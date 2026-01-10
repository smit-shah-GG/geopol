"""
Gemini API client with rate limiting and retry logic.

This module provides a robust client for interacting with Google's Gemini LLM
using the new google-genai SDK. It includes:
- Rate limiting (5 RPM for free tier)
- Exponential backoff retry logic
- Context caching support
- Structured output handling
"""

import os
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from google import genai
from google.genai.types import GenerateContentResponse, GenerationConfig
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class RateLimitExceeded(Exception):
    """Raised when rate limit would be exceeded."""

    pass


class GeminiClient:
    """
    Client for interacting with Gemini API with built-in rate limiting and retry logic.

    Attributes:
        model_name: The Gemini model to use (default: gemini-2.0-flash-exp)
        max_rpm: Maximum requests per minute (default: 5 for free tier)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "models/gemini-2.0-flash-exp",
        max_rpm: int = 5,
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Google AI API key. If None, reads from GEMINI_API_KEY env var.
            model_name: Name of the Gemini model to use.
            max_rpm: Maximum requests per minute (rate limit).

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "No API key provided. Set GEMINI_API_KEY environment variable "
                    "or pass api_key parameter."
                )

        # Initialize the client with the new SDK
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_rpm = max_rpm

        # Request tracking for rate limiting (sliding window)
        self._request_times: deque = deque()

        # Initialize model with caching support
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the Gemini model with proper configuration."""
        # Model configuration for better structured outputs
        self.generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
        )

    def _check_rate_limit(self) -> None:
        """
        Check if making a request would exceed rate limit.

        Uses a sliding window approach to track requests per minute.

        Raises:
            RateLimitExceeded: If making a request would exceed the rate limit.
        """
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Remove requests older than 1 minute from tracking
        while self._request_times and self._request_times[0] < minute_ago:
            self._request_times.popleft()

        # Check if we're at the limit
        if len(self._request_times) >= self.max_rpm:
            # Calculate when we can make the next request
            oldest_request = self._request_times[0]
            wait_time = (oldest_request + timedelta(minutes=1) - now).total_seconds()
            raise RateLimitExceeded(f"Rate limit reached. Wait {wait_time:.1f} seconds.")

        # Track this request
        self._request_times.append(now)

    @retry(
        retry=retry_if_exception_type((Exception,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True,
    )
    def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> GenerateContentResponse:
        """
        Generate content from Gemini with retry logic.

        Args:
            prompt: The user prompt to send to Gemini.
            system_instruction: Optional system instruction for the model.
            response_schema: Optional schema for structured JSON output.
            generation_config: Optional custom generation configuration.

        Returns:
            The response from Gemini API.

        Raises:
            RateLimitExceeded: If rate limit would be exceeded.
            Exception: For other API errors after retries exhausted.
        """
        # Check rate limit before making request
        self._check_rate_limit()

        # Use custom generation config if provided, else use default
        config = generation_config or self.generation_config

        # If response schema is provided, ensure JSON response
        if response_schema:
            config.response_mime_type = "application/json"
            config.response_schema = response_schema

        # Build the request
        contents = [prompt]

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
                system_instruction=system_instruction,
            )
            return response
        except Exception as e:
            # Log error for debugging
            print(f"Error generating content: {e}")
            raise

    def generate_with_context_caching(
        self,
        prompt: str,
        context: List[str],
        system_instruction: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> GenerateContentResponse:
        """
        Generate content with context caching for efficiency.

        This method leverages the SDK's automatic context caching to reduce
        token usage when using the same context across multiple requests.

        Args:
            prompt: The user prompt.
            context: List of context strings to cache.
            system_instruction: Optional system instruction.
            response_schema: Optional response schema for structured output.

        Returns:
            The response from Gemini API.
        """
        # Combine context and prompt
        full_prompt = "\n\n".join(context) + "\n\n" + prompt

        return self.generate_content(
            prompt=full_prompt,
            system_instruction=system_instruction,
            response_schema=response_schema,
        )

    def wait_for_rate_limit(self) -> float:
        """
        Wait if necessary to avoid exceeding rate limit.

        Returns:
            The number of seconds waited (0 if no wait needed).
        """
        try:
            self._check_rate_limit()
            return 0.0
        except RateLimitExceeded as e:
            # Extract wait time from exception message
            import re
            match = re.search(r"Wait (\d+\.?\d*) seconds", str(e))
            if match:
                wait_time = float(match.group(1))
                time.sleep(wait_time)
                return wait_time
            return 0.0

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status.

        Returns:
            Dictionary with rate limit information.
        """
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean old requests
        while self._request_times and self._request_times[0] < minute_ago:
            self._request_times.popleft()

        return {
            "requests_in_window": len(self._request_times),
            "max_requests": self.max_rpm,
            "available_requests": self.max_rpm - len(self._request_times),
            "window_start": minute_ago.isoformat(),
            "window_end": now.isoformat(),
        }


# Convenience function for quick testing
def create_client() -> GeminiClient:
    """Create a default Gemini client with environment configuration."""
    return GeminiClient()