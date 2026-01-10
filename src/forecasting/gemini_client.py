"""Google Gemini API client with rate limiting and retry logic.

This module provides a wrapper around the Google GenAI SDK with:
- Automatic retry logic using exponential backoff
- Rate limiting (5 RPM for free tier)
- Context caching support
- Sliding window rate limit tracking
"""

import os
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Optional

import google.genai as genai
from google.genai.types import GenerateContentResponse
from tenacity import retry, stop_after_attempt, wait_exponential


class RateLimiter:
    """Sliding window rate limiter for API calls."""

    def __init__(self, max_requests: int = 5, window_minutes: int = 1):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in the window
            window_minutes: Time window in minutes
        """
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.request_times: deque = deque()

    def is_allowed(self) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        # Remove requests outside the window
        while self.request_times and self.request_times[0] < now - self.window_seconds:
            self.request_times.popleft()

        return len(self.request_times) < self.max_requests

    def record_request(self) -> None:
        """Record a request timestamp."""
        self.request_times.append(time.time())

    def wait_if_needed(self) -> None:
        """Wait until a request is allowed."""
        while not self.is_allowed():
            if self.request_times:
                wait_time = self.window_seconds - (time.time() - self.request_times[0])
                if wait_time > 0:
                    time.sleep(min(wait_time, 1))  # Sleep in 1-second intervals max


class GeminiClient:
    """Client for Google Gemini API with rate limiting and retry logic."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "models/gemini-2.0-flash-exp",
        rate_limit_rpm: int = 5,
    ):
        """Initialize Gemini client.

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model: Model to use for generation
            rate_limit_rpm: Rate limit in requests per minute (default 5 for free tier)

        Raises:
            ValueError: If API key is not provided and GOOGLE_API_KEY env var not set
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "API key must be provided or GOOGLE_API_KEY environment variable set"
                )

        genai.configure(api_key=api_key)
        self.model = model
        self.rate_limiter = RateLimiter(max_requests=rate_limit_rpm, window_minutes=1)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    def generate_content(
        self,
        prompt: str,
        response_schema: Optional[Any] = None,
        response_mime_type: Optional[str] = None,
        **kwargs,
    ) -> GenerateContentResponse:
        """Generate content using Gemini with automatic retry.

        Args:
            prompt: Input prompt for generation
            response_schema: Pydantic model for structured output (optional)
            response_mime_type: MIME type for response format (optional)
            **kwargs: Additional arguments passed to generate_content

        Returns:
            GenerateContentResponse from Gemini API

        Raises:
            Exception: If API call fails after all retries
        """
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()

        # Build request config
        generation_config = {}
        if response_schema:
            generation_config["response_schema"] = response_schema
        if response_mime_type:
            generation_config["response_mime_type"] = response_mime_type

        # Merge with any additional kwargs
        generation_config.update(kwargs.get("generation_config", {}))

        # Create client and generate
        client = genai.Client()
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                **generation_config,
            ),
        )

        # Record request for rate limiting
        self.rate_limiter.record_request()

        return response

    def get_rate_limit_status(self) -> dict:
        """Get current rate limit status.

        Returns:
            Dictionary with rate limit info (requests_used, max_requests, reset_time)
        """
        now = time.time()
        active_requests = sum(
            1
            for t in self.rate_limiter.request_times
            if t > now - self.rate_limiter.window_seconds
        )

        if self.rate_limiter.request_times:
            oldest_request = self.rate_limiter.request_times[0]
            reset_time = datetime.fromtimestamp(
                oldest_request + self.rate_limiter.window_seconds
            )
        else:
            reset_time = datetime.now()

        return {
            "requests_used": active_requests,
            "max_requests": self.rate_limiter.max_requests,
            "window_seconds": self.rate_limiter.window_seconds,
            "reset_time": reset_time.isoformat(),
        }
