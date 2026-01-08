"""
Rate limiting with exponential backoff for API requests.
"""

import time
import logging
from typing import Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)


class RateLimiter:
    """Implements exponential backoff with configurable retry logic."""

    def __init__(self, base_delay: float = 1.0, max_retries: int = 5, max_delay: float = 60.0):
        """
        Initialize rate limiter.

        Args:
            base_delay: Base delay in seconds (will be exponentially increased)
            max_retries: Maximum number of retry attempts
            max_delay: Maximum delay between retries in seconds
        """
        self.base_delay = base_delay
        self.max_retries = max_retries
        self.max_delay = max_delay
        self.request_timestamps = []
        self.min_interval = 0.1  # Minimum time between requests (100ms)

    def wait_if_needed(self):
        """Check last request timestamp and wait if necessary."""
        current_time = time.time()

        if self.request_timestamps:
            last_request = self.request_timestamps[-1]
            time_since_last = current_time - last_request

            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)

        self.request_timestamps.append(time.time())

        # Keep only last 100 timestamps to prevent memory issues
        if len(self.request_timestamps) > 100:
            self.request_timestamps = self.request_timestamps[-100:]

    def with_exponential_backoff(self, func: Callable) -> Callable:
        """
        Decorator that adds exponential backoff retry logic to a function.

        Args:
            func: Function to wrap with retry logic

        Returns:
            Wrapped function with exponential backoff
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(self.max_retries):
                try:
                    # Wait if needed based on request history
                    self.wait_if_needed()

                    # Try to execute the function
                    result = func(*args, **kwargs)

                    # Success - reset any backoff state
                    if attempt > 0:
                        logger.info(f"Request succeeded after {attempt} retries")

                    return result

                except (ConnectionError, TimeoutError) as e:
                    last_exception = e
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)

                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                        f"\nRetrying in {delay:.1f}s..."
                    )

                    if attempt < self.max_retries - 1:
                        time.sleep(delay)

                except Exception as e:
                    # For HTTP 429 (rate limit) errors from requests library
                    if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                        if e.response.status_code == 429:
                            last_exception = e
                            delay = min(self.base_delay * (2 ** attempt), self.max_delay)

                            # Check if server provided Retry-After header
                            retry_after = e.response.headers.get('Retry-After')
                            if retry_after:
                                try:
                                    delay = float(retry_after)
                                    logger.info(f"Using server-provided Retry-After: {delay}s")
                                except ValueError:
                                    pass

                            logger.warning(
                                f"Rate limit hit (HTTP 429) - attempt {attempt + 1}/{self.max_retries}"
                                f"\nRetrying in {delay:.1f}s..."
                            )

                            if attempt < self.max_retries - 1:
                                time.sleep(delay)
                        else:
                            # Re-raise non-rate-limit errors
                            raise
                    else:
                        # Re-raise unexpected errors
                        raise

            # All retries exhausted
            logger.error(f"All {self.max_retries} retry attempts failed")
            raise last_exception

        return wrapper


# Global rate limiter instance for module-level use
default_rate_limiter = RateLimiter()
with_retry = default_rate_limiter.with_exponential_backoff