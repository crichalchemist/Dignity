"""
Retry utilities for robust data collection with exponential backoff.

Provides decorators and helper functions for handling transient failures
in API calls and file operations with proper jitter and rate limiting.
"""

import time
import random
import functools
import logging
from typing import Type, Union, List, Callable, Optional, Any

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """
    Retry decorator with exponential backoff and jitter.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for exponential backoff
        jitter: Whether to add random jitter to delays
        exceptions: Exception type(s) to catch and retry on
        on_retry: Optional callback function called on each retry

    Returns:
        Decorated function that retries on failure

    Example:
        @retry_with_backoff(max_retries=3, base_delay=5, max_delay=60)
        def download_data(url):
            return requests.get(url)
    """
    if isinstance(exceptions, type):
        exceptions = [exceptions]

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except tuple(exceptions) as e:
                    last_exception = e

                    # Don't retry on the final attempt
                    if attempt == max_retries:
                        logger.error(
                            f"[retry] {func.__name__} failed after {max_retries} retries: {str(e)[:100]}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor**attempt), max_delay)

                    # Add jitter if enabled
                    if jitter:
                        jitter_range = delay * 0.1
                        delay += random.uniform(-jitter_range, jitter_range)
                        delay = max(0, delay)  # Ensure non-negative

                    logger.warning(
                        f"[retry] {func.__name__} attempt {attempt + 1} failed: {str(e)[:50]}... "
                        f"Retrying in {delay:.1f}s"
                    )

                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(e, attempt + 1)
                        except Exception as callback_error:
                            logger.warning(f"[retry] Callback failed: {callback_error}")

                    # Wait before retry
                    time.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def rate_limit(calls_per_second: float = 1.0) -> Callable:
    """
    Rate limiting decorator to prevent API rate limit violations.

    Args:
        calls_per_second: Maximum number of calls per second

    Returns:
        Decorated function with rate limiting

    Example:
        @rate_limit(calls_per_second=0.5)  # 0.5 calls per second (1 call every 2 seconds)
        def api_call():
            return requests.get(url)
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]  # Use list to allow modification in closure

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_time = time.time()
            elapsed = current_time - last_called[0]

            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                logger.debug(f"[rate_limit] Sleeping {sleep_time:.2f}s to respect rate limit")
                time.sleep(sleep_time)

            last_called[0] = time.time()
            return func(*args, **kwargs)

        return wrapper

    return decorator


def resilient_file_operation(
    max_retries: int = 3, base_delay: float = 0.1, max_delay: float = 5.0
) -> Callable:
    """
    Decorator for resilient file operations (read/write).

    Specifically handles file system related issues like:
    - Temporary file locks
    - Network storage timeouts
    - Permission issues

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Decorated function with resilient file operations
    """
    file_exceptions = [OSError, IOError, PermissionError, FileNotFoundError, TimeoutError]

    return retry_with_backoff(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exceptions=file_exceptions,
    )


def api_call_with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    rate_limit_calls: float = 1.0,
) -> Callable:
    """
    Combined decorator for API calls with both retry and rate limiting.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        rate_limit_calls: Maximum calls per second

    Returns:
        Decorated function with retry and rate limiting
    """

    def decorator(func: Callable) -> Callable:
        # Apply rate limiting first
        rate_limited = rate_limit(rate_limit_calls)(func)

        # Then apply retry logic with only transient/network errors
        retried = retry_with_backoff(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            exceptions=[
                # HTTP/network errors that are transient
                TimeoutError,
                ConnectionError,
                # Add request library specific exceptions if available
                # OSError for file/network issues
                OSError,
            ],
        )(rate_limited)

        return retried

    return decorator


class RetryContext:
    """
    Context manager for retry operations with state tracking.

    Useful when you need more control over retry logic or want to
    track retry statistics across multiple operations.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exceptions = exceptions if isinstance(exceptions, list) else [exceptions]
        self.attempts = 0
        self.successes = 0
        self.failures = 0
        self.last_exception = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.failures += 1
            self.last_exception = exc_val
        else:
            self.successes += 1
        return False  # Don't suppress exceptions

    def attempt(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic within this context.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function return value

        Raises:
            Last exception if all retries fail
        """
        for attempt in range(self.max_retries + 1):
            self.attempts += 1

            try:
                return func(*args, **kwargs)

            except tuple(self.exceptions) as e:
                self.last_exception = e

                if attempt == self.max_retries:
                    logger.error(
                        f"[retry_context] Failed after {self.max_retries} retries: {str(e)[:100]}"
                    )
                    raise

                delay = min(self.base_delay * (2**attempt), self.max_delay)
                jitter = delay * 0.1 * (2 * random.random() - 1)
                # Ensure sleep time is non-negative
                sleep_time = max(0, delay + jitter)

                logger.warning(
                    f"[retry_context] Attempt {attempt + 1} failed: {str(e)[:50]}... "
                    f"Retrying in {sleep_time:.1f}s"
                )

                time.sleep(sleep_time)

    def get_stats(self) -> dict:
        """Get retry statistics."""
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": self.successes / max(self.attempts, 1),
            "last_exception": str(self.last_exception) if self.last_exception else None,
        }


# Convenience decorators for common use cases
http_retry = retry_with_backoff(
    max_retries=3,
    base_delay=2.0,
    max_delay=30.0,
    exceptions=[
        TimeoutError,
        ConnectionError,
        # Request-specific exceptions will be caught by generic Exception handling
    ],
)

file_retry = resilient_file_operation(max_retries=2, base_delay=0.1, max_delay=1.0)

api_retry = api_call_with_retry(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    rate_limit_calls=0.5,  # Conservative rate limiting
)
