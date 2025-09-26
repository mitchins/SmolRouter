"""
Google GenAI Request Funnel - Rate limiting for Google GenAI API calls

Configurable via environment variables:
- GOOGLE_GENAI_MAX_CONCURRENT: Maximum concurrent requests (default: 3)
- GOOGLE_GENAI_MAX_REQUESTS_PER_WINDOW: Maximum requests per rolling window (default: 12)
- GOOGLE_GENAI_WINDOW_MINUTES: Rolling window size in minutes (default: 4)
- GOOGLE_GENAI_RATE_LIMITING_ENABLED: Enable/disable rate limiting (default: true)

Based on empirical analysis of Google's undocumented IP-based rate limits:
- ~3 concurrent requests maximum
- ~12 requests per 4-minute rolling window
- Burst protection prevents rapid successive requests
"""

import asyncio
import os
import time
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger("model-rerouter")

# Configuration with environment variable overrides
GOOGLE_GENAI_MAX_CONCURRENT = int(os.getenv("GOOGLE_GENAI_MAX_CONCURRENT", "3"))
GOOGLE_GENAI_MAX_REQUESTS_PER_WINDOW = int(os.getenv("GOOGLE_GENAI_MAX_REQUESTS_PER_WINDOW", "12"))
GOOGLE_GENAI_WINDOW_MINUTES = int(os.getenv("GOOGLE_GENAI_WINDOW_MINUTES", "4"))
GOOGLE_GENAI_RATE_LIMITING_ENABLED = os.getenv("GOOGLE_GENAI_RATE_LIMITING_ENABLED", "true").lower() in ("true", "1", "yes")


class GoogleGenAIRequestFunnel:
    """Thread-safe request funnel for Google GenAI API rate limiting"""

    def __init__(self,
                 max_concurrent: int = GOOGLE_GENAI_MAX_CONCURRENT,
                 max_requests_per_window: int = GOOGLE_GENAI_MAX_REQUESTS_PER_WINDOW,
                 window_minutes: int = GOOGLE_GENAI_WINDOW_MINUTES,
                 enabled: bool = GOOGLE_GENAI_RATE_LIMITING_ENABLED):

        self.enabled = enabled
        if not self.enabled:
            logger.info("Google GenAI rate limiting is DISABLED")
            return

        # Concurrent request limiting
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0

        # Rolling window tracking
        self._window_seconds = window_minutes * 60
        self._max_requests_per_window = max_requests_per_window
        self._request_timestamps = deque()  # Track request start times

        # Thread safety
        self._lock = asyncio.Lock()

        # Stats
        self._total_requests = 0
        self._total_waits = 0

        logger.info(f"Google GenAI request funnel initialized: {max_concurrent} concurrent, "
                   f"{max_requests_per_window} requests per {window_minutes}min window")

    async def acquire_slot(self) -> None:
        """
        Acquire a request slot, respecting both:
        1. Concurrent request limit (max_concurrent)
        2. Rolling window limit (max_requests_per_window per window_minutes)

        If rate limiting is disabled, this is a no-op.
        """
        if not self.enabled:
            return

        # First, wait for rolling window availability
        await self._wait_for_window_slot()

        # Then acquire concurrent slot
        await self._semaphore.acquire()

        async with self._lock:
            self._active_count += 1
            self._total_requests += 1

    def release_slot(self) -> None:
        """Release concurrent slot when request completes"""
        if not self.enabled:
            return

        self._active_count -= 1
        self._semaphore.release()

    async def _wait_for_window_slot(self) -> None:
        """Wait until rolling window allows new request"""
        while True:
            async with self._lock:
                now = time.time()

                # Remove old timestamps outside window
                while (self._request_timestamps and
                       now - self._request_timestamps[0] > self._window_seconds):
                    self._request_timestamps.popleft()

                # Check if we can make a new request
                if len(self._request_timestamps) < self._max_requests_per_window:
                    # Record this request timestamp
                    self._request_timestamps.append(now)
                    return

                # Calculate wait time until oldest request expires
                oldest_timestamp = self._request_timestamps[0]
                wait_time = self._window_seconds - (now - oldest_timestamp)
                self._total_waits += 1

                if self._total_waits % 10 == 0:  # Log every 10th wait to avoid spam
                    logger.info(f"Google GenAI rate limit: waiting {wait_time:.1f}s for rolling window slot "
                              f"({len(self._request_timestamps)}/{self._max_requests_per_window} used)")

            # Wait outside the lock to avoid blocking other coroutines
            if wait_time > 0:
                await asyncio.sleep(min(wait_time + 0.1, 5.0))  # Check at least every 5 seconds, add small buffer

    @property
    def stats(self) -> dict:
        """Get funnel statistics"""
        if not self.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "active_requests": self._active_count,
            "total_requests": self._total_requests,
            "total_waits": self._total_waits,
            "window_usage": f"{len(self._request_timestamps)}/{self._max_requests_per_window}",
            "window_remaining_seconds": self._window_seconds - (time.time() - self._request_timestamps[0]) if self._request_timestamps else 0
        }

# Global instance - shared across all Google GenAI providers
google_genai_funnel = GoogleGenAIRequestFunnel()