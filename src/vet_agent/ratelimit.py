import math
import threading
import time
from collections import deque
from collections.abc import Callable
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from vet_agent.config import Settings, get_settings


class SlidingWindowLimiter:
    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_requests = max_requests
        self._window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        now = self._clock()
        with self._lock:
            self._purge(now)
            hits = self._hits.setdefault(key, deque())
            if len(hits) >= self._max_requests:
                return max(self._window - (now - hits[0]), 0.0)
            hits.append(now)
            return None

    def _purge(self, now: float) -> None:
        horizon = now - self._window
        for key in list(self._hits):
            hits = self._hits[key]
            while hits and hits[0] <= horizon:
                hits.popleft()
            if not hits:
                del self._hits[key]


@lru_cache
def build_limiter(max_requests: int, window_seconds: int) -> SlidingWindowLimiter:
    return SlidingWindowLimiter(max_requests, window_seconds)


def get_rate_limiter(settings: Annotated[Settings, Depends(get_settings)]) -> SlidingWindowLimiter:
    return build_limiter(
        settings.agent_rate_limit_requests, settings.agent_rate_limit_window_seconds
    )


LimiterDep = Annotated[SlidingWindowLimiter, Depends(get_rate_limiter)]


def enforce_rate_limit(request: Request, limiter: LimiterDep) -> None:
    key = request.client.host if request.client else "unknown"
    retry_after = limiter.check(key)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Try again later.",
            headers={"Retry-After": str(max(math.ceil(retry_after), 1))},
        )
