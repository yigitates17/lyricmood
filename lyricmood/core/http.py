"""Async HTTP base shared by all API clients: rate limiting and retries.

Each API gets its own `ApiClient` (and so its own limiter); clients never
share a limiter because each upstream has independent limits.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from lyricmood.core.errors import ApiError, RateLimited

RETRY_STATUSES = {429, 500, 502, 503, 504}


class RateLimiter:
    """Token bucket: `rate` requests per second with bursts up to `burst`."""

    def __init__(self, rate: float, burst: int = 1):
        self.rate = rate
        self.capacity = float(burst)
        self._tokens = float(burst)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                await asyncio.sleep((1 - self._tokens) / self.rate)


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 4
    wait_initial: float = 0.5
    wait_max: float = 20.0
    # A Retry-After longer than this means a quota is exhausted (Spotify can
    # answer with hours); fail fast instead of hanging a chat turn.
    max_retry_after: float = 60.0


DEFAULT_RETRY = RetryPolicy()


class _Retryable(Exception):
    def __init__(self, response: httpx.Response | None, retry_after: float | None = None):
        self.response = response
        self.retry_after = retry_after


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class ApiClient:
    """Thin wrapper over `httpx.AsyncClient` with per-API limits and retries."""

    name = "api"

    def __init__(
        self,
        base_url: str,
        *,
        limiter: RateLimiter,
        headers: dict[str, str] | None = None,
        retry: RetryPolicy = DEFAULT_RETRY,
        timeout: float = 20.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.limiter = limiter
        self.retry = retry
        self._http = httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=timeout, transport=transport
        )

    async def _auth_headers(self) -> dict[str, str]:
        return {}

    async def _on_unauthorized(self) -> bool:
        """Hook for clients that can refresh credentials. Return True to retry once."""
        return False

    async def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        await self.limiter.acquire()
        headers = {**kwargs.pop("headers", {}), **await self._auth_headers()}
        try:
            response = await self._http.request(method, path, headers=headers, **kwargs)
        except httpx.TransportError as exc:
            raise _Retryable(None) from exc
        if response.status_code in RETRY_STATUSES:
            retry_after = _retry_after(response)
            if retry_after is not None and retry_after > self.retry.max_retry_after:
                raise RateLimited(self.name, retry_after)
            raise _Retryable(response, retry_after)
        return response

    async def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        retried_auth = False
        while True:
            response = await self._request_with_retries(method, path, **kwargs)
            if response.status_code == 401 and not retried_auth and await self._on_unauthorized():
                retried_auth = True
                continue
            if response.is_error:
                raise ApiError(self.name, response.status_code, _error_detail(response))
            return response

    async def _request_with_retries(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        exp_wait = wait_exponential_jitter(
            multiplier=self.retry.wait_initial, max=self.retry.wait_max
        )

        def wait(state) -> float:
            exc = state.outcome.exception()
            hinted = getattr(exc, "retry_after", None)
            return max(hinted or 0.0, exp_wait(state))

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.retry.attempts),
                wait=wait,
                retry=retry_if_exception_type(_Retryable),
                reraise=True,
            ):
                with attempt:
                    return await self._send(method, path, **kwargs)
        except _Retryable as exc:
            if exc.response is None:
                raise ApiError(self.name, 0, "network error after retries") from exc.__cause__
            if exc.response.status_code == 429:
                raise RateLimited(self.name, exc.retry_after) from None
            detail = _error_detail(exc.response)
            raise ApiError(self.name, exc.response.status_code, detail) from None
        raise AssertionError("unreachable")

    async def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        response = await self.request("GET", path, params=params)
        return response.json()

    async def aclose(self) -> None:
        await self._http.aclose()


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)[:200]
        if err:
            return str(body.get("message") or err)[:200]
    return str(body)[:200]
