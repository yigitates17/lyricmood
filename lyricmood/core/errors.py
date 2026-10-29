"""Errors whose messages are safe to show to the user or an LLM as-is."""


class LyricmoodError(Exception):
    """Base class. The message should tell the user what to do next."""


class ConfigError(LyricmoodError):
    pass


class AuthRequired(LyricmoodError):
    pass


class RateLimited(LyricmoodError):
    def __init__(self, api: str, retry_after: float | None):
        self.api = api
        self.retry_after = retry_after
        wait = f" Retry in ~{int(retry_after)}s." if retry_after else ""
        super().__init__(f"{api} rate limit hit.{wait}")


class ApiError(LyricmoodError):
    def __init__(self, api: str, status: int, detail: str = ""):
        self.api = api
        self.status = status
        super().__init__(f"{api} returned HTTP {status}{': ' + detail if detail else ''}")


class NotFound(LyricmoodError):
    pass


class InvalidInput(LyricmoodError):
    """Arguments that are well-formed but can't be used (e.g. a track name
    where a Spotify ID is required)."""
