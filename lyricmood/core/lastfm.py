"""Last.fm client: tags, top tracks, similar tracks/artists.

Last.fm reports errors as JSON `{"error": code, "message": ...}`, sometimes
with HTTP 200, and returns numbers as strings in some methods.
"""

import httpx
from pydantic import BaseModel

from lyricmood.core.cache import DAY, Cache
from lyricmood.core.errors import ApiError, ConfigError
from lyricmood.core.http import DEFAULT_RETRY, ApiClient, RateLimiter, Retryable, RetryPolicy
from lyricmood.core.text import normalize

API_BASE = "https://ws.audioscrobbler.com/2.0/"
TTL = 30 * DAY

_TRANSIENT = {8, 11, 16, 29}  # operation failed, offline, temporary error, rate limit
_BAD_KEY = {10, 26}  # invalid key, suspended key
_NOT_FOUND = 6

# Tags that describe the listener, not the music.
_TAG_BLOCKLIST = {
    "seen live",
    "favorite",
    "favorites",
    "favourite",
    "favourites",
    "my favorite",
    "love",
    "loved",
    "awesome",
    "beautiful",
    "best",
    "albums i own",
    "spotify",
    "check out",
}


class Tag(BaseModel):
    name: str
    count: int


class LfmTrack(BaseModel):
    artist: str
    title: str
    duration_s: int | None = None
    match: float | None = None
    listeners: int | None = None


def _int(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_list(value) -> list:
    # Single results sometimes come back as an object instead of a list.
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


class LastfmClient(ApiClient):
    name = "Last.fm"

    def __init__(
        self,
        cache: Cache,
        *,
        api_key: str,
        user_agent: str,
        rps: float = 4.0,
        retry: RetryPolicy = DEFAULT_RETRY,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        super().__init__(
            API_BASE,
            limiter=RateLimiter(rps, burst=4),
            headers={"User-Agent": user_agent},
            retry=retry,
            transport=transport,
        )
        self.cache = cache
        self._key = api_key

    def _check_response(self, response: httpx.Response) -> None:
        try:
            code = response.json().get("error")
        except (ValueError, AttributeError):
            return
        if code in _TRANSIENT:
            raise Retryable(response)

    async def call(self, method: str, **params) -> dict | None:
        """Call a Last.fm method. Returns None when Last.fm says 'not found'."""
        params = {"method": method, "api_key": self._key, "format": "json", **params}
        response = await self.request_with_retries("GET", "", params=params)
        try:
            data = response.json()
        except ValueError:
            raise ApiError(self.name, response.status_code, response.text[:200]) from None
        code = data.get("error") if isinstance(data, dict) else None
        if code == _NOT_FOUND:
            return None
        if code in _BAD_KEY:
            raise ConfigError(f"Last.fm rejected LASTFM_API_KEY: {data.get('message')}")
        if code or response.is_error:
            raise ApiError(self.name, response.status_code, str(data.get("message", code)))
        return data

    async def _cached(self, method: str, key: str, **params) -> dict | None:
        return await self.cache.get_or_fetch(
            f"lastfm:{method}", key, lambda: self.call(method, **params), ttl=TTL
        )

    async def track_tags(self, artist: str, title: str, limit: int = 8) -> list[Tag]:
        data = await self._cached(
            "track.getTopTags",
            f"{normalize(artist)}|{normalize(title)}",
            artist=artist,
            track=title,
            autocorrect=1,
        )
        return _tags(data, limit, exclude=artist)

    async def artist_tags(self, artist: str, limit: int = 8) -> list[Tag]:
        data = await self._cached(
            "artist.getTopTags", normalize(artist), artist=artist, autocorrect=1
        )
        return _tags(data, limit, exclude=artist)

    async def artist_top_tracks(self, artist: str, limit: int = 10) -> list[LfmTrack]:
        data = await self._cached(
            "artist.getTopTracks",
            f"{normalize(artist)}|{limit}",
            artist=artist,
            limit=limit,
            autocorrect=1,
        )
        items = _as_list(((data or {}).get("toptracks") or {}).get("track"))
        return [_track(t) for t in items][:limit]

    async def similar_tracks(self, artist: str, title: str, limit: int = 30) -> list[LfmTrack]:
        data = await self._cached(
            "track.getSimilar",
            f"{normalize(artist)}|{normalize(title)}|{limit}",
            artist=artist,
            track=title,
            limit=limit,
            autocorrect=1,
        )
        items = _as_list(((data or {}).get("similartracks") or {}).get("track"))
        return [_track(t) for t in items][:limit]

    async def similar_artists(self, artist: str, limit: int = 20) -> list[tuple[str, float]]:
        data = await self._cached(
            "artist.getSimilar",
            f"{normalize(artist)}|{limit}",
            artist=artist,
            limit=limit,
            autocorrect=1,
        )
        items = _as_list(((data or {}).get("similarartists") or {}).get("artist"))
        return [(a["name"], _float(a.get("match")) or 0.0) for a in items][:limit]

    async def tag_top_tracks(self, tag: str, limit: int = 30) -> list[LfmTrack]:
        data = await self._cached(
            "tag.getTopTracks", f"{normalize(tag)}|{limit}", tag=tag, limit=limit
        )
        items = _as_list(((data or {}).get("tracks") or {}).get("track"))
        return [_track(t) for t in items][:limit]


def _track(item: dict) -> LfmTrack:
    artist = item.get("artist")
    return LfmTrack(
        artist=artist.get("name", "") if isinstance(artist, dict) else str(artist or ""),
        title=item.get("name", ""),
        # 0 means unknown on Last.fm.
        duration_s=_int(item.get("duration")) or None,
        match=_float(item.get("match")),
        listeners=_int(item.get("listeners")),
    )


def _tags(data: dict | None, limit: int, exclude: str) -> list[Tag]:
    items = _as_list(((data or {}).get("toptags") or {}).get("tag"))
    skip = _TAG_BLOCKLIST | {normalize(exclude)}
    tags = [
        Tag(name=t["name"].strip().lower(), count=_int(t.get("count")) or 0)
        for t in items
        if t.get("name") and normalize(t["name"]) not in skip
    ]
    # Counts are relative (the top tag is 100); below ~5 is mostly noise.
    return [t for t in tags if t.count >= 5][:limit]
