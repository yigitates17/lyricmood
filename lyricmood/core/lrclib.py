"""LRCLIB lyrics client (https://lrclib.net/docs). No key; identify via User-Agent."""

from typing import Literal

import httpx
from pydantic import BaseModel

from lyricmood.core.cache import DAY, MISS, Cache
from lyricmood.core.errors import ApiError
from lyricmood.core.http import DEFAULT_RETRY, ApiClient, RateLimiter, RetryPolicy
from lyricmood.core.models import Track
from lyricmood.core.text import clean_title, same_name

API_BASE = "https://lrclib.net/api"
NOT_FOUND_TTL = 14 * DAY


class Lyrics(BaseModel):
    plain: str | None
    instrumental: bool = False
    lrclib_id: int
    # "exact": /api/get matched artist + title + duration.
    # "search": artist + title match from /api/search; closest duration wins.
    match: Literal["exact", "search"]


class LrclibClient(ApiClient):
    name = "LRCLIB"

    def __init__(
        self,
        cache: Cache,
        *,
        user_agent: str,
        rps: float = 2.0,
        retry: RetryPolicy = DEFAULT_RETRY,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        super().__init__(
            API_BASE,
            limiter=RateLimiter(rps, burst=2),
            headers={"User-Agent": user_agent},
            retry=retry,
            # /api/get may query external sources on a miss, which is slow.
            timeout=30.0,
            transport=transport,
        )
        self.cache = cache

    async def get(self, artist: str, title: str, duration_s: float) -> dict | None:
        # album_name is optional in practice, and Spotify album names
        # ("Once (Remastered)") often differ from LRCLIB's, causing misses.
        try:
            return await self.get_json(
                "/get",
                {"artist_name": artist, "track_name": title, "duration": round(duration_s)},
            )
        except ApiError as exc:
            if exc.status == 404:
                return None
            raise

    async def search(self, artist: str, title: str) -> list[dict]:
        return await self.get_json("/search", {"artist_name": artist, "track_name": title})

    async def find_lyrics(self, track: Track) -> Lyrics | None:
        """Lyrics for a Spotify track, cached by Spotify ID (misses for 14 days)."""
        cached = self.cache.get("lrclib", track.spotify_id)
        if cached is not MISS:
            return Lyrics.model_validate(cached) if cached else None
        lyrics = await self._lookup(track)
        self.cache.set(
            "lrclib",
            track.spotify_id,
            lyrics.model_dump() if lyrics else None,
            ttl=None if lyrics else NOT_FOUND_TTL,
        )
        return lyrics

    async def _lookup(self, track: Track) -> Lyrics | None:
        artist, title = track.primary_artist, clean_title(track.title)
        duration = track.duration_ms / 1000

        if hit := await self.get(artist, title, duration):
            return _to_lyrics(hit, "exact")

        # Plain lyrics are the same across versions (remasters, compilations,
        # radio edits), so artist + title decide the match. Duration only
        # ranks the matches; it never rejects one.
        candidates = [
            r
            for r in await self.search(artist, title)
            if (r.get("plainLyrics") or r.get("instrumental"))
            and same_name(r.get("artistName") or "", artist)
            and same_name(clean_title(r.get("trackName") or ""), title)
        ]
        # Prefer entries with lyrics; an instrumental entry only wins when
        # it is all LRCLIB has.
        candidates = [r for r in candidates if r.get("plainLyrics")] or candidates
        if not candidates:
            return None
        best = min(candidates, key=lambda r: abs((r.get("duration") or 0) - duration))
        return _to_lyrics(best, "search")


def _to_lyrics(record: dict, match: Literal["exact", "search"]) -> Lyrics:
    return Lyrics(
        plain=(record.get("plainLyrics") or "").strip() or None,
        instrumental=bool(record.get("instrumental")),
        lrclib_id=record["id"],
        match=match,
    )
