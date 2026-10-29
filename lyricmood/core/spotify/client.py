"""Async Spotify Web API client.

Written against the post-February-2026 Development Mode API: search returns
at most 10 results per page, there is no batch `GET /tracks`, and artist
top-tracks, recommendations, audio-features and related-artists are gone.
"""

import re
from collections.abc import Awaitable, Callable

import httpx
from pydantic import BaseModel

from lyricmood.core.cache import DAY, MISS, Cache
from lyricmood.core.errors import ApiError, NotFound
from lyricmood.core.http import DEFAULT_RETRY, ApiClient, RateLimiter, RetryPolicy
from lyricmood.core.models import Track
from lyricmood.core.text import clean_title, same_name

API_BASE = "https://api.spotify.com/v1"
SEARCH_PAGE_MAX = 10
SEARCH_TTL = 7 * DAY

_ID_RE = re.compile(r"^[A-Za-z0-9]{22}$")
_TRACK_REF_RE = re.compile(
    r"(?:open\.spotify\.com/(?:intl-[a-z-]+/)?track/|spotify:track:)([A-Za-z0-9]{22})"
)


def parse_track_id(ref: str) -> str | None:
    """Extract a track ID from a Spotify URL, URI, or bare ID; else None."""
    ref = ref.strip()
    if match := _TRACK_REF_RE.search(ref):
        return match.group(1)
    return ref if _ID_RE.match(ref) else None


class SpotifyClient(ApiClient):
    name = "Spotify"

    def __init__(
        self,
        token_provider: Callable[[bool], Awaitable[str]],
        cache: Cache,
        *,
        rps: float = 5.0,
        retry: RetryPolicy = DEFAULT_RETRY,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        super().__init__(
            API_BASE, limiter=RateLimiter(rps, burst=5), retry=retry, transport=transport
        )
        self._token = token_provider
        self.cache = cache

    async def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {await self._token(False)}"}

    async def _on_unauthorized(self) -> bool:
        await self._token(True)
        return True

    async def search_tracks(self, query: str, limit: int = 5) -> list[Track]:
        """Search tracks, paginating past Spotify's 10-per-page cap if needed."""
        limit = max(1, min(limit, 50))
        key = f"{query.strip().lower()}|{limit}"
        ids = self.cache.get("spotify:search", key)
        if ids is not MISS:
            tracks = [t for i in ids if (t := self.cache.get_track(i))]
            if len(tracks) == len(ids):
                return tracks

        tracks: list[Track] = []
        offset = 0
        while len(tracks) < limit:
            page_size = min(SEARCH_PAGE_MAX, limit - len(tracks))
            data = await self.get_json(
                "/search", {"q": query, "type": "track", "limit": page_size, "offset": offset}
            )
            items = [i for i in data["tracks"]["items"] if i]
            tracks.extend(Track.from_spotify(i) for i in items)
            if len(items) < page_size or not data["tracks"].get("next"):
                break
            offset += page_size

        self.cache.put_tracks(tracks)
        self.cache.set("spotify:search", key, [t.spotify_id for t in tracks], ttl=SEARCH_TTL)
        return tracks

    async def get_track(self, track_id: str) -> Track:
        if cached := self.cache.get_track(track_id):
            return cached
        try:
            data = await self.get_json(f"/tracks/{track_id}")
        except ApiError as exc:
            if exc.status in (400, 404):
                raise NotFound(f"No Spotify track with ID {track_id}.") from None
            raise
        track = Track.from_spotify(data)
        self.cache.put_tracks([track])
        return track

    async def resolve(self, ref: str) -> Track:
        """Turn a Spotify URL/URI/ID or a free-text query into one track.

        Every tool that takes a `track` argument goes through this.
        """
        if track_id := parse_track_id(ref):
            return await self.get_track(track_id)
        results = await self.search_tracks(ref, limit=1)
        if not results:
            raise NotFound(f"No Spotify track found for {ref!r}.")
        return results[0]

    async def find_by_isrc(self, isrc: str) -> Track | None:
        if cached := self.cache.get_track_by_isrc(isrc):
            return cached
        results = await self.search_tracks(f"isrc:{isrc}", limit=1)
        return results[0] if results else None

    async def find_by_name(self, artist: str, title: str) -> Track | None:
        """Resolve an artist + title (e.g. from Last.fm) to a Spotify track,
        rejecting search hits whose artist or title doesn't match."""
        query = f'artist:"{artist}" track:"{clean_title(title)}"'
        for track in await self.search_tracks(query, limit=3):
            if any(same_name(a, artist) for a in track.artists) and same_name(
                clean_title(track.title), clean_title(title)
            ):
                return track
        return None

    async def currently_playing(self) -> "NowPlaying | None":
        """What the user is playing now; None when nothing is playing."""
        response = await self.request("GET", "/me/player/currently-playing")
        if response.status_code == 204 or not response.content:
            return None
        data = response.json()
        item = data.get("item")
        track = None
        if item and data.get("currently_playing_type") == "track":
            track = Track.from_spotify(item)
            self.cache.put_tracks([track])
        return NowPlaying(
            track=track,
            is_playing=bool(data.get("is_playing")),
            progress_ms=data.get("progress_ms"),
            kind=data.get("currently_playing_type") or "unknown",
        )


class NowPlaying(BaseModel):
    track: Track | None
    is_playing: bool
    progress_ms: int | None = None
    # "track", "episode", "ad" or "unknown"
    kind: str
