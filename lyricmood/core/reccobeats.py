"""ReccoBeats client: Spotify-style audio features and recommendations.

ReccoBeats is unreliable and has gaps: unknown tracks are silently left out
of responses. A missing track is cached as None (a normal answer), and
callers must work without features.
"""

import re

import httpx
from pydantic import BaseModel

from lyricmood.core.cache import DAY, MISS, Cache
from lyricmood.core.http import DEFAULT_RETRY, ApiClient, RateLimiter, RetryPolicy

API_BASE = "https://api.reccobeats.com/v1"
BATCH = 40
MISSING_TTL = 14 * DAY
_SPOTIFY_ID = re.compile(r"open\.spotify\.com/(?:track|artist)/([A-Za-z0-9]{22})")


class AudioFeatures(BaseModel):
    valence: float | None = None
    energy: float | None = None
    danceability: float | None = None
    acousticness: float | None = None
    instrumentalness: float | None = None
    speechiness: float | None = None
    liveness: float | None = None
    loudness: float | None = None
    tempo: float | None = None
    key: int | None = None
    mode: int | None = None


class RecTrack(BaseModel):
    spotify_id: str
    title: str
    artists: list[str]
    duration_ms: int | None = None
    isrc: str | None = None


def _spotify_id(href: str | None) -> str | None:
    match = _SPOTIFY_ID.search(href or "")
    return match.group(1) if match else None


class ReccoBeatsClient(ApiClient):
    name = "ReccoBeats"

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
            transport=transport,
        )
        self.cache = cache

    async def audio_features(self, spotify_ids: list[str]) -> dict[str, AudioFeatures | None]:
        """Features per Spotify ID; None where ReccoBeats has no data."""
        result: dict[str, AudioFeatures | None] = {}
        todo = []
        for sid in dict.fromkeys(spotify_ids):
            cached = self.cache.get("reccobeats:features", sid)
            if cached is MISS:
                todo.append(sid)
            else:
                result[sid] = AudioFeatures.model_validate(cached) if cached else None

        for start in range(0, len(todo), BATCH):
            batch = todo[start : start + BATCH]
            data = await self.get_json("/audio-features", {"ids": ",".join(batch)})
            found = {
                sid: AudioFeatures.model_validate(item)
                for item in data.get("content") or []
                if (sid := _spotify_id(item.get("href")))
            }
            for sid in batch:
                features = found.get(sid)
                result[sid] = features
                self.cache.set(
                    "reccobeats:features",
                    sid,
                    features.model_dump() if features else None,
                    ttl=None if features else MISSING_TTL,
                )
        return result

    async def recommendations(self, seed_ids: list[str], size: int = 20) -> list[RecTrack]:
        """Tracks similar to up to 5 Spotify seed IDs."""
        seeds = list(dict.fromkeys(seed_ids))[:5]
        key = f"{','.join(sorted(seeds))}|{size}"

        async def fetch() -> list[dict]:
            data = await self.get_json(
                "/track/recommendation", {"seeds": ",".join(seeds), "size": size}
            )
            tracks = []
            for item in data.get("content") or []:
                if not (sid := _spotify_id(item.get("href"))):
                    continue
                tracks.append(
                    RecTrack(
                        spotify_id=sid,
                        title=item.get("trackTitle") or "",
                        artists=[a["name"] for a in item.get("artists") or [] if a.get("name")],
                        duration_ms=item.get("durationMs"),
                        isrc=item.get("isrc"),
                    ).model_dump()
                )
            return tracks

        items = await self.cache.get_or_fetch("reccobeats:recs", key, fetch, ttl=7 * DAY)
        return [RecTrack.model_validate(i) for i in items]
