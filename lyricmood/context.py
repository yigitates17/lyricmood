"""Per-process dependency container handed to every tool.

Clients are built lazily, so a missing API key only fails the tools that
need it, with a message saying which key to set.
"""

from lyricmood.config import Settings, get_settings
from lyricmood.core.cache import Cache
from lyricmood.core.errors import ConfigError
from lyricmood.core.http import ApiClient
from lyricmood.core.lastfm import LastfmClient
from lyricmood.core.lrclib import LrclibClient
from lyricmood.core.reccobeats import ReccoBeatsClient
from lyricmood.core.spotify.auth import SpotifyAuth
from lyricmood.core.spotify.client import SpotifyClient


class AppContext:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        cache: Cache | None = None,
        spotify: SpotifyClient | None = None,
        lrclib: LrclibClient | None = None,
        lastfm: LastfmClient | None = None,
        reccobeats: ReccoBeatsClient | None = None,
    ):
        self.settings = settings or get_settings()
        self._cache = cache
        self._spotify = spotify
        self._spotify_auth: SpotifyAuth | None = None
        self._lrclib = lrclib
        self._lastfm = lastfm
        self._reccobeats = reccobeats

    @property
    def cache(self) -> Cache:
        if self._cache is None:
            self._cache = Cache(self.settings.cache_path)
        return self._cache

    @property
    def spotify_auth(self) -> SpotifyAuth:
        if self._spotify_auth is None:
            self._spotify_auth = SpotifyAuth(self.settings)
        return self._spotify_auth

    @property
    def spotify(self) -> SpotifyClient:
        if self._spotify is None:
            self._spotify = SpotifyClient(
                self.spotify_auth.access_token, self.cache, rps=self.settings.spotify_rps
            )
        return self._spotify

    @property
    def lrclib(self) -> LrclibClient:
        if self._lrclib is None:
            self._lrclib = LrclibClient(
                self.cache, user_agent=self.settings.user_agent, rps=self.settings.lrclib_rps
            )
        return self._lrclib

    @property
    def has_lastfm(self) -> bool:
        return self._lastfm is not None or self.settings.lastfm_api_key is not None

    @property
    def lastfm(self) -> LastfmClient:
        if self._lastfm is None:
            if self.settings.lastfm_api_key is None:
                raise ConfigError("LASTFM_API_KEY is not set. Add it to .env (see README).")
            self._lastfm = LastfmClient(
                self.cache,
                api_key=self.settings.lastfm_api_key.get_secret_value(),
                user_agent=self.settings.user_agent,
                rps=self.settings.lastfm_rps,
            )
        return self._lastfm

    @property
    def reccobeats(self) -> ReccoBeatsClient:
        if self._reccobeats is None:
            self._reccobeats = ReccoBeatsClient(
                self.cache, user_agent=self.settings.user_agent, rps=self.settings.reccobeats_rps
            )
        return self._reccobeats

    async def aclose(self) -> None:
        clients: list[ApiClient | None] = [
            self._spotify,
            self._lrclib,
            self._lastfm,
            self._reccobeats,
        ]
        for client in clients:
            if client is not None:
                await client.aclose()
        if self._cache is not None:
            self._cache.close()

    async def __aenter__(self) -> "AppContext":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
