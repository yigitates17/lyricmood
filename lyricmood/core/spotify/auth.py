"""Spotify OAuth (Authorization Code + PKCE) via spotipy.

spotipy is used only for the login flow and token refresh; API calls go
through the async `SpotifyClient`. The interactive login runs only from the
CLI (`lyricmood auth spotify`): the MCP server speaks over stdin/stdout, so
it must never prompt, and instead raises `AuthRequired`.
"""

import asyncio
import time

from spotipy.cache_handler import CacheFileHandler
from spotipy.oauth2 import SpotifyOauthError, SpotifyPKCE

from lyricmood.config import SPOTIFY_SCOPES, Settings
from lyricmood.core.errors import AuthRequired, ConfigError

LOGIN_HINT = "Run `uv run lyricmood auth spotify` in the lyricmood folder to log in."


class SpotifyAuth:
    def __init__(self, settings: Settings):
        if not settings.spotify_client_id:
            raise ConfigError("SPOTIFY_CLIENT_ID is not set. Add it to .env (see README).")
        self._token_path = settings.spotify_token_path
        self._pkce = SpotifyPKCE(
            client_id=settings.spotify_client_id,
            redirect_uri=settings.spotify_redirect_uri,
            scope=" ".join(SPOTIFY_SCOPES),
            cache_handler=CacheFileHandler(cache_path=str(self._token_path)),
            open_browser=True,
        )
        self._lock = asyncio.Lock()

    def login(self) -> None:
        """Interactive browser login. Blocks until the redirect is received."""
        self._pkce.get_access_token(check_cache=False)

    def logout(self) -> None:
        self._token_path.unlink(missing_ok=True)

    def cached_token_info(self) -> dict | None:
        return self._pkce.cache_handler.get_cached_token()

    def _token_sync(self, force_refresh: bool) -> str:
        info = self.cached_token_info()
        if info is None:
            raise AuthRequired(f"Not logged in to Spotify. {LOGIN_HINT}")
        try:
            if force_refresh:
                info = self._pkce.refresh_access_token(info["refresh_token"])
            else:
                # Returns None when the cached scopes don't cover SPOTIFY_SCOPES.
                info = self._pkce.validate_token(info)
        except SpotifyOauthError as exc:
            # Refresh tokens expire after six months, or may be revoked.
            raise AuthRequired(f"Spotify session expired ({exc}). {LOGIN_HINT}") from None
        if info is None:
            raise AuthRequired(f"Spotify login is missing required permissions. {LOGIN_HINT}")
        return info["access_token"]

    async def access_token(self, force_refresh: bool = False) -> str:
        async with self._lock:
            return await asyncio.to_thread(self._token_sync, force_refresh)

    def status(self) -> str:
        info = self.cached_token_info()
        if info is None:
            return "not logged in"
        if self._pkce.validate_token(info) is None:
            return "logged in, but missing scopes (log in again)"
        remaining = int(info.get("expires_at", 0) - time.time())
        return f"logged in (access token valid for {max(remaining, 0) // 60} more min)"
