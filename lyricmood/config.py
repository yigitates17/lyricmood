"""Settings loaded from environment variables and the project's .env file.

The .env path is resolved relative to the package, not the current working
directory, because MCP clients launch the server from arbitrary directories.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Requested once at login so later milestones (playlists, library) don't force
# a re-login. Spotify rejects a cached token whose scopes are a subset.
SPOTIFY_SCOPES = (
    "user-top-read",
    "user-library-read",
    "playlist-read-private",
    "playlist-read-collaborative",
    "playlist-modify-private",
    "playlist-modify-public",
    "user-read-currently-playing",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("~/.lyricmood"), alias="LYRICMOOD_DATA_DIR")

    spotify_client_id: str | None = Field(default=None, alias="SPOTIFY_CLIENT_ID")
    spotify_redirect_uri: str = Field(
        default="http://127.0.0.1:8888/callback", alias="SPOTIFY_REDIRECT_URI"
    )

    lastfm_api_key: SecretStr | None = Field(default=None, alias="LASTFM_API_KEY")

    # Sent to keyless APIs (LRCLIB asks for one that identifies the app).
    user_agent: str = Field(
        default="lyricmood/0.1 (+https://github.com/yigitates17/lyricmood)",
        alias="LYRICMOOD_USER_AGENT",
    )

    # Spotify's limits are an undisclosed rolling 30s window shared by all of
    # a developer's dev-mode apps, so stay well below anything plausible.
    spotify_rps: float = Field(default=5.0, alias="LYRICMOOD_SPOTIFY_RPS")

    @property
    def resolved_data_dir(self) -> Path:
        path = self.data_dir.expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cache_path(self) -> Path:
        return self.resolved_data_dir / "cache.sqlite3"

    @property
    def spotify_token_path(self) -> Path:
        return self.resolved_data_dir / "spotify_token.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
