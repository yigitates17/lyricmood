import json
from pathlib import Path

import pytest
import respx

from lyricmood.config import Settings
from lyricmood.context import AppContext
from lyricmood.core.cache import Cache
from lyricmood.core.http import RetryPolicy
from lyricmood.core.spotify.client import API_BASE, SpotifyClient

FIXTURES = Path(__file__).parent / "fixtures"
FAST_RETRY = RetryPolicy(attempts=3, wait_initial=0, wait_max=0)


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def settings(tmp_path) -> Settings:
    # _env_file=None: never read the developer's real .env in tests.
    return Settings(
        _env_file=None,
        LYRICMOOD_DATA_DIR=str(tmp_path / "data"),
        SPOTIFY_CLIENT_ID="test-client-id",
    )


@pytest.fixture
def cache(tmp_path) -> Cache:
    c = Cache(tmp_path / "cache.sqlite3")
    yield c
    c.close()


class FakeToken:
    def __init__(self):
        self.refreshes = 0

    async def __call__(self, force_refresh: bool) -> str:
        if force_refresh:
            self.refreshes += 1
        return f"token-{self.refreshes}"


@pytest.fixture
def token() -> FakeToken:
    return FakeToken()


@pytest.fixture
async def spotify(cache, token) -> SpotifyClient:
    client = SpotifyClient(token, cache, rps=1000, retry=FAST_RETRY)
    yield client
    await client.aclose()


@pytest.fixture
def spotify_api():
    with respx.mock(base_url=API_BASE, assert_all_called=False) as router:
        yield router


@pytest.fixture
async def ctx(settings, cache, spotify) -> AppContext:
    return AppContext(settings, cache=cache, spotify=spotify)
