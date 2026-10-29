import httpx
import pytest

from lyricmood.core.errors import AuthRequired, ConfigError, NotFound
from lyricmood.core.spotify.auth import SpotifyAuth
from lyricmood.core.spotify.client import parse_track_id
from tests.conftest import load_fixture

NEMO_ID = "1pVuCVMH2Ex4ab4bNIpw6S"


@pytest.mark.parametrize(
    "ref",
    [
        NEMO_ID,
        f"spotify:track:{NEMO_ID}",
        f"https://open.spotify.com/track/{NEMO_ID}?si=abc123",
        f"https://open.spotify.com/intl-tr/track/{NEMO_ID}",
    ],
)
def test_parse_track_id(ref):
    assert parse_track_id(ref) == NEMO_ID


@pytest.mark.parametrize("ref", ["nemo nightwish", "https://open.spotify.com/album/x", ""])
def test_parse_track_id_rejects_non_tracks(ref):
    assert parse_track_id(ref) is None


async def test_search_parses_and_caches(spotify, spotify_api, token):
    route = spotify_api.get("/search").respond(json=load_fixture("spotify_search_nemo.json"))

    tracks = await spotify.search_tracks("Nemo", limit=5)
    assert [t.title for t in tracks][:2] == ["Nemo", "Nemo - Orchestral Version"]
    assert tracks[0].isrc == "FIXTUREISRC01"
    assert tracks[1].isrc is None
    assert tracks[2].artists == ["Nightwish", "Example Guest"]
    assert route.calls[0].request.headers["Authorization"] == "Bearer token-0"
    assert route.calls[0].request.url.params["limit"] == "5"

    # Same query (case-insensitive) is served from SQLite.
    again = await spotify.search_tracks("nemo ", limit=5)
    assert again == tracks
    assert route.call_count == 1
    assert spotify.cache.get_track_by_isrc("FIXTUREISRC01").spotify_id == NEMO_ID


async def test_search_paginates_past_ten(spotify, spotify_api):
    item = load_fixture("spotify_track.json")

    def page(request):
        offset = int(request.url.params["offset"])
        limit = int(request.url.params["limit"])
        assert limit <= 10
        items = [{**item, "id": f"{offset + i:022d}"} for i in range(limit)]
        return httpx.Response(200, json={"tracks": {"items": items, "next": "more"}})

    route = spotify_api.get("/search").mock(side_effect=page)
    tracks = await spotify.search_tracks("many", limit=15)
    assert len(tracks) == 15
    assert [c.request.url.params["offset"] for c in route.calls] == ["0", "10"]


async def test_get_track_and_not_found(spotify, spotify_api):
    spotify_api.get(f"/tracks/{NEMO_ID}").respond(json=load_fixture("spotify_track.json"))
    spotify_api.get(f"/tracks/{'x' * 22}").respond(404, json={"error": {"status": 404}})

    assert (await spotify.resolve(f"spotify:track:{NEMO_ID}")).title == "Nemo"
    with pytest.raises(NotFound):
        await spotify.get_track("x" * 22)


async def test_resolve_free_text_with_no_results(spotify, spotify_api):
    spotify_api.get("/search").respond(json={"tracks": {"items": [], "next": None}})
    with pytest.raises(NotFound):
        await spotify.resolve("zzzz no such song")


def test_auth_requires_login_without_token(settings):
    auth = SpotifyAuth(settings)
    assert auth.status() == "not logged in"
    with pytest.raises(AuthRequired, match="lyricmood auth spotify"):
        auth._token_sync(force_refresh=False)


def test_auth_requires_client_id(settings):
    settings.spotify_client_id = None
    with pytest.raises(ConfigError, match="SPOTIFY_CLIENT_ID"):
        SpotifyAuth(settings)
