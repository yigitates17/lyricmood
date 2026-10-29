"""LRCLIB, Last.fm and ReccoBeats clients against recorded responses."""

import httpx
import pytest

from lyricmood.core.errors import ConfigError
from lyricmood.core.lastfm import API_BASE as LASTFM
from lyricmood.core.lrclib import API_BASE as LRCLIB
from lyricmood.core.models import Track
from lyricmood.core.reccobeats import API_BASE as RECCO
from tests.conftest import load_fixture

NEMO = Track(
    spotify_id="1RNwniZkJqdw9LMkkyynjx",
    title="Nemo - Remastered",
    artists=["Nightwish"],
    duration_ms=276_200,
)
CARNIVAL = Track(
    spotify_id="2DtPEKs3Pj7E06EoWW4vw8",
    title="Carnival of Rust",
    artists=["Poets of the Fall"],
    duration_ms=262_946,
)

# -- LRCLIB ------------------------------------------------------------------


async def test_lrclib_exact_match_is_cached(lrclib, router):
    route = router.get(f"{LRCLIB}/get").respond(json=load_fixture("lrclib_get_nemo.json"))

    lyrics = await lrclib.find_lyrics(NEMO)
    assert lyrics.match == "exact"
    assert lyrics.plain.startswith("Paper boats")
    params = route.calls[0].request.url.params
    assert (params["track_name"], params["artist_name"], params["duration"]) == (
        "Nemo",  # version suffix stripped
        "Nightwish",
        "276",
    )
    assert route.calls[0].request.headers["User-Agent"] == "lyricmood-tests"

    assert await lrclib.find_lyrics(NEMO) == lyrics
    assert route.call_count == 1


async def test_lrclib_search_fallback_matches_names_closest_duration_wins(lrclib, router):
    router.get(f"{LRCLIB}/get").respond(404, json={"code": 404})
    router.get(f"{LRCLIB}/search").respond(json=load_fixture("lrclib_search_carnival.json"))

    lyrics = await lrclib.find_lyrics(CARNIVAL)
    # "(Live)" and "Someone Else" don't match names; of the two that do,
    # 262.4s is closer to Spotify's 262.9s than 259s.
    assert lyrics.match == "search"
    assert lyrics.lrclib_id == 101


def search_record(id: int, duration: float, *, lyrics: str | None, instrumental=False) -> dict:
    return {
        "id": id,
        "trackName": "Carnival of Rust",
        "artistName": "Poets of the Fall",
        "duration": duration,
        "instrumental": instrumental,
        "plainLyrics": lyrics,
    }


async def test_lrclib_duration_never_rejects_a_name_match(lrclib, router):
    router.get(f"{LRCLIB}/get").respond(404)
    router.get(f"{LRCLIB}/search").respond(json=[search_record(7, 221.0, lyrics="Invented")])
    assert (await lrclib.find_lyrics(CARNIVAL)).lrclib_id == 7


async def test_lrclib_prefers_lyrics_over_closer_instrumental(lrclib, router):
    router.get(f"{LRCLIB}/get").respond(404)
    router.get(f"{LRCLIB}/search").respond(
        json=[
            search_record(1, 263.0, lyrics=None, instrumental=True),
            search_record(2, 240.0, lyrics="Invented"),
        ]
    )
    lyrics = await lrclib.find_lyrics(CARNIVAL)
    assert (lyrics.lrclib_id, lyrics.plain, lyrics.instrumental) == (2, "Invented", False)


async def test_lrclib_instrumental_only(lrclib, router):
    router.get(f"{LRCLIB}/get").respond(404)
    router.get(f"{LRCLIB}/search").respond(
        json=[search_record(1, 263.0, lyrics=None, instrumental=True)]
    )
    lyrics = await lrclib.find_lyrics(CARNIVAL)
    assert (lyrics.plain, lyrics.instrumental) == (None, True)


async def test_lrclib_miss_is_cached(lrclib, router):
    router.get(f"{LRCLIB}/get").respond(404)
    search = router.get(f"{LRCLIB}/search").respond(json=[])

    assert await lrclib.find_lyrics(CARNIVAL) is None
    assert await lrclib.find_lyrics(CARNIVAL) is None
    assert search.call_count == 1


async def test_lrclib_busy_server_is_retried(lrclib, router):
    route = router.get(f"{LRCLIB}/get").mock(
        side_effect=[
            httpx.Response(503, json={"name": "ServerOverloaded"}),
            httpx.Response(200, json=load_fixture("lrclib_get_nemo.json")),
        ]
    )
    assert (await lrclib.find_lyrics(NEMO)).match == "exact"
    assert route.call_count == 2


# -- Last.fm -----------------------------------------------------------------


def lastfm_route(router, method: str):
    return router.get(LASTFM, params={"method": method})


async def test_lastfm_tags_filter_noise(lastfm, router):
    route = lastfm_route(router, "artist.getTopTags").respond(
        json=load_fixture("lastfm_artist_tags.json")
    )
    tags = await lastfm.artist_tags("Poets of the Fall")
    # Counts below 5 (e.g. "indie rock": 2) are dropped.
    assert [t.name for t in tags] == ["alternative rock", "rock", "finnish", "alternative", "indie"]
    assert route.calls[0].request.url.params["api_key"] == "test-key"

    await lastfm.artist_tags("poets of the fall")
    assert route.call_count == 1


async def test_lastfm_blocklist_and_artist_name(lastfm, router):
    lastfm_route(router, "artist.getTopTags").respond(
        json={
            "toptags": {
                "tag": [
                    {"name": "Seen Live", "count": 100},
                    {"name": "Nightwish", "count": 90},
                    {"name": "symphonic metal", "count": 80},
                ]
            }
        }
    )
    assert [t.name for t in await lastfm.artist_tags("Nightwish")] == ["symphonic metal"]


async def test_lastfm_not_found_is_empty(lastfm, router):
    lastfm_route(router, "track.getTopTags").respond(
        json=load_fixture("lastfm_error_not_found.json")
    )
    assert await lastfm.track_tags("Nobody", "Nothing") == []


async def test_lastfm_bad_key_is_config_error(lastfm, router):
    lastfm_route(router, "artist.getTopTags").respond(
        403, json=load_fixture("lastfm_error_bad_key.json")
    )
    with pytest.raises(ConfigError, match="LASTFM_API_KEY"):
        await lastfm.artist_tags("x")


async def test_lastfm_rate_limit_error_in_body_is_retried(lastfm, router):
    route = lastfm_route(router, "artist.getTopTracks").mock(
        side_effect=[
            httpx.Response(200, json={"error": 29, "message": "Rate Limit Exceeded"}),
            httpx.Response(200, json=load_fixture("lastfm_artist_toptracks.json")),
        ]
    )
    tracks = await lastfm.artist_top_tracks("Poets of the Fall", limit=3)
    assert route.call_count == 2
    assert tracks[0].title == "Carnival of Rust"
    assert tracks[0].artist == "Poets of the Fall"
    assert tracks[0].listeners == 250834  # parsed from a string


# -- ReccoBeats --------------------------------------------------------------


async def test_reccobeats_features_with_gaps_are_cached(reccobeats, router):
    route = router.get(f"{RECCO}/audio-features").respond(
        json=load_fixture("reccobeats_audio_features.json")
    )
    ids = [CARNIVAL.spotify_id, NEMO.spotify_id, "0" * 22]

    features = await reccobeats.audio_features(ids)
    assert features[CARNIVAL.spotify_id].valence == pytest.approx(0.124)
    assert features[NEMO.spotify_id].key == 0
    assert features["0" * 22] is None

    assert await reccobeats.audio_features(ids) == features
    assert route.call_count == 1


async def test_reccobeats_batches_requests(reccobeats, router):
    route = router.get(f"{RECCO}/audio-features").respond(json={"content": []})
    await reccobeats.audio_features([f"{i:022d}" for i in range(45)])
    sizes = [len(c.request.url.params["ids"].split(",")) for c in route.calls]
    assert sizes == [40, 5]


async def test_reccobeats_recommendations(reccobeats, router):
    router.get(f"{RECCO}/track/recommendation").respond(
        json=load_fixture("reccobeats_recommendations.json")
    )
    recs = await reccobeats.recommendations([NEMO.spotify_id], size=3)
    assert len(recs) == 3
    assert recs[0].spotify_id == "43Q4Evn5AcIkxYZFkLr1pb"
    assert recs[0].artists == ["DE SOFFER", "Yann Muller"]
