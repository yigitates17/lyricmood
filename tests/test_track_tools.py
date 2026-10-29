"""Milestone 2 tools end to end, with every API mocked."""

import json

import httpx

from lyricmood.context import AppContext
from lyricmood.core.lastfm import API_BASE as LASTFM
from lyricmood.core.lrclib import API_BASE as LRCLIB
from lyricmood.core.reccobeats import API_BASE as RECCO
from lyricmood.core.spotify.client import API_BASE as SPOTIFY
from lyricmood.registry import load_tools
from tests.conftest import load_fixture

TRACK_ID = "1pVuCVMH2Ex4ab4bNIpw6S"  # the id in spotify_track.json


def mock_track(router):
    router.get(f"{SPOTIFY}/tracks/{TRACK_ID}").respond(json=load_fixture("spotify_track.json"))


def features_for(spotify_id: str) -> dict:
    item = load_fixture("reccobeats_audio_features.json")["content"][1]
    return {"content": [{**item, "href": f"https://open.spotify.com/track/{spotify_id}"}]}


async def call(ctx, name, **args) -> dict:
    return json.loads(await load_tools().call_json(ctx, name, args))


async def test_track_profile_combines_sources(full_ctx, router):
    mock_track(router)
    router.get(f"{LRCLIB}/get").respond(json=load_fixture("lrclib_get_nemo.json"))
    router.get(f"{RECCO}/audio-features").respond(json=features_for(TRACK_ID))
    router.get(LASTFM, params={"method": "track.getTopTags"}).respond(
        json=load_fixture("lastfm_track_tags_empty.json")
    )
    router.get(LASTFM, params={"method": "artist.getTopTags"}).respond(
        json={"toptags": {"tag": [{"name": "Symphonic Metal", "count": 100}]}}
    )

    out = await call(full_ctx, "get_track_profile", track=TRACK_ID)
    assert out["track"]["title"] == "Nemo"
    assert out["lyrics"]["match"] == "exact"
    assert out["lyrics"]["stats"]["repeated_lines"] == [
        "Hold the lantern, hold it higher",
        "We will find the morning anyway",
    ]
    assert out["features"] == {
        "valence": 0.36,
        "energy": 0.73,
        "danceability": 0.49,
        "acousticness": 0.0,
        "instrumentalness": 0.0,
        "speechiness": 0.03,
        "tempo": 125,
        "loudness": -6.3,
        "key": "C minor",
    }
    # Track has no tags on Last.fm, so the artist's are used.
    assert out["tags"] == ["symphonic metal"]
    assert out["tags_source"] == "artist"
    assert "missing" not in out


async def test_track_profile_survives_failing_sources(full_ctx, router):
    mock_track(router)
    router.get(f"{LRCLIB}/get").respond(404)
    router.get(f"{LRCLIB}/search").respond(json=[])
    router.get(f"{RECCO}/audio-features").respond(500)
    router.get(LASTFM).respond(json={"toptags": {"tag": []}})

    out = await call(full_ctx, "get_track_profile", track=TRACK_ID)
    assert out["track"]["id"] == TRACK_ID
    assert set(out["missing"]) == {"lyrics", "features", "tags"}
    assert "HTTP 500" in out["missing"]["features"]


async def test_track_profile_without_lastfm_key(
    settings, cache, spotify, lrclib, reccobeats, router
):
    ctx = AppContext(settings, cache=cache, spotify=spotify, lrclib=lrclib, reccobeats=reccobeats)
    mock_track(router)
    router.get(f"{LRCLIB}/get").respond(json=load_fixture("lrclib_get_nemo.json"))
    router.get(f"{RECCO}/audio-features").respond(json=features_for(TRACK_ID))

    out = await call(ctx, "get_track_profile", track=TRACK_ID)
    assert out["lyrics"]["match"] == "exact"
    assert "LASTFM_API_KEY" in out["missing"]["tags"]


async def test_get_lyrics_truncates(full_ctx, router):
    full_ctx.settings.lyrics_max_chars = 70
    mock_track(router)
    router.get(f"{LRCLIB}/get").respond(json=load_fixture("lrclib_get_nemo.json"))

    out = await call(full_ctx, "get_lyrics", track=TRACK_ID)
    assert out["lyrics"]["truncated"] is True
    assert out["lyrics"]["text"] == "Paper boats on a silver river\nCounting lights that never stay"
    assert "stats" not in out["lyrics"]


async def test_get_lyrics_not_found(full_ctx, router):
    mock_track(router)
    router.get(f"{LRCLIB}/get").respond(404)
    router.get(f"{LRCLIB}/search").respond(json=[])
    out = await call(full_ctx, "get_lyrics", track=TRACK_ID)
    assert out["note"] == "Lyrics not found on LRCLIB."
    assert "lyrics" not in out


def spotify_hit(spotify_id: str, title: str, artist: str) -> dict:
    item = load_fixture("spotify_track.json")
    item = {**item, "id": spotify_id, "name": title}
    item["artists"] = [{**item["artists"][0], "name": artist}]
    return {"tracks": {"items": [item], "next": None}}


async def test_get_artist_tracks_ranks_by_lastfm(full_ctx, router):
    router.get(LASTFM, params={"method": "artist.getTopTracks"}).respond(
        json={
            "toptracks": {
                "track": [
                    {"name": "Carnival of Rust", "artist": {"name": "Poets of the Fall"}},
                    {
                        "name": "Carnival of Rust - Remastered",
                        "artist": {"name": "Poets of the Fall"},
                    },
                    {"name": "Obscure B-Side", "artist": {"name": "Poets of the Fall"}},
                    {"name": "Late Goodbye", "artist": {"name": "Poets of the Fall"}},
                ]
            }
        }
    )

    def search(request: httpx.Request) -> httpx.Response:
        q = request.url.params["q"]
        if "Obscure" in q:
            return httpx.Response(200, json={"tracks": {"items": [], "next": None}})
        title = q.split('track:"')[1].rstrip('"')
        sid = "a" * 22 if "Carnival" in title else "b" * 22
        return httpx.Response(200, json=spotify_hit(sid, title, "Poets of the Fall"))

    router.get(f"{SPOTIFY}/search").mock(side_effect=search)

    out = await call(full_ctx, "get_artist_tracks", artist="poets of the fall", limit=5)
    assert out["ranking"] == "lastfm"
    assert out["artist"] == "Poets of the Fall"
    # The remaster collapses into the original; the B-side isn't on Spotify.
    assert [t["title"] for t in out["tracks"]] == ["Carnival of Rust", "Late Goodbye"]
    assert out["not_on_spotify"] == ["Obscure B-Side"]


async def test_get_artist_tracks_without_lastfm(settings, cache, spotify, router):
    ctx = AppContext(settings, cache=cache, spotify=spotify)
    router.get(f"{SPOTIFY}/search").respond(json=load_fixture("spotify_search_nemo.json"))
    out = await call(ctx, "get_artist_tracks", artist="Nightwish", limit=2)
    assert out["ranking"] == "spotify_search"
    assert len(out["tracks"]) == 2


async def test_now_playing_nothing(full_ctx, router):
    router.get(f"{SPOTIFY}/me/player/currently-playing").respond(204)
    assert await call(full_ctx, "now_playing") == {"status": "nothing_playing"}


async def test_now_playing_track(full_ctx, router):
    router.get(f"{SPOTIFY}/me/player/currently-playing").respond(
        json=load_fixture("spotify_currently_playing.json")
    )
    out = await call(full_ctx, "now_playing", with_profile=False)
    assert out == {
        "status": "playing",
        "progress": "1:23",
        "track": {
            "id": TRACK_ID,
            "title": "Nemo",
            "artists": "Nightwish",
            "album": "Once",
            "year": "2004",
            "duration": "4:36",
            "isrc": "FIXTUREISRC01",
        },
    }


async def test_now_playing_episode(full_ctx, router):
    router.get(f"{SPOTIFY}/me/player/currently-playing").respond(
        json={"is_playing": True, "item": None, "currently_playing_type": "episode"}
    )
    assert (await call(full_ctx, "now_playing"))["status"] == "not_a_song"
