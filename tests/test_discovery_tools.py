"""find_similar_by_lyrics and analyze_artists end to end, all APIs mocked.

A small fake "world" of songs: lyrics about rain and the sea should rank
above lyrics about dancing when the seed is about rain and the sea.
"""

import json

import httpx
import pytest

from lyricmood.core.errors import NotFound
from lyricmood.core.lastfm import API_BASE as LASTFM
from lyricmood.core.lrclib import API_BASE as LRCLIB
from lyricmood.core.reccobeats import API_BASE as RECCO
from lyricmood.core.spotify.client import API_BASE as SPOTIFY
from lyricmood.registry import load_tools
from tests.conftest import load_fixture

PAD = "and the night goes on and on"
# title -> (artist, spotify id, invented lyrics or None)
WORLD = {
    "Seed Song": (
        "Seed Band",
        "S" * 22,
        f"rain falls on the grey sea {PAD}\n\nthe sea takes the rain away {PAD}",
    ),
    "Sea Twin": (
        "Wave Band",
        "W" * 22,
        f"grey sea and falling rain {PAD}\n\nrain on the sea tonight {PAD}",
    ),
    "Dance Floor": (
        "Party Band",
        "P" * 22,
        f"dance dance on the floor {PAD}\n\nlights up party all {PAD}",
    ),
    "Recco Pick": (
        "Recco Band",
        "R" * 22,
        f"the sea is calling rain {PAD}\n\nwe dance a little {PAD}",
    ),
    "No Words": ("Quiet Band", "Q" * 22, None),
}
BY_ID = {sid: (title, artist, lyrics) for title, (artist, sid, lyrics) in WORLD.items()}


def spotify_track(title: str) -> dict:
    artist, sid, _ = WORLD[title]
    item = load_fixture("spotify_track.json")
    return {
        **item,
        "id": sid,
        "name": title,
        "external_ids": {},
        "artists": [{**item["artists"][0], "name": artist, "id": "A" * 22}],
    }


def mock_world(router, *, similar=("Sea Twin", "Dance Floor", "No Words"), tags=None):
    def spotify_search(request):
        q = request.url.params["q"]
        title = q.split('track:"')[1].rstrip('"') if 'track:"' in q else q
        items = [spotify_track(title)] if title in WORLD else []
        return httpx.Response(200, json={"tracks": {"items": items, "next": None}})

    def spotify_get(request):
        sid = request.url.path.rsplit("/", 1)[1]
        return httpx.Response(200, json=spotify_track(BY_ID[sid][0]))

    def lrclib_get(request):
        title = request.url.params["track_name"]
        lyrics = WORLD.get(title, (None, None, None))[2]
        if lyrics is None:
            return httpx.Response(404)
        return httpx.Response(
            200,
            json={"id": 1, "trackName": title, "artistName": WORLD[title][0], "duration": 276.0,
                  "instrumental": False, "plainLyrics": lyrics},
        )  # fmt: skip

    def lastfm(request):
        method = request.url.params["method"]
        if method == "track.getSimilar":
            items = [{"name": t, "artist": {"name": WORLD[t][0]}, "match": "0.5"} for t in similar]
            return httpx.Response(200, json={"similartracks": {"track": items}})
        if method in ("track.getTopTags", "artist.getTopTags"):
            items = (
                [
                    {"name": t, "count": 100}
                    for t in (tags or {}).get(request.url.params["artist"], [])
                ]
                if method == "artist.getTopTags"
                else []
            )
            return httpx.Response(200, json={"toptags": {"tag": items}})
        if method == "artist.getTopTracks":
            artist = request.url.params["artist"]
            items = [
                {"name": t, "artist": {"name": a}} for t, (a, _, _) in WORLD.items() if a == artist
            ]
            return httpx.Response(200, json={"toptracks": {"track": items}})
        return httpx.Response(200, json={})

    router.get(url__startswith=f"{SPOTIFY}/search").mock(side_effect=spotify_search)
    router.get(url__regex=rf"{SPOTIFY}/tracks/\w+").mock(side_effect=spotify_get)
    router.get(f"{LRCLIB}/get").mock(side_effect=lrclib_get)
    router.get(f"{LRCLIB}/search").respond(json=[])
    router.get(url__startswith=LASTFM).mock(side_effect=lastfm)
    router.get(f"{RECCO}/track/recommendation").respond(
        json={"content": [{"trackTitle": "Recco Pick", "artists": [{"name": "Recco Band"}],
                           "href": f"https://open.spotify.com/track/{'R' * 22}"}]}
    )  # fmt: skip
    features = {"S": (0.2, 0.3), "W": (0.9, 0.9), "P": (0.25, 0.3), "R": (0.2, 0.35)}
    router.get(f"{RECCO}/audio-features").respond(
        json={"content": [{"href": f"https://open.spotify.com/track/{k * 22}", "valence": v, "energy": e}
                          for k, (v, e) in features.items()]}
    )  # fmt: skip


async def call(ctx, name, **args) -> dict:
    return json.loads(await load_tools().call_json(ctx, name, args))


async def test_find_similar_ranks_by_lyrics(sim_ctx, router):
    mock_world(router)
    out = await call(sim_ctx, "find_similar_by_lyrics", track="S" * 22, k=5)

    titles = [r["track"]["title"] for r in out["results"]]
    assert titles[0] == "Sea Twin"
    assert titles[-1] == "Dance Floor"
    assert set(titles) == {"Sea Twin", "Recco Pick", "Dance Floor"}
    assert out["candidates"] == 4  # "No Words" is a candidate without lyrics
    assert out["candidates_with_lyrics"] == 3
    top = out["results"][0]
    assert top["found_via"] == ["lastfm_similar"]
    assert "sea" in top["matching_lines"]["this"]
    assert "mood" not in top


async def test_find_similar_mood_blend(sim_ctx, router):
    mock_world(router)
    out = await call(sim_ctx, "find_similar_by_lyrics", track="S" * 22, k=3, mood_weight=0.9)
    # Sea Twin's lyrics match best, but its mood (0.9, 0.9) is far from the seed's.
    assert out["results"][0]["track"]["title"] != "Sea Twin"
    assert all("mood" in r for r in out["results"])


async def test_find_similar_without_expanding_uses_earlier_results(sim_ctx, router):
    mock_world(router)
    await call(sim_ctx, "find_similar_by_lyrics", track="S" * 22)
    lastfm_calls = sum(1 for c in router.calls if str(c.request.url).startswith(LASTFM))

    out = await call(sim_ctx, "find_similar_by_lyrics", track="S" * 22, expand_pool=False)
    assert out["results"][0]["track"]["title"] == "Sea Twin"
    assert all(r["found_via"] == ["earlier_searches"] for r in out["results"])
    assert sum(1 for c in router.calls if str(c.request.url).startswith(LASTFM)) == lastfm_calls


async def test_find_similar_needs_seed_lyrics(sim_ctx, router):
    mock_world(router)
    with pytest.raises(NotFound, match="No lyrics found for No Words"):
        await call(sim_ctx, "find_similar_by_lyrics", track="Q" * 22)


async def test_analyze_artists_bundles(sim_ctx, router):
    mock_world(router, tags={"Seed Band": ["dark", "rock"], "Wave Band": ["rock", "sea"]})
    out = await call(sim_ctx, "analyze_artists", artists=["Seed Band", "Wave Band", "Quiet Band"])

    bundles = {a["artist"]: a for a in out["artists"]}
    assert bundles["Seed Band"]["songs"][0]["title"] == "Seed Song"
    assert bundles["Seed Band"]["songs"][0]["excerpt"].startswith("rain falls")
    # Fields ReccoBeats didn't return are dropped from the JSON.
    assert bundles["Seed Band"]["mood"] == {"valence": 0.2, "energy": 0.3, "songs_with_features": 1}
    assert "excerpt" not in bundles["Quiet Band"]["songs"][0]  # no lyrics
    assert out["shared_tags"] == {"rock": ["Seed Band", "Wave Band"]}
