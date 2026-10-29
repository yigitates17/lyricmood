"""Library and playlist tools against a mocked Spotify (no real account writes)."""

import json

import httpx
import pytest

from lyricmood.core.errors import InvalidInput
from lyricmood.core.spotify.client import API_BASE as SPOTIFY
from lyricmood.registry import load_tools
from tests.conftest import load_fixture

PL = "P" * 22
ME = "me-user"


def track_json(tid: str, title: str = "Song") -> dict:
    return {**load_fixture("spotify_track.json"), "id": tid, "name": title}


def ids(*letters: str) -> list[str]:
    return [c * 22 for c in letters]


class FakePlaylist:
    """Just enough of Spotify's playlist endpoints to check what was sent."""

    def __init__(self, router, tracks: list[str], owner: str = ME, extra_items: int = 0):
        self.tracks = list(tracks)
        self.owner = owner
        self.extra = extra_items  # episodes / local files
        self.writes: list[tuple[str, dict]] = []
        router.get(f"{SPOTIFY}/me").respond(json={"id": ME})
        router.get(f"{SPOTIFY}/playlists/{PL}").mock(side_effect=self.meta)
        router.get(f"{SPOTIFY}/playlists/{PL}/items").mock(side_effect=self.items)
        router.route(
            url=f"{SPOTIFY}/playlists/{PL}/items", method__in=["POST", "PUT", "DELETE"]
        ).mock(side_effect=self.write)

    def meta(self, request):
        return httpx.Response(200, json={"id": PL, "name": "Mine", "owner": {"id": self.owner},
                                         "items": {"total": len(self.tracks) + self.extra}, "snapshot_id": "s0"})  # fmt: skip

    def items(self, request):
        entries = [
            {"item": {**track_json(t), "type": "track"}, "is_local": False} for t in self.tracks
        ]
        entries += [{"item": {"type": "episode", "id": "E" * 22}, "is_local": False}] * self.extra
        return httpx.Response(200, json={"items": entries, "next": None})

    def write(self, request):
        body = json.loads(request.content)
        self.writes.append((request.method, body))
        if request.method == "POST":
            self.tracks += [u.split(":")[-1] for u in body["uris"]]
        elif request.method == "DELETE":
            gone = {i["uri"].split(":")[-1] for i in body["items"]}
            self.tracks = [t for t in self.tracks if t not in gone]
        else:
            self.tracks = [u.split(":")[-1] for u in body["uris"]]
        return httpx.Response(200, json={"snapshot_id": f"s{len(self.writes)}"})


async def call(ctx, tool_name, **args) -> dict:
    return json.loads(await load_tools().call_json(ctx, tool_name, args))


async def test_update_removes_adds_and_reorders(ctx, router):
    pl = FakePlaylist(router, ids("a", "b", "c"))
    out = await call(
        ctx, "update_spotify_playlist", playlist_id=f"https://open.spotify.com/playlist/{PL}",
        remove=ids("b"), add=ids("d", "a"), reorder=ids("d", "c", "a"),
    )  # fmt: skip
    assert pl.tracks == ids("d", "c", "a")
    assert [m for m, _ in pl.writes] == ["DELETE", "POST", "PUT"]
    assert out["added"] == 1 and out["removed"] == 1 and out["reordered"] is True
    assert out["already_there"] == 1  # "a" was already in the playlist


async def test_bad_reorder_changes_nothing(ctx, router):
    pl = FakePlaylist(router, ids("a", "b"))
    with pytest.raises(InvalidInput, match="Nothing was changed"):
        await call(
            ctx, "update_spotify_playlist", playlist_id=PL, remove=ids("a"), reorder=ids("a", "b")
        )
    assert pl.writes == []


async def test_refuses_other_peoples_playlists(ctx, router):
    pl = FakePlaylist(router, ids("a"), owner="someone-else")
    with pytest.raises(InvalidInput, match="own or collaborative"):
        await call(ctx, "update_spotify_playlist", playlist_id=PL, add=ids("b"))
    assert pl.writes == []


async def test_refuses_reorder_when_episodes_would_be_lost(ctx, router):
    FakePlaylist(router, ids("a", "b"), extra_items=1)
    with pytest.raises(InvalidInput, match="episodes or local files"):
        await call(ctx, "update_spotify_playlist", playlist_id=PL, reorder=ids("b", "a"))


async def test_track_names_are_rejected(ctx, router):
    FakePlaylist(router, [])
    with pytest.raises(InvalidInput, match="search_track"):
        await call(ctx, "update_spotify_playlist", playlist_id=PL, add=["Nemo by Nightwish"])


async def test_create_playlist_private_by_default_and_batches_adds(ctx, router):
    router.get(f"{SPOTIFY}/me").respond(json={"id": ME})
    create = router.post(f"{SPOTIFY}/me/playlists").respond(
        201, json={"id": PL, "name": "Rainy", "owner": {"id": ME}, "public": False}
    )
    add = router.post(f"{SPOTIFY}/playlists/{PL}/items").respond(201, json={"snapshot_id": "x"})
    tracks = [f"{i:022d}" for i in range(150)]

    out = await call(ctx, "create_spotify_playlist", name="Rainy", tracks=tracks)
    assert json.loads(create.calls[0].request.content) == {
        "name": "Rainy",
        "description": "",
        "public": False,
    }
    assert [len(json.loads(c.request.content)["uris"]) for c in add.calls] == [100, 50]
    assert out["added"] == 150 and out["playlist"]["editable"] is True


async def test_library_tools(ctx, router):
    router.get(f"{SPOTIFY}/me").respond(json={"id": ME})
    top = router.get(f"{SPOTIFY}/me/top/tracks").respond(
        json={"items": [track_json("a" * 22, "Top")]}
    )
    router.get(f"{SPOTIFY}/me/tracks").respond(
        json={"items": [{"track": track_json("b" * 22, "Liked")}], "total": 1808}
    )
    router.get(f"{SPOTIFY}/me/playlists").respond(
        json={"items": [{"id": PL, "name": "Mine", "owner": {"id": ME}, "items": {"total": 3}},
                        {"id": "Q" * 22, "name": "Theirs", "owner": {"id": "x"}, "items": {"total": 9}}],
              "total": 2}
    )  # fmt: skip

    out = await call(ctx, "get_my_top_tracks", time_range="short_term", limit=5)
    assert out["tracks"][0]["title"] == "Top"
    assert top.calls[0].request.url.params["time_range"] == "short_term"
    assert (await call(ctx, "get_saved_tracks"))["total"] == 1808
    playlists = (await call(ctx, "get_my_playlists"))["playlists"]
    assert [(p["name"], p["songs"], p["editable"]) for p in playlists] == [
        ("Mine", 3, True),
        ("Theirs", 9, False),
    ]


async def test_reading_someone_elses_playlist_explains_restriction(ctx, router):
    router.get(f"{SPOTIFY}/me").respond(json={"id": ME})
    router.get(f"{SPOTIFY}/playlists/{PL}").respond(
        json={"id": PL, "name": "Theirs", "owner": {"id": "x"}}
    )
    router.get(f"{SPOTIFY}/playlists/{PL}/items").respond(403, json={"error": {"status": 403}})
    with pytest.raises(Exception, match="own or collaborate"):
        await call(ctx, "get_playlist_tracks", playlist=PL)
