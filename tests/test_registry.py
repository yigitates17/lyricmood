import json

import pytest
from pydantic import BaseModel

from lyricmood.registry import ToolRegistry, load_tools
from tests.conftest import load_fixture


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int | None = None


def test_registry_reads_models_and_docstring():
    reg = ToolRegistry()

    @reg.tool
    async def double(ctx, args: In) -> Out:
        """Doubles x."""
        return Out(y=args.x * 2)

    spec = reg.get("double")
    assert spec.description == "Doubles x."
    assert spec.openai_schema()["function"]["parameters"]["required"] == ["x"]


def test_registry_rejects_bad_signatures():
    reg = ToolRegistry()
    with pytest.raises(TypeError):

        @reg.tool
        async def no_models(ctx, args: dict) -> dict:
            return {}


async def test_call_json_is_compact_and_drops_nulls():
    reg = ToolRegistry()

    @reg.tool
    async def nothing(ctx, args: In) -> Out:
        return Out()

    assert await reg.call_json(None, "nothing", {"x": "3"}) == "{}"


def test_builtin_tools_are_discovered():
    names = {spec.name for spec in load_tools().all()}
    assert "search_track" in names


async def test_search_track_tool(ctx, spotify_api):
    spotify_api.get("/search").respond(json=load_fixture("spotify_search_nemo.json"))
    out = json.loads(await load_tools().call_json(ctx, "search_track", {"query": "nemo"}))
    first = out["results"][0]
    assert first == {
        "id": "1pVuCVMH2Ex4ab4bNIpw6S",
        "title": "Nemo",
        "artists": "Nightwish",
        "album": "Once",
        "year": "2004",
        "duration": "4:36",
        "isrc": "FIXTUREISRC01",
    }
    assert "isrc" not in out["results"][1]


async def test_search_track_tool_accepts_links(ctx, spotify_api):
    route = spotify_api.get("/tracks/1pVuCVMH2Ex4ab4bNIpw6S").respond(
        json=load_fixture("spotify_track.json")
    )
    out = await load_tools().call(
        ctx, "search_track", {"query": "https://open.spotify.com/track/1pVuCVMH2Ex4ab4bNIpw6S"}
    )
    assert [r.title for r in out.results] == ["Nemo"]
    assert route.call_count == 1
