import json

import pytest
from fastmcp import Client
from pydantic import BaseModel, Field

from lyricmood.core.errors import NotFound
from lyricmood.mcp_server import build_server
from lyricmood.registry import load_tools, registry
from tests.conftest import load_fixture


@pytest.fixture
def server(ctx):
    return build_server(lambda: ctx)


async def test_lists_every_registry_tool_with_flat_schemas(server):
    async with Client(server) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert set(tools) == {spec.name for spec in load_tools().all()}

    schema = tools["search_track"].input_schema
    assert schema["required"] == ["query"]
    assert schema["properties"]["limit"] == {
        "default": 5,
        "description": "Max results (1-10).",
        "maximum": 10,
        "minimum": 1,
        "type": "integer",
    }
    assert tools["search_track"].description.startswith("Search Spotify for tracks.")
    assert tools["search_track"].output_schema is None


async def test_call_returns_compact_json_text(server, spotify_api):
    spotify_api.get("/search").respond(json=load_fixture("spotify_search_nemo.json"))
    async with Client(server) as client:
        result = await client.call_tool("search_track", {"query": "nemo", "limit": 3})
    assert len(result.content) == 1
    text = result.content[0].text
    assert " " not in text.split('"title"')[0]  # no pretty-printing whitespace
    assert json.loads(text)["results"][0]["title"] == "Nemo"


async def test_lyricmood_errors_become_tool_errors(server, spotify_api):
    spotify_api.get("/search").respond(json={"tracks": {"items": [], "next": None}})
    async with Client(server) as client:
        result = await client.call_tool(
            "get_lyrics", {"track": "no such song"}, raise_on_error=False
        )
    assert result.is_error
    assert "No Spotify track found" in result.content[0].text


async def test_invalid_arguments_are_reported(server):
    async with Client(server) as client:
        result = await client.call_tool("search_track", {"limit": 3}, raise_on_error=False)
    assert result.is_error
    assert "query" in result.content[0].text


async def test_new_registry_tools_appear_without_server_changes(ctx, monkeypatch):
    class PingInput(BaseModel):
        word: str = Field(description="Echoed back.")

    class PingOutput(BaseModel):
        word: str

    async def ping(ctx, args: PingInput) -> PingOutput:
        """Echo a word."""
        if args.word == "fail":
            raise NotFound("nothing here")
        return PingOutput(word=args.word)

    load_tools()
    monkeypatch.setattr(registry, "_tools", dict(registry._tools))
    registry.tool(ping)

    async with Client(build_server(lambda: ctx)) as client:
        assert "ping" in {t.name for t in await client.list_tools()}
        ok = await client.call_tool("ping", {"word": "hi"})
        bad = await client.call_tool("ping", {"word": "fail"}, raise_on_error=False)
    assert json.loads(ok.content[0].text) == {"word": "hi"}
    assert bad.is_error and "nothing here" in bad.content[0].text
