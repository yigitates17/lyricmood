"""MCP server: exposes every registry tool over stdio (FastMCP).

Run with `lyricmood-mcp` (or `uv run lyricmood mcp`). The tool list comes
from `lyricmood.registry`, the same source the terminal chat uses, so a tool
added to `lyricmood/tools/` shows up here without touching this file.

stdout carries the MCP protocol: nothing in the server may print to it.
"""

import inspect
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import Tool
from pydantic import Field, ValidationError
from pydantic_core import PydanticUndefined

from lyricmood.context import AppContext
from lyricmood.core.errors import LyricmoodError
from lyricmood.registry import ToolRegistry, ToolSpec, load_tools

INSTRUCTIONS = """\
lyricmood is the user's personal, lyrics-aware music assistant (Spotify,
LRCLIB lyrics, Last.fm tags, ReccoBeats audio features).

The user is often not a native speaker of the song's language. When asked
what a song means or why they might like it, call get_track_profile (or
get_lyrics), then explain the themes in plain words, translate key lines,
and explain idioms, slang and cultural references. Quote only short
fragments of the lyrics.

"This song" / "what's playing" means now_playing. Refer to tracks by the
`id` returned from search_track. find_similar_by_lyrics can take about a
minute the first time for a song; say so before calling it. Tool results
include a `missing` field when a source had no data; mention it briefly
instead of guessing.
"""


def _parameters(spec: ToolSpec) -> tuple[list[inspect.Parameter], dict[str, Any]]:
    """Flatten the tool's input model into keyword parameters, so MCP
    clients see `track`, `k`, ... instead of one nested `args` object."""
    params, annotations = [], {}
    for name, field in spec.input_model.model_fields.items():
        annotation = Annotated[
            (field.annotation, *field.metadata, Field(description=field.description))
        ]
        default = (
            inspect.Parameter.empty
            if field.is_required()
            else (field.default if field.default is not PydanticUndefined else None)
        )
        params.append(
            inspect.Parameter(
                name, inspect.Parameter.KEYWORD_ONLY, annotation=annotation, default=default
            )
        )
        annotations[name] = annotation
    return params, annotations


def _make_tool(spec: ToolSpec, registry: ToolRegistry, get_ctx: Callable[[], AppContext]) -> Tool:
    async def run(**kwargs: Any) -> str:
        try:
            return await registry.call_json(get_ctx(), spec.name, kwargs)
        except LyricmoodError as exc:
            raise ToolError(str(exc)) from None
        except ValidationError as exc:
            raise ToolError(f"Invalid arguments: {exc}") from None

    params, annotations = _parameters(spec)
    run.__name__ = spec.name
    run.__doc__ = spec.description
    run.__signature__ = inspect.Signature(params, return_annotation=str)
    run.__annotations__ = {**annotations, "return": str}
    # output_schema=None: results go out as one compact JSON text block.
    # A structured copy next to it would double the tokens in some clients.
    return Tool.from_function(run, name=spec.name, description=spec.description, output_schema=None)


def build_server(context_factory: Callable[[], AppContext] = AppContext) -> FastMCP:
    registry = load_tools()
    state: dict[str, AppContext] = {}

    @asynccontextmanager
    async def lifespan(_server):
        state["ctx"] = context_factory()
        try:
            yield {}
        finally:
            await state.pop("ctx").aclose()

    def get_ctx() -> AppContext:
        return state["ctx"]

    server = FastMCP("lyricmood", instructions=INSTRUCTIONS, lifespan=lifespan)
    for spec in registry.all():
        server.add_tool(_make_tool(spec, registry, get_ctx))
    return server


def main() -> None:
    build_server().run(show_banner=False)
