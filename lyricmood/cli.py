"""Command-line entry point: `uv run lyricmood --help`."""

import asyncio
import contextlib
import json
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from lyricmood.config import get_settings
from lyricmood.context import AppContext
from lyricmood.core.errors import LyricmoodError
from lyricmood.core.spotify.auth import SpotifyAuth
from lyricmood.registry import load_tools

app = typer.Typer(no_args_is_help=True, help="Lyrics-aware personal music assistant.")
auth_app = typer.Typer(no_args_is_help=True, help="Log in to music services.")
app.add_typer(auth_app, name="auth")
console = Console()


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(1)


@auth_app.command("spotify")
def auth_spotify(
    logout: Annotated[bool, typer.Option(help="Delete the saved Spotify token.")] = False,
) -> None:
    """Log in to Spotify in your browser (PKCE; no client secret needed)."""
    try:
        auth = SpotifyAuth(get_settings())
    except LyricmoodError as exc:
        _fail(str(exc))
    if logout:
        auth.logout()
        console.print("Logged out of Spotify.")
        return
    console.print("Opening your browser to log in to Spotify...")
    auth.login()
    console.print(f"[green]Spotify: {auth.status()}[/green]")


@app.command()
def status() -> None:
    """Show which API keys are configured, login state, and cache size."""
    settings = get_settings()
    table = Table(show_header=False, box=None)
    table.add_row("data dir", str(settings.resolved_data_dir))
    if settings.spotify_client_id:
        spotify = SpotifyAuth(settings).status()
    else:
        spotify = "[red]SPOTIFY_CLIENT_ID missing[/red]"
    table.add_row("spotify", spotify)
    table.add_row("last.fm", "configured" if settings.lastfm_api_key else "not configured")
    ctx = AppContext(settings)
    for name, count in ctx.cache.stats().items():
        table.add_row(f"cache {name}", str(count))
    ctx.cache.close()
    console.print(table)


@app.command("tools")
def list_tools(
    schema: Annotated[bool, typer.Option(help="Print each tool's JSON input schema.")] = False,
) -> None:
    """List the tools that the MCP server and the chat expose."""
    for spec in load_tools().all():
        console.print(f"[bold cyan]{spec.name}[/bold cyan]  {' '.join(spec.description.split())}")
        if schema:
            console.print_json(data=spec.input_schema())


def _parse_kv(pairs: list[str]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            _fail(f"expected key=value, got {pair!r}")
        # Scalars stay strings (pydantic coerces "5" -> 5 where an int is
        # expected), so a query like "1999" isn't turned into a number.
        args[key] = raw
        if raw.startswith(("[", "{")):
            with contextlib.suppress(json.JSONDecodeError):
                args[key] = json.loads(raw)
    return args


@app.command()
def call(
    name: Annotated[str, typer.Argument(help="Tool name, e.g. search_track.")],
    args: Annotated[
        list[str] | None, typer.Argument(help='key=value pairs, e.g. query="nemo"')
    ] = None,
    json_args: Annotated[
        str | None, typer.Option("--json", help="Arguments as a JSON object.")
    ] = None,
) -> None:
    """Call a tool directly and print its JSON result (what the LLM would see)."""
    payload = json.loads(json_args) if json_args else _parse_kv(args or [])
    registry = load_tools()

    async def run() -> str:
        async with AppContext() as ctx:
            return await registry.call_json(ctx, name, payload)

    try:
        console.print_json(asyncio.run(run()))
    except (LyricmoodError, KeyError) as exc:
        _fail(str(exc).strip("'\""))
    except ValidationError as exc:
        _fail(f"invalid arguments for {name}:\n{exc}")


def main() -> None:
    app()
