from pydantic import BaseModel, Field

from lyricmood.context import AppContext
from lyricmood.core.spotify.client import parse_track_id
from lyricmood.registry import tool
from lyricmood.tools.schemas import TrackOut


class SearchTrackInput(BaseModel):
    query: str = Field(
        min_length=1,
        description='Free text like "nemo nightwish", Spotify field filters like '
        '"artist:Nightwish track:Nemo", or a Spotify track URL/URI/ID.',
    )
    limit: int = Field(default=5, ge=1, le=10, description="Max results (1-10).")


class SearchTrackOutput(BaseModel):
    results: list[TrackOut]


@tool
async def search_track(ctx: AppContext, args: SearchTrackInput) -> SearchTrackOutput:
    """Search Spotify for tracks. Returns compact track info (id, title,
    artists, album, year, duration, isrc). Use the returned `id` to refer to a
    track in other tools. A Spotify link or ID returns exactly that track."""
    if track_id := parse_track_id(args.query):
        tracks = [await ctx.spotify.get_track(track_id)]
    else:
        tracks = await ctx.spotify.search_tracks(args.query, limit=args.limit)
    return SearchTrackOutput(results=[TrackOut.of(t) for t in tracks])
