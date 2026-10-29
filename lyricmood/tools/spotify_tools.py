from typing import Literal

from pydantic import BaseModel, Field

from lyricmood.context import AppContext
from lyricmood.core.spotify.client import parse_track_id
from lyricmood.registry import tool
from lyricmood.tools.lyrics_tools import TrackProfile, build_profile
from lyricmood.tools.schemas import TrackOut, format_ms


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


class NowPlayingInput(BaseModel):
    with_profile: bool = Field(
        default=True,
        description="Include the song's full profile (lyrics, features, tags).",
    )


class NowPlayingOutput(BaseModel):
    status: Literal["playing", "paused", "nothing_playing", "not_a_song"]
    progress: str | None = None
    track: TrackOut | None = None
    profile: TrackProfile | None = None


@tool
async def now_playing(ctx: AppContext, args: NowPlayingInput) -> NowPlayingOutput:
    """The song currently playing on the user's Spotify, with its profile
    (lyrics, audio features, tags) so you can explain it right away. Use it
    when the user says "this song", "what's playing", etc."""
    current = await ctx.spotify.currently_playing()
    if current is None:
        return NowPlayingOutput(status="nothing_playing")
    if current.track is None:
        return NowPlayingOutput(status="not_a_song")
    return NowPlayingOutput(
        status="playing" if current.is_playing else "paused",
        progress=format_ms(current.progress_ms) if current.progress_ms is not None else None,
        track=None if args.with_profile else TrackOut.of(current.track),
        profile=await build_profile(ctx, current.track) if args.with_profile else None,
    )
