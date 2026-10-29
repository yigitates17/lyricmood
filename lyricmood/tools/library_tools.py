from typing import Literal

from pydantic import BaseModel, Field

from lyricmood.context import AppContext
from lyricmood.core.errors import InvalidInput
from lyricmood.core.spotify.client import Playlist, parse_playlist_id
from lyricmood.registry import tool
from lyricmood.tools.schemas import TrackOut

PLAYLIST_ARG = "Spotify playlist ID, URL or URI (use `id` from get_my_playlists)."


def playlist_id_or_fail(ref: str) -> str:
    if pid := parse_playlist_id(ref):
        return pid
    raise InvalidInput(
        f"{ref!r} is not a Spotify playlist ID or link. Use get_my_playlists to find the ID."
    )


class TopTracksInput(BaseModel):
    time_range: Literal["short_term", "medium_term", "long_term"] = Field(
        default="medium_term",
        description="short_term ≈ last 4 weeks, medium_term ≈ 6 months, long_term ≈ 1 year.",
    )
    limit: int = Field(default=20, ge=1, le=50, description="How many tracks (1-50).")


class TracksOutput(BaseModel):
    tracks: list[TrackOut]
    total: int | None = None


@tool
async def get_my_top_tracks(ctx: AppContext, args: TopTracksInput) -> TracksOutput:
    """The user's most-played Spotify tracks over a time range, most played
    first. Good starting point for "why do I like what I like?"."""
    tracks = await ctx.spotify.top_tracks(args.time_range, args.limit)
    return TracksOutput(tracks=[TrackOut.of(t) for t in tracks])


class SavedTracksInput(BaseModel):
    limit: int = Field(default=20, ge=1, le=50, description="How many tracks (1-50).")
    offset: int = Field(default=0, ge=0, description="Skip this many (for paging).")


@tool
async def get_saved_tracks(ctx: AppContext, args: SavedTracksInput) -> TracksOutput:
    """The user's Liked Songs on Spotify, most recently liked first. `total`
    is the full count; page with `offset`."""
    tracks, total = await ctx.spotify.saved_tracks(args.limit, args.offset)
    return TracksOutput(tracks=[TrackOut.of(t) for t in tracks], total=total)


class PlaylistOut(BaseModel):
    id: str
    name: str
    songs: int | None = None
    # True when the user can read and edit its songs (owner or collaborator).
    editable: bool
    public: bool | None = None
    url: str

    @classmethod
    def of(cls, p: Playlist, user_id: str) -> "PlaylistOut":
        return cls(
            id=p.id,
            name=p.name,
            songs=p.total,
            editable=p.owner_id == user_id or p.collaborative,
            public=p.public,
            url=p.url,
        )


class MyPlaylistsInput(BaseModel):
    limit: int = Field(default=20, ge=1, le=50, description="How many playlists (1-50).")
    offset: int = Field(default=0, ge=0, description="Skip this many (for paging).")


class MyPlaylistsOutput(BaseModel):
    playlists: list[PlaylistOut]
    total: int


@tool
async def get_my_playlists(ctx: AppContext, args: MyPlaylistsInput) -> MyPlaylistsOutput:
    """Playlists in the user's Spotify library (own and followed). Only
    `editable` ones can be read song-by-song or changed: Spotify hides the
    contents of other people's playlists from apps since 2026."""
    playlists, total = await ctx.spotify.my_playlists(args.limit, args.offset)
    user = await ctx.spotify.user_id()
    return MyPlaylistsOutput(playlists=[PlaylistOut.of(p, user) for p in playlists], total=total)


class PlaylistTracksInput(BaseModel):
    playlist: str = Field(min_length=1, description=PLAYLIST_ARG)
    limit: int = Field(default=100, ge=1, le=500, description="Max songs to return (1-500).")


class PlaylistTracksOutput(BaseModel):
    playlist: PlaylistOut
    tracks: list[TrackOut]


@tool
async def get_playlist_tracks(ctx: AppContext, args: PlaylistTracksInput) -> PlaylistTracksOutput:
    """The songs of one of the user's own (or collaborative) playlists, in
    order. Needed before reordering or removing songs."""
    pid = playlist_id_or_fail(args.playlist)
    playlist = await ctx.spotify.playlist(pid)
    tracks = await ctx.spotify.playlist_tracks(pid, limit=args.limit)
    return PlaylistTracksOutput(
        playlist=PlaylistOut.of(playlist, await ctx.spotify.user_id()),
        tracks=[TrackOut.of(t) for t in tracks],
    )
