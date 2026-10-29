import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from lyricmood.context import AppContext
from lyricmood.core.text import clean_title, normalize, same_name
from lyricmood.registry import tool
from lyricmood.tools.schemas import TrackOut


class GetArtistTracksInput(BaseModel):
    artist: str = Field(min_length=1, description="Artist name.")
    limit: int = Field(default=10, ge=1, le=25, description="Max tracks (1-25).")


class GetArtistTracksOutput(BaseModel):
    artist: str
    tracks: list[TrackOut]
    # "lastfm": ranked by Last.fm listeners (popularity).
    # "spotify_search": Spotify search relevance (Last.fm had nothing).
    ranking: Literal["lastfm", "spotify_search"]
    # Last.fm top tracks that couldn't be found on Spotify.
    not_on_spotify: list[str] | None = None


@tool
async def get_artist_tracks(ctx: AppContext, args: GetArtistTracksInput) -> GetArtistTracksOutput:
    """An artist's most popular tracks (by Last.fm listeners), resolved to
    Spotify. Falls back to Spotify search order when Last.fm has no data."""
    if ctx.has_lastfm:
        # Ask for extra: some won't resolve, and versions of one song collapse.
        top = await ctx.lastfm.artist_top_tracks(args.artist, limit=args.limit + 10)
        if top:
            resolved = await asyncio.gather(
                *(ctx.spotify.find_by_name(t.artist or args.artist, t.title) for t in top)
            )
            tracks, missing, seen = [], [], set()
            for lfm, track in zip(top, resolved, strict=True):
                if track is None:
                    missing.append(lfm.title)
                    continue
                key = normalize(clean_title(track.title))
                if key in seen or track.spotify_id in {t.spotify_id for t in tracks}:
                    continue
                seen.add(key)
                tracks.append(track)
            return GetArtistTracksOutput(
                artist=top[0].artist or args.artist,
                tracks=[TrackOut.of(t) for t in tracks[: args.limit]],
                ranking="lastfm",
                not_on_spotify=missing[:10] or None,
            )

    found = await ctx.spotify.search_tracks(f'artist:"{args.artist}"', limit=args.limit * 2)
    tracks = [t for t in found if any(same_name(a, args.artist) for a in t.artists)]
    return GetArtistTracksOutput(
        artist=args.artist,
        tracks=[TrackOut.of(t) for t in tracks[: args.limit]],
        ranking="spotify_search",
    )
