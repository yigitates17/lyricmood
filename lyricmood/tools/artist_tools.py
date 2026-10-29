import asyncio
from dataclasses import dataclass
from typing import Literal

import polars as pl
from pydantic import BaseModel, Field

from lyricmood.context import AppContext
from lyricmood.core.errors import LyricmoodError
from lyricmood.core.lyrics import excerpt
from lyricmood.core.models import Track
from lyricmood.core.text import clean_title, normalize, same_name
from lyricmood.registry import tool
from lyricmood.tools.schemas import TrackOut

Ranking = Literal["lastfm", "spotify_search"]


@dataclass
class ArtistTop:
    artist: str
    tracks: list[Track]
    ranking: Ranking
    not_on_spotify: list[str]


async def artist_top(ctx: AppContext, artist: str, limit: int) -> ArtistTop:
    """Top tracks by Last.fm listeners, resolved to Spotify; Spotify search
    order when Last.fm is unavailable or has nothing."""
    if ctx.has_lastfm:
        # Ask for extra: some won't resolve, and versions of one song collapse.
        top = await ctx.lastfm.artist_top_tracks(artist, limit=limit + 10)
        if top:
            resolved = await asyncio.gather(
                *(ctx.spotify.find_by_name(t.artist or artist, t.title) for t in top)
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
            return ArtistTop(top[0].artist or artist, tracks[:limit], "lastfm", missing[:10])

    found = await ctx.spotify.search_tracks(f'artist:"{artist}"', limit=limit * 2)
    tracks = [t for t in found if any(same_name(a, artist) for a in t.artists)]
    return ArtistTop(artist, tracks[:limit], "spotify_search", [])


class GetArtistTracksInput(BaseModel):
    artist: str = Field(min_length=1, description="Artist name.")
    limit: int = Field(default=10, ge=1, le=25, description="Max tracks (1-25).")


class GetArtistTracksOutput(BaseModel):
    artist: str
    tracks: list[TrackOut]
    # "lastfm": ranked by Last.fm listeners (popularity).
    # "spotify_search": Spotify search relevance (Last.fm had nothing).
    ranking: Ranking
    # Last.fm top tracks that couldn't be found on Spotify.
    not_on_spotify: list[str] | None = None


@tool
async def get_artist_tracks(ctx: AppContext, args: GetArtistTracksInput) -> GetArtistTracksOutput:
    """An artist's most popular tracks (by Last.fm listeners), resolved to
    Spotify. Falls back to Spotify search order when Last.fm has no data."""
    top = await artist_top(ctx, args.artist, args.limit)
    return GetArtistTracksOutput(
        artist=top.artist,
        tracks=[TrackOut.of(t) for t in top.tracks],
        ranking=top.ranking,
        not_on_spotify=top.not_on_spotify or None,
    )


class AnalyzeArtistsInput(BaseModel):
    artists: list[str] = Field(min_length=1, max_length=6, description="1-6 artist names.")
    tracks_per_artist: int = Field(
        default=5, ge=1, le=8, description="Top songs sampled per artist (1-8)."
    )


class SongSample(BaseModel):
    title: str
    year: str | None = None
    # Opening stanza + chorus, cut to ~500 characters. None: no lyrics found.
    excerpt: str | None = None


class MoodAverages(BaseModel):
    """Means over the sampled songs; 0-1 scales (valence = musical positivity)."""

    valence: float | None = None
    energy: float | None = None
    danceability: float | None = None
    acousticness: float | None = None
    songs_with_features: int


class ArtistBundle(BaseModel):
    artist: str
    tags: list[str] = []
    mood: MoodAverages | None = None
    songs: list[SongSample]


class AnalyzeArtistsOutput(BaseModel):
    artists: list[ArtistBundle]
    # Last.fm tags shared by 2+ of the artists.
    shared_tags: dict[str, list[str]] | None = None
    missing: dict[str, str] | None = None


@tool
async def analyze_artists(ctx: AppContext, args: AnalyzeArtistsInput) -> AnalyzeArtistsOutput:
    """Sample each artist's most popular songs (lyrics excerpts, average audio
    mood, Last.fm tags) so you can find what they have in common, e.g. to
    answer "why do I like these artists?". Read the excerpts for recurring
    themes, imagery, perspective and emotional tone; use shared_tags and
    mood averages for the musical side."""
    missing: dict[str, str] = {}
    names = list(dict.fromkeys(a.strip() for a in args.artists if a.strip()))

    tops = await asyncio.gather(
        *(artist_top(ctx, a, args.tracks_per_artist) for a in names), return_exceptions=True
    )
    artists: list[ArtistTop] = []
    for name, top in zip(names, tops, strict=True):
        if isinstance(top, LyricmoodError):
            missing[name] = str(top)
        elif isinstance(top, BaseException):
            raise top
        else:
            artists.append(top)

    all_tracks = [t for a in artists for t in a.tracks]
    lyrics = await asyncio.gather(
        *(ctx.lrclib.find_lyrics(t) for t in all_tracks), return_exceptions=True
    )
    texts: dict[str, str] = {}
    for track, result in zip(all_tracks, lyrics, strict=True):
        if isinstance(result, BaseException) and not isinstance(result, LyricmoodError):
            raise result
        if not isinstance(result, BaseException) and result and result.plain:
            texts[track.spotify_id] = result.plain

    tags: dict[str, list[str]] = {}
    if ctx.has_lastfm:
        for a in artists:
            try:
                tags[a.artist] = [t.name for t in await ctx.lastfm.artist_tags(a.artist)]
            except LyricmoodError as exc:
                missing[f"tags:{a.artist}"] = str(exc)
    else:
        missing["tags"] = "LASTFM_API_KEY is not set"

    moods = await _mood_averages(ctx, artists, missing)

    bundles = [
        ArtistBundle(
            artist=a.artist,
            tags=tags.get(a.artist, []),
            mood=moods.get(a.artist),
            songs=[
                SongSample(
                    title=t.title,
                    year=(t.release_date or "")[:4] or None,
                    excerpt=excerpt(texts[t.spotify_id], ctx.settings.excerpt_max_chars)
                    if t.spotify_id in texts
                    else None,
                )
                for t in a.tracks
            ],
        )
        for a in artists
    ]

    by_tag: dict[str, list[str]] = {}
    for artist, artist_tags in tags.items():
        for tag in artist_tags:
            by_tag.setdefault(tag, []).append(artist)
    shared = {tag: who for tag, who in by_tag.items() if len(who) > 1}

    return AnalyzeArtistsOutput(
        artists=bundles,
        shared_tags=shared or None,
        missing=missing or None,
    )


async def _mood_averages(
    ctx: AppContext, artists: list[ArtistTop], missing: dict[str, str]
) -> dict[str, MoodAverages]:
    ids = [t.spotify_id for a in artists for t in a.tracks]
    try:
        features = await ctx.reccobeats.audio_features(ids)
    except LyricmoodError as exc:
        missing["mood"] = str(exc)
        return {}
    rows = [
        {
            "artist": a.artist,
            **f.model_dump(include={"valence", "energy", "danceability", "acousticness"}),
        }
        for a in artists
        for t in a.tracks
        if (f := features.get(t.spotify_id))
    ]
    if not rows:
        missing["mood"] = "ReccoBeats has no audio features for these songs"
        return {}
    cols = ["valence", "energy", "danceability", "acousticness"]
    df = (
        pl.DataFrame(rows, schema={"artist": pl.String, **{c: pl.Float64 for c in cols}})
        .group_by("artist")
        .agg(*(pl.col(c).mean().round(2) for c in cols), songs_with_features=pl.len())
    )
    return {
        row["artist"]: MoodAverages(**{k: v for k, v in row.items() if k != "artist"})
        for row in df.to_dicts()
    }
