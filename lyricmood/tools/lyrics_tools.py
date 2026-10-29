import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from lyricmood.context import AppContext
from lyricmood.core import lyrics as lyrics_text
from lyricmood.core.errors import LyricmoodError
from lyricmood.core.lastfm import Tag
from lyricmood.core.lrclib import Lyrics
from lyricmood.core.lyrics import LyricsStats
from lyricmood.core.models import Track
from lyricmood.core.reccobeats import AudioFeatures
from lyricmood.registry import tool
from lyricmood.tools.schemas import TRACK_ARG, TrackOut

_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class TrackInput(BaseModel):
    track: str = Field(min_length=1, description=TRACK_ARG)


class LyricsOut(BaseModel):
    text: str | None = None
    truncated: bool | None = None
    instrumental: bool | None = None
    # "exact" or "search": how confidently the lyrics match this recording.
    match: str
    stats: LyricsStats | None = None

    @classmethod
    def of(cls, lyrics: Lyrics, max_chars: int, with_stats: bool) -> "LyricsOut":
        text, truncated = (None, None)
        if lyrics.plain:
            text, truncated = lyrics_text.truncate(lyrics.plain, max_chars)
        return cls(
            text=text,
            truncated=truncated or None,
            instrumental=lyrics.instrumental or None,
            match=lyrics.match,
            stats=lyrics_text.stats(lyrics.plain) if with_stats and lyrics.plain else None,
        )


class FeaturesOut(BaseModel):
    """0-1 scales except tempo (BPM) and loudness (dB)."""

    valence: float | None = None
    energy: float | None = None
    danceability: float | None = None
    acousticness: float | None = None
    instrumentalness: float | None = None
    speechiness: float | None = None
    tempo: int | None = None
    loudness: float | None = None
    key: str | None = None

    @classmethod
    def of(cls, f: AudioFeatures) -> "FeaturesOut":
        def r(v: float | None) -> float | None:
            return round(v, 2) if v is not None else None

        key = None
        if f.key is not None and 0 <= f.key < 12:
            key = f"{_KEYS[f.key]} {'major' if f.mode == 1 else 'minor'}"
        return cls(
            valence=r(f.valence),
            energy=r(f.energy),
            danceability=r(f.danceability),
            acousticness=r(f.acousticness),
            instrumentalness=r(f.instrumentalness),
            speechiness=r(f.speechiness),
            tempo=round(f.tempo) if f.tempo else None,
            loudness=round(f.loudness, 1) if f.loudness is not None else None,
            key=key,
        )


class TrackProfile(BaseModel):
    track: TrackOut
    lyrics: LyricsOut | None = None
    features: FeaturesOut | None = None
    tags: list[str] = []
    # "track" when Last.fm has tags for the song itself, else "artist".
    tags_source: Literal["track", "artist"] | None = None
    # Sources that failed or had no data, with the reason.
    missing: dict[str, str] | None = None


async def track_tags(ctx: AppContext, track: Track) -> tuple[list[Tag], str | None]:
    tags = await ctx.lastfm.track_tags(track.primary_artist, track.title)
    if tags:
        return tags, "track"
    tags = await ctx.lastfm.artist_tags(track.primary_artist)
    return tags, "artist" if tags else None


async def _features(ctx: AppContext, track: Track) -> AudioFeatures | None:
    return (await ctx.reccobeats.audio_features([track.spotify_id]))[track.spotify_id]


async def build_profile(ctx: AppContext, track: Track) -> TrackProfile:
    lyrics_r, features_r, tags_r = await asyncio.gather(
        ctx.lrclib.find_lyrics(track),
        _features(ctx, track),
        track_tags(ctx, track),
        return_exceptions=True,
    )
    for result in (lyrics_r, features_r, tags_r):
        if isinstance(result, BaseException) and not isinstance(result, LyricmoodError):
            raise result

    profile = TrackProfile(track=TrackOut.of(track))
    missing: dict[str, str] = {}

    if isinstance(lyrics_r, LyricmoodError):
        missing["lyrics"] = str(lyrics_r)
    elif lyrics_r is None:
        missing["lyrics"] = "not found on LRCLIB"
    else:
        profile.lyrics = LyricsOut.of(lyrics_r, ctx.settings.lyrics_max_chars, with_stats=True)

    if isinstance(features_r, LyricmoodError):
        missing["features"] = str(features_r)
    elif features_r is None:
        missing["features"] = "ReccoBeats has no audio features for this track"
    else:
        profile.features = FeaturesOut.of(features_r)

    if isinstance(tags_r, LyricmoodError):
        missing["tags"] = str(tags_r)
    else:
        tags, source = tags_r
        profile.tags = [t.name for t in tags]
        profile.tags_source = source
        if not tags:
            missing["tags"] = "no Last.fm tags"

    profile.missing = missing or None
    return profile


class GetLyricsOutput(BaseModel):
    track: TrackOut
    lyrics: LyricsOut | None = None
    note: str | None = None


@tool
async def get_lyrics(ctx: AppContext, args: TrackInput) -> GetLyricsOutput:
    """Get a song's plain lyrics (from LRCLIB) so you can translate, explain
    or interpret them. Long lyrics are cut at a stanza boundary
    (`truncated: true`). Use `get_track_profile` instead when mood or genre
    context also matters."""
    track = await ctx.spotify.resolve(args.track)
    lyrics = await ctx.lrclib.find_lyrics(track)
    if lyrics is None:
        return GetLyricsOutput(track=TrackOut.of(track), note="Lyrics not found on LRCLIB.")
    return GetLyricsOutput(
        track=TrackOut.of(track),
        lyrics=LyricsOut.of(lyrics, ctx.settings.lyrics_max_chars, with_stats=False),
    )


@tool
async def get_track_profile(ctx: AppContext, args: TrackInput) -> TrackProfile:
    """Everything about one song in a single call: lyrics (possibly
    truncated) with simple stats (repeated lines usually mark the chorus),
    audio features from ReccoBeats (valence = musical positivity, energy,
    danceability, acousticness, tempo, key), and Last.fm mood/genre tags.
    Any source can be missing; `missing` says which and why. Use this to
    explain what a song is about or why someone might like it."""
    track = await ctx.spotify.resolve(args.track)
    return await build_profile(ctx, track)
