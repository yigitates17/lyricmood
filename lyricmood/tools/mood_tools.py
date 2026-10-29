import asyncio
import itertools

import numpy as np
from pydantic import BaseModel, Field

from lyricmood.context import AppContext
from lyricmood.core.embeddings import normalize_rows
from lyricmood.core.errors import LyricmoodError, NotFound
from lyricmood.core.models import Track
from lyricmood.core.mood import Target, assign, interpolate, score
from lyricmood.core.similarity import embed_lyrics
from lyricmood.registry import tool
from lyricmood.tools.artist_tools import artist_top
from lyricmood.tools.schemas import TrackOut

MOOD_TEXT = (
    "What the lyrics should be about or feel like, e.g. 'missing someone, lonely "
    "nights'. Several lines are averaged: give the mood in each language of the "
    "music (e.g. English and Finnish), because the model favours lyrics in the "
    "query's language."
)
VALENCE = "Target musical positivity 0-1 (sad ≈ 0.15, neutral ≈ 0.5, happy ≈ 0.85)."
ENERGY = "Target intensity 0-1 (calm ≈ 0.2, mid ≈ 0.5, intense ≈ 0.85)."


class ArcStage(BaseModel):
    mood_text: str = Field(min_length=1, description=MOOD_TEXT)
    valence: float | None = Field(default=None, ge=0, le=1, description=VALENCE)
    energy: float | None = Field(default=None, ge=0, le=1, description=ENERGY)


class MoodPlaylistInput(BaseModel):
    mood_text: str = Field(min_length=1, description=MOOD_TEXT)
    valence: float | None = Field(default=None, ge=0, le=1, description=VALENCE)
    energy: float | None = Field(default=None, ge=0, le=1, description=ENERGY)
    seed_artists: list[str] = Field(
        default=[],
        max_length=5,
        description="Draw songs from these artists and similar ones. Empty: use the "
        "user's own top and liked songs.",
    )
    size: int = Field(default=20, ge=5, le=50, description="Number of songs (5-50).")
    arc: list[ArcStage] | None = Field(
        default=None,
        min_length=2,
        max_length=5,
        description="Optional journey, e.g. sad -> hopeful: 2-5 stages in order. Songs move "
        "gradually from the first stage to the last. Overrides mood_text/valence/energy "
        "for ordering.",
    )
    max_per_artist: int = Field(default=3, ge=1, le=10, description="Variety cap per artist.")
    feature_weight: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="How much valence/energy count against lyrics (0 = lyrics only).",
    )


class MoodTrack(BaseModel):
    track: TrackOut
    stage: str | None = None
    valence: float | None = None
    energy: float | None = None
    score: float


class MoodPlaylistOutput(BaseModel):
    tracks: list[MoodTrack]
    candidates: int
    with_lyrics: int
    with_features: int
    missing: dict[str, str] | None = None


async def _pool(
    ctx: AppContext, seeds: list[str], cap: int, missing: dict[str, str]
) -> list[Track]:
    if seeds:
        lists: list[list[Track]] = []
        tops = await asyncio.gather(
            *(artist_top(ctx, a, 10) for a in seeds), return_exceptions=True
        )
        for name, top in zip(seeds, tops, strict=True):
            if isinstance(top, LyricmoodError):
                missing[f"artist:{name}"] = str(top)
            elif isinstance(top, BaseException):
                raise top
            else:
                lists.append(top.tracks)
        if ctx.has_lastfm:
            similar = await asyncio.gather(
                *(ctx.lastfm.similar_artists(a, 3) for a in seeds), return_exceptions=True
            )
            names = [n for r in similar if isinstance(r, list) for n, _ in r]
            names = [n for n in dict.fromkeys(names) if n.lower() not in {s.lower() for s in seeds}]
            sim_tops = await asyncio.gather(
                *(artist_top(ctx, n, 5) for n in names), return_exceptions=True
            )
            lists += [t.tracks for t in sim_tops if not isinstance(t, BaseException)]
    else:
        top = await ctx.spotify.top_tracks("medium_term", 50)
        saved, _ = await ctx.spotify.saved_tracks(50)
        lists = [top, saved]
    # Interleave so seeds and similar artists share the pool fairly.
    pool: dict[str, Track] = {}
    for track in itertools.chain.from_iterable(itertools.zip_longest(*lists)):
        if track is not None and len(pool) < cap:
            pool.setdefault(track.spotify_id, track)
    return list(pool.values())


def _query(ctx: AppContext, text: str) -> np.ndarray:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    vecs = ctx.embedder.embed(lines, kind="query")
    return normalize_rows(vecs.mean(0, keepdims=True))[0]


@tool
async def build_mood_playlist(ctx: AppContext, args: MoodPlaylistInput) -> MoodPlaylistOutput:
    """Pick and order songs for a mood, judging both what the lyrics are
    about (mood_text) and how the music feels (valence/energy targets you
    choose from the user's request). With `arc`, songs move gradually
    through 2-5 moods (e.g. heartbroken -> hopeful). Songs come from the
    seed artists and similar artists, or from the user's top and liked songs.
    This only proposes a list: show it to the user, then call
    create_spotify_playlist with the ids if they want it saved. The first run
    for new artists can take a minute or two."""
    missing: dict[str, str] = {}
    seeds = list(dict.fromkeys(a.strip() for a in args.seed_artists if a.strip()))
    pool = await _pool(ctx, seeds, ctx.settings.mood_pool_size, missing)
    if not pool:
        raise NotFound("No candidate songs found for these artists.")

    lyrics = await asyncio.gather(
        *(ctx.lrclib.find_lyrics(t) for t in pool), return_exceptions=True
    )
    texts = {}
    for track, result in zip(pool, lyrics, strict=True):
        if isinstance(result, BaseException) and not isinstance(result, LyricmoodError):
            raise result
        if not isinstance(result, BaseException) and result and result.plain:
            texts[track.spotify_id] = result.plain

    try:
        feats = await ctx.reccobeats.audio_features([t.spotify_id for t in pool])
    except LyricmoodError as exc:
        missing["features"] = str(exc)
        feats = {}

    stored = await embed_lyrics(
        ctx.embedder, ctx.vectors, [(t, texts[t.spotify_id]) for t in pool if t.spotify_id in texts]
    )

    stages = args.arc or [
        ArcStage(mood_text=args.mood_text, valence=args.valence, energy=args.energy)
    ]
    stage_targets = [
        Target(
            label=stage.mood_text.splitlines()[0][:60],
            query=await asyncio.to_thread(_query, ctx, stage.mood_text),
            valence=stage.valence,
            energy=stage.energy,
        )
        for stage in stages
    ]
    targets = interpolate(stage_targets, args.size)

    features = [(f.valence, f.energy) if (f := feats.get(t.spotify_id)) else None for t in pool]
    scores = score(
        targets,
        [stored[t.spotify_id].vector if t.spotify_id in stored else None for t in pool],
        features,
        args.feature_weight,
    )
    picks = assign(scores, [t.primary_artist.lower() for t in pool], args.max_per_artist)

    tracks = []
    for ti, si in picks:
        track, f = pool[si], features[si]
        tracks.append(
            MoodTrack(
                track=TrackOut.of(track),
                stage=targets[ti].label if args.arc else None,
                valence=round(f[0], 2) if f and f[0] is not None else None,
                energy=round(f[1], 2) if f and f[1] is not None else None,
                score=round(float(scores[ti, si]), 2),
            )
        )
    if len(tracks) < args.size:
        missing["size"] = (
            f"only {len(tracks)} songs could be judged; add seed artists or raise max_per_artist"
        )
    return MoodPlaylistOutput(
        tracks=tracks,
        candidates=len(pool),
        with_lyrics=len(stored),
        with_features=sum(f is not None for f in features),
        missing=missing or None,
    )
