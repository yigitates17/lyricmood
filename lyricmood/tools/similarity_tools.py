import asyncio

from pydantic import BaseModel, Field

from lyricmood.context import AppContext
from lyricmood.core.discovery import Candidate, gather_candidates
from lyricmood.core.errors import LyricmoodError, NotFound
from lyricmood.core.models import Track
from lyricmood.core.similarity import best_chunk_pair, embed_lyrics, rank
from lyricmood.registry import tool
from lyricmood.tools.lyrics_tools import track_tags
from lyricmood.tools.schemas import TRACK_ARG, TrackOut


class FindSimilarInput(BaseModel):
    track: str = Field(min_length=1, description=TRACK_ARG)
    k: int = Field(default=10, ge=1, le=25, description="How many results (1-25).")
    expand_pool: bool = Field(
        default=True,
        description="True: gather fresh candidates from Last.fm and ReccoBeats (slow the "
        "first time, cached after). False: rank only songs analyzed in earlier searches "
        "(instant, no network).",
    )
    mood_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0 = lyrics only. Raise (e.g. 0.3) to also prefer songs with similar "
        "musical mood (valence/energy). Use when the user cares how it sounds, not just "
        "what it says.",
    )


class MatchingLines(BaseModel):
    """The most alike passages: one from the seed, one from this song."""

    seed: str
    this: str


class SimilarTrack(BaseModel):
    track: TrackOut
    score: float
    lyrics: float
    mood: float | None = None
    found_via: list[str] = []
    matching_lines: MatchingLines | None = None


class FindSimilarOutput(BaseModel):
    seed: TrackOut
    results: list[SimilarTrack]
    candidates: int
    candidates_with_lyrics: int
    missing: dict[str, str] | None = None


def _snippet(text: str, max_chars: int = 120) -> str:
    lines = " / ".join(line.strip() for line in text.splitlines()[:2] if line.strip())
    return lines if len(lines) <= max_chars else lines[: max_chars - 1].rstrip() + "…"


@tool
async def find_similar_by_lyrics(ctx: AppContext, args: FindSimilarInput) -> FindSimilarOutput:
    """Find songs whose lyrics are about similar things as the given song
    (works across languages). Candidates come from Last.fm (similar tracks,
    tags, similar artists) and ReccoBeats, and are ranked by lyrics meaning
    with a local multilingual model. Scores are 0-1 relative to this search,
    not absolute. `matching_lines` shows the passages that matched best;
    use them to explain the connection. The first search for a song can
    take ~30-60 s; repeats are fast."""
    missing: dict[str, str] = {}
    seed = await ctx.spotify.resolve(args.track)
    seed_lyrics = await ctx.lrclib.find_lyrics(seed)
    if seed_lyrics is None or not seed_lyrics.plain:
        raise NotFound(
            f"No lyrics found for {seed.title} by {seed.primary_artist}, so lyrical "
            "similarity isn't possible for this song."
        )

    if args.expand_pool:
        seed_tags: list[str] = []
        if ctx.has_lastfm:
            try:
                seed_tags = [t.name for t in (await track_tags(ctx, seed))[0]]
            except LyricmoodError as exc:
                missing["tags"] = str(exc)
        else:
            missing["lastfm"] = "LASTFM_API_KEY is not set; using ReccoBeats candidates only"
        candidates, errors = await gather_candidates(
            seed,
            spotify=ctx.spotify,
            lastfm=ctx.lastfm if ctx.has_lastfm else None,
            reccobeats=ctx.reccobeats,
            seed_tags=seed_tags,
            pool_size=ctx.settings.similar_pool_size,
        )
        missing.update(errors)
        lyrics = await asyncio.gather(
            *(ctx.lrclib.find_lyrics(c.track) for c in candidates), return_exceptions=True
        )
        items: list[tuple[Track, str]] = [(seed, seed_lyrics.plain)]
        for cand, result in zip(candidates, lyrics, strict=True):
            if isinstance(result, BaseException) and not isinstance(result, LyricmoodError):
                raise result
            if not isinstance(result, BaseException) and result and result.plain:
                items.append((cand.track, result.plain))
        stored = await embed_lyrics(ctx.embedder, ctx.vectors, items)
    else:
        model = ctx.embedder.model_id
        ids = [i for i in ctx.vectors.ids(model) if i != seed.spotify_id]
        candidates = [
            Candidate(t, ["earlier_searches"]) for i in ids if (t := ctx.cache.get_track(i))
        ]
        stored = await embed_lyrics(ctx.embedder, ctx.vectors, [(seed, seed_lyrics.plain)])
        stored.update(ctx.vectors.get(model, [c.track.spotify_id for c in candidates]))

    seed_vec = stored.pop(seed.spotify_id, None)
    if seed_vec is None:
        raise NotFound(f"The lyrics of {seed.title} were too short to analyze.")
    by_id = {c.track.spotify_id: c for c in candidates if c.track.spotify_id in stored}
    pool = {sid: stored[sid] for sid in by_id}

    seed_mood, moods = None, {}
    if args.mood_weight > 0 and pool:
        try:
            features = await ctx.reccobeats.audio_features([seed.spotify_id, *pool])
            moods = {
                sid: (f.valence, f.energy)
                for sid, f in features.items()
                if f and f.valence is not None and f.energy is not None
            }
            seed_mood = moods.pop(seed.spotify_id, None)
            if seed_mood is None:
                missing["mood"] = "no audio features for the seed song; ranked on lyrics only"
        except LyricmoodError as exc:
            missing["mood"] = str(exc)

    ranked = rank(seed_vec, pool, seed_mood, moods, args.mood_weight).head(args.k)
    results = []
    for row in ranked.iter_rows(named=True):
        sid = row["spotify_id"]
        seed_part, cand_part = best_chunk_pair(seed_vec, pool[sid])
        results.append(
            SimilarTrack(
                track=TrackOut.of(by_id[sid].track),
                score=round(row["score"], 2),
                lyrics=round(row["lyrics"], 2),
                mood=round(row["mood"], 2) if row["mood"] is not None else None,
                found_via=by_id[sid].sources,
                matching_lines=MatchingLines(seed=_snippet(seed_part), this=_snippet(cand_part)),
            )
        )
    return FindSimilarOutput(
        seed=TrackOut.of(seed),
        results=results,
        candidates=len(candidates),
        candidates_with_lyrics=len(pool),
        missing=missing or None,
    )
