"""Lyrics similarity: chunking, embedding with storage, and ranking.

Scores are relative to the songs being compared. Raw e5 cosines sit in a
narrow band (~0.8-0.9) and same-language pairs score high regardless of
topic, so vectors are centered on the compared set before comparing.
On a 12-line, 3-topic, 2-language check, centering raised
nearest-neighbour topic accuracy from 0.92 to 1.00 (docs/DESIGN.md §8.2).
"""

import asyncio

import numpy as np
import polars as pl

from lyricmood.core.embeddings import Embedder, normalize_rows
from lyricmood.core.lyrics import stanzas
from lyricmood.core.models import Track
from lyricmood.core.text import normalize
from lyricmood.core.vectors import StoredLyrics, VectorStore

MIN_CHUNK_CHARS = 60
MAX_CHUNKS = 16
# Centering needs enough songs for a meaningful average.
MIN_CENTER = 5


def chunk_lyrics(text: str) -> list[str]:
    """Stanzas, each repeated stanza (chorus) once, with short stanzas
    merged into the next so a chunk carries enough meaning to embed."""
    chunks: list[str] = []
    seen: set[str] = set()
    pending = ""
    for stanza in stanzas(text):
        key = normalize(stanza)
        if key in seen:
            continue
        seen.add(key)
        pending = f"{pending}\n{stanza}".strip()
        if len(pending) >= MIN_CHUNK_CHARS:
            chunks.append(pending)
            pending = ""
    if pending:
        if chunks and len(pending) < MIN_CHUNK_CHARS:
            chunks[-1] = f"{chunks[-1]}\n{pending}"
        else:
            chunks.append(pending)
    return chunks[:MAX_CHUNKS]


async def embed_lyrics(
    embedder: Embedder, store: VectorStore, items: list[tuple[Track, str]]
) -> dict[str, StoredLyrics]:
    """Vectors for each (track, lyrics) pair: from the store when present,
    otherwise embedded now (in a worker thread) and stored."""
    ids = [t.spotify_id for t, _ in items]
    found = store.get(embedder.model_id, ids)
    todo = [(t, chunk_lyrics(text)) for t, text in items if t.spotify_id not in found]
    todo = [(t, chunks) for t, chunks in todo if chunks]
    if todo:
        flat = [c for _, chunks in todo for c in chunks]
        vectors = await asyncio.to_thread(embedder.embed, flat)
        offset = 0
        for track, chunks in todo:
            chunk_vecs = vectors[offset : offset + len(chunks)]
            offset += len(chunks)
            vector = normalize_rows(chunk_vecs.mean(0, keepdims=True))[0]
            store.put(embedder.model_id, track.spotify_id, track.isrc, vector, chunks, chunk_vecs)
            found[track.spotify_id] = StoredLyrics(vector, chunks, chunk_vecs)
    return found


def center(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) < MIN_CENTER:
        return matrix
    return normalize_rows(matrix - matrix.mean(0, keepdims=True))


def best_chunk_pair(a: StoredLyrics, b: StoredLyrics) -> tuple[str, str]:
    """The stanza of `a` and the stanza of `b` that are most alike."""
    sims = a.chunk_vectors @ b.chunk_vectors.T
    i, j = np.unravel_index(int(sims.argmax()), sims.shape)
    return a.chunks[i], b.chunks[j]


def rank(
    seed: StoredLyrics,
    candidates: dict[str, StoredLyrics],
    seed_mood: tuple[float, float] | None,
    moods: dict[str, tuple[float, float] | None],
    mood_weight: float,
) -> pl.DataFrame:
    """Rank candidates by lyrics similarity to the seed, optionally blended
    with closeness in (valence, energy).

    Columns: spotify_id, lyrics (0-1, relative to this pool),
    mood (0-1, or null when the candidate has no features), score.
    """
    ids = list(candidates)
    if not ids:
        return pl.DataFrame(
            schema={
                "spotify_id": pl.String,
                "lyrics": pl.Float64,
                "mood": pl.Float64,
                "score": pl.Float64,
            }
        )
    matrix = center(np.stack([seed.vector] + [candidates[i].vector for i in ids]))
    cosine = matrix[1:] @ matrix[0]

    def mood_sim(sid: str) -> float | None:
        mood = moods.get(sid)
        if seed_mood is None or mood is None:
            return None
        distance = np.hypot(mood[0] - seed_mood[0], mood[1] - seed_mood[1])
        return float(1 - distance / np.sqrt(2))

    df = pl.DataFrame(
        {"spotify_id": ids, "cosine": cosine.astype(float), "mood": [mood_sim(i) for i in ids]},
        schema={"spotify_id": pl.String, "cosine": pl.Float64, "mood": pl.Float64},
    )
    low, high = df["cosine"].min(), df["cosine"].max()
    spread = (high - low) or 1.0
    return (
        df.with_columns(lyrics=(pl.col("cosine") - low) / spread)
        .with_columns(
            # A candidate without features counts as typical (the pool's
            # median mood), so missing data neither helps nor hurts it.
            score=pl.when(pl.col("mood").is_not_null().any())
            .then(
                (1 - mood_weight) * pl.col("lyrics")
                + mood_weight * pl.col("mood").fill_null(pl.col("mood").median())
            )
            .otherwise(pl.col("lyrics"))
        )
        .sort("score", descending=True)
        .select("spotify_id", "lyrics", "mood", "score")
    )
