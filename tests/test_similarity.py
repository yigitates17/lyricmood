import numpy as np
import pytest

from lyricmood.core.lyrics import excerpt
from lyricmood.core.models import Track
from lyricmood.core.similarity import (
    MIN_CHUNK_CHARS,
    best_chunk_pair,
    center,
    chunk_lyrics,
    embed_lyrics,
    rank,
)
from lyricmood.core.vectors import StoredLyrics
from tests.conftest import FakeEmbedder

LONG = "x" * MIN_CHUNK_CHARS


def test_chunks_dedupe_repeated_stanzas_and_merge_short_ones():
    text = f"short one\n\n{LONG} verse\n\nchorus {LONG}\n\nshort two\n\nchorus {LONG}"
    chunks = chunk_lyrics(text)
    assert chunks == [f"short one\n{LONG} verse", f"chorus {LONG}\nshort two"]


def test_chunks_of_tiny_lyrics():
    assert chunk_lyrics("la la la") == ["la la la"]
    assert chunk_lyrics("") == []


def test_center_skips_small_sets():
    m = np.eye(3, dtype=np.float32)
    assert center(m) is m
    big = np.random.default_rng(0).random((6, 4)).astype(np.float32)
    centered = center(big)
    assert np.allclose(np.linalg.norm(centered, axis=1), 1)


def stored(vec, chunks=("a",), chunk_vecs=None) -> StoredLyrics:
    vec = np.asarray(vec, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    cv = np.asarray(chunk_vecs if chunk_vecs is not None else [vec], dtype=np.float32)
    return StoredLyrics(vec, list(chunks), cv)


def test_rank_by_lyrics_then_mood_blend():
    seed = stored([1, 0, 0])
    pool = {"close": stored([0.9, 0.1, 0]), "far": stored([0, 1, 0]), "mid": stored([0.6, 0.6, 0])}

    lyrics_only = rank(seed, pool, None, {}, 0.0)
    assert lyrics_only["spotify_id"].to_list() == ["close", "mid", "far"]
    assert lyrics_only["score"].to_list()[0] == 1.0  # relative to the pool
    assert lyrics_only["mood"].is_null().all()

    # "far" matches the seed's mood exactly; "close" is the opposite mood.
    moods = {"close": (0.0, 0.0), "far": (0.9, 0.9), "mid": (0.5, 0.5)}
    blended = rank(seed, pool, (0.9, 0.9), moods, 0.9)
    assert blended["spotify_id"].to_list()[0] == "far"


def test_rank_counts_missing_features_as_typical():
    seed = stored([1, 0])
    pool = {"a": stored([1, 0.1]), "b": stored([0.1, 1]), "c": stored([0.7, 0.7])}
    moods = {"b": (0.5, 0.5), "c": (0.0, 0.0)}  # "a" has no features
    out = rank(seed, pool, (0.5, 0.5), moods, 0.5)
    rows = {r["spotify_id"]: r for r in out.iter_rows(named=True)}
    assert rows["a"]["mood"] is None
    median_mood = (rows["b"]["mood"] + rows["c"]["mood"]) / 2
    assert rows["a"]["score"] == pytest.approx(0.5 * rows["a"]["lyrics"] + 0.5 * median_mood)
    # Not scored on lyrics alone, which would have given it a perfect 1.0.
    assert rows["a"]["score"] < 1.0


def test_rank_empty_pool():
    assert rank(stored([1, 0]), {}, None, {}, 0.0).is_empty()


def test_best_chunk_pair():
    a = stored([1, 0], chunks=["rain", "fire"], chunk_vecs=[[1, 0], [0, 1]])
    b = stored([1, 0], chunks=["sun", "flame"], chunk_vecs=[[0.7, 0.7], [0.1, 1]])
    assert best_chunk_pair(a, b) == ("fire", "flame")


def test_excerpt_takes_opening_and_chorus():
    text = "verse one\nline\n\nchorus line\nchorus line\n\nverse two\n\nchorus line\nchorus line"
    assert excerpt(text, 500) == "verse one\nline\n\nchorus line\nchorus line"
    assert excerpt(text, 15) == "verse one\nline"


async def test_embed_lyrics_stores_and_reuses(vectors):
    class CountingEmbedder(FakeEmbedder):
        calls = 0

        def embed(self, texts):
            self.calls += 1
            return super().embed(texts)

    embedder = CountingEmbedder()
    track = Track(spotify_id="a" * 22, title="T", artists=["A"], duration_ms=1, isrc="ISRC")
    text = f"rain on the ocean {LONG}\n\nwaves and rain again {LONG}"

    first = await embed_lyrics(embedder, vectors, [(track, text)])
    again = await embed_lyrics(embedder, vectors, [(track, text)])
    assert embedder.calls == 1
    assert len(first[track.spotify_id].chunks) == 2
    assert np.allclose(first[track.spotify_id].vector, again[track.spotify_id].vector)
    assert vectors.ids(embedder.model_id) == [track.spotify_id]
    assert vectors.ids("another-model") == []


def test_vector_store_replaces_chunks(vectors):
    v = np.ones(FakeEmbedder.dim, dtype=np.float32)
    vectors.put("m", "id1", None, v, ["a", "b"], np.stack([v, v]))
    vectors.put("m", "id1", None, v, ["c"], np.stack([v]))
    assert vectors.get("m", ["id1"])["id1"].chunks == ["c"]
    assert vectors.count() == 1


@pytest.mark.parametrize("quote", ["it's", "o'clock"])
def test_vector_store_quotes_ids(vectors, quote):
    v = np.ones(FakeEmbedder.dim, dtype=np.float32)
    vectors.put("m", quote, None, v, ["a"], np.stack([v]))
    assert list(vectors.get("m", [quote])) == [quote]
