"""LanceDB store for lyrics embeddings, keyed by (model, Spotify ID).

Two tables:
- `lyrics`: one vector per track (mean of its chunk vectors).
- `lyrics_chunks`: one vector per lyrics chunk (stanza), used to show
  which parts of two songs are most alike.

Only tracks that come up as candidates are ever embedded; nothing is
pre-computed for a catalog.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lyricmood.core.embeddings import INSTALL_HINT
from lyricmood.core.errors import ConfigError


@dataclass
class StoredLyrics:
    vector: np.ndarray
    chunks: list[str]
    chunk_vectors: np.ndarray


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class VectorStore:
    def __init__(self, path: Path, dim: int):
        try:
            import lancedb
            import pyarrow as pa
        except ImportError:
            raise ConfigError(INSTALL_HINT) from None
        self._db = lancedb.connect(str(path))
        vector = pa.list_(pa.float32(), dim)
        self._tracks = self._db.create_table(
            "lyrics",
            schema=pa.schema(
                [
                    ("key", pa.string()),
                    ("model", pa.string()),
                    ("spotify_id", pa.string()),
                    ("isrc", pa.string()),
                    ("vector", vector),
                    ("updated_at", pa.float64()),
                ]
            ),
            exist_ok=True,
        )
        self._chunks = self._db.create_table(
            "lyrics_chunks",
            schema=pa.schema(
                [
                    ("key", pa.string()),
                    ("model", pa.string()),
                    ("spotify_id", pa.string()),
                    ("chunk_ix", pa.int32()),
                    ("text", pa.string()),
                    ("vector", vector),
                ]
            ),
            exist_ok=True,
        )

    def put(
        self,
        model: str,
        spotify_id: str,
        isrc: str | None,
        vector: np.ndarray,
        chunks: list[str],
        chunk_vectors: np.ndarray,
    ) -> None:
        key = f"{model}:{spotify_id}"
        self._tracks.merge_insert(
            "key"
        ).when_matched_update_all().when_not_matched_insert_all().execute(
            [
                {
                    "key": key,
                    "model": model,
                    "spotify_id": spotify_id,
                    "isrc": isrc,
                    "vector": vector.astype(np.float32),
                    "updated_at": time.time(),
                }
            ]
        )
        self._chunks.delete(f"model = {_quote(model)} AND spotify_id = {_quote(spotify_id)}")
        self._chunks.add(
            [
                {
                    "key": f"{key}:{i}",
                    "model": model,
                    "spotify_id": spotify_id,
                    "chunk_ix": i,
                    "text": text,
                    "vector": vec.astype(np.float32),
                }
                for i, (text, vec) in enumerate(zip(chunks, chunk_vectors, strict=True))
            ]
        )

    def get(self, model: str, spotify_ids: list[str]) -> dict[str, StoredLyrics]:
        if not spotify_ids:
            return {}
        ids = ", ".join(_quote(i) for i in dict.fromkeys(spotify_ids))
        where = f"model = {_quote(model)} AND spotify_id IN ({ids})"
        tracks = self._tracks.search().where(where).limit(None).to_list()
        chunks = self._chunks.search().where(where).limit(None).to_list()
        by_id: dict[str, list[dict]] = {}
        for row in chunks:
            by_id.setdefault(row["spotify_id"], []).append(row)
        result = {}
        for row in tracks:
            parts = sorted(by_id.get(row["spotify_id"], []), key=lambda r: r["chunk_ix"])
            result[row["spotify_id"]] = StoredLyrics(
                vector=np.asarray(row["vector"], dtype=np.float32),
                chunks=[p["text"] for p in parts],
                chunk_vectors=np.asarray([p["vector"] for p in parts], dtype=np.float32),
            )
        return result

    def ids(self, model: str) -> list[str]:
        rows = (
            self._tracks.search()
            .where(f"model = {_quote(model)}")
            .select(["spotify_id"])
            .limit(None)
            .to_list()
        )
        return [r["spotify_id"] for r in rows]

    def count(self) -> int:
        return self._tracks.count_rows()
