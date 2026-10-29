"""SQLite cache: canonical tracks keyed by Spotify ID (and ISRC), plus a
namespaced key/value store for raw API responses.

Synchronous sqlite3 is deliberate: local queries take microseconds, and WAL
mode lets the MCP server and the terminal chat share one file safely.
"""

import json
import sqlite3
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from lyricmood.core.models import Track

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    spotify_id  TEXT PRIMARY KEY,
    isrc        TEXT,
    data        TEXT NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS tracks_isrc ON tracks(isrc);

CREATE TABLE IF NOT EXISTS kv (
    namespace   TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    expires_at  REAL,
    PRIMARY KEY (namespace, key)
);
"""

DAY = 86_400.0


class _Miss:
    def __repr__(self) -> str:
        return "MISS"


MISS: Any = _Miss()
"""Returned by `get` on a miss, so a cached `None` (a known-empty answer,
e.g. ReccoBeats having no features for a track) stays distinguishable."""


class Cache:
    def __init__(self, path: Path | str):
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- key/value -----------------------------------------------------------

    def get(self, namespace: str, key: str) -> Any:
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expires_at FROM kv WHERE namespace=? AND key=?", (namespace, key)
            ).fetchone()
        if row is None or (row[1] is not None and row[1] < time.time()):
            return MISS
        return json.loads(row[0])

    def set(self, namespace: str, key: str, value: Any, ttl: float | None = None) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv(namespace, key, value, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (namespace, key, json.dumps(value), now, now + ttl if ttl else None),
            )

    async def get_or_fetch(
        self,
        namespace: str,
        key: str,
        fetch: Callable[[], Awaitable[Any]],
        ttl: float | None = None,
    ) -> Any:
        value = self.get(namespace, key)
        if value is MISS:
            value = await fetch()
            self.set(namespace, key, value, ttl)
        return value

    # -- tracks --------------------------------------------------------------

    def put_tracks(self, tracks: list[Track]) -> None:
        now = time.time()
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO tracks(spotify_id, isrc, data, updated_at) "
                "VALUES (?, ?, ?, ?)",
                [(t.spotify_id, t.isrc, t.model_dump_json(), now) for t in tracks],
            )

    def get_track(self, spotify_id: str) -> Track | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM tracks WHERE spotify_id=?", (spotify_id,)
            ).fetchone()
        return Track.model_validate_json(row[0]) if row else None

    def get_track_by_isrc(self, isrc: str) -> Track | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM tracks WHERE isrc=? ORDER BY updated_at DESC LIMIT 1", (isrc,)
            ).fetchone()
        return Track.model_validate_json(row[0]) if row else None

    # -- maintenance ---------------------------------------------------------

    def stats(self) -> dict[str, int]:
        with self._lock:
            tracks = self._conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
            rows = self._conn.execute(
                "SELECT namespace, COUNT(*) FROM kv GROUP BY namespace ORDER BY namespace"
            ).fetchall()
        return {"tracks": tracks, **{f"kv:{ns}": n for ns, n in rows}}

    def purge_expired(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM kv WHERE expires_at IS NOT NULL AND expires_at < ?", (time.time(),)
            )
        return cur.rowcount
