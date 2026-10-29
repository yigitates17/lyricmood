import time

from lyricmood.core.cache import MISS
from lyricmood.core.models import Track


def make_track(spotify_id="a" * 22, isrc="ISRC1") -> Track:
    return Track(spotify_id=spotify_id, title="T", artists=["A"], duration_ms=1000, isrc=isrc)


def test_kv_roundtrip_and_miss(cache):
    assert cache.get("ns", "k") is MISS
    cache.set("ns", "k", {"x": [1, 2]})
    assert cache.get("ns", "k") == {"x": [1, 2]}
    assert cache.get("other", "k") is MISS


def test_cached_none_is_not_a_miss(cache):
    cache.set("reccobeats", "id", None)
    assert cache.get("reccobeats", "id") is None


def test_expired_entries_are_misses_and_purged(cache, monkeypatch):
    cache.set("ns", "k", 1, ttl=10)
    real = time.time()
    monkeypatch.setattr(time, "time", lambda: real + 11)
    assert cache.get("ns", "k") is MISS
    assert cache.purge_expired() == 1


async def test_get_or_fetch_fetches_once(cache):
    calls = 0

    async def fetch():
        nonlocal calls
        calls += 1
        return None

    for _ in range(3):
        assert await cache.get_or_fetch("ns", "k", fetch) is None
    assert calls == 1


def test_tracks_by_id_and_isrc(cache):
    track = make_track()
    cache.put_tracks([track])
    assert cache.get_track(track.spotify_id) == track
    assert cache.get_track_by_isrc("ISRC1") == track
    assert cache.get_track("b" * 22) is None
    assert cache.stats()["tracks"] == 1


def test_cache_persists_across_connections(tmp_path):
    from lyricmood.core.cache import Cache

    first = Cache(tmp_path / "c.sqlite3")
    first.put_tracks([make_track()])
    first.close()
    second = Cache(tmp_path / "c.sqlite3")
    assert second.get_track("a" * 22) is not None
    second.close()
