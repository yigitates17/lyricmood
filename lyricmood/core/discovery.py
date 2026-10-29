"""Candidate songs for "similar to X": Last.fm similar tracks, Last.fm tag
top tracks, similar artists' top tracks, and ReccoBeats recommendations.

Every candidate is resolved to a Spotify track before embedding, so it can
get audio features for the mood blend (docs/DESIGN.md §8.3).
"""

import asyncio
import itertools
from dataclasses import dataclass, field

from lyricmood.core.errors import LyricmoodError
from lyricmood.core.lastfm import LastfmClient
from lyricmood.core.models import Track
from lyricmood.core.reccobeats import ReccoBeatsClient
from lyricmood.core.spotify.client import SpotifyClient
from lyricmood.core.text import clean_title, normalize


@dataclass
class Candidate:
    track: Track
    sources: list[str] = field(default_factory=list)


def _name_key(artist: str, title: str) -> str:
    return f"{normalize(artist)}|{normalize(clean_title(title))}"


async def gather_candidates(
    seed: Track,
    *,
    spotify: SpotifyClient,
    lastfm: LastfmClient | None,
    reccobeats: ReccoBeatsClient,
    seed_tags: list[str],
    pool_size: int,
) -> tuple[list[Candidate], dict[str, str]]:
    """Up to `pool_size` candidates (seed excluded), plus sources that failed."""
    errors: dict[str, str] = {}

    async def attempt(name: str, coro):
        try:
            return await coro
        except LyricmoodError as exc:
            errors[name] = str(exc)
            return []

    artist, title = seed.primary_artist, seed.title
    jobs: dict[str, object] = {
        "reccobeats": attempt("reccobeats", reccobeats.recommendations([seed.spotify_id], size=15)),
    }
    if lastfm is not None:
        jobs["lastfm_similar"] = attempt("lastfm_similar", lastfm.similar_tracks(artist, title, 25))
        jobs["similar_artists"] = attempt("similar_artists", lastfm.similar_artists(artist, 5))
        for tag in seed_tags[:2]:
            jobs[f"tag:{tag}"] = attempt(f"tag:{tag}", lastfm.tag_top_tracks(tag, 10))
    results = dict(zip(jobs, await asyncio.gather(*jobs.values()), strict=True))

    if lastfm is not None and results.get("similar_artists"):
        top_lists = await asyncio.gather(
            *(
                attempt("similar_artists", lastfm.artist_top_tracks(a, 2))
                for a, _ in results["similar_artists"]
            )
        )
        results["similar_artists"] = [t for tracks in top_lists for t in tracks]

    # (source, artist, title, spotify_id or None), one list per source.
    per_source: list[list[tuple[str, str, str, str | None]]] = []
    for source, items in results.items():
        if source == "reccobeats":
            per_source.append(
                [(source, r.artists[0] if r.artists else "", r.title, r.spotify_id) for r in items]
            )
        else:
            per_source.append([(source, t.artist, t.title, None) for t in items])

    # Round-robin across sources so one source can't fill the pool alone.
    seed_key = _name_key(artist, title)
    picked: dict[str, tuple[list[str], str, str, str | None]] = {}
    for item in itertools.chain.from_iterable(itertools.zip_longest(*per_source)):
        if item is None:
            continue
        source, c_artist, c_title, sid = item
        key = _name_key(c_artist, c_title)
        if key == seed_key or sid == seed.spotify_id:
            continue
        if key in picked:
            picked[key][0].append(source)
        elif len(picked) < pool_size:
            picked[key] = ([source], c_artist, c_title, sid)

    async def resolve(entry) -> Candidate | None:
        sources, c_artist, c_title, sid = entry
        try:
            track = await (
                spotify.get_track(sid) if sid else spotify.find_by_name(c_artist, c_title)
            )
        except LyricmoodError:
            return None
        return Candidate(track, sources) if track else None

    resolved = await asyncio.gather(*(resolve(e) for e in picked.values()))
    candidates: dict[str, Candidate] = {}
    for cand in resolved:
        if cand is None or cand.track.spotify_id == seed.spotify_id:
            continue
        if existing := candidates.get(cand.track.spotify_id):
            existing.sources.extend(s for s in cand.sources if s not in existing.sources)
        else:
            candidates[cand.track.spotify_id] = cand
    return list(candidates.values()), errors
