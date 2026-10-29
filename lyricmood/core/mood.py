"""Mood playlists: score songs against mood targets and order them along an arc.

A target combines a mood-text vector (matched against lyrics) and optional
valence/energy (matched against ReccoBeats features). Turning words like
"sad" into numbers is the calling LLM's job; this module only does math.

Mood queries use raw cosine, unlike song-to-song similarity: centering the
lyrics but not the query made a sample "partying" query rank a sad song first
(docs/DESIGN.md §8.4).
"""

from dataclasses import dataclass

import numpy as np

from lyricmood.core.embeddings import normalize_rows


@dataclass
class Target:
    label: str
    query: np.ndarray | None = None  # normalized mood-text vector
    valence: float | None = None
    energy: float | None = None


def _lerp(u, v, frac: float):
    # Exactly on a stage, use that stage's value even if the neighbour has none.
    if frac == 0:
        return u
    if frac == 1:
        return v
    return None if u is None or v is None else u + (v - u) * frac


def interpolate(stages: list[Target], n: int) -> list[Target]:
    """`n` targets moving evenly from the first stage to the last.

    Queries are blended with normalized lerp; valence/energy linearly, but
    only where both neighbouring stages define them."""
    if len(stages) == 1 or n == 1:
        return [stages[0]] * n
    out = []
    for pos in range(n):
        x = pos * (len(stages) - 1) / (n - 1)
        i = min(int(x), len(stages) - 2)
        frac = x - i
        a, b = stages[i], stages[i + 1]

        query = None
        if a.query is not None and b.query is not None:
            query = normalize_rows(_lerp(a.query, b.query, frac)[None, :])[0]
        out.append(
            Target(
                label=(a if frac < 0.5 else b).label,
                query=query,
                valence=_lerp(a.valence, b.valence, frac),
                energy=_lerp(a.energy, b.energy, frac),
            )
        )
    return out


def score(
    targets: list[Target],
    lyric_vecs: list[np.ndarray | None],
    features: list[tuple[float | None, float | None] | None],
    feature_weight: float,
) -> np.ndarray:
    """[targets × tracks] scores in 0-1; -inf where a track has neither
    lyrics nor features.

    Lyrics similarity is min-max scaled per target across the tracks that
    have lyrics. Feature closeness is 1 - distance / sqrt(dims) over the
    dimensions the target defines."""
    n_t, n_s = len(targets), len(lyric_vecs)
    out = np.full((n_t, n_s), -np.inf)
    has_vec = [v is not None for v in lyric_vecs]
    matrix = np.stack([v for v in lyric_vecs if v is not None]) if any(has_vec) else None

    for ti, target in enumerate(targets):
        lyric = np.full(n_s, np.nan)
        if target.query is not None and matrix is not None:
            sims = matrix @ target.query
            spread = (sims.max() - sims.min()) or 1.0
            lyric[np.array(has_vec)] = (sims - sims.min()) / spread

        feat = np.full(n_s, np.nan)
        dims = [(k, getattr(target, k)) for k in ("valence", "energy")]
        dims = [(k, v) for k, v in dims if v is not None]
        if dims:
            for si, f in enumerate(features):
                if f is None:
                    continue
                values = {"valence": f[0], "energy": f[1]}
                diffs = [values[k] - v for k, v in dims if values[k] is not None]
                if len(diffs) == len(dims):
                    feat[si] = 1 - np.sqrt(np.sum(np.square(diffs))) / np.sqrt(len(dims))

        # A song missing one side counts as typical on that side (the median
        # of the songs that have it). Scoring it on the other side alone
        # would let songs with less information outrank complete ones.
        known = ~np.isnan(lyric) | ~np.isnan(feat)
        lyric_part = _fill_median(lyric) if target.query is not None else None
        feat_part = _fill_median(feat) if dims else None
        if lyric_part is not None and feat_part is not None:
            combined = (1 - feature_weight) * lyric_part + feature_weight * feat_part
        else:
            combined = lyric_part if lyric_part is not None else feat_part
        if combined is not None:
            out[ti, known] = combined[known]
    return out


def _fill_median(values: np.ndarray) -> np.ndarray:
    present = values[~np.isnan(values)]
    fill = float(np.median(present)) if present.size else 0.5
    return np.where(np.isnan(values), fill, values)


def assign(scores: np.ndarray, artists: list[str], max_per_artist: int) -> list[tuple[int, int]]:
    """Greedy, position by position: the best unused track for each target,
    with at most `max_per_artist` songs per artist. Positions with no
    eligible track left are dropped, so the result can be shorter.
    Returns (target index, track index) pairs in playlist order."""
    used: set[int] = set()
    per_artist: dict[str, int] = {}
    order: list[tuple[int, int]] = []
    for ti, row in enumerate(scores):
        for si in np.argsort(-row):
            si = int(si)
            if not np.isfinite(row[si]):
                break
            if si in used or per_artist.get(artists[si], 0) >= max_per_artist:
                continue
            used.add(si)
            per_artist[artists[si]] = per_artist.get(artists[si], 0) + 1
            order.append((ti, si))
            break
    return order
