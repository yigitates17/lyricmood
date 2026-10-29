import numpy as np
import pytest

from lyricmood.core.mood import Target, assign, interpolate, score


def unit(*xs):
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_interpolate_moves_from_first_to_last_stage():
    sad = Target("sad", unit(1, 0), valence=0.1, energy=0.3)
    happy = Target("happy", unit(0, 1), valence=0.9, energy=None)
    steps = interpolate([sad, happy], 5)
    assert [t.label for t in steps] == ["sad", "sad", "happy", "happy", "happy"]
    assert [round(t.valence, 2) for t in steps] == [0.1, 0.3, 0.5, 0.7, 0.9]
    # Energy is only defined on one side, so it isn't invented in between.
    assert steps[0].energy == 0.3 and steps[2].energy is None
    assert np.allclose([np.linalg.norm(t.query) for t in steps], 1)
    assert steps[0].query @ unit(1, 0) > steps[-1].query @ unit(1, 0)


def test_interpolate_single_stage_or_position():
    t = Target("x", unit(1, 0))
    assert interpolate([t], 3) == [t, t, t]
    assert interpolate([t, Target("y", unit(0, 1))], 1) == [t]


def test_score_blends_and_counts_missing_side_as_typical():
    target = Target("t", unit(1, 0), valence=0.2, energy=0.3)
    vecs = [unit(1, 0), unit(0, 1), None, unit(1, 1), None]
    feats = [(0.2, 0.3), (0.2, 0.3), (0.2, 0.3), None, None]
    s = score([target], vecs, feats, feature_weight=0.5)[0]
    assert s[0] == pytest.approx(1.0)  # best lyrics, perfect features
    assert s[0] > s[3] > s[1]  # missing features don't beat complete matches
    assert s[2] < s[0]  # no lyrics: counts as the median lyrics score
    assert s[4] == -np.inf  # neither lyrics nor features


def test_score_lyrics_only_without_feature_targets():
    target = Target("t", unit(1, 0))
    s = score([target], [unit(1, 0), unit(0, 1)], [(0.9, 0.9), None], 0.5)[0]
    assert list(s) == [1.0, 0.0]


def test_assign_respects_artist_cap_and_skips_exhausted_positions():
    scores = np.array([[0.9, 0.8, 0.1], [0.9, 0.8, 0.1], [0.9, 0.8, -np.inf]])
    picks = assign(scores, ["a", "a", "b"], max_per_artist=1)
    assert picks == [(0, 0), (1, 2)]
