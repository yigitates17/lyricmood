"""Checks the real e5 model on a tiny invented, bilingual topic set.

Runs only when the model is already downloaded (e.g. after one real
similarity search); CI skips it. Numbers recorded in docs/DESIGN.md §8.2.
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from lyricmood.core.embeddings import MODEL_REPO, MODEL_REVISION, E5Embedder  # noqa: E402
from lyricmood.core.similarity import center  # noqa: E402

MODELS_DIR = Path(os.environ.get("LYRICMOOD_DATA_DIR", "~/.lyricmood")).expanduser() / "models"


def _model_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(
        MODEL_REPO, "onnx/model.onnx", revision=MODEL_REVISION, cache_dir=MODELS_DIR
    )
    return isinstance(path, str)


pytestmark = pytest.mark.skipif(not _model_cached(), reason="e5 model not downloaded")

# (topic, language, invented lyric-like line)
SAMPLES = [
    ("loss", "en", "I lost you and the house is quiet, I miss your voice every night"),
    ("loss", "en", "Your coat still hangs by the door though you are gone for good"),
    ("loss", "fi", "Menetin sinut ja talo on hiljainen, kaipaan ääntäsi joka yö"),
    ("loss", "fi", "Sinä lähdit, ja minä istun yksin pimeässä ja odotan turhaan"),
    ("party", "en", "Let's dance all night, the party never stops, turn the music up"),
    ("party", "en", "Friday lights, everybody jump, we're going out until the sun comes up"),
    ("party", "fi", "Bileet jatkuu aamuun asti, tanssitaan koko yö"),
    ("party", "fi", "Perjantai-ilta, kaikki hyppää, mennään ulos auringonnousuun asti"),
    ("rebel", "en", "We won't obey your rules, we'll burn the walls you built around us"),
    ("rebel", "en", "Raise your fists against the kings who sold our future"),
    ("rebel", "fi", "Emme tottele sääntöjäsi, poltamme muurit jotka rakensit"),
    ("rebel", "fi", "Nostakaa nyrkit kuninkaita vastaan jotka myivät tulevaisuutemme"),
]


def test_nearest_neighbour_shares_topic_after_centering(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # never download in tests
    vectors = center(E5Embedder(MODELS_DIR).embed([text for *_, text in SAMPLES]))
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    topics = [topic for topic, *_ in SAMPLES]
    hits = [topics[i] == topics[j] for i, j in enumerate(sims.argmax(1))]
    assert np.mean(hits) >= 0.9
