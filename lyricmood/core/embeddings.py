"""Local text embeddings: intfloat/multilingual-e5-small via ONNX Runtime.

Uses the ONNX export from the model's official repo, so there is no
PyTorch dependency. The model (~470 MB) downloads once into
`<data dir>/models` on first use. The packages come from the optional
`embeddings` extra: `uv sync --extra embeddings`.
"""

import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from lyricmood.core.errors import ConfigError

MODEL_REPO = "intfloat/multilingual-e5-small"
# Pinned so a model update can't silently change vectors already stored.
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MAX_TOKENS = 512

INSTALL_HINT = (
    "Lyrics similarity needs the optional embeddings packages. "
    "Run `uv sync --extra embeddings` in the lyricmood folder."
)


class Embedder(Protocol):
    # Part of every stored vector's key, so switching models never mixes
    # vector spaces; old vectors are simply recomputed lazily.
    model_id: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normalized float32 vectors, one row per text."""
        ...


class E5Embedder:
    """e5 expects a prefix: "passage: " for lyrics. Lyrics are compared with
    lyrics, so every stored vector uses the same prefix."""

    model_id = f"multilingual-e5-small@{MODEL_REVISION[:7]}:passage"
    dim = 384

    def __init__(self, models_dir: Path, batch_size: int = 16):
        self._models_dir = models_dir
        self._batch_size = batch_size
        self._session = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        with self._lock:
            if self._session is not None:
                return
            try:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                from tokenizers import Tokenizer
            except ImportError:
                raise ConfigError(INSTALL_HINT) from None

            def fetch(name: str) -> str:
                return hf_hub_download(
                    MODEL_REPO,
                    f"onnx/{name}",
                    revision=MODEL_REVISION,
                    cache_dir=self._models_dir,
                )

            tokenizer = Tokenizer.from_file(fetch("tokenizer.json"))
            tokenizer.enable_truncation(MAX_TOKENS)
            tokenizer.enable_padding()
            self._tokenizer = tokenizer
            self._session = ort.InferenceSession(
                fetch("model.onnx"), providers=["CPUExecutionProvider"]
            )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._load()
        out = []
        for start in range(0, len(texts), self._batch_size):
            batch = [f"passage: {t}" for t in texts[start : start + self._batch_size]]
            enc = self._tokenizer.encode_batch(batch)
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            hidden = self._session.run(
                None,
                {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)},
            )[0]
            # e5 uses mean pooling over real (non-padding) tokens.
            pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
            out.append(pooled)
        return normalize_rows(np.concatenate(out).astype(np.float32))


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)
