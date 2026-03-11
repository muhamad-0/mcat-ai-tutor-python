from __future__ import annotations

import os
from functools import lru_cache
from typing import List

import numpy as np

from .config import settings

# Prevent terminal/progress-bar related runtime issues in detached Windows runs.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@lru_cache(maxsize=1)
def _get_model():
    # Import lazily so app startup can bind port quickly on small instances.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model_name)
    return model


def _is_bge_model(name: str) -> bool:
    return "bge" in name.lower()


def embed_passages(texts: List[str]) -> np.ndarray:
    """Embed passages (document chunks). Returns float32 array, L2-normalized."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    model = _get_model()

    # For BGE models, passage embeddings are typically computed on raw passages.
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a query. Returns float32 vector, L2-normalized."""
    model = _get_model()
    q = query

    # For BGE, prefix query with the recommended retrieval instruction.
    if _is_bge_model(settings.embedding_model_name):
        q = "Represent this sentence for searching relevant passages: " + query

    vec = model.encode([q], normalize_embeddings=True)
    return np.asarray(vec[0], dtype=np.float32)
