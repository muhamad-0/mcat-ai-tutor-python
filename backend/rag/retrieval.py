from __future__ import annotations

from typing import List

import numpy as np

from .embeddings import embed_query
from .types import RetrievedChunk
from .store import VectorStore


def retrieve(store: VectorStore, query: str, top_k: int = 6) -> List[RetrievedChunk]:
    if store.is_empty:
        return []

    q = embed_query(query)

    # Cosine similarity via dot product because embeddings are L2-normalized
    scores = store.embeddings @ q

    if top_k <= 0:
        top_k = 1
    top_k = min(top_k, len(store.chunks))

    # Argpartition for speed
    idxs = np.argpartition(-scores, top_k - 1)[:top_k]
    idxs_sorted = idxs[np.argsort(-scores[idxs])]

    results: List[RetrievedChunk] = []
    for i in idxs_sorted:
        results.append(RetrievedChunk(chunk=store.chunks[int(i)], score=float(scores[int(i)])))

    return results
