from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import settings
from .types import Chunk


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class VectorStore:
    """A tiny local vector store backed by numpy + jsonl.

    Designed for small prototypes (hundreds/thousands of chunks).
    """

    def __init__(self, chunks: Optional[List[Chunk]] = None, embeddings: Optional[np.ndarray] = None, meta: Optional[Dict] = None):
        self.chunks: List[Chunk] = chunks or []
        self.embeddings: np.ndarray = embeddings if embeddings is not None else np.zeros((0, 1), dtype=np.float32)
        self.meta: Dict = meta or {}

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    @property
    def dim(self) -> int:
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] == 0:
            return 0
        return int(self.embeddings.shape[1])

    @classmethod
    def load(cls) -> "VectorStore":
        idx = settings.index_dir
        docs_path = idx / "documents.jsonl"
        emb_path = idx / "embeddings.npy"
        meta_path = idx / "meta.json"

        if not docs_path.exists() or not emb_path.exists() or not meta_path.exists():
            return cls(meta={"created_at": None, "pdfs": {}, "embedding_model": settings.embedding_model_name})

        chunks: List[Chunk] = []
        with docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                chunks.append(Chunk(**obj))

        embeddings = np.load(str(emb_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        store = cls(chunks=chunks, embeddings=embeddings.astype(np.float32), meta=meta)
        return store

    def save(self) -> None:
        idx = settings.index_dir
        idx.mkdir(parents=True, exist_ok=True)

        docs_path = idx / "documents.jsonl"
        emb_path = idx / "embeddings.npy"
        meta_path = idx / "meta.json"

        with docs_path.open("w", encoding="utf-8") as f:
            for ch in self.chunks:
                f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")

        np.save(str(emb_path), self.embeddings.astype(np.float32))
        meta_path.write_text(json.dumps(self.meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def add_pdf(self, pdf_name: str, page_count: int, new_chunks: List[Chunk], new_embeddings: np.ndarray) -> None:
        if len(new_chunks) == 0:
            return
        if new_embeddings.ndim != 2 or new_embeddings.shape[0] != len(new_chunks):
            raise ValueError("Embeddings shape mismatch")

        if self.embeddings.shape[0] == 0:
            self.embeddings = new_embeddings.astype(np.float32)
        else:
            if self.embeddings.shape[1] != new_embeddings.shape[1]:
                raise ValueError(
                    f"Embedding dimension mismatch. Existing dim={self.embeddings.shape[1]}, new dim={new_embeddings.shape[1]}"
                )
            self.embeddings = np.vstack([self.embeddings, new_embeddings.astype(np.float32)])

        self.chunks.extend(new_chunks)

        self.meta.setdefault("pdfs", {})
        self.meta.setdefault("embedding_model", settings.embedding_model_name)
        self.meta.setdefault("created_at", self.meta.get("created_at") or _utc_now_iso())
        self.meta["updated_at"] = _utc_now_iso()

        self.meta["pdfs"][pdf_name] = {
            "added_at": _utc_now_iso(),
            "page_count": int(page_count),
            "chunk_count": int(len(new_chunks)),
        }

    def remove_pdf(self, pdf_name: str) -> None:
        """Remove all chunks/embeddings associated with a PDF (if present)."""
        if self.is_empty:
            return

        keep_indices = [i for i, ch in enumerate(self.chunks) if ch.source_pdf != pdf_name]
        if len(keep_indices) == len(self.chunks):
            return  # nothing to remove

        self.chunks = [self.chunks[i] for i in keep_indices]
        self.embeddings = (
            self.embeddings[keep_indices, :].astype(np.float32)
            if len(keep_indices) > 0
            else np.zeros((0, 1), dtype=np.float32)
        )

        if "pdfs" in self.meta and pdf_name in self.meta["pdfs"]:
            del self.meta["pdfs"][pdf_name]
            self.meta["updated_at"] = _utc_now_iso()

    def list_pdfs(self) -> List[Dict]:
        pdfs = self.meta.get("pdfs", {})
        out = []
        for name, info in pdfs.items():
            out.append({"name": name, **info})
        # Sort by name for stable display
        out.sort(key=lambda x: x["name"])
        return out

    def sample_chunks(self, limit: int = 5) -> List[Dict]:
        out = []
        for ch in self.chunks[:limit]:
            out.append({
                "chunk_id": ch.chunk_id,
                "source_pdf": ch.source_pdf,
                "page_number": ch.page_number,
                "preview": (ch.text[:240] + "…") if len(ch.text) > 240 else ch.text,
            })
        return out
