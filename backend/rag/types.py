from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    source_pdf: str
    page_number: int
    text: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
