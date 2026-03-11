from __future__ import annotations

from pathlib import Path
from typing import List

from .chunking import chunk_text
from .config import settings
from .ocr import ocr_pdf_to_pages
from .types import Chunk


def ingest_pdf(pdf_path: Path) -> List[Chunk]:
    """OCR + chunk a PDF into retrievable chunks."""
    pages = ocr_pdf_to_pages(pdf_path)
    chunks: List[Chunk] = []

    for page_idx, page_text in enumerate(pages):
        page_chunks = chunk_text(
            page_text,
            chunk_size=settings.chunk_size_chars,
            overlap=settings.chunk_overlap_chars,
        )
        for j, ch_text in enumerate(page_chunks):
            chunk_id = f"{pdf_path.stem}__p{page_idx+1:03d}__c{j:03d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_pdf=pdf_path.name,
                    page_number=page_idx + 1,
                    text=ch_text,
                )
            )

    return chunks
