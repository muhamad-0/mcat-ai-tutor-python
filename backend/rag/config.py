from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Directories
    base_dir: Path = Path(__file__).resolve().parents[1]  # backend/
    data_dir: Path = Path(__file__).resolve().parents[1] / "data"
    pdf_dir: Path = Path(__file__).resolve().parents[1] / "data" / "pdfs"
    ocr_cache_dir: Path = Path(__file__).resolve().parents[1] / "data" / "ocr_cache"
    index_dir: Path = Path(__file__).resolve().parents[1] / "data" / "index"

    # Models
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")

    # RAG settings
    rag_top_k: int = _env_int("RAG_TOP_K", 6)
    chunk_size_chars: int = _env_int("CHUNK_SIZE_CHARS", 1400)
    chunk_overlap_chars: int = _env_int("CHUNK_OVERLAP_CHARS", 200)

    # OCR
    ocr_zoom: float = _env_float("OCR_ZOOM", 2.0)


settings = Settings()

# Ensure folders exist
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.pdf_dir.mkdir(parents=True, exist_ok=True)
settings.ocr_cache_dir.mkdir(parents=True, exist_ok=True)
settings.index_dir.mkdir(parents=True, exist_ok=True)
