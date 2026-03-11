from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from .config import settings
from .text_utils import normalize_math_text


_tesseract_cmd = os.getenv("TESSERACT_CMD")
if _tesseract_cmd:
    # Allow explicit binary path on systems where Tesseract is not on PATH.
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd


def _fingerprint_pdf(pdf_path: Path) -> str:
    """Fast-ish fingerprint for caching OCR results."""
    stat = pdf_path.stat()
    h = hashlib.sha1()
    h.update(str(pdf_path.name).encode("utf-8"))
    h.update(str(stat.st_size).encode("utf-8"))
    h.update(str(int(stat.st_mtime)).encode("utf-8"))

    # Add a small content sample (first 1MB) for a bit more stability.
    with pdf_path.open("rb") as f:
        h.update(f.read(1024 * 1024))

    return h.hexdigest()[:16]


def ocr_pdf_to_pages(pdf_path: Path) -> List[str]:
    """Run OCR on every page and return a list of per-page text."""
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    fp = _fingerprint_pdf(pdf_path)
    cache_dir = settings.ocr_cache_dir / fp
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_path = cache_dir / "meta.json"

    # If cache exists and matches page count, reuse.
    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("pdf_name") == pdf_path.name and (cache_dir / "page_0.txt").exists():
                page_count = meta.get("page_count")
                if isinstance(page_count, int) and page_count > 0:
                    pages = []
                    all_present = True
                    for i in range(page_count):
                        p = cache_dir / f"page_{i}.txt"
                        if not p.exists():
                            all_present = False
                            break
                        pages.append(p.read_text(encoding="utf-8"))
                    if all_present:
                        return pages
    except Exception:
        # If cache is corrupted, fall through and rebuild.
        pass

    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count

    pages_text: List[str] = []

    zoom = settings.ocr_zoom
    matrix = fitz.Matrix(zoom, zoom)

    # Tesseract settings: PSM 6 works well for dense text blocks.
    tesseract_config = "--oem 1 --psm 6"

    for page_index in range(page_count):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # Convert to grayscale to help OCR
        img = img.convert("L")

        raw_text = pytesseract.image_to_string(img, lang="eng", config=tesseract_config)
        cleaned = normalize_math_text(raw_text)

        pages_text.append(cleaned)
        (cache_dir / f"page_{page_index}.txt").write_text(cleaned, encoding="utf-8")

    meta = {
        "pdf_name": pdf_path.name,
        "page_count": page_count,
        "ocr_zoom": zoom,
        "tesseract_config": tesseract_config,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return pages_text
