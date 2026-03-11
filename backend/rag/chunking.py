from __future__ import annotations

import re
from typing import List

from .text_utils import looks_like_math_line


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunk text into overlapping segments, trying to avoid breaking equations."""
    if not text:
        return []

    # Split into paragraphs first
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current = ""

    def flush_current():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        # If adding this paragraph would overflow, flush what we have.
        if current:
            flush_current()

        # If paragraph is too large, split it by lines.
        if len(para) > chunk_size:
            lines = [ln.rstrip() for ln in para.splitlines() if ln.strip()]
            buf = ""
            for ln in lines:
                cand = (buf + "\n" + ln).strip() if buf else ln
                if len(cand) <= chunk_size:
                    buf = cand
                    continue

                # Before flushing, avoid ending on a math-looking line when possible.
                if buf:
                    buf_lines = buf.splitlines()
                    if buf_lines and looks_like_math_line(buf_lines[-1]) and len(buf_lines) >= 2:
                        # Move the last line into next chunk
                        carry = buf_lines[-1]
                        buf = "\n".join(buf_lines[:-1]).strip()
                        if buf:
                            chunks.append(buf)
                        buf = carry
                    else:
                        chunks.append(buf.strip())
                        # overlap by chars
                        if overlap > 0:
                            tail = chunks[-1][-overlap:]
                            buf = (tail + "\n" + ln).strip()
                        else:
                            buf = ln
                else:
                    buf = ln

            if buf.strip():
                chunks.append(buf.strip())
            current = ""
        else:
            # Start new chunk with this paragraph
            current = para

    if current.strip():
        flush_current()

    # Apply overlap between chunks (char-based) if not already done in line splitting.
    if overlap > 0 and len(chunks) >= 2:
        overlapped: List[str] = []
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped.append(ch)
                continue
            prev = overlapped[-1]
            tail = prev[-overlap:]
            # avoid duplicating if already included
            if ch.startswith(tail):
                overlapped.append(ch)
            else:
                overlapped.append((tail + "\n" + ch).strip())
        chunks = overlapped

    return chunks
