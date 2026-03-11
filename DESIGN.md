# Design note (RAG tutor prototype)

This prototype follows the exercise requirements:

- Ingest two PDFs as the knowledge base
- Chunk into retrievable pieces
- Create embeddings locally
- Retrieve relevant chunks for a user query
- Use retrieved context to generate:
  - Tutor-style explanations
  - MCAT-style multiple choice questions
- Keep **math formatting stable** (LaTeX or clear plain text) through the pipeline

## 1) Ingestion & OCR

The provided PDFs are scanned/image-based, so text extraction via standard PDF parsers returns empty.

**Approach**

1. Render each PDF page to a PNG via **PyMuPDF** (zoom configurable via `OCR_ZOOM`).
2. Run **Tesseract OCR** on each rendered page using `--psm 6` (dense text block mode).
3. Cache OCR output per PDF fingerprint under `backend/data/ocr_cache/` to avoid reprocessing.

## 2) Math-safe text normalization

OCR output is normalized before chunking/embedding:

- Fix hyphenation across line breaks (e.g., `incom-\npressible → incompressible`)
- Map common Greek letters to ASCII names (`ρ → rho`, `μ → mu`, `Δ → Delta`, …)
- Convert superscripts/subscripts to `^` / `_` patterns when possible (`² → ^2`)
- Collapse extra whitespace while keeping newlines

Goal: not “perfect LaTeX”, but **stable, searchable equation strings** (e.g., `P + 1/2 rho v^2 + rho g h`).

## 3) Chunking

Chunking is done **per page**, then split into ~`CHUNK_SIZE_CHARS` character chunks with overlap.

- Primary split: paragraph boundaries (blank lines)
- If a single paragraph is too large: split by lines
- A small heuristic avoids ending a chunk on a line that “looks like math” (contains `=`, operators, digits, greek tokens)
- Overlap is character-based (`CHUNK_OVERLAP_CHARS`) to preserve continuity

Each chunk stores metadata:

- `source_pdf`
- `page_number`
- `chunk_id` (stable id like `Princeton__p003__c002`)

## 4) Embeddings & storage

**Embedding model (local)**

Default: `BAAI/bge-base-en-v1.5` via `sentence-transformers`.

- Runs locally (CPU is fine for a small corpus)
- Query embeddings use the recommended BGE instruction prefix:
  - `Represent this sentence for searching relevant passages: <query>`

**Vector storage**

Since the corpus is small, the store is implemented with:

- `documents.jsonl` → chunk text + metadata
- `embeddings.npy` → float32 numpy matrix (L2-normalized)

Retrieval uses cosine similarity computed as dot product.

## 5) Retrieval

At query time:

1. Embed the user query
2. Compute cosine similarity against all chunk vectors
3. Select top-K chunks (`RAG_TOP_K`)

The API returns the retrieved chunk previews + scores to make debugging easier.

## 6) Prompting & explanation style control

The exercise asks for tutor-style explanations with a specific structure.

This prototype enforces that via a **system prompt** that requires:

1. Toolkit (3–5 equations/concepts)
2. Think it through (step-by-step reasoning)
3. Analogy
4. MCAT Trap
5. Memory Rule

Both `/user/explanation` and `/user/mcq` pass the retrieved chunks as a “Context” block.

To keep math intact, the prompt requires **LaTeX equations** (inline `$...$`) and consistent variable naming.

## 7) Admin UI

The admin page (`/admin/rag`) supports:

- PDF upload
- Display of which PDFs are currently indexed
- Total chunk count + embedding model + a sample of chunks

This satisfies “upload PDF and current PDF in vector data” requirements.

## 8) Improvements with more time

- Better OCR: deskewing, thresholding, per-region OCR for equations/figures
- Hybrid retrieval: combine vector search + BM25 keyword search
- Reranking: add a lightweight cross-encoder reranker for better relevance
- Structured outputs: enforce JSON schemas for MCQs to avoid formatting drift
- Math rendering: add KaTeX/MathJax on the frontend for rendered equations
- Source citations: attach page references directly inside the answer
