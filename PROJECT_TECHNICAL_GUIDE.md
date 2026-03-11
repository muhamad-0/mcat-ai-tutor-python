# MCAT AI Tutor Technical Guide

## 1) Design Structure

### 1.1 High-level architecture
This project is a retrieval-augmented generation (RAG) web app with 3 layers:

1. Interface layer (HTML + JS templates)
2. Application/API layer (FastAPI routes)
3. Knowledge layer (OCR, chunking, embeddings, retrieval, vector store)

Request flow:

1. User submits a question/topic from `/user/explanation` or `/user/mcq`.
2. Backend embeds the query.
3. Backend computes cosine similarity against local chunk embeddings.
4. Top-K chunks are injected into a prompt.
5. OpenAI chat model generates final response.
6. UI renders answer + retrieved sources.

Admin ingestion flow:

1. Admin uploads PDFs on `/admin/rag`.
2. PDFs are OCR-processed page by page.
3. Text is normalized and chunked.
4. Chunks are embedded locally.
5. Index is persisted in `backend/data/index/`.

### 1.2 Key modules

- `backend/main.py`: Route wiring and orchestration.
- `backend/rag/ocr.py`: PDF page rendering + Tesseract OCR + cache.
- `backend/rag/text_utils.py`: OCR normalization and math heuristics.
- `backend/rag/chunking.py`: Chunk splitter with overlap and math-aware split guard.
- `backend/rag/embeddings.py`: SentenceTransformer loading and embedding functions.
- `backend/rag/retrieval.py`: Similarity scoring and top-K retrieval.
- `backend/rag/prompts.py`: System prompts and message builders.
- `backend/rag/llm.py`: OpenAI client wrapper.
- `backend/rag/store.py`: Local vector DB (`documents.jsonl`, `embeddings.npy`, `meta.json`).

---

## 2) Frontend

### 2.1 Pages

- `backend/templates/admin.html`
  - Upload PDFs and trigger indexing.
  - Shows current indexed PDFs, chunk count, embedding dim, sample chunks.

- `backend/templates/explanation.html`
  - Takes free-text question + top_k.
  - Calls `POST /user/explanation`.
  - Displays response and retrieved chunk previews.

- `backend/templates/mcq.html`
  - Takes topic + top_k.
  - Calls `POST /user/mcq`.
  - Displays generated MCQ and retrieved sources.

### 2.2 UI implementation notes

- Server-rendered templates via Jinja.
- Vanilla JS fetch calls (no framework state management).
- Shared styling in `backend/static/styles.css`.
- Current UX strengths: simple and debuggable.
- Current UX gap: no streamed responses, no LaTeX rendering (raw text only).

---

## 3) Backend

### 3.1 API routes

- `GET /admin/rag`, `POST /admin/rag`
- `GET /user/explanation`, `POST /user/explanation`
- `GET /user/mcq`, `POST /user/mcq`

### 3.2 Core behavior

- Store loaded once at startup into `STORE` singleton.
- Uploaded PDF replaces older index entries with same filename.
- Retrieval returns chunk metadata and scores for transparency.
- Explanation/MCQ generation both enforce context-grounded prompting.

### 3.3 Local DB format

Stored under `backend/data/index/`:

- `documents.jsonl`: chunk text + metadata
- `embeddings.npy`: float32 matrix `[num_chunks, dim]`
- `meta.json`: ingestion metadata, PDF stats, model name

This is optimized for small corpora and quick local use.

---

## 4) Agents

There is no formal multi-agent runtime in code today. Operationally, the system has role-like components:

1. Admin role: curates and ingests source PDFs.
2. Retrieval role: selects top-K grounded context chunks.
3. Tutor role: generates explanation/MCQ text from context.

Recommended next step if you want explicit agents:

1. Ingestion Agent: OCR quality checks, duplicate detection, chunk diagnostics.
2. Retrieval Agent: query rewriting + hybrid retrieval + reranking.
3. Response Agent: answer drafting with citation constraints.
4. Critic Agent: post-check faithfulness and math consistency before final output.

---

## 5) Embedding Model

### 5.1 Current behavior

- Code default: `BAAI/bge-base-en-v1.5` (from env fallback).
- Query instruction prefix is applied for BGE models:
  - `Represent this sentence for searching relevant passages: ...`
- Embeddings are L2-normalized.
- Similarity is dot product = cosine similarity under normalization.

### 5.2 Current runtime setup

Your `.env` is configured for:

- `EMBEDDING_MODEL_NAME=BAAI/bge-large-en-v1.5`

This increases representation capacity compared with base model, at higher latency/memory cost.

---

## 6) System Prompt Methodology

Prompt design is centralized in `backend/rag/prompts.py` and uses:

1. Strong system constraints:
   - Use only retrieved context for factual claims.
   - Admit missing context rather than hallucinating.
   - Keep equations and variable naming consistent.

2. Fixed pedagogical structure:
   - Toolkit
   - Think it through
   - Analogy
   - MCAT Trap
   - Memory Rule

3. Context serialization:
   - Retrieved chunks are injected as a structured context block with source/page/score.

Why this works:

- Reduces style drift.
- Improves answer consistency across prompts.
- Makes grading and automated evaluation easier due to predictable structure.

---

## 7) Main Response Model

Current model config is from `OPENAI_MODEL` env, defaulting to:

- `gpt-4o`

Usage:

- Explanation endpoint: lower temperature (`0.3`) for stable tutoring.
- MCQ endpoint: slightly higher temperature (`0.4`) for varied question construction.

Dependency:

- Requires `OPENAI_API_KEY` at runtime.

---

## 8) Tips to Improve

### 8.1 Retrieval quality

1. Add hybrid retrieval (vector + BM25 keyword search).
2. Add reranker (cross-encoder) on top 20 candidates.
3. Add metadata filtering (chapter, concept, formula tags).
4. Tune chunk size/overlap using measured retrieval recall.

### 8.2 Response quality

1. Add explicit inline source citations in generated answer.
2. Enforce structured JSON output before rendering.
3. Add math post-processor for equation normalization checks.
4. Add refusal behavior for out-of-scope topics.

### 8.3 Reliability and ops

1. Add startup validation for OCR, model availability, and env variables.
2. Add background job queue for ingestion with progress tracking.
3. Add logs for retrieval distribution and model latency.
4. Add versioning in `meta.json` for backward-compatible index upgrades.

---

## 9) Making a Test Set

Create a small but high-quality benchmark first (50-150 items), then expand.

### 9.1 Test item schema

For each sample, store:

- `id`
- `task_type`: `explanation` or `mcq`
- `query`
- `expected_concepts`: list of required ideas/equations
- `gold_sources`: expected PDF/page references
- `difficulty`: easy/medium/hard
- `reference_answer` (optional but useful)

### 9.2 Data collection strategy

1. Extract candidate questions from the two source PDFs.
2. Ask a domain reviewer to label expected concepts and key equations.
3. Include adversarial prompts:
   - vague wording
   - near-duplicate concepts
   - common misconception prompts
4. Include negative tests where context is intentionally insufficient.

### 9.3 Splits

- `dev`: prompt/retrieval iteration
- `test`: frozen evaluation set (never tune directly on this)

---

## 10) Measure Similarity

Use metrics at both retrieval and generation levels.

### 10.1 Retrieval metrics

1. Recall@K: whether at least one gold source is retrieved.
2. MRR: rank quality for first relevant chunk.
3. nDCG@K: graded ranking quality if multiple relevant chunks exist.

### 10.2 Semantic similarity metrics

For generated answers vs reference answers:

1. Embedding cosine similarity (SentenceTransformer/BGE embedding of full answer).
2. BERTScore or token-level semantic overlap.
3. Concept coverage score:
   - `% expected_concepts mentioned correctly`

### 10.3 Faithfulness checks

1. Citation overlap:
   - Are cited claims traceable to retrieved chunks?
2. Unsupported claim rate:
   - Count of statements not found in context.

---

## 11) LLM-as-Judge to Quantify Quality

Use a separate evaluator model with a strict rubric and forced JSON output.

### 11.1 Judge rubric (0-5 each)

1. Groundedness: factual consistency with provided context.
2. Concept correctness: physics correctness.
3. Pedagogical clarity: step quality for MCAT learner.
4. Structure compliance: required section order and completeness.
5. Conciseness: no unnecessary verbosity.

Final score example:

`final = 0.30*groundedness + 0.25*concept + 0.20*clarity + 0.15*structure + 0.10*conciseness`

### 11.2 Judge prompt pattern

Inputs:

- user query
- retrieved context
- model answer
- (optional) reference answer

Output (strict JSON):

- per-dimension scores
- short rationales
- hallucination flags
- missing-concept flags
- pass/fail

### 11.3 Calibration process

1. Manually score 30-50 samples.
2. Compare human vs judge agreement.
3. Adjust rubric wording until agreement stabilizes.
4. Freeze rubric and judge model version before reporting metrics.

---

## 12) Suggested Evaluation Dashboard

Track these over time:

1. Retrieval Recall@6
2. nDCG@6
3. Avg groundedness score (LLM judge)
4. Avg concept correctness score
5. Hallucination rate
6. Median latency (retrieval and generation)
7. Cost per request

This gives a clear quality/cost/performance view for iterative improvement.
