from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

# Load environment variables from backend/.env before importing settings.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from rag.config import settings
from rag.llm import chat_completion
from rag.prompts import build_explanation_messages, build_mcq_messages
from rag.retrieval import retrieve
from rag.store import VectorStore

app = FastAPI(title="MCAT AI Tutor (RAG Prototype)")


class JSONErrorMiddleware(BaseHTTPMiddleware):
    """Catch all exceptions and return JSON for API routes (POST /user/*, /admin/rag)."""

    async def dispatch(self, request: Request, call_next):
        if request.method != "POST" or not (
            request.url.path.startswith("/user/")
            or request.url.path == "/admin/rag"
        ):
            return await call_next(request)
        try:
            return await call_next(request)
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": str(exc),
                    "detail": traceback.format_exc() if os.getenv("DEBUG") else None,
                },
            )


app.add_middleware(JSONErrorMiddleware)


@app.exception_handler(Exception)
async def json_exception_handler(request: Request, exc: Exception):
    """Return JSON for all unhandled exceptions so the frontend can display them."""
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": str(exc),
            "detail": traceback.format_exc() if os.getenv("DEBUG") else None,
        },
    )


# Templates + static
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# In-memory singleton store (persisted to disk on updates)
STORE: VectorStore = VectorStore.load()
ADMIN_INGEST_ENABLED = os.getenv("ENABLE_ADMIN_RAG", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def _pdf_page_count(pdf_path: Path) -> int:
    import fitz

    doc = fitz.open(str(pdf_path))
    return int(doc.page_count)


def _embedding_model_mismatch_error() -> Optional[str]:
    indexed_model = STORE.meta.get("embedding_model")
    if not indexed_model or STORE.is_empty:
        return None
    if indexed_model == settings.embedding_model_name:
        return None
    return (
        "Embedding model mismatch: index built with "
        f"'{indexed_model}' but app is configured with '{settings.embedding_model_name}'. "
        "Set EMBEDDING_MODEL_NAME to match and restart, or re-upload PDFs in /admin/rag."
    )


@app.get("/", response_class=HTMLResponse)
def home() -> RedirectResponse:
    return RedirectResponse(url="/user/explanation")


if ADMIN_INGEST_ENABLED:

    @app.get("/admin/rag", response_class=HTMLResponse)
    def admin_page(request: Request):
        return templates.TemplateResponse(
            "admin.html",
            {
                "request": request,
                "pdfs": STORE.list_pdfs(),
                "chunk_count": len(STORE.chunks),
                "embedding_model": STORE.meta.get("embedding_model", settings.embedding_model_name),
                "dim": STORE.dim,
                "sample_chunks": STORE.sample_chunks(limit=6),
            },
        )


    @app.post("/admin/rag")
    async def admin_upload(files: List[UploadFile] = File(...)):
        global STORE
        from rag.embeddings import embed_passages
        from rag.ingest import ingest_pdf

        if not files:
            return JSONResponse({"ok": False, "error": "No files uploaded"}, status_code=400)

        added = []
        for uf in files:
            if not uf.filename.lower().endswith(".pdf"):
                return JSONResponse(
                    {"ok": False, "error": f"Only PDFs are supported: {uf.filename}"},
                    status_code=400,
                )

            pdf_path = settings.pdf_dir / uf.filename

            # Save to disk
            content = await uf.read()
            pdf_path.write_bytes(content)

            # If already indexed, remove old entries first
            STORE.remove_pdf(uf.filename)

            # OCR + chunk
            chunks = ingest_pdf(pdf_path)
            page_count = _pdf_page_count(pdf_path)

            # Embed
            texts = [c.text for c in chunks]
            embs = embed_passages(texts)

            # Store
            STORE.add_pdf(pdf_name=uf.filename, page_count=page_count, new_chunks=chunks, new_embeddings=embs)
            STORE.save()

            added.append({"pdf": uf.filename, "pages": page_count, "chunks": len(chunks)})

        return {
            "ok": True,
            "added": added,
            "total_pdfs": len(STORE.list_pdfs()),
            "total_chunks": len(STORE.chunks),
            "embedding_model": STORE.meta.get("embedding_model", settings.embedding_model_name),
            "dim": STORE.dim,
        }

else:

    @app.get("/admin/rag", response_class=HTMLResponse)
    def admin_page_disabled():
        return RedirectResponse(url="/user/explanation")


    @app.post("/admin/rag")
    async def admin_upload_disabled():
        return JSONResponse(
            {
                "ok": False,
                "error": "RAG ingestion is disabled on this deployment. Set ENABLE_ADMIN_RAG=1 to enable.",
            },
            status_code=403,
        )


@app.get("/user/explanation", response_class=HTMLResponse)
def explanation_page(request: Request):
    return templates.TemplateResponse(
        "explanation.html",
        {
            "request": request,
            "has_index": not STORE.is_empty,
            "top_k": settings.rag_top_k,
        },
    )


@app.post("/user/explanation")
async def explanation_api(request: Request):
    global STORE
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Invalid JSON: {e}"}, status_code=400)
    question = (payload.get("question") or "").strip()
    try:
        top_k = int(payload.get("top_k") or settings.rag_top_k)
    except (TypeError, ValueError):
        return JSONResponse({"ok": False, "error": "Invalid 'top_k': expected an integer"}, status_code=400)

    if not question:
        return JSONResponse({"ok": False, "error": "Missing 'question'"}, status_code=400)
    if STORE.is_empty:
        return JSONResponse({"ok": False, "error": "Vector store is empty. Upload PDFs in /admin/rag first."}, status_code=400)
    mismatch_error = _embedding_model_mismatch_error()
    if mismatch_error:
        return JSONResponse({"ok": False, "error": mismatch_error}, status_code=400)

    try:
        retrieved = retrieve(STORE, question, top_k=top_k)
        messages = build_explanation_messages(question, retrieved)
        answer = chat_completion(messages, temperature=0.3)
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": str(e)},
            status_code=500,
        )

    sources = [
        {
            "chunk_id": rc.chunk.chunk_id,
            "pdf": rc.chunk.source_pdf,
            "page": rc.chunk.page_number,
            "score": rc.score,
            "preview": rc.chunk.text[:220] + ("…" if len(rc.chunk.text) > 220 else ""),
        }
        for rc in retrieved
    ]

    return {"ok": True, "answer": answer, "sources": sources}


@app.get("/user/mcq", response_class=HTMLResponse)
def mcq_page(request: Request):
    return templates.TemplateResponse(
        "mcq.html",
        {
            "request": request,
            "has_index": not STORE.is_empty,
            "top_k": settings.rag_top_k,
        },
    )


@app.post("/user/mcq")
async def mcq_api(request: Request):
    global STORE
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Invalid JSON: {e}"}, status_code=400)
    topic = (payload.get("topic") or "").strip()
    try:
        top_k = int(payload.get("top_k") or settings.rag_top_k)
    except (TypeError, ValueError):
        return JSONResponse({"ok": False, "error": "Invalid 'top_k': expected an integer"}, status_code=400)

    if not topic:
        return JSONResponse({"ok": False, "error": "Missing 'topic'"}, status_code=400)
    if STORE.is_empty:
        return JSONResponse({"ok": False, "error": "Vector store is empty. Upload PDFs in /admin/rag first."}, status_code=400)
    mismatch_error = _embedding_model_mismatch_error()
    if mismatch_error:
        return JSONResponse({"ok": False, "error": mismatch_error}, status_code=400)

    try:
        retrieved = retrieve(STORE, topic, top_k=top_k)
        messages = build_mcq_messages(topic, retrieved)
        output = chat_completion(messages, temperature=0.4)
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": str(e)},
            status_code=500,
        )

    sources = [
        {
            "chunk_id": rc.chunk.chunk_id,
            "pdf": rc.chunk.source_pdf,
            "page": rc.chunk.page_number,
            "score": rc.score,
            "preview": rc.chunk.text[:220] + ("…" if len(rc.chunk.text) > 220 else ""),
        }
        for rc in retrieved
    ]

    return {"ok": True, "output": output, "sources": sources}
