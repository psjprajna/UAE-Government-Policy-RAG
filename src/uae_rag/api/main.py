"""FastAPI entry point — wires the RAG pipeline behind ``POST /query``.

The lifespan handler composes ``MultiQueryRetriever → HybridRetriever →
RerankRetriever → Generator`` once at boot and stores the bundle on
``app.state.pipeline``. Adapter construction is side-effect-light
(``@cached_property`` defers every model load to first use), so the boot
itself stays under a couple of seconds; the first ``/query`` pays the model-
load tax.

Wire ``Citation`` carries the ``[N]`` marker the in-answer text uses, deduped
on ``(source, article)`` so two chunks from the same article collapse to one
entry. Transport failures from the local LLM surface as a 503; upstream
``ValueError`` (e.g. whitespace-only question) surfaces as a 400 — the
``detail`` is the upstream exception's own message verbatim, not a hand-
rolled string.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from uae_rag import config
from uae_rag.generation.answer import AnswerPayload, Generator
from uae_rag.ingestion.chunker import Chunk, chunk_articles
from uae_rag.ingestion.parser import parse
from uae_rag.ingestion.registry import SOURCES
from uae_rag.ports import LLMUnavailableError, RetrievalPort
from uae_rag.retrieval.multi_query import MultiQueryRetriever
from uae_rag.retrieval.rerank import RerankRetriever

logger = logging.getLogger(__name__)

_APP_ROOT = Path(__file__).resolve().parents[3]
_CHROMA_DIR = _APP_ROOT / "data" / "chroma_db"
_RAW_DIR = _APP_ROOT / "data" / "raw"
_MAX_TOKENS = 500  # matches scripts/build_index.py
_RETRIEVE_TOP_K = 5
_CANDIDATE_TOP_K = 20
_N_VARIATIONS = 3


class Citation(BaseModel):
    marker: str = Field(..., description="In-answer marker, e.g. '[1]'")
    source: str = Field(..., description="Document slug, e.g. 'labour-law-en'")
    article: str = Field(..., description="Article id or breadcrumb tail")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    language: str = Field(..., description="Detected language: 'en' or 'ar'")


@dataclass(frozen=True, slots=True)
class Pipeline:
    """Bundle the composed retriever + generator stored on ``app.state``."""

    retriever: RetrievalPort
    generator: Generator


def _load_chunks() -> list[Chunk]:
    """Reparse every PDF in ``_RAW_DIR`` listed in ``SOURCES``; return all chunks.

    Mirrors ``tests/integration/test_generation_pipeline.py::_reparse_chunks``:
    missing PDFs are skipped (``continue``) so the server boots cleanly even
    without a populated corpus — the system then degrades to refusal-mode on
    every query. ``count_tokens`` defaults to the word-count proxy so this
    helper never triggers an embedder model load at boot.
    """
    chunks: list[Chunk] = []
    for src in SOURCES:
        pdf = _RAW_DIR / src.local_filename
        if not pdf.exists():
            continue
        articles = parse(pdf, src)
        chunks.extend(
            chunk_articles(articles, source_slug=src.slug, max_tokens=_MAX_TOKENS)
        )
    return chunks


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Compose the RAG pipeline once at server boot; tear nothing down.

    Composition is intentionally cheap: every adapter sets ``model_id`` /
    ``dimension`` from env vars in ``__init__`` and defers the actual weight
    load to ``@cached_property`` on first call. First request pays the model-
    load tax; boot stays under a few seconds.
    """
    try:
        embedder = config.get_embeddings()
        vector_index = config.get_vector_index(persist_dir=_CHROMA_DIR, embedder=embedder)
        chunks = _load_chunks()
        llm = config.get_llm()
        hybrid = config.get_retriever(
            chunks=chunks, embedder=embedder, vector_index=vector_index
        )
        multi = MultiQueryRetriever(retriever=hybrid, llm=llm, n_variations=_N_VARIATIONS)
        reranker = config.get_reranker()
        reranked = RerankRetriever(
            retriever=multi, reranker=reranker, candidate_top_k=_CANDIDATE_TOP_K
        )
        app.state.pipeline = Pipeline(retriever=reranked, generator=Generator(llm=llm))
        logger.info(
            "pipeline composed: %d chunks, embedder=%s, reranker=%s, llm=%s",
            len(chunks),
            embedder.model_id,
            reranker.model_id,
            llm.model_id,
        )
    except Exception:
        logger.exception(
            "pipeline composition failed; /query will return 503 until "
            "the cause is fixed and the server is restarted"
        )
        app.state.pipeline = None
    yield


app = FastAPI(
    title="UAE Government Policy RAG",
    version="0.0.1",
    description="Retrieval-augmented QA over UAE Labour Law and related regulations.",
    lifespan=lifespan,
)


def get_pipeline(request: Request) -> Pipeline:
    """Return the lifespan-composed pipeline; 503 if composition failed.

    Tests override this dependency via ``app.dependency_overrides`` — the
    override is consulted at request time regardless of whether lifespan ran.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503, detail="pipeline unavailable (see server logs)"
        )
    return pipeline


def to_api_response(payload: AnswerPayload) -> QueryResponse:
    """Dedup domain citations on ``(source, article)``; promote markers to wire."""
    seen: set[tuple[str, str]] = set()
    api_citations: list[Citation] = []
    for c in payload.citations:
        key = (c.source, c.article)
        if key in seen:
            continue
        seen.add(key)
        api_citations.append(Citation(marker=c.marker, source=c.source, article=c.article))
    return QueryResponse(
        answer=payload.answer,
        citations=api_citations,
        language=payload.language,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    pipeline: Annotated[Pipeline, Depends(get_pipeline)],
) -> QueryResponse:
    """Retrieve grounded passages and generate a cited answer in the question's language."""
    try:
        hits = pipeline.retriever.retrieve(request.question, top_k=_RETRIEVE_TOP_K)
        payload = pipeline.generator.generate(request.question, hits)
    except LLMUnavailableError as exc:
        logger.exception("LLM unavailable during /query")
        raise HTTPException(status_code=503, detail="LLM unavailable") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.debug(
        "/query language=%s hits=%d answer_len=%d",
        payload.language,
        len(payload.citations),
        len(payload.answer),
    )
    return to_api_response(payload)
