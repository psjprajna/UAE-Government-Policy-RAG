"""DRY composition of the ``/query`` pipeline for Phase 8 RAGAS evaluation.

Mirrors the recipe wired in :func:`uae_rag.api.main.lifespan`
(``MultiQuery → Hybrid → Rerank → Generator``) so the evaluation harness
runs against the same composition end users hit through ``/query``. Both
call sites stay duplicated for Phase 8 — the lifespan and the
``test_generation_pipeline::pipeline`` fixture are intentionally left
untouched so the RAGAS baseline measures the exact code path ``/query``
ships today. Collapsing all three onto this helper is a post-Phase 8
cleanup, not a Phase 8 deliverable.

Per ADR-0002 this module imports only ``ports``, sibling domain modules,
and :mod:`uae_rag.config` — never ``uae_rag.adapters``. Enforced by
``tests/fitness/test_layer_boundaries.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from uae_rag import config
from uae_rag.generation.answer import Generator
from uae_rag.ingestion.chunker import Chunk, chunk_articles
from uae_rag.ingestion.parser import parse
from uae_rag.ingestion.registry import SOURCES
from uae_rag.ports import EmbeddingsPort, LLMPort, RetrievalPort
from uae_rag.retrieval.multi_query import MultiQueryRetriever
from uae_rag.retrieval.rerank import RerankRetriever

logger = logging.getLogger(__name__)

_MAX_TOKENS = 500
_WARMUP_PROMPT = "Reply with the single word: ready"
_WARMUP_MAX_TOKENS = 8


@dataclass(frozen=True, slots=True)
class ComposedPipeline:
    """The retriever + generator bundle plus the components RAGAS needs.

    ``embedder`` and ``llm`` are surfaced so :mod:`uae_rag.evals.wrappers`
    (Slice B2) can wrap them as the RAGAS judge embeddings and judge LLM
    without re-resolving them through ``config.py``.
    """

    retriever: RetrievalPort
    generator: Generator
    embedder: EmbeddingsPort
    llm: LLMPort


def _load_chunks(raw_dir: Path) -> list[Chunk]:
    """Reparse every registered PDF present in ``raw_dir``; return all chunks.

    Mirrors ``uae_rag.api.main._load_chunks`` but parameterised on
    ``raw_dir`` so an evaluation run can point at a non-default corpus.
    Missing PDFs are skipped so the helper boots cleanly without a fully
    populated corpus — downstream code then degrades to refusal-mode
    answers, which is a useful "no-evidence" floor for the eval.
    """
    chunks: list[Chunk] = []
    for src in SOURCES:
        pdf = raw_dir / src.local_filename
        if not pdf.exists():
            continue
        articles = parse(pdf, src)
        chunks.extend(
            chunk_articles(articles, source_slug=src.slug, max_tokens=_MAX_TOKENS)
        )
    return chunks


def compose_pipeline(
    *,
    chroma_dir: Path,
    raw_dir: Path,
    n_variations: int = 3,
    candidate_top_k: int = 20,
    warmup: bool = True,
) -> ComposedPipeline:
    """Compose ``MultiQuery → Hybrid → Rerank → Generator``.

    Identical recipe to :func:`uae_rag.api.main.lifespan` (see
    ``app/src/uae_rag/api/main.py`` ``lifespan``). Unlike the lifespan,
    exceptions are NOT swallowed — callers (the eval CLI, integration
    tests) decide whether to skip or exit on failure.

    ``warmup=True`` runs a trivial LLM probe to surface a down/unpulled
    Ollama daemon as :class:`uae_rag.ports.LLMUnavailableError` here
    rather than mid-evaluation, plus a one-hit retrieve to force the
    cross-encoder cold-start.
    """
    embedder = config.get_embeddings()
    vector_index = config.get_vector_index(persist_dir=chroma_dir, embedder=embedder)
    chunks = _load_chunks(raw_dir)
    llm = config.get_llm()
    hybrid = config.get_retriever(
        chunks=chunks, embedder=embedder, vector_index=vector_index
    )
    multi = MultiQueryRetriever(retriever=hybrid, llm=llm, n_variations=n_variations)
    reranker = config.get_reranker()
    reranked = RerankRetriever(
        retriever=multi, reranker=reranker, candidate_top_k=candidate_top_k
    )
    if warmup:
        llm.generate(_WARMUP_PROMPT, max_output_tokens=_WARMUP_MAX_TOKENS)
        reranked.retrieve("warmup", top_k=1)
    logger.info(
        "pipeline composed: %d chunks, embedder=%s, reranker=%s, llm=%s",
        len(chunks),
        embedder.model_id,
        reranker.model_id,
        llm.model_id,
    )
    return ComposedPipeline(
        retriever=reranked,
        generator=Generator(llm=llm),
        embedder=embedder,
        llm=llm,
    )


__all__ = ["ComposedPipeline", "compose_pipeline"]
