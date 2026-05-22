"""End-to-end integration test for the Phase 6 generation pipeline.

Wraps the production hybrid retriever (BM25 + dense + RRF) with multi-query
expansion, the cross-encoder reranker (BGE-v2-m3), and the local Ollama LLM
to confirm three canonical questions (EN annual leave, EN probation, AR
annual leave) produce grounded answers with bracketed citations referring to
the right article. Marked ``adapter_local`` and ``slow``; further guarded by
file-presence / SDK-import / daemon-reachability checks so a missing corpus,
absent model, or down daemon just skips cleanly.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from uae_rag import config
from uae_rag.generation.answer import AnswerPayload, Generator
from uae_rag.ingestion.chunker import chunk_articles
from uae_rag.ingestion.parser import parse
from uae_rag.ingestion.registry import SOURCES
from uae_rag.ports import LLMUnavailableError
from uae_rag.retrieval.multi_query import MultiQueryRetriever
from uae_rag.retrieval.rerank import RerankRetriever

if TYPE_CHECKING:
    from uae_rag.ingestion.chunker import Chunk

pytestmark = [pytest.mark.adapter_local, pytest.mark.slow]

_APP_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _APP_ROOT / "data"
_CHROMA_DIR = _DATA_DIR / "chroma_db"
_RAW_DIR = _DATA_DIR / "raw"
_MAX_TOKENS = 500  # matches scripts/build_index.py
_MARKER_RE = re.compile(r"\[\d+\]")
_ANSWER_MAX_CHARS = 600


@dataclass(frozen=True, slots=True)
class _CanonicalQuery:
    question: str
    expected_article: str
    label: str


_CANONICAL_QUERIES = (
    _CanonicalQuery(
        question="What is the annual leave entitlement?",
        expected_article="29",
        label="en-annual-leave",
    ),
    _CanonicalQuery(
        question="How long is the probationary period?",
        expected_article="9",
        label="en-probation",
    ),
    _CanonicalQuery(
        question="ما هي مدة الإجازة السنوية؟",
        expected_article="29",
        label="ar-annual-leave",
    ),
)


def _skip_unless_corpus_and_model_available() -> None:
    if not _CHROMA_DIR.exists():
        pytest.skip(f"corpus index missing: {_CHROMA_DIR}")
    if not _RAW_DIR.exists():
        pytest.skip(f"raw PDFs missing: {_RAW_DIR}")
    if importlib.util.find_spec("chromadb") is None:
        pytest.skip("chromadb not installed")
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence_transformers not installed")


def _reparse_chunks(count_tokens: Callable[[str], int]) -> list[Chunk]:
    """Rebuild the in-memory chunk corpus using the same pipeline as build_index.py."""
    chunks: list[Chunk] = []
    for src in SOURCES:
        pdf = _RAW_DIR / src.local_filename
        if not pdf.exists():
            continue
        articles = parse(pdf, src)
        chunks.extend(
            chunk_articles(
                articles,
                source_slug=src.slug,
                count_tokens=count_tokens,
                max_tokens=_MAX_TOKENS,
            )
        )
    return chunks


@dataclass(frozen=True, slots=True)
class _Pipeline:
    """Bundle the wrapped retriever and generator the test queries against."""

    retriever: RerankRetriever
    generator: Generator


@pytest.fixture(scope="module")
def pipeline() -> _Pipeline:
    """Build MultiQuery -> Hybrid -> Rerank -> Generator against the real corpus + LLM.

    Module-scoped so the cross-encoder cold-start, e5-large model load, and
    Ollama daemon warmup are paid once. A trivial LLM warmup call forces a
    clean skip when the daemon is unreachable or ``llama3.1:8b`` isn't pulled
    locally, rather than failing inside the first test.
    """
    _skip_unless_corpus_and_model_available()

    llm = config.get_llm()
    try:
        llm.generate("Reply with the single word: ready", max_output_tokens=8)
    except LLMUnavailableError as exc:
        pytest.skip(f"Ollama daemon / model unavailable: {exc}")
    except ConnectionError as exc:
        # ollama>=0.6 raises a builtin ConnectionError (instead of httpx.ConnectError)
        # when the daemon isn't reachable; the local adapter doesn't yet wrap it.
        # Skipping cleanly here matches the spec's "auto-skip when Ollama daemon
        # unreachable" intent without requiring an adapter change in this slice.
        pytest.skip(f"Ollama daemon unreachable: {exc}")

    embedder = config.get_embeddings()
    index = config.get_vector_index(persist_dir=_CHROMA_DIR, embedder=embedder)
    chunks = _reparse_chunks(embedder.count_tokens)
    if not chunks:
        pytest.skip("re-parse produced no chunks; raw PDFs may be incomplete")

    hybrid = config.get_retriever(chunks=chunks, embedder=embedder, vector_index=index)
    multi = MultiQueryRetriever(retriever=hybrid, llm=llm, n_variations=3)
    reranker = config.get_reranker()
    reranked = RerankRetriever(retriever=multi, reranker=reranker, candidate_top_k=20)
    # Force the cross-encoder model load now rather than on the first assertion.
    reranked.retrieve("warmup", top_k=1)

    return _Pipeline(retriever=reranked, generator=Generator(llm=llm))


@pytest.mark.parametrize("query", _CANONICAL_QUERIES, ids=lambda q: q.label)
def test_canonical_query_produces_grounded_cited_answer(
    pipeline: _Pipeline,
    query: _CanonicalQuery,
) -> None:
    """Each canonical question returns a bounded, marker-bearing, gold-citing answer."""
    hits = pipeline.retriever.retrieve(query.question, top_k=5)
    assert hits, f"no hits retrieved for {query.label!r}"

    payload = pipeline.generator.generate(query.question, hits)

    assert isinstance(payload, AnswerPayload)
    assert payload.answer.strip(), f"empty answer for {query.label!r}"
    assert len(payload.answer) <= _ANSWER_MAX_CHARS, (
        f"answer exceeds {_ANSWER_MAX_CHARS} chars for {query.label!r}: {len(payload.answer)} chars"
    )
    assert _MARKER_RE.search(payload.answer), (
        f"no [N] citation marker in answer for {query.label!r}: {payload.answer!r}"
    )
    assert payload.citations, f"no citations rendered for {query.label!r}"
    assert payload.citations[0].article == query.expected_article, (
        f"citations[0].article={payload.citations[0].article!r} "
        f"does not match expected {query.expected_article!r} for {query.label!r}"
    )


def test_pipeline_arabic_question_uses_arabic_template(pipeline: _Pipeline) -> None:
    """An Arabic question routes through the AR prompt template (per ADR-0008)."""
    question = "ما هي مدة الإجازة السنوية؟"
    hits = pipeline.retriever.retrieve(question, top_k=5)

    payload = pipeline.generator.generate(question, hits)

    assert payload.language == "ar"
    assert "المقاطع" in payload.prompt_used  # AR template marker block heading
