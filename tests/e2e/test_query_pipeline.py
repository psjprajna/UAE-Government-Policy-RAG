"""Live-lane end-to-end test for ``POST /query``.

Runs the real lifespan (parses PDFs, opens ChromaDB, composes the
``MultiQuery → Hybrid → Rerank → Generator`` stack) and exercises three
canonical questions against the local Ollama daemon. Marked ``adapter_local``
and ``slow``; auto-skipped when the corpus, the SDKs, or the daemon are
unavailable.

Acceptance criterion (per Phase 7 spec): every marker present in
``response.citations`` is a valid ``[N]`` bracketed integer, and the answer
contains at least one ``[N]`` marker. The reverse — every in-answer marker
resolving to a citation — is *not* asserted (the LLM is permitted to
hallucinate a marker; strict enforcement is deferred to Phase 8).
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from uae_rag import config
from uae_rag.api.main import app
from uae_rag.ports import LLMUnavailableError

pytestmark = [pytest.mark.adapter_local, pytest.mark.slow]

_APP_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _APP_ROOT / "data"
_CHROMA_DIR = _DATA_DIR / "chroma_db"
_RAW_DIR = _DATA_DIR / "raw"
_MARKER_RE = re.compile(r"\[\d+\]")


@dataclass(frozen=True, slots=True)
class _CanonicalQuery:
    question: str
    expected_article: str
    expected_language: str
    label: str


_CANONICAL_QUERIES = (
    _CanonicalQuery(
        question="What is the annual leave entitlement?",
        expected_article="29",
        expected_language="en",
        label="en-annual-leave",
    ),
    _CanonicalQuery(
        question="How long is the probationary period?",
        expected_article="9",
        expected_language="en",
        label="en-probation",
    ),
    _CanonicalQuery(
        question="ما هي مدة الإجازة السنوية؟",
        expected_article="29",
        expected_language="ar",
        label="ar-annual-leave",
    ),
)


def _skip_unless_environment_ready() -> None:
    if not _CHROMA_DIR.exists():
        pytest.skip(f"corpus index missing: {_CHROMA_DIR}")
    if not _RAW_DIR.exists():
        pytest.skip(f"raw PDFs missing: {_RAW_DIR}")
    if importlib.util.find_spec("chromadb") is None:
        pytest.skip("chromadb not installed")
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence_transformers not installed")
    try:
        config.get_llm().generate("Reply with the single word: ready", max_output_tokens=8)
    except LLMUnavailableError as exc:
        pytest.skip(f"Ollama daemon / model unavailable: {exc}")


@pytest.fixture(scope="module")
def live_client() -> TestClient:
    """Module-scoped TestClient running under the real lifespan + LLM warmup."""
    _skip_unless_environment_ready()
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("query", _CANONICAL_QUERIES, ids=lambda q: q.label)
def test_query_returns_grounded_cited_answer(
    live_client: TestClient,
    query: _CanonicalQuery,
) -> None:
    response = live_client.post("/query", json={"question": query.question})

    assert response.status_code == 200, response.text
    body = response.json()

    assert isinstance(body["answer"], str) and body["answer"].strip()
    assert _MARKER_RE.search(body["answer"]), (
        f"no [N] marker in answer for {query.label!r}: {body['answer']!r}"
    )
    assert body["language"] == query.expected_language

    citations = body["citations"]
    assert isinstance(citations, list) and citations, (
        f"no citations returned for {query.label!r}"
    )
    for citation in citations:
        assert re.fullmatch(r"\[\d+\]", citation["marker"]), citation
    articles = {c["article"] for c in citations}
    assert query.expected_article in articles, (
        f"expected article {query.expected_article!r} missing from citations "
        f"{articles!r} for {query.label!r}"
    )
