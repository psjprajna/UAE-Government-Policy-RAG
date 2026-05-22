"""Fast-lane API smoke tests.

Validation, health, and OpenAPI assertions stay on a bare ``TestClient`` so
they don't reach the pipeline. The ``/query`` handler tests share one
module-scoped ``TestClient`` running under the real lifespan (cheap because
every adapter defers weight loads via ``@cached_property``) and swap a
``_FakePipeline`` in via ``app.dependency_overrides`` per test — never
touching Ollama or any model weights.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from uae_rag.api.main import Pipeline, app, get_pipeline
from uae_rag.generation.answer import AnswerPayload
from uae_rag.generation.citations import Citation as DomainCitation
from uae_rag.ports import LLMUnavailableError
from uae_rag.ports.retrieval import RetrievalHit


class _FakeRetriever:
    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                chunk_id="labour-law-en::art-29",
                text="Article 29 — Annual leave.",
                metadata={"source_slug": "labour-law-en", "article_id": "29", "language": "en"},
                score=0.9,
                rank=1,
                source="reranked",
            )
        ]


class _FakeGenerator:
    def __init__(self, *, raises: BaseException | None = None) -> None:
        self._raises = raises

    def generate(
        self,
        question: str,
        hits: Sequence[RetrievalHit],
        *,
        language: Literal["en", "ar"] | None = None,
    ) -> AnswerPayload:
        if self._raises is not None:
            raise self._raises
        citation = DomainCitation(
            marker="[1]",
            source="labour-law-en",
            article="29",
            chunk_id="labour-law-en::art-29",
            language="en",
        )
        return AnswerPayload(
            answer="Thirty days per year of extended service [1].",
            citations=[citation],
            language="en",
            prompt_used="<test prompt>",
        )


def _fake_pipeline(*, raises: BaseException | None = None) -> Pipeline:
    return Pipeline(retriever=_FakeRetriever(), generator=_FakeGenerator(raises=raises))


@pytest.fixture
def client() -> TestClient:
    """Bare client — does not enter lifespan; sufficient for non-/query tests."""
    return TestClient(app)


@pytest.fixture(scope="module")
def pipeline_client() -> Iterator[TestClient]:
    """Module-scoped client under the real (side-effect-light) lifespan.

    Lifespan parses PDFs and builds the BM25 index once for the whole module;
    individual tests swap the pipeline via ``dependency_overrides`` and never
    invoke the real retriever or LLM.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def fake_pipeline_client(pipeline_client: TestClient) -> Iterator[TestClient]:
    app.dependency_overrides[get_pipeline] = lambda: _fake_pipeline()
    try:
        yield pipeline_client
    finally:
        app.dependency_overrides.pop(get_pipeline, None)


@pytest.fixture
def llm_unavailable_client(pipeline_client: TestClient) -> Iterator[TestClient]:
    app.dependency_overrides[get_pipeline] = lambda: _fake_pipeline(
        raises=LLMUnavailableError("daemon down")
    )
    try:
        yield pipeline_client
    finally:
        app.dependency_overrides.pop(get_pipeline, None)


@pytest.fixture
def value_error_client(pipeline_client: TestClient) -> Iterator[TestClient]:
    app.dependency_overrides[get_pipeline] = lambda: _fake_pipeline(
        raises=ValueError("question must be non-empty")
    )
    try:
        yield pipeline_client
    finally:
        app.dependency_overrides.pop(get_pipeline, None)


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_answer_with_at_least_one_citation(
    fake_pipeline_client: TestClient,
) -> None:
    response = fake_pipeline_client.post(
        "/query", json={"question": "What is the annual leave entitlement?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["answer"], str) and len(body["answer"]) > 0
    assert isinstance(body["citations"], list) and len(body["citations"]) >= 1
    first = body["citations"][0]
    assert {"marker", "source", "article"} <= set(first.keys())
    assert re.fullmatch(r"\[\d+\]", first["marker"])
    assert body["language"] in {"en", "ar"}


def test_query_returns_503_on_llm_unavailable(
    llm_unavailable_client: TestClient,
) -> None:
    response = llm_unavailable_client.post(
        "/query", json={"question": "What is the annual leave entitlement?"}
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "LLM unavailable"}


def test_query_returns_400_on_generator_value_error(
    value_error_client: TestClient,
) -> None:
    response = value_error_client.post(
        "/query", json={"question": "What is the annual leave entitlement?"}
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "question must be non-empty"}


def test_query_rejects_empty_question(client: TestClient) -> None:
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422  # Pydantic min_length=1


def test_query_rejects_oversized_question(client: TestClient) -> None:
    response = client.post("/query", json={"question": "a" * 2001})
    assert response.status_code == 422  # Pydantic max_length=2000


def test_openapi_schema_is_served(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/query" in schema["paths"]
