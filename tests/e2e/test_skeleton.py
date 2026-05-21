"""Walking-skeleton smoke test.

Verifies the FastAPI /query endpoint exists, accepts a question, and returns
a well-formed response with a citations array. This is the proof that the
architecture wires up end-to-end before any real RAG components are added.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from uae_rag.api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_answer_with_at_least_one_citation(client: TestClient) -> None:
    response = client.post("/query", json={"question": "What is the annual leave entitlement?"})

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["answer"], str) and len(body["answer"]) > 0
    assert isinstance(body["citations"], list) and len(body["citations"]) >= 1
    first = body["citations"][0]
    assert "source" in first and "article" in first
    assert body["language"] in {"en", "ar"}


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
