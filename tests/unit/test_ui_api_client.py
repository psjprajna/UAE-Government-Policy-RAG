"""Unit tests for ``ui/api_client.py`` — the Streamlit UI's view of the wire.

Uses ``httpx.MockTransport`` (no new dep) to inject canned responses, so the
tests run in the fast lane without binding sockets or touching the FastAPI
app. Each scenario asserts one concept; the request shape, the happy path,
each HTTP error class, the network-failure mapping, the refusal-case parse,
and env-var resolution of ``UAE_RAG_API_BASE``.
"""

from __future__ import annotations

import json

import httpx
import pytest
from api_client import ApiClient, ApiError, CitationView, QueryResult


def _ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "answer": "Workers get 30 days [1].",
            "citations": [
                {"marker": "[1]", "source": "labour-law-en", "article": "29"},
                {"marker": "[2]", "source": "mohre-resolutions", "article": "18"},
            ],
            "language": "en",
        },
    )


def _make_client(handler: httpx.MockTransport | object) -> ApiClient:
    transport = (
        handler if isinstance(handler, httpx.MockTransport) else httpx.MockTransport(handler)
    )
    return ApiClient(base_url="http://test", timeout=5.0, transport=transport)


def test_query_returns_parsed_query_result_on_200() -> None:
    client = _make_client(_ok_handler)
    result = client.query("What is the annual leave entitlement?")
    assert isinstance(result, QueryResult)
    assert result.answer == "Workers get 30 days [1]."
    assert result.language == "en"
    assert result.citations == [
        CitationView(marker="[1]", source="labour-law-en", article="29"),
        CitationView(marker="[2]", source="mohre-resolutions", article="18"),
    ]


def test_query_posts_question_to_query_path() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["body"] = request.content.decode()
        return _ok_handler(request)

    client = _make_client(handler)
    client.query("What is the annual leave entitlement?")
    assert seen["method"] == "POST"
    assert seen["url"] == "http://test/query"
    assert json.loads(seen["body"]) == {"question": "What is the annual leave entitlement?"}  # type: ignore[arg-type]


def test_query_raises_api_error_on_400() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "question must not be empty"})

    client = _make_client(handler)
    with pytest.raises(ApiError) as exc_info:
        client.query("   ")
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "question must not be empty"


def test_query_raises_api_error_on_503_llm_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "LLM unavailable"})

    client = _make_client(handler)
    with pytest.raises(ApiError) as exc_info:
        client.query("What is the annual leave entitlement?")
    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "LLM unavailable"


def test_query_maps_network_error_to_api_error_503() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _make_client(handler)
    with pytest.raises(ApiError) as exc_info:
        client.query("What is the annual leave entitlement?")
    assert exc_info.value.status_code == 503
    assert "cannot reach API" in exc_info.value.detail


def test_query_parses_refusal_case_empty_citations() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answer": "I don't have enough information in the cited sources to answer that.",
                "citations": [],
                "language": "en",
            },
        )

    client = _make_client(handler)
    result = client.query("What is the meaning of life?")
    assert result.citations == []
    assert result.language == "en"
    assert "don't have enough information" in result.answer


def test_default_base_url_falls_back_to_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UAE_RAG_API_BASE", raising=False)
    client = ApiClient()
    assert client.base_url == "http://127.0.0.1:8000"


def test_default_base_url_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UAE_RAG_API_BASE", "http://custom-host:9000")
    client = ApiClient()
    assert client.base_url == "http://custom-host:9000"
