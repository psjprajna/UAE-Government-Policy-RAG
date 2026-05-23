"""HTTP client for the ``POST /query`` endpoint exposed by ``uae_rag.api``.

Lives outside ``src/uae_rag/`` — this file is the UI's local view of the wire
contract, not a package module. Field shapes mirror
``src/uae_rag/api/main.py:50-63`` verbatim; we intentionally do **not** import
the Pydantic models so the UI stays decoupled from the package.

Errors collapse to a single ``ApiError`` carrying ``status_code`` + ``detail``:
upstream 4xx/5xx surface verbatim, and network failures (DNS, refused, timeout)
map to ``ApiError(503, …)`` so the Streamlit caller has one path to render.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:8000"
# First /query after a fresh server boot pays the model-load tax (~190 s in
# practice — see app/README.md). Default well past that to avoid spurious
# timeouts on the first interaction; subsequent calls are sub-10s.
_DEFAULT_TIMEOUT_S = 240.0


def _base_url_from_env() -> str:
    return os.environ.get("UAE_RAG_API_BASE", _DEFAULT_BASE_URL)


@dataclass(frozen=True, slots=True)
class CitationView:
    """Wire ``Citation`` — see ``uae_rag.api.main.Citation`` (lines 50-53)."""

    marker: str
    source: str
    article: str


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Wire ``QueryResponse`` — see ``uae_rag.api.main.QueryResponse`` (lines 60-63)."""

    answer: str
    citations: list[CitationView]
    language: str


class ApiError(Exception):
    """Raised when ``/query`` returns non-200 or the API is unreachable."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ApiClient:
    base_url: str = field(default_factory=_base_url_from_env)
    timeout: float = _DEFAULT_TIMEOUT_S
    transport: httpx.BaseTransport | None = None  # test injection seam

    def query(self, question: str) -> QueryResult:
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                response = client.post("/query", json={"question": question})
        except httpx.RequestError as exc:
            logger.warning("API request failed: %s", exc)
            raise ApiError(
                503, f"cannot reach API at {self.base_url} ({exc.__class__.__name__})"
            ) from exc

        if response.status_code != 200:
            raise ApiError(response.status_code, _extract_detail(response))

        data = response.json()
        return QueryResult(
            answer=data["answer"],
            citations=[CitationView(**c) for c in data["citations"]],
            language=data["language"],
        )


def _extract_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text or response.reason_phrase
    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])
    return response.text or response.reason_phrase
