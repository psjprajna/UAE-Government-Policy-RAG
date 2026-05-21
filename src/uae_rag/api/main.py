from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="UAE Government Policy RAG",
    version="0.0.1",
    description="Retrieval-augmented QA over UAE Labour Law and related regulations.",
)


class Citation(BaseModel):
    source: str = Field(..., description="Document name, e.g. 'UAE Labour Law'")
    article: str = Field(..., description="Article reference, e.g. 'Article 29'")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    language: str = Field(..., description="Detected language: 'en' or 'ar'")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Walking skeleton: returns a deterministic stubbed answer with one citation.

    Real pipeline arrives in Phase 7. Until then, this endpoint exists to prove
    the architecture wires up end-to-end and the response shape is stable.
    """
    return QueryResponse(
        answer=(
            "This is a walking-skeleton response. The retrieval and generation "
            "pipeline is added in later phases. Your question was: "
            f"'{request.question}'."
        ),
        citations=[
            Citation(
                source="UAE Labour Law (Federal Law No. 33 of 2021)",
                article="Article 29 — Annual leave (placeholder)",
            )
        ],
        language="en",
    )
