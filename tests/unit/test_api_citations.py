"""Unit tests for ``api.main.to_api_response`` — domain → wire Citation mapping.

The mapper dedups on ``(source, article)`` so two chunks from the same article
collapse to one wire citation, preserving the marker assigned by
``render_citations``. Answer text and detected language pass through unchanged.
"""

from __future__ import annotations

from uae_rag.api.main import Citation, QueryResponse, to_api_response
from uae_rag.generation.answer import AnswerPayload
from uae_rag.generation.citations import Citation as DomainCitation


def _domain_citation(
    marker: str,
    source: str,
    article: str,
    chunk_id: str = "labour-law-en::art-29",
    language: str = "en",
) -> DomainCitation:
    return DomainCitation(
        marker=marker,
        source=source,
        article=article,
        chunk_id=chunk_id,
        language=language,
    )


def _payload(citations: list[DomainCitation], *, answer: str = "ok", language: str = "en") -> AnswerPayload:
    return AnswerPayload(
        answer=answer,
        citations=citations,
        language=language,  # type: ignore[arg-type]
        prompt_used="",
    )


def test_dedups_on_source_and_article() -> None:
    citations = [
        _domain_citation("[1]", "labour-law-en", "29", chunk_id="labour-law-en::art-29"),
        _domain_citation("[1]", "labour-law-en", "29", chunk_id="labour-law-en::art-29#p1-p1"),
        _domain_citation("[2]", "mohre-resolutions", "18"),
        _domain_citation("[3]", "mohre-resolutions", "19"),
        _domain_citation("[3]", "mohre-resolutions", "19", chunk_id="mohre-resolutions::art-19#p1"),
    ]

    response = to_api_response(_payload(citations))

    assert isinstance(response, QueryResponse)
    assert len(response.citations) == 3
    keys = [(c.source, c.article) for c in response.citations]
    assert keys == [
        ("labour-law-en", "29"),
        ("mohre-resolutions", "18"),
        ("mohre-resolutions", "19"),
    ]


def test_preserves_first_marker_per_pair() -> None:
    citations = [
        _domain_citation("[1]", "labour-law-en", "29"),
        _domain_citation("[1]", "labour-law-en", "29"),
        _domain_citation("[2]", "mohre-resolutions", "18"),
    ]

    response = to_api_response(_payload(citations))

    assert [c.marker for c in response.citations] == ["[1]", "[2]"]
    assert isinstance(response.citations[0], Citation)


def test_passes_through_answer_and_language() -> None:
    payload = _payload(
        [_domain_citation("[1]", "labour-law-ar", "29", language="ar")],
        answer="ب. يومان عن كل شهر [1].",
        language="ar",
    )

    response = to_api_response(payload)

    assert response.answer == "ب. يومان عن كل شهر [1]."
    assert response.language == "ar"
    assert response.citations[0].marker == "[1]"
    assert response.citations[0].source == "labour-law-ar"
    assert response.citations[0].article == "29"


def test_empty_citations_yields_empty_list() -> None:
    response = to_api_response(_payload([], answer="refusal", language="en"))

    assert response.citations == []
    assert response.answer == "refusal"
    assert response.language == "en"
