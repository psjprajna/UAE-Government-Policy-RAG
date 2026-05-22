"""Unit tests for the ``generation`` package — language, citations, prompts, Generator.

Self-contained. Reuses the ``FakeLLM`` defined in ``test_llm_port.py`` (same
pattern Phase 5 used for ``FakeReranker`` re-imported in ``test_rerank.py``).
"""

from __future__ import annotations

import pytest

from tests.unit.test_llm_port import FakeLLM
from uae_rag.generation.answer import AnswerPayload, Generator
from uae_rag.generation.citations import Citation, render_citations
from uae_rag.generation.language import detect_language
from uae_rag.generation.prompts import (
    PROMPT_TEMPLATE_AR,
    PROMPT_TEMPLATE_EN,
    REFUSAL_AR,
    REFUSAL_EN,
    render_prompt,
)
from uae_rag.ports import RetrievalHit


def _hit(
    chunk_id: str,
    text: str,
    *,
    source_slug: str = "labour-law-en",
    article_id: str | None = "29",
    breadcrumb: str = "UAE Labour Law > Article 29",
    language: str = "en",
    rank: int = 1,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        text=text,
        metadata={
            "source_slug": source_slug,
            "article_id": article_id,
            "breadcrumb": breadcrumb,
            "language": language,
        },
        score=0.1,
        rank=rank,
        source="reranked",
    )


# --- detect_language -------------------------------------------------------------


def test_detect_language_english() -> None:
    assert detect_language("What is the annual leave entitlement?") == "en"


def test_detect_language_arabic() -> None:
    assert detect_language("ما هي مدة الإجازة السنوية؟") == "ar"


def test_detect_language_empty_falls_back_to_en() -> None:
    assert detect_language("") == "en"


def test_detect_language_whitespace_only_falls_back_to_en() -> None:
    assert detect_language("   \n\t") == "en"


def test_detect_language_digits_only_falls_back_to_en() -> None:
    """Lingua returns ``None`` for digit-only input; we default to English."""
    assert detect_language("12345") == "en"


def test_detect_language_mixed_picks_dominant() -> None:
    """Question with majority-EN tokens but embedded AR phrase → English."""
    assert detect_language("What does المادة 29 say about leave?") == "en"


def test_detect_language_returns_only_en_or_ar() -> None:
    """The detector is restricted to EN+AR; nothing else escapes."""
    for sample in ("Bonjour le monde", "Hola mundo", "你好"):
        assert detect_language(sample) in {"en", "ar"}


# --- Citation + render_citations -------------------------------------------------


def test_citation_is_frozen_and_slotted() -> None:
    c = Citation(
        marker="[1]",
        source="labour-law-en",
        article="29",
        chunk_id="labour-law-en::art-29",
        language="en",
    )
    with pytest.raises((AttributeError, TypeError)):
        c.marker = "[2]"  # type: ignore[misc]


def test_render_citations_assigns_one_based_markers() -> None:
    hits = [
        _hit("labour-law-en::art-29", "annual leave...", article_id="29"),
        _hit(
            "labour-law-en::art-35",
            "probation...",
            article_id="35",
            breadcrumb="UAE Labour Law > Article 35",
        ),
    ]

    citations = render_citations(hits)

    assert [c.marker for c in citations] == ["[1]", "[2]"]


def test_render_citations_dedupes_on_source_and_article() -> None:
    """Two chunks from the same article collapse to a single marker."""
    hits = [
        _hit("labour-law-en::art-29", "p1...", article_id="29"),
        _hit("labour-law-en::art-29#p2-p2", "p2...", article_id="29"),
        _hit(
            "labour-law-en::art-35",
            "probation...",
            article_id="35",
            breadcrumb="UAE Labour Law > Article 35",
        ),
    ]

    citations = render_citations(hits)

    assert [c.marker for c in citations] == ["[1]", "[1]", "[2]"]
    assert citations[0].article == "29"
    assert citations[2].article == "35"


def test_render_citations_preserves_source_chunk_id_language() -> None:
    hits = [
        _hit(
            "labour-law-ar::art-29",
            "الإجازة...",
            source_slug="labour-law-ar",
            article_id="29",
            breadcrumb="قانون العمل > المادة 29",
            language="ar",
        ),
    ]

    citations = render_citations(hits)

    assert citations[0].source == "labour-law-ar"
    assert citations[0].chunk_id == "labour-law-ar::art-29"
    assert citations[0].language == "ar"
    assert citations[0].article == "29"


def test_render_citations_fallback_chunk_uses_breadcrumb_tail() -> None:
    """A chunk whose ``article_id`` is None falls back to the breadcrumb tail."""
    hits = [
        _hit(
            "labour-law-en::sec-1",
            "definitions...",
            article_id=None,
            breadcrumb="UAE Labour Law > Chapter 1 > Definitions",
        ),
    ]

    citations = render_citations(hits)

    assert citations[0].article == "Definitions"


def test_render_citations_truncates_long_breadcrumb_tail() -> None:
    long_tail = "x" * 200
    hits = [
        _hit(
            "labour-law-en::sec-1",
            "long-definitions...",
            article_id=None,
            breadcrumb=f"UAE Labour Law > {long_tail}",
        ),
    ]

    citations = render_citations(hits)

    assert len(citations[0].article) <= 80


def test_render_citations_empty_input_returns_empty() -> None:
    assert render_citations([]) == []


# --- Prompts ---------------------------------------------------------------------


def test_render_prompt_en_includes_question_and_passages() -> None:
    hits = [_hit("labour-law-en::art-29", "Every worker is entitled to thirty days...")]
    citations = render_citations(hits)

    prompt = render_prompt("en", question="What is annual leave?", hits=hits, citations=citations)

    assert "What is annual leave?" in prompt
    assert "Every worker is entitled to thirty days" in prompt
    assert "[1]" in prompt
    assert "cite every claim" in prompt.lower()


def test_render_prompt_ar_includes_question_and_passages() -> None:
    hits = [
        _hit(
            "labour-law-ar::art-29",
            "الإجازة السنوية ثلاثون يوماً",
            source_slug="labour-law-ar",
            breadcrumb="قانون العمل > المادة 29",
            language="ar",
        )
    ]
    citations = render_citations(hits)

    prompt = render_prompt(
        "ar", question="ما هي مدة الإجازة السنوية؟", hits=hits, citations=citations
    )

    assert "ما هي مدة الإجازة السنوية؟" in prompt
    assert "الإجازة السنوية ثلاثون يوماً" in prompt
    assert "[1]" in prompt


def test_refusal_strings_are_distinct_per_language() -> None:
    assert REFUSAL_EN != REFUSAL_AR
    assert "I don't have enough information" in REFUSAL_EN
    assert "لا تتوفر" in REFUSAL_AR


def test_prompt_templates_carry_refusal_instructions() -> None:
    """Each template tells the model to use the language-specific refusal verbatim."""
    assert REFUSAL_EN in PROMPT_TEMPLATE_EN
    assert REFUSAL_AR in PROMPT_TEMPLATE_AR


def test_render_prompt_uses_dedup_markers() -> None:
    """When two chunks share an article, the passage block must reflect the dedup."""
    hits = [
        _hit("labour-law-en::art-29", "passage one", article_id="29"),
        _hit("labour-law-en::art-29#p2-p2", "passage two", article_id="29"),
    ]
    citations = render_citations(hits)

    prompt = render_prompt("en", question="q", hits=hits, citations=citations)

    # Both passages appear, but they share marker [1] in the passage block.
    assert prompt.count("[1]") >= 3  # template instruction + 2 passages
    assert "[2]" not in prompt


# --- Generator -------------------------------------------------------------------


def test_generator_satisfies_happy_path() -> None:
    """Default temperature=0.0, EN question → EN prompt → LLM response surfaces verbatim."""
    fake = FakeLLM(canned_response="Workers get 30 days of leave [1].")
    gen = Generator(llm=fake)
    hits = [_hit("labour-law-en::art-29", "Every worker is entitled to thirty days...")]

    payload = gen.generate("What is the annual leave entitlement?", hits)

    assert isinstance(payload, AnswerPayload)
    assert payload.answer == "Workers get 30 days of leave [1]."
    assert payload.language == "en"
    assert len(payload.citations) == 1
    assert payload.citations[0].marker == "[1]"
    assert payload.citations[0].article == "29"
    assert "What is the annual leave entitlement?" in payload.prompt_used
    assert fake.last_temperature == 0.0


def test_generator_uses_arabic_template_for_arabic_question() -> None:
    fake = FakeLLM(canned_response="ثلاثون يوماً [1]")
    gen = Generator(llm=fake)
    hits = [
        _hit(
            "labour-law-ar::art-29",
            "الإجازة السنوية ثلاثون يوماً",
            source_slug="labour-law-ar",
            breadcrumb="قانون العمل > المادة 29",
            language="ar",
        )
    ]

    payload = gen.generate("ما هي مدة الإجازة السنوية؟", hits)

    assert payload.language == "ar"
    assert "المقاطع" in payload.prompt_used  # AR template marker


def test_generator_language_override_wins_over_detection() -> None:
    """Explicit ``language`` parameter overrides ``detect_language``."""
    fake = FakeLLM(canned_response="x")
    gen = Generator(llm=fake)
    hits = [_hit("labour-law-en::art-29", "...")]

    payload = gen.generate("English question", hits, language="ar")

    assert payload.language == "ar"
    assert "المقاطع" in payload.prompt_used


def test_generator_empty_hits_returns_refusal_without_llm_call() -> None:
    fake = FakeLLM(canned_response="should-not-be-used")
    gen = Generator(llm=fake)

    payload = gen.generate("What is annual leave?", [])

    assert payload.answer == REFUSAL_EN
    assert payload.citations == []
    assert payload.prompt_used == ""
    assert fake.call_count == 0


def test_generator_empty_hits_arabic_returns_arabic_refusal() -> None:
    fake = FakeLLM(canned_response="x")
    gen = Generator(llm=fake)

    payload = gen.generate("ما هي مدة الإجازة السنوية؟", [])

    assert payload.answer == REFUSAL_AR
    assert payload.language == "ar"
    assert fake.call_count == 0


def test_generator_question_validation_empty() -> None:
    gen = Generator(llm=FakeLLM(canned_response="x"))
    with pytest.raises(ValueError, match="question"):
        gen.generate("", [_hit("c", "x")])
    with pytest.raises(ValueError, match="question"):
        gen.generate("   ", [_hit("c", "x")])


def test_generator_question_validation_too_long() -> None:
    gen = Generator(llm=FakeLLM(canned_response="x"))
    with pytest.raises(ValueError, match="2000"):
        gen.generate("x" * 2001, [_hit("c", "x")])


def test_generator_forwards_max_output_tokens_and_temperature() -> None:
    fake = FakeLLM(canned_response="x")
    gen = Generator(llm=fake, max_output_tokens=256, temperature=0.3)

    gen.generate("What is annual leave?", [_hit("labour-law-en::art-29", "...")])

    assert fake.last_max_output_tokens == 256
    assert fake.last_temperature == 0.3


def test_generator_llm_exceptions_propagate() -> None:
    """The generator does not catch transport errors — caller decides."""

    def boom(_prompt: str) -> str:
        raise RuntimeError("transport down")

    gen = Generator(llm=FakeLLM(callable_=boom))

    with pytest.raises(RuntimeError, match="transport down"):
        gen.generate("What is annual leave?", [_hit("labour-law-en::art-29", "...")])


def test_generator_answer_payload_is_frozen() -> None:
    payload = AnswerPayload(
        answer="ok",
        citations=[],
        language="en",
        prompt_used="",
    )
    with pytest.raises((AttributeError, TypeError)):
        payload.answer = "changed"  # type: ignore[misc]
