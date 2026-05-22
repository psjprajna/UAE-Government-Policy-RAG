"""Generator — composes an ``LLMPort`` with language detection, citations, and prompts.

The generator is the single domain object Phase 7's ``/query`` will consume
(after retrieve + rerank). It accepts a question + retrieved hits, detects
the question's language (English / Arabic), renders citations and a
language-appropriate prompt, calls the LLM, and packages the result.

Empty ``hits`` short-circuits to the language-specific refusal phrase without
contacting the LLM — keeping ``/query`` honest about "no evidence found"
and avoiding a guess from the model.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from uae_rag.generation.citations import Citation, render_citations
from uae_rag.generation.language import detect_language
from uae_rag.generation.prompts import REFUSAL_AR, REFUSAL_EN, render_prompt
from uae_rag.ports.llm import LLMPort
from uae_rag.ports.retrieval import RetrievalHit

logger = logging.getLogger(__name__)

_MAX_QUESTION_CHARS = 2000


@dataclass(frozen=True, slots=True)
class AnswerPayload:
    """End-to-end generator output: answer text + citations + detected language + prompt."""

    answer: str
    citations: list[Citation]
    language: Literal["en", "ar"]
    prompt_used: str


class Generator:
    """Orchestrates language detection, citation rendering, prompt assembly, and the LLM call."""

    def __init__(
        self,
        *,
        llm: LLMPort,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self._llm = llm
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature

    def generate(
        self,
        question: str,
        hits: Sequence[RetrievalHit],
        *,
        language: Literal["en", "ar"] | None = None,
    ) -> AnswerPayload:
        """Return an ``AnswerPayload`` for ``question`` grounded in ``hits``.

        ``language`` explicitly overrides ``detect_language(question)``. Empty
        ``hits`` short-circuits to the language-specific refusal phrase
        without invoking the LLM.
        """
        if not question.strip():
            raise ValueError("question must be non-empty")
        if len(question) > _MAX_QUESTION_CHARS:
            raise ValueError(f"question exceeds {_MAX_QUESTION_CHARS} chars")

        resolved_language = language or detect_language(question)

        if not hits:
            logger.debug("empty hits — returning refusal in %s", resolved_language)
            return AnswerPayload(
                answer=REFUSAL_AR if resolved_language == "ar" else REFUSAL_EN,
                citations=[],
                language=resolved_language,
                prompt_used="",
            )

        citations = render_citations(hits)
        prompt = render_prompt(
            resolved_language,
            question=question,
            hits=hits,
            citations=citations,
        )
        answer = self._llm.generate(
            prompt,
            max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
        )

        return AnswerPayload(
            answer=answer,
            citations=citations,
            language=resolved_language,
            prompt_used=prompt,
        )


__all__ = ["AnswerPayload", "Generator"]
