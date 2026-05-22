"""Prompt templates — language-specific scaffolds with citation-marker instructions.

Per ADR-0008, the templates instruct the model to cite every claim by its
1-based bracketed marker (``[1]``, ``[2]``, …) and to refuse with a verbatim
language-specific phrase when no passage supports the answer. Both refusal
phrases also serve as the generator's empty-hits short-circuit response.

The passage block is rendered as ``[N] (<breadcrumb>) <text>`` and joined by
blank lines. Markers honor the dedup performed in ``render_citations`` — two
chunks from the same article share a marker.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from uae_rag.generation.citations import Citation
from uae_rag.ports.retrieval import RetrievalHit

REFUSAL_EN = "I don't have enough information in the cited sources to answer that."
REFUSAL_AR = "لا تتوفر لدي معلومات كافية في المصادر المستشهد بها للإجابة عن هذا السؤال."


PROMPT_TEMPLATE_EN = f"""\
You are a careful assistant answering questions about UAE government policy.

Rules:
1. Use ONLY the passages below as evidence. Do not rely on outside knowledge.
2. Cite every claim by the bracketed marker next to its passage (e.g. "[1]").
3. Do NOT write "(Article N)" or put article numbers in parentheses — use only the bracketed markers.
4. Every sentence must end with at least one bracketed marker.
5. If the passages do not contain the answer, reply exactly: "{REFUSAL_EN}"

Passages:
{{passages}}

Question: {{question}}

Answer:"""


PROMPT_TEMPLATE_AR = f"""\
أنت مساعد دقيق يجيب على أسئلة حول السياسات الحكومية الإماراتية.

القواعد:
1. استخدم المقاطع أدناه فقط كمصدر للأدلة. لا تعتمد على معرفة خارجية.
2. استشهد بكل ادعاء بالرمز المعقوف المقابل للمقطع المستخدم (مثال: "[1]").
3. لا تستشهد بكتابة "(المادة N)" أو وضع أرقام المواد بين قوسين — استخدم فقط الرموز المعقوفة.
4. يجب أن تنتهي كل جملة برمز معقوف واحد على الأقل.
5. إذا لم تكن المقاطع تحتوي على الإجابة، فقم بالرد حرفيًا: "{REFUSAL_AR}"

المقاطع:
{{passages}}

السؤال: {{question}}

الإجابة:"""


def render_prompt(
    language: Literal["en", "ar"],
    *,
    question: str,
    hits: Sequence[RetrievalHit],
    citations: Sequence[Citation],
) -> str:
    """Assemble the final prompt string for ``language`` from ``hits`` and ``citations``.

    ``citations`` must align positionally with ``hits``; the same marker is
    reused when two hits share an article (per ``render_citations``).
    """
    if len(citations) != len(hits):
        raise ValueError("citations and hits must have the same length")

    template = PROMPT_TEMPLATE_AR if language == "ar" else PROMPT_TEMPLATE_EN
    passages = _format_passages(hits, citations)
    return template.format(passages=passages, question=question)


def _format_passages(hits: Sequence[RetrievalHit], citations: Sequence[Citation]) -> str:
    blocks: list[str] = []
    for hit, citation in zip(hits, citations, strict=True):
        breadcrumb = str(hit.metadata.get("breadcrumb", "")).strip()
        prefix = f"{citation.marker} ({breadcrumb})" if breadcrumb else citation.marker
        blocks.append(f"{prefix} {hit.text}".strip())
    return "\n\n".join(blocks)


__all__ = [
    "PROMPT_TEMPLATE_AR",
    "PROMPT_TEMPLATE_EN",
    "REFUSAL_AR",
    "REFUSAL_EN",
    "render_prompt",
]
