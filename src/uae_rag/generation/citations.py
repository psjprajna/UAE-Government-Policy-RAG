"""Citation rendering — turn ranked ``RetrievalHit``s into deduped, marker-bearing citations.

Per ADR-0008, citation markers are 1-based bracketed integers (``[1]``,
``[2]``, …) and are deduped on the ``(source_slug, article_id)`` key so two
chunks from the same article share a marker. For chunks with no
``article_id`` (fallback chunks emitted by the chunker), the breadcrumb tail
is used in its place — truncated to keep prompts compact.

Domain shape — the API's wire ``Citation`` model lives in ``api/main.py`` and
will be mapped from this dataclass at the boundary in Phase 7.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from uae_rag.ports.retrieval import RetrievalHit

_MAX_ARTICLE_LABEL = 80


@dataclass(frozen=True, slots=True)
class Citation:
    """One deduped citation, ready to be rendered into a prompt or API response.

    Attributes:
        marker: 1-based bracketed marker, e.g. ``"[1]"``. Shared across hits
            that resolve to the same ``(source, article)``.
        source: Originating document slug, e.g. ``"labour-law-en"``.
        article: Article identifier, or the breadcrumb tail when the hit's
            ``article_id`` is ``None``.
        chunk_id: Full chunk id of the *first* hit that produced this citation
            — retained for downstream RAGAS traceability.
        language: Language code of the cited chunk (``"en"`` | ``"ar"``).
    """

    marker: str
    source: str
    article: str
    chunk_id: str
    language: str


def render_citations(hits: Sequence[RetrievalHit]) -> list[Citation]:
    """Return one ``Citation`` per input hit, sharing a marker on ``(source, article)``.

    The output preserves input order and length. Repeated ``(source, article)``
    pairs reuse the marker assigned to their first occurrence.
    """
    if not hits:
        return []

    marker_by_key: dict[tuple[str, str], str] = {}
    citations: list[Citation] = []
    next_index = 1

    for hit in hits:
        source = str(hit.metadata.get("source_slug", ""))
        article = _resolve_article_label(hit)
        key = (source, article)

        marker = marker_by_key.get(key)
        if marker is None:
            marker = f"[{next_index}]"
            marker_by_key[key] = marker
            next_index += 1

        citations.append(
            Citation(
                marker=marker,
                source=source,
                article=article,
                chunk_id=hit.chunk_id,
                language=str(hit.metadata.get("language", "")),
            )
        )

    return citations


def _resolve_article_label(hit: RetrievalHit) -> str:
    """Use ``article_id`` when present; otherwise the breadcrumb tail (truncated)."""
    article_id = hit.metadata.get("article_id")
    if article_id is not None:
        return str(article_id)

    breadcrumb = str(hit.metadata.get("breadcrumb", ""))
    tail = breadcrumb.rsplit(">", 1)[-1].strip() if breadcrumb else hit.chunk_id
    if len(tail) > _MAX_ARTICLE_LABEL:
        tail = tail[: _MAX_ARTICLE_LABEL - 1].rstrip() + "…"
    return tail


__all__ = ["Citation", "render_citations"]
