"""Golden-set schema + JSONL loader for Phase 8 RAGAS evaluation.

The committed ``app/data/golden_set.jsonl`` is the input fixture for every
RAGAS run. This module is the gate: anything reaching the evaluator first
passes through :func:`load_golden`, which validates structure and
enum-bounded fields per row and raises :class:`GoldenSetError` with the
offending line number on any violation. Source slugs are validated against
the canonical :data:`uae_rag.ingestion.registry.SOURCES` list — no
hand-maintained copy.

Imports are restricted to stdlib and the sibling ``ingestion`` domain
module per the Phase-8 spec; never ``adapters``. Enforced by
``tests/fitness/test_layer_boundaries.py``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from uae_rag.ingestion.registry import SOURCES

logger = logging.getLogger(__name__)

_VALID_LANGUAGES: frozenset[str] = frozenset({"en", "ar"})
_VALID_TOPICS: frozenset[str] = frozenset({"labour-law", "mohre", "visa"})
_VALID_ORIGINS: frozenset[str] = frozenset({"manual", "llm-seeded-reviewed"})
_VALID_SOURCE_SLUGS: frozenset[str] = frozenset(s.slug for s in SOURCES)
_REQUIRED_KEYS: tuple[str, ...] = (
    "id",
    "question",
    "ground_truth",
    "expected_articles",
    "language",
    "topic",
    "origin",
)


class GoldenSetError(Exception):
    """Raised when a row in the golden-set JSONL fails validation.

    Messages are prefixed with ``"line N: "`` so a curator can jump
    straight to the offending row in a 50-line file.
    """


@dataclass(frozen=True, slots=True)
class ExpectedArticle:
    """One ``(source, article)`` pair pointing at the chunk(s) expected in retrieval."""

    source: str
    article: str


@dataclass(frozen=True, slots=True)
class GoldenItem:
    """One curated evaluation row: question, reference answer, expected provenance."""

    id: str
    question: str
    ground_truth: str
    expected_articles: tuple[ExpectedArticle, ...]
    language: str
    topic: str
    origin: str


def load_golden(path: Path) -> list[GoldenItem]:
    """Parse and validate ``path`` as JSONL. Return one :class:`GoldenItem` per row.

    Empty lines are skipped silently (common JSONL convention). Every
    populated line must be a JSON object with all keys in
    :data:`_REQUIRED_KEYS`, language / topic / origin in their allowed
    sets, ``expected_articles`` a non-empty list whose ``source`` values
    are all in :data:`_VALID_SOURCE_SLUGS`. IDs must be unique across the
    file.
    """
    items: list[GoldenItem] = []
    seen_ids: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            items.append(_parse_row(raw, line_no, seen_ids))
    logger.info("loaded %d golden items from %s", len(items), path)
    return items


def _parse_row(raw: str, line_no: int, seen_ids: dict[str, int]) -> GoldenItem:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GoldenSetError(f"line {line_no}: malformed JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise GoldenSetError(
            f"line {line_no}: row must be a JSON object, got {type(payload).__name__}"
        )

    for key in _REQUIRED_KEYS:
        if key not in payload:
            raise GoldenSetError(f"line {line_no}: missing required key {key!r}")

    item_id = payload["id"]
    if not isinstance(item_id, str) or not item_id.strip():
        raise GoldenSetError(f"line {line_no}: 'id' must be a non-empty string")
    if item_id in seen_ids:
        raise GoldenSetError(
            f"line {line_no}: duplicate id {item_id!r} (first seen on line {seen_ids[item_id]})"
        )

    language = payload["language"]
    if language not in _VALID_LANGUAGES:
        raise GoldenSetError(
            f"line {line_no}: invalid 'language' {language!r}; allowed: {sorted(_VALID_LANGUAGES)}"
        )

    topic = payload["topic"]
    if topic not in _VALID_TOPICS:
        raise GoldenSetError(
            f"line {line_no}: invalid 'topic' {topic!r}; allowed: {sorted(_VALID_TOPICS)}"
        )

    origin = payload["origin"]
    if origin not in _VALID_ORIGINS:
        raise GoldenSetError(
            f"line {line_no}: invalid 'origin' {origin!r}; allowed: {sorted(_VALID_ORIGINS)}"
        )

    expected_articles = _parse_expected_articles(payload["expected_articles"], line_no)

    seen_ids[item_id] = line_no
    return GoldenItem(
        id=item_id,
        question=str(payload["question"]),
        ground_truth=str(payload["ground_truth"]),
        expected_articles=expected_articles,
        language=language,
        topic=topic,
        origin=origin,
    )


def _parse_expected_articles(raw: object, line_no: int) -> tuple[ExpectedArticle, ...]:
    if not isinstance(raw, list):
        raise GoldenSetError(
            f"line {line_no}: 'expected_articles' must be a list, got {type(raw).__name__}"
        )
    if not raw:
        raise GoldenSetError(f"line {line_no}: 'expected_articles' must be non-empty")

    parsed: list[ExpectedArticle] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise GoldenSetError(f"line {line_no}: 'expected_articles'[{idx}] must be an object")
        for sub_key in ("source", "article"):
            if sub_key not in entry:
                raise GoldenSetError(
                    f"line {line_no}: 'expected_articles'[{idx}] missing key {sub_key!r}"
                )
        source = entry["source"]
        article = entry["article"]
        if source not in _VALID_SOURCE_SLUGS:
            raise GoldenSetError(
                f"line {line_no}: 'expected_articles'[{idx}] has unknown source "
                f"{source!r}; allowed: {sorted(_VALID_SOURCE_SLUGS)}"
            )
        if not isinstance(article, str) or not article.strip():
            raise GoldenSetError(
                f"line {line_no}: 'expected_articles'[{idx}] 'article' must be a non-empty string"
            )
        parsed.append(ExpectedArticle(source=source, article=article))
    return tuple(parsed)


__all__ = ["ExpectedArticle", "GoldenItem", "GoldenSetError", "load_golden"]
