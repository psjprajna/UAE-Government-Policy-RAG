"""Unit tests for the Phase 8 golden-set schema loader.

These tests are fast-lane: they write temp JSONL fixtures and verify
``load_golden`` parses, validates, and dedups deterministically. The
committed-set test auto-skips while ``data/golden_set.jsonl`` is absent so
this file can land before the data file does (TDD ordering).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uae_rag.evals.golden import (
    ExpectedArticle,
    GoldenItem,
    GoldenSetError,
    load_golden,
)

_SENTINEL = object()


def _valid_row(
    *,
    id: str = "Q01",
    question: str = "What is the annual leave entitlement?",
    ground_truth: str = (
        "Article 29 entitles every worker to at least 30 days of annual leave per "
        "completed year of service."
    ),
    expected_articles: object = _SENTINEL,
    language: str = "en",
    topic: str = "labour-law",
    origin: str = "manual",
) -> dict:
    if expected_articles is _SENTINEL:
        expected_articles = [{"source": "labour-law-en", "article": "29"}]
    return {
        "id": id,
        "question": question,
        "ground_truth": ground_truth,
        "expected_articles": expected_articles,
        "language": language,
        "topic": topic,
        "origin": origin,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def test_loads_valid_jsonl_round_trip(tmp_path: Path) -> None:
    rows = [
        _valid_row(),
        _valid_row(
            id="Q02",
            question="ما هي مدة الإجازة السنوية؟",
            ground_truth="تنص المادة 29 على إجازة سنوية لا تقل عن 30 يومًا.",
            expected_articles=[{"source": "labour-law-ar", "article": "29"}],
            language="ar",
            origin="llm-seeded-reviewed",
        ),
    ]
    path = _write_jsonl(tmp_path / "golden.jsonl", rows)
    items = load_golden(path)

    assert len(items) == 2
    assert all(isinstance(item, GoldenItem) for item in items)
    assert items[0].id == "Q01"
    assert items[0].language == "en"
    assert items[0].topic == "labour-law"
    assert items[0].origin == "manual"
    assert items[0].expected_articles == (ExpectedArticle(source="labour-law-en", article="29"),)
    assert items[1].id == "Q02"
    assert items[1].language == "ar"
    assert items[1].expected_articles[0].source == "labour-law-ar"


@pytest.mark.parametrize(
    "missing_key",
    ["id", "question", "ground_truth", "expected_articles", "language", "topic", "origin"],
)
def test_raises_on_missing_required_key(tmp_path: Path, missing_key: str) -> None:
    row = _valid_row()
    del row[missing_key]
    path = _write_jsonl(tmp_path / "golden.jsonl", [row])

    with pytest.raises(GoldenSetError) as exc_info:
        load_golden(path)

    assert "line 1" in str(exc_info.value)
    assert missing_key in str(exc_info.value)


def test_raises_on_invalid_language(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "golden.jsonl", [_valid_row(language="fr")])

    with pytest.raises(GoldenSetError) as exc_info:
        load_golden(path)

    assert "line 1" in str(exc_info.value)
    assert "language" in str(exc_info.value)


def test_raises_on_invalid_topic(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "golden.jsonl", [_valid_row(topic="tax")])

    with pytest.raises(GoldenSetError) as exc_info:
        load_golden(path)

    assert "line 1" in str(exc_info.value)
    assert "topic" in str(exc_info.value)


def test_raises_on_invalid_source_slug(tmp_path: Path) -> None:
    row = _valid_row(expected_articles=[{"source": "not-a-real-source", "article": "29"}])
    path = _write_jsonl(tmp_path / "golden.jsonl", [row])

    with pytest.raises(GoldenSetError) as exc_info:
        load_golden(path)

    assert "line 1" in str(exc_info.value)
    assert "not-a-real-source" in str(exc_info.value)


def test_raises_on_empty_expected_articles(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "golden.jsonl", [_valid_row(expected_articles=[])])

    with pytest.raises(GoldenSetError) as exc_info:
        load_golden(path)

    assert "line 1" in str(exc_info.value)
    assert "expected_articles" in str(exc_info.value)


def test_raises_on_invalid_origin(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "golden.jsonl", [_valid_row(origin="crowdsourced")])

    with pytest.raises(GoldenSetError) as exc_info:
        load_golden(path)

    assert "line 1" in str(exc_info.value)
    assert "origin" in str(exc_info.value)


def test_raises_on_duplicate_id(tmp_path: Path) -> None:
    rows = [_valid_row(id="Q01"), _valid_row(id="Q01")]
    path = _write_jsonl(tmp_path / "golden.jsonl", rows)

    with pytest.raises(GoldenSetError) as exc_info:
        load_golden(path)

    assert "line 2" in str(exc_info.value)
    assert "Q01" in str(exc_info.value)


def test_raises_on_malformed_json_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_valid_row()) + "\n")
        fh.write("not valid json\n")

    with pytest.raises(GoldenSetError) as exc_info:
        load_golden(path)

    assert "line 2" in str(exc_info.value)


def test_committed_golden_set_is_50_rows_schema_valid() -> None:
    repo_data = Path(__file__).resolve().parents[2] / "data" / "golden_set.jsonl"
    if not repo_data.exists():
        pytest.skip("data/golden_set.jsonl not committed yet (Slice A in progress)")

    items = load_golden(repo_data)

    assert len(items) == 50, f"expected 50 rows, got {len(items)}"

    languages = {item.language for item in items}
    assert languages == {"en", "ar"}, f"language coverage incomplete: {languages}"
    en_count = sum(1 for item in items if item.language == "en")
    ar_count = sum(1 for item in items if item.language == "ar")
    assert abs(en_count - ar_count) <= 10, (
        f"EN/AR imbalance: en={en_count} ar={ar_count} (allowed ±10 of 25/25)"
    )

    topics = {item.topic for item in items}
    assert topics == {"labour-law", "mohre", "visa"}, f"topic coverage incomplete: {topics}"
    for topic in ("labour-law", "mohre", "visa"):
        rows_for_topic = sum(1 for item in items if item.topic == topic)
        assert rows_for_topic >= 5, f"topic {topic!r} has only {rows_for_topic} rows; need ≥5"

    manual = sum(1 for item in items if item.origin == "manual")
    assert manual >= 10, f"need ≥10 manual-origin rows; got {manual}"
