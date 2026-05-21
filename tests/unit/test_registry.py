"""Unit tests for the document registry — pure I/O over tmp_path, no network."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from uae_rag.ingestion.registry import (
    SOURCES,
    DocumentRecord,
    DocumentSource,
    HashMismatchError,
    compute_sha256,
    find_source,
    load_registry,
    save_registry,
    verify_or_record,
)


@pytest.fixture
def sample_source() -> DocumentSource:
    return DocumentSource(
        slug="labour-law-en",
        title="UAE Labour Law (EN) — sample",
        language="en",
        url="https://example.test/labour-law-en.pdf",
        local_filename="labour-law-en.pdf",
    )


@pytest.fixture
def pdf_bytes() -> bytes:
    return b"%PDF-1.4\nhello world"


@pytest.fixture
def pdf_on_disk(tmp_path: Path, pdf_bytes: bytes) -> Path:
    target = tmp_path / "labour-law-en.pdf"
    target.write_bytes(pdf_bytes)
    return target


def test_load_registry_returns_empty_dict_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-file.json"
    assert load_registry(missing) == {}


def test_load_registry_returns_empty_dict_for_empty_json(tmp_path: Path) -> None:
    target = tmp_path / "registry.json"
    target.write_text("{}\n", encoding="utf-8")
    assert load_registry(target) == {}


def test_save_then_load_round_trip(tmp_path: Path, sample_source: DocumentSource) -> None:
    record = DocumentRecord(
        slug=sample_source.slug,
        url=sample_source.url,
        sha256="a" * 64,
        bytes=42,
        fetched_at="2026-05-21T10:00:00+00:00",
        title=sample_source.title,
        language=sample_source.language,
    )
    target = tmp_path / "registry.json"

    save_registry(target, {record.slug: record})
    loaded = load_registry(target)

    assert loaded == {record.slug: record}


def test_compute_sha256_matches_known_value(pdf_on_disk: Path, pdf_bytes: bytes) -> None:
    expected = hashlib.sha256(pdf_bytes).hexdigest()
    assert compute_sha256(pdf_on_disk) == expected


def test_verify_or_record_creates_record_on_first_run(
    sample_source: DocumentSource, pdf_on_disk: Path
) -> None:
    registry: dict[str, DocumentRecord] = {}

    record = verify_or_record(sample_source, pdf_on_disk, registry)

    assert record.slug == sample_source.slug
    assert record.bytes == pdf_on_disk.stat().st_size
    assert record.sha256 == compute_sha256(pdf_on_disk)
    assert registry == {sample_source.slug: record}


def test_verify_or_record_returns_existing_when_hash_matches(
    sample_source: DocumentSource, pdf_on_disk: Path
) -> None:
    registry: dict[str, DocumentRecord] = {}
    first = verify_or_record(sample_source, pdf_on_disk, registry)

    again = verify_or_record(sample_source, pdf_on_disk, registry)

    assert again is first
    assert len(registry) == 1


def test_verify_or_record_raises_hash_mismatch_without_force(
    sample_source: DocumentSource, pdf_on_disk: Path
) -> None:
    registry: dict[str, DocumentRecord] = {}
    verify_or_record(sample_source, pdf_on_disk, registry)

    pdf_on_disk.write_bytes(b"%PDF-1.4\ndifferent content")

    with pytest.raises(HashMismatchError) as exc:
        verify_or_record(sample_source, pdf_on_disk, registry)

    assert sample_source.slug in str(exc.value)
    assert "--force" in str(exc.value)


def test_verify_or_record_overwrites_when_force_true(
    sample_source: DocumentSource, pdf_on_disk: Path
) -> None:
    registry: dict[str, DocumentRecord] = {}
    first = verify_or_record(sample_source, pdf_on_disk, registry)

    pdf_on_disk.write_bytes(b"%PDF-1.4\ndifferent content")
    new_record = verify_or_record(sample_source, pdf_on_disk, registry, force=True)

    assert new_record.sha256 != first.sha256
    assert registry[sample_source.slug].sha256 == new_record.sha256


def test_find_source_raises_keyerror_on_unknown_slug() -> None:
    with pytest.raises(KeyError):
        find_source("not-a-real-slug")


def test_sources_contains_four_expected_slugs() -> None:
    """SOURCES constant pins the corpus shape — 4 PDFs per Phase 1 spec."""
    slugs = {source.slug for source in SOURCES}
    assert slugs == {
        "labour-law-en",
        "labour-law-ar",
        "mohre-resolutions",
        "visa-regulations",
    }


def test_save_registry_is_atomic_via_tmp_file(
    tmp_path: Path, sample_source: DocumentSource
) -> None:
    """save_registry must not leave a .tmp sibling after success."""
    record = DocumentRecord(
        slug=sample_source.slug,
        url=sample_source.url,
        sha256="b" * 64,
        bytes=1,
        fetched_at="2026-05-21T10:00:00+00:00",
        title=sample_source.title,
        language=sample_source.language,
    )
    target = tmp_path / "registry.json"

    save_registry(target, {record.slug: record})

    assert target.exists()
    assert not target.with_suffix(".json.tmp").exists()
    # Sanity: file is valid JSON.
    json.loads(target.read_text(encoding="utf-8"))
