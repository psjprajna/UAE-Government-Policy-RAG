"""Document registry: SHA-256-verified manifest of corpus PDFs.

Pure local-FS module. The CLI fetcher in ``scripts/fetch_corpus.py`` is the only
component that performs network I/O; everything here operates on bytes already on
disk so the logic stays testable without Ollama, ChromaDB, or the internet.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_HASH_CHUNK_BYTES = 64 * 1024


class HashMismatchError(Exception):
    """Raised when an on-disk file's SHA-256 differs from the registry record."""


@dataclass(frozen=True, slots=True)
class DocumentSource:
    """Static metadata for one corpus document — slug, language, URL, target filename."""

    slug: str
    title: str
    language: str
    url: str
    local_filename: str
    requires_browser: bool = False


class DocumentRecord(BaseModel):
    """Hash-verified manifest entry persisted to ``data/registry.json``."""

    slug: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(..., ge=0)
    fetched_at: str = Field(..., min_length=1)
    title: str
    language: str


SOURCES: tuple[DocumentSource, ...] = (
    DocumentSource(
        slug="labour-law-en",
        title="UAE Labour Law (Federal Decree-Law No. 33 of 2021) — English",
        language="en",
        url="https://uaelegislation.gov.ae/en/legislations/1541/download",
        local_filename="labour-law-en.pdf",
        requires_browser=True,
    ),
    DocumentSource(
        slug="labour-law-ar",
        title="قانون العمل الاتحادي رقم 33 لسنة 2021 — Arabic",
        language="ar",
        url="https://uaelegislation.gov.ae/ar/legislations/1541/download",
        local_filename="labour-law-ar.pdf",
        requires_browser=True,
    ),
    DocumentSource(
        slug="mohre-resolutions",
        title="MOHRE Cabinet Resolution No. 1 of 2022 — Executive Regulations of Decree-Law No. 33",
        language="en",
        url="https://www.mohre.gov.ae/assets/download/46bdbfda/Cabinet%20Resolution%20_Executive%20Regulations%20Decree-Law%20No.%2033.pdf.aspx",
        local_filename="mohre-resolutions.pdf",
        requires_browser=True,
    ),
    DocumentSource(
        slug="visa-regulations",
        title="ICP Services Guide v4.2 (2024) — UAE entry and residence regulations",
        language="en",
        url="https://icp.gov.ae/wp-content/uploads/2024/05/%D8%AF%D9%84%D9%8A%D9%84-%D8%A7%D9%84%D8%AE%D8%AF%D9%85%D8%A7%D8%AA-%D8%A5%D8%B5%D8%AF%D8%A7%D8%B1-4.2-2024EN.pdf",
        local_filename="visa-regulations.pdf",
        requires_browser=False,
    ),
)


def find_source(slug: str) -> DocumentSource:
    """Return the ``DocumentSource`` for ``slug``; raise ``KeyError`` if absent."""
    for source in SOURCES:
        if source.slug == slug:
            return source
    raise KeyError(f"Unknown source slug: {slug!r}. Known: {[s.slug for s in SOURCES]}")


def compute_sha256(path: Path) -> str:
    """Stream ``path`` in 64 KiB chunks and return the hex SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(path: Path) -> dict[str, DocumentRecord]:
    """Load registry JSON. Missing file or empty object both yield an empty dict."""
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Registry root must be a JSON object, got {type(payload).__name__}")
    return {slug: DocumentRecord(**entry) for slug, entry in payload.items()}


def save_registry(path: Path, records: dict[str, DocumentRecord]) -> None:
    """Atomically write ``records`` to ``path`` via a ``.tmp`` sibling + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {slug: record.model_dump() for slug, record in sorted(records.items())}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def verify_or_record(
    source: DocumentSource,
    file_path: Path,
    registry: dict[str, DocumentRecord],
    *,
    force: bool = False,
) -> DocumentRecord:
    """Reconcile ``file_path`` against ``registry[source.slug]``.

    First run / backfill → write new record. Hash matches → return existing.
    Hash differs → raise ``HashMismatchError`` unless ``force=True``, in which
    case the existing record is replaced.
    """
    disk_sha = compute_sha256(file_path)
    existing = registry.get(source.slug)

    if existing is not None and existing.sha256 == disk_sha:
        return existing

    if existing is not None and not force:
        raise HashMismatchError(
            f"Hash mismatch for {source.slug}: registry={existing.sha256}, disk={disk_sha}. "
            "Re-run with --force after confirming the source has updated."
        )

    new_record = DocumentRecord(
        slug=source.slug,
        url=source.url,
        sha256=disk_sha,
        bytes=file_path.stat().st_size,
        fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        title=source.title,
        language=source.language,
    )
    registry[source.slug] = new_record
    if existing is None:
        logger.info("registered %s sha=%s", source.slug, disk_sha[:12])
    else:
        logger.warning(
            "overwrote %s sha=%s -> %s", source.slug, existing.sha256[:12], disk_sha[:12]
        )
    return new_record


__all__ = [
    "SOURCES",
    "DocumentRecord",
    "DocumentSource",
    "HashMismatchError",
    "compute_sha256",
    "find_source",
    "load_registry",
    "save_registry",
    "verify_or_record",
]
