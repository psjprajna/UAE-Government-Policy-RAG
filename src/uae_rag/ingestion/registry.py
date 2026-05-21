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
        title="UAE Labour Law (Federal Law No. 33 of 2021) — English",
        language="en",
        url="https://uaelegislation.gov.ae/api/v1/legislations/1290/downloadPublication/en",
        local_filename="labour-law-en.pdf",
    ),
    DocumentSource(
        slug="labour-law-ar",
        title="قانون العمل الاتحادي رقم 33 لسنة 2021",
        language="ar",
        url="https://uaelegislation.gov.ae/api/v1/legislations/1290/downloadPublication/ar",
        local_filename="labour-law-ar.pdf",
    ),
    DocumentSource(
        slug="mohre-resolutions",
        title="MOHRE Cabinet Resolutions implementing the Labour Law",
        language="en",
        url="https://www.mohre.gov.ae/handlers/getfile.ashx?fileurl=/-/media/files/mohre/resolutions/labour-law-resolutions.pdf",
        local_filename="mohre-resolutions.pdf",
    ),
    DocumentSource(
        slug="visa-regulations",
        title="UAE Federal Visa and Entry Regulations",
        language="en",
        url="https://icp.gov.ae/wp-content/uploads/2023/visa-regulations.pdf",
        local_filename="visa-regulations.pdf",
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
