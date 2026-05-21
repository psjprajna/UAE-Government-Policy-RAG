#!/usr/bin/env python3
"""Download UAE government policy PDFs and reconcile them against the registry.

Re-running is a no-op on unchanged sources. Hash drift aborts the run unless
``--force`` is given. ``--skip-download`` only verifies what's already on disk.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path

# scripts/ sits outside the domain layer enforced by tests/fitness/test_layer_boundaries.py,
# so direct imports from adapters/ are permitted here (and only here, plus config.py).
from uae_rag.adapters.local.browser_fetcher import BrowserFetcher, BrowserFetchError
from uae_rag.ingestion.registry import (
    SOURCES,
    DocumentSource,
    HashMismatchError,
    find_source,
    load_registry,
    save_registry,
    verify_or_record,
)

USER_AGENT = "uae-rag-fetcher/0.1"
TIMEOUT_SECONDS = 60.0
DOWNLOAD_CHUNK_BYTES = 64 * 1024
PDF_MAGIC = b"%PDF"
DATA_ROOT = Path("data") / "raw"
REGISTRY_PATH = Path("data") / "registry.json"


class FetchError(Exception):
    """Wraps every download-or-validate failure with the slug that caused it."""


def _selected_sources(requested: list[str]) -> list[DocumentSource]:
    if not requested:
        return list(SOURCES)
    return [find_source(slug) for slug in requested]


def _urllib_to_tmp(url: str, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with (
            urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response,
            tmp.open("wb") as fh,
        ):
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                fh.write(chunk)
    except (urllib.error.URLError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise FetchError(f"network: {exc}") from exc
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise FetchError(f"disk: {exc}") from exc

    head = tmp.read_bytes()[: len(PDF_MAGIC)]
    if head != PDF_MAGIC:
        tmp.unlink(missing_ok=True)
        raise FetchError(f"not a PDF (first bytes: {head!r})")
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise FetchError("0-byte download")
    tmp.replace(target)


def _download(source: DocumentSource, target: Path) -> None:
    """Dispatch to the right fetcher: BrowserFetcher for JS-walled sources, urllib otherwise."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.requires_browser:
        try:
            BrowserFetcher(timeout_seconds=TIMEOUT_SECONDS).download(source.url, target)
        except BrowserFetchError as exc:
            raise FetchError(f"browser: {exc}") from exc
    else:
        _urllib_to_tmp(source.url, target)


def _process(
    source: DocumentSource,
    *,
    skip_download: bool,
    force: bool,
    registry: dict,
) -> str:
    """Reconcile one source. Returns a one-line status for the summary table."""
    target = DATA_ROOT / source.local_filename

    if not target.exists():
        if skip_download:
            raise FetchError("file missing and --skip-download set")
        _download(source, target)

    existed_before = source.slug in registry
    record = verify_or_record(source, target, registry, force=force)
    state = (
        "unchanged"
        if existed_before and not force
        else ("forced" if force and existed_before else "recorded")
    )
    return f"  {source.slug:20s} {state:10s} {record.bytes:>10d} B  sha={record.sha256[:12]}…"


def _run(args: argparse.Namespace) -> int:
    selected = _selected_sources(args.source)
    registry = load_registry(REGISTRY_PATH)
    rows: list[str] = []
    failed: list[tuple[str, str]] = []

    for i, source in enumerate(selected, start=1):
        print(f"[{i}/{len(selected)}] {source.slug} ...")
        try:
            rows.append(
                _process(
                    source,
                    skip_download=args.skip_download,
                    force=args.force,
                    registry=registry,
                )
            )
        except HashMismatchError as exc:
            failed.append((source.slug, str(exc)))
            print(f"  HASH MISMATCH: {exc}", file=sys.stderr)
        except FetchError as exc:
            failed.append((source.slug, str(exc)))
            print(f"  FAILED: {exc}", file=sys.stderr)
            print(
                f"  Manual fallback: download {source.url}\n"
                f"  and place it at app/{DATA_ROOT}/{source.local_filename}, "
                f"then re-run with --skip-download.",
                file=sys.stderr,
            )

    save_registry(REGISTRY_PATH, registry)

    print()
    print("Registry summary:")
    for row in rows:
        print(row)
    if failed:
        print()
        print(f"{len(failed)} source(s) failed:", file=sys.stderr)
        for slug, reason in failed:
            print(f"  - {slug}: {reason}", file=sys.stderr)
        return 1
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download UAE government policy PDFs and update the SHA-256 registry.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only verify on-disk hashes against the registry; never fetch.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing registry entries on hash mismatch (use after confirming a real upstream update).",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="SLUG",
        help=f"Limit to specific slug(s). Repeatable. Known: {', '.join(s.slug for s in SOURCES)}.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        return _run(args)
    except KeyError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
