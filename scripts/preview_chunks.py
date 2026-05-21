"""Preview the chunk output for each registered corpus PDF.

Reads ``data/registry.json``, parses every (or selected) source via the
heading-aware parser + chunker, and prints a summary table to stdout.

Exit codes:
    0  every source produced ≥1 chunk
    1  one or more sources failed to parse or produced 0 chunks
    2  unknown ``--source`` slug supplied on the CLI
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections import Counter
from pathlib import Path

from uae_rag.ingestion.chunker import Chunk, chunk_articles
from uae_rag.ingestion.parser import parse
from uae_rag.ingestion.registry import SOURCES, DocumentSource, find_source

_DEFAULT_REGISTRY = Path("data/registry.json")
_DEFAULT_RAW_DIR = Path("data/raw")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="preview_chunks.py",
        description="Parse and chunk corpus PDFs, print a summary table.",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        metavar="SLUG",
        help="Restrict to a specific source slug (repeatable). Default: all registered sources.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_DEFAULT_RAW_DIR,
        help=f"Directory containing the raw PDFs (default: {_DEFAULT_RAW_DIR}).",
    )
    return parser.parse_args(argv)


def _resolve_sources(slugs: list[str] | None) -> list[DocumentSource]:
    if not slugs:
        return list(SOURCES)
    resolved: list[DocumentSource] = []
    for slug in slugs:
        try:
            resolved.append(find_source(slug))
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)
    return resolved


def _process_source(source: DocumentSource, raw_dir: Path) -> tuple[int, list[Chunk]] | None:
    pdf_path = raw_dir / source.local_filename
    if not pdf_path.exists():
        print(f"  [{source.slug}] missing PDF at {pdf_path}", file=sys.stderr)
        return None
    try:
        articles = parse(pdf_path, source)
        chunks = chunk_articles(articles, source_slug=source.slug)
        return len(articles), chunks
    except Exception as exc:
        print(f"  [{source.slug}] parse failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def _dominant_mode(chunks: list[Chunk]) -> str:
    if not chunks:
        return "—"
    counts = Counter(c.mode for c in chunks)
    return counts.most_common(1)[0][0]


def _print_summary(rows: list[tuple[str, int, int, str]]) -> None:
    header = ("slug", "articles", "chunks", "mode")
    widths = [max(len(header[i]), max((len(str(r[i])) for r in rows), default=0)) for i in range(4)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print()
    print(fmt.format(*header))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    sources = _resolve_sources(args.sources)

    rows: list[tuple[str, int, int, str]] = []
    failures: list[str] = []
    for i, source in enumerate(sources, start=1):
        print(f"[{i}/{len(sources)}] parsing {source.slug} ...", file=sys.stderr)
        result = _process_source(source, args.raw_dir)
        if result is None or not result[1]:
            failures.append(source.slug)
            rows.append((source.slug, 0, 0, "—"))
            continue
        articles_count, chunks = result
        rows.append((source.slug, articles_count, len(chunks), _dominant_mode(chunks)))

    _print_summary(rows)

    if failures:
        print(f"\nfailed sources: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
