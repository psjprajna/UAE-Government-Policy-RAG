"""Build the local ChromaDB vector index from the registered corpus PDFs.

Walks every source in ``data/registry.json`` (or those passed via
``--source``), parses each PDF, chunks with the real embedder tokenizer
(~500 tokens / chunk for e5-large), embeds, and upserts into a persistent
collection rooted at ``data/chroma_db``.

Exit codes:
    0  every selected source produced ≥1 chunk and upserted successfully
    1  one or more sources failed, or the dimension/model guard refused
    2  bad CLI usage (unknown source slug, conflicting flags)
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from uae_rag import config
from uae_rag.adapters.local.vector_index import DimensionMismatchError
from uae_rag.ingestion.chunker import Chunk, chunk_articles
from uae_rag.ingestion.parser import parse
from uae_rag.ingestion.registry import SOURCES, DocumentSource, find_source
from uae_rag.ports import EmbeddingsPort, VectorIndexPort, VectorRecord

_DEFAULT_PERSIST_DIR = Path("data/chroma_db")
_DEFAULT_RAW_DIR = Path("data/raw")
_DEFAULT_COLLECTION = "uae_policy_chunks"
_MAX_TOKENS = 500  # buffer below e5-large's 512 max sequence length

logger = logging.getLogger("build_index")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_index.py",
        description="Embed and index the UAE policy corpus into a local ChromaDB collection.",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        metavar="SLUG",
        help="Restrict to a specific source slug (repeatable). Default: all registered sources.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=_DEFAULT_PERSIST_DIR,
        help=f"ChromaDB persist directory (default: {_DEFAULT_PERSIST_DIR}).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_DEFAULT_RAW_DIR,
        help=f"Directory containing the raw PDFs (default: {_DEFAULT_RAW_DIR}).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=_DEFAULT_COLLECTION,
        help=f"ChromaDB collection name (default: {_DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse + chunk only; print the summary table without loading the embedder or writing the index.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the existing collection before indexing. Required when swapping embedding models.",
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


def _chunk_source(
    source: DocumentSource,
    raw_dir: Path,
    *,
    count_tokens,
    max_tokens: int,
) -> list[Chunk] | None:
    pdf_path = raw_dir / source.local_filename
    if not pdf_path.exists():
        print(f"  [{source.slug}] missing PDF at {pdf_path}", file=sys.stderr)
        return None
    try:
        articles = parse(pdf_path, source)
        return chunk_articles(
            articles,
            source_slug=source.slug,
            count_tokens=count_tokens,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        print(f"  [{source.slug}] parse/chunk failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def _to_record(chunk: Chunk, vector: list[float]) -> VectorRecord:
    return VectorRecord(
        id=chunk.chunk_id,
        embedding=vector,
        document=chunk.text,
        metadata={
            "source_slug": chunk.source_slug,
            "breadcrumb": chunk.breadcrumb,
            "article_id": chunk.article_id,
            "language": chunk.language,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "mode": chunk.mode,
        },
    )


def _print_summary(rows: list[tuple[str, int, int, str]]) -> None:
    header = ("slug", "articles", "chunks", "mode")
    widths = [max(len(header[i]), max((len(str(r[i])) for r in rows), default=0)) for i in range(4)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print()
    print(fmt.format(*header))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


def _dominant_mode(chunks: list[Chunk]) -> str:
    if not chunks:
        return "—"
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.mode] = counts.get(c.mode, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _dry_run(sources: list[DocumentSource], raw_dir: Path) -> int:
    """Parse + chunk with the default word-count tokenizer; no model load."""
    rows: list[tuple[str, int, int, str]] = []
    failures: list[str] = []
    for i, source in enumerate(sources, start=1):
        print(f"[{i}/{len(sources)}] (dry-run) parsing {source.slug} ...", file=sys.stderr)
        chunks = _chunk_source(
            source,
            raw_dir,
            count_tokens=lambda t: len(t.split()),
            max_tokens=600,
        )
        if chunks is None or not chunks:
            failures.append(source.slug)
            rows.append((source.slug, 0, 0, "—"))
            continue
        articles = len({c.article_id for c in chunks if c.article_id})
        rows.append((source.slug, articles or len(chunks), len(chunks), _dominant_mode(chunks)))
    _print_summary(rows)
    return 1 if failures else 0


def _build(
    sources: list[DocumentSource],
    raw_dir: Path,
    *,
    embedder: EmbeddingsPort,
    index: VectorIndexPort,
    reset: bool,
) -> int:
    if reset:
        print("resetting collection (--reset) ...", file=sys.stderr)
        index.reset()

    rows: list[tuple[str, int, int, str]] = []
    failures: list[str] = []
    total_upserted = 0

    for i, source in enumerate(sources, start=1):
        print(f"[{i}/{len(sources)}] embedding {source.slug} ...", file=sys.stderr)
        chunks = _chunk_source(
            source,
            raw_dir,
            count_tokens=embedder.count_tokens,
            max_tokens=_MAX_TOKENS,
        )
        if chunks is None or not chunks:
            failures.append(source.slug)
            rows.append((source.slug, 0, 0, "—"))
            continue

        vectors = embedder.embed_documents([c.text for c in chunks])
        records = [_to_record(c, v) for c, v in zip(chunks, vectors, strict=True)]
        try:
            index.upsert(records)
        except DimensionMismatchError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        total_upserted += len(records)
        articles = len({c.article_id for c in chunks if c.article_id})
        rows.append((source.slug, articles or len(chunks), len(chunks), _dominant_mode(chunks)))

    _print_summary(rows)
    final_count = index.count()
    print(f"\nupserted: {total_upserted} | collection count: {final_count}", file=sys.stderr)
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv or sys.argv[1:])
    sources = _resolve_sources(args.sources)

    if args.dry_run:
        return _dry_run(sources, args.raw_dir)

    embedder = config.get_embeddings()
    try:
        index = config.get_vector_index(
            persist_dir=args.persist_dir,
            embedder=embedder,
            collection_name=args.collection,
        )
    except DimensionMismatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return _build(sources, args.raw_dir, embedder=embedder, index=index, reset=args.reset)


if __name__ == "__main__":
    sys.exit(main())
