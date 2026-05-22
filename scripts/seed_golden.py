"""Seed Phase-8 golden-set candidates by prompting an LLM over the corpus.

For each registered source the script parses + chunks the PDF (same recipe as
``build_index.py``), samples ``--num-per-source`` article-mode chunks, and asks
``llama3.1:8b`` (or whatever ``--model`` overrides) for one
``{question, ground_truth, expected_articles}`` triple per chunk under
``format="json"`` grammar constraint (Ollama-only; bypasses the ``LLMPort``
since scripts may import ``ollama`` directly).

Outputs three sibling JSONL files under ``--output-dir``:

* ``golden_set.draft.jsonl`` — every seed candidate that parsed AND validated.
* ``golden_set.rejected.jsonl`` — each ``{chunk_id, raw_response,
  rejection_reason}`` for parsed-but-rejected rows.
* ``golden_set.seed_log.jsonl`` — one append-only row per script invocation
  (``started_at``, ``completed_at``, ``model``, ``temperature``, source args,
  accepted/rejected/parse-failed counts, ``git_sha``).

Humans then review ``.draft.jsonl``, add manual anchors, and save the curated
set as ``golden_set.jsonl``.

Exit codes: 0 (≥1 row produced), 1 (LLM unavailable or all rejected),
2 (bad CLI usage).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import ollama

from uae_rag.evals.golden import (
    GoldenSetError,
    load_golden,
)
from uae_rag.ingestion.chunker import Chunk, chunk_articles
from uae_rag.ingestion.parser import parse
from uae_rag.ingestion.registry import SOURCES, DocumentSource, find_source

_DEFAULT_RAW_DIR = Path("data/raw")
_DEFAULT_OUTPUT_DIR = Path("data")
_DEFAULT_MODEL = "llama3.1:8b"
_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_TIMEOUT_S = 120.0
_DEFAULT_NUM_PER_SOURCE = 10
_DRAFT_FILENAME = "golden_set.draft.jsonl"
_REJECTED_FILENAME = "golden_set.rejected.jsonl"
_SEED_LOG_FILENAME = "golden_set.seed_log.jsonl"

_TOPIC_FOR_SOURCE: dict[str, str] = {
    "labour-law-en": "labour-law",
    "labour-law-ar": "labour-law",
    "mohre-resolutions": "mohre",
    "visa-regulations": "visa",
}

logger = logging.getLogger("seed_golden")


@dataclass(frozen=True, slots=True)
class _SeedCounts:
    accepted: int
    rejected: int
    parse_failed: int


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="seed_golden.py",
        description="LLM-seed golden-set candidates from the corpus PDFs.",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        choices=[s.slug for s in SOURCES],
        metavar="SLUG",
        help="Restrict to a specific source slug (repeatable). Default: all sources.",
    )
    parser.add_argument(
        "--num-per-source",
        type=int,
        default=_DEFAULT_NUM_PER_SOURCE,
        help=f"Candidates to seed per source (default: {_DEFAULT_NUM_PER_SOURCE}).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=_DEFAULT_RAW_DIR,
        help=f"Directory containing the raw PDFs (default: {_DEFAULT_RAW_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory to write draft/rejected/seed_log JSONL (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="Ollama model id.")
    parser.add_argument("--host", default=_DEFAULT_HOST, help="Ollama host URL.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Decoding temperature; 0.0 (default) per ADR-0008 determinism.",
    )
    return parser.parse_args(argv)


def _resolve_sources(slugs: list[str] | None) -> list[DocumentSource]:
    if not slugs:
        return list(SOURCES)
    return [find_source(slug) for slug in slugs]


def _sample_chunks(chunks: list[Chunk], n: int) -> list[Chunk]:
    eligible = [c for c in chunks if c.article_id is not None]
    if not eligible:
        return []
    if len(eligible) <= n:
        return eligible
    step = len(eligible) / n
    return [eligible[int(i * step)] for i in range(n)]


def _build_prompt(chunk: Chunk, source_slug: str) -> str:
    target_lang = "Arabic" if chunk.language == "ar" else "English"
    return f"""You are reading one passage from the UAE government legal corpus. Produce ONE valid JSON object with exactly these keys:
"question": a natural-language question in {target_lang} whose answer is contained in the passage.
"ground_truth": a 2-4 sentence answer in {target_lang}, grounded entirely in the passage.
"expected_articles": a JSON array with one object: {{"source": "{source_slug}", "article": "{chunk.article_id}"}}.

Rules:
1. Use ONLY the article number that appears in the passage breadcrumb ({chunk.article_id}).
2. Do NOT invent facts. Every claim in ground_truth must be supported by the passage text.
3. Output ONLY the JSON object. No markdown fences, no commentary.
4. The question must be specific to this passage, not generic.

PASSAGE:
{chunk.text}
"""


def _call_llm(client: ollama.Client, *, model: str, prompt: str, temperature: float) -> str:
    response = client.generate(
        model=model,
        prompt=prompt,
        stream=False,
        format="json",
        options={"temperature": temperature, "num_predict": 800, "num_ctx": 8192},
        keep_alive="5m",
    )
    return str(response.response).strip()


def _validate_seed(raw: str, chunk: Chunk, source_slug: str) -> tuple[dict | None, str | None]:
    """Parse the LLM JSON and verify article matches. Return (row, None) or (None, reason)."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"json_decode: {exc.msg}"
    if not isinstance(payload, dict):
        return None, f"not_an_object: got {type(payload).__name__}"
    for key in ("question", "ground_truth", "expected_articles"):
        if key not in payload:
            return None, f"missing_key: {key}"
    ea = payload["expected_articles"]
    if not isinstance(ea, list) or not ea or not isinstance(ea[0], dict):
        return None, "bad_expected_articles_shape"
    if ea[0].get("article") != chunk.article_id:
        return None, f"article_mismatch: model={ea[0].get('article')!r} chunk={chunk.article_id!r}"
    if ea[0].get("source") != source_slug:
        return None, f"source_mismatch: model={ea[0].get('source')!r} chunk={source_slug!r}"
    return payload, None


def _seed_source(
    source: DocumentSource,
    chunks: list[Chunk],
    *,
    num: int,
    client: ollama.Client,
    model: str,
    temperature: float,
    draft_fh,
    rejected_fh,
    next_id: int,
) -> tuple[_SeedCounts, int]:
    sampled = _sample_chunks(chunks, num)
    topic = _TOPIC_FOR_SOURCE[source.slug]
    accepted = rejected = parse_failed = 0
    for i, chunk in enumerate(sampled, start=1):
        prompt = _build_prompt(chunk, source.slug)
        try:
            raw = _call_llm(client, model=model, prompt=prompt, temperature=temperature)
        except (httpx.HTTPError, ConnectionError, ollama.ResponseError) as exc:
            print(
                f"  [{source.slug}] {i}/{len(sampled)} chunk={chunk.chunk_id}: LLM unavailable ({exc})",
                file=sys.stderr,
            )
            raise
        row, reason = _validate_seed(raw, chunk, source.slug)
        if row is None:
            if reason and reason.startswith("json_decode"):
                parse_failed += 1
            else:
                rejected += 1
            rejected_fh.write(
                json.dumps(
                    {"chunk_id": chunk.chunk_id, "raw_response": raw, "rejection_reason": reason},
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(
                f"  [{source.slug}] {i}/{len(sampled)} chunk={chunk.chunk_id}: REJECTED ({reason})",
                file=sys.stderr,
            )
            continue
        draft_row = {
            "id": f"Q{next_id:02d}",
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "expected_articles": row["expected_articles"],
            "language": chunk.language,
            "topic": topic,
            "origin": "llm-seeded-reviewed",
        }
        draft_fh.write(json.dumps(draft_row, ensure_ascii=False) + "\n")
        accepted += 1
        next_id += 1
        print(
            f"  [{source.slug}] {i}/{len(sampled)} chunk={chunk.chunk_id}: ok (id=Q{next_id - 1:02d})",
            file=sys.stderr,
        )
    return _SeedCounts(accepted, rejected, parse_failed), next_id


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _next_draft_id(draft_path: Path) -> int:
    if not draft_path.exists():
        return 1
    try:
        existing = load_golden(draft_path)
    except GoldenSetError:
        existing = []
    used = []
    for item in existing:
        if item.id.startswith("Q") and item.id[1:].isdigit():
            used.append(int(item.id[1:]))
    return (max(used) + 1) if used else 1


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv or sys.argv[1:])
    if args.num_per_source < 1:
        print("error: --num-per-source must be ≥ 1", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = _resolve_sources(args.sources)
    client = ollama.Client(host=args.host, timeout=_DEFAULT_TIMEOUT_S)

    draft_path = args.output_dir / _DRAFT_FILENAME
    rejected_path = args.output_dir / _REJECTED_FILENAME
    next_id = _next_draft_id(draft_path)

    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    total = _SeedCounts(0, 0, 0)
    try:
        with (
            draft_path.open("a", encoding="utf-8") as draft_fh,
            rejected_path.open("a", encoding="utf-8") as rejected_fh,
        ):
            for i, source in enumerate(sources, start=1):
                print(
                    f"[{i}/{len(sources)}] seeding {source.slug} ({args.num_per_source} candidates) ...",
                    file=sys.stderr,
                )
                pdf = args.raw_dir / source.local_filename
                if not pdf.exists():
                    print(f"  [{source.slug}] missing PDF at {pdf}; skipping", file=sys.stderr)
                    continue
                chunks = chunk_articles(parse(pdf, source), source_slug=source.slug)
                counts, next_id = _seed_source(
                    source,
                    chunks,
                    num=args.num_per_source,
                    client=client,
                    model=args.model,
                    temperature=args.temperature,
                    draft_fh=draft_fh,
                    rejected_fh=rejected_fh,
                    next_id=next_id,
                )
                total = _SeedCounts(
                    total.accepted + counts.accepted,
                    total.rejected + counts.rejected,
                    total.parse_failed + counts.parse_failed,
                )
    except (httpx.HTTPError, ConnectionError, ollama.ResponseError) as exc:
        print(f"\nfatal: LLM transport error ({exc})", file=sys.stderr)
        return 1

    completed_at = datetime.now(UTC).isoformat(timespec="seconds")
    seed_log_path = args.output_dir / _SEED_LOG_FILENAME
    with seed_log_path.open("a", encoding="utf-8") as log_fh:
        log_fh.write(
            json.dumps(
                {
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "model": args.model,
                    "temperature": args.temperature,
                    "source_args": [s.slug for s in sources],
                    "num_per_source": args.num_per_source,
                    "accepted_count": total.accepted,
                    "rejected_count": total.rejected,
                    "parse_failed_count": total.parse_failed,
                    "git_sha": _git_sha(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    print(
        f"\nwrote {draft_path} (+{total.accepted}), {rejected_path} "
        f"(+{total.rejected} rejected, +{total.parse_failed} parse-failed)",
        file=sys.stderr,
    )
    return 0 if total.accepted > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
