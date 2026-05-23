"""Operator CLI for the Phase 8 RAGAS evaluation harness.

Run from ``app/``::

    uv run python scripts/run_eval.py --limit 5

Composes the same ``MultiQuery → Hybrid → Rerank → Generator`` pipeline as
``/query`` (via :func:`uae_rag.evals.compose_pipeline`), scores every golden
question with the six production RAGAS metrics, and writes a self-contained
run directory to ``data/eval_runs/<UTC-timestamp>/`` containing
``results.json`` (verbatim per-question records), ``summary.md`` (mean ± std
per metric with EN/AR + per-topic splits), ``config.json`` (provenance),
``metrics_bar.png`` and ``per_question.png``. The directory is the audit
trail — a human can re-grade any question without replaying.

Exit codes: 0 = success (at least one question scored), 1 = evaluation
failure (warmup fault or every question errored), 2 = bad CLI usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import statistics
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ragas

from uae_rag import config
from uae_rag.evals import (
    DEFAULT_METRIC_NAMES,
    RunConfig,
    build_default_metrics,
    compose_pipeline,
    load_golden,
    run_evaluation,
)
from uae_rag.ports import LLMUnavailableError

logger = logging.getLogger("run_eval")

_APP_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CHROMA_DIR = _APP_ROOT / "data" / "chroma_db"
_DEFAULT_RAW_DIR = _APP_ROOT / "data" / "raw"
_DEFAULT_GOLDEN_PATH = Path("data/golden_set.jsonl")
_DEFAULT_OUTPUT_DIR = Path("data/eval_runs")
_RERANKER_MODEL_ID = "bge-reranker-v2-m3"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer; got {value!r}")
    return parsed


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description=(
            "Run RAGAS evaluation over the golden set and write a self-contained "
            "run directory. Defaults assume the script is invoked from app/."
        ),
    )
    parser.add_argument(
        "--limit", type=_positive_int, default=None, help="evaluate only the first N golden items"
    )
    parser.add_argument(
        "--judge-profile",
        choices=["local", "openai"],
        default="local",
        help="RAGAS judge LLM profile (sets RAGAS_JUDGE_PROFILE env var)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="parent directory; the CLI appends <UTC-timestamp>/ inside",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="skip metrics_bar.png and per_question.png"
    )
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=_DEFAULT_GOLDEN_PATH,
        help="path to the golden-set JSONL",
    )
    return parser.parse_args(argv)


def _compute_golden_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_APP_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _make_run_dir(parent: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = parent / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _progress_line(idx: int, total: int, item: Any, record: Any) -> str:
    head = f"[{idx}/{total}] {item.id} (lang={item.language}, topic={item.topic})"
    if record.errors:
        return f"{head}: ERROR: {record.errors[0][:120]}"
    latency = record.latency_ms
    return (
        f"{head}: retrieve={latency['retrieve']}ms "
        f"generate={latency['generate']}ms judge={latency['judge']}ms ok"
    )


def _print_progress(idx: int, total: int, item: Any, record: Any) -> None:
    print(_progress_line(idx, total, item, record), file=sys.stderr, flush=True)


def _bucket_stats(
    records: list[Any], metric_names: tuple[str, ...]
) -> dict[str, tuple[float, float, int]]:
    stats: dict[str, tuple[float, float, int]] = {}
    for name in metric_names:
        finite = [r.scores[name] for r in records if r.scores.get(name) is not None]
        if not finite:
            stats[name] = (float("nan"), float("nan"), 0)
            continue
        mean = statistics.fmean(finite)
        std = statistics.pstdev(finite) if len(finite) > 1 else 0.0
        stats[name] = (mean, std, len(finite))
    return stats


def _aggregate(records: list[Any], metric_names: tuple[str, ...]) -> dict[str, Any]:
    by_language: dict[str, dict[str, tuple[float, float, int]]] = {}
    for lang in ("en", "ar"):
        subset = [r for r in records if r.language == lang]
        if subset:
            by_language[lang] = _bucket_stats(subset, metric_names)
    by_topic: dict[str, dict[str, tuple[float, float, int]]] = {}
    for topic in ("labour-law", "mohre", "visa"):
        subset = [r for r in records if r.topic == topic]
        if subset:
            by_topic[topic] = _bucket_stats(subset, metric_names)
    ok = sum(1 for r in records if not r.errors)
    return {
        "overall": _bucket_stats(records, metric_names),
        "by_language": by_language,
        "by_topic": by_topic,
        "counts": {"total": len(records), "ok": ok, "errored": len(records) - ok},
    }


def _write_results_json(run_dir: Path, result: Any) -> None:
    rows = [asdict(r) for r in result.records]
    (run_dir / "results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_config_json(
    run_dir: Path,
    run_config: RunConfig,
    args: argparse.Namespace,
    started_at: datetime,
    ended_at: datetime,
) -> None:
    payload = {
        **asdict(run_config),
        "cli_args": {
            "limit": args.limit,
            "judge_profile": args.judge_profile,
            "no_plot": args.no_plot,
            "golden_set": str(args.golden_set),
            "output_dir": str(args.output_dir),
        },
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": int((ended_at - started_at).total_seconds()),
    }
    (run_dir / "config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _stats_table(metric_names: tuple[str, ...], stats: dict[str, tuple[float, float, int]]) -> str:
    lines = ["| Metric | Mean | Std | N |", "|---|---:|---:|---:|"]
    for name in metric_names:
        mean, std, n = stats[name]
        if n == 0:
            lines.append(f"| {name} | — | — | 0 |")
        else:
            lines.append(f"| {name} | {mean:.3f} | {std:.3f} | {n} |")
    return "\n".join(lines)


def _write_summary_md(
    run_dir: Path,
    aggregates: dict[str, Any],
    run_config: RunConfig,
    metric_names: tuple[str, ...],
) -> None:
    counts = aggregates["counts"]
    git_short = (run_config.git_sha or "unknown")[:7]
    lines = [
        f"# Eval run {run_dir.name}",
        "",
        f"- **Adapter profile:** {run_config.adapter_profile}",
        f"- **Judge profile:** {run_config.judge_profile}",
        f"- **Answerer:** {run_config.answerer_model_id} · **Judge:** {run_config.judge_model_id}",
        f"- **Embedder:** {run_config.embedder_model_id} · "
        f"**Reranker:** {run_config.reranker_model_id}",
        f"- **RAGAS:** {run_config.ragas_version} · **Git:** {git_short}",
        f"- **Counts:** {counts['total']} total · {counts['ok']} ok · {counts['errored']} errored",
        "",
        "## Overall (mean ± std)",
        "",
        _stats_table(metric_names, aggregates["overall"]),
        "",
        "## By language",
    ]
    for lang in ("en", "ar"):
        if lang in aggregates["by_language"]:
            lines.append("")
            lines.append(f"### {lang.upper()}")
            lines.append("")
            lines.append(_stats_table(metric_names, aggregates["by_language"][lang]))
    lines.append("")
    lines.append("## By topic")
    for topic in ("labour-law", "mohre", "visa"):
        if topic in aggregates["by_topic"]:
            lines.append("")
            lines.append(f"### {topic}")
            lines.append("")
            lines.append(_stats_table(metric_names, aggregates["by_topic"][topic]))
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(
    run_dir: Path,
    records: list[Any],
    aggregates: dict[str, Any],
    metric_names: tuple[str, ...],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    means = [aggregates["overall"][n][0] for n in metric_names]
    stds = [aggregates["overall"][n][1] for n in metric_names]
    x = np.arange(len(metric_names))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"RAGAS metrics — mean ± std (N={len(records)})")
    fig.tight_layout()
    fig.savefig(run_dir / "metrics_bar.png", dpi=120)
    plt.close(fig)

    matrix = np.array(
        [
            [(r.scores[n] if r.scores.get(n) is not None else np.nan) for n in metric_names]
            for r in records
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(1.1 * len(metric_names) + 2, max(2.0, 0.28 * len(records) + 2)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(records)))
    ax.set_yticklabels([r.id for r in records])
    fig.colorbar(im, ax=ax, label="Score")
    ax.set_title("Per-question scores")
    fig.tight_layout()
    fig.savefig(run_dir / "per_question.png", dpi=120)
    plt.close(fig)


def _build_run_config(
    *,
    pipeline: Any,
    judge_llm: Any,
    args: argparse.Namespace,
) -> RunConfig:
    return RunConfig(
        adapter_profile=os.environ.get("ADAPTER_PROFILE", "local"),
        judge_profile=args.judge_profile,
        embedder_model_id=pipeline.embedder.model_id,
        reranker_model_id=_RERANKER_MODEL_ID,
        answerer_model_id=pipeline.llm.model_id,
        judge_model_id=judge_llm.model_id,
        ragas_version=ragas.__version__,
        metrics_used=DEFAULT_METRIC_NAMES,
        git_sha=_git_sha(),
        golden_set_sha256=_compute_golden_sha256(args.golden_set),
    )


def _run(args: argparse.Namespace) -> int:
    os.environ["RAGAS_JUDGE_PROFILE"] = args.judge_profile

    golden_items = load_golden(args.golden_set)
    if args.limit is not None:
        golden_items = golden_items[: args.limit]
    if not golden_items:
        logger.error("no golden items to evaluate (limit=%s)", args.limit)
        return 1

    try:
        pipeline = compose_pipeline(
            chroma_dir=_DEFAULT_CHROMA_DIR, raw_dir=_DEFAULT_RAW_DIR, warmup=True
        )
    except LLMUnavailableError as exc:
        logger.error("Ollama warmup failed: %s", exc)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(args.output_dir)
    started_at = datetime.now(UTC)

    judge_llm = config.get_judge_llm()
    judge_embeddings = config.get_embeddings()
    metrics = build_default_metrics(judge_llm, judge_embeddings)
    run_config = _build_run_config(pipeline=pipeline, judge_llm=judge_llm, args=args)

    result = run_evaluation(
        pipeline=pipeline,
        golden_items=golden_items,
        judge_llm=judge_llm,
        judge_embeddings=judge_embeddings,
        metrics=metrics,
        config=run_config,
        progress_cb=_print_progress,
    )
    ended_at = datetime.now(UTC)

    aggregates = _aggregate(list(result.records), DEFAULT_METRIC_NAMES)
    _write_results_json(run_dir, result)
    _write_summary_md(run_dir, aggregates, run_config, DEFAULT_METRIC_NAMES)
    _write_config_json(run_dir, run_config, args, started_at, ended_at)
    if not args.no_plot:
        _write_plots(run_dir, list(result.records), aggregates, DEFAULT_METRIC_NAMES)

    counts = aggregates["counts"]
    print(
        f"wrote {run_dir} ({counts['total']} records, "
        f"{counts['ok']} ok, {counts['errored']} errored)",
        file=sys.stderr,
    )
    return 0 if counts["ok"] > 0 else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    try:
        return _run(args)
    except Exception:
        logger.exception("eval run aborted")
        return 1


if __name__ == "__main__":
    sys.exit(main())
