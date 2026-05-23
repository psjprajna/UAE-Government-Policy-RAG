"""Pure orchestration: walk the golden set, run the pipeline, score with RAGAS.

The runner exists so the CLI script (Slice B3) can be a thin shell — argparse,
disk writes, plot rendering — with the per-question loop, latency capture,
and error containment living in one place that's exercised by both the
integration test and the CLI.

Three structural choices worth flagging:

* **No I/O.** :func:`run_evaluation` never opens a file, never prints, never
  reads env vars (the env-derived bits show up only through ``judge_llm`` and
  via the optional ``RunConfig`` the caller passes in). Disk writes happen in
  ``scripts/run_eval.py``.
* **Per-metric failure isolation.** Per the ``sdk-exception-leak`` lesson,
  one bad judge call (small-model JSON parse failure, transient timeout)
  must not kill the run. RAGAS itself returns NaN for failed metrics when
  ``raise_exceptions=False``; we coerce NaN → ``None`` for JSON-friendliness.
* **Determinism guard.** ADR-0008 requires temperature=0.0 on the answerer.
  We check the documented ``_temperature`` attribute on the Generator at
  startup; future drift fails fast here rather than silently producing a
  baseline number that can't be reproduced.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import ragas

from uae_rag.evals.golden import GoldenItem
from uae_rag.evals.pipeline import ComposedPipeline
from uae_rag.evals.wrappers import RagasEmbeddingsWrapper, RagasLLMWrapper
from uae_rag.ports import EmbeddingsPort, LLMPort, LLMUnavailableError, RetrievalHit

logger = logging.getLogger(__name__)

_RETRIEVE_TOP_K = 5
_LLM_UNAVAILABLE_ERROR_TAG = "LLM unavailable"


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Provenance for a single eval run. Populated by the CLI; carried in the result."""

    adapter_profile: str
    judge_profile: str
    embedder_model_id: str
    reranker_model_id: str
    answerer_model_id: str
    judge_model_id: str
    ragas_version: str
    metrics_used: tuple[str, ...]
    git_sha: str | None = None
    golden_set_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ContextRecord:
    """One retrieved chunk shipped into ``results.json`` verbatim."""

    marker: str
    source: str
    article: str
    chunk_id: str
    text: str


@dataclass(frozen=True, slots=True)
class CitationRecord:
    """One citation emitted by the generator, deduped to ``(source, article)``."""

    marker: str
    source: str
    article: str


@dataclass(frozen=True, slots=True)
class ResultRecord:
    """One row of the eval output — answer + provenance + scores + diagnostics."""

    id: str
    question: str
    ground_truth: str
    language: str
    topic: str
    generated_answer: str
    retrieved_contexts: tuple[ContextRecord, ...]
    citations: tuple[CitationRecord, ...]
    scores: dict[str, float | None]
    latency_ms: dict[str, int]
    errors: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Aggregate output of a full run."""

    records: tuple[ResultRecord, ...]
    config: RunConfig


ProgressCallback = Callable[[int, int, GoldenItem, ResultRecord], None]


def run_evaluation(
    *,
    pipeline: ComposedPipeline,
    golden_items: Sequence[GoldenItem],
    judge_llm: LLMPort,
    judge_embeddings: EmbeddingsPort,
    metrics: Sequence[Any],
    config: RunConfig,
    progress_cb: ProgressCallback | None = None,
) -> EvaluationResult:
    """Run the golden set through the pipeline + RAGAS metrics; return structured records.

    Catches :class:`LLMUnavailableError` per-question (answerer-side) and any
    RAGAS-side exception per-metric, surfacing both through the
    :class:`ResultRecord` ``errors`` and ``scores`` channels.
    """
    _assert_determinism_guard(pipeline)
    wrapped_judge = RagasLLMWrapper(judge_llm)
    wrapped_embeddings = RagasEmbeddingsWrapper(judge_embeddings)
    metric_names = tuple(_metric_name(m) for m in metrics)
    records: list[ResultRecord] = []
    total = len(golden_items)
    for index, item in enumerate(golden_items, start=1):
        record = _evaluate_one(
            item=item,
            pipeline=pipeline,
            metrics=metrics,
            metric_names=metric_names,
            wrapped_judge=wrapped_judge,
            wrapped_embeddings=wrapped_embeddings,
        )
        records.append(record)
        if progress_cb is not None:
            progress_cb(index, total, item, record)
    return EvaluationResult(records=tuple(records), config=config)


def _evaluate_one(
    *,
    item: GoldenItem,
    pipeline: ComposedPipeline,
    metrics: Sequence[Any],
    metric_names: tuple[str, ...],
    wrapped_judge: RagasLLMWrapper,
    wrapped_embeddings: RagasEmbeddingsWrapper,
) -> ResultRecord:
    """Retrieve → generate → judge for one ``GoldenItem``. Failures stay contained."""
    retrieve_ms, hits = _timed(
        lambda: pipeline.retriever.retrieve(item.question, top_k=_RETRIEVE_TOP_K)
    )
    try:
        generate_ms, payload = _timed(lambda: pipeline.generator.generate(item.question, hits))
    except LLMUnavailableError as exc:
        logger.warning("LLM unavailable on %s: %s", item.id, exc)
        return _failed_record(
            item=item,
            metric_names=metric_names,
            retrieve_ms=retrieve_ms,
            error=f"{_LLM_UNAVAILABLE_ERROR_TAG}: {exc}",
        )
    scores, judge_ms, judge_errors = _score_with_ragas(
        question=item.question,
        answer=payload.answer,
        contexts=[h.text for h in hits],
        reference=item.ground_truth,
        metrics=metrics,
        metric_names=metric_names,
        wrapped_judge=wrapped_judge,
        wrapped_embeddings=wrapped_embeddings,
    )
    return ResultRecord(
        id=item.id,
        question=item.question,
        ground_truth=item.ground_truth,
        language=item.language,
        topic=item.topic,
        generated_answer=payload.answer,
        retrieved_contexts=_to_context_records(hits, payload),
        citations=tuple(
            CitationRecord(marker=c.marker, source=c.source, article=c.article)
            for c in payload.citations
        ),
        scores=scores,
        latency_ms={"retrieve": retrieve_ms, "generate": generate_ms, "judge": judge_ms},
        errors=tuple(judge_errors),
    )


def _score_with_ragas(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str,
    metrics: Sequence[Any],
    metric_names: tuple[str, ...],
    wrapped_judge: RagasLLMWrapper,
    wrapped_embeddings: RagasEmbeddingsWrapper,
) -> tuple[dict[str, float | None], int, list[str]]:
    """Single-row RAGAS evaluate; on whole-run failure all metrics → None."""
    from ragas.dataset_schema import EvaluationDataset

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": reference,
            }
        ]
    )
    errors: list[str] = []
    try:
        judge_ms, ragas_result = _timed(
            lambda: ragas.evaluate(
                dataset,
                metrics=list(metrics),
                llm=wrapped_judge,
                embeddings=wrapped_embeddings,
                show_progress=False,
                raise_exceptions=False,
            )
        )
    except Exception as exc:
        # Per the sdk-exception-leak lesson: never propagate a judge failure.
        logger.exception("ragas.evaluate raised on question; recording per-metric null")
        errors.append(f"ragas.evaluate: {type(exc).__name__}: {str(exc)[:200]}")
        return ({name: None for name in metric_names}, 0, errors)
    return (_extract_row_scores(ragas_result, metric_names), judge_ms, errors)


def _extract_row_scores(
    ragas_result: Any, metric_names: tuple[str, ...]
) -> dict[str, float | None]:
    """Map RAGAS's first-row scores dict into our schema, coercing NaN → None."""
    if not ragas_result.scores:
        return {name: None for name in metric_names}
    row = ragas_result.scores[0]
    out: dict[str, float | None] = {}
    for name in metric_names:
        raw = row.get(name)
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            out[name] = None
        else:
            out[name] = float(raw)
    return out


def _to_context_records(hits: Sequence[RetrievalHit], payload: Any) -> tuple[ContextRecord, ...]:
    """Pair retrieved hits with citation markers; missing markers fall back to '[i]'."""
    marker_by_chunk = {c.chunk_id: c.marker for c in payload.citations if hasattr(c, "chunk_id")}
    records: list[ContextRecord] = []
    for i, hit in enumerate(hits, start=1):
        records.append(
            ContextRecord(
                marker=marker_by_chunk.get(hit.chunk_id, f"[{i}]"),
                source=str(hit.metadata.get("source", "")),
                article=str(hit.metadata.get("article", "")),
                chunk_id=hit.chunk_id,
                text=hit.text,
            )
        )
    return tuple(records)


def _failed_record(
    *,
    item: GoldenItem,
    metric_names: tuple[str, ...],
    retrieve_ms: int,
    error: str,
) -> ResultRecord:
    return ResultRecord(
        id=item.id,
        question=item.question,
        ground_truth=item.ground_truth,
        language=item.language,
        topic=item.topic,
        generated_answer="",
        retrieved_contexts=(),
        citations=(),
        scores={name: None for name in metric_names},
        latency_ms={"retrieve": retrieve_ms, "generate": 0, "judge": 0},
        errors=(error,),
    )


def _timed(fn: Callable[[], Any]) -> tuple[int, Any]:
    start = time.perf_counter_ns()
    result = fn()
    return ((time.perf_counter_ns() - start) // 1_000_000, result)


def _metric_name(metric: Any) -> str:
    return getattr(metric, "name", type(metric).__name__)


def _assert_determinism_guard(pipeline: ComposedPipeline) -> None:
    """Fail fast if the generator drifts off temperature=0.0 (ADR-0008)."""
    temperature = getattr(pipeline.generator, "_temperature", None)
    if temperature is not None and temperature != 0.0:
        raise AssertionError(
            f"Generator._temperature must be 0.0 for reproducible eval; got {temperature!r}"
        )


DEFAULT_METRIC_NAMES: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
    "answer_correctness",
    "domain_quality",
)

_ASPECT_CRITIC_DEFINITION = (
    "The answer cites at least one specific UAE legal article number in "
    "bracket notation (e.g. '[1]'), uses formal authoritative language "
    "appropriate for a legal/HR audience, and does not introduce "
    "qualifications or marketing language not supported by the cited "
    "passages."
)


def build_default_metrics(judge_llm: LLMPort, judge_embeddings: EmbeddingsPort) -> tuple[Any, ...]:
    """Instantiate the six production RAGAS metrics against the wrapped judge.

    One source of truth for the CLI runner and the integration test.
    Deprecation warnings from RAGAS 0.4's path migration to ``.collections``
    are suppressed at import time; the old-style classes still work and match
    the spec's metric inventory.
    """
    wrapped_llm = RagasLLMWrapper(judge_llm)
    wrapped_embeddings = RagasEmbeddingsWrapper(judge_embeddings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (
            AnswerCorrectness,
            AnswerRelevancy,
            AspectCritic,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )
    return (
        Faithfulness(llm=wrapped_llm),
        AnswerRelevancy(llm=wrapped_llm, embeddings=wrapped_embeddings),
        LLMContextPrecisionWithReference(llm=wrapped_llm),
        LLMContextRecall(llm=wrapped_llm),
        AnswerCorrectness(llm=wrapped_llm, embeddings=wrapped_embeddings),
        AspectCritic(
            name="domain_quality",
            definition=_ASPECT_CRITIC_DEFINITION,
            llm=wrapped_llm,
        ),
    )


__all__ = [
    "DEFAULT_METRIC_NAMES",
    "CitationRecord",
    "ContextRecord",
    "EvaluationResult",
    "ResultRecord",
    "RunConfig",
    "build_default_metrics",
    "run_evaluation",
]
