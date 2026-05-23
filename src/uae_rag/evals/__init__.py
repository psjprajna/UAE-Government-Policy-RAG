"""Evaluation domain layer for Phase 8 — RAGAS harness."""

from __future__ import annotations

from uae_rag.evals.golden import (
    ExpectedArticle,
    GoldenItem,
    GoldenSetError,
    load_golden,
)
from uae_rag.evals.pipeline import ComposedPipeline, compose_pipeline
from uae_rag.evals.ragas_runner import (
    DEFAULT_METRIC_NAMES,
    CitationRecord,
    ContextRecord,
    EvaluationResult,
    ResultRecord,
    RunConfig,
    build_default_metrics,
    run_evaluation,
)
from uae_rag.evals.wrappers import RagasEmbeddingsWrapper, RagasLLMWrapper

__all__ = [
    "DEFAULT_METRIC_NAMES",
    "CitationRecord",
    "ComposedPipeline",
    "ContextRecord",
    "EvaluationResult",
    "ExpectedArticle",
    "GoldenItem",
    "GoldenSetError",
    "RagasEmbeddingsWrapper",
    "RagasLLMWrapper",
    "ResultRecord",
    "RunConfig",
    "build_default_metrics",
    "compose_pipeline",
    "load_golden",
    "run_evaluation",
]
