"""Evaluation domain layer for Phase 8 — RAGAS harness."""

from __future__ import annotations

from uae_rag.evals.golden import (
    ExpectedArticle,
    GoldenItem,
    GoldenSetError,
    load_golden,
)
from uae_rag.evals.pipeline import ComposedPipeline, compose_pipeline

__all__ = [
    "ComposedPipeline",
    "ExpectedArticle",
    "GoldenItem",
    "GoldenSetError",
    "compose_pipeline",
    "load_golden",
]
