"""End-to-end integration test for the Phase 8 RAGAS evaluation harness.

Runs the first 5 golden-set questions through the real composed pipeline
(``MultiQuery → Hybrid → Rerank → Generator``) and scores them with the six
production metrics against the local Ollama judge. Marked ``adapter_local``
and ``slow``; skips cleanly when any of corpus / model / golden set / RAGAS
SDK / Ollama daemon is unavailable.

The faithfulness sanity bound (``mean > 0.3``) is calibration only — NOT a
quality bar. It's there to catch a totally broken pipeline (all answers
hallucinated, every metric NaN), not to enforce a target. The real
calibration is the Slice C baseline number written to ``baseline/``.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pytest

from uae_rag import config
from uae_rag.evals.golden import load_golden
from uae_rag.evals.pipeline import compose_pipeline
from uae_rag.evals.ragas_runner import RunConfig, run_evaluation
from uae_rag.ports import LLMUnavailableError

pytestmark = [pytest.mark.adapter_local, pytest.mark.slow]

_APP_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _APP_ROOT / "data"
_CHROMA_DIR = _DATA_DIR / "chroma_db"
_RAW_DIR = _DATA_DIR / "raw"
_GOLDEN_PATH = _DATA_DIR / "golden_set.jsonl"
_FIRST_N = 5
_FAITHFULNESS_SANITY_BOUND = 0.3
_EXPECTED_METRIC_NAMES = (
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


def _skip_unless_eval_stack_ready() -> None:
    if not _CHROMA_DIR.exists():
        pytest.skip(f"corpus index missing: {_CHROMA_DIR}")
    if not _RAW_DIR.exists():
        pytest.skip(f"raw PDFs missing: {_RAW_DIR}")
    if importlib.util.find_spec("chromadb") is None:
        pytest.skip("chromadb not installed")
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence_transformers not installed")
    if importlib.util.find_spec("ragas") is None:
        pytest.skip("ragas not installed")
    if not _GOLDEN_PATH.exists():
        pytest.skip(f"golden set missing: {_GOLDEN_PATH}")
    items = load_golden(_GOLDEN_PATH)
    if len(items) < _FIRST_N:
        pytest.skip(f"golden set has only {len(items)} rows (<{_FIRST_N})")


def _build_six_metrics(judge_llm, judge_embeddings):
    """Instantiate the six production metrics against wrapped judge + embeddings."""
    from uae_rag.evals.wrappers import RagasEmbeddingsWrapper, RagasLLMWrapper

    wrapped_llm = RagasLLMWrapper(judge_llm)
    wrapped_embeddings = RagasEmbeddingsWrapper(judge_embeddings)

    # Deprecation warnings come from RAGAS 0.4's path migration to .collections;
    # the old-style classes still work and match the spec's metric inventory.
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

    return [
        Faithfulness(llm=wrapped_llm),
        AnswerRelevancy(llm=wrapped_llm, embeddings=wrapped_embeddings),
        LLMContextPrecisionWithReference(llm=wrapped_llm),
        LLMContextRecall(llm=wrapped_llm),
        AnswerCorrectness(llm=wrapped_llm, embeddings=wrapped_embeddings),
        AspectCritic(name="domain_quality", definition=_ASPECT_CRITIC_DEFINITION, llm=wrapped_llm),
    ]


@pytest.fixture(scope="module")
def evaluation_result():
    """Compose the pipeline, run RAGAS on the first 5 golden questions, cache the result."""
    _skip_unless_eval_stack_ready()
    try:
        pipeline = compose_pipeline(chroma_dir=_CHROMA_DIR, raw_dir=_RAW_DIR, warmup=True)
    except LLMUnavailableError as exc:
        pytest.skip(f"Ollama daemon / model unavailable: {exc}")

    import ragas

    judge_llm = config.get_judge_llm()
    judge_embeddings = config.get_embeddings()
    metrics = _build_six_metrics(judge_llm, judge_embeddings)
    items = load_golden(_GOLDEN_PATH)[:_FIRST_N]
    run_config = RunConfig(
        adapter_profile="local",
        judge_profile="local",
        embedder_model_id=pipeline.embedder.model_id,
        reranker_model_id="bge-reranker-v2-m3",
        answerer_model_id=pipeline.llm.model_id,
        judge_model_id=judge_llm.model_id,
        ragas_version=ragas.__version__,
        metrics_used=_EXPECTED_METRIC_NAMES,
    )
    return run_evaluation(
        pipeline=pipeline,
        golden_items=items,
        judge_llm=judge_llm,
        judge_embeddings=judge_embeddings,
        metrics=metrics,
        config=run_config,
    )


def test_run_evaluation_first_five_records_have_full_schema(evaluation_result) -> None:
    """Every record carries all six metric keys, non-empty answer + contexts, latency."""
    assert len(evaluation_result.records) == _FIRST_N
    for record in evaluation_result.records:
        assert set(record.scores.keys()) == set(_EXPECTED_METRIC_NAMES), (
            f"missing metric keys on {record.id}: {record.scores.keys()}"
        )
        assert record.generated_answer, f"empty answer on {record.id}"
        assert record.retrieved_contexts, f"no retrieved contexts on {record.id}"
        assert record.latency_ms.keys() == {"retrieve", "generate", "judge"}
        assert all(v >= 0 for v in record.latency_ms.values())


def test_run_evaluation_first_five_faithfulness_passes_sanity_bound(evaluation_result) -> None:
    """Sanity check: faithfulness mean > 0.3. Calibration only, NOT a quality bar."""
    finite = [
        r.scores["faithfulness"]
        for r in evaluation_result.records
        if r.scores["faithfulness"] is not None
    ]
    if not finite:
        pytest.skip("every faithfulness score was None (judge failed on every question)")
    mean = sum(finite) / len(finite)
    assert mean > _FAITHFULNESS_SANITY_BOUND, (
        f"faithfulness mean {mean:.3f} suggests a broken pipeline "
        f"(every answer ungrounded). Re-run after diagnosing — this is not a "
        f"quality threshold, it's a sanity floor."
    )
