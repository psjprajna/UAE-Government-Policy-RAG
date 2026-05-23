"""Unit tests for the RAGAS wrappers and the evaluation runner.

Wrappers: prove the langchain-shaped ``BaseRagasLLM`` / ``BaseRagasEmbeddings``
surfaces delegate to our :class:`LLMPort` / :class:`EmbeddingsPort` exactly.

Runner: prove the per-question loop survives a missing LLM, propagates RAGAS
score failures only to the affected metric, fails fast on a determinism-guard
violation, and feeds the progress callback the right shape. All tests use
inline fakes — no Ollama, no ChromaDB, no real ``ragas.evaluate``.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import StringPromptValue

from uae_rag.evals.golden import ExpectedArticle, GoldenItem
from uae_rag.evals.pipeline import ComposedPipeline
from uae_rag.evals.ragas_runner import (
    EvaluationResult,
    ResultRecord,
    RunConfig,
    run_evaluation,
)
from uae_rag.evals.wrappers import RagasEmbeddingsWrapper, RagasLLMWrapper
from uae_rag.ports import LLMUnavailableError, RetrievalHit

# --- Fakes ---------------------------------------------------------------------


class FakeLLM:
    """Records the prompt and returns either a canned string or a callable result."""

    model_id: str = "fake-llm"

    def __init__(self, *, canned_response: str = "OK") -> None:
        self._canned = canned_response
        self.last_prompt: str | None = None
        self.last_temperature: float | None = None
        self.call_count: int = 0

    def generate(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        self.last_prompt = prompt
        self.last_temperature = temperature
        self.call_count += 1
        return self._canned


class FakeEmbeddings:
    """Deterministic vectors so the wrapper round-trip is verifiable."""

    model_id: str = "fake-embeddings"
    dimension: int = 4
    passage_prefix: str = ""
    query_prefix: str = ""

    def __init__(self) -> None:
        self.last_query: str | None = None
        self.last_documents: list[str] | None = None

    def embed_query(self, text: str) -> list[float]:
        self.last_query = text
        return [0.1, 0.2, 0.3, 0.4]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.last_documents = list(texts)
        return [[float(i)] * self.dimension for i, _ in enumerate(texts)]

    def count_tokens(self, text: str) -> int:
        return len(text.split())


@dataclass
class _FakeCitation:
    marker: str
    source: str
    article: str


@dataclass
class _FakeAnswerPayload:
    answer: str
    citations: list[_FakeCitation]
    language: str = "en"
    prompt_used: str = "stub"


class _FakeRetriever:
    """Returns a canned list of :class:`RetrievalHit` regardless of query."""

    def __init__(self, hits: list[RetrievalHit]) -> None:
        self._hits = hits

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        return list(self._hits[:top_k])


class _FakeGenerator:
    """Yields a canned payload or raises a chosen exception."""

    _temperature: float = 0.0

    def __init__(
        self,
        payload: _FakeAnswerPayload | None = None,
        *,
        raise_exc: Exception | None = None,
        temperature: float = 0.0,
    ) -> None:
        self._payload = payload or _FakeAnswerPayload(
            answer="canned", citations=[_FakeCitation(marker="[1]", source="src", article="1")]
        )
        self._raise = raise_exc
        self._temperature = temperature

    def generate(self, question: str, hits: Any) -> _FakeAnswerPayload:
        if self._raise is not None:
            raise self._raise
        return self._payload


class _FakeMetric:
    """Minimal metric stub the runner just needs ``.name`` from."""

    def __init__(self, name: str) -> None:
        self.name = name


@dataclass
class _FakeRagasResult:
    """Mimics :class:`ragas.dataset_schema.EvaluationResult` for monkeypatching."""

    scores: list[dict[str, float | None]]


def _hit(chunk_id: str = "labour-law-en::art-29", text: str = "Article 29 …") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        text=text,
        metadata={"source": "labour-law-en", "article": "29"},
        score=1.0,
        rank=1,
        source="reranked",
    )


def _golden(item_id: str = "Q01") -> GoldenItem:
    return GoldenItem(
        id=item_id,
        question="What is the annual leave entitlement?",
        ground_truth="Article 29 entitles every worker to 30 days of annual leave.",
        expected_articles=(ExpectedArticle(source="labour-law-en", article="29"),),
        language="en",
        topic="labour-law",
        origin="manual",
    )


def _pipeline(generator: _FakeGenerator | None = None) -> ComposedPipeline:
    return ComposedPipeline(
        retriever=_FakeRetriever([_hit()]),
        generator=generator or _FakeGenerator(),
        embedder=FakeEmbeddings(),
        llm=FakeLLM(),
    )


def _config(metric_names: tuple[str, ...] = ("faithfulness",)) -> RunConfig:
    return RunConfig(
        adapter_profile="local",
        judge_profile="local",
        embedder_model_id="fake-embeddings",
        reranker_model_id="fake-reranker",
        answerer_model_id="fake-llm",
        judge_model_id="fake-llm",
        ragas_version="0.4.3",
        metrics_used=metric_names,
    )


# --- Wrapper tests --------------------------------------------------------------


def test_ragas_llm_wrapper_locks_temperature_at_zero() -> None:
    """RAGAS's default 0.01 must be overridden — ADR-0008 determinism."""
    fake = FakeLLM(canned_response="grounded")
    wrapper = RagasLLMWrapper(fake)

    result = wrapper.generate_text(StringPromptValue(text="question"))

    assert isinstance(result, LLMResult)
    assert result.generations[0][0].text == "grounded"
    assert fake.last_temperature == 0.0
    assert fake.last_prompt == "question"


def test_ragas_llm_wrapper_async_path_delegates_to_sync() -> None:
    fake = FakeLLM(canned_response="async-ok")
    wrapper = RagasLLMWrapper(fake)

    result = asyncio.run(wrapper.agenerate_text(StringPromptValue(text="ping")))

    assert result.generations[0][0].text == "async-ok"
    assert fake.last_prompt == "ping"


def test_ragas_llm_wrapper_is_finished_returns_true() -> None:
    """Our adapter doesn't stream; every response is complete."""
    wrapper = RagasLLMWrapper(FakeLLM())
    result = wrapper.generate_text(StringPromptValue(text="x"))
    assert wrapper.is_finished(result) is True


def test_ragas_embeddings_wrapper_round_trips_through_port() -> None:
    fake = FakeEmbeddings()
    wrapper = RagasEmbeddingsWrapper(fake)

    qvec = wrapper.embed_query("annual leave")
    dvecs = wrapper.embed_documents(["a", "b"])

    assert qvec == [0.1, 0.2, 0.3, 0.4]
    assert dvecs == [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
    assert fake.last_query == "annual leave"
    assert fake.last_documents == ["a", "b"]


def test_ragas_embeddings_wrapper_async_path_delegates_to_sync() -> None:
    fake = FakeEmbeddings()
    wrapper = RagasEmbeddingsWrapper(fake)

    qvec = asyncio.run(wrapper.aembed_query("x"))
    dvecs = asyncio.run(wrapper.aembed_documents(["y"]))

    assert qvec == [0.1, 0.2, 0.3, 0.4]
    assert dvecs == [[0.0, 0.0, 0.0, 0.0]]


# --- Runner tests --------------------------------------------------------------


def test_run_evaluation_happy_path_populates_all_score_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_evaluate(dataset: Any, **kwargs: Any) -> _FakeRagasResult:
        return _FakeRagasResult(scores=[{"faithfulness": 0.83, "answer_relevancy": 0.91}])

    monkeypatch.setattr("uae_rag.evals.ragas_runner.ragas.evaluate", fake_evaluate)
    metrics = [_FakeMetric("faithfulness"), _FakeMetric("answer_relevancy")]

    result = run_evaluation(
        pipeline=_pipeline(),
        golden_items=[_golden("Q01"), _golden("Q02")],
        judge_llm=FakeLLM(),
        judge_embeddings=FakeEmbeddings(),
        metrics=metrics,
        config=_config(("faithfulness", "answer_relevancy")),
    )

    assert isinstance(result, EvaluationResult)
    assert len(result.records) == 2
    for record in result.records:
        assert record.scores == {"faithfulness": 0.83, "answer_relevancy": 0.91}
        assert record.errors == ()
        assert record.generated_answer == "canned"
        assert record.latency_ms.keys() == {"retrieve", "generate", "judge"}


def test_run_evaluation_coerces_nan_scores_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """RAGAS marks a failed metric as NaN; JSON-safe schema needs None."""
    def fake_evaluate(dataset: Any, **kwargs: Any) -> _FakeRagasResult:
        return _FakeRagasResult(scores=[{"faithfulness": math.nan, "answer_relevancy": 0.5}])

    monkeypatch.setattr("uae_rag.evals.ragas_runner.ragas.evaluate", fake_evaluate)
    metrics = [_FakeMetric("faithfulness"), _FakeMetric("answer_relevancy")]

    result = run_evaluation(
        pipeline=_pipeline(),
        golden_items=[_golden()],
        judge_llm=FakeLLM(),
        judge_embeddings=FakeEmbeddings(),
        metrics=metrics,
        config=_config(("faithfulness", "answer_relevancy")),
    )

    assert result.records[0].scores == {"faithfulness": None, "answer_relevancy": 0.5}


def test_run_evaluation_llm_unavailable_records_error_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One LLMUnavailableError marks that record, but later items still process."""
    calls = {"n": 0}

    def fake_evaluate(dataset: Any, **kwargs: Any) -> _FakeRagasResult:
        calls["n"] += 1
        return _FakeRagasResult(scores=[{"faithfulness": 0.5}])

    monkeypatch.setattr("uae_rag.evals.ragas_runner.ragas.evaluate", fake_evaluate)

    class _SometimesFailingGenerator(_FakeGenerator):
        def __init__(self) -> None:
            super().__init__()
            self._calls = 0

        def generate(self, question: str, hits: Any) -> _FakeAnswerPayload:
            self._calls += 1
            if self._calls == 1:
                raise LLMUnavailableError("daemon down")
            return _FakeAnswerPayload(
                answer="recovered",
                citations=[_FakeCitation(marker="[1]", source="src", article="1")],
            )

    pipeline = ComposedPipeline(
        retriever=_FakeRetriever([_hit()]),
        generator=_SometimesFailingGenerator(),
        embedder=FakeEmbeddings(),
        llm=FakeLLM(),
    )

    result = run_evaluation(
        pipeline=pipeline,
        golden_items=[_golden("Q01"), _golden("Q02")],
        judge_llm=FakeLLM(),
        judge_embeddings=FakeEmbeddings(),
        metrics=[_FakeMetric("faithfulness")],
        config=_config(),
    )

    failed, recovered = result.records
    assert failed.scores == {"faithfulness": None}
    assert failed.errors and failed.errors[0].startswith("LLM unavailable")
    assert failed.generated_answer == ""
    assert failed.latency_ms["generate"] == 0
    assert recovered.scores == {"faithfulness": 0.5}
    assert recovered.errors == ()
    assert calls["n"] == 1  # RAGAS only called for the successful question


def test_run_evaluation_metric_failure_records_per_metric_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``ragas.evaluate`` itself raises, every metric in that record is None."""

    def fake_evaluate(dataset: Any, **kwargs: Any) -> _FakeRagasResult:
        raise RuntimeError("judge JSON unparseable")

    monkeypatch.setattr("uae_rag.evals.ragas_runner.ragas.evaluate", fake_evaluate)

    result = run_evaluation(
        pipeline=_pipeline(),
        golden_items=[_golden()],
        judge_llm=FakeLLM(),
        judge_embeddings=FakeEmbeddings(),
        metrics=[_FakeMetric("faithfulness"), _FakeMetric("answer_relevancy")],
        config=_config(("faithfulness", "answer_relevancy")),
    )

    record = result.records[0]
    assert record.scores == {"faithfulness": None, "answer_relevancy": None}
    assert record.errors and record.errors[0].startswith("ragas.evaluate: RuntimeError")


def test_run_evaluation_determinism_guard_rejects_nonzero_temperature() -> None:
    """Future drift away from temperature=0.0 must fail fast at startup."""
    pipeline = ComposedPipeline(
        retriever=_FakeRetriever([_hit()]),
        generator=_FakeGenerator(temperature=0.5),
        embedder=FakeEmbeddings(),
        llm=FakeLLM(),
    )
    with pytest.raises(AssertionError, match=r"temperature must be 0\.0"):
        run_evaluation(
            pipeline=pipeline,
            golden_items=[_golden()],
            judge_llm=FakeLLM(),
            judge_embeddings=FakeEmbeddings(),
            metrics=[_FakeMetric("faithfulness")],
            config=_config(),
        )


def test_run_evaluation_passes_run_config_with_extended_timeout_and_no_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RAGAS executor config must override RAGAS defaults — a regression here
    silently corrupts baseline runs (3 of 6 metrics returned N=0 with the
    default 180s timeout; default 10 retries pushed full-50 runtime to ~12 h)."""
    from ragas.run_config import RunConfig as RagasRunConfig

    captured: dict[str, Any] = {}

    def fake_evaluate(dataset: Any, **kwargs: Any) -> _FakeRagasResult:
        captured.update(kwargs)
        return _FakeRagasResult(scores=[{"faithfulness": 0.5}])

    monkeypatch.setattr("uae_rag.evals.ragas_runner.ragas.evaluate", fake_evaluate)

    run_evaluation(
        pipeline=_pipeline(),
        golden_items=[_golden()],
        judge_llm=FakeLLM(),
        judge_embeddings=FakeEmbeddings(),
        metrics=[_FakeMetric("faithfulness")],
        config=_config(),
    )

    run_config = captured.get("run_config")
    assert isinstance(run_config, RagasRunConfig)
    assert run_config.timeout >= 900
    assert run_config.max_retries == 1


def test_run_evaluation_progress_callback_invoked_per_item(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_evaluate(dataset: Any, **kwargs: Any) -> _FakeRagasResult:
        return _FakeRagasResult(scores=[{"faithfulness": 0.5}])

    monkeypatch.setattr("uae_rag.evals.ragas_runner.ragas.evaluate", fake_evaluate)

    invocations: list[tuple[int, int, str]] = []

    def callback(index: int, total: int, item: GoldenItem, record: ResultRecord) -> None:
        invocations.append((index, total, item.id))
        assert isinstance(record, ResultRecord)

    run_evaluation(
        pipeline=_pipeline(),
        golden_items=[_golden("Q01"), _golden("Q02"), _golden("Q03")],
        judge_llm=FakeLLM(),
        judge_embeddings=FakeEmbeddings(),
        metrics=[_FakeMetric("faithfulness")],
        config=_config(),
        progress_cb=callback,
    )

    assert invocations == [(1, 3, "Q01"), (2, 3, "Q02"), (3, 3, "Q03")]
