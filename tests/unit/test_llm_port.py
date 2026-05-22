"""Unit tests for ``LLMPort`` — Protocol surface + FakeLLM + OllamaLLM adapter.

Self-contained. ``FakeLLM`` is a deterministic no-daemon LLM that returns a
canned response (or invokes a callable hook) and records the last prompt /
options it saw. It mirrors the role of ``FakeReranker`` in
``test_reranker_port.py`` and will be re-imported by ``test_generation.py``
and ``test_multi_query.py`` in later slices.

The Ollama-specific tests monkeypatch ``ollama.Client.generate`` so they run
without a live daemon: they verify the adapter's input plumbing
(model id, options, keep_alive) and its exception wrapping
(``ollama.ResponseError`` / ``httpx.ConnectError`` / ``httpx.ReadTimeout`` →
``LLMUnavailableError``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import ollama
import pytest

from uae_rag.adapters.local.llm import (
    _DEFAULT_HOST,
    _DEFAULT_KEEP_ALIVE,
    _DEFAULT_MODEL,
    _DEFAULT_TIMEOUT_S,
    OllamaLLM,
)
from uae_rag.ports import LLMPort, LLMUnavailableError

# --- FakeLLM ---------------------------------------------------------------------


class FakeLLM:
    """Deterministic ``LLMPort`` fake.

    Either returns ``canned_response`` verbatim, or — when ``callable_`` is
    supplied — invokes it with the prompt and uses the return value. Records
    ``last_prompt``, ``last_max_output_tokens``, and ``last_temperature`` so
    downstream tests can assert wiring.
    """

    model_id: str = "fake-llm"

    def __init__(
        self,
        *,
        canned_response: str = "",
        callable_: Callable[[str], str] | None = None,
    ) -> None:
        self._canned = canned_response
        self._callable = callable_
        self.last_prompt: str | None = None
        self.last_max_output_tokens: int | None = None
        self.last_temperature: float | None = None
        self.call_count: int = 0

    def generate(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be ≥ 1")
        self.last_prompt = prompt
        self.last_max_output_tokens = max_output_tokens
        self.last_temperature = temperature
        self.call_count += 1
        if self._callable is not None:
            return self._callable(prompt)
        return self._canned


# --- Protocol conformance --------------------------------------------------------


def test_fake_llm_satisfies_port() -> None:
    """Structural fake conforms to the runtime-checkable Protocol."""
    assert isinstance(FakeLLM(), LLMPort)


def test_ollama_llm_satisfies_port() -> None:
    """Real adapter is structurally compliant without contacting the daemon."""
    assert isinstance(OllamaLLM(), LLMPort)


# --- Env-var propagation (no daemon contact) -------------------------------------


def test_env_overrides_propagate_to_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("LOCAL_LLM_HOST", "http://example.local:11500")
    monkeypatch.setenv("LOCAL_LLM_TIMEOUT_S", "30")
    monkeypatch.setenv("LOCAL_LLM_KEEP_ALIVE", "1h")

    adapter = OllamaLLM()

    assert adapter.model_id == "qwen2.5:7b"
    assert adapter.host == "http://example.local:11500"
    assert adapter.timeout_s == 30.0
    assert adapter.keep_alive == "1h"


def test_default_env_vars_match_adr_0008(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset env → defaults reflect ADR-0008 (llama3.1:8b on localhost)."""
    for var in (
        "LOCAL_LLM_MODEL",
        "LOCAL_LLM_HOST",
        "LOCAL_LLM_TIMEOUT_S",
        "LOCAL_LLM_KEEP_ALIVE",
    ):
        monkeypatch.delenv(var, raising=False)

    adapter = OllamaLLM()

    assert adapter.model_id == _DEFAULT_MODEL == "llama3.1:8b"
    assert adapter.host == _DEFAULT_HOST == "http://localhost:11434"
    assert adapter.timeout_s == _DEFAULT_TIMEOUT_S == 120.0
    assert adapter.keep_alive == _DEFAULT_KEEP_ALIVE == "5m"


def test_construction_does_not_contact_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """``OllamaLLM()`` is cheap — the underlying ``ollama.Client`` is lazy."""

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("ollama.Client should not be constructed eagerly")

    monkeypatch.setattr(ollama, "Client", explode)

    adapter = OllamaLLM()

    assert adapter.model_id  # constructed fine
    # ``_client`` is a cached_property — if it never executed, the slot is empty.
    assert "_client" not in adapter.__dict__


# --- FakeLLM behavior (the contract callers depend on) ---------------------------


def test_fake_llm_returns_canned_response() -> None:
    assert FakeLLM(canned_response="hello").generate("anything") == "hello"


def test_fake_llm_invokes_callable_hook() -> None:
    fake = FakeLLM(callable_=lambda p: f"echo:{p}")

    assert fake.generate("ping") == "echo:ping"


def test_fake_llm_records_invocation() -> None:
    fake = FakeLLM(canned_response="ok")

    fake.generate("the prompt", max_output_tokens=128, temperature=0.7)

    assert fake.last_prompt == "the prompt"
    assert fake.last_max_output_tokens == 128
    assert fake.last_temperature == 0.7
    assert fake.call_count == 1


def test_fake_llm_rejects_zero_or_negative_max_output_tokens() -> None:
    fake = FakeLLM(canned_response="ok")

    with pytest.raises(ValueError, match="max_output_tokens"):
        fake.generate("p", max_output_tokens=0)
    with pytest.raises(ValueError, match="max_output_tokens"):
        fake.generate("p", max_output_tokens=-5)


# --- Adapter happy path + plumbing (mocked client) -------------------------------


class _StubResponse:
    """Minimal stand-in for ``ollama.GenerateResponse`` — only ``.response`` is read."""

    def __init__(self, response: str) -> None:
        self.response = response


def test_adapter_returns_stripped_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: adapter forwards prompt + options and strips whitespace."""
    captured: dict[str, Any] = {}

    def fake_generate(self: ollama.Client, **kwargs: Any) -> _StubResponse:
        captured.update(kwargs)
        return _StubResponse(response="  the answer  \n")

    monkeypatch.setattr(ollama.Client, "generate", fake_generate)

    adapter = OllamaLLM()
    result = adapter.generate("a prompt", max_output_tokens=64, temperature=0.2)

    assert result == "the answer"
    assert captured["model"] == adapter.model_id
    assert captured["prompt"] == "a prompt"
    assert captured["stream"] is False
    assert captured["keep_alive"] == adapter.keep_alive
    assert captured["options"]["temperature"] == 0.2
    assert captured["options"]["num_predict"] == 64


def test_adapter_rejects_zero_or_negative_max_output_tokens() -> None:
    """Validation happens before any client construction."""
    adapter = OllamaLLM()

    with pytest.raises(ValueError, match="max_output_tokens"):
        adapter.generate("p", max_output_tokens=0)
    with pytest.raises(ValueError, match="max_output_tokens"):
        adapter.generate("p", max_output_tokens=-1)
    # The client cache stays cold.
    assert "_client" not in adapter.__dict__


# --- Adapter exception wrapping --------------------------------------------------


def test_connect_error_raises_llm_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """``httpx.ConnectError`` → domain ``LLMUnavailableError`` (with cause preserved)."""

    def raise_connect(self: ollama.Client, **kwargs: Any) -> _StubResponse:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(ollama.Client, "generate", raise_connect)

    adapter = OllamaLLM()

    with pytest.raises(LLMUnavailableError) as exc_info:
        adapter.generate("p")

    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)


def test_read_timeout_raises_llm_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """``httpx.ReadTimeout`` → ``LLMUnavailableError`` with a 'timed out' message."""

    def raise_timeout(self: ollama.Client, **kwargs: Any) -> _StubResponse:
        raise httpx.ReadTimeout("read timed out")

    monkeypatch.setattr(ollama.Client, "generate", raise_timeout)

    adapter = OllamaLLM()

    with pytest.raises(LLMUnavailableError, match="timed out"):
        adapter.generate("p")


def test_ollama_response_error_raises_llm_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-200 / API-level failures → ``LLMUnavailableError``."""

    def raise_response_error(self: ollama.Client, **kwargs: Any) -> _StubResponse:
        raise ollama.ResponseError("model not found", status_code=404)

    monkeypatch.setattr(ollama.Client, "generate", raise_response_error)

    adapter = OllamaLLM()

    with pytest.raises(LLMUnavailableError) as exc_info:
        adapter.generate("p")

    assert isinstance(exc_info.value.__cause__, ollama.ResponseError)


def test_llm_unavailable_error_message_omits_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The user-visible message must not leak the configured host into logs."""
    monkeypatch.setenv("LOCAL_LLM_HOST", "http://secret-internal-host:11434")

    def raise_connect(self: ollama.Client, **kwargs: Any) -> _StubResponse:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(ollama.Client, "generate", raise_connect)

    adapter = OllamaLLM()

    with pytest.raises(LLMUnavailableError) as exc_info:
        adapter.generate("p")

    assert "secret-internal-host" not in str(exc_info.value)
