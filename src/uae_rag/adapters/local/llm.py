"""Local LLM adapter — Ollama Python SDK wrapping ``llama3.1:8b``.

Defaults to ``llama3.1:8b`` (ADR-0008). Model id, host, per-request timeout,
and the Ollama ``keep_alive`` duration are read from env vars so a different
local model can be swapped in without editing code:

    LOCAL_LLM_MODEL=qwen2.5:7b
    LOCAL_LLM_HOST=http://localhost:11434
    LOCAL_LLM_TIMEOUT_S=120
    LOCAL_LLM_KEEP_ALIVE=5m

The underlying ``ollama.Client`` is constructed lazily on the first
``generate`` call. Construction is cheap; no daemon contact happens at import
or at ``OllamaLLM()`` time. Mirrors the lazy-load pattern in
``adapters/local/embeddings.SentenceTransformersEmbeddings`` and
``adapters/local/reranker.SentenceTransformersReranker``.

Transport failures (``ollama.ResponseError``, ``httpx.ConnectError``,
``httpx.ReadTimeout``) are wrapped in ``LLMUnavailableError`` with the
original exception preserved on ``__cause__``. The user-visible message omits
the configured host so logs don't leak internal infrastructure.
"""

from __future__ import annotations

import logging
import os
from functools import cached_property

import httpx
import ollama

from uae_rag.ports.llm import LLMUnavailableError

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama3.1:8b"
_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_TIMEOUT_S = 120.0
_DEFAULT_KEEP_ALIVE = "5m"
_DEFAULT_NUM_CTX = 8192


class OllamaLLM:
    """``LLMPort`` implementation backed by ``ollama.Client``.

    Attributes are populated from env vars at construction; the client is not
    instantiated (and the daemon is not contacted) until the first
    ``generate`` call.
    """

    def __init__(self) -> None:
        self.model_id: str = os.environ.get("LOCAL_LLM_MODEL", _DEFAULT_MODEL)
        self.host: str = os.environ.get("LOCAL_LLM_HOST", _DEFAULT_HOST)
        self.timeout_s: float = float(
            os.environ.get("LOCAL_LLM_TIMEOUT_S", str(_DEFAULT_TIMEOUT_S))
        )
        self.keep_alive: str = os.environ.get("LOCAL_LLM_KEEP_ALIVE", _DEFAULT_KEEP_ALIVE)
        self.num_ctx: int = _DEFAULT_NUM_CTX

    @cached_property
    def _client(self) -> ollama.Client:
        logger.info(
            "constructing Ollama client for model %s (timeout=%.1fs, keep_alive=%s)",
            self.model_id,
            self.timeout_s,
            self.keep_alive,
        )
        return ollama.Client(host=self.host, timeout=self.timeout_s)

    def generate(
        self,
        prompt: str,
        *,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be ≥ 1")

        try:
            response = self._client.generate(
                model=self.model_id,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": temperature,
                    "num_predict": max_output_tokens,
                    "num_ctx": self.num_ctx,
                },
                keep_alive=self.keep_alive,
            )
        except httpx.ReadTimeout as exc:
            raise LLMUnavailableError(f"LLM timed out after {self.timeout_s:.0f}s") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError("LLM transport error") from exc
        except ollama.ResponseError as exc:
            raise LLMUnavailableError(f"LLM returned an error (status {exc.status_code})") from exc

        return str(response.response).strip()


__all__ = ["OllamaLLM"]
