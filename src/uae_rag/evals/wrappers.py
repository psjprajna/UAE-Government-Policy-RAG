"""Adapters from our :mod:`uae_rag.ports` Protocols to RAGAS 0.4 base classes.

RAGAS metrics consume judge calls through :class:`ragas.llms.base.BaseRagasLLM`
and similarity vectors through :class:`ragas.embeddings.base.BaseRagasEmbeddings`.
These shims let any :class:`uae_rag.ports.LLMPort` /
:class:`uae_rag.ports.EmbeddingsPort` implementation plug into RAGAS without
the evaluation layer having to know about langchain or RAGAS internals.

Two design decisions worth surfacing for future maintainers:

* **Temperature is locked at 0.0** (ADR-0008 determinism guard) regardless of
  the value RAGAS passes — judge prompts shouldn't introduce sampling noise
  on top of the answerer's own. The RAGAS 0.4 default is ``0.01``.
* **Async bridges to sync via** :func:`asyncio.to_thread`. Our LLMPort is
  synchronous (the Phase 8.5 async-port promotion is the planned fix); RAGAS
  runs its judge loop on asyncio so we have to bridge somehow. ``to_thread``
  preserves the same execution semantics RAGAS would get from a real async
  LLM client without forcing our adapters to grow an async surface.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from langchain_core.outputs import Generation, LLMResult
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig as RagasRunConfig

from uae_rag.ports import EmbeddingsPort, LLMPort

if TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from langchain_core.prompt_values import PromptValue

logger = logging.getLogger(__name__)

_JUDGE_TEMPERATURE = 0.0
_JUDGE_MAX_OUTPUT_TOKENS = 1024


class RagasLLMWrapper(BaseRagasLLM):
    """Adapt :class:`uae_rag.ports.LLMPort` to RAGAS's judge LLM interface."""

    def __init__(self, llm: LLMPort) -> None:
        super().__init__()
        self._llm = llm
        self.run_config = RagasRunConfig()

    @property
    def model_id(self) -> str:
        return self._llm.model_id

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        del temperature, stop, callbacks  # locked at 0.0; stop tokens unused
        text = self._llm.generate(
            prompt.to_string(),
            max_output_tokens=_JUDGE_MAX_OUTPUT_TOKENS,
            temperature=_JUDGE_TEMPERATURE,
        )
        generations = [[Generation(text=text)] for _ in range(max(n, 1))]
        return LLMResult(generations=generations)

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        return await asyncio.to_thread(
            self.generate_text, prompt, n, temperature, stop, callbacks
        )

    def is_finished(self, response: LLMResult) -> bool:
        del response
        return True

    def set_run_config(self, run_config: RagasRunConfig) -> None:
        self.run_config = run_config


class RagasEmbeddingsWrapper(BaseRagasEmbeddings):
    """Adapt :class:`uae_rag.ports.EmbeddingsPort` to RAGAS's embeddings interface."""

    def __init__(self, embeddings: EmbeddingsPort) -> None:
        super().__init__()
        self._embeddings = embeddings
        self.run_config = RagasRunConfig()

    @property
    def model_id(self) -> str:
        return self._embeddings.model_id

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    def set_run_config(self, run_config: RagasRunConfig) -> None:
        self.run_config = run_config


__all__ = ["RagasEmbeddingsWrapper", "RagasLLMWrapper"]
