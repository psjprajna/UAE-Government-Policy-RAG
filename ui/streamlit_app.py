"""Streamlit demo UI for the UAE Government Policy RAG service.

Single-shot Q&A: type a question, click Ask, see the language-tagged grounded
answer with expandable citations. Talks to the FastAPI ``/query`` endpoint
over HTTP (base URL via ``UAE_RAG_API_BASE``); never imports the API package,
so the hexagonal boundary at ``tests/fitness/test_layer_boundaries.py`` stays
trivially preserved.

Launch with::

    uv run streamlit run ui/streamlit_app.py

Expects ``uv run uvicorn uae_rag.api.main:app`` to be reachable at the
configured base URL.
"""

from __future__ import annotations

import logging

import streamlit as st
from api_client import ApiClient, ApiError, QueryResult

logger = logging.getLogger(__name__)

# Copied verbatim from src/uae_rag/generation/prompts.py:21-22 (kept here
# instead of imported so the UI has no module-level dep on the package).
_REFUSAL_EN = "I don't have enough information in the cited sources to answer that."
_REFUSAL_AR = "لا تتوفر لدي معلومات كافية في المصادر المستشهد بها للإجابة عن هذا السؤال."
_REFUSAL_PHRASES = frozenset({_REFUSAL_EN, _REFUSAL_AR})

_LANGUAGE_FLAG = {"en": "🇬🇧", "ar": "🇦🇪"}


def _is_refusal(result: QueryResult) -> bool:
    return not result.citations and result.answer.strip() in _REFUSAL_PHRASES


def _render_result(result: QueryResult) -> None:
    flag = _LANGUAGE_FLAG.get(result.language, "")
    st.caption(f"lang: {result.language.upper()} {flag}".rstrip())
    if _is_refusal(result):
        st.warning(result.answer)
        return
    st.markdown(result.answer)
    if not result.citations:
        return
    st.subheader("Citations")
    for citation in result.citations:
        header = f"{citation.marker} {citation.source} · Article {citation.article}"
        with st.expander(header):
            st.markdown(
                f"**Source:** `{citation.source}`  \n"
                f"**Article:** {citation.article}  \n"
                f"**In-answer marker:** `{citation.marker}`"
            )


def _render_error(exc: ApiError) -> None:
    st.error(f"**{exc.status_code}** — {exc.detail}")
    if exc.status_code == 503:
        st.caption(
            "Check that the API is running "
            "(`uv run uvicorn uae_rag.api.main:app`) and Ollama is reachable."
        )


def main() -> None:
    st.set_page_config(page_title="UAE Gov Policy RAG", layout="centered")
    st.title("UAE Government Policy RAG")
    st.caption("Ask a question about UAE Labour Law, MOHRE resolutions, or UAE Visa regulations.")

    with st.form("query_form", clear_on_submit=False):
        question = st.text_input(
            "Question",
            max_chars=2000,
            placeholder="e.g. What is the annual leave entitlement?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask")

    if not submitted:
        return
    question = (question or "").strip()
    if not question:
        st.info("Type a question above, then click **Ask**.")
        return

    with st.spinner("Querying the RAG pipeline…"):
        try:
            result = ApiClient().query(question)
        except ApiError as exc:
            _render_error(exc)
            return

    _render_result(result)


main()
