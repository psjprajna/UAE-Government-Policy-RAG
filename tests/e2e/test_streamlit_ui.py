"""Playwright smoke test for the Streamlit demo UI.

Drives a real browser against the Streamlit app launched by
``live_ui_url`` (see ``conftest.py``). Confirms the page renders and one
canonical EN question round-trips end-to-end: the cited ``[1]`` marker
shows up in the answer and at least one ``labour-law-en`` expander row
appears under Citations. Pure UI-rendering coverage — retrieval/
generation correctness is asserted in
``tests/e2e/test_query_pipeline.py``.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

pytestmark = [pytest.mark.adapter_local, pytest.mark.slow]

_PLACEHOLDER = "e.g. What is the annual leave entitlement?"
_QUESTION = "What is the annual leave entitlement?"
_ANSWER_TIMEOUT_MS = 30_000
_PAGE_TIMEOUT_MS = 30_000


def test_page_loads(page: Page, live_ui_url: str) -> None:
    page.goto(live_ui_url)
    expect(page.get_by_role("heading", name="UAE Government Policy RAG")).to_be_visible(
        timeout=_PAGE_TIMEOUT_MS
    )


def test_ask_returns_grounded_answer(page: Page, live_ui_url: str) -> None:
    page.goto(live_ui_url)
    page.get_by_placeholder(_PLACEHOLDER).fill(_QUESTION)
    page.get_by_role("button", name="Ask").click()

    expect(page.get_by_text("[1]", exact=False)).to_be_visible(timeout=_ANSWER_TIMEOUT_MS)
    expect(page.get_by_text("Citations", exact=False)).to_be_visible(timeout=_ANSWER_TIMEOUT_MS)
    expect(page.get_by_text("labour-law-en", exact=False).first).to_be_visible(
        timeout=_ANSWER_TIMEOUT_MS
    )
