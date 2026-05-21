"""Playwright-backed downloader for JS-walled corpus sources.

Used by ``scripts/fetch_corpus.py`` for documents whose hosting portal
serves the PDF through a Cloudflare-style JavaScript challenge that a
plain ``urllib`` request cannot defeat (e.g. uaelegislation.gov.ae).

This adapter is optional — install with ``uv sync --extra fetch`` and
``uv run playwright install chromium`` before use.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF"
DEFAULT_USER_AGENT = "uae-rag-fetcher/0.1 (+playwright)"


class BrowserFetchError(Exception):
    """Raised when the headless browser run fails or the result is not a PDF."""


class BrowserFetcher:
    """Navigate to a direct-download URL with headless chromium and save the resulting PDF.

    The class is constructed lazily — Playwright is imported only when ``download`` runs,
    so the module remains importable on machines that haven't installed the ``fetch`` extra.
    """

    def __init__(self, timeout_seconds: float = 60.0, user_agent: str = DEFAULT_USER_AGENT) -> None:
        self._timeout_ms = int(timeout_seconds * 1000)
        self._user_agent = user_agent

    def download(self, url: str, target: Path) -> None:
        """Fetch ``url`` and write a verified PDF to ``target`` atomically."""
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserFetchError(
                "playwright is not installed. Run `uv sync --extra fetch` "
                "and `uv run playwright install chromium`."
            ) from exc

        tmp = target.with_suffix(target.suffix + ".tmp")
        target.parent.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                context = browser.new_context(
                    accept_downloads=True,
                    user_agent=self._user_agent,
                )
                page = context.new_page()
                with page.expect_download(timeout=self._timeout_ms) as dl_info:
                    try:
                        page.goto(url, timeout=self._timeout_ms)
                    except PlaywrightError as exc:
                        # Direct-download URLs interrupt navigation; that's expected.
                        if "Download is starting" not in str(exc):
                            raise BrowserFetchError(f"navigation failed: {exc}") from exc
                dl_info.value.save_as(tmp)
            finally:
                browser.close()

        head = tmp.read_bytes()[: len(PDF_MAGIC)]
        if head != PDF_MAGIC:
            tmp.unlink(missing_ok=True)
            raise BrowserFetchError(f"response was not a PDF (first bytes: {head!r})")
        if tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            raise BrowserFetchError("0-byte download")
        tmp.replace(target)
        logger.info("browser-fetched %s -> %s (%d bytes)", url, target, target.stat().st_size)


__all__ = ["BrowserFetchError", "BrowserFetcher"]
