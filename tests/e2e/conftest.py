"""Subprocess launcher for live Streamlit + uvicorn behind Playwright tests.

The ``live_ui_url`` fixture spins up the API on one ephemeral 127.0.0.1
port and Streamlit on another (pointing at the API via
``UAE_RAG_API_BASE``), health-polls both, then issues one warm-up
``/query`` to pay the LLM cold-path tax (~190 s on a fresh Ollama) so
that each Playwright test can use a sane per-action timeout. Teardown
SIGTERMs both children with a SIGKILL fallback.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from tests.e2e.test_query_pipeline import (
    _skip_unless_environment_ready as _skip_unless_ready,
)

_APP_ROOT = Path(__file__).resolve().parents[2]
_HEALTH_TIMEOUT_S = 60.0
_STREAMLIT_TIMEOUT_S = 60.0
_WARMUP_TIMEOUT_S = 240.0
_POLL_INTERVAL_S = 0.5
_TEARDOWN_GRACE_S = 10.0
_WARMUP_QUESTION = "What is the annual leave entitlement?"


def _ephemeral_port() -> int:
    """Bind to port 0 to let the kernel pick an unused port; release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_http(url: str, timeout_s: float) -> bool:
    """Poll url with GET until a sub-500 response or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
        except httpx.HTTPError:
            pass
        else:
            if response.status_code < 500:
                return True
        time.sleep(_POLL_INTERVAL_S)
    return False


def _terminate(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_TEARDOWN_GRACE_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=_TEARDOWN_GRACE_S)


def _launch_api(port: int) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "uae_rag.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=_APP_ROOT,
    )


def _launch_streamlit(port: int, api_port: int) -> subprocess.Popen[bytes]:
    env = {**os.environ, "UAE_RAG_API_BASE": f"http://127.0.0.1:{api_port}"}
    return subprocess.Popen(
        [
            "uv",
            "run",
            "streamlit",
            "run",
            "ui/streamlit_app.py",
            "--server.port",
            str(port),
            "--server.address",
            "127.0.0.1",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=_APP_ROOT,
        env=env,
    )


@pytest.fixture(scope="module")
def live_ui_url() -> Iterator[str]:
    """Yield a live Streamlit URL backed by a warm uvicorn API subprocess."""
    _skip_unless_ready()

    api_port = _ephemeral_port()
    ui_port = _ephemeral_port()

    api_proc: subprocess.Popen[bytes] | None = None
    ui_proc: subprocess.Popen[bytes] | None = None
    try:
        api_proc = _launch_api(api_port)
        if not _wait_for_http(f"http://127.0.0.1:{api_port}/health", _HEALTH_TIMEOUT_S):
            pytest.skip(
                f"uvicorn did not become healthy on port {api_port} within {_HEALTH_TIMEOUT_S}s"
            )

        ui_proc = _launch_streamlit(ui_port, api_port)
        if not _wait_for_http(f"http://127.0.0.1:{ui_port}/", _STREAMLIT_TIMEOUT_S):
            pytest.skip(
                f"streamlit did not become healthy on port {ui_port} within {_STREAMLIT_TIMEOUT_S}s"
            )

        try:
            httpx.post(
                f"http://127.0.0.1:{api_port}/query",
                json={"question": _WARMUP_QUESTION},
                timeout=_WARMUP_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            pytest.skip(f"warm-up /query failed: {exc!r}")

        yield f"http://127.0.0.1:{ui_port}"
    finally:
        _terminate(ui_proc)
        _terminate(api_proc)
