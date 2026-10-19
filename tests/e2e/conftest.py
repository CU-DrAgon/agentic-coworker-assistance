"""Browser end-to-end fixtures: the real app on a free local port, the UI in English, no external hosts.

Run with `pytest -m e2e tests/e2e` after `playwright install chromium`; the default run deselects these
tests. The page runs in `?mock=1` mode, so nothing here needs a sign-in, a network or a GCP project."""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn


@pytest.fixture(scope="session")
def live_server() -> Iterator[str]:
    os.environ.setdefault("STORE", "memory")
    from meeting_scribe.app import app  # late: STORE must be set before the app reads its settings

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("the e2e server did not start within 10 s")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(5)


@pytest.fixture(scope="session")
def base_url(live_server: str) -> str:
    """Overrides pytest-playwright's, so `page.goto("/center")` reaches the live server."""
    return live_server


@pytest.fixture(autouse=True)
def block_external(page) -> None:
    """Google sign-in and web fonts stay offline: the mock page needs neither."""
    page.route("https://accounts.google.com/**", lambda route: route.abort())
    page.route("https://fonts.*/**", lambda route: route.abort())


@pytest.fixture(autouse=True)
def english_ui(page) -> None:
    """The UI language is deterministic: the page's saved choice (one key for every Salid page) is
    English before any navigation."""
    page.add_init_script("try { localStorage.setItem('salid-language', 'en'); } catch (e) {}")
