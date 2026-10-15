"""salid.llm: request timeout and retry behaviour (offline)."""

from types import SimpleNamespace

import httpx
import pytest

from salid import llm
from salid.config import Settings


class Models:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def generate_content(self, **kw):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_client_has_a_request_timeout():
    client = llm.make_client(Settings(gcp_project="p", gemini_location="global"))
    assert client._api_client._http_options.timeout == llm.REQUEST_TIMEOUT_MS


def test_timeout_is_retried(monkeypatch):
    """A hung call used to block a voice meeting's worker forever; now it times out and retries."""
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    ok = SimpleNamespace(text="hi")
    client = SimpleNamespace(models=Models([httpx.ReadTimeout("slow"), ok]))
    assert llm.generate(client, "m", "x", None) is ok
    assert client.models.calls == 2


def test_timeout_raises_after_all_retries(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    n = len(llm.RETRY_DELAYS) + 1
    client = SimpleNamespace(models=Models([httpx.ReadTimeout("slow")] * n))
    with pytest.raises(httpx.TimeoutException):
        llm.generate(client, "m", "x", None)
    assert client.models.calls == n
