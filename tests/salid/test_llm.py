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


def test_chat_agent_model_calls_have_the_request_timeout_and_keep_retries(seeded, settings):
    """The ADK chat agent built its own genai client with no timeout: one stuck Vertex call left
    the answer (and its worker thread) hanging. The timeout rides on every request's config."""
    from conftest import EXEC_ID, GUILD, scripted, text_resp

    from salid.qa import RETRY_OPTIONS, ask, build_agent

    agent = build_agent(settings, [], scoped=False)
    assert agent.model.retry_options is RETRY_OPTIONS  # 429 / 5xx are still retried
    assert agent.generate_content_config.http_options.timeout == llm.REQUEST_TIMEOUT_MS
    model = scripted(text_resp("ok"))
    ask(settings, seeded, "x", guild_id=GUILD, is_exec=True, user_id=EXEC_ID, model=model)
    assert model.requests[0].config.http_options.timeout == llm.REQUEST_TIMEOUT_MS


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


def test_meeting_transcription_uses_the_regional_client():
    from salid.meeting import MeetingManager

    s = Settings(gcp_project="p", gemini_location="global", transcribe_location="asia-southeast1")
    shared = object()
    mgr = MeetingManager(SimpleNamespace(s=s, llm=shared))
    stt = mgr.stt_client
    assert stt is not shared and stt is mgr.stt_client  # created once
    assert stt._api_client.location == "asia-southeast1"
    same = MeetingManager(
        SimpleNamespace(s=s.model_copy(update={"transcribe_location": ""}), llm=shared)
    )
    assert same.stt_client is shared
