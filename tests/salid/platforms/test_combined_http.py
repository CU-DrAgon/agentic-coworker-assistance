"""Cloud Run: one HTTP port serving /health plus the LINE and WhatsApp webhooks."""

import asyncio

from fastapi.testclient import TestClient

from salid.platforms import line, runtime, whatsapp
from salid.platforms.config import PlatformSettings


class Rec:
    async def handle(self, *a, **k):
        pass

    async def handle_payload(self, *a, **k):
        pass


def combined():
    http = runtime.PlatformsHttp(["slack"])
    http.register(line.WEBHOOK_PATH, line.create_app("line-secret", Rec()), "line")
    http.register(
        whatsapp.WEBHOOK_PATH, whatsapp.create_app("wa-secret", "verify-tok", Rec()), "whatsapp"
    )
    return TestClient(http)


def test_paths_match_the_modules():
    assert set(runtime.WEBHOOK_PATHS) == {line.WEBHOOK_PATH, whatsapp.WEBHOOK_PATH}


def test_health_lists_started_platforms():
    r = combined().get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "platforms": ["slack", "line", "whatsapp"]}


def test_line_signature_failure_is_401():
    c = combined()
    assert c.post(line.WEBHOOK_PATH, content=b"{}").status_code == 401
    assert (
        c.post(line.WEBHOOK_PATH, content=b"{}", headers={"X-Line-Signature": "AAAA"}).status_code
        == 401
    )


def test_whatsapp_verify_challenge():
    c = combined()
    q = {"hub.mode": "subscribe", "hub.verify_token": "verify-tok", "hub.challenge": "123"}
    r = c.get(whatsapp.WEBHOOK_PATH, params=q)
    assert r.status_code == 200 and r.text == "123"
    assert c.get(whatsapp.WEBHOOK_PATH, params={**q, "hub.verify_token": "x"}).status_code == 403


def test_server_runs_without_any_platform_and_unready_webhook_is_503():
    c = TestClient(runtime.PlatformsHttp())
    assert c.get("/health").json() == {"ok": True, "platforms": []}
    assert c.post(line.WEBHOOK_PATH).status_code == 503
    assert c.get("/nope").status_code == 404


async def test_failed_platform_is_dropped_but_not_raised():
    http = runtime.PlatformsHttp(["line", "whatsapp"])

    async def boom():
        raise RuntimeError("down")

    await runtime._guarded("line", boom(), http)
    assert http.platforms == ["whatsapp"]


def test_port_selection(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    assert PlatformSettings(_env_file=None).combined_port == 0  # old behaviour: separate ports
    assert PlatformSettings(_env_file=None, platforms_port=9000).combined_port == 9000
    monkeypatch.setenv("PORT", "8080")
    assert PlatformSettings(_env_file=None).combined_port == 8080


async def test_run_without_port_starts_no_http_server(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    calls = []

    async def fake_serve(*a, **k):
        calls.append("http")

    async def fake_line(a, ps, http=None):
        calls.append(("line", http))

    monkeypatch.setattr(runtime, "serve_http", fake_serve)
    monkeypatch.setattr(
        runtime, "build_assistant", lambda s, ps: type("A", (), {"identity": None})()
    )
    monkeypatch.setattr(runtime, "memory_loop", lambda a: asyncio.sleep(0))
    monkeypatch.setattr(runtime, "run_line", fake_line)
    ps = PlatformSettings(_env_file=None, line_channel_secret="s", line_channel_access_token="t")
    await runtime.run(object(), ps)
    assert calls == [("line", None)]
