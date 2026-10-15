"""Hub API tests: Google token verification is replaced by a fake, no network, no Firestore."""

import pytest
from fastapi.testclient import TestClient

from meeting_scribe import hub, pipeline
from meeting_scribe.app import app
from meeting_scribe.config import get_settings
from salid.models import Receipt
from salid.store import SQLiteStore
from salid.team_settings import TeamSettingsStore

CLIENT_ID = "123-abc.apps.googleusercontent.com"
TOKENS = {
    "good": {"email": "Boss@Example.com", "email_verified": True},
    "stranger": {"email": "x@example.com", "email_verified": True},
    "unverified": {"email": "boss@example.com", "email_verified": False},
}
AUTH = {"Authorization": "Bearer good"}


class FakeFirestore:
    """Just enough Firestore for one document: collection(c).document(d).get()/set()."""

    def __init__(self):
        self.cols = {}

    def collection(self, name):
        docs = self.cols.setdefault(name, {})
        outer = type("Col", (), {"docs": docs})()
        outer.document = lambda doc_id: FakeDoc(docs, doc_id)
        return outer


class FakeDoc:
    def __init__(self, docs, doc_id):
        self.docs, self.id = docs, doc_id

    def get(self):
        data = self.docs.get(self.id)
        return type("Snap", (), {"exists": data is not None, "to_dict": lambda s: data})()

    def set(self, data):
        self.docs[self.id] = dict(data)


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("STORE", "memory")
    monkeypatch.setenv("HUB_GOOGLE_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("HUB_ADMIN_EMAILS", "boss@example.com, other@example.com")
    monkeypatch.setenv("SALID_DISCORD_CLIENT_ID", "123456789012345678")
    get_settings.cache_clear()
    pipeline.get_store.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()
    pipeline.get_store.cache_clear()
    app.dependency_overrides.clear()


@pytest.fixture
def seen():
    return []


@pytest.fixture
def client(env, seen, tmp_path):
    def verifier():
        def verify(token, client_id):
            seen.append((token, client_id))
            if token not in TOKENS:
                raise ValueError("bad token")
            return TOKENS[token]

        return verify

    sqlite = SQLiteStore(str(tmp_path / "l.db"))
    team = TeamSettingsStore(client=FakeFirestore())
    app.dependency_overrides[hub.google_token_verifier] = verifier
    app.dependency_overrides[hub.get_team_store] = lambda: team
    app.dependency_overrides[hub.get_salid_runtime] = lambda: (object(), sqlite)
    c = TestClient(app)
    c.sqlite, c.team = sqlite, team
    return c


# ------------------------------------------------------------------ auth
def test_page_serves_and_injects_client_id(client):
    r = client.get("/hub")
    assert r.status_code == 200 and "accounts.google.com/gsi/client" in r.text
    assert CLIENT_ID in r.text and "__HUB_GOOGLE_CLIENT_ID__" not in r.text


def test_page_rejects_odd_client_id(client, env):
    env.setenv("HUB_GOOGLE_CLIENT_ID", '";alert(1)//')
    get_settings.cache_clear()
    assert 'const CLIENT_ID = "";' in client.get("/hub").text


@pytest.mark.parametrize("path", ["me", "settings", "activity"])
def test_api_requires_a_valid_admin_token(client, path):
    url = f"/api/hub/{path}"
    assert client.get(url).status_code == 401
    assert client.get(url, headers={"Authorization": "Basic x"}).status_code == 401
    assert client.get(url, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get(url, headers={"Authorization": "Bearer stranger"}).status_code == 403
    assert client.get(url, headers={"Authorization": "Bearer unverified"}).status_code == 403
    assert client.get(url, headers=AUTH).status_code == 200


def test_token_is_verified_against_the_configured_client_id(client, seen):
    client.get("/api/hub/me", headers=AUTH)
    assert seen == [("good", CLIENT_ID)]


@pytest.mark.parametrize("client_id,emails", [("", "a@b.c"), (CLIENT_ID, ""), ("", "")])
def test_fails_closed_when_not_configured(client, env, seen, client_id, emails):
    env.setenv("HUB_GOOGLE_CLIENT_ID", client_id)
    env.setenv("HUB_ADMIN_EMAILS", emails)
    get_settings.cache_clear()
    for method, path in [("get", "me"), ("get", "settings"), ("get", "activity")]:
        assert getattr(client, method)(f"/api/hub/{path}", headers=AUTH).status_code == 503
    assert client.post("/api/hub/ask", json={"question": "x"}, headers=AUTH).status_code == 503
    assert client.put("/api/hub/settings", json={}, headers=AUTH).status_code == 503
    assert seen == []  # nothing is even verified


def test_default_verifier_uses_google_with_audience(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "google.oauth2.id_token.verify_oauth2_token",
        lambda token, request, audience: calls.append((token, audience)) or {"email": "a@b.c"},
    )
    assert hub.google_token_verifier()("tok", "cid") == {"email": "a@b.c"}
    assert calls == [("tok", "cid")]


# ------------------------------------------------------------------ endpoints
def test_me_returns_email_and_invite(client, env):
    me = client.get("/api/hub/me", headers=AUTH).json()
    assert me["email"] == "boss@example.com"
    assert "client_id=123456789012345678" in me["invite_url"]
    env.setenv("SALID_DISCORD_CLIENT_ID", "")
    get_settings.cache_clear()
    assert client.get("/api/hub/me", headers=AUTH).json()["invite_url"] is None


def test_settings_defaults_then_round_trip(client):
    s = client.get("/api/hub/settings", headers=AUTH).json()
    assert s["features"] == dict(receipts=True, qa=True, digest=True, deep_tasks=True)
    body = {
        "features": {"digest": False},
        "receipt_channel_ids": ["111", "222"],
        "ask_channel_ids": [],
        "digest_channel_id": "333",
        "digest_time": "07:45",
    }
    put = client.put("/api/hub/settings", json=body, headers=AUTH).json()
    assert put["updated_by"] == "boss@example.com" and put["updated_at"]
    got = client.get("/api/hub/settings", headers=AUTH).json()
    assert got == put and got["features"]["digest"] is False and got["features"]["qa"] is True
    doc = client.team.client.collection("salid_teams").docs["default"]
    assert doc["receipt_channel_ids"] == ["111", "222"]


@pytest.mark.parametrize(
    "bad",
    [{"receipt_channel_ids": ["abc"]}, {"digest_time": "7pm"}, {"digest_channel_id": "1x"}],
)
def test_settings_validation(client, bad):
    assert client.put("/api/hub/settings", json=bad, headers=AUTH).status_code == 422
    assert client.team.client.collection("salid_teams").docs == {}


def test_activity_merges_receipts_and_meetings_newest_first(client):
    for n, (vendor, created, conf) in enumerate(
        [("Makro", "2026-10-14T10:00:00+07:00", 0.9), ("Grab", "2026-10-14T12:00:00+07:00", 0.3)]
    ):
        client.sqlite.add_receipt(
            Receipt(
                submitter_id=1, submitter_name="A", sha256=f"s{n}", vendor=vendor, total=100.0,
                effective_date="2026-10-14", created_at=created, confidence=conf,
            )
        )  # fmt: skip
    meeting = pipeline.get_store().create("Weekly sync")
    items = client.get("/api/hub/activity", headers=AUTH).json()
    assert [i["kind"] for i in items][0] == "meeting" and meeting.created_at.year >= 2026
    rest = [i for i in items if i["kind"] == "receipt"]
    assert [(i["title"], i["status"]) for i in rest] == [("Grab", "review"), ("Makro", "ok")]
    assert items[0]["title"] == "Weekly sync" and items[0]["status"] == "recording"
    assert len(items) == 3


def test_activity_survives_a_broken_receipt_store(client):
    app.dependency_overrides[hub.get_salid_runtime] = lambda: (object(), object())
    pipeline.get_store().create("Only meeting")
    items = client.get("/api/hub/activity", headers=AUTH).json()
    assert [i["kind"] for i in items] == ["meeting"]


def test_ask_runs_salid_qa_as_executive(client, monkeypatch):
    calls = []

    def fake_ask(settings, store, question, *, is_exec, user_id):
        calls.append((question, is_exec))
        return type("A", (), {"text": "ยอดรวม ฿1,000"})()

    monkeypatch.setattr("salid.qa.ask", fake_ask)
    r = client.post("/api/hub/ask", json={"question": " เดือนนี้ใช้เท่าไร "}, headers=AUTH)
    assert r.json() == {"answer": "ยอดรวม ฿1,000"} and calls == [("เดือนนี้ใช้เท่าไร", True)]
    assert client.post("/api/hub/ask", json={"question": ""}, headers=AUTH).status_code == 422
    assert client.post("/api/hub/ask", json={"question": "x"}).status_code == 401


def test_ask_failure_is_a_502_without_details(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internals")

    monkeypatch.setattr("salid.qa.ask", boom)
    r = client.post("/api/hub/ask", json={"question": "x"}, headers=AUTH)
    assert r.status_code == 502 and "secret" not in r.text


def test_home_page_links_to_hub(client):
    assert 'href="/hub"' in client.get("/").text
