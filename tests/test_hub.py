"""Hub API tests: Google token verification is replaced by a fake, no network, no Firestore."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from meeting_scribe import hub, pipeline
from meeting_scribe.app import app
from meeting_scribe.config import get_settings
from salid import guild_settings as gs
from salid.guild_settings import GuildSettingsUpdate, MemoryGuildStore
from salid.models import Receipt
from salid.store import SQLiteStore

CLIENT_ID = "123-abc.apps.googleusercontent.com"
A, B = 1000, 2000  # two guilds
TOKENS = {
    "boss": {"email": "Boss@Example.com", "email_verified": True},  # admin of A
    "other": {"email": "other@example.com", "email_verified": True},  # admin of B
    "super": {"email": "root@example.com", "email_verified": True},  # HUB_ADMIN_EMAILS
    "stranger": {"email": "x@example.com", "email_verified": True},  # admin of nothing
    "unverified": {"email": "boss@example.com", "email_verified": False},
    "noemail": {"email_verified": True},
}
AUTH = {"Authorization": "Bearer boss"}
OTHER = {"Authorization": "Bearer other"}
SUPER = {"Authorization": "Bearer super"}
STRANGER = {"Authorization": "Bearer stranger"}


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("STORE", "memory")
    monkeypatch.setenv("HUB_GOOGLE_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("HUB_ADMIN_EMAILS", "root@example.com")
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
    guilds = MemoryGuildStore()
    for gid, name in ((A, "ร้านกาแฟ"), (B, "Bar B")):
        guilds.upsert_guild(
            gid,
            name=name,
            icon_url=f"https://cdn/{gid}.png",
            channels=[{"id": str(gid + 1), "name": "ใบเสร็จ"}, {"id": str(gid + 2), "name": "chat"}],
            installed=True,
        )
    guilds.redeem_link_code(guilds.create_link_code(A, 1, "u"), "boss@example.com")
    guilds.redeem_link_code(guilds.create_link_code(B, 2, "v"), "other@example.com")
    app.dependency_overrides[hub.google_token_verifier] = verifier
    app.dependency_overrides[hub.get_guild_store] = lambda: guilds
    app.dependency_overrides[hub.get_salid_runtime] = lambda: (object(), sqlite)
    c = TestClient(app)
    c.sqlite, c.guilds = sqlite, guilds
    return c


def add_receipt(client, gid, vendor, created="2026-10-14T10:00:00+07:00", conf=0.9, sha=None):
    client.sqlite.add_receipt(
        Receipt(
            guild_id=gid, submitter_id=1, submitter_name="A", sha256=sha or f"{gid}-{vendor}",
            vendor=vendor, total=100.0, effective_date="2026-10-14", created_at=created,
            confidence=conf,
        )
    )  # fmt: skip


# ------------------------------------------------------------------ auth
def test_page_serves_and_injects_client_id(client):
    r = client.get("/hub")
    assert r.status_code == 200 and "accounts.google.com/gsi/client" in r.text
    assert CLIENT_ID in r.text and "__HUB_GOOGLE_CLIENT_ID__" not in r.text
    assert r.headers["referrer-policy"] == "no-referrer"  # the page URL may carry a link code


def test_page_rejects_odd_client_id(client, env):
    env.setenv("HUB_GOOGLE_CLIENT_ID", '";alert(1)//')
    get_settings.cache_clear()
    assert 'const CLIENT_ID = "";' in client.get("/hub").text


GUILD_PATHS = [f"/api/hub/guilds/{A}/{p}" for p in ("settings", "activity")]


@pytest.mark.parametrize("url", ["/api/hub/me", "/api/hub/guilds", *GUILD_PATHS])
def test_api_requires_a_valid_google_token(client, url):
    assert client.get(url).status_code == 401
    assert client.get(url, headers={"Authorization": "Basic x"}).status_code == 401
    assert client.get(url, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get(url, headers={"Authorization": "Bearer unverified"}).status_code == 403
    assert client.get(url, headers={"Authorization": "Bearer noemail"}).status_code == 403
    assert client.get(url, headers=AUTH).status_code == 200


def test_any_verified_google_account_may_sign_in(client):
    assert client.get("/api/hub/me", headers=STRANGER).json()["email"] == "x@example.com"
    assert client.get("/api/hub/guilds", headers=STRANGER).json() == []


def test_token_is_verified_against_the_configured_client_id(client, seen):
    client.get("/api/hub/me", headers=AUTH)
    assert seen == [("boss", CLIENT_ID)]


def test_fails_closed_without_a_google_client_id(client, env, seen):
    env.setenv("HUB_GOOGLE_CLIENT_ID", "")
    get_settings.cache_clear()
    for path in ("me", "guilds", f"guilds/{A}/settings", f"guilds/{A}/activity"):
        assert client.get(f"/api/hub/{path}", headers=AUTH).status_code == 503
    assert (
        client.post(f"/api/hub/guilds/{A}/ask", json={"question": "x"}, headers=AUTH).status_code
        == 503
    )
    assert client.post("/api/hub/link", json={"code": "x" * 20}, headers=AUTH).status_code == 503
    assert client.put(f"/api/hub/guilds/{A}/settings", json={}, headers=AUTH).status_code == 503
    assert seen == []  # nothing is even verified


def test_admin_emails_setting_is_optional(client, env):
    env.setenv("HUB_ADMIN_EMAILS", "")
    get_settings.cache_clear()
    assert client.get("/api/hub/guilds", headers=AUTH).status_code == 200


def test_default_verifier_uses_google_with_audience(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "google.oauth2.id_token.verify_oauth2_token",
        lambda token, request, audience: calls.append((token, audience)) or {"email": "a@b.c"},
    )
    assert hub.google_token_verifier()("tok", "cid") == {"email": "a@b.c"}
    assert calls == [("tok", "cid")]


# ------------------------------------------------------------------ me / guild list
def test_me_returns_email_invite_and_super_flag(client, env):
    me = client.get("/api/hub/me", headers=AUTH).json()
    assert me["email"] == "boss@example.com" and me["is_super"] is False
    assert "client_id=123456789012345678" in me["invite_url"]
    assert client.get("/api/hub/me", headers=SUPER).json()["is_super"] is True
    env.setenv("SALID_DISCORD_CLIENT_ID", "")
    get_settings.cache_clear()
    assert client.get("/api/hub/me", headers=AUTH).json()["invite_url"] is None


def test_guild_list_shows_only_guilds_the_user_administers(client):
    got = client.get("/api/hub/guilds", headers=AUTH).json()
    assert got == [
        {"id": str(A), "name": "ร้านกาแฟ", "icon_url": f"https://cdn/{A}.png", "active": True}
    ]
    assert [g["id"] for g in client.get("/api/hub/guilds", headers=OTHER).json()] == [str(B)]
    assert [g["id"] for g in client.get("/api/hub/guilds", headers=SUPER).json()] == [
        str(B),
        str(A),
    ]  # super admins see everything (sorted by name)
    client.guilds.set_active(A, False)
    assert client.get("/api/hub/guilds", headers=AUTH).json()[0]["active"] is False


# ------------------------------------------------------------------ link flow
def link(client, code, headers=STRANGER):
    return client.post("/api/hub/link", json={"code": code}, headers=headers)


def test_link_code_makes_the_signed_in_account_a_guild_admin(client):
    code = client.guilds.create_link_code(A, 77, "Boss#1")
    assert client.get(f"/api/hub/guilds/{A}/settings", headers=STRANGER).status_code == 403
    r = link(client, code)
    assert r.status_code == 200 and r.json()["id"] == str(A) and r.json()["name"] == "ร้านกาแฟ"
    assert "x@example.com" in client.guilds.get(A).admin_emails
    assert client.get(f"/api/hub/guilds/{A}/settings", headers=STRANGER).status_code == 200
    assert [g["id"] for g in client.get("/api/hub/guilds", headers=STRANGER).json()] == [str(A)]
    assert client.get(f"/api/hub/guilds/{B}/settings", headers=STRANGER).status_code == 403


def test_link_code_works_once_only(client):
    code = client.guilds.create_link_code(A, 77, "Boss#1")
    assert link(client, code).status_code == 200
    assert link(client, code, OTHER).status_code == 400
    assert "other@example.com" in client.guilds.get(B).admin_emails  # still only B
    assert "other@example.com" not in client.guilds.get(A).admin_emails


def test_expired_unknown_and_malformed_codes_are_rejected(client, monkeypatch):
    code = client.guilds.create_link_code(A, 77, "u")
    real = gs._now()
    monkeypatch.setattr(gs, "_now", lambda: real + timedelta(minutes=16))
    assert link(client, code).status_code == 400
    assert link(client, "does-not-exist-123456").status_code == 400
    assert link(client, "short").status_code == 422
    assert link(client, code, {}).status_code == 401  # needs a Google sign-in first
    assert "x@example.com" not in client.guilds.get(A).admin_emails


# ------------------------------------------------------------------ guild-scoped access
@pytest.mark.parametrize(
    "method,path,body",
    [
        ("get", "settings", None),
        ("put", "settings", {}),
        ("get", "activity", None),
        ("post", "ask", {"question": "x"}),
    ],
)
def test_a_guild_admin_cannot_touch_another_guild(client, monkeypatch, method, path, body):
    asked = []
    monkeypatch.setattr("salid.qa.ask", lambda *a, **k: asked.append(1))
    url = f"/api/hub/guilds/{B}/{path}"
    for who in (AUTH, STRANGER):
        assert (
            getattr(client, method)(
                url, headers=who, **({"json": body} if body else {})
            ).status_code
            == 403
        )
    assert not asked and client.guilds.get(B).updated_at == ""


def test_unknown_guild_is_403_for_users_and_404_for_super_admins(client):
    url = "/api/hub/guilds/31337/settings"
    assert client.get(url, headers=AUTH).status_code == 403
    assert client.get(url, headers=SUPER).status_code == 404
    assert client.put(url, json={}, headers=SUPER).status_code == 404
    assert client.guilds.find(31337) is None  # never created by a PUT


def test_super_admin_can_open_any_guild(client):
    for gid in (A, B):
        assert client.get(f"/api/hub/guilds/{gid}/settings", headers=SUPER).status_code == 200


# ------------------------------------------------------------------ settings
def test_settings_defaults_round_trip_and_layout(client):
    s = client.get(f"/api/hub/guilds/{A}/settings", headers=AUTH).json()
    assert s["features"] == dict(
        receipts=True,
        qa=True,
        digest=True,
        deep_tasks=True,
        voice_meetings=True,
        auto_join_meetings=True,
        memory=True,
        image_memory=True,
        media_memory=True,
        doc_memory=True,
        read_receipts=True,
        web=True,
        documents=True,
        scheduling=True,
        email=True,
        followups=True,
        reports=True,
        briefings=True,
        polls=True,
    )
    assert s["exec_role_names"] == ["ผู้บริหาร", "Executive"] and s["guild_name"] == "ร้านกาแฟ"
    assert [c["name"] for c in s["channels"]] == ["ใบเสร็จ", "chat"]
    body = {
        "features": {"digest": False},
        "receipt_channel_ids": [str(A + 1)],
        "ask_channel_ids": [],
        "digest_channel_id": str(A + 2),
        "digest_time": "07:45",
        "exec_role_names": ["หัวหน้า", "Executive"],
    }
    put = client.put(f"/api/hub/guilds/{A}/settings", json=body, headers=AUTH).json()
    assert put["updated_by"] == "boss@example.com" and put["updated_at"]
    got = client.get(f"/api/hub/guilds/{A}/settings", headers=AUTH).json()
    assert got == put and got["features"]["digest"] is False and got["features"]["qa"] is True
    doc = client.guilds.client.collection("salid_guilds").docs[str(A)]
    assert doc["receipt_channel_ids"] == [str(A + 1)] and doc["exec_role_names"][0] == "หัวหน้า"
    assert client.guilds.get(B).updated_at == ""  # the other guild is untouched


def test_hub_cannot_change_admins_channels_or_active_flag(client):
    body = {"admin_emails": ["evil@x.com"], "channels": [], "active": False, "guild_name": "pwned"}
    client.put(f"/api/hub/guilds/{A}/settings", json=body, headers=AUTH)
    g = client.guilds.get(A)
    assert g.admin_emails == ["boss@example.com"] and g.active and g.guild_name == "ร้านกาแฟ"
    assert len(g.channels) == 2


@pytest.mark.parametrize(
    "bad",
    [
        {"receipt_channel_ids": ["abc"]},
        {"digest_time": "7pm"},
        {"digest_channel_id": "1x"},
        {"exec_role_names": ["x" * 101]},
        {"receipt_channel_ids": ["999"]},  # numeric but not a channel of this guild
        {"ask_channel_ids": [str(B + 1)]},  # a channel of the OTHER guild
        {"digest_channel_id": "12345"},
        {"followup_time": "25:00"},
        {"report_time": "6pm"},
        {"report_channel_id": "1x"},
        {"report_channel_id": "12345"},  # numeric but not a channel of this guild
    ],
)
def test_settings_validation(client, bad):
    r = client.put(f"/api/hub/guilds/{A}/settings", json=bad, headers=AUTH)
    assert r.status_code == 422
    assert client.guilds.get(A).updated_at == ""


def test_hub_saves_coordinator_settings_and_features(client):
    body = {
        "features": {"followups": False, "reports": False},
        "followup_time": "09:15",
        "report_time": "19:00",
        "report_channel_id": str(A + 2),
    }
    put = client.put(f"/api/hub/guilds/{A}/settings", json=body, headers=AUTH).json()
    assert put["followup_time"] == "09:15" and put["report_time"] == "19:00"
    assert put["report_channel_id"] == str(A + 2)
    assert put["features"]["followups"] is False and put["features"]["reports"] is False
    assert client.guilds.get(B).followup_time == "10:00"  # defaults elsewhere: 10:00 / 18:30
    assert client.guilds.get(B).report_time == "18:30" and client.guilds.get(B).features.followups


def test_hub_page_has_both_languages_for_the_coordinator_controls():
    import json
    import re
    from pathlib import Path

    html = (Path(__file__).parent.parent / "src/meeting_scribe/static/hub.html").read_text()
    for key in (
        "f_followups_n", "f_followups_d", "f_followups_w", "f_reports_n", "f_reports_d",
        "f_reports_w", "followup_time_l", "report_time_l", "report_id_l", "report_auto",
        "report_id_s", "f_briefings_n", "f_briefings_d", "f_briefings_w", "morning_time_l",
        "evening_time_l",
    ):  # fmt: skip
        m = re.search(rf'"{key}": (\{{.*?\}})', html, re.S)
        assert m, key
        d = json.loads(m.group(1))
        assert d["th"] and d["en"], key
    for ident in ("followup-time", "report-time", "report-id", "morning-time", "evening-time"):
        assert f'id="{ident}"' in html
    assert 'id: "followups"' in html and 'id: "reports"' in html and 'id: "briefings"' in html


def test_hub_saves_channel_modes_and_default_mode(client):
    body = {"channel_modes": {str(A + 1): "listen", str(A + 2): "join"}, "default_mode": "chat"}
    put = client.put(f"/api/hub/guilds/{A}/settings", json=body, headers=AUTH).json()
    assert put["channel_modes"] == body["channel_modes"] and put["default_mode"] == "chat"
    assert client.guilds.get(A).channel_modes == {str(A + 1): "listen", str(A + 2): "join"}
    # saving again without a channel removes it (the hub sends the full map)
    client.put(
        f"/api/hub/guilds/{A}/settings", json={"channel_modes": {str(A + 2): "off"}}, headers=AUTH
    )
    g = client.guilds.get(A)
    assert g.channel_modes == {str(A + 2): "off"} and g.default_mode == "call"
    assert client.guilds.get(B).channel_modes == {}


@pytest.mark.parametrize(
    "bad",
    [
        {"channel_modes": {"999": "join"}},  # not a channel of this guild
        {"channel_modes": {str(B + 1): "join"}},  # a channel of the OTHER guild
    ],
)
def test_hub_rejects_modes_for_unknown_channels(client, bad):
    r = client.put(f"/api/hub/guilds/{A}/settings", json=bad, headers=AUTH)
    assert r.status_code == 422 and client.guilds.get(A).channel_modes == {}


def test_hub_drops_unknown_mode_values(client):
    body = {"channel_modes": {str(A + 1): "bogus", str(A + 2): "off"}, "default_mode": "bogus"}
    put = client.put(f"/api/hub/guilds/{A}/settings", json=body, headers=AUTH).json()
    assert put["channel_modes"] == {str(A + 2): "off"} and put["default_mode"] == "call"


def test_hub_saves_channel_languages_and_default_language(client):
    body = {"channel_languages": {str(A + 1): "en", str(A + 2): "auto"}, "default_language": "en"}
    put = client.put(f"/api/hub/guilds/{A}/settings", json=body, headers=AUTH).json()
    assert put["channel_languages"] == body["channel_languages"] and put["default_language"] == "en"
    assert client.guilds.get(A).channel_languages == body["channel_languages"]
    # the hub sends the full map: a channel left out is removed, the default falls back to th
    client.put(
        f"/api/hub/guilds/{A}/settings",
        json={"channel_languages": {str(A + 2): "th"}},
        headers=AUTH,
    )
    g = client.guilds.get(A)
    assert g.channel_languages == {str(A + 2): "th"} and g.default_language == "th"
    assert client.guilds.get(B).channel_languages == {}


@pytest.mark.parametrize(
    "bad",
    [
        {"channel_languages": {"999": "en"}},  # not a channel of this guild
        {"channel_languages": {str(B + 1): "en"}},  # a channel of the OTHER guild
    ],
)
def test_hub_rejects_languages_for_unknown_channels(client, bad):
    r = client.put(f"/api/hub/guilds/{A}/settings", json=bad, headers=AUTH)
    assert r.status_code == 422 and client.guilds.get(A).channel_languages == {}


def test_hub_drops_unknown_language_values(client):
    body = {"channel_languages": {str(A + 1): "fr", str(A + 2): "en"}, "default_language": "fr"}
    put = client.put(f"/api/hub/guilds/{A}/settings", json=body, headers=AUTH).json()
    assert put["channel_languages"] == {str(A + 2): "en"} and put["default_language"] == "th"


def test_hub_page_has_the_language_selectors(client):
    html = client.get("/hub").text
    assert 'id="default-language"' in html and "channel_languages" in html
    assert "lang-sel" in html and "default_language" in html


def test_hub_page_has_the_mode_section(client):
    html = client.get("/hub").text
    assert "โหมดของแต่ละห้อง" in html and 'id="default-mode"' in html and "channel_modes" in html


def test_any_numeric_channel_is_accepted_until_the_bot_has_synced_the_list(client):
    client.guilds.upsert_guild(A, name="n", icon_url="", channels=[])
    r = client.put(
        f"/api/hub/guilds/{A}/settings", json={"receipt_channel_ids": ["12345"]}, headers=AUTH
    )
    assert r.status_code == 200


# ------------------------------------------------------------------ activity
def test_activity_shows_only_this_guilds_receipts_newest_first(client):
    add_receipt(client, A, "Makro", "2026-10-14T10:00:00+07:00")
    add_receipt(client, A, "Grab", "2026-10-14T12:00:00+07:00", conf=0.3)
    add_receipt(client, B, "SecretBarTab", "2026-10-14T13:00:00+07:00")
    items = client.get(f"/api/hub/guilds/{A}/activity", headers=AUTH).json()
    assert [(i["title"], i["status"]) for i in items] == [("Grab", "review"), ("Makro", "ok")]
    assert "SecretBarTab" not in str(items)
    items_b = client.get(f"/api/hub/guilds/{B}/activity", headers=OTHER).json()
    assert [i["title"] for i in items_b] == ["SecretBarTab"]


def test_meetings_are_only_shown_to_super_admins(client):
    # Meetings are not per guild yet, so a guild admin must never see them.
    add_receipt(client, A, "Makro")
    pipeline.get_store().create("Weekly sync")
    assert [
        i["kind"] for i in client.get(f"/api/hub/guilds/{A}/activity", headers=AUTH).json()
    ] == ["receipt"]
    kinds = [i["kind"] for i in client.get(f"/api/hub/guilds/{A}/activity", headers=SUPER).json()]
    assert kinds[0] == "meeting" and "receipt" in kinds


def test_activity_survives_a_broken_receipt_store(client):
    app.dependency_overrides[hub.get_salid_runtime] = lambda: (object(), object())
    pipeline.get_store().create("Only meeting")
    items = client.get(f"/api/hub/guilds/{A}/activity", headers=SUPER).json()
    assert [i["kind"] for i in items] == ["meeting"]


# ------------------------------------------------------------------ ask
def test_ask_runs_salid_qa_as_executive_of_that_guild_only(client, monkeypatch):
    add_receipt(client, A, "Makro")
    add_receipt(client, B, "SecretBarTab")
    calls = []

    def fake_ask(settings, store, question, *, guild_id, is_exec, user_id):
        calls.append((question, guild_id, is_exec))
        vendors = [r.vendor for r in store.recent(guild_id, 50)]  # what Q&A could read
        return type("A", (), {"text": ",".join(vendors)})()

    monkeypatch.setattr("salid.qa.ask", fake_ask)
    r = client.post(f"/api/hub/guilds/{A}/ask", json={"question": " เดือนนี้ใช้เท่าไร "}, headers=AUTH)
    assert r.json() == {"answer": "Makro"} and calls == [("เดือนนี้ใช้เท่าไร", A, True)]
    r = client.post(f"/api/hub/guilds/{B}/ask", json={"question": "x"}, headers=OTHER)
    assert r.json() == {"answer": "SecretBarTab"} and calls[-1][1] == B
    assert (
        client.post(f"/api/hub/guilds/{A}/ask", json={"question": ""}, headers=AUTH).status_code
        == 422
    )
    assert client.post(f"/api/hub/guilds/{A}/ask", json={"question": "x"}).status_code == 401


def test_ask_failure_is_a_502_without_details(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internals")

    monkeypatch.setattr("salid.qa.ask", boom)
    r = client.post(f"/api/hub/guilds/{A}/ask", json={"question": "x"}, headers=AUTH)
    assert r.status_code == 502 and "secret" not in r.text


def test_old_single_team_endpoints_are_gone(client):
    for path in ("settings", "activity"):
        assert client.get(f"/api/hub/{path}", headers=AUTH).status_code == 404


def test_home_page_links_to_hub(client):
    assert 'href="/hub"' in client.get("/").text


def test_update_model_is_what_the_hub_writes():
    assert set(GuildSettingsUpdate.model_fields) == {
        "features", "receipt_channel_ids", "ask_channel_ids", "digest_channel_id", "receipt_home_channel_id", "receipt_forward",
        "digest_time", "exec_role_names", "channel_modes", "default_mode", "channel_languages", "default_language", "memory_retention_days",
        "followup_time", "report_time", "report_channel_id", "morning_brief_time", "evening_brief_time",
    }  # fmt: skip


def test_hub_saves_and_validates_memory_retention(client):
    url = f"/api/hub/guilds/{A}/settings"
    assert client.get(url, headers=AUTH).json()["memory_retention_days"] == 90
    put = client.put(
        url, json={"memory_retention_days": 30, "features": {"memory": False}}, headers=AUTH
    )
    assert put.json()["memory_retention_days"] == 30 and put.json()["features"]["memory"] is False
    assert client.guilds.get(A).memory_retention_days == 30
    assert client.guilds.get(B).memory_retention_days == 90  # per guild
    for bad in (0, 3651, -3, "abc", None):  # invalid values fall back to the default
        got = client.put(url, json={"memory_retention_days": bad}, headers=AUTH).json()
        assert got["memory_retention_days"] == 90
    assert (
        client.put(url, json={"memory_retention_days": "365"}, headers=AUTH).json()[
            "memory_retention_days"
        ]
        == 365
    )


def test_hub_page_has_the_memory_controls(client):
    html = (
        __import__("pathlib").Path("src/meeting_scribe/static/hub.html").read_text(encoding="utf-8")
    )
    assert 'id="mem-days"' in html and 'id: "memory"' in html and "memory_retention_days" in html


def test_hub_memory_status_shows_import_progress(client):
    from salid.memory_store import ChannelState, GuildMemoryState, SQLiteMemoryStore

    mem = SQLiteMemoryStore(":memory:")
    mem.put_channel_state(ChannelState(guild_id=A, channel_id=1, complete=True, messages=8421))
    mem.put_channel_state(ChannelState(guild_id=A, channel_id=2, messages=10))
    today = datetime.now(UTC).date().isoformat()
    mem.put_guild_state(GuildMemoryState(guild_id=A, day=today, images_done=120, channels_total=15))
    app.dependency_overrides[hub.get_memory_store] = lambda: mem
    app.dependency_overrides[hub.get_salid_runtime] = lambda: (SimpleNamespace(), client.sqlite)
    url = f"/api/hub/guilds/{A}/memory-status"
    got = client.get(url, headers=AUTH).json()
    assert got["channels_done"] == 1 and got["channels_total"] == 15 and got["messages"] == 8431
    assert got["images_cap"] == 300 and got["images_done"] == 120
    assert client.get(url, headers=OTHER).status_code == 403  # other guild's admin
    assert client.get(f"/api/hub/guilds/{B}/memory-status", headers=AUTH).status_code == 403


# ------------------------------------------------------------------ channels Salid cannot see
def test_settings_expose_channel_visibility_for_the_hidden_channels_card(client):
    client.guilds.upsert_guild(
        A,
        name="n",
        icon_url="",
        channels=[
            {"id": "1", "name": "ok"},  # old synced data without the flags counts as visible
            {"id": "2", "name": "secret", "visible": False, "readable_history": True},
            {"id": "3", "name": "no-history", "visible": True, "readable_history": False},
        ],
    )
    chans = client.get(f"/api/hub/guilds/{A}/settings", headers=AUTH).json()["channels"]
    assert [(c["name"], c["visible"], c["readable_history"]) for c in chans] == [
        ("ok", True, True),
        ("secret", False, True),
        ("no-history", True, False),
    ]


def test_hub_page_has_the_hidden_channels_card():
    from importlib.resources import files

    html = (files("meeting_scribe") / "static" / "hub.html").read_text(encoding="utf-8")
    assert 'id="hidden-channels"' in html and "ห้องที่สลิดยังมองไม่เห็น" in html
    assert "Read Message History" in html and "renderHidden" in html


# ------------------------------------------------------------------ bilingual page (TH | EN)
def _page_dict_and_html():
    import json
    import re
    from pathlib import Path

    html = (Path(__file__).parents[1] / "src/meeting_scribe/static/hub.html").read_text("utf-8")
    m = re.search(r'<script type="application/json" id="i18n-data">(.*?)</script>', html, re.S)
    assert m, "the dictionary must be an inline JSON block"
    return json.loads(m.group(1)), html


def test_hub_dictionary_has_same_keys_and_placeholders_in_both_languages():
    import re

    d, _ = _page_dict_and_html()
    assert d
    for key, entry in d.items():
        assert set(entry) == {"th", "en"}, key
        assert entry["th"].strip() and entry["en"].strip(), key
        assert set(re.findall(r"\{(\w+)\}", entry["th"])) == set(
            re.findall(r"\{(\w+)\}", entry["en"])
        ), key


def test_hub_every_used_key_exists_and_markup_has_no_hardcoded_thai():
    import re

    d, html = _page_dict_and_html()
    used = set(re.findall(r'data-i18n(?:-html)?="([\w]+)"', html))
    for attrs in re.findall(r'data-i18n-attr="([^"]+)"', html):
        used |= {p.split(":")[1] for p in attrs.split(";")}
    script = html.split('id="i18n-data"', 1)[1]
    used |= set(re.findall(r'\bt\("(\w+)"', script))
    used |= set(re.findall(r'\btx\((?:\$\("[\w-]+"\)|[^,()]*\([^()]*\)),\s*"(\w+)"', script))
    used |= set(re.findall(r'\boption\([^,]+,\s*"(\w+)"\)', script))
    used |= {
        "f_" + f + s for f in re.findall(r'\{ id: "(\w+)"', script) for s in ("_n", "_d", "_w")
    }
    used |= {"mode_" + m for m in ("off", "listen", "call", "join", "chat")}
    used |= {"lang_th", "lang_en", "lang_auto", "st_ok", "st_review", "st_recording"}
    used |= {"st_summarizing", "st_done", "st_failed", "hc_why_hidden", "hc_why_history"}
    used.discard("f_")  # prefix of the per-feature keys, covered above
    assert used - set(d) == set(), used - set(d)
    markup = html.split('<script type="application/json"', 1)[0]
    markup = re.sub(r"<style>.*?</style>", "", markup, flags=re.S)
    assert not re.search(r"[฀-๿]", markup.split("</head>", 1)[1])


def test_hub_has_language_toggle_with_safe_storage_and_browser_default():
    _, html = _page_dict_and_html()
    assert html.count('data-lang="th"') >= 1 and html.count('data-lang="en"') >= 1
    assert 'startsWith("th") ? "th" : "en"' in html
    assert "localStorage.setItem(LANG_KEY" in html and "catch" in html
    assert "toLocaleString(locale()" in html


def test_activity_item_carries_currency(client, env):
    from meeting_scribe.hub import ActivityItem

    assert ActivityItem(kind="receipt", title="x", at="t", status="ok").currency is None
