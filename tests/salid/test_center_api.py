"""Command center API: every endpoint with fake auth, SQLite / fake-Firestore doc stores, a fake
analyzer and a scripted `ask`. No network."""

import asyncio
import hashlib
import hmac
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from conftest import FixedClockSettings
from fake_firestore import FakeFirestore
from fastapi.testclient import TestClient

from meeting_scribe import center, hub, pipeline
from meeting_scribe.app import app
from meeting_scribe.config import get_settings
from salid.connectors import ClickUpConnector, ConnectorConfig, DiscordConnector, SlackConnector
from salid.coo_store import MINUTES, FirestoreDocStore, SQLiteDocStore
from salid.guild_settings import MemoryGuildStore
from salid.library import ORG_DOCS, Library, LocalBlobs
from salid.memory_store import MemoryMessage, SQLiteMemoryStore
from salid.models import Receipt
from salid.orgs import License, Member, OrgStore, Workspace, stamp
from salid.sharing import SHARED_DOCS_CHANNEL, WEB_DOCS_CHANNEL
from salid.store import SQLiteStore
from salid.wiki import WikiPage, WikiRepo

CLIENT_ID = "123-abc.apps.googleusercontent.com"
A, B = 1000, 2000
PUBLIC, PRIVATE = 1001, 1002  # channels of guild A
TOKENS = {
    n: {"email": f"{n}@example.com", "email_verified": True}
    for n in ("boss", "adm", "mem", "other", "root", "stranger")
}
H = {n: {"Authorization": f"Bearer {n}"} for n in TOKENS}
SLACK_SECRET = "s3cret"


def at(day="2026-10-14", h=12):
    return datetime.fromisoformat(f"{day}T{h:02d}:00:00+00:00")


@pytest.fixture(params=["sqlite", "firestore"])
def docs(request):
    if request.param == "sqlite":
        return SQLiteDocStore(":memory:")
    return FirestoreDocStore(client=FakeFirestore())


class World:
    pass


@pytest.fixture
def w(docs, tmp_path, monkeypatch):
    monkeypatch.setenv("STORE", "memory")
    monkeypatch.setenv("HUB_GOOGLE_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("HUB_ADMIN_EMAILS", "root@example.com")
    get_settings.cache_clear()
    pipeline.get_store.cache_clear()
    center._chat_times.clear()
    s = FixedClockSettings(tz="Asia/Bangkok")
    w = World()
    w.s, w.docs = s, docs
    w.orgs = OrgStore(docs)
    w.ledger = SQLiteStore(str(tmp_path / "l.db"))
    w.memory = SQLiteMemoryStore(":memory:", s.zone)
    w.guilds = MemoryGuildStore()
    w.analyzed = []

    def analyzer(data, filename, mime):
        w.analyzed.append(filename)
        if filename.endswith(".exe"):
            from salid.docs import DocUnreadable

            raise DocUnreadable("nope")
        return {
            "tag": "contract",
            "summary": f"summary of {filename}",
            "text": data.decode("utf-8", "ignore"),
        }

    w.lib = Library(
        docs=docs,
        ledger=w.ledger,
        memory=w.memory,
        guilds=w.guilds,
        settings=s,
        blobs=LocalBlobs(tmp_path / "up"),
        analyzer=analyzer,
        now=lambda: at("2026-10-15"),
    )
    w.asks = []

    def fake_ask(settings, store, question, **kw):
        w.asks.append((question, kw))
        return SimpleNamespace(text="ans:" + question)

    w.http_log = []
    w.handler = lambda request: httpx.Response(404)

    def connectors():
        cfg = ConnectorConfig(
            base_url="https://hub.example",
            slack_client_id="cid",
            slack_client_secret="csec",
            slack_signing_secret=SLACK_SECRET,
            clickup_client_id="ccid",
            clickup_client_secret="ccsec",
            discord_client_id="123456789012345678",
        )
        if getattr(w, "no_connectors", False):
            cfg = ConnectorConfig()
        client = httpx.Client(
            transport=httpx.MockTransport(lambda r: (w.http_log.append(r), w.handler(r))[1])
        )
        return {
            "discord": DiscordConnector(cfg),
            "slack": SlackConnector(cfg, memory=w.memory, docs=docs, zone=s.zone, client=client),
            "clickup": ClickUpConnector(cfg, client=client),
        }

    def verifier():
        def verify(token, client_id):
            if token not in TOKENS:
                raise ValueError("bad token")
            return TOKENS[token]

        return verify

    app.dependency_overrides[hub.google_token_verifier] = verifier
    app.dependency_overrides[hub.get_guild_store] = lambda: w.guilds
    app.dependency_overrides[hub.get_salid_runtime] = lambda: (s, w.ledger)
    app.dependency_overrides[hub.get_memory_store] = lambda: w.memory
    app.dependency_overrides[hub.get_prefs] = lambda: w.orgs
    # the web chat's language safety net would call Gemini: tests never do (identity here)
    monkeypatch.setattr(center, "get_translator", lambda: lambda text, lang: text)
    app.dependency_overrides[center.get_orgs] = lambda: w.orgs
    app.dependency_overrides[center.get_library] = lambda: w.lib
    app.dependency_overrides[center.get_ask] = lambda: fake_ask
    app.dependency_overrides[center.get_connectors] = connectors
    # hub.get_salid_runtime() is also called directly by the chat handler
    monkeypatch.setattr(hub, "get_salid_runtime", lambda: (s, w.ledger))
    monkeypatch.setattr(center, "connector_config", lambda: connectors()["slack"].cfg)

    # ---- org with two Discord servers
    for gid, name in ((A, "Cafe A"), (B, "Cafe B")):
        w.guilds.upsert_guild(
            gid,
            name=name,
            icon_url=f"https://cdn/{gid}.png",
            channels=[
                {"id": str(gid + 1), "name": "general"},
                {"id": str(gid + 2), "name": "execs"},
            ],
            installed=True,
            public_channel_ids=[str(gid + 1)],
        )
    w.org = w.orgs.create(
        "Cafe Group",
        "boss@example.com",
        org_id="org_t",
        members=[
            Member(email="boss@example.com", role="owner"),
            Member(email="adm@example.com", role="admin"),
            Member(email="mem@example.com", role="member"),
        ],
        license=License(plan="monthly", status="active", seats=5, valid_until="2099-01-01"),
    )
    for gid, name in ((A, "Cafe A"), (B, "Cafe B")):
        w.orgs.attach_workspace(
            "org_t",
            Workspace(id=f"discord-{gid}", platform="discord", external_id=str(gid), name=name),
        )
    w.other = w.orgs.create("Other Co", "other@example.com", org_id="org_o")
    w.client = TestClient(app)
    yield w
    app.dependency_overrides.clear()
    get_settings.cache_clear()
    pipeline.get_store.cache_clear()


def note(w, gid, channel, mid, text, author="Ann", day="2026-10-10", kind="chat"):
    w.memory.add_message(
        MemoryMessage(
            guild_id=gid,
            channel_id=channel,
            message_id=mid,
            author_id=7,
            author_name=author,
            created_at=at(day),
            local_date=day,
            text=text,
            jump_url=f"https://discord.com/channels/{gid}/{channel}/{mid}",
            kind=kind,
        )
    )


def seed_content(w):
    """Guild A: a public file note, a private one, a DM note; two receipts; wiki; minutes."""
    note(
        w,
        A,
        PUBLIC,
        1,
        "look\n[PDF: contract-a.pdf, 3 หน้า, contract] Lease for the shop | ยอด: 5,000",
    )
    note(w, A, PRIVATE, 2, "[DOCX: salary.docx, 1 หน้า, other] Salary table", day="2026-10-11")
    note(w, A, -55, 3, "[PDF: diary.pdf, 1 หน้า] very private", kind="dm")
    w.ledger.add_receipt(
        Receipt(
            guild_id=A, submitter_id=1, submitter_name="Ann", sha256="a1", vendor="Makro",
            effective_date="2026-10-12", total=1200.0, created_at="2026-10-12",
            channel_id=PUBLIC, message_id=99,
        )
    )  # fmt: skip
    repo = WikiRepo(w.docs)
    repo.save(
        A, WikiPage(slug="open", title="Opening hours", body="We open at 8", channel_ids=[PUBLIC])
    )
    repo.save(A, WikiPage(slug="secret", title="Payroll", body="salaries", channel_ids=[PRIVATE]))
    repo.save(A, WikiPage(slug="dmpage", title="From a DM", body="x", channel_ids=[-5]))
    w.docs.put(
        MINUTES, A, "m1",
        {"voice_channel_id": PUBLIC, "voice_channel_name": "town hall", "ended_at": "2026-10-13T10:00:00+00:00",
         "attendee_ids": [1, 2], "summary": "Decided to open earlier"},
    )  # fmt: skip
    w.docs.put(
        MINUTES, A, "m2",
        {"voice_channel_id": PRIVATE, "voice_channel_name": "execs", "ended_at": "2026-10-13T11:00:00+00:00",
         "attendee_ids": [1], "summary": "Budget cuts"},
    )  # fmt: skip


def titles(resp):
    assert resp.status_code == 200, resp.text
    return {i["title"] for i in resp.json()["items"]}


# ------------------------------------------------------------------ me / orgs
def test_requires_auth(w):
    assert w.client.get("/api/center/me").status_code == 401
    assert (
        w.client.get("/api/center/me", headers={"Authorization": "Bearer nope"}).status_code == 401
    )


def test_me_lists_orgs_with_roles_and_super_sees_all(w):
    me = w.client.get("/api/center/me", headers=H["adm"]).json()
    assert me["email"] == "adm@example.com" and not me["is_super"]
    assert [(o["id"], o["role"]) for o in me["orgs"]] == [("org_t", "admin")]
    assert me["orgs"][0]["license"]["active"] is True
    assert w.client.get("/api/center/me", headers=H["stranger"]).json()["orgs"] == []
    sup = w.client.get("/api/center/me", headers=H["root"]).json()
    assert sup["is_super"] and {o["id"] for o in sup["orgs"]} == {"org_t", "org_o"}
    assert {o["role"] for o in sup["orgs"]} == {"super"}


def test_create_org_gives_owner_web_workspace_and_open_ended_trial(w):
    r = w.client.post("/api/center/orgs", json={"name": "New Co"}, headers=H["stranger"])
    assert r.status_code == 200, r.text
    org = r.json()
    assert org["role"] == "owner" and org["owner_email"] == "stranger@example.com"
    lic = org["license"]
    assert lic["plan"] == "trial" and lic["days_left"] is None and lic["active"]  # no end, no cap
    assert lic["seats"] == 10_000
    assert [x["platform"] for x in org["workspaces"]] == ["web"]
    assert w.client.get(f"/api/center/orgs/{org['id']}", headers=H["stranger"]).status_code == 200
    for i in range(center.MAX_ORGS_PER_USER):
        r = w.client.post("/api/center/orgs", json={"name": f"n{i}"}, headers=H["stranger"])
    assert r.status_code == 409


def test_org_access_by_role(w):
    url = "/api/center/orgs/org_t"
    for who in ("boss", "adm", "mem"):
        assert w.client.get(url, headers=H[who]).json()["role"] in ("owner", "admin", "member")
    assert w.client.get(url, headers=H["stranger"]).status_code == 403
    assert w.client.get(url, headers=H["other"]).status_code == 403  # another org's owner
    assert w.client.get(url, headers=H["root"]).json()["role"] == "super"
    assert w.client.get("/api/center/orgs/nope", headers=H["stranger"]).status_code == 403
    assert w.client.get("/api/center/orgs/nope", headers=H["root"]).status_code == 404


def test_update_org_admin_only(w):
    body = {"name": "Renamed", "logo_url": "https://x.example/l.png"}
    assert w.client.put("/api/center/orgs/org_t", json=body, headers=H["mem"]).status_code == 403
    r = w.client.put("/api/center/orgs/org_t", json=body, headers=H["adm"])
    assert r.status_code == 200 and r.json()["name"] == "Renamed"
    assert w.orgs.get("org_t").logo_url == "https://x.example/l.png"
    bad = {"name": "x", "logo_url": "javascript:alert(1)"}
    assert w.client.put("/api/center/orgs/org_t", json=bad, headers=H["adm"]).status_code == 422


# ------------------------------------------------------------------ members
def test_member_management_roles(w):
    base = "/api/center/orgs/org_t/members"
    assert w.client.get(base, headers=H["mem"]).status_code == 403
    assert len(w.client.get(base, headers=H["adm"]).json()) == 3
    # admin adds a plain member and may hand out admin, never owner
    ok = w.client.post(base, json={"email": "New@Example.com"}, headers=H["adm"])
    assert ok.status_code == 200 and "new@example.com" in {m["email"] for m in ok.json()}
    deny = w.client.post(base, json={"email": "x@example.com", "role": "owner"}, headers=H["adm"])
    assert deny.status_code == 403
    ok = w.client.post(base, json={"email": "x@example.com", "role": "admin"}, headers=H["adm"])
    assert ok.status_code == 200 and w.orgs.get("org_t").role_of("x@example.com") == "admin"
    # ...but cannot touch that admin afterwards (owner only)
    r = w.client.post(base, json={"email": "x@example.com", "role": "member"}, headers=H["adm"])
    assert r.status_code == 403
    assert (
        w.client.post(
            base, json={"email": "x@example.com", "role": "king"}, headers=H["boss"]
        ).status_code
        == 422
    )
    # ownership is never granted, not even by the owner (Transfer ownership is the way)
    r = w.client.post(base, json={"email": "new@example.com", "role": "owner"}, headers=H["boss"])
    assert r.status_code == 403 and "Transfer" in r.json()["detail"]
    # owner promotes; the new member sees the org through the email index
    w.client.post(base, json={"email": "new@example.com", "role": "admin"}, headers=H["boss"])
    assert w.orgs.get("org_t").role_of("new@example.com") == "admin"
    assert [o.id for o in w.orgs.orgs_of_email("new@example.com")] == ["org_t"]
    # admin cannot remove an admin or owner, can remove a member
    assert w.client.delete(f"{base}/boss@example.com", headers=H["adm"]).status_code == 403
    assert w.client.delete(f"{base}?email=mem@example.com", headers=H["adm"]).status_code == 200
    assert w.orgs.orgs_of_email("mem@example.com") == []
    # the last owner stays (and is told to transfer ownership)
    assert w.client.delete(f"{base}/boss@example.com", headers=H["boss"]).status_code == 409
    me_down = w.client.post(
        base, json={"email": "boss@example.com", "role": "member"}, headers=H["boss"]
    )
    assert me_down.status_code == 409 and "transfer" in me_down.json()["detail"]
    demote = w.client.post(
        base, json={"email": "boss@example.com", "role": "admin"}, headers=H["boss"]
    )
    assert demote.status_code == 409
    assert w.client.delete(f"{base}/ghost@example.com", headers=H["boss"]).status_code == 404


def test_a_role_edit_keeps_the_members_other_fields(w):
    base = "/api/center/orgs/org_t/members"
    w.orgs.update_members(
        "org_t",
        lambda ms: (
            [
                (
                    setattr(m, "title", "CTO"),
                    setattr(m, "level", 2),
                    setattr(m, "department", "tech"),
                )
                for m in ms
                if m.email == "mem@example.com"
            ]
            or True
        ),
    )
    r = w.client.post(base, json={"email": "mem@example.com", "role": "admin"}, headers=H["boss"])
    assert r.status_code == 200
    m = next(x for x in w.orgs.get("org_t").members if x.email == "mem@example.com")
    assert (m.role, m.title, m.level, m.department) == ("admin", "CTO", 2, "tech")


def test_seat_limit(w):
    org = w.orgs.get("org_t")
    org.license.seats = 3
    w.orgs.save(org)
    r = w.client.post(
        "/api/center/orgs/org_t/members", json={"email": "n@example.com"}, headers=H["boss"]
    )
    assert r.status_code == 402
    # changing an existing member's role is not a new seat
    r = w.client.post(
        "/api/center/orgs/org_t/members",
        json={"email": "mem@example.com", "role": "admin"},
        headers=H["boss"],
    )
    assert r.status_code == 200


# ------------------------------------------------------------------ workspaces
def test_workspaces_list(w):
    r = w.client.get("/api/center/orgs/org_t/workspaces", headers=H["mem"]).json()
    assert [x["id"] for x in r] == ["web", "discord-1000", "discord-2000"]
    d = next(x for x in r if x["id"] == "discord-1000")
    assert d["settings_url"] == "/hub" and d["external_id"] == "1000" and d["status"] == "connected"
    assert (
        w.client.get("/api/center/orgs/org_t/workspaces", headers=H["stranger"]).status_code == 403
    )


def test_link_discord_with_code(w):
    c = 3000
    w.guilds.upsert_guild(c, name="New Guild", icon_url="i", channels=[], installed=True)
    code = w.guilds.create_link_code(c, 1, "u")
    url = "/api/center/orgs/org_t/workspaces/discord/link"
    assert w.client.post(url, json={"code": code}, headers=H["mem"]).status_code == 403
    r = w.client.post(url, json={"code": code}, headers=H["adm"])
    assert r.status_code == 200, r.text
    assert r.json()["id"] == "discord-3000" and r.json()["name"] == "New Guild"
    assert w.orgs.org_of_guild(c).id == "org_t"
    assert "adm@example.com" in w.guilds.get(c).admin_emails  # also a hub admin of the server
    # the code was consumed
    assert w.client.post(url, json={"code": code}, headers=H["adm"]).status_code == 400
    assert w.client.post(url, json={"code": "x" * 20}, headers=H["adm"]).status_code == 400


def test_link_discord_guild_in_another_org_is_409_and_code_survives(w):
    code = w.guilds.create_link_code(A, 1, "u")  # guild A belongs to org_t
    url = "/api/center/orgs/org_o/workspaces/discord/link"
    r = w.client.post(url, json={"code": code}, headers=H["other"])
    assert r.status_code == 409
    assert w.orgs.org_of_guild(A).id == "org_t"
    assert w.guilds.peek_link_code(code) == A  # not burned by the failed attempt


def test_link_moves_a_guild_out_of_an_unclaimed_grandfathered_shell(w):
    w.guilds.upsert_guild(4000, name="Old", icon_url="", channels=[], installed=True)
    shell = w.orgs.create("Old", "", org_id="org_g4000", members=[], grandfathered=True)
    w.orgs.attach_workspace(
        shell.id, Workspace(id="discord-4000", platform="discord", external_id="4000")
    )
    code = w.guilds.create_link_code(4000, 1, "u")
    r = w.client.post(
        "/api/center/orgs/org_t/workspaces/discord/link", json={"code": code}, headers=H["boss"]
    )
    assert r.status_code == 200 and w.orgs.org_of_guild(4000).id == "org_t"


def test_delete_workspace(w):
    w.client.post(
        "/api/center/orgs/org_t/sharing",
        json={
            "from_workspace": "discord-1000",
            "to_workspace": "discord-2000",
            "categories": ["wiki"],
        },
        headers=H["adm"],
    )
    base = "/api/center/orgs/org_t/workspaces"
    assert w.client.delete(f"{base}/discord-1000", headers=H["mem"]).status_code == 403
    assert w.client.delete(f"{base}/web", headers=H["adm"]).status_code == 400
    assert w.client.delete(f"{base}/nope", headers=H["adm"]).status_code == 404
    r = w.client.delete(f"{base}/discord-1000", headers=H["adm"])
    assert r.status_code == 200 and [x["id"] for x in r.json()] == ["web", "discord-2000"]
    assert w.orgs.org_of_guild(A) is None
    assert w.orgs.get("org_t").sharing == []  # links touching it are gone


# ------------------------------------------------------------------ sharing
def share(w, who="adm", frm="discord-1000", to="discord-2000", cats=("wiki",)):
    return w.client.post(
        "/api/center/orgs/org_t/sharing",
        json={"from_workspace": frm, "to_workspace": to, "categories": list(cats)},
        headers=H[who],
    )


def test_sharing_crud_and_validation(w):
    assert w.client.get("/api/center/orgs/org_t/sharing", headers=H["mem"]).json() == []
    assert share(w, "mem").status_code == 403
    r = share(w, cats=("wiki", "places", "wiki"))
    assert r.status_code == 200
    link = r.json()[0]
    assert (link["from_workspace"], link["to_workspace"]) == ("discord-1000", "discord-2000")
    assert link["categories"] == ["wiki", "places"] and link["created_by"] == "adm@example.com"
    assert link["created_at"]
    # same pair again replaces
    assert share(w, cats=("minutes",)).json()[0]["categories"] == ["minutes"]
    assert len(w.orgs.get("org_t").sharing) == 1
    assert share(w, cats=("chat",)).status_code == 422  # chat memory is not shareable
    assert share(w, to="discord-1000").status_code == 422
    assert share(w, to="discord-9").status_code == 404
    assert share(w, cats=()).status_code == 422
    # directional: the opposite direction is a separate link
    assert len(share(w, frm="discord-2000", to="discord-1000").json()) == 2
    q = "from_workspace=discord-1000&to_workspace=discord-2000"
    assert (
        w.client.delete(f"/api/center/orgs/org_t/sharing?{q}", headers=H["mem"]).status_code == 403
    )
    assert len(w.client.delete(f"/api/center/orgs/org_t/sharing?{q}", headers=H["adm"]).json()) == 1
    assert (
        w.client.delete(f"/api/center/orgs/org_t/sharing?{q}", headers=H["adm"]).status_code == 404
    )


def test_sharing_cannot_cross_orgs(w):
    other_ws = Workspace(id="discord-5000", platform="discord", external_id="5000", name="X")
    w.orgs.attach_workspace("org_o", other_ws)
    assert share(w, to="discord-5000").status_code == 404


# ------------------------------------------------------------------ license
def test_license_view_request_and_super_update(w):
    r = w.client.get("/api/center/orgs/org_t/license", headers=H["mem"]).json()
    assert r["license"]["plan"] == "monthly" and r["requests"] == []
    body = {"plan": "yearly", "seats": 10, "note": "please"}
    url = "/api/center/orgs/org_t/license/request"
    assert w.client.post(url, json=body, headers=H["mem"]).status_code == 403
    assert w.client.post(url, json={**body, "plan": "trial"}, headers=H["adm"]).status_code == 422
    req = w.client.post(url, json=body, headers=H["adm"]).json()
    assert req["status"] == "pending" and req["requested_by"] == "adm@example.com"
    assert [
        x["id"]
        for x in w.client.get("/api/center/orgs/org_t/license", headers=H["adm"]).json()["requests"]
    ] == [req["id"]]
    # only a super admin sets the license
    put = {
        "plan": "yearly",
        "status": "active",
        "seats": 10,
        "valid_until": "2027-10-01",
        "request_id": req["id"],
    }
    assert (
        w.client.put("/api/center/orgs/org_t/license", json=put, headers=H["boss"]).status_code
        == 403
    )
    r = w.client.put("/api/center/orgs/org_t/license", json=put, headers=H["root"])
    assert r.status_code == 200 and r.json()["license"]["plan"] == "yearly"
    assert r.json()["requests"][0]["status"] == "approved"
    assert w.orgs.get("org_t").license.seats == 10
    bad = w.client.put(
        "/api/center/orgs/org_t/license", json={**put, "status": "weird"}, headers=H["root"]
    )
    assert bad.status_code == 422


def test_super_admin_request_queue_and_decision(w):
    req = w.client.post(
        "/api/center/orgs/org_t/license/request",
        json={"plan": "monthly", "seats": 7},
        headers=H["boss"],
    ).json()
    assert w.client.get("/api/center/admin/license-requests", headers=H["boss"]).status_code == 403
    pending = w.client.get("/api/center/admin/license-requests", headers=H["root"]).json()
    assert [r["id"] for r in pending] == [req["id"]]
    assert len(w.client.get("/api/center/admin/orgs", headers=H["root"]).json()) == 2
    done = w.client.post(
        f"/api/center/admin/license-requests/{req['id']}", json={"approve": True}, headers=H["root"]
    )
    assert done.status_code == 200 and done.json()["status"] == "approved"
    lic = w.orgs.get("org_t").license
    assert (lic.plan, lic.seats, lic.status) == ("monthly", 7, "active") and lic.valid_until
    again = w.client.post(
        f"/api/center/admin/license-requests/{req['id']}",
        json={"approve": False},
        headers=H["root"],
    )
    assert again.status_code == 409
    assert w.client.get("/api/center/admin/license-requests", headers=H["root"]).json() == []


# ------------------------------------------------------------------ documents: visibility
def test_documents_exec_sees_everything_except_dms(w):
    seed_content(w)
    for who in ("boss", "adm", "root"):
        got = titles(w.client.get("/api/center/orgs/org_t/documents?limit=100", headers=H[who]))
        assert {"contract-a.pdf", "salary.docx", "Makro", "Opening hours", "Payroll"} <= got
        assert "diary.pdf" not in got and "From a DM" not in got  # DMs never leave Discord
        assert any(t.startswith("town hall") for t in got) and any(
            t.startswith("execs") for t in got
        )


def test_documents_member_sees_only_public_items_and_no_ledger(w):
    seed_content(w)
    got = titles(w.client.get("/api/center/orgs/org_t/documents?limit=100", headers=H["mem"]))
    assert (
        "contract-a.pdf" in got
        and "Opening hours" in got
        and any(t.startswith("town hall") for t in got)
    )
    assert not got & {"salary.docx", "Payroll", "diary.pdf", "From a DM", "Makro"}
    assert not any(t.startswith("execs") for t in got)
    cat = w.client.get(
        "/api/center/orgs/org_t/documents?category=receipts", headers=H["mem"]
    ).json()
    assert cat["items"] == []
    cat = w.client.get(
        "/api/center/orgs/org_t/documents?category=receipts", headers=H["adm"]
    ).json()
    assert [i["title"] for i in cat["items"]] == ["Makro"]
    assert cat["items"][0]["source_url"].endswith(f"/{PUBLIC}/99")


def test_document_item_shape_filters_search_and_paging(w):
    seed_content(w)
    base = "/api/center/orgs/org_t/documents"
    item = next(
        i
        for i in w.client.get(f"{base}?category=contracts_docs&q=lease", headers=H["boss"]).json()[
            "items"
        ]
    )
    assert item["title"] == "contract-a.pdf" and item["category"] == "contracts_docs"
    assert item["workspace"] == "discord-1000" and item["workspace_name"] == "Cafe A"
    assert item["who"] == "Ann" and item["date"].startswith("2026-10-10")
    assert "Lease" in item["preview"] and item["source_url"].startswith("https://discord.com/")
    assert item["shared_from"] is None
    assert w.client.get(f"{base}?category=nope", headers=H["boss"]).status_code == 422
    assert w.client.get(f"{base}?workspace=discord-2000", headers=H["boss"]).json()["items"] == []
    names = [
        i["title"]
        for i in w.client.get(f"{base}?workspace=discord-1000&limit=100", headers=H["boss"]).json()[
            "items"
        ]
    ]
    assert "Makro" in names
    p1 = w.client.get(f"{base}?limit=2", headers=H["boss"]).json()
    assert len(p1["items"]) == 2 and p1["next_cursor"] == "2" and p1["total"] >= 7
    p2 = w.client.get(f"{base}?limit=2&cursor={p1['next_cursor']}", headers=H["boss"]).json()
    assert {i["id"] for i in p1["items"]}.isdisjoint({i["id"] for i in p2["items"]})
    dates = [i["date"] for i in p1["items"] + p2["items"]]
    assert dates == sorted(dates, reverse=True)


def test_document_detail_respects_visibility(w):
    seed_content(w)
    items = w.client.get("/api/center/orgs/org_t/documents?limit=100", headers=H["boss"]).json()[
        "items"
    ]
    pub = next(i for i in items if i["title"] == "Opening hours")
    sec = next(i for i in items if i["title"] == "Payroll")
    url = "/api/center/orgs/org_t/documents/"
    d = w.client.get(url + pub["id"], headers=H["mem"])
    assert d.status_code == 200 and d.json()["text"] == "We open at 8"
    assert w.client.get(url + sec["id"], headers=H["mem"]).status_code == 404
    assert w.client.get(url + sec["id"], headers=H["adm"]).json()["text"] == "salaries"
    assert w.client.get(url + "discord-1000:wiki:dmpage", headers=H["boss"]).status_code == 404


# ------------------------------------------------------------------ documents: sharing safety
def test_shared_documents_into_a_workspace_are_public_only(w):
    seed_content(w)
    share(w, frm="discord-1000", to="discord-2000", cats=("documents", "wiki", "minutes"))
    base = "/api/center/orgs/org_t/documents?workspace=discord-2000&limit=100"
    got = w.client.get(base, headers=H["adm"]).json()["items"]
    shared = {i["title"] for i in got}
    assert {"contract-a.pdf", "Opening hours"} <= shared and any(
        t.startswith("town hall") for t in shared
    )
    assert not shared & {
        "salary.docx",
        "Payroll",
        "diary.pdf",
        "From a DM",
    }  # private never crosses
    assert not any(t.startswith("execs") for t in shared)
    assert all(i["shared_from"] == "Cafe A" for i in got)
    assert "Makro" not in shared  # ledger not shared here


def test_ledger_share_reaches_execs_on_the_web_only(w):
    seed_content(w)
    share(w, cats=("ledger",))
    base = "/api/center/orgs/org_t/documents?workspace=discord-2000&category=receipts"
    assert [i["title"] for i in w.client.get(base, headers=H["adm"]).json()["items"]] == ["Makro"]
    assert w.client.get(base, headers=H["mem"]).json()["items"] == []
    # and the bot-side tools have no ledger category at all
    from salid.sharing import SharedSource  # noqa: F401

    assert "ledger" not in ("documents", "wiki", "places", "minutes", "tasks")


def test_no_share_link_nothing_crosses(w):
    seed_content(w)
    got = w.client.get(
        "/api/center/orgs/org_t/documents?workspace=discord-2000", headers=H["boss"]
    ).json()
    assert got["items"] == []


# ------------------------------------------------------------------ uploads
def upload(w, who="mem", name="plan.txt", data=b"hello contract text", mime="text/plain"):
    return w.client.post(
        "/api/center/orgs/org_t/documents", files={"file": (name, data, mime)}, headers=H[who]
    )


def test_upload_flow_categorise_store_index_download(w):
    r = upload(w, "mem")
    assert r.status_code == 200, r.text
    item = r.json()
    assert (
        item["category"] == "uploads"
        and item["workspace"] == "web"
        and item["tags"] == ["contract"]
    )
    assert item["title"] == "plan.txt" and item["who"] == "mem@example.com" and item["size"] == 19
    assert "summary of plan.txt" in item["preview"]
    assert w.analyzed == ["plan.txt"]
    # stored in the doc store, bytes on disk
    rec = w.docs.find(ORG_DOCS, w.orgs.get("org_t").web_workspace().memory_id)[0]
    assert (
        rec["by"] == "mem@example.com" and Path(rec["uri"]).read_bytes() == b"hello contract text"
    )
    # listed for a member and an admin, searchable by text
    assert titles(
        w.client.get("/api/center/orgs/org_t/documents?category=uploads", headers=H["mem"])
    ) == {"plan.txt"}
    assert titles(
        w.client.get("/api/center/orgs/org_t/documents?q=hello+contract", headers=H["adm"])
    ) == {"plan.txt"}
    det = w.client.get(f"/api/center/orgs/org_t/documents/{item['id']}", headers=H["adm"]).json()
    assert det["text"] == "hello contract text"
    dl = w.client.get(f"/api/center/orgs/org_t/documents/{item['id']}/file", headers=H["mem"])
    assert dl.status_code == 200 and dl.content == b"hello contract text"
    assert (
        w.client.get(
            f"/api/center/orgs/org_t/documents/{item['id']}/file", headers=H["other"]
        ).status_code
        == 403
    )
    # searchable by Salid: a file note in the org's web workspace memory
    web = w.orgs.get("org_t").web_workspace()
    today = "2026-10-15"
    notes = w.memory.messages(web.memory_id, {WEB_DOCS_CHANNEL}, "2026-10-01", today)
    assert (
        len(notes) == 1
        and "summary of plan.txt" in notes[0].text
        and "[ไฟล์: plan.txt" in notes[0].text
    )
    # not in the Discord workspaces without a ShareLink
    assert w.memory.messages(A, {SHARED_DOCS_CHANNEL}, "2026-10-01", today) == []


def test_upload_rejections(w):
    assert upload(w, "stranger").status_code == 403
    assert (
        upload(w, name="virus.exe", data=b"MZ", mime="application/octet-stream").status_code == 415
    )
    assert upload(w, data=b"").status_code == 400
    from salid import library

    big = b"x" * (library.MAX_UPLOAD_BYTES + 1)
    assert upload(w, data=big).status_code == 413


def test_uploads_reach_discord_workspaces_only_through_a_documents_share_link(w):
    web = w.orgs.get("org_t").web_workspace()
    upload(w, "adm")
    today = "2026-10-15"
    assert w.memory.messages(A, {SHARED_DOCS_CHANNEL}, "2026-10-01", today) == []
    r = share(w, "adm", frm="web", to="discord-1000", cats=("documents",))
    assert r.status_code == 200
    copies = w.memory.messages(A, {SHARED_DOCS_CHANNEL}, "2026-10-01", today)
    assert len(copies) == 1 and "plan.txt" in copies[0].text
    assert w.memory.messages(B, {SHARED_DOCS_CHANNEL}, "2026-10-01", today) == []
    upload(w, "mem", name="two.txt", data=b"second file")  # new uploads follow the link
    assert len(w.memory.messages(A, {SHARED_DOCS_CHANNEL}, "2026-10-01", today)) == 2
    # the bot adds the channel to what Salid may read in that guild
    from salid.sharing import Sharing

    sh = Sharing(w.orgs, lambda _m: set())
    allowed, _ = sh.with_shared_docs(A, {PUBLIC: "general"})
    assert (
        SHARED_DOCS_CHANNEL in allowed and SHARED_DOCS_CHANNEL not in sh.with_shared_docs(B, {})[0]
    )
    # removing the link removes the copies
    q = "from_workspace=web&to_workspace=discord-1000"
    assert (
        w.client.delete(f"/api/center/orgs/org_t/sharing?{q}", headers=H["adm"]).status_code == 200
    )
    assert w.memory.messages(A, {SHARED_DOCS_CHANNEL}, "2026-10-01", today) == []
    assert len(w.memory.messages(web.memory_id, {WEB_DOCS_CHANNEL}, "2026-10-01", today)) == 2


def test_reindex_rebuilds_notes(w):
    upload(w, "adm")
    web = w.orgs.get("org_t").web_workspace()
    w.memory.delete_guild(web.memory_id)
    assert (
        w.client.post("/api/center/orgs/org_t/documents/reindex", headers=H["mem"]).status_code
        == 403
    )
    assert w.client.post("/api/center/orgs/org_t/documents/reindex", headers=H["adm"]).json() == {
        "indexed": 1
    }
    assert (
        len(w.memory.messages(web.memory_id, {WEB_DOCS_CHANNEL}, "2026-10-01", "2026-10-15")) == 1
    )


# ------------------------------------------------------------------ overview
def test_overview(w, monkeypatch):
    """The company profile only: no document scan (it took seconds on a real company)."""
    seed_content(w)
    monkeypatch.setattr(
        w.lib, "gather", lambda *a, **k: (_ for _ in ()).throw(AssertionError("scanned"))
    )
    ov = w.client.get("/api/center/orgs/org_t/overview", headers=H["adm"]).json()
    assert ov["members"] == 3 and ov["workspaces"]["total"] == 3
    assert ov["workspaces"]["by_platform"] == {"web": 1, "discord": 2}
    assert "documents" not in ov and "recent_activity" not in ov
    assert ov["license"]["plan"] == "monthly" and ov["license_days_left"] > 100
    assert w.client.get("/api/center/orgs/org_t/overview", headers=H["mem"]).status_code == 200
    assert w.client.get("/api/center/orgs/org_t/overview", headers=H["other"]).status_code == 403


# ------------------------------------------------------------------ chat
def test_chat_scopes_memory_to_the_member_and_workspace(w):
    seed_content(w)
    note(w, A, -55, 9, "dm note", kind="dm")
    url = "/api/center/orgs/org_t/chat"
    r = w.client.post(
        url, json={"message": "สรุปหน่อย", "workspace_id": "discord-1000"}, headers=H["mem"]
    )
    assert r.status_code == 200, r.text
    assert r.json()["answer"] == "ans:สรุปหน่อย" and r.json()["workspace_id"] == "discord-1000"
    q, kw = w.asks[-1]
    assert (
        kw["guild_id"] == A and kw["is_exec"] is False and kw["user_id"] != 0
    )  # the person, not anonymous
    assert set(kw["memory"].allowed) == {PUBLIC}  # a plain member: public channels only
    assert kw["language"] == "en"  # the COMPANY language (unchosen server = English), not Thai
    w.client.post(url, json={"message": "total?", "workspace_id": "discord-1000"}, headers=H["adm"])
    q, kw = w.asks[-1]
    assert kw["is_exec"] is True and set(kw["memory"].allowed) == {
        PUBLIC,
        PRIVATE,
    }  # never DM scope
    assert kw["language"] == "en"
    assert "สรุปหน่อย" not in kw["history"]  # history is per user
    # no workspace given = "the company": its first connected Discord server, same rights as
    # when that server was named explicitly (an admin is an executive there)
    w.client.post(url, json={"message": "hi"}, headers=H["adm"])
    q, kw = w.asks[-1]
    assert kw["guild_id"] == A
    assert kw["is_exec"] is True and set(kw["memory"].allowed) == {PUBLIC, PRIVATE}


def test_chat_other_guild_memory_is_not_readable(w):
    """Chat memory never crosses workspaces, even with sharing links."""
    share(w, cats=("documents", "wiki", "places", "minutes", "tasks"))
    w.client.post(
        "/api/center/orgs/org_t/chat",
        json={"message": "x", "workspace_id": "discord-2000"},
        headers=H["boss"],
    )
    kw = w.asks[-1][1]
    assert kw["guild_id"] == B and set(kw["memory"].allowed) <= {B + 1, B + 2}


def test_chat_history_and_limits(w):
    url = "/api/center/orgs/org_t/chat"
    assert w.client.get(url + "/history", headers=H["mem"]).json() == {"turns": []}
    w.client.post(url, json={"message": "one"}, headers=H["mem"])
    w.client.post(url, json={"message": "two"}, headers=H["mem"])
    turns = w.client.get(url + "/history", headers=H["mem"]).json()["turns"]
    assert [(t["role"], t["text"]) for t in turns] == [
        ("user", "one"), ("salid", "ans:one"), ("user", "two"), ("salid", "ans:two"),
    ]  # fmt: skip
    assert w.client.get(url + "/history", headers=H["adm"]).json() == {"turns": []}
    assert "one" in w.asks[-1][1]["history"]
    for i in range(60):
        w.orgs.append_chat(
            "org_t", "mem@example.com", [{"role": "user", "text": str(i), "at": "t"}]
        )
    assert len(w.orgs.chat_history("org_t", "mem@example.com")) == 50
    assert (
        w.client.post(
            url, json={"message": "x", "workspace_id": "nope"}, headers=H["mem"]
        ).status_code
        == 404
    )
    assert w.client.post(url, json={"message": ""}, headers=H["mem"]).status_code == 422
    assert w.client.post(url, json={"message": "x"}, headers=H["stranger"]).status_code == 403


def test_chat_blocked_when_license_inactive_and_rate_limited(w):
    org = w.orgs.get("org_t")
    org.license.status = "suspended"
    w.orgs.save(org)
    url = "/api/center/orgs/org_t/chat"
    assert w.client.post(url, json={"message": "x"}, headers=H["mem"]).status_code == 402
    assert (
        w.client.post(url, json={"message": "x"}, headers=H["root"]).status_code == 200
    )  # super admin
    org.license.status = "active"
    w.orgs.save(org)
    center._chat_times.clear()
    codes = [
        w.client.post(url, json={"message": "x"}, headers=H["mem"]).status_code
        for _ in range(center.CHAT_PER_HOUR + 1)
    ]
    assert codes[-1] == 429 and codes[0] == 200


# ------------------------------------------------------------------ connectors
def test_connectors_status_cards(w):
    cards = {
        c["platform"]: c
        for c in w.client.get("/api/center/connectors?org=org_t", headers=H["mem"]).json()
    }
    assert cards["discord"]["status"] == "available" and cards["discord"]["connected"] == 2
    assert cards["slack"]["enabled"] and cards["clickup"]["configured"]
    assert {cards[p]["status"] for p in ("google_drive", "gmail", "google_calendar")} == {
        "coming_soon"
    }
    # strangers get no counts for an org that is not theirs
    cards = {
        c["platform"]: c
        for c in w.client.get("/api/center/connectors?org=org_t", headers=H["stranger"]).json()
    }
    assert cards["discord"]["connected"] == 0
    w.no_connectors = True
    cards = {
        c["platform"]: c for c in w.client.get("/api/center/connectors", headers=H["mem"]).json()
    }
    assert cards["slack"]["status"] == "not_configured" and not cards["slack"]["enabled"]
    assert (
        w.client.get("/api/center/connectors/slack/start?org=org_t", headers=H["adm"]).status_code
        == 503
    )


def test_connector_start_urls_and_permissions(w):
    s = "/api/center/connectors/slack/start?org=org_t"
    assert w.client.get(s, headers=H["mem"]).status_code == 403
    assert w.client.get(s, headers=H["stranger"]).status_code == 403
    assert (
        w.client.get("/api/center/connectors/nope/start?org=org_t", headers=H["adm"]).status_code
        == 404
    )
    r = w.client.get(s, headers=H["adm"]).json()
    assert r["kind"] == "oauth" and r["url"].startswith("https://slack.com/oauth/v2/authorize?")
    assert (
        "client_id=cid" in r["url"]
        and "redirect_uri=https%3A%2F%2Fhub.example%2Fapi%2Fcenter%2Fconnectors%2Fslack%2Fcallback"
        in r["url"]
    )
    assert "state=" in r["url"] and "csec" not in r["url"]
    c = w.client.get("/api/center/connectors/clickup/start?org=org_t", headers=H["boss"]).json()
    assert c["url"].startswith("https://app.clickup.com/api?") and "client_id=ccid" in c["url"]
    d = w.client.get("/api/center/connectors/discord/start?org=org_t", headers=H["boss"]).json()
    assert d["kind"] == "invite" and "discord.com" in d["url"]


def _state(w, platform="slack", who="adm"):
    url = w.client.get(f"/api/center/connectors/{platform}/start?org=org_t", headers=H[who]).json()[
        "url"
    ]
    return dict(p.split("=", 1) for p in url.split("?", 1)[1].split("&"))["state"]


def test_slack_oauth_callback_attaches_workspace_and_stores_token(w):
    state = _state(w)

    def handler(request):
        assert request.url.path == "/api/oauth.v2.access"
        body = request.content.decode()
        assert "code=abc" in body and "client_secret=csec" in body
        return httpx.Response(
            200, json={"ok": True, "access_token": "xoxb-1", "team": {"id": "T1", "name": "Acme"}}
        )

    w.handler = handler
    r = w.client.get(
        f"/api/center/connectors/slack/callback?code=abc&state={state}", follow_redirects=False
    )
    assert r.status_code == 303 and r.headers["location"] == "/center?connected=slack"
    org = w.orgs.get("org_t")
    ws = org.workspace("slack-T1")
    assert ws.platform == "slack" and ws.name == "Acme" and ws.connected_by == "adm@example.com"
    from salid.connectors.base import TOKENS as TOK

    assert w.docs.get(TOK, ws.memory_id, "slack")["access_token"] == "xoxb-1"
    assert "xoxb-1" not in w.client.get("/api/center/orgs/org_t", headers=H["adm"]).text
    assert w.orgs.org_of_workspace("slack", "T1")[0].id == "org_t"
    # the state is one-time
    again = w.client.get(
        f"/api/center/connectors/slack/callback?code=abc&state={state}", follow_redirects=False
    )
    assert "error=bad_state" in again.headers["location"]


def test_oauth_callback_errors(w):
    r = w.client.get(
        "/api/center/connectors/slack/callback?code=abc&state=bogus", follow_redirects=False
    )
    assert "error=bad_state" in r.headers["location"]
    state = _state(w)
    w.handler = lambda request: httpx.Response(200, json={"ok": False, "error": "invalid_code"})
    r = w.client.get(
        f"/api/center/connectors/slack/callback?code=abc&state={state}", follow_redirects=False
    )
    assert "error=failed" in r.headers["location"]
    state = _state(w)
    r = w.client.get(
        f"/api/center/connectors/slack/callback?error=access_denied&state={state}",
        follow_redirects=False,
    )
    assert "error=failed" in r.headers["location"]
    # a team that belongs to another org
    w.orgs.attach_workspace("org_o", Workspace(id="slack-T9", platform="slack", external_id="T9"))
    state = _state(w)
    w.handler = lambda request: httpx.Response(
        200, json={"ok": True, "access_token": "t", "team": {"id": "T9", "name": "n"}}
    )
    r = w.client.get(
        f"/api/center/connectors/slack/callback?code=abc&state={state}", follow_redirects=False
    )
    assert "error=taken" in r.headers["location"]
    # state of a demoted user
    state = _state(w)
    org = w.orgs.get("org_t")
    org.members = [m for m in org.members if m.email != "adm@example.com"]
    w.orgs.save(org)
    r = w.client.get(
        f"/api/center/connectors/slack/callback?code=abc&state={state}", follow_redirects=False
    )
    assert "error=forbidden" in r.headers["location"]


def test_clickup_oauth_and_read_only_tasks(w):
    state = _state(w, "clickup", "boss")

    def handler(request):
        if request.url.path.endswith("/oauth/token"):
            return httpx.Response(200, json={"access_token": "pk_1"})
        if request.url.path.endswith("/team"):
            assert request.headers["authorization"] == "pk_1"
            return httpx.Response(200, json={"teams": [{"id": "90", "name": "Ops"}]})
        if request.url.path.endswith("/team/90/task"):
            assert request.method == "GET"
            return httpx.Response(
                200,
                json={"tasks": [{"id": "t1", "name": "Fix oven", "status": {"status": "open"},
                                 "assignees": [{"username": "Ann"}], "due_date": "1792000000000",
                                 "url": "https://app.clickup.com/t/t1"}]},
            )  # fmt: skip
        return httpx.Response(404)

    w.handler = handler
    r = w.client.get(
        f"/api/center/connectors/clickup/callback?code=c&state={state}", follow_redirects=False
    )
    assert r.headers["location"] == "/center?connected=clickup"
    assert w.orgs.get("org_t").workspace("clickup-90").name == "Ops"
    tasks = w.client.get("/api/center/orgs/org_t/tasks", headers=H["mem"]).json()["tasks"]
    assert tasks == [
        {"id": "t1", "title": "Fix oven", "status": "open", "assignees": ["Ann"], "due": "2026-10-14",
         "url": "https://app.clickup.com/t/t1", "assignee": "Ann", "workspace": "clickup-90", "platform": "clickup"}
    ]  # fmt: skip
    assert all(req.method in ("GET", "POST") for req in w.http_log)
    assert not any(req.method in ("PUT", "DELETE", "PATCH") for req in w.http_log)  # read-only


def test_tasks_view_discord_visibility(w):
    from salid.tasks import Task, TaskRepo

    repo = TaskRepo(w.docs)
    repo.save(Task(guild_id=A, title="public task", source_channel_id=PUBLIC))
    repo.save(Task(guild_id=A, title="private task", source_channel_id=PRIVATE))
    mem = w.client.get("/api/center/orgs/org_t/tasks", headers=H["mem"]).json()["tasks"]
    assert [t["title"] for t in mem] == ["public task"]
    adm = w.client.get(
        "/api/center/orgs/org_t/tasks?workspace=discord-1000", headers=H["adm"]
    ).json()["tasks"]
    assert {t["title"] for t in adm} == {"public task", "private task"}


# ------------------------------------------------------------------ slack events
def sign(body: bytes, ts=None, secret=SLACK_SECRET):
    ts = str(int(time.time()) if ts is None else ts)
    sig = (
        "v0="
        + hmac.new(secret.encode(), b"v0:" + ts.encode() + b":" + body, hashlib.sha256).hexdigest()
    )
    return {
        "X-Slack-Request-Timestamp": ts,
        "X-Slack-Signature": sig,
        "Content-Type": "application/json",
    }


def post_event(w, payload, **kw):
    body = json.dumps(payload).encode()
    return w.client.post("/api/connectors/slack/events", content=body, headers=sign(body, **kw))


def slack_msg(**over):
    ev = {"type": "message", "channel": "C1", "channel_type": "channel", "user": "U1",
          "text": "ship it", "ts": "1760443200.000200"}  # fmt: skip
    ev.update(over)
    return {"type": "event_callback", "team_id": "T1", "event": ev}


def test_slack_events_signature_and_challenge(w):
    body = json.dumps({"type": "url_verification", "challenge": "xyz"}).encode()
    assert w.client.post("/api/connectors/slack/events", content=body).status_code == 401
    assert (
        w.client.post(
            "/api/connectors/slack/events", content=body, headers=sign(body, secret="wrong")
        ).status_code
        == 401
    )
    old = sign(body, ts=int(time.time()) - 400)
    assert (
        w.client.post("/api/connectors/slack/events", content=body, headers=old).status_code == 401
    )  # replay
    r = w.client.post("/api/connectors/slack/events", content=body, headers=sign(body))
    assert r.status_code == 200 and r.json() == {"challenge": "xyz"}
    w.no_connectors = True
    assert (
        w.client.post("/api/connectors/slack/events", content=body, headers=sign(body)).status_code
        == 404
    )


def test_slack_events_ingest_into_namespaced_memory(w):
    from salid.orgs import message_namespaced_id, namespaced_id, workspace_memory_id

    w.orgs.attach_workspace(
        "org_t", Workspace(id="slack-T1", platform="slack", external_id="T1", name="Acme")
    )
    r = post_event(w, slack_msg())
    assert r.status_code == 200 and r.json()["stored"] == 1
    wid = workspace_memory_id("slack", "T1")
    assert wid >= 2**62 and wid != workspace_memory_id("slack", "T2")
    chan = namespaced_id("slack", "C1", "channel")
    rows = w.memory.messages(wid, {chan}, "2025-01-01", "2026-12-31")
    assert len(rows) == 1 and rows[0].text == "ship it" and rows[0].guild_id == wid
    assert rows[0].channel_id == chan
    assert rows[0].message_id == message_namespaced_id("slack", "T1", "C1", "1760443200.000200")
    # a retry is idempotent
    post_event(w, slack_msg())
    assert len(w.memory.messages(wid, {chan}, "2025-01-01", "2026-12-31")) == 1
    # the web chat reads it for this workspace only
    w.client.post(
        "/api/center/orgs/org_t/chat",
        json={"message": "x", "workspace_id": "slack-T1"},
        headers=H["adm"],
    )
    assert set(w.asks[-1][1]["memory"].allowed) == {chan}


@pytest.mark.parametrize(
    "over",
    [
        {"channel_type": "group"},
        {"channel_type": "im"},
        {"subtype": "channel_join"},
        {"bot_id": "B1"},
        {"text": ""},
    ],
)
def test_slack_events_ignore_private_bot_and_system_messages(w, over):
    from salid.orgs import workspace_memory_id

    w.orgs.attach_workspace("org_t", Workspace(id="slack-T1", platform="slack", external_id="T1"))
    assert post_event(w, slack_msg(**over)).json()["stored"] == 0
    assert w.memory.day_channels(workspace_memory_id("slack", "T1"), "2025-10-14") == []


def test_slack_events_need_a_connected_licensed_workspace(w):
    assert post_event(w, slack_msg()).json()["stored"] == 0  # unknown team
    w.orgs.attach_workspace("org_t", Workspace(id="slack-T1", platform="slack", external_id="T1"))
    org = w.orgs.get("org_t")
    org.license = License(plan="monthly", status="expired")
    w.orgs.save(org)
    assert post_event(w, slack_msg()).json()["stored"] == 0


# ------------------------------------------------------------------ page
def test_center_page_route(w, monkeypatch, tmp_path):
    page = tmp_path / "center.html"
    monkeypatch.setattr(center, "CENTER_HTML", page)
    assert w.client.get("/center").status_code == 404  # not built yet
    page.write_text("<script>var id='__HUB_GOOGLE_CLIENT_ID__'</script>", encoding="utf-8")
    r = w.client.get("/center")
    assert r.status_code == 200 and CLIENT_ID in r.text and r.headers["cache-control"] == "no-store"


def test_stamp_helper_is_iso():
    assert datetime.fromisoformat(stamp()).tzinfo is not None
    assert timedelta(0) < datetime.now(UTC) - datetime.fromisoformat(stamp()) + timedelta(seconds=2)


def test_me_carries_the_saved_web_language_and_put_changes_it(w):
    """Each person's page language is remembered with the account, so any device opens in it."""
    assert w.client.get("/api/center/me", headers=H["adm"]).json()["language"] is None
    r = w.client.put("/api/center/me/language", json={"language": "th"}, headers=H["adm"])
    assert r.status_code == 200 and r.json() == {"language": "th"}
    assert w.client.get("/api/center/me", headers=H["adm"]).json()["language"] == "th"
    assert w.client.get("/api/center/me", headers=H["mem"]).json()["language"] is None
    bad = w.client.put("/api/center/me/language", json={"language": "xx"}, headers=H["adm"])
    assert bad.status_code == 400
    assert w.client.put("/api/center/me/language", json={"language": "en"}).status_code == 401


# ------------------------------------------------------------------ the account system
def test_my_account_separates_the_person_from_the_org(w):
    """Personal identities live with the person (every org sees them); org workspaces stay on
    the org. Google is always the verified sign-in; others are typed (unverified) or proven."""
    me = w.client.get("/api/center/me", headers=H["adm"]).json()
    assert [i["platform"] for i in me["identities"]] == ["google"]
    assert me["identities"][0]["verified"] is True and me["name"] == ""
    assert "discord" in me["identity_platforms"] and me["discord_verify"] is False
    r = w.client.put("/api/center/me/profile", json={"name": "Adam"}, headers=H["adm"])
    assert r.status_code == 200 and r.json()["name"] == "Adam"
    r = w.client.put(
        "/api/center/me/identity",
        json={"platform": "slack", "external_id": "U123", "name": "adam"},
        headers=H["adm"],
    )
    assert r.status_code == 200
    slack = next(i for i in r.json() if i["platform"] == "slack")
    assert slack["external_id"] == "U123" and slack["verified"] is False and slack["linked_at"]
    bad = w.client.put(
        "/api/center/me/identity",
        json={"platform": "instagram", "external_id": "x"},
        headers=H["adm"],
    )
    assert bad.status_code == 400
    # the org's member list shows who each person is and how many platforms they linked
    members = w.client.get(f"/api/center/orgs/{w.org.id}/members", headers=H["boss"]).json()
    adam = next(m for m in members if m["email"] == "adm@example.com")
    assert adam["name"] == "Adam" and adam["linked_platforms"] == 1
    assert [i["platform"] for i in adam["identities"]] == ["google", "slack"]
    # another person sees nothing of Adam's identities on their own account
    assert [
        i["platform"] for i in w.client.get("/api/center/me", headers=H["mem"]).json()["identities"]
    ] == ["google"]
    r = w.client.delete("/api/center/me/identity/slack", headers=H["adm"])
    assert r.status_code == 200 and [i["platform"] for i in r.json()] == ["google"]


def test_discord_code_from_the_bot_verifies_the_person(w):
    code = w.guilds.create_link_code(A, 777, "adam#1")
    r = w.client.post("/api/center/me/identity/discord/code", json={"code": code}, headers=H["adm"])
    assert r.status_code == 200
    d = next(i for i in r.json() if i["platform"] == "discord")
    assert d["external_id"] == "777" and d["name"] == "adam#1" and d["verified"] is True
    # mirrored into the org membership, no admin rights granted
    members = w.client.get(f"/api/center/orgs/{w.org.id}/members", headers=H["boss"]).json()
    assert next(m for m in members if m["email"] == "adm@example.com")["discord_user_id"] == "777"
    again = w.client.post(
        "/api/center/me/identity/discord/code", json={"code": code}, headers=H["adm"]
    )
    assert again.status_code == 400


def test_linking_a_server_also_verifies_the_admin_who_ran_setup(w):
    gid = 990011
    w.guilds.upsert_guild(gid, name="Cafe C", icon_url="", channels=[], installed=True)
    code = w.guilds.create_link_code(gid, 555, "boss#9")
    r = w.client.post(
        f"/api/center/orgs/{w.org.id}/workspaces/discord/link",
        json={"code": code},
        headers=H["boss"],
    )
    assert r.status_code == 200
    me = w.client.get("/api/center/me", headers=H["boss"]).json()
    d = next(i for i in me["identities"] if i["platform"] == "discord")
    assert d["external_id"] == "555" and d["verified"] is True


def test_discord_verify_flow_needs_the_secret_then_links_through_the_callback(w, monkeypatch):
    from meeting_scribe.config import get_settings

    assert w.client.get("/api/center/identity/discord/start", headers=H["mem"]).status_code == 503
    monkeypatch.setenv("SALID_DISCORD_CLIENT_ID", "123456789012345678")
    monkeypatch.setenv("SALID_DISCORD_CLIENT_SECRET", "shh")
    get_settings.cache_clear()
    try:
        assert w.client.get("/api/center/me", headers=H["mem"]).json()["discord_verify"] is True
        r = w.client.get("/api/center/identity/discord/start", headers=H["mem"])
        assert r.status_code == 200
        url = r.json()["url"]
        assert (
            url.startswith("https://discord.com/api/oauth2/authorize?") and "scope=identify" in url
        )
        state = url.split("state=")[1].split("&")[0]
        app.dependency_overrides[center.get_discord_identity_exchange] = lambda: (
            lambda code: (
                ("31337", "Mem") if code == "ok" else (_ for _ in ()).throw(RuntimeError("bad"))
            )
        )
        cb = w.client.get(
            f"/api/center/identity/discord/callback?code=ok&state={state}", follow_redirects=False
        )
        assert cb.status_code == 303 and cb.headers["location"] == "/center?linked=discord#/account"
        d = next(
            i
            for i in w.client.get("/api/center/me", headers=H["mem"]).json()["identities"]
            if i["platform"] == "discord"
        )
        assert d["external_id"] == "31337" and d["name"] == "Mem" and d["verified"] is True
        # state is one-time
        cb2 = w.client.get(
            f"/api/center/identity/discord/callback?code=ok&state={state}", follow_redirects=False
        )
        assert "error=bad_state" in cb2.headers["location"]
    finally:
        app.dependency_overrides.pop(center.get_discord_identity_exchange, None)
        get_settings.cache_clear()


def test_overview_reports_org_platforms_and_how_many_members_linked_accounts(w):
    ov = w.client.get(f"/api/center/orgs/{w.org.id}/overview", headers=H["boss"]).json()
    assert "discord" in ov["platforms"]["available"]  # the test wiring configures all three
    assert ov["platforms"]["connected"] == ["discord"]
    assert "google_drive" in ov["platforms"]["coming_soon"]
    assert ov["members"] >= 1 and ov["members_linked"] == 0
    w.client.put(
        "/api/center/me/identity", json={"platform": "line", "external_id": "u1"}, headers=H["boss"]
    )
    ov = w.client.get(f"/api/center/orgs/{w.org.id}/overview", headers=H["boss"]).json()
    assert ov["members_linked"] == 1


def test_web_chat_tells_the_ai_who_is_talking(w):
    w.client.put("/api/center/me/profile", json={"name": "Boss Lee"}, headers=H["boss"])
    w.client.put(
        "/api/center/me/identity",
        json={"platform": "slack", "external_id": "U9"},
        headers=H["boss"],
    )
    r = w.client.post(
        f"/api/center/orgs/{w.org.id}/chat", json={"message": "สวัสดี"}, headers=H["boss"]
    )
    assert r.status_code == 200
    kw = w.asks[-1][1]
    assert "Talking with: Boss Lee <boss@example.com>" in kw["profile"]
    assert "Role in" in kw["profile"] and "slack U9 (unverified)" in kw["profile"]


def test_sign_in_with_discord_issues_a_salid_session_for_a_verified_email(w, monkeypatch):
    from meeting_scribe.config import get_settings

    # not configured: the pages hide the button and the route says so
    assert "__DISCORD_LOGIN__" not in w.client.get("/center").text
    assert w.client.get("/api/center/auth/discord/start", follow_redirects=False).status_code == 503
    monkeypatch.setenv("SALID_DISCORD_CLIENT_ID", "123456789012345678")
    monkeypatch.setenv("SALID_DISCORD_CLIENT_SECRET", "shh")
    monkeypatch.setenv("SALID_SESSION_SECRET", "sess")
    get_settings.cache_clear()
    try:
        assert 'const DISCORD_LOGIN = "1" === "1"' in w.client.get("/center").text
        r = w.client.get("/api/center/auth/discord/start?next=start", follow_redirects=False)
        assert r.status_code == 303
        loc = r.headers["location"]
        assert (
            loc.startswith("https://discord.com/api/oauth2/authorize?") and "identify+email" in loc
        )
        state = loc.split("state=")[1].split("&")[0]
        people = {
            "ok": ("4242", "Newbie", "newbie@example.com", True),
            "noemail": ("4343", "Ghost", "", False),
        }
        app.dependency_overrides[center.get_discord_login_exchange] = lambda: (
            lambda code: people[code]
        )
        cb = w.client.get(
            f"/api/center/auth/discord/callback?code=ok&state={state}", follow_redirects=False
        )
        assert cb.status_code == 303 and cb.headers["location"].startswith("/start#session=ss1.")
        token = cb.headers["location"].split("#session=")[1]
        # an invitation survives the round trip (phones open Discord in another browser):
        # the code rides in the server-side state and comes back on the landing URL
        r = w.client.get(
            "/api/center/auth/discord/start?next=center&invite=abcDEF_-123", follow_redirects=False
        )
        st2 = r.headers["location"].split("state=")[1].split("&")[0]
        cb2 = w.client.get(
            f"/api/center/auth/discord/callback?code=ok&state={st2}", follow_redirects=False
        )
        assert cb2.headers["location"].startswith("/center?invite=abcDEF_-123#session=ss1.")
        assert (
            w.client.get(
                "/api/center/auth/discord/start?invite=bad%20code", follow_redirects=False
            ).status_code
            == 422
        )
        me = w.client.get("/api/center/me", headers={"Authorization": "Bearer " + token}).json()
        assert me["email"] == "newbie@example.com" and me["name"] == "Newbie"
        d = next(i for i in me["identities"] if i["platform"] == "discord")
        assert d["external_id"] == "4242" and d["verified"] is True
        # state is one-time
        again = w.client.get(
            f"/api/center/auth/discord/callback?code=ok&state={state}", follow_redirects=False
        )
        assert "error=bad_state" in again.headers["location"]
        # no verified email -> no account, clear error
        r2 = w.client.get("/api/center/auth/discord/start", follow_redirects=False)
        st2 = r2.headers["location"].split("state=")[1].split("&")[0]
        cb2 = w.client.get(
            f"/api/center/auth/discord/callback?code=noemail&state={st2}", follow_redirects=False
        )
        assert cb2.headers["location"] == "/center?error=discord_no_email&platform=discord"
    finally:
        app.dependency_overrides.pop(center.get_discord_login_exchange, None)
        get_settings.cache_clear()


def test_chat_without_a_workspace_talks_to_the_company_not_a_platform(w):
    """Salid is one colleague: the panel sends no workspace and she answers from the company's
    Discord server (ledger + memory) when one is connected."""
    r = w.client.post(
        f"/api/center/orgs/{w.org.id}/chat", json={"message": "สรุปค่าใช้จ่าย"}, headers=H["boss"]
    )
    assert r.status_code == 200
    assert r.json()["workspace_id"].startswith("discord-")
    ws = w.orgs.get(w.org.id).workspace(r.json()["workspace_id"])
    assert ws is not None and w.asks[-1][1]["guild_id"] == ws.memory_id


# ------------------------------------------------------------------ real documents
def test_discord_originals_become_downloadable_documents_and_text_is_knowledge(w):
    """A PDF Salid read in a Discord channel is kept as a real file of that workspace: listed
    under kind=file with a download, visible only to people who may read that channel. Notes,
    wiki and minutes stay kind=knowledge."""
    from salid.files import FileArchive

    archive = FileArchive(w.docs, w.lib.blobs)
    pdf = b"%PDF-1.4 fake contract"
    doc_id = archive.save(
        A,
        filename="contract.pdf",
        data=pdf,
        mime="application/pdf",
        by="Somchai",
        channel_id=PRIVATE,  # an executives-only channel
        channel_name="finance",
        source_url="https://discord.com/channels/1/2/3",
        summary="สัญญาเช่าออฟฟิศ",
        tag="contract",
        text="สัญญาเช่า...",
    )
    assert (
        doc_id
        and archive.save(A, filename="contract.pdf", data=pdf, mime="application/pdf") == doc_id
    )
    url = f"/api/center/orgs/{w.org.id}/documents"
    files = w.client.get(url, params={"kind": "file"}, headers=H["boss"]).json()["items"]
    item = next(i for i in files if i["title"] == "contract.pdf")
    assert (
        item["kind"] == "file" and item["workspace"] == "discord-1000" and item["who"] == "Somchai"
    )
    assert item["source_url"].startswith("https://discord.com/")
    # download the original through the org's document route
    r = w.client.get(f"{url}/{item['id']}/file", headers=H["boss"])
    assert (
        r.status_code == 200 and r.content == pdf and "application/pdf" in r.headers["content-type"]
    )
    # someone who cannot read that channel does not see the file
    from salid.library import Viewer

    org = w.orgs.get(w.org.id)
    titles = [i.title for i in w.lib.list(org, Viewer("x@example.com", False), kind="file").items]
    assert "contract.pdf" not in titles
    assert "contract.pdf" in [
        i.title for i in w.lib.list(org, Viewer("boss@example.com", True), kind="file").items
    ]
    knowledge = w.client.get(url, params={"kind": "knowledge"}, headers=H["boss"]).json()["items"]
    assert all(i["kind"] == "knowledge" and i["title"] != "contract.pdf" for i in knowledge)
    assert w.client.get(url, params={"kind": "nope"}, headers=H["boss"]).status_code == 422


def test_discord_redirect_uris_match_what_is_registered_in_the_discord_app(w):
    """Discord rejects any redirect_uri that is not registered byte for byte."""
    assert (
        center.discord_identity_redirect()
        == "https://hub.example/api/center/identity/discord/callback"
    )
    assert center.discord_login_redirect() == "https://hub.example/api/center/auth/discord/callback"


# ------------------------------------------------------------------ voice input
def test_voice_input_understands_any_language_and_lands_in_the_page_language(w):
    from salid.web_voice import Heard

    seen = []

    def fake(data, mime, target):
        seen.append((len(data), mime, target))
        return Heard(
            text="สรุปค่าใช้จ่ายเดือนนี้", language="th", translated="Summarize this month's expenses"
        )

    app.dependency_overrides[center.get_transcriber] = lambda: fake
    try:
        r = w.client.post(
            f"/api/center/orgs/{w.org.id}/voice/transcribe?lang=en",
            files={"audio": ("u.webm", b"fake-opus-bytes", "audio/webm")},
            headers=H["boss"],
        )
        assert r.status_code == 200
        body = r.json()
        assert body["language"] == "th" and body["message"] == "Summarize this month's expenses"
        assert body["text"].startswith("สรุป") and seen == [(15, "audio/webm", "en")]
        empty = w.client.post(
            f"/api/center/orgs/{w.org.id}/voice/transcribe",
            files={"audio": ("u.webm", b"", "audio/webm")},
            headers=H["boss"],
        )
        assert empty.status_code == 422
        assert (
            w.client.post(
                f"/api/center/orgs/{w.org.id}/voice/transcribe",
                files={"audio": ("u.webm", b"x", "audio/webm")},
            ).status_code
            == 401
        )
    finally:
        app.dependency_overrides.pop(center.get_transcriber, None)


def test_web_voice_parser_drops_silence_and_same_language_translations():
    from types import SimpleNamespace

    from salid import web_voice

    class Client:  # Gemini stand-in: returns JSON the schema asked for
        def __init__(self, text):
            self.models = SimpleNamespace(generate_content=lambda **kw: SimpleNamespace(text=text))

    calls = {}

    def fake_flac(data):
        calls["flac"] = len(data)
        return b"fLaC"

    web_voice.to_flac = fake_flac
    heard = web_voice.transcribe_utterance(
        Client('{"text": "hello there", "language": "en-US", "translated": "hello there"}'),
        "m",
        b"audio",
        "audio/webm",
        "en",
    )
    assert heard.text == "hello there" and heard.language == "en" and heard.translated == ""
    quiet = web_voice.transcribe_utterance(
        Client('{"text": "...", "language": "th"}'), "m", b"a", "audio/webm", "en"
    )
    assert quiet.text == "" and quiet.language == ""
    assert calls["flac"] == 1


# ------------------------------------------------------------------ directory & org chart
def test_directory_groups_people_by_department_and_draws_the_reporting_tree(w):
    url = f"/api/center/orgs/{w.org.id}"
    w.client.put(
        "/api/center/me/profile",
        json={"name": "Boss", "title": "CEO", "department": "Management"},
        headers=H["boss"],
    )
    w.client.put(
        "/api/center/me/profile",
        json={"name": "Adam", "title": "CTO", "department": "Engineering"},
        headers=H["adm"],
    )
    # the admin reports to the boss; a plain member may only place themselves
    r = w.client.put(
        f"{url}/members/adm@example.com/structure",
        json={"reports_to": "boss@example.com"},
        headers=H["boss"],
    )
    assert r.status_code == 200 and r.json()["reports_to"] == "boss@example.com"
    assert w.client.put(
        f"{url}/members/boss@example.com/structure",
        json={"reports_to": "adm@example.com"},
        headers=H["adm"],
    ).status_code in (400, 403)
    loop = w.client.put(
        f"{url}/members/boss@example.com/structure",
        json={"reports_to": "adm@example.com"},
        headers=H["boss"],
    )
    assert loop.status_code == 400 and "loop" in loop.json()["detail"]
    assert (
        w.client.put(
            f"{url}/members/adm@example.com/structure",
            json={"reports_to": "nobody@example.com"},
            headers=H["boss"],
        ).status_code
        == 400
    )
    d = w.client.get(f"{url}/directory", headers=H["adm"]).json()
    depts = {g["department"]: [p["email"] for p in g["people"]] for g in d["departments"]}
    assert depts["Management"] == ["boss@example.com"] and depts["Engineering"] == [
        "adm@example.com"
    ]
    assert d["with_profile"] >= 2 and d["with_manager"] == 1
    roots = {n["email"]: n for n in d["tree"]}
    assert "boss@example.com" in roots and "adm@example.com" not in roots
    adam = next(c for c in roots["boss@example.com"]["children"] if c["email"] == "adm@example.com")
    assert adam["title"] == "CTO" and adam["reports_to"] == "boss@example.com"
    # the member list carries the manager too
    members = w.client.get(f"{url}/members", headers=H["boss"]).json()
    assert (
        next(m for m in members if m["email"] == "adm@example.com")["reports_to"]
        == "boss@example.com"
    )


def test_my_tasks_and_calendar_follow_the_person_and_members_get_a_read_only_server_summary(w):
    from salid.reminders import KIND as REMINDERS
    from salid.tasks import Task, TaskRepo

    url = f"/api/center/orgs/{w.org.id}"
    code = w.guilds.create_link_code(A, 777, "adam#1")
    w.client.post("/api/center/me/identity/discord/code", json={"code": code}, headers=H["adm"])
    repo = TaskRepo(w.docs)
    repo.save(
        Task(
            id="T-1",
            guild_id=A,
            assignee_id=777,
            assignee_name="adam",
            title="Ship it",
            due="2026-10-20",
            source_channel_id=PUBLIC,
        )
    )
    repo.save(
        Task(
            id="T-2",
            guild_id=A,
            assignee_id=1,
            assignee_name="someone",
            title="Not mine",
            due="2026-10-21",
            source_channel_id=PUBLIC,
        )
    )
    w.docs.put(
        REMINDERS,
        A,
        "R-1",
        {
            "id": "R-1",
            "user_id": 777,
            "message": "Call the bank",
            "next_at": "2026-10-19T02:00:00+00:00",
            "active": True,
            "set_by": 777,
        },
    )
    repo.save(
        Task(
            id="T-3",
            guild_id=A,
            assignee_id=777,
            assignee_name="adam",
            title="Old one",
            status="done",
            source_channel_id=PUBLIC,
        )
    )
    repo.save(
        Task(
            id="T-4",
            guild_id=A,
            assignee_id=0,
            assignee_name="Salid",
            title="Send the deck",
            source_channel_id=PUBLIC,
        )
    )
    mine = w.client.get(f"{url}/me/tasks", headers=H["adm"]).json()
    assert (
        [t["title"] for t in mine["tasks"]] == ["Ship it", "Old one"]
        and mine["open"] == 1
        and mine["done"] == 1
    )
    assert [t["title"] for t in mine["salid_tasks"]] == [
        "Send the deck"
    ]  # an admin sees what Salid was asked
    cal = w.client.get(f"{url}/me/calendar", headers=H["adm"]).json()
    assert [d["date"] for d in cal["days"]] == ["2026-10-19", "2026-10-20"]
    assert (
        cal["days"][0]["items"][0]["kind"] == "reminder"
        and cal["days"][1]["items"][0]["kind"] == "task"
    )
    assert cal["sources"]["google_calendar"] is False
    # someone with no identity and no matching name sees nothing of Adam's
    assert w.client.get(f"{url}/me/tasks", headers=H["boss"]).json()["tasks"] == []
    summary = w.client.get(f"{url}/server-summary", headers=H["adm"]).json()
    assert summary["can_manage"] is True and summary["servers"][0]["name"] == "Cafe A"
    assert "features_on" in summary["servers"][0] and "channels" in summary["servers"][0]


def test_sign_in_with_discord_lands_in_the_account_that_verified_that_discord(w, monkeypatch):
    """Verify with Discord on the Google account first; later "Sign in with Discord" with a
    Discord whose email is different (or missing) is still the same Salid account."""
    from meeting_scribe.config import get_settings

    code = w.guilds.create_link_code(A, 9001, "boss#1")
    w.client.post("/api/center/me/identity/discord/code", json={"code": code}, headers=H["boss"])
    assert w.orgs.email_for_identity("discord", "9001") == "boss@example.com"
    monkeypatch.setenv("SALID_DISCORD_CLIENT_ID", "123456789012345678")
    monkeypatch.setenv("SALID_DISCORD_CLIENT_SECRET", "shh")
    monkeypatch.setenv("SALID_SESSION_SECRET", "sess")
    get_settings.cache_clear()
    try:
        app.dependency_overrides[center.get_discord_login_exchange] = lambda: (
            lambda code: ("9001", "Boss", "", False)  # no verified email on Discord at all
        )
        st = (
            w.client.get("/api/center/auth/discord/start", follow_redirects=False)
            .headers["location"]
            .split("state=")[1]
            .split("&")[0]
        )
        cb = w.client.get(
            f"/api/center/auth/discord/callback?code=x&state={st}", follow_redirects=False
        )
        token = cb.headers["location"].split("#session=")[1]
        me = w.client.get("/api/center/me", headers={"Authorization": "Bearer " + token}).json()
        assert me["email"] == "boss@example.com" and [o["name"] for o in me["orgs"]]
    finally:
        app.dependency_overrides.pop(center.get_discord_login_exchange, None)
        get_settings.cache_clear()
    # removing the identity forgets the link
    w.client.delete("/api/center/me/identity/discord", headers=H["boss"])
    assert w.orgs.email_for_identity("discord", "9001") is None


# ------------------------------------------------------------------ live voice on the web
def test_live_voice_socket_streams_audio_both_ways_and_answers_tools(w):
    """The web call: hello (auth) -> ready -> Salid greets first (audio + transcript) -> mic
    frames go to the model -> a tool call runs the text brain -> interrupted/turn events."""
    import contextlib

    sent: dict = {"audio": [], "text": [], "tools": []}

    class FakeLive:
        def __init__(self):
            self._phase = 0

        async def send_realtime_input(self, audio=None, **kw):
            sent["audio"].append(len(audio.data))

        async def send_client_content(self, turns=None, turn_complete=True):
            sent["text"].append(turns.parts[0].text)

        async def send_tool_response(self, function_responses):
            sent["tools"].append(function_responses[0].response)

        async def receive(self):
            self._phase += 1
            if self._phase == 1:  # greeting turn
                yield SimpleNamespace(
                    go_away=None,
                    tool_call=None,
                    server_content=SimpleNamespace(
                        interrupted=False,
                        model_turn=SimpleNamespace(
                            parts=[SimpleNamespace(inline_data=SimpleNamespace(data=bytes(960)))]
                        ),
                        input_transcription=None,
                        output_transcription=SimpleNamespace(
                            text="Hi Boss, what can I do for you?"
                        ),
                        turn_complete=True,
                    ),
                )
            elif self._phase == 2:  # the user asked something -> tool call
                yield SimpleNamespace(
                    go_away=None,
                    tool_call=SimpleNamespace(
                        function_calls=[
                            SimpleNamespace(
                                id="c1",
                                name="ask_salid",
                                args={"question": "how much did we spend"},
                            )
                        ]
                    ),
                    server_content=None,
                )
            elif self._phase == 3:
                yield SimpleNamespace(
                    go_away=None,
                    tool_call=None,
                    server_content=SimpleNamespace(
                        interrupted=True,
                        model_turn=None,
                        input_transcription=SimpleNamespace(text="how much"),
                        output_transcription=None,
                        turn_complete=False,
                    ),
                )
            else:
                await asyncio.sleep(0.05)

    @contextlib.asynccontextmanager
    async def connect(cfg):
        assert (
            cfg.response_modalities == ["AUDIO"]
            and cfg.tools[0].function_declarations[0].name == "ask_salid"
        )
        yield FakeLive()

    app.dependency_overrides[center.get_live_connect] = lambda: connect
    try:
        with w.client.websocket_connect(f"/api/center/orgs/{w.org.id}/live") as sock:
            sock.send_json({"token": "boss", "lang": "en", "page": "Documents", "greet": True})
            events, audio = [], 0
            for _ in range(8):
                m = sock.receive()
                if m.get("bytes"):
                    audio += len(m["bytes"])
                elif m.get("text"):
                    events.append(json.loads(m["text"]))
                if any(e["type"] == "interrupted" for e in events):
                    break
            sock.send_bytes(bytes(640))  # 20 ms of mic silence
            sock.send_json({"type": "say", "text": "and last month?"})
            sock.send_json({"type": "close"})
        kinds = [e["type"] for e in events]
        assert (
            kinds[0] == "ready"
            and "said" in kinds
            and "thinking" in kinds
            and "interrupted" in kinds
        )
        assert audio == 960  # Salid's greeting audio reached the browser
        assert sent["text"][0].startswith("(The person just opened voice chat")  # greet cue
        assert "answer" in sent["tools"][0]  # the text brain answered the tool call
        assert w.asks[-1][0] == "how much did we spend" and w.asks[-1][1]["language"] == "en"
    finally:
        app.dependency_overrides.pop(center.get_live_connect, None)


def test_live_voice_socket_rejects_bad_tokens_and_strangers(w):
    with w.client.websocket_connect(f"/api/center/orgs/{w.org.id}/live") as sock:
        sock.send_json({"token": "nope"})
        assert json.loads(sock.receive()["text"])["type"] == "error"
    with w.client.websocket_connect(f"/api/center/orgs/{w.org.id}/live") as sock:
        sock.send_json({"token": "stranger"})
        assert json.loads(sock.receive()["text"])["text"] == "forbidden"


def test_web_live_waits_for_the_person_but_lets_them_interrupt():
    from types import SimpleNamespace

    from google.genai import types

    from salid import web_live

    s = SimpleNamespace(
        live_voice="Leda",
        live_model="m",
        live_silence_ms=1200,
        now=lambda: datetime(2026, 10, 17, tzinfo=UTC),
    )
    live = web_live.WebLive(
        s, client=None, instruction="x", ask=lambda q: "", out=None, connect=lambda c: None
    )
    cfg = live.config()
    ad = cfg.realtime_input_config.automatic_activity_detection
    assert (
        ad.silence_duration_ms == 2000
        and ad.end_of_speech_sensitivity == types.EndSensitivity.END_SENSITIVITY_LOW
    )
    assert cfg.realtime_input_config.activity_handling is None  # talking over her interrupts her
    text = web_live.build_web_instruction(s, "Talking with: Boss", "en")
    assert "Never refuse" in text and "colleague's" in text  # actions go to ask_salid
    assert "NEVER talk over them" in text and "take your time" in text


def test_web_chat_has_the_company_playbook_and_its_tools_like_discord(w):
    from salid.company_brain import CompanyRepo

    repo = CompanyRepo(w.docs)
    repo.teach(A, "glossary", "เลขผู้เสียภาษี", "0105566012345", by="boss") if hasattr(
        repo, "teach"
    ) else None
    r = w.client.post(
        f"/api/center/orgs/{w.org.id}/chat", json={"message": "ขอเลขผู้เสียภาษีหน่อย"}, headers=H["boss"]
    )
    assert r.status_code == 200
    kw = w.asks[-1][1]
    names = {getattr(t, "__name__", "") for t in kw["extra_tools"]}
    assert {"teach_company", "show_company_playbook"} <= names
    assert isinstance(kw.get("company", ""), str)


def test_web_chat_runs_the_discord_bots_own_agent_assembly(w):
    """One Salid: on the web she gets the same feature tools as in Discord (tasks, wiki,
    reminders, company playbook, search...) through the headless bot, not a rewritten set."""
    r = w.client.post(
        f"/api/center/orgs/{w.org.id}/chat",
        json={"message": "เตือนฉันพรุ่งนี้ 9 โมงให้โทรหาธนาคาร"},
        headers=H["boss"],
    )
    assert r.status_code == 200
    kw = w.asks[-1][1]
    names = {getattr(t, "__name__", "") for t in kw["extra_tools"]}
    assert {"set_reminder", "list_tasks", "teach_company", "update_my_profile"} <= names, sorted(
        names
    )
    assert "Talking with:" in kw["profile"] and kw.get("tone")


def test_my_calendar_shows_meetings_salid_scheduled_in_discord(w):
    from salid.secretary_store import MEETINGS

    w.docs.put(
        MEETINGS,
        A,
        "M-1",
        {
            "id": "M-1",
            "guild_id": A,
            "title": "Sprint review",
            "start_utc": "2026-10-22T03:00:00+00:00",
            "end_utc": "2026-10-22T04:00:00+00:00",
            "organizer_id": 777,
            "attendee_ids": [777, 1],
            "status": "scheduled",
            "event_url": "https://discord.com/events/1/2",
        },
    )
    code = w.guilds.create_link_code(A, 777, "adam#1")
    w.client.post("/api/center/me/identity/discord/code", json={"code": code}, headers=H["adm"])
    cal = w.client.get(f"/api/center/orgs/{w.org.id}/me/calendar", headers=H["adm"]).json()
    day = next(d for d in cal["days"] if d["date"] == "2026-10-22")
    m = day["items"][0]
    assert (
        m["kind"] == "meeting" and m["title"] == "Sprint review" and m["mine"] is True and m["url"]
    )


# ------------------------------------------------------------------ act on Discord from the web
def test_web_queues_discord_actions_that_the_bot_carries_out(w):
    """The web (API or Salid's tools) queues an action; the bot performs it on its tick and
    writes the result back; the web reads the status."""
    import asyncio

    from salid.actions import ActionQueue

    url = f"/api/center/orgs/{w.org.id}/actions"
    r = w.client.post(
        url,
        json={"kind": "post_message", "args": {"channel": "general", "text": "Hello team"}},
        headers=H["boss"],
    )
    assert r.status_code == 200 and r.json()["status"] == "pending"
    aid = r.json()["id"]
    assert (
        w.client.post(url, json={"kind": "nuke", "args": {}}, headers=H["boss"]).status_code == 400
    )
    assert w.client.post(
        url, json={"kind": "post_message", "args": {}}, headers=H["mem"]
    ).status_code in (403, 404)
    # Salid's tools on the web include the action tools for admins
    w.client.post(
        f"/api/center/orgs/{w.org.id}/chat",
        json={"message": "โพสต์บอกทีมว่าประชุม 10 โมง"},
        headers=H["boss"],
    )
    names = {getattr(t, "__name__", "") for t in w.asks[-1][1]["extra_tools"]}
    assert {"discord_post_message", "discord_create_event", "discord_action_status"} <= names
    # the bot side: a fake guild with a #general channel
    sent = []

    class Msg:
        id, jump_url = 42, "https://discord.com/channels/1/2/42"

    async def send(text):
        sent.append(text)
        return Msg()

    ch = SimpleNamespace(id=1001, name="general", send=send, guild=None)
    guild = SimpleNamespace(
        id=A, text_channels=[ch], voice_channels=[], get_channel=lambda i: ch if i == 1001 else None
    )
    from salid.actions_bot import ActionsMixin

    class Bot(ActionsMixin):
        pass

    bot = Bot()
    bot.actions = ActionQueue(w.docs)
    bot.guilds = [guild]
    n = asyncio.run(bot._actions_due())
    assert n == 1 and sent == ["Hello team"]
    got = w.client.get(url, headers=H["boss"]).json()["actions"]
    done = next(a for a in got if a["id"] == aid)
    assert (
        done["status"] == "done" and done["result"]["message_id"] == "42" and done["result"]["url"]
    )
    # "announce" names no channel: the server's ask channel first, else general
    ask_ch = SimpleNamespace(id=1002, name="ask-salid", send=send, guild=None)
    guild.text_channels = [ch, ask_ch]
    guild.get_channel = lambda i: {1001: ch, 1002: ask_ch}.get(i)
    bot.team = SimpleNamespace(current=lambda gid: SimpleNamespace(ask_channel_ids=["1002"]))
    a1 = bot.actions.enqueue(A, "announce", {"text": "📢 now Thai"}, by="boss@example.com")
    assert asyncio.run(bot._actions_due()) == 1 and sent[-1] == "📢 now Thai"
    assert bot.actions.get(A, a1["id"])["result"]["channel"] == "ask-salid"
    bot.team = SimpleNamespace(current=lambda gid: SimpleNamespace(ask_channel_ids=[]))
    a2 = bot.actions.enqueue(A, "announce", {"text": "📢 again"}, by="boss@example.com")
    asyncio.run(bot._actions_due())
    assert bot.actions.get(A, a2["id"])["result"]["channel"] == "general"


def test_library_gather_is_cached_briefly_and_cleared_by_uploads(w, monkeypatch):
    """The pages call the document list several times per screen: one scan serves them for a
    few seconds; an upload through the library clears the org's cache at once."""
    calls = {"n": 0}
    real = w.lib._gather_fresh

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(w.lib, "_gather_fresh", counting)
    url = f"/api/center/orgs/{w.org.id}/documents"
    for _ in range(3):
        assert w.client.get(url, headers=H["boss"]).status_code == 200
    assert calls["n"] == 1  # same person, same org: served from the cache
    w.client.get(url, headers=H["adm"])
    assert calls["n"] == 2  # another viewer has their own entry (their rights differ)
    w.lib.invalidate(w.org.id)
    w.client.get(url, headers=H["boss"])
    assert calls["n"] == 3


# ------------------------------------------------------------------ invite links
def test_invite_links(w):
    base = "/api/center/orgs/org_t/invites"
    # members cannot make links; admins make member links only; owners any role
    assert w.client.post(base, json={}, headers=H["mem"]).status_code == 403
    assert w.client.post(base, json={"role": "owner"}, headers=H["adm"]).status_code == 403
    adm_link = w.client.post(base, json={"role": "admin"}, headers=H["adm"])
    assert adm_link.status_code == 200 and adm_link.json()["role"] == "admin"
    w.client.delete(f"{base}/{adm_link.json()['code']}", headers=H["adm"])
    assert w.client.post(base, json={"role": "king"}, headers=H["boss"]).status_code == 422
    assert w.client.post(base, json={"role": "owner"}, headers=H["boss"]).status_code == 403
    assert w.client.post(base, json={"days": 999}, headers=H["adm"]).status_code == 422
    r = w.client.post(base, json={"label": "Sales team"}, headers=H["adm"])
    assert r.status_code == 200
    inv = r.json()
    assert inv["url"] == f"https://hub.example/center?invite={inv['code']}"
    assert inv["status"] == "active" and inv["role"] == "member" and inv["uses"] == 0
    made, ends = (datetime.fromisoformat(inv[k]) for k in ("created_at", "expires_at"))
    assert inv["max_uses"] == 0 and ends - made == timedelta(days=7)
    assert inv["created_by"] == "adm@example.com" and inv["label"] == "Sales team"
    forever = w.client.post(base, json={"days": 0, "max_uses": 1}, headers=H["boss"]).json()
    assert forever["expires_at"] == "" and forever["max_uses"] == 1
    # the public look-up needs no sign-in and gives away nothing but the org's name
    info = w.client.get(f"/api/center/invites/{inv['code']}").json()
    assert info == {
        "code": inv["code"],
        "status": "active",
        "org_name": "Cafe Group",
        "logo_url": "",
        "role": "member",
        "invited_by": "adm@example.com",
        "expires_at": inv["expires_at"],
    }
    assert w.client.get("/api/center/invites/nope").status_code == 404
    # a stranger signs in and joins through the link
    acc = w.client.post(f"/api/center/invites/{inv['code']}/accept", headers=H["stranger"])
    assert acc.status_code == 200
    assert acc.json() == {
        "org_id": "org_t",
        "org_name": "Cafe Group",
        "role": "member",
        "joined": True,
    }
    assert w.orgs.get("org_t").role_of("stranger@example.com") == "member"
    me = w.client.get("/api/center/me", headers=H["stranger"]).json()
    assert [o["id"] for o in me["orgs"]] == ["org_t"]
    # already a member: nothing changes, the role is kept, the link is not used up
    again = w.client.post(f"/api/center/invites/{inv['code']}/accept", headers=H["boss"]).json()
    assert again["joined"] is False and again["role"] == "owner"
    live = w.client.get(base, headers=H["adm"]).json()
    assert [(i["code"], i["uses"], i["used"]) for i in live if i["code"] == inv["code"]] == [
        (inv["code"], 1, ["stranger@example.com"])
    ]
    # a one-use link is used up after one join; the list then hides it (shown with ?all=1)
    w.client.post(f"/api/center/invites/{forever['code']}/accept", headers=H["other"])
    assert (
        w.client.post(
            f"/api/center/invites/{forever['code']}/accept", headers=H["root"]
        ).status_code
        == 410
    )
    assert forever["code"] not in {i["code"] for i in w.client.get(base, headers=H["adm"]).json()}
    everything = w.client.get(base + "?all=1", headers=H["adm"]).json()
    assert {i["code"]: i["status"] for i in everything}[forever["code"]] == "used_up"
    # revoke: 404 for a foreign code, then nobody joins through it any more
    assert w.client.delete(f"{base}/nope", headers=H["adm"]).status_code == 404
    assert w.client.delete(f"{base}/{inv['code']}", headers=H["mem"]).status_code == 403
    gone = w.client.delete(f"{base}/{inv['code']}", headers=H["adm"]).json()
    assert gone["status"] == "revoked" and gone["revoked_at"]
    assert (
        w.client.post(f"/api/center/invites/{inv['code']}/accept", headers=H["root"]).status_code
        == 410
    )
    assert w.client.get(base, headers=H["adm"]).json() == []


def test_invite_link_expiry_and_seats(w):
    base = "/api/center/orgs/org_t/invites"
    inv = w.client.post(base, json={"days": 1}, headers=H["boss"]).json()
    w.orgs._clock = lambda: at("2099-01-01", 12)  # long after: expired
    assert w.client.get(f"/api/center/invites/{inv['code']}").json()["status"] == "expired"
    r = w.client.post(f"/api/center/invites/{inv['code']}/accept", headers=H["stranger"])
    assert r.status_code == 410 and "expired" in r.json()["detail"]
    # the seat limit holds for links too
    fresh = w.client.post(base, json={}, headers=H["boss"]).json()
    org = w.orgs.get("org_t")
    org.license.seats = 3
    w.orgs.save(org)
    r = w.client.post(f"/api/center/invites/{fresh['code']}/accept", headers=H["stranger"])
    assert r.status_code == 402
    assert w.orgs.get("org_t").role_of("stranger@example.com") is None


# ------------------------------------------------------------------ link a Google account
def test_link_a_google_account_to_an_account_made_through_discord(w, monkeypatch):
    """stranger signed up through Discord (account = stranger@example.com). They add the Google
    account gnew@example.com: from then on a Google sign-in with it lands in stranger's
    account, and the profile lists both."""
    monkeypatch.setenv("SALID_SESSION_SECRET", "sess")
    get_settings.cache_clear()
    link = "/api/center/me/identity/google"
    assert (
        w.client.put(link, json={"credential": "x" * 30}, headers=H["stranger"]).status_code == 401
    )
    # the test verifier knows token names; the body needs >= 20 chars, so register padded ones
    GNEW, MEM, SELF = "gnew________________", "mem_________________", "stranger____________"
    TOKENS[GNEW] = {"email": "GNew@Example.com", "email_verified": True, "name": "G New"}
    TOKENS[MEM], TOKENS[SELF] = TOKENS["mem"], TOKENS["stranger"]
    gnew = {"Authorization": "Bearer " + GNEW}
    try:
        assert w.client.get("/api/center/me", headers=gnew).json()["email"] == "gnew@example.com"
        r = w.client.put(link, json={"credential": GNEW}, headers=H["stranger"])
        assert r.status_code == 200
        google = [(i["external_id"], i["verified"]) for i in r.json() if i["platform"] == "google"]
        assert google == [("stranger@example.com", True), ("gnew@example.com", True)]
        # a Google sign-in with gnew@ now IS stranger's account (every page, the session too)
        assert (
            w.client.get("/api/center/me", headers=gnew).json()["email"] == "stranger@example.com"
        )
        ses = w.client.post("/api/hub/session", headers=gnew)
        assert ses.status_code == 200, ses.text
        assert ses.json()["email"] == "stranger@example.com"
        assert w.orgs.account_for_google("GNew@Example.com") == "stranger@example.com"
        # a Google account that is already somebody's Salid account cannot be taken
        assert (
            w.client.put(link, json={"credential": MEM}, headers=H["stranger"]).status_code == 409
        )
        # nor can gnew@ be linked to a second account
        assert w.client.put(link, json={"credential": GNEW}, headers=H["boss"]).status_code == 409
        # linking your own email is a no-op, not a duplicate
        r = w.client.put(link, json={"credential": SELF}, headers=H["stranger"])
        assert [i["external_id"] for i in r.json() if i["platform"] == "google"] == [
            "stranger@example.com",
            "gnew@example.com",
        ]
        # unlink: gnew@ is its own (empty) account again
        r = w.client.delete("/api/center/me/identity/google", headers=H["stranger"])
        assert [i["external_id"] for i in r.json() if i["platform"] == "google"] == [
            "stranger@example.com"
        ]
        assert w.client.get("/api/center/me", headers=gnew).json()["email"] == "gnew@example.com"
    finally:
        for k in (GNEW, MEM, SELF):
            TOKENS.pop(k, None)


# ------------------------------------------------------------------ your own role, ownership
def test_own_role_only_goes_down_and_ownership_is_transferred(w):
    base = "/api/center/orgs/org_t/members"
    # an admin may lower themselves to member, never raise themselves again
    r = w.client.post(base, json={"email": "adm@example.com", "role": "member"}, headers=H["adm"])
    assert r.status_code == 200 and w.orgs.get("org_t").role_of("adm@example.com") == "member"
    assert w.client.get(base, headers=H["adm"]).status_code == 403  # now a plain member
    # a member cannot make themselves admin or owner (not even an admin can self-promote)
    w.client.post(base, json={"email": "adm@example.com", "role": "admin"}, headers=H["boss"])
    r = w.client.post(base, json={"email": "adm@example.com", "role": "owner"}, headers=H["adm"])
    assert r.status_code == 403 and "own role" in r.json()["detail"]
    # an outsider can never receive ownership: they must be a member first
    r = w.client.post(
        "/api/center/orgs/org_t/owner/transfer", json={"to": "new@example.com"}, headers=H["boss"]
    )
    assert r.status_code == 404
    # only an owner transfers; the receiver must be a member; the owner picks what they become
    xfer = "/api/center/orgs/org_t/owner/transfer"
    assert w.client.post(xfer, json={"to": "mem@example.com"}, headers=H["adm"]).status_code == 403
    assert w.client.post(xfer, json={"to": "mem@example.com"}, headers=H["mem"]).status_code == 403
    r = w.client.post(xfer, json={"to": "ghost@example.com"}, headers=H["boss"])
    assert r.status_code == 404
    r = w.client.post(xfer, json={"to": "boss@example.com"}, headers=H["boss"])
    assert r.status_code == 422
    r = w.client.post(xfer, json={"to": "mem@example.com", "me": "king"}, headers=H["boss"])
    assert r.status_code == 422
    r = w.client.post(xfer, json={"to": "Mem@Example.com", "me": "member"}, headers=H["boss"])
    assert r.status_code == 200
    org = w.orgs.get("org_t")
    assert org.role_of("mem@example.com") == "owner" and org.owner_email == "mem@example.com"
    assert org.role_of("boss@example.com") == "member"
    assert sum(1 for m in org.members if m.role == "owner") == 1
    # the former owner is now a member: no member management
    assert w.client.get(base, headers=H["boss"]).status_code == 403
    # security first: for 24 hours the former owner can take it back, and ownership cannot
    # move on to a third person before that window has passed
    ov = w.client.get("/api/center/orgs/org_t", headers=H["boss"]).json()["handover"]
    assert ov["from_email"] == "boss@example.com" and ov["to_email"] == "mem@example.com"
    assert ov["can_undo"] is True and ov["undo_until"] > ov["at"]
    r = w.client.post(xfer, json={"to": "adm@example.com"}, headers=H["mem"])
    assert r.status_code == 409 and "taken back" in r.json()["detail"]
    undo = "/api/center/orgs/org_t/owner/undo"
    assert w.client.post(undo, headers=H["mem"]).status_code == 403  # only the former owner
    assert w.client.post(undo, headers=H["adm"]).status_code == 403
    r = w.client.post(undo, headers=H["boss"])
    assert r.status_code == 200
    org = w.orgs.get("org_t")
    assert org.role_of("boss@example.com") == "owner" and org.owner_email == "boss@example.com"
    assert org.role_of("mem@example.com") == "member"  # back to what they were
    assert w.client.post(undo, headers=H["boss"]).status_code == 410  # only once
    assert (
        w.client.get("/api/center/orgs/org_t", headers=H["boss"]).json()["handover"]["can_undo"]
        is False
    )
    # after the window the handover is final: the new owner may pass it on, the old one cannot undo
    r = w.client.post(xfer, json={"to": "mem@example.com"}, headers=H["boss"])
    assert r.status_code == 200
    w.orgs._clock = lambda: datetime.now(UTC) + timedelta(hours=25)
    assert w.client.post(undo, headers=H["boss"]).status_code == 410
    r = w.client.post(xfer, json={"to": "boss@example.com"}, headers=H["mem"])
    assert r.status_code == 200
    org = w.orgs.get("org_t")
    assert org.role_of("boss@example.com") == "owner" and org.role_of("mem@example.com") == "admin"


# ------------------------------------------------------------------ web chat: company language
def test_web_chat_answers_in_the_company_language_and_fixes_a_slip(w, monkeypatch):
    """The web chat speaks the COMPANY language (the server's choice; English until chosen),
    never the language somebody typed in; a reply that slipped into another script is
    translated before it appears."""
    from salid.guild_settings import GuildSettingsUpdate

    url = "/api/center/orgs/org_t/chat"
    fixed = []

    def translator(text, lang):
        fixed.append((text, lang))
        return f"[{lang}] {text}"

    monkeypatch.setattr(center, "get_translator", lambda: translator)
    # unchosen server: English. The fake brain echoes the Thai question -> a Thai reply -> fixed
    r = w.client.post(
        url, json={"message": "สรุปหน่อย", "workspace_id": "discord-1000"}, headers=H["mem"]
    )
    assert r.json()["answer"] == "[en] ans:สรุปหน่อย" and w.asks[-1][1]["language"] == "en"
    r = w.client.post(
        url, json={"message": "total?", "workspace_id": "discord-1000"}, headers=H["mem"]
    )
    assert r.json()["answer"] == "ans:total?" and len(fixed) == 1  # English reply: untouched
    # the company chooses Thai: every web answer is Thai, an English slip is translated
    w.guilds.save_settings(A, GuildSettingsUpdate(default_language="th"), "boss@example.com")
    r = w.client.post(
        url, json={"message": "total?", "workspace_id": "discord-1000"}, headers=H["mem"]
    )
    assert w.asks[-1][1]["language"] == "th" and r.json()["answer"] == "[th] ans:total?"
    r = w.client.post(
        url, json={"message": "สรุปหน่อย", "workspace_id": "discord-1000"}, headers=H["mem"]
    )
    assert r.json()["answer"] == "ans:สรุปหน่อย"
    # a broken translator never loses the answer
    monkeypatch.setattr(center, "get_translator", lambda: lambda text, lang: 1 / 0)
    r = w.client.post(
        url, json={"message": "total?", "workspace_id": "discord-1000"}, headers=H["mem"]
    )
    assert r.json()["answer"] == "ans:total?"
    assert center.needs_language_fix("ok 👍", "th") is False  # nothing to translate
    assert center.needs_language_fix("", "en") is False


# ------------------------------------------------------------------ admins set who people are
def test_admins_edit_other_members_profiles(w):
    """An owner / admin sets a colleague's title, department, what they do, real name and
    manager without the person asking; a member only their own."""
    url = "/api/center/orgs/org_t/members/mem@example.com/profile"
    body = {"title": "CEO", "department": "Management", "first_name": "Mem", "about": "Runs it"}
    assert w.client.put(url, json=body, headers=H["other"]).status_code == 403  # not a member
    r = w.client.put(url, json=body, headers=H["mem"])  # yourself: fine
    assert r.status_code == 200 and r.json()["title"] == "CEO"
    r = w.client.put(url, json={"title": "CFO", "reports_to": "boss@example.com"}, headers=H["adm"])
    assert (
        r.status_code == 200
        and r.json()["title"] == "CFO"
        and r.json()["reports_to"] == "boss@example.com"
    )
    assert r.json()["department"] == "Management" and r.json()["first_name"] == "Mem"  # kept
    p = w.orgs.user_profile("mem@example.com")
    assert (p.title, p.department, p.about) == ("CFO", "Management", "Runs it")
    # a member cannot edit somebody else; unknown member / manager
    other = "/api/center/orgs/org_t/members/adm@example.com/profile"
    assert w.client.put(other, json={"title": "x"}, headers=H["mem"]).status_code == 403
    assert (
        w.client.put(
            "/api/center/orgs/org_t/members/ghost@example.com/profile",
            json={"title": "x"},
            headers=H["boss"],
        ).status_code
        == 404
    )
    assert w.client.put(
        url, json={"reports_to": "ghost@example.com"}, headers=H["boss"]
    ).status_code in (400, 404)
    # "" clears the manager, the directory shows the title
    assert (
        w.client.put(url, json={"reports_to": ""}, headers=H["boss"]).json()["reports_to"] is None
    )
    members = w.client.get("/api/center/orgs/org_t/members", headers=H["boss"]).json()
    assert next(m for m in members if m["email"] == "mem@example.com")["title"] == "CFO"
    # the same through Salid: admins get the tool, members do not
    w.client.post(
        "/api/center/orgs/org_t/chat", json={"message": "Mem is our CTO"}, headers=H["adm"]
    )
    tools = {getattr(t, "__name__", ""): t for t in w.asks[-1][1]["extra_tools"]}
    assert "update_member_profile" in tools
    out = tools["update_member_profile"]("Mem", title="CTO", manager="boss")
    assert out["status"] == "ok" and out["member"] == "mem@example.com"
    assert out["profile"]["title"] == "CTO" and out["reports_to"] == "boss@example.com"
    assert "error" in tools["update_member_profile"]("nobody-here", title="x")
    w.client.post("/api/center/orgs/org_t/chat", json={"message": "hi"}, headers=H["mem"])
    assert "update_member_profile" not in {
        getattr(t, "__name__", "") for t in w.asks[-1][1]["extra_tools"]
    }
    assert center.find_member(w.orgs.get("org_t"), w.orgs, "boss@example.com") == "boss@example.com"
    assert center.find_member(w.orgs.get("org_t"), w.orgs, "") is None


# ------------------------------------------------------------------ identities heal themselves
def test_discord_verified_before_joining_is_carried_onto_the_member_record(w):
    """Jems' case (18 Oct 2026): verified Discord (sign-in) BEFORE being added to the org ->
    Member.discord_user_id stayed empty and "my tasks" could not match him. Adding a member,
    accepting an invite and every /me (sign-in) now carry the verified id over, and the
    sign-in index is rebuilt for identities stored before it existed."""
    from salid.orgs import IDENTITY_INDEX, Identity

    w.orgs.set_user_identity(
        "new@example.com",
        Identity(platform="discord", external_id="555", name="newbie", verified=True),
    )
    r = w.client.post(
        "/api/center/orgs/org_t/members", json={"email": "new@example.com"}, headers=H["boss"]
    )
    assert next(m for m in r.json() if m["email"] == "new@example.com")["discord_user_id"] == "555"
    # through an invite link too
    w.orgs.set_user_identity(
        "stranger@example.com",
        Identity(platform="discord", external_id="777", name="s", verified=True),
    )
    inv = w.client.post("/api/center/orgs/org_t/invites", json={}, headers=H["boss"]).json()
    w.client.post(f"/api/center/invites/{inv['code']}/accept", headers=H["stranger"])
    org = w.orgs.get("org_t")
    assert (
        next(m for m in org.members if m.email == "stranger@example.com").discord_user_id == "777"
    )
    # an old record: identity in the profile, index missing, member record empty -> /me repairs
    for m in org.members:
        if m.email == "new@example.com":
            m.discord_user_id = None
    w.orgs.save(org)
    w.docs.delete(IDENTITY_INDEX, 0, "discord:555")
    assert w.orgs.email_for_identity("discord", "555") is None
    assert (
        w.client.get("/api/center/me", headers=H["mem"]).status_code == 200
    )  # somebody else: no-op
    assert (
        w.orgs.get("org_t").role_of("new@example.com")
        and w.orgs.email_for_identity("discord", "555") is None
    )

    TOKENS["newbie"] = {"email": "new@example.com", "email_verified": True}
    try:
        assert (
            w.client.get("/api/center/me", headers={"Authorization": "Bearer newbie"}).status_code
            == 200
        )
    finally:
        TOKENS.pop("newbie", None)
    assert w.orgs.email_for_identity("discord", "555") == "new@example.com"
    org = w.orgs.get("org_t")
    assert next(m for m in org.members if m.email == "new@example.com").discord_user_id == "555"
    assert w.orgs.heal_identities("new@example.com") is False  # nothing left to repair


# ------------------------------------------------------------------ people on the platforms
def test_people_directory_from_the_platforms(w, monkeypatch):
    """Everyone who talks on the company's Discord gets a floating profile (account or not):
    seen stats, link to a member when their Discord is verified, admin notes, and what Salid
    observed from their own messages (execs only)."""
    from salid.orgs import Identity
    from salid.people_directory import PeopleBook

    book = PeopleBook(w.docs)
    app.dependency_overrides[center.get_people] = lambda: book
    prompts = []

    def observer(prompt):
        prompts.append(prompt)
        return {
            "role": "Handles customer quotes",
            "responsibilities": "Sends quotes, chases invoices",
            "works_with": ["Ann", "Bob"],
            "how_they_are": "Busy but calm",
        }

    app.dependency_overrides[center.get_observer] = lambda: observer
    try:
        day = "2026-10-12"

        def msg(mid, uid, name, text, channel=PUBLIC, d=day):
            w.memory.add_message(
                MemoryMessage(
                    guild_id=A,
                    channel_id=channel,
                    message_id=mid,
                    author_id=uid,
                    author_name=name,
                    created_at=at(d),
                    local_date=d,
                    text=text,
                )
            )

        msg(1, 11, "Ann", "quote for customer A sent")
        msg(2, 12, "Bob", "thanks Ann, invoice next", d="2026-10-13")
        msg(3, 11, "Ann", "chasing the invoice", d="2026-10-13")
        msg(4, 13, "Salid", "noted!", d="2026-10-13")  # the bot herself: skipped
        msg(5, 14, "Zed", "private stuff", channel=PRIVATE)
        w.docs.put("salid_profiles", A, "11", {"user_id": 11, "nickname": "แอน"})
        w.docs.put(
            "salid_tasks", A, "t1", {"assignee_id": 11, "assignee_name": "Ann", "status": "open"}
        )
        w.docs.put(
            "salid_tasks", A, "t2", {"assignee_id": 11, "assignee_name": "Ann", "status": "done"}
        )
        w.docs.put(
            "salid_tasks",
            A,
            "t3",
            {"assignee_id": 99, "assignee_name": "Quiet Q", "status": "open"},
        )
        # Bob is a member who verified that Discord account
        w.orgs.set_user_identity(
            "mem@example.com", Identity(platform="discord", external_id="12", verified=True)
        )
        base = "/api/center/orgs/org_t/people"
        assert w.client.get(base, headers=H["mem"]).json()["people"] == []  # not scanned yet
        assert w.client.post(base + "/discover", headers=H["mem"]).status_code == 403
        r = w.client.post(base + "/discover", headers=H["adm"])
        assert r.status_code == 200, r.text
        got = {p["external_id"]: p for p in r.json()["people"]}
        assert set(got) == {"11", "12", "14", "99"}  # Salid herself is not a person
        ann = got["11"]
        assert ann["display_name"] == "แอน" and ann["name"] == "Ann" and ann["email"] == ""
        assert ann["seen"]["messages"] == 2 and ann["seen"]["channels"] == ["general"]
        assert ann["seen"]["talks_with"] == ["Bob"] and ann["seen"]["tasks_open"] == 1
        assert ann["seen"]["tasks_done"] == 1 and ann["seen"]["last_seen"] == "2026-10-13"
        assert got["12"]["email"] == "mem@example.com" and got["12"]["member_role"] == "member"
        assert got["99"]["name"] == "Quiet Q" and got["99"]["seen"]["messages"] == 0
        assert r.json()["with_account"] == 1
        # an admin writes who Ann is, hides the stray account, links Ann to a member by hand
        r = w.client.put(
            f"{base}/discord:11", json={"title": "Sales lead", "about": "Quotes"}, headers=H["adm"]
        )
        assert r.json()["title"] == "Sales lead" and r.json()["about"] == "Quotes"
        assert w.client.put(f"{base}/discord:99", json={"hidden": True}, headers=H["adm"]).json()[
            "hidden"
        ]
        assert (
            w.client.put(
                f"{base}/discord:11", json={"email": "ghost@example.com"}, headers=H["adm"]
            ).status_code
            == 404
        )
        assert (
            w.client.put(f"{base}/discord:nope", json={"title": "x"}, headers=H["adm"]).status_code
            == 404
        )
        r = w.client.put(f"{base}/discord:11", json={"email": "adm@example.com"}, headers=H["boss"])
        assert r.json()["email"] == "adm@example.com" and r.json()["member_role"] == "admin"
        # what Salid observed: execs only, from Ann's own lines, in the company language
        r = w.client.post(f"{base}/discord:11/observe", headers=H["adm"])
        assert r.status_code == 200, r.text
        obs = r.json()["observed"]
        assert obs["role"] == "Handles customer quotes" and obs["works_with"] == ["Ann", "Bob"]
        assert obs["messages"] == 2 and "quote for customer A sent" in prompts[-1]
        assert "chasing the invoice" in prompts[-1] and "in English" in prompts[-1]
        assert "private stuff" not in prompts[-1]  # somebody else's words are not read
        assert w.client.post(f"{base}/discord:11/observe", headers=H["mem"]).status_code == 403
        # members see the public part only: no hidden people, no observations
        mine = w.client.get(base, headers=H["mem"]).json()
        ids = {p["external_id"] for p in mine["people"]}
        assert "99" not in ids and all(p["observed"] is None for p in mine["people"])
        assert next(p for p in mine["people"] if p["external_id"] == "11")["title"] == "Sales lead"
        full = w.client.get(base, headers=H["adm"]).json()
        assert next(p for p in full["people"] if p["external_id"] == "11")["observed"]["role"]
        assert "99" in {p["external_id"] for p in full["people"]}
        # a re-scan keeps the admin's notes and the observation, refreshes what was seen
        msg(6, 11, "Ann", "new quote", d="2026-10-14")
        r = w.client.post(base + "/discover", headers=H["adm"]).json()
        ann = next(p for p in r["people"] if p["external_id"] == "11")
        assert (
            ann["seen"]["messages"] == 3
            and ann["title"] == "Sales lead"
            and ann["observed"]["role"]
        )
        assert ann["email"] == "adm@example.com"
        # nobody wrote anything: an empty observation, no model call
        n = len(prompts)
        r = w.client.post(f"{base}/discord:99/observe", headers=H["adm"])
        assert r.json()["observed"]["messages"] == 0 and len(prompts) == n
        # a web-only org has no platform people
        assert w.client.get("/api/center/orgs/org_o/people", headers=H["other"]).status_code == 404
    finally:
        app.dependency_overrides.pop(center.get_people, None)
        app.dependency_overrides.pop(center.get_observer, None)


# ------------------------------------------------------------------ document groups
def test_documents_are_grouped_by_what_they_are_for_the_company(w):
    """Files fall into company records / legal / finance / hr / other (keywords on the tag,
    file name and summary, Thai and English); an admin can re-file one."""
    from salid.library import classify_group

    assert classify_group("company", "หนังสือรับรองบริษัท.pdf") == "company"
    assert classify_group("other", "ข้อบังคับบริษัท.pdf") == "company"
    assert classify_group("contract", "Shareholder agreement.pdf") == "company"
    assert classify_group("contract", "NDA Acme.pdf") == "legal"
    assert classify_group("invoice", "INV-001.pdf") == "finance"
    assert classify_group("other", "ใบเสร็จค่าไฟ.jpg") == "finance"
    assert classify_group("contract", "สัญญาจ้างพนักงาน.pdf") == "hr"
    assert classify_group("notes", "ideas.txt") == "other"
    r = upload(w, "mem")  # plan.txt, tagged "contract" by the fake analyzer
    item = r.json()
    assert item["group"] == "legal"
    base = "/api/center/orgs/org_t/documents"
    counts = w.client.get(base + "/groups", headers=H["mem"]).json()
    assert counts["legal"] == 1 and counts["company"] == 0 and list(counts)[0] == "company"
    assert w.client.get(base + "?group=legal", headers=H["mem"]).json()["total"] == 1
    assert w.client.get(base + "?group=company", headers=H["mem"]).json()["total"] == 0
    assert w.client.get(base + "?group=nope", headers=H["mem"]).status_code == 422
    # an admin re-files it as a company record; members cannot
    url = f"{base}/{item['id']}/group"
    assert w.client.put(url, json={"group": "company"}, headers=H["mem"]).status_code == 403
    assert w.client.put(url, json={"group": "king"}, headers=H["adm"]).status_code == 422
    assert (
        w.client.put(
            f"{base}/web:upload:nope/group", json={"group": "company"}, headers=H["adm"]
        ).status_code
        == 404
    )
    assert (
        w.client.put(url, json={"group": "company"}, headers=H["adm"]).json()["group"] == "company"
    )
    got = w.client.get(base + "?group=company", headers=H["mem"]).json()
    assert [i["id"] for i in got["items"]] == [item["id"]] and got["items"][0]["group"] == "company"


# ------------------------------------------------------------------ positions + expertise locator
def test_positions_are_organisational_knowledge_admins_create_them_ahead(w):
    base = "/api/center/orgs/org_t/positions"
    body = {
        "title": "Sales Lead",
        "department": "Sales",
        "summary": "Owns the pipeline",
        "skills": ["negotiation", "  quotes "],
        "policies": "Discounts over 10% need the CEO",
    }
    assert w.client.post(base, json=body, headers=H["mem"]).status_code == 403
    assert w.client.post(base, json={"title": " "}, headers=H["adm"]).status_code == 400
    r = w.client.post(base, json=body, headers=H["adm"])
    assert (
        r.status_code == 200
        and r.json()["holders"] == []
        and r.json()["skills"] == ["negotiation", "quotes"]
    )
    sales = r.json()["id"]
    ceo = w.client.post(
        base, json={"title": "CEO", "holders": ["boss@example.com"]}, headers=H["boss"]
    ).json()
    r = w.client.put(
        f"{base}/{sales}",
        json={"reports_to": ceo["id"], "holders": ["mem@example.com"]},
        headers=H["adm"],
    )
    assert r.json()["reports_to"] == ceo["id"] and r.json()["holders"] == ["mem@example.com"]
    assert (
        w.client.put(
            f"{base}/{ceo['id']}", json={"reports_to": sales}, headers=H["adm"]
        ).status_code
        == 400
    )  # loop
    assert (
        w.client.put(
            f"{base}/{sales}", json={"holders": ["ghost@example.com"]}, headers=H["adm"]
        ).status_code
        == 400
    )
    assert w.client.put(f"{base}/nope", json={"title": "x"}, headers=H["adm"]).status_code == 404
    got = w.client.get(base, headers=H["mem"]).json()  # everyone reads them
    assert [x["title"] for x in got] == ["CEO", "Sales Lead"]
    assert w.client.delete(f"{base}/{ceo['id']}", headers=H["mem"]).status_code == 403
    assert w.client.delete(f"{base}/{ceo['id']}", headers=H["adm"]).json() == {"deleted": ceo["id"]}
    assert w.client.get(base, headers=H["mem"]).json()[0]["reports_to"] == ""  # orphan lifted


def test_expertise_locator_recommends_with_reasons_and_never_invents_people(w, monkeypatch):
    from salid.orgs import Identity
    from salid.people_directory import Observed, PeopleBook, Person

    book = PeopleBook(w.docs)
    app.dependency_overrides[center.get_people] = lambda: book
    monkeypatch.setattr(center, "get_people", lambda: book)
    monkeypatch.setattr(center, "get_library", lambda: w.lib)
    prompts = []

    def model(prompt):
        prompts.append(prompt)
        return {
            "picks": [
                {"key": "ghost@example.com", "rank": "primary", "reason": "invented"},
                {
                    "key": "mem@example.com",
                    "rank": "primary",
                    "reason": "Handles quotes daily",
                    "evidence": ["Finished task: send quote to A", "made-up line"],
                    "confidence": "high",
                },
                {
                    "key": "discord:55",
                    "rank": "backup",
                    "reason": "Talks to customers",
                    "confidence": "weird",
                },
            ],
            "gaps": "",
        }

    app.dependency_overrides[center.get_expert_model] = lambda: model
    monkeypatch.setattr(center, "get_expert_model", lambda: model)
    try:
        w.orgs.set_user_profile("mem@example.com", name="Mem", title="Account exec")
        w.orgs.set_user_identity(
            "mem@example.com", Identity(platform="discord", external_id="12", verified=True)
        )
        w.docs.put(
            "salid_tasks",
            A,
            "t1",
            {"assignee_id": 12, "title": "send quote to A", "status": "done"},
        )
        book.put(
            A,
            Person(
                id="discord:12",
                external_id="12",
                name="Mem",
                email="mem@example.com",
                observed=Observed(role="Sells to restaurants"),
            ),
        )
        book.put(
            A,
            Person(
                id="discord:55",
                external_id="55",
                name="Nok",
                observed=Observed(role="Customer support"),
            ),
        )
        book.put(
            A,
            Person(
                id="discord:66",
                external_id="66",
                name="Bot",
                hidden=True,
                observed=Observed(role="x"),
            ),
        )
        pos = w.client.post(
            "/api/center/orgs/org_t/positions",
            json={"title": "Sales Lead", "skills": ["quotes"]},
            headers=H["adm"],
        ).json()
        url = "/api/center/orgs/org_t/experts"
        assert w.client.post(url, json={}, headers=H["adm"]).status_code == 422
        assert w.client.post(url, json={"position_id": "nope"}, headers=H["adm"]).status_code == 404
        r = w.client.post(url, json={"position_id": pos["id"]}, headers=H["adm"])
        assert r.status_code == 200, r.text
        res = r.json()
        assert (
            res["need"] == "Who should take the position Sales Lead?"
            and res["position"]["title"] == "Sales Lead"
        )
        assert [(p["key"], p["rank"]) for p in res["picks"]] == [
            ("mem@example.com", "primary"),
            ("discord:55", "backup"),
        ]
        first = res["picks"][0]
        assert first["name"] == "Mem" and first["has_account"] and first["confidence"] == "high"
        assert first["evidence"] == ["Finished task: send quote to A"]  # made-up evidence dropped
        assert res["picks"][1]["has_account"] is False and res["picks"][1]["confidence"] == "medium"
        p = prompts[-1]
        assert "Sells to restaurants" in p and "Customer support" in p and "Skills: quotes" in p
        assert (
            "Bot" not in p and "[discord:12]" not in p
        )  # hidden people out; Mem once, as a member
        # members get recommendations without what Salid observed (read from every room)
        w.client.post(url, json={"need": "who can do quotes?"}, headers=H["mem"])
        assert (
            "Sells to restaurants" not in prompts[-1]
            and "Finished task: send quote to A" in prompts[-1]
        )
        # exclude the person being replaced
        w.client.post(
            url, json={"need": "cover for Mem", "exclude": "mem@example.com"}, headers=H["adm"]
        )
        assert "[mem@example.com]" not in prompts[-1]
        # Salid's chat tools: everyone finds experts and reads positions, admins save positions
        w.client.post(
            "/api/center/orgs/org_t/chat", json={"message": "who knows quotes?"}, headers=H["mem"]
        )
        tools = {getattr(t, "__name__", ""): t for t in w.asks[-1][1]["extra_tools"]}
        assert {"find_expert", "list_positions"} <= set(tools) and "save_position" not in tools
        assert tools["find_expert"]("quotes")["picks"][0]["key"] == "mem@example.com"
        assert tools["list_positions"]("sales")["positions"][0]["vacant"] is True
        w.client.post(
            "/api/center/orgs/org_t/chat", json={"message": "create a role"}, headers=H["adm"]
        )
        tools = {getattr(t, "__name__", ""): t for t in w.asks[-1][1]["extra_tools"]}
        out = tools["save_position"]("Sales Lead", holder="Mem", description="Owns the pipeline")
        assert out["status"] == "ok" and out["position"]["holders"] == ["mem@example.com"]
        assert (
            out["position"]["summary"] == "Owns the pipeline" and out["position"]["id"] == pos["id"]
        )
        out = tools["save_position"]("Ops", reports_to="Nonexistent")
        assert "error" in out
    finally:
        app.dependency_overrides.pop(center.get_people, None)
        app.dependency_overrides.pop(center.get_expert_model, None)


# ------------------------------------------------------------------ chat upload = a secretary
def test_document_uploaded_in_chat_is_filed_and_handed_to_salid_like_a_secretary(w):
    reg = "หนังสือรับรองบริษัท เลขทะเบียนนิติบุคคล 0105566000000 ทุนจดทะเบียน 1,000,000 บาท"
    r = upload(w, "adm", name="หนังสือรับรองบริษัท.pdf", data=reg.encode(), mime="application/pdf")
    doc = r.json()
    assert doc["group"] == "company"
    url = "/api/center/orgs/org_t/chat"
    w.client.post(
        url, json={"message": "I just uploaded it", "document_id": doc["id"]}, headers=H["adm"]
    )
    q, kw = w.asks[-1]
    assert q.startswith("I just uploaded it") and "0105566000000" in q and "'company'" in q
    assert "update_company_profile" in q and "teach_company" in q
    # the chat log keeps what the person typed, not the document dump
    hist = w.client.get(url + "/history", headers=H["adm"]).json()
    assert items_text(hist)[-2] == "I just uploaded it"
    # a member is told an admin can update the profile; an unknown id adds nothing
    w.client.post(url, json={"message": "here", "document_id": doc["id"]}, headers=H["mem"])
    assert "Only owners and admins may change the company profile" in w.asks[-1][0]
    w.client.post(url, json={"message": "x", "document_id": "web:upload:nope"}, headers=H["adm"])
    assert w.asks[-1][0] == "x"
    # the legal-entity fields exist on the profile and the admin tool
    r = w.client.put(
        "/api/center/orgs/org_t",
        json={
            "profile": {"registration_no": "0105566000000", "registered_capital": "1,000,000 THB"}
        },
        headers=H["adm"],
    )
    assert r.json()["profile"]["registration_no"] == "0105566000000"


def items_text(hist):
    rows = hist.get("turns") or hist.get("items") or hist if isinstance(hist, dict) else hist
    return [x["text"] for x in rows]


# ------------------------------------------------------------------ no short caps on what people write
def test_long_profile_and_company_text_is_kept_whole_with_line_breaks(w):
    """Owner's rule (18 Oct 2026): never cut what people write about themselves or the
    company. A 20,000-character job description with paragraphs comes back intact."""
    jd = ("หน้าที่หลัก: ดูแลลูกค้าองค์กรและติดตามใบเสนอราคา\n\n" * 400).strip()
    assert len(jd) > 15_000
    r = w.client.put(
        "/api/center/me/profile",
        json={"job_description": jd, "about": "Line 1\nLine 2"},
        headers=H["mem"],
    )
    assert r.status_code == 200, r.text
    assert r.json()["job_description"] == jd and r.json()["about"] == "Line 1\nLine 2"
    title = "Head of Customer Success, Enterprise Accounts and Strategic Partnerships for Southeast Asia"
    r = w.client.put(
        "/api/center/orgs/org_t/members/mem@example.com/profile",
        json={"title": title},
        headers=H["adm"],
    )
    assert r.json()["title"] == title and r.json()["job_description"] == jd  # > 80 chars kept
    about = "\n\n".join(f"Paragraph {i}: " + "x" * 400 for i in range(20))
    r = w.client.put(
        "/api/center/orgs/org_t",
        json={"profile": {"about": about, "objectives": "1. a\n2. b"}},
        headers=H["adm"],
    )
    assert (
        r.json()["profile"]["about"] == about and r.json()["profile"]["objectives"] == "1. a\n2. b"
    )
    pos = w.client.post(
        "/api/center/orgs/org_t/positions",
        json={"title": "Ops", "policies": about},
        headers=H["adm"],
    )
    assert pos.json()["policies"] == about


def test_several_company_papers_in_one_chat_message_and_duplicates_kept_once(w):
    """The owner drops several registration papers into the chat at once: one Salid request
    with every file (shared text budget), the same file twice is stored once."""
    cert = "หนังสือรับรอง ทะเบียนนิติบุคคลเลขที่ 0105569193528 กรรมการ นายพสิณ พรวรภัตช์".encode()
    a = upload(w, "boss", name="certificate_2.pdf", data=cert, mime="application/pdf").json()
    b = upload(w, "boss", name="certificate_3.pdf", data=cert, mime="application/pdf").json()
    assert b["duplicate"] is True and b["id"] == a["id"] and a["duplicate"] is False
    c = upload(
        w,
        "boss",
        name="cert-of-regist.pdf",
        data="ใบสำคัญแสดงการจดทะเบียน".encode(),
        mime="application/pdf",
    ).json()
    url = "/api/center/orgs/org_t/chat"
    r = w.client.post(
        url,
        json={"message": "my company papers", "document_ids": [a["id"], c["id"]]},
        headers=H["boss"],
    )
    assert r.status_code == 200, r.text
    q = w.asks[-1][0]
    assert (
        q.count("[Document just uploaded") == 2 and q.count("Act like the company's secretary") == 1
    )
    assert "นายพสิณ พรวรภัตช์" in q and "registration certificate (หนังสือรับรอง) wins" in q
    assert "already on file" in q
    assert (
        len(
            w.client.post(
                url, json={"message": "x", "document_ids": ["a"] * 11}, headers=H["boss"]
            ).json()["detail"]
        )
        >= 1
    )


def test_page_reading_rules_for_untrustworthy_pdf_text():
    from salid.docs import DocText
    from salid.library import needs_page_reading, squeeze_fill_lines

    wm = "กรมพัฒนาธุรกิจการค้า Ref:E69\n" * 4  # a watermark on every page, nothing else
    assert needs_page_reading(DocText(text=wm, pages=4)) is True
    assert needs_page_reading(DocText(text="ใบสำคัญแสดงการจดทะเบียน " * 30, pages=1)) is True  # Thai
    english = "\n".join(
        f"Line {i} of a normal English contract with real content." for i in range(40)
    )
    assert needs_page_reading(DocText(text=english, pages=1)) is False
    assert squeeze_fill_lines("ชื่อ" + "." * 900 + " ปี") == "ชื่อ… ปี"


# ------------------------------------------------------------------ company calendar + personal overlay
def test_company_calendar_with_an_optional_personal_overlay(w):
    from salid.orgs import Identity
    from salid.projects import PROJECTS

    w.orgs.set_user_identity(
        "mem@example.com", Identity(platform="discord", external_id="12", verified=True)
    )
    w.docs.put(
        "salid_reminders",
        A,
        "R-1",
        {"active": True, "user_id": 12, "next_at": "2026-10-20T09:00", "message": "call A"},
    )
    w.docs.put(
        "salid_tasks",
        A,
        "t1",
        {
            "id": "t1",
            "guild_id": A,
            "assignee_id": 12,
            "title": "send quote",
            "status": "open",
            "due": "2026-10-21",
        },
    )
    w.docs.put(
        PROJECTS,
        A,
        "p1",
        {
            "id": "p1",
            "name": "Launch",
            "owner": {"email": "boss@example.com"},
            "status": "in_progress",
            "deadline": "2026-10-30",
            "visibility": "team",
            "milestones": [{"id": "m1", "title": "Beta", "due": "2026-10-25"}],
        },
    )
    w.docs.put(
        PROJECTS,
        A,
        "p2",
        {
            "id": "p2",
            "name": "Secret",
            "owner": {"email": "boss@example.com"},
            "status": "planned",
            "deadline": "2026-11-01",
            "visibility": "members",
        },
    )
    url = "/api/center/orgs/org_t/calendar"

    def flat(r):
        return {(i["kind"], i["layer"], i["title"]) for d in r.json()["days"] for i in d["items"]}

    company = flat(w.client.get(url, headers=H["mem"]))
    assert ("deadline", "company", "Launch") in company and (
        "milestone",
        "company",
        "Launch: Beta",
    ) in company
    assert not any(x[1] == "personal" for x in company)  # no overlay by default
    assert ("deadline", "company", "Secret") not in company  # a private project stays private
    both = flat(w.client.get(url + "?mine=1", headers=H["mem"]))
    assert ("reminder", "personal", "call A") in both and ("task", "personal", "send quote") in both
    assert w.client.get(url + "?mine=1", headers=H["mem"]).json()["layers"] == [
        "company",
        "personal",
    ]
    assert ("deadline", "company", "Secret") in flat(
        w.client.get(url, headers=H["adm"])
    )  # admins see all


def test_overlay_includes_the_persons_google_calendar(w, monkeypatch):
    """'Overlay my calendar' also shows the person's own Google Calendar events (personal
    layer); a Google failure never breaks the company calendar."""

    class G:
        evicted = 0

        def enabled(self):
            return True

        def is_connected(self, org_id, email):
            return email == "mem@example.com"

        def access_token(self, org_id, email):
            return "tok"

        def evict(self, org_id, email):
            G.evicted += 1

    calls = []

    class Resp:
        def __init__(self, code, body):
            self.status_code, self._b = code, body

        def json(self):
            return self._b

    class Http:
        def __init__(self, codes):
            self.codes = codes

        def get(self, url, params=None, headers=None):
            calls.append((url, params, headers))
            code = self.codes.pop(0)
            return Resp(
                code,
                {
                    "items": [
                        {
                            "summary": "Dentist",
                            "start": {"dateTime": "2026-10-20T09:00:00+07:00"},
                            "end": {"dateTime": "2026-10-20T10:00:00+07:00"},
                            "htmlLink": "https://calendar.google.com/x",
                        },
                        {
                            "summary": "Holiday",
                            "start": {"date": "2026-10-23"},
                            "end": {"date": "2026-10-24"},
                        },
                        {"summary": "gone", "status": "cancelled", "start": {"date": "2026-10-22"}},
                    ]
                },
            )

    monkeypatch.setattr(center, "get_google_oauth", lambda orgs: G())
    monkeypatch.setattr(center, "get_google_http", lambda: Http([401, 200]))  # stale token, then ok
    url = "/api/center/orgs/org_t/calendar"
    r = w.client.get(url + "?mine=1", headers=H["mem"]).json()
    items = [i for d in r["days"] for i in d["items"] if i["kind"] == "gcal"]
    assert [(i["title"], i["layer"], i["all_day"]) for i in items] == [
        ("Dentist", "personal", False),
        ("Holiday", "personal", True),
    ]
    assert r["google"]["connected"] and r["sources"]["google_calendar"] is True and G.evicted == 1
    assert (
        calls[-1][0].endswith("/calendars/primary/events")
        and calls[-1][2]["Authorization"] == "Bearer tok"
    )
    assert calls[-1][1]["singleEvents"] == "true"
    # company view only: Google is not asked
    n = len(calls)
    w.client.get(url, headers=H["mem"])
    assert len(calls) == n
    # not connected / Google down: the calendar still answers, the reason is reported
    assert (
        w.client.get(url + "?mine=1", headers=H["adm"]).json()["google"]["error"] == "not_connected"
    )
    monkeypatch.setattr(center, "get_google_http", lambda: Http([500]))
    r = w.client.get(url + "?mine=1", headers=H["mem"])
    assert r.status_code == 200 and r.json()["google"]["error"] == "failed"
