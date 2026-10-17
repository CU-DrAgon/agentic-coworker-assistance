"""Command center API: every endpoint with fake auth, SQLite / fake-Firestore doc stores, a fake
analyzer and a scripted `ask`. No network."""

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
    # admin adds a plain member, cannot hand out admin
    ok = w.client.post(base, json={"email": "New@Example.com"}, headers=H["adm"])
    assert ok.status_code == 200 and "new@example.com" in {m["email"] for m in ok.json()}
    deny = w.client.post(base, json={"email": "x@example.com", "role": "admin"}, headers=H["adm"])
    assert deny.status_code == 403
    assert (
        w.client.post(
            base, json={"email": "x@example.com", "role": "king"}, headers=H["boss"]
        ).status_code
        == 422
    )
    # owner promotes; the new member sees the org through the email index
    w.client.post(base, json={"email": "new@example.com", "role": "admin"}, headers=H["boss"])
    assert w.orgs.get("org_t").role_of("new@example.com") == "admin"
    assert [o.id for o in w.orgs.orgs_of_email("new@example.com")] == ["org_t"]
    # admin cannot remove an admin or owner, can remove a member
    assert w.client.delete(f"{base}/boss@example.com", headers=H["adm"]).status_code == 403
    assert w.client.delete(f"{base}?email=mem@example.com", headers=H["adm"]).status_code == 200
    assert w.orgs.orgs_of_email("mem@example.com") == []
    # the last owner stays
    assert w.client.delete(f"{base}/boss@example.com", headers=H["boss"]).status_code == 409
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
    upload(w, "mem", name="two.txt")  # new uploads follow the link
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
def test_overview(w):
    seed_content(w)
    ov = w.client.get("/api/center/orgs/org_t/overview", headers=H["adm"]).json()
    assert ov["members"] == 3 and ov["workspaces"]["total"] == 3
    assert ov["workspaces"]["by_platform"] == {"web": 1, "discord": 2}
    assert (
        ov["documents"]["by_category"]["receipts"] == 1
        and ov["documents"]["by_category"]["wiki"] == 2
    )
    assert ov["license"]["plan"] == "monthly" and ov["license_days_left"] > 100
    assert ov["recent_activity"] and {"title", "at", "category", "workspace_name"} <= set(
        ov["recent_activity"][0]
    )
    mem = w.client.get("/api/center/orgs/org_t/overview", headers=H["mem"]).json()
    assert (
        mem["documents"]["by_category"]["receipts"] == 0
        and mem["documents"]["by_category"]["wiki"] == 1
    )
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
    assert kw["guild_id"] == A and kw["is_exec"] is False and kw["user_id"] == 0
    assert set(kw["memory"].allowed) == {PUBLIC}  # a plain member: public channels only
    assert kw["language"] == "th"
    w.client.post(url, json={"message": "total?", "workspace_id": "discord-1000"}, headers=H["adm"])
    q, kw = w.asks[-1]
    assert kw["is_exec"] is True and set(kw["memory"].allowed) == {
        PUBLIC,
        PRIVATE,
    }  # never DM scope
    assert kw["language"] == "en"
    assert "สรุปหน่อย" not in kw["history"]  # history is per user
    # the default workspace is the web one: uploads only, no ledger
    w.client.post(url, json={"message": "hi"}, headers=H["adm"])
    q, kw = w.asks[-1]
    assert kw["guild_id"] == w.orgs.get("org_t").web_workspace().memory_id
    assert kw["is_exec"] is False and set(kw["memory"].allowed) == {WEB_DOCS_CHANNEL}


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
