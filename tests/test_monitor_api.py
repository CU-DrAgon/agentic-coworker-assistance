"""Workflow monitor API (`/api/center/orgs/{org}/monitor/*`, `/api/center/admin/monitor/*`):
fake Google tokens, a SQLite / fake-Firestore doc store, a fake `video_ad` source and a fixed
clock. No network."""

import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent / "salid"))

from fake_firestore import FakeFirestore  # noqa: E402

from meeting_scribe import center, hub, monitor_api  # noqa: E402
from meeting_scribe.app import app  # noqa: E402
from meeting_scribe.config import get_settings  # noqa: E402
from salid import monitor, monitor_sources  # noqa: E402
from salid.coo_store import FirestoreDocStore, SQLiteDocStore  # noqa: E402
from salid.monitor import Audience, MonitorItem, Step  # noqa: E402
from salid.orgs import License, Member, OrgStore, Workspace  # noqa: E402

CLIENT_ID = "123-abc.apps.googleusercontent.com"
GUILD = 1000
WS = f"discord-{GUILD}"
ORG = "org_t"
NOW = datetime(2026, 10, 14, 5, 0, tzinfo=UTC)  # 12:00 in Bangkok
BKK = ZoneInfo("Asia/Bangkok")
NAMES = ("boss", "adm", "mem", "root", "stranger", "solo")
TOKENS = {n: {"email": f"{n}@example.com", "email_verified": True} for n in NAMES}
H = {n: {"Authorization": f"Bearer {n}"} for n in NAMES}
BASE = f"/api/center/orgs/{ORG}/monitor"
OPS = "/api/center/admin/monitor"
SENTINEL = "SENSITIVE-7f3a"
ROUTES = ("/summary", "/items", "/items/video_ad/j1", "/changes?since=0")


def strings(obj):
    """Every string value and key in `obj`, recursively."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from strings(k)
            yield from strings(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from strings(v)


def item(
    item_id,
    *,
    kind="video_ad",
    status="running",
    people=(),
    admin_only=False,
    dm=False,
    requester="",
    at=NOW - timedelta(minutes=1),
    ws=WS,
    steps=(),
):
    return MonitorItem(
        kind=kind,
        id=item_id,
        org_id=ORG,
        workspace=ws,
        title="Video ad",
        status=status,
        raw_status=status,
        requester_id=requester,
        started_at=at.isoformat(),
        updated_at=at.isoformat(),
        steps=list(steps),
        audience=Audience(admin_only=admin_only, dm=dm, people=list(people)),
    )


def fake_source(items):
    """A source over `items` that, like the real ones, reads only the org's Discord workspaces."""

    def source(docs, org, now):
        wss = {ws.id for ws in monitor_sources.discord_workspaces(org)}
        return [it for it in items if it.workspace in wss]

    def finder(docs, org, item_id):
        return next((it for it in source(docs, org, None) if it.id == item_id), None)

    return source, finder


@pytest.fixture(params=["sqlite", "firestore"])
def docs(request):
    if request.param == "sqlite":
        return SQLiteDocStore(":memory:")
    return FirestoreDocStore(client=FakeFirestore())


class World:
    pass


@pytest.fixture
def w(docs, monkeypatch):
    monkeypatch.setenv("HUB_GOOGLE_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("HUB_ADMIN_EMAILS", "root@example.com")
    monkeypatch.setenv("MONITOR_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(monitor_api, "now", lambda: NOW)
    monkeypatch.setattr(monitor_api, "LIMITER", monitor_api.TokenBucket())
    runtime = (SimpleNamespace(deep_task_timeout=600), None)
    monkeypatch.setattr(hub, "get_salid_runtime", lambda: runtime)
    w = World()
    w.docs = docs
    w.orgs = OrgStore(docs)
    w.orgs.create(
        "Acme",
        "boss@example.com",
        org_id=ORG,
        license=License(),
        members=[
            Member(email="boss@example.com", role="owner"),
            Member(email="adm@example.com", role="admin"),
            Member(email="mem@example.com", discord_user_id="7"),
        ],
    )
    w.orgs.attach_workspace(ORG, Workspace(id=WS, platform="discord", external_id=str(GUILD)))
    w.items = []
    monkeypatch.setitem(monitor_sources.SOURCES, "video_ad", fake_source(w.items))
    monitor.write_heartbeat(docs, "bot", queues={"actions": 2}, now=NOW)

    def verifier():
        def verify(token, client_id):
            if token not in TOKENS:
                raise ValueError("bad token")
            return TOKENS[token]

        return verify

    app.dependency_overrides[hub.google_token_verifier] = verifier
    app.dependency_overrides[center.get_orgs] = lambda: w.orgs
    app.dependency_overrides[hub.get_prefs] = lambda: w.orgs
    w.c = TestClient(app)
    yield w
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def get(w, who, path, base=BASE, headers=None, **kw):
    return w.c.get(base + path, headers={**H[who], **(headers or {})}, **kw)


def ids_of(r):
    assert r.status_code == 200, r.text
    return [i["id"] for i in r.json()["items"]]


# ------------------------------------------------------------------ flag, roles, licence
def test_flag_off_is_404(w, monkeypatch):
    w.items.append(item("j1"))
    monkeypatch.setenv("MONITOR_ENABLED", "false")
    get_settings.cache_clear()
    for path in ROUTES:
        assert get(w, "boss", path).status_code == 404, path
        assert get(w, "root", path).status_code == 404, path
        assert w.c.get(BASE + path).status_code == 404, path  # hidden even before sign-in
    for path in ("/summary", "/items"):
        assert get(w, "root", path, base=OPS).status_code == 404, path
        assert w.c.get(OPS + path).status_code == 404, path


def test_roles_see_their_items(w):
    w.items += [
        item("a_admin", admin_only=True, people=["discord:7"]),  # admin_only beats people
        item("a_mem", people=["discord:7"], requester="discord:7"),
        item("a_other", people=["discord:8"]),
        item("a_dm", dm=True, people=["discord:7"]),  # a private conversation: never the owner
    ]

    def keys(who):
        return set(ids_of(get(w, who, "/items")))

    assert keys("boss") == keys("adm") == keys("root") == {"a_admin", "a_mem", "a_other"}
    assert keys("mem") == {"a_mem", "a_dm"}
    rows = {r["id"]: r for r in get(w, "mem", "/items").json()["items"]}
    assert rows["a_mem"]["mine"] is True and rows["a_dm"]["mine"] is False
    assert get(w, "boss", "/items").json()["items"][0]["mine"] is False
    assert get(w, "stranger", "/items").status_code == 403
    assert w.c.get(BASE + "/items").status_code == 401
    assert get(w, "mem", "/items", base="/api/center/orgs/nope/monitor").status_code == 403
    assert get(w, "root", "/items", base="/api/center/orgs/nope/monitor").status_code == 404
    assert get(w, "boss", "/items/video_ad/a_mem").json()["controls"] == ["cancel"]
    assert get(w, "mem", "/items/video_ad/a_mem").json()["controls"] == ["cancel"]  # requester
    assert get(w, "root", "/items/video_ad/a_mem").json()["controls"] == []  # read-only
    assert get(w, "mem", "/summary").json()["counts"] == {"running": 2}
    assert get(w, "boss", "/summary").json()["counts"] == {"running": 3}


def test_unlicensed_org_is_402_for_everyone_but_ops_view_works(w):
    w.items.append(item("j1"))
    org = w.orgs.get(ORG)
    org.license = License(status="expired")
    w.orgs.save(org)
    for who in ("boss", "adm", "mem", "root"):
        for path in ROUTES:
            assert get(w, who, path).status_code == 402, (who, path)
    assert get(w, "stranger", "/summary").status_code == 403  # membership is checked first
    s = get(w, "root", "/summary", base=OPS)
    assert s.status_code == 200, s.text
    assert [(o["id"], o["counts"]) for o in s.json()["orgs"]] == [(ORG, {"running": 1})]
    assert ids_of(get(w, "root", "/items", base=OPS)) == ["j1"]


# ------------------------------------------------------------------ items
def test_paging_and_ids_batch(w):
    w.items += [
        item(f"j{n:03d}", people=["discord:7"], at=NOW - timedelta(seconds=n)) for n in range(120)
    ]
    p1 = get(w, "boss", "/items").json()
    assert len(p1["items"]) == 50 and p1["next_cursor"] == "50"
    assert [r["id"] for r in p1["items"][:2]] == ["j000", "j001"]  # newest first
    p3 = get(w, "boss", "/items", params={"cursor": "100"}).json()
    assert len(p3["items"]) == 20 and p3["next_cursor"] is None
    assert len(get(w, "boss", "/items", params={"limit": 100}).json()["items"]) == 100
    assert get(w, "boss", "/items", params={"limit": 101}).status_code == 422
    assert get(w, "boss", "/items", params={"limit": 0}).status_code == 422
    too_many = ",".join(f"video_ad:j{n:03d}" for n in range(51))
    assert get(w, "boss", "/items", params={"ids": too_many}).status_code == 422
    fifty = ",".join(f"video_ad:j{n:03d}" for n in range(50))
    assert len(ids_of(get(w, "boss", "/items", params={"ids": fifty}))) == 50
    w.items.append(item("secret", admin_only=True))
    batch = "video_ad:j005,video_ad:nope,video_ad:secret,research:j001,junk,video_ad:j003"
    assert ids_of(get(w, "mem", "/items", params={"ids": batch})) == ["j003", "j005"]
    assert ids_of(get(w, "boss", "/items", params={"ids": "video_ad:secret"})) == ["secret"]
    assert ids_of(get(w, "boss", "/items", params={"ids": ""})) == []


def test_filters_and_bangkok_days(w):
    midnight = datetime(2026, 10, 14, tzinfo=BKK)  # 2026-10-13 17:00 UTC
    w.items += [
        item("early", status="done", at=midnight - timedelta(minutes=1)),  # 13 Oct in Bangkok
        item("late", status="failed", requester="discord:7", at=midnight + timedelta(minutes=1)),
    ]

    def ids(**params):
        return ids_of(get(w, "boss", "/items", params=params))

    assert ids() == ["late", "early"]
    assert ids(**{"from": "2026-10-14"}) == ["late"]
    assert ids(to="2026-10-13") == ["early"]
    assert ids(**{"from": "2026-10-13", "to": "2026-10-13"}) == ["early"]
    assert ids(status="failed") == ["late"] and ids(status=["done", "failed"]) == ["late", "early"]
    assert ids(status="done,failed") == ["late", "early"]  # comma lists work too
    assert ids(attention="true") == ["late"] and ids(attention="false") == ["early"]
    assert ids(owner="DISCORD:7") == ["late"] and ids(workspace="discord") == ["late", "early"]
    assert ids(workspace=WS) == ["late", "early"] and ids(workspace="slack") == []
    assert ids(kind="change") == [] and ids(kind=["change", "video_ad"]) == ["late", "early"]
    assert ids(q="LATE") == ["late"] and ids(priority="high") == ["late", "early"]  # no effect yet
    assert get(w, "boss", "/items", params={"from": "soon"}).status_code == 422


def test_detail_404_steps_events_controls(w):
    j1 = item(
        "j1",
        people=["discord:7"],
        requester="discord:7",
        steps=[
            Step(name="storyboard", state="done", ms=1200),
            Step(name="stills", state="running"),
        ],
    )
    hidden = item("hidden", admin_only=True)
    w.items += [j1, hidden]
    monitor.emit(w.docs, ORG, monitor.event_for(j1), now=NOW)
    monitor.emit(w.docs, ORG, monitor.event_for(hidden), now=NOW)
    monitor.emit(w.docs, ORG, monitor.event_for(j1.model_copy(update={"status": "failed"})))
    d = get(w, "boss", "/items/video_ad/j1").json()
    assert [(s["name"], s["state"], s["ms"]) for s in d["steps"]] == [
        ("storyboard", "done", 1200),
        ("stills", "running", None),
    ]
    assert [(e["seq"], e["status"], e["severity"]) for e in d["events"]] == [
        (1, "running", "info"),
        (3, "failed", "error"),
    ]
    assert all("audience" not in e for e in d["events"]) and "audience" not in d
    assert d["controls"] == ["cancel"] and d["mine"] is False and d["title"] == "Video ad"
    assert get(w, "mem", "/items/video_ad/j1").json()["mine"] is True
    assert get(w, "root", "/items/video_ad/j1").json()["controls"] == []
    assert get(w, "boss", "/items/video_ad/nope").status_code == 404
    assert get(w, "boss", "/items/zzz/j1").status_code == 404
    assert get(w, "mem", "/items/video_ad/hidden").status_code == 404  # invisible = unknown
    assert get(w, "adm", "/items/video_ad/hidden").status_code == 200


def test_attention_uses_the_bot_heartbeat_and_deep_timeout(w, monkeypatch):
    w.items += [
        item("fresh"),
        item("slow", at=NOW - timedelta(seconds=601)),  # a video step past 600 s
        item("broken", status="failed"),
    ]
    rows = {r["id"]: r for r in get(w, "boss", "/items").json()["items"]}
    assert {k: r["attention_reason"] for k, r in rows.items()} == {
        "fresh": "",
        "slow": "stuck",
        "broken": "failed",
    }
    s = get(w, "boss", "/summary").json()
    assert s["attention"] == 2 and s["failed_24h"] == 1 and s["done_today"] == 0
    assert s["heartbeat"] == {"bot": {"state": "online", "tick_at": NOW.isoformat()}}
    assert s["now"] == NOW.isoformat() and s["seq"] == 0 and s["by_kind"] == {"video_ad": 3}
    monitor.write_heartbeat(w.docs, "bot", queues={}, now=NOW - timedelta(minutes=6))
    rows = {r["id"]: r for r in get(w, "boss", "/items").json()["items"]}
    assert rows["fresh"]["attention_reason"] == "bot_offline"
    assert get(w, "boss", "/summary").json()["heartbeat"]["bot"]["state"] == "offline"


def test_deep_timeout_reads_the_salid_settings_and_falls_back(monkeypatch):
    monkeypatch.setattr(
        hub, "get_salid_runtime", lambda: (SimpleNamespace(deep_task_timeout=900), 0)
    )
    assert monitor_api.get_deep_timeout() == 900

    def broken():
        raise RuntimeError("no store")

    monkeypatch.setattr(hub, "get_salid_runtime", broken)
    assert monitor_api.get_deep_timeout() == 600


def test_a_failing_source_is_skipped_and_logged_without_its_text(w, monkeypatch, caplog):
    w.items.append(item("j1"))

    def broken(docs, org, now):
        raise RuntimeError(SENTINEL)

    def broken_finder(docs, org, item_id):
        raise RuntimeError(SENTINEL)

    monkeypatch.setitem(monitor_sources.SOURCES, "research", (broken, broken_finder))
    with caplog.at_level(logging.WARNING, logger="salid.monitor_sources"):
        assert ids_of(get(w, "boss", "/items")) == ["j1"]
        assert get(w, "boss", "/items/research/r1").status_code == 404
    assert "RuntimeError" in caplog.text and SENTINEL not in caplog.text


def test_collect_keeps_active_items_and_the_newest_history(monkeypatch):
    org = SimpleNamespace(id=ORG, workspaces=[])
    done = [item(f"d{n:03d}", status="done", at=NOW - timedelta(seconds=n)) for n in range(4)]
    active = [item("old_run", at=NOW - timedelta(days=30))]
    monkeypatch.setattr(monitor_sources, "HISTORY_MAX", 2)
    monkeypatch.setattr(monitor_sources, "SOURCES", {})
    monitor_sources.register("video_ad", lambda d, o, n: done + active, lambda d, o, i: None)
    got = monitor_sources.collect(None, org, NOW)
    assert sorted(it.id for it in got) == ["d000", "d001", "old_run"]
    assert monitor_sources.collect(None, org, NOW, kinds=["change"]) == []
    assert monitor_sources.find(None, org, "nope", "x") is None


def test_monitor_sources_never_imports_discord():
    text = Path(monitor_sources.__file__).read_text(encoding="utf-8")
    assert "import discord" not in text and "from discord" not in text


def test_discord_workspaces_are_connected_discord_only():
    org = SimpleNamespace(
        workspaces=[
            Workspace(id="web", platform="web", external_id="o"),
            Workspace(id=WS, platform="discord", external_id=str(GUILD)),
            Workspace(id="discord-2", platform="discord", external_id="2", status="disconnected"),
            Workspace(id="slack-T1", platform="slack", external_id="T1"),
        ]
    )
    assert [ws.id for ws in monitor_sources.discord_workspaces(org)] == [WS]


# ------------------------------------------------------------------ change feed
def test_changes_etag_304_and_rate_limit(w, monkeypatch):
    clock = SimpleNamespace(t=100.0)
    monkeypatch.setattr(monitor_api, "LIMITER", monitor_api.TokenBucket(clock=lambda: clock.t))

    def poll(who, since=0, **kw):
        clock.t += 2.0  # one token every 2 s: never limited
        return get(w, who, "/changes", params={"since": since}, **kw)

    monitor.emit(w.docs, ORG, monitor.event_for(item("j1", people=["discord:7"])))
    r = poll("boss")
    assert r.status_code == 200 and r.headers["etag"] == '"1"'
    assert r.headers["cache-control"] == "no-store"
    body = r.json()
    assert set(body) == {"seq", "floor", "now", "events", "reset"}
    assert body["seq"] == 1 and body["floor"] == 0 and body["reset"] is False
    assert body["now"] == NOW.isoformat() and [e["id"] for e in body["events"]] == ["j1"]
    assert "audience" not in body["events"][0]
    etag = r.headers["etag"]
    nm = poll("boss", headers={"If-None-Match": etag})
    assert nm.status_code == 304 and nm.headers["etag"] == etag and nm.content == b""
    assert poll("boss", 1, headers={"If-None-Match": f'"0", W/{etag}'}).status_code == 304
    assert poll("boss", 1, headers={"If-None-Match": '"0"'}).status_code == 200
    assert poll("mem", 5).json()["reset"] is True  # ahead of the log
    assert poll("mem", -1).status_code == 422

    for _ in range(3):  # burst 3, all inside 2 s
        assert get(w, "adm", "/changes?since=0").status_code == 200
        clock.t += 0.5
    r = get(w, "adm", "/changes?since=0")
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert get(w, "mem", "/changes?since=0").status_code == 200  # another person is unaffected
    clock.t += 2.0
    assert get(w, "adm", "/changes?since=0").status_code == 200  # a token came back


def test_token_bucket_refills_one_token_per_rate():
    clock = SimpleNamespace(t=0.0)
    bucket = monitor_api.TokenBucket(rate_s=2.0, burst=3, clock=lambda: clock.t)
    assert [bucket.take("a") for _ in range(3)] == [0.0, 0.0, 0.0]
    assert bucket.take("a") == pytest.approx(2.0)
    clock.t = 1.0
    assert bucket.take("a") == pytest.approx(1.0)
    clock.t = 2.0
    assert bucket.take("a") == 0.0 and bucket.take("b") == 0.0


def test_demoted_admin_loses_admin_items_and_events_next_poll(w):
    queued = item("act1", admin_only=True, people=["adm@example.com"], requester="adm@example.com")
    w.items.append(queued)
    monitor.emit(w.docs, ORG, monitor.event_for(queued))
    batch = {"ids": "video_ad:act1"}
    assert ids_of(get(w, "adm", "/items", params=batch)) == ["act1"]
    assert [e["id"] for e in get(w, "adm", "/changes?since=0").json()["events"]] == ["act1"]
    r = w.c.post(
        f"/api/center/orgs/{ORG}/members",
        headers=H["boss"],
        json={"email": "adm@example.com", "role": "member"},
    )
    assert r.status_code == 200, r.text
    assert ids_of(get(w, "adm", "/items", params=batch)) == []
    assert get(w, "adm", "/changes?since=0").json()["events"] == []
    assert get(w, "adm", "/items/video_ad/act1").status_code == 404
    r = w.c.delete(f"/api/center/orgs/{ORG}/members/adm@example.com", headers=H["boss"])
    assert r.status_code == 200, r.text
    assert get(w, "adm", "/items", params=batch).status_code == 403
    for path in ROUTES:
        assert get(w, "adm", path).status_code == 403, path


def test_web_only_org_is_empty_and_never_bot_offline(w, monkeypatch):
    w.orgs.create("Solo", "solo@example.com", org_id="org_web", license=License())
    base = "/api/center/orgs/org_web/monitor"
    monitor.write_heartbeat(w.docs, "bot", queues={}, now=NOW - timedelta(hours=1))  # offline
    w.items.append(item("j1"))  # in org_t's Discord workspace
    s = get(w, "solo", "/summary", base=base)
    assert s.status_code == 200, s.text
    body = s.json()
    assert body["counts"] == {} and body["by_kind"] == {} and body["attention"] == 0
    assert body["heartbeat"] == {} and body["seq"] == 0
    assert get(w, "solo", "/items", base=base).json() == {"items": [], "next_cursor": None}
    web = item("r1", kind="research", ws="web")
    monkeypatch.setitem(
        monitor_sources.SOURCES,
        "research",
        (
            lambda d, o, n: [web] if o.id == "org_web" else [],
            lambda d, o, i: web if (o.id, i) == ("org_web", "r1") else None,
        ),
    )
    rows = get(w, "solo", "/items", base=base).json()["items"]
    assert [(r["id"], r["attention"], r["attention_reason"]) for r in rows] == [("r1", False, "")]
    s = get(w, "boss", "/summary").json()  # the Discord org does see the bot offline
    assert s["heartbeat"]["bot"]["state"] == "offline" and s["attention"] == 1


def test_platform_heartbeat_only_for_orgs_with_a_bridge(w):
    w.orgs.attach_workspace(ORG, Workspace(id="slack-T1", platform="slack", external_id="T1"))
    monitor.write_heartbeat(w.docs, "platforms", queues={}, now=NOW - timedelta(seconds=120))
    hb = get(w, "boss", "/summary").json()["heartbeat"]
    assert hb == {
        "bot": {"state": "online", "tick_at": NOW.isoformat()},
        "platforms": {"state": "delayed", "tick_at": (NOW - timedelta(seconds=120)).isoformat()},
    }


# ------------------------------------------------------------------ operations view
def test_ops_view_lists_orgs_and_unlinked_logs(w):
    w.items += [item("j1"), item("j2", status="failed", at=NOW - timedelta(minutes=5))]
    w.orgs.create("Solo", "solo@example.com", org_id="org_web", license=License())
    monitor.emit(w.docs, ORG, monitor.event_for(w.items[0]), now=NOW)
    event = {"kind": "action", "id": "a1", "status": "queued", "attention": False}
    monitor.emit(w.docs, "guild:999", event, now=NOW - timedelta(minutes=2))
    s = get(w, "root", "/summary", base=OPS)
    assert s.status_code == 200, s.text
    body = s.json()
    assert [o["id"] for o in body["orgs"]] == [ORG, "org_web"]  # by name: Acme, Solo
    assert body["orgs"][0] == {
        "id": ORG,
        "name": "Acme",
        "counts": {"running": 1, "failed": 1},
        "attention": 1,
        "seq": 1,
    }
    assert body["orgs"][1] == {
        "id": "org_web",
        "name": "Solo",
        "counts": {},
        "attention": 0,
        "seq": 0,
    }
    assert body["unlinked"] == [
        {"key": "guild:999", "seq": 1, "last_at": (NOW - timedelta(minutes=2)).isoformat()}
    ]
    bot = body["heartbeats"]["bot"]
    assert bot["state"] == "online" and bot["queues"]["actions"] == 2 and bot["tick_at"]
    assert body["now"] == NOW.isoformat()
    rows = get(w, "root", "/items", base=OPS).json()
    assert [(r["org_id"], r["id"]) for r in rows["items"]] == [(ORG, "j1"), (ORG, "j2")]
    assert rows["next_cursor"] is None and "audience" not in rows["items"][0]
    assert ids_of(get(w, "root", "/items", base=OPS, params={"status": "failed"})) == ["j2"]
    assert ids_of(get(w, "root", "/items", base=OPS, params={"ids": "video_ad:j2"})) == ["j2"]
    assert get(w, "root", "/items", base=OPS, params={"limit": 101}).status_code == 422
    for who in ("mem", "boss", "adm"):
        assert get(w, who, "/summary", base=OPS).status_code == 403, who
        assert get(w, who, "/items", base=OPS).status_code == 403, who
    assert w.c.get(OPS + "/summary").status_code == 401


# ------------------------------------------------------------------ privacy
def test_audience_and_sentinels_never_returned(w):
    j1 = item(
        "j1",
        people=["discord:" + SENTINEL, "discord:7"],
        steps=[Step(name="render", state="failed", error_code="render")],
    )
    w.items.append(j1)
    monitor.emit(w.docs, ORG, monitor.event_for(j1))
    monitor.emit(w.docs, "guild:999", monitor.event_for(j1))
    responses = [
        get(w, "boss", "/summary"),
        get(w, "boss", "/items"),
        get(w, "mem", "/items", params={"ids": "video_ad:j1"}),
        get(w, "boss", "/items/video_ad/j1"),
        get(w, "mem", "/items/video_ad/j1"),
        get(w, "boss", "/changes?since=0"),
        get(w, "mem", "/changes?since=0"),
        get(w, "root", "/summary", base=OPS),
        get(w, "root", "/items", base=OPS),
    ]
    assert [r.status_code for r in responses] == [200] * len(responses)
    bodies = [r.json() for r in responses]
    assert SENTINEL not in "".join(strings(bodies))
    assert "audience" not in set(strings(bodies))  # never a key anywhere
    item_keys = set(bodies[1]["items"][0]) | set(bodies[3]) | set(bodies[5]["events"][0])
    assert "audience" not in item_keys and "mine" in item_keys
