"""Workflow monitor foundation: item model, access matrix, attention rules and error codes;
the change log, the emitter, heartbeats and the list helpers."""

from __future__ import annotations

import ast
import contextlib
import json
import logging
import threading
from datetime import UTC, datetime, timedelta
from importlib import metadata
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fake_firestore import FakeFirestore

from salid import monitor
from salid.coo_store import FirestoreDocStore, SQLiteDocStore
from salid.monitor import (
    ACTIVE,
    BOT_KINDS,
    EMIT_STATS,
    ERROR_TEXT,
    EVENT_V,
    FINAL,
    HEARTBEAT,
    HEARTBEAT_EVERY_S,
    KINDS,
    LOG,
    MAX_EVENT_PEOPLE,
    RING,
    STATUSES,
    Audience,
    Emitter,
    MonitorItem,
    SafeError,
    Step,
    Viewer,
    allowed,
    app_version,
    apply_attention,
    changes_since,
    controls,
    emit,
    emit_items,
    emit_many,
    event_for,
    filter_items,
    heartbeat_state,
    item_events,
    log_key,
    mine,
    org_key_resolver,
    paginate,
    person_key,
    read_heartbeats,
    read_log,
    safe_error,
    stuck_after_s,
    summarize,
    viewer_for,
    visible,
    write_heartbeat,
)
from salid.orgs import Member, Org, OrgStore, Workspace

NOW = datetime(2026, 10, 18, 9, 0, tzinfo=UTC)
BANGKOK = ZoneInfo("Asia/Bangkok")


def V(role: str, ids: tuple[str, ...] = (), ex: bool = False) -> Viewer:
    return Viewer("x@y.com", role, frozenset({"x@y.com", *ids}), ex)


def item(
    kind: str = "video_ad",
    status: str = "running",
    *,
    age_s: float = 0,
    audience: Audience | None = None,
    **kw,
) -> MonitorItem:
    at = (NOW - timedelta(seconds=age_s)).isoformat()
    return MonitorItem(
        kind=kind,
        id="j1",
        status=status,
        started_at=at,
        updated_at=at,
        audience=audience or Audience(),
        **kw,
    )


def test_constants_and_item_key():
    assert ACTIVE == ("queued", "running", "waiting", "paused")
    assert FINAL == ("done", "failed", "cancelled")
    assert set(STATUSES) == set(ACTIVE) | set(FINAL) and len(STATUSES) == 7
    assert set(KINDS) == {
        "video_ad",
        "deep_task",
        "research",
        "change",
        "action",
        "meeting",
        "task",
        "approval",
        "vote",
        "form",
        "reminder",
    }
    assert set(BOT_KINDS) == {"video_ad", "deep_task", "research", "action"}
    it = item("deep_task", steps=[Step(name="plan")])
    assert it.key == "deep_task:j1"
    assert it.steps[0].model_dump() == {
        "name": "plan",
        "state": "pending",
        "started_at": "",
        "ms": None,
        "attempts": 0,
        "error_code": "",
    }
    assert (it.attention, it.attention_reason, it.error, it.cost_usd) == (False, "", None, None)
    assert Audience().model_dump() == {
        "admin_only": False,
        "exec_only": False,
        "dm": False,
        "people": [],
        "channel_ids": [],
    }
    assert person_key("discord", 7) == "discord:7" and person_key("slack", "U1") == "slack:U1"


@pytest.mark.parametrize(
    "role,ex,kind,aud,ids,ok",
    [
        ("owner", False, "video_ad", Audience(people=["discord:7"]), (), True),
        ("owner", False, "video_ad", Audience(dm=True, people=["discord:7"]), (), False),
        ("admin", False, "deep_task", Audience(exec_only=True, people=["discord:7"]), (), False),
        ("admin", True, "deep_task", Audience(exec_only=True, people=["discord:7"]), (), True),
        ("member", False, "change", Audience(admin_only=True), (), False),
        ("member", False, "action", Audience(admin_only=True, people=["x@y.com"]), (), False),
        ("member", False, "video_ad", Audience(people=["discord:7"]), ("discord:7",), True),
        ("member", False, "video_ad", Audience(people=["discord:8"]), ("discord:7",), False),
        ("super", False, "deep_task", Audience(exec_only=True), (), True),
        ("super", False, "video_ad", Audience(dm=True, people=["discord:7"]), (), False),
        # beyond the pinned rows: the dm owner, admin_only for staff, a non-exec admin's own job
        (
            "member",
            False,
            "video_ad",
            Audience(dm=True, people=["discord:7"]),
            ("discord:7",),
            True,
        ),
        ("admin", False, "change", Audience(admin_only=True), (), True),
        (
            "admin",
            False,
            "deep_task",
            Audience(exec_only=True, people=["discord:7"]),
            ("discord:7",),
            True,
        ),
        # deep tasks are executive-only per spec 4.1 even when an adapter forgot the flag
        ("admin", False, "deep_task", Audience(people=["discord:7"]), (), False),
        ("admin", False, "video_ad", Audience(exec_only=True, people=["discord:7"]), (), False),
        ("", False, "video_ad", Audience(people=["discord:7"]), (), False),
    ],
)
def test_visible_roles_by_kinds_matrix(role, ex, kind, aud, ids, ok):
    viewer = V(role, ids, ex)
    assert allowed(kind, aud, viewer) is ok
    assert visible(MonitorItem(kind=kind, id="1", status="running", audience=aud), viewer) is ok


def test_controls_matrix():
    seven = Audience(people=["discord:7"])
    staff_only = Audience(admin_only=True, people=["x@y.com"])
    own_deep = Audience(exec_only=True, people=["discord:7"])
    me = ("discord:7",)
    assert controls(item("video_ad", audience=seven), V("owner")) == ["cancel"]
    assert controls(item("video_ad", "queued", audience=seven), V("owner")) == ["cancel"]
    assert controls(item("action", "failed", audience=staff_only), V("owner")) == ["retry"]
    assert controls(item("change", "failed", audience=Audience(admin_only=True)), V("owner")) == [
        "resend"
    ]
    assert controls(item("change", "failed", audience=Audience(admin_only=True)), V("admin")) == [
        "resend"
    ]
    assert controls(item("video_ad", "done", audience=seven), V("owner")) == []
    assert controls(item("research", audience=seven), V("admin")) == ["cancel"]
    mine_video = item("video_ad", audience=seven, requester_id="discord:7")
    assert controls(mine_video, V("member", me)) == ["cancel"]
    mine_deep = item("deep_task", audience=own_deep, requester_id="discord:7")
    assert controls(mine_deep, V("member", me)) == []
    assert controls(mine_deep, V("member", me, ex=True)) == ["cancel"]
    # a non-executive admin cannot see someone else's deep task, so cannot cancel it
    assert controls(item("deep_task", audience=own_deep), V("admin")) == []
    assert controls(item("deep_task", audience=own_deep), V("admin", ex=True)) == ["cancel"]
    mine_action = item("action", "failed", audience=staff_only, requester_id="x@y.com")
    assert controls(mine_action, V("member")) == []
    # in the audience but not the requester: no cancel for a member
    assert (
        controls(item("video_ad", audience=seven, assignee_id="discord:7"), V("member", me)) == []
    )
    assert controls(item("video_ad", audience=seven), V("super")) == []
    # never a control on an item the viewer cannot see (dm items are hidden from the owner)
    dm = Audience(dm=True, people=["discord:7"])
    assert controls(item("video_ad", audience=dm, requester_id="discord:7"), V("owner")) == []


def test_mine_matches_requester_assignee_or_waiting_on():
    me = V("member", ("discord:7",))
    assert mine(item(requester_id="discord:7"), me)
    assert mine(item(assignee_id="x@y.com"), me)
    assert mine(item(waiting_on="discord:7"), me)
    assert not mine(item(requester_id="discord:8"), me)
    assert not mine(item(), me)


def _org(level: int, role: str = "admin") -> Org:
    member = Member(
        email="a@x.com",
        role=role,
        discord_user_id="7",
        identities={"slack": "U1", "line": ""},
        level=level,
    )
    return Org(id="org_t", name="T", members=[member])


def test_viewer_for_collects_verified_ids_and_executive_level():
    v = viewer_for(_org(3), "A@x.com", "admin")
    assert v.ids == {"a@x.com", "discord:7", "slack:U1"}
    assert (v.email, v.role, v.executive) == ("a@x.com", "admin", True)
    assert viewer_for(_org(5), "a@x.com", "admin").executive is False
    assert viewer_for(_org(0), "a@x.com", "admin").executive is False
    assert viewer_for(_org(0, "owner"), "a@x.com", "owner").executive is True
    assert viewer_for(_org(1, "member"), "a@x.com", "member").executive is True
    assert viewer_for(_org(5, "member"), "a@x.com", "member").executive is False
    assert viewer_for(_org(0, "member"), "a@x.com", "member").executive is False
    ops = viewer_for(_org(3), "ops@salid.app", "super")
    assert (ops.role, ops.ids, ops.executive) == ("super", frozenset({"ops@salid.app"}), False)


def test_exec_level_member_cancels_own_deep_task_but_sees_no_others():
    # spec 4.1: members cancel their own jobs, deep tasks only when they are executives
    def deep(requester: str) -> MonitorItem:
        aud = Audience(exec_only=True, people=[requester])
        return item("deep_task", audience=aud, requester_id=requester)

    director = viewer_for(_org(2, "member"), "a@x.com", "member")
    staff = viewer_for(_org(5, "member"), "a@x.com", "member")
    assert controls(deep("discord:7"), director) == ["cancel"]
    assert controls(deep("discord:7"), staff) == []
    # being an executive never widens what a member may see
    other = deep("discord:8")
    assert not visible(other, director) and controls(other, director) == []


def test_heartbeat_thresholds_and_clock_skew():
    ages = {90: "online", 91: "delayed", 300: "delayed", 301: "offline", -120: "online"}
    for age, want in ages.items():
        assert heartbeat_state((NOW - timedelta(seconds=age)).isoformat(), NOW) == want
    assert heartbeat_state(None, NOW) == heartbeat_state("garbage", NOW) == "offline"
    assert heartbeat_state("", NOW) == "offline"
    # a naive timestamp is read as UTC
    assert heartbeat_state("2026-10-18T08:59:00", NOW) == "online"


def test_stuck_thresholds_per_kind():
    assert stuck_after_s("video_ad") == 600
    assert stuck_after_s("deep_task") == 720
    assert stuck_after_s("deep_task", deep_timeout_s=900) == 1020
    assert stuck_after_s("research") == 900
    assert stuck_after_s("change") is None and stuck_after_s("action") is None


def test_attention_rules():
    def att(it: MonitorItem, bot: str = "online", **kw) -> tuple[bool, str]:
        out = apply_attention(it, NOW, bot, **kw)
        return out.attention, out.attention_reason

    none = (False, "")
    assert att(item("video_ad", age_s=601)) == (True, "stuck")
    assert att(item("video_ad", age_s=599)) == none
    assert att(item("deep_task", age_s=721), deep_timeout_s=600) == (True, "stuck")
    assert att(item("deep_task", age_s=719), deep_timeout_s=600) == none
    assert att(item("deep_task", age_s=1019), deep_timeout_s=900) == none
    assert att(item("research", age_s=901)) == (True, "stuck")
    assert att(item("research", age_s=899)) == none
    assert att(item("video_ad", "done", age_s=10_000)) == none
    assert att(item("action", "queued", age_s=61)) == (True, "stuck")
    assert att(item("action", "queued", age_s=61), "delayed") == none
    assert att(item("action", "queued", age_s=59)) == none
    assert att(item("video_ad"), "offline") == (True, "bot_offline")
    assert att(item("action", "queued"), "offline") == (True, "bot_offline")
    assert att(item("deep_task", "done"), "offline") == none
    assert att(item("change"), "offline") == none
    for kind in KINDS:
        assert att(item(kind, "failed", age_s=10_000), "offline") == (True, "failed")
    # adapters set their own attention (changes: stuck); it is kept when no rule applies
    flagged = item("change", "queued", attention=True, attention_reason="stuck")
    assert att(flagged) == (True, "stuck")
    # pure: the input item is never changed
    original = item("video_ad", age_s=601)
    apply_attention(original, NOW)
    assert (original.attention, original.attention_reason) == none


def test_safe_error_never_echoes_input():
    assert safe_error("discord: SECRET").model_dump() == {
        "code": "unknown",
        "message": ERROR_TEXT["unknown"],
    }
    assert safe_error("month_limit") == SafeError(
        code="month_limit", message=ERROR_TEXT["month_limit"]
    )
    assert safe_error("").code == "unknown"


def test_error_text_codes_are_short_english_sentences():
    assert set(ERROR_TEXT) == {
        "unknown",
        "restarted",
        "failed",
        "error",
        "month_limit",
        "storyboard",
        "render",
        "upload_refused",
        "not_confirmed",
        "delivery_failed",
        "missing_permission",
        "not_found",
        "bad_request",
        "discord_error",
        "expired",
    }
    for text in ERROR_TEXT.values():
        assert text.isascii() and text[0].isupper() and text.endswith(".")
        assert len(text) <= 80 and text.count(".") == 1


def test_monitor_never_imports_discord():
    tree = ast.parse(Path(monitor.__file__).read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [n for n in names if n == "discord" or n.startswith("discord.")]


# ------------------------------------------------------------------ change log
@pytest.fixture(params=["sqlite", "firestore"])
def docs(request):
    if request.param == "sqlite":
        return SQLiteDocStore(":memory:")
    return FirestoreDocStore(client=FakeFirestore())


def ev(i: int, kind: str = "video_ad", status: str = "running", **audience) -> dict:
    """A producer event as `event_for` builds it."""
    aud = Audience(**audience).model_dump()
    return {"kind": kind, "id": f"j{i}", "status": status, "attention": False, "audience": aud}


class Broken:
    """A store whose every write fails."""

    transact = staticmethod(lambda fn: (_ for _ in ()).throw(RuntimeError("down")))

    def put(self, *a):
        raise RuntimeError("down")


EVENT_KEYS = {"v", "seq", "at", "kind", "id", "status", "attention", "audience"}


def test_log_constants_and_keys():
    assert (LOG, HEARTBEAT, RING, EVENT_V) == (
        "salid_monitor_log",
        "salid_monitor_heartbeat",
        200,
        1,
    )
    assert (MAX_EVENT_PEOPLE, HEARTBEAT_EVERY_S) == (20, 30.0)
    assert log_key("org_t", 1000) == "org_t"
    assert log_key(None, 1000) == log_key("", 1000) == "guild:1000"


def test_emit_assigns_increasing_seq_and_trims_ring_to_200(docs):
    assert emit_many(docs, "org_t", [ev(i) for i in range(205)], now=NOW) == 205
    log = read_log(docs, "org_t")
    head = (log["seq"], log["floor"], len(log["events"]), log["events"][0]["seq"])
    assert head == (205, 5, 200, 6)
    assert set(log["events"][0]) == EVENT_KEYS
    assert [e["seq"] for e in log["events"]] == list(range(6, 206))
    first = log["events"][0]
    assert (first["v"], first["at"], first["id"]) == (EVENT_V, NOW.isoformat(), "j5")
    # the next write continues the sequence; the ring stays at 200
    assert emit(docs, "org_t", ev(999), now=NOW) == 206
    log = read_log(docs, "org_t")
    assert (log["seq"], log["floor"], len(log["events"])) == (206, 6, 200)
    # each key has its own sequence; a missing log reads as empty
    assert emit(docs, "guild:5", ev(1)) == 1
    assert read_log(docs, "nope") == {"seq": 0, "floor": 0, "events": []}
    # fields a producer adds never reach the log (events carry no content)
    emit(docs, "org_x", {**ev(1), "title": "SECRET brief", "error": "Traceback"}, now=NOW)
    stored = read_log(docs, "org_x")["events"][0]
    assert set(stored) == EVENT_KEYS and "SECRET" not in json.dumps(stored)


def test_changes_since_floor_or_future_resets(docs):
    emit_many(docs, "org_t", [ev(i) for i in range(205)], now=NOW)
    log, owner = read_log(docs, "org_t"), V("owner")
    assert changes_since(log, 4, owner)["reset"] is True
    got = changes_since(log, 5, owner)
    assert (got["reset"], len(got["events"]), got["seq"], got["floor"]) == (False, 200, 205, 5)
    assert changes_since(log, 205, owner) == {"seq": 205, "floor": 5, "events": [], "reset": False}
    assert changes_since(log, 206, owner)["reset"] is True
    assert [e["seq"] for e in changes_since(log, 203, owner)["events"]] == [204, 205]
    # an empty log is no reset at since=0
    empty = changes_since(read_log(docs, "none"), 0, owner)
    assert empty == {"seq": 0, "floor": 0, "events": [], "reset": False}


def test_changes_since_sorts_by_seq_and_strips_audience():
    def raw(seq: int) -> dict:
        return {**ev(seq), "v": 1, "seq": seq, "at": NOW.isoformat()}

    log = {"seq": 3, "floor": 0, "events": [raw(3), raw(1), raw(2)]}
    out = changes_since(log, 0, V("owner"))["events"]
    assert [e["seq"] for e in out] == [1, 2, 3]
    assert set(out[0]) == EVENT_KEYS - {"audience"}


def test_concurrent_emits_never_lose_or_repeat_seq(docs):
    start, results = threading.Barrier(2), []

    def worker(t: int) -> None:
        start.wait()
        results.extend(emit(docs, "org_t", ev(t * 100 + i)) for i in range(20))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
        assert not t.is_alive()
    log = read_log(docs, "org_t")
    assert sorted(e["seq"] for e in log["events"]) == list(range(1, 41))
    assert sorted(results) == list(range(1, 41))
    want = sorted(f"j{t * 100 + i}" for t in range(2) for i in range(20))
    assert sorted(e["id"] for e in log["events"]) == want


def test_conflicting_emits_retry_and_keep_seq_unique(docs):
    """Both writers read the log before either commits: Firestore aborts one and re-runs it on
    the new log; SQLite serialises them (the barrier times out). Never the same seq twice."""
    seen = threading.Barrier(2)
    transact = docs.transact

    def racy(fn):
        def first_read_waits(tx):
            get = tx.get

            def get_then_wait(*a):
                doc = get(*a)
                with contextlib.suppress(threading.BrokenBarrierError):
                    seen.wait(timeout=0.5)
                return doc

            tx.get = get_then_wait
            return fn(tx)

        return transact(first_read_waits)

    docs.transact = racy
    results: list[int | None] = []
    threads = [
        threading.Thread(target=lambda i=i: results.append(emit(docs, "org_t", ev(i))))
        for i in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert sorted(results) == [1, 2]
    assert [e["seq"] for e in read_log(docs, "org_t")["events"]] == [1, 2]
    if isinstance(docs, FirestoreDocStore):
        assert docs.client.aborts >= 1  # the conflict really happened


def test_emit_failure_is_counted_and_never_raises(caplog):
    n = EMIT_STATS["failed"]
    with caplog.at_level(logging.WARNING, logger="salid.monitor"):
        assert emit(Broken(), "org_t", ev(1)) is None
    assert EMIT_STATS["failed"] == n + 1
    assert emit_many(Broken(), "org_t", [ev(1), ev(2)]) is None
    assert EMIT_STATS["failed"] == n + 2
    assert emit_items(Broken(), "org_t", [item()]) is None
    assert EMIT_STATS["failed"] == n + 3
    # only the error type is logged, never its text
    assert "RuntimeError" in caplog.text and "down" not in caplog.text
    # nothing to write: no transaction, no failure
    assert emit_items(Broken(), "org_t", []) is None
    assert EMIT_STATS["failed"] == n + 3


def test_emit_items_writes_one_event_per_item(docs):
    emit_items(docs, "org_t", [item(), item("deep_task", "failed")])
    log = read_log(docs, "org_t")
    assert [(e["seq"], e["kind"], e["status"]) for e in log["events"]] == [
        (1, "video_ad", "running"),
        (2, "deep_task", "failed"),
    ]


def test_changes_are_filtered_per_viewer(docs):
    events = [
        ev(1, admin_only=True),
        ev(2, people=["discord:7"]),
        ev(3, dm=True, people=["discord:7"]),
        ev(4, kind="deep_task", exec_only=True, people=["discord:7"]),
    ]
    emit_many(docs, "org_t", events, now=NOW)
    log = read_log(docs, "org_t")

    def ids(viewer: Viewer) -> list[str]:
        return [e["id"] for e in changes_since(log, 0, viewer)["events"]]

    assert ids(V("member", ("discord:7",))) == ["j2", "j3", "j4"]
    assert ids(V("member", ("discord:8",))) == []
    assert ids(V("owner")) == ["j1", "j2", "j4"]  # the dm event is hidden from the owner
    assert ids(V("admin")) == ["j1", "j2"]
    assert ids(V("admin", ex=True)) == ["j1", "j2", "j4"]
    assert ids(V("super")) == ["j1", "j2", "j4"]
    # the audience never leaves the server
    assert all("audience" not in e for e in changes_since(log, 0, V("owner"))["events"])
    # a malformed audience fails closed, even for the owner
    bad = {**log, "events": [{**log["events"][1], "audience": {"people": 5}}]}
    assert changes_since(bad, 0, V("owner"))["events"] == []


def test_event_carries_no_content():
    it = item(
        title="Video ad · SECRET brief",
        error=safe_error("render"),
        raw_status="render",
        requester_id="discord:7",
        steps=[Step(name="render", error_code="render")],
        links={"discord": "https://discord.com/channels/1/2/3"},
        audience=Audience(
            people=[f"discord:{n}" for n in range(22)],
            channel_ids=[str(n) for n in range(22)],
        ),
    )
    e = event_for(it)
    assert set(e) == {"kind", "id", "status", "attention", "audience"}
    assert (e["kind"], e["id"], e["status"], e["attention"]) == ("video_ad", "j1", "running", False)
    assert "SECRET" not in json.dumps(e) and "render" not in json.dumps(e)
    assert e["audience"]["people"] == [f"discord:{n}" for n in range(20)]
    assert len(e["audience"]["channel_ids"]) == 20
    assert it.audience.people == [f"discord:{n}" for n in range(22)]  # the item is unchanged


def test_item_events_for_one_item_with_severity(docs):
    emit_many(
        docs,
        "org_t",
        [
            ev(1),
            ev(2),
            {**ev(1), "attention": True},
            ev(1, status="failed"),
            ev(1, kind="deep_task"),
        ],
        now=NOW,
    )
    out = item_events(read_log(docs, "org_t"), "video_ad", "j1")
    assert [(e["seq"], e["severity"]) for e in out] == [(1, "info"), (3, "warning"), (4, "error")]
    assert all("audience" not in e for e in out)
    assert item_events(read_log(docs, "org_t"), "video_ad", "nope") == []


class SpyStore:
    """A real SQLite store that counts transactions."""

    def __init__(self):
        self.inner, self.n = SQLiteDocStore(":memory:"), 0

    def transact(self, fn):
        self.n += 1
        return self.inner.transact(fn)


def test_emitter_batches_one_transaction_per_key_per_flush():
    spy = SpyStore()
    em = Emitter(spy, enabled=True, resolve=lambda m: f"guild:{m}", now=lambda: NOW)
    for i in range(5):
        em.add("org_a", ev(i))
    for i in range(3):
        em.add_for_guild(77, ev(10 + i))
    assert em.flush() == 8 and spy.n == 2
    assert read_log(spy.inner, "org_a")["seq"] == 5
    guild = read_log(spy.inner, "guild:77")
    assert [e["id"] for e in guild["events"]] == ["j10", "j11", "j12"]
    assert guild["events"][0]["at"] == NOW.isoformat()
    assert em.flush() == 0 and spy.n == 2  # nothing buffered, nothing written
    # a guild that resolves to the org joins its batch, in the order they were added
    same = Emitter(spy, enabled=True, resolve=lambda m: "org_b")
    same.add("org_b", ev(1))
    same.add_for_guild(77, ev(2))
    same.add("org_b", ev(3))
    assert same.flush() == 3 and spy.n == 3
    assert [e["id"] for e in read_log(spy.inner, "org_b")["events"]] == ["j1", "j2", "j3"]
    # no resolver: the unlinked key
    plain = Emitter(spy, enabled=True)
    plain.add_for_guild(5, ev(1))
    assert plain.flush() == 1 and read_log(spy.inner, "guild:5")["seq"] == 1
    # disabled: nothing is buffered or written
    off = Emitter(spy, enabled=False)
    off.add("org_a", ev(1))
    off.add_for_guild(77, ev(2))
    assert off.flush() == 0 and spy.n == 4


def test_emitter_never_raises():
    spy = SpyStore()
    n = EMIT_STATS["failed"]
    bad = Emitter(spy, enabled=True, resolve=lambda m: 1 / 0)
    bad.add_for_guild(1, ev(1))
    bad.add("org_a", ev(2))
    assert bad.flush() == 1 and read_log(spy.inner, "org_a")["seq"] == 1
    assert EMIT_STATS["failed"] == n + 1  # the unresolved guild's events are dropped
    down = Emitter(Broken(), enabled=True)
    down.add("org_a", ev(1))
    down.add_for_guild(2, ev(2))
    assert down.flush() == 0 and EMIT_STATS["failed"] == n + 3
    assert down.flush() == 0  # failed events are not kept for the next tick
    down.add("", ev(3))  # no key: dropped, not raised
    assert down.flush() == 0


def test_emitter_adds_from_many_threads():
    spy = SpyStore()
    em = Emitter(spy, enabled=True)

    def worker(t: int) -> None:
        for i in range(50):
            em.add("org_a", ev(t * 100 + i))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert em.flush() == 200 and read_log(spy.inner, "org_a")["seq"] == 200


def test_org_key_resolver_maps_linked_guilds_and_caches(docs):
    clock = [0.0]
    resolve = org_key_resolver(docs, ttl_s=60.0, clock=lambda: clock[0])
    assert resolve(1000) == "guild:1000"
    orgs = OrgStore(docs)
    orgs.create("T", "boss@t.com", org_id="org_t")
    orgs.attach_workspace(
        "org_t", Workspace(id="discord-1000", platform="discord", external_id="1000")
    )
    assert resolve(1000) == "guild:1000"  # cached for the TTL
    clock[0] = 60.0
    assert resolve(1000) == "org_t"
    assert org_key_resolver(docs)(1000) == "org_t"
    assert org_key_resolver(docs)(2000) == "guild:2000"


# ------------------------------------------------------------------ heartbeat
def test_heartbeat_round_trip(docs):
    assert read_heartbeats(docs) == {}
    queues = {"actions": 2, "outbox_pending": 1, "memory": 0}
    assert write_heartbeat(docs, "bot", queues=queues, now=NOW) is True
    later = NOW + timedelta(seconds=30)
    assert write_heartbeat(docs, "platforms", queues={"memory": 3}, now=later) is True
    beats = read_heartbeats(docs)
    assert beats == {
        "bot": {
            "process": "bot",
            "version": app_version(),
            "tick_at": NOW.isoformat(),
            "queues": queues,
        },
        "platforms": {
            "process": "platforms",
            "version": app_version(),
            "tick_at": later.isoformat(),
            "queues": {"actions": 0, "outbox_pending": 0, "memory": 3},
        },
    }
    assert heartbeat_state(beats["bot"]["tick_at"], NOW + timedelta(seconds=60)) == "online"
    assert write_heartbeat(Broken(), "bot", queues=queues, now=NOW) is False


def test_app_version(monkeypatch):
    monkeypatch.setenv("K_REVISION", "meeting-scribe-00042-abc")
    assert app_version() == "meeting-scribe-00042-abc"
    monkeypatch.delenv("K_REVISION")
    assert app_version() == metadata.version("meeting-scribe")

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(monitor.metadata, "version", missing)
    assert app_version() == "dev"


# ------------------------------------------------------------------ list helpers
def test_filter_paginate_and_summarize():
    items = [MonitorItem(kind="video_ad", id=f"i{n:03d}", status="running") for n in range(120)]
    page, nxt = paginate(items, None, 50)
    assert (len(page), nxt, page[0].id) == (50, "50", "i000")
    page, nxt = paginate(items, "50", 50)
    assert (page[0].id, nxt) == ("i050", "100")
    page, nxt = paginate(items, "100", 50)
    assert (len(page), nxt, page[0].id) == (20, None, "i100")
    assert paginate(items, "junk", 10)[0][0].id == "i000"
    assert len(paginate(items, None, 500)[0]) == 100  # the page size is capped
    assert paginate([], None, 50) == ([], None)

    # NOW is 16:00 in Bangkok; 2026-10-17T18:00Z is 01:00 on the 18th there, 16:59Z is 23:59 on
    # the 17th: the same UTC date, different Bangkok dates
    done = [
        MonitorItem(kind="change", id="d1", status="done", finished_at="2026-10-17T18:00:00+00:00"),
        MonitorItem(kind="change", id="d2", status="done", finished_at="2026-10-17T16:59:00+00:00"),
        MonitorItem(kind="change", id="d3", status="done", updated_at="2026-10-18T08:00:00+00:00"),
    ]
    ago = lambda **kw: (NOW - timedelta(**kw)).isoformat()  # noqa: E731
    mixed = [
        MonitorItem(kind="video_ad", id="r", status="running"),
        MonitorItem(kind="video_ad", id="q", status="queued", attention=True),
        MonitorItem(
            kind="deep_task", id="f1", status="failed", attention=True, finished_at=ago(hours=23)
        ),
        MonitorItem(
            kind="deep_task", id="f2", status="failed", attention=True, finished_at=ago(hours=25)
        ),
        *done,
    ]
    assert summarize(mixed, NOW, BANGKOK) == {
        "counts": {"running": 1, "queued": 1, "failed": 2, "done": 3},
        "by_kind": {"video_ad": 2, "deep_task": 2, "change": 3},
        "attention": 3,
        "failed_24h": 1,
        "done_today": 2,
    }
    empty = {"counts": {}, "by_kind": {}, "attention": 0, "failed_24h": 0, "done_today": 0}
    assert summarize([], NOW, BANGKOK) == empty


def test_filter_items():
    a = MonitorItem(
        kind="video_ad",
        id="a",
        status="running",
        title="Video ad · 5 scenes",
        workspace="discord-1000",
        requester_id="discord:7",
        attention=True,
        attention_reason="stuck",
        updated_at=(NOW - timedelta(hours=1)).isoformat(),
    )
    b = MonitorItem(
        kind="deep_task",
        id="b",
        status="failed",
        title="Deep task",
        workspace="web",
        assignee_id="a@x.com",
        finished_at=(NOW - timedelta(days=3)).isoformat(),
    )
    c = MonitorItem(
        kind="change",
        id="c",
        status="done",
        title="Change → Slack",
        workspace="slack-T1",
        waiting_on="discord:7",
        started_at=(NOW - timedelta(hours=2)).isoformat(),
    )
    d = MonitorItem(kind="action", id="d", status="queued")

    def ids(**kw) -> list[str]:
        return [i.id for i in filter_items([a, b, c, d], **kw)]

    assert ids() == ids(kinds=[], statuses=[]) == ["a", "b", "c", "d"]
    assert ids(kinds=["video_ad", "change"]) == ["a", "c"]
    assert ids(statuses=["failed", "queued"]) == ["b", "d"]
    assert ids(attention=True) == ["a"] and ids(attention=False) == ["b", "c", "d"]
    assert ids(owner="discord:7") == ["a"]  # requester or assignee, never waiting_on
    assert ids(owner="A@X.com") == ["b"]
    assert ids(workspace="discord-1000") == ids(workspace="discord") == ["a"]
    assert ids(workspace="web") == ["b"] and ids(workspace="disc") == []
    # date range on the last activity, inclusive; items without a time are left out
    assert ids(start=NOW - timedelta(days=1)) == ["a", "c"]
    assert ids(end=(NOW - timedelta(days=1)).isoformat()) == ["b"]
    assert ids(start=NOW - timedelta(hours=2), end=NOW - timedelta(hours=2)) == ["c"]
    assert ids(q="slack") == ["c"] and ids(q="DEEP") == ["b"] and ids(q="  ") == ids()
    assert ids(kinds=["video_ad"], attention=False) == []
