"""Workflow monitor delivery adapters: change outbox and web -> Discord action queue records as
monitor items (status, steps, `stuck`, safe error codes, no content), and their registered
sources and finders over a SQLite / fake-Firestore doc store. Offline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fake_firestore import FakeFirestore

from salid import monitor_sources
from salid.actions import ActionQueue
from salid.change_outbox import KIND as OUTBOX
from salid.change_outbox import Change, Target, org_guild
from salid.coo_store import FirestoreDocStore, SQLiteDocStore
from salid.monitor import Audience
from salid.monitor_sources import (
    action_error_code,
    actions_finder,
    actions_source,
    adapt_action,
    adapt_change,
    adapt_video,
    changes_finder,
    changes_source,
    video_finder,
    video_source,
)
from salid.orgs import Org, Workspace

T0 = datetime(2026, 10, 18, 9, 0, tzinfo=UTC)
SENTINEL = "SENSITIVE-7f3a"
GUILD = 1000


@pytest.fixture(params=["sqlite", "firestore"])
def docs(request):
    if request.param == "sqlite":
        return SQLiteDocStore(":memory:")
    return FirestoreDocStore(client=FakeFirestore())


def target(status, *, platform="discord", kind="apply", attempts=1, last_error="", sent_at=""):
    return Target(
        workspace_id=f"{platform}-1",
        platform=platform,
        kind=kind,
        person="7" if kind == "notify" else "",
        status=status,
        attempts=attempts,
        sent_at=sent_at,
        last_error=last_error,
    )


def change(*targets, change_id="c1", org_id="o1", at=T0, kind="role"):
    return Change(
        id=change_id,
        org_id=org_id,
        at=at.isoformat(),
        by="boss@example.com",
        kind=kind,
        subject_email=SENTINEL + "@x.com",
        subject_name=SENTINEL,
        before=SENTINEL,
        after=SENTINEL,
        targets=list(targets),
    )


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from strings(k)
            yield from strings(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from strings(v)


# ------------------------------------------------------------------ change adapter
def test_change_status_precedence_and_steps():
    later = (T0 + timedelta(minutes=5)).isoformat()
    failed = adapt_change(
        change(
            target("acked", sent_at=later),
            target("failed", kind="notify", attempts=8, last_error="not confirmed in time"),
        ),
        now=T0 + timedelta(minutes=10),
    )
    assert (failed.status, failed.attention, failed.attention_reason) == ("failed", True, "failed")
    assert failed.error is not None and failed.error.code == "not_confirmed"
    assert [(s.name, s.state, s.attempts, s.error_code) for s in failed.steps] == [
        ("discord apply", "done", 1, ""),
        ("discord notify", "failed", 8, "not_confirmed"),
    ]
    assert failed.steps[0].started_at == later and failed.steps[1].error_code == "not_confirmed"
    assert failed.raw_status == "1/2 acked" and failed.finished_at == later
    now = T0 + timedelta(minutes=1)
    queued = adapt_change(change(target("pending"), target("sent", kind="notify")), now=now)
    assert (queued.status, queued.attention) == ("queued", False)
    assert [s.state for s in queued.steps] == ["pending", "running"]
    waiting = adapt_change(change(target("sent"), target("acked", kind="notify")), now=now)
    assert (waiting.status, waiting.raw_status, waiting.finished_at) == ("waiting", "1/2 acked", "")
    empty = adapt_change(change(), now=now)
    assert (empty.status, empty.raw_status, empty.steps) == ("done", "0/0 acked", [])
    assert empty.finished_at == T0.isoformat() and empty.error is None
    done = adapt_change(change(target("acked"), target("acked", kind="notify")), now=now)
    assert (done.status, done.raw_status, done.attention) == ("done", "2/2 acked", False)
    other = adapt_change(
        change(target("failed", last_error=SENTINEL), target("pending", last_error=SENTINEL)),
        now=now,
    )
    assert [(s.state, s.error_code) for s in other.steps] == [
        ("failed", "delivery_failed"),
        ("pending", "delivery_failed"),  # its last attempt failed; it is still being retried
    ]
    assert other.error is not None and other.error.code == "delivery_failed"


def test_change_item_labels_audience_and_no_subject():
    it = adapt_change(change(target("failed", last_error=SENTINEL)), now=T0)
    assert (it.kind, it.id, it.org_id, it.workspace) == ("change", "c1", "o1", "web")
    assert it.title == "Change · role" and it.started_at == T0.isoformat()
    assert it.audience == Audience(admin_only=True)
    assert (it.requester_id, it.assignee_id, it.waiting_on) == ("", "", "")
    assert SENTINEL not in "".join(strings(it.model_dump()))
    assert adapt_change(change(kind=SENTINEL), now=T0).title == "Change · other"


def test_change_open_past_deadline_is_stuck():
    deadline = T0 + timedelta(minutes=15 * (8 + 1))
    open_change = change(target("sent", attempts=8, sent_at=T0.isoformat()))
    early = adapt_change(open_change, now=deadline - timedelta(seconds=1))
    assert (early.status, early.attention, early.attention_reason) == ("waiting", False, "")
    late = adapt_change(open_change, now=deadline)
    assert (late.status, late.attention, late.attention_reason) == ("waiting", True, "stuck")
    pending = adapt_change(change(target("pending", attempts=0)), now=deadline)
    assert pending.attention_reason == "stuck"  # nobody serves the platform: it never fails
    soon = T0 + timedelta(minutes=2 * (2 + 1))
    custom = adapt_change(open_change, now=soon, resend_minutes=2, max_attempts=2)
    assert custom.attention_reason == "stuck"
    assert adapt_change(change(target("acked")), now=deadline).attention is False
    failed = adapt_change(change(target("failed", last_error="x")), now=deadline)
    assert failed.attention_reason == "failed"


# ------------------------------------------------------------------ action adapter
def action(status="pending", **kw):
    rec = {
        "id": "A-1",
        "guild_id": GUILD,
        "kind": "post_message",
        "args": {"channel": "general", "text": SENTINEL},
        "by": "Adm@Example.com",
        "status": status,
        "created_at": "2026-10-18T16:00:00+07:00",  # the bot's clock is Bangkok time
        "done_at": "",
        "result": {},
        "error": "",
    }
    return {**rec, **kw}


def test_action_mapping_and_error_codes():
    assert [
        action_error_code(e)
        for e in (
            "missing permission",
            "expired before the bot could do it",
            "channel not found: x",
            "discord: 50013",
            "bad request",
            "KeyError",
            "",
        )
    ] == ["missing_permission", "expired", "not_found", "discord_error", "bad_request", "error", ""]
    queued = adapt_action(action())
    assert (queued.kind, queued.id) == ("action", "A-1")
    assert (queued.status, queued.raw_status) == ("queued", "pending")
    assert queued.title == "Discord action · post_message"
    assert queued.workspace == f"discord-{GUILD}"
    assert queued.requester_id == "adm@example.com"
    assert queued.audience == Audience(admin_only=True, people=["adm@example.com"])
    assert queued.started_at == queued.updated_at == "2026-10-18T09:00:00+00:00"  # UTC
    assert (queued.finished_at, queued.error, queued.links) == ("", None, {})
    assert queued.attention is False
    url = "https://discord.com/channels/1000/1/2"
    done = adapt_action(
        action(
            "done",
            done_at="2026-10-18T16:00:05+07:00",
            result={"url": url, "channel": SENTINEL, "delivered_to": SENTINEL},
        )
    )
    assert (done.status, done.links, done.error) == ("done", {"discord": url}, None)
    assert done.updated_at == done.finished_at == "2026-10-18T09:00:05+00:00"
    failed = adapt_action(
        action("failed", done_at="2026-10-18T16:01:00+07:00", error="discord: " + SENTINEL)
    )
    assert (failed.status, failed.attention, failed.attention_reason) == ("failed", True, "failed")
    assert failed.error is not None and failed.error.code == "discord_error"
    for item in (queued, done, failed):
        assert SENTINEL not in "".join(strings(item.model_dump()))
    for bad in ("http://discord.com/x", "https://evil.example/discord.com/", "javascript:x", 5):
        assert adapt_action(action("done", result={"url": bad})).links == {}
    assert adapt_action(action(kind=SENTINEL)).title == "Discord action · other"


# ------------------------------------------------------------------ video adapter
VIDEO_STEPS = ["storyboard", "stills", "hero", "audio", "render"]


def vstep(name, state="done", ms=1000, error_code=""):
    return {"name": name, "state": state, "started_at": "2026-10-18T16:00:00+07:00", "ms": ms,
            "attempts": 1, "error_code": error_code}  # fmt: skip


def video(state="stills", steps=(), reason="", **kw):
    """A `salid_video_jobs` record as `videoad_bot._video_save` writes it (Bangkok times)."""
    rec = {
        "guild_id": GUILD,
        "channel_id": 55,
        "requester_id": 7,
        "state": state,
        "created_at": "2026-10-18T16:00:00+07:00",
        "updated_at": "2026-10-18T16:02:00+07:00",
        "cost_usd": 0.1008,
        "reason": reason,
        "hero_id": SENTINEL,  # an Omni interaction id: never shown
        "steps": list(steps),
    }
    return {**rec, **kw}


def test_adapt_video_mapping():
    running = adapt_video(
        "j1", video("stills", [vstep("storyboard"), vstep("stills", "running", None)])
    )
    assert (running.kind, running.id, running.title) == ("video_ad", "j1", "Video ad")
    assert (running.status, running.raw_status, running.attention) == ("running", "stills", False)
    assert [s.name for s in running.steps] == VIDEO_STEPS  # 5 steps
    assert [s.state for s in running.steps] == ["done", "running", "pending", "pending", "pending"]
    assert running.steps[0].ms == 1000 and running.steps[0].attempts == 1
    assert running.steps[0].started_at == "2026-10-18T09:00:00+00:00"  # UTC
    assert running.steps[2].started_at == "" and running.steps[2].ms is None
    assert running.requester_id == "discord:7" and running.workspace == f"discord-{GUILD}"
    assert running.audience == Audience(people=["discord:7"])
    assert running.cost_usd == 0.1008 and running.error is None and running.org_id == ""
    assert running.links == {"discord": f"https://discord.com/channels/{GUILD}/55"}
    assert running.started_at == "2026-10-18T09:00:00+00:00"
    assert running.updated_at == "2026-10-18T09:02:00+00:00" and running.finished_at == ""
    assert adapt_video("j1", video("queued")).status == "queued"
    assert [s.state for s in adapt_video("j1", video("queued")).steps] == ["pending"] * 5

    done_steps = [vstep(n) for n in ("storyboard", "stills", "audio", "render")]
    done = adapt_video("j1", video("done", done_steps))
    assert (done.status, done.attention, done.error) == ("done", False, None)
    assert {s.name: s.state for s in done.steps}["hero"] == "skipped"
    assert done.finished_at == done.updated_at == "2026-10-18T09:02:00+00:00"

    refused = adapt_video(
        "j1", video("failed", [vstep("render", "failed", 5, "upload_refused")], "upload_refused")
    )
    assert (refused.status, refused.attention, refused.attention_reason) == (
        "failed",
        True,
        "failed",
    )
    assert refused.error is not None and refused.error.code == "upload_refused"
    assert {s.name: s.error_code for s in refused.steps}["render"] == "upload_refused"
    assert [s.state for s in refused.steps] == ["skipped"] * 4 + ["failed"]

    returned = adapt_video("j1", video("returned", reason=""))
    assert (returned.status, returned.attention) == ("failed", True)
    assert returned.error is not None and returned.error.code == "unknown"
    assert adapt_video("j1", video("returned", reason="cancelled")).status == "cancelled"
    cancelled = adapt_video("j1", video("cancelled", [vstep("stills", "cancelled", 3)]))
    assert (cancelled.status, cancelled.attention, cancelled.error) == ("cancelled", False, None)
    assert {s.name: s.state for s in cancelled.steps}["stills"] == "cancelled"
    restarted = adapt_video("j1", video("failed", reason="restarted"))
    assert restarted.error.code == "restarted"


def test_adapt_video_never_echoes_stored_text():
    rec = video(
        SENTINEL,
        [vstep("stills", SENTINEL, error_code=SENTINEL), {"name": SENTINEL}, SENTINEL],
        SENTINEL,
        brief=SENTINEL,
        storyboard={"title": SENTINEL},
        channel_id=SENTINEL,
    )
    it = adapt_video("j1", rec)
    assert (it.status, it.raw_status, it.error.code) == ("failed", "unknown", "unknown")
    assert [s.name for s in it.steps] == VIDEO_STEPS
    assert {s.name: (s.state, s.error_code) for s in it.steps}["stills"] == ("pending", "unknown")
    assert it.links == {}  # no usable channel id
    assert SENTINEL not in "".join(strings(it.model_dump()))
    bare = adapt_video("j2", {"state": "render"})
    assert (bare.status, bare.requester_id, bare.audience, bare.links) == (
        "running",
        "",
        Audience(),
        {},
    )
    assert (bare.workspace, bare.cost_usd, bare.started_at) == ("", None, "")


# ------------------------------------------------------------------ sources and finders
def org(org_id="o1", *workspaces):
    return Org(id=org_id, name=org_id, workspaces=list(workspaces))


def test_sources_are_registered_on_import():
    assert monitor_sources.SOURCES["change"] == (changes_source, changes_finder)
    assert monitor_sources.SOURCES["action"] == (actions_source, actions_finder)
    assert monitor_sources.SOURCES["video_ad"] == (video_source, video_finder)


def test_change_source_and_finder_read_only_the_orgs_changes(docs, monkeypatch):
    mine = change(target("sent"), change_id="c1", org_id="o1")
    theirs = change(target("sent"), change_id="c2", org_id="o2")
    for c in (mine, theirs):
        docs.put(OUTBOX, org_guild(c.org_id), c.id, c.model_dump())
    o1 = org("o1")
    items = changes_source(docs, o1, T0 + timedelta(minutes=1))
    assert [(it.id, it.org_id, it.status) for it in items] == [("c1", "o1", "waiting")]
    monkeypatch.setattr(monitor_sources, "wall_clock", lambda: T0 + timedelta(minutes=1))
    found = changes_finder(docs, o1, "c1")
    assert found is not None and (found.id, found.org_id, found.attention) == ("c1", "o1", False)
    monkeypatch.setattr(monitor_sources, "wall_clock", lambda: T0 + timedelta(days=1))
    assert changes_finder(docs, o1, "c1").attention_reason == "stuck"  # the finder's clock
    assert changes_finder(docs, o1, "c2") is None  # another org's change
    assert changes_finder(docs, o1, "nope") is None
    assert changes_finder(docs, org(""), "c1") is None


def test_action_source_and_finder_read_the_orgs_discord_workspaces(docs):
    clock = lambda: T0  # noqa: E731
    q = ActionQueue(docs, clock=clock)
    a1 = q.enqueue(GUILD, "post_message", {"text": SENTINEL}, by="adm@example.com")
    a2 = q.enqueue(2000, "announce", {"text": SENTINEL}, by="adm@example.com")
    a3 = q.enqueue(3000, "announce", {"text": SENTINEL}, by="adm@example.com")  # another org
    q.finish(2000, a2["id"], error="missing permission")
    o1 = org(
        "o1",
        Workspace(id=f"discord-{GUILD}", platform="discord", external_id=str(GUILD)),
        Workspace(id="discord-2000", platform="discord", external_id="2000"),
        Workspace(id="discord-3000", platform="discord", external_id="3000", status="disconnected"),
        Workspace(id="slack-T1", platform="slack", external_id="T1"),
    )
    items = {it.id: it for it in actions_source(docs, o1, T0)}
    assert set(items) == {a1["id"], a2["id"]}
    assert {it.org_id for it in items.values()} == {"o1"}
    assert items[a2["id"]].status == "failed" and items[a2["id"]].workspace == "discord-2000"
    assert items[a2["id"]].error.code == "missing_permission"
    found = actions_finder(docs, o1, a2["id"])
    assert found is not None and (found.id, found.org_id) == (a2["id"], "o1")
    assert actions_finder(docs, o1, a3["id"]) is None  # a disconnected server
    assert actions_finder(docs, org("o9"), a1["id"]) is None  # an org without that server
    assert actions_finder(docs, o1, "A-nope") is None


def test_video_source_and_finder_read_the_orgs_discord_workspaces(docs):
    docs.put(monitor_sources.VIDEO_JOBS, GUILD, "j1", video("stills"))
    docs.put(monitor_sources.VIDEO_JOBS, 2000, "j2", video("done", guild_id=2000))
    docs.put(monitor_sources.VIDEO_JOBS, 3000, "j3", video("render", guild_id=3000))  # other org
    docs.put(monitor_sources.VIDEO_JOBS, 0, "_active", {"j1": GUILD})  # the bot's index
    o1 = org(
        "o1",
        Workspace(id=f"discord-{GUILD}", platform="discord", external_id=str(GUILD)),
        Workspace(id="discord-2000", platform="discord", external_id="2000"),
        Workspace(id="discord-3000", platform="discord", external_id="3000", status="disconnected"),
    )
    items = {it.id: it for it in video_source(docs, o1, T0)}
    assert set(items) == {"j1", "j2"}
    assert {it.org_id for it in items.values()} == {"o1"}
    assert (items["j1"].status, items["j2"].status) == ("running", "done")
    assert items["j2"].workspace == "discord-2000"
    found = video_finder(docs, o1, "j2")
    assert found is not None and (found.id, found.org_id, found.status) == ("j2", "o1", "done")
    assert video_finder(docs, o1, "j3") is None  # a disconnected server
    assert video_finder(docs, org("o9"), "j1") is None  # an org without that server
    assert video_finder(docs, o1, "_active") is None and video_finder(docs, o1, "nope") is None
