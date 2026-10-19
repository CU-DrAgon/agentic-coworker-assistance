# ruff: noqa: F811, F401
"""Deep task and deep research job records (`salid.jobs.JobBook`: DocStore `salid_jobs` + the
`_active` index), the restart sweep, the bot wiring (`salid.jobs_bot.JobsMixin`, `_run_deep`,
`_run_research`), the monitor hook and the `deep_task` / `research` adapter and sources. Offline:
SQLite DocStore / fake Firestore, fake Discord objects, scripted clients."""

from __future__ import annotations

import asyncio
import gc
import json
import threading
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import discord
import pytest
from conftest import GUILD
from fake_firestore import FakeFirestore
from test_bot_multitenant import Msg
from test_coordinator import PEER, PUBLIC, STAFF, world
from test_modes import bot  # noqa: F401  (fixture)
from test_search import Dispatch, finish, start

from salid import bot as botmod
from salid import jobs, jobs_bot, monitor_sources
from salid.coo_store import FirestoreDocStore, SQLiteDocStore
from salid.deep_task import DeepCancelled, DeepHandle, DeepResult, DeepTaskError
from salid.jobs import FINAL, INDEX_GUILD, INDEX_KEY, KIND, KINDS, JobBook, new_id
from salid.monitor import ERROR_TEXT, Audience, read_log
from salid.monitor_sources import adapt_job
from salid.orgs import Org, Workspace
from salid.search_bot import _ResearchJob

BKK = ZoneInfo("Asia/Bangkok")
T0 = datetime(2026, 10, 18, 16, 0, tzinfo=BKK)  # 09:00 UTC
SENTINEL = "SENSITIVE-7f3a"


@pytest.fixture(params=["sqlite", "firestore"])
def docs(request):
    if request.param == "sqlite":
        return SQLiteDocStore(":memory:")
    return FirestoreDocStore(client=FakeFirestore())


def book(docs, seen=None):
    return JobBook(docs, lambda: T0, on_change=None if seen is None else seen.append)


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


# ------------------------------------------------------------------ JobBook
def test_constants_and_ids():
    assert KIND == "salid_jobs" and (INDEX_GUILD, INDEX_KEY) == (0, "_active")
    assert KINDS == ("deep_task", "research")
    assert set(FINAL) == {"done", "failed", "cancelled", "restarted"}
    a, b = new_id(), new_id()
    assert len(a) == 12 and int(a, 16) >= 0 and a != b
    assert not set(jobs.REASONS) - {""} - set(ERROR_TEXT)  # every stored reason has a safe text


def test_job_lifecycle_and_index(docs):
    seen: list[dict] = []
    b = book(docs, seen)
    assert b.start("deep_task", job_id="j1", guild_id=GUILD, channel_id=55, requester_id=7)
    assert b.active() == {"j1": GUILD}
    rec = docs.get(KIND, GUILD, "j1")
    at = T0.isoformat(timespec="seconds")
    expected = {
        "job_id": "j1", "kind": "deep_task", "guild_id": GUILD, "channel_id": 55,
        "requester_id": 7, "dm": False, "state": "running", "created_at": at, "updated_at": at,
        "finished_at": "", "reason": "",
    }  # fmt: skip
    assert rec == expected and seen == [expected]
    assert b.finish(GUILD, "j1", "done")
    assert b.active() == {}
    rec = docs.get(KIND, GUILD, "j1")
    assert (rec["state"], rec["reason"], rec["finished_at"]) == ("done", "", at)
    assert [r["state"] for r in seen] == ["running", "done"]
    assert b.start("research", job_id="j2", guild_id=GUILD, channel_id=1, requester_id=2, dm=True)
    assert docs.get(KIND, GUILD, "j2")["dm"] is True
    assert b.finish(GUILD, "j2", "failed", "error")
    assert docs.get(KIND, GUILD, "j2")["reason"] == "error"
    assert b.active() == {}


def test_bad_kinds_states_and_reasons_store_nothing_odd(docs):
    b = book(docs)
    assert not b.start("video_ad", job_id="x", guild_id=GUILD, channel_id=1, requester_id=2)
    assert not b.start("research", job_id="", guild_id=GUILD, channel_id=1, requester_id=2)
    assert docs.find(KIND, GUILD) == [] and b.active() == {}
    assert b.start("research", job_id="j1", guild_id=GUILD, channel_id=1, requester_id=2)
    assert not b.finish(GUILD, "j1", "running")  # not a final state
    assert docs.get(KIND, GUILD, "j1")["state"] == "running" and b.active() == {"j1": GUILD}
    assert b.finish(GUILD, "j1", "failed", f"❌ {SENTINEL}")  # never free text
    assert docs.get(KIND, GUILD, "j1")["reason"] == "error"
    assert not b.finish(GUILD, "nope", "done")  # no record: nothing written, not indexed


def test_finish_without_a_record_only_leaves_the_index(docs):
    b = book(docs)
    assert b.start("deep_task", job_id="j1", guild_id=GUILD, channel_id=1, requester_id=2)
    docs.delete(KIND, GUILD, "j1")  # the record write never happened
    assert not b.finish(GUILD, "j1", "done")
    assert b.active() == {} and docs.get(KIND, GUILD, "j1") is None


def test_sweep_marks_cut_off_jobs_restarted_and_skips_live(docs):
    seen: list[dict] = []
    b = book(docs, seen)
    for key in ("j1", "j2", "j3", "j4"):
        assert b.start("deep_task", job_id=key, guild_id=GUILD, channel_id=1, requester_id=2)
    docs.put(KIND, GUILD, "j3", {**docs.get(KIND, GUILD, "j3"), "state": "done"})  # ended
    docs.delete(KIND, GUILD, "j4")  # never written
    seen.clear()
    swept = b.sweep(lambda job_id: job_id == "j2")
    at = T0.isoformat(timespec="seconds")
    assert [(r["job_id"], r["state"], r["reason"], r["finished_at"]) for r in swept] == [
        ("j1", "restarted", "restarted", at)
    ]
    assert seen == swept  # the monitor hears about it
    assert docs.get(KIND, GUILD, "j1")["state"] == "restarted"
    assert docs.get(KIND, GUILD, "j2")["state"] == "running"  # live: left alone
    assert docs.get(KIND, GUILD, "j3")["state"] == "done"
    assert b.active() == {"j2": GUILD}  # ended or missing ones just leave the index
    assert b.sweep(lambda job_id: False)[0]["job_id"] == "j2"  # a later sweep, not live any more


class Broken:
    """A store whose every call fails."""

    def __getattr__(self, name):
        def fail(*a, **k):
            raise RuntimeError(SENTINEL)

        return fail


def test_job_book_never_raises_on_store_errors(caplog):
    b = JobBook(Broken(), lambda: T0, on_change=lambda rec: 1 / 0)
    assert not b.start("deep_task", job_id="j1", guild_id=GUILD, channel_id=1, requester_id=2)
    assert not b.finish(GUILD, "j1", "done")
    assert b.active() == {} and b.sweep(lambda job_id: False) == []
    assert SENTINEL not in caplog.text and "RuntimeError" in caplog.text


def test_a_bad_index_entry_is_skipped_not_fatal(docs, caplog):
    b = book(docs)
    docs.put(KIND, INDEX_GUILD, INDEX_KEY, {"bad": SENTINEL, "none": None, "ok": GUILD})
    assert b.active() == {"ok": GUILD}
    assert b.start("deep_task", job_id="j1", guild_id=GUILD, channel_id=1, requester_id=2)
    assert b.active() == {"ok": GUILD, "j1": GUILD}  # the write drops the bad entries
    assert [r["job_id"] for r in b.sweep(lambda job_id: job_id == "j1")] == []
    assert b.active() == {"j1": GUILD}  # "ok" had no record: it just left the index
    assert "ValueError" in caplog.text and SENTINEL not in caplog.text


def test_a_failing_hook_or_record_write_is_contained(docs, caplog):
    b = JobBook(docs, lambda: T0, on_change=lambda rec: 1 / 0)
    assert b.start("deep_task", job_id="j1", guild_id=GUILD, channel_id=1, requester_id=2)
    assert b.finish(GUILD, "j1", "done")  # the hook raised: the records are still written
    real_put = docs.put

    def put(kind, gid, key, doc):
        if key != INDEX_KEY:
            raise RuntimeError(SENTINEL)
        real_put(kind, gid, key, doc)

    docs.put = put
    b = book(docs)
    assert not b.start("deep_task", job_id="j2", guild_id=GUILD, channel_id=1, requester_id=2)
    assert b.active() == {"j2": GUILD}  # the next start's sweep drops the entry
    assert b.sweep(lambda job_id: False) == [] and b.active() == {}
    assert SENTINEL not in caplog.text


def test_a_failed_final_write_leaves_the_job_for_the_sweep(docs):
    b = book(docs)
    assert b.start("research", job_id="j1", guild_id=GUILD, channel_id=1, requester_id=2)
    real_put = docs.put
    docs.put = lambda kind, gid, key, doc: (_ for _ in ()).throw(RuntimeError("down"))
    assert not b.finish(GUILD, "j1", "done")
    docs.put = real_put
    assert b.active() == {"j1": GUILD}  # still listed: a restart marks it restarted
    assert [r["state"] for r in b.sweep(lambda job_id: False)] == ["restarted"]


# ------------------------------------------------------------------ the bot: deep tasks
def deep_msg(world, text="งานใหญ่: วิเคราะห์", channel=None):
    return Msg(world.g, channel or world.pub, world.staff, text)


async def settle(bot):
    """Every deep task has ended and every end record written after a slow start record."""
    await asyncio.gather(*list(bot._deep_tasks), return_exceptions=True)
    await asyncio.gather(*list(bot._job_tails), return_exceptions=True)
    await asyncio.sleep(0)


def records(bot):
    return bot.docs.find(KIND, GUILD)


@pytest.mark.parametrize(
    "outcome, expected",
    [
        ("done", ("done", "")),
        ("deep_error", ("failed", "failed")),
        ("crash", ("failed", "error")),
        ("cli_cancel", ("cancelled", "")),
    ],
)
async def test_deep_task_outcomes_are_recorded(bot, world, monkeypatch, outcome, expected):
    def fake(s, store, task, **kw):
        if outcome == "deep_error":
            raise DeepTaskError(f"bad {SENTINEL}")
        if outcome == "crash":
            raise RuntimeError(SENTINEL)
        if outcome == "cli_cancel":
            raise DeepCancelled()
        return DeepResult(text=f"report {SENTINEL}")

    monkeypatch.setattr(botmod, "run_deep_task", fake)
    msg = deep_msg(world)
    bot._launch_deep(msg, GUILD, f"วิเคราะห์ {SENTINEL}", "th")
    job_id = bot._deep_jobs[GUILD].job_id
    assert len(job_id) == 12
    await settle(bot)
    (rec,) = records(bot)
    assert (rec["job_id"], rec["kind"], rec["requester_id"], rec["channel_id"]) == (
        job_id,
        "deep_task",
        STAFF,
        PUBLIC,
    )
    assert (rec["state"], rec["reason"], rec["dm"]) == (*expected, False)
    assert bot.jobs.active() == {}


async def test_records_never_hold_the_task_text(bot, world, monkeypatch):
    monkeypatch.setattr(
        botmod, "run_deep_task", lambda s, store, task, **kw: DeepResult(text=f"{task} done")
    )
    bot._launch_deep(deep_msg(world, f"งานใหญ่: {SENTINEL}"), GUILD, f"ทำ {SENTINEL}", "th")
    await settle(bot)
    assert [r["state"] for r in records(bot)] == ["done"]
    assert SENTINEL not in json.dumps(bot.docs.find("salid_jobs", GUILD))
    assert SENTINEL not in json.dumps(bot.docs.find("salid_jobs", INDEX_GUILD))


async def test_cancel_deep_task_records_cancelled(bot, world, monkeypatch):
    gate = threading.Event()

    def slow(s, store, task, **kw):
        gate.wait(5)
        return DeepResult(text="late")

    monkeypatch.setattr(botmod, "run_deep_task", slow)
    msg = deep_msg(world)
    bot._launch_deep(msg, GUILD, "วิเคราะห์", "th")
    for _ in range(100):  # the record is written before the CLI runs
        if records(bot):
            break
        await asyncio.sleep(0.01)
    assert [r["state"] for r in records(bot)] == ["running"]
    await bot._cancel_deep(GUILD, world.staff, "th")
    await settle(bot)
    gate.set()
    assert [(r["state"], r["reason"]) for r in records(bot)] == [("cancelled", "")]
    assert msg.replies == [] and bot.jobs.active() == {}


class Gate:
    """`bot.jobs.start` held until `release()` (a slow store), then the real one."""

    def __init__(self, bot):
        self.event, self.real = threading.Event(), bot.jobs.start

    def __call__(self, *a, **kw):
        self.event.wait(5)
        return self.real(*a, **kw)

    def release(self):
        self.event.set()


async def until(check, tries=200):
    for _ in range(tries):
        if check():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out")


async def test_a_cancel_while_the_record_is_written_still_ends_it(bot, world, monkeypatch):
    """The user cancels while the start record is still being written: the cancel does not
    wait for it, and the end record is written after it, so nothing is left running."""
    gate = Gate(bot)
    monkeypatch.setattr(bot.jobs, "start", gate)
    monkeypatch.setattr(botmod, "run_deep_task", lambda *a, **k: DeepResult(text="x"))
    bot._launch_deep(deep_msg(world), GUILD, "วิเคราะห์", "th")
    await asyncio.sleep(0.02)
    await bot._cancel_deep(GUILD, world.staff, "th")
    await asyncio.gather(*list(bot._deep_tasks), return_exceptions=True)
    assert not bot._deep_tasks and records(bot) == []  # the job ended; its record still waits
    gate.release()
    await settle(bot)
    assert [(r["state"], r["reason"]) for r in records(bot)] == [("cancelled", "")]
    assert bot.jobs.active() == {} and bot._job_writes == {}


async def test_a_slow_store_never_holds_the_deep_task_back(bot, world, monkeypatch):
    monkeypatch.setattr(jobs_bot, "START_WAIT_S", 0.05)
    gate = Gate(bot)
    monkeypatch.setattr(bot.jobs, "start", gate)
    monkeypatch.setattr(botmod, "run_deep_task", lambda *a, **k: DeepResult(text="report"))
    msg = deep_msg(world)
    bot._launch_deep(msg, GUILD, "วิเคราะห์", "th")
    job_id = bot._deep_jobs[GUILD].job_id
    await until(lambda: msg.replies)  # posted while the start record is still being written
    assert "report" in msg.replies[0] and records(bot) == []
    await asyncio.gather(*list(bot._deep_tasks), return_exceptions=True)
    assert bot._job_live(job_id)  # its records are still being written: the sweep skips it
    gate.release()
    await settle(bot)
    assert [(r["state"], r["reason"]) for r in records(bot)] == [("done", "")]  # start, then end
    assert bot.jobs.active() == {} and not bot._job_live(job_id)


@pytest.mark.parametrize("store", ["raising", "broken"])
async def test_a_failing_store_leaves_the_deep_task_result_unchanged(
    bot, world, monkeypatch, caplog, store
):
    if store == "raising":
        monkeypatch.setattr(bot.jobs, "start", lambda *a, **k: 1 / 0)
    else:
        bot.jobs = JobBook(Broken(), bot.s.now, on_change=bot._monitor_job)
    monkeypatch.setattr(botmod, "run_deep_task", lambda *a, **k: DeepResult(text="report"))
    msg = deep_msg(world)
    bot._launch_deep(msg, GUILD, "วิเคราะห์", "th")
    await settle(bot)
    assert len(msg.replies) == 1 and "report" in msg.replies[0]
    assert records(bot) == [] and bot._job_writes == {}
    gc.collect()
    await asyncio.sleep(0)
    assert "never retrieved" not in caplog.text and SENTINEL not in caplog.text


async def test_a_shutdown_leaves_the_deep_task_for_the_sweep(bot, world, monkeypatch):
    gate = threading.Event()

    def slow(s, store, task, **kw):
        gate.wait(5)
        return DeepResult(text="late")

    monkeypatch.setattr(botmod, "run_deep_task", slow)
    msg = deep_msg(world)
    bot._launch_deep(msg, GUILD, "วิเคราะห์", "th")
    job = bot._deep_jobs[GUILD]
    await until(lambda: records(bot))
    job.task.cancel()  # cancelled without the user's flag: the bot is stopping
    await settle(bot)
    gate.set()
    assert [r["state"] for r in records(bot)] == ["running"]
    assert bot.jobs.active() == {job.job_id: GUILD} and msg.replies == []
    assert job.handle.cancelled  # the CLI process is still stopped
    await bot._sweep_jobs()  # the next start
    assert [(r["state"], r["reason"]) for r in records(bot)] == [("restarted", "restarted")]
    assert bot.jobs.active() == {}


async def test_a_job_asked_for_in_private_is_dm(bot, world, monkeypatch):
    monkeypatch.setattr(botmod, "run_deep_task", lambda *a, **k: DeepResult(text="x"))
    bot._pc_owners[world.pub.id] = STAFF  # the requester's private Salid channel
    bot._launch_deep(deep_msg(world), GUILD, "วิเคราะห์", "th")
    await settle(bot)
    dm_msg = Msg(None, world.pub, world.staff, "งานใหญ่: x")  # a DM resolved to the guild
    bot._launch_deep(dm_msg, GUILD, "วิเคราะห์", "th")
    await settle(bot)
    assert [r["dm"] for r in records(bot)] == [True, True]


async def test_job_records_emit_monitor_events_only_while_enabled(bot, world, monkeypatch):
    monkeypatch.setattr(botmod, "run_deep_task", lambda *a, **k: DeepResult(text="x"))
    bot._launch_deep(deep_msg(world), GUILD, "วิเคราะห์", "th")
    await settle(bot)
    await bot._monitor_due()
    assert read_log(bot.docs, f"guild:{GUILD}")["events"] == []  # the flag is off
    bot.monitor.enabled = True
    bot._launch_deep(deep_msg(world), GUILD, "วิเคราะห์", "th")
    await settle(bot)
    await bot._monitor_due()
    events = read_log(bot.docs, f"guild:{GUILD}")["events"]
    assert [(e["kind"], e["status"]) for e in events] == [
        ("deep_task", "running"),
        ("deep_task", "done"),
    ]
    assert (
        events[0]["audience"] == Audience(exec_only=True, people=[f"discord:{STAFF}"]).model_dump()
    )
    bot._monitor_job({"guild_id": "junk"})  # never raises


def test_job_live_sees_unfinished_deep_and_research_jobs(bot):
    class Task:
        def __init__(self, done):
            self._done = done

        def done(self):
            return self._done

    bot._deep_jobs[GUILD] = botmod._DeepJob(Task(False), DeepHandle(), STAFF, "d1")
    bot._research_jobs[GUILD] = _ResearchJob(Task(False), None, STAFF, "r1")
    bot._research_jobs[GUILD + 1] = _ResearchJob(Task(True), None, STAFF, "r2")
    assert bot._job_live("d1") and bot._job_live("r1")
    assert not bot._job_live("r2") and not bot._job_live("nope") and not bot._job_live("")
    assert botmod._DeepJob(Task(False), DeepHandle(), STAFF).job_id == ""  # old call shape


async def test_the_sweep_runs_once_off_the_event_loop(bot, world):
    bot.monitor.enabled = True
    assert bot.jobs.start("deep_task", job_id="cut", guild_id=GUILD, channel_id=1, requester_id=2)
    assert bot.jobs.start("research", job_id="live", guild_id=GUILD, channel_id=1, requester_id=2)
    bot._research_jobs[GUILD] = _ResearchJob(
        asyncio.get_running_loop().create_future(), None, 2, "live"
    )
    threads = []
    real_get = bot.docs.get

    def get(kind, gid, key):
        threads.append(threading.current_thread() is threading.main_thread())
        return real_get(kind, gid, key)

    bot.docs.get = get
    await bot._sweep_jobs()
    assert threads and not any(threads)  # every store read in a worker thread
    state = {r["job_id"]: r["state"] for r in records(bot)}
    assert state == {"cut": "restarted", "live": "running"}
    assert bot.jobs.active() == {"live": GUILD}
    await bot._monitor_due()
    events = read_log(bot.docs, f"guild:{GUILD}")["events"]
    assert (events[-1]["id"], events[-1]["status"], events[-1]["attention"]) == (
        "cut",
        "failed",
        True,
    )
    assert bot.jobs.start("deep_task", job_id="later", guild_id=GUILD, channel_id=1, requester_id=2)
    await bot._sweep_jobs()  # on_ready again after a reconnect: no second sweep
    assert {r["job_id"]: r["state"] for r in records(bot)}["later"] == "running"


async def test_the_sweep_never_raises(bot, monkeypatch, caplog):
    monkeypatch.setattr(bot.jobs, "sweep", lambda is_live: 1 / 0)
    await bot._sweep_jobs()  # logged, not raised
    assert "ZeroDivisionError" in caplog.text


async def test_on_ready_sweeps_left_over_jobs(bot, world, monkeypatch):
    called = []

    async def sweep():
        called.append(1)

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(bot, "_sweep_jobs", sweep)
    monkeypatch.setattr(bot, "_sweep_video_jobs", noop)
    monkeypatch.setattr(bot, "_sync_guild", noop)
    monkeypatch.setattr(bot, "_pc_backfill_start", lambda g: None)
    monkeypatch.setattr(bot, "_start_import", lambda: None)
    bot.meetings = SimpleNamespace(resume_all=noop, scan_all=lambda: None)
    await bot.on_ready()
    assert called == [1]


# ------------------------------------------------------------------ the bot: deep research
async def test_research_job_is_recorded_and_ends_done(bot, world):
    bot.llm = Dispatch()
    fns, _out = await start(bot, world)
    res = await asyncio.to_thread(fns["deep_research"], f"เปรียบเทียบ POS {SENTINEL}")
    assert res["status"] == "started"
    job_id = bot._research_jobs[GUILD].job_id
    await finish(bot)
    assert [r["state"] for r in bot.docs.find("salid_jobs", GUILD)] == ["done"]
    (rec,) = records(bot)
    assert (rec["job_id"], rec["kind"], rec["requester_id"], rec["channel_id"]) == (
        job_id,
        "research",
        STAFF,
        PUBLIC,
    )
    assert rec["dm"] is False and bot.jobs.active() == {}
    assert SENTINEL not in json.dumps(bot.docs.find("salid_jobs", GUILD))


async def test_cancelled_research_ends_cancelled(bot, world):
    gate = threading.Event()
    client = Dispatch()
    orig = client.generate_content

    def slow(model, contents, config):
        if config.tools:
            gate.wait(5)
        return orig(model, contents, config)

    client.models = SimpleNamespace(generate_content=slow)
    bot.llm = client
    fns, out = await start(bot, world)
    await asyncio.to_thread(fns["deep_research"], "POS")
    await asyncio.sleep(0.05)
    await bot._cancel_deep(GUILD, world.g.members[STAFF], "th")
    gate.set()
    await finish(bot)
    assert out.sent == []
    assert [(r["state"], r["reason"]) for r in records(bot)] == [("cancelled", "")]
    assert bot.jobs.active() == {}


async def test_crashed_research_ends_failed(bot, world, monkeypatch):
    monkeypatch.setattr(
        "salid.search_bot.run_research", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    )
    bot.llm = Dispatch()
    fns, out = await start(bot, world)
    await asyncio.to_thread(fns["deep_research"], "POS")
    await finish(bot)
    assert [(r["state"], r["reason"]) for r in records(bot)] == [("failed", "error")]


async def test_a_slow_or_failing_store_never_holds_the_research_back(bot, world, monkeypatch):
    monkeypatch.setattr(jobs_bot, "START_WAIT_S", 0.05)
    gate = Gate(bot)
    monkeypatch.setattr(bot.jobs, "start", gate)
    bot.llm = Dispatch()
    fns, out = await start(bot, world)
    await asyncio.to_thread(fns["deep_research"], "POS")
    await until(lambda: out.sent)  # the report is posted while the start record waits
    assert out.sent[0][1]["file"].filename == "research.pdf" and records(bot) == []
    gate.release()
    await finish(bot)
    await asyncio.gather(*list(bot._job_tails), return_exceptions=True)
    assert [(r["state"], r["reason"]) for r in records(bot)] == [("done", "")]
    monkeypatch.setattr(bot.jobs, "start", lambda *a, **k: 1 / 0)
    await asyncio.to_thread(fns["deep_research"], "POS again")
    await finish(bot)
    assert len(out.sent) == 2  # the second report too, with no start record at all


async def test_a_shutdown_leaves_the_research_for_the_sweep(bot, world):
    gate = threading.Event()
    client = Dispatch()
    orig = client.generate_content

    def slow(model, contents, config):
        if config.tools:
            gate.wait(5)
        return orig(model, contents, config)

    client.models = SimpleNamespace(generate_content=slow)
    bot.llm = client
    fns, out = await start(bot, world)
    await asyncio.to_thread(fns["deep_research"], "POS")
    job = bot._research_jobs[GUILD]
    await until(lambda: records(bot))
    job.task.cancel()  # no user cancel: a shutdown
    await finish(bot)
    gate.set()
    assert [r["state"] for r in records(bot)] == ["running"] and out.sent == []
    assert bot.jobs.active() == {job.job_id: GUILD}
    await bot._sweep_jobs()
    assert [(r["state"], r["reason"]) for r in records(bot)] == [("restarted", "restarted")]


async def test_a_research_report_that_could_not_be_posted_is_failed(bot, world):
    bot.llm = Dispatch()
    fns, out = await start(bot, world)

    async def refuse(text=None, **kw):
        raise discord.HTTPException(SimpleNamespace(status=500, reason="x"), "boom")

    out.send = refuse
    await asyncio.to_thread(fns["deep_research"], "POS")
    await finish(bot)
    assert [(r["state"], r["reason"]) for r in records(bot)] == [("failed", "error")]


# ------------------------------------------------------------------ adapter and sources
def job(kind="deep_task", state="running", **extra):
    at = T0.isoformat(timespec="seconds")
    return {"job_id": "j1", "kind": kind, "guild_id": GUILD, "channel_id": 55,
            "requester_id": 7, "dm": False, "state": state, "created_at": at,
            "updated_at": at, "finished_at": at if state in FINAL else "", "reason": "",
            **extra}  # fmt: skip


def test_adapt_job_mapping():
    deep = adapt_job(job())
    assert (deep.kind, deep.id, deep.title, deep.status) == (
        "deep_task",
        "j1",
        "Deep task",
        "running",
    )
    assert (deep.raw_status, deep.attention, deep.error, deep.org_id) == (
        "running",
        False,
        None,
        "",
    )
    assert deep.requester_id == "discord:7" and deep.workspace == f"discord-{GUILD}"
    assert deep.audience == Audience(exec_only=True, people=["discord:7"])
    assert deep.started_at == deep.updated_at == "2026-10-18T09:00:00+00:00"
    assert deep.finished_at == "" and deep.steps == []
    assert deep.links == {"discord": f"https://discord.com/channels/{GUILD}/55"}
    research = adapt_job(job("research", "done"))
    assert (research.title, research.status, research.error) == ("Deep research", "done", None)
    assert research.audience == Audience(people=["discord:7"])
    assert research.finished_at == "2026-10-18T09:00:00+00:00"
    failed = adapt_job(job(state="failed", reason="failed"))
    assert (failed.status, failed.attention, failed.attention_reason) == ("failed", True, "failed")
    assert failed.error.code == "failed"
    assert adapt_job(job(state="failed", reason="error")).error.code == "error"
    restarted = adapt_job(job("research", "restarted", reason="restarted"))
    assert (restarted.status, restarted.raw_status, restarted.attention) == (
        "failed",
        "restarted",
        True,
    )
    assert restarted.error.code == "restarted"
    cancelled = adapt_job(job(state="cancelled"))
    assert (cancelled.status, cancelled.attention, cancelled.error) == ("cancelled", False, None)
    private = adapt_job(job(dm=True))
    assert private.audience == Audience(exec_only=True, dm=True, people=["discord:7"])
    assert private.links == {}  # never a link that names the private channel
    odd = adapt_job(job(state="weird", requester_id=0))
    assert (odd.status, odd.raw_status, odd.requester_id) == ("failed", "unknown", "")
    assert odd.audience.people == []


def test_adapt_job_never_echoes_stored_text():
    rec = job(state=SENTINEL, reason=SENTINEL, task=SENTINEL, question=SENTINEL)
    rec["job_id"] = "j1"
    item = adapt_job(rec).model_dump()
    assert SENTINEL not in "".join(strings(item))


def test_job_sources_are_registered_and_read_the_orgs_discord_workspaces(docs):
    assert set(KINDS) <= set(monitor_sources.SOURCES)
    b = book(docs)
    assert b.start("deep_task", job_id="d1", guild_id=GUILD, channel_id=1, requester_id=7)
    assert b.start("research", job_id="r1", guild_id=2000, channel_id=1, requester_id=7)
    assert b.start("research", job_id="r9", guild_id=3000, channel_id=1, requester_id=7)
    o1 = Org(
        id="o1",
        name="o1",
        workspaces=[
            Workspace(id=f"discord-{GUILD}", platform="discord", external_id=str(GUILD)),
            Workspace(id="discord-2000", platform="discord", external_id="2000"),
            Workspace(
                id="discord-3000", platform="discord", external_id="3000", status="disconnected"
            ),
        ],
    )
    deep_source, deep_finder = monitor_sources.SOURCES["deep_task"]
    research_source, research_finder = monitor_sources.SOURCES["research"]
    assert [(it.id, it.org_id, it.kind) for it in deep_source(docs, o1, T0)] == [
        ("d1", "o1", "deep_task")
    ]
    assert [(it.id, it.org_id) for it in research_source(docs, o1, T0)] == [("r1", "o1")]
    found = research_finder(docs, o1, "r1")
    assert found is not None and (found.id, found.org_id, found.workspace) == (
        "r1",
        "o1",
        "discord-2000",
    )
    assert deep_finder(docs, o1, "r1") is None  # another kind
    assert research_finder(docs, o1, "r9") is None  # a disconnected server
    assert deep_finder(docs, o1, INDEX_KEY) is None and deep_finder(docs, o1, "nope") is None
    assert research_finder(docs, Org(id="o9", name="o9"), "r1") is None  # not its server


def test_jobs_module_never_imports_discord():
    from pathlib import Path

    text = Path(jobs.__file__).read_text(encoding="utf-8")
    assert "import discord" not in text and "from discord" not in text
