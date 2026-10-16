# ruff: noqa: F811, F401
"""Deleting Salid's own messages by reply, and cancelling a running deep task (fakes only)."""

import asyncio
import subprocess
import sys
import threading
from functools import partial
from types import SimpleNamespace

import pytest
from conftest import ALICE, GUILD
from test_bot_multitenant import ME, member
from test_modes import EXEC, MANAGER, STAFF, TMsg, bot, world

from salid import bot as botmod
from salid.deep_task import DeepCancelled, DeepHandle, _run
from salid.message_control import AnswerLog, parse_cancel_task, parse_delete_message
from salid.models import Receipt
from salid.qa import Answer


@pytest.mark.parametrize(
    "text, kind",
    [("ลบ", "plain"), ("delete", "plain"), ("Delete!", "plain"),
     ("ลบข้อความ", "message"), ("ยกเลิกข้อความ", "message"), ("เอาออก", "message"),
     ("remove", "message"), ("unsend", "message"), ("delete message", "message"),
     ("ลบข้อความนี้ทั้งหมดเลยนะ", None), ("please delete the file", None), ("", None)],
)  # fmt: skip
def test_parse_delete_message(text, kind):
    assert parse_delete_message(text) == kind


@pytest.mark.parametrize(
    "text", ["ยกเลิกงาน", "ยกเลิกงานใหญ่", "cancel task", "Cancel the task", "stop task"]
)
def test_parse_cancel_task(text):
    assert parse_cancel_task(text)
    assert not parse_cancel_task("cancel my subscription")


def test_answer_log_expires_after_a_day():
    now = [0.0]
    log = AnswerLog(lambda: now[0])
    log.add(1, 2, [10, 11])
    assert log.get(11).asker_id == 2
    now[0] = 24 * 3600 + 1
    assert log.get(10) is None


class Partial:
    def __init__(self, deleted, mid):
        self.deleted, self.mid = deleted, mid

    async def delete(self):
        self.deleted.append(self.mid)


class Bot:
    """Wires the fake channel and the memory queue of the `bot` fixture."""

    def __init__(self, bot, world):
        self.bot, self.world = bot, world
        self.deleted, self.jobs = [], []
        world.gen.get_partial_message = lambda mid: Partial(self.deleted, mid)
        bot._memory_enqueue = self.jobs.append

    def ask_reply(self, asker, content, ref_id, **kw):
        m = TMsg(self.world.g, self.world.gen, asker, content, **kw)
        m.reference = SimpleNamespace(
            message_id=ref_id, resolved=SimpleNamespace(author=ME, reference=None)
        )
        return m


@pytest.fixture
def env(bot, world):
    return Bot(bot, world)


async def test_asker_deletes_all_chunks_with_thumbs_up_and_memory_removal(env):
    asker = member(env.world.g, STAFF)
    env.bot._answers.add(env.world.gen.id, asker.id, [501, 502, 503])
    m = env.ask_reply(asker, "ลบ", 502)
    await env.bot.on_message(m)
    assert env.deleted == [501, 502, 503]
    assert m.reactions == ["👍"] and m.replies == []
    (job,) = [j for j in env.jobs if getattr(j, "func", None) == env.bot.memory.delete]
    assert isinstance(job, partial) and job.func == env.bot.memory.delete
    assert job.args == (GUILD, [501, 502, 503])
    assert env.bot._answers.get(501) is None  # forgotten


async def test_other_people_are_refused_but_managers_and_executives_may(env):
    g = env.world.g
    asker = member(g, STAFF)
    for who, ok in ((member(g, 77), False), (member(g, MANAGER, manage_guild=True), True)):
        env.deleted.clear()
        env.bot._answers.add(env.world.gen.id, asker.id, [600, 601])
        m = env.ask_reply(who, "delete", 601)
        await env.bot.on_message(m)
        assert bool(env.deleted) is ok
        assert (m.reactions == ["👍"]) is ok
        if not ok:
            assert m.replies and "Salid" in m.replies[0] or "สลิด" in m.replies[0]
    env.deleted.clear()
    env.bot._answers.add(env.world.gen.id, asker.id, [700])
    m = env.ask_reply(member(g, EXEC), "unsend", 700)
    await env.bot.on_message(m)
    assert env.deleted == [700]


async def test_untracked_salid_message_uses_the_replied_to_author(env):
    g = env.world.g
    asker = member(g, STAFF)
    m = env.ask_reply(asker, "เอาออก", 900)
    m.reference.resolved.reference = SimpleNamespace(resolved=SimpleNamespace(author=asker))
    await env.bot.on_message(m)
    assert env.deleted == [900] and m.reactions == ["👍"]
    # a message that is not Salid's is left alone (and the text goes on as normal chat)
    env.deleted.clear()
    other = env.ask_reply(asker, "ลบ", 901)
    other.reference.resolved = SimpleNamespace(author=asker, reference=None)
    await env.bot.on_message(other)
    assert env.deleted == []


async def test_real_answer_tracks_every_chunk(env, monkeypatch):
    del env.bot._answer
    monkeypatch.setattr(
        botmod, "ask", lambda *a, **k: Answer(text="x" * 4500, rounds=1, seconds=0.1, tool_calls=[])
    )
    asker = member(env.world.g, STAFF)
    sent_ids = iter([801, 802, 803])

    class S:
        def __init__(self):
            self.id = next(sent_ids)

    m = TMsg(env.world.g, env.world.gen, asker, "ยอดเท่าไร", mention=True)

    async def reply(text, **kw):
        return S()

    m.reply = reply
    env.world.gen.send = lambda text, **kw: asyncio.sleep(0, S())
    await env.bot._answer(m, env.world.g, asker)
    rec = env.bot._answers.get(801)
    assert rec.message_ids == [801, 802, 803] and rec.asker_id == STAFF


def _receipt(store):
    r = store.add_receipt(
        Receipt(
            guild_id=GUILD, submitter_id=ALICE, submitter_name="Alice", sha256="s1", vendor="V",
            receipt_date="2026-10-10", effective_date="2026-10-10", total=10.0, category="อื่นๆ",
            confidence=0.9, created_at="2026-10-10",
        )
    )  # fmt: skip
    store.set_reply_message(r.id, 555)
    return r


async def test_receipt_reply_voids_and_only_delete_message_removes_the_message(env):
    r = _receipt(env.bot.store)
    alice = member(env.world.g, ALICE)
    m = env.ask_reply(alice, "ลบ", 555)
    await env.bot.on_message(m)
    assert env.bot.store.get(r.id).status == "void"
    assert env.deleted == [] and m.replies  # voided, message kept
    env.bot.store.set_status(r.id, "active")
    m2 = env.ask_reply(alice, "ลบข้อความ", 555)
    await env.bot.on_message(m2)
    assert env.bot.store.get(r.id).status == "void"
    assert env.deleted == [555]
    assert m2.replies == [f"ยกเลิกใบเสร็จ #{r.id} แล้ว (กู้คืนได้)"]


# ------------------------------------------------------------------ cancel deep task
class Job:
    def __init__(self, done=False):
        self.cancelled, self._done = False, done

    def done(self):
        return self._done

    def cancel(self):
        self.cancelled = True


async def test_cancel_permissions_and_effect(bot, world):
    g = world.g
    handle = DeepHandle()
    job = Job()
    bot._deep_jobs[GUILD] = botmod._DeepJob(job, handle, STAFF)
    assert "ผู้จัดการ" in await bot._cancel_deep(GUILD, member(g, 77), "th")
    assert not job.cancelled
    ok = await bot._cancel_deep(GUILD, member(g, STAFF), "en")
    assert "cancelled" in ok and job.cancelled and handle.cancelled
    job2, h2 = Job(), DeepHandle()
    bot._deep_jobs[GUILD] = botmod._DeepJob(job2, h2, STAFF)
    await bot._cancel_deep(GUILD, member(g, MANAGER, manage_guild=True), "th")
    assert job2.cancelled
    bot._deep_jobs.clear()
    assert "ไม่มีงาน" in await bot._cancel_deep(GUILD, member(g, STAFF), "th")


async def test_typed_cancel_command(bot, world):
    bot._deep_jobs[GUILD] = botmod._DeepJob(Job(), DeepHandle(), STAFF)
    m = TMsg(world.g, world.gen, member(world.g, STAFF), "ยกเลิกงาน", mention=True)
    await bot.on_message(m)
    assert m.replies == ["ยกเลิกงานใหญ่ที่กำลังทำอยู่แล้วค่ะ"]


def test_handle_kills_a_real_process_and_run_reports_cancel(tmp_path):
    handle = DeepHandle()
    code = "import time; time.sleep(60)"
    threading.Timer(0.5, handle.cancel).start()
    with pytest.raises(DeepCancelled):
        _run([sys.executable, "-c", code], {}, tmp_path, 30, handle)
    assert handle.proc.poll() is not None
    assert isinstance(handle.proc, subprocess.Popen)
