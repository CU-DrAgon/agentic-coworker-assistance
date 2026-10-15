# ruff: noqa: F811, F401
"""The coordinator: DM notes, task tracking and follow-ups, disclosure requests, the reports
channel and /รายงาน. Offline: fake Discord objects, SQLite, scripted models."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import discord
import pytest
from conftest import GUILD, FakeClient, FixedClockSettings, fc_resp, json_resp, scripted, text_resp
from test_bot_multitenant import FakeChannel, FakeGuild, member
from test_modes import Chan, TMsg, bot  # noqa: F401  (fixture)

from salid import coo, qa
from salid import reports as rep
from salid.bot import SecretaryBot
from salid.coo import REPORT_CHANNEL_NAME, parse_private_marker, plan_report_overwrites
from salid.coo_store import DISCLOSURES, STATE, FirestoreDocStore, SQLiteDocStore
from salid.coo_tools import CooTools, build_coo_tools
from salid.disclosure import DisclosureRepo, describe_hits, pick_owner
from salid.i18n import t
from salid.memory import MemoryScheduler, build_message
from salid.memory_access import Scope, dm_scope, hierarchy_authors
from salid.memory_store import SQLiteMemoryStore
from salid.tasks import (
    Task,
    TaskRepo,
    TaskTracker,
    due_for_followup,
    match_person,
    parse_due,
    similar,
)

BKK = ZoneInfo("Asia/Bangkok")
OWNER_ID, ADMIN_ID, BOSS, STAFF, PEER, INTERN = 1, 2, 25, 26, 27, 28  # not 3/4/6 (test_modes)
PUBLIC, SECRET = 10, 11


# ------------------------------------------------------------------ fakes
class NS(SimpleNamespace):
    """Discord objects are hashable (permission overwrites are keyed by them)."""

    __hash__ = object.__hash__


class RChan(Chan):
    """A channel only `readers` may read; `sent` keeps what Salid posted."""

    def __init__(self, id, name, readers=None):
        super().__init__(id, name)
        self.readers, self.mention = readers, f"<#{id}>"

    def permissions_for(self, m):
        ok = self.readers is None or m.id in self.readers
        return SimpleNamespace(read_messages=ok, read_message_history=ok, send_messages=True)


def person(g, uid, name, pos=1, roles=(), **perms):
    m = NS(**vars(member(g, uid, roles=roles, **perms)))
    g.members[uid] = m
    m.display_name, m.top_role, m.dms = name, SimpleNamespace(position=pos), []

    async def send(text=None, **kw):
        m.dms.append((text, kw))

    m.send = send
    return m


@pytest.fixture
def world(monkeypatch):
    from test_modes import ME

    monkeypatch.setattr(SecretaryBot, "user", property(lambda s: ME))
    chans = [
        RChan(PUBLIC, "ทั่วไป"),
        RChan(SECRET, "ลับ", {OWNER_ID, ADMIN_ID, BOSS}),
    ]
    g = FakeGuild(GUILD, "A", chans)
    g.default_role, g.roles = NS(name="@everyone"), []
    monkeypatch.setattr(SecretaryBot, "guilds", property(lambda s: [g]))
    people = SimpleNamespace(
        owner=person(g, OWNER_ID, "Owner", 9),
        admin=person(g, ADMIN_ID, "Admin", 7, administrator=True),
        boss=person(g, BOSS, "Boss", 5, roles=["Executive"]),
        staff=person(g, STAFF, "Somchai", 2),
        peer=person(g, PEER, "Peer", 2),
        intern=person(g, INTERN, "Ploy", 1),
    )
    chat = SimpleNamespace(id=999)  # test_modes' bot fixture wants an ask channel; none here
    return SimpleNamespace(g=g, pub=chans[0], secret=chans[1], chat=chat, **vars(people))


def at(h, m=0, day=14):
    return datetime(2026, 10, day, h, m, tzinfo=BKK)


def add_msg(bot, mid, uid, name, text, cid, when=None, kind="chat", visibility="normal"):
    m = build_message(
        bot.s, guild_id=GUILD, channel_id=cid, message_id=mid, author_id=uid, author_name=name,
        created_at=(when or at(10)).astimezone(UTC), text=text, kind=kind,
    )  # fmt: skip
    bot.memory_store.add_message(m.model_copy(update={"visibility": visibility}))


def dm(world, uid, text, mid=None):
    author = SimpleNamespace(id=uid, bot=False)
    msg = TMsg(None, Chan(900 + uid), author, text)
    msg.created_at = at(12).astimezone(UTC)
    return msg


# ------------------------------------------------------------------ private marker
@pytest.mark.parametrize(
    "text, expected",
    [("เรื่องนี้เก็บเป็นความลับ", (True, True)), ("เป็นความลับนะครับ", (True, True)),
     ("private", (True, True)), ("keep this private please", (True, True)),
     ("พรุ่งนี้ผมลา เรื่องนี้เก็บเป็นความลับนะ", (True, False)),
     ("ลูกค้า X ขอเลื่อนส่งของเป็นศุกร์", (False, False)), ("", (False, False))],
)  # fmt: skip
def test_private_marker(text, expected):
    assert parse_private_marker(text) == expected


# ------------------------------------------------------------------ DM notes
async def test_dm_is_recorded_as_a_private_scope_note_with_a_one_time_notice(bot, world):
    m = dm(world, STAFF, "พรุ่งนี้ผมลา")
    await bot.on_message(m)
    await bot.drain_memory()
    rows = bot.memory_store.messages(GUILD, {dm_scope(STAFF)}, "2026-10-14", "2026-10-14")
    assert [(r.kind, r.channel_id, r.author_id, r.visibility) for r in rows] == [
        ("dm", dm_scope(STAFF), STAFF, "normal")
    ]
    assert rows[0].text == "พรุ่งนี้ผมลา" and rows[0].author_name == "Somchai"
    assert m.replies == ["ans"] and m.channel.sent == [t("dm_notice", "th")]  # first DM only
    m2 = dm(world, STAFF, "ลูกค้า X ขอเลื่อนส่งของเป็นศุกร์")
    await bot.on_message(m2)
    assert m2.channel.sent == []


async def test_a_member_dm_question_is_recorded_once_and_answered_once(bot, world):
    m = dm(world, STAFF, "ยอดใช้จ่ายเดือนนี้เท่าไร")
    await bot.on_message(m)
    await bot.drain_memory()
    rows = bot.memory_store.messages(GUILD, {dm_scope(STAFF)}, "2026-10-14", "2026-10-14")
    assert [r.text for r in rows] == ["ยอดใช้จ่ายเดือนนี้เท่าไร"]  # one note
    assert bot.calls == [("answer", "ยอดใช้จ่ายเดือนนี้เท่าไร")]  # one agent answer
    assert m.replies == ["ans"]


async def test_salids_own_dm_is_neither_recorded_nor_answered(bot, world):
    from test_modes import ME

    own = TMsg(None, Chan(901), ME, "สวัสดีครับ")
    own.created_at = at(12).astimezone(UTC)
    await bot._direct_message(own)
    await bot.drain_memory()
    assert bot.memory_store.messages(GUILD, {dm_scope(STAFF)}, "2026-10-14", "2026-10-14") == []
    assert bot.calls == [] and own.replies == []


async def test_a_dm_reason_after_a_meeting_reminder_goes_to_the_absence_path_only(
    bot, world, monkeypatch
):
    now = datetime.now(UTC)
    bot.scheduler.store.put_meeting({
        "id": "m1", "guild_id": GUILD, "title": "Sync", "status": "scheduled", "kind": "text",
        "text_channel_id": PUBLIC, "attendee_ids": [STAFF], "attendee_names": ["Somchai"],
        "dm_sent": {str(STAFF): now.isoformat(timespec="seconds")}, "attendance_notes": {},
        "start_utc": (now + timedelta(minutes=5)).isoformat(timespec="seconds"),
    })  # fmt: skip
    monkeypatch.setattr(bot.scheduler, "get_guild", lambda gid: world.g)
    monkeypatch.setattr(bot.scheduler, "clock", lambda: now)

    async def post(guild, channel_id, text):
        world.pub.log.append(text)

    monkeypatch.setattr(bot.scheduler, "post", post)
    m = dm(world, STAFF, "ไปช้า 10 นาที")
    await bot.on_message(m)
    await bot.drain_memory()
    assert m.replies == [t("sched_absence_ack", "th", title="Sync")]  # the absence ack only
    assert bot.calls == []  # not answered a second time by the agent
    assert bot.scheduler.store.get_meeting("m1")["attendance_notes"][str(STAFF)]["note"]
    # a long question in the same window is not a reason: it is noted and answered normally
    q = dm(world, STAFF, "ช่วยสรุปยอดใช้จ่ายเดือนนี้ให้หน่อย " + "x" * 250)
    await bot.on_message(q)
    assert [c[0] for c in bot.calls] == ["answer"]


async def test_private_note_is_marked_and_a_bare_marker_flips_the_previous_note(bot, world):
    await bot.on_message(dm(world, STAFF, "งานที่ผมทำอยู่คือแผนปรับโครงสร้าง"))
    bare = dm(world, STAFF, "เรื่องนี้เก็บเป็นความลับ")
    await bot.on_message(bare)
    await bot.drain_memory()
    assert bare.replies == [t("dm_private_marked", "th")]  # answered without the agent
    (row,) = bot.memory_store.messages(GUILD, {dm_scope(STAFF)}, "2026-10-14", "2026-10-14")
    assert row.visibility == "private"
    inline = dm(world, STAFF, "ผมจะลาออก เป็นความลับนะ")
    await bot.on_message(inline)
    await bot.drain_memory()
    rows = bot.memory_store.messages(GUILD, {dm_scope(STAFF)}, "2026-10-14", "2026-10-14")
    assert sorted(r.visibility for r in rows) == ["private", "private"]
    assert bot.answer_kw[-1]["note"] == t("dm_private_note", "th")


async def test_dm_is_not_recorded_with_memory_off_or_when_the_guild_is_ambiguous(bot, world):
    bot.guilds_store.save_settings(
        GUILD, __import__("salid.guild_settings", fromlist=["x"]).GuildSettingsUpdate(
            features={"memory": False}), "x")  # fmt: skip
    bot.team._cache.clear()
    await bot.on_message(dm(world, STAFF, "พรุ่งนี้ผมลา"))
    await bot.drain_memory()
    assert bot.memory_store.messages(GUILD, {dm_scope(STAFF)}, "2026-10-14", "2026-10-14") == []


async def test_dm_notes_visible_to_higher_ranks_only_through_the_tools(bot, world):
    await bot.on_message(dm(world, STAFF, "พรุ่งนี้ผมลา"))
    await bot.on_message(dm(world, STAFF, "เรื่องส่วนตัวนี้เก็บเป็นความลับ"))
    await bot.drain_memory()
    where = SimpleNamespace(guild=None, channel=None)  # a DM: the answer reaches one person

    async def seen_by(who):
        tools = await bot._memory_tools(where, world.g, who)
        r = tools.chat_messages("2026-10-14", "2026-10-14", person="somchai")
        return [x["text"] for x in r["messages"]]

    assert await seen_by(world.boss) == ["พรุ่งนี้ผมลา"]  # higher rank: the work note only
    assert len(await seen_by(world.staff)) == 2  # the author sees both
    assert await seen_by(world.peer) == []  # same rank
    assert await seen_by(world.intern) == []  # lower rank
    in_public = SimpleNamespace(guild=world.g, channel=world.pub)  # a channel everyone could read
    tools = await bot._memory_tools(in_public, world.g, world.boss)
    assert tools.chat_messages("2026-10-14", "2026-10-14", person="somchai")["count"] == 0


# ------------------------------------------------------------------ tasks: pure helpers
TODAY = datetime(2026, 10, 14).date()  # a Wednesday


@pytest.mark.parametrize(
    "text, expected",
    [("2026-10-16", "2026-10-16"), ("วันนี้", "2026-10-14"), ("พรุ่งนี้เช้า", "2026-10-15"),
     ("tomorrow", "2026-10-15"), ("มะรืนนี้", "2026-10-16"), ("ศุกร์", "2026-10-16"),
     ("by Friday", "2026-10-16"), ("พุธ", "2026-10-21"),  # the same weekday = next week
     ("เร็วๆ นี้", ""), ("", ""), (None, ""), ("2026-13-45", "")],
)  # fmt: skip
def test_parse_due(text, expected):
    assert parse_due(text, TODAY) == expected


def test_match_person_by_name_mention_and_ambiguity():
    people = {1: "Somchai", 2: "Somsri", 3: "แอน", 4: "Ann Lee"}
    assert match_person("somchai", people) == 1
    assert match_person("<@3>", people) == 3
    assert match_person("@แอน", people) == 3
    assert match_person("Ann", people) == 4  # unique substring
    assert match_person("Som", people) == 0  # ambiguous: two matches
    assert match_person("ใครสักคน", people) == 0 and match_person("", people) == 0
    assert match_person("<@99>", people) == 0  # not a member


def test_similar_titles():
    assert similar("ส่งใบเสนอราคาลูกค้า X", "ส่งใบเสนอราคา ลูกค้า X!")
    assert not similar("ส่งใบเสนอราคาลูกค้า X", "จองห้องประชุม")


# ------------------------------------------------------------------ tasks: repo, tracker
@pytest.fixture(params=["sqlite", "firestore"])
def docs(request):
    if request.param == "sqlite":
        return SQLiteDocStore(":memory:")
    from fake_firestore import FakeFirestore

    return FirestoreDocStore(client=FakeFirestore())


PEOPLE = {STAFF: "Somchai", BOSS: "Boss", INTERN: "Ploy"}


def tracker(docs, settings, llm=None, **kw):
    return TaskTracker(TaskRepo(docs), settings, llm, people_of=lambda gid: PEOPLE, **kw)


def summary(*items):
    return {"action_items": [dict(who=w, what=x, due=d) for w, x, d in items]}


def test_action_items_become_tasks_with_assignee_due_and_dedupe(docs):
    s = FixedClockSettings()
    tr = tracker(docs, s)
    msgs = [SimpleNamespace(jump_url="https://d/1", text="x", kind="chat")]
    new = tr.ingest(GUILD, PUBLIC, "2026-10-14", summary(
        ("Somchai", "ส่งใบเสนอราคาลูกค้า X", "ศุกร์"), ("ใครไม่รู้", "จองห้องประชุม", ""),
        (f"<@{INTERN}>", "ทำโพสต์โปรโมชัน", "2026-10-15"), ("Somchai", "", "")), msgs)  # fmt: skip
    assert [(t_.assignee_id, t_.assignee_name, t_.due) for t_ in new] == [
        (STAFF, "Somchai", "2026-10-16"),
        (0, "ใครไม่รู้", ""),
        (INTERN, "Ploy", "2026-10-15"),
    ]
    assert new[0].source_link == "https://d/1" and new[0].created_from == "summary"
    # the living summary lists the same item again (rephrased): no duplicate
    again = tr.ingest(GUILD, PUBLIC, "2026-10-14", summary(
        ("Somchai", "ส่งใบเสนอราคา ลูกค้า X ให้ด้วย", "2026-10-16")), msgs)  # fmt: skip
    assert again == [] and len(tr.repo.list(GUILD, status=None)) == 3
    # other guilds are separate
    assert TaskRepo(docs).list(GUILD + 1, status=None) == []


def test_a_finished_task_is_not_recreated_by_a_later_summary(docs):
    tr = tracker(docs, FixedClockSettings())
    (task,) = tr.ingest(GUILD, PUBLIC, "2026-10-14", summary(("Somchai", "ส่งรายงานยอดขาย", "")), [])
    tr.repo.set_status(GUILD, task.id, "done", "ส่งแล้ว")
    assert (
        tr.ingest(GUILD, PUBLIC, "2026-10-15", summary(("Somchai", "ส่งรายงานยอดขาย", "")), []) == []
    )
    assert tr.repo.list(GUILD) == [] and tr.repo.get(GUILD, task.id).done_at


def test_voice_and_dm_sources_are_labelled(docs):
    tr = tracker(docs, FixedClockSettings(), kind_of=lambda g, c: "meeting")
    (a,) = tr.ingest(GUILD, 30, "2026-10-14", summary(("Somchai", "สรุปประชุม", "")), [])
    (b,) = tr.ingest(GUILD, dm_scope(STAFF), "2026-10-14", summary(("", "ส่งของวันศุกร์", "ศุกร์")), [])
    assert a.created_from == "meeting"
    assert (b.created_from, b.assignee_id, b.source_link) == ("dm", STAFF, "")  # own note, no link


def test_done_marking_is_one_batched_cheap_call_and_only_for_known_open_tasks(docs):
    s = FixedClockSettings()
    llm = FakeClient([])
    tr = tracker(docs, s, llm)
    a, b = tr.ingest(GUILD, PUBLIC, "2026-10-14", summary(
        ("Somchai", "ส่งใบเสนอราคาลูกค้า X", ""), ("Ploy", "ทำโพสต์โปรโมชัน", "")), [])  # fmt: skip
    llm.models.script = [json_resp({"done": [{"id": a.id, "note": "ส่งแล้ว"}, {"id": "nope"}]})]
    msgs = [build_message(s, guild_id=GUILD, channel_id=PUBLIC, message_id=i, author_id=STAFF,
                          author_name="Somchai", created_at=at(11), text=x)
            for i, x in ((1, "ส่งใบเสนอราคาลูกค้า X ให้แล้วครับ"), (2, "เดี๋ยวทำโพสต์ต่อ"))]  # fmt: skip
    assert tr.mark_done(GUILD, msgs) == [a.id]
    assert len(llm.models.calls) == 1 and llm.models.calls[0][0] == s.gate_model  # Flash-Lite
    prompt = llm.models.calls[0][1]
    assert a.id in prompt and b.id in prompt and "เดี๋ยวทำโพสต์ต่อ" in prompt
    assert [x.id for x in tr.repo.list(GUILD)] == [b.id]
    assert tr.repo.get(GUILD, a.id).status == "done" and tr.repo.get(GUILD, a.id).note == "ส่งแล้ว"
    # no open task or no messages: no model call at all
    llm.models.script = []
    tr.repo.set_status(GUILD, b.id, "done")
    assert tr.mark_done(GUILD, msgs) == [] and len(llm.models.calls) == 1


def test_a_failing_model_never_breaks_the_summary_hook(docs):
    s = FixedClockSettings()
    tr = tracker(docs, s, FakeClient([]))  # script exhausted: the call raises
    tr.ingest(GUILD, PUBLIC, "2026-10-14", summary(("Somchai", "ทำงาน A", "")), [])
    msg = build_message(s, guild_id=GUILD, channel_id=PUBLIC, message_id=1, author_id=1,
                        author_name="x", created_at=at(9), text="ทำเสร็จแล้ว")  # fmt: skip
    tr.on_summary(GUILD, PUBLIC, "2026-10-14", {}, [msg])  # no exception
    assert len(tr.repo.list(GUILD)) == 1


def test_scheduler_hook_creates_tasks_after_a_summary_update(tmp_path):
    s = FixedClockSettings()
    store = SQLiteMemoryStore(str(tmp_path / "m.db"))
    tr = tracker(SQLiteDocStore(":memory:"), s, FakeClient([]))
    items = summary(("Somchai", "ส่งใบเสนอราคาลูกค้า X", "2026-10-16"))
    llm = FakeClient([json_resp({"overview": "o", **items})])
    clock = lambda: at(10, 40)  # noqa: E731
    sched = MemoryScheduler(store, s, llm, now=clock, on_summary=tr.on_summary)
    for i in range(30):
        m = build_message(s, guild_id=GUILD, channel_id=PUBLIC, message_id=i + 1, author_id=STAFF,
                          author_name="Somchai", created_at=at(10, 0, 14), text=f"m{i}",
                          jump_url=f"https://d/{i}")  # fmt: skip
        store.add_message(m)
        sched.note(GUILD, PUBLIC, "2026-10-14")
    assert sched.tick([GUILD]) == 1
    (task,) = tr.repo.list(GUILD)
    assert (task.assignee_id, task.due, task.source_channel_id) == (STAFF, "2026-10-16", PUBLIC)


def test_private_dm_notes_are_never_summarised(tmp_path):
    s = FixedClockSettings()
    store = SQLiteMemoryStore(str(tmp_path / "m.db"))
    llm = FakeClient([json_resp({"overview": "o"})])
    sched = MemoryScheduler(store, s, llm, now=lambda: at(10, 40))
    for i, vis in enumerate(["normal", "private"] * 15):
        m = build_message(s, guild_id=GUILD, channel_id=dm_scope(STAFF), message_id=i + 1,
                          author_id=STAFF, author_name="Somchai", created_at=at(10), kind="dm",
                          text=f"note-{i}-{vis}")  # fmt: skip
        store.add_message(m.model_copy(update={"visibility": vis}))
        sched.note(GUILD, dm_scope(STAFF), "2026-10-14")
    assert sched.tick([GUILD]) == 1
    prompt = llm.models.calls[0][1]
    assert "-normal" in prompt and "-private" not in prompt


def test_followup_selection_due_today_and_overdue_only(docs):
    repo = TaskRepo(docs)
    mk_ = lambda title, due, uid=STAFF, st="open": repo.save(  # noqa: E731
        Task(guild_id=GUILD, assignee_id=uid, title=title, due=due, status=st)
    )
    mk_("today", "2026-10-14"), mk_("late", "2026-10-10"), mk_("future", "2026-10-20")
    mk_("nodue", ""), mk_("done", "2026-10-10", st="done"), mk_("nobody", "2026-10-10", uid=0)
    mk_("boss late", "2026-10-13", uid=BOSS)
    got = due_for_followup(repo.list(GUILD), TODAY)
    assert {u: [x.title for x in v] for u, v in got.items()} == {
        STAFF: ["late", "today"],
        BOSS: ["boss late"],
    }


# ------------------------------------------------------------------ follow-up DMs (fake clock)
def seed_tasks(bot, world):
    bot.tasks.save(Task(guild_id=GUILD, assignee_id=STAFF, assignee_name="Somchai",
                        title="ส่งใบเสนอราคาลูกค้า X", due="2026-10-14", id="aaaa1111"))  # fmt: skip
    bot.tasks.save(Task(guild_id=GUILD, assignee_id=STAFF, title="งานเก่า", due="2026-10-10"))
    bot.tasks.save(Task(guild_id=GUILD, assignee_id=INTERN, title="ยังไม่ถึง", due="2026-10-20"))
    bot.tasks.save(Task(guild_id=GUILD, assignee_id=PEER, title="เสร็จแล้ว", due="2026-10-12", status="done"))  # fmt: skip


async def test_followups_dm_each_member_once_a_day_at_the_configured_time(bot, world):
    seed_tasks(bot, world)
    assert await bot._followups_due(at(9, 59)) == 0  # before 10:00 (the default)
    assert await bot._followups_due(at(10, 5)) == 1
    ((text, _),) = world.staff.dms
    assert "#aaaa1111" in text and "ส่งใบเสนอราคาลูกค้า X" in text and "งานเก่า" in text
    assert "2026-10-14" in text and "ยังไม่ถึง" not in text
    assert world.intern.dms == [] and world.peer.dms == []  # nothing due / already done
    assert await bot._followups_due(at(10, 30)) == 0  # once a day, even after a restart:
    assert bot.docs.get(STATE, GUILD, "followup:2026-10-14") is not None
    assert await bot._followups_due(at(14, 0)) == 0  # outside the window later in the day
    assert await bot._followups_due(at(10, 5, day=15)) == 1  # next day again


async def test_followups_respect_the_flag_and_the_hub_time(bot, world):
    seed_tasks(bot, world)
    from salid.guild_settings import GuildSettingsUpdate

    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(followup_time="08:00"), "x")
    bot.team._cache.clear()
    assert await bot._followups_due(at(8, 1)) == 1  # the configured time, not 10:00
    world.staff.dms.clear()
    bot.guilds_store.save_settings(
        GUILD, GuildSettingsUpdate(followup_time="08:00", features={"followups": False}), "x"
    )
    bot.team._cache.clear()
    assert await bot._followups_due(at(8, 2, day=15)) == 0 and world.staff.dms == []


# ------------------------------------------------------------------ update_task / list_tasks
def coo_tools(docs, asker, *, is_exec=False, scope=None):
    repo = TaskRepo(docs)
    return repo, CooTools(guild_id=GUILD, asker_id=asker, is_exec=is_exec, scope=scope, tasks=repo,
                          people={**PEOPLE, PEER: "Peer"}, today=lambda: TODAY)  # fmt: skip


def test_update_task_is_limited_to_own_tasks_unless_executive(docs):
    repo, tools = coo_tools(docs, STAFF)
    mine = repo.save(Task(guild_id=GUILD, assignee_id=STAFF, title="ของฉัน"))
    other = repo.save(Task(guild_id=GUILD, assignee_id=INTERN, title="ของคนอื่น"))
    nobody = repo.save(Task(guild_id=GUILD, assignee_id=0, title="ไม่มีเจ้าของ"))
    assert tools.update_task(mine.id, "done", "เสร็จแล้ว")["updated"]["status"] == "done"
    assert tools.update_task(f"#{mine.id}", "open")["updated"]["status"] == "open"
    assert "not allowed" in tools.update_task(other.id, "done")["error"]
    assert "not allowed" in tools.update_task(nobody.id, "done")["error"]
    assert "no such task" in tools.update_task("zzzz", "done")["error"]
    assert "status" in tools.update_task(mine.id, "weird")["error"]
    assert repo.get(GUILD, other.id).status == "open"
    _, boss = coo_tools(docs, BOSS, is_exec=True)
    assert boss.update_task(other.id, "done")["updated"]["status"] == "done"


def test_list_tasks_respects_the_hierarchy(docs):
    ranks = {BOSS: 50, STAFF: 20, PEER: 20, INTERN: 10}
    repo = TaskRepo(docs)
    repo.save(
        Task(guild_id=GUILD, assignee_id=INTERN, title="intern secret", source_channel_id=SECRET)
    )
    repo.save(Task(guild_id=GUILD, assignee_id=BOSS, title="boss secret", source_channel_id=SECRET))
    repo.save(Task(guild_id=GUILD, assignee_id=STAFF, title="public one", source_channel_id=PUBLIC))
    repo.save(Task(guild_id=GUILD, assignee_id=PEER, title="peer secret", source_channel_id=SECRET))
    repo.save(Task(guild_id=GUILD, assignee_id=INTERN, title="dm note task",
                   source_channel_id=dm_scope(INTERN)))  # fmt: skip

    def titles(asker, **kw):
        sc = Scope(
            asker_id=asker, channels=frozenset({PUBLIC}), authors=hierarchy_authors(asker, ranks)
        )
        tools = CooTools(guild_id=GUILD, asker_id=asker, is_exec=False, scope=sc, tasks=repo,
                         people=PEOPLE | {PEER: "Peer"}, today=lambda: TODAY)  # fmt: skip
        return sorted(x["title"] for x in tools.list_tasks(**kw)["tasks"])

    assert titles(BOSS) == sorted(
        ["intern secret", "boss secret", "peer secret", "public one", "dm note task"]
    )
    assert titles(STAFF) == sorted(
        ["intern secret", "public one", "dm note task"]
    )  # not the boss's
    assert "peer secret" not in titles(STAFF)  # same rank: only through channels both can read
    assert titles(PEER) == sorted(["peer secret", "intern secret", "public one", "dm note task"])
    assert titles(INTERN) == sorted(["intern secret", "dm note task", "public one"])  # own + public
    assert titles(BOSS, person="Somchai") == ["public one"]
    assert titles(INTERN, person="me") == sorted(["intern secret", "dm note task"])
    sc = Scope(asker_id=BOSS)
    assert "cannot tell" in CooTools(guild_id=GUILD, asker_id=BOSS, is_exec=True, scope=sc,
                                     tasks=repo, people=PEOPLE, today=lambda: TODAY
                                     ).list_tasks(person="ใครสักคน")["error"]  # fmt: skip


# ------------------------------------------------------------------ disclosure: storage and scope
def hit(mid, author, channel=SECRET, day="2026-10-14", kind="chat"):
    return SimpleNamespace(message_id=mid, author_id=author, channel_id=channel, local_date=day,
                           kind=kind, visibility="normal")  # fmt: skip


def test_pick_owner_is_the_highest_ranked_author_and_description_has_no_content():
    ranks = {BOSS: 50, STAFF: 20, INTERN: 10}
    hits = [hit(1, STAFF), hit(2, STAFF), hit(3, BOSS), hit(4, 0, kind="salid")]
    assert pick_owner(hits, ranks) == BOSS
    assert pick_owner([hit(1, STAFF), hit(2, INTERN)], ranks) == STAFF
    assert pick_owner([hit(4, 0, kind="salid")], ranks) == 0
    text = describe_hits([hit(1, BOSS), hit(2, BOSS, dm_scope(BOSS), "2026-10-15", "dm")],
                         {SECRET: "ลับ"}, "th")  # fmt: skip
    assert "2 ข้อความ" in text and "2026-10-14 - 2026-10-15" in text and "#ลับ" in text
    assert "บันทึก DM" in text


def test_grants_are_stored_and_unlock_only_the_approved_messages_for_the_grantee(docs):
    now = [datetime(2026, 10, 14, 12, tzinfo=UTC)]
    repo = DisclosureRepo(docs, now=lambda: now[0])
    hits = [hit(1, BOSS), hit(2, BOSS), hit(3, STAFF)]
    req, created = repo.request(guild_id=GUILD, owner_id=BOSS, grantee_id=INTERN, topic="งบโฆษณา",
                                question="งบเท่าไร", hits=hits, names={SECRET: "ลับ"})  # fmt: skip
    assert created and req.status == "pending" and req.message_ids == [1, 2]  # the owner's only
    assert repo.unlocked_ids(GUILD, INTERN) == frozenset()  # nothing before approval
    again, created2 = repo.request(guild_id=GUILD, owner_id=BOSS, grantee_id=INTERN,
                                   topic=" งบโฆษณา ", question="x", hits=hits, names={})  # fmt: skip
    assert not created2 and again.id == req.id  # no second pending request
    done = repo.approve(req.id)
    assert done.status == "approved" and done.granted_at and done.owner_id == BOSS
    assert repo.unlocked_ids(GUILD, INTERN) == {1, 2}
    assert repo.unlocked_ids(GUILD, PEER) == frozenset()  # only that grantee
    assert repo.unlocked_ids(GUILD + 1, INTERN) == frozenset()  # only that guild
    assert repo.approve(req.id) is None and repo.decline(req.id) is None  # decided once
    stored = docs.get(DISCLOSURES, GUILD, req.id)
    assert stored["status"] == "approved" and stored["message_ids"] == [1, 2]
    assert docs.find_any(DISCLOSURES, req.id)["grantee_id"] == INTERN


def test_declined_and_expired_grants_unlock_nothing(docs):
    now = [datetime(2026, 10, 14, 12, tzinfo=UTC)]
    repo = DisclosureRepo(docs, now=lambda: now[0])
    kw = dict(guild_id=GUILD, owner_id=BOSS, grantee_id=STAFF, question="q", names={})
    no, _ = repo.request(topic="a", hits=[hit(1, BOSS)], **kw)
    ok, _ = repo.request(topic="b", hits=[hit(2, BOSS)], **kw)
    assert repo.decline(no.id).status == "declined"
    repo.approve(ok.id, expires_in=timedelta(days=1))
    assert repo.unlocked_ids(GUILD, STAFF) == {2}
    now[0] += timedelta(days=2)
    assert repo.unlocked_ids(GUILD, STAFF) == frozenset()


# ------------------------------------------------------------------ disclosure: the Discord flow
class Click:
    """A button interaction by `user` on a disclosure DM."""

    def __init__(self, user, content="ASK"):
        self.user, self.message = user, SimpleNamespace(content=content)
        self.edited, self.sent = [], []

        async def edit_message(**kw):
            self.edited.append(kw)

        async def send_message(text, ephemeral=False):
            self.sent.append((text, ephemeral))

        self.response = SimpleNamespace(edit_message=edit_message, send_message=send_message)


async def request_flow(bot, world, topic="งบโฆษณา"):
    """STAFF asks in a DM about something only BOSS wrote; returns the pending request."""
    add_msg(bot, 101, BOSS, "Boss", "งบโฆษณาปรับขึ้น 20% ห้ามบอกทีม", SECRET)
    add_msg(bot, 102, BOSS, "Boss", "ตัวเลขงบโฆษณาไตรมาสหน้า 500k", SECRET, at(11))
    add_msg(bot, 104, BOSS, "Boss", "เรื่องอื่นของบอส", SECRET, at(12))
    add_msg(bot, 103, INTERN, "Ploy", "ทำโพสต์งบโฆษณาเสร็จแล้ว", PUBLIC)
    where = SimpleNamespace(guild=None, channel=None)
    mem = await bot._memory_tools(where, world.g, world.staff)
    r = mem.chat_search(["งบโฆษณา"], "2026-10-14", "2026-10-14")
    assert "withheld" in r and "500k" not in json.dumps(r, ensure_ascii=False)
    res = await asyncio.to_thread(
        bot._request_disclosure, world.g, world.staff, "งบโฆษณาเหลือเท่าไหร่", mem,
        asyncio.get_running_loop(), topic,
    )  # fmt: skip
    return mem, res


async def test_request_dms_the_owner_with_buttons_and_no_content(bot, world):
    _, res = await request_flow(bot, world)
    assert "request sent" in res["status"]
    ((text, kw),) = world.boss.dms  # the highest-ranked author of the hidden records
    assert "Somchai" in text and "งบโฆษณา" in text and "2 ข้อความ" in text and "#ลับ" in text
    assert "20%" not in text and "500k" not in text  # the content is never in the request
    ids = [b.custom_id for b in kw["view"].children]
    assert [i.rsplit(":", 2)[1] for i in ids] == ["ok", "no"] and kw["view"].timeout is None
    rx = coo.DisclosureButton.__discord_ui_compiled_template__
    assert all(rx.fullmatch(i) for i in ids)  # persistent: survives a restart via the custom id
    assert world.staff.dms == [] and world.peer.dms == []


async def test_a_second_identical_request_is_not_sent_twice(bot, world):
    mem, _ = await request_flow(bot, world)
    res = await asyncio.to_thread(bot._request_disclosure, world.g, world.staff, "q", mem,
                                  asyncio.get_running_loop(), "งบโฆษณา")  # fmt: skip
    assert "already requested" in res["status"] and len(world.boss.dms) == 1


async def test_approve_unlocks_only_those_messages_and_dms_the_answer(bot, world, monkeypatch):
    await request_flow(bot, world)
    ((_, kw),) = world.boss.dms
    req_id = kw["view"].children[0].custom_id.rsplit(":", 1)[1]
    seen = {}

    def fake_ask(settings, store, q, **kw):
        seen["memory"], seen["q"] = kw["memory"], q
        r = kw["memory"].chat_messages("2026-10-14", "2026-10-14", person="boss")
        seen["texts"] = sorted(m["text"] for m in r["messages"])
        return SimpleNamespace(text="งบโฆษณาเพิ่ม 20% และไตรมาสหน้า 500k ครับ")

    monkeypatch.setattr(qa, "ask", fake_ask)
    stranger = Click(world.peer)
    await bot.handle_disclosure(stranger, "ok", req_id)
    assert stranger.sent == [(t("disc_not_yours", "th"), True)] and world.staff.dms == []
    click = Click(world.boss)
    await bot.handle_disclosure(click, "ok", req_id)
    assert "✅" in click.edited[0]["content"] and click.edited[0]["view"] is None
    # the answer was made with exactly the approved records unlocked (101, 102; not 104)
    assert seen["memory"].vis.scope.unlocked == {101, 102} and seen["memory"].vis.scope.dm
    assert seen["texts"] == sorted(["ตัวเลขงบโฆษณาไตรมาสหน้า 500k", "งบโฆษณาปรับขึ้น 20% ห้ามบอกทีม"])
    ((answer, _),) = world.staff.dms
    assert t("disc_asker_approved", "th", topic="งบโฆษณา") in answer and "500k" in answer
    again = Click(world.boss)
    await bot.handle_disclosure(again, "no", req_id)  # a decided request stays decided
    assert again.sent == [(t("disc_handled", "th"), True)] and len(world.staff.dms) == 1
    # Salid now remembers it was disclosed: the same asker can keep talking about it in a DM ...
    dm_where = SimpleNamespace(guild=None, channel=None)
    later = await bot._memory_tools(dm_where, world.g, world.staff)
    got = later.chat_messages("2026-10-14", "2026-10-14", person="boss")
    assert sorted(m["text"] for m in got["messages"]) == seen["texts"]  # 104 stays hidden
    # ... but nobody else, and not in front of an audience that was never granted anything
    other = await bot._memory_tools(dm_where, world.g, world.peer)
    assert other.chat_messages("2026-10-14", "2026-10-14", person="boss")["count"] == 0
    public = SimpleNamespace(guild=world.g, channel=world.pub)
    in_channel = await bot._memory_tools(public, world.g, world.staff)
    assert in_channel.chat_messages("2026-10-14", "2026-10-14", person="boss")["count"] == 0


async def test_decline_tells_the_asker_politely_and_unlocks_nothing(bot, world, monkeypatch):
    await request_flow(bot, world)
    ((_, kw),) = world.boss.dms
    req_id = kw["view"].children[1].custom_id.rsplit(":", 1)[1]
    monkeypatch.setattr(qa, "ask", lambda *a, **k: pytest.fail("no answer after a refusal"))
    click = Click(world.boss)
    await bot.handle_disclosure(click, "no", req_id)
    assert "🚫" in click.edited[0]["content"]
    ((text, _),) = world.staff.dms
    assert text == t("disc_asker_declined", "th", topic="งบโฆษณา")
    assert bot.disclosures.unlocked_ids(GUILD, STAFF) == frozenset()
    mem = await bot._memory_tools(SimpleNamespace(guild=None, channel=None), world.g, world.staff)
    assert mem.chat_messages("2026-10-14", "2026-10-14", person="boss")["count"] == 0


async def test_request_tool_needs_something_withheld_and_a_reachable_owner(bot, world):
    mem = await bot._memory_tools(SimpleNamespace(guild=None, channel=None), world.g, world.staff)
    tools = bot._coo_tools(world.g, world.staff, False, mem, "q", "th")
    request = next(f for f in tools if f.__name__ == "request_disclosure")
    assert "nothing was withheld" in request("งบโฆษณา")["error"]
    add_msg(bot, 101, BOSS, "Boss", "งบโฆษณา ลับ", SECRET)
    mem.chat_search(["งบโฆษณา"], "2026-10-14", "2026-10-14")
    assert mem.hidden
    del world.g.members[BOSS]  # the owner left the server
    res = await asyncio.to_thread(request, "งบโฆษณา")
    assert "could not message" in res["error"]


# ------------------------------------------------------------------ reports channel
def role(name, **perms):
    return NS(name=name, managed=perms.pop("managed", False),
                           permissions=SimpleNamespace(administrator=perms.get("administrator", False),
                                                       manage_guild=perms.get("manage_guild", False)))  # fmt: skip


def test_overwrites_hide_the_channel_from_everyone_but_leadership(world):
    g, me = world.g, NS(id=999)
    admin, manager, execs = (
        role("Admins", administrator=True),
        role("Mods", manage_guild=True),
        role("Executive"),
    )
    g.roles = [
        g.default_role,
        admin,
        manager,
        execs,
        role("staff"),
        role("Bot", administrator=True, managed=True),
    ]
    ow = plan_report_overwrites(g, me, {"Executive"}, {BOSS})
    assert ow[g.default_role].view_channel is False
    for r in (admin, manager, execs):
        assert ow[r].view_channel is True and ow[r].read_message_history is True
    assert g.roles[4] not in ow and g.roles[5] not in ow  # plain and bot-managed roles stay out
    assert world.owner in ow and world.boss in ow and world.staff not in ow  # owner, env exec user
    assert ow[me].view_channel and ow[me].send_messages  # Salid itself


class NewChannel(RChan):
    pass


def with_create(world, *, manage=True):
    g = world.g
    g.me = NS(id=999, guild_permissions=SimpleNamespace(manage_channels=manage))
    g.created = []

    async def create_text_channel(name, **kw):
        ch = NewChannel(77, name, readers={OWNER_ID, ADMIN_ID, BOSS})
        ch.guild, ch.kw = g, kw
        g.text_channels.append(ch)
        g.created.append(ch)
        return ch

    g.create_text_channel = create_text_channel


async def test_reports_channel_is_created_private_once_and_found_afterwards(bot, world):
    with_create(world)
    ch = await bot._report_channel(world.g)
    assert ch.name == REPORT_CHANNEL_NAME and world.g.created == [ch]
    assert ch.kw["overwrites"][world.g.default_role].view_channel is False
    assert t("report_channel_topic", "th") == ch.kw["topic"]
    assert await bot._report_channel(world.g) is ch and len(world.g.created) == 1  # no duplicates
    assert await bot._report_channel(world.g, create=False) is ch


async def test_without_manage_channels_an_admin_is_asked_once_a_day(bot, world):
    with_create(world, manage=False)
    assert await bot._report_channel(world.g) is None and world.g.created == []
    assert [t_ for t_, _ in world.owner.dms] == [
        t("report_need_channel", "th", name=REPORT_CHANNEL_NAME)
    ]
    assert await bot._report_channel(world.g) is None and len(world.owner.dms) == 1  # not nagging
    assert await bot._post_report(world.g, TODAY) is None  # nowhere private to post


async def test_a_channel_picked_in_the_hub_is_used_and_never_replaced(bot, world):
    from salid.guild_settings import GuildSettingsUpdate

    with_create(world)
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(report_channel_id=str(SECRET)), "x")
    bot.team._cache.clear()
    assert await bot._report_channel(world.g) is world.secret and world.g.created == []
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(report_channel_id="999999"), "x")
    bot.team._cache.clear()  # a deleted channel: no silent creation of another one
    assert await bot._report_channel(world.g) is None and world.g.created == []


# ------------------------------------------------------------------ report content
def seed_day(bot, world):
    from conftest import mk

    from salid.memory_store import DaySummary

    mk(bot.store, 1, STAFF, "Somchai", "Makro", "2026-10-14", 1500.0)
    mk(bot.store, 2, STAFF, "Somchai", "Grab", "2026-10-03", 250.0)
    mk(bot.store, 3, STAFF, "Somchai", "AWS", "2026-10-14", 12.5, conf=0.3)
    vc = RChan(30, "ประชุมทีม")
    world.g.voice_channels = [vc]
    top = RChan(12, "owner-only", {OWNER_ID})
    world.g.text_channels.append(top)
    add_msg(bot, 1, STAFF, "Somchai", "ส่งใบเสนอราคาลูกค้า X ให้แล้วครับ", PUBLIC)
    add_msg(bot, 2, INTERN, "Ploy", "ทำโพสต์โปรโมชันเสร็จแล้ว รอตรวจ", PUBLIC, at(11))
    add_msg(bot, 3, BOSS, "Boss", "อนุมัติงบโฆษณาเดือนหน้า", SECRET, at(12))
    add_msg(bot, 4, STAFF, "Somchai", "พรุ่งนี้ผมลา ไปงานแต่ง", dm_scope(STAFF), at(13), "dm")
    add_msg(
        bot,
        5,
        STAFF,
        "Somchai",
        "ต้องเก็บเรื่องสุขภาพเป็นความลับ",
        dm_scope(STAFF),
        at(13, 5),
        "dm",
        "private",
    )
    add_msg(bot, 6, ADMIN_ID, "Admin", "บันทึกส่วนตัวของแอดมิน", dm_scope(ADMIN_ID), at(13), "dm")
    add_msg(bot, 7, OWNER_ID, "Owner", "OWNER-ONLY-CHANNEL-TEXT", 12, at(14))
    for cid, ov in ((PUBLIC, "คุยเรื่องใบเสนอราคาและโปรโมชัน"), (SECRET, "ผู้บริหารคุยงบโฆษณา"),
                    (12, "OWNER-ONLY-SUMMARY"), (30, "ประชุมทีมสรุปแผนไตรมาส")):  # fmt: skip
        bot.memory_store.put_summary(DaySummary(
            guild_id=GUILD, channel_id=cid, local_date="2026-10-14", message_count=3,
            summary={"overview": ov, "decisions": [f"ตกลง {ov}"] if cid == 30 else []}))  # fmt: skip
    created = "2026-10-14T03:00:00+00:00"
    bot.tasks.save(Task(guild_id=GUILD, assignee_id=STAFF, title="ส่งใบเสนอราคาลูกค้า X",
                        due="2026-10-14", created_at=created, source_channel_id=PUBLIC))  # fmt: skip
    bot.tasks.save(Task(guild_id=GUILD, assignee_id=INTERN, title="ทำโพสต์โปรโมชัน", status="done",
                        done_at="2026-10-14T09:00:00+00:00", source_channel_id=PUBLIC))  # fmt: skip
    bot.tasks.save(Task(guild_id=GUILD, assignee_id=INTERN, title="จัดอีเวนต์ล่าช้า", due="2026-10-01",
                        source_channel_id=PUBLIC))  # fmt: skip
    bot.tasks.save(
        Task(guild_id=GUILD, assignee_id=0, title="หาผู้ขายรายใหม่", source_channel_id=PUBLIC)
    )
    bot.tasks.save(Task(guild_id=GUILD, assignee_id=OWNER_ID, title="OWNER-SECRET-TASK",
                        source_channel_id=12))  # fmt: skip


AUDIENCE = {OWNER_ID, ADMIN_ID, BOSS}  # the people who can read the reports channel


def test_report_input_is_compact_and_limited_to_what_the_audience_may_read(bot, world):
    seed_day(bot, world)
    bot.llm = FakeClient([text_resp("**รายงานจากโมเดล**")])
    text = bot._report_text(
        world.g, AUDIENCE - {OWNER_ID}, TODAY, "th"
    )  # no owner: owner-only is out
    assert text == "**รายงานจากโมเดล**"
    ((model, prompt, cfg),) = bot.llm.models.calls
    assert model == bot.s.gemini_model  # the main model, not the cheap one
    assert '"THB": 1512.5' in prompt and '"month_count": 3' in prompt  # ledger brief
    assert '"review": 1' in prompt and "PERSON Somchai" in prompt and "PERSON Ploy" in prompt
    assert "MEETING #ประชุมทีม" in prompt and "ตกลง ประชุมทีมสรุปแผนไตรมาส" in prompt
    assert "ผู้บริหารคุยงบโฆษณา" in prompt  # a channel every report reader may read
    assert "ส่งใบเสนอราคาลูกค้า X" in prompt
    assert "พรุ่งนี้ผมลา" in prompt and "1 DM briefings" in prompt  # a lower rank's DM briefing ...
    assert "เก็บเรื่องสุขภาพ" not in prompt  # ... never the private note
    assert "บันทึกส่วนตัวของแอดมิน" not in prompt  # same rank as a reader: not for the report
    assert "OWNER-ONLY" not in prompt  # a channel the whole audience cannot read
    assert "opened: ส่งใบเสนอราคาลูกค้า X" in prompt and "done: ทำโพสต์โปรโมชัน" in prompt
    assert "จัดอีเวนต์ล่าช้า (2026-10-01)" in prompt and "หาผู้ขายรายใหม่" in prompt
    assert len(prompt) < rep.MAX_INPUT_CHARS + 2500  # compact input plus the instructions


def test_the_owner_alone_reads_everything_including_owner_only_sources(bot, world):
    seed_day(bot, world)
    bot.llm = FakeClient([text_resp("ok")])
    bot._report_text(world.g, {OWNER_ID}, TODAY, "th")
    prompt = bot.llm.models.calls[0][1]
    assert "OWNER-ONLY-SUMMARY" in prompt and "PERSON Owner: 1 msgs in owner-only" in prompt
    assert (
        "open tasks: 1" in prompt.split("PERSON Owner")[1]
    )  # his task from the owner-only channel
    assert "บันทึกส่วนตัวของแอดมิน" in prompt  # the owner outranks the admin


def test_report_falls_back_to_a_plain_report_when_the_model_fails(bot, world):
    seed_day(bot, world)
    bot.llm = FakeClient([])  # the call raises
    text = bot._report_text(world.g, AUDIENCE, TODAY, "th")
    assert t("rep_title", "th", day="2026-10-14") in text
    assert "฿1,512.50" in text and "Somchai" in text and "ประชุมวันนี้" in text
    en = bot._report_text(world.g, AUDIENCE, TODAY, "en")
    assert t("rep_title", "en", day="2026-10-14") in en and "Company brief" in en


LEDGER = dict(today_by_currency={"THB": 10.0}, today_count=1, month_by_currency={"THB": 99.0},
              month_count=3, review=0, duplicates=0, duplicates_by_currency={})  # fmt: skip


def test_report_input_has_a_hard_size_cap():
    people = [rep.PersonDay(i, f"Person {i}", messages=50, channels=["c"] * 3,
                            snippets=["ก" * rep.SNIPPET_CHARS] * rep.PERSON_SNIPPETS,
                            opened=["งาน " * 30] * 9, overdue=["ช้า " * 30] * 9, open_n=9)
              for i in range(rep.MAX_PEOPLE)]  # fmt: skip
    chans = [{"name": f"ch{i}", "meeting": i % 2 == 0, "summary": "ข" * 700} for i in range(40)]
    inp = rep.CompanyInput("2026-10-14", LEDGER, chans, people, ["x" * 100] * 10)
    text = rep.compact_text(inp)
    assert len(text) <= rep.MAX_INPUT_CHARS + len("(more data omitted)") + 1
    assert text.endswith("(more data omitted)") and text.startswith("DATA ")
    assert len(rep.render_plain(inp, "th")) <= rep.MAX_PLAIN_CHARS


# ------------------------------------------------------------------ a member's own day
def test_personal_report_lists_only_the_members_own_work(bot, world):
    seed_day(bot, world)
    chans = {PUBLIC: "ทั่วไป", SECRET: "ลับ"}
    data = rep.person_day(guild_id=GUILD, user_id=STAFF, day=TODAY, zone=bot.s.zone,
                          memory=bot.memory_store, tasks=bot.tasks, channels=chans)  # fmt: skip
    assert data["messages_by_channel"] == {"ทั่วไป": 1, "DM": 2}  # his own notes, private ones too
    assert [x["title"] for x in data["tasks_open"]] == ["ส่งใบเสนอราคาลูกค้า X"]
    text = rep.render_personal(data, "th")
    assert "ส่งใบเสนอราคาลูกค้า X ให้แล้วครับ" in text and "โปรโมชัน" not in text
    assert "Ploy" not in text and "อนุมัติงบโฆษณา" not in text
    other = rep.person_day(guild_id=GUILD, user_id=INTERN, day=TODAY, zone=bot.s.zone,
                           memory=bot.memory_store, tasks=bot.tasks, channels=chans)  # fmt: skip
    assert [x["title"] for x in other["tasks_open"]] == ["จัดอีเวนต์ล่าช้า"]
    assert other["tasks_done_today"] == ["ทำโพสต์โปรโมชัน"] and other["tasks_open"][0]["overdue"]
    empty = rep.person_day(guild_id=GUILD, user_id=PEER, day=TODAY, zone=bot.s.zone,
                           memory=bot.memory_store, tasks=bot.tasks, channels=chans)  # fmt: skip
    assert rep.render_personal(empty, "en").count(
        "\n"
    ) == 1 and "not recorded" in rep.render_personal(empty, "en")


# ------------------------------------------------------------------ /รายงาน
class SlashInteraction:
    def __init__(self, guild, user, channel):
        self.guild, self.user, self.channel, self.locale = guild, user, channel, "th"
        self.sent, self.deferred = [], False

        async def send_message(text, ephemeral=False):
            self.sent.append((text, ephemeral))

        async def defer(ephemeral=False):
            self.deferred = True

        async def followup_send(text, ephemeral=False):
            self.sent.append((text, ephemeral))

        self.response = SimpleNamespace(send_message=send_message, defer=defer)
        self.followup = SimpleNamespace(send=followup_send)


async def test_report_command_permissions_and_dates(bot, world):
    seed_day(bot, world)
    with_create(world)
    inter = lambda who: SlashInteraction(world.g, who, world.pub)  # noqa: E731
    denied = inter(world.staff)
    await bot.handle_report(denied, None, False)
    assert denied.sent == [(t("report_denied", "th"), True)] and not denied.deferred
    assert world.g.created == []  # a refused request creates nothing
    bad = inter(world.owner)
    await bot.handle_report(bad, "14/10/2026", False)
    assert bad.sent == [(t("report_bad_date", "th"), True)]
    ok = inter(world.boss)  # an executive role: allowed
    await bot.handle_report(ok, "2026-10-14", False)
    (ch,) = world.g.created
    posted = "".join(ch.sent)
    assert t("rep_title", "th", day="2026-10-14") in posted  # plain fallback: bot.llm is None
    assert ok.sent == [(t("report_posted", "th", day="2026-10-14", channel=ch.mention), True)]
    admin = inter(world.admin)  # Administrator: allowed, reuses the existing channel
    await bot.handle_report(admin, "2026-10-13", False)
    assert len(world.g.created) == 1 and t("rep_title", "th", day="2026-10-13") in "".join(ch.sent)


async def test_report_command_for_one_self_is_open_to_everyone_and_private(bot, world):
    seed_day(bot, world)
    it = SlashInteraction(world.g, world.staff, world.pub)
    await bot.handle_report(it, None, True)
    ((text, eph),) = it.sent
    assert eph and "ส่งใบเสนอราคาลูกค้า X ให้แล้วครับ" in text and "Ploy" not in text
    assert world.g.text_channels == [world.pub, world.secret, *world.g.text_channels[2:]]


async def test_report_command_without_a_private_channel_answers_the_executive_only(bot, world):
    seed_day(bot, world)
    with_create(world, manage=False)
    it = SlashInteraction(world.g, world.boss, world.pub)
    await bot.handle_report(it, None, False)
    ((text, eph),) = it.sent
    assert eph and t("rep_title", "th", day="2026-10-14") in text
    assert "OWNER-ONLY" not in text and "บันทึกส่วนตัวของแอดมิน" not in text  # scoped to the asker


async def test_scheduled_report_posts_once_a_day_in_the_window_and_respects_the_flag(bot, world):
    from salid.guild_settings import GuildSettingsUpdate

    seed_day(bot, world)
    with_create(world)
    assert await bot._reports_due(at(18, 0)) == 0  # before 18:30
    assert await bot._reports_due(at(18, 31)) == 1
    (ch,) = world.g.created
    assert t("rep_title", "th", day="2026-10-14") in "".join(ch.sent)
    assert await bot._reports_due(at(19, 0)) == 0  # once a day
    assert await bot._reports_due(at(23, 59, day=15)) == 0  # window passed on the next day
    bot.guilds_store.save_settings(
        GUILD, GuildSettingsUpdate(report_time="20:00", features={"reports": False}), "x"
    )
    bot.team._cache.clear()
    assert await bot._reports_due(at(20, 5, day=15)) == 0


# ------------------------------------------------------------------ agent wiring
def test_the_coordinator_tools_and_prompt_reach_the_agent(bot, world, tmp_path):
    repo, tools = coo_tools(SQLiteDocStore(":memory:"), STAFF)
    repo.save(
        Task(guild_id=GUILD, assignee_id=STAFF, title="ส่งรายงาน", id="abcd1234", due="2026-10-14")
    )
    llm = scripted(
        fc_resp(("list_tasks", {})),
        fc_resp(("update_task", {"task_id": "abcd1234", "status": "done", "note": "ส่งแล้ว"})),
        text_resp("อัปเดตแล้วครับ"),
    )
    a = qa.ask(bot.s, bot.store, "ส่งรายงานเสร็จแล้ว", guild_id=GUILD, is_exec=False, user_id=STAFF,
               model=llm, extra_tools=build_coo_tools(tools))  # fmt: skip
    names = {d.name for t_ in llm.requests[0].config.tools for d in t_.function_declarations}
    assert {"list_tasks", "update_task", "request_disclosure", "my_day"} <= names
    assert "ผู้ประสานงานของทีม" in llm.requests[0].config.system_instruction
    assert "request_disclosure" in llm.requests[0].config.system_instruction
    assert repo.get(GUILD, "abcd1234").status == "done" and a.text == "อัปเดตแล้วครับ"
    # nothing coordinator-related without the tools
    plain = scripted(text_resp("ok"))
    qa.ask(bot.s, bot.store, "hi", guild_id=GUILD, is_exec=False, user_id=STAFF, model=plain)
    assert "request_disclosure" not in plain.requests[0].config.system_instruction
    names = {d.name for t_ in plain.requests[0].config.tools for d in t_.function_declarations}
    assert not {"list_tasks", "update_task", "my_day"} & names


def test_persona_is_the_executive_assistant():
    for lang, needle in (
        ("th", "ผู้ช่วยผู้บริหาร"),
        ("en", "executive assistant"),
        ("auto", "executive assistant"),
    ):
        assert needle in qa.PERSONA[lang]
    assert "รายงานจากสลิด" in qa.CAPABILITIES_PROMPT and "เก็บเป็นความลับ" in qa.CAPABILITIES_PROMPT
