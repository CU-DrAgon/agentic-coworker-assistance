# ruff: noqa: F811, F401
"""Organisational memory: raw log, summaries, retention, access control, tools, bot wiring.
Offline: SQLite and the Firestore fake, scripted LLMs, small Discord fakes."""

import asyncio
import json
import time
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from conftest import (
    GUILD,
    FakeClient,
    FakeEmbedder,
    FixedClockSettings,
    fc_resp,
    json_resp,
    scripted,
    text_resp,
)
from fake_firestore import offline_memory_store
from test_bot_multitenant import ME, member
from test_modes import (  # noqa: F401  (fixtures)
    EXEC,
    MANAGER,
    STAFF,
    Chan,
    Interaction,
    TMsg,
    bot,
    send,
    set_mode,
    world,
)

from salid import memory_summary
from salid import qa as qa_mod
from salid.embed import unpack
from salid.guild_settings import (
    GuildSettings,
    GuildSettingsUpdate,
    MemoryGuildStore,
    default_retention_days,
)
from salid.i18n import t
from salid.memory import (
    MemoryScheduler,
    MemoryService,
    build_message,
    message_text,
    vectorize,
    vectors_usable,
)
from salid.memory_access import ChannelAccess, allowed_channel_ids
from salid.memory_index import IndexedMemoryStore
from salid.memory_store import DaySummary, SQLiteMemoryStore
from salid.memory_summary import summary_items
from salid.qa import MemoryTools, ask

BKK = ZoneInfo("Asia/Bangkok")


def bkk(day, hh, mm=0, ss=0):
    return datetime(2026, 10, day, hh, mm, ss, tzinfo=BKK)


@pytest.fixture(params=["sqlite", "firestore"])
def mstore(request, tmp_path):
    if request.param == "sqlite":
        return SQLiteMemoryStore(str(tmp_path / "m.db"))
    return offline_memory_store()


@pytest.fixture
def settings():
    return FixedClockSettings()


def mk(settings, mid, at, text="hello", cid=10, gid=GUILD, kind="chat", author="แอน", uid=1):
    return build_message(
        settings, guild_id=gid, channel_id=cid, message_id=mid, author_id=uid,
        author_name=author, created_at=at, text=text, jump_url=f"https://d/{cid}/{mid}", kind=kind,
    )  # fmt: skip


# ------------------------------------------------------------------ message building
def test_local_date_uses_the_configured_timezone(settings):
    late = mk(settings, 1, datetime(2026, 10, 14, 17, 30, tzinfo=UTC))  # 00:30 next day, Bangkok
    assert late.local_date == "2026-10-15" and late.created_at.tzinfo is not None
    assert mk(settings, 2, datetime(2026, 10, 14, 16, 59, tzinfo=UTC)).local_date == "2026-10-14"


def test_attachment_descriptors():
    atts = [
        SimpleNamespace(filename="receipt.jpg", content_type="image/jpeg"),
        SimpleNamespace(filename="a.pdf", content_type="application/pdf"),
    ]
    assert message_text("ดูนี่", atts) == "ดูนี่ [image: receipt.jpg] [file: a.pdf]"
    assert message_text("", []) == ""


# ------------------------------------------------------------------ store parity
def test_store_reads_never_leave_the_allowed_channels(mstore, settings):
    now = bkk(14, 10).astimezone(UTC)
    mstore.add_message(mk(settings, 1, now, "public งบโฆษณา", cid=10))
    mstore.add_message(mk(settings, 2, now + timedelta(minutes=1), "secret งบโฆษณา", cid=11))
    mstore.add_message(mk(settings, 3, now, "other guild", cid=10, gid=GUILD + 1))
    got = mstore.messages(GUILD, {10}, "2026-10-14", "2026-10-14")
    assert [m.text for m in got] == ["public งบโฆษณา"]
    assert mstore.messages(GUILD, set(), "2026-10-14", "2026-10-14") == []
    assert mstore.messages(GUILD, {10}, "2026-10-14", "2026-10-14", channel_id=11) == []
    both = mstore.messages(GUILD, {10, 11}, "2026-10-14", "2026-10-14", keyword="งบ")
    assert [m.message_id for m in both] == [2, 1]  # newest first
    assert (
        mstore.messages(GUILD, {10, 11}, "2026-10-14", "2026-10-14", keyword="SECRET")[0].channel_id
        == 11
    )
    assert mstore.messages(GUILD, {10, 11}, "2026-10-14", "2026-10-14", limit=1)[0].message_id == 2
    assert mstore.messages(GUILD, {10}, "2026-10-13", "2026-10-13") == []


def test_store_edit_delete_and_dirty_flag(mstore, settings):
    now = bkk(14, 10).astimezone(UTC)
    mstore.add_message(mk(settings, 1, now, "old"))
    mstore.add_message(mk(settings, 2, now, "keep"))
    mstore.put_summary(DaySummary(guild_id=GUILD, channel_id=10, local_date="2026-10-14",
                                  summary={"overview": "x"}, message_count=2))  # fmt: skip
    assert mstore.update_message_text(GUILD, 1, "new", now)
    assert not mstore.update_message_text(GUILD, 99, "x", now)
    assert mstore.messages(GUILD, {10}, "2026-10-14", "2026-10-14", keyword="new")[0].edited_at
    assert mstore.delete_messages(GUILD, [1, 77]) == [(10, "2026-10-14")]
    assert [m.message_id for m in mstore.messages(GUILD, {10}, "2026-10-14", "2026-10-14")] == [2]
    assert mstore.get_summary(GUILD, 10, "2026-10-14").dirty
    assert [s.channel_id for s in mstore.open_summaries(GUILD)] == [10]


def test_store_cursor_and_day_channels(mstore, settings):
    now = bkk(14, 10).astimezone(UTC)
    for i in range(1, 5):
        mstore.add_message(mk(settings, i, now + timedelta(seconds=i), f"m{i}", cid=10 + i % 2))
    assert mstore.day_channels(GUILD, "2026-10-14") == [10, 11]
    rows = mstore.messages_after(GUILD, 11, "2026-10-14")
    assert [m.message_id for m in rows] == [1, 3]
    assert [m.message_id for m in mstore.messages_after(GUILD, 11, "2026-10-14", rows[0].key)] == [
        3
    ]


def test_store_delete_guild(mstore, settings):
    now = bkk(14, 10).astimezone(UTC)
    mstore.add_message(mk(settings, 1, now))
    mstore.add_message(mk(settings, 2, now, gid=GUILD + 1))
    assert mstore.delete_guild(GUILD) == 1
    assert mstore.messages(GUILD, {10}, "2026-10-14", "2026-10-14") == []
    assert len(mstore.messages(GUILD + 1, {10}, "2026-10-14", "2026-10-14")) == 1


def test_purge_uses_each_guilds_current_retention(mstore, settings):
    now = datetime(2026, 10, 14, 5, tzinfo=UTC)
    for gid in (GUILD, GUILD + 1):
        for days_ago, mid in ((100, 1), (40, 2), (5, 3)):
            at = now - timedelta(days=days_ago)
            mstore.add_message(mk(settings, mid + gid, at, gid=gid))
            mstore.put_summary(DaySummary(guild_id=gid, channel_id=10,
                                          local_date=at.astimezone(BKK).date().isoformat(),
                                          message_count=1, final=True))  # fmt: skip
    ids = lambda g: sorted(m.message_id - g for m in mstore.messages(  # noqa: E731
        g, {10}, "2026-01-01", "2026-12-31", limit=50))  # fmt: skip
    cut = lambda days: now - timedelta(days=days)  # noqa: E731
    # guild A keeps 30 days, guild B keeps 365
    assert mstore.purge({GUILD: cut(30), GUILD + 1: cut(365)}, cut(90)) == (2, 2)
    assert ids(GUILD) == [3] and ids(GUILD + 1) == [1, 2, 3]
    assert len(mstore.summaries(GUILD, {10}, "2026-01-01", "2026-12-31")) == 1
    # shortening applies to data already stored; lengthening never resurrects deleted data
    assert mstore.purge({GUILD: cut(365), GUILD + 1: cut(7)}, cut(90)) == (2, 2)
    assert ids(GUILD) == [3] and ids(GUILD + 1) == [3]
    # a guild the bot does not know uses the default cutoff
    mstore.add_message(mk(settings, 50, cut(200), gid=GUILD + 2))
    assert mstore.purge({}, cut(90))[0] == 1


# ------------------------------------------------------------------ summaries
def summary_json(tag):
    return json_resp({"overview": f"overview {tag}", "topics": [{"title": "งบโฆษณา", "details": tag, "people": ["แอน"]}],
                      "decisions": [], "action_items": [], "numbers": [], "open_questions": []})  # fmt: skip


class Clock:
    def __init__(self, at):
        self.at = at

    def __call__(self):
        return self.at


def sched(mstore, settings, clock, llm, **kw):
    return MemoryScheduler(mstore, settings, llm, now=clock, **kw)


def log(svc, settings, mid, at, text=None, cid=10):
    svc.record(mk(settings, mid, at.astimezone(UTC), text or f"msg {mid}", cid=cid))


def prompts(llm):
    return [c[1] for c in llm.models.calls]


def test_update_triggers_on_count_and_on_quiet_time(mstore, settings):
    clock, llm = Clock(bkk(14, 10)), FakeClient([summary_json("a"), summary_json("b")])
    s = sched(mstore, settings, clock, llm)
    svc = MemoryService(mstore, settings, s)
    for i in range(29):
        log(svc, settings, i + 1, clock.at)
    assert s.tick([GUILD]) == 0 and llm.models.calls == []  # 29 < 30 and not quiet yet
    log(svc, settings, 30, clock.at)
    assert s.tick([GUILD]) == 1  # 30th message: count trigger
    first = mstore.get_summary(GUILD, 10, "2026-10-14")
    assert (
        first.message_count == 30 and not first.final and first.summary["overview"] == "overview a"
    )
    assert llm.models.calls[0][0] == settings.gate_model
    # 3 more messages: below the count, so wait for the quiet period
    for i in range(3):
        log(svc, settings, 100 + i, clock.at, f"late-{i}")
    clock.at += timedelta(minutes=19)
    assert s.tick([GUILD]) == 0
    clock.at += timedelta(minutes=2)
    assert s.tick([GUILD]) == 1
    second = prompts(llm)[1]
    assert "overview a" in second  # previous summary goes in
    assert all(f"late-{i}" in second for i in range(3))  # only the new messages ...
    assert "msg 5" not in second  # ... never the old ones again
    assert mstore.get_summary(GUILD, 10, "2026-10-14").message_count == 33
    clock.at += timedelta(minutes=30)
    assert s.tick([GUILD]) == 0  # nothing new: no model call


def test_at_most_three_updates_per_tick(mstore, settings):
    clock = Clock(bkk(14, 10))
    llm = FakeClient([summary_json(str(i)) for i in range(5)])
    s = sched(mstore, settings, clock, llm)
    svc = MemoryService(mstore, settings, s)
    for cid in range(10, 15):
        log(svc, settings, cid, clock.at, "x", cid=cid)
    clock.at += timedelta(minutes=21)
    assert s.tick([GUILD]) == 3 and s.tick([GUILD]) == 2 and s.tick([GUILD]) == 0


def test_day_end_finalises_once_and_never_rewrites(mstore, settings):
    clock, llm = Clock(bkk(14, 22)), FakeClient([summary_json("a"), summary_json("b")])
    s = sched(mstore, settings, clock, llm)
    svc = MemoryService(mstore, settings, s)
    log(svc, settings, 1, bkk(14, 22))
    clock.at = bkk(14, 22, 30)
    assert s.tick([GUILD]) == 1  # quiet update
    clock.at = bkk(14, 23, 50)
    log(svc, settings, 2, clock.at)  # arrives just before midnight
    clock.at = bkk(14, 23, 55)
    assert s.tick([GUILD]) == 0 and not mstore.get_summary(GUILD, 10, "2026-10-14").final
    clock.at = bkk(15, 0, 5)  # the local day ended
    assert s.tick([GUILD]) == 1
    done = mstore.get_summary(GUILD, 10, "2026-10-14")
    assert done.final and done.message_count == 2 and len(llm.models.calls) == 2
    s.note(GUILD, 10, "2026-10-14")  # a straggler: ignored, no rewrite
    clock.at = bkk(15, 2)
    assert s.tick([GUILD]) == 0 and len(llm.models.calls) == 2
    assert mstore.get_summary(GUILD, 10, "2026-10-14").summary["overview"] == "overview b"


def test_finalising_a_quiet_day_needs_no_model_call(mstore, settings):
    clock, llm = Clock(bkk(14, 10)), FakeClient([summary_json("a")])
    s = sched(mstore, settings, clock, llm)
    svc = MemoryService(mstore, settings, s)
    log(svc, settings, 1, clock.at)
    clock.at = bkk(14, 11)
    s.tick([GUILD])
    clock.at = bkk(15, 0, 1)
    s.tick([GUILD])
    assert mstore.get_summary(GUILD, 10, "2026-10-14").final and len(llm.models.calls) == 1


def test_deleting_a_message_regenerates_the_day_once(mstore, settings):
    clock = Clock(bkk(14, 10))
    llm = FakeClient([summary_json("a"), summary_json("regen")])
    s = sched(mstore, settings, clock, llm)
    svc = MemoryService(mstore, settings, s)
    log(svc, settings, 1, clock.at, "keep me")
    log(svc, settings, 2, clock.at + timedelta(seconds=1), "secret to delete")
    clock.at = bkk(14, 11)
    s.tick([GUILD])
    clock.at = bkk(15, 0, 5)
    s.tick([GUILD])
    assert mstore.get_summary(GUILD, 10, "2026-10-14").final
    assert svc.delete(GUILD, [2]) == 1
    assert mstore.get_summary(GUILD, 10, "2026-10-14").dirty
    assert s.tick([GUILD]) == 1  # regenerated from the remaining raw messages only
    redo = prompts(llm)[1]
    assert "keep me" in redo and "secret to delete" not in redo and "overview a" not in redo
    got = mstore.get_summary(GUILD, 10, "2026-10-14")
    assert got.final and not got.dirty and got.message_count == 1
    assert got.summary["overview"] == "overview regen"
    assert s.tick([GUILD]) == 0 and len(llm.models.calls) == 2  # and then final again


def test_a_failed_summary_call_is_retried_later(mstore, settings):
    clock = Clock(bkk(14, 10))
    llm = FakeClient([RuntimeError("boom"), summary_json("ok")])

    def boom(contents, config):
        raise RuntimeError("boom")

    llm.models.script[0] = boom
    s = sched(mstore, settings, clock, llm)
    svc = MemoryService(mstore, settings, s)
    log(svc, settings, 1, clock.at)
    clock.at += timedelta(minutes=21)
    assert s.tick([GUILD]) == 0 and mstore.get_summary(GUILD, 10, "2026-10-14") is None
    clock.at += timedelta(minutes=1)
    assert s.tick([GUILD]) == 0  # still backing off
    clock.at += timedelta(minutes=5)
    assert s.tick([GUILD]) == 1 and mstore.get_summary(GUILD, 10, "2026-10-14")


def test_scheduler_picks_up_unfinished_work_after_a_restart(mstore, settings):
    clock, llm = Clock(bkk(14, 10)), FakeClient([summary_json("a")])
    mstore.add_message(mk(settings, 1, bkk(14, 9).astimezone(UTC)))
    s = sched(mstore, settings, clock, llm)  # fresh scheduler: nothing noted in memory
    s.tick([GUILD])  # seeds from the store; the channel is not quiet yet
    clock.at += timedelta(minutes=21)
    assert s.tick([GUILD]) == 1


def test_scheduler_purges_once_a_day_by_guild_retention(mstore, settings):
    clock = Clock(bkk(14, 10))
    s = sched(mstore, settings, clock, FakeClient([]), retention_of=lambda gid: 30)
    mstore.add_message(mk(settings, 1, bkk(14, 9).astimezone(UTC) - timedelta(days=40)))
    mstore.add_message(mk(settings, 2, bkk(14, 9).astimezone(UTC) - timedelta(days=10)))
    s.tick([GUILD])
    assert [m.message_id for m in mstore.messages(GUILD, {10}, "2026-01-01", "2026-12-31")] == [2]
    assert s.maybe_purge(clock.at) is None  # already ran today


def test_summary_prompt_and_rendering():
    prompt = memory_summary.build_prompt({"overview": "prev"}, [], channel="ads", day="2026-10-14",
                                         lang="th", zone=BKK)  # fmt: skip
    assert "Thai" in prompt and "#ads" in prompt and "prev" in prompt and "do not invent" in prompt
    text = memory_summary.render_summary({
        "overview": "o", "topics": [{"title": "T", "details": "d", "people": ["a"]}],
        "decisions": ["dec"], "action_items": [{"who": "บอย", "what": "ส่งงาน", "due": "ศุกร์"}],
        "numbers": ["50,000 บาท"], "open_questions": ["q?"]})  # fmt: skip
    assert all(
        x in text
        for x in ("Topic: T (a): d", "Decision: dec", "บอย: ส่งงาน (due ศุกร์)", "50,000", "q?")
    )


# ------------------------------------------------------------------ access function
def acc(cid, *viewers):
    return ChannelAccess(cid, f"c{cid}", frozenset(viewers))


def test_access_asker_must_read_the_source_channel():
    chans = [acc(1, 7, 8), acc(2, 8)]
    assert allowed_channel_ids(chans, 7, None) == {1}  # DM: the asker's channels
    assert allowed_channel_ids(chans, 9, None) == set()


def test_access_wider_audience_cannot_see_a_private_source():
    private, public = acc(1, 7), acc(2, 7, 8, 9)
    chans = [private, public]
    assert allowed_channel_ids(chans, 7, 2) == {2}  # public answer: private source excluded
    assert allowed_channel_ids(chans, 7, 1) == {1, 2}  # private answer: everything 7 reads
    # same channel is always allowed, even when the asker is not in the viewer cache
    assert allowed_channel_ids(chans, 99, 2) == {2}
    assert allowed_channel_ids(chans, 7, 555) == set()  # unknown answering channel: nothing


def test_access_equal_audience_is_fine():
    chans = [acc(1, 7, 8), acc(2, 7, 8)]
    assert allowed_channel_ids(chans, 7, 2) == {1, 2}


# ------------------------------------------------------------------ discord adapter
class RChan(Chan):
    def __init__(self, id, name, readers):
        super().__init__(id, name)
        self.readers = set(readers)

    def permissions_for(self, m):
        ok = m.id in self.readers
        return SimpleNamespace(read_messages=ok, read_message_history=ok, send_messages=True)


@pytest.fixture
def rworld(world, bot):
    g = world.g
    g.text_channels[:] = [RChan(10, "ทั่วไป", {STAFF, MANAGER}), RChan(11, "ผู้บริหาร", {EXEC}),
                          RChan(12, "ads", {STAFF, MANAGER, EXEC})]  # fmt: skip
    for c in g.text_channels:
        c.guild = g
    for uid in (STAFF, MANAGER, EXEC):
        member(g, uid)
    return SimpleNamespace(g=g, staff=g.members[STAFF], boss=g.members[EXEC])


def scope(bot, w, who, answering):
    bot._viewers.clear()
    return bot._memory_scope(w.g, who, answering)


def test_adapter_applies_permissions_audience_and_modes(bot, rworld):
    w = rworld
    assert scope(bot, w, w.staff, None) == {10: "ทั่วไป", 12: "ads"}  # DM: what staff can read
    assert set(scope(bot, w, w.boss, None)) == {11, 12}
    # answering in #ทั่วไป (staff+manager can read): #ads has more readers than staff? no: audience
    # {STAFF, MANAGER} is inside #ads's readers, so #ads is allowed; #ผู้บริหาร is not readable
    assert set(scope(bot, w, w.staff, 10)) == {10, 12}
    assert set(scope(bot, w, w.boss, 10)) == {10, 12}  # boss cannot read 10: only the same channel
    set_mode(bot, 12, "off")
    assert set(scope(bot, w, w.staff, None)) == {10}  # off channels are never remembered


async def test_memory_tools_for_several_answering_channels_intersect(bot, rworld):
    w = rworld
    bot._viewers.clear()
    wide = await bot._memory_tools_for(w.g, w.boss, 11)
    both = await bot._memory_tools_for(w.g, w.boss, 11, 12)
    assert set(wide.allowed) == {11, 12} and set(both.allowed) == {12}


# ------------------------------------------------------------------ stale reader cache
def stale_world(rworld):
    """#hr {STAFF, MANAGER} is a private source; #ทั่วไป {STAFF, MANAGER} is where the answer goes."""
    w = rworld
    hr = RChan(13, "hr", {STAFF, MANAGER})
    hr.guild = w.g
    w.g.text_channels.append(hr)
    w.hr, w.gen = hr, w.g.text_channels[0]
    return w


def test_answer_channel_widening_inside_the_ttl_excludes_the_private_source(bot, rworld):
    w = stale_world(rworld)
    bot._viewers.clear()
    assert set(bot._memory_scope(w.g, w.staff, 10)) == {10, 12, 13}  # primes the cache
    w.gen.readers.add(EXEC)  # the answer channel just opened up to the boss; no event seen yet
    assert 13 not in bot._memory_scope(w.g, w.staff, 10)  # live audience: hr is private
    assert set(bot._memory_scope(w.g, w.staff, 10)) == {10, 12}


async def test_source_narrowing_is_picked_up_by_a_member_update(bot, rworld):
    w = stale_world(rworld)
    bot._viewers.clear()
    assert 12 in bot._memory_scope(w.g, w.staff, 10)  # #ads readers {STAFF, MANAGER, EXEC}
    w.g.text_channels[2].readers.discard(MANAGER)  # the manager lost access to #ads
    old = SimpleNamespace(id=MANAGER, guild=w.g, roles=["a"])
    new = SimpleNamespace(id=MANAGER, guild=w.g, roles=[])
    await bot.on_member_update(old, new)
    assert 12 not in bot._memory_scope(w.g, w.staff, 10)  # audience {S,M} no longer inside


async def test_room_scope_uses_the_live_voice_chat_audience(bot, rworld):
    w = stale_world(rworld)
    bot._viewers.clear()
    first = await bot._room_memory_tools(w.g, w.gen, [w.staff], asker=w.staff)
    assert 13 in first.allowed
    w.gen.readers.add(EXEC)  # the voice chat opens up inside the TTL
    again = await bot._room_memory_tools(w.g, w.gen, [w.staff], asker=w.staff)
    assert 13 not in again.allowed and 10 in again.allowed


def _ev(guild, **kw):
    return SimpleNamespace(guild=guild, **kw)


@pytest.mark.parametrize(
    "event",
    ["channel_update", "category_update", "member_update", "role_update", "role_delete",
     "member_join", "member_remove", "channel_delete"],
)  # fmt: skip
async def test_permission_events_drop_the_cached_viewers(bot, rworld, event):
    w = rworld
    g = w.g
    g.me = SimpleNamespace(roles=[])
    bot._viewers.clear()
    bot._memory_scope(g, w.staff, 10)
    assert bot._viewers
    chan = g.text_channels[0]
    cat = _ev(g, id=900, name="c", overwrites={"a": 1}, channels=[chan])
    chan.category = cat
    if event == "channel_update":
        await bot.on_guild_channel_update(
            _ev(g, id=10, name="x", overwrites={}), _ev(g, id=10, name="x", overwrites={"a": 1})
        )
    elif event == "category_update":
        await bot.on_guild_channel_update(
            _ev(g, id=900, name="c", overwrites={}, channels=[chan]), cat
        )
    elif event == "member_update":
        await bot.on_member_update(
            SimpleNamespace(id=STAFF, guild=g, roles=[]),
            SimpleNamespace(id=STAFF, guild=g, roles=["r"]),
        )
    elif event == "role_update":
        await bot.on_guild_role_update(_ev(g, permissions=1), _ev(g, permissions=2))
    elif event == "role_delete":
        await bot.on_guild_role_delete(_ev(g, id=5))
    elif event == "member_join":
        await bot.on_member_join(SimpleNamespace(id=STAFF, guild=g, bot=True))
    elif event == "member_remove":
        await bot.on_member_remove(SimpleNamespace(id=STAFF, guild=g, bot=True))
    elif event == "channel_delete":
        await bot.on_guild_channel_delete(_ev(g, id=10))
    if event in ("channel_update", "category_update", "channel_delete"):
        assert (GUILD, 10) not in bot._viewers and (GUILD, 11) in bot._viewers  # just that one
    else:
        assert not bot._viewers  # the whole guild


async def test_unrelated_events_keep_the_cache(bot, rworld):
    w = rworld
    bot._viewers.clear()
    bot._memory_scope(w.g, w.staff, 10)
    await bot.on_guild_channel_update(  # a rename only
        _ev(w.g, id=10, name="a", overwrites={}), _ev(w.g, id=10, name="b", overwrites={})
    )
    assert bot._viewers


# ------------------------------------------------------------------ tools
def seeded_tools(mstore, settings, allowed, now=None):
    now = now or bkk(14, 9).astimezone(UTC)
    mstore.add_message(mk(settings, 1, now, "งบโฆษณาเดือนนี้ 50,000 บาท", cid=10, author="แอน"))
    mstore.add_message(
        mk(settings, 2, now + timedelta(minutes=5), "ลองใช้ ads ใหม่", cid=10, author="บอย")
    )
    mstore.add_message(
        mk(settings, 3, now + timedelta(minutes=6), "ads ลับของผู้บริหาร", cid=11, author="เจ้านาย")
    )
    mstore.add_message(mk(settings, 4, now - timedelta(days=3), "ads old", cid=10))
    return MemoryTools(mstore, GUILD, allowed, settings, retention_days=30)


def test_tools_never_return_channels_outside_the_scope(mstore, settings):
    tools = seeded_tools(mstore, settings, {10: "general"})
    for fn, args in (
        (tools.chat_messages, ("2026-10-01", "2026-10-14")),
        (tools.chat_search, (["ads"],)),
        (tools.chat_days, ("2026-10-14", "2026-10-14")),
    ):
        assert "ลับ" not in json.dumps(fn(*args), ensure_ascii=False)
    # the model names a channel it cannot see
    r = tools.call(
        "chat_messages",
        {"start_date": "2026-10-01", "end_date": "2026-10-14", "channel_name": "ผู้บริหาร"},
    )
    assert r["count"] == 0 and "no channel" in r["note"]
    r = tools.call(
        "chat_messages",
        {"start_date": "2026-10-01", "end_date": "2026-10-14", "allowed": [11], "channel_id": 11},
    )
    assert {m["channel"] for m in r["messages"]} == {"general"}  # undeclared args are dropped
    assert tools.call("chat_days", {"start_date": "x", "end_date": "y"})["error"]
    assert tools.call("drop", {})["error"]


def test_chat_messages_keyword_order_and_links(mstore, settings):
    tools = seeded_tools(mstore, settings, {10: "general", 11: "exec"})
    r = tools.chat_messages("2026-10-01", "2026-10-14", keyword="ADS")
    assert [m["author"] for m in r["messages"]][:2] == ["เจ้านาย", "บอย"]  # newest first
    assert r["messages"][0]["link"] == "https://d/11/3" and r["messages"][0]["time"].startswith(
        "2026-10-14 "
    )
    assert (
        tools.chat_messages("2026-10-01", "2026-10-14", channel_name="#GENERAL", keyword="ads")[
            "count"
        ]
        == 2
    )


def test_chat_days_returns_summaries_and_todays_raw_tail(mstore, settings):
    tools = seeded_tools(mstore, settings, {10: "general"})
    cursor_msg = mstore.messages_after(GUILD, 10, "2026-10-14")[0]  # the 09:00 message
    mstore.put_summary(DaySummary(
        guild_id=GUILD, channel_id=10, local_date="2026-10-14", message_count=1,
        summary={"overview": "คุยเรื่องงบ", "topics": [], "decisions": ["อนุมัติ"]},
        last_message_at=cursor_msg.key[0], last_message_id=cursor_msg.message_id))  # fmt: skip
    r = tools.chat_days("2026-10-14", "2026-10-14")
    assert r["days"][0]["channel"] == "general" and "อนุมัติ" in r["days"][0]["summary"]
    # only what is newer than the summary comes as raw messages
    assert [m["text"] for m in r["newer_than_summary"]] == ["ลองใช้ ads ใหม่"]
    empty = tools.chat_days("2026-10-01", "2026-10-02")
    assert empty["count"] == 0 and "nothing recorded" in empty["note"]
    assert "error" in tools.call(
        "chat_days", {"start_date": "2026-01-01", "end_date": "2026-12-31"}
    )


def test_firestore_messages_after_filters_the_channel_server_side(settings):
    st = offline_memory_store()
    st.add_message(mk(settings, 1, bkk(14, 9).astimezone(UTC), cid=10))
    st.add_message(mk(settings, 2, bkk(14, 9).astimezone(UTC), cid=11))
    st.client.streams.clear()
    assert [m.message_id for m in st.messages_after(GUILD, 10, "2026-10-14")] == [1]
    assert ("channel_id", "==", 10) in st.client.streams[-1][1]


def test_chat_days_reads_each_tail_day_once(settings):
    st = offline_memory_store()
    for i, cid in enumerate((10, 11, 12) * 2):
        day = 13 + i // 3
        st.add_message(mk(settings, i + 1, bkk(day, 9, i).astimezone(UTC), "x", cid=cid))
    tools = MemoryTools(st, GUILD, {10: "a", 11: "b", 12: "c"}, settings)
    st.client.streams.clear()
    r = tools.chat_days("2026-10-13", "2026-10-14")
    msg_reads = [f for name, f in st.client.streams if name == "salid_messages"]
    assert len(msg_reads) == 2 and len(r["newer_than_summary"]) == 6


def test_chat_days_output_is_capped_newest_days_first(mstore, settings):
    tools = MemoryTools(mstore, GUILD, {10: "general"}, settings)
    for d in range(1, 15):
        mstore.put_summary(DaySummary(guild_id=GUILD, channel_id=10, local_date=f"2026-10-{d:02d}",
                                      message_count=5, final=True, summary={"overview": "x" * 2400}))  # fmt: skip
    r = tools.chat_days("2026-10-01", "2026-10-14")
    assert len(json.dumps(r, ensure_ascii=False)) < 7500 and "narrow" in r["note"]
    dates = [d["date"] for d in r["days"]]
    assert "2026-10-14" in dates and "2026-10-01" not in dates


def test_chat_search_ranks_summary_hits_higher_and_matches_any_keyword(mstore, settings):
    tools = MemoryTools(mstore, GUILD, {10: "general", 12: "ads"}, settings, retention_days=30)
    mstore.put_summary(DaySummary(guild_id=GUILD, channel_id=12, local_date="2026-10-10", message_count=3,
        summary={"overview": "", "topics": [{"title": "งบโฆษณา", "details": "ค่า ads 20,000", "people": ["แอน"]}],
                 "decisions": [], "action_items": [], "numbers": [], "open_questions": []}))  # fmt: skip
    mk_m = lambda mid, day, text, cid: mstore.add_message(  # noqa: E731
        mk(settings, mid, bkk(day, 10).astimezone(UTC), text, cid=cid))  # fmt: skip
    mk_m(1, 13, "มี ads ใหม่", 10)
    mk_m(2, 12, "ads และ งบโฆษณา เหลือน้อย", 10)  # two keywords in one message
    mk_m(3, 11, "ไม่เกี่ยว", 10)
    r = tools.chat_search(["งบโฆษณา", "ADS", "marketing budget"])
    keys = [(h["date"], h["channel"]) for h in r["hits"]]
    assert keys == [("2026-10-10", "ads"), ("2026-10-12", "general"), ("2026-10-13", "general")]
    top = r["hits"][0]
    assert top["score"] == 6 and top["summary_hits"] and top["matching_messages"] == 0
    assert r["hits"][1]["score"] == 2 > r["hits"][2]["score"] == 1
    assert r["hits"][1]["messages"][0]["link"] == "https://d/10/2"
    assert tools.chat_search(["nothing-here"])["hits"] == []
    assert "error" in tools.chat_search([" ", ""])
    # date range: defaults to the retention window, explicit range narrows it
    assert [h["date"] for h in tools.chat_search(["ads"], "2026-10-13", "2026-10-13")["hits"]] == [
        "2026-10-13"
    ]


def test_chat_search_ties_go_newest_first_and_limits_groups(mstore, settings):
    tools = MemoryTools(mstore, GUILD, {10: "g"}, settings)
    for d in (12, 13, 14):  # inside the raw window (last 3 days)
        mstore.add_message(mk(settings, d, bkk(d, 10).astimezone(UTC), "ads", cid=10))
    assert [h["date"] for h in tools.chat_search(["ads"])["hits"]] == [
        "2026-10-14",
        "2026-10-13",
        "2026-10-12",
    ]
    assert len(tools.chat_search(["ads"], limit=2)["hits"]) == 2


def test_chat_search_output_is_capped(mstore, settings):
    tools = MemoryTools(mstore, GUILD, {10: "g"}, settings)
    long = "ads " + "ก" * 450
    for d in range(1, 15):
        mstore.put_summary(DaySummary(guild_id=GUILD, channel_id=10, local_date=f"2026-10-{d:02d}",
                                      message_count=4, final=True,
                                      summary={"overview": long, "decisions": [long] * 6}))  # fmt: skip
    r = tools.chat_search(["ads"], limit=50)
    assert len(json.dumps(r, ensure_ascii=False)) <= 4500 and "truncated" in r["note"]
    assert all(len(h["messages"]) <= 3 for h in r["hits"])


def test_old_raw_only_days_point_to_chat_messages(mstore, settings):
    tools = MemoryTools(mstore, GUILD, {10: "g"}, settings, retention_days=30)
    mstore.add_message(mk(settings, 1, bkk(5, 10).astimezone(UTC), "ads", cid=10))
    r = tools.chat_search(["ads"])
    assert r["hits"] == [] and "chat_messages" in r["note"]


# ------------------------------------------------------------------ hybrid search
def topic_day(st, day, cid, title, details=""):
    """A final day summary with one topic, vectorised with the fake embedder."""
    summary = {"overview": "", "topics": [{"title": title, "details": details, "people": []}]}
    v = vectorize(summary, None, FakeEmbedder())
    st.put_summary(DaySummary(guild_id=GUILD, channel_id=cid, local_date=day, message_count=3,
                              final=True, summary=summary, vec_model=v.model_id,
                              vec_hashes=v.hashes, vecs=v.data))  # fmt: skip


def hybrid_tools(mstore, settings, allowed=None, emb=None):
    st = IndexedMemoryStore(mstore, emb or FakeEmbedder())
    return st, MemoryTools(st, GUILD, allowed or {10: "general", 12: "ads"}, settings,
                           retention_days=30, embedder=emb or FakeEmbedder())  # fmt: skip


def test_english_query_finds_a_thai_only_day(mstore, settings):
    st, tools = hybrid_tools(mstore, settings)
    topic_day(st, "2026-10-05", 12, "งบโฆษณา", "ค่าโฆษณาเฟซบุ๊ก 20,000")
    topic_day(st, "2026-10-06", 10, "ซื้อน้ำแข็ง")
    r = tools.chat_search(["marketing budget"], query="what is our marketing budget?")
    assert [(h["date"], h["channel"], h["match"]) for h in r["hits"]] == [
        ("2026-10-05", "ads", "meaning")
    ]
    assert "งบโฆษณา" in r["hits"][0]["summary_hits"][0]


def test_thai_query_finds_an_english_only_day(mstore, settings):
    st, tools = hybrid_tools(mstore, settings)
    topic_day(st, "2026-10-05", 10, "Facebook ads plan")
    assert tools.chat_search(["งบโฆษณา"], query="งบโฆษณาเท่าไร")["hits"][0]["date"] == "2026-10-05"


def test_keyword_and_meaning_together_rank_first(mstore, settings):
    st, tools = hybrid_tools(mstore, settings)
    topic_day(st, "2026-10-03", 10, "Q4 ads plan")  # keyword "ads" and meaning
    topic_day(st, "2026-10-04", 10, "โฆษณา")  # meaning only (newer, so it wins the meaning tie)
    r = tools.chat_search(["ads"], query="ads")
    # RRF: 10-03 = 1/61 + 1/62 beats 10-04 = 1/61
    assert [(h["date"], h["match"]) for h in r["hits"]] == [
        ("2026-10-03", "both"),
        ("2026-10-04", "meaning"),
    ]


def test_embedding_failure_falls_back_to_keywords(mstore, settings):
    st, tools = hybrid_tools(mstore, settings, emb=FakeEmbedder(fail=True))
    topic_day(st, "2026-10-05", 12, "งบโฆษณา")
    r = tools.chat_search(["งบโฆษณา"], query="marketing budget")
    assert [h["match"] for h in r["hits"]] == ["keyword"] and "error" not in r


def test_semantic_false_makes_no_embedding_call(mstore, settings):
    emb = FakeEmbedder()
    st, tools = hybrid_tools(mstore, settings, emb=emb)
    topic_day(st, "2026-10-05", 12, "งบโฆษณา")
    tools.chat_search(["งบโฆษณา"], semantic=False)
    assert not [c for c in emb.calls if c[1] == "query"]


def test_meaning_hits_never_leave_the_allowed_channels(mstore, settings):
    st, tools = hybrid_tools(mstore, settings, allowed={10: "general"})
    topic_day(st, "2026-10-05", 11, "งบโฆษณาลับ")
    assert tools.chat_search(["marketing"], query="marketing budget")["hits"] == []


def test_stale_vectors_are_ignored_not_crashed_on(mstore, settings):
    st, tools = hybrid_tools(mstore, settings)
    topic_day(st, "2026-10-05", 12, "งบโฆษณา")
    st.put_summary_vectors(GUILD, 12, "2026-10-05", "other@8", ["x"], b"\0" * 32)
    assert tools.chat_search(["marketing"], query="marketing")["hits"] == []


def test_query_only_search_works(mstore, settings):
    st, tools = hybrid_tools(mstore, settings)
    topic_day(st, "2026-10-05", 12, "งบโฆษณา")
    assert tools.chat_search([], query="marketing budget")["hits"][0]["date"] == "2026-10-05"
    assert "error" in tools.chat_search([], query=" ")


def test_chat_search_query_budget_on_firestore(settings):
    base = offline_memory_store()
    st, tools = hybrid_tools(base, settings, allowed={10: "general", 11: "b"})
    start = date(2026, 10, 14)
    for n in range(90):
        day = (start - timedelta(days=n)).isoformat()
        for cid in (10, 11):
            base.put_summary(DaySummary(guild_id=GUILD, channel_id=cid, local_date=day,
                                        message_count=2, final=n > 1,
                                        summary={"overview": "ads" if n % 7 == 0 else "x"}))  # fmt: skip
            at = datetime.fromisoformat(day).replace(hour=3, tzinfo=UTC)
            for i in range(2):
                base.add_message(
                    mk(settings, n * 100 + cid * 2 + i, at + timedelta(minutes=i), "ads", cid=cid)
                )
    base.client.streams.clear()
    assert tools.chat_search(["ads"], query="ads")["hits"]
    assert len(base.client.streams) <= 10
    base.client.streams.clear()
    tools.chat_days("2026-10-08", "2026-10-14")
    assert len(base.client.streams) <= 3


def test_ask_wires_the_memory_tools_and_prompt(mstore, settings, tmp_path):
    from salid.store import SQLiteStore

    tools = seeded_tools(mstore, settings, {10: "general"})
    llm = scripted(fc_resp(("chat_search", {"keywords": ["ads", "งบโฆษณา"]})), text_resp("พบครับ"))
    a = ask(settings, SQLiteStore(str(tmp_path / "l.db")), "งบโฆษณาเหลือเท่าไหร่", guild_id=GUILD,
            is_exec=False, user_id=7, model=llm, memory=tools)  # fmt: skip
    decls = {d.name: d for t_ in llm.requests[0].config.tools for d in t_.function_declarations}
    assert {"chat_days", "chat_messages", "chat_search", "spending_summary"} <= set(decls)
    search_params = decls["chat_search"].parameters_json_schema["properties"]
    assert "query" in search_params and "semantic" not in search_params
    prompt = llm.requests[0].config.system_instruction
    assert "chat_search" in prompt and "ความจำของทีม" in prompt and "query" in prompt
    blob = json.dumps(
        [
            str(p.function_response)
            for c in llm.requests[1].contents
            for p in c.parts or []
            if p.function_response
        ],
        ensure_ascii=False,
    )
    assert a.text == "พบครับ" and "50,000" in blob and "ลับ" not in blob
    # without memory tools the prompt and tools do not mention them
    llm2 = scripted(text_resp("ok"))
    ask(
        settings,
        SQLiteStore(str(tmp_path / "l.db")),
        "hi",
        guild_id=GUILD,
        is_exec=True,
        user_id=7,
        model=llm2,
    )
    assert "chat_search" not in llm2.requests[0].config.system_instruction


# ------------------------------------------------------------------ bot: recording
def recorded(bot, channel_ids=(10,), day="2026-10-14"):
    return bot.memory_store.messages(
        GUILD, set(channel_ids), day, day, limit=100, newest_first=False
    )


async def test_modes_decide_what_is_recorded(bot, world):
    set_mode(bot, 10, "off")
    await send(bot, world, world.gen, "private chat")
    await send(bot, world, world.named, "in a call-mode channel")
    for mode in ("listen", "call", "join", "chat"):
        set_mode(bot, 10, mode)
        await send(bot, world, world.gen, f"in {mode}")
    await bot.drain_memory()
    assert [m.text for m in recorded(bot)] == ["in listen", "in call", "in join", "in chat"]
    assert [m.text for m in recorded(bot, (12,))] == ["in a call-mode channel"]
    assert all(
        m.kind == "chat" and m.author_name == "u4" and m.author_id == STAFF for m in recorded(bot)
    )


async def test_message_fields_and_attachments(bot, world):
    from test_bot_multitenant import photo

    m = await send(bot, world, world.gen, "ใบเสร็จวันนี้", attachments=[photo()])
    m.jump_url = "https://discord.com/channels/1/10/5"
    await bot.drain_memory()
    (rec,) = recorded(bot)
    assert rec.text == "ใบเสร็จวันนี้ [image: r.jpg]" and rec.channel_id == 10 and rec.thread_id is None
    assert rec.local_date == "2026-10-14" and rec.message_id == m.id


async def test_salids_own_replies_are_recorded_as_salid(bot, world):
    own = TMsg(world.g, world.gen, ME, "ตอบแล้วครับ")
    await bot.on_message(own)
    await bot.drain_memory()
    (rec,) = recorded(bot)
    assert (rec.kind, rec.author_name, rec.text) == ("salid", "Salid", "ตอบแล้วครับ")


async def test_other_bots_are_not_recorded(bot, world):
    author = member(world.g, 77)
    author.bot = True
    await bot.on_message(TMsg(world.g, world.gen, author, "beep"))
    await bot.drain_memory()
    assert recorded(bot) == []


async def test_threads_use_the_parent_channel(bot, world):
    thread = Chan(500, "thread-x")
    thread.parent_id, thread.parent, thread.guild = 12, world.named, world.g
    await send(bot, world, thread, "in a thread")
    private = Chan(501, "secret-thread")
    private.parent_id, private.parent, private.guild = 12, world.named, world.g
    private.is_private = lambda: True
    await send(bot, world, private, "private thread")
    set_mode(bot, 12, "off")
    await send(bot, world, thread, "parent is off")
    await bot.drain_memory()
    (rec,) = recorded(bot, (12, 500))
    assert (rec.channel_id, rec.thread_id, rec.text) == (12, 500, "in a thread")


async def test_the_memory_feature_flag_turns_recording_off(bot, world):
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features=dict(memory=False)), "x")
    bot.team._cache.clear()
    await send(bot, world, world.gen, "nothing kept")
    await bot.drain_memory()
    assert recorded(bot) == []
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features=dict(memory=True)), "x")
    bot.team._cache.clear()
    assert await bot._memory_tools(
        TMsg(world.g, world.gen, member(world.g, STAFF)), world.g, member(world.g, STAFF)
    )


async def test_edit_and_delete_events(bot, world):
    a = await send(bot, world, world.gen, "first")
    b = await send(bot, world, world.gen, "second")
    await bot.drain_memory()
    edit = SimpleNamespace(guild_id=GUILD, message_id=a.id, data={
        "content": "first (edited)", "attachments": [{"filename": "x.png", "content_type": "image/png"}],
        "edited_timestamp": "2026-10-14T05:00:00+00:00"})  # fmt: skip
    await bot.on_raw_message_edit(edit)
    await bot.on_raw_message_edit(
        SimpleNamespace(guild_id=GUILD, message_id=b.id, data={"embeds": []})
    )
    await bot.on_raw_message_edit(
        SimpleNamespace(guild_id=None, message_id=1, data={"content": "dm"})
    )
    await bot.drain_memory()
    first, second = recorded(bot)
    assert (
        first.text == "first (edited) [image: x.png]"
        and first.edited_at
        and second.text == "second"
    )
    bot.memory_store.put_summary(
        DaySummary(
            guild_id=GUILD, channel_id=10, local_date="2026-10-14", message_count=2, final=True
        )
    )
    await bot.on_raw_message_delete(SimpleNamespace(guild_id=GUILD, message_id=a.id))
    await bot.drain_memory()
    assert [m.text for m in recorded(bot)] == ["second"]
    assert bot.memory_store.get_summary(GUILD, 10, "2026-10-14").dirty
    await bot.on_raw_bulk_message_delete(SimpleNamespace(guild_id=GUILD, message_ids={b.id, 123}))
    await bot.drain_memory()
    assert recorded(bot) == []


async def test_recording_failure_never_breaks_the_handler(bot, world, monkeypatch):
    def boom(msg):
        raise RuntimeError("store down")

    monkeypatch.setattr(bot.memory, "record", boom)
    m = await send(bot, world, world.gen, "สลิด ยอดเท่าไร")
    await bot.drain_memory()
    assert bot.calls == [("answer", "สลิด ยอดเท่าไร")] and m.replies == ["ans"]


async def test_the_message_is_logged_before_any_reply_work(bot, world):
    order = []
    orig = bot._memory_enqueue
    bot._memory_enqueue = lambda job: (order.append("logged"), orig(job))[1]

    async def slow_answer(m, guild, who):
        order.append("answering")

    bot._answer = slow_answer
    await send(bot, world, world.gen, "สลิด ช่วยหน่อย")
    assert order == ["logged", "answering"]


async def test_voice_segments_are_logged_as_voice(bot, world):
    voice = Chan(77, "ห้องประชุม")
    voice.guild = world.g
    at = datetime(2026, 10, 14, 5, 0, tzinfo=UTC)
    await bot.record_voice(voice, STAFF, "แอน", "เลื่อนประชุมเป็นวันศุกร์", at)
    await bot.record_voice(voice, STAFF, "แอน", "   ", at)
    set_mode(bot, 77, "off")
    await bot.record_voice(voice, STAFF, "แอน", "ไม่ควรเก็บ", at)
    await bot.drain_memory()
    (rec,) = recorded(bot, (77,))
    assert (rec.kind, rec.channel_id, rec.author_name) == ("voice", 77, "แอน")
    assert rec.text == "เลื่อนประชุมเป็นวันศุกร์" and "77" in rec.jump_url


# ------------------------------------------------------------------ notices, /memory
async def test_switching_a_channel_on_posts_a_one_time_notice(bot, world):
    set_mode(bot, 10, "off")
    await send(bot, world, world.gen, "hi")
    assert world.gen.sent == []
    it = Interaction(world.g, member(world.g, MANAGER, manage_guild=True), world.gen)
    await bot.handle_mode(it, "call")
    assert world.gen.sent == [t("memory_notice", "th", days=90)] and "90" in world.gen.sent[0]
    await bot.handle_mode(it, "join")  # on -> on: no second notice
    assert len(world.gen.sent) == 1
    await bot.handle_mode(it, "off")
    await bot.handle_mode(it, "chat")
    assert len(world.gen.sent) == 2


async def test_a_hub_change_from_off_is_noticed_on_the_next_message(bot, world):
    set_mode(bot, 10, "off")
    await send(bot, world, world.gen, "hi")
    set_mode(bot, 10, "call")  # changed in the hub: the bot only sees it later
    await send(bot, world, world.gen, "hello")
    await send(bot, world, world.gen, "again")
    assert world.gen.sent == [t("memory_notice", "th", days=90)]


async def test_notice_shows_the_guilds_retention_days(bot, world):
    bot.guilds_store.set_memory_retention_days(GUILD, 30)
    bot.team._cache.clear()
    set_mode(bot, 10, "off")
    await send(bot, world, world.gen, "hi")
    await bot.handle_mode(
        Interaction(world.g, member(world.g, MANAGER, manage_guild=True), world.gen), "call"
    )
    assert "30" in world.gen.sent[0] and "90" not in world.gen.sent[0]
    it = Interaction(world.g, member(world.g, STAFF), world.gen)
    await bot.handle_mode(it, None)
    assert "30 วัน" in it.sent[0][0]


def test_mode_help_and_welcome_mention_memory():
    for lang in ("th", "en"):
        assert t("mode_help_off", lang, days=7) and "7" in t("mode_help_chat", lang, days=7)
        assert "7" in t("welcome", lang, mention="@S", chat="#c", days=7)
        assert "7" in t("memory_notice", lang, days=7)


async def test_memory_command_status_and_permissions(bot, world):
    it = Interaction(world.g, member(world.g, STAFF), world.gen)
    await bot.handle_memory(it, None)
    ((text, ephemeral),) = it.sent
    assert ephemeral and "90" in text and t("memory_on", "th") in text
    it2 = Interaction(world.g, member(world.g, STAFF, roles=["พนักงาน"]), world.gen)
    await bot.handle_memory(it2, 30)
    assert it2.sent == [(t("memory_denied", "th"), True)]
    assert bot.team.memory_retention_days(GUILD) == 90
    boss = Interaction(world.g, member(world.g, MANAGER, manage_guild=True), world.gen)
    await bot.handle_memory(boss, 30)
    assert boss.sent == [(t("memory_set", "th", days=30), False)]  # public confirmation
    assert bot.team.memory_retention_days(GUILD) == 30
    for bad in (0, 3651, -5):
        bad_it = Interaction(world.g, member(world.g, MANAGER, manage_guild=True), world.gen)
        await bot.handle_memory(bad_it, bad)
        assert bad_it.sent[0][1] is True and bad_it.sent[0][0] == t(
            "memory_bad_days", "th", lo=1, hi=3650
        )
    assert bot.team.memory_retention_days(GUILD) == 30
    exec_it = Interaction(world.g, member(world.g, EXEC), world.gen)
    await bot.handle_memory(exec_it, 365)  # executives (env id) may change it too
    assert bot.team.memory_retention_days(GUILD) == 365
    bot.guilds_store.save_settings(
        GUILD, GuildSettingsUpdate(features=dict(memory=False), memory_retention_days=365), "x"
    )
    bot.team._cache.clear()
    off = Interaction(world.g, member(world.g, STAFF), world.gen)
    await bot.handle_memory(off, None)
    assert t("memory_off", "th") in off.sent[0][0] and "365" in off.sent[0][0]


# ------------------------------------------------------------------ retention setting
@pytest.mark.parametrize("value, expected", [(30, 30), (1, 1), (3650, 3650), ("180", 180), (365.0, 365),
                                              (0, 90), (3651, 90), (-1, 90), ("abc", 90), (None, 90),
                                              (True, 90), (12.5, 90), ([7], 90)])  # fmt: skip
def test_retention_validation_bounds(value, expected):
    assert GuildSettingsUpdate(memory_retention_days=value).memory_retention_days == expected
    assert GuildSettings().memory_retention_days == default_retention_days() == 90


def test_retention_env_default(monkeypatch):
    monkeypatch.setenv("SALID_MEMORY_RETENTION_DAYS", "45")
    assert GuildSettingsUpdate().memory_retention_days == 45
    monkeypatch.setenv("SALID_MEMORY_RETENTION_DAYS", "99999")
    assert GuildSettingsUpdate().memory_retention_days == 90


def test_retention_is_stored_per_guild():
    store = MemoryGuildStore()
    store.save_settings(GUILD, GuildSettingsUpdate(memory_retention_days=30), "x")
    store.save_settings(GUILD + 1, GuildSettingsUpdate(), "x")
    assert (
        store.get(GUILD).memory_retention_days == 30
        and store.get(GUILD + 1).memory_retention_days == 90
    )
    store.set_memory_retention_days(GUILD, 7)
    assert store.get(GUILD).memory_retention_days == 7
    with pytest.raises(ValueError):
        store.set_memory_retention_days(GUILD, 0)


async def test_purge_follows_the_setting_changed_after_the_data_was_stored(bot, world, tmp_path):
    old = bkk(14, 9).astimezone(UTC) - timedelta(days=60)
    bot.memory_store.add_message(mk(bot.s, 1, old))
    bot.team.set_memory_retention_days(GUILD, 30)  # shortened: the 60-day-old message goes
    sch = bot.memory.scheduler
    sch._known_guilds.add(GUILD)
    sch.maybe_purge(bkk(14, 10))
    assert bot.memory_store.messages(GUILD, {10}, "2026-01-01", "2026-12-31") == []


# ------------------------------------------------------------------ deleted / expired content
@pytest.mark.parametrize("indexed", [True, False])
def test_a_deleted_message_is_gone_from_every_summary_read(mstore, settings, indexed):
    st = IndexedMemoryStore(mstore, FakeEmbedder()) if indexed else mstore
    tools = MemoryTools(st, GUILD, {10: "general"}, settings, retention_days=30,
                        embedder=FakeEmbedder())  # fmt: skip
    summary = {"overview": "", "topics": [{"title": "รหัสลับ", "details": "เรื่องลับสุดยอด",
                                           "people": []}]}  # fmt: skip
    v = vectorize(summary, None, FakeEmbedder())
    st.put_summary(DaySummary(guild_id=GUILD, channel_id=10, local_date="2026-10-10",
                              message_count=1, final=True, summary=summary, vec_model=v.model_id,
                              vec_hashes=v.hashes, vecs=v.data))  # fmt: skip
    st.add_message(mk(settings, 1, bkk(10, 9).astimezone(UTC), "เรื่องลับสุดยอด"))
    assert tools.chat_search(["รหัสลับ"], query="รหัสลับ")["hits"]  # visible before the delete
    assert tools.chat_days("2026-10-10", "2026-10-10")["count"] == 1
    st.delete_messages(GUILD, [1])
    kw = tools.chat_search(["รหัสลับ"], semantic=False)
    meaning = tools.chat_search([], query="รหัสลับ")
    assert kw["hits"] == [] and meaning["hits"] == []
    assert "รหัสลับ" not in json.dumps(
        tools.chat_days("2026-10-10", "2026-10-10"), ensure_ascii=False
    )
    assert tools.chat_days("2026-10-10", "2026-10-10")["count"] == 0
    # once the day is regenerated it is served again
    st.put_summary(DaySummary(guild_id=GUILD, channel_id=10, local_date="2026-10-10",
                              message_count=2, final=True, summary={"overview": "สรุปใหม่"}))  # fmt: skip
    assert tools.chat_days("2026-10-10", "2026-10-10")["count"] == 1


def test_a_dirty_index_entry_keeps_no_items_or_vectors(mstore, settings):
    st = IndexedMemoryStore(mstore, FakeEmbedder())
    topic_day(st, "2026-10-10", 10, "ลับ")
    assert st.summary_entries(GUILD, {10}, "2026-10-01", "2026-10-31")  # loads the guild
    st.add_message(mk(settings, 1, bkk(10, 9).astimezone(UTC)))
    st.delete_messages(GUILD, [1])
    [e] = [e for e in st._guilds[GUILD].values()]
    assert e.summary.dirty and e.items == [] and e.vectors == []
    assert st.summary_entries(GUILD, {10}, "2026-10-01", "2026-10-31") == []


@pytest.mark.parametrize("indexed", [True, False])
def test_tools_clamp_every_date_range_to_the_current_retention(mstore, settings, indexed):
    st = IndexedMemoryStore(mstore, FakeEmbedder()) if indexed else mstore
    days = [30]
    tools = MemoryTools(st, GUILD, {10: "general"}, settings, retention_days=30,
                        retention_of=lambda: days[0], embedder=FakeEmbedder())  # fmt: skip
    topic_day(st, "2026-10-08", 10, "ประชุมเก่า")  # 6 days before "today" (2026-10-14)
    topic_day(st, "2026-10-14", 10, "ประชุมวันนี้")
    st.add_message(mk(settings, 1, bkk(8, 9).astimezone(UTC), "ประชุมเก่า"))
    assert tools.chat_days("2026-10-01", "2026-10-14")["count"] == 2
    days[0] = 1  # retention shortened: read at call time, not when the tools were built
    r = tools.chat_days("2026-10-01", "2026-10-14")
    assert [d["date"] for d in r["days"]] == ["2026-10-14"]
    hits = tools.chat_search(["ประชุมเก่า"], start_date="2026-10-01", end_date="2026-10-14")
    assert "2026-10-08" not in [h["date"] for h in hits["hits"]]
    found = tools.chat_search([], query="ประชุมเก่า", start_date="2026-10-01")["hits"]
    assert "2026-10-08" not in [h["date"] for h in found]
    assert tools.chat_messages("2026-10-01", "2026-10-14")["count"] == 0
    assert tools.chat_days("2026-10-01", "2026-10-07")["count"] == 0  # wholly expired range


async def test_shortening_retention_purges_the_guild_and_drops_its_index(bot, world):
    old = bkk(14, 9).astimezone(UTC) - timedelta(days=60)
    bot.memory_store.add_message(mk(bot.s, 1, old))
    topic_day(bot.memory_store, "2026-08-15", 10, "เก่ามาก")
    other = GUILD + 1
    bot.memory_store.add_message(mk(bot.s, 2, old, gid=other))
    assert bot.memory_store.summaries(GUILD, {10}, "2026-08-01", "2026-08-31")  # index loaded
    boss = Interaction(world.g, member(world.g, MANAGER, manage_guild=True), world.gen)
    await bot.handle_memory(boss, 30)
    await asyncio.gather(*bot._bg_tasks)
    assert bot.memory_store.messages(GUILD, {10}, "2026-01-01", "2026-12-31") == []
    assert bot.memory_store.summaries(GUILD, {10}, "2026-08-01", "2026-08-31") == []
    # another guild's data is untouched by this guild's purge
    assert bot.memory_store.messages(other, {10}, "2026-01-01", "2026-12-31")


async def test_a_retention_change_seen_by_the_tick_also_purges(bot, world):
    old = bkk(14, 9).astimezone(UTC) - timedelta(days=60)
    bot.memory_store.add_message(mk(bot.s, 1, old))
    bot._memory_triggers()  # remembers 90 days
    bot.team.set_memory_retention_days(GUILD, 30)  # e.g. from the hub
    bot._memory_triggers()
    await asyncio.gather(*bot._bg_tasks)
    assert bot.memory_store.messages(GUILD, {10}, "2026-01-01", "2026-12-31") == []


# ------------------------------------------------------------------ the end-to-end scenario
class FakeClockNow:
    pass


async def test_end_to_end_ads_budget_question_from_a_dm(bot, world):
    """A member posts the budget in a channel; 18 s later another member who can read it asks in
    a DM. No summary exists yet, but chat_search and chat_days(today) both return the raw
    message. A member who cannot read the channel gets nothing."""
    g = world.g
    g.text_channels[:] = [RChan(10, "ads-team", {STAFF, MANAGER}), RChan(12, "ผู้บริหาร", {EXEC})]
    for c in g.text_channels:
        c.guild = g
    poster, asker, outsider = member(g, STAFF), member(g, MANAGER), member(g, EXEC)
    text = "งบโฆษณาเดือนนี้ 50,000 บาท ใช้ไปแล้ว 32,000"
    posted = TMsg(g, g.text_channels[0], poster, text)
    posted.jump_url = "https://discord.com/channels/1000/10/42"
    await bot.on_message(posted)  # the log is written by the time the queue has run, no summary
    await bot.drain_memory()
    bot.clock.advance(18)
    assert bot.memory_store.get_summary(GUILD, 10, "2026-10-14") is None

    dm = SimpleNamespace(guild=None)  # a DM: no answering channel
    tools = await bot._memory_tools(dm, g, asker)
    assert tools.allowed == {10: "ads-team"}
    found = tools.chat_search(["งบโฆษณา", "ads"])
    assert found["count"] == 1 and found["hits"][0]["channel"] == "ads-team"
    msg = found["hits"][0]["messages"][0]
    assert msg["text"] == text and msg["link"].endswith("/42") and msg["author"] == "u4"
    today = tools.chat_days("2026-10-14", "2026-10-14")
    assert today["days"] == [] and [m["text"] for m in today["newer_than_summary"]] == [text]

    nothing = await bot._memory_tools(dm, g, outsider)  # cannot read #ads-team
    assert nothing.allowed == {12: "ผู้บริหาร"}
    assert nothing.chat_search(["งบโฆษณา", "ads"])["hits"] == []
    assert "newer_than_summary" not in nothing.chat_days("2026-10-14", "2026-10-14")
    assert nothing.chat_messages("2026-10-14", "2026-10-14", keyword="งบ")["count"] == 0


def test_latest_message_at_parity(mstore, settings):
    now = bkk(14, 10).astimezone(UTC)
    mstore.add_message(mk(settings, 1, now - timedelta(days=2)))
    mstore.add_message(mk(settings, 2, now, cid=10))
    mstore.add_message(mk(settings, 3, now + timedelta(hours=1), cid=11))
    assert mstore.latest_message_at(GUILD, 10, "2026-10-01") == now
    assert mstore.latest_message_at(GUILD, 12, "2026-10-01") is None
    assert mstore.latest_message_at(GUILD, 10, "2026-10-15") is None


def test_import_state_parity(mstore):
    from salid.memory_store import ChannelState, GuildMemoryState

    assert mstore.get_channel_state(GUILD, 10) is None and mstore.list_channel_states(GUILD) == []
    st = ChannelState(
        guild_id=GUILD,
        channel_id=10,
        imported_from="a",
        last_message_at="b",
        summary_days=["2026-10-01"],
    )
    mstore.put_channel_state(st)
    mstore.put_channel_state(ChannelState(guild_id=GUILD + 1, channel_id=10))
    assert mstore.get_channel_state(GUILD, 10) == st
    assert [s.channel_id for s in mstore.list_channel_states(GUILD)] == [10]
    assert mstore.get_guild_state(GUILD) == GuildMemoryState(guild_id=GUILD)
    mstore.put_guild_state(GuildMemoryState(guild_id=GUILD, day="2026-10-14", images_done=5))
    assert mstore.get_guild_state(GUILD).images_done == 5
    mstore.delete_guild(GUILD)
    assert (
        mstore.get_channel_state(GUILD, 10) is None
        and mstore.get_guild_state(GUILD).images_done == 0
    )
    assert mstore.get_channel_state(GUILD + 1, 10) is not None


# ------------------------------------------------------------------ vectors
def test_summary_items_lists_every_part():
    s = {"overview": "o", "topics": [{"title": "t", "details": "d", "people": ["แอน"]}],
         "decisions": ["x"], "action_items": [{"who": "บอย", "what": "w", "due": ""}],
         "numbers": ["5"], "open_questions": ["q"]}  # fmt: skip
    assert summary_items(s) == [
        "o", "Topic: t d แอน", "Decision: x", "Action: บอย w", "Number: 5", "Open: q",
    ]  # fmt: skip


def test_update_writes_vectors_and_reuses_unchanged_items(mstore, settings):
    first = json_resp({"overview": "งบโฆษณา", "decisions": ["d1"]})
    second = json_resp({"overview": "งบโฆษณา", "decisions": ["d1", "ซื้อน้ำแข็ง"]})
    clock, llm, emb = Clock(bkk(14, 10)), FakeClient([first, second]), FakeEmbedder()
    s = sched(mstore, settings, clock, llm, embedder=emb)
    svc = MemoryService(mstore, settings, s)
    for i in range(30):
        log(svc, settings, i + 1, bkk(14, 9, i))
    s.tick([GUILD])
    for i in range(30, 60):
        log(svc, settings, i + 1, bkk(14, 10, i - 30))  # after the first update's cursor
    s.tick([GUILD])
    assert emb.calls[1] == (["Decision: ซื้อน้ำแข็ง"], "document")
    got = mstore.get_summary(GUILD, 10, "2026-10-14")
    assert vectors_usable(got, "fake@4") and len(unpack(got.vecs, 4)) == 3


def test_summary_is_saved_when_embedding_fails(mstore, settings):
    clock, llm = Clock(bkk(14, 10)), FakeClient([summary_json("a")])
    s = sched(mstore, settings, clock, llm, embedder=FakeEmbedder(fail=True))
    svc = MemoryService(mstore, settings, s)
    for i in range(30):
        log(svc, settings, i + 1, bkk(14, 9, i))
    s.tick([GUILD])
    got = mstore.get_summary(GUILD, 10, "2026-10-14")
    assert got.summary["overview"] == "overview a" and got.vec_model == ""


def test_backfill_vectorises_old_summaries_newest_first_with_a_cap(mstore, settings):
    for d in range(1, 13):
        mstore.put_summary(DaySummary(guild_id=GUILD, channel_id=10, local_date=f"2026-10-{d:02d}",
                                      final=True, message_count=3, summary={"overview": f"day {d}"}))  # fmt: skip
    emb = FakeEmbedder()
    s = sched(mstore, settings, Clock(bkk(14, 10)), FakeClient([]), embedder=emb)
    s.tick([GUILD])
    assert [c[0] for c in emb.calls][0] == ["day 12"] and len(emb.calls) == 10
    s.tick([GUILD])
    assert len(emb.calls) == 12
    s.tick([GUILD])
    assert len(emb.calls) == 12  # nothing left: the guild is skipped
    assert all(vectors_usable(x, "fake@4") for x in mstore.all_summaries(GUILD))


def test_backfill_redoes_vectors_of_another_model(mstore, settings):
    old = vectorize({"overview": "งบโฆษณา"}, None, FakeEmbedder())
    mstore.put_summary(DaySummary(guild_id=GUILD, channel_id=10, local_date="2026-10-01", final=True,
                                  summary={"overview": "งบโฆษณา"}, vec_model="other@8",
                                  vec_hashes=old.hashes, vecs=old.data))  # fmt: skip
    emb = FakeEmbedder()
    sched(mstore, settings, Clock(bkk(14, 10)), FakeClient([]), embedder=emb).tick([GUILD])
    assert mstore.get_summary(GUILD, 10, "2026-10-01").vec_model == "fake@4"


def test_empty_summary_counts_as_vectorised():
    s = DaySummary(guild_id=GUILD, channel_id=10, local_date="2026-10-01", summary={})
    assert vectors_usable(s, "fake@4") and vectorize({}, None, FakeEmbedder()).embedded == 0


def test_put_summary_vectors_and_all_summaries_parity(mstore):
    mstore.put_summary(DaySummary(guild_id=GUILD, channel_id=10, local_date="2026-10-01",
                                  summary={"overview": "a"}, message_count=4))  # fmt: skip
    mstore.put_summary(DaySummary(guild_id=GUILD + 1, channel_id=10, local_date="2026-10-01",
                                  summary={}))  # fmt: skip
    assert mstore.put_summary_vectors(GUILD, 10, "2026-10-01", "fake@4", ["h"], b"\0" * 16)
    assert not mstore.put_summary_vectors(GUILD, 99, "2026-10-01", "fake@4", ["h"], b"")
    got = mstore.all_summaries(GUILD)
    assert [(x.channel_id, x.vec_model, x.vecs, x.message_count) for x in got] == [
        (10, "fake@4", b"\0" * 16, 4)
    ]


def test_old_sqlite_summary_table_gets_vector_columns(tmp_path):
    import sqlite3

    path = str(tmp_path / "old.db")
    with sqlite3.connect(path) as c:
        c.execute(
            "CREATE TABLE salid_day_summaries (guild_id INTEGER, channel_id INTEGER, "
            "local_date TEXT, summary TEXT, last_message_at TEXT, last_message_id INTEGER DEFAULT 0, "
            "message_count INTEGER DEFAULT 0, updated_at TEXT, final INTEGER DEFAULT 0, "
            "dirty INTEGER DEFAULT 0, PRIMARY KEY (guild_id, channel_id, local_date))"
        )
        c.execute(
            "INSERT INTO salid_day_summaries (guild_id, channel_id, local_date, summary, updated_at) "
            "VALUES (1000, 10, '2026-10-01', '{}', '')"
        )
    st = SQLiteMemoryStore(path)
    assert st.get_summary(GUILD, 10, "2026-10-01").vec_model == ""
    assert st.put_summary_vectors(GUILD, 10, "2026-10-01", "fake@4", ["h"], b"\1" * 16)
    assert st.get_summary(GUILD, 10, "2026-10-01").vecs == b"\1" * 16


# ------------------------------------------------------------------ review fixes
def test_backfill_stops_at_its_time_budget(mstore, settings):
    for d in range(1, 13):
        mstore.put_summary(DaySummary(guild_id=GUILD, channel_id=10, local_date=f"2026-10-{d:02d}",
                                      final=True, message_count=3, summary={"overview": f"day {d}"}))  # fmt: skip
    seconds = iter(range(0, 1000, 2))  # every timer read: 2 s later
    emb = FakeEmbedder()
    s = sched(mstore, settings, Clock(bkk(14, 10)), FakeClient([]), embedder=emb,
              timer=lambda: next(seconds))  # fmt: skip
    s.tick([GUILD])
    assert len(emb.calls) == 2  # the tick loop is not held for the whole batch


def test_backfill_rechecks_a_done_guild_for_summaries_written_elsewhere(mstore, settings):
    clock, emb = Clock(bkk(14, 10)), FakeEmbedder()
    s = sched(mstore, settings, clock, FakeClient([]), embedder=emb)
    s.tick([GUILD])  # nothing to do yet: the guild is marked done
    mstore.put_summary(DaySummary(guild_id=GUILD, channel_id=10, local_date="2026-10-01",
                                  final=True, summary={"overview": "imported"}))  # fmt: skip
    s.tick([GUILD])  # e.g. written by the history importer, not by the scheduler
    assert emb.calls == []
    clock.at = bkk(14, 10, 11)
    s.tick([GUILD])
    assert vectors_usable(mstore.get_summary(GUILD, 10, "2026-10-01"), "fake@4")


def test_one_unembeddable_summary_does_not_stall_the_backfill(mstore, settings):
    for d, text in ((1, "day 1"), (2, "day 2"), (3, "poison")):
        mstore.put_summary(DaySummary(guild_id=GUILD, channel_id=10, local_date=f"2026-10-{d:02d}",
                                      final=True, message_count=3, summary={"overview": text}))  # fmt: skip

    class Picky(FakeEmbedder):
        def embed(self, texts, task):
            if any("poison" in t for t in texts):
                raise RuntimeError("rejected")
            return super().embed(texts, task)

    s = sched(mstore, settings, Clock(bkk(14, 10)), FakeClient([]), embedder=Picky())
    s.tick([GUILD])
    s.tick([GUILD])
    for day in ("2026-10-01", "2026-10-02"):
        assert vectors_usable(mstore.get_summary(GUILD, 10, day), "fake@4")


def test_slow_query_embedding_times_out_to_keywords(mstore, settings, monkeypatch):
    monkeypatch.setattr(qa_mod, "QUERY_EMBED_TIMEOUT_S", 0.05)

    class Slow(FakeEmbedder):
        model_id = "slow@4"

        def embed(self, texts, task):
            if task == "query":
                time.sleep(0.5)
            return super().embed(texts, task)

    st, tools = hybrid_tools(mstore, settings, emb=Slow())
    topic_day(st, "2026-10-05", 12, "งบโฆษณา")
    started = time.monotonic()
    r = tools.chat_search(["งบโฆษณา"], query="a slow marketing question")
    assert time.monotonic() - started < 0.4 and [h["match"] for h in r["hits"]] == ["keyword"]


def test_month_range_ending_in_the_future_stays_within_budget(settings):
    base = offline_memory_store()
    st, tools = hybrid_tools(base, settings, allowed={10: "general"})
    base.add_message(mk(settings, 1, bkk(14, 9).astimezone(UTC), "ads", cid=10))
    base.client.streams.clear()
    r = tools.chat_search(["ads"], "2026-10-01", "2026-10-31", query="ads this month")
    assert r["hits"] and len(base.client.streams) <= 10


def test_query_only_search_uses_its_words_when_meaning_search_is_off(mstore, settings):
    tools = MemoryTools(mstore, GUILD, {12: "ads"}, settings, retention_days=30)  # no embedder
    mstore.put_summary(DaySummary(guild_id=GUILD, channel_id=12, local_date="2026-10-05",
                                  message_count=3, final=True,
                                  summary={"overview": "marketing plan for Q4"}))  # fmt: skip
    r = tools.chat_search([], query="what is the marketing plan?")
    assert [h["date"] for h in r["hits"]] == ["2026-10-05"]


def test_query_only_search_uses_its_words_when_embedding_fails(mstore, settings):
    st, tools = hybrid_tools(mstore, settings, emb=FakeEmbedder(fail=True))
    topic_day(st, "2026-10-05", 12, "marketing plan")
    r = tools.chat_search([], query="the marketing plan please")
    assert [(h["date"], h["match"]) for h in r["hits"]] == [("2026-10-05", "keyword")]
