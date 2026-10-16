# ruff: noqa: F811
"""Attendance log: who entered / left the voice meeting and when. Fully offline."""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from conftest import FixedClockSettings, text_resp
from test_meeting import FakeVoiceChannel, env, make_member  # noqa: F401  (fixture + helpers)

from salid import attendance as att
from salid import meeting_doc as md
from salid.guild_settings import GuildSettingsUpdate
from salid.i18n import t

TZ = ZoneInfo("Asia/Bangkok")
T0 = datetime(2026, 10, 14, 23, 12, tzinfo=TZ)
KEY = (1000, 77)


def ev(uid, name, kind, minutes, mono=None):
    at = T0 + timedelta(minutes=minutes)
    return att.AttendanceEvent(uid, name, kind, at, (minutes * 60.0) if mono is None else mono)


# ------------------------------------------------------------------ pure logic
def test_chat_lines_th_batch_and_en():
    one = [ev(1, "พีท", "joined", 0)]
    assert att.chat_lines(one, "th") == "🟢 23:12 คุณพีท เข้าห้องประชุม"
    assert att.chat_lines([ev(1, "พีท", "left", 18)], "th") == "🔴 23:30 คุณพีท ออกจากห้องประชุม"
    two = [ev(1, "พีท", "joined", 0), ev(2, "คุณมิว", "joined", 0)]
    assert att.chat_lines(two, "th") == "🟢 23:12 คุณพีท, คุณมิว เข้าห้องประชุม"  # no "คุณคุณ"
    assert att.chat_lines(one, "en") == "🟢 23:12 พีท joined the meeting"
    assert att.chat_lines([ev(1, "Pete", "left", 18)], "en") == "🔴 23:30 Pete left the meeting"


def test_record_ignores_duplicates_and_retracts_a_reconnect():
    log = []
    assert att.record(log, ev(1, "พีท", "joined", 0))[0]
    assert att.record(log, ev(1, "พีท", "joined", 1)) == (False, None)  # already present
    assert att.record(log, ev(2, "มิว", "left", 1)) == (False, None)  # never joined
    left = ev(1, "พีท", "left", 5)
    assert att.record(log, left)[0]
    added, retracted = att.record(log, ev(1, "พีท", "joined", 5, mono=5 * 60 + 19.0))
    assert not added and retracted is left
    assert [e.kind for e in log] == ["joined"]  # as if nothing happened
    assert att.record(log, ev(1, "พีท", "left", 6))[0]
    assert att.record(log, ev(1, "พีท", "joined", 7, mono=6 * 60 + 21.0))[0]  # a real return
    assert [e.kind for e in log] == ["joined", "left", "joined"]


def test_presences_intervals_still_present_and_minutes_block():
    end = T0 + timedelta(minutes=33)
    log = [
        ev(1, "พีท", "joined", 0),
        ev(2, "มิว", "joined", 0),
        ev(2, "มิว", "left", 8),
        ev(2, "มิว", "joined", 18),
        ev(1, "พีท", "left", 33),
    ]
    people = att.presences(log, end)
    assert [(p.name, p.minutes) for p in people] == [("พีท", 33), ("มิว", 8 + 15)]
    assert att.minutes_block(people, "th") == (
        "👥 ผู้เข้าร่วม\n"
        "คุณพีท 23:12–23:45 (33 นาที)\n"
        "คุณมิว 23:12–23:20 (8 นาที), 23:30–23:45 (15 นาที)\n"
        "ทั้งหมด 2 คน\n\n"
    )
    assert att.minutes_block(people, "en").startswith("👥 Attendees\nพีท 23:12–23:45 (33 min)")
    assert att.minutes_block([], "th") == ""
    short = att.presences([ev(3, "บอย", "joined", 0)], T0 + timedelta(seconds=10))
    assert "ไม่ถึง 1 นาที" in att.minutes_block(short, "th")


def test_document_section_and_off_state():
    people = att.presences([ev(1, "พีท", "joined", 0), ev(1, "พีท", "left", 30)], T0)
    info = md.MeetingInfo(room="ห้องประชุม", started_at=T0, attendance=people)
    text = md.render_markdown(info, [], "th")
    assert "## ผู้เข้าร่วม" in text and "| คุณพีท | 23:12–23:42 | 30 |" in text
    assert "ทั้งหมด 1 คน" in text
    en = md.render_markdown(info, [], "en")
    assert "## Attendance" in en and "| พีท | 23:12–23:42 | 30 |" in en
    assert "## ผู้เข้าร่วม" not in md.render_markdown(md.MeetingInfo(started_at=T0), [], "th")


# ------------------------------------------------------------------ manager
class Clock:
    def __init__(self):
        self.t = 0.0

    def wall(self):
        return T0 + timedelta(seconds=self.t)


@pytest.fixture
def clk(env, monkeypatch):
    c = Clock()
    monkeypatch.setattr(FixedClockSettings, "now", lambda self: c.wall())
    env.mgr._now = lambda: c.t

    async def sleep(d):
        c.t += d
        await asyncio.sleep(0)

    env.mgr._sleep = sleep
    env.bot.s.voice_state_posts = True
    return c


async def settle():
    for _ in range(10):
        await asyncio.sleep(0)


def vs(channel):
    return SimpleNamespace(channel=channel)


async def begin(env, *names):
    """Meeting in room 77 with `names` ((id, name) pairs) already inside."""
    members = [make_member(env.guild, uid, n, env.chan) for uid, n in names]
    env.chan.members = list(members)
    await env.mgr.join(members[0])
    await settle()
    return env.mgr.sessions[KEY], members


def lines(env):
    return [m for m in env.chan.sent if m[:1] in "🟢🔴" or "\n🔴" in m]


async def test_everyone_in_the_room_is_logged_at_the_start_in_one_message(env, clk):
    s, _ = await begin(env, (10, "พีท"), (11, "มิว"))
    assert [(e.user_id, e.kind, e.at) for e in s.attendance] == [
        (10, "joined", T0),
        (11, "joined", T0),
    ]
    assert lines(env) == ["🟢 23:12 คุณพีท, คุณมิว เข้าห้องประชุม"]
    await env.mgr.end(KEY, "shutdown")


async def test_later_join_and_leave_are_logged_and_posted(env, clk):
    s, (pete, _) = await begin(env, (10, "พีท"), (11, "มิว"))
    new = make_member(env.guild, 12, "บอย")
    clk.t = 60
    await env.mgr.on_voice_activity(new, vs(None), vs(env.chan))
    await settle()
    assert lines(env)[-1] == "🟢 23:13 คุณบอย เข้าห้องประชุม"
    clk.t = 18 * 60
    await env.mgr.on_voice_activity(pete, vs(env.chan), vs(None))
    await settle()
    assert lines(env)[-1] == "🔴 23:30 คุณพีท ออกจากห้องประชุม"
    assert [(e.user_id, e.kind) for e in s.attendance][-2:] == [(12, "joined"), (10, "left")]
    await env.mgr.end(KEY, "shutdown")


async def test_moving_to_another_room_is_a_leave(env, clk):
    s, (pete, _) = await begin(env, (10, "พีท"), (11, "มิว"))
    other = FakeVoiceChannel(id=88, name="อีกห้อง")
    other.guild = env.guild
    clk.t = 600
    await env.mgr.on_voice_activity(pete, vs(env.chan), vs(other))
    await settle()
    assert s.attendance[-1].kind == "left" and s.attendance[-1].user_id == 10
    assert lines(env)[-1].startswith("🔴 23:22 คุณพีท")
    await env.mgr.end(KEY, "shutdown")


async def test_mute_and_deafen_toggles_and_bots_are_ignored(env, clk):
    s, (pete, _) = await begin(env, (10, "พีท"), (11, "มิว"))
    before = list(env.chan.sent)
    await env.mgr.on_voice_activity(pete, vs(env.chan), vs(env.chan))  # same channel
    bot_member = make_member(env.guild, 50, "บอท")
    bot_member.bot = True
    await env.mgr.on_voice_activity(bot_member, vs(None), vs(env.chan))
    await settle()
    assert len(s.attendance) == 2 and env.chan.sent == before
    await env.mgr.end(KEY, "shutdown")


async def test_a_quick_reconnect_is_not_logged_twice(env, clk):
    s, (pete, _) = await begin(env, (10, "พีท"), (11, "มิว"))
    posted = list(env.chan.sent)
    clk.t = 300
    await env.mgr.on_voice_activity(pete, vs(env.chan), vs(None))
    clk.t = 310  # back within 20 s
    await env.mgr.on_voice_activity(pete, vs(None), vs(env.chan))
    await settle()
    assert [(e.user_id, e.kind) for e in s.attendance] == [(10, "joined"), (11, "joined")]
    assert env.chan.sent == posted  # neither a leave nor a join line
    await env.mgr.end(KEY, "shutdown")


async def test_events_close_together_share_one_message(env, clk):
    s, _ = await begin(env, (10, "พีท"))
    a, b = make_member(env.guild, 12, "บอย"), make_member(env.guild, 13, "แจม")
    clk.t = 120
    await env.mgr.on_voice_activity(a, vs(None), vs(env.chan))
    clk.t = 121.5
    await env.mgr.on_voice_activity(b, vs(None), vs(env.chan))
    await settle()
    assert lines(env)[-1] == "🟢 23:14 คุณบอย, คุณแจม เข้าห้องประชุม"
    assert len(lines(env)) == 2
    await env.mgr.end(KEY, "shutdown")


async def test_english_lines_and_no_mention_pings(env, clk):
    env.bot.s.voice_state_posts = True
    s, _ = await begin(env, (10, "@everyone Pete"))
    s.fallback_lang = "en"
    s.language = "en"
    new = make_member(env.guild, 12, "Mew")
    clk.t = 60
    await env.mgr.on_voice_activity(new, vs(None), vs(env.chan))
    await settle()
    assert lines(env)[-1] == "🟢 23:13 Mew joined the meeting"
    assert "@everyone" not in lines(env)[0].replace("@​everyone", "")
    await env.mgr.end(KEY, "shutdown")


async def test_end_of_meeting_section_in_the_posted_minutes(env, clk):
    s, (pete, mew) = await begin(env, (10, "พีท"), (11, "มิว"))
    clk.t = 8 * 60
    await env.mgr.on_voice_activity(mew, vs(env.chan), vs(None))
    clk.t = 18 * 60
    await env.mgr.on_voice_activity(mew, vs(None), vs(env.chan))
    clk.t = 33 * 60
    s.transcript.add(clk.wall(), "พีท", "ให้บอยส่งรายงาน")
    env.llm.models.script = [text_resp("**สรุปการประชุม**\nบอยส่งรายงาน")]
    await env.mgr.end(KEY, "command")
    posted = env.chan.sent[-1]
    assert "👥 ผู้เข้าร่วม" in posted
    assert posted.index("👥 ผู้เข้าร่วม") < posted.index("สรุปการประชุม**")
    assert "คุณพีท 23:12–23:45 (33 นาที)" in posted  # still in the room: until the end
    assert "คุณมิว 23:12–23:20 (8 นาที), 23:30–23:45 (15 นาที)" in posted
    assert "ทั้งหมด 2 คน" in posted
    assert "ผู้เข้าร่วม" not in env.llm.models.calls[0][1][0]  # the summary prompt is untouched


async def test_minutes_document_carries_the_attendance_section(env, clk):
    from test_meeting import FileChannel

    env.chan = FileChannel()
    env.chan.guild = env.guild
    s, _ = await begin(env, (10, "พีท"), (11, "มิว"))
    clk.t = 33 * 60
    s.transcript.add(clk.wall(), "พีท", "ให้บอยส่งรายงาน")
    env.llm.models.script = [text_resp("**สรุปการประชุม**\nx")]
    ended = clk.wall()
    people = env.mgr._attendance_people(s, ended)
    assert [p.name for p in people] == ["พีท", "มิว"]
    info = md.MeetingInfo(started_at=s.started_at, ended_at=ended, attendance=people)
    assert "| คุณมิว | 23:12–23:45 | 33 |" in md.render_markdown(info, [], "th")
    await env.mgr.end(KEY, "command")
    assert env.chan.files[-1]  # the document is still attached


async def test_flag_off_means_no_lines_and_no_section(env, clk):
    env.bot.guilds_store.save_settings(
        1000, GuildSettingsUpdate(features=dict(attendance_log=False)), "x"
    )
    await asyncio.to_thread(env.bot.team.current, 1000)
    s, (pete, _) = await begin(env, (10, "พีท"), (11, "มิว"))
    new = make_member(env.guild, 12, "บอย")
    await env.mgr.on_voice_activity(new, vs(None), vs(env.chan))
    await env.mgr.on_voice_activity(pete, vs(env.chan), vs(None))
    await settle()
    assert s.attendance == [] and lines(env) == []
    s.transcript.add(clk.wall(), "พีท", "ให้บอยส่งรายงาน")
    env.llm.models.script = [text_resp("**สรุปการประชุม**\nx")]
    await env.mgr.end(KEY, "command")
    assert "ผู้เข้าร่วม" not in env.chan.sent[-1]
    assert env.mgr._attendance_people(s, clk.wall()) == []


def test_catalog_has_both_languages():
    for key in ("attend_joined", "attend_left", "attend_header", "attend_total", "attend_minutes"):
        kw = dict(icon="x", time="1", names="a", n=1)
        assert t(key, "th", **kw) != t(key, "en", **kw)
