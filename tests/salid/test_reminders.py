# ruff: noqa: F811, F401
"""Personal reminders: parsing, schedule maths, limits, who may remind whom, cancelling, the tick
(fake clock) and the glue (tools in the agent, voice rebinding, prompt, flag). Fakes only."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from conftest import GUILD
from test_briefings import set_features
from test_coordinator import BKK, BOSS, PEER, STAFF, at, person, world
from test_modes import bot  # noqa: F401  (fixture)

from salid import reminders as R
from salid.coo_store import FirestoreDocStore, SQLiteDocStore
from salid.i18n import CATALOG, t
from salid.mentions import match_members
from salid.qa import stable_prompt
from salid.voice_actions import ACTION_TOOL_NAMES

NOW = at(12)  # Wednesday 2026-10-14 12:00 Bangkok
GONE = 999


def iso(dt):
    return dt.astimezone(UTC).isoformat(timespec="seconds")


# ------------------------------------------------------------------ parsing
def test_parse_at_every_accepted_form():
    assert R.parse_at("2026-10-15 09:00", NOW) == at(9, day=15)
    assert R.parse_at("2026-10-15T09:30", NOW) == at(9, 30, day=15)
    assert R.parse_at("14:30", NOW) == at(14, 30)  # still ahead: today
    assert R.parse_at("09:00", NOW) == at(9, day=15)  # already passed: tomorrow
    assert R.parse_at("12:00", NOW) == at(12, day=15)  # not strictly ahead: tomorrow
    assert R.parse_at("+30m", NOW) == at(12, 30)
    assert R.parse_at("+2h", NOW) == at(14)
    assert R.parse_at("+1d", NOW) == at(12, day=15)
    assert R.parse_at("9.00 น.", NOW) == at(9, day=15)
    for bad in ("", "tomorrow", "25:00", "2026-13-01 10:00", "+0m", "+x", "10"):
        with pytest.raises(R.ReminderError) as e:
            R.parse_at(bad, NOW)
        assert e.value.code == "bad_time"


def test_parse_every_and_limits():
    assert R.parse_every("30m") == 1800 and R.parse_every("2h") == 7200
    assert R.parse_every("1d") == 86400 and R.parse_every("15 min") == 900
    with pytest.raises(R.ReminderError) as e:
        R.parse_every("10m")
    assert e.value.code == "min_interval"
    for bad in ("", "often", "0h", "2x", "-1h"):
        with pytest.raises(R.ReminderError) as e:
            R.parse_every(bad)
        assert e.value.code == "bad_every"


def test_parse_weekdays_english_thai_groups_and_ranges():
    assert R.parse_weekdays("daily") == list(range(7))
    assert R.parse_weekdays("weekdays") == [0, 1, 2, 3, 4]
    assert R.parse_weekdays("mon, wed,Fri") == [0, 2, 4]
    assert R.parse_weekdays("จ,อ,พ,พฤ,ศ,ส,อา") == list(range(7))
    assert R.parse_weekdays("พ พฤ") == [2, 3]  # พ and พฤ are different days
    assert R.parse_weekdays("จันทร์,วันศุกร์") == [0, 4]
    assert R.parse_weekdays("mon-fri") == [0, 1, 2, 3, 4]
    assert R.parse_weekdays("weekend") == [5, 6]
    for bad in ("", "someday", "mon,xyz"):
        with pytest.raises(R.ReminderError) as e:
            R.parse_weekdays(bad)
        assert e.value.code == "bad_weekdays"


def test_message_is_cleaned_and_broadcast_mentions_defused():
    assert R.clean_message("  ออกกำลัง   กาย ") == "ออกกำลัง กาย"
    out = R.clean_message("hey @everyone and @here <@&5>")
    assert "@everyone" not in out and "@here" not in out and "<@&5>" not in out
    with pytest.raises(R.ReminderError):
        R.clean_message("  ")
    with pytest.raises(R.ReminderError) as e:
        R.clean_message("x" * 301)
    assert e.value.code == "too_long"
    assert R.norm_id("r-12") == "R-0012" and R.norm_id("12") == "R-0012"
    assert R.norm_id("exercise") is None


# ------------------------------------------------------------------ next-slot maths
def rec(kind, **kw):
    base = {
        "id": "R-0001", "guild_id": GUILD, "user_id": STAFF, "set_by": STAFF, "set_by_name": "x",
        "message": "m", "kind": kind, "next_at": iso(NOW), "every_s": 0, "at_hhmm": "",
        "weekdays": [], "until": None, "quiet": False, "where": "me", "channel_id": None,
        "lang": "th", "created_at": iso(NOW), "fired": 0, "active": True,
    }  # fmt: skip
    return {**base, **kw}


def test_interval_next_slot_keeps_cadence_and_is_in_the_future():
    r = rec("interval", every_s=7200, next_at=iso(at(12)))
    assert R.next_slot(r, at(12)) == at(14)
    assert R.next_slot(r, at(12, 1)) == at(14)
    assert R.next_slot(r, at(14)) == at(16)  # strictly after
    # the bot was down for a day: next slot on the cadence, a single one
    assert R.next_slot(r, at(13, day=15)) == at(14, day=15)


def test_interval_across_the_quiet_window():
    r = rec("interval", every_s=7200, quiet=True, next_at=iso(at(20)))
    assert R.next_slot(r, at(20)) == at(8, day=15)  # 22:00 -> 08:00, nothing at night
    r = rec("interval", every_s=7200, quiet=True, next_at=iso(at(8, day=15)))
    assert R.next_slot(r, at(8, day=15)) == at(10, day=15)
    r = rec("interval", every_s=3 * 3600, quiet=True, next_at=iso(at(19)))
    assert R.next_slot(r, at(19)) == at(8, day=15)  # 22:00 is inside the window
    r = rec("interval", every_s=7200, quiet=True, next_at=iso(at(2, day=15)))
    assert R.next_slot(r, at(2, day=15)) == at(8, day=15)
    loud = rec("interval", every_s=7200, quiet=False, next_at=iso(at(20)))
    assert R.next_slot(loud, at(20)) == at(22)  # quiet_hours=false is honoured


def test_weekly_over_a_weekend_and_today_if_ahead():
    friday = at(10, day=16)
    r = rec("weekly", weekdays=[0, 1, 2, 3, 4], at_hhmm="09:00")
    assert R.next_slot(r, friday) == at(9, day=19)  # Monday
    assert R.next_slot(r, at(8, day=16)) == at(9, day=16)  # today, still ahead
    thu = rec("weekly", weekdays=[3], at_hhmm="08:00")
    assert R.next_slot(thu, NOW) == at(8, day=15)
    assert R.next_slot(thu, at(8, day=15)) == at(8, day=22)  # strictly after, next week


def test_until_expires_the_reminder_and_advance_never_goes_back():
    r = rec("weekly", weekdays=list(range(7)), at_hhmm="09:00", until="2026-10-15")
    assert R.next_slot(r, at(10)) == at(9, day=15)
    assert R.next_slot(r, at(10, day=15)) is None
    done = R.advance(r, at(10, day=15))
    assert done["active"] is False and done["done_at"]
    once = R.advance(rec("once"), NOW)
    assert once["active"] is False
    i = R.advance(rec("interval", every_s=900, next_at=iso(at(11))), NOW)
    assert R._from_iso(i["next_at"]) > NOW and i["active"]


# ------------------------------------------------------------------ the tools
def members(*people):
    pool = [SimpleNamespace(id=i, display_name=n, bot=False) for i, n in people]
    return pool


def make_ctx(is_exec=False, uid=STAFF, name="Somchai", lang="th", docs=None, here=None, now=NOW):
    docs = docs or SQLiteDocStore(":memory:")
    pool = members((STAFF, "Somchai"), (PEER, "Peer"), (BOSS, "Boss"), (30, "Peerapat"))
    guild = SimpleNamespace(members=pool)
    return R.ReminderCtx(
        guild_id=GUILD, asker_id=uid, asker_name=name, is_exec=is_exec, lang=lang,
        book=R.ReminderBook(docs), now=lambda: now,
        resolve=lambda n: match_members(guild, n),
        name_of=lambda i: next((m.display_name for m in pool if m.id == i), str(i)),
        channel_id=here,
    )  # fmt: skip


def tools_of(ctx):
    return {f.__name__: f for f in R.reminder_tools(ctx)}


def test_set_once_and_the_stored_fields():
    ctx = make_ctx()
    out = tools_of(ctx)["set_reminder"]("โทรหาลูกค้า", at="+30m")
    assert out["ok"] and out["id"] == "R-0001" and "12:30" in out["next"]
    assert out["summary"].startswith("R-0001 · โทรหาลูกค้า")
    doc = ctx.book.get(GUILD, "R-0001")
    assert doc["kind"] == "once" and doc["next_at"] == iso(at(12, 30)) and doc["active"]
    assert doc["user_id"] == STAFF == doc["set_by"] and doc["where"] == "me" and doc["lang"] == "th"
    assert tools_of(ctx)["set_reminder"]("อีกอัน", at="09:00")["id"] == "R-0002"


def test_set_interval_defaults_quiet_and_first_fire_after_one_interval():
    ctx = make_ctx()
    out = tools_of(ctx)["set_reminder"]("ออกกำลังกาย", every="2h")
    assert out["ok"] and "14:00" in out["next"] and out["quiet_hours"]
    d = ctx.book.get(GUILD, out["id"])
    assert d["kind"] == "interval" and d["every_s"] == 7200 and d["quiet"] is True
    late = make_ctx(now=at(21, 30))
    out = tools_of(late)["set_reminder"]("ยืด", every="2h")
    assert "08:00" in out["next"]  # 23:30 would be quiet
    off = tools_of(late)["set_reminder"]("ยืด", every="2h", quiet_hours=False)
    assert "23:30" in off["next"] and "quiet_hours" not in off


def test_set_weekly_and_daily():
    ctx = make_ctx()
    out = tools_of(ctx)["set_reminder"]("ส่งรายงาน", at="08:00", weekdays="จ")
    d = ctx.book.get(GUILD, out["id"])
    assert d["kind"] == "weekly" and d["weekdays"] == [0] and d["at_hhmm"] == "08:00"
    assert d["next_at"] == iso(at(8, day=19))
    # for a recurring reminder only the time of day of `at` counts, never its date
    daily = tools_of(ctx)["set_reminder"]("กินยา", at="2026-10-01 20:00", weekdays="daily")
    assert daily["ok"] and ctx.book.get(GUILD, daily["id"])["next_at"] == iso(at(20))
    ok = tools_of(ctx)["set_reminder"]("กินยา", at="20:00", weekdays="daily", until="2026-10-20")
    assert ok["ok"] and ctx.book.get(GUILD, ok["id"])["until"] == "2026-10-20"


def refused(out, code):
    assert out["ok"] is False and out["error"] == code and out["reason"]


def test_limits_and_refusals_never_crash():
    ctx = make_ctx()
    s = tools_of(ctx)["set_reminder"]
    refused(s("x", every="10m"), "min_interval")
    refused(s("x", at="2026-10-14 11:00"), "past")
    assert s("x", at="2026-10-14 11:59")["ok"]  # within the 1 minute grace
    refused(s("x", at="2027-11-01 09:00"), "too_far")
    assert s("x", at="2027-10-10 09:00")["ok"]
    refused(s("x", at="+400d"), "too_far")
    refused(s("x", every="2h", weekdays="mon", at="08:00"), "both")
    refused(s("x", weekdays="mon"), "weekly_needs_time")
    refused(s("x", weekdays="mon", at="+2h"), "weekly_needs_time")
    refused(s("x", weekdays="blah", at="08:00"), "bad_weekdays")
    refused(s("x"), "no_time")
    refused(s("x", at="soon"), "bad_time")
    refused(s("", at="+5m"), "empty")
    refused(s("x" * 400, at="+5m"), "too_long")
    refused(s("x", every="2h", until="2026-10-01"), "bad_until")
    refused(s("x", every="2h", until="nope"), "bad_until")
    refused(s("x", at="+5m", where="elsewhere"), "bad_where")
    refused(s("x", at="+5m", where="here"), "bad_where")  # no channel in this context
    en = make_ctx(lang="en")
    assert tools_of(en)["set_reminder"]("x", every="1m")["reason"].startswith("The shortest")


def test_twenty_active_reminders_per_person_cap():
    ctx = make_ctx()
    s = tools_of(ctx)["set_reminder"]
    for i in range(R.MAX_ACTIVE):
        assert s(f"r{i}", at="+1h")["ok"]
    refused(s("one more", at="+1h"), "cap")
    tools_of(ctx)["cancel_reminder"]("R-0001")
    assert s("one more", at="+1h")["ok"]
    other = make_ctx(uid=PEER, name="Peer", docs=ctx.book.docs)
    assert tools_of(other)["set_reminder"]("mine", at="+1h")["ok"]  # a separate cap per person


def test_for_user_is_executives_only():
    plain = make_ctx(is_exec=False)
    refused(tools_of(plain)["set_reminder"]("x", at="+1h", for_user="Peer"), "exec_only")
    assert tools_of(plain)["set_reminder"]("x", at="+1h", for_user="me")["ok"]
    boss = make_ctx(is_exec=True, uid=BOSS, name="Boss", docs=plain.book.docs)
    s = tools_of(boss)["set_reminder"]
    out = s("ส่งงานพรุ่งนี้", at="+1h", for_user="Peer")
    assert out["ok"] and out["for"] == "Peer"
    d = boss.book.get(GUILD, out["id"])
    assert d["user_id"] == PEER and d["set_by"] == BOSS and d["set_by_name"] == "Boss"
    refused(s("x", at="+1h", for_user="Nobody"), "person_not_found")
    refused(s("x", at="+1h", for_user="Pee"), "ambiguous_person")  # Peer / Peerapat
    assert s("x", at="+1h", for_user="Boss")["ok"]  # oneself by name: no "for"
    refused(s("โอนเงินเข้าบัญชีนี้ให้หน่อย", at="+1h", for_user="Somchai"), "sensitive")
    assert s("โอนเงินเข้าบัญชี", at="+1h")["ok"]  # one's own text is fine


def test_list_shows_own_and_execs_see_what_they_set_for_others():
    docs = SQLiteDocStore(":memory:")
    boss = make_ctx(is_exec=True, uid=BOSS, name="Boss", docs=docs)
    tools_of(boss)["set_reminder"]("ของบอส", at="+1h")
    tools_of(boss)["set_reminder"]("ฝากสมชาย", at="+2h", for_user="Somchai")
    staff = make_ctx(docs=docs)
    tools_of(staff)["set_reminder"]("ของสมชาย", at="+3h")
    mine = tools_of(staff)["list_reminders"]()
    assert mine["count"] == 2 and {x["text"] for x in mine["reminders"]} == {"ฝากสมชาย", "ของสมชาย"}
    theirs = tools_of(boss)["list_reminders"]()
    assert {x["text"] for x in theirs["reminders"]} == {"ของบอส", "ฝากสมชาย"}
    assert next(x for x in theirs["reminders"] if x["text"] == "ฝากสมชาย")["for"] == "Somchai"
    assert all({"id", "schedule", "next"} <= set(x) for x in theirs["reminders"])
    peer = make_ctx(uid=PEER, name="Peer", docs=docs)
    assert tools_of(peer)["list_reminders"]()["count"] == 0


def test_cancel_by_id_fragment_ambiguous_and_all():
    ctx = make_ctx()
    s, cancel = tools_of(ctx)["set_reminder"], tools_of(ctx)["cancel_reminder"]
    s("ออกกำลังกาย", every="2h")
    s("โทรหาลูกค้า A", at="+1h")
    s("โทรหาลูกค้า B", at="+2h")
    assert cancel("r-1")["cancelled"] == ["R-0001"]
    assert ctx.book.get(GUILD, "R-0001")["active"] is False
    amb = cancel("โทรหา")
    assert amb["ok"] is False and amb["error"] == "ambiguous"
    assert {c["id"] for c in amb["candidates"]} == {"R-0002", "R-0003"}
    assert cancel("ลูกค้า b")["cancelled"] == ["R-0003"]  # a fragment matching exactly one
    refused(cancel("nothing like this"), "not_found")
    refused(cancel("R-0001"), "not_found")  # already cancelled
    s("อื่นๆ", at="+1h")
    assert sorted(cancel("all")["cancelled"]) == ["R-0002", "R-0004"]
    assert ctx.book.active(GUILD) == []


def test_a_person_can_cancel_what_an_exec_set_for_them_and_not_others():
    docs = SQLiteDocStore(":memory:")
    boss = make_ctx(is_exec=True, uid=BOSS, name="Boss", docs=docs)
    rid = tools_of(boss)["set_reminder"]("ส่งงาน", at="+1h", for_user="Somchai")["id"]
    peer = make_ctx(uid=PEER, name="Peer", docs=docs)
    refused(tools_of(peer)["cancel_reminder"](rid), "not_found")
    assert tools_of(peer)["cancel_reminder"]("all")["cancelled"] == []
    staff = make_ctx(docs=docs)
    assert tools_of(staff)["cancel_reminder"](rid)["cancelled"] == [rid]
    again = tools_of(boss)["set_reminder"]("อีกอัน", at="+1h", for_user="Somchai")["id"]
    assert tools_of(boss)["cancel_reminder"](again)["cancelled"] == [again]  # the setter too


def test_firestore_docstore_works_too():
    from fake_firestore import FakeFirestore

    ctx = make_ctx(docs=FirestoreDocStore(client=FakeFirestore()))
    out = tools_of(ctx)["set_reminder"]("x", every="1h")
    assert out["ok"] and ctx.book.due(GUILD, NOW + timedelta(hours=1))[0]["id"] == out["id"]
    assert ctx.book.due(GUILD, NOW) == []


def test_i18n_keys_exist_in_both_languages():
    keys = [k for k in CATALOG if k.startswith("rem_")]
    assert len(keys) > 25
    for k in keys:
        assert CATALOG[k]["th"] and CATALOG[k]["en"], k
    codes = {
        "empty", "too_long", "no_time", "bad_time", "past", "too_far", "bad_every", "min_interval",
        "both", "bad_weekdays", "weekly_needs_time", "bad_until", "cap", "exec_only",
        "person_not_found", "ambiguous_person", "bad_where", "sensitive", "not_found", "ambiguous",
    }  # fmt: skip
    assert {f"rem_err_{c}" for c in codes} <= set(CATALOG)
    r = rec("interval", every_s=7200, next_at=iso(at(14)))
    th = R.fire_text(r, "th", NOW, nxt=at(14), who="")
    assert th.startswith("⏰ เตือนค่ะ: m") and "ทุก 2 ชั่วโมง" in th and "ยกเลิกเตือน R-0001" in th
    en = R.fire_text(r, "en", NOW, nxt=at(14), who="")
    assert en.startswith("⏰ Reminder: m") and "every 2 h" in en and "cancel reminder R-0001" in en
    assert "ฝากเตือน" in R.fire_text(r, "th", NOW, who="คุณพีท")
    assert R.fire_text(r, "th", NOW).count("\n") == 0  # no footer without a next time


# ------------------------------------------------------------------ the bot
def agent_tools(bot, world, member, is_exec=False, here=None, **kw):
    msg = bot.room_message(world.g, member, here or world.pub)
    k = bot._agent_kwargs(
        msg, world.g, member, is_exec, setting="th", lang="th", question="q", memory=None,
        profile=None, **kw,
    )  # fmt: skip
    return {f.__name__: f for f in k.get("extra_tools", [])}


def tick_at(bot, now):
    bot._reminders_last = float("-inf")
    return bot._reminders_due(now)


async def test_tools_are_in_the_agent_and_the_flag_turns_them_off(bot, world):
    me = world.g.members[STAFF]
    names = {"set_reminder", "list_reminders", "cancel_reminder"}
    assert names <= set(agent_tools(bot, world, me))
    assert not names & set(agent_tools(bot, world, me, interjection=True))
    set_features(bot, reminders=False)
    assert not names & set(agent_tools(bot, world, me))
    assert bot._reminder_tools(SimpleNamespace(channel=world.pub), world.g, me, False, "th") == []
    set_features(bot, reminders=True)
    assert names <= set(agent_tools(bot, world, me))


def test_voice_binds_the_reminder_tools_to_the_speaker_and_flag_exists():
    assert {"set_reminder", "cancel_reminder", "list_reminders"} <= ACTION_TOOL_NAMES
    from salid.guild_settings import FEATURES, Features

    assert "reminders" in FEATURES and Features().reminders is True


def test_prompt_section_follows_the_tool():
    assert "set_reminder" in stable_prompt("th", False, reminders=True)
    assert "+30m" in stable_prompt("th", False, reminders=True)  # the how-to section
    assert "+30m" not in stable_prompt("th", False)
    assert "ตั้งเตือนความจำ" in stable_prompt("th", False)  # the capabilities list always says so


async def test_where_here_posts_in_the_channel_mentioning_the_target(bot, world):
    boss, staff = world.g.members[BOSS], world.g.members[STAFF]
    staff.mention = f"<@{STAFF}>"
    tools = agent_tools(bot, world, boss, is_exec=True, here=world.pub)
    out = tools["set_reminder"]("ส่งรายงาน", at="+30m", for_user="Somchai", where="here")
    assert out["ok"]
    assert await tick_at(bot, at(12, 31)) == 1
    assert world.pub.sent and world.pub.sent[-1].startswith(f"<@{STAFF}> ⏰")
    assert "ส่งรายงาน" in world.pub.sent[-1] and "Boss ฝากเตือน" in world.pub.sent[-1]
    assert staff.dms == []  # not also sent privately


async def test_tick_fires_once_and_advances_a_recurring_reminder(bot, world):
    staff = world.g.members[STAFF]
    tools = agent_tools(bot, world, staff)
    tools["set_reminder"]("ออกกำลังกาย", every="2h")  # next 14:00
    tools["set_reminder"]("ประชุม", at="+10m")
    assert await tick_at(bot, at(12, 5)) == 0 and staff.dms == []
    assert await tick_at(bot, at(12, 10)) == 1
    assert "ประชุม" in staff.dms[0][0] and "ยกเลิกเตือน" not in staff.dms[0][0]
    assert await tick_at(bot, at(12, 11)) == 0  # once: done
    assert bot.reminder_book.get(GUILD, "R-0002")["active"] is False
    assert await tick_at(bot, at(14, 0)) == 1
    text = staff.dms[1][0]
    assert "ออกกำลังกาย" in text and "ทุก 2 ชั่วโมง" in text and "16:00" in text
    assert "R-0001" in text
    d = bot.reminder_book.get(GUILD, "R-0001")
    assert d["active"] and d["next_at"] == iso(at(16)) and d["fired"] == 1
    assert await tick_at(bot, at(14, 1)) == 0 and len(staff.dms) == 2


async def test_tick_is_throttled_by_the_bot_clock(bot, world):
    staff = world.g.members[STAFF]
    agent_tools(bot, world, staff)["set_reminder"]("x", at="+1m")
    assert await tick_at(bot, at(12, 2)) == 1
    agent_tools(bot, world, staff)["set_reminder"]("y", at="+1m")
    assert await bot._reminders_due(at(12, 5)) == 0  # clock did not move: under 30 s
    bot.clock.advance(31)
    assert await bot._reminders_due(at(12, 5)) == 1


async def test_a_late_once_reminder_is_still_sent_with_a_note(bot, world):
    staff = world.g.members[STAFF]
    agent_tools(bot, world, staff)["set_reminder"]("ต่อภาษี", at="+10m")  # due 12:10
    assert await tick_at(bot, at(14, 0)) == 1  # the bot was down for hours
    text = staff.dms[0][0]
    assert "ต่อภาษี" in text and "เลยเวลาที่ตั้งไว้" in text and "12:10" in text
    assert bot.reminder_book.get(GUILD, "R-0001")["active"] is False
    agent_tools(bot, world, staff)["set_reminder"]("ใกล้ๆ", at="+10m")  # due 12:10 (clock fixed)
    assert await tick_at(bot, at(12, 20)) == 1  # 10 minutes late: no note
    assert "เลยเวลา" not in staff.dms[1][0]


async def test_recurring_reminders_skip_missed_slots_without_a_burst(bot, world):
    staff = world.g.members[STAFF]
    agent_tools(bot, world, staff)["set_reminder"]("ดื่มน้ำ", every="1h", quiet_hours=False)
    # due 13:00; the bot comes back at 18:20, five slots later
    assert await tick_at(bot, at(18, 20)) == 0
    assert staff.dms == []
    d = bot.reminder_book.get(GUILD, "R-0001")
    assert d["active"] and d["next_at"] == iso(at(19, 0)) and d["fired"] == 0
    assert await tick_at(bot, at(19, 0)) == 1  # fires at the next future slot, once
    assert len(staff.dms) == 1


async def test_weekly_reminder_with_until_deactivates_after_the_last_day(bot, world):
    staff = world.g.members[STAFF]
    agent_tools(bot, world, staff)["set_reminder"](
        "รายงาน", at="09:00", weekdays="daily", until="2026-10-15"
    )  # first: 15 Oct 09:00
    assert await tick_at(bot, at(9, 0, day=15)) == 1
    d = bot.reminder_book.get(GUILD, "R-0001")
    assert d["active"] is False
    assert "ครั้งต่อไป" not in staff.dms[0][0]  # no next time: no footer


async def test_member_who_left_is_deactivated_not_sent(bot, world):
    staff = world.g.members[STAFF]
    agent_tools(bot, world, staff)["set_reminder"]("x", at="+5m")
    del world.g.members[STAFF]
    assert await tick_at(bot, at(12, 6)) == 0
    assert bot.reminder_book.get(GUILD, "R-0001")["active"] is False
    assert staff.dms == []


async def test_a_failing_delivery_does_not_stop_the_others(bot, world, monkeypatch):
    staff, peer = world.g.members[STAFF], world.g.members[PEER]
    agent_tools(bot, world, staff)["set_reminder"]("first", at="+5m")
    agent_tools(bot, world, peer)["set_reminder"]("second", at="+5m")
    real = bot.deliver_to_person

    async def flaky(guild, member, text, files=None, view=None):
        if member.id == STAFF:
            raise RuntimeError("discord is down")
        return await real(guild, member, text, files, view)

    monkeypatch.setattr(bot, "deliver_to_person", flaky)
    assert await tick_at(bot, at(12, 6)) == 1
    assert "second" in peer.dms[0][0] and staff.dms == []
    # the failed ONCE reminder was not lost: it is tried again shortly
    d = bot.reminder_book.get(GUILD, "R-0001")
    assert d["active"] is True


async def test_failed_once_reminder_retries_then_gives_up(bot, world, monkeypatch):
    from salid.dm import DMResult

    staff = world.g.members[STAFF]
    agent_tools(bot, world, staff)["set_reminder"]("x", at="+5m")

    async def closed(guild, member, text, files=None, view=None):
        return DMResult(False, "closed")

    monkeypatch.setattr(bot, "deliver_to_person", closed)
    for minutes in (6, 9, 12):
        assert await tick_at(bot, at(12, minutes)) == 0
    assert bot.reminder_book.get(GUILD, "R-0001")["active"] is False  # gave up after 3 tries


async def test_the_flag_off_holds_reminders_back(bot, world):
    staff = world.g.members[STAFF]
    agent_tools(bot, world, staff)["set_reminder"]("x", at="+5m")
    set_features(bot, reminders=False)
    assert await tick_at(bot, at(12, 6)) == 0 and staff.dms == []
    set_features(bot, reminders=True)
    assert await tick_at(bot, at(12, 7)) == 1


async def test_finished_reminders_are_pruned_after_a_month(bot, world):
    staff = world.g.members[STAFF]
    tools = agent_tools(bot, world, staff)
    tools["set_reminder"]("old", at="+5m")
    assert await tick_at(bot, at(12, 6)) == 1
    assert bot.reminder_book.get(GUILD, "R-0001")["active"] is False
    assert bot.reminder_book.prune(GUILD, at(12) + timedelta(days=29)) == 0
    assert bot.reminder_book.prune(GUILD, at(12) + timedelta(days=31)) == 1
    assert bot.reminder_book.get(GUILD, "R-0001") is None


async def test_a_reminder_to_a_thai_preferring_person_on_an_english_server_has_both_blocks(
    bot, world
):
    from salid.guild_settings import GuildSettingsUpdate
    from salid.profiles import Profile

    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(default_language="en"), "x")
    bot.team._cache.clear()
    bot.profiles.save(GUILD, STAFF, Profile(language="th"))
    staff = world.g.members[STAFF]
    agent_tools(bot, world, staff)["set_reminder"]("send the report", at="+10m")
    assert await tick_at(bot, at(12, 10)) == 1
    text = staff.dms[0][0]
    en, th = text.split("\n\n", 1)  # the company language first, then their own
    assert en.startswith("⏰ Reminder:") and th.startswith("⏰ เตือนค่ะ:")
    staff.dms.clear()
    bot.profiles.save(GUILD, STAFF, Profile(language="en"))  # same as the company: one block
    agent_tools(bot, world, staff)["set_reminder"]("again", at="+10m")
    assert await tick_at(bot, at(12, 20)) == 1
    assert "\n\n" not in staff.dms[0][0] and staff.dms[0][0].startswith("⏰ Reminder:")
