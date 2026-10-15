# ruff: noqa: F811, F401
"""Personal briefings: schedule (fake clock), content assembly, empty skip, the morning brief
replacing the follow-up DM, important-document alerts (targeting, access, rate limit, dedupe),
pending asks (detection, answered, one reminder), and the preferences tools. Offline."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import discord
import pytest
from conftest import GUILD, FakeClient, text_resp
from test_coordinator import (
    ADMIN_ID,
    BOSS,
    INTERN,
    OWNER_ID,
    PEER,
    PUBLIC,
    SECRET,
    STAFF,
    at,
    person,
    world,
)  # fmt: skip
from test_modes import Chan, TMsg, bot  # noqa: F401  (fixture)

from salid import briefings as bf
from salid.briefing_bot import BriefingMixin
from salid.chat_tools import ChatActions, build_chat_tools
from salid.coo_store import ALERTS, ASKS, STATE
from salid.guild_settings import GuildSettingsUpdate
from salid.i18n import t
from salid.tasks import Task

BKK = at(0).tzinfo


def utc_at(h, m=0, day=14):
    return at(h, m, day).astimezone(UTC)


def set_features(bot, **features):
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features=features), "x")
    bot.team._cache.clear()


def meeting(bot, mid, title, start, attendees, **extra):
    m = {
        "id": mid, "guild_id": GUILD, "title": title, "status": "scheduled", "kind": "voice",
        "voice_channel_id": 600, "attendee_ids": attendees, "attendee_names": ["x"] * len(attendees),
        "attended": [], "attendance_notes": {}, "dm_sent": {}, "duration_minutes": 60,
        "start_utc": start.astimezone(UTC).isoformat(timespec="seconds"),
        "end_utc": (start + timedelta(hours=1)).astimezone(UTC).isoformat(timespec="seconds"),
        **extra,
    }  # fmt: skip
    bot.secretary_store.put_meeting(m)
    return m


def seed_morning(bot):
    meeting(bot, "m1", "Weekly sync", at(10, 0), [STAFF, PEER])
    bot.tasks.save(Task(guild_id=GUILD, assignee_id=STAFF, title="ส่งใบเสนอราคา", due="2026-10-14",
                        id="aaaa1111"))  # fmt: skip
    bot.tasks.save(Task(guild_id=GUILD, assignee_id=STAFF, title="งานเก่า", due="2026-10-10"))


# ------------------------------------------------------------------ pure pieces
@pytest.mark.parametrize(
    "text, expected",
    [("8:30", "08:30"), ("08.30", "08:30"), ("7", "07:00"), ("17:00 น.", "17:00"), ("24:00", None),
     ("9:75", None), ("soon", None), ("", None), (None, None)],
)  # fmt: skip
def test_parse_hhmm(text, expected):
    assert bf.parse_hhmm(text) == expected


def test_schedule_weekdays_weekends_and_per_user_times():
    p = bf.NotifPrefs()
    wed, sat = at(8, 31, day=14), at(8, 31, day=17)  # 2026-10-14 is a Wednesday
    assert bf.brief_wanted("morning", wed, p, "08:30")
    assert not bf.brief_wanted("morning", at(8, 29), p, "08:30")  # before
    assert not bf.brief_wanted("morning", at(11, 31), p, "08:30")  # window (3 h) over
    assert not bf.brief_wanted("morning", sat, p, "08:30")  # weekdays only by default
    assert bf.brief_wanted("morning", sat, bf.NotifPrefs(weekends=True), "08:30")
    assert bf.brief_wanted("morning", at(7, 5), bf.NotifPrefs(morning_time="07:00"), "08:30")
    assert not bf.brief_wanted("morning", at(10, 5), bf.NotifPrefs(morning_time="07:00"), "08:30")
    assert not bf.brief_wanted("morning", wed, bf.NotifPrefs(morning=False), "08:30")
    assert bf.brief_wanted("evening", at(18, 1), p, "18:00")
    assert not bf.brief_wanted("evening", at(18, 1), bf.NotifPrefs(evening=False), "18:00")


def test_prefs_repo_validates_and_resets(docs):
    repo = bf.PrefsRepo(docs)
    p, err = repo.update(GUILD, STAFF, morning_time="7.30", doc_alerts="all", weekends=True)
    assert err == "" and (p.morning_time, p.doc_alerts, p.weekends) == ("07:30", "all", True)
    assert repo.get(GUILD, STAFF).morning_time == "07:30" and repo.get(GUILD, PEER).morning
    assert repo.update(GUILD, STAFF, evening_time="late")[0] is None
    assert repo.update(GUILD, STAFF, doc_alerts="some")[0] is None
    assert repo.update(GUILD, STAFF, morning="yes")[0] is None
    reset, _ = repo.update(GUILD, STAFF, morning_time="default")
    assert reset.morning_time == "" and set(repo.all(GUILD)) == {STAFF}


@pytest.fixture
def docs():
    from salid.coo_store import SQLiteDocStore

    return SQLiteDocStore(":memory:")


def test_brief_input_is_capped_and_skips_empty():
    many = [
        {"id": str(i), "title": "x" * 300, "due": "2026-10-14", "overdue": False} for i in range(40)
    ]
    inp = bf.BriefInput(kind="morning", name="Somchai", day="2026-10-14", tasks_due=many)
    text = bf.compact_text(inp)
    assert text.count("\n- ") <= bf.CAPS["tasks"] and len(text) <= bf.MAX_INPUT_CHARS + 50
    assert not inp.is_empty()
    assert bf.BriefInput(kind="morning", name="x", day="d").is_empty()
    # open tasks alone do not make an evening brief
    ev = bf.BriefInput(kind="evening", name="x", day="d", still_open=[{"id": "1", "title": "t"}])
    assert ev.is_empty()


def test_brief_falls_back_to_plain_when_the_model_fails_or_drops_a_link():
    inp = bf.BriefInput(kind="morning", name="Somchai", day="2026-10-14",
                        meetings=[{"time": "10:00", "title": "Sync", "link": "https://x.test/1"}])  # fmt: skip
    plain = bf.render_plain(inp, "en")
    assert "Sync" in plain and "https://x.test/1" in plain
    good = FakeClient([text_resp("Morning! Sync at 10:00 https://x.test/1")])
    assert bf.build_brief(good, "m", inp, "en").startswith("Morning!")
    dropped = FakeClient([text_resp("Morning! Sync at 10:00")])  # link lost: use the plain one
    assert bf.build_brief(dropped, "m", inp, "en") == plain
    assert bf.build_brief(None, "m", inp, "en") == plain  # model unavailable


# ------------------------------------------------------------------ morning / evening DMs
async def test_morning_brief_dms_members_with_something_to_say_once(bot, world):
    seed_morning(bot)
    assert await bot._briefings_due(at(8, 20)) == 0  # not yet
    assert await bot._briefings_due(at(8, 31)) == 2  # Somchai and Peer (the meeting)
    ((text, _),) = world.staff.dms
    assert "Weekly sync" in text and "10:00" in text and "https://discord.com/channels" in text
    assert "ส่งใบเสนอราคา" in text and "งานเก่า" in text and "#aaaa1111" in text
    assert world.peer.dms and "Weekly sync" in world.peer.dms[0][0]
    assert "งาน" not in world.peer.dms[0][0].split("นัดวันนี้")[-1].split("**")[0]
    assert world.intern.dms == [] and world.boss.dms == []  # nothing to say: no DM
    assert await bot._briefings_due(at(9, 0)) == 0  # once a day
    assert bot.docs.get(STATE, GUILD, f"brief:morning:{STAFF}:2026-10-14") is not None
    world.staff.dms.clear()
    assert await bot._briefings_due(at(8, 31, day=15)) == 1  # next day: Somchai's overdue tasks


async def test_morning_brief_is_composed_by_the_model_from_capped_input(bot, world):
    seed_morning(bot)
    bot.llm = FakeClient([text_resp("สวัสดีตอนเช้า https://x")] * 2)
    await bot._briefings_due(at(8, 31))
    prompt = bot.llm.models.calls[0][1]
    assert "Weekly sync" in prompt and "ส่งใบเสนอราคา" in prompt and len(prompt) < 4000
    # the model dropped the meeting link -> the deterministic brief goes out instead
    assert "https://discord.com/channels" in world.staff.dms[0][0]


async def test_empty_brief_sends_nothing_and_weekend_is_skipped(bot, world):
    assert await bot._briefings_due(at(8, 31)) == 0  # no data at all
    assert all(not m.dms for m in (world.staff, world.peer, world.boss))
    seed_morning(bot)
    assert await bot._briefings_due(at(8, 31, day=17)) == 0  # Saturday
    bot.prefs.update(GUILD, STAFF, weekends=True)
    bot._prefs_cache.clear()
    assert await bot._briefings_due(at(8, 32, day=17)) == 1


async def test_per_user_time_and_opt_out(bot, world):
    seed_morning(bot)
    bot.prefs.update(GUILD, STAFF, morning_time="07:00")
    bot.prefs.update(GUILD, PEER, morning=False)
    bot._prefs_cache.clear()
    assert await bot._briefings_due(at(7, 5)) == 1 and len(world.staff.dms) == 1
    assert await bot._briefings_due(at(8, 31)) == 0  # Peer opted out, Somchai already got it
    assert world.peer.dms == []


async def test_closed_dms_are_skipped_silently(bot, world):
    seed_morning(bot)

    async def closed(text=None, **kw):
        raise discord.Forbidden(SimpleNamespace(status=403, reason="x"), "closed")

    world.staff.send = closed
    assert await bot._briefings_due(at(8, 31)) == 1  # only Peer's went out; no crash


async def test_briefings_flag_off_sends_nothing(bot, world):
    seed_morning(bot)
    set_features(bot, briefings=False)
    assert await bot._briefings_due(at(8, 31)) == 0


async def test_the_morning_brief_replaces_the_old_ten_oclock_followup(bot, world):
    seed_morning(bot)
    assert await bot._followups_due(at(10, 5)) == 0  # briefings on: never both
    await bot._briefings_due(at(8, 31))
    assert len(world.staff.dms) == 1
    await bot._coo_due()  # the combined timer (real clock-free path) does not re-send either
    assert len(world.staff.dms) == 1
    set_features(bot, briefings=False)  # off: the old follow-up works again
    world.staff.dms.clear()
    assert await bot._followups_due(at(10, 5)) == 1


async def test_evening_brief_lists_the_day_and_tomorrow(bot, world):
    meeting(bot, "m2", "Kickoff", at(9, 0), [STAFF], status="done", attended=[STAFF],
                summary="ตกลงเริ่มโครงการวันจันทร์")  # fmt: skip
    meeting(bot, "m3", "Review", at(9, 0, day=15), [STAFF])
    bot.tasks.save(Task(guild_id=GUILD, assignee_id=STAFF, title="งานใหม่วันนี้", id="new1",
                        created_at=utc_at(11).isoformat()))  # fmt: skip
    bot.tasks.save(Task(guild_id=GUILD, assignee_id=STAFF, title="งานที่เสร็จ", status="done",
                        done_at=utc_at(15).isoformat()))  # fmt: skip
    bot.tasks.save(Task(guild_id=GUILD, assignee_id=STAFF, title="งานค้าง", due="2026-10-20"))
    assert await bot._briefings_due(at(18, 5)) == 1
    ((text, _),) = world.staff.dms
    for needle in ("Kickoff", "ตกลงเริ่มโครงการวันจันทร์", "งานใหม่วันนี้", "งานที่เสร็จ", "งานค้าง", "Review"):
        assert needle in text, needle
    assert world.peer.dms == []
    world.staff.dms.clear()
    # another day with only open tasks left: no evening brief
    assert await bot._briefings_due(at(18, 5, day=15)) == 0 and world.staff.dms == []


# ------------------------------------------------------------------ pending asks
@pytest.mark.parametrize(
    "text, expected",
    [("<@5> ช่วยดูสัญญาหน่อย", True), ("<@5> รบกวนส่งไฟล์ด้วยครับ", True), ("<@5> ขอยอดเดือนนี้", True),
     ("<@5> can you review this", True), ("<@5> please send the invoice", True),
     ("<@5> เสร็จยัง?", True), ("<@5> ฝากด้วยนะ", True),
     ("<@5> ขอบคุณมากครับ", False), ("<@5> สวัสดีครับ", False), ("<@5> thanks, great job", False),
     ("<@5> ขอโทษที", False), ("", False)],
)  # fmt: skip
def test_looks_like_ask(text, expected):
    assert bf.looks_like_ask(text) is expected


def ask_msg(world, bot, text, mentions, author=STAFF, channel=None, when=None, mid=None):
    ch = channel or world.pub
    m = TMsg(world.g, ch, world.g.members[author], text)
    m.raw_mentions = list(mentions)
    m.created_at = when or utc_at(9)
    m.jump_url = f"https://discord.com/channels/{GUILD}/{ch.id}/{m.id}"
    if mid:
        m.id = mid
    return m


async def test_ask_is_tracked_then_answered_by_a_later_post(bot, world):
    ask = ask_msg(world, bot, f"<@{PEER}> ช่วยดูสัญญาหน่อย", [PEER])
    await bot.on_message(ask)
    (open_ask,) = bot.asks.open_for(GUILD, PEER, utc_at(9, 30))
    assert open_ask["author_name"] == "Somchai" and open_ask["jump_url"] == ask.jump_url
    # the author's own follow-up, someone else's reply and a post in ANOTHER channel do not answer
    await bot.on_message(ask_msg(world, bot, "ขอบคุณ", [], author=STAFF))
    await bot.on_message(ask_msg(world, bot, "ผมดูให้", [], author=BOSS))
    await bot.on_message(ask_msg(world, bot, "ok", [], author=PEER, channel=world.secret))
    assert len(bot.asks.open_for(GUILD, PEER, utc_at(9, 30))) == 1
    await bot.on_message(ask_msg(world, bot, "ดูแล้วครับ", [], author=PEER))
    assert bot.asks.open_for(GUILD, PEER, utc_at(9, 30)) == []
    assert bot.docs.get(ASKS, GUILD, f"{ask.id}-{PEER}")["status"] == "answered"


async def test_only_user_mentions_that_ask_something_count(bot, world):
    await bot.on_message(ask_msg(world, bot, "<@&77> ช่วยดูหน่อย @everyone", []))  # roles / everyone
    await bot.on_message(ask_msg(world, bot, f"<@{PEER}> ขอบคุณครับ", [PEER]))  # not a request
    await bot.on_message(ask_msg(world, bot, f"<@{STAFF}> ช่วยหน่อย", [STAFF]))  # a self-mention
    await bot.on_message(ask_msg(world, bot, f"<@{INTERN}> ช่วยหน่อย", [INTERN, INTERN]))
    assert bot.asks.open_for(GUILD, PEER, utc_at(10)) == []
    assert bot.asks.open_for(GUILD, STAFF, utc_at(10)) == []
    assert len(bot.asks.open_for(GUILD, INTERN, utc_at(10))) == 1  # deduplicated


async def test_asks_in_off_channels_are_not_tracked(bot, world):
    bot.guilds_store.set_channel_mode(GUILD, PUBLIC, "off")
    bot.team._cache.clear()
    await bot.on_message(ask_msg(world, bot, f"<@{PEER}> ช่วยหน่อย", [PEER]))
    assert bot.docs.find(ASKS, GUILD) == []


async def test_a_reaction_answers_an_ask(bot, world):
    ask = ask_msg(world, bot, f"<@{PEER}> ช่วยดูหน่อย", [PEER])
    await bot.on_message(ask)
    await bot.on_raw_reaction_add(
        SimpleNamespace(guild_id=GUILD, message_id=ask.id, user_id=BOSS)  # someone else: no
    )
    assert len(bot.asks.open_for(GUILD, PEER, utc_at(10))) == 1
    await bot.on_raw_reaction_add(SimpleNamespace(guild_id=GUILD, message_id=ask.id, user_id=PEER))
    assert bot.asks.open_for(GUILD, PEER, utc_at(10)) == []


async def test_deleting_the_message_drops_the_ask(bot, world):
    ask = ask_msg(world, bot, f"<@{PEER}> ช่วยดูหน่อย", [PEER])
    await bot.on_message(ask)
    await bot.on_raw_message_delete(SimpleNamespace(guild_id=GUILD, message_id=ask.id))
    assert bot.asks.open_for(GUILD, PEER, utc_at(10)) == []


async def test_unanswered_ask_gets_one_gentle_reminder_after_two_hours(bot, world):
    await bot.on_message(ask_msg(world, bot, f"<@{PEER}> ช่วยดูสัญญาหน่อย", [PEER], when=utc_at(9, 30)))
    assert await bot._asks_due(at(11, 0)) == 0  # 1.5 h: too early
    assert await bot._asks_due(at(11, 31)) == 1
    ((text, _),) = world.peer.dms
    assert "Somchai" in text and "ช่วยดูสัญญาหน่อย" in text and "discord.com/channels" in text
    assert await bot._asks_due(at(13, 0)) == 0  # once
    assert len(world.peer.dms) == 1


async def test_answered_asks_and_opted_out_members_are_not_reminded(bot, world):
    await bot.on_message(ask_msg(world, bot, f"<@{PEER}> ช่วยดูหน่อย", [PEER], when=utc_at(8)))
    await bot.on_message(ask_msg(world, bot, "เรียบร้อย", [], author=PEER))
    await bot.on_message(ask_msg(world, bot, f"<@{INTERN}> ช่วยส่งไฟล์", [INTERN], when=utc_at(8)))
    bot.prefs.update(GUILD, INTERN, ask_reminders=False)
    bot._prefs_cache.clear()
    assert await bot._asks_due(at(12, 0)) == 0
    assert world.peer.dms == [] and world.intern.dms == []


async def test_no_reminder_in_the_middle_of_the_night_or_without_channel_access(bot, world):
    await bot.on_message(ask_msg(world, bot, f"<@{PEER}> ช่วยดูหน่อย", [PEER], when=utc_at(8)))
    assert await bot._asks_due(at(22, 30)) == 0  # quiet hours: waits
    assert await bot._asks_due(at(12, 0)) == 1  # sent when the day is back
    secret = ask_msg(world, bot, f"<@{INTERN}> ช่วยดูหน่อย", [INTERN], channel=world.secret,
                     author=BOSS, when=utc_at(8))  # fmt: skip
    await bot.on_message(secret)  # Ploy cannot read this channel: no reminder, no leak
    assert await bot._asks_due(at(12, 5)) == 0 and world.intern.dms == []


async def test_pending_asks_appear_in_the_morning_brief(bot, world):
    await bot.on_message(
        ask_msg(world, bot, f"<@{PEER}> ช่วยดูสัญญาหน่อย", [PEER], when=utc_at(17, day=13))
    )
    assert await bot._briefings_due(at(8, 31)) == 1
    text = world.peer.dms[0][0]
    assert "ช่วยดูสัญญาหน่อย" in text and "Somchai" in text and "discord.com/channels" in text


# ------------------------------------------------------------------ important-document alerts
def doc_msg(world, channel, author=STAFF, mentions=(), reference=None, mid=None):
    m = TMsg(world.g, channel, world.g.members[author], "ไฟล์ครับ")
    m.raw_mentions = list(mentions)
    m.jump_url = f"https://discord.com/channels/{GUILD}/{channel.id}/{m.id}"
    m.reference = reference
    return m


def info(doc_type="contract", amount=None, addressed=(), source="pdf", summary="สัญญาจ้าง ABC"):
    return bf.DocInfo("c.pdf", source, doc_type, summary, amount, list(addressed))


@pytest.fixture
def jemm(world):
    return person(world.g, 40, "Jemm", 2)


def test_classification_and_importance():
    assert bf.category("Contract") == "contract" and bf.category("สัญญาจ้าง") == "contract"
    assert bf.category("ใบเสนอราคา") == "quotation" and bf.category("tax invoice") == "invoice"
    assert bf.category("หนังสือราชการ") == "official" and bf.category("report") == "other"
    assert bf.is_important(info("contract"), 50000)
    assert bf.is_important(info("invoice", 60000), 50000)
    assert not bf.is_important(info("invoice", 1200), 50000)  # under the threshold
    assert not bf.is_important(info("report"), 50000)
    assert bf.is_important(info("report", addressed=["Jemm"]), 50000)  # addressed to someone
    assert bf.is_important(info("report"), 50000, mentioned=True)  # mentions a member
    assert not bf.is_important(info("meme", source="image"), 50000, mentioned=True)  # chat


async def test_a_contract_alerts_executives_who_can_read_the_channel_not_the_uploader(bot, world):
    world.boss.send_log = []
    await bot._doc_alert(doc_msg(world, world.pub, author=STAFF), 0, info("contract"))
    assert len(world.boss.dms) == 1 and len(world.admin.dms) == 1 and len(world.owner.dms) == 1
    text = world.boss.dms[0][0]
    assert "สัญญาจ้าง ABC" in text and "#ทั่วไป" in text and "discord.com/channels" in text
    assert world.staff.dms == [] and world.peer.dms == []  # the uploader / ordinary members: no


async def test_alerts_respect_channel_access(bot, world):
    # the secret channel is readable by owner, admin and boss only
    await bot._doc_alert(
        doc_msg(world, world.secret, author=BOSS, mentions=[INTERN, PEER]), 0, info("report")
    )
    assert world.intern.dms == [] and world.peer.dms == []  # mentioned but cannot read the channel
    await bot._doc_alert(doc_msg(world, world.secret, author=BOSS), 1, info("contract"))
    assert world.boss.dms == [] and len(world.admin.dms) == 1  # never the uploader himself


async def test_mentioned_members_and_addressees_are_told(bot, world, jemm):
    ref = SimpleNamespace(resolved=SimpleNamespace(author=world.peer), message_id=5)
    msg = doc_msg(world, world.pub, mentions=[INTERN], reference=ref)
    await bot._doc_alert(msg, 0, info("report", addressed=["Jemm"], summary="จดหมายถึงคุณเจม"))
    assert world.intern.dms and world.peer.dms and jemm.dms  # mention, replied-to, addressee
    assert world.boss.dms == []  # a report is not an executive document


async def test_unimportant_documents_alert_nobody_unless_asked(bot, world):
    await bot._doc_alert(doc_msg(world, world.pub), 0, info("report"))
    assert all(not m.dms for m in (world.boss, world.admin, world.peer, world.intern))
    bot.prefs.update(GUILD, PEER, doc_alerts="all")
    bot._prefs_cache.clear()
    await bot._doc_alert(doc_msg(world, world.pub), 1, info("report"))
    assert len(world.peer.dms) == 1 and world.boss.dms == []


async def test_doc_alert_preferences_off_and_important(bot, world):
    bot.prefs.update(GUILD, BOSS, doc_alerts="off")
    bot._prefs_cache.clear()
    await bot._doc_alert(doc_msg(world, world.pub), 0, info("contract"))
    assert world.boss.dms == [] and len(world.admin.dms) == 1


async def test_alerts_are_deduplicated_and_rate_limited_per_person(bot, world):
    bot.s.doc_alerts_per_day = 3
    msg = doc_msg(world, world.pub)
    await bot._doc_alert(msg, 0, info("contract"))
    await bot._doc_alert(msg, 0, info("contract"))  # the same document again
    assert len(world.boss.dms) == 1
    for n in range(1, 6):
        await bot._doc_alert(doc_msg(world, world.pub), 0, info("contract", summary=f"d{n}"))
    assert len(world.boss.dms) == 3  # the daily limit
    rows = bot.docs.find(ALERTS, GUILD)
    assert len(rows) == 6 and rows[0]["targets"]  # all are kept for the briefs


async def test_closed_dm_does_not_break_the_other_alerts(bot, world):
    async def closed(text=None, **kw):
        raise discord.Forbidden(SimpleNamespace(status=403, reason="x"), "closed")

    world.boss.send = closed
    assert await bot._doc_alert(doc_msg(world, world.pub), 0, info("contract")) == 2
    assert len(world.admin.dms) == 1 and len(world.owner.dms) == 1


async def test_important_documents_since_the_last_evening_go_into_the_morning_brief(bot, world):
    def rec(key, when, important=True):
        bot.alerts.record(
            GUILD, key,
            {"targets": [BOSS], "important": important, "kind": "contract", "summary": key,
             "channel_name": "ทั่วไป", "jump_url": "https://x/" + key, "at": bf._iso(when)},
        )  # fmt: skip

    rec("before-the-evening-brief", at(17, 0, day=14))  # already in yesterday's evening brief
    rec("after-the-evening-brief", at(19, 0, day=14))
    rec("not-important", at(20, 0, day=14), important=False)
    assert await bot._briefings_due(at(8, 31, day=15)) == 1
    ((text, _),) = world.boss.dms
    assert "after-the-evening-brief" in text and "https://x/after-the-evening-brief" in text
    assert "before-the-evening-brief" not in text and "not-important" not in text


async def test_alerts_use_the_documents_the_note_pipeline_analysed(bot, world, monkeypatch):
    from salid import docs as docs_mod

    note = docs_mod.DocNote(doc_type="contract", summary="สัญญา X", amount_value=10.0,
                            addressed_to="Jemm, Ploy")  # fmt: skip
    got = []
    monkeypatch.setattr(type(bot), "_doc_alert_soon", lambda s, m, i, info: got.append(info))
    monkeypatch.setattr(docs_mod, "extract_text", lambda data, kind: docs_mod.DocText("x" * 400, 3))
    monkeypatch.setattr(docs_mod, "analyze", lambda *a, **k: note)

    class Att(SimpleNamespace):
        async def read(self):
            return b"%PDF"

    msg = doc_msg(world, world.pub)
    att = Att(filename="c.pdf", content_type="application/pdf", size=10)
    msg.attachments = [att]
    await bot._doc_message(msg, [(0, att)])
    assert got and got[0].doc_type == "contract" and got[0].addressed == ["Jemm", "Ploy"]
    assert got[0].amount == 10.0


# ------------------------------------------------------------------ preferences tools
async def test_set_and_show_my_notifications_tools(bot, world):
    calls: list = []
    msg = TMsg(world.g, world.pub, world.staff, "ไม่ต้องส่งสรุปเย็น")
    actions = bot._chat_actions(msg, world.g, world.staff, False, "th", asyncio.get_running_loop())
    tools = {f.__name__: f for f in build_chat_tools(actions, calls)}
    out = tools["set_my_notifications"](evening=False, morning_time="7:30", weekends=True)
    assert out["status"] == "ok" and out["settings"]["evening_brief"] is False
    assert out["settings"]["morning_time"] == "07:30" and out["settings"]["weekends"] is True
    assert out["settings"]["evening_time"] == "18:00"  # the server default
    assert bot.prefs.get(GUILD, STAFF).evening is False
    assert bot.prefs.get(GUILD, PEER).evening is True  # per (guild, user)
    assert "error" in tools["set_my_notifications"](doc_alerts="sometimes")
    shown = tools["show_my_notifications"]()["settings"]
    assert shown["morning_time"] == "07:30" and shown["doc_alerts"] == "important"
    assert [c[0] for c in calls][:2] == ["set_my_notifications", "set_my_notifications"]


async def test_notification_tools_are_hidden_when_briefings_are_off(bot, world):
    set_features(bot, briefings=False)
    msg = TMsg(world.g, world.pub, world.staff, "x")
    actions = bot._chat_actions(msg, world.g, world.staff, False, "th", asyncio.get_running_loop())
    names = {f.__name__ for f in build_chat_tools(actions, [])}
    assert "set_my_notifications" not in names and "show_my_notifications" not in names
    assert "deliver_file" in names


def test_guild_defaults_for_brief_times():
    from salid.guild_settings import GuildSettings

    g = GuildSettings()
    assert (g.morning_brief_time, g.evening_brief_time) == ("08:30", "18:00")
    assert g.features.briefings is True
    with pytest.raises(ValueError):
        GuildSettingsUpdate(morning_brief_time="8am")
