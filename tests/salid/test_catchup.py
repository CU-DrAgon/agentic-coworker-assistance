# ruff: noqa: F811, F401
"""'What did I miss?': start-point parsing, scope (only readable channels), content and caps,
the agent tool (same in chat and voice), /ตามทัน, and the morning brief section. Offline."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from conftest import GUILD
from test_briefings import BKK, meeting, set_features
from test_coordinator import (
    BOSS,
    PEER,
    PUBLIC,
    SECRET,
    STAFF,
    SlashInteraction,
    add_msg,
    at,
    person,
    world,
)  # fmt: skip
from test_modes import bot  # noqa: F401  (fixture)

from salid import catchup as cu
from salid.i18n import t
from salid.memory_store import DaySummary
from salid.qa import MemoryTools
from salid.tasks import Task


# ------------------------------------------------------------------ pure pieces
@pytest.mark.parametrize(
    "text, expected_day",
    [("yesterday", 13), ("เมื่อวาน", 13), ("เมื่อวานซืน", 12), ("3 days", 11), ("ช่วง 3 วันที่ไม่อยู่", 11),
     ("1 week", 7), ("สัปดาห์ที่แล้ว", 7), ("monday", 12), ("วันจันทร์", 12), ("friday", 9),
     ("2026-10-01", 1), ("today", 14)],
)  # fmt: skip
def test_parse_since(text, expected_day):
    now = at(9)  # Wednesday 2026-10-14
    got = cu.parse_since(text, now, BKK)
    assert got is not None and got.day == expected_day and (got.hour, got.minute) == (0, 0)


def test_parse_since_gives_none_for_nonsense_and_empty():
    assert cu.parse_since("blah", at(9), BKK) is None and cu.parse_since("", at(9), BKK) is None
    assert cu.parse_since("2026-13-45", at(9), BKK) is None


def test_last_active_is_the_message_before_the_latest_silence():
    now = at(9)
    chatting = [at(8, 30), at(8, 0), at(6, 0), at(1, 0, day=13)]
    since, found = cu.last_active_since(chatting, now)
    assert found and since == at(1, 0, day=13)  # the current session does not count
    gone, _ = cu.last_active_since([at(18, 0, day=11)], now)
    assert gone == at(18, 0, day=11)
    assert cu.last_active_since([at(8, 0), at(7, 0)], now) == (None, True)  # always around
    assert cu.last_active_since([], now) == (None, False)


def test_working_days_away_skips_weekends_unless_asked():
    zone = BKK
    fri_eve, mon_morning = at(18, day=9), at(8, day=12)
    assert cu.working_days_away(fri_eve, mon_morning, zone) == 0  # only Sat + Sun between
    assert cu.working_days_away(fri_eve, mon_morning, zone, weekends=True) == 2
    assert cu.working_days_away(at(18, day=12), at(8, day=14), zone) == 1  # Tuesday
    assert cu.working_days_away(at(18, day=13), at(8, day=14), zone) == 0


# ------------------------------------------------------------------ scope and content
def tools_for(bot, world, member, allowed, now):
    (tool,) = cu.catchup_tools(bot._catchup_sources(world.g, member, allowed), lambda: now)
    return tool


def seed(bot, world):
    me = f"<@{STAFF}>"
    add_msg(
        bot, 1, STAFF, "Somchai", "ผมลาไปต่างจังหวัดนะ", PUBLIC, at(17, day=10)
    )  # last seen Sat? no: Mon
    add_msg(bot, 2, PEER, "Peer", f"{me} ช่วยดูใบเสนอราคา ABC ด้วย", PUBLIC, at(10, day=13))
    add_msg(bot, 3, PEER, "Peer", f"{me} เรื่องลับ งบปี 2027", SECRET, at(11, day=13))
    add_msg(bot, 4, PEER, "Peer", f"{me} [PDF: สัญญา.pdf]", PUBLIC, at(12, day=13))
    add_msg(bot, 5, BOSS, "Boss", "ไม่เกี่ยวกับใคร", PUBLIC, at(13, day=13))


def test_tool_returns_only_readable_channels_with_links_and_ignores_the_secret_one(bot, world):
    seed(bot, world)
    tool = tools_for(bot, world, world.staff, {PUBLIC: "ทั่วไป"}, at(9))
    data = tool(None)
    assert data["start_was_guessed_from_last_activity"] is True
    assert data["since"].startswith("2026-10-10")  # his last message before the long silence
    texts = [m["text"] for m in data["mentions"]]
    assert any("ใบเสนอราคา ABC" in x for x in texts)
    assert all("งบปี 2027" not in x for x in texts) and "SECRET" not in str(data)
    assert data["files"] and "สัญญา.pdf" in data["files"][0]["text"]
    assert (
        data["messages_in_readable_channels"] == 3
    )  # his own and the secret channel are not counted
    assert "<@" not in str(data["mentions"])  # the mention is rendered by name


def test_explicit_since_overrides_last_activity_and_bad_since_is_an_error(bot, world):
    seed(bot, world)
    tool = tools_for(bot, world, world.staff, {PUBLIC: "ทั่วไป"}, at(9))
    assert tool("yesterday")["since"].startswith("2026-10-13")
    assert "error" in tool("whenever the moon is full")
    assert tool("2026-10-14").get("note")  # nothing since this morning


def test_tasks_decisions_highlights_and_meetings(bot, world):
    seed(bot, world)
    bot.tasks.save(Task(guild_id=GUILD, assignee_id=STAFF, title="ส่งใบเสนอราคา", due="2026-10-12",
                        id="aaaa1111", created_at=at(9, day=13).astimezone(UTC).isoformat()))  # fmt: skip
    bot.tasks.save(Task(guild_id=GUILD, assignee_id=PEER, title="งานของเป้ย", id="bbbb2222"))
    bot.memory_store.put_summary(DaySummary(
        guild_id=GUILD, channel_id=PUBLIC, local_date="2026-10-13", message_count=12, final=True,
        summary={"overview": "คุยเรื่องแคมเปญเดือนหน้า", "decisions": ["เลื่อนเปิดตัวเป็น 1 พ.ย."],
                 "action_items": [{"who": "Somchai", "what": "ทำสไลด์", "due": "2026-10-20"}]},
    ))  # fmt: skip
    meeting(
        bot,
        "mm",
        "Kickoff",
        at(14, day=13),
        [STAFF, PEER],
        status="done",
        summary_url="https://m.test/1",
    )
    meeting(bot, "mn", "Retro", at(10, day=15), [STAFF], status="scheduled")
    data = tools_for(bot, world, world.staff, {PUBLIC: "ทั่วไป"}, at(9))(None)
    assert [x["id"] for x in data["tasks"]] == ["aaaa1111"] and data["tasks"][0]["overdue"]
    assert data["decisions"][0].get("assigned") and "ทำสไลด์" in data["decisions"][0]["text"]
    assert any("เลื่อนเปิดตัว" in d["text"] for d in data["decisions"])
    assert data["channels"][0]["messages"] == 12 and "แคมเปญ" in data["channels"][0]["text"]
    assert data["missed"] == [
        {"time": "13/10 14:00", "title": "Kickoff", "minutes": "https://m.test/1"}
    ]
    assert [m["title"] for m in data["upcoming"]] == ["Retro"]


def test_caps_keep_the_result_short(bot, world):
    me = f"<@{STAFF}>"
    for i in range(30):
        add_msg(bot, 100 + i, PEER, "Peer", f"{me} ข้อ {i}", PUBLIC, at(10, i, day=13))
    data = tools_for(bot, world, world.staff, {PUBLIC: "ทั่วไป"}, at(9))("yesterday")
    assert len(data["mentions"]) == cu.CAPS["mentions"]
    assert "ข้อ 29" in data["mentions"][0]["text"]  # newest first


def test_render_is_bilingual_and_short_form_is_two_to_three_lines(bot, world):
    seed(bot, world)
    src = bot._catchup_sources(world.g, world.staff, {PUBLIC: "ทั่วไป"})
    data = cu.gather(src, at(0, day=13), at(9))
    th, en = cu.render(data, "th", BKK), cu.render(data, "en", BKK)
    assert "อยากดูละเอียดห้องไหนบอกได้นะคะ" in th and "Tell me which channel" in en
    assert "ใบเสนอราคา ABC" in th and "https://" not in "งบปี 2027"
    short = cu.render(data, "th", BKK, short=True)
    assert 2 <= len(short.splitlines()) <= 4 and "งบปี 2027" not in short
    empty = cu.gather(src, at(9, day=14), at(9, 5))
    assert cu.render(empty, "en", BKK).startswith("Nothing new")


# ------------------------------------------------------------------ the bot
async def test_agent_kwargs_register_catch_up_and_the_flag_turns_it_off(bot, world):
    mem = MemoryTools(bot.memory_store, GUILD, {PUBLIC: "ทั่วไป"}, bot.s, 30)
    names = lambda: {f.__name__ for f in bot._catchup_tools(world.g, world.staff, mem)}  # noqa: E731
    assert names() == {"catch_up_tool"}
    set_features(bot, catchup=False)
    assert names() == set()
    assert bot._catchup_tools(world.g, world.staff, None) == []


async def test_slash_command_is_ephemeral_and_scoped(bot, world):
    now = bot.s.now()
    day = (now - timedelta(days=1)).replace(hour=10, minute=0)
    add_msg(bot, 7, PEER, "Peer", f"<@{STAFF}> ดูสัญญาด้วย", PUBLIC, day)
    add_msg(bot, 8, PEER, "Peer", f"<@{STAFF}> ลับสุดยอด", SECRET, day)
    it = SlashInteraction(world.g, world.staff, world.pub)
    await bot.handle_catchup(it, "2 days")
    text = "".join(s for s, _ in it.sent)
    assert it.sent and all(eph for _, eph in it.sent)
    assert "ดูสัญญาด้วย" in text and "ลับสุดยอด" not in text
    bad = SlashInteraction(world.g, world.staff, world.pub)
    await bot.handle_catchup(bad, "never ever")
    assert bad.sent == [(t("cu_bad_since", "th"), True)]
    set_features(bot, catchup=False)
    off = SlashInteraction(world.g, world.staff, world.pub)
    await bot.handle_catchup(off, None)
    assert off.sent == [(t("cu_off", "th"), True)]


async def test_morning_brief_adds_a_short_catch_up_only_after_a_working_day_away(bot, world):
    add_msg(bot, 1, STAFF, "Somchai", "เจอกันนะ", PUBLIC, at(17, day=12))  # Monday evening
    add_msg(bot, 2, PEER, "Peer", f"<@{STAFF}> ช่วยดูสัญญาหน่อย", PUBLIC, at(10, day=13))
    assert await bot._briefings_due(at(8, 31)) >= 1  # Wednesday: Tuesday was a whole day away
    text = world.staff.dms[0][0]
    assert "สิ่งที่คุณพลาด" in text and "ช่วยดูสัญญาหน่อย" in text
    assert "อยากดูละเอียดห้องไหนบอกได้นะคะ" in text


async def test_morning_brief_has_no_catch_up_when_the_person_was_around(bot, world):
    add_msg(bot, 1, STAFF, "Somchai", "ผมอยู่นี่", PUBLIC, at(17, day=13))  # yesterday evening
    add_msg(bot, 2, PEER, "Peer", f"<@{STAFF}> ช่วยดูสัญญาหน่อย", PUBLIC, at(18, day=13))
    meeting(bot, "m1", "Weekly sync", at(10, 0), [STAFF])
    await bot._briefings_due(at(8, 31))
    assert "Weekly sync" in world.staff.dms[0][0] and "สิ่งที่คุณพลาด" not in world.staff.dms[0][0]
    set_features(bot, catchup=False)  # and the flag removes it for the long-absent too
