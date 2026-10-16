# ruff: noqa: F811, F401
"""Incident 16 Oct 01:19 UTC: member C asked "Salid สรุปประชุมเมื่อคืนให้หน่อย" in a public chat
channel and was told the meeting records were restricted. Voice transcripts of a channel everyone
can read are public; only the owner's DM notes are protected, and they are a footnote, never a
reason to refuse the whole summary. Fake data mirrors the real shape (ids, kinds, times)."""

from datetime import UTC, datetime
from types import SimpleNamespace

from conftest import GUILD
from test_coordinator import RChan, at, person
from test_coordinator import world as world  # noqa: PLC0414  (fixture)
from test_modes import bot  # noqa: F401  (fixture)

from salid.memory import build_message
from salid.memory_access import dm_scope
from salid.qa import MemoryTools, system_prompt

CHAT, VOICE = 1560242607320342560, 1548606560001982467
PASSIN, C = 845668986125025280, 1090215946817376276
SALID = 1560235651545309284


def put(bot, mid, cid, uid, name, text, when_utc, kind="chat", visibility="normal"):
    m = build_message(
        bot.s, guild_id=GUILD, channel_id=cid, message_id=mid, author_id=uid, author_name=name,
        created_at=when_utc, text=text, kind=kind,
    )  # fmt: skip
    bot.memory_store.add_message(m.model_copy(update={"visibility": visibility}))


def seed(bot):
    d = lambda day, h, m: datetime(2026, 10, day, h, m, tzinfo=UTC)  # noqa: E731
    # 23:26 Bangkok on the 15th: local_date 15th; 01:07 Bangkok: local_date 16th
    put(bot, 1, VOICE, SALID, "Salid", "ยินดีที่ได้รู้จัก ประชุมทีมเรื่อง NeurIPS", d(15, 16, 26), "salid")
    put(
        bot,
        2,
        VOICE,
        PASSIN,
        "passin",
        "ประชุมคืนนี้: ตกลงงบทริป NeurIPS 50,000",
        d(15, 18, 16),
        "voice",
    )
    put(
        bot,
        3,
        VOICE,
        1229824117914599466,
        "dafssd",
        "ประชุมเรื่อง Pointer 2 มิติ",
        d(15, 18, 17),
        "voice",
    )
    put(bot, 4, dm_scope(PASSIN), PASSIN, "passin", "ประกาศประชุมเร่งด่วน 9 โมง", d(15, 19, 40), "dm")
    put(bot, 5, dm_scope(PASSIN), PASSIN, "passin", "ประชุมนี้ส่วนตัว", d(15, 19, 43), "dm", "private")


async def as_c(bot, world):
    g = world.g
    g.id = GUILD
    g.text_channels[:] = [RChan(CHAT, "คุยกับน้องสลิด")]
    voice = RChan(VOICE, "General")  # no overwrites: every member can read it
    g.voice_channels = [voice]
    for ch in (*g.text_channels, voice):
        ch.guild = g
    g.owner_id = PASSIN
    person(g, PASSIN, "passin", 9)
    c = person(g, C, "C", 2)
    bot.team._cache.clear()
    # the seeded records are from the 15th/16th: the clock must not be before them (hybrid
    # search only scans raw messages of days that already happened)
    object.__setattr__(bot.s, "now", lambda: at(12, day=16))
    seed(bot)
    where = SimpleNamespace(guild=g, channel=g.text_channels[0])
    return await bot._memory_tools(where, g, c)


async def test_public_voice_transcripts_are_summarisable_by_a_lower_ranked_member(bot, world):
    tools = await as_c(bot, world)
    assert tools.allowed == {CHAT: "คุยกับน้องสลิด", VOICE: "General"}
    r = tools.chat_messages("2026-10-15", "2026-10-16")
    assert {m["text"] for m in r["messages"]} >= {
        "ประชุมคืนนี้: ตกลงงบทริป NeurIPS 50,000",
        "ประชุมเรื่อง Pointer 2 มิติ",
    }  # the owner outranks C, yet her spoken words in a public room are shared
    s = tools.chat_search(["ประชุม", "meeting"], "2026-10-15", "2026-10-16")
    assert {h["channel"] for h in s["hits"]} == {"General"}
    assert not [m for h in s["hits"] for m in h["messages"] if "ส่วนตัว" in m["text"]]


async def test_only_the_dm_notes_are_withheld_and_the_note_is_a_footnote_not_a_refusal(bot, world):
    tools = await as_c(bot, world)
    s = tools.chat_search(["ประชุม"], "2026-10-15", "2026-10-16")
    assert s["count"] >= 1  # the public part is returned...
    assert {m.channel_id for m in tools.hidden} == {dm_scope(PASSIN)}  # ...DM notes only withheld
    note = s["withheld"]
    assert "answer the question fully" in note and "never refuse" in note
    assert "NOT allowed" not in note and "cannot share" not in note


async def test_last_night_spans_two_local_dates(bot, world):
    """15 Oct 23:26-01:07 Bangkok is stored under 15th and 16th: searching only 'yesterday' misses
    the meeting, so the prompt tells the model to cover both dates."""
    tools = await as_c(bot, world)
    assert tools.chat_messages("2026-10-15", "2026-10-15", channel_name="General")["count"] == 1
    assert tools.chat_messages("2026-10-16", "2026-10-16", channel_name="General")["count"] == 2
    p = system_prompt(bot.s, scoped=True, memory=True, coo=True, now=at(8, 19, day=16))
    assert "เมื่อคืน" in p and "ทั้งวันที่เมื่อวานและวันนี้" in p
    assert "ห้ามปฏิเสธทั้งคำตอบ" in p


async def test_since_until_and_channel_filters_cover_the_night_not_just_the_newest(bot, world):
    tools = await as_c(bot, world)
    night = tools.chat_messages(
        "2026-10-15", "2026-10-16", channel_name="General",
        since="2026-10-15T23:00", until="2026-10-16T01:30",
    )  # fmt: skip
    assert [m["time"][-5:] for m in night["messages"]] == ["01:17", "01:16", "23:26"]  # newest 1st
    assert "since/until" not in night.get("note", "")
    few = tools.chat_messages("2026-10-15", "2026-10-16", channel_name="General", limit=1)
    assert few["count"] == 1 and "narrow with since/until" in few["note"]
    nope = tools.chat_days("2026-10-15", "2026-10-16", channel_name="nope")
    assert nope["count"] == 0 and "no channel" in nope["note"]
