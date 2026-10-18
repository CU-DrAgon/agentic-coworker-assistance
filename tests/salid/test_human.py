# ruff: noqa: F811, F401
"""Salid as a person: acknowledgement of slow work, tagging and DMs, settings by talking, quiet
read receipts, receipts learned from content (never from channel names). Offline."""

import asyncio
import threading
from types import SimpleNamespace

import discord
import pytest
from conftest import GUILD, FakeClient
from test_bot_multitenant import ME, member
from test_memory_import import FakeAtt, att, iw, note_resp, send_images, set_llm, settle
from test_modes import EXEC, STAFF, Sent, TMsg, bot, set_mode, world
from test_salid_files import Agent, wire

from salid import ack as ackmod
from salid import bot as botmod
from salid.chat_tools import ChatActions, build_chat_tools
from salid.extract import IngestResult
from salid.i18n import t
from salid.mentions import (
    allowed_mentions_for,
    find_member,
    match_members,
    resolve_mentions,
    tag_candidates,
    wants_everyone,
)
from salid.qa import system_prompt


def closed_dm():
    return discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), "Cannot send")


def recording(m):
    """Texts passed to m.reply, in order, plus the kwargs of each call."""
    texts, kws = [], []
    orig = m.reply

    async def reply(text, **kw):
        texts.append(text)
        kws.append(kw)
        return await orig(text, **kw)

    m.reply = reply
    return SimpleNamespace(texts=texts, kws=kws)


async def until(cond, tries=300):
    for _ in range(tries):
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true")


def people(world, *specs):
    """Guild members with a display name: specs are (uid, name, extra attrs)."""
    out = []
    for uid, name, *rest in specs:
        m = member(world.g, uid)
        m.display_name = name
        m.sent, m.dm_error = [], None
        for k, v in (rest[0] if rest else {}).items():
            setattr(m, k, v)

        async def send(text, files=(), _m=m, **kw):
            if _m.dm_error:
                raise _m.dm_error
            _m.sent.append((text, [f.filename for f in files]))

        m.send = send
        out.append(m)
    return out


# ------------------------------------------------------------------ mentions: resolution
def test_mention_resolution_exact_prefix_ambiguous_unknown(world):
    mew, mei, ann, ann2 = people(
        world, (21, "Mew", {"global_name": "Mew Mew"}), (22, "Mei"), (23, "Ann"), (24, "annie")
    )
    g = world.g
    assert resolve_mentions("hi @[mew]", g) == "hi <@21>"  # case-insensitive exact
    assert resolve_mentions("@[Mew Mew]", g) == "<@21>"  # global name
    assert resolve_mentions("@[Me]", g) == "@Me"  # prefix of two (Mew, Mei): ambiguous
    assert resolve_mentions("@[Ann]", g) == "<@23>"  # exact beats the prefix of annie
    assert resolve_mentions("@[ann", g) == "@[ann"  # not markup
    assert resolve_mentions("@[Mei]", g) == "<@22>"
    assert resolve_mentions("@[Mew", g) == "@[Mew"
    assert resolve_mentions("@[annie]", g) == "<@24>"
    assert resolve_mentions("@[Zed]", g) == "@Zed"  # unknown: plain text
    assert resolve_mentions("@[Mi]", g) == "@Mi"  # no member starts with "Mi" except Mei: unique
    assert [m.id for m in match_members(g, "Me")] == [21, 22]
    assert find_member(g, "Me") is None and find_member(g, "mei").id == 22


def test_prefix_that_is_unique_resolves(world):
    people(world, (21, "Mewmew"), (23, "Ann"))
    assert resolve_mentions("@[Mew]", world.g) == "<@21>"


def test_role_mentions_only_when_mentionable(world):
    world.g.roles = [
        SimpleNamespace(id=7, name="Sales", mentionable=True),
        SimpleNamespace(id=8, name="Admins", mentionable=False),
    ]
    assert resolve_mentions("@[Sales] ready", world.g) == "<@&7> ready"
    assert resolve_mentions("@[Admins] ready", world.g) == "@Admins ready"
    allowed = allowed_mentions_for("<@&7> <@5>")
    assert allowed.users is True and [r.id for r in allowed.roles] == [7]
    assert allowed.everyone is False and allowed.replied_user is False
    assert allowed_mentions_for("hello").roles is False


def test_everyone_and_here_are_stripped_unless_allowed(world):
    g = world.g
    assert resolve_mentions("@everyone and @here: meeting", g) == "everyone and here: meeting"
    assert resolve_mentions("@everyone", g, allow_everyone=True) == "@everyone"
    assert allowed_mentions_for("@everyone", allow_everyone=True).everyone is True
    assert allowed_mentions_for("@everyone").everyone is False
    assert wants_everyone("แจ้งทุกคนว่าประชุมพรุ่งนี้", True)
    assert wants_everyone("ping @everyone please", True)
    assert not wants_everyone("แจ้งทุกคนว่าประชุมพรุ่งนี้", False)  # not a manager / executive
    assert not wants_everyone("ฝากบอกมิวว่าลูกค้าโทรมา", True)  # never asked for everyone


def test_tag_candidates_recent_people_first_and_capped(world):
    ch = SimpleNamespace(
        members=[SimpleNamespace(display_name=f"m{i}", bot=False) for i in range(80)]
        + [SimpleNamespace(display_name="botty", bot=True)]
    )
    names = tag_candidates(ch, ["Zed", "สลิด", "m5"])
    assert names[:2] == ["Zed", "m5"] and "สลิด" not in names and "botty" not in names
    assert len(names) == 50 and names.count("m5") == 1


# ------------------------------------------------------------------ mentions in the bot
async def test_answer_tags_people_and_sends_allowed_mentions(bot, world, monkeypatch):
    w = wire(bot, monkeypatch)
    (mew,) = people(world, (21, "Mew"))
    world.chat.members = [mew]
    monkeypatch.setattr(
        botmod,
        "ask",
        lambda s, store, q, **kw: (
            w.agent.calls.append((q, kw))
            or SimpleNamespace(text="@[Mew] ลูกค้าโทรมา (คุณเอฝากบอก)", seconds=0, rounds=0)
        ),
    )
    m = TMsg(world.g, world.chat, member(world.g, STAFF), "ฝากบอกมิวว่าลูกค้าโทรมา")
    rec = recording(m)
    await bot.on_message(m)
    assert rec.texts == ["<@21> ลูกค้าโทรมา (คุณเอฝากบอก)"]
    allowed = rec.kws[0]["allowed_mentions"]
    assert allowed.users is True and allowed.everyone is False and allowed.replied_user is False
    (_q, kw) = w.agent.calls[0]
    assert kw["people"] == ["Mew"]  # names only, never ids


async def test_everyone_ping_needs_a_manager_who_asked(bot, world, monkeypatch):
    wire(bot, monkeypatch)
    monkeypatch.setattr(
        botmod,
        "ask",
        lambda s, store, q, **kw: SimpleNamespace(text="@everyone ประชุมนะ", seconds=0, rounds=0),
    )
    boss = TMsg(world.g, world.chat, member(world.g, EXEC), "แจ้งทุกคนว่าประชุมพรุ่งนี้")
    rec = recording(boss)
    await bot.on_message(boss)
    assert rec.texts == ["@everyone ประชุมนะ"] and rec.kws[0]["allowed_mentions"].everyone is True
    staff = TMsg(world.g, world.chat, member(world.g, STAFF), "แจ้งทุกคนว่าประชุมพรุ่งนี้")
    rec = recording(staff)
    await bot.on_message(staff)
    assert rec.texts == ["everyone ประชุมนะ"] and rec.kws[0]["allowed_mentions"].everyone is False


def test_prompt_lists_people_and_the_mention_markup(settings):
    p = system_prompt(settings, scoped=False, people=["มิว", "เอ"])
    assert "@[มิว]" in p and "คนที่แท็กได้ (ชื่อที่แสดง): มิว, เอ" in p
    assert "ฝากบอก" in p and "@everyone" in p
    assert "คนที่แท็กได้" not in system_prompt(settings, scoped=False)  # no list: no tagging


# ------------------------------------------------------------------ send_dm tool
async def test_dm_to_me_sends_plain_text(bot, world):
    (me,) = people(world, (STAFF, "Som"))
    r = await bot._send_dm_tool(world.g, me, "สรุป: ยอด 500", "me", None, "th")
    assert r == {"status": "sent", "to": "requester"}
    assert me.sent == [("สรุป: ยอด 500", [])]


async def test_dm_to_another_person_is_a_relay_that_says_who_it_is_from(bot, world):
    som, mew = people(world, (STAFF, "Som"), (21, "Mew"))
    r = await bot._send_dm_tool(world.g, som, "ลูกค้าโทรมา", "Mew", None, "th")
    assert r == {"status": "sent", "to": "Mew"}
    ((text, _),) = mew.sent
    assert "Som" in text and "ลูกค้าโทรมา" in text and "ผ่านสลิด" in text
    assert som.sent == []


async def test_dm_ambiguous_and_unknown_names(bot, world):
    som, a1, a2 = people(world, (STAFF, "Som"), (21, "Mat"), (22, "Mal"))
    r = await bot._send_dm_tool(world.g, som, "x", "Ma", None, "th")
    assert r["error"] == "ambiguous" and sorted(r["candidates"]) == ["Mal", "Mat"]
    r = await bot._send_dm_tool(world.g, som, "x", "Zed", None, "th")
    assert r["error"] == "person_not_found"
    assert a1.sent == a2.sent == []


async def test_dm_closed_gives_the_guidance(bot, world):
    som, mew = people(world, (STAFF, "Som"), (21, "Mew", {"dm_error": closed_dm()}))
    r = await bot._send_dm_tool(world.g, som, "x", "Mew", None, "th")
    assert r["error"] == "dms_closed" and "Privacy Settings" in r["guidance"]
    assert "Mew" in r["guidance"] and world.g.name in r["guidance"]
    r = await bot._send_dm_tool(world.g, som, "x", "Mew", None, "en")
    assert "turned off" in r["guidance"]
    assert bot._dm_sent == {}  # a failed DM does not count


async def test_dm_attaches_a_pdf_for_me_and_removes_it_from_the_channel_reply(bot, world):
    (som,) = people(world, (STAFF, "Som"))
    bot.secretary_state.remember_pdf(GUILD, STAFF, "report.pdf", b"%PDF-1")
    secretary = SimpleNamespace(attachments=[("report.pdf", b"%PDF-1"), ("other.pdf", b"x")])
    r = await bot._send_dm_tool(world.g, som, "นี่ครับ", "me", "report.pdf", "th", secretary)
    assert r["status"] == "sent" and som.sent == [("นี่ครับ", ["report.pdf"])]
    assert secretary.attachments == [("other.pdf", b"x")]  # delivered privately only
    r = await bot._send_dm_tool(world.g, som, "x", "me", "nope.pdf", "th")
    assert r["error"] == "attachment_not_found"


async def test_dm_pdf_to_others_only_when_they_can_see_at_least_as_much(bot, world):
    som, boss, mew = people(world, (STAFF, "Som"), (EXEC, "Boss"), (21, "Mew"))
    bot.secretary_state.remember_pdf(GUILD, STAFF, "report.pdf", b"%PDF-1")
    r = await bot._send_dm_tool(world.g, som, "รายงาน", "Mew", "report.pdf", "th")
    assert r["error"] == "attachment_only_for_requester_or_executives" and mew.sent == []
    r = await bot._send_dm_tool(world.g, som, "รายงาน", "Boss", "report.pdf", "th")
    assert r["status"] == "sent" and boss.sent[0][1] == ["report.pdf"]


async def test_dm_rate_limit_is_per_guild_per_hour(bot, world):
    (som,) = people(world, (STAFF, "Som"))
    for _ in range(botmod.DM_PER_GUILD_PER_HOUR):
        assert (await bot._send_dm_tool(world.g, som, "x", "me", None, "th"))["status"] == "sent"
    assert (await bot._send_dm_tool(world.g, som, "x", "me", None, "th"))["error"] == "rate_limited"
    bot._clock = lambda: 10**9 + 4000  # an hour later
    assert (await bot._send_dm_tool(world.g, som, "x", "me", None, "th"))["status"] == "sent"


async def test_shared_dm_helper_survives_closed_dms():
    from salid.dm import send_dm

    class M:
        id = 1

        def __init__(self, err):
            self.err = err

        async def send(self, text, **kw):
            if self.err:
                raise self.err

    assert (await send_dm(M(None), "hi")).ok
    assert (await send_dm(M(closed_dm()), "hi")).reason == "closed"
    http = discord.HTTPException(SimpleNamespace(status=500, reason="x"), "x")
    assert (await send_dm(M(http), "hi")).reason == "error"
    assert (await send_dm(None, "hi")).reason == "not_found"


def test_prompt_says_dms_are_possible_and_forbids_invented_limits(settings):
    p = system_prompt(settings, scoped=False, chat=True, people=["มิว"])
    assert "send_dm" in p and "ส่งสรุปเข้า DM ผม" in p and "DM รายงานให้หัวหน้า" in p
    assert "ส่ง PDF ให้ผมทาง DM" in p and "ฝากบอกพี่มิวทาง DM" in p
    assert "ห้ามอ้างข้อจำกัดของ Discord" in p and "ยังทำไม่ได้ค่ะ" in p
    assert "บอตส่ง DM ไม่ได้" in p  # named as the example of what never to claim


def test_prompt_follow_up_and_emoji_rules(settings):
    p = system_prompt(settings, scoped=False)
    assert "ห้ามหยิบเรื่องเก่าที่ไม่เกี่ยวมาพูดหรือเสนอต่อยอด" in p
    assert "อย่างมาก 1 ประโยคสั้นๆ" in p and "ห้ามปิดท้ายด้วยการเชียร์" in p
    assert "อีโมจิ 0-1 ตัว" in p


def test_prompt_persona_and_exchange_rate_rules(settings):
    p = system_prompt(settings, scoped=False, language="th")
    assert "ไม่ใช่หุ่นยนต์" in p and "ระบบได้รับคำสั่งแล้ว" in p and "ในฐานะ AI" in p
    assert "ห้ามประมาณหรือเดาอัตราแลกเปลี่ยน" in p
    en = system_prompt(settings, scoped=False, language="en")
    assert "Talk like a real person" in en


def test_capabilities_are_examples_not_a_manual(settings):
    p = system_prompt(settings, scoped=False)
    assert "4-6 อย่าง" in p and "แค่บอกสลิดเป็นคำพูดก็ได้" in p


# ------------------------------------------------------------------ settings by talking
def test_chat_tools_register_only_bound_actions():
    names = lambda a: [f.__name__ for f in build_chat_tools(a, [])]  # noqa: E731
    assert names(ChatActions()) == []
    full = ChatActions(*([lambda *a: {}] * 7))
    assert names(full) == [
        "set_channel_mode", "set_channel_language", "set_memory_days", "show_settings",
        "hidden_channels", "cancel_deep_task", "send_dm",
    ]  # fmt: skip
    calls = []
    (tool,) = build_chat_tools(ChatActions(show_settings=lambda: {"ok": 1}), calls)
    assert tool() == {"ok": 1} and calls == [("show_settings", {})]


async def tools_for(bot, world, who, text="x"):
    author = member(world.g, who)
    m = TMsg(world.g, world.gen, author, text)
    return bot._chat_actions(
        m, world.g, author, bot.is_exec_member(author), "th", asyncio.get_running_loop()
    )


async def test_set_channel_mode_by_talking_follows_the_slash_rules(bot, world):
    a = await tools_for(bot, world, EXEC)
    r = await asyncio.to_thread(a.set_channel_mode, "off", None)
    assert r == {"status": "ok", "channel": "ทั่วไป", "mode": "off"}
    assert bot.team.channel_mode(GUILD, 10, "ทั่วไป", set()) == "off"
    r = await asyncio.to_thread(a.set_channel_mode, "listen", "ถาม-สลิด")  # another channel, by name
    assert r["channel"] == "ถาม-สลิด" and bot.team.channel_mode(GUILD, 12, "", set()) == "listen"
    assert (await asyncio.to_thread(a.set_channel_mode, "loud", None))["error"] == "unknown mode"
    assert (await asyncio.to_thread(a.set_channel_mode, "off", "nope"))[
        "error"
    ] == "channel_not_found"
    staff = await tools_for(bot, world, STAFF)
    r = await asyncio.to_thread(staff.set_channel_mode, "call", None)
    assert r["error"] == "not_allowed"
    assert bot.team.channel_mode(GUILD, 10, "ทั่วไป", set()) == "off"  # unchanged


async def test_language_memory_show_hidden_and_cancel_tools(bot, world):
    a = await tools_for(bot, world, EXEC)
    # the language is the company's (no per-channel language any more): set_server_language
    r = await asyncio.to_thread(a.set_server_language, "en", True)
    assert r["status"] == "ok" and r["language"] == "en"
    assert bot.team.channel_language(GUILD, 10) == bot.team.channel_language(GUILD, 12) == "en"
    assert (await asyncio.to_thread(a.set_server_language, "fr", True))[
        "error"
    ] == "unknown language"
    assert (await asyncio.to_thread(a.set_memory_days, 30)) == {"status": "ok", "days": 30}
    assert bot.team.memory_retention_days(GUILD) == 30
    assert (await asyncio.to_thread(a.set_memory_days, -1))["error"] == "bad_days"  # 0 = forever
    shown = await asyncio.to_thread(a.show_settings)
    assert (
        shown["mode"] == "call" and shown["reply_language"] == "en" and shown["memory_days"] == 30
    )
    world.named.permissions_for = lambda me: SimpleNamespace(
        view_channel=False, read_message_history=False
    )
    hidden = await asyncio.to_thread(a.hidden_channels)
    assert hidden["count"] >= 1 and hidden["channels"][0]["mention"].startswith("<#")
    assert "result" in await asyncio.to_thread(a.cancel_deep_task)
    staff = await tools_for(bot, world, STAFF)
    for call in (lambda: staff.set_memory_days(10), staff.hidden_channels):
        assert (await asyncio.to_thread(call))["error"] == "not_allowed"
    assert (await asyncio.to_thread(staff.show_settings))["mode"] == "call"  # anyone may look


async def test_chat_tools_are_handed_to_the_model(bot, world, monkeypatch):
    w = wire(bot, monkeypatch)
    m = TMsg(world.g, world.chat, member(world.g, EXEC), "ห้องนี้ไม่ต้องตอบนะ")
    await bot.on_message(m)
    ((_q, kw),) = w.agent.calls
    assert kw["chat_actions"].set_channel_mode is not None


# ------------------------------------------------------------------ acknowledgement
def msg_with(*atts, content=""):
    return SimpleNamespace(attachments=list(atts), content=content)


def test_ack_flavor_by_content():
    pdf, png = att("a.pdf", "application/pdf"), att("a.png", "image/png")
    mp3, mp4 = att("a.mp3", "audio/mpeg"), att("a.mp4", "video/mp4")
    assert ackmod.ack_flavor(msg_with(pdf), "") == "doc"
    assert ackmod.ack_flavor(msg_with(att("a.docx", "")), "") == "doc"
    assert ackmod.ack_flavor(msg_with(png), "") == "image"
    assert ackmod.ack_flavor(msg_with(mp3), "") == "audio"
    assert ackmod.ack_flavor(msg_with(mp4, png), "") == "audio"  # audio / video wins
    assert ackmod.ack_flavor(msg_with(), "ดูนี่ https://example.com/x") == "link"
    assert ackmod.ack_flavor(msg_with(), "เดือนนี้จ่ายไปเท่าไร") is None
    assert ackmod.ack_flavor(msg_with(), "x", deep=True) == "deep"
    assert ackmod.ack_text("doc", "th", lambda xs: xs[0]) == t("ack_doc_1", "th")
    assert ackmod.ack_text("doc", "th", lambda xs: xs[0], avoid=[t("ack_doc_1", "th")]) == t(
        "ack_doc_2", "th"
    )


def test_every_ack_has_both_languages_and_variants():
    assert ackmod.VARIANTS >= 8
    for flavor in ackmod.FLAVORS:
        for n in range(1, ackmod.VARIANTS + 1):
            for lang in ("th", "en"):
                assert t(f"ack_{flavor}_{n}", lang)


@pytest.mark.parametrize(
    "name, ctype, text, flavor",
    [
        ("a.pdf", "application/pdf", "", "doc"),
        ("a.jpg", "image/jpeg", "", "image"),
        ("clip.mp3", "audio/mpeg", "", "audio"),
        (None, None, "ช่วยดูหน่อย https://example.com/a", "link"),
    ],
)
async def test_ack_is_sent_at_once_then_edited_into_the_answer(
    bot, world, monkeypatch, name, ctype, text, flavor
):
    wire(bot, monkeypatch)
    atts = [att(name, ctype)] if name else []
    m = TMsg(world.g, world.chat, member(world.g, STAFF), text, attachments=atts)
    rec = recording(m)
    await bot.on_message(m)
    await settle(bot)
    assert rec.texts == [t(f"ack_{flavor}_1", "th")]  # exactly one message was sent ...
    assert m.replies == ["agent says"]  # ... and it became the answer
    assert bot._answers.get(Sent.id) is not None  # delete-by-reply finds the edited message


async def test_an_english_server_gets_an_english_ack(bot, world, monkeypatch):
    wire(bot, monkeypatch)
    bot.guilds_store.set_server_language(GUILD, "en")  # the company language
    bot.team._cache.clear()
    m = TMsg(
        world.g,
        world.chat,
        member(world.g, STAFF),
        "",
        attachments=[att("a.pdf", "application/pdf")],
    )
    rec = recording(m)
    await bot.on_message(m)
    assert rec.texts == [t("ack_doc_1", "en")] and "document" in rec.texts[0]


async def test_long_answer_edits_the_first_chunk_and_sends_the_rest(bot, world, monkeypatch):
    wire(bot, monkeypatch)
    long = "ก" * 1900 + "ข" * 100
    monkeypatch.setattr(
        botmod, "ask", lambda *a, **k: SimpleNamespace(text=long, seconds=0, rounds=0)
    )
    sent_more = []

    async def send(text, **kw):
        sent_more.append(text)
        return SimpleNamespace(id=9001)

    world.chat.send = send
    m = TMsg(
        world.g,
        world.chat,
        member(world.g, STAFF),
        "",
        attachments=[att("a.pdf", "application/pdf")],
    )
    await bot.on_message(m)
    assert m.replies == ["ก" * 1900] and sent_more == ["ข" * 100]
    assert bot._answers.get(9001) is bot._answers.get(Sent.id)  # both chunks are one answer


async def test_failed_edit_sends_normally_and_deletes_the_ack(bot, world, monkeypatch):
    wire(bot, monkeypatch)

    async def boom(self, content=None, **kw):
        raise discord.HTTPException(SimpleNamespace(status=500, reason="x"), "x")

    monkeypatch.setattr(Sent, "edit", boom)
    m = TMsg(
        world.g, world.chat, member(world.g, STAFF), "", attachments=[att("a.jpg", "image/jpeg")]
    )
    await bot.on_message(m)
    # reply 0 was the ack (deleted), reply 1 is the answer sent the normal way
    assert m.replies == [None, "agent says"]


async def test_receipt_reply_link_points_at_the_edited_message(bot, world, monkeypatch):
    w = wire(bot, monkeypatch)
    w.agent.receipts = [(41, "⚠️")]
    m = TMsg(
        world.g,
        world.chat,
        member(world.g, STAFF),
        "",
        attachments=[att("IMG_1.jpg", "image/jpeg")],
    )
    await bot.on_message(m)
    assert w.linked == [(41, Sent.id)] and m.replies == ["agent says"]


class Slow:
    """A model call that blocks (in its worker thread) until released."""

    def __init__(self):
        self.release, self.started = threading.Event(), threading.Event()

    def __call__(self, *a, **k):
        self.started.set()
        assert self.release.wait(5)
        return SimpleNamespace(text="เสร็จแล้ว", seconds=0, rounds=0)


async def test_slow_plain_question_is_acked_after_six_seconds(bot, world, monkeypatch):
    wire(bot, monkeypatch)
    slow = Slow()
    monkeypatch.setattr(botmod, "ask", slow)
    gate, slept = asyncio.Event(), []

    async def fake_sleep(seconds):  # the 6 second timer, on a fake clock
        slept.append(seconds)
        await gate.wait()

    bot._sleep = fake_sleep
    m = TMsg(world.g, world.chat, member(world.g, STAFF), "เดือนนี้จ่ายไปเท่าไร")
    rec = recording(m)
    task = asyncio.create_task(bot.on_message(m))
    await until(slow.started.is_set)
    assert slept == [ackmod.SLOW_AFTER_S] == [6.0] and rec.texts == []  # nothing yet
    gate.set()  # six seconds pass
    await until(lambda: rec.texts == [t("ack_slow_1", "th")])
    slow.release.set()
    await task
    assert m.replies == ["เสร็จแล้ว"]  # the ack became the answer


class WriterStub:
    """Stands in for FillerWriter: a fixed line after `delay` seconds, or an error."""

    def __init__(self, line=None, delay=0.0, boom=False):
        self.line, self.delay, self.boom, self.calls = line, delay, boom, []

    def enabled(self):
        return True

    async def write(self, **kw):
        self.calls.append(kw)
        await asyncio.sleep(self.delay)
        if self.boom:
            raise RuntimeError("model down")
        return self.line


async def doc_message(world):
    return TMsg(
        world.g,
        world.chat,
        member(world.g, STAFF),
        "ช่วยสรุปไฟล์นี้",
        attachments=[att("a.pdf", "application/pdf")],
    )


@pytest.mark.parametrize(
    ("writer", "expected"),
    [
        (WriterStub("ขออ่านไฟล์นี้ก่อนนะคะ"), "ขออ่านไฟล์นี้ก่อนนะคะ"),  # fast: the model's line
        (WriterStub("ช้าไปค่ะ", delay=1.0), None),  # slow: the fallback, the late line is dropped
        (WriterStub(boom=True), None),  # failing: the fallback
        (WriterStub(None), None),  # nothing usable: the fallback
    ],
)
async def test_instant_ack_is_one_message_whether_the_model_is_fast_slow_or_failing(
    bot, world, monkeypatch, writer, expected
):
    wire(bot, monkeypatch)
    bot.fillers = writer
    bot.s.ack_llm_timeout_s = 0.1
    m = await doc_message(world)
    rec = recording(m)
    await bot.on_message(m)
    assert len(rec.texts) == 1 and m.replies == ["agent says"]  # ONE ack, edited into the answer
    pool = [t(f"ack_doc_{n}", "th") for n in range(1, 9)]
    assert rec.texts[0] == expected if expected else rec.texts[0] in pool
    assert writer.calls[0]["kind"] == "doc" and writer.calls[0]["question"] == "ช่วยสรุปไฟล์นี้"


async def test_slow_ack_line_is_written_while_the_timer_runs(bot, world, monkeypatch):
    wire(bot, monkeypatch)
    slow = Slow()
    monkeypatch.setattr(botmod, "ask", slow)
    bot.fillers = WriterStub("ขอเช็กยอดเดือนนี้แป๊บนะคะ")
    gate = asyncio.Event()

    async def fake_sleep(seconds):
        await gate.wait()

    bot._sleep = fake_sleep
    m = TMsg(world.g, world.chat, member(world.g, STAFF), "เดือนนี้จ่ายไปเท่าไร")
    rec = recording(m)
    task = asyncio.create_task(bot.on_message(m))
    await until(slow.started.is_set)
    assert bot.fillers.calls and rec.texts == []  # generation started with the timer
    gate.set()
    await until(lambda: rec.texts == ["ขอเช็กยอดเดือนนี้แป๊บนะคะ"])
    slow.release.set()
    await task
    assert m.replies == ["เสร็จแล้ว"] and bot.recent_acks.get(world.chat.id)


async def test_consecutive_acks_in_a_channel_avoid_the_last_lines(bot, world, monkeypatch):
    wire(bot, monkeypatch)
    bot.fillers = WriterStub(None)
    bot._choose = lambda xs: xs[0]  # the first free line each time
    seen = []
    for _ in range(3):
        m = await doc_message(world)
        rec = recording(m)
        await bot.on_message(m)
        seen += rec.texts
    assert len(set(seen)) == 3


async def test_quick_answer_gets_no_ack(bot, world, monkeypatch):
    wire(bot, monkeypatch)
    gate, slept = asyncio.Event(), []

    async def fake_sleep(seconds):
        slept.append(seconds)
        await gate.wait()  # never released: the answer is back first

    bot._sleep = fake_sleep
    m = TMsg(world.g, world.chat, member(world.g, STAFF), "สวัสดี")
    rec = recording(m)
    await bot.on_message(m)
    assert rec.texts == ["agent says"] and m.replies == ["agent says"]
    assert slept in ([], [6.0])  # the timer may have started but never fired


async def test_interjections_are_never_acked(bot, world, monkeypatch):
    wire(bot, monkeypatch)
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    bot._sleep = fake_sleep
    monkeypatch.setattr(
        botmod, "ask", lambda *a, **k: SimpleNamespace(text="ลองดูนี่", seconds=0, rounds=0)
    )
    m = TMsg(world.g, world.gen, member(world.g, STAFF), "ดู https://example.com ไหม")
    rec = recording(m)
    await bot._answer(m, world.g, m.author, interjection=True)
    assert slept == [] and rec.texts == ["ลองดูนี่"]


async def test_a_crashing_answer_does_not_leave_the_ack_hanging(bot, world, monkeypatch):
    wire(bot, monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("model down")

    monkeypatch.setattr(botmod, "ask", boom)
    m = TMsg(
        world.g,
        world.chat,
        member(world.g, STAFF),
        "",
        attachments=[att("a.pdf", "application/pdf")],
    )
    await bot.on_message(m)  # on_message logs the failure
    assert m.replies == [t("qa_failed", "th")]


async def never_fires(seconds):  # the 6 second ack timer, on a clock that never gets there
    await asyncio.Event().wait()


async def test_a_question_that_fails_before_the_ack_still_gets_a_reply(bot, world, monkeypatch):
    """A quick failure (a 400 / 403, the model-call cap) inside the 6 s ack delay left no ack to
    edit, so the user got no reply at all."""
    wire(bot, monkeypatch)
    bot._sleep = never_fires

    def boom(*a, **k):
        raise RuntimeError("400 bad request")

    monkeypatch.setattr(botmod, "ask", boom)
    m = TMsg(world.g, world.chat, member(world.g, STAFF), "เดือนนี้จ่ายไปเท่าไร")
    rec = recording(m)
    await bot.on_message(m)  # on_message logs the failure
    assert rec.texts == [t("qa_failed", "th")] and m.replies == [t("qa_failed", "th")]


async def test_a_failed_interjection_stays_quiet(bot, world, monkeypatch):
    wire(bot, monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("model down")

    monkeypatch.setattr(botmod, "ask", boom)
    m = TMsg(world.g, world.gen, member(world.g, STAFF), "ดู https://example.com ไหม")
    rec = recording(m)
    with pytest.raises(RuntimeError):
        await bot._answer(m, world.g, m.author, interjection=True)
    assert rec.texts == []  # nobody asked Salid: no apology either


def test_answer_deadline_default_and_env(monkeypatch):
    from salid.config import Settings

    monkeypatch.delenv("SALID_ANSWER_TIMEOUT_SECONDS", raising=False)
    assert Settings(_env_file=None).answer_timeout_seconds == 120.0
    monkeypatch.setenv("SALID_ANSWER_TIMEOUT_SECONDS", "45")
    assert Settings(_env_file=None).answer_timeout_seconds == 45.0


class TypingProbe:
    """channel.typing(): records when the indicator starts and stops."""

    def __init__(self):
        self.log = []

    def __call__(self):
        return self

    async def __aenter__(self):
        self.log.append("on")

    async def __aexit__(self, *a):
        self.log.append("off")


@pytest.mark.parametrize(
    ("text", "atts"),
    [
        ("เดือนนี้จ่ายไปเท่าไร", []),  # the 6 s ack never went out: the apology is a reply
        ("", [("a.pdf", "application/pdf")]),  # instant ack: it becomes the apology
    ],
)
async def test_an_answer_past_the_deadline_gets_the_apology_and_stops_typing(
    bot, world, monkeypatch, caplog, text, atts
):
    """A Vertex call that hangs used to leave "one moment..." and typing up forever."""
    wire(bot, monkeypatch)
    bot._sleep = never_fires
    bot.s.answer_timeout_seconds = 0.05
    hung = Slow()
    monkeypatch.setattr(botmod, "ask", hung)
    world.chat.typing = typing = TypingProbe()
    m = TMsg(world.g, world.chat, member(world.g, STAFF), text, attachments=[att(*a) for a in atts])
    rec = recording(m)
    try:
        with caplog.at_level("WARNING", logger="salid.bot"):
            await bot.on_message(m)
    finally:
        hung.release.set()  # the abandoned worker thread ends
    assert m.replies == [t("qa_failed", "th")] and len(rec.texts) == 1
    assert typing.log == ["on", "off"]
    (warn,) = [r for r in caplog.records if r.name == "salid.bot" and r.levelname == "WARNING"]
    assert str(m.id) in warn.getMessage() and str(world.chat.id) in warn.getMessage()
    assert "เดือนนี้" not in warn.getMessage()  # ids only, never the question
    assert not [r for r in caplog.records if r.levelname == "ERROR"]  # handled, not a crash


async def test_an_interjection_past_the_deadline_stays_quiet(bot, world, monkeypatch):
    wire(bot, monkeypatch)
    bot.s.answer_timeout_seconds = 0.05
    hung = Slow()
    monkeypatch.setattr(botmod, "ask", hung)
    m = TMsg(world.g, world.gen, member(world.g, STAFF), "ดู https://example.com ไหม")
    rec = recording(m)
    try:
        assert await bot._answer(m, world.g, m.author, interjection=True) is False
    finally:
        hung.release.set()
    assert rec.texts == []


async def test_reply_summary_acks_by_the_referenced_message_and_registers_the_edit(
    bot, world, monkeypatch
):
    wire(bot, monkeypatch)
    target = TMsg(
        world.g,
        world.chat,
        member(
            world.g,
            EXEC,
        ),
        "",
        attachments=[att("deck.pdf", "application/pdf")],
    )

    async def summarize(self, tgt, guild, member, setting):
        return "สรุปแล้วครับ"

    monkeypatch.setattr(type(bot), "_summarize_message", summarize)
    m = TMsg(world.g, world.chat, member(world.g, STAFF), "สรุปหน่อย")
    m.reference = SimpleNamespace(message_id=target.id, resolved=target)
    rec = recording(m)
    await bot._answer(m, world.g, m.author)
    assert rec.texts == [t("ack_doc_1", "th")] and m.replies == ["สรุปแล้วครับ"]
    assert bot._answers.get(Sent.id) is not None


# ------------------------------------------------------------------ 👀 read receipts
async def test_eyes_after_a_quiet_image_description(bot, iw, world):
    set_llm(bot, FakeClient([note_resp()]))
    m = await send_images(bot, world, iw.a, "ดูนี่", [att()])
    await settle(bot)
    assert m.reactions == ["👀"]


async def test_no_eyes_when_the_description_fails(bot, iw, world):
    set_llm(bot, FakeClient([]))  # the model call raises
    m = await send_images(bot, world, iw.a, "ดูนี่", [att()])
    await settle(bot)
    assert m.reactions == []


async def test_no_eyes_when_the_flag_is_off_or_the_channel_is_off(bot, iw, world):
    from salid.guild_settings import GuildSettingsUpdate

    set_llm(bot, FakeClient([note_resp(), note_resp()]))
    bot.guilds_store.save_settings(
        GUILD, GuildSettingsUpdate(features={"read_receipts": False}), "x"
    )
    bot.team._cache.clear()
    m = await send_images(bot, world, iw.a, "a", [att()])
    await settle(bot)
    assert m.reactions == []
    bot.guilds_store.save_settings(
        GUILD, GuildSettingsUpdate(features={"read_receipts": True}), "x"
    )
    set_mode(bot, 12, "off")
    m = await send_images(bot, world, iw.b, "b", [att()])
    await settle(bot)
    assert m.reactions == []


async def test_eyes_also_in_listen_mode_and_for_media_and_documents(bot, iw, world, monkeypatch):
    set_mode(bot, 10, "listen")

    async def media_note(self, message, idx, a):
        return "[เสียง: x]"

    async def doc_note(self, message, idx, a, order=0, *, gid):
        return "[PDF: x]"

    monkeypatch.setattr(type(bot), "_media_note", media_note)
    monkeypatch.setattr(type(bot), "_doc_note", doc_note)
    monkeypatch.setattr(botmod, "ingest_attachment", lambda *a, **k: IngestResult("ignored"))
    m = await send_images(bot, world, iw.a, "เสียง", [att("v.mp3", "audio/mpeg")])
    await settle(bot)
    assert m.reactions == ["👀"]
    m = await send_images(bot, world, iw.a, "เอกสาร", [att("d.pdf", "application/pdf")])
    await settle(bot)
    assert m.reactions == ["👀"]
    monkeypatch.setattr(type(bot), "_doc_note", lambda *a, **k: _none())
    m = await send_images(bot, world, iw.a, "เอกสาร2", [att("e.pdf", "application/pdf")])
    await settle(bot)
    assert m.reactions == []  # nothing could be read: no sign of having read it


async def _none():
    return None


async def test_no_eyes_for_backlog_items(bot, iw, world):
    from test_memory_import import ago, hmsg, run_import

    set_llm(bot, FakeClient([note_resp()]))
    old = hmsg(world, iw.a, 1, ago(3), "old pic", attachments=[att("1.png")])
    await run_import(bot)
    assert old.reactions == []


async def test_no_eyes_on_a_message_addressed_to_salid(bot, world, monkeypatch):
    wire(bot, monkeypatch, FakeClient([note_resp()]))
    m = TMsg(
        world.g, world.chat, member(world.g, STAFF), "", attachments=[att("a.jpg", "image/jpeg")]
    )
    await bot.on_message(m)
    await settle(bot)
    assert "👀" not in m.reactions


# ------------------------------------------------------------------ receipts by content
async def test_counts_come_from_the_ledger_and_are_cached(bot, world):
    from salid.models import Receipt

    for i in range(3):
        bot.store.add_receipt(
            Receipt(guild_id=GUILD, submitter_id=1, submitter_name="A", sha256=f"s{i}", channel_id=10,
                    effective_date="2026-10-10", created_at="2026-10-10")
        )  # fmt: skip
    bot.store.add_receipt(
        Receipt(guild_id=GUILD, submitter_id=1, submitter_name="A", sha256="old", channel_id=12,
                effective_date="2026-01-01", created_at="2026-01-01")
    )  # fmt: skip
    assert await bot._counts(GUILD) == {10: 3}  # the old one is outside the 30 days
    bot.store.add_receipt(
        Receipt(guild_id=GUILD, submitter_id=1, submitter_name="A", sha256="n", channel_id=10,
                effective_date="2026-10-11", created_at="2026-10-11")
    )  # fmt: skip
    assert await bot._counts(GUILD) == {10: 3}  # cached for ~10 minutes
    bot._clock = lambda: 10**9
    assert await bot._counts(GUILD) == {10: 4}


async def test_pdf_goes_through_intake_by_content_and_images_through_the_describer(
    bot, iw, world, monkeypatch
):
    seen = []

    def ingest(*a, **k):
        seen.append(k["filename"])
        return IngestResult("ignored")

    monkeypatch.setattr(botmod, "ingest_attachment", ingest)
    del bot._intake  # the real one
    set_llm(bot, FakeClient([note_resp("ใบเสร็จ", "ยอด 100", "receipt/invoice")]))
    await send_images(bot, world, iw.a, "", [att("bill.pdf", "application/pdf")])
    await settle(bot)
    assert seen == ["bill.pdf"]  # a PDF in an ordinary channel is judged by its content
    await send_images(bot, world, iw.a, "", [att("pic.png")])
    await settle(bot)
    assert seen == ["bill.pdf", "pic.png"]  # describer says receipt-like -> intake
