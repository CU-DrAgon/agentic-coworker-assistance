# ruff: noqa: F401, F811
"""The server-wide language switch: parser, store (clear + restore), command and tool (role
gating), private channel rename, and the English persona."""

import asyncio
import re

import pytest
from conftest import GUILD
from test_bot_multitenant import member
from test_human import tools_for
from test_modes import EXEC, STAFF, bot, manager, send, set_lang, world
from test_private_chat import Cat, human, prep

from salid.i18n import parse_server_language_command, t
from salid.qa import PERSONA, volatile_prompt

THAI = re.compile(r"[฀-๿]")


@pytest.mark.parametrize(
    "text, expected",
    [("ใช้ภาษาอังกฤษทั้งเซิร์ฟเวอร์", "en"), ("เปลี่ยนสลิดเป็นภาษาอังกฤษทั้งหมด", "en"),
     ("switch the whole server to English", "en"), ("speak English everywhere", "en"),
     ("Please use English server-wide", "en"), ("ใช้ภาษาไทยทั้งเซิร์ฟเวอร์", "th"),
     ("switch the whole server back to Thai", "th"), ("speak Thai everywhere", "th"),
     ("change the whole server from Thai to English", "en"),
     ("speak English", None), ("พูดภาษาอังกฤษ", None),
     ("what is English for everywhere?", None), ("how much did the whole server spend", None),
     ("", None)],
)  # fmt: skip
def test_parse_server_language_command(text, expected):
    assert parse_server_language_command(text) == expected


def test_store_switch_clears_overrides_and_switching_back_restores(bot):
    store = bot.guilds_store
    store.set_server_language(GUILD, "th", False)  # the language the overrides are saved under
    store.set_channel_language(GUILD, 10, "auto")
    store.set_channel_language(GUILD, 11, "th")
    bot.team.set_server_language(GUILD, "en", True)
    g = store.get(GUILD)
    assert g.default_language == "en" and g.channel_languages == {}
    assert g.previous_channel_languages == {"10": "auto", "11": "th"}
    bot.team.set_server_language(GUILD, "en", True)  # again: the saved copy survives
    assert store.get(GUILD).previous_channel_languages == {"10": "auto", "11": "th"}
    bot.team.set_server_language(GUILD, "th", True)
    g = store.get(GUILD)
    assert g.default_language == "th" and g.channel_languages == {"10": "auto", "11": "th"}
    assert g.previous_channel_languages == {}
    bot.team.set_server_language(GUILD, "en", False)  # default only
    g = store.get(GUILD)
    assert g.default_language == "en" and g.channel_languages == {"10": "auto", "11": "th"}
    with pytest.raises(ValueError):
        bot.team.set_server_language(GUILD, "auto")


async def test_command_switches_the_whole_server_and_replies_in_the_new_language(bot, world):
    set_lang(bot, 10, "th")
    m = await send(
        bot, world, world.gen, "<@999> switch the whole server to English",
        who=manager(world), mention=True,
    )  # fmt: skip
    assert m.replies == [t("server_lang_set", "en", label="English")]
    assert not THAI.search(m.replies[0]) and bot.calls == []
    g = bot.guilds_store.get(GUILD)
    assert g.default_language == "en" and g.channel_languages == {}
    assert bot.team.channel_language(GUILD, 10) == "en"
    back = await send(
        bot, world, world.gen, "<@999> ใช้ภาษาไทยทั้งเซิร์ฟเวอร์", who=manager(world), mention=True
    )
    assert back.replies == [t("server_lang_set", "th", label=t("lang_label_th", "th"))]
    assert bot.team.channel_language(GUILD, 10) == "th"  # restored


async def test_command_is_refused_for_regular_members(bot, world):
    m = await send(
        bot, world, world.gen, "<@999> speak English everywhere", who=STAFF, mention=True
    )
    assert m.replies == [t("lang_denied", "th")]
    assert bot.guilds_store.get(GUILD).default_language == "th"


async def test_tool_is_gated_and_validates(bot, world):
    staff = await tools_for(bot, world, STAFF)
    r = await asyncio.to_thread(staff.set_server_language, "en", True)
    assert r["error"] == "not_allowed" and bot.team.current(GUILD).default_language == "th"
    a = await tools_for(bot, world, EXEC)
    assert (await asyncio.to_thread(a.set_server_language, "fr", True))["error"]
    r = await asyncio.to_thread(a.set_server_language, "en", True)
    assert r["status"] == "ok" and r["language"] == "en"
    assert bot.team.current(GUILD).default_language == "en"
    r = await asyncio.to_thread(a.set_server_language, "th", False)
    assert r["status"] == "ok" and bot.team.current(GUILD).default_language == "th"


@pytest.fixture
def g(world, bot):
    return prep(world.g)


class RenCat(Cat):
    id = 4242

    async def edit(self, **kw):
        self.name = kw["name"]


async def test_private_channels_are_renamed_on_switch(bot, g):
    bot._pc_pace = 0
    m = human(g, STAFF)
    ch, created, _ = await bot.ensure_private_channel(g, m)
    assert ch.name == "สลิด-somchai"
    cat = g.categories[0]
    cat.__class__ = RenCat
    ch.category = cat

    async def edit(**kw):
        ch.name = kw["name"]
        ch.topic = kw.get("topic")

    ch.edit = edit
    await bot._apply_server_language(g, "en", True)
    await asyncio.gather(*list(bot._bg_tasks))
    assert cat.name == "Salid private chats" and ch.name == "salid-somchai"
    assert not THAI.search(ch.topic)
    await bot._apply_server_language(g, "th", True)
    await asyncio.gather(*list(bot._bg_tasks))
    assert cat.name == "คุยกับสลิด (ส่วนตัว)" and ch.name == "สลิด-somchai"


async def test_new_private_channels_in_an_english_server_use_english_names(bot, g):
    bot.team.set_server_language(GUILD, "en", True)
    m = human(g, STAFF)
    ch, _, _ = await bot.ensure_private_channel(g, m)
    assert ch.name == "salid-somchai" and g.categories[0].name == "Salid private chats"
    assert not THAI.search(ch.kw["topic"]) and not THAI.search(ch.sent[0][0])


def test_english_persona_has_no_thai_and_translates_content(bot):
    persona = PERSONA["en"]
    # the one Thai mention is the deliberate "switch to Thai ... ค่ะ/คะ" clause
    assert (
        not THAI.search(persona.replace("ค่ะ/คะ", ""))
        and "Salid" in persona
        and "translate" in persona
    )
    tail = volatile_prompt(bot.s, bot.s.now(), scoped=False, interjection=False, language="en")
    assert "English" in tail and "translate" in tail
