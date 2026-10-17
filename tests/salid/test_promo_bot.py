# ruff: noqa: F811, F401
"""Promo kits in chat: when the tool is offered, what it is bound to, the flag and the hub copy."""

from conftest import GUILD
from promo_helpers import fake_client, image_response, png_bytes
from test_coordinator import world  # fixture: world.g, .pub, .secret, .staff, .boss
from test_modes import TMsg, bot  # fixture

from salid import promo, qa
from salid.guild_settings import FEATURES, Features, GuildSettingsUpdate
from salid.qa import FileInput


def img_input(i=0, mime="image/png"):
    return FileInput(i, "p.png", mime, data=png_bytes((900, 900)))


def names(tools):
    return {f.__name__ for f in tools}


async def test_offered_only_with_a_usable_photo(bot, world):
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "ทำโพสต์")
    assert names(bot._promo_tools(msg, world.g, m, [img_input()])) == {"make_promo_kit"}
    assert bot._promo_tools(msg, world.g, m, []) == []
    assert bot._promo_tools(msg, world.g, m, None) == []  # voice path
    assert bot._promo_tools(msg, world.g, m, [img_input(mime="image/gif")]) == []
    skipped = img_input()
    skipped.skip = "over 15 MB"
    assert bot._promo_tools(msg, world.g, m, [skipped]) == []


async def test_remembered_photo_offers_the_tool_to_the_same_person_in_the_same_channel(bot, world):
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "ขอโทนอุ่นกว่านี้")
    bot.promo_memory.put(
        (GUILD, world.pub.id, m.id),
        promo.prepare_photo(png_bytes((900, 900))),
        promo.merge_spec(None, headline="h", scene="cafe"),
    )
    assert names(bot._promo_tools(msg, world.g, m, [])) == {"make_promo_kit"}
    other = world.boss
    assert bot._promo_tools(TMsg(world.g, world.pub, other, "x"), world.g, other, []) == []
    assert bot._promo_tools(TMsg(world.g, world.secret, m, "x"), world.g, m, []) == []


async def test_flag_off_or_no_raqm_means_no_tool(bot, world, monkeypatch):
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "x")
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={"promo": False}), "x")
    bot.team._cache.clear()
    assert bot._promo_tools(msg, world.g, m, [img_input()]) == []
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={"promo": True}), "x")
    bot.team._cache.clear()
    monkeypatch.setattr("salid.promo_bot.cards.thai_ready", lambda: False)
    assert bot._promo_tools(msg, world.g, m, [img_input()]) == []


async def test_agent_kwargs_add_the_tool_for_chat_not_interjections(bot, world):
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "x")
    kw = dict(setting="th", lang="th", question="x", memory=None, profile=None)
    assert "make_promo_kit" in names(
        bot._agent_kwargs(msg, world.g, m, False, files=[img_input()], **kw)["extra_tools"]
    )
    assert "make_promo_kit" not in names(
        bot._agent_kwargs(msg, world.g, m, False, **kw).get("extra_tools", [])
    )
    inter = bot._agent_kwargs(msg, world.g, m, False, files=[img_input()], interjection=True, **kw)
    assert "make_promo_kit" not in names(inter.get("extra_tools", []))


async def test_kit_is_bound_to_guild_channel_author_and_budget(bot, world):
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "x")
    bot.llm = fake_client(image_response(png_bytes((768, 1376))))
    [tool] = bot._promo_tools(msg, world.g, m, [img_input()])
    out = tool(headline="ลาเต้", scene="cafe")
    assert out["status"] == "ok" and len(tool.outbox) == 3
    assert qa.tool_files([tool]) == tool.outbox  # what Answer.attachments carries
    assert bot.promo_memory.get((GUILD, world.pub.id, m.id)) is not None
    assert bot.promo_budget.used(GUILD, bot.s.today().isoformat()) == 1
    assert bot.llm.models.calls[0][0] == "gemini-nano-banana-2.1"


async def test_the_daily_kit_cap_survives_a_restart(bot, world):
    m, day = world.staff, bot.s.today().isoformat()
    bot.docs.put("salid_media_usage", GUILD, day, {"promo_kits": bot.s.promo_kits_per_day})
    bot._init_promo()  # a restart: fresh process state, the same DocStore
    bot.llm = fake_client(RuntimeError("must not be called"))
    [tool] = bot._promo_tools(TMsg(world.g, world.pub, m, "x"), world.g, m, [img_input()])
    out = tool(headline="ลาเต้", scene="cafe")
    assert (out["status"], out["ai_scene"], out["reason"]) == ("ok", False, "daily_limit")
    assert bot.llm.models.calls == [] and out["kits_left_today"] == 0


async def test_a_remembered_kit_can_still_be_revised_by_voice(bot, world):
    """Voice passes files=[] on purpose (origin/main): the spoken flag does not hide the kit."""
    m = world.staff
    bot.promo_memory.put(
        (GUILD, world.pub.id, m.id),
        promo.prepare_photo(png_bytes((900, 900))),
        promo.merge_spec(None, headline="h", scene="cafe"),
    )
    room = bot.room_message(world.g, m, world.pub, room={m.id})
    kw = dict(setting="th", lang="th", question="x", memory=None, profile=None)
    voice = bot._agent_kwargs(room, world.g, m, False, files=[], spoken=True, **kw)
    assert "make_promo_kit" in names(voice["extra_tools"])
    assert bot._promo_tools(room, world.g, m, None) == []  # no file list at all: still none


def test_promo_flag_defaults_on_and_capability_line():
    assert "promo" in FEATURES and Features().promo is True
    assert "โพสต์โปรโมทจากรูปสินค้า" in qa.CAPABILITIES_PROMPT
