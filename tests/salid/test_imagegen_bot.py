# ruff: noqa: F811, F401
"""create_image in chat: when the tool is offered, what it is bound to, the flag and limits."""

from conftest import GUILD
from promo_helpers import fake_client, image_response, png_bytes
from test_coordinator import world  # fixture: world.g, .pub, .secret, .staff, .boss
from test_modes import TMsg, bot  # fixture

from salid import qa
from salid.guild_settings import FEATURES, Features, GuildSettingsUpdate
from salid.qa import FileInput


def img_input(i=0, mime="image/png"):
    return FileInput(i, "p.png", mime, data=png_bytes((900, 900)))


def names(tools):
    return {f.__name__ for f in tools}


async def test_offered_for_chat_with_or_without_a_photo_not_for_voice(bot, world):
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "วาดรูปแมว")
    assert names(bot._image_tools(msg, world.g, m, [])) == {"create_image"}
    assert names(bot._image_tools(msg, world.g, m, [img_input()])) == {"create_image"}
    assert bot._image_tools(msg, world.g, m, None) == []  # voice path


async def test_flag_off_or_no_raqm_means_no_tool(bot, world, monkeypatch):
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "x")
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={"images": False}), "x")
    bot.team._cache.clear()
    assert bot._image_tools(msg, world.g, m, []) == []
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={"images": True}), "x")
    bot.team._cache.clear()
    monkeypatch.setattr("salid.imagegen_bot.cards.thai_ready", lambda: False)
    assert bot._image_tools(msg, world.g, m, []) == []


async def test_agent_kwargs_add_the_tool_for_chat_not_interjections(bot, world):
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "x")
    kw = dict(setting="th", lang="th", question="x", memory=None, profile=None)
    assert "create_image" in names(
        bot._agent_kwargs(msg, world.g, m, False, files=[], **kw)["extra_tools"]
    )
    assert "create_image" not in names(
        bot._agent_kwargs(msg, world.g, m, False, **kw).get("extra_tools", [])
    )
    inter = bot._agent_kwargs(msg, world.g, m, False, files=[], interjection=True, **kw)
    assert "create_image" not in names(inter.get("extra_tools", []))


async def test_tool_is_bound_to_guild_channel_author_and_both_budgets(bot, world):
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "x")
    img = image_response(png_bytes((1024, 1024)))
    bot.llm = fake_client(img, img)
    [tool] = bot._image_tools(msg, world.g, m, [img_input(1)])
    out = tool(prompt="a latte on a wooden table", count=2, file_index=1)
    assert out["status"] == "ok" and len(tool.outbox) == 2
    assert qa.tool_files([tool]) == tool.outbox  # what Answer.attachments carries
    day = bot.s.today().isoformat()
    assert bot.image_budget.used(GUILD, day) == 2 and bot.image_user_budget.used(m.id, day) == 2
    assert bot.image_budget.cap == 20 and bot.image_user_budget.cap == 5
    assert len(bot.image_memory.get((GUILD, world.pub.id, m.id))) == 2
    assert bot.llm.models.calls[0][0] == "gemini-nano-banana-2.1"
    assert bot.llm.models.calls[0][1][0].inline_data is not None  # the attached photo


async def test_unusable_attachments_are_not_offered_as_references(bot, world):
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "x")
    skipped = img_input(0)
    skipped.skip = "over 15 MB"
    bot.llm = fake_client()
    [tool] = bot._image_tools(msg, world.g, m, [skipped, img_input(1, mime="image/gif")])
    for i in (0, 1):
        assert tool(prompt="a cup", file_index=i)["status"] == "error"
    assert bot.llm.models.calls == []


async def test_never_raises(bot, world, monkeypatch):
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "x")
    monkeypatch.setattr(
        "salid.imagegen_bot.image_tools", lambda ctx: (_ for _ in ()).throw(RuntimeError("x"))
    )
    assert bot._image_tools(msg, world.g, m, []) == []


def test_images_flag_defaults_on():
    assert "images" in FEATURES and Features().images is True
