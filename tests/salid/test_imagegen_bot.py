# ruff: noqa: F811, F401
"""create_image in chat: when the tool is offered, what it is bound to, the flag and limits."""

from conftest import GUILD
from promo_helpers import fake_client, image_response, png_bytes
from test_coordinator import world  # fixture: world.g, .pub, .secret, .staff, .boss
from test_modes import TMsg, bot  # fixture

from salid import imagegen, imagegen_bot, qa
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


async def test_a_voice_room_gets_no_tool_even_with_an_empty_file_list(bot, world):
    """Newer voice code passes files=[] and a room stand-in (no `reply`): still no tool."""
    m = world.staff
    kw = dict(setting="th", lang="th", question="x", memory=None, profile=None)
    for room in (
        bot.room_message(world.g, m, world.pub, room={m.id}),  # an answer spoken to the room
        bot.room_message(world.g, m, world.pub),  # a private answer asked by voice
    ):
        assert bot._image_tools(room, world.g, m, []) == []
        tools = bot._agent_kwargs(room, world.g, m, False, files=[], **kw).get("extra_tools", [])
        assert "create_image" not in names(tools)


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
    user_budget = bot.image_user_budget.for_guild(GUILD)
    assert bot.image_budget.used(GUILD, day) == 2 and user_budget.used(m.id, day) == 2
    assert bot.image_budget.cap == 20 and bot.image_user_budget.cap == user_budget.cap == 5
    # persisted per guild and day: a restart keeps both counts
    assert bot.docs.get("salid_media_usage", GUILD, day) == {"images": 2, f"images_user_{m.id}": 2}
    assert len(bot.image_memory.get((GUILD, world.pub.id, m.id))) == 2
    assert bot.llm.models.calls[0][0] == "gemini-nano-banana-2.1"
    assert bot.llm.models.calls[0][1][0].inline_data is not None  # the attached photo


async def test_daily_caps_survive_a_restart(bot, world):
    m, day = world.staff, bot.s.today().isoformat()
    bot.docs.put("salid_media_usage", GUILD, day, {"images": 19})  # made before the restart
    bot._init_images()  # a restart: fresh process state, the same DocStore
    img = image_response(png_bytes((64, 64)))
    bot.llm = fake_client(img, img)
    [tool] = bot._image_tools(TMsg(world.g, world.pub, m, "x"), world.g, m, [])
    out = tool(prompt="a cup", count=2)
    assert out["status"] == "ok" and out["count_made"] == 1 and out["images_left_today"] == 0
    assert "Only 1 of 2" in out["note"] and len(bot.llm.models.calls) == 1
    assert bot.docs.get("salid_media_usage", GUILD, day) == {"images": 20, f"images_user_{m.id}": 1}
    assert tool(prompt="a cup")["status"] == "limit" and len(bot.llm.models.calls) == 1
    other = world.boss  # the per-person cap is stored too
    bot.docs.put("salid_media_usage", GUILD, day, {f"images_user_{other.id}": 5})
    bot._init_images()
    [tool] = bot._image_tools(TMsg(world.g, world.pub, other, "x"), world.g, other, [])
    assert tool(prompt="a cup")["status"] == "limit" and len(bot.llm.models.calls) == 1


def count_usage_io(bot):
    """Counts DocStore reads and writes of the media usage documents."""
    io = {"get": 0, "put": 0}
    get, put = bot.docs.get, bot.docs.put

    def counted_get(kind, *a):
        io["get"] += kind == "salid_media_usage"
        return get(kind, *a)

    def counted_put(kind, *a):
        io["put"] += kind == "salid_media_usage"
        return put(kind, *a)

    bot.docs.get, bot.docs.put = counted_get, counted_put
    return io


async def test_one_read_before_the_take_and_one_read_modify_write_for_it(bot, world):
    m, day = world.staff, bot.s.today().isoformat()
    img = image_response(png_bytes((64, 64)))
    bot.llm = fake_client(img, img)
    [tool] = bot._image_tools(TMsg(world.g, world.pub, m, "x"), world.g, m, [])
    io = count_usage_io(bot)
    out = tool(prompt="a cup", count=2)
    assert out["status"] == "ok" and out["images_left_today"] == 3
    assert io == {"get": 2, "put": 1}  # the check, then one read-modify-write of both counts
    assert bot.docs.get("salid_media_usage", GUILD, day) == {"images": 2, f"images_user_{m.id}": 2}


async def test_units_without_an_image_go_back_to_both_stored_counts(bot, world):
    m, day = world.staff, bot.s.today().isoformat()
    bot.llm = fake_client(image_response(png_bytes((64, 64))), RuntimeError("x"))
    [tool] = bot._image_tools(TMsg(world.g, world.pub, m, "x"), world.g, m, [])
    out = tool(prompt="a cup", count=2)
    assert out["count_made"] == 1 and out["images_left_today"] == 4
    assert bot.docs.get("salid_media_usage", GUILD, day) == {"images": 1, f"images_user_{m.id}": 1}
    bot.llm = None  # no client: nothing can be made, nothing stays taken
    [tool] = bot._image_tools(TMsg(world.g, world.pub, m, "x"), world.g, m, [])
    assert tool(prompt="a cup")["status"] == "error"
    assert bot.docs.get("salid_media_usage", GUILD, day) == {"images": 1, f"images_user_{m.id}": 1}


def test_the_stored_budget_has_no_unused_available():
    assert not hasattr(imagegen_bot.StoredDailyBudget, "available")


async def test_explicit_spoken_flag_keeps_the_tool_out_of_a_voice_answer(bot, world):
    """Voice callers say so (`spoken=True`), even with a message that looks like chat."""
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "วาดรูปแมว")
    kw = dict(setting="th", lang="th", question="x", memory=None, profile=None)
    assert bot._image_tools(msg, world.g, m, [], spoken=True) == []
    voice = bot._agent_kwargs(msg, world.g, m, False, files=[], spoken=True, **kw)
    assert "create_image" not in names(voice.get("extra_tools", []))
    chat = bot._agent_kwargs(msg, world.g, m, False, files=[], spoken=False, **kw)
    assert "create_image" in names(chat["extra_tools"])


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


async def test_upload_budget_follows_the_servers_limit(bot, world, monkeypatch):
    m = world.staff
    msg = TMsg(world.g, world.pub, m, "x")
    seen = []
    monkeypatch.setattr("salid.imagegen_bot.image_tools", lambda ctx: seen.append(ctx) or [])
    bot._image_tools(msg, world.g, m, [])  # no filesize_limit: discord.py's 10 MiB default
    world.g.filesize_limit = 10 * 2**20  # unboosted / tier 1
    bot._image_tools(msg, world.g, m, [])
    world.g.filesize_limit = 100 * 2**20  # tier 3
    bot._image_tools(msg, world.g, m, [])
    assert [c.max_bytes for c in seen] == [9 * 2**20, 9 * 2**20, imagegen.OUTBOX_MAX_BYTES]


async def test_a_dm_uses_the_dm_upload_limit_not_the_servers(bot, world, monkeypatch):
    m = world.staff
    world.g.filesize_limit = 100 * 2**20  # the server the DM is about is boosted
    seen = []
    monkeypatch.setattr("salid.imagegen_bot.image_tools", lambda ctx: seen.append(ctx) or [])
    bot._image_tools(TMsg(None, world.pub, m, "x"), world.g, m, [])  # a DM: message.guild None
    assert [c.max_bytes for c in seen] == [9 * 2**20]


def test_the_capability_list_names_images_and_video_ads():
    caps = qa.CAPABILITIES_PROMPT
    assert "\n20) สร้างรูปจากคำบรรยาย:" in caps and "\n21) วิดีโอโฆษณาสั้น:" in caps
    assert caps.index("19) สไลด์นำเสนอ") < caps.index("20) ") < caps.index("21) ")


def test_images_flag_defaults_on():
    assert "images" in FEATURES and Features().images is True
