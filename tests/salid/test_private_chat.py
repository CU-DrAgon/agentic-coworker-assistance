# ruff: noqa: F401, F811
"""The private Salid channel per member (salid.private_chat / private_chat_bot)."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import discord
import pytest
from test_bot_multitenant import GUILD, STAFF, FakeChannel, bot, member, world  # noqa: F401

from salid import dm as dm_module
from salid.guild_settings import GuildSettingsUpdate
from salid.i18n import t
from salid.private_chat import (
    GUILD_CHANNEL_CAP,
    PrivateChatRepo,
    channel_name,
    overwrites,
    pick_category,
    wants_delete,
    wants_private_channel,
)


class H(SimpleNamespace):
    __hash__ = object.__hash__  # usable as an overwrite key, like discord objects


class Cat:
    def __init__(self, name, n=0):
        self.name, self.channels = name, [object()] * n


class Made(FakeChannel):
    def __init__(self, id, name, guild, **kw):
        super().__init__(id, name)
        self.guild, self.kw, self.deleted = guild, kw, False

    async def delete(self, **_):
        self.deleted = True
        self.guild.text_channels.remove(self)

    async def send(self, text, **kw):
        self.sent.append((text, kw))


def prep(guild, manage=True):
    guild.default_role = H(name="@everyone")
    guild.me = H(
        id=999, mention="<@999>", guild_permissions=SimpleNamespace(manage_channels=manage)
    )
    guild.categories, guild.created, guild.channels = [], [], []
    guild._next = 700

    async def create_category(name, **_):
        c = Cat(name)
        guild.categories.append(c)
        return c

    async def create_text_channel(name, **kw):
        guild._next += 1
        ch = Made(guild._next, name, guild, **kw)
        guild.text_channels.append(ch)
        guild.channels.append(ch)
        guild.created.append(ch)
        return ch

    guild.create_category, guild.create_text_channel = create_category, create_text_channel
    return guild


def human(guild, uid, name="Somchai", bot_=False):
    m = H(**vars(member(guild, uid)))
    guild.members[uid] = m
    m.display_name, m.bot, m.mention = name, bot_, f"<@{uid}>"
    return m


@pytest.fixture
def g(world, bot):  # noqa: F811
    return prep(world.a)


def test_pure_helpers():
    assert channel_name("Somchai Jai") == "สลิด-somchai-jai"
    assert channel_name("สมชาย!!") == "สลิด-สมชาย"
    assert wants_private_channel("ขอห้องคุยส่วนตัวกับสลิด")
    assert wants_private_channel("make me a private channel")
    assert not wants_private_channel("ลบห้องส่วนตัว") and wants_delete("ลบห้องส่วนตัว")
    assert wants_delete("delete my private channel")
    assert not wants_private_channel("สรุปยอดให้หน่อย")


def test_overwrites_hide_it_from_everyone_but_the_member_and_salid(g):
    m = human(g, STAFF)
    ow = overwrites(g, m)
    assert ow[g.default_role].view_channel is False
    mine = ow[m]
    assert (
        mine.view_channel and mine.send_messages and mine.read_message_history and mine.attach_files
    )
    assert ow[g.me].manage_channels and ow[g.me].view_channel and ow[g.me].send_messages
    assert set(ow) == {g.default_role, m, g.me}  # nobody else is named


def test_category_picking_overflows_to_numbered_ones():
    full = Cat("คุยกับสลิด (ส่วนตัว)", 50)
    assert pick_category([], "X") == (None, "X")
    assert pick_category([Cat("X", 3)], "X")[0].name == "X"
    assert pick_category([Cat("X", 50)], "X") == (None, "X 2")
    assert pick_category([Cat("X", 50), Cat("X 2", 50)], "X") == (None, "X 3")
    assert full.name


async def test_one_channel_per_member_with_chat_mode_and_honest_welcome(bot, g):
    m = human(g, STAFF)
    ch, created, err = await bot.ensure_private_channel(g, m)
    assert created and not err and ch.name == "สลิด-somchai"
    assert g.categories[0].name == "คุยกับสลิด (ส่วนตัว)" and ch.kw["category"] is g.categories[0]
    assert "Administrator" in ch.sent[0][0] and "<@4>" in ch.sent[0][0]
    assert bot.team.channel_mode(GUILD, ch.id, ch.name, set()) == "chat"
    again, created2, _ = await bot.ensure_private_channel(g, m)
    assert again is ch and not created2 and len(g.created) == 1


async def test_recreated_when_deleted_by_hand(bot, g):
    m = human(g, STAFF)
    ch, *_ = await bot.ensure_private_channel(g, m)
    g.text_channels.remove(ch)  # an admin deleted it
    ch2, created, _ = await bot.ensure_private_channel(g, m)
    assert created and ch2 is not ch


async def test_missing_manage_channels_says_exactly_what(bot, g):
    prep(g, manage=False)
    ch, created, err = await bot.ensure_private_channel(g, human(g, STAFF))
    assert ch is None and "Manage Channels" in err and g.created == []


async def test_join_creates_and_welcomes_bots_get_none_flag_off_nothing(bot, g):
    await bot.on_member_join(human(g, STAFF))
    assert len(g.created) == 1 and g.created[0].sent
    await bot.on_member_join(human(g, 50, "Robo", bot_=True))
    assert len(g.created) == 1
    bot.guilds_store.save_settings(
        GUILD, GuildSettingsUpdate(features={"private_channels": False}), "x"
    )
    bot.team._cache.clear()
    await bot.on_member_join(human(g, 51, "Late"))
    assert len(g.created) == 1  # off: nothing new (existing ones stay)
    ch, _, err = await bot.ensure_private_channel(g, human(g, 52, "Ask"))
    assert ch is None and err == t("pc_off", "th")


async def test_backfill_is_idempotent_paced_and_skips_bots(bot, g):
    for uid in (10, 11, 12):
        human(g, uid, f"P{uid}")
    human(g, 13, "Robo", bot_=True)
    pauses = []

    async def sleep(s):
        pauses.append(s)

    bot._pc_sleep = sleep
    await bot._pc_backfill(g)
    assert len(g.created) == 3 and pauses == [2.0, 2.0, 2.0]
    await bot._pc_backfill(g)
    assert len(g.created) == 3 and len(pauses) == 3  # nothing new, no pacing


async def test_leave_schedules_deletion_in_24h_and_it_is_persisted(bot, g):
    m = human(g, STAFF)
    ch, *_ = await bot.ensure_private_channel(g, m)
    await bot.on_member_remove(m)
    repo = PrivateChatRepo(bot.docs)  # a fresh object over the same store: survives a restart
    soon = datetime.now(UTC) + timedelta(hours=23)
    later = datetime.now(UTC) + timedelta(hours=25)
    assert repo.due_deletions(GUILD, soon) == [] and repo.due_deletions(GUILD, later) == [STAFF]
    del g.members[STAFF]
    await bot._pc_sweep()
    assert not ch.deleted  # not yet due
    bot.private_chats.schedule_delete(GUILD, STAFF, datetime.now(UTC) - timedelta(minutes=1))
    await bot._pc_sweep()
    assert ch.deleted and bot.private_chats.get(GUILD, STAFF) is None


async def test_rejoin_within_24h_cancels_and_reuses_the_channel(bot, g):
    m = human(g, STAFF)
    ch, *_ = await bot.ensure_private_channel(g, m)
    await bot.on_member_remove(m)
    await bot.on_member_join(m)
    assert (
        len(g.created) == 1
        and bot.private_chats.due_deletions(GUILD, datetime.now(UTC) + timedelta(days=9)) == []
    )
    assert (await bot.private_channel_for(g, STAFF)) is ch


async def test_category_overflow_uses_a_second_category(bot, g):
    g.categories.append(Cat("คุยกับสลิด (ส่วนตัว)", 50))
    ch, *_ = await bot.ensure_private_channel(g, human(g, STAFF))
    assert [c.name for c in g.categories][-1] == "คุยกับสลิด (ส่วนตัว) 2" and ch.kw[
        "category"
    ].name.endswith("2")


async def test_channel_cap_falls_back_to_dm_and_tells_admins_once(bot, g):
    g.channels = [object()] * GUILD_CHANNEL_CAP
    told = []

    async def report(guild, create=True):
        return SimpleNamespace(send=lambda text: _send(told, text))

    async def _send(lst, text):
        lst.append(text)

    bot._report_channel = report
    m = human(g, STAFF)
    for _ in range(2):
        ch, created, err = await bot.ensure_private_channel(g, m)
        assert ch is None and err == t("pc_cap", "th")
    assert g.created == [] and len(told) == 1


def dm_member(g, uid=STAFF):
    m = human(g, uid)
    m.dms = []

    async def send(text, **kw):
        m.dms.append((text, kw))
        return SimpleNamespace(id=len(m.dms) + 9000)

    m.send = send
    return m


async def test_proactive_messages_go_to_the_private_channel_with_a_mention(bot, g):
    m = dm_member(g)
    ch, *_ = await bot.ensure_private_channel(g, m)
    ch.sent.clear()
    res = await bot.deliver_to_person(g, m, "สรุปเช้า", view="V")
    assert res.ok and m.dms == []
    text, kw = ch.sent[0]
    assert text == "<@4> สรุปเช้า" and kw["view"] == "V" and kw["allowed_mentions"].users == [m]
    # the module-level send_dm (briefings, reminders, forms, bulk DMs) takes the same road
    await dm_module.send_dm(m, "เตือนงาน")
    assert ch.sent[1][0] == "<@4> เตือนงาน" and m.dms == []


async def test_without_a_channel_or_by_preference_or_on_failure_it_is_a_dm(bot, g):
    m = dm_member(g)
    await bot.deliver_to_person(g, m, "a")  # no channel yet
    assert [d[0] for d in m.dms] == ["a"]
    ch, *_ = await bot.ensure_private_channel(g, m)
    bot.profiles.repo = None if False else bot.profiles
    from salid.profiles import ProfileTools

    ProfileTools(bot.profiles, GUILD, STAFF).remember("delivery", "dm")
    await bot.deliver_to_person(g, m, "b")
    assert [d[0] for d in m.dms] == ["a", "b"]  # "ส่งทาง DM แทน"
    ProfileTools(bot.profiles, GUILD, STAFF).remember("delivery", "channel")
    ch.sent.clear()
    await bot.deliver_to_person(g, m, "c")
    assert ch.sent and [d[0] for d in m.dms] == ["a", "b"]

    async def boom(text, **kw):
        raise discord.HTTPException(SimpleNamespace(status=500, reason="x"), "x")

    ch.send = boom
    await bot.deliver_to_person(g, m, "d")
    assert [d[0] for d in m.dms] == ["a", "b", "d"]  # could not post: falls back to the DM


async def test_private_channel_talk_is_a_private_note_not_a_channel_record(bot, g):
    m = dm_member(g)
    ch, *_ = await bot.ensure_private_channel(g, m)
    from test_bot_multitenant import Msg

    msg = Msg(g, ch, m, "พรุ่งนี้ผมลา")
    msg.created_at = datetime.now(UTC)
    msg.id = 424242
    await bot.on_message(msg)
    await bot.drain_memory()
    from salid.memory_access import dm_scope

    notes = bot.memory_store.messages(GUILD, {dm_scope(STAFF)}, "2000-01-01", "2100-01-01")
    assert [(n.kind, n.text) for n in notes] == [("dm", "พรุ่งนี้ผมลา")]
    assert bot.memory_store.messages(GUILD, {ch.id}, "2000-01-01", "2100-01-01") == []
    assert bot.calls == [("answer", GUILD)]  # answered without a tag
    other = human(g, 77, "Other")
    await bot.on_message(Msg(g, ch, other, "hello"))
    assert len(bot.calls) == 1  # somebody else's words in it are ignored


async def test_audience_of_the_channel_is_its_viewers_including_admins(bot, g):
    m = human(g, STAFF)
    admin = human(g, 2, "Admin")
    plain = human(g, 3, "Plain")
    ch, *_ = await bot.ensure_private_channel(g, m)

    def perms(who):
        ok = who.id in (STAFF, 2)  # the owner, and Administrator who sees every channel
        return SimpleNamespace(read_messages=ok, read_message_history=ok)

    ch.permissions_for = perms
    assert bot._channel_viewers(g, ch) == frozenset({STAFF, admin.id})
    assert plain.id not in bot._channel_viewers(g, ch)


async def test_delete_command_asks_then_deletes(bot, g):
    from test_bot_multitenant import Msg

    m = dm_member(g)
    ch, *_ = await bot.ensure_private_channel(g, m)
    a = Msg(g, ch, m, "ลบห้องส่วนตัว")
    await bot.on_message(a)
    assert a.replies == [t("pc_delete_ask", "th")] and not ch.deleted
    b = Msg(g, ch, m, "ยืนยัน")
    await bot.on_message(b)
    assert ch.deleted and bot.private_chats.get(GUILD, STAFF) is None


async def test_dm_picker_links_the_private_channel(bot, g, world):
    from test_dm_guild import DM

    m = dm_member(g)
    member(world.b, STAFF)
    ch, *_ = await bot.ensure_private_channel(g, m)
    msg = DM("ยอด")
    await bot.on_message(msg)
    assert f"https://discord.com/channels/{GUILD}/{ch.id}" in msg.replies[0][0]


def test_salid_overwrite_only_grants_permissions_salid_is_invited_with():
    """Discord refuses (403) a channel whose overwrites grant the bot a permission it lacks."""
    import discord as _d

    from salid import private_chat
    from salid.permissions import SALID_PERMISSIONS

    class _G:
        default_role = object()
        me = object()

    ow = private_chat.overwrites(_G(), object())[_G.me]
    allowed, _ = ow.pair()
    assert allowed.value & ~_d.Permissions(SALID_PERMISSIONS).value == 0
