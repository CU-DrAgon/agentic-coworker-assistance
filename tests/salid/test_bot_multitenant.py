"""Bot behaviour with no setup: intake rules, executives, per-guild toggles/digest, join/leave,
the /ตั้งค่า link. Discord objects are small fakes; nothing connects to Discord."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from conftest import ALICE, GUILD, OTHER_GUILD, FixedClockSettings

from salid import bot as botmod
from salid.bot import SecretaryBot
from salid.guild_settings import GuildSettingsUpdate, MemoryGuildStore

ME = SimpleNamespace(id=999, mention="<@999>")
OWNER, ADMIN, MANAGER, STAFF, BOSS = 1, 2, 3, 4, 5


class Perms(SimpleNamespace):
    def __init__(self, administrator=False, manage_guild=False):
        super().__init__(administrator=administrator, manage_guild=manage_guild)


class FakeChannel:
    def __init__(self, id, name="general"):
        self.id, self.name, self.sent = id, name, []

    def permissions_for(self, _):
        return SimpleNamespace(send_messages=True)

    async def send(self, text, **_):
        self.sent.append(text)

    def typing(self):
        return _Typing()


class _Typing:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakeGuild:
    def __init__(self, id, name="Shop", channels=(), system_channel=None):
        self.id, self.name, self.owner_id = id, name, OWNER
        self.icon = SimpleNamespace(url="https://cdn/icon.png")
        self.text_channels = list(channels)
        self.system_channel, self.me = system_channel, ME
        for c in self.text_channels:
            c.guild = self
        self.members = {}

    def get_channel(self, cid):
        return next((c for c in self.text_channels if c.id == cid), None)

    def get_member(self, uid):
        return self.members.get(uid)


def member(guild, uid, roles=(), **perms):
    m = SimpleNamespace(
        id=uid, guild=guild, roles=[SimpleNamespace(name=r) for r in roles],
        guild_permissions=Perms(**perms), display_name=f"u{uid}", bot=False,
    )  # fmt: skip
    guild.members[uid] = m
    return m


class Att(SimpleNamespace):
    async def read(self):
        return b"img"


def photo():
    return Att(filename="r.jpg", content_type="image/jpeg", size=10)


class Msg:
    def __init__(self, guild, channel, author, content="hi", attachments=(), mention=False):
        self.guild, self.channel, self.author = guild, channel, author
        self.content, self.attachments = content, list(attachments)
        self.mentions = [ME] if mention else []
        self.reference, self.replies, self.id = None, [], 1

    async def reply(self, text, **kw):
        self.replies.append(text)


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setattr(SecretaryBot, "user", property(lambda s: ME))
    ga = FakeGuild(GUILD, "A", [FakeChannel(10, "ทั่วไป"), FakeChannel(11, "ใบเสร็จ-ร้าน")])
    gb = FakeGuild(OTHER_GUILD, "B", [FakeChannel(20, "Receipts"), FakeChannel(21, "chat")])
    monkeypatch.setattr(SecretaryBot, "guilds", property(lambda s: [ga, gb]))
    return SimpleNamespace(a=ga, b=gb)


@pytest.fixture
def bot(world, seeded):
    b = SecretaryBot(
        FixedClockSettings(exec_user_ids={BOSS}), seeded, None, guilds=MemoryGuildStore()
    )
    b.calls = []

    async def intake(m):
        b.calls.append(("intake", m.guild.id))

    async def answer(m, guild, who):
        b.calls.append(("answer", guild.id))

    b._intake, b._answer = intake, answer
    return b


# ------------------------------------------------------------------ receipt intake without setup
async def test_receipt_named_channels_work_without_any_config(bot, world):
    u = member(world.a, STAFF)
    await bot.on_message(Msg(world.a, world.a.text_channels[1], u, attachments=[photo()]))
    await bot.on_message(Msg(world.b, world.b.text_channels[0], u, attachments=[photo()]))
    assert bot.calls == [("intake", GUILD), ("intake", OTHER_GUILD)]  # ใบเสร็จ / Receipts (any case)


async def test_other_channels_need_a_mention_with_a_file(bot, world):
    u = member(world.a, STAFF)
    general = world.a.text_channels[0]
    await bot.on_message(Msg(world.a, general, u, attachments=[photo()]))  # no mention: ignored
    await bot.on_message(Msg(world.a, general, u, "รูป", [photo()], mention=True))
    txt = Att(filename="n.txt", content_type="text/plain", size=1)
    await bot.on_message(Msg(world.a, general, u, "อ่านนี่", [txt], mention=True))
    assert bot.calls == [("intake", GUILD), ("answer", GUILD)]  # a .txt mention is a question


async def test_configured_channels_still_work_and_are_per_guild(bot, world):
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(receipt_channel_ids=["10"]), "x")
    u = member(world.a, STAFF)
    await bot.on_message(Msg(world.a, world.a.text_channels[0], u, attachments=[photo()]))
    ub = member(world.b, STAFF + 100)
    await bot.on_message(Msg(world.b, FakeChannel(10, "other"), ub, attachments=[photo()]))
    assert bot.calls == [("intake", GUILD)]  # channel id 10 means nothing in guild B


async def test_toggles_are_per_guild(bot, world):
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={"receipts": False}), "x")
    for g in (world.a, world.b):
        ch = g.text_channels[1] if g is world.a else g.text_channels[0]
        await bot.on_message(Msg(g, ch, member(g, STAFF), attachments=[photo()]))
    assert bot.calls == [("intake", OTHER_GUILD)]


async def test_ask_channels_and_mentions_answer_in_their_guild(bot, world):
    bot.guilds_store.save_settings(OTHER_GUILD, GuildSettingsUpdate(ask_channel_ids=["21"]), "x")
    await bot.on_message(Msg(world.b, world.b.text_channels[1], member(world.b, STAFF)))
    await bot.on_message(
        Msg(world.a, world.a.text_channels[0], member(world.a, STAFF), mention=True)
    )
    await bot.on_message(Msg(world.a, world.a.text_channels[0], member(world.a, STAFF)))  # chatter
    assert bot.calls == [("answer", OTHER_GUILD), ("answer", GUILD)]


async def test_mentioning_the_bot_role_also_answers(bot, world):
    m = Msg(world.a, world.a.text_channels[0], member(world.a, STAFF))
    m.role_mentions = [SimpleNamespace(id=5, tags=SimpleNamespace(bot_id=ME.id))]
    await bot.on_message(m)
    other = Msg(world.a, world.a.text_channels[0], member(world.a, STAFF))
    other.role_mentions = [SimpleNamespace(id=6, tags=None)]  # an ordinary role
    await bot.on_message(other)
    assert bot.calls == [("answer", GUILD)]


# ------------------------------------------------------------------ real handlers, scoped per guild
async def test_answer_asks_with_the_message_guild(bot, world, monkeypatch):
    seen = []
    monkeypatch.setattr(
        "salid.bot.ask",
        lambda s, store, q, **kw: (
            seen.append(kw) or SimpleNamespace(text="ok", seconds=0, rounds=0)
        ),
    )
    del bot._answer  # use the real one
    m = Msg(world.b, world.b.text_channels[1], member(world.b, STAFF), "ยอดเท่าไร", mention=True)
    await bot._answer(m, world.b, m.author)
    assert seen == [{"guild_id": OTHER_GUILD, "is_exec": False, "user_id": STAFF, "history": ""}]
    assert m.replies == ["ok"]


async def test_deep_task_runs_on_the_asking_guild(bot, world, monkeypatch):
    seen = []
    monkeypatch.setattr(
        "salid.bot.run_deep_task",
        lambda s, store, task, **kw: seen.append(kw) or (_ for _ in ()).throw(RuntimeError("x")),
    )
    bot.s = bot.s.model_copy(update={"gemini_cli_path": "/opt/gemini"})
    sent = []
    msg = Msg(world.b, FakeChannel(21), member(world.b, OWNER), "งานใหญ่: ทำรายงาน")
    msg.channel.send = lambda t, **k: _append(sent, t)
    del bot._answer
    await bot._answer(msg, world.b, msg.author)  # the owner is an executive
    await asyncio.gather(*bot._deep_tasks)
    assert seen == [{"guild_id": OTHER_GUILD}]


async def _append(lst, v):
    lst.append(v)


async def test_cannot_correct_another_guilds_receipt(bot, world, seeded):
    from conftest import mk

    r = mk(seeded, 50, ALICE, "Alice", "Zeta", "2026-10-01", 5.0, guild=OTHER_GUILD)
    seeded.set_reply_message(r.id, 777)
    m = Msg(world.a, world.a.text_channels[0], member(world.a, OWNER), "ยอดจริง 1")
    m.reference = SimpleNamespace(message_id=777)
    assert await bot._maybe_correction(m) is False
    assert seeded.get(r.id).total == 5.0


# ------------------------------------------------------------------ executives
def test_executive_rules(bot, world):
    a = world.a
    people = {
        "owner": member(a, OWNER),
        "admin": member(a, ADMIN, administrator=True),
        "manager": member(a, MANAGER, manage_guild=True),
        "role": member(a, 6, roles=["Executive"]),
        "thai role": member(a, 7, roles=["ผู้บริหาร"]),
        "env id": member(a, BOSS),
        "staff": member(a, STAFF, roles=["พนักงาน"]),
    }
    assert {k for k, m in people.items() if bot.is_exec_member(m)} == set(people) - {"staff"}


def test_exec_roles_follow_the_guilds_own_setting(bot, world):
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(exec_role_names=["หัวหน้า"]), "x")
    assert bot.is_exec_member(member(world.a, 8, roles=["หัวหน้า"]))
    assert not bot.is_exec_member(member(world.a, 9, roles=["Executive"]))  # replaced list
    assert bot.is_exec_member(member(world.b, 9, roles=["Executive"]))  # guild B keeps defaults


# ------------------------------------------------------------------ DMs
async def test_dm_answers_only_when_exactly_one_shared_guild(bot, world):
    del bot._answer
    answered = []

    async def answer(m, guild, who):
        answered.append(guild.id)

    bot._answer = answer
    member(world.a, STAFF)
    dm = Msg(None, FakeChannel(1), SimpleNamespace(id=STAFF, bot=False), "ยอด")
    await bot.on_message(dm)
    assert answered == [GUILD]
    member(world.b, STAFF)  # now in both guilds: ambiguous
    dm2 = Msg(None, FakeChannel(1), SimpleNamespace(id=STAFF, bot=False), "ยอด")
    await bot.on_message(dm2)
    assert answered == [GUILD] and "แท็ก <@999>" in dm2.replies[0]


# ------------------------------------------------------------------ lifecycle
async def test_join_registers_guild_and_welcomes(bot, world):
    sys_ch = FakeChannel(30, "general")
    world.a.system_channel = sys_ch
    await bot.on_guild_join(world.a)
    g = bot.guilds_store.get(GUILD)
    assert g.guild_name == "A" and g.icon_url.endswith("icon.png") and g.active and g.installed_at
    assert [c.name for c in g.channels] == ["ทั่วไป", "ใบเสร็จ-ร้าน"]
    welcome = sys_ch.sent[0]
    assert "สลิด" in welcome and "ใบเสร็จ" in welcome and "receipt" in welcome
    assert "<@999>" in welcome and "/ตั้งค่า" in welcome


async def test_welcome_falls_back_to_first_sendable_channel(bot, world):
    await bot.on_guild_join(world.b)  # no system channel
    assert world.b.text_channels[0].sent and not world.b.text_channels[1].sent


async def test_remove_marks_inactive_but_keeps_data(bot, world):
    await bot.on_guild_join(world.a)
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(digest_time="07:00"), "x")
    await bot.on_guild_remove(world.a)
    g = bot.guilds_store.get(GUILD)
    assert g.active is False and g.digest_time == "07:00" and g.guild_name == "A"


async def test_on_ready_syncs_every_guild(bot, world):
    await bot.on_ready()
    assert bot.guilds_store.get(GUILD).guild_name == "A"
    assert bot.guilds_store.get(OTHER_GUILD).channels[0].name == "Receipts"


async def test_channel_events_are_throttled_and_end_with_latest_state(bot, world, monkeypatch):
    monkeypatch.setattr(botmod, "SYNC_MIN_INTERVAL_S", 0.05)
    writes = []
    real = bot.guilds_store.upsert_guild
    bot.guilds_store.upsert_guild = lambda gid, **kw: (
        writes.append(kw["channels"]) or real(gid, **kw)
    )
    await bot.on_ready()
    writes.clear()
    for n in range(5):  # a burst of channel changes
        world.a.text_channels.append(FakeChannel(100 + n, f"c{n}"))
        world.a.text_channels[-1].guild = world.a
        await bot.on_guild_channel_create(world.a.text_channels[-1])
    await asyncio.sleep(0.2)
    assert len(writes) == 1 and len(writes[0]) == 7  # one trailing write with everything


# ------------------------------------------------------------------ /ตั้งค่า
class Interaction:
    def __init__(self, guild, user):
        self.guild, self.user, self.sent = guild, user, []
        self.response = SimpleNamespace(send_message=self._send)

    async def _send(self, text, ephemeral=False):
        self.sent.append((text, ephemeral))


async def test_setup_command_gives_an_ephemeral_one_time_link(bot, world):
    bot.s = bot.s.model_copy(update={"hub_url": "https://hub.example/"})
    it = Interaction(world.a, member(world.a, MANAGER, manage_guild=True))
    await bot.handle_setup(it)
    ((text, ephemeral),) = it.sent
    assert ephemeral is True and "https://hub.example/hub?link=" in text
    code = text.split("link=")[1].split()[0]
    doc = bot.guilds_store.client.collection("salid_link_codes").docs[code]
    assert doc["guild_id"] == GUILD and doc["discord_user_id"] == MANAGER and not doc["used"]


async def test_setup_command_refused_for_regular_members(bot, world):
    it = Interaction(world.a, member(world.a, STAFF, roles=["Executive"]))
    await bot.handle_setup(it)
    assert it.sent == [(botmod.SETUP_DENIED_MESSAGE, True)]
    assert bot.guilds_store.client.collection("salid_link_codes").docs == {}
    dm = Interaction(None, SimpleNamespace(id=1))
    await bot.handle_setup(dm)
    assert dm.sent[0][1] is True


async def test_setup_command_allows_the_owner_and_admins(bot, world):
    for who in (member(world.a, OWNER), member(world.a, ADMIN, administrator=True)):
        it = Interaction(world.a, who)
        await bot.handle_setup(it)
        assert "link=" in it.sent[0][0]


def test_slash_commands_are_registered_with_thai_name_and_english_alias(bot):
    bot._register_commands()
    cmds = {c.name: c for c in bot.tree.get_commands()}
    assert set(cmds) == {"ตั้งค่า", "setup", "เข้าประชุม", "join", "ออกประชุม", "leave"}
    assert all(c.guild_only for c in cmds.values())


# ------------------------------------------------------------------ digest
def at(hh, mm=0):
    return datetime(2026, 10, 14, hh, mm, tzinfo=ZoneInfo("Asia/Bangkok"))


def test_digest_is_per_guild_channel_and_time(bot, world):
    bot.guilds_store.save_settings(
        GUILD, GuildSettingsUpdate(digest_channel_id="10", digest_time="18:00"), "x"
    )
    bot.guilds_store.save_settings(
        OTHER_GUILD, GuildSettingsUpdate(digest_channel_id="21", digest_time="07:30"), "x"
    )
    assert [(g.id, c.id) for g, c in bot.due_digests(at(18))] == [(GUILD, 10)]
    assert [(g.id, c.id) for g, c in bot.due_digests(at(7, 30))] == [(OTHER_GUILD, 21)]
    assert bot.due_digests(at(12)) == []


def test_digest_skips_guilds_without_a_channel_or_with_the_feature_off(bot, world):
    bot.guilds_store.save_settings(
        GUILD, GuildSettingsUpdate(digest_time="18:00"), "x"
    )  # no channel
    bot.guilds_store.save_settings(
        OTHER_GUILD,
        GuildSettingsUpdate(digest_channel_id="21", features={"digest": False}),
        "x",
    )
    assert bot.due_digests(at(18)) == []


def test_env_digest_channel_only_applies_in_the_guild_that_owns_it(bot, world):
    bot.s = bot.s.model_copy(update={"digest_channel_id": 21, "digest_time": "09:00"})
    assert [(g.id, c.id) for g, c in bot.due_digests(at(9))] == [(OTHER_GUILD, 21)]


async def test_digest_sends_that_guilds_numbers_once_per_day(bot, world, seeded):
    from conftest import mk

    mk(seeded, 60, ALICE, "Alice", "Zeta Corp", "2026-10-14", 7777.0, guild=OTHER_GUILD)
    mk(seeded, 61, ALICE, "Alice", "Makro", "2026-10-14", 100.0, guild=GUILD)
    for g, ch in ((GUILD, "10"), (OTHER_GUILD, "21")):
        bot.guilds_store.save_settings(g, GuildSettingsUpdate(digest_channel_id=ch), "x")
    await bot._send_digests(at(18))
    await bot._send_digests(at(18))  # same day: not sent again
    a, b = world.a.get_channel(10), world.b.get_channel(21)
    assert len(a.sent) == 1 and len(b.sent) == 1
    assert "Makro" in a.sent[0] and "Zeta" not in a.sent[0]
    assert "Zeta Corp" in b.sent[0] and "Makro" not in b.sent[0]


async def test_bare_mention_uses_previous_message_or_greets(bot, world, monkeypatch):
    from datetime import UTC, datetime, timedelta

    from salid.bot import GREETING_MESSAGE

    seen = []
    monkeypatch.setattr(
        "salid.bot.ask",
        lambda s, store, q, **kw: seen.append(q) or SimpleNamespace(text="ok", seconds=0, rounds=0),
    )
    del bot._answer  # use the real one
    now = datetime(2026, 10, 15, 10, 0, tzinfo=UTC)
    author = member(world.a, STAFF)

    class Chan(FakeChannel):
        def __init__(self, prev):
            super().__init__(10, "ทั่วไป")
            self.prev = prev

        async def history(self, limit, before):
            for p in self.prev:
                yield p

    def msg(chan, content, at):
        m = Msg(world.a, chan, author, content)
        m.created_at = at
        return m

    hi = msg(None, "สวัสดีครับ", now - timedelta(seconds=5))
    chan = Chan([hi])
    bare = msg(chan, "<@999>", now)
    await bot._answer(bare, world.a, author)
    assert seen == ["สวัสดีครับ"]

    stale = Chan([msg(None, "เก่ามาก", now - timedelta(minutes=10))])
    lonely = msg(stale, "<@999>", now)
    await bot._answer(lonely, world.a, author)
    assert lonely.replies == [GREETING_MESSAGE]


async def test_follow_up_without_mention_within_the_conversation_window(bot, world, monkeypatch):
    import salid.bot as botmod

    ch = world.a.text_channels[0]
    author = member(world.a, STAFF)
    await bot.on_message(Msg(world.a, ch, author, "ยอดเดือนนี้"))  # never talked: ignored
    assert bot.calls == []
    bot._note_talking(Msg(world.a, ch, author))
    await bot.on_message(Msg(world.a, ch, author, "แล้วเดือนก่อนล่ะ"))
    other = Msg(world.a, ch, author, "ไปกินข้าวกัน")
    other.mentions = [SimpleNamespace(id=5)]  # talking to a person, not Salid
    await bot.on_message(other)
    assert bot.calls == [("answer", GUILD)]
    monkeypatch.setattr(botmod, "CONVERSATION_SECONDS", 0)
    await bot.on_message(Msg(world.a, ch, author, "ยังอยู่ไหม"))
    assert bot.calls == [("answer", GUILD)]


async def test_reply_to_a_salid_message_is_answered(bot, world):
    m = Msg(world.a, world.a.text_channels[0], member(world.a, STAFF), "อธิบายเพิ่มหน่อย")
    m.reference = SimpleNamespace(message_id=1, resolved=SimpleNamespace(author=ME))
    await bot.on_message(m)
    assert bot.calls == [("answer", GUILD)]


async def test_channels_named_after_salid_answer_everything(bot, world):
    ch = FakeChannel(12, "ถาม-สลิด")
    await bot.on_message(Msg(world.a, ch, member(world.a, STAFF), "ทำอะไรได้บ้าง"))
    assert bot.calls == [("answer", GUILD)]


async def test_recent_chat_keeps_only_this_person_and_salid(bot, world):
    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 10, 15, 10, 0, tzinfo=UTC)
    me = member(world.a, STAFF)
    me.display_name = "พี่"
    other = member(world.a, BOSS)

    def line(author, text, ago):
        m = SimpleNamespace(author=author, content=text, created_at=now - timedelta(seconds=ago))
        m.role_mentions = []
        return m

    class Chan(FakeChannel):
        async def history(self, limit, before):
            for p in [
                line(ME, "สวัสดีครับ", 10),
                line(other, "เรื่องอื่น", 20),
                line(me, "ทักครับ", 30),
                line(me, "เก่าเกินไป", 3600),
            ]:
                yield p

    msg = Msg(world.a, Chan(10), me, "คุยต่อ")
    msg.created_at = now
    assert await bot._recent_chat(msg) == "พี่: ทักครับ\nสลิด: สวัสดีครับ"


async def test_join_creates_the_chat_channel_and_welcomes_there(bot, world):
    from salid.bot import CHAT_CHANNEL_NAME

    g = world.a
    g.me = SimpleNamespace(guild_permissions=SimpleNamespace(manage_channels=True))

    async def create_text_channel(name, **kw):
        ch = FakeChannel(40, name)
        ch.mention = f"<#{ch.id}>"
        g.text_channels.append(ch)
        return ch

    g.create_text_channel = create_text_channel
    await bot.on_guild_join(g)
    chat = g.text_channels[-1]
    assert chat.name == CHAT_CHANNEL_NAME and "<#40>" in chat.sent[0]
    await bot.on_guild_join(g)  # rejoin: reuses the existing channel
    assert [c.name for c in g.text_channels].count(CHAT_CHANNEL_NAME) == 1


async def test_join_without_manage_channels_still_welcomes(bot, world):
    await bot.on_guild_join(world.b)
    assert "คุยกับน้องสลิด" in world.b.text_channels[0].sent[0]
