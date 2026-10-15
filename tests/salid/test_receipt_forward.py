"""Receipts posted in the wrong channel are copied to the receipt home channel."""

from types import SimpleNamespace

import pytest
from conftest import GUILD, FixedClockSettings

from salid import bot as botmod
from salid.bot import SecretaryBot, is_receipt_channel_name
from salid.extract import IngestResult
from salid.guild_settings import GuildSettings, GuildSettingsUpdate, MemoryGuildStore
from salid.receipt_forward import MAX_FILES, forward_receipt, home_channel

ME = SimpleNamespace(id=999, mention="<@999>")
STAFF = 4


class Chan:
    def __init__(self, id, name, position=0, can_send=True):
        self.id, self.name, self.position, self.can_send = id, name, position, can_send
        self.sent = []

    def permissions_for(self, _):
        return SimpleNamespace(view_channel=True, send_messages=self.can_send, attach_files=True)

    async def send(self, text, files=(), **kw):
        msg = SimpleNamespace(id=5000 + len(self.sent), channel=self, text=text, files=files)
        self.sent.append(msg)
        return msg

    def typing(self):
        return _Typing()


class _Typing:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class Guild:
    def __init__(self, channels):
        self.id, self.me, self.text_channels = GUILD, ME, list(channels)
        self.members = {}

    def get_channel(self, cid):
        return next((c for c in self.text_channels if c.id == cid), None)

    def get_member(self, uid):
        return self.members.get(uid)


class Msg:
    def __init__(self, guild, channel, n_files=1):
        self.guild, self.channel, self.id, self.content = guild, channel, 77, "lunch"
        self.author = SimpleNamespace(
            id=STAFF,
            display_name="Som",
            mention="<@4>",
            guild_permissions=SimpleNamespace(administrator=False, manage_guild=False),
        )
        self.attachments = [
            SimpleNamespace(filename=f"r{i}.jpg", content_type="image/jpeg", size=3)
            for i in range(n_files)
        ]
        self.author.guild, self.author.roles = SimpleNamespace(owner_id=1, id=GUILD), []
        self.jump_url = "https://discord.com/channels/1/2/77"
        self.reactions, self.replies = [], []

    async def reply(self, text, **kw):
        self.replies.append(text)
        return SimpleNamespace(id=6000 + len(self.replies))

    async def add_reaction(self, e):
        self.reactions.append(e)


def settings(**kw):
    return GuildSettings(**kw)


def names(ch):
    return ch.name if ch else None


# ------------------------------------------------------------------ home channel pick
def test_home_prefers_configured_then_receipt_channels_then_named():
    a, b, c = Chan(1, "general"), Chan(2, "misc", 5), Chan(3, "ใบเสร็จ", 2)
    g = Guild([a, b, c])
    pick = lambda s, env=(): home_channel(g, s, env, is_receipt_channel_name)  # noqa: E731
    assert pick(settings(receipt_home_channel_id="1")) is a
    assert pick(settings(receipt_channel_ids=["2", "1"])) is b  # first listed
    assert pick(settings(), {1}) is a  # env ids count too
    assert pick(settings()) is c  # by name: "misc" has no hint, "ใบเสร็จ" does
    c.can_send = False
    assert pick(settings(receipt_channel_ids=["3", "1"])) is a  # skips one it cannot send in
    assert pick(settings()) is None
    assert pick(settings(receipt_home_channel_id="3")) is None  # explicit but unusable
    assert pick(settings(receipt_home_channel_id="99")) is None  # gone


def test_named_pick_goes_by_position():
    a, b = Chan(1, "expense", 7), Chan(2, "receipt", 3)
    assert home_channel(Guild([a, b]), settings(), set(), is_receipt_channel_name) is b


# ------------------------------------------------------------------ bot wiring
@pytest.fixture
def world(monkeypatch, seeded):
    monkeypatch.setattr(SecretaryBot, "user", property(lambda s: ME))
    gen, home = Chan(10, "ทั่วไป"), Chan(11, "ใบเสร็จ")
    g = Guild([gen, home])
    monkeypatch.setattr(SecretaryBot, "guilds", property(lambda s: [g]))
    bot = SecretaryBot(FixedClockSettings(), seeded, None, guilds=MemoryGuildStore())
    receipt = seeded.get(1)
    state = SimpleNamespace(status="created", receipt=receipt)

    def ingest(*a, **k):
        r = IngestResult(state.status, "saved #1", "✅", state.receipt)
        return r

    monkeypatch.setattr(botmod, "ingest_attachment", ingest)

    async def read():
        return b"bytes"

    return SimpleNamespace(bot=bot, g=g, gen=gen, home=home, store=seeded, state=state, read=read)


def conf(w, **kw):
    w.bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(**kw), "x")
    w.bot.team._cache.clear()


def msg(w, channel, n=1):
    m = Msg(w.g, channel, n)
    for a in m.attachments:
        a.read = w.read
    return m


async def test_forward_from_a_wrong_channel(world):
    w = world
    m = msg(w, w.gen)
    await w.bot._intake(m)
    (fwd,) = w.home.sent
    assert "<@4>" in fwd.text and "#ทั่วไป" in fwd.text and m.jump_url in fwd.text
    assert "saved #1" in fwd.text
    assert [f.filename for f in fwd.files] == ["r0.jpg"]
    assert w.store.get_by_reply_message(fwd.id).id == w.state.receipt.id  # reply = the copy
    assert len(m.replies) == 1 and "ส่งต่อไปที่ #ใบเสร็จ ให้แล้วครับ" in m.replies[0]
    assert m.replies[0].startswith("saved #1")  # normal reply kept
    assert m.reactions == ["✅"]


async def test_duplicate_is_copied_but_keeps_its_reply_link(world):
    w = world
    w.store.set_reply_message(w.state.receipt.id, 4242)
    w.state.status = "duplicate"
    await w.bot._intake(msg(w, w.gen))
    assert len(w.home.sent) == 1
    assert w.store.get_by_reply_message(4242).id == w.state.receipt.id


async def test_ignored_is_not_forwarded(world):
    world.state.status = "ignored"
    await world.bot._intake(msg(world, world.gen))
    assert world.home.sent == []


async def test_no_forward_from_a_receipt_channel_or_when_off_or_no_home(world):
    w = world
    await w.bot._intake(msg(w, w.home))  # named like a receipt channel
    assert w.home.sent == []
    conf(w, receipt_channel_ids=["10"], receipt_home_channel_id="11")
    await w.bot._intake(msg(w, w.gen))  # configured receipt channel
    assert w.home.sent == []
    conf(w, receipt_home_channel_id="11", receipt_forward=False)
    await w.bot._intake(msg(w, w.gen))
    assert w.home.sent == []
    w.home.name = "random"  # no receipt channel any more: nowhere to go
    conf(w)
    await w.bot._intake(msg(w, w.gen))
    assert w.home.sent == []


async def test_listen_mode_reacts_only(world):
    w = world
    m = msg(w, w.gen)
    await w.bot._intake(m, quiet=True)
    assert m.replies == [] and m.reactions == ["✅"] and len(w.home.sent) == 1
    assert w.store.get_by_reply_message(w.home.sent[0].id) is not None


async def test_missing_send_permission_falls_back_quietly(world):
    w = world
    w.home.can_send = False
    m = msg(w, w.gen)
    await w.bot._intake(m)
    assert w.home.sent == [] and m.replies == ["saved #1"]  # normal reply, no ack
    assert m.reactions == ["✅"]


async def test_http_error_while_sending_is_survived(world):
    import discord

    w = world

    async def boom(*a, **k):
        raise discord.HTTPException(SimpleNamespace(status=500, reason="x"), "x")

    w.home.send = boom
    m = msg(w, w.gen)
    await w.bot._intake(m)
    assert m.replies == ["saved #1"]


async def test_files_are_capped(world):
    files = [(f"f{i}.jpg", b"x") for i in range(6)] + [("big.jpg", b"x" * (8 * 1024 * 1024 + 1))]
    w = world
    sent = await forward_receipt(
        guild=w.g, settings=settings(), env_receipt_ids=set(), name_matches=is_receipt_channel_name,
        source=w.gen, author=SimpleNamespace(mention="@u"), jump_url="u", files=files,
        text="t", lang="en",
    )  # fmt: skip
    assert [f.filename for f in sent.files] == [f"f{i}.jpg" for i in range(MAX_FILES)]
    sent = await forward_receipt(
        guild=w.g, settings=settings(), env_receipt_ids=set(), name_matches=is_receipt_channel_name,
        source=w.gen, author=SimpleNamespace(mention="@u"), jump_url="u", files=files[-1:],
        text="t", lang="en",
    )  # fmt: skip
    assert sent.files == []  # too big: header and summary only


async def test_dm_forwards_when_the_agent_recorded_a_receipt(world, monkeypatch):
    w = world
    dm = SimpleNamespace(id=1, name=None)
    m = msg(w, dm)
    m.guild = None  # a DM
    res = IngestResult("created", "ok", "✅", w.state.receipt)
    fwd = await w.bot._forward(m, w.g, [("r.jpg", b"b")], res, "th")
    assert fwd is not None and "DM" in fwd.text and "ข้อความต้นฉบับ" not in fwd.text
    assert w.home.sent == [fwd]


async def test_agent_path_forwards_and_acks(world, monkeypatch):
    w = world
    from salid.qa import FileInput

    seen = {}

    def fake_ask(s, store, q, *, files=(), actions=None, **kw):
        f = FileInput(0, "a.jpg", "image/jpeg", b"bytes")
        res = actions.record_receipt(f, "")
        seen["res"] = res
        return SimpleNamespace(text="done", seconds=0, rounds=1, receipts=[(res.receipt.id, "✅")])

    monkeypatch.setattr(botmod, "ask", fake_ask)
    m = msg(w, w.gen)
    m.mentions, m.reference, m.role_mentions = [ME], None, []
    await w.bot._answer(m, w.g, m.author)
    assert len(w.home.sent) == 1 and w.home.sent[0].files[0].filename == "a.jpg"
    assert w.store.get_by_reply_message(w.home.sent[0].id) is not None
    assert any("ส่งต่อไปที่ #ใบเสร็จ" in r for r in m.replies)
