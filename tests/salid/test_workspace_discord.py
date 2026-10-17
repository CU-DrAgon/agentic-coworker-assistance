"""DiscordWorkspace over fake discord objects: no network, no gateway."""

from types import SimpleNamespace as NS

import discord
import pytest

from salid.workspace import (
    BUTTONS,
    CAPABILITIES,
    VOICE,
    Button,
    FileOut,
    MsgRef,
    WorkspaceError,
    button_id,
)
from salid.workspace_discord import CHUNK, DiscordWorkspace, render_view

ME, ANA, BEN, BEN2, BOT = 1, 100, 200, 201, 300
DEFAULT_ROLE = object()


class FakeMessage:
    def __init__(self, mid):
        self.id = mid


class Partial:
    def __init__(self, channel, mid):
        self.channel, self.id = channel, mid

    async def edit(self, **kw):
        self.channel.edits.append((self.id, kw))


class FakeChannel:
    def __init__(self, cid, name, readers, *, public=False, bot_reads=True):
        self.id, self.name = cid, name
        self.readers = readers  # None = everyone; else a set of member ids
        self.public, self.bot_reads = public, bot_reads
        self.history = True
        self.sent: list[dict] = []
        self.edits: list = []

    def permissions_for(self, who):
        if who is DEFAULT_ROLE:
            return NS(read_messages=self.public, read_message_history=self.public)
        if who.id == ME:
            return NS(read_messages=self.bot_reads, read_message_history=self.bot_reads)
        ok = self.readers is None or who.id in self.readers
        return NS(read_messages=ok, read_message_history=ok and self.history)

    async def send(self, **kw):
        self.sent.append(kw)
        return FakeMessage(7000 + len(self.sent))

    def get_partial_message(self, mid):
        return Partial(self, mid)


def member(uid, name, *, bot=False, global_name=None, user=None):
    dm = FakeChannel(uid + 5000, "dm", None)

    async def create_dm():
        return dm

    return NS(
        id=uid,
        display_name=name,
        global_name=global_name,
        name=user or name.lower(),
        bot=bot,
        dm_channel=None,
        create_dm=create_dm,
        dm=dm,
    )


class FakeGuild:
    id = 4242
    me = NS(id=ME)
    default_role = DEFAULT_ROLE
    filesize_limit = 26214400

    def __init__(self):
        self.members = {
            m.id: m
            for m in (
                member(ANA, "Ana", global_name="Ana G", user="ana_x"),
                member(BEN, "Ben"),
                member(BEN2, "ben", user="ben2"),
                member(BOT, "Salid", bot=True),
            )
        }
        self.general = FakeChannel(10, "general", None, public=True)
        self.secret = FakeChannel(11, "secret", {ANA})
        self.hidden = FakeChannel(12, "hidden", {BEN}, bot_reads=False)
        self.text_channels = [self.general, self.secret, self.hidden]
        self.fetched: list[int] = []

    def get_member(self, uid):
        return self.members.get(uid)

    async def fetch_member(self, uid):
        self.fetched.append(uid)
        raise discord.NotFound(NS(status=404, reason="x"), "unknown member")

    def get_channel(self, cid):
        return next((c for c in self.text_channels if c.id == cid), None)


@pytest.fixture
def guild():
    return FakeGuild()


@pytest.fixture
def ws(guild):
    async def fetch(cid):
        raise discord.NotFound(NS(status=404, reason="x"), "unknown channel")

    client = NS(get_channel=lambda cid: None, fetch_channel=fetch)
    return DiscordWorkspace(
        client,
        guild,
        exec_check=lambda m: m.id == ANA,
        rank_fn=lambda g, m: 7 if m.id == ANA else 1,
    )


def test_identity_and_capabilities(ws, guild):
    assert ws.platform == "discord" and ws.tenant_id == 4242
    assert all(ws.supports(f) for f in CAPABILITIES) and ws.supports(VOICE)
    assert not ws.supports("telepathy")
    assert ws.upload_limit == 26214400
    guild.filesize_limit = 0
    assert ws.upload_limit == 10 * 1024 * 1024
    assert BUTTONS in CAPABILITIES


async def test_person_by_mention_id_and_names(ws):
    assert (await ws.person("<@100>")).id == 100
    assert (await ws.person("<@!100>")).platform_id == "100"
    assert (await ws.person("100")).name == "Ana"
    assert (await ws.person("@ana g")).id == 100  # global name, case-insensitive
    assert (await ws.person("ANA_X")).id == 100  # username
    assert (await ws.person("Ana")).id == 100
    assert await ws.person("") is None and await ws.person("nobody") is None


async def test_person_never_guesses_and_skips_bots(ws, guild):
    assert await ws.person("ben") is None  # "Ben" and "ben" display names collide
    assert (await ws.person("ben2")).id == 201  # a unique username still works
    assert await ws.person("<@300>") is None and await ws.person("salid") is None
    assert await ws.person("<@777>") is None and guild.fetched == [777]  # fetch fallback


async def test_people_exec_and_rank(ws):
    assert sorted(p.id for p in await ws.people()) == [100, 200, 201]
    assert await ws.is_exec(100) and not await ws.is_exec(200) and not await ws.is_exec(5)
    assert await ws.rank(100) == 7 and await ws.rank(200) == 1 and await ws.rank(5) == 0


async def test_rooms_only_readable_and_private_flag(ws):
    rooms = {r.name: r for r in await ws.rooms()}
    assert set(rooms) == {"general", "secret"}
    assert not rooms["general"].private and rooms["secret"].private
    assert rooms["secret"].platform_id == "11" and not rooms["secret"].shared


async def test_room_view_is_the_live_reader_set(ws, guild):
    view = await ws.room_view(11)
    assert view.audience == frozenset({ANA})
    assert {c.name: c.viewers for c in view.rooms} == {"general": frozenset({ANA, BEN, BEN2})}
    guild.secret.readers = {ANA, BEN}  # permissions change: nothing is cached
    assert (await ws.room_view(11)).audience == frozenset({ANA, BEN})
    guild.secret.history = False  # must read the history too
    assert (await ws.room_view(11)).audience == frozenset()


async def test_room_view_unknown_or_unreadable_fails_closed(ws):
    assert (await ws.room_view(12)).audience is None  # the bot cannot read it
    assert (await ws.room_view(999)).audience is None


async def test_post_text_file_buttons_and_thread(ws, guild):
    btn = Button(button_id("task", "done", ws.tenant_id, "t1"), "Done", "success")
    ref = await ws.post(
        10,
        "hello <@100>",
        thread=MsgRef("discord", 10, "55"),
        files=[FileOut("r.pdf", b"%PDF", "application/pdf")],
        buttons=[btn],
    )
    assert ref == MsgRef("discord", 10, "7001")
    kw = guild.general.sent[0]
    assert kw["content"] == "hello <@100>"
    assert kw["allowed_mentions"].everyone is False and kw["allowed_mentions"].roles is False
    assert kw["reference"].id == 55
    (f,) = kw["files"]
    assert f.filename == "r.pdf" and f.fp.read() == b"%PDF"
    (item,) = kw["view"].children
    assert item.custom_id == btn.id and item.label == "Done"
    assert item.style is discord.ButtonStyle.success


async def test_long_text_is_split_and_extras_ride_the_last_chunk(ws, guild):
    text = "\n".join(["x" * 900] * 5)
    ref = await ws.post(
        10, text, files=[FileOut("a", b"1")], buttons=[Button("salid:a:b:1:c", "Go")]
    )
    sent = guild.general.sent
    assert len(sent) >= 3 and all(len(k["content"]) <= CHUNK for k in sent)
    assert all("view" not in k and "files" not in k for k in sent[:-1])
    assert "view" in sent[-1] and "files" in sent[-1]
    assert ref.platform_ref == str(7000 + len(sent))


async def test_edit_replaces_or_clears_buttons(ws, guild):
    await ws.edit(MsgRef("discord", 10, "9"), "new")
    mid, kw = guild.general.edits[0]
    assert mid == 9 and kw["content"] == "new" and kw["view"] is None
    await ws.edit(MsgRef("discord", 10, "9"), "new", buttons=[Button("salid:a:b:1:c", "Go")])
    assert guild.general.edits[1][1]["view"].children[0].custom_id == "salid:a:b:1:c"


async def test_unknown_room_and_too_many_buttons_raise(ws):
    with pytest.raises(WorkspaceError):
        await ws.post(404, "x")
    with pytest.raises(WorkspaceError):
        await ws.post(10, "x", buttons=[Button(f"salid:a:b:1:{i}", "L") for i in range(26)])


async def test_dm_sends_in_the_dm_channel_and_handles_closed_dms(ws, guild):
    ref = await ws.dm(ANA, "psst")
    dm = guild.members[ANA].dm
    assert ref == MsgRef("discord", dm.id, "7001") and dm.sent[0]["content"] == "psst"
    assert await ws.dm(BOT, "x") is None and await ws.dm(555, "x") is None

    async def closed():
        raise discord.Forbidden(NS(status=403, reason="x"), "cannot send")

    guild.members[BEN].create_dm = closed
    assert await ws.dm(BEN, "x") is None


def test_mention_and_render_view():
    assert DiscordWorkspace.mention(NS(), 5) == "<@5>"
    assert render_view([]) is None
