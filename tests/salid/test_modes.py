"""Per-channel modes: resolution, name detection, routing per mode, conversation window and
sign-off, join-mode gating and budget, snooze, /โหมด. Discord objects are small fakes, the gate
model is replaced, nothing touches the network."""

import itertools
from datetime import UTC, datetime
from types import SimpleNamespace

import discord
import pytest
from conftest import GUILD, FixedClockSettings
from test_bot_multitenant import ME, FakeChannel, FakeGuild, Msg, member, photo

from salid import bot as botmod
from salid.bot import SecretaryBot
from salid.gate import is_join_candidate
from salid.guild_settings import GuildSettingsUpdate, MemoryGuildStore
from salid.i18n import t
from salid.modes import (
    CHAT_CHANNEL_NAME,
    MODES,
    mode_label,
    parse_snooze_command,
    resolve_mode,
)
from salid.naming import mentions_name
from salid.store import SQLiteStore

MANAGER, STAFF, EXEC = 3, 4, 6
_ids = itertools.count(1000)


# ------------------------------------------------------------------ pure functions
def test_mode_resolution_order():
    kw = dict(channel_modes={}, ask_channel_ids=set())
    assert resolve_mode(1, "general", **kw) == "call"  # guild default
    assert resolve_mode(1, "general", **kw, default_mode="join") == "join"
    assert resolve_mode(1, "general", channel_modes={}, ask_channel_ids={1}) == "chat"
    assert resolve_mode(1, f"x-{CHAT_CHANNEL_NAME}", **kw) == "chat"
    assert resolve_mode(1, CHAT_CHANNEL_NAME, **kw, default_mode="off") == "chat"
    # an explicit channel setting beats ask ids, the chat name and the default
    explicit = dict(channel_modes={"1": "listen"}, ask_channel_ids={1})
    assert resolve_mode(1, CHAT_CHANNEL_NAME, **explicit, default_mode="join") == "listen"
    assert resolve_mode(1, "x", channel_modes={"1": "bogus"}, ask_channel_ids=set()) == "call"


@pytest.mark.parametrize("name", ["salid", "Salid-chat", "ถาม-สลิด", "สลิด"])
def test_channels_merely_named_after_salid_are_no_longer_chat_channels(name):
    assert resolve_mode(1, name, channel_modes={}, ask_channel_ids=set()) == "call"


@pytest.mark.parametrize(
    "text",
    ["สลิดช่วยหน่อย", "ช่วยสรุปหน่อยสลิด", "สะลิดอยู่ไหม", "สลิตๆ", "ซาลิด ยอดเท่าไร", "Salid ยอด?",
     "hey SALID", "sa lid help", "สลิดครับ"],
)  # fmt: skip
def test_name_detection_catches_misspellings(text):
    assert mentions_name(text)


@pytest.mark.parametrize(
    "text",
    ["ส่งสลิปให้หน่อย", "สลิปโอนเงินนี้ยอดเท่าไร", "แนบสลิปแล้วนะ", "สลิปสลิป", "unsalidated",
     "usa lid", "<:salid:123456> ฮ่าๆ", "ดู https://x.example/salid ด้วย", "", "สลิ"],
)  # fmt: skip
def test_name_detection_has_no_false_positives(text):
    assert not mentions_name(text)


@pytest.mark.parametrize(
    "text, expected",
    [("เงียบก่อน", "snooze"), ("เงียบ", "snooze"), ("พักก่อนนะ", "snooze"),
     ("คุยได้แล้ว", "unsnooze"), ("กลับมาคุยได้", "unsnooze"),
     ("be quiet", "snooze"), ("Quiet", "snooze"), ("shh", "snooze"), ("SHUSH please", "snooze"),
     ("mute", "snooze"), ("hush", "snooze"), ("snooze", "snooze"), ("  be   quiet!! ", "snooze"),
     ("you can talk", "unsnooze"), ("talk again", "unsnooze"), ("unmute", "unsnooze"),
     ("Unsnooze", "unsnooze"), ("wake up", "unsnooze"), ("come back", "unsnooze"),
     ("quietly tell me the total", None), ("how much did we mute", None),
     ("ยอดเท่าไร", None), ("", None), ("เงียบก่อน" + "ก" * 60, None)],
)  # fmt: skip
def test_parse_snooze_command(text, expected):
    assert parse_snooze_command(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [("เดือนนี้จ่ายไปเท่าไหร่", True), ("พรุ่งนี้ประชุมกี่โมง", True), ("ไปกินข้าวกัน", False),
     ("ok", False), ("ใครเบิกค่าน้ำมัน", True), ("นี่ใช่ไหม", True), ("สรุปให้หน่อย", True),
     ("ฮ่าๆๆ", False), ("จ่าย 500 บาท", True), ("what?", True),
     ("what did we spend on ingredients", True), ("How much was the Makro receipt", True),
     ("can someone check the budget", True), ("is there a summary of the meeting", True),
     ("who reimbursed the taxi", True), ("Could we split the cost", True),
     ("expenses look high", True), ("it was 500 baht", True), ("where is the invoice", True),
     ("lunch at noon", False), ("thanks", False), ("scan it later", False)],
)  # fmt: skip
def test_join_free_filter(text, expected):
    assert is_join_candidate(text) is expected


# ------------------------------------------------------------------ fakes
class Clock:
    def __init__(self):
        self.t = 10_000.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class Sent:
    def __init__(self):
        self.reactions = []

    async def add_reaction(self, e):
        self.reactions.append(e)


class TMsg(Msg):
    def __init__(self, guild, channel, author, content="hi", attachments=(), mention=False):
        super().__init__(guild, channel, author, content, attachments, mention)
        self.id, self.reactions, self.role_mentions = next(_ids), [], []
        self.created_at = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
        self.sent = []
        channel.log.append(self)

    async def reply(self, text, **kw):
        self.replies.append(text)
        sent = Sent()
        self.sent.append(sent)
        return sent

    async def add_reaction(self, e):
        self.reactions.append(e)


class Chan(FakeChannel):
    def __init__(self, id, name="ทั่วไป"):
        super().__init__(id, name)
        self.log = []

    async def history(self, limit, before=None):
        for m in list(reversed(self.log))[:limit]:
            if m is not before:
                yield m


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setattr(SecretaryBot, "user", property(lambda s: ME))
    g = FakeGuild(GUILD, "A", [Chan(10), Chan(11, CHAT_CHANNEL_NAME), Chan(12, "ถาม-สลิด")])
    monkeypatch.setattr(SecretaryBot, "guilds", property(lambda s: [g]))
    return SimpleNamespace(g=g, gen=g.text_channels[0], chat=g.text_channels[1],
                           named=g.text_channels[2])  # fmt: skip


@pytest.fixture
def bot(world, monkeypatch, tmp_path):
    clock = Clock()
    b = SecretaryBot(
        FixedClockSettings(exec_user_ids={EXEC}),
        SQLiteStore(str(tmp_path / "t.db")),
        None,
        guilds=MemoryGuildStore(),
        clock=clock,
    )
    b.clock, b.calls = clock, []
    b.answer_kw, b.hints_seen = [], []
    b.gate = SimpleNamespace(window=False, pick="first", window_calls=0, pick_calls=0, silent=False)

    async def intake(m, quiet=False):
        b.calls.append(("intake-quiet" if quiet else "intake", m.content))

    async def answer(m, guild, who, **kw):  # like the real one: reply, then note the answer
        b.calls.append(("answer", m.content))
        b.answer_kw.append(kw)
        if kw.get("interjection") and b.gate.silent:
            return False  # the model said NO_REPLY: nothing posted
        b._note_answer(m, await m.reply("ans", mention_author=False))
        return True

    def window_gate(client, model, history, line, *, channel_id):
        b.gate.window_calls += 1
        return b.gate.window

    def pick_gate(client, model, lines, ids, *, channel_id, hints=None):
        b.gate.pick_calls += 1
        b.hints_seen.append((list(ids), hints))
        return ids[0] if b.gate.pick == "first" else b.gate.pick

    monkeypatch.setattr(botmod.gate, "addressed_to_salid", window_gate)
    monkeypatch.setattr(botmod.gate, "pick_interjection", pick_gate)
    b._intake, b._answer = intake, answer
    return b


def set_mode(bot, channel_id, mode):
    bot.guilds_store.set_channel_mode(GUILD, channel_id, mode)
    bot.team._cache.clear()


def say(world, channel, content="hi", who=STAFF, **kw):
    author = who if not isinstance(who, int) else member(world.g, who)
    return TMsg(world.g, channel, author, content, **kw)


async def send(bot, world, channel, content="hi", who=STAFF, **kw):
    m = say(world, channel, content, who, **kw)
    await bot.on_message(m)
    return m


# ------------------------------------------------------------------ off / listen
async def test_off_ignores_everything(bot, world):
    set_mode(bot, 10, "off")
    ch = world.gen
    for kw in ({"attachments": [photo()]}, {"mention": True}):
        m = await send(bot, world, ch, "ยอดเท่าไร", **kw)
        assert not m.reactions and not m.replies
    await send(bot, world, ch, "สลิดช่วยหน่อย")
    assert bot.calls == []


async def test_off_does_not_stop_the_slash_command(bot, world):
    set_mode(bot, 10, "off")
    it = Interaction(world.g, member(world.g, MANAGER, manage_guild=True), world.gen)
    await bot.handle_mode(it, "call")
    assert bot._mode(world.gen, GUILD) == "call"


async def test_listen_reads_receipts_with_reaction_only(bot, world):
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(receipt_channel_ids=["10"]), "x")
    set_mode(bot, 10, "listen")
    await send(bot, world, world.gen, "ร้านนี้", attachments=[photo()])
    assert bot.calls == [("intake-quiet", "ร้านนี้")]


async def test_listen_answers_a_tag_but_not_a_bare_name(bot, world):
    set_mode(bot, 10, "listen")
    m = await send(bot, world, world.gen, "ยอดเท่าไร", mention=True)
    n = await send(bot, world, world.gen, "สลิด ยอดเท่าไร")
    assert bot.calls == [("answer", "ยอดเท่าไร")] and not n.reactions and not m.reactions
    assert not n.replies and len(bot.calls) == 1  # never unprompted, no name-only answers


async def test_listen_still_accepts_meeting_commands_by_mention(bot, world, monkeypatch):
    set_mode(bot, 10, "listen")
    joined = []

    async def join(who, fallback_channel=None, lang="th"):
        joined.append(who.id)
        return "เข้าแล้ว"

    monkeypatch.setattr(bot.meetings, "join", join)
    author = member(world.g, STAFF)
    author.voice = object()
    m = TMsg(world.g, world.gen, author, "<@999> เข้าประชุม", mention=True)
    await bot.on_message(m)
    assert joined == [STAFF] and m.replies == ["เข้าแล้ว"] and not m.reactions


async def test_intake_quiet_adds_only_the_reaction(bot, world, monkeypatch):
    from salid.extract import IngestResult

    monkeypatch.setattr(
        botmod, "ingest_attachment", lambda *a, **k: IngestResult("created", "ok", "✅", None)
    )
    m = say(world, world.gen, "x", attachments=[photo()])
    await SecretaryBot._intake(bot, m, quiet=True)
    assert m.reactions == ["✅"] and m.replies == []


# ------------------------------------------------------------------ call
async def test_call_answers_tag_and_name_without_tag(bot, world):
    ch = world.gen
    await send(bot, world, ch, "ยอดเท่าไร", mention=True)
    await send(bot, world, ch, "สะลิด ยอดวันนี้เท่าไร")
    await send(bot, world, ch, "ช่วยสรุปหน่อยสลิตๆ")
    await send(bot, world, ch, "ส่งสลิปให้แล้วนะ")  # a payment slip, not Salid
    await send(bot, world, ch, "ไปกินข้าวกัน")
    assert [c[1] for c in bot.calls] == ["ยอดเท่าไร", "สะลิด ยอดวันนี้เท่าไร", "ช่วยสรุปหน่อยสลิตๆ"]
    assert bot.gate.window_calls == 2  # only the two plain messages after Salid spoke


async def test_in_window_messages_go_through_the_gate(bot, world):
    ch = world.gen
    await send(bot, world, ch, "ยอดเท่าไร", mention=True)
    bot.calls.clear()
    bot.gate.window = True
    await send(bot, world, ch, "แล้วเดือนก่อนล่ะ")
    bot.gate.window = False
    await send(bot, world, ch, "เย็นนี้กินอะไรดี")
    assert bot.calls == [("answer", "แล้วเดือนก่อนล่ะ")] and bot.gate.window_calls == 2


async def test_window_is_per_channel_and_expires(bot, world):
    await send(bot, world, world.gen, "ยอดเท่าไร", mention=True)
    bot.gate.window = True
    await send(bot, world, world.named, "คนละห้อง")  # another channel: no window there
    assert bot.gate.window_calls == 0
    bot.clock.advance(botmod.CONVERSATION_SECONDS + 1)
    await send(bot, world, world.gen, "นานแล้ว")
    assert bot.gate.window_calls == 0 and len(bot.calls) == 1


async def test_mention_of_someone_else_is_ignored_in_the_window(bot, world):
    await send(bot, world, world.gen, "ยอดเท่าไร", mention=True)
    bot.gate.window = True
    other = say(world, world.gen, "<@5> ไปกินข้าวไหม")
    other.mentions = [SimpleNamespace(id=5)]
    await bot.on_message(other)
    assert len(bot.calls) == 1 and bot.gate.window_calls == 0


async def test_reply_to_salid_is_always_answered(bot, world):
    first = await send(bot, world, world.gen, "ยอดเท่าไร", mention=True)
    bot.clock.advance(botmod.CONVERSATION_SECONDS * 5)  # long after: still a direct reply
    m = say(world, world.gen, "อธิบายเพิ่มหน่อย")
    m.reference = SimpleNamespace(message_id=first.id, resolved=SimpleNamespace(author=ME))
    await bot.on_message(m)
    assert bot.calls[-1] == ("answer", "อธิบายเพิ่มหน่อย") and bot.gate.window_calls == 0


async def test_bot_role_mention_counts_as_a_call(bot, world):
    m = say(world, world.gen, "ยอดเท่าไร")
    m.role_mentions = [SimpleNamespace(id=5, tags=SimpleNamespace(bot_id=ME.id))]
    await bot.on_message(m)
    assert bot.calls == [("answer", "ยอดเท่าไร")]


# ------------------------------------------------------------------ chat
async def test_chat_answers_every_message(bot, world):
    set_mode(bot, 10, "chat")
    await send(bot, world, world.gen, "อะไรก็ได้")
    await send(bot, world, world.chat, "ห้องคุยกับน้องสลิด")  # by channel name
    await send(bot, world, world.named, "ห้องชื่อธรรมดา")  # old heuristic: gone
    assert [c[1] for c in bot.calls] == ["อะไรก็ได้", "ห้องคุยกับน้องสลิด"]


async def test_ask_channel_ids_mean_chat(bot, world):
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(ask_channel_ids=["12"]), "x")
    await send(bot, world, world.named, "ทำอะไรได้บ้าง")
    assert bot.calls == [("answer", "ทำอะไรได้บ้าง")]


async def test_chat_mode_has_no_conversation_or_signoff(bot, world):
    await send(bot, world, world.chat, "a")
    await send(bot, world, world.chat, "b")
    bot.clock.advance(1000)
    await bot.tick()
    assert world.chat.sent == []


async def test_default_mode_of_the_guild_applies(bot, world):
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(default_mode="chat"), "x")
    await send(bot, world, world.gen, "อะไรก็ได้")
    assert bot.calls == [("answer", "อะไรก็ได้")]


# ------------------------------------------------------------------ sign-off
async def test_two_replies_then_silence_posts_one_signoff(bot, world):
    ch = world.gen
    await send(bot, world, ch, "ยอดเท่าไร", mention=True)
    bot.clock.advance(60)
    await send(bot, world, ch, "สลิด แล้วเดือนก่อน")
    bot.clock.advance(botmod.CONVERSATION_SECONDS - 1)
    await bot.tick()
    assert ch.sent == []  # the second answer refreshed the window
    bot.clock.advance(2)
    await bot.tick()
    await bot.tick()
    assert ch.sent == ["ถ้ามีอะไรก็ทักน้องสลิดมาได้เลยนะครับ 👋"]  # once


async def test_single_answer_gets_a_wave_reaction_instead(bot, world):
    m = await send(bot, world, world.gen, "ยอดเท่าไร", mention=True)
    bot.clock.advance(botmod.CONVERSATION_SECONDS + 1)
    await bot.tick()
    assert world.gen.sent == [] and m.sent[0].reactions == ["👋"]
    await bot.tick()
    assert m.sent[0].reactions == ["👋"]


async def test_new_conversation_after_signoff_counts_again(bot, world):
    ch = world.gen
    for _ in range(2):
        await send(bot, world, ch, "สลิด ถามหน่อย")
    bot.clock.advance(botmod.CONVERSATION_SECONDS + 1)
    await bot.tick()
    m = await send(bot, world, ch, "สลิด อีกที")
    bot.clock.advance(botmod.CONVERSATION_SECONDS + 1)
    await bot.tick()
    assert len(ch.sent) == 1 and m.sent[0].reactions == ["👋"]


# ------------------------------------------------------------------ join
async def test_join_candidate_waits_for_quiet_then_one_gate_call(bot, world):
    set_mode(bot, 10, "join")
    ch = world.gen
    await send(bot, world, ch, "เดือนนี้จ่ายค่าวัตถุดิบไปเท่าไหร่")
    await send(bot, world, ch, "ฮ่าๆ")  # not a candidate, but it is activity
    bot.clock.advance(20)
    await bot.tick()
    assert bot.gate.pick_calls == 0  # 20 s since the last message: not quiet yet
    bot.clock.advance(botmod.QUIET_SECONDS)
    await bot.tick()
    await bot.tick()
    assert bot.gate.pick_calls == 1
    assert bot.calls == [("answer", "เดือนนี้จ่ายค่าวัตถุดิบไปเท่าไหร่")]


async def test_join_gate_no_means_silence(bot, world):
    set_mode(bot, 10, "join")
    bot.gate.pick = None
    await send(bot, world, world.gen, "ใครไปประชุมบ้าง")
    bot.clock.advance(60)
    await bot.tick()
    assert bot.gate.pick_calls == 1 and bot.calls == []


async def test_join_skips_chatter_and_messages_that_mention_people(bot, world):
    set_mode(bot, 10, "join")
    await send(bot, world, world.gen, "ไปกินข้าวกัน")
    tagged = say(world, world.gen, "<@5> ใครเบิกเงินไหม")
    tagged.mentions = [SimpleNamespace(id=5)]
    await bot.on_message(tagged)
    bot.clock.advance(60)
    await bot.tick()
    assert bot.gate.pick_calls == 0 and bot.calls == []


async def test_call_mode_never_interjects(bot, world):
    await send(bot, world, world.gen, "ใครเบิกเงินไหม")
    bot.clock.advance(60)
    await bot.tick()
    assert bot.gate.pick_calls == 0 and bot.calls == []


async def test_join_budget_is_three_per_hour_per_channel(bot, world):
    set_mode(bot, 10, "join")
    ch = world.gen

    users = itertools.count(100)  # one person is on a 10 min cooldown, so everyone is new

    async def attempt():
        await send(bot, world, ch, "ยอดค่าใช้จ่ายเท่าไหร่", who=next(users))
        bot.clock.advance(60)
        await bot.tick()

    for _ in range(5):
        await attempt()
    assert len(bot.calls) == 3
    assert bot.gate.pick_calls == 3  # out of budget: no more gate calls either
    bot.clock.advance(3600)
    await attempt()
    assert len(bot.calls) == 4


async def test_budget_is_per_channel(bot, world):
    set_mode(bot, 10, "join")
    set_mode(bot, 12, "join")
    users = itertools.count(100)
    for ch in (world.gen, world.named):
        for _ in range(3):
            await send(bot, world, ch, "เบิกได้ไหม", who=next(users))
            bot.clock.advance(60)
            await bot.tick()
    assert len(bot.calls) == 6


async def test_explicit_calls_do_not_use_the_budget(bot, world):
    set_mode(bot, 10, "join")
    ch = world.gen
    for _ in range(5):
        await send(bot, world, ch, "สลิด ยอดเท่าไร")
    assert len(bot.calls) == 5
    bot.clock.advance(botmod.CONVERSATION_SECONDS + 1)
    await bot.tick()
    await send(bot, world, ch, "ใครเบิกเงินไหม")
    bot.clock.advance(60)
    await bot.tick()
    assert bot.calls[-1] == ("answer", "ใครเบิกเงินไหม")  # full budget was still there


async def test_join_picks_only_among_candidates(bot, world):
    set_mode(bot, 10, "join")
    bot.gate.pick = "999999"  # not a candidate id
    await send(bot, world, world.gen, "ใครเบิกเงินไหม")
    bot.clock.advance(60)
    await bot.tick()
    assert bot.calls == []


async def test_join_mode_also_has_call_behaviour(bot, world):
    set_mode(bot, 10, "join")
    await send(bot, world, world.gen, "สลิด ยอดเท่าไร")
    assert bot.calls == [("answer", "สลิด ยอดเท่าไร")]


# ------------------------------------------------------------------ snooze
async def test_snooze_makes_the_channel_listen_for_an_hour(bot, world):
    ch = world.gen
    ack = await send(bot, world, ch, "<@999> เงียบก่อน", mention=True)
    assert ack.replies == [t("snooze", "th")] and bot.calls == []
    m = await send(bot, world, ch, "ยอดเท่าไร", mention=True)
    named = await send(bot, world, ch, "สลิด ยอดเท่าไร")
    assert bot.calls == [] and m.reactions == ["👂"] and not named.reactions
    bot.clock.advance(botmod.SNOOZE_SECONDS + 1)
    await send(bot, world, ch, "ยอดเท่าไร", mention=True)
    assert bot.calls == [("answer", "ยอดเท่าไร")]


async def test_snooze_is_per_channel_and_open_to_everyone(bot, world):
    await send(bot, world, world.gen, "<@999> พักก่อน", mention=True)
    await send(bot, world, world.named, "สลิด ยอดเท่าไร")
    assert bot.calls == [("answer", "สลิด ยอดเท่าไร")]


async def test_unsnooze_brings_salid_back(bot, world):
    ch = world.gen
    await send(bot, world, ch, "<@999> เงียบก่อน", mention=True)
    back = await send(bot, world, ch, "<@999> คุยได้แล้ว", mention=True)
    assert back.replies == [t("unsnooze", "th")]
    await send(bot, world, ch, "สลิด ยอดเท่าไร")
    assert bot.calls == [("answer", "สลิด ยอดเท่าไร")]
    again = await send(bot, world, ch, "<@999> คุยได้แล้ว", mention=True)
    assert again.replies == [t("not_snoozed", "th")]


async def test_snooze_stops_the_signoff_and_join_batches(bot, world):
    set_mode(bot, 10, "join")
    await send(bot, world, world.gen, "สลิด ยอดเท่าไร")
    await send(bot, world, world.gen, "ใครเบิกเงินไหม")
    await send(bot, world, world.gen, "<@999> เงียบก่อน", mention=True)
    bot.calls.clear()
    bot.clock.advance(botmod.CONVERSATION_SECONDS + 60)
    await bot.tick()
    assert bot.calls == [] and world.gen.sent == []


async def test_snooze_does_not_wake_an_off_channel(bot, world):
    set_mode(bot, 10, "off")
    m = await send(bot, world, world.gen, "<@999> เงียบก่อน", mention=True)
    assert m.replies == []


# ------------------------------------------------------------------ /โหมด
class Interaction:
    def __init__(self, guild, user, channel):
        self.guild, self.user, self.channel, self.sent = guild, user, channel, []
        self.response = SimpleNamespace(send_message=self._send)

    async def _send(self, text, ephemeral=False):
        self.sent.append((text, ephemeral))


async def test_mode_without_argument_shows_current_mode_and_help(bot, world):
    set_mode(bot, 10, "join")
    it = Interaction(world.g, member(world.g, STAFF), world.gen)
    await bot.handle_mode(it, None)
    ((text, ephemeral),) = it.sent
    assert ephemeral and "🎧 ร่วมวง" in text.splitlines()[0]
    assert all(mode_label(m, "th") in text for m in MODES)


async def test_mode_change_is_refused_for_regular_members(bot, world):
    it = Interaction(world.g, member(world.g, STAFF, roles=["พนักงาน"]), world.gen)
    await bot.handle_mode(it, "join")
    assert it.sent == [(t("mode_denied", "th"), True)]
    assert bot._mode(world.gen, GUILD) == "call"
    assert bot.guilds_store.get(GUILD).channel_modes == {}


async def test_mode_change_by_manager_is_public_persisted_and_immediate(bot, world):
    it = Interaction(world.g, member(world.g, MANAGER, manage_guild=True), world.gen)
    await bot.handle_mode(it, "join")
    assert it.sent == [("ตั้งห้องนี้เป็น 🎧 ร่วมวง แล้วครับ", False)]
    assert bot.guilds_store.get(GUILD).channel_modes == {"10": "join"}
    assert bot._mode(world.gen, GUILD) == "join"  # no waiting for the cache TTL
    assert bot._mode(world.named, GUILD) == "call"  # other channels untouched


async def test_executives_may_change_modes_and_it_clears_a_snooze(bot, world):
    await send(bot, world, world.gen, "<@999> เงียบก่อน", mention=True)
    it = Interaction(world.g, member(world.g, EXEC), world.gen)
    await bot.handle_mode(it, "chat")
    assert bot._mode(world.gen, GUILD) == "chat"
    it2 = Interaction(world.g, member(world.g, 7, roles=["Executive"]), world.gen)
    await bot.handle_mode(it2, "off")
    assert bot._mode(world.gen, GUILD) == "off"


async def test_explicit_call_mode_overrides_the_chat_channel_name(bot, world):
    it = Interaction(world.g, member(world.g, MANAGER, manage_guild=True), world.chat)
    await bot.handle_mode(it, "call")
    assert bot._mode(world.chat, GUILD) == "call"


def test_mode_command_has_five_choices_in_both_languages(bot):
    bot._register_commands()
    cmds = {c.name: c for c in bot.tree.get_commands(type=discord.AppCommandType.chat_input)}
    for name in ("โหมด", "mode"):
        (param,) = cmds[name].parameters
        assert [c.value for c in param.choices] == list(MODES) and not param.required
        assert cmds[name].guild_only


# ------------------------------------------------------------------ on_guild_join
async def test_join_saves_chat_mode_for_a_reused_channel(bot, world):
    reused = Chan(50, f"เก่า-{CHAT_CHANNEL_NAME}")
    world.g.text_channels[:] = [world.gen, reused]
    reused.guild, reused.mention = world.g, "<#50>"
    await bot.on_guild_join(world.g)
    assert bot.guilds_store.get(GUILD).channel_modes == {"50": "chat"}
    set_mode(bot, 50, None)  # even if the setting went missing, the name still means chat
    assert bot._mode(reused, GUILD) == "chat"
    assert reused.sent  # welcomed there


async def test_join_saves_chat_mode_for_a_created_channel(bot, world):
    world.g.text_channels[:] = [world.gen]
    world.g.me = SimpleNamespace(guild_permissions=SimpleNamespace(manage_channels=True))

    async def create_text_channel(name, **kw):
        ch = Chan(60, name)
        ch.mention, ch.guild = "<#60>", world.g
        world.g.text_channels.append(ch)
        return ch

    world.g.create_text_channel = create_text_channel
    await bot.on_guild_join(world.g)
    assert bot.guilds_store.get(GUILD).channel_modes == {"60": "chat"}


# ------------------------------------------------------------------ English commands and replies
def manager(world):
    return member(world.g, MANAGER, manage_guild=True)


def set_lang(bot, channel_id, language):
    bot.guilds_store.set_channel_language(GUILD, channel_id, language)
    bot.team._cache.clear()


async def test_english_snooze_and_unsnooze_by_mention(bot, world):
    ch = world.gen
    set_lang(bot, 10, "en")
    ack = await send(bot, world, ch, "<@999> Be quiet!", mention=True)
    assert ack.replies == [t("snooze", "en")] and "stay quiet" in ack.replies[0]
    assert bot._is_snoozed(10)
    back = await send(bot, world, ch, "<@999> you can talk", mention=True)
    assert back.replies == [t("unsnooze", "en")] and not bot._is_snoozed(10)
    again = await send(bot, world, ch, "<@999> unmute", mention=True)
    assert again.replies == [t("not_snoozed", "en")]


async def test_fixed_replies_are_thai_by_default_and_english_when_set(bot, world):
    th = await send(
        bot, world, world.gen, "<@999> shh", mention=True
    )  # English words, Thai channel
    assert th.replies == [t("snooze", "th")]
    set_lang(bot, 12, "en")
    en = await send(bot, world, world.named, "<@999> เงียบก่อน", mention=True)  # Thai words, English
    assert en.replies == [t("snooze", "en")]


async def test_auto_replies_follow_the_language_the_person_wrote_in(bot, world):
    set_lang(bot, 10, "auto")
    en = await send(bot, world, world.gen, "<@999> snooze", mention=True)
    th = await send(bot, world, world.gen, "<@999> เงียบก่อน", mention=True)
    assert en.replies == [t("snooze", "en")] and th.replies == [t("snooze", "th")]


async def test_english_meeting_commands_by_mention(bot, world, monkeypatch):
    seen = []

    async def join(who, fallback_channel=None, lang="th"):
        seen.append(("join", lang))
        return "joined"

    async def leave(who, lang="th"):
        seen.append(("leave", lang))
        return "left"

    monkeypatch.setattr(bot.meetings, "join", join)
    monkeypatch.setattr(bot.meetings, "leave", leave)
    set_lang(bot, 10, "auto")
    author = member(world.g, STAFF)
    author.voice = object()
    for text in ("join the meeting", "Come to the meeting", "JOIN", "leave meeting", "end meeting"):
        m = TMsg(world.g, world.gen, author, f"<@999> {text}", mention=True)
        await bot.on_message(m)
        assert m.replies in (["joined"], ["left"]), text
    assert seen == [("join", "en")] * 3 + [("leave", "en")] * 2
    thai = TMsg(world.g, world.gen, author, "<@999> เข้าประชุม", mention=True)
    await bot.on_message(thai)
    assert seen[-1] == ("join", "th")  # Thai still works, and auto follows the Thai text


async def test_greeting_and_signoff_follow_the_channel_language(bot, world, monkeypatch):
    del bot._answer  # the real one: a bare mention greets
    set_lang(bot, 10, "en")
    m = await send(bot, world, world.gen, "<@999>", mention=True)
    assert m.replies == [t("greeting", "en")] and "Salid here" in m.replies[0]
    monkeypatch.setattr(
        "salid.bot.ask", lambda *a, **kw: SimpleNamespace(text="ok", seconds=0, rounds=0)
    )
    await send(bot, world, world.gen, "<@999> how much?", mention=True)  # reply count reaches 2
    bot.clock.advance(botmod.CONVERSATION_SECONDS + 1)
    await bot.tick()
    assert world.gen.sent == [t("signoff", "en")]


async def test_answer_passes_the_language_setting_and_follows_it_for_fixed_text(
    bot, world, monkeypatch
):
    seen = []
    monkeypatch.setattr(
        "salid.bot.ask",
        lambda s, store, q, **kw: (
            seen.append(kw["language"]) or SimpleNamespace(text="ok", seconds=0, rounds=0)
        ),
    )
    del bot._answer
    for ch, setting in ((world.gen, "th"), (world.named, "en"), (world.chat, "auto")):
        set_lang(bot, ch.id, setting)
        m = TMsg(world.g, ch, member(world.g, STAFF), "<@999> hello?", mention=True)
        await bot._answer(m, world.g, m.author)
    assert seen == ["th", "en", "auto"]
    # features off: the refusal speaks the resolved language (auto + English text -> English)
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features=dict(qa=False)), "x")
    bot.team._cache.clear()
    set_lang(bot, 11, "auto")
    m = TMsg(world.g, world.chat, member(world.g, STAFF), "hello?")
    await bot._answer(m, world.g, m.author)
    assert m.replies == [t("qa_off", "en")]
    m = TMsg(world.g, world.chat, member(world.g, STAFF), "ยอดเท่าไร")
    await bot._answer(m, world.g, m.author)
    assert m.replies == [t("qa_off", "th")]


async def test_direct_messages_use_auto(bot, world, monkeypatch):
    author = member(world.g, STAFF)
    other = FakeGuild(2000, "B")  # shares nobody with the author: the DM is refused
    monkeypatch.setattr(SecretaryBot, "guilds", property(lambda s: [world.g, other]))
    other.get_member = lambda uid: author
    world.g.get_member = lambda uid: author  # two shared guilds
    en = Msg(None, world.gen, author, "hello there")
    th = Msg(None, world.gen, author, "สวัสดี")
    for m in (en, th):
        await bot.on_message(m)
    assert en.replies == [t("dm", "en", mention="<@999>")]
    assert th.replies == [t("dm", "th", mention="<@999>")]


# ------------------------------------------------------------------ typed language commands
@pytest.mark.parametrize(
    "text, expected",
    [("พูดภาษาอังกฤษ", "en"), ("speak English", "en"), ("English please", "en"),
     ("พูดไทย", "th"), ("speak Thai", "th"), ("ภาษาไทย", "th"),
     ("auto language", "auto"), ("ภาษาอัตโนมัติ", "auto")],
)  # fmt: skip
async def test_manager_changes_the_channel_language_by_mention(bot, world, text, expected):
    set_lang(bot, 10, "en" if expected != "en" else "th")
    m = await send(bot, world, world.gen, f"<@999> {text}", who=manager(world), mention=True)
    assert bot.guilds_store.get(GUILD).channel_languages == {"10": expected}
    assert bot.team.channel_language(GUILD, 10) == expected  # immediate, no cache wait
    assert len(m.replies) == 1 and bot.calls == []  # a command, not a question
    assert bot.team.channel_language(GUILD, 11) == "th"  # other channels untouched


async def test_language_command_confirms_in_the_new_language(bot, world):
    en = await send(bot, world, world.gen, "<@999> speak English", who=manager(world), mention=True)
    assert en.replies == [t("lang_set", "en", label=t("lang_label_en", "en"))]
    th = await send(bot, world, world.gen, "<@999> พูดไทย", who=manager(world), mention=True)
    assert th.replies == [t("lang_set", "th", label=t("lang_label_th", "th"))]


async def test_language_command_is_refused_for_regular_members(bot, world):
    m = await send(bot, world, world.gen, "<@999> speak English", who=STAFF, mention=True)
    assert m.replies == [t("lang_denied", "th")]  # refusal in the channel's (Thai) language
    assert bot.guilds_store.get(GUILD).channel_languages == {}
    set_lang(bot, 10, "en")
    m = await send(bot, world, world.gen, "<@999> พูดไทย", who=STAFF, mention=True)
    assert m.replies == [t("lang_denied", "en")] and "Only server admins" in m.replies[0]
    assert bot.guilds_store.get(GUILD).channel_languages == {"10": "en"}


async def test_executives_may_change_the_language_by_mention(bot, world):
    await send(bot, world, world.gen, "<@999> speak English", who=EXEC, mention=True)
    assert bot.team.channel_language(GUILD, 10) == "en"
    boss = member(world.g, 7, roles=["Executive"])
    await send(bot, world, world.gen, "<@999> speak Thai", who=boss, mention=True)
    assert bot.team.channel_language(GUILD, 10) == "th"


async def test_language_words_in_a_question_are_not_a_command(bot, world):
    await send(
        bot,
        world,
        world.gen,
        "<@999> English is hard, how much did we spend?",
        who=MANAGER,
        mention=True,
    )
    assert bot.guilds_store.get(GUILD).channel_languages == {}
    assert bot.calls == [("answer", "<@999> English is hard, how much did we spend?")]


async def test_language_command_needs_a_mention(bot, world):
    await send(bot, world, world.gen, "speak English", who=manager(world))
    assert bot.guilds_store.get(GUILD).channel_languages == {}


# ------------------------------------------------------------------ /ภาษา
async def test_language_without_argument_shows_the_current_language(bot, world):
    it = Interaction(world.g, member(world.g, STAFF), world.gen)
    await bot.handle_language(it, None)
    ((text, ephemeral),) = it.sent
    assert ephemeral and text.startswith(t("lang_current", "th", label="ไทย"))
    set_lang(bot, 10, "en")
    it = Interaction(world.g, member(world.g, STAFF), world.gen)
    await bot.handle_language(it, None)
    assert it.sent[0][0].startswith("Salid's reply language in this channel right now: English")


async def test_language_change_is_refused_for_regular_members(bot, world):
    it = Interaction(world.g, member(world.g, STAFF, roles=["พนักงาน"]), world.gen)
    await bot.handle_language(it, "en")
    assert it.sent == [(t("lang_denied", "th"), True)]
    assert bot.guilds_store.get(GUILD).channel_languages == {}


async def test_language_change_by_manager_is_persisted_and_immediate(bot, world):
    it = Interaction(world.g, member(world.g, MANAGER, manage_guild=True), world.gen)
    await bot.handle_language(it, "en")
    assert it.sent == [(t("lang_set", "en", label="English"), False)]
    assert bot.guilds_store.get(GUILD).channel_languages == {"10": "en"}
    assert bot.team.channel_language(GUILD, 10) == "en"
    assert bot.team.channel_language(GUILD, 12) == "th"
    ex = Interaction(world.g, member(world.g, EXEC), world.gen)
    await bot.handle_language(ex, "auto")
    assert bot.team.channel_language(GUILD, 10) == "auto"
    await bot.handle_language(ex, "fr")  # not a choice
    assert ex.sent[-1][1] is True  # an ephemeral failure notice, nothing saved
    assert bot.team.channel_language(GUILD, 10) == "auto"


async def test_default_language_of_the_guild_applies_until_a_channel_overrides(bot, world):
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(default_language="en"), "x")
    bot.team._cache.clear()
    m = await send(bot, world, world.gen, "<@999> shh", mention=True)
    assert m.replies == [t("snooze", "en")]
    set_lang(bot, 12, "th")
    m = await send(bot, world, world.named, "<@999> shh", mention=True)
    assert m.replies == [t("snooze", "th")]


async def test_interaction_language_for_auto_follows_the_discord_locale(bot, world):
    set_lang(bot, 10, "auto")
    it = Interaction(world.g, member(world.g, STAFF), world.gen)
    assert bot._interaction_lang(it) == "th"  # unknown locale: today's default
    it.locale = "en-US"
    assert bot._interaction_lang(it) == "en"
    it.locale = "th"
    assert bot._interaction_lang(it) == "th"


def test_language_command_has_three_choices_in_both_languages(bot):
    bot._register_commands()
    cmds = {c.name: c for c in bot.tree.get_commands(type=discord.AppCommandType.chat_input)}
    for name in ("ภาษา", "language"):
        (param,) = cmds[name].parameters
        assert [c.value for c in param.choices] == ["th", "en", "auto"] and not param.required
        assert cmds[name].guild_only
        assert all(len(c.description) <= 100 for c in cmds.values())
    names = [c.name for c in cmds["language"].parameters[0].choices]
    assert names == ["ไทย · Thai", "English · อังกฤษ", "อัตโนมัติ · Auto"]


def test_mode_choices_are_bilingual(bot):
    bot._register_commands()
    cmds = {c.name: c for c in bot.tree.get_commands(type=discord.AppCommandType.chat_input)}
    labels = [c.name for c in cmds["mode"].parameters[0].choices]
    assert "🙋 เรียกแล้วคุย · Call" in labels and "💬 ตอบทุกข้อความ · Chat" in labels
    assert all(len(x) <= 100 for x in labels)


async def test_mode_help_and_confirmation_in_english(bot, world):
    set_lang(bot, 10, "en")
    set_mode(bot, 10, "join")
    it = Interaction(world.g, member(world.g, STAFF), world.gen)
    await bot.handle_mode(it, None)
    text = it.sent[0][0]
    assert text.splitlines()[0] == "This channel's mode right now: 🎧 Join"
    assert "💬 Chat — Replies to every message" in text and "ร่วมวง" not in text
    it = Interaction(world.g, member(world.g, MANAGER, manage_guild=True), world.gen)
    await bot.handle_mode(it, "chat")
    assert it.sent == [("This channel is now set to 💬 Chat.", False)]


def test_welcome_message_mentions_both_languages():
    th = t("welcome", "th", mention="@S", chat="#c", days=90)
    en = t("welcome", "en", mention="@S", chat="#c", days=90)
    forms = ("/โหมด", "/mode", "/ภาษา", "/language", "/เข้าประชุม", "/join-meeting", "/ออกประชุม",
             "/leave-meeting", "/ตั้งค่า", "/setup", "speak English")  # fmt: skip
    for text in (th, en):
        assert all(f in text for f in forms), text
    assert "พูดภาษาอังกฤษ" in th and "พูดภาษาไทย" in en
    assert "สวัสดีครับ" in th and "Hi, I'm" in en


async def test_welcome_is_sent_in_the_guild_default_language(bot, world):
    world.g.text_channels[:] = [world.gen]
    world.g.system_channel = world.gen
    world.g.me = SimpleNamespace(guild_permissions=SimpleNamespace(manage_channels=False))
    world.g.me.permissions_for = lambda *_: None
    world.gen.permissions_for = lambda _: SimpleNamespace(send_messages=True)
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(default_language="en"), "x")
    await bot.on_guild_join(world.g)
    assert world.gen.sent and world.gen.sent[0].startswith("Hi, I'm")


# ------------------------------------------------------------------ receipt channel names
@pytest.mark.parametrize(
    "name",
    ["tax-invoice", "Tax-Invoice", "ใบกำกับภาษี", "สลิปโอน", "slips", "บิลร้าน", "bills",
     "expenses", "ค่าใช้จ่าย", "เบิกเงิน", "ใบเสร็จ-ร้าน", "Receipts"],
)  # fmt: skip
def test_receipt_channel_name_hints(name):
    assert botmod.is_receipt_channel_name(name)


@pytest.mark.parametrize("name", ["ทั่วไป", "general", "", None])
def test_non_receipt_channel_names(name):
    assert not botmod.is_receipt_channel_name(name)


async def test_tax_invoice_channel_takes_receipts(bot, world):
    world.gen.name = "tax-invoice"
    await send(bot, world, world.gen, "", attachments=[photo()])
    assert bot.calls == [("intake", "")]


def test_every_slash_command_is_bilingual(bot):
    from discord import app_commands

    bot._register_commands()
    cmds = [c for c in bot.tree.get_commands() if isinstance(c, app_commands.Command)]
    names = {c.name for c in cmds}
    pairs = [
        ("สรุป", "summary"),
        ("ตั้งค่า", "setup"),
        ("โหมด", "mode"),
        ("ภาษา", "language"),
        ("ความจำ", "memory"),
        ("ห้อง", "channels"),
        ("เข้าประชุม", "join-meeting"),
        ("ออกประชุม", "leave-meeting"),
        ("ยกเลิก", "cancel"),
    ]
    for th_name, en_name in pairs:
        assert {th_name, en_name} <= names
    for c in cmds:
        assert " · " in c.description and len(c.description) <= 100, c.name
        for p in c.parameters:
            assert " · " in p.description, (c.name, p.name)
            for ch in p.choices:
                assert " · " in ch.name, ch.name
    th_cmds = [c for c in cmds if not c.name.isascii()]
    en_cmds = [c for c in cmds if c.name.isascii()]
    assert len(th_cmds) == len(en_cmds) == len(pairs)


def test_language_choice_labels_name_both_languages(bot):
    bot._register_commands()
    cmds = {c.name: c for c in bot.tree.get_commands()}
    for ch in cmds["language"].parameters[0].choices:
        assert " · " in ch.name


async def test_ephemeral_replies_follow_locale_for_auto(bot, world):
    set_lang(bot, 10, "auto")
    for cb in (
        lambda it: bot.handle_mode(it, "bogus"),
        lambda it: bot.handle_mode(it, "join"),  # refused for a regular member
    ):
        for locale, lang in (("en-US", "en"), ("th", "th")):
            it = Interaction(world.g, member(world.g, STAFF, roles=["x"]), world.gen)
            it.locale = locale
            await cb(it)
            ((text, ephemeral),) = it.sent
            assert ephemeral and text in (t("mode_failed", lang), t("mode_denied", lang))
    it = Interaction(world.g, member(world.g, STAFF, roles=["x"]), world.gen)
    it.locale = "en-GB"
    await bot.handle_mode(it, None)
    assert "Mode" in it.sent[0][0] or "mode" in it.sent[0][0].lower()
    # in a DM (no guild) the denial also follows the locale
    dm = Interaction(None, member(world.g, STAFF), None)
    dm.locale = "en-US"
    await bot.handle_language(dm, None)
    assert dm.sent == [(t("setup_denied", "en"), True)]


def test_typed_commands_have_thai_and_english_forms():
    from salid.deep_task import parse_deep_trigger
    from salid.i18n import parse_language_command
    from salid.meeting import parse_meeting_command
    from salid.modes import parse_snooze_command

    assert parse_meeting_command("เข้าประชุม") == parse_meeting_command("join") == "join"
    assert parse_meeting_command("ออกประชุม") == parse_meeting_command("leave meeting") == "leave"
    assert parse_snooze_command("เงียบก่อน") == parse_snooze_command("be quiet") == "snooze"
    assert parse_snooze_command("คุยได้แล้ว") == parse_snooze_command("you can talk") == "unsnooze"
    assert parse_language_command("speak English") == "en"
    assert parse_language_command("พูดไทย") == "th"
    assert parse_deep_trigger("งานใหญ่: x") == parse_deep_trigger("deep task: x") == "x"
