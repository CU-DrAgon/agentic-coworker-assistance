"""Per-channel modes: resolution, name detection, routing per mode, conversation window and
sign-off, join-mode gating and budget, snooze, /โหมด. Discord objects are small fakes, the gate
model is replaced, nothing touches the network."""

import itertools
from types import SimpleNamespace

import pytest
from conftest import GUILD, FixedClockSettings
from test_bot_multitenant import ME, FakeChannel, FakeGuild, Msg, member, photo

from salid import bot as botmod
from salid.bot import SecretaryBot
from salid.gate import is_join_candidate
from salid.guild_settings import GuildSettingsUpdate, MemoryGuildStore
from salid.modes import CHAT_CHANNEL_NAME, MODES, parse_snooze_command, resolve_mode
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
     ("ยอดเท่าไร", None), ("", None), ("เงียบก่อน" + "ก" * 60, None)],
)  # fmt: skip
def test_parse_snooze_command(text, expected):
    assert parse_snooze_command(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [("เดือนนี้จ่ายไปเท่าไหร่", True), ("พรุ่งนี้ประชุมกี่โมง", True), ("ไปกินข้าวกัน", False),
     ("ok", False), ("ใครเบิกค่าน้ำมัน", True), ("นี่ใช่ไหม", True), ("สรุปให้หน่อย", True),
     ("ฮ่าๆๆ", False), ("จ่าย 500 บาท", True), ("what?", True)],
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
    b.gate = SimpleNamespace(window=False, pick="first", window_calls=0, pick_calls=0)

    async def intake(m, quiet=False):
        b.calls.append(("intake-quiet" if quiet else "intake", m.content))

    async def answer(m, guild, who):  # like the real one: reply, then note the answer
        b.calls.append(("answer", m.content))
        b._note_answer(m, await m.reply("ans", mention_author=False))

    def window_gate(client, model, history, line, *, channel_id):
        b.gate.window_calls += 1
        return b.gate.window

    def pick_gate(client, model, lines, ids, *, channel_id):
        b.gate.pick_calls += 1
        return ids[0] if b.gate.pick == "first" else b.gate.pick

    monkeypatch.setattr(botmod.gate, "addressed_to_salid", window_gate)
    monkeypatch.setattr(botmod.gate, "pick_interjection", pick_gate)
    b._intake, b._answer = intake, answer
    return b


def set_mode(bot, channel_id, mode):
    bot.guilds_store.set_channel_mode(GUILD, channel_id, mode)
    bot.team._cache.clear()


def say(world, channel, content="hi", who=STAFF, **kw):
    return TMsg(world.g, channel, member(world.g, who), content, **kw)


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


async def test_listen_answers_nothing_and_reacts_to_a_mention(bot, world):
    set_mode(bot, 10, "listen")
    m = await send(bot, world, world.gen, "ยอดเท่าไร", mention=True)
    n = await send(bot, world, world.gen, "สลิด ยอดเท่าไร")
    assert bot.calls == [] and m.reactions == ["👂"] and not n.reactions and not m.replies


async def test_listen_still_accepts_meeting_commands_by_mention(bot, world, monkeypatch):
    set_mode(bot, 10, "listen")
    joined = []

    async def join(who, fallback_channel=None):
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

    async def attempt():
        await send(bot, world, ch, "ยอดค่าใช้จ่ายเท่าไหร่")
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
    for ch in (world.gen, world.named):
        for _ in range(3):
            await send(bot, world, ch, "เบิกได้ไหม")
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
    assert ack.replies == [botmod.SNOOZE_MESSAGE] and bot.calls == []
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
    assert back.replies == [botmod.UNSNOOZE_MESSAGE]
    await send(bot, world, ch, "สลิด ยอดเท่าไร")
    assert bot.calls == [("answer", "สลิด ยอดเท่าไร")]
    again = await send(bot, world, ch, "<@999> คุยได้แล้ว", mention=True)
    assert again.replies == [botmod.NOT_SNOOZED_MESSAGE]


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
    assert all(label in text for label in botmod.MODE_LABELS.values())


async def test_mode_change_is_refused_for_regular_members(bot, world):
    it = Interaction(world.g, member(world.g, STAFF, roles=["พนักงาน"]), world.gen)
    await bot.handle_mode(it, "join")
    assert it.sent == [(botmod.MODE_DENIED_MESSAGE, True)]
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
    cmds = {c.name: c for c in bot.tree.get_commands()}
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
