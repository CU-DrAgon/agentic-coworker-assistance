# ruff: noqa: F811, F401
"""Translation in chat: filters, model calls, the agent tools bound to the asker (reply / link /
last-from, read scope), the context menu, per-channel auto-translate. Offline, fake Discord."""

import asyncio
from types import SimpleNamespace

import discord
import pytest
from conftest import GUILD, FakeClient, fc_resp, json_resp, scripted, text_resp
from test_bot_multitenant import ME, STAFF, FakeChannel, Interaction, Msg, bot, member, world  # noqa

from salid import translate as tr
from salid import translate_bot as tb
from salid.coo_store import SQLiteDocStore
from salid.profiles import Profile

BOSS_ID = 5


# ------------------------------------------------------------------ pure logic
@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("ok", False),
        ("😀😀😀😀😀", False),
        ("https://example.com/a/b/c/d/e", False),
        ("/summary today please now", False),
        ("see you at the office tomorrow", True),
        ("ประชุมพรุ่งนี้เช้าสิบโมงที่ห้องใหญ่", True),
        ("明日の会議は10時です", True),
        ("สวัสดีครับ", False),
        ("<@123> <@456> <@789> <@1>", False),
    ],
)
def test_worth_translating(text, ok):
    assert tr.worth_translating(text) is ok


def test_languages_are_normalised_in_thai_and_english():
    assert tr.normalize_targets("อังกฤษ, Thai") == ["en", "th"]
    assert tr.normalize_targets("English and Japanese") == ["en", "ja"]
    assert tr.normalize_targets(["en", "EN", "Swiss German"]) == ["en", "Swiss German"]
    assert len(tr.normalize_targets("en,th,ja,ko,zh")) == tr.MAX_TARGETS
    assert tr.english_name("ญี่ปุ่น") == "Japanese"


def test_script_language_and_menu_target():
    assert tr.script_language("ประชุมพรุ่งนี้เช้านะ") == "th"
    assert tr.script_language("Please send me the file that you have") == "en"
    assert tr.script_language("こんにちは、明日の会議について") is None
    assert tr.menu_target("en", None) == "th"  # English -> Thai
    assert tr.menu_target("th", None) == "en"  # Thai -> English
    assert tr.menu_target("ja", None) == "en"
    assert tr.menu_target("en", "th") == "th"  # preference wins ...
    assert tr.menu_target("th", "th") == "en"  # ... unless it is already in that language


def test_translations_never_ping():
    assert "@everyone" not in tr.neutralize("hi @everyone and @here")
    assert tr.format_auto({"en": "Hello"}) == "🌐 Hello"
    assert tr.format_auto({"en": "Hi", "th": "หวัดดี"}).startswith("🌐 [EN] Hi\n🌐 [TH]")


def test_rate_limiter():
    now = [0.0]
    rl = tr.RateLimiter(limit=2, window=100, gap=5, clock=lambda: now[0])
    assert rl.allow("c") and not rl.allow("c")  # too soon
    now[0] = 6
    assert rl.allow("c") and rl.allow("other")
    now[0] = 12
    assert not rl.allow("c")  # 2 in the window
    now[0] = 200
    assert rl.allow("c")


def test_model_calls_use_the_given_model_and_parse():
    c = FakeClient([json_resp({"source_language": "TH", "translation": "Hello @everyone"})])
    r = tr.translate_text(c, "m1", "สวัสดีทุกคน", "อังกฤษ")
    assert (r.source_language, "@everyone" not in r.translation) == ("th", True)
    model, contents, _ = c.models.calls[0]
    assert model == "m1" and "English" in contents and "สวัสดีทุกคน" in contents
    c = FakeClient(
        [
            json_resp(
                {
                    "source_language": "th",
                    "translations": [
                        {"language": "English", "text": "See you at 10"},
                        {"language": "Thai", "text": "เจอกัน 10 โมง"},  # same as the source: dropped
                    ],
                }
            )
        ]
    )
    assert tr.auto_translate(c, "lite", "เจอกัน 10 โมง", ["en", "th"]) == {"en": "See you at 10"}


def test_repo_roundtrip_cache_and_off():
    now = [0.0]
    repo = tr.AutoTranslateRepo(SQLiteDocStore(":memory:"), clock=lambda: now[0])
    assert repo.targets(GUILD, 10) == []
    repo.set(GUILD, 10, ["en"], by=1)
    assert repo.targets(GUILD, 10) == ["en"] and repo.targets(GUILD, 11) == []
    assert repo.targets(GUILD + 1, 10) == []  # per guild
    repo.set(GUILD, 10, [], by=1)
    assert repo.targets(GUILD, 10) == []


# ------------------------------------------------------------------ fakes
class Hist(FakeChannel):
    def __init__(self, id, name="general", readable=True):
        super().__init__(id, name)
        self.msgs, self.readable = [], readable

    def permissions_for(self, _):
        return SimpleNamespace(read_messages=self.readable, read_message_history=self.readable)

    async def history(self, limit=100, before=None):
        older = [m for m in self.msgs if before is None or m.id < before.id]
        for m in sorted(older, key=lambda m: -m.id)[:limit]:
            yield m

    async def fetch_message(self, mid):
        return next(m for m in self.msgs if m.id == mid)


class Post(Msg):
    def __init__(self, guild, channel, author, content, id, ref=None):
        super().__init__(guild, channel, author, content)
        self.id, self.reference, self.kw = id, ref, {}
        channel.msgs.append(self)

    async def reply(self, text, **kw):
        self.replies.append(text)
        self.kw = kw


@pytest.fixture
def room(bot, world):
    g = world.a
    ch = Hist(10)
    g.text_channels[:] = [ch]
    ch.guild = g
    g.get_channel_or_thread = lambda cid: ch if cid == ch.id else None
    asker = member(g, STAFF)
    boss = member(g, BOSS_ID, manage_guild=True)
    kenji = member(g, 77)
    kenji.display_name = "Kenji"
    bot.llm = object()

    def fake(text, target):
        return tr.TranslateResult(source_language="ja", translation=f"[{target}] {text}")

    bot._translate_text = fake
    return SimpleNamespace(g=g, ch=ch, asker=asker, boss=boss, kenji=kenji)


def tools_of(bot, room, msg, who=None):
    tools = bot._translate_tools(msg, room.g, who or room.asker, False, asyncio.get_running_loop())
    return {f.__name__: f for f in tools}


@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    def fake(client, model, text, target):
        return tr.TranslateResult(source_language="ja", translation=f"[{target}] {text}")

    monkeypatch.setattr(tb, "translate_text", fake)


async def call(fn, *a, **kw):
    return await asyncio.to_thread(fn, *a, **kw)  # tools run in a worker thread, like in ask()


# ------------------------------------------------------------------ the translate tool
async def test_translate_reply_target_and_default(room, bot):
    old = Post(room.g, room.ch, room.kenji, "明日の会議は10時です", 1)
    ref = SimpleNamespace(resolved=old, message_id=1)
    msg = Post(room.g, room.ch, room.asker, "แปลให้หน่อย", 2, ref)
    t = tools_of(bot, room, msg)
    got = await call(t["translate"], "English")
    assert got["status"] == "ok" and got["author"] == "Kenji"
    assert got["translation"] == "[English] 明日の会議は10時です"
    assert got["target_language"] == "English"


async def test_translate_last_from_name_and_last_from_anyone_else(room, bot):
    Post(room.g, room.ch, room.kenji, "明日の会議は10時です", 1)
    Post(room.g, room.ch, room.asker, "my own message here", 2)
    msg = Post(room.g, room.ch, room.asker, "what did he say?", 3)
    t = tools_of(bot, room, msg)
    assert (await call(t["translate"], "Thai", message="last from kenji"))["author"] == "Kenji"
    assert (await call(t["translate"], "Thai", message="last"))["author"] == "Kenji"  # not me
    assert (await call(t["translate"], "Thai"))["author"] == "Kenji"  # default: no reply -> last
    assert (await call(t["translate"], "Thai", message="last from nobody"))["error"]


async def test_translate_literal_text_and_defaults(room, bot):
    msg = Post(room.g, room.ch, room.asker, "x", 3)
    t = tools_of(bot, room, msg)
    got = await call(t["translate"], "English", text="Bonjour tout le monde")
    assert got["translation"] == "[English] Bonjour tout le monde" and "author" not in got
    got = await call(t["translate"], "", text="Please send the file to me today")
    assert got["target_language"] == "Thai"  # English source -> Thai
    assert (await call(t["translate"], "en", text="😀 😀"))["error"] == "nothing_to_translate"


async def test_translate_link_respects_read_scope(room, bot, world):
    secret = Hist(11, "private", readable=False)
    room.g.text_channels.append(secret)
    secret.guild = room.g
    Post(room.g, room.ch, room.kenji, "公開メッセージです", 1)
    hidden = Post(room.g, secret, room.kenji, "秘密のメッセージです", 2)
    room.g.get_channel_or_thread = lambda cid: {10: room.ch, 11: secret}.get(cid)
    msg = Post(room.g, room.ch, room.asker, "translate this", 9)
    t = tools_of(bot, room, msg)
    ok = await call(t["translate"], "English", message=f"https://discord.com/channels/{GUILD}/10/1")
    assert ok["status"] == "ok"
    bad = await call(
        t["translate"], "English", message=f"https://discord.com/channels/{GUILD}/11/2"
    )
    assert bad == {"error": "cannot_read_that_message"}
    other = await call(t["translate"], "English", message="https://discord.com/channels/5/10/1")
    assert "error" in other  # another server's link
    assert hidden.content not in str(bad)


async def test_translate_unreadable_current_channel_gives_nothing(room, bot):
    room.ch.readable = False
    msg = Post(room.g, room.ch, room.asker, "x", 3)
    t = tools_of(bot, room, msg)
    assert "error" in await call(t["translate"], "English", message="last")


# ------------------------------------------------------------------ set_auto_translate
async def test_only_admins_and_executives_turn_auto_translate_on(room, bot):
    msg = Post(room.g, room.ch, room.asker, "x", 3)
    denied = await call(tools_of(bot, room, msg)["set_auto_translate"], "English")
    assert denied["error"] == "not_allowed" and bot.autotranslate.targets(GUILD, 10) == []
    bmsg = Post(room.g, room.ch, room.boss, "x", 4)
    t = tools_of(bot, room, bmsg, room.boss)
    ok = await call(t["set_auto_translate"], "อังกฤษ, ไทย")
    assert ok["status"] == "ok" and ok["auto_translate"] == ["English", "Thai"]
    assert bot.autotranslate.targets(GUILD, 10) == ["en", "th"]
    assert (await call(t["set_auto_translate"], "off"))["auto_translate"] == "off"
    assert bot.autotranslate.targets(GUILD, 10) == []
    assert (await call(t["set_auto_translate"], "en", channel="nope"))["error"]


def test_tools_ride_the_one_agent_builder_and_keep_the_coo_prompt_out(bot, room):
    msg = Post(room.g, room.ch, room.asker, "x", 3)

    async def build():
        return bot._agent_kwargs(
            msg, room.g, room.asker, False, setting="th", lang="th", question="q",
            memory=None, profile=Profile(),
        )  # fmt: skip

    kw = asyncio.run(build())
    names = [f.__name__ for f in kw["extra_tools"]]
    assert names == ["translate", "set_auto_translate"]
    from salid.qa import COO_PROMPT, ask

    llm = scripted(text_resp("ok"))
    ask(bot.s, bot.store, "q", guild_id=GUILD, is_exec=False, user_id=1, model=llm, **kw)
    sent = llm.requests[0]
    assert "translate" in sent.tools_dict
    assert COO_PROMPT not in str(sent.config.system_instruction)


# ------------------------------------------------------------------ context menu
async def test_menu_translates_into_the_clickers_language(room, bot):
    jp = Post(room.g, room.ch, room.kenji, "明日の会議は10時です", 1)
    en = Post(room.g, room.ch, room.kenji, "The meeting starts at ten tomorrow", 2)
    th = Post(room.g, room.ch, room.kenji, "ประชุมพรุ่งนี้สิบโมงเช้า", 3)
    it = Interaction(room.g, room.asker)
    it.locale = "th"
    it.channel = room.ch
    it.followup = SimpleNamespace(send=lambda text, **kw: _record(it, text, kw))
    it.response = SimpleNamespace(defer=_noop)
    await bot.handle_translate_menu(it, en)  # English source -> Thai
    await bot.handle_translate_menu(it, th)  # Thai source -> English
    await bot.handle_translate_menu(it, jp)  # other -> the channel language
    targets = [s[0].split("[")[1].split("]")[0] for s in it.sent]
    assert targets == ["th", "en", "th"]  # other language: the channel's (Thai)
    assert all(s[1] and s[2] for s in it.sent)  # ephemeral, no pings
    bot.profiles.save(GUILD, room.asker.id, Profile(language="en"))
    it.sent.clear()
    await bot.handle_translate_menu(it, jp)
    assert "[en]" in it.sent[0][0]  # profile language wins over the channel's


async def _noop(**kw):
    return None


async def _record(it, text, kw):
    it.sent.append((text, kw.get("ephemeral"), kw.get("allowed_mentions") is not None))


async def test_menu_with_no_text(room, bot):
    empty = Post(room.g, room.ch, room.kenji, "  ", 1)
    it = Interaction(room.g, room.asker)
    it.locale = "en"
    it.channel = room.ch
    it.followup = SimpleNamespace(send=lambda text, **kw: _record(it, text, kw))
    it.response = SimpleNamespace(defer=_noop)
    await bot.handle_translate_menu(it, empty)
    assert "no text" in it.sent[0][0].lower() or "ไม่มี" in it.sent[0][0]


def test_menu_is_registered_with_a_thai_name(bot):
    bot._register_commands()
    menus = {c.name for c in bot.tree.get_commands(type=discord.AppCommandType.message)}
    assert {"Summarize with Salid", "Translate with Salid"} <= menus


# ------------------------------------------------------------------ auto-translate
@pytest.fixture
def auto(room, bot, monkeypatch):
    calls = []

    def fake(client, model, text, targets):
        calls.append((model, text, targets))
        return {targets[0]: "Meeting is at ten tomorrow @everyone"}

    monkeypatch.setattr(tb, "auto_translate", fake)
    bot.autotranslate.set(GUILD, 10, ["en"], by=1)
    bot._auto_limit = tr.RateLimiter(limit=3, window=100, gap=0)
    return calls


async def settle(bot):
    await asyncio.gather(*list(bot._bg_tasks))


async def test_auto_translate_replies_compactly_without_pings(room, bot, auto):
    m = Post(room.g, room.ch, room.kenji, "明日の会議は10時です。資料を確認してください", 1)
    await bot._autotranslate_maybe(m)
    await settle(bot)
    assert auto and auto[0][0] == bot.s.gate_model and auto[0][2] == ["en"]
    assert m.replies[0].startswith("🌐 Meeting is at ten tomorrow")
    assert "@everyone" not in m.replies[0]
    assert m.kw["allowed_mentions"].everyone is False and m.kw["allowed_mentions"].users is False
    assert m.kw["suppress_embeds"] is True and m.kw["mention_author"] is False


@pytest.mark.parametrize(
    "text",
    [
        "ok thanks",  # < 4 words
        "😀😀😀😀😀😀",
        "https://example.com/some/long/link/here",
        "/summary today for the team please",
        "Please send me the report that you have today",  # already English
    ],
)
async def test_auto_translate_skips(room, bot, auto, text):
    m = Post(room.g, room.ch, room.kenji, text, 1)
    await bot._autotranslate_maybe(m)
    await settle(bot)
    assert not auto and not m.replies


async def test_auto_translate_skips_bots_other_channels_and_when_off(room, bot, auto):
    jp = "明日の会議は10時です。資料を確認してください"
    robot = member(room.g, 300)
    robot.bot = True
    m = Post(room.g, room.ch, robot, jp, 1)
    await bot._autotranslate_maybe(m)
    elsewhere = Hist(12)
    elsewhere.guild = room.g
    m2 = Post(room.g, elsewhere, room.kenji, jp, 2)
    await bot._autotranslate_maybe(m2)
    to_salid = Post(room.g, room.ch, room.kenji, jp, 3)
    to_salid.mentions = [ME]
    await bot._autotranslate_maybe(to_salid)
    bot.autotranslate.set(GUILD, 10, [], by=1)
    m3 = Post(room.g, room.ch, room.kenji, jp, 4)
    await bot._autotranslate_maybe(m3)
    await settle(bot)
    assert not auto


async def test_auto_translate_is_rate_limited_per_channel(room, bot, auto):
    jp = "明日の会議は10時です。資料を確認してください"
    posts = [Post(room.g, room.ch, room.kenji, jp, i) for i in range(1, 6)]
    for p in posts:
        await bot._autotranslate_maybe(p)
    await settle(bot)
    assert len(auto) == 3


async def test_auto_translate_nothing_to_say_stays_quiet(room, bot, auto, monkeypatch):
    monkeypatch.setattr(tb, "auto_translate", lambda *a: {})
    m = Post(room.g, room.ch, room.kenji, "Bonjour tout le monde, la réunion est à dix heures", 1)
    await bot._autotranslate_maybe(m)
    await settle(bot)
    assert not m.replies


async def test_on_message_hook_runs_for_humans(room, bot, auto):
    bot._answer = None
    seen = []

    async def hook(m):
        seen.append(m.id)

    bot._autotranslate_maybe = hook
    await bot.on_message(Post(room.g, room.ch, room.kenji, "hello there everyone", 1))
    assert seen == [1]
