"""English by default, and the main-language question each server answers once: unchosen servers
run on `auto`, the join/startup question with its persistent select menu, who may answer, what
the answer changes, and the settings rules behind it. Discord objects are small fakes."""

from types import SimpleNamespace

import discord
import pytest
from conftest import GUILD, FixedClockSettings
from test_bot_multitenant import ME, FakeChannel, FakeGuild, member

from salid import guild_settings as gs
from salid import language_ask as la
from salid.bot import SecretaryBot
from salid.guild_settings import GuildSettings, MemoryGuildStore
from salid.hints import hint
from salid.i18n import LANGS, locale_lang, resolve_lang, t
from salid.modes import CHAT_CHANNEL_NAME, CHAT_CHANNEL_NAME_EN
from salid.store import SQLiteStore

ADMIN, STAFF = 3, 4


# ------------------------------------------------------------------ fakes
class Chan(FakeChannel):
    def __init__(self, id, name="general"):
        super().__init__(id, name)
        self.views, self.mention, self.renamed = [], f"<#{id}>", []

    async def send(self, text, **kw):
        self.sent.append(text)
        self.views.append(kw.get("view"))

    async def edit(self, **kw):
        self.name = kw["name"]
        self.renamed.append(kw["name"])


class Resp:
    def __init__(self):
        self.calls = []

    async def send_message(self, text, ephemeral=False, **kw):
        self.calls.append(("send", text, ephemeral))

    async def edit_message(self, content=None, view="keep", **kw):
        self.calls.append(("edit", content, view))

    async def defer(self, **kw):
        self.calls.append(("defer",))


def inter(guild, user, values=None, locale="en-US", channel=None):
    return SimpleNamespace(
        guild=guild, user=user, channel=channel, locale=locale, response=Resp(),
        data={"values": values or []}, client=None,
    )  # fmt: skip


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setattr(SecretaryBot, "user", property(lambda s: ME))
    chat = Chan(11, CHAT_CHANNEL_NAME_EN)
    g = FakeGuild(GUILD, "A", [Chan(10), chat], system_channel=None)
    monkeypatch.setattr(SecretaryBot, "guilds", property(lambda s: [g]))
    return SimpleNamespace(g=g, gen=g.text_channels[0], chat=chat)


@pytest.fixture
def bot(world, tmp_path):
    b = SecretaryBot(
        FixedClockSettings(), SQLiteStore(str(tmp_path / "t.db")), None, guilds=MemoryGuildStore()
    )
    b._spawn = lambda coro: coro.close()  # the private-channel rename is not under test
    return b


def stored(bot):
    return bot.guilds_store.get(GUILD)


# ------------------------------------------------------------------ English default, pure parts
@pytest.mark.parametrize(
    ("setting", "text", "expected"),
    [(None, "", "en"), ("auto", "", "en"), ("auto", None, "en"), ("auto", "👍", "en"),
     ("auto", "ยอดเท่าไร", "th"), ("auto", "how much", "en"), ("bogus", "ยอด", "en"),
     ("th", "", "th"), ("ja", "", "ja")],
)  # fmt: skip
def test_resolve_lang_defaults_to_english(setting, text, expected):
    assert resolve_lang(setting, text) == expected


@pytest.mark.parametrize(
    ("locale", "expected"),
    [("", "en"), (None, "en"), ("en-US", "en"), ("en-GB", "en"), ("th", "th"), ("th-TH", "th"),
     ("zh-CN", "zh"), ("zh-TW", "zh"), ("hi", "hi"), ("id", "id"), ("ms", "ms"), ("vi", "vi"),
     ("ko", "ko"), ("ja", "ja"), ("fr", "en"), ("de", "en"), ("auto", "en"), ("xx-YY", "en")],
)  # fmt: skip
def test_locale_lang_table(locale, expected):
    assert locale_lang(locale) == expected


def test_interaction_language_follows_the_setting_then_the_locale(bot, world):
    ch = world.gen
    for locale, expected in [("", "en"), ("fr", "en"), ("th", "th"), ("th-TH", "th"),
                             ("zh-CN", "zh"), ("ja", "ja"), ("ko", "ko")]:  # fmt: skip
        it = SimpleNamespace(guild=world.g, channel=ch, locale=locale)
        assert bot._interaction_lang(it) == expected  # unchosen server: auto, so the locale
    bot.guilds_store.set_server_language(GUILD, "th")
    bot.team._cache.clear()
    assert bot._interaction_lang(SimpleNamespace(guild=world.g, channel=ch, locale="ja")) == "th"


# ------------------------------------------------------------------ unchosen = auto, chosen = fixed
def test_unchosen_server_runs_on_auto_and_chosen_is_fixed(bot):
    assert bot.team.channel_language(GUILD, 10) == "auto"
    assert not bot.team.language_chosen(GUILD)
    bot.guilds_store.set_server_language(GUILD, "ja")
    bot.team._cache.clear()
    assert bot.team.channel_language(GUILD, 10) == "ja" and bot.team.language_chosen(GUILD)
    assert stored(bot).language_chosen and stored(bot).default_language == "ja"


def test_a_document_with_a_default_language_counts_as_chosen_without_a_write():
    store = MemoryGuildStore()
    ref = store._ref(1)
    ref.set(
        {"guild_name": "Old", "default_language": "th"}
    )  # set earlier with /language or the hub
    old = store.get(1)
    assert old.language_chosen and old.language_setting == "th"
    assert "language_chosen" not in ref.get().to_dict()  # migrated when read, not written
    ref2 = store._ref(2)
    ref2.set({"guild_name": "New"})
    assert not store.get(2).language_chosen and store.get(2).language_setting == "auto"
    assert GuildSettings().default_language == "auto" and not GuildSettings().language_chosen


def test_the_hub_echoing_auto_does_not_choose_but_a_real_pick_does():
    store = MemoryGuildStore()
    upd = gs.GuildSettingsUpdate
    store.save_settings(1, upd(default_language="auto", digest_time="19:00"), "a@x.com")
    g = store.get(1)
    assert not g.language_chosen and g.language_setting == "auto" and g.digest_time == "19:00"
    store.save_settings(1, upd(default_language="en"), "a@x.com")
    assert store.get(1).language_chosen and store.get(1).language_setting == "en"


def test_a_channel_language_command_on_an_unchosen_server_no_longer_answers_the_question():
    store = MemoryGuildStore()
    store.set_channel_language(1, 5, "en")
    g = store.get(1)
    # the company language is still to be chosen (salid.audience): only the channel has an override
    assert not g.language_chosen and g.language_setting == "auto"
    assert g.channel_languages == {"5": "en"}
    store.set_channel_language(1, 5, None)  # removing an override changes nothing about the choice
    store.set_channel_language(2, 5, None)
    assert not store.get(2).language_chosen


# ------------------------------------------------------------------ set_server_language restore
def test_switching_back_to_the_default_the_overrides_were_saved_under_restores_them():
    store = MemoryGuildStore()
    store.set_server_language(1, "en")
    store.set_channel_language(1, 5, "th")
    store.set_channel_language(1, 6, "ja")
    store.set_server_language(1, "zh")  # clears the overrides, remembers them under "en"
    g = store.get(1)
    assert g.channel_languages == {} and g.previous_default_language == "en"
    assert g.previous_channel_languages == {"5": "th", "6": "ja"}
    store.set_server_language(1, "th")  # not the language they were saved under: stay cleared
    g = store.get(1)
    assert g.channel_languages == {} and g.previous_channel_languages == {"5": "th", "6": "ja"}
    store.set_server_language(1, "en")  # back to it: restored
    g = store.get(1)
    assert g.channel_languages == {"5": "th", "6": "ja"} and g.previous_channel_languages == {}
    assert g.default_language == "en" and g.previous_default_language == ""


def test_there_is_no_thai_special_case_in_the_restore_rule():
    store = MemoryGuildStore()
    store.set_server_language(1, "th")
    store.set_channel_language(1, 5, "en")
    store.set_server_language(1, "en")  # saved under th
    store.set_server_language(1, "th")  # back to th: restored
    assert store.get(1).channel_languages == {"5": "en"}
    store.set_server_language(1, "en")
    store.set_server_language(1, "ja")  # overrides cleared under en; ja is not en
    assert store.get(1).channel_languages == {}
    # a server that never chose runs on auto, so overrides saved then restore on no pick at all
    fresh = MemoryGuildStore()
    fresh.set_channel_language(2, 5, "th")
    fresh.set_server_language(2, "en")
    assert fresh.get(2).previous_default_language == "auto"


def test_overrides_saved_before_previous_default_language_existed_restore_on_thai():
    store = MemoryGuildStore()
    store._ref(1).set({"default_language": "en", "channel_languages": {},
                       "previous_channel_languages": {"5": "en"}})  # fmt: skip
    store.set_server_language(1, "th")  # old documents were always saved under Thai
    assert store.get(1).channel_languages == {"5": "en"}


# ------------------------------------------------------------------ the menu
def test_the_menu_lists_the_nine_languages_in_their_own_script_and_english():
    select = la.LanguageSelect().item
    assert [o.value for o in select.options] == [
        "en",
        "th",
        "zh",
        "hi",
        "ms",
        "id",
        "vi",
        "ko",
        "ja",
    ]
    assert {o.value for o in select.options} == set(LANGS) - {"auto"}
    labels = {o.value: o.label for o in select.options}
    assert labels["ja"] == "日本語 · Japanese" and labels["th"] == "ไทย · Thai"
    assert select.custom_id == la.CUSTOM_ID == "salid:lang:ask"
    assert "language" in la.QUESTION_FULL and "ภาษา" in la.QUESTION_FULL
    assert isinstance(la.ask_view(), discord.ui.View) and la.ask_view().timeout is None


async def test_the_dynamic_item_rebuilds_from_its_custom_id_and_hands_the_pick_to_the_bot(
    bot, world
):
    item = await la.LanguageSelect.from_custom_id(None, None, None)
    assert isinstance(item, la.LanguageSelect)
    got = []

    async def handle(interaction, value):
        got.append(value)

    it = inter(world.g, None, ["th"])
    it.client = SimpleNamespace(handle_language_pick=handle)
    await item.callback(it)
    assert got == ["th"]


async def test_an_admin_pick_saves_the_language_and_confirms_in_it(bot, world):
    admin = member(world.g, ADMIN, manage_guild=True)
    it = inter(world.g, admin, ["th"], channel=world.chat)
    await bot.handle_language_pick(it, "th")
    g = stored(bot)
    assert g.language_chosen and g.default_language == "th"
    assert bot.team.channel_language(GUILD, 10) == "th"
    kind, text, view = it.response.calls[-1]
    assert kind == "edit" and view is None
    assert text.startswith(t("lang_ask_done", "th", label=t("lang_label_th", "th")))
    assert t("welcome", "th", mention=ME.mention, chat=world.chat.mention, days=90)[:20] in text
    assert hint("welcome_help", "th") in text  # how to ask for help is in the welcome
    # the chat channel Salid created follows the language
    assert world.chat.renamed == [CHAT_CHANNEL_NAME] and world.chat.name == CHAT_CHANNEL_NAME


async def test_a_pick_in_a_second_tier_language_confirms_from_its_catalog(bot, world):
    admin = member(world.g, ADMIN, manage_guild=True)
    it = inter(world.g, admin, ["ja"], channel=world.chat)
    await bot.handle_language_pick(it, "ja")
    assert stored(bot).default_language == "ja"
    kind, text, view = it.response.calls[-1]
    assert (
        kind == "edit"
        and view is None
        and text.startswith(t("lang_ask_done", "ja", label=t("lang_label_ja", "ja")))
    )
    assert world.chat.name == CHAT_CHANNEL_NAME_EN  # not Thai: the English name stays


async def test_a_reused_chat_channel_keeps_its_name(bot, world):
    world.chat.name = f"team-{CHAT_CHANNEL_NAME_EN}"
    admin = member(world.g, ADMIN, manage_guild=True)
    await bot.handle_language_pick(inter(world.g, admin, ["th"]), "th")
    assert world.chat.renamed == [] and world.chat.name == f"team-{CHAT_CHANNEL_NAME_EN}"


async def test_executives_may_pick_too(bot, world):
    bot.s.exec_user_ids = {77}
    boss = member(world.g, 77)
    it = inter(world.g, boss, ["en"])
    await bot.handle_language_pick(it, "en")
    assert stored(bot).language_chosen and it.response.calls[-1][0] == "edit"


async def test_others_get_an_ephemeral_ask_an_admin_line_and_nothing_changes(bot, world):
    staff = member(world.g, STAFF)
    it = inter(world.g, staff, ["th"], channel=world.chat)
    await bot.handle_language_pick(it, "th")
    kind, text, ephemeral = it.response.calls[-1]
    assert (kind, ephemeral) == ("send", True)
    assert t("lang_ask_denied", "en") in text and t("lang_ask_denied", "th") in text
    assert not stored(bot).language_chosen and world.chat.renamed == []


@pytest.mark.parametrize("value", ["", "auto", "klingon", "xx"])
async def test_an_unknown_value_is_ignored(bot, world, value):
    admin = member(world.g, ADMIN, manage_guild=True)
    it = inter(world.g, admin, [value])
    await bot.handle_language_pick(it, value)
    assert it.response.calls == [("defer",)] and not stored(bot).language_chosen


async def test_a_failing_save_says_so_in_both_languages(bot, world, monkeypatch):
    admin = member(world.g, ADMIN, manage_guild=True)

    def boom(*a, **k):
        raise RuntimeError("store down")

    monkeypatch.setattr(bot.team, "set_server_language", boom)
    it = inter(world.g, admin, ["th"])
    await bot.handle_language_pick(it, "th")
    kind, text, ephemeral = it.response.calls[-1]
    assert kind == "send" and ephemeral and t("lang_failed", "th") in text


# ------------------------------------------------------------------ asking
async def test_joining_welcomes_in_english_then_asks_once(bot, world):
    world.g.text_channels[:] = [world.gen, world.chat]
    await bot.on_guild_join(world.g)
    sent = world.chat.sent
    assert sent[0].startswith("Hi, I'm Salid")  # English welcome
    assert hint("welcome_help", "en") in sent[0]
    assert hint("lang_how", "en") in sent[1] and hint("lang_how", "th") in sent[1]
    assert sent[1] == la.QUESTION_FULL and isinstance(world.chat.views[1], discord.ui.View)
    assert len(sent) == 2
    assert stored(bot).language_asked_at and not stored(bot).language_chosen
    assert bot.team.channel_language(GUILD, 10) == "auto"  # until answered


async def test_a_server_that_already_chose_is_not_asked_again_on_rejoin(bot, world):
    bot.guilds_store.set_server_language(GUILD, "th")
    bot.team._cache.clear()
    await bot.on_guild_join(world.g)
    assert len(world.chat.sent) == 1 and world.chat.sent[0].startswith("สวัสดีค่ะ")


async def test_startup_asks_every_unanswered_server_once_in_the_chat_channel(bot, world):
    await bot._ask_unanswered_languages()
    assert world.chat.sent == [la.QUESTION_FULL] and world.gen.sent == []
    assert stored(bot).language_asked_at
    await bot._ask_unanswered_languages()  # a restart, a second ready event
    assert world.chat.sent == [la.QUESTION_FULL]


async def test_startup_falls_back_to_the_system_channel_and_skips_chosen_servers(bot, world):
    world.g.text_channels[:] = [world.gen]
    world.g.system_channel = world.gen
    await bot._ask_unanswered_languages()
    assert world.gen.sent == [la.QUESTION_FULL]
    other = FakeGuild(2, "B", [Chan(30, CHAT_CHANNEL_NAME)])
    bot.guilds_store.set_server_language(2, "en")
    assert await bot._post_language_question(other) is False and other.text_channels[0].sent == []


async def test_without_a_channel_to_write_in_nothing_is_marked_asked(bot, world):
    world.g.text_channels[:] = [world.gen]  # no Salid chat channel, no system channel
    await bot._ask_unanswered_languages()
    assert world.gen.sent == [] and not stored(bot).language_asked_at


async def test_on_ready_asks_after_the_guilds_are_synced(bot, world):
    async def noop(*a, **k):
        return None

    bot._sync_guild, bot._pc_backfill_start = noop, lambda g: None
    bot._start_import = lambda: None
    bot.meetings.resume_all, bot.meetings.scan_all = noop, lambda: None
    await bot.on_ready()
    assert world.chat.sent == [la.QUESTION_FULL]
    await bot.on_ready()
    assert world.chat.sent == [la.QUESTION_FULL]


# ------------------------------------------------------------------ DMs
def test_dm_setting_is_the_persons_language_then_the_chosen_company_language_then_auto(bot, world):
    bot._shared_guilds = lambda uid: [(world.g, None)]
    bot._profile_of = lambda gid, uid: SimpleNamespace(language="")
    dm = SimpleNamespace(author=SimpleNamespace(id=7))
    assert bot._dm_setting(dm) == "auto"  # the company has not chosen
    bot.guilds_store.set_server_language(GUILD, "ja")
    bot.team._cache.clear()
    assert bot._dm_setting(dm) == "ja"
    bot._profile_of = lambda gid, uid: SimpleNamespace(language="th")
    assert bot._dm_setting(dm) == "ja"  # a personal language is added next to it, never instead


# ------------------------------------------------------------------ prompts
@pytest.mark.parametrize(
    ("language", "needle"),
    [("en", "main language is English"), ("th", "ภาษาหลักของเซิร์ฟเวอร์นี้คือภาษาไทย"),
     ("auto", "English when you cannot tell")],
)  # fmt: skip
def test_the_prompt_names_the_main_language_and_allows_switching(language, needle):
    from salid import qa

    s = FixedClockSettings()
    tail = qa.volatile_prompt(s, s.now(), scoped=False, interjection=False, language=language)
    assert needle in tail
    assert "Thai" in tail or "ภาษา" in tail  # may switch when the person writes in / asks for it


def test_the_english_persona_defaults_to_english_but_follows_a_request_to_switch():
    from salid import qa

    en = qa.persona("en")
    assert "reply in English" in en and "explicitly ask" in en
    assert "Switch to Thai" not in en and "writes to you in it" not in en
    assert qa.persona("zzz") == qa.persona("en")  # unknown: English, not Thai
