# ruff: noqa: F811, F401
"""One company language, personal languages added for the people involved (salid.audience)."""

import asyncio
from types import SimpleNamespace

import pytest
from conftest import GUILD, FixedClockSettings
from test_coordinator import BOSS, STAFF, bot, world  # noqa: F401  (fixtures)
from test_meeting import PCM, FakeGuild, env, make_member, start, voice  # noqa: F401
from test_task_handshake import Click, hs, new_task, set_settings  # noqa: F401  (fixtures)

from salid import audience as aud
from salid import live_voice as lvm
from salid import meeting as mt
from salid import qa
from salid.audience_bot import AudienceMixin
from salid.guild_settings import (
    GuildSettings,
    GuildSettingsUpdate,
    MemoryGuildStore,
)
from salid.i18n import CATALOG, t
from salid.profiles import Profile


@pytest.fixture(autouse=True)
def _notices_on(monkeypatch):
    monkeypatch.setattr(aud, "MODE_NOTICES", True)


# ------------------------------------------------------------------ the pure language set
def test_langs_for_puts_the_company_first_then_distinct_personal_ones():
    assert aud.langs_for("en") == ["en"]
    assert aud.langs_for("en", "th") == ["en", "th"]
    assert aud.langs_for("en", "en", "") == ["en"]  # same as the company: nothing added
    assert aud.langs_for("en", None, "th", "th") == ["en", "th"]  # deduped
    assert aud.langs_for("th", "en", "ja") == ["th", "en", "ja"]


def test_langs_for_caps_the_audience_at_three_and_honours_the_limit():
    assert aud.langs_for("en", "th", "ja", "ko", "zh") == ["en", "th", "ja"]
    assert aud.langs_for("en", "th", "ja", limit=2) == ["en", "th"]


def test_an_unchosen_server_keeps_the_auto_path_whatever_the_people_prefer():
    assert aud.langs_for("auto", "th") == ["auto"]
    assert aud.langs_for("zzz", "th") == ["auto"]


def test_company_only_ignores_personal_languages():
    assert aud.langs_for("en", "th", "ja", mode="company_only") == ["en"]
    assert aud.voice_langs("th", ["en", "ja"], mode="company_only") == ["th"]


def test_company_and_person_language_readers():
    chosen = GuildSettings(default_language="en", language_chosen=True)
    assert aud.company_lang(chosen) == "en"
    assert aud.company_lang(GuildSettings()) == "auto"
    assert aud.person_lang(Profile(language="th")) == "th"
    assert aud.person_lang(Profile()) == "" and aud.person_lang("auto") == ""
    assert aud.person_lang(Profile(language="th"), allowed=False) == ""


def test_voice_langs_take_profiles_or_plain_codes():
    assert aud.voice_langs("en", [Profile(language="th"), Profile(), "th"]) == ["en", "th"]
    assert aud.voice_langs("en", []) == ["en"]


def test_tt_renders_each_language_and_dedupes_identical_ones():
    both = aud.tt("aud_personal_unset", ["en", "th"])
    assert both == t("aud_personal_unset", "en") + "\n\n" + t("aud_personal_unset", "th")
    assert aud.tt("aud_personal_unset", ["en", "en"]) == t("aud_personal_unset", "en")
    # a key that reads the same in every language is shown once
    CATALOG["aud_test_same"] = {"th": "OK", "en": "OK"}
    assert aud.tt("aud_test_same", ["en", "th"]) == "OK"
    # a callable value is asked for its text per language
    text = aud.tt("aud_personal_set", ["en", "th"], lname=aud.name_in("th"))
    assert "Thai" in text and "ภาษาไทย" in text


def test_join_blocks_and_render_keep_the_audience_order():
    assert aud.join_blocks({"th": "ก", "en": "A"}, ["en", "th"]) == "A\n\nก"
    assert aud.join_blocks({"en": "A"}, ["en", "th"]) == "A"
    assert aud.join_blocks({"en": "A"}, ["ja"]) == "A"  # nothing matches: the first block
    assert aud.render(lambda lg: lg.upper(), ["en", "th", "en"]) == "EN\n\nTH"


# ------------------------------------------------------------------ the mode statements
def test_the_three_greeting_modes():
    assert aud.mode_line(["en"]) == "Hi, I'll take notes and answer in English."
    assert aud.mode_line(["th"]) == "สวัสดีค่ะ สลิดจะจดและตอบเป็นภาษาไทยนะคะ"
    mixed = aud.mode_line(["en", "th"])
    assert mixed.startswith(
        "Hi everyone. Someone here prefers Thai, so I'll answer in English and then in Thai."
    )
    assert "สวัสดีค่ะทุกคน" in mixed and "ภาษาอังกฤษ" in mixed  # the Thai sentence follows


def test_company_only_mode_line_says_so():
    assert aud.mode_line(["en"], company_only=True) == "I'll answer in English only in this room."
    assert aud.mode_line(["th"], company_only=True) == "ห้องนี้ตอบภาษาไทยอย่างเดียวค่ะ"


def test_the_hint_says_how_to_change_it_and_the_adding_line_is_bilingual():
    assert "talk to me in Thai" in aud.mode_line_with_hint(["en"])
    line = aud.adding_line(["th"])
    assert line == "adding Thai from now on / เพิ่มภาษาไทยตั้งแต่นี้ไป"


def test_second_language_rule_only_with_two_or_more_languages():
    assert aud.second_language_rule(["en"]) == "" and aud.second_language_rule(["auto"]) == ""
    rule = aud.second_language_rule(["en", "th"])
    assert "English" in rule and "Thai" in rule and "key points" in rule
    assert "ภาษาอังกฤษ" in aud.second_language_rule(["en", "th"], "th")


def test_split_blocks_gives_each_paragraph_its_language():
    pieces = aud.split_blocks("Done.\n\nเรียบร้อยแล้วค่ะ", ["en", "th"])
    assert pieces == [("en", "Done."), ("th", "เรียบร้อยแล้วค่ะ")]
    assert aud.split_blocks("one\n\ntwo", ["en", "th"]) == [("en", "one\n\ntwo")]


# ------------------------------------------------------------------ typed forms
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("talk to me in Thai", ("set", "th")),
        ("Salid, talk to me in Japanese please", ("set", "ja")),
        ("my language is Thai", ("set", "th")),
        ("สลิด คุยกับฉันเป็นภาษาไทย", ("set", "th")),
        ("คุยกับผมเป็นภาษาอังกฤษนะ", ("set", "en")),
        ("use the company language", ("unset", None)),
        ("ใช้ภาษาบริษัท", ("unset", None)),
        ("ใช้ภาษาบริษัทอย่างเดียว", None),  # that one is the executive's policy
        ("talk to me in Klingon", None),
        ("what is the company language of Acme", None),
    ],
)
def test_parse_personal_language(text, expected):
    assert aud.parse_personal_language(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("company language only", "company_only"),
        ("English only", "company_only"),
        ("ใช้ภาษาบริษัทอย่างเดียว", "company_only"),
        ("ใช้ภาษาเดียว", "company_only"),
        ("allow personal languages", "audience"),
        ("ตอบหลายภาษาได้", "audience"),
        ("English only in this meeting", None),  # the per-meeting form
        ("how are you", None),
    ],
)
def test_parse_language_mode(text, expected):
    assert aud.parse_language_mode(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Salid, English only in this meeting", "company_only"),
        ("สลิด ประชุมนี้ใช้ภาษาอังกฤษอย่างเดียว", "company_only"),
        ("both languages in this meeting", "audience"),
        ("ประชุมนี้ใช้สองภาษา", "audience"),
        ("English only", None),
    ],
)
def test_parse_meeting_mode(text, expected):
    assert aud.parse_meeting_mode(text) == expected


# ------------------------------------------------------------------ the model's prompts
def _tail(language, languages=None):
    s = FixedClockSettings()
    return qa.volatile_prompt(
        s, s.now(), scoped=False, interjection=False, language=language, languages=languages
    )


@pytest.mark.parametrize("language", ["en", "th"])
def test_persona_prompts_have_no_switch_to_the_users_language_clause(language):
    prompt = qa.persona(language) + _tail(language)
    assert "unless the person writes" not in prompt
    assert "Switch to Thai, or another language, only when" not in prompt
    assert "เว้นแต่ผู้ใช้พิมพ์มาเป็นภาษาอังกฤษ" not in prompt
    assert "explicit" in prompt or "ชัดเจน" in prompt  # only an explicit request switches


def test_the_second_language_rule_is_in_the_prompt_only_for_two_or_more_languages():
    one = _tail("en", ["en"])
    two = _tail("en", ["en", "th"])
    assert "repeat it in Thai" in two and "repeat it in" not in one
    assert "repeat it in" not in _tail("en", None)
    th = _tail("th", ["th", "en"])
    assert "ตอบซ้ำเป็นภาษาอังกฤษ" in th


def test_the_live_instruction_carries_the_rule_and_the_mode_line_once():
    s = FixedClockSettings()
    en = lvm.build_instruction(s, "en", lang="en", langs=["en", "th"])
    assert "repeat it in Thai" in en and "Someone here prefers Thai" in en
    assert "never switch language because someone speaks" in en
    th = lvm.build_instruction(s, "th", lang="th", langs=["th", "en"])
    assert "ตอบซ้ำเป็นภาษาอังกฤษ" in th and "ห้ามเปลี่ยนภาษาเอง" in th
    one = lvm.build_instruction(s, "en", lang="en", langs=["en"])
    assert "repeat it in" not in one and "I'll take notes and answer in English." in one
    only = lvm.build_instruction(s, "en", lang="en", langs=["en"], company_only=True)
    assert "I'll answer in English only in this room." in only
    plain = lvm.build_instruction(s, "en", lang="en")  # an unchosen server: as before
    assert "take notes and answer in" not in plain


# ------------------------------------------------------------------ settings
def test_language_mode_is_stored_and_validated():
    store = MemoryGuildStore()
    assert store.get(1).language_mode == "audience"
    store.set_language_mode(1, "company_only")
    assert store.get(1).language_mode == "company_only"
    with pytest.raises(ValueError):
        store.set_language_mode(1, "everything")
    assert GuildSettingsUpdate(language_mode="nonsense").language_mode == "audience"
    store.save_settings(1, GuildSettingsUpdate(language_mode="company_only"), "a@x.com")
    assert store.get(1).language_mode == "company_only"  # the hub round trip


# ------------------------------------------------------------------ resolution on the bot
class FakeBot(AudienceMixin):
    """The resolution parts of SecretaryBot over a real guild store and profiles."""

    def __init__(self, store, profiles=None, allowed=True):
        from salid.guild_settings import BotGuildSettings

        self.team = BotGuildSettings(store, ttl=0)
        self.langs = profiles or {}
        self.allowed = allowed

    def _profile_of(self, gid, uid):
        return Profile(language=self.langs.get(uid) if self.allowed else None)

    def _person_lang(self, gid, uid, text=None):
        from salid.i18n import resolve_lang

        return resolve_lang(self.team.current(gid).language_setting, text)


def _chosen(language="en", **kw):
    store = MemoryGuildStore()
    store.save_settings(1, GuildSettingsUpdate(default_language=language, **kw), "a@x.com")
    return store


def test_a_chosen_english_server_stays_english_for_a_thai_message():
    bot = FakeBot(_chosen("en"))
    assert bot.langs_for(1, 5, 7) == ["en"]
    assert bot.team.channel_language(1, 5) == "en"  # the words people write decide nothing


def test_the_asker_s_personal_language_is_added_not_substituted():
    bot = FakeBot(_chosen("en"), {7: "th"})
    assert bot.langs_for(1, 5, 7) == ["en", "th"]
    assert bot.langs_for(1, None, 7, 8) == ["en", "th"]  # 8 has none
    assert bot.langs_for(1, 5, 8) == ["en"]


def test_personal_settings_off_and_company_only_both_drop_the_personal_language():
    assert FakeBot(_chosen("en"), {7: "th"}, allowed=False).langs_for(1, 5, 7) == ["en"]
    only = _chosen("en", language_mode="company_only")
    assert FakeBot(only, {7: "th"}).langs_for(1, 5, 7) == ["en"]


def test_an_auto_channel_override_is_ignored_once_chosen_but_kept_when_unchosen():
    store = _chosen("en")
    store.set_channel_language(1, 5, "auto")
    bot = FakeBot(store)
    assert bot.team.channel_language(1, 5) == "en"
    unchosen = MemoryGuildStore()
    unchosen.set_channel_language(1, 5, "auto")
    ub = FakeBot(unchosen, {7: "th"})
    assert ub.langs_for(1, 5, 7) == ["auto"]  # today's path: follow the words
    assert ub.team.channel_language(1, 5) == "auto"


def test_a_channel_command_on_an_unchosen_server_does_not_choose_for_it():
    store = MemoryGuildStore()
    store.set_channel_language(1, 5, "en")
    assert not store.get(1).language_chosen


def test_restoring_the_previous_default_drops_the_old_auto_overrides():
    store = _chosen("th")
    store.set_channel_language(1, 10, "auto")
    store.set_channel_language(1, 11, "en")
    store.set_server_language(1, "en", True)
    store.set_server_language(1, "th", True)
    assert store.get(1).channel_languages == {"11": "en"}


# ------------------------------------------------------------------ personal commands
def _cmd_bot(store, allowed=True):
    bot = FakeBot(store, allowed=allowed)
    saved = {}

    class Repo:
        def get(self, gid, uid):
            return saved.get(uid, Profile())

        def save(self, gid, uid, p):
            saved[uid] = p

    bot.profiles = Repo()
    bot._personal_allowed = lambda gid: allowed
    bot.saved = saved
    return bot


def test_set_and_unset_the_personal_language_confirms_in_both_languages():
    bot = _cmd_bot(_chosen("en"))
    reply = bot._set_personal_language(1, 7, ("set", "th"), ["en"])
    assert bot.saved[7].language == "th"
    assert "Thai" in reply and "ภาษาไทย" in reply  # the company language, then theirs
    reply = bot._set_personal_language(1, 7, ("unset", None), ["en"])
    assert bot.saved[7].language is None and "company language" in reply


def test_a_personal_language_equal_to_the_company_one_is_not_stored():
    bot = _cmd_bot(_chosen("en"))
    assert "already the company language" in bot._set_personal_language(1, 7, ("set", "en"), ["en"])
    assert bot.saved[7].language is None


def test_personal_command_is_refused_when_the_org_turned_personal_settings_off():
    bot = _cmd_bot(_chosen("en"), allowed=False)
    reply = bot._set_personal_language(1, 7, ("set", "th"), ["en"])
    assert "turned personal settings off" in reply and 7 not in bot.saved


async def test_org_language_mode_is_for_executives_only():
    store = _chosen("en")
    bot = _cmd_bot(store)
    bot.is_exec_member = lambda m: m.id == 1
    denied = await bot._set_org_mode(SimpleNamespace(id=2), 1, "company_only", ["en"])
    assert "Only an executive" in denied and store.get(1).language_mode == "audience"
    ok = await bot._set_org_mode(SimpleNamespace(id=1), 1, "company_only", ["en"])
    assert "company language (English)" in ok and store.get(1).language_mode == "company_only"
    bot.team._cache.clear()
    assert bot.langs_for(1, 5, 7) == ["en"]


def test_the_agent_tool_sets_the_org_mode_for_executives_and_refuses_others():
    store = _chosen("en")
    bot = _cmd_bot(store)
    msg = SimpleNamespace(channel=SimpleNamespace(id=5))
    guild = SimpleNamespace(id=1)
    (tool,) = bot._audience_tools(msg, guild, SimpleNamespace(id=2), False, "en")
    assert tool("company_only")["error"] == "not_executive"
    (tool,) = bot._audience_tools(msg, guild, SimpleNamespace(id=1), True, "en")
    assert tool("company_only")["status"] == "ok"
    assert store.get(1).language_mode == "company_only"
    assert tool("nonsense")["error"]


# ------------------------------------------------------------------ voice meetings
def _english_room(env, people):
    """English company language; `people` = {id: personal language or None} are in the room."""
    env.bot.team.set_server_language(1000, "en")
    for uid, lang in people.items():
        m = make_member(env.guild, uid, f"P{uid}", env.chan)
        env.chan.members.append(m)
        if lang:
            env.bot.profiles.save(1000, uid, Profile(language=lang))
    return env.guild.members[next(iter(people))]


async def test_joining_an_english_room_announces_english_only(env):
    env.bot.s.voice_state_posts = True
    starter = _english_room(env, {10: None})
    await env.mgr.join(starter, lang="en")
    s = env.mgr.sessions[(1000, 77)]
    assert s.langs == ["en"] and s.starter_id == 10
    assert any(x.endswith("Hi, I'll take notes and answer in English.") for x in env.chan.sent)


async def test_joining_a_thai_company_room_announces_thai(env):
    env.bot.s.voice_state_posts = True
    m = make_member(env.guild, 10, "แอน", env.chan)
    env.chan.members.append(m)
    await env.mgr.join(m, lang="th")  # the test store is a Thai company
    assert env.mgr.sessions[(1000, 77)].langs == ["th"]
    assert any(x.endswith("สวัสดีค่ะ สลิดจะจดและตอบเป็นภาษาไทยนะคะ") for x in env.chan.sent)


async def test_a_mixed_room_announces_english_then_thai(env):
    env.bot.s.voice_state_posts = True
    starter = _english_room(env, {10: None, 11: "th"})
    await env.mgr.join(starter, lang="en")
    s = env.mgr.sessions[(1000, 77)]
    assert s.langs == ["en", "th"]
    (line,) = [x for x in env.chan.sent if "Someone here prefers" in x]
    assert "Someone here prefers Thai, so I'll answer in English and then in Thai." in line
    assert "สวัสดีค่ะทุกคน" in line


async def test_someone_joining_mid_meeting_adds_the_language_once_as_a_text_line(env):
    starter = _english_room(env, {10: None})
    await env.mgr.join(starter, lang="en")
    s = env.mgr.sessions[(1000, 77)]
    env.chan.sent.clear()
    late = make_member(env.guild, 12, "Late", env.chan)
    env.chan.members.append(late)
    env.bot.profiles.save(1000, 12, Profile(language="th"))
    before = SimpleNamespace(channel=None)
    after = SimpleNamespace(channel=env.chan)
    await env.mgr.on_voice_activity(late, before, after)
    assert s.langs == ["en", "th"]
    assert env.chan.sent.count("adding Thai from now on / เพิ่มภาษาไทยตั้งแต่นี้ไป") == 1
    another = make_member(env.guild, 13, "More", env.chan)
    env.chan.members.append(another)
    env.bot.profiles.save(1000, 13, Profile(language="th"))
    await env.mgr.on_voice_activity(another, before, after)  # Thai is already there
    assert env.chan.sent.count("adding Thai from now on / เพิ่มภาษาไทยตั้งแต่นี้ไป") == 1


async def test_company_only_policy_keeps_the_room_in_the_company_language(env):
    starter = _english_room(env, {10: None, 11: "th"})
    env.bot.team.set_language_mode(1000, "company_only")
    await env.mgr.join(starter, lang="en")
    assert env.mgr.sessions[(1000, 77)].langs == ["en"]


async def test_the_chair_can_force_one_language_for_one_meeting(env):
    starter = _english_room(env, {10: None, 11: "th"})
    await env.mgr.join(starter, lang="en")
    s = env.mgr.sessions[(1000, 77)]
    assert s.langs == ["en", "th"]
    env.bot.meetings = env.mgr
    env.bot.is_exec_member = lambda m: m.id == 5
    bot = env.bot
    chair = bot._set_meeting_mode(starter, 1000, "company_only", ["en"])
    assert "English only" in chair and s.langs == ["en"] and s.language_mode == "company_only"
    await asyncio.sleep(0)
    assert any("I'll answer in English only in this room." in x for x in env.chan.sent)
    other = env.guild.members[11]
    refusal = bot._set_meeting_mode(other, 1000, "audience", ["en"])
    assert "Only whoever invited Salid" in refusal and s.language_mode == "company_only"
    bot.meetings.set_language_mode(s, "audience")
    assert s.langs == ["en", "th"]


async def test_classic_speak_uses_each_languages_voice_and_says_the_mode_once(voice):
    voice.bot.team.set_server_language(1000, "en")
    for uid, lang in ((10, None), (11, "th")):
        m = make_member(voice.guild, uid, f"P{uid}", voice.chan)
        voice.chan.members.append(m)
        if lang:
            voice.bot.profiles.save(1000, uid, Profile(language=lang))
    s = await start(voice)
    assert s.langs == ["en", "th"]
    task = asyncio.create_task(voice.mgr.speak(s, "Done.\n\nเรียบร้อยแล้วค่ะ", "q"))
    for _ in range(80):
        await asyncio.sleep(0.01)
        if voice.vc.played:
            break
    voice.vc.playing = False
    voice.vc.after(None)
    await task
    voices = [(c[1], c[2]) for c in voice.calls if c[0] == "tts"]
    langs = [v[0] for v in voices]
    # the mode line in both languages, then the answer: English voice first, then the Thai voice
    assert langs == ["en", "th", "en", "th"]
    assert s.mode_spoken is True
    voice.calls.clear()
    voice.vc.played.clear()
    voice.vc.playing = False
    task = asyncio.create_task(voice.mgr.speak(s, "Done.\n\nเรียบร้อยแล้วค่ะ", "q"))
    for _ in range(80):
        await asyncio.sleep(0.01)
        if voice.vc.played:
            break
    voice.vc.playing = False
    voice.vc.after(None)
    await task
    assert [c[1] for c in voice.calls if c[0] == "tts"] == ["en", "th"]  # no mode line again
    await voice.mgr.end((1000, 77))


async def test_the_summary_stays_in_the_company_language(env, monkeypatch):
    starter = _english_room(env, {10: None, 11: "th"})
    await env.mgr.join(starter, lang="en")
    s = env.mgr.sessions[(1000, 77)]
    assert s.lang == "en"  # minutes, summary and PDF follow the session's company language
    s.language_mode = "audience"
    assert s.lang == "en" and s.language == "en"


def test_pcm_fixture_is_bytes():
    assert isinstance(PCM, bytes)
    assert isinstance(FakeGuild, type) and mt.MIN_SEGMENT_S > 0


# ------------------------------------------------------------------ task cards and giver lines
async def test_a_card_to_a_thai_preferring_assignee_on_an_english_server_has_both_blocks(hs, world):
    set_settings(hs, default_language="en")
    hs.profiles.save(GUILD, STAFF, Profile(language="th"))
    task = new_task(hs)
    await hs._handshake_offer(GUILD, [task])
    ((card, _),) = world.staff.dms
    en_block = hs._handshake_text(world.g, [hs.tasks.get(GUILD, task.id)], "en", first_time=True)
    th_block = hs._handshake_text(world.g, [hs.tasks.get(GUILD, task.id)], "th", first_time=True)
    assert card == en_block + "\n\n" + th_block  # company language first, then theirs
    assert en_block != th_block


async def test_a_card_to_a_same_language_assignee_is_one_block(hs, world):
    set_settings(hs, default_language="en")
    hs.profiles.save(GUILD, STAFF, Profile(language="en"))
    task = new_task(hs)
    await hs._handshake_offer(GUILD, [task])
    ((card, _),) = world.staff.dms
    assert card.count(task.title) == 1


async def test_the_giver_line_is_english_then_the_givers_language(hs, world):
    set_settings(hs, default_language="en")
    hs.profiles.save(GUILD, BOSS, Profile(language="th"))
    task = new_task(hs)
    await hs._handshake_offer(GUILD, [task])
    click = Click(world.staff)
    await hs.handle_task_answer(click, "ok", GUILD, task.id)
    ((line, _),) = world.boss.dms
    en = t("hs_giver_ok", "en", name="Somchai", title=task.title, due="2026-10-16", id=task.id)
    th = t("hs_giver_ok", "th", name="Somchai", title=task.title, due="2026-10-16", id=task.id)
    assert line.startswith(en) and th in line and line.index(en) < line.index(th)


async def test_personal_settings_off_gives_everyone_the_company_language_only(hs, world):
    set_settings(hs, default_language="en", personal_settings=False)
    hs.profiles.save(GUILD, STAFF, Profile(language="th"))
    task = new_task(hs)
    await hs._handshake_offer(GUILD, [task])
    ((card, _),) = world.staff.dms
    assert card.count(task.title) == 1
