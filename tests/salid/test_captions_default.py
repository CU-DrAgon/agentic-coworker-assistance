# ruff: noqa: F811, F401
"""Captions default: the server setting, the start / resume hook, the typed forms and the hub."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from test_captions import tool_msg, until  # noqa: F401
from test_meeting import _bot_guild, env, make_member, start

from salid.captions import captions_tools, parse_captions_default_command
from salid.guild_settings import GuildSettings, GuildSettingsUpdate, MemoryGuildStore
from salid.i18n import t


def set_default(env, value):
    env.bot.guilds_store.save_settings(1000, GuildSettingsUpdate(captions_default=value), "x")
    env.bot.team._cache.pop(1000, None)


@pytest.mark.parametrize(
    "text,want",
    [
        ("แสดงคำบรรยายทุกประชุม", True),
        ("show captions in every meeting", True),
        ("captions on by default", True),
        ("ซ่อนคำบรรยายทุกประชุม", False),
        ("hide captions by default", False),
        ("captions off by default", False),
        ("เปิดซับ", None),
        ("hide captions", None),
        ("show me the report", None),
        ("", None),
    ],
)
def test_typed_forms(text, want):
    assert parse_captions_default_command(text) is want


def test_old_guild_docs_load_with_captions_off():
    assert GuildSettings().captions_default == "off"
    assert GuildSettings(captions_default="nonsense").captions_default == "off"
    store = MemoryGuildStore()
    store.save_settings(7, GuildSettingsUpdate(captions_default="on"), "x")
    assert store.get(7).captions_default == "on"
    store.set_captions_default(7, False)
    assert store.get(7).captions_default == "off"


async def test_default_off_means_no_captions_at_start(env):
    s = await start(env)
    assert not s.captions.on
    await env.mgr.end((1000, 77))


async def test_default_on_turns_captions_on_once_at_start_with_the_suffix(env, monkeypatch):
    set_default(env, "on")
    env.bot.s.voice_state_posts = True
    calls = []
    orig = env.mgr.set_captions
    monkeypatch.setattr(env.mgr, "set_captions", lambda *a: calls.append(a) or orig(*a))
    s = await start(env)
    assert s.captions.on and len(calls) == 1 and calls[0][1] is True
    assert any(t("captions_suffix", s.lang) in m for m in env.chan.sent)
    env.mgr.set_captions(s, False)  # "ซ่อนคำบรรยาย" for this meeting still works
    assert not s.captions.on
    await env.mgr.end((1000, 77))


async def test_no_suffix_when_off_and_none_when_not_recording(env):
    env.bot.s.voice_state_posts = True
    s = await start(env)
    assert not any("📝" in m for m in env.chan.sent)
    await env.mgr.end((1000, 77))
    set_default(env, "on")
    s = env.mgr._captions_from_default
    sess = SimpleNamespace(recording="not_recording", guild_id=1000, captions_defaulted=False)
    assert s(sess) is False  # no notes, no captions


async def test_resume_applies_the_default_once(env, monkeypatch):
    await start(env)
    await env.mgr.shutdown()
    set_default(env, "on")
    env.chan.members = [make_member(env.guild, 10, "แอน", env.chan)]
    env.vc.disconnected = False
    _bot_guild(env, monkeypatch)
    await env.mgr.resume_all()
    new = env.mgr.sessions[(1000, 77)]
    assert new.captions.on
    await env.mgr.end((1000, 77), "command")


def test_default_tool_calls_the_setter_and_is_listed_only_with_one():
    got = []
    assert len(captions_tools(lambda: None, lambda *a: {})) == 1
    tools = captions_tools(
        lambda: None, lambda *a: {}, lambda on: got.append(on) or {"status": "x"}
    )
    assert [f.__name__ for f in tools] == ["set_captions", "set_captions_default"]
    assert tools[1](True) == {"status": "x"} and got == [True]


def fake_message(env, text, user):
    sent = []

    async def reply(content, **kw):
        sent.append(content)

    msg = SimpleNamespace(
        content=text, guild=env.guild, author=user, mentions=[], reply=reply,
        channel=env.chan, channel_mentions=[], role_mentions=[],
    )  # fmt: skip
    return msg, sent


async def test_typed_form_needs_an_executive_and_sets_the_setting(env):
    boss = make_member(env.guild, 5, "เจ้านาย", env.chan)
    boss.guild_permissions = SimpleNamespace(administrator=True, manage_guild=True)
    other = make_member(env.guild, 12, "ซี", env.chan)
    other.guild_permissions = SimpleNamespace(administrator=False, manage_guild=False)
    msg, sent = fake_message(env, "แสดงคำบรรยายทุกประชุม", other)
    assert await env.bot._captions_default_command(msg) is True
    assert sent == [t("captions_default_denied", "th")] and not env.bot.team.captions_default(1000)
    msg, sent = fake_message(env, "แสดงคำบรรยายทุกประชุม", boss)
    assert await env.bot._captions_default_command(msg) is True
    assert sent == [t("captions_default_on", "th")] and env.bot.team.captions_default(1000)
    msg, sent = fake_message(env, "hide captions by default", boss)
    assert await env.bot._captions_default_command(msg) is True
    assert not env.bot.team.captions_default(1000)
    msg, _ = fake_message(env, "สวัสดี", boss)
    assert await env.bot._captions_default_command(msg) is False


def test_hub_html_has_the_toggle_and_both_languages():
    html = (Path(__file__).parents[2] / "src/meeting_scribe/static/hub.html").read_text("utf-8")
    for k in (
        "captions_default_l",
        "captions_default_s",
        'id="captions-default"',
        "captions_default:",
    ):
        assert k in html
    assert "แสดงคำบรรยายสด (ข้อความที่ถอดจากเสียง) ในแชตห้องประชุมทุกครั้ง" in html
    assert "Show live captions (transcribed speech) in the meeting chat by default" in html


# ------------------------------------------------------------------ auto-joined meetings
from test_meeting import segment  # noqa: E402
from test_meeting_behaviour import auto_session  # noqa: E402

from salid import meeting as mt  # noqa: E402


async def test_auto_joined_consent_yes_turns_captions_on_once(env, monkeypatch):
    set_default(env, "on")
    s = await auto_session(env)
    assert s.recording == "unknown" and not s.captions.on
    s.consent, s.consent_until = "pending", mt.time.monotonic() + 30
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "ใช่ จดเลยครับ")
    await env.mgr.handle_segment(s, segment(10))
    assert s.recording == "recording" and s.captions.on and s.captions_defaulted
    env.mgr.set_captions(s, False)  # hidden for this meeting...
    env.mgr.set_recording(s, True)  # ...another transition does not undo it
    s.recording = "unknown"
    s.consent, s.consent_until = "pending", mt.time.monotonic() - 1
    env.mgr._consent_tick(s, mt.time.monotonic())
    assert not s.captions.on
    await env.mgr.end((1000, 77), "shutdown")


async def test_unknown_to_recording_via_set_recording_applies_once(env):
    set_default(env, "on")
    s = await auto_session(env)
    assert not s.captions.on
    env.mgr.set_recording(s, True)
    assert s.captions.on
    env.mgr.set_captions(s, False)
    env.mgr.set_recording(s, False)
    env.mgr.set_recording(s, True)
    assert not s.captions.on
    await env.mgr.end((1000, 77), "shutdown")


async def test_silence_consent_and_default_off_do_nothing_wrong(env):
    s = await auto_session(env)
    s.consent, s.consent_until = "pending", mt.time.monotonic() - 1
    env.mgr._consent_tick(s, mt.time.monotonic())
    assert s.recording == "recording" and not s.captions.on and not s.captions_defaulted
    await env.mgr.end((1000, 77), "shutdown")


async def test_resume_keeps_a_hidden_meeting_hidden(env, monkeypatch):
    set_default(env, "on")
    s = await start(env)
    assert s.captions.on
    env.mgr.set_captions(s, False)
    await env.mgr._persist(s)
    rec = env.bot.secretary_store.get_active_meeting(1000, 77)
    assert rec["captions_hidden"] is True
    await env.mgr.shutdown()
    env.chan.members = [make_member(env.guild, 10, "แอน", env.chan)]
    env.vc.disconnected = False
    _bot_guild(env, monkeypatch)
    await env.mgr.resume_all()
    assert not env.mgr.sessions[(1000, 77)].captions.on
    await env.mgr.end((1000, 77), "command")
