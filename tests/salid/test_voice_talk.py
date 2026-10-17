# ruff: noqa: F811, F401
"""Salid's voice in meetings: the human listen-only / may-talk switch (never an agent tool)."""

import asyncio
from types import SimpleNamespace

import pytest
from test_captions import until  # noqa: F401
from test_captions_default import fake_message
from test_live_voice import lv
from test_meeting import PCM, fake_ask, make_member, segment, start, voice
from test_meeting import env as env  # noqa: F401

from salid import meeting as mt
from salid.guild_settings import GuildSettings, GuildSettingsUpdate
from salid.i18n import t
from salid.meeting import parse_voice_talk_command, parse_voice_talk_default_command


@pytest.mark.parametrize(
    "text,want",
    [
        ("สลิด ฟังอย่างเดียว", "listen_only"),
        ("ฟังอย่างเดียวนะครับ", "listen_only"),
        ("Salid, listen only", "listen_only"),
        ("listen only please.", "listen_only"),
        ("สลิด พูดได้", "may_talk"),
        ("พูดได้แล้วค่ะ", "may_talk"),
        ("Salid, you may talk", "may_talk"),
        ("you can talk", "may_talk"),
        ("talk mode", "may_talk"),
        ("พูดได้ไหม", None),
        ("can you talk about the budget", None),
        ("ฟังอย่างเดียวก็ได้ แต่ขอถามหน่อย", None),
        ("listen only by default", None),
        ("", None),
    ],
)
def test_parse_voice_talk_command(text, want):
    assert parse_voice_talk_command(text) == want


@pytest.mark.parametrize(
    "text,want",
    [
        ("ประชุมทุกครั้งฟังอย่างเดียว", "listen_only"),
        ("listen only by default", "listen_only"),
        ("พูดได้ทุกประชุม", "may_talk"),
        ("may talk by default", "may_talk"),
        ("listen only", None),
        ("พูดได้", None),
    ],
)
def test_parse_voice_talk_default_command(text, want):
    assert parse_voice_talk_default_command(text) == want


def test_setting_default_and_validation():
    assert GuildSettings().voice_talk == "may_talk"
    assert GuildSettings(voice_talk="nonsense").voice_talk == "may_talk"
    assert GuildSettingsUpdate(voice_talk="listen_only").voice_talk == "listen_only"


async def test_switch_closes_live_and_back_restarts_it(lv):
    s, live = await lv.join(5)
    assert s.voice_mode == "live"
    await lv.mgr.set_voice_talk(s, "listen_only")
    assert s.live is None and s.voice_mode == "classic" and live.closed
    assert lv.chan.sent[-1] == t("voice_talk_listen", s.lang)
    rec = lv.bot.secretary_store.get_active_meeting(1000, 77)
    assert rec["voice_talk"] == "listen_only"
    await lv.mgr.set_voice_talk(s, "may_talk")
    await until(lambda: len(lv.connect.sessions) == 2)
    assert s.voice_mode == "live" and s.live is not None
    assert lv.chan.sent[-1] == t("voice_talk_may", s.lang)
    await lv.mgr.end((1000, 77))


async def test_org_default_listen_only_never_starts_live_and_is_announced(lv):
    lv.bot.guilds_store.save_settings(1000, GuildSettingsUpdate(voice_talk="listen_only"), "x")
    lv.bot.team._cache.pop(1000, None)
    lv.bot.s.voice_state_posts = True
    m = make_member(lv.guild, 5, "boss", lv.chan)
    lv.chan.members = [m]
    await lv.mgr.join(m)
    s = lv.mgr.sessions[(1000, 77)]
    assert s.live is None and lv.connect.configs == []
    assert any(t("voice_talk_listen", s.lang) in x for x in lv.chan.sent)
    s.voice_talk = "may_talk"  # the meeting's own switch beats the org default
    assert lv.mgr.talk_mode(s) == "may_talk"
    await lv.mgr.end((1000, 77))


async def test_listen_only_name_plus_question_is_a_text_answer_and_no_tts(voice, monkeypatch):
    s = await start(voice)
    voice.bot.s.meeting_text_echo = False  # the usual voice-only default
    seen = {}
    import salid.qa

    def ask(*a, **kw):
        seen.update(kw)
        return SimpleNamespace(text="เลื่อนเป็นวันศุกร์ค่ะ")

    monkeypatch.setattr(salid.qa, "ask", ask)
    await voice.mgr.set_voice_talk(s, "listen_only")
    voice.chan.sent.clear()
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด เมื่อกี้ตกลงอะไรกัน")
    await voice.mgr.handle_segment(s, segment(11))
    await asyncio.sleep(0.05)
    assert voice.chan.sent == ["<@11> เลื่อนเป็นวันศุกร์ค่ะ"]
    assert seen["spoken"] is False
    assert voice.vc.played == [] and not [c for c in voice.calls if c[0] == "tts"]
    await voice.mgr.end((1000, 77))


async def test_chair_and_executive_switch_by_voice_others_are_refused(env, monkeypatch):
    s = await start(env)  # member 10 invited Salid: the chair
    make_member(env.guild, 5, "boss", env.chan)
    saying = {}
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: saying["text"])
    env.chan.sent.clear()
    saying["text"] = "สลิด ฟังอย่างเดียว"
    await env.mgr.handle_segment(s, segment(11))  # not the chair, not an executive
    assert s.voice_talk == "" and env.chan.sent == [t("voice_talk_denied", s.lang)]
    await env.mgr.handle_segment(s, segment(10))
    assert s.voice_talk == "listen_only"
    saying["text"] = "Salid, you may talk"
    await env.mgr.handle_segment(s, segment(5))  # executive
    assert s.voice_talk == "may_talk"
    await env.mgr.end((1000, 77))


async def test_typed_forms(env):
    s = await start(env)
    other = make_member(env.guild, 12, "ซี", env.chan)
    chair = env.guild.members[10]
    msg, sent = fake_message(env, "listen only", other)
    assert await env.bot._voice_talk_command(msg) is True
    assert sent == [t("voice_talk_denied", s.lang)] and s.voice_talk == ""
    msg, sent = fake_message(env, "สลิด ฟังอย่างเดียว", chair)
    assert await env.bot._voice_talk_command(msg) is True and s.voice_talk == "listen_only"
    msg, sent = fake_message(env, "listen only by default", other)  # org form: executives only
    assert await env.bot._voice_talk_command(msg) is True
    assert sent == [t("voice_talk_default_denied", "th")]
    assert env.bot.team.voice_talk(1000) == "may_talk"
    boss = make_member(env.guild, 5, "boss", env.chan)
    boss.guild_permissions = SimpleNamespace(administrator=True, manage_guild=True)
    msg, sent = fake_message(env, "ประชุมทุกครั้งฟังอย่างเดียว", boss)
    assert await env.bot._voice_talk_command(msg) is True
    assert env.bot.team.voice_talk(1000) == "listen_only"
    msg, _ = fake_message(env, "สวัสดี", boss)
    assert await env.bot._voice_talk_command(msg) is False
    await env.mgr.end((1000, 77))


def test_hub_html_has_the_select_and_both_languages():
    from pathlib import Path

    html = (Path(__file__).parents[2] / "src/meeting_scribe/static/hub.html").read_text("utf-8")
    for k in ('id="voice-talk"', "voice_talk_l", "voice_talk:", "listen_only"):
        assert k in html
    assert "เสียงของสลิดในห้องประชุม" in html and "Salid's voice in meetings" in html
