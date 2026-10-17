# ruff: noqa: F811, F401
"""v0.11: the Live model decides wake / stay / end. Code keeps a few hard rules: the name for the
mouth gate, the 10 s bare-name timeout, the 90 s cap, the hush command, a two-names backup.
Offline: fake Live session and fake transcriber."""

import time
from pathlib import Path

from test_false_wake import record_replies, say, seg
from test_live_voice import audio_msg, done_msg, lv, tool_msg, until
from test_meeting import SPEECH_PCM
from test_mouth_gate import gl

from salid import meeting as mt
from salid.guild_settings import GuildSettingsUpdate
from salid.live_voice import build_instruction
from salid.meeting import Heard

from test_live_voice import FixedClockSettings  # isort: skip


async def test_end_conversation_tool_goes_back_to_listening_with_the_reason(lv):
    lv.bot.s.voice_state_posts = True
    s, live = await lv.join(5)
    lv.mgr._set_engaged(s, True, name="มิว", by=5)
    assert s.engaged and s.caller == "มิว"
    live.push(tool_msg("end_conversation", {"reason": "they thanked me"}))
    await until(lambda: live.tool_responses)
    assert s.engaged is False and s.caller == ""
    await until(lambda: any("said thanks" in m or "ขอบคุณแล้ว" in m for m in lv.chan.sent))
    assert any("มิว" in m and "เรียกชื่อ" in m for m in lv.chan.sent)
    await lv.mgr.end((1000, 77))


def test_end_reason_maps_to_a_line_and_defaults_to_ended():
    assert mt.MeetingManager._END_WHY[0][1].search("thanked")
    assert mt.MeetingManager._END_WHY[1][1].search("went back to each other")
    assert mt.MeetingManager._END_WHY[2][1].search("told to be quiet")


async def test_live_mode_ignores_the_dismissal_list_and_the_three_strikes(lv, monkeypatch):
    s, live = await lv.join(5)
    lv.mgr._set_engaged(s, True, by=5)
    for text in ("โอเค คนต่อไป", "ok next", "ขอบคุณครับ", "ร้านนี้อร่อยดี", "ไปกินข้าวกัน", "มึงเอาไหม"):
        say(monkeypatch, Heard(text))
        await lv.mgr.handle_segment(s, seg())
    assert s.engaged is True and s.unaddressed == 0  # only the Live model (or the cap) ends it
    await lv.mgr.end((1000, 77))


async def test_bare_name_engagement_lasts_10_seconds(lv):
    s, live = await lv.join(5)
    lv.mgr._set_engaged(s, True, by=5, bare=True)
    assert 9.0 < s.engaged_until - time.monotonic() <= 10.0
    lv.mgr._consent_tick(s, time.monotonic() + 11)
    assert s.engaged is False
    await lv.mgr.end((1000, 77))


async def test_the_cap_is_90_seconds_and_her_own_speech_keeps_it_away(lv):
    s, live = await lv.join(5)
    lv.mgr._set_engaged(s, True, by=5)
    assert 89.0 < s.engaged_until - time.monotonic() <= 90.0
    s.engaged_until = time.monotonic() + 5
    live.push(audio_msg())
    await until(lambda: s.engaged_until - time.monotonic() > 80)
    await lv.mgr.end((1000, 77))


async def test_name_twice_within_15_seconds_forces_one_cue(lv, monkeypatch):
    s, live = await lv.join(5)
    calls = record_replies(s)
    say(monkeypatch, Heard("สลิด", True))
    await lv.mgr.handle_segment(s, seg())
    assert calls == [] and s.engaged is False  # one bare name: the Live model decides
    await lv.mgr.handle_segment(s, seg())
    assert len(calls) == 1 and s.engaged is True and s.engaged_bare is True
    await lv.mgr.handle_segment(s, seg())
    assert len(calls) == 1  # the counter restarted: no second cue for one more name
    await lv.mgr.end((1000, 77))


async def test_name_twice_but_she_already_spoke_forces_nothing(lv, monkeypatch):
    s, live = await lv.join(5)
    calls = record_replies(s)
    say(monkeypatch, Heard("สลิด", True))
    await lv.mgr.handle_segment(s, seg())
    s.live._real_audio_at = time.monotonic() + 1
    await lv.mgr.handle_segment(s, seg())
    assert calls == []
    await lv.mgr.end((1000, 77))


async def test_strict_gate_blocks_unnamed_audio_natural_opens_8s_after_her_words(gl):
    s, live = await gl.join(5)
    assert s.live._wake_policy() == "strict" and not s.live._gate_open()
    live.push(audio_msg(), done_msg())
    await until(lambda: not s.live._turn_open)
    assert gl.vc.played == []
    s.live._wake_policy = lambda: "natural"
    s.live._real_audio_at = time.monotonic() - 9
    assert not s.live._gate_open()
    s.live._real_audio_at = time.monotonic() - 2
    assert s.live._gate_open()
    live.push(audio_msg(), done_msg())
    await until(lambda: gl.vc.played)
    assert s.engaged is True  # a direct reply re-engages her
    await gl.mgr.end((1000, 77))


def test_wake_policy_round_trip_and_instruction_text():
    assert GuildSettingsUpdate().wake_policy == "strict"
    assert GuildSettingsUpdate(wake_policy="natural").wake_policy == "natural"
    assert GuildSettingsUpdate(wake_policy="bogus").wake_policy == "strict"
    hub = Path(mt.__file__).parents[1] / "meeting_scribe" / "static" / "hub.html"
    html = hub.read_text(encoding="utf-8")
    assert 'id="wake-policy"' in html and "wake_policy:" in html and "s.wake_policy" in html
    s = FixedClockSettings()
    for lang in ("th", "en"):
        strict = build_instruction(s, lang, wake_policy="strict")
        natural = build_instruction(s, lang, wake_policy="natural")
        assert "end_conversation" in strict and len(natural) > len(strict)
