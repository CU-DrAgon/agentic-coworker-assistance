"""Mouth gate (v0.8.1): Salid's Live voice is played only when someone called her name, and name
hits need real speech behind them. Offline: fake Live session through the real LiveVoice.
The shared `lv` fixture has the gate OFF (other suites test playback mechanics); `gl` turns it on."""

import asyncio
import logging
import time
from array import array

import pytest
from test_live_voice import (
    audio_msg,
    capture_cues,
    done_msg,
    heard_msg,
    lv,  # noqa: F401  (fixture)
    pcm48,
    said_msg,
    tool_msg,
    until,
)
from test_meeting import SPEECH_FRAME

from salid import live_voice as lvm
from salid import meeting as mt
from salid.config import Settings

MUTED = "live: unsolicited audio muted (listening)"


@pytest.fixture
def gl(lv):  # noqa: F811
    lv.bot.s.live_mouth_gate = True
    lv.bot.s.live_call_grace_seconds = 0.15
    return lv


async def settle(t=0.2):
    await asyncio.sleep(t)


def muted_count(caplog):
    return sum(1 for r in caplog.records if r.getMessage() == MUTED)


# ------------------------------------------------------------------ audio gate
async def test_audio_while_listening_is_muted_and_logged_once(gl, caplog):
    caplog.set_level(logging.INFO, logger="salid.live_voice")
    s, live = await gl.join(5)
    live.push(audio_msg(), said_msg("หวัดดีค่ะ"))
    await settle(0.25)  # past the grace window
    live.push(audio_msg(), audio_msg(), said_msg(" ทุกคน"), done_msg())
    await settle()
    assert gl.vc.played == [] and s.engaged is False
    assert muted_count(caplog) == 1
    assert not any("🔊" in m for m in gl.chan.sent)  # a muted turn posts no spoken line
    await gl.mgr.end((1000, 77))


async def test_short_unsolicited_turn_is_discarded_at_turn_end(gl, caplog):
    caplog.set_level(logging.INFO, logger="salid.live_voice")
    gl.bot.s.live_call_grace_seconds = 5.0
    s, live = await gl.join(5)
    live.push(audio_msg(), said_msg("ฮัลโหล"), done_msg())
    await settle()
    assert gl.vc.played == [] and muted_count(caplog) == 1
    assert not any("🔊" in m for m in gl.chan.sent)
    assert s.live._held == []
    await gl.mgr.end((1000, 77))


async def test_audio_while_engaged_plays_and_posts(gl):
    s, live = await gl.join(5)
    gl.mgr._set_engaged(s, True)
    live.push(audio_msg(), said_msg("ค่ะ"), done_msg())
    await until(lambda: gl.vc.played)
    await until(lambda: "🔊 ค่ะ" in gl.chan.sent)
    await gl.mgr.end((1000, 77))


async def test_tool_answer_plays_even_if_engaged_dropped_meanwhile(gl):
    s, live = await gl.join(5)
    gl.mgr._set_engaged(s, True)
    live.push(tool_msg("set_captions", {"on": True}))
    await until(lambda: live.tool_responses)
    gl.mgr._set_engaged(s, False)  # back to listening while the answer is being produced
    assert s.engaged is False
    live.push(audio_msg(), said_msg("เปิดซับแล้วค่ะ"), done_msg())
    await until(lambda: gl.vc.played)
    await until(lambda: "🔊 เปิดซับแล้วค่ะ" in gl.chan.sent)
    await gl.mgr.end((1000, 77))


async def test_grace_buffer_is_released_when_the_name_arrives_in_time(gl, caplog):
    caplog.set_level(logging.INFO, logger="salid.live_voice")
    gl.bot.s.live_call_grace_seconds = 1.0
    s, live = await gl.join(5)
    live.push(audio_msg(), audio_msg())
    await settle(0.1)
    assert gl.vc.played == []  # held, not played, not lost
    gl.mgr._set_engaged(s, True)  # the name reaches the transcript a moment later
    await until(lambda: gl.vc.played)
    assert muted_count(caplog) == 0
    await gl.mgr.end((1000, 77))


async def test_grace_buffer_is_discarded_when_the_name_comes_too_late(gl, caplog):
    caplog.set_level(logging.INFO, logger="salid.live_voice")
    gl.bot.s.live_call_grace_seconds = 0.05
    s, live = await gl.join(5)
    live.push(audio_msg())
    await settle(0.15)
    gl.mgr._set_engaged(s, True)
    live.push(audio_msg())  # the rest of the unsolicited turn stays muted too
    await settle()
    assert gl.vc.played == [] and muted_count(caplog) == 1
    await gl.mgr.end((1000, 77))


async def test_mouth_gate_off_is_the_old_behaviour(gl):
    gl.bot.s.live_mouth_gate = False
    s, live = await gl.join(5)
    live.push(audio_msg(), said_msg("สวัสดี"), done_msg())
    await until(lambda: gl.vc.played)
    await until(lambda: "🔊 สวัสดี" in gl.chan.sent)
    live.push(tool_msg("set_captions", {"on": True}, id="c9"))
    await until(lambda: live.tool_responses)
    assert s.captions.on
    await gl.mgr.end((1000, 77))


# ------------------------------------------------------------------ tool gate
async def test_refused_tools_have_no_side_effect_while_listening(gl):
    s, live = await gl.join(5)
    calls = [
        ("set_captions", {"on": True}),
        ("set_recording", {"recording": False}),
        ("set_minutes_destination", {"channel": "general"}),
        ("ask_consent", {"text": "ประชุมไหมคะ", "medium": "voice"}),
        ("ask_salid", {"question": "งานอะไรค้าง"}),
        ("answer_privately", {"question": "งานอะไรค้าง"}),
    ]
    before = (s.recording, s.captions.on, s.consent)
    for i, (name, args) in enumerate(calls, 1):
        live.push(tool_msg(name, args, id=f"c{i}"))
        await until(lambda i=i: len(live.tool_responses) == i)
        assert live.tool_responses[-1][0].response == {"error": "nobody asked: stay silent"}
    assert (s.recording, s.captions.on, s.consent) == before
    assert s.live._awaiting_answer is False  # nothing will be answered out loud
    await gl.mgr.end((1000, 77))


async def test_post_note_and_text_consent_still_work_while_listening(gl):
    s, live = await gl.join(5)
    live.push(tool_msg("post_note", {"text": "ส่งรายงานวันศุกร์"}, id="n1"))
    await until(lambda: live.tool_responses)
    assert "error" not in live.tool_responses[-1][0].response
    live.push(tool_msg("ask_consent", {"text": "ประชุมไหมคะ"}, id="n2"))
    await until(lambda: len(live.tool_responses) == 2)
    assert live.tool_responses[-1][0].response != {"error": "nobody asked: stay silent"}
    await gl.mgr.end((1000, 77))


async def test_tools_work_when_engaged(gl):
    s, live = await gl.join(5)
    gl.mgr._set_engaged(s, True)
    live.push(tool_msg("set_captions", {"on": True}))
    await until(lambda: live.tool_responses)
    assert s.captions.on
    await gl.mgr.end((1000, 77))


# ------------------------------------------------------------------ join / unmute is no call
async def test_joining_does_not_engage_or_cue(gl):
    capture = []
    s, live = await gl.join(5)
    capture = [n for n in live.notes if "เรียกชื่อคุณ" in n[0] or "called your name" in n[0]]
    assert s.engaged is False and capture == []
    s.on_pcm(5, pcm48(1000), time.monotonic())  # a mic opens: audio goes to the model, no cue
    await settle()
    assert s.engaged is False
    assert live.audio  # still sent: it may contain the name
    await gl.mgr.end((1000, 77))


# ------------------------------------------------------------------ energy gate for name hits
def test_voiced_fraction_silence_noise_and_speech():
    assert mt.voiced_fraction(b"") == 0.0
    assert mt.voiced_fraction(bytes(3840 * 50)) == 0.0
    noise = array("h", [60] * 1920).tobytes() * 50
    assert mt.voiced_fraction(noise) == 0.0
    assert mt.voiced_fraction(SPEECH_FRAME * 50) == 1.0
    mixed = SPEECH_FRAME * 15 + bytes(3840 * 35)
    assert mt.voiced_fraction(mixed) == pytest.approx(0.3)


def test_transcription_prompt_no_longer_primes_the_greeting():
    assert "หวัดดีครับ สลิด" not in mt.TRANSCRIBE_PROMPT
    assert "สะกดว่า สลิด" in mt.TRANSCRIBE_PROMPT and "ตอบ text ว่าง" in mt.TRANSCRIBE_PROMPT


def test_energy_settings_defaults():
    s = Settings(_env_file=None)
    assert s.name_min_voiced_seconds == 0.4 and s.name_min_rms == 250.0
    assert s.live_mouth_gate is True and s.live_call_grace_seconds == 2.5


async def test_classic_greeting_on_low_energy_audio_does_not_engage(lv, monkeypatch, caplog):  # noqa: F811
    caplog.set_level(logging.INFO, logger="salid.meeting")
    s, live = await lv.join(5)
    sent = capture_cues(live)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "หวัดดีครับ สลิด")  # 15 chars
    quiet = array("h", [150] * 1920).tobytes() * 100  # above the send floor, below the voice floor
    await lv.mgr.handle_segment(s, mt.Segment(5, quiet, time.monotonic(), 2.0))
    await settle()
    assert s.engaged is False and not [t for t in sent if t[1]]
    assert "name hit rejected: no voiced audio" in caplog.text
    await lv.mgr.end((1000, 77))


async def test_classic_greeting_on_real_speech_engages(lv, monkeypatch):  # noqa: F811
    s, live = await lv.join(5)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "หวัดดีครับ สลิด")
    await lv.mgr.handle_segment(s, mt.Segment(5, SPEECH_FRAME * 100, time.monotonic(), 2.0))
    assert s.engaged is True
    await lv.mgr.end((1000, 77))


async def test_live_name_hit_without_voiced_audio_is_ignored(lv, caplog):  # noqa: F811
    caplog.set_level(logging.INFO, logger="salid.live_voice")
    s, live = await lv.join(5)
    lv.bot.s.live_name_reply_seconds = 0.05
    sent = capture_cues(live)
    for _ in range(5):
        s.live.feed(5, pcm48(40))  # a faint hiss
    live.push(heard_msg("หวัดดีครับ สลิด"))
    await settle(0.3)
    assert s.engaged is False and not [t for t in sent if t[1]]  # no cue
    assert "live: name hit rejected: no voiced audio" in caplog.text
    await lv.mgr.end((1000, 77))


async def test_live_name_hit_with_voiced_audio_engages(lv):  # noqa: F811
    s, live = await lv.join(5)
    lv.bot.s.live_name_reply_seconds = 0.05
    for _ in range(30):
        s.live.feed(5, pcm48(800))  # 0.6 s of speech
    live.push(heard_msg("หวัดดีครับ สลิด"))
    await until(lambda: s.engaged)
    await lv.mgr.end((1000, 77))


def test_min_rms_is_raised():
    assert mt.MIN_RMS == 120 and lvm.FRAME_S == 0.02
