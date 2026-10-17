"""Mouth gate (v0.8.1): Salid's Live voice is played only when someone called her name, and name
hits need real speech behind them. Offline: fake Live session through the real LiveVoice.
The shared `lv` fixture has the gate OFF (other suites test playback mechanics); `gl` turns it on."""

import asyncio
import logging
import math
import random
import re
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
FRAME = 3840  # 20 ms, 48 kHz stereo s16le


def _pcm(values):
    return array("h", [max(-32768, min(32767, int(v))) for v in values for _ in (0, 1)]).tobytes()


def frames_of(pcm):
    return [pcm[i : i + FRAME] for i in range(0, len(pcm) - FRAME + 1, FRAME)]


def noise(seconds, rms_=250.0, seed=1):
    r = random.Random(seed)
    return _pcm(r.gauss(0, rms_) for _ in range(int(48000 * seconds)))


def hum(seconds, seed=2):  # 50 Hz hum + hiss, quiet
    r = random.Random(seed)
    return _pcm(
        330 * math.sin(2 * math.pi * 50 * i / 48000) + r.gauss(0, 12)
        for i in range(int(48000 * seconds))
    )


def bursts(seconds, seed=3):  # steady noise bursts: 100 ms at RMS ~600 every 200 ms-ish
    r = random.Random(seed)
    out = []
    for i in range(int(48000 * seconds)):
        loud = (i // 4800) % 2 == 0
        out.append(r.gauss(0, 330 if loud else 20))
    return _pcm(out)


def click(seconds=1.2, seed=4):  # one 30 ms click in hiss
    r = random.Random(seed)
    n = int(48000 * seconds)
    return _pcm(r.gauss(0, 3000 if 20000 <= i < 21440 else 40) for i in range(n))


def speechlike(seconds, seed=5):
    """Bursts of 60-150 ms with short gaps and varied amplitude (900-5000)."""
    r = random.Random(seed)
    out = []
    n = int(48000 * seconds)
    while len(out) < n:
        amp = r.uniform(900, 5000)
        for _ in range(int(48000 * r.uniform(0.06, 0.15))):
            out.append(r.gauss(0, amp / 1.2))
        out.extend(r.gauss(0, 40) for _ in range(int(48000 * 0.03)))
    return _pcm(out[:n])


def verdict(pcm):
    stats = mt.speech_stats(pcm)
    return mt.looks_like_speech(stats, Settings(_env_file=None)), stats


@pytest.mark.parametrize(
    "name,pcm,ok",
    [
        ("white noise", noise(0.9), False),
        ("hum + hiss", hum(1.5), False),
        ("noise bursts", bursts(1.0), False),
        ("mic click", click(), False),
        ("speech-like sentence", speechlike(2.0), True),
        ("speech-like bare name 0.5 s", speechlike(0.5, seed=7), True),
    ],
)
def test_looks_like_speech_table(name, pcm, ok):
    (got, reason), stats = verdict(pcm)
    assert got is ok, (name, reason, stats)
    assert (reason == "") is ok


def test_speech_stats_basics():
    z = mt.speech_stats(b"")
    assert z == mt.SpeechStats(0.0, 0.0, 0.0, 0.0, 0.0)
    q = mt.speech_stats(bytes(FRAME * 10))
    assert q.cv == 0.0 and q.voiced_s == 0.0 and q.duration_s == pytest.approx(0.2)
    c = mt.speech_stats(SPEECH_FRAME * 25)
    assert c.voiced_s == pytest.approx(0.5) and c.voiced_fraction == 1.0 and c.cv < 0.01


def test_gate_off_setting_is_zero_voiced_seconds():
    assert Settings(_env_file=None, name_min_voiced_seconds=0).name_min_voiced_seconds == 0


async def test_classic_decision_log_has_numbers_and_no_text(lv, monkeypatch, caplog):  # noqa: F811
    caplog.set_level(logging.INFO, logger="salid.meeting")
    s, live = await lv.join(5)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด")
    await lv.mgr.handle_segment(s, mt.Segment(5, speechlike(0.8, seed=7), time.monotonic(), 0.8))
    assert s.engaged is False and len(s.name_calls) == 1  # live: one bare name is left to the model
    await lv.mgr.end((1000, 77))
    line = next(r.getMessage() for r in caplog.records if "name hit" in r.getMessage())
    assert re.fullmatch(
        r"meeting \d+: name hit accepted dur=0\.80 rms=\d+ voiced=[\d.]+ vf=[\d.]+ cv=[\d.]+", line
    ), line
    assert "สลิด" not in line


async def test_classic_steady_noise_is_rejected_with_reason(lv, monkeypatch, caplog):  # noqa: F811
    caplog.set_level(logging.INFO, logger="salid.meeting")
    s, live = await lv.join(5)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด")
    await lv.mgr.handle_segment(s, mt.Segment(5, noise(0.9, 400), time.monotonic(), 0.9))
    assert s.engaged is False
    assert "name hit rejected: steady noise dur=0.90" in caplog.text
    await lv.mgr.end((1000, 77))


async def test_live_steady_noise_is_rejected_and_logged(lv, caplog):  # noqa: F811
    caplog.set_level(logging.INFO, logger="salid.live_voice")
    s, live = await lv.join(5)
    for f in frames_of(noise(0.9, 400)):
        s.live.feed(5, f)
    live.push(heard_msg("สลิด"))
    await settle(0.3)
    assert s.engaged is False
    assert re.search(r"live: name hit rejected: steady noise dur=0\.\d+ rms=\d+ ", caplog.text)
    await lv.mgr.end((1000, 77))


async def test_live_accept_log_shape(lv, caplog):  # noqa: F811
    caplog.set_level(logging.INFO, logger="salid.live_voice")
    s, live = await lv.join(5)
    lv.bot.s.live_name_reply_seconds = 0.05
    for f in frames_of(speechlike(0.5, seed=7)):
        s.live.feed(5, f)
    live.push(heard_msg("สลิด"))
    await until(lambda: s.engaged)
    assert re.search(
        r"live: name hit accepted dur=0\.50 rms=\d+ voiced=[\d.]+ vf=[\d.]+ cv=[\d.]+", caplog.text
    )
    await lv.mgr.end((1000, 77))


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
    assert "สะกดว่า สลิด" not in mt.TRANSCRIBE_PROMPT and "ตอบ text ว่าง" in mt.TRANSCRIBE_PROMPT


def test_energy_settings_defaults():
    s = Settings(_env_file=None)
    assert s.name_min_voiced_seconds == 0.35 and s.name_min_rms == 250.0
    assert s.name_min_cv == 0.25
    assert s.live_mouth_gate is True and s.live_call_grace_seconds == 4.0


async def test_classic_greeting_on_low_energy_audio_does_not_engage(lv, monkeypatch, caplog):  # noqa: F811
    caplog.set_level(logging.INFO, logger="salid.meeting")
    s, live = await lv.join(5)
    sent = capture_cues(live)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "หวัดดีครับ สลิด")  # 15 chars
    quiet = array("h", [150] * 1920).tobytes() * 100  # above the send floor, below the voice floor
    await lv.mgr.handle_segment(s, mt.Segment(5, quiet, time.monotonic(), 2.0))
    await settle()
    assert s.engaged is False and not [t for t in sent if t[1]]
    assert "name hit rejected: too little voiced audio" in caplog.text
    await lv.mgr.end((1000, 77))


async def test_classic_greeting_on_real_speech_engages(lv, monkeypatch):  # noqa: F811
    s, live = await lv.join(5)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "หวัดดีครับ สลิด")
    await lv.mgr.handle_segment(s, mt.Segment(5, speechlike(1.5), time.monotonic(), 1.5))
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
    assert "live: name hit rejected: too little voiced audio" in caplog.text
    await lv.mgr.end((1000, 77))


async def test_live_name_hit_with_voiced_audio_engages(lv):  # noqa: F811
    s, live = await lv.join(5)
    lv.bot.s.live_name_reply_seconds = 0.05
    for f in frames_of(speechlike(0.5)):  # a bare "สลิด": 0.5 s
        s.live.feed(5, f)
    live.push(heard_msg("หวัดดีครับ สลิด"))
    await until(lambda: s.engaged)
    await lv.mgr.end((1000, 77))


def test_min_rms_is_raised():
    assert mt.MIN_RMS == 120 and lvm.FRAME_S == 0.02


# ------------------------------------------------------------------ re-engage grace, ask_salid evidence
async def test_recently_engaged_grace_opens_gates_and_reengages(gl):
    s, live = await gl.join(5)
    gl.mgr._set_engaged(s, True)
    gl.mgr._set_engaged(s, False)  # the room kept talking: dropped back to listening
    assert s.engaged is False and s.live._gate_open()
    live.push(tool_msg("set_captions", {"on": True}))
    await until(lambda: live.tool_responses)
    assert s.captions.on and s.engaged is True  # re-engaged in code
    await gl.mgr.end((1000, 77))


async def test_grace_expires(gl):
    s, live = await gl.join(5)
    gl.mgr._set_engaged(s, True)
    gl.mgr._set_engaged(s, False)
    s.live._last_engaged_at = time.monotonic() - 61
    assert not s.live._gate_open()
    live.push(tool_msg("set_captions", {"on": True}))
    await until(lambda: live.tool_responses)
    assert live.tool_responses[-1][0].response == {"error": "nobody asked: stay silent"}
    await gl.mgr.end((1000, 77))


async def test_grace_also_opens_the_mouth_gate(gl):
    s, live = await gl.join(5)
    gl.mgr._set_engaged(s, True)
    gl.mgr._set_engaged(s, False)
    live.push(audio_msg(), said_msg("ค่ะ"), done_msg())
    await until(lambda: gl.vc.played)
    await gl.mgr.end((1000, 77))


async def test_ask_salid_with_real_speech_engages(gl, caplog):
    caplog.set_level(logging.INFO, logger="salid.live_voice")
    s, live = await gl.join(5)

    async def fake_run(name, args):
        return {"answer": "ok"}

    s.live._run_tool = fake_run
    for f in frames_of(speechlike(1.5)):
        s.live.feed(5, f)
    s.name_calls = [time.monotonic()]  # strict policy: her name was heard (ASR) a moment ago
    live.push(tool_msg("ask_salid", {"question": "งานอะไรค้าง"}))
    await until(lambda: live.tool_responses)
    assert live.tool_responses[-1][0].response != {"error": "nobody asked: stay silent"}
    assert s.engaged is True
    assert "live: ask_salid accepted: speech heard, engaging" in caplog.text
    await gl.mgr.end((1000, 77))


async def test_ask_salid_with_noise_is_refused_and_side_effects_stay_gated(gl):
    s, live = await gl.join(5)
    for f in frames_of(noise(1.0, 400)):
        s.live.feed(5, f)
    live.push(tool_msg("ask_salid", {"question": "x"}, id="a"))
    await until(lambda: len(live.tool_responses) == 1)
    assert live.tool_responses[-1][0].response == {"error": "nobody asked: stay silent"}
    assert s.engaged is False
    for f in frames_of(speechlike(1.5)):
        s.live.feed(5, f)
    live.push(tool_msg("set_captions", {"on": True}, id="b"))
    await until(lambda: len(live.tool_responses) == 2)
    assert live.tool_responses[-1][0].response == {"error": "nobody asked: stay silent"}
    assert not s.captions.on and s.engaged is False
    await gl.mgr.end((1000, 77))


async def test_ask_evidence_accepts_the_live_numbers_that_were_wrongly_refused(gl):
    """Seen live: dur=0.96 rms=363 voiced=0.34 vf=0.35 cv=1.37 is a real question (name hits still
    need 0.35 s voiced and 30% voiced frames; ask_salid only has to exclude noise)."""
    from salid.meeting import SpeechStats, looks_like_speech

    s, live = await gl.join(5)
    stats = SpeechStats(0.96, 363.0, 0.34, 0.35, 1.37)
    assert s.live._ask_evidence(stats) == (True, "")
    assert looks_like_speech(stats, gl.bot.s)[0] is False  # the name check is unchanged
    assert s.live._ask_evidence(SpeechStats(0.96, 363.0, 0.1, 0.1, 1.37))[0] is False  # too short
    assert s.live._ask_evidence(SpeechStats(1.0, 363.0, 0.5, 0.5, 0.1))[0] is False  # steady noise
    assert s.live._ask_evidence(SpeechStats(1.0, 100.0, 0.5, 0.5, 1.4))[0] is False  # too quiet
    await gl.mgr.end((1000, 77))


async def test_ask_evidence_looks_back_six_seconds_names_keep_three(gl):
    s, live = await gl.join(5)
    now = time.monotonic()
    s.live._frames = [(now - 5.0 + i * 0.02, 900.0 if i % 3 else 300.0) for i in range(40)]
    assert s.live._speech_stats(now).voiced_s == 0.0  # the 3 s window sees nothing
    assert s.live._speech_stats(now, lvm.ASK_EVIDENCE_S).voiced_s >= 0.2
    await gl.mgr.end((1000, 77))
