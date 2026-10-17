# ruff: noqa: F811, F401
"""False wake-ups (v0.9.9): a one-word clip the transcriber wrote as "สลิด" never forces a turn,
the cue never orders a reply to nothing, the caller gets no automatic pass while engaged, and a
bare-name engagement expires fast. Offline, fakes only."""

import asyncio
import time
from array import array
from types import SimpleNamespace

import pytest
from test_live_voice import capture_cues, lv, until
from test_meeting import ME, SPEECH_FRAME, SPEECH_PCM, env, make_member, segment, start

from salid import live_voice as lvm
from salid import meeting as mt
from salid.config import Settings
from salid.live_voice import FORCE_CUE, FORCE_CUE_EN
from salid.meeting import BARE_WAKE_FOLLOW_S, Heard, SpeechStats, bare_name, looks_like_speech


def say(monkeypatch, *heard):
    """The transcriber returns these Heard values in order (the last one repeats)."""
    it = iter(heard)
    last = [heard[-1]]

    def fake(client, model, pcm, language=None):
        last[0] = next(it, last[0])
        return last[0]

    monkeypatch.setattr(mt, "transcribe_heard", fake)


def record_replies(s):
    calls = []
    s.live.ensure_reply = lambda since, heard="", **kw: calls.append((heard, kw))
    return calls


def seg(uid=5):
    return mt.Segment(uid, SPEECH_PCM, time.monotonic(), 2.0)


# ------------------------------------------------------------------ live: a bare name
async def test_live_bare_name_from_the_transcriber_only_sets_pending(lv, monkeypatch):
    s, live = await lv.join(5)
    calls = record_replies(s)
    say(monkeypatch, Heard("สลิด", False))
    await lv.mgr.handle_segment(s, seg())
    assert s.engaged is False and calls == []
    assert 5 in s.pending_wake
    await lv.mgr.end((1000, 77))


async def test_live_bare_name_even_with_name_called_true_is_pending_only(lv, monkeypatch):
    s, live = await lv.join(5)
    calls = record_replies(s)
    say(monkeypatch, Heard("สลิด", True))
    await lv.mgr.handle_segment(s, seg())
    assert s.engaged is False and calls == [] and 5 in s.pending_wake
    await lv.mgr.end((1000, 77))


async def test_live_bare_name_then_a_question_within_5_seconds_is_the_call(lv, monkeypatch):
    s, live = await lv.join(5)
    calls = record_replies(s)
    say(monkeypatch, Heard("สลิด", False), Heard("วันนี้มีนัดอะไรบ้าง", False))
    await lv.mgr.handle_segment(s, seg())
    await lv.mgr.handle_segment(s, seg())
    assert s.engaged is True
    assert len(calls) == 1 and calls[0][1]["question"] == "วันนี้มีนัดอะไรบ้าง"
    assert 5 not in s.pending_wake
    await lv.mgr.end((1000, 77))


async def test_live_bare_name_then_three_plain_words_is_the_call_too(lv, monkeypatch):
    s, live = await lv.join(5)
    calls = record_replies(s)
    say(monkeypatch, Heard("Salid", False), Heard("summarize the budget", False))
    await lv.mgr.handle_segment(s, seg())
    await lv.mgr.handle_segment(s, seg())
    assert len(calls) == 1 and s.engaged is True
    await lv.mgr.end((1000, 77))


async def test_live_question_after_the_follow_window_is_nothing(lv, monkeypatch):
    s, live = await lv.join(5)
    calls = record_replies(s)
    say(monkeypatch, Heard("สลิด", False), Heard("วันนี้มีนัดอะไรบ้าง", False))
    await lv.mgr.handle_segment(s, seg())
    s.pending_wake[5] -= BARE_WAKE_FOLLOW_S + 1  # the 5 s are over (the old window was 15 s)
    await lv.mgr.handle_segment(s, seg())
    assert s.engaged is False and calls == []
    await lv.mgr.end((1000, 77))


async def test_live_follow_up_must_come_from_the_same_speaker(lv, monkeypatch):
    s, live = await lv.join(5, 6)
    calls = record_replies(s)
    say(monkeypatch, Heard("สลิด", False), Heard("วันนี้มีนัดอะไรบ้าง", False))
    await lv.mgr.handle_segment(s, seg(5))
    await lv.mgr.handle_segment(s, seg(6))
    assert s.engaged is False and calls == []
    await lv.mgr.end((1000, 77))


async def test_live_name_with_words_behaves_as_before(lv, monkeypatch):
    s, live = await lv.join(5)
    calls = record_replies(s)
    say(monkeypatch, Heard("Salid, what's the status?", True))
    await lv.mgr.handle_segment(s, seg())
    assert s.engaged is True and s.engaged_bare is False
    assert len(calls) == 1 and calls[0][1]["question"] == "what's the status"
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ classic: a bare name
async def test_classic_bare_name_the_transcriber_says_was_addressed_asks_back(env, monkeypatch):
    s = await start(env)
    say(monkeypatch, Heard("สลิด", True))
    await env.mgr.handle_segment(s, segment(10))
    assert s.engaged is True and s.engaged_bare is True
    assert 10 in s.pending_wake  # the "คะ?" path: the next words are the question
    assert any("<@10>" in m for m in env.chan.sent)  # the "I'm listening" notice
    await env.mgr.end((1000, 77))


async def test_classic_bare_name_not_addressed_is_silent(env, monkeypatch):
    s = await start(env)
    before = list(env.chan.sent)
    say(monkeypatch, Heard("สลิด", False))
    await env.mgr.handle_segment(s, segment(10))
    assert s.engaged is False and s.pending_wake == {}
    assert env.chan.sent == before and env.vc.played == []
    await env.mgr.end((1000, 77))


async def test_classic_name_with_words_is_a_call_whatever_name_called_says(env, monkeypatch):
    s = await start(env)
    say(monkeypatch, Heard("สลิด เมื่อวานตกลงอะไร", False))
    answered = []

    async def answer(sess, member, name, question):
        answered.append(question)

    monkeypatch.setattr(env.mgr, "answer", answer)
    await env.mgr.handle_segment(s, segment(10))
    assert s.engaged is True and s.engaged_bare is False and answered == ["เมื่อวานตกลงอะไร"]
    await env.mgr.end((1000, 77))


# ------------------------------------------------------------------ pure helpers
def test_bare_name_detection():
    assert bare_name("สลิด", "")
    assert bare_name("ครับ สลิด", "")
    assert bare_name("Hey Salid", "")
    assert bare_name("สลิดคะ", "")
    assert not bare_name("Salid, what's the status?", "what's the status?")
    assert not bare_name("สลิด สรุปให้หน่อย", "สรุปให้หน่อย")
    assert not bare_name("เมื่อวานตกลงอะไร", None)


def stats(voiced, rms=900.0, cv=0.6, fraction=0.7):
    return SpeechStats(1.0, rms, voiced, fraction, cv)


def test_bare_name_needs_more_voiced_audio_than_a_sentence():
    st = Settings(_env_file=None)
    assert st.name_bare_min_voiced_seconds == 0.45
    assert not looks_like_speech(stats(0.30), st, bare=True)[0]
    assert looks_like_speech(stats(0.50), st, bare=True)[0]
    assert not looks_like_speech(stats(0.30), st)[0]  # a sentence: the old 0.35 s floor
    assert looks_like_speech(stats(0.40), st)[0]
    assert not looks_like_speech(stats(0.40), st, bare=True)[0]


def test_loud_short_shortcut_for_a_bare_name_needs_0_3_voiced():
    st = Settings(_env_file=None)
    loud = dict(rms=2500.0)
    assert looks_like_speech(stats(0.16, **loud), st)[0]
    assert not looks_like_speech(stats(0.16, **loud), st, bare=True)[0]
    assert looks_like_speech(stats(0.30, **loud), st, bare=True)[0]


async def test_a_short_bare_clip_is_rejected_by_the_gate(lv, monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="salid.meeting")
    s, live = await lv.join(5)
    calls = record_replies(s)
    say(monkeypatch, Heard("สลิด", True))
    soft = array("h", [300] * (len(SPEECH_FRAME) // 2)).tobytes()
    short = (SPEECH_FRAME + soft) * 10  # 0.4 s, voiced and varying, not loud
    await lv.mgr.handle_segment(s, mt.Segment(5, short, time.monotonic(), 0.4))
    assert s.engaged is False and calls == [] and s.pending_wake == {}
    assert "rejected: too little voiced audio for a bare name" in caplog.text
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ prompts
def test_prompt_choice_follows_the_reply_language_and_keeps_both_scripts():
    th, en = mt.transcribe_prompt("th"), mt.transcribe_prompt("en")
    assert th == mt.TRANSCRIBE_PROMPT and en == mt.TRANSCRIBE_PROMPT_EN
    assert th != en
    assert "ภาษาไทยหรือภาษาอังกฤษ" in th and "ตามที่พูดจริงทุกคำ" in th and "ไม่ต้องแปล" in th
    assert "Transcribe Thai or English exactly as spoken, verbatim, in its own script" in en
    assert "never translate" in en
    for p in (th, en):
        assert "สลิด" in p and "name_called" in p
    # no priming of the name any more
    assert "ถ้าได้ยินคำนี้ให้เขียนว่า สลิด" not in th and "สะกดว่า สลิด" not in th
    assert "slide" in en and "NOT the name" in en
    assert "If you hear" not in en and "something like the name" not in en
    # a second-tier reply language keeps its prefix on the Thai prompt
    assert mt.transcribe_prompt("ja").endswith(mt.TRANSCRIBE_PROMPT)


def test_transcribe_heard_reads_name_called(monkeypatch):
    from conftest import FakeClient  # noqa: F401
    from test_meeting import FRAME, json_resp

    monkeypatch.setattr(mt, "to_flac", lambda wav: b"flac")
    monkeypatch.undo()  # drop the shim: the real functions
    monkeypatch.setattr(mt, "to_flac", lambda wav: b"flac")
    from test_meeting import FakeClient as FC

    h = mt.transcribe_heard(
        FC([json_resp({"text": " สลิด ", "name_called": True})]), "m", FRAME, "en"
    )
    assert h == Heard("สลิด", True)
    h = mt.transcribe_heard(FC([json_resp({"text": "so"})]), "m", FRAME)
    assert h == Heard("so", False)
    h = mt.transcribe_heard(FC([json_resp({"text": "...", "name_called": True})]), "m", FRAME)
    assert h == Heard("", False)


# ------------------------------------------------------------------ the cue
async def cue_text(lv, heard, lang_en):
    s, live = await lv.join(5)
    sent = capture_cues(live)
    s.live._en = lambda: lang_en
    await s.live._cue(live, heard)
    await lv.mgr.end((1000, 77))
    return sent[0][0]


async def test_cue_text_forbids_inventing_a_question(lv):
    assert "Never invent a question" in FORCE_CUE_EN and "Yes?" in FORCE_CUE_EN
    assert not any("\u0e00" <= ch <= "\u0e7f" for ch in FORCE_CUE_EN)  # English session: no Thai
    assert "ห้ามแต่งคำถามขึ้นเอง" in FORCE_CUE and "คะ?" in FORCE_CUE
    assert await cue_text(lv, "สลิด", True) == FORCE_CUE_EN
    assert await cue_text(lv, "สลิด", False) == FORCE_CUE
    assert await cue_text(lv, "Hey Salid", True) == FORCE_CUE_EN


async def test_cue_adds_what_they_said_only_with_words_beyond_the_name(lv):
    text = await cue_text(lv, "Salid, what's the status?", True)
    assert text.startswith(FORCE_CUE_EN) and "What they said" in text
    text = await cue_text(lv, "วันนี้มีนัดอะไรบ้าง", False)  # a follow-up question (no name)
    assert "ข้อความที่เขาพูดคือ" in text


# ------------------------------------------------------------------ engaged: no automatic pass
def engaged_session(env, s, *, by=10, ago=60.0):
    env.mgr._set_engaged(s, True, by=by)
    s.called_at = time.monotonic() - ago  # the call is long ago, Salid is not speaking


async def test_callers_hmm_and_self_talk_are_not_addressed_and_three_end_it(env, monkeypatch):
    env.bot.s.voice_state_posts = False
    s = await start(env)
    engaged_session(env, s)
    now = time.monotonic()
    env.mgr._engaged_utterance(s, "อืม", now, 10)
    assert s.engaged and s.unaddressed == 1
    until_before = s.engaged_until
    env.mgr._engaged_utterance(s, "เอางานมึงไปทำสิ", now, 10)  # slang, and it even has a "สิ"
    assert s.unaddressed == 2 and s.engaged_until == until_before  # not renewed
    env.mgr._engaged_utterance(s, "เออ ใช่", now, 10)
    assert s.engaged is False
    await env.mgr.end((1000, 77))


@pytest.mark.parametrize("text", ["dude can you pass the file", "you guys ready?", "กูว่าไม่ไหว"])
async def test_self_talk_markers_override_a_question_shape(env, text):
    s = await start(env)
    engaged_session(env, s)
    env.mgr._engaged_utterance(s, text, time.monotonic(), 10)
    assert s.unaddressed == 1
    await env.mgr.end((1000, 77))


async def test_google_and_guru_are_not_self_talk(env):
    s = await start(env)
    engaged_session(env, s, ago=10)
    env.mgr._engaged_utterance(s, "ช่วยค้นใน กูเกิล ให้หน่อย", time.monotonic(), 10)
    assert s.unaddressed == 0
    await env.mgr.end((1000, 77))


async def test_addressing_another_attendee_by_name_is_not_addressed_and_not_renewed(env):
    s = await start(env)
    s.attendee_ids.update({10, 11})
    engaged_session(env, s)
    s.called_at = time.monotonic() - 3  # even inside the 6 s reply window
    until_before = s.engaged_until
    env.mgr._engaged_utterance(s, "บอย ไปกินข้าวกันไหม", time.monotonic(), 10)
    assert s.unaddressed == 1 and s.engaged_until == until_before
    env.mgr._engaged_utterance(s, "แอน ไปกินข้าวกันไหม", time.monotonic(), 10)  # own name: fine
    assert s.unaddressed == 0
    await env.mgr.end((1000, 77))


async def test_a_question_within_20_seconds_of_her_words_is_still_addressed(env):
    s = await start(env)
    engaged_session(env, s, ago=15)
    env.mgr._engaged_utterance(s, "ตกลงมันเป็นยังไง", time.monotonic(), 10)
    assert s.unaddressed == 0 and s.engaged
    await env.mgr.end((1000, 77))


async def test_reply_within_6_seconds_of_her_voice_is_addressed_for_anyone(env):
    s = await start(env)
    engaged_session(env, s, ago=2)
    env.mgr._engaged_utterance(s, "อืม โอเค", time.monotonic(), 11)
    assert s.unaddressed == 0
    await env.mgr.end((1000, 77))


# ------------------------------------------------------------------ bare engagement expires
async def test_bare_engagement_goes_back_to_listening_after_12_seconds(env):
    env.bot.s.voice_state_posts = False
    s = await start(env)
    assert Settings(_env_file=None).engaged_bare_timeout_seconds == 12.0
    env.mgr._set_engaged(s, True, by=10, bare=True)
    assert s.engaged_bare is True
    assert 11.0 < s.engaged_until - time.monotonic() <= 12.0
    env.mgr._consent_tick(s, time.monotonic() + 5)
    assert s.engaged is True
    env.mgr._consent_tick(s, time.monotonic() + 13)
    assert s.engaged is False and s.engaged_bare is False
    await env.mgr.end((1000, 77))


async def test_an_addressed_utterance_makes_a_bare_engagement_a_full_one(env):
    s = await start(env)
    env.mgr._set_engaged(s, True, by=10, bare=True)
    env.mgr._engaged_utterance(s, "ช่วยสรุปหน่อย", time.monotonic(), 10)  # within 6 s of the call
    assert s.engaged_bare is False
    assert s.engaged_until - time.monotonic() > 40
    await env.mgr.end((1000, 77))


async def test_a_bare_name_never_shortens_a_running_conversation(env):
    s = await start(env)
    env.mgr._set_engaged(s, True, by=10)
    full = s.engaged_until
    env.mgr._set_engaged(s, True, by=10, bare=True)
    assert s.engaged_bare is False and s.engaged_until >= full
    await env.mgr.end((1000, 77))


async def test_live_transcript_bare_name_engages_in_bare_mode(lv):
    s, live = await lv.join(5)
    lv.bot.s.live_name_reply_seconds = 0.05
    capture_cues(live)
    for i in range(0, len(SPEECH_PCM), 3840):
        s.live.feed(5, SPEECH_PCM[i : i + 3840])
    s.live._on_heard("สลิด")
    assert s.engaged is True and s.engaged_bare is True
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ hush
@pytest.mark.parametrize(
    "text",
    [
        "เงียบ", "เงียบก่อน", "เงียบไปเลย", "หยุดพูด", "หยุดก่อน", "พอแล้ว", "พอแล้วครับ", "พอก่อน",
        "ฟังอย่างเดียว", "ฟังเฉย ๆ", "ไม่ต้องพูด", "ไม่ต้องตอบ", "หุบปาก", "สลิด เงียบก่อนนะ",
        "เงียบก่อนนะ เดี๋ยวเรียก", "be quiet", "Be quiet for now", "quiet", "Quiet please",
        "shut up", "stop talking", "stop", "Stop.", "that's enough", "enough", "just listen",
        "listen only", "don't answer", "no need to answer", "Salid, be quiet", "ok stop please",
    ],
)  # fmt: skip
def test_is_hush_positives(text):
    assert mt.is_hush(text), text


@pytest.mark.parametrize(
    "text",
    [
        "", "stop the meeting", "ออกประชุม", "enough about that", "quiet down the budget talk",
        "สลิด สรุปให้หน่อย", "ขอบคุณครับ", "the room is quiet today", "stop recording the demo",
        "เมื่อวานเงียบสงบดี", "ok next", "we should not stop here because the deadline is near " * 2,
    ],
)  # fmt: skip
def test_is_hush_negatives(text):
    assert not mt.is_hush(text), text


def hushed_lines(sent, s):
    want = lvm.STATE_HUSHED_EN if s.live._en() else lvm.STATE_HUSHED
    return [t for t, _ in sent if t == want]


async def test_hush_while_speaking_stops_audio_mutes_the_turn_and_goes_quiet(lv):
    s, live = await lv.join(5)
    sent = capture_cues(live)
    s.live._turn_open = True
    lv.mgr._set_engaged(s, True, by=5)
    await asyncio.sleep(0.05)
    sent.clear()
    s.pending_wake[5] = time.monotonic()
    stopped = []
    s.live.stop_playback = lambda: stopped.append(1)
    s.live._on_heard("be quiet")
    await asyncio.sleep(0.05)
    assert stopped and s.live._muted_turn is True
    assert s.engaged is False and s.pending_wake == {}
    assert s.hushed_until > time.monotonic() + 80
    assert len(hushed_lines(sent, s)) == 1
    # the classic transcript of the same utterance and a second hush: still one state line
    s.live._on_heard(" please")
    lv.mgr._hush(s)  # a second hush: no second state line
    await asyncio.sleep(0.05)
    assert len(hushed_lines(sent, s)) == 1
    await lv.mgr.end((1000, 77))


async def test_classic_hush_makes_no_reply_and_state_line_is_sent_once(lv, monkeypatch):
    s, live = await lv.join(5)
    sent = capture_cues(live)
    calls = record_replies(s)
    lv.mgr._set_engaged(s, True, by=5)
    say(monkeypatch, Heard("เงียบก่อนนะ", False))
    posted = list(lv.chan.sent)
    await lv.mgr.handle_segment(s, seg())
    s.live._on_heard("เงียบก่อนนะ")  # the Live transcript of the same utterance
    await asyncio.sleep(0.05)
    assert s.engaged is False and calls == []
    assert len(hushed_lines(sent, s)) == 1
    assert lv.vc.played == []
    assert not any("เงียบ" in m for m in lv.chan.sent[len(posted) :])  # no acknowledgement
    await lv.mgr.end((1000, 77))


async def test_hush_cancels_a_pending_forced_turn(lv):
    s, live = await lv.join(5)
    sent = capture_cues(live)
    task = asyncio.create_task(
        s.live._force_reply(time.monotonic(), "Salid, be", None, "be", wait=0.3)
    )
    await asyncio.sleep(0.05)
    lv.mgr._hush(s)
    await task
    await asyncio.sleep(0.05)
    assert not [t for t, done in sent if done]  # no cue (turn_complete=True) was sent
    await lv.mgr.end((1000, 77))


async def test_during_the_hold_a_bare_name_does_nothing_even_corroborated(lv, monkeypatch):
    s, live = await lv.join(5)
    calls = record_replies(s)
    lv.mgr._hush(s)
    say(monkeypatch, Heard("สลิด", True))
    await lv.mgr.handle_segment(s, seg())
    assert s.engaged is False and calls == [] and s.pending_wake == {}
    s.live._on_heard("สลิด")  # the Live transcript path too
    assert s.engaged is False
    assert s.hushed_until > time.monotonic()
    await lv.mgr.end((1000, 77))


async def test_during_the_hold_name_with_a_request_answers_and_clears_the_hold(lv, monkeypatch):
    s, live = await lv.join(5)
    calls = record_replies(s)
    lv.mgr._hush(s)
    say(monkeypatch, Heard("Salid, read the last decision", True))
    await lv.mgr.handle_segment(s, seg())
    assert s.hushed_until == 0.0 and s.engaged is True
    assert len(calls) == 1 and calls[0][1]["question"] == "read the last decision"
    assert s.live._hushed is False
    await lv.mgr.end((1000, 77))


async def test_during_the_hold_other_talk_is_not_an_engaged_conversation(lv, monkeypatch):
    s, live = await lv.join(5)
    calls = record_replies(s)
    lv.mgr._hush(s)
    say(monkeypatch, Heard("แล้วเดือนที่แล้วล่ะเท่าไหร่", False))
    await lv.mgr.handle_segment(s, seg())
    assert s.engaged is False and calls == [] and s.unaddressed == 0
    await lv.mgr.end((1000, 77))


async def test_the_hold_expires_and_listening_returns(lv, monkeypatch):
    s, live = await lv.join(5)
    sent = capture_cues(live)
    monkeypatch.setattr(lv.mgr.bot.s, "hush_hold_seconds", 90.0, raising=False)
    lv.mgr._hush(s)
    await asyncio.sleep(0.05)
    lv.mgr._consent_tick(s, time.monotonic() + 91)
    await asyncio.sleep(0.05)
    assert s.hushed_until == 0.0 and s.live._hushed is False
    assert lvm.STATE_LISTENING in [t for t, _ in sent] or lvm.STATE_LISTENING_EN in [
        t for t, _ in sent
    ]
    say(monkeypatch, Heard("สลิด", False))
    await lv.mgr.handle_segment(s, seg())
    assert 5 in s.pending_wake  # the normal bare-name rules are back
    await lv.mgr.end((1000, 77))


async def test_classic_mode_hush_stops_playback_and_stays_silent(env, monkeypatch):
    s = await start(env)
    env.mgr._set_engaged(s, True, by=10)
    say(monkeypatch, Heard("be quiet", False))
    before = list(env.chan.sent)
    await env.mgr.handle_segment(s, segment(10))
    assert s.engaged is False and s.hushed_until > time.monotonic()
    assert env.vc.played == [] and env.chan.sent == before
    say(monkeypatch, Heard("สลิด", True))
    await env.mgr.handle_segment(s, segment(10))
    assert s.engaged is False and env.vc.played == []
    await env.mgr.end((1000, 77))


# ------------------------------------------------------------------ Salid's own to-dos are not tasks
def test_summary_prompt_says_salids_own_jobs_are_not_action_items():
    from salid.memory_summary import PROMPT

    sentence = "Things Salid itself was asked to do (look up, remind, record) are not action items."
    assert sentence in PROMPT


def test_ingest_skips_salid_and_the_bot_but_keeps_people():
    from conftest import FixedClockSettings
    from test_coordinator import GUILD, PUBLIC, summary, tracker

    from salid.coo_store import SQLiteDocStore

    tr = tracker(SQLiteDocStore(":memory:"), FixedClockSettings())
    items = summary(
        ("Salid", "ค้นหาข้อมูล flight duty period ตามที่ C สอบถาม", ""),
        ("สลิด", "เรียนรู้การจดจำชื่อคนในทีม", ""),
        ("the bot", "remind everyone", ""),
        ("บอท", "จดบันทึก", ""),
        ("Somchai", "ส่งใบเสนอราคาลูกค้า X", ""),
    )
    new = tr.ingest(GUILD, PUBLIC, "2026-10-14", items, [])
    assert [t_.assignee_name for t_ in new] == ["Somchai"]
    assert len(tr.repo.list(GUILD, status=None)) == 1


# ------------------------------------------------------------------ background work is quiet
class _Msg:
    def __init__(self, log, text):
        self.log, self.text = log, text

    async def edit(self, content=None, **_):
        self.text = content
        self.log.append(("edit", content))


def status_echo(live_voice, fail_edit=False):
    log: list = []

    async def echo(text, **_):
        log.append(("send", text))
        m = _Msg(log, text)
        if fail_edit:

            async def boom(**_k):
                raise RuntimeError("gone")

            m.edit = boom
        return m

    live_voice.echo = echo
    return log


async def long_job(lv, monkeypatch, question=None, answer=None, **kw):
    from test_voice_async import Brains, Q, patch_brain, setup_async

    setup_async(lv, **kw)
    brains = Brains()
    patch_brain(monkeypatch, brains)
    q = question or Q
    if answer is not None:

        async def brain(self, qq):
            await brains.gates.setdefault(qq, asyncio.Event()).wait()
            return {"answer": answer}

        monkeypatch.setattr(lvm.LiveVoice, "_ask_salid", brain)
    s, live = await lv.join(5)
    return s, live, brains, q


async def test_working_re_ask_has_no_say_and_one_status_line_is_edited_in_place(lv, monkeypatch):
    from test_live_voice import tool_msg

    s, live, brains, q = await long_job(lv, monkeypatch)
    log = status_echo(s.live)
    live.push(tool_msg("ask_salid", {"question": q}, id="a"))
    await until(lambda: len(live.tool_responses) == 1)  # hand-off
    live.push(tool_msg("ask_salid", {"question": q}, id="b"))
    await until(lambda: len(live.tool_responses) == 2)
    got = {r[0].id: r[0].response for r in live.tool_responses}
    assert got["a"]["status"] == got["b"]["status"] == "working"
    assert "say" not in got["a"] and "say" not in got["b"]
    brains.release(q)
    await until(lambda: any(k == "edit" and "✅" in x for k, x in log))
    sends = [x for k, x in log if k == "send" and not x.startswith("💬")]
    assert len(sends) == 1 and "⏳" in sends[0]  # ONE line, no "sent to the background" line
    edits = [x for k, x in log if k == "edit"]
    assert any(x.startswith("⏳") and "ยังทำอยู่" in x for x in edits)
    assert edits[-1].startswith("✅") and q[:20] in edits[-1]
    assert not any("ส่งไปทำพื้นหลัง" in x for _, x in log)
    await lv.mgr.end((1000, 77))


async def test_a_failing_status_edit_falls_back_to_a_new_line(lv, monkeypatch):
    from test_live_voice import tool_msg

    s, live, brains, q = await long_job(lv, monkeypatch)
    log = status_echo(s.live, fail_edit=True)
    live.push(tool_msg("ask_salid", {"question": q}, id="a"))
    await until(lambda: len(live.tool_responses) == 1)
    brains.release(q)
    await until(lambda: any(k == "send" and x.startswith("✅") for k, x in log))
    await lv.mgr.end((1000, 77))


async def finish_and_get_cue(lv, monkeypatch, answer, mode=None, expect=True):
    from test_live_voice import tool_msg
    from test_voice_async import cues, model_idle

    kw = {} if mode is None else {"live_result_voice": mode}
    s, live, brains, q = await long_job(lv, monkeypatch, answer=answer, **kw)
    live.push(tool_msg("ask_salid", {"question": q}, id="a"))
    await until(lambda: live.tool_responses)
    model_idle(s)
    brains.release(q)
    if expect:
        await until(lambda: cues(live))
        out = cues(live)[0][0]
    else:
        await asyncio.sleep(0.6)
        out = cues(live)
    await lv.mgr.end((1000, 77))
    return out, q


async def test_long_result_is_one_short_sentence_by_default(lv, monkeypatch):
    text, q = await finish_and_get_cue(lv, monkeypatch, "ผลลัพธ์ยาวมาก " * 30)
    assert "รายละเอียดอยู่ในแชต" in text and "ผลลัพธ์ยาวมาก" not in text
    assert text.startswith("(ผลงานที่ขอไว้เสร็จแล้ว") and q in text


async def test_short_answer_is_spoken_in_full(lv, monkeypatch):
    text, q = await finish_and_get_cue(lv, monkeypatch, "ยอดรวม 1,250 บาท")
    assert "ยอดรวม 1,250 บาท" in text and "ห้ามเรียบเรียงใหม่" in text


async def test_full_mode_keeps_the_whole_result(lv, monkeypatch):
    long = "ผลลัพธ์ยาวมาก " * 30
    text, q = await finish_and_get_cue(lv, monkeypatch, long, mode="full")
    assert "ผลลัพธ์ยาวมาก" in text and "รายละเอียดอยู่ในแชต" not in text


async def test_off_mode_announces_nothing_by_voice(lv, monkeypatch):
    out, q = await finish_and_get_cue(lv, monkeypatch, "x" * 300, mode="off", expect=False)
    assert out == []
    assert Settings(_env_file=None).live_result_voice == "short"


def test_short_result_cue_constants_and_instruction_say_nothing_about_progress():
    from salid.live_voice import build_instruction

    assert "รายละเอียดอยู่ในแชต" in lvm.RESULT_CUE_SHORT
    assert "result is in the chat" in lvm.RESULT_CUE_SHORT_EN
    settings = Settings(_env_file=None)
    en = " ".join(build_instruction(settings, "en").split())
    th = build_instruction(settings, "th")
    assert "say nothing about it (not even “one moment”) and go back to listening" in en
    assert "แม้แต่ “แป๊บนึงนะคะ”" in th and "กลับไปฟังต่อ" in th
