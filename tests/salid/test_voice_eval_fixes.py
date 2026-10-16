# ruff: noqa: F811, F401
"""Fixes found by the live voice evaluation (tests/salid/live_eval): wake-word variants the speech
model produces, the spoken answer to a bare name, the Live force cue and the Live prompt rules."""

import asyncio
import time

import pytest
from test_live_voice import lv, until  # noqa: F401  (fixtures / helpers)
from test_meeting import env, segment, start

from salid import meeting as mt
from salid.config import Settings
from salid.live_voice import build_instruction
from salid.meeting import parse_wake
from salid.naming import find_call


@pytest.mark.parametrize(
    "text",
    ["เฮ้ โซรี่", "เฮ้ ซิรี่ วันนี้วันอะไร", "หวัดดีครับ สนิท", "เฮ้ Siri", "สวัสดีค่ะ ซิริ", "hey siri"],
)
def test_misheard_siri_after_a_greeting_is_a_call(text):
    assert find_call(text) is not None


@pytest.mark.parametrize(
    "text",
    ["เรากับเขาสนิทกันมาก", "ซอรี่นะ ขอโทษที", "โซรี่ เป็นชื่อแมว", "ดูซีรี่เรื่องนี้ยัง", "เฮ้ย ไปกินข้าวกัน",
     "เมื่อวานกินข้าวที่ไหน"],
)  # fmt: skip
def test_chatter_is_not_a_call(text):
    assert parse_wake(text) is None


async def test_bare_name_gets_a_spoken_yes_when_voice_replies_are_on(env, monkeypatch):
    env.bot.s.voice_reply = True
    s = await start(env)
    said = []

    async def fake_say(self_, sess, text):
        said.append(text)

    monkeypatch.setattr(type(env.mgr), "_say_text", fake_say)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "ครับ สลิด")
    await env.mgr.handle_segment(s, segment(10))
    await asyncio.sleep(0.05)
    assert said == ["ค่ะ? ว่ามาได้เลยค่ะ"]
    await env.mgr.end((1000, 77), "shutdown")


async def test_live_cue_waits_while_a_tool_answer_is_pending(lv):
    s, live = await lv.join(5)
    lv.bot.s.live_force_reply_seconds = 0.05
    sent = []

    async def send_client_content(turns=None, turn_complete=True):
        sent.append(turns)

    live.send_client_content = send_client_content
    s.live._awaiting_answer = True  # ask_salid is running: the real answer comes next
    s.live.ensure_reply(time.monotonic())
    await asyncio.sleep(0.2)
    assert sent == []
    await lv.mgr.end((1000, 77))


def test_force_cue_default_is_longer_than_the_models_own_first_audio():
    assert Settings().live_force_reply_seconds >= 4.0


def test_live_prompt_keeps_quiet_for_chatter_and_lookalike_words():
    th = build_instruction(Settings(), "th")
    assert "เมื่อวานกินข้าวที่ไหน" in th and "“สนิท” ใน “เขาสนิทกันมาก”" in th


async def test_live_cue_carries_the_transcript_of_what_was_said(lv):
    s, live = await lv.join(5)
    lv.bot.s.live_force_reply_seconds = 0.05
    sent = []

    async def send_client_content(turns=None, turn_complete=True):
        sent.append(turns.parts[0].text)

    live.send_client_content = send_client_content
    s.live.ensure_reply(time.monotonic(), "สลิด วันนี้วันอะไร")
    await until(lambda: sent)
    assert "เรียกชื่อ" in sent[0] and "สลิด วันนี้วันอะไร" in sent[0]
    await lv.mgr.end((1000, 77))


def test_live_prompt_pins_the_date_answer_to_the_given_date():
    assert "ตอบจากวันที่นี้เท่านั้น" in build_instruction(Settings(), "th")
