# ruff: noqa: F811, F401
"""v0.9.2 hotfixes seen in a real room: a loud but short "สลิด!" is a call; a stale model turn /
tool answer no longer blocks a result announcement; the hand-off line is played when the model
stays silent; the quotation refusal says where rights come from."""

import asyncio
import time

import pytest
from test_live_voice import FillerStub, lv, tool_msg, until
from test_voice_async import Brains, patch_brain, setup_async

from salid import live_voice as lvm
from salid import meeting as mt
from salid.i18n import t


def test_loud_short_name_call_passes_the_speech_gate():
    s = mt.SpeechStats(duration_s=0.70, rms=7476, voiced_s=0.22, voiced_fraction=0.31, cv=1.2)
    ok, _ = mt.looks_like_speech(s, object())
    assert ok  # the real rejection of 2026-10-16 17:07:58
    quiet = mt.SpeechStats(duration_s=0.70, rms=400, voiced_s=0.22, voiced_fraction=0.31, cv=1.2)
    assert not mt.looks_like_speech(quiet, object())[0]  # short AND quiet: still rejected
    hum = mt.SpeechStats(duration_s=1.5, rms=3000, voiced_s=0.5, voiced_fraction=0.9, cv=0.05)
    assert not mt.looks_like_speech(hum, object())[0]  # loud steady noise: rejected (cv)


@pytest.mark.asyncio
async def test_a_stale_tool_answer_does_not_block_the_pause(lv, monkeypatch):
    setup_async(lv)
    monkeypatch.setattr(lvm, "STALE_TURN_S", 0.05)
    s, live = await lv.join(5)
    v = s.live
    v._awaiting_answer = True
    v._awaiting_since = time.monotonic()
    assert not v._quiet(0.0)  # fresh: the model may still answer out loud
    await asyncio.sleep(0.08)
    assert v._quiet(0.0)  # stale: the announcement may go ahead
    v._awaiting_answer = False
    v._turn_open, v._turn_open_since = True, time.monotonic()
    assert not v._quiet(0.0)
    await asyncio.sleep(0.08)
    assert v._quiet(0.0)
    await lv.mgr.end((1000, 77))


@pytest.mark.asyncio
async def test_noise_frames_do_not_count_as_talking(lv):
    setup_async(lv)
    s, live = await lv.join(5)
    v = s.live
    now = time.monotonic()
    v._frames = [(now, 50.0)]  # an open mic sending near-silence
    assert v._quiet(0.5)
    v._frames = [(now, 900.0)]  # someone talking
    assert not v._quiet(0.5)
    await lv.mgr.end((1000, 77))


@pytest.mark.asyncio
async def test_handoff_line_is_played_when_the_model_stays_silent(lv, monkeypatch):
    setup_async(lv, live_handoff_say_seconds=0.02)
    brains = Brains()
    patch_brain(monkeypatch, brains)
    s, live = await lv.join(5)
    v = s.live
    played = []

    async def render(text, lang, cache=False):
        return b"\x00\x00" * 960

    async def play(pcm):
        played.append(pcm)
        return True

    monkeypatch.setattr(v, "_render", render)
    monkeypatch.setattr(v, "_play_filler", play)
    s.engaged = True
    res = await v._one_brain("ทำสรุปยอดไตรมาสให้หน่อย")
    assert res["status"] == "working"
    await asyncio.sleep(0.1)
    assert len(played) == 1  # the model said nothing: the clip was played
    played.clear()
    v._real_audio_at = time.monotonic() + 10  # the model did speak: nothing is played
    res = await v._one_brain("อีกงานหนึ่ง")
    assert res["status"] == "working"
    await asyncio.sleep(0.1)
    assert played == []
    brains.release("ทำสรุปยอดไตรมาสให้หน่อย")
    brains.release("อีกงานหนึ่ง")
    await lv.mgr.end((1000, 77))


def test_quotation_refusal_says_where_rights_come_from():
    th = t("quote_refusal", "th", roles="")
    en = t("quote_refusal", "en", roles="")
    assert "เปลี่ยนสิทธิ์ให้ใครเองไม่ได้" in th and "Discord" in th
    assert "cannot grant or change" in en
