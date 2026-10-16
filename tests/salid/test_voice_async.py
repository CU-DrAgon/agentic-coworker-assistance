# ruff: noqa: F811, F401
"""Voice v0.9.0: the spoken brain has the chat tools, and long jobs run in the background while the
room keeps talking (result announced at the next pause, always posted to the text chat). Offline."""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from conftest import EXEC_ID, GUILD, fc_resp, scripted, text_resp
from test_live_voice import (
    FillerStub,
    asked,
    audio_msg,
    interrupted_msg,
    lv,
    pcm48,
    tool_msg,
    until,
)

from salid import live_voice as lvm
from salid.i18n import t
from salid.live_voice import LiveVoice, build_instruction
from salid.qa import ask

Q = "ทำสรุปยอดไตรมาสให้หน่อย"
LINE = "เดี๋ยวสลิดทำให้นะคะ ได้แล้วจะบอกค่ะ"


def setup_async(lv, **kw):
    """Short windows for the tests; the hand-off line comes from a stub."""
    lv.bot.fillers = FillerStub(LINE, "ยังอยู่ค่ะ")
    lv.bot.s.live_foreground_seconds = kw.pop("fg", 0.05)
    lv.bot.s.live_result_pause_seconds = kw.pop("pause", 0.1)
    lv.bot.s.live_result_wait_seconds = kw.pop("wait", 3.0)
    for k, v in kw.items():
        setattr(lv.bot.s, k, v)


class Brains:
    """`_ask_salid` stand-in: per-question gates, optional failure, records starts."""

    def __init__(self, fail=()):
        self.gates: dict[str, asyncio.Event] = {}
        self.started, self.fail = [], set(fail)

    async def __call__(self, lv_self, q):
        self.started.append(q)
        await self.gates.setdefault(q, asyncio.Event()).wait()
        if q in self.fail:
            raise RuntimeError("backend down")
        return {"answer": f"ตอบ {q}"}

    def release(self, q):
        self.gates.setdefault(q, asyncio.Event()).set()


def patch_brain(monkeypatch, brains):
    monkeypatch.setattr(LiveVoice, "_ask_salid", lambda self, q: brains(self, q))


def model_idle(s):
    """The Live model has finished saying the hand-off line."""
    s.live._awaiting_answer = False
    s.live._turn_audio = False
    s.live._playing_src = s.live._source = None  # the filler clip (fake player) has played out
    s.live._filler_until = 0.0


def cues(live):
    return [n for n in live.notes if "ผลงานที่ขอไว้เสร็จแล้ว" in n[0] or "requested job" in n[0]]


async def talk(s, seconds, uid=5):
    """The room is talking for `seconds`."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        s.live.feed(uid, pcm48(2000))
        await asyncio.sleep(0.02)


# ------------------------------------------------------------------ foreground
async def test_a_quick_answer_is_returned_as_before(lv, monkeypatch):
    setup_async(lv, fg=5.0)
    brains = Brains()
    patch_brain(monkeypatch, brains)
    brains.release(Q)
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": Q}))
    await until(lambda: live.tool_responses)
    assert live.tool_responses[0][0].response == {"answer": f"ตอบ {Q}"}
    await until(lambda: f"💬 **สลิด:** ตอบ {Q}" in lv.chan.sent)
    await asyncio.sleep(0.2)
    assert cues(live) == []  # answered at once: no announcement
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ background hand-off
async def test_slow_job_returns_working_with_a_line_and_the_live_model_is_not_blocked(
    lv, monkeypatch
):
    setup_async(lv)
    brains = Brains()
    patch_brain(monkeypatch, brains)
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": Q}, id="a"))
    await until(lambda: live.tool_responses)
    res = live.tool_responses[0][0].response
    assert res["status"] == "working" and res["say"] == LINE and len(res["job"]) == 4
    first = next(c for c in lv.bot.fillers.calls if c.get("facts"))
    assert first["stage"] == "first" and first["kind"] == "voice"
    assert first["facts"] == [lvm.BG_FACT["th"]]
    # the room goes on: another tool call is answered while the job still runs
    live.push(tool_msg("post_note", {"text": "x"}, id="n"))
    await until(lambda: len(live.tool_responses) == 2)
    assert brains.started == [Q] and not [x for x in lv.chan.sent if x.startswith("💬")]
    brains.release(Q)
    await lv.mgr.end((1000, 77))


async def test_handoff_line_falls_back_to_the_pool_in_both_languages(lv, monkeypatch):
    setup_async(lv)
    lv.bot.fillers = FillerStub(boom=True)
    brains = Brains()
    patch_brain(monkeypatch, brains)
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": Q}))
    await until(lambda: live.tool_responses)
    say = live.tool_responses[0][0].response["say"]
    assert any(say.startswith(x[:12]) for x in lvm.FILLER_BG["th"])
    assert "คุยเรื่องอื่น" in say or "ทำอยู่" in say or "เบื้องหลัง" in say
    assert all(x for x in lvm.FILLER_BG["en"])
    brains.release(Q)
    await lv.mgr.end((1000, 77))


async def test_finished_job_is_posted_to_chat_and_announced_only_after_a_pause(lv, monkeypatch):
    setup_async(lv, pause=0.15)
    brains = Brains()
    patch_brain(monkeypatch, brains)
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": Q}))
    await until(lambda: live.tool_responses)
    model_idle(s)
    job = live.tool_responses[0][0].response["job"]
    busy = asyncio.create_task(talk(s, 0.5))  # the room is busy when the job finishes
    await asyncio.sleep(0.05)
    brains.release(Q)
    await until(lambda: f"💬 **สลิด:** ตอบ {Q}" in lv.chan.sent)  # chat first, always
    await asyncio.sleep(0.2)
    assert cues(live) == []  # no announcement while people talk
    await busy
    await until(lambda: cues(live))  # the room went quiet
    text, complete = cues(live)[0]
    assert complete is True and text.startswith("(ผลงานที่ขอไว้เสร็จแล้ว")
    assert f"งาน {job}" in text and Q in text and f"ผลลัพธ์: ตอบ {Q}" in text
    assert "ห้ามเรียบเรียงใหม่" in text
    assert len(cues(live)) == 1
    await lv.mgr.end((1000, 77))


async def test_result_longer_than_the_cap_is_cut_in_the_cue_not_in_chat(lv, monkeypatch):
    setup_async(lv)

    async def long(self, q):
        await asyncio.sleep(0.2)
        return {"answer": "ก" * 900}

    monkeypatch.setattr(LiveVoice, "_ask_salid", long)
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": Q}))
    await until(lambda: live.tool_responses)
    model_idle(s)
    await until(lambda: cues(live))
    assert "ก" * lvm.ANSWER_MAX_CHARS in cues(live)[0][0]
    assert "ก" * (lvm.ANSWER_MAX_CHARS + 1) not in cues(live)[0][0]
    await lv.mgr.end((1000, 77))


async def test_english_room_gets_the_english_cue(lv, monkeypatch):
    setup_async(lv)
    brains = Brains()
    patch_brain(monkeypatch, brains)
    s, live = await lv.join(5)
    s.language = "en"
    live.push(tool_msg("ask_salid", {"question": "Draft the Q3 summary"}))
    await until(lambda: live.tool_responses)
    model_idle(s)
    brains.release("Draft the Q3 summary")
    await until(lambda: cues(live))
    assert cues(live)[0][0].startswith("(The requested job is done.")
    assert "💬 **Salid:** ตอบ Draft the Q3 summary" in lv.chan.sent
    await lv.mgr.end((1000, 77))


async def test_no_pause_in_time_means_chat_only_and_no_speech(lv, monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="salid.live_voice")
    setup_async(lv, pause=0.2, wait=0.3)
    brains = Brains()
    patch_brain(monkeypatch, brains)
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": Q}))
    await until(lambda: live.tool_responses)
    model_idle(s)
    brains.release(Q)
    await talk(s, 0.6)  # never quiet for 0.2 s within the 0.3 s wait
    assert f"💬 **สลิด:** ตอบ {Q}" in lv.chan.sent and cues(live) == []
    assert any("not spoken" in r.getMessage() for r in caplog.records)
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ mouth gate
async def test_announcement_audio_passes_the_mouth_gate_while_listening(lv, monkeypatch):
    setup_async(lv)
    lv.bot.s.live_mouth_gate = True
    lv.bot.s.live_call_grace_seconds = 0.15
    brains = Brains()
    patch_brain(monkeypatch, brains)
    s, live = await lv.join(5)
    lv.mgr._set_engaged(s, True)
    live.push(tool_msg("ask_salid", {"question": Q}))
    await until(lambda: live.tool_responses)
    model_idle(s)
    lv.mgr._set_engaged(s, False)  # the room is back in listening state
    s.live._last_engaged_at = -1e9
    assert not s.live._gate_open()
    brains.release(Q)
    await until(lambda: cues(live))
    assert s.live._gate_open()  # the cue counts as a name hit (60 s grace for follow-ups)
    live.push(audio_msg(), audio_msg())
    await until(lambda: lv.vc.played)  # not held, not dropped
    assert s.live._gated_turn is False
    await lv.mgr.end((1000, 77))


async def test_without_a_cue_the_same_audio_is_still_muted_while_listening(lv):
    lv.bot.s.live_mouth_gate = True
    lv.bot.s.live_call_grace_seconds = 0.05
    s, live = await lv.join(5)
    live.push(audio_msg())
    await asyncio.sleep(0.15)
    live.push(audio_msg())
    await asyncio.sleep(0.15)
    assert lv.vc.played == []
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ known-long tools
async def test_a_known_long_tool_goes_to_the_background_at_once(lv, asked, monkeypatch):
    setup_async(lv, fg=30.0)  # the window would be 30 s: the tool name decides
    release = threading.Event()

    def fake_ask(*a, **kw):
        kw["on_tool_call"]("create_deck")
        release.wait(5)
        return SimpleNamespace(text="ทำสไลด์เสร็จแล้ว", seconds=1.0, rounds=1, attachments=[])

    monkeypatch.setattr("salid.qa.ask", fake_ask)
    s, live = await lv.join(5)
    t0 = time.monotonic()
    live.push(tool_msg("ask_salid", {"question": "ทำสไลด์สรุปผลประชุม"}))
    await until(lambda: live.tool_responses, timeout=3)
    assert time.monotonic() - t0 < 3 and live.tool_responses[0][0].response["status"] == "working"
    model_idle(s)
    release.set()
    await until(lambda: cues(live))
    assert "ทำสไลด์เสร็จแล้ว" in cues(live)[0][0]
    await lv.mgr.end((1000, 77))


def test_background_tools_are_the_documented_list():
    assert {
        "deep_research", "make_promo_kit", "create_deck", "market_benchmark", "business_framework",
        "idea_export", "make_quotation", "revise_quotation", "financial_projection", "meeting_document", "contest_plan", "contest_pitch",
        "wiki_write", "wiki_publish",
    } == lvm.BACKGROUND_TOOLS  # fmt: skip


async def test_a_quick_tool_does_not_trigger_the_background(lv, monkeypatch):
    setup_async(lv, fg=5.0)

    def fake_ask(*a, **kw):
        kw["on_tool_call"]("list_tasks")
        return SimpleNamespace(text="มีสามงานค่ะ", seconds=0.1, rounds=1, attachments=[])

    monkeypatch.setattr("salid.qa.ask", fake_ask)
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": "งานของฉันมีอะไรบ้าง"}))
    await until(lambda: live.tool_responses)
    assert live.tool_responses[0][0].response == {"answer": "มีสามงานค่ะ"}
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ job limits
async def test_the_same_question_again_says_still_working(lv, monkeypatch):
    setup_async(lv)
    brains = Brains()
    patch_brain(monkeypatch, brains)
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": Q}, id="a"))
    await until(lambda: len(live.tool_responses) == 1)
    live.push(tool_msg("ask_salid", {"question": Q + "?"}, id="b"))
    await until(lambda: len(live.tool_responses) == 2)
    again = live.tool_responses[1][0].response
    assert again["status"] == "working" and again["say"] == "ยังทำอยู่ค่ะ อีกแป๊บนะคะ"
    assert brains.started == [Q]
    brains.release(Q)
    await lv.mgr.end((1000, 77))


async def test_a_fourth_job_is_refused_until_one_finishes(lv, monkeypatch):
    setup_async(lv)
    brains = Brains()
    patch_brain(monkeypatch, brains)
    s, live = await lv.join(5)
    qs = ["ยอดขายเดือนนี้", "ใครลางานพรุ่งนี้", "ร่างอีเมลถึงลูกค้า", "สรุปประชุมเมื่อวาน"]
    for i, q in enumerate(qs):
        live.push(tool_msg("ask_salid", {"question": q}, id=str(i)))
        await until(lambda n=i: len(live.tool_responses) == n + 1)
    got = [r[0].response for r in live.tool_responses]
    assert [g.get("status") for g in got[:3]] == ["working"] * 3
    assert got[3]["error"] and got[3]["say"] == "ขอให้งานก่อนหน้าเสร็จก่อนนะคะ"
    assert brains.started == qs[:3]
    brains.release(qs[0])
    await until(lambda: f"💬 **สลิด:** ตอบ {qs[0]}" in lv.chan.sent)
    live.push(tool_msg("ask_salid", {"question": qs[3]}, id="again"))
    await until(lambda: len(live.tool_responses) == 5)
    assert live.tool_responses[4][0].response["status"] == "working"  # a slot is free again
    for q in qs:
        brains.release(q)
    await lv.mgr.end((1000, 77))


async def test_barge_in_does_not_cancel_a_running_job(lv, monkeypatch):
    setup_async(lv)
    brains = Brains()
    patch_brain(monkeypatch, brains)
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": Q}))
    await until(lambda: live.tool_responses)
    live.push(interrupted_msg())
    s.live._stop_requested()
    s.live._interrupt()
    await asyncio.sleep(0.1)
    job = next(iter(s.live._jobs.values()))
    assert not job.task.done()
    model_idle(s)
    brains.release(Q)
    await until(lambda: f"💬 **สลิด:** ตอบ {Q}" in lv.chan.sent)
    await until(lambda: cues(live))
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ failures
async def test_a_failed_background_job_is_a_warning_in_chat_and_an_honest_spoken_line(
    lv, monkeypatch
):
    setup_async(lv)
    brains = Brains(fail=[Q])
    patch_brain(monkeypatch, brains)
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": Q}))
    await until(lambda: live.tool_responses)
    model_idle(s)
    brains.release(Q)
    await until(lambda: any("⚠️" in x and "backend down" in x for x in lv.chan.sent))
    await until(lambda: cues(live) or any("ทำไม่สำเร็จ" in n[0] for n in live.notes))
    text = next(n[0] for n in live.notes if "ทำไม่สำเร็จ" in n[0])
    assert t("voice_tool_failed_say", "th") in text and "ผลลัพธ์:" not in text
    await lv.mgr.end((1000, 77))


async def test_a_job_that_times_out_in_the_background_is_reported_the_same_way(lv, monkeypatch):
    setup_async(lv, live_ask_timeout_seconds=0.2)
    brains = Brains()
    patch_brain(monkeypatch, brains)
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": Q}))
    await until(lambda: live.tool_responses)
    model_idle(s)
    await until(lambda: any("⚠️" in x and "timeout" in x for x in lv.chan.sent))
    await until(lambda: any("ทำไม่สำเร็จ" in n[0] for n in live.notes))
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ files from tools
async def test_files_made_by_tools_go_to_the_chat_post_and_the_answer_says_so(lv, asked):
    setup_async(lv, fg=5.0)
    sent = []

    async def send(text, files=None, **_):
        sent.append((text, [f.filename for f in files or []]))

    lv.chan.send = send

    def fake_ask(*a, **kw):
        asked.update(kw)
        return SimpleNamespace(
            text="ทำสไลด์ให้แล้วค่ะ", seconds=1.0, rounds=2, attachments=[("deck.pptx", b"PK")]
        )

    import salid.qa

    salid.qa.ask = fake_ask  # restored by the `asked` fixture's monkeypatch
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": "ทำสไลด์"}))
    await until(lambda: live.tool_responses)
    res = live.tool_responses[0][0].response
    assert res["answer"] == "ทำสไลด์ให้แล้วค่ะ " + t("voice_files_say", "th")
    assert "files" not in res  # never into the Live model
    await until(lambda: any(f for _, f in sent))
    text, names = next(x for x in sent if x[1])
    assert names == ["deck.pptx"] and "💬 **สลิด:** ทำสไลด์ให้แล้วค่ะ" in text
    assert t("voice_files_echo", "th") in text
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ chat flags
async def test_voice_passes_the_chat_flags_decks_and_no_files(lv, asked, monkeypatch):
    setup_async(lv, fg=5.0)
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": "a"}, id="1"))
    await until(lambda: live.tool_responses)
    assert asked["deck"] is True and callable(asked["on_tool_call"]) and asked["spoken"] is True
    await lv.mgr.end((1000, 77))


async def test_decks_flag_off_in_the_guild_is_off_in_voice_too(lv, asked, monkeypatch):
    setup_async(lv, fg=5.0)
    real = lv.bot.team.enabled
    monkeypatch.setattr(
        lv.bot.team, "enabled", lambda gid, f: False if f == "decks" else real(gid, f)
    )
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": "a"}, id="1"))
    await until(lambda: live.tool_responses)
    assert asked["deck"] is False
    await lv.mgr.end((1000, 77))


def test_ask_reports_every_tool_call_to_the_hook(seeded, settings):
    names = []
    llm = scripted(fc_resp(("flagged_receipts", {})), text_resp("สรุป"))
    a = ask(settings, seeded, "x", guild_id=GUILD, is_exec=True, user_id=EXEC_ID, model=llm,
            spoken=True, on_tool_call=names.append)  # fmt: skip
    assert a.text == "สรุป" and names == ["flagged_receipts"]


def test_a_failing_hook_never_breaks_the_answer(seeded, settings):
    def boom(name):
        raise RuntimeError("hook")

    llm = scripted(fc_resp(("flagged_receipts", {})), text_resp("สรุป"))
    a = ask(settings, seeded, "x", guild_id=GUILD, is_exec=True, user_id=EXEC_ID, model=llm,
            on_tool_call=boom)  # fmt: skip
    assert a.text == "สรุป"


# ------------------------------------------------------------------ prompt + settings
def test_prompts_explain_working_status_and_the_result_message_in_both_languages():
    from conftest import FixedClockSettings

    s = FixedClockSettings()
    th, en = build_instruction(s, "th"), build_instruction(s, "en")
    assert "status=working" in th and "ไม่ต้องรอ" in th and "(ผลงานที่ขอไว้เสร็จแล้ว" in th
    assert "ยังทำอยู่" in th and "ห้ามเรียบเรียงใหม่" in th and "ห้ามพูดคำเติมเอง" in th
    assert "status=working" in en and "do not wait" in en and "(The requested job is done" in en
    assert "still being done" in en and "do not rephrase" not in en.split("status=working")[0]
    assert "superseded" not in th and "superseded" not in en


def test_async_settings_defaults():
    from salid.config import Settings

    f = Settings.model_fields
    assert f["live_foreground_seconds"].default == 12.0
    assert f["live_max_jobs"].default == 3
    assert f["live_result_pause_seconds"].default == 1.5
    assert f["live_result_wait_seconds"].default == 90.0
    assert f["live_slow_answer_seconds"].default == 40.0  # still accepted, no longer used
