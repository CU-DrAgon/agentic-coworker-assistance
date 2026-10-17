# ruff: noqa: F811, F401
"""Listen-first voice (v0.8.0): dismissal phrases, engaged drop-out, visible state, slow answers
that are never lost, one brain at a time, the spoken tool list, answer/failure echo, prompts and
the duplicate memory search. Fully offline."""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from conftest import EXEC_ID, GUILD, FixedClockSettings, fc_resp, scripted, text_resp
from test_live_voice import (
    FillerStub,
    lv,
    play_source,
    slow_tool,
    tool_msg,
    tts_spy,
    until,
)
from test_meeting import ME, env, make_member, segment, start

from salid import live_voice as lvm
from salid import meeting as mt
from salid import search as sr
from salid.i18n import t
from salid.live_voice import LiveVoice, build_instruction, same_question
from salid.qa import MAX_ROUNDS, SPOKEN_MAX_ROUNDS, ask


# ------------------------------------------------------------------ dismissal phrases
@pytest.mark.parametrize(
    "text",
    [
        "ขอบคุณครับ", "โอเคครับ ขอบคุณมากครับ", "พอแล้วครับ", "จบแค่นี้", "thanks, that's all",
        "โอเค", "โอเคครับ", "คนต่อไป", "ต่อไป", "เชิญคนต่อไป", "เชิญต่อ", "ไปต่อ", "เอาล่ะ",
        "ok next", "moving on", "next", "next person", "let's continue", "Okay, moving on.",
    ],
)  # fmt: skip
def test_dismissal_phrases_old_and_new(text):
    assert mt.MeetingManager.is_dismissal(text)


@pytest.mark.parametrize(
    "text",
    [
        "ต่อไปเราจะคุยเรื่องงบประมาณของไตรมาสหน้ากัน", "next quarter we need a budget",
        "โอเค แล้วงบเท่าไหร่", "ออกประชุม", "leave the meeting", "ขอบคุณครับ แล้วเรื่องงบล่ะ",
    ],
)  # fmt: skip
def test_whole_short_utterances_only(text):
    assert not mt.MeetingManager.is_dismissal(text)


# ------------------------------------------------------------------ engaged drop-out
async def quiet_session(env, **settings):
    env.bot.s.meeting_text_echo = False
    env.bot.s.voice_state_posts = False
    for k, v in settings.items():
        setattr(env.bot.s, k, v)
    return await start(env)


def test_default_engaged_timeout_is_45_seconds():
    from salid.config import Settings

    assert Settings.model_fields["engaged_timeout_seconds"].default == 45.0
    assert Settings.model_fields["live_ask_timeout_seconds"].default == 150.0
    assert Settings.model_fields["live_slow_answer_seconds"].default == 40.0
    assert Settings.model_fields["voice_state_posts"].default is True


async def test_three_unaddressed_utterances_drop_back_to_listening(env, monkeypatch):
    s = await quiet_session(env)
    env.mgr._set_engaged(s, True)
    s.called_at = time.monotonic() - 60  # the call is long ago, Salid is not speaking
    heard = iter(["เมื่อวานไปกินข้าวที่ไหนมา", "ร้านใหม่หน้าออฟฟิศ", "อร่อยมาก"])
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: next(heard))
    await env.mgr.handle_segment(s, segment(10))
    assert s.engaged is True and s.unaddressed == 1
    await env.mgr.handle_segment(s, segment(11))
    assert s.engaged is True and s.unaddressed == 2
    await env.mgr.handle_segment(s, segment(12))
    assert s.engaged is False  # silently: nothing was said
    assert env.vc.played == [] and env.chan.sent == []
    await env.mgr.end((1000, 77), "shutdown")


async def test_callers_own_utterances_are_never_unaddressed(env, monkeypatch):
    s = await quiet_session(env)
    env.mgr._set_engaged(s, True, by=10)
    assert s.called_by == 10
    s.called_at = time.monotonic() - 60
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "เมื่อวานไปกินข้าวมา")
    monkeypatch.setattr(env.mgr, "answer", lambda *a, **k: asyncio.sleep(0))
    for _ in range(5):
        await env.mgr.handle_segment(s, segment(10))
    assert s.engaged is True and s.unaddressed == 0
    await env.mgr.handle_segment(s, segment(11))  # someone else: counts
    assert s.unaddressed == 1
    await env.mgr.end((1000, 77), "shutdown")


async def test_an_addressed_utterance_resets_the_count(env, monkeypatch):
    s = await quiet_session(env)
    env.mgr._set_engaged(s, True)
    s.called_at = time.monotonic() - 60
    heard = iter(["ตอนเย็นไปไหนกัน", "Salid ช่วยดูยอดหน่อย", "แล้วไง"])
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: next(heard))
    monkeypatch.setattr(env.mgr, "answer", lambda *a, **k: asyncio.sleep(0))
    await env.mgr.handle_segment(s, segment(10))
    assert s.unaddressed == 1
    await env.mgr.handle_segment(s, segment(10))  # name: engaged again, count reset
    assert s.engaged is True and s.unaddressed == 0
    await env.mgr.handle_segment(s, segment(10))  # right after the call: a reply to her
    assert s.engaged is True and s.unaddressed == 0
    await env.mgr.end((1000, 77), "shutdown")


async def test_a_question_soon_after_she_spoke_counts_as_addressed(env):
    s = await quiet_session(env)
    env.mgr._set_engaged(s, True)
    now = time.monotonic()
    s.called_at = now - 15
    assert env.mgr._addressed(s, "แล้วเดือนที่แล้วล่ะเท่าไหร่", now)  # a question, 15 s ago
    assert not env.mgr._addressed(s, "ร้านนี้อร่อยดี", now)  # not a question
    s.called_at = now - 40
    assert not env.mgr._addressed(s, "แล้วเดือนที่แล้วล่ะเท่าไหร่", now)
    await env.mgr.end((1000, 77), "shutdown")


async def test_engaged_times_out_after_45_seconds_by_default(env):
    s = await quiet_session(env)
    assert env.mgr._engage_s() == 45.0
    env.mgr._set_engaged(s, True)
    env.mgr._consent_tick(s, time.monotonic() + 30)
    assert s.engaged is True
    env.mgr._consent_tick(s, time.monotonic() + 60)
    assert s.engaged is False
    await env.mgr.end((1000, 77), "shutdown")


async def test_ok_while_engaged_returns_to_listening_without_a_spoken_reply(env, monkeypatch):
    s = await quiet_session(env)
    env.mgr._set_engaged(s, True)
    said = []
    monkeypatch.setattr(env.mgr, "_say_text", lambda *a: said.append(a) or asyncio.sleep(0))
    for text, spoken in (("โอเค คนต่อไป", 0), ("ขอบคุณครับ", 1)):
        env.mgr._set_engaged(s, True)
        monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm, x=text: x)
        await env.mgr.handle_segment(s, segment(10))
        await asyncio.sleep(0.01)
        assert s.engaged is False and len(said) == spoken  # only a direct thank-you is answered
    await env.mgr.end((1000, 77), "shutdown")


# ------------------------------------------------------------------ visible state
class Msg:
    def __init__(self, text):
        self.text = text
        self.edits = []

    async def edit(self, content):
        self.edits.append(content)
        self.text = content


def editable(env):
    env.chan.msgs = []

    async def send(text, **_):
        env.chan.sent.append(text)
        m = Msg(text)
        env.chan.msgs.append(m)
        return m

    env.chan.send = send


async def state_session(env, flip_s=15.0):
    env.bot.s.meeting_text_echo = False
    env.bot.s.voice_state_posts = True
    editable(env)
    mt.MeetingManager.STATE_FLIP_S = flip_s
    s = await start(env)
    await asyncio.sleep(0)
    return s


async def test_join_engaged_listening_state_lines_and_the_quick_flip_is_an_edit(env):
    s = await state_session(env)
    assert env.chan.sent == [t("voice_state_join", "th")] and "👂" in env.chan.sent[0]
    env.mgr._set_engaged(s, True, name="มิว")
    await asyncio.sleep(0.01)
    # within 15 s of the join line: edited, not a second post
    assert env.chan.msgs[0].edits == ["💬 สลิดกำลังคุยกับคุณมิวอยู่ค่ะ"] and len(env.chan.sent) == 1
    env.mgr._set_engaged(s, False)
    await asyncio.sleep(0.01)
    assert env.chan.msgs[0].text == "👂 สลิดกลับไปฟังอย่างเดียวแล้วค่ะ" and len(env.chan.sent) == 1
    await env.mgr.end((1000, 77), "shutdown")


async def test_state_after_15_seconds_is_a_new_line_and_no_name_has_its_own_text(env):
    s = await state_session(env)
    s.state_at -= 20  # the join line is old
    env.mgr._set_engaged(s, True)
    await asyncio.sleep(0.01)
    assert env.chan.sent[-1] == "💬 สลิดกำลังคุยอยู่ค่ะ" and len(env.chan.sent) == 2
    await env.mgr.end((1000, 77), "shutdown")


async def test_english_state_lines(env):
    env.bot.s.meeting_text_echo = False
    env.bot.s.voice_state_posts = True
    editable(env)
    s = await start(env)
    s.language = "en"
    s.state_at -= 20
    env.mgr._set_engaged(s, True, name="Mew")
    await asyncio.sleep(0.01)
    assert env.chan.sent[-1] == "💬 Salid is talking with Mew."
    await env.mgr.end((1000, 77), "shutdown")


async def test_state_posts_can_be_switched_off(env):
    env.bot.s.meeting_text_echo = False
    env.bot.s.voice_state_posts = False
    s = await start(env)
    env.mgr._set_engaged(s, True)
    await asyncio.sleep(0.01)
    assert env.chan.sent == []
    await env.mgr.end((1000, 77), "shutdown")


async def test_a_failing_post_never_raises(env):
    env.bot.s.voice_state_posts = True

    async def boom(text, **_):
        raise RuntimeError("rate limited")

    env.chan.send = boom
    s = await start(env)
    assert await env.mgr._post(s, "x") is None
    await env.mgr.end((1000, 77), "shutdown")


# ------------------------------------------------------------------ slow answers / failures
async def test_tool_timeout_returns_an_honest_line_to_read_and_posts_the_failure(lv, monkeypatch):
    lv.bot.s.live_ask_timeout_seconds = 0.05

    async def slow(self, q):
        await asyncio.sleep(5)

    monkeypatch.setattr(LiveVoice, "_ask_salid", slow)
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": "ยอดเดือนนี้"}))
    await until(lambda: live.tool_responses)
    res = live.tool_responses[0][0].response
    assert res["error"] == "timeout"
    assert res["say"] == "ขออภัยค่ะ ตรวจสอบให้ไม่ได้ตอนนี้ เครื่องมือค้นข้อมูลขัดข้องค่ะ"
    await until(lambda: any("⚠️" in x for x in lv.chan.sent))
    assert "ask_salid ขัดข้อง (timeout)" in [x for x in lv.chan.sent if "⚠️" in x][0]
    await lv.mgr.end((1000, 77))


async def test_a_slow_answer_that_arrives_before_the_limit_is_delivered(lv, monkeypatch):
    lv.bot.s.live_ask_timeout_seconds = 5

    async def slowish(self, q):
        await asyncio.sleep(0.2)
        return {"answer": "ยอดรวม 10,370.50 บาท"}

    monkeypatch.setattr(LiveVoice, "_ask_salid", slowish)
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": "ยอดเดือนนี้"}))
    await until(lambda: live.tool_responses)
    assert live.tool_responses[0][0].response == {"answer": "ยอดรวม 10,370.50 บาท"}
    await lv.mgr.end((1000, 77))


async def test_answer_is_echoed_to_the_text_chat(lv, monkeypatch):
    async def ans(self, q):
        return {"answer": "คุณแอนรับงานนี้ ส่งวันศุกร์"}

    monkeypatch.setattr(LiveVoice, "_ask_salid", ans)
    lv.bot.s.meeting_text_echo = False  # the echo is not the optional chatter
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": "ใครรับงาน"}))
    await until(lambda: "💬 **สลิด:** คุณแอนรับงานนี้ ส่งวันศุกร์" in lv.chan.sent)
    await lv.mgr.end((1000, 77))


async def test_tool_exception_is_echoed_and_the_echo_never_raises_into_the_loop(lv, monkeypatch):
    async def boom(self, q):
        raise RuntimeError("search backend down")

    async def bad_post(text, **_):
        raise RuntimeError("discord down")

    monkeypatch.setattr(LiveVoice, "_ask_salid", boom)
    s, live = await lv.join(5)
    lv.chan.send = bad_post
    live.push(tool_msg("ask_salid", {"question": "q"}))
    await until(lambda: live.tool_responses)
    assert "say" in live.tool_responses[0][0].response
    live.push(tool_msg("post_note", {"text": "x"}, id="n"))  # the loop still runs
    await until(lambda: len(live.tool_responses) == 2)
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ one brain at a time
@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("ยอดเดือนนี้เท่าไร", "ยอดเดือนนี้เท่าไร?", True),
        ("ยอดเดือนนี้เท่าไร", "ขอถามอีกที ยอดเดือนนี้เท่าไร", True),  # one contains the other
        ("who owns the Q4 plan", "Who owns the Q4 plan?", True),
        ("ยอดเดือนนี้เท่าไร", "ใครรับงานแผนไตรมาสสี่", False),
        ("", "x", False),
    ],
)
def test_same_question(a, b, same):
    assert same_question(a, b) is same


class Brain:
    """ask_salid stand-in: records starts, finishes only when released."""

    def __init__(self):
        self.started, self.finished, self.cancelled = [], [], []
        self.gates: dict[str, asyncio.Event] = {}

    async def __call__(self, lv_self, q):
        self.started.append(q)
        gate = self.gates.setdefault(q, asyncio.Event())
        try:
            await gate.wait()
        except asyncio.CancelledError:
            self.cancelled.append(q)
            raise
        self.finished.append(q)
        return {"answer": f"ตอบ {q}"}


async def test_same_question_while_a_job_runs_gets_still_working_and_no_second_agent(
    lv, monkeypatch
):
    lv.bot.s.live_foreground_seconds = 0.05
    brain = Brain()
    monkeypatch.setattr(LiveVoice, "_ask_salid", lambda self, q: brain(self, q))
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": "ยอดเดือนนี้เท่าไร"}, id="a"))
    await until(lambda: len(live.tool_responses) == 1)  # the first one went to the background
    live.push(tool_msg("ask_salid", {"question": "ยอดเดือนนี้เท่าไร นะ"}, id="b"))
    await until(lambda: len(live.tool_responses) == 2)
    got = {r[0].id: r[0].response for r in live.tool_responses}
    assert got["a"]["status"] == "working" and got["b"]["status"] == "working"
    assert got["b"]["say"] == t("voice_still_working", "th") and got["b"]["job"] == got["a"]["job"]
    assert len(brain.started) == 1 and not brain.cancelled  # no second agent
    brain.gates["ยอดเดือนนี้เท่าไร"].set()
    await until(lambda: "💬 **สลิด:** ตอบ ยอดเดือนนี้เท่าไร" in lv.chan.sent)
    await lv.mgr.end((1000, 77))


async def test_a_different_question_is_a_new_job_and_nothing_is_cancelled(lv, monkeypatch):
    lv.bot.s.live_foreground_seconds = 0.05
    brain = Brain()
    monkeypatch.setattr(LiveVoice, "_ask_salid", lambda self, q: brain(self, q))
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": "ยอดเดือนนี้เท่าไร"}, id="a"))
    await until(lambda: len(live.tool_responses) == 1)
    live.push(tool_msg("ask_salid", {"question": "ใครรับงานแผนไตรมาสสี่"}, id="b"))
    await until(lambda: len(brain.started) == 2)
    await until(lambda: len(live.tool_responses) == 2)
    assert not brain.cancelled
    for q in list(brain.gates):
        brain.gates[q].set()
    await until(lambda: len([x for x in lv.chan.sent if x.startswith("💬 **สลิด:**")]) == 2)
    echoes = {x for x in lv.chan.sent if x.startswith("💬 **สลิด:**")}
    assert echoes == {"💬 **สลิด:** ตอบ ยอดเดือนนี้เท่าไร", "💬 **สลิด:** ตอบ ใครรับงานแผนไตรมาสสี่"}
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ prompts
def test_prompts_are_listen_first_in_both_languages():
    s = FixedClockSettings()
    th, en = build_instruction(s, "th"), build_instruction(s, "en")
    assert "ผู้จดบันทึกประจำทีม" in th and "ไม่ใช่เพื่อนคุย" in th
    assert "โอเค ขอบคุณ คนต่อไป" in th and "โดยไม่พูดอะไรเลย" in th
    assert "ตามที่ได้มา" in th and "ห้ามเรียบเรียงใหม่" in th and "say" in th
    assert "ขออภัยค่ะ ตรวจสอบให้ไม่ได้ตอนนี้" in th
    for gone in ("เข้าร่วมบทสนทนาอย่างเป็นธรรมชาติ", "ถามชวนคุย", "เรียบเรียงคำตอบเป็นคำพูดของคุณเอง"):
        assert gone not in th
    assert "note-taker" in en and "LISTEN" in en and "WITHOUT saying anything" in en
    assert "as it is" in en and "never rewrite" in en and "“say” field" in en
    for gone in ("join the conversation naturally", "ask questions like a good colleague"):
        assert gone not in en
    assert "ได้ค่ะ” สั้น ๆ แล้วกลับไปเงียบฟัง" not in lvm.STATE_ENGAGED
    assert "without saying anything" in lvm.STATE_ENGAGED_EN


# ------------------------------------------------------------------ one memory search per question
def test_chat_and_files_adapters_share_one_memory_search():
    from datetime import UTC, date, datetime
    from zoneinfo import ZoneInfo

    calls = []
    lock = threading.Lock()

    class Memory:
        def chat_search(self, **kw):
            with lock:
                calls.append(kw)
            time.sleep(0.05)  # the twin adapter arrives while this one is still running
            return {"hits": []}

    src = sr.build_sources(
        memory=Memory(), tools=[], zone=ZoneInfo("Asia/Bangkok"), now=lambda: None, user_id=1
    )
    out = sr.search_everything(
        src,
        lambda q: ["quotation"],
        "quotation",
        now=datetime(2026, 10, 16, tzinfo=UTC),
        zone=ZoneInfo("Asia/Bangkok"),
    )
    assert out["count"] == 0 and set(out["searched"]) >= {"chat", "files"}
    assert len(calls) == 1  # was 2 identical searches per question


def test_a_failing_shared_search_fails_both_adapters_not_hangs():
    from datetime import date

    class Memory:
        def chat_search(self, **kw):
            raise RuntimeError("index down")

    shared = sr.SharedChatSearch(Memory())
    for _ in range(2):
        with pytest.raises(RuntimeError):
            shared(["a"], date(2026, 10, 1), date(2026, 10, 16))


# ------------------------------------------------------------------ spoken rounds cap
def test_spoken_answers_get_five_tool_rounds_chat_keeps_six(seeded, settings):
    assert SPOKEN_MAX_ROUNDS == 5 and MAX_ROUNDS == 6
    llm = scripted(*[fc_resp(("flagged_receipts", {}))] * SPOKEN_MAX_ROUNDS, text_resp("สรุป"))
    a = ask(settings, seeded, "x", guild_id=GUILD, is_exec=True, user_id=EXEC_ID, model=llm,
            spoken=True)  # fmt: skip
    assert a.text == "สรุป" and a.rounds == SPOKEN_MAX_ROUNDS
    assert len(llm.requests) == SPOKEN_MAX_ROUNDS + 1
    assert not llm.requests[SPOKEN_MAX_ROUNDS].config.tools  # tools withdrawn after round 5


def test_a_stray_call_after_the_cap_is_not_counted_as_a_round(seeded, settings):
    llm = scripted(*[fc_resp(("flagged_receipts", {}))] * (MAX_ROUNDS + 1), text_resp("สรุป"))
    a = ask(settings, seeded, "x", guild_id=GUILD, is_exec=True, user_id=EXEC_ID, model=llm)
    assert a.rounds == MAX_ROUNDS  # was MAX_ROUNDS + 1 (the "7 rounds" seen in the log)
