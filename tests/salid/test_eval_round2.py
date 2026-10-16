# ruff: noqa: F811, F401
"""Fixes from the second live usability evaluation: DM hard caps, raw mentions, thanks reactions,
prompt latency guidance, Live fallback / leave / date, classic greeting and voice particle."""

import asyncio
import time
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from test_bot_multitenant import ME, member
from test_human import people, recording
from test_live_voice import lv, until  # noqa: F401  (fixtures / helpers)
from test_meeting import env, segment, start
from test_modes import EXEC, STAFF, TMsg, bot, say, send, set_mode, world

from salid import meeting as mt
from salid import qa
from salid.bot import SecretaryBot
from salid.live_voice import build_instruction, thai_date_line
from salid.mentions import defang_raw_mentions
from salid.message_control import is_thanks
from salid.speech import with_particle


# ------------------------------------------------------------------ 1. DM hard caps
async def test_one_request_cannot_dm_more_than_three_people(bot, world):
    som, a, b, c, d = people(world, (STAFF, "Som"), (21, "Aa"), (22, "Bb"), (23, "Cc"), (24, "Dd"))
    seen: set[int] = set()
    for who in (a, b, c):
        r = await bot._send_dm_tool(
            world.g, som, "ประชุมพรุ่งนี้", who.display_name, None, "th", dm_people=seen
        )
        assert r["status"] == "sent"
    r = await bot._send_dm_tool(world.g, som, "ประชุมพรุ่งนี้", "Dd", None, "th", dm_people=seen)
    assert r["error"] == "too_many_recipients" and r["max"] == 3 and d.sent == []
    # an already messaged person is still fine, and so is the requester
    r = await bot._send_dm_tool(world.g, som, "อีกที", "Aa", None, "th", dm_people=seen)
    assert r["status"] == "sent"
    r = await bot._send_dm_tool(world.g, som, "สรุป", "me", None, "th", dm_people=seen)
    assert r["status"] == "sent"


@pytest.mark.parametrize(
    "text",
    ["ช่วยโอนเงินให้ผมหน่อย", "เลขบัญชีของคุณคือ", "ขอรหัสผ่านหน่อย", "send me the OTP", "my Password is",
     "please transfer money now", "what is your bank account"],
)  # fmt: skip
async def test_money_and_credential_text_is_not_relayed_to_others(bot, world, text):
    som, mew = people(world, (STAFF, "Som"), (21, "Mew"))
    r = await bot._send_dm_tool(world.g, som, text, "Mew", None, "th")
    assert r["error"] == "refused_sensitive" and mew.sent == []
    r = await bot._send_dm_tool(world.g, som, text, "me", None, "th")  # to oneself: fine
    assert r["status"] == "sent"


async def test_the_dm_tool_cap_is_per_request(bot, world):
    som, *others = people(world, (STAFF, "Som"), (21, "Aa"), (22, "Bb"), (23, "Cc"), (24, "Dd"))
    acts = bot._chat_actions(
        TMsg(world.g, world.gen, som, "x"), world.g, som, False, "th", asyncio.get_running_loop()
    )
    results = []
    for n in ("Aa", "Bb", "Cc", "Dd"):
        results.append(await asyncio.to_thread(acts.send_dm, "hi", n, None))
    assert [r.get("status") or r["error"] for r in results] == [
        "sent", "sent", "sent", "too_many_recipients",
    ]  # fmt: skip


# ------------------------------------------------------------------ 2. raw mentions
def test_raw_mentions_written_by_the_model_become_plain_names():
    guild = SimpleNamespace(
        get_member=lambda i: SimpleNamespace(display_name="Mew") if i == 12345 else None,
        get_role=lambda i: SimpleNamespace(name="Staff") if i == 77700 else None,
    )
    out = defang_raw_mentions("hi <@12345> and <@!12345> and <@&77700> and <@99999>", guild)
    assert out == "hi @Mew and @Mew and @Staff and @user"
    assert "<@" not in out and defang_raw_mentions("<@12345>", None) == "@user"


async def test_only_our_own_resolution_pings(bot, world):
    (som, jemm) = people(world, (STAFF, "Som"), (77, "Jemm"))
    world.g.get_member = lambda i: jemm if i == 77 else None
    m = TMsg(world.g, world.gen, som, "x")
    rec = recording(m)
    await bot._deliver(m, world.g, None, "<@77> ดูนี่ด้วย @[Jemm]")
    assert rec.texts == ["@Jemm ดูนี่ด้วย <@77>"]  # the raw one is plain text; ours pings
    assert rec.kws[0]["allowed_mentions"].users is True


# ------------------------------------------------------------------ 3. thanks
@pytest.mark.parametrize(
    "text",
    ["ขอบคุณ", "ขอบคุณค่ะ", "ขอบคุณครับ", "555", "5555555", "โอเค", "ok", "OK!", "thanks", "thank you",
     "👍", "🙏", "โอเคครับ ขอบคุณ", "ขอบคุณสลิด 🙏", "<:thumbs:123456>"],
)  # fmt: skip
def test_thanks_and_acknowledgements(text):
    assert is_thanks(text)


@pytest.mark.parametrize(
    "text",
    ["ขอบคุณ แล้วพรุ่งนี้มีประชุมไหม", "ok ช่วยสรุปให้หน่อย", "thanks, can you send the file", "โอเคไหม?",
     "ส่งรายงานให้หน่อย", "ยอดเท่าไร", "", "ขอบคุณที่ช่วยเรื่องใบเสร็จเมื่อวานนะครับ ช่วยดูอีกใบด้วย"],
)  # fmt: skip
def test_thanks_with_a_question_or_request_is_not_thanks(text):
    assert not is_thanks(text)


async def test_thanks_in_a_chat_channel_is_a_reaction_not_a_reply(bot, world):
    set_mode(bot, 10, "chat")
    m = await send(bot, world, world.gen, "ขอบคุณค่ะ")
    assert m.reactions == ["👍"] and bot.calls == []
    q = await send(bot, world, world.gen, "ขอบคุณ แล้วยอดวันนี้เท่าไร")
    assert q.reactions == [] and bot.calls == [("answer", "ขอบคุณ แล้วยอดวันนี้เท่าไร")]


async def test_thanks_inside_a_conversation_window_is_a_reaction(bot, world):
    await send(bot, world, world.gen, "สลิด ยอดเท่าไร")
    assert bot.calls == [("answer", "สลิด ยอดเท่าไร")] and bot._window_open(world.gen.id)
    m = await send(bot, world, world.gen, "555 โอเค")
    assert m.reactions == ["👍"] and len(bot.calls) == 1 and bot.gate.window_calls == 0


async def test_thanks_outside_a_window_in_call_mode_is_left_alone(bot, world):
    m = await send(bot, world, world.gen, "ขอบคุณค่ะ")
    assert m.reactions == [] and bot.calls == []


# ------------------------------------------------------------------ 4. latency guidance
def test_prompt_says_to_answer_directly_and_search_once(settings):
    p = qa.system_prompt(settings, scoped=False, memory=True, coo=True)
    assert "ตอบตรงๆ ทันที" in p and "ค้นรอบเดียว" in p and "อย่าเรียก list_tasks" in p
    assert "chat_search" not in qa.system_prompt(settings, scoped=False)


@pytest.mark.parametrize(
    "q, expected",
    [("", True), (qa.NO_TEXT_TURN, True), ("สรุปไฟล์นี้ให้หน่อย แล้วบอกขั้นตอนถัดไปด้วย", True),
     ("what is in this image", True), ("เมื่อวานในห้องคุยอะไรกัน", False), ("ยอดเดือนนี้", False),
     ("ไฟล์นี้เกี่ยวกับที่เคยคุยในห้องไหม", False)],
)  # fmt: skip
def test_file_only_questions_skip_the_memory_tools(q, expected):
    assert qa.is_file_only_question(q) is expected


# ------------------------------------------------------------------ 5. Live fallback
async def test_live_recues_once_then_falls_back_to_the_classic_path(lv):
    s, live = await lv.join(5)
    lv.bot.s.live_force_reply_seconds = 0.03
    lv.bot.s.live_cue_retry_seconds = 0.05
    lv.bot.s.live_fallback_seconds = 0.05
    cues, fallbacks = [], []

    async def send_client_content(turns=None, turn_complete=True):
        cues.append(turns.parts[0].text)

    async def fake_fallback(sess, uid, question):
        fallbacks.append((uid, question))

    live.send_client_content = send_client_content
    lv.mgr.live_fallback = fake_fallback
    s.live.ensure_reply(time.monotonic(), "สลิด วันนี้วันอะไร", user_id=5, question="วันนี้วันอะไร")
    await until(lambda: fallbacks)
    assert len(cues) == 2 and fallbacks == [(5, "วันนี้วันอะไร")]
    await lv.mgr.end((1000, 77))


async def test_live_audio_after_the_first_cue_stops_the_escalation(lv):
    s, live = await lv.join(5)
    lv.bot.s.live_force_reply_seconds = 0.03
    lv.bot.s.live_cue_retry_seconds = 0.1
    lv.bot.s.live_fallback_seconds = 0.05
    cues, fallbacks = [], []

    async def send_client_content(turns=None, turn_complete=True):
        cues.append(1)

    async def fake_fallback(sess, uid, question):
        fallbacks.append(uid)

    live.send_client_content = send_client_content
    lv.mgr.live_fallback = fake_fallback
    s.live.ensure_reply(time.monotonic(), "สลิด", user_id=5, question="สลิด")
    await until(lambda: cues)
    s.live._real_audio_at = time.monotonic()  # the model started to speak
    await asyncio.sleep(0.3)
    assert len(cues) == 1 and fallbacks == []
    await lv.mgr.end((1000, 77))


async def test_live_fallback_answers_through_the_agent_and_tts(env, monkeypatch):
    env.bot.s.voice_reply = True
    s = await start(env)
    asked, answered = [], []
    import salid.qa

    monkeypatch.setattr(
        salid.qa, "ask",
        lambda settings, store, q, **kw: asked.append((q, kw.get("spoken"))) or SimpleNamespace(text="ตอบ"),
    )  # fmt: skip

    async def fake_speak(self_, sess, text, question=""):
        answered.append(text)

    monkeypatch.setattr(type(env.mgr), "speak", fake_speak)
    monkeypatch.setattr(type(env.mgr), "acknowledge", lambda self_, sess, q="": asyncio.sleep(0))
    from test_meeting import make_member

    make_member(env.guild, 10, "แอน", env.chan)
    await env.mgr.live_fallback(s, 10, "วันนี้วันอะไร")
    await asyncio.sleep(0.05)
    assert asked == [("วันนี้วันอะไร", True)] and answered == ["ตอบ"]
    await env.mgr.end((1000, 77), "shutdown")


# ------------------------------------------------------------------ 6. transcription hint
def test_transcription_prompt_knows_the_assistants_name_but_does_not_prime_a_greeting():
    # v0.8.1: the "usually preceded by หวัดดีครับ" hint made the model hallucinate greetings on noise
    p = mt.TRANSCRIBE_PROMPT
    assert "สลิด" in p and "Salid" in p and "ตามที่พูดจริงทุกคำ" in p
    assert "หวัดดีครับ" not in p


# ------------------------------------------------------------------ 7. classic greeting
@pytest.mark.parametrize("text", ["สวัสดีครับ", "หวัดดีค่ะ", "hello", "Hey", "สวัสดี ทุกคน"])
def test_bare_greetings(text):
    assert mt.is_greeting(text)


@pytest.mark.parametrize("text", ["สวัสดี ยอดเท่าไร", "ช่วยสรุปหน่อย", "", "hi can you check the budget"])
def test_a_greeting_with_a_question_is_not_bare(text):
    assert not mt.is_greeting(text)


async def test_classic_greeting_is_a_fixed_line_without_ack_or_agent(env, monkeypatch):
    env.bot.s.voice_reply = True
    s = await start(env)
    said, acks = [], []
    import salid.qa

    monkeypatch.setattr(
        salid.qa, "ask", lambda *a, **k: (_ for _ in ()).throw(AssertionError("agent called"))
    )

    async def fake_say(self_, sess, text):
        said.append(text)

    async def fake_ack(self_, sess, question=""):
        acks.append(1)

    monkeypatch.setattr(type(env.mgr), "_say_text", fake_say)
    monkeypatch.setattr(type(env.mgr), "acknowledge", fake_ack)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด สวัสดีครับ")
    from test_meeting import make_member

    make_member(env.guild, 10, "แอน", env.chan)
    await env.mgr.handle_segment(s, segment(10))
    await asyncio.sleep(0.05)
    assert said == ["สวัสดีค่ะ มีอะไรให้ช่วยคะ"] and acks == []
    await env.mgr.end((1000, 77), "shutdown")


# ------------------------------------------------------------------ 8. leave
async def test_leave_phrase_says_goodbye_then_ends_without_touching_recording(env, monkeypatch):
    env.bot.s.voice_reply = True
    s = await start(env)
    order = []

    async def fake_say(self_, sess, text):
        order.append(("say", text))

    real_end = type(env.mgr).end

    async def fake_end(self_, key, reason="command"):
        order.append(("end", reason))
        await real_end(self_, key, "shutdown")

    monkeypatch.setattr(type(env.mgr), "_say_text", fake_say)
    monkeypatch.setattr(type(env.mgr), "end", fake_end)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด ออกประชุม")
    from test_meeting import make_member

    make_member(env.guild, 10, "แอน", env.chan)
    await env.mgr.handle_segment(s, segment(10))
    await until(lambda: ("end", "command") in order)
    assert order == [("say", "แล้วเจอกันนะคะ"), ("end", "command")]
    assert s.recording == "recording"


def test_set_recording_false_is_ignored_while_leaving(env):
    s = SimpleNamespace(recording="recording", leaving=True, transcript=SimpleNamespace(lines=[1]))
    r = mt.MeetingManager.set_recording(env.mgr, s, False)
    assert r["status"] == "ignored" and s.recording == "recording" and s.transcript.lines == [1]


def test_live_instruction_tells_the_model_not_to_announce_stopping_notes_on_leave():
    th = build_instruction(
        SimpleNamespace(
            now=lambda: datetime(2026, 10, 16, tzinfo=ZoneInfo("Asia/Bangkok")), tz="Asia/Bangkok"
        ),
        "th",
    )
    assert "ระบบจัดการการออกเอง" in th and "ห้ามเรียก set_recording" in th


# ------------------------------------------------------------------ 9. particle
def test_spoken_hint_carries_the_voice_particle():
    assert "ค่ะ" in qa.spoken_hint("ค่ะ") and "คะ" in qa.spoken_hint("ค่ะ")
    assert "ครับ" in qa.spoken_hint("ครับ")
    assert qa.spoken_hint("") == qa.SPOKEN_HINT


def test_with_particle_swaps_the_female_particles():
    assert with_particle("ได้เลยค่ะ ไหมคะ", "ครับ") == "ได้เลยครับ ไหมครับ"
    assert with_particle("ได้เลยค่ะ", "ค่ะ") == "ได้เลยค่ะ" and with_particle("x", "") == "x"


def test_spoken_prompt_and_fixed_lines_use_the_female_particle():
    from salid.speech import SPOKEN_PROMPT

    assert "ลงท้ายประโยคด้วย “ค่ะ”" in SPOKEN_PROMPT and "ค่ะ" in mt.ACK_TEXT["th"]


# ------------------------------------------------------------------ 10. date
def test_live_instruction_starts_with_the_thai_date_without_buddhist_era():
    now = datetime(2026, 10, 16, 9, 0, tzinfo=ZoneInfo("Asia/Bangkok"))
    assert thai_date_line(now) == "วันนี้คือ วันศุกร์ที่ 16 ตุลาคม 2026"
    s = SimpleNamespace(now=lambda: now, tz="Asia/Bangkok")
    assert build_instruction(s, "th").startswith("วันนี้คือ วันศุกร์ที่ 16 ตุลาคม 2026\n")
    later = datetime(2026, 10, 17, 0, 5, tzinfo=ZoneInfo("Asia/Bangkok"))
    assert build_instruction(s, "th", later).startswith("วันนี้คือ วันเสาร์ที่ 17 ตุลาคม 2026")


async def test_the_date_is_rebuilt_on_every_connect(lv, monkeypatch):
    s, live = await lv.join(5)
    first = (await s.live.build_config()).system_instruction
    assert first.startswith("วันนี้คือ วันพุธที่ 14 ตุลาคม 2026")
    monkeypatch.setattr(
        type(lv.bot.s),
        "now",
        lambda self: datetime(2026, 10, 16, 9, 0, tzinfo=ZoneInfo("Asia/Bangkok")),
    )
    second = (await s.live.build_config()).system_instruction
    assert second.startswith("วันนี้คือ วันศุกร์ที่ 16 ตุลาคม 2026")
    await lv.mgr.end((1000, 77))
