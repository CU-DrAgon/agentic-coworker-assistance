# ruff: noqa: F811, F401
"""Natural voice-meeting behaviour: Live turn-taking config, nothing posted in text by default,
recording state (unknown / recording / not_recording), notes, the minutes destination, join / leave
phrases and the cooldown. Fully offline."""

import asyncio
import time
from types import SimpleNamespace

import pytest
from test_live_voice import lv, pcm48, until  # noqa: F401  (fixtures / helpers)
from test_meeting import ME, FakeVoiceChannel, env, make_member, segment, start

from salid import meeting as mt
from salid.coo_store import MINUTES
from salid.i18n import t
from salid.live_voice import LiveVoice, build_realtime_input
from salid.meeting import parse_meeting_command
from salid.minutes import parse_minutes_destination


class TChan:
    def __init__(self, id, name, view=True, send=True):
        self.id, self.name, self.view, self.send_ok, self.sent = id, name, view, send, []
        self.guild = None

    def permissions_for(self, who):
        return SimpleNamespace(view_channel=self.view, read_messages=self.view,
                               send_messages=self.send_ok)  # fmt: skip

    async def send(self, text, **_):
        self.sent.append(text)
        return SimpleNamespace(jump_url=f"https://x/{self.id}")


def add_text(env, *chans):
    for c in chans:
        c.guild = env.guild
    env.guild.text_channels = list(chans)
    env.guild.get_channel = lambda cid: next((c for c in chans if c.id == cid), None)


# ------------------------------------------------------------------ Live turn-taking
def test_realtime_input_config_for_thai_turn_taking():
    s = SimpleNamespace(live_silence_ms=1200, live_prefix_padding_ms=300,
                        live_start_sensitivity="high", live_end_sensitivity="low")  # fmt: skip
    vad = build_realtime_input(s).automatic_activity_detection
    assert vad.start_of_speech_sensitivity.name == "START_SENSITIVITY_HIGH"
    assert vad.end_of_speech_sensitivity.name == "END_SENSITIVITY_LOW"
    assert vad.silence_duration_ms == 1200 and vad.prefix_padding_ms == 300


async def test_live_config_has_vad_proactivity_and_meeting_tools(lv):
    s, live = await lv.join(5)
    cfg = lv.connect.configs[0]
    vad = cfg.realtime_input_config.automatic_activity_detection
    assert vad.silence_duration_ms == 1200 and vad.prefix_padding_ms == 300
    assert cfg.proactivity.proactive_audio is True
    names = [d.name for d in cfg.tools[0].function_declarations]
    assert {"post_note", "ask_consent", "set_recording", "set_minutes_destination"} <= set(names)
    assert "สลิด" in cfg.system_instruction and "เพื่อนร่วมทีม" in cfg.system_instruction
    await lv.mgr.end((1000, 77))


async def test_a_short_greeting_is_forwarded_even_before_the_session_is_ready():
    live = LiveVoice(SimpleNamespace(), SimpleNamespace(), post=lambda t: None)
    live.feed(1, pcm48(900))  # one 20 ms frame, heard while connecting
    assert live.mixer.pop() != b""


# ------------------------------------------------------------------ voice stays voice
async def test_join_posts_and_says_nothing_by_default(env):
    env.bot.s.meeting_text_echo = False
    env.bot.s.voice_reply = True
    a = make_member(env.guild, 10, "แอน", env.chan)
    await env.mgr.join(a, origin="auto")
    await asyncio.sleep(0.05)
    assert env.chan.sent == [] and env.vc.played == []
    await env.mgr.end((1000, 77), "shutdown")


async def test_wake_answer_is_not_echoed_in_text_by_default(env, monkeypatch):
    env.bot.s.meeting_text_echo = False
    s = await start(env)
    import salid.qa

    monkeypatch.setattr(salid.qa, "ask", lambda *a, **k: SimpleNamespace(text="ตอบ"))
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด ขอถามหน่อย")
    await env.mgr.handle_segment(s, segment(10))
    assert env.chan.sent == []
    await env.mgr.end((1000, 77), "shutdown")


async def test_summary_is_still_posted_when_echo_is_off(env):
    env.bot.s.meeting_text_echo = False
    s = await start(env)
    s.transcript.add(env.bot.s.now(), "แอน", "x")
    env.llm.models.script = [mt.text_resp("สรุป") if hasattr(mt, "text_resp") else None]
    from conftest import text_resp

    env.llm.models.script = [text_resp("สรุปการประชุม")]
    await env.mgr.end((1000, 77), "command")
    assert any("สรุปการประชุม" in m for m in env.chan.sent)


# ------------------------------------------------------------------ recording state
async def auto_session(env, **settings):
    env.bot.s.meeting_text_echo = False
    for k, v in settings.items():
        setattr(env.bot.s, k, v)
    a = make_member(env.guild, 10, "แอน", env.chan)
    make_member(env.guild, 11, "บอย", env.chan)
    env.chan.members = [a]
    await env.mgr.join(a, origin="auto")
    return env.mgr.sessions[(1000, 77)]


async def test_states_unknown_for_auto_known_for_command_and_schedule(env):
    s = await auto_session(env)
    assert s.recording == "unknown"
    await env.mgr.adopt(s, "m1", None)  # a schedule says it is a meeting
    assert s.recording == "recording"
    await env.mgr.end((1000, 77), "shutdown")


async def test_consent_is_asked_once_after_the_delay_in_text(env, monkeypatch):
    s = await auto_session(env, consent_delay_seconds=20.0)

    async def fake_line(self_, sess):
        return "สลิดขออนุญาตจดนะคะ"

    monkeypatch.setattr(type(env.mgr), "_consent_line", fake_line)
    s.first_speech_at = time.monotonic() - 5
    env.mgr._consent_tick(s, time.monotonic())
    await asyncio.sleep(0.02)
    assert env.chan.sent == []  # too early
    s.first_speech_at = time.monotonic() - 25
    env.mgr._consent_tick(s, time.monotonic())
    env.mgr._consent_tick(s, time.monotonic())
    await asyncio.sleep(0.05)
    assert env.chan.sent == ["สลิดขออนุญาตจดนะคะ"] and s.consent == "pending"
    await env.mgr.end((1000, 77), "shutdown")


async def test_no_question_when_salid_was_addressed(env):
    s = await auto_session(env)
    s.first_speech_at, s.introduced = time.monotonic() - 60, True
    env.mgr._consent_tick(s, time.monotonic())
    await asyncio.sleep(0.02)
    assert env.chan.sent == [] and s.consent is None
    await env.mgr.end((1000, 77), "shutdown")


async def test_silence_after_the_question_means_recording(env):
    s = await auto_session(env)
    s.consent, s.consent_until = "pending", time.monotonic() - 1
    env.mgr._consent_tick(s, time.monotonic())
    assert s.recording == "recording" and s.consent == "assumed"
    await env.mgr.end((1000, 77), "shutdown")


async def test_a_yes_by_voice_starts_recording(env, monkeypatch):
    s = await auto_session(env)
    s.consent, s.consent_until = "pending", time.monotonic() + 30
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "ใช่ จดเลยครับ")
    await env.mgr.handle_segment(s, segment(10))
    assert s.recording == "recording" and len(s.transcript) == 1
    await env.mgr.end((1000, 77), "shutdown")


async def test_a_no_by_voice_stops_recording_and_acks_in_voice_but_stays(env, monkeypatch):
    s = await auto_session(env, voice_reply=True)
    s.transcript.add(env.bot.s.now(), "บอย", "เมื่อกี้พูดอะไรไว้")
    s.consent, s.consent_until = "pending", time.monotonic() + 30
    said = []

    async def fake_say(self_, sess, text):
        said.append(text)

    monkeypatch.setattr(type(env.mgr), "_say_text", fake_say)
    logged = []

    async def rec(*a, **k):
        logged.append(a)

    monkeypatch.setattr(env.bot, "record_voice", rec)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "ไม่ต้องจดนะ คุยเล่นเฉยๆ")
    await env.mgr.handle_segment(s, segment(10))
    assert s.recording == "not_recording" and len(s.transcript) == 0 and logged == []
    assert said == [t("meet_not_recording", "th")]
    assert (1000, 77) in env.mgr.sessions  # stays quietly
    await env.mgr.end((1000, 77), "command")
    assert env.chan.sent == []  # no summary while not recording


async def test_a_no_in_text_acks_in_text_and_privacy_leaves_with_cooldown(env):
    s = await auto_session(env)
    s.consent, s.consent_until = "pending", time.monotonic() + 30
    msg = SimpleNamespace(channel=env.chan, content="ไม่ต้องจด ขอส่วนตัวนะ")
    assert await env.mgr.on_chat_text(msg) is True
    await asyncio.sleep(0.1)
    assert (1000, 77) not in env.mgr.sessions  # left
    assert env.chan.sent and env.mgr._cooldown[(1000, 77)] > env.mgr._now()
    assert env.chan.sent[-1] in [t(f"meet_consent_no_{i}", "th") for i in (1, 2, 3)]


async def test_not_recording_can_resume(env):
    s = await auto_session(env)
    env.mgr.set_recording(s, False)
    assert (
        await env.mgr.on_chat_text(SimpleNamespace(channel=env.chan, content="เริ่มประชุมแล้ว"))
        is False
    )
    assert s.recording == "recording"
    await env.mgr.end((1000, 77), "shutdown")


def test_consent_answer_parsing():
    assert mt.MeetingManager.is_decline("ไม่ต้อง")
    assert mt.MeetingManager.is_decline("ไม่ครับ")
    assert mt.MeetingManager.is_decline("คุยเล่นเฉยๆ")
    assert not mt.MeetingManager.is_decline("ใช่ประชุมกัน")


# ------------------------------------------------------------------ notes
async def test_notes_for_decisions_not_chit_chat(env, monkeypatch):
    s = await start(env)
    now = env.bot.s.now()
    s.transcript.add(now, "แอน", "เมื่อวานกินอะไรกัน")
    monkeypatch.setattr(mt, "notes_from_transcript", lambda *a, **k: [])
    await env.mgr._note_pass(s)
    assert env.chan.sent == [t("meet_announce", "th")]  # only the (echo-on) fixture announcement
    s.transcript.add(now.replace(microsecond=1) + (now - now), "บอย", "ตกลงส่งรายงานวันศุกร์")
    monkeypatch.setattr(
        mt, "notes_from_transcript", lambda *a, **k: ["บอย ส่งรายงานภายในวันศุกร์", "x"]
    )
    s.last_note_pass = None
    await env.mgr._note_pass(s)
    assert "📌 บอย ส่งรายงานภายในวันศุกร์" in env.chan.sent
    await env.mgr.end((1000, 77), "shutdown")


async def test_post_note_is_sparing_and_blocked_when_not_recording(env):
    s = await start(env)
    assert (await env.mgr.post_note(s, "ตกลงใช้งบ 5,000 บาท"))["status"] == "posted"
    assert (await env.mgr.post_note(s, "ตกลงใช้งบ 5,000 บาท"))["error"] == "duplicate"
    s.last_note_at = 0
    env.mgr.set_recording(s, False)
    assert (await env.mgr.post_note(s, "อะไรก็ได้ยาวๆ"))["error"] == "not_recording"
    await env.mgr.end((1000, 77), "shutdown")


# ------------------------------------------------------------------ minutes destination
@pytest.mark.parametrize(
    "text, name",
    [
        ("ส่งมติการประชุมไปห้อง #บัญชี", "บัญชี"),
        ("ส่งสรุปประชุมไปที่ห้องเปเปอร์นะครับ", "เปเปอร์"),
        ("send the minutes to #accounting", "accounting"),
        ("post the summary in channel reports", "reports"),
    ],
)
def test_parse_minutes_destination(text, name):
    assert parse_minutes_destination(text).name == name


def test_not_a_minutes_instruction():
    assert parse_minutes_destination("วันนี้อากาศดีนะ") is None


async def test_voice_instruction_sets_destination_and_summary_goes_there_only(env, monkeypatch):
    from conftest import text_resp

    s = await start(env)
    acct, other = TChan(5, "บัญชี"), TChan(6, "general")
    add_text(env, acct, other)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "ส่งมติการประชุมไปห้องบัญชีด้วยนะครับ")
    await env.mgr.handle_segment(s, segment(10))
    assert s.minutes_channel_id == 5
    env.llm.models.script = [text_resp("**สรุป** ตกลงเลื่อน")]
    await env.mgr.end((1000, 77), "command")
    assert any("ตกลงเลื่อน" in m for m in acct.sent)
    assert not any("ตกลงเลื่อน" in m for m in env.chan.sent)


async def test_destination_the_room_cannot_see_is_refused(env):
    s = await start(env)
    add_text(env, TChan(5, "บัญชี", view=False))
    res = await env.mgr.set_minutes_destination(s, "บัญชี")
    assert res == {"error": "not_allowed"} and s.minutes_channel_id is None
    await env.mgr.end((1000, 77), "shutdown")


async def test_summary_goes_to_the_meeting_chat_when_no_destination(env):
    from conftest import text_resp

    s = await start(env)
    s.transcript.add(env.bot.s.now(), "แอน", "x")
    env.llm.models.script = [text_resp("สรุปเฉยๆ")]
    await env.mgr.end((1000, 77), "command")
    assert any("สรุปเฉยๆ" in m for m in env.chan.sent)


async def test_summary_is_stored_for_deliver_minutes(env):
    from conftest import text_resp

    from salid.coo_store import SQLiteDocStore

    env.bot.docs = SQLiteDocStore(":memory:")
    s = await start(env)
    s.attendee_ids.add(10)  # heard speaking
    s.transcript.add(env.bot.s.now(), "แอน", "x")
    env.llm.models.script = [text_resp("สรุปเก็บไว้")]
    await env.mgr.end((1000, 77), "command")
    rec = env.bot.docs.find(MINUTES, 1000)
    assert len(rec) == 1 and rec[0]["summary"] == "สรุปเก็บไว้" and 10 in rec[0]["attendee_ids"]


# ------------------------------------------------------------------ deliver_minutes (chat)
class Req:
    def __init__(self, id, name="ผู้ขอ"):
        self.id, self.display_name, self.guild = id, name, None


async def test_deliver_minutes_access_rules(env):
    from salid.coo_store import SQLiteDocStore
    from salid.deliver import DeliverMixin

    docs = SQLiteDocStore(":memory:")
    docs.put(MINUTES, 1000, "2026-10-14T12:00:00+00:00-77", {
        "voice_channel_id": 77, "voice_channel_name": "ห้องประชุม", "attendee_ids": [10],
        "started_at": "2026-10-14T11:00:00", "ended_at": "2026-10-14T12:00:00+00:00",
        "summary": "**สรุป** ตกลงเลื่อน"})  # fmt: skip
    dest = TChan(5, "บัญชี")
    nopost = TChan(6, "readonly", send=False)
    add_text(env, dest, nopost)
    env.guild.me = SimpleNamespace(id=999)
    bot = SimpleNamespace(docs=docs)

    def run(m, d, meeting="last"):
        return DeliverMixin._deliver_minutes(bot, env.guild, m, "th", d, meeting)

    env.chan.view = True
    env.chan.permissions_for = lambda who: SimpleNamespace(connect=True, view_channel=who.id != 12)
    attendee, outsider = Req(10), Req(12)
    res = await run(attendee, "บัญชี")
    assert res["status"] == "posted" and "ตกลงเลื่อน" in dest.sent[0]
    assert (await run(outsider, "บัญชี"))["error"] == "no_meeting"  # cannot see the meeting
    assert (await run(attendee, "readonly"))["error"] == "not_allowed"  # cannot post there
    assert (await run(attendee, "ไม่มีห้องนี้"))["error"] == "destination_not_found"


# ------------------------------------------------------------------ join / leave phrases, cooldown
@pytest.mark.parametrize(
    "text",
    ["เข้ามาประชุมหน่อยครับ", "สลิด เข้ามาประชุมหน่อย", "มาประชุมด้วยกัน", "เข้ามาหน่อย", "เข้าห้องเสียง",
     "มาฟังด้วย", "มาจดประชุม", "join us", "come in", "hop in", "please join the meeting",
     "ช่วยเข้าประชุมหน่อย"],
)  # fmt: skip
def test_join_phrases(text):
    assert parse_meeting_command(text) == "join"


@pytest.mark.parametrize(
    "text",
    ["ออกไปก่อนนะ", "ไปได้แล้ว", "ออกจากห้องได้เลย", "leave us", "you can go", "สลิด ออกประชุม"],
)
def test_leave_phrases(text):
    assert parse_meeting_command(text) == "leave"


async def test_cooldown_default_cleared_when_empty_and_overridden_by_join(env):
    from salid.config import Settings

    assert Settings().auto_join_cooldown_minutes == 10
    key = (1000, 77)
    env.mgr._cooldown[key] = env.mgr._now() + 600
    env.chan.members = []
    before = SimpleNamespace(channel=env.chan)
    member = make_member(env.guild, 10, "แอน")
    await env.mgr.on_voice_activity(member, before, SimpleNamespace(channel=None))
    assert key not in env.mgr._cooldown  # the room emptied: a new gathering, a new chance
    env.mgr._cooldown[key] = env.mgr._now() + 600
    a = make_member(env.guild, 11, "บอย", env.chan)
    await env.mgr.join(a)  # explicit request
    assert key not in env.mgr._cooldown and key in env.mgr.sessions
    await env.mgr.end(key, "shutdown")


async def test_join_voice_tool_for_a_natural_request(env):
    from salid.bot import SecretaryBot

    a = make_member(env.guild, 10, "แอน", env.chan)
    msg = SimpleNamespace(channel=SimpleNamespace(id=500))
    res = await SecretaryBot._join_voice_tool(env.bot, msg, a, "th")
    assert res == {"status": "joined", "channel": "ห้องประชุม"}
    nobody = make_member(env.guild, 12, "ไม่อยู่ห้อง")
    res = await SecretaryBot._join_voice_tool(env.bot, msg, nobody, "th")
    assert res["error"] == "not_in_voice"
    await env.mgr.end((1000, 77), "shutdown")


def test_join_voice_tool_is_exposed_to_the_agent():
    from salid.chat_tools import ChatActions, build_chat_tools

    calls = []
    tools = build_chat_tools(
        ChatActions(join_voice=lambda: {"status": "joined"}, leave_voice=lambda: {"result": "ok"}),
        calls,
    )
    names = [f.__name__ for f in tools]
    assert names == ["join_voice", "leave_voice"]
    assert tools[0]() == {"status": "joined"} and calls[0][0] == "join_voice"


# ------------------------------------------------------------------ listening / engaged
from salid.naming import find_call  # noqa: E402


@pytest.mark.parametrize(
    "text",
    ["สวัสดีครับ สลิด", "หวัดดีครับ สนิท", "สวัสดีค่ะสะหลิด", "สลิตครับ", "ซาลิด ช่วยหน่อย",
     "hey salid", "saleed how are you", "สวัสดีครับ ซะลิด", "ฮัลโหล สิริ ครับ", "Hey Siri",
     "น้องสนิทเป็นไงบ้าง", "สนิท ช่วยสรุปหน่อย", "ช่วยหน่อยนะ สนิท"],
)  # fmt: skip
def test_the_name_in_every_spelling_is_a_call(text):
    assert find_call(text) is not None
    assert mt.parse_wake(text) is not None


@pytest.mark.parametrize(
    "text", ["สนิทกันมาก", "เขาเป็นคนสนิทของผม", "เราสนิทกันตั้งแต่เด็ก", "ผมชอบสลิปเงินเดือน"]
)
def test_risky_mishearings_in_the_middle_of_a_sentence_are_not_calls(text):
    assert find_call(text) is None


async def listening_session(env, voice_mode="classic"):
    env.bot.s.meeting_text_echo = False
    s = await start(env)
    s.voice_mode = voice_mode
    return s


async def test_name_engages_and_every_next_utterance_is_answered_without_it(env, monkeypatch):
    s = await listening_session(env)
    import salid.qa

    asked = []
    monkeypatch.setattr(
        salid.qa,
        "ask",
        lambda settings, store, q, **kw: asked.append(q) or SimpleNamespace(text="ok"),
    )
    heard = iter(["พี่เห็นด้วยไหม", "สวัสดีครับ สลิด", "แล้วงบเท่าไหร่", "ขอบคุณครับ พอแล้ว", "แล้วไงต่อ"])
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: next(heard))
    for _ in range(5):
        await env.mgr.handle_segment(s, segment(10))
    # listening: ignored; name: answered + engaged; engaged: answered; dismissed: listening again
    assert asked == ["สวัสดีครับ", "แล้วงบเท่าไหร่"] or asked[1] == "แล้วงบเท่าไหร่"
    assert "แล้วไงต่อ" not in asked and s.engaged is False
    await env.mgr.end((1000, 77), "shutdown")


@pytest.mark.parametrize(
    "text",
    ["ขอบคุณครับ", "โอเคครับ ขอบคุณมากครับ", "พอแล้วครับ", "ยุติได้", "จบแค่นี้", "ออกได้ครับ",
     "thanks, that's all", "that's it", "we're good"],
)  # fmt: skip
def test_dismissal_phrases(text):
    assert mt.MeetingManager.is_dismissal(text)


@pytest.mark.parametrize(
    "text", ["ออกประชุม", "ออกจากห้องได้เลย", "ไปได้แล้ว", "leave the meeting", "ขอบคุณครับ แล้วเรื่องงบล่ะ"]
)
def test_leave_phrases_and_real_questions_are_not_dismissals(text):
    assert not mt.MeetingManager.is_dismissal(text)


async def test_dismissal_returns_to_listening_without_leaving(env, monkeypatch):
    s = await listening_session(env)
    env.mgr._set_engaged(s, True)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "ขอบคุณครับ")
    await env.mgr.handle_segment(s, segment(10))
    assert s.engaged is False and (1000, 77) in env.mgr.sessions
    await env.mgr.end((1000, 77), "shutdown")


async def test_leave_phrase_while_engaged_leaves_the_room(env, monkeypatch):
    s = await listening_session(env)
    env.mgr._set_engaged(s, True)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "ออกจากห้องได้เลย")
    await env.mgr.handle_segment(s, segment(10))
    await asyncio.sleep(0.1)
    assert (1000, 77) not in env.mgr.sessions


async def test_engaged_times_out_back_to_listening(env):
    s = await listening_session(env)
    env.mgr._set_engaged(s, True)
    env.mgr._consent_tick(s, time.monotonic() + 10)
    assert s.engaged is True
    env.mgr._consent_tick(s, time.monotonic() + 100)
    assert s.engaged is False
    await env.mgr.end((1000, 77), "shutdown")


async def test_live_forces_a_turn_when_the_model_stays_silent_after_the_name(lv):
    s, live = await lv.join(5)
    lv.bot.s.live_force_reply_seconds = 0.05
    live.cues = []
    sent = []

    async def send_client_content(turns=None, turn_complete=True):
        sent.append((turns.parts[0].text, turn_complete))

    live.send_client_content = send_client_content
    s.live.ensure_reply(time.monotonic())
    await until(lambda: sent)
    assert sent[0][1] is True and "เรียกชื่อ" in sent[0][0]
    # an answer already under way: no cue
    sent.clear()
    s.live._last_audio_at = time.monotonic() + 1
    s.live.ensure_reply(time.monotonic())
    await asyncio.sleep(0.2)
    assert sent == []
    await lv.mgr.end((1000, 77))


async def test_live_is_told_the_state(lv):
    s, live = await lv.join(5)
    sent = []

    async def send_client_content(turns=None, turn_complete=True):
        sent.append((turns.parts[0].text, turn_complete))

    live.send_client_content = send_client_content
    lv.mgr._set_engaged(s, True)
    await until(lambda: sent)
    assert "กำลังคุยกับคุณอยู่" in sent[0][0] and sent[0][1] is False
    lv.mgr._set_engaged(s, False)
    await until(lambda: len(sent) == 2)
    assert "กำลังฟัง" in sent[1][0]
    await lv.mgr.end((1000, 77))
