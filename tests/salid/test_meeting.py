"""Voice meetings, fully offline: segmenter, wake word, transcript window, command guards,
wake-word answers and the summary on leave. No Discord connection, no Vertex."""

import asyncio
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from conftest import FakeClient, FixedClockSettings, json_resp, text_resp

from salid import meeting as mt
from salid.bot import SecretaryBot
from salid.guild_settings import GuildSettingsUpdate, MemoryGuildStore
from salid.meeting import (
    PCM_BYTES_PER_SECOND,
    Segmenter,
    Transcript,
    parse_meeting_command,
    parse_wake,
)

FRAME = bytes(PCM_BYTES_PER_SECOND // 50)  # 20 ms of silence-valued PCM
T0 = datetime(2026, 10, 14, 12, 0, tzinfo=ZoneInfo("Asia/Bangkok"))


def speak(seg, uid, start, seconds, step=0.02):
    """Feed `seconds` of continuous speech starting at `start`; return finished segments."""
    out, t = [], start
    for _ in range(round(seconds / step)):
        out += seg.feed(uid, FRAME, t)
        t += step
    return out


# ------------------------------------------------------------------ segmenter
def test_segmenter_cuts_on_silence():
    s = Segmenter()
    assert speak(s, 1, 0.0, 2.0) == []
    assert s.tick(2.0 + 1.0) == []  # a 1.0 s pause is not enough
    (seg,) = s.tick(2.0 + 1.3)
    assert seg.user_id == 1 and 1.9 < seg.duration_s < 2.1


def test_segmenter_cuts_when_the_speaker_resumes_after_a_pause():
    s = Segmenter()
    speak(s, 1, 0.0, 1.0)
    done = speak(s, 1, 3.0, 1.0)  # 2 s later: the old segment closes, a new one starts
    assert len(done) == 1 and 0.9 < done[0].duration_s < 1.1
    assert len(s.flush_all()) == 1


def test_segmenter_cuts_at_max_length():
    s = Segmenter(max_s=20.0)
    done = speak(s, 1, 0.0, 45.0)
    assert [round(d.duration_s) for d in done] == [20, 20]
    assert 4.5 < s.flush_all()[0].duration_s < 5.5


def test_segmenter_drops_short_segments():
    s = Segmenter()
    speak(s, 1, 0.0, 0.4)
    assert s.tick(5.0) == [] and s.bufs == {}


def test_segmenter_keeps_speakers_apart():
    s = Segmenter()
    speak(s, 1, 0.0, 1.0)
    speak(s, 2, 0.5, 1.0)
    assert {x.user_id for x in s.tick(10.0)} == {1, 2}


def test_segmenter_pads_short_pauses():
    s = Segmenter()
    s.feed(1, FRAME, 0.0)
    s.feed(1, FRAME * 25, 0.52)  # 0.5 s gap before this packet
    (seg,) = s.flush_all()
    assert seg.duration_s > 0.9  # 0.02 + 0.5 s of padding + 0.5 s of audio


def test_rms_ignores_silence():
    assert mt.rms(FRAME) == 0
    loud = (b"\xff\x3f" * 480) * 2
    assert mt.rms(loud) > 1000


# ------------------------------------------------------------------ wake word
@pytest.mark.parametrize(
    ("text", "question"),
    [
        ("สลิด เมื่อกี้ตกลงอะไรกัน", "เมื่อกี้ตกลงอะไรกัน"),
        ("สลิดครับ ช่วยสรุปหน่อย", "ช่วยสรุปหน่อย"),
        ("Salid, what did we decide?", "what did we decide"),
        ("SALID เดือนนี้ใช้จ่ายไปเท่าไร", "เดือนนี้ใช้จ่ายไปเท่าไร"),
        ("สะลิด วันนี้วันอะไร", "วันนี้วันอะไร"),
        ("สลิตร ขอสรุปหน่อย", "ขอสรุปหน่อย"),
        ("ช่วยสรุปให้หน่อยสลิด", "ช่วยสรุปให้หน่อย"),
        ("สลิด", ""),
        ("สลิดครับ", ""),
        ("เรามาประชุมกันเลย", None),
        ("", None),
    ],
)
def test_parse_wake(text, question):
    assert parse_wake(text) == question


@pytest.mark.parametrize(
    ("text", "cmd"),
    [
        ("เข้าประชุม", "join"),
        ("เข้าประชุมหน่อย", "join"),
        ("/เข้าประชุม", "join"),
        ("join", "join"),
        ("ออกประชุม", "leave"),
        ("Leave", "leave"),
        ("เดือนนี้จ่ายอะไรบ้างและจะเข้าประชุมวันไหน ลองสรุปให้หน่อยนะ", None),
        ("สวัสดี", None),
    ],
)
def test_parse_meeting_command(text, cmd):
    assert parse_meeting_command(text) == cmd


# ------------------------------------------------------------------ transcript
def test_transcript_history_window_and_format():
    t = Transcript()
    t.add(T0, "แอน", "เก่ามาก")
    t.add(T0 + timedelta(minutes=8), "บอย", "ขอเลื่อนเป็นวันศุกร์")
    t.add(T0 + timedelta(minutes=9), "แอน", "ตกลงครับ")
    now = T0 + timedelta(minutes=10)
    assert t.history(now) == "บอย: ขอเลื่อนเป็นวันศุกร์\nแอน: ตกลงครับ"
    assert t.history(now, max_chars=15) == "แอน: ตกลงครับ"  # newest kept when trimmed


def test_transcript_is_bounded():
    t = Transcript(keep_s=3600, max_lines=3)
    for i in range(5):
        t.add(T0 + timedelta(seconds=i), "a", f"m{i}")
    assert [ln.text for ln in t.lines] == ["m2", "m3", "m4"]
    t.add(T0 + timedelta(hours=2), "a", "late")
    assert [ln.text for ln in t.lines] == ["late"]


# ------------------------------------------------------------------ manager fakes
ME = SimpleNamespace(id=999, mention="<@999>")


class FakeVoiceChannel:
    def __init__(self, id=77, name="ห้องประชุม", connect=True):
        self.id, self.name, self.members, self.sent = id, name, [], []
        self._connect = connect
        self.guild = None

    def permissions_for(self, _):
        return SimpleNamespace(connect=self._connect)

    async def send(self, text, **_):
        self.sent.append(text)


class FakeVC:
    def __init__(self):
        self.listening, self.disconnected = None, False
        self._connection = None
        self.connected, self.playing, self.played, self.after = True, False, [], None

    def is_connected(self):
        return self.connected

    def is_playing(self):
        return self.playing

    def play(self, source, after=None):
        self.played.append(source)
        self.playing, self.after = True, after

    def stop(self):
        self.playing = False

    def listen(self, sink):
        self.listening = sink

    def stop_listening(self):
        self.listening = None

    async def disconnect(self):
        self.disconnected = True


class FakeGuild:
    def __init__(self, id=1000):
        self.id, self.me, self.owner_id, self.members = id, ME, 1, {}

    def get_member(self, uid):
        return self.members.get(uid)


def make_member(guild, uid, name, channel=None, **perms):
    m = SimpleNamespace(
        id=uid, guild=guild, display_name=name, mention=f"<@{uid}>", bot=False,
        roles=[], voice=SimpleNamespace(channel=channel) if channel else None,
        guild_permissions=SimpleNamespace(administrator=False, manage_guild=False, **perms),
    )  # fmt: skip
    guild.members[uid] = m
    return m


@pytest.fixture
def env(monkeypatch):
    guild = FakeGuild()
    chan = FakeVoiceChannel()
    chan.guild = guild
    llm = FakeClient([])
    bot = SecretaryBot(
        FixedClockSettings(exec_user_ids={5}, voice_reply=False),
        None,
        llm,
        guilds=MemoryGuildStore(),
    )
    monkeypatch.setattr(SecretaryBot, "user", property(lambda s: ME))
    mgr = bot.meetings
    vc = FakeVC()

    async def connect(channel):
        return vc

    monkeypatch.setattr(mgr, "_connect", connect)
    return SimpleNamespace(guild=guild, chan=chan, llm=llm, bot=bot, mgr=mgr, vc=vc)


# ------------------------------------------------------------------ join / leave guards
async def test_join_requires_being_in_a_voice_channel(env):
    m = make_member(env.guild, 10, "แอน")
    assert await env.mgr.join(m) == mt.NOT_IN_VOICE_MESSAGE
    assert env.mgr.sessions == {}


async def test_join_needs_connect_permission(env):
    env.chan._connect = False
    m = make_member(env.guild, 10, "แอน", env.chan)
    assert await env.mgr.join(m) == mt.NO_PERMISSION_MESSAGE


async def test_join_respects_the_feature_toggle(env):
    env.bot.guilds_store.save_settings(
        1000, GuildSettingsUpdate(features=dict(voice_meetings=False)), "x"
    )
    await asyncio.to_thread(env.bot.team.current, 1000)
    m = make_member(env.guild, 10, "แอน", env.chan)
    assert await env.mgr.join(m) == mt.MEETINGS_OFF_MESSAGE


async def test_join_announces_listens_and_blocks_a_second_meeting(env):
    m = make_member(env.guild, 10, "แอน", env.chan)
    assert "ห้องประชุม" in await env.mgr.join(m)
    assert env.chan.sent == [mt.ANNOUNCE_MESSAGE]
    assert "สลิดเข้ามาฟังและจดบันทึกการประชุมครับ" in env.chan.sent[0]
    assert isinstance(env.vc.listening, mt.MeetingSink)
    again = await env.mgr.join(m)
    assert "กำลังอยู่ในการประชุม" in again
    await env.mgr.end(1000, "command")


async def test_leave_without_meeting(env):
    m = make_member(env.guild, 10, "แอน", env.chan)
    assert await env.mgr.leave(m) == mt.NO_MEETING_MESSAGE


async def test_leave_denied_for_outsiders(env):
    a = make_member(env.guild, 10, "แอน", env.chan)
    outsider = make_member(env.guild, 11, "ซี")
    await env.mgr.join(a)
    assert await env.mgr.leave(outsider) == mt.LEAVE_DENIED_MESSAGE
    assert 1000 in env.mgr.sessions
    await env.mgr.end(1000)


# ------------------------------------------------------------------ transcription + wake word
def segment(uid):
    return mt.Segment(uid, FRAME * 100, 0.0, 2.0)


async def start(env):
    a = make_member(env.guild, 10, "แอน", env.chan)
    make_member(env.guild, 11, "บอย", env.chan)
    await env.mgr.join(a)
    return env.mgr.sessions[1000]


async def test_segment_is_transcribed_with_the_speaker_name(env, monkeypatch):
    s = await start(env)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "เรามาเริ่มกันเลย")
    await env.mgr.handle_segment(s, segment(10))
    assert [(ln.speaker, ln.text) for ln in s.transcript.lines] == [("แอน", "เรามาเริ่มกันเลย")]
    assert env.chan.sent == [mt.ANNOUNCE_MESSAGE]  # no wake word, no reply
    await env.mgr.end(1000)


async def test_wake_word_answers_with_transcript_history(env, monkeypatch):
    s = await start(env)
    asked = {}

    def fake_ask(settings, store, q, **kw):
        asked.update(q=q, **kw)
        return SimpleNamespace(text="ตกลงเลื่อนเป็นวันศุกร์ครับ", rounds=0, seconds=0.1)

    import salid.qa

    monkeypatch.setattr(salid.qa, "ask", fake_ask)
    s.transcript.add(env.bot.s.now(), "บอย", "ขอเลื่อนเป็นวันศุกร์นะ")
    spoken = iter(["สลิด เมื่อกี้ตกลงอะไรกัน"])
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: next(spoken))
    await env.mgr.handle_segment(s, segment(11))
    assert asked["q"] == "เมื่อกี้ตกลงอะไรกัน"
    assert asked["guild_id"] == 1000 and asked["user_id"] == 11 and asked["is_exec"] is False
    assert "บอย: ขอเลื่อนเป็นวันศุกร์นะ" in asked["history"]
    assert env.chan.sent[-1] == "<@11> ตกลงเลื่อนเป็นวันศุกร์ครับ"
    await env.mgr.end(1000)


async def test_wake_word_exec_flag_follows_the_speaker(env, monkeypatch):
    s = await start(env)
    boss = make_member(env.guild, 5, "เจ้านาย", env.chan)
    flags = []
    import salid.qa

    monkeypatch.setattr(
        salid.qa, "ask", lambda *a, **kw: flags.append(kw["is_exec"]) or SimpleNamespace(text="ok")
    )
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด สรุปให้หน่อย")
    await env.mgr.handle_segment(s, segment(boss.id))
    assert flags == [True]
    await env.mgr.end(1000)


async def test_bare_wake_word_takes_the_next_segment_as_the_question(env, monkeypatch):
    s = await start(env)
    questions = []
    import salid.qa

    monkeypatch.setattr(
        salid.qa,
        "ask",
        lambda s_, st, q, **kw: questions.append(q) or SimpleNamespace(text="ได้ครับ"),
    )
    texts = iter(["สลิด", "วันนี้มีนัดอะไรบ้าง"])
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: next(texts))
    await env.mgr.handle_segment(s, segment(10))
    assert env.chan.sent[-1].endswith(mt.WAKE_EMPTY_MESSAGE)
    await env.mgr.handle_segment(s, segment(10))
    assert questions == ["วันนี้มีนัดอะไรบ้าง"]
    await env.mgr.end(1000)


def test_context_for_the_meeting_text_chat(env):
    s = SimpleNamespace(text_channel=env.chan, transcript=Transcript())
    s.transcript.add(env.bot.s.now(), "แอน", "ส่งงานวันศุกร์")
    env.mgr.sessions[1000] = s
    assert env.mgr.context_for(env.chan.id) == "แอน: ส่งงานวันศุกร์"
    assert env.mgr.context_for(123) == ""


# ------------------------------------------------------------------ summary on leave
async def test_leave_posts_a_summary_and_drops_the_transcript(env, monkeypatch):
    s = await start(env)
    s.transcript.add(env.bot.s.now(), "แอน", "ให้บอยส่งรายงานวันศุกร์")
    env.llm.models.script = [text_resp("**สรุปการประชุม**\nบอยส่งรายงานวันศุกร์")]
    await env.mgr.end(1000, "command")
    assert env.vc.disconnected and 1000 not in env.mgr.sessions
    assert "สรุปการประชุม" in env.chan.sent[-1]
    prompt = env.llm.models.calls[0][1][0]
    assert "แอน: ให้บอยส่งรายงานวันศุกร์" in prompt and "งานที่ต้องทำ" in prompt
    assert len(s.transcript) == 0  # nothing persisted


async def test_leave_with_nothing_heard(env):
    await start(env)
    await env.mgr.end(1000, "empty")
    assert env.chan.sent[-1] == mt.NO_TRANSCRIPT_MESSAGE
    assert env.llm.models.calls == []


async def test_long_summary_is_split_for_discord(env):
    s = await start(env)
    s.transcript.add(env.bot.s.now(), "แอน", "x")
    env.llm.models.script = [text_resp("ก" * 4500)]
    await env.mgr.end(1000, "command")
    parts = env.chan.sent[1:]
    assert len(parts) == 3 and all(len(p) <= 2000 for p in parts)


# ------------------------------------------------------------------ transcribe_pcm
def test_transcribe_pcm_parses_json_and_filters_noise(monkeypatch):
    monkeypatch.setattr(mt, "to_flac", lambda wav: b"flac")
    ok = FakeClient([json_resp({"text": " สวัสดีครับ "})])
    assert mt.transcribe_pcm(ok, "m", FRAME) == "สวัสดีครับ"
    junk = FakeClient([json_resp({"text": " ... "})])
    assert mt.transcribe_pcm(junk, "m", FRAME) == ""
    plain = FakeClient([text_resp("ข้อความธรรมดา")])  # model ignored the JSON schema
    assert mt.transcribe_pcm(plain, "m", FRAME) == "ข้อความธรรมดา"


# ------------------------------------------------------------------ voice sink (real Opus, fake DAVE)
def _opus_frame():
    import math
    import struct

    from discord.opus import Encoder

    enc = Encoder()
    n = Encoder.SAMPLES_PER_FRAME
    pcm = b"".join(
        struct.pack("<hh", v, v)
        for v in (int(8000 * math.sin(2 * math.pi * 440 * i / 48000)) for i in range(n))
    )
    return enc.encode(pcm, n)


class _Session:
    def __init__(self, conn):
        self.vc, self.got = SimpleNamespace(_connection=conn), []

    def push_threadsafe(self, uid, pcm):
        self.got.append((uid, pcm))


def _data(opus):
    return SimpleNamespace(opus=opus)


def test_sink_decodes_plain_opus_when_dave_is_off():
    opus = _opus_frame()
    s = _Session(SimpleNamespace(dave_session=None, dave_protocol_version=0))
    sink = mt.MeetingSink(s)
    sink.write(SimpleNamespace(id=7), _data(opus))
    sink.write(SimpleNamespace(id=7), _data(b"\xf8\xff\xfe"))  # silence frame: skipped
    sink.write(None, _data(opus))  # unknown speaker: skipped
    assert sink.wants_opus() and len(s.got) == 1 and s.got[0][0] == 7
    assert len(s.got[0][1]) == 3840  # 20 ms, 48 kHz, stereo, s16


def test_sink_removes_the_dave_layer_per_speaker(monkeypatch):
    import davey

    opus, seen = _opus_frame(), []

    class Dave:
        ready = True

        def decrypt(self, uid, media, data):
            seen.append((uid, media, data))
            if data == b"junk":
                raise RuntimeError("decrypt failed")
            return opus

    s = _Session(SimpleNamespace(dave_session=Dave(), dave_protocol_version=1))
    sink = mt.MeetingSink(s)
    sink.write(SimpleNamespace(id=7), _data(b"ciphertext"))
    sink.write(SimpleNamespace(id=7), _data(b"junk"))  # undecryptable: dropped, no crash
    assert seen[0] == (7, davey.MediaType.audio, b"ciphertext")
    assert len(s.got) == 1 and sink.dave_dropped == 1


def test_sink_survives_undecodable_audio():
    s = _Session(SimpleNamespace(dave_session=None, dave_protocol_version=0))
    mt.MeetingSink(s).write(SimpleNamespace(id=7), _data(b"\x01\x02\x03garbage"))
    assert s.got == [] or len(s.got) == 1  # either dropped or decoded; never raises


# ------------------------------------------------------------------ spoken replies
PCM = bytes(24000 * 2)  # 1 s of 24 kHz mono silence


@pytest.fixture
def voice(env, monkeypatch):
    """voice_reply on, with speech.py's Vertex/ffmpeg parts faked."""
    env.bot.s.voice_reply = True
    calls = []
    monkeypatch.setattr(
        mt, "spoken_text", lambda c, m, q, a: calls.append(("spoken", m, q, a)) or "พูดสั้น ๆ"
    )
    monkeypatch.setattr(mt, "synthesize", lambda c, m, v, t: calls.append(("tts", m, v, t)) or PCM)
    monkeypatch.setattr(mt, "pcm_source", lambda pcm: ("source", len(pcm)))
    env.calls = calls
    return env


def fake_ask(monkeypatch, text="ตกลงเลื่อนเป็นวันศุกร์ครับ"):
    import salid.qa

    monkeypatch.setattr(salid.qa, "ask", lambda *a, **kw: SimpleNamespace(text=text))


async def test_speak_plays_and_clears_speaking_after_the_callback(voice):
    s = await start(voice)
    task = asyncio.create_task(voice.mgr.speak(s, "คำตอบเต็ม", "คำถาม"))
    for _ in range(50):
        await asyncio.sleep(0.01)
        if voice.vc.played:
            break
    assert voice.vc.played == [("source", len(PCM))] and s.speaking is True
    assert voice.calls[0] == ("spoken", "gemini-3.5-flash-lite", "คำถาม", "คำตอบเต็ม")
    assert voice.calls[1] == ("tts", "gemini-2.5-flash-tts", "Puck", "พูดสั้น ๆ")
    voice.vc.playing = False
    voice.vc.after(None)  # the player thread says it is done
    await task
    assert s.speaking is False and s.speaking_until > 0
    await voice.mgr.end(1000)


async def test_speak_skips_when_not_connected_or_already_playing(voice):
    s = await start(voice)
    voice.vc.connected = False
    await voice.mgr.speak(s, "x")
    voice.vc.connected, voice.vc.playing = True, True
    await voice.mgr.speak(s, "x")
    assert voice.calls == [] and voice.vc.played == []
    voice.vc.playing = False
    await voice.mgr.end(1000)


async def test_speak_skips_when_the_meeting_is_ending(voice):
    s = await start(voice)
    s.ending = True
    await voice.mgr.speak(s, "x")
    assert voice.calls == [] and voice.vc.played == []
    s.ending = False
    await voice.mgr.end(1000)


async def test_speak_swallows_tts_errors(voice, monkeypatch):
    s = await start(voice)

    def boom(*a):
        raise RuntimeError("TTS returned no audio")

    monkeypatch.setattr(mt, "synthesize", boom)
    await voice.mgr.speak(s, "x")  # must not raise
    assert voice.vc.played == [] and s.speaking is False
    await voice.mgr.end(1000)


async def test_speak_times_out_stops_playback_and_clears_speaking(voice, monkeypatch):
    s = await start(voice)
    monkeypatch.setattr(mt, "pcm_duration_s", lambda pcm: 0.05)
    monkeypatch.setattr(mt, "SPEAK_TIMEOUT_PAD_S", 0.0)
    await voice.mgr.speak(s, "x")  # the after-callback never fires
    assert voice.vc.played and voice.vc.playing is False  # stopped
    assert s.speaking is False and s.speaking_until > 0
    await voice.mgr.end(1000)


async def test_answer_posts_text_then_speaks(voice, monkeypatch):
    s = await start(voice)
    fake_ask(monkeypatch)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด เมื่อกี้ตกลงอะไรกัน")
    await voice.mgr.handle_segment(s, segment(11))
    assert voice.chan.sent[-1] == "<@11> ตกลงเลื่อนเป็นวันศุกร์ครับ"
    for _ in range(50):
        await asyncio.sleep(0.01)
        if voice.vc.played:
            break
    assert voice.calls[0][2:] == ("เมื่อกี้ตกลงอะไรกัน", "ตกลงเลื่อนเป็นวันศุกร์ครับ")
    voice.vc.playing = False
    voice.vc.after(None)
    await voice.mgr.end(1000)


async def test_answer_does_not_speak_when_voice_reply_is_off(voice, monkeypatch):
    voice.bot.s.voice_reply = False
    s = await start(voice)
    fake_ask(monkeypatch)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด สรุปให้หน่อย")
    await voice.mgr.handle_segment(s, segment(11))
    await asyncio.sleep(0.05)
    assert voice.calls == [] and voice.vc.played == []
    assert voice.chan.sent[-1].endswith("ตกลงเลื่อนเป็นวันศุกร์ครับ")
    await voice.mgr.end(1000)


async def test_answer_text_still_posts_when_speech_fails(voice, monkeypatch):
    s = await start(voice)
    fake_ask(monkeypatch)

    def boom(*a):
        raise RuntimeError("tts down")

    monkeypatch.setattr(mt, "synthesize", boom)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด สรุปให้หน่อย")
    await voice.mgr.handle_segment(s, segment(11))
    await asyncio.sleep(0.05)
    assert voice.chan.sent[-1] == "<@11> ตกลงเลื่อนเป็นวันศุกร์ครับ"
    assert voice.vc.played == [] and s.speaking is False
    await voice.mgr.end(1000)


async def test_echo_guard_ignores_wake_words_while_speaking_but_keeps_the_line(voice, monkeypatch):
    s = await start(voice)
    asked = []
    import salid.qa

    monkeypatch.setattr(
        salid.qa, "ask", lambda *a, **kw: asked.append(a[2]) or SimpleNamespace(text="ok")
    )
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด สรุปให้หน่อย")
    s.speaking = True
    await voice.mgr.handle_segment(s, segment(11))
    assert asked == [] and [ln.text for ln in s.transcript.lines] == ["สลิด สรุปให้หน่อย"]
    s.speaking, s.speaking_until = False, time.monotonic() + 5  # inside the cooldown
    await voice.mgr.handle_segment(s, segment(11))
    assert asked == [] and len(s.transcript) == 2
    await voice.mgr.end(1000)


async def test_echo_guard_uses_the_audio_timestamps_when_transcription_lags(voice, monkeypatch):
    s = await start(voice)
    asked = []
    import salid.qa

    monkeypatch.setattr(
        salid.qa, "ask", lambda *a, **kw: asked.append(a[2]) or SimpleNamespace(text="ok")
    )
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด สรุปให้หน่อย")
    now = time.monotonic()
    # Salid spoke from now-20 to now-6 (cooldown over); the segment was heard at now-15.
    s.speaking_since, s.speaking_until = now - 20, now - 6
    echo = mt.Segment(11, FRAME * 100, now - 15, 2.0)
    await voice.mgr.handle_segment(s, echo)
    assert asked == []
    later = mt.Segment(11, FRAME * 100, now - 4, 2.0)  # after the cooldown: a real question
    await voice.mgr.handle_segment(s, later)
    assert asked == ["สรุปให้หน่อย"]
    await voice.mgr.end(1000)


async def test_end_stops_playback_before_disconnecting(voice):
    await start(voice)
    voice.vc.playing = True
    await voice.mgr.end(1000)
    assert voice.vc.playing is False and voice.vc.disconnected


async def test_speak_skips_when_voice_reply_is_disabled(voice):
    s = await start(voice)
    voice.bot.s.voice_reply = False
    await voice.mgr.speak(s, "x")
    assert voice.calls == [] and voice.vc.played == []
    await voice.mgr.end(1000)


async def test_concurrent_speak_calls_only_generate_one_reply(voice, monkeypatch):
    import threading

    s = await start(voice)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def slow_spoken(*args):
        calls.append(args)
        entered.set()
        release.wait(timeout=1)
        return "พูดสั้น ๆ"

    monkeypatch.setattr(mt, "spoken_text", slow_spoken)
    first = asyncio.create_task(voice.mgr.speak(s, "คำตอบแรก"))
    assert await asyncio.to_thread(entered.wait, 1)
    second = asyncio.create_task(voice.mgr.speak(s, "คำตอบที่สอง"))
    await asyncio.sleep(0.05)
    release.set()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if voice.vc.played:
            break
    voice.vc.playing = False
    voice.vc.after(None)
    await asyncio.gather(first, second)
    assert len(calls) == 1
    assert len([call for call in voice.calls if call[0] == "tts"]) == 1
    await voice.mgr.end(1000)


async def test_speak_cleans_ffmpeg_source_when_play_rejects_it(voice, monkeypatch):
    s = await start(voice)

    class Source:
        cleaned = False

        def cleanup(self):
            self.cleaned = True

    source = Source()
    monkeypatch.setattr(mt, "pcm_source", lambda pcm: source)

    def reject(source, after=None):
        raise RuntimeError("player refused source")

    monkeypatch.setattr(voice.vc, "play", reject)
    await voice.mgr.speak(s, "x")
    assert source.cleaned is True
    assert s.speaking is False
    await voice.mgr.end(1000)
