"""Voice meetings, fully offline: segmenter, wake word, transcript window, command guards,
wake-word answers and the summary on leave. No Discord connection, no Vertex."""

import asyncio
import time
from array import array
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from conftest import FakeClient, FixedClockSettings, json_resp, text_resp

from salid import meeting as mt
from salid.bot import SecretaryBot
from salid.guild_settings import GuildSettingsUpdate, MemoryGuildStore
from salid.i18n import t
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
    speak(s, 1, 0.0, 0.3)
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
        ("Hey Salid, what did we spend?", "what did we spend"),
        ("hey salid what did we spend this month", "what did we spend this month"),
        ("what did we spend, salid", "what did we spend"),
        ("sa lid summarize the meeting", "summarize the meeting"),
        ("สลิด", ""),
        ("สลิดครับ", ""),
        ("hey salid", ""),
        ("Hi, Salid", ""),
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
        ("Join meeting", "join"),
        ("join the meeting", "join"),
        ("  JOIN   THE  MEETING! ", "join"),
        ("come to the meeting", "join"),
        ("come to meeting please", "join"),
        ("ออกประชุม", "leave"),
        ("Leave", "leave"),
        ("leave meeting", "leave"),
        ("leave the meeting", "leave"),
        ("end meeting", "leave"),
        ("Stop the meeting", "leave"),
        (", leave.", "leave"),
        ("joined the call yesterday", None),
        ("what did we decide in the meeting", None),
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

    async def disconnect(self, force=False):
        self.disconnected = True


class FakeGuild:
    def __init__(self, id=1000):
        self.id, self.me, self.owner_id, self.members = id, ME, 1, {}

    def get_member(self, uid):
        return self.members.get(uid)

    def get_channel(self, cid):
        return None


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
        FixedClockSettings(
            exec_user_ids={5},
            voice_reply=False,
            voice_mode="classic",
            meeting_text_echo=True,
            voice_state_posts=False,
        ),
        None,
        llm,
        guilds=MemoryGuildStore(),
    )
    monkeypatch.setattr(SecretaryBot, "user", property(lambda s: ME))
    mgr = bot.meetings
    vc = FakeVC()

    async def connect(body, channel):
        return vc

    monkeypatch.setattr(mgr, "_connect", connect)
    return SimpleNamespace(guild=guild, chan=chan, llm=llm, bot=bot, mgr=mgr, vc=vc)


# ------------------------------------------------------------------ join / leave guards
async def test_join_requires_being_in_a_voice_channel(env):
    m = make_member(env.guild, 10, "แอน")
    assert await env.mgr.join(m) == t("meet_not_in_voice", "th")
    assert env.mgr.sessions == {}


async def test_join_needs_connect_permission(env):
    env.chan._connect = False
    m = make_member(env.guild, 10, "แอน", env.chan)
    reply = await env.mgr.join(m)
    assert "Connect" in reply and "Manage" not in reply and "ตอนนี้สลิดยังขาดสิทธิ์" in reply


async def test_join_respects_the_feature_toggle(env):
    env.bot.guilds_store.save_settings(
        1000, GuildSettingsUpdate(features=dict(voice_meetings=False)), "x"
    )
    await asyncio.to_thread(env.bot.team.current, 1000)
    m = make_member(env.guild, 10, "แอน", env.chan)
    assert await env.mgr.join(m) == t("meet_off", "th")


async def test_join_announces_listens_and_blocks_a_second_meeting(env):
    m = make_member(env.guild, 10, "แอน", env.chan)
    assert "ห้องประชุม" in await env.mgr.join(m)
    assert env.chan.sent == [t("meet_announce", "th")]
    assert "สลิดเข้ามาฟังและจดบันทึกการประชุมค่ะ" in env.chan.sent[0]
    assert isinstance(env.vc.listening, mt.MeetingSink)
    again = await env.mgr.join(m)
    assert "กำลังอยู่ในการประชุม" in again
    await env.mgr.end((1000, 77), "command")


async def test_leave_without_meeting(env):
    m = make_member(env.guild, 10, "แอน", env.chan)
    assert await env.mgr.leave(m) == t("meet_none", "th")


async def test_leave_denied_for_outsiders(env):
    a = make_member(env.guild, 10, "แอน", env.chan)
    outsider = make_member(env.guild, 11, "ซี")
    await env.mgr.join(a)
    assert await env.mgr.leave(outsider) == t("meet_leave_denied", "th")
    assert (1000, 77) in env.mgr.sessions
    await env.mgr.end((1000, 77))


# ------------------------------------------------------------------ transcription + wake word
SPEECH_FRAME = array("h", [800] * (PCM_BYTES_PER_SECOND // 100)).tobytes()  # 20 ms, RMS 800


LOUD_FRAME = array("h", [2400] * (PCM_BYTES_PER_SECOND // 100)).tobytes()


SPEECH_PCM = (SPEECH_FRAME + LOUD_FRAME) * 50  # 2 s, varying energy


def segment(uid):
    # speech-like energy (frames alternate RMS 800 / 2400): name hits behind silent or steady
    # audio are rejected by the energy gate
    return mt.Segment(uid, SPEECH_PCM, 0.0, 2.0)


async def start(env):
    a = make_member(env.guild, 10, "แอน", env.chan)
    make_member(env.guild, 11, "บอย", env.chan)
    await env.mgr.join(a)
    return env.mgr.sessions[(1000, 77)]


async def test_segment_is_transcribed_with_the_speaker_name(env, monkeypatch):
    s = await start(env)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "เรามาเริ่มกันเลย")
    await env.mgr.handle_segment(s, segment(10))
    assert [(ln.speaker, ln.text) for ln in s.transcript.lines] == [("แอน", "เรามาเริ่มกันเลย")]
    assert env.chan.sent == [t("meet_announce", "th")]  # no wake word, no reply
    await env.mgr.end((1000, 77))


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
    await env.mgr.end((1000, 77))


async def test_wake_word_answers_get_memory_scoped_to_the_voice_channel(env, monkeypatch):
    s = await start(env)
    env.chan.members = [env.guild.members[10], env.guild.members[11]]  # the room
    scoped, asked = [], {}

    async def tools_for(guild, voice_channel, humans, *also, asker=None):
        scoped.append((guild.id, asker.id, voice_channel.id, also, sorted(h.id for h in humans)))
        return "MEMORY"

    monkeypatch.setattr(env.bot, "_room_memory_tools", tools_for)
    import salid.qa

    monkeypatch.setattr(
        salid.qa, "ask", lambda *a, **kw: asked.update(kw) or SimpleNamespace(text="ok")
    )
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด เมื่อวานตกลงงบไว้เท่าไร")
    await env.mgr.handle_segment(s, segment(11))
    # the speaker is scoped with the whole room, against the voice channel's chat readers
    assert scoped == [(1000, 11, env.chan.id, (), [10, 11])] and asked["memory"] == "MEMORY"
    await env.mgr.end((1000, 77))


async def test_voice_memory_scope_also_covers_the_fallback_channel(env, monkeypatch):
    # answers fall back to the channel the meeting was started from: its readers bound the scope too
    fallback = SimpleNamespace(id=55, name="ทั่วไป", parent_id=None, sent=[])

    async def fallback_send(text):
        fallback.sent.append(text)

    fallback.send = fallback_send
    a = make_member(env.guild, 10, "แอน", env.chan)
    make_member(env.guild, 11, "บอย", env.chan)
    await env.mgr.join(a, fallback_channel=fallback)
    s = env.mgr.sessions[(1000, 77)]
    scoped = []

    async def tools_for(guild, voice_channel, humans, *also, asker=None):
        scoped.append((voice_channel.id, also))
        return None

    monkeypatch.setattr(env.bot, "_room_memory_tools", tools_for)
    import salid.qa

    monkeypatch.setattr(salid.qa, "ask", lambda *a, **kw: SimpleNamespace(text="ok"))
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด เมื่อวานตกลงอะไร")
    await env.mgr.handle_segment(s, segment(11))
    assert scoped == [(env.chan.id, (55,))]
    await env.mgr.end((1000, 77))


async def test_voice_answers_log_a_metrics_line(env, monkeypatch, caplog):
    s = await start(env)
    import salid.qa

    monkeypatch.setattr(
        salid.qa, "ask", lambda *a, **kw: SimpleNamespace(text="ok", rounds=0, seconds=0.1)
    )
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด ยอดเท่าไร")
    with caplog.at_level("INFO", logger="salid.bot"):
        await env.mgr.handle_segment(s, segment(11))
    assert any(r.getMessage().startswith("answer guild=1000 channel=77 ") for r in caplog.records)
    await env.mgr.end((1000, 77))


async def test_wake_word_without_the_memory_feature_passes_no_memory(env, monkeypatch):
    env.bot.guilds_store.save_settings(1000, GuildSettingsUpdate(features=dict(memory=False)), "x")
    s = await start(env)
    asked = {}
    import salid.qa

    monkeypatch.setattr(
        salid.qa, "ask", lambda *a, **kw: asked.update(kw) or SimpleNamespace(text="ok")
    )
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด สรุปเมื่อวานหน่อย")
    await env.mgr.handle_segment(s, segment(11))
    assert asked["memory"] is None
    await env.mgr.end((1000, 77))


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
    await env.mgr.end((1000, 77))


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
    assert env.chan.sent[-1].endswith(t("meet_wake_empty", "th"))
    await env.mgr.handle_segment(s, segment(10))
    assert questions == ["วันนี้มีนัดอะไรบ้าง"]
    await env.mgr.end((1000, 77))


def test_context_for_the_meeting_text_chat(env):
    s = SimpleNamespace(text_channel=env.chan, transcript=Transcript())
    s.transcript.add(env.bot.s.now(), "แอน", "ส่งงานวันศุกร์")
    env.mgr.sessions[(1000, 77)] = s
    assert env.mgr.context_for(env.chan.id) == "แอน: ส่งงานวันศุกร์"
    assert env.mgr.context_for(123) == ""


# ------------------------------------------------------------------ summary on leave
async def test_leave_posts_a_summary_and_drops_the_transcript(env, monkeypatch):
    s = await start(env)
    s.transcript.add(env.bot.s.now(), "แอน", "ให้บอยส่งรายงานวันศุกร์")
    env.llm.models.script = [text_resp("**สรุปการประชุม**\nบอยส่งรายงานวันศุกร์")]
    await env.mgr.end((1000, 77), "command")
    assert env.vc.disconnected and (1000, 77) not in env.mgr.sessions
    assert "สรุปการประชุม" in env.chan.sent[-1]
    prompt = env.llm.models.calls[0][1][0]
    assert "แอน: ให้บอยส่งรายงานวันศุกร์" in prompt and "งานที่ต้องทำ" in prompt
    assert len(s.transcript) == 0  # nothing persisted


async def test_leave_with_nothing_heard(env):
    await start(env)
    await env.mgr.end((1000, 77), "empty")
    assert env.chan.sent[-1] == t("meet_no_transcript", "th")
    assert env.llm.models.calls == []


async def test_long_summary_is_split_for_discord(env):
    s = await start(env)
    s.transcript.add(env.bot.s.now(), "แอน", "x")
    env.llm.models.script = [text_resp("ก" * 4500)]
    await env.mgr.end((1000, 77), "command")
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
    env.bot.s.ack_llm = False  # no model-written acknowledgement racing the other clips
    calls = []
    monkeypatch.setattr(
        mt, "spoken_text", lambda c, m, q, a, *_: calls.append(("spoken", m, q, a)) or "พูดสั้น ๆ"
    )
    monkeypatch.setattr(
        mt, "say", lambda st, c, t, lang="th": calls.append(("tts", lang, t)) or PCM
    )
    monkeypatch.setattr(mt, "pcm_source", lambda pcm, pitch=1.0: ("source", len(pcm)))
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
    assert voice.calls[1] == ("tts", "th", "พูดสั้น ๆ")
    voice.vc.playing = False
    voice.vc.after(None)  # the player thread says it is done
    await task
    assert s.speaking is False and s.speaking_until > 0
    await voice.mgr.end((1000, 77))


async def test_speak_skips_when_not_connected_or_already_playing(voice, monkeypatch):
    s = await start(voice)
    voice.vc.connected = False
    await voice.mgr.speak(s, "x")
    assert voice.calls == []
    # Something else still playing (not ours): the reply is dropped, never overlapped.
    voice.vc.connected, voice.vc.playing = True, True
    await voice.mgr.speak(s, "x")
    assert voice.vc.played == []
    voice.vc.playing = False
    await voice.mgr.end((1000, 77))


async def test_speak_skips_when_the_meeting_is_ending(voice):
    s = await start(voice)
    s.ending = True
    await voice.mgr.speak(s, "x")
    assert voice.calls == [] and voice.vc.played == []
    s.ending = False
    await voice.mgr.end((1000, 77))


async def test_speak_swallows_tts_errors(voice, monkeypatch):
    s = await start(voice)

    def boom(*a):
        raise RuntimeError("TTS returned no audio")

    monkeypatch.setattr(mt, "say", boom)
    await voice.mgr.speak(s, "x")  # must not raise
    assert voice.vc.played == [] and s.speaking is False
    await voice.mgr.end((1000, 77))


async def test_speak_times_out_stops_playback_and_clears_speaking(voice, monkeypatch):
    s = await start(voice)
    monkeypatch.setattr(mt, "pcm_duration_s", lambda pcm: 0.05)
    monkeypatch.setattr(mt, "SPEAK_TIMEOUT_PAD_S", 0.0)
    await voice.mgr.speak(s, "x")  # the after-callback never fires
    assert voice.vc.played and voice.vc.playing is False  # stopped
    assert s.speaking is False and s.speaking_until > 0
    await voice.mgr.end((1000, 77))


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
    # First the instant acknowledgment, then the shortened answer.
    assert voice.calls[0][:2] == ("tts", "th") and voice.calls[0][2] in ACK_POOL_TH
    assert voice.vc.played == [("source", len(PCM))]
    voice.vc.playing = False
    voice.vc.after(None)  # acknowledgment finished
    for _ in range(50):
        await asyncio.sleep(0.01)
        if len(voice.vc.played) == 2:
            break
    spoken = [c for c in voice.calls if c[0] == "spoken"]
    assert spoken[0][2:] == ("เมื่อกี้ตกลงอะไรกัน", "ตกลงเลื่อนเป็นวันศุกร์ครับ")
    # the acknowledgment is skipped when the answer is already speakable by the time it is ready
    assert len(voice.vc.played) in (1, 2)
    voice.vc.playing = False
    voice.vc.after(None)
    await voice.mgr.end((1000, 77))


ACK_POOL_TH = [t(f"ack_voice_{n}", "th") for n in range(1, 9)]


async def ack_once(voice, s, question=""):
    task = asyncio.create_task(voice.mgr.acknowledge(s, question))
    for _ in range(100):
        await asyncio.sleep(0.01)
        if s.speaking or task.done():
            break
    voice.vc.playing = False
    voice.vc.after(None)
    await task


class FakeWriter:
    def __init__(self, line=None, delay=0.0, boom=False):
        self.line, self.delay, self.boom, self.questions = line, delay, boom, []

    def enabled(self):
        return True

    async def write(self, **kw):
        self.questions.append(kw["question"])
        await asyncio.sleep(self.delay)
        if self.boom:
            raise RuntimeError("model down")
        return self.line


async def test_acknowledgment_never_repeats_the_last_three_lines(voice):
    s = await start(voice)
    voice.bot.fillers = FakeWriter(None)  # the model says nothing: hand-written pool
    lines, add = [], voice.mgr._ack_recent.add
    voice.mgr._ack_recent.add = lambda key, line: (lines.append(line), add(key, line))
    for _ in range(6):
        await ack_once(voice, s)
    assert len(lines) == 6 and all(x in ACK_POOL_TH for x in lines)
    for i, x in enumerate(lines):
        assert x not in lines[max(0, i - 3) : i]
    assert len(voice.vc.played) == 6
    await voice.mgr.end((1000, 77))


async def test_acknowledgment_uses_the_line_written_for_the_question(voice):
    s = await start(voice)
    voice.bot.fillers = FakeWriter("ขอเช็กยอดเดือนนี้แป๊บนะคะ")
    await ack_once(voice, s, "ยอดเดือนนี้เท่าไร")
    assert voice.bot.fillers.questions == ["ยอดเดือนนี้เท่าไร"]
    assert [c for c in voice.calls if c[0] == "tts"] == [("tts", "th", "ขอเช็กยอดเดือนนี้แป๊บนะคะ")]
    await voice.mgr.end((1000, 77))


@pytest.mark.parametrize("writer", [FakeWriter("ช้าไปค่ะ", delay=1.0), FakeWriter(boom=True)])
async def test_slow_or_failing_line_falls_back_and_plays_one_clip(voice, writer):
    s = await start(voice)
    voice.bot.s.voice_ack_timeout_s = 0.05
    voice.bot.fillers = writer
    await ack_once(voice, s, "q")
    lines = [c[2] for c in voice.calls if c[0] == "tts"]
    assert len(lines) == 1 and lines[0] in ACK_POOL_TH  # the late line is dropped
    assert len(voice.vc.played) == 1
    await voice.mgr.end((1000, 77))


async def test_answer_does_not_speak_when_voice_reply_is_off(voice, monkeypatch):
    voice.bot.s.voice_reply = False
    s = await start(voice)
    fake_ask(monkeypatch)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด สรุปให้หน่อย")
    await voice.mgr.handle_segment(s, segment(11))
    await asyncio.sleep(0.05)
    assert voice.calls == [] and voice.vc.played == []
    assert voice.chan.sent[-1].endswith("ตกลงเลื่อนเป็นวันศุกร์ครับ")
    await voice.mgr.end((1000, 77))


async def test_answer_text_still_posts_when_speech_fails(voice, monkeypatch):
    s = await start(voice)
    fake_ask(monkeypatch)

    def boom(*a):
        raise RuntimeError("tts down")

    monkeypatch.setattr(mt, "say", boom)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด สรุปให้หน่อย")
    await voice.mgr.handle_segment(s, segment(11))
    await asyncio.sleep(0.05)
    assert voice.chan.sent[-1] == "<@11> ตกลงเลื่อนเป็นวันศุกร์ครับ"
    assert voice.vc.played == [] and s.speaking is False
    await voice.mgr.end((1000, 77))


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
    await voice.mgr.end((1000, 77))


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
    echo = mt.Segment(11, SPEECH_PCM, now - 15, 2.0)
    await voice.mgr.handle_segment(s, echo)
    assert asked == []
    later = mt.Segment(11, SPEECH_PCM, now - 4, 2.0)  # after the cooldown: a real question
    await voice.mgr.handle_segment(s, later)
    assert asked == ["สรุปให้หน่อย"]
    await voice.mgr.end((1000, 77))


async def test_end_stops_playback_before_disconnecting(voice):
    await start(voice)
    voice.vc.playing = True
    await voice.mgr.end((1000, 77))
    assert voice.vc.playing is False and voice.vc.disconnected


async def test_speak_skips_when_voice_reply_is_disabled(voice):
    s = await start(voice)
    voice.bot.s.voice_reply = False
    await voice.mgr.speak(s, "x")
    assert voice.calls == [] and voice.vc.played == []
    await voice.mgr.end((1000, 77))


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
    await voice.mgr.end((1000, 77))


async def test_speak_cleans_ffmpeg_source_when_play_rejects_it(voice, monkeypatch):
    s = await start(voice)

    class Source:
        cleaned = False

        def cleanup(self):
            self.cleaned = True

    source = Source()
    monkeypatch.setattr(mt, "pcm_source", lambda pcm, pitch=1.0: source)

    def reject(source, after=None):
        raise RuntimeError("player refused source")

    monkeypatch.setattr(voice.vc, "play", reject)
    await voice.mgr.speak(s, "x")
    assert source.cleaned is True
    assert s.speaking is False
    await voice.mgr.end((1000, 77))


# ------------------------------------------------------------------ language
async def test_english_replies_for_the_invoker(env):
    m = make_member(env.guild, 10, "Ann")
    assert await env.mgr.join(m, lang="en") == t("meet_not_in_voice", "en")
    assert "voice channel" in t("meet_not_in_voice", "en")
    assert await env.mgr.leave(m, lang="en") == "Salid is not in a meeting right now."


async def test_meeting_follows_the_text_channels_language(env, monkeypatch):
    env.bot.guilds_store.set_channel_language(1000, env.chan.id, "en")
    m = make_member(env.guild, 10, "Ann", env.chan)
    assert await env.mgr.join(m, lang="th") == t("meet_joined", "th", channel=env.chan.name)
    s = env.mgr.sessions[(1000, 77)]
    assert s.language == "en" and s.lang == "en"
    assert env.chan.sent == [t("meet_announce", "en")]  # announce uses the channel language
    asked = {}
    import salid.qa

    monkeypatch.setattr(
        salid.qa, "ask", lambda *a, **kw: asked.update(kw) or SimpleNamespace(text="Done")
    )
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m_, pcm: "Salid, what did we decide?")
    await env.mgr.handle_segment(s, segment(10))
    assert asked["language"] == "en" and env.chan.sent[-1] == "<@10> Done"
    s.transcript.add(env.bot.s.now(), "Ann", "Boy sends the report on Friday")
    env.llm.models.script = [text_resp("**Meeting summary**\nBoy sends the report")]
    await env.mgr.end((1000, 77), "command")
    prompt = env.llm.models.calls[0][1][0]
    assert "in English" in prompt and "Action items" in prompt
    assert env.chan.sent[-1].startswith("📝 Meeting summary (someone told Salid to leave)")


async def test_auto_meeting_language_uses_the_starters_language_for_fixed_text(env):
    env.bot.guilds_store.set_channel_language(1000, env.chan.id, "auto")
    m = make_member(env.guild, 10, "Ann", env.chan)
    await env.mgr.join(m, lang="en")
    s = env.mgr.sessions[(1000, 77)]
    assert s.language == "auto" and s.lang == "en"
    assert env.chan.sent == [t("meet_announce", "en")]
    await env.mgr.end((1000, 77), "empty")
    assert env.chan.sent[-1] == t("meet_no_transcript", "en")


def test_summary_language_auto_follows_the_transcript(env):
    mt_ = mt
    client = FakeClient([text_resp("x"), text_resp("y"), text_resp("z")])
    mt_.summarize_transcript(client, "m", "[10:00] Ann: ship it on Friday", "auto")
    mt_.summarize_transcript(client, "m", "[10:00] แอน: ส่งวันศุกร์", "auto")
    mt_.summarize_transcript(client, "m", "[10:00] แอน: ส่งวันศุกร์")
    en, th, default = (c[1][0] for c in client.models.calls)
    assert "in English" in en and "เป็นภาษาไทย" in th and "เป็นภาษาไทย" in default


async def test_transcribed_segments_are_logged_as_voice_memory(env, monkeypatch):
    s = await start(env)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "เรามาเริ่มกันเลย")
    await env.mgr.handle_segment(s, segment(10))
    await env.bot.drain_memory()
    day = env.bot.s.now().date().isoformat()
    (rec,) = env.bot.memory_store.messages(1000, {env.chan.id}, day, day)
    assert (rec.kind, rec.author_name, rec.text) == ("voice", "แอน", "เรามาเริ่มกันเลย")
    assert s.transcript.lines  # the in-memory transcript still works
    await env.mgr.end((1000, 77))


def test_classic_answer_passes_memory_and_web_to_ask(monkeypatch):
    """Merged path: the wake-word answer uses our `ask` (memory tools + web client) and is both
    posted as text and handed to the spoken-reply pipeline."""
    import asyncio
    from types import SimpleNamespace

    from salid import meeting as m

    seen = {}

    def fake_ask(*a, **kw):
        seen.update(kw)
        return SimpleNamespace(text="answer")

    monkeypatch.setattr("salid.qa.ask", fake_ask)
    posted, spoken = [], []

    async def go():
        async def memory_tools(guild, voice_channel, humans, *also, asker=None):
            seen["asker"] = asker.id
            seen["humans"] = [h.id for h in humans]
            return "MEM"

        bot = SimpleNamespace(
            s=SimpleNamespace(voice_reply=True, now=lambda: None, meeting_text_echo=True),
            team=SimpleNamespace(enabled=lambda g, f: True),
            store=None,
            is_exec_member=lambda mem: True,
            _room_memory_tools=memory_tools,
            _log_answer=lambda *a: None,
            _web_kw=lambda gid: {"web_client": "W"},
        )
        mgr = m.MeetingManager(bot)
        s = SimpleNamespace(
            guild_id=1,
            transcript=SimpleNamespace(history=lambda now: ""),
            answer_lock=asyncio.Lock(),
            text_channel=None,
            fallback_channel=None,
            voice_channel=SimpleNamespace(id=77, members=[SimpleNamespace(id=6, bot=False)]),
            language="th",
            ending=False,
            vc=SimpleNamespace(is_connected=lambda: True),
            tasks=[],
        )

        async def post(sess, text):
            posted.append(text)

        async def speak(sess, text, question=""):
            spoken.append(text)

        mgr._post, mgr.speak = post, speak
        member = SimpleNamespace(id=5, guild=object(), mention="@u")
        await mgr.answer(s, member, "u", "q?")
        await asyncio.gather(*s.tasks)

    asyncio.run(go())
    assert seen["memory"] == "MEM" and seen["web_client"] == "W"
    assert seen["spoken"] is True and sorted(seen["humans"]) == [5, 6]  # the whole room
    assert posted == ["@u answer"] and spoken == ["answer"]


# ------------------------------------------------------------------ persistence / rejoin / reconnect
def _bot_guild(env, monkeypatch, text_ch=None):
    env.guild.get_channel = lambda cid: env.chan if cid == env.chan.id else text_ch
    monkeypatch.setattr(SecretaryBot, "get_guild", lambda self, gid: env.guild)


async def test_active_meeting_is_persisted_and_cleared_on_a_normal_end(env):
    store = env.bot.secretary_store
    await start(env)
    rec = store.get_active_meeting(1000, 77)
    assert rec["voice_channel_id"] == env.chan.id and rec["starter"] == "แอน"
    assert rec["language"] and rec["started_at"] and "meeting_id" in rec
    await env.mgr.end((1000, 77), "command")
    assert store.get_active_meeting(1000, 77) is None


async def test_shutdown_keeps_the_record_and_posts_no_summary(env):
    s = await start(env)
    sent = len(env.chan.sent)
    await env.mgr.shutdown()
    assert env.bot.secretary_store.get_active_meeting(1000, 77) is not None
    assert env.vc.disconnected and (1000, 77) not in env.mgr.sessions and s.ending
    assert len(env.chan.sent) == sent


async def test_rejoin_on_ready_continues_the_meeting_with_logged_lines(env, monkeypatch):
    s = await start(env)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "ส่งงานวันศุกร์")
    await env.mgr.handle_segment(s, segment(10))
    await env.bot.drain_memory()
    await env.mgr.shutdown()  # the restart
    env.chan.members = [make_member(env.guild, 10, "แอน", env.chan)]
    env.chan.sent.clear()
    env.vc.disconnected = False
    _bot_guild(env, monkeypatch)
    await env.mgr.resume_all()
    new = env.mgr.sessions[(1000, 77)]
    assert new is not s and isinstance(env.vc.listening, mt.MeetingSink)
    assert [ln.text for ln in new.transcript.lines] == ["ส่งงานวันศุกร์"]
    assert env.chan.sent == ["สลิดกลับเข้ามาจดประชุมต่อแล้วค่ะ (หลุดไปครู่หนึ่ง)"]
    assert env.bot.secretary_store.get_active_meeting(1000, 77) is not None
    await env.mgr.end((1000, 77), "command")


async def test_rejoin_with_an_empty_channel_ends_with_a_summary(env, monkeypatch):
    s = await start(env)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "ตกลงตามนี้")
    await env.mgr.handle_segment(s, segment(10))
    await env.bot.drain_memory()
    await env.mgr.shutdown()
    env.chan.members = []  # everyone left while the bot was down
    env.chan.sent.clear()
    _bot_guild(env, monkeypatch)
    monkeypatch.setattr(mt, "summarize_transcript", lambda c, m, text, lang: "สรุป: " + text)
    await env.mgr.resume_all()
    assert (1000, 77) not in env.mgr.sessions
    assert env.bot.secretary_store.get_active_meeting(1000, 77) is None
    assert any("ตกลงตามนี้" in m and "สรุป" in m for m in env.chan.sent)


async def test_unexpected_disconnect_retries_then_continues(env, monkeypatch):
    monkeypatch.setattr(mt, "RECONNECT_DELAYS", (0, 0, 0))
    s = await start(env)
    env.chan.members = [make_member(env.guild, 10, "แอน", env.chan)]
    calls = []
    new_vc = FakeVC()

    async def flaky(body, channel):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("voice down")
        return new_vc

    monkeypatch.setattr(env.mgr, "_connect", flaky)
    await env.mgr.on_disconnected(1000)
    assert len(calls) == 3 and s.vc is new_vc and isinstance(new_vc.listening, mt.MeetingSink)
    assert t("meet_reconnected", "th") in env.chan.sent and (1000, 77) in env.mgr.sessions
    await env.mgr.end((1000, 77))


async def test_disconnect_gives_up_after_three_tries_and_announces(env, monkeypatch):
    monkeypatch.setattr(mt, "RECONNECT_DELAYS", (0, 0, 0))
    await start(env)
    env.chan.members = [make_member(env.guild, 10, "แอน", env.chan)]
    calls = []

    async def dead(body, channel):
        calls.append(1)
        raise RuntimeError("down")

    monkeypatch.setattr(env.mgr, "_connect", dead)
    await env.mgr.on_disconnected(1000)
    assert len(calls) == 3 and (1000, 77) not in env.mgr.sessions
    assert t("meet_reconnect_failed", "th") in env.chan.sent
    assert env.bot.secretary_store.get_active_meeting(1000, 77) is None


async def test_disconnect_with_nobody_left_just_ends(env):
    await start(env)
    env.chan.members = []
    await env.mgr.on_disconnected(1000)
    assert (1000, 77) not in env.mgr.sessions


async def _tick_once(env, s, monkeypatch):
    monkeypatch.setattr(mt, "TICK_S", 0)
    ends = []

    async def fake_end(gid, reason="command"):
        ends.append(reason)

    monkeypatch.setattr(env.mgr, "end", fake_end)
    task = asyncio.create_task(env.mgr._ticker(s))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return ends


async def test_meeting_leaves_thirty_seconds_after_humans_leave(env, monkeypatch):
    s = await start(env)
    assert s.empty_grace == 30.0
    s.had_humans, s.empty_since = True, time.monotonic() - 20  # a brief disconnect
    env.chan.members = []
    assert await _tick_once(env, s, monkeypatch) == []
    s.empty_since = time.monotonic() - 31
    assert await _tick_once(env, s, monkeypatch) == ["empty"]


async def test_auto_join_nobody_came_uses_the_short_grace(env, monkeypatch):
    a = make_member(env.guild, 10, "แอน", env.chan)
    await env.mgr.join(a, first_grace=120.0)
    s = env.mgr.sessions[(1000, 77)]
    env.chan.members = []
    s.empty_since = time.monotonic() - 121
    assert await _tick_once(env, s, monkeypatch) == ["empty"]
    # once people were there, the normal grace applies even for an auto-join
    s.had_humans, s.empty_since = True, time.monotonic() - 10
    assert await _tick_once(env, s, monkeypatch) == []


async def test_max_time_warns_ten_minutes_before_the_cap(env, monkeypatch):
    s = await start(env)
    assert mt.MEETING_MAX_S == 6 * 3600
    env.chan.members = [make_member(env.guild, 10, "แอน", env.chan)]
    s.started = time.monotonic() - (mt.MEETING_MAX_S - 300)
    assert await _tick_once(env, s, monkeypatch) == []
    assert env.chan.sent.count(t("meet_max_warning", "th")) == 1
    s.started = time.monotonic() - mt.MEETING_MAX_S - 1
    assert await _tick_once(env, s, monkeypatch) == ["timeout"]


async def test_spoken_leave_ends_the_meeting(env, monkeypatch):
    s = await start(env)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด ออกประชุม")
    await env.mgr.handle_segment(s, segment(10))
    await asyncio.sleep(0.05)
    assert (1000, 77) not in env.mgr.sessions


# ------------------------------------------------------------------ minutes document at the end
class FileChannel(FakeVoiceChannel):
    def __init__(self):
        super().__init__()
        self.files = []

    async def send(self, text, files=None, **_):
        self.sent.append(text)
        self.files.append(list(files or []))


async def _doc_env(env):
    env.chan = FileChannel()
    env.chan.guild = env.guild
    return await start(env)


async def test_end_attaches_the_minutes_pdf_to_the_first_summary_message(env):
    s = await _doc_env(env)
    now = env.bot.s.now()
    s.transcript.add(now, "แอน", "ให้บอยส่งรายงานวันศุกร์")
    s.transcript.add(now, mt.md.SALID_SPEAKER, "รับทราบค่ะ")
    s.attendee_ids |= {10, 11}
    env.llm.models.script = [text_resp("**สรุปการประชุม**\nบอยส่งรายงานวันศุกร์")]
    await env.mgr.end((1000, 77), "command")
    assert "สรุปการประชุม" in env.chan.sent[-1]  # the summary text is unchanged
    doc = env.chan.files[-1]
    assert len(doc) == 1 and doc[0].filename.startswith("minutes-")
    assert doc[0].filename.endswith((".pdf", ".md"))
    doc[0].fp.seek(0)
    data = doc[0].fp.read()
    if doc[0].filename.endswith(".pdf"):
        assert data.startswith(b"%PDF")
    else:
        text = data.decode()
        assert "แอน" in text and "สลิด" in text and "บทพูดของแต่ละคน" in text


async def test_no_minutes_document_when_the_flag_is_off(env):
    env.bot.guilds_store.save_settings(
        1000, GuildSettingsUpdate(features=dict(meeting_transcript_doc=False)), "x"
    )
    await asyncio.to_thread(env.bot.team.current, 1000)
    s = await _doc_env(env)
    s.transcript.add(env.bot.s.now(), "แอน", "ให้บอยส่งรายงานวันศุกร์")
    env.llm.models.script = [text_resp("**สรุปการประชุม**\nx")]
    await env.mgr.end((1000, 77), "command")
    assert env.chan.sent and all(f == [] for f in env.chan.files)


async def test_no_minutes_document_when_not_recording(env):
    s = await _doc_env(env)
    s.transcript.add(env.bot.s.now(), "แอน", "ก่อนหยุด")
    env.mgr.set_recording(s, False)
    assert s.transcript_from is not None and len(s.transcript) == 0
    await env.mgr.end((1000, 77), "command")
    assert all(f == [] for f in env.chan.files)


async def test_lines_before_a_pause_are_not_in_the_document(env):
    s = await _doc_env(env)
    s.transcript.add(env.bot.s.now(), "แอน", "ความลับก่อนหยุด")
    env.mgr.set_recording(s, False)
    env.mgr.set_recording(s, True)
    s.transcript.add(env.bot.s.now(), "บอย", "หลังกลับมา")
    s.attendee_ids.add(10)
    env.llm.models.script = [text_resp("**สรุปการประชุม**\nx")]
    await env.mgr.end((1000, 77), "command")
    doc = env.chan.files[-1][0]
    doc.fp.seek(0)
    data = doc.fp.read()
    if doc.filename.endswith(".md"):
        assert (
            b"\xe0\xb8\x84\xe0\xb8\xa7\xe0\xb8\xb2\xe0\xb8\xa1\xe0\xb8\xa5\xe0\xb8\xb1\xe0\xb8\x9a"
            not in data
        )
    assert "ความลับก่อนหยุด" not in env.llm.models.calls[0][1][0]


async def test_salid_spoken_line_is_logged_to_the_transcript(env):
    s = await _doc_env(env)
    await env.mgr._log_salid_line(s, "รับทราบค่ะ")
    assert [(ln.speaker, ln.text) for ln in s.transcript.lines] == [
        (mt.md.SALID_SPEAKER, "รับทราบค่ะ")
    ]
    env.mgr.set_recording(s, False)
    await env.mgr._log_salid_line(s, "ไม่ควรถูกเก็บ")
    assert len(s.transcript) == 0
