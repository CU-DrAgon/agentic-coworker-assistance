"""Voice meetings, fully offline: segmenter, wake word, transcript window, command guards,
wake-word answers and the summary on leave. No Discord connection, no Vertex."""

import asyncio
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
    bot = SecretaryBot(FixedClockSettings(exec_user_ids={5}), None, llm, guilds=MemoryGuildStore())
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
    assert await env.mgr.join(m) == t("meet_not_in_voice", "th")
    assert env.mgr.sessions == {}


async def test_join_needs_connect_permission(env):
    env.chan._connect = False
    m = make_member(env.guild, 10, "แอน", env.chan)
    assert await env.mgr.join(m) == t("meet_no_permission", "th")


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
    assert "สลิดเข้ามาฟังและจดบันทึกการประชุมครับ" in env.chan.sent[0]
    assert isinstance(env.vc.listening, mt.MeetingSink)
    again = await env.mgr.join(m)
    assert "กำลังอยู่ในการประชุม" in again
    await env.mgr.end(1000, "command")


async def test_leave_without_meeting(env):
    m = make_member(env.guild, 10, "แอน", env.chan)
    assert await env.mgr.leave(m) == t("meet_none", "th")


async def test_leave_denied_for_outsiders(env):
    a = make_member(env.guild, 10, "แอน", env.chan)
    outsider = make_member(env.guild, 11, "ซี")
    await env.mgr.join(a)
    assert await env.mgr.leave(outsider) == t("meet_leave_denied", "th")
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
    assert env.chan.sent == [t("meet_announce", "th")]  # no wake word, no reply
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
    assert env.chan.sent[-1].endswith(t("meet_wake_empty", "th"))
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
    assert env.chan.sent[-1] == t("meet_no_transcript", "th")
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
    s = env.mgr.sessions[1000]
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
    await env.mgr.end(1000, "command")
    prompt = env.llm.models.calls[0][1][0]
    assert "in English" in prompt and "Action items" in prompt
    assert env.chan.sent[-1].startswith("📝 Meeting summary (someone told Salid to leave)")


async def test_auto_meeting_language_uses_the_starters_language_for_fixed_text(env):
    env.bot.guilds_store.set_channel_language(1000, env.chan.id, "auto")
    m = make_member(env.guild, 10, "Ann", env.chan)
    await env.mgr.join(m, lang="en")
    s = env.mgr.sessions[1000]
    assert s.language == "auto" and s.lang == "en"
    assert env.chan.sent == [t("meet_announce", "en")]
    await env.mgr.end(1000, "empty")
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
    await env.mgr.end(1000)
