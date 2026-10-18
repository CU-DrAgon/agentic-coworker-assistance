"""Live voice mode, fully offline: audio math, the playback source, and a fake Gemini Live session
driving the real LiveVoice through MeetingManager. No Discord, no Vertex."""

import asyncio
import contextlib
import logging
import time
from array import array
from types import SimpleNamespace

import pytest
from conftest import ALICE, FakeClient, FixedClockSettings, ThaiGuildStore, mk
from google.genai import types
from test_meeting import SPEECH_PCM, FakeGuild, FakeVC, FakeVoiceChannel, make_member

from salid import live_voice as lvm
from salid import meeting as mt
from salid.bot import SecretaryBot
from salid.guild_settings import GuildSettingsUpdate
from salid.i18n import t
from salid.live_voice import (
    FRAME_BYTES,
    SILENCE,
    LiveAudioSource,
    LiveVoice,
    Mixer,
    build_instruction,
    to_live_input,
    upsample_output,
)
from salid.profiles import Profile
from salid.store import SQLiteStore

ME = SimpleNamespace(id=999, mention="<@999>")


def pcm48(left, right=None, frames=960):
    """20 ms of constant 48 kHz stereo s16le."""
    right = left if right is None else right
    return array("h", [left, right] * frames).tobytes()


def samples(b):
    a = array("h")
    a.frombytes(b)
    return list(a)


# ------------------------------------------------------------------ audio math
def test_downmix_and_decimate_length_and_average():
    out = to_live_input(pcm48(3000, 1000))  # 20 ms
    assert len(out) == 320 and set(out) == {2000}  # 16 kHz mono, L/R averaged
    assert len(to_live_input(bytes(3840 * 5))) == 1600  # 100 ms -> 1600 samples
    assert len(to_live_input(b"\x00" * 7)) == 0  # less than one group of 3 frames


def test_mixer_sums_speakers_and_clips():
    m = Mixer()
    m.feed(1, pcm48(1000))
    m.feed(2, pcm48(2500))
    assert set(samples(m.pop())) == {3500}
    m.feed(1, pcm48(30000))
    m.feed(2, pcm48(30000))
    m.feed(3, pcm48(-30000))
    m.feed(4, pcm48(-30000))
    m.feed(5, pcm48(-30000))
    m.feed(6, pcm48(-30000))
    assert set(samples(m.pop())) == {-32768}  # 60000 - 120000 clipped at the low end
    m.feed(1, pcm48(30000))
    m.feed(2, pcm48(30000))
    assert set(samples(m.pop())) == {32767}
    assert m.pop() == b""


def test_mixer_drains_in_bounded_chunks_and_keeps_speakers_contiguous():
    m = Mixer()
    for _ in range(10):
        m.feed(1, pcm48(100))  # 200 ms
    m.feed(2, pcm48(50))  # 20 ms
    first = samples(m.pop(1600))
    assert len(first) == 1600 and first[0] == 150 and first[-1] == 100
    assert len(samples(m.pop(1600))) == 1600  # the rest of speaker 1
    assert m.pop() == b""


def test_mixer_bounds_a_speakers_backlog():
    m = Mixer(max_s=0.1)
    for _ in range(20):
        m.feed(1, pcm48(7))
    assert len(samples(m.pop(10_000))) == 1600


def test_upsample_output_repeats_samples_into_stereo():
    out = samples(upsample_output(array("h", [1, -2]).tobytes()))
    assert out == [1, 1, 1, 1, -2, -2, -2, -2]  # 24k mono -> 48k stereo
    assert len(upsample_output(bytes(1920))) == FRAME_BYTES * 2  # 40 ms of 24 kHz = 2 frames


# ------------------------------------------------------------------ playback source
def test_source_returns_exact_frames_silence_while_generating_then_ends():
    src = LiveAudioSource()
    assert src.read() == SILENCE and not src.is_opus()  # nothing yet, still generating
    src.feed(array("h", range(1, 961)).tobytes())  # 40 ms of model audio = 2 frames
    a, b = src.read(), src.read()
    assert len(a) == len(b) == FRAME_BYTES and a != SILENCE and b != SILENCE
    assert src.read() == SILENCE  # buffer ran dry but the turn is open
    src.feed(bytes(1920))
    src.finish()
    assert len(src.read()) == FRAME_BYTES and len(src.read()) == FRAME_BYTES
    assert src.read() == b"" and src.ended  # complete and drained: playback ends
    assert src.feed(bytes(100)) is False  # a spent source takes no more audio


def test_source_pads_the_last_partial_frame_and_keeps_odd_bytes():
    src = LiveAudioSource()
    src.feed(b"\x01\x00\x02")  # one sample + a dangling byte
    src.feed(b"\x00")  # completes the second sample
    assert src.buffered_bytes == 16
    assert src.read() == SILENCE  # a partial frame is not played while the turn is open
    src.finish()
    last = src.read()
    assert (
        len(last) == FRAME_BYTES and last[:16] != bytes(16) and last[16:] == bytes(FRAME_BYTES - 16)
    )
    assert src.read() == b""


def test_source_clear_drops_queued_audio_and_reopening_continues():
    src = LiveAudioSource()
    src.feed(bytes(1920))
    src.finish()
    src.clear()
    assert src.buffered_bytes == 0
    src.feed(bytes(1920))  # feeding again re-opens a finished turn
    assert len(src.read()) == FRAME_BYTES and src.read() != b""


def test_source_ends_when_the_model_never_finishes():
    src = LiveAudioSource(stall_s=0.0)
    time.sleep(0.01)
    assert src.read() == b"" and src.ended
    other = LiveAudioSource()
    other.cleanup()
    assert other.read() == b""


# ------------------------------------------------------------------ instruction
def test_instruction_has_date_persona_tools_and_language():
    s = FixedClockSettings()
    th = build_instruction(s, "th")
    assert "2026-10-14" in th and "วันพุธ" in th and "สัปดาห์เริ่มวันจันทร์" in th
    assert "สลิด" in th and "Salid" in th and "เงียบ" in th and "ask_salid" in th
    assert "กำลังฟัง" in th and "end_conversation" in th and "พูดภาษาไทย" in th
    assert "always speak English" in build_instruction(s, "en")
    assert "ภาษาเดียวกับคนที่พูด" in build_instruction(s, "auto")
    assert "ดูข้อมูลของทีมไม่ได้" in build_instruction(s, "th", qa=False)
    assert "ask_salid" not in build_instruction(s, "th", qa=False)
    assert "แอน: ส่งงาน" in build_instruction(s, "th", history="แอน: ส่งงาน")


def test_instruction_answers_general_questions_without_tools_and_never_raises_money():
    th = build_instruction(FixedClockSettings(), "th")
    assert "ตอบเองตรง ๆ ห้ามเรียกเครื่องมือ" in th  # general knowledge, opinions, small talk
    assert "ห้ามพูดถึงเงิน ใบเสร็จ หรือค่าใช้จ่ายเองเด็ดขาด" in th
    assert "ห้ามพูดคำเติมเองก่อนหรือระหว่างเรียก ask_salid" in th  # the system speaks the filler
    assert "spending_summary" not in th


# ------------------------------------------------------------------ fake Live API
def audio_msg(n=1920):
    blob = types.Blob(data=array("h", [100] * (n // 2)).tobytes(), mime_type="audio/pcm;rate=24000")
    content = types.Content(role="model", parts=[types.Part(inline_data=blob)])
    return types.LiveServerMessage(server_content=types.LiveServerContent(model_turn=content))


def said_msg(text):
    sc = types.LiveServerContent(output_transcription=types.Transcription(text=text))
    return types.LiveServerMessage(server_content=sc)


def done_msg():
    return types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))


def interrupted_msg():
    return types.LiveServerMessage(server_content=types.LiveServerContent(interrupted=True))


def tool_msg(name, args, id="call-1"):
    call = types.FunctionCall(id=id, name=name, args=args)
    return types.LiveServerMessage(tool_call=types.LiveServerToolCall(function_calls=[call]))


def resume_msg(handle):
    upd = types.LiveServerSessionResumptionUpdate(new_handle=handle, resumable=True)
    return types.LiveServerMessage(session_resumption_update=upd)


def go_away_msg():
    return types.LiveServerMessage(go_away=types.LiveServerGoAway(time_left="30s"))


class FakeLive:
    """The part of a Live session LiveVoice uses. receive() ends after a turn_complete, like the
    real one, and raises an exception pushed into it (connection lost)."""

    def __init__(self):
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.audio: list[bytes] = []
        self.stream_ends = 0
        self.tool_responses: list = []
        self.closed = False
        self.notes: list[tuple[str, bool]] = []

    async def send_client_content(self, turns=None, turn_complete=True):
        self.notes.append((turns.parts[0].text, turn_complete))

    def push(self, *msgs):
        for m in msgs:
            self.inbox.put_nowait(m)

    async def send_realtime_input(self, audio=None, audio_stream_end=None):
        if audio is not None:
            assert audio.mime_type == "audio/pcm;rate=16000"
            self.audio.append(audio.data)
        if audio_stream_end:
            self.stream_ends += 1

    async def send_tool_response(self, function_responses):
        self.tool_responses.append(function_responses)

    async def receive(self):
        while True:
            m = await self.inbox.get()
            if isinstance(m, BaseException):
                raise m
            yield m
            if m.server_content is not None and m.server_content.turn_complete:
                return


class FakeConnect:
    def __init__(self, fail_first=0):
        self.configs: list[types.LiveConnectConfig] = []
        self.sessions: list[FakeLive] = []
        self.fail_first = fail_first

    @contextlib.asynccontextmanager
    async def __call__(self, config):
        self.configs.append(config)
        if len(self.configs) <= self.fail_first:
            raise ConnectionError("connect failed")
        live = FakeLive()
        self.sessions.append(live)
        try:
            yield live
        finally:
            live.closed = True


async def until(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached in time")


@pytest.fixture
def lv(monkeypatch, tmp_path):
    guild = FakeGuild()
    chan = FakeVoiceChannel()
    chan.guild = guild
    store = SQLiteStore(str(tmp_path / "t.db"))
    mk(store, 1, ALICE, "Alice", "Makro", "2026-10-01", 1000.0, guild=guild.id)
    mk(store, 2, ALICE, "Alice", "Grab", "2026-10-05", 250.5, "ค่าเดินทาง", guild=guild.id)
    bot = SecretaryBot(
        FixedClockSettings(
            exec_user_ids={5},
            voice_reply=True,
            voice_mode="live",
            meeting_text_echo=True,
            voice_state_posts=False,
            # these suites test playback / tool mechanics, not the gate: it is off here on purpose
            # (the gate has its own suite, test_mouth_gate.py, where it is on)
            live_mouth_gate=False,
        ),
        store,
        FakeClient([]),
        guilds=ThaiGuildStore(),
    )
    monkeypatch.setattr(SecretaryBot, "user", property(lambda s: ME))
    monkeypatch.setattr(lvm, "SEND_INTERVAL_S", 0.01)
    monkeypatch.setattr(lvm, "STREAM_IDLE_S", 0.06)
    monkeypatch.setattr(lvm, "BACKOFF_S", (0.0, 0.0, 0.0, 0.0))
    monkeypatch.setattr(lvm, "say", lambda *a, **k: bytes(4800))  # no Cloud TTS offline
    monkeypatch.setattr(lvm, "pitch_pcm", lambda pcm, pitch=1.0: pcm)
    vc = FakeVC()
    connect = FakeConnect()

    async def vc_connect(body, channel):
        return vc

    monkeypatch.setattr(bot.meetings, "_connect", vc_connect)
    monkeypatch.setattr(LiveVoice, "_open_default", lambda self, config: connect(config))
    env = SimpleNamespace(guild=guild, chan=chan, bot=bot, mgr=bot.meetings, vc=vc, connect=connect)

    async def join(*member_ids):
        """Join with the given members in the room (id 5 is an executive)."""
        members = [make_member(guild, uid, f"m{uid}", chan) for uid in member_ids]
        chan.members = members
        await env.mgr.join(members[0])
        s = env.mgr.sessions[(guild.id, chan.id)]
        await until(lambda: connect.sessions and s.live._live is not None)
        return s, connect.sessions[0]

    env.join = join
    return env


def play_source(env):
    return env.vc.played[0]


def drain(src, limit=50):
    """Frames of real audio the source hands out before it goes silent or ends."""
    n = 0
    while n < limit and src.read() not in (SILENCE, b""):
        n += 1
    return n


# ------------------------------------------------------------------ wiring
async def test_join_connects_live_at_once_and_builds_the_session_config(lv):
    s, live = await lv.join(5)
    assert s.voice_mode == "live" and s.live is not None
    cfg = lv.connect.configs[0]
    assert cfg.response_modalities == ["AUDIO"]
    assert cfg.proactivity.proactive_audio is True
    assert cfg.input_audio_transcription is not None and cfg.output_audio_transcription is not None
    assert cfg.speech_config.voice_config.prebuilt_voice_config.voice_name == "Leda"
    assert cfg.session_resumption.handle is None
    assert cfg.context_window_compression.sliding_window is not None
    names = [d.name for d in cfg.tools[0].function_declarations]
    assert names == [
        "ask_salid", "answer_privately", "set_minutes_destination", "post_note", "ask_consent",
        "set_recording", "set_captions", "end_conversation",
    ]  # fmt: skip
    assert "spending_summary" not in names  # no raw ledger tools in Live mode
    assert "2026-10-14" in cfg.system_instruction
    assert live.closed is False
    await lv.mgr.end((1000, 77))
    assert live.closed


async def test_live_not_started_when_voice_reply_off_or_classic_mode(lv):
    lv.bot.s.voice_reply = False
    m = make_member(lv.guild, 5, "boss", lv.chan)
    lv.chan.members = [m]
    await lv.mgr.join(m)
    s = lv.mgr.sessions[(1000, 77)]
    assert s.live is None and s.voice_mode == "classic" and lv.connect.configs == []
    await lv.mgr.end((1000, 77))
    lv.bot.s.voice_reply, lv.bot.s.voice_mode = True, "classic"
    await lv.mgr.join(m)
    assert lv.mgr.sessions[(1000, 77)].live is None
    await lv.mgr.end((1000, 77))


async def test_room_audio_is_mixed_streamed_and_ended_after_a_pause(lv):
    lv.bot.s.live_silence_ms = 0  # idle = STREAM_IDLE_S (patched short)
    s, live = await lv.join(5, 11)
    for _ in range(5):
        s.on_pcm(5, pcm48(1000), time.monotonic())
        s.on_pcm(11, pcm48(500), time.monotonic())
    await until(lambda: live.audio)
    first = samples(b"".join(live.audio))
    assert first[0] == 1500  # both speakers summed into one mono stream
    await until(lambda: live.stream_ends == 1)
    # real silence went out between the voice and the stream end, so the server VAD can cut the turn
    assert any(set(samples(c)) == {0} for c in live.audio)
    n = len(live.audio)
    await asyncio.sleep(0.1)
    assert live.stream_ends == 1 and len(live.audio) == n  # idle: nothing more is sent
    s.on_pcm(5, pcm48(900), time.monotonic())  # speaking again resumes the stream
    await until(lambda: len(live.audio) > n)
    await lv.mgr.end((1000, 77))


async def test_audio_heard_while_connecting_is_kept_so_a_first_greeting_is_not_lost():
    live = LiveVoice(SimpleNamespace(), SimpleNamespace(), post=lambda t: None)
    live.feed(1, pcm48(1000))
    assert live.mixer.pop() != b""


# ------------------------------------------------------------------ output
async def test_model_audio_plays_once_posts_the_transcript_and_updates_speaking(lv, caplog):
    caplog.set_level(logging.INFO, logger="salid.live_voice")
    s, live = await lv.join(5)
    s.on_pcm(5, pcm48(1000), time.monotonic())
    await until(lambda: live.audio)
    live.push(audio_msg(), audio_msg(), said_msg("สวัสดี"), said_msg("นะ"))
    await until(lambda: lv.vc.played)
    await asyncio.sleep(0.03)
    assert len(lv.vc.played) == 1 and s.speaking is True and s.speaking_since > 0
    live.push(done_msg())
    await until(lambda: "🔊 สวัสดีนะ" in lv.chan.sent)
    src = play_source(lv)
    assert drain(src) == 4 and src.read() == b"" and src.ended  # 2 x 40 ms, then playback ends
    lv.vc.playing = False
    lv.vc.after(None)  # the player thread reports it is done
    await until(lambda: s.speaking is False)
    assert s.speaking_until > time.monotonic()
    assert "live: user audio ended -> first audio" in caplog.text
    await lv.mgr.end((1000, 77))


async def test_a_second_turn_while_the_first_drains_continues_the_same_playback(lv):
    s, live = await lv.join(5)
    live.push(audio_msg(), done_msg())
    await until(lambda: lv.vc.played and play_source(lv)._complete)
    live.push(audio_msg())
    await until(lambda: not play_source(lv)._complete)
    assert len(lv.vc.played) == 1  # one stream, no second vc.play
    await lv.mgr.end((1000, 77))


async def test_turn_complete_right_after_a_tool_call_does_not_end_playback(lv):
    s, live = await lv.join(5)
    live.push(audio_msg())  # "let me check" filler
    await until(lambda: lv.vc.played)
    live.push(tool_msg("spending_summary", {"start_date": "2026-10-01", "end_date": "2026-10-31"}))
    live.push(done_msg())  # spurious: the answer is still to come
    await until(lambda: live.tool_responses)
    src = play_source(lv)
    assert drain(src) == 2
    assert src.read() == SILENCE and not src.ended  # still open: waiting for the answer
    live.push(audio_msg(), done_msg())  # the real answer, then the real end
    await until(lambda: src._complete)
    assert drain(src) == 2 and src.read() == b""
    assert len(lv.vc.played) == 1
    await lv.mgr.end((1000, 77))


async def test_interruption_stops_playback_and_drops_queued_audio(lv):
    s, live = await lv.join(5)
    live.push(audio_msg(), said_msg("ยังพูดไม่จบ"))
    await until(lambda: lv.vc.played and s.speaking)
    src = play_source(lv)
    live.push(interrupted_msg())
    await until(lambda: lv.vc.playing is False)
    assert src.buffered_bytes == 0 and s.speaking is False
    live.push(done_msg())
    await asyncio.sleep(0.03)
    assert all("ยังพูดไม่จบ" not in m for m in lv.chan.sent)  # the cut-off turn is not logged
    live.push(audio_msg())  # the next answer plays on a fresh source
    await until(lambda: len(lv.vc.played) == 2)
    assert lv.vc.played[1] is not src
    await lv.mgr.end((1000, 77))


async def test_nothing_plays_when_voice_reply_is_switched_off_or_meeting_ends(lv):
    s, live = await lv.join(5)
    lv.bot.s.voice_reply = False
    live.push(audio_msg())
    await asyncio.sleep(0.05)
    assert lv.vc.played == []
    lv.bot.s.voice_reply = True
    s.ending = True
    live.push(audio_msg())
    await asyncio.sleep(0.05)
    assert lv.vc.played == []
    s.ending = False
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ tools
@pytest.fixture
def asked(monkeypatch, lv):
    """Records what ask_salid hands to the text agent; the room memory scope is stubbed."""
    seen = {}

    def fake_ask(*a, **kw):
        seen.update(kw, question=a[2])
        return SimpleNamespace(text="x" * 900, seconds=1.5, rounds=2)

    async def room_memory(guild, voice_channel, humans, *also, asker=None):
        seen["room"] = sorted(h.id for h in humans)
        seen["mem_asker"], seen["mem_also"] = getattr(asker, "id", None), also
        return "MEM"

    monkeypatch.setattr("salid.qa.ask", fake_ask)
    monkeypatch.setattr(lv.bot, "_room_memory_tools", room_memory)
    monkeypatch.setattr(lv.bot, "_web_kw", lambda gid: {"web_client": "W"})

    def room_secretary(guild, asker, is_exec, lang, text_channel_id):
        seen["sec_args"] = (asker.id, is_exec, text_channel_id)
        return "SEC"

    def coo_tools(guild, asker, is_exec, memory, question, lang):
        seen["coo_args"] = (asker.id, is_exec, memory)
        return ["COO"]

    monkeypatch.setattr(lv.bot, "_room_secretary", room_secretary)
    monkeypatch.setattr(lv.bot, "_coo_tools", coo_tools)
    monkeypatch.setattr(lv.bot, "_freetime_tools", lambda *a: [])  # covered in test_freetime.py
    monkeypatch.setattr(lv.bot, "_wiki_tools", lambda *a, **k: [])  # covered by test_wiki.py
    monkeypatch.setattr(lv.bot, "_approval_tools", lambda *a: [])  # covered in test_approvals
    monkeypatch.setattr(lv.bot, "_ranks", lambda g: {5: 9, 11: 1, 12: 4})
    monkeypatch.setattr(lv.bot, "_profile_of", lambda gid, uid: Profile(nickname=f"p{uid}"))
    return seen


async def test_ask_salid_runs_the_text_agent_scoped_to_the_room(lv, asked):
    s, live = await lv.join(5)
    s.transcript.add(lv.bot.s.now(), "แอน", "คุยเรื่องแผนไตรมาสสี่")
    live.push(tool_msg("ask_salid", {"question": "ใครรับงานแผนไตรมาสสี่"}))
    await until(lambda: live.tool_responses)
    (resp,) = live.tool_responses[0]
    assert resp.name == "ask_salid" and len(resp.response["answer"]) == lvm.ANSWER_MAX_CHARS
    assert asked["room"] == [5] and asked["memory"] == "MEM" and asked["web_client"] == "W"
    assert asked["is_exec"] is True and asked["spoken"] is True
    assert "คุยเรื่องแผนไตรมาสสี่" in asked["history"]  # this meeting's transcript is visible
    assert asked["language"] == s.language
    await lv.mgr.end((1000, 77))


async def test_ask_salid_memory_is_bound_to_the_text_echo_audience(lv, asked):
    """The spoken answer is also typed into the meeting chat / fallback channel (text echo), so
    memory must be limited to what the effective asker and everyone reading those can see."""
    s, live = await lv.join(5, 11, 12)
    s.fallback_channel = SimpleNamespace(id=4242, guild=lv.guild)
    live.push(tool_msg("ask_salid", {"question": "งบโฆษณา"}))
    await until(lambda: live.tool_responses)
    assert asked["mem_asker"] == 11  # the lowest-ranked human present (ranks 5:9, 11:1, 12:4)
    assert asked["mem_also"] == (lv.bot._answering_id(s.fallback_channel),)
    await lv.mgr.end((1000, 77))


async def test_ask_salid_gets_the_secretary_and_coo_tools_for_the_lone_human(lv, asked):
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": "ประชุมพรุ่งนี้มีอะไรบ้าง"}))
    await until(lambda: live.tool_responses)
    assert asked["secretary"] == "SEC" and asked["extra_tools"][:1] == ["COO"]
    names = {getattr(f, "__name__", f) for f in asked["extra_tools"]}
    assert "set_captions" in names
    assert {"create_poll", "poll_results", "start_form", "form_results"} <= names
    assert {"post_message", "send_dm_many"} <= names
    assert asked["sec_args"] == (5, True, s.text_channel.id)
    assert asked["coo_args"] == (5, True, "MEM")
    assert "p5" in asked["profile"]  # one person: their own profile
    await lv.mgr.end((1000, 77))


async def test_ask_salid_in_a_room_acts_as_the_lowest_ranked_human_without_profile(lv, asked):
    s, live = await lv.join(5, 12, 11)  # ranks 9, 4, 1
    live.push(tool_msg("ask_salid", {"question": "เตือนงานให้หน่อย"}))
    await until(lambda: live.tool_responses)
    assert asked["sec_args"][0] == 11 and asked["coo_args"][0] == 11
    assert asked["sec_args"][1] is False  # not everyone is an executive
    assert "profile" not in asked  # several people: no one's personal profile
    await lv.mgr.end((1000, 77))


class _Stop(Exception):
    pass


@pytest.fixture
def tool_names(monkeypatch):
    """Runs the real qa.ask up to building the agent and returns the names of its tools."""
    seen = {}

    def grab(settings, tools, **kw):
        seen["names"] = {getattr(f, "__name__", None) or getattr(f, "name", str(f)) for f in tools}
        raise _Stop

    monkeypatch.setattr("salid.qa.build_agent", grab)
    return seen


async def test_chat_and_voice_build_the_same_agent_tools(lv, tool_names, monkeypatch):
    from salid import qa
    from salid.qa import MemoryTools

    bot = lv.bot
    s, live = await lv.join(5)
    mem = MemoryTools(bot.memory_store, 1000, {}, bot.s, 30)

    async def room_memory(guild, voice_channel, humans, *also, asker=None):
        return mem

    monkeypatch.setattr(bot, "_room_memory_tools", room_memory)
    monkeypatch.setattr(bot, "_web_kw", lambda gid: {"web_client": object()})
    member = lv.chan.members[0]
    msg = SimpleNamespace(
        id=1, guild=lv.guild, channel=lv.chan, author=member, content="", attachments=[],
        reference=None,
    )  # fmt: skip
    kw = bot._agent_kwargs(
        msg, lv.guild, member, True, setting="th", lang="th", question="q", memory=mem,
        profile=Profile(),
    )  # fmt: skip
    await asyncio.to_thread(_swallow, qa.ask, bot.s, bot.store, "q", guild_id=1000,
                            is_exec=True, user_id=5, deck=bot.team.enabled(1000, "decks"),
                            **kw)  # fmt: skip
    chat = set(tool_names.pop("names"))
    live.push(tool_msg("ask_salid", {"question": "q"}))
    await until(lambda: live.tool_responses)
    voice = tool_names["names"]
    # one assistant: exactly the same tools in chat and in a voice room (decks flag included)
    assert voice == chat - {"concept_image"}  # the one exception: an image cannot be spoken
    assert {"create_deck", "idea_export", "business_framework", "wiki_write"} <= voice
    assert {"send_dm", "deliver_file", "read_web"} <= voice
    assert {"meeting_attendance", "my_schedule"} <= voice  # personal schedule tools
    assert {"project_create", "project_show", "my_projects_and_tasks"} <= voice  # PM projects
    await lv.mgr.end((1000, 77))


def _swallow(fn, *a, **kw):
    with contextlib.suppress(_Stop):
        fn(*a, **kw)


async def test_voice_confirmation_works_across_consecutive_ask_salid_calls(lv):
    s, live = await lv.join(5)
    bot, asker = lv.bot, lv.chan.members[0]
    first = bot._room_secretary(lv.guild, asker, True, "th", lv.chan.id)  # call 1: propose
    second = bot._room_secretary(lv.guild, asker, True, "th", lv.chan.id)  # call 2: confirm
    key = ("meeting", 1000, asker.id, "same-args")
    assert first.request_id != second.request_id
    assert bot.secretary_state.confirm(key, first.request_id) is False
    assert bot.secretary_state.confirm(key, first.request_id) is False  # same call: not confirmed
    assert bot.secretary_state.confirm(key, second.request_id) is True
    await lv.mgr.end((1000, 77))


async def test_ask_salid_in_a_mixed_room_has_no_ledger_and_says_so(lv, asked):
    s, live = await lv.join(5, 11)  # 11 is not an executive
    live.push(tool_msg("ask_salid", {"question": "ยอดใช้จ่ายเดือนนี้"}))
    await until(lambda: live.tool_responses)
    assert asked["room"] == [5, 11]  # memory is scoped to everyone present
    assert asked["is_exec"] is False and asked["user_id"] == 0  # nobody's receipts
    assert lvm.LEDGER_NOTE in asked["question"]
    await lv.mgr.end((1000, 77))


async def test_ask_salid_single_non_exec_gets_only_their_own_data(lv, asked):
    s, live = await lv.join(11)
    live.push(tool_msg("ask_salid", {"question": "ใบเสร็จของฉัน"}))
    await until(lambda: live.tool_responses)
    assert asked["is_exec"] is False and asked["user_id"] == 11
    assert lvm.LEDGER_NOTE not in asked["question"]
    await lv.mgr.end((1000, 77))


async def test_ask_salid_refuses_unknown_tools_empty_questions_and_an_empty_room(lv, asked):
    s, live = await lv.join(5)
    live.push(
        tool_msg("spending_summary", {}, id="a"), tool_msg("ask_salid", {"question": " "}, id="b")
    )
    await until(lambda: len(live.tool_responses) == 2)
    by_id = {r[0].id: r[0].response for r in live.tool_responses}
    assert "unknown tool" in by_id["a"]["error"] and by_id["b"] == {"error": "empty question"}
    lv.chan.members = []
    live.push(tool_msg("ask_salid", {"question": "ใครมา"}, id="c"))
    await until(lambda: len(live.tool_responses) == 3)
    assert live.tool_responses[2][0].response == {"error": lvm.NO_ROOM}
    await lv.mgr.end((1000, 77))


async def test_ask_salid_failure_is_data_for_the_model(lv, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("agent down")

    monkeypatch.setattr("salid.qa.ask", boom)
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": "อะไรก็ได้"}))
    await until(lambda: live.tool_responses)
    res = live.tool_responses[0][0].response
    assert res["error"] == "agent down" and res["say"] == t("voice_tool_failed_say", "th")
    assert any("⚠️" in x and "ask_salid" in x and "agent down" in x for x in lv.chan.sent)
    await lv.mgr.end((1000, 77))


async def test_room_scope_is_the_channels_every_human_can_read(lv):
    class Chan:
        def __init__(self, id, name):
            self.id, self.name = id, name

    a, b = Chan(1, "all"), Chan(2, "execs-only")
    lv.guild.text_channels = [a, b]
    lv.guild.voice_channels = []
    lv.bot._channel_viewers = lambda g, c: frozenset({5, 11}) if c is a else frozenset({5})
    lv.bot._can_read = lambda c, m: m.id in ({5, 11} if c is a else {5})
    lv.bot.team.channel_mode = lambda *x: "listen"
    h5, h11 = SimpleNamespace(id=5), SimpleNamespace(id=11)
    vc = SimpleNamespace(id=77)
    assert lv.bot._room_scope(lv.guild, vc, [h5]) == {1: "all", 2: "execs-only"}
    assert lv.bot._room_scope(lv.guild, vc, [h5, h11]) == {1: "all"}  # #execs-only: 11 cannot read


async def test_qa_feature_off_means_no_tools_in_the_config_and_a_refusal(lv):
    lv.bot.guilds_store.save_settings(
        1000, GuildSettingsUpdate(features=dict(qa=False), default_language="th"), "x"
    )
    await asyncio.to_thread(lv.bot.team.current, 1000)
    s, live = await lv.join(5)
    names = [d.name for d in lv.connect.configs[0].tools[0].function_declarations]
    assert names == [
        "set_minutes_destination", "post_note", "ask_consent", "set_recording", "set_captions",
        "end_conversation",
    ]  # fmt: skip
    assert "ดูข้อมูลของทีมไม่ได้" in lv.connect.configs[0].system_instruction
    live.push(tool_msg("ask_salid", {"question": "อะไร"}))
    await until(lambda: live.tool_responses)
    assert live.tool_responses[0][0].response == {"error": lvm.QA_OFF}
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ reconnecting
async def test_go_away_reconnects_with_the_resumption_handle(lv):
    s, live = await lv.join(5)
    live.push(resume_msg("handle-1"), go_away_msg())
    await until(lambda: len(lv.connect.sessions) == 2)
    assert lv.connect.configs[1].session_resumption.handle == "handle-1"
    second = lv.connect.sessions[1]
    await until(lambda: s.live._live is second)
    assert live.closed
    s.on_pcm(5, pcm48(1000), time.monotonic())  # audio now flows to the new session
    await until(lambda: second.audio)
    assert s.voice_mode == "live"
    await lv.mgr.end((1000, 77))
    assert second.closed


async def test_lost_connection_reconnects_and_a_fresh_session_gets_recent_context(lv):
    s, live = await lv.join(5)
    s.transcript.add(lv.bot.s.now(), "แอน", "ตกลงส่งวันศุกร์")
    live.push(ConnectionError("1011 keepalive ping timeout"))
    await until(lambda: len(lv.connect.sessions) == 2)
    cfg = lv.connect.configs[1]
    assert cfg.session_resumption.handle is None  # never got a handle: starts over...
    assert "แอน: ตกลงส่งวันศุกร์" in cfg.system_instruction  # ...with the recent transcript
    assert s.voice_mode == "live"
    await lv.mgr.end((1000, 77))


async def test_reconnect_ends_playback_that_was_in_flight(lv):
    s, live = await lv.join(5)
    live.push(audio_msg())
    await until(lambda: lv.vc.played)
    live.push(ConnectionError("lost"))
    await until(lambda: len(lv.connect.sessions) == 2)
    assert play_source(lv)._complete  # plays out what it has, then stops; never hangs open
    await lv.mgr.end((1000, 77))


async def test_connect_failure_falls_back_to_classic_wake_words(lv, monkeypatch):
    lv.connect.fail_first = 99
    m = make_member(lv.guild, 5, "boss", lv.chan)
    lv.chan.members = [m]
    await lv.mgr.join(m)
    s = lv.mgr.sessions[(1000, 77)]
    await until(lambda: s.voice_mode == "classic")
    assert len(lv.connect.configs) == lvm.MAX_FAILURES and s.live._task.done()
    # wake words are answered by the classic pipeline again (spoken part off to stay offline)
    lv.bot.s.voice_reply = False
    import salid.qa

    monkeypatch.setattr(salid.qa, "ask", lambda *a, **kw: SimpleNamespace(text="ได้เลยนะ"))
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, mdl, pcm: "สลิด สรุปให้หน่อย")
    await lv.mgr.handle_segment(s, mt.Segment(5, SPEECH_PCM, 0.0, 2.0))
    assert lv.chan.sent[-1] == "<@5> ได้เลยนะ"
    await lv.mgr.end((1000, 77))


async def test_a_flaky_connect_that_recovers_stays_in_live_mode(lv):
    lv.connect.fail_first = 2
    s, live = await lv.join(5)
    assert len(lv.connect.configs) == 3 and s.voice_mode == "live"
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ transcript pipeline in live mode
async def test_live_mode_records_the_transcript_but_does_not_answer_wake_words(lv, monkeypatch):
    s, live = await lv.join(5, 11)
    asked = []
    import salid.qa

    monkeypatch.setattr(
        salid.qa, "ask", lambda *a, **kw: asked.append(a[2]) or SimpleNamespace(text="x")
    )
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, mdl, pcm: "สลิด เมื่อกี้ตกลงอะไรกัน")
    await lv.mgr.handle_segment(s, mt.Segment(11, bytes(3840 * 100), 0.0, 2.0))
    assert [(ln.speaker, ln.text) for ln in s.transcript.lines] == [("m11", "สลิด เมื่อกี้ตกลงอะไรกัน")]
    assert asked == [] and s.pending_wake == {}
    assert lv.chan.sent == [mt.t("meet_announce", "th")]  # nothing posted, no acknowledgment
    assert lv.vc.played == [] and s.speech_pending is False
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ lifecycle
async def test_end_closes_the_session_and_leaves_no_tasks_behind(lv):
    s, live = await lv.join(5)
    live.push(audio_msg(), said_msg("จบแล้วนะ"), done_msg())
    await until(lambda: "🔊 จบแล้วนะ" in lv.chan.sent)
    lv.vc.playing = True
    live_task = s.live._task
    await lv.mgr.end((1000, 77))
    assert live_task.done() and live.closed and lv.vc.disconnected
    assert s.live._bg == set() and s.live._live is None
    assert lv.vc.playing is False


async def test_close_is_idempotent_and_cancel_propagates(lv):
    s, live = await lv.join(5)
    await s.live.close()
    await s.live.close()
    assert s.live._task.done() and live.closed
    await lv.mgr.end((1000, 77))


async def test_feed_after_close_is_ignored(lv):
    s, live = await lv.join(5)
    await s.live.close()
    s.on_pcm(5, pcm48(1000), time.monotonic())
    assert s.live.mixer.pop() == b""
    await lv.mgr.end((1000, 77))


async def test_next_turn_waits_for_the_previous_player_after_callback(lv):
    s, live = await lv.join(5)
    live.push(audio_msg(), done_msg())
    await until(lambda: lv.vc.played and play_source(lv)._complete)
    old = play_source(lv)
    assert drain(old) == 2 and old.read() == b""

    # discord.py sets AudioPlayer._end before its player thread sends final silence and calls
    # `after`, so VoiceClient.is_playing() is already false during this winding-down window.
    old_after = lv.vc.after
    lv.vc.playing = False
    live.push(audio_msg())
    await until(lambda: s.live._source is not old)
    await asyncio.sleep(0.02)
    assert len(lv.vc.played) == 1

    old_after(None)
    await until(lambda: len(lv.vc.played) == 2)
    await lv.mgr.end((1000, 77))


async def test_bot_audio_is_not_sent_to_live_but_still_reaches_transcription(lv):
    s, live = await lv.join(5)
    frame = pcm48(1000)
    s.push_threadsafe(42, frame, is_bot=True)
    await until(lambda: 42 in s.segmenter.bufs)
    assert live.audio == [] and s.segmenter.bufs[42].size == len(frame)
    await lv.mgr.end((1000, 77))


async def test_close_releases_owned_genai_transports_once(lv):
    s, live = await lv.join(5)

    class FakeClientClose:
        def __init__(self):
            self.aio = self
            self.async_closes = 0
            self.sync_closes = 0

        async def aclose(self):
            self.async_closes += 1

        def close(self):
            self.sync_closes += 1

    client = FakeClientClose()
    s.live._client = client
    await s.live.close()
    await s.live.close()
    assert client.async_closes == client.sync_closes == 1
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ fast name detection
def heard_msg(text):
    sc = types.LiveServerContent(input_transcription=types.Transcription(text=text))
    return types.LiveServerMessage(server_content=sc)


def capture_cues(live):
    sent = []

    async def send_client_content(turns=None, turn_complete=True):
        sent.append((turns.parts[0].text, turn_complete))

    live.send_client_content = send_client_content
    return sent


async def test_name_in_the_live_input_transcript_forces_one_turn_and_dedupes_classic(lv):
    s, live = await lv.join(5)
    lv.bot.s.live_name_reply_seconds = 0.05
    sent = capture_cues(live)
    for i in range(0, len(SPEECH_PCM), 3840):
        s.live.feed(5, SPEECH_PCM[i : i + 3840])  # real speech behind the name (energy gate)
    live.push(heard_msg("ช่วยสรุปหน่อย"), heard_msg("สลิด "), heard_msg("วันนี้มีนัดอะไรบ้าง"))
    await until(lambda: any(t[1] for t in sent))
    cue = [t for t in sent if t[1]]
    assert len(cue) == 1 and "สลิด" in cue[0][0] and s.engaged is True
    # the classic transcription of the same utterance arrives later: no second forced turn
    s.live.ensure_reply(time.monotonic(), "สลิด วันนี้มีนัดอะไรบ้าง", user_id=5, question="q")
    await asyncio.sleep(0.2)
    assert len([t for t in sent if t[1]]) == 1
    assert s.live._fallback == (5, "q")  # but it is kept as the classic fallback
    await lv.mgr.end((1000, 77))


async def test_other_speech_in_the_input_transcript_does_nothing(lv):
    s, live = await lv.join(5)
    lv.bot.s.live_name_reply_seconds = 0.05
    sent = capture_cues(live)
    live.push(heard_msg("เขาสนิทกันมาก"), heard_msg(" แล้วก็ไปกินข้าว"), heard_msg(" สลิด ออกประชุม"))
    await asyncio.sleep(0.25)
    assert sent == [] and s.engaged is False  # leaving is the meeting manager's job
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ filler while a tool runs
@pytest.fixture
def slow_tool(monkeypatch):
    gate = asyncio.Event()

    async def run_tool(self, name, args):
        if name == "ask_salid":
            await gate.wait()
        return {"answer": "ok"}

    monkeypatch.setattr(LiveVoice, "_run_tool", run_tool)
    return gate


REAL = 7680  # a 1920-byte 24 kHz model chunk after upsampling to 48 kHz stereo


async def test_filler_plays_on_a_slow_tool_and_is_cut_when_the_answer_audio_arrives(lv, slow_tool):
    s, live = await lv.join(5)
    await until(lambda: len(s.live._filler_pcm) == 4)  # prerendered at startup
    live.push(tool_msg("ask_salid", {"question": "q"}))
    await until(lambda: lv.vc.played)
    src = play_source(lv)
    assert src.buffered_bytes > 0 and s.live._turn_audio is False  # filler, not model audio
    slow_tool.set()
    await until(lambda: live.tool_responses)
    live.push(audio_msg())
    await until(lambda: s.live._turn_audio)
    assert src.buffered_bytes == REAL and len(lv.vc.played) == 1  # filler dropped, same stream
    await lv.mgr.end((1000, 77))


async def test_filler_is_skipped_when_the_model_already_started_speaking(lv, slow_tool):
    s, live = await lv.join(5)
    live.push(audio_msg())
    await until(lambda: lv.vc.played)
    live.push(tool_msg("ask_salid", {"question": "q"}))
    await asyncio.sleep(0.15)
    assert play_source(lv).buffered_bytes == REAL  # nothing added
    slow_tool.set()
    await lv.mgr.end((1000, 77))


async def test_no_filler_for_fast_tools_and_a_typed_note_if_still_waiting(lv, slow_tool):
    s, live = await lv.join(5)
    await until(lambda: len(s.live._filler_pcm) == 4)
    lv.bot.s.live_filler_again_seconds = 0.05
    live.push(tool_msg("post_note", {"text": "x"}))
    await asyncio.sleep(0.15)
    assert not lv.vc.played
    live.push(tool_msg("ask_salid", {"question": "q"}))
    await until(lambda: lv.vc.played)
    one = play_source(lv).buffered_bytes
    await until(lambda: t("voice_working_text", "th", question="q") in lv.chan.sent)
    await asyncio.sleep(0.15)
    assert play_source(lv).buffered_bytes == one  # the 8 s note is typed, no second clip
    assert lv.chan.sent.count(t("voice_working_text", "th", question="q")) == 1  # once per job
    slow_tool.set()
    await lv.mgr.end((1000, 77))


async def test_filler_does_not_repeat_twice_in_a_row(lv):
    s, live = await lv.join(5)
    picks = [s.live._pick_fallback("th") for _ in range(40)]
    for i, p in enumerate(picks):  # never one of the 3 before it
        assert p not in picks[max(0, i - 3) : i]
    await lv.mgr.end((1000, 77))


async def test_spoken_questions_use_minimal_thinking_chat_keeps_low(monkeypatch, tmp_path):
    from salid import qa
    from salid.store import SQLiteStore

    seen = []

    def grab(settings, tools, **kw):
        seen.append(kw.get("thinking_level"))
        raise _Stop

    monkeypatch.setattr("salid.qa.build_agent", grab)
    store = SQLiteStore(str(tmp_path / "q.db"))
    for spoken in (True, False):
        await asyncio.to_thread(
            _swallow, qa.ask, FixedClockSettings(), store, "q", guild_id=1, is_exec=True,
            user_id=5, spoken=spoken,
        )  # fmt: skip
    assert seen == ["MINIMAL", "LOW"]


# ------------------------------------------------------------------ who is talking / who asked
async def test_speaker_note_is_sent_only_on_change_and_throttled(lv, monkeypatch):
    monkeypatch.setattr(lvm, "SPEAKER_NOTE_MIN_S", 0.3)
    s, live = await lv.join(5, 11)
    pcm = pcm48(1000)
    s.live.feed(5, pcm)
    await until(lambda: live.notes)
    assert live.notes == [("(ตอนนี้ m5 กำลังพูด)", False)]  # context only: the turn stays open
    s.live.feed(5, pcm)  # same speaker again: nothing
    s.live.feed(11, pcm)  # changed, but too soon after the last note: held back
    await asyncio.sleep(0.1)
    assert len(live.notes) == 1
    await asyncio.sleep(0.3)
    s.live.feed(11, pcm)  # the held-back change goes out on the next frame
    await until(lambda: len(live.notes) == 2)
    assert live.notes[1][0] == "(ตอนนี้ m11 กำลังพูด)"
    s.live.feed(11, pcm)
    await asyncio.sleep(0.1)
    assert len(live.notes) == 2
    await lv.mgr.end((1000, 77))


async def test_recent_speakers_single_then_overlapping(lv):
    s, live = await lv.join(5, 12, 11)
    pcm = pcm48(1000)
    assert s.live.recent_speakers() == []
    s.live.feed(12, pcm)
    assert s.live.recent_speakers() == [12]
    for uid in (5, 12, 5, 12):  # two people talking over each other
        s.live.feed(uid, pcm)
        await asyncio.sleep(0.01)
    assert s.live.recent_speakers() == [5, 12]
    assert s.live.recent_speakers(time.monotonic() + 60) == []  # too long ago: unknown
    await lv.mgr.end((1000, 77))


async def test_exec_speaker_with_a_junior_present_reads_as_junior_and_acts_as_exec(
    lv, asked, monkeypatch
):
    s, live = await lv.join(5, 11)  # 5 is an executive, 11 is not
    chat_calls = []
    orig = lv.bot._chat_actions

    def rec(message, guild, member, is_exec, *a, **k):
        chat_calls.append((member.id, is_exec))
        return orig(message, guild, member, is_exec, *a, **k)

    monkeypatch.setattr(lv.bot, "_chat_actions", rec)

    def coo(guild, asker, is_exec, memory, question, lang):
        def list_tasks():
            return asker.id

        def update_task():
            return asker.id

        return [list_tasks, update_task]

    monkeypatch.setattr(lv.bot, "_coo_tools", coo)
    monkeypatch.setattr(lv.bot, "_room_secretary", lambda *a: None)
    s.live.feed(5, pcm48(1000))
    live.push(tool_msg("ask_salid", {"question": "ส่งนัดให้หน่อย"}))
    await until(lambda: live.tool_responses)
    assert chat_calls == [(11, False), (5, True)]  # the room agent, then the speaker's actions
    reads = {
        f.__name__: f() for f in asked["extra_tools"] if f.__name__ in ("list_tasks", "update_task")
    }
    assert reads == {"list_tasks": 11, "update_task": 11}  # every read is built for the junior
    over = {f.__name__: f for f in asked["tool_overrides"]}
    assert over["update_task"]() == 5 and "send_dm" in over  # actions: the exec speaker
    assert "list_tasks" not in over
    assert asked["memory"] == "MEM"
    assert asked["is_exec"] is False and asked["user_id"] == 0  # no ledger: 11 is not an exec
    assert "ไม่ใช่ผู้บริหาร" in asked["question"] and "DM" in asked["question"]
    await lv.mgr.end((1000, 77))


async def test_single_human_room_is_unchanged_no_rebinding(lv, asked):
    s, live = await lv.join(5)
    s.live.feed(5, pcm48(1000))
    live.push(tool_msg("ask_salid", {"question": "q"}))
    await until(lambda: live.tool_responses)
    assert "tool_overrides" not in asked and asked["sec_args"][:2] == (5, True)
    await lv.mgr.end((1000, 77))


async def test_answer_privately_dms_the_speaker_and_returns_no_content(lv, asked, monkeypatch):
    from salid.dm import DMResult

    sent = []

    async def fake_dm(member, text, files=()):
        sent.append((member.id, text))
        return DMResult(True)

    async def mem(message, guild, member):
        asked["private_for"] = (message.guild, member.id)
        return "PMEM"

    monkeypatch.setattr("salid.dm.send_dm", fake_dm)
    monkeypatch.setattr(lv.bot, "_memory_tools", mem)
    s, live = await lv.join(5, 11)
    s.live.feed(5, pcm48(1000))
    live.push(tool_msg("answer_privately", {"question": "ยอดใช้จ่ายเดือนนี้"}))
    await until(lambda: live.tool_responses)
    resp = live.tool_responses[0][0].response
    assert resp == {"status": "sent"}  # nothing of the answer reaches the voice model
    assert sent and sent[0][0] == 5 and sent[0][1].startswith("xxx")
    assert asked["is_exec"] is True and asked["user_id"] == 5  # the speaker's own full rights
    assert asked["memory"] == "PMEM" and asked["private_for"] == (None, 5)
    assert "spoken" not in asked
    await lv.mgr.end((1000, 77))


def spy_agent_kwargs(bot, monkeypatch) -> list:
    """The `spoken` keyword of every `_agent_kwargs` call."""
    seen, real = [], bot._agent_kwargs

    def spy(*a, **kw):
        seen.append(kw.get("spoken"))
        return real(*a, **kw)

    monkeypatch.setattr(bot, "_agent_kwargs", spy)
    return seen


async def test_ask_salid_builds_the_agent_as_a_spoken_answer(lv, asked, monkeypatch):
    seen = spy_agent_kwargs(lv.bot, monkeypatch)
    s, live = await lv.join(5, 12, 11)  # a room: the agent is built for two people
    live.push(tool_msg("ask_salid", {"question": "q"}))
    await until(lambda: live.tool_responses)
    assert seen and all(x is True for x in seen)
    await lv.mgr.end((1000, 77))


async def test_answer_privately_builds_the_agent_as_a_spoken_answer(lv, asked, monkeypatch):
    from salid.dm import DMResult

    async def fake_dm(member, text, files=()):
        return DMResult(True)

    async def mem(message, guild, member):
        return "PMEM"

    monkeypatch.setattr("salid.dm.send_dm", fake_dm)
    monkeypatch.setattr(lv.bot, "_memory_tools", mem)
    seen = spy_agent_kwargs(lv.bot, monkeypatch)
    s, live = await lv.join(5, 11)
    s.live.feed(5, pcm48(1000))
    live.push(tool_msg("answer_privately", {"question": "q"}))
    await until(lambda: live.tool_responses)
    assert seen == [True]
    await lv.mgr.end((1000, 77))


async def test_answer_privately_needs_a_known_speaker(lv, asked):
    s, live = await lv.join(5, 11)
    live.push(tool_msg("answer_privately", {"question": "q"}))
    await until(lambda: live.tool_responses)
    assert "error" in live.tool_responses[0][0].response
    await lv.mgr.end((1000, 77))


def test_rebind_actions_swaps_only_whitelisted_names_and_never_adds():
    from salid.voice_actions import rebind_actions

    def mk(tag, *names):
        out = []
        for n in names:
            f = lambda tag=tag: tag  # noqa: E731
            f.__name__ = n
            out.append(f)
        return out

    room = {"extra_tools": mk("room", "list_tasks", "request_approval"), "x": 1}
    speaker = {"extra_tools": mk("spk", "list_tasks", "request_approval", "set_availability")}
    out = rebind_actions(room, speaker, {"request_approval", "set_availability"})
    assert out["x"] == 1 and [f() for f in out["tool_overrides"]] == ["spk", "spk"]

    from salid import qa

    seen = {}

    def grab(settings, tools, **kw):
        seen["tools"] = tools
        raise _Stop

    import pytest as _p

    mp = _p.MonkeyPatch()
    mp.setattr("salid.qa.build_agent", grab)
    try:
        from conftest import FixedClockSettings as S

        from salid.store import SQLiteStore

        store = SQLiteStore(":memory:")
        _swallow(qa.ask, S(), store, "q", guild_id=1, is_exec=False, user_id=1,
                 extra_tools=room["extra_tools"], tool_overrides=out["tool_overrides"])  # fmt: skip
    finally:
        mp.undo()
    got = {
        f.__name__: f() for f in seen["tools"] if f.__name__ in ("list_tasks", "request_approval")
    }
    assert got == {"list_tasks": "room", "request_approval": "spk"}


async def test_overlapping_speakers_act_as_the_lowest_ranked_of_them(lv, asked):
    s, live = await lv.join(5, 12, 11)  # ranks 9, 4, 1; 11 stayed quiet
    pcm = pcm48(1000)
    for uid in (5, 12, 5, 12):
        s.live.feed(uid, pcm)
        await asyncio.sleep(0.01)
    live.push(tool_msg("ask_salid", {"question": "q"}))
    await until(lambda: live.tool_responses)
    assert asked["sec_args"][0] == 12 and asked["sec_args"][1] is False
    assert "profile" not in asked
    await lv.mgr.end((1000, 77))


# ------------------------------------------------------------------ intentional barge-in
def test_realtime_input_never_lets_noise_interrupt_unless_configured():
    on = lvm.build_realtime_input(SimpleNamespace())
    assert on.activity_handling == types.ActivityHandling.NO_INTERRUPTION
    old = lvm.build_realtime_input(SimpleNamespace(live_barge_in="any"))
    assert old.activity_handling is None


@pytest.mark.parametrize(
    "text, stop",
    [
        ("สลิด หยุดก่อน", True), ("เดี๋ยวก่อนนะ", True), ("พอแล้ว", True), ("ขอแทรกหน่อย", True),
        ("Salid", True), ("wait, stop", True), ("hold on", True),
        ("วันนี้ฝนตกหนักเลย", False), ("เอกสารอยู่ในโฟลเดอร์ไหน", False), ("the stopwatch", False),
        ("", False),
    ],
)  # fmt: skip
def test_stop_phrases(text, stop):
    assert lvm.is_stop_request(text) is stop


async def test_side_talk_does_not_stop_salid_but_stop_does_and_drops_the_rest(lv):
    s, live = await lv.join(5, 11)
    s.live._partners = {5}
    live.push(audio_msg(), said_msg("คำตอบยาวๆ"))
    await until(lambda: lv.vc.played and s.speaking)
    s.live._partners = {5}  # set again: the first audio of the turn recomputed it
    s.live._turn_open = True
    live.push(heard_msg("ไปกินข้าวที่ไหนกันดี"))  # someone else chatting nearby
    await asyncio.sleep(0.05)
    assert lv.vc.playing
    live.push(heard_msg(" สลิด หยุดก่อน"))
    await until(lambda: lv.vc.playing is False)
    live.push(audio_msg())  # the rest of the stopped answer never plays
    await asyncio.sleep(0.05)
    assert len(lv.vc.played) == 1
    live.push(done_msg())
    await asyncio.sleep(0.03)
    assert all("คำตอบยาวๆ" not in m for m in lv.chan.sent)
    live.push(audio_msg())  # the next answer plays again
    await until(lambda: len(lv.vc.played) == 2)
    await lv.mgr.end((1000, 77))


async def test_only_the_partner_talking_over_salid_stops_her(lv):
    s, live = await lv.join(5, 11)
    live.push(audio_msg())
    await until(lambda: lv.vc.played and s.speaking)
    v = s.live
    v._partners, v._turn_open = {5}, True
    v._turn_started_at = 900.0  # her answer began long ago: no early cut
    loud, quiet = pcm48(8000), pcm48(50)
    t0 = 1000.0
    for i in range(150):  # 3 s of loud talk from a bystander (id 11): ignored
        v._check_talkover(11, loud, t0 + i * 0.02)
    for i in range(150):  # 3 s of the partner's mic noise below the voice threshold: ignored
        v._check_talkover(5, quiet, t0 + i * 0.02)
    await asyncio.sleep(0.02)
    assert lv.vc.playing
    for i in range(30):  # 0.6 s of the partner talking: not enough
        v._check_talkover(5, loud, t0 + 10 + i * 0.02)
    assert lv.vc.playing
    for i in range(30, 45):  # 0.8 s of clear speech over her: stop
        v._check_talkover(5, loud, t0 + 10 + i * 0.02)
    await until(lambda: lv.vc.playing is False)
    assert v._muted_turn
    await lv.mgr.end((1000, 77))


async def test_stop_after_the_turn_finished_does_not_mute_the_next_answer(lv):
    s, live = await lv.join(5)
    live.push(audio_msg(48000), said_msg("ยาว"), done_msg())  # whole turn received, still playing
    await until(lambda: lv.vc.played and s.speaking)
    assert s.live._turn_open is False
    live.push(heard_msg("หยุดก่อน"))
    await until(lambda: lv.vc.playing is False)
    assert s.live._muted_turn is False
    live.push(audio_msg())
    await until(lambda: len(lv.vc.played) == 2)
    await lv.mgr.end((1000, 77))


async def test_any_mode_keeps_the_old_server_barge_in(lv):
    lv.bot.s.live_barge_in = "any"
    s, live = await lv.join(5)
    live.push(audio_msg())
    await until(lambda: lv.vc.played and s.speaking)
    live.push(heard_msg("เดี๋ยวก่อน"))  # no local stop in "any" mode; the server decides
    await asyncio.sleep(0.05)
    assert lv.vc.playing
    live.push(interrupted_msg())
    await until(lambda: lv.vc.playing is False)
    await lv.mgr.end((1000, 77))


class FillerStub:
    """Stands in for FillerWriter in the filler tests: per-stage lines, optional delay / error."""

    def __init__(self, first=None, again=None, delay=0.0, boom=False):
        self.lines = {"first": first, "again": again}
        self.delay, self.boom, self.calls = delay, boom, []

    def enabled(self):
        return True

    async def write(self, **kw):
        self.calls.append(kw)
        await asyncio.sleep(self.delay)
        if self.boom:
            raise RuntimeError("model down")
        return self.lines[kw["stage"]]


@pytest.fixture
def tts_spy(lv, monkeypatch):
    lv.said = []

    def say(settings, client, text, lang="th"):
        lv.said.append(text)
        return bytes(4800)

    monkeypatch.setattr(lvm, "say", say)
    return lv


async def test_filler_is_written_for_the_question_and_the_again_line_is_typed_not_spoken(
    tts_spy, slow_tool
):
    lv = tts_spy
    stub = lv.bot.fillers = FillerStub("ขอเช็กยอดเดือนนี้แป๊บนะคะ", "ยังหาอยู่นะคะ ข้อมูลเยอะหน่อย")
    lv.bot.s.live_filler_again_seconds = 0.05
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": "ยอดเดือนนี้เท่าไร"}))
    note = t("voice_working_text", "th", question="ยอดเดือนนี้เท่าไร")
    await until(lambda: note in lv.chan.sent)
    assert "ขอเช็กยอดเดือนนี้แป๊บนะคะ" in lv.said
    assert "ยังหาอยู่นะคะ ข้อมูลเยอะหน่อย" not in lv.said  # no second clip
    assert [c["stage"] for c in stub.calls] == ["first"]  # no "again" line is even written
    assert all(c["question"] == "ยอดเดือนนี้เท่าไร" and c["kind"] == "voice" for c in stub.calls)
    slow_tool.set()
    await lv.mgr.end((1000, 77))


@pytest.mark.parametrize("stub", [FillerStub("ช้าไปค่ะ", delay=1.0), FillerStub(boom=True)])
async def test_slow_or_failing_line_plays_a_prerendered_fallback_only(tts_spy, slow_tool, stub):
    lv = tts_spy
    lv.bot.fillers = stub
    lv.bot.s.voice_ack_timeout_s = 0.05
    s, live = await lv.join(5)
    await until(lambda: len(s.live._filler_pcm) == 4)
    before = list(lv.said)
    live.push(tool_msg("ask_salid", {"question": "q"}))
    await until(lambda: lv.vc.played)
    await asyncio.sleep(0.2)
    pool = [catalog_th(n) for n in range(1, 9)]
    assert lv.said == before  # the fallback clip was already rendered; the late line never was
    assert "ช้าไปค่ะ" not in lv.said and all(x in pool for x in before[:4])
    slow_tool.set()
    await lv.mgr.end((1000, 77))


def catalog_th(n):
    from salid.i18n import t

    return t(f"ack_voice_{n}", "th")


async def test_prerender_is_bounded_not_the_whole_pool(tts_spy):
    lv = tts_spy
    s, _live = await lv.join(5)
    await until(lambda: len(s.live._filler_pcm) == 4)
    await asyncio.sleep(0.1)
    assert len(lv.said) == 4  # 4 of the 8 first-line fallbacks, no "again" clips
    await lv.mgr.end((1000, 77))


async def test_generic_line_clip_is_cached_on_disk_and_not_synthesized_twice(tts_spy, slow_tool):
    from salid.fillers import Line

    lv = tts_spy

    class Pooled(FillerStub):
        async def write_ex(self, **kw):
            self.calls.append(kw)
            return Line("เดี๋ยวสลิดเปิดดูให้นะคะ", True)

    lv.bot.fillers = Pooled()
    s, live = await lv.join(5)
    await until(lambda: len(s.live._filler_pcm) == 4)
    n = len(lv.said)
    live.push(tool_msg("ask_salid", {"question": "q"}))
    await until(lambda: lv.said.count("เดี๋ยวสลิดเปิดดูให้นะคะ") == 1)
    slow_tool.set()
    await until(lambda: live.tool_responses)
    assert len(lv.said) == n + 1
    # a second question: the same pooled line comes from disk, no TTS call
    s.live._turn_audio = False
    live.push(tool_msg("ask_salid", {"question": "again"}, id="call-2"))
    await asyncio.sleep(0.3)
    assert lv.said.count("เดี๋ยวสลิดเปิดดูให้นะคะ") == 1
    await lv.mgr.end((1000, 77))
