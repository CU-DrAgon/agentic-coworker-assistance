"""Live voice mode, fully offline: audio math, the playback source, and a fake Gemini Live session
driving the real LiveVoice through MeetingManager. No Discord, no Vertex."""

import asyncio
import contextlib
import logging
import time
from array import array
from types import SimpleNamespace

import pytest
from conftest import ALICE, FakeClient, FixedClockSettings, mk
from google.genai import types
from test_meeting import FakeGuild, FakeVC, FakeVoiceChannel, make_member

from salid import live_voice as lvm
from salid import meeting as mt
from salid.bot import SecretaryBot
from salid.guild_settings import GuildSettingsUpdate, MemoryGuildStore
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
    assert "กำลังฟัง" in th and "กำลังคุยอยู่" in th and "พูดภาษาไทย" in th
    assert "always speak English" in build_instruction(s, "en")
    assert "ภาษาเดียวกับคนที่พูด" in build_instruction(s, "auto")
    assert "ดูข้อมูลของทีมไม่ได้" in build_instruction(s, "th", qa=False)
    assert "ask_salid" not in build_instruction(s, "th", qa=False)
    assert "แอน: ส่งงาน" in build_instruction(s, "th", history="แอน: ส่งงาน")


def test_instruction_answers_general_questions_without_tools_and_never_raises_money():
    th = build_instruction(FixedClockSettings(), "th")
    assert "ตอบเองตรง ๆ ห้ามเรียกเครื่องมือ" in th  # general knowledge, opinions, small talk
    assert "ห้ามพูดถึงเงิน ใบเสร็จ หรือค่าใช้จ่ายเองเด็ดขาด" in th
    assert "แป๊บนึงนะคะ" in th  # a short natural filler while ask_salid runs
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
            exec_user_ids={5}, voice_reply=True, voice_mode="live", meeting_text_echo=True
        ),
        store,
        FakeClient([]),
        guilds=MemoryGuildStore(),
    )
    monkeypatch.setattr(SecretaryBot, "user", property(lambda s: ME))
    monkeypatch.setattr(lvm, "SEND_INTERVAL_S", 0.01)
    monkeypatch.setattr(lvm, "STREAM_IDLE_S", 0.06)
    monkeypatch.setattr(lvm, "BACKOFF_S", (0.0, 0.0, 0.0, 0.0))
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
        "ask_salid", "set_minutes_destination", "post_note", "ask_consent", "set_recording",
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

    async def room_memory(guild, voice_channel, humans):
        seen["room"] = sorted(h.id for h in humans)
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


async def test_ask_salid_gets_the_secretary_and_coo_tools_for_the_lone_human(lv, asked):
    s, live = await lv.join(5)
    live.push(tool_msg("ask_salid", {"question": "ประชุมพรุ่งนี้มีอะไรบ้าง"}))
    await until(lambda: live.tool_responses)
    assert asked["secretary"] == "SEC" and asked["extra_tools"] == ["COO"]
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
        seen["names"] = {getattr(f, "__name__", str(f)) for f in tools}
        raise _Stop

    monkeypatch.setattr("salid.qa.build_agent", grab)
    return seen


async def test_chat_and_voice_build_the_same_agent_tools(lv, tool_names, monkeypatch):
    from salid import qa
    from salid.qa import MemoryTools

    bot = lv.bot
    s, live = await lv.join(5)
    mem = MemoryTools(bot.memory_store, 1000, {}, bot.s, 30)

    async def room_memory(guild, voice_channel, humans):
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
                            is_exec=True, user_id=5, **kw)  # fmt: skip
    chat = set(tool_names.pop("names"))
    live.push(tool_msg("ask_salid", {"question": "q"}))
    await until(lambda: live.tool_responses)
    assert tool_names["names"] == chat  # one assistant: same tools in chat and in a voice room
    assert {"send_dm", "deliver_file", "read_web"} <= chat  # the chat-only tools are there too
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
    assert live.tool_responses[0][0].response == {"error": "agent down"}
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
    lv.bot.guilds_store.save_settings(1000, GuildSettingsUpdate(features=dict(qa=False)), "x")
    await asyncio.to_thread(lv.bot.team.current, 1000)
    s, live = await lv.join(5)
    names = [d.name for d in lv.connect.configs[0].tools[0].function_declarations]
    assert names == ["set_minutes_destination", "post_note", "ask_consent", "set_recording"]
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
    await lv.mgr.handle_segment(s, mt.Segment(5, bytes(3840 * 100), 0.0, 2.0))
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
