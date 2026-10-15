"""Live voice mode: a Gemini Live (native audio) session hears the whole voice channel and talks back.

Compared with the classic pipeline (wake word -> transcription -> text answer -> spoken rewrite ->
TTS, ~11 s) the model answers ~1.5 s after someone stops talking, can be interrupted (barge-in) and,
with proactive audio on, stays silent while people talk among themselves.

    Discord (48 kHz stereo, one stream per speaker)
      -> Mixer: downmix, 16 kHz, sum all speakers      -> Live session (realtime audio in)
    Live session (24 kHz mono audio out, transcripts, tool calls)
      -> LiveAudioSource (48 kHz stereo, 20 ms frames) -> VoiceClient.play

The per-speaker transcription pipeline in meeting.py keeps running for the transcript and summary
(speaker names); in live mode it just does not answer wake words. Audio is never stored or logged.
Account data stays behind the same ledger tools as the text Q&A; because the mixed stream does not
say who is speaking, tools are only usable when everyone in the room is an executive.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import threading
import time
from array import array
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import discord
from google.genai import types

from .meeting import SPEAK_COOLDOWN_S, _connected, _stop_playback
from .qa import THAI_DAYS, LedgerTools, build_tools

if TYPE_CHECKING:
    from .bot import SecretaryBot
    from .meeting import MeetingSession

logger = logging.getLogger(__name__)

# ---- audio formats
DISCORD_RATE = 48000  # what MeetingSink delivers (stereo s16le)
LIVE_IN_RATE = 16000  # what the Live API wants (mono s16le)
LIVE_OUT_RATE = 24000  # what the Live API returns (mono s16le)
FRAME_BYTES = 3840  # 20 ms of 48 kHz stereo s16le: what discord.py asks for per read()
SILENCE = bytes(FRAME_BYTES)

# ---- streaming in
SEND_INTERVAL_S = 0.1  # mixed audio goes out in ~100 ms chunks
MAX_CHUNK_SAMPLES = LIVE_IN_RATE * 3 // 10  # catch up on a backlog, 300 ms at most per send
STREAM_IDLE_S = 1.0  # silence is sent this long after the last voice (lets the server VAD end the
# turn), then audio_stream_end once; nothing more until someone speaks again
MIXER_MAX_S = 3  # per speaker; older audio is dropped

# ---- output / session
SOURCE_STALL_S = 20.0  # an open turn with no audio for this long is ended (model never answered)
MAX_FAILURES = 4  # consecutive failed (re)connects before falling back to classic mode
BACKOFF_S = (0.5, 1.0, 2.0, 4.0)
STABLE_S = 10.0  # a connection that lived this long resets the failure count
PLAY_RETRIES = 5


def _sensitivity(enum_cls, name: str, default: str):
    """`enum_cls` member for "high" / "low" (the default when unknown)."""
    prefix = "START_SENSITIVITY_" if "Start" in enum_cls.__name__ else "END_SENSITIVITY_"
    return getattr(enum_cls, prefix + str(name or default).upper(), None) or getattr(
        enum_cls, prefix + default.upper()
    )


def build_realtime_input(settings: Any) -> types.RealtimeInputConfig:
    """Turn-taking for natural Thai talk: short greetings start a turn quickly (start sensitivity
    high), a mid-sentence pause does not end it (end sensitivity low, long silence_duration_ms)."""
    return types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(
            start_of_speech_sensitivity=_sensitivity(
                types.StartSensitivity, getattr(settings, "live_start_sensitivity", "high"), "high"
            ),
            end_of_speech_sensitivity=_sensitivity(
                types.EndSensitivity, getattr(settings, "live_end_sensitivity", "low"), "low"
            ),
            prefix_padding_ms=int(getattr(settings, "live_prefix_padding_ms", 300)),
            silence_duration_ms=int(getattr(settings, "live_silence_ms", 1200)),
        )
    )


def set_minutes_destination(channel: str) -> dict:
    """Choose the text channel where the meeting minutes / resolutions are posted when the meeting
    ends. Call it when someone says e.g. "ส่งมติการประชุมไปห้องบัญชี" or "send the minutes to
    #accounting". Then confirm in one short spoken sentence. If the result has an error, tell them
    briefly that the channel was not found / is ambiguous / not everyone can see it.

    Args:
        channel: the channel name as said (without "ห้อง" if possible).
    """
    return {}  # declaration only: executed by LiveVoice._run_tool


def post_note(text: str) -> dict:
    """Write ONE short line in the meeting's text chat: a decision, an action item (who, what, by
    when) or a number someone asked you to remember. Use sparingly, never to repeat what was said
    and never for chit-chat. Talking with people stays voice.

    Args:
        text: the note, one short line in the language of the meeting.
    """
    return {}


def ask_consent(text: str, medium: str = "text") -> dict:
    """Ask, once, whether this is a meeting and whether you may take notes. Use it when people
    have been talking for a while, nobody has talked to you and it is unclear what this is (no
    schedule). Ask like a colleague, briefly; they can say no.

    Args:
        text: your short, natural question.
        medium: "text" (default; does not interrupt) or "voice" (the room is quiet or someone is
            talking to you; the text is then spoken for you, do not also say it yourself).
    """
    return {}


def set_recording(recording: bool) -> dict:
    """Switch note-taking. Call with false when people say it is not a meeting or they do not want
    notes ("ไม่ต้องจด", "คุยเล่น"); true when they say a meeting is starting or ask you to take notes.

    Args:
        recording: true to take notes, false to stop.
    """
    return {}


_MEETING_TOOLS = {
    "set_minutes_destination": "set_minutes_destination",
    "post_note": "post_note",
    "ask_consent": "ask_consent",
    "set_recording": "set_recording",
}

STATE_LISTENING = "(สถานะ: กำลังฟังอยู่ ยังไม่มีใครคุยกับคุณ เงียบไว้จนกว่าจะมีคนเรียกชื่อสลิด)"
STATE_ENGAGED = (
    "(สถานะ: กำลังคุยกับคุณอยู่ ตอบทุกประโยคที่คนในห้องพูดกับคุณเหมือนบทสนทนาปกติ "
    "ไม่ต้องรอให้เรียกชื่อ ถ้าเขาบอกพอแล้ว/ขอบคุณ ให้ตอบ “ได้ค่ะ” สั้น ๆ แล้วกลับไปเงียบฟัง)"
)
FORCE_CUE = "(มีคนเรียกชื่อคุณ: ตอบกลับทันทีตามบริบท)"

NO_TOOLS_MIXED_ROOM = "ข้อมูลบัญชีของร้านดูได้เฉพาะเมื่อทุกคนในห้องเป็นผู้บริหาร ถามในแชทแทนได้นะ"
QA_OFF = "ฟีเจอร์ถามตอบข้อมูลบัญชีถูกปิดอยู่"


# ------------------------------------------------------------------ audio math
def to_live_input(pcm: bytes) -> array:
    """48 kHz stereo s16le -> 16 kHz mono samples: L and R averaged, every 3 frames averaged."""
    a = array("h")
    a.frombytes(pcm[: len(pcm) // 12 * 12])  # whole groups of 3 stereo frames (6 samples)
    return array(
        "h",
        [
            (p + q + r + s + t + u) // 6
            for p, q, r, s, t, u in zip(
                a[0::6], a[1::6], a[2::6], a[3::6], a[4::6], a[5::6], strict=True
            )
        ],
    )


def upsample_output(pcm: bytes) -> bytes:
    """24 kHz mono s16le -> 48 kHz stereo s16le (each sample repeated twice in time, L = R)."""
    s = array("h")
    s.frombytes(pcm[: len(pcm) // 2 * 2])
    out = array("h", bytes(len(s) * 8))
    for k in range(4):
        out[k::4] = s
    return out.tobytes()


class Mixer:
    """Sums every speaker into one 16 kHz mono stream. Each speaker has a contiguous buffer
    (Discord only sends frames while someone talks), so speakers need no sample-exact alignment."""

    def __init__(self, max_s: float = MIXER_MAX_S):
        self._bufs: dict[int, array] = {}
        self._max = int(LIVE_IN_RATE * max_s)

    def feed(self, user_id: int, pcm48: bytes) -> None:
        buf = self._bufs.setdefault(user_id, array("h"))
        buf.extend(to_live_input(pcm48))
        if len(buf) > self._max:
            del buf[: len(buf) - self._max]

    def pop(self, max_samples: int = MAX_CHUNK_SAMPLES) -> bytes:
        """Up to `max_samples` mixed samples (s16le), b"" when nobody has audio pending."""
        take = min(max_samples, max((len(b) for b in self._bufs.values()), default=0))
        if not take:
            return b""
        acc = [0] * take
        for uid in list(self._bufs):
            buf = self._bufs[uid]
            n = min(take, len(buf))
            for i in range(n):
                acc[i] += buf[i]
            del buf[:n]
            if not buf:
                del self._bufs[uid]
        return array(
            "h", [-32768 if x < -32768 else 32767 if x > 32767 else x for x in acc]
        ).tobytes()

    def clear(self) -> None:
        self._bufs.clear()


class LiveAudioSource(discord.AudioSource):
    """What the model says, as an endless-until-finished stream for `VoiceClient.play`.

    The player thread calls read() every 20 ms; the event loop feeds audio in. read() returns exactly
    3840 bytes while the turn is open (silence when the buffer ran dry: the model is still
    generating) and b"" once the turn is finished and the buffer is drained, which ends playback."""

    def __init__(self, stall_s: float = SOURCE_STALL_S):
        self._lock = threading.Lock()
        self._buf = bytearray()
        self._carry = b""  # odd byte left over from a chunk
        self._complete = False
        self._stall_s = stall_s
        self._last_feed = time.monotonic()
        self.ended = False  # read() returned b"" (or cleanup ran): this source is spent

    def feed(self, pcm24k: bytes) -> bool:
        """Queue model audio (24 kHz mono). False when the source is already spent."""
        with self._lock:
            if self.ended:
                return False
            data = self._carry + pcm24k
            n = len(data) // 2 * 2
            self._carry = data[n:]
            if n:
                self._buf += upsample_output(data[:n])
            self._complete = False  # more audio: the turn goes on (also re-opens a finished turn)
            self._last_feed = time.monotonic()
            return True

    def finish(self) -> None:
        """The turn is complete: end after the queued audio has played."""
        with self._lock:
            self._complete = True

    def clear(self) -> None:
        """Drop everything queued (barge-in)."""
        with self._lock:
            self._buf.clear()
            self._carry = b""

    @property
    def buffered_bytes(self) -> int:
        with self._lock:
            return len(self._buf)

    def read(self) -> bytes:
        with self._lock:
            if self.ended:
                return b""
            if len(self._buf) >= FRAME_BYTES:
                out = bytes(self._buf[:FRAME_BYTES])
                del self._buf[:FRAME_BYTES]
                return out
            done = self._complete or time.monotonic() - self._last_feed > self._stall_s
            if not done:
                return SILENCE  # still generating
            if self._buf:  # last partial frame, padded
                out = bytes(self._buf).ljust(FRAME_BYTES, b"\0")
                self._buf.clear()
                return out
            self.ended = True
            return b""

    def is_opus(self) -> bool:
        return False

    def cleanup(self) -> None:
        with self._lock:
            self.ended = True
            self._buf.clear()


# ------------------------------------------------------------------ instruction
def build_instruction(
    settings: Any, language: str = "th", now=None, qa: bool = True, history: str = ""
) -> str:
    """System instruction for the Live session (Thai, built per connect so the date is current)."""
    now = now or settings.now()
    p = (
        "คุณชื่อ “สลิด” (Salid) เป็นเพื่อนร่วมทีมและผู้ช่วยที่นั่งอยู่ในห้องเสียง Discord ของทีมและฟังทุกคนคุยกันอยู่ "
        "อ่านบรรยากาศในห้องเป็น เข้าร่วมบทสนทนาอย่างเป็นธรรมชาติ ถามชวนคุยเหมือนเพื่อนร่วมงานที่ดี "
        "ไม่สั่งสอน ไม่อ่านกฎหรือนโยบายให้ใคร\n"
        "กติกาการพูด:\n"
        "- มี 2 สถานะ ระบบจะบอกสถานะให้เป็นข้อความในวงเล็บ: (1) กำลังฟัง (ค่าเริ่มต้น) คุณฟังและจดเงียบ ๆ "
        "ห้ามพูด ยกเว้นเมื่อมีคนเรียกชื่อ “สลิด” (Salid, สะลิด, สลิต, ซาลิด) ให้ตอบทันทีสั้น ๆ โดยเฉพาะถ้าเป็นการทักทาย ให้ทักกลับ "
        "(2) กำลังคุยอยู่ ตอบทุกประโยคที่พูดกับคุณเหมือนบทสนทนาปกติ ไม่ต้องรอชื่อ ถ้าเขาบอกพอแล้ว/ขอบคุณ ให้ตอบ “ได้ค่ะ” สั้น ๆ แล้วเงียบ\n"
        "- ถ้าคนในห้องคุยกันเอง ให้เงียบสนิท ไม่ต้องตอบหรือแสดงความเห็น\n"
        "- ตอบสั้น 1-3 ประโยค เป็นธรรมชาติเหมือนคุยกันจริง ไม่อ่านเป็นรายการยาว ๆ\n"
        "- บุคลิกน่ารัก ร่าเริง เป็นกันเอง เสียงผู้หญิง ลงท้ายด้วย “ค่ะ” “คะ” “นะคะ” ห้ามส่งเสียงสัตว์\n"
        "- ตอนเข้าห้องอย่าเพิ่งพูดอะไร รอจนมีคนทักหรือคุยกับคุณ\n"
        "- ถ้าไม่ชัดว่านี่คือการประชุมไหม (ไม่มีนัดหมาย คุยกันสบาย ๆ) และไม่มีใครคุยกับคุณมาสักพัก คุณถามได้เองแบบเพื่อนร่วมงาน "
        "เช่น “วันนี้ประชุมกันหรือเปล่าคะ” โดยเรียก ask_consent (ปกติเป็นข้อความ จะได้ไม่รบกวน; พูดเมื่อห้องเงียบหรือมีคนคุยกับคุณ) "
        "ถ้าตอบว่าไม่ใช่การประชุม/ไม่ต้องจด ให้เรียก set_recording(false) แล้วพูดสั้น ๆ ว่าไม่จดแล้ว และเงียบรอให้เรียก "
        "ถ้าเป็นการประชุมให้จดต่อ\n"
        "- การคุยกับคนต้องเป็นเสียงเท่านั้น ห้ามส่งข้อความซ้ำสิ่งที่พูด ส่วนสิ่งที่ควรจดจริง ๆ (มติ งานที่ต้องทำพร้อมเจ้าของและกำหนดเวลา "
        "ตัวเลขที่มีคนขอให้จำ) ให้เรียก post_note บรรทัดสั้น ๆ ต่อหนึ่งเรื่อง ไม่ใช่ทุกประโยค\n"
        "- ถ้ามีคนบอกให้ส่งมติ/สรุปการประชุมไปห้องไหน ให้เรียกเครื่องมือ set_minutes_destination แล้วยืนยันสั้น ๆ\n"
        f"วันนี้คือ {now.date().isoformat()} (วัน{THAI_DAYS[now.weekday()]}) เวลา {now:%H:%M} เขตเวลา {settings.tz}. "
        "สัปดาห์เริ่มวันจันทร์. ใช้วันที่นี้แปลง “เดือนนี้”, “สัปดาห์ก่อน” ฯลฯ เป็นช่วงวันที่ YYYY-MM-DD ตอนเรียกเครื่องมือ\n"
    )
    if qa:
        p += (
            "เรื่องเงินและใบเสร็จของร้าน: ต้องเรียกเครื่องมือเสมอ ห้ามคำนวณหรือเดาตัวเลขเอง "
            "พูดจำนวนเงินให้เป็นธรรมชาติเป็นบาท ถ้าไม่มีข้อมูลให้บอกตรง ๆ "
            "ถ้าเครื่องมือตอบว่าดูไม่ได้ ให้บอกเหตุผลสั้น ๆ ตามนั้น\n"
        )
    else:
        p += "ตอนนี้คุณดูข้อมูลบัญชีของร้านไม่ได้ ถ้ามีคนถามเรื่องเงินหรือใบเสร็จให้บอกตรง ๆ ว่าตอนนี้ดูไม่ได้\n"
    if language == "en":
        p += "Language: always speak English (keep Thai names and vendors as they are).\n"
    elif language == "auto":
        p += "ภาษา: ตอบด้วยภาษาเดียวกับคนที่พูดกับคุณ (ไทยหรืออังกฤษ)\n"
    else:
        p += "ภาษา: พูดภาษาไทย\n"
    if history:
        p += "\nบริบท: สิ่งที่คนในห้องเพิ่งคุยกัน (ใช้อ้างอิงเท่านั้น ไม่ต้องตอบ):\n" + history + "\n"
    return p


# ------------------------------------------------------------------ the live session
class LiveVoice:
    """One Live session per meeting. `connect(config)` returns an async context manager yielding the
    Live session (injected so tests run offline); `post(text)` writes to the meeting's text chat;
    `on_failed()` is called once when connecting is given up on."""

    def __init__(
        self,
        bot: SecretaryBot,
        session: MeetingSession,
        post: Callable[[str], Any],
        connect: Callable[[types.LiveConnectConfig], Any] | None = None,
        on_failed: Callable[[], None] | None = None,
    ):
        self.bot, self.session, self.post = bot, session, post
        self._connect = connect or self._open_default
        self.on_failed = on_failed
        self.mixer = Mixer()
        self._task: asyncio.Task | None = None
        self._bg: set[asyncio.Task] = set()
        self._client: Any = None
        self._live: Any = None  # the open Live session, None while (re)connecting
        self._closing = False
        self.handle: str | None = None  # latest session-resumption handle
        # sender state
        self._streaming = False  # audio has been sent and the stream has not been ended yet
        self._last_voice = 0.0
        self._latency_pending = False
        # receiver / output state
        self._source: LiveAudioSource | None = None
        self._playing_src: LiveAudioSource | None = None
        self._said: list[str] = []
        self._turn_audio = False
        self._awaiting_answer = False  # a tool call was answered: its spoken answer comes next
        self._last_audio_at = 0.0  # when the model last produced audio (monotonic)
        self._engaged = False

    # ---- lifecycle
    def start(self) -> asyncio.Task:
        self._task = asyncio.create_task(self.run())
        return self._task

    async def close(self) -> None:
        """Stop everything: the session task, background posts, playback. Safe to call twice."""
        self._closing = True
        task = self._task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for t in list(self._bg):
            t.cancel()
        if self._bg:
            await asyncio.gather(*self._bg, return_exceptions=True)
        await self._close_client()
        self.stop_playback()

    async def _close_client(self) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        try:
            await client.aio.aclose()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("live: could not close async client", exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                client.close()

    def feed(self, user_id: int, pcm: bytes) -> None:
        """One 20 ms frame (48 kHz stereo) from a human. Kept (the last few seconds) while the
        session is still connecting, so a greeting right after joining is not lost."""
        if not self._closing:
            self.mixer.feed(user_id, pcm)

    # ---- connecting
    def _open_default(self, config: types.LiveConnectConfig):  # pragma: no cover - network
        from google import genai

        s = self.bot.s
        if self._client is None:
            self._client = genai.Client(
                vertexai=True, project=s.gcp_project, location=s.live_location
            )
        return self._client.aio.live.connect(model=s.live_model, config=config)

    async def _qa_enabled(self) -> bool:
        return await asyncio.to_thread(self.bot.team.enabled, self.session.guild_id, "qa")

    async def build_config(self) -> types.LiveConnectConfig:
        s, sess = self.bot.s, self.session
        qa = await self._qa_enabled()
        fns = [set_minutes_destination, post_note, ask_consent, set_recording]
        if qa:
            fns = [*build_tools(LedgerTools(self.bot.store, sess.guild_id)), *fns]
        decls = [
            types.FunctionDeclaration.from_callable_with_api_option(
                callable=fn, api_option="ENTERPRISE"
            )
            for fn in fns
        ]
        tools = [types.Tool(function_declarations=decls)]
        history = ""
        if self.handle is None:  # a resumed session already remembers the conversation
            history = sess.transcript.history(s.now(), window_s=120, max_chars=1500)
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=build_instruction(s, sess.language, s.now(), qa=qa, history=history),
            tools=tools,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=s.live_voice)
                )
            ),
            input_audio_transcription={},
            output_audio_transcription={},
            realtime_input_config=build_realtime_input(s),
            # With proactive audio the model stays silent for chatter that is not addressed to it.
            proactivity=(
                types.ProactivityConfig(proactive_audio=True)
                if getattr(s, "live_proactive_audio", True)
                else None
            ),
            session_resumption=types.SessionResumptionConfig(handle=self.handle),
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()
            ),
        )

    async def run(self) -> None:
        """Connect, serve, and reconnect (with the resumption handle) until closed. Gives up after
        MAX_FAILURES consecutive failures and calls `on_failed`."""
        failures = 0
        while not self._closing:
            err: Exception | None = None
            t_conn: float | None = None
            try:
                config = await self.build_config()
                async with self._connect(config) as live:
                    t_conn = time.monotonic()
                    self._live = live
                    logger.info("live: connected%s", " (resumed)" if self.handle else "")
                    await self._serve(live)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                err = e
                logger.warning("live: session error: %s: %s", type(e).__name__, e)
            finally:
                self._live = None
                self._end_turn_audio()
            if self._closing:
                return
            lived = time.monotonic() - t_conn if t_conn is not None else 0.0
            if lived >= STABLE_S:
                failures = 0
                if err is None:
                    logger.info("live: reconnecting (server asked)")
                    continue
            failures += 1
            if failures >= MAX_FAILURES:
                logger.error("live: giving up after %d failed connects", failures)
                await self._close_client()
                if self.on_failed is not None:
                    self.on_failed()
                return
            delay = BACKOFF_S[min(failures - 1, len(BACKOFF_S) - 1)]
            logger.info("live: reconnecting in %.1fs (attempt %d)", delay, failures)
            await asyncio.sleep(delay)

    async def _serve(self, live: Any) -> None:
        self._streaming = False  # audio heard while connecting stays queued: it is sent first
        if self._engaged:
            self._spawn(self._send_state())
        tasks = [
            asyncio.create_task(self._sender(live)),
            asyncio.create_task(self._receiver(live)),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                t.result()  # re-raises the failure that ended the session
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # ---- listening / engaged state, forced turns
    def notify_state(self, engaged: bool) -> None:
        """Tell the model whether it is engaged in a conversation or only listening."""
        self._engaged = engaged
        self._spawn(self._send_state())

    async def _send_state(self) -> None:
        live = self._live
        if live is None or self._closing:
            return  # the current state is sent again when the session connects
        text = STATE_ENGAGED if self._engaged else STATE_LISTENING
        await live.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=text)]), turn_complete=False
        )

    def ensure_reply(self, since: float) -> None:
        """Someone said the name (the utterance ended at `since`, monotonic). If the model has not
        started answering `live_force_reply_seconds` after that, force a turn with a cue."""
        self._spawn(self._force_reply(since))

    async def _force_reply(self, since: float) -> None:
        wait = float(getattr(self.bot.s, "live_force_reply_seconds", 1.2))
        delay = since + wait - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        live = self._live
        if live is None or self._closing or self.session.ending:
            return
        if self._last_audio_at >= since or self._turn_audio:
            return  # the model is answering (or answered) already
        logger.info("live: name heard but no reply: forcing a turn")
        self._last_audio_at = time.monotonic()  # one cue per call
        await live.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=FORCE_CUE)]),
            turn_complete=True,
        )

    # ---- audio in
    def _idle_s(self) -> float:
        """Silence is sent long enough for the server to end the turn (silence_duration_ms) first."""
        ms = int(getattr(self.bot.s, "live_silence_ms", 0) or 0)
        return max(STREAM_IDLE_S, ms / 1000 + 0.8)

    async def _sender(self, live: Any) -> None:
        while True:
            await asyncio.sleep(SEND_INTERVAL_S)
            if self.session.ending:
                continue
            now = time.monotonic()
            chunk = self.mixer.pop()
            if chunk:
                self._last_voice, self._latency_pending, self._streaming = now, True, True
                await live.send_realtime_input(
                    audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={LIVE_IN_RATE}")
                )
            elif self._streaming:
                if now - self._last_voice < self._idle_s():
                    # Real silence after the voice, so the server-side VAD can end the turn.
                    silence = bytes(int(LIVE_IN_RATE * SEND_INTERVAL_S) * 2)
                    await live.send_realtime_input(
                        audio=types.Blob(data=silence, mime_type=f"audio/pcm;rate={LIVE_IN_RATE}")
                    )
                else:
                    await live.send_realtime_input(audio_stream_end=True)
                    self._streaming = False

    # ---- messages in
    async def _receiver(self, live: Any) -> None:
        empty = 0
        while True:
            seen = False
            async for msg in live.receive():  # ends after every turn / tool call: re-create it
                seen = True
                if await self._on_message(live, msg) == "reconnect":
                    return
            empty = 0 if seen else empty + 1
            if empty >= 3:
                raise ConnectionError("live session closed")
            await asyncio.sleep(0)

    async def _on_message(self, live: Any, msg: Any) -> str | None:
        upd = getattr(msg, "session_resumption_update", None)
        if upd is not None and getattr(upd, "new_handle", None):
            self.handle = upd.new_handle
        if getattr(msg, "go_away", None) is not None:
            logger.info("live: server go_away (time left %s)", msg.go_away.time_left)
            return "reconnect"
        if getattr(msg, "tool_call", None) is not None:
            await self._on_tool_call(live, msg.tool_call)
        sc = getattr(msg, "server_content", None)
        if sc is None:
            return None
        if sc.interrupted:
            self._interrupt()
        if sc.model_turn is not None:
            for part in sc.model_turn.parts or []:
                data = part.inline_data.data if part.inline_data is not None else None
                if data:
                    self._on_audio(data)
        ot = sc.output_transcription
        if ot is not None and ot.text:
            self._said.append(ot.text)
        if sc.turn_complete:
            self._turn_complete()
        return None

    # ---- tools
    def _scope_ok(self) -> str | None:
        """None when the tools may run, else the refusal to hand to the model. Executed in a thread
        (permission lookups may touch the guild store)."""
        bot, s = self.bot, self.session
        if not bot.team.enabled(s.guild_id, "qa"):
            return QA_OFF
        humans = [m for m in s.voice_channel.members if not getattr(m, "bot", False)]
        if not humans or not all(bot.is_exec_member(m) for m in humans):
            return NO_TOOLS_MIXED_ROOM
        return None

    async def _run_tool(self, name: str, args: dict) -> dict:
        if name in _MEETING_TOOLS:  # not account data: no executive check
            mgr, a = self.bot.meetings, args or {}
            if name == "set_minutes_destination":
                return await mgr.set_minutes_destination(self.session, str(a.get("channel", "")))
            if name == "post_note":
                return await mgr.post_note(self.session, str(a.get("text", "")))
            if name == "ask_consent":
                return await mgr.ask_consent(
                    self.session, str(a.get("text", "")), str(a.get("medium", "text"))
                )
            return mgr.set_recording(self.session, bool(a.get("recording", True)))

        def call() -> dict:
            refusal = self._scope_ok()
            if refusal:
                return {"error": refusal}
            # Everyone present is an executive at the point of access, so the whole guild ledger
            # is in scope. The check and query share this worker call; no await can widen the room.
            ledger = LedgerTools(self.bot.store, self.session.guild_id, forced_submitter_id=None)
            fn = {f.__name__: f for f in build_tools(ledger)}.get(name)
            if fn is None:
                return {"error": f"unknown tool {name}"}
            params = inspect.signature(fn).parameters
            clean = {k: v for k, v in (args or {}).items() if k in params and v is not None}
            try:
                result = fn(**clean)
            except Exception as e:
                logger.warning("live: tool %s failed: %s", name, e)
                return {"error": str(e)}
            return result if isinstance(result, dict) else {"result": result}

        return await asyncio.to_thread(call)

    async def _on_tool_call(self, live: Any, tool_call: Any) -> None:
        self._awaiting_answer = True  # the turn_complete that follows is not the end of the answer
        responses = []
        for fc in tool_call.function_calls or []:
            logger.info("live: tool call %s", fc.name)  # name only: arguments may hold PII
            result = await self._run_tool(fc.name, dict(fc.args or {}))
            responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=result))
        if responses:
            await live.send_tool_response(function_responses=responses)

    # ---- audio out
    def _on_audio(self, data: bytes) -> None:
        s = self.session
        if s.ending or not self.bot.s.voice_reply or not _connected(s.vc):
            return
        if not self._turn_audio:
            self._turn_audio = True
            if self._latency_pending:
                logger.info(
                    "live: user audio ended -> first audio %.1fs",
                    time.monotonic() - self._last_voice,
                )
                self._latency_pending = False
        self._awaiting_answer = False
        s.introduced = True  # Salid has spoken: the room knows it is there
        self._last_audio_at = time.monotonic()
        src = self._source
        if src is None or not src.feed(data):  # no source yet, or the last one is spent
            src = self._source = LiveAudioSource()
            src.feed(data)
        self._ensure_playing(src)

    def _ensure_playing(self, src: LiveAudioSource, retries: int = PLAY_RETRIES) -> None:
        s = self.session
        if src is not self._source or src.ended or s.ending or not _connected(s.vc):
            return
        if self._playing_src is not None:
            if self._playing_src is src and s.vc.is_playing():
                return
            busy = RuntimeError("previous voice player still winding down")
        elif s.vc.is_playing():
            busy = RuntimeError("voice client still busy")
        else:
            busy = None
        try:
            if busy is not None:
                raise busy
            s.vc.play(src, after=lambda err, src=src: self._after_threadsafe(src, err))
        except Exception as e:
            if retries > 0:  # e.g. the previous source is still winding down
                s.loop.call_later(0.05, self._ensure_playing, src, retries - 1)
            else:
                logger.warning("live: could not start playback: %s", e)
            return
        self._playing_src = src
        if not s.speaking:
            s.speaking, s.speaking_since = True, time.monotonic()

    def _after_threadsafe(self, src: LiveAudioSource, err: Exception | None) -> None:
        with contextlib.suppress(RuntimeError):  # loop already closed
            self.session.loop.call_soon_threadsafe(self._on_play_end, src, err)

    def _on_play_end(self, src: LiveAudioSource, err: Exception | None) -> None:
        if err:
            logger.warning("live: playback error: %s", err)
        if self._playing_src is src:  # a newer source may already have taken over
            self._mark_silent()
            self._playing_src = None
        if self._source is src:
            self._source = None

    def _mark_silent(self) -> None:
        s = self.session
        s.speaking = False
        s.speaking_until = time.monotonic() + SPEAK_COOLDOWN_S

    def _turn_complete(self) -> None:
        text = "".join(self._said).strip()
        self._said.clear()
        if text:
            self._spawn(self.post(f"🔊 {text}"))
        if self._awaiting_answer:
            return  # right after a tool call: the spoken answer is the next turn
        self._turn_audio = False
        if self._source is not None:
            self._source.finish()

    def _interrupt(self) -> None:
        """Barge-in: somebody talked over Salid. Drop queued audio and stop playing now."""
        logger.info("live: interrupted")
        self.stop_playback()
        self._said.clear()
        self._turn_audio = False
        self._awaiting_answer = False

    def stop_playback(self) -> None:
        src, self._source = self._source, None
        if src is not None:
            src.clear()
            src.finish()
        if self._playing_src is not None:
            _stop_playback(self.session.vc)
            self._playing_src = None
            self._mark_silent()

    def _end_turn_audio(self) -> None:
        """The connection ended: let whatever is queued play out, then stop."""
        if self._source is not None:
            self._source.finish()
        self._said.clear()
        self._turn_audio = self._awaiting_answer = False

    def _spawn(self, coro: Any) -> None:
        async def guarded() -> None:
            try:
                await coro
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("live: could not post to the text chat", exc_info=True)

        task = asyncio.create_task(guarded())
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)
