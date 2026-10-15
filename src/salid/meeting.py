"""Voice meetings: Salid joins a Discord voice channel, listens, transcribes who said what, answers
when called by name, and posts a Thai summary when the meeting ends.

Privacy: audio is never written to disk or any bucket. Per-speaker PCM lives in memory only until
a segment is transcribed, and the transcript stays in memory only (per guild, last 3 hours) and is
dropped when the meeting ends. Salid announces itself in the channel's text chat when it joins.

Receiving voice: `discord-ext-voice-recv` hands us transport-decrypted frames, which are still
DAVE (MLS end-to-end) ciphertext. discord.py (with `davey`) owns the MLS session, so we decrypt each
frame with `dave_session.decrypt(user_id, ...)` ourselves and decode Opus in our own try/except
(the extension's decoder thread dies on the first undecodable frame). Speaker identity is exact:
one Discord SSRC maps to one member, so no diarization is needed.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import re
import time
import wave
from array import array
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import discord
from discord.ext import voice_recv
from google.genai import types
from pydantic import BaseModel

from meeting_scribe.audio import to_flac

from .llm import generate
from .naming import _WAKE_RE

if TYPE_CHECKING:
    from .bot import SecretaryBot

logger = logging.getLogger(__name__)

# ---- audio format coming out of the Opus decoder (discord.opus.Decoder)
PCM_RATE = 48000
PCM_CHANNELS = 2
PCM_BYTES_PER_SECOND = PCM_RATE * PCM_CHANNELS * 2

# ---- segmentation / limits
SILENCE_S = 1.2  # a pause this long ends a segment
MAX_SEGMENT_S = 20.0
MIN_SEGMENT_S = 0.6  # shorter segments are dropped (coughs, clicks)
PAD_GAP_S = 0.15  # shorter gaps inside a segment are filled with silence to keep timing
MIN_RMS = 80  # int16 RMS below this is treated as silence/noise and not sent to the model
TRANSCRIPT_KEEP_S = 3 * 3600
TRANSCRIPT_MAX_LINES = 6000
MEETING_MAX_S = 3 * 3600
EMPTY_GRACE_S = 20.0  # everyone else left: end after this long
CONTEXT_WINDOW_S = 5 * 60
CONTEXT_MAX_CHARS = 6000
SUMMARY_MAX_CHARS = 150_000
PENDING_WAKE_S = 15.0
TRANSCRIBE_WORKERS = 2
QUEUE_MAX = 60
TICK_S = 0.5

ANNOUNCE_MESSAGE = (
    "สลิดเข้ามาฟังและจดบันทึกการประชุมครับ 🎙️ "
    "(เก็บเป็นข้อความในหน่วยความจำเท่านั้น ไม่บันทึกเสียง) "
    "เรียก “สลิด” แล้วถามได้เลย หรือพิมพ์ /ออกประชุม เพื่อสรุปและให้ผมออก"
)
NOT_IN_VOICE_MESSAGE = "ต้องเข้าห้องเสียง (Voice Channel) ก่อน แล้วค่อยเรียกสลิดเข้าประชุมนะครับ"
ALREADY_MESSAGE = "สลิดกำลังอยู่ในการประชุมของเซิร์ฟเวอร์นี้แล้วครับ ({channel}) พิมพ์ /ออกประชุม เพื่อจบก่อนนะครับ"
NO_MEETING_MESSAGE = "ตอนนี้สลิดไม่ได้อยู่ในการประชุมครับ"
LEAVE_DENIED_MESSAGE = "ต้องอยู่ในห้องประชุมเดียวกัน (หรือเป็นผู้ดูแลเซิร์ฟเวอร์) ถึงจะสั่งให้สลิดออกได้ครับ"
MEETINGS_OFF_MESSAGE = "ตอนนี้ปิดฟังการประชุมอยู่ครับ เปิดได้ที่หน้าศูนย์บัญชาการของสลิด"
NO_PERMISSION_MESSAGE = "สลิดไม่มีสิทธิ์เชื่อมต่อ (Connect) ห้องเสียงนี้ครับ รบกวนผู้ดูแลเปิดสิทธิ์ให้ก่อน"
CONNECT_FAILED_MESSAGE = "เข้าห้องเสียงไม่สำเร็จครับ ลองใหม่อีกครั้งภายหลังนะครับ"
JOINED_MESSAGE = "สลิดเข้าห้อง {channel} แล้วครับ"
LEFT_MESSAGE = "สลิดออกจากการประชุมแล้วครับ กำลังสรุปให้ในแชตของห้องเสียง"
WAKE_EMPTY_MESSAGE = "ครับ? ว่ามาได้เลยครับ"
NO_TRANSCRIPT_MESSAGE = "จบการประชุมแล้วครับ แต่ยังไม่ได้ยินเนื้อหาให้สรุปเลย 🙂"
SUMMARY_FAILED_MESSAGE = "จบการประชุมแล้วครับ แต่สรุปไม่สำเร็จ ขออภัยด้วยนะครับ"
END_REASONS = {
    "command": "ผู้ใช้สั่งให้ออกจากการประชุม",
    "empty": "ทุกคนออกจากห้องแล้ว",
    "timeout": "ประชุมครบ 3 ชั่วโมง",
    "disconnected": "สลิดหลุดจากห้องเสียง",
    "shutdown": "สลิดกำลังปิดระบบ",
}

# ------------------------------------------------------------------ text triggers
_JOIN_RE = re.compile(r"^(เข้า\s?ประชุม|เข้ามา\s?ฟัง|มา\s?ประชุม|join)", re.I)
_LEAVE_RE = re.compile(r"^(ออก\s?ประชุม|เลิก\s?ประชุม|จบ\s?ประชุม|leave)", re.I)


def parse_meeting_command(text: str) -> str | None:
    """ "เข้าประชุม" / "join" -> "join", "ออกประชุม" / "leave" -> "leave", anything else None.
    Only the start of the (mention-stripped) message counts, so a long question that merely
    contains the words is not a command."""
    t = (text or "").strip().lstrip("/").strip()
    if not t or len(t) > 40:
        return None
    if _JOIN_RE.match(t):
        return "join"
    if _LEAVE_RE.match(t):
        return "leave"
    return None


# ------------------------------------------------------------------ wake word
# The wake word regex lives in naming.py (shared with the text-channel name detection).
_LEAD_FILLER = re.compile(r"^[\s,.:;!?\-–—…ๆ]*(?:(?:ครับ|ค่ะ|คะ|นะ|จ๊ะ|จ้ะ|เอ่อ|อ่า|ว่า)\s*)*")
_TRAIL_FILLER = re.compile(r"[\s,.:;!?\-–—…]+$")


def _clean_question(s: str) -> str:
    s = _LEAD_FILLER.sub("", s)
    return _TRAIL_FILLER.sub("", s).strip()


def parse_wake(text: str) -> str | None:
    """None when Salid was not called. Otherwise the question: the words after the wake word, or,
    when it ends the sentence ("ช่วยสรุปหน่อยสลิด"), the words before it. "" = called with no
    question yet (the caller may take the speaker's next segment as the question)."""
    m = _WAKE_RE.search(text or "")
    if m is None:
        return None
    after = _clean_question(text[m.end() :])
    if len(after) >= 2:
        return after
    before = _clean_question(text[: m.start()])
    return before if len(before) >= 2 else ""


# ------------------------------------------------------------------ transcript
@dataclass
class Line:
    at: datetime
    speaker: str
    text: str


class Transcript:
    """Rolling in-memory transcript (memory only; bounded by age and line count)."""

    def __init__(self, keep_s: float = TRANSCRIPT_KEEP_S, max_lines: int = TRANSCRIPT_MAX_LINES):
        self.keep_s, self.max_lines = keep_s, max_lines
        self.lines: deque[Line] = deque()

    def add(self, at: datetime, speaker: str, text: str) -> None:
        self.lines.append(Line(at, speaker, text))
        while len(self.lines) > self.max_lines:
            self.lines.popleft()
        while self.lines and (at - self.lines[0].at).total_seconds() > self.keep_s:
            self.lines.popleft()

    def __len__(self) -> int:
        return len(self.lines)

    def recent(self, now: datetime, window_s: float = CONTEXT_WINDOW_S) -> list[Line]:
        return [ln for ln in self.lines if 0 <= (now - ln.at).total_seconds() <= window_s]

    def history(
        self, now: datetime, window_s: float = CONTEXT_WINDOW_S, max_chars: int = CONTEXT_MAX_CHARS
    ) -> str:
        """The last `window_s` seconds as "ชื่อ: ข้อความ" lines, oldest first, newest kept when
        trimmed to `max_chars`."""
        out: list[str] = []
        size = 0
        for ln in reversed(self.recent(now, window_s)):
            row = f"{ln.speaker}: {ln.text}"
            size += len(row) + 1
            if size > max_chars:
                break
            out.append(row)
        return "\n".join(reversed(out))

    def full_text(self, max_chars: int = SUMMARY_MAX_CHARS) -> str:
        rows = [f"[{ln.at:%H:%M}] {ln.speaker}: {ln.text}" for ln in self.lines]
        text = "\n".join(rows)
        return text[-max_chars:] if len(text) > max_chars else text


# ------------------------------------------------------------------ segmenter
@dataclass
class Segment:
    user_id: int
    pcm: bytes  # 48 kHz stereo s16le
    started: float  # time.monotonic() when the first frame arrived
    duration_s: float


@dataclass
class _Buf:
    chunks: list[bytes] = field(default_factory=list)
    size: int = 0
    started: float = 0.0
    last: float = 0.0
    last_len: int = 0  # bytes of the most recent packet


class Segmenter:
    """Per-speaker buffers cut on silence (no frames for `silence_s`) or at `max_s`; segments
    shorter than `min_s` are dropped. Time is passed in, so it is fully testable. Discord stops
    sending frames while someone is silent, so silence is detected by the gap between frames
    (`feed`) or by `tick` when nothing arrives at all."""

    def __init__(
        self,
        silence_s: float = SILENCE_S,
        max_s: float = MAX_SEGMENT_S,
        min_s: float = MIN_SEGMENT_S,
        pad_gap_s: float = PAD_GAP_S,
    ):
        self.silence_s, self.max_s, self.min_s, self.pad_gap_s = silence_s, max_s, min_s, pad_gap_s
        self.bufs: dict[int, _Buf] = {}

    def _close(self, uid: int) -> Segment | None:
        buf = self.bufs.pop(uid, None)
        if buf is None or not buf.size:
            return None
        dur = buf.size / PCM_BYTES_PER_SECOND
        if dur < self.min_s:
            return None
        return Segment(uid, b"".join(buf.chunks), buf.started, dur)

    def feed(self, uid: int, pcm: bytes, now: float) -> list[Segment]:
        done: list[Segment] = []
        buf = self.bufs.get(uid)
        if buf is not None and now - buf.last > self.silence_s:
            if seg := self._close(uid):
                done.append(seg)
            buf = None
        if buf is None:
            buf = self.bufs[uid] = _Buf(started=now)
        else:
            gap = now - buf.last - buf.last_len / PCM_BYTES_PER_SECOND
            if gap > self.pad_gap_s:  # keep real timing across short pauses
                frames = int(gap * PCM_RATE)
                filler = bytes(frames * PCM_CHANNELS * 2)
                buf.chunks.append(filler)
                buf.size += len(filler)
        buf.chunks.append(pcm)
        buf.size += len(pcm)
        buf.last, buf.last_len = now, len(pcm)
        if buf.size / PCM_BYTES_PER_SECOND >= self.max_s and (seg := self._close(uid)):
            done.append(seg)
        return done

    def tick(self, now: float) -> list[Segment]:
        """Close segments whose speaker has been quiet for `silence_s`."""
        done = []
        for uid in [u for u, b in self.bufs.items() if now - b.last > self.silence_s]:
            if seg := self._close(uid):
                done.append(seg)
        return done

    def flush_all(self) -> list[Segment]:
        return [s for uid in list(self.bufs) if (s := self._close(uid))]


def rms(pcm: bytes) -> float:
    """Cheap loudness estimate (every 16th int16 sample)."""
    n = len(pcm) // 2
    if not n:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[: n * 2])
    sub = samples[::16] or samples
    return (sum(x * x for x in sub) / len(sub)) ** 0.5


# ------------------------------------------------------------------ transcription / summary
class _Heard(BaseModel):
    text: str


TRANSCRIBE_PROMPT = (
    "ถอดเสียงพูดภาษาไทยในคลิปนี้ (เสียงของคนคนเดียวในการประชุม) ตามที่พูดจริงทุกคำ "
    "ภาษาอังกฤษที่พูดปนให้คงเป็นภาษาอังกฤษ ไม่ต้องแปล ไม่ต้องใส่ชื่อผู้พูด\n"
    "ผู้ช่วยในประชุมชื่อ “สลิด” (Salid) ถ้าได้ยินคำนี้ให้เขียนว่า สลิด\n"
    "ถ้าไม่มีเสียงพูดชัดเจน (เงียบ เสียงรบกวน เสียงหายใจ) ให้ตอบ text เป็นสตริงว่าง ห้ามแต่งข้อความขึ้นเอง "
    'ตอบเป็น JSON {"text": "..."}'
)
SUMMARY_PROMPT = """คุณคือ “สลิด” เลขาประชุมของทีม จากบันทึกการประชุมด้านล่าง (รูปแบบ [เวลา] ชื่อ: ข้อความ ถอดจากเสียงอัตโนมัติ อาจมีคำผิด) \
เขียนสรุปการประชุมเป็นภาษาไทยที่กระชับ ใช้เฉพาะข้อมูลที่อยู่ในบันทึก ห้ามแต่งเพิ่ม ใช้หัวข้อตามนี้:
**สรุปการประชุม** (2-4 บรรทัด)
**มติ/ข้อตกลง** (รายการ; ถ้าไม่มีเขียนว่า ไม่มี)
**งานที่ต้องทำ** (รายการรูปแบบ "ใคร: ทำอะไร (กำหนดเมื่อไรถ้ามี)"; ระบุเจ้าของงานเป็นชื่อผู้พูดตามบันทึก ถ้าไม่ชัดเจนเขียนว่า ยังไม่ระบุผู้รับผิดชอบ)
**เรื่องที่ยังค้างอยู่/คำถามเปิด** (รายการ; ถ้าไม่มีเขียนว่า ไม่มี)
อย่าใส่ข้อมูลส่วนตัวที่ไม่เกี่ยวกับงาน

บันทึกการประชุม:
{transcript}"""


def pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(PCM_CHANNELS)
        w.setsampwidth(2)
        w.setframerate(PCM_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def transcribe_pcm(client: Any, model: str, pcm: bytes) -> str:
    """Blocking: 48 kHz stereo PCM -> 16 kHz mono FLAC (ffmpeg, in memory) -> Gemini on Vertex.
    Returns the Thai text, "" when there was no speech."""
    flac = to_flac(pcm_to_wav(pcm))
    config = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=_Heard, temperature=0.0
    )
    resp = generate(
        client,
        model,
        [types.Part.from_bytes(data=flac, mime_type="audio/flac"), TRANSCRIBE_PROMPT],
        config,
    )
    raw = (getattr(resp, "text", None) or "").strip()
    try:
        text = str(json.loads(raw).get("text", ""))
    except (ValueError, AttributeError):
        text = raw
    text = text.strip()
    return "" if re.fullmatch(r"[\s\W_]*", text) else text


def summarize_transcript(client: Any, model: str, transcript: str) -> str:
    """Blocking: Thai meeting summary (decisions, action items with owners, open questions)."""
    resp = generate(
        client,
        model,
        [SUMMARY_PROMPT.format(transcript=transcript)],
        types.GenerateContentConfig(temperature=0.2),
    )
    return (getattr(resp, "text", None) or "").strip()


# ------------------------------------------------------------------ voice receive sink
class MeetingSink(voice_recv.AudioSink):
    """Runs in the extension's router thread. Asks for raw Opus (the extension then never decodes),
    removes the DAVE layer, decodes with a per-speaker decoder and hands PCM to the event loop."""

    def __init__(self, session: MeetingSession):
        super().__init__()
        self.session = session
        self._decoders: dict[int, Any] = {}
        self.dave_dropped = 0

    def wants_opus(self) -> bool:
        return True

    def cleanup(self) -> None:
        self._decoders.clear()

    def _dave_decrypt(self, user_id: int, opus: bytes) -> bytes | None:
        conn = getattr(self.session.vc, "_connection", None)
        dave = getattr(conn, "dave_session", None)
        if dave is None or not getattr(conn, "dave_protocol_version", 0) or not dave.ready:
            return opus  # no end-to-end layer on this call
        try:
            import davey

            return dave.decrypt(user_id, davey.MediaType.audio, opus)
        except Exception as e:
            if "nencrypted" in str(e):  # passthrough frame: it is plain Opus
                return opus
            self.dave_dropped += 1
            if self.dave_dropped in (1, 50, 500):
                logger.warning("DAVE decrypt failed (%d so far): %s", self.dave_dropped, e)
            return None

    def write(self, user, data) -> None:
        opus = data.opus
        if user is None or not opus or opus == b"\xf8\xff\xfe":  # unknown speaker / lost / silence
            return
        plain = self._dave_decrypt(user.id, opus)
        if not plain:
            return
        dec = self._decoders.get(user.id)
        if dec is None:
            from discord.opus import Decoder

            dec = self._decoders[user.id] = Decoder()
        try:
            pcm = dec.decode(plain, fec=False)
        except Exception:
            logger.debug("opus decode failed for %s", user.id, exc_info=True)
            return
        self.session.push_threadsafe(user.id, pcm)


# ------------------------------------------------------------------ meeting session
@dataclass
class MeetingSession:
    guild_id: int
    voice_channel: Any
    text_channel: Any  # the voice channel's built-in text chat
    fallback_channel: Any | None
    vc: Any
    loop: asyncio.AbstractEventLoop
    started: float
    transcript: Transcript = field(default_factory=Transcript)
    segmenter: Segmenter = field(default_factory=Segmenter)
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=QUEUE_MAX))
    pending_wake: dict[int, float] = field(default_factory=dict)
    tasks: list[asyncio.Task] = field(default_factory=list)
    answer_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    empty_since: float | None = None
    ending: bool = False
    dropped: int = 0

    def push_threadsafe(self, user_id: int, pcm: bytes) -> None:
        with contextlib.suppress(RuntimeError):  # loop already closed on shutdown
            self.loop.call_soon_threadsafe(self.on_pcm, user_id, pcm, time.monotonic())

    def on_pcm(self, user_id: int, pcm: bytes, now: float) -> None:
        if not self.ending:
            self.enqueue(self.segmenter.feed(user_id, pcm, now))

    def enqueue(self, segments: list[Segment]) -> None:
        for seg in segments:
            if rms(seg.pcm) < MIN_RMS:
                continue
            try:
                self.queue.put_nowait(seg)
            except asyncio.QueueFull:
                self.dropped += 1
                logger.warning("transcription backlog full, dropped a segment")


class MeetingManager:
    """One meeting per guild. `bot` supplies settings, the Gemini client, permissions and Q&A."""

    def __init__(self, bot: SecretaryBot):
        self.bot = bot
        self.sessions: dict[int, MeetingSession] = {}
        self._joining: set[int] = set()  # guilds with a join in flight

    # ---- helpers
    def session_for_channel(self, channel_id: int | None) -> MeetingSession | None:
        return next((s for s in self.sessions.values() if s.text_channel.id == channel_id), None)

    def context_for(self, channel_id: int | None) -> str:
        """Recent meeting transcript for a message sent in the meeting's text chat ("" if none)."""
        s = self.session_for_channel(channel_id)
        if s is None:
            return ""
        return s.transcript.history(self.bot.s.now())

    async def _connect(self, channel):
        return await channel.connect(cls=voice_recv.VoiceRecvClient)

    async def _post(self, session: MeetingSession, text: str) -> None:
        for ch in (session.text_channel, session.fallback_channel):
            if ch is None:
                continue
            try:
                await ch.send(text)
                return
            except discord.HTTPException:
                logger.warning("could not post to channel %s", getattr(ch, "id", "?"))

    # ---- commands (return the Thai reply for the invoker)
    async def join(self, member, fallback_channel=None) -> str:
        guild = member.guild
        if not self.bot.team.enabled(guild.id, "voice_meetings"):
            return MEETINGS_OFF_MESSAGE
        existing = self.sessions.get(guild.id)
        if existing is not None:
            return ALREADY_MESSAGE.format(channel=existing.voice_channel.name)
        if guild.id in self._joining:
            return ALREADY_MESSAGE.format(channel="กำลังเข้า")
        voice = getattr(member, "voice", None)
        channel = getattr(voice, "channel", None)
        if channel is None:
            return NOT_IN_VOICE_MESSAGE
        if not channel.permissions_for(guild.me).connect:
            return NO_PERMISSION_MESSAGE
        # Reserve the guild slot before the first await so two commands cannot both join.
        self._joining.add(guild.id)
        try:
            vc = await self._connect(channel)
        except Exception:
            logger.exception("could not join voice channel %s", channel.id)
            return CONNECT_FAILED_MESSAGE
        finally:
            self._joining.discard(guild.id)
        loop = asyncio.get_running_loop()
        session = MeetingSession(
            guild_id=guild.id, voice_channel=channel, text_channel=channel,
            fallback_channel=fallback_channel, vc=vc, loop=loop, started=time.monotonic(),
        )  # fmt: skip
        self.sessions[guild.id] = session
        try:
            vc.listen(MeetingSink(session))
        except Exception:
            logger.exception("could not start listening")
            self.sessions.pop(guild.id, None)
            with contextlib.suppress(Exception):
                await vc.disconnect()
            return CONNECT_FAILED_MESSAGE
        session.tasks = [
            asyncio.create_task(self._worker(session)) for _ in range(TRANSCRIBE_WORKERS)
        ]
        session.tasks.append(asyncio.create_task(self._ticker(session)))
        await self._post(session, ANNOUNCE_MESSAGE)
        logger.info("meeting started in guild %s channel %s", guild.id, channel.id)
        return JOINED_MESSAGE.format(channel=channel.name)

    async def leave(self, member) -> str:
        session = self.sessions.get(member.guild.id)
        if session is None:
            return NO_MEETING_MESSAGE
        in_channel = (
            getattr(getattr(member, "voice", None), "channel", None) == session.voice_channel
        )
        from .bot import can_manage_guild

        if not in_channel and not can_manage_guild(member):
            return LEAVE_DENIED_MESSAGE
        asyncio.create_task(self.end(member.guild.id, "command"))
        return LEFT_MESSAGE

    # ---- background work
    async def _ticker(self, s: MeetingSession) -> None:
        try:
            while not s.ending:
                await asyncio.sleep(TICK_S)
                now = time.monotonic()
                s.enqueue(s.segmenter.tick(now))
                if now - s.started > MEETING_MAX_S:
                    asyncio.create_task(self.end(s.guild_id, "timeout"))
                    return
                humans = [m for m in s.voice_channel.members if not m.bot]
                if humans:
                    s.empty_since = None
                elif s.empty_since is None:
                    s.empty_since = now
                elif now - s.empty_since > EMPTY_GRACE_S:
                    asyncio.create_task(self.end(s.guild_id, "empty"))
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("meeting ticker crashed")

    async def _worker(self, s: MeetingSession) -> None:
        while True:
            seg = await s.queue.get()
            try:
                await self.handle_segment(s, seg)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("segment handling failed")
            finally:
                s.queue.task_done()

    async def handle_segment(self, s: MeetingSession, seg: Segment) -> None:
        member = s.voice_channel.guild.get_member(seg.user_id)
        name = member.display_name if member else f"ผู้ร่วมประชุม {seg.user_id % 1000}"
        text = await asyncio.to_thread(
            transcribe_pcm, self.bot.llm, self.bot.s.gemini_model, seg.pcm
        )
        if not text:
            return
        s.transcript.add(self.bot.s.now(), name, text)
        logger.info("meeting %s: %s said %d chars", s.guild_id, seg.user_id, len(text))
        question = parse_wake(text)
        now = time.monotonic()
        if question is None:
            if now - s.pending_wake.pop(seg.user_id, -1e9) < PENDING_WAKE_S:
                question = text  # "สลิด ..." then a pause, then the actual question
            else:
                return
        if not question:
            s.pending_wake[seg.user_id] = now
            if member is not None:
                await self._post(s, f"{member.mention} {WAKE_EMPTY_MESSAGE}")
            return
        s.pending_wake.pop(seg.user_id, None)
        await self.answer(s, member, name, question)

    async def answer(self, s: MeetingSession, member, name: str, question: str) -> None:
        from .bot import chunks
        from .qa import ask

        bot = self.bot
        if member is None or not bot.team.enabled(s.guild_id, "qa"):
            return
        async with s.answer_lock:
            history = s.transcript.history(bot.s.now())
            ans = await asyncio.to_thread(
                ask,
                bot.s,
                bot.store,
                question,
                guild_id=s.guild_id,
                is_exec=bot.is_exec_member(member),
                user_id=member.id,
                history=history,
            )
            for i, part in enumerate(chunks(ans.text, 1800)):
                await self._post(s, f"{member.mention} {part}" if i == 0 else part)

    # ---- ending
    async def end(self, guild_id: int, reason: str = "command") -> None:
        s = self.sessions.get(guild_id)
        if s is None or s.ending:
            return
        s.ending = True
        self.sessions.pop(guild_id, None)
        logger.info("meeting ending in guild %s (%s)", guild_id, reason)
        with contextlib.suppress(Exception):
            s.vc.stop_listening()
        with contextlib.suppress(Exception):
            await s.vc.disconnect()
        s.enqueue(s.segmenter.flush_all())
        with contextlib.suppress(asyncio.TimeoutError):  # let pending segments finish
            await asyncio.wait_for(s.queue.join(), timeout=90)
        for t in s.tasks:
            if t is not asyncio.current_task():
                t.cancel()
        await self._post_summary(s, reason)

    async def _post_summary(self, s: MeetingSession, reason: str) -> None:
        from .bot import chunks

        if not len(s.transcript):
            await self._post(s, NO_TRANSCRIPT_MESSAGE)
            return
        try:
            body = await asyncio.to_thread(
                summarize_transcript,
                self.bot.llm,
                self.bot.s.gemini_model,
                s.transcript.full_text(),
            )
        except Exception:
            logger.exception("meeting summary failed")
            body = ""
        if not body:
            await self._post(s, SUMMARY_FAILED_MESSAGE)
            return
        header = f"📝 สรุปการประชุม ({END_REASONS.get(reason, reason)})\n"
        for part in chunks(header + body, 1900):
            await self._post(s, part)
        s.transcript.lines.clear()  # transcript is never persisted

    async def shutdown(self) -> None:
        for gid in list(self.sessions):
            await self.end(gid, "shutdown")
