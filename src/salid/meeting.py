"""Voice meetings: Salid joins a Discord voice channel, listens, transcribes who said what, answers
when called by name, and posts a summary (in the text channel's language) when the meeting ends.

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
import random
import re
import time
import wave
from array import array
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import discord
from discord.ext import voice_recv
from google.genai import types
from pydantic import BaseModel

from meeting_scribe.audio import to_flac

from .coo_store import MINUTES
from .gate import GateLine, notes_from_transcript
from .i18n import resolve_lang, t
from .llm import generate
from .minutes import (
    can_see,
    can_see_and_post,
    parse_minutes_destination,
    resolve_channel,
)
from .naming import find_call
from .permissions import channel_missing, permission_message
from .speech import pcm_duration_s, pcm_source, say, spoken_text
from .voice_pool import Claim, VoiceBody, VoicePool, pick_victim, room_score

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
MIN_SEGMENT_S = 0.4  # shorter segments are dropped (coughs, clicks); 0.6 lost short greetings
PAD_GAP_S = 0.15  # shorter gaps inside a segment are filled with silence to keep timing
MIN_RMS = 80  # int16 RMS below this is treated as silence/noise and not sent to the model
TRANSCRIPT_KEEP_S = 3 * 3600
TRANSCRIPT_MAX_LINES = 6000
MEETING_MAX_S = 6 * 3600
MAX_WARN_S = 600.0  # warn in the text chat this long before the cap
RECONNECT_DELAYS = (2.0, 5.0, 10.0)
EMPTY_GRACE_S = 30.0  # no humans left: end after this long (settings override)
CONTEXT_WINDOW_S = 5 * 60
CONTEXT_MAX_CHARS = 6000
SUMMARY_MAX_CHARS = 150_000
PENDING_WAKE_S = 15.0
TRANSCRIBE_WORKERS = 2
QUEUE_MAX = 60
NOTE_MIN_GAP_S = 5.0
MAX_NOTES = 40
TICK_S = 0.5
SPEAK_COOLDOWN_S = 1.5  # after Salid speaks, ignore wake words this long (room echo)
SPEAK_TIMEOUT_PAD_S = 10.0
# Said right after the wake question is heard, so the room knows Salid is on it (pre-synthesized).
ACK_TEXT = {"th": "ได้เลย แป๊บนึงนะ", "en": "Sure, one sec!"}

END_REASONS = ("command", "declined", "empty", "timeout", "disconnected", "shutdown", "preempted")
BLOCK_AFTER_FAILURE_S = 300.0  # no auto-rejoin this soon after a timeout / lost connection

# ------------------------------------------------------------------ text triggers
_JOIN_RE = re.compile(
    r"^(?:เข้า(?:มา)?\s?(?:ประชุม|ห้อง(?:เสียง)?|มา|หน่อย|ฟัง|จด)|เข้ามา$|เข้ามา\s?(?:ฟัง|จด)"
    r"|(?:เข้า)?มา\s?(?:ประชุม|ฟัง|จด|หน่อย|ด้วย)|มาจด(?:ประชุม)?"
    r"|join(?:\s+(?:us|me|the\s+(?:meeting|call|voice))?)?\b|come\s+(?:in|join|to\s+the\s+meeting)\b"
    r"|come\s+to\s+(?:the\s+)?meeting\b|hop\s+in\b|jump\s+in\b)",
    re.I,
)
_LEAVE_RE = re.compile(
    r"^(?:ออก\s?ประชุม|เลิก\s?ประชุม|จบ\s?ประชุม|ออกไป(?:ก่อน)?|ออกจาก(?:ห้อง|ประชุม)|ไปได้แล้ว"
    r"|leave(?:\s+(?:us|the\s+(?:meeting|call|room)))?\b|you\s+can\s+go\b"
    r"|(?:end|stop)\s+(?:the\s+)?meeting\b)",
    re.I,
)
_POLITE_PREFIX = re.compile(
    r"^(?:สลิด|salid|ช่วย|รบกวน|ขอ|กรุณา|ให้|please|pls|hey|hi|can\s+you|could\s+you)[\s,]*", re.I
)


def parse_meeting_command(text: str) -> str | None:
    """ "เข้าประชุม" / "เข้ามาประชุมหน่อยครับ" / "join us" -> "join", "ออกประชุม" / "ไปได้แล้ว" /
    "leave" -> "leave", anything else None. Only the start of the (mention-stripped, politeness-
    and name-stripped) message counts, so a long question that merely contains the words is not
    a command."""
    s = (text or "").strip().strip(" \t\n/,.:;!-–—…")
    if not s or len(s) > 60:
        return None
    while (m := _POLITE_PREFIX.match(s)) and m.end() > 0:
        s = s[m.end() :]
    if _JOIN_RE.match(s):
        return "join"
    if _LEAVE_RE.match(s):
        return "leave"
    return None


# ------------------------------------------------------------------ wake word
# The wake word regex lives in naming.py (shared with the text-channel name detection).
_LEAD_FILLER = re.compile(r"^[\s,.:;!?\-–—…ๆ]*(?:(?:ครับ|ค่ะ|คะ|นะ|จ๊ะ|จ้ะ|เอ่อ|อ่า|ว่า)\s*)*")
_TRAIL_FILLER = re.compile(r"[\s,.:;!?\-–—…]+$")


_GREETINGS = {"hey", "hi", "hello", "yo", "ok", "okay", "hey there", "hi there", "ว่าไง", "หวัดดี"}


def _clean_question(s: str) -> str:
    s = _LEAD_FILLER.sub("", s)
    return _TRAIL_FILLER.sub("", s).strip()


def parse_wake(text: str) -> str | None:
    """None when Salid was not called. Otherwise the question: the words after the wake word, or,
    when it ends the sentence ("ช่วยสรุปหน่อยสลิด"), the words before it. "" = called with no
    question yet (the caller may take the speaker's next segment as the question)."""
    span = find_call(text or "")
    if span is None:
        return None
    after = _clean_question(text[span[1] :])
    if len(after) >= 2:
        return after
    before = _clean_question(text[: span[0]])
    if before.casefold() in _GREETINGS:  # "Hey Salid": a greeting, not a question
        return ""
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
    "ถอดเสียงพูดภาษาไทยหรือภาษาอังกฤษในคลิปนี้ (เสียงของคนคนเดียวในการประชุม) ตามที่พูดจริงทุกคำ "
    "ภาษาอังกฤษที่พูดให้คงเป็นภาษาอังกฤษ ไม่ต้องแปล ไม่ต้องใส่ชื่อผู้พูด\n"
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


SUMMARY_PROMPT_EN = """You are “Salid”, the team's meeting secretary. Below is a meeting transcript (format [time] name: text, \
auto-transcribed so it may contain errors). Write a concise summary of the meeting in English. Use only \
what is in the transcript and do not invent anything. Use these headings:
**Meeting summary** (2-4 lines)
**Decisions/agreements** (list; write None if there are none)
**Action items** (list as "who: what (deadline if any)"; use the speaker names from the transcript as owners, \
write Owner not specified if unclear)
**Open issues/questions** (list; write None if there are none)
Do not include personal information unrelated to work.

Meeting transcript:
{transcript}"""
SUMMARY_PROMPTS = {"th": SUMMARY_PROMPT, "en": SUMMARY_PROMPT_EN}


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


def summarize_transcript(client: Any, model: str, transcript: str, language: str = "th") -> str:
    """Blocking: meeting summary (decisions, action items with owners, open questions) in
    `language` (th, en, or auto = the language the transcript is mostly in)."""
    prompt = SUMMARY_PROMPTS[resolve_lang(language, transcript)]
    resp = generate(
        client,
        model,
        [prompt.format(transcript=transcript)],
        types.GenerateContentConfig(temperature=0.2),
    )
    return (getattr(resp, "text", None) or "").strip()


def _connected(vc: Any) -> bool:
    try:
        return bool(vc.is_connected())
    except Exception:
        return False


def _stop_playback(vc: Any) -> None:
    with contextlib.suppress(Exception):
        if vc.is_playing():
            vc.stop()


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
        if getattr(user, "bot", False):
            self.session.push_threadsafe(user.id, pcm, is_bot=True)
        else:
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
    speech_pending: bool = False  # one shorten/TTS/playback pipeline at a time
    speaking: bool = False  # Salid is playing a spoken reply
    speech_ready: bool = (
        False  # the answer's audio is synthesized: a late acknowledgment is pointless
    )
    play_lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # one clip at a time, in order
    speaking_since: float = 0.0
    speaking_until: float = 0.0  # end of the last spoken reply + cooldown (monotonic)
    language: str = "th"  # th / en / auto: the text channel's setting when the meeting started
    fallback_lang: str = "th"  # th / en: what fixed messages use when `language` is auto
    empty_grace: float = EMPTY_GRACE_S  # no humans in the channel: end after this long
    first_grace: float | None = None  # auto-join only: shorter wait while nobody has come yet
    had_humans: bool = False
    max_warned: bool = False
    reconnecting: bool = False
    started_at: datetime | None = None  # wall clock (UTC) start, persisted for restarts
    starter: str = ""
    on_end: Any = None  # async callable(session), run after the summary was posted
    meeting_id: str | None = None  # the scheduled meeting this session takes notes for
    summary_text: str = ""
    summary_url: str = ""  # jump link of the first summary message
    # "live": a Gemini Live session (live_voice.LiveVoice) talks; wake words are not answered by
    # the transcription pipeline. Falls back to "classic" when the Live session cannot connect.
    voice_mode: str = "classic"
    live: Any = None
    body: Any = None  # the VoiceBody (main bot or helper) holding this session's voice connection
    origin: str = "command"  # command (a person asked) | scheduled | auto (joined by itself)
    released: asyncio.Event = field(default_factory=asyncio.Event)  # voice connection let go
    # Where the end-of-meeting summary is posted (set by voice / chat instruction); None = the
    # meeting's own text channel.
    minutes_channel_id: int | None = None
    attendee_ids: set[int] = field(default_factory=set)  # everyone who was in the room or spoke
    # Recording state: "unknown" (joined by itself, not known to be a meeting), "recording" or
    # "not_recording" (nothing is logged to memory, no summary). See MeetingManager.set_recording.
    recording: str = "recording"
    consent: str | None = None  # None = not asked, "pending", "answered", "assumed"
    consent_until: float = 0.0
    consent_medium: str = "text"
    first_speech_at: float = 0.0
    introduced: bool = False  # Salid was addressed or has spoken: no need to ask who it is
    notes_posted: list[str] = field(default_factory=list)
    last_note_at: float = 0.0
    last_note_pass: datetime | None = None
    note_pass_at: float = 0.0
    engaged: bool = False  # False = listening (speaks only when its name is called)
    engaged_until: float = 0.0

    @property
    def key(self) -> tuple[int, int]:
        return (self.guild_id, self.voice_channel.id)

    @property
    def lang(self) -> str:
        """Language of fixed messages (auto has no text to follow, so the starter's language)."""
        return self.language if self.language in ("th", "en") else self.fallback_lang

    def push_threadsafe(self, user_id: int, pcm: bytes, *, is_bot: bool = False) -> None:
        with contextlib.suppress(RuntimeError):  # loop already closed on shutdown
            self.loop.call_soon_threadsafe(self.on_pcm, user_id, pcm, time.monotonic(), is_bot)

    def on_pcm(self, user_id: int, pcm: bytes, now: float, is_bot: bool = False) -> None:
        if not self.ending:
            self.enqueue(self.segmenter.feed(user_id, pcm, now))
            if not is_bot and self.live is not None and self.voice_mode == "live":
                self.live.feed(user_id, pcm)

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
    """Meetings keyed by (guild id, voice channel id), each held by one voice body: the main bot
    or a helper account (see voice_pool). `bot` supplies settings, the Gemini client, permissions
    and Q&A."""

    def __init__(self, bot: SecretaryBot):
        self.bot = bot
        pool = getattr(bot, "voice_pool", None)
        self.pool: VoicePool = pool if pool is not None else VoicePool(bot)
        self.sessions: dict[tuple[int, int], MeetingSession] = {}
        self._joining: set[tuple[int, int]] = set()  # channels with a join in flight
        self._reserved: set[tuple[int, int]] = (
            set()
        )  # (guild, body index) in use but not yet/no more a session
        self._cooldown: dict[
            tuple[int, int], float
        ] = {}  # channel -> no auto-join until (monotonic)
        self._auto_tasks: dict[tuple[int, int], asyncio.Task] = {}
        self._bg: set[asyncio.Task] = set()
        self._now = time.monotonic
        self._sleep = asyncio.sleep
        self._stt_client: Any = None
        self._ack_pcm: dict[str, bytes] = {}

    @property
    def stt_client(self) -> Any:
        """Vertex client for segment transcription (regional when `transcribe_location` is set)."""
        loc = getattr(self.bot.s, "transcribe_location", "")
        if not loc or loc == getattr(self.bot.s, "gemini_location", loc):
            return self.bot.llm
        if self._stt_client is None:
            from .llm import make_client

            self._stt_client = make_client(self.bot.s.model_copy(update={"gemini_location": loc}))
        return self._stt_client

    # ---- helpers
    def session_for_channel(self, channel_id: int | None) -> MeetingSession | None:
        return next((s for s in self.sessions.values() if s.text_channel.id == channel_id), None)

    def session_in(self, guild_id: int, channel_id: int) -> MeetingSession | None:
        return self.sessions.get((guild_id, channel_id))

    def sessions_in_guild(self, guild_id: int) -> list[MeetingSession]:
        return [s for s in self.sessions.values() if s.guild_id == guild_id]

    def context_for(self, channel_id: int | None) -> str:
        """Recent meeting transcript for a message sent in the meeting's text chat ("" if none)."""
        s = self.session_for_channel(channel_id)
        if s is None:
            return ""
        return s.transcript.history(self.bot.s.now())

    @staticmethod
    def _body_channel(body: VoiceBody, guild, channel):
        """(guild, channel) as seen by `body`'s own client; (None, None) when it cannot see it."""
        if body.index == 0:
            return guild, channel
        hguild = body.client.get_guild(guild.id)
        hchan = hguild.get_channel(channel.id) if hguild is not None else None
        return hguild, hchan

    async def _connect(self, body: VoiceBody, channel):
        _, hchan = self._body_channel(body, channel.guild, channel)
        if hchan is None:
            raise RuntimeError(f"voice body {body.index} cannot see channel {channel.id}")
        return await hchan.connect(cls=voice_recv.VoiceRecvClient)

    async def _post(self, session: MeetingSession, text: str):
        """Post to the meeting's text chat (else the fallback channel); the sent message or None."""
        for ch in (session.text_channel, session.fallback_channel):
            if ch is None:
                continue
            try:
                return await ch.send(text)
            except discord.HTTPException:
                logger.warning("could not post to channel %s", getattr(ch, "id", "?"))
        return None

    async def _notice(self, session: MeetingSession, text: str):
        """A status line for the meeting's text chat. Meetings stay voice: nothing is posted unless
        `meeting_text_echo` is on."""
        if getattr(self.bot.s, "meeting_text_echo", False):
            return await self._post(session, text)
        return None

    def _spawn(self, session: MeetingSession, coro) -> None:
        """Run a background coroutine for the session (cancelled when the meeting ends)."""
        session.tasks = [x for x in session.tasks if not x.done()]
        session.tasks.append(asyncio.create_task(coro))

    def _sp(self, s: MeetingSession, key: str, **kw) -> str:
        """A fixed spoken line with the voice's particles (ค่ะ / คะ, or ครับ)."""
        p = str(getattr(self.bot.s, "voice_particle", "ค่ะ") or "ค่ะ")
        text = t(key, s.lang, **kw)
        return text if p == "ค่ะ" else text.replace("ค่ะ", p).replace("คะ", p)

    _NO_RE = re.compile(
        r"ไม่ต้อง|ไม่เป็นไร|ไม่ได้ประชุม|ไม่ใช่(?:การ)?ประชุม|คุยเล่น|ไม่บันทึก|ไม่เอา|ไม่จด"
        r"|^\W*ไม่\W*(?:ครับ|ค่ะ|คะ)?\W*$"
        r"|(?<![a-z])(?:no|nope|don'?t|not\s+a\s+meeting|just\s+chatting|no\s+notes)(?![a-z])",
        re.I,
    )
    _PRIVATE_RE = re.compile(
        r"ส่วนตัว|ออกไป|ไปก่อน|ไม่ต้องอยู่|ขอตัว|(?<![a-z])(?:private|privacy|leave|go\s+away)(?![a-z])",
        re.I,
    )
    _START_RE = re.compile(
        r"เริ่มประชุม|จดด้วย|บันทึกด้วย|จดให้หน่อย|ประชุมแล้ว|(?<![a-z])(?:start\s+(?:the\s+)?meeting"
        r"|take\s+notes|we'?re\s+(?:in\s+a\s+)?meeting)(?![a-z])",
        re.I,
    )

    _YES_RE = re.compile(
        r"ใช่|ประชุม|จดเลย|บันทึก|ได้เลย|โอเค|(?<![a-z])(?:yes|yeah|sure|ok|okay)(?![a-z])", re.I
    )

    @classmethod
    def is_decline(cls, text: str) -> bool:
        """An answer to "is this a meeting / may I take notes?" that says no."""
        return bool(cls._NO_RE.search((text or "").strip()))

    @classmethod
    def wants_privacy(cls, text: str) -> bool:
        return bool(cls._PRIVATE_RE.search(text or ""))

    @classmethod
    def is_start(cls, text: str) -> bool:
        return bool(cls._START_RE.search(text or ""))

    # ---- recording state
    def set_recording(self, s: MeetingSession, recording: bool) -> dict:
        """unknown / recording <-> not_recording. Going to not_recording forgets the transcript so
        far; nothing is logged to memory and no summary is posted while it lasts."""
        if recording:
            s.recording = "recording"
            return {"status": "recording"}
        s.recording = "not_recording"
        s.transcript.lines.clear()
        return {"status": "not_recording"}

    # ---- consent: a colleague asking whether this is a meeting
    async def _consent_line(self, s: MeetingSession) -> str:
        p = str(getattr(self.bot.s, "voice_particle", "ค่ะ") or "ค่ะ")
        lang = "English" if s.lang == "en" else "Thai"
        prompt = (
            "You are Salid, a friendly team member who just sat in on a voice channel to help "
            "and take meeting notes. Nobody has said what this is. Write ONE short, natural line "
            f"in {lang} asking, like a colleague would, whether this is a meeting and whether it "
            "is okay to take notes; they can say no. Vary the wording; no lecture, no policy. "
            f"A woman speaks it; use the particle {p} (questions: "
            f"{'คะ' if p == 'ค่ะ' else p}). Reply with the line only."
        )
        try:
            resp = await asyncio.to_thread(
                generate,
                self.bot.llm,
                self.bot.s.gate_model,
                prompt,
                types.GenerateContentConfig(temperature=1.0, max_output_tokens=100),
            )
            line = (getattr(resp, "text", None) or "").strip().strip('"“”')
            if 4 <= len(line) <= 220:
                return line
        except Exception:
            logger.warning("consent line generation failed", exc_info=True)
        return self._sp(s, f"meet_consent_ask_{random.randint(1, 3)}")

    async def ask_consent(self, s: MeetingSession, text: str | None = None, medium: str = "text"):
        """Ask once whether this is a meeting / notes are welcome. Text by default (does not
        interrupt); voice when the room is quiet or someone is talking to Salid."""
        if s.consent is not None or s.ending or s.recording == "not_recording":
            return {"status": "already_asked"}
        s.consent = "pending"
        s.consent_medium = "voice" if medium == "voice" and _connected(s.vc) else "text"
        s.consent_until = time.monotonic() + float(
            getattr(self.bot.s, "consent_window_seconds", 90.0)
        )
        line = (text or "").strip()[:300] or await self._consent_line(s)
        if s.consent_medium == "voice":
            await self._say_text(s, line)
        else:
            await self._post(s, line)
        return {"status": "asked", "medium": s.consent_medium}

    async def _not_a_meeting(self, s: MeetingSession, text: str, via: str) -> None:
        """ "Not a meeting" / "no notes": stop recording, say so briefly in the same medium. Stays
        quietly unless they ask for privacy; then leaves politely (60 min cooldown)."""
        s.consent = "answered"
        self.set_recording(s, False)
        leave = self.wants_privacy(text)
        key = f"meet_consent_no_{random.randint(1, 3)}" if leave else "meet_not_recording"
        line = self._sp(s, key)
        if via == "voice":
            await self._say_text(s, line)
        else:
            await self._post(s, line)
        if leave:
            await self.end(s.key, "declined")

    async def on_chat_text(self, message) -> bool:
        """A message typed in a meeting's text chat. True when it answered the notes question
        (then it is not a request for the assistant)."""
        s = self.session_for_channel(getattr(message.channel, "id", None))
        if s is None:
            return False
        text = message.content or ""
        if s.recording == "not_recording":
            if self.is_start(text):
                self.set_recording(s, True)
            return False
        if s.consent == "pending" and time.monotonic() <= s.consent_until and self.is_decline(text):
            await self._not_a_meeting(s, text, "text")
            return True
        return False

    # ---- notes worth writing down
    async def post_note(self, s: MeetingSession, text: str) -> dict:
        """One short line for the text chat: a decision, an action item or a number to remember.
        Sparing: no duplicates, at most one every few seconds, never while not recording."""
        text = " ".join((text or "").split())[:300]
        now = time.monotonic()
        if s.recording == "not_recording" or s.ending:
            return {"error": "not_recording"}
        if len(text) < 4:
            return {"error": "empty"}
        if any(text.casefold() == n.casefold() for n in s.notes_posted):
            return {"error": "duplicate"}
        if now - s.last_note_at < NOTE_MIN_GAP_S or len(s.notes_posted) >= MAX_NOTES:
            return {"error": "too_many_notes"}
        s.notes_posted.append(text)
        s.last_note_at = now
        await self._post(s, f"📌 {text}")
        return {"status": "posted"}

    async def _note_pass(self, s: MeetingSession) -> None:
        """Classic mode: the cheap gate reads the transcript since the last pass and posts one
        line per decision / action item."""
        since = s.last_note_pass
        s.last_note_pass = self.bot.s.now()
        lines = [
            GateLine(str(i), ln.speaker, ln.text)
            for i, ln in enumerate(s.transcript.lines)
            if since is None or ln.at > since
        ][-40:]
        if not lines:
            return
        notes = await asyncio.to_thread(
            notes_from_transcript,
            self.bot.llm,
            self.bot.s.gate_model,
            lines,
            channel_id=s.voice_channel.id,
        )
        for n in notes[:3]:
            await self.post_note(s, n)
            s.last_note_at = 0.0  # a batch may post up to three lines at once

    async def _say_text(self, s: MeetingSession, text: str) -> None:
        """Say a short fixed line in the voice channel (TTS). Failures are logged, never raised."""
        if not self.bot.s.voice_reply or s.ending or not _connected(s.vc):
            return
        try:
            pcm = self._ack_pcm.get(f"{s.lang}:{text}")
            if pcm is None:
                pcm = await asyncio.to_thread(say, self.bot.s, self.bot.llm, text, s.lang)
                if len(self._ack_pcm) < 64:
                    self._ack_pcm[f"{s.lang}:{text}"] = pcm
            await self._play(s, pcm)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("spoken line failed", exc_info=True)

    # ---- bodies and importance
    def _busy_indexes(self, guild_id: int) -> set[int]:
        return {i for g, i in self._reserved if g == guild_id} | {
            getattr(s.body, "index", 0) for s in self.sessions_in_guild(guild_id)
        }

    def _free_body(self, guild_id: int) -> VoiceBody | None:
        busy = self._busy_indexes(guild_id)
        return next((b for b in self.pool.usable(guild_id) if b.index not in busy), None)

    @staticmethod
    def _humans(channel) -> list:
        return [m for m in getattr(channel, "members", []) if not getattr(m, "bot", False)]

    def _is_exec(self, member) -> bool:
        try:
            return bool(self.bot.is_exec_member(member))
        except Exception:
            return False

    def _claim(self, key, channel, origin: str, scheduled: bool, started: float) -> Claim:
        humans = self._humans(channel)
        execs = sum(1 for m in humans if self._is_exec(m))
        return Claim(
            key, room_score(len(humans), execs, scheduled), len(humans), started, origin, scheduled
        )

    def _busy_reply(self, guild_id: int, lang: str) -> str:
        rooms = ", ".join(f"#{s.voice_channel.name}" for s in self.sessions_in_guild(guild_id))
        return t("meet_busy", lang, rooms=rooms or "-")

    async def _claim_body(self, guild, channel, origin: str, scheduled: bool, lang: str):
        """(body, "") with the body reserved, or (None, the reply explaining why not). When every
        body is taken, a weaker auto-joined session gives its body up (see pick_victim)."""
        body = self._free_body(guild.id)
        if body is None:
            cand = self._claim((guild.id, channel.id), channel, origin, scheduled, self._now())
            active = [
                self._claim(s.key, s.voice_channel, s.origin, s.meeting_id is not None, s.started)
                for s in self.sessions_in_guild(guild.id)
                if not s.ending
            ]
            victim = pick_victim(cand, active)
            if victim is None:
                return None, self._busy_reply(guild.id, lang)
            await self._preempt(self.sessions[victim.key])
            body = self._free_body(guild.id)
            if body is None:
                return None, self._busy_reply(guild.id, lang)
        self._reserved.add((guild.id, body.index))
        return body, ""

    async def _preempt(self, s: MeetingSession) -> None:
        """Tell the old room, end its session with a summary, wait until its body is free."""
        logger.info("meeting in %s gives its voice body to a more important one", s.key)
        await self._notice(s, t("meet_preempted", s.lang))
        task = asyncio.create_task(self.end(s.key, "preempted"))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(s.released.wait(), timeout=20)

    # ---- commands (return the reply for the invoker, in `lang`)
    async def join(
        self,
        member,
        fallback_channel=None,
        lang: str = "th",
        *,
        announce: str = "meet_announce",
        empty_grace: float | None = None,
        first_grace: float | None = None,
        on_end=None,
        meeting_id: str | None = None,
        origin: str | None = None,
    ) -> str:
        guild = member.guild
        if not self.bot.team.enabled(guild.id, "voice_meetings"):
            return t("meet_off", lang)
        voice = getattr(member, "voice", None)
        channel = getattr(voice, "channel", None)
        if channel is None:
            return t("meet_not_in_voice", lang)
        key = (guild.id, channel.id)
        origin = origin or ("scheduled" if meeting_id else "command")
        if origin == "command":
            self._cooldown.pop(key, None)  # an explicit request always overrides the cooldown
        existing = self.sessions.get(key)
        if existing is not None:
            return t("meet_already", lang, channel=existing.voice_channel.name)
        if key in self._joining:
            return t("meet_already", lang, channel=t("meet_joining", lang))
        # Reserve the channel before the first await so two commands cannot both join it.
        self._joining.add(key)
        reserved: tuple[int, int] | None = None
        try:
            body, why = await self._claim_body(guild, channel, origin, bool(meeting_id), lang)
            if body is None:
                return why
            reserved = (guild.id, body.index)
            hguild, hchan = self._body_channel(body, guild, channel)
            if hchan is None:
                return t("meet_connect_failed", lang)
            lacking = channel_missing(hchan, hguild, ("connect",))
            if lacking:
                return permission_message(
                    lacking, lang, getattr(body, "application_id", None) or
                    getattr(self.bot, "application_id", None), helper=body.index > 0,
                )  # fmt: skip
            try:
                vc = await self._connect(body, channel)
            except Exception:
                logger.exception("could not join voice channel %s", channel.id)
                return t("meet_connect_failed", lang)
            if empty_grace is None:
                empty_grace = float(getattr(self.bot.s, "leave_empty_seconds", 30.0))
            loop = asyncio.get_running_loop()
            session = MeetingSession(
                guild_id=guild.id, voice_channel=channel, text_channel=channel,
                fallback_channel=fallback_channel, vc=vc, loop=loop, started=time.monotonic(),
                language=self.bot.team.channel_language(guild.id, channel.id), fallback_lang=lang,
                empty_grace=empty_grace, first_grace=first_grace, on_end=on_end,
                meeting_id=meeting_id, started_at=self.bot.s.now().astimezone(UTC),
                starter=str(getattr(member, "display_name", "") or ""), body=body, origin=origin,
            )  # fmt: skip
            session.recording = "unknown" if origin == "auto" else "recording"
            session.introduced = origin == "command"
            self.sessions[key] = session
        finally:
            self._joining.discard(key)
            if reserved is not None:
                self._reserved.discard(reserved)
        try:
            vc.listen(MeetingSink(session))
        except Exception:
            logger.exception("could not start listening")
            self.sessions.pop(key, None)
            with contextlib.suppress(Exception):
                await vc.disconnect()
            return t("meet_connect_failed", lang)
        session.tasks = [
            asyncio.create_task(self._worker(session)) for _ in range(TRANSCRIBE_WORKERS)
        ]
        session.tasks.append(asyncio.create_task(self._ticker(session)))
        self._start_live(session)
        await self._persist(session, getattr(member, "id", 0))
        for m in self._humans(channel):
            session.attendee_ids.add(m.id)
        await self._notice(session, t(announce, session.lang))
        logger.info(
            "meeting started in guild %s channel %s on voice body %d (%s)",
            guild.id, channel.id, body.index, origin,
        )  # fmt: skip
        return t("meet_joined", lang, channel=channel.name)

    async def adopt(self, s: MeetingSession, meeting_id: str, on_end) -> None:
        """A scheduled meeting starts in a room Salid already sits in: the session now takes its
        notes (and counts as scheduled for priority)."""
        s.meeting_id, s.on_end, s.origin = meeting_id, on_end, "scheduled"
        if s.recording == "unknown":
            s.recording = "recording"  # the schedule says it is a meeting
        await self._persist(s)

    # ---- auto-join: Salid joins by itself when people are in a voice channel
    def _auto_wanted(self, ch) -> bool:
        guild = ch.guild
        key = (guild.id, ch.id)
        if key in self.sessions or key in self._joining:
            return False
        if self._cooldown.get(key, 0.0) > self._now():
            return False
        afk = getattr(guild, "afk_channel", None)
        if afk is not None and afk.id == ch.id:
            return False
        team = self.bot.team
        if not (
            team.enabled(guild.id, "voice_meetings")
            and team.enabled(guild.id, "auto_join_meetings")
        ):
            return False
        if team.channel_mode(guild.id, ch.id, ch.name, self.bot.s.ask_channel_ids) == "off":
            return False
        need = max(1, int(getattr(self.bot.s, "auto_join_min_humans", 1)))
        return len(self._humans(ch)) >= need

    async def on_voice_activity(self, member, before, after) -> None:
        """A person moved in the voice channels: consider joining the rooms they left/entered."""
        if getattr(member, "bot", False):
            return
        entered = getattr(after, "channel", None)
        if entered is not None and entered != getattr(before, "channel", None):
            s = self.sessions.get((entered.guild.id, entered.id))
            if s is not None:  # a new arrival may greet Salid without the name
                s.attendee_ids.add(member.id)
        left = getattr(before, "channel", None)
        if left is not None and not self._humans(left):
            self._cooldown.pop((left.guild.id, left.id), None)  # a new gathering: a new chance
        for ch in {
            c for c in (getattr(before, "channel", None), getattr(after, "channel", None)) if c
        }:
            self.kick(ch)

    def kick(self, ch) -> None:
        """Start (or keep) the auto-join attempt for one channel. Rapid duplicate events for the
        same channel share one attempt; with the default 0 s delay nothing waits."""
        key = (ch.guild.id, ch.id)
        pending = self._auto_tasks.get(key)
        if not self._auto_wanted(ch):
            if pending is not None and not pending.done():
                pending.cancel()  # only a delayed attempt can still be waiting
            return
        if pending is not None and not pending.done():
            return
        task = asyncio.create_task(self._auto_run(ch))
        self._auto_tasks[key] = task

        def done(t_: asyncio.Task, key=key) -> None:
            if self._auto_tasks.get(key) is t_:
                self._auto_tasks.pop(key, None)

        task.add_done_callback(done)

    async def _auto_run(self, ch) -> None:
        try:
            delay = float(getattr(self.bot.s, "auto_join_delay_seconds", 0.0) or 0.0)
            if delay > 0:  # optional stability wait (off by default)
                await self._sleep(delay)
                if not self._auto_wanted(ch):
                    return
            await self._auto_join(ch)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("auto-join failed for channel %s", getattr(ch, "id", "?"))

    async def _auto_join(self, ch) -> None:
        guild = ch.guild
        scheduler = getattr(self.bot, "scheduler", None)
        sched = scheduler.meeting_now(guild.id, ch.id) if scheduler is not None else None
        if sched is not None:  # a scheduled meeting: same allocator, +100, notes for the meeting
            await scheduler.auto_join_now(sched)
            return
        lang = self.bot.team.channel_language(guild.id, ch.id)
        starter = SimpleNamespace(
            id=0, guild=guild, voice=SimpleNamespace(channel=ch), display_name="Salid"
        )
        await self.join(
            starter,
            lang=lang if lang in ("th", "en") else "th",
            announce="meet_announce_auto",
            first_grace=float(getattr(self.bot.s, "meeting_empty_minutes", 2)) * 60.0,
            origin="auto",
        )

    def _voice_channels(self, guild) -> list:
        return [*getattr(guild, "voice_channels", ()), *getattr(guild, "stage_channels", ())]

    def scan_all(self) -> None:
        """After startup / when a body frees up: look at every voice channel with people in it."""
        for guild in getattr(self.bot, "guilds", ()):
            for ch in self._voice_channels(guild):
                self.kick(ch)

    def _rescan(self, guild_id: int) -> None:
        guild = self.bot.get_guild(guild_id)
        if guild is None:
            return
        for ch in self._voice_channels(guild):
            self.kick(ch)

    def _note_end(self, key, reason: str) -> None:
        if reason in ("command", "declined"):
            minutes = float(getattr(self.bot.s, "auto_join_cooldown_minutes", 60))
            self._cooldown[key] = self._now() + minutes * 60.0
        elif reason in ("timeout", "disconnected"):
            self._cooldown[key] = self._now() + BLOCK_AFTER_FAILURE_S

    # ---- persistence: a restart must not lose the meeting
    @property
    def _store(self):
        return getattr(self.bot, "secretary_store", None)

    async def _persist(self, s: MeetingSession, starter_id: int = 0) -> None:
        """Remember the active meeting (so a restart can rejoin it). Never raises."""
        store = self._store
        if store is None:
            return
        rec = {
            "guild_id": s.guild_id,
            "voice_channel_id": s.voice_channel.id,
            "body_index": getattr(s.body, "index", 0),
            "origin": s.origin,
            "text_channel_id": getattr(s.fallback_channel, "id", None),
            "started_at": (s.started_at or self.bot.s.now()).isoformat(timespec="seconds"),
            "language": s.language,
            "fallback_lang": s.fallback_lang,
            "meeting_id": s.meeting_id,
            "starter": s.starter,
            "starter_id": starter_id,
            "empty_grace": s.empty_grace,
            "first_grace": s.first_grace,
            "minutes_channel_id": s.minutes_channel_id,
            "recording": s.recording,
        }
        try:
            await asyncio.to_thread(store.put_active_meeting, rec)
        except Exception:
            logger.warning("could not persist the active meeting", exc_info=True)

    async def _unpersist(self, key: tuple[int, int]) -> None:
        store = self._store
        if store is None:
            return
        try:
            await asyncio.to_thread(store.clear_active_meeting, key[0], key[1])
        except Exception:
            logger.warning("could not clear the active meeting record", exc_info=True)

    def _logged_lines(self, guild, voice_channel, since: datetime) -> list[tuple]:
        """Voice lines of this meeting already logged to memory: (at, speaker, text)."""
        mem = getattr(self.bot, "memory_store", None)
        if mem is None:
            return []
        now = self.bot.s.now()
        start = since.astimezone(now.tzinfo).date().isoformat()
        msgs = mem.messages(
            guild.id, [voice_channel.id], start, now.date().isoformat(),
            channel_id=voice_channel.id, limit=2000, newest_first=False,
        )  # fmt: skip
        return [
            (m.created_at, m.author_name, m.text)
            for m in msgs
            if m.kind == "voice" and m.created_at >= since
        ]

    async def resume_all(self) -> None:
        """After a restart (on_ready): rejoin persisted meetings that still have people in them;
        end the others properly. Safe to call again on a gateway re-ready."""
        await self.pool.wait_ready()  # helpers log in a little after the main bot
        store = self._store
        if store is None:
            return
        try:
            records = await asyncio.to_thread(store.list_active_meetings)
        except Exception:
            logger.warning("could not list active meetings", exc_info=True)
            return
        for rec in records:
            try:
                await self._resume(rec)
            except Exception:
                logger.exception("could not resume meeting in guild %s", rec.get("guild_id"))

    async def _resume(self, rec: dict) -> None:
        gid = int(rec["guild_id"])
        guild = self.bot.get_guild(gid)
        channel = guild.get_channel(int(rec["voice_channel_id"])) if guild else None
        if guild is None or channel is None:
            await self._unpersist((gid, int(rec["voice_channel_id"])))
            return
        key = (gid, channel.id)
        if key in self.sessions or key in self._joining:
            return
        started_at = datetime.fromisoformat(rec["started_at"])
        text_ch = guild.get_channel(rec["text_channel_id"]) if rec.get("text_channel_id") else None
        humans = self._humans(channel)
        elapsed = max(0.0, (self.bot.s.now() - started_at).total_seconds())
        lang = rec.get("fallback_lang") or "th"
        on_end = None
        if rec.get("meeting_id") and getattr(self.bot, "scheduler", None) is not None:
            mid = rec["meeting_id"]

            async def on_end(session) -> None:
                await self.bot.scheduler.on_voice_end(mid, session)

        # The same body as before when it is still around, else any free one.
        busy = self._busy_indexes(gid)
        body = self.pool.body(rec.get("body_index", 0))
        if body is None or not body.available or not body.in_guild(gid) or body.index in busy:
            body = self._free_body(gid)
        vc = None
        if humans and body is not None:
            self._joining.add(key)
            self._reserved.add((gid, body.index))
            try:
                vc = await self._connect(body, channel)
            except Exception:
                logger.exception("could not rejoin voice channel %s", channel.id)
            finally:
                self._joining.discard(key)
                self._reserved.discard((gid, body.index))
        if vc is None:  # no people, no body, or could not connect: end it with what the log has
            humans = []
        session = MeetingSession(
            guild_id=gid, voice_channel=channel, text_channel=channel, fallback_channel=text_ch,
            vc=vc, loop=asyncio.get_running_loop(), started=time.monotonic() - elapsed,
            language=rec.get("language") or "th", fallback_lang=lang,
            empty_grace=float(rec.get("empty_grace") or EMPTY_GRACE_S),
            first_grace=rec.get("first_grace"), on_end=on_end, meeting_id=rec.get("meeting_id"),
            started_at=started_at, starter=rec.get("starter", ""), had_humans=True,
            body=body or self.pool.main, origin=rec.get("origin") or "command",
            minutes_channel_id=rec.get("minutes_channel_id"),
            recording=rec.get("recording") or "recording",
        )  # fmt: skip
        for m in humans:
            session.attendee_ids.add(m.id)
        try:
            for at, speaker, text in await asyncio.to_thread(
                self._logged_lines, guild, channel, started_at
            ):
                session.transcript.add(at, speaker, text)
        except Exception:
            logger.warning("could not reload the logged meeting lines", exc_info=True)
        self.sessions[key] = session
        if not humans:
            await self.end(key, "empty")
            return
        try:
            vc.listen(MeetingSink(session))
        except Exception:
            logger.exception("could not start listening after rejoin")
            await self.end(key, "disconnected")
            return
        session.tasks = [
            asyncio.create_task(self._worker(session)) for _ in range(TRANSCRIBE_WORKERS)
        ]
        session.tasks.append(asyncio.create_task(self._ticker(session)))
        self._start_live(session)
        await self._notice(session, t("meet_rejoined", session.lang))
        logger.info("meeting resumed in guild %s channel %s", gid, channel.id)

    async def on_disconnected(self, guild_id: int, body_index: int = 0) -> None:
        """A voice body's connection dropped unexpectedly (not /leave, not shutdown): retry with
        backoff, then give up and end that meeting."""
        s = next(
            (
                x
                for x in self.sessions_in_guild(guild_id)
                if getattr(x.body, "index", 0) == body_index
            ),
            None,
        )
        if s is None or s.ending or s.reconnecting:
            return
        key = s.key
        if not self._humans(s.voice_channel):
            await self.end(key, "disconnected")
            return
        s.reconnecting = True
        try:
            for attempt, delay in enumerate(RECONNECT_DELAYS, 1):
                await asyncio.sleep(delay)
                if s.ending or self.sessions.get(key) is not s:
                    return
                try:
                    with contextlib.suppress(Exception):
                        await s.vc.disconnect(force=True)
                    vc = await self._connect(s.body, s.voice_channel)
                    vc.listen(MeetingSink(s))
                except Exception:
                    logger.warning("voice reconnect %d failed", attempt, exc_info=True)
                    continue
                s.vc = vc
                if s.live is not None and s.voice_mode == "live":
                    with contextlib.suppress(Exception):
                        s.live.stop_playback()
                await self._notice(s, t("meet_reconnected", s.lang))
                return
            await self._post(s, t("meet_reconnect_failed", s.lang))
            s.reconnecting = False
            await self.end(key, "disconnected")
        finally:
            s.reconnecting = False

    def _start_live(self, session: MeetingSession) -> None:
        """Live voice mode: connect a Gemini Live session now (it takes ~2 s) so the first question
        is answered fast. If it cannot connect the meeting falls back to classic wake words."""
        s = self.bot.s
        if not s.voice_reply or getattr(s, "voice_mode", "classic") != "live":
            return
        from .live_voice import LiveVoice

        session.voice_mode = "live"
        session.live = LiveVoice(
            self.bot,
            session,
            post=lambda text: self._notice(session, text),
            on_failed=lambda: self._live_failed(session),
        )
        session.live.start()

    def _live_failed(self, session: MeetingSession) -> None:
        logger.error("live voice unavailable in guild %s: back to wake words", session.guild_id)
        session.voice_mode = "classic"
        if session.live is not None:
            session.live.stop_playback()

    async def leave(self, member, lang: str = "th", text_channel_id: int | None = None) -> str:
        """End the session of the room the person is in (else the one whose text chat they typed
        in, else, for a manager, the only one in the guild)."""
        from .bot import can_manage_guild

        guild_id = member.guild.id
        mine = getattr(getattr(member, "voice", None), "channel", None)
        sessions = self.sessions_in_guild(guild_id)
        if not sessions:
            return t("meet_none", lang)
        session = self.sessions.get((guild_id, mine.id)) if mine is not None else None
        if session is None and text_channel_id is not None:
            session = next((x for x in sessions if x.text_channel.id == text_channel_id), None)
        if session is None:
            if not can_manage_guild(member):
                return t("meet_leave_denied", lang)
            if len(sessions) > 1:
                rooms = ", ".join(f"#{x.voice_channel.name}" for x in sessions)
                return t("meet_leave_which", lang, rooms=rooms)
            session = sessions[0]
        elif mine != session.voice_channel and not can_manage_guild(member):
            return t("meet_leave_denied", lang)
        asyncio.create_task(self.end(session.key, "command"))
        return t("meet_left", lang)

    # ---- background work
    async def _ticker(self, s: MeetingSession) -> None:
        try:
            while not s.ending:
                await asyncio.sleep(TICK_S)
                now = time.monotonic()
                s.enqueue(s.segmenter.tick(now))
                self._consent_tick(s, now)
                if now - s.started > MEETING_MAX_S:
                    asyncio.create_task(self.end(s.key, "timeout"))
                    return
                if not s.max_warned and now - s.started > MEETING_MAX_S - MAX_WARN_S:
                    s.max_warned = True
                    await self._notice(s, t("meet_max_warning", s.lang))
                humans = self._humans(s.voice_channel)
                grace = s.empty_grace
                if humans:
                    s.empty_since = None
                    s.had_humans = True
                    continue
                if not s.had_humans and s.first_grace is not None:
                    grace = s.first_grace  # an auto-join nobody came to
                if s.empty_since is None:
                    s.empty_since = now
                elif now - s.empty_since > grace:
                    asyncio.create_task(self.end(s.key, "empty"))
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("meeting ticker crashed")

    def _consent_tick(self, s: MeetingSession, now: float) -> None:
        """Nobody addressed Salid soon after the first speech: ask once. Silence after the question
        counts as an OK."""
        delay = float(getattr(self.bot.s, "consent_delay_seconds", 20.0))
        if (
            s.recording == "unknown"
            and s.consent is None
            and not s.introduced
            and s.first_speech_at
            and now - s.first_speech_at > delay
        ):
            self._spawn(s, self.ask_consent(s))
        if s.engaged and now > s.engaged_until and not s.speaking:
            self._set_engaged(s, False)  # nobody addressed Salid for a while: back to listening
        if s.consent == "pending" and now > s.consent_until:
            s.consent = "assumed"
            if s.recording == "unknown":
                s.recording = "recording"
        if (
            s.voice_mode != "live"
            and s.recording != "not_recording"
            and len(s.transcript)
            and now - s.note_pass_at >= float(getattr(self.bot.s, "notes_batch_seconds", 60.0))
        ):
            s.note_pass_at = now
            self._spawn(s, self._note_pass(s))

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
        name = (
            member.display_name if member else t("meet_participant", s.lang, n=seg.user_id % 1000)
        )
        text = await asyncio.to_thread(
            transcribe_pcm, self.stt_client, self.bot.s.gemini_model, seg.pcm
        )
        if not text:
            return
        now_m = time.monotonic()
        if not s.first_speech_at:
            s.first_speech_at = now_m
        if s.recording == "not_recording":
            if self.is_start(text):
                self.set_recording(s, True)
        elif s.consent == "pending" and now_m <= s.consent_until and not self._is_echo(s, seg):
            if self.is_decline(text):
                await self._not_a_meeting(s, text, "voice")
                return
            if self._YES_RE.search(text):  # "ใช่ จดเลย" / "yes": it is a meeting
                s.consent = "answered"
                s.recording = "recording"
        if s.recording != "not_recording":
            s.transcript.add(self.bot.s.now(), name, text)
            s.attendee_ids.add(seg.user_id)
            await self.bot.record_voice(s.voice_channel, seg.user_id, name, text, self.bot.s.now())
        logger.info("meeting %s: %s said %d chars", s.guild_id, seg.user_id, len(text))
        said = parse_wake(text)
        called = said is not None
        if called and parse_meeting_command(said) == "leave":
            asyncio.create_task(self.end(s.key, "command"))  # "สลิด ออกประชุม"
            return
        if s.engaged and not called and parse_meeting_command(text) == "leave":
            asyncio.create_task(self.end(s.key, "command"))  # "ออกจากห้องได้เลย" while engaged
            return
        if await self._minutes_instruction(s, member, text):
            return
        now = time.monotonic()
        if s.engaged and not called and self.is_dismissal(text):
            self._set_engaged(s, False)  # "ขอบคุณครับ พอแล้ว": back to listening, not leaving
            if s.voice_mode != "live" and not self._is_echo(s, seg):
                self._spawn(s, self._say_text(s, self._sp(s, "meet_dismiss_ack")))
            return
        if called:
            self._set_engaged(s, True)
        elif s.engaged and not self._is_echo(s, seg):
            s.engaged_until = now + self._engage_s()  # talking with Salid keeps it engaged
        if s.voice_mode == "live":
            if called and s.live is not None:
                # Deterministic: the name was said, so Salid must answer. If the Live model has
                # not started to speak by itself, give it a cue.
                s.live.ensure_reply(seg.started + seg.duration_s)
            return  # otherwise the Live session answers by itself; this is only the record
        if self._is_echo(s, seg):
            return  # probably Salid's own voice coming back through someone's mic
        question = said
        if question is None:
            if now - s.pending_wake.pop(seg.user_id, -1e9) < PENDING_WAKE_S:
                question = text  # "สลิด ..." then a pause, then the actual question
            elif s.engaged:
                question = text  # a conversation is going: no name needed
            else:
                return
        if not question:
            s.pending_wake[seg.user_id] = now
            if member is not None:
                await self._notice(s, f"{member.mention} {t('meet_wake_empty', s.lang)}")
            return
        s.pending_wake.pop(seg.user_id, None)
        if self.bot.s.voice_reply and member is not None:
            s.tasks = [t for t in s.tasks if not t.done()]
            s.tasks.append(asyncio.create_task(self.acknowledge(s)))
        await self.answer(s, member, name, question)

    # ---- listening / engaged
    _DISMISS_RE = re.compile(
        r"^\W*(?:(?:โอเค|ok|okay|เออ|ได้|งั้น)\s*)?(?:(?:ครับ|ค่ะ|คะ|นะ)\s*)?"
        r"(?:(?:ขอบคุณ(?:มาก)?|พอแล้ว|พอ|ยุติได้|ยุติ|จบแค่นี้|จบแล้ว|ออกได้|ไม่มีอะไรแล้ว|เท่านี้)"
        r"(?:\s*(?:ครับ|ค่ะ|คะ|นะ|แล้ว|ได้|เลย|ก่อน|มาก))*\W*)+$"
        r"|^\W*(?:ok(?:ay)?[,. ]*)?(?:thanks?(?:\s+you)?|thank\s+you)(?:[,. ]+(?:that'?s\s+all|that'?s\s+it|"
        r"we'?re\s+good|bye))?\W*$"
        r"|^\W*(?:that'?s\s+(?:all|it)|we'?re\s+good|all\s+good|no\s+more\s+questions)\W*$",
        re.I,
    )

    @classmethod
    def is_dismissal(cls, text: str) -> bool:
        """ "พอแล้วครับ" / "ขอบคุณครับ" / "thanks, that's all" said on its own: back to listening
        (NOT leaving the room: "ออกประชุม" / "ออกจากห้อง" / "leave" do that)."""
        text = (text or "").strip()
        if len(text) > 50 or parse_meeting_command(text) == "leave":
            return False
        return bool(cls._DISMISS_RE.search(text))

    def _engage_s(self) -> float:
        return float(getattr(self.bot.s, "engaged_timeout_seconds", 90.0))

    def _set_engaged(self, s: MeetingSession, on: bool) -> None:
        """listening <-> engaged. The Live model is told the state so it stays silent when
        listening."""
        s.engaged_until = time.monotonic() + self._engage_s() if on else 0.0
        if s.engaged == on:
            return
        s.engaged = on
        s.introduced = True if on else s.introduced
        if s.live is not None:
            s.live.notify_state(on)

    # ---- minutes destination
    def _minutes_allowed(self, s: MeetingSession, member=None):
        """Channels the minutes may go to: Salid can post there and the speaker (voice instruction
        by a known person) or everyone in the room (instruction without a speaker) can see them."""
        guild = s.voice_channel.guild
        me = getattr(guild, "me", None)
        people = [member] if member is not None else self._humans(s.voice_channel)

        def allowed(ch) -> bool:
            if me is not None and not can_see_and_post(ch, me):
                return False
            if member is not None:
                return can_see_and_post(ch, member)
            return all(can_see(ch, p) for p in people)

        return allowed

    async def set_minutes_destination(self, s: MeetingSession, name: str, member=None) -> dict:
        """Point the end-of-meeting summary at a channel. Returns a small dict (status / error) for
        the Live tool and the spoken confirmation."""
        guild = s.voice_channel.guild
        res = await asyncio.to_thread(
            lambda: resolve_channel(guild.text_channels, name, self._minutes_allowed(s, member))
        )
        if res.status != "ok":
            out: dict = {"error": res.status}
            if res.candidates:
                out["candidates"] = [c.name for c in res.candidates[:5]]
            return out
        s.minutes_channel_id = res.channel.id
        await self._persist(s)
        return {"status": "set", "channel": res.channel.name}

    async def _minutes_instruction(self, s: MeetingSession, member, text: str) -> bool:
        """ "ส่งมติการประชุมไปห้อง X" heard in the room: remember X, confirm by voice (classic mode;
        in live mode the model confirms through its own tool call). True when it was one."""
        req = parse_minutes_destination(text)
        if req is None or s.ending:
            return False
        res = await self.set_minutes_destination(s, req.name, member)
        live = s.voice_mode == "live"
        if res.get("status") == "set":
            if not live:
                line = self._sp(s, "minutes_set_spoken", channel=res["channel"])
                self._spawn(s, self._say_text(s, line))
            return True
        if not req.explicit:
            return False  # "ส่งสรุปให้ทุกคน": not a channel after all
        if not live:
            key = {
                "not_allowed": "minutes_not_allowed_spoken",
                "ambiguous": "minutes_ambiguous_spoken",
            }.get(res.get("error", ""), "minutes_not_found_spoken")
            self._spawn(s, self._say_text(s, self._sp(s, key)))
        return True

    @staticmethod
    def _is_echo(s: MeetingSession, seg: Segment) -> bool:
        """True when the segment was (partly) spoken while Salid was talking, or just after.
        Transcription lags the audio, so the audio's own timestamps are compared as well."""
        if s.speaking or time.monotonic() < s.speaking_until:
            return True
        return bool(s.speaking_until) and (
            seg.started + seg.duration_s >= s.speaking_since and seg.started <= s.speaking_until
        )

    async def answer(self, s: MeetingSession, member, name: str, question: str) -> None:
        from .bot import chunks
        from .qa import ask

        bot = self.bot
        s.introduced = True  # somebody is talking to Salid
        if member is None or not bot.team.enabled(s.guild_id, "qa"):
            return
        async with s.answer_lock:
            history = s.transcript.history(bot.s.now())
            memory = None
            guild = getattr(member, "guild", None)
            if guild is not None:  # the asker's readable channels, as for a text question
                where = SimpleNamespace(guild=guild, channel=s.text_channel)
                memory = await bot._memory_tools(where, guild, member)
            ans = await asyncio.to_thread(
                ask,
                bot.s,
                bot.store,
                question,
                guild_id=s.guild_id,
                is_exec=bot.is_exec_member(member),
                user_id=member.id,
                history=history,
                language=s.language,
                memory=memory,
                **bot._web_kw(s.guild_id),
            )
            if getattr(bot.s, "meeting_text_echo", False):  # meetings stay voice by default
                for i, part in enumerate(chunks(ans.text, 1800)):
                    await self._post(s, f"{member.mention} {part}" if i == 0 else part)
        if bot.s.voice_reply and not s.ending and _connected(s.vc):
            # In the background: playback takes seconds and must not hold a transcription worker.
            s.tasks = [t for t in s.tasks if not t.done()]
            s.tasks.append(asyncio.create_task(self.speak(s, ans.text, question)))

    async def acknowledge(self, s: MeetingSession) -> None:
        """Play a short cached "one sec" right away, while the answer is being worked out."""
        if not self.bot.s.voice_reply or s.ending or s.speaking:
            return
        lang = s.lang if s.lang in ACK_TEXT else "th"
        try:
            pcm = self._ack_pcm.get(lang)
            if pcm is None:
                pcm = await asyncio.to_thread(say, self.bot.s, self.bot.llm, ACK_TEXT[lang], lang)
                self._ack_pcm[lang] = pcm
            if not s.speech_ready:
                await self._play(s, pcm)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("acknowledgment failed", exc_info=True)

    async def speak(self, s: MeetingSession, text: str, question: str = "") -> None:
        """Say a short spoken version of `text` in the voice channel. The written answer is
        already posted, so any failure here is logged and swallowed."""
        bot = self.bot
        if not bot.s.voice_reply or s.ending or s.speech_pending or not _connected(s.vc):
            return
        # Claim the pipeline before its first await. Two transcription workers can answer at once;
        # only one should pay for shortening and TTS when Discord can play only one source.
        s.speech_pending = True
        try:
            # The cheap model: ~1-2 s versus ~8 s for the thinking model, and it is plenty here.
            spoken = await asyncio.to_thread(spoken_text, bot.llm, bot.s.gate_model, question, text)
            if not spoken:
                return
            pcm = await asyncio.to_thread(say, bot.s, bot.llm, spoken, s.lang)
            s.speech_ready = True
            if not bot.s.voice_reply or s.ending:
                return
            await self._play(s, pcm)  # waits for a playing acknowledgment instead of cutting it off
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("spoken reply failed", exc_info=True)
        finally:
            s.speech_pending = False
            s.speech_ready = False

    async def _play(self, s: MeetingSession, pcm: bytes) -> None:
        """Play PCM (24 kHz mono) and wait until it ends; clips queue on `play_lock` so they never
        overlap or reorder. Sets the echo-guard flags."""
        async with s.play_lock:
            vc = s.vc
            if s.ending or not self.bot.s.voice_reply or not _connected(vc) or vc.is_playing():
                return
            loop = asyncio.get_running_loop()
            done: asyncio.Future = loop.create_future()

            def finished(err: Exception | None) -> None:  # runs on the player thread
                def resolve() -> None:
                    if not done.done():
                        done.set_result(err)

                with contextlib.suppress(RuntimeError):  # loop already closed
                    loop.call_soon_threadsafe(resolve)

            source: discord.AudioSource = pcm_source(pcm, getattr(self.bot.s, "tts_pitch", 1.0))
            owned = False
            s.speaking, s.speaking_since = True, time.monotonic()
            try:
                vc.play(source, after=finished)
                owned = True
                err = await asyncio.wait_for(done, pcm_duration_s(pcm) + SPEAK_TIMEOUT_PAD_S)
                if err:
                    logger.warning("spoken reply playback error: %s", err)
            except TimeoutError:
                logger.warning("spoken reply playback timed out")
            finally:
                s.speaking = False
                s.speaking_until = time.monotonic() + SPEAK_COOLDOWN_S
                _stop_playback(vc)
                if not owned:
                    with contextlib.suppress(Exception):
                        source.cleanup()

    # ---- ending
    async def end(self, key: tuple[int, int], reason: str = "command") -> None:
        s = self.sessions.get(key)
        if s is None or s.ending:
            return
        s.ending = True
        self.sessions.pop(key, None)
        held = (s.guild_id, getattr(s.body, "index", 0))
        self._reserved.add(held)  # the body stays taken until its voice connection is let go
        self._note_end(key, reason)
        logger.info("meeting ending in guild %s channel %s (%s)", key[0], key[1], reason)
        if reason != "shutdown":  # a restart keeps the record so the meeting is rejoined
            await self._unpersist(key)
        if s.live is not None:
            with contextlib.suppress(Exception):
                await s.live.close()
        _stop_playback(s.vc)
        with contextlib.suppress(Exception):
            s.vc.stop_listening()
        with contextlib.suppress(Exception):
            await s.vc.disconnect()
        self._reserved.discard(held)
        s.released.set()
        if reason not in ("shutdown", "preempted"):
            self._rescan(key[0])  # a body is free: a waiting room may take it
        s.enqueue(s.segmenter.flush_all())
        with contextlib.suppress(asyncio.TimeoutError):  # let pending segments finish
            await asyncio.wait_for(s.queue.join(), timeout=90)
        for task in s.tasks:
            if task is not asyncio.current_task():
                task.cancel()
        try:
            await self._post_summary(s, reason)
        finally:
            if s.on_end is not None:
                try:
                    await s.on_end(s)
                except Exception:
                    logger.exception("meeting end callback failed")

    async def _post_minutes(self, s: MeetingSession, text: str):
        """The summary goes to the channel chosen for the minutes, else (or if that fails) to the
        meeting's own text channel."""
        if s.minutes_channel_id is not None:
            ch = s.voice_channel.guild.get_channel(s.minutes_channel_id)
            if ch is not None:
                try:
                    return await ch.send(text)
                except discord.HTTPException:
                    logger.warning("could not post the minutes to channel %s", ch.id)
        return await self._post(s, text)

    def _store_minutes(self, s: MeetingSession, reason: str) -> None:
        """Keep the summary text so it can be sent elsewhere later (deliver_minutes). Never raises."""
        docs = getattr(self.bot, "docs", None)
        if docs is None or not s.summary_text:
            return
        ended = self.bot.s.now().astimezone(UTC).isoformat(timespec="seconds")
        try:
            docs.put(
                MINUTES,
                s.guild_id,
                f"{ended}-{s.voice_channel.id}",
                {
                    "voice_channel_id": s.voice_channel.id,
                    "voice_channel_name": getattr(s.voice_channel, "name", ""),
                    "attendee_ids": sorted(s.attendee_ids),
                    "started_at": (s.started_at or self.bot.s.now()).isoformat(timespec="seconds"),
                    "ended_at": ended,
                    "reason": reason,
                    "language": s.language,
                    "summary": s.summary_text[:SUMMARY_MAX_CHARS],
                },
            )
        except Exception:
            logger.warning("could not store the meeting minutes", exc_info=True)

    async def _post_summary(self, s: MeetingSession, reason: str) -> None:
        from .bot import chunks

        if reason == "declined" or s.recording == "not_recording":
            return
        if not len(s.transcript):
            await self._post_minutes(s, t("meet_no_transcript", s.lang))
            return
        try:
            body = await asyncio.to_thread(
                summarize_transcript,
                self.bot.llm,
                self.bot.s.gemini_model,
                s.transcript.full_text(),
                s.language,
            )
        except Exception:
            logger.exception("meeting summary failed")
            body = ""
        if not body:
            await self._post_minutes(s, t("meet_summary_failed", s.lang))
            return
        why = t(f"end_{reason}", s.lang) if reason in END_REASONS else reason
        header = t("meet_summary_header", s.lang, reason=why)
        s.summary_text = body
        for i, part in enumerate(chunks(header + body, 1900)):
            sent = await self._post_minutes(s, part)
            if i == 0 and sent is not None:
                s.summary_url = getattr(sent, "jump_url", "") or ""
        await asyncio.to_thread(self._store_minutes, s, reason)
        s.transcript.lines.clear()  # transcript is never persisted

    async def shutdown(self) -> None:
        """The bot is stopping (deploy/restart): leave quietly. The meeting is NOT ended: its record
        stays so on_ready rejoins it, and its transcript is in the memory log."""
        for task in self._auto_tasks.values():
            task.cancel()
        for key in list(self.sessions):
            s = self.sessions.pop(key, None)
            if s is None:
                continue
            s.ending = True
            if s.live is not None:
                with contextlib.suppress(Exception):
                    await s.live.close()
            for task in s.tasks:
                task.cancel()
            _stop_playback(s.vc)
            with contextlib.suppress(Exception):
                s.vc.stop_listening()
            with contextlib.suppress(Exception):
                await s.vc.disconnect()
