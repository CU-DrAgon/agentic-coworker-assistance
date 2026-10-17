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
The model has ONE data tool, ask_salid(question): it runs the full text agent (team memory scoped to
what everyone in the room may read, web, and the ledger only when every human present is an
executive; the mixed stream does not say who is speaking). General knowledge and chat need no tool.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import difflib
import io
import logging
import random
import re
import secrets
import threading
import time
from array import array
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import discord
from google.genai import types

from .captions import normalize_language
from .fillers import POOL_SIZE, ClipCache, FillerWriter, write_line
from .i18n import EXTRA, lang_name, resolve_lang
from .i18n import t as catalog
from .meeting import (
    SPEAK_COOLDOWN_S,
    SpeechStats,
    _connected,
    _stop_playback,
    looks_like_speech,
    parse_meeting_command,
    parse_wake,
    rms,
    stats_from_frame_rms,
    stats_line,
)
from .mentions import tag_candidates
from .qa import THAI_DAYS
from .speech import pcm_duration_s, pitch_pcm, say, with_particle
from .tone import DEFAULT_TONE, effective_tone, voice_line, voice_note
from .voice_actions import rebind_actions

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


def intentional_barge_in(settings: Any) -> bool:
    return str(getattr(settings, "live_barge_in", "intentional") or "").lower() != "any"


# Said to Salid while she talks = "stop now". Short, explicit phrases (or her name) only, so a noisy
# room, a side conversation or a cough never cuts her off mid-sentence.
_STOP = re.compile(
    r"(?:สลิด|salid|ซาลิด)"
    r"|หยุด|พอ\s*(?:แล้ว|ก่อน|ละ)|เดี๋ยว\s*(?:ก่อน|ๆ|เดี๋ยว)|ขอ\s*(?:แทรก|พูด|ขัด)|เงียบ|ฟังก่อน"
    r"|รอ\s*ก่อน"
    r"|\b(?:stop|wait|hold on|hang on|enough|be quiet|quiet please|excuse me|sorry to interrupt)\b",
    re.I,
)


def is_stop_request(text: str) -> bool:
    """Is this (partial) utterance, heard while Salid speaks, a deliberate interruption?"""
    return bool(_STOP.search(" ".join(str(text or "").split())))


def build_realtime_input(settings: Any) -> types.RealtimeInputConfig:
    """Turn-taking for natural Thai talk: short greetings start a turn quickly (start sensitivity
    high), a mid-sentence pause does not end it (end sensitivity low, long silence_duration_ms).
    With intentional barge-in the server never interrupts Salid for incoming audio
    (NO_INTERRUPTION); `LiveVoice._on_heard` stops her when someone deliberately asks."""
    extra = (
        {"activity_handling": types.ActivityHandling.NO_INTERRUPTION}
        if intentional_barge_in(settings)
        else {}
    )
    return types.RealtimeInputConfig(
        **extra,
        automatic_activity_detection=types.AutomaticActivityDetection(
            start_of_speech_sensitivity=_sensitivity(
                types.StartSensitivity, getattr(settings, "live_start_sensitivity", "high"), "high"
            ),
            end_of_speech_sensitivity=_sensitivity(
                types.EndSensitivity, getattr(settings, "live_end_sensitivity", "low"), "low"
            ),
            prefix_padding_ms=int(getattr(settings, "live_prefix_padding_ms", 300)),
            silence_duration_ms=int(getattr(settings, "live_silence_ms", 1200)),
        ),
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
    schedule). Ask briefly; they can say no. Text by default.

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


def set_captions(on: bool, language: str = "original") -> dict:
    """Live captions (subtitles) in the meeting's text chat. They are OFF unless someone asks. Call
    with on=true when people ask for captions ("เปิดซับ", "เปิดซับให้หน่อย", "turn on captions"), with
    language "th" or "en" when they also want a translation ("แปลเป็นอังกฤษให้ด้วย", "captions in
    Thai"), and on=false for "ปิดซับ" / "captions off". Then confirm in one short sentence.

    Args:
        on: true to show captions, false to stop them.
        language: "original" (as spoken, default), "th" (Thai) or "en" (English).
    """
    return {}


def answer_privately(question: str) -> dict:
    """Answer the person who just asked in a PRIVATE DM, with everything they personally may see.
    Use it when ask_salid says the answer cannot be said in the room (not everyone present may
    hear it) and the asker says yes to a DM, or when they ask for a private answer / "ส่ง DM ให้".
    The answer never comes back to you: just say you sent it, in one short sentence.

    Args:
        question: what the person wants answered privately, in their words.
    """
    return {}


def status_question(question: str) -> str:
    """The question as quoted in a typed status line: one line, at most `STATUS_Q_CHARS`."""
    q = " ".join(str(question).split())
    return q if len(q) <= STATUS_Q_CHARS else q[: STATUS_Q_CHARS - 1].rstrip() + "…"


def ask_salid(question: str) -> dict:
    """Ask the full Salid text assistant when the question needs TEAM information or current web
    info: what was said or decided in meetings/chats, tasks and owners, files, receipts and
    spending, schedules, or fresh news, or when someone asks Salid to DO something (create a
    document, deck, poll, event, send something). Do NOT call it for general knowledge, opinions,
    small talk or explanations you can give yourself. Returns {"answer": ...} to say as it is;
    or {"status": "working", "job": ...}: the job runs in the background and a note is already in
    the text chat, so say nothing about it (at most “ค่ะ” / “Okay”), do not wait, go on with the
    conversation (the result is sent later as a bracketed message); or an error with
    {"say": ...} to read as it is. A "working" answer that has a "say" line: say that line.

    Args:
        question: the question, complete and self-contained (names, dates, the topic).
    """
    return {}


# Mouth gate: tools the Live model may not call while nobody has called Salid (side effects, or a
# spoken question). `post_note` (taking notes is her job) is always allowed.
GATED_TOOLS = frozenset(
    {"set_captions", "set_recording", "set_minutes_destination", "ask_salid", "answer_privately"}
)
NOBODY_ASKED = {"error": "nobody asked: stay silent"}
FRAME_S = 0.02  # one frame from `feed`
ASK_EVIDENCE_S = 6.0  # a tool call arrives 1-2 s after the utterance: look this far back
ASK_MIN_VOICED_S = 0.2  # the model heard a question: only noise is excluded by this check
HEARD_GAP_S = 3.0  # input-transcript text after this much quiet starts a new utterance
NAME_DEDUPE_S = 10.0  # a classic name hit this soon after a Live-transcript hit is the same call
# Hand-written fallback pool (i18n ack_voice_*): a line written for the question is
# used when it is ready in time (`Settings.voice_ack_timeout_s`), otherwise one of these.
FILLERS = {
    lang: [catalog(f"ack_voice_{n}", lang) for n in range(1, POOL_SIZE + 1)]
    for lang in ("th", "en", *EXTRA)
}
PRERENDER_FIRST = 4  # clips synthesized ahead per language (bounded TTS cost)
# Status lines ("still working", "sent to the background", "vote sent") are typed in the meeting chat,
# not spoken: voice is for real communication with the people.
STATUS_Q_CHARS = 80  # the question quoted in a status line
FILLER_MEMORY = 3  # the last lines said in this session are not picked again
FILLER_TOOLS = {
    "ask_salid",
    "answer_privately",
}  # tools that take seconds: worth a spoken "one moment"
SPEAKER_NOTE_MIN_S = 1.0  # speaker-change notes are at least this far apart
BURST_GAP_S = 1.0  # a speaker's frames this close together are one stretch of speech
SPEAKER_KEEP_S = 120.0  # how long speech stretches are remembered
ASKER_MAX_AGE_S = 30.0
# Intentional barge-in: the person Salid is answering talks over her for this long (loud frames,
# short gaps allowed) = a real interruption. Other people's side talk and noise never count.
TALKOVER_S = 0.8
TALKOVER_NO_WORDS_S = 1.6  # ... or this long when the Live transcript has no words for it yet
TALKOVER_GAP_S = 0.35
TALKOVER_RMS = 500.0  # the last utterance counts as "who asked" if it ended this recently
ROOM_NOTE = (
    "(ผู้ถามคือ {asker}. คำตอบนี้จะถูกพูดให้ทุกคนในห้องเสียงฟัง: การกระทำ (นัด อีเมล DM โพสต์ อนุมัติ งาน) "
    "ทำในนามผู้ถามด้วยสิทธิ์ของเขา แต่ข้อมูลที่พูดต้องเป็นสิ่งที่ทุกคนในห้องดูได้เท่านั้น. "
    "ถ้าสิ่งที่ถามถูกจำกัดเพราะมีคนในห้องที่ไม่มีสิทธิ์ ให้ตอบสั้น ๆ ว่าพูดตรงนี้ไม่ได้เพราะไม่ใช่ทุกคนในห้องฟังได้ "
    "และเสนอส่งให้ทาง DM; ถ้าผู้ถามตกลงให้ใช้ send_dm ถึง “me”)"
)
ROOM_NOTE_EN = (
    "(The asker is {asker}. This answer will be spoken to everyone in the voice channel: actions "
    "(schedule, email, DM, post, approve, tasks) are done on the asker's behalf with their rights, "
    "but the information spoken must be only what everyone in the room may see. If what was asked is "
    "restricted because someone in the room lacks access, answer briefly that it cannot be said here "
    "because not everyone in the room may hear it, and offer to send it by DM; if the asker agrees "
    "use send_dm to “me”.)"
)
ASK_TIMEOUT_S = 150.0  # default of `Settings.live_ask_timeout_seconds`
STALE_TURN_S = (
    15.0  # a model turn / tool answer older than this no longer blocks a result announcement
)
# The spoken brain has exactly the chat tools (parity test). Work whose first tool call is one of
# these takes tens of seconds (documents, decks, kits, research, PDFs), so the job goes to the
# background at once instead of making the Live model wait for `live_foreground_seconds`.
BACKGROUND_TOOLS = frozenset(
    {
        "deep_research",  # background web research + PDF report
        "make_promo_kit",  # image edit + three post images
        "create_deck",  # slide deck (.pptx)
        "market_benchmark",  # 20-60 s web benchmark + PDF
        "business_framework",  # long framework + PDF
        "idea_export",  # PDF / docx / md of an idea
        "make_quotation",  # quotation PDF + docx
        "revise_quotation",  # new revision of a quotation, PDF + docx
        "financial_projection",  # spreadsheet / PDF
        "meeting_document",  # minutes document file
        "contest_plan",  # plan, optionally PDF
        "contest_pitch",  # pitch scripts
        "wiki_write",  # writes a long page
        "wiki_publish",  # posts a page to a channel
    }
)
ANNOUNCE_AUTH_S = 30.0  # the announcement turn may be heard for this long after the cue is sent


# TODO(web timeout): grounded web search has a fixed 20 s timeout (llm.GROUNDED_TIMEOUT_S); there is
# no per-call knob yet, so the 15 s voice target is not enforced.
ANSWER_MAX_CHARS = 600
LEDGER_NOTE = (
    "(ในห้องเสียงมีผู้ที่ไม่ใช่ผู้บริหาร: ห้ามใช้หรือเปิดเผยข้อมูลบัญชี/ใบเสร็จ/การใช้จ่ายของร้าน "
    "ถ้าถามเรื่องนี้ให้ตอบสั้น ๆ ว่าดูได้เฉพาะเมื่อทุกคนในห้องเป็นผู้บริหาร ถามในแชทแทนได้)"
)

LEDGER_NOTE_EN = (
    "(Someone in the voice channel is not an executive: do not use or reveal the shop's "
    "accounting/receipt/spending data. If asked, answer briefly that it can only be viewed when "
    "everyone in the room is an executive, and suggest asking in chat instead.)"
)

_MEETING_TOOLS = {
    "set_minutes_destination": "set_minutes_destination",
    "post_note": "post_note",
    "ask_consent": "ask_consent",
    "set_recording": "set_recording",
    "set_captions": "set_captions",
}

STATE_LISTENING = "(สถานะ: กำลังฟังอยู่ ยังไม่มีใครคุยกับคุณ จดบันทึกเงียบ ๆ ห้ามพูดจนกว่าจะมีคนเรียกชื่อสลิด)"
STATE_ENGAGED = (
    "(สถานะ: กำลังคุยกับคนที่เรียกคุณอยู่ ตอบเฉพาะประโยคที่พูดกับคุณ ไม่ต้องรอให้เรียกชื่อ "
    "ถ้าเขาบอกโอเค/ต่อไป/พอแล้ว หรือเริ่มคุยกันเอง ให้เงียบแล้วกลับไปฟังโดยไม่พูดอะไร)"
)
FORCE_CUE = "(มีคนเรียกชื่อคุณ: ตอบกลับทันทีตามบริบท)"
STATE_LISTENING_EN = (
    "(State: listening. Nobody is talking to you yet. Take notes silently and say nothing until "
    "someone calls the name Salid.)"
)
STATE_ENGAGED_EN = (
    "(State: talking with the person who called you. Answer only what is said to you, with no need "
    "to hear your name. If they say ok / next / enough, or start talking among themselves, go quiet "
    "and back to listening without saying anything.)"
)
FORCE_CUE_EN = "(Someone called your name: reply at once, in context.)"
HEARD_EN = " What they said: “{heard}”"
RESULT_CUE = (
    "(ผลงานที่ขอไว้เสร็จแล้ว งาน {job} คำถาม: {question} ผลลัพธ์: {answer} — "
    "เริ่มด้วยประโยคสั้น ๆ เช่น “ได้ผลเรื่อง … แล้วค่ะ:” แล้วพูดผลลัพธ์ตามนี้ ห้ามเรียบเรียงใหม่)"
)
RESULT_CUE_EN = (
    "(The requested job is done. Job {job}, question: {question}. Result: {answer} — "
    "open with one short line such as “Here is the result on …:” and say the result as given; "
    "do not rephrase it.)"
)
FAILED_CUE = (
    "(งานที่ขอไว้ งาน {job} คำถาม: {question} ทำไม่สำเร็จ — แจ้งสั้น ๆ ว่าทำให้ไม่ได้ "
    "โดยพูดตามนี้ ห้ามเรียบเรียงใหม่: {answer})"
)
FAILED_CUE_EN = (
    "(The requested job failed. Job {job}, question: {question}. Tell them briefly that it "
    "could not be done, saying this as given and not rephrased: {answer})"
)
NO_ROOM_EN = "Nobody is in the voice channel right now."
QA_OFF_EN = "The team Q&A feature is turned off."

NO_ROOM = "ไม่มีใครอยู่ในห้องเสียงตอนนี้"
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
THAI_MONTHS = [
    "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม",
]  # fmt: skip


def thai_date_line(now) -> str:
    """ "วันนี้คือ วันศุกร์ที่ 16 ตุลาคม 2026": Thai weekday and month, Gregorian year (no Buddhist
    era, which the model would otherwise mix up), for the very top of the Live instruction."""
    return f"วันนี้คือ วัน{THAI_DAYS[now.weekday()]}ที่ {now.day} {THAI_MONTHS[now.month - 1]} {now.year}"


def build_instruction_en(
    settings: Any,
    language: str = "en",
    now=None,
    qa: bool = True,
    history: str = "",
    tone: str = DEFAULT_TONE,
    tone_note: str = "",
    reply_lang: str | None = None,
) -> str:
    """The English system instruction (guild language "en"): English persona, no Thai particles.
    `reply_lang`: a second-tier language (zh, ja...) to speak instead of English.
    `tone` (salid.tone) sets the persona line, the same style as the text answers."""
    now = now or settings.now()
    p = (
        f"Today is {now:%A, %B} {now.day}, {now.year}.\n"
        "You are “Salid”, the team's note-taker sitting in the team's Discord voice channel. Your base "
        "job is to LISTEN and take the minutes quietly; you are not a chat companion. You speak only "
        "when someone calls your name, then you talk with that person (and whoever joins in). Do not "
        "lecture, do not start conversations and do not recite rules or policies.\n"
        "Speaking rules:\n"
        "- There are 2 states; the system tells you the state in a bracketed message: (1) Listening "
        "(default): you listen and take notes silently and never speak, except when someone calls "
        "the name “Salid” (it may be heard as Salit, Saleed or Sa-lid): then answer "
        "briefly at once, and if it is a greeting, greet them back. (2) In conversation: talk with "
        "the person who called you and anyone who turns to you; answer what is said to you, with no "
        "need for the name, and never ask chit-chat questions yourself.\n"
        "- When someone says ok / thanks / next person / moving on / that is enough, or people start "
        "talking among themselves or to someone else about something else, go back to listening "
        "WITHOUT saying anything (say “Sure.” only if they thanked you directly).\n"
        "- If people in the room are talking among themselves, stay completely silent: do not reply "
        "or comment, even if the topic concerns the team or the past (for example “where did we eat "
        "yesterday”), unless they call your name or turn to ask you.\n"
        "- A word that sounds like your name but sits mid-sentence and is not addressing you "
        "(for example “Salad”, “Siri”, “slip”) is not a call: stay quiet.\n"
        "- Answer in 1-3 short sentences, natural like real talk. Never read long lists aloud.\n"
        f"- {voice_line(tone, 'en')}\n"
        "- When you join the room, do not say anything yet: wait until someone greets you or talks to you. "
        "If nobody calls your name, the system never plays your voice at all, so do not speak on your own "
        "when you join or when you hear someone unmute their mic / background noise.\n"
        "- If it is unclear whether this is a meeting (no schedule, relaxed chat) and nobody has talked "
        "to you for a while, you may ask like a colleague, e.g. “Is this a meeting today?”, by calling "
        "ask_consent (text by default so it does not disturb; spoken only when someone is talking "
        "to you). If the answer is that it is not a meeting / no notes, call "
        "set_recording(false), say briefly that you have stopped taking notes, and stay quiet until "
        "called. If it is a meeting, keep taking notes.\n"
        "- If someone tells you to leave the room / end the meeting (“leave”, “you can go”, "
        "“end the meeting”), the system handles leaving itself: do not call set_recording and do not "
        "say you stopped taking notes. Say one short goodbye, such as “See you later!”, then stay quiet.\n"
        "- Talk to people by voice. The system also posts your answers to the text chat, so do not "
        "post them yourself. For what truly "
        "deserves a note (decisions, action items with owner and deadline, numbers someone asks you to "
        "remember) call post_note with one short line per topic, not every sentence.\n"
        "- Live captions are always off until someone asks: if asked to turn on captions / translate, "
        "call set_captions(on=true, language) (original/th/en); if asked to turn them off call "
        "set_captions(on=false) and confirm briefly.\n"
        "- If someone says where to send the decisions/meeting summary, call the tool "
        "set_minutes_destination and confirm briefly.\n"
        f"Today is {now.date().isoformat()} ({now:%A}), time {now:%H:%M}, timezone {settings.tz}. "
        "If someone asks what day today/tomorrow/yesterday is, answer only from this date; never guess "
        "from memory. The week starts on Monday. Use this date to turn “this month”, “last week” etc. "
        "into YYYY-MM-DD ranges when calling tools.\n"
        "The team's stored memory and tool results may contain Thai: translate it and always speak "
        "English.\n"
    )
    if qa:
        p += (
            "The only information tool is ask_salid (the full assistant that knows the team's memory "
            "and can search the web).\n"
            "- General knowledge questions, theory, opinions, explanations, chit-chat, greetings: answer "
            "directly yourself and do not call a tool.\n"
            "- Call ask_salid only when the question needs the team's data (who said what / decisions, "
            "tasks and owners, files, receipts/expenses, the schedule) or fresh information from the "
            "web, then say the ask_salid answer as it is: shorten only by cutting sentences, never "
            "rewrite it, keep every number, name and date. Never guess or compute numbers yourself.\n"
            "- Never bring up money, receipts or expenses yourself unless someone asks about them directly.\n"
            "- The system sends bracketed messages saying who is speaking right now (for example "
            "“(Now speaking: Mew)”). Use them to address people naturally (for example “Sure, Mew”). "
            "Never read these messages aloud and never answer them.\n"
            "- If ask_salid says it cannot be said here because not everyone in the room may hear it, "
            "tell the asker that briefly and offer to send it by DM; if they agree (or ask for a private "
            "answer) call answer_privately(question), then say briefly that you sent it by DM. Never "
            "guess or say the content yourself.\n"
            "- Do not say a filler yourself before or while calling ask_salid (such as “one moment”): "
            "the system says it for you. Call the tool at once without speaking, then answer when you "
            "have the result.\n"
            "- If a tool result has an error and a “say” field, read “say” as it is (for example “Sorry, "
            "I can't check that right now. The lookup tool is not working.”); never invent an answer.\n"
            "- ask_salid is also how you DO things (documents, decks, polls, events, messages): it has "
            "every tool the text assistant has.\n"
            "- ask_salid may answer at once, or reply status=working: the job runs in the background and "
            "the system has already typed a note in the text chat, so say nothing about it (at most "
            "“Okay” if someone is clearly waiting for an answer), do not wait, and go on with the "
            "conversation (other questions are fine; the job keeps running). If status=working comes "
            "with a “say” line (the same job asked again), say that short line as it is. When the "
            "system later sends “(The requested job is done ...)”, announce it briefly and say the "
            "result as given. If a file was made, the answer says it was put in the text chat.\n"
        )
    else:
        p += (
            "You cannot look up the team's data right now. If someone asks about team data, money or "
            "receipts, say plainly that you cannot look it up at the moment; you can still answer general "
            "knowledge yourself.\n"
        )
    if reply_lang in EXTRA:
        p += (
            f"Language: always speak {lang_name(reply_lang)} ({lang_name(reply_lang, 0)}); "
            "keep names and vendors as they are. Polite, natural register for that language. "
            "Never use Thai particles.\n"
        )
    elif language == "auto":
        p += "Language: reply in the language the speaker uses; default to English. Never use Thai particles.\n"
    else:
        p += "Language: always speak English (keep names and vendors as they are).\n"
    if note := voice_note(tone_note, "en"):
        p += note + "\n"
    if history:
        p += (
            "\nContext: what people in the room just said (reference only, do not answer it):\n"
            + history
            + "\n"
        )
    if reply_lang in EXTRA:
        p = p.replace(
            "Speak plain natural English only.", f"Speak plain, natural {lang_name(reply_lang)}."
        )
    return p


def build_instruction(
    settings: Any,
    language: str = "th",
    now=None,
    qa: bool = True,
    history: str = "",
    lang: str | None = None,
    tone: str = DEFAULT_TONE,
    tone_note: str = "",
) -> str:
    """System instruction for the Live session (built per connect so the date is current). Thai by
    default; English persona and rules when the session's resolved `lang` (or `language`) is en.
    `tone`: the salid.tone preset of the meeting's text channel (persona line, TH and EN)."""
    reply = lang or language
    if reply in EXTRA:
        return build_instruction_en(settings, language, now, qa, history, tone, tone_note, reply)
    if lang == "en" or language == "en":
        return build_instruction_en(settings, language, now, qa, history, tone, tone_note)
    now = now or settings.now()
    p = (
        f"{thai_date_line(now)}\n"
        "คุณชื่อ “สลิด” (Salid) เป็นผู้จดบันทึกประจำทีม นั่งอยู่ในห้องเสียง Discord ของทีม งานหลักคือฟังและจดบันทึกการประชุมเงียบ ๆ "
        "ไม่ใช่เพื่อนคุย คุณพูดเมื่อมีคนเรียกชื่อเท่านั้น แล้วคุยกับคนที่เรียกคุณ (และคนที่หันมาคุยด้วย) "
        "ไม่สั่งสอน ไม่ชวนคุยเอง ไม่อ่านกฎหรือนโยบายให้ใคร\n"
        "กติกาการพูด:\n"
        "- มี 2 สถานะ ระบบจะบอกสถานะให้เป็นข้อความในวงเล็บ: (1) กำลังฟัง (ค่าเริ่มต้น) คุณฟังและจดเงียบ ๆ "
        "ห้ามพูดเด็ดขาด ยกเว้นเมื่อมีคนเรียกชื่อ “สลิด” (Salid, สะลิด, สลิต, ซาลิด) ให้ตอบทันทีสั้น ๆ โดยเฉพาะถ้าเป็นการทักทาย ให้ทักกลับ "
        "(2) กำลังคุยอยู่ คุยกับคนที่เรียกคุณและคนที่หันมาคุยกับคุณ ตอบเฉพาะประโยคที่พูดกับคุณ ไม่ต้องรอชื่อ ห้ามชวนคุยเอง\n"
        "- เมื่อมีคนพูดว่า โอเค ขอบคุณ คนต่อไป ต่อไป พอแล้ว หรือเริ่มคุยกันเอง/คุยกับคนอื่นเรื่องอื่น "
        "ให้กลับไปฟังเงียบ ๆ โดยไม่พูดอะไรเลย (ตอบ “ได้ค่ะ” สั้น ๆ เฉพาะเมื่อเขาขอบคุณคุณโดยตรง)\n"
        "- ถ้าคนในห้องคุยกันเอง ให้เงียบสนิท ไม่ต้องตอบหรือแสดงความเห็น แม้เรื่องที่คุยจะเกี่ยวกับทีมหรืออดีต "
        "(เช่น “เมื่อวานกินข้าวที่ไหน”) เว้นแต่เขาเรียกชื่อคุณหรือหันมาถามคุณ\n"
        "- คำที่เสียงคล้ายชื่อคุณแต่อยู่กลางประโยคและไม่ใช่การเรียกคุณ (เช่น “สนิท” ใน “เขาสนิทกันมาก”, “สลิป”, “สิริ”) "
        "ไม่ใช่การเรียกชื่อ ให้เงียบ\n"
        "- ตอบสั้น 1-3 ประโยค เป็นธรรมชาติเหมือนคุยกันจริง ไม่อ่านเป็นรายการยาว ๆ\n"
        f"- {voice_line(tone, 'th')}\n"
        "- ตอนเข้าห้องอย่าเพิ่งพูดอะไร รอจนมีคนทักหรือคุยกับคุณ ถ้าไม่มีใครเรียกชื่อคุณ ระบบจะไม่ส่งเสียงของคุณออกไปเลย "
        "ดังนั้นอย่าพูดเองตอนเข้าห้องหรือตอนได้ยินคนเปิดไมค์ / เสียงรบกวน\n"
        "- ถ้าไม่ชัดว่านี่คือการประชุมไหม (ไม่มีนัดหมาย คุยกันสบาย ๆ) และไม่มีใครคุยกับคุณมาสักพัก คุณถามได้เองแบบเพื่อนร่วมงาน "
        "เช่น “วันนี้ประชุมกันหรือเปล่าคะ” โดยเรียก ask_consent (เป็นข้อความโดยปริยาย จะได้ไม่รบกวน; พูดเฉพาะเมื่อมีคนคุยกับคุณอยู่) "
        "ถ้าตอบว่าไม่ใช่การประชุม/ไม่ต้องจด ให้เรียก set_recording(false) แล้วพูดสั้น ๆ ว่าไม่จดแล้ว และเงียบรอให้เรียก "
        "ถ้าเป็นการประชุมให้จดต่อ\n"
        "- ถ้ามีคนบอกให้คุณออกจากห้อง/ออกประชุม/เลิกประชุม (“ออกประชุม”, “ออกไปก่อน”, “leave”) ระบบจัดการการออกเอง: "
        "ห้ามเรียก set_recording และห้ามพูดว่า “ไม่จดแล้ว” ให้พูดลาสั้น ๆ ประโยคเดียว เช่น “แล้วเจอกันนะคะ” แล้วเงียบ\n"
        "- คุยกับคนด้วยเสียง ระบบโพสต์คำตอบของคุณลงแชทข้อความให้เองอยู่แล้ว ห้ามโพสต์ซ้ำ ส่วนสิ่งที่ควรจดจริง ๆ (มติ งานที่ต้องทำพร้อมเจ้าของและกำหนดเวลา "
        "ตัวเลขที่มีคนขอให้จำ) ให้เรียก post_note บรรทัดสั้น ๆ ต่อหนึ่งเรื่อง ไม่ใช่ทุกประโยค\n"
        "- ซับไตเติล (คำบรรยายสด) ปิดอยู่เสมอจนกว่าจะมีคนขอ: ถ้าขอเปิดซับ/แปลให้ด้วย เรียก set_captions(on=true, language) "
        "(original/th/en) ถ้าขอปิดซับเรียก set_captions(on=false) แล้วยืนยันสั้น ๆ\n"
        "- ถ้ามีคนบอกให้ส่งมติ/สรุปการประชุมไปห้องไหน ให้เรียกเครื่องมือ set_minutes_destination แล้วยืนยันสั้น ๆ\n"
        f"วันนี้คือ {now.date().isoformat()} (วัน{THAI_DAYS[now.weekday()]}) เวลา {now:%H:%M} เขตเวลา {settings.tz}. "
        "ถ้ามีคนถามว่าวันนี้/พรุ่งนี้/เมื่อวานวันอะไร ให้ตอบจากวันที่นี้เท่านั้น ห้ามเดาจากความจำ. "
        "สัปดาห์เริ่มวันจันทร์. ใช้วันที่นี้แปลง “เดือนนี้”, “สัปดาห์ก่อน” ฯลฯ เป็นช่วงวันที่ YYYY-MM-DD ตอนเรียกเครื่องมือ\n"
    )
    if qa:
        p += (
            "เครื่องมือข้อมูลมีอันเดียวคือ ask_salid (ผู้ช่วยเต็มรูปแบบที่รู้ความจำของทีมและค้นเว็บได้)\n"
            "- คำถามความรู้ทั่วไป ทฤษฎี ความเห็น การอธิบาย คุยเล่น ทักทาย: ตอบเองตรง ๆ ห้ามเรียกเครื่องมือ\n"
            "- เรียก ask_salid เฉพาะเมื่อคำถามต้องใช้ข้อมูลของทีม (ใครพูดอะไร/มติ งานและเจ้าของ ไฟล์ ใบเสร็จ/ค่าใช้จ่าย "
            "ตารางนัด) หรือข้อมูลล่าสุดจากเว็บ แล้วพูดคำตอบของ ask_salid ตามที่ได้มา: ย่อได้เฉพาะการตัดประโยคออก "
            "ห้ามเรียบเรียงใหม่ ต้องคงตัวเลข ชื่อ และวันที่ครบ ห้ามเดาหรือคำนวณตัวเลขเอง\n"
            "- ห้ามพูดถึงเงิน ใบเสร็จ หรือค่าใช้จ่ายเองเด็ดขาด เว้นแต่มีคนถามเรื่องนั้นโดยตรง\n"
            "- ระบบจะส่งข้อความในวงเล็บบอกว่าตอนนี้ใครกำลังพูด (เช่น “(ตอนนี้ คุณมิว กำลังพูด)”) ใช้เรียกชื่อคนที่คุยด้วยให้เป็นธรรมชาติ "
            "(เช่น “ได้ค่ะคุณมิว”) ห้ามอ่านข้อความเหล่านี้ออกเสียงและห้ามตอบมัน\n"
            "- ถ้า ask_salid บอกว่าพูดตรงนี้ไม่ได้เพราะไม่ใช่ทุกคนในห้องฟังได้ ให้บอกผู้ถามสั้น ๆ ตามนั้นและเสนอส่งให้ทาง DM "
            "ถ้าเขาตกลง (หรือขอคำตอบแบบส่วนตัว) เรียก answer_privately(question) แล้วบอกสั้น ๆ ว่าส่ง DM ให้แล้ว ห้ามเดาหรือพูดเนื้อหาเอง\n"
            "- ห้ามพูดคำเติมเองก่อนหรือระหว่างเรียก ask_salid (เช่น “แป๊บนึงนะคะ”) ระบบพูดให้แล้ว ให้เรียกเครื่องมือทันทีโดยไม่พูดอะไร "
            "แล้วค่อยตอบเมื่อได้ผลลัพธ์\n"
            "- ถ้าผลลัพธ์ของเครื่องมือมี error และ say ให้อ่าน say ตามที่เป็น (เช่น “ขออภัยค่ะ ตรวจสอบให้ไม่ได้ตอนนี้ "
            "เครื่องมือค้นข้อมูลขัดข้องค่ะ”) ห้ามแต่งคำตอบเอง\n"
            "- ask_salid ใช้ลงมือทำงานได้ด้วย (เอกสาร สไลด์ โพล นัดหมาย ส่งข้อความ) เพราะมีเครื่องมือครบเท่าผู้ช่วยในแชท\n"
            "- ask_salid อาจตอบทันที หรือตอบ status=working: งานทำอยู่เบื้องหลังและระบบพิมพ์โน้ตในแชตให้แล้ว "
            "ดังนั้นไม่ต้องพูดถึงมัน (อย่างมากพูดแค่ “ค่ะ” ถ้ามีคนรอคำตอบอยู่ชัด ๆ) ไม่ต้องรอ "
            "แล้วคุยเรื่องอื่นต่อได้เลย (ถามเรื่องอื่นเพิ่มได้ งานยังทำอยู่) ถ้า status=working มี “say” มาด้วย "
            "(ถามงานเดิมซ้ำ) ให้พูดประโยคสั้น ๆ นั้นตามที่เป็น เมื่อระบบส่ง "
            "“(ผลงานที่ขอไว้เสร็จแล้ว ...)” มาภายหลัง ให้ประกาศสั้น ๆ แล้วพูดผลลัพธ์ตามที่ได้ ห้ามเรียบเรียงใหม่ "
            "ถ้ามีไฟล์ คำตอบจะบอกว่าส่งไว้ในแชตแล้ว\n"
        )
    else:
        p += "ตอนนี้คุณดูข้อมูลของทีมไม่ได้ ถ้ามีคนถามเรื่องข้อมูลทีม เงิน หรือใบเสร็จให้บอกตรง ๆ ว่าตอนนี้ดูไม่ได้ ส่วนความรู้ทั่วไปตอบเองได้\n"
    if language == "en":
        p += "Language: always speak English (keep Thai names and vendors as they are).\n"
    elif language == "auto":
        p += "ภาษา: ตอบด้วยภาษาเดียวกับคนที่พูดกับคุณ (ไทยหรืออังกฤษ)\n"
    else:
        p += "ภาษา: พูดภาษาไทย\n"
    if note := voice_note(tone_note, "th"):
        p += note + "\n"
    if history:
        p += "\nบริบท: สิ่งที่คนในห้องเพิ่งคุยกัน (ใช้อ้างอิงเท่านั้น ไม่ต้องตอบ):\n" + history + "\n"
    return p


def _norm_question(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text or "").lower())


def same_question(a: str, b: str) -> bool:
    """The same question asked again (re-asked while waiting): normalised, one contains the
    other or they are at least 80% similar."""
    a, b = _norm_question(a), _norm_question(b)
    if not a or not b:
        return False
    return a in b or b in a or difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


class _Job:
    """One ask_salid job: it runs to the end whatever the room does. `background`: a known-long tool
    was called (hand it off at once); `detached`: the Live model was told "working" (the result is
    announced when it finishes)."""

    def __init__(self, question: str):
        self.id = secrets.token_hex(2)
        self.question = question
        self.task: asyncio.Task | None = None
        self.background = asyncio.Event()
        self.detached = False
        self.done = False
        self.result: dict | None = None


_JOB: contextvars.ContextVar[_Job | None] = contextvars.ContextVar("salid_voice_job", default=None)


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
        echo: Callable[[str], Any] | None = None,
    ):
        """`echo(text)`: posts to the meeting's text chat always (answers and failures are shown
        there even when `post`, the optional chatter, is off); default: `post`."""
        self.bot, self.session, self.post = bot, session, post
        self.echo = echo or post
        self._jobs: dict[str, _Job] = {}  # ask_salid jobs still running (`live_max_jobs`)
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
        self._muted_turn = False  # someone said "stop": drop the rest of this model turn
        self._turn_open = False  # the server is still sending the current model turn
        self._turn_open_since = 0.0  # monotonic: when the current model turn began
        self._awaiting_since = 0.0  # monotonic: when a tool call started waiting for its answer
        self._partners: set[int] = set()  # who Salid is answering (talking over her stops her)
        self._talkover: dict[int, list[float]] = {}  # partner id -> [start, last] loud speech
        self._heard_while_speaking = ""
        self._last_audio_at = 0.0  # when the model last produced audio, or a cue was sent
        self._real_audio_at = 0.0  # when the model last really produced audio (monotonic)
        self._engaged = False
        self._last_engaged_at = -1e9  # monotonic: engaged last (grace window for the gates)
        self._frames: list[tuple[float, float]] = []  # (arrival time, RMS) of the last 3 s
        self._rejected_logged = False
        # mouth gate (see `_gate_audio`)
        self._held: list[
            bytes
        ] = []  # audio of a not-yet-authorised turn, kept for the grace window
        self._held_since = 0.0  # monotonic time the first held chunk arrived
        self._turn_ok = False  # this model turn is authorised to be heard
        self._answer_auth = False  # the last tool call was made while engaged
        self._announce_until = 0.0  # monotonic: a job result was cued, its audio is authorised
        self._muted_logged = False  # the "muted" line was logged for this turn
        self._gated_turn = False  # this turn had audio the gate held or dropped
        # name heard in the Live input transcript (fast path)
        self._heard_buf = ""
        self._heard_at = 0.0
        self._heard_hit = False
        self._live_hit_at = -1e9
        self._fallback: tuple[int, str] | None = None  # who/what to answer if the model stays mute
        # filler clips ("one moment") for slow tools
        self._filler_pcm: dict[str, bytes] = {}
        self._filler_used: list[str] = []  # the last lines said (generated or fallback)
        self._filler_until = 0.0  # monotonic: filler audio may still sit in the playback buffer
        self._filler_task: asyncio.Task | None = None
        self._filler_question = ""
        # who is talking (Discord gives one stream per user)
        self._bursts: dict[int, list] = {}  # user id -> its open [uid, start, end]
        self._spans: list[list] = []  # [uid, start, end] stretches of speech, oldest first
        self._noted_uid: int | None = None  # the speaker the Live model was last told about
        self._note_at = -1e9

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
            now = time.monotonic()
            self._track_speaker(user_id, now)
            self._check_talkover(user_id, pcm, now)
            self._frames.append((now, rms(pcm)))  # see `_speech_stats`
            if len(self._frames) > 400:  # ~8 s of frames: keep the ring short
                self._frames = [f for f in self._frames if now - f[0] <= ASK_EVIDENCE_S]

    def _speech_stats(self, now: float, window: float = HEARD_GAP_S) -> SpeechStats:
        """Energy statistics of the frames received in the last `window` s (3 by default), any speaker."""
        self._frames = [f for f in self._frames if now - f[0] <= ASK_EVIDENCE_S]
        floor = float(getattr(getattr(self.bot, "s", None), "name_min_rms", 250.0))
        return stats_from_frame_rms([r for t, r in self._frames if now - t <= window], floor)

    def _check_talkover(self, uid: int, pcm: bytes, now: float) -> None:
        """Stop when the person Salid is answering clearly talks over her. Anyone else (a side
        conversation, a TV, a noisy mic) is ignored; they can still say "สลิด หยุด" / "stop"."""
        if uid not in self._partners or not self._speaking():
            self._talkover.pop(uid, None)
            return
        if not intentional_barge_in(self.bot.s) or rms(pcm) < TALKOVER_RMS:
            return
        cur = self._talkover.get(uid)
        if cur is None or now - cur[1] > TALKOVER_GAP_S:
            cur = self._talkover[uid] = [now, now]
        cur[1] = now
        words = len(self._heard_while_speaking.strip()) >= 2
        if cur[1] - cur[0] >= (TALKOVER_S if words else TALKOVER_NO_WORDS_S):
            self._talkover.clear()
            self._stop_requested()

    # ---- who is talking / who asked
    def _track_speaker(self, uid: int, now: float) -> None:
        cur = self._bursts.get(uid)
        if cur is not None and now - cur[2] < BURST_GAP_S:
            cur[2] = now
        else:
            cur = self._bursts[uid] = [uid, now, now]
            self._spans.append(cur)
            while self._spans and self._spans[0][2] < now - SPEAKER_KEEP_S:
                self._spans.pop(0)
        if (
            uid != self._noted_uid
            and now - self._note_at >= SPEAKER_NOTE_MIN_S
            and self._live is not None
        ):
            self._noted_uid, self._note_at = uid, now
            self._spawn(self._send_speaker(uid))

    def speaker_name(self, uid: int) -> str:
        s = self.session
        guild = s.voice_channel.guild
        try:
            nick = self.bot._profile_of(s.guild_id, uid).nickname
        except Exception:
            nick = ""
        member = guild.get_member(uid)
        return nick or (member.display_name if member else "")

    async def _send_speaker(self, uid: int) -> None:
        live, name = self._live, self.speaker_name(uid)
        if live is None or self._closing or not name:
            return
        note = f"(Now speaking: {name})" if self._en() else f"(ตอนนี้ {name} กำลังพูด)"
        await live.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=note)]), turn_complete=False
        )

    def recent_speakers(self, now: float | None = None) -> list[int]:
        """Users whose voice was active in the last utterance (the one that ended most recently,
        and everyone who overlapped it). [] when nobody spoke lately."""
        now = time.monotonic() if now is None else now
        spans = [b for b in self._spans if b[2] >= now - ASKER_MAX_AGE_S]
        if not spans:
            return []
        last = max(spans, key=lambda b: b[2])
        return sorted({b[0] for b in spans if b[1] <= last[2] and b[2] >= last[1]})

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

    def _tone_kw(self) -> dict:
        """The tone of the meeting's text channel (so the voice sounds like the chat). Only read at
        connect: a tone changed mid-meeting reaches the spoken answers of `ask_salid` at once and
        the persona line at the next (re)connect."""
        sess = self.session
        try:
            cid = sess.minutes_channel_id or getattr(sess.text_channel, "id", 0)
            g = self.bot.team.current(sess.guild_id)
            return {"tone": effective_tone(g, cid), "tone_note": g.tone_note}
        except Exception:
            return {}

    async def build_config(self) -> types.LiveConnectConfig:
        s, sess = self.bot.s, self.session
        qa = await self._qa_enabled()
        fns = [set_minutes_destination, post_note, ask_consent, set_recording, set_captions]
        if qa:
            fns = [ask_salid, answer_privately, *fns]
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
            system_instruction=build_instruction(
                s, sess.language, s.now(), qa=qa, history=history, lang=sess.lang, **self._tone_kw()
            ),
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
        self._noted_uid = None  # a new session knows nobody yet: tell it who is talking
        if self._engaged:
            self._spawn(self._send_state())
        if getattr(self.bot.s, "live_filler", True) and not self._filler_pcm:
            self._spawn(self._prerender_fillers())
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
        if engaged:
            self._last_engaged_at = time.monotonic()
            self._release_held()  # the name arrived while the first words were being held back
        self._spawn(self._send_state())

    async def _send_state(self) -> None:
        live = self._live
        if live is None or self._closing:
            return  # the current state is sent again when the session connects
        if self._en():
            text = STATE_ENGAGED_EN if self._engaged else STATE_LISTENING_EN
        else:
            text = STATE_ENGAGED if self._engaged else STATE_LISTENING
        await live.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=text)]), turn_complete=False
        )

    def ensure_reply(
        self,
        since: float,
        heard: str = "",
        *,
        user_id: int | None = None,
        question: str = "",
    ) -> None:
        """Someone said the name (the utterance ended at `since`, monotonic). If the model has not
        started answering `live_force_reply_seconds` after that, force a turn with a cue; no audio
        `live_cue_retry_seconds` after the cue: cue once more; still nothing after another
        `live_fallback_seconds`: the classic path (agent + TTS) answers `question` instead.

        The Live input transcript usually sees the name first (`_on_heard`); the classic hit that
        follows for the same utterance must not force a second turn, but it does tell the running
        one who to answer through the classic path if the model stays mute."""
        now = time.monotonic()
        if now - self._live_hit_at < NAME_DEDUPE_S and since - self._live_hit_at < 3.0:
            if user_id is not None and (question or heard).strip():
                self._fallback = (user_id, question or heard)
            return
        self._spawn(self._force_reply(since, heard, user_id, question or heard))

    # ---- the name in the Live session's own input transcript (arrives while the person talks)
    def _on_heard(self, text: str) -> None:
        now = time.monotonic()
        if now - self._heard_at > HEARD_GAP_S:
            self._flush_heard()
        self._heard_at = now
        self._heard_buf += text
        if self._speaking():
            self._heard_while_speaking += text
        if (
            self._speaking()
            and intentional_barge_in(self.bot.s)
            and is_stop_request(self._heard_buf)
        ):
            self._stop_requested()
            if not parse_wake(self._heard_buf):
                return  # "หยุด" / "stop": just stop; a name call goes on to be answered
        if self._heard_hit:
            return
        said = parse_wake(self._heard_buf)  # the same detector as the classic path
        if said is None:
            return
        if float(getattr(self.bot.s, "name_min_voiced_seconds", 0.35)) > 0:
            stats = self._speech_stats(now)
            ok, reason = looks_like_speech(stats, self.bot.s)
            if not ok:
                # noise / an unmuted mic hallucinated into a greeting: not a call (a later part of
                # the same utterance with real speech can still be one)
                if not self._rejected_logged:
                    self._rejected_logged = True
                    logger.info("live: name hit rejected: %s %s", reason, stats_line(stats))
                return
            logger.info("live: name hit accepted %s", stats_line(stats))
        self._last_engaged_at = now
        self._heard_hit = True
        if parse_meeting_command(said) == "leave":
            return  # "สลิด ออกประชุม": the meeting manager leaves; no reply to force
        s = self.session
        if s.ending or s.leaving:
            return
        self._live_hit_at = now
        self._fallback = None
        logger.info("live: name heard in the live transcript")
        who = self.recent_speakers()
        self.bot.meetings._set_engaged(
            s,
            True,
            name=self.speaker_name(who[0]) if len(who) == 1 else "",
            by=who[0] if len(who) == 1 else 0,
        )
        self._spawn(
            self._force_reply(
                now,
                self._heard_buf,
                None,
                said or self._heard_buf,
                wait=float(getattr(self.bot.s, "live_name_reply_seconds", 1.5)),
                track_voice=True,
            )
        )

    def _flush_heard(self) -> None:
        """The utterance is over: a thank-you / "that's enough" said to Salid while it is engaged
        puts it back to listening (like the classic path), then start a fresh buffer."""
        text, hit = self._heard_buf.strip(), self._heard_hit
        self._heard_buf, self._heard_hit = "", False
        self._rejected_logged = False
        if text and not hit and self._engaged:
            mgr = self.bot.meetings
            if mgr.is_dismissal(text):
                mgr._set_engaged(self.session, False)

    def _answering(self, since: float) -> bool:
        return (
            self._real_audio_at >= since
            or self._last_audio_at >= since
            or self._turn_audio
            or self._awaiting_answer
        )

    async def _cue(self, live: Any, heard: str) -> None:
        en = self._en()
        cue = FORCE_CUE_EN if en else FORCE_CUE
        if heard.strip():  # the model may have dropped the audio: say what was heard (transcript)
            if en:
                cue += HEARD_EN.format(heard=heard.strip()[:300])
            else:
                cue += f" ข้อความที่เขาพูดคือ: “{heard.strip()[:300]}”"
        await live.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=cue)]),
            turn_complete=True,
        )

    async def _force_reply(
        self,
        since: float,
        heard: str = "",
        user_id: int | None = None,
        question: str = "",
        *,
        wait: float | None = None,
        track_voice: bool = False,
    ) -> None:
        sets = self.bot.s
        if wait is None:
            wait = float(getattr(sets, "live_force_reply_seconds", 1.2))
        while True:  # `track_voice`: the person may still be talking when the name is seen
            delay = max(since, self._last_voice if track_voice else 0.0) + wait - time.monotonic()
            if delay <= 0:
                break
            await asyncio.sleep(delay)
        live = self._live
        if live is None or self._closing or self.session.ending:
            return
        if self._answering(since):
            return  # the model is answering (or answered, or waiting for a tool result) already
        logger.info("live: name heard but no reply: forcing a turn")
        cue_at = time.monotonic()
        self._last_audio_at = cue_at  # one cue per call
        await self._cue(live, heard)
        await asyncio.sleep(float(getattr(sets, "live_cue_retry_seconds", 6.0)))
        live = self._live
        if live is None or self._closing or self.session.ending or self.session.leaving:
            return
        if self._real_audio_at >= cue_at or self._turn_audio or self._awaiting_answer:
            return
        logger.info("live: still no audio after the cue: cueing once more")
        cue_at = time.monotonic()
        await self._cue(live, heard)
        await asyncio.sleep(float(getattr(sets, "live_fallback_seconds", 4.0)))
        if self._closing or self.session.ending or self.session.leaving:
            return
        if self._real_audio_at >= cue_at or self._turn_audio or self._awaiting_answer:
            return
        if user_id is None and self._fallback is not None:
            user_id, question = self._fallback
        if user_id is None or not question.strip():
            return
        logger.warning("live: no audio from the model: answering through the classic path")
        self._spawn(self.bot.meetings.live_fallback(self.session, user_id, question))

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
            if not self._turn_open:
                self._turn_open_since = time.monotonic()
            self._turn_open = True
        if sc.model_turn is not None and not self._muted_turn:
            for part in sc.model_turn.parts or []:
                data = part.inline_data.data if part.inline_data is not None else None
                if data and not self._muted_turn:  # a gate drop mutes the rest of the message too
                    self._gate_audio(data)
        it = sc.input_transcription
        if it is not None and it.text:
            self._on_heard(it.text)
        ot = sc.output_transcription
        if ot is not None and ot.text:
            self._said.append(ot.text)
        if sc.turn_complete:
            self._flush_heard()
            self._turn_complete()
        return None

    # ---- tools
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
            if name == "set_captions":
                lang = normalize_language(a.get("language", "original"))
                if lang is None:
                    return {"error": "language must be original, th or en"}
                return mgr.set_captions(self.session, bool(a.get("on", False)), lang)
            return mgr.set_recording(self.session, bool(a.get("recording", True)))

        if name not in ("ask_salid", "answer_privately"):
            return {"error": f"unknown tool {name}"}
        question = str((args or {}).get("question", ""))
        if name == "answer_privately":
            return await self._guarded(name, self._answer_privately, question)
        return await self._one_brain(question)

    # ---- jobs: ask_salid answers at once or keeps working in the background
    async def _one_brain(self, question: str) -> dict:
        """ask_salid as a job. It answers within `live_foreground_seconds` (quick Q&A stays
        conversational), or at once with "working" when its first tool call is a known-long one
        (`BACKGROUND_TOOLS`); otherwise the job runs on and `_announce` tells the room at the next
        pause. The hand-off is a typed note in the meeting chat, never spoken. The same question
        while it runs gets "still working" (a note + one short spoken sentence); a different one is a new job
        (at most `live_max_jobs`). Nothing here cancels a job: not a new question, not barge-in."""
        sets, lang = self.bot.s, self._filler_lang()
        running = [j for j in self._jobs.values() if not j.done]
        for j in running:
            if same_question(j.question, question):
                logger.info("live: ask_salid asked again, job %s still running", j.id)
                self._echo_text("voice_still_working_text", question=status_question(question))
                return {
                    "status": "working",
                    "job": j.id,
                    "say": catalog("voice_still_working", lang),  # answers the person: spoken
                }
        if len(running) >= int(getattr(sets, "live_max_jobs", 3)):
            logger.info("live: ask_salid refused: %d jobs running", len(running))
            return {"error": "too many jobs running", "say": catalog("voice_jobs_full", lang)}
        job = _Job(question)
        self._jobs[job.id] = job
        job.task = asyncio.create_task(self._job_main(job))
        self._bg.add(job.task)
        job.task.add_done_callback(self._bg.discard)
        waiter = asyncio.create_task(job.background.wait())
        try:
            await asyncio.wait(
                {job.task, waiter},
                timeout=float(getattr(sets, "live_foreground_seconds", 12.0)),
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            waiter.cancel()
        if job.done and job.result is not None:
            return job.result
        job.detached = True  # from here the result is announced, not returned
        logger.info("live: job %s goes to the background", job.id)
        self._echo_text("voice_handoff_text", question=status_question(question))
        return {"status": "working", "job": job.id}  # no `say`: the chat note is the status

    async def _job_main(self, job: _Job) -> dict:
        """Runs the brain with the time limit; a finished background job is announced."""
        _JOB.set(job)
        res: dict = {}
        try:
            res = await self._guarded("ask_salid", self._ask_salid, job.question)
        finally:
            job.result, job.done = res, True
            self._jobs.pop(job.id, None)
        if job.detached:
            self._spawn(self._announce(job))
        return res

    def _quiet(self, pause: float) -> bool:
        """Nobody has spoken for `pause` seconds and Salid is neither speaking nor about to."""
        now = time.monotonic()
        floor = float(getattr(self.bot.s, "name_min_rms", 250.0))
        # only voiced frames count as "someone is talking": an open mic sends silence/noise frames
        last_room = max(
            self._last_voice, max((t for t, r in self._frames if r >= floor), default=0.0)
        )
        # a turn / tool answer the model never finished must not block the announcement forever
        turn_stuck = self._turn_open and now - self._turn_open_since > STALE_TURN_S
        answer_stuck = self._awaiting_answer and now - self._awaiting_since > STALE_TURN_S
        return (
            now - last_room >= pause
            and self._live is not None
            and not self._speaking()
            and not self._turn_audio
            and (not self._turn_open or turn_stuck)
            and (not self._awaiting_answer or answer_stuck)
            and now >= self._filler_until
        )

    async def _wait_for_pause(self) -> bool:
        sets = self.bot.s
        pause = float(getattr(sets, "live_result_pause_seconds", 1.5))
        end = time.monotonic() + float(getattr(sets, "live_result_wait_seconds", 90.0))
        step = max(0.01, min(0.1, pause / 3))
        while time.monotonic() < end:
            s = self.session
            if self._closing or s.ending or s.leaving:
                return False
            if self._quiet(pause):
                return True
            await asyncio.sleep(step)
        return False

    async def _announce(self, job: _Job) -> None:
        """A background job finished: at the next pause tell the Live model to say the result (the
        full text is already in the chat). No pause within `live_result_wait_seconds`: chat only."""
        en, res = self._en(), job.result or {}
        q = " ".join(job.question.split())[:160]
        if res.get("answer"):
            cue = (RESULT_CUE_EN if en else RESULT_CUE).format(
                job=job.id, question=q, answer=str(res["answer"])[:ANSWER_MAX_CHARS]
            )
        else:
            say = str(res.get("say") or catalog("voice_tool_failed_say", self._filler_lang()))
            cue = (FAILED_CUE_EN if en else FAILED_CUE).format(job=job.id, question=q, answer=say)
        await self.announce_cue(cue, f"job {job.id}")

    async def announce_cue(self, cue: str, tag: str = "cue") -> None:
        """Tell the Live model, at the next pause, to say a prepared cue (a job result, a vote
        result). The room asked for it, so the mouth gate lets that audio by. No pause within
        `live_result_wait_seconds`: nothing is spoken (the chat has it)."""
        if not await self._wait_for_pause():
            logger.info("live: %s not spoken: no pause in time (chat has it)", tag)
            return
        live = self._live
        if live is None or self._closing or self.session.ending:
            return
        now = time.monotonic()
        self._announce_until = now + ANNOUNCE_AUTH_S  # the room asked for this: the gate lets it by
        self._last_engaged_at = now  # like a name hit: follow-ups work for the grace window
        logger.info("live: announcing %s", tag)
        await live.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=cue)]), turn_complete=True
        )

    async def _guarded(self, name: str, fn, question: str) -> dict:
        """Run a brain tool with the time limit. A failure or timeout returns the honest line the
        model reads as it is (`say`) and is shown in the text chat; nothing is invented. An answer
        is always posted to the text chat, with the files its tools made (never sent to the model)."""
        limit = float(getattr(self.bot.s, "live_ask_timeout_seconds", ASK_TIMEOUT_S))
        try:
            res = await asyncio.wait_for(fn(question), limit)
        except TimeoutError:
            reason = "timeout"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("live: %s failed: %s", name, type(e).__name__)
            reason = str(e) or type(e).__name__
        else:
            if name == "ask_salid" and isinstance(res, dict):
                res = dict(res)
                files = list(res.pop("files", None) or [])
                if res.get("answer") or files:
                    self._echo_answer(str(res.get("answer") or ""), files)
                if files and res.get("answer"):
                    res["answer"] = (
                        f"{res['answer']} {catalog('voice_files_say', self._filler_lang())}"
                    )
            return res
        lang = self._filler_lang()
        self._echo_text("voice_tool_failed", tool=name, reason=reason[:80])
        return {"error": reason, "say": catalog("voice_tool_failed_say", lang)}

    def _echo_answer(self, text: str, files: list) -> None:
        """The answer (and the files its tools made) in the meeting's text chat."""
        lang = self._filler_lang()
        body = catalog("voice_answer_echo", lang, text=text) if text else ""
        kw: dict = {}
        if files:
            body = f"{body}\n{catalog('voice_files_echo', lang)}".strip()
            kw["files"] = [
                discord.File(io.BytesIO(data), filename=name) for name, data in files[:10]
            ]

        async def send() -> None:
            try:
                await self.echo(body, **kw)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("live: could not post to the text chat (%s)", type(e).__name__)

        with contextlib.suppress(RuntimeError):
            self._spawn(send())

    def _echo_text(self, key: str, **fmt: Any) -> None:
        """Post a fixed line to the meeting's text chat; never raises into the Live loop."""
        text = catalog(key, self._filler_lang(), **fmt)

        async def send() -> None:
            try:
                await self.echo(text)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("live: could not post to the text chat (%s)", type(e).__name__)

        with contextlib.suppress(RuntimeError):
            self._spawn(send())

    async def _ask_salid(self, question: str) -> dict:
        """The full text agent for this room: team memory limited to what every human present may
        read, the ledger only when all of them are executives, web when enabled, and the last
        minutes of this meeting as history. The answer is short enough to be spoken."""
        from .qa import ask

        bot, s = self.bot, self.session
        if not question.strip():
            return {"error": "empty question"}
        if not bot.team.enabled(s.guild_id, "qa"):
            return {"error": QA_OFF_EN if self._en() else QA_OFF}
        humans = [m for m in s.voice_channel.members if not getattr(m, "bot", False)]
        if not humans:
            return {"error": NO_ROOM_EN if self._en() else NO_ROOM}
        guild = getattr(humans[0], "guild", None)
        everyone_exec = all(bot.is_exec_member(m) for m in humans)
        if everyone_exec or len(humans) == 1:  # one identified person: their own receipts only
            user_id = humans[0].id
        else:  # unknown speaker: no account data at all
            user_id, question = 0, f"{question}\n{LEDGER_NOTE_EN if self._en() else LEDGER_NOTE}"
        # Same agent as in chat (bot._agent_kwargs). With several people in the room the WHOLE agent
        # is built for the lowest-ranked one, so every read (tasks, approvals, meetings, memory,
        # ledger) is safe to say out loud. Only the outward ACTIONS are re-bound to the person who
        # asked (the speaker heard just before this call; several overlapping: the lowest-ranked of
        # them) with their own identity and rights.
        asker = humans[0]
        ranks: dict = {}
        if len(humans) > 1 and guild is not None:
            ranks = await asyncio.to_thread(bot._ranks, guild)
            asker = min(humans, key=lambda h: ranks.get(h.id, -1))
        # The answer may be typed into the meeting chat or the fallback channel (text echo), so
        # memory is also limited to what everyone reading those places can see, for this asker.
        fallback = s.fallback_channel
        also = (bot._answering_id(fallback),) if fallback is not None else ()
        memory = (
            await bot._room_memory_tools(guild, s.voice_channel, humans, *also, asker=asker)
            if guild is not None
            else None
        )
        recent = set(self.recent_speakers())
        speakers = [h for h in humans if h.id in recent]
        speaker = min(speakers, key=lambda h: ranks.get(h.id, -1)) if speakers else asker
        known_asker = len(humans) == 1 or len(speakers) == 1
        if len(humans) > 1:
            note = ROOM_NOTE_EN if self._en() else ROOM_NOTE
            question = f"{question}\n{note.format(asker=speaker.display_name)}"
        kw: dict = {"language": self._qa_setting(), "memory": memory, **bot._web_kw(s.guild_id)}
        if guild is not None:
            here = guild.get_channel(s.minutes_channel_id or 0) or s.text_channel

            def agent_for(member, is_exec: bool, profile) -> dict:
                return bot._agent_kwargs(
                    bot.room_message(guild, member, here, room={h.id for h in humans}),
                    guild,
                    member,
                    is_exec,
                    setting=self._qa_setting(),
                    lang=self._qa_lang(question),
                    question=question,
                    memory=memory,
                    profile=profile,
                    people=tag_candidates(here),
                    files=[],  # no attachments by voice; a remembered promo kit can be revised
                )

            kw = agent_for(
                asker,
                everyone_exec,
                bot._profile_of(s.guild_id, speaker.id) if known_asker else None,
            )
            if speaker.id != asker.id:
                kw = rebind_actions(
                    kw,
                    agent_for(speaker, bot.is_exec_member(speaker), None),
                )
        job = _JOB.get()
        loop = asyncio.get_running_loop()

        def on_tool(name: str) -> None:  # from the agent's thread: a known-long tool = background
            if job is not None and name in BACKGROUND_TOOLS:
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(job.background.set)

        ans = await asyncio.to_thread(
            ask,
            bot.s,
            bot.store,
            question,
            guild_id=s.guild_id,
            **kw,
            is_exec=everyone_exec,
            user_id=user_id,
            history=s.transcript.history(bot.s.now(), window_s=180, max_chars=3000),
            spoken=True,
            deck=bot.team.enabled(s.guild_id, "decks"),  # as in chat
            on_tool_call=on_tool,
        )
        # counts only: the question and answer may hold personal data
        logger.info(
            "live: ask_salid answered in %.1fs (%d rounds, %d chars)",
            ans.seconds,
            ans.rounds,
            len(ans.text or ""),
        )
        out: dict = {"answer": (ans.text or "")[:ANSWER_MAX_CHARS]}
        files = list(getattr(ans, "attachments", ()) or ())
        if files:  # posted to the text chat by `_guarded`, never sent to the Live model
            out["files"] = files
        return out

    async def _answer_privately(self, question: str) -> dict:
        """A full `ask` as the speaker with their own rights (as if they had DMed Salid), sent to
        them by DM. Only a status goes back to the voice model: the content never enters it."""
        from .dm import send_dm
        from .qa import ask

        bot, s = self.bot, self.session
        if not question.strip():
            return {"error": "empty question"}
        if not bot.team.enabled(s.guild_id, "qa"):
            return {"error": QA_OFF_EN if self._en() else QA_OFF}
        humans = [m for m in s.voice_channel.members if not getattr(m, "bot", False)]
        recent = set(self.recent_speakers())
        speakers = [h for h in humans if h.id in recent]
        who = humans[0] if len(humans) == 1 else (speakers[0] if len(speakers) == 1 else None)
        guild = getattr(who, "guild", None)
        if who is None or guild is None:
            return {"error": "do not know who is asking: ask who wants the private answer"}
        here = guild.get_channel(s.minutes_channel_id or 0) or s.text_channel
        is_exec = bot.is_exec_member(who)
        # as in a DM: the asker's own readable channels, not the room's
        memory = await bot._memory_tools(SimpleNamespace(guild=None, channel=here), guild, who)
        kw = bot._agent_kwargs(
            bot.room_message(guild, who, here),
            guild,
            who,
            is_exec,
            setting=self._qa_setting(),
            lang=self._qa_lang(question),
            question=question,
            memory=memory,
            profile=bot._profile_of(s.guild_id, who.id),
            people=tag_candidates(here),
            files=[],
        )
        ans = await asyncio.to_thread(
            ask,
            bot.s,
            bot.store,
            question,
            guild_id=s.guild_id,
            **kw,
            is_exec=is_exec,
            user_id=who.id,
            history=s.transcript.history(bot.s.now(), window_s=180, max_chars=3000),
            deck=bot.team.enabled(s.guild_id, "decks"),  # as in chat
        )
        res = await send_dm(who, ans.text or "")
        logger.info(
            "live: answer_privately %s in %.1fs", "sent" if res.ok else "failed", ans.seconds
        )
        return {"status": "sent"} if res.ok else {"status": "failed", "reason": res.reason}

    async def _on_tool_call(self, live: Any, tool_call: Any) -> None:
        """The tools run in the background: the receiver keeps handling audio, transcripts and
        interruptions while the brain works (a re-asked question can then join / replace it)."""
        engaged = self._authorize_tools([fc.name for fc in tool_call.function_calls or []])
        self._answer_auth = engaged  # a spoken answer to a call made while engaged may be heard
        if any(
            self._refusal(fc.name, dict(fc.args or {}), engaged) is None
            for fc in tool_call.function_calls or []
        ):
            # the turn_complete that follows is not the end of the answer (a call refused below
            # produces no answer to wait for)
            self._awaiting_answer = True
            self._awaiting_since = time.monotonic()
        task = asyncio.create_task(self._run_tool_call(live, tool_call, engaged))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    def _gate_open(self) -> bool:
        """Someone called Salid (or the mouth gate is off) or she was engaged within
        `live_reengage_grace_seconds` (a lively room drops the state while a question is open)."""
        if not getattr(self.bot.s, "live_mouth_gate", True) or self.session.engaged:
            return True
        grace = float(getattr(self.bot.s, "live_reengage_grace_seconds", 60.0))
        return time.monotonic() - self._last_engaged_at <= grace

    def _authorize_tools(self, names: list[str]) -> bool:
        """The gate for a tool call batch. Passing only thanks to the grace re-engages in code (the
        state line in chat is right). `ask_salid` / `answer_privately` while not engaged is
        evidence: if the last 3 s of user audio is real speech, someone did ask, so engage."""
        if self._gate_open():
            if (
                not self.session.engaged
                and getattr(self.bot.s, "live_mouth_gate", True)
                and any(n != "post_note" for n in names)  # a note alone is not a conversation
            ):
                self.bot.meetings._set_engaged(self.session, True)
            return True
        if any(n in ("ask_salid", "answer_privately") for n in names):
            stats = self._speech_stats(time.monotonic(), ASK_EVIDENCE_S)
            ok, reason = self._ask_evidence(stats)
            if ok:
                logger.info(
                    "live: ask_salid accepted: speech heard, engaging %s", stats_line(stats)
                )
                self.bot.meetings._set_engaged(self.session, True)
                return True
            logger.info("live: ask_salid refused: %s %s", reason, stats_line(stats))
        return False

    def _ask_evidence(self, stats: SpeechStats) -> tuple[bool, str]:
        """Is there speech behind an ask_salid made while listening? The Live model already judged
        that someone addressed Salid, so this only excludes noise: some voiced audio, speech-like
        variation, loud enough (no voiced-fraction rule, unlike name hits)."""
        sets = self.bot.s
        if stats.voiced_s < ASK_MIN_VOICED_S:
            return False, "too little voiced audio"
        if stats.cv < float(getattr(sets, "name_min_cv", 0.25)):
            return False, "steady noise"
        if stats.rms < float(getattr(sets, "name_min_rms", 250.0)):
            return False, "too quiet"
        return True, ""

    @staticmethod
    def _refusal(name: str, args: dict, engaged: bool) -> dict | None:
        """Mouth gate for tools: while nobody has called Salid the model may only take notes. A
        refused call has no side effect (and no filler); `ask_consent` stays free as text."""
        if engaged or name == "post_note":
            return None
        if name == "ask_consent":
            return None if str(args.get("medium", "text")) == "text" else NOBODY_ASKED
        return NOBODY_ASKED if name in GATED_TOOLS else None

    async def _run_tool_call(self, live: Any, tool_call: Any, engaged: bool = True) -> None:
        try:
            responses = []
            for fc in tool_call.function_calls or []:
                logger.info("live: tool call %s", fc.name)  # name only: arguments may hold PII
                args = dict(fc.args or {})
                result = self._refusal(fc.name, args, engaged)
                if result is not None:
                    logger.info("live: tool call %s refused (listening)", fc.name)
                    responses.append(
                        types.FunctionResponse(id=fc.id, name=fc.name, response=result)
                    )
                    continue
                filler = self._start_filler(fc.name, args)
                try:
                    result = await self._run_tool(fc.name, args)
                finally:
                    if filler is not None:
                        filler.cancel()
                responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=result))
            if responses and not self._closing:
                await live.send_tool_response(function_responses=responses)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # never into the Live loop
            logger.warning("live: tool call failed: %s", type(e).__name__)
            self._awaiting_answer = False

    # ---- audio out
    # Mouth gate. The model is told to stay silent while listening, but sometimes speaks anyway
    # (an unmuted mic or a noise starts a turn). So the rule is in code: model audio is played only
    # if the session is engaged (the name was heard by either detector), or if it answers a tool
    # call made while engaged. A turn that is neither is held for `live_call_grace_seconds`
    # (the name can reach the transcript a moment after the model starts to answer); the name in
    # that window releases it (`notify_state(True)`), otherwise it is dropped for the whole turn.
    def _gate_audio(self, data: bytes) -> None:
        if not getattr(self.bot.s, "live_mouth_gate", True):
            self._on_audio(data)
            return
        if (
            self._turn_ok
            or self._gate_open()
            or (self._awaiting_answer and self._answer_auth)
            or time.monotonic() < self._announce_until  # a job result the room asked for
        ):
            self._turn_ok = True
            self._release_held()
            self._on_audio(data)
            return
        now = time.monotonic()
        if not self._held:
            self._held_since = now
        grace = float(getattr(self.bot.s, "live_call_grace_seconds", 2.5))
        if now - self._held_since <= grace:
            self._held.append(data)
            self._gated_turn = True
            return
        self._drop_unsolicited()  # no name in time: this turn stays muted to its end

    def _release_held(self) -> None:
        """The name arrived: play what was held back, if it was held for no longer than the grace."""
        held, self._held = self._held, []
        if not held:
            return
        grace = float(getattr(self.bot.s, "live_call_grace_seconds", 2.5))
        if time.monotonic() - self._held_since > grace:  # too late: the rest of the turn too
            self._gated_turn = True
            self._muted_turn = self._turn_open
            self._log_muted()
            return
        self._turn_ok = True
        self._gated_turn = False
        for chunk in held:
            self._on_audio(chunk)

    def _drop_unsolicited(self) -> None:
        self._held.clear()
        self._gated_turn = True
        self._muted_turn = self._turn_open  # the rest of this turn is dropped like a "stop"
        self._log_muted()

    def _log_muted(self) -> None:
        if not self._muted_logged:
            self._muted_logged = True
            logger.info("live: unsolicited audio muted (listening)")

    def _on_audio(self, data: bytes) -> None:
        s = self.session
        if s.ending or s.leaving or not self.bot.s.voice_reply or not _connected(s.vc):
            return
        if not self._turn_audio:
            self._turn_audio = True
            # the people this answer is for: whoever spoke just before it (and overlapped them)
            self._partners = set(self.recent_speakers())
            self._talkover.clear()
            self._heard_while_speaking = ""
            if self._latency_pending:
                logger.info(
                    "live: user audio ended -> first audio %.1fs",
                    time.monotonic() - self._last_voice,
                )
                self._latency_pending = False
        self._awaiting_answer = False
        s.introduced = True  # Salid has spoken: the room knows it is there
        self._last_audio_at = self._real_audio_at = time.monotonic()
        src = self._source
        if src is not None and self._filler_until > self._real_audio_at:
            src.clear()  # the real answer is here: cut the rest of the "one moment" clip
        self._filler_until = 0.0
        if src is None or not src.feed(data):  # no source yet, or the last one is spent
            src = self._source = LiveAudioSource()
            src.feed(data)
        self._ensure_playing(src)

    # ---- filler ("one moment") while a slow tool runs
    def _en(self) -> bool:
        """The session speaks English: guild/channel language en (auto falls back to the guild's)."""
        return getattr(self.session, "lang", "th") == "en"

    def _qa_setting(self) -> str:
        s = self.session
        return "en" if self._en() else s.language

    def _qa_lang(self, question: str) -> str:
        return "en" if self._en() else resolve_lang(self.session.language, question)

    def _filler_lang(self) -> str:
        return "en" if self._en() else "th"

    def _particle(self) -> str:
        return str(getattr(self.bot.s, "voice_particle", "") or "")

    def _pool(self, lang: str) -> list[str]:
        """The fallback lines of a pool as they are spoken (Thai with the persona particle)."""
        return [with_particle(x, self._particle()) for x in FILLERS[lang]]

    async def _prerender_fillers(self) -> None:
        """Synthesize a few fallback clips once (Cloud TTS, the classic voice) so one is ready when
        a generated line is not. Bounded: not the whole pool, to save TTS calls."""
        lang = self._filler_lang()
        picks = random.sample(self._pool(lang), PRERENDER_FIRST)
        for text in picks:
            try:
                await self._filler_clip(text, lang)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("live: filler clip failed", exc_info=True)

    async def _render(self, text: str, lang: str, cache: bool = False) -> bytes:
        """Cloud TTS + pitch. `cache`: a reusable line, its clip is loaded from / saved to disk."""
        line = with_particle(text, self._particle())
        clips = ClipCache.for_settings(self.bot.s) if cache else None
        if clips is not None:
            pcm = await asyncio.to_thread(clips.get, lang, line)
            if pcm:
                return pcm
        pcm = await asyncio.to_thread(say, self.bot.s, self.bot.llm, line, lang)
        pcm = await asyncio.to_thread(pitch_pcm, pcm, getattr(self.bot.s, "tts_pitch", 1.0))
        if clips is not None:
            await asyncio.to_thread(clips.put, lang, line, pcm)
        return pcm

    async def _filler_clip(self, text: str, lang: str) -> bytes:
        """A cached clip of a fallback line."""
        key = f"{lang}:{text}"
        pcm = self._filler_pcm.get(key)
        if pcm is None:
            pcm = self._filler_pcm[key] = await self._render(text, lang, cache=True)
        return pcm

    def _pick_fallback(self, lang: str) -> str:
        """A fallback line: not one of the last 3 said, preferring the ones whose clip is already
        rendered."""
        pool = self._pool(lang)
        avoid = set(self._filler_used[-FILLER_MEMORY:])
        fresh = [x for x in pool if x not in avoid] or pool
        ready = [x for x in fresh if f"{lang}:{x}" in self._filler_pcm]
        text = random.choice(ready or fresh)
        self._remember(text)
        return text

    def _remember(self, text: str) -> None:
        self._filler_used = [*self._filler_used, text][-FILLER_MEMORY:]

    def _start_filler(self, tool: str, args: dict | None = None) -> asyncio.Task | None:
        """A slow tool starts and Salid has not spoken in this turn: say "one moment" at once.
        `args` carry the question the line is written for."""
        s = self.session
        if (
            tool not in FILLER_TOOLS
            or not getattr(self.bot.s, "live_filler", True)
            or self._turn_audio  # the model is already talking: no second voice on top
            or s.ending
            or s.leaving
        ):
            return None
        question = str((args or {}).get("question") or "")
        running = self._filler_task is not None and not self._filler_task.done()
        if running and same_question(self._filler_question, question):
            return None  # the question was re-asked: the "one moment" already running covers it
        if running:
            self._filler_task.cancel()  # a different question replaces the old one
        self._filler_question = question
        task = self._filler_task = asyncio.create_task(self._filler_run(question))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)
        return task

    def _writer(self) -> FillerWriter:
        return getattr(self.bot, "fillers", None) or FillerWriter(
            getattr(self.bot, "llm", None), self.bot.s
        )

    async def _generated_clip(self, question: str, lang: str) -> tuple[str, bytes] | None:
        """A line written for this question and its clip (None: nothing usable)."""
        got = await write_line(
            self._writer(),
            kind="voice",
            question=question,
            lang=lang,
            particle=self._particle() or "ค่ะ",
            stage="first",
            avoid=self._filler_used,
        )
        if got is None or got.text in self._filler_used:
            return None
        return got.text, await self._render(got.text, lang, cache=got.generic)

    async def _filler_run(self, question: str = "") -> None:
        """The spoken "one moment" at once; if the tool is still running after
        `live_filler_again_seconds`, ONE typed note in the meeting chat (never a second clip)."""
        lang = self._filler_lang()
        writer = self._writer()
        first: asyncio.Task | None = None
        try:
            started = time.monotonic()
            timeout = float(getattr(self.bot.s, "voice_ack_timeout_s", 3.0))
            if writer.enabled():
                first = asyncio.create_task(self._generated_clip(question, lang))
            made = None
            if first is not None:
                done, _ = await asyncio.wait({first}, timeout=timeout)
                if first in done and not first.cancelled() and first.exception() is None:
                    made = first.result()
                if made is None:
                    first.cancel()  # a late line is dropped: one filler only
            if made is not None:
                text, pcm = made
                self._remember(text)
            else:
                text = self._pick_fallback(lang)
                pcm = await self._filler_clip(text, lang)
            await self._play_filler(pcm)  # False: the model is already talking; the note still goes
            wait = float(getattr(self.bot.s, "live_filler_again_seconds", 8.0))
            await asyncio.sleep(max(0.0, wait - (time.monotonic() - started)))
            s = self.session
            if not (self._closing or s.ending or s.leaving):
                self._echo_text("voice_working_text", question=status_question(question))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("live: filler failed", exc_info=True)
        finally:
            if first is not None:
                if not first.done():
                    first.cancel()
                else:
                    with contextlib.suppress(BaseException):
                        first.exception()  # a failed line is already logged / ignored

    async def _play_filler(self, pcm: bytes) -> bool:
        """Queue a filler clip on the Live playback path. False when the model started to speak."""
        s = self.session
        if self._turn_audio or s.ending or s.leaving or not _connected(s.vc):
            return False
        src = self._source
        if src is None or not src.feed(pcm):
            src = self._source = LiveAudioSource()
            src.feed(pcm)
        self._filler_until = time.monotonic() + pcm_duration_s(pcm)
        self._ensure_playing(src)
        return True

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
        self._turn_open = False
        if self._gated_turn and not self._turn_ok:
            # unsolicited turn: its held audio is discarded and its words are never posted
            self._held.clear()
            self._log_muted()
            self._muted_turn = True
            self._awaiting_answer = False  # nothing will be answered out loud
        self._turn_ok = self._gated_turn = self._muted_logged = False
        self._announce_until = 0.0  # the announcement turn is over
        if self._muted_turn:  # the rest of a turn somebody stopped: never posted or played
            self._muted_turn = False
            self._said.clear()
            return
        text = "".join(self._said).strip()
        self._said.clear()
        if text:
            self._spawn(self.post(f"🔊 {text}"))
        if self._awaiting_answer:
            return  # right after a tool call: the spoken answer is the next turn
        self._turn_audio = False
        if self._source is not None:
            self._source.finish()

    def _speaking(self) -> bool:
        """Salid's own voice is playing (or queued) right now."""
        return self._playing_src is not None or (self._source is not None and self._turn_audio)

    def _stop_requested(self) -> None:
        """Someone deliberately talked over Salid: stop now and drop the rest of this answer (the
        server keeps generating it because it never interrupts with NO_INTERRUPTION)."""
        logger.info("live: stopped on request")
        self._interrupt()
        # only the turn still being generated is dropped; if it already finished (its audio was
        # just playing out of the buffer) the next turn is the answer to whoever stopped her
        self._muted_turn = self._turn_open

    def _interrupt(self) -> None:
        """Barge-in: somebody talked over Salid. Drop queued audio and stop playing now."""
        logger.info("live: interrupted")
        if self._filler_task is not None:
            self._filler_task.cancel()
        self.stop_playback()
        self._said.clear()
        self._held.clear()
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
        self._held.clear()
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
