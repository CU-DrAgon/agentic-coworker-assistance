"""Audio and video files people send: extract the audio track, transcribe the speech with Gemini.

Pipeline (blocking, run with asyncio.to_thread): bytes -> temp file -> ffmpeg (no video, mono,
16 kHz FLAC, at most MAX_DURATION_S) -> ~10 minute chunks with a 2 s overlap -> one Gemini call per
chunk (segments with speaker labels and times relative to the chunk) -> chunk offsets added in code,
overlap duplicates dropped. Whole videos are never sent to the model, only the audio. Temp files
live in a TemporaryDirectory that is always removed; nothing is stored.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google.genai import types
from pydantic import BaseModel, Field

from .llm import generate

logger = logging.getLogger(__name__)

MAX_BYTES = 100 * 1024 * 1024
MAX_DURATION_S = 90 * 60
CHUNK_S = 10 * 60
OVERLAP_S = 2.0
CHUNK_WORKERS = 3
FFMPEG_TIMEOUT_S = 600
PROMPT_CHARS = 30_000  # transcript text handed to the agent
NOTE_CHARS = 1800  # transcript text kept in the memory log
VOICE_QUESTION_MAX_S = 120  # a voice message this short with no text is the question itself
AUDIO_EXTS = {"mp3", "m4a", "wav", "ogg", "opus", "aac", "flac"}
VIDEO_EXTS = {"webm", "mp4", "mov", "mkv"}

TRANSCRIBE_PROMPT = (
    "Transcribe the speech in this audio verbatim, in the language actually spoken (Thai, English "
    "or a mix; keep English words in English, do not translate). Split it into segments of one "
    "speaker turn (at most about 30 seconds each). For each segment give: start (time from the "
    "beginning of THIS audio as MM:SS), speaker (a label by voice: Speaker A, Speaker B, ...; use "
    "the person's name instead only when it is said aloud in the audio), text. Do not invent "
    "anything: mark unclear words as [inaudible], and return no segments when there is no speech "
    "(music, silence, noise). Also report language (the main spoken language, e.g. th, en)."
)
SUMMARY_PROMPT = (
    "Below is the transcript of a recording posted in a team chat ([MM:SS] speaker: text). Write "
    "a 1-2 line summary (at most 250 characters) of what it is and the key points, in the same "
    "language as the transcript. Use only what is in the transcript.\n\n{transcript}"
)


class MediaError(RuntimeError):
    """The media could not be read or transcribed."""


# ------------------------------------------------------------------ data
@dataclass
class Seg:
    start: float  # seconds from the start of the file
    speaker: str
    text: str


@dataclass
class Transcript:
    segments: list[Seg] = field(default_factory=list)
    duration: float = 0.0  # seconds of audio transcribed (capped at MAX_DURATION_S)
    language: str = ""
    truncated: bool = False  # the file was longer than MAX_DURATION_S

    @property
    def text(self) -> str:
        return format_lines(self.segments)


class RawSeg(BaseModel):
    start: str = "00:00"
    speaker: str = "Speaker A"
    text: str = ""


class RawTranscript(BaseModel):
    segments: list[RawSeg] = Field(default_factory=list)
    language: str = ""


# ------------------------------------------------------------------ detection
def _ext(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in (name or "") else ""


def media_kind(att) -> str | None:
    """ "voice" (Discord voice message), "audio", "video" or None for a Discord attachment."""
    if getattr(att, "waveform", None) or getattr(att, "duration", None):
        return "voice"
    name = str(getattr(att, "filename", "") or "")
    ct = str(getattr(att, "content_type", "") or "").split(";")[0].strip().lower()
    ext = _ext(name)
    if ct.startswith("audio/") or ext in AUDIO_EXTS:
        return "audio"
    if ct.startswith("video/") or ext in VIDEO_EXTS:
        return "video"
    return None


def is_voice_message(message, att=None) -> bool:
    """A Discord voice message (message flag, or the attachment's waveform / duration)."""
    flags = getattr(message, "flags", None)
    if flags is not None and getattr(flags, "voice", False) is True:
        return True
    return att is not None and media_kind(att) == "voice"


def media_candidates(message) -> list[tuple[int, Any]]:
    """(attachment index, attachment) for each audio/video attachment of a message."""
    found = []
    for i, a in enumerate(getattr(message, "attachments", ()) or ()):
        if media_kind(a) is not None or (is_voice_message(message) and i == 0):
            found.append((i, a))
    return found


_LINK = re.compile(r"channels/(?:\d+|@me)/(\d+)/(\d+)")


def parse_message_ref(ref: str) -> tuple[int | None, int] | None:
    """(channel id or None, message id) from a Discord message link or a bare message id."""
    ref = str(ref or "").strip()
    if m := _LINK.search(ref):
        return int(m.group(1)), int(m.group(2))
    if ref.isdigit():
        return None, int(ref)
    return None


# ------------------------------------------------------------------ time helpers
def fmt_time(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def parse_time(s: str) -> float:
    """ "MM:SS" / "H:MM:SS" / "SS" (also with a fraction) -> seconds; 0 when unparsable."""
    parts = re.findall(r"\d+(?:\.\d+)?", s or "")
    if not parts or len(parts) > 3:
        return 0.0
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return secs


# ------------------------------------------------------------------ chunking and merging
def plan_chunks(
    duration: float, chunk_s: float = CHUNK_S, overlap_s: float = OVERLAP_S
) -> list[tuple[float, float]]:
    """(start, length) windows covering `duration`: one window up to `chunk_s`, else windows of
    `chunk_s` + `overlap_s` every `chunk_s`; the last runs to the end. A trailing window that would
    only repeat the overlap is dropped."""
    if duration <= chunk_s:
        return [(0.0, float(duration))]
    out: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        if out and duration - start <= overlap_s:
            break
        length = min(chunk_s + overlap_s, duration - start)
        out.append((start, length))
        start += chunk_s
    return out


def merge_chunks(chunks: list[tuple[float, list[Seg]]], overlap_s: float = OVERLAP_S) -> list[Seg]:
    """Join per-chunk segments (times already absolute). From every chunk after the first, drop
    segments starting inside the overlap with the previous chunk (the previous one has them) and
    exact repeats of the last kept line."""
    out: list[Seg] = []
    for i, (offset, segs) in enumerate(chunks):
        for seg in sorted(segs, key=lambda x: x.start):
            if i and seg.start < offset + overlap_s:
                continue
            if out and out[-1].text == seg.text and abs(out[-1].start - seg.start) < 3:
                continue
            out.append(seg)
    return out


def segments_from_raw(raw: RawTranscript, offset: float, length: float) -> list[Seg]:
    """Model segments (chunk-relative "MM:SS") -> absolute `Seg`s. Empty / inaudible-only lines
    are dropped; times are clamped into the chunk."""
    out = []
    for r in raw.segments:
        text = " ".join((r.text or "").split())
        if not text or re.fullmatch(r"[\s\W_]*|\[inaudible\]", text, re.I):
            continue
        rel = min(max(parse_time(r.start), 0.0), max(length, 0.0))
        out.append(Seg(offset + rel, (r.speaker or "").strip() or "Speaker A", text))
    return out


# ------------------------------------------------------------------ formatting
def format_lines(segments: list[Seg]) -> str:
    return "\n".join(f"[{fmt_time(s.start)}] {s.speaker}: {s.text}" for s in segments)


def prompt_text(tr: Transcript, max_chars: int = PROMPT_CHARS) -> str:
    """Timestamped lines for the agent; over `max_chars` the first part is kept and the cut is
    said out loud. A recording cut at MAX_DURATION_S says so too."""
    lines = [f"[{fmt_time(s.start)}] {s.speaker}: {s.text}" for s in tr.segments]
    out, size, cut = [], 0, False
    for ln in lines:
        if size + len(ln) + 1 > max_chars:
            cut = True
            break
        out.append(ln)
        size += len(ln) + 1
    body = "\n".join(out) if out else "(ไม่พบเสียงพูด / no speech detected)"
    if cut:
        last = fmt_time(tr.segments[len(out) - 1].start) if out else "0:00"
        body += f"\n[ตัดทอน: แสดงถึง {last} จากทั้งหมด {fmt_time(tr.duration)} / truncated]"
    if tr.truncated:
        body += (
            f"\n[ไฟล์ยาวเกิน {MAX_DURATION_S // 60} นาที ถอดเสียงเฉพาะ {MAX_DURATION_S // 60} "
            "นาทีแรก / only the first part was transcribed]"
        )
    return body


def label_for(index: int, name: str, tr: Transcript) -> str:
    return f"[ถอดเสียงจากไฟล์ #{index}: {name}, ยาว {fmt_time(tr.duration)}]"


def memory_note(kind: str, tr: Transcript, summary: str = "", max_chars: int = NOTE_CHARS) -> str:
    """ "[เสียง: 3:42] summary | Speaker A: … | Speaker B: …" for the memory log: consecutive lines
    of one speaker are joined, the whole is capped at `max_chars`."""
    head = f"[{'วิดีโอ' if kind == 'video' else 'เสียง'}: {fmt_time(tr.duration)}]"
    turns: list[list[str]] = []
    for s in tr.segments:
        if turns and turns[-1][0] == s.speaker:
            turns[-1][1] += " " + s.text
        else:
            turns.append([s.speaker, s.text])
    body = " | ".join(f"{sp}: {tx}" for sp, tx in turns)
    summary = " ".join((summary or "").split())
    if summary:
        out = f"{head} {summary} | {body}" if body else f"{head} {summary}"
    else:
        out = f"{head} {body}" if body else f"{head} (ไม่พบเสียงพูด)"
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out


# ------------------------------------------------------------------ ffmpeg
def _run(args: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, capture_output=True, timeout=FFMPEG_TIMEOUT_S, check=False)
    except FileNotFoundError as e:
        raise MediaError("ffmpeg is not installed") from e
    except subprocess.TimeoutExpired as e:
        raise MediaError("ffmpeg timed out") from e


def probe_duration(path: Path) -> float:
    proc = _run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)]
    )
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (ValueError, KeyError, TypeError) as e:
        raise MediaError("could not read the duration") from e


def extract_audio(src: Path, dst: Path, max_s: float = MAX_DURATION_S) -> float:
    """Audio track of `src` -> mono 16 kHz FLAC `dst` (first `max_s` seconds). Returns the
    duration of `dst`. Raises MediaError when there is no decodable audio."""
    proc = _run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(src),
         "-vn", "-t", str(max_s), "-ac", "1", "-ar", "16000", "-c:a", "flac", str(dst)]
    )  # fmt: skip
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        err = proc.stderr.decode("utf-8", "replace").strip()[-300:]
        raise MediaError(f"no audio could be extracted: {err}")
    return probe_duration(dst)


def cut_chunk(src: Path, dst: Path, start: float, length: float) -> bytes:
    proc = _run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src), "-c:a", "flac", str(dst)]
    )  # fmt: skip
    if proc.returncode != 0 or not dst.exists():
        raise MediaError("could not cut an audio chunk")
    return dst.read_bytes()


# ------------------------------------------------------------------ model
def transcribe_chunk(client, model: str, flac: bytes) -> RawTranscript:
    resp = generate(
        client,
        model,
        [types.Part.from_bytes(data=flac, mime_type="audio/flac"), TRANSCRIBE_PROMPT],
        types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=RawTranscript,
        ),
    )
    return RawTranscript.model_validate(json.loads(resp.text))


def summarize_note(client, model: str, tr: Transcript) -> str:
    """One cheap call: 1-2 line summary of the transcript ('' when empty or on failure)."""
    if not tr.segments:
        return ""
    try:
        resp = generate(
            client,
            model,
            [SUMMARY_PROMPT.format(transcript=prompt_text(tr, 12_000))],
            types.GenerateContentConfig(temperature=0.2, max_output_tokens=200),
        )
        return " ".join((getattr(resp, "text", None) or "").split())[:300]
    except Exception as e:  # never log contents
        logger.warning("media summary failed (%s)", type(e).__name__)
        return ""


def transcribe_media(
    client, model: str, data: bytes, filename: str = "", *, max_s: float = MAX_DURATION_S
) -> Transcript:
    """Blocking: transcribe the speech in an audio/video file's bytes."""
    if not data:
        raise MediaError("empty file")
    if len(data) > MAX_BYTES:
        raise MediaError("file too large")
    with tempfile.TemporaryDirectory(prefix="salid-media-") as tmp:
        root = Path(tmp)
        src = root / ("in." + (_ext(filename) or "bin"))
        src.write_bytes(data)
        del data
        flac = root / "audio.flac"
        try:
            src_duration = probe_duration(src)
        except MediaError:
            src_duration = 0.0
        duration = extract_audio(src, flac, max_s)
        truncated = src_duration > max_s + 1
        plan = plan_chunks(duration)
        blobs = (
            [
                (start, length, cut_chunk(flac, root / f"c{i}.flac", start, length))
                for i, (start, length) in enumerate(plan)
            ]
            if len(plan) > 1
            else [(0.0, duration, flac.read_bytes())]
        )

        def one(item):
            start, length, blob = item
            try:
                raw = transcribe_chunk(client, model, blob)
                return start, segments_from_raw(raw, start, length), raw.language
            except Exception as e:
                logger.warning("media chunk transcription failed (%s)", type(e).__name__)
                return start, None, ""

        with ThreadPoolExecutor(max_workers=CHUNK_WORKERS) as pool:
            results = list(pool.map(one, blobs))
        del blobs
    if all(segs is None for _s, segs, _l in results):
        raise MediaError("transcription failed")
    parts: list[tuple[float, list[Seg]]] = []
    for start, segs, _l in results:
        if segs is None:
            segs = [Seg(start, "-", f"[ถอดเสียงช่วง {fmt_time(start)} ไม่สำเร็จ]")]
        parts.append((start, segs))
    lang = next((lg.strip() for _s, _g, lg in results if lg and lg.strip()), "")
    return Transcript(merge_chunks(parts), duration, lang, truncated)


# ------------------------------------------------------------------ daily budget
class DailyBudget:
    """Units (seconds of audio, pages of documents) used per guild per local day, in memory (a
    restart resets it). `cap` 0 = nothing allowed."""

    def __init__(self, cap: float):
        self.cap = max(0.0, float(cap))
        self._used: dict[int, tuple[str, float]] = {}

    def used(self, guild_id: int, day: str) -> float:
        d, s = self._used.get(guild_id, (day, 0.0))
        return s if d == day else 0.0

    def available(self, guild_id: int, day: str) -> bool:
        return self.used(guild_id, day) < self.cap

    def add(self, guild_id: int, day: str, amount: float) -> None:
        self._used[guild_id] = (day, self.used(guild_id, day) + max(0.0, amount))

    def take(self, guild_id: int, day: str, amount: float) -> bool:
        """Add `amount` unless that would pass `cap` (False, nothing added, then)."""
        if self.used(guild_id, day) + max(0.0, amount) > self.cap:
            return False
        self.add(guild_id, day, amount)
        return True

    def give_back(self, guild_id: int, day: str, amount: float) -> None:
        """Return `amount` units taken today (never below 0)."""
        self._used[guild_id] = (day, max(0.0, self.used(guild_id, day) - max(0.0, amount)))


def MediaBudget(minutes_per_day: int) -> DailyBudget:  # noqa: N802 - reads as a constructor
    return DailyBudget(max(0, minutes_per_day) * 60)
