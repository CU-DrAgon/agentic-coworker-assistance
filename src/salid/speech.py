"""Spoken replies for voice meetings: shorten an answer for the ear, synthesize it with Gemini TTS
on Vertex, and turn the PCM into something discord.py can play. Nothing here touches the disk."""

from __future__ import annotations

import io
import logging
import re
import unicodedata
from typing import Any

import discord
from google.genai import types

from .llm import generate

logger = logging.getLogger(__name__)

SPOKEN_MAX_CHARS = 220
TTS_RATE = 24000  # Gemini TTS returns raw 16-bit mono PCM at 24 kHz
TTS_BYTES_PER_SECOND = TTS_RATE * 2

SPOKEN_PROMPT = """คุณคือ “สลิด” ผู้ช่วยในห้องประชุมเสียง เรียบเรียงคำตอบด้านล่างให้เป็นประโยคพูดสั้น ๆ 1-2 ประโยค \
เพื่อให้ผู้ร่วมประชุมฟังรู้เรื่องในครั้งเดียว

กติกา:
- ใช้ภาษาเดียวกับคำตอบ (คำตอบเป็นภาษาไทยก็ตอบภาษาไทย)
- ใช้เฉพาะข้อมูลที่อยู่ในคำตอบ ห้ามเพิ่มข้อเท็จจริงใหม่
- ไม่ใช้ markdown ไม่ใช้อีโมจิ ไม่ใส่ลิงก์หรือ URL ไม่ใช้ตาราง ไม่อ่านรายการยาว ๆ สรุปเฉพาะประเด็นหลัก
- เขียนตัวเลขให้อ่านออกเสียงได้ลื่น เช่น “หนึ่งหมื่นสองพัน” หรือ “12,450 บาท” ตามที่ฟังเป็นธรรมชาติ
- พูดแบบผู้ชายสุภาพ ลงท้ายด้วย “ครับ” เมื่อเหมาะสม (ไม่ใช้ ค่ะ/คะ)
- ความยาวไม่เกิน {max_chars} ตัวอักษร
- ตอบเฉพาะประโยคที่จะพูด ไม่ต้องมีคำนำ

คำถาม: {question}

คำตอบเต็ม:
{answer}"""

_URL = re.compile(r"https?://\S+|www\.\S+")
_LINK = re.compile(r"\[([^\]]*)\]\((?:[^)]*)\)")
_CODE_FENCE = re.compile(r"```.*?```", re.S)
_MD_MARKS = re.compile(r"[*_`~>#|]+")
_BULLET = re.compile(r"^\s*(?:[-+•]|\d+[.)])\s+", re.M)
_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\u2600-\u27bf\u2300-\u23ff\u2b00-\u2bff\ufe0f\u200d\u20e3]"
)
_SPACES = re.compile(r"\s+")


def strip_markdown(text: str) -> str:
    """Plain speakable text: no markdown, links, URLs, emoji; whitespace collapsed."""
    t = _CODE_FENCE.sub(" ", text or "")
    t = _LINK.sub(r"\1", t)
    t = _URL.sub(" ", t)
    t = _BULLET.sub("", t)
    t = _MD_MARKS.sub(" ", t)
    t = _EMOJI.sub("", t)
    return _SPACES.sub(" ", t).strip()


def clip(text: str, max_chars: int = SPOKEN_MAX_CHARS) -> str:
    """Cut to at most `max_chars` characters, preferably at whitespace. Works on str, and never
    leaves a Thai combining mark (vowel/tone mark) dangling at the start of the cut-off rest."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = max_chars
    space = text.rfind(" ", 0, max_chars + 1)
    if space >= max_chars // 2:
        cut = space
    else:
        # No usable whitespace (Thai has few): do not split a base character from its marks.
        while cut > 0 and unicodedata.category(text[cut]).startswith("M"):
            cut -= 1
    return text[:cut].rstrip()


def fallback_text(answer: str, max_chars: int = SPOKEN_MAX_CHARS) -> str:
    return clip(strip_markdown(answer), max_chars)


def spoken_text(client: Any, model: str, question: str, answer: str) -> str:
    """Blocking. One short Gemini call turning the written answer into 1-2 spoken sentences.
    Any failure (API error, empty reply) falls back to the markdown-stripped, clipped answer."""
    try:
        resp = generate(
            client,
            model,
            [
                SPOKEN_PROMPT.format(
                    max_chars=SPOKEN_MAX_CHARS, question=question.strip(), answer=answer.strip()
                )
            ],
            types.GenerateContentConfig(temperature=0.3),
        )
        text = fallback_text(getattr(resp, "text", None) or "")
        if text:
            return text
    except Exception:
        logger.warning("spoken_text failed, using the plain answer", exc_info=True)
    return fallback_text(answer)


def _rate_of(mime: str) -> int | None:
    m = re.search(r"rate=(\d+)", mime or "")
    return int(m.group(1)) if m else None


def synthesize(client: Any, model: str, voice: str, text: str) -> bytes:
    """Blocking. Gemini TTS -> raw 16-bit mono PCM at 24 kHz. Raises RuntimeError when the model
    returns no audio (or audio in a format we cannot play)."""
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
    )
    resp = generate(client, model, text, config)
    for cand in getattr(resp, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            blob = getattr(part, "inline_data", None)
            data = getattr(blob, "data", None)
            if not data:
                continue
            mime = getattr(blob, "mime_type", "") or ""
            if not mime.lower().startswith("audio/l16") or _rate_of(mime) not in (None, TTS_RATE):
                raise RuntimeError(f"unexpected TTS audio format: {mime!r}")
            return bytes(data)
    raise RuntimeError("TTS returned no audio")


def pcm_duration_s(pcm: bytes) -> float:
    return len(pcm) / TTS_BYTES_PER_SECOND


def pcm_source(pcm24k_mono: bytes) -> discord.AudioSource:
    """Playable source for `VoiceClient.play`. ffmpeg resamples to Discord's 48 kHz stereo."""
    return discord.FFmpegPCMAudio(
        io.BytesIO(pcm24k_mono), pipe=True, before_options=f"-f s16le -ar {TTS_RATE} -ac 1"
    )
