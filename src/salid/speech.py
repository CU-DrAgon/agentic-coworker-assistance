"""Spoken replies for voice meetings: shorten an answer for the ear, synthesize it (Cloud
Text-to-Speech Chirp 3 HD by default, ~1.5 s; Gemini TTS as fallback, ~6 s) and turn the PCM into
something discord.py can play. Nothing here touches the disk.

Voice persona: a cute, playful cat ("แมวฉอเลาะ"). The spoken rewrite adds the cat tone; the voice is
Chirp 3 HD Leda (the brightest Thai voice) nudged up in pitch by ffmpeg."""

from __future__ import annotations

import base64
import io
import logging
import re
import unicodedata
import wave
from typing import Any

import discord
from google.genai import types

from .llm import generate

logger = logging.getLogger(__name__)

SPOKEN_MAX_CHARS = 220
TTS_RATE = 24000  # both backends return raw 16-bit mono PCM at 24 kHz
TTS_BYTES_PER_SECOND = TTS_RATE * 2

SPOKEN_PROMPT = """คุณคือ “สลิด” ผู้ช่วยในห้องประชุมเสียง เรียบเรียงคำตอบด้านล่างให้เป็นประโยคพูดสั้น ๆ 1-2 ประโยค \
เพื่อให้ผู้ร่วมประชุมฟังรู้เรื่องในครั้งเดียว

กติกา:
- ใช้ภาษาเดียวกับคำตอบ (คำตอบเป็นภาษาไทยก็ตอบภาษาไทย)
- ใช้เฉพาะข้อมูลที่อยู่ในคำตอบ ห้ามเพิ่มข้อเท็จจริงใหม่
- ไม่ใช้ markdown ไม่ใช้อีโมจิ ไม่ใส่ลิงก์หรือ URL ไม่ใช้ตาราง ไม่อ่านรายการยาว ๆ สรุปเฉพาะประเด็นหลัก
- เขียนตัวเลขให้อ่านออกเสียงได้ลื่น เช่น “หนึ่งหมื่นสองพัน” หรือ “12,450 บาท” ตามที่ฟังเป็นธรรมชาติ
- บุคลิก: “แมวฉอเลาะ” น่ารัก ขี้อ้อน ร่าเริง พูดเป็นกันเอง ใช้คำลงท้ายอย่าง “น้า” “นะเมี้ยว” “เมี้ยว~” แทน ครับ/ค่ะ \
(ใส่ “เมี้ยว” ได้ไม่เกินหนึ่งครั้งต่อคำตอบ) ถ้าเป็นภาษาอังกฤษ ใช้น้ำเสียงน่ารักขี้เล่นแบบเดียวกัน เช่น “meow~”
- ความน่ารักต้องไม่ทำให้ข้อมูลผิดหรือหายไป ตัวเลข ชื่อ และวันที่ต้องครบตามคำตอบ
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


CLOUD_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
_LANG_CODES = {"th": "th-TH", "en": "en-US"}


def _cloud_voice(voice: str, lang: str) -> tuple[str, str]:
    """('th-TH', 'th-TH-Chirp3-HD-Leda') from a short voice name ('Leda') and a language."""
    code = _LANG_CODES.get(lang, "th-TH")
    return code, voice if "-" in voice else f"{code}-Chirp3-HD-{voice}"


def synthesize_cloud(
    project: str, voice: str, text: str, lang: str = "th", rate: float = 1.0
) -> bytes:
    """Blocking. Cloud Text-to-Speech (Chirp 3 HD) -> raw 16-bit mono PCM at 24 kHz, ~1.5 s."""
    import google.auth
    import google.auth.transport.requests
    import httpx

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    code, name = _cloud_voice(voice, lang)
    body = {
        "input": {"text": text},
        "voice": {"languageCode": code, "name": name},
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": TTS_RATE,
            "speakingRate": rate,
        },
    }
    headers = {"Authorization": f"Bearer {creds.token}", "x-goog-user-project": project}
    r = httpx.post(CLOUD_TTS_URL, json=body, headers=headers, timeout=20.0)
    if r.status_code != 200:
        raise RuntimeError(f"Cloud TTS HTTP {r.status_code}")
    audio = base64.b64decode(r.json().get("audioContent") or b"")
    if not audio:
        raise RuntimeError("Cloud TTS returned no audio")
    if audio[:4] == b"RIFF":  # LINEAR16 comes wrapped in a WAV header
        with wave.open(io.BytesIO(audio)) as w:
            if w.getframerate() != TTS_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise RuntimeError("unexpected Cloud TTS audio format")
            return w.readframes(w.getnframes())
    return audio


def say(settings: Any, client: Any, text: str, lang: str = "th") -> bytes:
    """Blocking. PCM for `text` with the configured backend; Cloud TTS falls back to Gemini TTS."""
    if getattr(settings, "tts_backend", "cloud") == "cloud":
        try:
            return synthesize_cloud(
                settings.gcp_project,
                settings.cloud_tts_voice,
                text,
                lang,
                settings.tts_speaking_rate,
            )
        except Exception:
            logger.warning("Cloud TTS failed, falling back to Gemini TTS", exc_info=True)
    return synthesize(client, settings.tts_model, settings.tts_voice, text)


def pcm_duration_s(pcm: bytes) -> float:
    return len(pcm) / TTS_BYTES_PER_SECOND


def pcm_source(pcm24k_mono: bytes, pitch: float = 1.0) -> discord.AudioSource:
    """Playable source for `VoiceClient.play`. ffmpeg resamples to Discord's 48 kHz stereo and,
    with pitch > 1, raises the voice without changing its speed (the cute-cat voice)."""
    options = None
    if pitch and abs(pitch - 1.0) > 0.01:
        options = (
            f"-af asetrate={int(TTS_RATE * pitch)},aresample={TTS_RATE},atempo={1 / pitch:.4f}"
        )
    return discord.FFmpegPCMAudio(
        io.BytesIO(pcm24k_mono),
        pipe=True,
        before_options=f"-f s16le -ar {TTS_RATE} -ac 1",
        options=options,
    )
