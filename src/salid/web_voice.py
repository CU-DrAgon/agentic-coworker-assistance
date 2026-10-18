"""Voice input for the command-center chat: any language in, the page's language out.

The browser records one utterance (MediaRecorder: webm/opus, ogg or mp4) and posts it. ffmpeg
turns it into 16 kHz mono FLAC, Gemini transcribes it verbatim in whatever language was spoken
and, when that is not the page's language, also translates it, so a Thai sentence spoken on the
English page lands in the chat in English (and the original is kept for the record).
"""

from __future__ import annotations

import json
import re
from typing import Any

from google.genai import types
from pydantic import BaseModel

from meeting_scribe.audio import to_flac

from .i18n import lang_name
from .llm import generate

MAX_AUDIO_BYTES = 8 * 1024 * 1024  # ~ a minute of opus


class Heard(BaseModel):
    text: str  # verbatim, in the language spoken
    language: str  # BCP-47 primary subtag, e.g. "th", "en", "ja"; "" when there was no speech
    translated: str = ""  # the same utterance in the target language ("" when not needed)


PROMPT = (
    "Transcribe this short recording verbatim in the language that was spoken (any language). "
    "Then, if that language is not {target} ({code}), translate it into {target} as "
    '`translated`; otherwise leave `translated` empty. If there is no speech, return text="" '
    'and language="". Return JSON only.'
)


def transcribe_utterance(
    client: Any, model: str, data: bytes, mime: str, target_lang: str = "en"
) -> Heard:
    """Blocking: browser audio -> FLAC -> Gemini. Raises on conversion or model failure."""
    if not data or len(data) > MAX_AUDIO_BYTES:
        raise ValueError("audio empty or too large")
    flac = to_flac(data)
    config = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=Heard, temperature=0.0
    )
    prompt = PROMPT.format(target=lang_name(target_lang), code=target_lang)
    resp = generate(
        client, model, [types.Part.from_bytes(data=flac, mime_type="audio/flac"), prompt], config
    )
    raw = (getattr(resp, "text", None) or "").strip()
    try:
        heard = Heard.model_validate(json.loads(raw))
    except (ValueError, AttributeError):
        heard = Heard(text=raw, language="")
    heard.text = heard.text.strip()
    if re.fullmatch(r"[\s\W_]*", heard.text):
        heard.text, heard.language, heard.translated = "", "", ""
    heard.language = heard.language.lower().split("-")[0][:8]
    if heard.language == target_lang:
        heard.translated = ""
    return heard
