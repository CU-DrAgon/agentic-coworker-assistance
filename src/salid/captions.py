"""Live captions for voice meetings: OFF unless someone asks ("เปิดซับ", "turn on captions in
English"). While on, what people say is batched and posted in the room's text chat as
"**Name:** text", optionally translated (th / en) with the cheap model.

This module is pure logic (no Discord): `CaptionState` buffers lines, `translate_lines` does one
LLM call per batch, `format_captions` builds the messages and `captions_tools` is the chat/voice
agent tool. The meeting manager (meeting.py) feeds it and posts what it returns. Only human
speech is ever fed in, never Salid's own notes."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from google.genai import types

from .i18n import LANGS, lang_name
from .llm import generate

logger = logging.getLogger(__name__)

LANGUAGES = ("original", *(x for x in LANGS if x != "auto"))
FLUSH_EVERY_S = 12.0
MAX_BATCH = 30  # lines per translation call
MAX_PENDING = 120  # older lines are dropped if posting cannot keep up
MAX_MESSAGE_CHARS = 1800
MAX_LINE_CHARS = 400

_ALIASES = {
    "original": "original", "orig": "original", "none": "original", "default": "original",
    "เดิม": "original", "ต้นฉบับ": "original", "": "original", "same": "original",
    "th": "th", "thai": "th", "ไทย": "th", "ภาษาไทย": "th",
    "en": "en", "english": "en", "eng": "en", "อังกฤษ": "en", "ภาษาอังกฤษ": "en",
}  # fmt: skip


def normalize_language(value: object) -> str | None:
    """'original' | 'th' | 'en', or None when the value is not a language we caption in."""
    return _ALIASES.get(str(value or "").strip().casefold())


@dataclass
class CaptionState:
    """Per-meeting caption switch and buffer. Off by default."""

    on: bool = False
    language: str = "original"
    pending: list[tuple[str, str]] = field(default_factory=list)  # (speaker, text)
    last_flush: float = 0.0

    def turn_on(self, language: str, now: float) -> None:
        if not self.on:
            self.pending.clear()
            self.last_flush = now
        self.on, self.language = True, language

    def turn_off(self) -> None:
        self.on = False
        self.pending.clear()

    def add(self, speaker: str, text: str) -> None:
        text = " ".join((text or "").split())
        if not self.on or not text:
            return
        self.pending.append((speaker, text[:MAX_LINE_CHARS]))
        del self.pending[:-MAX_PENDING]

    def due(self, now: float) -> bool:
        return self.on and bool(self.pending) and now - self.last_flush >= FLUSH_EVERY_S

    def take(self, now: float) -> list[tuple[str, str]]:
        batch, self.pending = self.pending[:MAX_BATCH], self.pending[MAX_BATCH:]
        self.last_flush = now
        return batch


_TRANSLATE_PROMPT = """Translate each caption line into {lang}. They are spoken sentences from a \
live meeting (speech-to-text, may contain errors). Keep names, numbers and meaning; no commentary. \
Lines already in {lang} stay as they are. Reply with ONLY a JSON array of {n} strings, in the same \
order.

{lines}"""


def translate_lines(client: Any, model: str, texts: list[str], language: str) -> list[str]:
    """Translate several lines in ONE call. Falls back to the original text on any failure."""
    if language not in LANGS or language == "auto" or not texts:
        return list(texts)
    lang = lang_name(language)
    prompt = _TRANSLATE_PROMPT.format(
        lang=lang, n=len(texts), lines=json.dumps(texts, ensure_ascii=False)
    )
    try:
        resp = generate(
            client,
            model,
            [prompt],
            types.GenerateContentConfig(
                temperature=0.0, max_output_tokens=2048, response_mime_type="application/json"
            ),
        )
        out = json.loads(getattr(resp, "text", "") or "[]")
        if isinstance(out, list) and len(out) == len(texts):
            return [str(x).strip() or texts[i] for i, x in enumerate(out)]
    except Exception:
        logger.warning("caption translation failed", exc_info=True)
    return list(texts)


def format_captions(lines: list[tuple[str, str]]) -> list[str]:
    """ "**Name:** text" messages; consecutive lines of one speaker are joined into one turn. Split
    to fit Discord's message limit."""
    turns: list[list[str]] = []
    for speaker, text in lines:
        if turns and turns[-1][0] == speaker:
            turns[-1][1] += " " + text
        else:
            turns.append([speaker, text])
    messages: list[str] = []
    cur = ""
    for speaker, text in turns:
        row = f"**{speaker}:** {text}"[:MAX_MESSAGE_CHARS]
        if cur and len(cur) + len(row) + 1 > MAX_MESSAGE_CHARS:
            messages.append(cur)
            cur = ""
        cur = f"{cur}\n{row}" if cur else row
    if cur:
        messages.append(cur)
    return messages


# ------------------------------------------------------------------ org default
_CAP = r"(?:คำบรรยาย|ซับ|captions?|subtitles?)"
_DEFAULT_ON_RE = re.compile(
    rf"แสดง{_CAP}(?:สด)?ทุก(?:ครั้ง|ประชุม)"
    rf"|(?:show|display)\s+(?:live\s+)?{_CAP}\s+in\s+every\s+meeting"
    rf"|{_CAP}\s+on\s+by\s+default|(?:show|turn\s+on)\s+{_CAP}\s+by\s+default",
    re.I,
)
_DEFAULT_OFF_RE = re.compile(
    rf"(?:ซ่อน|ปิด){_CAP}(?:สด)?ทุก(?:ครั้ง|ประชุม)"
    rf"|(?:hide|stop\s+showing)\s+(?:live\s+)?{_CAP}\s+(?:in\s+every\s+meeting|by\s+default)"
    rf"|{_CAP}\s+off\s+by\s+default|(?:turn\s+off|hide)\s+{_CAP}\s+by\s+default",
    re.I,
)


def parse_captions_default_command(text: str) -> bool | None:
    """True / False from "แสดงคำบรรยายทุกประชุม" / "captions on by default" and "ซ่อนคำบรรยายทุกประชุม"
    / "hide captions by default"; else None."""
    s = " ".join((text or "").split()).strip(" \t/,.:;!-")
    if not s or len(s) > 100:
        return None
    if _DEFAULT_OFF_RE.search(s):
        return False
    return True if _DEFAULT_ON_RE.search(s) else None


# ------------------------------------------------------------------ agent tool
def captions_tools(
    session_of: Callable[[], Any],
    set_captions: Callable[[Any, bool, str], dict],
    set_default: Callable[[bool], dict] | None = None,
) -> list[Callable]:
    """The `set_captions` agent tool for chat and voice. `session_of()` returns the meeting the
    asker is sitting in (their voice channel) or None; the scope is decided in code."""

    def set_captions_tool(on: bool, language: str = "original") -> dict:
        """Turn live captions (subtitles) of the voice meeting the asker is in on or off, optionally
        translated. Call it when someone asks e.g. "เปิดซับ", "เปิดซับให้หน่อย", "แปลเป็นอังกฤษให้ด้วย",
        "turn on captions in Thai", "แสดงคำบรรยาย", "ซ่อนคำบรรยาย", "hide captions", "ปิดซับ",
        "captions off". Captions are posted in the meeting's
        text chat. Only works for someone who is in the meeting's voice channel. Confirm in one
        short sentence.

        Args:
            on: true to show captions, false to stop them.
            language: "original" (as spoken), "th" (Thai) or "en" (English).
        """
        lang = normalize_language(language)
        if lang is None:
            return {"error": "language must be original, th or en"}
        session = session_of()
        if session is None:
            return {"error": "not_in_meeting_voice_channel"}
        return set_captions(session, bool(on), lang)

    set_captions_tool.__name__ = "set_captions"
    if set_default is None:
        return [set_captions_tool]

    def set_captions_default_tool(on: bool) -> dict:
        """Executives and admins only: make every future voice meeting of this server start with
        live captions in its chat (on) or without them (off, as before). Call it for "แสดงคำบรรยาย
        ทุกประชุม", "captions on by default", "ซ่อนคำบรรยายทุกประชุม", "hide captions by default".
        Not for the meeting happening now (that is set_captions). Confirm in one short sentence.

        Args:
            on: true to start meetings with captions, false to stop doing that.
        """
        return set_default(bool(on))

    set_captions_default_tool.__name__ = "set_captions_default"
    return [set_captions_tool, set_captions_default_tool]
