"""Cheap yes/no gates that decide whether Salid should speak (call/join modes).

Both functions are blocking (run them with asyncio.to_thread), use one small JSON-mode call at
temperature 0 and fail quiet: any error or unparsable reply means "do not answer". Only
decisions are logged, never message contents.
"""

from __future__ import annotations

import json
import logging
import re
from typing import NamedTuple

from google.genai import types
from pydantic import BaseModel

from .llm import generate

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 50
LINE_CHARS = 300

# Free pre-filter for join mode: looks like a question or touches Salid's domain.
_QUESTION_RE = re.compile(
    r"\?|ไหม|มั้ย|หรือเปล่า|อะไร|ยังไง|อย่างไร|เท่าไหร่|เท่าไร|ใคร|ที่ไหน|เมื่อไหร่|ทำไม|กี่"
    r"|ค่าใช้จ่าย|งบ|บาท|ใบเสร็จ|เบิก|ประชุม|สรุป"
    # English: question words, and the same domain words (whole words only)
    r"|(?<![a-z])(?:what|how|when|where|who|why|which|can|could|should|does|is\s+there"
    r"|are\s+there|expenses?|costs?|budget|receipts?|reimburs\w*|meetings?|summar\w+|baht"
    r"|spend|spent)(?![a-z])",
    re.I,
)

_ABOUT = (
    "Salid (สลิด) is a bilingual (Thai and English) team assistant in a Discord server. It reads "
    "receipts into an expense ledger, answers questions about spending, and can summarize "
    "meetings. Messages are in Thai or English."
)


class GateLine(NamedTuple):
    id: str
    author: str
    text: str


class _Yes(BaseModel):
    answer: bool


class _Pick(BaseModel):
    answer: bool
    message_id: str = ""


def is_join_candidate(text: str) -> bool:
    """Free filter (no LLM): a question, or a message about Salid's domain."""
    return bool(_QUESTION_RE.search(text or ""))


# ---- free memory lookup helpers for stage 2 (no LLM)
HINT_TOTAL_CHARS = 600
HINT_HITS_PER_CANDIDATE = 3
MAX_KEYWORDS = 8
WINDOW_LEN = 6
MAX_WINDOWS = 4
_THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
_STOP_TH = (
    "หรือเปล่า|เท่าไหร่|เท่าไร|เมื่อไหร่|ที่ไหน|อย่างไร|ยังไง|อะไร|ทำไม|ใคร|ไหม|มั้ย|กี่"
    "|หน่อย|ครับ|ค่ะ|คะ|นะ|ด้วย|แล้ว|และ|ของ|ที่|ได้|เป็น|จะ|ให้|กับ|ก็|ใน|ว่า|ช่วย|สลิด"
    "|ไป|มา|มี|หรือ|ไม่|คือ"
)
_STOP_EN = {
    "what", "how", "when", "where", "who", "why", "which", "can", "could", "should", "does",
    "did", "the", "and", "for", "are", "was", "were", "has", "have", "any", "there", "this",
    "that", "with", "from", "about", "please", "much", "many", "anyone", "someone", "salid",
    "you", "our", "your", "his", "her", "its", "not", "but", "will", "would",
}  # fmt: skip
_STOP_TH_RE = re.compile(_STOP_TH)
_TOKEN_RE = re.compile(r"[0-9A-Za-z\u0E00-\u0E7F][0-9A-Za-z\u0E00-\u0E7F,.]*")


def extract_keywords(text: str) -> list[str]:
    """Deterministic keywords for the free memory lookup: tokens of >= 3 chars after removing
    question words and stopwords. Thai is written without spaces, so it is split on common
    particles and question words ("ค่าน้ำมันเดือนนี้เท่าไหร่" -> "ค่าน้ำมันเดือนนี้"); a long Thai
    run also contributes a few overlapping 6-character windows so that "น้ำแข็งซื้อ" still finds
    "ซื้อน้ำแข็ง". Windows can match by accident: the gate is told the hints may be irrelevant."""
    runs: list[str] = []
    for tok in _TOKEN_RE.findall(_STOP_TH_RE.sub(" ", text or "")):
        for part in re.split(r"[\s,.]+", tok):
            if len(part) >= 3 and part.casefold() not in _STOP_EN and part not in runs:
                runs.append(part)
    out: list[str] = []
    for run in runs:
        out.append(run)
    for run in runs:
        if len(run) > WINDOW_LEN + 1 and _THAI_RE.search(run):
            windows = [run[i : i + WINDOW_LEN] for i in range(0, len(run) - WINDOW_LEN + 1, 3)]
            out += [w for w in windows[:MAX_WINDOWS] if w not in out]
    return out[:MAX_KEYWORDS]


def hint_line(result: dict, per_hit: int = 160, exclude_text: str = "") -> str:
    """One compact line from a `chat_search` result: "memory has: [date #channel] snippet; ...".
    A message that starts with `exclude_text` (the candidate itself) is not a hint."""
    skip = " ".join((exclude_text or "").split())
    parts: list[str] = []
    for h in result.get("hits") or []:
        snippet = ""
        if h.get("summary_hits"):
            snippet = h["summary_hits"][0]
        else:
            for m in h.get("messages") or []:
                text = " ".join(str(m.get("text", "")).split())
                if not (skip and text.startswith(skip)):
                    snippet = text
                    break
        snippet = " ".join(str(snippet).split())[:per_hit]
        if snippet:
            parts.append(f"[{h.get('date')} #{h.get('channel')}] {snippet}")
        if len(parts) >= HINT_HITS_PER_CANDIDATE:
            break
    return "memory has: " + "; ".join(parts) if parts else ""


def fit_hints(hints: dict[str, str], total: int = HINT_TOTAL_CHARS) -> dict[str, str]:
    """Keep the hint lines within `total` characters in all (earlier candidates first)."""
    out: dict[str, str] = {}
    left = total
    for cid, line in hints.items():
        if not line or left <= 0:
            continue
        line = line[:left]
        out[cid] = line
        left -= len(line)
    return out


def _format(lines: list[GateLine], with_ids: bool = False) -> str:
    return "\n".join(
        (f"[{ln.id}] " if with_ids else "") + f"{ln.author}: {ln.text[:LINE_CHARS]}" for ln in lines
    )


def _decide(client, model: str, prompt: str, schema: type[BaseModel]):
    resp = generate(
        client,
        model,
        prompt,
        types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    return schema.model_validate(json.loads(resp.text))


def addressed_to_salid(
    client, model: str, history: list[GateLine], message: GateLine, *, channel_id: int
) -> bool:
    """In-window gate: is `message` addressed to Salid or does it expect Salid's input?"""
    prompt = (
        f"{_ABOUT}\nSalid just took part in this conversation. Recent chat, oldest first "
        f"(author 'สลิด' is Salid):\n{_format(history)}\n\nNew message:\n{_format([message])}\n\n"
        "Is the new message addressed to Salid, or does it expect Salid's input (a follow-up "
        "to Salid's last answer, a question Salid can answer)? Answer false if people are talking "
        'to each other about something else. Reply JSON: {"answer": true|false}.'
    )
    try:
        result = _decide(client, model, prompt, _Yes).answer
    except Exception:
        logger.warning("gate (in-window) failed; staying quiet", exc_info=True)
        result = False
    logger.info("gate in_window channel=%s answer=%s", channel_id, result)
    return result


class _Notes(BaseModel):
    notes: list[str] = []


def notes_from_transcript(
    client, model: str, lines: list[GateLine], *, channel_id: int
) -> list[str]:
    """Batch gate over a meeting transcript: one short line per real decision, action item (owner
    and date when said) or number someone asked Salid to remember; [] for chit-chat. Fails quiet."""
    prompt = (
        f"{_ABOUT}\nMeeting transcript, oldest first:\n{_format(lines)}\n\n"
        "List ONLY things worth writing down: decisions, action items (who does what, by when if "
        "said), numbers or facts someone asked Salid to remember. Casual chat, greetings and "
        "discussion without an outcome are NOT notes. One short line each in the language spoken, "
        'at most 3, never quote speech. Reply JSON: {"notes": ["..."]} ({"notes": []} if none).'
    )
    try:
        resp = generate(
            client,
            model,
            prompt,
            types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=300,
                response_mime_type="application/json",
                response_schema=_Notes,
            ),
        )
        notes = [n.strip() for n in _Notes.model_validate(json.loads(resp.text)).notes if n.strip()]
    except Exception:
        logger.warning("gate (notes) failed; no notes", exc_info=True)
        notes = []
    logger.info("gate notes channel=%s count=%d", channel_id, len(notes))
    return notes


def pick_interjection(
    client,
    model: str,
    history: list[GateLine],
    candidate_ids: list[str],
    *,
    channel_id: int,
    hints: dict[str, str] | None = None,
) -> str | None:
    """Join-mode batch gate: the id of the candidate message Salid has something genuinely useful
    to add to, or None."""
    hint_block = ""
    if hints:
        hint_block = (
            "Free memory lookup per candidate (keyword search of past chat; may be irrelevant):\n"
            + "\n".join(f"[{cid}] {line}" for cid, line in hints.items())
            + "\n"
        )
    prompt = (
        f"{_ABOUT}\nSalid was not called, but may chime in. Recent chat, oldest first, each line "
        f"prefixed with its message id:\n{_format(history, with_ids=True)}\n\n"
        f"Candidate message ids: {', '.join(candidate_ids)}.\n"
        + hint_block
        + "Does Salid have something genuinely useful to add to one of the candidates (it can look "
        "up spending and receipts; do not interrupt casual chat or questions meant for other "
        "people)? Prefer candidates that memory or the expense ledger can answer; answer false if "
        "the memory hints are irrelevant to the question. "
        'If yes reply {"answer": true, "message_id": "<id>"}, otherwise {"answer": false}.'
    )
    picked = None
    try:
        r = _decide(client, model, prompt, _Pick)
        if r.answer and r.message_id in candidate_ids:
            picked = r.message_id
    except Exception:
        logger.warning("gate (join) failed; staying quiet", exc_info=True)
    logger.info("gate join channel=%s answer=%s", channel_id, picked is not None)
    return picked
