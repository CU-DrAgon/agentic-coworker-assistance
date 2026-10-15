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


def pick_interjection(
    client, model: str, history: list[GateLine], candidate_ids: list[str], *, channel_id: int
) -> str | None:
    """Join-mode batch gate: the id of the candidate message Salid has something genuinely useful
    to add to, or None."""
    prompt = (
        f"{_ABOUT}\nSalid was not called, but may chime in. Recent chat, oldest first, each line "
        f"prefixed with its message id:\n{_format(history, with_ids=True)}\n\n"
        f"Candidate message ids: {', '.join(candidate_ids)}.\n"
        "Does Salid have something genuinely useful to add to one of the candidates (it can look up "
        "spending and receipts; do not interrupt casual chat or questions meant for other people)? "
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
