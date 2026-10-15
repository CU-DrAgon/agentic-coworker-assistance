"""Layer 2 of Salid's memory: the cheap-model day summary of one channel.

One call updates a channel-day's summary: the previous summary plus only the messages that arrived
since the last update go in, the new summary comes out. Each message is therefore read by the cheap
model about once. The call is blocking (run it with asyncio.to_thread), uses JSON mode at
temperature 0, and raises on failure; the scheduler retries later. Only counts are logged, never
message contents.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from zoneinfo import ZoneInfo

from google.genai import types
from pydantic import BaseModel, Field

from .llm import generate
from .memory_store import MemoryMessage

logger = logging.getLogger(__name__)

LINE_CHARS = 700  # per message sent to the model (image descriptions are long)
MAX_BATCH = 150  # messages per model call; larger backlogs are split into sequential calls
MAX_OUTPUT_TOKENS = 1800
RENDER_CHARS = 2500  # cap of one rendered summary handed to the answering model


class Topic(BaseModel):
    title: str
    details: str = ""
    people: list[str] = Field(default_factory=list)


class ActionItem(BaseModel):
    who: str = ""
    what: str
    due: str = ""


class SummaryDoc(BaseModel):
    overview: str = ""
    topics: list[Topic] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


LANG_NAME = {"th": "Thai", "en": "English"}

PROMPT = """You maintain the day summary of ONE Discord channel for a team assistant.
Language of the summary: {language}. Keep names, amounts, dates and IDs exactly as written; do not invent anything; \
if something is unclear leave it out. Messages starting with "[voice]" are spoken in a voice meeting, "Salid" is the assistant.
Channel: #{channel}   Day: {day}

Previous summary (JSON, may be empty):
{previous}

New messages since then (format HH:MM name: text), oldest first:
{messages}

Return the UPDATED summary of the whole day as JSON with keys: overview (2-4 sentences), \
topics (list of {{title, details, people}}), decisions (list of strings), \
action_items (list of {{who, what, due}}; who = the person's name as written; due only if stated, as YYYY-MM-DD when the date is given or can be worked out from the day above, else as written), \
numbers (amounts, dates, IDs mentioned that matter, each with its context), open_questions (list of strings). \
Merge the new messages into the previous summary: keep what is still true, add what is new, \
update items that changed. Be concise."""


def format_lines(messages: Sequence[MemoryMessage], zone: ZoneInfo) -> str:
    out = []
    for m in messages:
        who = "Salid" if m.kind == "salid" else m.author_name or str(m.author_id)
        tag = "[voice] " if m.kind == "voice" else ""
        out.append(f"{m.created_at.astimezone(zone):%H:%M} {tag}{who}: {m.text[:LINE_CHARS]}")
    return "\n".join(out)


def build_prompt(
    previous: dict | None,
    messages: Sequence[MemoryMessage],
    *,
    channel: str,
    day: str,
    lang: str,
    zone: ZoneInfo,
) -> str:
    return PROMPT.format(
        language=LANG_NAME.get(lang, "Thai"),
        channel=channel,
        day=day,
        previous=json.dumps(previous, ensure_ascii=False) if previous else "{}",
        messages=format_lines(messages, zone),
    )


def update_summary(
    client,
    model: str,
    previous: dict | None,
    messages: Sequence[MemoryMessage],
    *,
    channel: str,
    day: str,
    lang: str,
    zone: ZoneInfo,
) -> dict:
    """The new summary dict. Splits big backlogs into MAX_BATCH sized calls."""
    current = previous
    for i in range(0, len(messages), MAX_BATCH):
        batch = messages[i : i + MAX_BATCH]
        prompt = build_prompt(current, batch, channel=channel, day=day, lang=lang, zone=zone)
        resp = generate(
            client,
            model,
            prompt,
            types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
                response_schema=SummaryDoc,
            ),
        )
        current = SummaryDoc.model_validate(json.loads(resp.text)).model_dump()
        logger.info("memory summary call: %d new messages", len(batch))
    return current or SummaryDoc().model_dump()


def render_summary(summary: dict, max_chars: int = RENDER_CHARS) -> str:
    """The summary as compact text for the answering model."""
    lines: list[str] = []
    if summary.get("overview"):
        lines.append(summary["overview"])
    for tp in summary.get("topics") or []:
        who = f" ({', '.join(tp.get('people') or [])})" if tp.get("people") else ""
        lines.append(f"- Topic: {tp.get('title', '')}{who}: {tp.get('details', '')}".rstrip(": "))
    for label, key in (
        ("Decision", "decisions"),
        ("Number", "numbers"),
        ("Open", "open_questions"),
    ):
        lines += [f"- {label}: {x}" for x in summary.get(key) or []]
    for a in summary.get("action_items") or []:
        due = f" (due {a['due']})" if a.get("due") else ""
        lines.append(f"- Action: {a.get('who') or '?'}: {a.get('what', '')}{due}")
    return "\n".join(lines)[:max_chars]
