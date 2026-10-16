"""Typed commands that control Salid's own messages and running work.

- delete a Salid message: reply to it with just "ลบ" / "delete" ...
- cancel the running deep task: "ยกเลิกงาน" / "cancel task"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ANSWER_TTL_S = 24 * 3600  # how long the chunk ids of an answer are remembered
_TRIM = " .!?,:;\"'`~-_*"

# Bare words that, on a receipt reply, only void the receipt (today's behaviour).
_PLAIN = {"ลบ", "delete"}
# Words that explicitly ask to remove the message itself (on a receipt reply: void + delete).
_MESSAGE = {
    "ลบข้อความ",
    "ยกเลิกข้อความ",
    "เอาออก",
    "remove",
    "unsend",
    "delete message",
    "remove message",
    "delete this message",
}
_CANCEL_TASK = re.compile(
    r"^(?:ยกเลิก\s?งาน(?:\s?ใหญ่)?|หยุด\s?งาน(?:\s?ใหญ่)?"
    r"|(?:cancel|stop|abort)\s+(?:the\s+)?(?:deep\s+|big\s+)?task)$",
    re.I,
)


def parse_delete_message(text: str) -> str | None:
    """ "plain" for "ลบ" / "delete", "message" for the explicit message-removal words, else None.
    The whole (cleaned) text must be the command, so a sentence that merely contains it is not."""
    s = re.sub(r"\s+", " ", (text or "").strip().casefold().strip(_TRIM))
    if s in _PLAIN:
        return "plain"
    if s.replace(" ", "") in {w.replace(" ", "") for w in _MESSAGE}:
        return "message"
    return None


def parse_cancel_task(text: str) -> bool:
    s = re.sub(r"\s+", " ", (text or "").strip().strip(_TRIM))
    return bool(_CANCEL_TASK.match(s))


@dataclass
class AnswerRecord:
    """The messages of one Salid answer (a long answer is sent in several chunks)."""

    channel_id: int
    asker_id: int
    message_ids: list[int] = field(default_factory=list)
    at: float = 0.0


class AnswerLog:
    """In-memory map message id -> answer record, for about 24 h."""

    def __init__(self, clock) -> None:
        self._clock = clock
        self._by_msg: dict[int, AnswerRecord] = {}

    def add(self, channel_id: int, asker_id: int, message_ids: list[int]) -> None:
        now = self._clock()
        self._prune(now)
        rec = AnswerRecord(channel_id, asker_id, list(message_ids), now)
        for mid in message_ids:
            self._by_msg[mid] = rec

    def get(self, message_id: int) -> AnswerRecord | None:
        rec = self._by_msg.get(message_id)
        if rec is not None and self._clock() - rec.at > ANSWER_TTL_S:
            self._by_msg.pop(message_id, None)
            return None
        return rec

    def forget(self, rec: AnswerRecord) -> None:
        for mid in rec.message_ids:
            self._by_msg.pop(mid, None)

    def _prune(self, now: float) -> None:
        for mid in [m for m, r in self._by_msg.items() if now - r.at > ANSWER_TTL_S]:
            del self._by_msg[mid]
