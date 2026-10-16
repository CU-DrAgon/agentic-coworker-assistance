"""Native Discord polls: validation, the stored record and the one-line result.

Salid posts a `discord.Poll` (so people vote with Discord's own UI) and keeps a small record
(`salid_polls`, same `DocStore` as tasks) to know which polls it created, when they end and
whether the result line was already posted. Discord glue is `salid.polls_bot`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from .coo_store import DocStore
from .i18n import t

POLLS = "salid_polls"
MIN_OPTIONS, MAX_OPTIONS = 2, 10  # Discord's limits
MAX_OPTION_CHARS = 55
MAX_QUESTION_CHARS = 300
MIN_HOURS, MAX_HOURS = 1, 768  # Discord allows up to 32 days
GRACE = timedelta(minutes=15)  # after the end, how long to wait for Discord to finalise


def clean_poll(question: str, options, hours) -> tuple[str, list[str], int] | dict:
    """(question, options, whole hours) within Discord's limits, or an error dict."""
    q = (question or "").strip()
    if not q:
        return {"error": "need_question"}
    seen: set[str] = set()
    opts: list[str] = []
    for o in options or []:
        o = str(o).strip()
        if o and o.casefold() not in seen:
            seen.add(o.casefold())
            opts.append(o)
    if len(opts) < MIN_OPTIONS:
        return {"error": "need_at_least_2_options"}
    if len(opts) > MAX_OPTIONS:
        return {"error": "too_many_options", "max": MAX_OPTIONS}
    if len(q) > MAX_QUESTION_CHARS or any(len(o) > MAX_OPTION_CHARS for o in opts):
        return {
            "error": "too_long",
            "max_question_chars": MAX_QUESTION_CHARS,
            "max_option_chars": MAX_OPTION_CHARS,
        }
    try:
        h = int(round(float(hours)))
    except (TypeError, ValueError):
        h = 24
    return q, opts, min(max(h, MIN_HOURS), MAX_HOURS)


def new_record(
    *,
    guild_id: int,
    channel_id: int,
    message_id: int,
    creator_id: int,
    question: str,
    options: list[str],
    hours: int,
    multiple: bool,
    now: datetime | None = None,
) -> dict:
    now = (now or datetime.now(UTC)).astimezone(UTC)
    return {
        "id": str(message_id),
        "guild_id": guild_id,
        "channel_id": channel_id,
        "message_id": message_id,
        "creator_id": creator_id,
        "question": question,
        "options": options,
        "multiple": multiple,
        "created_at": now.isoformat(timespec="seconds"),
        "ends_at": (now + timedelta(hours=hours)).isoformat(timespec="seconds"),
        "done": False,
    }


class PollRepo:
    def __init__(self, docs: DocStore):
        self.docs = docs

    def put(self, rec: dict) -> dict:
        self.docs.put(POLLS, rec["guild_id"], rec["id"], rec)
        return rec

    def all(self, guild_id: int) -> list[dict]:
        return sorted(self.docs.find(POLLS, guild_id), key=lambda r: r["created_at"])

    def due(self, guild_id: int, now: datetime) -> list[dict]:
        return [
            r
            for r in self.all(guild_id)
            if not r["done"] and datetime.fromisoformat(r["ends_at"]) <= now
        ]

    def find(self, guild_id: int, ref: str | None, visible=lambda rec: True):
        """(record, candidates): `ref` = message id / link, part of the question, or empty / "last"
        for the newest poll `visible` allows."""
        recs = [r for r in self.all(guild_id) if visible(r)]
        if not recs:
            return None, []
        key = (ref or "").strip().casefold()
        digits = "".join(c for c in key.rsplit("/", 1)[-1] if c.isdigit())
        if key in ("", "last", "latest", "the last poll", "ล่าสุด", "โพลล่าสุด"):
            return recs[-1], []
        by_id = [r for r in recs if digits and r["id"] == digits]
        if by_id:
            return by_id[0], []
        hits = [r for r in recs if key in r["question"].casefold()]
        if len(hits) == 1:
            return hits[0], []
        return None, (hits or recs)[-5:]


def tally(options: list[str], counts: list[int]) -> dict:
    """Votes per option, the total and the winner(s) (all tied options)."""
    total = sum(counts)
    top = max(counts, default=0)
    winners = [o for o, c in zip(options, counts, strict=False) if c == top and top > 0]
    return {
        "options": [{"option": o, "votes": c} for o, c in zip(options, counts, strict=False)],
        "total_votes": total,
        "winners": winners,
    }


def result_line(question: str, options: list[str], counts: list[int], lang: str) -> str:
    """One line for the channel: who won, or that nobody voted."""
    r = tally(options, counts)
    if r["total_votes"] == 0:
        return t("poll_result_none", lang, question=question)
    if len(r["winners"]) > 1:
        return t(
            "poll_result_tie",
            lang,
            question=question,
            winners=" / ".join(r["winners"]),
            votes=max(counts),
        )
    return t(
        "poll_result",
        lang,
        question=question,
        winner=r["winners"][0],
        votes=max(counts),
        total=r["total_votes"],
    )
