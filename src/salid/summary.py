"""Direct summary requests: the /สรุป question, and "summarize this message" detection."""

from __future__ import annotations

import re
from datetime import date, timedelta

from .i18n import t
from .naming import command_text

PERIODS = ("today", "yesterday", "7d", "30d")
_SUMMARIZE_RE = re.compile(
    r"^(?:ช่วย)?(?:สรุป(?:ให้|ที|หน่อย)?(?:\s?(?:ข้อความ|อันนี้|นี้|นี่))?"
    r"|(?:please\s+)?summari[sz]e(?:\s+(?:this|it|that))?|tl\s?;?\s?dr)$",
    re.I,
)


def period_range(period: str, today: date) -> tuple[date, date]:
    """Inclusive (start, end) local dates; an unknown period means today."""
    if period == "yesterday":
        d = today - timedelta(days=1)
        return d, d
    if period == "7d":
        return today - timedelta(days=6), today
    if period == "30d":
        return today - timedelta(days=29), today
    return today, today


def summary_question(
    lang: str,
    *,
    period: str,
    today: date,
    tz: str,
    channel_name: str | None = None,
    topic: str | None = None,
) -> str:
    """The question /สรุป sends through the normal ask. `channel_name=None`: all channels."""
    start, end = period_range(period, today)
    dates = (
        t("summary_dates_one", lang, start=start.isoformat(), tz=tz)
        if start == end
        else t("summary_dates_range", lang, start=start.isoformat(), end=end.isoformat(), tz=tz)
    )
    q = (
        t("summary_q_channel", lang, channel=channel_name, dates=dates)
        if channel_name
        else t("summary_q_all", lang, dates=dates)
    )
    if topic and topic.strip():
        q += t("summary_focus", lang, topic=topic.strip()[:200])
    return q


def is_summarize_request(text: str) -> bool:
    """True when the whole (mention-stripped) message is just "สรุป" / "summarize" / "tl;dr"."""
    rest, _named = command_text(text or "")
    rest = rest.strip(" \t\n.!?…")
    return bool(rest) and len(rest) <= 30 and _SUMMARIZE_RE.match(rest) is not None


def message_summary_question(lang: str, author: str, text: str) -> str:
    q = t("summary_msg_q", lang, author=author)
    if text.strip():
        q += "\n\n" + t("summary_msg_text", lang, text=text.strip()[:4000])
    return q
