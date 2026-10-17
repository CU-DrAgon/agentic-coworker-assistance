"""Meeting minutes as a document with a speaker-by-speaker transcript ("who said what").

Pure logic: grouping lines into turns, talk-time stats, the optional "clean" pass (one cheap-model
call per chunk that may only rewrite the TEXT of a turn; the speaker / time skeleton stays in code),
the Markdown document, and the `meeting_document` agent tool. Discord glue (who may receive the
file, private delivery, memory lookups) lives in `meeting_doc_bot.py` and `MeetingManager`.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from google.genai import types

from . import attendance as att
from .i18n import LANGS, base_lang, t
from .ideas_pdf import markdown_docx, markdown_pdf
from .llm import generate
from .minutes import latest_visible

logger = logging.getLogger(__name__)

SALID_SPEAKER = "สลิด"  # speaker name used for Salid's own spoken answers
MERGE_GAP_S = 60.0  # consecutive lines of one speaker closer than this are one turn
CLEAN_CHUNK_CHARS = 6000
CLEAN_CHUNK_TURNS = 40
MAX_TRANSCRIPT_CHARS = 150_000
FORMATS = ("pdf", "docx", "md")
MODES = ("verbatim", "clean")

Line = tuple[datetime, str, str]  # (at, speaker, text)


@dataclass(frozen=True)
class Turn:
    at: datetime
    speaker: str
    text: str


def group_turns(lines: Iterable[Line], gap_s: float = MERGE_GAP_S) -> list[Turn]:
    """Consecutive lines of the same speaker (each within `gap_s` of the previous one) become one
    turn at the first line's time. Lines are ordered by time first."""
    out: list[Turn] = []
    last_at: datetime | None = None
    for at, speaker, text in sorted(lines, key=lambda ln: ln[0]):
        text = " ".join((text or "").split())
        if not text:
            continue
        if (
            out
            and out[-1].speaker == speaker
            and last_at is not None
            and (at - last_at).total_seconds() <= gap_s
        ):
            out[-1] = Turn(out[-1].at, speaker, f"{out[-1].text} {text}")
        else:
            out.append(Turn(at, speaker, text))
        last_at = at
    return out


def speaker_stats(turns: Iterable[Turn]) -> list[tuple[str, int, int]]:
    """(speaker, number of turns, share of talk in whole percent by characters), busiest first.
    Salid's own answers are not counted."""
    count: dict[str, int] = defaultdict(int)
    chars: dict[str, int] = defaultdict(int)
    for tn in turns:
        if tn.speaker == SALID_SPEAKER:
            continue
        count[tn.speaker] += 1
        chars[tn.speaker] += len(tn.text)
    total = sum(chars.values()) or 1
    rows = [(s, count[s], round(100 * chars[s] / total)) for s in count]
    return sorted(rows, key=lambda r: (-r[2], r[0]))


# ------------------------------------------------------------------ clean mode
_CLEAN_PROMPT = {
    "th": (
        'ด้านล่างคือข้อความที่ถอดจากเสียงประชุมอัตโนมัติ เป็น JSON list ของ {{"i": เลขลำดับ, "t": ข้อความ}}. '
        "แก้เฉพาะคำที่ถอดผิดอย่างชัดเจน ตัดคำฟุ่มเฟือย (เอ่อ อืม แบบว่า) และจัดเว้นวรรคให้อ่านง่าย "
        "ห้ามเปลี่ยนความหมาย ห้ามเพิ่มหรือตัดเนื้อหาสำคัญ ห้ามแก้ตัวเลข ชื่อ วันที่ ห้ามรวมหรือแยกรายการ "
        "ถ้าไม่แน่ใจให้คงข้อความเดิม ตอบเป็น JSON list รูปแบบเดิมเท่านั้น จำนวนรายการเท่าเดิม\n\n{items}"
    ),
    "en": (
        'Below is text auto-transcribed from a meeting, as a JSON list of {{"i": index, "t": text}}. '
        "Fix only clearly wrong transcription, remove filler words (um, uh, you know) and tidy spacing. "
        "Never change the meaning, never add or drop substantive content, never change numbers, names "
        "or dates, never merge or split items. When unsure keep the original. Answer with the same JSON "
        "list only, same number of items.\n\n{items}"
    ),
}


def _digits(s: str) -> set[str]:
    return set(re.findall(r"\d+", s))


def _acceptable(orig: str, new: Any) -> bool:
    if not isinstance(new, str):
        return False
    new = " ".join(new.split())
    if not new:
        return False
    if len(orig) >= 12 and not (0.4 * len(orig) <= len(new) <= 1.5 * len(orig) + 20):
        return False
    return _digits(orig) <= _digits(new)  # numbers are never altered or dropped


def _chunks(turns: list[Turn]) -> list[list[int]]:
    out: list[list[int]] = []
    cur: list[int] = []
    size = 0
    for i, tn in enumerate(turns):
        if cur and (size + len(tn.text) > CLEAN_CHUNK_CHARS or len(cur) >= CLEAN_CHUNK_TURNS):
            out.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += len(tn.text)
    if cur:
        out.append(cur)
    return out


def clean_turns(client: Any, model: str, turns: list[Turn], lang: str = "en") -> list[Turn]:
    """The turns with obvious transcription errors / filler words fixed. Only the text of a turn is
    sent and replaced: speakers and times are kept from `turns`, so the model cannot merge or
    reassign anyone. Any failure of a chunk (or an answer that changes numbers, length, or the item
    count) keeps that chunk verbatim."""
    out = list(turns)
    if client is None:
        return out
    prompt = _CLEAN_PROMPT[base_lang(lang)]
    for idx in _chunks(turns):
        try:
            items = json.dumps(
                [{"i": n, "t": turns[i].text} for n, i in enumerate(idx)], ensure_ascii=False
            )
            resp = generate(
                client,
                model,
                [prompt.format(items=items)],
                types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json"),
            )
            data = json.loads(getattr(resp, "text", "") or "")
            if isinstance(data, dict):
                data = data.get("items") or []
            if not isinstance(data, list) or len(data) != len(idx):
                continue
            for n, i in enumerate(idx):
                row = data[n]
                if not isinstance(row, dict) or row.get("i") != n:
                    continue
                if _acceptable(turns[i].text, row.get("t")):
                    out[i] = Turn(turns[i].at, turns[i].speaker, " ".join(row["t"].split()))
        except Exception:
            logger.warning(
                "transcript clean-up failed for a chunk; keeping it verbatim", exc_info=True
            )
    return out


# ------------------------------------------------------------------ the document
@dataclass
class MeetingInfo:
    room: str = ""
    started_at: datetime | None = None
    ended_at: datetime | None = None
    attendees: list[str] = field(default_factory=list)
    summary: str = ""
    attendance: list[att.Presence] = field(default_factory=list)  # who was in the room, when


def file_stem(started_at: datetime | None) -> str:
    return f"minutes-{started_at:%Y-%m-%d-%H%M}" if started_at else "minutes"


def _summary_sections(summary: str) -> str:
    """The stored summary with its **bold heading** lines turned into real headings."""
    rows = []
    for line in (summary or "").strip().splitlines():
        m = re.fullmatch(r"\s*\*\*(.+?)\*\*\s*:?\s*", line)
        rows.append(f"## {m.group(1).strip()}" if m else line)
    return "\n".join(rows)


def _hm(at: datetime | None) -> str:
    return f"{at:%H:%M}" if at else ""


def render_markdown(
    info: MeetingInfo,
    turns: list[Turn],
    lang: str,
    *,
    include_transcript: bool = True,
    stats: bool = True,
    transcript_note: str = "",
) -> str:
    """The minutes document. `transcript_note` replaces the heading note (e.g. "cleaned")."""
    lang = lang if lang in LANGS and lang != "auto" else "en"
    out = [f"# {t('mdoc_title', lang)}", ""]
    meta = []
    if info.room:
        meta.append(f"- **{t('mdoc_room', lang)}:** {info.room}")
    if info.started_at:
        meta.append(f"- **{t('mdoc_date', lang)}:** {info.started_at:%Y-%m-%d}")
        span = _hm(info.started_at)
        if info.ended_at:
            span += f" - {_hm(info.ended_at)}"
            mins = max(0, round((info.ended_at - info.started_at).total_seconds() / 60))
            span += f" ({t('mdoc_minutes', lang, n=mins)})"
        meta.append(f"- **{t('mdoc_time', lang)}:** {span}")
    if info.attendees:
        meta.append(f"- **{t('mdoc_attendees', lang)}:** {', '.join(info.attendees)}")
    out += meta + [""]
    if info.attendance:
        out += [
            f"## {t('mdoc_attendance', lang)}",
            "",
            f"| {t('mdoc_col_person', lang)} | {t('mdoc_col_present', lang)} "
            f"| {t('mdoc_col_total', lang)} |",
            "|---|---|---|",
            *[
                f"| {att.label(p.name, lang)} | "
                + ", ".join(f"{att.hm(a)}–{att.hm(b)}" for a, b in p.spans)
                + f" | {p.minutes} |"
                for p in info.attendance
            ],
            "",
            t("attend_total", lang, n=len(info.attendance)),
            "",
        ]
    if info.summary.strip():
        out += [_summary_sections(info.summary), ""]
    if include_transcript:
        out += [f"## {t('mdoc_transcript', lang)}", ""]
        out += [f"*{transcript_note or t('mdoc_note', lang)}*", ""]
        if not turns:
            out += [t("mdoc_no_transcript", lang), ""]
        else:
            if stats:
                rows = speaker_stats(turns)
                if rows:
                    out += [
                        f"### {t('mdoc_stats', lang)}",
                        "",
                        f"| {t('mdoc_col_person', lang)} | {t('mdoc_col_turns', lang)} "
                        f"| {t('mdoc_col_share', lang)} |",
                        "|---|---|---|",
                        *[f"| {s} | {n} | {p}% |" for s, n, p in rows],
                        "",
                    ]
            for tn in turns:
                who = t("mdoc_salid", lang) if tn.speaker == SALID_SPEAKER else tn.speaker
                out += [f"**{_hm(tn.at)} {who}:** {tn.text}", ""]
    return "\n".join(out).rstrip() + "\n"


def build_file(
    fmt: str,
    markdown: str,
    *,
    org: str,
    lang: str,
    generated_at: datetime,
    fonts: Callable[[], Any],
) -> bytes:
    """The document as bytes (pdf / docx / md)."""
    title = t("mdoc_title", lang)
    if fmt == "md":
        return markdown.encode("utf-8")
    if fmt == "docx":
        return markdown_docx(title, markdown)
    return markdown_pdf(
        org=org, title=title, markdown=markdown, lang=lang, generated_at=generated_at, fonts=fonts()
    )


# ------------------------------------------------------------------ the agent tool
def pick_record(
    records: list[dict], meeting: str, asker_id: int, can_view_voice: Callable[[int], bool]
) -> dict | None:
    """The newest stored meeting the asker attended or can see the room of. `meeting`: "last" (or
    empty), a part of the room name, or a date YYYY-MM-DD."""
    want = (meeting or "last").strip().casefold()
    pool = records
    if want not in ("last", "latest", "", "ล่าสุด", "เมื่อกี้"):
        pool = [
            r
            for r in records
            if want in str(r.get("voice_channel_name", "")).casefold()
            or want == str(r.get("started_at", ""))[:10]
            or want == str(r.get("ended_at", ""))[:10]
        ]
    return latest_visible(pool, asker_id, can_view_voice)


@dataclass
class MeetingDocCtx:
    """Everything the tool needs for one request; scope is decided by the callables, in code."""

    asker_id: int
    lang: str
    org: str
    records: Callable[[], list[dict]]
    can_view_voice: Callable[[int], bool]
    lines: Callable[[dict], list[Line]]  # the meeting's voice lines (memory rules apply)
    audience_ok: Callable[[dict], bool]  # may every reader of this conversation see the room?
    send_private: Callable[[str, bytes, str], bool]  # (file name, data, caption) -> delivered
    fonts: Callable[[], Any]
    now: Callable[[], datetime]
    client: Any = None
    model: str = ""
    outbox: list[tuple[str, bytes]] = field(default_factory=list)

    def info_of(self, rec: dict) -> MeetingInfo:
        def when(key: str) -> datetime | None:
            try:
                return datetime.fromisoformat(str(rec.get(key)))
            except Exception:
                return None

        return MeetingInfo(
            room=str(rec.get("voice_channel_name") or ""),
            started_at=when("started_at"),
            ended_at=when("ended_at"),
            attendees=list(rec.get("attendee_names") or []),
            summary=str(rec.get("summary") or ""),
        )

    def meeting_document(
        self,
        meeting: str = "last",
        format: str = "pdf",  # noqa: A002
        mode: str = "verbatim",
        include_transcript: bool = True,
    ) -> dict:
        fmt = (format or "pdf").casefold().lstrip(".")
        fmt = {"markdown": "md", "word": "docx", "doc": "docx"}.get(fmt, fmt)
        if fmt not in FORMATS:
            return {"status": "error", "error": "format must be pdf, docx or md"}
        mode = (mode or "verbatim").casefold()
        if mode not in MODES:
            mode = "verbatim"
        rec = pick_record(self.records(), meeting, self.asker_id, self.can_view_voice)
        if rec is None:
            return {
                "status": "no_meeting",
                "note": "no finished meeting this requester attended or can see",
            }
        info = self.info_of(rec)
        turns: list[Turn] = []
        if include_transcript:
            try:
                turns = group_turns(self.lines(rec))
            except Exception:
                logger.warning("could not rebuild the meeting transcript", exc_info=True)
        cleaned = False
        if turns and mode == "clean":
            new = clean_turns(self.client, self.model, turns, self.lang)
            cleaned = new != turns
            turns = new
        md = render_markdown(
            info,
            turns,
            self.lang,
            include_transcript=include_transcript,
            transcript_note=t("mdoc_note_clean", self.lang) if cleaned else "",
        )
        if len(md) > MAX_TRANSCRIPT_CHARS + len(info.summary):
            md = md[: MAX_TRANSCRIPT_CHARS + len(info.summary)]
        name = f"{file_stem(info.started_at)}.{fmt}"
        try:
            data = build_file(
                fmt, md, org=self.org, lang=self.lang, generated_at=self.now(), fonts=self.fonts
            )
        except Exception:
            logger.exception("meeting document failed")
            if fmt == "md":
                return {"status": "error", "error": "could not build the file"}
            fmt, name, data = "md", f"{file_stem(info.started_at)}.md", md.encode("utf-8")
        out = {
            "status": "ok",
            "file": name,
            "format": fmt,
            "room": info.room,
            "mode": mode if cleaned or mode == "verbatim" else "verbatim",
            "transcript_available": bool(turns),
        }
        if include_transcript and not turns:
            out["note"] = "no transcript is available for this meeting (not kept); summary only"
        if self.audience_ok(rec):
            self.outbox.append((name, data))
            out["delivery"] = "attached to the reply"
            return out
        caption = t("mdoc_private_caption", self.lang, room=info.room)
        if self.send_private(name, data, caption):
            out["delivery"] = "sent_privately"
            out["note"] = (
                (out.get("note", "") + " ").lstrip()
                + "The file was sent to the requester privately (this place has readers who "
                "cannot see the meeting room). Say so; do not paste the transcript."
            )
            return out
        return {"status": "error", "error": "could not deliver the file privately (DMs closed?)"}


def meeting_document_tools(ctx: MeetingDocCtx) -> list[Callable]:
    """The function `_agent_kwargs` appends to `extra_tools` (its docstring is the tool description).
    Attached files are collected in `ctx.outbox` (`outbox` attribute: see `qa.tool_files`)."""

    def meeting_document(
        meeting: str = "last",
        format: str = "pdf",  # noqa: A002
        mode: str = "verbatim",
        include_transcript: bool = True,
    ) -> dict:
        """Make the minutes of a FINISHED voice meeting as a document: title, room, date, times,
        attendees, summary, decisions, action items and (by default) the speaker-by-speaker
        transcript "who said what". Use for "ขอรายงานการประชุมเมื่อกี้เป็นเอกสาร", "ขอบทพูดของแต่ละคน
        ในประชุมเมื่อวาน", "export the meeting transcript as Word". Only for people who attended or
        can see the voice room; the file goes to the requester (privately when this place has other
        readers). Reply in one or two lines saying the document is attached / sent; never paste the
        transcript.

        Args:
            meeting: "last" (default), part of the voice room's name, or a date YYYY-MM-DD.
            format: "pdf" (default), "docx" (Word) or "md".
            mode: "verbatim" (default, as transcribed) or "clean" (obvious transcription errors and
                filler words fixed, nothing else changed).
            include_transcript: false for the summary sections only.
        """
        try:
            return ctx.meeting_document(meeting, format, mode, include_transcript)
        except Exception:
            logger.exception("meeting_document failed")
            return {"status": "error", "error": "the meeting document failed; try again"}

    meeting_document.outbox = ctx.outbox  # type: ignore[attr-defined]
    return [meeting_document]
