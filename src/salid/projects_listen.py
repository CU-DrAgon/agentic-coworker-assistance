"""Projects that update themselves from normal chat.

People talk, update each other and meet as usual; Salid listens. `ProjectListener.on_summary` is
chained after `TaskTracker.on_summary` (the memory scheduler's hook, see `coo.py`). For each batch
of new messages it makes ONE cheap-model call that lists the project-level changes the messages
clearly state as decided or done (status, deadline, priority, a milestone finished, a new risk, a
short progress line). Everything the model returns is checked in code against the real projects:
unknown ids are dropped, a value equal to the current one is dropped, a completed / cancelled
project only takes a progress line, owners are never changed from chat, and an update whose quote
is not in the messages is dropped. Changes go through `projects.apply_updates`, so the change log
carries the quoted line as the reason. Then Salid says what it recorded, once per batch, in the
room (throttled, never in DM scope, per-guild switch). Everything here is blocking: it runs in the
summary's worker thread, and `note` must be safe to call from there.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import date

from google.genai import types
from pydantic import BaseModel, Field

from .i18n import CATALOG, resolve_lang, t
from .llm import generate
from .memory_access import is_dm_scope
from .memory_store import MemoryMessage
from .memory_summary import format_lines
from .projects import (
    DONE_STATUSES,
    MAX_UPDATES,
    PRIORITIES,
    STATUSES,
    Person,
    Project,
    ProjectRepo,
    Risk,
    Update,
    apply_updates,
    next_item_id,
    record_change,
    stamp,
)
from .tasks import MAX_LINES_FOR_CHECK, match_person

logger = logging.getLogger(__name__)

KINDS = ("status", "deadline", "priority", "milestone_done", "risk", "note")
LAST_WINS = ("status", "deadline", "priority")
MAX_UPDATES_PER_BATCH = 8
MAX_PROJECTS_IN_PROMPT = 30
NOTE_GAP_S = 600.0  # at most one note per channel in 10 minutes
MAX_ITEMS_IN_NOTE = 6
QUOTE_CHARS = 200
_TH_MONTHS = ("ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค.")  # fmt: skip
_EN_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")  # fmt: skip


class _Heard(BaseModel):
    project_id: str = ""
    kind: str = ""
    value: str = ""
    milestone_id: str = ""
    quote: str = ""
    speaker: str = ""


class _HeardList(BaseModel):
    updates: list[_Heard] = Field(default_factory=list)


LISTEN_PROMPT = """You keep a team's project records up to date from their normal chat. Nobody is \
talking to you; you only listen.
Today (the day of the messages): {day}

Projects (id | name | status | priority | deadline | open milestones (id: title) | open risks):
{projects}

New messages (HH:MM name: text):
{messages}

Return JSON {{"updates": [{{"project_id", "kind", "value", "milestone_id", "quote", "speaker"}}]}} \
listing ONLY changes to the projects above that the messages clearly state as decided or done. \
Proposals, questions, wishes, jokes and plans are NOT changes. Kinds:
- "status": value is one of draft planned in_progress on_hold completed cancelled (delivered / \
finished = completed, paused / waiting = on_hold, started = in_progress, dropped = cancelled).
- "deadline": value is the new deadline as YYYY-MM-DD, resolved relative to {day}.
- "priority": value is high, medium or low.
- "milestone_done": a milestone of that project is finished; milestone_id is its id from the list.
- "risk": a newly raised risk or blocker; value is a short title.
- "note": a short progress line worth keeping (one per project at most).
"quote" is the exact message text (or a part of it) that says it; "speaker" is who said it. Never \
change an owner. Use only ids from the list. When unsure, leave it out. At most {limit} updates; \
an empty list is the usual answer."""


STRINGS = {
    "pl_noted": {
        "th": "📌 บันทึกแล้ว: {items} (ถ้าไม่ใช่ บอกสลิดได้)",
        "en": "📌 Noted: {items} (if that's wrong, tell Salid)",
    },
    "pl_status": {"th": "{name} → {value}", "en": "{name} → {value}"},
    "pl_deadline": {"th": "{name} กำหนดส่งเป็น {value}", "en": "{name} deadline set to {value}"},
    "pl_priority": {
        "th": "{name} ความสำคัญ → {value}",
        "en": "{name} priority → {value}",
    },
    "pl_milestone": {
        "th": "{name}: “{title}” เสร็จแล้ว",
        "en": "{name}: “{title}” is done",
    },
    "pl_risk": {"th": "{name}: ความเสี่ยงใหม่ “{title}”", "en": "{name}: new risk “{title}”"},
    "pl_progress": {"th": "{name}: จดความคืบหน้า", "en": "{name}: progress noted"},
    "pl_st_draft": {"th": "ฉบับร่าง", "en": "draft"},
    "pl_st_planned": {"th": "วางแผน", "en": "planned"},
    "pl_st_in_progress": {"th": "กำลังดำเนินการ", "en": "in progress"},
    "pl_st_on_hold": {"th": "พักไว้", "en": "on hold"},
    "pl_st_completed": {"th": "เสร็จสิ้น", "en": "completed"},
    "pl_st_cancelled": {"th": "ยกเลิก", "en": "cancelled"},
    "pl_pr_high": {"th": "สูง", "en": "high"},
    "pl_pr_medium": {"th": "กลาง", "en": "medium"},
    "pl_pr_low": {"th": "ต่ำ", "en": "low"},
}
CATALOG.update(STRINGS)  # one import registers the strings


def _clean(text: str, limit: int) -> str:
    return " ".join((text or "").split())[:limit]


def _norm(text: str) -> str:
    return " ".join((text or "").casefold().split())


def human_date(iso: str, lang: str, today: date) -> str:
    """ "30 พ.ย." / "30 Nov" (with the year when it is not this year)."""
    try:
        d = date.fromisoformat(iso)
    except ValueError:
        return iso
    months = _TH_MONTHS if lang == "th" else _EN_MONTHS
    out = f"{d.day} {months[d.month - 1]}"
    return out if d.year == today.year else f"{out} {d.year}"


class ProjectListener:
    def __init__(
        self,
        project_repo: ProjectRepo,
        task_repo,
        settings,
        llm,
        *,
        people_of: Callable[[int], dict[int, str]] = lambda gid: {},
        note: Callable[[int, int, str], None] | None = None,
        lang_of: Callable[[int, int], str] = lambda gid, cid: "auto",  # th / en / auto
        enabled: Callable[[int], bool] = lambda gid: True,
        notes_on: Callable[[int], bool] = lambda gid: True,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.repo, self.tasks, self.s, self.llm = project_repo, task_repo, settings, llm
        self.people_of, self.note, self.lang_of = people_of, note, lang_of
        self.enabled, self.notes_on, self._clock = enabled, notes_on, clock
        self._last_note: dict[tuple[int, int], float] = {}

    # ------------------------------------------------------------------ hook
    def on_summary(
        self,
        guild_id: int,
        channel_id: int,
        day: str,
        summary: dict,
        messages: list[MemoryMessage],
    ) -> list[str]:
        """Never raises. Returns the lines applied (for the log and tests)."""
        try:
            if not self.enabled(guild_id):
                return []
            return self.listen(guild_id, channel_id, day, messages)
        except Exception:
            logger.warning("project listening failed", exc_info=True)
            return []

    def listen(
        self, guild_id: int, channel_id: int, day: str, messages: list[MemoryMessage]
    ) -> list[str]:
        lines = [m for m in messages if m.kind != "salid" and m.text.strip()]
        lines = lines[-MAX_LINES_FOR_CHECK:]
        if not lines:
            return []
        projects = [p for p in self.repo.list(guild_id) if p.status != "cancelled"]
        if not projects:
            return []
        projects.sort(key=lambda p: p.updated_at, reverse=True)
        projects = projects[:MAX_PROJECTS_IN_PROMPT]
        heard = self._ask(day, projects, lines)
        if not heard:
            return []
        return self._apply(guild_id, channel_id, day, projects, lines, heard)

    # ------------------------------------------------------------------ model
    @staticmethod
    def _listing(projects: list[Project]) -> str:
        rows = []
        for p in projects:
            ms = "; ".join(f"{m.id}: {m.title}" for m in p.milestones if m.status != "done")
            rk = "; ".join(r.title for r in p.risks if r.status == "open")
            rows.append(
                f"{p.id} | {p.name} | {p.status} | {p.priority} | {p.deadline or '-'} | "
                f"{ms or '-'} | {rk or '-'}"
            )
        return "\n".join(rows)

    def _ask(self, day: str, projects: list[Project], lines: list[MemoryMessage]) -> list[_Heard]:
        prompt = LISTEN_PROMPT.format(
            day=day,
            projects=self._listing(projects),
            messages=format_lines(lines, self.s.zone),
            limit=MAX_UPDATES_PER_BATCH,
        )
        try:
            resp = generate(
                self.llm,
                self.s.gate_model,
                prompt,
                types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=1200,
                    response_mime_type="application/json",
                    response_schema=_HeardList,
                ),
            )
            return _HeardList.model_validate_json(resp.text).updates[:MAX_UPDATES_PER_BATCH]
        except Exception:
            logger.warning("project listening call failed", exc_info=True)
            return []

    # ------------------------------------------------------------------ apply
    def _by(self, guild_id: int, speaker: str, line: MemoryMessage) -> Person:
        uid = match_person(speaker, self.people_of(guild_id))
        if uid:
            return Person(
                discord_user_id=str(uid),
                name=self.people_of(guild_id).get(uid) or speaker or line.author_name,
            )
        if line.author_id > 0:
            return Person(discord_user_id=str(line.author_id), name=line.author_name or speaker)
        return Person(name=_clean(speaker or line.author_name, 120) or "chat")

    @staticmethod
    def _line_of(quote: str, lines: list[MemoryMessage]) -> MemoryMessage | None:
        q = _norm(quote)
        if len(q) < 3:
            return None
        for m in reversed(lines):
            if q in _norm(m.text):
                return m
        return None

    def _apply(
        self,
        guild_id: int,
        channel_id: int,
        day: str,
        projects: list[Project],
        lines: list[MemoryMessage],
        heard: list[_Heard],
    ) -> list[str]:
        now = self.s.now()
        try:
            today = date.fromisoformat(day)
        except ValueError:
            today = now.date()
        lang = resolve_lang(  # the company language; an unchosen server follows the words
            self.lang_of(guild_id, channel_id), " ".join(m.text for m in lines[-5:])
        )
        known = {p.id: p for p in projects}
        # the last statement wins for the plain fields; keep the order of first appearance
        last: dict[tuple[str, str], int] = {}
        for i, h in enumerate(heard):
            if h.kind in LAST_WINS:
                last[(h.project_id, h.kind)] = i
        work: dict[str, Project] = {}
        noted: set[str] = set()
        said: list[tuple[str, str]] = []  # (project id, line for the room)
        for i, h in enumerate(heard):
            p0 = known.get(h.project_id)
            if p0 is None or h.kind not in KINDS:
                continue
            if h.kind in LAST_WINS and last[(h.project_id, h.kind)] != i:
                continue
            if h.kind != "note" and p0.status in DONE_STATUSES:
                continue
            quote = _clean(h.quote, QUOTE_CHARS)
            line = self._line_of(quote, lines)
            if line is None:  # not said in these messages: never record it
                continue
            p = work.get(p0.id) or p0.model_copy(deep=True)
            by = self._by(guild_id, h.speaker, line)
            reason = f'heard in chat: "{quote}"'
            done = self._one(p, h, by, quote, reason, line, channel_id, noted, now)
            if done is None:
                continue
            work[p.id] = done[0]
            said.append((p.id, self._say(done[1], p.name, done[2], lang, today)))
        if not work:
            return []
        saved: set[str] = set()
        for pid, p in work.items():
            try:
                self.repo.save(guild_id, p)
                saved.add(pid)
            except Exception:
                logger.warning("saving heard project update failed", exc_info=True)
        out = [line for pid, line in said if pid in saved]
        if not out:
            return []
        logger.info("projects: %d heard in guild=%s channel=%s", len(out), guild_id, channel_id)
        self._tell(guild_id, channel_id, out, lang)
        return out

    def _one(
        self,
        p: Project,
        h: _Heard,
        by: Person,
        quote: str,
        reason: str,
        line: MemoryMessage,
        channel_id: int,
        noted: set[str],
        now,
    ) -> tuple[Project, str, str] | None:
        """Apply one heard update to the working copy `p`. (project, kind, shown value) or None."""
        value = _clean(h.value, 300)
        kind = h.kind
        if kind in LAST_WINS:
            if kind == "status" and (value not in STATUSES or value == p.status):
                return None
            if kind == "priority" and (value not in PRIORITIES or value == p.priority):
                return None
            if kind == "deadline" and value == p.deadline:
                return None
            try:
                new, _ = apply_updates(p, {kind: value}, by, reason=reason, impact="", now=now)
            except Exception:  # a bad date or enum value (pydantic ValidationError)
                return None
            return new, kind, value
        if kind == "milestone_done":
            ms = next((m for m in p.milestones if m.id == h.milestone_id), None)
            if ms is None or ms.status == "done":
                return None
            ms.status = "done"
            record_change(p, "milestone", ms.title, "done", by, reason=reason, now=now)
            p.updated_at = stamp(now)
            return p, kind, ms.title
        if kind == "risk":
            title = _clean(value, 200)
            if not title or any(r.title.casefold() == title.casefold() for r in p.risks):
                return None
            p.risks.append(
                Risk(
                    id=next_item_id(p.risks, "r"),
                    title=title,
                    severity="medium",
                    source="team",
                    note=quote,
                    created_at=stamp(now),
                )
            )
            p.updated_at = stamp(now)
            return p, kind, title
        # note: at most one per project per batch
        text = _clean(value, 300)
        if not text or p.id in noted:
            return None
        noted.add(p.id)
        p.updates.append(
            Update(
                at=stamp(now),
                by=by,
                text=text,
                quote=quote,
                channel_id=0 if is_dm_scope(channel_id) else channel_id,
                jump_url="" if is_dm_scope(channel_id) else line.jump_url,
            )
        )
        del p.updates[:-MAX_UPDATES]
        p.updated_at = stamp(now)
        return p, kind, text

    # ------------------------------------------------------------------ the room
    @staticmethod
    def _say(kind: str, name: str, value: str, lang: str, today: date) -> str:
        if kind == "status":
            return t("pl_status", lang, name=name, value=t(f"pl_st_{value}", lang))
        if kind == "deadline":
            return t("pl_deadline", lang, name=name, value=human_date(value, lang, today))
        if kind == "priority":
            return t("pl_priority", lang, name=name, value=t(f"pl_pr_{value}", lang))
        if kind == "milestone_done":
            return t("pl_milestone", lang, name=name, title=value)
        if kind == "risk":
            return t("pl_risk", lang, name=name, title=value)
        return t("pl_progress", lang, name=name)

    def _tell(self, guild_id: int, channel_id: int, said: list[str], lang: str) -> None:
        """One note per batch; skipped (the changes stay) when off, DM scope or too soon."""
        if self.note is None or is_dm_scope(channel_id) or not said:
            return
        try:
            if not self.notes_on(guild_id):
                return
            key = (guild_id, channel_id)
            last = self._last_note.get(key)
            if last is not None and self._clock() - last < NOTE_GAP_S:
                return
            text = t("pl_noted", lang, items=" · ".join(said[:MAX_ITEMS_IN_NOTE]))
            self._last_note[key] = self._clock()
            self.note(guild_id, channel_id, text)
        except Exception:
            logger.warning("project note failed", exc_info=True)
