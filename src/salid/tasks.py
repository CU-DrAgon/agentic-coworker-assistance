"""Work tracking: a task list built from what the team says, with no one reporting to Salid.

`TaskTracker.on_summary` is the memory scheduler's hook. After each channel-day summary update it
(1) turns the summary's `action_items` into tasks (assignee resolved from names or mentions when
possible, duplicates merged by text similarity) and (2) makes one batched cheap-model call that
marks open tasks done when the new messages say so. Daily follow-ups and the `list_tasks` /
`update_task` agent tools use `TaskRepo`. Tasks live in the `salid_tasks` collection
(see `salid.coo_store`). Everything here is blocking: run it in a worker thread.
"""

from __future__ import annotations

import difflib
import logging
import re
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

from google.genai import types
from pydantic import BaseModel, Field

from .coo_store import TASKS, DocStore
from .llm import generate
from .memory_access import Scope, is_dm_scope
from .memory_store import MemoryMessage
from .memory_summary import format_lines

logger = logging.getLogger(__name__)

STATUSES = ("open", "done")
SIMILARITY = 0.78  # titles at least this similar (same assignee) are one task
MAX_OPEN_FOR_CHECK = 40
MAX_LINES_FOR_CHECK = 80
MAX_TASKS_PER_SUMMARY = 20
_MENTION = re.compile(r"<@!?(\d{1,25})>")
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_WEEKDAYS = {
    "จันทร์": 0, "monday": 0, "mon": 0, "อังคาร": 1, "tuesday": 1, "tue": 1,
    "พุธ": 2, "wednesday": 2, "wed": 2, "พฤหัส": 3, "thursday": 3, "thu": 3,
    "ศุกร์": 4, "friday": 4, "fri": 4, "เสาร์": 5, "saturday": 5, "sat": 5,
    "อาทิตย์": 6, "sunday": 6, "sun": 6,
}  # fmt: skip


class Task(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    guild_id: int
    assignee_id: int = 0  # 0 = not resolved to a member
    assignee_name: str = ""
    title: str
    due: str = ""  # YYYY-MM-DD or ""
    source_link: str = ""
    source_channel_id: int = 0  # access check: who may see where this came from
    source_day: str = ""
    status: str = "open"
    created_from: str = "summary"  # summary | meeting | dm | manual
    note: str = ""
    created_at: str = ""
    updated_at: str = ""
    done_at: str = ""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------- pure helpers
def parse_due(text: str | None, today: date) -> str:
    """A due date from "2026-10-16", "วันนี้", "พรุ่งนี้", "ศุกร์", "tomorrow", "friday"; else ""."""
    s = (text or "").strip().casefold()
    if not s:
        return ""
    m = _ISO.search(s)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
        except ValueError:
            return ""
    if "มะรืน" in s or "day after tomorrow" in s:
        return (today + timedelta(days=2)).isoformat()
    if "พรุ่งนี้" in s or "tomorrow" in s:
        return (today + timedelta(days=1)).isoformat()
    if "วันนี้" in s or "today" in s or "ภายในวัน" in s or "eod" in s:
        return today.isoformat()
    for name, wd in _WEEKDAYS.items():
        if name in s:
            ahead = (wd - today.weekday()) % 7 or 7
            return (today + timedelta(days=ahead)).isoformat()
    return ""


def match_person(name: str, people: dict[int, str]) -> int:
    """Member id for a name or mention among `people` (id -> display name); 0 when unknown or
    ambiguous. Exact (case-insensitive) first, then a unique substring match either way."""
    s = (name or "").strip()
    if not s:
        return 0
    m = _MENTION.search(s)
    if m:
        uid = int(m[1])
        return uid if uid in people else 0
    if s.isdigit() and int(s) in people:
        return int(s)
    low = s.casefold().lstrip("@")
    exact = [i for i, n in people.items() if n.casefold() == low]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return 0
    loose = [
        i
        for i, n in people.items()
        if len(low) >= 2 and (low in n.casefold() or (len(n) >= 2 and n.casefold() in low))
    ]
    return loose[0] if len(loose) == 1 else 0


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.casefold())


def similar(a: Task | str, b: Task | str) -> bool:
    ta = a if isinstance(a, str) else a.title
    tb = b if isinstance(b, str) else b.title
    x, y = _norm(ta), _norm(tb)
    if not x or not y:
        return False
    return x == y or difflib.SequenceMatcher(None, x, y).ratio() >= SIMILARITY


def task_visible(task: Task, scope: Scope) -> bool:
    """Who may see a task: whoever can read where it came from, the assignee, and anyone who
    outranks the assignee (for a DM note: the DM's author) under the scope's audience rule."""
    if is_dm_scope(task.source_channel_id):
        return -task.source_channel_id in scope.authors
    return task.source_channel_id in scope.channels or (
        task.assignee_id != 0 and task.assignee_id in scope.authors
    )


def is_overdue(task: Task, today: date) -> bool:
    return task.status == "open" and bool(task.due) and task.due < today.isoformat()


def due_for_followup(tasks: list[Task], today: date) -> dict[int, list[Task]]:
    """Open tasks with a known assignee that are due today or overdue, per member, oldest due
    first."""
    out: dict[int, list[Task]] = {}
    for t in tasks:
        if t.status == "open" and t.assignee_id and t.due and t.due <= today.isoformat():
            out.setdefault(t.assignee_id, []).append(t)
    for v in out.values():
        v.sort(key=lambda t: (t.due, t.id))
    return out


# ---------------------------------------------------------------- repository
class TaskRepo:
    def __init__(self, docs: DocStore):
        self.docs = docs

    def get(self, guild_id: int, task_id: str) -> Task | None:
        d = self.docs.get(TASKS, guild_id, task_id)
        return Task(**d) if d else None

    def list(
        self,
        guild_id: int,
        *,
        status: str | None = "open",
        assignee_id: int | None = None,
    ) -> list[Task]:
        rows = self.docs.find(TASKS, guild_id)
        out = [
            Task(**d)
            for d in rows
            if (status in (None, "all") or d.get("status") == status)
            and (assignee_id is None or d.get("assignee_id") == assignee_id)
        ]
        return sorted(out, key=lambda t: (t.due or "9999-99-99", t.created_at, t.id))

    def save(self, task: Task) -> Task:
        task.updated_at = _now_iso()
        self.docs.put(TASKS, task.guild_id, task.id, task.model_dump())
        return task

    def add(self, task: Task) -> tuple[Task, bool]:
        """Store a task unless a similar one exists (any status: a finished item that a living
        summary still lists must not come back). Returns (task, created)."""
        for old in self.list(task.guild_id, status=None):
            if (
                old.assignee_id == task.assignee_id
                or not old.assignee_id
                or not task.assignee_id
                or old.assignee_name.casefold() == task.assignee_name.casefold()
            ) and similar(old, task):
                changed = False
                if task.due and not old.due and old.status == "open":
                    old.due, changed = task.due, True
                if task.assignee_id and not old.assignee_id:
                    old.assignee_id, old.assignee_name, changed = (
                        task.assignee_id,
                        task.assignee_name,
                        True,
                    )
                if changed:
                    self.save(old)
                return old, False
        task.created_at = task.created_at or _now_iso()
        return self.save(task), True

    def set_status(self, guild_id: int, task_id: str, status: str, note: str = "") -> Task | None:
        task = self.get(guild_id, task_id)
        if task is None or status not in STATUSES:
            return None
        task.status = status
        task.done_at = _now_iso() if status == "done" else ""
        if note:
            task.note = note[:500]
        return self.save(task)


# ---------------------------------------------------------------- the summary hook
class _Done(BaseModel):
    id: str
    note: str = ""


class _DoneList(BaseModel):
    done: list[_Done] = Field(default_factory=list)


DONE_PROMPT = """You check a team's task list against new chat messages.
Open tasks (id | assignee | task):
{tasks}

New messages (HH:MM name: text):
{messages}

Return JSON {{"done": [{{"id": ..., "note": short reason}}]}} listing ONLY open tasks that the \
messages clearly say are finished (delivered, sent, paid, fixed, done, completed). If a task is only \
discussed, postponed or in progress, leave it out. When unsure, leave it out."""


class TaskTracker:
    def __init__(
        self,
        repo: TaskRepo,
        settings,
        llm,
        *,
        people_of: Callable[[int], dict[int, str]] = lambda gid: {},
        kind_of: Callable[[int, int], str] = lambda gid, cid: "summary",
        enabled: Callable[[int], bool] = lambda gid: True,
    ):
        self.repo, self.s, self.llm = repo, settings, llm
        self.people_of, self.kind_of, self.enabled = people_of, kind_of, enabled

    def on_summary(
        self,
        guild_id: int,
        channel_id: int,
        day: str,
        summary: dict,
        messages: list[MemoryMessage],
    ) -> None:
        if not self.enabled(guild_id):
            return
        self.ingest(guild_id, channel_id, day, summary, messages)
        if messages:
            self.mark_done(guild_id, messages)

    def ingest(
        self, guild_id: int, channel_id: int, day: str, summary: dict, messages
    ) -> list[Task]:
        """Action items of a summary -> tasks. Returns the newly created ones."""
        items = [a for a in (summary.get("action_items") or []) if (a.get("what") or "").strip()]
        if not items:
            return []
        people = self.people_of(guild_id)
        try:
            today = date.fromisoformat(day)
        except ValueError:
            today = self.s.now().date()
        link = next((m.jump_url for m in messages if m.jump_url), "")
        kind = "dm" if is_dm_scope(channel_id) else self.kind_of(guild_id, channel_id)
        created = []
        for a in items[:MAX_TASKS_PER_SUMMARY]:
            who = (a.get("who") or "").strip()
            uid = match_person(who, people)
            if not uid and is_dm_scope(channel_id) and not who:
                uid = -channel_id  # a note someone gave about their own work
            task = Task(
                guild_id=guild_id,
                assignee_id=uid,
                assignee_name=people.get(uid) or who,
                title=a["what"].strip()[:300],
                due=parse_due(a.get("due"), today),
                source_link="" if is_dm_scope(channel_id) else link,
                source_channel_id=channel_id,
                source_day=day,
                created_from=kind,
            )
            t, new = self.repo.add(task)
            if new:
                created.append(t)
        if created:
            logger.info(
                "tasks: %d new from guild=%s channel=%s day=%s",
                len(created),
                guild_id,
                channel_id,
                day,
            )
        return created

    def mark_done(self, guild_id: int, messages: list[MemoryMessage]) -> list[str]:
        """One cheap-model call for the whole batch of new messages. Returns the ids marked."""
        open_tasks = self.repo.list(guild_id, status="open")[:MAX_OPEN_FOR_CHECK]
        lines = [m for m in messages if m.kind != "salid" and m.text.strip()]
        if not open_tasks or not lines:
            return []
        listing = "\n".join(f"{t.id} | {t.assignee_name or '?'} | {t.title}" for t in open_tasks)
        prompt = DONE_PROMPT.format(
            tasks=listing,
            messages=format_lines(lines[-MAX_LINES_FOR_CHECK:], self.s.zone),
        )
        try:
            resp = generate(
                self.llm,
                self.s.gate_model,
                prompt,
                types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=600,
                    response_mime_type="application/json",
                    response_schema=_DoneList,
                ),
            )
            done = _DoneList.model_validate_json(resp.text).done
        except Exception:
            logger.warning("task completion check failed", exc_info=True)
            return []
        known = {t.id for t in open_tasks}
        marked = []
        for d in done:
            if d.id in known and self.repo.set_status(guild_id, d.id, "done", d.note):
                marked.append(d.id)
        if marked:
            logger.info("tasks: %d marked done in guild=%s", len(marked), guild_id)
        return marked
