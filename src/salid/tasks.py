"""Work tracking: a task list built from what the team says, with no one reporting to Salid.

`TaskTracker.on_summary` is the memory scheduler's hook. After each channel-day summary update it
(1) turns the summary's `action_items` into tasks (assignee resolved from names or mentions when
possible, duplicates merged by text similarity) and (2) makes one batched cheap-model call that
marks open tasks done when the new messages say so. Daily follow-ups and the `list_tasks` /
`update_task` agent tools use `TaskRepo`. Tasks live in the `salid_tasks` collection
(see `salid.coo_store`). Everything here is blocking: run it in a worker thread.
"""

from __future__ import annotations

import contextlib
import difflib
import logging
import re
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

from google.genai import types
from pydantic import BaseModel, Field, PrivateAttr

from .coo_store import TASKS, DocStore
from .llm import generate
from .memory_access import Scope, is_dm_scope
from .memory_store import MemoryMessage
from .memory_summary import format_lines
from .naming import mentions_name

logger = logging.getLogger(__name__)

_BOT_WHO_RE = re.compile(r"^\W*(?:บอท|ผู้ช่วย|bot|the\s+bot|assistant|ai|the\s+assistant)\W*$", re.I)


def is_assistant_who(who: str) -> bool:
    """The doer named in an action item is Salid itself (or "the bot"): not a team task."""
    return bool(who) and (mentions_name(who) or bool(_BOT_WHO_RE.match(who)))


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
    project_id: str = ""  # PM dashboard: the project this task belongs to (salid.projects)
    assignee_email: str = ""  # web members are assigned by email (assignee_id stays a Discord id)
    milestone_id: str = ""
    # task handshake (salid.task_handshake): the assignee confirms, the giver is told
    assigner_id: int = 0  # who gave / asked for the task (0 = unknown)
    assigner_name: str = ""
    confirm: str = ""  # "" | pending | accepted | declined | awaiting_approval | owner_asked
    confirm_at: str = ""  # when `confirm` last changed (the 24 h nudge counts from "pending")
    handshake_message: str = ""  # "channel_id:message_id:index" of the card, for editing it
    owner_message: str = ""  # same, for the "who takes this task?" card sent to the giver
    group_id: str = ""  # tasks made from one action item that named several people
    # approve mode: the giver signs off after the assignee accepts (confirm = awaiting_approval)
    approver_id: int = 0
    approval_message: str = ""  # "channel_id:message_id" of the giver's approval card
    approval_rounds: int = 0  # how many times it was sent back
    edit_from: str = ""  # JSON {title, due, note} before the assignee's edits (before -> after)
    send_back_note: str = ""
    # plan (Gantt) fields, see salid.plan_import / salid.progress
    start: str = ""  # planned start YYYY-MM-DD or ""
    step: int = 0  # position in the plan, 1-2-3 (0 = not a numbered step)
    depends_on: list[str] = Field(default_factory=list)  # ids of the tasks that come before
    pct_done: int = 0  # 0-100 (a done task counts as 100)
    nodue_asked_at: str = ""  # when the assignee was asked once about a missing deadline
    # not stored: a team / several people were named but nobody could be picked (giver is asked)
    _needs_owner: bool = PrivateAttr(default=False)


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


_WHO_SPLIT = re.compile(
    r"\s*กับ\s*|\s*และ\s*|\s*,\s*|\s+and\s+|\s*&\s*|\s*/\s*|\s+(?=@)|(?<=>)\s*(?=<@)", re.I
)
MAX_GROUP = 5  # a team larger than this is not split into one task per member


def split_who(who: str) -> list[str]:
    """The people named in an action item's `who` ("เตยกับบี", "A, B and C", "<@1> <@2>")."""
    return [p.strip() for p in _WHO_SPLIT.split(who or "") if p and p.strip()]


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", (text or "").casefold()) if len(w) >= 3]


def guess_assigner(
    messages, people: dict[int, str], assignee_id: int, assignee_name: str, title: str
) -> int:
    """Who asked for a task nobody named a giver for. With an assignee: the author of the newest
    message in the batch that names them (name or mention) AND shares a word with the title.
    Without one (0): the author of the newest message that shares at least 2 title words, or one
    word of 4+ characters; else 0."""
    name = (
        (assignee_name or people.get(assignee_id) or "").casefold().strip() if assignee_id else ""
    )
    words = list(dict.fromkeys(_words(title)))
    for m in reversed(list(messages)):  # oldest first in, so newest first here
        if getattr(m, "kind", "chat") == "salid" or getattr(m, "author_id", 0) not in people:
            continue
        text = (getattr(m, "text", "") or "").casefold()
        if not assignee_id:
            hits = [w for w in words if w in text]
            if len(hits) >= 2 or any(len(w) >= 4 for w in hits):
                return m.author_id
            continue
        named = f"<@{assignee_id}>" in text or f"<@!{assignee_id}>" in text
        named = named or (len(name) >= 2 and name in text)
        if named and any(w in text for w in words):
            return m.author_id
    return 0


def project_in_text(projects: list, text: str) -> str:
    """Id of the project an action item names ("โปรเจกต์ X" / "project X"), else ""."""
    if not projects:
        return ""
    from .projects import project_in_text as find

    p = find(projects, text)
    return p.id if p else ""


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


def _local_date(iso: str, zone) -> date | None:
    try:
        at = datetime.fromisoformat(iso)
    except ValueError:
        return None
    return (at if at.tzinfo else at.replace(tzinfo=UTC)).astimezone(zone).date()


def working_days_between(start: date, end: date) -> int:
    """Mon-Fri days after `start` up to and including `end`."""
    return sum(
        1 for i in range(1, max((end - start).days, 0) + 1) if (start + timedelta(i)).weekday() < 5
    )


def no_deadline_due(tasks: list[Task], today: date, zone, working_days: int = 3) -> list[Task]:
    """Open tasks with an assignee and no due date, never asked about it, that were created (or
    accepted, when later) at least `working_days` working days (Mon-Fri in `zone`) ago."""
    out = []
    for t in tasks:
        if t.status != "open" or not t.assignee_id or t.due or t.nodue_asked_at:
            continue
        if t.confirm == "declined":
            continue
        start = max(
            (d for d in (_local_date(t.created_at, zone),
                         _local_date(t.confirm_at, zone) if t.confirm == "accepted" else None) if d),
            default=None,
        )  # fmt: skip
        if start and working_days_between(start, today) >= working_days:
            out.append(t)
    return sorted(out, key=lambda t: (t.created_at, t.id))


# ---------------------------------------------------------------- repository
class TaskRepo:
    def __init__(self, docs: DocStore):
        self.docs = docs
        # called with the task after it became done (company knowledge: "delivered X")
        self.on_done: Callable[[Task], None] | None = None

    def get(self, guild_id: int, task_id: str) -> Task | None:
        d = self.docs.get(TASKS, guild_id, task_id)
        return Task(**d) if d else None

    def list(
        self,
        guild_id: int,
        *,
        status: str | None = "open",
        assignee_id: int | None = None,
        project_id: str | None = None,
    ) -> list[Task]:
        rows = self.docs.find(TASKS, guild_id)
        out = [
            Task(**d)
            for d in rows
            if (status in (None, "all") or d.get("status") == status)
            and (assignee_id is None or d.get("assignee_id") == assignee_id)
            and (project_id is None or d.get("project_id", "") == project_id)
        ]
        return sorted(out, key=lambda t: (t.due or "9999-99-99", t.created_at, t.id))

    def for_person(
        self,
        guild_id: int,
        email: str | None,
        discord_id: int | str | None,
        *,
        status: str | None = "open",
    ) -> list[Task]:
        """Tasks assigned to a person known by web email and / or Discord id."""
        e = (email or "").strip().lower()
        d = int(discord_id) if str(discord_id or "").isdigit() else 0
        return [
            t
            for t in self.list(guild_id, status=status)
            if (e and t.assignee_email.lower() == e) or (d and t.assignee_id == d)
        ]

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
                if task.project_id and not old.project_id:
                    old.project_id, changed = task.project_id, True
                if task.assignee_id and not old.assignee_id and old.confirm != "declined":
                    old.assignee_id, old.assignee_name, changed = (
                        task.assignee_id,
                        task.assignee_name,
                        True,
                    )
                    if old.confirm == "owner_asked":
                        old.confirm = ""  # a summary named the owner: the question is moot
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
        saved = self.save(task)
        if status == "done" and self.on_done is not None:
            with contextlib.suppress(Exception):  # learning never blocks a status change
                self.on_done(saved)
        return saved


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
        role_members_of: Callable[[int, str], list[int] | None] = lambda gid, name: None,
    ):
        self.repo, self.s, self.llm = repo, settings, llm
        self.people_of, self.kind_of, self.enabled = people_of, kind_of, enabled
        # member ids of the guild role / team a text names; None = no such role
        self.role_members_of = role_members_of

    def on_summary(
        self,
        guild_id: int,
        channel_id: int,
        day: str,
        summary: dict,
        messages: list[MemoryMessage],
    ) -> list[Task]:
        """Tasks and done-marks for one summary update. Returns the newly created tasks."""
        if not self.enabled(guild_id):
            return []
        created = self.ingest(guild_id, channel_id, day, summary, messages)
        if messages:
            self.mark_done(guild_id, messages)
        return created

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
        project_ids = None  # loaded once, only if there is an item
        for a in items[:MAX_TASKS_PER_SUMMARY]:
            who = (a.get("who") or "").strip()
            if is_assistant_who(who):
                logger.debug("tasks: skipped an action item for the assistant itself")
                continue
            by = (a.get("by") or "").strip()
            title = a["what"].strip()[:300]
            due = parse_due(a.get("due"), today)
            uids, overflow = self._assignees(guild_id, who, people)
            if not uids and not who and is_dm_scope(channel_id):
                uids = [-channel_id]  # a note someone gave about their own work
            group = uuid.uuid4().hex[:8] if len(uids) > 1 else ""
            batch = []
            for uid in uids or [0]:
                giver = match_person(by, people)
                if not by and uid >= 0:
                    giver = guess_assigner(messages, people, uid, people.get(uid) or who, title)
                task = Task(
                    guild_id=guild_id,
                    assignee_id=uid,
                    assignee_name=people.get(uid) or who,
                    assigner_id=giver,
                    assigner_name=people.get(giver) or by,
                    title=title,
                    due=due,
                    source_link="" if is_dm_scope(channel_id) else link,
                    source_channel_id=channel_id,
                    source_day=day,
                    created_from=kind,
                    group_id=group,
                )
                task._needs_owner = overflow
                batch.append(task)
            if project_ids is None:
                project_ids = self._projects(guild_id)
            pid = project_in_text(project_ids, title) or ""
            for task in batch:
                task.project_id = pid
                t, new = self.repo.add(task)
                if new:
                    t._needs_owner = task._needs_owner
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

    def _assignees(self, guild_id: int, who: str, people: dict[int, str]) -> tuple[list[int], bool]:
        """(member ids, overflow): everyone `who` names. Several people -> all that resolve (if
        2+); one role / team of up to MAX_GROUP members -> its members; a bigger team, or several
        names of which fewer than 2 resolve -> ([], overflow=True when several/team were named)."""
        parts = split_who(who)
        if len(parts) >= 2:
            ids = list(dict.fromkeys(i for i in (match_person(p, people) for p in parts) if i))
            return (ids, False) if len(ids) >= 1 else ([], True)
        uid = match_person(who, people)
        if uid:
            return [uid], False
        if not who:
            return [], False
        try:
            members = self.role_members_of(guild_id, who)
        except Exception:
            logger.warning("role lookup for tasks failed", exc_info=True)
            members = None
        if members is None:
            return [], False
        members = [m for m in dict.fromkeys(members) if m in people]
        if 1 <= len(members) <= MAX_GROUP:
            return members, False
        return [], True

    def _projects(self, guild_id: int) -> list:
        """The guild's running projects (for "โปรเจกต์ X" in an action item); [] on any trouble."""
        try:
            from .projects import ProjectRepo

            return ProjectRepo(self.repo.docs).list(guild_id)
        except Exception:
            logger.warning("project lookup for tasks failed", exc_info=True)
            return []

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
