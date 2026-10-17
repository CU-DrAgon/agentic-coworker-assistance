"""Agent tools of the coordinator role: tasks, disclosure requests and "what did I do today".

Bound in code to one guild and one asker, like the memory tools: the ids are never part of a tool
schema. `build_coo_tools` returns plain functions that ADK wraps; errors come back as data.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime

from . import task_handshake as H
from .hints import hint, legend, status_marker
from .memory_access import Scope
from .tasks import (
    STATUSES,
    Task,
    TaskRepo,
    is_overdue,
    match_person,
    parse_due,
    split_who,
    task_visible,
)

MAX_TASKS = 40
ME_WORDS = ("me", "mine", "myself", "i", "my own", "ฉัน", "ผม", "เรา", "หนู", "ของฉัน", "ของผม",
            "ตัวเอง", "ฉันเอง", "ผมเอง")  # fmt: skip


class CooTools:
    def __init__(
        self,
        *,
        guild_id: int,
        asker_id: int,
        is_exec: bool,
        scope: Scope | None,
        tasks: TaskRepo,
        people: dict[int, str],
        today: Callable[[], date],
        memory=None,  # MemoryTools: `hidden` holds what was withheld in this request
        request_disclosure: Callable[[str], dict] | None = None,
        my_day: Callable[[date], dict] | None = None,
        reassigned: Callable[[Task], None] | None = None,
        created: Callable[[list[Task]], None] | None = None,
        owner_answered: Callable[[list[Task], list[Task]], None] | None = None,
        find_project: Callable[[str], tuple[str, str]] | None = None,
        channel_id: int = 0,
        lang: str = "en",
    ):
        self.lang = lang
        self.guild_id, self.asker_id, self.is_exec, self.scope = guild_id, asker_id, is_exec, scope
        self.tasks, self.people, self.today, self.memory = tasks, people, today, memory
        self._request, self._my_day = request_disclosure, my_day
        self._reassigned = reassigned
        self._owner_answered = owner_answered
        self._created, self._find_project, self.channel_id = created, find_project, channel_id

    def _view(self, t: Task) -> dict:
        return {
            "id": t.id,
            "assignee": t.assignee_name or None,
            "title": t.title,
            "due": t.due or None,
            "status": t.status,
            "overdue": is_overdue(t, self.today()),
            "marker": status_marker(t.confirm, is_overdue(t, self.today()), self.lang) or None,
            "link": t.source_link or None,
            "note": t.note or None,
            "confirm": t.confirm or None,
            "awaiting_confirmation": t.confirm == "pending",
            "owner_pending": t.confirm == "owner_asked",
            "assigned_by": t.assigner_name or None,
            "group": t.group_id or None,
        }

    def _can_see(self, t: Task) -> bool:
        if t.assignee_id == self.asker_id:
            return True
        return self.scope is not None and task_visible(t, self.scope)

    def list_tasks(self, person: str | None = None, status: str = "open") -> dict:
        if status not in (*STATUSES, "all"):
            return {"error": "status must be open, done or all"}
        who = None
        if person and person.strip().casefold() not in ("me", "ผม", "ฉัน", "ตัวเอง"):
            who = match_person(person, self.people)
            if not who:
                return {
                    "error": f"cannot tell who {person!r} is; use the member's name or @mention"
                }
        elif person:
            who = self.asker_id
        rows = [
            t
            for t in self.tasks.list(self.guild_id, status=status, assignee_id=who)
            if self._can_see(t)
        ]
        out: dict = {"count": len(rows), "tasks": [self._view(t) for t in rows[:MAX_TASKS]]}
        if len(rows) > MAX_TASKS:
            out["note"] = f"showing the first {MAX_TASKS}"
        if not rows:
            out["note"] = "no matching tasks that this person may see"
        if any(v["marker"] for v in out["tasks"]):
            out["legend"] = legend(self.lang)  # show once, as the last line, when a marker is used
        return out

    def add_task(
        self, title: str, assignee: str = "", due: str = "", project: str = "", note: str = ""
    ) -> dict:
        """A task the asker hands out now (tag or reply): one task per named person, the asker as
        the giver. Each assignee gets a card to confirm (taking it yourself needs none)."""
        title = " ".join((title or "").split())[:300]
        if not title:
            return {"error": "give the task a title"}
        when = ""
        if (due or "").strip():
            when = parse_due(due, self.today())
            if not when:
                return {"error": f"cannot read the due date {due!r}; use YYYY-MM-DD or a weekday"}
        pid = ""
        if (project or "").strip():
            if self._find_project is None:
                return {"error": "projects are not available here"}
            pid, err = self._find_project(project)
            if err:
                return {"error": err}
        ids: list[int] = []
        unresolved: list[str] = []
        for part in split_who(assignee):
            uid = match_person(part, self.people)
            if uid and uid not in ids:
                ids.append(uid)
            elif not uid:
                unresolved.append(part)
        group = uuid.uuid4().hex[:8] if len(ids) > 1 else ""
        giver = self.people.get(self.asker_id, "")
        rows, fresh = [], []
        for uid in ids or [0]:
            task = Task(
                guild_id=self.guild_id,
                assignee_id=uid,
                assignee_name=self.people.get(uid) or (assignee or "").strip()[:120],
                assigner_id=self.asker_id,
                assigner_name=giver,
                title=title,
                due=when,
                note=(note or "")[:500],
                project_id=pid,
                source_channel_id=self.channel_id,
                source_day=self.today().isoformat(),
                created_from="manual",
                group_id=group,
                confirm="accepted" if uid == self.asker_id else "",
            )
            saved, new = self.tasks.add(task)
            rows.append(saved)
            if new:
                fresh.append(saved)
        if fresh and self._created is not None:
            self._created(fresh)
        views = [self._view(x) for x in rows]
        out: dict = {"task": views[0], "tasks": views, "unresolved": unresolved}
        if fresh:
            out["asked_to_confirm"] = [
                x.assignee_name for x in fresh if x.assignee_id and not x.confirm
            ]
        return out

    def update_task(
        self, task_id: str, status: str = "", note: str = "", assignee: str = "", due: str = ""
    ) -> dict:
        task = self.tasks.get(self.guild_id, (task_id or "").lstrip("#").strip())
        if task is None:
            return {"error": "no such task"}
        if not status and not (assignee or "").strip() and not (due or "").strip():
            return {"error": "give a status (done / open), an assignee or a due date"}
        if (due or "").strip():
            if not (
                self.is_exec
                or task.assignee_id == self.asker_id
                or (task.assigner_id and task.assigner_id == self.asker_id)
            ):
                return {"error": "not allowed: only the assignee, the giver or an executive"}
            when = parse_due(due, self.today())
            if not when:
                return {
                    "error": f"cannot read {due!r} as a date",
                    "hint": hint("due_formats", self.lang),
                }
            task.due = when
            if note:
                task.note = note[:500]
            self.tasks.save(task)
            if not status and not (assignee or "").strip():
                return {"updated": self._view(task)}
        if status:
            if not (self.is_exec or task.assignee_id == self.asker_id):
                return {"error": "not allowed: a member may only update their own tasks"}
            if status not in STATUSES:
                return {"error": "status must be open or done"}
        if (assignee or "").strip():
            return self._reassign(task, assignee, status, note)
        done = self.tasks.set_status(self.guild_id, task.id, status, note)
        return {"updated": self._view(done)} if done else {"error": "could not update"}

    def _reassign(self, task: Task, assignee: str, status: str, note: str) -> dict:
        """Move a task to another member. Executives and the person who gave the task may. The new
        assignee gets the confirmation card (nobody, when the asker takes it themself)."""
        if not (self.is_exec or (task.assigner_id and task.assigner_id == self.asker_id)):
            return {"error": "not allowed: only the person who gave the task or an executive"}
        uid = match_person(assignee, self.people)
        if (assignee or "").strip().casefold().strip(".!") in ME_WORDS:
            uid = self.asker_id
        if not uid:
            return {"error": f"cannot tell who {assignee!r} is; use the member's name or @mention"}
        if uid == task.assignee_id and task.confirm != "declined":
            return {"updated": self._view(task), "note": "already assigned to this person"}
        task.assignee_id, task.assignee_name = uid, self.people[uid]
        task.assigner_id, task.assigner_name = self.asker_id, self.people.get(self.asker_id, "")
        task.handshake_message, task.confirm_at = "", ""
        task.confirm = "accepted" if uid == self.asker_id else ""
        if note:
            task.note = note[:500]
        self.tasks.save(task)
        if status:
            self.tasks.set_status(self.guild_id, task.id, status, note)
            task = self.tasks.get(self.guild_id, task.id) or task
        if task.confirm == "" and self._reassigned is not None:
            self._reassigned(task)
        return {"updated": self._view(task)}

    def answer_task_question(self, task_id: str, answer: str, name: str = "") -> dict:
        """The giver's answer to "who takes this task?": mine | other (+ name) | none (not a
        task). Only the giver or an executive; the task must still be waiting for it."""
        task = self.tasks.get(self.guild_id, (task_id or "").lstrip("#").strip())
        if task is None:
            return {"error": "no such task"}
        if not (self.is_exec or (task.assigner_id and task.assigner_id == self.asker_id)):
            return {"error": "not allowed: only the person who gave the task or an executive"}
        if task.confirm != H.OWNER_ASKED:
            return {"error": "this task is not waiting for an owner"}
        if answer not in ("mine", "other", "none"):
            return {"error": "answer must be mine, other or none"}
        if answer == "other" and not (name or "").strip():
            return {"error": "give the name of the person it goes to"}
        now = datetime.now(UTC).isoformat(timespec="seconds")
        res = H.apply_owner_answer(self.tasks, task, answer, name, self.people, now)
        if res.unresolved:
            return {
                "error": f"cannot tell who {res.unresolved!r} is; use the member's name or @mention"
            }
        if self._owner_answered is not None:
            self._owner_answered(res.tasks, res.offer)
        return {"updated": [self._view(x) for x in res.tasks]}

    def request_disclosure(self, topic: str) -> dict:
        if self._request is None or self.memory is None or not self.memory.hidden:
            return {
                "error": "nothing was withheld in this conversation, so there is nothing to ask for"
            }
        if not (topic or "").strip():
            return {"error": "give the topic in a few words"}
        return self._request(topic.strip())

    def my_day(self, day: str | None = None) -> dict:
        if self._my_day is None:
            return {"error": "not available"}
        try:
            d = date.fromisoformat(day) if day else self.today()
        except ValueError:
            return {"error": "date must be YYYY-MM-DD"}
        return self._my_day(d)


def build_coo_tools(c: CooTools, calls: list | None = None) -> list:
    """Plain functions bound to `c` (their docstrings are the tool descriptions)."""
    log = calls if calls is not None else []

    def safe(name: str, args: dict, fn):
        log.append((name, {k: v for k, v in args.items() if v is not None}))
        try:
            return fn()
        except (TypeError, ValueError) as e:
            return {"error": str(e)}

    def list_tasks(person: str | None = None, status: str = "open") -> dict:
        """The team's task list (work Salid noticed people promising or being given), limited to
        what the asker may see. Each task has an id, assignee, title, due date, status, link.

        Args:
            person: only this person's tasks (name or @mention; "me" for the asker), optional.
            status: open (default), done or all.
        """
        a = dict(person=person, status=status)
        return safe("list_tasks", a, lambda: c.list_tasks(**a))

    def add_task(
        title: str, assignee: str = "", due: str = "", project: str = "", note: str = ""
    ) -> dict:
        """Create a task the asker is handing out right now (they tagged Salid on a message that
        gives work, or replied to one). The asker is the giver; each named person gets a card to
        confirm. Call once per task (several names for the SAME task are fine: one task each).

        Args:
            title: what has to be done, short.
            assignee: who does it: a name, @mention, or several joined by "and" / comma; empty if
                nobody is named.
            due: date such as 2026-10-30, "ศุกร์", "tomorrow", optional.
            project: part of a project name, optional.
            note: extra detail, optional.
        """
        a = dict(title=title, assignee=assignee, due=due, project=project, note=note)
        return safe("add_task", a, lambda: c.add_task(**a))

    def update_task(
        task_id: str, status: str = "", note: str = "", assignee: str = "", due: str = ""
    ) -> dict:
        """Mark a task done or reopen it, with a short note (for example what was delivered), or
        hand it to someone else. Members may only update their own tasks; executives may update
        any. Only an executive or the person who gave a task may reassign it; the new assignee
        then gets a card to confirm.

        Args:
            task_id: the task id from list_tasks.
            status: done or open; empty to leave it as it is.
            note: short note, optional.
            assignee: member name or @mention to hand the task to, optional.
            due: set the deadline ("friday", "tomorrow", 2026-10-20); the assignee, the giver
                or an executive may.
        """
        a = dict(task_id=task_id, status=status, note=note, assignee=assignee, due=due)
        return safe("update_task", a, lambda: c.update_task(**a))

    def answer_task_question(task_id: str, answer: str, name: str = "") -> dict:
        """Answer "who takes this task?" for a task Salid asked the asker about (it shows
        owner_pending in list_tasks). Use when they say it is theirs (answer=mine), is for
        someone (answer=other, name=that person) or is not a task / should be dropped
        (answer=none). Only the person who gave the task or an executive.

        Args:
            task_id: the task id.
            answer: mine, other or none.
            name: member name or @mention, only with answer=other.
        """
        a = dict(task_id=task_id, answer=answer, name=name)
        return safe("answer_task_question", a, lambda: c.answer_task_question(**a))

    def request_disclosure(topic: str) -> dict:
        """Ask the owner of withheld information for permission to share it with the asker. Only
        call it after a memory tool reported withheld records AND the asker said yes to asking.
        The owner gets a private message with Approve / Decline buttons (never the content); if
        approved, Salid answers the asker privately.

        Args:
            topic: the subject in a few neutral words (what the asker wants to know).
        """
        a = dict(topic=topic)
        return safe("request_disclosure", a, lambda: c.request_disclosure(**a))

    def my_day(date: str | None = None) -> dict:
        """What the asker themself did on a day: their messages per channel, their DM briefings to
        Salid, tasks done and still open. Use for "what did I do today".

        Args:
            date: ISO date YYYY-MM-DD, default today.
        """
        a = dict(day=date)
        return safe("my_day", a, lambda: c.my_day(**a))

    return [list_tasks, add_task, update_task, answer_task_question, request_disclosure, my_day]
