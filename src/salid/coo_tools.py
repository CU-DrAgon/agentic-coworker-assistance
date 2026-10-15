"""Agent tools of the coordinator role: tasks, disclosure requests and "what did I do today".

Bound in code to one guild and one asker, like the memory tools: the ids are never part of a tool
schema. `build_coo_tools` returns plain functions that ADK wraps; errors come back as data.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from .memory_access import Scope
from .tasks import STATUSES, Task, TaskRepo, is_overdue, match_person, task_visible

MAX_TASKS = 40


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
    ):
        self.guild_id, self.asker_id, self.is_exec, self.scope = guild_id, asker_id, is_exec, scope
        self.tasks, self.people, self.today, self.memory = tasks, people, today, memory
        self._request, self._my_day = request_disclosure, my_day

    def _view(self, t: Task) -> dict:
        return {
            "id": t.id,
            "assignee": t.assignee_name or None,
            "title": t.title,
            "due": t.due or None,
            "status": t.status,
            "overdue": is_overdue(t, self.today()),
            "link": t.source_link or None,
            "note": t.note or None,
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
        return out

    def update_task(self, task_id: str, status: str, note: str = "") -> dict:
        task = self.tasks.get(self.guild_id, (task_id or "").lstrip("#").strip())
        if task is None:
            return {"error": "no such task"}
        if not (self.is_exec or task.assignee_id == self.asker_id):
            return {"error": "not allowed: a member may only update their own tasks"}
        if status not in STATUSES:
            return {"error": "status must be open or done"}
        done = self.tasks.set_status(self.guild_id, task.id, status, note)
        return {"updated": self._view(done)} if done else {"error": "could not update"}

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

    def update_task(task_id: str, status: str, note: str = "") -> dict:
        """Mark a task done or reopen it, with a short note (for example what was delivered).
        Members may only update their own tasks; executives may update any.

        Args:
            task_id: the task id from list_tasks.
            status: done or open.
            note: short note, optional.
        """
        a = dict(task_id=task_id, status=status, note=note)
        return safe("update_task", a, lambda: c.update_task(**a))

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

    return [list_tasks, update_task, request_disclosure, my_day]
