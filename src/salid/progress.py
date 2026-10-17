"""How far a project really is, and SEPARATELY how close its deadline is. Pure (no network).

`progress(project, tasks, today)` returns two independent facts: `done_pct` (the work finished:
tasks done / tasks, or milestones done / milestones when there are no tasks) and `time_pct` (the
share of the time between the start and the deadline that has passed), plus `days_left`, a
`state` and the nearest open item. `progress_line` prints them as ONE line (Thai or English) for
chat, the project summary and the dashboard. The pictures in `progress_chart.py` draw the same
numbers.

States: `on_track` | `behind` (done% is over 20 points under time%) | `at_risk` (behind and a week
or less left, or past the deadline and not completed) | `done` | `no_deadline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from .i18n import CATALOG, t

if TYPE_CHECKING:
    from .projects import Project
    from .tasks import Task

BEHIND_MARGIN = 20  # done% this far under time% = behind
RISK_DAYS = 7  # behind and this close to the deadline = at risk
STATES = ("on_track", "behind", "at_risk", "done", "no_deadline")
STATE_ORDER = {"at_risk": 0, "behind": 1, "on_track": 2, "no_deadline": 3, "done": 4}
DONE_STATUSES = ("completed", "cancelled")

STRINGS = {
    "prog_done": {"th": "ความคืบหน้าจริง {pct}%", "en": "Real progress {pct}%"},
    "prog_time": {"th": "เวลาผ่านไป {pct}%", "en": "{pct}% of the time used"},
    "prog_left": {"th": "เหลือ {n} วัน", "en": "{n} days left"},
    "prog_left_one": {"th": "เหลือ 1 วัน", "en": "1 day left"},
    "prog_left_today": {"th": "ครบกำหนดวันนี้", "en": "due today"},
    "prog_over": {"th": "เลยกำหนด {n} วัน", "en": "{n} days overdue"},
    "prog_over_one": {"th": "เลยกำหนด 1 วัน", "en": "1 day overdue"},
    "prog_no_deadline": {"th": "ยังไม่มีกำหนดส่ง", "en": "no deadline yet"},
    "prog_no_work": {"th": "ยังไม่มีงานให้วัดผล", "en": "nothing to measure yet"},
    "prog_state_on_track": {"th": "✅ ตามแผน", "en": "✅ on track"},
    "prog_state_behind": {"th": "⚠️ ช้ากว่าแผน", "en": "⚠️ behind plan"},
    "prog_state_at_risk": {"th": "🔴 เสี่ยงไม่ทัน", "en": "🔴 at risk"},
    "prog_state_done": {"th": "🏁 เสร็จแล้ว", "en": "🏁 done"},
    "prog_state_no_deadline": {"th": "➖ ไม่มีกำหนดส่ง", "en": "➖ no deadline"},
    "prog_word_on_track": {"th": "ตามแผน", "en": "on track"},
    "prog_word_behind": {"th": "ช้ากว่าแผน", "en": "behind"},
    "prog_word_at_risk": {"th": "เสี่ยงไม่ทัน", "en": "at risk"},
    "prog_word_done": {"th": "เสร็จแล้ว", "en": "done"},
    "prog_word_no_deadline": {"th": "ไม่มีกำหนดส่ง", "en": "no deadline"},
    "prog_step": {"th": "ขั้นที่ {n}", "en": "Step {n}"},
    "prog_gap_left": {"th": "เหลือ {n} วัน", "en": "{n} d left"},
    "prog_gap_over": {"th": "เลย {n} วัน", "en": "{n} d over"},
    "prog_gap_today": {"th": "วันนี้", "en": "today"},
    "prog_lbl_work": {"th": "งานเสร็จ", "en": "Work done"},
    "prog_lbl_time": {"th": "เวลา", "en": "Time"},
    "prog_title_running": {"th": "โปรเจกต์ที่กำลังทำอยู่", "en": "Projects running now"},
    "prog_lbl_steps": {"th": "ขั้นตอน", "en": "Steps"},
}
CATALOG.update(STRINGS)


def _date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _stamp_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).date()
    except ValueError:
        return _date(s)


def task_pct(task: Task) -> int:
    """How far one task is: 100 when done, else its own `pct_done`."""
    return 100 if task.status == "done" else max(0, min(100, int(task.pct_done or 0)))


def ordered_tasks(project: Project | None, tasks: list[Task]) -> list[Task]:
    """Tasks in step order: explicit `step` first (ascending), then by milestone due date, due
    date, start, and title."""
    due_of = {m.id: (m.due or "9999-12-31") for m in (project.milestones if project else [])}

    def key(x: Task) -> tuple:
        return (
            x.step if x.step > 0 else 10**6,
            due_of.get(x.milestone_id, "9999-12-31") if x.milestone_id else "9999-12-31",
            x.due or "9999-12-31",
            x.start or "9999-12-31",
            x.title,
        )

    return sorted(tasks, key=key)


@dataclass
class TaskProgress:
    id: str
    title: str
    n: int  # the number shown: "step n"
    start: str
    due: str
    gap_days: int | None  # due - today (negative = overdue); None when done or no due date
    pct: int
    done: bool
    overdue: bool
    assignee: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Progress:
    done_pct: int | None
    time_pct: int | None
    days_left: int | None  # negative = past the deadline
    state: str
    next_due: dict[str, Any] | None = None  # {kind, title, due, days}
    tasks: list[TaskProgress] = field(default_factory=list)

    def to_dict(self, tasks: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "done_pct": self.done_pct,
            "time_pct": self.time_pct,
            "days_left": self.days_left,
            "state": self.state,
            "next_due": self.next_due,
        }
        if tasks:
            out["tasks"] = [x.to_dict() for x in self.tasks]
        return out


def project_start(project: Project, tasks: list[Task]) -> date | None:
    """When the project began: its creation date, or an earlier planned start of a task."""
    starts = [d for d in (_stamp_date(project.created_at),) if d]
    starts += [d for x in tasks if (d := _date(x.start))]
    if not starts:
        starts = [d for x in tasks if (d := _stamp_date(x.created_at))]
    return min(starts) if starts else None


def progress(project: Project, tasks: list[Task], today: date) -> Progress:
    mine = [x for x in tasks if x.project_id == project.id]
    ordered = ordered_tasks(project, mine)
    total = len(mine)
    ms = project.milestones
    if total:
        done_pct: int | None = round(100 * sum(1 for x in mine if x.status == "done") / total)
    elif ms:
        done_pct = round(100 * sum(1 for m in ms if m.status == "done") / len(ms))
    else:
        done_pct = None

    deadline = _date(project.deadline)
    days_left = (deadline - today).days if deadline else None
    time_pct: int | None = None
    if deadline is not None:
        start = project_start(project, mine)
        if start is None or start >= deadline:
            time_pct = 100 if today >= deadline else 0
        else:
            span = (deadline - start).days
            time_pct = max(0, min(100, round(100 * (today - start).days / span)))

    finished = project.status in DONE_STATUSES or done_pct == 100
    if finished:
        state = "done"
    elif deadline is None:
        state = "no_deadline"
    elif days_left is not None and days_left < 0:
        state = "at_risk"
    elif done_pct is not None and time_pct is not None and done_pct < time_pct - BEHIND_MARGIN:
        state = "at_risk" if days_left is not None and days_left <= RISK_DAYS else "behind"
    else:
        state = "on_track"

    rows: list[TaskProgress] = []
    for i, x in enumerate(ordered, 1):
        due = _date(x.due)
        done = x.status == "done"
        gap = (due - today).days if (due and not done) else None
        rows.append(
            TaskProgress(
                id=x.id, title=x.title, n=x.step if x.step > 0 else i, start=x.start, due=x.due,
                gap_days=gap, pct=task_pct(x), done=done, overdue=bool(gap is not None and gap < 0),
                assignee=x.assignee_name,
            )
        )  # fmt: skip

    cands: list[tuple[str, str, str]] = [
        ("task", x.title, x.due) for x in mine if x.status == "open" and _date(x.due)
    ] + [("milestone", m.title, m.due) for m in ms if m.status != "done" and _date(m.due)]
    nxt = None
    if cands and project.status not in DONE_STATUSES:
        kind, title, due = min(cands, key=lambda c: (c[2], c[0]))
        nxt = {"kind": kind, "title": title, "due": due, "days": (date.fromisoformat(due) - today).days}  # fmt: skip
    return Progress(done_pct, time_pct, days_left, state, nxt, rows)


def gap_text(days: int | None, lang: str) -> str:
    """ "เหลือ 3 วัน" / "เลย 2 วัน" for one task or milestone ('' without a date)."""
    if days is None:
        return ""
    if days < 0:
        return t("prog_gap_over", lang, n=-days)
    if days == 0:
        return t("prog_gap_today", lang)
    return t("prog_gap_left", lang, n=days)


def days_text(days: int | None, lang: str) -> str:
    if days is None:
        return t("prog_no_deadline", lang)
    if days < 0:
        return t("prog_over_one", lang) if days == -1 else t("prog_over", lang, n=-days)
    if days == 0:
        return t("prog_left_today", lang)
    return t("prog_left_one", lang) if days == 1 else t("prog_left", lang, n=days)


def state_word(state: str, lang: str) -> str:
    return t(f"prog_word_{state}", lang)


def progress_line(p: Progress, lang: str) -> str:
    """ONE line: real progress, time used, days left, state."""
    bits = [
        t("prog_done", lang, pct=p.done_pct) if p.done_pct is not None else t("prog_no_work", lang)
    ]
    if p.time_pct is not None:
        bits.append(t("prog_time", lang, pct=p.time_pct))
    if p.state not in ("done", "no_deadline"):
        bits.append(days_text(p.days_left, lang))
    bits.append(t(f"prog_state_{p.state}", lang))
    return " · ".join(bits)
