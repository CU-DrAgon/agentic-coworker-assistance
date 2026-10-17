"""Projects for the PM dashboard (`/pm`) and for Salid's agent tools (logic; Discord glue in
`projects_bot.py`, web API in `meeting_scribe/pm_api.py`).

One company, one workspace; one project, one owner. Projects live in the `DocStore` kind
`salid_projects` (key = project id `p-xxxxxx`, `guild_id` = the workspace memory id, so a Discord
server's projects sit next to its tasks). The tasks of a project ARE Salid's own tasks
(`salid_tasks`, see `tasks.py`): a `Task` carries `project_id`, `assignee_email`, `milestone_id`.

Everything here is pure or blocking on the doc store. Who may see / edit what is decided here in
code (`can_read`, `can_contribute`, `can_edit`, `is_finance`), never by the model:

* admin (org owner / admin / super admin): everything;
* project owner: edit everything of the project;
* project member (listed in `members`, or the creator): tasks, milestones, risks, links, overview;
* any org member: read projects with `visibility == "team"`; `members` projects stay hidden;
* finance data (`budget`) only for admins and members flagged `finance` (Authorized Approver).
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .coo_store import DocStore
from .i18n import CATALOG, t
from .tasks import Task, TaskRepo, is_overdue, parse_due

logger = logging.getLogger(__name__)

PROJECTS = "salid_projects"
PM_SETTINGS = "salid_pm_settings"  # guild 0, key = org id: {"visibility": ...}

DEPARTMENTS = ("tech", "business", "finance", "other")
KINDS = ("standard", "hackathon")
PRIORITIES = ("high", "medium", "low")
STATUSES = ("draft", "planned", "in_progress", "on_hold", "completed", "cancelled")
VISIBILITIES = ("team", "members")
DONE_STATUSES = ("completed", "cancelled")
SUBMISSION_STATUSES = (
    "not_started", "in_progress", "submitted", "results_pending", "won", "lost",
)  # fmt: skip
BIG_FIELDS = ("status", "owner", "deadline", "priority")  # a change needs a reason (+ impact)
ON_HOLD_REVIEW_DAYS = 7
URGENT_DAYS = 3
DEADLINE_DAYS = 14
MAX_PROPOSED_RISKS = 5

Department = Literal["tech", "business", "finance", "other"]
Kind = Literal["standard", "hackathon"]
Priority = Literal["high", "medium", "low"]
Status = Literal["draft", "planned", "in_progress", "on_hold", "completed", "cancelled"]
Visibility = Literal["team", "members"]
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:?\d{2}|Z)?)?$"
)


class ReasonRequired(Exception):
    """A big change (status / owner / deadline / priority) came without a reason."""

    def __init__(self, field: str = ""):
        super().__init__("reason_required")
        self.field = field


def _now() -> datetime:
    return datetime.now(UTC)


def stamp(dt: datetime | None = None) -> str:
    return (dt or _now()).isoformat(timespec="seconds")


def _date_ok(v: str, what: str = "date") -> str:
    v = (v or "").strip()
    if v and not _DATE_RE.match(v):
        raise ValueError(f"{what} must be YYYY-MM-DD")
    if v:
        date.fromisoformat(v)
    return v


# ------------------------------------------------------------------ models
class Person(BaseModel):
    """A person ref: a web member (email), a Discord user, or just a name."""

    email: str | None = None
    discord_user_id: str | None = None
    name: str = ""

    @field_validator("email", mode="before")
    @classmethod
    def _email(cls, v: Any) -> str | None:
        v = str(v or "").strip().lower()
        return v or None

    @field_validator("discord_user_id", mode="before")
    @classmethod
    def _discord(cls, v: Any) -> str | None:
        v = str(v or "").strip()
        return v or None

    @field_validator("name", mode="before")
    @classmethod
    def _name(cls, v: Any) -> str:
        return str(v or "").strip()[:120]

    @model_validator(mode="after")
    def _some_identity(self) -> Person:
        if not (self.email or self.discord_user_id or self.name):
            raise ValueError("a person needs an email, a Discord id or a name")
        return self

    @property
    def label(self) -> str:
        return self.name or self.email or self.discord_user_id or ""


class Milestone(BaseModel):
    id: str = ""
    title: str
    due: str = ""
    status: Literal["pending", "done"] = "pending"
    owner: Person | None = None

    @field_validator("due")
    @classmethod
    def _due(cls, v: str) -> str:
        return _date_ok(v, "due")

    @field_validator("title")
    @classmethod
    def _title(cls, v: str) -> str:
        v = v.strip()[:200]
        if not v:
            raise ValueError("title is required")
        return v


class Risk(BaseModel):
    id: str = ""
    title: str
    severity: Literal["high", "medium", "low"] = "medium"
    status: Literal["open", "mitigated", "closed"] = "open"
    owner: Person | None = None
    note: str = ""
    source: Literal["team", "ai"] = "team"
    created_at: str = ""

    @field_validator("title")
    @classmethod
    def _title(cls, v: str) -> str:
        v = v.strip()[:200]
        if not v:
            raise ValueError("title is required")
        return v


class Link(BaseModel):
    id: str = ""
    title: str = ""
    url: str
    kind: Literal["doc", "repo", "demo", "video", "deck", "other"] = "other"

    @field_validator("url")
    @classmethod
    def _url(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^https?://\S+$", v):
            raise ValueError("url must start with http:// or https://")
        return v[:1000]


class ProjectMember(BaseModel):
    person: Person
    role: str = ""


class TeamRole(BaseModel):
    person: Person
    role: str = ""


class Deliverable(BaseModel):
    id: str = ""
    title: str
    kind: Literal["github", "demo", "video", "deck", "other"] = "other"
    url: str = ""
    owner: Person | None = None
    done: bool = False


class Hackathon(BaseModel):
    competition: str = ""
    official_deadline: str = ""  # YYYY-MM-DD or YYYY-MM-DDTHH:MM (Bangkok time)
    registration_open: str = ""
    rules: str = ""
    eligibility: str = ""
    judging_criteria: str = ""
    team_roles: list[TeamRole] = Field(default_factory=list)
    deliverables: list[Deliverable] = Field(default_factory=list)
    submission_status: Literal[
        "not_started", "in_progress", "submitted", "results_pending", "won", "lost"
    ] = "not_started"
    submitted_at: str = ""
    results: str = ""
    contest_id: str | None = None

    @field_validator("official_deadline")
    @classmethod
    def _deadline(cls, v: str) -> str:
        v = (v or "").strip()
        if v and not _DATETIME_RE.match(v):
            raise ValueError("official_deadline must be YYYY-MM-DD or YYYY-MM-DDTHH:MM")
        return v


class Budget(BaseModel):
    currency: str = "THB"
    planned: float = 0
    spent: float = 0
    notes: str = ""
    travel: float = 0


class Change(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    at: str = ""
    by: Person | None = None
    field: str
    from_: str = Field(default="", alias="from")
    to: str = ""
    reason: str = ""
    impact: str = ""


MAX_UPDATES = 50


class Update(BaseModel):
    """A short progress line Salid heard in chat and kept on the project (newest last)."""

    at: str = ""
    by: Person | None = None
    text: str
    quote: str = ""
    channel_id: int = 0
    jump_url: str = ""


class Project(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str = ""
    workspace: str = ""
    name: str
    kind: Kind = "standard"
    department: Department = "other"
    owner: Person
    priority: Priority = "medium"
    status: Status = "planned"
    target_outcome: str = ""
    deadline: str = ""
    visibility: Visibility = "team"
    members: list[ProjectMember] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    overview: str = ""
    created_by: Person | None = None
    created_at: str = ""
    updated_at: str = ""
    milestones: list[Milestone] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)
    hackathon: Hackathon | None = None
    budget: Budget = Field(default_factory=Budget)
    changes: list[Change] = Field(default_factory=list)
    updates: list[Update] = Field(default_factory=list)
    deleted: bool = False

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = v.strip()[:120]
        if not v:
            raise ValueError("name is required")
        return v

    @field_validator("deadline")
    @classmethod
    def _deadline(cls, v: str) -> str:
        return _date_ok(v, "deadline")

    @field_validator("tags")
    @classmethod
    def _tags(cls, v: list[str]) -> list[str]:
        return [x.strip()[:40] for x in v if x and x.strip()][:20]


# ------------------------------------------------------------------ people and permissions
def person_matches(person: Person | None, email: str | None, discord_id: Any = None) -> bool:
    """Is `person` the one with this email / Discord id? ("Me" in the dashboard.)"""
    if person is None:
        return False
    e = (email or "").strip().lower()
    d = str(discord_id or "").strip()
    return bool((e and person.email == e) or (d and person.discord_user_id == d))


def is_me(person: Person | None, me: Person) -> bool:
    return person_matches(person, me.email, me.discord_user_id)


def is_owner(p: Project, me: Person) -> bool:
    return is_me(p.owner, me)


def is_team(p: Project, me: Person) -> bool:
    return any(is_me(m.person, me) for m in p.members)


def can_read(p: Project, me: Person, is_admin: bool) -> bool:
    if p.deleted:
        return False
    if is_admin or p.visibility == "team":
        return True
    return is_owner(p, me) or is_team(p, me) or is_me(p.created_by, me)


def can_contribute(p: Project, me: Person, is_admin: bool) -> bool:
    """Tasks, milestones, risks, links, overview: owner, listed members, creator, admins."""
    return not p.deleted and (
        is_admin or is_owner(p, me) or is_team(p, me) or is_me(p.created_by, me)
    )


def can_edit(p: Project, me: Person, is_admin: bool) -> bool:
    """Everything else (status, owner, deadline, priority, members...): owner and admins."""
    return not p.deleted and (is_admin or is_owner(p, me))


def is_finance(member: Any, is_admin: bool) -> bool:
    """Finance data: admins and members flagged `finance` (Authorized Approver)."""
    return bool(is_admin or (member is not None and getattr(member, "finance", False)))


def strip_finance(doc: dict) -> dict:
    """The project JSON without any finance field (non-finance viewers)."""
    return {k: v for k, v in doc.items() if k != "budget"}


# ------------------------------------------------------------------ templates
SECTION_LABELS = {
    "overview": ("ภาพรวม", "Overview"),
    "tasks": ("งาน", "Tasks"),
    "milestones": ("เหตุการณ์สำคัญ", "Milestones"),
    "risks": ("ความเสี่ยงและปัญหา", "Risks & Issues"),
    "links": ("ไฟล์และลิงก์", "Files & Links"),
    "competition": ("ชื่อการแข่งขันและเส้นตายทางการ", "Competition Name & Official Deadline"),
    "rules": ("กติกา คุณสมบัติ และเกณฑ์ตัดสิน", "Rules, Eligibility & Judging Criteria"),
    "team": ("ทีมและบทบาท", "Team Members & Roles"),
    "deliverables": ("โปรโตไทป์ เดโม และสไลด์นำเสนอ", "Prototype, Demo & Pitch Deck"),
    "submission": ("เช็กลิสต์ส่งงานและการยืนยัน", "Submission Checklist & Confirmation"),
    "budget": ("งบ ค่าเดินทาง และผลการแข่งขัน", "Budget, Travel Costs & Results"),
}
_STANDARD_SECTIONS = ["overview", "tasks", "milestones", "risks", "links"]
_HACKATHON_SECTIONS = [
    *_STANDARD_SECTIONS, "competition", "rules", "team", "deliverables", "submission", "budget",
]  # fmt: skip
DEFAULT_DELIVERABLES = (
    ("GitHub repo", "github"),
    ("Demo", "demo"),
    ("Video", "video"),
    ("Pitch deck", "deck"),
)
TEMPLATES: dict[str, dict[str, Any]] = {
    "standard": {"sections": _STANDARD_SECTIONS, "defaults": {}},
    "hackathon": {
        "sections": _HACKATHON_SECTIONS,
        "defaults": {"deliverables": [title for title, _ in DEFAULT_DELIVERABLES]},
    },
}


def templates_view() -> list[dict]:
    """Templates for Settings: `sections` are the English labels; `section_labels` carry both."""
    out = []
    for kind, tpl in TEMPLATES.items():
        out.append(
            {
                "kind": kind,
                "sections": [f"sec_{k}" for k in tpl["sections"]],
                "section_labels": [
                    {"key": f"sec_{k}", "th": SECTION_LABELS[k][0], "en": SECTION_LABELS[k][1]}
                    for k in tpl["sections"]
                ],
                "defaults": tpl["defaults"],
            }
        )
    return out


def apply_template(p: Project) -> Project:
    """Fill the kind's template: a hackathon gets its block with the 4 default deliverables."""
    if p.kind == "hackathon":
        if p.hackathon is None:
            p.hackathon = Hackathon()
        if not p.hackathon.deliverables:
            p.hackathon.deliverables = [
                Deliverable(id=f"d{i}", title=title, kind=kind)  # type: ignore[arg-type]
                for i, (title, kind) in enumerate(DEFAULT_DELIVERABLES, 1)
            ]
    return p


def next_item_id(items: list, prefix: str) -> str:
    """m1, m2... (max + 1, so a deleted id is not reused right away)."""
    top = 0
    for it in items:
        m = re.fullmatch(rf"{prefix}(\d+)", getattr(it, "id", "") or "")
        if m:
            top = max(top, int(m[1]))
    return f"{prefix}{top + 1}"


# ------------------------------------------------------------------ changes
def _same(a: Any, b: Any) -> bool:
    def norm(x: Any) -> Any:
        return x.model_dump() if isinstance(x, BaseModel) else x

    return norm(a) == norm(b)


def _show(v: Any) -> str:
    if isinstance(v, Person):
        return v.label
    if isinstance(v, list):
        return ", ".join(_show(x) for x in v)
    return "" if v is None else str(v)


def record_change(
    p: Project,
    field: str,
    old: Any,
    new: Any,
    by: Person | None,
    *,
    reason: str = "",
    impact: str = "",
    now: datetime | None = None,
) -> Change:
    """Append a change entry. status / owner / deadline / priority need a reason
    (`ReasonRequired`); impact is optional. No approval step."""
    if field in BIG_FIELDS and not (reason or "").strip():
        raise ReasonRequired(field)
    ch = Change(
        at=stamp(now),
        by=by,
        field=field,
        **{"from": _show(old)},
        to=_show(new),
        reason=(reason or "").strip()[:500],
        impact=(impact or "").strip()[:500],
    )
    p.changes.append(ch)
    return ch


EDITABLE_FIELDS = (
    "name", "department", "owner", "priority", "status", "target_outcome", "deadline",
    "visibility", "tags", "overview", "members",
)  # fmt: skip
CONTRIBUTOR_FIELDS = ("overview",)  # what a plain project member may edit through PATCH


def apply_updates(
    p: Project,
    updates: dict[str, Any],
    by: Person | None,
    *,
    reason: str = "",
    impact: str = "",
    now: datetime | None = None,
) -> tuple[Project, list[Change]]:
    """`p` with `updates` applied and validated (pydantic `ValidationError` on a bad value).
    Raises `ReasonRequired` BEFORE anything changes when a big field changes without a reason.
    Returns (new project, change entries recorded for the big fields)."""
    merged = {**p.model_dump(by_alias=True), **updates}
    new = Project.model_validate(merged)
    changed = [
        f for f in updates if f in EDITABLE_FIELDS and not _same(getattr(p, f), getattr(new, f))
    ]
    for f in changed:
        if f in BIG_FIELDS and not (reason or "").strip():
            raise ReasonRequired(f)
    new.updated_at = stamp(now)
    out: list[Change] = []
    for f in changed:
        if f in BIG_FIELDS:
            out.append(
                record_change(
                    new,
                    f,
                    getattr(p, f),
                    getattr(new, f),
                    by,
                    reason=reason,
                    impact=impact,
                    now=now,
                )  # fmt: skip
            )
    return new, out


# ------------------------------------------------------------------ stats
def _on_hold_since(p: Project) -> date | None:
    stamps = [c.at for c in p.changes if c.field == "status" and c.to == "on_hold"]
    raw = max(stamps) if stamps else p.updated_at
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        return None


def _unassigned(t: Task) -> bool:
    return not (t.assignee_id or t.assignee_email or t.assignee_name)


def project_stats(p: Project, tasks: list[Task], today: date) -> dict:
    """Counts for a card plus the "needs review" flag: a HIGH priority project that is late
    (deadline passed and not finished, or >= 1 overdue task, or on hold for over 7 days)."""
    mine = [x for x in tasks if x.project_id == p.id]
    overdue = sum(1 for x in mine if is_overdue(x, today))
    reasons: list[str] = []
    if p.priority == "high" and p.status not in DONE_STATUSES:
        if p.deadline and p.deadline < today.isoformat():
            reasons.append("pm_review_deadline")
        if overdue:
            reasons.append("pm_review_tasks")
        if p.status == "on_hold":
            since = _on_hold_since(p)
            if since is not None and (today - since).days > ON_HOLD_REVIEW_DAYS:
                reasons.append("pm_review_on_hold")
    return {
        "tasks_total": len(mine),
        "tasks_done": sum(1 for x in mine if x.status == "done"),
        "tasks_overdue": overdue,
        "tasks_unassigned": sum(1 for x in mine if x.status == "open" and _unassigned(x)),
        "milestones_done": sum(1 for m in p.milestones if m.status == "done"),
        "milestones_total": len(p.milestones),
        "needs_review": bool(reasons),
        "review_reasons": reasons,
        "review_reasons_text": [{"key": r, "th": t(r, "th"), "en": t(r, "en")} for r in reasons],
    }


# ------------------------------------------------------------------ tasks as JSON
def task_assignee(task: Task) -> dict:
    return {
        "email": task.assignee_email or None,
        "discord_user_id": str(task.assignee_id) if task.assignee_id > 0 else None,
        "name": task.assignee_name or "",
    }


def task_mine(task: Task, email: str | None, discord_id: Any) -> bool:
    e = (email or "").strip().lower()
    d = str(discord_id or "").strip()
    return bool(
        (e and task.assignee_email.lower() == e)
        or (d and task.assignee_id > 0 and str(task.assignee_id) == d)
    )


def task_json(task: Task, project_name: str | None = None) -> dict:
    out = {
        "id": task.id,
        "project_id": task.project_id,
        "title": task.title,
        "assignee": task_assignee(task),
        "due": task.due,
        "status": task.status,
        "note": task.note,
        "url": task.source_link,
        "milestone_id": task.milestone_id,
        "created_from": task.created_from,
        "updated_at": task.updated_at,
    }
    if project_name is not None:
        out["project_name"] = project_name
    return out


def apply_assignee(task: Task, person: Person | None) -> None:
    """Point a task at a person (None clears it)."""
    if person is None:
        task.assignee_id, task.assignee_name, task.assignee_email = 0, "", ""
        return
    task.assignee_email = person.email or ""
    task.assignee_id = (
        int(person.discord_user_id) if (person.discord_user_id or "").isdigit() else 0
    )
    task.assignee_name = person.name or (person.email or "").split("@")[0]


def make_task(
    guild_id: int,
    project_id: str,
    title: str,
    *,
    assignee: Person | None = None,
    due: str = "",
    milestone_id: str = "",
    note: str = "",
    created_from: str = "manual",
) -> Task:
    task = Task(
        guild_id=guild_id,
        project_id=project_id,
        title=title.strip()[:300],
        due=_date_ok(due, "due"),
        milestone_id=milestone_id,
        note=note[:500],
        created_from=created_from,
        created_at=stamp(),
    )
    apply_assignee(task, assignee)
    return task


# ------------------------------------------------------------------ views
def project_json(
    p: Project, *, finance_ok: bool, stats: dict, tasks: list[Task] | None = None
) -> dict:
    doc = p.model_dump(by_alias=True)
    doc.pop("deleted", None)
    if not finance_ok:
        doc = strip_finance(doc)
    doc["stats"] = stats
    if tasks is not None:
        doc["tasks"] = [task_json(x) for x in tasks]
    return doc


CARD_DROP = ("milestones", "risks", "links", "changes", "updates", "overview", "budget")


def card_json(p: Project, stats: dict) -> dict:
    doc = p.model_dump(by_alias=True)
    for k in (*CARD_DROP, "deleted"):
        doc.pop(k, None)
    if p.hackathon is not None:
        doc["hackathon"] = {
            "submission_status": p.hackathon.submission_status,
            "official_deadline": p.hackathon.official_deadline,
            "checklist_done": sum(1 for d in p.hackathon.deliverables if d.done),
            "checklist_total": len(p.hackathon.deliverables),
        }
    doc["stats"] = stats
    return doc


def deadlines_window(
    projects: list[Project], days: int = DEADLINE_DAYS, today: date | None = None
) -> list[dict]:
    """Project, milestone and hackathon deadlines from today through `days` days ahead."""
    today = today or date.today()
    last = (today + timedelta(days=days)).isoformat()
    first = today.isoformat()
    out: list[dict] = []

    def add(when: str, kind: str, p: Project, title: str) -> None:
        if when and first <= when[:10] <= last:
            out.append(
                {
                    "when": when, "kind": kind, "project_id": p.id,
                    "project_name": p.name, "title": title,
                }
            )  # fmt: skip

    for p in projects:
        if p.deleted or p.status in DONE_STATUSES:
            continue
        add(p.deadline, "project", p, p.name)
        for m in p.milestones:
            if m.status != "done":
                add(m.due, "milestone", p, m.title)
        if p.hackathon and p.hackathon.official_deadline:
            add(p.hackathon.official_deadline, "hackathon", p, p.hackathon.competition or p.name)
    return sorted(out, key=lambda d: (d["when"], d["project_name"], d["title"]))


# ------------------------------------------------------------------ insights
def _person_key(person: Person, members: list[Any]) -> tuple[str, Person, str]:
    """(key, canonical person, department): a person who is an org member is merged by email."""
    for m in members:
        if person_matches(person, m.email, m.discord_user_id):
            canon = Person(email=m.email, discord_user_id=m.discord_user_id, name=person.name)
            return m.email.lower(), canon, (getattr(m, "department", "") or "other")
    key = person.email or (f"d:{person.discord_user_id}" if person.discord_user_id else "")
    return key or f"n:{person.name.casefold()}", person, "other"


def insights(
    projects: list[Project],
    tasks: list[Task],
    members: list[Any],
    finance_ok: bool,
    today: date | None = None,
) -> dict:
    """Company Insights: progress by department / status, who is loaded, hackathons, budget
    (None unless `finance_ok`). `projects` must already be filtered to what the viewer may read."""
    today = today or date.today()
    ids = {p.id for p in projects}
    by_proj: dict[str, list[Task]] = {}
    for x in tasks:
        if x.project_id in ids:
            by_proj.setdefault(x.project_id, []).append(x)

    by_dept = {
        d: {
            "department": d, "projects": 0, "active": 0, "completed": 0,
            "tasks_done": 0, "tasks_total": 0,
        }
        for d in DEPARTMENTS
    }  # fmt: skip
    by_status = {s: 0 for s in STATUSES}
    people: dict[str, dict] = {}

    def entry(person: Person) -> dict:
        key, canon, dept = _person_key(person, members)
        e = people.get(key)
        if e is None:
            e = people[key] = {
                "person": canon.model_dump(), "open_tasks": 0, "projects": set(),
                "department": dept,
            }  # fmt: skip
        elif person.name and not e["person"]["name"]:
            e["person"]["name"] = person.name
        return e

    for p in projects:
        row = by_dept[p.department]
        row["projects"] += 1
        row["active"] += p.status == "in_progress"
        row["completed"] += p.status == "completed"
        mine = by_proj.get(p.id, [])
        row["tasks_total"] += len(mine)
        row["tasks_done"] += sum(1 for x in mine if x.status == "done")
        by_status[p.status] += 1
        if p.status in DONE_STATUSES:
            continue
        entry(p.owner)["projects"].add(p.id)
        for m in p.members:
            entry(m.person)["projects"].add(p.id)
        for x in mine:
            if x.status == "open" and not _unassigned(x):
                who = Person(
                    email=x.assignee_email or None,
                    discord_user_id=str(x.assignee_id) if x.assignee_id > 0 else None,
                    name=x.assignee_name,
                )
                e = entry(who)
                e["open_tasks"] += 1
                e["projects"].add(p.id)

    plist = []
    by_dept_people: dict[str, list[dict]] = {d: [] for d in DEPARTMENTS}
    for e in sorted(
        people.values(),
        key=lambda e: (
            -e["open_tasks"],
            e["person"]["name"] or "",
        ),
    ):
        item = {
            "person": e["person"], "open_tasks": e["open_tasks"],
            "projects": len(e["projects"]),
        }  # fmt: skip
        plist.append(item)
        by_dept_people[e["department"] if e["department"] in DEPARTMENTS else "other"].append(item)

    hacks = []
    for p in projects:
        if p.kind != "hackathon" or p.hackathon is None or p.deleted:
            continue
        h = p.hackathon
        due = h.official_deadline if h.official_deadline[:10] >= today.isoformat() else ""
        hacks.append(
            {
                "project_id": p.id,
                "name": p.name,
                "competition": h.competition,
                "next_deadline": due,
                "checklist_done": sum(1 for d in h.deliverables if d.done),
                "checklist_total": len(h.deliverables),
                "submission_status": h.submission_status,
            }
        )
    hacks.sort(key=lambda h: (h["next_deadline"] or "9999", h["name"]))

    budget = None
    if finance_ok:
        rows = [
            {
                "project_id": p.id, "name": p.name, "department": p.department,
                "currency": p.budget.currency, "planned": p.budget.planned,
                "spent": p.budget.spent, "travel": p.budget.travel,
            }
            for p in projects
        ]  # fmt: skip
        budget = {
            "by_project": rows,
            "total_planned": sum(r["planned"] for r in rows),
            "total_spent": sum(r["spent"] for r in rows),
            "currency": (rows[0]["currency"] if rows else "THB"),
        }
    return {
        "progress": {"by_department": list(by_dept.values()), "by_status": by_status},
        "resources": {"people": plist, "by_department_people": by_dept_people},
        "hackathons": hacks,
        "budget": budget,
    }


# ------------------------------------------------------------------ repository
class ProjectRepo:
    def __init__(self, docs: DocStore, clock: Callable[[], datetime] | None = None):
        self.docs = docs
        self._clock = clock or _now

    @staticmethod
    def _load(d: dict | None) -> Project | None:
        if not d:
            return None
        try:
            return Project.model_validate({k: v for k, v in d.items() if k != "guild_id"})
        except ValueError:
            logger.warning("unreadable project %s", d.get("id"))
            return None

    def get(
        self, guild_id: int, project_id: str, *, include_deleted: bool = False
    ) -> Project | None:
        p = self._load(self.docs.get(PROJECTS, guild_id, project_id))
        return p if p and (include_deleted or not p.deleted) else None

    def list(self, guild_id: int, *, include_deleted: bool = False) -> list[Project]:
        out = [p for d in self.docs.find(PROJECTS, guild_id) if (p := self._load(d))]
        out = [p for p in out if include_deleted or not p.deleted]
        return sorted(out, key=lambda p: (p.name.casefold(), p.id))

    def create(self, guild_id: int, p: Project) -> Project:
        for _ in range(8):
            pid = "p-" + uuid.uuid4().hex[:6]
            if self.docs.get(PROJECTS, guild_id, pid) is None:
                break
        p.id = pid
        p.created_at = p.updated_at = stamp(self._clock())
        return self.save(guild_id, p, touch=False)

    def save(self, guild_id: int, p: Project, *, touch: bool = True) -> Project:
        if touch:
            p.updated_at = stamp(self._clock())
        self.docs.put(PROJECTS, guild_id, p.id, p.model_dump(by_alias=True))
        return p

    def soft_delete(self, guild_id: int, project_id: str, by: Person | None = None) -> bool:
        p = self.get(guild_id, project_id)
        if p is None:
            return False
        p.status, p.deleted = "cancelled", True
        record_change(
            p, "deleted", False, True, by, reason="deleted", now=self._clock()
        )  # not a big field: no reason needed, text only
        self.save(guild_id, p)
        return True


class PmSettings:
    """Per-org dashboard defaults (kept apart from the org document)."""

    def __init__(self, docs: DocStore):
        self.docs = docs

    def get(self, org_id: str) -> dict:
        d = self.docs.get(PM_SETTINGS, 0, org_id) or {}
        vis = d.get("visibility")
        return {"visibility": vis if vis in VISIBILITIES else "team"}

    def set_visibility(self, org_id: str, visibility: str) -> dict:
        if visibility not in VISIBILITIES:
            raise ValueError("visibility must be team or members")
        self.docs.put(PM_SETTINGS, 0, org_id, {"visibility": visibility})
        return self.get(org_id)


# ------------------------------------------------------------------ finding projects by words
def find_projects(projects: list[Project], fragment: str) -> list[Project]:
    """Projects named by `fragment`: an id, an exact name, else a unique-ish name fragment."""
    s = (fragment or "").strip().casefold()
    if not s:
        return []
    live = [p for p in projects if not p.deleted]
    by_id = [p for p in live if p.id.casefold() == s]
    if by_id:
        return by_id
    exact = [p for p in live if p.name.casefold() == s]
    if exact:
        return exact
    return [p for p in live if s in p.name.casefold()]


_PROJECT_WORDS = ("โปรเจกต์", "โปรเจ็กต์", "โปรเจค", "โครงการ", "project")


def project_in_text(projects: list[Project], text: str) -> Project | None:
    """The (longest-named) running project a sentence points at with "โปรเจกต์ X" / "project X"."""
    best: Project | None = None
    for p in projects:
        if p.deleted or p.status in DONE_STATUSES or len(p.name) < 2:
            continue
        for w in _PROJECT_WORDS:
            if re.search(
                rf"{re.escape(w)}\s*[:：]?\s*[\"'“”‘’]?{re.escape(p.name)}",
                text or "",
                re.IGNORECASE,
            ) and (best is None or len(p.name) > len(best.name)):
                best = p
    return best


def link_task_to_project(docs: DocStore, guild_id: int, task: Task) -> bool:
    """Set `task.project_id` when its text names a known project. Never raises."""
    try:
        if task.project_id:
            return False
        p = project_in_text(ProjectRepo(docs).list(guild_id), task.title)
        if p is None:
            return False
        task.project_id = p.id
        return True
    except Exception:
        logger.warning("task -> project link failed", exc_info=True)
        return False


_DEPT_WORDS = {
    "tech": (
        "เทค", "ซอฟต์แวร์", "แอป", "เว็บ", "ระบบ", "โค้ด", "dev", "app", "website", "api", "ai",
        "software", "backend", "frontend", "bot", "hackathon", "แฮกกะธอน",
    ),
    "business": (
        "ขาย", "การตลาด", "ลูกค้า", "พาร์ทเนอร์", "แคมเปญ", "sales", "marketing", "customer",
        "partner", "campaign", "ธุรกิจ", "business",
    ),
    "finance": (
        "การเงิน", "งบ", "บัญชี", "ภาษี", "finance", "budget", "invoice", "accounting", "tax",
    ),
}  # fmt: skip


def guess_department(text: str) -> str:
    """A department when the words point at exactly one, else "" (then Salid asks)."""
    low = (text or "").casefold()
    hits = {d: sum(w in low for w in ws) for d, ws in _DEPT_WORDS.items()}
    top = max(hits.values())
    best = [d for d, n in hits.items() if n == top]
    return best[0] if top and len(best) == 1 else ""


# ------------------------------------------------------------------ agent tools
@dataclass
class ProjectsCtx:
    """Who asks and what they may do, fixed by the bot in code."""

    guild_id: int
    asker: Person
    is_admin: bool
    lang: str
    repo: ProjectRepo
    tasks: TaskRepo
    now: Callable[[], datetime]
    resolve: Callable[[str], list[Person]]  # members matching a name / @mention
    workspace: str = ""


STRINGS = {
    "pm_review_deadline": {
        "th": "โปรเจกต์ความสำคัญสูง เลยกำหนดส่งแล้ว",
        "en": "high priority and past its deadline",
    },
    "pm_review_tasks": {
        "th": "โปรเจกต์ความสำคัญสูง มีงานเลยกำหนด",
        "en": "high priority with overdue tasks",
    },
    "pm_review_on_hold": {
        "th": "โปรเจกต์ความสำคัญสูง พักไว้เกิน 7 วัน",
        "en": "high priority and on hold for over 7 days",
    },
    "pm_created": {
        "th": "สร้างโปรเจกต์ “{name}” แล้วค่ะ (เจ้าของ {owner}, {department}, ความสำคัญ {priority}, รหัส {id})",
        "en": "Created project “{name}” (owner {owner}, {department}, priority {priority}, id {id}).",
    },
    "pm_exists": {
        "th": "มีโปรเจกต์ชื่อ “{name}” อยู่แล้วค่ะ (รหัส {id})",
        "en": "A project called “{name}” already exists (id {id}).",
    },
    "pm_need_name": {"th": "ขอชื่อโปรเจกต์ด้วยค่ะ", "en": "I need a project name."},
    "pm_need_department": {
        "th": "โปรเจกต์นี้อยู่แผนกไหนคะ (เทค ธุรกิจ การเงิน หรืออื่นๆ)",
        "en": "Which department is it for: tech, business, finance or other?",
    },
    "pm_bad_value": {
        "th": "ค่า “{value}” ใช้กับ {field} ไม่ได้ค่ะ",
        "en": "“{value}” is not valid for {field}.",
    },
    "pm_owner_unknown": {
        "th": "ไม่พบสมาชิกชื่อ “{name}” ค่ะ",
        "en": "I cannot find a member called “{name}”.",
    },
    "pm_owner_ambiguous": {
        "th": "มีหลายคนชื่อใกล้เคียงกับ “{name}”: {names} ระบุให้ชัดอีกนิดนะคะ",
        "en": "Several members match “{name}”: {names}. Which one?",
    },
    "pm_not_found": {
        "th": "ไม่พบโปรเจกต์ “{name}” ที่คุณเปิดดูได้ค่ะ",
        "en": "I cannot find a project “{name}” that you can open.",
    },
    "pm_ambiguous": {
        "th": "มีหลายโปรเจกต์ที่ตรงกับ “{name}”: {names} ระบุให้ชัดอีกนิดนะคะ",
        "en": "Several projects match “{name}”: {names}. Which one?",
    },
    "pm_forbidden": {
        "th": "ขอโทษค่ะ เรื่องนี้ต้องเป็นเจ้าของโปรเจกต์หรือผู้ดูแลเท่านั้น",
        "en": "Sorry, only the project owner or an admin can do that.",
    },
    "pm_forbidden_member": {
        "th": "ขอโทษค่ะ ต้องเป็นสมาชิกของโปรเจกต์ถึงจะทำเรื่องนี้ได้",
        "en": "Sorry, you need to be on the project to do that.",
    },
    "pm_reason_required": {
        "th": "การเปลี่ยน {field} ต้องมีเหตุผลค่ะ ขอเหตุผล (และผลกระทบ ถ้ามี) หน่อย",
        "en": "Changing {field} needs a reason. Please give one (and the impact, if any).",
    },
    "pm_updated": {
        "th": "อัปเดต {field} ของ “{name}” เป็น {value} แล้วค่ะ",
        "en": "Updated {field} of “{name}” to {value}.",
    },
    "pm_unchanged": {
        "th": "{field} ของ “{name}” เป็น {value} อยู่แล้วค่ะ",
        "en": "{field} of “{name}” is already {value}.",
    },
    "pm_task_added": {
        "th": "เพิ่มงาน “{title}” ในโปรเจกต์ “{name}” แล้วค่ะ (รหัส {id})",
        "en": "Added task “{title}” to “{name}” (id {id}).",
    },
    "pm_task_unassigned": {
        "th": "งานนี้ยังไม่มีผู้รับผิดชอบ ให้ใครดูแลคะ",
        "en": "The task has no assignee yet. Who should take it?",
    },
    "pm_none": {"th": "ยังไม่มีโปรเจกต์ค่ะ", "en": "No projects yet."},
    "pm_lbl_owner": {"th": "เจ้าของ", "en": "Owner"},
    "pm_lbl_deadline": {"th": "กำหนดส่ง", "en": "Deadline"},
    "pm_lbl_goal": {"th": "เป้าหมาย", "en": "Goal"},
    "pm_lbl_tasks": {"th": "งาน", "en": "Tasks"},
    "pm_lbl_done": {"th": "เสร็จ", "en": "done"},
    "pm_lbl_overdue": {"th": "เลยกำหนด", "en": "overdue"},
    "pm_lbl_unassigned": {"th": "ยังไม่มีผู้รับ", "en": "unassigned"},
    "pm_lbl_milestones": {"th": "เหตุการณ์สำคัญถัดไป", "en": "Next milestones"},
    "pm_lbl_risks": {"th": "ความเสี่ยงที่เปิดอยู่", "en": "Open risks"},
    "pm_lbl_hackathon": {"th": "แฮกกะธอน", "en": "Hackathon"},
    "pm_lbl_checklist": {"th": "เช็กลิสต์ส่งงาน", "en": "Submission checklist"},
    "pm_lbl_review": {"th": "ต้องทบทวน", "en": "Needs review"},
    "pm_lbl_mine": {"th": "โปรเจกต์ของฉัน", "en": "My projects"},
    "pm_lbl_my_tasks": {"th": "งานของฉันที่ค้างอยู่", "en": "My open tasks"},
    "pm_lbl_no_project": {"th": "ไม่มีโปรเจกต์", "en": "no project"},
    "pm_lbl_none_open": {"th": "ไม่มีงานค้างค่ะ", "en": "No open tasks."},
    "pm_lbl_updates": {"th": "ความเคลื่อนไหวล่าสุด", "en": "Latest updates"},
}
CATALOG.update(STRINGS)

PROJECTS_PROMPT = (
    "\nโปรเจกต์ (project_create / project_update / project_list / project_show / project_add_task / "
    "my_projects_and_tasks): เมื่อมีคนขอสร้างโปรเจกต์ (“เปิดโปรเจกต์ X”, “create a project called X”) ให้เรียก "
    "project_create ทันที ต้องการแค่ชื่อกับแผนก (tech | business | finance | other); ถ้าเดาแผนกจากคำที่เขาพูดได้ให้เดาเลย "
    "ไม่งั้นถามแผนกครั้งเดียว. เจ้าของโปรเจกต์มีคนเดียวเสมอ ถ้าไม่ได้บอกให้เป็นคนที่ขอ. "
    "kind='hackathon' สำหรับโปรเจกต์แข่งแฮกกะธอน (ได้เช็กลิสต์ส่งงานสำเร็จรูป). deadline พูดเป็นวันที่หรือ “ศุกร์นี้” ได้. "
    "การเปลี่ยน status / owner / deadline / priority ด้วย project_update ต้องมีเหตุผล (reason) เสมอ ถ้าเขาไม่บอกให้ถามเหตุผลสั้นๆ "
    "ก่อน (impact ใส่ถ้ามี); ไม่ต้องขออนุมัติใคร. status: draft | planned | in_progress | on_hold | completed | cancelled. "
    "เพิ่มงานเข้าโปรเจกต์ด้วย project_add_task (assignee ว่าง = ยังไม่มีผู้รับ ให้ถามว่าใครดูแล). "
    "“โปรเจกต์ของฉัน/งานของฉัน/my tasks” ใช้ my_projects_and_tasks; “ภาพรวมโปรเจกต์ X / status of project X” ใช้ project_show. "
    "ระบุโปรเจกต์ด้วยชื่อบางส่วนได้; ถ้าเครื่องมือบอกว่าไม่พบหรือกำกวม ให้บอกตรงๆ และถามชื่อที่ชัดเจน. "
    "เห็นเฉพาะโปรเจกต์ที่ผู้ถามมีสิทธิ์ดู งบประมาณ/การเงินไม่มีในเครื่องมือเหล่านี้ ถ้าถามให้บอกว่าดูได้ในหน้า /pm เฉพาะฝ่ายการเงิน. "
    "ถ้ามีคนบอกว่าสลิดจดอะไรจากแชตผิด (ข้อความขึ้นต้น “บันทึกแล้ว”) ให้แก้ด้วย project_update ตั้งค่ากลับ โดยใส่ reason “correction”. "
    "If someone says Salid recorded something wrong from the chat, set the field back with project_update and reason “correction”. "
    "Projects: create with project_create (name + department are enough), one owner per project, "
    "big changes (status, owner, deadline, priority) need a reason, no approval step."
)


def _err(ctx: ProjectsCtx, reason: str, key: str, **fmt: Any) -> dict:
    return {"ok": False, "reason": reason, "say": t(key, ctx.lang, **fmt)}


def _today(ctx: ProjectsCtx) -> date:
    return ctx.now().date()


def _readable(ctx: ProjectsCtx) -> list[Project]:
    return [p for p in ctx.repo.list(ctx.guild_id) if can_read(p, ctx.asker, ctx.is_admin)]


def _pick(ctx: ProjectsCtx, fragment: str) -> tuple[Project | None, dict | None]:
    found = find_projects(_readable(ctx), fragment)
    if not found:
        return None, _err(ctx, "not_found", "pm_not_found", name=fragment)
    if len(found) > 1:
        names = "; ".join(f"{p.name} ({p.id})" for p in found[:6])
        res = _err(ctx, "ambiguous", "pm_ambiguous", name=fragment, names=names)
        res["matches"] = [{"id": p.id, "name": p.name} for p in found[:6]]
        return None, res
    return found[0], None


def _person_arg(ctx: ProjectsCtx, name: str) -> tuple[Person | None, dict | None]:
    """A member named in words (empty = the asker)."""
    if not (name or "").strip():
        return ctx.asker, None
    found = ctx.resolve(name)
    if not found:
        return None, _err(ctx, "owner_not_found", "pm_owner_unknown", name=name)
    if len(found) > 1:
        names = ", ".join(x.label for x in found[:6])
        return None, _err(ctx, "owner_ambiguous", "pm_owner_ambiguous", name=name, names=names)
    return found[0], None


def _date_arg(ctx: ProjectsCtx, text: str) -> str | None:
    """YYYY-MM-DD from words ("2026-11-30", "ศุกร์", "tomorrow"); "" for empty; None if unreadable."""
    if not (text or "").strip():
        return ""
    return parse_due(text, _today(ctx)) or None


def project_tools(ctx: ProjectsCtx) -> list:
    """The project tools bound to one guild and one asker (chat or voice)."""

    def project_create(
        name: str,
        department: str = "",
        owner: str = "",
        priority: str = "medium",
        kind: str = "standard",
        target_outcome: str = "",
        deadline: str = "",
    ) -> dict:
        """Create a project (anyone may). Only a name and a department are needed.

        Args:
            name: project name.
            department: tech | business | finance | other. Guess it from the words when obvious,
                otherwise leave empty and the tool asks.
            owner: member name or @mention; empty = the person asking. One owner per project.
            priority: high | medium | low.
            kind: standard | hackathon (a hackathon gets the competition block and a checklist).
            target_outcome: the goal in a sentence, optional.
            deadline: date such as 2026-11-30, "ศุกร์", "tomorrow"; optional.
        """
        name = (name or "").strip()
        if not name:
            return _err(ctx, "name_required", "pm_need_name")
        dept = (department or "").strip().lower()
        if dept not in DEPARTMENTS:
            dept = guess_department(f"{name} {target_outcome}") if not dept else ""
        if not dept:
            return _err(ctx, "department_required", "pm_need_department")
        pr = (priority or "medium").strip().lower()
        kd = (kind or "standard").strip().lower()
        if pr not in PRIORITIES:
            return _err(ctx, "bad_value", "pm_bad_value", field="priority", value=priority)
        if kd not in KINDS:
            return _err(ctx, "bad_value", "pm_bad_value", field="kind", value=kind)
        dl = _date_arg(ctx, deadline)
        if dl is None:
            return _err(ctx, "bad_value", "pm_bad_value", field="deadline", value=deadline)
        who, bad = _person_arg(ctx, owner)
        if bad:
            return bad
        for p in ctx.repo.list(ctx.guild_id):
            if p.name.casefold() == name.casefold() and p.status not in DONE_STATUSES:
                return _err(ctx, "exists", "pm_exists", name=p.name, id=p.id) | {"id": p.id}
        p = Project(
            name=name, kind=kd, department=dept, owner=who, priority=pr,  # type: ignore[arg-type]
            status="planned", target_outcome=target_outcome.strip()[:500], deadline=dl,
            visibility="team", created_by=ctx.asker, workspace=ctx.workspace,
        )  # fmt: skip
        p = ctx.repo.create(ctx.guild_id, apply_template(p))
        return {
            "ok": True,
            "id": p.id,
            "say": t(
                "pm_created",
                ctx.lang,
                name=p.name,
                owner=who.label,
                department=dept,
                priority=pr,
                id=p.id,
            ),  # fmt: skip
        }

    def project_update(
        project: str, field: str, value: str, reason: str = "", impact: str = ""
    ) -> dict:
        """Change one field of a project. status, owner, deadline and priority need a `reason`
        (ask for it if the person did not give one); `impact` is optional. No approval step.

        Args:
            project: project name (or part of it) or id.
            field: status | owner | deadline | priority | name | department | target_outcome |
                overview | visibility.
            value: the new value (status: draft | planned | in_progress | on_hold | completed |
                cancelled; priority: high | medium | low; owner: member name; deadline: a date).
            reason: why (required for status, owner, deadline, priority).
            impact: what it changes for the team, optional.
        """
        p, bad = _pick(ctx, project)
        if bad:
            return bad
        f = (field or "").strip().lower()
        if f not in EDITABLE_FIELDS or f in ("members", "tags"):
            return _err(ctx, "bad_value", "pm_bad_value", field=field, value=value)
        if not can_edit(p, ctx.asker, ctx.is_admin) and not (
            f in CONTRIBUTOR_FIELDS and can_contribute(p, ctx.asker, ctx.is_admin)
        ):
            return _err(ctx, "forbidden", "pm_forbidden")
        new_value: Any = (value or "").strip()
        if f == "owner":
            new_value, bad = _person_arg(ctx, value)
            if bad or not (value or "").strip():
                return bad or _err(ctx, "owner_not_found", "pm_owner_unknown", name=value)
            new_value = new_value.model_dump()
        elif f == "deadline":
            new_value = _date_arg(ctx, value)
            if new_value is None:
                return _err(ctx, "bad_value", "pm_bad_value", field=f, value=value)
        elif f in ("status", "priority", "department", "visibility"):
            new_value = str(new_value).lower().replace(" ", "_").replace("-", "_")
        try:
            new, changes = apply_updates(
                p, {f: new_value}, ctx.asker, reason=reason, impact=impact, now=ctx.now()
            )
        except ReasonRequired:
            return _err(ctx, "reason_required", "pm_reason_required", field=f)
        except ValueError:
            return _err(ctx, "bad_value", "pm_bad_value", field=f, value=value)
        shown = getattr(new, f)
        shown = shown.label if isinstance(shown, Person) else shown
        if _same(getattr(p, f), getattr(new, f)):
            return {"ok": True, "changed": False, "say": t(
                "pm_unchanged", ctx.lang, field=f, name=p.name, value=shown or "-")}  # fmt: skip
        ctx.repo.save(ctx.guild_id, new)
        return {
            "ok": True,
            "changed": True,
            "say": t("pm_updated", ctx.lang, field=f, name=p.name, value=shown or "-"),
        }

    def project_list(mine: bool = False, department: str = "", status: str = "") -> dict:
        """List projects the asker may see (up to 20), optionally only theirs, or by department
        (tech | business | finance | other) or status."""
        tasks = ctx.tasks.list(ctx.guild_id, status=None)
        out = []
        for p in _readable(ctx):
            if mine and not (is_owner(p, ctx.asker) or is_team(p, ctx.asker)):
                continue
            if department and p.department != department.strip().lower():
                continue
            if status and p.status != status.strip().lower():
                continue
            st = project_stats(p, tasks, _today(ctx))
            out.append(
                {
                    "id": p.id, "name": p.name, "department": p.department, "owner": p.owner.label,
                    "priority": p.priority, "status": p.status, "deadline": p.deadline,
                    "tasks": f"{st['tasks_done']}/{st['tasks_total']}",
                    "needs_review": st["needs_review"],
                }
            )  # fmt: skip
        if not out:
            return {"ok": True, "projects": [], "say": t("pm_none", ctx.lang)}
        return {"ok": True, "total": len(out), "projects": out[:20]}

    def project_show(project: str) -> dict:
        """A short markdown summary of one project: overview, next milestones, open risks, task
        counts, and for a hackathon the next deadline and checklist."""
        p, bad = _pick(ctx, project)
        if bad:
            return bad
        return {
            "ok": True, "id": p.id,
            "markdown": render_summary(p, ctx.tasks.list(ctx.guild_id, status=None, project_id=p.id),
                                       ctx.lang, _today(ctx)),
        }  # fmt: skip

    def project_add_task(
        project: str, title: str, assignee: str = "", due: str = "", milestone: str = ""
    ) -> dict:
        """Add a task to a project (owner, project members and admins). Empty assignee = nobody
        yet: then ask who should take it.

        Args:
            project: project name (or part of it) or id.
            title: the task.
            assignee: member name or @mention, optional.
            due: date such as 2026-10-30, "ศุกร์", "tomorrow", optional.
            milestone: part of a milestone title of this project, optional.
        """
        p, bad = _pick(ctx, project)
        if bad:
            return bad
        if not can_contribute(p, ctx.asker, ctx.is_admin):
            return _err(ctx, "forbidden", "pm_forbidden_member")
        if not (title or "").strip():
            return _err(ctx, "bad_value", "pm_bad_value", field="title", value=title)
        who = None
        if (assignee or "").strip():
            who, bad = _person_arg(ctx, assignee)
            if bad:
                return bad
        dl = _date_arg(ctx, due)
        if dl is None:
            return _err(ctx, "bad_value", "pm_bad_value", field="due", value=due)
        mid = ""
        if (milestone or "").strip():
            low = milestone.strip().casefold()
            hit = [m for m in p.milestones if low in m.title.casefold() or m.id.casefold() == low]
            if len(hit) != 1:
                return _err(ctx, "bad_value", "pm_bad_value", field="milestone", value=milestone)
            mid = hit[0].id
        task = ctx.tasks.save(
            make_task(ctx.guild_id, p.id, title, assignee=who, due=dl, milestone_id=mid)
        )
        res = {
            "ok": True,
            "id": task.id,
            "say": t("pm_task_added", ctx.lang, title=task.title, name=p.name, id=task.id),
        }
        if who is None:
            res["hint"] = t("pm_task_unassigned", ctx.lang)
        return res

    def my_projects_and_tasks() -> dict:
        """The asker's projects (owner or member) and their open tasks across all projects."""
        projects = _readable(ctx)
        names = {p.id: p.name for p in projects}
        mine = [p for p in projects if is_owner(p, ctx.asker) or is_team(p, ctx.asker)]
        tasks = [
            x
            for x in ctx.tasks.list(ctx.guild_id, status="open")
            if task_mine(x, ctx.asker.email, ctx.asker.discord_user_id)
        ]
        return {
            "ok": True,
            "projects": [
                {
                    "id": p.id,
                    "name": p.name,
                    "status": p.status,
                    "deadline": p.deadline,
                    "role": "owner" if is_owner(p, ctx.asker) else "member",
                }
                for p in mine
            ],  # fmt: skip
            "tasks": [
                {
                    "id": x.id,
                    "title": x.title,
                    "due": x.due,
                    "project": names.get(x.project_id, "") if x.project_id else "",
                    "overdue": is_overdue(x, _today(ctx)),
                }
                for x in tasks[:30]
            ],
        }

    return [
        project_create, project_update, project_list, project_show, project_add_task,
        my_projects_and_tasks,
    ]  # fmt: skip


def render_summary(p: Project, tasks: list[Task], lang: str, today: date) -> str:
    """Markdown summary of a project, at most 25 lines."""
    st = project_stats(p, tasks, today)
    lines = [f"**{p.name}** · {p.department} · {p.priority} · {p.status}"]
    meta = [f"{t('pm_lbl_owner', lang)}: {p.owner.label}"]
    if p.deadline:
        meta.append(f"{t('pm_lbl_deadline', lang)}: {p.deadline}")
    lines.append(" · ".join(meta))
    if p.target_outcome:
        lines.append(f"{t('pm_lbl_goal', lang)}: {p.target_outcome[:200]}")
    if p.overview.strip():
        lines.append(" ".join(p.overview.split())[:300])
    counts = f"{t('pm_lbl_tasks', lang)}: {st['tasks_done']}/{st['tasks_total']} {t('pm_lbl_done', lang)}"
    if st["tasks_overdue"]:
        counts += f", {st['tasks_overdue']} {t('pm_lbl_overdue', lang)}"
    if st["tasks_unassigned"]:
        counts += f", {st['tasks_unassigned']} {t('pm_lbl_unassigned', lang)}"
    lines.append(counts)
    nxt = sorted(
        (m for m in p.milestones if m.status != "done"), key=lambda m: (m.due or "9999", m.id)
    )
    if nxt:
        lines.append(f"{t('pm_lbl_milestones', lang)}:")
        lines += [f"- {m.title}" + (f" ({m.due})" if m.due else "") for m in nxt[:3]]
    risks = [r for r in p.risks if r.status == "open"]
    risks.sort(key=lambda r: {"high": 0, "medium": 1, "low": 2}[r.severity])
    if risks:
        lines.append(f"{t('pm_lbl_risks', lang)}:")
        lines += [f"- {r.title} ({r.severity})" for r in risks[:3]]
    if p.hackathon is not None:
        h = p.hackathon
        bits = [h.competition or p.name]
        if h.official_deadline:
            bits.append(h.official_deadline.replace("T", " "))
        bits.append(h.submission_status)
        lines.append(f"{t('pm_lbl_hackathon', lang)}: " + " · ".join(bits))
        if h.deliverables:
            done = sum(1 for d in h.deliverables if d.done)
            lines.append(f"{t('pm_lbl_checklist', lang)}: {done}/{len(h.deliverables)}")
    if st["needs_review"]:
        lines.append(
            f"{t('pm_lbl_review', lang)}: " + "; ".join(t(r, lang) for r in st["review_reasons"])
        )
    lines = lines[:25]
    if p.updates:  # heard in chat (salid.projects_listen); the newest five, after the capped body
        lines.append(f"{t('pm_lbl_updates', lang)}:")
        lines += [f"- {u.at[:10]} {u.text}" for u in p.updates[-5:][::-1]]
    return "\n".join(lines)
