"""PM dashboard API: `/pm` page and `/api/center/orgs/{org_id}/pm/*` (projects, tasks, milestones,
risks, links, hackathon block, budget, my tasks, insights, settings).

The contract (shared with the page, `static/pm.html`) is `brief-pm-contract.md`; the logic lives in
`salid.projects`. Auth is the command center's: a Google ID token as Bearer, `member_ctx` decides who
belongs to the org. Roles: Admin (org owner / admin / super admin), Project Owner, Member, Finance
(`Member.finance`). Who may read / edit what is decided in `salid.projects`, never by the client.

Projects live in the org's Discord workspace (its memory id = the guild id, next to the Discord
tasks) when there is one, else in the web workspace. Lists run across all connected workspaces.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from meeting_scribe import hub
from meeting_scribe.center import Ctx, get_orgs, member_ctx, role_change_error
from salid import projects as P
from salid.i18n import LANG_NAMES, lang_name
from salid.orgs import Member, OrgStore, Workspace
from salid.tasks import Task, TaskRepo, is_overdue

logger = logging.getLogger(__name__)

ZONE = ZoneInfo("Asia/Bangkok")
ASSESS_TIMEOUT_S = 20.0
HOME_TASKS = 10

router = APIRouter(prefix="/api/center/orgs/{org_id}/pm")

DEPARTMENT_LABELS = {
    "tech": ("เทค", "Tech"),
    "business": ("ธุรกิจ", "Business"),
    "finance": ("การเงิน", "Finance"),
    "other": ("อื่นๆ", "Other"),
}


def now() -> datetime:
    """Bangkok wall clock (tests replace it)."""
    return datetime.now(ZONE)


def get_assessor() -> Callable[[str], str]:
    """Cheap-model call for risk proposals: prompt -> JSON text (tests replace it)."""

    def run(prompt: str) -> str:
        from google.genai import types

        from salid.llm import generate, make_client

        s, _ = hub.get_salid_runtime()
        resp = generate(
            make_client(s),
            s.gate_model,
            prompt,
            types.GenerateContentConfig(
                temperature=0.3, max_output_tokens=900, response_mime_type="application/json"
            ),
        )
        return resp.text or ""

    return run


# ------------------------------------------------------------------ request context
@dataclass
class Pm:
    ctx: Ctx
    orgs: OrgStore
    projects: P.ProjectRepo
    tasks: TaskRepo
    me: P.Person
    member: Member | None
    admin: bool
    finance: bool
    wss: list[Workspace]
    when: datetime

    @property
    def today(self) -> date:
        return self.when.date()

    @property
    def home(self) -> Workspace:
        """Where a new project goes: the Discord workspace when there is one, else web."""
        for ws in self.wss:
            if ws.platform == "discord":
                return ws
        return next((w for w in self.wss if w.platform == "web"), None) or self.wss[0]

    def workspace(self, ws_id: str | None) -> Workspace:
        if not ws_id:
            return self.home
        ws = next((w for w in self.wss if w.id == ws_id), None)
        if ws is None:
            raise HTTPException(422, "unknown workspace")
        return ws


def pm_ctx(ctx: Ctx = Depends(member_ctx), orgs: OrgStore = Depends(get_orgs)) -> Pm:
    email = ctx.email.lower()
    member = next((m for m in ctx.org.members if m.email.lower() == email), None)
    wss = [w for w in ctx.org.workspaces if w.status == "connected"]
    if not wss:
        raise HTTPException(409, "the organization has no connected workspace")
    name = email.split("@")[0]
    me = P.Person(
        email=email,
        discord_user_id=member.discord_user_id if member else None,
        name=name,
    )
    return Pm(
        ctx=ctx,
        orgs=orgs,
        projects=P.ProjectRepo(orgs.docs),
        tasks=TaskRepo(orgs.docs),
        me=me,
        member=member,
        admin=ctx.is_exec,
        finance=P.is_finance(member, ctx.is_exec),
        wss=wss,
        when=now(),
    )


# ------------------------------------------------------------------ helpers
def forbid(detail: str = "Not allowed") -> HTTPException:
    return HTTPException(403, detail)


def enrich(pm: Pm, person: P.Person | None) -> P.Person | None:
    """Fill in what the org knows about a person ref (email <-> Discord id, display name)."""
    if person is None:
        return None
    for m in pm.ctx.org.members:
        if P.person_matches(person, m.email, m.discord_user_id):
            return P.Person(
                email=person.email or m.email.lower(),
                discord_user_id=person.discord_user_id or m.discord_user_id,
                name=person.name or m.email.split("@")[0],
            )
    if not person.name and person.email:
        return P.Person(email=person.email, name=person.email.split("@")[0])
    return person


def enrich_project(pm: Pm, p: P.Project) -> P.Project:
    p.owner = enrich(pm, p.owner) or p.owner
    for m in p.members:
        m.person = enrich(pm, m.person) or m.person
    return p


def invalid(e: ValidationError) -> HTTPException:
    err = e.errors()[0] if e.errors() else {}
    loc = ".".join(str(x) for x in err.get("loc", ()) if x != "body")
    return HTTPException(422, f"{loc}: {err.get('msg', 'invalid')}".strip(": "))


def locate(pm: Pm, project_id: str) -> tuple[P.Project, Workspace]:
    for ws in pm.wss:
        p = pm.projects.get(ws.memory_id, project_id)
        if p is not None:
            p.workspace = ws.id
            return p, ws
    raise HTTPException(404, "Unknown project")


def locate_readable(pm: Pm, project_id: str) -> tuple[P.Project, Workspace]:
    p, ws = locate(pm, project_id)
    if not P.can_read(p, pm.me, pm.admin):
        raise forbid("No access to this project")
    return p, ws


def locate_contributor(pm: Pm, project_id: str) -> tuple[P.Project, Workspace]:
    p, ws = locate_readable(pm, project_id)
    if not P.can_contribute(p, pm.me, pm.admin):
        raise forbid("Only the owner, project members or an admin can change this")
    return p, ws


def all_projects(pm: Pm) -> list[tuple[Workspace, P.Project]]:
    out = []
    for ws in pm.wss:
        for p in pm.projects.list(ws.memory_id):
            p.workspace = ws.id
            out.append((ws, p))
    return out


def readable(pm: Pm) -> list[tuple[Workspace, P.Project]]:
    return [(ws, p) for ws, p in all_projects(pm) if P.can_read(p, pm.me, pm.admin)]


def tasks_by_ws(pm: Pm, pairs: list[tuple[Workspace, P.Project]]) -> dict[str, list[Task]]:
    """All tasks (any status) of each workspace that holds one of `pairs`."""
    cache: dict[str, list[Task]] = {}
    for ws, _ in pairs:
        if ws.id not in cache:
            cache[ws.id] = pm.tasks.list(ws.memory_id, status=None)
    return cache


def stats_of(pm: Pm, p: P.Project, tasks: list[Task]) -> dict:
    return P.project_stats(p, tasks, pm.today)


def mine(pm: Pm, p: P.Project) -> bool:
    return P.is_owner(p, pm.me) or P.is_team(p, pm.me)


def save(pm: Pm, ws: Workspace, p: P.Project) -> P.Project:
    return pm.projects.save(ws.memory_id, p)


def full_json(pm: Pm, ws: Workspace, p: P.Project, with_tasks: bool = False) -> dict:
    tasks = pm.tasks.list(ws.memory_id, status=None, project_id=p.id)
    p.workspace = ws.id
    return P.project_json(
        p,
        finance_ok=pm.finance,
        stats=stats_of(pm, p, tasks),
        tasks=tasks if with_tasks else None,
    )


def new_item_id(items: list, prefix: str) -> str:
    return P.next_item_id(items, prefix)


# ------------------------------------------------------------------ home
@router.get("/home")
def home(pm: Pm = Depends(pm_ctx)) -> dict:
    pairs = readable(pm)
    by_ws = tasks_by_ws(pm, pairs)
    names = {p.id: p.name for _, p in pairs}
    my_open: list[Task] = []
    for ws in pm.wss:
        my_open += pm.tasks.for_person(ws.memory_id, pm.me.email, pm.me.discord_user_id)
    my_open.sort(key=lambda t: (t.due or "9999-99-99", t.id))
    my_project_ids = {p.id for _, p in pairs if mine(pm, p)}
    soon = pm.today.toordinal() + P.URGENT_DAYS
    urgent: dict[str, Task] = {}
    for ws, _ in pairs:
        for t in by_ws[ws.id]:
            if t.status != "open" or not t.due:
                continue
            related = (
                P.task_mine(t, pm.me.email, pm.me.discord_user_id) or t.project_id in my_project_ids
            )
            try:
                due_ord = date.fromisoformat(t.due).toordinal()
            except ValueError:
                continue
            if related and (is_overdue(t, pm.today) or due_ord <= soon):
                urgent[f"{ws.id}:{t.id}"] = t
    at_risk = []
    active = 0
    for ws, p in pairs:
        st = stats_of(pm, p, by_ws[ws.id])
        active += p.status == "in_progress"
        st = P.with_progress_reason(st, p)
        if st["needs_review"] or st.get("at_risk_progress"):
            at_risk.append(P.card_json(p, st))
    return {
        "me": pm.me.model_dump(),
        "counts": {
            "projects": len(pairs),
            "active": active,
            "needs_review": len(at_risk),
            "my_open_tasks": len(my_open),
        },
        "my_tasks": [P.task_json(t, names.get(t.project_id, "")) for t in my_open[:HOME_TASKS]],
        "urgent_tasks": [
            P.task_json(t, names.get(t.project_id, ""))
            for t in sorted(urgent.values(), key=lambda t: (t.due, t.id))
        ],
        "at_risk": at_risk,
        "deadlines": P.deadlines_window([p for _, p in pairs], P.DEADLINE_DAYS, pm.today),
    }


# ------------------------------------------------------------------ projects
@router.get("/projects")
def list_projects(
    department: str = "",
    status: str = "",
    kind: str = "",
    q: str = "",
    mine_only: str = Query(default="", alias="mine"),
    pm: Pm = Depends(pm_ctx),
) -> dict:
    pairs = readable(pm)
    by_ws = tasks_by_ws(pm, pairs)
    needle = q.strip().casefold()
    out = []
    for ws, p in pairs:
        if department and p.department != department:
            continue
        if status and p.status != status:
            continue
        if kind and p.kind != kind:
            continue
        if mine_only and mine_only not in ("0", "false") and not mine(pm, p):
            continue
        if needle and needle not in " ".join([p.name, p.target_outcome, *p.tags]).casefold():
            continue
        out.append(P.card_json(p, stats_of(pm, p, by_ws[ws.id])))
    return {"projects": out}


class ProjectCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    department: P.Department
    owner: P.Person
    priority: P.Priority
    kind: P.Kind = "standard"
    status: P.Status = "planned"
    target_outcome: str = ""
    deadline: str = ""
    visibility: P.Visibility | None = None
    workspace: str | None = None


@router.post("/projects", status_code=201)
def create_project(body: ProjectCreate, pm: Pm = Depends(pm_ctx)) -> dict:
    ws = pm.workspace(body.workspace)
    try:
        p = P.Project(
            name=body.name,
            kind=body.kind,
            department=body.department,
            owner=body.owner,
            priority=body.priority,
            status=body.status,
            target_outcome=body.target_outcome.strip()[:500],
            deadline=body.deadline,
            visibility=body.visibility
            or P.PmSettings(pm.orgs.docs).get(pm.ctx.org.id)["visibility"],
            created_by=pm.me,
            workspace=ws.id,
        )
    except ValidationError as e:
        raise invalid(e) from e
    enrich_project(pm, p)
    p = pm.projects.create(ws.memory_id, P.apply_template(p))
    p.workspace = ws.id
    return full_json(pm, ws, p)


@router.get("/projects/{project_id}")
def get_project(project_id: str, pm: Pm = Depends(pm_ctx)) -> dict:
    p, ws = locate_readable(pm, project_id)
    return full_json(pm, ws, p, with_tasks=True)


class ProjectPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    department: P.Department | None = None
    owner: P.Person | None = None
    priority: P.Priority | None = None
    status: P.Status | None = None
    target_outcome: str | None = None
    deadline: str | None = None
    visibility: P.Visibility | None = None
    tags: list[str] | None = None
    overview: str | None = None
    members: list[P.ProjectMember] | None = None
    reason: str = ""
    impact: str = ""


@router.patch("/projects/{project_id}")
def patch_project(project_id: str, body: ProjectPatch, pm: Pm = Depends(pm_ctx)) -> dict:
    p, ws = locate_readable(pm, project_id)
    updates = {
        k: v
        for k, v in body.model_dump(exclude_unset=True).items()
        if k not in ("reason", "impact") and v is not None
    }
    if not updates:
        return full_json(pm, ws, p)
    if not P.can_edit(p, pm.me, pm.admin) and not (
        set(updates) <= set(P.CONTRIBUTOR_FIELDS) and P.can_contribute(p, pm.me, pm.admin)
    ):
        raise forbid("Only the project owner or an admin can change this")
    if "owner" in updates:
        updates["owner"] = enrich(pm, P.Person.model_validate(updates["owner"])).model_dump()
    if "members" in updates:
        updates["members"] = [
            {**m, "person": enrich(pm, P.Person.model_validate(m["person"])).model_dump()}
            for m in updates["members"]
        ]
    try:
        new, _ = P.apply_updates(
            p, updates, pm.me, reason=body.reason, impact=body.impact, now=pm.when
        )
    except P.ReasonRequired as e:
        raise HTTPException(422, "reason_required") from e
    except ValidationError as e:
        raise invalid(e) from e
    new.workspace = ws.id
    save(pm, ws, new)
    return full_json(pm, ws, new)


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: str, pm: Pm = Depends(pm_ctx)) -> Response:
    if not pm.admin:
        raise forbid("Admin only")
    p, ws = locate(pm, project_id)
    pm.projects.soft_delete(ws.memory_id, p.id, pm.me)
    return Response(status_code=204)


# ------------------------------------------------------------------ tasks
class TaskCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=300)
    assignee: P.Person | None = None
    due: str = ""
    milestone_id: str = ""
    note: str = ""
    start: str = ""
    step: int = Field(default=0, ge=0, le=999)
    pct_done: int = Field(default=0, ge=0, le=100)


class TaskPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, min_length=1, max_length=300)
    assignee: P.Person | None = None
    due: str | None = None
    status: Literal["open", "done"] | None = None
    note: str | None = None
    milestone_id: str | None = None
    start: str | None = None
    step: int | None = Field(default=None, ge=0, le=999)
    pct_done: int | None = Field(default=None, ge=0, le=100)


def check_milestone(p: P.Project, mid: str) -> None:
    if mid and not any(m.id == mid for m in p.milestones):
        raise HTTPException(422, "unknown milestone_id")


@router.get("/projects/{project_id}/tasks")
def project_tasks(project_id: str, status: str = "all", pm: Pm = Depends(pm_ctx)) -> dict:
    p, ws = locate_readable(pm, project_id)
    tasks = pm.tasks.list(ws.memory_id, status=status if status != "all" else None, project_id=p.id)
    return {"tasks": [P.task_json(t) for t in tasks]}


@router.post("/projects/{project_id}/tasks")
def add_task(project_id: str, body: TaskCreate, pm: Pm = Depends(pm_ctx)) -> dict:
    p, ws = locate_contributor(pm, project_id)
    check_milestone(p, body.milestone_id)
    try:
        task = P.make_task(
            ws.memory_id,
            p.id,
            body.title,
            assignee=enrich(pm, body.assignee),
            due=body.due,
            milestone_id=body.milestone_id,
            note=body.note,
        )
        task.start = P._date_ok(body.start, "start")
        task.step, task.pct_done = body.step, body.pct_done
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return P.task_json(pm.tasks.save(task))


@router.patch("/projects/{project_id}/tasks/{task_id}")
def patch_task(project_id: str, task_id: str, body: TaskPatch, pm: Pm = Depends(pm_ctx)) -> dict:
    p, ws = locate_readable(pm, project_id)
    task = pm.tasks.get(ws.memory_id, task_id)
    if task is None or task.project_id != p.id:
        raise HTTPException(404, "Unknown task")
    if not (
        P.task_mine(task, pm.me.email, pm.me.discord_user_id) or P.can_edit(p, pm.me, pm.admin)
    ):
        raise forbid("Only the assignee, the project owner or an admin can change this task")
    data = body.model_dump(exclude_unset=True)
    if "milestone_id" in data:
        check_milestone(p, data["milestone_id"] or "")
    if "title" in data and data["title"] is not None:
        task.title = data["title"].strip()
    if "due" in data:
        try:
            task.due = P._date_ok(data["due"] or "", "due")
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
    if "start" in data:
        try:
            task.start = P._date_ok(data["start"] or "", "start")
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
    if data.get("step") is not None:
        task.step = data["step"]
    if data.get("pct_done") is not None:
        task.pct_done = data["pct_done"]
    if "note" in data:
        task.note = (data["note"] or "")[:500]
    if "milestone_id" in data:
        task.milestone_id = data["milestone_id"] or ""
    if "assignee" in data:
        P.apply_assignee(task, enrich(pm, body.assignee))
    if data.get("status"):
        task.status = data["status"]
        task.done_at = P.stamp(pm.when) if task.status == "done" else ""
    return P.task_json(pm.tasks.save(task))


# ------------------------------------------------------------------ milestones / risks / links
def _create_item(
    pm: Pm, project_id: str, field: str, prefix: str, model: type[BaseModel], data: dict
):
    p, ws = locate_contributor(pm, project_id)
    items = getattr(p, field)
    data = {**data, "id": new_item_id(items, prefix)}
    if "owner" in data and data["owner"]:
        data["owner"] = enrich(pm, P.Person.model_validate(data["owner"])).model_dump()
    if field == "risks":
        data["created_at"] = P.stamp(pm.when)
    try:
        item = model.model_validate(data)
    except ValidationError as e:
        raise invalid(e) from e
    items.append(item)
    save(pm, ws, p)
    return item


def _patch_item(
    pm: Pm, project_id: str, field: str, item_id: str, model: type[BaseModel], data: dict
):
    p, ws = locate_contributor(pm, project_id)
    items = getattr(p, field)
    idx = next((i for i, x in enumerate(items) if x.id == item_id), None)
    if idx is None:
        raise HTTPException(404, f"Unknown {field[:-1]}")
    if data.get("owner"):
        data["owner"] = enrich(pm, P.Person.model_validate(data["owner"])).model_dump()
    try:
        items[idx] = model.model_validate({**items[idx].model_dump(), **data, "id": item_id})
    except ValidationError as e:
        raise invalid(e) from e
    save(pm, ws, p)
    return items[idx]


def _delete_item(pm: Pm, project_id: str, field: str, item_id: str) -> Response:
    p, ws = locate_contributor(pm, project_id)
    items = getattr(p, field)
    if not any(x.id == item_id for x in items):
        raise HTTPException(404, f"Unknown {field[:-1]}")
    setattr(p, field, [x for x in items if x.id != item_id])
    if field == "milestones":  # tasks that pointed at it just lose the link
        for t in pm.tasks.list(ws.memory_id, status=None, project_id=p.id):
            if t.milestone_id == item_id:
                t.milestone_id = ""
                pm.tasks.save(t)
    save(pm, ws, p)
    return Response(status_code=204)


class MilestoneIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    due: str = ""
    status: Literal["pending", "done"] = "pending"
    owner: P.Person | None = None


class MilestonePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = None
    due: str | None = None
    status: Literal["pending", "done"] | None = None
    owner: P.Person | None = None


@router.post("/projects/{project_id}/milestones")
def add_milestone(project_id: str, body: MilestoneIn, pm: Pm = Depends(pm_ctx)) -> dict:
    return _create_item(
        pm, project_id, "milestones", "m", P.Milestone, body.model_dump()
    ).model_dump()


@router.patch("/projects/{project_id}/milestones/{mid}")
def patch_milestone(
    project_id: str, mid: str, body: MilestonePatch, pm: Pm = Depends(pm_ctx)
) -> dict:
    data = body.model_dump(exclude_unset=True)
    return _patch_item(pm, project_id, "milestones", mid, P.Milestone, data).model_dump()


@router.delete("/projects/{project_id}/milestones/{mid}", status_code=204)
def delete_milestone(project_id: str, mid: str, pm: Pm = Depends(pm_ctx)) -> Response:
    return _delete_item(pm, project_id, "milestones", mid)


class RiskIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    severity: Literal["high", "medium", "low"] = "medium"
    status: Literal["open", "mitigated", "closed"] = "open"
    owner: P.Person | None = None
    note: str = ""
    source: Literal["team", "ai"] = "team"


class RiskPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = None
    severity: Literal["high", "medium", "low"] | None = None
    status: Literal["open", "mitigated", "closed"] | None = None
    owner: P.Person | None = None
    note: str | None = None


@router.post("/projects/{project_id}/risks")
def add_risk(project_id: str, body: RiskIn, pm: Pm = Depends(pm_ctx)) -> dict:
    return _create_item(pm, project_id, "risks", "r", P.Risk, body.model_dump()).model_dump()


@router.patch("/projects/{project_id}/risks/{rid}")
def patch_risk(project_id: str, rid: str, body: RiskPatch, pm: Pm = Depends(pm_ctx)) -> dict:
    data = body.model_dump(exclude_unset=True)
    return _patch_item(pm, project_id, "risks", rid, P.Risk, data).model_dump()


@router.delete("/projects/{project_id}/risks/{rid}", status_code=204)
def delete_risk(project_id: str, rid: str, pm: Pm = Depends(pm_ctx)) -> Response:
    return _delete_item(pm, project_id, "risks", rid)


def get_pm_guilds():
    """The Salid guild settings store (the bot reads task_confirm_mode / personal_settings from
    it); None when it cannot be opened. Tests override this dependency."""
    try:
        return hub.get_guild_store()
    except Exception:
        logger.warning("guild settings store unavailable for the PM settings", exc_info=True)
        return None


RISK_PROMPT = """You are a careful project manager. Propose up to {n} concrete risks for this project.
Skip anything already listed under "Known risks".{language_rule}

Project: {name} ({kind}, {department}, priority {priority}, status {status})
Goal: {goal}
Deadline: {deadline}
Overview: {overview}
Milestones: {milestones}
Tasks: {done}/{total} done, {overdue} overdue, {unassigned} without an assignee
Known risks: {known}

Return JSON {{"risks": [{{"title": short, "severity": "high"|"medium"|"low", "note": one sentence
on why and what to do}}]}}. Only risks that follow from the facts above; no filler."""


def _risk_prompt(p: P.Project, stats: dict, language: str = "") -> str:
    """`language`: the company language (salid.audience): documents stay in one language, so the
    risks are written in it, whatever language the project text is in."""
    rule = (
        f" Write the risks in {lang_name(language)}."
        if language in LANG_NAMES
        else " Answer in the language of the project text."
    )
    return RISK_PROMPT.format(
        language_rule=rule,
        n=P.MAX_PROPOSED_RISKS,
        name=p.name,
        kind=p.kind,
        department=p.department,
        priority=p.priority,
        status=p.status,
        goal=p.target_outcome or "-",
        deadline=p.deadline or "-",
        overview=" ".join(p.overview.split())[:800] or "-",
        milestones="; ".join(f"{m.title} ({m.due or '?'}, {m.status})" for m in p.milestones[:10])
        or "-",
        done=stats["tasks_done"],
        total=stats["tasks_total"],
        overdue=stats["tasks_overdue"],
        unassigned=stats["tasks_unassigned"],
        known="; ".join(r.title for r in p.risks[:15]) or "-",
    )


def parse_proposed(text: str, when: datetime) -> list[dict]:
    data = json.loads(text)
    rows = data.get("risks", []) if isinstance(data, dict) else data
    out = []
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict):
            continue
        sev = str(r.get("severity", "medium")).lower()
        try:
            risk = P.Risk(
                id="",
                title=str(r.get("title", "")),
                severity=sev if sev in P.PRIORITIES else "medium",  # type: ignore[arg-type]
                note=str(r.get("note", ""))[:500],
                source="ai",
                created_at=P.stamp(when),
            )
        except ValueError:
            continue
        out.append(risk.model_dump())
        if len(out) >= P.MAX_PROPOSED_RISKS:
            break
    return out


@router.post("/projects/{project_id}/risks/assess")
async def assess_risks(
    project_id: str,
    pm: Pm = Depends(pm_ctx),
    assess: Callable[[str], str] = Depends(get_assessor),
    guilds=Depends(get_pm_guilds),
) -> dict:
    p, ws = await asyncio.to_thread(locate_contributor, pm, project_id)
    tasks = await asyncio.to_thread(pm.tasks.list, ws.memory_id, status=None, project_id=p.id)
    prompt = _risk_prompt(
        p, stats_of(pm, p, tasks), await asyncio.to_thread(company_language, pm, guilds)
    )
    try:
        text = await asyncio.wait_for(asyncio.to_thread(assess, prompt), ASSESS_TIMEOUT_S)
        return {"proposed": parse_proposed(text, pm.when)}
    except Exception as e:  # fail soft: the page shows the note, the team adds risks by hand
        logger.warning("risk assessment failed: %s", type(e).__name__, exc_info=True)
        return {"proposed": [], "note": "AI risk assessment is unavailable right now"}


class LinkIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = ""
    url: str
    kind: Literal["doc", "repo", "demo", "video", "deck", "other"] = "other"


class LinkPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = None
    url: str | None = None
    kind: Literal["doc", "repo", "demo", "video", "deck", "other"] | None = None


@router.post("/projects/{project_id}/links")
def add_link(project_id: str, body: LinkIn, pm: Pm = Depends(pm_ctx)) -> dict:
    return _create_item(pm, project_id, "links", "l", P.Link, body.model_dump()).model_dump()


@router.patch("/projects/{project_id}/links/{lid}")
def patch_link(project_id: str, lid: str, body: LinkPatch, pm: Pm = Depends(pm_ctx)) -> dict:
    data = body.model_dump(exclude_unset=True)
    return _patch_item(pm, project_id, "links", lid, P.Link, data).model_dump()


@router.delete("/projects/{project_id}/links/{lid}", status_code=204)
def delete_link(project_id: str, lid: str, pm: Pm = Depends(pm_ctx)) -> Response:
    return _delete_item(pm, project_id, "links", lid)


# ------------------------------------------------------------------ hackathon / budget
class DeliverablePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = ""
    title: str | None = None
    kind: Literal["github", "demo", "video", "deck", "other"] | None = None
    url: str | None = None
    owner: P.Person | None = None
    done: bool | None = None
    delete: bool = False


class HackathonPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    competition: str | None = None
    official_deadline: str | None = None
    registration_open: str | None = None
    rules: str | None = None
    eligibility: str | None = None
    judging_criteria: str | None = None
    team_roles: list[P.TeamRole] | None = None
    deliverables: list[DeliverablePatch] | None = None
    submission_status: (
        Literal["not_started", "in_progress", "submitted", "results_pending", "won", "lost"] | None
    ) = None
    submitted_at: str | None = None
    results: str | None = None
    contest_id: str | None = None
    reason: str = ""
    impact: str = ""


@router.patch("/projects/{project_id}/hackathon")
def patch_hackathon(project_id: str, body: HackathonPatch, pm: Pm = Depends(pm_ctx)) -> dict:
    p, ws = locate_contributor(pm, project_id)
    if p.kind != "hackathon" or p.hackathon is None:
        raise HTTPException(422, "not_a_hackathon")
    sent = body.model_dump(exclude_unset=True)
    h = p.hackathon.model_dump()
    old_status = p.hackathon.submission_status
    for k, v in sent.items():
        if k in ("reason", "impact", "deliverables"):
            continue
        if k == "team_roles":
            v = [
                {**r, "person": enrich(pm, P.Person.model_validate(r["person"])).model_dump()}
                for r in v or []
            ]
        h[k] = v if v is not None else h.get(k)
    if body.deliverables is not None:
        items = {d["id"]: d for d in h["deliverables"]}
        order = [d["id"] for d in h["deliverables"]]
        for patch in body.deliverables:
            fields = patch.model_dump(exclude_unset=True, exclude={"id", "delete"})
            if patch.owner is not None:
                fields["owner"] = enrich(pm, patch.owner).model_dump()
            if patch.id and patch.id in items:
                if patch.delete:
                    items.pop(patch.id)
                    order.remove(patch.id)
                else:
                    items[patch.id].update({k: v for k, v in fields.items() if v is not None})
            elif not patch.delete:
                new_id = P.next_item_id([P.Deliverable(id=i, title="x") for i in items], "d")
                items[new_id] = {
                    "id": new_id,
                    "title": patch.title or "",
                    **{k: v for k, v in fields.items() if v is not None and k != "title"},
                }
                order.append(new_id)
        h["deliverables"] = [items[i] for i in order]
    try:
        new_h = P.Hackathon.model_validate(h)
    except ValidationError as e:
        raise invalid(e) from e
    if new_h.submission_status != old_status:
        P.record_change(
            p, "submission_status", old_status, new_h.submission_status, pm.me,
            reason=body.reason, impact=body.impact, now=pm.when,
        )  # fmt: skip
        if new_h.submission_status == "submitted" and not new_h.submitted_at:
            new_h.submitted_at = P.stamp(pm.when)
    p.hackathon = new_h
    save(pm, ws, p)
    return full_json(pm, ws, p)


class BudgetPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    currency: str | None = Field(default=None, min_length=1, max_length=8)
    planned: float | None = Field(default=None, ge=0)
    spent: float | None = Field(default=None, ge=0)
    notes: str | None = None
    travel: float | None = Field(default=None, ge=0)


@router.patch("/projects/{project_id}/budget")
def patch_budget(project_id: str, body: BudgetPatch, pm: Pm = Depends(pm_ctx)) -> dict:
    if not pm.finance:
        raise forbid("Finance or admin only")
    p, ws = locate_readable(pm, project_id)
    p.budget = P.Budget.model_validate(
        {
            **p.budget.model_dump(),
            **{k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None},
        }
    )
    save(pm, ws, p)
    return p.budget.model_dump()


# ------------------------------------------------------------------ my tasks / insights
@router.get("/my-tasks")
def my_tasks(status: Literal["open", "done", "all"] = "open", pm: Pm = Depends(pm_ctx)) -> dict:
    out: list[tuple[str, Task, str]] = []
    for ws in pm.wss:
        names = {p.id: p.name for p in pm.projects.list(ws.memory_id)}
        for t in pm.tasks.for_person(
            ws.memory_id,
            pm.me.email,
            pm.me.discord_user_id,
            status=None if status == "all" else status,
        ):
            if t.project_id and t.project_id not in names:
                continue  # its project was deleted
            out.append((t.due or "9999-99-99", t, names.get(t.project_id, "")))
    out.sort(key=lambda r: (r[0], r[1].id))
    return {"tasks": [P.task_json(t, name) for _, t, name in out]}


@router.get("/insights")
def company_insights(pm: Pm = Depends(pm_ctx)) -> dict:
    pairs = readable(pm)
    by_ws = tasks_by_ws(pm, pairs)
    tasks = [t for tl in by_ws.values() for t in tl]
    return P.insights([p for _, p in pairs], tasks, pm.ctx.org.members, pm.finance, pm.today)


# ------------------------------------------------------------------ settings
def discord_guild_ids(pm: Pm) -> list[int]:
    out = []
    for ws in pm.wss:
        if ws.platform == "discord" and str(ws.external_id).isdigit():
            out.append(int(ws.external_id))
    return out


def company_language(pm: Pm, guilds=None) -> str:
    """The company language (the linked Discord server's chosen main language), "" when it is not
    known or not chosen yet. Documents, reports and this page stay in it."""
    for gid in discord_guild_ids(pm)[:1]:
        try:
            g = guilds.find(gid) if guilds is not None else None
        except Exception:
            g = None
        if g is not None and g.language_setting in LANG_NAMES:
            return g.language_setting
    return ""


def settings_view(pm: Pm, guilds=None) -> dict:
    org = pm.ctx.org
    defaults = P.PmSettings(pm.orgs.docs).get(org.id)
    for gid in discord_guild_ids(pm)[:1]:  # the bot's own setting is the truth when it is known
        try:
            g = guilds.find(gid) if guilds is not None else None
        except Exception:
            g = None
        if g is not None:
            defaults = {
                **defaults,
                "task_confirm_mode": g.task_confirm_mode,
                "personal_settings": g.personal_settings,
            }
    return {
        "members": [{**m.model_dump(), "name": m.email.split("@")[0]} for m in org.members],
        "departments": [
            {"key": k, "label_th": th, "label_en": en} for k, (th, en) in DEPARTMENT_LABELS.items()
        ],
        "templates": P.templates_view(),
        "defaults": defaults,
        "language": company_language(pm, guilds),  # the page's default UI language
        "company": {"name": org.name, "logo_url": org.logo_url},
    }


@router.get("/settings")
def get_pm_settings(pm: Pm = Depends(pm_ctx), guilds=Depends(get_pm_guilds)) -> dict:
    return settings_view(pm, guilds)


class MemberPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str
    department: Literal["tech", "business", "finance", "other", ""] | None = None
    finance: bool | None = None
    role: Literal["admin", "member"] | None = None


class DefaultsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    visibility: P.Visibility | None = None
    task_confirm_mode: Literal["notify", "approve"] | None = None
    personal_settings: bool | None = None


class SettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    members: list[MemberPatch] | None = None
    defaults: DefaultsPatch | None = None


@router.patch("/settings")
def patch_pm_settings(
    body: SettingsPatch, pm: Pm = Depends(pm_ctx), guilds=Depends(get_pm_guilds)
) -> dict:
    if not pm.admin:
        raise forbid("Admin only")
    org_id = pm.ctx.org.id
    with pm.orgs.lock:
        org = pm.orgs.get(org_id)
        if org is None:
            raise HTTPException(404, "Unknown organization")
        for patch in body.members or []:
            m = next(
                (x for x in org.members if x.email.lower() == patch.email.strip().lower()), None
            )
            if m is None:
                raise HTTPException(422, "unknown_member")
            sent = patch.model_dump(exclude_unset=True)
            if "role" in sent and sent["role"] is not None:
                if m.email.lower() == org.owner_email.lower() or m.role == "owner":
                    raise HTTPException(422, "owner_role_fixed")
                if sent["role"] != m.role:  # the Members page's rules (`role_change_error`)
                    me = m.email.lower() == pm.ctx.email.lower()
                    if why := role_change_error(pm.ctx, m.role, sent["role"], me):
                        raise HTTPException(*why)
                m.role = sent["role"]
            if sent.get("department") is not None:
                m.department = sent["department"]
            if sent.get("finance") is not None:
                m.finance = sent["finance"]
        pm.orgs.save(org)
        pm.ctx.org = org
    if body.defaults and body.defaults.visibility:
        P.PmSettings(pm.orgs.docs).set_visibility(org_id, body.defaults.visibility)
    d = body.defaults
    if d and d.task_confirm_mode:
        P.PmSettings(pm.orgs.docs).set_task_confirm_mode(org_id, d.task_confirm_mode)
    if d and d.personal_settings is not None:
        P.PmSettings(pm.orgs.docs).set_personal_settings(org_id, d.personal_settings)
    if guilds is not None and d and (d.task_confirm_mode or d.personal_settings is not None):
        for gid in discord_guild_ids(pm):  # the bot reads these from the guild settings
            try:
                if d.task_confirm_mode:
                    guilds.set_task_confirm_mode(gid, d.task_confirm_mode)
                if d.personal_settings is not None:
                    guilds.set_personal_settings(gid, d.personal_settings)
            except Exception:
                logger.warning("could not mirror PM defaults to guild %s", gid, exc_info=True)
    return settings_view(pm, guilds)
