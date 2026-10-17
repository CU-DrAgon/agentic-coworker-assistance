"""Progress pictures and plan import, Discord / agent side (logic in `progress.py`,
`progress_chart.py` and `plan_import.py`).

`progress_tools(ctx)` gives the agent `project_picture` (a PNG of what is running, what a person
works on, or one project) and `project_import_plan` (a Gantt / spreadsheet / text turned into a
project with steps; preview first, then "confirm"). They are built on the same `ProjectsCtx` as the
other project tools, so who may see what is decided there (`can_read`, `can_contribute`), never by
the model. Pictures leave through the tool function's `outbox` (see `qa.tool_files`).

`ProgressMixin._plan_confirm_command` answers a bare "ยืนยัน" / "confirm" without the model when a
plan draft is waiting for that person.
"""

from __future__ import annotations

import asyncio
import io
import logging

import discord

from . import plan_import as PI
from .i18n import CATALOG, t
from .progress import Progress, progress, progress_line
from .progress_chart import person_png, portfolio_png, project_png
from .projects import (
    DONE_STATUSES,
    Milestone,
    Person,
    Project,
    ProjectsCtx,
    _err,
    _person_arg,
    _pick,
    _readable,
    _today,
    apply_assignee,
    apply_template,
    can_contribute,
    find_projects,
    guess_department,
    is_owner,
    is_team,
    make_task,
    next_item_id,
    render_summary,
    task_mine,
)
from .tasks import Task

logger = logging.getLogger(__name__)

MAX_SUMMARY_LINES = 12
WHATS = ("auto", "portfolio", "person", "project")

STRINGS = {
    "pic_failed": {
        "th": "วาดภาพไม่สำเร็จค่ะ ลองใหม่อีกครั้งนะคะ",
        "en": "I could not draw the picture. Please try again.",
    },
    "pic_person_none": {
        "th": "{name} ยังไม่มีงานหรือโปรเจกต์ที่เปิดดูได้ค่ะ",
        "en": "{name} has no open work or projects that I can show you.",
    },
    "pic_bad_what": {
        "th": "ชนิดภาพ “{value}” ไม่มีค่ะ (portfolio, person, project)",
        "en": "“{value}” is not a picture type (portfolio, person, project).",
    },
}
CATALOG.update(STRINGS)


def _active(p: Project) -> bool:
    return p.status not in (*DONE_STATUSES, "draft")


def _line(p: Project, pr: Progress, lang: str) -> str:
    return f"{p.name}: {progress_line(pr, lang)}"


def _stats_of(ctx: ProjectsCtx, projects: list[Project], tasks: list[Task]):
    today = _today(ctx)
    return [(p, progress(p, [x for x in tasks if x.project_id == p.id], today)) for p in projects]


# ------------------------------------------------------------------ pictures
def make_picture(ctx: ProjectsCtx, project: str, person: str, what: str) -> dict:
    today = _today(ctx)
    kind = (what or "auto").strip().lower()
    if kind not in WHATS:
        return _err(ctx, "bad_value", "pic_bad_what", value=what)
    if kind == "auto":
        kind = "person" if (person or "").strip() else "project" if (project or "").strip() else "portfolio"  # fmt: skip
    all_tasks = ctx.tasks.list(ctx.guild_id, status=None)
    readable = _readable(ctx)
    if kind == "project":
        p, bad = _pick(ctx, project)
        if bad:
            return bad
        mine = [x for x in all_tasks if x.project_id == p.id]
        pr = progress(p, mine, today)
        png = project_png(p, mine, ctx.lang, today, prog=pr)
        lines = [progress_line(pr, ctx.lang)]
        name = f"progress-project-{p.id}.png"
    elif kind == "person":
        who, bad = _person_arg(ctx, person)
        if bad:
            return bad
        mine = [
            x for x in all_tasks
            if x.status == "open" and task_mine(x, who.email, who.discord_user_id)
        ]  # fmt: skip
        ids = {x.project_id for x in mine if x.project_id}
        projs = [
            p for p in readable
            if _active(p) and (p.id in ids or is_owner(p, who) or is_team(p, who))
        ]  # fmt: skip
        # only tasks of projects the asker may read
        visible = {p.id for p in readable}
        mine = [x for x in mine if x.project_id in visible]
        if not projs and not mine:
            return _err(ctx, "empty", "pic_person_none", name=who.label)
        items = _stats_of(ctx, projs, all_tasks)
        png = person_png(who.label, items, mine, ctx.lang, today)
        lines = [_line(p, pr, ctx.lang) for p, pr in items][:MAX_SUMMARY_LINES]
        name = "progress-person.png"
    else:
        projs = [p for p in readable if _active(p)]
        if not projs:
            return _err(ctx, "empty", "pm_none")
        items = _stats_of(ctx, projs, all_tasks)
        png = portfolio_png(items, ctx.lang, today=today)
        order = {"at_risk": 0, "behind": 1, "on_track": 2, "no_deadline": 3, "done": 4}
        lines = [
            _line(p, pr, ctx.lang)
            for p, pr in sorted(items, key=lambda it: (order.get(it[1].state, 9), it[0].name))
        ][:MAX_SUMMARY_LINES]
        name = "progress-projects.png"
    ctx.outbox.append((name, png))
    return {"ok": True, "attached": True, "what": kind, "summary_lines": lines}


# ------------------------------------------------------------------ plan import
def _asker_key(ctx: ProjectsCtx) -> str:
    return str(ctx.asker.discord_user_id or ctx.asker.email or ctx.asker.name)


def _asker_id(ctx: ProjectsCtx) -> int:
    d = ctx.asker.discord_user_id or ""
    return int(d) if d.isdigit() else 0


def _resolve_owner(ctx: ProjectsCtx, name: str, cache: dict[str, Person | None]) -> Person | None:
    key = name.strip().casefold()
    if not key:
        return None
    if key not in cache:
        try:
            found = ctx.resolve(name)
        except Exception:
            found = []
        cache[key] = found[0] if len(found) == 1 else None
    return cache[key]


def _owner_notes(ctx: ProjectsCtx, draft: PI.PlanDraft) -> list[str]:
    cache: dict[str, Person | None] = {}
    notes: list[str] = []
    seen: set[str] = set()
    blank = 0
    for s in draft.steps:
        if not s.owner.strip():
            blank += 1
        elif _resolve_owner(ctx, s.owner, cache) is None and s.owner.casefold() not in seen:
            seen.add(s.owner.casefold())
            notes.append(t("plan_note_owner", ctx.lang, n=s.n, name=s.owner))
    if blank:
        notes.append(t("plan_note_noowner", ctx.lang, n=blank))
    return notes


def _target_project(ctx: ProjectsCtx, fragment: str, draft: PI.PlanDraft) -> tuple[Project | None, str, dict | None]:  # fmt: skip
    """(existing project or None, name for a new one, error)."""
    frag = (fragment or "").strip()
    if frag:
        found = find_projects(_readable(ctx), frag)
        if len(found) == 1:
            return found[0], found[0].name, None
        if len(found) > 1:
            p, bad = _pick(ctx, frag)  # builds the "which one?" answer
            return p, frag, bad
        return None, frag, None
    name = draft.project_name.strip()
    if not name:
        return None, "", _err(ctx, "name_required", "pm_need_name")
    for p in _readable(ctx):
        if p.name.casefold() == name.casefold() and p.status not in DONE_STATUSES:
            return p, p.name, None
    return None, name, None


def apply_plan(ctx: ProjectsCtx, draft: PI.PlanDraft, fragment: str) -> dict:
    """Create / update the project, milestones and tasks of a confirmed plan."""
    today = _today(ctx)
    project, name, bad = _target_project(ctx, fragment, draft)
    if bad:
        return bad
    dues = [d for d in [*(s.due for s in draft.steps), *(p.end for p in draft.phases)] if d]
    if project is None:
        project = Project(
            name=name, department=guess_department(name) or "other",  # type: ignore[arg-type]
            owner=ctx.asker, priority="medium", status="in_progress",
            deadline=max(dues, default=""), visibility="team", created_by=ctx.asker,
            workspace=ctx.workspace,
        )  # fmt: skip
        project = ctx.repo.create(ctx.guild_id, apply_template(project))
    elif not can_contribute(project, ctx.asker, ctx.is_admin):
        return _err(ctx, "forbidden", "plan_forbidden", name=project.name)

    # phases -> milestones (due = the phase's end)
    mid_of: dict[str, str] = {}
    for ph in draft.phases:
        hit = next(
            (m for m in project.milestones if m.title.casefold() == ph.title.casefold()), None
        )
        if hit is None:
            hit = Milestone(id=next_item_id(project.milestones, "m"), title=ph.title, due=ph.end)
            project.milestones.append(hit)
        elif ph.end:
            hit.due = ph.end
        mid_of[ph.title.casefold()] = hit.id
    ctx.repo.save(ctx.guild_id, project)

    existing = {
        PI._norm_title(x.title): x
        for x in ctx.tasks.list(ctx.guild_id, status=None, project_id=project.id)
    }
    cache: dict[str, Person | None] = {}
    asker_id = _asker_id(ctx)
    task_of: dict[int, Task] = {}
    new: list[Task] = []
    updated: list[Task] = []
    unresolved: list[str] = []
    for s in draft.steps:
        mid = mid_of.get(s.phase.casefold(), "") if s.phase else ""
        task = existing.pop(PI._norm_title(s.title), None)
        who = _resolve_owner(ctx, s.owner, cache)
        if task is None:
            task = make_task(
                ctx.guild_id, project.id, s.title, assignee=who, due=s.due, milestone_id=mid,
                created_from="plan",
            )  # fmt: skip
            if who is None and s.owner.strip():
                task.note = f"owner in plan: {s.owner.strip()}"[:500]
                task.assigner_id, task.assigner_name = asker_id, ctx.asker.name
                unresolved.append(s.owner.strip())
            elif task.assignee_id > 0 and asker_id:
                if task.assignee_id == asker_id:
                    task.confirm = "accepted"
                else:
                    task.assigner_id, task.assigner_name = asker_id, ctx.asker.name
            new.append(task)
        else:
            if s.due:
                task.due = s.due
            if mid:
                task.milestone_id = mid
            if who is not None and not (
                task.assignee_id or task.assignee_email or task.assignee_name
            ):
                apply_assignee(task, who)
            updated.append(task)
        task.start, task.step, task.pct_done = s.start, s.n, s.pct_done
        if s.pct_done >= 100 and task.status != "done":
            task.status, task.done_at = "done", ctx.now().isoformat(timespec="seconds")
        task_of[s.n] = task
    for s in draft.steps:
        task_of[s.n].depends_on = [task_of[d].id for d in s.depends_on if d in task_of]
    for task in [*new, *updated]:
        ctx.tasks.save(task)
    # a card for each task given to someone else, an owner question for each unresolved owner
    cards = [x for x in new if x.status == "open" and x.assigner_id and x.confirm != "accepted"]
    if cards:
        try:
            if ctx.on_assigned_many is not None:
                ctx.on_assigned_many(cards)
            elif ctx.on_assigned is not None:
                for x in cards:
                    ctx.on_assigned(x)
        except Exception:
            logger.warning("task handshake hook failed", exc_info=True)

    project = ctx.repo.get(ctx.guild_id, project.id) or project
    all_tasks = ctx.tasks.list(ctx.guild_id, status=None, project_id=project.id)
    pr = progress(project, all_tasks, today)
    ctx.outbox.append((f"plan-{project.id}.png", project_png(project, all_tasks, ctx.lang, today, prog=pr)))  # fmt: skip
    return {
        "ok": True,
        "id": project.id,
        "attached": True,
        "say": t(
            "plan_created",
            ctx.lang,
            name=project.name,
            new=len(new),
            upd=len(updated),
            ms=len(draft.phases),
        ),  # fmt: skip
        "progress": progress_line(pr, ctx.lang),
        "unresolved_owners": sorted(set(unresolved)),
        "markdown": render_summary(project, all_tasks, ctx.lang, today),
    }


def import_plan(ctx: ProjectsCtx, project: str, file_index: int, text: str, confirm: bool) -> dict:
    key = _asker_key(ctx)
    if confirm:
        loaded = PI.load_draft(ctx.docs, ctx.guild_id, key, ctx.now())
        if loaded is None:
            return _err(ctx, "no_draft", "plan_no_draft")
        draft, stored = loaded
        res = apply_plan(ctx, draft, project or stored)
        if res.get("ok"):
            PI.drop_draft(ctx.docs, ctx.guild_id, key)
        return res
    files = [
        f for f in ctx.files
        if not getattr(f, "skip", "") and (file_index < 0 or getattr(f, "index", -1) == file_index)
    ]  # fmt: skip
    draft = PI.extract_plan(ctx.llm, ctx.model, files, ctx.lang, text, today=_today(ctx))
    if draft.error == "no_input":
        return _err(ctx, "no_input", "plan_no_input")
    if draft.error == "empty":
        return _err(ctx, "empty", "plan_empty")
    if draft.error:
        return _err(ctx, draft.error, "plan_error", why=draft.error)
    PI.save_draft(ctx.docs, ctx.guild_id, key, draft, project, ctx.now())
    return {
        "ok": True,
        "awaiting_confirm": True,
        "steps": len(draft.steps),
        "preview": PI.preview_md(draft, ctx.lang, _today(ctx), _owner_notes(ctx, draft)),
    }


def progress_tools(ctx: ProjectsCtx) -> list:
    """`project_picture` and `project_import_plan` for one request (chat or voice)."""

    def project_picture(project: str = "", person: str = "", what: str = "auto") -> dict:
        """Draw a picture (PNG, attached to your reply automatically) of how far projects really
        are AND, separately, how close their deadlines are. Use it when someone asks what projects
        are running, how a project or a person's work is going, or says "show me" / "ดู" / "ภาพ" /
        "กราฟ" about projects. Also say the numbers in words, in one line per project.

        Args:
            project: project name (or part of it) or id, for one project's picture.
            person: member name or @mention, for "what is X working on" (empty = nobody).
            what: auto (person if given, else project if given, else all running projects) |
                portfolio | person | project.
        """
        try:
            return make_picture(ctx, project, person, what)
        except Exception:
            logger.warning("project picture failed", exc_info=True)
            return _err(ctx, "failed", "pic_failed")

    def project_import_plan(
        project: str = "", file_index: int = -1, text: str = "", confirm: bool = False
    ) -> dict:
        """Turn a plan the person sent (a Gantt chart picture, PDF, Excel / CSV with columns like
        step, task, owner, start, end, status, %, or typed steps in `text`) into a project with
        steps 1-2-3, dates, owners and progress. Always call it first with confirm=False: it
        returns a preview table, show it and ask the person to confirm. Only after they say
        "ยืนยัน" / "confirm" call it again with confirm=True (nothing else needed); it then creates
        the project (or fills the one named in `project`) and attaches the picture. Sending the
        same plan again updates the same steps.

        Args:
            project: existing project name to put the plan into; empty = the plan's own title.
            file_index: the number of the attached file (#N), -1 = every attached file.
            text: the plan typed or pasted in the message, optional.
            confirm: True only after the person confirmed the preview.
        """
        try:
            return import_plan(ctx, project, file_index, text, confirm)
        except Exception:
            logger.warning("plan import failed", exc_info=True)
            return _err(ctx, "failed", "plan_error", why="error")

    tools = [project_picture, project_import_plan]
    for fn in tools:
        fn.outbox = ctx.outbox  # type: ignore[attr-defined]
    return tools


# ------------------------------------------------------------------ Discord glue
class ProgressMixin:
    async def _plan_confirm_command(self, message, guild=None, member=None) -> bool:
        """A bare "ยืนยัน" / "confirm" with a plan draft waiting for this person: create the
        project without asking the model. False when there is no such draft (the message goes on
        as usual)."""
        if not PI.parse_plan_confirm(self._command_text(message)):
            return False
        guild = guild or message.guild
        member = member or message.author
        if guild is None or not self._projects_on(guild.id):
            return False
        key = str(member.id)
        waiting = await asyncio.to_thread(PI.load_draft, self.docs, guild.id, key, self.s.now())
        if waiting is None:
            return False
        lang = self._person_lang(guild.id, member.id, message.content or "")
        ctx = await asyncio.to_thread(
            self._project_ctx, message, guild, member, self.is_exec_member(member), lang, []
        )
        if ctx is None:
            return False
        res = await asyncio.to_thread(import_plan, ctx, "", -1, "", True)
        text = res.get("say") or ""
        if res.get("ok"):
            text = f"{text}\n{res.get('progress', '')}".strip()
        files = [discord.File(io.BytesIO(data), filename=name) for name, data in ctx.outbox[:3]]
        await message.reply(
            text or t("pic_failed", lang), files=files or None, mention_author=False
        )
        return True
