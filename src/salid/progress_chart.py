"""Pictures of project progress for chat (matplotlib Agg only, PNG under 1 MB, Thai/CJK fonts).

Three charts, all showing real progress and time to the deadline as TWO separate things:

* `portfolio_png`: what is running now: one row per active project, a "work done" bar and a
  "time used" bar, days left and the state colour; the worst first (at risk, behind, on track).
* `person_png`: what one person is working on: a block per project with their open tasks as a
  mini-Gantt (start to due, today line, overdue in red, "เหลือ 3 วัน / เลย 2 วัน").
* `project_png`: one project: the two bars and days left on top, then milestones (diamonds) and
  the steps 1-2-3 as bars on a time axis with their done percentage.

Pure functions of already-scoped data: the caller decides who may see which project.
"""

from __future__ import annotations

import io
import logging
from datetime import date, datetime, timedelta

from .i18n import t
from .progress import (
    Progress,
    TaskProgress,
    days_text,
    gap_text,
    ordered_tasks,
    progress,
    project_start,
    state_word,
)
from .projects import Milestone, Project
from .tasks import Task

logger = logging.getLogger(__name__)

MAX_BYTES = 1_000_000
WIDTH_PX = 1200
MAX_PORTFOLIO = 20
MAX_BLOCKS = 8
MAX_TASKS_PER_BLOCK = 6
MAX_STEPS = 30
MAX_MILESTONES = 10
STATE_COLOR = {
    "on_track": "#009E73",
    "behind": "#E69F00",
    "at_risk": "#D55E00",
    "done": "#0072B2",
    "no_deadline": "#7A7A7A",
}
TIME_COLOR = "#8497B0"
TRACK = "#E7EAEE"
OVERDUE = "#C0392B"
INK = "#222222"
MUTED = "#666666"


# ------------------------------------------------------------------ plumbing
def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _created(task: Task) -> date | None:
    try:
        return datetime.fromisoformat(task.created_at).date() if task.created_at else None
    except ValueError:
        return None


def _families(texts: list[str], lang: str) -> list[str]:
    from matplotlib import font_manager

    from .data_studio import chart_font_paths

    families: list[str] = []
    try:
        paths = chart_font_paths(texts, lang)
    except Exception:
        paths = []
    for p in paths:
        try:
            font_manager.fontManager.addfont(p)
            families.append(font_manager.FontProperties(fname=p).get_name())
        except Exception:
            logger.warning("chart font %s not usable", p)
    families.append("DejaVu Sans")
    return families


def _figure(height_px: int):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(WIDTH_PX / 100, max(height_px, 240) / 100), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    return fig


def _rc(families: list[str]):
    import matplotlib

    return matplotlib.rc_context({"font.family": families, "axes.unicode_minus": False})


def _png(fig) -> bytes:
    for dpi in (100, 80, 60):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        if buf.tell() <= MAX_BYTES:
            return buf.getvalue()
    raise ValueError("the chart image is too large")


def _short(d: date) -> str:
    return f"{d.day}/{d.month}"


def _axes(fig, left: float, top_px: int, bottom_px: int, height_px: int, right: float = 0.74):
    """An axes in pixel margins: `left` / `right` are figure fractions."""
    h = height_px
    return fig.add_axes([left, bottom_px / h, right - left, (h - top_px - bottom_px) / h])


def _style(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def _legend(fig, entries: list[tuple[str, str]], x: float, y: float) -> None:
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=c, label=name) for name, c in entries]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(x, y), ncol=len(handles),
        frameon=False, fontsize=11, handlelength=1.2, columnspacing=1.6,
    )  # fmt: skip


def _time_ticks(ax, lo: int, hi: int, today: date) -> None:
    """Date ticks (d/m) on an ordinal x axis, a vertical line for today."""
    n = 6
    step = max(1, (hi - lo) // n)
    ticks = list(range(lo, hi + 1, step))
    ax.set_xticks(ticks, [_short(date.fromordinal(x)) for x in ticks], fontsize=10, color=MUTED)
    ax.grid(axis="x", color="#EEEEEE", linewidth=1)
    ax.set_axisbelow(True)
    ax.xaxis.tick_top()


def _today_line(ax, today: date, label: str, n_rows: float) -> None:
    x = today.toordinal()
    ax.axvline(x, color=OVERDUE, linestyle=(0, (4, 3)), linewidth=1.4, zorder=5)
    ax.text(
        x, n_rows + 0.1, f"{label} {_short(today)}", color=OVERDUE, fontsize=10,
        ha="center", va="top", zorder=6,
    )  # fmt: skip


def _range(dates: list[date], today: date) -> tuple[int, int]:
    pts = [d.toordinal() for d in dates] + [today.toordinal()]
    lo, hi = min(pts), max(pts)
    if hi - lo < 14:
        hi = lo + 14
    pad = max(1, (hi - lo) // 25)
    return lo - pad, hi + max(pad, (hi - lo) // 7)


def _bar(ax, y: float, start: date, end: date, pct: int, color: str, h: float = 0.5) -> None:
    from matplotlib.patches import FancyBboxPatch

    s, e = start.toordinal(), max(end.toordinal(), start.toordinal() + 1)
    ax.add_patch(
        FancyBboxPatch((s, y - h / 2), e - s, h, boxstyle="round,pad=0,rounding_size=0.15",
                       facecolor=color, alpha=0.28, edgecolor="none", mutation_aspect=0.05, zorder=3)
    )  # fmt: skip
    if pct > 0:
        ax.barh(y, (e - s) * pct / 100, left=s, height=h, color=color, zorder=4)


# ------------------------------------------------------------------ 1. what is running
def portfolio_png(
    items: list[tuple[Project, Progress]], lang: str, title: str = "", today: date | None = None
) -> bytes:
    """One row per active project: work done, time used, days left, state. Worst first."""
    order = {"at_risk": 0, "behind": 1, "on_track": 2, "no_deadline": 3, "done": 4}
    rows = sorted(
        items,
        key=lambda it: (order.get(it[1].state, 9), it[1].days_left if it[1].days_left is not None else 9999, it[0].name),
    )[:MAX_PORTFOLIO]  # fmt: skip
    n = len(rows)
    row_h = 58
    h = 150 + row_h * max(n, 1)
    texts = [title, t("prog_lbl_work", lang), t("prog_lbl_time", lang)] + [p.name for p, _ in rows]
    fig = _figure(h)
    with _rc(_families(texts, lang)):
        fig.text(0.02, 1 - 38 / h, title or t("prog_title_running", lang), fontsize=20, color=INK, weight="bold", va="center")  # fmt: skip
        _legend(fig, [(t("prog_lbl_work", lang), STATE_COLOR["on_track"]), (t("prog_lbl_time", lang), TIME_COLOR)], 0.02, 1 - 62 / h)  # fmt: skip
        ax = _axes(fig, 0.27, 100, 24, h, right=0.72)
        _style(ax)
        ax.set_xlim(0, 100)
        ax.set_ylim(max(n, 1), 0)
        ax.set_xticks([])
        ax.set_yticks([])
        if n == 0:
            ax.text(50, 0.5, t("pm_none", lang), ha="center", va="center", fontsize=14, color=MUTED)
        side = ax.get_yaxis_transform()
        for i, (p, pr) in enumerate(rows):
            col = STATE_COLOR.get(pr.state, STATE_COLOR["no_deadline"])
            ax.text(-0.02, i + 0.5, _clip(p.name, 30), transform=side, ha="right", va="center", fontsize=13, color=INK)  # fmt: skip
            for k, (val, c) in enumerate(((pr.done_pct, col), (pr.time_pct, TIME_COLOR))):
                y = i + 0.34 + 0.3 * k
                ax.barh(y, 100, height=0.2, color=TRACK, zorder=2)
                if val:
                    ax.barh(y, val, height=0.2, color=c, zorder=3)
                label = f"{val}%" if val is not None else "-"
                ax.text(min(val or 0, 92) + 1.5, y, label, va="center", fontsize=9, color=MUTED, zorder=4)  # fmt: skip
            ax.text(1.03, i + 0.36, days_text(pr.days_left, lang) if pr.state != "done" else state_word("done", lang), transform=side, va="center", fontsize=12, color=INK)  # fmt: skip
            ax.text(1.03, i + 0.66, state_word(pr.state, lang), transform=side, va="center", fontsize=11, color=col, weight="bold")  # fmt: skip
        return _png(fig)


# ------------------------------------------------------------------ 2. what a person works on
def person_png(
    person: str,
    projects: list[tuple[Project, Progress]],
    tasks: list[Task],
    lang: str,
    today: date | None = None,
) -> bytes:
    """A block per project with the person's open tasks as a mini-Gantt."""
    today = today or date.today()
    names = {p.id: (p, pr) for p, pr in projects}
    by_project: dict[str, list[Task]] = {}
    for x in tasks:
        if x.status == "open":
            by_project.setdefault(x.project_id if x.project_id in names else "", []).append(x)
    order = [pid for pid, _ in names.items() if pid in by_project]
    order += [pid for pid in names if pid not in by_project]  # owned, nothing open of theirs
    if "" in by_project:
        order.append("")
    order = order[:MAX_BLOCKS]

    rows: list[tuple] = []  # (kind, label, payload)
    dates: list[date] = []
    for pid in order:
        pr_tuple = names.get(pid)
        own = ordered_tasks(pr_tuple[0] if pr_tuple else None, by_project.get(pid, []))
        head = pr_tuple[0].name if pr_tuple else t("pm_lbl_no_project", lang)
        rows.append(("head", head, pr_tuple[1] if pr_tuple else None))
        if not own:
            rows.append(("note", t("pm_lbl_none_open", lang), None))
        for x in own[:MAX_TASKS_PER_BLOCK]:
            due = _date(x.due)
            start = _date(x.start) or _created(x) or (due - timedelta(days=7) if due else today)
            if due and start > due:
                start = due - timedelta(days=1)
            rows.append(("task", x.title, (x, start, due)))
            dates += [d for d in (start, due) if d]
        if len(own) > MAX_TASKS_PER_BLOCK:
            rows.append(("note", f"+{len(own) - MAX_TASKS_PER_BLOCK}", None))
    n = len(rows)
    lo, hi = _range(dates, today)
    row_h = 36
    h = 170 + row_h * max(n, 1)
    texts = [person] + [r[1] for r in rows]
    fig = _figure(h)
    with _rc(_families(texts, lang)):
        fig.text(0.02, 1 - 38 / h, _clip(person, 40), fontsize=20, color=INK, weight="bold", va="center")  # fmt: skip
        ax = _axes(fig, 0.30, 96, 20, h, right=0.86)
        _style(ax)
        ax.set_xlim(lo, hi)
        ax.set_ylim(max(n, 1), 0)
        ax.set_yticks([])
        _time_ticks(ax, lo, hi, today)
        side = ax.get_yaxis_transform()
        for i, (kind, label, payload) in enumerate(rows):
            y = i + 0.5
            if kind == "head":
                ax.axhspan(i + 0.05, i + 0.95, color="#F2F5F9", zorder=1)
                ax.text(-0.02, y, _clip(label, 34), transform=side, ha="right", va="center", fontsize=13, color=INK, weight="bold")  # fmt: skip
                if payload is not None:
                    col = STATE_COLOR.get(payload.state, MUTED)
                    txt = f"{state_word(payload.state, lang)}"
                    if payload.done_pct is not None:
                        txt += f" · {t('prog_lbl_work', lang)} {payload.done_pct}%"
                    if payload.time_pct is not None:
                        txt += f" · {t('prog_lbl_time', lang)} {payload.time_pct}%"
                    ax.text(lo + (hi - lo) * 0.01, y, txt, va="center", fontsize=11, color=col, zorder=6)  # fmt: skip
            elif kind == "note":
                ax.text(-0.02, y, label, transform=side, ha="right", va="center", fontsize=11, color=MUTED)  # fmt: skip
            else:
                x, start, due = payload
                late = bool(due and due < today)
                ax.text(-0.02, y, _clip(label, 34), transform=side, ha="right", va="center", fontsize=12, color=OVERDUE if late else INK)  # fmt: skip
                if due:
                    col = OVERDUE if late else STATE_COLOR["on_track"] if x.pct_done else "#0072B2"
                    _bar(ax, y, start, due, x.pct_done, col, h=0.46)
                    ax.text(due.toordinal() + 0.6, y, gap_text((due - today).days, lang), va="center", fontsize=10, color=OVERDUE if late else MUTED, zorder=6)  # fmt: skip
                else:
                    ax.text(today.toordinal() + 0.6, y, "-", va="center", fontsize=10, color=MUTED)
        _today_line(ax, today, t("prog_gap_today", lang), n)
        return _png(fig)


# ------------------------------------------------------------------ 3. one project
def project_png(
    project: Project,
    tasks: list[Task],
    lang: str,
    today: date | None = None,
    milestones: list[Milestone] | None = None,
    prog: Progress | None = None,
) -> bytes:
    """Header with the two bars and days left, then milestones (diamonds) and steps (bars)."""
    today = today or date.today()
    mine = [x for x in tasks if x.project_id == project.id]
    prog = prog or progress(project, mine, today)
    ms = sorted(milestones if milestones is not None else project.milestones, key=lambda m: (m.due or "9999", m.id))[:MAX_MILESTONES]  # fmt: skip
    steps = ordered_tasks(project, mine)[:MAX_STEPS]
    rows: list[tuple] = []  # (kind, label, payload)
    dates: list[date] = []
    start0 = project_start(project, mine)
    for d in (start0, _date(project.deadline)):
        if d:
            dates.append(d)
    numbered = {x.id: row for x, row in zip(ordered_tasks(project, mine), prog.tasks, strict=False)}

    def task_row(x: Task) -> None:
        tp: TaskProgress = numbered[x.id]
        due = _date(x.due)
        start = _date(x.start) or _created(x) or (due - timedelta(days=7) if due else today)
        if due and start > due:
            start = due - timedelta(days=1)
        rows.append(("task", f"{tp.n}. {x.title}", (x, tp, start, due)))
        dates.extend(d for d in (start, due) if d)

    used: set[str] = set()
    for m in ms:
        due = _date(m.due)
        rows.append(("milestone", m.title, (m, due)))
        if due:
            dates.append(due)
        for x in steps:
            if x.milestone_id == m.id:
                task_row(x)
                used.add(x.id)
    rest = [x for x in steps if x.id not in used]
    if rest and ms:
        rows.append(("head", t("prog_lbl_steps", lang), None))
    for x in rest:
        task_row(x)
    n = len(rows)
    lo, hi = _range(dates, today)
    head_h = 190
    row_h = 34
    h = head_h + 60 + row_h * max(n, 1)
    texts = [project.name, t("prog_lbl_work", lang), t("prog_lbl_time", lang)] + [
        r[1] for r in rows
    ]
    col = STATE_COLOR.get(prog.state, STATE_COLOR["no_deadline"])
    fig = _figure(h)
    with _rc(_families(texts, lang)):
        fig.text(0.02, 1 - 34 / h, _clip(project.name, 48), fontsize=21, color=INK, weight="bold", va="center")  # fmt: skip
        fig.text(0.98, 1 - 34 / h, state_word(prog.state, lang), fontsize=16, color=col, weight="bold", ha="right", va="center")  # fmt: skip
        # the two separate bars
        hx = _axes(fig, 0.16, 62, h - 62 - 84, h, right=0.74)
        _style(hx)
        hx.set_xlim(0, 100)
        hx.set_ylim(2, 0)
        hx.set_xticks([])
        hx.set_yticks([])
        side = hx.get_yaxis_transform()
        for k, (label, val, c) in enumerate(
            ((t("prog_lbl_work", lang), prog.done_pct, col), (t("prog_lbl_time", lang), prog.time_pct, TIME_COLOR))
        ):  # fmt: skip
            y = k + 0.5
            hx.barh(y, 100, height=0.5, color=TRACK, zorder=2)
            if val:
                hx.barh(y, val, height=0.5, color=c, zorder=3)
            hx.text(-0.015, y, label, transform=side, ha="right", va="center", fontsize=13, color=INK)  # fmt: skip
            hx.text(min(val or 0, 93) + 1.5, y, f"{val}%" if val is not None else "-", va="center", fontsize=12, color=INK, zorder=4)  # fmt: skip
        hx.text(1.02, 0.5, days_text(prog.days_left, lang) if prog.state != "done" else state_word("done", lang), transform=side, va="center", fontsize=14, color=INK)  # fmt: skip
        if prog.next_due:
            nd = prog.next_due
            hx.text(1.02, 1.5, f"{_clip(nd['title'], 24)} {_short(date.fromisoformat(nd['due']))}", transform=side, va="center", fontsize=10, color=MUTED)  # fmt: skip
        # the plan
        ax = _axes(fig, 0.30, head_h, 20, h, right=0.88)
        _style(ax)
        ax.set_xlim(lo, hi)
        ax.set_ylim(max(n, 1), 0)
        ax.set_yticks([])
        _time_ticks(ax, lo, hi, today)
        side = ax.get_yaxis_transform()
        if n == 0:
            ax.text((lo + hi) / 2, 0.5, t("pm_lbl_none_open", lang), ha="center", va="center", fontsize=13, color=MUTED)  # fmt: skip
        dl = _date(project.deadline)
        if dl:
            ax.axvline(dl.toordinal(), color=INK, linewidth=1.6, zorder=4)
        for i, (kind, label, payload) in enumerate(rows):
            y = i + 0.5
            if kind == "head":
                ax.text(-0.02, y, label, transform=side, ha="right", va="center", fontsize=12, color=MUTED, weight="bold")  # fmt: skip
            elif kind == "milestone":
                m, due = payload
                ax.axhspan(i + 0.05, i + 0.95, color="#F2F5F9", zorder=1)
                ax.text(-0.02, y, _clip(label, 36), transform=side, ha="right", va="center", fontsize=12, color=INK, weight="bold")  # fmt: skip
                if due:
                    c = (
                        STATE_COLOR["done"]
                        if m.status == "done"
                        else (OVERDUE if due < today else "#444444")
                    )
                    ax.scatter([due.toordinal()], [y], marker="D", s=120, color=c, zorder=7)
                    if m.status != "done":
                        ax.text(due.toordinal() + 0.8, y, gap_text((due - today).days, lang), va="center", fontsize=10, color=MUTED, zorder=6)  # fmt: skip
            else:
                x, tp, start, due = payload
                ax.text(-0.02, y, _clip(label, 38), transform=side, ha="right", va="center", fontsize=11.5, color=OVERDUE if tp.overdue else INK)  # fmt: skip
                if due:
                    c = (
                        STATE_COLOR["done"]
                        if tp.done
                        else OVERDUE
                        if tp.overdue
                        else STATE_COLOR["on_track"]
                    )
                    _bar(ax, y, start, due, tp.pct, c, h=0.46)
                    tail = "✓ " if tp.done else ""
                    pct = f"{tp.pct}%" if tp.pct else ""
                    ax.text(due.toordinal() + 0.8, y, f"{tail}{pct}  {gap_text(tp.gap_days, lang)}".strip(), va="center", fontsize=10, color=OVERDUE if tp.overdue else MUTED, zorder=6)  # fmt: skip
        _today_line(ax, today, t("prog_gap_today", lang), n)
        return _png(fig)


def chart_name(kind: str) -> str:
    return f"progress-{kind}.png"
