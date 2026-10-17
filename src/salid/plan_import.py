"""A Gantt chart / plan sent to Salid becomes a project with steps, dates and progress.

`extract_plan(llm, model, files, lang, text=...)` makes ONE cheap-model call over what was sent (a
picture of a Gantt or timeline, a PDF, an xlsx / csv with columns like step, task, owner, start,
end, status, %, or pasted text) and returns a `PlanDraft`: the project name, phases and numbered
steps. Everything the model says is checked in code (dates `YYYY-MM-DD`, Buddhist years turned into
Gregorian, "สัปดาห์ที่ 3" counted from the plan start when there is one and flagged when there is not,
unknown dependencies dropped, percentages clamped) and it fails soft with `error`.

The tool `project_import_plan` works in two turns so nothing is created by surprise: the first call
shows a preview table and keeps the draft (DocStore `salid_plan_drafts`, one per person, 1 hour);
"ยืนยัน" / "confirm" (the tool with confirm=True, or the typed command) then creates the project
(or uses the named one), phases as milestones and steps as tasks (`Task.start` `step` `depends_on`
`pct_done`). Importing the same plan again updates the steps with the same title.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from .i18n import CATALOG, t
from .llm import generate
from .progress import gap_text

logger = logging.getLogger(__name__)

PLAN_DRAFTS = "salid_plan_drafts"  # DocStore kind: key = asker id, one draft each
DRAFT_TTL = timedelta(hours=1)
MAX_STEPS = 60
MAX_PHASES = 20
PREVIEW_ROWS = 30
MAX_TEXT_CHARS = 30_000
MAX_FILE_BYTES = 8_000_000
_IMAGE_PREFIX = "image/"
_PDF = "application/pdf"

STRINGS = {
    "plan_no_input": {
        "th": "ส่งไฟล์แผนงาน (รูป Gantt, PDF, Excel/CSV) หรือพิมพ์รายการงานมาด้วยนะคะ แล้วสลิดจะแปลงเป็นโปรเจกต์ให้",
        "en": "Send the plan (a Gantt picture, PDF, Excel/CSV) or type the list of steps and I will turn it into a project.",
    },
    "plan_error": {
        "th": "อ่านแผนงานนี้ไม่ได้ค่ะ ({why}) ลองส่งภาพที่ชัดขึ้นหรือเป็น Excel ดูนะคะ",
        "en": "I could not read this plan ({why}). Try a clearer picture or an Excel file.",
    },
    "plan_empty": {
        "th": "ไม่เจอรายการงานในแผนนี้ค่ะ",
        "en": "I found no steps in this plan.",
    },
    "plan_no_draft": {
        "th": "ไม่มีแผนที่รอยืนยันค่ะ (หรือเกิน 1 ชั่วโมงแล้ว) ส่งแผนมาใหม่ได้เลย",
        "en": "No plan is waiting for confirmation (or it is over an hour old). Send the plan again.",
    },
    "plan_preview_title": {
        "th": "แผนงาน “{name}”: อ่านได้ {steps} ขั้นตอน, {phases} ช่วง",
        "en": "Plan “{name}”: {steps} steps, {phases} phases",
    },
    "plan_hdr": {
        "th": "| ขั้น | งาน | ผู้รับผิดชอบ | เริ่ม | กำหนดส่ง | เหลือ | % |",
        "en": "| Step | Task | Owner | Start | Due | Gap | % |",
    },
    "plan_more": {"th": "…และอีก {n} ขั้น", "en": "…and {n} more steps"},
    "plan_unreadable": {"th": "อ่านไม่ได้ / ไม่แน่ใจ", "en": "Could not read / unsure"},
    "plan_confirm": {
        "th": "ถ้าถูกต้อง ตอบ “ยืนยัน” แล้วสลิดจะสร้างโปรเจกต์และงานให้ค่ะ (แก้ได้ก่อน: บอกว่าจะเปลี่ยนอะไร)",
        "en": "If this looks right, reply “confirm” and I will create the project and tasks.",
    },
    "plan_note_weeks": {
        "th": "ขั้น {n} บอกเป็น “สัปดาห์ที่” แต่ไม่รู้วันเริ่มแผน เลยยังไม่ใส่วันที่ (บอกวันเริ่มได้ค่ะ)",
        "en": "Step {n} is given in weeks but the plan start is unknown, so its dates are empty (tell me the start date).",
    },
    "plan_note_date": {
        "th": "ขั้น {n}: วันที่ “{value}” อ่านไม่ได้ จึงเว้นว่าง",
        "en": "Step {n}: the date “{value}” could not be read, left empty.",
    },
    "plan_note_order": {
        "th": "ขั้น {n}: วันเริ่มอยู่หลังกำหนดส่ง จึงตัดวันเริ่มออก",
        "en": "Step {n}: the start is after the due date, start dropped.",
    },
    "plan_note_owner": {
        "th": "ขั้น {n}: ไม่พบสมาชิกชื่อ “{name}” จะถามว่าใครรับ",
        "en": "Step {n}: no member called “{name}”, I will ask who takes it.",
    },
    "plan_note_noowner": {
        "th": "{n} ขั้นยังไม่ระบุผู้รับผิดชอบ",
        "en": "{n} steps have no owner yet.",
    },
    "plan_forbidden": {
        "th": "ขอโทษค่ะ ต้องเป็นสมาชิกโปรเจกต์ “{name}” ถึงจะใส่แผนเข้าไปได้",
        "en": "Sorry, you need to be on project “{name}” to put a plan into it.",
    },
    "plan_created": {
        "th": "นำเข้าแผนใน “{name}” แล้วค่ะ: งานใหม่ {new}, อัปเดต {upd}, ช่วง/เหตุการณ์สำคัญ {ms}",
        "en": "Imported the plan into “{name}”: {new} new tasks, {upd} updated, {ms} milestones.",
    },
}
CATALOG.update(STRINGS)

_CONFIRM = re.compile(
    r"^(ยืนยัน(แผน)?|ตกลง|ใช่|ok|okay|yes|confirm( plan)?|go ahead)\s*(ค่ะ|คะ|ครับ|นะ|เลย|please|!)*$",
    re.I,
)


def parse_plan_confirm(text: str | None) -> bool:
    """A bare "ยืนยัน" / "confirm" (the caller checks that a draft is waiting)."""
    return bool(_CONFIRM.match((text or "").strip().strip("!.。 ")))


# ------------------------------------------------------------------ models
class PlanPhase(BaseModel):
    title: str
    start: str = ""
    end: str = ""


class PlanStep(BaseModel):
    n: int
    title: str
    phase: str = ""
    owner: str = ""
    start: str = ""
    due: str = ""
    pct_done: int = 0
    depends_on: list[int] = Field(default_factory=list)


class PlanDraft(BaseModel):
    project_name: str = ""
    plan_start: str = ""
    phases: list[PlanPhase] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # what could not be read (already in words)
    error: str = ""


class _RawPhase(BaseModel):
    title: str = ""
    start: str = ""
    end: str = ""


class _RawStep(BaseModel):
    n: int = 0
    title: str = ""
    phase: str = ""
    owner: str = ""
    start: str = ""
    due: str = ""
    start_week: int = 0
    due_week: int = 0
    pct_done: int = 0
    depends_on: list[int] = Field(default_factory=list)


class _RawPlan(BaseModel):
    project_name: str = ""
    plan_start: str = ""
    phases: list[_RawPhase] = Field(default_factory=list)
    steps: list[_RawStep] = Field(default_factory=list)
    unreadable: list[str] = Field(default_factory=list)


PROMPT = """You read project plans: a Gantt chart or timeline (picture or PDF), a spreadsheet with
columns such as step / task / owner / start / end / due / status / %, or plain text. Return ONLY JSON
for the plan in the input. Today is {today}. Reply language of any free text: {language}.

Fields:
- project_name: the plan's own title, "" if none.
- plan_start: the first day of the plan as YYYY-MM-DD when the plan states it, else "".
- phases: the bands / groups / milestones of the plan [{{title, start, end}}], dates YYYY-MM-DD or "".
- steps: every task row in the order of the plan [{{n, title, phase, owner, start, due, start_week,
  due_week, pct_done, depends_on}}]. n = the step number written in the plan, else 1, 2, 3 in order.
  start / due as YYYY-MM-DD ("due" is the end of the bar). Convert Buddhist-era years (2569 -> 2026) and
  day-first dates (17/10/2026 = 17 Oct). If the plan only says a week ("สัปดาห์ที่ 3", "week 3",
  "W3"), put the week number in start_week / due_week and leave the dates "". pct_done 0-100 from a
  % or a status (done = 100, in progress without a number = 0). depends_on = the n of the steps it
  waits for (arrows, "after step 2"), [] if none. owner = the name written, "" if none.
- unreadable: short notes about parts you could not read or were unsure of.
Never invent steps, names or dates that are not in the input. At most {max_steps} steps.
"""


# ------------------------------------------------------------------ pure helpers
_DMY = re.compile(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})$")
_YMD = re.compile(r"^(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})$")


def norm_date(value: str | None) -> str:
    """YYYY-MM-DD from "2026-10-17", "17/10/2026", "2569-10-17" (Buddhist year); "" if unreadable."""
    s = (value or "").strip()[:10].strip()
    if not s:
        return ""
    m = _YMD.match(s)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
    else:
        m = _DMY.match(s)
        if not m:
            return ""
        d, mo, y = int(m[1]), int(m[2]), int(m[3])
    if y > 2400:  # Buddhist era
        y -= 543
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return ""


def _week_start(plan_start: str, week: int) -> str:
    return (date.fromisoformat(plan_start) + timedelta(days=7 * (week - 1))).isoformat()


def _week_end(plan_start: str, week: int) -> str:
    return (date.fromisoformat(plan_start) + timedelta(days=7 * week - 1)).isoformat()


def clean_plan(raw: _RawPlan, lang: str) -> PlanDraft:
    """The model's plan, checked: real dates, week numbers resolved when the plan start is known,
    steps numbered 1.. without duplicates, dependencies that exist, percentages 0-100."""
    notes = [x.strip()[:200] for x in raw.unreadable if x and x.strip()][:6]
    plan_start = norm_date(raw.plan_start)
    steps: list[PlanStep] = []
    seen: set[int] = set()
    next_n = 1
    flagged_weeks: list[int] = []
    for rs in raw.steps[:MAX_STEPS]:
        title = " ".join(rs.title.split())[:200]
        if not title:
            continue
        n = rs.n if rs.n > 0 and rs.n not in seen else 0
        if not n:
            while next_n in seen:
                next_n += 1
            n = next_n
        seen.add(n)
        start, due = norm_date(rs.start), norm_date(rs.due)
        for _label, got, value in (("start", start, rs.start), ("due", due, rs.due)):
            if value.strip() and not got:
                notes.append(t("plan_note_date", lang, n=n, value=value.strip()[:20]))
        if not start and rs.start_week > 0 and plan_start:
            start = _week_start(plan_start, rs.start_week)
        if not due and rs.due_week > 0 and plan_start:
            due = _week_end(plan_start, rs.due_week)
        if not due and not start and rs.start_week > 0 and plan_start:
            due = _week_end(plan_start, rs.start_week)
        if (rs.start_week > 0 or rs.due_week > 0) and not plan_start and not (start and due):
            flagged_weeks.append(n)
        if start and due and start > due:
            notes.append(t("plan_note_order", lang, n=n))
            start = ""
        steps.append(
            PlanStep(
                n=n, title=title, phase=" ".join(rs.phase.split())[:120],
                owner=" ".join(rs.owner.split())[:80], start=start, due=due,
                pct_done=max(0, min(100, int(rs.pct_done))), depends_on=list(rs.depends_on),
            )
        )  # fmt: skip
    for n in flagged_weeks[:5]:
        notes.append(t("plan_note_weeks", lang, n=n))
    have = {s.n for s in steps}
    for s in steps:
        s.depends_on = sorted({d for d in s.depends_on if d in have and d != s.n})
    steps.sort(key=lambda s: s.n)
    phases = []
    for rp in raw.phases[:MAX_PHASES]:
        title = " ".join(rp.title.split())[:120]
        if title:
            phases.append(PlanPhase(title=title, start=norm_date(rp.start), end=norm_date(rp.end)))
    for ph in phases:  # a phase without dates takes the span of its steps
        mine = [s for s in steps if s.phase.casefold() == ph.title.casefold()]
        if not ph.end:
            ph.end = max((s.due for s in mine if s.due), default="")
        if not ph.start:
            ph.start = min((s.start for s in mine if s.start), default="")
    return PlanDraft(
        project_name=" ".join(raw.project_name.split())[:120],
        plan_start=plan_start, phases=phases, steps=steps, notes=notes[:10],
    )  # fmt: skip


# ------------------------------------------------------------------ reading the input
def _parts_of(files: list[Any], text: str) -> list[Any]:
    """The model input: pictures and PDFs as bytes, spreadsheets / documents as text."""
    parts: list[Any] = []
    chunks: list[str] = []
    for f in files:
        if getattr(f, "skip", ""):
            continue
        mime = (getattr(f, "mime", "") or "").lower()
        data = getattr(f, "data", b"") or b""
        if mime.startswith(_IMAGE_PREFIX) or mime == _PDF:
            if data and len(data) <= MAX_FILE_BYTES:
                parts.append(types.Part.from_bytes(data=data, mime_type=mime))
            continue
        body = data.decode("utf-8", "replace") if data else ""
        raw = getattr(f, "raw", b"") or b""
        if not body.strip() and raw:
            try:
                from . import docs

                _, body, _ = docs.extract_document(raw, getattr(f, "filename", ""), None)
            except Exception:
                logger.warning("plan file not readable as text", exc_info=True)
        if body.strip():
            chunks.append(f"[{getattr(f, 'filename', 'file')}]\n{body[:MAX_TEXT_CHARS]}")
    if (text or "").strip():
        chunks.append(text.strip()[:MAX_TEXT_CHARS])
    if chunks:
        parts.append("\n\n".join(chunks))
    return parts


def extract_plan(
    llm: Any,
    model: str,
    files: list[Any],
    lang: str,
    text: str = "",
    today: date | None = None,
) -> PlanDraft:
    """One cheap-model call over the pictures / documents / text. Never raises: `error` is set."""
    from .i18n import lang_name

    parts = _parts_of(files, text)
    if not parts:
        return PlanDraft(error="no_input")
    prompt = PROMPT.format(
        today=(today or date.today()).isoformat(), language=lang_name(lang), max_steps=MAX_STEPS
    )
    try:
        resp = generate(
            llm, model, [*parts, prompt],
            types.GenerateContentConfig(
                temperature=0, max_output_tokens=8000,
                response_mime_type="application/json", response_schema=_RawPlan,
            ),
        )  # fmt: skip
        raw = _RawPlan.model_validate_json(resp.text)
    except (ValidationError, ValueError) as e:
        logger.warning("plan reply not valid (%s)", type(e).__name__)
        return PlanDraft(error="unreadable")
    except Exception as e:
        logger.warning("plan extraction failed (%s)", type(e).__name__)
        return PlanDraft(error="unavailable")
    draft = clean_plan(raw, lang)
    if not draft.steps:
        draft.error = "empty"
    return draft


# ------------------------------------------------------------------ preview
def preview_md(
    draft: PlanDraft, lang: str, today: date, owners_note: list[str] | None = None
) -> str:
    """The table the person confirms: at most 30 rows, then what could not be read."""
    lines = [
        t("plan_preview_title", lang, name=draft.project_name or "-", steps=len(draft.steps), phases=len(draft.phases)),
        "",
        t("plan_hdr", lang),
        "|" + "---|" * 7,
    ]  # fmt: skip
    for s in draft.steps[:PREVIEW_ROWS]:
        gap = gap_text((date.fromisoformat(s.due) - today).days, lang) if s.due else ""
        cell = lambda v: (v or "-").replace("|", "/")  # noqa: E731
        lines.append(
            f"| {s.n} | {cell(s.title)[:60]} | {cell(s.owner)[:20]} | {s.start or '-'} | "
            f"{s.due or '-'} | {gap or '-'} | {s.pct_done}% |"
        )
    if len(draft.steps) > PREVIEW_ROWS:
        lines.append(t("plan_more", lang, n=len(draft.steps) - PREVIEW_ROWS))
    notes = [*draft.notes, *(owners_note or [])]
    if notes:
        lines += ["", t("plan_unreadable", lang) + ":"] + [f"- {n}" for n in notes[:10]]
    lines += ["", t("plan_confirm", lang)]
    return "\n".join(lines)


# ------------------------------------------------------------------ draft store
def save_draft(
    docs: Any, guild_id: int, asker_key: str, draft: PlanDraft, project: str, now: datetime
) -> None:
    docs.put(
        PLAN_DRAFTS, guild_id, asker_key,
        {"at": now.isoformat(timespec="seconds"), "project": project, "plan": draft.model_dump()},
    )  # fmt: skip


def load_draft(
    docs: Any, guild_id: int, asker_key: str, now: datetime
) -> tuple[PlanDraft, str] | None:
    """The waiting draft and the project name it was for; None when there is none or it is over
    an hour old (an old one is deleted)."""
    doc = docs.get(PLAN_DRAFTS, guild_id, asker_key)
    if not doc:
        return None
    try:
        at = datetime.fromisoformat(doc["at"])
        if at.tzinfo is None and now.tzinfo is not None:
            at = at.replace(tzinfo=now.tzinfo)
        if now - at > DRAFT_TTL:
            docs.delete(PLAN_DRAFTS, guild_id, asker_key)
            return None
        return PlanDraft.model_validate(doc["plan"]), str(doc.get("project", ""))
    except (KeyError, ValueError, ValidationError):
        docs.delete(PLAN_DRAFTS, guild_id, asker_key)
        return None


def drop_draft(docs: Any, guild_id: int, asker_key: str) -> None:
    docs.delete(PLAN_DRAFTS, guild_id, asker_key)


def _norm_title(s: str) -> str:
    return " ".join((s or "").casefold().split())
