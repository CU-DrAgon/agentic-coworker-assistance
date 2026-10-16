"""Forms and surveys by DM: the pure logic (storage, state machine, results, files).

Salid DMs each recipient the questions one at a time; the recipient just replies in the DM. A form
is one document (`salid_forms`, kept in the same `DocStore` as tasks: SQLite or Firestore) with a
`recipients` map, one entry per person: state, the index of the question being asked, the answers.
The Discord side (resolving people, sending, the DM hook, timers) is `salid.polls_bot`.

Hard limits live here: at most MAX_RECIPIENTS people, more than BIG_BATCH only for executives /
managers. Results are visible to the creator and to people ranked above the creator, never reverse.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import UTC, datetime, timedelta

from .coo_store import DocStore
from .i18n import t
from .pdf_report import FontSet, _finish, _Pdf
from .secretary_store import new_id, utc_iso

FORMS = "salid_forms"
MAX_RECIPIENTS = 50  # hard limit of people one form may be sent to
BIG_BATCH = 10  # more than this many people: executives / managers only
MAX_QUESTIONS = 10
MAX_QUESTION_CHARS = 300
MAX_ANSWER_CHARS = 1500
DEFAULT_HOURS = 48.0
MAX_HOURS = 24 * 30
MIN_REMINDER_LEAD = timedelta(minutes=10)
MAX_REMINDER_LEAD = timedelta(hours=24)

# recipient states
INVITED, ANSWERING, DONE, DECLINED, UNREACHABLE = (
    "invited",
    "answering",
    "done",
    "declined",
    "unreachable",
)
OPEN_STATES = (INVITED, ANSWERING)

SKIP_RE = re.compile(r"^\s*(?:ข้าม|ขอข้าม|ไม่ตอบข้อนี้|skip|pass|n/?a)\s*[.!ค่ะครับคะ]*\s*$", re.I)
STOP_RE = re.compile(
    r"^\s*(?:ไม่ขอตอบ|ขอไม่ตอบ|ไม่ตอบแล้ว|หยุดถาม|ไม่เข้าร่วม|stop|opt\s*out|no\s+thanks)\s*"
    r"[.!ค่ะครับคะ]*\s*$",
    re.I,
)


def _parse(iso: str) -> datetime:
    return datetime.fromisoformat(iso).astimezone(UTC)


# ---------------------------------------------------------------------------- limits
def recipients_refusal(count: int, is_manager: bool) -> dict | None:
    """The hard recipient limits, decided in code: None when `count` people are fine."""
    if count < 1:
        return {"error": "no_recipients"}
    if count > MAX_RECIPIENTS:
        return {
            "error": "too_many_recipients",
            "max": MAX_RECIPIENTS,
            "reason": f"one form may go to at most {MAX_RECIPIENTS} people; suggest a smaller "
            "group or a channel poll",
        }
    if count > BIG_BATCH and not is_manager:
        return {
            "error": "not_allowed",
            "reason": f"sending a form to more than {BIG_BATCH} people needs an executive or a "
            "server manager; offer to ask one, or to pick fewer people",
        }
    return None


def clean_questions(questions) -> list[str] | dict:
    """Trimmed questions, or an error dict."""
    qs = [str(q).strip() for q in (questions or []) if str(q).strip()]
    if not qs:
        return {"error": "need_questions"}
    if len(qs) > MAX_QUESTIONS:
        return {"error": "too_many_questions", "max": MAX_QUESTIONS}
    if any(len(q) > MAX_QUESTION_CHARS for q in qs):
        return {"error": "question_too_long", "max_chars": MAX_QUESTION_CHARS}
    return qs


def clamp_hours(hours) -> float:
    try:
        h = float(hours)
    except (TypeError, ValueError):
        return DEFAULT_HOURS
    return min(max(h, 1.0), float(MAX_HOURS))


# ---------------------------------------------------------------------------- the form record
def new_form(
    *,
    guild_id: int,
    creator_id: int,
    creator_name: str,
    title: str,
    questions: list[str],
    people: dict[int, str],
    hours: float,
    lang: str,
    now: datetime | None = None,
) -> dict:
    now = (now or datetime.now(UTC)).astimezone(UTC)
    return {
        "id": new_id(),
        "guild_id": guild_id,
        "creator_id": creator_id,
        "creator_name": creator_name,
        "title": title.strip()[:120] or "Form",
        "questions": questions,
        "created_at": utc_iso(now),
        "deadline": utc_iso(now + timedelta(hours=hours)),
        "status": "open",
        "lang": lang,
        "closed_notified": False,
        "recipients": {
            str(uid): {
                "name": name,
                "state": INVITED,
                "idx": 0,
                "answers": [],
                "reminded": False,
                "asked_at": "",
            }
            for uid, name in people.items()
        },
    }


class FormRepo:
    def __init__(self, docs: DocStore):
        self.docs = docs

    def put(self, form: dict) -> dict:
        self.docs.put(FORMS, form["guild_id"], form["id"], form)
        return form

    def get(self, guild_id: int, form_id: str) -> dict | None:
        return self.docs.get(FORMS, guild_id, form_id)

    def all(self, guild_id: int) -> list[dict]:
        return sorted(self.docs.find(FORMS, guild_id), key=lambda f: f["created_at"])

    def find(
        self, guild_id: int, ref: str | None, visible=lambda f: True
    ) -> tuple[dict | None, list[dict]]:
        """(form, candidates): `ref` is an id, a title (partial), or empty / "last" for the newest
        of the forms `visible` allows (the others do not exist for the caller)."""
        forms = [f for f in self.all(guild_id) if visible(f)]
        if not forms:
            return None, []
        key = (ref or "").strip().lstrip("#").casefold()
        if key in ("", "last", "latest", "the last form", "ล่าสุด", "อันล่าสุด"):
            return forms[-1], []
        exact = [f for f in forms if f["id"].casefold() == key]
        if exact:
            return exact[0], []
        hits = [f for f in forms if key in f["title"].casefold()]
        if len(hits) == 1:
            return hits[0], []
        return None, (hits or forms)[-5:]


# ---------------------------------------------------------------------------- state machine
def is_open(form: dict, now: datetime) -> bool:
    return form.get("status") == "open" and _parse(form["deadline"]) > now


def current_question(form: dict, uid: int) -> str | None:
    r = form["recipients"].get(str(uid))
    if r is None or r["state"] not in OPEN_STATES or r["idx"] >= len(form["questions"]):
        return None
    return form["questions"][r["idx"]]


def asking_form(forms: list[dict], uid: int, now: datetime) -> dict | None:
    """The open form whose question `uid` was most recently asked (None when none is open)."""
    best, when = None, ""
    for f in forms:
        r = f["recipients"].get(str(uid))
        if r and r["state"] == ANSWERING and is_open(f, now) and r["asked_at"] >= when:
            best, when = f, r["asked_at"]
    return best


def mark_asked(form: dict, uid: int, now: datetime) -> None:
    r = form["recipients"][str(uid)]
    r["state"] = ANSWERING
    r["asked_at"] = utc_iso(now)


def record_answer(form: dict, uid: int, text: str | None, now: datetime) -> int | None:
    """Store the answer to the current question (None = skipped). Returns the index of the next
    question, or None when the person has answered everything (state -> done)."""
    r = form["recipients"][str(uid)]
    r["answers"] = [*r["answers"][: r["idx"]], (text or "").strip()[:MAX_ANSWER_CHARS] or None]
    r["idx"] += 1
    if r["idx"] >= len(form["questions"]):
        r["state"] = DONE
        return None
    r["asked_at"] = utc_iso(now)
    return r["idx"]


def decline(form: dict, uid: int) -> None:
    form["recipients"][str(uid)]["state"] = DECLINED


def due_reminders(form: dict, now: datetime) -> list[int]:
    """People to nudge once: still in the form, not yet reminded, and the deadline is within the
    reminder lead (a quarter of the form's life, between 10 minutes and 24 hours)."""
    if not is_open(form, now):
        return []
    deadline, created = _parse(form["deadline"]), _parse(form["created_at"])
    lead = min(max((deadline - created) / 4, MIN_REMINDER_LEAD), MAX_REMINDER_LEAD)
    if now < deadline - lead:
        return []
    return [
        int(uid)
        for uid, r in form["recipients"].items()
        if r["state"] in OPEN_STATES and not r["reminded"]
    ]


def progress(form: dict) -> tuple[int, int]:
    """(people who finished, people the form reached or is still waiting for)."""
    rs = form["recipients"].values()
    return sum(r["state"] == DONE for r in rs), sum(r["state"] != UNREACHABLE for r in rs)


def waiting(form: dict) -> list[str]:
    return [r["name"] for r in form["recipients"].values() if r["state"] in OPEN_STATES]


# ---------------------------------------------------------------------------- visibility
def can_view_results(viewer_id: int, form: dict, ranks: dict[int, int]) -> bool:
    """The creator, and people ranked strictly above the creator. Never reverse, never peers."""
    if viewer_id == form["creator_id"]:
        return True
    return ranks.get(viewer_id, -1) > ranks.get(form["creator_id"], 10**9)


# ---------------------------------------------------------------------------- results
STATE_TH = {INVITED: "ยังไม่ตอบ", ANSWERING: "กำลังตอบ", DONE: "ตอบแล้ว", DECLINED: "ปฏิเสธ"}
STATE_EN = {INVITED: "no reply", ANSWERING: "answering", DONE: "done", DECLINED: "declined"}


def state_label(state: str, lang: str) -> str:
    if state == UNREACHABLE:
        return "DM ปิด" if lang != "en" else "DMs closed"
    return (STATE_EN if lang == "en" else STATE_TH).get(state, state)


def rows(form: dict, lang: str) -> list[list[str]]:
    """One row per person: name, state, then one cell per question."""
    n = len(form["questions"])
    out = []
    for r in form["recipients"].values():
        answers = [*r["answers"], *[None] * (n - len(r["answers"]))]
        out.append([r["name"], state_label(r["state"], lang), *[a or "" for a in answers[:n]]])
    return sorted(out, key=lambda x: x[0].casefold())


def counts_line(form: dict, lang: str) -> str:
    done, reached = progress(form)
    return t("form_progress", lang, title=form["title"], done=done, total=reached)


def text_table(form: dict, lang: str) -> str:
    """Per-person table as plain text (for a DM to the viewer)."""
    lines = [counts_line(form, lang)]
    for row in rows(form, lang):
        head = f"- {row[0]} ({row[1]})"
        qa = [f"    {i + 1}. {form['questions'][i]}: {a}" for i, a in enumerate(row[2:]) if a]
        lines.append("\n".join([head, *qa]))
    return "\n".join(lines)


def csv_bytes(form: dict, lang: str) -> bytes:
    """UTF-8 with BOM so Excel opens Thai text correctly."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [t("form_col_name", lang), t("form_col_state", lang), *form["questions"]],
    )
    w.writerows(rows(form, lang))
    return buf.getvalue().encode("utf-8-sig")


def form_pdf(form: dict, *, org: str, lang: str, generated_at: datetime, fonts: FontSet) -> bytes:
    pdf = _Pdf(fonts, org, lang, generated_at)
    pdf.add_page()
    pdf.h1(form["title"])
    pdf.p(counts_line(form, lang))
    pdf.h2(t("form_pdf_people", lang))
    for row in rows(form, lang):
        pdf.p(f"{row[0]} - {row[1]}", bold=True)
        for i, a in enumerate(row[2:]):
            if a:
                pdf.p(f"  {i + 1}. {form['questions'][i]}\n     {a}", size=9)
    return _finish(pdf)


def summary_for_agent(form: dict, lang: str, *, with_answers: bool) -> dict:
    """What the form_results tool hands the model. Answers only when `with_answers` (a private
    place); otherwise only counts, so nothing leaks into a public channel."""
    done, reached = progress(form)
    out: dict = {
        "id": form["id"],
        "title": form["title"],
        "status": form["status"],
        "deadline": form["deadline"],
        "questions": form["questions"],
        "finished": done,
        "reached": reached,
        "waiting": waiting(form),
        "declined": [r["name"] for r in form["recipients"].values() if r["state"] == DECLINED],
        "dms_closed": [r["name"] for r in form["recipients"].values() if r["state"] == UNREACHABLE],
    }
    if with_answers:
        out["people"] = [{"name": r[0], "state": r[1], "answers": r[2:]} for r in rows(form, lang)]
    return out


# ---------------------------------------------------------------------------- the answer gate
def answer_gate_prompt(question: str, reply: str) -> str:
    return (
        "Salid, a team assistant in Discord, sent a person one survey question by private "
        "message. Decide whether the person's reply is an ANSWER to that question (even a short, "
        "informal, partial or negative one such as 'no', 'maybe Friday', 'A') or whether they are "
        "doing something else: asking Salid for something unrelated, asking a question back, "
        "chatting, or giving an instruction.\n"
        f"Question: {question[:MAX_QUESTION_CHARS]}\nReply: {reply[:600]}\n"
        'Reply JSON: {"answer": true|false} (true = the reply answers the question).'
    )


def looks_like_answer(client, model: str, question: str, reply: str) -> bool:
    """Cheap one-call gate (blocking; run it in a thread). Any failure means "not an answer", so
    the message falls through to normal chat."""
    from . import gate  # lazy: gate imports the Gemini client

    try:
        return gate._decide(client, model, answer_gate_prompt(question, reply), gate._Yes).answer
    except Exception:
        gate.logger.warning("gate (form answer) failed; treating as chat", exc_info=True)
        return False
