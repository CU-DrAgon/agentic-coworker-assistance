"""Task handshake, pure part: the texts and decisions (no Discord, no network).

A task Salid hears in chat is sent to the person who has to do it, as a card with three buttons
(accept / edit / not mine). When they answer, the task is recorded and the person who gave it is
told. `task_handshake_bot.py` binds this to Discord; the card state lives on the `Task`
(`confirm`, `confirm_at`, `handshake_message`, `assigner_*`).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .i18n import CATALOG, t
from .memory_access import is_dm_scope
from .tasks import Task, parse_due

MAX_CARDS = 5  # one message holds 5 rows of buttons (Discord: 25 components, 3 used per card)
NUDGE_AFTER = timedelta(hours=24)
PENDING, ACCEPTED, DECLINED = "pending", "accepted", "declined"
AWAITING = "awaiting_approval"  # approve mode: the assignee accepted, the giver must sign off
ANSWERED = (ACCEPTED, DECLINED, AWAITING)
MAX_ROUNDS = 3  # send-backs before the task stays pending and an executive is told
TEXT_MAX = 1900

STRINGS = {
    "hs_card": {
        "th": "📋 งานใหม่ จาก #{channel} ({giver}): **{title}** · กำหนด {due} · โปรเจกต์ {project}",
        "en": "📋 New task from #{channel} ({giver}): **{title}** · due {due} · project {project}",
    },
    "hs_giver_unknown": {"th": "จากบทสนทนา", "en": "from the conversation"},
    "hs_btn_ok": {"th": "✅ รับงาน", "en": "✅ Accept"},
    "hs_btn_edit": {"th": "✏️ แก้ไข", "en": "✏️ Edit"},
    "hs_btn_no": {"th": "❌ ไม่ใช่ของฉัน", "en": "❌ Not mine"},
    "hs_status_ok": {"th": "✅ รับงานแล้ว", "en": "✅ Accepted"},
    "hs_status_edit": {"th": "✏️ รับงานแล้ว (แก้ไขแล้ว)", "en": "✏️ Accepted with edits"},
    "hs_status_no": {"th": "❌ ไม่ใช่ของฉัน", "en": "❌ Not mine"},
    "hs_giver_ok": {
        "th": "✅ {name} รับงาน “{title}” แล้ว (กำหนด {due}) · งาน #{id}",
        "en": "✅ {name} accepted “{title}” (due {due}) · task #{id}",
    },
    "hs_giver_edit": {
        "th": "✏️ {name} รับงานแล้ว และแก้เป็น {changes} · งาน #{id}",
        "en": "✏️ {name} accepted the task and changed it to {changes} · task #{id}",
    },
    "hs_giver_no": {
        "th": "❌ {name} บอกว่างาน “{title}” ไม่ใช่ของเขา — ช่วยระบุผู้รับผิดชอบใหม่ (บอกสลิดได้เลย เช่น “งาน #{id} ให้ …”)",
        "en": "❌ {name} says “{title}” is not theirs — please name who should take it (tell Salid, for example “task #{id} to …”)",
    },
    "hs_count": {"th": " ({n}/{total} ยืนยันแล้ว)", "en": " ({n}/{total} confirmed)"},
    "hs_team_unclear": {
        "th": "งาน “{title}” ระบุหลายคน/ทีม ช่วยบอกสลิดว่าใครเป็นเจ้าของหลัก (งาน #{id} ให้ …)",
        "en": "“{title}” names several people or a team — please tell Salid who the main owner is (task #{id} to …)",
    },
    "hs_change_title": {"th": "ชื่องาน “{v}”", "en": "title “{v}”"},
    "hs_change_due": {"th": "กำหนด {v}", "en": "due {v}"},
    "hs_change_note": {"th": "โน้ต “{v}”", "en": "note “{v}”"},
    "hs_not_yours": {
        "th": "งานนี้ไม่ใช่ของคุณค่ะ ให้เจ้าของงานกดนะคะ",
        "en": "This task is not yours. Its assignee has to answer.",
    },
    "hs_handled": {
        "th": "งานนี้ตอบไปแล้วค่ะ",
        "en": "This task has already been answered.",
    },
    "hs_gone": {"th": "ไม่พบงานนี้แล้วค่ะ", "en": "That task no longer exists."},
    "hs_modal_title": {"th": "แก้งานก่อนรับ", "en": "Edit, then accept"},
    "hs_f_title": {"th": "ชื่องาน", "en": "Task"},
    "hs_f_due": {
        "th": "กำหนดส่ง (เช่น ศุกร์, 2026-10-30)",
        "en": "Due (for example friday, 2026-10-30)",
    },
    "hs_f_note": {"th": "โน้ต", "en": "Note"},
    "hs_bad_due": {
        "th": "อ่านวันที่ไม่ออก เลยคงกำหนดเดิมไว้ค่ะ",
        "en": "I could not read that date, so the due date stays as it was.",
    },
    "hs_pending_marker": {"th": "⏳ รอยืนยัน", "en": "⏳ awaiting confirmation"},
    "hs_awaiting_marker": {"th": "⏳ รออนุมัติ", "en": "⏳ awaiting approval"},
    "hs_status_wait": {"th": "⏳ รับงานแล้ว รอหัวหน้าอนุมัติ", "en": "⏳ Accepted, waiting for approval"},
    "hs_back_line": {"th": "↩️ หัวหน้าส่งกลับ: {note}", "en": "↩️ Sent back: {note}"},
    "hs_btn_ap": {"th": "✅ อนุมัติ", "en": "✅ Approve"},
    "hs_btn_ae": {"th": "✏️ แก้ไข", "en": "✏️ Edit"},
    "hs_btn_sb": {"th": "↩️ ส่งกลับ", "en": "↩️ Send back"},
    "hs_boss_card": {
        "th": "📋 {name} รับงาน “{title}” (กำหนด {due}{edits}) · งาน #{id}",
        "en": "📋 {name} accepted “{title}” (due {due}{edits}) · task #{id}",
    },
    "hs_boss_edits": {"th": ", แก้จาก: {diff}", "en": ", changed from: {diff}"},
    "hs_diff_title": {"th": "ชื่อ “{a}” → “{b}”", "en": "title “{a}” → “{b}”"},
    "hs_diff_due": {"th": "กำหนด {a} → {b}", "en": "due {a} → {b}"},
    "hs_diff_note": {"th": "โน้ต “{a}” → “{b}”", "en": "note “{a}” → “{b}”"},
    "hs_boss_ok": {"th": "✅ อนุมัติแล้ว", "en": "✅ Approved"},
    "hs_boss_edited": {"th": "✏️ แก้ไขและอนุมัติแล้ว", "en": "✏️ Edited and approved"},
    "hs_boss_back": {"th": "↩️ ส่งกลับแล้ว", "en": "↩️ Sent back"},
    "hs_to_assignee_ok": {
        "th": "✅ หัวหน้าอนุมัติแล้ว: “{title}” (กำหนด {due}) · งาน #{id}",
        "en": "✅ Approved by your boss: “{title}” (due {due}) · task #{id}",
    },
    "hs_to_assignee_edit": {
        "th": "✏️ หัวหน้าแก้เป็น {changes} และอนุมัติแล้ว · งาน #{id}",
        "en": "✏️ Your boss changed it to {changes} and approved it · task #{id}",
    },
    "hs_modal_boss": {"th": "แก้งานก่อนอนุมัติ", "en": "Edit, then approve"},
    "hs_modal_back": {"th": "ส่งกลับให้ผู้รับงาน", "en": "Send back"},
    "hs_f_back": {"th": "บอกผู้รับงานว่าให้แก้อะไร", "en": "What should they change?"},
    "hs_back_cap": {
        "th": "ส่งกลับครบ {n} รอบแล้ว งานนี้ค้างรอยืนยัน ผู้บริหารจะได้รับแจ้งค่ะ",
        "en": "Sent back {n} times already; the task stays pending and an executive is told.",
    },
    "hs_back_cap_exec": {
        "th": "งาน “{title}” (#{id}) ถูกส่งกลับ {n} รอบแล้วยังไม่ลงตัว ช่วยดูให้หน่อยค่ะ",
        "en": "Task “{title}” (#{id}) was sent back {n} times and is still not settled; please have a look.",
    },
    "hs_mode_set_notify": {
        "th": "ตั้งโหมดมอบหมายงานของเซิร์ฟเวอร์เป็น “แบบแจ้ง” แล้วค่ะ ผู้รับงานกดรับ แล้วสลิดแจ้งคนมอบงาน",
        "en": "Task mode for this server is now “notify”: the assignee accepts and the giver is told.",
    },
    "hs_mode_set_approve": {
        "th": "ตั้งโหมดมอบหมายงานของเซิร์ฟเวอร์เป็น “แบบอนุมัติ” แล้วค่ะ คนมอบงานต้องอนุมัติหลังผู้รับงานตอบรับ",
        "en": "Task mode for this server is now “approve”: the giver signs off after the assignee accepts.",
    },
    "hs_mode_me_notify": {
        "th": "ได้ค่ะ งานที่คุณมอบหมายจะแจ้งให้ทราบเฉยๆ",
        "en": "Done: tasks you give will just notify you.",
    },
    "hs_mode_me_approve": {
        "th": "ได้ค่ะ งานที่คุณมอบหมายจะส่งให้คุณอนุมัติก่อน",
        "en": "Done: tasks you give will come to you for approval.",
    },
    "hs_mode_denied": {
        "th": "เฉพาะแอดมินหรือผู้บริหารที่ตั้งโหมดของเซิร์ฟเวอร์ได้ค่ะ (ตั้งของตัวเองได้ด้วย “โหมดมอบหมายงานของฉัน …”)",
        "en": "Only admins and executives can change the server's task mode (yours: “my task mode approve / notify”).",
    },
}
CATALOG.update(STRINGS)

HANDSHAKE_PROMPT = (
    "\nถ้ามีคนแท็กสลิดบนข้อความที่มอบงาน หรือตอบกลับข้อความนั้นพร้อมแท็ก (“@สลิด มอบหมายตามนี้”, “เตย ทำสไลด์ศุกร์นี้ @สลิด”, "
    "“@Salid note the tasks above”) ให้อ่านข้อความนั้น (ข้อความที่ตอบกลับอยู่ในบริบทแล้ว) และข้อความแท็ก แล้วเรียก add_task หนึ่งครั้งต่อหนึ่งงาน "
    "(หลายคนในงานเดียวกันใส่ชื่อรวมกันได้) จากนั้นตอบสั้นๆ บรรทัดเดียวว่าสร้างอะไรไว้ (รหัสงาน ผู้รับ กำหนดส่ง) และผู้รับจะถูกถามให้ยืนยัน "
    "ถามกลับเฉพาะเมื่อไม่รู้จริงๆ ว่าใครเป็นผู้รับ. "
    "When tagged in, or replied to with a tag on, a message that hands out work, read it (the replied-to text is already in "
    "the context) plus the tag message, call add_task once per task (several names for one task are fine), then answer with ONE short "
    "line listing what was created (ids, assignees, due) and that the assignees will be asked to confirm; ask only when the assignee is genuinely unknown. "
    "งานที่ได้ยินในแชตจะถูกส่งให้ผู้รับผิดชอบกดยืนยัน (รับงาน/แก้ไข/ไม่ใช่ของฉัน) และคนมอบงานจะได้รับแจ้ง "
    "ถ้ามีคนบอกว่าไม่ใช่ของเขา ให้ย้ายงานด้วย update_task(task_id, assignee=ชื่อ) แล้วสลิดส่งการ์ดให้คนใหม่เอง. "
    "Tasks heard in chat are sent to the assignee to confirm (accept / edit / not mine) and the giver is "
    "told; reassign with update_task(task_id, assignee=name) and Salid sends the new person the card.\n"
)


def pending_marker(task: Task, lang: str) -> str:
    """A small marker for lists and reports: "" unless the task still awaits confirmation."""
    if task.confirm == AWAITING:
        return t("hs_awaiting_marker", lang)
    return t("hs_pending_marker", lang) if task.confirm == PENDING else ""


def giver_label(task: Task, lang: str) -> str:
    return task.assigner_name or t("hs_giver_unknown", lang)


def card_text(task: Task, lang: str, *, channel: str, project: str = "") -> str:
    return t(
        "hs_card",
        lang,
        channel=channel or "-",
        giver=giver_label(task, lang),
        title=task.title,
        due=task.due or "-",
        project=project or "-",
    )


def status_line(task: Task, lang: str, edited: bool = False) -> str:
    if task.confirm == DECLINED:
        return t("hs_status_no", lang)
    if task.confirm == AWAITING:
        return t("hs_status_wait", lang)
    return t("hs_status_edit" if edited else "hs_status_ok", lang)


def render_message(
    tasks: list[Task], lang: str, *, channel_of, project_of, edited: frozenset[str] = frozenset()
) -> str:
    """The text of a card message: one card per task; a task already answered shows its final
    status line instead of waiting for buttons. `channel_of(task)` / `project_of(task)` give names."""
    many = len(tasks) > 1
    blocks = []
    for i, task in enumerate(tasks, 1):
        head = card_text(task, lang, channel=channel_of(task), project=project_of(task))
        if many:
            head = f"{i}. {head}"
        if task.confirm == PENDING and task.send_back_note:
            head += "\n" + t("hs_back_line", lang, note=task.send_back_note)
        if task.confirm in ANSWERED:
            head += "\n" + status_line(task, lang, task.id in edited)
        blocks.append(head)
    return "\n\n".join(blocks)[:TEXT_MAX]


def batches(tasks: list[Task]) -> list[list[Task]]:
    return [tasks[i : i + MAX_CARDS] for i in range(0, len(tasks), MAX_CARDS)]


def describe_changes(old: Task, title: str, due: str, note: str, lang: str) -> list[str]:
    out = []
    if title and title != old.title:
        out.append(t("hs_change_title", lang, v=title))
    if due != old.due:
        out.append(t("hs_change_due", lang, v=due or "-"))
    if note != old.note:
        out.append(t("hs_change_note", lang, v=note or "-"))
    return out


def edited_fields(old: Task, title: str, due_text: str, note: str, today) -> tuple[dict, bool]:
    """(changes to apply, due_text_unreadable). Empty title keeps the old one; an unreadable due
    keeps the old due; an empty due text clears nothing (keeps the old due)."""
    new_title = (title or "").strip()[:300] or old.title
    bad = False
    due = old.due
    if (due_text or "").strip():
        parsed = parse_due(due_text, today)
        if parsed:
            due = parsed
        else:
            bad = True
    new_note = (note or "").strip()[:500]
    if not new_note:
        new_note = old.note
    return {"title": new_title, "due": due, "note": new_note}, bad


def giver_message(
    kind: str,
    task: Task,
    name: str,
    lang: str,
    changes: list[str] | None = None,
    group: tuple[int, int] | None = None,
) -> str:
    """The line for the person who gave the task. kind: ok | edit | no. `group` = (confirmed,
    total) for a task shared by several people."""
    if kind == "no":
        return t("hs_giver_no", lang, name=name, title=task.title, id=task.id)
    if kind == "edit" and changes:
        line = t("hs_giver_edit", lang, name=name, changes=", ".join(changes), id=task.id)
    else:
        line = t("hs_giver_ok", lang, name=name, title=task.title, due=task.due or "-", id=task.id)
    if group and group[1] > 1:
        line += t("hs_count", lang, n=group[0], total=group[1])
    return line


def nudge_due(task: Task, now: datetime) -> bool:
    """A task that has waited for its answer for more than 24 hours."""
    if task.confirm not in (PENDING, AWAITING) or task.status != "open" or not task.assignee_id:
        return False
    try:
        since = datetime.fromisoformat(task.confirm_at)
    except ValueError:
        return False
    if since.tzinfo is None:
        since = since.replace(tzinfo=now.tzinfo)
    return now - since >= NUDGE_AFTER


def offerable(task: Task) -> bool:
    """Whether a freshly created task is sent to its assignee for confirmation."""
    return (
        task.assignee_id > 0
        and task.created_from != "dm"
        and not is_dm_scope(task.source_channel_id)
        and not task.confirm
    )


# ---------------------------------------------------------------- approve mode
def diff_text(task: Task, lang: str) -> str:
    """ "title “a” → “b”, due x → y" for the edits the assignee made (from `task.edit_from`)."""
    import json

    try:
        old = json.loads(task.edit_from) if task.edit_from else {}
    except ValueError:
        return ""
    out = []
    if old.get("title") is not None and old["title"] != task.title:
        out.append(t("hs_diff_title", lang, a=old["title"], b=task.title))
    if old.get("due") is not None and old["due"] != task.due:
        out.append(t("hs_diff_due", lang, a=old["due"] or "-", b=task.due or "-"))
    if old.get("note") is not None and old["note"] != task.note:
        out.append(t("hs_diff_note", lang, a=old["note"] or "-", b=task.note or "-"))
    return "; ".join(out)


def boss_card_text(task: Task, name: str, lang: str) -> str:
    diff = diff_text(task, lang)
    return t(
        "hs_boss_card",
        lang,
        name=name,
        title=task.title,
        due=task.due or "-",
        edits=t("hs_boss_edits", lang, diff=diff) if diff else "",
        id=task.id,
    )


def effective_mode(org_mode: str, giver_mode: str, personal_ok: bool) -> str:
    """The giver's own mode when the organisation allows personal settings and they set one."""
    return giver_mode if personal_ok and giver_mode in ("notify", "approve") else org_mode


_APPROVE = r"(?:อนุมัติ|approve|approval)"
_NOTIFY = r"(?:แจ้ง|notify|inform)"


def parse_task_mode_command(text: str) -> tuple[str, bool] | None:
    """("approve" | "notify", personal) from "โหมดมอบหมายงานแบบอนุมัติ", "task mode notify",
    "งานที่ฉันมอบหมาย ขออนุมัติก่อน", "my task mode approve"; else None."""
    import re

    s = (text or "").strip().strip(" \t\n/,.:;!-")
    if not s or len(s) > 100:
        return None

    def mode_of(rest: str) -> str | None:
        a = re.search(_APPROVE, rest, re.I)
        n = re.search(_NOTIFY, rest, re.I)
        if a and not n:
            return "approve"
        if n and not a:
            return "notify"
        if a and n:
            return "approve" if a.start() < n.start() else "notify"
        return None

    low = s.casefold()
    personal = re.search(
        r"\bmy task mode\b|งานที่(?:ฉัน|ผม|เรา|หนู|ดิฉัน)มอบหมาย|โหมดมอบหมายงานของ(?:ฉัน|ผม|เรา|หนู)", low
    )
    if personal:
        m = mode_of(low[personal.end() :])
        return (m, True) if m else None
    server = re.search(r"โหมดมอบหมายงาน|\btask (?:assignment )?mode\b", low)
    if server:
        m = mode_of(low[server.end() :])
        return (m, False) if m else None
    return None
