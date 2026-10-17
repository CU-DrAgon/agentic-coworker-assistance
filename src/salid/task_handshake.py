"""Task handshake, pure part: the texts and decisions (no Discord, no network).

A task Salid hears in chat is sent to the person who has to do it, as a card with three buttons
(accept / edit / not mine). When they answer, the task is recorded and the person who gave it is
told. `task_handshake_bot.py` binds this to Discord; the card state lives on the `Task`
(`confirm`, `confirm_at`, `handshake_message`, `assigner_*`).
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta

from .hints import confirm_marker, help_tasks, with_hint
from .i18n import CATALOG, t
from .memory_access import is_dm_scope
from .progress import STRINGS as _PROGRESS_STRINGS  # noqa: F401  (registers the catalog keys)
from .tasks import Task, match_person, parse_due, split_who

MAX_CARDS = 5  # one message holds 5 rows of buttons (Discord: 25 components, 3 used per card)
NUDGE_AFTER = timedelta(hours=24)
PENDING, ACCEPTED, DECLINED = "pending", "accepted", "declined"
AWAITING = "awaiting_approval"  # approve mode: the assignee accepted, the giver must sign off
OWNER_ASKED = "owner_asked"  # nobody is assigned: the giver was asked who takes it
ANSWERED = (ACCEPTED, DECLINED, AWAITING)
NOT_A_TASK = "not a task"  # the note of a task its giver dropped
MAX_ROUNDS = 3  # send-backs before the task stays pending and an executive is told
TEXT_MAX = 1900
HINT_ROOM = 320  # characters kept free below a card message for its hint lines

STRINGS = {
    "hs_card": {
        "th": "📋 งานใหม่ จาก #{channel} ({giver}): **{title}** · กำหนด {due} · โปรเจกต์ {project}",
        "en": "📋 New task from #{channel} ({giver}): **{title}** · due {due} · project {project}",
    },
    "hs_card_id": {"th": " · งาน #{id}", "en": " · task #{id}"},
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
    "hs_owner_card": {
        "th": "❓ งานนี้ให้ใครทำ — จาก #{channel}: **{title}** · กำหนด {due}",
        "en": "❓ Who takes this task? — from #{channel}: **{title}** · due {due}",
    },
    "hs_btn_om": {"th": "🙋 ของฉันเอง", "en": "🙋 Mine"},
    "hs_btn_oo": {"th": "👤 ให้คนอื่น", "en": "👤 Someone else"},
    "hs_btn_on": {"th": "❌ ไม่ใช่งาน", "en": "❌ Not a task"},
    "hs_owner_mine": {"th": "✅ งานของคุณ · งาน #{id}", "en": "✅ Yours · task #{id}"},
    "hs_owner_other": {
        "th": "👤 ส่งให้ {name} แล้ว · งาน #{id}",
        "en": "👤 Sent to {name} · task #{id}",
    },
    "hs_owner_none": {
        "th": "🚫 ไม่ใช่งาน ปิดแล้ว · งาน #{id}",
        "en": "🚫 Not a task, closed · task #{id}",
    },
    "hs_owner_modal": {"th": "ให้ใครทำ", "en": "Who takes it"},
    "hs_owner_field": {"th": "ชื่อหรือ @ ", "en": "Name or @mention"},
    "hs_owner_unknown": {
        "th": "ไม่พบ “{name}” ลองพิมพ์ @ชื่อ ค่ะ",
        "en": "I could not find “{name}”, try @name.",
    },
    "hs_owner_not_giver": {
        "th": "เฉพาะคนที่มอบงานนี้ (หรือผู้บริหาร) ตอบได้ค่ะ",
        "en": "Only the person who gave this task (or an executive) can answer.",
    },
    "hs_owner_which": {
        "th": "งานไหนคะ ตอบพร้อมรหัสงาน เช่น “งาน #รหัส ของฉัน”:",
        "en": "Which task? Reply with its id, for example “task #id mine”:",
    },
    "hs_owner_say_none": {"th": "โอเคค่ะ ไม่จดเป็นงานนะคะ", "en": "Okay, I won't keep it as a task."},
    "hs_owner_say_mine": {"th": "โอเคค่ะ เป็นงานของคุณนะคะ", "en": "Okay, it's yours."},
    "hs_owner_say_other": {"th": "โอเคค่ะ ส่งให้เขาแล้วนะคะ", "en": "Okay, I've passed it on."},
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
    "told; reassign with update_task(task_id, assignee=name) and Salid sends the new person the card. "
    "ถ้าสลิดไม่แน่ใจว่าใครรับงานที่ได้ยินในแชต สลิดจะถามคนพูดเป็นการส่วนตัวว่างานนี้ให้ใคร (ของฉัน / ให้คนอื่น / ไม่ใช่งาน); "
    'ถ้าคนพูดตอบในแชตว่า “งาน #id ของฉัน” ให้เรียก update_task(task_id, assignee="me"). '
    "When Salid is unsure who takes a task heard in chat, she asks the speaker privately (mine / someone else / not a task); "
    'if the speaker answers in chat "task #id is mine", call update_task(task_id, assignee="me"). '
    "เมื่อมีคนบอกว่างานที่สลิดถามถึง (owner_pending) เป็นของเขา / ให้ใคร / ไม่ใช่งาน ให้เรียก answer_task_question(task_id, mine|other|none, name) "
    "แล้วตอบสั้นๆ หนึ่งบรรทัด. When someone says a task Salid asked about (owner_pending) is theirs / for X / not a task, call "
    "answer_task_question(task_id, mine|other|none, name) and reply with one short line.\n"
    "ถ้า list_tasks / my_projects_and_tasks ส่ง legend มา (มีงานที่มีสัญลักษณ์สถานะ) ให้แสดง legend นั้นครั้งเดียวเป็นบรรทัดสุดท้าย "
    "และเมื่อผู้ใช้ถามว่าใช้งานยังไง / สัญลักษณ์แปลว่าอะไร / ตอบยังไง ให้ตอบจากคู่มือนี้ (ไม่ต้องเรียกเครื่องมือ): "
    "If list_tasks / my_projects_and_tasks return a legend (some tasks carry a status marker), show that legend once as the last line; "
    "when someone asks how to use Salid, what a symbol means or how to answer, answer from this guide (no tool call):\n"
    + help_tasks("th")
    + "\n"
    + help_tasks("en")
    + "\n"
)


def pending_marker(task: Task, lang: str) -> str:
    """A small marker for lists and reports: "" unless the task still awaits confirmation."""
    return confirm_marker(task.confirm, lang)


def giver_label(task: Task, lang: str) -> str:
    return task.assigner_name or t("hs_giver_unknown", lang)


def card_text(task: Task, lang: str, *, channel: str, project: str = "") -> str:
    line = t(
        "hs_card",
        lang,
        channel=channel or "-",
        giver=giver_label(task, lang),
        title=task.title,
        due=task.due or "-",
        project=project or "-",
    )
    if task.step > 0:  # a step of an imported plan (salid.plan_import)
        line += " · " + t("prog_step", lang, n=task.step)
    return line + t("hs_card_id", lang, id=task.id)


def status_line(task: Task, lang: str, edited: bool = False) -> str:
    if task.confirm == DECLINED:
        return t("hs_status_no", lang)
    if task.confirm == AWAITING:
        return t("hs_status_wait", lang)
    return t("hs_status_edit" if edited else "hs_status_ok", lang)


def render_message(
    tasks: list[Task],
    lang: str,
    *,
    channel_of,
    project_of,
    edited: frozenset[str] = frozenset(),
    first_time: bool = False,
) -> str:
    """The text of a card message: one card per task; a task already answered shows its final
    status line instead of waiting for buttons. `channel_of(task)` / `project_of(task)` give names.
    While a card still waits, two hint lines close the message: how to answer and what #id means
    (a person's very first card swaps the id line for a "first time?" pointer to the help)."""
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
    text = "\n\n".join(blocks)
    if any(x.confirm not in ANSWERED for x in tasks):
        keys = ("answer_confirm", "first_time" if first_time else "task_id")
        return with_hint(text[: TEXT_MAX - HINT_ROOM], *keys, lang=lang)
    return text[:TEXT_MAX]


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
        line = t("hs_giver_no", lang, name=name, title=task.title, id=task.id)
        return with_hint(line, "reassign", "mention", lang=lang)
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
    return _waited(task, now)


def owner_nudge_due(task: Task, now: datetime) -> bool:
    """An owner question the giver has not answered for more than 24 hours."""
    if task.confirm != OWNER_ASKED or task.status != "open" or task.assignee_id:
        return False
    return _waited(task, now)


def _waited(task: Task, now: datetime) -> bool:
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


def ownerless(task: Task) -> bool:
    """A fresh task from a team channel that nobody takes and whose giver is known: the giver is
    asked who it goes to (a named team nobody could pick has its own message)."""
    return (
        task.assignee_id == 0
        and task.assigner_id > 0
        and task.created_from != "dm"
        and not is_dm_scope(task.source_channel_id)
        and not task.confirm
        and task.status != "done"
    )


def owner_waiting(task: Task) -> bool:
    """Still waiting for the giver's answer (also while the card is being sent)."""
    return task.status == "open" and not task.assignee_id and task.confirm in ("", OWNER_ASKED)


def owner_status(task: Task, lang: str, names: str = "") -> str:
    """The final line of an owner card, from the task's state ("" while it still waits)."""
    if owner_waiting(task):
        return ""
    if task.status == "done" and task.note == NOT_A_TASK:
        return t("hs_owner_none", lang, id=task.id)
    if task.assignee_id and task.assignee_id == task.assigner_id:
        return t("hs_owner_mine", lang, id=task.id)
    return t("hs_owner_other", lang, name=names or task.assignee_name or "-", id=task.id)


def render_owner_message(tasks: list[Task], lang: str, *, channel_of, names_of) -> str:
    """The owner question: one block per task, with its final line once answered."""
    many = len(tasks) > 1
    blocks = []
    for i, task in enumerate(tasks, 1):
        head = t(
            "hs_owner_card",
            lang,
            channel=channel_of(task) or "-",
            title=task.title,
            due=task.due or "-",
        )
        if many:
            head = f"{i}. {head}"
        final = owner_status(task, lang, names_of(task))
        if final:
            head += "\n" + final
        blocks.append(head)
    text = "\n\n".join(blocks)
    if any(owner_waiting(x) for x in tasks):
        return with_hint(text[: TEXT_MAX - HINT_ROOM], "answer_owner", "mention", lang=lang)
    return text[:TEXT_MAX]


# ---------------------------------------------------------------- the owner's answer
_PARTICLES = r"(?:ครับ|ค่ะ|คะ|นะครับ|นะคะ|นะ|จ้า|จ้ะ|เลย|please|pls|thanks)"
_NONE = re.compile(
    r"ไม่ใช่งาน|ไม่ต้องจด|ไม่ต้องสั่ง|ไม่ต้องเป็นงาน|ลบทิ้ง|ลบออก|ไม่เอางาน|"
    r"\bnot a task\b|\bnot task\b|\bignore (?:that|it|this)\b|\bdrop (?:it|that|this)\b|"
    r"\bnever ?mind\b|\bforget (?:it|that)\b|\bdelete (?:it|that)\b|\bno task\b"
)
_MINE = re.compile(
    r"(?:ของ|เป็นของ)(?:ฉัน|ผม|เรา|หนู|ดิฉัน|ตัวเอง)(?:เอง)?|(?:ฉัน|ผม|เรา|หนู|ดิฉัน)(?:ทำ)?เอง|"
    r"mine|it'?s mine|this is mine|me|i(?:'ll| will) (?:do|take) (?:it|this)|i'll do it|my own"
)
_OTHER = (
    re.compile(r"(?:(?:อันนี้|งานนี้|งาน)\s*)?(?:เป็น)?ให้\s*(.+?)(?:\s*ทำ|\s*รับ)?"),
    re.compile(r"(?:it'?s |this is |give it |assign it |assign this |hand it )?(?:for|to)\s+(.+?)"),
    re.compile(r"(?:ของ|เป็นของ)\s*(.+?)"),
    re.compile(r"(<@!?\d+>.*)"),
)
_REF = re.compile(r"^(?:งาน|task)\s*#?[0-9a-z_-]{4,40}\s*", re.I)
_LEAD = re.compile(r"^(?:อ๋อ|อ่อ|อ๋า|โอเค|ok(?:ay)?|ah|oh|well|เออ|ครับ|ค่ะ)\s*[,.!]*\s*", re.I)


def parse_owner_answer(text: str) -> tuple[str, str] | None:
    """("mine", "") | ("other", name) | ("none", "") from a short reply to the owner question
    ("ไม่ใช่งาน", "ของฉัน", "ให้บี", "not a task", "it's for Bee"); None = not an answer."""
    s = (text or "").strip().casefold()
    s = s.strip(" \t\n.,!?:;~-…")
    if not s or len(s) > 90:
        return None
    s = _REF.sub("", s).strip(" ,.:;")
    if _NONE.search(s):
        return "none", ""
    for _ in range(3):  # greetings in front, particles behind
        s2 = _LEAD.sub("", s, count=1)
        s2 = re.sub(r"\s*" + _PARTICLES + r"\s*$", "", s2).strip(" ,.!?:;~")
        if s2 == s:
            break
        s = s2
    if s in ("ไม่ใช่", "ไม่", "no", "nope", "nah") or s.startswith("ไม่ใช่ "):
        return "none", ""
    if _MINE.fullmatch(s):
        return "mine", ""
    for rx in _OTHER:
        m = rx.fullmatch(s)
        if m:
            name = m[1].strip(" ,.")
            if name and len(name) <= 40 and not _MINE.fullmatch(name):
                return "other", name
    return None


class OwnerResult:
    """What `apply_owner_answer` did: the tasks it changed or made, the ones now needing the
    normal assignee card, and the name it could not resolve (nothing changed then)."""

    def __init__(self) -> None:
        self.tasks: list[Task] = []
        self.offer: list[Task] = []
        self.unresolved = ""


def apply_owner_answer(repo, task: Task, kind: str, name: str, people: dict, now: str):
    """Blocking. kind mine | other | none. Several names: one task per person (shared group_id),
    the original becomes the first person's."""
    out = OwnerResult()
    if kind == "none":
        task.status, task.note, task.confirm, task.confirm_at, task.done_at = (
            "done",
            NOT_A_TASK,
            DECLINED,
            now,
            "",
        )
        out.tasks = [repo.save(task)]
        return out
    if kind == "mine":
        ids = [task.assigner_id]
    else:
        parts = split_who(name)
        ids = []
        for part in parts:
            uid = match_person(part, people)
            if not uid:
                out.unresolved = part
                return out
            if uid not in ids:
                ids.append(uid)
        if not ids:
            out.unresolved = name.strip()
            return out
    group = uuid.uuid4().hex[:8] if len(ids) > 1 else task.group_id
    for i, uid in enumerate(ids):
        row = (
            task
            if i == 0
            else task.model_copy(update={"id": uuid.uuid4().hex[:8], "owner_message": ""})
        )
        row.assignee_id = uid
        row.assignee_name = people.get(uid) or (
            task.assigner_name if uid == task.assigner_id else ""
        )
        row.group_id = group
        row.confirm_at = now
        if uid == task.assigner_id:
            row.confirm = ACCEPTED  # taking it yourself needs no confirmation
        else:
            row.confirm, row.confirm_at = "", ""
            row.handshake_message = ""
        out.tasks.append(repo.save(row))
        if not row.confirm:
            out.offer.append(row)
    return out


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


def boss_card_text(task: Task, name: str, lang: str, *, hint: bool = True) -> str:
    """The approval card for the giver; `hint=False` once it is answered (no buttons left)."""
    diff = diff_text(task, lang)
    text = t(
        "hs_boss_card",
        lang,
        name=name,
        title=task.title,
        due=task.due or "-",
        edits=t("hs_boss_edits", lang, diff=diff) if diff else "",
        id=task.id,
    )
    return with_hint(text, "answer_boss", lang=lang) if hint else text


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
