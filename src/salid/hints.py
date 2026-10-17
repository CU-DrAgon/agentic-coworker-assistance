"""How to answer Salid: one shared vocabulary of short hints (pure, no Discord, no network).

Whenever Salid sends something that expects a typed answer, the message ends with ONE or TWO
"ℹ️" lines that explain the symbols (#id, @name), the words that work and the status markers.
The words live in the catalog (`hint_<key>`, Thai + English, second tier by the translate script);
`with_hint(text, "answer_owner", "mention", lang=...)` appends them, never twice.

The status legend is built from the same marker strings the task lists use (`hs_*_marker`), so
cards, lists, reports and the legend always show the same emoji and words.
"""

from __future__ import annotations

import re

from .i18n import CATALOG, t

# ---------------------------------------------------------------- the status markers
M_PENDING, M_APPROVAL, M_OWNER, M_OVERDUE, M_DONE = "⏳", "🔏", "❓", "⚠️", "✅"
MARKER_KEYS = (
    "hs_pending_marker",  # ⏳ awaiting confirmation
    "hs_awaiting_marker",  # 🔏 awaiting approval
    "hs_owner_marker",  # ❓ owner pending
    "hs_overdue_marker",  # ⚠️ overdue
    "hs_done_marker",  # ✅ done
)
MARKER_EMOJI = (M_PENDING, M_APPROVAL, M_OWNER, M_OVERDUE, M_DONE)

STRINGS: dict[str, dict[str, str]] = {
    "hs_pending_marker": {"th": "⏳ รอยืนยัน", "en": "⏳ awaiting confirmation"},
    "hs_awaiting_marker": {"th": "🔏 รออนุมัติ", "en": "🔏 awaiting approval"},
    "hs_owner_marker": {"th": "❓ รอระบุผู้รับ", "en": "❓ owner pending"},
    "hs_overdue_marker": {"th": "⚠️ เลยกำหนด", "en": "⚠️ overdue"},
    "hs_done_marker": {"th": "✅ เสร็จ", "en": "✅ done"},
    "hint_prefix": {"th": "ℹ️ ", "en": "ℹ️ "},
    "hint_task_id": {
        "th": "#a1b2c3 = รหัสงาน ใช้อ้างถึงงานได้ เช่น “งาน #a1b2c3 เสร็จแล้ว”",
        "en": "#a1b2c3 is the task id: refer to it like “task #a1b2c3 done”",
    },
    "hint_mention": {
        "th": "@ชื่อ = แท็กคนใน Discord (พิมพ์ @ แล้วเลือกชื่อ)",
        "en": "@name tags a person in Discord (type @ and pick the name)",
    },
    "hint_answer_owner": {
        "th": "ตอบได้: กดปุ่มด้านล่าง หรือพิมพ์ “ของฉัน” · “ให้ @ชื่อ” · “ไม่ใช่งาน”",
        "en": "Answer with a button, or type “mine” · “for @name” · “not a task”",
    },
    "hint_answer_confirm": {
        "th": "ตอบได้: กดปุ่มด้านล่าง (รับงาน / แก้ไข / ไม่ใช่ของฉัน) แล้วพิมพ์ “งาน #id เสร็จแล้ว” เมื่อทำเสร็จ",
        "en": "Answer with a button (Accept / Edit / Not mine), then type “task #id done” when finished",
    },
    "hint_answer_boss": {
        "th": "ตอบได้: กดปุ่มด้านล่าง (อนุมัติ / แก้ไข / ส่งกลับ)",
        "en": "Answer with a button below (Approve / Edit / Send back)",
    },
    "hint_reassign": {
        "th": "พิมพ์ “งาน #id ให้ @ชื่อ” เพื่อมอบให้คนอื่น",
        "en": "Type “task #id to @name” to hand it to someone",
    },
    "hint_due_formats": {
        "th": "กำหนดส่งพิมพ์ได้ เช่น วันนี้ · พรุ่งนี้ · ศุกร์ · 2026-10-20",
        "en": "Due dates can be: today · tomorrow · friday · 2026-10-20",
    },
    "hint_time_formats": {
        "th": "บอกเวลาได้ เช่น “พรุ่งนี้ 9 โมง” · “อีก 2 ชั่วโมง” · “ทุกวันจันทร์ 10:00”",
        "en": "Say the time like “tomorrow 9am” · “in 2 hours” · “every monday 10:00”",
    },
    "hint_nodue": {
        "th": "ไม่มีกำหนด: กด ✏️ แก้ไข เพื่อใส่ หรือพิมพ์ “งาน #{id} กำหนด ศุกร์”",
        "en": "No deadline: press ✏️ Edit to add one, or type “task #{id} due Friday”",
    },
    "hint_ph_due": {
        "th": "วันนี้ · พรุ่งนี้ · ศุกร์ · 2026-10-20",
        "en": "today · tomorrow · friday · 2026-10-20",
    },
    "hint_ph_who": {"th": "@ชื่อ หรือ ชื่อ", "en": "@name or a name"},
    "hint_lang_how": {
        "th": "ตอบได้: เลือกจากเมนูด้านบน (แอดมินเท่านั้น) หรือพิมพ์ “/language th” · “/language en”",
        "en": "Pick from the menu above (admins only), or type “/language th” · “/language en”",
    },
    "hint_vote_how": {
        "th": "โหวต: กดปุ่มตัวเลือกด้านล่าง เปลี่ยนใจได้จนกว่าจะปิดโหวต",
        "en": "To vote, press an option button below; you can change it until the vote closes",
    },
    "hint_first_time": {
        "th": "ครั้งแรกใช่ไหมคะ: การ์ดนี้คือการถามว่าคุณรับงานนี้ไหม กดปุ่มด้านล่างได้เลย "
        "พิมพ์ “สลิด ใช้งานยังไง” เพื่อดูคำอธิบายสัญลักษณ์ทั้งหมดค่ะ",
        "en": "First time? This card asks whether you take the task: press a button below. "
        "Type “Salid help” to see what every symbol means.",
    },
    "hint_welcome_help": {
        "th": "พิมพ์ “สลิด ใช้งานยังไง” ได้ทุกเมื่อ",
        "en": "Type “Salid help” any time",
    },
    "hint_legend_title": {"th": "สัญลักษณ์สถานะ: ", "en": "Status symbols: "},
    "hint_help_tasks": {
        "th": "**วิธีใช้งานกับสลิด**\n"
        "• สั่งงานได้ 3 ทาง: พูดคุยกันตามปกติ สลิดจะจับงานเอง · แท็กสลิดใต้ข้อความที่มอบงาน · "
        "หรือบอกสลิดตรงๆ เช่น “ให้ @เตย ทำสไลด์ศุกร์นี้”\n"
        "• คนรับงานจะได้การ์ด กดปุ่ม รับงาน / แก้ไข / ไม่ใช่ของฉัน ถ้าสลิดไม่รู้ว่างานเป็นของใคร จะถามคนมอบ: "
        "ของฉัน / ให้คนอื่น / ไม่ใช่งาน\n"
        "• พิมพ์ตอบแทนปุ่มได้: “งาน #รหัส เสร็จแล้ว” · “งาน #รหัส ให้ @ชื่อ” · “ของฉัน” · “ไม่ใช่งาน”\n"
        "• #รหัส = รหัสงาน · @ชื่อ = แท็กคน (พิมพ์ @ แล้วเลือกชื่อ) · กำหนดส่ง: วันนี้ · พรุ่งนี้ · ศุกร์ · 2026-10-20\n"
        "• {legend}\n"
        "• ดูงานได้ที่ “งานของฉัน” หรือหน้า /pm",
        "en": "**How to work with Salid**\n"
        "• Three ways to hand out work: just talk and Salid picks tasks up · tag Salid under a message "
        "that gives work · or tell Salid directly, like “give @Toey the slides by friday”\n"
        "• The assignee gets a card: Accept / Edit / Not mine. If Salid cannot tell who owns a task she "
        "asks the giver: Mine / Someone else / Not a task\n"
        "• You can type instead of pressing: “task #id done” · “task #id to @name” · “mine” · “not a task”\n"
        "• #id = task id · @name tags a person (type @ and pick the name) · due: today · tomorrow · friday · 2026-10-20\n"
        "• {legend}\n"
        "• See tasks with “my tasks” or on the /pm page",
    },
}
CATALOG.update(STRINGS)

# Every one-line hint (`legend` is built from the markers; `help_tasks` is the multi-line guide).
LINE_HINTS = (
    "task_id",
    "mention",
    "answer_owner",
    "answer_confirm",
    "answer_boss",
    "reassign",
    "due_formats",
    "nodue",
    "time_formats",
    "ph_due",
    "ph_who",
    "lang_how",
    "vote_how",
    "first_time",
    "welcome_help",
)


def hint(key: str, lang: str = "en") -> str:
    """One hint line (without the ℹ️ prefix). `legend` and `help_tasks` are built here."""
    if key == "legend":
        return legend(lang)
    if key == "help_tasks":
        return help_tasks(lang)
    return t(f"hint_{key}", lang)


def legend(lang: str = "en") -> str:
    """The status legend on one line: every marker with its word, from the shared marker strings."""
    return t("hint_legend_title", lang) + " · ".join(t(k, lang) for k in MARKER_KEYS)


def help_tasks(lang: str = "en") -> str:
    """The "how to answer Salid" guide (<= 12 lines)."""
    return t("hint_help_tasks", lang, legend=legend(lang))


def with_hint(text: str, *keys: str, lang: str = "en") -> str:
    """`text` with the hints for `keys` appended as "ℹ️ ..." lines at the very end. A hint whose
    text already appears in `text` (or twice in `keys`) is skipped; no keys left = `text` as is."""
    lines: list[str] = []
    for key in keys:
        line = hint(key, lang)
        if not line or line in text or line in lines:
            continue
        lines.append(line)
    if not lines:
        return text
    prefix = t("hint_prefix", lang)
    return text.rstrip() + "\n" + "\n".join(prefix + line for line in lines)


_CONFIRM_MARKER = {
    "pending": "hs_pending_marker",
    "awaiting_approval": "hs_awaiting_marker",
    "owner_asked": "hs_owner_marker",
}


def confirm_marker(confirm: str, lang: str = "en") -> str:
    """The marker of a task's `confirm` state ("" when nothing is pending)."""
    key = _CONFIRM_MARKER.get(confirm or "")
    return t(key, lang) if key else ""


def overdue_marker(lang: str = "en") -> str:
    return t("hs_overdue_marker", lang)


def status_marker(confirm: str, overdue: bool, lang: str = "en") -> str:
    """The marker a task list shows for a task: its confirm state first, then overdue ("" = none)."""
    return confirm_marker(confirm, lang) or (overdue_marker(lang) if overdue else "")


def has_marker(text: str) -> bool:
    """Whether `text` shows one of the status emoji of the legend."""
    return any(e in (text or "") for e in MARKER_EMOJI[:4])


def with_legend(text: str, lang: str = "en", *, markers: bool | None = None) -> str:
    """`text` plus the legend line, only when a status marker is present (or `markers` says so)."""
    show = has_marker(text) if markers is None else markers
    return with_hint(text, "legend", lang=lang) if show else text


# ---------------------------------------------------------------- the help command
_HELP_PATTERNS = (
    r"(?:ช่วย)?(?:อธิบาย|บอก)?(?:วิธี)?(?:ใช้งาน|ใช้)(?:สลิด|salid)?(?:ยังไง|อย่างไร|ไง|ยังงัย)",
    r"(?:ช่วย)?(?:อธิบาย|บอก|ดู)(?:ความหมาย)?(?:ของ)?สัญลักษณ์(?:ให้)?(?:หน่อย)?",
    r"สัญลักษณ์(?:พวกนี้|เหล่านี้|ต่างๆ|ต่าง ๆ)?(?:คือ|หมายถึง)?(?:อะไร|ยังไง|ไง)?",
    r"(?:ตอบ|ตอบกลับ|ตอบสลิด)(?:ยังไง|อย่างไร|ไง)",
    r"วิธี(?:ใช้งาน|ใช้|ตอบ)(?:สลิด|salid)?",
    r"ช่วยเหลือ|ขอความช่วยเหลือ|คู่มือ|วิธีใช้",
    r"help(?: me)?(?: please)?|salid help|help salid|usage|guide|user guide",
    r"how (?:do|can|should) (?:i|we) (?:use|answer|reply to|respond to)(?: salid| you| this| it)?",
    r"how (?:does|do) (?:this|salid|it) work",
    r"how to (?:use|answer|reply to)(?: salid| this)?",
    r"what do(?:es)? (?:the |these |those )?(?:symbols?|icons?|emoji|markers?|signs?)(?: mean| stand for)?",
    r"explain (?:the )?(?:symbols?|icons?|emoji|markers?)",
    r"what (?:are|is) (?:the )?(?:symbols?|statuses|status icons?)",
)
_HELP_RX = re.compile(r"(?:" + "|".join(_HELP_PATTERNS) + r")", re.I)
_HELP_LEAD = re.compile(r"^(?:ช่วย|ขอ|อยากรู้|อยากทราบ|please|pls|can you|could you|tell me)\s*", re.I)
_PUNCT = " \t\n.,!?:;~-…“”\"'“”"


def parse_help_command(text: str) -> bool:
    """True for a short message that asks how to use Salid / what the symbols mean ("ใช้งานยังไง",
    "ช่วยอธิบายสัญลักษณ์", "help", "how do I answer"). The whole message must be the question."""
    s = (text or "").strip(_PUNCT).casefold()
    if not s or len(s) > 70:
        return False
    s = re.sub(r"^(?:สลิด|salid)\s*[,:]?\s*", "", s).strip(_PUNCT)
    s = _HELP_LEAD.sub("", s, count=1).strip(_PUNCT)
    s = re.sub(r"\s*(?:ครับ|ค่ะ|คะ|นะครับ|นะคะ|นะ|หน่อย|ด้วย|เลย|please|pls)\s*$", "", s).strip(_PUNCT)
    return bool(s) and _HELP_RX.fullmatch(s) is not None
