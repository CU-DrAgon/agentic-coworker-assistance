"""Salid's reply language: Thai, English or auto, plus the catalog of fixed user-facing strings.

Language setting values: `th`, `en`, `auto` (reply in the language the person wrote in). A
resolved language is always `th` or `en`. Thai entries are the original wording and the default.
"""

from __future__ import annotations

import re

LANGS = ("th", "en", "auto")
DEFAULT_LANGUAGE = "th"
LANGUAGE_LABELS = {"th": "ไทย · Thai", "en": "English", "auto": "อัตโนมัติ · Auto"}
_THAI_RE = re.compile(r"[฀-๿]")


def has_thai(text: str | None) -> bool:
    return _THAI_RE.search(text or "") is not None


def resolve_lang(setting: str | None, text: str | None = None) -> str:
    """The language to reply in. `auto` follows `text`: Thai characters -> th, other letters ->
    en, and no text (or no letters, e.g. only emoji) -> th, today's default."""
    if setting in ("th", "en"):
        return setting
    if setting == "auto":
        if has_thai(text):
            return "th"
        return "en" if any(c.isalpha() for c in text or "") else "th"
    return DEFAULT_LANGUAGE


def resolve_language(
    channel_id: int,
    *,
    channel_languages: dict[str, str],
    default_language: str = DEFAULT_LANGUAGE,
) -> str:
    """The channel's language setting: its own override, else the guild default (th/en/auto)."""
    explicit = channel_languages.get(str(channel_id))
    if explicit in LANGS:
        return explicit
    return default_language if default_language in LANGS else DEFAULT_LANGUAGE


_FILLER = r"(?:\s*(?:ครับ|ค่ะ|คะ|นะ|หน่อย|ด้วย|เถอะ|เลย|please|pls|plz|now|mode))*[\s.!?]*"
_LANG_COMMANDS = (
    (
        "en",
        r"(?:(?:พูด|ตอบ|คุย)\s?)?ภาษา\s?อังกฤษ|(?:พูด|ตอบ|คุย)\s?อังกฤษ"
        r"|(?:(?:please\s+)?(?:speak|reply|answer|talk|respond)(?:\s+in)?\s+english)"
        r"|(?:in\s+)?english|use\s+english",
    ),
    (
        "th",
        r"(?:(?:พูด|ตอบ|คุย)\s?)?ภาษา\s?ไทย|(?:พูด|ตอบ|คุย)\s?ไทย"
        r"|(?:(?:please\s+)?(?:speak|reply|answer|talk|respond)(?:\s+in)?\s+thai)"
        r"|(?:in\s+)?thai|use\s+thai",
    ),
    (
        "auto",
        r"(?:(?:พูด|ตอบ|คุย)\s?)?ภาษา\s?(?:อัตโนมัติ|ออโต้|auto)|อัตโนมัติ"
        r"|auto(?:matic)?(?:\s+language)?|(?:language|lang)\s+auto(?:matic)?",
    ),
)
_LANG_RES = [(v, re.compile(rf"^(?:{p})(?:{_FILLER})$", re.I)) for v, p in _LANG_COMMANDS]


def parse_language_command(text: str) -> str | None:
    """ "พูดภาษาอังกฤษ" / "speak English" -> "en", "พูดไทย" / "speak Thai" -> "th", "ภาษาอัตโนมัติ" /
    "auto language" -> "auto", anything else None. The whole (mention-stripped) message must be
    the command, so a question that merely starts with "English" is not one."""
    s = (text or "").strip().strip(" \t\n/,.:;!-–—…")
    if not s or len(s) > 40:
        return None
    for value, rx in _LANG_RES:
        if rx.match(s):
            return value
    return None


def _bi(th: str, en: str) -> dict[str, str]:
    both = f"{th} · {en}"
    return {"th": both, "en": both}


def t(key: str, lang: str = "th", **fmt) -> str:
    """Catalog text for a resolved language (anything but `en` falls back to Thai)."""
    entry = CATALOG[key]
    text = entry["en"] if lang == "en" else entry["th"]
    return text.format(**fmt) if fmt else text


CATALOG: dict[str, dict[str, str]] = {
    # ---------------------------------------------------------------- bot: general
    "qa_off": {
        "th": "ตอนนี้ปิดการถามตอบอยู่ครับ เปิดได้ที่หน้าศูนย์บัญชาการของสลิด",
        "en": "Q&A is switched off right now. You can turn it on in Salid's command center.",
    },
    "deep_off": {
        "th": "ตอนนี้ปิดงานใหญ่อยู่ครับ เปิดได้ที่หน้าศูนย์บัญชาการของสลิด",
        "en": "Big tasks are switched off right now. You can turn them on in Salid's command center.",
    },
    "dm": {
        "th": "สลิดแยกข้อมูลของแต่ละเซิร์ฟเวอร์ครับ รบกวนแท็ก {mention} ถามในห้องแชตของเซิร์ฟเวอร์ที่ต้องการนะครับ",
        "en": "Salid keeps each server's data separate. Please tag {mention} in the chat channel of "
        "the server you mean.",
    },
    "welcome": {
        "th": "สวัสดีครับ ผม “สลิด” เลขาและเพื่อนร่วมทีม 🐱\n"
        "• คุยกับผมได้ทุกเรื่องในห้อง {chat} พิมพ์ได้เลยไม่ต้องแท็ก "
        "ห้องอื่นแท็ก {mention} หรือเรียกชื่อ “สลิด” ได้เลยครับ "
        "(พิมพ์ /โหมด หรือ /mode เพื่อดูหรือปรับว่าผมจะคุยในแต่ละห้องแค่ไหน)\n"
        "• ส่งรูปใบเสร็จในห้องที่ชื่อมีคำว่า “ใบเสร็จ” หรือ “receipt” "
        "หรือแท็ก {mention} พร้อมรูป ผมจะอ่านยอดและบันทึกให้เอง\n"
        "• คุยได้ทั้งไทยและอังกฤษ พิมพ์ /ภาษา หรือ /language เพื่อเลือกภาษาที่ผมตอบ "
        "(หรือแท็กแล้วพิมพ์ “พูดภาษาอังกฤษ” / “speak English”)\n"
        "• สั่งเข้าห้องประชุมเสียงด้วย /เข้าประชุม หรือ /join และให้ออกด้วย /ออกประชุม หรือ /leave "
        "(หรือแท็กแล้วพิมพ์ “เข้าประชุม” / “join”)\n"
        "• ผู้ดูแลพิมพ์ /ตั้งค่า หรือ /setup เพื่อเชื่อมกับศูนย์บัญชาการ (หน้าเว็บสำหรับตั้งค่า)",
        "en": "Hi, I'm “Salid”, your team's secretary and teammate 🐱\n"
        "• Chat with me about anything in {chat}, no tag needed. In other channels tag {mention} "
        "or call my name “Salid” (type /mode or /โหมด to see or change how much I join in, "
        "channel by channel).\n"
        "• Post a receipt photo in a channel whose name has “receipt” or “ใบเสร็จ”, or tag "
        "{mention} with the photo, and I will read the total and record it.\n"
        "• I speak Thai and English. Type /language or /ภาษา to choose my reply language "
        "(or tag me and say “speak English” / “พูดภาษาไทย”).\n"
        "• Ask me to join a voice meeting with /join or /เข้าประชุม and to leave with /leave or "
        "/ออกประชุม (or tag me and say “join” / “leave”).\n"
        "• Admins: type /setup or /ตั้งค่า to link the server to the command center (the web "
        "page for settings).",
    },
    "chat_topic": {
        "th": "คุยกับน้องสลิดได้ทุกเรื่อง พิมพ์ได้เลยไม่ต้องแท็ก",
        "en": "Chat with Salid about anything, no tag needed",
    },
    "greeting": {
        "th": 'สวัสดีครับ สลิดเองครับ 🐱 ส่งรูปใบเสร็จมาได้เลย หรือแท็กสลิดพร้อมคำถาม เช่น "เดือนนี้จ่ายค่าวัตถุดิบไปเท่าไร" ครับ',
        "en": 'Hi, Salid here 🐱 Send me a receipt photo, or tag Salid with a question, e.g. "how '
        'much did we spend on ingredients this month?"',
    },
    "signoff": {
        "th": "ถ้ามีอะไรก็ทักน้องสลิดมาได้เลยนะครับ 👋",
        "en": "If you need anything, just ping Salid 👋",
    },
    "snooze": {
        "th": "ได้เลยครับ สลิดเงียบในห้องนี้ 1 ชั่วโมง 🤫 พิมพ์ “@Salid คุยได้แล้ว” เมื่อไรก็เรียกกลับได้ครับ",
        "en": "Sure, Salid will stay quiet in this channel for 1 hour 🤫 Type “@Salid you can talk” "
        "any time to call me back.",
    },
    "unsnooze": {"th": "กลับมาคุยแล้วครับ 😊", "en": "I'm back 😊"},
    "not_snoozed": {
        "th": "สลิดคุยได้ตามปกติอยู่แล้วครับ 🙂",
        "en": "Salid is already chatting as usual 🙂",
    },
    "history_meeting": {
        "th": "บันทึกการประชุมล่าสุด (ใครพูดอะไร):\n{meeting}\n\nแชต:\n{chat}",
        "en": "Latest meeting transcript (who said what):\n{meeting}\n\nChat:\n{chat}",
    },
    "correction_denied": {
        "th": "แก้ไขได้เฉพาะผู้ส่งใบเสร็จหรือผู้บริหารเท่านั้น",
        "en": "Only the person who sent the receipt, or an executive, can edit it.",
    },
    "deep_failed": {
        "th": "❌ งานใหญ่ล้มเหลว ลองใหม่อีกครั้งภายหลังครับ",
        "en": "❌ The big task failed. Please try again later.",
    },
    # ---------------------------------------------------------------- bot: slash commands
    "setup_denied": {
        "th": "คำสั่งนี้ใช้ได้เฉพาะเจ้าของเซิร์ฟเวอร์หรือผู้ที่มีสิทธิ์จัดการเซิร์ฟเวอร์ครับ",
        "en": "This command is only for the server owner or people who can manage the server.",
    },
    "setup_message": {
        "th": "ลิงก์เชื่อม {name} กับศูนย์บัญชาการของสลิด 🔗\n{url}\n"
        "เปิดลิงก์แล้วเข้าสู่ระบบด้วยบัญชี Google ของคุณ ลิงก์ใช้ได้ครั้งเดียวและหมดอายุใน 15 นาที "
        "อย่าส่งต่อให้คนอื่นครับ",
        "en": "Link to connect {name} to Salid's command center 🔗\n{url}\n"
        "Open the link and sign in with your Google account. It works once and expires in 15 "
        "minutes. Please do not share it.",
    },
    "setup_failed": {
        "th": "สร้างลิงก์ไม่สำเร็จ ลองพิมพ์ /ตั้งค่า อีกครั้งภายหลังครับ",
        "en": "Could not create the link. Please try /setup again later.",
    },
    "mode_denied": {
        "th": "ปรับโหมดได้เฉพาะผู้ดูแลเซิร์ฟเวอร์หรือผู้บริหารครับ",
        "en": "Only server admins or executives can change the mode.",
    },
    "mode_failed": {
        "th": "ตั้งโหมดไม่สำเร็จ ลองใหม่อีกครั้งภายหลังนะครับ",
        "en": "Could not set the mode. Please try again later.",
    },
    "mode_current": {
        "th": "โหมดของห้องนี้ตอนนี้: {label}",
        "en": "This channel's mode right now: {label}",
    },
    "mode_snoozed": {
        "th": "(สลิดกำลังพักเงียบชั่วคราว ทำตัวเหมือนโหมดฟังเงียบ)",
        "en": "(Salid is snoozed for now and acts like Listen mode)",
    },
    "mode_footer": {
        "th": "ผู้ดูแลหรือผู้บริหารเปลี่ยนได้ด้วย /โหมด แล้วเลือกโหมดครับ",
        "en": "Admins or executives can change it with /mode and then pick a mode.",
    },
    "mode_set": {"th": "ตั้งห้องนี้เป็น {label} แล้วครับ", "en": "This channel is now set to {label}."},
    "lang_current": {
        "th": "ภาษาที่สลิดตอบในห้องนี้ตอนนี้: {label}",
        "en": "Salid's reply language in this channel right now: {label}",
    },
    "lang_footer": {
        "th": "ผู้ดูแลหรือผู้บริหารเปลี่ยนได้ด้วย /ภาษา แล้วเลือกภาษาครับ "
        "(หรือแท็กสลิดแล้วพิมพ์ “speak English” / “พูดไทย”)",
        "en": "Admins or executives can change it with /language and then pick a language "
        "(or tag Salid and say “speak English” / “พูดไทย”).",
    },
    "lang_denied": {
        "th": "เปลี่ยนภาษาได้เฉพาะผู้ดูแลเซิร์ฟเวอร์หรือผู้บริหารครับ",
        "en": "Only server admins or executives can change the language.",
    },
    "lang_failed": {
        "th": "ตั้งภาษาไม่สำเร็จ ลองใหม่อีกครั้งภายหลังนะครับ",
        "en": "Could not set the language. Please try again later.",
    },
    "lang_set": {
        "th": "ตั้งภาษาของห้องนี้เป็น {label} แล้วครับ",
        "en": "This channel's language is now {label}.",
    },
    "lang_label_th": {"th": "ไทย", "en": "Thai"},
    "lang_label_en": {"th": "อังกฤษ (English)", "en": "English"},
    "lang_label_auto": {
        "th": "อัตโนมัติ (ตอบตามภาษาที่พิมพ์มา)",
        "en": "Auto (reply in the language you write in)",
    },
    "mode_label_off": {"th": "🔇 ปิด", "en": "🔇 Off"},
    "mode_label_listen": {"th": "👂 ฟังเงียบ", "en": "👂 Listen"},
    "mode_label_call": {"th": "🙋 เรียกแล้วคุย", "en": "🙋 Call"},
    "mode_label_join": {"th": "🎧 ร่วมวง", "en": "🎧 Join"},
    "mode_label_chat": {"th": "💬 ตอบทุกข้อความ", "en": "💬 Chat"},
    "mode_help_off": {
        "th": "สลิดไม่ยุ่งกับห้องนี้เลย (ไม่อ่านใบเสร็จ ไม่ตอบ แท็กก็ไม่ตอบ)",
        "en": "Salid stays out of this channel completely (no receipts, no replies, even if tagged)",
    },
    "mode_help_listen": {
        "th": "ทำงานเบื้องหลังเท่านั้น อ่านใบเสร็จแล้วแปะอีโมจิ ไม่ตอบเป็นข้อความ",
        "en": "Works in the background only: reads receipts and adds an emoji, no text replies",
    },
    "mode_help_call": {
        "th": "ตอบเมื่อถูกแท็กหรือถูกเรียกชื่อ แล้วคุยต่อได้ 3 นาทีโดยไม่ต้องแท็ก",
        "en": "Replies when tagged or called by name, then keeps chatting for 3 minutes without a tag",
    },
    "mode_help_join": {
        "th": "เหมือนโหมดเรียกแล้วคุย และสลิดอาจเข้าร่วมวงเองเมื่อมีคำถามที่ช่วยได้ (จำกัดชั่วโมงละ 3 ครั้ง)",
        "en": "Like Call, and Salid may chime in on its own when it can help with a question "
        "(at most 3 times an hour)",
    },
    "mode_help_chat": {
        "th": "ตอบทุกข้อความในห้อง",
        "en": "Replies to every message in the channel",
    },
    # ---------------------------------------------------------------- slash command texts
    # (Discord shows one description, so these are bilingual in both columns)
    "cmd_setup": _bi("เชื่อมเซิร์ฟเวอร์กับศูนย์บัญชาการ (ผู้ดูแล)", "Link server to command center (admins)"),
    "cmd_mode": _bi("ดูหรือตั้งโหมดการคุยของสลิดในห้องนี้", "View or set Salid's mode here"),
    "cmd_mode_arg": _bi("โหมดใหม่ (ไม่เลือก = ดูโหมดปัจจุบัน)", "New mode (empty = view)"),
    "cmd_language": _bi("ดูหรือตั้งภาษาที่สลิดตอบในห้องนี้", "View or set Salid's language here"),
    "cmd_language_arg": _bi("ภาษาใหม่ (ไม่เลือก = ดูภาษาปัจจุบัน)", "New language (empty = view)"),
    "cmd_join": _bi("ให้สลิดเข้าห้องเสียงเพื่อจดบันทึกประชุม", "Salid joins your voice channel"),
    "cmd_leave": _bi("ให้สลิดออกจากห้องเสียงและสรุปการประชุม", "Salid leaves and summarizes"),
    # ---------------------------------------------------------------- meeting
    "meet_announce": {
        "th": "สลิดเข้ามาฟังและจดบันทึกการประชุมครับ 🎙️ "
        "(เก็บเป็นข้อความในหน่วยความจำเท่านั้น ไม่บันทึกเสียง) "
        "เรียก “สลิด” แล้วถามได้เลย หรือพิมพ์ /ออกประชุม เพื่อสรุปและให้ผมออก",
        "en": "Salid has joined to listen and take meeting notes 🎙️ "
        "(kept as text in memory only, no audio is recorded). "
        "Say “Salid” and ask away, or type /leave to get the summary and let me go.",
    },
    "meet_not_in_voice": {
        "th": "ต้องเข้าห้องเสียง (Voice Channel) ก่อน แล้วค่อยเรียกสลิดเข้าประชุมนะครับ",
        "en": "Please join a voice channel first, then ask Salid to join the meeting.",
    },
    "meet_already": {
        "th": "สลิดกำลังอยู่ในการประชุมของเซิร์ฟเวอร์นี้แล้วครับ ({channel}) พิมพ์ /ออกประชุม เพื่อจบก่อนนะครับ",
        "en": "Salid is already in a meeting on this server ({channel}). Type /leave to end it first.",
    },
    "meet_joining": {"th": "กำลังเข้า", "en": "joining now"},
    "meet_none": {
        "th": "ตอนนี้สลิดไม่ได้อยู่ในการประชุมครับ",
        "en": "Salid is not in a meeting right now.",
    },
    "meet_leave_denied": {
        "th": "ต้องอยู่ในห้องประชุมเดียวกัน (หรือเป็นผู้ดูแลเซิร์ฟเวอร์) ถึงจะสั่งให้สลิดออกได้ครับ",
        "en": "You need to be in the same meeting channel (or be a server admin) to tell Salid to leave.",
    },
    "meet_off": {
        "th": "ตอนนี้ปิดฟังการประชุมอยู่ครับ เปิดได้ที่หน้าศูนย์บัญชาการของสลิด",
        "en": "Meeting listening is switched off right now. You can turn it on in Salid's command center.",
    },
    "meet_no_permission": {
        "th": "สลิดไม่มีสิทธิ์เชื่อมต่อ (Connect) ห้องเสียงนี้ครับ รบกวนผู้ดูแลเปิดสิทธิ์ให้ก่อน",
        "en": "Salid does not have permission to Connect to this voice channel. Please ask an admin to allow it.",
    },
    "meet_connect_failed": {
        "th": "เข้าห้องเสียงไม่สำเร็จครับ ลองใหม่อีกครั้งภายหลังนะครับ",
        "en": "Could not join the voice channel. Please try again later.",
    },
    "meet_joined": {"th": "สลิดเข้าห้อง {channel} แล้วครับ", "en": "Salid has joined {channel}."},
    "meet_left": {
        "th": "สลิดออกจากการประชุมแล้วครับ กำลังสรุปให้ในแชตของห้องเสียง",
        "en": "Salid has left the meeting. The summary is coming in the voice channel's chat.",
    },
    "meet_wake_empty": {"th": "ครับ? ว่ามาได้เลยครับ", "en": "Yes? Go ahead."},
    "meet_no_transcript": {
        "th": "จบการประชุมแล้วครับ แต่ยังไม่ได้ยินเนื้อหาให้สรุปเลย 🙂",
        "en": "The meeting is over, but I did not hear anything to summarize 🙂",
    },
    "meet_summary_failed": {
        "th": "จบการประชุมแล้วครับ แต่สรุปไม่สำเร็จ ขออภัยด้วยนะครับ",
        "en": "The meeting is over, but the summary failed. Sorry about that.",
    },
    "meet_summary_header": {
        "th": "📝 สรุปการประชุม ({reason})\n",
        "en": "📝 Meeting summary ({reason})\n",
    },
    "meet_participant": {"th": "ผู้ร่วมประชุม {n}", "en": "Participant {n}"},
    "end_command": {
        "th": "ผู้ใช้สั่งให้ออกจากการประชุม",
        "en": "someone told Salid to leave",
    },
    "end_empty": {"th": "ทุกคนออกจากห้องแล้ว", "en": "everyone left the channel"},
    "end_timeout": {"th": "ประชุมครบ 3 ชั่วโมง", "en": "the meeting reached 3 hours"},
    "end_disconnected": {"th": "สลิดหลุดจากห้องเสียง", "en": "Salid was disconnected"},
    "end_shutdown": {"th": "สลิดกำลังปิดระบบ", "en": "Salid is shutting down"},
    # ---------------------------------------------------------------- digest
    "digest_title": {
        "th": "📒 สรุปค่าใช้จ่ายประจำวันที่ {day}",
        "en": "📒 Daily expense summary for {day}",
    },
    "digest_none": {
        "th": "ไม่มีใบเสร็จที่บันทึกในวันนี้",
        "en": "No receipts were recorded today",
    },
    "digest_total": {
        "th": "ยอดรวม: {total} ({count} ใบเสร็จ)",
        "en": "Total: {total} ({count} receipts)",
    },
    "digest_by_category": {"th": "แยกตามหมวด:", "en": "By category:"},
    "digest_top_vendors": {
        "th": "ร้านที่ใช้จ่ายสูงสุด 3 อันดับ:",
        "en": "Top 3 vendors by spending:",
    },
    "digest_flagged": {
        "th": "⚠️ รายการที่ต้องตรวจสอบ ({n}):",
        "en": "⚠️ Needs review ({n}):",
    },
    "digest_dup": {"th": "อาจซ้ำ", "en": "possible duplicate"},
    "digest_unclear": {"th": "ข้อมูลไม่ชัด", "en": "unclear data"},
    "digest_no_total": {"th": "ไม่มียอด", "en": "no total"},
    "digest_flag_line": {
        "th": "• #{id} {vendor} {amount} ({why}) โดย {who}",
        "en": "• #{id} {vendor} {amount} ({why}) by {who}",
    },
    # ---------------------------------------------------------------- receipts
    "unknown": {"th": "ไม่ทราบ", "en": "unknown"},
    "rc_check": {"th": "⚠️ กรุณาตรวจสอบ", "en": "⚠️ Please check"},
    "rc_saved": {"th": "✅ บันทึกแล้ว", "en": "✅ Saved"},
    "rc_vendor": {"th": "ร้าน: {v}", "en": "Vendor: {v}"},
    "rc_date": {"th": "วันที่: {v}", "en": "Date: {v}"},
    "rc_date_unknown": {"th": "ไม่ทราบ (ใช้วันที่ส่ง)", "en": "unknown (using the date sent)"},
    "rc_total": {"th": "ยอดรวม: {v}", "en": "Total: {v}"},
    "rc_category": {"th": "หมวด: {v}", "en": "Category: {v}"},
    "rc_unclear": {
        "th": 'อ่านข้อมูลไม่ชัด ตอบข้อความนี้เพื่อแก้ไข เช่น "ยอดจริง 450" หรือ "หมวด ค่าเดินทาง"',
        "en": 'Some details were hard to read. Reply to this message to fix them, e.g. "total 450" '
        'or "category ค่าเดินทาง"',
    },
    "rc_possible_dup": {
        "th": "⚠️ อาจซ้ำกับใบเสร็จ #{id} (ร้าน/วันที่/ยอดเท่ากัน)",
        "en": "⚠️ Possible duplicate of receipt #{id} (same vendor, date and total)",
    },
    "rc_duplicate": {
        "th": "ℹ️ ใบเสร็จนี้เคยบันทึกแล้ว (#{id} ร้าน {vendor} ยอด {total}) ไม่บันทึกซ้ำ",
        "en": "ℹ️ This receipt was already recorded (#{id}, vendor {vendor}, total {total}). Not "
        "saving it again.",
    },
    "corr_unclear": {
        "th": 'ไม่เข้าใจคำแก้ไข ลองพิมพ์ เช่น "ยอดจริง 450", "หมวด ค่าเดินทาง", "ร้าน ABC" หรือ "วันที่ 2026-10-12"',
        "en": 'I did not understand that correction. Try e.g. "total 450", "category ค่าเดินทาง", '
        '"vendor ABC" or "date 2026-10-12"',
    },
    "corr_total": {"th": "ยอดรวม", "en": "total"},
    "corr_vat": {"th": "VAT", "en": "VAT"},
    "corr_vendor": {"th": "ร้าน", "en": "vendor"},
    "corr_date": {"th": "วันที่", "en": "date"},
    "corr_category": {"th": "หมวด", "en": "category"},
    "corr_notes": {"th": "หมายเหตุ", "en": "notes"},
    "corr_done": {
        "th": "✏️ แก้ไขใบเสร็จ #{id} แล้ว: {changed}\nตอนนี้: {now}",
        "en": "✏️ Receipt #{id} updated: {changed}\nNow: {now}",
    },
    "qa_failed": {
        "th": "ขออภัย ตอบไม่ได้ในตอนนี้ ลองถามใหม่อีกครั้งนะครับ",
        "en": "Sorry, I can't answer right now. Please try asking again.",
    },
    # ---------------------------------------------------------------- deep tasks
    "deep_disabled": {
        "th": "ฟีเจอร์งานใหญ่ยังไม่ได้เปิดใช้งานครับ (ยังไม่ได้ตั้งค่า SALID_GEMINI_CLI_PATH) ลองถามคำถามสั้นๆ แทนได้เลย",
        "en": "Big tasks are not enabled yet (SALID_GEMINI_CLI_PATH is not set). Please ask a "
        "short question instead.",
    },
    "deep_not_exec": {
        "th": "ขออภัยครับ งานใหญ่ใช้ได้เฉพาะผู้บริหารเท่านั้น ถามคำถามสั้นๆ เรื่องใบเสร็จของคุณได้เลยครับ",
        "en": "Sorry, big tasks are for executives only. You can still ask short questions about "
        "your own receipts.",
    },
    "deep_cli_failed": {
        "th": "Gemini CLI ทำงานไม่สำเร็จ ลองใหม่อีกครั้งภายหลังครับ",
        "en": "Gemini CLI failed. Please try again later.",
    },
    "deep_no_result": {
        "th": "ไม่ได้รับผลลัพธ์จากงานนี้ครับ",
        "en": "This task returned no result.",
    },
    "deep_ack": {
        "th": "รับงานแล้วครับ 📋 งานนี้ใหญ่กว่าคำถามทั่วไป สลิดจะวิเคราะห์ข้อมูลทั้งหมดแล้วเขียนรายงาน "
        "อาจใช้เวลาประมาณ 1-5 นาที เสร็จแล้วจะส่งผลในห้องนี้ครับ",
        "en": "Got it 📋 This is bigger than a normal question. Salid will analyze all the data and "
        "write a report. It may take about 1-5 minutes, and I will post the result in this channel.",
    },
    "deep_busy": {
        "th": "ตอนนี้สลิดกำลังทำงานใหญ่อีกงานอยู่ครับ รอให้เสร็จก่อนแล้วสั่งใหม่อีกครั้งนะครับ",
        "en": "Salid is busy with another big task. Please wait for it to finish and ask again.",
    },
    "deep_usage": {
        "th": "บอกโจทย์ด้วยครับ เช่น งานใหญ่: วิเคราะห์ค่าใช้จ่ายเดือนนี้ หาหมวดที่โตผิดปกติ แล้วเขียนรายงาน",
        "en": "Please tell me the task, e.g. deep task: analyze this month's expenses, find "
        "categories that grew unusually, and write a report",
    },
    "deep_attached": {
        "th": "(รายงานฉบับเต็มแนบเป็นไฟล์ {name})",
        "en": "(The full report is attached as {name})",
    },
    "deep_stats": {
        "th": "⏱ {seconds:.0f} วินาที · tokens {tokens:,} · โมเดล: {models}",
        "en": "⏱ {seconds:.0f} s · tokens {tokens:,} · model: {models}",
    },
    "deep_report_lang": {
        "th": "เป็นภาษาไทย",
        "en": "เป็นภาษาอังกฤษ (English)",
    },
    "deep_report_lang_auto": {
        "th": "ด้วยภาษาเดียวกับที่เขียนโจทย์ (ไทยหรืออังกฤษ)",
        "en": "ด้วยภาษาเดียวกับที่เขียนโจทย์ (ไทยหรืออังกฤษ)",
    },
}
