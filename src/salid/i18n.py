"""Salid's reply language: Thai, English or auto, plus the catalog of fixed user-facing strings.

Language setting values: `th`, `en`, `auto` (reply in the language the person wrote in). A
resolved language is always `th` or `en`. Thai entries are the original wording and the default.
"""

from __future__ import annotations

import re

LANGS = ("th", "en", "auto")
DEFAULT_LANGUAGE = "th"
LANGUAGE_LABELS = {"th": "ไทย · Thai", "en": "English · อังกฤษ", "auto": "อัตโนมัติ · Auto"}
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
        "th": "ตอนนี้ปิดการถามตอบอยู่ค่ะ เปิดได้ที่หน้าศูนย์บัญชาการของสลิด",
        "en": "Q&A is switched off right now. You can turn it on in Salid's command center.",
    },
    "deep_off": {
        "th": "ตอนนี้ปิดงานใหญ่อยู่ค่ะ เปิดได้ที่หน้าศูนย์บัญชาการของสลิด",
        "en": "Big tasks are switched off right now. You can turn them on in Salid's command center.",
    },
    "dm": {
        "th": "สลิดแยกข้อมูลของแต่ละเซิร์ฟเวอร์ค่ะ รบกวนแท็ก {mention} ถามในห้องแชตของเซิร์ฟเวอร์ที่ต้องการนะคะ",
        "en": "Salid keeps each server's data separate. Please tag {mention} in the chat channel of "
        "the server you mean.",
    },
    "dmg_ask": {
        "th": "คุยเรื่องบริษัทไหนคะ?",
        "en": "Which company is this about?",
    },
    "dmg_switched": {
        "th": "ได้ค่ะ ต่อไปนี้สลิดคุยเรื่องบริษัท {name} นะคะ",
        "en": "Done. I'll talk about {name} from now on.",
    },
    "dmg_confirm": {
        "th": "เปลี่ยนไปคุยเรื่องบริษัท {name} ใช่ไหมคะ?",
        "en": "Switch to talking about {name}?",
    },
    "dmg_btn_ok": {"th": "ใช่ค่ะ", "en": "Yes"},
    "dmg_btn_no": {"th": "ยกเลิก", "en": "Cancel"},
    "dmg_kept": {"th": "ได้ค่ะ คุยเรื่องเดิมต่อนะคะ", "en": "OK, staying on the current company."},
    "pc_category": {"th": "คุยกับสลิด (ส่วนตัว)", "en": "Salid private chats"},
    "pc_topic": {
        "th": "ห้องส่วนตัวของคุณกับสลิด",
        "en": "Your private room with Salid",
    },
    "pc_welcome": {
        "th": "สวัสดีค่ะ {mention} นี่คือห้องส่วนตัวของคุณกับสลิดค่ะ "
        "ห้องนี้เห็นแค่คุณกับสลิด (และแอดมินของเซิร์ฟเวอร์ที่มีสิทธิ์ Administrator) "
        "คุยได้เลยไม่ต้องแท็ก สลิดจะส่งสรุป เตือนงาน และขออนุมัติมาที่นี่ด้วยค่ะ "
        "อยากรับทาง DM แทนก็บอกได้ หรือพิมพ์ “ลบห้องส่วนตัว” เพื่อลบห้องนี้ค่ะ",
        "en": "Hi {mention}, this is your private room with Salid. Only you and Salid can see it "
        "(plus server admins who have the Administrator permission). Just talk, no tag needed. "
        "Salid will send your briefings, reminders and approval requests here. Prefer DMs? Say so. "
        'Type "delete my private channel" to remove this room.',
    },
    "pc_created": {
        "th": "สร้างห้องส่วนตัวให้แล้วค่ะ {channel} (เซิร์ฟเวอร์ {server})",
        "en": "Your private channel is ready: {channel} ({server})",
    },
    "pc_exists": {
        "th": "คุณมีห้องส่วนตัวอยู่แล้วค่ะ {channel} (เซิร์ฟเวอร์ {server})",
        "en": "You already have a private channel: {channel} ({server})",
    },
    "pc_failed": {
        "th": "สร้างห้องส่วนตัวไม่สำเร็จค่ะ ลองใหม่อีกครั้งนะคะ",
        "en": "I couldn't create the private channel. Please try again.",
    },
    "pc_off": {
        "th": "แอดมินปิดห้องส่วนตัวไว้ค่ะ",
        "en": "Private channels are switched off by the admins.",
    },
    "pc_cap": {
        "th": "เซิร์ฟเวอร์นี้ห้องเต็มใกล้ขีดจำกัดของ Discord แล้ว สลิดจะส่งทาง DM แทนนะคะ",
        "en": "This server is near Discord's channel limit, so I'll use DMs instead.",
    },
    "pc_cap_admin": {
        "th": "ห้องในเซิร์ฟเวอร์ใกล้ถึงขีดจำกัดของ Discord (500) สลิดเลยหยุดสร้างห้องส่วนตัวให้สมาชิกเพิ่ม และจะส่งทาง DM แทนค่ะ",
        "en": "The server is close to Discord's channel limit (500), so I've paused creating private "
        "channels and will use DMs for the rest.",
    },
    "pc_none": {"th": "คุณยังไม่มีห้องส่วนตัวค่ะ", "en": "You don't have a private channel."},
    "pc_delete_ask": {
        "th": "ลบห้องส่วนตัวของคุณใช่ไหมคะ? ตอบ “ยืนยัน” เพื่อลบค่ะ",
        "en": 'Delete your private channel? Reply "confirm" to delete it.',
    },
    "pc_deleted": {"th": "ลบห้องส่วนตัวให้แล้วค่ะ", "en": "Your private channel is deleted."},
    "pc_offer": {
        "th": "หรือจะคุยในห้องส่วนตัวในเซิร์ฟเวอร์ก็ได้นะคะ (พิมพ์ “ขอห้องส่วนตัว” ในเซิร์ฟเวอร์นั้น)",
        "en": 'Or talk in your private channel inside the server (say "make me a private channel").',
    },
    "pc_offer_link": {"th": "ห้องของคุณใน {server}: {url}", "en": "Your room in {server}: {url}"},
    "cmd_private": _bi("สร้างห้องคุยส่วนตัวกับสลิด", "Create your private chat with Salid"),
    "dmg_footer": {"th": "— บริษัท: {name}", "en": "— Company: {name}"},
    "dmg_current": {"th": "ตอนนี้: {name}", "en": "Now: {name}"},
    "dmg_not_yours": {
        "th": "ปุ่มนี้ของคนอื่นค่ะ",
        "en": "These buttons are for someone else.",
    },
    "dmg_gone": {
        "th": "ดูเหมือนคุณไม่ได้อยู่ในเซิร์ฟเวอร์นั้นแล้วค่ะ เลือกใหม่ได้เลยนะคะ",
        "en": "You no longer seem to be in that server. Please pick again.",
    },
    "dmg_none": {
        "th": "สลิดไม่พบเซิร์ฟเวอร์ที่คุณอยู่ร่วมกับสลิดค่ะ",
        "en": "I can't find a server we share.",
    },
    "cmd_company": _bi("เลือกบริษัทที่สลิดจะคุยด้วยใน DM", "Choose which company Salid talks about in DMs"),
    "cmd_company_arg": _bi("ชื่อบริษัท (ไม่ใส่ = เลือกจากปุ่ม)", "Company name (empty = pick with buttons)"),
    "welcome": {
        "th": "สวัสดีค่ะ สลิด เลขาประจำทีมค่ะ 🐱 ทักสลิดได้เลย พิมพ์หรือส่งไฟล์ รูป เสียงมาได้หมด "
        "ห้องนี้คุยกับสลิดได้เลยไม่ต้องแท็ก ห้องอื่นแท็ก {mention} หรือเรียกชื่อสลิดก็พอ ({chat})\n"
        "สลิดจำแค่ห้องที่สลิดเห็นได้ ({days} วัน) ไม่อยากให้จำห้องไหน บอกสลิดได้เลยว่า “ห้องนี้ปิดไว้นะ”",
        "en": "Hi, I'm Salid, your team's secretary 🐱 Just talk to me: type, or send files, photos "
        "or voice, anything goes. In {chat} you can chat with me without tagging; elsewhere tag "
        "{mention} or call my name.\n"
        "I only remember channels I'm allowed to see ({days} days). If you'd rather I skip one, "
        "just tell me “switch this channel off”.",
    },
    "chat_topic": {
        "th": "คุยกับน้องสลิดได้ทุกเรื่อง พิมพ์ได้เลยไม่ต้องแท็ก",
        "en": "Chat with Salid about anything, no tag needed",
    },
    "greeting": {
        "th": "ว่าไงค่ะ มีอะไรให้ช่วยไหม 🙂",
        "en": "Hey, what can I do for you? 🙂",
    },
    "greeting_2": {
        "th": "ค่ะ ว่าเลยค่ะ",
        "en": "Yes? I'm all ears.",
    },
    "greeting_3": {
        "th": "สลิดอยู่ค่ะ มีอะไรให้ช่วยบอกได้เลย",
        "en": "Salid here, tell me what you need.",
    },
    "ack_doc_1": {
        "th": "ขอเวลาแป๊บนึงนะคะ ขออ่านเอกสารให้เข้าใจก่อน 📄",
        "en": "Give me a moment, I want to read the document properly first 📄",
    },
    "ack_doc_2": {
        "th": "ขออ่านเอกสารก่อนนะคะ เดี๋ยวมาตอบ 📄",
        "en": "Let me read through this first, back in a moment 📄",
    },
    "ack_image_1": {
        "th": "ขอดูรูปแป๊บนึงนะคะ 🖼️",
        "en": "Let me take a look at the picture 🖼️",
    },
    "ack_image_2": {
        "th": "แป๊บนึงนะคะ ขอดูรูปให้ชัดก่อน",
        "en": "One sec, looking at it closely",
    },
    "ack_audio_1": {"th": "ขอฟังคลิปก่อนนะคะ 🎧", "en": "Let me listen to the clip first 🎧"},
    "ack_audio_2": {
        "th": "ขอเวลาฟังแป๊บนึงนะคะ 🎧",
        "en": "Give me a minute to listen to it 🎧",
    },
    "ack_link_1": {"th": "ขอเปิดลิงก์ดูแป๊บนึงค่ะ 🔗", "en": "Let me open that link real quick 🔗"},
    "ack_link_2": {
        "th": "เดี๋ยวขออ่านลิงก์ก่อนนะคะ 🔗",
        "en": "I'll read the link first and get back to you 🔗",
    },
    "ack_deep_1": {"th": "เดี๋ยวสลิดไล่ดูให้นะคะ", "en": "I'll go through it for you"},
    "ack_deep_2": {
        "th": "รับทราบค่ะ ขอเวลาไล่ดูข้อมูลก่อนนะ",
        "en": "On it, give me a bit to go through the data",
    },
    "ack_slow_1": {"th": "แป๊บนึงนะคะ กำลังดูให้อยู่", "en": "One moment, I'm on it"},
    "ack_slow_2": {
        "th": "ขอเวลาเช็กให้แป๊บนึงนะคะ",
        "en": "Give me a second to check that",
    },
    "relay_dm": {
        "th": "ข้อความจากคุณ {sender} ({server}) ผ่านสลิดค่ะ: {text}",
        "en": "A message from {sender} ({server}), passed on by Salid: {text}",
    },
    "bc_post": {"th": "📢 แจ้งจากคุณ{sender}: {text}", "en": "📢 From {sender}: {text}"},
    "bc_preview_post": {
        "th": "จะโพสต์ใน #{channel} {ping}\n“{text}”\nยืนยันไหมคะ",
        "en": "I'll post in #{channel} {ping}\n“{text}”\nShall I go ahead?",
    },
    "bc_cant_see": {
        "th": "ไม่แท็ก {names} เพราะมองไม่เห็นห้องนี้ ให้สลิดส่ง DM บอกแทนไหมคะ",
        "en": "Not tagging {names}: they can't see this channel. Shall I DM them instead?",
    },
    "bc_preview_dm": {
        "th": "จะส่ง DM ถึง {count} คน: {names}\n“{text}”\nยืนยันไหมคะ",
        "en": "I'll DM {count} people: {names}\n“{text}”\nShall I go ahead?",
    },
    "bc_report": {
        "th": "ส่ง DM เสร็จแล้วค่ะ: ส่งสำเร็จ {sent} คน",
        "en": "DMs done: sent to {sent}",
    },
    "bc_report_closed": {
        "th": "ปิดรับ DM {count} คน ({names})",
        "en": "DMs closed for {count} ({names})",
    },
    "bc_report_failed": {
        "th": "ส่งไม่สำเร็จ {count} คน ({names})",
        "en": "failed for {count} ({names})",
    },
    "dm_closed": {
        "th": "ส่ง DM หา {name} ไม่ได้ค่ะ น่าจะปิดรับข้อความส่วนตัวจากสมาชิกในเซิร์ฟเวอร์ไว้ "
        "ให้เปิดที่ ชื่อเซิร์ฟเวอร์ ({server}) > ความเป็นส่วนตัว (Privacy Settings) > อนุญาตข้อความส่วนตัว (Direct Messages) แล้วบอกสลิดอีกที",
        "en": "I couldn't DM {name}; they probably have server DMs turned off. They can switch them "
        "on under {server} > Privacy Settings > Direct Messages, then ask me again.",
    },
    "signoff": {
        "th": "ถ้ามีอะไรก็ทักน้องสลิดมาได้เลยนะคะ 👋",
        "en": "If you need anything, just ping Salid 👋",
    },
    "snooze": {
        "th": "ได้เลยค่ะ สลิดเงียบในห้องนี้ 1 ชั่วโมง 🤫 พิมพ์ “@Salid คุยได้แล้ว” เมื่อไรก็เรียกกลับได้ค่ะ",
        "en": "Sure, Salid will stay quiet in this channel for 1 hour 🤫 Type “@Salid you can talk” "
        "any time to call me back.",
    },
    "unsnooze": {"th": "กลับมาคุยแล้วค่ะ 😊", "en": "I'm back 😊"},
    "not_snoozed": {
        "th": "สลิดคุยได้ตามปกติอยู่แล้วค่ะ 🙂",
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
        "th": "❌ งานใหญ่ล้มเหลว ลองใหม่อีกครั้งภายหลังค่ะ",
        "en": "❌ The big task failed. Please try again later.",
    },
    # ---------------------------------------------------------------- bot: slash commands
    "hidden_none": {
        "th": "สลิดเห็นทุกห้องแล้วค่ะ 👍",
        "en": "Salid can see every channel now 👍",
    },
    "hidden_header": {
        "th": "🔒 ห้องที่สลิดยังมองไม่เห็น ({n} ห้อง)",
        "en": "🔒 Channels Salid cannot see yet ({n})",
    },
    "hidden_reason_view": {"th": "ไม่เห็นห้อง", "en": "cannot see the channel"},
    "hidden_reason_history": {"th": "อ่านย้อนหลังไม่ได้", "en": "cannot read history"},
    "hidden_more": {
        "th": "และอีก {m} ห้อง (ดูทั้งหมดในหน้า Hub)",
        "en": "and {m} more (see all of them in the Hub)",
    },
    "hidden_steps": {
        "th": "วิธีให้สลิดเข้าห้อง:\n"
        "1. กดชื่อห้องด้านบน แล้วกด ⚙️ (Edit Channel)\n"
        "2. ไปที่ Permissions → กด + (Add members or roles)\n"
        "3. พิมพ์ Salid แล้วเลือก\n"
        "4. ติ๊ก ✅ View Channel และ ✅ Read Message History แล้วกด Save",
        "en": "How to let Salid in:\n"
        "1. Click the channel name above, then click ⚙️ (Edit Channel)\n"
        "2. Go to Permissions → click + (Add members or roles)\n"
        "3. Type Salid and pick it\n"
        "4. Tick ✅ View Channel and ✅ Read Message History, then click Save",
    },
    "hidden_footer": {
        "th": "สลิดจะเริ่มอ่านห้องนั้นและประวัติย้อนหลังให้เองทันที ห้องลับที่ไม่อยากให้สลิดรู้ ไม่ต้องเพิ่มค่ะ",
        "en": "Salid will start reading that channel and its history by itself right away. "
        "Private channels you don't want Salid to know about: just leave them out.",
    },
    "cmd_channels": _bi(
        "ดูห้องที่สลิดยังมองไม่เห็น และวิธีเพิ่ม", "Channels Salid cannot see, and how to add it"
    ),
    "setup_denied": {
        "th": "คำสั่งนี้ใช้ได้เฉพาะเจ้าของเซิร์ฟเวอร์หรือผู้ที่มีสิทธิ์จัดการเซิร์ฟเวอร์ค่ะ",
        "en": "This command is only for the server owner or people who can manage the server.",
    },
    "setup_message": {
        "th": "ลิงก์เชื่อม {name} กับศูนย์บัญชาการของสลิด 🔗\n{url}\n"
        "เปิดลิงก์แล้วเข้าสู่ระบบด้วยบัญชี Google ของคุณ ลิงก์ใช้ได้ครั้งเดียวและหมดอายุใน 15 นาที "
        "อย่าส่งต่อให้คนอื่นค่ะ",
        "en": "Link to connect {name} to Salid's command center 🔗\n{url}\n"
        "Open the link and sign in with your Google account. It works once and expires in 15 "
        "minutes. Please do not share it.",
    },
    "setup_failed": {
        "th": "สร้างลิงก์ไม่สำเร็จ ลองพิมพ์ /ตั้งค่า อีกครั้งภายหลังค่ะ",
        "en": "Could not create the link. Please try /setup again later.",
    },
    "mode_denied": {
        "th": "ปรับโหมดได้เฉพาะผู้ดูแลเซิร์ฟเวอร์หรือผู้บริหารค่ะ",
        "en": "Only server admins or executives can change the mode.",
    },
    "mode_failed": {
        "th": "ตั้งโหมดไม่สำเร็จ ลองใหม่อีกครั้งภายหลังนะคะ",
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
        "th": "ผู้ดูแลหรือผู้บริหารเปลี่ยนได้ด้วย /โหมด แล้วเลือกโหมดค่ะ",
        "en": "Admins or executives can change it with /mode and then pick a mode.",
    },
    "mode_set": {"th": "ตั้งห้องนี้เป็น {label} แล้วค่ะ", "en": "This channel is now set to {label}."},
    "lang_current": {
        "th": "ภาษาที่สลิดตอบในห้องนี้ตอนนี้: {label}",
        "en": "Salid's reply language in this channel right now: {label}",
    },
    "lang_footer": {
        "th": "ผู้ดูแลหรือผู้บริหารเปลี่ยนได้ด้วย /ภาษา แล้วเลือกภาษาค่ะ "
        "(หรือแท็กสลิดแล้วพิมพ์ “speak English” / “พูดไทย”)",
        "en": "Admins or executives can change it with /language and then pick a language "
        "(or tag Salid and say “speak English” / “พูดไทย”).",
    },
    "lang_denied": {
        "th": "เปลี่ยนภาษาได้เฉพาะผู้ดูแลเซิร์ฟเวอร์หรือผู้บริหารค่ะ",
        "en": "Only server admins or executives can change the language.",
    },
    "lang_failed": {
        "th": "ตั้งภาษาไม่สำเร็จ ลองใหม่อีกครั้งภายหลังนะคะ",
        "en": "Could not set the language. Please try again later.",
    },
    "lang_set": {
        "th": "ตั้งภาษาของห้องนี้เป็น {label} แล้วค่ะ",
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
        "th": "สลิดไม่ยุ่งกับห้องนี้เลย (ไม่อ่านใบเสร็จ ไม่ตอบ แท็กก็ไม่ตอบ) และไม่บันทึกข้อความ",
        "en": "Salid stays out of this channel completely (no receipts, no replies, even if tagged) "
        "and nothing is recorded",
    },
    "mode_help_listen": {
        "th": "ฟังเงียบ ไม่พูดเอง อ่านใบเสร็จแล้วแปะอีโมจิ ตอบเมื่อถูกแท็กเท่านั้น (ไม่ตอบเมื่อเรียกชื่อเฉยๆ) "
        "สลิดจำข้อความของห้องนี้ไว้ {days} วัน",
        "en": "Listens quietly and never speaks unprompted: reads receipts with an emoji and replies "
        "only when tagged (not when just called by name). Salid remembers this channel's messages "
        "for {days} days",
    },
    "mode_help_call": {
        "th": "ตอบเมื่อถูกแท็กหรือถูกเรียกชื่อ แล้วคุยต่อได้ 3 นาทีโดยไม่ต้องแท็ก สลิดจำข้อความของห้องนี้ไว้ {days} วัน",
        "en": "Replies when tagged or called by name, then keeps chatting for 3 minutes without a tag. "
        "Salid remembers this channel's messages for {days} days",
    },
    "mode_help_join": {
        "th": "เหมือนโหมดเรียกแล้วคุย และสลิดอาจเข้าร่วมวงเองเมื่อมีคำถามที่ช่วยได้ (จำกัดชั่วโมงละ 3 ครั้ง) "
        "สลิดจำข้อความของห้องนี้ไว้ {days} วัน",
        "en": "Like Call, and Salid may chime in on its own when it can help with a question "
        "(at most 3 times an hour). Salid remembers this channel's messages for {days} days",
    },
    "mode_help_chat": {
        "th": "ตอบทุกข้อความในห้อง สลิดจำข้อความของห้องนี้ไว้ {days} วัน",
        "en": "Replies to every message in the channel. Salid remembers this channel's messages "
        "for {days} days",
    },
    "cmd_memory": _bi("ดูหรือตั้งจำนวนวันที่สลิดจำข้อความ", "View or set how long Salid remembers chats"),
    "cmd_memory_arg": _bi("จำนวนวัน 1-3650 (ไม่ใส่ = ดูสถานะ)", "Days 1-3650 (empty = view status)"),
    "memory_on": {"th": "เปิด", "en": "on"},
    "memory_off": {"th": "ปิดอยู่", "en": "off"},
    "memory_status": {
        "th": "ความจำของทีม: {state} · เก็บข้อความไว้ {days} วัน",
        "en": "Team memory: {state} · messages are kept for {days} days",
    },
    "memory_import": {
        "th": "นำเข้าประวัติแล้ว {done}/{total} ห้อง, {messages} ข้อความ, รูป {images}/{cap}",
        "en": "History imported: {done}/{total} channels, {messages} messages, images {images}/{cap}",
    },
    "memory_modes": {
        "th": "ห้องที่ถูกบันทึก: ทุกโหมดยกเว้น {off} (โหมด {recorded})",
        "en": "Recorded channels: every mode except {off} (modes {recorded})",
    },
    "memory_footer": {
        "th": "ผู้ดูแลหรือผู้บริหารเปลี่ยนจำนวนวันได้ด้วย /ความจำ days:<จำนวนวัน> "
        "ปิดห้องเดียวด้วย /โหมด 🔇 หรือปิดทั้งเซิร์ฟเวอร์ที่หน้าศูนย์บัญชาการ",
        "en": "Admins or executives can change the days with /memory days:<number>. Switch one "
        "channel off with /mode 🔇, or the whole server in the command center.",
    },
    "memory_set": {
        "th": "ตั้งให้สลิดจำข้อความไว้ {days} วันแล้วค่ะ (ใช้กับข้อความเก่าด้วยในการล้างรอบถัดไป)",
        "en": "Salid will now keep messages for {days} days (this also applies to existing "
        "messages at the next daily cleanup).",
    },
    "memory_denied": {
        "th": "เปลี่ยนจำนวนวันที่จำข้อความได้เฉพาะผู้ดูแลเซิร์ฟเวอร์หรือผู้บริหารค่ะ",
        "en": "Only server admins or executives can change how long messages are kept.",
    },
    "memory_bad_days": {
        "th": "ใส่จำนวนวันเป็นตัวเลข {lo} ถึง {hi} ค่ะ",
        "en": "Please give a number of days from {lo} to {hi}.",
    },
    "memory_failed": {
        "th": "ตั้งค่าไม่สำเร็จ ลองใหม่อีกครั้งภายหลังนะคะ",
        "en": "Could not save the setting. Please try again later.",
    },
    "memory_notice": {
        "th": "📝 ต่อไปนี้สลิดจะจำข้อความในห้องนี้ไว้ {days} วัน เพื่อช่วยตอบคำถามอย่าง “เมื่อวานคุยเรื่องอะไร” "
        "และจะให้ข้อมูลจากห้องนี้เฉพาะกับคนที่เห็นห้องนี้เท่านั้น "
        "ถ้าไม่ต้องการ พิมพ์ /โหมด แล้วเลือก 🔇 ปิด",
        "en": "📝 From now on Salid will remember messages in this channel for {days} days to help "
        "answer questions like “what did we discuss yesterday?”, and will only share information "
        "from it with people who can see this channel. To stop, type /mode and pick 🔇 Off.",
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
    "cmd_summary": _bi("สรุปสิ่งที่คุยกันในห้อง", "Summarize what was discussed"),
    "cmd_summary_period": _bi("ช่วงเวลา (ค่าเริ่มต้น วันนี้)", "Period (default today)"),
    "cmd_summary_channel": _bi(
        "ห้องที่จะสรุป (ค่าเริ่มต้น ห้องนี้)", "Channel to summarize (default this one)"
    ),
    "cmd_summary_all": _bi("ทุกห้องที่ฉันเห็น", "All channels I can see"),
    "cmd_summary_topic": _bi("เรื่องที่อยากให้เน้น (ไม่บังคับ)", "Topic to focus on (optional)"),
    "cmd_summary_private": _bi("เห็นคนเดียว", "Only visible to me"),
    "period_today": _bi("วันนี้", "Today"),
    "period_yesterday": _bi("เมื่อวาน", "Yesterday"),
    "period_7d": _bi("7 วัน", "Last 7 days"),
    "period_30d": _bi("30 วัน", "Last 30 days"),
    "menu_summarize": {"th": "สรุปด้วยสลิด", "en": "Summarize with Salid"},
    "menu_translate": {"th": "แปลด้วยสลิด", "en": "Translate with Salid"},
    "translate_nothing": _bi(
        "ข้อความนี้ไม่มีตัวอักษรให้แปลค่ะ", "There is no text in that message to translate."
    ),
    "translate_failed": _bi(
        "แปลไม่สำเร็จค่ะ ลองใหม่อีกครั้งนะคะ", "I couldn't translate that, please try again."
    ),
    "summary_q_channel": {
        "th": "สรุปสิ่งที่คุยกันใน #{channel} ช่วง {dates}: หัวข้อ, สิ่งที่ตัดสินใจ, ใครรับงานอะไร, "
        "ตัวเลขสำคัญ, เรื่องที่ค้าง",
        "en": "Summarize what was discussed in #{channel} for {dates}: topics, decisions made, "
        "who took which task, key numbers, open items",
    },
    "summary_q_all": {
        "th": "สรุปสิ่งที่คุยกันในทุกห้องที่ฉันเห็น ช่วง {dates} แยกตามห้อง: หัวข้อ, สิ่งที่ตัดสินใจ, "
        "ใครรับงานอะไร, ตัวเลขสำคัญ, เรื่องที่ค้าง",
        "en": "Summarize what was discussed in all the channels I can see for {dates}, per channel: "
        "topics, decisions made, who took which task, key numbers, open items",
    },
    "summary_focus": {"th": " [เน้น: {topic}]", "en": " [focus: {topic}]"},
    "summary_dates_one": {"th": "{start} (เขตเวลา {tz})", "en": "{start} (time zone {tz})"},
    "summary_dates_range": {
        "th": "{start} ถึง {end} (เขตเวลา {tz})",
        "en": "{start} to {end} (time zone {tz})",
    },
    "summary_msg_q": {
        "th": "สรุปข้อความนี้ของ {author}: หัวข้อ ประเด็นสำคัญ ตัวเลข และสิ่งที่ต้องทำต่อ "
        "(ถ้ามีไฟล์แนบหรือลิงก์ ให้อ่านและสรุปด้วย อย่าบันทึกเป็นใบเสร็จ)",
        "en": "Summarize this message from {author}: topic, key points, numbers and next steps "
        "(read and summarize any attached files or links too; do not record anything as a receipt)",
    },
    "summary_msg_text": {"th": "ข้อความที่ต้องสรุป:\n{text}", "en": "Message to summarize:\n{text}"},
    "summary_unavailable": {
        "th": "สรุปไม่ได้ตอนนี้ เพราะการถามตอบหรือความจำของทีมถูกปิดอยู่ (เปิดได้ที่หน้าศูนย์บัญชาการ)",
        "en": "I can't summarize right now: Q&A or team memory is switched off (turn it on in the command center).",
    },
    "summary_no_access": {
        "th": "คุณไม่มีสิทธิ์เห็นห้องนั้น หรือสลิดไม่ได้จดห้องนั้นไว้ค่ะ",
        "en": "You can't see that channel, or Salid isn't recording it.",
    },
    "summary_nothing": {
        "th": "ไม่มีข้อความที่สรุปได้ค่ะ (ข้อความว่าง ไม่มีไฟล์หรือลิงก์)",
        "en": "There is nothing to summarize (no text, files or links).",
    },
    # ---------------------------------------------------------------- meeting
    "meet_announce": {
        "th": "สลิดเข้ามาฟังและจดบันทึกการประชุมค่ะ 🎙️ "
        "(เก็บเป็นข้อความในหน่วยความจำเท่านั้น ไม่บันทึกเสียง) "
        "เรียก “สลิด” แล้วถามได้เลย หรือพิมพ์ /ออกประชุม เพื่อสรุปและให้สลิดออก",
        "en": "Salid has joined to listen and take meeting notes 🎙️ "
        "(kept as text in memory only, no audio is recorded). "
        "Say “Salid” and ask away, or type /leave-meeting to get the summary and let me go.",
    },
    # Fallbacks only: the consent question is normally generated. ค่ะ / คะ = voice particles.
    "meet_consent_ask_1": {
        "th": "สลิดขออนุญาตจดประชุมนี้นะคะ ถ้าไม่สะดวกบอกได้เลยค่ะ",
        "en": "Salid would like to take notes of this meeting. Just say if you'd rather not.",
    },
    "meet_consent_ask_2": {
        "th": "ขออนุญาตจดบันทึกการประชุมนะคะ ถ้าไม่ต้องการ บอกสลิดได้เลยนะคะ",
        "en": "May I take notes here? Tell me if that's not okay.",
    },
    "meet_consent_ask_3": {
        "th": "สลิดนั่งฟังและจดการประชุมอยู่นะคะ ถ้าไม่สะดวกบอกได้เลยค่ะ",
        "en": "I'm listening in to take notes. Let me know if you'd prefer I didn't.",
    },
    "meet_dismiss_ack": {"th": "ได้ค่ะ", "en": "Okay!"},
    "meet_not_recording": {
        "th": "งั้นสลิดไม่จดนะคะ มีอะไรเรียกได้เลยค่ะ",
        "en": "Then I won't take notes. Just call me if you need anything.",
    },
    "meet_consent_no_1": {
        "th": "ได้ค่ะ งั้นสลิดขอตัวนะคะ",
        "en": "Okay, I'll leave you to it.",
    },
    "meet_consent_no_2": {
        "th": "รับทราบค่ะ สลิดไม่จดแล้วนะคะ ขอตัวก่อน",
        "en": "Understood, no notes. I'll head out.",
    },
    "meet_consent_no_3": {
        "th": "ไม่เป็นไรค่ะ สลิดออกไปก่อนนะคะ",
        "en": "No problem, I'll step out.",
    },
    "minutes_set_spoken": {
        "th": "ได้เลยค่ะ เดี๋ยวส่งสรุปไปที่ห้อง {channel} นะคะ",
        "en": "Sure, I'll send the minutes to {channel}.",
    },
    "minutes_not_found_spoken": {
        "th": "หาห้องที่บอกไม่เจอค่ะ ลองบอกชื่อห้องอีกทีนะคะ",
        "en": "I couldn't find that channel. Please say its name again.",
    },
    "minutes_ambiguous_spoken": {
        "th": "มีหลายห้องที่ชื่อใกล้เคียงกันค่ะ ช่วยบอกชื่อให้ชัดขึ้นหน่อยนะคะ",
        "en": "Several channels match. Please be more specific.",
    },
    "minutes_not_allowed_spoken": {
        "th": "ห้องนั้นคนในที่ประชุมบางคนยังมองไม่เห็นค่ะ เลยส่งสรุปไปที่นั่นไม่ได้",
        "en": "Not everyone here can see that channel, so I can't send the minutes there.",
    },
    "minutes_forwarded": {
        "th": "📝 สรุปการประชุม #{room} ({when}) ส่งต่อโดย {who}\n",
        "en": "📝 Minutes of the meeting in #{room} ({when}), forwarded by {who}\n",
    },
    "meet_not_in_voice": {
        "th": "ต้องเข้าห้องเสียง (Voice Channel) ก่อน แล้วค่อยเรียกสลิดเข้าประชุมนะคะ",
        "en": "Please join a voice channel first, then ask Salid to join the meeting.",
    },
    "meet_already": {
        "th": "สลิดกำลังอยู่ในการประชุมของเซิร์ฟเวอร์นี้แล้วค่ะ ({channel}) พิมพ์ /ออกประชุม เพื่อจบก่อนนะคะ",
        "en": "Salid is already in a meeting on this server ({channel}). Type /leave-meeting to end it first.",
    },
    "meet_joining": {"th": "กำลังเข้า", "en": "joining now"},
    "meet_none": {
        "th": "ตอนนี้สลิดไม่ได้อยู่ในการประชุมค่ะ",
        "en": "Salid is not in a meeting right now.",
    },
    "meet_leave_denied": {
        "th": "ต้องอยู่ในห้องประชุมเดียวกัน (หรือเป็นผู้ดูแลเซิร์ฟเวอร์) ถึงจะสั่งให้สลิดออกได้ค่ะ",
        "en": "You need to be in the same meeting channel (or be a server admin) to tell Salid to leave.",
    },
    "meet_off": {
        "th": "ตอนนี้ปิดฟังการประชุมอยู่ค่ะ เปิดได้ที่หน้าศูนย์บัญชาการของสลิด",
        "en": "Meeting listening is switched off right now. You can turn it on in Salid's command center.",
    },
    "meet_connect_failed": {
        "th": "เข้าห้องเสียงไม่สำเร็จค่ะ ลองใหม่อีกครั้งภายหลังนะคะ",
        "en": "Could not join the voice channel. Please try again later.",
    },
    "meet_joined": {"th": "สลิดเข้าห้อง {channel} แล้วค่ะ", "en": "Salid has joined {channel}."},
    "meet_left": {
        "th": "สลิดออกจากการประชุมแล้วค่ะ กำลังสรุปให้ในแชตของห้องเสียง",
        "en": "Salid has left the meeting. The summary is coming in the voice channel's chat.",
    },
    "meet_wake_empty": {"th": "คะ? ว่ามาได้เลยค่ะ", "en": "Yes? Go ahead."},
    "meet_wake_spoken": {"th": "ค่ะ? ว่ามาได้เลยค่ะ", "en": "Yes? Go ahead."},
    "meet_greet_spoken": {"th": "สวัสดีค่ะ มีอะไรให้ช่วยคะ", "en": "Hi! How can I help?"},
    "meet_bye_spoken": {"th": "แล้วเจอกันนะคะ", "en": "Bye for now!"},
    "meet_no_transcript": {
        "th": "จบการประชุมแล้วค่ะ แต่ยังไม่ได้ยินเนื้อหาให้สรุปเลย 🙂",
        "en": "The meeting is over, but I did not hear anything to summarize 🙂",
    },
    "meet_summary_failed": {
        "th": "จบการประชุมแล้วค่ะ แต่สรุปไม่สำเร็จ ขออภัยด้วยนะคะ",
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
    "end_timeout": {"th": "ประชุมครบ 6 ชั่วโมง", "en": "the meeting reached 6 hours"},
    "meet_rejoined": {
        "th": "สลิดกลับเข้ามาจดประชุมต่อแล้วค่ะ (หลุดไปครู่หนึ่ง)",
        "en": "Salid is back and taking notes again (it dropped out for a moment).",
    },
    "meet_reconnected": {
        "th": "สลิดหลุดไปครู่หนึ่ง แต่กลับเข้าห้องและจดประชุมต่อแล้วค่ะ",
        "en": "Salid dropped out for a moment but is back in the room and still taking notes.",
    },
    "meet_reconnect_failed": {
        "th": "สลิดต่อเข้าห้องเสียงกลับไม่ได้ค่ะ ขอปิดการประชุมและสรุปให้ตอนนี้",
        "en": "Salid could not reconnect to the voice channel, so it is ending the meeting now.",
    },
    "meet_max_warning": {
        "th": "ประชุมใกล้ครบ 6 ชั่วโมงแล้ว สลิดจะออกและสรุปในอีก 10 นาทีค่ะ",
        "en": "The meeting hits the 6-hour limit in 10 minutes; Salid will then leave and summarize.",
    },
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
    "digest_dups": {
        "th": "🔁 อาจซ้ำ {n} รายการ รวม {amount} รอตรวจ",
        "en": "🔁 Possible duplicates: {n} receipts, {amount} awaiting review",
    },
    "delete_denied": {
        "th": "ลบได้เฉพาะคนที่สลิดตอบอยู่ ผู้จัดการเซิร์ฟเวอร์ หรือผู้บริหารเท่านั้นค่ะ",
        "en": "Only the person Salid was answering, a server manager or an executive can delete this.",
    },
    "receipt_void_ack": {
        "th": "ยกเลิกใบเสร็จ #{id} แล้ว (กู้คืนได้)",
        "en": "Receipt #{id} is cancelled (it can be restored).",
    },
    "cancel_done": {
        "th": "ยกเลิกงานใหญ่ที่กำลังทำอยู่แล้วค่ะ",
        "en": "The running big task has been cancelled.",
    },
    "research_ack": {
        "th": "กำลังค้นข้อมูลให้นะคะ ประมาณ 1–2 นาที เสร็จแล้วสลิดส่งรายงานพร้อมไฟล์ PDF ให้ที่ห้องนี้ค่ะ (สั่ง “ยกเลิกงาน” ได้ถ้าไม่ต้องการแล้ว)",
        "en": "I'm researching this now, about 1-2 minutes. I'll post a report with a PDF here when it's done (say “cancel task” to stop).",
    },
    "research_busy": {
        "th": "ตอนนี้สลิดกำลังค้นข้อมูลเชิงลึกเรื่องอื่นอยู่ค่ะ รอให้เสร็จก่อนนะคะ",
        "en": "I'm already running another deep research here. Please wait until it finishes.",
    },
    "research_limit": {
        "th": "วันนี้ค้นข้อมูลเชิงลึกครบ {n} ครั้งแล้วค่ะ ลองใหม่พรุ่งนี้ หรือให้สลิดค้นแบบสั้นๆ (web_search) แทนก็ได้นะคะ",
        "en": "The daily limit of {n} deep researches has been reached. Try again tomorrow, or I can do a quick web search instead.",
    },
    "research_done": {
        "th": "ค้นข้อมูลเสร็จแล้วค่ะ รายงานฉบับเต็มอยู่ในไฟล์ PDF:",
        "en": "The research is done. The full report is in the attached PDF:",
    },
    "research_failed": {
        "th": "ขอโทษค่ะ ค้นข้อมูลไม่สำเร็จ ลองใหม่อีกครั้งนะคะ",
        "en": "Sorry, the research failed. Please try again.",
    },
    "research_cancelled": {
        "th": "หยุดการค้นข้อมูลให้แล้วค่ะ",
        "en": "Research stopped.",
    },
    "research_nothing": {
        "th": "## สรุป\nค้นเว็บแล้วไม่ได้ข้อมูลที่ใช้ได้ค่ะ ลองถามให้เฉพาะเจาะจงขึ้นอีกนิดนะคะ",
        "en": "## Summary\nThe web search returned nothing usable. Try a more specific question.",
    },
    "research_sources": {"th": "แหล่งอ้างอิง", "en": "Sources"},
    "research_raw_head": {
        "th": "ผลการค้นหา (ยังไม่ได้สรุป)",
        "en": "Search results (not summarised)",
    },
    "research_title": {"th": "รายงานการค้นคว้า", "en": "Research report"},
    "research_dated": {"th": "ข้อมูล ณ วันที่ {at}", "en": "Information as of {at}"},
    "cancel_none": {
        "th": "ตอนนี้ไม่มีงานใหญ่ที่กำลังทำอยู่ค่ะ",
        "en": "There is no big task running right now.",
    },
    "cancel_denied": {
        "th": "ยกเลิกงานใหญ่ได้เฉพาะคนที่สั่งงานนั้น ผู้จัดการเซิร์ฟเวอร์ หรือผู้บริหารเท่านั้นค่ะ",
        "en": "Only the person who started the task, a server manager or an executive can cancel it.",
    },
    "cmd_cancel": _bi("ยกเลิกงานใหญ่ที่กำลังทำอยู่", "Cancel the running big task"),
    "corr_void_done": {
        "th": "🗑️ หักใบเสร็จ #{id} ({vendor} {total}) ออกจากยอดแล้ว (ถือว่าซ้ำ) ตอบ “กู้คืน” เพื่อนำกลับมา",
        "en": "🗑️ Receipt #{id} ({vendor} {total}) is deducted from the totals (marked as a "
        'duplicate). Reply "restore" to bring it back.',
    },
    "corr_restore_done": {
        "th": "♻️ กู้คืนใบเสร็จ #{id} ({vendor} {total}) กลับเข้ายอดแล้ว",
        "en": "♻️ Receipt #{id} ({vendor} {total}) is counted again.",
    },
    "corr_already_void": {
        "th": "ใบเสร็จ #{id} ถูกหักออกไปแล้ว ตอบ “กู้คืน” เพื่อนำกลับมา",
        "en": 'Receipt #{id} is already deducted. Reply "restore" to bring it back.',
    },
    "corr_not_void": {
        "th": "ใบเสร็จ #{id} ยังนับอยู่ในยอด ไม่ต้องกู้คืน",
        "en": "Receipt #{id} is still counted, nothing to restore.",
    },
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
    "fwd_header": {
        "th": "📎 ใบเสร็จจาก {user} (ส่งมาใน #{channel}) · [ข้อความต้นฉบับ]({url})",
        "en": "📎 Receipt from {user} (sent in #{channel}) · [original message]({url})",
    },
    "fwd_header_dm": {
        "th": "📎 ใบเสร็จจาก {user} (ส่งมาทาง DM)",
        "en": "📎 Receipt from {user} (sent by DM)",
    },
    "fwd_ack": {
        "th": "ส่งต่อไปที่ #{home} ให้แล้วค่ะ (ข้อความเดิมลบได้ถ้าต้องการ)",
        "en": "Forwarded to #{home} (you can delete your original message if you like).",
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
        "th": "ขออภัย ตอบไม่ได้ในตอนนี้ ลองถามใหม่อีกครั้งนะคะ",
        "en": "Sorry, I can't answer right now. Please try asking again.",
    },
    # ---------------------------------------------------------------- deep tasks
    "deep_disabled": {
        "th": "ฟีเจอร์งานใหญ่ยังไม่ได้เปิดใช้งานค่ะ (ยังไม่ได้ตั้งค่า SALID_GEMINI_CLI_PATH) ลองถามคำถามสั้นๆ แทนได้เลย",
        "en": "Big tasks are not enabled yet (SALID_GEMINI_CLI_PATH is not set). Please ask a "
        "short question instead.",
    },
    "deep_not_exec": {
        "th": "ขออภัยค่ะ งานใหญ่ใช้ได้เฉพาะผู้บริหารเท่านั้น ถามคำถามสั้นๆ เรื่องใบเสร็จของคุณได้เลยค่ะ",
        "en": "Sorry, big tasks are for executives only. You can still ask short questions about "
        "your own receipts.",
    },
    "deep_cli_failed": {
        "th": "Gemini CLI ทำงานไม่สำเร็จ ลองใหม่อีกครั้งภายหลังค่ะ",
        "en": "Gemini CLI failed. Please try again later.",
    },
    "deep_no_result": {
        "th": "ไม่ได้รับผลลัพธ์จากงานนี้ค่ะ",
        "en": "This task returned no result.",
    },
    "deep_ack": {
        "th": "เดี๋ยวสลิดไล่ดูให้นะคะ งานนี้ใหญ่หน่อย อาจใช้เวลา 1-5 นาที เสร็จแล้วส่งผลในห้องนี้ค่ะ",
        "en": "I'll go through it for you. This one's a big job, so it may take 1-5 minutes; I'll "
        "post the result right here.",
    },
    "deep_busy": {
        "th": "ตอนนี้สลิดกำลังทำงานใหญ่อีกงานอยู่ค่ะ รอให้เสร็จก่อนแล้วสั่งใหม่อีกครั้งนะคะ",
        "en": "Salid is busy with another big task. Please wait for it to finish and ask again.",
    },
    "deep_usage": {
        "th": "บอกโจทย์ด้วยค่ะ เช่น งานใหญ่: วิเคราะห์ค่าใช้จ่ายเดือนนี้ หาหมวดที่โตผิดปกติ แล้วเขียนรายงาน",
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
    "pdf_generated": {
        "th": "สร้างเมื่อ {at}",
        "en": "Generated {at}",
    },
    "pdf_page": {
        "th": "หน้า {n}/{total}",
        "en": "Page {n}/{total}",
    },
    "pdf_title_expense_summary": {
        "th": "สรุปค่าใช้จ่าย",
        "en": "Expense summary",
    },
    "pdf_title_receipt_list": {
        "th": "รายการใบเสร็จ",
        "en": "Receipt list",
    },
    "pdf_title_meeting_minutes": {
        "th": "รายงานการประชุม",
        "en": "Meeting minutes",
    },
    "pdf_title_custom": {
        "th": "เอกสาร",
        "en": "Document",
    },
    "pdf_period": {
        "th": "ช่วงเวลา: {start} ถึง {end}",
        "en": "Period: {start} to {end}",
    },
    "pdf_scope_own": {
        "th": "แสดงเฉพาะใบเสร็จของ {name}",
        "en": "Showing only receipts submitted by {name}",
    },
    "pdf_totals": {
        "th": "ยอดรวมแยกตามสกุลเงิน",
        "en": "Totals by currency",
    },
    "pdf_currency": {
        "th": "สกุลเงิน",
        "en": "Currency",
    },
    "pdf_amount": {
        "th": "จำนวนเงิน",
        "en": "Amount",
    },
    "pdf_no_fx": {
        "th": "ไม่รวมต่างสกุลเงินเข้าด้วยกันและไม่แปลงอัตราแลกเปลี่ยน",
        "en": "Currencies are never added together and no exchange rate is applied.",
    },
    "pdf_by_category": {
        "th": "แยกตามหมวด",
        "en": "By category",
    },
    "pdf_by_submitter": {
        "th": "แยกตามผู้ส่ง",
        "en": "By submitter",
    },
    "pdf_name": {
        "th": "ชื่อ",
        "en": "Name",
    },
    "pdf_receipts": {
        "th": "รายการใบเสร็จ ({n} ใบ)",
        "en": "Receipts ({n})",
    },
    "pdf_date": {
        "th": "วันที่",
        "en": "Date",
    },
    "pdf_vendor": {
        "th": "ร้านค้า",
        "en": "Vendor",
    },
    "pdf_category": {
        "th": "หมวด",
        "en": "Category",
    },
    "pdf_cur": {
        "th": "สกุล",
        "en": "Cur.",
    },
    "pdf_flags": {
        "th": "หมายเหตุ",
        "en": "Flags",
    },
    "pdf_flag_low": {
        "th": "ความมั่นใจต่ำ",
        "en": "low confidence",
    },
    "pdf_flag_dup": {
        "th": "อาจซ้ำกับ #{id}",
        "en": "possible duplicate of #{id}",
    },
    "pdf_total_line": {
        "th": "รวม: {total}",
        "en": "Total: {total}",
    },
    "pdf_void_note": {
        "th": "ไม่รวมใบเสร็จที่ถูกหักออก (void) {n} ใบ",
        "en": "Excludes {n} voided receipt(s).",
    },
    "pdf_when": {
        "th": "เวลา: {when}",
        "en": "When: {when}",
    },
    "pdf_attendees": {
        "th": "ผู้เข้าร่วม: {names}",
        "en": "Attendees: {names}",
    },
    "pdf_attendance": {
        "th": "การเข้าร่วมและเหตุผลที่ไม่เข้า",
        "en": "Attendance and absence reasons",
    },
    "pdf_summary": {
        "th": "สรุปการประชุม",
        "en": "Summary",
    },
    "pdf_no_summary": {
        "th": "(ยังไม่มีสรุป)",
        "en": "(no summary yet)",
    },
    "sched_invite_dm": {
        "th": "สวัสดีค่ะ มีนัดประชุมใหม่ในเซิร์ฟเวอร์ {guild} จาก {by}\n📅 {title}\n🕑 {when}\n🔗 {link}",
        "en": "Hi! You are invited to a meeting on {guild} by {by}\n📅 {title}\n🕑 {when}\n🔗 {link}",
    },
    "place_offer_dm": {
        "th": "สลิดเห็นว่าคุณพิมพ์ที่อยู่ “{name}” ไว้ในห้องแชต: {address}\nให้สลิดจำไว้เป็นที่ส่วนตัวของคุณไหมคะ (เห็นได้คนเดียว แชร์ให้ใครทีหลังได้ ลืมได้ทุกเมื่อ) ตอบ “จำไว้” หรือ “ไม่ต้อง” ได้เลยค่ะ สลิดจะถามแค่ครั้งนี้ครั้งเดียว",
        "en": "Salid noticed you wrote the address “{name}” in a channel: {address}\nShall Salid remember it as your own private place (only you can see it; you can share it later and I can forget it any time)? Just reply “remember it” or “no thanks”. I will only ask this once.",
    },
    "bf_leave_line": {
        "th": "ควรออกจาก{origin} ประมาณ {leave} (ใช้เวลา ~{minutes} นาที) ไปประชุม “{title}” ที่{place}",
        "en": "Leave {origin} around {leave} (~{minutes} min) for “{title}” at {place}",
    },
    "bf_weather_line": {
        "th": "อากาศที่{place}วันนี้: {summary}",
        "en": "Weather at {place} today: {summary}",
    },
    "place_event_dm": {
        "th": "📍 {where}\n🗺️ {maps}",
        "en": "📍 {where}\n🗺️ {maps}",
    },
    "sched_cancelled_dm": {
        "th": "การประชุม “{title}” ถูกยกเลิกแล้วค่ะ",
        "en": "The meeting “{title}” has been cancelled.",
    },
    "sched_moved_dm": {
        "th": "การประชุม “{title}” เลื่อนเวลาเป็น {when}\n🔗 {link}",
        "en": "The meeting “{title}” was moved to {when}\n🔗 {link}",
    },
    "sched_remind_pre": {
        "th": "⏰ อีก {mins} นาทีถึงการประชุม “{title}” ({when})\nเข้าร่วม: {link}",
        "en": "⏰ “{title}” starts in {mins} minutes ({when})\nJoin: {link}",
    },
    "sched_remind_start": {
        "th": "🔔 การประชุม “{title}” เริ่มแล้วค่ะ\nเข้าร่วม: {link}",
        "en": "🔔 “{title}” is starting now\nJoin: {link}",
    },
    "sched_reply_if_stuck": {
        "th": "ถ้าเข้าไม่ได้ ตอบกลับสลิดได้เลยค่ะ",
        "en": "If you can't join, just reply to me.",
    },
    "sched_nudge_dm": {
        "th": "การประชุม “{title}” เริ่มแล้วค่ะ มีปัญหาอะไรหรือเปล่า? ถ้าหาลิงก์ไม่เจอ กดเข้าตรงนี้ได้เลย: {link}\nถ้าเข้าไม่ได้ ตอบกลับสลิดได้เลยค่ะ",
        "en": "“{title}” has started. Is something wrong? If you can't find the link, join here: {link}\nIf you can't make it, just reply to me.",
    },
    "sched_attended": {
        "th": "✅ เข้าร่วม: {names}",
        "en": "✅ Attended: {names}",
    },
    "sched_absent": {
        "th": "🚫 ไม่ได้เข้า: {names}",
        "en": "🚫 Absent: {names}",
    },
    "sched_summary_link": {
        "th": "📝 สรุปการประชุม: {url}",
        "en": "📝 Meeting summary: {url}",
    },
    "sched_absence_note": {
        "th": "{name} แจ้งว่า: {reason} (การประชุม “{title}”)",
        "en": "{name} says: {reason} (meeting “{title}”)",
    },
    "sched_absence_ack": {
        "th": "รับทราบค่ะ แจ้งผู้จัดการประชุม “{title}” แล้ว",
        "en": "Got it, I let the organiser of “{title}” know.",
    },
    "perm_missing": {
        "th": "ตอนนี้สลิดยังขาดสิทธิ์ {what} อยู่ค่ะ ให้แอดมินเปิดลิงก์นี้ครั้งเดียวแล้วเลือกเซิร์ฟเวอร์นี้ก็เรียบร้อย {url}",
        "en": "I'm missing the {what} permission here. An admin just needs to open this link once "
        "and pick this server: {url}",
    },
    "perm_missing_nolink": {
        "th": "ตอนนี้สลิดยังขาดสิทธิ์ {what} อยู่ค่ะ ให้แอดมินเปิดสิทธิ์นี้ให้บทบาทของสลิดหน่อยนะคะ",
        "en": "I'm missing the {what} permission here. An admin needs to allow it for my role.",
    },
    "email_not_configured": {
        "th": "ยังไม่ได้ตั้งค่าการส่งอีเมล (บันทึกร่างไว้แล้ว แต่ยังไม่ได้ส่งจริง)",
        "en": "Email sending is not configured yet (the draft was recorded but nothing was sent).",
    },
    "meet_announce_scheduled": {
        "th": "สลิดเข้ามาจดประชุมตามนัดค่ะ 🎙️ (เก็บเป็นข้อความในความจำเท่านั้น ไม่บันทึกเสียง)",
        "en": "Salid has joined to take notes for the scheduled meeting 🎙️ (kept as text only, no audio is recorded).",
    },
    "meet_announce_auto": {
        "th": "สลิดเข้ามาจดประชุมนะคะ ไม่ต้องการพิมพ์ “@Salid ออกประชุม” ได้เลย "
        "(เก็บเป็นข้อความในความจำเท่านั้น ไม่บันทึกเสียง)",
        "en": "Salid has joined to take meeting notes. If you don't need me, just type "
        "“@Salid leave meeting” (kept as text only, no audio is recorded).",
    },
    "meet_preempted": {
        "th": "ขอไปช่วยประชุมที่นัดไว้ก่อนนะคะ ส่วนที่คุยมาสรุปไว้ให้แล้ว",
        "en": "I have to go help with a scheduled meeting, sorry! The summary of what was said is below.",
    },
    "end_preempted": {
        "th": "สลิดต้องไปประชุมที่สำคัญกว่า",
        "en": "Salid moved to a more important meeting",
    },
    "meet_busy": {
        "th": "ตอนนี้ร่างของสลิดติดประชุมอยู่ครบแล้วค่ะ ({rooms}) ถ้ามีผู้ช่วยเสียงเพิ่มจะแบ่งร่างไปได้อีก",
        "en": "All of Salid's voices are in meetings right now ({rooms}). "
        "More voice helpers would let Salid split further.",
    },
    "meet_leave_which": {
        "th": "สลิดอยู่ในหลายห้อง ให้พิมพ์สั่งจากในห้องนั้นหรือเข้าห้องที่ต้องการก่อนนะคะ: {rooms}",
        "en": "Salid is in several rooms. Type it from that room or join the room first: {rooms}",
    },
    "voice_bodies_title": {
        "th": "ร่างเสียงของสลิด",
        "en": "Salid's voices",
    },
    "voice_body_in": {"th": "อยู่ในเซิร์ฟเวอร์นี้", "en": "in this server"},
    "voice_body_out": {"th": "ยังไม่ได้เชิญเข้าเซิร์ฟเวอร์นี้", "en": "not invited to this server"},
    "voice_body_down": {"th": "ยังไม่พร้อมใช้งาน", "en": "unavailable"},
    # ---------------------------------------------------------------- coordinator: disclosure
    "disc_where_dm": {"th": "บันทึก DM ส่วนตัว", "en": "a private DM note"},
    "disc_desc": {
        "th": "{n} ข้อความที่คุณเขียนไว้ ช่วงวันที่ {span} ({where})",
        "en": "{n} message(s) you wrote, dated {span} ({where})",
    },
    "disc_ask": {
        "th": "{asker} ถามสลิดเรื่อง “{topic}”\nคำถามเดิม: “{question}”\n"
        "สลิดมีข้อมูลที่เป็นของคุณที่เกี่ยวข้อง: {description}\n"
        "(สลิดยังไม่ได้บอกเนื้อหาให้ใคร) อนุญาตให้สลิดบอก {asker} เรื่องนี้ไหมคะ?",
        "en": "{asker} asked Salid about “{topic}”.\nOriginal question: “{question}”\n"
        "Salid holds related information of yours: {description}\n"
        "(Nothing has been shared yet.) May Salid tell {asker} about it?",
    },
    "disc_btn_ok": {"th": "อนุญาต", "en": "Approve"},
    "disc_btn_no": {"th": "ไม่อนุญาต", "en": "Decline"},
    "disc_owner_approved": {
        "th": "✅ อนุญาตแล้ว สลิดจะบอก {asker} เฉพาะข้อมูลชุดนี้ค่ะ",
        "en": "✅ Approved. Salid will tell {asker} only this information.",
    },
    "disc_owner_declined": {
        "th": "🚫 ไม่อนุญาต สลิดจะไม่บอก {asker} ค่ะ",
        "en": "🚫 Declined. Salid will not tell {asker}.",
    },
    "disc_handled": {
        "th": "คำขอนี้ตัดสินไปแล้วค่ะ",
        "en": "This request has already been decided.",
    },
    "disc_not_yours": {
        "th": "คำขอนี้ไม่ใช่ของคุณค่ะ",
        "en": "This request is not yours to answer.",
    },
    "disc_asker_declined": {
        "th": "เรื่อง “{topic}”: เจ้าของข้อมูลยังไม่อนุญาตให้บอกค่ะ ขอโทษด้วยนะคะ ลองถามเจ้าของโดยตรงได้เลย",
        "en": "About “{topic}”: the owner of that information did not allow sharing it, sorry. "
        "You can ask them directly.",
    },
    "disc_asker_approved": {
        "th": "เรื่อง “{topic}”: เจ้าของข้อมูลอนุญาตแล้วค่ะ\n\n",
        "en": "About “{topic}”: the owner approved sharing it.\n\n",
    },
    "disc_asker_failed": {
        "th": "เจ้าของข้อมูลอนุญาตแล้ว แต่สลิดเรียบเรียงคำตอบไม่สำเร็จ ลองถามอีกครั้งทาง DM ได้เลยค่ะ",
        "en": "The owner approved, but Salid could not put the answer together. Please ask again in a DM.",
    },
    # ---------------------------------------------------------------- coordinator: DMs
    "dm_notice": {
        "th": "ℹ️ เรื่องงานที่คุณบอกสลิดทาง DM สลิดจะจดไว้ให้ทีม และผู้ที่ตำแหน่ง (role) สูงกว่าคุณอาจเห็นได้ "
        "ถ้าเรื่องไหนเป็นความลับ พิมพ์ “เรื่องนี้เก็บเป็นความลับ” แล้วสลิดจะเก็บให้คุณคนเดียวค่ะ",
        "en": "ℹ️ Work information you tell Salid in a DM is noted for the team, and people with a "
        "higher role than yours may see it. If something is confidential, type “keep this private” "
        "and Salid will keep it to you alone.",
    },
    "dm_private_marked": {
        "th": "รับทราบค่ะ เรื่องนี้เก็บเป็นความลับ เห็นเฉพาะคุณคนเดียว สลิดจะไม่เอาไปรายงานหรือบอกใครคะ 🔒",
        "en": "Understood. That stays confidential: only you can see it, and Salid will not report "
        "it or tell anyone. 🔒",
    },
    "dm_private_note": {
        "th": "(ผู้ใช้ระบุว่าเรื่องนี้เป็นความลับ เก็บไว้ให้เขาคนเดียว)",
        "en": "(The user said this is confidential: keep it to them alone.)",
    },
    # ---------------------------------------------------------------- coordinator: follow-ups
    "fu_header": {
        "th": "สวัสดีค่ะ {name} สลิดขอติดตามงานที่ถึงกำหนดหรือเลยกำหนดค่ะ:",
        "en": "Hi {name}, Salid is following up on tasks that are due or overdue:",
    },
    "fu_line": {"th": "• #{id} {title} (กำหนด {due})", "en": "• #{id} {title} (due {due})"},
    "fu_footer": {
        "th": "ตอบกลับที่นี่ได้เลยว่าถึงไหนแล้ว เช่น “#{id} เสร็จแล้ว” หรือ “ติดเรื่อง…” แล้วสลิดจะอัปเดตให้ค่ะ",
        "en": "Just reply here with the status, e.g. “#{id} done” or “blocked on…”, and Salid will "
        "update it.",
    },
    # ---------------------------------------------------------------- coordinator: reports
    "rep_title": {"th": "**📋 รายงานประจำวันที่ {day}**", "en": "**📋 Daily report for {day}**"},
    "rep_brief": {"th": "**ภาพรวมบริษัท**", "en": "**Company brief**"},
    "rep_spend": {
        "th": "- ใช้จ่ายวันนี้ {today} ({n} ใบ) · เดือนนี้ {month}",
        "en": "- Spent today {today} ({n} receipts) · this month {month}",
    },
    "rep_review": {
        "th": "- ใบเสร็จที่ต้องตรวจ {n} ใบ · อาจซ้ำ {d} ใบ",
        "en": "- Receipts to review: {n} · possible duplicates: {d}",
    },
    "rep_meetings": {"th": "**ประชุมวันนี้**", "en": "**Meetings today**"},
    "rep_people": {"th": "**งานของแต่ละคน**", "en": "**Work by person**"},
    "rep_person": {
        "th": "- **{name}**: {msgs} ข้อความ · งานค้าง {open}",
        "en": "- **{name}**: {msgs} messages · {open} open task(s)",
    },
    "rep_opened": {"th": "เปิดใหม่: ", "en": "opened: "},
    "rep_done": {"th": "เสร็จ: ", "en": "done: "},
    "rep_overdue": {"th": "เลยกำหนด: ", "en": "overdue: "},
    "rep_risks": {"th": "**ความเสี่ยง / สิ่งที่ควรตาม**", "en": "**Risks / to follow up**"},
    "pr_title": {"th": "**วันนี้ของคุณ ({day})**", "en": "**Your day ({day})**"},
    "pr_where": {"th": "คุณเขียนใน: {where}", "en": "You wrote in: {where}"},
    "pr_none": {
        "th": "วันนี้ยังไม่พบข้อความของคุณในห้องที่สลิดจดไว้ค่ะ",
        "en": "Salid has not recorded any message from you today.",
    },
    "pr_done": {"th": "งานที่เสร็จวันนี้: ", "en": "Tasks done today: "},
    "pr_open": {"th": "งานที่ยังค้าง:", "en": "Open tasks:"},
    "report_channel_topic": {
        "th": "รายงานประจำวันจากสลิด สำหรับเจ้าของ แอดมิน และผู้บริหาร",
        "en": "Daily reports from Salid for the owner, admins and executives",
    },
    "report_need_channel": {
        "th": "สลิดอยากส่งรายงานเย็นให้ผู้บริหารในห้องส่วนตัว “{name}” แต่ไม่มีสิทธิ์จัดการช่อง (Manage Channels) "
        "รบกวนให้สิทธิ์สลิด หรือสร้างห้องที่เห็นเฉพาะผู้บริหารแล้วเลือกเป็นห้องรายงานในหน้า Hub ค่ะ",
        "en": "Salid would like to post the evening report for leadership in a private channel "
        "“{name}” but lacks the Manage Channels permission. Please grant it, or create a "
        "leadership-only channel and pick it as the report channel in the hub.",
    },
    "report_denied": {
        "th": "รายงานบริษัทดูได้เฉพาะเจ้าของ แอดมิน และผู้บริหารค่ะ (ดูของตัวเองด้วย /รายงาน ของฉัน)",
        "en": "The company report is for the owner, admins and executives (use /report mine for yours).",
    },
    "report_bad_date": {
        "th": "วันที่ต้องเป็นรูปแบบ ปี-เดือน-วัน เช่น 2026-10-14 ค่ะ",
        "en": "The date must look like 2026-10-14.",
    },
    "report_off": {
        "th": "ตอนนี้ปิดความจำหรือรายงานอยู่ค่ะ เปิดได้ที่หน้าศูนย์บัญชาการของสลิด",
        "en": "Memory or reports are switched off. You can turn them on in Salid's command center.",
    },
    "report_posted": {
        "th": "ส่งรายงานวันที่ {day} เข้า {channel} แล้วค่ะ",
        "en": "Posted the report for {day} in {channel}.",
    },
    "report_failed": {
        "th": "สร้างรายงานไม่สำเร็จ ลองใหม่ภายหลังค่ะ",
        "en": "Could not build the report. Please try again later.",
    },
    "cmd_report": _bi("รายงานประจำวันจากสลิด", "Daily report from Salid"),
    "cmd_report_date": _bi("วันที่ (ปี-เดือน-วัน) ไม่ใส่ = วันนี้", "Date (YYYY-MM-DD), default today"),
    "cmd_report_mine": _bi("ดูของฉันเอง", "Show only my own day"),
    # ---------------------------------------------------------------- briefings and alerts
    "bf_title_morning": {
        "th": "**สรุปเช้าของคุณ {name} ({day})**",
        "en": "**Good morning {name}, your brief for {day}**",
    },
    "bf_title_evening": {
        "th": "**สรุปเย็นของคุณ {name} ({day})**",
        "en": "**Evening brief for {name} ({day})**",
    },
    "bf_meetings": {"th": "**นัดวันนี้**", "en": "**Today's meetings**"},
    "bf_meetings_tomorrow": {"th": "**นัดพรุ่งนี้**", "en": "**Tomorrow's meetings**"},
    "bf_meeting_line": {"th": "• {time} {title} {link}", "en": "• {time} {title} {link}"},
    "bf_tasks_due": {"th": "**งานที่ครบกำหนดหรือเลยกำหนด**", "en": "**Tasks due or overdue**"},
    "bf_task_line": {
        "th": "• #{id} {title} (กำหนด {due})",
        "en": "• #{id} {title} (due {due})",
    },
    "bf_docs": {"th": "**เอกสารสำคัญที่ได้รับ**", "en": "**Important documents received**"},
    "bf_doc_line": {
        "th": "• {kind}: {summary} ({channel}) {link}",
        "en": "• {kind}: {summary} ({channel}) {link}",
    },
    "bf_asks": {"th": "**เรื่องที่รอคุณตอบ**", "en": "**Waiting for your reply**"},
    "bf_ask_line": {
        "th": "• {who} ({channel}): {text} {link}",
        "en": "• {who} ({channel}): {text} {link}",
    },
    "bf_attended": {"th": "**ประชุมที่คุณเข้าวันนี้**", "en": "**Meetings you attended today**"},
    "bf_new_tasks": {"th": "**งานใหม่ที่ได้รับวันนี้**", "en": "**New tasks assigned today**"},
    "bf_done": {"th": "**งานที่เสร็จวันนี้**", "en": "**Done today**"},
    "bf_open": {"th": "**งานที่ยังค้าง**", "en": "**Still open**"},
    "bf_footer": {
        "th": "ปรับหรือปิดสรุปนี้ได้โดยบอกสลิด เช่น “ไม่ต้องส่งสรุปเย็น”",
        "en": "To change or stop these briefs just tell Salid, e.g. “no evening brief”.",
    },
    # ---------------------------------------------------------------- catch-up ("what did I miss")
    "cu_title": {
        "th": "**สิ่งที่คุณพลาดไปตั้งแต่ {since}**",
        "en": "**What you missed since {since}**",
    },
    "cu_none": {
        "th": "ตั้งแต่ {since} ยังไม่มีอะไรใหม่ที่เกี่ยวกับคุณในห้องที่คุณอ่านได้ค่ะ",
        "en": "Nothing new for you since {since} in the channels you can read.",
    },
    "cu_mentions": {"th": "**มีคนแท็กหรือพูดถึงคุณ**", "en": "**Mentions of you**"},
    "cu_mention_line": {
        "th": "• {who} ({channel}): {text} {link}",
        "en": "• {who} ({channel}): {text} {link}",
    },
    "cu_tasks": {"th": "**งานของคุณ**", "en": "**Your tasks**"},
    "cu_task_line": {
        "th": "• #{id} {title} (กำหนด {due}) {flag} {link}",
        "en": "• #{id} {title} (due {due}) {flag} {link}",
    },
    "cu_late": {"th": "เลยกำหนด", "en": "overdue"},
    "cu_new": {"th": "ใหม่", "en": "new"},
    "cu_decisions": {"th": "**มติและเรื่องที่ตกลงกัน**", "en": "**Decisions and agreements**"},
    "cu_missed": {"th": "**ประชุมที่คุณไม่ได้เข้า**", "en": "**Meetings you missed**"},
    "cu_upcoming": {"th": "**นัดที่กำลังจะถึง**", "en": "**Coming up**"},
    "cu_meeting_line": {"th": "• {time} {title} {link}", "en": "• {time} {title} {link}"},
    "cu_no_minutes": {"th": "(ยังไม่มีสรุปประชุม)", "en": "(no minutes yet)"},
    "cu_channels": {"th": "**ไฮไลต์รายห้อง**", "en": "**Highlights by channel**"},
    "cu_channel_line": {
        "th": "• {channel} ({n} ข้อความ): {text} {link}",
        "en": "• {channel} ({n} messages): {text} {link}",
    },
    "cu_files": {"th": "**ไฟล์ที่เกี่ยวกับคุณ**", "en": "**Files for you**"},
    "cu_ask_more": {
        "th": "อยากดูละเอียดห้องไหนบอกได้นะคะ",
        "en": "Tell me which channel you want in detail.",
    },
    "cu_s_mentions": {"th": "แท็กคุณ {n}", "en": "{n} mention(s)"},
    "cu_s_tasks": {"th": "งาน {n}", "en": "{n} task(s)"},
    "cu_s_missed": {"th": "ประชุมที่พลาด {n}", "en": "{n} missed meeting(s)"},
    "cu_s_decisions": {"th": "มติ {n}", "en": "{n} decision(s)"},
    "cu_off": {
        "th": "ตอนนี้ปิดสรุปสิ่งที่พลาดหรือความจำอยู่ค่ะ เปิดได้ที่หน้าศูนย์บัญชาการของสลิด",
        "en": "Catch-up or memory is switched off. You can turn it on in Salid's command center.",
    },
    "cu_failed": {
        "th": "สรุปสิ่งที่พลาดไม่สำเร็จ ลองใหม่ภายหลังนะคะ",
        "en": "Could not build your catch-up. Please try again later.",
    },
    "cu_bad_since": {
        "th": "ไม่เข้าใจช่วงเวลานี้ค่ะ ลองเช่น “เมื่อวาน” “3 วัน” “จันทร์” หรือ 2026-10-12",
        "en": "I did not understand that period. Try “yesterday”, “3 days”, “monday” or 2026-10-12.",
    },
    "cmd_catchup": _bi("สรุปสิ่งที่คุณพลาดไป", "Catch up on what you missed"),
    "cmd_catchup_since": _bi(
        "ตั้งแต่เมื่อไร เช่น เมื่อวาน, 3 วัน, จันทร์ (ไม่ใส่ = ตั้งแต่ที่คุณหายไป)",
        "Since when, e.g. yesterday, 3 days, monday (empty = since you were last around)",
    ),
    "alert_doc": {
        "th": "**เอกสารสำคัญ** ({kind}): {summary}\nห้อง {channel} · {link}",
        "en": "**Important document** ({kind}): {summary}\nChannel {channel} · {link}",
    },
    "ask_reminder": {
        "th": "ขอเตือนเบาๆ ค่ะ {who} ฝากเรื่องนี้ไว้ในห้อง {channel} และยังไม่มีคำตอบจากคุณ: “{text}” {link}",
        "en": "A gentle reminder: {who} asked you something in {channel} and there is no reply "
        "from you yet: “{text}” {link}",
    },
    # ---------------------------------------------------------------- deliver_file
    "deliver_footer": {
        "th": "ส่งต่อจาก {who} · [ต้นฉบับ]({url})",
        "en": "Forwarded from {who} · [original]({url})",
    },
    "deliver_oversize": {
        "th": "ไฟล์ใหญ่เกินกว่าจะแนบ (เกิน 25 MB) ดูได้ที่ต้นฉบับ: {url}",
        "en": "The file is too large to attach (over 25 MB); see the original: {url}",
    },
    "deliver_email_body": {
        "th": "เรียนทุกท่าน\nขอส่งไฟล์ {name} มาพร้อมอีเมลนี้ค่ะ",
        "en": "Hello,\nPlease find the file {name} attached.",
    },
    "deliver_footer_plain": {"th": "ส่งต่อจาก {who}", "en": "Forwarded from {who}"},
    "deliver_original": {"th": "[ต้นฉบับ]({url})", "en": "[original]({url})"},
    "deliver_dm_from": {
        "th": "ส่งต่อจาก {who} ({server}) ผ่านสลิดค่ะ",
        "en": "Forwarded by Salid from {who} ({server}).",
    },
    "deliver_warn_audience": {
        "th": "เอกสารนี้มาจาก #{source} ซึ่ง {who} มองไม่เห็น ยืนยันจะส่งไหมคะ",
        "en": "This document comes from #{source}, which {who} cannot see. Send it anyway?",
    },
    # ---------------------------------------------------------------- polls and forms
    "poll_result": {
        "th": "ผลโหวต “{question}”: {winner} ชนะค่ะ ({votes} จาก {total} โหวต)",
        "en": "Poll result for “{question}”: {winner} wins ({votes} of {total} votes).",
    },
    "poll_result_tie": {
        "th": "ผลโหวต “{question}”: เสมอกันระหว่าง {winners} ({votes} โหวตเท่ากัน) ค่ะ",
        "en": "Poll result for “{question}”: a tie between {winners} ({votes} votes each).",
    },
    "poll_result_none": {
        "th": "โหวต “{question}” จบแล้ว แต่ยังไม่มีใครโหวตเลยค่ะ",
        "en": "The poll “{question}” has ended, but nobody voted.",
    },
    "poll_memory": {
        "th": "[ผลโหวต] {line}",
        "en": "[Poll result] {line}",
    },
    "form_invite": {
        "th": "สวัสดีค่ะ {creator} ({server}) ฝากให้สลิดถามเรื่อง “{title}” มี {n} ข้อ "
        "ตอบกลับในแชตนี้ได้เลยค่ะ (พิมพ์ “ข้าม” ถ้าไม่อยากตอบข้อไหน หรือ “ไม่ขอตอบ” เพื่อหยุด) "
        "ตอบภายใน {deadline} นะคะ",
        "en": "Hi! {creator} ({server}) asked me to collect answers for “{title}”: {n} questions. "
        "Just reply here (type “skip” to pass a question, or “stop” to opt out). "
        "Please answer by {deadline}.",
    },
    "form_question": {
        "th": "ข้อ {i}/{n}: {question}",
        "en": "Question {i}/{n}: {question}",
    },
    "form_thanks": {
        "th": "ขอบคุณค่ะ ตอบครบแล้ว สลิดส่งคำตอบให้ {creator} เรียบร้อยค่ะ",
        "en": "Thank you, that's everything. I'll pass your answers to {creator}.",
    },
    "form_declined_ack": {
        "th": "รับทราบค่ะ ไม่ถามเรื่องนี้อีกแล้วนะคะ",
        "en": "Got it, I won't ask about this again.",
    },
    "form_reminder": {
        "th": "เตือนความจำเบาๆ ค่ะ “{title}” ของ {creator} ใกล้ถึงกำหนด ({deadline}) แล้ว "
        "ยังตอบไม่ครบนะคะ ตอบกลับในแชตนี้ได้เลยค่ะ",
        "en": "Gentle reminder: “{title}” from {creator} closes soon ({deadline}) and I still "
        "have some unanswered questions. Just reply here.",
    },
    "form_progress": {
        "th": "“{title}” ตอบแล้ว {done} จาก {total} คน",
        "en": "“{title}”: {done} of {total} people have answered.",
    },
    "form_closed": {
        "th": "แบบสอบถาม “{title}” ปิดแล้วค่ะ ตอบแล้ว {done} จาก {total} คน "
        "ขอดูผลได้เลยนะคะ (บอกว่า “ขอผลแบบสอบถาม”)",
        "en": "The form “{title}” has closed: {done} of {total} people answered. "
        "Ask me for the results any time.",
    },
    "form_col_name": {"th": "ชื่อ", "en": "Name"},
    "form_col_state": {"th": "สถานะ", "en": "Status"},
    "form_pdf_people": {"th": "คำตอบรายคน", "en": "Answers by person"},
    "form_results_dm": {
        "th": "ผลแบบสอบถามส่งทาง DM แล้วค่ะ",
        "en": "I sent the results to you by DM.",
    },
    # ---------------------------------------------------------------- wiki (salid.wiki_bot)
    "wiki_post_footer": {
        "th": "อัปเดตล่าสุด {date} · เขียนโดย {who} · สลิด",
        "en": "Last updated {date} · written by {who} · via Salid",
    },
}
