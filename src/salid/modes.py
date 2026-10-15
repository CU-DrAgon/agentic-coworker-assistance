"""Per-channel modes: how much Salid takes part in a Discord channel."""

from __future__ import annotations

import re

MODES = ("off", "listen", "call", "join", "chat")
DEFAULT_MODE = "call"
# Created on join (needs Manage Channels); a channel with this name answers every message.
CHAT_CHANNEL_NAME = "คุยกับน้องสลิด"
MODE_LABELS = {
    "off": "🔇 ปิด",
    "listen": "👂 ฟังเงียบ",
    "call": "🙋 เรียกแล้วคุย",
    "join": "🎧 ร่วมวง",
    "chat": "💬 ตอบทุกข้อความ",
}
MODE_HELP = {
    "off": "สลิดไม่ยุ่งกับห้องนี้เลย (ไม่อ่านใบเสร็จ ไม่ตอบ แท็กก็ไม่ตอบ)",
    "listen": "ทำงานเบื้องหลังเท่านั้น อ่านใบเสร็จแล้วแปะอีโมจิ ไม่ตอบเป็นข้อความ",
    "call": "ตอบเมื่อถูกแท็กหรือถูกเรียกชื่อ แล้วคุยต่อได้ 3 นาทีโดยไม่ต้องแท็ก",
    "join": "เหมือนโหมดเรียกแล้วคุย และสลิดอาจเข้าร่วมวงเองเมื่อมีคำถามที่ช่วยได้ (จำกัดชั่วโมงละ 3 ครั้ง)",
    "chat": "ตอบทุกข้อความในห้อง",
}


def is_chat_channel_name(name: str | None) -> bool:
    return CHAT_CHANNEL_NAME in (name or "")


def resolve_mode(
    channel_id: int,
    channel_name: str | None,
    *,
    channel_modes: dict[str, str],
    ask_channel_ids: set[int],
    default_mode: str = DEFAULT_MODE,
) -> str:
    """Explicit per-channel setting, else ask channel -> chat, else the Salid chat channel name
    -> chat, else the guild default."""
    explicit = channel_modes.get(str(channel_id))
    if explicit in MODES:
        return explicit
    if channel_id in ask_channel_ids or is_chat_channel_name(channel_name):
        return "chat"
    return default_mode if default_mode in MODES else DEFAULT_MODE


_SNOOZE_RE = re.compile(r"^(เงียบ\s?ก่อน|เงียบ|พัก\s?ก่อน)")
_UNSNOOZE_RE = re.compile(r"^(คุย\s?ได้\s?แล้ว|กลับ\s?มา\s?คุย\s?ได้)")


def parse_snooze_command(text: str) -> str | None:
    """ "เงียบก่อน" -> "snooze", "คุยได้แล้ว" -> "unsnooze", anything else None. Only the start of
    the (mention-stripped) message counts."""
    t = (text or "").strip().lstrip("/").strip()
    if not t or len(t) > 40:
        return None
    if _SNOOZE_RE.match(t):
        return "snooze"
    if _UNSNOOZE_RE.match(t):
        return "unsnooze"
    return None
