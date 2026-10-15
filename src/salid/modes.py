"""Per-channel modes: how much Salid takes part in a Discord channel."""

from __future__ import annotations

import re

from .i18n import t

MODES = ("off", "listen", "call", "join", "chat")
DEFAULT_MODE = "call"
# Created on join (needs Manage Channels); a channel with this name answers every message.
CHAT_CHANNEL_NAME = "คุยกับน้องสลิด"


def mode_label(mode: str, lang: str = "th") -> str:
    return t(f"mode_label_{mode}", lang)


def mode_help(mode: str, lang: str = "th") -> str:
    return t(f"mode_help_{mode}", lang)


def mode_choice_name(mode: str) -> str:
    """Bilingual label for the slash command choice, e.g. "🙋 เรียกแล้วคุย · Call"."""
    th, en = mode_label(mode, "th"), mode_label(mode, "en")
    return f"{th} · {en.split(' ', 1)[1]}"


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


_LEAD_JUNK = " \t\n/,.:;!-–—…"
_SNOOZE_RE = re.compile(
    r"^(เงียบ\s?ก่อน|เงียบ|พัก\s?ก่อน"
    r"|(?:be\s+)?quiet\b|shh+\b|shush\b|mute\b|hush\b|snooze\b)",
    re.I,
)
_UNSNOOZE_RE = re.compile(
    r"^(คุย\s?ได้\s?แล้ว|กลับ\s?มา\s?คุย\s?ได้"
    r"|you\s+can\s+talk\b|talk\s+again\b|unmute\b|unsnooze\b|wake\s+up\b|come\s+back\b)",
    re.I,
)


def parse_snooze_command(text: str) -> str | None:
    """ "เงียบก่อน" / "be quiet" -> "snooze", "คุยได้แล้ว" / "you can talk" -> "unsnooze", anything
    else None. Only the start of the (mention-stripped) message counts."""
    s = (text or "").strip().strip(_LEAD_JUNK)
    if not s or len(s) > 40:
        return None
    if _SNOOZE_RE.match(s):
        return "snooze"
    if _UNSNOOZE_RE.match(s):
        return "unsnooze"
    return None
