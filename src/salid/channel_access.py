"""Which channels can Salid not read? Pure helpers shared by `/ห้อง`, `/ตั้งค่า` and the hub sync.

A channel is "hidden" when Salid lacks View Channel or Read Message History there. Nothing here
talks to Discord beyond `channel.permissions_for(me)`, so it works with plain fakes in tests.
"""

from __future__ import annotations

from .i18n import t

MAX_LISTED = 15  # channel mentions shown in a chat message
DISCORD_LIMIT = 2000


def access_flags(channel, me) -> tuple[bool, bool]:
    """(can view the channel, can read its history). Unknown permissions count as allowed."""
    try:
        perms = channel.permissions_for(me)
    except Exception:
        return True, True
    visible = bool(getattr(perms, "view_channel", True))
    history = bool(getattr(perms, "read_message_history", True))
    return visible, history


def is_public(channel, guild) -> bool:
    """Everyone in the server (the @everyone role) can see the channel and read its history.
    Unknown permissions count as NOT public: sharing across workspaces needs a positive answer."""
    role = getattr(guild, "default_role", None)
    if role is None:
        return False
    try:
        perms = channel.permissions_for(role)
    except Exception:
        return False
    return bool(getattr(perms, "view_channel", False)) and bool(
        getattr(perms, "read_message_history", False)
    )


def public_channel_ids(guild) -> list[str]:
    return [str(c.id) for c in _channels(guild) if is_public(c, guild)]


def _channels(guild) -> list:
    """Text and announcement, forum and voice channels (categories and stages are skipped)."""
    out: list = []
    for attr in ("text_channels", "forums", "voice_channels"):
        out += list(getattr(guild, attr, None) or [])
    return out


def hidden_channels(guild, me) -> list[dict]:
    """Channels Salid cannot fully read, sorted by category then position. Each entry:
    `id`, `name`, `category` ("" when none), `missing` ("view" | "history") and `reason`
    (Thai text: "ไม่เห็นห้อง" or "อ่านย้อนหลังไม่ได้")."""
    rows = []
    for ch in _channels(guild):
        visible, history = access_flags(ch, me)
        if visible and history:
            continue
        cat = getattr(ch, "category", None)
        missing = "view" if not visible else "history"
        rows.append(
            (
                (
                    -1 if cat is None else getattr(cat, "position", 0),
                    getattr(cat, "name", "") if cat else "",
                    getattr(ch, "position", 0),
                    ch.id,
                ),
                {
                    "id": ch.id,
                    "name": ch.name,
                    "category": getattr(cat, "name", "") if cat else "",
                    "missing": missing,
                    "reason": t(f"hidden_reason_{missing}", "th"),
                },
            )
        )
    return [e for _, e in sorted(rows, key=lambda x: x[0])]


def split_message(text: str, limit: int = DISCORD_LIMIT) -> list[str]:
    """Split on line boundaries into chunks of at most `limit` characters."""
    chunks, cur = [], ""
    for line in text.split("\n"):
        while len(line) > limit:  # a single over-long line: hard cut
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if cur and len(cur) + 1 + len(line) > limit:
            chunks.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


def hidden_channels_messages(entries: list[dict], lang: str = "en") -> list[str]:
    """The friendly "channels Salid cannot see" message(s), each within Discord's limit."""
    if not entries:
        return [t("hidden_none", lang)]
    shown, rest = entries[:MAX_LISTED], len(entries) - MAX_LISTED
    lines = [t("hidden_header", lang, n=len(entries))]
    lines += [f"• <#{e['id']}> ({t('hidden_reason_' + e['missing'], lang)})" for e in shown]
    if rest > 0:
        lines.append(t("hidden_more", lang, m=rest))
    lines += ["", t("hidden_steps", lang), "", t("hidden_footer", lang)]
    return split_message("\n".join(lines))
