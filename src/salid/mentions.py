"""Tagging people: the model writes `@[display name]`, the bot turns it into a real Discord
mention (`<@id>` / `<@&id>`) before sending, and only people it can resolve unambiguously get one.

Who actually gets pinged is decided here too (never by model text alone): users and mentionable
roles named in the answer, `@everyone` / `@here` only for a manager or executive who explicitly
asked to notify everyone.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import discord

MENTION_RE = re.compile(r"@\[([^\[\]\n]{1,80})\]")
BROADCAST_RE = re.compile(r"@(everyone|here)\b")
ASKED_EVERYONE_RE = re.compile(
    r"@everyone|@here|\beveryone\b|\beverybody\b|all\s+hands|ทุกคน|ทั้งทีม|ทั้งเซิร์ฟ", re.I
)
MAX_PEOPLE = 50


def _names(member) -> list[str]:
    out = []
    for attr in ("display_name", "global_name", "name"):
        v = getattr(member, attr, None)
        if v:
            out.append(str(v).casefold())
    return out


def _members(guild) -> list:
    found = getattr(guild, "members", None) or []
    return list(found.values()) if isinstance(found, dict) else list(found)


def match_members(guild, name: str, pool: Iterable | None = None) -> list:
    """Members `name` could mean: the exact (case-insensitive) matches on display name, global
    name or username if any, else everyone whose name starts with it. Bots are skipped."""
    want = name.strip().lstrip("@").casefold()
    if not want:
        return []
    members = [m for m in (pool if pool is not None else _members(guild)) if not m.bot]
    exact = [m for m in members if want in _names(m)]
    if exact:
        return exact
    return [m for m in members if any(n.startswith(want) for n in _names(m))]


def find_member(guild, name: str, pool: Iterable | None = None):
    """The one member `name` means, or None when unknown or ambiguous."""
    found = match_members(guild, name, pool)
    return found[0] if len(found) == 1 else None


def find_role(guild, name: str):
    """A mentionable role with that name (case-insensitive, unique), else None."""
    want = name.strip().lstrip("@").casefold()
    roles = [
        r
        for r in getattr(guild, "roles", [])
        if str(getattr(r, "name", "")).casefold() == want and getattr(r, "mentionable", False)
    ]
    return roles[0] if len(roles) == 1 else None


def wants_everyone(requester_text: str, can_broadcast: bool) -> bool:
    """A manager / executive explicitly asked to notify everyone."""
    return bool(can_broadcast and ASKED_EVERYONE_RE.search(requester_text or ""))


RAW_MENTION_RE = re.compile(r"<@([!&]?)(\d{1,25})>")


def defang_raw_mentions(text: str, guild) -> str:
    """Rewrite `<@id>` / `<@!id>` / `<@&id>` the model wrote itself into plain `@DisplayName`
    (no ping). Run BEFORE resolve_mentions, so only our own `@[name]` resolution pings."""

    def one(m: re.Match) -> str:
        kind, num = m.group(1), int(m.group(2))
        name = None
        if guild is not None:
            if kind == "&":
                role = guild.get_role(num) if hasattr(guild, "get_role") else None
                name = getattr(role, "name", None)
            else:
                member = guild.get_member(num) if hasattr(guild, "get_member") else None
                name = getattr(member, "display_name", None)
        return f"@{name}" if name else "@" + ("role" if kind == "&" else "user")

    return RAW_MENTION_RE.sub(one, text or "")


def resolve_mentions(text: str, guild, *, allow_everyone: bool = False, pool=None) -> str:
    """Replace `@[name]` with real mentions where unambiguous (members first, then mentionable
    roles); anything else becomes plain `@name`. `@everyone` / `@here` are stripped unless
    `allow_everyone`."""

    def one(m: re.Match) -> str:
        name = m.group(1)
        if guild is not None:
            member = find_member(guild, name, pool)
            if member is not None:
                return f"<@{member.id}>"
            role = find_role(guild, name)
            if role is not None:
                return f"<@&{role.id}>"
        return f"@{name}"

    out = MENTION_RE.sub(one, text or "")
    if not allow_everyone:
        out = BROADCAST_RE.sub(lambda m: m.group(1), out)
    return out


def allowed_mentions_for(text: str, *, allow_everyone: bool = False) -> discord.AllowedMentions:
    """Ping the users and the (resolved, so mentionable) roles in `text`; never a reply ping."""
    roles = [discord.Object(id=int(i)) for i in dict.fromkeys(re.findall(r"<@&(\d+)>", text))]
    return discord.AllowedMentions(
        users=True, roles=roles or False, everyone=allow_everyone, replied_user=False
    )


def tag_candidates(
    channel, recent_authors: Iterable[str] = (), limit: int = MAX_PEOPLE
) -> list[str]:
    """Display names Salid may tag: the recent chat's people first, then members who can see
    the channel (no bots), at most `limit`. Names only, never ids."""
    out: list[str] = []
    for n in [
        *recent_authors,
        *(m.display_name for m in getattr(channel, "members", []) if not m.bot),
    ]:
        if n and n not in out and n not in ("สลิด", "Salid"):
            out.append(n)
        if len(out) >= limit:
            break
    return out
