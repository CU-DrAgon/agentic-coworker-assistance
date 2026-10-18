"""Org-wide scope for a verified 1:1: which sources may Salid use for ONE member of an org?

Design: docs/superpowers/specs/2026-10-18-salid-workspace-parity-design.md, "Scope across
platforms". The rule: an answer may use a source only if the asker may read it AND so can everyone
who reads the answer. A 1:1 chat with a verified member has exactly one reader, the member, so
everything that member may read in every workspace of the org is allowed. Anything else (a group,
a channel, an unverified sender, an unknown audience) gets no widening at all and keeps today's
single-tenant scope.

This module is pure: no network, no clock, no stores. The platform specific "what may this
account read in that workspace?" questions are answered by injected `RoomProvider`s
(`platforms/org_sources.py`); a provider that cannot answer (error, no token, workspace not
reachable from this process) EXCLUDES that workspace: fail closed, never a guess.

Also here: the pure Discord permission resolver (`readable_channels`), which follows Discord's
documented algorithm over data fetched by REST (guild roles, the member's roles, channel
overwrites): owner and Administrator see everything, otherwise base permissions are the @everyone
role plus the member's roles, then the channel overwrites apply in the documented order
(@everyone, then all the member's roles together, then the member itself, each one first removing
its denies and then adding its allows). A channel is readable with View Channel and Read Message
History; the text chat of a voice or stage channel is read the same way (Connect is only for
joining the call, not for reading).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .orgs import TENANT_BOUND_PLATFORMS, Member, Org, Workspace, namespaced_id

logger = logging.getLogger(__name__)

# platforms that have a chat account and a ledger / memory of their own
CHAT_PLATFORMS = ("discord", "slack", "line", "telegram", "whatsapp")
PLATFORM_NAMES = {
    "discord": "Discord",
    "slack": "Slack",
    "line": "LINE",
    "telegram": "Telegram",
    "whatsapp": "WhatsApp",
}

# ---------------------------------------------------------------- Discord permissions
ADMINISTRATOR = 1 << 3
VIEW_CHANNEL = 1 << 10
READ_MESSAGE_HISTORY = 1 << 16
CONNECT = 1 << 20  # joining a call; NOT needed to read a voice channel's text chat
ALL_PERMISSIONS = (1 << 64) - 1

# channel types whose message history Salid records (threads count under their parent)
TEXT_TYPES = frozenset({0, 5, 15, 16})  # text, announcement, forum, media
VOICE_TYPES = frozenset({2, 13})  # voice, stage: their text chat
OVERWRITE_ROLE, OVERWRITE_MEMBER = 0, 1


def _perm(value: Any) -> int:
    """Discord sends permission bit sets as decimal strings."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def base_permissions(guild: Mapping, member: Mapping) -> int:
    """The member's guild-wide permissions: owner / Administrator = everything, else the
    @everyone role OR-ed with the permissions of every role the member has."""
    guild_id = str(guild.get("id"))
    user_id = str((member.get("user") or {}).get("id") or member.get("id") or "")
    if user_id and str(guild.get("owner_id")) == user_id:
        return ALL_PERMISSIONS
    roles = {str(r.get("id")): _perm(r.get("permissions")) for r in guild.get("roles") or ()}
    perms = roles.get(guild_id, 0)  # the @everyone role has the guild's id
    for rid in member.get("roles") or ():
        perms |= roles.get(str(rid), 0)
    return ALL_PERMISSIONS if perms & ADMINISTRATOR else perms


def channel_permissions(
    guild: Mapping, member: Mapping, channel: Mapping, base: int | None = None
) -> int:
    """The member's permissions in one channel (Discord's documented overwrite order)."""
    perms = base_permissions(guild, member) if base is None else base
    if perms & ADMINISTRATOR or perms == ALL_PERMISSIONS:
        return ALL_PERMISSIONS
    guild_id = str(guild.get("id"))
    user_id = str((member.get("user") or {}).get("id") or member.get("id") or "")
    member_roles = {str(r) for r in member.get("roles") or ()}
    overwrites = list(channel.get("permission_overwrites") or ())

    def find(kind: int, oid: str) -> Mapping | None:
        for o in overwrites:
            if int(o.get("type", -1)) == kind and str(o.get("id")) == oid:
                return o
        return None

    everyone = find(OVERWRITE_ROLE, guild_id)
    if everyone is not None:
        perms &= ~_perm(everyone.get("deny"))
        perms |= _perm(everyone.get("allow"))
    deny = allow = 0
    for o in overwrites:
        if int(o.get("type", -1)) == OVERWRITE_ROLE and str(o.get("id")) in member_roles:
            deny |= _perm(o.get("deny"))
            allow |= _perm(o.get("allow"))
    perms &= ~deny
    perms |= allow
    own = find(OVERWRITE_MEMBER, user_id) if user_id else None
    if own is not None:
        perms &= ~_perm(own.get("deny"))
        perms |= _perm(own.get("allow"))
    return perms


def readable_channels(guild: Mapping, member: Mapping, channels: list[Mapping]) -> dict[int, str]:
    """Channel id -> name of the channels this member can read the history of."""
    base = base_permissions(guild, member)
    out: dict[int, str] = {}
    for ch in channels:
        kind = int(ch.get("type", -1))
        if kind not in TEXT_TYPES and kind not in VOICE_TYPES:
            continue
        perms = channel_permissions(guild, member, ch, base)
        need = VIEW_CHANNEL | READ_MESSAGE_HISTORY
        if perms & need == need:
            out[int(ch["id"])] = str(ch.get("name") or ch["id"])
    return out


# ---------------------------------------------------------------- the scope
# platform account id -> {native channel key: name}; "" as a name marks the member's 1:1 chat
RoomProvider = Callable[[Workspace, str], Awaitable[dict[str, str] | None]]


@dataclass(frozen=True)
class Source:
    """What one org workspace contributes to the memory scope."""

    guild_id: int  # the tenant id every store uses for that workspace
    platform: str
    channels: dict[int, str]  # channel id -> citation label ("Discord #general")
    direct: frozenset[int] = frozenset()  # of those: the member's own 1:1 chats


@dataclass(frozen=True)
class LedgerWorkspace:
    guild_id: int
    platform: str
    submitter_id: int | None  # the member's id in that workspace's ledger; None = no account


@dataclass(frozen=True)
class OrgView:
    """What a verified member may use across the org in a 1:1."""

    org_id: str
    email: str
    is_admin: bool  # org owner / admin: sees every receipt of every workspace
    sources: tuple[Source, ...] = ()
    workspaces: tuple[LedgerWorkspace, ...] = ()
    skipped: tuple[str, ...] = field(default=())  # workspace ids that could not be resolved

    def memory_channels(self) -> dict[int, tuple[int, str]]:
        """channel id -> (guild id, label), over every source."""
        return {c: (s.guild_id, label) for s in self.sources for c, label in s.channels.items()}

    def ledger_sources(self, everything: bool) -> list[tuple[int, int | None]]:
        """(guild id, submitter id | None = every receipt) pairs for the ledger adapter."""
        if everything:
            return [(w.guild_id, None) for w in self.workspaces]
        return [(w.guild_id, w.submitter_id) for w in self.workspaces if w.submitter_id is not None]


def submitter_id(platform: str, account: str) -> int | None:
    """The id a member's receipts carry in that platform's ledger (Discord: the user id)."""
    if platform == "discord":
        return int(account) if str(account).isdigit() else None
    return namespaced_id(platform, account, "user")


def channel_id(platform: str, key: str) -> int | None:
    if platform == "discord":
        return int(key) if str(key).isdigit() else None
    return namespaced_id(platform, key, "channel")


def member_of(org: Org, email: str) -> Member | None:
    email = (email or "").lower()
    return next((m for m in org.members if m.email.lower() == email), None)


def widening_allowed(*, is_direct: bool, audience: frozenset[str] | None, asker: str) -> bool:
    """Only a 1:1 whose exact audience is the asker alone widens the scope."""
    return bool(is_direct) and audience is not None and audience == frozenset({asker})


def account_for(member: Member, workspace: Workspace, asking: tuple[str, str] | None) -> str | None:
    """The member's verified account id to use in THIS workspace, or None.

    LINE user ids are per provider, Telegram's per bot and WhatsApp's per business, so such an
    identity only counts in the exact workspace it was verified in (`identity_tenants`). One
    with no recorded workspace counts only in the workspace the question is coming from;
    anything else fails closed. Slack and Discord ids are global."""
    account = member.all_identities().get(workspace.platform)
    if not account:
        return None
    if workspace.platform not in TENANT_BOUND_PLATFORMS:
        return account
    linked_in = member.identity_tenant(workspace.platform)
    if linked_in:
        return account if linked_in == workspace.external_id else None
    here = asking is not None and asking == (workspace.platform, workspace.external_id)
    return account if here else None


def _label(platform: str, name: str, workspace: Workspace, several: bool) -> str:
    base = PLATFORM_NAMES.get(platform, platform.title())
    if several:
        base = f"{base} {workspace.name or workspace.external_id}"
    return f"{base} #{name.lstrip('#')}" if name else f"{base} 1:1 chat"


async def resolve(
    org: Org,
    email: str,
    *,
    is_direct: bool,
    audience: frozenset[str] | None,
    asker: str,
    providers: Mapping[str, RoomProvider],
    asking: tuple[str, str] | None = None,
) -> OrgView | None:
    """The org-wide view of a verified member, or None when the message gets no widening
    (not a 1:1 with exactly the asker as reader, or not a member of the org). Never raises: a
    workspace whose provider is missing or fails is left out and listed in `skipped`.
    `asking` = (platform, tenant) the message comes from (see `account_for`)."""
    if not widening_allowed(is_direct=is_direct, audience=audience, asker=asker):
        return None
    member = member_of(org, email)
    if member is None:
        return None
    connected = [
        w for w in org.workspaces if w.status == "connected" and w.platform in CHAT_PLATFORMS
    ]
    per_platform: dict[str, int] = {}
    for w in connected:
        per_platform[w.platform] = per_platform.get(w.platform, 0) + 1
    sources: list[Source] = []
    ledger: list[LedgerWorkspace] = []
    skipped: list[str] = []
    seen_channels: set[int] = set()
    for w in connected:
        account = account_for(member, w, asking)
        guild = w.memory_id
        ledger.append(
            LedgerWorkspace(
                guild, w.platform, submitter_id(w.platform, account) if account else None
            )
        )
        if not account:
            continue  # nothing of this member's on that platform is verified
        provider = providers.get(w.platform)
        if provider is None:
            skipped.append(w.id)
            continue
        try:
            rooms = await provider(w, account)
        except Exception as e:  # fail closed: this workspace contributes nothing
            logger.warning("org scope: %s unavailable (%s)", w.platform, type(e).__name__)
            rooms = None
        if rooms is None:
            skipped.append(w.id)
            continue
        several = per_platform[w.platform] > 1
        channels: dict[int, str] = {}
        direct: set[int] = set()
        for key, name in rooms.items():
            cid = channel_id(w.platform, key)
            if cid is None or cid in seen_channels:
                continue
            label = _label(w.platform, name, w, several)
            if label in channels.values():
                label = f"{label} ({key})"
            channels[cid] = label
            seen_channels.add(cid)
            if not name and w.platform != "discord":
                direct.add(cid)
        if channels:
            sources.append(Source(guild, w.platform, channels, frozenset(direct)))
    return OrgView(
        org.id,
        member.email.lower(),
        member.role in ("owner", "admin"),
        tuple(sources),
        tuple(ledger),
        tuple(skipped),
    )
