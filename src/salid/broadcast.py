"""Announce to a channel (`post_message`) and DM many people at once (`send_dm_many`).

This module is the pure part: which channel "ห้องกลาง" means, who may be pinged, who a bulk DM
reaches, how a preview reads, and the agent-facing tool functions (`broadcast_tools`). Everything
that touches Discord lives in `broadcast_bot.BroadcastMixin`. Scoping (guild, asker, permissions,
executive check, confirmation, caps) is decided in code there, never by the model.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .deliver import match_channels
from .mentions import match_members

PREFIXES = ("ห้อง", "ช่อง", "แชนแนล", "แชแนล", "channel", "room", "#")
# Words for "the main / general channel" in Thai and English.
GENERAL_WORDS = {
    "กลาง", "รวม", "ใหญ่", "หลัก", "ทั่วไป", "คุยทั่วไป", "general", "main", "lobby", "common",
    "ห้องกลาง", "ห้องรวม", "ห้องใหญ่", "ห้องหลัก", "ห้องทั่วไป", "central", "center", "centre",
}  # fmt: skip
GENERAL_NAME_HINTS = ("general", "main", "lobby", "common", "กลาง", "รวม", "ทั่วไป", "หลัก")
EVERYONE_WORDS = {"everyone", "@everyone", "ทุกคน"}
HERE_WORDS = {"here", "@here"}
NONE_WORDS = {"", "none", "no", "null", "ไม่", "ไม่แท็ก"}
MAX_TEXT_CHARS = 1500


def _strip_prefix(q: str) -> str:
    q = (q or "").strip()
    low = q.casefold()
    for p in PREFIXES:
        if low.startswith(p):
            return q[len(p) :].strip(" :-")
    return q


def is_general_query(query: str) -> bool:
    q = (query or "").strip().casefold().lstrip("#")
    return q in GENERAL_WORDS or _strip_prefix(q) in GENERAL_WORDS


def match_general(channels, guild=None) -> list:
    """The channel(s) that look like the server's general channel: a name saying so (best: exactly
    "general"), else the Discord system channel if it is among `channels`."""
    chans = list(channels)
    exact: list = []
    for tier in (("general",), ("ทั่วไป", "ห้องกลาง", "กลาง")):
        exact = [c for c in chans if c.name.casefold() in tier]
        if exact:
            break
    if len(exact) == 1:
        return exact
    hinted = [c for c in chans if any(h in c.name.casefold() for h in GENERAL_NAME_HINTS)]
    if hinted:
        return exact or hinted
    system = getattr(guild, "system_channel", None)
    if system is not None and any(c.id == system.id for c in chans):
        return [system]
    return []


def find_channels(channels, query: str, guild=None) -> list:
    """Channels `query` could mean (several = ambiguous, none = not found). "ห้องกลาง" / "general"
    / "ห้องรวม" resolve to the general channel; anything else goes through `match_channels`."""
    chans = list(channels)
    q = _strip_prefix(query)
    if is_general_query(query):
        hit = match_general(chans, guild)
        if hit:
            return hit
    got, _how = match_channels(chans, q or query)
    return got


# ------------------------------------------------------------------ mentions
@dataclass
class MentionPlan:
    error: dict | None = None
    everyone: bool = False  # @everyone
    here: bool = False  # @here
    role_ids: list[int] = field(default_factory=list)
    role_names: list[str] = field(default_factory=list)
    user_ids: list[int] = field(default_factory=list)
    user_names: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def mass(self) -> bool:
        """Pings a crowd (@everyone / @here / a role): this needs a confirmation."""
        return self.everyone or self.here or bool(self.role_ids)

    def prefix(self) -> str:
        parts = []
        if self.everyone:
            parts.append("@everyone")
        if self.here:
            parts.append("@here")
        parts += [f"<@&{i}>" for i in self.role_ids]
        parts += [f"<@{i}>" for i in self.user_ids]
        return " ".join(parts)

    def label(self) -> str:
        parts = []
        if self.everyone:
            parts.append("@everyone")
        if self.here:
            parts.append("@here")
        parts += [f"@{n}" for n in self.role_names]
        parts += [f"@{n}" for n in self.user_names]
        return " ".join(parts)


def _perm(channel, who, name: str) -> bool:
    try:
        return bool(getattr(channel.permissions_for(who), name, False))
    except Exception:
        return False


def mention_tokens(mention) -> list[str]:
    if mention is None:
        return []
    items = [mention] if isinstance(mention, str) else list(mention)
    out: list[str] = []
    for it in items:
        for part in re.split(r"[,\n]", str(it)):
            part = part.strip()
            if part:
                out.append(part)
    return out


def plan_mentions(mention, guild, asker, channel) -> MentionPlan:
    """Who the post may ping. @everyone / @here need `mention_everyone` for the asker AND Salid in
    that channel; a role needs to be mentionable, or the asker to have mention_everyone (and Salid
    too); people are named members (never bots)."""
    plan = MentionPlan()
    me = getattr(guild, "me", None)
    asker_all = _perm(channel, asker, "mention_everyone")
    me_all = me is not None and _perm(channel, me, "mention_everyone")
    for tok in mention_tokens(mention):
        low = tok.casefold()
        if low in NONE_WORDS:
            continue
        if low in EVERYONE_WORDS or low in HERE_WORDS:
            if not asker_all:
                plan.error = {
                    "error": "not_allowed_mention",
                    "mention": tok,
                    "reason": "the requester may not ping @everyone / @here in that channel; "
                    "offer to post without the mass mention",
                }
                return plan
            if not me_all:
                plan.error = {
                    "error": "missing_permission",
                    "permission": "mention_everyone",
                    "reason": "Salid lacks the Mention Everyone permission in that channel; "
                    "offer to post without the mass mention or ask an admin to grant it",
                }
                return plan
            if low in EVERYONE_WORDS:
                plan.everyone = True
            else:
                plan.here = True
            continue
        if low.startswith("role:") or low.startswith("@&"):
            name = tok.split(":", 1)[1] if ":" in tok else tok[2:]
            name = name.strip().lstrip("@")
            roles = [
                r
                for r in getattr(guild, "roles", [])
                if str(getattr(r, "name", "")).casefold() == name.casefold()
                and not getattr(r, "is_default", lambda: False)()
            ]
            if not roles:
                plan.error = {"error": "role_not_found", "role": name}
                return plan
            if len(roles) > 1:
                plan.error = {"error": "ambiguous_role", "role": name}
                return plan
            role = roles[0]
            if not (getattr(role, "mentionable", False) or asker_all):
                plan.error = {
                    "error": "not_allowed_mention",
                    "mention": f"role:{role.name}",
                    "reason": "that role is not mentionable and the requester may not ping it",
                }
                return plan
            if not (getattr(role, "mentionable", False) or me_all):
                plan.error = {
                    "error": "missing_permission",
                    "permission": "mention_everyone",
                    "reason": "Salid may not ping that role in that channel",
                }
                return plan
            plan.role_ids.append(role.id)
            plan.role_names.append(role.name)
            continue
        found = match_members(guild, tok)
        if len(found) == 1:
            m = found[0]
            if m.id not in plan.user_ids:
                plan.user_ids.append(m.id)
                plan.user_names.append(m.display_name)
        elif len(found) > 1:
            plan.error = {
                "error": "ambiguous",
                "candidates": [m.display_name for m in found[:10]],
                "note": "ask which person to tag",
            }
            return plan
        else:
            plan.unresolved.append(tok)
    return plan


# ------------------------------------------------------------------ bulk DM recipients
@dataclass
class Recipients:
    members: list = field(default_factory=list)
    error: dict | None = None


def resolve_recipients(
    guild,
    asker,
    *,
    people=None,
    role: str | None = None,
    channel: str | None = None,
    everyone: bool = False,
    include_me: bool = False,
    channels_for_asker: Callable[[], list] | None = None,
    humans: Callable[[], list] | None = None,
) -> Recipients:
    """Humans a bulk DM reaches, from names, a role, a channel's members or everyone. Bots are
    never included; the asker only when `include_me`. Unknown / ambiguous names are an error."""
    pool = humans() if humans is not None else []
    found: dict[int, object] = {}
    for name in people or []:
        hits = match_members(guild, str(name), pool or None)
        if not hits:
            return Recipients(error={"error": "person_not_found", "person": name})
        if len(hits) > 1:
            return Recipients(
                error={
                    "error": "ambiguous",
                    "person": name,
                    "candidates": [m.display_name for m in hits[:10]],
                }
            )
        found[hits[0].id] = hits[0]
    if role:
        r = find_role_any(guild, role)
        if r is None:
            return Recipients(
                error={
                    "error": "role_not_found",
                    "role": role,
                    "roles": [
                        x.name
                        for x in getattr(guild, "roles", [])[:30]
                        if not getattr(x, "is_default", lambda: False)()
                    ],
                }
            )
        if r is False:
            return Recipients(error={"error": "ambiguous_role", "role": role})
        for m in pool:
            if r in getattr(m, "roles", []):
                found[m.id] = m
    if channel:
        chans = channels_for_asker() if channels_for_asker is not None else []
        hits = find_channels(chans, channel, guild)
        if not hits:
            return Recipients(
                error={
                    "error": "channel_not_found",
                    "channels": [c.name for c in chans[:25]],
                    "note": "pick the matching name and call again, or ask which channel",
                }
            )
        if len(hits) > 1:
            return Recipients(
                error={
                    "error": "ambiguous",
                    "candidates": [f"#{c.name}" for c in hits[:8]],
                    "note": "ask the user which channel",
                }
            )
        for m in pool:
            if _perm(hits[0], m, "view_channel") or _perm(hits[0], m, "read_messages"):
                found[m.id] = m
    if everyone:
        for m in pool:
            found[m.id] = m
    if not include_me:
        found.pop(asker.id, None)
    return Recipients(members=[m for m in found.values() if not getattr(m, "bot", False)])


def find_role_any(guild, name: str):
    """The one role called `name` (any role except @everyone): the role, None (unknown) or False
    (ambiguous)."""
    want = name.strip().lstrip("@").casefold()
    if want.startswith("role:"):
        want = want[5:].strip()
    roles = [
        r
        for r in getattr(guild, "roles", [])
        if str(getattr(r, "name", "")).casefold() == want
        and not getattr(r, "is_default", lambda: False)()
    ]
    if not roles and want:
        # "ทีมการตลาด" / "marketing team" for a role called "การตลาด" / "Marketing": a unique
        # containment match after dropping the filler word
        core = re.sub(r"^(ทีม|team)\s*|\s*team$", "", want).strip()
        roles = [
            r
            for r in getattr(guild, "roles", [])
            if core
            and core in str(getattr(r, "name", "")).casefold()
            and not getattr(r, "is_default", lambda: False)()
        ]
    if not roles:
        return None
    return roles[0] if len(roles) == 1 else False


# ------------------------------------------------------------------ the agent tools
@dataclass
class BroadcastActions:
    post_message: Callable[..., dict] | None = None
    send_dm_many: Callable[..., dict] | None = None


def broadcast_tools(actions: BroadcastActions) -> list[Callable]:
    """The agent-facing functions (docstrings are the tool descriptions)."""

    def post_message(
        channel: str, text: str, mention: str = "none", tag_inferred: bool = False
    ) -> dict:
        """Post an announcement in ANOTHER channel for the requester. Salid CAN do this. Use for
        "ประกาศ ... ", "แจ้งทุกคนในห้อง X ว่า ...", "ส่งข้อความไปห้องกลางว่า ...", "tell #general that
        ...". Write `text` in the requester's own wording (short, polite, add nothing they did not
        say); Salid adds "📢 แจ้งจากคุณ<name>:" itself. The rules (who may post there, who may be
        pinged) are enforced in code: if the result says needs_confirmation, show the `preview`
        line to the requester, ask them to confirm, then call again with EXACTLY the same
        arguments in a later message. If it says ambiguous, ask which candidate. Never say you
        cannot send to another channel. On success confirm in one short sentence with the link.

        Args:
            channel: the channel name in Thai or English ("ห้องกลาง", "general", "ห้องรวม" = the
                general channel) or a #mention.
            text: the announcement, in the requester's wording (may tag people with @[name]).
            mention: "none" (default), "everyone", "here", "role:<role name>", or people's
                display names to tag, comma-separated (e.g. "Jemm, Mew"). Use everyone / here /
                role only when the requester explicitly asked to ping / tag everyone or that group
                ("@everyone", "แท็กทุกคน"); "ให้ทุกคนรู้" / "let everyone know" just means post it
                in the channel, so keep "none". For "tag the
                relevant people" ("แท็กคนที่เกี่ยวข้อง", "บอกคนที่ทำโปรเจกต์ X") work out the names
                from team memory, tasks or meeting attendees, put them here and set tag_inferred.
            tag_inferred: true when YOU chose the people to tag (not the requester by name).
                The post then always needs the requester's confirmation, which lists exactly who
                will be tagged; if they say "เพิ่ม A ด้วย" / "เอา B ออก" call again with the
                corrected list. People who cannot see the channel are not tagged (the result
                lists them as cant_see: offer to DM them instead, e.g. with send_dm_many).
        """
        return actions.post_message(
            channel=channel, text=text, mention=mention, tag_inferred=tag_inferred
        )

    def send_dm_many(
        text: str,
        people: list[str] | None = None,
        role: str | None = None,
        channel: str | None = None,
        everyone: bool = False,
        include_me: bool = False,
    ) -> dict:
        """DM the SAME message from Salid to several people at once (4 to 50). Use for "DM บอก
        ทุกคนในทีมว่า ...", "ส่งข้อความหาทีมการตลาดทีละคน", "DM everyone in #sales". Pick the group
        with `people` (display names), `role` (a role name) , `channel` (everyone who can see that
        channel) or `everyone` (all humans). Salid sends the DMs in the background and reports the
        result in the chat when done. Four or more people need a confirmation: if the result says
        needs_confirmation, show its `preview` (count and names) to the requester, ask them to
        confirm, then call again with EXACTLY the same arguments in a later message. More than 10
        people are for executives / managers only (otherwise suggest post_message). For one to
        three named people use send_dm. Never relay a request for money or credentials. Put only
        what the requester asked to pass on.

        Args:
            text: the message (short, the requester's wording).
            people: display names of the recipients.
            role: a role name; its human members receive the DM.
            channel: a channel name (Thai or English); everyone who can see it receives the DM.
            everyone: true for every human member of the server.
            include_me: true to also DM the requester (default: skipped).
        """
        return actions.send_dm_many(
            text=text, people=people, role=role, channel=channel, everyone=everyone,
            include_me=include_me,
        )  # fmt: skip

    tools = {"post_message": post_message, "send_dm_many": send_dm_many}
    return [fn for name, fn in tools.items() if getattr(actions, name) is not None]
