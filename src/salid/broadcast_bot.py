"""Discord side of `post_message` and `send_dm_many` (a mixin of `SecretaryBot`).

Rules in code: the asker must see and write in the target channel; @everyone / @here / role
pings need the permissions in `broadcast.plan_mentions`; a mass ping or a channel the asker is not
in needs a confirmation in a LATER message (the secretary confirm gate); a bulk DM of 4 or more
people needs one too, more than 10 people are for executives / managers, 50 is the hard cap, and
money / credential text is never relayed. Bulk DMs go out in the background with a small pause
between them and the result is reported in the chat afterwards.
"""

from __future__ import annotations

import asyncio
import logging
import re

import discord

from .broadcast import (
    MAX_TEXT_CHARS,
    BroadcastActions,
    broadcast_tools,
    find_channels,
    plan_mentions,
    resolve_recipients,
)
from .dm import DM_CONFIRM_OVER, bulk_refusal, name_list, send_dm
from .i18n import t
from .mentions import defang_raw_mentions, resolve_mentions
from .permissions import channel_missing, permission_message
from .secretary import fingerprint

logger = logging.getLogger(__name__)

POST_PER_GUILD_PER_HOUR = 10  # announcements Salid may post per guild per hour
BULK_PER_GUILD_PER_HOUR = 2  # bulk DM requests (4 or more people) per guild per hour
DM_PER_GUILD_PER_HOUR = 20  # DMs of the small (1 to 3 people) path, shared with send_dm
BULK_DM_PACE_S = 0.7  # pause between two DMs of a bulk send
_RAW_MENTION = re.compile(r"<@[!&]?\d+>")


def _can(channel, who, *names: str) -> bool:
    try:
        perms = channel.permissions_for(who)
    except Exception:
        return False
    return all(bool(getattr(perms, n, False)) for n in names)


def _can_view(channel, who) -> bool:
    return _can(channel, who, "view_channel") or _can(channel, who, "read_messages")


class BroadcastMixin:
    # ------------------------------------------------------------- tool entry
    def _broadcast_tools(self, message, guild, asker, lang: str, request_id: str) -> list:
        """The sync tool functions bound to this message (they run in the agent's thread)."""
        loop = asyncio.get_running_loop()

        def run(coro):
            return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=180)

        def post(channel, text, mention="none", tag_inferred=False):
            return run(
                self._post_message(
                    message,
                    guild,
                    asker,
                    lang,
                    request_id,
                    channel=channel,
                    text=text,
                    mention=mention,
                    tag_inferred=tag_inferred,
                )  # fmt: skip
            )

        def many(text, people=None, role=None, channel=None, everyone=False, include_me=False):
            return run(
                self._send_dm_many(
                    message,
                    guild,
                    asker,
                    lang,
                    request_id,
                    text=text,
                    people=people,
                    role=role,
                    channel=channel,
                    everyone=everyone,
                    include_me=include_me,
                )  # fmt: skip
            )

        return broadcast_tools(BroadcastActions(post_message=post, send_dm_many=many))

    def _recent(self, name: str, gid: int, per_hour_window: float = 3600.0) -> list[float]:
        book: dict = self.__dict__.setdefault(name, {})
        now = self._clock()
        book[gid] = [x for x in book.get(gid, []) if now - x < per_hour_window]
        return book[gid]

    # ------------------------------------------------------------- post_message
    async def _post_message(
        self, message, guild, asker, lang, request_id, *, channel, text, mention,
        tag_inferred=False,
    ) -> dict:  # fmt: skip
        gid = guild.id
        clean = _RAW_MENTION.sub("", defang_raw_mentions(text or "", guild)).strip()
        clean = clean[:MAX_TEXT_CHARS]
        if not clean:
            return {"error": "empty_text"}
        everything = list(guild.text_channels)
        visible = [c for c in everything if _can_view(c, asker)]
        digits = re.fullmatch(r"\s*<#(\d+)>\s*", channel or "")
        if digits:
            hits = [c for c in visible if c.id == int(digits[1])]
        else:
            hits = find_channels(visible, channel, guild)
        if not hits:
            if not digits and find_channels(everything, channel, guild):
                return {"error": "not_allowed", "note": "the requester cannot see that channel"}
            return {
                "error": "channel_not_found",
                "channels": [c.name for c in visible[:25]],
                "note": "if the user named a channel in Thai, pick the matching name from "
                "channels and call again; otherwise ask which channel",
            }
        if len(hits) > 1:
            return {
                "error": "ambiguous",
                "candidates": [f"#{c.name}" for c in hits[:8]],
                "note": "ask the user which channel",
            }
        target = hits[0]
        if not _can(target, asker, "send_messages"):
            return {
                "error": "not_allowed",
                "note": f"the requester cannot send messages in #{target.name}",
            }
        bad = channel_missing(target, guild, ("view_channel", "send_messages"))
        if bad:
            return {
                "error": "missing_permission",
                "channel": target.name,
                "how_to_grant": permission_message(bad, lang, self.application_id),
            }
        plan = plan_mentions(mention, guild, asker, target)
        if plan.error:
            return plan.error
        # only members who can see the channel are tagged; the others are listed (offer a DM)
        cant_see: list[str] = []
        for uid, name in list(zip(plan.user_ids, plan.user_names, strict=True)):
            m = guild.get_member(uid)
            if m is None or not _can_view(target, m):
                cant_see.append(name)
                plan.user_ids.remove(uid)
                plan.user_names.remove(name)
        if len(self._recent("_post_sent", gid)) >= POST_PER_GUILD_PER_HOUR:
            return {"error": "rate_limited", "per_hour": POST_PER_GUILD_PER_HOUR}
        resolved = resolve_mentions(clean, guild)
        for uid in {int(i) for i in re.findall(r"<@(\d+)>", resolved)} - set(plan.user_ids):
            m = guild.get_member(uid)
            if m is None or not _can_view(target, m):
                name = getattr(m, "display_name", str(uid))
                resolved = resolved.replace(f"<@{uid}>", f"@{name}")
                cant_see.append(name)
        for uid, name in zip(plan.user_ids, plan.user_names, strict=True):
            resolved = resolved.replace(f"<@{uid}>", f"@{name}")  # tagged once, in the prefix
        body = t("bc_post", lang, sender=asker.display_name, text=resolved)
        if plan.prefix():
            body = f"{plan.prefix()}\n{body}"
        body = body[:2000]
        here = message.channel
        current = {getattr(here, "id", None), getattr(getattr(here, "parent", None), "id", None)}
        inferred = bool(tag_inferred and (plan.user_ids or cant_see))
        if plan.mass or inferred or target.id not in current:
            key = (
                "post", gid, asker.id,
                fingerprint(target.id, clean, inferred, sorted(plan.role_ids), plan.everyone, plan.here,
                            sorted(plan.user_ids)),
            )  # fmt: skip
            if not self.secretary_state.confirm(key, request_id):
                ping = plan.label()
                return {
                    "status": "needs_confirmation",
                    "channel": target.name,
                    "pings": ping or None,
                    "cant_see_channel": cant_see,
                    "dm_offer": bool(cant_see),
                    "preview": t(
                        "bc_preview_post", lang, channel=target.name, ping=ping, text=clean
                    ).replace("  ", " ")
                    + (
                        "\n" + t("bc_cant_see", lang, names=", ".join(cant_see)) if cant_see else ""
                    ),
                    "instruction": "Show the preview line, ask the requester to confirm, then "
                    "call post_message again with EXACTLY the same arguments in a later "
                    "message. Do not claim it was posted.",
                }
        user_ids = {int(i) for i in re.findall(r"<@(\d+)>", body)}
        allowed = discord.AllowedMentions(
            everyone=plan.everyone or plan.here,
            users=[discord.Object(id=i) for i in sorted(user_ids)] or False,
            roles=[discord.Object(id=i) for i in plan.role_ids] or False,
            replied_user=False,
        )
        try:
            sent = await target.send(body, allowed_mentions=allowed)
        except discord.HTTPException:
            logger.warning("post_message: posting failed", exc_info=True)
            return {"error": "could_not_post", "channel": target.name}
        self._recent("_post_sent", gid).append(self._clock())
        out = {
            "status": "posted",
            "channel": target.name,
            "mention": f"<#{target.id}>",
            "link": getattr(sent, "jump_url", ""),
        }
        if plan.label():
            out["pinged"] = plan.label()
        if cant_see:
            out["cant_see_channel"] = cant_see
        if plan.unresolved:
            out["could_not_tag"] = plan.unresolved
        return out

    # ------------------------------------------------------------- send_dm_many
    async def _send_dm_many(
        self, message, guild, asker, lang, request_id, *, text, people, role, channel, everyone,
        include_me,
    ) -> dict:  # fmt: skip
        gid = guild.id
        clean = _RAW_MENTION.sub("", text or "").strip()[:1500]
        if not clean:
            return {"error": "empty_text"}
        if not (people or role or channel or everyone):
            return {"error": "no_recipients", "note": "ask who should get the message"}
        rec = await asyncio.to_thread(
            resolve_recipients,
            guild, asker,
            people=people, role=role, channel=channel, everyone=everyone,
            include_me=include_me,
            channels_for_asker=lambda: [c for c in guild.text_channels if _can_view(c, asker)],
            humans=lambda: self._members(guild),
        )  # fmt: skip
        if rec.error:
            return rec.error
        members = rec.members
        others = [m for m in members if m.id != asker.id]
        refused = bulk_refusal(clean, len(others), self._may_configure(asker)) if others else None
        if not members:
            refused = {"error": "no_recipients"}
        if refused:
            return refused
        n = len(members)
        bulk = n > DM_CONFIRM_OVER
        if bulk:
            if len(self._recent("_bulk_sent", gid)) >= BULK_PER_GUILD_PER_HOUR:
                return {"error": "rate_limited", "bulk_per_hour": BULK_PER_GUILD_PER_HOUR}
            names = [m.display_name for m in members]
            key = ("dm_many", gid, asker.id, fingerprint(sorted(m.id for m in members), clean))
            if not self.secretary_state.confirm(key, request_id):
                return {
                    "status": "needs_confirmation",
                    "count": n,
                    "preview": t(
                        "bc_preview_dm", lang, count=n, names=name_list(names, lang), text=clean
                    ),
                    "instruction": "Show the preview (count and names), ask the requester to "
                    "confirm, then call send_dm_many again with EXACTLY the same arguments in a "
                    "later message. Do not claim anything was sent.",
                }
            self._recent("_bulk_sent", gid).append(self._clock())
            task = asyncio.create_task(
                self._bulk_dm_run(message, guild, asker, lang, members, clean)
            )
            tasks: set = self.__dict__.setdefault("_bulk_tasks", set())
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            return {
                "status": "started",
                "count": n,
                "note": "the DMs go out one by one in the background; the result is posted here "
                "when done. Say it has started, not that it is finished.",
            }
        # one to three people: sent now
        recent = self._recent("_dm_sent", gid)
        if len(recent) + n > DM_PER_GUILD_PER_HOUR:
            return {"error": "rate_limited", "per_hour": DM_PER_GUILD_PER_HOUR}
        sent, closed, failed = await self._dm_all(guild, asker, lang, members, clean, pace=0.0)
        recent.extend([self._clock()] * len(sent))
        return {"status": "sent", "sent": sent, "dms_closed": closed, "failed": failed}

    async def _dm_all(self, guild, asker, lang, members, text, pace: float):
        sent: list[str] = []
        closed: list[str] = []
        failed: list[str] = []
        for i, m in enumerate(members):
            body = (
                text
                if m.id == asker.id
                else t(
                    "relay_dm", lang, sender=asker.display_name, server=guild.name, text=text[:1500]
                )
            )
            res = await send_dm(m, body[:1900])
            if res.ok:
                sent.append(m.display_name)
            elif res.reason == "closed":
                closed.append(m.display_name)
            else:
                failed.append(m.display_name)
            if pace and i < len(members) - 1:
                await asyncio.sleep(pace)
        return sent, closed, failed

    async def _bulk_dm_run(self, message, guild, asker, lang, members, text) -> None:
        try:
            sent, closed, failed = await self._dm_all(
                guild, asker, lang, members, text, pace=BULK_DM_PACE_S
            )
            parts = [t("bc_report", lang, sent=len(sent))]
            if closed:
                parts.append(
                    t("bc_report_closed", lang, count=len(closed), names=name_list(closed, lang))
                )
            if failed:
                parts.append(
                    t("bc_report_failed", lang, count=len(failed), names=name_list(failed, lang))
                )
            report = f"<@{asker.id}> " + ("; " if lang != "th" else " · ").join(parts)
            await message.channel.send(
                report[:2000],
                allowed_mentions=discord.AllowedMentions(
                    users=[discord.Object(id=asker.id)], roles=False, everyone=False
                ),
            )
        except Exception:
            logger.warning("bulk DM report failed", exc_info=True)
