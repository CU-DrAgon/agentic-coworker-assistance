"""The bot's side of `salid.actions`: carry out, in Discord, what the web queued.

Runs from the bot tick (`_actions_due`, every tick) for every guild the bot is in: each pending
action is performed with the bot's permissions and finished with a result (ids, links) or an
error the web can show ("missing permission", "channel not found"...). Kinds:

* post_message {channel, text}            -> message id + jump link
* announce     {text}                     -> same, in the server's ask channel, else general /
  the system channel, else the first text channel Salid may write in (company-wide notices
  such as "the company language is now Thai")
* send_dm      {user_id, text}            -> delivered
* create_channel {name, kind=text|voice, topic} -> channel id
* create_event {name, start, end, where, description} -> event id + url (voice channel name or
  a free-text location; `start`/`end` ISO instants). Tracked by the scheduler like any event.
* edit_event   {event_id, name?, start?, end?, description?}
* delete_event {event_id}
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import discord

from .actions import ActionQueue
from .broadcast import find_channels, match_general

logger = logging.getLogger(__name__)


def _iso(s: str) -> datetime:
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=UTC)


class ActionsMixin:
    def _init_actions(self) -> None:
        self.actions = ActionQueue(self.docs, clock=lambda: self.s.now())

    async def _actions_due(self) -> int:
        """Perform every pending action of every guild; returns how many were handled."""
        n = 0
        for guild in list(self.guilds):
            try:
                self.actions.expire(guild.id)
                for a in self.actions.pending(guild.id):
                    n += 1
                    try:
                        result = await self._perform_action(guild, a)
                        self.actions.finish(guild.id, a["id"], result=result)
                    except discord.Forbidden:
                        self.actions.finish(guild.id, a["id"], error="missing permission")
                    except (KeyError, ValueError) as e:
                        self.actions.finish(guild.id, a["id"], error=str(e)[:200] or "bad request")
                    except discord.HTTPException as e:
                        self.actions.finish(
                            guild.id, a["id"], error=f"discord: {getattr(e, 'text', e)}"[:200]
                        )
                    except Exception as e:  # never let one action stop the tick
                        logger.exception("action %s failed", a.get("id"))
                        self.actions.finish(guild.id, a["id"], error=type(e).__name__)
            except Exception:
                logger.exception("action queue of guild %s failed", guild.id)
        return n

    def _action_channel(self, guild, ref: Any):
        if ref is None or ref == "":
            raise ValueError("channel is required")
        if str(ref).isdigit():
            ch = guild.get_channel(int(ref))
            if ch is not None:
                return ch
        hits = find_channels(
            list(guild.text_channels) + list(guild.voice_channels), str(ref), guild
        )
        if not hits:
            raise ValueError(f"channel not found: {ref}")
        return hits[0]

    def _announce_channel(self, guild):
        """Where a company-wide notice goes: the first configured ask channel, else "general"
        (`broadcast.general_channels`), else the first text channel Salid may write in."""
        me = getattr(guild, "me", None)

        def writable(ch) -> bool:
            try:
                return me is None or bool(ch.permissions_for(me).send_messages)
            except Exception:
                return True

        try:
            ids = list(self.team.current(guild.id).ask_channel_ids)
        except Exception:
            ids = []
        for cid in ids:
            ch = guild.get_channel(int(cid)) if str(cid).isdigit() else None
            if ch is not None and writable(ch):
                return ch
        for ch in match_general(list(guild.text_channels), guild):
            if writable(ch):
                return ch
        return next((ch for ch in guild.text_channels if writable(ch)), None)

    async def _perform_action(self, guild, a: dict) -> dict:
        kind, args = a.get("kind"), a.get("args") or {}
        if kind == "post_message":
            ch = self._action_channel(guild, args.get("channel"))
            text = str(args.get("text", "")).strip()[:2000]
            if not text:
                raise ValueError("text is required")
            m = await ch.send(text)
            return {"message_id": str(m.id), "url": getattr(m, "jump_url", ""), "channel": ch.name}
        if kind == "announce":
            text = str(args.get("text", "")).strip()[:2000]
            if not text:
                raise ValueError("text is required")
            ch = self._announce_channel(guild)
            if ch is None:
                raise ValueError("no channel Salid may write in")
            m = await ch.send(text)
            return {"message_id": str(m.id), "url": getattr(m, "jump_url", ""), "channel": ch.name}
        if kind == "send_dm":
            uid = int(args.get("user_id") or 0)
            member = guild.get_member(uid) or await guild.fetch_member(uid)
            text = str(args.get("text", "")).strip()[:2000]
            if not text:
                raise ValueError("text is required")
            await member.send(text)
            return {"delivered_to": member.display_name}
        if kind == "create_channel":
            name = str(args.get("name", "")).strip()[:100]
            if not name:
                raise ValueError("name is required")
            if args.get("kind") == "voice":
                ch = await guild.create_voice_channel(name)
            else:
                ch = await guild.create_text_channel(
                    name, topic=str(args.get("topic", ""))[:1024] or None
                )
            return {"channel_id": str(ch.id), "name": ch.name}
        if kind == "create_event":
            name = str(args.get("name", "")).strip()[:100]
            if not name or not args.get("start"):
                raise ValueError("name and start are required")
            start = _iso(str(args["start"]))
            end = _iso(str(args["end"])) if args.get("end") else start + timedelta(hours=1)
            if start < datetime.now(UTC):
                raise ValueError("start is in the past")
            kw: dict[str, Any] = dict(
                name=name,
                start_time=start,
                end_time=end,
                description=str(args.get("description", ""))[:1000],
                privacy_level=discord.PrivacyLevel.guild_only,
            )
            where = str(args.get("where", "")).strip()
            voice = None
            if where:
                hits = find_channels(list(guild.voice_channels), where, guild)
                voice = hits[0] if hits else None
            if voice is not None:
                kw["channel"] = voice
            else:
                kw.update(
                    entity_type=discord.EntityType.external, location=(where or "Online")[:100]
                )
            ev = await guild.create_scheduled_event(**kw)
            return {"event_id": str(ev.id), "url": getattr(ev, "url", ""), "name": ev.name}
        if kind in ("edit_event", "delete_event"):
            eid = int(args.get("event_id") or 0)
            ev = guild.get_scheduled_event(eid) or await guild.fetch_scheduled_event(eid)
            if kind == "delete_event":
                await ev.delete()
                return {"deleted": str(eid)}
            kw = {}
            if args.get("name"):
                kw["name"] = str(args["name"])[:100]
            if args.get("start"):
                kw["start_time"] = _iso(str(args["start"]))
            if args.get("end"):
                kw["end_time"] = _iso(str(args["end"]))
            if args.get("description") is not None and "description" in args:
                kw["description"] = str(args["description"])[:1000]
            ev = await ev.edit(**kw)
            return {"event_id": str(ev.id), "url": getattr(ev, "url", "")}
        raise ValueError(f"unknown action {kind!r}")
