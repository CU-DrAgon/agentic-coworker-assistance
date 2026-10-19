"""ONE Salid for every transport: the Discord bot's brain, tools and persona, driven from the web.

The product rule: Salid is one colleague. Discord, the web command center (chat and live voice)
and later Slack must not grow separate chat systems. The bot already has the single place where
the agent is assembled (`SecretaryBot._agent_kwargs`: memory, secretary, every feature's tools,
company playbook, tone, web search). So the web service runs a HEADLESS bot: the same
`SecretaryBot` object, never connected to the Discord gateway, and talks to it through small
stand-ins for the Discord objects its tool factories read (guild, member, channel, message).

What differs on the web is only the environment, not the brain:
* the person is already addressed (no wake word, no ASR), so answers come at once;
* anything a tool would "send to the channel" lands in the web chat (the stand-in channel keeps
  an outbox) instead of Discord; DMs become lines in the same panel;
* the person is identified by their verified Discord id when they have one (then Salid treats
  them exactly as in Discord: tasks, reminders, rank), else by a stable id derived from the email.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

from .orgs import _email_bucket

logger = logging.getLogger(__name__)

WEB_CHANNEL_ID = -1  # the stand-in channel's id: never a real Discord channel


class _Typing:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


class WebChannel(SimpleNamespace):
    """The "channel" a web conversation happens in: collects what Salid would post."""

    def __init__(self, guild, name: str = "web"):
        super().__init__(id=WEB_CHANNEL_ID, name=name, guild=guild, outbox=[], mention=f"#{name}")

    def typing(self):
        return _Typing()

    async def send(self, content: str = "", **kw) -> Any:
        self.outbox.append({"content": content, **{k: v for k, v in kw.items() if k != "view"}})
        return SimpleNamespace(id=0, jump_url="", edit=self._noop, delete=self._noop)

    async def _noop(self, *a, **k):
        return None

    def permissions_for(self, member):
        return SimpleNamespace(view_channel=True, read_message_history=True, send_messages=True)


class WebMember(SimpleNamespace):
    """A person on the web: looks like a Discord member to the tool factories."""

    def __init__(self, guild, uid: int, name: str, *, admin: bool, email: str = ""):
        perms = SimpleNamespace(administrator=admin, manage_guild=admin, manage_channels=admin)
        super().__init__(
            id=uid,
            name=name,
            display_name=name,
            global_name=name,
            guild=guild,
            guild_permissions=perms,
            roles=[],
            voice=None,
            bot=False,
            mention=f"@{name}",
            email=email,
            web=True,
        )
        self.guild = guild

    async def send(self, content: str = "", **kw):  # a DM: shown in the web panel instead
        return await self.guild.web_channel.send(content, **kw)

    def __str__(self) -> str:
        return self.name


class WebGuild(SimpleNamespace):
    """The company's Discord server as the bot's code sees it, without the gateway."""

    def __init__(self, gid: int, name: str, *, owner_id: int = 0):
        super().__init__(
            id=gid,
            name=name,
            me=None,
            owner_id=owner_id,
            text_channels=[],
            voice_channels=[],
            channels=[],
            members=[],
            roles=[],
            scheduled_events=[],
            icon=None,
            web=True,
        )
        self.web_channel = WebChannel(self)
        self._members: dict[int, WebMember] = {}

    def add_member(self, m: WebMember) -> WebMember:
        self._members[m.id] = m
        self.members = list(self._members.values())
        return m

    def get_member(self, uid: int):
        return self._members.get(uid)

    def get_channel(self, cid: int):
        return self.web_channel if cid == WEB_CHANNEL_ID else None

    def get_role(self, rid: int):
        return None


def person_id(discord_id: str | None, email: str) -> int:
    """The id Salid knows the person by: their verified Discord id, else a stable id from the
    email (so reminders, tasks and preferences stick to them across sessions)."""
    return int(discord_id) if discord_id and discord_id.isdigit() else _email_bucket(email)


def web_context(gid: int, guild_name: str, *, uid: int, name: str, admin: bool, email: str = ""):
    """(guild, member, channel, message) stand-ins for one web request. `message` comes from
    `SecretaryBot.room_message`, the same stand-in the voice rooms use."""
    guild = WebGuild(gid, guild_name)
    member = guild.add_member(WebMember(guild, uid, name, admin=admin, email=email))
    return guild, member, guild.web_channel


@lru_cache
def _headless() -> Any:
    """One headless bot per process (its memory index and caches are per process anyway)."""
    from .bot import SecretaryBot
    from .config import Settings
    from .llm import make_client
    from .store import make_store

    s = Settings.from_env()
    store = make_store(s)
    bot = SecretaryBot(s, store, make_client(s))
    bot.headless = True
    # Never connected, so `setup_hook` never starts the monitor loop and nothing would flush the
    # emitter: keep it off whatever `SALID_MONITOR_ENABLED` says (the web emits its own events).
    bot.monitor.enabled = False
    return bot


def get_headless_bot() -> Any:
    """The web service's Salid. Overridden in tests (`meeting_scribe.center.get_bot`)."""
    return _headless()


def agent_kwargs_in_thread(bot: Any, **kw) -> dict:
    """`SecretaryBot._agent_kwargs` wants a running event loop (some factories keep it for
    follow-ups). From a worker thread, run it on a short-lived loop."""

    async def build() -> dict:
        return bot._agent_kwargs(**kw)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(build())
    return bot._agent_kwargs(**kw)  # pragma: no cover - called on the loop already
