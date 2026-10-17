"""`DiscordWorkspace`: the `Workspace` interface over one `discord.Guild`.

It only wraps what the bot already does; no behaviour changes. Executive and rank rules stay in
the bot and are injected (`exec_check(member)`, `rank_fn(guild, member)`), so `bot.py` can pass
`SecretaryBot.is_exec_member` and `coo`'s `_rank_of` later without moving them.

Buttons are only rendered: a `discord.ui.View` of Buttons whose `custom_id` is `Button.id`.
Their callbacks stay in the persistent `DynamicItem` handlers registered at startup (a view with
no callbacks still posts the components; Discord routes the click by `custom_id`).
"""

from __future__ import annotations

import io
import re
from collections.abc import Callable, Sequence

import discord

from .memory_access import ChannelAccess
from .platforms import textfmt
from .workspace import (
    CAPABILITIES,
    Button,
    FileOut,
    MsgRef,
    Person,
    Room,
    RoomView,
    WorkspaceError,
    check_buttons,
)

PLATFORM = "discord"
CHUNK = 2000
MAX_FILES = 10
DEFAULT_UPLOAD = 10 * 1024 * 1024  # used when a guild reports no limit
MENTION = re.compile(r"^<@!?(\d{1,25})>$")
_STYLES = {
    "primary": discord.ButtonStyle.primary,
    "secondary": discord.ButtonStyle.secondary,
    "success": discord.ButtonStyle.success,
    "danger": discord.ButtonStyle.danger,
}
# Only people Salid names are pinged; never @everyone or roles.
NO_BROADCAST = discord.AllowedMentions(everyone=False, roles=False, users=True, replied_user=False)


def _humans(guild) -> list:
    members = guild.members.values() if isinstance(guild.members, dict) else guild.members
    return [m for m in members if not getattr(m, "bot", False)]


def _can_read(channel, member) -> bool:
    perms = channel.permissions_for(member)
    return bool(
        getattr(perms, "read_messages", False) and getattr(perms, "read_message_history", False)
    )


def _person(m) -> Person:
    return Person(m.id, m.display_name, str(m.id), None, bool(getattr(m, "bot", False)))


def render_view(buttons: Sequence[Button]):
    """A persistent-style `discord.ui.View` (no timeout) for `buttons`, or None."""
    check_buttons(buttons)
    if not buttons:
        return None
    view = discord.ui.View(timeout=None)
    for b in buttons:
        view.add_item(discord.ui.Button(custom_id=b.id, label=b.label[:80], style=_STYLES[b.style]))
    return view


class DiscordWorkspace:
    platform = PLATFORM

    def __init__(
        self,
        client,
        guild,
        *,
        exec_check: Callable[[object], bool],
        rank_fn: Callable[[object, object], int],
    ) -> None:
        self.client, self.guild = client, guild
        self._exec_check, self._rank_fn = exec_check, rank_fn
        self.tenant_id = guild.id

    @property
    def upload_limit(self) -> int:
        return int(getattr(self.guild, "filesize_limit", 0) or DEFAULT_UPLOAD)

    # ------------------------------------------------------------------ people
    async def _member(self, user: int):
        m = self.guild.get_member(user)
        if m is not None:
            return m
        try:
            return await self.guild.fetch_member(user)
        except (discord.NotFound, discord.HTTPException):
            return None

    async def person(self, ref: str) -> Person | None:
        text = ref.strip()
        if not text:
            return None
        found = MENTION.match(text)
        if found or text.isdigit():
            m = await self._member(int(found.group(1) if found else text))
            return _person(m) if m is not None and not getattr(m, "bot", False) else None
        key = text.lstrip("@").strip().casefold()
        # Display name, then global name, then username: the first tier with exactly one hit
        # wins. Two people with the same name are never guessed between (returns None).
        for attr in ("display_name", "global_name", "name"):
            hits = [
                m for m in _humans(self.guild) if (getattr(m, attr, None) or "").casefold() == key
            ]
            if len(hits) == 1:
                return _person(hits[0])
            if len(hits) > 1:
                return None
        return None

    async def people(self) -> list[Person]:
        return [_person(m) for m in _humans(self.guild)]

    async def is_exec(self, user: int) -> bool:
        m = await self._member(user)
        return m is not None and bool(self._exec_check(m))

    async def rank(self, user: int) -> int:
        m = await self._member(user)
        return 0 if m is None else int(self._rank_fn(self.guild, m))

    # ------------------------------------------------------------------ rooms
    def _readable(self) -> list:
        me = self.guild.me
        return [c for c in self.guild.text_channels if c.permissions_for(me).read_messages]

    async def rooms(self) -> list[Room]:
        everyone = self.guild.default_role
        return [
            Room(
                c.id,
                c.name,
                str(c.id),
                private=not c.permissions_for(everyone).read_messages,
            )
            for c in self._readable()
        ]

    async def room_view(self, room: int) -> RoomView:
        """Who can read `room` (live, never cached), plus every other readable room's viewers.
        A room that cannot be found or read gives an unknown audience (fail closed)."""
        channel = self.guild.get_channel(room)
        if channel is None or channel not in self._readable():
            return RoomView(None, [])
        humans = _humans(self.guild)
        audience = frozenset(m.id for m in humans if _can_read(channel, m))
        others = [
            ChannelAccess(c.id, c.name, frozenset(m.id for m in humans if _can_read(c, m)))
            for c in self._readable()
            if c.id != room
        ]
        return RoomView(audience, others)

    # ------------------------------------------------------------------ output
    async def _channel(self, room: int):
        ch = self.guild.get_channel(room) or self.client.get_channel(room)
        if ch is None:
            try:
                ch = await self.client.fetch_channel(room)
            except (discord.NotFound, discord.HTTPException, discord.Forbidden):
                ch = None
        if ch is None:
            raise WorkspaceError(f"unknown room {room}")
        return ch

    @staticmethod
    def _files(files: Sequence[FileOut]) -> list:
        if len(files) > MAX_FILES:
            raise WorkspaceError(f"at most {MAX_FILES} files per message")
        return [discord.File(io.BytesIO(f.data), filename=f.name) for f in files]

    async def _send(self, channel, text, thread, files, buttons) -> MsgRef:
        view = render_view(buttons)
        outs = self._files(files)
        parts = textfmt.split_text(text, CHUNK) or [""]
        reference = None
        if thread is not None:
            reference = channel.get_partial_message(int(thread.platform_ref))
        sent = None
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            kw: dict = {"allowed_mentions": NO_BROADCAST}
            if part:
                kw["content"] = part
            if last and outs:
                kw["files"] = outs
            if last and view is not None:
                kw["view"] = view
            if i == 0 and reference is not None:
                kw["reference"] = reference
            if not part and "files" not in kw and "view" not in kw:
                continue
            sent = await channel.send(**kw)
        if sent is None:
            raise WorkspaceError("nothing to post")
        return MsgRef(PLATFORM, channel.id, str(sent.id))

    async def post(
        self,
        room: int,
        text: str,
        *,
        thread: MsgRef | None = None,
        files: Sequence[FileOut] = (),
        buttons: Sequence[Button] = (),
    ) -> MsgRef:
        return await self._send(await self._channel(room), text, thread, files, buttons)

    async def dm(
        self, user: int, text: str, *, files: Sequence[FileOut] = (), buttons: Sequence[Button] = ()
    ) -> MsgRef | None:
        m = await self._member(user)
        if m is None or getattr(m, "bot", False):
            return None
        try:
            channel = m.dm_channel or await m.create_dm()
            return await self._send(channel, text, None, files, buttons)
        except discord.Forbidden:  # DMs closed
            return None

    async def edit(self, ref: MsgRef, text: str, *, buttons: Sequence[Button] = ()) -> None:
        channel = await self._channel(ref.room_id)
        await channel.get_partial_message(int(ref.platform_ref)).edit(
            content=text[:CHUNK], view=render_view(buttons), allowed_mentions=NO_BROADCAST
        )

    def mention(self, user: int) -> str:
        return f"<@{int(user)}>"

    def supports(self, feature: str) -> bool:
        return feature in CAPABILITIES  # Discord offers everything, including voice and events
