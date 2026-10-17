"""`Workspace`: the one interface between Salid's features and a chat platform.

Design and rules: docs/superpowers/specs/2026-10-18-salid-workspace-parity-design.md.
A feature module never imports `discord` or a Slack SDK: it takes ids, text and a `Workspace`.
One object exists per (platform, tenant). Every id is an int (Discord ids are their own; other
platforms use `salid.platforms.ids`), so the same value keys every store.

This module has no platform imports. Adapters: `salid.workspace_discord.DiscordWorkspace`,
`salid.platforms.workspace_slack.SlackWorkspace`; `FakeWorkspace` below is for feature tests.

`RoomView` here is the int-id twin of `platforms.core.RoomView` (which speaks platform string ids
and is tied to an incoming message). `audience` None = the readers of the room are unknown, and
memory is then limited to that room alone (fail closed). `rooms` are the OTHER rooms Salid can
read, with their viewers (`memory_access.ChannelAccess`), so `compute_scope` can intersect them
with the asker and the audience.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .memory_access import ChannelAccess

# --- capability names for Workspace.supports(...) ------------------------------------------
VOICE = "voice"  # voice meetings, live captions (Discord only)
BUTTONS = "buttons"  # interactive buttons (Discord components, Slack Block Kit)
EVENTS = "events"  # native scheduled events (Discord Scheduled Events)
THREADS = "threads"  # replying under a message
EDIT = "edit"  # editing a posted message
DM = "dm"  # direct messages to a person
FILES = "files"  # attachments
HISTORY = "history"  # reading and importing past messages
CAPABILITIES = (VOICE, BUTTONS, EVENTS, THREADS, EDIT, DM, FILES, HISTORY)

MAX_BUTTONS = 25  # Slack: 25 elements per actions block; Discord: 5 rows x 5
MAX_BUTTON_ID = 100  # Discord custom_id limit (Slack allows 255)
BUTTON_STYLES = ("primary", "secondary", "success", "danger")


class WorkspaceError(Exception):
    """An output call that cannot be done (unknown room, uninstalled workspace, bad button)."""


@dataclass(frozen=True)
class Person:
    id: int
    name: str
    platform_id: str  # the id the platform itself uses ("123", "U0123ABC")
    email: str | None = None
    is_bot: bool = False


@dataclass(frozen=True)
class Room:
    id: int
    name: str
    platform_id: str
    private: bool
    shared: bool = False  # people from outside the workspace (Slack Connect): audience unknown


@dataclass(frozen=True)
class MsgRef:
    """A posted message, enough to edit it later."""

    platform: str
    room_id: int
    platform_ref: str  # Discord message id / Slack ts


@dataclass(frozen=True)
class Button:
    id: str  # see `button_id`
    label: str
    style: str = "secondary"  # primary | secondary | success | danger


@dataclass(frozen=True)
class FileOut:
    name: str
    data: bytes
    mime: str = "application/octet-stream"


@dataclass
class RoomView:
    audience: frozenset[int] | None
    rooms: list[ChannelAccess] = field(default_factory=list)


@dataclass(frozen=True)
class ButtonId:
    feature: str
    action: str
    tenant: int
    obj: str


def button_id(feature: str, action: str, tenant: int, obj: str) -> str:
    """`salid:<feature>:<action>:<tenant>:<obj>`: the persistent callback id of a button.
    `obj` may contain ':' (it is the rest of the id); the other fields may not."""
    for part in (feature, action):
        if not part or ":" in part:
            raise ValueError(f"bad button id part {part!r}")
    if not obj:
        raise ValueError("a button id needs an object")
    out = f"salid:{feature}:{action}:{int(tenant)}:{obj}"
    if len(out) > MAX_BUTTON_ID:
        raise ValueError(f"button id longer than {MAX_BUTTON_ID} characters")
    return out


def parse_button_id(value: str) -> ButtonId | None:
    """The inverse of `button_id`; None for anything that is not one (other ids stay valid)."""
    parts = value.split(":", 4)
    if len(parts) != 5 or parts[0] != "salid" or not parts[4]:
        return None
    _, feature, action, tenant, obj = parts
    if not feature or not action or not tenant.isdigit():
        return None
    return ButtonId(feature, action, int(tenant), obj)


def check_buttons(buttons: Sequence[Button]) -> None:
    if len(buttons) > MAX_BUTTONS:
        raise WorkspaceError(f"at most {MAX_BUTTONS} buttons per message")
    for b in buttons:
        if b.style not in BUTTON_STYLES:
            raise WorkspaceError(f"unknown button style {b.style!r}")
        if not b.id or len(b.id) > MAX_BUTTON_ID or not b.label:
            raise WorkspaceError("a button needs an id (<= 100 chars) and a label")


class Workspace(Protocol):
    platform: str  # "discord" | "slack" | "line" | "telegram" | "whatsapp"
    tenant_id: int  # the guild_id every store uses
    upload_limit: int  # bytes per attachment

    # people
    async def person(self, ref: str) -> Person | None:
        """A name, @mention, email or platform id -> one person; None when none or ambiguous."""
        ...

    async def people(self) -> list[Person]: ...  # members Salid may tag (no bots)

    async def is_exec(self, user: int) -> bool: ...

    async def rank(self, user: int) -> int: ...  # role hierarchy (COO rules); higher = senior

    # rooms and memory scope
    async def rooms(self) -> list[Room]: ...  # readable channels

    async def room_view(self, room: int) -> RoomView: ...

    # output
    async def post(
        self,
        room: int,
        text: str,
        *,
        thread: MsgRef | None = None,
        files: Sequence[FileOut] = (),
        buttons: Sequence[Button] = (),
    ) -> MsgRef:
        """Post `text` (model markdown; long text is split). Files and buttons go with the last
        message. Returns the message that carries the buttons (the last one)."""
        ...

    async def dm(
        self, user: int, text: str, *, files: Sequence[FileOut] = (), buttons: Sequence[Button] = ()
    ) -> MsgRef | None: ...  # None when the person cannot be messaged

    async def edit(self, ref: MsgRef, text: str, *, buttons: Sequence[Button] = ()) -> None:
        """Replace the text; `buttons` REPLACES the old ones (empty = remove them)."""
        ...

    def mention(self, user: int) -> str: ...  # "<@id>" / "<@U..>" / a plain name

    # capabilities
    def supports(self, feature: str) -> bool: ...


# ------------------------------------------------------------------------- test double
@dataclass
class Posted:
    ref: MsgRef
    text: str
    thread: MsgRef | None = None
    files: tuple[FileOut, ...] = ()
    buttons: tuple[Button, ...] = ()
    user: int | None = None  # set for a DM


class FakeWorkspace:
    """A recording `Workspace` for feature tests: no platform, no network.

    `posts` / `dms` / `edits` hold what the feature sent. Capabilities default to everything
    on; pass `supports={...}` to turn the set down. `views` maps a room to its `RoomView`
    (default: unknown audience)."""

    def __init__(
        self,
        *,
        platform: str = "fake",
        tenant_id: int = 1,
        people: Sequence[Person] = (),
        rooms: Sequence[Room] = (),
        execs: Sequence[int] = (),
        ranks: dict[int, int] | None = None,
        views: dict[int, RoomView] | None = None,
        supports: Sequence[str] | None = None,
        upload_limit: int = 8 * 1024 * 1024,
        can_dm: Callable[[int], bool] = lambda _user: True,
    ) -> None:
        self.platform, self.tenant_id, self.upload_limit = platform, tenant_id, upload_limit
        self._people, self._rooms = list(people), list(rooms)
        self._execs, self._ranks = set(execs), dict(ranks or {})
        self._views = dict(views or {})
        self._supports = set(CAPABILITIES if supports is None else supports)
        self._can_dm = can_dm
        self.posts: list[Posted] = []
        self.dms: list[Posted] = []
        self.edits: list[tuple[MsgRef, str, tuple[Button, ...]]] = []
        self._n = 0

    # people
    async def person(self, ref: str) -> Person | None:
        key = ref.strip().lstrip("@").strip("<>").lstrip("@").lower()
        hits = [
            p
            for p in self._people
            if key in {p.name.lower(), p.platform_id.lower(), str(p.id), (p.email or "").lower()}
        ]
        return hits[0] if len(hits) == 1 else None

    async def people(self) -> list[Person]:
        return [p for p in self._people if not p.is_bot]

    async def is_exec(self, user: int) -> bool:
        return user in self._execs

    async def rank(self, user: int) -> int:
        return self._ranks.get(user, 0)

    # rooms
    async def rooms(self) -> list[Room]:
        return list(self._rooms)

    async def room_view(self, room: int) -> RoomView:
        return self._views.get(room, RoomView(None, []))

    # output
    def _ref(self, room: int) -> MsgRef:
        self._n += 1
        return MsgRef(self.platform, room, f"m{self._n}")

    async def post(self, room, text, *, thread=None, files=(), buttons=()):
        check_buttons(buttons)
        if not self.supports(BUTTONS) and buttons:
            raise WorkspaceError("buttons are not supported here")
        p = Posted(self._ref(room), text, thread, tuple(files), tuple(buttons))
        self.posts.append(p)
        return p.ref

    async def dm(self, user, text, *, files=(), buttons=()):
        if not self._can_dm(user):
            return None
        check_buttons(buttons)
        p = Posted(self._ref(-user), text, None, tuple(files), tuple(buttons), user=user)
        self.dms.append(p)
        return p.ref

    async def edit(self, ref, text, *, buttons=()):
        check_buttons(buttons)
        self.edits.append((ref, text, tuple(buttons)))

    def mention(self, user: int) -> str:
        for p in self._people:
            if p.id == user:
                return f"@{p.name}"
        return f"@{user}"

    def supports(self, feature: str) -> bool:
        return feature in self._supports
