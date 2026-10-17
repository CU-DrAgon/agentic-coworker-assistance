"""`SlackWorkspace`: the `Workspace` interface over one `SlackPort` (one Slack workspace).

Ids: Slack ids are strings, so every id the interface speaks is an int from `platforms.ids`
(`user_id`, `room_id`). The reverse maps (int -> Slack id) are filled whenever a person or room
is seen (`people()`, `person()`, `rooms()`, `learn_user`, `learn_room`, an opened DM); an outbound
call with an int never seen is looked up once in the user / channel list, else it is refused.

Buttons are rendered as Block Kit `actions` with `action_id` = `Button.id`. Routing the
`block_actions` payload back to the feature handlers (Bolt `app.action(...)` keyed on
`parse_button_id`) is a later step; until then a click on a Slack button is not handled.

Slack API surface used (slack_sdk AsyncWebClient): users.list, users.info, users.conversations,
conversations.open, chat.postMessage, chat.update, files.upload (`files_upload_v2`; scope
`files:write`). Writes go through `SlackPort._call` (per-channel pacing, 429 Retry-After).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from ..memory_access import ChannelAccess
from ..workspace import (
    CAPABILITIES,
    EVENTS,
    VOICE,
    Button,
    FileOut,
    MsgRef,
    Person,
    Room,
    RoomView,
    WorkspaceError,
    check_buttons,
)
from . import ids, textfmt
from .core import safe_exc
from .slack import CHUNK, LIST_TTL_S, MAX_ROOMS, PLATFORM, SlackPort, share_state

logger = logging.getLogger(__name__)

# Slack documents 1 GB as the largest upload; Salid's own paths (Gemini, PDFs, decks) never need
# that, so be conservative.
UPLOAD_LIMIT = 100 * 1024 * 1024
USERS_TTL_S = 600.0
MENTION = re.compile(r"^<@([UW][A-Z0-9]+)(?:\|[^>]*)?>$")
RAW_ID = re.compile(r"^[UW][A-Z0-9]+$")
# `mention()` hands out a sentinel, not "<@U..>": the Slack formatter escapes `<`/`>` (so the
# model or a user cannot ping anyone), and only the sentinel of a local human is turned into a
# real mention after formatting.
MENTION_OPEN, MENTION_CLOSE = "\ue000", "\ue001"
_SENTINEL = re.compile(f"{MENTION_OPEN}([UW][A-Z0-9]+){MENTION_CLOSE}")
_STYLES = {
    "primary": "primary",
    "success": "primary",
    "danger": "danger",
}  # secondary: Slack default
UNSUPPORTED = frozenset({VOICE, EVENTS})  # Huddles have no bot API; no scheduled-event object


def _name(u: dict) -> str:
    prof = u.get("profile") or {}
    return prof.get("display_name") or u.get("real_name") or u.get("name") or u.get("id") or ""


def blocks(text: str, buttons: Sequence[Button]) -> list[dict]:
    """Block Kit for a text (already Slack mrkdwn) and its buttons."""
    out: list[dict] = [{"type": "section", "text": {"type": "mrkdwn", "text": text or " "}}]
    if buttons:
        elements = []
        for b in buttons:
            el: dict = {
                "type": "button",
                "action_id": b.id,
                "text": {"type": "plain_text", "text": b.label[:75]},
            }
            if b.style in _STYLES:
                el["style"] = _STYLES[b.style]
            elements.append(el)
        out.append({"type": "actions", "elements": elements})
    return out


class SlackWorkspace:
    platform = PLATFORM
    upload_limit = UPLOAD_LIMIT

    def __init__(self, port: SlackPort) -> None:
        self.port = port
        self.tenant_id = ids.tenant_id(PLATFORM, port.team_id)
        self._user_ids: dict[int, str] = {}
        self._room_ids: dict[int, str] = {}
        self._humans: set[str] = set()  # Slack ids of this workspace's own humans (seen so far)
        self._users: tuple[float, list[dict]] | None = None
        self._rooms: tuple[float, list[dict]] | None = None

    # ------------------------------------------------------------------ id maps
    def learn_user(self, uid: str) -> int:
        n = ids.user_id(PLATFORM, uid)
        self._user_ids[n] = uid
        return n

    def learn_room(self, channel: str) -> int:
        n = ids.room_id(PLATFORM, self.port.team_id, channel)
        self._room_ids[n] = channel
        return n

    async def _slack_user(self, user: int) -> str | None:
        if user not in self._user_ids:
            await self._user_list(fresh=True)
        return self._user_ids.get(user)

    async def _slack_room(self, room: int) -> str | None:
        if room not in self._room_ids:
            await self._room_list(fresh=True)
        return self._room_ids.get(room)

    # ------------------------------------------------------------------ people
    async def _user_list(self, *, fresh: bool = False) -> list[dict]:
        now = self.port._clock()
        if not fresh and self._users and now - self._users[0] < USERS_TTL_S:
            return self._users[1]
        try:
            rows = await self.port._pages(self.port.client.users_list, "members", limit=200)
        except Exception as e:
            logger.warning("users.list failed (%s)", safe_exc(e))
            return self._users[1] if self._users else []
        # This workspace's own humans only: no bots, deleted accounts, or Slack Connect guests
        # of another team.
        keep = [
            u
            for u in rows
            if u.get("id")
            and not u.get("is_bot")
            and not u.get("deleted")
            and u.get("id") != "USLACKBOT"
            and u.get("team_id") == self.port.team_id
        ]
        for u in keep:
            self.learn_user(u["id"])
            self._humans.add(u["id"])
        self._users = (now, keep)
        return keep

    def _make(self, u: dict) -> Person:
        email = (u.get("profile") or {}).get("email") or None
        self._humans.add(u["id"])
        return Person(self.learn_user(u["id"]), _name(u), u["id"], email, bool(u.get("is_bot")))

    async def people(self) -> list[Person]:
        return [self._make(u) for u in await self._user_list()]

    async def person(self, ref: str) -> Person | None:
        text = ref.strip()
        if not text:
            return None
        found = MENTION.match(text)
        uid = found.group(1) if found else (text if RAW_ID.match(text) else None)
        if uid is not None:
            # An explicit mention / platform id never falls through to name matching.
            u = await self.port.user(uid)  # users.info, cached
            if not u:
                return None
            # Same fail-closed rule as the admin check: this workspace's own humans only.
            own = u.get("team_id") == self.port.team_id
            if u.get("is_bot") or u.get("deleted") or not own:
                return None
            return self._make({**u, "id": u.get("id") or uid})
        key = text.lstrip("@").strip().casefold()
        # Every alias of every local human is a candidate; a person is returned only when
        # exactly one human matches (the display name of A equal to the username of B: None).
        hits: dict[str, dict] = {}
        for u in await self._user_list():
            prof = u.get("profile") or {}
            aliases = (
                prof.get("display_name"),
                u.get("real_name"),
                prof.get("real_name"),
                u.get("name"),
                prof.get("email"),
            )
            if any((a or "").strip().casefold() == key for a in aliases):
                hits[u["id"]] = u
        return self._make(next(iter(hits.values()))) if len(hits) == 1 else None

    async def is_exec(self, user: int) -> bool:
        uid = await self._slack_user(user)
        return uid is not None and await self.port.is_admin_user(uid)

    async def rank(self, user: int) -> int:
        """Slack has admins/owners only: 1 for them, else 0 (see the design doc)."""
        return 1 if await self.is_exec(user) else 0

    # ------------------------------------------------------------------ rooms
    async def _room_list(self, *, fresh: bool = False) -> list[dict]:
        now = self.port._clock()
        if not fresh and self._rooms and now - self._rooms[0] < LIST_TTL_S:
            return self._rooms[1]
        try:
            rows = await self.port._pages(
                self.port.client.users_conversations,
                "channels",
                types="public_channel,private_channel,mpim",
                exclude_archived=True,
                limit=200,
            )
        except Exception as e:
            logger.warning("users.conversations failed (%s)", safe_exc(e))
            return self._rooms[1] if self._rooms else []
        rows = [r for r in rows if r.get("id")]
        for r in rows:
            self.learn_room(r["id"])
        self._rooms = (now, rows)
        return rows

    @staticmethod
    def _shared(row: dict) -> bool:
        """Shared, or not provably non-shared (flags omitted): the audience is unknown."""
        return share_state(row) is not False

    async def rooms(self) -> list[Room]:
        """Conversations Salid is in (public, private, group DMs)."""
        return [
            Room(
                self.learn_room(r["id"]),
                r.get("name") or r["id"],
                r["id"],
                private=bool(r.get("is_private") or r.get("is_mpim")),
                shared=self._shared(r),
            )
            for r in await self._room_list()
        ]

    async def _local_human(self, uid: str) -> bool:
        u = await self.port.user(uid)
        return (
            bool(u)
            and u.get("team_id") == self.port.team_id
            and not (u.get("is_bot") or u.get("deleted"))
        )

    async def _audience(self, channel: str) -> frozenset[str] | None:
        """The exact readers of one conversation, or None (unknown). The only way a room's
        audience enters a `RoomView`, for the answering room and for every other room.

        - 1:1 DM: only after an explicit `is_im` and a validated (own, human) peer.
        - group DM: only after an explicit `is_mpim`, not shared, every member validated.
        - channels: `SlackPort.room_audience` (private and explicitly non-shared only)."""
        info = await self.port._conversation_info(channel)
        if info is None:
            return None
        if info.get("is_im") is True:
            peer = info.get("user")
            return frozenset({peer}) if peer and await self._local_human(peer) else None
        if info.get("is_mpim") is True:
            if share_state(info) is True:
                return None
            members = await self.port.members(channel, fresh=True)
            if members is None:
                return None
            for m in members:
                if not await self._local_human(m):
                    return None
            return members
        return await self.port.room_audience(channel)

    async def room_view(self, room: int) -> RoomView:
        """Same rules as `SlackPort.room_view`: exact audience only for a private, explicitly
        non-shared conversation (or a validated DM); anything else (public, Slack Connect,
        omitted share flags, error) is unknown = fail closed. `rooms` lists the OTHER
        conversations Salid is in whose audience is exact by the same rule."""
        channel = await self._slack_room(room)
        if channel is None:
            return RoomView(None, [])
        try:
            audience = await self._audience(channel)
        except Exception as e:
            logger.warning("room audience failed (%s)", safe_exc(e))
            audience = None
        if audience is None:
            return RoomView(None, [])
        others: list[ChannelAccess] = []
        for r in (await self._room_list())[: MAX_ROOMS + 1]:
            if r["id"] == channel:
                continue
            try:
                members = await self._audience(r["id"])
            except Exception as e:
                logger.warning("room audience failed (%s)", safe_exc(e))
                members = None
            if members is not None:
                others.append(
                    ChannelAccess(
                        self.learn_room(r["id"]),
                        r.get("name") or r["id"],
                        frozenset(self.learn_user(m) for m in members),
                    )
                )
            if len(others) >= MAX_ROOMS:
                break
        return RoomView(frozenset(self.learn_user(m) for m in audience), others)

    # ------------------------------------------------------------------ output
    def _format(self, text: str) -> str:
        """mrkdwn for `text`; only `mention()` sentinels of local humans become real pings."""
        out = textfmt.to_slack(text)
        return _SENTINEL.sub(lambda m: f"<@{m[1]}>" if m[1] in self._humans else "@someone", out)

    async def _send(
        self, channel: str, text: str, thread_ts, files: Sequence[FileOut], buttons
    ) -> MsgRef:
        if self.port.revoked:
            raise WorkspaceError("the Slack app was uninstalled from this workspace")
        check_buttons(buttons)
        parts = textfmt.split_text(self._format(text), CHUNK) or [""]
        resp: dict = {}
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            kw: dict = {
                "channel": channel,
                "text": part or " ",  # the notification / fallback text
                "mrkdwn": True,
                "unfurl_links": False,
            }
            if last and buttons:
                kw["blocks"] = blocks(part, buttons)
            if thread_ts:
                kw["thread_ts"] = thread_ts
            resp = await self.port._call(self.port.client.chat_postMessage, **kw)
        ts = resp.get("ts") or ""
        if files:
            # chat.postMessage cannot carry files; upload them into the same place right after.
            uploads = [{"file": f.data, "filename": f.name, "title": f.name} for f in files]
            kw = {"channel": channel, "file_uploads": uploads}
            if thread_ts:
                kw["thread_ts"] = thread_ts
            await self.port._call(self.port.client.files_upload_v2, **kw)
        return MsgRef(PLATFORM, self.learn_room(channel), ts)

    async def post(
        self,
        room: int,
        text: str,
        *,
        thread: MsgRef | None = None,
        files: Sequence[FileOut] = (),
        buttons: Sequence[Button] = (),
    ) -> MsgRef:
        channel = await self._slack_room(room)
        if channel is None:
            raise WorkspaceError(f"unknown room {room}")
        return await self._send(
            channel, text, thread.platform_ref if thread else None, files, buttons
        )

    async def dm(
        self, user: int, text: str, *, files: Sequence[FileOut] = (), buttons: Sequence[Button] = ()
    ) -> MsgRef | None:
        uid = await self._slack_user(user)
        if uid is None:
            return None
        try:
            opened = await self.port._call(self.port.client.conversations_open, users=uid)
            channel = (opened.get("channel") or {}).get("id")
            if not channel:
                return None
            return await self._send(channel, text, None, files, buttons)
        except WorkspaceError:
            raise
        except Exception as e:
            logger.warning("could not DM a user (%s)", safe_exc(e))
            return None

    async def edit(self, ref: MsgRef, text: str, *, buttons: Sequence[Button] = ()) -> None:
        channel = await self._slack_room(ref.room_id)
        if channel is None:
            raise WorkspaceError(f"unknown room {ref.room_id}")
        check_buttons(buttons)
        body = self._format(text)[:CHUNK]
        # Always send blocks: chat.update keeps the old ones when none are given, so an empty
        # button list would not remove them.
        await self.port._call(
            self.port.client.chat_update,
            channel=channel,
            ts=ref.platform_ref,
            text=body or " ",
            blocks=blocks(body, buttons),
            mrkdwn=True,
        )

    def mention(self, user: int) -> str:
        uid = self._user_ids.get(user)
        return f"{MENTION_OPEN}{uid}{MENTION_CLOSE}" if uid in self._humans else "@someone"

    def supports(self, feature: str) -> bool:
        return feature in CAPABILITIES and feature not in UNSUPPORTED
