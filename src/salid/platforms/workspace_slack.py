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
from .slack import CHUNK, LIST_TTL_S, MAX_ROOMS, PLATFORM, SlackPort

logger = logging.getLogger(__name__)

# Slack documents 1 GB as the largest upload; Salid's own paths (Gemini, PDFs, decks) never need
# that, so be conservative.
UPLOAD_LIMIT = 100 * 1024 * 1024
USERS_TTL_S = 600.0
MENTION = re.compile(r"^<@([UW][A-Z0-9]+)(?:\|[^>]*)?>$")
RAW_ID = re.compile(r"^[UW][A-Z0-9]+$")
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
        self._users = (now, keep)
        return keep

    def _make(self, u: dict) -> Person:
        email = (u.get("profile") or {}).get("email") or None
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
            u = await self.port.user(uid)  # users.info, cached
            if u:
                # Same fail-closed rule as the admin check: this workspace's own humans only.
                own = u.get("team_id") == self.port.team_id
                if u.get("is_bot") or u.get("deleted") or not own:
                    return None
                return self._make({**u, "id": u.get("id") or uid})
            if found:
                return None
            # not a user id after all (a name that looks like one): fall through to names
        everyone = await self._user_list()
        if "@" in text and not text.startswith("@") and " " not in text:  # an email
            key = text.casefold()
            hits = [
                u
                for u in everyone
                if ((u.get("profile") or {}).get("email") or "").casefold() == key
            ]
        else:
            key = text.lstrip("@").strip().casefold()
            hits = []
            for pick in (
                lambda u: (u.get("profile") or {}).get("display_name"),
                lambda u: u.get("real_name") or (u.get("profile") or {}).get("real_name"),
                lambda u: u.get("name"),
            ):
                hits = [u for u in everyone if (pick(u) or "").casefold() == key]
                if hits:
                    break
        # Ambiguous names are never guessed between.
        return self._make(hits[0]) if len(hits) == 1 else None

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
        return any(bool(row.get(k)) for k in ("is_shared", "is_ext_shared", "is_org_shared"))

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

    async def room_view(self, room: int) -> RoomView:
        """Same rules as `SlackPort.room_view`: exact audience only for a private, non-shared
        conversation; anything else (public, Slack Connect, error) is unknown = fail closed.
        `rooms` lists the other non-shared conversations Salid is in whose members are known."""
        channel = await self._slack_room(room)
        if channel is None:
            return RoomView(None, [])
        try:
            if channel.startswith("D"):  # a 1:1 DM: its audience is the other person
                info = await self.port._conversation_info(channel)
                peer = (info or {}).get("user")
                audience = frozenset({peer}) if peer else None
            else:
                audience = await self.port.room_audience(channel)
        except Exception as e:
            logger.warning("room audience failed (%s)", safe_exc(e))
            audience = None
        if audience is None:
            return RoomView(None, [])
        others: list[ChannelAccess] = []
        for r in (await self._room_list())[: MAX_ROOMS + 1]:
            if r["id"] == channel or self._shared(r):
                continue
            members = await self.port.members(r["id"])
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
    async def _send(
        self, channel: str, text: str, thread_ts, files: Sequence[FileOut], buttons
    ) -> MsgRef:
        if self.port.revoked:
            raise WorkspaceError("the Slack app was uninstalled from this workspace")
        check_buttons(buttons)
        parts = textfmt.split_text(textfmt.to_slack(text), CHUNK) or [""]
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
        body = textfmt.to_slack(text)[:CHUNK]
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
        return f"<@{uid}>" if uid else "@someone"

    def supports(self, feature: str) -> bool:
        return feature in CAPABILITIES and feature not in UNSUPPORTED
