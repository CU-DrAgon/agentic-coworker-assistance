"""Slack adapter: Socket Mode events -> `Incoming`, and Salid's replies back to Slack.

`SlackPort` is the `core.Port` for Slack; `SlackAdapter` translates raw events (and is what the
tests drive with a fake Web API client); `build_app` wires both to a Bolt app in Socket Mode, so
no public endpoint is needed. Tokens never reach a log line.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from ..qa import MAX_FILE_BYTES
from . import textfmt
from .config import PlatformSettings
from .core import Assistant, FileRef, HistLine, Incoming, RoomInfo, RoomView, safe_exc

logger = logging.getLogger(__name__)

PLATFORM = "slack"
CHUNK = 3000  # characters per message (Slack allows far more; shorter reads better)
MAX_MESSAGES = 10
MEMBERS_TTL_S = 60.0
LIST_TTL_S = 300.0
USER_TTL_S = 600.0
MAX_MEMBERS = 5000  # a bigger room: audience unknown, so memory stays in that room
MAX_ROOMS = 60  # other rooms considered for memory scope
MAX_PAGES = 20
ACCEPTED_SUBTYPES = {None, "file_share", "thread_broadcast"}


class SlackPort:
    """`core.Port` for one Slack workspace. `client` is a `slack_sdk.web.async_client.AsyncWebClient`."""

    platform = PLATFORM
    max_messages = MAX_MESSAGES

    def __init__(self, client, team_id: str, bot_user_id: str, clock=time.monotonic) -> None:
        self.client, self.team_id, self.bot_user_id, self._clock = (
            client,
            team_id,
            bot_user_id,
            clock,
        )
        self._members: dict[str, tuple[float, frozenset[str]]] = {}
        self._bot_rooms: tuple[float, dict[str, tuple[str, str]]] | None = None
        self._users: dict[str, tuple[float, dict]] = {}
        self._room_names: dict[str, str] = {}

    # ------------------------------------------------------------ lookups
    async def user(self, uid: str) -> dict:
        """users.info (cached 10 min); {} when it cannot be read."""
        hit = self._users.get(uid)
        if hit is not None and self._clock() - hit[0] < USER_TTL_S:
            return hit[1]
        try:
            info = (await self.client.users_info(user=uid)).get("user") or {}
        except Exception as e:
            logger.warning("users.info failed (%s)", safe_exc(e))
            info = {}
        self._users[uid] = (self._clock(), info)
        return info

    async def user_name(self, uid: str) -> str:
        u = await self.user(uid)
        prof = u.get("profile") or {}
        return prof.get("display_name") or u.get("real_name") or u.get("name") or uid

    async def is_admin(self, msg: Incoming) -> bool:
        u = await self.user(msg.user)
        return bool(u.get("is_admin") or u.get("is_owner") or u.get("is_primary_owner"))

    async def room_name(self, channel: str) -> str:
        if channel in self._room_names:
            return self._room_names[channel]
        try:
            info = (await self.client.conversations_info(channel=channel)).get("channel") or {}
        except Exception as e:
            logger.warning("conversations.info failed (%s)", safe_exc(e))
            return ""
        name = info.get("name") or ""
        if name:
            self._room_names[channel] = name
        return name

    async def _pages(self, call, key: str, **kw) -> list[Any]:
        out: list[Any] = []
        cursor = ""
        for _ in range(MAX_PAGES):
            resp = await call(**kw, **({"cursor": cursor} if cursor else {}))
            out.extend(resp.get(key) or [])
            cursor = ((resp.get("response_metadata") or {}).get("next_cursor")) or ""
            if not cursor:
                return out
        raise RuntimeError("too many pages")

    async def members(self, channel: str, *, fresh: bool = False) -> frozenset[str] | None:
        """Human members of a conversation (the bot removed); None when unknown or too large."""
        hit = self._members.get(channel)
        if not fresh and hit is not None and self._clock() - hit[0] < MEMBERS_TTL_S:
            return hit[1]
        try:
            got = await self._pages(
                self.client.conversations_members, "members", channel=channel, limit=200
            )
        except Exception as e:
            logger.warning("conversations.members failed (%s)", safe_exc(e))
            return None
        if len(got) > MAX_MEMBERS:
            return None
        members = frozenset(m for m in got if m != self.bot_user_id)
        self._members[channel] = (self._clock(), members)
        return members

    async def _convs(self, **kw) -> dict[str, tuple[str, str]]:
        rows = await self._pages(
            self.client.users_conversations,
            "channels",
            types="public_channel,private_channel,mpim",
            exclude_archived=True,
            limit=200,
            **kw,
        )
        return {r["id"]: (r.get("name") or r["id"], r.get("id", "")) for r in rows if r.get("id")}

    async def bot_rooms(self) -> dict[str, tuple[str, str]]:
        """Conversations Salid is in (public, private, group DMs): id -> (name, id)."""
        if self._bot_rooms and self._clock() - self._bot_rooms[0] < LIST_TTL_S:
            return self._bot_rooms[1]
        rooms = await self._convs()
        self._bot_rooms = (self._clock(), rooms)
        return rooms

    # ------------------------------------------------------------ core.Port
    async def room_view(self, msg: Incoming) -> RoomView:
        if msg.is_direct:
            audience: frozenset[str] | None = frozenset({msg.user})
        else:
            audience = await self.members(msg.room, fresh=True)
        if audience is None:
            return RoomView(None, [])
        mine = await self._convs(user=msg.user)  # the asker's conversations, read live
        shared = await self.bot_rooms()
        candidates = [c for c in mine if c in shared and c != msg.room][:MAX_ROOMS]
        rooms = []
        for cid in candidates:
            members = await self.members(cid)
            if members is not None:
                rooms.append(RoomInfo(cid, shared[cid][0], members))
        return RoomView(audience, rooms)

    def format(self, text: str) -> list[str]:
        return textfmt.split_text(textfmt.to_slack(text), CHUNK)

    def _thread(self, msg: Incoming) -> str | None:
        return msg.thread or (None if msg.is_direct else msg.message)

    async def ack(self, msg: Incoming, text: str) -> object | None:
        resp = await self.client.chat_postMessage(
            channel=msg.room, thread_ts=self._thread(msg), text=textfmt.to_slack(text), mrkdwn=True
        )
        return (msg.room, resp.get("ts")) if resp.get("ts") else None

    async def drop_ack(self, ack: object) -> None:
        channel, ts = ack
        try:
            await self.client.chat_delete(channel=channel, ts=ts)
        except Exception as e:
            logger.warning("could not remove the acknowledgement (%s)", safe_exc(e))

    async def deliver(self, msg: Incoming, texts: list[str], ack: object | None) -> list[str]:
        sent: list[str] = []
        rest = list(texts)
        if ack is not None and rest:
            channel, ts = ack
            try:
                await self.client.chat_update(channel=channel, ts=ts, text=rest[0], mrkdwn=True)
                sent.append(ts)
                rest.pop(0)
            except Exception as e:
                logger.warning("could not edit the acknowledgement (%s)", safe_exc(e))
                await self.drop_ack(ack)
        for text in rest:
            resp = await self.client.chat_postMessage(
                channel=msg.room,
                thread_ts=self._thread(msg),
                text=text,
                mrkdwn=True,
                unfurl_links=False,
                reply_broadcast=False,
            )
            sent.append(resp.get("ts") or "")
        return sent

    async def history(self, msg: Incoming, limit: int) -> list[HistLine]:
        """Recent messages of the thread (or the room), oldest first, read from Slack."""
        try:
            if msg.thread:
                resp = await self.client.conversations_replies(
                    channel=msg.room, ts=msg.thread, limit=limit
                )
                raw = resp.get("messages") or []
            else:
                resp = await self.client.conversations_history(channel=msg.room, limit=limit)
                raw = list(reversed(resp.get("messages") or []))
        except Exception as e:
            logger.warning("could not read Slack history (%s)", safe_exc(e))
            return []
        out = []
        for m in raw:
            if m.get("subtype") not in ACCEPTED_SUBTYPES:
                continue
            mine = m.get("user") == self.bot_user_id
            if m.get("bot_id") and not mine:
                continue
            uid = m.get("user") or ""
            text = textfmt.slack_to_plain(m.get("text") or "", self.bot_user_id)
            if not text:
                continue
            who = "Salid" if mine else await self.user_name(uid)
            out.append(HistLine(_ts(m.get("ts")), who, text, mine))
        return out


def _ts(ts: str | None) -> datetime:
    try:
        return datetime.fromtimestamp(float(ts), UTC)
    except (TypeError, ValueError):
        return datetime.now(UTC)


def download(token: str, url: str, get=None) -> bytes:
    """A Slack private file (blocking; called in a worker thread). Refuses a login page and files
    over the size limit; the token is sent only to Slack file hosts."""
    host = httpx.URL(url).host or ""
    if not (
        host == "files.slack.com" or host.endswith(".slack.com") or host.endswith(".slack-edge.com")
    ):
        raise ValueError("not a Slack file host")
    headers = {"Authorization": f"Bearer {token}"}
    if get is not None:
        return get(url, headers)
    data = bytearray()
    with httpx.stream("GET", url, headers=headers, timeout=30.0, follow_redirects=False) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("content-type", ""):
            raise ValueError("Slack returned a web page (is files:read granted?)")
        for chunk in r.iter_bytes():
            data.extend(chunk)
            if len(data) > MAX_FILE_BYTES:
                raise ValueError("file over the size limit")
    return bytes(data)


class SlackAdapter:
    """Translates Slack events. `port` answers the Web API calls; `fetch(url) -> bytes` downloads
    a private file (the token stays inside it)."""

    def __init__(
        self,
        assistant: Assistant,
        port: SlackPort,
        fetch: Callable[[str], bytes],
        *,
        on_joined: Callable[[str], None] | None = None,
    ) -> None:
        self.a, self.port, self.fetch, self.on_joined = assistant, port, fetch, on_joined

    async def on_message(
        self, event: dict, team_id: str | None = None, *, mention: bool = False
    ) -> None:
        """A `message` or `app_mention` event. Edits and deletions update the memory; everything
        from a bot (Salid included) is ignored."""
        team = team_id or event.get("team") or self.port.team_id
        sub = event.get("subtype")
        if sub == "message_changed":
            await self._edited(event, team)
            return
        if sub == "message_deleted":
            if event.get("deleted_ts") and event.get("channel"):
                await self.a.on_delete(PLATFORM, team, event["channel"], event["deleted_ts"])
            return
        if sub not in ACCEPTED_SUBTYPES or event.get("bot_id"):
            return
        user = event.get("user")
        if (
            not user
            or user == self.port.bot_user_id
            or not event.get("channel")
            or not event.get("ts")
        ):
            return
        msg = await self._incoming(event, team, mention)
        await self.a.handle(msg, self.port)

    async def _incoming(self, event: dict, team: str, mention: bool) -> Incoming:
        raw = event.get("text") or ""
        bot = self.port.bot_user_id
        mentioned = textfmt.slack_mentions(raw)
        names = {}
        for uid in dict.fromkeys(m for m in mentioned if m != bot):
            names[uid] = await self.port.user_name(uid)
        text = textfmt.slack_to_plain(raw, bot, names.get)
        ts = event["ts"]
        thread = event.get("thread_ts")
        direct = event.get("channel_type") == "im"
        room_name = "" if direct else await self.port.room_name(event["channel"])
        return Incoming(
            platform=PLATFORM,
            tenant=team,
            room=event["channel"],
            user=event["user"],
            user_name=await self.port.user_name(event["user"]),
            text=text,
            message=ts,
            at=_ts(ts),
            is_direct=direct,
            mentions_bot=mention or bot in mentioned,
            mentions_others=any(m != bot for m in mentioned),
            thread=thread if thread and thread != ts else None,
            replies_to_bot=bool(event.get("parent_user_id") == bot and thread and thread != ts),
            files=self._files(event.get("files") or []),
            room_name=room_name,
        )

    def _files(self, raw: list[dict]) -> list[FileRef]:
        out = []
        for f in raw:
            url = f.get("url_private_download") or f.get("url_private")
            if not url or f.get("mode") == "tombstone":
                continue
            out.append(
                FileRef(
                    name=f.get("name") or f.get("title") or "file",
                    mime=f.get("mimetype"),
                    size=int(f.get("size") or 0),
                    fetch=lambda u=url: self.fetch(u),
                )
            )
        return out

    async def _edited(self, event: dict, team: str) -> None:
        new = event.get("message") or {}
        if new.get("bot_id") or not new.get("user") or new.get("user") == self.port.bot_user_id:
            return
        if not event.get("channel") or not new.get("ts"):
            return
        text = textfmt.slack_to_plain(new.get("text") or "", self.port.bot_user_id)
        msg = Incoming(
            platform=PLATFORM,
            tenant=team,
            room=event["channel"],
            user=new["user"],
            user_name="",
            text=text,
            message=new["ts"],
            at=_ts(new["ts"]),
            thread=new.get("thread_ts") if new.get("thread_ts") != new["ts"] else None,
        )
        await self.a.on_edit(msg)

    def joined(self, event: dict) -> None:
        """The bot was added to a channel: its history can be imported."""
        if event.get("user") == self.port.bot_user_id and self.on_joined is not None:
            self.on_joined(event.get("channel") or "")


async def connect(ps: PlatformSettings, assistant: Assistant):
    """Log in, build the Bolt app and return (handler, adapter, port). The caller runs
    `await handler.start_async()`. Tokens are only handed to Slack's SDK."""
    from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
    from slack_bolt.async_app import AsyncApp
    from slack_sdk.http_retry.builtin_async_handlers import AsyncRateLimitErrorRetryHandler
    from slack_sdk.web.async_client import AsyncWebClient

    client = AsyncWebClient(token=ps.slack_bot_token)
    client.retry_handlers.append(AsyncRateLimitErrorRetryHandler(max_retry_count=2))
    auth = await client.auth_test()
    team, bot = auth["team_id"], auth["user_id"]
    port = SlackPort(client, team, bot)
    token = ps.slack_bot_token

    adapter = SlackAdapter(assistant, port, lambda url: download(token, url))
    app = AsyncApp(client=client)

    @app.event("message")
    async def _message(event, context):
        await adapter.on_message(event, context.team_id)

    @app.event("app_mention")
    async def _mention(event, context):
        await adapter.on_message(event, context.team_id, mention=True)

    @app.event("member_joined_channel")
    async def _joined(event):
        adapter.joined(event)

    return AsyncSocketModeHandler(app, ps.slack_app_token), adapter, port


async def check(ps: PlatformSettings) -> dict:
    """Read-only token check for the smoke script: auth.test. No secrets in the result."""
    from slack_sdk.web.async_client import AsyncWebClient

    r = await AsyncWebClient(token=ps.slack_bot_token).auth_test()
    return {"team": r.get("team"), "team_id": r.get("team_id"), "bot_user_id": r.get("user_id")}
