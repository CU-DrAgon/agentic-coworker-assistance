"""Slack adapter: Socket Mode events -> `Incoming`, and Salid's replies back to Slack.

`SlackPort` is the `core.Port` for Slack; `SlackAdapter` translates raw events (and is what the
tests drive with a fake Web API client); `build_app` wires both to a Bolt app in Socket Mode, so
no public endpoint is needed. Tokens never reach a log line.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from ..qa import MAX_FILE_BYTES
from . import ids, textfmt
from .config import PlatformSettings
from .core import AccountState, Assistant, FileRef, HistLine, Incoming, RoomInfo, RoomView, safe_exc

logger = logging.getLogger(__name__)

PLATFORM = "slack"
CHUNK = 3000  # characters per message (Slack allows far more; shorter reads better)
MAX_MESSAGES = 10
MEMBERS_TTL_S = 60.0
LIST_TTL_S = 300.0
USER_TTL_S = 600.0
USER_FAIL_TTL_S = 30.0  # a failed users.info is retried after this long, not after USER_TTL_S
MAX_MEMBERS = 5000  # a bigger room: audience unknown, so memory stays in that room
MAX_ROOMS = 60  # other rooms considered for memory scope
MAX_PAGES = 20
ACCEPTED_SUBTYPES = {None, "file_share", "thread_broadcast"}
ROOM_TTL_S = 600.0  # channel name / privacy caches expire after this long
HISTORY_TIMEOUT_S = 3.0  # the live read in the answer path gives up after this long
HISTORY_BLOCK_MAX_S = 300.0  # after a 429 the live read is skipped for Retry-After (capped)
POST_GAP_S = 1.1  # chat.postMessage: about one message per second per channel
POST_TRIES = 5  # attempts per message when Slack answers 429
POST_WAIT_MAX_S = 60.0
MENTION_WAIT_S = 1.0  # an app_mention waits this long for its twin `message` event
MAX_FILE_LOOKUPS = 4
REVOKED_EVENTS = {"app_uninstalled", "tokens_revoked"}
USER_EVENTS = {"user_change"}  # a profile / role / deactivation change: re-verify the person
ROOM_EVENTS = {
    "channel_rename", "group_rename", "channel_archive", "group_archive", "channel_unarchive",
    "group_unarchive", "channel_left", "group_left", "channel_deleted", "group_deleted",
    "member_joined_channel", "member_left_channel",
}  # fmt: skip
# One Bolt listener for every non-message event (keeps the listener count at three).
LIFECYCLE_EVENTS = re.compile(
    "^(" + "|".join(sorted(ROOM_EVENTS | REVOKED_EVENTS | USER_EVENTS)) + ")$"
)


def retry_after(e: BaseException) -> float | None:
    """Seconds Slack asks us to wait when `e` is an HTTP 429, else None."""
    resp = getattr(e, "response", None)
    if resp is None or getattr(resp, "status_code", None) != 429:
        return None
    try:
        headers = getattr(resp, "headers", None) or {}
        for k, v in headers.items():
            if str(k).lower() == "retry-after":
                return max(float(v), 0.0)
    except (TypeError, ValueError):
        pass
    return 1.0


SHARE_FIELDS = ("is_shared", "is_ext_shared", "is_org_shared")


def share_state(info: dict) -> bool | None:
    """Tri-state Slack Connect / org-share flag of a conversation (users.conversations row or
    conversations.info): True when any flag is set, False ONLY when all three flags are present
    and explicitly False, None (unknown) when Slack omitted any of them."""
    if any(bool(info.get(k)) for k in SHARE_FIELDS):
        return True
    if all(info.get(k) is False for k in SHARE_FIELDS):
        return False
    return None


class SlackPort:
    """`core.Port` for one Slack workspace. `client` is a `slack_sdk.web.async_client.AsyncWebClient`."""

    platform = PLATFORM
    max_messages = MAX_MESSAGES

    def __init__(
        self,
        client,
        team_id: str,
        bot_user_id: str,
        clock=time.monotonic,
        *,
        history_client=None,
        import_client=None,
        sleep=asyncio.sleep,
        history_timeout: float = HISTORY_TIMEOUT_S,
    ) -> None:
        self.client, self.team_id, self.bot_user_id, self._clock = (
            client,
            team_id,
            bot_user_id,
            clock,
        )
        # Clients without the SDK's sleeping 429 handler: the live read must never stall an
        # answer and the importer paces itself (see slack_import.Throttle).
        self.history_client = history_client or client
        self.import_client = import_client or client
        self._sleep, self._history_timeout = sleep, history_timeout
        self.revoked = False  # app_uninstalled / tokens_revoked: this tenant is gone
        self._history_blocked_until = 0.0
        self._history_warned = False
        self._next_post: dict[str, float] = {}
        self._members: dict[str, tuple[float, frozenset[str]]] = {}
        self._bot_rooms: tuple[float, dict[str, tuple[str, str]]] | None = None
        self._users: dict[str, tuple[float, dict]] = {}
        self._room_names: dict[str, tuple[float, str]] = {}
        self._room_private: dict[str, tuple[float, bool]] = {}
        self._room_shared: dict[str, tuple[float, bool | None]] = {}

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
        # a failure is remembered only briefly, so one blip does not blind us for ten minutes
        stamp = self._clock() if info else self._clock() - USER_TTL_S + USER_FAIL_TTL_S
        self._users[uid] = (stamp, info)
        return info

    def forget_user(self, uid: str) -> None:
        """The person changed (user_change): the next lookup asks Slack again."""
        self._users.pop(uid, None)

    async def user_name(self, uid: str) -> str:
        u = await self.user(uid)
        prof = u.get("profile") or {}
        return prof.get("display_name") or u.get("real_name") or u.get("name") or uid

    async def account_state(self, uid: str) -> AccountState:
        """What Slack says about this person NOW (users.info, cached up to 10 minutes):

        * ``member``: a real, full member of THIS workspace with a readable profile email;
        * ``revoked``: a bot, deleted, guest (single / multi-channel) or a person of another
          workspace (Slack Connect): not a member whose address means anything here;
        * ``unavailable``: the lookup failed or the email is not readable (scope missing).
          Nothing is known, so the caller must fail closed without removing any link."""
        u = await self.user(uid)
        if not u:
            return AccountState("unavailable", "")
        if u.get("is_bot") or u.get("deleted") or u.get("team_id") != self.team_id:
            return AccountState("revoked", "")
        if u.get("is_restricted") or u.get("is_ultra_restricted"):
            return AccountState("revoked", "")  # guests: their address is not the company's
        email = str((u.get("profile") or {}).get("email") or "").strip().lower()
        return AccountState("member", email) if email else AccountState("unavailable", "")

    async def profile_email(self, uid: str) -> str:
        """The verified-by-Slack profile email (lower case) of a real, current member of THIS
        workspace; "" otherwise (see `account_state`). Fails closed."""
        return (await self.account_state(uid)).email

    async def is_admin(self, msg: Incoming) -> bool:
        return await self.is_admin_user(msg.user)

    async def is_admin_user(self, uid: str) -> bool:
        """Workspace admin/owner of THIS workspace only. A Slack Connect user from another
        workspace can be an admin there (users.info says so in a shared channel): that must never
        make them an executive here. A bot, a deleted user or a missing team_id fails closed."""
        u = await self.user(uid)
        if u.get("is_bot") or u.get("deleted") or u.get("team_id") != self.team_id:
            return False
        return bool(u.get("is_admin") or u.get("is_owner") or u.get("is_primary_owner"))

    def _fresh(self, cache: dict, channel: str):
        hit = cache.get(channel)
        if hit is not None and self._clock() - hit[0] < ROOM_TTL_S:
            return hit[1]
        return None

    async def room_name(self, channel: str) -> str:
        known = self._fresh(self._room_names, channel)
        if known is not None:
            return known
        info = await self._conversation_info(channel)
        if info is None:
            return ""
        return info.get("name") or ""

    async def _conversation_info(self, channel: str) -> dict | None:
        try:
            info = (await self.client.conversations_info(channel=channel)).get("channel") or {}
        except Exception as e:
            logger.warning("conversations.info failed (%s)", safe_exc(e))
            return None
        private = info.get("is_private")
        if isinstance(private, bool):
            self._room_private[channel] = (self._clock(), private)
        # Slack Connect / org-shared: people of other workspaces or orgs are in the room, so its
        # audience cannot be listed from this workspace.
        self._room_shared[channel] = (self._clock(), share_state(info))
        if info.get("name"):
            self._room_names[channel] = (self._clock(), info["name"])
        return info

    # ------------------------------------------------------------ cache invalidation
    def forget_room(self, channel: str) -> None:
        """A channel was renamed, archived, left or deleted: drop what we cached about it."""
        self._room_names.pop(channel, None)
        self._room_private.pop(channel, None)
        self._room_shared.pop(channel, None)
        self._members.pop(channel, None)
        self._bot_rooms = None

    def members_changed(self, channel: str, user: str | None = None) -> None:
        self._members.pop(channel, None)
        if user == self.bot_user_id:
            self._bot_rooms = None  # the list of rooms Salid is in changed

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

    async def readable_rooms(self, uid: str) -> dict[str, str] | None:
        """Conversations Salid is in that this member of THIS workspace can read: channel id ->
        name. Public non-shared channels (a full member may read any; guests get none of them),
        plus private channels and group DMs that are not shared and where Slack lists the member
        (users.conversations with ``user``). Slack Connect / org-shared rooms and rooms whose
        share flags are missing are never included. None when the person is not a current member
        of this team or Slack cannot answer: nothing is known, so the caller uses nothing."""
        u = await self.user(uid)
        if not u or u.get("is_bot") or u.get("deleted") or u.get("team_id") != self.team_id:
            return None
        guest = bool(u.get("is_restricted") or u.get("is_ultra_restricted"))
        kw = dict(types="public_channel,private_channel,mpim", exclude_archived=True, limit=200)
        try:
            ours = await self._pages(self.client.users_conversations, "channels", **kw)
            mine = await self._pages(self.client.users_conversations, "channels", user=uid, **kw)
        except Exception as e:
            logger.warning("users.conversations failed (%s)", safe_exc(e))
            return None
        member_of = {r["id"] for r in mine if r.get("id")}
        out: dict[str, str] = {}
        for r in ours:
            cid = r.get("id")
            if not cid or share_state(r) is not False:
                continue
            private = bool(r.get("is_private") or r.get("is_mpim"))
            if (private and cid in member_of) or (not private and not guest):
                out[cid] = r.get("name") or cid
        return out

    async def bot_rooms(self) -> dict[str, tuple[str, str]]:
        """Conversations Salid is in (public, private, group DMs): id -> (name, id)."""
        if self._bot_rooms and self._clock() - self._bot_rooms[0] < LIST_TTL_S:
            return self._bot_rooms[1]
        rooms = await self._convs()
        self._bot_rooms = (self._clock(), rooms)
        return rooms

    # ------------------------------------------------------------ core.Port
    async def room_audience(self, channel: str) -> frozenset[str] | None:
        """Readers of a channel when Slack lists them exactly, else None (fail closed).
        Shared (Slack Connect) and unknown-privacy channels are None; conversations.members is
        the full audience only for private conversations (a public channel is readable by people
        who have not joined)."""
        private = self._fresh(self._room_private, channel)
        hit = self._room_shared.get(channel)
        fresh = hit is not None and self._clock() - hit[0] < ROOM_TTL_S
        if private is None or not fresh:
            await self._conversation_info(channel)
            private = self._fresh(self._room_private, channel)
            hit = self._room_shared.get(channel)
            fresh = hit is not None and self._clock() - hit[0] < ROOM_TTL_S
        shared = hit[1] if fresh and hit is not None else None
        if shared is not False or not private:  # shared, unknown (fields omitted) or public
            return None
        return await self.members(channel, fresh=True)

    async def room_view(self, msg: Incoming) -> RoomView:
        if msg.is_direct:
            audience: frozenset[str] | None = frozenset({msg.user})
        else:
            audience = await self.room_audience(msg.room)
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

    async def _pace(self, channel: str) -> None:
        """Reserve the next posting slot of a channel (>= POST_GAP_S after the previous one)."""
        now = self._clock()
        start = max(now, self._next_post.get(channel, 0.0))
        self._next_post[channel] = start + POST_GAP_S
        if len(self._next_post) > 1000:
            self._next_post = {c: t for c, t in self._next_post.items() if t > now}
        if start > now:
            await self._sleep(start - now)

    async def _call(self, fn, **kw):
        """A Web API write: paced per channel, and retried when Slack answers 429 (waiting the
        Retry-After it asks for). The `channel` keyword is passed through to Slack."""
        channel = kw.get("channel")
        for attempt in range(POST_TRIES):
            if channel is not None:
                await self._pace(channel)
            try:
                return await fn(**kw)
            except Exception as e:
                wait = retry_after(e)
                if wait is None or attempt == POST_TRIES - 1:
                    raise
                logger.info("slack asked us to wait %.0fs (429)", wait)
                await self._sleep(min(wait, POST_WAIT_MAX_S))
        raise AssertionError("unreachable")

    async def ack(self, msg: Incoming, text: str) -> object | None:
        if self.revoked:
            return None
        resp = await self._call(
            self.client.chat_postMessage,
            channel=msg.room,
            thread_ts=self._thread(msg),
            text=textfmt.to_slack(text),
            mrkdwn=True,
        )
        return (msg.room, resp.get("ts")) if resp.get("ts") else None

    async def drop_ack(self, ack: object) -> None:
        channel, ts = ack
        try:
            await self.client.chat_delete(channel=channel, ts=ts)
        except Exception as e:
            logger.warning("could not remove the acknowledgement (%s)", safe_exc(e))

    async def deliver(self, msg: Incoming, texts: list[str], ack: object | None) -> list[str]:
        """Post the chunks in order, at most one per second per channel. A chunk that cannot be
        posted is logged and the later ones are still tried; the first failure is raised after."""
        if self.revoked:
            return []  # uninstalled while the answer was in flight: post nothing
        sent: list[str] = []
        rest = list(texts)
        if ack is not None and rest:
            channel, ts = ack
            try:
                await self._call(
                    self.client.chat_update, channel=channel, ts=ts, text=rest[0], mrkdwn=True
                )
                sent.append(ts)
                rest.pop(0)
            except Exception as e:
                logger.warning("could not edit the acknowledgement (%s)", safe_exc(e))
                await self.drop_ack(ack)
        failure: Exception | None = None
        for i, text in enumerate(rest):
            try:
                resp = await self._call(
                    self.client.chat_postMessage,
                    channel=msg.room,
                    thread_ts=self._thread(msg),
                    text=text,
                    mrkdwn=True,
                    unfurl_links=False,
                    reply_broadcast=False,
                )
            except Exception as e:
                logger.error(
                    "reply chunk %d/%d was not delivered (%s)", i + 1, len(rest), safe_exc(e)
                )
                failure = failure or e
                sent.append("")
                continue
            sent.append(resp.get("ts") or "")
        if failure is not None:
            raise failure
        return sent

    def _history_failed(self, why: str, wait: float = 0.0) -> None:
        if wait:
            self._history_blocked_until = self._clock() + min(wait, HISTORY_BLOCK_MAX_S)
        if not self._history_warned:
            self._history_warned = True
            logger.warning("could not read Slack history, using the live buffer (%s)", why)

    async def history(self, msg: Incoming, limit: int) -> list[HistLine]:
        """Recent messages of the thread (or the room), oldest first, read from Slack. Bounded
        by a short timeout, never retried and skipped while Slack is rate limiting us: the caller
        falls back to what the live events already filled in."""
        if self.revoked or self._clock() < self._history_blocked_until:
            return []
        try:
            out = await asyncio.wait_for(self._read_history(msg, limit), self._history_timeout)
        except TimeoutError:
            self._history_failed("timed out")
            return []
        except Exception as e:
            self._history_failed(safe_exc(e), retry_after(e) or 0.0)
            return []
        self._history_warned = False
        return out

    async def _read_history(self, msg: Incoming, limit: int) -> list[HistLine]:
        client = self.history_client
        if msg.thread:
            resp = await client.conversations_replies(channel=msg.room, ts=msg.thread, limit=limit)
            raw = resp.get("messages") or []
        else:
            resp = await client.conversations_history(channel=msg.room, limit=limit)
            raw = list(reversed(resp.get("messages") or []))
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

    async def file_info(self, file_id: str) -> dict | None:
        """files.info: the real file object of a Slack Connect stub (`check_file_info`)."""
        try:
            return (await self.client.files_info(file=file_id)).get("file") or None
        except Exception as e:
            logger.warning("files.info failed (%s)", safe_exc(e))
            return None


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
        merge_wait: float = MENTION_WAIT_S,
        delete_data: bool = True,
    ) -> None:
        self.a, self.port, self.fetch, self.on_joined = assistant, port, fetch, on_joined
        self.merge_wait, self.delete_data = merge_wait, delete_data
        # (channel, ts) of messages that arrived as a `message` event / app_mentions waiting for it
        self._message_seen: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._mention_held: dict[tuple[str, str], asyncio.Event] = {}

    async def on_message(
        self, event: dict, team_id: str | None = None, *, mention: bool = False
    ) -> None:
        """A `message` or `app_mention` event. Edits and deletions update the memory; everything
        from a bot (Salid included) is ignored."""
        if self.port.revoked:
            return
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
        key = (event["channel"], event["ts"])
        if mention:
            if key in self._message_seen:
                return  # the `message` event came first and carries everything
            if not event.get("files") and event.get("channel_type") != "im":
                # app_mention may arrive without its attachments: give the twin `message` event
                # a moment, it is the one to answer (core drops the second event of a message).
                held = self._mention_held[key] = asyncio.Event()
                try:
                    await asyncio.wait_for(held.wait(), self.merge_wait)
                    return
                except TimeoutError:
                    pass
                finally:
                    self._mention_held.pop(key, None)
        else:
            self._message_seen[key] = None
            while len(self._message_seen) > 500:
                self._message_seen.popitem(last=False)
            held = self._mention_held.get(key)
            if held is not None:
                held.set()
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
            files=await self._files(event.get("files") or []),
            room_name=room_name,
        )

    async def _files(self, raw: list[dict]) -> list[FileRef]:
        out = []
        lookups = 0
        for f in raw:
            if f.get("mode") == "tombstone":
                continue
            if (
                not (f.get("url_private_download") or f.get("url_private"))
                and f.get("file_access") == "check_file_info"
                and f.get("id")
                and lookups < MAX_FILE_LOOKUPS
            ):
                # Slack Connect: the event holds a stub, files.info has the real file object
                lookups += 1
                f = await self.port.file_info(f["id"]) or {}
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

    async def on_event(self, event: dict, team_id: str | None = None) -> None:
        """Everything that is not a message: channel changes (caches), membership and removal."""
        kind = event.get("type")
        team = team_id or event.get("team") or self.port.team_id
        if kind in REVOKED_EVENTS:
            await self._revoked(kind, event, team)
            return
        if kind in USER_EVENTS:
            await self._user_changed(event, team)
            return
        if self.port.revoked or kind not in ROOM_EVENTS:
            return
        ch = event.get("channel")
        channel = ch.get("id") if isinstance(ch, dict) else ch
        if not channel:
            return
        if kind in ("member_joined_channel", "member_left_channel"):
            self.port.members_changed(channel, event.get("user"))
            if kind == "member_joined_channel":
                self.joined(event)
            return
        if kind in ("channel_rename", "group_rename"):
            self.port.forget_room(channel)
            return
        self.port.forget_room(channel)  # archived, unarchived, left, deleted

    async def _user_changed(self, event: dict, team: str) -> None:
        """user_change: forget what we cached about the person and re-verify the account at
        once, so a deactivation, a switch to guest or a new email takes effect now instead of
        after the next refresh."""
        user = event.get("user")
        uid = user.get("id") if isinstance(user, dict) else user
        if not uid or self.port.revoked or team != self.port.team_id:
            return
        self.port.forget_user(uid)
        await self.a.recheck_account(PLATFORM, team, uid, self.port)

    async def _revoked(self, kind: str, event: dict, team: str) -> None:
        """app_uninstalled / tokens_revoked: stop answering this workspace and erase what Salid
        remembered of it (memory with its vectors). Only a revoked *bot* token counts."""
        if team != self.port.team_id or self.port.revoked:
            return
        if kind == "tokens_revoked" and not (event.get("tokens") or {}).get("bot"):
            return  # only user tokens were revoked; Salid uses the bot token
        self.port.revoked = True
        gid = ids.tenant_id(PLATFORM, team)
        logger.warning("slack %s: this workspace is removed, no more answers tenant=%d", kind, gid)
        await asyncio.to_thread(self.a.tenant_revoked, PLATFORM, team)
        if not self.delete_data:
            return
        try:
            await self.a.drain()  # queued memory writes must not land after the deletion
            n = await asyncio.to_thread(self.a.memory.delete_guild, gid)
            logger.info("slack tenant=%d memory erased (%s)", gid, n)
        except Exception as e:
            logger.error("could not erase the memory of tenant=%d (%s)", gid, safe_exc(e))
        self.a.tenants.pop(gid, None)
        store = getattr(self.a, "guilds_store", None)
        if store is not None:
            try:
                await asyncio.to_thread(store.set_active, gid, False)
            except Exception as e:
                logger.warning("could not mark tenant=%d inactive (%s)", gid, safe_exc(e))


async def connect(ps: PlatformSettings, assistant: Assistant):
    """Log in, build the Bolt app and return (handler, adapter, port). The caller runs
    `await handler.start_async()`. Tokens are only handed to Slack's SDK."""
    from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
    from slack_bolt.async_app import AsyncApp
    from slack_sdk.http_retry.builtin_async_handlers import AsyncRateLimitErrorRetryHandler
    from slack_sdk.web.async_client import AsyncWebClient

    client = AsyncWebClient(token=ps.slack_bot_token)
    client.retry_handlers.append(AsyncRateLimitErrorRetryHandler(max_retry_count=2))
    # no sleeping 429 handler on these: the live history read has a short timeout and the
    # importer paces itself and honours Retry-After
    live = AsyncWebClient(
        token=ps.slack_bot_token, timeout=int(HISTORY_TIMEOUT_S), retry_handlers=[]
    )
    plain = AsyncWebClient(token=ps.slack_bot_token, retry_handlers=[])
    auth = await client.auth_test()
    team, bot = auth["team_id"], auth["user_id"]
    port = SlackPort(client, team, bot, history_client=live, import_client=plain)
    token = ps.slack_bot_token

    adapter = SlackAdapter(assistant, port, lambda url: download(token, url))
    app = AsyncApp(client=client)

    @app.event("message")
    async def _message(event, context):
        await adapter.on_message(event, context.team_id)

    @app.event("app_mention")
    async def _mention(event, context):
        await adapter.on_message(event, context.team_id, mention=True)

    @app.event(LIFECYCLE_EVENTS)
    async def _lifecycle(event, context):
        await adapter.on_event(event, context.team_id)

    return AsyncSocketModeHandler(app, ps.slack_app_token), adapter, port


async def check(ps: PlatformSettings) -> dict:
    """Read-only token check for the smoke script: auth.test. No secrets in the result."""
    from slack_sdk.web.async_client import AsyncWebClient

    r = await AsyncWebClient(token=ps.slack_bot_token).auth_test()
    return {"team": r.get("team"), "team_id": r.get("team_id"), "bot_user_id": r.get("user_id")}
