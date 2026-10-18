"""LINE adapter: webhook (signature-checked) -> `Incoming`, and Salid's replies back to LINE.

* `verify_signature`: X-Line-Signature, HMAC-SHA256 of the raw body, compared in constant time.
* `create_app`: a small FastAPI app. It answers 200 at once and handles events in the background
  (LINE retries slow webhooks).
* `LinePort`: reply token first (it lasts about a minute and a reply is free), push messages only
  when the token is gone or the answer needs more than the 5 message objects of one reply
  (5000 UTF-16 units per text message). A push counts once per recipient against the monthly
  quota, so chunks are merged to fill every message object.
* `LineApi`: the few Messaging API calls Salid needs, over `line-bot-sdk` (blocking, so every call
  runs in a worker thread). Tests replace it with a fake.

LINE does not give a bot the member list of a group without a verified account, so memory scope
is always the room itself: a group's memory stays in that group, a 1:1 chat sees only itself.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request

from ..naming import mentions_name
from ..qa import MAX_FILE_BYTES
from . import ids, textfmt
from .config import PlatformSettings
from .core import Assistant, FileRef, HistLine, Incoming, RoomView, safe_exc

logger = logging.getLogger(__name__)

PLATFORM = "line"
TEXT_LIMIT = 4900  # UTF-16 code units per text message when splitting (room for a note)
MERGE_LIMIT = 5000  # what LINE allows per text message: chunks are merged up to this
REPLY_MAX = 5  # message objects per reply or push call
MAX_MESSAGES = 5  # one reply (free) or one push (counted once per recipient) holds the answer
MAX_BODY = 1_000_000
WEBHOOK_PATH = "/line/webhook"
LOADING_SECONDS = 30
NAME_TTL_S = 3600.0
REPLY_TOKEN_SECONDS = 50.0  # the token is valid ~1 minute; never give up on it sooner than this
PUSH_RETRY_DELAY_S = 1.0  # one retry of an ambiguous push failure, with the same retry key
SEEN_EVENTS = 5000  # webhookEventIds remembered for de-duplication
SEEN_EDITS = 5000
FILE_CACHE_S = 600.0  # a quoted image / file is still found for 10 minutes
FILE_CACHE_PER_ROOM = 20
FILE_CACHE_ROOMS = 500
QUOTES_KEEP = 1000
REDELIVERY_TOKEN_MAX_AGE_S = 1200.0  # LINE: a redelivered token is dead 20 minutes after the event
ANON_NAME = "ผู้ใช้ LINE (ไม่ระบุตัวตน)"
LINE_RETENTION_DAYS = 7  # team memory kept this long in LINE rooms unless an admin chose otherwise


def verify_signature(channel_secret: str, body: bytes, signature: str | None) -> bool:
    """True only for a correct X-Line-Signature (base64 HMAC-SHA256 of `body`)."""
    if not channel_secret or not signature:
        return False
    digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest)
    return hmac.compare_digest(expected, signature.strip().encode("utf-8"))


class LineApi(Protocol):
    """The Messaging API calls Salid uses (blocking)."""

    def reply(self, token: str, texts: list[str], quote_token: str | None = None) -> list[str]: ...

    def push(
        self,
        to: str,
        texts: list[str],
        retry_key: str | None = None,
        quote_token: str | None = None,
    ) -> list[str]: ...

    def loading(self, chat_id: str, seconds: int) -> None: ...

    def profile(self, kind: str, room: str, user: str) -> str: ...

    def group_name(self, kind: str, room: str) -> str: ...

    def content(self, message_id: str) -> bytes: ...

    def bot_info(self) -> dict: ...


def reply_rejected(e: Exception) -> bool:
    """True when LINE definitely did not deliver a reply: an HTTP 4xx answer (invalid or expired
    reply token, bad request). Anything else (timeouts, 5xx, connection errors) is ambiguous."""
    status = getattr(e, "status", None)
    return isinstance(status, int) and 400 <= status < 500


def _status(e: Exception) -> int | None:
    status = getattr(e, "status", None)
    return status if isinstance(status, int) else None


def retry_key(msg: Incoming, index: int, texts: list[str]) -> str:
    """A stable x-line-retry-key (a UUID) for one logical push: the same answer chunk of the same
    message always gets the same key, so a retried push is accepted by LINE only once (a second
    attempt answers 409). Different content gets a different key."""
    digest = hashlib.sha1("\x1f".join(texts).encode("utf-8")).hexdigest()
    name = f"salid-line:{msg.tenant}:{msg.room}:{msg.message}:{index}:{digest}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def pack_texts(texts: list[str], limit: int = MERGE_LIMIT) -> list[str]:
    """Merge neighbouring chunks into as few text messages as fit `limit` UTF-16 units each, so
    the whole answer needs as few of the 5 message objects of a reply as possible."""
    out: list[str] = []
    for t in texts:
        if not t:
            continue
        if out and textfmt.utf16_len(out[-1]) + 2 + textfmt.utf16_len(t) <= limit:
            out[-1] = f"{out[-1]}\n\n{t}"
        else:
            out.append(t)
    return out


class SdkLineApi:
    """`LineApi` over line-bot-sdk v3."""

    def __init__(self, access_token: str) -> None:
        from linebot.v3.messaging import ApiClient, Configuration, MessagingApi, MessagingApiBlob

        self._client = ApiClient(Configuration(access_token=access_token))
        self.api = MessagingApi(self._client)
        self.blob = MessagingApiBlob(self._client)

    @staticmethod
    def _messages(texts: list[str], quote_token: str | None = None):
        from linebot.v3.messaging import TextMessage

        return [
            TextMessage(text=x, quoteToken=quote_token if i == 0 and quote_token else None)
            for i, x in enumerate(texts)
        ]

    def reply(self, token: str, texts: list[str], quote_token: str | None = None) -> list[str]:
        from linebot.v3.messaging import ReplyMessageRequest

        r = self.api.reply_message(
            ReplyMessageRequest(replyToken=token, messages=self._messages(texts, quote_token))
        )
        return [m.id for m in (r.sent_messages or [])]

    def push(
        self,
        to: str,
        texts: list[str],
        retry_key: str | None = None,
        quote_token: str | None = None,
    ) -> list[str]:
        from linebot.v3.messaging import PushMessageRequest

        # The retry key makes a retried push idempotent on LINE's side (no duplicate messages).
        r = self.api.push_message(
            PushMessageRequest(to=to, messages=self._messages(texts, quote_token)),
            x_line_retry_key=retry_key or str(uuid.uuid4()),
        )
        return [m.id for m in (r.sent_messages or [])]

    def loading(self, chat_id: str, seconds: int) -> None:
        from linebot.v3.messaging import ShowLoadingAnimationRequest

        self.api.show_loading_animation(
            ShowLoadingAnimationRequest(chatId=chat_id, loadingSeconds=seconds)
        )

    def profile(self, kind: str, room: str, user: str) -> str:
        if kind == "group":
            return self.api.get_group_member_profile(room, user).display_name
        if kind == "room":
            return self.api.get_room_member_profile(room, user).display_name
        return self.api.get_profile(user).display_name

    def group_name(self, kind: str, room: str) -> str:
        return self.api.get_group_summary(room).group_name if kind == "group" else ""

    def content(self, message_id: str) -> bytes:
        data = bytes(self.blob.get_message_content(message_id))
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("file over the size limit")
        return data

    def bot_info(self) -> dict:
        r = self.api.get_bot_info()
        return {"user_id": r.user_id, "display_name": r.display_name, "basic_id": r.basic_id}


class LinePort:
    platform = PLATFORM
    max_messages = MAX_MESSAGES

    def __init__(self, api: LineApi, ps: PlatformSettings, clock=time.monotonic) -> None:
        self.api, self.ps, self._clock = api, ps, clock
        self._names: dict[tuple, tuple[float, str]] = {}
        self._tokens: set[str] = set()
        # (room, message id) -> quote token of the message Salid is answering (groups)
        self.quotes: OrderedDict[tuple[str, str], str] = OrderedDict()

    def remember_quote(self, room: str, message: str, token: str) -> None:
        self.quotes[(room, message)] = token
        while len(self.quotes) > QUOTES_KEEP:
            self.quotes.popitem(last=False)

    async def display_name(self, kind: str, room: str, user: str) -> str:
        key = (kind, room, user)
        hit = self._names.get(key)
        if hit is not None and self._clock() - hit[0] < NAME_TTL_S:
            return hit[1]
        try:
            name = await asyncio.to_thread(self.api.profile, kind, room, user)
        except Exception as e:
            logger.warning("LINE profile lookup failed (%s)", safe_exc(e))
            name = ""
        self._names[key] = (self._clock(), name)
        return name

    async def group_name(self, kind: str, room: str) -> str:
        key = ("name", kind, room)
        hit = self._names.get(key)
        if hit is not None and self._clock() - hit[0] < NAME_TTL_S:
            return hit[1]
        try:
            name = await asyncio.to_thread(self.api.group_name, kind, room)
        except Exception as e:
            logger.warning("LINE group summary failed (%s)", safe_exc(e))
            name = ""
        self._names[key] = (self._clock(), name)
        return name

    # ------------------------------------------------------------ core.Port
    async def room_view(self, msg: Incoming) -> RoomView:
        return RoomView(None, [])  # memory never crosses rooms on LINE

    async def is_admin(self, msg: Incoming) -> bool:
        return False  # executives come from SALID_LINE_EXEC_USER_IDS only

    def format(self, text: str) -> list[str]:
        return textfmt.split_text(textfmt.to_line(text), TEXT_LIMIT, textfmt.utf16_len)

    async def start_loading(self, msg: Incoming) -> None:
        """LINE's typing indicator: 1:1 chats only (a group id is a 400). Free, no quota."""
        if not msg.is_direct:
            return
        try:
            await asyncio.to_thread(self.api.loading, msg.room, LOADING_SECONDS)
        except Exception as e:
            logger.warning("LINE loading animation failed (%s)", safe_exc(e))

    async def ack(self, msg: Incoming, text: str) -> object | None:
        # A text "one moment" would burn the single-use reply token (or a push quota): only the
        # typing indicator (started at once; this renews it for a long answer).
        await self.start_loading(msg)
        return None

    async def drop_ack(self, ack: object) -> None:
        return None

    async def history(self, msg: Incoming, limit: int) -> list[HistLine]:
        return []  # the Messaging API cannot read chat history

    def token_window(self) -> float:
        return max(float(self.ps.line_reply_token_seconds), REPLY_TOKEN_SECONDS)

    def _token_ok(self, msg: Incoming) -> bool:
        return (
            bool(msg.reply_token)
            and msg.reply_token not in self._tokens
            and self._clock() - msg.raw_at < self.token_window()
        )

    def _use_token(self, token: str) -> None:
        self._tokens.add(token)
        if len(self._tokens) > 2000:
            self._tokens.clear()

    async def notice(self, token: str, text: str) -> None:
        """A one-off message through a join / follow reply token (free). Never pushed."""
        if not token or token in self._tokens:
            return
        self._use_token(token)
        try:
            await asyncio.to_thread(self.api.reply, token, [text], None)
        except Exception as e:
            logger.warning("LINE notice not sent (%s)", safe_exc(e))

    async def _push(self, msg: Incoming, index: int, batch: list[str], quote: str | None):
        """One push request, retried once on an ambiguous failure with the same retry key.
        Returns the sent ids; None when LINE refused for good (logged)."""
        key = retry_key(msg, index, batch)
        for attempt in (1, 2):
            try:
                return await asyncio.to_thread(self.api.push, msg.room, batch, key, quote)
            except Exception as e:
                status = _status(e)
                if status == 409:  # same retry key: LINE already accepted this request
                    logger.info("LINE push already accepted (409), not re-sending")
                    return []
                if status == 429:
                    logger.error(
                        "LINE push refused (429): the monthly message quota is used up or the "
                        "push rate limit was hit; this part of the answer was NOT delivered "
                        "(tenant=%s room=%s). Replies stay free: check the plan and "
                        "GET /v2/bot/message/quota/consumption.",
                        ids.tenant_id("line", msg.tenant),
                        ids.room_id("line", msg.tenant, msg.room),
                    )
                    return None
                if reply_rejected(e) or attempt == 2:
                    raise
                logger.warning("LINE push failed, retrying once (%s)", safe_exc(e))
                await asyncio.sleep(PUSH_RETRY_DELAY_S)
        return None  # unreachable

    async def deliver(self, msg: Incoming, texts: list[str], ack: object | None) -> list[str]:
        sent: list[str] = []
        rest = pack_texts(texts)
        quote = None if msg.is_direct else self.quotes.get((msg.room, msg.message))
        if rest and self._token_ok(msg):
            self._use_token(msg.reply_token)
            first, rest = rest[:REPLY_MAX], rest[REPLY_MAX:]
            try:
                sent += await asyncio.to_thread(self.api.reply, msg.reply_token, first, quote)
                quote = None
            except Exception as e:
                quote = None if not reply_rejected(e) else quote
                if _status(e) == 429:
                    # Rate limited, not an expired token: a (paid) push would hit the same wall.
                    logger.error("LINE reply rate limited (429), answer not delivered")
                    return sent
                if not reply_rejected(e):
                    # Timeout / network error: LINE may have delivered it. Pushing the same text
                    # again could duplicate the answer, so only the remaining batches go out.
                    logger.warning("LINE reply outcome unknown, not re-sending (%s)", safe_exc(e))
                else:  # LINE refused the token (expired / used): push instead
                    logger.warning("LINE reply refused, pushing (%s)", safe_exc(e))
                    rest = first + rest
        index = 0
        while rest:
            batch, rest = rest[:REPLY_MAX], rest[REPLY_MAX:]
            sent_ids = await self._push(msg, index, batch, quote)
            if sent_ids is None:  # refused (429): the remaining batches would be refused as well
                break
            sent += sent_ids
            quote = None
            index += len(batch)
        return sent


# ------------------------------------------------------------------ notices

NOTICE_TH = 'สวัสดีค่ะ สลิดเองนะคะ ผู้ช่วยของทีม ยินดีที่ได้ช่วยค่ะ\nเรียกสลิดได้ด้วยการแท็ก @ หรือพิมพ์ชื่อ "สลิด"'
NOTICE_EN = (
    'Hi, this is Salid, your team assistant.\nCall me by tagging me with @ or by writing "Salid".'
)
MEMORY_TH = (
    "\n\nในแชทนี้สลิดจะจำข้อความไว้เป็นความจำของทีม เพื่อช่วยตอบเรื่องที่เคยคุยกัน "
    "เก็บไว้ {days} วัน และใช้เฉพาะในแชทนี้เท่านั้น\n"
    "• ให้ลืมข้อความหนึ่ง: ยกเลิกการส่งข้อความนั้น สลิดจะลบออกจากความจำ\n"
    "• ให้หยุดจำ: {stop_th} ข้อความที่จำไว้แล้วจะหมดอายุเองตามกำหนด"
)
MEMORY_EN = (
    "\n\nIn this chat I remember messages as team memory, to help answer things you discussed "
    "before. They are kept for {days} days and used only in this chat.\n"
    "• To make me forget a message: unsend it and I delete it from memory.\n"
    "• To stop me remembering: {stop_en} What is already stored expires on schedule."
)
STOP_GROUP = ("เชิญสลิดออกจากกลุ่ม", "remove me from the group.")
STOP_DIRECT = ("บล็อกหรือลบแชทกับสลิด", "block or delete this chat.")


def notice_text(days: int | None, direct: bool) -> str:
    """The join / follow message (Thai, then English; no links). `days` None: memory is off."""
    th, en = NOTICE_TH, NOTICE_EN
    if days is not None:
        stop_th, stop_en = STOP_DIRECT if direct else STOP_GROUP
        th += MEMORY_TH.format(days=days, stop_th=stop_th)
        en += MEMORY_EN.format(days=days, stop_en=stop_en)
    return f"{th}\n\n— — —\n\n{en}"


def _is_file(m: dict) -> bool:
    return m.get("type") in ("image", "file")


class LineAdapter:
    def __init__(
        self, assistant: Assistant, port: LinePort, bot_user_id: str = "", clock=time.monotonic
    ) -> None:
        self.a, self.port, self.bot_user_id, self._clock = assistant, port, bot_user_id, clock
        self._events: OrderedDict[str, None] = OrderedDict()  # webhookEventIds handled
        self._edit_ts: OrderedDict[tuple[str, str], float] = OrderedDict()
        # room -> {message id: (arrival time, FileRef)} of recent images / files
        self._files: OrderedDict[str, OrderedDict[str, tuple[float, FileRef]]] = OrderedDict()
        self._retention_set: set[str] = set()

    # ------------------------------------------------------------ de-duplication
    def _first_delivery(self, event: dict) -> bool:
        """False for an event already handled (a LINE redelivery or a network duplicate)."""
        eid = event.get("webhookEventId")
        if not eid:
            return True
        redelivery = bool((event.get("deliveryContext") or {}).get("isRedelivery"))
        if eid in self._events:
            self._events.move_to_end(eid)
            logger.info("LINE event %s skipped: already handled (redelivery=%s)", eid, redelivery)
            return False
        self._events[eid] = None
        while len(self._events) > SEEN_EVENTS:
            self._events.popitem(last=False)
        if redelivery:
            logger.info("LINE event %s is a redelivery we had not seen: handling it", eid)
        return True

    async def on_event(self, event: dict, destination: str = "") -> None:
        """One webhook event. Never raises."""
        try:
            await self._event(event, destination)
        except Exception as e:
            logger.error("LINE event failed (%s)", safe_exc(e))

    async def _event(self, event: dict, destination: str) -> None:
        tenant = destination or self.bot_user_id
        if not tenant or (self.bot_user_id and destination and destination != self.bot_user_id):
            return
        if event.get("mode") == "standby":
            return
        if not self._first_delivery(event):
            return
        if (event.get("deliveryContext") or {}).get("isRedelivery"):
            ts = event.get("timestamp")
            if isinstance(ts, (int, float)):
                age = datetime.now(UTC).timestamp() - ts / 1000
                if age > REDELIVERY_TOKEN_MAX_AGE_S:
                    event = {**event, "replyToken": ""}  # dead token: do not try it
        kind = event.get("type")
        if kind == "unsend":
            src = event.get("source") or {}
            room = src.get("groupId") or src.get("roomId") or src.get("userId")
            mid = (event.get("unsend") or {}).get("messageId")
            if room and mid:
                await self.a.on_delete(PLATFORM, tenant, room, mid)
            return
        if kind in ("join", "follow"):
            await self._welcome(event, tenant, direct=kind == "follow")
            return
        if kind == "messageEdited":
            await self._edited(event, tenant)
            return
        if kind != "message":
            return
        msg = await self._incoming(event, tenant)
        if msg is None:
            return
        if msg.is_direct:
            await self.port.start_loading(msg)  # at once: free, and the answer can take a while
        await self.a.handle(msg, self.port)

    # ------------------------------------------------------------ join / follow
    def _memory_days(self, gid: int, tenant: str) -> int | None:
        """Retention shown in the notice (None: team memory is off for this tenant). Blocking.
        A tenant still on the global default gets LINE's shorter default, once per run."""
        team = self.a.team
        if not team.enabled(gid, "memory"):
            return None
        ns = SimpleNamespace(platform=PLATFORM, tenant=tenant)
        if tenant not in self._retention_set and self.a._owner_ok(ns, gid):
            self._retention_set.add(tenant)
            from ..guild_settings import default_retention_days, effective_retention_days

            try:
                # The STORED value (0 = keep forever, the global default), not the effective
                # window. A hub save (updated_at) makes even the default a company choice.
                g, default = team.current(gid), default_retention_days()
                if g.memory_retention_days == default and not g.updated_at:
                    team.set_memory_retention_days(
                        gid, min(LINE_RETENTION_DAYS, effective_retention_days(default))
                    )
            except Exception as e:
                logger.warning("could not set LINE memory retention (%s)", safe_exc(e))
        return team.memory_retention_days(gid)

    def _welcome_info(
        self, gid: int, rid: int, tenant: str, direct: bool
    ) -> tuple[bool, int | None]:
        """(room is switched off, retention to tell). Blocking."""
        self.a.team.current(gid)
        if self.a._mode(gid, rid, "", direct) == "off":
            return True, None
        return False, self._memory_days(gid, tenant)

    async def _welcome(self, event: dict, tenant: str, *, direct: bool) -> None:
        """Tell the room (reply token, free) what Salid remembers and how to stop it."""
        src = event.get("source") or {}
        room = src.get("groupId") or src.get("roomId") or src.get("userId")
        token = event.get("replyToken") or ""
        if not room or not token:
            return
        gid = ids.tenant_id(PLATFORM, tenant)
        rid = ids.room_id(PLATFORM, tenant, room)
        try:
            off, days = await asyncio.to_thread(self._welcome_info, gid, rid, tenant, direct)
        except Exception as e:
            logger.warning("LINE notice settings unavailable (%s)", safe_exc(e))
            off, days = False, LINE_RETENTION_DAYS
        if off:
            return
        await self.port.notice(token, notice_text(days, direct))

    # ------------------------------------------------------------ edits
    async def _edited(self, event: dict, tenant: str) -> None:
        """messageEdited (groups): keep memory current. Edits can arrive out of order: only the
        newest timestamp wins. Salid does not answer an edit."""
        msg = await self._incoming(event, tenant, edit=True)
        if msg is None:
            return
        ts = event.get("timestamp")
        if isinstance(ts, (int, float)):
            key = (msg.room, msg.message)
            if ts <= self._edit_ts.get(key, 0):
                return
            self._edit_ts[key] = ts
            while len(self._edit_ts) > SEEN_EDITS:
                self._edit_ts.popitem(last=False)
        await self.a.on_edit(msg)

    # ------------------------------------------------------------ incoming
    def _cache_file(self, room: str, mid: str, ref: FileRef) -> None:
        now = self._clock()
        files = self._files.setdefault(room, OrderedDict())
        self._files.move_to_end(room)
        files[mid] = (now, ref)
        for k in [k for k, (t, _r) in files.items() if now - t >= FILE_CACHE_S]:
            del files[k]
        while len(files) > FILE_CACHE_PER_ROOM:
            files.popitem(last=False)
        while len(self._files) > FILE_CACHE_ROOMS:
            self._files.popitem(last=False)

    def _cached_file(self, room: str, mid: str) -> FileRef | None:
        hit = (self._files.get(room) or {}).get(mid)
        if hit is None or self._clock() - hit[0] >= FILE_CACHE_S:
            return None
        return hit[1]

    async def _incoming(self, event: dict, tenant: str, *, edit: bool = False) -> Incoming | None:
        src = event.get("source") or {}
        stype = src.get("type")
        if stype not in ("user", "group", "room"):
            return None
        user = src.get("userId")
        direct = stype == "user"
        room = src.get("groupId") or src.get("roomId") or user
        m = event.get("message") or {}
        mtype, mid = m.get("type"), m.get("id")
        if not mid:
            return None
        anonymous = not user
        if anonymous:
            if direct or not room:
                logger.warning("LINE message without a userId ignored (no room to answer in)")
                return None
            # LINE for PC users have no userId in groups. Remembered (and answerable) as an
            # anonymous non-executive: a fresh identity per message, so it can never see another
            # person's receipts.
            user = f"anon:{room}:{mid}"
            logger.info("LINE group message without a userId handled as anonymous")
        text, mentions_bot, mentions_others, files = "", False, False, []
        if mtype == "text":
            text, mentions_bot, mentions_others = _strip_self_mentions(
                m.get("text") or "", (m.get("mention") or {}).get("mentionees") or []
            )
        elif mtype == "image":
            files = [
                FileRef(f"image-{mid}.jpg", "image/jpeg", 0, lambda i=mid: self.port.api.content(i))
            ]
        elif mtype == "file":
            name = m.get("fileName") or f"file-{mid}"
            files = [
                FileRef(
                    name, None, int(m.get("fileSize") or 0), lambda i=mid: self.port.api.content(i)
                )
            ]
        else:
            return None
        if not edit:
            if files:
                self._cache_file(room, mid, files[0])
            if not direct and m.get("quoteToken") and mtype in ("text", "image"):
                self.port.remember_quote(room, mid, m["quoteToken"])
            quoted = m.get("quotedMessageId")
            if mtype == "text" and quoted and (direct or mentions_bot or mentions_name(text)):
                # "@Salid what is this?" quoting an image: the text applies to that file
                ref = self._cached_file(room, quoted)
                if ref is not None:
                    files = [ref]
        ts = event.get("timestamp")
        at = (
            datetime.fromtimestamp(ts / 1000, UTC)
            if isinstance(ts, (int, float))
            else datetime.now(UTC)
        )
        if anonymous:
            name = ANON_NAME
        else:
            name = await self.port.display_name(stype, room, user)
        return Incoming(
            platform=PLATFORM,
            tenant=tenant,
            room=room,
            user=user,
            user_name=name,
            text=text,
            message=mid,
            at=at,
            is_direct=direct,
            mentions_bot=mentions_bot,
            mentions_others=mentions_others,
            files=files,
            room_name="" if direct else await self.port.group_name(stype, room),
            reply_token="" if edit else (event.get("replyToken") or ""),
            raw_at=self._clock(),
        )


def _strip_self_mentions(text: str, mentionees: list[dict]) -> tuple[str, bool, bool]:
    """(text without Salid's own @mention, tagged Salid, tagged someone else). Mentionee indexes
    count UTF-16 code units, like every LINE text offset."""
    units = text.encode("utf-16-le")
    cut, me, others = [], False, False
    for x in mentionees:
        if x.get("isSelf"):
            me = True
            cut.append((int(x.get("index", 0)), int(x.get("length", 0))))
        elif x.get("type") in ("user", "all"):  # someone else, or @All: not meant for Salid alone
            others = True
    for index, length in sorted(cut, reverse=True):
        units = units[: index * 2] + units[(index + length) * 2 :]
    plain = units.decode("utf-16-le", errors="ignore")
    return " ".join(plain.split()), me, others


def create_app(channel_secret: str, adapter: LineAdapter, spawn: Callable | None = None):
    """The webhook app. `spawn(coro)` schedules background handling (default: asyncio task)."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    tasks: set[asyncio.Task] = set()

    def run(coro) -> None:
        if spawn is not None:
            spawn(coro)
            return
        task = asyncio.create_task(coro)
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    @app.post(WEBHOOK_PATH)
    async def webhook(request: Request):
        signature = request.headers.get("x-line-signature")
        if not signature:
            raise HTTPException(status_code=401)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY:
                raise HTTPException(status_code=413)
        raw = bytes(body)
        if not verify_signature(channel_secret, raw, signature):
            raise HTTPException(status_code=401)
        try:
            data = json.loads(raw)
        except ValueError:
            raise HTTPException(status_code=400) from None
        for ev in data.get("events") or []:
            run(adapter.on_event(ev, data.get("destination") or ""))
        return {}

    @app.get("/health")
    async def health():
        return {"ok": True}

    app.state.tasks = tasks
    return app


async def check(ps: PlatformSettings) -> dict:
    """Read-only token check for the smoke script: get bot info. No secrets in the result."""
    api = SdkLineApi(ps.line_channel_access_token)
    return await asyncio.to_thread(api.bot_info)
