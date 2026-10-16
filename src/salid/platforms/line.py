"""LINE adapter: webhook (signature-checked) -> `Incoming`, and Salid's replies back to LINE.

* `verify_signature`: X-Line-Signature, HMAC-SHA256 of the raw body, compared in constant time.
* `create_app`: a small FastAPI app. It answers 200 at once and handles events in the background
  (LINE retries slow webhooks).
* `LinePort`: reply token first (it expires quickly), push messages when the answer is slow or too
  long for one reply (5 messages per call, 5000 characters per text message).
* `LineApi`: the few Messaging API calls Salid needs, over `line-bot-sdk` (blocking, so every call
  runs in a worker thread). Tests replace it with a fake.

LINE does not tell a bot who is in a group (that needs a verified account), so memory scope is
always the room itself: a group's memory stays in that group, a 1:1 chat sees only itself.
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
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request

from ..qa import MAX_FILE_BYTES
from . import textfmt
from .config import PlatformSettings
from .core import Assistant, FileRef, HistLine, Incoming, RoomView, safe_exc

logger = logging.getLogger(__name__)

PLATFORM = "line"
TEXT_LIMIT = 4900  # UTF-16 code units per text message (LINE allows 5000)
REPLY_MAX = 5  # messages per reply or push call
MAX_MESSAGES = 8
MAX_BODY = 1_000_000
WEBHOOK_PATH = "/line/webhook"
LOADING_SECONDS = 30
NAME_TTL_S = 3600.0


def verify_signature(channel_secret: str, body: bytes, signature: str | None) -> bool:
    """True only for a correct X-Line-Signature (base64 HMAC-SHA256 of `body`)."""
    if not channel_secret or not signature:
        return False
    digest = hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest)
    return hmac.compare_digest(expected, signature.strip().encode("utf-8"))


class LineApi(Protocol):
    """The Messaging API calls Salid uses (blocking)."""

    def reply(self, token: str, texts: list[str]) -> list[str]: ...

    def push(self, to: str, texts: list[str]) -> list[str]: ...

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


class SdkLineApi:
    """`LineApi` over line-bot-sdk v3."""

    def __init__(self, access_token: str) -> None:
        from linebot.v3.messaging import ApiClient, Configuration, MessagingApi, MessagingApiBlob

        self._client = ApiClient(Configuration(access_token=access_token))
        self.api = MessagingApi(self._client)
        self.blob = MessagingApiBlob(self._client)

    @staticmethod
    def _messages(texts: list[str]):
        from linebot.v3.messaging import TextMessage

        return [TextMessage(text=x) for x in texts]

    def reply(self, token: str, texts: list[str]) -> list[str]:
        from linebot.v3.messaging import ReplyMessageRequest

        r = self.api.reply_message(
            ReplyMessageRequest(replyToken=token, messages=self._messages(texts))
        )
        return [m.id for m in (r.sent_messages or [])]

    def push(self, to: str, texts: list[str]) -> list[str]:
        from linebot.v3.messaging import PushMessageRequest

        # A retry key makes a retried push idempotent on LINE's side (no duplicate messages).
        r = self.api.push_message(
            PushMessageRequest(to=to, messages=self._messages(texts)),
            x_line_retry_key=str(uuid.uuid4()),
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
        return RoomView(None, [])  # LINE does not say who is in a group

    async def is_admin(self, msg: Incoming) -> bool:
        return False  # executives come from SALID_LINE_EXEC_USER_IDS only

    def format(self, text: str) -> list[str]:
        return textfmt.split_text(textfmt.to_line(text), TEXT_LIMIT, textfmt.utf16_len)

    async def ack(self, msg: Incoming, text: str) -> object | None:
        # A text "one moment" would burn the single-use reply token (or a push quota): only the
        # typing indicator, which LINE offers in 1:1 chats.
        if msg.is_direct:
            try:
                await asyncio.to_thread(self.api.loading, msg.room, LOADING_SECONDS)
            except Exception as e:
                logger.warning("LINE loading animation failed (%s)", safe_exc(e))
        return None

    async def drop_ack(self, ack: object) -> None:
        return None

    async def history(self, msg: Incoming, limit: int) -> list[HistLine]:
        return []  # the Messaging API cannot read chat history

    def _token_ok(self, msg: Incoming) -> bool:
        return (
            bool(msg.reply_token)
            and msg.reply_token not in self._tokens
            and self._clock() - msg.raw_at < self.ps.line_reply_token_seconds
        )

    async def deliver(self, msg: Incoming, texts: list[str], ack: object | None) -> list[str]:
        sent: list[str] = []
        rest = list(texts)
        if rest and self._token_ok(msg):
            self._tokens.add(msg.reply_token)
            if len(self._tokens) > 2000:
                self._tokens.clear()
            first, rest = rest[:REPLY_MAX], rest[REPLY_MAX:]
            try:
                sent += await asyncio.to_thread(self.api.reply, msg.reply_token, first)
            except Exception as e:
                if not reply_rejected(e):
                    # Timeout / network error: LINE may have delivered it. Pushing the same text
                    # again could duplicate the answer, so only the remaining batches go out.
                    logger.warning("LINE reply outcome unknown, not re-sending (%s)", safe_exc(e))
                else:  # LINE refused the token (expired / used): push instead
                    logger.warning("LINE reply refused, pushing (%s)", safe_exc(e))
                    rest = first + rest
        while rest:
            batch, rest = rest[:REPLY_MAX], rest[REPLY_MAX:]
            sent += await asyncio.to_thread(self.api.push, msg.room, batch)
        return sent


class LineAdapter:
    def __init__(
        self, assistant: Assistant, port: LinePort, bot_user_id: str = "", clock=time.monotonic
    ) -> None:
        self.a, self.port, self.bot_user_id, self._clock = assistant, port, bot_user_id, clock

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
        kind = event.get("type")
        if kind == "unsend":
            src = event.get("source") or {}
            room = src.get("groupId") or src.get("roomId") or src.get("userId")
            mid = (event.get("unsend") or {}).get("messageId")
            if room and mid:
                await self.a.on_delete(PLATFORM, tenant, room, mid)
            return
        if kind != "message":
            return
        msg = await self._incoming(event, tenant)
        if msg is not None:
            await self.a.handle(msg, self.port)

    async def _incoming(self, event: dict, tenant: str) -> Incoming | None:
        src = event.get("source") or {}
        stype = src.get("type")
        user = src.get("userId")
        if stype not in ("user", "group", "room") or not user:
            return None
        room = src.get("groupId") or src.get("roomId") or user
        m = event.get("message") or {}
        mtype, mid = m.get("type"), m.get("id")
        if not mid:
            return None
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
        ts = event.get("timestamp")
        at = (
            datetime.fromtimestamp(ts / 1000, UTC)
            if isinstance(ts, (int, float))
            else datetime.now(UTC)
        )
        direct = stype == "user"
        return Incoming(
            platform=PLATFORM,
            tenant=tenant,
            room=room,
            user=user,
            user_name=await self.port.display_name(stype, room, user),
            text=text,
            message=mid,
            at=at,
            is_direct=direct,
            mentions_bot=mentions_bot,
            mentions_others=mentions_others,
            files=files,
            room_name="" if direct else await self.port.group_name(stype, room),
            reply_token=event.get("replyToken") or "",
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
        elif x.get("type") == "user":
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
