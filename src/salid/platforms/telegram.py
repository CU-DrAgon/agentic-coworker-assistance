"""Telegram adapter: Bot API long polling -> `Incoming`, and Salid's replies back to Telegram.

* `TelegramApi`: the few Bot API calls Salid needs, over `httpx` (no extra dependency). The bot
  token is part of every API URL, so URLs are never logged: the httpx / httpcore loggers are
  pinned to WARNING and every failure is re-raised as `TelegramError` (method and status code
  only, never the URL or the response text).
* `Poller`: `deleteWebhook` once, then `getUpdates` with a long timeout. The offset only advances
  after an update was handed over, so a failed poll is retried (with backoff) without losing any.
* `TelegramAdapter`: one update -> `Incoming` (tenant = the bot itself, room = chat id, thread =
  forum topic id), `edited_message` -> `Assistant.on_edit`.
* `TelegramPort`: plain-text replies (4096 UTF-16 units per message), `reply_parameters` to the
  triggering message in groups, a typing indicator while Salid thinks.

Telegram does not tell a bot who is in a group (no member list for bots), so memory scope is the
room itself, like LINE: a group's memory stays in that group, a private chat sees only itself.
Telegram sends no event when a message is deleted, so deleted messages stay in memory until
retention removes them (`Assistant.on_delete` is never called from here).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from datetime import UTC, datetime
from typing import Protocol

import httpx

from ..qa import MAX_FILE_BYTES
from . import textfmt
from .core import Assistant, FileRef, HistLine, Incoming, RoomView, safe_exc

# The token is inside every request URL and httpx logs "HTTP Request: POST <url>" at INFO.
for _noisy in ("httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

PLATFORM = "telegram"
API_BASE = "https://api.telegram.org"
TEXT_LIMIT = 4096  # characters per message; counted in UTF-16 units to be safe
MAX_MESSAGES = 6
POLL_SECONDS = 30
ALLOWED_UPDATES = ["message", "edited_message"]
BACKOFF_START, BACKOFF_MAX = 1.0, 60.0
MAX_RETRY_AFTER = 30.0  # a 429 asking for a longer wait is not retried inline
DOWNLOAD_MAX = 20 * 1024 * 1024  # the Bot API serves files up to 20 MB
TYPING_EVERY = 4.0  # a typing indicator lasts 5 s
TYPING_MAX_S = 180.0


class TelegramError(Exception):
    """A failed Bot API call. Carries the method and status code only: no URL, no response text."""

    def __init__(self, method: str, code: int | None, retry_after: float | None = None):
        super().__init__(f"telegram {method} failed ({code})")
        self.method, self.code, self.retry_after = method, code, retry_after


class Api(Protocol):
    """The Bot API calls the adapter uses; tests replace it with a fake."""

    async def get_me(self) -> dict: ...

    async def delete_webhook(self) -> None: ...

    async def get_updates(self, offset: int | None, timeout: int) -> list[dict]: ...

    async def send_message(self, params: dict) -> dict: ...

    async def send_chat_action(self, chat_id: int, thread: int | None) -> None: ...

    def fetch_file(self, file_id: str) -> bytes:
        """getFile + download (blocking: called in a worker thread)."""
        ...


class TelegramApi:
    def __init__(
        self, token: str, *, transport: httpx.AsyncBaseTransport | None = None, sleep=None
    ):
        self._token = token
        self._transport = transport
        self._sleep = sleep or asyncio.sleep
        self._client = httpx.AsyncClient(timeout=15.0, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _url(self, method: str) -> str:
        return f"{API_BASE}/bot{self._token}/{method}"

    async def call(self, method: str, params: dict | None = None, *, timeout: float = 15.0):
        for attempt in (0, 1):
            try:
                r = await self._client.post(self._url(method), json=params or {}, timeout=timeout)
            except httpx.HTTPError:
                raise TelegramError(method, None) from None  # its text may quote the URL
            try:
                body = r.json()
            except ValueError:
                body = {}
            if not isinstance(body, dict):
                body = {}
            if r.status_code == 200 and body.get("ok"):
                return body.get("result")
            retry = (body.get("parameters") or {}).get("retry_after")
            err = TelegramError(
                method, body.get("error_code") or r.status_code, float(retry) if retry else None
            )
            if (
                err.code == 429
                and attempt == 0
                and err.retry_after is not None
                and err.retry_after <= MAX_RETRY_AFTER
            ):
                await self._sleep(err.retry_after)
                continue
            raise err
        raise AssertionError("unreachable")  # pragma: no cover

    async def get_me(self) -> dict:
        return await self.call("getMe")

    async def delete_webhook(self) -> None:
        await self.call("deleteWebhook", {"drop_pending_updates": False})

    async def get_updates(self, offset: int | None, timeout: int) -> list[dict]:
        params: dict = {"timeout": timeout, "allowed_updates": ALLOWED_UPDATES}
        if offset is not None:
            params["offset"] = offset
        return await self.call("getUpdates", params, timeout=timeout + 15.0) or []

    async def send_message(self, params: dict) -> dict:
        return await self.call("sendMessage", params)

    async def send_chat_action(self, chat_id: int, thread: int | None) -> None:
        params: dict = {"chat_id": chat_id, "action": "typing"}
        if thread is not None:
            params["message_thread_id"] = thread
        await self.call("sendChatAction", params)

    def fetch_file(self, file_id: str) -> bytes:
        limit = min(DOWNLOAD_MAX, MAX_FILE_BYTES)
        try:
            with httpx.Client(timeout=60.0, transport=self._transport) as c:
                r = c.post(self._url("getFile"), json={"file_id": file_id})
                body = r.json() if r.status_code == 200 else {}
                info = body.get("result") if body.get("ok") else None
                if not info or not info.get("file_path"):
                    raise TelegramError("getFile", r.status_code)
                if int(info.get("file_size") or 0) > limit:
                    raise ValueError("file over the size limit")
                url = f"{API_BASE}/file/bot{self._token}/{info['file_path']}"
                data = bytearray()
                with c.stream("GET", url) as resp:
                    if resp.status_code != 200:
                        raise TelegramError("download", resp.status_code)
                    for chunk in resp.iter_bytes():
                        data += chunk
                        if len(data) > limit:
                            raise ValueError("file over the size limit")
                return bytes(data)
        except httpx.HTTPError:
            raise TelegramError("download", None) from None


# ------------------------------------------------------------------ port (Salid -> Telegram)


class TelegramPort:
    platform = PLATFORM
    max_messages = MAX_MESSAGES

    def __init__(self, api: Api) -> None:
        self.api = api

    async def room_view(self, msg: Incoming) -> RoomView:
        return RoomView(None, [])  # a bot cannot list the members of a group

    async def is_admin(self, msg: Incoming) -> bool:
        return False  # executives come from SALID_TELEGRAM_EXEC_USER_IDS only

    def format(self, text: str) -> list[str]:
        return textfmt.split_text(textfmt.to_telegram(text), TEXT_LIMIT, textfmt.utf16_len)

    async def history(self, msg: Incoming, limit: int) -> list[HistLine]:
        return []  # the Bot API cannot read past messages

    @staticmethod
    def _thread(msg: Incoming) -> int | None:
        return int(msg.thread) if msg.thread else None

    async def typing(self, msg: Incoming) -> None:
        try:
            await self.api.send_chat_action(int(msg.room), self._thread(msg))
        except Exception as e:
            logger.warning("telegram typing indicator failed (%s)", safe_exc(e))

    async def _typing_loop(self, msg: Incoming) -> None:
        waited = 0.0
        while waited < TYPING_MAX_S:
            await self.typing(msg)
            await asyncio.sleep(TYPING_EVERY)
            waited += TYPING_EVERY

    async def ack(self, msg: Incoming, text: str) -> object | None:
        """No "one moment" text: keep the typing indicator alive until `deliver` stops it."""
        return asyncio.create_task(self._typing_loop(msg))

    async def drop_ack(self, ack: object) -> None:
        await self._stop(ack)

    @staticmethod
    async def _stop(ack: object | None) -> None:
        if isinstance(ack, asyncio.Task):
            ack.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await ack

    async def deliver(self, msg: Incoming, texts: list[str], ack: object | None) -> list[str]:
        await self._stop(ack)
        sent: list[str] = []
        for i, text in enumerate(texts):
            params: dict = {"chat_id": int(msg.room), "text": text}
            thread = self._thread(msg)
            if thread is not None:
                params["message_thread_id"] = thread
            if not msg.is_direct and i == 0:
                params["reply_parameters"] = {
                    "message_id": int(msg.message),
                    "allow_sending_without_reply": True,
                }
            result = await self.api.send_message(params)
            sent.append(str(result["message_id"]))
        return sent


# ------------------------------------------------------------------ adapter (Telegram -> Salid)


def _utf16_slice(text: str, offset: int, length: int) -> str:
    units = text.encode("utf-16-le")
    return units[offset * 2 : (offset + length) * 2].decode("utf-16-le", errors="ignore")


def strip_self_mentions(
    text: str, entities: list[dict], bot_id: str, username: str
) -> tuple[str, bool, bool]:
    """(text without Salid's own @mention, tagged Salid, tagged someone else). Entity offsets
    count UTF-16 code units, like every Telegram text offset."""
    units = text.encode("utf-16-le")
    cut: list[tuple[int, int]] = []
    me = others = False
    for e in entities:
        kind = e.get("type")
        off, length = int(e.get("offset", 0)), int(e.get("length", 0))
        if kind == "mention":
            tagged = _utf16_slice(text, off, length).lstrip("@")
            if username and tagged.lower() == username.lower():
                me = True
                cut.append((off, length))
            else:
                others = True
        elif kind == "text_mention":
            if str((e.get("user") or {}).get("id")) == bot_id:
                me = True
                cut.append((off, length))
            else:
                others = True
    for off, length in sorted(cut, reverse=True):
        units = units[: off * 2] + units[(off + length) * 2 :]
    plain = units.decode("utf-16-le", errors="ignore")
    return re.sub(r"[ \t]{2,}", " ", plain).strip(), me, others


class TelegramAdapter:
    def __init__(self, assistant: Assistant, port: TelegramPort, bot_id: str, username: str = ""):
        self.a, self.port, self.bot_id, self.username = assistant, port, bot_id, username

    async def on_update(self, update: dict) -> None:
        """One update. Never raises."""
        try:
            await self._update(update)
        except Exception as e:
            logger.error("telegram update failed (%s)", safe_exc(e))

    async def _update(self, update: dict) -> None:
        edited = "edited_message" in update
        raw = update.get("edited_message") if edited else update.get("message")
        msg = self._incoming(raw or {})
        if msg is None:
            return
        if edited:
            await self.a.on_edit(msg)
            return
        if msg.is_direct:  # always addressed: show "typing" at once
            self.a.spawn(self.port.typing(msg))
        await self.a.handle(msg, self.port)

    def _incoming(self, m: dict) -> Incoming | None:
        sender, chat = m.get("from") or {}, m.get("chat") or {}
        uid, mid = sender.get("id"), m.get("message_id")
        ctype = chat.get("type")
        if uid is None or mid is None or chat.get("id") is None:
            return None
        # bots (including this one and the anonymous-admin bot) and posts made as a channel
        if sender.get("is_bot") or str(uid) == self.bot_id or m.get("sender_chat"):
            return None
        if ctype not in ("private", "group", "supergroup"):
            return None
        direct = ctype == "private"
        text = m.get("text") or m.get("caption") or ""
        entities = (m.get("entities") if m.get("text") else m.get("caption_entities")) or []
        plain, mentions_bot, mentions_others = strip_self_mentions(
            text, entities, self.bot_id, self.username
        )
        if direct and re.fullmatch(
            r"/start(@\w+)?(\s.*)?", plain
        ):  # the first thing everyone sends
            plain = ""
        replied = m.get("reply_to_message") or {}
        replies_to_bot = (
            str((replied.get("from") or {}).get("id")) == self.bot_id
            and "forum_topic_created" not in replied
        )
        files = self._files(m, str(mid))
        if not plain and not files and not (mentions_bot or direct):
            return None
        name = " ".join(x for x in (sender.get("first_name"), sender.get("last_name")) if x)
        return Incoming(
            platform=PLATFORM,
            tenant=self.bot_id,
            room=str(chat["id"]),
            user=str(uid),
            user_name=name or sender.get("username") or str(uid),
            text=plain,
            message=str(mid),
            at=datetime.fromtimestamp(m["date"], UTC) if m.get("date") else datetime.now(UTC),
            is_direct=direct,
            mentions_bot=mentions_bot or replies_to_bot,  # core only honours replies in threads
            mentions_others=mentions_others,
            thread=str(m["message_thread_id"])
            if not direct and m.get("is_topic_message") and m.get("message_thread_id")
            else None,
            replies_to_bot=replies_to_bot,
            files=files,
            room_name="" if direct else (chat.get("title") or ""),
        )

    def _files(self, m: dict, mid: str) -> list[FileRef]:
        api = self.port.api
        out: list[FileRef] = []
        photos = m.get("photo") or []
        if photos:
            best = max(
                photos,
                key=lambda p: (
                    p.get("file_size") or 0,
                    (p.get("width") or 0) * (p.get("height") or 0),
                ),
            )
            out.append(
                FileRef(
                    f"photo-{mid}.jpg",
                    "image/jpeg",
                    int(best.get("file_size") or 0),
                    lambda fid=best["file_id"]: api.fetch_file(fid),
                )
            )
        doc = m.get("document")
        if doc and doc.get("file_id"):
            out.append(
                FileRef(
                    doc.get("file_name") or f"file-{mid}",
                    doc.get("mime_type"),
                    int(doc.get("file_size") or 0),
                    lambda fid=doc["file_id"]: api.fetch_file(fid),
                )
            )
        return out


# ------------------------------------------------------------------ long polling


class Poller:
    def __init__(
        self,
        api: Api,
        adapter: TelegramAdapter,
        *,
        sleep=asyncio.sleep,
        timeout: int = POLL_SECONDS,
    ) -> None:
        self.api, self.adapter, self._sleep, self.timeout = api, adapter, sleep, timeout
        self.offset: int | None = None
        self.backoff = BACKOFF_START
        self._webhook_cleared = False
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    async def poll_once(self) -> int:
        """One getUpdates round; returns how many updates were handed to Salid. Raises on an API
        error (the offset is unchanged, so Telegram re-delivers those updates)."""
        if not self._webhook_cleared:  # a webhook and getUpdates exclude each other
            await self.api.delete_webhook()
            self._webhook_cleared = True
        updates = await self.api.get_updates(self.offset, self.timeout)
        for u in updates:
            self.adapter.a.spawn(self.adapter.on_update(u))
            self.offset = max(self.offset or 0, int(u["update_id"]) + 1)
        return len(updates)

    async def run(self) -> None:
        while not self._stop:
            try:
                await self.poll_once()
                self.backoff = BACKOFF_START
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    "telegram poll failed, retrying in %.0fs (%s)", self.backoff, safe_exc(e)
                )
                await self._sleep(self.backoff)
                self.backoff = min(self.backoff * 2, BACKOFF_MAX)


async def check(ps) -> dict:
    """Read-only token check for the smoke script: getMe. Only the username and id are returned."""
    api = TelegramApi(ps.telegram_bot_token)
    try:
        me = await api.get_me()
    finally:
        await api.aclose()
    return {"username": me.get("username"), "id": me.get("id")}
