"""Telegram adapter: Bot API long polling -> `Incoming`, and Salid's replies back to Telegram.

* `TelegramApi`: the few Bot API calls Salid needs, over `httpx` (no extra dependency). The bot
  token is part of every API URL, so URLs are never logged: the httpx / httpcore loggers are
  pinned to WARNING and every failure is re-raised as `TelegramError` (method and status code
  only, never the URL or the response text).
* `Poller`: `deleteWebhook` once, then `getUpdates` with a long timeout. A batch is handled
  concurrently (album grouping needs that) and awaited; the offset only advances past updates
  whose handler finished without a transient error, so a failed poll or handler is retried
  (with backoff) without losing any. Handlers are idempotent per `update_id`.
* `TelegramAdapter`: one update -> `Incoming` (tenant = the bot itself, room = chat id, thread =
  forum topic id, also in private chats with topics), `edited_message` -> `Assistant.on_edit`.
  `/cmd@thisbot` commands address the bot, `/help` is answered without the model, an album
  (`media_group_id`) becomes one `Incoming`, `my_chat_member` greets / forgets a chat.
* `TelegramPort`: plain-text replies (4096 UTF-16 units per message), `reply_parameters` to the
  triggering message in groups, a typing indicator while Salid thinks, and paced sending (>= 1 s
  between messages to one chat, <= 20 a minute to a group, 429 `retry_after` honoured).

A bot has no call that lists all members of a group (only admins and the member count), so memory
scope is the room itself, like LINE: a group's memory stays in that group, a private chat sees
only itself.
Telegram sends no event when a message is deleted, so deleted messages stay in memory until
retention removes them (`Assistant.on_delete` is never called from here).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import logging
import re
import time
from collections import OrderedDict, deque
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
ALLOWED_UPDATES = ["message", "edited_message", "my_chat_member"]
BACKOFF_START, BACKOFF_MAX = 1.0, 60.0
CONFLICT_BACKOFF_MIN = 5.0  # a 409 means another consumer: do not hammer it
IDLE_RESET_S = 6 * 86400  # after ~a week without updates Telegram may renumber update_id
MAX_RETRY_AFTER = 120.0  # a 429 asking for a longer wait is surfaced, not slept through
MAX_RATE_RETRIES = 3
SEND_GAP_S = 1.0  # at most about one message a second to one chat
GROUP_PER_MINUTE = 20  # and 20 a minute to one group
ALBUM_DEBOUNCE_S = 1.5  # wait for the other photos of an album
ALBUM_MAX_WAIT_S = 10.0
BATCH_TIMEOUT_S = 180.0  # one stuck handler must not block polling for ever
MAX_HANDLER_RETRIES = 3  # a transient handler failure is redelivered this many times at most
DONE_KEEP = 2000  # update_ids remembered as handled (redeliveries are skipped)
MAX_LOCKS = 2000  # per-chat send locks kept (idle ones are evicted)
MAX_WINDOWS = 2000  # per-group rate windows kept
MAX_GONE = 10_000  # chats remembered as removed / blocked
GONE_TTL_S = 30 * 86400  # a removed chat is forgotten after a month (it is retried then)
COMMANDS = [
    {"command": "start", "description": "เริ่มใช้งาน Salid / Start Salid"},
    {"command": "help", "description": "วิธีใช้ / How to use Salid"},
]
HELP_TEXT = (
    "สลิด (Salid) ผู้ช่วยของทีม\n"
    '• ถามได้เลยในแชทส่วนตัว ส่วนในกลุ่มให้ @mention, ตอบข้อความของ Salid หรือเรียกชื่อ "สลิด"\n'
    "• ส่งรูปใบเสร็จหรือ PDF มาให้บันทึกลงบัญชี แล้วถามยอดค่าใช้จ่ายได้\n"
    "• ในกลุ่ม Salid จำข้อความไว้เป็นความจำของทีม (ไม่ข้ามไปกลุ่มอื่น) "
    "ลบ Salid ออกจากกลุ่มเพื่อให้หยุดจำ\n"
    "\n"
    "Salid, your team assistant\n"
    '• Ask anything in a private chat. In a group, @mention me, reply to my message or say "Salid".\n'
    "• Send a receipt photo or PDF to record it, then ask about spending.\n"
    "• In groups Salid remembers messages as team memory (never shared with other groups). "
    "Remove the bot from the group to stop it."
)
WELCOME_TEXT = (
    "สวัสดี! Salid เข้ากลุ่มนี้แล้ว\n"
    "• Salid จำข้อความและไฟล์ในกลุ่มนี้ไว้เป็นความจำของทีม เพื่อใช้ตอบคำถามในกลุ่มนี้เท่านั้น "
    "(ไม่ข้ามไปกลุ่มอื่น) เนื้อหาอาจถูกส่งให้ Google Vertex AI (Gemini) ประมวลผลเพื่อสรุปและตอบ\n"
    "• ความจำถูกลบเองเมื่อครบกำหนดเก็บ ขอลบก่อนได้โดยแจ้งผู้ดูแลบอท\n"
    "• ไม่ต้องการให้จำ: ให้แอดมินลบ Salid ออกจากกลุ่ม\n"
    "• พิมพ์ /help ดูวิธีใช้\n"
    "\n"
    "Hi! Salid joined this group.\n"
    "• Salid remembers the messages and files posted here as team memory, used only to answer "
    "questions in this group (never shared with other groups). Content may be processed by "
    "Google Vertex AI (Gemini) to summarise and answer.\n"
    "• Memory expires automatically after the retention period; ask the bot operator to delete "
    "it earlier.\n"
    "• To stop it, a group admin removes Salid from the group.\n"
    "• Send /help for usage."
)
DOWNLOAD_MAX = 20 * 1024 * 1024  # the Bot API serves files up to 20 MB
TYPING_EVERY = 4.0  # a typing indicator lasts 5 s
TYPING_MAX_S = 180.0


class TelegramError(Exception):
    """A failed Bot API call. Carries the method and status code only: no URL, no response text."""

    def __init__(self, method: str, code: int | None, retry_after: float | None = None):
        super().__init__(f"telegram {method} failed ({code})")
        self.method, self.code, self.retry_after = method, code, retry_after


def _ref(chat_id: int) -> str:
    """A short, stable digest of a chat id for logs: the raw id never reaches them."""
    return hashlib.sha256(str(chat_id).encode()).hexdigest()[:10]


def is_transient(e: BaseException) -> bool:
    """A failure worth redelivering the update for: a network error, a 429 or a 5xx."""
    if isinstance(e, TelegramError):
        return e.code is None or e.code == 429 or e.code >= 500
    return isinstance(e, (TimeoutError, OSError))


class _Gone:
    """Chats that removed / blocked the bot: a bounded (cap + TTL) set."""

    def __init__(self, clock) -> None:
        self._clock = clock
        self._at: OrderedDict[int, float] = OrderedDict()

    def add(self, chat_id: int) -> None:
        self._at[chat_id] = self._clock()
        self._at.move_to_end(chat_id)
        while len(self._at) > MAX_GONE:
            self._at.popitem(last=False)

    def discard(self, chat_id: int) -> None:
        self._at.pop(chat_id, None)

    def __contains__(self, chat_id: object) -> bool:
        at = self._at.get(chat_id)  # type: ignore[arg-type]
        if at is None:
            return False
        if self._clock() - at > GONE_TTL_S:
            del self._at[chat_id]  # type: ignore[arg-type]
            return False
        return True

    def __len__(self) -> int:
        return len(self._at)


class Api(Protocol):
    """The Bot API calls the adapter uses; tests replace it with a fake."""

    async def get_me(self) -> dict: ...

    async def delete_webhook(self) -> None: ...

    async def set_my_commands(self, commands: list[dict]) -> None: ...

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
        for attempt in range(MAX_RATE_RETRIES + 1):
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
                and attempt < MAX_RATE_RETRIES
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

    async def set_my_commands(self, commands: list[dict]) -> None:
        await self.call("setMyCommands", {"commands": commands})

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

    def __init__(self, api: Api, *, clock=time.monotonic, sleep=asyncio.sleep) -> None:
        self.api = api
        self._clock, self._sleep = clock, sleep
        self._locks: OrderedDict[int, asyncio.Lock] = OrderedDict()
        self._last: dict[int, float] = {}  # chat -> time of the last send
        self._window: dict[int, deque[float]] = {}  # group -> send times in the last minute
        self.gone = _Gone(clock)  # chats that removed / blocked the bot: never send there

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

    def mark_gone(self, chat_id: int) -> None:
        if chat_id not in self.gone:
            logger.warning(
                "telegram chat %s removed or blocked the bot: not sending there", _ref(chat_id)
            )
        self.gone.add(chat_id)

    def mark_alive(self, chat_id: int) -> None:
        self.gone.discard(chat_id)

    async def _pace(self, chat_id: int) -> None:
        """Wait until the next message to `chat_id` is within Telegram's limits."""
        now = self._clock()
        delay = 0.0
        last = self._last.get(chat_id)
        if last is not None:
            delay = last + SEND_GAP_S - now
        window = self._window.setdefault(chat_id, deque()) if chat_id < 0 else None
        if window is not None:
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= GROUP_PER_MINUTE:
                delay = max(delay, window[0] + 60 - now)
        if delay > 0:
            await self._sleep(delay)
            now = self._clock()
            if window is not None:
                while window and now - window[0] >= 60:
                    window.popleft()
        self._last[chat_id] = now
        if window is not None:
            window.append(now)
        if len(self._last) > 2000:  # forget chats idle for a minute
            self._last = {c: t for c, t in self._last.items() if now - t < 60}
        if len(self._window) > MAX_WINDOWS:
            self._trim_windows(now)

    def _trim_windows(self, now: float) -> None:
        """Drop the rate windows of groups idle for a minute, then the oldest if still too many."""
        for c in [c for c, w in self._window.items() if not w or now - w[-1] >= 60]:
            del self._window[c]
        while len(self._window) > MAX_WINDOWS:
            del self._window[next(iter(self._window))]

    def _lock_for(self, chat_id: int) -> asyncio.Lock:
        lock = self._locks.get(chat_id)
        if lock is None:
            lock = self._locks[chat_id] = asyncio.Lock()
        self._locks.move_to_end(chat_id)
        if len(self._locks) > MAX_LOCKS:  # evict the least recently used idle locks only
            for c in [c for c, lk in self._locks.items() if c != chat_id and not lk.locked()]:
                if len(self._locks) <= MAX_LOCKS:
                    break
                del self._locks[c]
        return lock

    async def _send(self, chat_id: int, params: dict) -> dict:
        lock = self._lock_for(chat_id)
        async with lock:
            await self._pace(chat_id)
            try:
                return await self.api.send_message(params)
            finally:
                self._last[chat_id] = self._clock()  # a 429 wait counts as time since the send

    async def notice(self, chat_id: int, text: str) -> None:
        """A message not tied to a user's question (the welcome note). Never raises."""
        if chat_id in self.gone:
            return
        try:
            await self._send(chat_id, {"chat_id": chat_id, "text": text})
        except Exception as e:
            if isinstance(e, TelegramError) and e.code == 403:
                self.mark_gone(chat_id)
            else:
                logger.warning("telegram notice failed (%s)", safe_exc(e))

    async def typing(self, msg: Incoming) -> None:
        if int(msg.room) in self.gone:
            return
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
        chat_id = int(msg.room)
        sent: list[str] = []
        if chat_id in self.gone:
            logger.warning(
                "telegram reply not sent: chat %s removed or blocked the bot", _ref(chat_id)
            )
            return sent
        for i, text in enumerate(texts):
            params: dict = {"chat_id": chat_id, "text": text}
            thread = self._thread(msg)
            if thread is not None:
                params["message_thread_id"] = thread
            if not msg.is_direct and i == 0:
                params["reply_parameters"] = {
                    "message_id": int(msg.message),
                    "allow_sending_without_reply": True,
                }
            try:
                result = await self._send(chat_id, params)
            except TelegramError as e:
                if e.code == 403:  # blocked / kicked / muted: stop, do not hammer the chat
                    self.mark_gone(chat_id)
                    return sent
                logger.error(
                    "telegram delivery failed: %d of %d messages sent (%s %s)",
                    len(sent), len(texts), e.method, e.code,
                )  # fmt: skip
                raise
            sent.append(str(result["message_id"]))
        return sent


# ------------------------------------------------------------------ adapter (Telegram -> Salid)


def _utf16_slice(text: str, offset: int, length: int) -> str:
    units = text.encode("utf-16-le")
    return units[offset * 2 : (offset + length) * 2].decode("utf-16-le", errors="ignore")


def strip_self_mentions(
    text: str, entities: list[dict], bot_id: str, username: str
) -> tuple[str, bool, bool]:
    """(text without Salid's own @mention / `@bot` command suffix, tagged Salid, tagged someone
    else). Entity offsets count UTF-16 code units, like every Telegram text offset."""
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
        elif kind == "bot_command":  # /help@SalidBot addresses this bot, /help@OtherBot another
            name, at, target = _utf16_slice(text, off, length).partition("@")
            if not at:
                continue
            if username and target.lower() == username.lower():
                me = True
                keep = textfmt.utf16_len(name)
                cut.append((off + keep, length - keep))  # drop the "@SalidBot" suffix only
            else:
                others = True
    for off, length in sorted(cut, reverse=True):
        units = units[: off * 2] + units[(off + length) * 2 :]
    plain = units.decode("utf-16-le", errors="ignore")
    return re.sub(r"[ \t]{2,}", " ", plain).strip(), me, others


class TelegramAdapter:
    def __init__(
        self,
        assistant: Assistant,
        port: TelegramPort,
        bot_id: str,
        username: str = "",
        *,
        sleep=asyncio.sleep,
    ):
        self.a, self.port, self.bot_id, self.username = assistant, port, bot_id, username
        self._sleep = sleep
        self._albums: dict[tuple[int, str], list[Incoming]] = {}

    async def process(self, update: dict) -> None:
        """One update. Raises when handling failed (the poller decides whether to redeliver)."""
        await self._update(update)

    async def on_update(self, update: dict) -> None:
        """One update. Never raises."""
        try:
            await self.process(update)
        except Exception as e:
            logger.error("telegram update failed (%s)", safe_exc(e))

    async def _update(self, update: dict) -> None:
        if update.get("my_chat_member"):
            await self._membership(update["my_chat_member"])
            return
        edited = "edited_message" in update
        raw = (update.get("edited_message") if edited else update.get("message")) or {}
        msg = self._incoming(raw)
        if msg is None:
            return
        self.port.mark_alive(int(msg.room))  # it is talking to us, so it did not remove us
        if edited:
            await self.a.on_edit(msg)
            return
        if raw.get("media_group_id"):
            msg = await self._album((int(msg.room), str(raw["media_group_id"])), msg)
            if msg is None:  # the first update of the album sends the combined message
                return
        if msg.is_direct:  # always addressed: show "typing" at once
            self.a.spawn(self.port.typing(msg))
        if (msg.is_direct or msg.mentions_bot) and re.fullmatch(r"/help(\s.*)?", msg.text, re.S):
            await self.port.deliver(msg, [HELP_TEXT], None)  # no model call
            return
        await self.a.handle(msg, self.port)

    async def _album(self, key: tuple[int, str], msg: Incoming) -> Incoming | None:
        """The photos of an album arrive as separate updates sharing a media_group_id. The first
        one waits until no more arrive (debounce) and returns them as one message; the others
        return None."""
        items = self._albums.get(key)
        if items is not None:
            items.append(msg)
            return None
        items = self._albums[key] = [msg]
        try:
            seen, waited = 0, 0.0
            while len(items) != seen and waited < ALBUM_MAX_WAIT_S:
                seen = len(items)
                await self._sleep(ALBUM_DEBOUNCE_S)
                waited += ALBUM_DEBOUNCE_S
        finally:
            items = self._albums.pop(key, items)
        items = sorted(items, key=lambda m: int(m.message))
        texts = list(dict.fromkeys(m.text for m in items if m.text))
        lead = next((m for m in items if m.text), items[0])
        return dataclasses.replace(
            lead,
            text="\n".join(texts),
            mentions_bot=any(m.mentions_bot for m in items),
            mentions_others=any(m.mentions_others for m in items),
            replies_to_bot=any(m.replies_to_bot for m in items),
            files=[f for m in items for f in m.files],
        )

    async def _membership(self, cm: dict) -> None:
        """The bot itself was added to / removed from a chat, or a user blocked / unblocked it."""
        chat = cm.get("chat") or {}
        new, old = cm.get("new_chat_member") or {}, cm.get("old_chat_member") or {}
        if (
            chat.get("id") is None
            or str((new.get("user") or {}).get("id", self.bot_id)) != self.bot_id
        ):
            return
        cid, status = int(chat["id"]), new.get("status")
        muted = status == "restricted" and not new.get("can_send_messages", True)
        if status in ("left", "kicked") or muted:
            self.port.mark_gone(cid)
            return
        self.port.mark_alive(cid)
        if chat.get("type") in ("group", "supergroup") and old.get("status") in (
            None,
            "left",
            "kicked",
        ):
            await self.port.notice(cid, WELCOME_TEXT)

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
        if (direct or mentions_bot) and re.fullmatch(
            r"/start(@\w+)?(\s.*)?", plain, re.S
        ):  # the first thing everyone sends (also /start@bot from a group deep link)
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
            if m.get("message_thread_id") and (m.get("is_topic_message") or direct)
            else None,  # a private chat with topics has them too; a reply chain is no topic
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


def privacy_warning(me: dict) -> str | None:
    """A clear warning when getMe says privacy mode is on (the bot sees only commands, mentions
    and replies in groups, so team memory would silently be incomplete)."""
    if me.get("can_read_all_group_messages") is False and me.get("can_join_groups") is not False:
        return (
            f"telegram privacy mode is ON for @{me.get('username') or me.get('id')}: in groups "
            "Salid only receives commands, @mentions and replies, so team memory is incomplete. "
            "Fix: @BotFather /setprivacy -> Disable, then REMOVE and RE-ADD the bot to every "
            "group (or make it a group admin)."
        )
    return None


class Poller:
    def __init__(
        self,
        api: Api,
        adapter: TelegramAdapter,
        *,
        sleep=asyncio.sleep,
        clock=time.time,
        timeout: int = POLL_SECONDS,
        batch_timeout: float = BATCH_TIMEOUT_S,
    ) -> None:
        self.api, self.adapter, self._sleep, self.timeout = api, adapter, sleep, timeout
        self.batch_timeout = batch_timeout
        self._done: OrderedDict[int, None] = OrderedDict()  # update_ids handled to the end
        self._failures: dict[int, int] = {}  # update_id -> transient handler failures so far
        self._bg: set[asyncio.Task] = set()
        self._clock = clock
        self.offset: int | None = None
        self.backoff = BACKOFF_START
        self._webhook_cleared = False
        self._last_update_at: float | None = None
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    async def startup(self) -> None:
        """Once per run: warn when privacy mode is on, register /start and /help. Never raises."""
        try:
            warning = privacy_warning(await self.api.get_me())
            if warning:
                logger.warning(warning)
        except Exception as e:
            logger.warning("telegram getMe check failed (%s)", safe_exc(e))
        try:
            await self.api.set_my_commands(COMMANDS)
        except Exception as e:
            logger.warning("telegram setMyCommands failed (%s)", safe_exc(e))

    async def _handle(self, update: dict) -> bool:
        """True when the update is finished with (handled, or failed for good); False on a
        transient failure that a redelivery may fix."""
        uid = int(update["update_id"])
        if uid in self._done:
            return True  # redelivered because a neighbour failed: handling it twice would repeat it
        try:
            await self.adapter.process(update)
        except Exception as e:
            if is_transient(e) and self._failures.get(uid, 0) + 1 < MAX_HANDLER_RETRIES:
                self._failures[uid] = self._failures.get(uid, 0) + 1
                logger.warning(
                    "telegram update handler failed, will be redelivered (%s)", safe_exc(e)
                )
                return False
            logger.error("telegram update failed for good (%s)", safe_exc(e))
        self._failures.pop(uid, None)
        self._done[uid] = None
        while len(self._done) > DONE_KEEP:
            self._done.popitem(last=False)
        return True

    async def poll_once(self) -> int:
        """One getUpdates round; returns how many updates were fetched. The batch is handled
        concurrently and awaited (at most `batch_timeout`); the offset moves past the updates
        that finished, stopping at the first transient handler failure so Telegram re-delivers
        it. Raises on an API error (the offset is unchanged)."""
        if not self._webhook_cleared:  # a webhook and getUpdates exclude each other
            await self.api.delete_webhook()
            self._webhook_cleared = True
        now = self._clock()
        if self._last_update_at is None:
            self._last_update_at = now
        elif self.offset is not None and now - self._last_update_at > IDLE_RESET_S:
            logger.warning(
                "telegram: no updates for %.1f days; resetting the offset because Telegram "
                "numbers the next update_id at random after a quiet week",
                (now - self._last_update_at) / 86400,
            )
            self.offset, self._last_update_at = None, now
        updates = await self.api.get_updates(self.offset, self.timeout)
        if updates:
            await self._handle_batch(updates)
            self._last_update_at = self._clock()
        return len(updates)

    async def _handle_batch(self, updates: list[dict]) -> None:
        tasks = [asyncio.create_task(self._handle(u)) for u in updates]
        try:
            _done, pending = await asyncio.wait(tasks, timeout=self.batch_timeout)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise
        if pending:  # keep them running, but do not hold the poll for them
            logger.warning("telegram: %d update handler(s) still running; moving on", len(pending))
            for t in pending:
                self._bg.add(t)
                t.add_done_callback(self._bg.discard)
        for u, t in zip(updates, tasks, strict=True):
            if t not in pending and not t.result():
                break  # redeliver this update (and, with it, the ones after it)
            self.offset = max(self.offset or 0, int(u["update_id"]) + 1)

    async def run(self) -> None:
        await self.startup()
        while not self._stop:
            try:
                await self.poll_once()
                self.backoff = BACKOFF_START
            except asyncio.CancelledError:
                raise
            except Exception as e:
                delay = self.backoff
                if isinstance(e, TelegramError) and e.code == 409:
                    self._webhook_cleared = False  # a webhook may have been set again
                    delay = max(delay, CONFLICT_BACKOFF_MIN)
                    logger.error(
                        "telegram getUpdates 409 Conflict: another process or a webhook is "
                        "using this bot token. Run ONE instance per token. Retrying in %.0fs",
                        delay,
                    )
                else:
                    logger.warning(
                        "telegram poll failed, retrying in %.0fs (%s)", delay, safe_exc(e)
                    )
                await self._sleep(delay)
                self.backoff = min(self.backoff * 2, BACKOFF_MAX)


async def check(ps) -> dict:
    """Read-only token check for the smoke script: getMe. Only the username and id are returned."""
    api = TelegramApi(ps.telegram_bot_token)
    try:
        me = await api.get_me()
    finally:
        await api.aclose()
    return {"username": me.get("username"), "id": me.get("id")}
