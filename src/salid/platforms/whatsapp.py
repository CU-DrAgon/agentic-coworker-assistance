"""WhatsApp adapter: Cloud API webhook (signature-checked) -> `Incoming`, and Salid's replies back.

* `verify_signature`: X-Hub-Signature-256, `sha256=` + HMAC-SHA256 of the raw body keyed with the
  app secret, compared in constant time before the body is parsed.
* `create_app`: a small FastAPI app. GET answers Meta's verification handshake, POST answers 200
  at once and handles the messages in the background (Meta retries slow webhooks, so message ids
  are de-duplicated).
* `WhatsAppApi`: the few Graph API calls Salid needs, over `httpx` (no extra dependency). Every
  failure is re-raised as `WhatsAppError` (operation and status code only): an httpx error text
  can quote a URL, and a Graph error body can quote a phone number.
* `WhatsAppPort`: plain replies of at most 4096 characters, quoting the question, a read receipt
  with a typing indicator while Salid thinks.

The official API only has 1:1 chats: the tenant is the business phone number (its id), every room
is one customer and is always addressed. Memory scope is the room itself, so a chat sees only its
own history. Salid only ever answers an inbound message, which keeps it inside the 24-hour customer
service window; it never sends a proactive message (that needs an approved template). A wa_id is a
phone number: it is never logged (core logs the internal room id).

Identity (usernames / BSUIDs, since April 2026): a message may carry no phone number (`from` and
`wa_id` absent) but always has `from_user_id`, the business-scoped user id (BSUID, `US.1349...`).
One person is one identity:
* user and room: the BSUID whenever the message has one, else the wa_id. The BSUID never changes,
  while the phone number can appear or vanish (the user adopts a username, or we messaged them in
  the last 30 days) and can be reused by a different person. Keying both on the BSUID keeps one
  chat in one room with one receipt history, and a new holder of an old number cannot see it.
* the phone number is kept only as the outbound recipient (`port.phones`) and for executive
  matching (`SALID_WHATSAPP_EXEC_NUMBERS`; `exec_bsuids` may also list BSUIDs).
* reply: `to` when a phone number is known (or is the user), `recipient` (BSUID) otherwise.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import mimetypes
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from ..qa import MAX_FILE_BYTES
from . import textfmt
from .config import PlatformSettings
from .core import Assistant, FileRef, HistLine, Incoming, RoomView, safe_exc

# The access token travels in a header, but URLs carry phone / media ids: keep these quiet.
for _noisy in ("httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

PLATFORM = "whatsapp"
GRAPH = "https://graph.facebook.com"
TEXT_LIMIT = 4096  # characters per text message
MAX_MESSAGES = 6
MAX_BODY = 3 * 1024 * 1024  # Meta documents webhook payloads of up to 3 MB
WEBHOOK_PATH = "/whatsapp/webhook"
SEEN_KEEP = 5000
SENT_KEEP = 2000
DOWNLOAD_MAX = 20 * 1024 * 1024
RETRY_CODES = frozenset({130429, 131056})  # throughput / pair rate limit: temporary
RETRY_DELAYS = (4, 16)  # seconds, Meta's 4^X backoff; bounded so a reply never hangs for long
HUMAN_REPLY = (
    "Salid is an AI assistant. To reach a person, please contact the administrators of this "
    "workspace directly (whoever set Salid up for you).\n\n"
    "Salid เป็นผู้ช่วย AI หากต้องการติดต่อเจ้าหน้าที่ "
    "กรุณาติดต่อผู้ดูแลของ workspace นี้โดยตรง (ผู้ที่ตั้งค่า Salid ให้คุณ)"
)
_HUMAN_EXACT = frozenset(
    {"human", "a human", "agent", "person", "คน", "คนจริง", "คนจริงๆ", "เจ้าหน้าที่"}
)
_HUMAN_PHRASES = (
    "talk to a human",
    "talk to human",
    "speak to a human",
    "speak to a person",
    "talk to a person",
    "real person",
    "human agent",
    "ติดต่อเจ้าหน้าที่",
    "คุยกับเจ้าหน้าที่",
    "คุยกับคน",
)


def is_human_request(text: str) -> bool:
    """A short message asking for a person ("human", "คน", "ติดต่อเจ้าหน้าที่", ...)."""
    t = " ".join(text.lower().split()).strip(" .!?,…\"'()")
    return t in _HUMAN_EXACT or (len(t) <= 80 and any(p in t for p in _HUMAN_PHRASES))


class WhatsAppError(Exception):
    """A failed Graph API call. Carries the operation and status codes only: no URL, no body."""

    def __init__(self, op: str, status: int | None, graph_code: int | None = None):
        super().__init__(f"whatsapp {op} failed ({status})")
        self.op, self.status, self.graph_code = op, status, graph_code


def verify_signature(app_secret: str, body: bytes, header: str | None) -> bool:
    """True only for a correct `X-Hub-Signature-256: sha256=<hex>` of `body`."""
    if not app_secret or not header:
        return False
    scheme, _, given = header.strip().partition("=")
    if scheme != "sha256" or not given:
        return False
    expected = hmac.new(app_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected.encode("utf-8"), given.lower().encode("utf-8"))


class Api(Protocol):
    """The Graph API calls the adapter uses; tests replace it with a fake."""

    async def send_text(self, recipient: str, body: str, reply_to: str | None) -> str:
        """`recipient` is a wa_id (digits) or a BSUID (sent as `recipient`, not `to`)."""
        ...

    async def mark_read(self, message_id: str) -> None: ...

    def fetch_media(self, media_id: str) -> bytes:
        """media lookup + download (blocking: called in a worker thread)."""
        ...


class WhatsAppApi:
    def __init__(
        self,
        access_token: str,
        phone_number_id: str,
        version: str = "v26.0",
        *,
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
    ) -> None:
        self._headers = {"Authorization": f"Bearer {access_token}"}
        self._phone = phone_number_id
        self._base = f"{GRAPH}/{version}"
        self._transport = transport
        async_t = transport if isinstance(transport, httpx.AsyncBaseTransport) else None
        self._client = httpx.AsyncClient(timeout=15.0, transport=async_t, headers=self._headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, op: str, payload: dict) -> dict:
        try:
            r = await self._client.post(f"{self._base}/{self._phone}/messages", json=payload)
        except httpx.HTTPError:
            raise WhatsAppError(op, None) from None  # its text may quote the URL
        if r.status_code != 200:
            code = None
            with contextlib.suppress(ValueError, AttributeError):
                code = ((r.json() or {}).get("error") or {}).get("code")
            raise WhatsAppError(op, r.status_code, code if isinstance(code, int) else None)
        try:
            body = r.json()
        except ValueError:
            body = {}
        return body if isinstance(body, dict) else {}

    async def send_text(self, recipient: str, body: str, reply_to: str | None) -> str:
        phone = recipient.isascii() and recipient.isdigit()
        payload: dict = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to" if phone else "recipient": recipient,
            "type": "text",
            "text": {"body": body, "preview_url": False},
        }
        if reply_to:
            payload["context"] = {"message_id": reply_to}
        res = await self._post("send", payload)
        return str(((res.get("messages") or [{}])[0]).get("id") or "")

    async def mark_read(self, message_id: str) -> None:
        """Read receipt plus the typing indicator (shown until Salid answers or 25 s pass)."""
        await self._post(
            "read",
            {
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id,
                "typing_indicator": {"type": "text"},
            },
        )

    async def profile(self) -> dict:
        """The business number's name, for the smoke check."""
        try:
            r = await self._client.get(
                f"{self._base}/{self._phone}",
                params={"fields": "display_phone_number,verified_name"},
            )
        except httpx.HTTPError:
            raise WhatsAppError("profile", None) from None
        if r.status_code != 200:
            raise WhatsAppError("profile", r.status_code)
        body = r.json()
        return body if isinstance(body, dict) else {}

    def fetch_media(self, media_id: str) -> bytes:
        limit = min(DOWNLOAD_MAX, MAX_FILE_BYTES)
        sync_t = self._transport if isinstance(self._transport, httpx.BaseTransport) else None
        try:
            with httpx.Client(timeout=60.0, transport=sync_t, headers=self._headers) as c:
                r = c.get(f"{self._base}/{media_id}")
                info = r.json() if r.status_code == 200 else {}
                url = info.get("url") if isinstance(info, dict) else None
                if not url or not str(url).startswith("https://"):
                    raise WhatsAppError("media", r.status_code)
                if int(info.get("file_size") or 0) > limit:
                    raise ValueError("file over the size limit")
                data = bytearray()
                with c.stream("GET", url) as resp:  # the same Bearer header, per Meta's docs
                    if resp.status_code != 200:
                        raise WhatsAppError("download", resp.status_code)
                    for chunk in resp.iter_bytes():
                        data += chunk
                        if len(data) > limit:
                            raise ValueError("file over the size limit")
                return bytes(data)
        except httpx.HTTPError:
            raise WhatsAppError("download", None) from None


# ------------------------------------------------------------------ port (Salid -> WhatsApp)


class WhatsAppPort:
    platform = PLATFORM
    max_messages = MAX_MESSAGES

    def __init__(
        self,
        api: Api,
        *,
        exec_bsuids: frozenset[str] | set[str] = frozenset(),
        exec_numbers: frozenset[str] | set[str] = frozenset(),
        sleep: Callable = asyncio.sleep,
    ) -> None:
        self.api = api
        self.exec_bsuids = frozenset(exec_bsuids)  # executives listed by BSUID
        self.exec_numbers = frozenset(exec_numbers)  # phone numbers; the adapter fills it if empty
        self.phones: OrderedDict[str, str] = OrderedDict()  # BSUID -> its current phone number
        self._sleep = sleep
        self.sent_ids: OrderedDict[str, None] = OrderedDict()  # to recognise replies to Salid

    async def room_view(self, msg: Incoming) -> RoomView:
        return RoomView(None, [])  # 1:1 only: the room alone

    async def is_admin(self, msg: Incoming) -> bool:
        # msg.user is the BSUID (or the phone number when there is none). Executives are the
        # listed BSUIDs, or whoever's phone number (as sent with this message) is listed.
        return msg.user in self.exec_bsuids or self.recipient(msg.user) in self.exec_numbers

    def remember_phone(self, user: str, phone: str) -> None:
        """Track the phone number a BSUID user is currently reachable on (none: forget it)."""
        if not phone or phone == user:
            self.phones.pop(user, None)
            return
        self.phones[user] = phone
        self.phones.move_to_end(user)
        while len(self.phones) > SENT_KEEP:
            self.phones.popitem(last=False)

    def recipient(self, user: str) -> str:
        return self.phones.get(user) or user

    def format(self, text: str) -> list[str]:
        return textfmt.split_text(textfmt.to_whatsapp(text), TEXT_LIMIT)

    async def history(self, msg: Incoming, limit: int) -> list[HistLine]:
        return []  # the Cloud API cannot read past messages

    async def read(self, msg: Incoming) -> None:
        try:
            await self.api.mark_read(msg.message)
        except Exception as e:
            logger.warning("whatsapp read receipt failed (%s)", safe_exc(e))

    async def ack(self, msg: Incoming, text: str) -> object | None:
        return None  # the typing indicator from `read` already says Salid is on it

    async def drop_ack(self, ack: object) -> None:
        return None

    async def _send(self, recipient: str, text: str, reply_to: str | None) -> str:
        """send_text, retried a bounded number of times on a temporary rate limit."""
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                return await self.api.send_text(recipient, text, reply_to)
            except WhatsAppError as e:
                if e.graph_code not in RETRY_CODES or attempt == len(RETRY_DELAYS):
                    raise
                logger.warning(
                    "whatsapp rate limited (code %s): retry %d in %d s",
                    e.graph_code,
                    attempt + 1,
                    RETRY_DELAYS[attempt],
                )
                await self._sleep(RETRY_DELAYS[attempt])
        raise AssertionError("unreachable")  # pragma: no cover

    async def deliver(self, msg: Incoming, texts: list[str], ack: object | None) -> list[str]:
        sent: list[str] = []
        for i, text in enumerate(texts):
            try:
                mid = await self._send(
                    self.recipient(msg.user), text, msg.message if i == 0 else None
                )
            except Exception as e:
                code = getattr(e, "graph_code", None)
                logger.error(
                    "whatsapp reply incomplete: %d of %d parts sent, %d not sent (%s, code %s)",
                    i,
                    len(texts),
                    len(texts) - i,
                    safe_exc(e),
                    code,
                )
                raise
            if mid:
                self.sent_ids[mid] = None
                while len(self.sent_ids) > SENT_KEEP:
                    self.sent_ids.popitem(last=False)
            sent.append(mid or f"{msg.message}.{i}")
        return sent


# ------------------------------------------------------------------ adapter (WhatsApp -> Salid)

_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "audio/ogg": ".ogg",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "audio/amr": ".amr",
}
_DEFAULT_EXT = {"image": ".jpg", "audio": ".ogg"}


class WhatsAppAdapter:
    def __init__(self, assistant: Assistant, port: WhatsAppPort, phone_number_id: str) -> None:
        self.a, self.port, self.phone_number_id = assistant, port, phone_number_id
        if not port.exec_numbers:
            port.exec_numbers = frozenset(getattr(assistant, "exec_ids", {}).get("whatsapp") or ())
        self._seen: OrderedDict[str, None] = OrderedDict()

    async def on_payload(self, payload: dict) -> None:
        """One webhook POST. Never raises."""
        try:
            for entry in payload.get("entry") or []:
                for change in entry.get("changes") or []:
                    await self._value(change.get("value") or {})
        except Exception as e:
            logger.error("whatsapp payload failed (%s)", safe_exc(e))

    async def _value(self, value: dict) -> None:
        if (value.get("metadata") or {}).get("phone_number_id") != self.phone_number_id:
            return  # another number of the same app (or a malformed payload)
        names: dict = {}
        for c in value.get("contacts") or []:
            name = (c.get("profile") or {}).get("name") or ""
            for key in (c.get("wa_id"), c.get("user_id")):  # phone number and/or BSUID
                if key:
                    names[key] = name
        for st in value.get("statuses") or []:  # delivery receipts: only failures are logged
            self._status(st)
        for m in value.get("messages") or []:
            try:
                msg = self._incoming(m, names)
                if msg is None:
                    continue
                if not msg.files and is_human_request(msg.text):
                    await self._escalate(msg)  # a fixed answer: the model is not called
                    continue
                self.a.spawn(self.port.read(msg))
                await self.a.handle(msg, self.port)
            except Exception as e:
                logger.error("whatsapp message failed (%s)", safe_exc(e))

    @staticmethod
    def _status(st: object) -> None:
        """Log a failed send by error code only (the status also carries a phone number)."""
        if not isinstance(st, dict) or st.get("status") != "failed":
            return
        codes = [
            e["code"]
            for e in st.get("errors") or []
            if isinstance(e, dict) and isinstance(e.get("code"), int)
        ]
        logger.warning("whatsapp delivery failed (error codes %s)", codes or "unknown")

    async def _escalate(self, msg: Incoming) -> None:
        try:
            await self.port.deliver(msg, [HUMAN_REPLY], None)
        except Exception as e:
            logger.error("whatsapp human-contact reply failed (%s)", safe_exc(e))

    def _first_seen(self, mid: str) -> bool:
        if mid in self._seen:
            return False
        self._seen[mid] = None
        while len(self._seen) > SEEN_KEEP:
            self._seen.popitem(last=False)
        return True

    def _incoming(self, m: dict, names: dict) -> Incoming | None:
        mid, kind = m.get("id"), m.get("type")
        phone, bsuid = str(m.get("from") or ""), str(m.get("from_user_id") or "")
        user = bsuid or phone  # one person, one identity: the BSUID, the phone only as a fallback
        if not mid or not user:
            return None
        text, files = "", []
        if kind == "text":
            text = (m.get("text") or {}).get("body") or ""
        elif kind in ("image", "document", "audio"):
            media = m.get(kind) or {}
            text = media.get("caption") or ""
            if media.get("id"):
                files = [self._file(kind, media, str(mid))]
        else:
            return None  # video, stickers, location, reactions, ...
        if not text.strip() and not files:
            return None
        if not self._first_seen(str(mid)):
            return None  # Meta retried a delivery
        self.port.remember_phone(user, phone)
        ts = m.get("timestamp")
        try:
            at = datetime.fromtimestamp(int(ts), UTC)
        except (TypeError, ValueError, OverflowError, OSError):
            at = datetime.now(UTC)
        replied = (m.get("context") or {}).get("id")
        return Incoming(
            platform=PLATFORM,
            tenant=self.phone_number_id,
            room=user,
            user=user,
            user_name=names.get(phone) or names.get(bsuid) or "",
            text=text.strip(),
            message=str(mid),
            at=at,
            is_direct=True,  # always addressed
            replies_to_bot=bool(replied and replied in self.port.sent_ids),
            files=files,
        )

    def _file(self, kind: str, media: dict, mid: str) -> FileRef:
        api, media_id = self.port.api, str(media["id"])
        mime = (media.get("mime_type") or "").split(";")[0].strip().lower() or None
        if kind == "document":
            name = media.get("filename") or f"file-{mid}"
        else:  # voice notes are "audio/ogg; codecs=opus"
            ext = _EXT.get(mime or "") or mimetypes.guess_extension(mime or "")
            name = f"{kind}-{mid}{ext or _DEFAULT_EXT[kind]}"
        return FileRef(name, mime, 0, lambda: api.fetch_media(media_id))


# ------------------------------------------------------------------ webhook


def create_app(
    app_secret: str,
    verify_token: str,
    adapter: WhatsAppAdapter,
    spawn: Callable | None = None,
):
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

    @app.get(WEBHOOK_PATH)
    async def verify(request: Request):
        q = request.query_params
        given = (q.get("hub.verify_token") or "").encode("utf-8")
        if (
            q.get("hub.mode") != "subscribe"
            or not verify_token
            or not hmac.compare_digest(given, verify_token.encode("utf-8"))
        ):
            raise HTTPException(status_code=403)
        return PlainTextResponse(q.get("hub.challenge") or "")

    @app.post(WEBHOOK_PATH)
    async def webhook(request: Request):
        signature = request.headers.get("x-hub-signature-256")
        if not signature:
            raise HTTPException(status_code=401)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY:
                raise HTTPException(status_code=413)
        raw = bytes(body)
        if not verify_signature(app_secret, raw, signature):
            raise HTTPException(status_code=401)
        try:
            data = json.loads(raw)
        except ValueError:
            raise HTTPException(status_code=400) from None
        if isinstance(data, dict):
            run(adapter.on_payload(data))
        return {}

    @app.get("/health")
    async def health():
        return {"ok": True}

    app.state.tasks = tasks
    return app


async def check(ps: PlatformSettings) -> dict:
    """Read-only token check for the smoke script. Only the verified business name is returned."""
    api = WhatsAppApi(
        ps.whatsapp_access_token, ps.whatsapp_phone_number_id, ps.whatsapp_api_version
    )
    try:
        info = await api.profile()
    finally:
        await api.aclose()
    return {"verified_name": info.get("verified_name")}
