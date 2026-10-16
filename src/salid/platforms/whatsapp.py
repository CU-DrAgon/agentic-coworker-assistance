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
is one customer (their wa_id) and is always addressed. Memory scope is the room itself, so a chat
sees only its own history. Salid only ever answers an inbound message, which keeps it inside the
24-hour customer service window; it never sends a proactive message (that needs an approved
template). A wa_id is a phone number: it is never logged (core logs the internal room id).
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
MAX_BODY = 1_000_000
WEBHOOK_PATH = "/whatsapp/webhook"
SEEN_KEEP = 5000
SENT_KEEP = 2000
DOWNLOAD_MAX = 20 * 1024 * 1024


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

    async def send_text(self, to: str, body: str, reply_to: str | None) -> str: ...

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

    async def send_text(self, to: str, body: str, reply_to: str | None) -> str:
        payload: dict = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
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

    def __init__(self, api: Api) -> None:
        self.api = api
        self.sent_ids: OrderedDict[str, None] = OrderedDict()  # to recognise replies to Salid

    async def room_view(self, msg: Incoming) -> RoomView:
        return RoomView(None, [])  # 1:1 only: the room alone

    async def is_admin(self, msg: Incoming) -> bool:
        return False  # executives come from SALID_WHATSAPP_EXEC_NUMBERS only

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

    async def deliver(self, msg: Incoming, texts: list[str], ack: object | None) -> list[str]:
        sent: list[str] = []
        for i, text in enumerate(texts):
            mid = await self.api.send_text(msg.room, text, msg.message if i == 0 else None)
            if mid:
                self.sent_ids[mid] = None
                while len(self.sent_ids) > SENT_KEEP:
                    self.sent_ids.popitem(last=False)
            sent.append(mid or f"{msg.message}.{i}")
        return sent


# ------------------------------------------------------------------ adapter (WhatsApp -> Salid)

_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


class WhatsAppAdapter:
    def __init__(self, assistant: Assistant, port: WhatsAppPort, phone_number_id: str) -> None:
        self.a, self.port, self.phone_number_id = assistant, port, phone_number_id
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
        names = {
            c.get("wa_id"): (c.get("profile") or {}).get("name") or ""
            for c in value.get("contacts") or []
        }
        for m in value.get("messages") or []:  # `statuses` are delivery receipts: ignored
            try:
                msg = self._incoming(m, names)
                if msg is None:
                    continue
                self.a.spawn(self.port.read(msg))
                await self.a.handle(msg, self.port)
            except Exception as e:
                logger.error("whatsapp message failed (%s)", safe_exc(e))

    def _first_seen(self, mid: str) -> bool:
        if mid in self._seen:
            return False
        self._seen[mid] = None
        while len(self._seen) > SEEN_KEEP:
            self._seen.popitem(last=False)
        return True

    def _incoming(self, m: dict, names: dict) -> Incoming | None:
        mid, sender, kind = m.get("id"), m.get("from"), m.get("type")
        if not mid or not sender:
            return None
        text, files = "", []
        if kind == "text":
            text = (m.get("text") or {}).get("body") or ""
        elif kind in ("image", "document"):
            media = m.get(kind) or {}
            text = media.get("caption") or ""
            if media.get("id"):
                files = [self._file(kind, media, str(mid))]
        else:
            return None  # audio, video, stickers, location, reactions, ...
        if not text.strip() and not files:
            return None
        if not self._first_seen(str(mid)):
            return None  # Meta retried a delivery
        ts = m.get("timestamp")
        try:
            at = datetime.fromtimestamp(int(ts), UTC)
        except (TypeError, ValueError, OverflowError, OSError):
            at = datetime.now(UTC)
        replied = (m.get("context") or {}).get("id")
        return Incoming(
            platform=PLATFORM,
            tenant=self.phone_number_id,
            room=str(sender),
            user=str(sender),
            user_name=names.get(sender) or "",
            text=text.strip(),
            message=str(mid),
            at=at,
            is_direct=True,  # always addressed
            replies_to_bot=bool(replied and replied in self.port.sent_ids),
            files=files,
        )

    def _file(self, kind: str, media: dict, mid: str) -> FileRef:
        api, media_id = self.port.api, str(media["id"])
        mime = media.get("mime_type")
        if kind == "document":
            name = media.get("filename") or f"file-{mid}"
        else:
            name = f"image-{mid}{_EXT.get(mime or '', mimetypes.guess_extension(mime or '') or '.jpg')}"
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
