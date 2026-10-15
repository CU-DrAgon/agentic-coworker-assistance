"""Outgoing email: a pluggable `EmailSender`.

* `LogOnlySender` (default): sends nothing. The caller records the email in the store and tells
  the user email sending is not configured yet.
* `GmailSender`: Gmail API (`users.messages.send`) with an OAuth refresh token of a dedicated
  Google account (scope gmail.send). Active only when SALID_GMAIL_CLIENT_ID / _CLIENT_SECRET /
  _REFRESH_TOKEN / _SENDER are all set. See README ("Email").

Whether to send at all (draft shown, confirmed in a later message, role allowed, daily limit) is
decided in `salid.secretary`; the senders only deliver.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
MAX_RECIPIENTS = 20
MAX_ATTACHMENT_BYTES = 15 * 1024 * 1024
_ADDR = re.compile(
    r"^[A-Za-z0-9._%+\-']+@[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?)+$"
)


def valid_address(addr: str) -> bool:
    a = (addr or "").strip()
    return len(a) <= 254 and bool(_ADDR.match(a)) and ".." not in a


def clean_addresses(values: list[str] | None) -> tuple[list[str], list[str]]:
    """-> (valid lowercase-domain addresses without duplicates, the invalid inputs)."""
    ok: list[str] = []
    bad: list[str] = []
    for v in values or []:
        a = (v or "").strip().strip("<>")
        if not a:
            continue
        if not valid_address(a):
            bad.append(a)
            continue
        local, domain = a.rsplit("@", 1)
        a = f"{local}@{domain.lower()}"
        if a.lower() not in {x.lower() for x in ok}:
            ok.append(a)
    return ok, bad


@dataclass
class OutgoingEmail:
    to: list[str]
    subject: str
    body: str
    cc: list[str] = field(default_factory=list)
    attachments: list[tuple[str, bytes, str]] = field(default_factory=list)  # name, data, mime
    sender: str = ""


@dataclass
class SendResult:
    ok: bool
    status: str  # "sent" | "logged" | "failed"
    detail: str = ""
    message_id: str = ""


class EmailSender(Protocol):
    name: str
    configured: bool

    def send(self, mail: OutgoingEmail) -> SendResult: ...


class LogOnlySender:
    """No real delivery: the email is only recorded by the caller."""

    name = "log_only"
    configured = False

    def send(self, mail: OutgoingEmail) -> SendResult:
        logger.info("email not sent (no sender configured): %d recipient(s)", len(mail.to))
        return SendResult(ok=False, status="logged", detail="no email sender configured")


def build_mime(mail: OutgoingEmail) -> EmailMessage:
    msg = EmailMessage()
    msg["To"] = ", ".join(mail.to)
    if mail.cc:
        msg["Cc"] = ", ".join(mail.cc)
    if mail.sender:
        msg["From"] = mail.sender
    msg["Subject"] = mail.subject.replace("\r", " ").replace("\n", " ")
    msg.set_content(mail.body)
    for name, data, mime in mail.attachments:
        main, _, sub = (mime or "application/octet-stream").partition("/")
        msg.add_attachment(data, maintype=main, subtype=sub or "octet-stream", filename=name)
    return msg


class GmailSender:
    name = "gmail"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        sender: str,
        http: httpx.Client | None = None,
    ) -> None:
        self.client_id, self.client_secret = client_id, client_secret
        self.refresh_token, self.sender = refresh_token, sender
        self.http = http or httpx.Client(timeout=30)

    @property
    def configured(self) -> bool:
        return all((self.client_id, self.client_secret, self.refresh_token, self.sender))

    def token_request(self) -> tuple[str, dict]:
        return TOKEN_URL, {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token",
        }

    def build_request(self, mail: OutgoingEmail, access_token: str) -> tuple[str, dict, dict]:
        """-> (url, headers, json body) of the Gmail `messages.send` call."""
        mail.sender = mail.sender or self.sender
        raw = base64.urlsafe_b64encode(build_mime(mail).as_bytes()).decode("ascii")
        return SEND_URL, {"Authorization": f"Bearer {access_token}"}, {"raw": raw}

    def send(self, mail: OutgoingEmail) -> SendResult:
        if not self.configured:
            return SendResult(False, "failed", "gmail is not configured")
        try:
            url, data = self.token_request()
            tok = self.http.post(url, data=data)
            tok.raise_for_status()
            access = tok.json()["access_token"]
            url, headers, body = self.build_request(mail, access)
            r = self.http.post(url, headers=headers, json=body)
            r.raise_for_status()
            return SendResult(True, "sent", message_id=str(r.json().get("id", "")))
        except Exception as e:  # never log addresses or contents
            logger.warning("gmail send failed (%s)", type(e).__name__)
            return SendResult(False, "failed", type(e).__name__)


def make_sender(settings) -> EmailSender:
    g = GmailSender(
        settings.gmail_client_id,
        settings.gmail_client_secret,
        settings.gmail_refresh_token,
        settings.gmail_sender,
    )
    return g if g.configured else LogOnlySender()
