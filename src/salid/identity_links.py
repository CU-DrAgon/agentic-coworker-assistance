"""One-time codes that bind a chat account to an org member, or attach a bot tenant to an org.

Two kinds of code, one flow (design: docs/superpowers/specs/2026-10-18-salid-workspace-parity-
design.md, "Central identity and cross-platform memory"):

* ``identity``: a signed-in member presses "Link LINE / Telegram / WhatsApp" on the web and gets a
  code for ONE platform. Typing it to Salid in a private chat binds the sender's platform account
  to that member (`orgs.Member.identities`). Only a verified identity widens what Salid may use.
* ``attach``: an org owner / admin asks for a code for a platform; typing it to the bot in a
  private chat attaches that bot tenant (LINE channel, Telegram bot, WhatsApp number, Slack team)
  to the org as an `orgs.Workspace`.

A code is 8 random digits shown as ``1234-5678``, valid 15 minutes, single use, and stored as the
document ``salid_identity_codes/{digits}`` in the `DocStore`. At most 5 live codes per
(org, creator, kind). Redeeming is one step under the org store's lock: every check first, then
the code is marked used, then the link / attach is applied. A failure after that point burns the
code (the person asks for a new one) so a code can never be applied twice.

Whatever goes wrong, the person typing the code is told only "invalid or expired"; `CodeError.
reason` is for logs. Codes are never accepted in group chats (the platform layer enforces that).
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .orgs import (
    IDENTITY_PLATFORMS,
    IdentityTaken,
    OrgStore,
    Workspace,
    WorkspaceTaken,
    stamp,
    workspace_id,
)

KIND = "salid_identity_codes"
CODE_TTL = timedelta(minutes=15)
MAX_LIVE_CODES = 5
CODE_DIGITS = 8
# Discord links through /ตั้งค่า, so it never takes a code
CODE_PLATFORMS = tuple(p for p in IDENTITY_PLATFORMS if p != "discord")
PRUNE_AFTER = timedelta(days=1)

# "link 1234-5678", "/link 12345678", "เชื่อมบัญชี 1234 5678", "Link: 1234–5678"; the whole message
_COMMAND = re.compile(
    r"^\s*/?(?:link|เชื่อมบัญชี)(?:@\w+)?\s*[:：]?\s*(?P<code>\d[\d\s\-–—_.]*\d)\s*$",
    re.IGNORECASE,
)

REPLY_LINKED = "เชื่อมบัญชีเรียบร้อยแล้วค่ะ / Account linked."
REPLY_ATTACHED = "เชื่อมแชตนี้กับองค์กรเรียบร้อยแล้วค่ะ / This chat is now connected to your organization."
REPLY_INVALID = "เชื่อมไม่สำเร็จ: รหัสไม่ถูกต้องหรือหมดอายุ / Could not link: the code is invalid or expired."
REPLY_PRIVATE = "กรุณาส่งรหัสในแชตส่วนตัวกับสลิดนะคะ / Please send the code in a private chat."
REPLY_SLOW_DOWN = "ลองหลายครั้งเกินไป กรุณารอสักครู่ / Too many attempts, please wait a few minutes."


def _now() -> datetime:
    return datetime.now(UTC)


def parse_command(text: str) -> str | None:
    """The 8 digits of a ``link ...`` command, or None when the message is anything else."""
    m = _COMMAND.match(text or "")
    if m is None:
        return None
    digits = re.sub(r"\D", "", m.group("code"))
    return digits if len(digits) == CODE_DIGITS else None


def format_code(digits: str) -> str:
    return f"{digits[:4]}-{digits[4:]}"


class CodeError(Exception):
    """A code could not be created or redeemed. `reason` is for logs, never for the user."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class TooManyCodes(CodeError):
    def __init__(self) -> None:
        super().__init__("too many live codes")


class NotAMember(CodeError):
    def __init__(self) -> None:
        super().__init__("not a member of the org")


@dataclass(frozen=True)
class IssuedCode:
    code: str  # "1234-5678"
    kind: str  # identity | attach
    platform: str
    expires_at: str


@dataclass(frozen=True)
class Redeemed:
    kind: str  # identity | attach
    org_id: str
    platform: str
    email: str  # the member (identity) or the admin who issued the code (attach)


class IdentityLinks:
    """Create and redeem the codes. Blocking: call from a worker thread."""

    def __init__(self, orgs: OrgStore, clock=_now):
        self.orgs, self.docs, self._clock = orgs, orgs.docs, clock

    # ---- create
    def create_code(self, org_id: str, email: str, platform: str) -> IssuedCode:
        """A link code for a member of the org. NotAMember / TooManyCodes / CodeError."""
        org = self.orgs.get(org_id)
        if org is None or org.role_of(email) is None:
            raise NotAMember()
        return self._issue("identity", org_id, email.lower(), platform)

    def create_attach_code(self, org_id: str, admin_email: str, platform: str) -> IssuedCode:
        """A code that attaches a bot tenant of `platform` to the org (the web layer has already
        checked that the caller is an owner / admin)."""
        if self.orgs.get(org_id) is None:
            raise CodeError("unknown org")
        return self._issue("attach", org_id, admin_email.lower(), platform)

    def _issue(self, kind: str, org_id: str, email: str, platform: str) -> IssuedCode:
        if platform not in CODE_PLATFORMS:
            raise CodeError("platform takes no code")
        now = self._clock()
        with self.orgs.lock:
            self._prune(now)
            live = self.docs.find(
                KIND,
                0,
                lambda d: (
                    d.get("kind") == kind
                    and d.get("org_id") == org_id
                    and d.get("email") == email
                    and self._live(d, now)
                ),
            )
            if len(live) >= MAX_LIVE_CODES:
                raise TooManyCodes()
            for _ in range(20):
                digits = f"{secrets.randbelow(10**CODE_DIGITS):0{CODE_DIGITS}d}"
                if self.docs.get(KIND, 0, digits) is None:
                    break
            else:  # pragma: no cover - 20 collisions in a row
                raise CodeError("no free code")
            expires = stamp(now + CODE_TTL)
            self.docs.put(
                KIND,
                0,
                digits,
                {
                    "kind": kind,
                    "org_id": org_id,
                    "email": email,
                    "platform": platform,
                    "created_at": stamp(now),
                    "expires_at": expires,
                    "used": False,
                },
            )
        return IssuedCode(format_code(digits), kind, platform, expires)

    @staticmethod
    def _live(d: dict, now: datetime) -> bool:
        return not d.get("used") and datetime.fromisoformat(d["expires_at"]) > now

    def _prune(self, now: datetime) -> None:
        for key, _d in self.docs.find_items(
            KIND,
            0,
            lambda d: datetime.fromisoformat(d["expires_at"]) + PRUNE_AFTER <= now,
        ):
            self.docs.delete(KIND, 0, key)

    # ---- redeem
    def redeem(self, code: str, platform: str, platform_user_id: str, tenant_key: str) -> Redeemed:
        """Apply a code typed by `platform_user_id` to the bot tenant `tenant_key`.

        Every check runs before anything changes; then the code is marked used; then the link /
        attach is applied. CodeError for any refusal (expired, used, unknown, wrong platform,
        another org's tenant, account already linked to someone else, ...)."""
        digits = re.sub(r"\D", "", code or "")
        pid = str(platform_user_id or "").strip()
        tenant = str(tenant_key or "").strip()
        if len(digits) != CODE_DIGITS or not pid or not tenant:
            raise CodeError("malformed")
        now = self._clock()
        with self.orgs.lock:
            doc = self.docs.get(KIND, 0, digits)
            if not doc:
                raise CodeError("unknown code")
            if doc.get("used"):
                raise CodeError("code already used")
            if datetime.fromisoformat(doc["expires_at"]) <= now:
                raise CodeError("code expired")
            if doc["platform"] != platform or platform not in CODE_PLATFORMS:
                raise CodeError("wrong platform")
            org = self.orgs.get(doc["org_id"])
            if org is None:
                raise CodeError("org is gone")
            owner = self.orgs.org_of_workspace(platform, tenant)
            if doc["kind"] == "identity":
                if org.role_of(doc["email"]) is None:
                    raise CodeError("member was removed")
                if owner is not None and owner[0].id != org.id:
                    raise CodeError("tenant belongs to another org")
                holder = self.orgs.member_by_identity(org.id, platform, pid)
                if holder is not None and holder.email.lower() != doc["email"]:
                    raise CodeError("account linked to another member")
            elif owner is not None and owner[0].id != org.id:
                raise CodeError("tenant belongs to another org")
            self.docs.put(KIND, 0, digits, {**doc, "used": True, "used_at": stamp(now)})
            try:
                if doc["kind"] == "identity":
                    self.orgs.link_identity(org.id, doc["email"], platform, pid)
                elif owner is None:
                    self.orgs.attach_workspace(
                        org.id,
                        Workspace(
                            id=workspace_id(platform, tenant),
                            platform=platform,
                            external_id=tenant,
                            name=f"{platform.title()} {tenant}",
                            status="connected",
                            connected_at=stamp(now),
                            connected_by=doc["email"],
                        ),
                    )
            except (IdentityTaken, WorkspaceTaken, KeyError, ValueError) as exc:
                raise CodeError(f"apply failed: {type(exc).__name__}") from exc
        return Redeemed(doc["kind"], org.id, platform, doc["email"])

    # ---- who is this account?
    def identity_of(
        self, platform: str, tenant_key: str, platform_user_id: str
    ) -> tuple[str, str] | None:
        """(org id, member email) when the tenant belongs to an org and this account is a
        VERIFIED member of that org; else None (a guest of the room)."""
        found = self.orgs.org_of_workspace(platform, tenant_key)
        if found is None or found[1].status != "connected":
            return None
        member = self.orgs.member_by_identity(found[0].id, platform, platform_user_id)
        return (found[0].id, member.email.lower()) if member else None

    def auto_link(
        self, platform: str, tenant_key: str, platform_user_id: str, email: str
    ) -> tuple[str, str] | None:
        """Platform-verified email (Slack): link the account to the org member with that email.
        The tenant must belong to an org. Never replaces a different account the member already
        linked, and never takes an account another member holds. Returns the identity."""
        email = (email or "").strip().lower()
        found = self.orgs.org_of_workspace(platform, tenant_key)
        if not email or found is None or found[1].status != "connected":
            return None
        org = found[0]
        if org.role_of(email) is None:
            return None
        with self.orgs.lock:
            org = self.orgs.get(org.id)
            who = next((m for m in org.members if m.email.lower() == email), None) if org else None
            if who is None:
                return None
            current = who.identity(platform)
            if current is None:
                try:
                    self.orgs.link_identity(org.id, who.email, platform, platform_user_id)
                except (IdentityTaken, ValueError):
                    return None
            elif current != str(platform_user_id):
                return None
        return org.id, email
