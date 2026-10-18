"""Google OAuth 2.0 (web server flow) for one member's Calendar access.

Google login on the web (Firebase) only proves who the member is. Calendar needs its own consent:
`GoogleOAuth.begin` creates a single-use server-side `state` bound to (org, member email, PKCE
verifier, 10 minute expiry); `auth_url` turns it into the consent URL; the browser comes back to
the redirect URI and `exchange` redeems code + state.

Tokens: only the REFRESH token is stored, Fernet-encrypted (`SALID_TOKEN_KEY`) in the DocStore kind
`salid_google_tokens`, key `{org_id}:{email}`. The access token lives in process memory until it
expires. Nothing secret is ever logged or put into an exception message. The Google account that
consents must be the member's own email (checked with userinfo), otherwise the grant is revoked.

The feature is off (`GoogleConfig.enabled()` is False) unless client id, client secret, redirect
URI and a valid Fernet key are all set.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

EVENTS_SCOPE = "https://www.googleapis.com/auth/calendar.events.owned"
FREEBUSY_SCOPE = "https://www.googleapis.com/auth/calendar.events.freebusy"
CALENDAR_SCOPES = (EVENTS_SCOPE, FREEBUSY_SCOPE)
# openid + email only so userinfo can prove WHICH Google account consented (non-sensitive scopes).
REQUEST_SCOPES = ("openid", "email", *CALENDAR_SCOPES)
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"

TOKENS_KIND = "salid_google_tokens"
STATE_KIND = "salid_google_state"
STATE_TTL = timedelta(minutes=10)
EXPIRY_SKEW_S = 60


class GoogleOAuthError(Exception):
    """`code` is a short machine-readable reason; the message never contains tokens or bodies."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class NotConnected(GoogleOAuthError):
    """The member has no (valid) Google Calendar connection; they must connect again."""

    def __init__(self) -> None:
        super().__init__("not_connected")


@dataclass(frozen=True)
class GoogleConfig:
    client_id: str = ""
    client_secret: str = field(default="", repr=False)
    redirect_uri: str = ""
    token_key: str = field(default="", repr=False)

    @classmethod
    def from_settings(cls, st: Any) -> GoogleConfig:
        return cls(
            client_id=getattr(st, "google_oauth_client_id", "") or "",
            client_secret=getattr(st, "google_oauth_client_secret", "") or "",
            redirect_uri=getattr(st, "google_oauth_redirect_uri", "") or "",
            token_key=getattr(st, "salid_token_key", "") or "",
        )

    def enabled(self) -> bool:
        if not (self.client_id and self.client_secret and self.redirect_uri and self.token_key):
            return False
        try:
            Fernet(self.token_key.encode())
        except (ValueError, TypeError):
            logger.warning("SALID_TOKEN_KEY is not a valid Fernet key; Google Calendar is off")
            return False
        return True


@dataclass(frozen=True)
class Account:
    """Whose Google Calendar: a member of an org, identified by org id and lower-case email."""

    org_id: str
    email: str


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def pkce_challenge(verifier: str) -> str:
    return _b64url(hashlib.sha256(verifier.encode()).digest())


def _key(org_id: str, email: str) -> str:
    return f"{org_id}:{email.strip().lower()}"


class GoogleOAuth:
    def __init__(
        self,
        docs: Any,
        cfg: GoogleConfig,
        client: httpx.Client | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        mono: Callable[[], float] = time.monotonic,
    ) -> None:
        self.docs, self.cfg = docs, cfg
        self._http = client or httpx.Client(timeout=15)
        self._clock, self._mono = clock, mono
        self._cache: dict[str, tuple[str, float]] = {}  # key -> (access token, mono expiry)

    def enabled(self) -> bool:
        return self.cfg.enabled()

    # ------------------------------------------------------------------ consent
    def begin(self, org_id: str, email: str) -> str:
        """A fresh single-use state for this member; store the PKCE verifier server-side."""
        if not self.enabled():
            raise GoogleOAuthError("not_configured")
        state = secrets.token_urlsafe(32)
        exp = (self._clock() + STATE_TTL).isoformat(timespec="seconds")
        self.docs.put(
            STATE_KIND,
            0,
            state,
            {
                "org_id": org_id,
                "email": email.strip().lower(),
                "verifier": secrets.token_urlsafe(48),
                "exp": exp,
            },
        )
        return state

    def auth_url(self, state: str) -> str:
        """The Google consent URL for a state made by `begin` (does not consume it)."""
        rec = self.docs.get(STATE_KIND, 0, state)
        if not rec:
            raise GoogleOAuthError("bad_state")
        params = {
            "client_id": self.cfg.client_id,
            "redirect_uri": self.cfg.redirect_uri,
            "response_type": "code",
            "scope": " ".join(REQUEST_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
            "include_granted_scopes": "true",
            "state": state,
            "code_challenge": pkce_challenge(rec["verifier"]),
            "code_challenge_method": "S256",
            "login_hint": rec["email"],
        }
        return f"{AUTH_URL}?{urlencode(params)}"

    def start(self, org_id: str, email: str) -> str:
        return self.auth_url(self.begin(org_id, email))

    def _consume_state(self, state: str) -> dict:
        def fn(tx: Any) -> dict | None:
            rec = tx.get(STATE_KIND, 0, state)
            if rec:
                tx.delete(STATE_KIND, 0, state)  # single use, even when it turns out expired
            return rec

        rec = self.docs.transact(fn) if state else None
        if not rec:
            raise GoogleOAuthError("bad_state")
        if datetime.fromisoformat(rec["exp"]) < self._clock():
            raise GoogleOAuthError("expired")
        return rec

    def exchange(self, code: str, state: str) -> Account:
        """Redeem the consent result. Stores the encrypted refresh token and returns whose it is."""
        if not self.enabled():
            raise GoogleOAuthError("not_configured")
        rec = self._consume_state(state)
        if not code:
            raise GoogleOAuthError("exchange_failed")
        tok = self._token_request(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.cfg.redirect_uri,
                "code_verifier": rec["verifier"],
            }
        )
        access, refresh = tok.get("access_token", ""), tok.get("refresh_token", "")
        if not access:
            raise GoogleOAuthError("exchange_failed")
        email = rec["email"]
        try:
            who = self._get(USERINFO_URL, access)
        except GoogleOAuthError:
            self._revoke_token(refresh or access)
            raise
        if who.get("email_verified") is not True or str(who.get("email", "")).lower() != email:
            self._revoke_token(refresh or access)
            raise GoogleOAuthError("email_mismatch")
        granted = set(str(tok.get("scope", "")).split())
        if not set(CALENDAR_SCOPES) <= granted:
            self._revoke_token(refresh or access)
            raise GoogleOAuthError("scope_missing")
        if not refresh:
            self._revoke_token(access)
            raise GoogleOAuthError("no_refresh_token")
        self._store(rec["org_id"], email, refresh, str(tok.get("scope", "")))
        self._remember(rec["org_id"], email, access, tok.get("expires_in"))
        return Account(rec["org_id"], email)

    # ------------------------------------------------------------------ tokens
    def _fernet(self) -> Fernet:
        return Fernet(self.cfg.token_key.encode())

    def _store(self, org_id: str, email: str, refresh: str, scope: str) -> None:
        # org and email travel inside the ciphertext so a copied record cannot be replayed for
        # another member.
        blob = json.dumps({"o": org_id, "e": email, "rt": refresh}).encode()
        self.docs.put(
            TOKENS_KIND,
            0,
            _key(org_id, email),
            {
                "org_id": org_id,
                "email": email,
                "enc": self._fernet().encrypt(blob).decode(),
                "scope": scope,
                "connected_at": self._clock().isoformat(timespec="seconds"),
            },
        )

    def _refresh_token(self, org_id: str, email: str) -> str:
        rec = self.docs.get(TOKENS_KIND, 0, _key(org_id, email))
        if not rec:
            raise NotConnected()
        try:
            data = json.loads(self._fernet().decrypt(rec["enc"].encode()))
        except (InvalidToken, ValueError, KeyError):
            logger.warning("google token record unreadable; dropping it")
            self.docs.delete(TOKENS_KIND, 0, _key(org_id, email))
            raise NotConnected() from None
        if data.get("o") != org_id or data.get("e") != email.strip().lower():
            raise NotConnected()
        return str(data["rt"])

    def status(self, org_id: str, email: str) -> dict:
        rec = self.docs.get(TOKENS_KIND, 0, _key(org_id, email))
        if not rec:
            return {"connected": False, "email": None, "connected_at": None}
        return {"connected": True, "email": rec["email"], "connected_at": rec.get("connected_at")}

    def is_connected(self, org_id: str, email: str) -> bool:
        return bool(self.docs.get(TOKENS_KIND, 0, _key(org_id, email)))

    def _remember(self, org_id: str, email: str, access: str, expires_in: Any) -> None:
        try:
            ttl = max(0, int(expires_in) - EXPIRY_SKEW_S)
        except (TypeError, ValueError):
            ttl = 0
        self._cache[_key(org_id, email)] = (access, self._mono() + ttl)

    def access_token(self, org_id: str, email: str) -> str:
        """A valid access token: cached until shortly before expiry, otherwise refreshed."""
        hit = self._cache.get(_key(org_id, email))
        if hit and hit[1] > self._mono():
            return hit[0]
        return self.refresh(Account(org_id, email.strip().lower()))

    def evict(self, org_id: str, email: str) -> None:
        """Forget the cached access token (e.g. Google answered 401); the next use refreshes."""
        self._cache.pop(_key(org_id, email), None)

    def refresh(self, member: Account) -> str:
        """Force a refresh. `invalid_grant` (revoked / expired consent) deletes the record."""
        rt = self._refresh_token(member.org_id, member.email)
        try:
            tok = self._token_request({"grant_type": "refresh_token", "refresh_token": rt})
        except GoogleOAuthError as e:
            if e.code == "invalid_grant":
                self.docs.delete(TOKENS_KIND, 0, _key(member.org_id, member.email))
                self._cache.pop(_key(member.org_id, member.email), None)
                raise NotConnected() from None
            raise
        access = tok.get("access_token", "")
        if not access:
            raise GoogleOAuthError("refresh_failed")
        self._remember(member.org_id, member.email, access, tok.get("expires_in"))
        return access

    def revoke(self, member: Account) -> None:
        """Revoke at Google (best effort) and always delete our copy."""
        with contextlib.suppress(NotConnected):
            self._revoke_token(self._refresh_token(member.org_id, member.email))
        self.docs.delete(TOKENS_KIND, 0, _key(member.org_id, member.email))
        self._cache.pop(_key(member.org_id, member.email), None)

    # ------------------------------------------------------------------ http
    def _token_request(self, form: dict) -> dict:
        body = {"client_id": self.cfg.client_id, "client_secret": self.cfg.client_secret, **form}
        try:
            r = self._http.post(TOKEN_URL, data=body)
        except httpx.HTTPError as e:
            raise GoogleOAuthError("network") from e
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError as e:
                raise GoogleOAuthError("exchange_failed") from e
        err = ""
        with contextlib.suppress(ValueError, AttributeError):
            err = str(r.json().get("error", ""))
        logger.info("google token endpoint answered %s %s", r.status_code, err)
        raise GoogleOAuthError(
            "invalid_grant" if err == "invalid_grant" else f"http_{r.status_code}"
        )

    def _get(self, url: str, access: str) -> dict:
        try:
            r = self._http.get(url, headers={"Authorization": f"Bearer {access}"})
        except httpx.HTTPError as e:
            raise GoogleOAuthError("network") from e
        if r.status_code != 200:
            raise GoogleOAuthError(f"http_{r.status_code}")
        try:
            return r.json()
        except ValueError as e:
            raise GoogleOAuthError("exchange_failed") from e

    def _revoke_token(self, token: str) -> None:
        if not token:
            return
        try:
            self._http.post(REVOKE_URL, data={"token": token})
        except httpx.HTTPError:
            logger.info("google revoke request failed")
