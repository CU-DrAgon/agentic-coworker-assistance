"""Salid command center: `/hub` page and `/api/hub/*` JSON API (multi-guild).

Auth is stateless and needs no server secret. The page signs the user in with Google Identity
Services and sends the Google ID token as `Authorization: Bearer ...`. Every API call verifies
the token against `HUB_GOOGLE_CLIENT_ID` (public OAuth client id); any verified Google account
may sign in. Which guilds (Discord servers) it may manage is decided by the guild's
`admin_emails` list in Firestore, filled in by the one-time `/ตั้งค่า` link flow
(`POST /api/hub/link`). `HUB_ADMIN_EMAILS` is an optional list of super admins who see every
guild. If `HUB_GOOGLE_CLIENT_ID` is empty every call fails closed with 503.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from meeting_scribe import pipeline
from meeting_scribe.config import Settings, get_settings
from salid.guild_settings import (
    GuildSettings,
    GuildSettingsUpdate,
    GuildStore,
    LinkError,
)

logger = logging.getLogger(__name__)

HUB_HTML = Path(__file__).parent / "static" / "hub.html"
CLIENT_ID_RE = re.compile(r"^[\w.-]+$")
ACTIVITY_LIMIT = 20

router = APIRouter()


# ------------------------------------------------------------------ auth
def super_admins(settings: Settings) -> set[str]:
    return {e.strip().lower() for e in settings.hub_admin_emails.split(",") if e.strip()}


def google_token_verifier() -> Callable[[str, str], dict[str, Any]]:
    """Dependency returning `verify(token, client_id) -> claims`; tests override this."""

    def verify(token: str, client_id: str) -> dict[str, Any]:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token

        return id_token.verify_oauth2_token(token, google_requests.Request(), audience=client_id)

    return verify


# ------------------------------------------------------------------ Salid sessions
# A Google ID token lives one hour. After the first sign-in the pages exchange it for a Salid
# session token ("ss1.<payload>.<mac>", HMAC-SHA256 with SALID_SESSION_SECRET, 30 days) and keep
# that in the browser, so nobody is signed out every hour or on every reload.
SESSION_PREFIX = "ss1."


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def issue_session(
    email: str, secret: str, days: int, now: datetime | None = None
) -> tuple[str, str]:
    """(token, expires_at ISO) for `email`."""
    exp = (now or datetime.now(UTC)) + timedelta(days=days)
    payload = _b64(json.dumps({"e": email.lower(), "x": int(exp.timestamp())}).encode())
    mac = _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
    return f"{SESSION_PREFIX}{payload}.{mac}", exp.isoformat(timespec="seconds")


def verify_session(token: str, secret: str, now: datetime | None = None) -> str:
    """The email a session token stands for. ValueError when forged, malformed or expired."""
    if not secret or not token.startswith(SESSION_PREFIX):
        raise ValueError("not a session token")
    try:
        payload, mac = token[len(SESSION_PREFIX) :].split(".", 1)
    except ValueError as exc:
        raise ValueError("malformed session") from exc
    want = _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(want, mac):
        raise ValueError("bad session signature")
    data = json.loads(_unb64(payload))
    if int(data.get("x", 0)) <= int((now or datetime.now(UTC)).timestamp()):
        raise ValueError("session expired")
    return str(data["e"]).lower()


@lru_cache
def _prefs() -> Any:
    from salid.orgs import make_org_store

    return make_org_store(*get_salid_runtime())


def get_prefs() -> Any:
    """Per-person web preferences (`OrgStore.user_language` / `set_user_language`). Overridden in tests."""
    return _prefs()


@lru_cache
def _doc_store() -> Any:
    from salid.coo_store import make_doc_store

    return make_doc_store(*get_salid_runtime())


def get_doc_store() -> Any:
    """The bot's DocStore (the action queue lives there). Overridden in tests."""
    return _doc_store()


def language_change_notice(before: str, after: str, who: str) -> str:
    """What Salid posts on the server when the company language changes: the line in the new
    language, then in the old one (so everybody understands the switch)."""
    from salid.i18n import lang_name, t

    blocks: list[str] = []
    for lang in (after, before):
        render = lang if lang in ("th", "en") or lang else "en"
        line = t(
            "company_language_changed",
            render,
            lname=lang_name(after, 2 if render == "th" else 1),
            who=who,
        )
        if line not in blocks:
            blocks.append(line)
    return "\n".join(blocks)


def announce_language_change(docs: Any, guild_id: int, before: str, after: str, who: str) -> None:
    """Queue the notice for the bot (`salid.actions`, kind `announce`: the ask channel, else
    general). Best effort: a queue problem never fails the settings save."""
    from salid.actions import ActionQueue

    try:
        ActionQueue(docs).enqueue(
            guild_id, "announce", {"text": language_change_notice(before, after, who)}, by=who
        )
    except Exception:
        logger.exception("could not queue the language notice for guild %s", guild_id)


def require_user(
    authorization: str | None = Header(default=None),
    verify: Callable[[str, str], dict[str, Any]] = Depends(google_token_verifier),
    prefs: Any = Depends(get_prefs),
) -> str:
    """Return the verified email (Salid session or Google ID token), or raise 503 / 401 / 403."""
    if not authorization or not authorization.lower().startswith("bearer "):
        if not get_settings().hub_google_client_id:
            raise HTTPException(503, "Hub is not configured")
        raise HTTPException(401, "Missing bearer token")
    return user_from_token(authorization[7:].strip(), verify, prefs)


def account_for_google(email: str, prefs: Any) -> str:
    """A Google sign-in lands in the Salid account that LINKED that Google account (someone who
    signed up through Discord and added Google later), else in the account of that email.
    Fail-soft: a broken preference store never blocks sign-in."""
    if prefs is None:
        return email
    try:
        return str(prefs.account_for_google(email) or email).lower()
    except Exception:
        logger.exception("could not resolve the Google account of %s", email)
        return email


def user_from_token(
    raw: str, verify: Callable[[str, str], dict[str, Any]], prefs: Any = None
) -> str:
    """The email behind a raw session / Google token (same rules as `require_user`); for places
    that cannot use the Authorization header, like WebSockets (pass `get_prefs()` there)."""
    settings = get_settings()
    if not settings.hub_google_client_id:
        raise HTTPException(503, "Hub is not configured")
    if not raw:
        raise HTTPException(401, "Missing bearer token")
    if raw.startswith(SESSION_PREFIX):
        try:
            return verify_session(raw, settings.salid_session_secret)
        except ValueError as exc:
            logger.info("Rejected session: %s", exc)
            raise HTTPException(401, "Invalid or expired session") from exc
    try:
        claims = verify(raw, settings.hub_google_client_id)
    except ValueError as exc:
        logger.info("Rejected hub token: %s", exc)
        raise HTTPException(401, "Invalid or expired token") from exc
    if claims.get("email_verified") is not True or not claims.get("email"):
        raise HTTPException(403, "This account has no verified email")
    return account_for_google(str(claims["email"]).lower(), prefs)


def is_super_admin(email: str) -> bool:
    return email in super_admins(get_settings())


# ------------------------------------------------------------------ collaborators (overridable)
@lru_cache
def get_guild_store() -> GuildStore:
    s = get_settings()
    return GuildStore(collection=s.salid_guild_collection, project=s.gcp_project)


@lru_cache
def _salid_runtime() -> tuple[Any, Any]:
    from salid.config import Settings as SalidSettings
    from salid.store import make_store

    s = SalidSettings.from_env()  # honours SALID_STORE like the bot and salid.cli
    return s, make_store(s)


def get_salid_runtime() -> tuple[Any, Any]:
    """(salid settings, salid store). Overridden in tests."""
    return _salid_runtime()


def get_memory_store() -> Any:
    """The bot's memory store (same SALID_STORE settings). Overridden in tests."""
    return _memory_store()


WEB_MEMORY_REFRESH_S = 120.0  # the bot writes summaries; the web re-reads them in the background


@lru_cache
def _memory_store() -> Any:
    """Day summaries are served from an in-process index (one Firestore read per guild, then
    memory), refreshed in the background every two minutes, so a web or voice question does not
    re-read every summary of the company (that took ~4.5 s per question)."""
    from salid.memory_index import IndexedMemoryStore
    from salid.memory_store import make_memory_store

    s, store = get_salid_runtime()
    return IndexedMemoryStore(make_memory_store(s, store), refresh_s=WEB_MEMORY_REFRESH_S)


@dataclass
class GuildAccess:
    guild_id: int
    email: str
    is_super: bool
    settings: GuildSettings


async def guild_admin(
    guild_id: int,
    email: str = Depends(require_user),
    guilds: GuildStore = Depends(get_guild_store),
) -> GuildAccess:
    """403 unless the user administers this guild (or is a super admin; 404 for unknown guilds)."""
    settings = await run_in_threadpool(guilds.find, guild_id)
    if settings is None:
        raise HTTPException(404 if is_super_admin(email) else 403, "No access to this guild")
    if not is_super_admin(email) and email not in settings.admin_emails:
        raise HTTPException(403, "No access to this guild")
    return GuildAccess(guild_id, email, is_super_admin(email), settings)


# ------------------------------------------------------------------ page
def invite_url() -> str:
    from meeting_scribe.app import salid_invite_url

    return salid_invite_url(get_settings().salid_discord_client_id)


@router.get("/hub", include_in_schema=False)
def hub_page() -> HTMLResponse:
    client_id = get_settings().hub_google_client_id
    if not CLIENT_ID_RE.match(client_id):
        client_id = ""
    html = HUB_HTML.read_text(encoding="utf-8").replace("__HUB_GOOGLE_CLIENT_ID__", client_id)
    return HTMLResponse(
        html, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}
    )


# ------------------------------------------------------------------ API
class Me(BaseModel):
    email: str
    invite_url: str | None
    is_super: bool
    language: str | None = None  # saved web UI language (th / en); None = not chosen yet


class LanguageBody(BaseModel):
    language: str


class GuildInfo(BaseModel):
    id: str  # a string: Discord ids exceed JavaScript's safe integers
    name: str
    icon_url: str
    active: bool


class LinkBody(BaseModel):
    code: str = Field(min_length=10, max_length=200)


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class AskReply(BaseModel):
    answer: str


class ActivityItem(BaseModel):
    kind: str  # receipt | meeting
    title: str
    at: str  # ISO 8601, UTC
    status: str  # receipt: ok | review; meeting: the meeting status
    amount: float | None = None
    currency: str | None = None  # ISO code of `amount`; missing counts as THB


class SessionView(BaseModel):
    token: str
    expires_at: str
    email: str


@router.post("/api/hub/session")
def hub_session(
    email: str = Depends(require_user),
    authorization: str | None = Header(default=None),
    verify: Callable[[str, str], dict[str, Any]] = Depends(google_token_verifier),
    prefs: Any = Depends(get_prefs),
) -> SessionView:
    """Exchange a fresh Google sign-in (or a still-valid session) for a Salid session token that
    every page keeps in the browser. 503 when SALID_SESSION_SECRET is not set. A Google token
    also fills the person's real first / last name (once) from the account."""
    s = get_settings()
    if not s.salid_session_secret:
        raise HTTPException(503, "Sessions are not configured")
    raw = (authorization or "")[7:].strip()
    if raw and not raw.startswith(SESSION_PREFIX):
        try:
            claims = verify(raw, s.hub_google_client_id)
            p = prefs.user_profile(email)
            first = str(claims.get("given_name") or "").strip()
            last = str(claims.get("family_name") or "").strip()
            fields = {
                "first_name": first if first and not p.first_name else None,
                "last_name": last if last and not p.last_name else None,
                "name": (first or str(claims.get("name") or "").strip()) if not p.name else None,
            }
            if any(v for v in fields.values()):
                prefs.set_user_profile(email, **fields)
        except Exception:  # names are a convenience: never block sign-in on them
            logger.exception("could not record the Google name of %s", email)
    token, exp = issue_session(email, s.salid_session_secret, s.salid_session_days)
    return SessionView(token=token, expires_at=exp, email=email)


@router.get("/api/hub/me")
async def hub_me(email: str = Depends(require_user), prefs: Any = Depends(get_prefs)) -> Me:
    try:  # the saved language is a convenience: never block sign-in on it
        language = await run_in_threadpool(prefs.user_language, email)
    except Exception:
        logger.exception("could not read the language preference of %s", email)
        language = None
    return Me(
        email=email,
        invite_url=invite_url() or None,
        is_super=is_super_admin(email),
        language=language,
    )


@router.put("/api/hub/me/language")
async def hub_set_language(
    body: LanguageBody, email: str = Depends(require_user), prefs: Any = Depends(get_prefs)
) -> dict[str, str]:
    """Remember the language this person wants every Salid page in (shared with /center)."""
    try:
        lang = await run_in_threadpool(prefs.set_user_language, email, body.language)
    except ValueError as exc:
        raise HTTPException(400, "Unsupported language (th or en)") from exc
    return {"language": lang}


def _info(guild_id: int, g: GuildSettings) -> GuildInfo:
    return GuildInfo(
        id=str(guild_id), name=g.guild_name or str(guild_id), icon_url=g.icon_url, active=g.active
    )


@router.get("/api/hub/guilds")
async def hub_guilds(
    email: str = Depends(require_user), guilds: GuildStore = Depends(get_guild_store)
) -> list[GuildInfo]:
    found = await run_in_threadpool(guilds.list_guilds, None if is_super_admin(email) else email)
    return [_info(gid, g) for gid, g in found]


@router.post("/api/hub/link")
async def hub_link(
    body: LinkBody,
    email: str = Depends(require_user),
    guilds: GuildStore = Depends(get_guild_store),
) -> GuildInfo:
    """Redeem a `/ตั้งค่า` code: the signed-in Google account becomes an admin of that guild."""
    try:
        gid, g = await run_in_threadpool(guilds.redeem_link_code, body.code.strip(), email)
    except LinkError as exc:
        raise HTTPException(400, "Link code is invalid, expired or already used") from exc
    logger.info("hub: linked a Google account to guild %s", gid)
    return _info(gid, g)


@router.get("/api/hub/guilds/{guild_id}/settings")
async def hub_get_settings(access: GuildAccess = Depends(guild_admin)) -> GuildSettings:
    return access.settings


@router.get("/api/hub/guilds/{guild_id}/memory-status")
async def hub_memory_status(
    access: GuildAccess = Depends(guild_admin),
    memory: Any = Depends(get_memory_store),
    runtime: tuple[Any, Any] = Depends(get_salid_runtime),
) -> dict:
    """History-import progress for the memory card: channels, messages, historical images."""
    from salid.memory_import import import_progress

    s, _store = runtime
    cap = getattr(s, "memory_import_images_per_day", 300)
    today = datetime.now(UTC).astimezone(getattr(s, "zone", UTC)).date().isoformat()
    return await run_in_threadpool(import_progress, memory, access.guild_id, cap, today)


@router.put("/api/hub/guilds/{guild_id}/settings")
async def hub_put_settings(
    body: GuildSettingsUpdate,
    access: GuildAccess = Depends(guild_admin),
    guilds: GuildStore = Depends(get_guild_store),
    docs: Any = Depends(get_doc_store),
) -> GuildSettings:
    """Save the server settings. A change of the COMPANY language is announced on the server by
    Salid (queued for the bot), so nobody is surprised by her switching languages."""
    known = {c.id for c in access.settings.channels}
    chosen = {
        *body.receipt_channel_ids,
        *body.ask_channel_ids,
        *body.channel_modes,
        *body.channel_languages,
        *body.channel_tones,
        body.digest_channel_id,
        body.receipt_home_channel_id,
        body.report_channel_id,
    } - {""}
    if known and chosen - known:  # the bot syncs the list; an empty list means "not synced yet"
        raise HTTPException(422, "Unknown channel id for this guild")
    before = access.settings.language_setting
    saved = await run_in_threadpool(guilds.save_settings, access.guild_id, body, access.email)
    after = saved.language_setting
    if after != before:
        await run_in_threadpool(
            announce_language_change, docs, access.guild_id, before, after, access.email
        )
    return saved


def _utc(value: str | datetime) -> datetime:
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def build_activity(
    salid_store: Any, guild_id: int, include_meetings: bool, limit: int = ACTIVITY_LIMIT
) -> list[ActivityItem]:
    items: list[tuple[datetime, ActivityItem]] = []
    try:
        from salid.store import duplicate_extras

        dup_ids = {r.id for r in duplicate_extras(salid_store.duplicate_groups(guild_id))}
        for r in salid_store.recent(guild_id, limit):  # void receipts are excluded
            flagged = (
                r.confidence < 0.7
                or r.total is None
                or r.possible_duplicate_of is not None
                or r.id in dup_ids
            )
            at = _utc(r.created_at or r.effective_date)
            items.append(
                (
                    at,
                    ActivityItem(
                        kind="receipt",
                        title=r.vendor or "",
                        at=at.isoformat(timespec="seconds"),
                        status="review" if flagged else "ok",
                        amount=r.total,
                        currency=(r.currency or "THB").upper(),
                    ),
                )
            )
    except Exception:
        logger.warning("hub: could not read receipts", exc_info=True)
    # Meetings are not tied to a guild yet, so only super admins (who see everything) get them.
    # Showing them to guild admins would leak one team's meetings to another.
    if include_meetings:
        try:
            for m in pipeline.get_store().list_recent(limit):
                at = _utc(m.created_at)
                items.append(
                    (
                        at,
                        ActivityItem(
                            kind="meeting",
                            title=(m.summary.title if m.summary else m.title),
                            at=at.isoformat(timespec="seconds"),
                            status=m.status.value,
                        ),
                    )
                )
        except Exception:
            logger.warning("hub: could not read meetings", exc_info=True)
    items.sort(key=lambda t: t[0], reverse=True)
    return [i for _, i in items[:limit]]


@router.get("/api/hub/guilds/{guild_id}/activity")
async def hub_activity(
    access: GuildAccess = Depends(guild_admin), runtime: tuple = Depends(get_salid_runtime)
) -> list[ActivityItem]:
    return await run_in_threadpool(build_activity, runtime[1], access.guild_id, access.is_super)


@router.post("/api/hub/guilds/{guild_id}/ask")
async def hub_ask(
    body: AskBody,
    access: GuildAccess = Depends(guild_admin),
    runtime: tuple = Depends(get_salid_runtime),
) -> AskReply:
    """Salid's expense Q&A as an executive of this guild (hub admins are). No deep tasks here."""
    from salid.qa import ask

    s, store = runtime
    try:
        ans = await run_in_threadpool(
            lambda: ask(
                s, store, body.question.strip(), guild_id=access.guild_id, is_exec=True, user_id=0
            )
        )
    except Exception as exc:
        logger.exception("hub ask failed")
        raise HTTPException(502, "Salid could not answer right now") from exc
    return AskReply(answer=ans.text)
