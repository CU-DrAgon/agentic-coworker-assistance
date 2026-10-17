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

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
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


def require_user(
    authorization: str | None = Header(default=None),
    verify: Callable[[str, str], dict[str, Any]] = Depends(google_token_verifier),
) -> str:
    """Return the verified Google email, or raise 503 (not configured) / 401 / 403."""
    settings = get_settings()
    if not settings.hub_google_client_id:
        raise HTTPException(503, "Hub is not configured")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    try:
        claims = verify(authorization[7:].strip(), settings.hub_google_client_id)
    except ValueError as exc:
        logger.info("Rejected hub token: %s", exc)
        raise HTTPException(401, "Invalid or expired token") from exc
    if claims.get("email_verified") is not True or not claims.get("email"):
        raise HTTPException(403, "This account has no verified email")
    return str(claims["email"]).lower()


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


@lru_cache
def _memory_store() -> Any:
    from salid.memory_store import make_memory_store

    s, store = get_salid_runtime()
    return make_memory_store(s, store)


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


@router.get("/api/hub/me")
def hub_me(email: str = Depends(require_user)) -> Me:
    return Me(email=email, invite_url=invite_url() or None, is_super=is_super_admin(email))


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
) -> GuildSettings:
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
    return await run_in_threadpool(guilds.save_settings, access.guild_id, body, access.email)


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
