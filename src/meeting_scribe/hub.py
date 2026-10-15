"""Salid command center: `/hub` page and `/api/hub/*` JSON API.

Auth is stateless and needs no server secret. The page signs the user in with Google Identity
Services and sends the Google ID token as `Authorization: Bearer ...`. Every API call verifies
the token against `HUB_GOOGLE_CLIENT_ID` (public OAuth client id) and requires the verified
email to be in `HUB_ADMIN_EMAILS`. If either setting is empty every call fails closed with 503.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
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
from salid.team_settings import TeamSettings, TeamSettingsStore

logger = logging.getLogger(__name__)

HUB_HTML = Path(__file__).parent / "static" / "hub.html"
CLIENT_ID_RE = re.compile(r"^[\w.-]+$")
ACTIVITY_LIMIT = 20

router = APIRouter()


# ------------------------------------------------------------------ auth
def allowed_emails(settings: Settings) -> set[str]:
    return {e.strip().lower() for e in settings.hub_admin_emails.split(",") if e.strip()}


def google_token_verifier() -> Callable[[str, str], dict[str, Any]]:
    """Dependency returning `verify(token, client_id) -> claims`; tests override this."""

    def verify(token: str, client_id: str) -> dict[str, Any]:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token

        return id_token.verify_oauth2_token(token, google_requests.Request(), audience=client_id)

    return verify


def require_admin(
    authorization: str | None = Header(default=None),
    verify: Callable[[str, str], dict[str, Any]] = Depends(google_token_verifier),
) -> str:
    """Return the verified admin email, or raise 503 (not configured) / 401 / 403."""
    settings = get_settings()
    admins = allowed_emails(settings)
    if not settings.hub_google_client_id or not admins:
        raise HTTPException(503, "Hub is not configured")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    try:
        claims = verify(authorization[7:].strip(), settings.hub_google_client_id)
    except ValueError as exc:
        logger.info("Rejected hub token: %s", exc)
        raise HTTPException(401, "Invalid or expired token") from exc
    email = str(claims.get("email", "")).lower()
    if claims.get("email_verified") is not True or email not in admins:
        raise HTTPException(403, "This account is not allowed")
    return email


# ------------------------------------------------------------------ collaborators (overridable)
@lru_cache
def get_team_store() -> TeamSettingsStore:
    s = get_settings()
    return TeamSettingsStore(
        collection=s.salid_team_collection, team_id=s.salid_team_id, project=s.gcp_project
    )


@lru_cache
def _salid_runtime() -> tuple[Any, Any]:
    from salid.config import Settings as SalidSettings
    from salid.store import make_store

    s = SalidSettings.from_env()  # honours SALID_STORE like the bot and salid.cli
    return s, make_store(s)


def get_salid_runtime() -> tuple[Any, Any]:
    """(salid settings, salid store). Overridden in tests."""
    return _salid_runtime()


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
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------ API
class Me(BaseModel):
    email: str
    invite_url: str | None


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


@router.get("/api/hub/me")
def hub_me(email: str = Depends(require_admin)) -> Me:
    return Me(email=email, invite_url=invite_url() or None)


@router.get("/api/hub/settings")
async def hub_get_settings(
    _: str = Depends(require_admin), store: TeamSettingsStore = Depends(get_team_store)
) -> TeamSettings:
    return await run_in_threadpool(store.get)


@router.put("/api/hub/settings")
async def hub_put_settings(
    body: TeamSettings,
    email: str = Depends(require_admin),
    store: TeamSettingsStore = Depends(get_team_store),
) -> TeamSettings:
    return await run_in_threadpool(store.save, body, email)


def _utc(value: str | datetime) -> datetime:
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def build_activity(salid_store: Any, limit: int = ACTIVITY_LIMIT) -> list[ActivityItem]:
    items: list[tuple[datetime, ActivityItem]] = []
    try:
        for r in salid_store.recent(limit):
            flagged = r.confidence < 0.7 or r.total is None or r.possible_duplicate_of is not None
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
                    ),
                )
            )
    except Exception:
        logger.warning("hub: could not read receipts", exc_info=True)
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


@router.get("/api/hub/activity")
async def hub_activity(
    _: str = Depends(require_admin), runtime: tuple = Depends(get_salid_runtime)
) -> list[ActivityItem]:
    return await run_in_threadpool(build_activity, runtime[1])


@router.post("/api/hub/ask")
async def hub_ask(
    body: AskBody, _: str = Depends(require_admin), runtime: tuple = Depends(get_salid_runtime)
) -> AskReply:
    """Salid's expense Q&A as an executive (hub admins are executives). No deep tasks here."""
    from salid.qa import ask

    s, store = runtime
    try:
        ans = await run_in_threadpool(ask, s, store, body.question.strip(), is_exec=True, user_id=0)
    except Exception as exc:
        logger.exception("hub ask failed")
        raise HTTPException(502, "Salid could not answer right now") from exc
    return AskReply(answer=ans.text)
