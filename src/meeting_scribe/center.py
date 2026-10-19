"""Salid command center API: `/center` page and `/api/center/*` (organizations, workspaces,
sharing, license, documents library, web chat, connectors).

Auth is the hub's: a Google ID token as Bearer (`hub.require_user`). Roles come from the
organization's member list (owner / admin / member); `HUB_ADMIN_EMAILS` are super admins who see
and manage every org and set licenses. The contract with request / response examples is written to `command-center-api.md`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import secrets
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from meeting_scribe import hub
from meeting_scribe.config import get_settings
from salid import monitor
from salid.actions import ActionQueue
from salid.change_outbox import ChangeOutbox, record_org_edit
from salid.connectors import (
    COMING_SOON,
    ClickUpConnector,
    ConnectorConfig,
    ConnectorError,
    DiscordConnector,
    SlackConnector,
)
from salid.connectors.base import TOKENS
from salid.guild_settings import LinkError
from salid.identity_links import (
    CODE_PLATFORMS,
    CodeError,
    IdentityLinks,
    IssuedCode,
    NotAMember,
    TooManyCodes,
)
from salid.library import (
    CATEGORIES as LIB_CATEGORIES,
)
from salid.library import (
    DocItem,
    Library,
    Viewer,
)
from salid.licensing import days_left, license_end, license_ok
from salid.monitor_sources import adapt_action, adapt_change
from salid.orgs import (
    CATEGORIES,
    HANDOVER_UNDO_HOURS,
    IDENTITY_PLATFORMS,
    INVITE_DAYS,
    INVITE_MAX_DAYS,
    INVITE_MAX_USES,
    PLANS,
    ROLES,
    STATUSES,
    CompanyProfile,
    Handover,
    Identity,
    Invite,
    InviteError,
    License,
    LicenseRequest,
    Member,
    Org,
    OrgStore,
    Position,
    ShareLink,
    SigninTaken,
    UserProfile,
    Workspace,
    WorkspaceTaken,
    stamp,
    trial_license,
    workspace_id,
)
from salid.people_directory import Observed, PeopleBook, Person, Seen
from salid.sharing import SHARED_DOCS_CHANNEL, SHARED_DOCS_NAME, WEB_DOCS_CHANNEL

logger = logging.getLogger(__name__)

CENTER_HTML = Path(__file__).parent / "static" / "center.html"
MAX_ORGS_PER_USER = 5
CHAT_PER_HOUR = 30
CHAT_HISTORY_PROMPT_TURNS = 6
STATE_KIND = "salid_oauth_state"
STATE_TTL = timedelta(minutes=10)

router = APIRouter(prefix="/api/center")
events_router = APIRouter()


# ------------------------------------------------------------------ collaborators (overridable)
@lru_cache
def _org_store() -> OrgStore:
    from salid.orgs import make_org_store

    s, store = hub.get_salid_runtime()
    return make_org_store(s, store)


def get_orgs() -> OrgStore:
    return _org_store()


@lru_cache
def _library() -> Library:
    from salid.coo_store import make_doc_store
    from salid.library import make_analyzer, make_blobs

    s, store = hub.get_salid_runtime()
    lazy: dict[str, Any] = {}

    def analyze(data: bytes, filename: str, mime: str) -> dict:
        if "fn" not in lazy:
            from salid.llm import make_client

            lazy["fn"] = make_analyzer(make_client(s), s.gate_model, read_model=s.gemini_model)
        return lazy["fn"](data, filename, mime)

    return Library(
        docs=make_doc_store(s, store),
        ledger=store,
        memory=hub.get_memory_store(),
        guilds=hub.get_guild_store(),
        settings=s,
        blobs=make_blobs(s),
        analyzer=analyze,
    )


def get_library() -> Library:
    return _library()


def get_ask() -> Callable[..., Any]:
    """`qa.ask`; tests replace it with a scripted function."""
    from salid.qa import ask

    return ask


def connector_config() -> ConnectorConfig:
    st = get_settings()
    try:
        base = hub.get_salid_runtime()[0].hub_url
    except Exception:
        base = ""
    return ConnectorConfig(
        base_url=base,
        slack_client_id=st.slack_client_id,
        slack_client_secret=st.slack_client_secret,
        slack_signing_secret=st.slack_signing_secret,
        clickup_client_id=st.clickup_client_id,
        clickup_client_secret=st.clickup_client_secret,
        discord_client_id=st.salid_discord_client_id,
    )


def get_connectors(
    orgs: OrgStore = Depends(get_orgs), lib: Library = Depends(get_library)
) -> dict[str, Any]:
    cfg = connector_config()
    return {
        "discord": DiscordConnector(cfg),
        "slack": SlackConnector(cfg, memory=lib.memory, docs=orgs.docs, zone=lib.s.zone),
        "clickup": ClickUpConnector(cfg),
    }


# ------------------------------------------------------------------ views
class LicenseView(BaseModel):
    plan: str
    status: str
    seats: int
    valid_until: str | None
    notes: str
    billing_provider: str
    active: bool  # the bot serves this org right now (status + date + 3 days of grace)
    days_left: int | None
    grace_until: str | None


class IdentityView(BaseModel):
    platform: str  # google (the sign-in, always verified) | discord | slack | clickup | line
    external_id: str
    name: str = ""
    verified: bool = False
    linked_at: str = ""


def identity_views(p: UserProfile) -> list[IdentityView]:
    """Who this person is on each platform. Google comes first: the account's own email (the
    sign-in itself, always verified), then any Google account linked later (someone who signed
    up through Discord), then the platforms."""
    google = [IdentityView(platform="google", external_id=p.email, name=p.name, verified=True)]
    google += [
        IdentityView(**i.model_dump())
        for i in p.identities
        if i.platform == "google" and i.external_id != p.email
    ]
    return google + [IdentityView(**i.model_dump()) for i in p.identities if i.platform != "google"]


class MemberView(BaseModel):
    email: str
    role: str
    discord_user_id: str | None = None
    reports_to: str | None = None  # manager's email in this org (the org chart)
    name: str = ""
    first_name: str = ""
    last_name: str = ""
    full_name: str = ""
    title: str = ""
    department: str = ""
    about: str = ""
    job_description: str = ""
    identities: list[IdentityView] = Field(default_factory=list)
    linked_platforms: int = 0  # platforms linked besides Google
    level: int = 0  # position level mirrored by the bot (salid.positions), 0 = unknown


def member_view(m: Member, p: UserProfile | None = None) -> MemberView:
    """A member as the pages see them: the org record (role, manager, Discord id, the level the
    bot mirrors) plus the person's profile (names, title, department, linked accounts). The
    profile's title wins; the bot-mirrored `Member.title` fills in when the profile has none.
    Platform accounts verified by the bot (`Member.identities`) show as verified identities too."""
    p = p or UserProfile(email=m.email)
    ids = identity_views(p)
    have = {(i.platform, i.external_id) for i in ids}
    for platform, ext in m.all_identities().items():
        if (platform, ext) in have or any(i.platform == platform for i in ids[1:]):
            continue
        # a bot-verified chat account (LINE, WhatsApp...): never the whole id (main's rule,
        # test_member_list_does_not_leak_platform_ids); Discord ids are public in Discord
        shown = ext if platform == "discord" else "…" + str(ext)[-4:]
        ids.append(IdentityView(platform=platform, external_id=shown, verified=True))
    return MemberView(
        email=m.email,
        role=m.role,
        discord_user_id=m.discord_user_id or m.identity("discord"),
        reports_to=m.reports_to,
        name=p.name,
        first_name=p.first_name,
        last_name=p.last_name,
        full_name=p.full_name,
        title=p.title or m.title,
        department=p.department or m.department,
        about=p.about,
        job_description=p.job_description,
        identities=ids,
        linked_platforms=len(ids) - 1,
        level=m.level,
    )


class WorkspaceView(BaseModel):
    id: str
    platform: str
    external_id: str
    name: str
    icon_url: str
    status: str
    connected_at: str
    connected_by: str
    settings_url: str | None  # discord: the existing /hub page for per-server settings


class OrgView(BaseModel):
    id: str
    name: str
    logo_url: str
    profile: CompanyProfile = Field(default_factory=CompanyProfile)
    created_at: str
    owner_email: str
    role: str  # the caller's role: owner | admin | member | super
    license: LicenseView
    members: list[MemberView]
    workspaces: list[WorkspaceView]
    sharing: list[ShareLink]
    handover: HandoverView | None = None


class HandoverView(BaseModel):
    from_email: str
    to_email: str
    at: str
    undo_until: str
    can_undo: bool  # within the window and not undone yet (the former owner acts on it)
    undone_at: str = ""


class OrgBrief(BaseModel):
    id: str
    name: str
    role: str
    license: LicenseView


class MeView(BaseModel):
    email: str
    is_super: bool
    orgs: list[OrgBrief]
    language: str | None = None  # saved web UI language (th / en), None = not chosen yet
    name: str = ""
    first_name: str = ""
    last_name: str = ""
    full_name: str = ""
    title: str = ""
    department: str = ""
    about: str = ""
    job_description: str = ""
    identities: list[IdentityView] = Field(default_factory=list)  # "my account" on each platform
    identity_platforms: list[str] = Field(default_factory=lambda: list(IDENTITY_PLATFORMS))
    discord_verify: bool = False  # True when "Verify with Discord" (OAuth) is configured


class LanguageBody(BaseModel):
    language: str


class ProfileBody(BaseModel):
    name: str | None = Field(default=None, max_length=1_000)
    first_name: str | None = Field(default=None, max_length=1_000)
    last_name: str | None = Field(default=None, max_length=1_000)
    title: str | None = Field(default=None, max_length=1_000)
    department: str | None = Field(default=None, max_length=1_000)
    about: str | None = Field(default=None, max_length=50_000)
    job_description: str | None = Field(default=None, max_length=50_000)


class IdentityBody(BaseModel):
    platform: str
    external_id: str = Field(min_length=1, max_length=120)
    name: str = Field(default="", max_length=120)


class IdentityCodeBody(BaseModel):
    code: str = Field(min_length=8, max_length=200)


def license_view(lic: License, now: datetime | None = None) -> LicenseView:
    now = now or datetime.now(UTC)
    end = license_end(lic)
    return LicenseView(
        **lic.model_dump(),
        active=license_ok(lic, now),
        days_left=days_left(lic, now),
        grace_until=end.date().isoformat() if end else None,
    )


def workspace_view(w: Workspace) -> WorkspaceView:
    return WorkspaceView(**w.model_dump(), settings_url="/hub" if w.platform == "discord" else None)


@dataclass
class Ctx:
    email: str
    is_super: bool
    org: Org
    role: str  # owner | admin | member | super

    @property
    def is_exec(self) -> bool:
        return self.is_super or self.role in ("owner", "admin")


def org_view(ctx: Ctx) -> OrgView:
    o = ctx.org
    return OrgView(
        id=o.id,
        name=o.name,
        logo_url=o.logo_url,
        profile=o.profile,
        created_at=o.created_at,
        owner_email=o.owner_email,
        role=ctx.role,
        license=license_view(o.license),
        members=[member_view(m) for m in o.members],
        workspaces=[workspace_view(w) for w in o.workspaces],
        sharing=o.sharing,
        handover=handover_view(o.handover),
    )


def handover_view(h: Handover | None, now: datetime | None = None) -> HandoverView | None:
    if h is None:
        return None
    return HandoverView(
        from_email=h.from_email,
        to_email=h.to_email,
        at=h.at,
        undo_until=h.undo_until,
        can_undo=h.can_undo(now or datetime.now(UTC)),
        undone_at=h.undone_at,
    )


# ------------------------------------------------------------------ access
async def member_ctx(
    org_id: str, email: str = Depends(hub.require_user), orgs: OrgStore = Depends(get_orgs)
) -> Ctx:
    """403 unless the caller belongs to the org (or is a super admin; 404 for unknown orgs)."""
    sup = hub.is_super_admin(email)
    org = await run_in_threadpool(orgs.get, org_id)
    if org is None:
        raise HTTPException(404 if sup else 403, "No access to this organization")
    role = org.role_of(email)
    if role is None and not sup:
        raise HTTPException(403, "No access to this organization")
    return Ctx(email, sup, org, role or "super")


async def admin_ctx(ctx: Ctx = Depends(member_ctx)) -> Ctx:
    if not ctx.is_exec:
        raise HTTPException(403, "Owner or admin only")
    return ctx


async def super_user(email: str = Depends(hub.require_user)) -> str:
    if not hub.is_super_admin(email):
        raise HTTPException(403, "Super admin only")
    return email


# ------------------------------------------------------------------ page
@router.get("/ping", include_in_schema=False)
def ping() -> dict:
    return {"ok": True}


def center_page() -> HTMLResponse:
    client_id = get_settings().hub_google_client_id
    if not hub.CLIENT_ID_RE.match(client_id):
        client_id = ""
    if not CENTER_HTML.exists():
        raise HTTPException(404, "center.html is not built yet")
    from meeting_scribe.app import salid_invite_url  # late: app imports this module

    invite = salid_invite_url(get_settings().salid_discord_client_id)
    html = (
        CENTER_HTML.read_text(encoding="utf-8")
        .replace("__HUB_GOOGLE_CLIENT_ID__", client_id)
        .replace("__INVITE_URL__", invite)
        .replace("__DISCORD_LOGIN__", "1" if discord_login_enabled() else "")
        .replace("__MONITOR__", "1" if get_settings().monitor_enabled else "")
    )
    return HTMLResponse(
        html, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}
    )


events_router.add_api_route("/center", center_page, methods=["GET"], include_in_schema=False)


PM_HTML = Path(__file__).parent / "static" / "pm.html"


def pm_page() -> HTMLResponse:
    client_id = get_settings().hub_google_client_id
    if not hub.CLIENT_ID_RE.match(client_id):
        client_id = ""
    if not PM_HTML.exists():
        raise HTTPException(404, "pm.html is not built yet")
    html = PM_HTML.read_text(encoding="utf-8").replace("__HUB_GOOGLE_CLIENT_ID__", client_id)
    return HTMLResponse(
        html, headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}
    )


events_router.add_api_route("/pm", pm_page, methods=["GET"], include_in_schema=False)


# ------------------------------------------------------------------ me / orgs
@router.get("/me")
async def me(email: str = Depends(hub.require_user), orgs: OrgStore = Depends(get_orgs)) -> MeView:
    sup = hub.is_super_admin(email)
    try:  # self-repair of the sign-in index / Member.discord_user_id (see OrgStore.heal_identities)
        await run_in_threadpool(orgs.heal_identities, email)
    except Exception:
        logger.warning("could not heal the identities of %s", email, exc_info=True)
    found = await run_in_threadpool(orgs.all_orgs if sup else (lambda: orgs.orgs_of_email(email)))
    profile = await run_in_threadpool(orgs.user_profile, email)
    return MeView(
        email=email,
        is_super=sup,
        language=profile.language if profile.language in ("th", "en") else None,
        name=profile.name,
        first_name=profile.first_name,
        last_name=profile.last_name,
        full_name=profile.full_name,
        title=profile.title,
        department=profile.department,
        about=profile.about,
        job_description=profile.job_description,
        identities=identity_views(profile),
        discord_verify=bool(get_settings().salid_discord_client_secret),
        orgs=[
            OrgBrief(
                id=o.id,
                name=o.name,
                role=o.role_of(email) or "super",
                license=license_view(o.license),
            )
            for o in found
        ],
    )


class CreateOrg(BaseModel):
    name: str = Field(min_length=1, max_length=120)


@router.put("/me/language")
async def set_my_language(
    body: LanguageBody, email: str = Depends(hub.require_user), orgs: OrgStore = Depends(get_orgs)
) -> dict[str, str]:
    """Remember the language this person wants every Salid page in (hub, center, start)."""
    try:
        lang = await run_in_threadpool(orgs.set_user_language, email, body.language)
    except ValueError as exc:
        raise HTTPException(400, "Unsupported language (th or en)") from exc
    return {"language": lang}


# ------------------------------------------------------------------ my account (the person)
# A PERSON links their own accounts here; an ORG links workspaces under /orgs/{id}/workspaces.
# Both are separate on purpose: the profile follows the person into every org they belong to.
@router.put("/me/profile")
async def set_my_profile(
    body: ProfileBody, email: str = Depends(hub.require_user), orgs: OrgStore = Depends(get_orgs)
) -> MeView:
    await run_in_threadpool(
        orgs.set_user_profile,
        email,
        name=body.name,
        first_name=body.first_name,
        last_name=body.last_name,
        title=body.title,
        department=body.department,
        about=body.about,
        job_description=body.job_description,
    )
    return await me(email, orgs)


@router.put("/me/identity")
async def set_my_identity(
    body: IdentityBody, email: str = Depends(hub.require_user), orgs: OrgStore = Depends(get_orgs)
) -> list[IdentityView]:
    """Type in your own account on a platform (unverified until the platform confirms it)."""
    if body.platform not in IDENTITY_PLATFORMS:
        raise HTTPException(400, f"platform must be one of {', '.join(IDENTITY_PLATFORMS)}")
    ident = Identity(platform=body.platform, external_id=body.external_id, name=body.name)
    try:
        p = await run_in_threadpool(orgs.set_user_identity, email, ident)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return identity_views(p)


class GoogleLinkBody(BaseModel):
    credential: str = Field(min_length=20, max_length=4000)  # a Google ID token from the button


@router.put("/me/identity/google")
async def link_my_google(
    body: GoogleLinkBody,
    email: str = Depends(hub.require_user),
    verify: Callable[[str, str], dict[str, Any]] = Depends(hub.google_token_verifier),
    orgs: OrgStore = Depends(get_orgs),
) -> list[IdentityView]:
    """Add a Google account to THIS account (for people who signed up with Discord): from now
    on "Sign in with Google" with it lands here. The token is verified like a sign-in; 409 when
    that Google account is already a Salid account of its own."""
    try:
        claims = verify(body.credential, get_settings().hub_google_client_id)
    except ValueError as exc:
        raise HTTPException(401, "Invalid Google sign-in") from exc
    if claims.get("email_verified") is not True or not claims.get("email"):
        raise HTTPException(403, "This Google account has no verified email")
    g = str(claims["email"]).lower()
    name = str(claims.get("name") or "").strip()
    try:
        p = await run_in_threadpool(orgs.link_google, email, g, name)
    except SigninTaken as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return identity_views(p)


@router.delete("/me/identity/{platform}")
async def remove_my_identity(
    platform: str, email: str = Depends(hub.require_user), orgs: OrgStore = Depends(get_orgs)
) -> list[IdentityView]:
    p = await run_in_threadpool(orgs.remove_user_identity, email, platform)
    return identity_views(p)


@router.post("/me/identity/discord/code")
async def verify_discord_by_code(
    body: IdentityCodeBody,
    email: str = Depends(hub.require_user),
    orgs: OrgStore = Depends(get_orgs),
    guilds=Depends(hub.get_guild_store),
) -> list[IdentityView]:
    """Redeem the one-time code Salid handed you in Discord (`/ลิงก์บัญชี` / `/linkaccount`):
    proof that this Google account is that Discord user."""
    try:
        _gid, uid, uname = await run_in_threadpool(
            guilds.redeem_identity_code, body.code.strip(), email
        )
    except LinkError as exc:
        raise HTTPException(400, "Code is invalid, expired or already used") from exc
    ident = Identity(platform="discord", external_id=uid, name=uname, verified=True)
    p = await run_in_threadpool(orgs.set_user_identity, email, ident)
    return identity_views(p)


DISCORD_OAUTH = "https://discord.com/api/oauth2"
IDENTITY_STATE = "discord_identity"


def discord_identity_redirect() -> str:
    # registered in the Discord app: <public url>/api/center/identity/discord/callback
    return f"{connector_config().base_url.rstrip('/')}/api/center/identity/discord/callback"


def exchange_discord_identity(code: str) -> tuple[str, str]:
    """OAuth2 `identify`: turn the callback code into (discord user id, display name).
    Overridden in tests."""
    from salid.connectors.base import http_client

    s = get_settings()
    with http_client() as http:
        tok = http.post(
            f"{DISCORD_OAUTH}/token",
            data={
                "client_id": s.salid_discord_client_id,
                "client_secret": s.salid_discord_client_secret,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": discord_identity_redirect(),
            },
        )
        tok.raise_for_status()
        me_ = http.get(
            "https://discord.com/api/v10/users/@me",
            headers={"Authorization": f"Bearer {tok.json()['access_token']}"},
        )
        me_.raise_for_status()
    u = me_.json()
    return str(u["id"]), str(u.get("global_name") or u.get("username") or "")


def get_discord_identity_exchange() -> Callable[[str], tuple[str, str]]:
    return exchange_discord_identity


# ------------------------------------------------------------------ sign in with Discord
# Same OAuth app, scope `identify email`: a Discord account with a VERIFIED email signs in as
# that email (the key of every org membership) and gets a Salid session, exactly like Google.
LOGIN_STATE = "discord_login"
LOGIN_NEXT = ("center", "start", "hub")


def discord_login_enabled() -> bool:
    s = get_settings()
    return bool(
        s.salid_discord_client_id and s.salid_discord_client_secret and s.salid_session_secret
    )


def discord_login_redirect() -> str:
    # registered in the Discord app: <public url>/api/center/auth/discord/callback
    return f"{connector_config().base_url.rstrip('/')}/api/center/auth/discord/callback"


def exchange_discord_login(code: str) -> tuple[str, str, str, bool]:
    """(discord user id, display name, email, email_verified). Overridden in tests."""
    from salid.connectors.base import http_client

    s = get_settings()
    with http_client() as http:
        tok = http.post(
            f"{DISCORD_OAUTH}/token",
            data={
                "client_id": s.salid_discord_client_id,
                "client_secret": s.salid_discord_client_secret,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": discord_login_redirect(),
            },
        )
        tok.raise_for_status()
        me_ = http.get(
            "https://discord.com/api/v10/users/@me",
            headers={"Authorization": f"Bearer {tok.json()['access_token']}"},
        )
        me_.raise_for_status()
    u = me_.json()
    name = str(u.get("global_name") or u.get("username") or "")
    return str(u["id"]), name, str(u.get("email") or "").lower(), bool(u.get("verified"))


def get_discord_login_exchange() -> Callable[[str], tuple[str, str, str, bool]]:
    return exchange_discord_login


@router.get("/auth/discord/start", include_in_schema=False)
async def discord_login_start(
    next: str = Query(default="center"),
    invite: str = Query(default="", max_length=64, pattern=r"^[A-Za-z0-9_-]*$"),
    orgs: OrgStore = Depends(get_orgs),
) -> RedirectResponse:
    """Browser navigation (no Bearer): send the person to Discord's consent screen. An invite
    code travels in the server-side state and comes back on the landing URL, so an invitation
    survives the round trip even when Discord opens in another browser (phones: the Discord
    app's in-app browser has its own localStorage)."""
    if not discord_login_enabled():
        raise HTTPException(503, "Sign in with Discord is not configured")
    nxt = next if next in LOGIN_NEXT else "center"
    state = secrets.token_urlsafe(24)
    exp = (datetime.now(UTC) + STATE_TTL).isoformat(timespec="seconds")
    await run_in_threadpool(
        orgs.docs.put,
        STATE_KIND,
        0,
        state,
        {"platform": LOGIN_STATE, "exp": exp, "next": nxt, "invite": invite},
    )
    q = urlencode(
        {
            "client_id": get_settings().salid_discord_client_id,
            "response_type": "code",
            "scope": "identify email",
            "redirect_uri": discord_login_redirect(),
            "state": state,
            "prompt": "consent",
        }
    )
    return RedirectResponse(f"{DISCORD_OAUTH}/authorize?{q}", status_code=303)


@router.get("/auth/discord/callback", include_in_schema=False)
async def discord_login_callback(
    request: Request,
    orgs: OrgStore = Depends(get_orgs),
    exchange: Callable[[str], tuple[str, str, str, bool]] = Depends(get_discord_login_exchange),
) -> RedirectResponse:
    """Discord sends the browser back here. Success lands on `/<next>#session=<token>` (the page
    stores the session and cleans the address bar); failure on `/<next>?error=<code>`."""
    params = dict(request.query_params)

    invite = ""

    def run() -> tuple[str, str | None, str | None]:
        nonlocal invite
        st = orgs.docs.get(STATE_KIND, 0, params.get("state", ""))
        if not st or st.get("platform") != LOGIN_STATE:
            return "center", "bad_state", None
        orgs.docs.delete(STATE_KIND, 0, params["state"])  # one time
        nxt = st.get("next") if st.get("next") in LOGIN_NEXT else "center"
        invite = str(st.get("invite") or "")[:64]
        if datetime.fromisoformat(st["exp"]) < datetime.now(UTC):
            return nxt, "expired", None
        if "code" not in params:
            return nxt, "denied", None
        try:
            uid, uname, email, verified = exchange(params["code"])
        except Exception:
            logger.exception("discord login exchange failed")
            return nxt, "exchange_failed", None
        # Someone who already verified this Discord account (Verify with Discord, or the
        # /linkaccount code) signs in as THAT person, whatever email Discord reports.
        owner = orgs.email_for_identity("discord", uid)
        if owner:
            email = owner
        elif not email or not verified:
            return nxt, "discord_no_email", None
        orgs.set_user_identity(
            email, Identity(platform="discord", external_id=uid, name=uname, verified=True)
        )
        if not orgs.user_profile(email).name and uname:
            orgs.set_user_name(email, uname)
        s = get_settings()
        token, _exp = hub.issue_session(email, s.salid_session_secret, s.salid_session_days)
        return nxt, None, token

    nxt, err, token = await run_in_threadpool(run)
    if err:
        return RedirectResponse(
            f"/{nxt}?{urlencode({'error': err, 'platform': 'discord'})}", status_code=303
        )
    back = f"/{nxt}" + (f"?{urlencode({'invite': invite})}" if invite else "")
    return RedirectResponse(f"{back}#session={token}", status_code=303)


@router.get("/identity/discord/start")
async def discord_identity_start(
    email: str = Depends(hub.require_user), orgs: OrgStore = Depends(get_orgs)
) -> dict[str, str]:
    """URL of Discord's consent screen (scope `identify` only). 503 until the client secret is set."""
    s = get_settings()
    if not (s.salid_discord_client_id and s.salid_discord_client_secret):
        raise HTTPException(503, "Discord verification is not configured")
    state = secrets.token_urlsafe(24)
    exp = (datetime.now(UTC) + STATE_TTL).isoformat(timespec="seconds")
    await run_in_threadpool(
        orgs.docs.put,
        STATE_KIND,
        0,
        state,
        {"email": email, "platform": IDENTITY_STATE, "exp": exp},
    )
    q = urlencode(
        {
            "client_id": s.salid_discord_client_id,
            "response_type": "code",
            "scope": "identify",
            "redirect_uri": discord_identity_redirect(),
            "state": state,
            "prompt": "consent",
        }
    )
    return {"url": f"{DISCORD_OAUTH}/authorize?{q}"}


@router.get("/identity/discord/callback", include_in_schema=False)
async def discord_identity_callback(
    request: Request,
    orgs: OrgStore = Depends(get_orgs),
    exchange: Callable[[str], tuple[str, str]] = Depends(get_discord_identity_exchange),
) -> RedirectResponse:
    """Browser redirect from Discord (no Bearer header): the one-time `state` names the person.
    Lands on `/center#/account?linked=discord` or `/center#/account?error=<code>`."""
    params = dict(request.query_params)

    def go(**q: str) -> RedirectResponse:
        return RedirectResponse(f"/center?{urlencode(q)}#/account", status_code=303)

    def run() -> str | None:
        st = orgs.docs.get(STATE_KIND, 0, params.get("state", ""))
        if not st or st.get("platform") != IDENTITY_STATE:
            return "bad_state"
        orgs.docs.delete(STATE_KIND, 0, params["state"])  # one time
        if datetime.fromisoformat(st["exp"]) < datetime.now(UTC):
            return "expired"
        if "code" not in params:
            return "denied"
        try:
            uid, uname = exchange(params["code"])
        except Exception:
            logger.exception("discord identity exchange failed")
            return "exchange_failed"
        orgs.set_user_identity(
            st["email"], Identity(platform="discord", external_id=uid, name=uname, verified=True)
        )
        return None

    err = await run_in_threadpool(run)
    return go(error=err, platform="discord") if err else go(linked="discord")


@router.post("/orgs")
async def create_org(
    body: CreateOrg, email: str = Depends(hub.require_user), orgs: OrgStore = Depends(get_orgs)
) -> OrgView:
    """The creator becomes owner; the license is the open-ended launch trial."""

    def run() -> Org:
        mine = [o for o in orgs.orgs_of_email(email) if o.role_of(email) == "owner"]
        if len(mine) >= MAX_ORGS_PER_USER:
            raise HTTPException(409, "Too many organizations")
        return orgs.create(body.name, email, license=trial_license())

    org = await run_in_threadpool(run)
    return org_view(Ctx(email, hub.is_super_admin(email), org, "owner"))


@router.get("/orgs/{org_id}")
async def get_org(ctx: Ctx = Depends(member_ctx)) -> OrgView:
    return org_view(ctx)


class CompanyProfileBody(BaseModel):
    tagline: str | None = Field(default=None, max_length=1_000)
    about: str | None = Field(default=None, max_length=50_000)
    industry: str | None = Field(default=None, max_length=1_000)
    founded: str | None = Field(default=None, max_length=1_000)
    location: str | None = Field(default=None, max_length=1_000)
    website: str | None = Field(default=None, max_length=2_000)
    email: str | None = Field(default=None, max_length=320)
    phone: str | None = Field(default=None, max_length=1_000)
    size: str | None = Field(default=None, max_length=1_000)
    values: list[str] | None = Field(default=None, max_length=8)
    registered_name: str | None = Field(default=None, max_length=1_000)
    registration_no: str | None = Field(default=None, max_length=1_000)
    tax_id: str | None = Field(default=None, max_length=1_000)
    registered_capital: str | None = Field(default=None, max_length=1_000)
    registered_address: str | None = Field(default=None, max_length=50_000)
    directors: str | None = Field(default=None, max_length=50_000)
    objectives: str | None = Field(default=None, max_length=50_000)


class UpdateOrg(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    logo_url: str | None = Field(default=None, max_length=500, pattern=r"^(https?://\S+)?$")
    profile: CompanyProfileBody | None = None


@router.put("/orgs/{org_id}")
async def update_org(
    body: UpdateOrg, ctx: Ctx = Depends(admin_ctx), orgs: OrgStore = Depends(get_orgs)
) -> OrgView:
    """Name, logo and the company profile (admins). Fields left out stay as they are."""

    def run() -> None:
        with orgs.lock:
            org = orgs.get(ctx.org.id)
            if body.name is not None:
                org.name = body.name.strip()
            if body.logo_url is not None:
                org.logo_url = body.logo_url
            orgs.save(org)
            if body.profile is not None:
                org = orgs.set_company_profile(ctx.org.id, **body.profile.model_dump())
            ctx.org = org

    await run_in_threadpool(run)
    return org_view(ctx)


# ------------------------------------------------------------------ members
RANK = {"member": 1, "admin": 2, "owner": 3}


def role_change_error(
    ctx: Ctx, current: str | None, new: str, self_change: bool
) -> tuple[int, str] | None:
    """Why `ctx` may not set this member's role to `new` (None = allowed). Pure, so the UI and
    the invite links follow the same rules (the owner's security model, 2026-10-18):
    * ONE owner. Ownership is never granted: the owner hands it over with `transfer_ownership`
      (to somebody who is already a member) and gives up their own;
    * admin and owner grant admin and member (by email or by invite link);
    * only the owner (or a super admin) changes somebody who is an admin;
    * your own role can only go DOWN (admin -> member); never up."""
    owner = ctx.is_super or ctx.role == "owner"
    if self_change:
        if ctx.is_super:
            return None
        if current and RANK.get(new, 0) > RANK.get(current, 0):
            return 403, "You cannot raise your own role"
        return None  # lowering yourself (the owner must transfer ownership first)
    if new == "owner":
        return 403, "Ownership is handed over with Transfer ownership, never granted"
    if new == "admin" and not ctx.is_exec:
        return 403, "Only an owner or admin can grant admin"
    if current in ("owner", "admin") and not owner:
        return 403, "Only an owner can change an admin or owner"
    return None


class TransferBody(BaseModel):
    to: str = Field(min_length=3, max_length=200, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    me: str = "admin"  # what the former owner becomes: admin | member


class MemberBody(BaseModel):
    email: str = Field(min_length=3, max_length=200, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    role: str = "member"
    discord_user_id: str | None = Field(default=None, max_length=30)


def _monitor_hook(docs: Any, org_id: str, adapt: Callable[[Any], Any]) -> Callable | None:
    """A delivery queue's `on_change` on the web: the changed records (`adapt` -> monitor items)
    are written to the org's change log at once, one best-effort transactional write per state
    change (the web has no tick to batch on; `monitor.emit_items` never raises). None while
    `MONITOR_ENABLED` is off."""
    if not get_settings().monitor_enabled:
        return None

    def emit(xs: list) -> None:
        monitor.emit_items(docs, org_id, [adapt(x) for x in xs])

    return emit


def _record_edit(orgs: OrgStore, before: Org, after: Org, by: str) -> int:
    """Record an org edit in the change outbox (never raises), with a monitor event per stored
    change while `MONITOR_ENABLED` is on."""
    hook = _monitor_hook(orgs.docs, after.id, lambda c: adapt_change(c, now=datetime.now(UTC)))
    return record_org_edit(orgs.docs, before, after, by, on_change=hook)


@router.get("/orgs/{org_id}/members")
async def list_members(
    ctx: Ctx = Depends(admin_ctx), orgs: OrgStore = Depends(get_orgs)
) -> list[MemberView]:
    """Every member with who they are on each platform (name, linked accounts, verified or not)."""
    profiles = await run_in_threadpool(orgs.member_profiles, ctx.org)
    return [member_view(m, profiles.get(m.email)) for m in ctx.org.members]


@router.post("/orgs/{org_id}/members")
async def add_member(
    body: MemberBody, ctx: Ctx = Depends(admin_ctx), orgs: OrgStore = Depends(get_orgs)
) -> list[MemberView]:
    """Add a member or change a member's role. The rules (`role_change_error`): an admin or an
    owner may grant admin, only an owner (or a super admin) may grant owner or change another
    admin / owner; you may only LOWER your own role (admin -> member; an owner hands the
    organization over with `transfer_ownership`); there is always at least one owner."""
    if body.role not in ROLES:
        raise HTTPException(422, "role must be owner, admin or member")
    email = body.email.lower()

    def run() -> Org:
        with orgs.lock:
            org = orgs.get(ctx.org.id)
            current = org.role_of(email)
            if why := role_change_error(ctx, current, body.role, email == ctx.email):
                raise HTTPException(*why)
            if current is None and len(org.members) >= max(org.license.seats, 1):
                raise HTTPException(402, "Seat limit reached; request more seats")
            owners = sum(1 for m in org.members if m.role == "owner")
            if current == "owner" and body.role != "owner" and owners <= 1:
                raise HTTPException(
                    409, "An organization needs at least one owner: transfer ownership first"
                )
            prev = next((m for m in org.members if m.email.lower() == email), None)
            org.members = [m for m in org.members if m.email.lower() != email]
            if prev is not None:  # an edit: keep title, level, department, identities, manager...
                kept = prev.model_copy(update={"role": body.role})
                if body.discord_user_id:
                    kept.discord_user_id = body.discord_user_id
                org.members.append(kept)
            else:
                org.members.append(
                    Member(
                        email=email,
                        role=body.role,
                        discord_user_id=body.discord_user_id or orgs.verified_discord_id(email),
                    )
                )
            if body.role == "owner" and not org.owner_email:
                org.owner_email = email
            return orgs.save(org)

    org = await run_in_threadpool(run)
    await run_in_threadpool(_record_edit, orgs, ctx.org, org, ctx.email)  # change outbox
    return [member_view(m) for m in org.members]


@router.post("/orgs/{org_id}/owner/transfer")
async def transfer_ownership(
    body: TransferBody, ctx: Ctx = Depends(admin_ctx), orgs: OrgStore = Depends(get_orgs)
) -> list[MemberView]:
    """The owner hands the organization to another member and becomes admin or member (their
    choice). Owners only; the receiver must already be a member (never an outsider: security)."""
    if ctx.role != "owner":
        raise HTTPException(403, "Only an owner can transfer ownership")
    if body.me not in ("admin", "member"):
        raise HTTPException(422, "me must be admin or member")
    to = body.to.lower()
    if to == ctx.email:
        raise HTTPException(422, "Choose another member to hand the organization to")

    def run() -> Org:
        with orgs.lock:
            org = orgs.get(ctx.org.id)
            now = orgs._clock()
            if org.handover and org.handover.can_undo(now):
                raise HTTPException(
                    409,
                    "The previous handover can still be taken back until "
                    f"{org.handover.undo_until}; ownership cannot move on before that",
                )
            if (was := org.role_of(to)) is None:
                raise HTTPException(404, "That person is not a member yet: add them first")
            for m in org.members:
                if m.email.lower() == to:
                    m.role = "owner"
                elif m.email.lower() == ctx.email:
                    m.role = body.me
            org.owner_email = to
            org.handover = Handover(
                from_email=ctx.email,
                to_email=to,
                at=stamp(now),
                undo_until=stamp(now + timedelta(hours=HANDOVER_UNDO_HOURS)),
                former_role=body.me,
                to_role=was,
            )
            return orgs.save(org)

    org = await run_in_threadpool(run)
    await run_in_threadpool(_record_edit, orgs, ctx.org, org, ctx.email)  # change outbox
    return [member_view(m) for m in org.members]


@router.post("/orgs/{org_id}/owner/undo")
async def undo_transfer(
    ctx: Ctx = Depends(member_ctx), orgs: OrgStore = Depends(get_orgs)
) -> list[MemberView]:
    """Security first: the former owner takes the organization back within 24 hours of handing
    it over (the receiver returns to the role they had). Only the former owner, only once."""

    def run() -> Org:
        with orgs.lock:
            org = orgs.get(ctx.org.id)
            h = org.handover
            if h is None or h.from_email != ctx.email:
                raise HTTPException(
                    403, "Only the owner who handed the organization over can take it back"
                )
            now = orgs._clock()
            if not h.can_undo(now):
                raise HTTPException(410, "The time to take ownership back has passed")
            if org.role_of(h.to_email) != "owner" or org.role_of(h.from_email) is None:
                raise HTTPException(409, "The organization has changed since the handover")
            for m in org.members:
                if m.email.lower() == h.to_email:
                    m.role = h.to_role if h.to_role in ("admin", "member") else "member"
                elif m.email.lower() == h.from_email:
                    m.role = "owner"
            org.owner_email = h.from_email
            h.undone_at = stamp(now)
            return orgs.save(org)

    org = await run_in_threadpool(run)
    await run_in_threadpool(_record_edit, orgs, ctx.org, org, ctx.email)  # change outbox
    return [member_view(m) for m in org.members]


def _remove_member(ctx: Ctx, orgs: OrgStore, email: str) -> list[MemberView]:
    email = email.lower()
    with orgs.lock:
        org = orgs.get(ctx.org.id)
        target = org.role_of(email)
        if target is None:
            raise HTTPException(404, "Not a member")
        can_grant = ctx.is_super or ctx.role == "owner"
        if target in ("owner", "admin") and not can_grant:
            raise HTTPException(403, "Only an owner can remove an admin or owner")
        if target == "owner" and sum(1 for m in org.members if m.role == "owner") <= 1:
            raise HTTPException(409, "An organization needs at least one owner")
        org.members = [m for m in org.members if m.email.lower() != email]
        orgs.save(org)
    _record_edit(orgs, ctx.org, org, ctx.email)  # change outbox (never raises)
    return [member_view(m) for m in org.members]


@router.delete("/orgs/{org_id}/members")
async def delete_member_q(
    email: str = Query(min_length=3),
    ctx: Ctx = Depends(admin_ctx),
    orgs: OrgStore = Depends(get_orgs),
) -> list[MemberView]:
    return await run_in_threadpool(_remove_member, ctx, orgs, email)


@router.delete("/orgs/{org_id}/members/{email}")
async def delete_member(
    email: str, ctx: Ctx = Depends(admin_ctx), orgs: OrgStore = Depends(get_orgs)
) -> list[MemberView]:
    return await run_in_threadpool(_remove_member, ctx, orgs, email)


@router.get("/orgs/{org_id}/changes")
async def list_changes(
    limit: int = Query(50, ge=1, le=200),
    ctx: Ctx = Depends(admin_ctx),
    orgs: OrgStore = Depends(get_orgs),
) -> list[dict]:
    """Recent central changes with the status of every platform target (who confirmed)."""
    found = await run_in_threadpool(ChangeOutbox(orgs.docs).list_changes, ctx.org.id, limit)
    return [c.model_dump(exclude={"subject_ids", "subject_tenants"}) for c in found]


# ------------------------------------------------------------------ invite links
class InviteBody(BaseModel):
    role: str = "member"
    days: int = Field(default=INVITE_DAYS, ge=0, le=INVITE_MAX_DAYS)  # 0 = never expires
    max_uses: int = Field(default=0, ge=0, le=INVITE_MAX_USES)  # 0 = unlimited
    label: str = Field(default="", max_length=60)


class InviteView(BaseModel):
    code: str
    url: str
    role: str
    label: str
    status: str  # active | revoked | expired | used_up
    created_by: str
    created_at: str
    expires_at: str  # "" = never
    max_uses: int  # 0 = unlimited
    uses: int
    used: list[str]
    revoked_at: str


class InviteInfo(BaseModel):
    """What a person sees before signing in: whose org the link opens (public, no secrets)."""

    code: str
    status: str
    org_name: str
    logo_url: str
    role: str
    invited_by: str
    expires_at: str


class InviteAccepted(BaseModel):
    org_id: str
    org_name: str
    role: str
    joined: bool  # False: already a member (role kept)


def public_base(request: Request) -> str:
    base = connector_config().base_url.rstrip("/")
    return base or str(request.base_url).rstrip("/")


def invite_url(base: str, code: str) -> str:
    return f"{base}/center?invite={code}"


def invite_view(inv: Invite, base: str, now: datetime | None = None) -> InviteView:
    return InviteView(
        code=inv.code,
        url=invite_url(base, inv.code),
        role=inv.role,
        label=inv.label,
        status=inv.status(now),
        created_by=inv.created_by,
        created_at=inv.created_at,
        expires_at=inv.expires_at,
        max_uses=inv.max_uses,
        uses=inv.uses,
        used=list(inv.used),
        revoked_at=inv.revoked_at,
    )


@router.get("/orgs/{org_id}/invites")
async def list_invites(
    request: Request,
    all: bool = False,
    ctx: Ctx = Depends(admin_ctx),
    orgs: OrgStore = Depends(get_orgs),
) -> list[InviteView]:
    """Invite links of the org: the live ones (default), or every link with `?all=1`."""
    rows = await run_in_threadpool(orgs.invites, ctx.org.id, active_only=not all)
    base, now = public_base(request), orgs._clock()
    return [invite_view(i, base, now) for i in rows]


@router.post("/orgs/{org_id}/invites")
async def create_invite(
    body: InviteBody,
    request: Request,
    ctx: Ctx = Depends(admin_ctx),
    orgs: OrgStore = Depends(get_orgs),
) -> InviteView:
    """Make an invite link for members or admins (admins and the owner). Never for owner:
    ownership is transferred, not granted (same rules as changing a role)."""
    if body.role not in ROLES:
        raise HTTPException(422, "role must be owner, admin or member")
    if why := role_change_error(ctx, None, body.role, False):
        raise HTTPException(*why)
    inv = await run_in_threadpool(
        lambda: orgs.create_invite(
            ctx.org.id,
            by=ctx.email,
            role=body.role,
            days=body.days,
            max_uses=body.max_uses,
            label=body.label,
        )
    )
    return invite_view(inv, public_base(request))


@router.delete("/orgs/{org_id}/invites/{code}")
async def revoke_invite(
    code: str,
    request: Request,
    ctx: Ctx = Depends(admin_ctx),
    orgs: OrgStore = Depends(get_orgs),
) -> InviteView:
    """Revoke a link: nobody can join through it any more (people who already did stay)."""
    inv = await run_in_threadpool(orgs.revoke_invite, ctx.org.id, code, by=ctx.email)
    if inv is None:
        raise HTTPException(404, "No such invite link")
    return invite_view(inv, public_base(request))


INVITE_ERRORS = {
    "expired": (410, "This invite link has expired"),
    "revoked": (410, "This invite link was cancelled"),
    "used_up": (410, "This invite link has already been used"),
    "seats": (402, "Seat limit reached; ask the organization for more seats"),
}


def invite_http_error(why: str) -> HTTPException:
    status, msg = INVITE_ERRORS.get(why, (404, "This invite link is not valid"))
    return HTTPException(status, msg)


@router.get("/invites/{code}")
async def invite_info(code: str, orgs: OrgStore = Depends(get_orgs)) -> InviteInfo:
    """Public: who is inviting (shown on the sign-in screen before the person has an account)."""
    inv = await run_in_threadpool(orgs.invite, code)
    org = await run_in_threadpool(orgs.get, inv.org_id) if inv else None
    if inv is None or org is None:
        raise HTTPException(404, "This invite link is not valid")
    return InviteInfo(
        code=inv.code,
        status=inv.status(orgs._clock()),
        org_name=org.name,
        logo_url=org.logo_url,
        role=inv.role,
        invited_by=inv.created_by,
        expires_at=inv.expires_at,
    )


@router.post("/invites/{code}/accept")
async def accept_invite(
    code: str, email: str = Depends(hub.require_user), orgs: OrgStore = Depends(get_orgs)
) -> InviteAccepted:
    """The signed-in person joins the org behind the link (seat limit applies)."""
    try:
        org, inv, joined = await run_in_threadpool(orgs.accept_invite, code, email)
    except InviteError as e:
        raise invite_http_error(str(e)) from None
    except KeyError:
        raise HTTPException(404, "This organization no longer exists") from None
    return InviteAccepted(
        org_id=org.id, org_name=org.name, role=org.role_of(email) or inv.role, joined=joined
    )


# ------------------------------------------------------------------ people on the platforms
class PersonView(BaseModel):
    id: str
    platform: str
    external_id: str
    name: str
    nickname: str
    display_name: str
    email: str  # linked Salid account, "" = no account yet
    member_role: str = ""  # their role in this org when linked
    title: str
    department: str
    about: str
    hidden: bool
    seen: Seen
    observed: Observed | None = None  # execs only
    updated_at: str


class PersonBody(BaseModel):
    nickname: str | None = Field(default=None, max_length=1_000)
    title: str | None = Field(default=None, max_length=1_000)
    department: str | None = Field(default=None, max_length=1_000)
    about: str | None = Field(default=None, max_length=50_000)
    hidden: bool | None = None
    email: str | None = Field(default=None, max_length=200)  # link to a member; "" unlinks


def get_people() -> PeopleBook:
    """The people directory on the bot's DocStore. Overridden in tests."""
    return PeopleBook(get_library().docs)


@lru_cache
def _observer() -> Callable[[str], dict]:
    from salid.llm import make_client
    from salid.people_directory import make_observer

    s, _ = hub.get_salid_runtime()
    return make_observer(make_client(s), s.gate_model)


def get_observer() -> Callable[[str], dict]:
    """(prompt) -> observed fields, through the cheap model. Overridden in tests."""
    return _observer()


def people_workspace(ctx: Ctx, workspace_id: str | None) -> Workspace:
    ws = ctx.org.workspace(workspace_id) if workspace_id else company_workspace(ctx.org)
    if ws is None or ws.status != "connected" or ws.platform != "discord":
        raise HTTPException(404, "No connected Discord workspace")
    return ws


def discord_emails(orgs: OrgStore, org: Org) -> dict[str, str]:
    """discord user id -> member email (the member record, else the verified identity)."""
    out: dict[str, str] = {}
    profiles = orgs.member_profiles(org)
    for m in org.members:
        did = m.discord_user_id or ""
        p = profiles.get(m.email)
        if not did and p is not None:
            i = p.identity("discord")
            did = i.external_id if i and i.verified else ""
        if did:
            out[str(did)] = m.email.lower()
    return out


def person_view(p: Person, org: Org, *, exec_view: bool) -> PersonView:
    return PersonView(
        **{k: v for k, v in p.model_dump().items() if k != "observed"},
        display_name=p.display_name(),
        member_role=(org.role_of(p.email) or "") if p.email else "",
        observed=p.observed if exec_view else None,
    )


@router.get("/orgs/{org_id}/people")
async def list_people(
    workspace_id: str | None = None,
    ctx: Ctx = Depends(member_ctx),
    people: PeopleBook = Depends(get_people),
) -> dict:
    """Everyone seen on the company's Discord (account or not). Members see the public part;
    owners / admins also see hidden people and what Salid observed."""
    ws = people_workspace(ctx, workspace_id)
    rows = await run_in_threadpool(people.list, ws.memory_id, include_hidden=ctx.is_exec)
    return {
        "workspace_id": ws.id,
        "people": [person_view(p, ctx.org, exec_view=ctx.is_exec) for p in rows],
        "with_account": sum(1 for p in rows if p.email),
        "scanned_at": max((p.updated_at for p in rows), default=""),
    }


@router.post("/orgs/{org_id}/people/discover")
async def discover_people(
    workspace_id: str | None = None,
    ctx: Ctx = Depends(admin_ctx),
    orgs: OrgStore = Depends(get_orgs),
    lib: Library = Depends(get_library),
    people: PeopleBook = Depends(get_people),
) -> dict:
    """Scan the last 90 days of the team memory, the tasks and the bot's profiles; link the
    people who have a Salid account (verified Discord)."""
    ws = people_workspace(ctx, workspace_id)
    channels = {cid: name for cid, name in chat_scope(lib, ctx.org, ws, True).items() if cid > 0}

    def run() -> list[Person]:
        tasks = lib.docs.find("salid_tasks", ws.memory_id)
        nicks = {
            str(d.get("user_id")): str(d.get("nickname") or "")
            for d in lib.docs.find("salid_profiles", ws.memory_id)
            if d.get("nickname")
        }
        return people.discover(
            ws.memory_id,
            lib.memory,
            channels,
            today=datetime.now(lib.s.zone).date(),
            tasks=tasks,
            nicknames=nicks,
            emails=discord_emails(orgs, ctx.org),
        )

    rows = await run_in_threadpool(run)
    return {
        "workspace_id": ws.id,
        "people": [person_view(p, ctx.org, exec_view=True) for p in rows],
        "with_account": sum(1 for p in rows if p.email),
        "scanned_at": stamp(),
    }


@router.put("/orgs/{org_id}/people/{pid}")
async def edit_person(
    pid: str,
    body: PersonBody,
    workspace_id: str | None = None,
    ctx: Ctx = Depends(admin_ctx),
    people: PeopleBook = Depends(get_people),
) -> PersonView:
    """What an admin writes about a person (nickname, title, department, what they do), hide
    them, or link / unlink the floating profile to a member's account."""
    ws = people_workspace(ctx, workspace_id)
    if body.email and ctx.org.role_of(body.email) is None:
        raise HTTPException(404, "That email is not a member of this organization")
    try:
        p = await run_in_threadpool(
            lambda: people.edit(
                ws.memory_id,
                pid,
                nickname=body.nickname,
                title=body.title,
                department=body.department,
                about=body.about,
                hidden=body.hidden,
                email=body.email,
            )
        )
    except KeyError:
        raise HTTPException(404, "Unknown person") from None
    return person_view(p, ctx.org, exec_view=True)


@router.post("/orgs/{org_id}/people/{pid}/observe")
async def observe_person(
    pid: str,
    workspace_id: str | None = None,
    ctx: Ctx = Depends(admin_ctx),
    lib: Library = Depends(get_library),
    guilds=Depends(hub.get_guild_store),
    people: PeopleBook = Depends(get_people),
    observer: Callable[[str], dict] = Depends(get_observer),
) -> PersonView:
    """Ask Salid what this person seems to do, from their own messages (one cheap-model call,
    in the company language). Execs only: it is derived from every room."""
    ws = people_workspace(ctx, workspace_id)
    p0 = await run_in_threadpool(people.get, ws.memory_id, pid)
    if p0 is None:
        raise HTTPException(404, "Unknown person")
    channels = {cid: name for cid, name in chat_scope(lib, ctx.org, ws, True).items() if cid > 0}
    lang = company_language(guilds, ws)

    def run() -> Person:
        lines = people.own_lines(
            ws.memory_id,
            lib.memory,
            channels,
            p0.external_id,
            today=datetime.now(lib.s.zone).date(),
        )
        return people.observe(ws.memory_id, pid, lines=lines, observer=observer, lang=lang)

    try:
        p = await run_in_threadpool(run)
    except Exception as exc:
        logger.exception("observe %s failed", pid)
        raise HTTPException(502, "Salid could not read the conversations right now") from exc
    return person_view(p, ctx.org, exec_view=True)


# ------------------------------------------------------------------ positions + expertise
class PositionBody(BaseModel):
    title: str | None = Field(default=None, max_length=1_000)
    department: str | None = Field(default=None, max_length=1_000)
    reports_to: str | None = Field(default=None, max_length=20)
    summary: str | None = Field(default=None, max_length=50_000)
    responsibilities: str | None = Field(default=None, max_length=50_000)
    policies: str | None = Field(default=None, max_length=50_000)
    skills: list[str] | None = Field(default=None, max_length=100)
    holders: list[str] | None = Field(default=None, max_length=20)


@router.get("/orgs/{org_id}/positions")
async def list_positions(ctx: Ctx = Depends(member_ctx)) -> list[Position]:
    """Every job of the company, filled or vacant (organisational knowledge: everyone reads)."""
    return sorted(ctx.org.positions, key=lambda x: (x.department.casefold(), x.title.casefold()))


def _save_position(ctx: Ctx, orgs: OrgStore, pid: str | None, body: PositionBody) -> Position:
    try:
        pos = orgs.save_position(ctx.org.id, by=ctx.email, pid=pid, **body.model_dump())
    except KeyError as exc:
        raise HTTPException(404, "Unknown position") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    after = orgs.get(ctx.org.id)
    if after is not None:  # a holder change reaches the person as a "company position" notice
        _record_edit(orgs, ctx.org, after, ctx.email)  # change outbox (never raises)
    return pos


@router.post("/orgs/{org_id}/positions")
async def create_position(
    body: PositionBody, ctx: Ctx = Depends(admin_ctx), orgs: OrgStore = Depends(get_orgs)
) -> Position:
    """Owners / admins create a job ahead of time (vacant until somebody holds it)."""
    return await run_in_threadpool(_save_position, ctx, orgs, None, body)


@router.put("/orgs/{org_id}/positions/{pid}")
async def update_position(
    pid: str,
    body: PositionBody,
    ctx: Ctx = Depends(admin_ctx),
    orgs: OrgStore = Depends(get_orgs),
) -> Position:
    return await run_in_threadpool(_save_position, ctx, orgs, pid, body)


@router.delete("/orgs/{org_id}/positions/{pid}")
async def delete_position(
    pid: str, ctx: Ctx = Depends(admin_ctx), orgs: OrgStore = Depends(get_orgs)
) -> dict:
    if not await run_in_threadpool(orgs.delete_position, ctx.org.id, pid):
        raise HTTPException(404, "Unknown position")
    after = await run_in_threadpool(orgs.get, ctx.org.id)
    if after is not None:
        await run_in_threadpool(_record_edit, orgs, ctx.org, after, ctx.email)
    return {"deleted": pid}


class ExpertBody(BaseModel):
    need: str = Field(default="", max_length=600)  # "who can handle our tax filing?"
    position_id: str | None = Field(default=None, max_length=20)  # match people to this job
    exclude: str | None = Field(default=None, max_length=200)  # e.g. the person being replaced


class ExpertPick(BaseModel):
    key: str
    name: str
    email: str = ""
    has_account: bool
    title: str = ""
    rank: str
    reason: str
    evidence: list[str]
    confidence: str


class ExpertResult(BaseModel):
    need: str
    position: Position | None = None
    picks: list[ExpertPick]
    gaps: str = ""
    candidates: int


@lru_cache
def _expert_model() -> Callable[[str], dict]:
    from salid.expertise import make_model_call
    from salid.llm import make_client

    s, _ = hub.get_salid_runtime()
    return make_model_call(make_client(s), s.gemini_model)


def get_expert_model() -> Callable[[str], dict]:
    """(prompt) -> ranking dict. Overridden in tests."""
    return _expert_model()


def expertise_candidates(
    ctx: Ctx, orgs: OrgStore, lib: Library, people: PeopleBook, *, exclude: str = ""
) -> tuple[list, dict[str, dict]]:
    """Evidence for everyone: members (profile, positions, observed, tasks) and people seen on
    the platforms without an account. What Salid OBSERVED (read from every room) is used only
    for owners / admins. Returns (candidates, key -> display info)."""
    from salid.expertise import candidate

    org = orgs.get(ctx.org.id) or ctx.org
    profiles = orgs.member_profiles(org)
    ws = company_workspace(org)
    gid = ws.memory_id if ws and ws.platform == "discord" else None
    persons = people.list(gid) if gid else []
    by_uid = {p.external_id: p for p in persons}
    by_email = {p.email: p for p in persons if p.email}
    tasks = lib.docs.find("salid_tasks", gid) if gid else []
    done: dict[str, list[str]] = defaultdict(list)
    open_: dict[str, list[str]] = defaultdict(list)
    for t in sorted(tasks, key=lambda x: x.get("created_at", ""), reverse=True):
        uid = str(t.get("assignee_id") or "")
        if uid and uid != "0" and t.get("title"):
            (open_ if t.get("status") == "open" else done)[uid].append(t["title"])
    held: dict[str, list[str]] = defaultdict(list)
    for pos in org.positions:
        for h in pos.holders:
            held[h].append(pos.title)
    skip = (exclude or "").strip().casefold()
    out, info = [], {}
    for m in org.members:
        p = profiles.get(m.email)
        person = by_email.get(m.email) or (
            by_uid.get(m.discord_user_id or "") if m.discord_user_id else None
        )
        did = m.discord_user_id or (person.external_id if person else "")
        name = (
            (p.full_name or p.name if p else "")
            or (person.display_name() if person else "")
            or m.email.split("@")[0]
        )
        if skip and skip in (m.email.casefold(), name.casefold()):
            continue
        obs = person.observed.model_dump() if person and person.observed and ctx.is_exec else None
        out.append(
            candidate(
                m.email,
                name,
                title=p.title if p else "",
                department=p.department if p else "",
                about=p.about if p else "",
                positions=held.get(m.email, []),
                observed=obs,
                channels=person.seen.channels if person else [],
                tasks_done=done.get(did, []),
                tasks_open=open_.get(did, []),
            )
        )
        info[m.email] = {
            "name": name,
            "email": m.email,
            "has_account": True,
            "title": p.title if p else "",
        }
    linked = {m.discord_user_id for m in org.members if m.discord_user_id}
    for person in persons:
        if person.hidden or person.email or person.external_id in linked:
            continue
        name = person.display_name()
        if skip and skip == name.casefold():
            continue
        obs = person.observed.model_dump() if person.observed and ctx.is_exec else None
        out.append(
            candidate(
                person.id,
                name,
                has_account=False,
                title=person.title,
                department=person.department,
                about=person.about,
                observed=obs,
                channels=person.seen.channels,
                tasks_done=done.get(person.external_id, []),
                tasks_open=open_.get(person.external_id, []),
            )
        )
        info[person.id] = {"name": name, "email": "", "has_account": False, "title": person.title}
    return out, info


def locate_experts(
    ctx: Ctx,
    orgs: OrgStore,
    lib: Library,
    people: PeopleBook,
    guilds: Any,
    model_call: Callable[[str], dict],
    need: str,
    position_id: str | None = None,
    exclude: str = "",
) -> ExpertResult:
    from salid.expertise import rank

    org = orgs.get(ctx.org.id) or ctx.org
    pos = next((x for x in org.positions if x.id == position_id), None) if position_id else None
    if position_id and pos is None:
        raise HTTPException(404, "Unknown position")
    need = need.strip() or (f"Who should take the position {pos.title}?" if pos else "")
    if not need:
        raise HTTPException(422, "Say what you need someone for, or pick a position")
    cands, info = expertise_candidates(ctx, orgs, lib, people, exclude=exclude)
    ws = company_workspace(org)
    lang = company_language(guilds, ws) if ws else "en"
    r = rank(
        need, cands, model_call=model_call, lang=lang, position=pos.model_dump() if pos else None
    )
    picks = [
        ExpertPick(
            **{**info[x.key], "key": x.key},
            rank=x.rank,
            reason=x.reason,
            evidence=x.evidence,
            confidence=x.confidence,
        )
        for x in r.picks
        if x.key in info
    ]
    return ExpertResult(need=need, position=pos, picks=picks, gaps=r.gaps, candidates=len(cands))


@router.post("/orgs/{org_id}/experts")
async def find_experts(
    body: ExpertBody,
    ctx: Ctx = Depends(member_ctx),
    orgs: OrgStore = Depends(get_orgs),
    lib: Library = Depends(get_library),
    people: PeopleBook = Depends(get_people),
    guilds=Depends(hub.get_guild_store),
    model_call: Callable[[str], dict] = Depends(get_expert_model),
) -> ExpertResult:
    """Expertise locator: the best person (and backups) for a need or a position, each with
    the reasons and the evidence from Salid's records."""
    try:
        return await run_in_threadpool(
            locate_experts,
            ctx,
            orgs,
            lib,
            people,
            guilds,
            model_call,
            body.need,
            body.position_id,
            body.exclude or "",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("expertise locator failed")
        raise HTTPException(502, "Salid could not look for people right now") from exc


# ------------------------------------------------------------------ workspaces
@router.get("/orgs/{org_id}/workspaces")
async def list_workspaces(ctx: Ctx = Depends(member_ctx)) -> list[WorkspaceView]:
    return [workspace_view(w) for w in ctx.org.workspaces]


class LinkBody(BaseModel):
    code: str = Field(min_length=10, max_length=200)


@router.post("/orgs/{org_id}/workspaces/discord/link")
async def link_discord(
    body: LinkBody,
    ctx: Ctx = Depends(admin_ctx),
    orgs: OrgStore = Depends(get_orgs),
    guilds=Depends(hub.get_guild_store),
    lib: Library = Depends(get_library),
) -> WorkspaceView:
    """Redeem a `/ตั้งค่า` code and attach that Discord server to this org. 409 when the server
    already belongs to another org. The Google account also becomes a hub admin of the server."""
    code = body.code.strip()

    def run() -> Workspace:
        try:
            gid = guilds.peek_link_code(code)
        except LinkError as exc:
            raise HTTPException(400, "Link code is invalid, expired or already used") from exc
        existing = orgs.org_of_guild(gid)
        if existing is not None and existing.id != ctx.org.id:
            if not (existing.grandfathered and not existing.members):
                raise HTTPException(409, "This server already belongs to another organization")
            # an unclaimed auto-created shell (no admin ever linked): move the server out of it
            orgs.detach_workspace(existing.id, f"discord-{gid}")
        try:
            info = guilds.link_code_info(code)
            _gid, g = guilds.redeem_link_code(code, ctx.email)
        except LinkError as exc:
            raise HTTPException(400, "Link code is invalid, expired or already used") from exc
        # the code was handed to one Discord user: this Google account is that person (verified)
        if info.get("discord_user_id"):
            orgs.set_user_identity(
                ctx.email,
                Identity(
                    platform="discord",
                    external_id=str(info["discord_user_id"]),
                    name=str(info.get("discord_user_name") or ""),
                    verified=True,
                ),
            )
        ws = Workspace(
            id=f"discord-{gid}",
            platform="discord",
            external_id=str(gid),
            name=g.guild_name or str(gid),
            icon_url=g.icon_url,
            status="connected",
            connected_at=stamp(),
            connected_by=ctx.email,
        )
        try:
            org = orgs.attach_workspace(ctx.org.id, ws)
        except WorkspaceTaken as exc:
            raise HTTPException(409, "This server already belongs to another organization") from exc
        lib.reconcile_shared_docs(org)
        return ws

    return workspace_view(await run_in_threadpool(run))


@router.delete("/orgs/{org_id}/workspaces/{ws_id}")
async def delete_workspace(
    ws_id: str,
    ctx: Ctx = Depends(admin_ctx),
    orgs: OrgStore = Depends(get_orgs),
    lib: Library = Depends(get_library),
) -> list[WorkspaceView]:
    def run() -> Org:
        try:
            org = orgs.detach_workspace(ctx.org.id, ws_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown workspace") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        lib.reconcile_shared_docs(org)
        return org

    org = await run_in_threadpool(run)
    return [workspace_view(w) for w in org.workspaces]


# ------------------------------------------------------------------ sharing
class ShareBody(BaseModel):
    from_workspace: str
    to_workspace: str
    categories: list[str] = Field(min_length=1)


@router.get("/orgs/{org_id}/sharing")
async def list_sharing(ctx: Ctx = Depends(member_ctx)) -> list[ShareLink]:
    return ctx.org.sharing


@router.post("/orgs/{org_id}/sharing")
async def put_sharing(
    body: ShareBody,
    ctx: Ctx = Depends(admin_ctx),
    orgs: OrgStore = Depends(get_orgs),
    lib: Library = Depends(get_library),
) -> list[ShareLink]:
    """Create or replace the link "`to_workspace` may read `categories` of `from_workspace`"."""
    cats = [c for c in dict.fromkeys(body.categories)]
    if any(c not in CATEGORIES for c in cats):
        raise HTTPException(422, f"categories must be among {', '.join(CATEGORIES)}")
    if body.from_workspace == body.to_workspace:
        raise HTTPException(422, "Choose two different workspaces")

    def run() -> Org:
        with orgs.lock:
            org = orgs.get(ctx.org.id)
            a, b = org.workspace(body.from_workspace), org.workspace(body.to_workspace)
            if a is None or b is None:
                raise HTTPException(404, "Unknown workspace")
            if a.status != "connected" or b.status != "connected":
                raise HTTPException(409, "Both workspaces must be connected")
            org.sharing = [
                s for s in org.sharing if (s.from_workspace, s.to_workspace) != (a.id, b.id)
            ] + [
                ShareLink(
                    from_workspace=a.id,
                    to_workspace=b.id,
                    categories=cats,
                    created_by=ctx.email,
                    created_at=stamp(),
                )
            ]
            orgs.save(org)
        lib.reconcile_shared_docs(org)
        return org

    org = await run_in_threadpool(run)
    return org.sharing


@router.delete("/orgs/{org_id}/sharing")
async def delete_sharing(
    from_workspace: str = Query(),
    to_workspace: str = Query(),
    ctx: Ctx = Depends(admin_ctx),
    orgs: OrgStore = Depends(get_orgs),
    lib: Library = Depends(get_library),
) -> list[ShareLink]:
    def run() -> Org:
        with orgs.lock:
            org = orgs.get(ctx.org.id)
            before = len(org.sharing)
            org.sharing = [
                s
                for s in org.sharing
                if (s.from_workspace, s.to_workspace) != (from_workspace, to_workspace)
            ]
            if len(org.sharing) == before:
                raise HTTPException(404, "No such share link")
            orgs.save(org)
        lib.reconcile_shared_docs(org)
        return org

    return (await run_in_threadpool(run)).sharing


# ------------------------------------------------------------------ license
class LicenseRequestBody(BaseModel):
    plan: str
    seats: int = Field(default=5, ge=1, le=10000)
    note: str = Field(default="", max_length=1000)


class LicenseBody(BaseModel):
    plan: str
    status: str = "active"
    seats: int = Field(default=5, ge=1, le=100000)
    valid_until: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}([T ].*)?$")
    notes: str = Field(default="", max_length=1000)
    billing_provider: str = Field(default="", max_length=40)
    request_id: str | None = None  # mark this request approved


class LicenseResponse(BaseModel):
    license: LicenseView
    requests: list[LicenseRequest]


@router.get("/orgs/{org_id}/license")
async def get_license(
    ctx: Ctx = Depends(member_ctx), orgs: OrgStore = Depends(get_orgs)
) -> LicenseResponse:
    reqs = await run_in_threadpool(orgs.license_requests, ctx.org.id) if ctx.is_exec else []
    return LicenseResponse(license=license_view(ctx.org.license), requests=reqs)


@router.post("/orgs/{org_id}/license/request")
async def request_license(
    body: LicenseRequestBody, ctx: Ctx = Depends(admin_ctx), orgs: OrgStore = Depends(get_orgs)
) -> LicenseRequest:
    """ "Subscribe" without a payment provider: stored for a super admin to approve."""
    if body.plan not in PLANS or body.plan == "trial":
        raise HTTPException(422, "plan must be monthly, yearly or enterprise")
    req = LicenseRequest(
        id=f"lr_{uuid.uuid4().hex[:10]}",
        org_id=ctx.org.id,
        org_name=ctx.org.name,
        plan=body.plan,
        seats=body.seats,
        note=body.note,
        requested_by=ctx.email,
        created_at=stamp(),
    )
    return await run_in_threadpool(orgs.add_license_request, req)


def _approve_until(plan: str, now: datetime) -> str | None:
    days = {"monthly": 31, "yearly": 366}.get(plan)
    return (now + timedelta(days=days)).date().isoformat() if days else None


@router.put("/orgs/{org_id}/license")
async def put_license(
    body: LicenseBody,
    ctx: Ctx = Depends(member_ctx),
    email: str = Depends(super_user),
    orgs: OrgStore = Depends(get_orgs),
) -> LicenseResponse:
    """Super admin only."""
    if body.plan not in PLANS or body.status not in STATUSES:
        raise HTTPException(422, "unknown plan or status")

    def run() -> Org:
        with orgs.lock:
            org = orgs.get(ctx.org.id)
            org.license = License(**body.model_dump(exclude={"request_id"}))
            orgs.save(org)
            if body.request_id:
                req = orgs.get_license_request(body.request_id)
                if req and req.org_id == org.id:
                    req.status, req.decided_by, req.decided_at = "approved", email, stamp()
                    orgs.add_license_request(req)
            return org

    org = await run_in_threadpool(run)
    reqs = await run_in_threadpool(orgs.license_requests, org.id)
    return LicenseResponse(license=license_view(org.license), requests=reqs)


class OrgSummary(BaseModel):
    id: str
    name: str
    owner_email: str
    members: int
    workspaces: int
    license: LicenseView


@router.get("/admin/orgs")
async def admin_orgs(
    _: str = Depends(super_user), orgs: OrgStore = Depends(get_orgs)
) -> list[OrgSummary]:
    found = await run_in_threadpool(orgs.all_orgs)
    return [
        OrgSummary(
            id=o.id,
            name=o.name,
            owner_email=o.owner_email,
            members=len(o.members),
            workspaces=len(o.workspaces),
            license=license_view(o.license),
        )
        for o in found
    ]


@router.get("/admin/license-requests")
async def admin_requests(
    status: str | None = Query(default="pending"),
    _: str = Depends(super_user),
    orgs: OrgStore = Depends(get_orgs),
) -> list[LicenseRequest]:
    rows = await run_in_threadpool(orgs.license_requests)
    return [r for r in rows if status in (None, "", "all") or r.status == status]


class Decision(BaseModel):
    approve: bool
    valid_until: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


@router.post("/admin/license-requests/{req_id}")
async def decide_request(
    req_id: str,
    body: Decision,
    email: str = Depends(super_user),
    orgs: OrgStore = Depends(get_orgs),
) -> LicenseRequest:
    """Approve (activates the requested plan, 31 days / 366 days unless `valid_until`) or reject."""

    def run() -> LicenseRequest:
        with orgs.lock:
            req = orgs.get_license_request(req_id)
            if req is None:
                raise HTTPException(404, "Unknown request")
            if req.status != "pending":
                raise HTTPException(409, "Already decided")
            if body.approve:
                org = orgs.get(req.org_id)
                if org is None:
                    raise HTTPException(404, "Unknown organization")
                org.license = License(
                    plan=req.plan,
                    status="active",
                    seats=req.seats,
                    valid_until=body.valid_until or _approve_until(req.plan, datetime.now(UTC)),
                    notes=f"approved {req.id}",
                )
                orgs.save(org)
            req.status = "approved" if body.approve else "rejected"
            req.decided_by, req.decided_at = email, stamp()
            return orgs.add_license_request(req)

    return await run_in_threadpool(run)


# ------------------------------------------------------------------ documents
class DocPage(BaseModel):
    items: list[DocItem]
    next_cursor: str | None
    total: int


def _viewer(ctx: Ctx) -> Viewer:
    return Viewer(ctx.email, ctx.is_exec)


@router.get("/orgs/{org_id}/documents")
async def list_documents(
    category: str | None = Query(default=None),
    kind: str | None = Query(default=None, pattern="^(file|knowledge)$"),
    workspace: str | None = Query(default=None),
    q: str | None = Query(default=None, max_length=200),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=30, ge=1, le=100),
    group: str | None = Query(default=None, pattern="^(company|legal|finance|hr|other)$"),
    ctx: Ctx = Depends(member_ctx),
    lib: Library = Depends(get_library),
) -> DocPage:
    """Documents of the org. `group` narrows FILES to one bucket of what they are for the
    company (company records / legal / finance / hr / other)."""
    if category and category not in LIB_CATEGORIES:
        raise HTTPException(422, f"category must be among {', '.join(LIB_CATEGORIES)}")
    page = await run_in_threadpool(
        lambda: lib.list(
            ctx.org,
            _viewer(ctx),
            category=category,
            workspace=workspace,
            kind=kind,
            q=q,
            cursor=cursor,
            limit=limit,
            group=group,
        )
    )
    return DocPage(items=page.items, next_cursor=page.next_cursor, total=page.total)


@router.get("/orgs/{org_id}/documents/groups")
async def document_groups(
    workspace: str | None = Query(default=None),
    ctx: Ctx = Depends(member_ctx),
    lib: Library = Depends(get_library),
) -> dict[str, int]:
    """How many files each group holds (company records first)."""
    return await run_in_threadpool(lib.group_counts, ctx.org, _viewer(ctx), workspace)


class GroupBody(BaseModel):
    group: str = Field(pattern="^(company|legal|finance|hr|other)$")


@router.put("/orgs/{org_id}/documents/{doc_id}/group")
async def set_document_group(
    doc_id: str,
    body: GroupBody,
    ctx: Ctx = Depends(admin_ctx),
    lib: Library = Depends(get_library),
) -> dict:
    """An owner / admin files a document under another group (e.g. a shareholder agreement
    the keywords put under Legal goes to Company records)."""
    ok = await run_in_threadpool(lib.set_group, ctx.org, doc_id, body.group)
    if not ok:
        raise HTTPException(404, "No such file")
    return {"id": doc_id, "group": body.group}


@router.post("/orgs/{org_id}/documents/reindex")
async def reindex_documents(
    ctx: Ctx = Depends(admin_ctx), lib: Library = Depends(get_library)
) -> dict:
    """Rebuild the search notes of web uploads (memory retention drops old ones)."""
    n = await run_in_threadpool(lib.reconcile_shared_docs, ctx.org)
    return {"indexed": n}


@router.post("/orgs/{org_id}/documents")
async def upload_document(
    file: UploadFile = File(),
    ctx: Ctx = Depends(member_ctx),
    lib: Library = Depends(get_library),
) -> DocItem:
    """Multipart upload (field `file`, <= 20 MB: PDF, Office, text, images). Flash-Lite reads,
    categorises (`tags`) and summarises it; Salid can then find it for this org."""
    from salid.docs import DocUnreadable
    from salid.library import MAX_UPLOAD_BYTES

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (20 MB max)")
    name = re.sub(r"[\x00-\x1f/\\]", "_", file.filename or "upload")[:200]
    mime = (file.content_type or "application/octet-stream").split(";")[0]
    try:
        return await run_in_threadpool(lib.upload, ctx.org, ctx.email, name, data, mime)
    except DocUnreadable as exc:
        raise HTTPException(415, "Unsupported or unreadable file type") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("upload failed")
        raise HTTPException(502, "Could not process the file right now") from exc


@router.get("/orgs/{org_id}/documents/{doc_id}")
async def get_document(
    doc_id: str,
    workspace: str | None = Query(default=None),
    ctx: Ctx = Depends(member_ctx),
    lib: Library = Depends(get_library),
):
    detail = await run_in_threadpool(lib.get, ctx.org, _viewer(ctx), doc_id, workspace)
    if detail is None:
        raise HTTPException(404, "Document not found")
    return detail


@router.get("/orgs/{org_id}/documents/{doc_id}/file")
async def download_document(
    doc_id: str, ctx: Ctx = Depends(member_ctx), lib: Library = Depends(get_library)
) -> Response:
    """Original bytes of a web upload (fetch with the Bearer header, then save as a blob)."""
    detail = await run_in_threadpool(lib.get, ctx.org, _viewer(ctx), doc_id, None)
    if detail is None or detail.category != "uploads":
        raise HTTPException(404, "Document not found")
    got = await run_in_threadpool(lib.read_upload, ctx.org, doc_id)
    if got is None:
        raise HTTPException(404, "Document not found")
    rec, data = got
    return Response(
        data,
        media_type=rec.get("mime") or "application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{rec["id"]}"',
            "Cache-Control": "no-store",
        },
    )


# ------------------------------------------------------------------ overview
@router.get("/orgs/{org_id}/overview")
async def overview(
    ctx: Ctx = Depends(member_ctx),
    lib: Library = Depends(get_library),
    orgs: OrgStore = Depends(get_orgs),
    connectors: dict = Depends(get_connectors),
) -> dict:
    """The organization's profile card: documents, platforms (org-level connections), members
    (how many have linked their own accounts), license, recent activity."""
    available = [p for p in ("discord", "slack", "clickup") if connectors[p].enabled()]

    def run() -> dict:
        # The company profile only: no document scan here (it took 5-7 s on a real company;
        # documents have their own page)
        profiles = orgs.member_profiles(ctx.org)
        linked_members = sum(1 for p in profiles.values() if p.identities)
        by_platform: dict[str, int] = defaultdict(int)
        for w in ctx.org.workspaces:
            by_platform[w.platform] += 1
        lic = license_view(ctx.org.license)
        rank = {"owner": 0, "admin": 1, "member": 2}
        leaders = [
            {
                "email": m.email,
                "role": m.role,
                "name": (profiles.get(m.email) or UserProfile(email=m.email)).name,
                "full_name": (profiles.get(m.email) or UserProfile(email=m.email)).full_name,
                "title": (profiles.get(m.email) or UserProfile(email=m.email)).title,
                "department": (profiles.get(m.email) or UserProfile(email=m.email)).department,
            }
            for m in sorted(ctx.org.members, key=lambda m: (rank.get(m.role, 9), m.email))
            if m.role in ("owner", "admin")
        ]
        depts = sorted({p.department for p in profiles.values() if p.department}, key=str.casefold)
        return {
            "org": {
                "id": ctx.org.id,
                "name": ctx.org.name,
                "logo_url": ctx.org.logo_url,
                "profile": ctx.org.profile.model_dump(),
                "created_at": ctx.org.created_at,
            },
            "leadership": leaders,
            "departments": depts,
            "headcount": len(ctx.org.members),
            "workspaces": {
                "total": len(ctx.org.workspaces),
                "connected": sum(1 for w in ctx.org.workspaces if w.status == "connected"),
                "by_platform": dict(by_platform),
            },
            "platforms": {
                "available": available,
                "connected": sorted(
                    {
                        w.platform
                        for w in ctx.org.workspaces
                        if w.platform != "web" and w.status == "connected"
                    }
                ),
                "coming_soon": list(COMING_SOON),
            },
            "members": len(ctx.org.members),
            "members_linked": linked_members,
            "license": lic.model_dump(),
            "license_days_left": lic.days_left,
        }

    return await run_in_threadpool(run)


# ------------------------------------------------------------------ tools Salid has on the web
def web_tools(ctx: Ctx, orgs: OrgStore, guilds: Any, ws: Workspace) -> list:
    """Agent tools for the command-center chat: personal ones for everyone (remember who I am,
    my language) and, for owners / admins talking about a Discord workspace, the server
    settings the bot also exposes (saved in the guild document, picked up by the bot within a
    minute, so the change reaches Discord)."""
    from salid.chat_tools import ChatActions, build_chat_tools
    from salid.i18n import LANGS

    email = ctx.email

    def update_my_profile(
        name: str = "",
        title: str = "",
        department: str = "",
        about: str = "",
        manager: str = "",
        first_name: str = "",
        last_name: str = "",
        job_description: str = "",
    ) -> dict:
        """Remember who the person talking to you is: their display name (nickname), real
        first / last name, job title (e.g. CTO), job_description (their full JD, any length),
        department
        (e.g. Engineering), what they do / are responsible for, and `manager` = who they report
        to (a member's name or email; "none" = top of the chart). Call it whenever they tell
        you any of these; leave a field empty to keep its current value. Other members and the
        company directory / org chart see title, department and manager."""
        p = orgs.set_user_profile(
            email,
            name=name or None,
            first_name=first_name or None,
            last_name=last_name or None,
            title=title or None,
            department=department or None,
            about=about or None,
            job_description=job_description or None,
        )
        out: dict = {
            "status": "ok",
            "profile": {
                "name": p.name,
                "full_name": p.full_name,
                "title": p.title,
                "department": p.department,
                "about": p.about,
            },
        }
        if manager:
            want = manager.strip().casefold()
            if want in ("none", "nobody", "-", "ไม่มี"):
                orgs.set_reports_to(ctx.org.id, email, None)
                out["reports_to"] = None
            else:
                profiles = orgs.member_profiles(ctx.org)
                hit = [
                    m.email
                    for m in ctx.org.members
                    if m.email.casefold() == want
                    or m.email.split("@")[0].casefold() == want
                    or (profiles.get(m.email) and profiles[m.email].name.casefold() == want)
                ]
                if len(hit) != 1:
                    out["manager_error"] = (
                        f"cannot tell which member {manager!r} is; ask for their name or email"
                    )
                else:
                    try:
                        orgs.set_reports_to(ctx.org.id, email, hit[0])
                        out["reports_to"] = hit[0]
                    except ValueError as exc:
                        out["manager_error"] = str(exc)
        return out

    def update_member_profile(
        member: str,
        title: str = "",
        department: str = "",
        about: str = "",
        manager: str = "",
        first_name: str = "",
        last_name: str = "",
        job_description: str = "",
        nickname: str = "",
    ) -> dict:
        """(Owners and admins only.) Set who ANOTHER member of the company is: `member` = their
        email (best) or name / nickname in any script; when it does not match, the result lists
        the members: pick the right one yourself and call again with the email. `manager` "me"
        = the person talking to you; job title (e.g. CEO, CTO), department, what they do, real first / last
        name, and `manager` = who they report to (name / email, "none" = top of the chart).
        Call it when the admin tells you somebody's role ("Jems is our CEO"). Leave a field
        empty to keep it. The directory and org chart show the result at once."""
        if not ctx.is_exec:
            return {"error": "only an owner or admin can edit other people's profiles"}
        target = find_member(ctx.org, orgs, member)
        if target is None:
            return {
                "error": f"no unique member matches {member!r}",
                "members": member_choices(ctx.org, orgs),
                "hint": "pick the right person from `members` (names may be written in "
                "another script or as a nickname) and call again with their email; ask the "
                "user only if it is truly unclear",
            }
        p = orgs.set_user_profile(
            target,
            name=nickname or None,
            first_name=first_name or None,
            last_name=last_name or None,
            title=title or None,
            department=department or None,
            about=about or None,
            job_description=job_description or None,
        )
        out: dict = {
            "status": "ok",
            "member": target,
            "profile": {
                "full_name": p.full_name,
                "title": p.title,
                "department": p.department,
                "about": p.about,
            },
        }
        if manager:
            want = manager.strip().casefold()
            me_words = ("me", "myself", "ผม", "ฉัน", "ดิฉัน", "เรา", "หนู")
            if want in ("none", "nobody", "-", "ไม่มี"):
                boss = None
            elif want in me_words:
                boss = ctx.email
            else:
                boss = find_member(ctx.org, orgs, manager)
            if want not in ("none", "nobody", "-", "ไม่มี") and boss is None:
                out["manager_error"] = f"no unique member matches {manager!r}"
                out["members"] = member_choices(ctx.org, orgs)
            else:
                try:
                    orgs.set_reports_to(ctx.org.id, target, boss)
                    out["reports_to"] = boss
                except ValueError as exc:
                    out["manager_error"] = str(exc)
        return out

    def set_my_language(language: str) -> dict:
        """Change the language of the person's own web pages (th or en). Only for them."""
        try:
            return {"status": "ok", "language": orgs.set_user_language(email, language)}
        except ValueError:
            return {"error": "language must be th or en"}

    tools: list = [update_my_profile, set_my_language]
    if ctx.is_exec:
        tools.append(update_member_profile)
    if ctx.role in ("owner", "admin", "super"):

        def update_company_profile(
            tagline: str = "",
            about: str = "",
            industry: str = "",
            founded: str = "",
            location: str = "",
            website: str = "",
            email: str = "",
            phone: str = "",
            size: str = "",
            registered_name: str = "",
            registration_no: str = "",
            tax_id: str = "",
            registered_capital: str = "",
            registered_address: str = "",
            directors: str = "",
            objectives: str = "",
        ) -> dict:
            """Fill in or change the COMPANY's profile (admins only): one-line tagline, about
            text, industry, founded (registration date or year), location, website, contact
            email, phone, team size, and the LEGAL ENTITY as on the registration papers:
            registered_name, registration_no (juristic person number), tax_id,
            registered_capital, registered_address, directors (authorised directors and the
            signing rule), objectives (business objectives). Fill these from an uploaded
            registration certificate / articles. Leave a field empty to keep it; say what you
            changed, and ask before overwriting a value that differs."""
            org = orgs.set_company_profile(
                ctx.org.id,
                **{
                    k: (v or None)
                    for k, v in {
                        "tagline": tagline,
                        "about": about,
                        "industry": industry,
                        "founded": founded,
                        "location": location,
                        "website": website,
                        "email": email,
                        "phone": phone,
                        "size": size,
                        "registered_name": registered_name,
                        "registration_no": registration_no,
                        "tax_id": tax_id,
                        "registered_capital": registered_capital,
                        "registered_address": registered_address,
                        "directors": directors,
                        "objectives": objectives,
                    }.items()
                },
            )
            return {"status": "ok", "profile": org.profile.model_dump()}

        tools.append(update_company_profile)

        def save_position(
            title: str,
            department: str = "",
            description: str = "",
            responsibilities: str = "",
            skills: str = "",
            policies: str = "",
            holder: str = "",
            reports_to: str = "",
            position_id: str = "",
        ) -> dict:
            """(Owners / admins.) Create a job position ahead of time, or update one
            (`position_id`, or the same `title`): job description, responsibilities, skills
            (comma separated), the company policies that apply to it, `holder` = a member who
            holds it (name or email; "none" = vacant), `reports_to` = another position's title.
            Organisational knowledge: used to answer who does what and to find successors."""
            org = orgs.get(ctx.org.id) or ctx.org
            pid = position_id or next(
                (x.id for x in org.positions if x.title.casefold() == title.strip().casefold()),
                None,
            )
            fields: dict = {
                "title": title or None,
                "department": department or None,
                "summary": description or None,
                "responsibilities": responsibilities or None,
                "policies": policies or None,
                "skills": [x.strip() for x in skills.split(",") if x.strip()] if skills else None,
            }
            if holder:
                if holder.strip().casefold() in ("none", "vacant", "ว่าง", "ไม่มี"):
                    fields["holders"] = []
                else:
                    who = find_member(org, orgs, holder)
                    if who is None:
                        return {
                            "error": f"no unique member matches {holder!r}",
                            "members": member_choices(org, orgs),
                        }
                    cur = next((x for x in org.positions if x.id == pid), None)
                    fields["holders"] = list(dict.fromkeys([*(cur.holders if cur else []), who]))
            if reports_to:
                boss = next(
                    (
                        x
                        for x in org.positions
                        if x.title.casefold() == reports_to.strip().casefold()
                    ),
                    None,
                )
                if boss is None:
                    return {
                        "error": f"no position titled {reports_to!r}",
                        "positions": [x.title for x in org.positions],
                    }
                fields["reports_to"] = boss.id
            try:
                pos = orgs.save_position(ctx.org.id, by=ctx.email, pid=pid, **fields)
            except (KeyError, ValueError) as exc:
                return {"error": str(exc)}
            return {
                "status": "ok",
                "position": pos.model_dump(exclude={"created_at", "updated_by"}),
            }

        tools.append(save_position)

    def list_positions_tool(query: str = "") -> dict:
        """The company's job positions (filled or VACANT) with their job description,
        responsibilities, skills, policies and who holds them. `query` filters by words in the
        title / department / description. Use it for "what does the X role do", "which roles
        are open", "what are the rules for this job"."""
        org = orgs.get(ctx.org.id) or ctx.org
        q = query.strip().casefold()
        rows = [
            x
            for x in org.positions
            if not q or q in f"{x.title} {x.department} {x.summary} {x.responsibilities}".casefold()
        ]
        return {
            "positions": [
                {
                    **x.model_dump(exclude={"created_at", "updated_by", "updated_at"}),
                    "vacant": x.vacant,
                }
                for x in rows[:40]
            ]
        }

    list_positions_tool.__name__ = "list_positions"
    tools.append(list_positions_tool)

    def find_expert(need: str, position: str = "", exclude: str = "") -> dict:
        """Expertise locator: who in the company is best for a need ("who knows our tax
        filing", "who can take over Jems' tasks", "who could fill the Sales Lead role"),
        with a primary pick, backups, the reasons and the evidence (profiles, positions, what
        Salid observed, finished tasks). `position` = a position title to match against its
        job description; `exclude` = the person being replaced. Present the picks WITH the
        reasons; say when the evidence is thin."""
        org = orgs.get(ctx.org.id) or ctx.org
        pid = None
        if position:
            pid = next(
                (x.id for x in org.positions if x.title.casefold() == position.strip().casefold()),
                None,
            )
        lib = get_library()
        r = locate_experts(
            ctx, orgs, lib, get_people(), guilds, get_expert_model(), need, pid, exclude
        )
        return r.model_dump(exclude={"position"})

    tools.append(find_expert)
    if ws.platform != "discord" or ctx.role not in ("owner", "admin", "super"):
        return tools
    gid = ws.memory_id
    calls: list[tuple[str, dict]] = []

    def set_server_lang(language: str, apply_to_all_channels: bool = True) -> dict:
        if language not in LANGS or language == "auto":
            return {"error": "unknown language", "languages": [x for x in LANGS if x != "auto"]}
        guilds.set_server_language(gid, language, bool(apply_to_all_channels))
        return {
            "status": "ok",
            "language": language,
            "applied_to_all_channels": bool(apply_to_all_channels),
            "note": "saved for the Discord server too; the bot switches within a minute",
        }

    def set_days(days: int) -> dict:
        from salid.guild_settings import (
            MAX_RETENTION_DAYS,
            MIN_RETENTION_DAYS,
            valid_retention_days,
        )

        if valid_retention_days(days) is None:
            return {"error": "bad_days", "min": MIN_RETENTION_DAYS, "max": MAX_RETENTION_DAYS}
        guilds.set_memory_retention_days(gid, days)
        return {"status": "ok", "memory_days": days}

    def show() -> dict:
        g = guilds.get(gid)
        return {
            "server": g.guild_name or str(gid),
            "default_language": g.default_language,
            "default_tone": g.default_tone,
            "memory_days": g.memory_retention_days,
            "note": "you are an administrator: confirm before changing server-wide settings",
        }

    actions = ChatActions(
        set_server_language=set_server_lang, set_memory_days=set_days, show_settings=show
    )
    return tools + build_chat_tools(actions, calls) + discord_action_tools(ctx, orgs.docs, ws)


def _action_queue(docs: Any, org_id: str) -> ActionQueue:
    """The web -> Discord action queue of the center; every action queued through it is a monitor
    event in the org's change log while `MONITOR_ENABLED` is on. The hub's
    `announce_language_change` queues without this hook (it knows only the guild): its action
    first shows up through the bot's done / failed event or the page's 60 s heal."""
    hook = _monitor_hook(docs, org_id, adapt_action)
    return ActionQueue(docs, on_change=(lambda rec: hook([rec])) if hook else None)


def discord_action_tools(ctx: Ctx, docs: Any, ws: Workspace) -> list:
    """Act on the company's Discord from the web (admins): the bot carries the action out
    within seconds (salid.actions). Each tool returns an action id to check with
    `discord_action_status`."""
    q = _action_queue(docs, ctx.org.id)
    gid = ws.memory_id

    def _queue(kind: str, args: dict) -> dict:
        try:
            a = q.enqueue(gid, kind, args, by=ctx.email)
        except ValueError as exc:
            return {"error": str(exc)}
        return {
            "status": "queued",
            "action_id": a["id"],
            "note": "Salid's Discord bot does this within a few seconds; check with discord_action_status",
        }

    def discord_post_message(channel: str, text: str) -> dict:
        """Post a message in a channel of the company's Discord server (channel name or id).
        Confirm the wording with the person first when it is an announcement."""
        return _queue("post_message", {"channel": channel, "text": text})

    def discord_send_dm(user_id: str, text: str) -> dict:
        """Send a Discord direct message to a member (their Discord user id, e.g. from the
        directory). Use it to notify or remind someone personally."""
        return _queue("send_dm", {"user_id": user_id, "text": text})

    def discord_create_event(
        name: str, start: str, end: str = "", where: str = "", description: str = ""
    ) -> dict:
        """Create a scheduled event (meeting) in the company's Discord: `start`/`end` as ISO
        datetimes with timezone (e.g. 2026-10-20T10:00:00+07:00), `where` = a voice channel name
        or a place / link. It appears in everyone's calendar."""
        return _queue(
            "create_event",
            {"name": name, "start": start, "end": end, "where": where, "description": description},
        )

    def discord_create_channel(name: str, kind: str = "text", topic: str = "") -> dict:
        """Create a text or voice channel in the company's Discord server."""
        return _queue("create_channel", {"name": name, "kind": kind, "topic": topic})

    def discord_action_status(action_id: str) -> dict:
        """Result of a queued Discord action: pending, done (with ids/links) or failed (why)."""
        a = q.get(gid, action_id)
        if not a:
            return {"error": "unknown action"}
        return {k: a.get(k) for k in ("id", "kind", "status", "result", "error", "created_at")}

    return [
        discord_post_message,
        discord_send_dm,
        discord_create_event,
        discord_create_channel,
        discord_action_status,
    ]


class ActionBody(BaseModel):
    kind: str
    args: dict = Field(default_factory=dict)


@router.get("/orgs/{org_id}/actions")
async def list_actions(ctx: Ctx = Depends(admin_ctx), lib: Library = Depends(get_library)) -> dict:
    """Recent web -> Discord actions of the company's server (admins)."""
    ws = company_workspace(ctx.org)
    if ws is None or ws.platform != "discord":
        return {"actions": []}
    q = ActionQueue(lib.docs)
    return {"actions": await run_in_threadpool(q.recent, ws.memory_id)}


@router.post("/orgs/{org_id}/actions")
async def queue_action(
    body: ActionBody, ctx: Ctx = Depends(admin_ctx), lib: Library = Depends(get_library)
) -> dict:
    """Queue an action for the Discord bot (admins): post_message, send_dm, create_event,
    create_channel, edit_event, delete_event."""
    ws = company_workspace(ctx.org)
    if ws is None or ws.platform != "discord":
        raise HTTPException(404, "No Discord server connected")
    try:
        a = await run_in_threadpool(
            _action_queue(lib.docs, ctx.org.id).enqueue,
            ws.memory_id,
            body.kind,
            body.args,
            by=ctx.email,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return a


class DirectoryPerson(BaseModel):
    email: str
    name: str = ""
    full_name: str = ""
    title: str = ""
    department: str = ""
    role: str
    linked_platforms: int = 0
    platforms: list[str] = Field(default_factory=list)


class DirectoryGroup(BaseModel):
    department: str  # "" = not assigned yet
    people: list[DirectoryPerson]


class StructureBody(BaseModel):
    reports_to: str | None = Field(default=None, max_length=200)


@router.put("/orgs/{org_id}/members/{email}/structure")
async def set_structure(
    email: str,
    body: StructureBody,
    ctx: Ctx = Depends(member_ctx),
    orgs: OrgStore = Depends(get_orgs),
) -> dict:
    """Place a person in the org chart (who they report to). Admins for anyone; a member only
    for themselves."""
    target = email.lower()
    if target != ctx.email and ctx.role not in ("owner", "admin", "super"):
        raise HTTPException(403, "Only an admin can change other people's place in the chart")
    try:
        await run_in_threadpool(orgs.set_reports_to, ctx.org.id, target, body.reports_to)
    except KeyError as exc:
        raise HTTPException(404, "Unknown member") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"email": target, "reports_to": (body.reports_to or "").lower() or None}


class MemberProfileBody(ProfileBody):
    reports_to: str | None = None  # "" = top of the chart; omitted = unchanged


def find_member(org: Org, orgs: OrgStore, who: str) -> str | None:
    """A member's email from an email, the part before the @, a display / real / first name or
    nickname (case-insensitive; a unique partial match counts too). None when nobody or more
    than one matches: callers then show `member_choices` so the model picks the email."""
    want = (who or "").strip().casefold().lstrip("@")
    if not want:
        return None
    profiles = orgs.member_profiles(org)
    names: dict[str, set[str]] = {}
    for m in org.members:
        p = profiles.get(m.email)
        ns = {m.email.casefold(), m.email.split("@")[0].casefold()}
        if p:
            ns |= {p.name.casefold(), p.full_name.casefold(), p.first_name.casefold()}
            ns |= {i.name.casefold() for i in p.identities if i.name}
        names[m.email] = ns - {""}
    exact = [e for e, ns in names.items() if want in ns]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None
    part = [
        e for e, ns in names.items() if any(want in n or (len(n) > 2 and n in want) for n in ns)
    ]
    return part[0] if len(part) == 1 else None


def member_choices(org: Org, orgs: OrgStore) -> list[dict]:
    """Every member with the names they go by, so the model can match a nickname in another
    script (e.g. Thai "เจมส์" = "Jems") and call again with the email."""
    profiles = orgs.member_profiles(org)
    out = []
    for m in org.members:
        p = profiles.get(m.email)
        out.append(
            {
                "email": m.email,
                "name": p.name if p else "",
                "full_name": p.full_name if p else "",
                "title": p.title if p else "",
                "accounts": [i.name for i in p.identities if i.name] if p else [],
                "role": m.role,
            }
        )
    return out


@router.put("/orgs/{org_id}/members/{email}/profile")
async def set_member_profile(
    email: str,
    body: MemberProfileBody,
    ctx: Ctx = Depends(member_ctx),
    orgs: OrgStore = Depends(get_orgs),
) -> MemberView:
    """Who a member is: job title (CEO...), department, what they do, real name, manager.
    Owners and admins set it for anyone (no request from the person needed); a member only
    for themselves (same as My profile)."""
    target = email.lower()
    if target != ctx.email and not ctx.is_exec:
        raise HTTPException(403, "Only an owner or admin can edit other people's profiles")
    if ctx.org.role_of(target) is None:
        raise HTTPException(404, "Unknown member")

    def run() -> MemberView:
        orgs.set_user_profile(
            target,
            name=body.name,
            first_name=body.first_name,
            last_name=body.last_name,
            title=body.title,
            department=body.department,
            about=body.about,
            job_description=body.job_description,
        )
        if body.reports_to is not None:
            try:
                orgs.set_reports_to(ctx.org.id, target, body.reports_to or None)
            except KeyError as exc:
                raise HTTPException(404, "Unknown manager") from exc
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        org = orgs.get(ctx.org.id)
        m = next(x for x in org.members if x.email.lower() == target)
        return member_view(m, orgs.user_profile(target))

    return await run_in_threadpool(run)


@router.get("/orgs/{org_id}/directory")
async def directory(ctx: Ctx = Depends(member_ctx), orgs: OrgStore = Depends(get_orgs)) -> dict:
    """Who works here, by department (the org chart). Every member may see it."""
    profiles = await run_in_threadpool(orgs.member_profiles, ctx.org)
    rank = {"owner": 0, "admin": 1, "member": 2}
    groups: dict[str, list[DirectoryPerson]] = defaultdict(list)
    for m in ctx.org.members:
        p = profiles.get(m.email) or UserProfile(email=m.email)
        groups[p.department].append(
            DirectoryPerson(
                email=m.email,
                name=p.name,
                full_name=p.full_name,
                title=p.title,
                department=p.department,
                role=m.role,
                linked_platforms=len(p.identities),
                platforms=[i.platform for i in p.identities if i.verified],
            )
        )
    out = []
    for dept in sorted(groups, key=lambda d: (d == "", d.casefold())):
        people = sorted(
            groups[dept], key=lambda x: (rank.get(x.role, 9), (x.name or x.email).casefold())
        )
        out.append(DirectoryGroup(department=dept, people=people))
    # the chart: who reports to whom (a forest; owners first among the roots)
    person = {p.email: p for g in out for p in g.people}
    kids: dict[str | None, list[str]] = defaultdict(list)
    emails = {m.email for m in ctx.org.members}
    for m in ctx.org.members:
        boss = (m.reports_to or "").lower()
        kids[boss if boss in emails and boss != m.email else None].append(m.email)

    def node(e: str, depth: int = 0) -> dict:
        p = person[e]
        children = sorted(kids.get(e, []), key=lambda c: (rank.get(person[c].role, 9), c))
        return {
            **p.model_dump(),
            "reports_to": next((m.reports_to for m in ctx.org.members if m.email == e), None),
            "children": [node(c, depth + 1) for c in children] if depth < 12 else [],
        }

    roots = sorted(kids.get(None, []), key=lambda c: (rank.get(person[c].role, 9), c))
    return {
        "org": {"id": ctx.org.id, "name": ctx.org.name},
        "departments": [g.model_dump() for g in out],
        "tree": [node(r) for r in roots],
        "members": len(ctx.org.members),
        "with_profile": sum(1 for p in profiles.values() if p.title or p.department),
        "with_manager": sum(1 for m in ctx.org.members if m.reports_to),
    }


# ------------------------------------------------------------------ me across the company
def _my_handles(ctx: Ctx, p: UserProfile) -> set[str]:
    """Names the person may appear under in task lists: display name, Discord name, email."""
    names = {ctx.email.split("@")[0], p.name}
    names |= {i.name for i in p.identities} | {i.external_id for i in p.identities}
    return {n.strip().casefold() for n in names if n and n.strip()}


@router.get("/orgs/{org_id}/me/tasks")
async def my_tasks(
    ctx: Ctx = Depends(member_ctx),
    orgs: OrgStore = Depends(get_orgs),
    lib: Library = Depends(get_library),
) -> dict:
    """The person's own open tasks across every connected Discord server (matched by their
    Discord id or any of their names), nearest due first."""

    def run() -> dict:
        from salid.tasks import TaskRepo

        out = my_tasks_sync(ctx, orgs, lib)
        out.sort(key=lambda x: (x["status"] != "open", x["due"] or "9999", x["title"]))
        # things the company asked SALID to do (executives see them: they are usually theirs)
        salid: list[dict] = []
        if ctx.is_exec:
            repo = TaskRepo(lib.docs)
            for ws in ctx.org.workspaces:
                if ws.platform != "discord" or ws.status != "connected":
                    continue
                for t in repo.list(ws.memory_id, status="open"):
                    if not t.assignee_id and (t.assignee_name or "").strip().casefold() in (
                        "salid",
                        "สลิด",
                    ):
                        salid.append(
                            {
                                "id": t.id,
                                "title": t.title,
                                "due": t.due,
                                "status": t.status,
                                "url": t.source_link,
                                "workspace": ws.id,
                                "workspace_name": ws.name,
                                "platform": "discord",
                            }
                        )
        return {
            "tasks": out,
            "open": sum(1 for x in out if x["status"] == "open"),
            "done": sum(1 for x in out if x["status"] != "open"),
            "salid_tasks": salid,
        }

    return await run_in_threadpool(run)


@router.get("/orgs/{org_id}/me/calendar")
async def my_calendar(
    ctx: Ctx = Depends(member_ctx),
    orgs: OrgStore = Depends(get_orgs),
    lib: Library = Depends(get_library),
    guilds=Depends(hub.get_guild_store),
    g: Any = Depends(lambda orgs=Depends(get_orgs): get_google_oauth(orgs)),
    http: Any = Depends(lambda: get_google_http()),
) -> dict:
    """What is coming up for the person: their reminders (set in Discord or here) and the due
    dates of their tasks, plus the company's meetings and events, grouped by day."""
    out = await run_in_threadpool(calendar_view, ctx, orgs, lib, guilds, True, True)
    return await run_in_threadpool(with_google_events, out, ctx, g, http)


@router.get("/orgs/{org_id}/calendar")
async def company_calendar(
    mine: bool = Query(default=False),  # overlay the caller's personal calendar
    ctx: Ctx = Depends(member_ctx),
    orgs: OrgStore = Depends(get_orgs),
    lib: Library = Depends(get_library),
    guilds=Depends(hub.get_guild_store),
    g: Any = Depends(lambda orgs=Depends(get_orgs): get_google_oauth(orgs)),
    http: Any = Depends(lambda: get_google_http()),
) -> dict:
    """The COMPANY calendar: meetings (held and scheduled), Discord events, project deadlines
    and milestones. `mine=1` overlays the caller's personal items (reminders, their task due
    dates); every item says its `layer` (company | personal)."""
    out = await run_in_threadpool(calendar_view, ctx, orgs, lib, guilds, True, mine)
    return await run_in_threadpool(with_google_events, out, ctx, g, http) if mine else out


GCAL_EVENTS = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
GCAL_BACK_DAYS, GCAL_AHEAD_DAYS, GCAL_MAX = 31, 92, 250


def get_google_http() -> Any:
    """The HTTP client for Google Calendar reads. Overridden in tests."""
    import httpx

    return httpx.Client(timeout=8)


def google_events(
    g: Any, org_id: str, email: str, http: Any, now: datetime | None = None
) -> list[dict]:
    """The person's own Google Calendar (primary) events from a month back to three months
    ahead, as calendar items on the PERSONAL layer. Raises on any failure (the caller keeps the
    rest of the calendar). One retry with a fresh token after a 401."""
    now = now or datetime.now(UTC)
    params = {
        "singleEvents": "true",
        "orderBy": "startTime",
        "maxResults": str(GCAL_MAX),
        "timeMin": (now - timedelta(days=GCAL_BACK_DAYS)).isoformat(timespec="seconds"),
        "timeMax": (now + timedelta(days=GCAL_AHEAD_DAYS)).isoformat(timespec="seconds"),
    }
    r = None
    for attempt in range(2):
        token = g.access_token(org_id, email)
        r = http.get(GCAL_EVENTS, params=params, headers={"Authorization": f"Bearer {token}"})
        if r.status_code != 401 or attempt:
            break
        g.evict(org_id, email)  # a stale cached token: refresh once
    if r.status_code >= 400:
        raise RuntimeError(f"google calendar {r.status_code}")
    out: list[dict] = []
    for ev in r.json().get("items", []):
        if ev.get("status") == "cancelled":
            continue
        start, end = ev.get("start") or {}, ev.get("end") or {}
        out.append(
            {
                "kind": "gcal",
                "layer": "personal",
                "at": start.get("dateTime") or start.get("date", ""),
                "end": end.get("dateTime") or end.get("date", ""),
                "all_day": "date" in start,
                "title": ev.get("summary") or "(busy)",
                "where": ev.get("location", ""),
                "url": ev.get("htmlLink", ""),
                "meet": ev.get("hangoutLink", ""),
                "workspace_name": "Google Calendar",
            }
        )
    return out


def with_google_events(view: dict, ctx: Ctx, g: Any, http: Any) -> dict:
    """Merge the caller's Google Calendar into a calendar view (personal layer). Fail-soft:
    not configured / not connected / a Google error only shows up in `google`."""
    info = {"connected": False, "error": ""}
    try:
        if g is None or not g.enabled() or ctx.org.role_of(ctx.email) is None:
            info["error"] = "not_configured"
        elif not g.is_connected(ctx.org.id, ctx.email):
            info["error"] = "not_connected"
        else:
            info["connected"] = True
            items = google_events(g, ctx.org.id, ctx.email, http)
            days: dict[str, list[dict]] = {
                d["date"]: list(d["items"]) for d in view.get("days", [])
            }
            for it in items:
                if it["at"]:
                    days.setdefault(it["at"][:10], []).append(it)
            view["days"] = [
                {"date": d, "items": sorted(days[d], key=lambda x: x.get("at", ""))}
                for d in sorted(days)
            ]
            info["events"] = len(items)
    except Exception as exc:
        logger.warning("google calendar overlay failed: %s", type(exc).__name__)
        info["error"] = "failed"
    finally:
        with contextlib.suppress(Exception):
            if hasattr(http, "close"):
                http.close()
    view.setdefault("sources", {})["google_calendar"] = info["connected"] and not info["error"]
    view["google"] = info
    return view


def calendar_view(
    ctx: Ctx, orgs: OrgStore, lib: Library, guilds: Any, company: bool, personal: bool
) -> dict:
    """Calendar items by day. Company layer: meetings, Discord events, project deadlines and
    milestones (projects the caller may read). Personal layer: their reminders and task dues."""
    from salid.reminders import KIND as REMINDERS

    def run() -> dict:
        p = orgs.user_profile(ctx.email)
        did = (p.identity("discord") or Identity(platform="discord", external_id="")).external_id
        items: list[dict] = []
        for ws in ctx.org.workspaces:
            if ws.platform != "discord" or ws.status != "connected":
                continue
            if did and personal:
                for r in lib.docs.find(
                    REMINDERS,
                    ws.memory_id,
                    lambda d: d.get("active") and str(d.get("user_id")) == did,
                ):
                    items.append(
                        {
                            "kind": "reminder",
                            "layer": "personal",
                            "at": r.get("next_at", ""),
                            "title": r.get("message", ""),
                            "repeat": r.get("every") or "",
                            "workspace_name": ws.name,
                        }
                    )
        tasks = my_tasks_sync(ctx, orgs, lib) if personal else []
        for t in tasks:
            if t["status"] == "open" and t["due"]:
                items.append(
                    {
                        "kind": "task",
                        "layer": "personal",
                        "at": t["due"],
                        "title": t["title"],
                        "url": t["url"],
                        "workspace_name": t["workspace_name"],
                    }
                )
        # meetings Salid took minutes of, where this person was in the room (their Discord id)
        from salid.coo_store import MINUTES

        public_by_ws = {}
        for ws in ctx.org.workspaces:
            if ws.platform != "discord" or ws.status != "connected":
                continue
            public_by_ws[ws.id] = lib._channel_sets(ws)[0]
            for d in lib.docs.find(MINUTES, ws.memory_id):
                attendees = {str(a) for a in d.get("attendee_ids", [])}
                mine = bool(did) and did in attendees
                visible = ctx.is_exec or int(d.get("voice_channel_id", 0)) in public_by_ws[ws.id]
                if not (mine or visible):
                    continue
                if not company and not mine:
                    continue
                items.append(
                    {
                        "kind": "meeting",
                        "layer": "company",
                        "at": d.get("started_at") or d.get("ended_at", ""),
                        "end": d.get("ended_at", ""),
                        "title": d.get("voice_channel_name") or "Meeting",
                        "mine": mine,
                        "attendees": len(attendees),
                        "summary": " ".join((d.get("summary") or "").split())[:200],
                        "workspace_name": ws.name,
                    }
                )
            # meetings Salid scheduled (chat "นัดประชุม…") and Discord scheduled events the bot
            # tracked: the secretary's meeting records
            try:
                from salid.secretary_store import MEETINGS

                for m in sorted(
                    lib.docs.find(MEETINGS, ws.memory_id), key=lambda x: x.get("start_utc", "")
                ):
                    if m.get("status") in ("cancelled", "canceled"):
                        continue
                    att = {str(a) for a in (m.get("attendee_ids") or [])}
                    items.append(
                        {
                            "kind": "meeting",
                            "layer": "company",
                            "at": m.get("start_utc", ""),
                            "end": m.get("end_utc", ""),
                            "title": m.get("title") or "Meeting",
                            "mine": bool(did) and (did in att or str(m.get("organizer_id")) == did),
                            "attendees": len(att),
                            "url": m.get("event_url", ""),
                            "status": m.get("status", ""),
                            "workspace_name": ws.name,
                        }
                    )
            except Exception:
                logger.exception("could not read scheduled meetings of %s", ws.id)
            # the company's upcoming Discord events (everyone sees them)
            try:
                g = guilds.get(ws.memory_id)
                for ev in getattr(g, "events", []) or []:
                    items.append(
                        {
                            "kind": "event",
                            "layer": "company",
                            "at": ev.get("start", ""),
                            "end": ev.get("end", ""),
                            "title": ev.get("name", ""),
                            "where": ev.get("where", ""),
                            "url": ev.get("url", ""),
                            "workspace_name": ws.name,
                        }
                    )
            except Exception:  # the calendar must never fail on a guild read
                logger.exception("could not read events of %s", ws.id)
        if company:
            items.extend(project_calendar_items(ctx, lib))
        items = [it for it in items if it.get("at")]
        items.sort(key=lambda x: x["at"])
        days: dict[str, list[dict]] = defaultdict(list)
        for it in items:
            days[it["at"][:10]].append(it)
        return {
            "days": [{"date": d, "items": days[d]} for d in sorted(days)],
            "sources": {
                "reminders": True,
                "tasks": True,
                "meetings": True,
                "scheduled_meetings": True,
                "discord_events": True,
                "projects": True,
                "google_calendar": False,
            },
            "layers": [x for x, on in (("company", company), ("personal", personal)) if on],
            "linked_discord": bool(did),
        }

    return run()


def project_calendar_items(ctx: Ctx, lib: Library) -> list[dict]:
    """Deadlines and open milestones of the projects the caller may read (team projects, all
    for owners / admins), on the company layer."""
    from salid.projects import PROJECTS, Project

    out: list[dict] = []
    for ws in ctx.org.workspaces:
        if ws.status != "connected" or ws.platform not in ("discord", "web"):
            continue
        for d in lib.docs.find(PROJECTS, ws.memory_id):
            try:
                p = Project.model_validate(d)
            except Exception:
                continue
            if p.deleted or not (ctx.is_exec or p.visibility == "team"):
                continue
            if p.deadline and p.status not in ("completed", "cancelled"):
                out.append(
                    {
                        "kind": "deadline",
                        "layer": "company",
                        "at": p.deadline,
                        "title": p.name,
                        "workspace_name": ws.name,
                    }
                )
            for m in p.milestones:
                if m.due and m.status != "done":
                    out.append(
                        {
                            "kind": "milestone",
                            "layer": "company",
                            "at": m.due,
                            "title": f"{p.name}: {m.title}",
                            "workspace_name": ws.name,
                        }
                    )
    return out


def my_tasks_sync(ctx: Ctx, orgs: OrgStore, lib: Library) -> list[dict]:
    from salid.tasks import TaskRepo

    p = orgs.user_profile(ctx.email)
    handles = _my_handles(ctx, p)
    did = (p.identity("discord") or Identity(platform="discord", external_id="")).external_id
    repo = TaskRepo(lib.docs)
    out = []
    for ws in ctx.org.workspaces:
        if ws.platform != "discord" or ws.status != "connected":
            continue
        for t in repo.list(ws.memory_id, status="all"):  # done ones too: "what did I finish"
            mine = (did and str(t.assignee_id) == did) or (
                t.assignee_name and t.assignee_name.strip().casefold() in handles
            )
            if mine:
                out.append(
                    {
                        "id": t.id,
                        "title": t.title,
                        "due": t.due,
                        "status": t.status,
                        "url": t.source_link,
                        "workspace": ws.id,
                        "workspace_name": ws.name,
                        "platform": "discord",
                    }
                )
    return out


@router.get("/orgs/{org_id}/server-summary")
async def server_summary(
    ctx: Ctx = Depends(member_ctx), guilds=Depends(hub.get_guild_store)
) -> dict:
    """What a regular member may know about the company's Discord servers: name, language,
    tone and which Salid features are on. No controls (those are the admins' Server settings)."""

    def run() -> dict:
        rows = []
        for ws in ctx.org.workspaces:
            if ws.platform != "discord" or ws.status != "connected":
                continue
            g = guilds.get(ws.memory_id)
            feats = (
                g.features.model_dump()
                if hasattr(g.features, "model_dump")
                else dict(g.features or {})
            )
            rows.append(
                {
                    "workspace": ws.id,
                    "name": g.guild_name or ws.name,
                    "icon_url": g.icon_url,
                    "default_language": g.default_language,
                    "default_tone": g.default_tone,
                    "features_on": sorted(k for k, v in feats.items() if v),
                    "channels": len(g.channels or []),
                }
            )
        return {"servers": rows, "can_manage": ctx.role in ("owner", "admin", "super")}

    return await run_in_threadpool(run)


# ------------------------------------------------------------------ voice input (any language)
def transcribe_web_audio(data: bytes, mime: str, target_lang: str):
    """Blocking: one utterance from the browser -> `salid.web_voice.Heard`. Overridden in tests."""
    from salid.llm import make_client
    from salid.web_voice import transcribe_utterance

    s, _store = hub.get_salid_runtime()
    return transcribe_utterance(make_client(s), s.gemini_model, data, mime, target_lang)


def get_transcriber() -> Callable[[bytes, str, str], Any]:
    return transcribe_web_audio


@router.post("/orgs/{org_id}/voice/transcribe")
async def voice_transcribe(
    audio: UploadFile = File(...),
    lang: str = Query(default="en", pattern="^(th|en)$"),
    ctx: Ctx = Depends(member_ctx),
    transcriber: Callable[[bytes, str, str], Any] = Depends(get_transcriber),
) -> dict:
    """Speak in any language; get the words back verbatim plus, when needed, in the page's
    language (`translated`). The browser puts `translated or text` into the chat box."""
    if not _chat_allowed(f"{ctx.org.id}:{ctx.email}:voice"):
        raise HTTPException(429, "Too many recordings; try again later")
    data = await audio.read()
    if not data:
        raise HTTPException(422, "empty recording")
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(413, "recording too long")
    try:
        heard = await run_in_threadpool(transcriber, data, audio.content_type or "audio/webm", lang)
    except Exception as exc:
        logger.exception("voice transcription failed")
        raise HTTPException(502, "Could not understand the recording") from exc
    return {
        "text": heard.text,
        "language": heard.language,
        "translated": heard.translated,
        "message": heard.translated or heard.text,
    }


# ------------------------------------------------------------------ who is talking
def who_block(ctx: Ctx, p: UserProfile, page: str | None = None) -> str:
    """Prompt data block: who the web user is (name, role in the org, linked accounts), so the AI
    knows whom it is talking to. Plain data, never instructions."""
    linked = ", ".join(
        f"{i.platform} {i.name or i.external_id}{'' if i.verified else ' (unverified)'}"
        for i in p.identities
    )
    who = p.name or ctx.email
    if p.full_name and p.full_name != p.name:
        who += f" ({p.full_name})"
    if p.title or p.department:
        who += " (" + ", ".join(x for x in (p.title, p.department) if x) + ")"
    lines = [
        f"Talking with: {who} <{ctx.email}>",
        f"Role in {ctx.org.name}: {ctx.role}" + (" (executive)" if ctx.is_exec else ""),
        f"Linked accounts: {linked or 'none yet (signed in with Google only)'}",
    ]
    if p.about:
        lines.append(f"What they do: {p.about}")
    else:
        lines.append(
            "Profile incomplete: when natural, ask who they are (name, job title, department, "
            "what they do) and remember it with update_my_profile."
        )
    if page:
        lines.append(f"They are looking at the {page} page of the command center right now.")
    return "\n".join(lines)


# ------------------------------------------------------------------ chat
class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    workspace_id: str | None = None
    page: str | None = Field(default=None, max_length=80)  # which page the person is looking at
    document_id: str | None = Field(default=None, max_length=200)  # just uploaded in the chat
    document_ids: list[str] = Field(default_factory=list, max_length=10)  # several at once


DOC_BLOCK_TEXT = 20_000  # characters of ONE uploaded file's text Salid reads in the chat
DOC_BLOCK_TOTAL = 45_000  # all files of one chat message together


def uploaded_docs_block(lib: Library, ctx: Ctx, doc_ids: list[str]) -> str:
    """The secretary request for SEVERAL files uploaded together: each file's block, the text
    budget shared (DOC_BLOCK_TOTAL), one set of instructions at the end."""
    per = max(4_000, min(DOC_BLOCK_TEXT, DOC_BLOCK_TOTAL // max(1, len(doc_ids))))
    blocks = [
        b for d in doc_ids if (b := uploaded_doc_block(lib, ctx, d, text_limit=per, tail=False))
    ]
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n" + secretary_instructions(ctx, len(blocks))


def secretary_instructions(ctx: Ctx, n: int) -> str:
    act = (
        "If a file is one of the company's own records (registration certificate, articles of "
        "association / memorandum, company regulations, shareholder papers, licences): call "
        "update_company_profile with every legal-entity fact it states (registered_name, "
        "registration_no, tax_id, registered_capital, registered_address, directors, "
        "objectives, founded = registration date) and, for work rules / working conditions / "
        "regulations, record each rule with teach_company (kind rule). Copy names, numbers "
        "and addresses EXACTLY as written. When several company papers disagree (a scanned "
        "copy can be misread), the registration certificate (หนังสือรับรอง) wins; mention the "
        "difference instead of guessing. "
        if ctx.is_exec
        else "Only owners and admins may change the company profile: if a file is a company "
        "record, say an admin can ask you to update the profile from it. "
    )
    return (
        (
            "Act like the company's secretary for these files. "
            if n > 1
            else "Act like the company's secretary: "
        )
        + "say in one line per file where it was filed (offer to move it if another group fits "
        "better; say so when it was already on file), "
        + act
        + "Then report briefly what you updated. Never invent facts that are not in the text."
    )


def uploaded_doc_block(
    lib: Library, ctx: Ctx, doc_id: str, *, text_limit: int = 0, tail: bool = True
) -> str:
    """What Salid needs to act like a secretary on a file just uploaded in the chat: where it
    was filed, its summary and text, and what to do with a company record. "" when unknown."""
    from salid.library import ORG_DOCS, classify_group

    ws_id, _, key = doc_id.partition(":upload:")
    ws = ctx.org.workspace(ws_id)
    rec = lib.docs.get(ORG_DOCS, ws.memory_id, key) if ws and key else None
    if not rec:
        return ""
    group = rec.get("group") or classify_group(
        rec.get("tag", ""), rec.get("filename", ""), rec.get("summary", "")
    )
    limit = text_limit or DOC_BLOCK_TEXT
    block = (
        "[Document just uploaded in this chat by the person: "
        f"{rec.get('filename', '')} — filed under the Documents group '{group}' "
        f"(id {doc_id}; groups: company records, legal, finance, hr, other). "
        f"Summary: {(rec.get('summary') or '').strip()[:600]}\n"
        f"Text:\n{(rec.get('text') or '').strip()[:limit]}\n]"
    )
    return block + ("\n" + secretary_instructions(ctx, 1) if tail else "")


class ChatReply(BaseModel):
    answer: str
    workspace_id: str
    at: str


class Turn(BaseModel):
    role: str  # user | salid
    text: str
    at: str
    workspace_id: str = ""


_chat_times: dict[str, deque[float]] = defaultdict(deque)


def _chat_allowed(key: str) -> bool:
    now = time.monotonic()
    q = _chat_times[key]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= CHAT_PER_HOUR:
        return False
    q.append(now)
    return True


def company_workspace(org: Org) -> Workspace | None:
    """Where "the company" lives for chat: the first connected Discord server, else the web."""
    discord = [w for w in org.workspaces if w.platform == "discord" and w.status == "connected"]
    return discord[0] if discord else org.web_workspace()


def chat_scope(lib: Library, org: Org, ws: Workspace, is_exec: bool) -> dict[int, str]:
    """Channels (id -> name) the web caller's memory tools may read in `ws`. Decided here."""
    from salid.sharing import sources_in_org

    allowed: dict[int, str] = {}
    if ws.platform == "web":
        allowed[WEB_DOCS_CHANNEL] = "uploads"
    elif ws.platform == "discord":
        g = lib.guilds.find(ws.memory_id)
        if g is not None:
            names = {int(c.id): c.name for c in g.channels}
            public = {int(x) for x in g.public_channel_ids}
            for cid in (public | set(names)) if is_exec else public:
                if cid > 0:
                    allowed[cid] = names.get(cid, str(cid))
        if any(s.platform == "web" for s in sources_in_org(org, ws.id, "documents")):
            allowed[SHARED_DOCS_CHANNEL] = SHARED_DOCS_NAME
    elif ws.platform == "slack":
        for cid in _slack_channel_ids(lib, ws):
            allowed[cid] = "slack"
    return allowed


def _slack_channel_ids(lib: Library, ws: Workspace) -> list[int]:
    # DocStore.find returns docs only; the channel's int id is the doc key = namespaced_id(...)
    from salid.connectors.slack import SLACK_CHANNELS
    from salid.orgs import namespaced_id

    return [
        namespaced_id("slack", d["slack_id"], "channel")
        for d in lib.docs.find(SLACK_CHANNELS, ws.memory_id)
        if d.get("slack_id")
    ]


def answer_question(
    ctx: Ctx,
    orgs: OrgStore,
    lib: Library,
    ask: Callable[..., Any],
    guilds: Any,
    ws: Workspace,
    question: str,
    page: str | None = None,
    language: str | None = None,
) -> str:
    """Blocking: Salid's text brain for one web question, scoped like the chat route (the org's
    workspace, the caller's memory rights, their profile, the web tools). Used by the chat route
    and by the live voice session's ask_salid tool."""
    from salid.qa import MemoryTools

    s, store = hub.get_salid_runtime()
    is_exec = ctx.is_exec and ws.platform == "discord"
    allowed = chat_scope(lib, ctx.org, ws, ctx.is_exec)
    memory = MemoryTools(lib.memory, ws.memory_id, allowed, s) if allowed else None
    turns = orgs.chat_history(ctx.org.id, ctx.email)
    history = "\n".join(
        f"{'ผู้ใช้' if t['role'] == 'user' else 'สลิด'}: {t['text']}"
        for t in turns[-CHAT_HISTORY_PROMPT_TURNS:]
    )
    lang = language or company_language(guilds, ws)
    profile = orgs.user_profile(ctx.email)
    tools = web_tools(ctx, orgs, guilds, ws)
    who = who_block(ctx, profile, page)
    # ONE Salid: the Discord bot's own agent assembly (every feature's tools, company playbook,
    # tone, web search), driven headless with web stand-ins. See salid.headless.
    kw = (
        headless_kwargs(ctx, profile, ws, question, lang, memory)
        if ws.platform == "discord"
        else None
    )
    if kw is not None:
        kw["extra_tools"] = list(kw.get("extra_tools") or []) + tools
        kw["profile"] = who + ("\n" + kw["profile"] if kw.get("profile") else "")
        ans = ask(
            s,
            store,
            question,
            guild_id=ws.memory_id,
            is_exec=is_exec,
            user_id=kw.pop("user_id", 0),
            history=history,
            **kw,
        )
        return conform_language(ans.text, lang)
    company = ""
    if ws.platform == "discord":  # fallback: the company knowledge without the full bot
        company, more = company_knowledge(ctx, profile, ws, lib, guilds, question, lang)
        tools = tools + more
    ans = ask(
        s,
        store,
        question,
        guild_id=ws.memory_id,
        is_exec=is_exec,
        user_id=0,
        history=history,
        language=lang,
        profile=who,
        company=company,
        extra_tools=tools,
        memory=memory,
    )
    return conform_language(ans.text, lang)


def get_bot() -> Any:
    """The headless Salid (the Discord bot object, never connected). None when it cannot be
    built here (then the chat falls back to the lighter tool set). Overridden in tests."""
    try:
        from salid.headless import get_headless_bot

        return get_headless_bot()
    except Exception:
        logger.exception("headless Salid unavailable")
        return None


def company_language(guilds: Any, ws: Workspace) -> str:
    """The language the web chat answers in: the COMPANY language of the org's Discord server
    (chosen in the hub; English until chosen), never the language somebody typed in. Web-only
    orgs: English."""
    if ws.platform != "discord":
        return "en"
    try:
        g = guilds.get(ws.memory_id)
        lang = getattr(g, "language_setting", None) or "en"
        return lang if lang != "auto" else "en"
    except Exception:
        logger.warning("company language of %s unknown; English", ws.memory_id, exc_info=True)
        return "en"


def needs_language_fix(text: str, lang: str) -> bool:
    """True when `text` is visibly not in `lang`: Thai letters in a non-Thai reply, or a Thai
    reply with no Thai at all (script check only: cheap and never wrong about Thai)."""
    from salid.i18n import base_lang, has_thai

    if not text:
        return False
    if base_lang(lang) == "th":
        return not has_thai(text) and re.search(r"[A-Za-z]{3,}", text) is not None
    return has_thai(text)


@lru_cache
def _translator() -> Callable[[str, str], str]:
    from salid.llm import make_client
    from salid.translate import translate_text

    s, _ = hub.get_salid_runtime()
    client = make_client(s)

    def run(text: str, lang: str) -> str:
        return translate_text(client, s.gate_model, text, lang).translation or text

    return run


def get_translator() -> Callable[[str, str], str]:
    """(text, lang) -> text in `lang`. Overridden in tests."""
    return _translator()


def conform_language(text: str, lang: str) -> str:
    """The safety net of the web chat: a reply in the wrong script is translated into the
    company language before it appears. Any failure keeps the original."""
    if not needs_language_fix(text, lang):
        return text
    try:
        return get_translator()(text, lang) or text
    except Exception:
        logger.warning("could not conform the web answer to %s", lang, exc_info=True)
        return text


def headless_kwargs(
    ctx: Ctx, profile: UserProfile, ws: Workspace, question: str, lang: str, memory: Any
) -> dict | None:
    """`SecretaryBot._agent_kwargs` for a web request, or None when the headless bot is off."""
    from salid.headless import agent_kwargs_in_thread, person_id, web_context

    bot = get_bot()
    if bot is None:
        return None
    try:
        did = (
            profile.identity("discord") or Identity(platform="discord", external_id="")
        ).external_id
        uid = person_id(did, ctx.email)
        guild, member, channel = web_context(
            ws.memory_id,
            ws.name,
            uid=uid,
            name=profile.name or ctx.email.split("@")[0],
            admin=ctx.role in ("owner", "admin", "super"),
            email=ctx.email,
        )
        message = bot.room_message(guild, member, channel)
        kw = agent_kwargs_in_thread(
            bot,
            message=message,
            guild=guild,
            asker=member,
            is_exec=ctx.is_exec,
            setting=lang,
            lang=lang,
            question=question,
            memory=memory,
            profile=None,
        )
        kw["user_id"] = uid
        kw.pop("chat_actions", None)  # Discord-only actions (modes, DMs); the web has its own
        return kw
    except Exception:
        logger.exception("headless agent assembly failed; using the lighter web tools")
        return None


def company_knowledge(
    ctx: Ctx,
    profile: UserProfile,
    ws: Workspace,
    lib: Library,
    guilds: Any,
    question: str,
    lang: str,
) -> tuple[str, list]:
    """The company playbook (what the company taught Salid: addresses, tax id, rules, glossary,
    SOPs, FAQ) as a prompt block plus the teach / show / forget tools, exactly like in Discord.
    ("" , []) when the feature is off or anything fails: the chat never breaks on it."""
    from salid.company_brain import CompanyCtx, CompanyRepo, company_block, company_tools
    from salid.i18n import base_lang
    from salid.orgs import _email_bucket

    try:
        gid = ws.memory_id
        g = guilds.get(gid)
        if not getattr(g.features, "company_learning", True):
            return "", []
        s, _store = hub.get_salid_runtime()
        repo = CompanyRepo(lib.docs, clock=lambda: s.now())
        pb = repo.get(gid)
        block = company_block(pb, question, lang=base_lang(lang))
        did = (
            profile.identity("discord") or Identity(platform="discord", external_id="")
        ).external_id
        cctx = CompanyCtx(
            guild_id=gid,
            asker_id=int(did) if did.isdigit() else _email_bucket(ctx.email),
            asker_name=profile.name or ctx.email,
            is_exec=ctx.is_exec,
            repo=repo,
            lang=base_lang(lang),
            notify=lambda sugg: 0,  # web: suggestions are stored, admins review them in Discord
        )
        return block, company_tools(cctx)
    except Exception:
        logger.exception("company knowledge unavailable for the web chat")
        return "", []


@router.post("/orgs/{org_id}/chat")
async def chat(
    body: ChatBody,
    ctx: Ctx = Depends(member_ctx),
    orgs: OrgStore = Depends(get_orgs),
    lib: Library = Depends(get_library),
    ask: Callable[..., Any] = Depends(get_ask),
    guilds=Depends(hub.get_guild_store),
) -> ChatReply:
    """Same `qa.ask` brain as Discord, as the logged-in member (owner / admin = executive).
    Salid is ONE colleague of the company: without `workspace_id` she answers from the company's
    workspace (its Discord server when connected, else the web workspace)."""
    ws = ctx.org.workspace(body.workspace_id) if body.workspace_id else company_workspace(ctx.org)
    if ws is None or ws.status != "connected":
        raise HTTPException(404, "Unknown workspace")
    if not license_ok(ctx.org.license, datetime.now(UTC)) and not ctx.is_super:
        raise HTTPException(402, "The license of this organization is not active")
    if not _chat_allowed(f"{ctx.org.id}:{ctx.email}"):
        raise HTTPException(429, "Too many messages; try again later")
    question = body.message.strip()
    asked = question
    ids = list(
        dict.fromkeys([*body.document_ids, *([body.document_id] if body.document_id else [])])
    )
    if ids:
        block = await run_in_threadpool(uploaded_docs_block, lib, ctx, ids)
        if block:
            asked = question + "\n\n" + block

    try:
        text = await run_in_threadpool(
            answer_question, ctx, orgs, lib, ask, guilds, ws, asked, body.page
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("center chat failed")
        raise HTTPException(502, "Salid could not answer right now") from exc
    at = stamp()
    await run_in_threadpool(
        orgs.append_chat,
        ctx.org.id,
        ctx.email,
        [
            {"role": "user", "text": question, "at": at, "workspace_id": ws.id},
            {"role": "salid", "text": text, "at": stamp(), "workspace_id": ws.id},
        ],
    )
    return ChatReply(answer=text, workspace_id=ws.id, at=at)


@router.get("/orgs/{org_id}/chat/history")
async def chat_history(ctx: Ctx = Depends(member_ctx), orgs: OrgStore = Depends(get_orgs)) -> dict:
    turns = await run_in_threadpool(orgs.chat_history, ctx.org.id, ctx.email)
    return {"turns": [Turn(**t) for t in turns]}


# ------------------------------------------------------------------ live voice (the Discord Salid, on the web)
LIVE_MAX_S = 30 * 60
_live_open: set[str] = set()  # "org:email" sessions currently open (one per person)


def get_live_connect() -> Callable[[Any], Any] | None:
    """`WebLive.connect` override for tests (None = real Gemini Live)."""
    return None


@router.websocket("/orgs/{org_id}/live")
async def live_voice(
    websocket: WebSocket,
    org_id: str,
    orgs: OrgStore = Depends(get_orgs),
    lib: Library = Depends(get_library),
    ask: Callable[..., Any] = Depends(get_ask),
    guilds=Depends(hub.get_guild_store),
    verify: Callable[[str, str], dict[str, Any]] = Depends(hub.google_token_verifier),
    connect: Callable[[Any], Any] | None = Depends(get_live_connect),
) -> None:
    """Two-way live conversation with Salid, exactly like a Discord voice room: the browser sends
    16 kHz PCM16 frames (binary), Salid's 24 kHz audio comes back (binary); JSON events carry
    transcripts, "interrupted", "thinking", "turn", "ready", "error". First message from the
    browser: {"token", "lang": th|en, "page", "greet": true} (greet = Salid speaks first)."""
    from salid.web_live import WebLive, build_web_instruction

    await websocket.accept()
    key = None
    live: WebLive | None = None
    try:
        hello = await asyncio.wait_for(websocket.receive_json(), 15)
        try:
            email = hub.user_from_token(str(hello.get("token", "")), verify, orgs)
        except HTTPException as exc:
            await websocket.send_json({"type": "error", "text": "auth", "status": exc.status_code})
            await websocket.close(code=4401)
            return
        org = await run_in_threadpool(orgs.get, org_id)
        sup = hub.is_super_admin(email)
        role = org.role_of(email) if org else None
        if org is None or (role is None and not sup):
            await websocket.send_json({"type": "error", "text": "forbidden"})
            await websocket.close(code=4403)
            return
        ctx = Ctx(email, sup, org, role or "super")
        ws = company_workspace(org)
        if ws is None or (not license_ok(org.license, datetime.now(UTC)) and not sup):
            await websocket.send_json({"type": "error", "text": "unavailable"})
            await websocket.close(code=4402)
            return
        key = f"{org.id}:{email}"
        if key in _live_open:
            await websocket.send_json({"type": "error", "text": "already_open"})
            await websocket.close(code=4409)
            return
        _live_open.add(key)
        lang = hello.get("lang") if hello.get("lang") in ("th", "en") else "en"
        page = str(hello.get("page") or "")[:80] or None
        s, _store = hub.get_salid_runtime()
        tone, note = "cheerful", ""
        if ws.platform == "discord":
            try:
                g = await run_in_threadpool(guilds.get, ws.memory_id)
                tone, note = g.default_tone or tone, getattr(g, "tone_note", "") or ""
            except Exception:  # tone is cosmetic: never block the call on it
                pass
        profile = await run_in_threadpool(orgs.user_profile, email)
        instruction = build_web_instruction(s, who_block(ctx, profile, page), lang, tone, note)

        def ask_fn(question: str) -> str:
            return answer_question(ctx, orgs, lib, ask, guilds, ws, question, page, language=lang)

        async def out(kind: str, payload: Any) -> None:
            if kind == "audio":
                await websocket.send_bytes(payload)
            else:
                await websocket.send_json({"type": kind, "text": payload})

        from google import genai

        # the Live model is served from `live_location` (us-central1), not `global`: same client
        # settings as the Discord room
        live_client = genai.Client(vertexai=True, project=s.gcp_project, location=s.live_location)
        live = WebLive(
            s, live_client, instruction=instruction, ask=ask_fn, out=out, connect=connect
        )
        task = live.start()
        if hello.get("greet", True):
            live.greet(lang)
        started = time.monotonic()
        while True:
            if task.done():
                break
            if time.monotonic() - started > LIVE_MAX_S:
                await websocket.send_json({"type": "error", "text": "time_limit"})
                break
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                live.feed(msg["bytes"])
            elif msg.get("text"):
                try:
                    ev = json.loads(msg["text"])
                except ValueError:
                    continue
                if ev.get("type") == "close":
                    break
                if ev.get("type") == "say" and ev.get("text"):
                    live.say(str(ev["text"])[:500])
    except (TimeoutError, WebSocketDisconnect):
        pass
    except Exception:
        logger.exception("live voice session failed")
    finally:
        if key:
            _live_open.discard(key)
        if live is not None:
            await live.close()
        with contextlib.suppress(Exception):
            await websocket.close()


# ------------------------------------------------------------------ tasks (read-only view)
@router.get("/orgs/{org_id}/tasks")
async def tasks_view(
    workspace: str | None = Query(default=None),
    ctx: Ctx = Depends(member_ctx),
    orgs: OrgStore = Depends(get_orgs),
    lib: Library = Depends(get_library),
    connectors: dict = Depends(get_connectors),
) -> dict:
    """Open tasks: Discord workspaces from Salid's COO list (plain members see tasks that came
    from public channels), ClickUp workspaces live and read-only."""
    from salid.tasks import TaskRepo

    def run() -> list[dict]:
        out: list[dict] = []
        repo = TaskRepo(lib.docs)
        for ws in ctx.org.workspaces:
            if workspace and ws.id != workspace or ws.status != "connected":
                continue
            if ws.platform == "discord":
                public = lib._channel_sets(ws)[0]
                for t in repo.list(ws.memory_id):
                    if not ctx.is_exec and t.source_channel_id not in public:
                        continue
                    out.append(
                        {
                            "id": t.id,
                            "title": t.title,
                            "assignee": t.assignee_name,
                            "due": t.due,
                            "status": t.status,
                            "url": t.source_link,
                            "workspace": ws.id,
                            "platform": "discord",
                        }
                    )
            elif ws.platform == "clickup":
                tok = (lib.docs.get(TOKENS, ws.memory_id, "clickup") or {}).get("access_token")
                if not tok or not connectors["clickup"].enabled():
                    continue
                try:
                    for t in connectors["clickup"].tasks(tok, ws.external_id):
                        out.append(
                            {
                                **t,
                                "assignee": ", ".join(t["assignees"]),
                                "workspace": ws.id,
                                "platform": "clickup",
                            }
                        )
                except ConnectorError:
                    logger.warning("clickup task import failed", exc_info=True)
        return out

    return {"tasks": await run_in_threadpool(run)}


# ------------------------------------------------------------------ connectors
PLATFORM_NAMES = {
    "discord": "Discord",
    "slack": "Slack",
    "clickup": "ClickUp",
    "google_drive": "Google Drive",
    "gmail": "Gmail",
    "google_calendar": "Google Calendar",
}


@router.get("/connectors")
async def list_connectors(
    org: str | None = Query(default=None),
    email: str = Depends(hub.require_user),
    orgs: OrgStore = Depends(get_orgs),
    connectors: dict = Depends(get_connectors),
) -> list[dict]:
    """Integration cards. `status`: available | not_configured | coming_soon. With `org` also
    the number of connected workspaces per platform (only for members of that org)."""
    connected: dict[str, int] = defaultdict(int)
    if org:
        o = await run_in_threadpool(orgs.get, org)
        if o and (o.role_of(email) or hub.is_super_admin(email)):
            for w in o.workspaces:
                if w.platform != "web" and w.status == "connected":
                    connected[w.platform] += 1
    out = []
    for platform in ("discord", "slack", "clickup"):
        on = connectors[platform].enabled()
        out.append(
            {
                "platform": platform,
                "name": PLATFORM_NAMES[platform],
                "enabled": on,
                "configured": on,
                "status": "available" if on else "not_configured",
                "connected": connected.get(platform, 0),
            }
        )
    for platform in COMING_SOON:
        out.append(
            {
                "platform": platform,
                "name": PLATFORM_NAMES[platform],
                "enabled": False,
                "configured": False,
                "status": "coming_soon",
                "connected": 0,
            }
        )
    return out


@router.get("/connectors/{platform}/start")
async def connector_start(
    platform: str,
    org: str = Query(),
    email: str = Depends(hub.require_user),
    orgs: OrgStore = Depends(get_orgs),
    connectors: dict = Depends(get_connectors),
) -> dict:
    """JSON `{url}`: navigate the browser there. Discord: the bot invite (then `/ตั้งค่า` and the
    link-code endpoint); Slack / ClickUp: the platform's OAuth consent page."""
    conn = connectors.get(platform)
    if conn is None:
        raise HTTPException(404, "Unknown connector")
    o = await run_in_threadpool(orgs.get, org)
    if o is None:
        raise HTTPException(
            404 if hub.is_super_admin(email) else 403, "No access to this organization"
        )
    role = o.role_of(email)
    if not hub.is_super_admin(email) and role not in ("owner", "admin"):
        raise HTTPException(403, "Owner or admin only")
    if not conn.enabled():
        raise HTTPException(503, f"{PLATFORM_NAMES[platform]} is not configured")
    if platform == "discord":
        return {"url": conn.oauth_start(org), "kind": "invite", "next": "/ตั้งค่า"}
    state = secrets.token_urlsafe(24)
    exp = (datetime.now(UTC) + STATE_TTL).isoformat(timespec="seconds")
    await run_in_threadpool(
        orgs.docs.put,
        STATE_KIND,
        0,
        state,
        {"org_id": org, "email": email, "platform": platform, "exp": exp},
    )
    return {"url": conn.oauth_start(org, state), "kind": "oauth"}


@router.get("/connectors/{platform}/callback", include_in_schema=False)
async def connector_callback(
    platform: str,
    request: Request,
    orgs: OrgStore = Depends(get_orgs),
    connectors: dict = Depends(get_connectors),
) -> RedirectResponse:
    """Browser redirect from the platform (no Bearer header): the one-time `state` created by
    /start identifies the org and the user. Redirects to `/center?connected=<platform>` or
    `/center?error=<code>`."""
    params = dict(request.query_params)
    conn = connectors.get(platform)

    def fail(code: str) -> RedirectResponse:
        return RedirectResponse(
            f"/center?{urlencode({'error': code, 'platform': platform})}", status_code=303
        )

    if conn is None or platform == "discord" or not conn.enabled():
        return fail("not_available")

    def run() -> str | None:
        st = orgs.docs.get(STATE_KIND, 0, params.get("state", ""))
        if not st or st.get("platform") != platform:
            return "bad_state"
        orgs.docs.delete(STATE_KIND, 0, params["state"])  # one time
        if datetime.fromisoformat(st["exp"]) < datetime.now(UTC):
            return "expired"
        org = orgs.get(st["org_id"])
        if org is None:
            return "forbidden"
        if org.role_of(st["email"]) not in ("owner", "admin") and not hub.is_super_admin(
            st["email"]
        ):
            return "forbidden"
        try:
            conns = conn.oauth_callback(params)
        except ConnectorError as exc:
            logger.info("connector %s failed: %s", platform, exc)
            return "failed"
        for c in conns:
            ws = Workspace(
                id=workspace_id(c.platform, c.external_id),
                platform=c.platform,
                external_id=c.external_id,
                name=c.name,
                connected_at=stamp(),
                connected_by=st["email"],
            )
            try:
                orgs.attach_workspace(org.id, ws)
            except WorkspaceTaken:
                return "taken"
            if c.token:
                orgs.docs.put(TOKENS, ws.memory_id, platform, {"access_token": c.token})
        return None

    err = await run_in_threadpool(run)
    if err:
        return fail(err)
    return RedirectResponse(f"/center?connected={platform}", status_code=303)


# ------------------------------------------------------------------ slack events
@events_router.post("/api/connectors/slack/events", include_in_schema=False)
async def slack_events(
    request: Request,
    x_slack_request_timestamp: str | None = Header(default=None),
    x_slack_signature: str | None = Header(default=None),
    orgs: OrgStore = Depends(get_orgs),
    connectors: dict = Depends(get_connectors),
) -> dict:
    """Slack Events API. Rejects anything without a valid signature (v0 HMAC, 5-minute window);
    answers the URL-verification challenge; stores public-channel messages of connected
    workspaces whose org license is active."""
    slack: SlackConnector = connectors["slack"]
    if not slack.enabled():
        raise HTTPException(404, "Slack is not configured")
    body = await request.body()
    if not slack.verify(body, x_slack_request_timestamp or "", x_slack_signature or ""):
        raise HTTPException(401, "Bad signature")
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(400, "Bad JSON") from exc
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge", "")}
    if payload.get("type") != "event_callback":
        return {"ok": True}

    def run() -> int:
        found = orgs.org_of_workspace("slack", str(payload.get("team_id", "")))
        if not found or found[1].status != "connected":
            return 0
        if not license_ok(found[0].license, datetime.now(UTC)):
            return 0
        return slack.ingest(payload)

    try:
        n = await run_in_threadpool(run)
    except Exception:
        logger.warning("slack ingest failed", exc_info=True)
        n = 0
    return {"ok": True, "stored": n}


# ------------------------------------------------------------------ verified identities
def get_identity_links(orgs: OrgStore = Depends(get_orgs)) -> IdentityLinks:
    return IdentityLinks(orgs)


class IssuedCodeView(BaseModel):
    code: str  # "1234-5678"
    platform: str
    kind: str  # identity | attach
    expires_at: str
    command: str  # what to send to Salid in a private chat


class LinkedIdentityView(BaseModel):
    platform: str
    account_hint: str  # the last characters of the linked account id, never the whole id


def _issued_view(issued: IssuedCode) -> IssuedCodeView:
    return IssuedCodeView(
        code=issued.code,
        platform=issued.platform,
        kind=issued.kind,
        expires_at=issued.expires_at,
        command=f"link {issued.code}",
    )


def _code_platform(platform: str) -> str:
    if platform not in CODE_PLATFORMS:
        raise HTTPException(
            400, f"Linking by code works for: {', '.join(CODE_PLATFORMS)} (Discord uses /setup)"
        )
    return platform


@router.post("/orgs/{org_id}/identities/{platform}/code")
async def create_identity_code(
    platform: str,
    ctx: Ctx = Depends(member_ctx),
    links: IdentityLinks = Depends(get_identity_links),
) -> IssuedCodeView:
    """A 15-minute single-use code that links the caller's `platform` chat account to their
    member record: they send `link 1234-5678` to Salid in a private chat. At most 5 live codes
    per member (429)."""
    _code_platform(platform)

    def run() -> IssuedCode:
        try:
            return links.create_code(ctx.org.id, ctx.email, platform)
        except NotAMember as exc:
            raise HTTPException(403, "Only members of the organization can link accounts") from exc
        except TooManyCodes as exc:
            raise HTTPException(429, "Too many unused codes; wait for one to expire") from exc
        except CodeError as exc:
            raise HTTPException(400, "Cannot create a code") from exc

    return _issued_view(await run_in_threadpool(run))


def _identity_views(org: Org, email: str) -> list[LinkedIdentityView]:
    me = next((m for m in org.members if m.email.lower() == email.lower()), None)
    linked = me.all_identities() if me else {}
    return [LinkedIdentityView(platform=p, account_hint=v[-4:]) for p, v in sorted(linked.items())]


@router.get("/orgs/{org_id}/identities")
async def my_identities(ctx: Ctx = Depends(member_ctx)) -> list[LinkedIdentityView]:
    """The caller's own linked chat accounts in this organization."""
    return _identity_views(ctx.org, ctx.email)


@router.delete("/orgs/{org_id}/identities/{platform}")
async def unlink_identity(
    platform: str,
    ctx: Ctx = Depends(member_ctx),
    orgs: OrgStore = Depends(get_orgs),
) -> list[LinkedIdentityView]:
    """Unlink the caller's own `platform` account. Discord is managed in the hub, not here."""
    _code_platform(platform)
    removed = await run_in_threadpool(orgs.unlink_identity, ctx.org.id, ctx.email, platform)
    if not removed:
        raise HTTPException(404, "Nothing is linked for this platform")
    return _identity_views(await run_in_threadpool(orgs.get, ctx.org.id), ctx.email)


@router.post("/orgs/{org_id}/workspaces/{platform}/attach-code")
async def create_attach_code(
    platform: str,
    ctx: Ctx = Depends(admin_ctx),
    links: IdentityLinks = Depends(get_identity_links),
) -> IssuedCodeView:
    """A 15-minute single-use code that attaches a bot (LINE channel, Telegram bot, WhatsApp
    number, Slack team) to this organization when sent to that bot in a private chat:
    `link 1234-5678`. The bot's tenant must not belong to another organization."""
    _code_platform(platform)

    def run() -> IssuedCode:
        try:
            return links.create_attach_code(ctx.org.id, ctx.email, platform)
        except TooManyCodes as exc:
            raise HTTPException(429, "Too many unused codes; wait for one to expire") from exc
        except CodeError as exc:
            raise HTTPException(400, "Cannot create a code") from exc

    return _issued_view(await run_in_threadpool(run))


# ------------------------------------------------------------------ Google Calendar (per member)
def get_google_oauth(orgs: OrgStore = Depends(get_orgs)) -> Any:
    from salid.google_oauth import GoogleConfig, GoogleOAuth

    return GoogleOAuth(orgs.docs, GoogleConfig.from_settings(get_settings()))


def _google_member(ctx: Ctx, g: Any) -> str:
    """503 when the feature is off; 403 unless the caller is a real member (a super admin who is
    not in the org has no calendar here). Returns the member's email."""
    if not g.enabled():
        raise HTTPException(503, "Google Calendar is not configured")
    if ctx.org.role_of(ctx.email) is None:
        raise HTTPException(403, "Only members of the organization can connect a calendar")
    return ctx.email


@router.get("/orgs/{org_id}/google/calendar")
async def google_calendar_status(
    ctx: Ctx = Depends(member_ctx), g: Any = Depends(get_google_oauth)
) -> dict:
    """The caller's own Google Calendar connection: `{connected, email, connected_at}`."""
    email = _google_member(ctx, g)
    return await run_in_threadpool(g.status, ctx.org.id, email)


@router.post("/orgs/{org_id}/google/calendar/connect")
async def google_calendar_connect(
    ctx: Ctx = Depends(member_ctx), g: Any = Depends(get_google_oauth)
) -> dict:
    """`{auth_url}`: navigate the browser there (Google consent, single-use state, 10 minutes)."""
    email = _google_member(ctx, g)
    return {"auth_url": await run_in_threadpool(g.start, ctx.org.id, email)}


@router.delete("/orgs/{org_id}/google/calendar")
async def google_calendar_disconnect(
    ctx: Ctx = Depends(member_ctx), g: Any = Depends(get_google_oauth)
) -> dict:
    """Revoke at Google and delete the stored token."""
    from salid.google_oauth import Account

    email = _google_member(ctx, g)
    await run_in_threadpool(g.revoke, Account(ctx.org.id, email))
    return {"connected": False, "email": None, "connected_at": None}


@router.get("/connectors/google/callback", include_in_schema=False)
async def google_calendar_callback(
    request: Request,
    orgs: OrgStore = Depends(get_orgs),
    g: Any = Depends(get_google_oauth),
) -> RedirectResponse:
    """Browser redirect from Google (no Bearer header): the single-use `state` made by /connect
    identifies the org and member. Redirects to `/center?calendar=connected` or
    `/center?calendar=error&reason=<code>`."""
    from salid.google_oauth import Account, GoogleOAuthError

    params = dict(request.query_params)

    def fail(reason: str) -> RedirectResponse:
        return RedirectResponse(
            f"/center?{urlencode({'calendar': 'error', 'reason': reason})}", status_code=303
        )

    if not g.enabled():
        return fail("not_available")
    if params.get("error"):  # the user pressed Cancel at Google
        if params.get("state"):
            await run_in_threadpool(lambda: g.docs.delete("salid_google_state", 0, params["state"]))
        return fail("denied")

    def run() -> tuple[str, Account | None]:
        try:
            acct = g.exchange(params.get("code", ""), params.get("state", ""))
        except GoogleOAuthError as exc:
            return exc.code, None
        org = orgs.get(acct.org_id)
        if org is None or org.role_of(acct.email) is None:  # removed meanwhile
            g.revoke(acct)
            return "forbidden", None
        return "", acct

    reason, acct = await run_in_threadpool(run)
    if acct is None:
        return fail(reason)
    return RedirectResponse(
        f"/center?{urlencode({'calendar': 'connected', 'org': acct.org_id})}", status_code=303
    )


# `/connectors/{platform}/callback` above would swallow `google`; routes match in order.
router.routes.insert(0, router.routes.pop())
