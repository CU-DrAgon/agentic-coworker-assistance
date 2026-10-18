"""Salid command center API: `/center` page and `/api/center/*` (organizations, workspaces,
sharing, license, documents library, web chat, connectors).

Auth is the hub's: a Google ID token as Bearer (`hub.require_user`). Roles come from the
organization's member list (owner / admin / member); `HUB_ADMIN_EMAILS` are super admins who see
and manage every org and set licenses. The contract with request / response examples is written to `command-center-api.md`.
"""

from __future__ import annotations

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

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from meeting_scribe import hub
from meeting_scribe.config import get_settings
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
from salid.orgs import (
    CATEGORIES,
    PLANS,
    ROLES,
    STATUSES,
    License,
    LicenseRequest,
    Member,
    Org,
    OrgStore,
    ShareLink,
    Workspace,
    WorkspaceTaken,
    stamp,
    trial_license,
    workspace_id,
)
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

            lazy["fn"] = make_analyzer(make_client(s), s.gate_model)
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


class MemberView(BaseModel):
    email: str
    role: str
    discord_user_id: str | None = None


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
    created_at: str
    owner_email: str
    role: str  # the caller's role: owner | admin | member | super
    license: LicenseView
    members: list[MemberView]
    workspaces: list[WorkspaceView]
    sharing: list[ShareLink]


class OrgBrief(BaseModel):
    id: str
    name: str
    role: str
    license: LicenseView


class MeView(BaseModel):
    email: str
    is_super: bool
    orgs: list[OrgBrief]


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
        created_at=o.created_at,
        owner_email=o.owner_email,
        role=ctx.role,
        license=license_view(o.license),
        members=[MemberView(**m.model_dump()) for m in o.members],
        workspaces=[workspace_view(w) for w in o.workspaces],
        sharing=o.sharing,
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
    found = await run_in_threadpool(orgs.all_orgs if sup else (lambda: orgs.orgs_of_email(email)))
    return MeView(
        email=email,
        is_super=sup,
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


@router.post("/orgs")
async def create_org(
    body: CreateOrg, email: str = Depends(hub.require_user), orgs: OrgStore = Depends(get_orgs)
) -> OrgView:
    """The creator becomes owner; the license is a 14-day trial."""

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


class UpdateOrg(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    logo_url: str = Field(default="", max_length=500, pattern=r"^(https?://\S+)?$")


@router.put("/orgs/{org_id}")
async def update_org(
    body: UpdateOrg, ctx: Ctx = Depends(admin_ctx), orgs: OrgStore = Depends(get_orgs)
) -> OrgView:
    def run() -> None:
        with orgs.lock:
            org = orgs.get(ctx.org.id)
            org.name, org.logo_url = body.name.strip(), body.logo_url
            orgs.save(org)
            ctx.org = org

    await run_in_threadpool(run)
    return org_view(ctx)


# ------------------------------------------------------------------ members
class MemberBody(BaseModel):
    email: str = Field(min_length=3, max_length=200, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    role: str = "member"
    discord_user_id: str | None = Field(default=None, max_length=30)


@router.get("/orgs/{org_id}/members")
async def list_members(ctx: Ctx = Depends(admin_ctx)) -> list[MemberView]:
    return [MemberView(**m.model_dump()) for m in ctx.org.members]


@router.post("/orgs/{org_id}/members")
async def add_member(
    body: MemberBody, ctx: Ctx = Depends(admin_ctx), orgs: OrgStore = Depends(get_orgs)
) -> list[MemberView]:
    """Add a member or change a member's role. Admins may add plain members; only an owner (or
    a super admin) may hand out admin / owner."""
    if body.role not in ROLES:
        raise HTTPException(422, "role must be owner, admin or member")
    email = body.email.lower()
    can_grant = ctx.is_super or ctx.role == "owner"
    if body.role != "member" and not can_grant:
        raise HTTPException(403, "Only an owner can grant admin or owner")

    def run() -> Org:
        with orgs.lock:
            org = orgs.get(ctx.org.id)
            current = org.role_of(email)
            if current in ("owner", "admin") and not can_grant:
                raise HTTPException(403, "Only an owner can change an admin or owner")
            if current is None and len(org.members) >= max(org.license.seats, 1):
                raise HTTPException(402, "Seat limit reached; request more seats")
            owners = sum(1 for m in org.members if m.role == "owner")
            if current == "owner" and body.role != "owner" and owners <= 1:
                raise HTTPException(409, "An organization needs at least one owner")
            org.members = [m for m in org.members if m.email.lower() != email]
            org.members.append(
                Member(email=email, role=body.role, discord_user_id=body.discord_user_id)
            )
            if body.role == "owner" and not org.owner_email:
                org.owner_email = email
            return orgs.save(org)

    org = await run_in_threadpool(run)
    await run_in_threadpool(record_org_edit, orgs.docs, ctx.org, org, ctx.email)  # change outbox
    return [MemberView(**m.model_dump()) for m in org.members]


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
    record_org_edit(orgs.docs, ctx.org, org, ctx.email)  # change outbox (never raises)
    return [MemberView(**m.model_dump()) for m in org.members]


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
            _gid, g = guilds.redeem_link_code(code, ctx.email)
        except LinkError as exc:
            raise HTTPException(400, "Link code is invalid, expired or already used") from exc
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
    workspace: str | None = Query(default=None),
    q: str | None = Query(default=None, max_length=200),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=30, ge=1, le=100),
    ctx: Ctx = Depends(member_ctx),
    lib: Library = Depends(get_library),
) -> DocPage:
    if category and category not in LIB_CATEGORIES:
        raise HTTPException(422, f"category must be among {', '.join(LIB_CATEGORIES)}")
    page = await run_in_threadpool(
        lambda: lib.list(
            ctx.org,
            _viewer(ctx),
            category=category,
            workspace=workspace,
            q=q,
            cursor=cursor,
            limit=limit,
        )
    )
    return DocPage(items=page.items, next_cursor=page.next_cursor, total=page.total)


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
async def overview(ctx: Ctx = Depends(member_ctx), lib: Library = Depends(get_library)) -> dict:
    def run() -> dict:
        viewer = _viewer(ctx)
        got = lib.gather(ctx.org, viewer)
        counts = dict.fromkeys(LIB_CATEGORIES, 0)
        for i in got.items.values():
            counts[i.category] += 1
        recent = sorted(got.items.values(), key=lambda i: (i.date, i.id), reverse=True)[:8]
        by_platform: dict[str, int] = defaultdict(int)
        for w in ctx.org.workspaces:
            by_platform[w.platform] += 1
        lic = license_view(ctx.org.license)
        return {
            "org": {"id": ctx.org.id, "name": ctx.org.name},
            "documents": {"total": sum(counts.values()), "by_category": counts},
            "workspaces": {
                "total": len(ctx.org.workspaces),
                "connected": sum(1 for w in ctx.org.workspaces if w.status == "connected"),
                "by_platform": dict(by_platform),
            },
            "members": len(ctx.org.members),
            "license": lic.model_dump(),
            "license_days_left": lic.days_left,
            "recent_activity": [
                {
                    "id": i.id,
                    "category": i.category,
                    "title": i.title,
                    "at": i.date,
                    "workspace": i.workspace,
                    "workspace_name": i.workspace_name,
                    "who": i.who,
                }
                for i in recent
            ],
        }

    return await run_in_threadpool(run)


# ------------------------------------------------------------------ chat
class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    workspace_id: str | None = None


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


@router.post("/orgs/{org_id}/chat")
async def chat(
    body: ChatBody,
    ctx: Ctx = Depends(member_ctx),
    orgs: OrgStore = Depends(get_orgs),
    lib: Library = Depends(get_library),
    ask: Callable[..., Any] = Depends(get_ask),
) -> ChatReply:
    """Same `qa.ask` brain as Discord, as the logged-in member (owner / admin = executive)."""
    ws = ctx.org.workspace(body.workspace_id) if body.workspace_id else ctx.org.web_workspace()
    if ws is None or ws.status != "connected":
        raise HTTPException(404, "Unknown workspace")
    if not license_ok(ctx.org.license, datetime.now(UTC)) and not ctx.is_super:
        raise HTTPException(402, "The license of this organization is not active")
    if not _chat_allowed(f"{ctx.org.id}:{ctx.email}"):
        raise HTTPException(429, "Too many messages; try again later")
    question = body.message.strip()

    def run() -> str:
        from salid.i18n import resolve_lang
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
        ans = ask(
            s,
            store,
            question,
            guild_id=ws.memory_id,
            is_exec=is_exec,
            user_id=0,
            history=history,
            language=resolve_lang("auto", question),
            memory=memory,
        )
        return ans.text

    try:
        text = await run_in_threadpool(run)
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


class IdentityView(BaseModel):
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


def _identity_views(org: Org, email: str) -> list[IdentityView]:
    me = next((m for m in org.members if m.email.lower() == email.lower()), None)
    linked = me.all_identities() if me else {}
    return [IdentityView(platform=p, account_hint=v[-4:]) for p, v in sorted(linked.items())]


@router.get("/orgs/{org_id}/identities")
async def my_identities(ctx: Ctx = Depends(member_ctx)) -> list[IdentityView]:
    """The caller's own linked chat accounts in this organization."""
    return _identity_views(ctx.org, ctx.email)


@router.delete("/orgs/{org_id}/identities/{platform}")
async def unlink_identity(
    platform: str,
    ctx: Ctx = Depends(member_ctx),
    orgs: OrgStore = Depends(get_orgs),
) -> list[IdentityView]:
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
