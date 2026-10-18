"""Organizations ("companies") for the Salid command center, and the store that keeps them.

An `Org` owns members (roles owner / admin / member), a license, workspaces (one connected
platform space each: a Discord server, a Slack team, a ClickUp team, or the org's own "web"
workspace) and `ShareLink`s (directional data sharing between workspaces of the same org, off by
default). A Discord guild belongs to at most ONE org.

Storage follows the repo's store pattern: everything lives in the `DocStore` (SQLite in dev,
Firestore in prod), org documents under guild id 0 and small index documents keyed by the
workspace's memory id / the member's email so lookups never scan:

* `salid_orgs`           guild 0, key org id      -> the whole `Org` document
* `salid_org_ws`         guild = workspace memory id, key "org" -> {org_id, workspace_id}
* `salid_org_email`      guild = hash(email), key org id        -> {email, org_id}
* `salid_license_reqs`   guild 0, key request id  -> `LicenseRequest`
* `salid_center_chat`    guild 0, key "org:email" -> web chat history

Memory id of a workspace (the "guild id" used in the memory store, ledger, wiki...):
a Discord workspace uses the real guild id; every other platform uses a namespaced id
`NAMESPACE_BASE + (sha256(platform + ":" + external_id)[:8] mod 2**61)`. Discord snowflakes stay
below 2**62 until the 22nd century, so the ranges never meet.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from .coo_store import DocStore

ORGS = "salid_orgs"
ORG_WS = "salid_org_ws"
ORG_EMAIL = "salid_org_email"
LICENSE_REQS = "salid_license_reqs"
CENTER_CHAT = "salid_center_chat"
USER_PREFS = (
    "salid_user_prefs"  # guild = hash(email), key email -> UserProfile (language, identities)
)
IDENTITY_INDEX = "salid_identity_index"  # guild 0, key "platform:external_id" -> {email}
UI_LANGS = ("th", "en")  # languages the web pages are written in
SIGNIN_PLATFORMS = ("google",)  # extra sign-ins linked to an account (external_id = the email)

ROLES = ("owner", "admin", "member")
PLATFORMS = ("discord", "slack", "line", "telegram", "whatsapp", "clickup", "web")
# Platforms where a person has a chat account that can be linked to an org member
IDENTITY_PLATFORMS = ("discord", "slack", "line", "telegram", "whatsapp")
MAX_IDENTITY_LEN = 200
INVITES = "salid_org_invites"  # guild 0, key = the code: an invite link to join an org
INVITE_DAYS = 7  # default life of an invite link; 0 = never expires (max INVITE_MAX_DAYS)
INVITE_MAX_DAYS = 365
INVITE_MAX_USES = 1000  # `max_uses` 0 = unlimited
PLANS = ("trial", "monthly", "yearly", "enterprise")
STATUSES = ("active", "past_due", "expired", "suspended")
CATEGORIES = ("documents", "wiki", "places", "minutes", "ledger", "tasks")
NAMESPACE_BASE = 2**62
TRIAL_DAYS = 14
TRIAL_SEATS = 10_000  # "no limit" for the launch trial (LicenseRequestBody caps seats at 10,000)


def _now() -> datetime:
    return datetime.now(UTC)


def stamp(dt: datetime | None = None) -> str:
    return (dt or _now()).isoformat(timespec="seconds")


def workspace_memory_id(platform: str, external_id: str) -> int:
    """The id under which this workspace's data lives in the memory store / ledger / wiki."""
    if platform == "discord":
        return int(external_id)
    digest = hashlib.sha256(f"{platform}:{external_id}".encode()).digest()
    return NAMESPACE_BASE + (int.from_bytes(digest[:8], "big") % 2**61)


def namespaced_id(platform: str, external_id: str, kind: str = "channel") -> int:
    """Stable positive id for a non-Discord channel / user (Slack channel ids are strings)."""
    digest = hashlib.sha256(f"{platform}:{kind}:{external_id}".encode()).digest()
    return NAMESPACE_BASE + (int.from_bytes(digest[:8], "big") % 2**61)


def message_namespaced_id(platform: str, tenant_key: str, room_key: str, key: str) -> int:
    """Stable positive id for a non-Discord message (Slack `ts`, LINE message id, ...).

    The memory store keys messages by (guild_id, message_id) with no channel in the key, and a
    Slack `ts` is only unique inside one channel, so the id hashes the workspace and the room
    together with the message key: two channels can never overwrite each other's records.
    """
    digest = hashlib.sha256(
        "\x1f".join((platform, "message", tenant_key, room_key, key)).encode()
    ).digest()
    return NAMESPACE_BASE + (int.from_bytes(digest[:8], "big") % 2**61)


def workspace_id(platform: str, external_id: str) -> str:
    return "web" if platform == "web" else f"{platform}-{external_id}"


def _email_bucket(email: str) -> int:
    return int.from_bytes(hashlib.sha256(email.lower().encode()).digest()[:7], "big")


# platforms whose account ids only mean something inside one workspace (LINE: per provider,
# WhatsApp: per business, Telegram: per bot), so a verified link authorizes only that tenant
TENANT_BOUND_PLATFORMS = frozenset({"line", "telegram", "whatsapp"})


# ------------------------------------------------------------------ models
class Member(BaseModel):
    email: str
    role: str = "member"  # owner | admin | member
    discord_user_id: str | None = None
    # platform -> platform user id, VERIFIED by Salid (Slack profile email, or a one-time link
    # code). Discord stays in `discord_user_id`; read it through `identity("discord")`.
    identities: dict[str, str] = Field(default_factory=dict)
    # platform -> the tenant key (LINE channel, WhatsApp number, ...) where that identity was
    # verified. LINE user ids are per provider and WhatsApp ids per business, so a link made in
    # one workspace must never vouch for another. Absent = linked before this was recorded.
    identity_tenants: dict[str, str] = Field(default_factory=dict)
    department: str = ""  # PM dashboard: tech | business | finance | other | ""
    finance: bool = False  # PM dashboard: Authorized Approver, may see budgets
    # Position in the company, mirrored by Salid (salid.positions): title as written and its
    # level (1 founder/owner ... 7 intern, 0 unknown). Default empty/0 so older documents and the
    # web keep working; the web may edit `title`, Salid then adopts it.
    title: str = ""
    level: int = 0

    def identity(self, platform: str) -> str | None:
        """The verified platform account id of this member, or None."""
        if platform == "discord":
            return self.identities.get("discord") or self.discord_user_id or None
        return self.identities.get(platform) or None

    def identity_tenant(self, platform: str) -> str | None:
        """The tenant key the platform account was verified in, or None (not recorded)."""
        return self.identity_tenants.get(platform) or None

    def all_identities(self) -> dict[str, str]:
        out = {k: v for k, v in self.identities.items() if v}
        if self.discord_user_id and "discord" not in out:
            out["discord"] = self.discord_user_id
        return out

    reports_to: str | None = None  # email of their manager in THIS org (the org chart)


class Identity(BaseModel):
    """One person's account on one platform. `verified` = proven by that platform (OAuth or a
    one-time code the bot handed to that very user), otherwise typed by hand."""

    platform: str  # see IDENTITY_PLATFORMS
    external_id: str
    name: str = ""
    verified: bool = False
    linked_at: str = ""


class UserProfile(BaseModel):
    """The person behind an email: what every Salid page and the AI know about them, across orgs."""

    email: str
    name: str = ""  # display name (what Salid calls them)
    first_name: str = ""  # legal first name (from the Google account at sign-in, editable)
    last_name: str = ""
    title: str = ""  # job title, e.g. "CTO"
    department: str = ""  # e.g. "Engineering"
    about: str = ""  # what they do / are responsible for, in their own words
    language: str | None = None  # web UI language (th / en)
    identities: list[Identity] = Field(default_factory=list)

    def identity(self, platform: str) -> Identity | None:
        return next((i for i in self.identities if i.platform == platform), None)

    @property
    def full_name(self) -> str:
        return " ".join(x for x in (self.first_name, self.last_name) if x).strip()


class License(BaseModel):
    plan: str = "trial"  # trial | monthly | yearly | enterprise
    status: str = "active"  # active | past_due | expired | suspended
    seats: int = 5
    valid_until: str | None = None  # ISO date (YYYY-MM-DD) or datetime; None = no end
    notes: str = ""
    billing_provider: str = ""  # seam for Stripe later; empty = manual


class Workspace(BaseModel):
    id: str
    platform: str  # discord | slack | line | telegram | whatsapp | clickup | web
    external_id: str  # guild id / slack team id / clickup team id / org id (web)
    name: str = ""
    icon_url: str = ""
    status: str = "connected"  # connected | pending | disconnected
    connected_at: str = ""
    connected_by: str = ""

    @property
    def memory_id(self) -> int:
        return workspace_memory_id(self.platform, self.external_id)


class ShareLink(BaseModel):
    from_workspace: str
    to_workspace: str
    categories: list[str] = Field(default_factory=list)
    created_by: str = ""
    created_at: str = ""


class CompanyProfile(BaseModel):
    """What the company says about itself: the public-facing profile (shown to members and,
    later, to outsiders). All optional; admins edit it on the web or by telling Salid."""

    tagline: str = ""  # one line, e.g. "AI assistants for Thai SMEs"
    about: str = ""  # a paragraph or two
    industry: str = ""
    founded: str = ""  # year or date, free text
    location: str = ""  # city, country
    website: str = ""
    email: str = ""
    phone: str = ""
    size: str = ""  # e.g. "11-50 people"
    values: list[str] = Field(default_factory=list)  # a few words each, max 8
    # the legal entity, as on the registration papers (filled from an uploaded registration
    # certificate / articles by Salid, or by an admin)
    registered_name: str = ""  # e.g. บริษัท เดอะ ฟรอนเทียร์ จำกัด
    registration_no: str = ""  # juristic person registration number
    tax_id: str = ""
    registered_capital: str = ""  # free text with currency
    registered_address: str = ""
    directors: str = ""  # authorised directors / signing rule
    objectives: str = ""  # business objectives (วัตถุประสงค์)


PROFILE_LIMITS = {
    "tagline": 160, "about": 2000, "industry": 80, "founded": 40, "location": 120,
    "website": 300, "email": 200, "phone": 60, "size": 40,
    "registered_name": 200, "registration_no": 40, "tax_id": 40, "registered_capital": 120,
    "registered_address": 400, "directors": 600, "objectives": 2000,
}  # fmt: skip


POSITION_LIMITS = {
    "title": 80,
    "department": 80,
    "summary": 1000,  # the job description
    "responsibilities": 2000,
    "policies": 2000,  # company rules that apply to this job
}
MAX_POSITIONS = 200


class Position(BaseModel):
    """A job in the company, created by an admin, filled or VACANT: its job description,
    responsibilities, required skills and the company policies that apply to it. Organisational
    knowledge: Salid reads it to answer "who does X", to match people (expertise locator) and to
    brief a successor. `holders` = member emails (several people may share a role)."""

    id: str
    title: str
    department: str = ""
    reports_to: str = ""  # another position's id ("" = top)
    summary: str = ""
    responsibilities: str = ""
    skills: list[str] = Field(default_factory=list)
    policies: str = ""
    holders: list[str] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    updated_by: str = ""

    @property
    def vacant(self) -> bool:
        return not self.holders


HANDOVER_UNDO_HOURS = 24


class Handover(BaseModel):
    """The last transfer of ownership: security first, so the former owner can take it back
    within `HANDOVER_UNDO_HOURS` (`undo_until`), and no second transfer happens inside that
    window. `to_role` is what the receiver was before (restored on undo)."""

    from_email: str
    to_email: str
    at: str
    undo_until: str
    former_role: str = "admin"  # what the former owner became
    to_role: str = "member"  # what the receiver was before
    undone_at: str = ""

    def can_undo(self, now: datetime) -> bool:
        return not self.undone_at and datetime.fromisoformat(self.undo_until) > now


class Org(BaseModel):
    id: str
    name: str
    logo_url: str = ""
    profile: CompanyProfile = Field(default_factory=CompanyProfile)
    created_at: str = ""
    owner_email: str = ""
    members: list[Member] = Field(default_factory=list)
    license: License = Field(default_factory=License)
    workspaces: list[Workspace] = Field(default_factory=list)
    sharing: list[ShareLink] = Field(default_factory=list)
    grandfathered: bool = False
    handover: Handover | None = None  # the last ownership transfer (undo window)
    positions: list[Position] = Field(default_factory=list)  # jobs, filled or vacant

    def role_of(self, email: str) -> str | None:
        email = email.lower()
        for m in self.members:
            if m.email.lower() == email:
                return m.role
        return None

    def workspace(self, ws_id: str) -> Workspace | None:
        return next((w for w in self.workspaces if w.id == ws_id), None)

    def web_workspace(self) -> Workspace | None:
        return next((w for w in self.workspaces if w.platform == "web"), None)


class LicenseRequest(BaseModel):
    id: str
    org_id: str
    org_name: str = ""
    plan: str = "monthly"
    seats: int = 5
    note: str = ""
    requested_by: str = ""
    created_at: str = ""
    status: str = "pending"  # pending | approved | rejected
    decided_by: str = ""
    decided_at: str = ""


class WorkspaceTaken(Exception):
    """The workspace already belongs to another organization."""

    def __init__(self, org_id: str):
        super().__init__(f"workspace already belongs to {org_id}")
        self.org_id = org_id


class IdentityTaken(Exception):
    """That platform account is already linked to another member of the org."""

    def __init__(self, email: str):
        super().__init__("platform account already linked to another member")
        self.email = email


class UnknownMember(KeyError):
    """The email is not a member of the org."""


# ------------------------------------------------------------------ store
def trial_license(days: int | None = None, now: datetime | None = None) -> License:
    """The launch trial: open-ended and effectively unlimited (no payment exists yet). Pass `days`
    for a time-boxed trial (`TRIAL_DAYS` is the intended length once plans go live)."""
    until = ((now or _now()) + timedelta(days=days)).date().isoformat() if days else None
    return License(plan="trial", status="active", seats=TRIAL_SEATS, valid_until=until)


class SigninTaken(ValueError):
    """That sign-in (a Google account) already belongs to another Salid account."""


class InviteError(ValueError):
    """Why an invite link cannot be used: unknown, revoked, expired, used_up, seats."""


class Invite(BaseModel):
    """An invite link: whoever opens it and signs in joins the org with `role`. Made by an
    admin (admin/owner roles only by an owner), lives `expires_at` ("" = forever) or `max_uses`
    joins (0 = unlimited), can be revoked. `used` keeps who joined through it, so the Members
    page can say which link is live and what it brought in."""

    code: str
    org_id: str
    role: str = "member"
    label: str = ""
    created_by: str
    created_at: str
    expires_at: str = ""
    max_uses: int = 0
    used: list[str] = Field(default_factory=list)
    revoked_at: str = ""
    revoked_by: str = ""

    @property
    def uses(self) -> int:
        return len(self.used)

    def status(self, now: datetime | None = None) -> str:
        """active | revoked | expired | used_up."""
        if self.revoked_at:
            return "revoked"
        if self.expires_at and datetime.fromisoformat(self.expires_at) <= (now or _now()):
            return "expired"
        if self.max_uses and self.uses >= self.max_uses:
            return "used_up"
        return "active"


class OrgStore:
    """Organizations on a `DocStore` (SQLite or Firestore).

    The workspace and identity read-modify-writes (`attach_workspace`, `detach_workspace`,
    `link_identity`, `unlink_identity`, `drop_platform_identities`, and the code redemption in
    `salid.identity_links`) run in ONE `DocStore.transact`, so they are atomic across processes
    and store instances. The `_tx_*` helpers take the transaction handle; they read through it
    and never touch the store directly. Other writes (`save`, member edits in the web layer)
    still take only the process lock: they are admin-rate and last-writer-wins.
    """

    def __init__(self, docs: DocStore, clock=_now):
        self.docs, self._clock = docs, clock
        self._lock = self.lock = threading.RLock()

    # ---- orgs
    def get(self, org_id: str) -> Org | None:
        d = self.docs.get(ORGS, 0, org_id)
        return Org.model_validate(d) if d else None

    def all_orgs(self) -> list[Org]:
        orgs = [Org.model_validate(d) for d in self.docs.find(ORGS, 0)]
        return sorted(orgs, key=lambda o: (o.name.casefold(), o.id))

    def create(
        self,
        name: str,
        owner_email: str,
        *,
        org_id: str | None = None,
        license: License | None = None,
        members: list[Member] | None = None,
        grandfathered: bool = False,
    ) -> Org:
        now = self._clock()
        oid = org_id or f"org_{uuid.uuid4().hex[:10]}"
        owner = owner_email.lower()
        org = Org(
            id=oid,
            name=name.strip()[:120] or oid,
            created_at=stamp(now),
            owner_email=owner,
            members=members if members is not None else [Member(email=owner, role="owner")],
            license=license or trial_license(now=now),
            grandfathered=grandfathered,
        )
        org.workspaces.append(
            Workspace(
                id="web",
                platform="web",
                external_id=oid,
                name="Salid Web",
                status="connected",
                connected_at=stamp(now),
                connected_by=owner,
            )
        )
        with self._lock:
            self.save(org)
            self.docs.put(ORG_WS, org.workspaces[0].memory_id, "org", self._ws_index(org, "web"))
        return org

    @staticmethod
    def _ws_index(org: Org, ws_id: str) -> dict[str, Any]:
        return {"org_id": org.id, "workspace_id": ws_id}

    def save(self, org: Org) -> Org:
        """Replace the org document and refresh the member index."""
        with self._lock:
            old = self.get(org.id)
            self.docs.put(ORGS, 0, org.id, org.model_dump())
            now_emails = {m.email.lower() for m in org.members}
            for m in org.members:
                e = m.email.lower()
                self.docs.put(ORG_EMAIL, _email_bucket(e), org.id, {"email": e, "org_id": org.id})
            if old:
                for m in old.members:
                    e = m.email.lower()
                    if e not in now_emails:
                        self.docs.delete(ORG_EMAIL, _email_bucket(e), org.id)
        return org

    # ---- lookups
    def org_of_workspace(self, platform: str, external_id: str) -> tuple[Org, Workspace] | None:
        idx = self.docs.get(ORG_WS, workspace_memory_id(platform, external_id), "org")
        if not idx:
            return None
        org = self.get(idx["org_id"])
        ws = org.workspace(idx["workspace_id"]) if org else None
        return (org, ws) if org and ws else None

    def org_of_guild(self, guild_id: int) -> Org | None:
        found = self.org_of_workspace("discord", str(guild_id))
        return found[0] if found else None

    def org_of_memory_id(self, memory_id: int) -> tuple[Org, Workspace] | None:
        """Org + workspace owning a memory id (any platform)."""
        idx = self.docs.get(ORG_WS, memory_id, "org")
        if not idx:
            return None
        org = self.get(idx["org_id"])
        ws = org.workspace(idx["workspace_id"]) if org else None
        return (org, ws) if org and ws else None

    def orgs_of_email(self, email: str) -> list[Org]:
        e = email.lower()
        rows = self.docs.find(ORG_EMAIL, _email_bucket(e), lambda d: d.get("email") == e)
        orgs = [o for r in rows if (o := self.get(r["org_id"])) and o.role_of(e)]
        return sorted(orgs, key=lambda o: (o.name.casefold(), o.id))

    # ---- transactional building blocks (see class docstring)
    def _tx_org(self, tx, org_id: str) -> Org | None:
        d = tx.get(ORGS, 0, org_id)
        return Org.model_validate(d) if d else None

    def _tx_save(self, tx, org: Org, old: Org | None) -> None:
        """Replace the org document and refresh the member index, inside a transaction."""
        tx.put(ORGS, 0, org.id, org.model_dump())
        now_emails = {m.email.lower() for m in org.members}
        for e in now_emails:
            tx.put(ORG_EMAIL, _email_bucket(e), org.id, {"email": e, "org_id": org.id})
        for m in old.members if old else []:
            e = m.email.lower()
            if e not in now_emails:
                tx.delete(ORG_EMAIL, _email_bucket(e), org.id)

    def _tx_workspace_owner(
        self, tx, platform: str, external_id: str
    ) -> tuple[Org, Workspace] | None:
        idx = tx.get(ORG_WS, workspace_memory_id(platform, external_id), "org")
        if not idx:
            return None
        org = self._tx_org(tx, idx["org_id"])
        ws = org.workspace(idx["workspace_id"]) if org else None
        return (org, ws) if org and ws else None

    def _tx_attach(self, tx, org_id: str, ws: Workspace) -> Org:
        org = self._tx_org(tx, org_id)
        if org is None:
            raise KeyError(org_id)
        mid = ws.memory_id
        idx = tx.get(ORG_WS, mid, "org")
        if idx and idx["org_id"] != org_id and self._tx_org(tx, idx["org_id"]) is not None:
            raise WorkspaceTaken(idx["org_id"])
        old = org.model_copy(deep=True)
        org.workspaces = [w for w in org.workspaces if w.id != ws.id] + [ws]
        self._tx_save(tx, org, old)
        tx.put(ORG_WS, mid, "org", self._ws_index(org, ws.id))
        return org

    def _tx_link(
        self, tx, org_id: str, email: str, platform: str, platform_user_id: str, tenant: str = ""
    ) -> Member:
        pid = str(platform_user_id or "").strip()
        if platform not in IDENTITY_PLATFORMS or not pid or len(pid) > MAX_IDENTITY_LEN:
            raise ValueError("bad platform or account id")
        org = self._tx_org(tx, org_id)
        if org is None:
            raise KeyError(org_id)
        old = org.model_copy(deep=True)
        who = next((m for m in org.members if m.email.lower() == email.lower()), None)
        if who is None:
            raise UnknownMember(email)
        for m in org.members:
            if m is not who and m.identity(platform) == pid:
                raise IdentityTaken(m.email)
        if platform == "discord":
            who.discord_user_id = pid
            who.identities.pop("discord", None)
        else:
            who.identities[platform] = pid
        if tenant and platform != "discord":
            who.identity_tenants[platform] = str(tenant)
        else:
            who.identity_tenants.pop(platform, None)
        self._tx_save(tx, org, old)
        return who

    def _tx_unlink(
        self, tx, org_id: str, email: str, platform: str, only_if: str | None = None
    ) -> bool:
        org = self._tx_org(tx, org_id)
        who = (
            next((m for m in org.members if m.email.lower() == email.lower()), None)
            if org
            else None
        )
        if org is None or who is None or platform not in IDENTITY_PLATFORMS:
            return False
        current = who.identity(platform)
        if current is None or (only_if is not None and current != str(only_if)):
            return False
        old = org.model_copy(deep=True)
        who.identities.pop(platform, None)
        who.identity_tenants.pop(platform, None)
        if platform == "discord":
            who.discord_user_id = None
        self._tx_save(tx, org, old)
        return True

    # ---- the person (profile shared by every org they belong to)
    def user_profile(self, email: str) -> UserProfile:
        e = email.lower()
        d = self.docs.get(USER_PREFS, _email_bucket(e), e) or {}
        d.pop("updated_at", None)
        return UserProfile.model_validate({**d, "email": e})

    def _save_profile(self, p: UserProfile) -> UserProfile:
        self.docs.put(
            USER_PREFS,
            _email_bucket(p.email),
            p.email,
            {**p.model_dump(), "updated_at": stamp(self._clock())},
        )
        return p

    def user_language(self, email: str) -> str | None:
        lang = self.user_profile(email).language
        return lang if lang in UI_LANGS else None

    def set_user_language(self, email: str, language: str) -> str:
        """Remember which language `email` wants the web pages in. ValueError when unsupported."""
        if language not in UI_LANGS:
            raise ValueError(f"unsupported language {language!r}")
        with self._lock:
            p = self.user_profile(email)
            p.language = language
            self._save_profile(p)
        return language

    def set_user_name(self, email: str, name: str) -> UserProfile:
        return self.set_user_profile(email, name=name)

    PROFILE_FIELDS = {
        "name": 80,
        "first_name": 80,
        "last_name": 80,
        "title": 80,
        "department": 80,
        "about": 600,
    }

    def set_user_profile(self, email: str, **fields: str | None) -> UserProfile:
        """Update who the person is (name, title, department, about). None = leave unchanged."""
        with self._lock:
            p = self.user_profile(email)
            for k, v in fields.items():
                if k not in self.PROFILE_FIELDS:
                    raise ValueError(f"unknown profile field {k!r}")
                if v is not None:
                    setattr(p, k, " ".join(str(v).split())[: self.PROFILE_FIELDS[k]])
            return self._save_profile(p)

    def set_user_identity(self, email: str, identity: Identity) -> UserProfile:
        """Link (or replace) the person's account on one platform. A verified link also fills
        `Member.discord_user_id` in every org they belong to. ValueError for unknown platforms."""
        if identity.platform not in IDENTITY_PLATFORMS + SIGNIN_PLATFORMS:
            raise ValueError(f"unsupported platform {identity.platform!r}")
        if not identity.external_id.strip():
            raise ValueError("empty account id")
        ext = identity.external_id.strip()
        if identity.platform in SIGNIN_PLATFORMS:
            ext = ext.lower()
        identity = identity.model_copy(
            update={"external_id": ext, "linked_at": stamp(self._clock())}
        )
        with self._lock:
            p = self.user_profile(email)
            old = p.identity(identity.platform)
            p.identities = [i for i in p.identities if i.platform != identity.platform] + [identity]
            self._save_profile(p)
            if old is not None and old.external_id != identity.external_id:
                self.docs.delete(IDENTITY_INDEX, 0, f"{old.platform}:{old.external_id}")
            key = f"{identity.platform}:{identity.external_id}"
            if identity.verified:  # only a PROVEN account may sign the person in
                self.docs.put(
                    IDENTITY_INDEX, 0, key, {"email": p.email, "platform": identity.platform}
                )
            else:
                self.docs.delete(IDENTITY_INDEX, 0, key)
            if identity.platform == "discord":
                self._sync_discord_user_id(p.email, identity.external_id)
        return p

    def verified_discord_id(self, email: str) -> str | None:
        """The person's VERIFIED Discord id from their profile, else None."""
        i = self.user_profile(email).identity("discord")
        return i.external_id if i and i.verified and i.external_id else None

    def heal_identities(self, email: str) -> bool:
        """Self-repair on sign-in: every verified identity is in the sign-in index, and the
        verified Discord id is on the person's Member record in every org (people who verified
        Discord BEFORE being added to an org, or before the index existed, were missing both;
        "my tasks" and Discord sign-in depend on them). Returns True when something changed."""
        changed = False
        with self._lock:
            p = self.user_profile(email)
            for i in p.identities:
                if not i.verified or not i.external_id:
                    continue
                key = f"{i.platform}:{i.external_id}"
                have = self.docs.get(IDENTITY_INDEX, 0, key) or {}
                if have.get("email") != p.email:
                    self.docs.put(
                        IDENTITY_INDEX, 0, key, {"email": p.email, "platform": i.platform}
                    )
                    changed = True
                if i.platform == "discord":
                    for org in self.orgs_of_email(p.email):
                        for m in org.members:
                            if m.email.lower() == p.email and m.discord_user_id != i.external_id:
                                changed = True
                    self._sync_discord_user_id(p.email, i.external_id)
        return changed

    def email_for_identity(self, platform: str, external_id: str) -> str | None:
        """The person (email) who VERIFIED this platform account, so signing in with that
        account lands in the same Salid account whatever email the platform reports."""
        d = self.docs.get(IDENTITY_INDEX, 0, f"{platform}:{external_id}")
        return d.get("email") if d else None

    def link_google(self, email: str, google_email: str, name: str = "") -> UserProfile:
        """Someone who signed up through Discord (account = the email Discord reported) adds a
        Google account: from then on "Sign in with Google" with that account lands in THIS
        Salid account. IdentityTaken when that Google account is already a Salid account of
        its own (a member somewhere, a profile, or linked to someone else)."""
        e, g = email.lower(), google_email.strip().lower()
        if not g:
            raise ValueError("empty Google account")
        with self._lock:
            p = self.user_profile(e)
            if g == e:  # the account's own email: nothing to link
                return p
            owner = self.email_for_identity("google", g)
            if owner and owner != e:
                raise SigninTaken("that Google account is linked to another Salid account")
            if (
                self.orgs_of_email(g)
                or self.docs.get(USER_PREFS, _email_bucket(g), g)
                or self.docs.find(IDENTITY_INDEX, 0, lambda d: d.get("email") == g)
            ):
                raise SigninTaken("that Google account already has a Salid account of its own")
            return self.set_user_identity(
                e, Identity(platform="google", external_id=g, name=name, verified=True)
            )

    def account_for_google(self, google_email: str) -> str:
        """The Salid account a Google sign-in lands in: the account that linked this Google
        account, else the Google email itself."""
        g = google_email.strip().lower()
        return self.email_for_identity("google", g) or g

    def remove_user_identity(self, email: str, platform: str) -> UserProfile:
        with self._lock:
            p = self.user_profile(email)
            old = p.identity(platform)
            if old is not None:
                self.docs.delete(IDENTITY_INDEX, 0, f"{platform}:{old.external_id}")
            p.identities = [i for i in p.identities if i.platform != platform]
            self._save_profile(p)
            if platform == "discord":
                self._sync_discord_user_id(p.email, None)
        return p

    def _sync_discord_user_id(self, email: str, user_id: str | None) -> None:
        for org in self.orgs_of_email(email):
            changed = False
            for m in org.members:
                if m.email == email and m.discord_user_id != user_id:
                    m.discord_user_id, changed = user_id, True
            if changed:
                self.save(org)

    # ---- positions (jobs, filled or vacant)
    def save_position(
        self, org_id: str, *, by: str, pid: str | None = None, **fields: Any
    ) -> Position:
        """Create (no `pid`) or update a position. Fields: title, department, reports_to,
        summary, responsibilities, policies (text), skills (list), holders (member emails).
        None = unchanged. ValueError on bad input, KeyError for an unknown position."""
        with self._lock:
            org = self.get(org_id)
            if org is None:
                raise KeyError(org_id)
            now = stamp(self._clock())
            if pid:
                pos = next((x for x in org.positions if x.id == pid), None)
                if pos is None:
                    raise KeyError(pid)
            else:
                if len(org.positions) >= MAX_POSITIONS:
                    raise ValueError("too many positions")
                if not str(fields.get("title") or "").strip():
                    raise ValueError("a position needs a title")
                pos = Position(id="P-" + uuid.uuid4().hex[:8], title="?", created_at=now)
                org.positions.append(pos)
            for k, v in fields.items():
                if v is None:
                    continue
                if k in POSITION_LIMITS:
                    v = " ".join(str(v).split()) if k in ("title", "department") else str(v).strip()
                    if k == "title" and not v:
                        raise ValueError("a position needs a title")
                    setattr(pos, k, v[: POSITION_LIMITS[k]])
                elif k == "skills":
                    pos.skills = [" ".join(str(x).split())[:60] for x in v if str(x).strip()][:30]
                elif k == "holders":
                    members = {m.email.lower() for m in org.members}
                    hs = [str(x).strip().lower() for x in v if str(x).strip()]
                    bad = [h for h in hs if h not in members]
                    if bad:
                        raise ValueError(f"not a member: {', '.join(bad)}")
                    pos.holders = list(dict.fromkeys(hs))[:20]
                elif k == "reports_to":
                    v = str(v or "")
                    if v and (v == pos.id or not any(x.id == v for x in org.positions)):
                        raise ValueError("reports_to must be another position")
                    cur, seen = v, set()
                    while cur and cur not in seen:  # no cycles
                        if cur == pos.id:
                            raise ValueError("that would make a loop in the chart")
                        seen.add(cur)
                        cur = next((x.reports_to for x in org.positions if x.id == cur), "")
                    pos.reports_to = v
                else:
                    raise ValueError(f"unknown field {k!r}")
            pos.updated_at, pos.updated_by = now, by.lower()
            self.save(org)
            return pos

    def delete_position(self, org_id: str, pid: str) -> bool:
        with self._lock:
            org = self.get(org_id)
            if org is None or not any(x.id == pid for x in org.positions):
                return False
            org.positions = [x for x in org.positions if x.id != pid]
            for x in org.positions:
                if x.reports_to == pid:
                    x.reports_to = ""
            self.save(org)
            return True

    def set_company_profile(self, org_id: str, **fields: Any) -> Org:
        """Update the company's public profile; None leaves a field unchanged."""
        with self._lock:
            org = self.get(org_id)
            if org is None:
                raise KeyError(org_id)
            for k, v in fields.items():
                if v is None:
                    continue
                if k == "values":
                    org.profile.values = [
                        " ".join(str(x).split())[:40] for x in list(v)[:8] if str(x).strip()
                    ]
                elif k in PROFILE_LIMITS:
                    setattr(org.profile, k, " ".join(str(v).split())[: PROFILE_LIMITS[k]])
                else:
                    raise ValueError(f"unknown profile field {k!r}")
            return self.save(org)

    def set_reports_to(self, org_id: str, email: str, manager: str | None) -> Org:
        """Who `email` reports to in this org (None = top of the chart). ValueError when the
        manager is not a member or the link would make a cycle."""
        with self._lock:
            org = self.get(org_id)
            if org is None:
                raise KeyError(org_id)
            e, m = email.lower(), (manager or "").lower() or None
            members = {x.email.lower(): x for x in org.members}
            if e not in members:
                raise KeyError(email)
            if m is not None:
                if m not in members:
                    raise ValueError("manager is not a member of this organization")
                if m == e:
                    raise ValueError("a person cannot report to themselves")
                seen, cur = {e}, m  # walk up from the manager: must not reach `e`
                while cur:
                    if cur in seen:
                        raise ValueError("that would make a loop in the org chart")
                    seen.add(cur)
                    cur = (members[cur].reports_to or "").lower() or None
            members[e].reports_to = m
            return self.save(org)

    def member_profiles(self, org: Org) -> dict[str, UserProfile]:
        """Profiles of every member of `org`, by email (one read per member)."""
        return {m.email: self.user_profile(m.email) for m in org.members}

    # ---- invite links
    def create_invite(
        self,
        org_id: str,
        *,
        by: str,
        role: str = "member",
        days: int = INVITE_DAYS,
        max_uses: int = 0,
        label: str = "",
    ) -> Invite:
        if role not in ROLES:
            raise ValueError("role must be owner, admin or member")
        days, max_uses = int(days), int(max_uses)
        if not 0 <= days <= INVITE_MAX_DAYS or not 0 <= max_uses <= INVITE_MAX_USES:
            raise ValueError("days or max_uses out of range")
        org = self.get(org_id)
        if org is None:
            raise KeyError(org_id)
        now = self._clock()
        inv = Invite(
            code=secrets.token_urlsafe(18),
            org_id=org.id,
            role=role,
            label=label.strip()[:60],
            created_by=by.lower(),
            created_at=stamp(now),
            expires_at=stamp(now + timedelta(days=days)) if days else "",
            max_uses=max_uses,
        )
        self.docs.put(INVITES, 0, inv.code, inv.model_dump())
        return inv

    def invite(self, code: str) -> Invite | None:
        d = self.docs.get(INVITES, 0, code) if code else None
        return Invite.model_validate(d) if d else None

    def invites(self, org_id: str, *, active_only: bool = False) -> list[Invite]:
        """The org's invite links, newest first (active ones first when all are listed)."""
        rows = self.docs.find(INVITES, 0, lambda d: d.get("org_id") == org_id)
        now = self._clock()
        out = [Invite.model_validate(d) for d in rows]
        if active_only:
            out = [i for i in out if i.status(now) == "active"]
        out.sort(key=lambda i: i.created_at, reverse=True)
        out.sort(key=lambda i: i.status(now) != "active")
        return out

    def revoke_invite(self, org_id: str, code: str, *, by: str) -> Invite | None:
        with self._lock:
            inv = self.invite(code)
            if inv is None or inv.org_id != org_id:
                return None
            if not inv.revoked_at:
                inv.revoked_at, inv.revoked_by = stamp(self._clock()), by.lower()
                self.docs.put(INVITES, 0, inv.code, inv.model_dump())
            return inv

    def accept_invite(self, code: str, email: str) -> tuple[Org, Invite, bool]:
        """Join the org behind `code` as `email`. Returns (org, invite, joined): `joined` is
        False when the person was already a member (their role is kept, the link is not used
        up). InviteError when the link cannot be used; KeyError when the org is gone."""
        e = email.lower()
        with self._lock:
            inv = self.invite(code)
            if inv is None:
                raise InviteError("unknown")
            st = inv.status(self._clock())
            if st != "active":
                raise InviteError(st)
            org = self.get(inv.org_id)
            if org is None:
                raise KeyError(inv.org_id)
            if org.role_of(e):
                return org, inv, False
            if len(org.members) >= max(org.license.seats, 1):
                raise InviteError("seats")
            org.members.append(
                Member(email=e, role=inv.role, discord_user_id=self.verified_discord_id(e))
            )
            org = self.save(org)
            inv.used.append(e)
            self.docs.put(INVITES, 0, inv.code, inv.model_dump())
            return org, inv, True

    # ---- workspaces
    def attach_workspace(self, org_id: str, ws: Workspace) -> Org:
        """Add (or refresh) a workspace. WorkspaceTaken when another org already has it. One
        transaction: two orgs racing for the same tenant cannot both win."""
        return self.docs.transact(lambda tx: self._tx_attach(tx, org_id, ws))

    def detach_workspace(self, org_id: str, ws_id: str) -> Org:
        """Remove a workspace and every share link touching it. The web workspace stays."""

        def run(tx):
            org = self._tx_org(tx, org_id)
            ws = org.workspace(ws_id) if org else None
            if org is None or ws is None:
                raise KeyError(ws_id)
            if ws.platform == "web":
                raise ValueError("the web workspace cannot be removed")
            old = org.model_copy(deep=True)
            org.workspaces = [w for w in org.workspaces if w.id != ws_id]
            org.sharing = [
                s for s in org.sharing if ws_id not in (s.from_workspace, s.to_workspace)
            ]
            self._drop_orphan_identities(org, ws.platform)
            self._tx_save(tx, org, old)
            tx.delete(ORG_WS, ws.memory_id, "org")
            return org

        return self.docs.transact(run)

    # ---- verified platform identities (see salid.identity_links)
    @staticmethod
    def _drop_orphan_identities(org: Org, platform: str) -> None:
        """No workspace of this platform left in the org: its member links mean nothing."""
        if platform not in IDENTITY_PLATFORMS or platform == "discord":
            return
        if any(w.platform == platform for w in org.workspaces):
            return
        for m in org.members:
            m.identities.pop(platform, None)
            m.identity_tenants.pop(platform, None)

    def member_by_identity(
        self, org_id: str, platform: str, platform_user_id: str
    ) -> Member | None:
        org = self.get(org_id)
        if org is None or not platform_user_id:
            return None
        return next((m for m in org.members if m.identity(platform) == str(platform_user_id)), None)

    def update_members(self, org_id: str, fn) -> bool:
        """Atomically apply `fn(members)` to the org's member list inside one transaction (it
        edits the Member objects in place and returns True when it changed anything). Members are
        never added or removed here. Returns True when the document was written."""

        def run(tx) -> bool:
            org = self._tx_org(tx, org_id)
            if org is None or not fn(org.members):
                return False
            tx.put(ORGS, 0, org.id, org.model_dump())
            return True

        return self.docs.transact(run)

    def link_identity(
        self, org_id: str, email: str, platform: str, platform_user_id: str, tenant: str = ""
    ) -> Member:
        """Bind a platform account to a member. A second claim on the same account by another
        member is refused (`IdentityTaken`); re-linking the same member replaces the old id.
        `tenant`: the workspace key where the account was verified (recorded per platform)."""
        return self.docs.transact(
            lambda tx: self._tx_link(tx, org_id, email, platform, platform_user_id, tenant)
        )

    def unlink_identity(
        self, org_id: str, email: str, platform: str, only_if: str | None = None
    ) -> bool:
        """Remove one linked account of a member. True when something was removed. With
        `only_if`, only when the member is still linked to exactly that account id."""
        return self.docs.transact(lambda tx: self._tx_unlink(tx, org_id, email, platform, only_if))

    def drop_platform_identities(
        self, org_id: str, platform: str, keep_workspace: str = ""
    ) -> None:
        """A platform workspace was uninstalled / revoked: forget the member links of that
        platform unless the org still has another workspace there (`keep_workspace` = the one
        going away, still listed in the org document)."""
        if platform not in IDENTITY_PLATFORMS or platform == "discord":
            return

        def run(tx):
            org = self._tx_org(tx, org_id)
            if org is None:
                return
            if any(w.platform == platform and w.id != keep_workspace for w in org.workspaces):
                return
            old = org.model_copy(deep=True)
            changed = [m.identities.pop(platform, None) for m in org.members]
            for m in org.members:
                m.identity_tenants.pop(platform, None)
            if any(c is not None for c in changed):
                self._tx_save(tx, org, old)

        self.docs.transact(run)

    # ---- license requests
    def add_license_request(self, req: LicenseRequest) -> LicenseRequest:
        self.docs.put(LICENSE_REQS, 0, req.id, req.model_dump())
        return req

    def get_license_request(self, req_id: str) -> LicenseRequest | None:
        d = self.docs.get(LICENSE_REQS, 0, req_id)
        return LicenseRequest.model_validate(d) if d else None

    def license_requests(self, org_id: str | None = None) -> list[LicenseRequest]:
        rows = self.docs.find(
            LICENSE_REQS, 0, (lambda d: d.get("org_id") == org_id) if org_id else None
        )
        return sorted(
            (LicenseRequest.model_validate(d) for d in rows),
            key=lambda r: r.created_at,
            reverse=True,
        )

    # ---- web chat history (last 50 turns per user and org)
    def chat_history(self, org_id: str, email: str) -> list[dict]:
        d = self.docs.get(CENTER_CHAT, 0, f"{org_id}:{email.lower()}")
        return list((d or {}).get("turns", []))

    def append_chat(self, org_id: str, email: str, turns: list[dict], keep: int = 50) -> None:
        with self._lock:
            cur = self.chat_history(org_id, email) + turns
            self.docs.put(CENTER_CHAT, 0, f"{org_id}:{email.lower()}", {"turns": cur[-keep:]})


def make_org_store(settings, store=None) -> OrgStore:
    from .coo_store import make_doc_store

    return OrgStore(make_doc_store(settings, store))
