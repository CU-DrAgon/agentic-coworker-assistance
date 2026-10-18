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

ROLES = ("owner", "admin", "member")
PLATFORMS = ("discord", "slack", "line", "telegram", "whatsapp", "clickup", "web")
# Platforms where a person has a chat account that can be linked to an org member
IDENTITY_PLATFORMS = ("discord", "slack", "line", "telegram", "whatsapp")
MAX_IDENTITY_LEN = 200
PLANS = ("trial", "monthly", "yearly", "enterprise")
STATUSES = ("active", "past_due", "expired", "suspended")
CATEGORIES = ("documents", "wiki", "places", "minutes", "ledger", "tasks")
NAMESPACE_BASE = 2**62
TRIAL_DAYS = 14


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


# ------------------------------------------------------------------ models
class Member(BaseModel):
    email: str
    role: str = "member"  # owner | admin | member
    discord_user_id: str | None = None
    # platform -> platform user id, VERIFIED by Salid (Slack profile email, or a one-time link
    # code). Discord stays in `discord_user_id`; read it through `identity("discord")`.
    identities: dict[str, str] = Field(default_factory=dict)
    department: str = ""  # PM dashboard: tech | business | finance | other | ""
    finance: bool = False  # PM dashboard: Authorized Approver, may see budgets

    def identity(self, platform: str) -> str | None:
        """The verified platform account id of this member, or None."""
        if platform == "discord":
            return self.identities.get("discord") or self.discord_user_id or None
        return self.identities.get(platform) or None

    def all_identities(self) -> dict[str, str]:
        out = {k: v for k, v in self.identities.items() if v}
        if self.discord_user_id and "discord" not in out:
            out["discord"] = self.discord_user_id
        return out


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


class Org(BaseModel):
    id: str
    name: str
    logo_url: str = ""
    created_at: str = ""
    owner_email: str = ""
    members: list[Member] = Field(default_factory=list)
    license: License = Field(default_factory=License)
    workspaces: list[Workspace] = Field(default_factory=list)
    sharing: list[ShareLink] = Field(default_factory=list)
    grandfathered: bool = False

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
def trial_license(days: int = TRIAL_DAYS, now: datetime | None = None) -> License:
    until = ((now or _now()) + timedelta(days=days)).date().isoformat()
    return License(plan="trial", status="active", seats=5, valid_until=until)


class OrgStore:
    """Organizations on a `DocStore` (SQLite or Firestore). Read-modify-write helpers take a
    process lock; cross-instance races on the same org are accepted (admin-rate writes)."""

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

    # ---- workspaces
    def attach_workspace(self, org_id: str, ws: Workspace) -> Org:
        """Add (or refresh) a workspace. WorkspaceTaken when another org already has it."""
        with self._lock:
            org = self.get(org_id)
            if org is None:
                raise KeyError(org_id)
            mid = ws.memory_id
            idx = self.docs.get(ORG_WS, mid, "org")
            if idx and idx["org_id"] != org_id and self.get(idx["org_id"]) is not None:
                raise WorkspaceTaken(idx["org_id"])
            org.workspaces = [w for w in org.workspaces if w.id != ws.id] + [ws]
            self.save(org)
            self.docs.put(ORG_WS, mid, "org", self._ws_index(org, ws.id))
            return org

    def detach_workspace(self, org_id: str, ws_id: str) -> Org:
        """Remove a workspace and every share link touching it. The web workspace stays."""
        with self._lock:
            org = self.get(org_id)
            ws = org.workspace(ws_id) if org else None
            if org is None or ws is None:
                raise KeyError(ws_id)
            if ws.platform == "web":
                raise ValueError("the web workspace cannot be removed")
            org.workspaces = [w for w in org.workspaces if w.id != ws_id]
            org.sharing = [
                s for s in org.sharing if ws_id not in (s.from_workspace, s.to_workspace)
            ]
            self._drop_orphan_identities(org, ws.platform)
            self.save(org)
            self.docs.delete(ORG_WS, ws.memory_id, "org")
            return org

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

    def member_by_identity(
        self, org_id: str, platform: str, platform_user_id: str
    ) -> Member | None:
        org = self.get(org_id)
        if org is None or not platform_user_id:
            return None
        return next((m for m in org.members if m.identity(platform) == str(platform_user_id)), None)

    def link_identity(
        self, org_id: str, email: str, platform: str, platform_user_id: str
    ) -> Member:
        """Bind a platform account to a member. A second claim on the same account by another
        member is refused (`IdentityTaken`); re-linking the same member replaces the old id."""
        pid = str(platform_user_id or "").strip()
        if platform not in IDENTITY_PLATFORMS or not pid or len(pid) > MAX_IDENTITY_LEN:
            raise ValueError("bad platform or account id")
        with self._lock:
            org = self.get(org_id)
            if org is None:
                raise KeyError(org_id)
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
            self.save(org)
            return who

    def unlink_identity(self, org_id: str, email: str, platform: str) -> bool:
        """Remove one linked account of a member. True when something was removed."""
        with self._lock:
            org = self.get(org_id)
            who = (
                next((m for m in org.members if m.email.lower() == email.lower()), None)
                if org
                else None
            )
            if org is None or who is None or platform not in IDENTITY_PLATFORMS:
                return False
            had = who.identity(platform) is not None
            who.identities.pop(platform, None)
            if platform == "discord":
                who.discord_user_id = None
            if had:
                self.save(org)
            return had

    def drop_platform_identities(
        self, org_id: str, platform: str, keep_workspace: str = ""
    ) -> None:
        """A platform workspace was uninstalled / revoked: forget the member links of that
        platform unless the org still has another workspace there (`keep_workspace` = the one
        going away, still listed in the org document)."""
        if platform not in IDENTITY_PLATFORMS or platform == "discord":
            return
        with self._lock:
            org = self.get(org_id)
            if org is None:
                return
            if any(w.platform == platform and w.id != keep_workspace for w in org.workspaces):
                return
            changed = [m.identities.pop(platform, None) for m in org.members]
            if any(c is not None for c in changed):
                self.save(org)

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
