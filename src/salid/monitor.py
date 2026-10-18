"""Workflow monitor: the item model, who may see and control what, attention rules, error codes.

Pure module: no ``discord`` import and no I/O. Design:
``docs/superpowers/specs/2026-10-18-salid-workflow-monitor-design.md`` section 3 (data model,
heartbeat states, ``stuck`` thresholds) and section 4.1 (permissions).

- ``MonitorItem`` is one unit of background work (a video ad, a deep task, a change delivery, a
  queued Discord action, ...) built by a per-kind adapter from the workflow's own record, which
  stays the source of truth. It carries safe labels and account keys only (``discord:<id>``,
  emails for web actions): never prompts, briefs, message text, raw errors or person names.
- ``Audience`` decides who may see an item and its change-log events. It never leaves the server.
- ``Viewer`` is the person asking, with every verified account key they own.
- ``allowed`` / ``visible`` / ``controls`` are the access rules; ``apply_attention`` flags items
  that need a human (failed, stuck, bot offline).
- Errors shown to people are codes from ``ERROR_TEXT`` (``safe_error``), never exception text.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .orgs import Org
from .positions import is_exec_level

KINDS = (
    "video_ad",
    "deep_task",
    "research",
    "change",
    "action",
    "meeting",
    "task",
    "approval",
    "vote",
    "form",
    "reminder",
)
ACTIVE = ("queued", "running", "waiting", "paused")
FINAL = ("done", "failed", "cancelled")
STATUSES = ACTIVE + FINAL
# kinds the bot carries out: while its heartbeat is offline their active items get `bot_offline`
BOT_KINDS = ("video_ad", "deep_task", "research", "action")
# kinds a person may cancel while they are queued or running (release 2 control)
CANCELLABLE = ("video_ad", "deep_task", "research")

# `stuck` thresholds in seconds (spec 3.2): a running item not updated for this long
VIDEO_STUCK_S = 600  # one video step
RESEARCH_STUCK_S = 900
DEEP_TASK_GRACE_S = 120  # added to the deep-task timeout
ACTION_STUCK_S = 60  # a queued Discord action not picked up while the bot is online

# heartbeat age in seconds (spec 3.4): online <= 90 < delayed <= 300 < offline
HEARTBEAT_ONLINE_S = 90
HEARTBEAT_DELAYED_S = 300

# the only error text that reaches people: one short English sentence per code
ERROR_TEXT: dict[str, str] = {
    "unknown": "Something went wrong.",
    "restarted": "Salid restarted while this was running.",
    "failed": "The job did not finish.",
    "error": "An unexpected error stopped this job.",
    "month_limit": "This month's limit has been reached.",
    "storyboard": "The storyboard could not be planned.",
    "render": "The video could not be rendered.",
    "upload_refused": "The file was made but could not be attached.",
    "not_confirmed": "The platform did not confirm delivery in time.",
    "delivery_failed": "The change could not be delivered.",
    "missing_permission": "Salid lacks a permission needed for this.",
    "not_found": "The target could not be found.",
    "bad_request": "The request was rejected as invalid.",
    "discord_error": "Discord returned an error.",
    "expired": "It expired before the bot could do it.",
}


# ------------------------------------------------------------------ models
class Step(BaseModel):
    name: str
    state: str = "pending"  # pending | running | done | failed
    started_at: str = ""  # ISO UTC
    ms: int | None = None
    attempts: int = 0
    error_code: str = ""  # a key of ERROR_TEXT, or ""


class Audience(BaseModel):
    """Who may see an item (and its change-log events). Server-side only."""

    admin_only: bool = False  # owners and admins only, even if listed in `people`
    exec_only: bool = False  # admins must also be executives (deep tasks)
    dm: bool = False  # private conversation: only the people listed, never the owner
    people: list[str] = Field(default_factory=list)  # account keys (`person_key`) and emails
    channel_ids: list[str] = Field(default_factory=list)


class SafeError(BaseModel):
    code: str
    message: str


class MonitorItem(BaseModel):
    kind: str  # one of KINDS
    id: str
    org_id: str = ""
    workspace: str = ""  # platform + workspace memory id, e.g. "discord-1000", "web"
    title: str = ""  # safe English label built by the adapter, never content
    status: str  # one of STATUSES
    attention: bool = False
    attention_reason: str = ""  # failed | stuck | waiting_too_long | overdue | bot_offline
    raw_status: str = ""  # the workflow's own value
    requester_id: str = ""  # account keys where they exist
    assignee_id: str = ""
    waiting_on: str = ""
    steps: list[Step] = Field(default_factory=list)
    started_at: str = ""  # ISO UTC
    updated_at: str = ""
    finished_at: str = ""
    cost_usd: float | None = None
    error: SafeError | None = None
    links: dict[str, str] = Field(default_factory=dict)
    audience: Audience = Field(default_factory=Audience)

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.id}"


@dataclass(frozen=True)
class Viewer:
    email: str
    role: str  # owner | admin | member | super (a super admin who is not in the org)
    ids: frozenset[str]  # lowercased email + every verified account key
    executive: bool = False


# ------------------------------------------------------------------ people
def person_key(platform: str, account_id: str | int) -> str:
    """The account key items and audiences use, e.g. ``discord:123``."""
    return f"{platform}:{account_id}"


def viewer_for(org: Org, email: str, role: str) -> Viewer:
    """The viewer for `email` in `org` with the role the center resolved for them.

    An executive is the owner, or an admin or member whose mirrored position level is an
    executive one (the web cannot see Discord roles, so this fails closed). For a member it only
    unlocks cancelling their own deep tasks, never what they see. A non-member (a super admin)
    only has their email and is never an executive.
    """
    email = email.lower()
    ids = {email}
    executive = False
    member = next((m for m in org.members if m.email.lower() == email), None)
    if member is not None:
        ids.update(person_key(p, i) for p, i in member.all_identities().items())
        executive = role == "owner" or (role in ("admin", "member") and is_exec_level(member.level))
    return Viewer(email, role, frozenset(ids), executive)


# ------------------------------------------------------------------ access
def allowed(kind: str, audience: Audience, viewer: Viewer) -> bool:
    """Whether `viewer` may see an item or event of `kind` with `audience` (spec 4.1)."""
    listed = not viewer.ids.isdisjoint(audience.people)
    if audience.dm:  # a private conversation: only the people in it, never the owner
        return listed
    if viewer.role in ("owner", "super"):
        return True
    if viewer.role == "admin":
        # deep tasks are executive-only (spec 4.1) even if an adapter forgot the flag
        if (audience.exec_only or kind == "deep_task") and not viewer.executive:
            return listed
        return True
    if audience.admin_only:
        return False
    return listed


def visible(item: MonitorItem, viewer: Viewer) -> bool:
    return allowed(item.kind, item.audience, viewer)


def _is_viewer(key: str, viewer: Viewer) -> bool:
    return bool(key) and key in viewer.ids


def mine(item: MonitorItem, viewer: Viewer) -> bool:
    """The viewer requested the item, is assigned to it, or it waits on them."""
    keys = (item.requester_id, item.assignee_id, item.waiting_on)
    return any(_is_viewer(k, viewer) for k in keys)


def controls(item: MonitorItem, viewer: Viewer) -> list[str]:
    """The actions `viewer` may take on `item`; never any on an item they cannot see."""
    if viewer.role == "super" or not visible(item, viewer):
        return []
    staff = viewer.role in ("owner", "admin")
    out: list[str] = []
    if item.kind in CANCELLABLE and item.status in ("queued", "running"):
        requester = viewer.role == "member" and _is_viewer(item.requester_id, viewer)
        if staff or (requester and (item.kind != "deep_task" or viewer.executive)):
            out.append("cancel")
    if staff and item.status == "failed":
        if item.kind == "action":
            out.append("retry")
        elif item.kind == "change":
            out.append("resend")
    return out


def safe_error(code: str) -> SafeError:
    """A code from ERROR_TEXT with its fixed text; anything else is `unknown`, never echoed."""
    key = code if isinstance(code, str) and code in ERROR_TEXT else "unknown"
    return SafeError(code=key, message=ERROR_TEXT[key])


# ------------------------------------------------------------------ time
def _parse(ts: str | None) -> datetime | None:
    """An ISO timestamp as an aware datetime (naive = UTC), or None when missing or invalid."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        at = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return at if at.tzinfo else at.replace(tzinfo=UTC)


def heartbeat_state(tick_at: str | None, now: datetime) -> str:
    """online / delayed / offline from the last heartbeat; a future `tick_at` counts as age 0."""
    at = _parse(tick_at)
    if at is None:
        return "offline"
    age = max(0.0, (now - at).total_seconds())
    if age <= HEARTBEAT_ONLINE_S:
        return "online"
    if age <= HEARTBEAT_DELAYED_S:
        return "delayed"
    return "offline"


def stuck_after_s(kind: str, deep_timeout_s: int = 600) -> int | None:
    """Seconds without an update after which a running item of `kind` is stuck (None = never)."""
    if kind == "video_ad":
        return VIDEO_STUCK_S
    if kind == "deep_task":
        return deep_timeout_s + DEEP_TASK_GRACE_S
    if kind == "research":
        return RESEARCH_STUCK_S
    return None


def _attention_reason(item: MonitorItem, now: datetime, bot: str, deep_timeout_s: int) -> str:
    if item.status == "failed":
        return "failed"
    if bot == "offline" and item.kind in BOT_KINDS and item.status in ACTIVE:
        return "bot_offline"
    at = _parse(item.updated_at or item.started_at)
    if at is None:
        return ""
    age = (now - at).total_seconds()
    limit = stuck_after_s(item.kind, deep_timeout_s)
    if item.status == "running" and limit is not None and age > limit:
        return "stuck"
    queued_action = item.kind == "action" and item.status == "queued"
    if queued_action and bot == "online" and age > ACTION_STUCK_S:
        return "stuck"
    return ""


def apply_attention(
    item: MonitorItem, now: datetime, bot: str = "online", *, deep_timeout_s: int = 600
) -> MonitorItem:
    """The item with attention set by the shared rules, in order: failed, bot offline (bot-run
    kinds, active), stuck (running past its kind's threshold), a queued action not picked up
    within 60 s while the bot is online. Otherwise the adapter's own attention is kept (changes
    set their own `stuck`). `bot` is the `heartbeat_state` of the bot. The input is never changed.
    """
    reason = _attention_reason(item, now, bot, deep_timeout_s)
    if not reason:
        return item
    return item.model_copy(update={"attention": True, "attention_reason": reason})
