"""Workflow monitor: the item model, who may see and control what, attention rules, error codes,
the change log, heartbeats and the list helpers.

No ``discord`` import. The models and rules are pure; only the change log and the heartbeat touch
storage, through a ``DocStore``. Design:
``docs/superpowers/specs/2026-10-18-salid-workflow-monitor-design.md`` section 3 (data model,
change log, heartbeat states, ``stuck`` thresholds), section 4.1 (permissions) and section 6.

- ``MonitorItem`` is one unit of background work (a video ad, a deep task, a change delivery, a
  queued Discord action, ...) built by a per-kind adapter from the workflow's own record, which
  stays the source of truth. It carries safe labels and account keys only (``discord:<id>``,
  emails for web actions): never prompts, briefs, message text, raw errors or person names.
- ``Audience`` decides who may see an item and its change-log events. It never leaves the server.
- ``Viewer`` is the person asking, with every verified account key they own.
- ``allowed`` / ``visible`` / ``controls`` are the access rules; ``apply_attention`` flags items
  that need a human (failed, stuck, bot offline).
- Errors shown to people are codes from ``ERROR_TEXT`` (``safe_error``), never exception text.
- The change log (DocStore kind ``salid_monitor_log``, guild 0, key = org id or
  ``guild:<memory id>``) is ``{seq, floor, events}``: a ring of the last ``RING`` content-free
  events, appended by ``emit_many`` in one transaction (best effort, never raises). Producers
  buffer in an ``Emitter`` that the bot flushes once per tick. ``changes_since`` gives a viewer
  the events they may see after a cursor, without the audience.
- Heartbeats (kind ``salid_monitor_heartbeat``, guild 0, key = process) say the bot and the
  platform bridges are alive, with their queue sizes.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import metadata
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, ValidationError

from .coo_store import DocStore
from .orgs import Org, OrgStore
from .positions import is_exec_level

logger = logging.getLogger(__name__)

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
HEARTBEAT_EVERY_S = 30.0  # how often the bot and the platform bridges write theirs
HEARTBEAT_QUEUES = ("actions", "outbox_pending", "memory")

# change log and heartbeat storage (DocStore kinds, guild 0)
LOG = "salid_monitor_log"  # key = org id, or `guild:<memory id>` for a workspace with no org
HEARTBEAT = "salid_monitor_heartbeat"  # key = process name: `bot` or `platforms`
RING = 200  # events kept per log
EVENT_V = 1  # event format version: fields are only ever added
MAX_EVENT_PEOPLE = 20  # audience people (and channel ids) copied into one event
PAGE_MAX = 100  # the largest list page
# best-effort write counters: "failed" = emit calls (and guild lookups) that wrote nothing
EMIT_STATS: Counter[str] = Counter()

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


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(at: datetime) -> str:
    """ISO UTC; a naive datetime is read as UTC."""
    return (at if at.tzinfo else at.replace(tzinfo=UTC)).astimezone(UTC).isoformat()


# ------------------------------------------------------------------ change log
def _failed(what: str, exc: BaseException) -> None:
    """Count and log a best-effort write that did not happen: the error type only, never its
    text (it may quote stored data)."""
    EMIT_STATS["failed"] += 1
    logger.warning("monitor %s failed: %s", what, type(exc).__name__)


def log_key(org_id: str | None, memory_id: int) -> str:
    """The change log of an org, or ``guild:<memory id>`` for a workspace linked to no org."""
    return org_id or f"guild:{memory_id}"


def _trimmed(audience: dict[str, Any]) -> dict[str, Any]:
    out = dict(audience)
    for field in ("people", "channel_ids"):
        if isinstance(out.get(field), list):
            out[field] = out[field][:MAX_EVENT_PEOPLE]
    return out


def event_for(item: MonitorItem) -> dict[str, Any]:
    """The change-log event of `item`: which item changed, to what, and who may see it. Never
    content (no title, error, steps or links); the audience keeps at most 20 people and 20
    channel ids."""
    return {
        "kind": item.kind,
        "id": item.id,
        "status": item.status,
        "attention": item.attention,
        "audience": _trimmed(item.audience.model_dump()),
    }


def _stamped(event: dict[str, Any], seq: int, at: str) -> dict[str, Any]:
    """Exactly the log fields of a producer event; anything else it carries is dropped."""
    return {
        "v": EVENT_V,
        "seq": seq,
        "at": at,
        "kind": str(event["kind"]),
        "id": str(event["id"]),
        "status": str(event["status"]),
        "attention": bool(event.get("attention")),
        "audience": _trimmed(event.get("audience") or {}),
    }


def emit_many(
    docs: DocStore, key: str, events: list[dict[str, Any]], *, now: datetime | None = None
) -> int | None:
    """Append `events` (as built by `event_for`) to the log `key` in ONE transaction, so `seq`
    only ever increases, and return the new `seq` (the last event's). The ring keeps the last
    `RING` events; `floor` is the highest seq no longer in it.

    Best effort: a failure is logged (error type only), counted in ``EMIT_STATS["failed"]`` and
    returns None; it never raises. An empty list writes nothing and returns None."""
    if not events:
        return None
    try:
        at = _iso(now or _utcnow())

        def append(tx) -> int:
            log = tx.get(LOG, 0, key) or {}
            seq = int(log.get("seq") or 0)
            ring = list(log.get("events") or [])
            for event in events:
                seq += 1
                ring.append(_stamped(event, seq, at))
            ring = ring[-RING:]
            tx.put(LOG, 0, key, {"seq": seq, "floor": ring[0]["seq"] - 1, "events": ring})
            return seq

        return docs.transact(append)
    except Exception as exc:
        _failed("emit", exc)
        return None


def emit(
    docs: DocStore, key: str, event: dict[str, Any], *, now: datetime | None = None
) -> int | None:
    """One event; see `emit_many`."""
    return emit_many(docs, key, [event], now=now)


def emit_items(docs: DocStore, key: str, items: list[MonitorItem]) -> None:
    """The events of `items` in one write. Best effort like `emit_many`: never raises."""
    try:
        events = [event_for(item) for item in items]
    except Exception as exc:
        _failed("emit", exc)
        return
    emit_many(docs, key, events)


def read_log(docs: DocStore, key: str) -> dict[str, Any]:
    """The log `key` as ``{seq, floor, events}``; empty (seq 0) when it does not exist."""
    doc = docs.get(LOG, 0, key) or {}
    return {
        "seq": int(doc.get("seq") or 0),
        "floor": int(doc.get("floor") or 0),
        "events": list(doc.get("events") or []),
    }


def _seq(event: dict[str, Any]) -> int:
    return int(event.get("seq") or 0)


def _public(event: dict[str, Any]) -> dict[str, Any]:
    """An event as it may leave the server: without its audience."""
    return {k: v for k, v in event.items() if k != "audience"}


def _event_allowed(event: dict[str, Any], viewer: Viewer) -> bool:
    try:
        audience = Audience.model_validate(event.get("audience") or {})
    except ValidationError:
        return False  # an unreadable audience fails closed
    return allowed(str(event.get("kind", "")), audience, viewer)


def changes_since(log: dict[str, Any], since: int, viewer: Viewer) -> dict[str, Any]:
    """``{seq, floor, events, reset}``: the events after `since` that `viewer` may see (by their
    audience, without reading any item), oldest first, without the audience.

    `reset` is true when the cursor is older than the ring (``since < floor``: events were
    missed) or ahead of the log (``since > seq``: the log was recreated); the client then reloads
    instead of patching. Gaps in `seq` are normal: other viewers' events are left out."""
    seq, floor = int(log.get("seq") or 0), int(log.get("floor") or 0)
    newer = sorted((e for e in log.get("events") or [] if _seq(e) > since), key=_seq)
    events = [_public(e) for e in newer if _event_allowed(e, viewer)]
    return {"seq": seq, "floor": floor, "events": events, "reset": since < floor or since > seq}


def _severity(event: dict[str, Any]) -> str:
    if event.get("status") == "failed":
        return "error"
    return "warning" if event.get("attention") else "info"


def item_events(log: dict[str, Any], kind: str, item_id: str) -> list[dict[str, Any]]:
    """The logged events of one item, oldest first, each with a `severity` (`error` when it
    failed, `warning` when it needed attention, else `info`), without the audience. The caller
    has already checked that the viewer may see the item."""
    own = [e for e in log.get("events") or [] if e.get("kind") == kind and e.get("id") == item_id]
    return [{**_public(e), "severity": _severity(e)} for e in sorted(own, key=_seq)]


def org_key_resolver(
    docs: DocStore, ttl_s: float = 60.0, clock: Callable[[], float] = time.monotonic
) -> Callable[[int], str]:
    """A workspace memory id -> its log key: the owning org's id, else ``guild:<id>``. Answers
    (linked or not) are cached for `ttl_s`. A failed lookup raises; the `Emitter` then drops
    that guild's events for the tick."""
    orgs = OrgStore(docs)
    cache: dict[int, tuple[float, str]] = {}
    lock = threading.Lock()

    def resolve(memory_id: int) -> str:
        t = clock()
        with lock:
            hit = cache.get(memory_id)
        if hit is not None and t - hit[0] < ttl_s:
            return hit[1]
        found = orgs.org_of_memory_id(memory_id)
        key = log_key(found[0].id if found else None, memory_id)
        with lock:
            cache[memory_id] = (t, key)
        return key

    return resolve


class Emitter:
    """Buffers change-log events between ticks; `flush` writes them with one `emit_many` per log
    key, to stay within Firestore's guidance against rapid writes to one document.

    `add` takes a log key (an org id). `add_for_guild` takes a workspace memory id that is
    resolved at flush time (off the event loop) by `resolve` (default: the unlinked key
    ``guild:<id>``). While `enabled` is false nothing is buffered or written. Thread-safe and
    never raises; events whose write fails are dropped (the client's 60 s heal covers the gap).
    """

    def __init__(
        self,
        docs: DocStore,
        *,
        enabled: bool,
        resolve: Callable[[int], str] | None = None,
        now: Callable[[], datetime] = _utcnow,
    ):
        self.docs, self.enabled, self.resolve, self.now = docs, enabled, resolve, now
        self._lock = threading.Lock()
        # (is a memory id, log key or memory id, event), in the order they were added
        self._buf: list[tuple[bool, Any, dict[str, Any]]] = []

    def add(self, key: str, event: dict[str, Any]) -> None:
        if self.enabled and key:
            with self._lock:
                self._buf.append((False, key, event))

    def add_for_guild(self, memory_id: int, event: dict[str, Any]) -> None:
        if self.enabled:
            with self._lock:
                self._buf.append((True, memory_id, event))

    def flush(self) -> int:
        """Write everything buffered; returns how many events were written."""
        with self._lock:
            pending, self._buf = self._buf, []
        if not self.enabled or not pending:
            return 0
        try:
            return self._write(pending)
        except Exception as exc:  # never break the tick
            _failed("emit", exc)
            return 0

    def _write(self, pending: list[tuple[bool, Any, dict[str, Any]]]) -> int:
        keys: dict[Any, str | None] = {}
        groups: dict[str, list[dict[str, Any]]] = {}
        for is_guild, target, event in pending:
            key = self._guild_key(target, keys) if is_guild else target
            if key:
                groups.setdefault(key, []).append(event)
        now, written = self.now(), 0
        for key, events in groups.items():
            if emit_many(self.docs, key, events, now=now) is not None:
                written += len(events)
        return written

    def _guild_key(self, memory_id: Any, keys: dict[Any, str | None]) -> str | None:
        """The log key of a memory id, looked up once per flush; None when the lookup failed."""
        if memory_id not in keys:
            try:
                resolve = self.resolve
                keys[memory_id] = resolve(memory_id) if resolve else log_key(None, memory_id)
            except Exception as exc:
                _failed("emit lookup", exc)
                keys[memory_id] = None
        return keys[memory_id]


# ------------------------------------------------------------------ heartbeat
def app_version() -> str:
    """The running revision: Cloud Run's ``K_REVISION``, else the installed package version,
    else ``dev``."""
    revision = os.environ.get("K_REVISION", "").strip()
    if revision:
        return revision
    try:
        return metadata.version("meeting-scribe")
    except metadata.PackageNotFoundError:
        return "dev"


def write_heartbeat(docs: DocStore, process: str, *, queues: dict[str, int], now: datetime) -> bool:
    """Record that `process` (`bot` or `platforms`) is alive, with its queue sizes (missing
    queues count 0). Never raises; False when the write failed."""
    try:
        doc = {
            "process": process,
            "version": app_version(),
            "tick_at": _iso(now),
            "queues": {q: int(queues.get(q) or 0) for q in HEARTBEAT_QUEUES},
        }
        docs.put(HEARTBEAT, 0, process, doc)
    except Exception as exc:
        logger.warning("monitor heartbeat failed: %s", type(exc).__name__)
        return False
    return True


def read_heartbeats(docs: DocStore) -> dict[str, dict[str, Any]]:
    """Every process's last heartbeat by process name (store fields such as `guild_id` left
    out). Feed `tick_at` to `heartbeat_state`."""
    return {
        key: {
            "process": doc.get("process") or key,
            "version": doc.get("version") or "",
            "tick_at": doc.get("tick_at") or "",
            "queues": dict(doc.get("queues") or {}),
        }
        for key, doc in docs.find_items(HEARTBEAT, 0)
    }


# ------------------------------------------------------------------ list helpers
def _bound(value: datetime | str | None) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    return _parse(value)


def _last_activity(item: MonitorItem) -> datetime | None:
    times = [_parse(t) for t in (item.started_at, item.updated_at, item.finished_at)]
    return max((t for t in times if t is not None), default=None)


def filter_items(
    items: Iterable[MonitorItem],
    *,
    kinds: Iterable[str] | None = None,
    statuses: Iterable[str] | None = None,
    attention: bool | None = None,
    owner: str | None = None,
    workspace: str | None = None,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    q: str | None = None,
) -> list[MonitorItem]:
    """The items matching every filter given, order kept; None or empty matches everything.

    - `kinds`, `statuses`: any of them; `attention`: equal.
    - `owner`: an account key or email equal (case-insensitive) to the requester or assignee.
    - `workspace`: a workspace id (``discord-1000``) or a platform (``discord``).
    - `start` / `end`: datetimes or ISO strings (naive = UTC), inclusive, on the item's latest
      time; an item with no time is left out while a bound is set. Turning a date into a
      Bangkok day is the caller's job; an unreadable bound is ignored.
    - `q`: case-insensitive text in the title, kind, id or status.
    """
    kind_set, status_set = set(kinds or ()), set(statuses or ())
    lo, hi = _bound(start), _bound(end)
    who = (owner or "").strip().casefold()
    needle = (q or "").strip().casefold()

    def keep(it: MonitorItem) -> bool:
        if kind_set and it.kind not in kind_set:
            return False
        if status_set and it.status not in status_set:
            return False
        if attention is not None and it.attention != attention:
            return False
        if who and who not in (it.requester_id.casefold(), it.assignee_id.casefold()):
            return False
        if workspace and it.workspace != workspace and not it.workspace.startswith(f"{workspace}-"):
            return False
        if lo or hi:
            at = _last_activity(it)
            if at is None or (lo and at < lo) or (hi and at > hi):
                return False
        text = f"{it.title} {it.kind} {it.id} {it.status} {it.raw_status}".casefold()
        return not needle or needle in text

    return [it for it in items if keep(it)]


def paginate[T](items: Sequence[T], cursor: str | None, limit: int) -> tuple[list[T], str | None]:
    """One page at the offset `cursor` (the documents pattern: ``"50"``; anything else is the
    start) and the next cursor, None on the last page. `limit` is kept within 1..100."""
    start = int(cursor) if cursor and cursor.isascii() and cursor.isdigit() else 0
    limit = max(1, min(int(limit), PAGE_MAX))
    end = start + limit
    return list(items[start:end]), (str(end) if end < len(items) else None)


def summarize(items: Iterable[MonitorItem], now: datetime, zone: ZoneInfo) -> dict[str, Any]:
    """The summary tiles: counts by status and by kind (non-zero only), items needing attention,
    failures finished in the last 24 h, and items done today (the calendar date in `zone`, e.g.
    Bangkok). An item finished at `finished_at`, else at `updated_at`."""
    items = list(items)
    today = now.astimezone(zone).date()
    failed_24h = done_today = 0
    for it in items:
        at = _parse(it.finished_at or it.updated_at)
        if at is None:
            continue
        if it.status == "failed" and now - at <= timedelta(hours=24):
            failed_24h += 1
        if it.status == "done" and at.astimezone(zone).date() == today:
            done_today += 1
    return {
        "counts": dict(Counter(it.status for it in items)),
        "by_kind": dict(Counter(it.kind for it in items)),
        "attention": sum(1 for it in items if it.attention),
        "failed_24h": failed_24h,
        "done_today": done_today,
    }
