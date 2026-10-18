"""Workflow monitor sources: where the items of each kind come from.

No ``discord`` import. A source turns one kind's own records (the source of truth) into
``MonitorItem``s for one org; a finder looks one item up by id. The per-kind adapters register
themselves here when the module is imported (today: the change outbox and the web -> Discord
action queue; later phases: video ads, deep tasks, research); this module holds the registry and
the helpers the web API (``meeting_scribe.monitor_api``) calls. Design:
``docs/superpowers/specs/2026-10-18-salid-workflow-monitor-design.md`` sections 3.2 and 4.

Sources return every item of their kind they may hold; who may see an item is decided later by
``monitor.visible``. Every adapter MUST set ``org_id`` to the org it was read for: the API drops
any item whose ``org_id`` is not the org asked about (a tenant guard on top of reading only that
org's records). A raising source or finder is logged (the error type only, never its text, which
may quote stored data) and skipped, so one broken kind never empties the whole monitor.

Adapters copy only safe labels, states, times and account keys: never action ``args``, raw
errors (``error``, ``last_error``), result fields other than a Discord link, or a change's
subject (email, name, before / after values). Errors become codes (``monitor.safe_error``).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from .actions import KINDS as ACTION_KINDS
from .actions import ActionQueue
from .change_outbox import KINDS as CHANGE_KINDS
from .change_outbox import MAX_ATTEMPTS, RESEND_MINUTES, Change, ChangeOutbox, Target
from .coo_store import DocStore
from .monitor import ACTIVE, FINAL, Audience, MonitorItem, Step, safe_error
from .orgs import Org, Workspace

logger = logging.getLogger(__name__)

Source = Callable[[DocStore, Org, datetime], list[MonitorItem]]
Finder = Callable[[DocStore, Org, str], MonitorItem | None]
SOURCES: dict[str, tuple[Source, Finder]] = {}  # kind -> (source, finder)
HISTORY_MAX = 500  # finished items listed per kind and org (the newest); active ones are all kept


def register(kind: str, source: Source, finder: Finder) -> None:
    """Make `kind` listable (`source`) and findable by id (`finder`); a later call replaces it."""
    SOURCES[kind] = (source, finder)


def discord_workspaces(org: Org) -> list[Workspace]:
    """The org's connected Discord workspaces (where the bot keeps its records)."""
    return [ws for ws in org.workspaces if ws.platform == "discord" and ws.status == "connected"]


def _newest_history(items: list[MonitorItem]) -> list[MonitorItem]:
    """Every active item plus the newest `HISTORY_MAX` finished ones."""
    active = [it for it in items if it.status in ACTIVE]
    finished = [it for it in items if it.status not in ACTIVE]
    finished.sort(key=lambda it: (it.updated_at, it.id), reverse=True)
    return active + finished[:HISTORY_MAX]


def _skipped(what: str, kind: str, exc: BaseException) -> None:
    logger.warning("monitor %s %s failed: %s", what, kind, type(exc).__name__)


def collect(
    docs: DocStore, org: Org, now: datetime, kinds: Iterable[str] | None = None
) -> list[MonitorItem]:
    """The items of every registered kind (or only `kinds`) for `org`, unfiltered by viewer. A
    raising source is logged and skipped."""
    wanted = None if kinds is None else set(kinds)
    out: list[MonitorItem] = []
    for kind, (source, _finder) in list(SOURCES.items()):
        if wanted is not None and kind not in wanted:
            continue
        try:
            items = list(source(docs, org, now))
        except Exception as exc:
            _skipped("source", kind, exc)
            continue
        out.extend(_newest_history(items))
    return out


def find(docs: DocStore, org: Org, kind: str, item_id: str) -> MonitorItem | None:
    """One item of `kind` by id, or None (unknown kind or id; a raising finder is logged)."""
    entry = SOURCES.get(kind)
    if entry is None:
        return None
    try:
        return entry[1](docs, org, item_id)
    except Exception as exc:
        _skipped("finder", kind, exc)
        return None


# ------------------------------------------------------------------ adapters: shared
def wall_clock() -> datetime:
    """UTC wall clock for the finders, which are given no time (tests replace it)."""
    return datetime.now(UTC)


def _utc(ts: Any) -> datetime | None:
    """An ISO timestamp as an aware UTC datetime (naive = UTC); None when missing or invalid."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        at = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return (at if at.tzinfo else at.replace(tzinfo=UTC)).astimezone(UTC)


def _iso(at: datetime | None) -> str:
    return at.isoformat() if at else ""


# ------------------------------------------------------------------ change outbox
NOT_CONFIRMED = "not confirmed in time"  # `ChangeOutbox.due`'s error for a target out of time
_TARGET_STATE = {"pending": "pending", "sent": "running", "acked": "done", "failed": "failed"}


def _change_status(statuses: list[str]) -> str:
    """Any failed target -> failed, else any pending -> queued, else any sent (awaiting the
    ack) -> waiting, else done (a change with no targets is done)."""
    if "failed" in statuses:
        return "failed"
    if "pending" in statuses:
        return "queued"
    if "sent" in statuses:
        return "waiting"
    return "done"


def _target_step(x: Target) -> Step:
    """One target as a step. A failed target, or one whose last attempt reported an error, gets
    an error code (`not_confirmed` when it ran out of time, else `delivery_failed`), never the
    error text."""
    state = _TARGET_STATE.get(x.status, "pending")
    code = ""
    if state == "failed" or x.last_error:
        code = "not_confirmed" if x.last_error == NOT_CONFIRMED else "delivery_failed"
    return Step(
        name=f"{x.platform} {x.kind}",
        state=state,
        started_at=_iso(_utc(x.sent_at)),
        attempts=x.attempts,
        error_code=code,
    )


def adapt_change(
    change: Change,
    *,
    now: datetime,
    resend_minutes: int = RESEND_MINUTES,
    max_attempts: int = MAX_ATTEMPTS,
) -> MonitorItem:
    """One change as an item; its targets are the steps. Still open past the outbox's deadline
    (`at + resend * (max_attempts + 1)`, e.g. a platform no running process serves) -> attention
    `stuck`; failed -> attention `failed` with the first failed target's error code. Admins and
    owners only; no subject email, name or values."""
    targets = change.targets
    status = _change_status([x.status for x in targets])
    steps = [_target_step(x) for x in targets]
    born = _utc(change.at)
    times = [born, *(_utc(x.sent_at) for x in targets), *(_utc(x.acked_at) for x in targets)]
    updated = max((t for t in times if t is not None), default=None)
    reason = ""
    if status == "failed":
        reason = "failed"
    elif status in ACTIVE and born is not None:
        resend = timedelta(minutes=max(int(resend_minutes), 1))
        deadline = born + resend * (max(int(max_attempts), 1) + 1)
        at = now if now.tzinfo else now.replace(tzinfo=UTC)
        if at >= deadline:
            reason = "stuck"
    code = next((s.error_code for s in steps if s.state == "failed"), "")
    acked = sum(1 for x in targets if x.status == "acked")
    kind = change.kind if change.kind in CHANGE_KINDS else "other"
    return MonitorItem(
        kind="change",
        id=change.id,
        org_id=change.org_id,
        workspace="web",
        title=f"Change · {kind}",
        status=status,
        attention=bool(reason),
        attention_reason=reason,
        raw_status=f"{acked}/{len(targets)} acked",
        steps=steps,
        started_at=_iso(born),
        updated_at=_iso(updated),
        finished_at=_iso(updated) if status in FINAL else "",
        error=safe_error(code) if status == "failed" else None,
        audience=Audience(admin_only=True),
    )


def changes_source(docs: DocStore, org: Org, now: datetime) -> list[MonitorItem]:
    """Every change of `org` (one read of the org's outbox guild)."""
    changes = ChangeOutbox(docs).list_changes(org.id, limit=sys.maxsize)
    return [adapt_change(c, now=now) for c in changes]


def changes_finder(docs: DocStore, org: Org, item_id: str) -> MonitorItem | None:
    """One change of `org` by id, read from the org's own outbox guild only."""
    if not org.id:
        return None
    change = ChangeOutbox(docs).get(item_id, org_id=org.id)
    if change is None or change.org_id != org.id:
        return None
    return adapt_change(change, now=wall_clock())


# ------------------------------------------------------------------ web -> Discord actions
DISCORD_URL = "https://discord.com/"  # the only links an action item carries
_ACTION_STATUS = {"pending": "queued", "done": "done", "failed": "failed"}


def action_error_code(error: str) -> str:
    """The `monitor.ERROR_TEXT` code of an action's stored error (as written by
    `actions_bot._actions_due` / `ActionQueue.expire`); "" when there is none."""
    text = error.strip() if isinstance(error, str) else ""
    if not text:
        return ""
    if text == "missing permission":
        return "missing_permission"
    if text.startswith("expired"):
        return "expired"
    if text.startswith("channel not found"):
        return "not_found"
    if text.startswith("discord:"):
        return "discord_error"
    if text == "bad request":
        return "bad_request"
    return "error"


def adapt_action(rec: dict) -> MonitorItem:
    """One queued web -> Discord action as an item: pending -> queued (the API adds `stuck` after
    60 s while the bot is online), done -> done, failed -> failed with attention (an unknown
    state also counts as failed). Admins and owners only; the requester is the email that queued
    it. Never the action's `args`, its raw `error` or result fields other than a Discord link.
    `org_id` is left for the source to set."""
    raw = str(rec.get("status") or "")
    status = _ACTION_STATUS.get(raw, "failed")
    by = str(rec.get("by") or "").lower()
    created, done = _utc(rec.get("created_at")), _utc(rec.get("done_at"))
    code = action_error_code(str(rec.get("error") or ""))
    result = rec.get("result")
    url = result.get("url") if isinstance(result, dict) else None
    kind = rec.get("kind") if rec.get("kind") in ACTION_KINDS else "other"
    return MonitorItem(
        kind="action",
        id=str(rec.get("id") or ""),
        workspace=f"discord-{rec.get('guild_id')}",
        title=f"Discord action · {kind}",
        status=status,
        attention=status == "failed",
        attention_reason="failed" if status == "failed" else "",
        raw_status=raw,
        requester_id=by,
        started_at=_iso(created),
        updated_at=_iso(done or created),
        finished_at=_iso(done) if status in FINAL else "",
        error=safe_error(code) if code else None,
        links={"discord": url} if isinstance(url, str) and url.startswith(DISCORD_URL) else {},
        audience=Audience(admin_only=True, people=[by] if by else []),
    )


def _org_action(rec: dict, org: Org) -> MonitorItem:
    return adapt_action(rec).model_copy(update={"org_id": org.id})


def actions_source(docs: DocStore, org: Org, now: datetime) -> list[MonitorItem]:
    """Every action queued for the org's connected Discord servers."""
    queue = ActionQueue(docs)
    return [
        _org_action(rec, org)
        for ws in discord_workspaces(org)
        for rec in queue.recent(ws.memory_id, limit=sys.maxsize)
        if rec.get("id")
    ]


def actions_finder(docs: DocStore, org: Org, item_id: str) -> MonitorItem | None:
    """One action by id, looked up in each of the org's connected Discord servers."""
    queue = ActionQueue(docs)
    for ws in discord_workspaces(org):
        rec = queue.get(ws.memory_id, item_id)
        if rec:
            return _org_action(rec, org)
    return None


register("change", changes_source, changes_finder)
register("action", actions_source, actions_finder)
