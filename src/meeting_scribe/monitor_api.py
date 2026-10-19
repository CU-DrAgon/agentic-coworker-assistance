"""Workflow monitor API: `/api/center/orgs/{org_id}/monitor/*` (summary, items, item detail,
change feed) and the super-admin operations view `/api/center/admin/monitor/*`.

Design: `docs/superpowers/specs/2026-10-18-salid-workflow-monitor-design.md` sections 4
(API, permissions), 6 (server behaviour) and 7 (privacy). Auth is the command center's:
`member_ctx` decides who belongs to the org, `super_user` guards the operations view.

- Every route answers 404 while `MONITOR_ENABLED` is off, even before sign-in.
- Org routes answer 402 for an org whose licence is not active, super admins included; the
  operations view is read-only and has no licence gate.
- Items come from the per-kind sources in `salid.monitor_sources`; who sees an item and which
  controls are offered is decided here by `salid.monitor` (`visible`, `controls`), never by the
  client. An item the viewer may not see answers 404, like an unknown one. An item whose
  `org_id` is not the requested org is never returned, to anyone (tenant guard).
- No response carries an item's `audience`, a prompt, a name or raw error text.
- `changes` is one change-log read: `ETag: "<seq>"`, 304 on `If-None-Match`, and at most one
  call per 2 s per person (token bucket, burst 3; in process, enough with max-instances=1).
- Store work runs in the thread pool, never on the event loop.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from meeting_scribe import hub
from meeting_scribe.center import Ctx, get_orgs, member_ctx, super_user
from meeting_scribe.config import get_settings
from salid import monitor as M
from salid import monitor_sources as S
from salid.coo_store import DocStore
from salid.licensing import license_ok
from salid.monitor import PAGE_MAX, MonitorItem, Viewer
from salid.orgs import Org, OrgStore

logger = logging.getLogger(__name__)

ZONE = ZoneInfo("Asia/Bangkok")  # `from` / `to` dates and "done today" are Bangkok days
PAGE, IDS_MAX = 50, 50  # PAGE_MAX (100) comes from salid.monitor
DEFAULT_DEEP_TIMEOUT_S = 600
BRIDGE_PLATFORMS = ("slack", "line", "telegram", "whatsapp")  # served by the `platforms` process
LIMITER_KEYS_MAX = 10_000  # people tracked by the rate limiter before idle ones are forgotten


# ------------------------------------------------------------------ collaborators (overridable)
def now() -> datetime:
    """UTC wall clock (tests replace it)."""
    return datetime.now(UTC)


def get_deep_timeout() -> int:
    """The bot's deep-task timeout in seconds (the deep-task `stuck` threshold); 600 on any
    error. Blocking on first use (it may open the Salid store): call it off the event loop."""
    try:
        return int(hub.get_salid_runtime()[0].deep_task_timeout)
    except Exception:
        return DEFAULT_DEEP_TIMEOUT_S


def monitor_enabled() -> None:
    """404 while `MONITOR_ENABLED` is off, so the feature stays hidden."""
    if not get_settings().monitor_enabled:
        raise HTTPException(404, "Not Found")


class TokenBucket:
    """Per-key rate limit: one token every `rate_s` seconds, at most `burst` saved up, so a few
    tabs polling independently are not starved. Thread-safe."""

    def __init__(
        self, rate_s: float = 2.0, burst: int = 3, clock: Callable[[], float] = time.monotonic
    ):
        self.rate_s, self.burst, self.clock = rate_s, burst, clock
        self._state: dict[str, tuple[float, float]] = {}  # key -> (tokens, at)
        self._lock = threading.Lock()

    def take(self, key: str) -> float:
        """0.0 when a token was taken (allowed), else the seconds until the next one."""
        t = self.clock()
        with self._lock:
            tokens, at = self._state.get(key, (float(self.burst), t))
            tokens = min(float(self.burst), tokens + max(0.0, t - at) / self.rate_s)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, t)
                self._forget_idle(t)
                return 0.0
            self._state[key] = (tokens, t)
            return (1.0 - tokens) * self.rate_s

    def _forget_idle(self, t: float) -> None:
        """Drop the keys whose bucket is full again (they behave like new ones) once many are
        tracked, so memory stays small."""
        if len(self._state) <= LIMITER_KEYS_MAX:
            return
        full_after = self.burst * self.rate_s
        for key in [k for k, (_, at) in self._state.items() if t - at >= full_after]:
            del self._state[key]


LIMITER = TokenBucket()


# ------------------------------------------------------------------ request context
@dataclass
class Mon:
    ctx: Ctx
    docs: DocStore
    viewer: Viewer
    when: datetime
    bot: str  # the bot's heartbeat state; "" when the org has no Discord workspace


def relevant_heartbeats(
    org: Org, beats: dict[str, dict[str, Any]], when: datetime
) -> dict[str, dict[str, str]]:
    """The heartbeats that matter to `org`: `bot` when it has a connected Discord workspace,
    `platforms` when it has a connected Slack / LINE / Telegram / WhatsApp one; each as
    `{state, tick_at}` (missing = offline). A web-only org gets `{}`. Queue sizes and versions
    span every company, so only the operations view shows them."""
    platforms = {ws.platform for ws in org.workspaces if ws.status == "connected"}
    out: dict[str, dict[str, str]] = {}
    for process, needed in (
        ("bot", "discord" in platforms),
        ("platforms", not platforms.isdisjoint(BRIDGE_PLATFORMS)),
    ):
        if needed:
            tick_at = str((beats.get(process) or {}).get("tick_at") or "")
            out[process] = {"state": M.heartbeat_state(tick_at, when), "tick_at": tick_at}
    return out


def _bot_state(org: Org, beats: dict[str, dict[str, Any]], when: datetime) -> str:
    return relevant_heartbeats(org, beats, when).get("bot", {}).get("state", "")


def _mon(ctx: Ctx, orgs: OrgStore) -> Mon:
    when = now()
    if not license_ok(ctx.org.license, when):
        raise HTTPException(402, "The organization's license is not active")
    return Mon(ctx, orgs.docs, M.viewer_for(ctx.org, ctx.email, ctx.role), when, bot="")


async def mon_ctx(
    ctx: Ctx = Depends(member_ctx),
    orgs: OrgStore = Depends(get_orgs),
    _: None = Depends(monitor_enabled),
) -> Mon:
    """The viewer and the bot state for the item routes (402 when the licence is not active)."""
    m = _mon(ctx, orgs)
    if S.discord_workspaces(ctx.org):
        beats = await run_in_threadpool(M.read_heartbeats, m.docs)
        m.bot = _bot_state(ctx.org, beats, m.when)
    return m


async def feed_ctx(
    ctx: Ctx = Depends(member_ctx),
    orgs: OrgStore = Depends(get_orgs),
    _: None = Depends(monitor_enabled),
) -> Mon:
    """`mon_ctx` without the bot state: the change feed stays one log read per poll."""
    return _mon(ctx, orgs)


@dataclass
class ItemQuery:
    kinds: list[str]
    statuses: list[str]
    attention: bool | None
    owner: str | None
    workspace: str | None
    start: datetime | None
    end: datetime | None
    q: str | None
    ids: list[tuple[str, str]] | None  # None = no batch asked for
    cursor: str | None
    limit: int


def _split(values: Iterable[str]) -> list[str]:
    """Repeated (`kind=a&kind=b`) and comma-separated (`kind=a,b`) values alike."""
    return [v.strip() for raw in values for v in raw.split(",") if v.strip()]


def parse_ids(raw: str | None) -> list[tuple[str, str]] | None:
    """`kind:id,kind:id` (at most `IDS_MAX`, else 422) as unique (kind, id) pairs, in order;
    malformed entries are dropped like unknown ones."""
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) > IDS_MAX:
        raise HTTPException(422, f"ids takes at most {IDS_MAX} items")
    keys: dict[tuple[str, str], None] = {}
    for part in parts:
        kind, sep, item_id = part.partition(":")
        if sep and kind and item_id:
            keys[(kind, item_id)] = None
    return list(keys)


def _day_start(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=ZONE)


def item_query(
    kind: Annotated[list[str] | None, Query()] = None,
    status: Annotated[list[str] | None, Query()] = None,
    attention: bool | None = None,
    owner: str | None = Query(None, max_length=200),
    workspace: str | None = Query(None, max_length=100),
    start: Annotated[date | None, Query(alias="from")] = None,
    to: date | None = None,
    priority: str | None = Query(None, max_length=40),  # accepted; tasks use it from phase 3
    q: str | None = Query(None, max_length=200),
    ids: str | None = Query(None, max_length=5_000),
    cursor: str | None = Query(None, max_length=20),
    limit: int = Query(PAGE, ge=1, le=PAGE_MAX),
) -> ItemQuery:
    """The list filters shared by the org and the operations `items` routes. `from` / `to` are
    Bangkok calendar days, both inclusive."""
    return ItemQuery(
        kinds=_split(kind or ()),
        statuses=_split(status or ()),
        attention=attention,
        owner=owner,
        workspace=workspace,
        start=_day_start(start) if start else None,
        end=_day_start(to) + timedelta(days=1, microseconds=-1) if to else None,
        q=q,
        ids=parse_ids(ids),
        cursor=cursor,
        limit=limit,
    )


# ------------------------------------------------------------------ helpers (blocking)
def _iso(at: datetime) -> str:
    return at.astimezone(UTC).isoformat()


def item_view(item: MonitorItem, viewer: Viewer) -> dict[str, Any]:
    """An item as it leaves the server: without its audience, with `mine` for this viewer."""
    return {**item.model_dump(exclude={"audience"}), "mine": M.mine(item, viewer)}


def _newest_first(items: list[MonitorItem]) -> list[MonitorItem]:
    return sorted(items, key=lambda it: (it.updated_at, it.id), reverse=True)


def _seen(m: Mon, items: Iterable[MonitorItem | None], deep_timeout_s: int) -> list[MonitorItem]:
    """The items of the requested org that `m.viewer` may see, attention applied, newest first.

    Tenant guard: an item whose `org_id` is not the requested org (or is empty) is dropped
    whoever asks, even though every source reads only that org's records."""
    out = []
    for it in items:
        if it is None or it.org_id != m.ctx.org.id:
            continue
        it = M.apply_attention(it, m.when, m.bot, deep_timeout_s=deep_timeout_s)
        if M.visible(it, m.viewer):
            out.append(it)
    return _newest_first(out)


def items_for(m: Mon, kinds: Iterable[str] | None = None) -> list[MonitorItem]:
    """The org's items (of `kinds`, default all) that `m.viewer` may see, newest first."""
    return _seen(m, S.collect(m.docs, m.ctx.org, m.when, kinds), get_deep_timeout())


def items_by_keys(m: Mon, keys: Iterable[tuple[str, str]]) -> list[MonitorItem]:
    """The items with these (kind, id) keys that exist and `m.viewer` may see, newest first."""
    found = [S.find(m.docs, m.ctx.org, kind, item_id) for kind, item_id in keys]
    return _seen(m, found, get_deep_timeout())


def _page(items: list[MonitorItem], f: ItemQuery, view: Callable[[MonitorItem], dict]) -> dict:
    hits = M.filter_items(
        items,
        kinds=f.kinds,
        statuses=f.statuses,
        attention=f.attention,
        owner=f.owner,
        workspace=f.workspace,
        start=f.start,
        end=f.end,
        q=f.q,
    )
    page, next_cursor = M.paginate(hits, f.cursor, f.limit)
    return {"items": [view(it) for it in page], "next_cursor": next_cursor}


def etag_matches(if_none_match: str | None, etag: str) -> bool:
    """Whether an `If-None-Match` header (a list, weak tags or `*`) matches `etag`."""
    for tag in (if_none_match or "").split(","):
        tag = tag.strip()
        if tag == "*" or (tag and tag.removeprefix("W/") == etag):
            return True
    return False


# ------------------------------------------------------------------ org routes
router = APIRouter(
    prefix="/api/center/orgs/{org_id}/monitor", dependencies=[Depends(monitor_enabled)]
)


@router.get("/summary")
async def summary(m: Mon = Depends(mon_ctx)) -> dict:
    """Counts by status and kind, attention, failures in 24 h, done today, the heartbeats that
    matter to this org, the server's `now` and the change-log `seq`."""

    def run() -> dict:
        tiles = M.summarize(items_for(m), m.when, ZONE)
        beats = M.read_heartbeats(m.docs)
        seq = M.read_log(m.docs, m.ctx.org.id)["seq"]
        heartbeat = relevant_heartbeats(m.ctx.org, beats, m.when)
        return {**tiles, "heartbeat": heartbeat, "now": _iso(m.when), "seq": seq}

    return await run_in_threadpool(run)


@router.get("/items")
async def list_items(m: Mon = Depends(mon_ctx), f: ItemQuery = Depends(item_query)) -> dict:
    """`{items, next_cursor}`, newest first; `ids=` refetches changed items in one batch (unknown
    and invisible keys are left out)."""

    def run() -> dict:
        items = items_by_keys(m, f.ids) if f.ids is not None else items_for(m, f.kinds or None)
        return _page(items, f, lambda it: item_view(it, m.viewer))

    return await run_in_threadpool(run)


@router.get("/items/{kind}/{item_id}")
async def item_detail(kind: str, item_id: str, m: Mon = Depends(mon_ctx)) -> dict:
    """One item with its steps, its logged events and the controls this viewer may use; 404
    when it is unknown or the viewer may not see it."""

    def run() -> dict | None:
        found = items_by_keys(m, [(kind, item_id)])
        if not found:
            return None
        it = found[0]
        events = M.item_events(M.read_log(m.docs, m.ctx.org.id), kind, item_id)
        return {**item_view(it, m.viewer), "events": events, "controls": M.controls(it, m.viewer)}

    out = await run_in_threadpool(run)
    if out is None:
        raise HTTPException(404, "Unknown item")
    return out


@router.get("/changes")
async def changes(
    since: int = Query(0, ge=0),
    if_none_match: str | None = Header(None),
    m: Mon = Depends(feed_ctx),
) -> Response:
    """`{seq, floor, now, events, reset}`: the events after `since` this viewer may see. ETag =
    the log's `seq` (304 when unchanged); 429 with `Retry-After` past 1 call per 2 s."""
    wait = LIMITER.take(m.viewer.email)
    if wait > 0:
        retry = str(max(1, math.ceil(wait)))
        raise HTTPException(429, "Too many requests", headers={"Retry-After": retry})
    log = await run_in_threadpool(M.read_log, m.docs, m.ctx.org.id)
    etag = f'"{log["seq"]}"'
    headers = {"ETag": etag, "Cache-Control": "no-store"}
    if etag_matches(if_none_match, etag):
        return Response(status_code=304, headers=headers)
    body = {**M.changes_since(log, since, m.viewer), "now": _iso(m.when)}
    return JSONResponse(body, headers=headers)


# ------------------------------------------------------------------ operations view
ops_router = APIRouter(prefix="/api/center/admin/monitor", dependencies=[Depends(monitor_enabled)])


def _ops_mons(
    email: str, orgs: OrgStore, when: datetime, beats: dict[str, dict[str, Any]]
) -> list[Mon]:
    """A read-only super-admin view of every org (DM items stay hidden)."""
    return [
        Mon(
            Ctx(email, True, org, "super"),
            orgs.docs,
            M.viewer_for(org, email, "super"),
            when,
            _bot_state(org, beats, when),
        )
        for org in orgs.all_orgs()
    ]


def _log_heads(docs: DocStore) -> dict[str, dict[str, Any]]:
    """Every change log's `seq` and last event time by key, in one read."""
    heads = {}
    for key, doc in docs.find_items(M.LOG, 0):
        events = doc.get("events") or []
        last_at = str(events[-1].get("at") or "") if events else ""
        heads[key] = {"seq": int(doc.get("seq") or 0), "last_at": last_at}
    return heads


@ops_router.get("/summary")
async def ops_summary(email: str = Depends(super_user), orgs: OrgStore = Depends(get_orgs)) -> dict:
    """Every org's counts, attention and `seq`, every process's heartbeat (with queue sizes),
    and the change logs of workspaces linked to no org."""
    when = now()

    def run() -> dict:
        heads = _log_heads(orgs.docs)
        beats = M.read_heartbeats(orgs.docs)
        rows = []
        for m in _ops_mons(email, orgs, when, beats):
            tiles = M.summarize(items_for(m), when, ZONE)
            org = m.ctx.org
            seq = heads.get(org.id, {}).get("seq", 0)
            row = {"id": org.id, "name": org.name, "counts": tiles["counts"]}
            rows.append({**row, "attention": tiles["attention"], "seq": seq})
        heartbeats = {
            process: {**beat, "state": M.heartbeat_state(beat["tick_at"], when)}
            for process, beat in beats.items()
        }
        unlinked = [{"key": key, **head} for key, head in heads.items() if key.startswith("guild:")]
        unlinked.sort(key=lambda u: u["key"])
        return {"orgs": rows, "heartbeats": heartbeats, "unlinked": unlinked, "now": _iso(when)}

    return await run_in_threadpool(run)


@ops_router.get("/items")
async def ops_items(
    email: str = Depends(super_user),
    orgs: OrgStore = Depends(get_orgs),
    f: ItemQuery = Depends(item_query),
) -> dict:
    """The org `items` filters across every org, newest first; each row carries its `org_id`."""
    when = now()

    def run() -> dict:
        viewers: dict[str, Viewer] = {}
        items: list[MonitorItem] = []
        for m in _ops_mons(email, orgs, when, M.read_heartbeats(orgs.docs)):
            viewers[m.ctx.org.id] = m.viewer
            org_items = items_for(m, f.kinds or None)
            items += [it.model_copy(update={"org_id": m.ctx.org.id}) for it in org_items]
        if f.ids is not None:
            wanted = set(f.ids)
            items = [it for it in items if (it.kind, it.id) in wanted]
        return _page(_newest_first(items), f, lambda it: item_view(it, viewers[it.org_id]))

    return await run_in_threadpool(run)
