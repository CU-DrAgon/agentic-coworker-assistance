"""Workflow monitor sources: where the items of each kind come from.

No ``discord`` import. A source turns one kind's own records (the source of truth) into
``MonitorItem``s for one org; a finder looks one item up by id. The per-kind adapters register
themselves here (later phases: change outbox, action queue, video ads, deep tasks, research);
this module holds the registry and the helpers the web API (``meeting_scribe.monitor_api``)
calls. Design: ``docs/superpowers/specs/2026-10-18-salid-workflow-monitor-design.md`` sections
3.2 and 4.

Sources return every item of their kind they may hold; who may see an item is decided later by
``monitor.visible``. A raising source or finder is logged (the error type only, never its text,
which may quote stored data) and skipped, so one broken kind never empties the whole monitor.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from datetime import datetime

from .coo_store import DocStore
from .monitor import ACTIVE, MonitorItem
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
