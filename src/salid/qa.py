from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.models import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

from .chat_tools import ChatActions, build_chat_tools
from .config import Settings
from .contests import CONTESTS_PROMPT
from .currency import CURRENCY_PROMPT
from .embed import NullEmbedder, cosine
from .flights import FLIGHT_PROMPT
from .freetime import FREETIME_PROMPT
from .gate import extract_keywords
from .gmaps import MAPS_PROMPT
from .i18n import resolve_lang, t
from .ideas import IDEAS_PROMPT
from .markets import MARKETS_PROMPT
from .media import parse_message_ref
from .memory_access import Visibility, hidden_but_askable, is_dm_scope, message_visible
from .memory_index import entries_for
from .memory_store import MAX_ROWS, MemoryMessage, MemoryStore
from .memory_summary import render_summary, summary_items
from .places import PLACES_PROMPT
from .search import SEARCH_PROMPT
from .secretary import SecretaryContext, SecretaryTools, build_secretary_tools
from .store import Store, duplicate_extras, is_active, pick_keep
from .web_read import make_read_web
from .wiki import WIKI_PROMPT

logger = logging.getLogger(__name__)
MAX_ROUNDS = 6
APP_NAME = "salid"
# Vertex 429s (shared quota) and 5xx are retried with backoff by the genai HTTP client
# (same policy as meeting_scribe.agents).
RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=5, initial_delay=2.0, max_delay=30.0, http_status_codes=[429, 500, 502, 503, 504]
)
# Answer field -> usage_metadata attribute (Vertex token accounting per model call).
USAGE_FIELDS = {
    "tokens_in": "prompt_token_count",
    "tokens_cached": "cached_content_token_count",
    "tokens_out": "candidates_token_count",
    "tokens_thought": "thoughts_token_count",
}
THAI_DAYS = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]


def baht(x: float) -> str:
    """Thai baht only. For anything that may be another currency use `money`."""
    return "฿" + f"{x:,.2f}".removesuffix(".00")


def money(amount: float, currency: str | None = "THB") -> str:
    """Format in the receipt's own currency: ฿ for THB (also when missing), $ for USD, otherwise
    "1,234.50 CODE". Never prefixes ฿ to another currency."""
    cur = (currency or "THB").strip().upper() or "THB"
    if cur == "THB":
        return baht(amount)
    if cur == "USD":
        return ("-$" if amount < 0 else "$") + f"{abs(amount):,.2f}"
    return f"{amount:,.2f} {cur}"


def fmt_totals(by: dict[str, float]) -> str:
    """Per-currency totals, e.g. "฿184,885.54, $131.99". Currencies are never added together."""
    items = sorted(by.items(), key=lambda kv: (kv[0] != "THB", kv[0]))
    return ", ".join(money(v, c) for c, v in items) or baht(0)


class LedgerTools:
    """Tools the model may call. All arithmetic happens here (SQL). `guild_id` (the tenant) and
    `forced_submitter_id` are injected into every query and are NOT part of any tool schema, so the
    model cannot widen the scope."""

    def __init__(
        self,
        store: Store,
        guild_id: int,
        forced_submitter_id: int | None = None,
        actor: str = "",
    ):
        self.store = store
        self.guild_id = guild_id
        self.forced = forced_submitter_id
        self.actor = actor  # who asked (recorded as `voided_by`)

    @property
    def is_exec(self) -> bool:
        """Executives are the unscoped askers; only they may void or restore receipts."""
        return self.forced is None

    def _fmt_summary(self, s: dict) -> dict:
        s["total_fmt"] = fmt_totals(s["totals_by_currency"])
        for g in s["groups"]:
            g["total_fmt"] = fmt_totals(g["totals_by_currency"])
        return s

    def spending_summary(self, start_date: str, end_date: str, group_by: str = "none") -> dict:
        return self._fmt_summary(
            self.store.summarize(
                self.guild_id, start_date, end_date, group_by, submitter_id=self.forced
            )
        )

    def list_receipts(
        self,
        start_date: str,
        end_date: str,
        category: str | None = None,
        vendor_contains: str | None = None,
        submitter: str | None = None,
        limit: int = 20,
        include_void: bool = False,
    ) -> dict:
        rows = self.store.list_receipts(
            self.guild_id,
            start_date,
            end_date,
            category,
            vendor_contains,
            submitter,
            limit,
            submitter_id=self.forced,
            include_void=include_void,
        )
        return {"count": len(rows), "receipts": [self._r(r) for r in rows]}

    def compare_periods(
        self, a_start: str, a_end: str, b_start: str, b_end: str, group_by: str = "none"
    ) -> dict:
        a = self.store.summarize(self.guild_id, a_start, a_end, group_by, submitter_id=self.forced)
        b = self.store.summarize(self.guild_id, b_start, b_end, group_by, submitter_id=self.forced)
        am = {g["key"]: g["totals_by_currency"] for g in a["groups"]}
        bm = {g["key"]: g["totals_by_currency"] for g in b["groups"]}
        rows = []
        for k in sorted(
            set(am) | set(bm),
            key=lambda k: -sum(am.get(k, {}).values()) - sum(bm.get(k, {}).values()),
        ):
            rows.append(self._delta(k, am.get(k, {}), bm.get(k, {})))
        return {
            "note": "amounts are compared per currency; never add different currencies",
            "period_a": {
                "start": a_start,
                "end": a_end,
                "total_fmt": fmt_totals(a["totals_by_currency"]),
                "total": a["total"],
                "totals_by_currency": a["totals_by_currency"],
            },
            "period_b": {
                "start": b_start,
                "end": b_end,
                "total_fmt": fmt_totals(b["totals_by_currency"]),
                "total": b["total"],
                "totals_by_currency": b["totals_by_currency"],
            },
            "overall": self._delta("all", a["totals_by_currency"], b["totals_by_currency"]),
            "groups": rows,
        }

    @staticmethod
    def _delta(key, a: dict, b: dict):
        """Per-currency difference a - b. With a single currency the flat fields a/b/diff/... are
        also present; with several they are None and `by_currency` carries each one."""
        by = {}
        for cur in sorted(set(a) | set(b), key=lambda c: (c != "THB", c)):
            av, bv = a.get(cur, 0.0), b.get(cur, 0.0)
            diff = round(av - bv, 2)
            by[cur] = {
                "a": av,
                "b": bv,
                "diff": diff,
                "diff_fmt": ("+" if diff > 0 else "-" if diff < 0 else "") + money(abs(diff), cur),
                "pct_change_vs_b": round(diff / bv * 100, 1) if bv else None,
            }
        out = {"key": key, "by_currency": by}
        flat = next(iter(by.values())) if len(by) == 1 else None
        for f in ("a", "b", "diff", "diff_fmt", "pct_change_vs_b"):
            out[f] = flat[f] if flat else None
        return out

    def flagged_receipts(self) -> dict:
        rows = self.store.flagged(self.guild_id, submitter_id=self.forced)
        return {"count": len(rows), "receipts": [self._r(r) for r in rows]}

    @staticmethod
    def _money(amount: float | None, currency: str = "THB") -> str | None:
        if amount is None:
            return None
        return money(amount, currency)

    def _sum_fmt(self, rows) -> tuple[dict, str]:
        by: dict[str, float] = {}
        for r in rows:
            by[r.currency or "THB"] = by.get(r.currency or "THB", 0.0) + (r.total or 0.0)
        by = {c: round(v, 2) for c, v in by.items()}
        return by, fmt_totals(by)

    def _dup_entry(self, r) -> dict:
        out = {
            "id": r.id,
            "vendor": r.vendor,
            "total": r.total,
            "total_fmt": self._money(r.total, r.currency),
            "date": r.effective_date,
            "submitter": r.submitter_name,
            "confidence": r.confidence,
            "file_type": "pdf"
            if (r.original_uri or "").lower().endswith(".pdf")
            else "image/other",
        }
        if r.guild_id and r.channel_id and r.message_id:
            out["source"] = {
                "channel_id": r.channel_id,
                "message_id": r.message_id,
                "jump_url": f"https://discord.com/channels/{r.guild_id}/{r.channel_id}/{r.message_id}",
            }
        return out

    def find_duplicates(self, start_date: str | None = None, end_date: str | None = None) -> dict:
        """Groups of ACTIVE receipts that look like the same bill (read-only). Scoped users only
        get duplicates among their own receipts."""
        start = start_date or end_date and "0000-01-01"
        end = end_date or start_date and "9999-12-31"
        groups = self.store.duplicate_groups(
            self.guild_id,
            start if start and end else None,
            end if start and end else None,
            submitter_id=self.forced,
        )
        out = []
        for g in groups:
            keep = pick_keep(g)
            extras = [r for r in g if r is not keep]
            _, fmt = self._sum_fmt(extras)
            out.append(
                {
                    "keep": self._dup_entry(keep),
                    "duplicates": [self._dup_entry(r) for r in extras],
                    "deduct_fmt": fmt,
                }
            )
        by, fmt = self._sum_fmt(duplicate_extras(groups))
        return {
            "group_count": len(out),
            "groups": out,
            "total_deduct_by_currency": by,
            "total_deduct_fmt": fmt,
            "can_void": self.is_exec,
        }

    def void_receipts(self, receipt_ids: list[int], reason: str) -> dict:
        """Void (soft delete, reversible) ACTIVE receipts of this guild. Executives only."""
        if not self.is_exec:
            return {"error": "not_allowed"}
        if not (reason or "").strip():
            return {"error": "reason is required"}
        ids = list(dict.fromkeys(int(i) for i in receipt_ids or []))
        ignored, todo = [], {}
        for rid in ids:
            r = self.store.get(rid)
            if r is None or r.guild_id != self.guild_id:
                ignored.append({"id": rid, "reason": "not found in this server"})
            elif not is_active(r):
                ignored.append({"id": rid, "reason": "already void"})
            else:
                todo[rid] = r
        refused = []
        for g in self.store.duplicate_groups(self.guild_id):
            gid = {r.id for r in g}
            if gid <= set(todo):  # would leave the group with no active receipt
                for rid in sorted(gid):
                    refused.append(
                        {"id": rid, "reason": "would void every copy of the bill; keep one"}
                    )
                    todo.pop(rid)
        done = [
            self.store.set_status(rid, "void", reason=reason.strip(), by=self.actor or "executive")
            for rid in todo
        ]
        by, fmt = self._sum_fmt(done)
        return {
            "voided": [self._dup_entry(r) for r in done],
            "ignored": ignored,
            "refused": refused,
            "deducted_by_currency": by,
            "total_delta_fmt": "-" + fmt if done else baht(0),
            "undo": "restore_receipts",
        }

    def restore_receipts(self, receipt_ids: list[int]) -> dict:
        """Undo a void. Executives only."""
        if not self.is_exec:
            return {"error": "not_allowed"}
        ignored, done = [], []
        for rid in dict.fromkeys(int(i) for i in receipt_ids or []):
            r = self.store.get(rid)
            if r is None or r.guild_id != self.guild_id:
                ignored.append({"id": rid, "reason": "not found in this server"})
            elif is_active(r):
                ignored.append({"id": rid, "reason": "not void"})
            else:
                done.append(self.store.set_status(rid, "active"))
        by, fmt = self._sum_fmt(done)
        return {
            "restored": [self._dup_entry(r) for r in done],
            "ignored": ignored,
            "added_by_currency": by,
            "total_delta_fmt": "+" + fmt if done else baht(0),
        }

    @staticmethod
    def _r(r) -> dict:
        extra = (
            {"status": "void", "void_reason": r.void_reason, "voided_by": r.voided_by}
            if not is_active(r)
            else {}
        )
        return {
            **extra,
            "id": r.id,
            "date": r.effective_date,
            "vendor": r.vendor,
            "total": r.total,
            "currency": r.currency or "THB",
            "total_fmt": money(r.total, r.currency) if r.total is not None else None,
            "category": r.category,
            "submitter": r.submitter_name,
            "confidence": r.confidence,
            "possible_duplicate_of": r.possible_duplicate_of,
            "notes": r.notes,
        }

    def call(self, name: str, args: dict) -> dict:
        allowed = {
            "spending_summary",
            "list_receipts",
            "compare_periods",
            "flagged_receipts",
            "find_duplicates",
            "void_receipts",
            "restore_receipts",
        }
        if name not in allowed:
            return {"error": f"unknown tool {name}"}
        fn = getattr(self, name)
        # only pass parameters the function declares; drops any attempt to inject submitter_id etc.
        params = fn.__code__.co_varnames[1 : fn.__code__.co_argcount]
        clean = {k: v for k, v in (args or {}).items() if k in params and v is not None}
        try:
            return fn(**clean)
        except (TypeError, ValueError) as e:
            return {"error": str(e)}


MAX_CHAT_DAYS = 31  # longest date range one memory tool call may cover
MAX_DAY_ROWS = 60  # summaries returned per chat_days call
TAIL_PER_CHANNEL = 50
TAIL_TOTAL = 150
MAX_MESSAGE_LIMIT = 100
SNIPPET_CHARS = 500
DAYS_CHARS = 6000  # cap of one chat_days result (JSON characters)
DAYS_TAIL_CHARS = 2500  # share of it for the raw tail newer than today's summary
SEARCH_CHARS = 4000  # cap of one chat_search result
SUMMARY_WEIGHT = 3  # a keyword hit in a summary counts this many raw-message hits
SEARCH_MSGS_PER_GROUP = 3
MAX_SEARCH_DAYS = 400
MAX_ROWS_SEARCH = 5000
# Hybrid search (keyword + meaning) over day summaries; raw messages only where it pays off.
RAW_RECENT_DAYS = 3  # raw messages of the last days are always scanned (not summarised yet)
RAW_GROUPS = 5  # older top hits whose raw messages are read for quotes and links
RAW_GROUP_FETCH = 600  # messages read per such channel-day
MIN_SIMILARITY = 0.55  # cosine a summary item needs to count as a meaning match (provisional)
RRF_K = 60  # reciprocal rank fusion constant
QUERY_CACHE_SIZE = 512  # recent query vectors kept (the model often repeats a search)
QUERY_EMBED_TIMEOUT_S = 2.0  # longest wait for the query vector before keyword-only search
RAW_NOTE = (
    f"raw messages older than {RAW_RECENT_DAYS} days are searched through summaries only; for an "
    "exact code, number or phrase use chat_messages with a keyword and dates"
)
_query_cache: OrderedDict[tuple[str, str], object] = OrderedDict()
_query_lock = threading.Lock()
_embed_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="salid-embed")


def _query_vector(embedder, text: str):
    """The query's vector, cached per (model, text). Raises when embedding fails."""
    key = (embedder.model_id, text)
    with _query_lock:
        if key in _query_cache:
            _query_cache.move_to_end(key)
            return _query_cache[key]
    vec = embedder.embed([text], "query")[0]
    with _query_lock:
        _query_cache[key] = vec
        while len(_query_cache) > QUERY_CACHE_SIZE:
            _query_cache.popitem(last=False)
    return vec


class MemoryTools:
    """Organisational-memory tools. Bound in code to one guild and to `allowed` (channel id ->
    name), computed by the bot from Discord permissions; neither is part of a tool schema, and a
    channel name the model passes is only matched against `allowed`, so it can never widen the
    scope. Reads go through MemoryStore, which also refuses anything outside `allowed`."""

    def __init__(
        self,
        store: MemoryStore,
        guild_id: int,
        allowed: dict[int, str],
        settings,
        retention_days: int | None = None,
        opener: Callable[[int | None, int, int], dict] | None = None,
        visibility: Visibility | None = None,
        embedder=None,
        retention_of: Callable[[], int] | None = None,
    ):
        self.opener = opener  # (channel id | None, message id, file index) -> {"text": ...}
        self.store, self.guild_id, self.allowed, self.s = store, guild_id, dict(allowed), settings
        self._retention_days = retention_days or settings.memory_retention_days
        self._retention_of = retention_of  # read at call time: a shortened setting applies at once
        # With `visibility` the role hierarchy and disclosure grants apply (salid.memory_access);
        # `hidden` collects matching records the asker may not see (never shown, only counted, so
        # Salid can offer a disclosure request).
        self.vis = visibility
        self.hidden: list[MemoryMessage] = []
        self.hidden_in_dm = 0  # of those: would be visible if asked in a DM
        self.embedder = embedder or NullEmbedder()

    def open_file(self, message_link_or_id: str, file_index: int = 0) -> dict:
        """Re-read an attachment (document, audio, video, image) of a remembered message. The
        channel check happens in the bot against `allowed`; nothing outside it is returned."""
        ref = parse_message_ref(message_link_or_id)
        if ref is None:
            return {"error": "give the message link (jump url) or message id"}
        if self.opener is None:
            return {"error": "opening files is not available here"}
        cid, mid = ref
        return self.opener(cid, mid, int(file_index))

    @property
    def retention_days(self) -> int:
        """The guild's CURRENT retention (not the value at construction)."""
        if self._retention_of is not None:
            try:
                return max(int(self._retention_of()), 1)
            except Exception:  # fail closed on the construction value
                pass
        return max(self._retention_days, 1)

    def floor(self) -> date:
        """Oldest day any tool may read: what the daily purge keeps (see `cutoff_date`)."""
        return self.s.now().date() - timedelta(days=self.retention_days)

    def _range(self, start_date: str, end_date: str) -> tuple[str, str]:
        a, b = date.fromisoformat(start_date), date.fromisoformat(end_date)
        if b < a:
            raise ValueError("end_date is before start_date")
        if (b - a).days + 1 > MAX_CHAT_DAYS:
            raise ValueError(f"range too long: at most {MAX_CHAT_DAYS} days per call")
        return max(a, self.floor()).isoformat(), b.isoformat()  # past retention: never read

    def _shown(self) -> dict[int, str]:
        """Channels whose day summaries may be read: the allowed ones plus, with a hierarchy,
        the DM briefings of people the asker outranks (single-author, never private notes)."""
        return {**self.allowed, **(self.vis.dm_names if self.vis else {})}

    def _name(self, channel_id: int) -> str:
        shown = {**(self.vis.read_channels if self.vis else {}), **self._shown()}
        if is_dm_scope(channel_id) and channel_id not in shown:
            return "DM"
        return shown.get(channel_id, str(channel_id))

    def _read_ids(self) -> set[int]:
        """Channels to read from the store: with a hierarchy, every channel (hidden records are
        counted, then filtered by `_visible`)."""
        return set(self.allowed) | (set(self.vis.read_channels) if self.vis else set())  # + hidden

    def _visible(self, m: MemoryMessage) -> bool:
        return self.vis is None or message_visible(m, self.vis.scope)

    def _person_ids(self, person: str | None) -> tuple[set[int], str]:
        """Member ids matching a name or mention, and the lower-cased text for author names."""
        text = (person or "").strip().lstrip("@").casefold()
        if not text:
            return set(), ""
        ids = {int(x) for x in re.findall(r"<@!?(\d{1,25})>", person or "")}
        if text.isdigit():
            ids.add(int(text))
        if self.vis:
            ids |= {i for i, n in self.vis.people.items() if text in n.casefold()}
        return ids, text

    def _by_person(self, m: MemoryMessage, ids: set[int], text: str) -> bool:
        return not text or m.author_id in ids or text in m.author_name.casefold()

    def _local_time(self, text: str | None) -> datetime | None:
        """A local (guild time zone) ISO date-time from the model -> aware datetime."""
        if not text or not str(text).strip():
            return None
        d = datetime.fromisoformat(str(text).strip())
        return d.replace(tzinfo=self.s.zone) if d.tzinfo is None else d

    def _hide(self, m: MemoryMessage) -> None:
        if self.vis is None or not hidden_but_askable(m, self.vis.scope):
            return
        if all(h.message_id != m.message_id for h in self.hidden):
            self.hidden.append(m)
            if m.author_id in self.vis.scope.dm_authors:
                self.hidden_in_dm += 1

    def _hidden_note(self) -> str | None:
        if not self.hidden:
            return None
        n = len(self.hidden)
        hint = (
            " Part of it may be shareable if the person asks in a DM with Salid."
            if self.hidden_in_dm
            else ""
        )
        return (
            f"{n} other matching record(s) (private notes or private channels) are left out of "
            "these results. This is only a footnote: everything returned here IS shareable, so "
            "answer the question fully from it first (never refuse or say you cannot summarise "
            "because of this). Then add at most one short sentence that some private notes exist "
            "and offer to ask the owner with request_disclosure; never guess, describe or call them "
            "meeting records."
            f"{hint}"
        )

    def _msg(self, m: MemoryMessage) -> dict:
        return {
            "time": f"{m.created_at.astimezone(self.s.zone):%Y-%m-%d %H:%M}",
            "channel": self._name(m.channel_id),
            "author": "Salid" if m.kind == "salid" else m.author_name,
            "kind": m.kind,
            "text": m.text[:SNIPPET_CHARS],
            "link": m.jump_url,
        }

    @staticmethod
    def _size(item) -> int:
        return len(json.dumps(item, ensure_ascii=False))

    def _fit(self, items: list, budget: int) -> tuple[list, bool]:
        """Leading items (already in priority order) whose JSON fits `budget` characters."""
        out, used = [], 0
        for it in items:
            used += self._size(it)
            if used > budget and out:
                return out, True
            out.append(it)
        return out, False

    _summary_lines = staticmethod(summary_items)  # searchable lines of a summary (also: wiki)

    def chat_days(self, start_date: str, end_date: str, channel_name: str | None = None) -> dict:
        start, end = self._range(start_date, end_date)
        shown = self._shown()
        if channel_name:
            want = channel_name.strip().lstrip("#").casefold()
            shown = {i: n for i, n in shown.items() if n.casefold() == want}
            if not shown:
                return {"count": 0, "days": [], "note": f"no channel named {channel_name!r}"}
        sums = [
            x for x in self.store.summaries(self.guild_id, shown, start, end) if x.message_count > 0
        ]
        # Messages newer than the (living) summary, so nothing between updates is missed. One
        # read per day, split by channel here.
        today = self.s.now().date()
        tail: list[dict] = []
        covered = {(x.channel_id, x.local_date): x for x in sums}
        for day in (today - timedelta(days=1), today):
            d = day.isoformat()
            if not start <= d <= end:
                continue
            by_channel: dict[int, list[MemoryMessage]] = {}
            for m in self.store.messages(
                self.guild_id, set(shown), d, d, limit=MAX_ROWS, newest_first=False
            ):
                by_channel.setdefault(m.channel_id, []).append(m)
            for cid in sorted(by_channel):
                x = covered.get((cid, d))
                if x is not None and x.final:
                    continue
                rows = [
                    m
                    for m in by_channel[cid]
                    if (x is None or x.cursor is None or m.key > x.cursor) and self._visible(m)
                ]
                tail += [self._msg(m) for m in rows[:TAIL_PER_CHANNEL]]
        tail = tail[-TAIL_TOTAL:]
        # Newest days first; the output is capped, so old days drop out first.
        all_days = [
            {
                "date": x.local_date,
                "channel": self._name(x.channel_id),
                "messages": x.message_count,
                "final": x.final,
                "summary": render_summary(x.summary),
            }
            for x in sorted(sums, key=lambda x: (x.local_date, x.channel_id), reverse=True)
        ][:MAX_DAY_ROWS]
        tail_fit, tail_cut = self._fit(list(reversed(tail)), DAYS_TAIL_CHARS)
        days, cut = self._fit(all_days, DAYS_CHARS - sum(self._size(m) for m in tail_fit))
        days.sort(key=lambda d: (d["date"], d["channel"]))
        out: dict = {"count": len(days), "days": days}
        if tail_fit:
            out["newer_than_summary"] = list(reversed(tail_fit))
        if cut or tail_cut or len(sums) > MAX_DAY_ROWS:
            out["note"] = (
                "output truncated (most recent days kept): narrow the date range or use chat_search"
            )
        elif not days and not tail_fit:
            out["note"] = "nothing recorded in the channels you can see for this period"
        return out

    def chat_search(
        self,
        keywords: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 20,
        person: str | None = None,
        query: str | None = None,
        semantic: bool = True,
    ) -> dict:
        """Hybrid search: keyword hits and meaning (embedding similarity) over the day summaries,
        fused by reciprocal rank; raw messages of the last RAW_RECENT_DAYS days are scanned for
        keywords, and the top older hits get their matching raw messages (quotes and links).
        `semantic=False` (join-mode hints) makes no embedding call."""
        started = time.monotonic()
        kws = [k.strip().casefold() for k in (keywords or []) if isinstance(k, str) and k.strip()]
        kws = list(dict.fromkeys(kws))[:12]
        query = (query or "").strip()
        if not kws and query:  # words of the question, so keyword search works without meaning
            kws = [k.casefold() for k in extract_keywords(query)]
        use_meaning = semantic and bool(self.embedder.model_id)
        if not kws and not (query and use_meaning):
            return {"error": "give at least one keyword or a query"}
        today = self.s.now().date()
        end = date.fromisoformat(end_date) if end_date else today
        start = (
            date.fromisoformat(start_date)
            if start_date
            else end - timedelta(days=max(self.retention_days, 1) - 1)
        )
        if end < start:
            raise ValueError("end_date is before start_date")
        start = max(start, end - timedelta(days=MAX_SEARCH_DAYS - 1), self.floor())
        a, b = start.isoformat(), end.isoformat()
        pids, ptext = self._person_ids(person)
        raw_from = (today - timedelta(days=RAW_RECENT_DAYS - 1)).isoformat()

        pending = None
        if use_meaning:  # embed the query while the summaries load
            pending = _embed_pool.submit(_query_vector, self.embedder, query or " ".join(kws))
        t_index = time.monotonic()
        entries = entries_for(self.store, self.guild_id, self._shown(), a, b, self.embedder)
        ms_index = (time.monotonic() - t_index) * 1000
        qv, ms_embed = None, 0.0
        if pending is not None:
            try:
                qv = pending.result(timeout=QUERY_EMBED_TIMEOUT_S)
            except Exception:
                logger.warning("query embedding failed; keyword search only", exc_info=True)
            ms_embed = (time.monotonic() - t_index) * 1000

        def hits(text: str) -> int:
            low = text.casefold()
            return sum(1 for k in kws if k in low)

        groups: dict[tuple[str, int], dict] = {}

        def group(day: str, cid: int) -> dict:
            return groups.setdefault(
                (day, cid),
                {"score": 0, "summary": [], "messages": [], "n_msgs": 0, "sim": 0.0, "near": []},
            )

        for e in entries:
            for line in e.items:
                h = hits(line)
                if h and (not ptext or ptext in line.casefold()):
                    g = group(e.local_date, e.channel_id)
                    g["score"] += h * SUMMARY_WEIGHT
                    g["summary"].append(line[:300])
            if qv is not None and e.vectors:
                sims = [
                    (cosine(qv, v), line)
                    for v, line in zip(e.vectors, e.items, strict=False)
                    if len(v) == len(qv)  # vectors of another model/dim are ignored
                    and (not ptext or ptext in line.casefold())
                ]
                best = max(sims, default=(0.0, ""))
                if best[0] >= MIN_SIMILARITY:
                    g = group(e.local_date, e.channel_id)
                    g["sim"] = best[0]
                    g["near"] = [line[:300] for _, line in sorted(sims, reverse=True)[:2]]

        def scan(m: MemoryMessage) -> None:
            """Count a keyword-matching raw message (role hierarchy and person filter apply)."""
            h = hits(f"{m.text} {m.author_name}")
            if not h or not self._by_person(m, pids, ptext):
                return
            if not self._visible(m):
                self._hide(m)
                return
            g = group(m.local_date, m.channel_id)
            g["score"] += h
            g["n_msgs"] += 1
            if len(g["messages"]) < SEARCH_MSGS_PER_GROUP:
                g["messages"].append(self._msg(m))

        raw_queries = 0
        raw_to = min(b, today.isoformat())  # no reads for days that have not happened yet
        if kws and max(a, raw_from) <= raw_to:  # recent raw messages (not summarised yet)
            raw_queries += 1
            rows = self.store.messages(
                self.guild_id, self._read_ids(), max(a, raw_from), raw_to,
                limit=MAX_ROWS_SEARCH, newest_first=True,
            )  # fmt: skip
            for m in rows:  # newest first
                scan(m)

        rankings = (
            sorted((k for k, g in groups.items() if g["score"] > 0),
                   key=lambda k: (groups[k]["score"], k[0]), reverse=True),
            sorted((k for k, g in groups.items() if g["sim"] >= MIN_SIMILARITY),
                   key=lambda k: (groups[k]["sim"], k[0]), reverse=True),
        )  # fmt: skip
        fused: dict[tuple[str, int], float] = {}
        for ranking in rankings:
            for rank, key in enumerate(ranking, start=1):
                fused[key] = fused.get(key, 0.0) + 1 / (RRF_K + rank)
        order = sorted(fused, key=lambda k: (fused[k], k[0]), reverse=True)
        order = order[: max(1, min(int(limit), 50))]

        if kws:  # quotes and links for the top older hits
            readable = self._read_ids()
            for day, cid in [k for k in order if k[0] < raw_from and k[1] in readable][:RAW_GROUPS]:
                raw_queries += 1
                g = groups[(day, cid)]
                for m in reversed(
                    self.store.messages_after(self.guild_id, cid, day, None, RAW_GROUP_FETCH)
                ):
                    h = hits(f"{m.text} {m.author_name}")
                    if not h or not self._by_person(m, pids, ptext):
                        continue
                    if not self._visible(m):
                        self._hide(m)
                        continue
                    g["n_msgs"] += 1
                    if len(g["messages"]) < SEARCH_MSGS_PER_GROUP:
                        g["messages"].append(self._msg(m))

        def match(g: dict) -> str:
            kw, mean = g["score"] > 0, g["sim"] >= MIN_SIMILARITY
            return "both" if kw and mean else "keyword" if kw else "meaning"

        items = [
            {
                "date": day,
                "channel": self._name(cid),
                "score": g["score"],
                "match": match(g),
                "summary_hits": (g["summary"] or g["near"])[:6],
                "matching_messages": g["n_msgs"],
                "messages": g["messages"],
            }
            for day, cid in order
            for g in (groups[(day, cid)],)
        ]
        fit, cut = self._fit(items, SEARCH_CHARS)
        out: dict = {"keywords": kws, "range": [a, b], "count": len(fit), "hits": fit}
        notes = []
        if cut:
            notes.append("output truncated: more hits exist, use narrower keywords or dates")
        if not fit:
            notes.append("no matches in the channels you can see")
        if a < raw_from:
            notes.append(RAW_NOTE)
        if notes:
            out["note"] = " | ".join(notes)
        if self._hidden_note():
            out["withheld"] = self._hidden_note()
        kinds = [h["match"] for h in fit]
        logger.info(
            "memory search guild=%s ms=%d ms_embed=%d ms_index=%d raw_queries=%d hits=%d "
            "keyword=%d meaning=%d both=%d",
            self.guild_id, (time.monotonic() - started) * 1000, ms_embed, ms_index, raw_queries,
            len(fit), kinds.count("keyword"), kinds.count("meaning"), kinds.count("both"),
        )  # fmt: skip
        return out

    def chat_messages(
        self,
        start_date: str,
        end_date: str,
        channel_name: str | None = None,
        keyword: str | None = None,
        limit: int = 50,
        person: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> dict:
        start, end = self._range(start_date, end_date)
        pids, ptext = self._person_ids(person)
        t0, t1 = self._local_time(since), self._local_time(until)
        channel_id = None
        if channel_name:
            want = channel_name.strip().lstrip("#").casefold()
            channel_id = next((i for i, n in self.allowed.items() if n.casefold() == want), None)
            if channel_id is None:
                return {
                    "count": 0,
                    "messages": [],
                    "note": f"no channel named {channel_name!r} available",
                }
        limit = max(1, min(int(limit), MAX_MESSAGE_LIMIT))
        if self.vis is None and not ptext and not (t0 or t1):
            rows = self.store.messages(
                self.guild_id,
                self.allowed,
                start,
                end,
                channel_id=channel_id,
                keyword=keyword,
                limit=limit,
                newest_first=True,
            )
            return {"count": len(rows), "messages": [self._msg(m) for m in rows]}
        rows = self.store.messages(
            self.guild_id,
            self._read_ids() if channel_id is None else {channel_id},
            start,
            end,
            channel_id=channel_id,
            keyword=keyword,
            limit=MAX_ROWS_SEARCH,
            newest_first=True,
        )
        shown = []
        for m in rows:
            if not self._by_person(m, pids, ptext):
                continue
            if (t0 and m.created_at < t0) or (t1 and m.created_at >= t1):
                continue
            if self._visible(m):
                shown.append(m)
            elif keyword or ptext:
                self._hide(m)
        out: dict = {"count": len(shown[:limit]), "messages": [self._msg(m) for m in shown[:limit]]}
        if len(shown) > limit:
            out["note"] = (
                f"{len(shown) - limit} older message(s) in this range were not returned: narrow "
                "with since/until (local date-time) or channel_name to read the whole period"
            )
        if self._hidden_note():
            out["withheld"] = self._hidden_note()
        return out

    def call(self, name: str, args: dict) -> dict:
        if name not in {"chat_days", "chat_messages", "chat_search"}:
            return {"error": f"unknown tool {name}"}
        fn = getattr(self, name)
        params = fn.__code__.co_varnames[1 : fn.__code__.co_argcount]
        clean = {k: v for k, v in (args or {}).items() if k in params and v is not None}
        clean.pop("semantic", None)  # internal switch, never the model's choice
        try:
            return fn(**clean)
        except (TypeError, ValueError) as e:
            return {"error": str(e)}


# ---------------------------------------------------------------- files sent to Salid
MAX_FILES = 4
MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024
MAX_TEXT_FILE_BYTES = 200 * 1024
NO_TEXT_TURN = "(ผู้ใช้ส่งไฟล์มาโดยไม่มีข้อความ)"
_IMAGE_MIMES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "heic": "image/heic",
    "heif": "image/heif",
}
_TEXT_EXTS = {"txt", "md", "csv", "tsv", "json", "log", "xml", "yaml", "yml", "html", "htm"}
_RECEIPT_MIMES = {"image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"}


@dataclass
class FileInput:
    """One attachment of the message being answered. `index` is its position in the message's
    attachments (the number the model passes to record_receipt). `size` defaults to len(data);
    the bot may pass a size without data to plan what to read. `skip` = why it is not used."""

    index: int
    filename: str
    mime: str | None
    data: bytes = b""
    size: int = 0
    skip: str = ""
    label: str = ""  # replaces the default "[ไฟล์ #N: name]" label (transcripts of audio/video)

    def __post_init__(self):
        self.size = self.size or len(self.data)


def file_mime(filename: str, content_type: str | None = None) -> str | None:
    """The MIME type Salid can read (image, PDF, plain text), or None for anything else."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    ct = (content_type or "").split(";")[0].strip().lower()
    if ext in _IMAGE_MIMES:
        return _IMAGE_MIMES[ext]
    if ext == "pdf":
        return "application/pdf"
    if ext in _TEXT_EXTS:
        return "text/plain"
    if ct in set(_IMAGE_MIMES.values()):
        return ct
    if ct == "application/pdf":
        return ct
    if ct.startswith("text/"):
        return "text/plain"
    return None


def select_files(files: list[FileInput]) -> tuple[list[FileInput], list[FileInput]]:
    """(usable, skipped): applies the count, per-file and total size limits in order. Skipped files
    carry the reason in `skip`."""
    ok: list[FileInput] = []
    skipped: list[FileInput] = []
    total = 0
    for f in files:
        reason = f.skip
        if not reason and not f.mime:
            reason = "unsupported file type"
        elif not reason and f.mime == "text/plain" and f.size > MAX_TEXT_FILE_BYTES:
            reason = f"text file over {MAX_TEXT_FILE_BYTES // 1024} KB"
        elif not reason and f.size > MAX_FILE_BYTES:
            reason = f"over {MAX_FILE_BYTES // 1024 // 1024} MB"
        elif not reason and len(ok) >= MAX_FILES:
            reason = f"more than {MAX_FILES} files"
        elif not reason and total + f.size > MAX_TOTAL_BYTES:
            reason = f"total size over {MAX_TOTAL_BYTES // 1024 // 1024} MB"
        if reason:
            f.skip = reason
            skipped.append(f)
        else:
            ok.append(f)
            total += f.size
    return ok, skipped


def _gif_first_frame(data: bytes) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        im.seek(0)
        out = io.BytesIO()
        im.convert("RGBA").save(out, format="PNG")
        return out.getvalue()


def file_parts(files: list[FileInput]) -> tuple[list[types.Part], list[FileInput]]:
    """Model input for the usable files: a label part (with the attachment number) and then the
    file as inline bytes (text files as text). Returns (parts, files that could not be read)."""
    parts: list[types.Part] = []
    bad: list[FileInput] = []
    for f in files:
        label = f.label or f"[ไฟล์ #{f.index}: {f.filename}]"
        try:
            if f.mime == "text/plain":
                body = f.data.decode("utf-8", errors="replace")
                parts.append(types.Part(text=f"{label}\n{body}"))
                continue
            data, mime = f.data, f.mime or ""
            if mime == "image/gif":
                data, mime = _gif_first_frame(data), "image/png"
            parts.append(types.Part(text=label))
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        except Exception:
            f.skip = "could not be read"
            bad.append(f)
    return parts, bad


def skipped_note(skipped: list[FileInput]) -> str:
    return "\n".join(f"[file skipped: {f.filename}, {f.skip}]" for f in skipped)


@dataclass
class FileActions:
    """What the file tools may do, bound by the bot to one message (guild, author, channel and
    message ids are closed over there, never model input). `record_receipt` None = the receipts
    feature is off; `start_deep_task` None = deep tasks are not available."""

    record_receipt: Callable[[FileInput, str], object] | None = None  # -> extract.IngestResult
    start_deep_task: Callable[[str], str] | None = None  # -> "started" or a reason


PERSONA = {
    "th": (
        "คุณคือ “สลิด” เพื่อนร่วมทีมและเลขาที่เก่งและใจดี อยู่ในแชต Discord ของทีม และเป็นผู้ช่วยผู้บริหารและผู้ประสานงานของทีม (ติดตามว่าใครทำอะไร ตามงาน รายงาน และรักษาความลับตามลำดับชั้น role โดยเสนอขออนุญาตเจ้าของข้อมูลเมื่อผู้ถามยังไม่มีสิทธิ์เห็น) พูดเหมือนคนจริงๆ ไม่ใช่หุ่นยนต์: "
        "สั้นและเป็นธรรมชาติเป็นค่าเริ่มต้น เป็นผู้หญิง เรียกตัวเองว่า “สลิด” (ห้ามใช้ ผม/ฉัน/ครับ) ใช้คำลงท้ายอย่าง ค่ะ นะคะ ได้ค่ะ ตามจังหวะที่คนไทยพูดกัน (ประโยคคำถามใช้ “คะ”) "
        "ปรับน้ำเสียงตามอีกฝ่าย (เขาเล่นมาก็เล่นด้วย เขาจริงจังก็จริงจัง) เรียกชื่อคนที่คุยด้วยบ้างเป็นครั้งคราว และจำเรื่องที่คุยกันมาก่อนหน้า\n"
    ),
    "en": (
        "You are “Salid” (สลิด), a capable, warm teammate and secretary in the team's Discord chat, and the team's executive assistant and coordinator: you track who is doing what, follow up on work, report, and keep confidential information according to the role hierarchy, offering to ask the owner when the asker may not see something. "
        "Talk like a real person, not a robot: short and natural by default, relaxed, polite, with a "
        "bit of warmth. Match the other person's energy (playful with playful, serious with serious), "
        "use their name now and then, and remember what was said earlier. ALWAYS reply in English, "
        "even if the user writes in Thai (keep Thai names, vendors and category names as they are).\n"
    ),
    "auto": (
        "คุณคือ “สลิด” (Salid) เพื่อนร่วมทีมและเลขาที่เก่งและใจดี อยู่ในแชต Discord ของทีม และเป็นผู้ช่วยผู้บริหารและผู้ประสานงานของทีม (ติดตามงานและรายงาน รักษาความลับตามลำดับชั้น / executive assistant and coordinator) พูดเหมือนคนจริงๆ ไม่ใช่หุ่นยนต์: "
        "สั้นและเป็นธรรมชาติเป็นค่าเริ่มต้น ปรับน้ำเสียงตามอีกฝ่าย เรียกชื่อคนที่คุยด้วยบ้าง และจำเรื่องที่คุยกันมา "
        "ตอบด้วยภาษาเดียวกับข้อความล่าสุดของผู้ใช้ (reply in the same language as the user's latest "
        "message): ถ้าพิมพ์ไทยตอบไทยธรรมชาติ เป็นผู้หญิง เรียกตัวเองว่า “สลิด” (ห้ามใช้ ผม/ฉัน/ครับ) ลงท้ายด้วยคำว่าค่ะ (ประโยคคำถามใช้ “คะ”) ถ้าพิมพ์อังกฤษตอบอังกฤษแบบเป็นกันเอง\n"
    ),
}

HUMAN_STYLE_PROMPT = (
    "สไตล์การตอบ: เหมือนเพื่อนร่วมงานที่เก่ง ไม่ใช่ระบบ. คุยเล่นหรือถามสั้นๆ ตอบ 1-2 ประโยค ไม่ใช้หัวข้อหรือบูลเล็ตในการคุยทั่วไป "
    "(ใช้รายการหรือโครงสร้างเฉพาะตอนเป็นรายการจริงๆ หรือรายงาน/สรุปที่ผู้ใช้ขอ). "
    "ห้ามใช้ประโยคแบบหุ่นยนต์ เช่น “ระบบได้รับคำสั่งแล้ว” “ในฐานะ AI” “ฉันสามารถช่วยคุณได้ในเรื่องต่อไปนี้” "
    "ห้ามแนะนำตัวซ้ำหรือไล่รายการสิ่งที่ทำได้ เว้นแต่ถูกถามตรงๆ. "
    "ถ้าไม่แน่ใจว่าอีกฝ่ายต้องการอะไร ให้ถามกลับสั้นๆ ข้อเดียว แทนที่จะโยนตัวเลือกยาวๆ. "
    "ทำเลยเมื่อสั่งชัดแล้ว แล้วบอกผลสั้นๆ. ถ้าเป็นรายงานหรือสรุป จะจัดโครงสร้างให้อ่านง่ายก็ได้. "
    "อีโมจิ 0-1 ตัวต่อข้อความ ไม่ต้องใส่ทุกข้อความ และห้ามใส่เป็นพรวน. "
    "ใช้ประวัติแชตเพื่อเข้าใจข้อความล่าสุดเท่านั้น ห้ามหยิบเรื่องเก่าที่ไม่เกี่ยวมาพูดหรือเสนอต่อยอด. "
    "ข้อเสนอต่อยอดมีได้อย่างมาก 1 ประโยคสั้นๆ และต้องเกี่ยวข้องกับสิ่งที่เพิ่งคุยโดยตรงเท่านั้น ห้ามปิดท้ายด้วยการเชียร์หรือเสนอบริการทุกครั้ง. "
    "ห้ามอ้างข้อจำกัดของ Discord หรือของระบบที่คุณไม่ได้ตรวจสอบจริง (เช่น “บอตส่ง DM ไม่ได้”: จริงๆ ส่งได้ด้วยเครื่องมือ send_dm). "
    "ถ้าไม่มีเครื่องมือสำหรับเรื่องที่ขอ ให้บอกตรงๆ ว่า “เรื่องนี้สลิดยังทำไม่ได้ค่ะ” แล้วเสนอทางที่ทำได้แทน ห้ามแต่งเหตุผลทางเทคนิคขึ้นมาเอง.\n"
)

# Settings by talking: the tools are bound in code to the guild, channel and author.
CHAT_TOOLS_PROMPT = (
    "\nตั้งค่าด้วยการพูดคุย (เครื่องมือ set_channel_mode, set_channel_language, set_memory_days, show_settings, "
    "hidden_channels, cancel_deep_task, send_dm, send_dm_many, post_message, deliver_file, set_my_notifications, show_my_notifications ถ้ามีให้ใช้): ทุกอย่างที่คำสั่ง / (slash) ทำได้ ผู้ใช้แค่บอกเป็นคำพูดก็พอ "
    "เช่น “ห้องนี้ไม่ต้องตอบนะ” (set_channel_mode off), “ช่วยพูดอังกฤษในห้องนี้” (set_channel_language en), "
    "“จำแค่ 30 วันพอ” (set_memory_days 30), “ตั้งค่าตอนนี้เป็นไง” (show_settings), “ทำไมสลิดไม่เห็นห้องบัญชี” (hidden_channels), "
    "“ยกเลิกงานที่ทำอยู่” (cancel_deep_task). เรียกเครื่องมือแล้วยืนยันสั้นๆ ประโยคเดียว (เช่น “ได้เลยค่ะ ห้องนี้สลิดเงียบไว้ให้นะคะ”). "
    "ถ้าเครื่องมือตอบ not_allowed ให้บอกเป็นกันเองว่าต้องให้แอดมินหรือผู้บริหารเป็นคนสั่ง. "
    "ห้ามบอกให้ผู้ใช้พิมพ์คำสั่ง / ถ้าทำให้เองได้ (คำสั่ง / เป็นแค่ทางลัดเสริม).\n"
    "ส่ง DM (เครื่องมือ send_dm ถ้ามี): สลิดส่งข้อความส่วนตัวหาสมาชิกในเซิร์ฟเวอร์ได้ เช่น “ส่งสรุปเข้า DM ผม” (to=me), "
    "“ส่ง PDF ให้ผมทาง DM” (to=me, attach=ชื่อไฟล์จาก make_report_pdf), “DM รายงานให้หัวหน้า”, “ฝากบอกพี่มิวทาง DM ว่า...”. "
    "ส่งให้คนอื่นจะมีบรรทัดบอกว่าใครฝาก จึงใส่เฉพาะสิ่งที่ผู้ถามบอกให้ฝากหรือข้อมูลที่ผู้รับเห็นได้อยู่แล้ว "
    "ห้ามใส่ตัวเลขบัญชี ความจำของทีม หรือรายงานที่เป็นสิทธิ์เฉพาะผู้ถามให้คนอื่น. "
    "ส่งสำเร็จให้ยืนยันในห้องสั้นๆ เช่น “ส่งเข้า DM ให้แล้วค่ะ”. ถ้าได้ dms_closed ให้บอกว่าอีกฝ่ายปิดรับ DM และบอกวิธีเปิดตาม guidance. "
    "ถ้า ambiguous ให้ถามกลับว่าหมายถึงคนไหนจากชื่อที่ได้.\n"
    "ประกาศไปห้องอื่น (เครื่องมือ post_message ถ้ามี): สลิดโพสต์ข้อความในห้องอื่นแทนผู้ถามได้ เช่น “ส่งข้อความไปห้องกลางว่าผมมีประชุมด่วน 9 โมง”, "
    "“ประกาศในห้องรวมว่า...”, “แจ้งทุกคนในห้องการตลาดว่า...”, “บอกใน #general ว่า...” (ห้องกลาง / ห้องรวม / general = ห้องทั่วไปของเซิร์ฟเวอร์). "
    "ห้ามบอกว่าส่งข้อความไปห้องอื่นไม่ได้ ห้ามเสนอแค่สร้างอีเวนต์หรือร่างอีเมลแทน. ใส่ text ตามถ้อยคำของผู้ถาม (สลิดเติม “📢 แจ้งจากคุณ…” ให้เอง) "
    "mention: none (ค่าเริ่มต้น) / everyone / here / role:ชื่อ role / ชื่อคนที่ให้แท็ก ใช้ everyone หรือ here เฉพาะเมื่อผู้ถามบอกให้แจ้งทุกคน. "
    "ถ้าให้ “แท็กคนที่เกี่ยวข้อง” ให้หาชื่อจากความจำทีม/งาน/ผู้เข้าประชุม ใส่ใน mention แล้วตั้ง tag_inferred=true (ต้องให้ผู้ถามยืนยันรายชื่อก่อนเสมอ; แก้ได้ เช่น “เพิ่ม A ด้วย” “เอา B ออก”; "
    "คนที่มองไม่เห็นห้องนั้นจะไม่ถูกแท็ก เสนอส่ง DM แทน). ถ้าได้ needs_confirmation ให้แสดงบรรทัด preview ให้ผู้ถามยืนยันในข้อความถัดไปก่อน แล้วเรียกซ้ำด้วยอาร์กิวเมนต์เดิมทุกตัว; "
    "ถ้า ambiguous ให้ถามว่าหมายถึงห้องไหน; ส่งสำเร็จตอบสั้นๆ พร้อมลิงก์ (link).\n"
    "DM หลายคน (เครื่องมือ send_dm_many ถ้ามี): “DM บอกทุกคนในทีมการตลาดว่า...”, “ส่งข้อความหาทุกคนในห้อง #sales ทีละคน” = send_dm_many (people / role / channel / everyone). "
    "ตั้งแต่ 4 คนขึ้นไปต้องให้ผู้ถามยืนยันรายชื่อก่อน (แสดง preview) เกิน 10 คนเฉพาะผู้บริหาร/ผู้จัดการ (ถ้าไม่ใช่ให้เสนอ post_message ในห้องแทน) สูงสุด 50 คน; "
    "ส่งเป็นเบื้องหลัง ให้บอกว่า “เริ่มส่งแล้ว” แล้วสลิดจะรายงานผลในแชตเอง (ห้ามบอกว่าส่งเสร็จแล้ว); ห้ามส่งเรื่องโอนเงิน/รหัสผ่านให้คนอื่น. "
    "send_dm ใช้กับ 1-3 คนเท่านั้น.\n"
    "เข้า/ออกห้องเสียง (join_voice, leave_voice ถ้ามี): “เข้ามาประชุมหน่อย”, “มาฟังด้วย”, “join us” = เรียก join_voice ทันที (ห้ามใช้ schedule_meeting; "
    "schedule_meeting ใช้เฉพาะเมื่อผู้ใช้ระบุเวลาในอนาคตชัดเจน); “ออกไปก่อน”, “ไปได้แล้ว” = leave_voice. "
    "ส่งสรุป/มติประชุมที่จบแล้วไปห้องอื่น (deliver_minutes ถ้ามี): “ส่งสรุปประชุมเมื่อกี้ไปห้อง X”.\n"
    "ส่งไฟล์/เอกสารให้คน (เครื่องมือ deliver_file ถ้ามี): สลิดส่งไฟล์หรือเอกสารต่อให้ได้ — โพสต์ในห้องอื่น ส่ง DM ให้คน หรือร่างอีเมลแนบไฟล์ "
    "ใช้เมื่อได้ยิน “ส่ง / เอาไปลง / ฝากให้ / forward / send this to …” เช่น ตอบกลับ (reply) ไฟล์ PDF แล้วพิมพ์ “นำเอกสารนี้ไปส่งที่ห้องเปเปอร์แล้วแท็ก jemm ด้วย”. "
    "ห้ามบอกว่าส่งไปห้องอื่นไม่ได้. เขียน note เองเป็นข้อความสั้นสุภาพ (ใช้ @[ชื่อ] เพื่อแท็ก ไม่ต้องแปะเนื้อหาเอกสาร) แล้วใส่ source=this "
    "(ไฟล์ในข้อความนี้หรือข้อความที่ตอบกลับ) หรือลิงก์ข้อความจาก chat_search หรือชื่อ PDF ที่สลิดเพิ่งทำ. "
    "destination ใส่ชื่อห้อง (ไทยหรืออังกฤษ ถ้าไม่เจอจะได้รายชื่อห้องให้เลือกชื่อที่ตรงแล้วเรียกใหม่) ชื่อคน หรือ email:ที่อยู่. "
    "ถ้าได้ needs_confirmation (เช่นคนในห้องปลายทางยังมองไม่เห็นห้องต้นทาง) ให้บอกคำเตือน ขอให้ผู้ใช้ยืนยัน แล้วเรียกซ้ำด้วยอาร์กิวเมนต์เดิมหลังผู้ใช้ตอบในข้อความถัดไปเท่านั้น. "
    "ถ้า ambiguous ให้ถามว่าหมายถึงห้อง/คนไหน; สำเร็จแล้วยืนยันสั้นๆ ว่าส่งไปที่ไหน.\n"
    "การแจ้งเตือนส่วนตัว (เครื่องมือ set_my_notifications / show_my_notifications ถ้ามี): ทุกคนปรับของตัวเองได้ด้วยการพูด เช่น “ไม่ต้องส่งสรุปเย็น”, "
    "“ส่งสรุปเช้า 7 โมง”, “ส่งเสาร์อาทิตย์ด้วย”, “แจ้งเอกสารสำคัญทุกฉบับ / เฉพาะที่สำคัญ / ปิด”, “เลิกเตือนเรื่องที่ค้างตอบ”. ส่งเฉพาะค่าที่เขาขอเปลี่ยน.\n"
    "ความชอบส่วนตัว (เครื่องมือ remember_my_preference / show_my_preferences / forget_my_preferences ถ้ามี): เมื่อผู้ใช้บอกความชอบแบบถาวรว่าอยากให้สลิดปฏิบัติกับเขาอย่างไร "
    "เช่น “เรียกผมว่าพี่ปัส” (nickname), “ตอบสั้นๆ นะ” (reply_length), “สรุปเป็นภาษาอังกฤษ” (language), “สนใจเรื่องการตลาด” (focus_add), “ผมดูแลบัญชี” (role_note), “ไม่ต้องใส่อีโมจิ” (style_add) "
    "ให้เรียกเครื่องมือแล้วยืนยันประโยคเดียว; อย่าบันทึกคำขอครั้งเดียว (“ตอบสั้นๆ ข้อนี้”) ถ้าไม่ชัดว่าถาวรหรือไม่ให้ถามสั้นๆ. แก้ได้เฉพาะของผู้พูดเอง."
)

MENTION_PROMPT = (
    "\nการแท็กคน: ถ้าต้องการเรียก ฝากข้อความ หรือบอกว่าใครรับผิดชอบอะไร ให้เขียนชื่อในรูป @[ชื่อที่แสดง] "
    "(เช่น @[มิว]) ระบบจะแปลงเป็นการแท็กจริงเอง ใช้ได้เฉพาะชื่อในรายชื่อ “คนที่แท็กได้” ด้านล่าง "
    "ห้ามเดาหรือแต่งชื่อ และห้ามเขียนรหัสผู้ใช้ ถ้าไม่มีชื่อที่ตรงให้ถามกลับสั้นๆ. "
    "ฝากบอกใคร (“ฝากบอกพี่มิวว่า...”, “tell Mew that...”): ตอบในห้องนี้โดยแท็กคนนั้นพร้อมข้อความ และบอกว่าใครฝาก "
    "(เช่น “@[มิว] ลูกค้าโทรมาค่ะ (คุณเอฝากบอก)”). ถ้าผู้ใช้บอกว่า “ส่วนตัว” / “DM” ให้เรียก send_dm แทน (to = ชื่อคนนั้น). "
    "“เรียกทุกคน” / “ทุกคนในทีม”: แท็กทุกคนที่อยู่ในรายชื่อ (ไม่เกินที่มี) ส่วน @everyone / @here ให้ใช้เฉพาะเมื่อผู้ถามเป็นแอดมินหรือผู้บริหาร "
    "และขอให้แจ้งทุกคนชัดเจน ไม่เช่นนั้นห้ามใช้."
)


LATENCY_PROMPT = (
    "\nความเร็ว: ตอบตรงๆ ทันทีเมื่อไฟล์แนบหรือข้อความมีคำตอบอยู่แล้ว (เช่น สรุปไฟล์ ขั้นตอนถัดไปจากเนื้อหาไฟล์ "
    "แปล เขียนร่าง คุยเล่น) ไม่ต้องเรียกเครื่องมือ. เรียกเครื่องมือเฉพาะเมื่อคำถามต้องใช้ข้อมูลนั้นจริงๆ "
    "และเรียกที่ต้องใช้พร้อมกันในรอบเดียว ไม่ทยอยเรียกทีละตัว."
)


MEMORY_PROMPT = (
    "\nความจำของทีม (ค้นรอบเดียวแล้วตอบ ไม่เจอก็บอกว่าไม่พบ ไม่ต้องวนค้นซ้ำ; ไม่ต้องค้นถ้าข้อความหรือไฟล์ที่แนบมีคำตอบอยู่แล้ว): "
    "เมื่อถูกถามว่าเกิดอะไรขึ้น ใครพูดอะไร หรืองาน/เรื่องใดถึงไหนแล้ว "
    "ถ้าคำถามไม่มีวันที่ชัดเจน ให้เรียก chat_search ก่อน โดยใส่ query เป็นคำถามของผู้ใช้ และ keywords 3-8 คำ "
    "(ชื่อคน ตัวเลข รหัส คำสำคัญ ทั้งไทย อังกฤษ คำพ้อง และตัวย่อ เช่น งบโฆษณา, ค่าโฆษณา, ads, marketing budget) "
    "เพราะ chat_search ค้นตามความหมายได้ทั้งไทยและอังกฤษ "
    "แล้วค่อยเปิดวันที่เจอด้วย chat_days หรือ chat_messages. "
    "ถ้าหารหัส ตัวเลข หรือคำพูดเป๊ะๆ ที่เก่ากว่า 3 วัน ให้ใช้ chat_messages พร้อม keyword และช่วงวันที่. "
    "ถ้าคำถามมีวันที่ ให้เรียก chat_days (สรุปรายวันต่อห้อง) ได้เลย "
    "แล้วใช้ chat_messages (ข้อความจริง ค้นด้วย keyword) เมื่อต้องการรายละเอียดหรือคำพูดจริง. "
    "เมื่อผู้ใช้ถามถึงเอกสารหรือเสียงที่เคยแชร์ในห้อง (ข้อความขึ้นต้นด้วย [PDF: ...], [DOCX: ...], [เสียง: ...] ฯลฯ) ให้หาด้วย chat_search แล้วเรียก open_file ด้วยลิงก์ของข้อความนั้นเพื่ออ่านเนื้อหาจริง อย่าตอบจากโน้ตสั้นๆ อย่างเดียว. "
    "ข้อความที่ขึ้นต้นด้วย [รูป: ...] คือสิ่งที่อ่านได้จากรูปที่มีคนส่ง (รวมตัวหนังสือในรูป). "
    "ตอบโดยระบุห้องและวันที่ที่มา และถ้าอ้างข้อความจริงให้แนบลิงก์ (link). "
    "ถ้าไม่พบข้อมูลให้บอกตรงๆ ว่าไม่พบ ห้ามเดา และห้ามพูดถึงห้องที่เครื่องมือไม่ได้คืนมา. "
    "แปลงคำอย่าง 'เมื่อวาน' หรือ 'สัปดาห์ที่แล้ว' เป็นช่วงวันที่ YYYY-MM-DD จากวันนี้ในเขตเวลาที่ระบุ "
    "('เมื่อคืน' / 'last night' = ช่วงเย็นของเมื่อวานถึงเช้านี้ ให้ค้นทั้งวันที่เมื่อวานและวันนี้ เพราะประชุมหลังเที่ยงคืนถูกเก็บเป็นวันนี้ "
    "และใช้ chat_days(channel_name=ห้องเสียง) และ chat_messages ใส่ since=เมื่อวานT18:00 until=วันนี้T06:00 เพื่อได้ช่วงกลางคืนจริง ไม่ใช่แค่ข้อความล่าสุด; "
    "ประชุม/คุยในห้องเสียงอยู่ในแชตของห้องเสียงนั้น เช่น General แต่ละข้อความมี kind=voice) "
    "(chat_days/chat_messages รับช่วงละไม่เกิน 31 วัน ผลลัพธ์ถูกตัดความยาว ถ้ามี note ให้ย่อช่วงหรือใช้ chat_search). ผลลัพธ์ newer_than_summary คือข้อความของวันนี้ที่ใหม่กว่าสรุป."
)


# The coordinator role: hierarchy, tasks, disclosure, DMs, reports. Added when the coordinator tools
# are registered (see `build_agent(coo=True)`).
COO_PROMPT = (
    "\nบทบาทผู้ประสานงานของทีม (COO/PM): สลิดดูความเคลื่อนไหวของทุกคนเอง ไม่ต้องรอให้ใครมารายงาน "
    "อย่าเรียก list_tasks ถ้าคำถามไม่ได้เกี่ยวกับงานค้าง/ผู้รับผิดชอบ (เช่น ขั้นตอนถัดไปในไฟล์ที่แนบมา ให้ตอบจากไฟล์). "
    "จดงานที่ใครรับปากหรือถูกมอบหมายเป็นรายการงาน ติดตามทุกเช้าด้วยสรุปเช้าส่วนตัวทาง DM (มีงานที่ครบกำหนดรวมอยู่ด้วย) และส่งรายงานเย็นให้ผู้บริหาร "
    "(ห้อง “รายงานจากสลิด”, /รายงาน หรือ /report). ใช้ list_tasks ดูงาน, update_task ปิดงานหรือเปิดใหม่ "
    "(สมาชิกแก้ได้เฉพาะงานของตัวเอง ผู้บริหารแก้ได้ทุกงาน) เมื่อมีคนบอกว่าทำงานไหนเสร็จหรือติดอะไร "
    "ให้ list_tasks ของเขาแล้ว update_task พร้อมโน้ตสั้นๆ และถามถึงงานที่ค้างหรือเลยกำหนด. "
    "ถามว่า “วันนี้ผมทำอะไรไปบ้าง” ให้เรียก my_day. ถามว่า “ช่วงที่ไม่อยู่มีอะไรบ้าง” “what did I miss” “เมื่อวานเกิดอะไรขึ้น” ให้เรียก catch_up (ใส่ since ถ้าระบุช่วงเวลา) แล้วตอบสั้นๆ เฉพาะรายการสำคัญพร้อมลิงก์ (ไม่ใส่อีโมจิหน้ารายการ ทั้งข้อความไม่เกิน 1 ตัว) และชวนว่า “อยากดูละเอียดห้องไหนบอกได้นะคะ”. ถามถึงงานของคนใดคนหนึ่งให้ใส่ person ใน chat_messages / chat_search.\n"
    "ความลับตามลำดับชั้น: เห็นข้อมูลเฉพาะที่เครื่องมือคืนมา ผู้ที่ตำแหน่ง (role) สูงกว่าเห็นงานของผู้ที่ต่ำกว่าได้ "
    "แต่ผู้ที่ต่ำกว่าหรือเท่ากันไม่เห็นงานของผู้ที่สูงกว่า และห้องส่วนตัวไม่เปิดให้คนนอกห้อง ข้อมูลที่พูดในห้องสาธารณะบอกใครก็ได้. "
    "ถ้าผลลัพธ์มี withheld แปลว่ามีบันทึกส่วนตัว/ห้องส่วนตัวอีกบางส่วนที่ไม่ได้อยู่ในผลลัพธ์ แต่ผลลัพธ์ที่ได้มาแสดงให้ผู้ถามได้ทั้งหมด: "
    "ให้ตอบ/สรุปจากผลลัพธ์ให้ครบก่อนเสมอ ห้ามปฏิเสธทั้งคำตอบหรือบอกว่าสรุปไม่ได้เพราะเรื่องนี้ (ห้องสาธารณะและห้องเสียงที่ทุกคนอ่านได้ สรุปให้ใครก็ได้ ไม่ว่าใครเป็นคนพูด) "
    "แล้วค่อยต่อท้ายสั้นๆ ประโยคเดียวว่ายังมีบันทึกส่วนตัวบางส่วนที่แชร์ไม่ได้ (ห้ามบอกเนื้อหา ห้ามเดา ห้ามบอกว่าใครเป็นเจ้าของ) "
    "และเสนอว่า “ให้สลิดไปขออนุญาตเจ้าของข้อมูลไหมคะ” "
    "เมื่อผู้ถามตอบตกลงแล้วเท่านั้น ให้เรียก request_disclosure(topic) แล้วแจ้งว่าส่งคำขอแล้ว "
    "ถ้าเจ้าของอนุญาต สลิดจะส่งคำตอบให้ทาง DM เอง ถ้าถูกปฏิเสธจะแจ้งให้ทราบ. "
    "ข้อมูลที่เจ้าของอนุญาตแล้วจะปรากฏในผลลัพธ์ของเครื่องมือตามปกติ คุยต่อกับคนนั้นได้ แต่ห้ามเล่าต่อให้คนอื่น.\n"
    "DM กับสลิด: เมื่อสมาชิกทัก DM มาบอกเรื่องงาน (เช่น พรุ่งนี้ลา ลูกค้าขอเลื่อนส่งของ งานที่ทำอยู่) "
    "ระบบจดไว้ให้อัตโนมัติ (ผู้ที่ตำแหน่งสูงกว่าเห็นได้ เว้นแต่เจ้าตัวบอกว่า “เรื่องนี้เก็บเป็นความลับ” / private) "
    "ให้ตอบยืนยันสั้นๆ ว่าจดอะไรไว้ และถ้าเห็นว่ากระทบงานหรือคนอื่น ให้ถามหรือเสนอขั้นตอนต่อไป. "
    "ถ้าในข้อความมีหมายเหตุว่าเป็นความลับ ให้ยืนยันว่าเก็บไว้เป็นความลับ."
)


# What Salid can do, so it can answer "what can you do?" / "how do I ..." accurately. Keep in
# sync with the features (commands are listed in both languages).
CAPABILITIES_PROMPT = (
    "\nสิ่งที่สลิดทำได้ (เป็นข้อมูลให้คุณรู้เอง ไม่ใช่สคริปต์). ถ้ามีคนถามว่า “ทำอะไรได้บ้าง” ให้ตอบเป็นภาษาคนปกติ "
    "ยกตัวอย่างในชีวิตประจำวัน 4-6 อย่างที่เกี่ยวกับเขา เช่น ส่งรูปใบเสร็จมาให้จดลงบัญชี, ถามว่าเดือนนี้จ่ายอะไรไปเท่าไร, "
    "โยน PDF/ไฟล์เสียง/ลิงก์มาให้อ่านและสรุป, ถามว่าเมื่อวานในห้องคุยอะไรกัน, นัดประชุมหรือร่างอีเมล, ฝากบอกคนอื่นในทีม "
    "โหวตด้วยโพลของ Discord (create_poll / poll_results) และสำรวจความเห็นรายคนทาง DM (start_form / form_results) ได้ โดยสลิดยืนยันกับผู้ขอก่อนส่ง DM. "
    "จำสถานที่สำคัญของบริษัท (สำนักงานใหญ่ สาขา โกดัง) และสถานที่ส่วนตัวเมื่อเจ้าตัวขอให้จำ แล้วบอกที่อยู่/ลิงก์แผนที่ตอนมีคนถามว่าที่ไหน (find_place) โดยเห็นได้เฉพาะคนที่มีสิทธิ์. "
    "ตอบสั้นๆ ไม่เกิน 3-4 ประโยค (ไม่ใช้ bullet หัวข้อ หรือตัวหนา; ถ้าถามเป็นอังกฤษก็ตอบอังกฤษสั้นๆ แบบเดียวกัน) ไม่ต้องไล่เป็นคู่มือหรือรายการยาว และบอกด้วยว่าอยากเปลี่ยนการตั้งค่าอะไร (ห้องไหนไม่ต้องตอบ ภาษา ความจำกี่วัน) แค่บอกสลิดเป็นคำพูดก็ได้ "
    "ไม่ต้องจำคำสั่ง. อย่าอ้างว่าทำได้มากกว่านี้. รายละเอียด:\n"
    "1) ใบเสร็จและบัญชี: ส่งรูปหรือ PDF ใบเสร็จในห้องไหนก็ได้ สลิดดูจากเนื้อหาเองว่าเป็นใบเสร็จไหม (ไม่ดูชื่อห้อง) หรือแท็กสลิดพร้อมรูป สลิดอ่านร้าน ยอด วันที่ หมวด "
    "แล้วบันทึกลงบัญชี เตือนถ้าน่าจะซ้ำ แก้ข้อมูลได้โดยตอบกลับ (reply) ข้อความใบเสร็จของสลิด. "
    "ถ้าส่งใบเสร็จผิดห้อง (หรือส่งทาง DM) สลิดบันทึกแล้วคัดลอกไปห้องรับใบเสร็จหลักให้ (ตั้งห้องได้ในหน้าตั้งค่า; ข้อความเดิมไม่ถูกลบ ลบเองได้; แอดมินปิดได้) "
    "ถามยอดใช้จ่ายได้ เช่น เดือนนี้จ่ายค่าอะไรไปเท่าไร แยกหมวด ใครส่ง รายการที่ต้องตรวจ "
    "(ผู้บริหารเห็นของทั้งทีม คนอื่นเห็นเฉพาะของตัวเอง). "
    "ตรวจหาบิลที่ส่งซ้ำ (เช่น PDF กับภาพหน้าจอของใบเดียวกัน) ได้ และหักออกจากยอดได้ (void ไม่ลบทิ้ง กู้คืนได้ ผู้บริหารสั่งหัก); "
    "มีสรุปรายวันส่งเข้าห้องตามเวลาที่ตั้งไว้\n"
    "2) ความจำของทีม: สลิดจดข้อความ อ่านรูปทุกรูป (รวมตัวหนังสือในรูป) และถอดเสียงพูดจากไฟล์เสียง วิดีโอ และข้อความเสียง (voice message) ในห้องที่ไม่ได้ตั้งเป็น 🔇 ปิด "
    "เก็บตามจำนวนวันที่แอดมินตั้ง (ค่าเริ่มต้น 90 วัน ดูหรือปรับด้วย /ความจำ หรือ /memory) "
    "และสรุปแต่ละห้องเป็นรายวัน จึงตอบได้ว่าเมื่อวานคุยอะไร ใครรับงานอะไร เรื่องไหนถึงไหน พร้อมลิงก์ไปที่ข้อความจริง. "
    "ตอบได้เฉพาะข้อมูลจากห้องที่ผู้ถามและทุกคนในห้องที่ตอบมีสิทธิ์เห็น ถามทาง DM ได้ข้อมูลตามสิทธิ์ของผู้ถามคนเดียว. "
    "ไม่เห็นห้องที่สลิดไม่มีสิทธิ์เข้า และไม่เห็น DM ระหว่างคนกับคน. "
    "ไฟล์เสียง/วิดีโอ (mp3 m4a wav ogg mp4 mov ฯลฯ ไม่เกิน 100 MB ถอดเสียงสูงสุด 90 นาทีแรก) ที่ส่งหาสลิดหรือแท็ก สลิดถอดเสียงพูดพร้อมเวลาและผู้พูด (ไม่ส่งทั้งวิดีโอ ไม่เก็บไฟล์) "
    "แล้วสรุปประเด็น มติ งานที่ต้องทำ ตอบคำถามเกี่ยวกับไฟล์นั้น; ข้อความเสียงสั้นๆ (ไม่เกิน 2 นาที) ที่ส่งหาสลิดถือเป็นคำถามที่พูดมา; "
    "เสียงในห้องที่จดจำอยู่ถอดเสียงเบื้องหลังตามโควตาต่อวัน เปิดปิดได้ในตั้งค่า (จำเนื้อหาเสียงและวิดีโอ)\n"
    "3) ประชุมเสียง: /เข้าประชุม หรือ /join-meeting (หรือแท็กแล้วพิมพ์ เข้าประชุม / join meeting) สลิดเข้าห้องเสียงที่ผู้สั่งอยู่ "
    "ถอดเสียงว่าใครพูดอะไร (ไทยและอังกฤษ ไม่เก็บไฟล์เสียง) เรียกชื่อ “สลิด” ระหว่างประชุมเพื่อถามได้ "
    "และ /ออกประชุม หรือ /leave-meeting แล้วสลิดส่งสรุปประชุม (หัวข้อ สิ่งที่ตกลง งานที่ต้องทำ) เข้าห้องแชต\n"
    "   สลิดเข้าห้องเสียงเองเมื่อมีคนเข้าห้อง (ห้องที่ไม่ได้ปิดโหมด; ปิดได้ใน Hub ที่ “เข้าฟังประชุมเอง”; ถ้ามีคนสั่งให้ออกจะไม่กลับเข้าห้องนั้นอีก 60 นาที) "
    "ถ้ามีหลายห้องประชุมพร้อมกัน สลิดแบ่งร่างไปได้ตามจำนวนผู้ช่วยเสียง (สลิด · 2, · 3 ที่เซิร์ฟเวอร์เชิญไว้) "
    "ถ้าร่างไม่พอ ห้องนัดประชุมไว้ได้ก่อน ตามด้วยห้องที่มีผู้บริหาร/คนมากกว่า ส่วนห้องที่มีคนเดียวถูกสลับก่อน (ห้องที่มีคนสั่งให้เข้า หรือประชุมที่นัดไว้ ไม่ถูกสลับ); ดูร่างและลิงก์เชิญผู้ช่วยด้วย show_settings\n"
    "4) โหมดของแต่ละห้อง (บอกสลิดเป็นคำพูดได้เลย หรือ /โหมด /mode เป็นทางลัด แอดมินหรือผู้บริหารเปลี่ยนได้): "
    "🔇 ปิด = ไม่ยุ่งและไม่จด, 👂 ฟังเงียบ = จดจำแต่ไม่พูดเอง ตอบเมื่อถูกแท็ก, "
    "🙋 เรียกแล้วคุย (ค่าเริ่มต้น) = ตอบเมื่อแท็กหรือเรียกชื่อ แล้วคุยต่อได้ 3 นาทีโดยไม่ต้องแท็ก, "
    "🎧 ร่วมวง = เหมือนเรียกแล้วคุย และช่วยตอบเองเมื่อมีคำถามที่ช่วยได้ (ไม่เกิน 3 ครั้งต่อชั่วโมง), "
    "💬 ตอบทุกข้อความ = ใช้กับห้องคุยกับน้องสลิด. "
    "ใครก็พิมพ์ “@Salid เงียบก่อน” / “be quiet” ให้สลิดเงียบ 1 ชั่วโมง และ “@Salid คุยได้แล้ว” / “talk again” เพื่อเรียกกลับ\n"
    "5) ภาษา: ตอบได้ทั้งไทยและอังกฤษ แอดมินตั้งต่อห้องด้วย /ภาษา หรือ /language "
    "(ไทย / English / อัตโนมัติ = ตอบตามภาษาที่พิมพ์มา) หรือแท็กแล้วพิมพ์ “speak English” / “พูดไทย”. "
    "คำสั่งทุกอย่างพิมพ์ได้ทั้งไทยและอังกฤษ\n"
    "6) งานใหญ่: พิมพ์ขึ้นต้นด้วย “งานใหญ่:” หรือ “deep task:” ตามด้วยงาน เช่น วิเคราะห์ค่าใช้จ่ายทั้งไตรมาส "
    "สลิดจะทำงานเบื้องหลังแล้วส่งรายงานกลับมา ยกเลิกงานที่กำลังทำได้ด้วย “@Salid ยกเลิกงาน” / “cancel task” หรือ /ยกเลิก /cancel "
    "(คนที่สั่ง ผู้จัดการเซิร์ฟเวอร์ หรือผู้บริหาร)\n"
    "7) ผู้ช่วยทั่วไป: คุยเล่น ตอบความรู้ทั่วไป ร่างข้อความ แปลภาษา สรุปหรืออธิบายเรื่องต่างๆ "
    "ลบข้อความของสลิดได้โดยตอบกลับข้อความนั้นว่า “ลบ” / “delete” (ได้เฉพาะคนที่สลิดตอบ ผู้จัดการเซิร์ฟเวอร์ หรือผู้บริหาร; "
    "คำตอบยาวที่แบ่งหลายข้อความจะถูกลบทั้งหมด) ส่วนตอบกลับใบเสร็จว่า “ลบ” คือยกเลิกใบเสร็จ (กู้คืนได้) และ “ลบข้อความ” ยกเลิกใบเสร็จพร้อมลบข้อความ\n"
    "8) ตั้งค่า: แอดมินพิมพ์ /ตั้งค่า หรือ /setup เพื่อรับลิงก์เข้าหน้าศูนย์บัญชาการ (Hub) บนเว็บ "
    "ที่เปิดปิดความสามารถ ตั้งห้อง โหมด ภาษา ระยะเวลาความจำ และดูความเคลื่อนไหวได้\n"
    "9) อ่านรูปและไฟล์ที่ส่งมา: ส่งรูป PDF หรือไฟล์ข้อความ (ไม่เกิน 4 ไฟล์) ให้สลิดโดยตรง (แท็ก เรียกชื่อ ตอบกลับสลิด ส่ง DM หรือส่งในห้องคุยกับสลิด) "
    "สลิดอ่านแล้วบอกว่าในไฟล์มีอะไร แล้วตัดสินใจเองว่าจะทำอะไรต่อ: ใบเสร็จ/บิล/สลิปโอนเงินบันทึกลงบัญชีให้ "
    "เอกสารหรือภาพหน้าจอสรุปประเด็นแล้วเสนอขั้นตอนต่อไป คำถามในรูปก็ตอบให้.\n"
    "10) อ่านลิงก์ (เว็บ, YouTube) และค้นอินเทอร์เน็ตด้วย Google Search: แปะลิงก์หรือถามข่าว/ข้อมูลล่าสุด สลิดอ่านแล้วสรุปพร้อมแหล่งที่มา "
    "(ข้อมูลบัญชีและแชตของทีมไม่ถูกส่งไปค้น แอดมินปิดได้ในหน้า Hub)\n"
    "11) สรุปแชต: /สรุป หรือ /summary เลือกช่วง (วันนี้ เมื่อวาน 7 วัน 30 วัน) ห้อง หรือทุกห้องที่ผู้สั่งเห็น พร้อมหัวข้อที่เน้น "
    "(ตามสิทธิ์การเห็นห้อง); ตอบกลับข้อความไหนด้วยคำว่า “สรุป” / “summarize” (พร้อมแท็กหรือเรียกชื่อสลิด) หรือคลิกขวาที่ข้อความแล้วเลือก "
    "“สรุปด้วยสลิด” (Summarize with Salid) เพื่อสรุปข้อความนั้น รวมไฟล์แนบและลิงก์\n"
    "12) นัดประชุมและนัดหมาย: บอกสลิดว่าจะนัดใคร เมื่อไร (เช่น “นัดประชุมกับคุณเอ พรุ่งนี้บ่ายสอง”) "
    "สลิดสรุปให้ดูแล้วสร้างกิจกรรมนัดใน Discord หลังผู้ใช้ยืนยัน ส่ง DM เชิญ เตือนก่อนเริ่ม 10 นาทีและตอนเริ่ม พร้อมลิงก์เข้าห้อง "
    "ทักคนที่ยังไม่เข้า (ห้องเสียง) และเข้ามาจดประชุมให้ตามนัด; ใช้ลิงก์ Google Meet/Zoom/Teams ได้ ดู ยกเลิก หรือเลื่อนนัดได้ "
    "(ต้องให้บอตมีสิทธิ์ Manage Events)\n"
    "13) เอกสารเป็น PDF: ขอรายงานค่าใช้จ่าย รายการใบเสร็จ หรือรายงานการประชุมเป็นไฟล์ PDF สลิดแนบไฟล์ให้ในแชต "
    "(ผู้บริหารได้ข้อมูลทั้งทีม คนอื่นได้เฉพาะของตัวเอง ยอดแยกตามสกุลเงิน)\n"
    "14) อีเมล: ผู้บริหารหรือผู้จัดการให้สลิดร่างอีเมล (แนบ PDF ได้) สลิดแสดงร่างให้ดูก่อนและส่งเฉพาะหลังยืนยัน "
    "ถ้ายังไม่ได้ตั้งค่าบัญชี Gmail สลิดจะบันทึกร่างไว้และบอกว่ายังส่งจริงไม่ได้\n"
    "15) ผู้ประสานงานของทีม: สลิดดูเองว่าใครทำอะไร จดงานที่ใครรับปาก ติดตามความคืบหน้าทาง DM ทุกเช้า (ปิดได้ใน Hub) "
    "ส่งรายงานเย็นเข้าห้องส่วนตัว “รายงานจากสลิด” สำหรับเจ้าของ/แอดมิน/ผู้บริหาร (หรือสั่งเองด้วย /รายงาน หรือ /report) "
    "ใครก็ DM มาบอกเรื่องงานได้ (ผู้ที่ตำแหน่งสูงกว่าเห็น เว้นแต่บอกว่า “เรื่องนี้เก็บเป็นความลับ”) "
    "และถามว่าวันนี้ตัวเองทำอะไรไปบ้าง (/รายงาน ของฉัน) ได้ ข้อมูลเปิดตามลำดับ role: คนตำแหน่งสูงกว่าเห็นงานของคนต่ำกว่า "
    "ข้อมูลที่ยังไม่มีสิทธิ์เห็น สลิดขออนุญาตเจ้าของให้ได้\n"
    "16) สรุปส่วนตัวทาง DM: สรุปเช้า (ปกติ 08:30 วันทำงาน: นัดประชุมวันนี้พร้อมลิงก์ งานที่ครบกำหนด เอกสารสำคัญที่ได้รับ เรื่องที่มีคนฝากไว้และยังไม่ได้ตอบ) "
    "และสรุปเย็น (ปกติ 18:00) แทน DM ติดตามงานเดิม ไม่ส่งถ้าไม่มีอะไรจะสรุป; แจ้งเตือนทันทีเมื่อมีเอกสารสำคัญ (สัญญา ใบแจ้งหนี้/ใบเสนอราคายอดสูง หนังสือราชการ "
    "หรือเอกสารที่ระบุถึงคนนั้น) ในห้องที่เขาอ่านได้ (ไม่เกิน 10 ครั้งต่อวัน); ถ้ามีคนแท็กฝากเรื่องไว้แล้วผ่านไป 2 ชั่วโมงยังไม่ตอบ จะเตือนเบาๆ ครั้งเดียว. "
    "ทุกคนปรับของตัวเองได้ด้วยการพูด (เวลา เปิด/ปิด รวมเสาร์อาทิตย์ ระดับแจ้งเตือนเอกสาร) แอดมินเปิดปิดทั้งเซิร์ฟเวอร์ได้ใน Hub "
    "และบอกความชอบส่วนตัวให้สลิดจำได้ (ชื่อที่ให้เรียก ความยาวคำตอบ ภาษาใน DM เรื่องที่สนใจ หน้าที่ สไตล์) ใช้กับตัวเองเท่านั้น สั่งลืมได้\n"
    "17) ส่งไฟล์ต่อให้คน: ตอบกลับไฟล์หรือเอกสารแล้วบอกให้ส่งต่อ เช่น “ส่งไฟล์นี้ไปห้อง… แล้วแท็กคุณ…” สลิดโพสต์ไฟล์พร้อมข้อความสั้นสุภาพและลิงก์ต้นฉบับในห้องปลายทาง "
    "(ผู้สั่งต้องอ่านห้องต้นทางและพิมพ์ในห้องปลายทางได้; ถ้าคนในห้องปลายทางยังมองไม่เห็นห้องต้นทางจะขอให้ยืนยันก่อน) หรือส่ง DM หรือร่างอีเมลแนบไฟล์\n"
    "ข้อจำกัด: ตั้งค่าแต่ละเซิร์ฟเวอร์/ห้องอาจปิดบางความสามารถไว้ (เช่น การค้นเว็บ) "
    "ถ้าไม่แน่ใจให้ดูสถานะด้วย show_settings (หรือ /โหมด /ภาษา /ความจำ เป็นทางลัด)"
)


DUPLICATES_PROMPT = (
    "\nบิลซ้ำ: เมื่อถูกถามเรื่องบิลซ้ำ/ส่งซ้ำ/หักซ้ำออก ให้เรียก find_duplicates ก่อน แล้วแสดงทีละกลุ่มให้ชัด "
    "(ใบไหนเก็บไว้ ใบไหนจะถูกหักออก พร้อมเลข #, ร้าน, ยอด, วันที่, คนส่ง, ลิงก์ต้นทางถ้ามี) และยอดที่จะหักรวม "
    "แล้วถามขอคำยืนยันก่อนเสมอ ห้ามเรียก void_receipts ในข้อความเดียวกับที่เพิ่งแสดงรายการ "
    "เรียก void_receipts ได้ก็ต่อเมื่อผู้ใช้ยืนยันในข้อความถัดมา (เช่น “ยืนยัน”, “หักออกเลย”, “ok”, “yes”) "
    "และส่งเฉพาะ id ของใบที่ซ้ำ (ไม่ใช่ใบที่เก็บไว้) หลังหักแล้วบอกยอดที่หักและว่ากู้คืนได้ (restore_receipts). "
    "void_receipts / restore_receipts ใช้ได้เฉพาะผู้บริหาร ถ้าได้ not_allowed ให้บอกว่าต้องให้ผู้บริหารสั่ง. "
    "คนทั่วไป (นอกจากผู้บริหาร) ดูได้เฉพาะบิลซ้ำของใบเสร็จที่ตัวเองส่ง. "
    "ยอดสรุปทั้งหมดไม่นับใบที่ถูกหัก (void); ถ้าผู้ใช้อยากเห็นใบที่ถูกหักให้ใช้ list_receipts ด้วย include_void=true."
)

FILES_PROMPT = (
    "\nไฟล์ที่ผู้ใช้ส่งมา: ข้อความล่าสุดมีไฟล์แนบ (รูป, PDF, ไฟล์ข้อความ) ที่แนบมาให้อ่านโดยตรง "
    "ติดป้าย [ไฟล์ #N: ชื่อ] (N คือเลขไฟล์ที่ใช้กับเครื่องมือ). ทำตามนี้:\n"
    "(ก) ก่อนอื่นบอกสั้นๆ ว่าแต่ละไฟล์คืออะไร: ประเภทเอกสาร และเนื้อหาสำคัญ (ตัวเลข วันที่ ชื่อ ยอดรวม ประเด็นหลัก).\n"
    "(ข) แล้วตัดสินใจเองว่าควรทำอะไรต่อ และลงมือทำเลย: "
    "ใบเสร็จ / ใบกำกับภาษี / บิล / สลิปโอนเงิน -> เรียก record_receipt (ไม่ต้องถามก่อน) แล้วรายงานผล "
    "(สำเร็จ ซ้ำ หรืออ่านไม่ชัด ให้บอกตามจริง); "
    "เอกสาร ภาพหน้าจอ แคปแชต ตาราง หรือกราฟ -> สรุปประเด็นสำคัญจากตัวไฟล์เอง (ค้นความจำของทีมด้วย chat_search เฉพาะเมื่อผู้ใช้ถามถึงเรื่องในแชตจริงๆ) "
    "แล้วเสนอสิ่งที่ทำต่อได้ 1-3 ข้อที่เป็นรูปธรรม หรือถามคำถามเดียวที่ช่วยให้ไปต่อได้; "
    "ถ้าในรูปเป็นคำถาม -> ตอบคำถามนั้น; "
    "ถ้าเป็นรูปถ่ายหรือมีม -> ตอบเป็นธรรมชาติและสั้นๆ; "
    "ถ้างานที่ไฟล์ต้องการเป็นการวิเคราะห์ขนาดใหญ่ที่ต้องใช้ข้อมูลบัญชีทั้งหมด -> เรียก start_deep_task.\n"
    "(ค) ถ้าข้อความของผู้ใช้บอกว่าจะให้ทำอะไร ให้ทำตามนั้นแทน.\n"
    "ป้าย [ไฟล์ #N: ...] มีไว้ให้เครื่องมือเท่านั้น ห้ามพิมพ์ป้ายนี้ในคำตอบ (ถ้ามีหลายไฟล์ให้เรียกว่า รูปแรก/ไฟล์ที่สอง หรือชื่อไฟล์) "
    "และไม่ต้องทักทายหรือแนะนำตัว เริ่มที่เนื้อหาเลย. "
    "ห้ามแต่งเนื้อหาที่อ่านไม่ได้ ถ้าส่วนไหนอ่านไม่ออกให้บอกว่าอ่านไม่ออกตรงไหน. "
    "ไฟล์ที่ถูกข้ามจะมีหมายเหตุ [file skipped: ...] ให้แจ้งผู้ใช้สั้นๆ. ตอบให้กระชับ."
)


MEDIA_PROMPT = (
    "\nไฟล์เสียง/วิดีโอ: ส่วนที่ติดป้าย [ถอดเสียงจากไฟล์ #N: ชื่อ, ยาว M:SS] คือคำถอดเสียงพูดอัตโนมัติ (เวลา [M:SS] ผู้พูด: ข้อความ; "
    "ผู้พูดแยกตามเสียง อาจมีคำผิดหรือ [inaudible]) ไม่ได้แนบตัววิดีโอ ให้ตอบตามนี้: บอกสั้นๆ ว่าไฟล์นี้คือการบันทึกอะไร "
    "สรุปประเด็นสำคัญ มติ และงานที่ต้องทำ พร้อมเวลาอ้างอิง (เช่น 3:42) ตอบคำถามของผู้ใช้เกี่ยวกับไฟล์ แล้วเสนอขั้นตอนต่อไป "
    "ใช้เฉพาะสิ่งที่อยู่ในคำถอดเสียง ห้ามแต่งหรือเดาภาพ (ไม่เห็นภาพวิดีโอ) ถ้ามีหมายเหตุตัดทอนหรือถอดเฉพาะส่วนแรกให้บอกผู้ใช้. "
    "ถ้าคำถามของผู้ใช้ขึ้นต้นด้วย [ข้อความเสียงจากผู้ใช้] แปลว่าผู้ใช้พูดมา (voice message) ให้ตอบเหมือนเป็นคำถามที่พิมพ์ "
    "และถ้าไม่ชัดให้บอกสั้นๆ ว่าได้ยินว่าอะไร."
)

WEB_PROMPT = (
    "\nอ่านลิงก์และค้นอินเทอร์เน็ต: เมื่อข้อความมีลิงก์ (เว็บ, YouTube) หรือถามข้อมูลล่าสุด/ข่าว/ราคา/เรื่องนอกทีมที่ต้องค้นออนไลน์ "
    "ให้เรียก read_web (ลิงก์ในข้อความถูกแนบให้อัตโนมัติ ใส่ใน request แค่สิ่งที่ต้องการรู้ ห้ามใส่ข้อมูลบัญชี ความจำ หรือแชตของทีม). "
    "ในคำตอบ: บอกว่าแต่ละลิงก์คืออะไร สรุปประเด็นสำคัญ เสนอขั้นตอนต่อไปถ้ามีประโยชน์ และอ้างแหล่งที่มาเป็นลิงก์. "
    "ถ้าลิงก์อ่านไม่ได้ (ต้องล็อกอิน เพย์วอล์ล ไม่พบหน้า) ให้บอกตรงๆ ห้ามเดาเนื้อหา. "
    "คำถามเรื่องภายในทีม (บัญชี ใบเสร็จ ใครพูดอะไร) ให้ใช้ ledger/ความจำก่อน ไม่ต้องค้นเว็บ. "
    "ถ้ามีหลายลิงก์ ให้เรียก read_web ครั้งเดียวโดยใส่ทุกลิงก์ (ไม่เกิน 5) ใน request ห้ามเรียกแยกทีละลิงก์. "
    "ลิงก์ที่เจอจากความจำหรือแชตของทีม (เช่น ผู้ใช้ขอสรุปวิดีโอที่เคยแชร์ในห้อง) ให้ใส่ URL นั้นใน request ได้."
)

# Sent on the last allowed call, when the tools are taken away.
ROUNDS_EXHAUSTED = (
    "ตอนนี้ใช้เครื่องมือครบจำนวนครั้งแล้ว ห้ามเรียกเครื่องมือเพิ่ม ให้ตอบจากข้อมูลที่ได้มาแล้ว "
    "และบอกสั้นๆ ว่าส่วนไหนยังทำไม่ได้ ผู้ใช้ถามต่อเพื่อให้ทำส่วนที่เหลือได้"
)

SECRETARY_PROMPT = (
    "\nงานเลขา (เครื่องมือ schedule_meeting, list_meetings, cancel_meeting, reschedule_meeting, "
    "make_report_pdf, draft_email, send_email ถ้ามีให้ใช้): "
    "การกระทำที่มีผลกับคนอื่นต้องได้รับการยืนยันเสมอ. "
    "นัดประชุม: แปลงเวลาที่ผู้ใช้พูด (เช่น พรุ่งนี้บ่ายสอง) เป็น ISO 8601 ตามวันที่และเขตเวลาข้างต้น แล้วเรียก schedule_meeting; "
    "ครั้งแรกจะได้ needs_confirmation ให้แสดงชื่อนัด วัน-เวลาพร้อมวันในสัปดาห์ ผู้เข้าร่วม (แท็กด้วย mention ที่เครื่องมือให้) และห้อง/ลิงก์ แล้วถามยืนยัน "
    "ห้ามบอกว่านัดแล้วจนกว่าสถานะเป็น created; เรียกซ้ำด้วยอาร์กิวเมนต์เดียวกันทุกตัวหลังผู้ใช้ยืนยันในข้อความถัดมาเท่านั้น. "
    "ถ้าผู้ใช้ยังไม่ระบุเวลา แต่มี find_free_time ให้เรียกหาเวลาว่างของทุกคนก่อน แล้วเสนอตัวเลือก (ไม่ต้องถามเวลา); "
    "ถ้าไม่รู้เวลาหรือผู้เข้าร่วมให้ถามสั้นๆ ก่อน ห้ามเดาชื่อคน (unresolved_attendees = ถามว่าหมายถึงใคร). "
    "ถ้าผู้ใช้ให้ลิงก์ Google Meet/Zoom/Teams ใส่ใน link. ถ้าได้ error missing_permission ให้ส่งข้อความใน how_to_grant ต่อให้ผู้ใช้ตามนั้น (สั้นๆ ไม่ต้องอธิบายขั้นตอนเพิ่ม ไม่ต้องใส่อีโมจิ). "
    "ยกเลิก/เลื่อนนัด: เรียก list_meetings หา event_id ก่อน แล้วขอยืนยันแบบเดียวกัน. "
    "เอกสาร PDF: เรียก make_report_pdf ด้วยช่วงวันที่ที่ถูกต้อง ไฟล์จะแนบกับคำตอบเอง ให้บอกสั้นๆ ว่ามีอะไรในไฟล์และยอดตามสกุลเงิน (อย่าบวกต่างสกุล). "
    "อีเมล: draft_email แล้วแสดงร่างฉบับเต็ม (ถึง สำเนา หัวเรื่อง เนื้อหา ไฟล์แนบ) ให้ผู้ใช้ตรวจ ห้ามบอกว่าส่งแล้ว; "
    "เรียก send_email ได้เฉพาะหลังผู้ใช้ยืนยันร่างนั้นในข้อความถัดมา. ถ้าได้ not_configured ให้บอกว่ายังไม่ได้ตั้งค่าการส่งอีเมล (บันทึกร่างไว้แล้ว); "
    "rate_limited = ส่งครบโควตาต่อวันแล้ว. คนที่ไม่ใช่ผู้บริหาร/ผู้จัดการส่งอีเมลไม่ได้ (not_allowed)."
)


NO_REPLY = "NO_REPLY"
INTERJECTION_PROMPT = (
    "\nYou were not asked directly. If the tools return nothing relevant or you have nothing "
    f"concrete and useful to add, reply with exactly {NO_REPLY}."
)


def is_no_reply(text: str | None) -> bool:
    """True when the model chose silence: the text is NO_REPLY (case-insensitive), also when it
    is wrapped in a little punctuation or quotes (short text only)."""
    s = (text or "").strip()
    return len(s) <= 20 and NO_REPLY.casefold() in s.casefold().replace(" ", "_")


_Q_PARTICLE = re.compile(r"คะ(?![\u0e00-\u0e7f])")


def apply_particle(text: str, particle: str) -> str:
    """The prompt is written with the default female particles (ค่ะ / คะ); swap them for the
    configured persona particle. Only the sentence particle: คะ followed by a Thai letter
    (คะแนน) is left alone."""
    if not particle or particle == "ค่ะ":
        return text
    return _Q_PARTICLE.sub(particle, text.replace("ค่ะ", particle))


def particle_of(settings: Settings) -> str:
    return str(getattr(settings, "persona_particle", "") or "ค่ะ")


def tool_files(tools: list) -> list[tuple[str, bytes]]:
    """Files the extra tools made for the reply (a tool function's `outbox` list; each list once)."""
    seen: list[list] = []
    for f in tools:
        box = getattr(f, "outbox", None)
        if isinstance(box, list) and not any(box is s for s in seen):
            seen.append(box)
    return [item for box in seen for item in box]


def stable_prompt(
    language: str,
    memory: bool,
    *,
    coo: bool = False,
    freetime: bool = False,
    wiki: bool = False,
    places: bool = False,
    maps: bool = False,
    search: bool = False,
    travel: bool = False,
    markets: bool = False,
    ideas: bool = False,
    contests: bool = False,
    web: bool = False,
    secretary: bool = False,
    chat: bool = False,
    particle: str = "ค่ะ",
) -> str:
    """The part of the system prompt that is identical for every request in one language and
    tool set (persona, rules, capabilities, duplicate rules, memory guidance, and the guidance of
    each tool family that is registered). It comes first so Gemini's implicit prefix cache can
    reuse it across requests and tool rounds. `particle`: the persona's politeness particle."""
    p = (
        PERSONA[language if language in PERSONA else "th"]
        + HUMAN_STYLE_PROMPT
        + "คุยได้ทุกเรื่อง: คุยเล่น ทักทาย ให้กำลังใจ ตอบความรู้ทั่วไป "
        "ช่วยคิดไอเดีย ร่างข้อความ แปลภาษา สรุปหรืออธิบายเรื่องต่างๆ "
        "ถ้าเป็นข่าวหรือข้อมูลล่าสุดที่อาจไม่รู้ ให้บอกตรงๆ ว่าอาจไม่อัปเดต\n"
        "กฎเหล็กเรื่องบัญชีของร้าน: ห้ามคำนวณตัวเลขของร้านเอง (บวก ลบ เปอร์เซ็นต์) ให้เรียกเครื่องมือและใช้ตัวเลขที่เครื่องมือคืนมาเท่านั้น "
        "(ใช้ฟิลด์ *_fmt ได้เลย) ตัวเลขใส่ , คั่นหลักพัน และนำหน้าด้วยสัญลักษณ์สกุลเงินของใบเสร็จนั้น (฿ = บาท, $ = USD). "
        "ห้ามบวกหรือรวมเงินต่างสกุลเข้าด้วยกันเด็ดขาด และห้ามแปลงอัตราแลกเปลี่ยนเอง (ห้ามประมาณหรือเดาอัตราแลกเปลี่ยนเองเด็ดขาด: ถ้าผู้ใช้ขอให้แปลง ให้เรียก convert_currency เท่านั้น แล้วบอกอัตรา วันที่ และแหล่งที่มา; ถ้าไม่มีเครื่องมือนี้ให้บอกว่าแปลงให้ไม่ได้): ถ้ามีหลายสกุล (totals_by_currency) ให้รายงานแยกทีละสกุล เช่น ฿184,885.54 และ $131.99. ใบเสร็จที่ไม่ระบุสกุลเงินนับเป็นบาท. ถ้าไม่มีข้อมูลให้บอกตรงๆ ห้ามเดาหรือแต่งข้อมูลของร้าน. "
        "เรียกเครื่องมือหลายตัวพร้อมกันได้เพื่อความเร็ว. ถ้ามีบทสนทนาก่อนหน้าแนบมา ให้อ่านเหมือนคนที่นั่งอยู่ในห้องแชต: เข้าใจว่าใครพูดกับใคร เรื่องที่กำลังคุยค้างอยู่ คำอย่าง “อันนั้น” “เมื่อกี้” “แล้วล่ะ” หมายถึงอะไร และสิ่งที่สลิดเคยตอบไปแล้ว (ไม่ต้องทักทายซ้ำหรือแนะนำตัวใหม่ทุกครั้ง) แล้วตอบเฉพาะข้อความล่าสุด"
    )
    p += CAPABILITIES_PROMPT
    p += LATENCY_PROMPT
    p += DUPLICATES_PROMPT
    if memory:
        p += MEMORY_PROMPT
    if coo:
        p += COO_PROMPT
    if freetime:
        p += FREETIME_PROMPT
    if wiki:
        p += WIKI_PROMPT
    if places:
        p += PLACES_PROMPT
    if maps:
        p += MAPS_PROMPT
    if search:
        p += SEARCH_PROMPT
    if travel:
        p += CURRENCY_PROMPT + FLIGHT_PROMPT
    if markets:
        p += MARKETS_PROMPT
    if ideas:
        p += IDEAS_PROMPT
    if contests:
        p += CONTESTS_PROMPT
    if web:
        p += WEB_PROMPT
    if secretary:
        p += SECRETARY_PROMPT
    if chat:
        p += CHAT_TOOLS_PROMPT
    return apply_particle(p, particle)


def volatile_prompt(
    settings: Settings,
    now,
    *,
    scoped: bool,
    interjection: bool,
    language: str,
    files: bool = False,
    profile: str = "",
    people: list[str] | None = None,
) -> str:
    """The per-request tail of the system prompt: date and time, the asker's scope, the
    interjection rule, the file-input rules, the asker's own preferences, the taggable people
    and the language line."""
    p = (
        f"\nวันนี้คือ {now.date().isoformat()} (วัน{THAI_DAYS[now.weekday()]}) เวลา {now:%H:%M} เขตเวลา {settings.tz}. "
        f"สัปดาห์เริ่มวันจันทร์. ใช้วันที่นี้แปลง 'เดือนนี้', 'สัปดาห์ก่อน' ฯลฯ เป็นช่วงวันที่ YYYY-MM-DD"
    )
    if scoped:
        p += "\nผู้ถามไม่ใช่ผู้บริหาร: ข้อมูลที่เห็นคือเฉพาะใบเสร็จที่ผู้ถามส่งเองเท่านั้น ให้ระบุในคำตอบว่าเป็นของผู้ถามเอง และห้ามอ้างถึงข้อมูลของคนอื่น"
    if interjection:
        p += INTERJECTION_PROMPT
    if files:
        p += FILES_PROMPT + MEDIA_PROMPT
    p += profile  # the asker's own preferences (salid.profiles.profile_block), "" when none
    if people is not None:
        p += MENTION_PROMPT
        p += "\nคนที่แท็กได้ (ชื่อที่แสดง): " + (", ".join(people) if people else "(ยังไม่มีรายชื่อ)")
    if language == "en":
        p += "\nLanguage: write your whole reply in English."
    elif language == "auto":
        p += (
            "\nLanguage: reply in the same language as the user's latest message (Thai or English)."
        )
    return apply_particle(p, particle_of(settings))


def system_prompt(
    settings: Settings,
    scoped: bool,
    now=None,
    language: str = "th",
    memory: bool = False,
    interjection: bool = False,
    files: bool = False,
    web: bool = False,
    secretary: bool = False,
    chat: bool = False,
    people: list[str] | None = None,
    coo: bool = False,
    profile: str = "",
    freetime: bool = False,
    wiki: bool = False,
    places: bool = False,
    maps: bool = False,
    search: bool = False,
    travel: bool = False,
    markets: bool = False,
    ideas: bool = False,
    contests: bool = False,
) -> str:
    """`language`: th / en answer in that language; auto = the language of the user's message.
    `chat`: the settings tools exist; `people`: display names that may be tagged (None = no
    tagging, e.g. a DM). Stable part first, per-request part last (see `stable_prompt`)."""
    now = now or settings.now()
    return stable_prompt(
        language,
        memory,
        coo=coo,
        freetime=freetime,
        wiki=wiki,
        places=places,
        maps=maps,
        search=search,
        travel=travel,
        markets=markets,
        ideas=ideas,
        contests=contests,
        web=web,
        secretary=secretary,
        chat=chat,
        particle=particle_of(settings),
    ) + volatile_prompt(
        settings,
        now,
        scoped=scoped,
        interjection=interjection,
        language=language,
        files=files,
        profile=profile,
        people=people,
    )


@dataclass
class Answer:
    text: str
    rounds: int
    seconds: float
    tool_calls: list[tuple[str, dict]]
    receipts: list[tuple[int, str | None]] = field(default_factory=list)  # (id, reaction) created
    attachments: list[tuple[str, bytes]] = field(
        default_factory=list
    )  # PDFs to upload (name, data)
    # Cost and latency (summed over every model call of the turn; 0 when not reported).
    model_calls: int = 0
    tokens_in: int = 0
    tokens_cached: int = 0
    tokens_out: int = 0
    tokens_thought: int = 0
    tool_ms: dict[str, float] = field(default_factory=dict)  # tool name -> total milliseconds


def build_tools(
    ledger: LedgerTools,
    calls: list[tuple[str, dict]] | None = None,
    memory: MemoryTools | None = None,
    timings: dict[str, float] | None = None,
) -> list:
    """The four ledger tools (plus the two memory tools when `memory` is given) as plain functions
    bound to `ledger` / `memory` (ADK wraps them as FunctionTools).

    Permission scoping lives in `ledger.forced`: none of these signatures has a submitter
    parameter, so the model cannot pass one (ADK also drops undeclared arguments). Errors are
    returned to the model as data instead of raised, so a bad date never kills the turn.
    `timings` (tool name -> milliseconds, summed) receives how long each tool took.
    """
    log = calls if calls is not None else []
    spent = timings if timings is not None else {}

    def safe(name: str, args: dict, fn):
        log.append((name, {k: v for k, v in args.items() if v is not None}))
        started = time.monotonic()
        try:
            return fn()
        except (TypeError, ValueError) as e:
            return {"error": str(e)}
        finally:
            spent[name] = spent.get(name, 0.0) + (time.monotonic() - started) * 1000

    def spending_summary(start_date: str, end_date: str, group_by: str = "none") -> dict:
        """Total spending and receipt count for a date range, optionally grouped.

        Args:
            start_date: ISO date YYYY-MM-DD, inclusive.
            end_date: ISO date YYYY-MM-DD, inclusive.
            group_by: one of category, vendor, submitter, day, none.
        """
        a = dict(start_date=start_date, end_date=end_date, group_by=group_by)
        return safe("spending_summary", a, lambda: ledger.spending_summary(**a))

    def list_receipts(
        start_date: str,
        end_date: str,
        category: str | None = None,
        vendor_contains: str | None = None,
        submitter: str | None = None,
        limit: int = 20,
        include_void: bool = False,
    ) -> dict:
        """List individual receipts in a date range with optional filters. Void (deducted)
        receipts are left out unless include_void is true; they are then marked status=void.

        Args:
            start_date: ISO date YYYY-MM-DD, inclusive.
            end_date: ISO date YYYY-MM-DD, inclusive.
            category: expense category, one of the Thai categories (e.g. วัตถุดิบ).
            vendor_contains: substring of the vendor name.
            submitter: substring of the submitter display name.
            limit: max rows, default 20.
            include_void: also list void (deducted) receipts, marked status=void.
        """
        a = dict(
            start_date=start_date,
            end_date=end_date,
            category=category,
            vendor_contains=vendor_contains,
            submitter=submitter,
            limit=limit,
            include_void=include_void,
        )
        return safe("list_receipts", a, lambda: ledger.list_receipts(**a))

    def compare_periods(
        a_start: str, a_end: str, b_start: str, b_end: str, group_by: str = "none"
    ) -> dict:
        """Compare spending of period A vs period B (diff = A - B, with percent change).

        Args:
            a_start: period A start, ISO date YYYY-MM-DD, inclusive.
            a_end: period A end, ISO date YYYY-MM-DD, inclusive.
            b_start: period B start, ISO date YYYY-MM-DD, inclusive.
            b_end: period B end, ISO date YYYY-MM-DD, inclusive.
            group_by: one of category, vendor, submitter, day, none.
        """
        a = dict(a_start=a_start, a_end=a_end, b_start=b_start, b_end=b_end, group_by=group_by)
        return safe("compare_periods", a, lambda: ledger.compare_periods(**a))

    def flagged_receipts() -> dict:
        """Receipts needing attention: low confidence, missing total, or possible duplicates."""
        return safe("flagged_receipts", {}, ledger.flagged_receipts)

    def find_duplicates(start_date: str | None = None, end_date: str | None = None) -> dict:
        """Find receipts that look like the same bill recorded twice (e.g. a PDF and a screenshot
        of one invoice): same total and currency, dates within 3 days, similar vendor or same
        invoice number. Read-only. Returns groups with which receipt would be kept, which would
        be deducted, and the amount deducted per group and in total. Does not change anything.

        Args:
            start_date: optional ISO date YYYY-MM-DD, inclusive (give both dates or neither).
            end_date: optional ISO date YYYY-MM-DD, inclusive.
        """
        a = dict(start_date=start_date, end_date=end_date)
        return safe("find_duplicates", a, lambda: ledger.find_duplicates(**a))

    def void_receipts(receipt_ids: list[int], reason: str) -> dict:
        """Deduct receipts from all totals by voiding them (soft delete, reversible with
        restore_receipts). Executives only. Call it ONLY after the user has confirmed, in a
        later message, the exact receipts to deduct. At least one receipt of each duplicate
        group stays active.

        Args:
            receipt_ids: ids of the receipts to void (the duplicates, not the one that stays).
            reason: short reason, e.g. "duplicate of #12".
        """
        a = dict(receipt_ids=receipt_ids, reason=reason)
        return safe("void_receipts", a, lambda: ledger.void_receipts(**a))

    def restore_receipts(receipt_ids: list[int]) -> dict:
        """Undo a void: count the receipts in the totals again. Executives only.

        Args:
            receipt_ids: ids of the void receipts to restore.
        """
        a = dict(receipt_ids=receipt_ids)
        return safe("restore_receipts", a, lambda: ledger.restore_receipts(**a))

    tools = [
        spending_summary,
        list_receipts,
        compare_periods,
        flagged_receipts,
        find_duplicates,
        void_receipts,
        restore_receipts,
    ]
    if memory is None:
        return tools

    def chat_days(start_date: str, end_date: str, channel_name: str | None = None) -> dict:
        """What was discussed: per-channel day summaries (overview, topics, decisions, action items,
        numbers, open questions) for a date range, in the channels the asker may see. For today it
        also returns the raw messages newer than the summary (newer_than_summary).

        Args:
            start_date: ISO date YYYY-MM-DD, inclusive.
            end_date: ISO date YYYY-MM-DD, inclusive (at most 31 days from start_date).
            channel_name: only this channel, e.g. a voice room such as General (no #), optional.
        """
        a = dict(start_date=start_date, end_date=end_date, channel_name=channel_name)
        return safe("chat_days", a, lambda: memory.chat_days(**a))

    def chat_messages(
        start_date: str,
        end_date: str,
        channel_name: str | None = None,
        keyword: str | None = None,
        limit: int = 50,
        person: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> dict:
        """Raw chat messages (newest first) with author, local time, channel, text and link.

        Args:
            start_date: ISO date YYYY-MM-DD, inclusive.
            end_date: ISO date YYYY-MM-DD, inclusive (at most 31 days from start_date).
            channel_name: only this channel (name without #), optional.
            keyword: case-insensitive text to look for in the message or author name, optional.
            limit: max messages, default 50.
            person: only messages written by this person (name or @mention), optional. Also
                finds what a person told Salid in a DM when the asker may see it.
            since: local date-time YYYY-MM-DDTHH:MM, optional: only messages at/after it (e.g.
                "last night" = yesterday T18:00 .. today T06:00; a voice meeting is in the voice
                channel's chat, kind=voice).
            until: local date-time YYYY-MM-DDTHH:MM, optional: only messages before it.
        """
        a = dict(
            start_date=start_date,
            end_date=end_date,
            channel_name=channel_name,
            keyword=keyword,
            limit=limit,
            person=person,
            since=since,
            until=until,
        )
        return safe("chat_messages", a, lambda: memory.chat_messages(**a))

    def chat_search(
        keywords: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 20,
        person: str | None = None,
        query: str | None = None,
    ) -> dict:
        """Search the team's memory by keywords AND by meaning (Thai and English), in the
        channels the asker may see. Day summaries are searched over the whole range; raw
        messages of the last 3 days too. Hits are grouped by date and channel and ranked by
        both signals (match = keyword, meaning or both), with matching summary lines and up to
        3 matching messages with links.

        Args:
            keywords: 2-5 search words: names, numbers, codes and key terms (Thai or English).
            start_date: ISO date YYYY-MM-DD, optional (default: the whole retention window).
            end_date: ISO date YYYY-MM-DD, optional (default: today).
            limit: max date/channel groups, default 20.
            person: only what this person (name or @mention) wrote or is named in, optional.
            query: the user's question in plain words, any language; improves meaning-based
                matching.
        """
        a = dict(
            keywords=keywords,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            person=person,
            query=query,
        )
        return safe("chat_search", a, lambda: memory.chat_search(**a))

    def open_file(message_link_or_id: str, file_index: int = 0) -> dict:
        """Read a document, recording or image that was shared earlier in a channel (PDF, Word,
        Excel, PowerPoint, text, audio/video transcript). Returns its full text (about 30k chars
        max). Use it after chat_search found the message, instead of answering from the short note.

        Args:
            message_link_or_id: the message's link (the `link` field of a memory result) or id.
            file_index: which attachment of that message, 0 for the first.
        """
        a = dict(message_link_or_id=message_link_or_id, file_index=file_index)
        return safe("open_file", a, lambda: memory.open_file(**a))

    return [*tools, chat_days, chat_messages, chat_search, *([open_file] if memory.opener else [])]


def build_file_tools(
    files: list[FileInput],
    actions: FileActions,
    calls: list[tuple[str, dict]],
    created: list[tuple[int, str | None]],
) -> list:
    """record_receipt / start_deep_task for the files of one message. The attachment number is
    the only model input that selects data; guild, author, channel and message are bound in
    `actions`. Receipts created are appended to `created` as (receipt id, reaction)."""
    by_index = {f.index: f for f in files}
    started: list[str] = []

    def record_receipt(file_index: int, note: str = "") -> dict:
        """Record an attachment of this message as a receipt / tax invoice / bill / bank transfer
        slip in the shop's expense ledger. Do not ask the user first. Returns the status
        (created, duplicate, ignored = not a receipt, disabled) and the saved fields.

        Args:
            file_index: the attachment number N of the file ([ไฟล์ #N: ...]).
            note: optional extra context from the user (what the expense was for).
        """
        calls.append(("record_receipt", {"file_index": file_index}))
        if actions.record_receipt is None:
            return {"status": "disabled"}
        f = by_index.get(file_index)
        if f is None:
            return {"status": "error", "error": f"no file #{file_index} in this message"}
        if f.mime not in _RECEIPT_MIMES and f.mime not in ("application/pdf", "image/gif"):
            return {"status": "ignored", "reason": "only an image or PDF can be a receipt"}
        try:
            res = actions.record_receipt(f, note or "")
        except Exception as e:  # a failed extraction must not kill the turn
            logger.warning("record_receipt failed (%s)", type(e).__name__)
            return {"status": "error", "error": "could not read this file as a receipt"}
        out: dict = {"status": res.status}
        r = res.receipt
        if r is not None:
            out.update(
                receipt_id=r.id,
                vendor=r.vendor,
                total=r.total,
                currency=r.currency or "THB",
                total_fmt=money(r.total, r.currency) if r.total is not None else None,
                date=r.receipt_date or r.effective_date,
                category=r.category,
                needs_check=res.reaction == "⚠️",
            )
        if res.reply:
            out["reply_text"] = res.reply
        if res.status == "created" and r is not None:
            created.append((r.id, res.reaction))
        return out

    def start_deep_task(task: str) -> dict:
        """Start a deep background analysis of the shop's ledger (executives only; the report is
        posted in the channel later). Use only for a heavy analysis the file calls for, then tell
        the user it has started.

        Args:
            task: what to analyse, in the user's language, including the relevant facts from the file.
        """
        calls.append(("start_deep_task", {}))
        if actions.start_deep_task is None:
            return {"status": "disabled"}
        if started:
            return {"status": "already_started"}
        status = actions.start_deep_task(task)
        if status == "started":
            started.append(task)
        return {"status": status}

    return [record_receipt, start_deep_task]


def _ensure_vertex_env(settings: Settings) -> None:
    # ADK reads these to route Gemini calls through Vertex AI (meeting_scribe sets them too).
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", settings.gcp_project)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", settings.gemini_location)


def build_agent(
    settings: Settings,
    tools: list,
    *,
    scoped: bool,
    now=None,
    model=None,
    on_model_call=None,
    language: str = "th",
    memory: bool = False,
    interjection: bool = False,
    files: bool = False,
    web: bool = False,
    secretary: bool = False,
    chat: bool = False,
    people: list[str] | None = None,
    coo: bool = False,
    profile: str = "",
    freetime: bool = False,
    wiki: bool = False,
    thinking_level: str = "LOW",
    places: bool = False,
    maps: bool = False,
    search: bool = False,
    travel: bool = False,
    markets: bool = False,
    ideas: bool = False,
    contests: bool = False,
) -> LlmAgent:
    """One ADK LlmAgent per request. `model` is injectable so tests stay offline."""
    _ensure_vertex_env(settings)
    prompt = system_prompt(
        settings,
        scoped=scoped,
        now=now,
        language=language,
        memory=memory,
        interjection=interjection,
        files=files,
        web=web,
        secretary=secretary,
        chat=chat,
        people=people,
        coo=coo,
        profile=profile,
        freetime=freetime,
        wiki=wiki,
        places=places,
        maps=maps,
        search=search,
        travel=travel,
        markets=markets,
        ideas=ideas,
        contests=contests,
    )
    model_calls = 0

    def before_model(callback_context, llm_request):
        # Hard cap: after MAX_ROUNDS tool rounds the next call has no tools, so it must answer.
        nonlocal model_calls
        model_calls += 1
        if on_model_call:
            on_model_call(model_calls)
        if model_calls > MAX_ROUNDS:
            # No tool declarations, but tools_dict stays: a model that still emits a call (its
            # history is full of them) would otherwise crash ADK with "Tool 'x' not found.
            # Available tools: " (empty). before_tool answers such a call without running it.
            llm_request.config.tools = None
            llm_request.append_instructions([ROUNDS_EXHAUSTED])
        return None

    def before_tool(tool, args, tool_context):
        if model_calls > MAX_ROUNDS:
            return {"error": ROUNDS_EXHAUSTED}  # skip the call; the model must now just answer
        return None

    return LlmAgent(
        name="salid_qa",
        model=model
        or Gemini(
            model=settings.gemini_model,
            client_kwargs={"location": settings.gemini_location},
            retry_options=RETRY_OPTIONS,
        ),
        description="Chats with the team and answers expense questions from the ledger.",
        # A callable skips ADK's {placeholder} templating of the instruction string.
        instruction=lambda _ctx: prompt,
        tools=tools,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.6, thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
        ),
        before_model_callback=before_model,
        before_tool_callback=before_tool,
    )


SPOKEN_THINKING = "MINIMAL"  # ThinkingLevel for answers that are read aloud in a voice room


def spoken_hint(particle: str = "") -> str:
    """Hint appended to a question whose answer will be spoken. `particle`: the voice's
    politeness particle (ค่ะ for the female voice; questions use คะ)."""
    hint = SPOKEN_HINT
    if particle:
        q = "คะ" if particle == "ค่ะ" else particle
        hint += (
            f"\n(เสียงพูดเป็นผู้หญิง ลงท้ายประโยคด้วย “{particle}” เสมอ ประโยคคำถามใช้ “{q}” "
            "ห้ามใช้ ครับ/นะครับ ปนกัน / end spoken sentences with the voice particle)"
        )
    return hint


SPOKEN_HINT = (
    "(คำตอบนี้จะถูกพูดออกเสียงในห้องเสียง: ตอบ 1-3 ประโยคสั้น ๆ พูดได้ลื่น ไม่ใช้รายการหรือ markdown / "
    "answer in 1-3 short sentences suitable for speaking, no lists or markdown)"
)


_FILE_REF_RE = re.compile(
    r"ไฟล์|เอกสาร|รูป|ภาพ|แนบ|pdf|docx?|\bfile\b|\bimage\b|\bphoto\b|\bdocument\b|ใบเสร็จ|สลิป",
    re.I,
)
_MEMORY_REF_RE = re.compile(
    r"เมื่อวาน|เมื่อกี้|ก่อนหน้า|ที่ผ่านมา|ในห้อง|ในแชต|ใครพูด|ใครบอก|คุยกัน|ประชุม|เคย|ค้น|หา(?:ใน)?"
    r"|chat|yesterday|earlier|previous|last\s+(?:week|time)|who\s+(?:said|told)|search|find|meeting",
    re.I,
)


def is_file_only_question(question: str) -> bool:
    """The message is just attachments, or a request about them that does not mention the chat's
    history ("สรุปไฟล์นี้ให้หน่อย"): the chat-memory tools would only cost a model round."""
    q = (question or "").strip()
    if not q or q == NO_TEXT_TURN:
        return True
    return bool(_FILE_REF_RE.search(q)) and not _MEMORY_REF_RE.search(q)


async def ask_async(
    settings: Settings,
    store: Store,
    question: str,
    *,
    guild_id: int,
    is_exec: bool,
    user_id: int,
    history: str = "",
    now=None,
    model=None,
    language: str = "th",
    memory: MemoryTools | None = None,
    interjection: bool = False,
    files: list[FileInput] | None = None,
    actions: FileActions | None = None,
    web_client=None,
    secretary: SecretaryContext | None = None,
    chat_actions: ChatActions | None = None,
    people: list[str] | None = None,
    extra_tools: list | None = None,
    profile: str = "",
    spoken: bool = False,
    tool_overrides: list | None = None,
) -> Answer:
    """Answer a question with an ADK agent. `history` is recent chat text, context only. Scope is decided here in code, before any data
    reaches the model: tools are built per request around a LedgerTools bound to `guild_id`
    whose `forced_submitter_id` is the asker for non-executives. `files`: the message's attachments
    (read by the model itself as multimodal parts, within the size limits); with `actions` the model
    also gets record_receipt / start_deep_task for them. `web_client` (a genai client): registers
    read_web (links + Google Search; see salid.web_read), fed only by `question` and the model's
    short request. `extra_tools`: the coordinator tools (see salid.coo_tools), registered with
    their prompt section."""
    started = time.monotonic()
    calls: list[tuple[str, dict]] = []
    tool_ms: dict[str, float] = {}
    model_calls = 0

    def count_call(n: int) -> None:
        nonlocal model_calls
        model_calls = n

    created: list[tuple[int, str | None]] = []
    ledger = LedgerTools(
        store,
        guild_id,
        forced_submitter_id=None if is_exec else user_id,
        actor=f"user {user_id}",
    )
    usable, skipped = select_files(list(files or []))
    f_parts, unreadable = file_parts(usable)
    usable = [f for f in usable if f not in unreadable]
    skipped += unreadable
    if memory is not None and usable and is_file_only_question(question):
        memory = None  # the question is about the attached file only: no chat-memory tools
    tools = build_tools(ledger, calls, memory, tool_ms)
    if usable and actions is not None:
        tools += build_file_tools(usable, actions, calls, created)
    if web_client is not None:
        tools.append(make_read_web(web_client, settings.gemini_model, question, calls))
    if secretary is not None:
        secretary.ledger = ledger  # receipt scoping for the PDFs comes from the asker's ledger
    secretary_tools = (
        build_secretary_tools(SecretaryTools(secretary), calls) if secretary is not None else []
    )
    tools += secretary_tools
    chat_tools = build_chat_tools(chat_actions, calls) if chat_actions is not None else []
    tools += chat_tools
    tools += extra_tools or []
    if tool_overrides:  # same-named tools bound to someone else (voice: the speaker's actions)
        swap = {getattr(f, "__name__", ""): f for f in tool_overrides}
        tools = [swap.get(getattr(f, "__name__", ""), f) for f in tools]
    tool_names = {getattr(f, "__name__", "") for f in tools}  # prompt sections follow the tools
    agent = build_agent(
        settings,
        tools,
        scoped=not is_exec,
        now=now,
        model=model,
        on_model_call=count_call,
        language=language,
        memory=memory is not None,
        interjection=interjection,
        files=bool(usable or skipped),
        web=web_client is not None,
        secretary=bool(secretary_tools),
        chat=bool(chat_tools),
        people=people,
        coo="list_tasks" in tool_names,
        profile=profile,
        freetime="find_free_time" in tool_names,
        wiki="wiki_search" in tool_names,
        # Spoken answers sit between a person and the sound of silence: think as little as possible.
        thinking_level=SPOKEN_THINKING if spoken else "LOW",
        places="find_place" in tool_names,
        maps="travel_time" in tool_names or "weather" in tool_names,
        search="search_everything" in tool_names,
        travel="convert_currency" in tool_names,
        markets="stock_quote" in tool_names,
        ideas="idea_save" in tool_names,
        contests="contest_add" in tool_names,
    )
    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    adk_user, session_id = str(user_id), uuid.uuid4().hex
    await runner.session_service.create_session(
        app_name=APP_NAME, user_id=adk_user, session_id=session_id
    )
    if files and not question.strip():
        question = NO_TEXT_TURN
    text_in = (
        f"บทสนทนาก่อนหน้าในห้องนี้ (หลายคนคุยกัน เรียงจากเก่าไปใหม่):\n{history}\n\n"
        f"ข้อความล่าสุดที่ต้องตอบ:\n{question}"
        if history
        else question
    )
    if skipped:
        text_in += "\n" + skipped_note(skipped)
    if spoken:
        text_in += "\n" + spoken_hint(str(getattr(settings, "voice_particle", "") or ""))
    message = types.Content(role="user", parts=[types.Part(text=text_in), *f_parts])
    rounds, text = 0, ""
    first_call_s = last_call_s = 0.0
    tokens = dict.fromkeys(USAGE_FIELDS, 0)
    async for event in runner.run_async(
        user_id=adk_user,
        session_id=session_id,
        new_message=message,
        run_config=RunConfig(max_llm_calls=MAX_ROUNDS + 2),
    ):
        if event.usage_metadata is not None:
            for name, attr in USAGE_FIELDS.items():
                tokens[name] += getattr(event.usage_metadata, attr, None) or 0
        if event.get_function_calls():
            rounds += 1
            last_call_s = time.monotonic() - started
            first_call_s = first_call_s or last_call_s
        elif event.is_final_response() and event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts if not p.thought).strip()
    total_s = time.monotonic() - started
    if spoken:  # timing only, no content
        logger.info(
            "qa spoken timing: first tool call %.1fs, last tool call %.1fs, total %.1fs, "
            "%d tool rounds",
            first_call_s,
            last_call_s,
            total_s,
            rounds,
        )
    if not text and interjection:
        text = NO_REPLY  # an unprompted answer that failed is never worth posting
    return Answer(
        text or t("qa_failed", resolve_lang(language, question)),
        rounds,
        time.monotonic() - started,
        calls,
        created,
        [*(secretary.attachments if secretary is not None else []), *tool_files(tools)],
        model_calls=model_calls,
        tool_ms=tool_ms,
        **tokens,
    )


def ask(
    settings: Settings,
    store: Store,
    question: str,
    *,
    guild_id: int,
    is_exec: bool,
    user_id: int,
    history: str = "",
    now=None,
    model=None,
    language: str = "th",
    memory: MemoryTools | None = None,
    interjection: bool = False,
    files: list[FileInput] | None = None,
    actions: FileActions | None = None,
    web_client=None,
    secretary: SecretaryContext | None = None,
    chat_actions: ChatActions | None = None,
    people: list[str] | None = None,
    extra_tools: list | None = None,
    profile: str = "",
    spoken: bool = False,
    tool_overrides: list | None = None,
) -> Answer:
    """Blocking wrapper around `ask_async` (call from a worker thread, e.g. asyncio.to_thread)."""
    return asyncio.run(
        ask_async(
            settings,
            store,
            question,
            guild_id=guild_id,
            is_exec=is_exec,
            user_id=user_id,
            history=history,
            now=now,
            model=model,
            language=language,
            memory=memory,
            interjection=interjection,
            files=files,
            actions=actions,
            web_client=web_client,
            secretary=secretary,
            chat_actions=chat_actions,
            people=people,
            extra_tools=extra_tools,
            profile=profile,
            spoken=spoken,
            tool_overrides=tool_overrides,
        )
    )
