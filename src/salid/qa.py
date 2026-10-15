from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.models import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

from .config import Settings
from .i18n import resolve_lang, t
from .memory_access import Visibility, hidden_but_askable, is_dm_scope, message_visible
from .memory_store import MemoryMessage, MemoryStore
from .memory_summary import render_summary
from .store import Store, duplicate_extras, is_active, pick_keep
from .web_read import make_read_web

logger = logging.getLogger(__name__)
MAX_ROUNDS = 4
APP_NAME = "salid"
# Vertex 429s (shared quota) and 5xx are retried with backoff by the genai HTTP client
# (same policy as meeting_scribe.agents).
RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=5, initial_delay=2.0, max_delay=30.0, http_status_codes=[429, 500, 502, 503, 504]
)
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
        visibility: Visibility | None = None,
    ):
        self.store, self.guild_id, self.allowed, self.s = store, guild_id, dict(allowed), settings
        self.retention_days = retention_days or settings.memory_retention_days
        # With `visibility` the role hierarchy and disclosure grants apply (salid.memory_access);
        # `hidden` collects matching records the asker may not see (never shown, only counted, so
        # Salid can offer a disclosure request).
        self.vis = visibility
        self.hidden: list[MemoryMessage] = []
        self.hidden_in_dm = 0  # of those: would be visible if asked in a DM

    def _range(self, start_date: str, end_date: str) -> tuple[str, str]:
        a, b = date.fromisoformat(start_date), date.fromisoformat(end_date)
        if b < a:
            raise ValueError("end_date is before start_date")
        if (b - a).days + 1 > MAX_CHAT_DAYS:
            raise ValueError(f"range too long: at most {MAX_CHAT_DAYS} days per call")
        return a.isoformat(), b.isoformat()

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
            f"{n} more matching record(s) exist that this person is NOT allowed to see (content "
            "withheld; never guess it). Tell them you cannot share that yet and offer to ask the "
            f"owner for permission with request_disclosure.{hint}"
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

    def chat_days(self, start_date: str, end_date: str) -> dict:
        start, end = self._range(start_date, end_date)
        sums = [
            x
            for x in self.store.summaries(self.guild_id, self._shown(), start, end)
            if x.message_count > 0
        ]
        # Messages newer than the (living) summary, so nothing between updates is missed.
        today = self.s.now().date()
        tail: list[dict] = []
        covered = {(x.channel_id, x.local_date): x for x in sums}
        for day in (today - timedelta(days=1), today):
            d = day.isoformat()
            if not start <= d <= end:
                continue
            for cid in self.store.day_channels(self.guild_id, d):
                if cid not in self._shown():
                    continue
                x = covered.get((cid, d))
                if x is not None and x.final:
                    continue
                rows = self.store.messages_after(
                    self.guild_id, cid, d, x.cursor if x else None, TAIL_PER_CHANNEL
                )
                tail += [self._msg(m) for m in rows if self._visible(m)]
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

    @staticmethod
    def _summary_lines(summary: dict) -> list[str]:
        """Searchable lines of a summary: topics, decisions, action items, numbers, the rest."""
        lines = [summary.get("overview") or ""]
        for tp in summary.get("topics") or []:
            people = ", ".join(tp.get("people") or [])
            lines.append(f"Topic: {tp.get('title', '')} {tp.get('details', '')} {people}".strip())
        lines += [f"Decision: {x}" for x in summary.get("decisions") or []]
        for a in summary.get("action_items") or []:
            lines.append(f"Action: {a.get('who', '')} {a.get('what', '')} {a.get('due', '')}")
        lines += [f"Number: {x}" for x in summary.get("numbers") or []]
        lines += [f"Open: {x}" for x in summary.get("open_questions") or []]
        return [x.strip() for x in lines if x and x.strip()]

    def chat_search(
        self,
        keywords: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 20,
        person: str | None = None,
    ) -> dict:
        kws = [k.strip().casefold() for k in (keywords or []) if isinstance(k, str) and k.strip()]
        kws = list(dict.fromkeys(kws))[:12]
        if not kws:
            return {"error": "give at least one keyword"}
        today = self.s.now().date()
        end = date.fromisoformat(end_date) if end_date else today
        start = (
            date.fromisoformat(start_date)
            if start_date
            else end - timedelta(days=max(self.retention_days, 1) - 1)
        )
        if end < start:
            raise ValueError("end_date is before start_date")
        start = max(start, end - timedelta(days=MAX_SEARCH_DAYS - 1))
        a, b = start.isoformat(), end.isoformat()
        pids, ptext = self._person_ids(person)

        def hits(text: str) -> int:
            low = text.casefold()
            return sum(1 for k in kws if k in low)

        groups: dict[tuple[str, int], dict] = {}

        def group(day: str, cid: int) -> dict:
            return groups.setdefault(
                (day, cid), {"score": 0, "summary": [], "messages": [], "n_msgs": 0}
            )

        for x in self.store.summaries(self.guild_id, self._shown(), a, b):
            for line in self._summary_lines(x.summary):
                h = hits(line)
                if h and (not ptext or ptext in line.casefold()):
                    g = group(x.local_date, x.channel_id)
                    g["score"] += h * SUMMARY_WEIGHT
                    g["summary"].append(line[:300])
        rows = self.store.messages(
            self.guild_id, self._read_ids(), a, b, limit=MAX_ROWS_SEARCH, newest_first=True
        )
        for m in rows:  # newest first
            h = hits(f"{m.text} {m.author_name}")
            if h and self._by_person(m, pids, ptext):
                if not self._visible(m):
                    self._hide(m)
                    continue
                g = group(m.local_date, m.channel_id)
                g["score"] += h
                g["n_msgs"] += 1
                if len(g["messages"]) < SEARCH_MSGS_PER_GROUP:
                    g["messages"].append(self._msg(m))
        ranked = sorted(groups.items(), key=lambda kv: (kv[1]["score"], kv[0][0]), reverse=True)
        items = [
            {
                "date": day,
                "channel": self._name(cid),
                "score": g["score"],
                "summary_hits": g["summary"][:6],
                "matching_messages": g["n_msgs"],
                "messages": g["messages"],
            }
            for (day, cid), g in ranked[: max(1, min(int(limit), 50))]
        ]
        fit, cut = self._fit(items, SEARCH_CHARS)
        out: dict = {"keywords": kws, "range": [a, b], "count": len(fit), "hits": fit}
        if cut:
            out["note"] = "output truncated: more hits exist, use narrower keywords or dates"
        if not fit:
            out["note"] = "no matches in the channels you can see"
        if self._hidden_note():
            out["withheld"] = self._hidden_note()
        return out

    def chat_messages(
        self,
        start_date: str,
        end_date: str,
        channel_name: str | None = None,
        keyword: str | None = None,
        limit: int = 50,
        person: str | None = None,
    ) -> dict:
        start, end = self._range(start_date, end_date)
        pids, ptext = self._person_ids(person)
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
        if self.vis is None and not ptext:
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
            if self._visible(m):
                shown.append(m)
            elif keyword or ptext:
                self._hide(m)
        out: dict = {"count": len(shown[:limit]), "messages": [self._msg(m) for m in shown[:limit]]}
        if self._hidden_note():
            out["withheld"] = self._hidden_note()
        return out

    def call(self, name: str, args: dict) -> dict:
        if name not in {"chat_days", "chat_messages", "chat_search"}:
            return {"error": f"unknown tool {name}"}
        fn = getattr(self, name)
        params = fn.__code__.co_varnames[1 : fn.__code__.co_argcount]
        clean = {k: v for k, v in (args or {}).items() if k in params and v is not None}
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
        label = f"[ไฟล์ #{f.index}: {f.filename}]"
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
        "คุณชื่อ “สลิด” เป็นผู้ช่วยผู้บริหารและผู้ประสานงานของทีม (มือขวาของ COO/PM: ดูว่าใครทำอะไรอยู่ "
        "ติดตามงาน รายงาน นัดหมาย และรักษาความลับตามลำดับชั้นของทีม) และเป็นเพื่อนร่วมทีมในแชต Discord "
        "นิสัยเป็นกันเอง สุภาพ อารมณ์ดี พูดภาษาไทยเป็นธรรมชาติ ลงท้ายด้วยคำว่าครับ "
        "ตอบตามภาษาที่ผู้ใช้พิมพ์มา และปรับความยาวตามคำถาม (คุยเล่นตอบสั้น งานที่ขอรายละเอียดตอบละเอียดได้)\n"
    ),
    "en": (
        "Your name is “Salid” (สลิด). You are the team's executive assistant and coordinator (a "
        "COO/PM's right hand: you keep track of who is doing what, follow up on work, report, "
        "schedule, and keep confidential information according to the team's role hierarchy) and a "
        "teammate in the team's Discord chat. You are friendly, polite and upbeat. ALWAYS reply in English, even if the "
        "user writes in Thai (keep Thai names, vendors and category names as they are). "
        "Adjust the length to the question (short for small talk, detailed when the user asks "
        "for detail)\n"
    ),
    "auto": (
        "คุณชื่อ “สลิด” (Salid) เป็นผู้ช่วยผู้บริหารและผู้ประสานงานของทีม (ติดตามว่าใครทำอะไร ติดตามงาน "
        "รายงาน นัดหมาย และรักษาความลับตามลำดับชั้น / executive assistant and coordinator) "
        "และเป็นเพื่อนร่วมทีมในแชต Discord "
        "นิสัยเป็นกันเอง สุภาพ อารมณ์ดี ตอบด้วยภาษาเดียวกับข้อความล่าสุดของผู้ใช้ "
        "(reply in the same language as the user's latest message): ถ้าผู้ใช้พิมพ์ไทยให้ตอบไทยเป็นธรรมชาติ "
        "ลงท้ายด้วยคำว่าครับ ถ้าพิมพ์อังกฤษให้ตอบเป็นอังกฤษ "
        "และปรับความยาวตามคำถาม (คุยเล่นตอบสั้น งานที่ขอรายละเอียดตอบละเอียดได้)\n"
    ),
}


MEMORY_PROMPT = (
    "\nความจำของทีม: เมื่อถูกถามว่าเกิดอะไรขึ้น ใครพูดอะไร หรืองาน/เรื่องใดถึงไหนแล้ว "
    "ถ้าคำถามไม่มีวันที่ชัดเจน ให้เรียก chat_search ก่อนด้วยคำค้น 3-8 คำ ทั้งไทย อังกฤษ คำพ้อง ชื่อคน และตัวย่อ "
    "(เช่น งบโฆษณา, ค่าโฆษณา, ads, marketing budget) แล้วค่อยเปิดวันที่เจอด้วย chat_days หรือ chat_messages. "
    "ถ้าคำถามมีวันที่ ให้เรียก chat_days (สรุปรายวันต่อห้อง) ได้เลย "
    "แล้วใช้ chat_messages (ข้อความจริง ค้นด้วย keyword) เมื่อต้องการรายละเอียดหรือคำพูดจริง. "
    "ข้อความที่ขึ้นต้นด้วย [รูป: ...] คือสิ่งที่อ่านได้จากรูปที่มีคนส่ง (รวมตัวหนังสือในรูป). "
    "ตอบโดยระบุห้องและวันที่ที่มา และถ้าอ้างข้อความจริงให้แนบลิงก์ (link). "
    "ถ้าไม่พบข้อมูลให้บอกตรงๆ ว่าไม่พบ ห้ามเดา และห้ามพูดถึงห้องที่เครื่องมือไม่ได้คืนมา. "
    "แปลงคำอย่าง 'เมื่อวาน' หรือ 'สัปดาห์ที่แล้ว' เป็นช่วงวันที่ YYYY-MM-DD จากวันนี้ในเขตเวลาที่ระบุ "
    "(chat_days/chat_messages รับช่วงละไม่เกิน 31 วัน ผลลัพธ์ถูกตัดความยาว ถ้ามี note ให้ย่อช่วงหรือใช้ chat_search). ผลลัพธ์ newer_than_summary คือข้อความของวันนี้ที่ใหม่กว่าสรุป."
)


# The coordinator role: hierarchy, tasks, disclosure, DMs, reports. Added when the coordinator tools
# are registered (see `build_agent(coo=True)`).
COO_PROMPT = (
    "\nบทบาทผู้ประสานงานของทีม (COO/PM): สลิดดูความเคลื่อนไหวของทุกคนเอง ไม่ต้องรอให้ใครมารายงาน "
    "จดงานที่ใครรับปากหรือถูกมอบหมายเป็นรายการงาน ติดตามทุกเช้า (DM ถามความคืบหน้า) และส่งรายงานเย็นให้ผู้บริหาร "
    "(ห้อง “รายงานจากสลิด”, /รายงาน หรือ /report). ใช้ list_tasks ดูงาน, update_task ปิดงานหรือเปิดใหม่ "
    "(สมาชิกแก้ได้เฉพาะงานของตัวเอง ผู้บริหารแก้ได้ทุกงาน) เมื่อมีคนบอกว่าทำงานไหนเสร็จหรือติดอะไร "
    "ให้ list_tasks ของเขาแล้ว update_task พร้อมโน้ตสั้นๆ และถามถึงงานที่ค้างหรือเลยกำหนด. "
    "ถามว่า “วันนี้ผมทำอะไรไปบ้าง” ให้เรียก my_day. ถามถึงงานของคนใดคนหนึ่งให้ใส่ person ใน chat_messages / chat_search.\n"
    "ความลับตามลำดับชั้น: เห็นข้อมูลเฉพาะที่เครื่องมือคืนมา ผู้ที่ตำแหน่ง (role) สูงกว่าเห็นงานของผู้ที่ต่ำกว่าได้ "
    "แต่ผู้ที่ต่ำกว่าหรือเท่ากันไม่เห็นงานของผู้ที่สูงกว่า และห้องส่วนตัวไม่เปิดให้คนนอกห้อง ข้อมูลที่พูดในห้องสาธารณะบอกใครก็ได้. "
    "ถ้าผลลัพธ์มี withheld แปลว่ามีข้อมูลที่เกี่ยวข้องแต่ผู้ถามยังไม่มีสิทธิ์เห็น: บอกตรงๆ อย่างสุภาพว่าตอนนี้ยังบอกไม่ได้ "
    "(ห้ามบอกเนื้อหา ห้ามเดา ห้ามบอกว่าใครเป็นเจ้าของ) และเสนอว่า “ให้สลิดไปขออนุญาตเจ้าของข้อมูลไหมครับ” "
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
    "\nสิ่งที่สลิดทำได้ (ใช้ตอบเมื่อมีคนถามว่าทำอะไรได้บ้าง หรือถามวิธีใช้ ตอบให้กระชับ เลือกเฉพาะที่เกี่ยวข้อง "
    "อย่าอ้างว่าทำได้มากกว่านี้):\n"
    "1) ใบเสร็จและบัญชี: ส่งรูปใบเสร็จในห้องที่ชื่อมีคำว่า ใบเสร็จ/receipt/invoice/ใบกำกับ/สลิป/slip/บิล/bill/expense/ค่าใช้จ่าย/เบิก หรือแท็กสลิดพร้อมรูป สลิดอ่านร้าน ยอด วันที่ หมวด "
    "แล้วบันทึกลงบัญชี เตือนถ้าน่าจะซ้ำ แก้ข้อมูลได้โดยตอบกลับ (reply) ข้อความใบเสร็จของสลิด. "
    "ถามยอดใช้จ่ายได้ เช่น เดือนนี้จ่ายค่าอะไรไปเท่าไร แยกหมวด ใครส่ง รายการที่ต้องตรวจ "
    "(ผู้บริหารเห็นของทั้งทีม คนอื่นเห็นเฉพาะของตัวเอง). "
    "ตรวจหาบิลที่ส่งซ้ำ (เช่น PDF กับภาพหน้าจอของใบเดียวกัน) ได้ และหักออกจากยอดได้ (void ไม่ลบทิ้ง กู้คืนได้ ผู้บริหารสั่งหัก); "
    "มีสรุปรายวันส่งเข้าห้องตามเวลาที่ตั้งไว้\n"
    "2) ความจำของทีม: สลิดจดข้อความและอ่านรูปทุกรูป (รวมตัวหนังสือในรูป) ในห้องที่ไม่ได้ตั้งเป็น 🔇 ปิด "
    "เก็บตามจำนวนวันที่แอดมินตั้ง (ค่าเริ่มต้น 90 วัน ดูหรือปรับด้วย /ความจำ หรือ /memory) "
    "และสรุปแต่ละห้องเป็นรายวัน จึงตอบได้ว่าเมื่อวานคุยอะไร ใครรับงานอะไร เรื่องไหนถึงไหน พร้อมลิงก์ไปที่ข้อความจริง. "
    "ตอบได้เฉพาะข้อมูลจากห้องที่ผู้ถามและทุกคนในห้องที่ตอบมีสิทธิ์เห็น ถามทาง DM ได้ข้อมูลตามสิทธิ์ของผู้ถามคนเดียว. "
    "ไม่เห็นห้องที่สลิดไม่มีสิทธิ์เข้า และไม่เห็น DM ระหว่างคนกับคน\n"
    "3) ประชุมเสียง: /เข้าประชุม หรือ /join (หรือแท็กแล้วพิมพ์ เข้าประชุม / join meeting) สลิดเข้าห้องเสียงที่ผู้สั่งอยู่ "
    "ถอดเสียงว่าใครพูดอะไร (ไทยและอังกฤษ ไม่เก็บไฟล์เสียง) เรียกชื่อ “สลิด” ระหว่างประชุมเพื่อถามได้ "
    "และ /ออกประชุม หรือ /leave แล้วสลิดส่งสรุปประชุม (หัวข้อ สิ่งที่ตกลง งานที่ต้องทำ) เข้าห้องแชต\n"
    "4) โหมดของแต่ละห้อง (/โหมด หรือ /mode, แอดมินหรือผู้บริหารเปลี่ยนได้): "
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
    "ผู้ประสานงานของทีม: สลิดดูเองว่าใครทำอะไร จดงานที่ใครรับปาก ติดตามความคืบหน้าทาง DM ทุกเช้า (ปิดได้ใน Hub) "
    "ส่งรายงานเย็นเข้าห้องส่วนตัว “รายงานจากสลิด” สำหรับเจ้าของ/แอดมิน/ผู้บริหาร (หรือสั่งเองด้วย /รายงาน หรือ /report) "
    "ใครก็ DM มาบอกเรื่องงานได้ (ผู้ที่ตำแหน่งสูงกว่าเห็น เว้นแต่บอกว่า “เรื่องนี้เก็บเป็นความลับ”) "
    "และถามว่าวันนี้ตัวเองทำอะไรไปบ้าง (/รายงาน ของฉัน) ได้ ข้อมูลเปิดตามลำดับ role: คนตำแหน่งสูงกว่าเห็นงานของคนต่ำกว่า "
    "ข้อมูลที่ยังไม่มีสิทธิ์เห็น สลิดขออนุญาตเจ้าของให้ได้\n"
    "ข้อจำกัด: ตั้งค่าแต่ละเซิร์ฟเวอร์/ห้องอาจปิดบางความสามารถไว้ (เช่น การค้นเว็บ) "
    "ถ้าไม่แน่ใจให้แนะนำ /โหมด /ภาษา /ความจำ เพื่อดูสถานะ"
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
    "เอกสาร ภาพหน้าจอ แคปแชต ตาราง หรือกราฟ -> สรุปประเด็นสำคัญ เชื่อมกับความจำของทีมถ้าเกี่ยวข้อง (chat_search) "
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


WEB_PROMPT = (
    "\nอ่านลิงก์และค้นอินเทอร์เน็ต: เมื่อข้อความมีลิงก์ (เว็บ, YouTube) หรือถามข้อมูลล่าสุด/ข่าว/ราคา/เรื่องนอกทีมที่ต้องค้นออนไลน์ "
    "ให้เรียก read_web (ลิงก์ในข้อความถูกแนบให้อัตโนมัติ ใส่ใน request แค่สิ่งที่ต้องการรู้ ห้ามใส่ข้อมูลบัญชี ความจำ หรือแชตของทีม). "
    "ในคำตอบ: บอกว่าแต่ละลิงก์คืออะไร สรุปประเด็นสำคัญ เสนอขั้นตอนต่อไปถ้ามีประโยชน์ และอ้างแหล่งที่มาเป็นลิงก์. "
    "ถ้าลิงก์อ่านไม่ได้ (ต้องล็อกอิน เพย์วอล์ล ไม่พบหน้า) ให้บอกตรงๆ ห้ามเดาเนื้อหา. "
    "คำถามเรื่องภายในทีม (บัญชี ใบเสร็จ ใครพูดอะไร) ให้ใช้ ledger/ความจำก่อน ไม่ต้องค้นเว็บ."
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


def system_prompt(
    settings: Settings,
    scoped: bool,
    now=None,
    language: str = "th",
    memory: bool = False,
    interjection: bool = False,
    files: bool = False,
    web: bool = False,
    coo: bool = False,
) -> str:
    """`language`: th / en answer in that language; auto = the language of the user's message."""
    now = now or settings.now()
    p = (
        PERSONA[language if language in PERSONA else "th"]
        + "คุยได้ทุกเรื่องเหมือนผู้ช่วย AI ทั่วไป: คุยเล่น ทักทาย ให้กำลังใจ ตอบความรู้ทั่วไป "
        "ช่วยคิดไอเดีย ร่างข้อความ แปลภาษา สรุปหรืออธิบายเรื่องต่างๆ "
        "ถ้าเป็นข่าวหรือข้อมูลล่าสุดที่อาจไม่รู้ ให้บอกตรงๆ ว่าอาจไม่อัปเดต\n"
        f"วันนี้คือ {now.date().isoformat()} (วัน{THAI_DAYS[now.weekday()]}) เวลา {now:%H:%M} เขตเวลา {settings.tz}. "
        f"สัปดาห์เริ่มวันจันทร์. ใช้วันที่นี้แปลง 'เดือนนี้', 'สัปดาห์ก่อน' ฯลฯ เป็นช่วงวันที่ YYYY-MM-DD\n"
        "กฎเหล็กเรื่องบัญชีของร้าน: ห้ามคำนวณตัวเลขของร้านเอง (บวก ลบ เปอร์เซ็นต์) ให้เรียกเครื่องมือและใช้ตัวเลขที่เครื่องมือคืนมาเท่านั้น "
        "(ใช้ฟิลด์ *_fmt ได้เลย) ตัวเลขใส่ , คั่นหลักพัน และนำหน้าด้วยสัญลักษณ์สกุลเงินของใบเสร็จนั้น (฿ = บาท, $ = USD). "
        "ห้ามบวกหรือรวมเงินต่างสกุลเข้าด้วยกันเด็ดขาด และห้ามแปลงอัตราแลกเปลี่ยน: ถ้ามีหลายสกุล (totals_by_currency) ให้รายงานแยกทีละสกุล เช่น ฿184,885.54 และ $131.99. ใบเสร็จที่ไม่ระบุสกุลเงินนับเป็นบาท. ถ้าไม่มีข้อมูลให้บอกตรงๆ ห้ามเดาหรือแต่งข้อมูลของร้าน. "
        "เรียกเครื่องมือหลายตัวพร้อมกันได้เพื่อความเร็ว. ถ้ามีบทสนทนาก่อนหน้าแนบมา ให้อ่านเหมือนคนที่นั่งอยู่ในห้องแชต: เข้าใจว่าใครพูดกับใคร เรื่องที่กำลังคุยค้างอยู่ คำอย่าง “อันนั้น” “เมื่อกี้” “แล้วล่ะ” หมายถึงอะไร และสิ่งที่สลิดเคยตอบไปแล้ว (ไม่ต้องทักทายซ้ำหรือแนะนำตัวใหม่ทุกครั้ง) แล้วตอบเฉพาะข้อความล่าสุด"
    )
    p += CAPABILITIES_PROMPT
    p += DUPLICATES_PROMPT
    if scoped:
        p += "\nผู้ถามไม่ใช่ผู้บริหาร: ข้อมูลที่เห็นคือเฉพาะใบเสร็จที่ผู้ถามส่งเองเท่านั้น ให้ระบุในคำตอบว่าเป็นของผู้ถามเอง และห้ามอ้างถึงข้อมูลของคนอื่น"
    if memory:
        p += MEMORY_PROMPT
    if coo:
        p += COO_PROMPT
    if interjection:
        p += INTERJECTION_PROMPT
    if files:
        p += FILES_PROMPT
    if web:
        p += WEB_PROMPT
    if language == "en":
        p += "\nLanguage: write your whole reply in English."
    elif language == "auto":
        p += (
            "\nLanguage: reply in the same language as the user's latest message (Thai or English)."
        )
    return p


@dataclass
class Answer:
    text: str
    rounds: int
    seconds: float
    tool_calls: list[tuple[str, dict]]
    receipts: list[tuple[int, str | None]] = field(default_factory=list)  # (id, reaction) created


def build_tools(
    ledger: LedgerTools,
    calls: list[tuple[str, dict]] | None = None,
    memory: MemoryTools | None = None,
) -> list:
    """The four ledger tools (plus the two memory tools when `memory` is given) as plain functions
    bound to `ledger` / `memory` (ADK wraps them as FunctionTools).

    Permission scoping lives in `ledger.forced`: none of these signatures has a submitter
    parameter, so the model cannot pass one (ADK also drops undeclared arguments). Errors are
    returned to the model as data instead of raised, so a bad date never kills the turn.
    """
    log = calls if calls is not None else []

    def safe(name: str, args: dict, fn):
        log.append((name, {k: v for k, v in args.items() if v is not None}))
        try:
            return fn()
        except (TypeError, ValueError) as e:
            return {"error": str(e)}

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

    def chat_days(start_date: str, end_date: str) -> dict:
        """What was discussed: per-channel day summaries (overview, topics, decisions, action items,
        numbers, open questions) for a date range, in the channels the asker may see. For today it
        also returns the raw messages newer than the summary (newer_than_summary).

        Args:
            start_date: ISO date YYYY-MM-DD, inclusive.
            end_date: ISO date YYYY-MM-DD, inclusive (at most 31 days from start_date).
        """
        a = dict(start_date=start_date, end_date=end_date)
        return safe("chat_days", a, lambda: memory.chat_days(**a))

    def chat_messages(
        start_date: str,
        end_date: str,
        channel_name: str | None = None,
        keyword: str | None = None,
        limit: int = 50,
        person: str | None = None,
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
        """
        a = dict(
            start_date=start_date,
            end_date=end_date,
            channel_name=channel_name,
            keyword=keyword,
            limit=limit,
            person=person,
        )
        return safe("chat_messages", a, lambda: memory.chat_messages(**a))

    def chat_search(
        keywords: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 20,
        person: str | None = None,
    ) -> dict:
        """Search day summaries AND raw messages for ANY of the keywords (case-insensitive
        substring), in the channels the asker may see. Hits are grouped by date and channel,
        ranked by keyword hits (summary hits count more), with matching summary lines and up to
        3 matching messages with links.

        Args:
            keywords: 3-8 search words: Thai and English forms, synonyms, names, abbreviations.
            start_date: ISO date YYYY-MM-DD, optional (default: the whole retention window).
            end_date: ISO date YYYY-MM-DD, optional (default: today).
            limit: max date/channel groups, default 20.
            person: only what this person (name or @mention) wrote or is named in, optional.
        """
        a = dict(
            keywords=keywords, start_date=start_date, end_date=end_date, limit=limit, person=person
        )
        return safe("chat_search", a, lambda: memory.chat_search(**a))

    return [*tools, chat_days, chat_messages, chat_search]


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
    coo: bool = False,
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
        coo=coo,
    )
    model_calls = 0

    def before_model(callback_context, llm_request):
        # Hard cap: after MAX_ROUNDS tool rounds the next call has no tools, so it must answer.
        nonlocal model_calls
        model_calls += 1
        if on_model_call:
            on_model_call(model_calls)
        if model_calls > MAX_ROUNDS:
            llm_request.config.tools = None
            llm_request.tools_dict.clear()
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
            temperature=0.6, thinking_config=types.ThinkingConfig(thinking_level="LOW")
        ),
        before_model_callback=before_model,
    )


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
    extra_tools: list | None = None,
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
    tools = build_tools(ledger, calls, memory)
    if usable and actions is not None:
        tools += build_file_tools(usable, actions, calls, created)
    if web_client is not None:
        tools.append(make_read_web(web_client, settings.gemini_model, question, calls))
    tools += extra_tools or []
    agent = build_agent(
        settings,
        tools,
        scoped=not is_exec,
        now=now,
        model=model,
        language=language,
        memory=memory is not None,
        interjection=interjection,
        files=bool(usable or skipped),
        web=web_client is not None,
        coo=bool(extra_tools),
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
    message = types.Content(role="user", parts=[types.Part(text=text_in), *f_parts])
    rounds, text = 0, ""
    async for event in runner.run_async(
        user_id=adk_user,
        session_id=session_id,
        new_message=message,
        run_config=RunConfig(max_llm_calls=MAX_ROUNDS + 2),
    ):
        if event.get_function_calls():
            rounds += 1
        elif event.is_final_response() and event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts if not p.thought).strip()
    if not text and interjection:
        text = NO_REPLY  # an unprompted answer that failed is never worth posting
    return Answer(
        text or t("qa_failed", resolve_lang(language, question)),
        rounds,
        time.monotonic() - started,
        calls,
        created,
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
    extra_tools: list | None = None,
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
            extra_tools=extra_tools,
        )
    )
