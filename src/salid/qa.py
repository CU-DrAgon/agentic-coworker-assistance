from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.models import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

from .config import Settings
from .embed import NullEmbedder, cosine
from .i18n import resolve_lang, t
from .memory_index import entries_for
from .memory_store import MAX_ROWS, MemoryMessage, MemoryStore
from .memory_summary import render_summary
from .store import Store

logger = logging.getLogger(__name__)
MAX_ROUNDS = 4
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
    return "฿" + f"{x:,.2f}".removesuffix(".00")


class LedgerTools:
    """Tools the model may call. All arithmetic happens here (SQL). `guild_id` (the tenant) and
    `forced_submitter_id` are injected into every query and are NOT part of any tool schema, so the
    model cannot widen the scope."""

    def __init__(self, store: Store, guild_id: int, forced_submitter_id: int | None = None):
        self.store = store
        self.guild_id = guild_id
        self.forced = forced_submitter_id

    def _fmt_summary(self, s: dict) -> dict:
        s["total_fmt"] = baht(s["total"])
        for g in s["groups"]:
            g["total_fmt"] = baht(g["total"])
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
        )
        return {"count": len(rows), "receipts": [self._r(r) for r in rows]}

    def compare_periods(
        self, a_start: str, a_end: str, b_start: str, b_end: str, group_by: str = "none"
    ) -> dict:
        a = self.store.summarize(self.guild_id, a_start, a_end, group_by, submitter_id=self.forced)
        b = self.store.summarize(self.guild_id, b_start, b_end, group_by, submitter_id=self.forced)
        am, bm = (
            {g["key"]: g["total"] for g in a["groups"]},
            {g["key"]: g["total"] for g in b["groups"]},
        )
        rows = []
        for k in sorted(set(am) | set(bm), key=lambda k: -(am.get(k, 0) + bm.get(k, 0))):
            rows.append(self._delta(k, am.get(k, 0.0), bm.get(k, 0.0)))
        return {
            "period_a": {
                "start": a_start,
                "end": a_end,
                "total_fmt": baht(a["total"]),
                "total": a["total"],
            },
            "period_b": {
                "start": b_start,
                "end": b_end,
                "total_fmt": baht(b["total"]),
                "total": b["total"],
            },
            "overall": self._delta("all", a["total"], b["total"]),
            "groups": rows,
        }

    @staticmethod
    def _delta(key, a, b):
        diff = round(a - b, 2)
        return {
            "key": key,
            "a": a,
            "b": b,
            "diff": diff,
            "diff_fmt": ("+" if diff > 0 else "-" if diff < 0 else "") + baht(abs(diff)),
            "pct_change_vs_b": round(diff / b * 100, 1) if b else None,
        }

    def flagged_receipts(self) -> dict:
        rows = self.store.flagged(self.guild_id, submitter_id=self.forced)
        return {"count": len(rows), "receipts": [self._r(r) for r in rows]}

    @staticmethod
    def _r(r) -> dict:
        return {
            "id": r.id,
            "date": r.effective_date,
            "vendor": r.vendor,
            "total": r.total,
            "total_fmt": baht(r.total) if r.total is not None else None,
            "category": r.category,
            "submitter": r.submitter_name,
            "confidence": r.confidence,
            "possible_duplicate_of": r.possible_duplicate_of,
            "notes": r.notes,
        }

    def call(self, name: str, args: dict) -> dict:
        allowed = {"spending_summary", "list_receipts", "compare_periods", "flagged_receipts"}
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
        embedder=None,
    ):
        self.store, self.guild_id, self.allowed, self.s = store, guild_id, dict(allowed), settings
        self.retention_days = retention_days or settings.memory_retention_days
        self.embedder = embedder or NullEmbedder()

    def _range(self, start_date: str, end_date: str) -> tuple[str, str]:
        a, b = date.fromisoformat(start_date), date.fromisoformat(end_date)
        if b < a:
            raise ValueError("end_date is before start_date")
        if (b - a).days + 1 > MAX_CHAT_DAYS:
            raise ValueError(f"range too long: at most {MAX_CHAT_DAYS} days per call")
        return a.isoformat(), b.isoformat()

    def _name(self, channel_id: int) -> str:
        return self.allowed.get(channel_id, str(channel_id))

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
            for x in self.store.summaries(self.guild_id, self.allowed, start, end)
            if x.message_count > 0
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
                self.guild_id, self.allowed, d, d, limit=MAX_ROWS, newest_first=False
            ):
                by_channel.setdefault(m.channel_id, []).append(m)
            for cid in sorted(by_channel):
                x = covered.get((cid, d))
                if x is not None and x.final:
                    continue
                rows = [
                    m for m in by_channel[cid] if x is None or x.cursor is None or m.key > x.cursor
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
        start = max(start, end - timedelta(days=MAX_SEARCH_DAYS - 1))
        a, b = start.isoformat(), end.isoformat()
        raw_from = (today - timedelta(days=RAW_RECENT_DAYS - 1)).isoformat()

        pending = None
        if use_meaning:  # embed the query while the summaries load
            pending = _embed_pool.submit(_query_vector, self.embedder, query or " ".join(kws))
        t_index = time.monotonic()
        entries = entries_for(self.store, self.guild_id, self.allowed, a, b, self.embedder)
        ms_index = (time.monotonic() - t_index) * 1000
        qv, ms_embed = None, 0.0
        if pending is not None:
            try:
                qv = pending.result()
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
                if h:
                    g = group(e.local_date, e.channel_id)
                    g["score"] += h * SUMMARY_WEIGHT
                    g["summary"].append(line[:300])
            if qv is not None and e.vectors:
                sims = [
                    (cosine(qv, v), line)
                    for v, line in zip(e.vectors, e.items, strict=False)
                    if len(v) == len(qv)  # vectors of another model/dim are ignored
                ]
                best = max(sims, default=(0.0, ""))
                if best[0] >= MIN_SIMILARITY:
                    g = group(e.local_date, e.channel_id)
                    g["sim"] = best[0]
                    g["near"] = [line[:300] for _, line in sorted(sims, reverse=True)[:2]]
        raw_queries = 0
        if kws and max(a, raw_from) <= b:  # recent raw messages (not summarised yet)
            raw_queries += 1
            rows = self.store.messages(
                self.guild_id, self.allowed, max(a, raw_from), b, limit=MAX_ROWS_SEARCH,
                newest_first=True,
            )  # fmt: skip
            for m in rows:  # newest first
                h = hits(f"{m.text} {m.author_name}")
                if h:
                    g = group(m.local_date, m.channel_id)
                    g["score"] += h
                    g["n_msgs"] += 1
                    if len(g["messages"]) < SEARCH_MSGS_PER_GROUP:
                        g["messages"].append(self._msg(m))

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
            for day, cid in [k for k in order if k[0] < raw_from][:RAW_GROUPS]:
                raw_queries += 1
                g = groups[(day, cid)]
                for m in reversed(
                    self.store.messages_after(self.guild_id, cid, day, None, RAW_GROUP_FETCH)
                ):
                    if hits(f"{m.text} {m.author_name}"):
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
    ) -> dict:
        start, end = self._range(start_date, end_date)
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
        rows = self.store.messages(
            self.guild_id,
            self.allowed,
            start,
            end,
            channel_id=channel_id,
            keyword=keyword,
            limit=max(1, min(int(limit), MAX_MESSAGE_LIMIT)),
            newest_first=True,
        )
        return {"count": len(rows), "messages": [self._msg(m) for m in rows]}

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


PERSONA = {
    "th": (
        "คุณชื่อ “สลิด” เป็นเลขาและเพื่อนร่วมทีมของร้าน อยู่ในแชต Discord ของทีม "
        "นิสัยเป็นกันเอง สุภาพ อารมณ์ดี พูดภาษาไทยเป็นธรรมชาติ ลงท้ายด้วยคำว่าครับ "
        "ตอบตามภาษาที่ผู้ใช้พิมพ์มา และปรับความยาวตามคำถาม (คุยเล่นตอบสั้น งานที่ขอรายละเอียดตอบละเอียดได้)\n"
    ),
    "en": (
        "Your name is “Salid” (สลิด). You are the shop's secretary and teammate in the team's "
        "Discord chat. You are friendly, polite and upbeat. ALWAYS reply in English, even if the "
        "user writes in Thai (keep Thai names, vendors and category names as they are). "
        "Adjust the length to the question (short for small talk, detailed when the user asks "
        "for detail)\n"
    ),
    "auto": (
        "คุณชื่อ “สลิด” (Salid) เป็นเลขาและเพื่อนร่วมทีมของร้าน อยู่ในแชต Discord ของทีม "
        "นิสัยเป็นกันเอง สุภาพ อารมณ์ดี ตอบด้วยภาษาเดียวกับข้อความล่าสุดของผู้ใช้ "
        "(reply in the same language as the user's latest message): ถ้าผู้ใช้พิมพ์ไทยให้ตอบไทยเป็นธรรมชาติ "
        "ลงท้ายด้วยคำว่าครับ ถ้าพิมพ์อังกฤษให้ตอบเป็นอังกฤษ "
        "และปรับความยาวตามคำถาม (คุยเล่นตอบสั้น งานที่ขอรายละเอียดตอบละเอียดได้)\n"
    ),
}


MEMORY_PROMPT = (
    "\nความจำของทีม: เมื่อถูกถามว่าเกิดอะไรขึ้น ใครพูดอะไร หรืองาน/เรื่องใดถึงไหนแล้ว "
    "ถ้าคำถามไม่มีวันที่ชัดเจน ให้เรียก chat_search ก่อน โดยใส่ query เป็นคำถามของผู้ใช้ และ keywords 2-5 คำ "
    "(ชื่อคน ตัวเลข รหัส คำสำคัญ) เพราะ chat_search ค้นตามความหมายได้ทั้งไทยและอังกฤษ "
    "แล้วค่อยเปิดวันที่เจอด้วย chat_days หรือ chat_messages. "
    "ถ้าหารหัส ตัวเลข หรือคำพูดเป๊ะๆ ที่เก่ากว่า 3 วัน ให้ใช้ chat_messages พร้อม keyword และช่วงวันที่. "
    "ถ้าคำถามมีวันที่ ให้เรียก chat_days (สรุปรายวันต่อห้อง) ได้เลย "
    "แล้วใช้ chat_messages (ข้อความจริง ค้นด้วย keyword) เมื่อต้องการรายละเอียดหรือคำพูดจริง. "
    "ข้อความที่ขึ้นต้นด้วย [รูป: ...] คือสิ่งที่อ่านได้จากรูปที่มีคนส่ง (รวมตัวหนังสือในรูป). "
    "ตอบโดยระบุห้องและวันที่ที่มา และถ้าอ้างข้อความจริงให้แนบลิงก์ (link). "
    "ถ้าไม่พบข้อมูลให้บอกตรงๆ ว่าไม่พบ ห้ามเดา และห้ามพูดถึงห้องที่เครื่องมือไม่ได้คืนมา. "
    "แปลงคำอย่าง 'เมื่อวาน' หรือ 'สัปดาห์ที่แล้ว' เป็นช่วงวันที่ YYYY-MM-DD จากวันนี้ในเขตเวลาที่ระบุ "
    "(chat_days/chat_messages รับช่วงละไม่เกิน 31 วัน ผลลัพธ์ถูกตัดความยาว ถ้ามี note ให้ย่อช่วงหรือใช้ chat_search). ผลลัพธ์ newer_than_summary คือข้อความของวันนี้ที่ใหม่กว่าสรุป."
)


# What Salid can do, so it can answer "what can you do?" / "how do I ..." accurately. Keep in
# sync with the features (commands are listed in both languages).
CAPABILITIES_PROMPT = (
    "\nสิ่งที่สลิดทำได้ (ใช้ตอบเมื่อมีคนถามว่าทำอะไรได้บ้าง หรือถามวิธีใช้ ตอบให้กระชับ เลือกเฉพาะที่เกี่ยวข้อง "
    "อย่าอ้างว่าทำได้มากกว่านี้):\n"
    "1) ใบเสร็จและบัญชี: ส่งรูปใบเสร็จในห้องที่ชื่อมีคำว่า ใบเสร็จ/receipt/invoice/ใบกำกับ/สลิป/slip/บิล/bill/expense/ค่าใช้จ่าย/เบิก หรือแท็กสลิดพร้อมรูป สลิดอ่านร้าน ยอด วันที่ หมวด "
    "แล้วบันทึกลงบัญชี เตือนถ้าน่าจะซ้ำ แก้ข้อมูลได้โดยตอบกลับ (reply) ข้อความใบเสร็จของสลิด. "
    "ถามยอดใช้จ่ายได้ เช่น เดือนนี้จ่ายค่าอะไรไปเท่าไร แยกหมวด ใครส่ง รายการที่ต้องตรวจ "
    "(ผู้บริหารเห็นของทั้งทีม คนอื่นเห็นเฉพาะของตัวเอง). มีสรุปรายวันส่งเข้าห้องตามเวลาที่ตั้งไว้\n"
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
    "สลิดจะทำงานเบื้องหลังแล้วส่งรายงานกลับมา\n"
    "7) ผู้ช่วยทั่วไป: คุยเล่น ตอบความรู้ทั่วไป ร่างข้อความ แปลภาษา สรุปหรืออธิบายเรื่องต่างๆ\n"
    "8) ตั้งค่า: แอดมินพิมพ์ /ตั้งค่า หรือ /setup เพื่อรับลิงก์เข้าหน้าศูนย์บัญชาการ (Hub) บนเว็บ "
    "ที่เปิดปิดความสามารถ ตั้งห้อง โหมด ภาษา ระยะเวลาความจำ และดูความเคลื่อนไหวได้.\n"
    "ข้อจำกัด: ค้นอินเทอร์เน็ตไม่ได้ (ข่าวหรือข้อมูลล่าสุดอาจไม่อัปเดต) และตั้งค่าแต่ละเซิร์ฟเวอร์/ห้องอาจปิดบางความสามารถไว้ "
    "ถ้าไม่แน่ใจให้แนะนำ /โหมด /ภาษา /ความจำ เพื่อดูสถานะ"
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


def stable_prompt(language: str, memory: bool) -> str:
    """The part of the system prompt that is identical for every request in one language (persona,
    rules, capabilities, memory guidance). It comes first so Gemini's implicit prefix cache can
    reuse it across requests and tool rounds."""
    p = (
        PERSONA[language if language in PERSONA else "th"]
        + "คุยได้ทุกเรื่องเหมือนผู้ช่วย AI ทั่วไป: คุยเล่น ทักทาย ให้กำลังใจ ตอบความรู้ทั่วไป "
        "ช่วยคิดไอเดีย ร่างข้อความ แปลภาษา สรุปหรืออธิบายเรื่องต่างๆ "
        "ถ้าเป็นข่าวหรือข้อมูลล่าสุดที่อาจไม่รู้ ให้บอกตรงๆ ว่าอาจไม่อัปเดต\n"
        "กฎเหล็กเรื่องบัญชีของร้าน: ห้ามคำนวณตัวเลขของร้านเอง (บวก ลบ เปอร์เซ็นต์) ให้เรียกเครื่องมือและใช้ตัวเลขที่เครื่องมือคืนมาเท่านั้น "
        "(ใช้ฟิลด์ *_fmt ได้เลย) ตัวเลขใส่ , คั่นหลักพัน และนำหน้าด้วย ฿. ถ้าไม่มีข้อมูลให้บอกตรงๆ ห้ามเดาหรือแต่งข้อมูลของร้าน. "
        "เรียกเครื่องมือหลายตัวพร้อมกันได้เพื่อความเร็ว. ถ้ามีบทสนทนาก่อนหน้าแนบมา ให้อ่านเหมือนคนที่นั่งอยู่ในห้องแชต: เข้าใจว่าใครพูดกับใคร เรื่องที่กำลังคุยค้างอยู่ คำอย่าง “อันนั้น” “เมื่อกี้” “แล้วล่ะ” หมายถึงอะไร และสิ่งที่สลิดเคยตอบไปแล้ว (ไม่ต้องทักทายซ้ำหรือแนะนำตัวใหม่ทุกครั้ง) แล้วตอบเฉพาะข้อความล่าสุด"
    )
    p += CAPABILITIES_PROMPT
    if memory:
        p += MEMORY_PROMPT
    return p


def volatile_prompt(
    settings: Settings, now, *, scoped: bool, interjection: bool, language: str
) -> str:
    """The per-request tail of the system prompt: date and time, the asker's scope, the
    interjection rule and the language line."""
    p = (
        f"\nวันนี้คือ {now.date().isoformat()} (วัน{THAI_DAYS[now.weekday()]}) เวลา {now:%H:%M} เขตเวลา {settings.tz}. "
        f"สัปดาห์เริ่มวันจันทร์. ใช้วันที่นี้แปลง 'เดือนนี้', 'สัปดาห์ก่อน' ฯลฯ เป็นช่วงวันที่ YYYY-MM-DD"
    )
    if scoped:
        p += "\nผู้ถามไม่ใช่ผู้บริหาร: ข้อมูลที่เห็นคือเฉพาะใบเสร็จที่ผู้ถามส่งเองเท่านั้น ให้ระบุในคำตอบว่าเป็นของผู้ถามเอง และห้ามอ้างถึงข้อมูลของคนอื่น"
    if interjection:
        p += INTERJECTION_PROMPT
    if language == "en":
        p += "\nLanguage: write your whole reply in English."
    elif language == "auto":
        p += (
            "\nLanguage: reply in the same language as the user's latest message (Thai or English)."
        )
    return p


def system_prompt(
    settings: Settings,
    scoped: bool,
    now=None,
    language: str = "th",
    memory: bool = False,
    interjection: bool = False,
) -> str:
    """`language`: th / en answer in that language; auto = the language of the user's message.
    Stable part first, per-request part last (see `stable_prompt`)."""
    now = now or settings.now()
    return stable_prompt(language, memory) + volatile_prompt(
        settings, now, scoped=scoped, interjection=interjection, language=language
    )


@dataclass
class Answer:
    text: str
    rounds: int
    seconds: float
    tool_calls: list[tuple[str, dict]]
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
    ) -> dict:
        """List individual receipts in a date range with optional filters.

        Args:
            start_date: ISO date YYYY-MM-DD, inclusive.
            end_date: ISO date YYYY-MM-DD, inclusive.
            category: expense category, one of the Thai categories (e.g. วัตถุดิบ).
            vendor_contains: substring of the vendor name.
            submitter: substring of the submitter display name.
            limit: max rows, default 20.
        """
        a = dict(
            start_date=start_date,
            end_date=end_date,
            category=category,
            vendor_contains=vendor_contains,
            submitter=submitter,
            limit=limit,
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

    tools = [spending_summary, list_receipts, compare_periods, flagged_receipts]
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
    ) -> dict:
        """Raw chat messages (newest first) with author, local time, channel, text and link.

        Args:
            start_date: ISO date YYYY-MM-DD, inclusive.
            end_date: ISO date YYYY-MM-DD, inclusive (at most 31 days from start_date).
            channel_name: only this channel (name without #), optional.
            keyword: case-insensitive text to look for in the message or author name, optional.
            limit: max messages, default 50.
        """
        a = dict(
            start_date=start_date,
            end_date=end_date,
            channel_name=channel_name,
            keyword=keyword,
            limit=limit,
        )
        return safe("chat_messages", a, lambda: memory.chat_messages(**a))

    def chat_search(
        keywords: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 20,
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
            query: the user's question in plain words, any language; improves meaning-based
                matching.
        """
        a = dict(
            keywords=keywords, start_date=start_date, end_date=end_date, limit=limit, query=query
        )
        return safe("chat_search", a, lambda: memory.chat_search(**a))

    return [*tools, chat_days, chat_messages, chat_search]


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
) -> Answer:
    """Answer a question with an ADK agent. `history` is recent chat text, context only. Scope is decided here in code, before any data
    reaches the model: tools are built per request around a LedgerTools bound to `guild_id`
    whose `forced_submitter_id` is the asker for non-executives."""
    started = time.monotonic()
    calls: list[tuple[str, dict]] = []
    tool_ms: dict[str, float] = {}
    model_calls = 0

    def count_call(n: int) -> None:
        nonlocal model_calls
        model_calls = n

    ledger = LedgerTools(store, guild_id, forced_submitter_id=None if is_exec else user_id)
    agent = build_agent(
        settings,
        build_tools(ledger, calls, memory, tool_ms),
        scoped=not is_exec,
        now=now,
        model=model,
        on_model_call=count_call,
        language=language,
        memory=memory is not None,
        interjection=interjection,
    )
    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    adk_user, session_id = str(user_id), uuid.uuid4().hex
    await runner.session_service.create_session(
        app_name=APP_NAME, user_id=adk_user, session_id=session_id
    )
    text_in = (
        f"บทสนทนาก่อนหน้าในห้องนี้ (หลายคนคุยกัน เรียงจากเก่าไปใหม่):\n{history}\n\n"
        f"ข้อความล่าสุดที่ต้องตอบ:\n{question}"
        if history
        else question
    )
    message = types.Content(role="user", parts=[types.Part(text=text_in)])
    rounds, text = 0, ""
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
        elif event.is_final_response() and event.content and event.content.parts:
            text = "".join(p.text or "" for p in event.content.parts if not p.thought).strip()
    if not text and interjection:
        text = NO_REPLY  # an unprompted answer that failed is never worth posting
    return Answer(
        text or t("qa_failed", resolve_lang(language, question)),
        rounds,
        time.monotonic() - started,
        calls,
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
        )
    )
