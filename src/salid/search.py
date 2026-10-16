"""`search_everything`: one agent tool that searches every internal source the asker may see.

Nothing here widens scope. Each source is an adapter around a tool object that the bot already
built for THIS asker and audience (`MemoryTools.chat_search`, the `wiki_search` / `list_tasks` /
`find_place` agent tools, the asker's meetings, and the ledger only when the asker is an
executive). The model never picks a scope: it only supplies the query, the source names and a date
range. Sources run in parallel; results are merged, deduplicated and ranked by a simple score
(keyword match + the source's own rank + recency) and the top ones are returned with source type,
date, who, a one-line snippet and a jump link.

Thai has no word spaces, so matching is substring based (like the memory search). The query is
expanded into a few keywords (Thai <-> English synonyms) by one cheap model call; without a model a
small built-in glossary and the query's own words are used.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from google.genai import types

from . import llm
from .catchup import parse_since

logger = logging.getLogger(__name__)

SOURCES = ("chat", "files", "wiki", "receipts", "tasks", "meetings", "places")
TOP_N = 10
PER_SOURCE_CAP = 5  # so one chatty source cannot fill the whole answer
SNIPPET = 160
MAX_KEYWORDS = 8
SOURCE_TIMEOUT_S = 25.0
RECENCY_HALF_LIFE_DAYS = 30.0
UNDATED_RECENCY = 0.4
RECEIPT_LOOKBACK_DAYS = 400
UPCOMING_DAYS = 60
_FILE_NOTE = re.compile(r"\[[^\]\n]{1,24}:[^\]]*\]")  # "[PDF: quotation.pdf]" descriptors

# Fallback synonyms when the model is not reachable. Lower-case; both directions are used.
GLOSSARY: tuple[tuple[str, ...], ...] = (
    ("ใบเสนอราคา", "quotation", "quote", "ราคาเสนอ"),
    ("ใบแจ้งหนี้", "invoice", "bill"),
    ("ใบเสร็จ", "receipt"),
    ("ใบกำกับภาษี", "tax invoice"),
    ("สัญญา", "contract", "agreement"),
    ("ประชุม", "meeting"),
    ("รายงาน", "report"),
    ("งบประมาณ", "budget", "งบ"),
    ("ลูกค้า", "client", "customer"),
    ("ผู้ขาย", "vendor", "supplier", "ซัพพลายเออร์"),
    ("ใบสั่งซื้อ", "purchase order", "po"),
    ("ลา", "leave", "วันลา"),
    ("เงินเดือน", "salary", "payroll"),
    ("โปรเจกต์", "project", "โครงการ"),
    ("กำหนดส่ง", "deadline", "due date"),
    ("ขั้นตอน", "procedure", "process", "วิธี"),
)


# ------------------------------------------------------------------ data
@dataclass
class Hit:
    source: str  # one of SOURCES
    title: str = ""
    snippet: str = ""
    when: datetime | None = None
    who: str = ""
    link: str = ""
    rank: float = 0.0  # the source's own opinion, 0..1 (higher = better)
    open_with: str = ""  # a follow-up tool call when there is no link (wiki)
    score: float = field(default=0.0, compare=False)


@dataclass
class SearchSources:
    """name -> adapter(keywords, query, start, end) returning hits. Absent = not searched."""

    adapters: dict[str, Callable[[list[str], str, date, date], list[Hit]]] = field(
        default_factory=dict
    )
    withheld: Callable[[], str | None] = lambda: None  # chat records the asker may not see


def _cut(text: str, n: int = SNIPPET) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = datetime.combine(date.fromisoformat(str(s)[:10]), datetime.min.time())
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


# ------------------------------------------------------------------ query expansion
def glossary_terms(query: str) -> list[str]:
    low = (query or "").casefold()
    out: list[str] = []
    for group in GLOSSARY:
        if any(term in low for term in group):
            out += [g for g in group if g not in out]
    return out


def base_keywords(query: str) -> list[str]:
    q = " ".join((query or "").casefold().split())
    words = [w for w in re.split(r"[\s,;/|]+", q) if len(w) >= 2]
    return list(dict.fromkeys([q, *words])) if q else []


class Expander:
    """query -> keywords (original words, Thai/English synonyms) with one cheap model call and a
    small cache; any failure falls back to the glossary."""

    def __init__(self, client, model: str, *, cache_size: int = 64):
        self.client, self.model = client, model
        self._cache: OrderedDict[str, list[str]] = OrderedDict()
        self._size = cache_size

    def __call__(self, query: str) -> list[str]:
        key = " ".join((query or "").casefold().split())
        if key in self._cache:
            self._cache.move_to_end(key)
            return list(self._cache[key])
        kws = merge_keywords(base_keywords(query), glossary_terms(query), self._ask(query))
        self._cache[key] = kws
        while len(self._cache) > self._size:
            self._cache.popitem(last=False)
        return list(kws)

    def _ask(self, query: str) -> list[str]:
        if self.client is None:
            return []
        prompt = (
            "Expand this workplace search query into search keywords for substring matching in a "
            "Thai/English company chat. Return the important words from the query plus synonyms "
            "and the same terms in the other language (Thai <-> English, e.g. ใบเสนอราคา <-> "
            f"quotation). At most {MAX_KEYWORDS} short items, no sentences. Reply JSON: "
            f'{{"keywords": ["..."]}}\nQuery: {(query or "")[:300]}'
        )
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.0, max_output_tokens=300
        )
        try:
            resp = llm.generate(
                self.client,
                self.model,
                [types.Content(role="user", parts=[types.Part(text=prompt)])],
                cfg,
            )
            data = json.loads(getattr(resp, "text", None) or "{}")
            items = data.get("keywords") if isinstance(data, dict) else None
        except Exception as e:  # never log the query
            logger.warning("search expansion failed (%s)", type(e).__name__)
            return []
        return [str(x) for x in items or [] if isinstance(x, str)]


def merge_keywords(*groups: list[str]) -> list[str]:
    out: list[str] = []
    for g in groups:
        for k in g:
            k = " ".join(str(k).casefold().split())[:60]
            if len(k) >= 2 and k not in out:
                out.append(k)
    return out[:MAX_KEYWORDS]


# ------------------------------------------------------------------ scoring / merging
def match_score(hit: Hit, keywords: list[str]) -> float:
    """0..1 from how many of the keywords occur in the hit's text (any one is a real match)."""
    text = f"{hit.title} {hit.snippet} {hit.who}".casefold()
    n = sum(1 for k in keywords if k in text)
    return 0.15 if n == 0 else min(1.0, 0.5 + 0.25 * (n - 1))


def recency(when: datetime | None, now: datetime) -> float:
    if when is None:
        return UNDATED_RECENCY
    days = max(0.0, (now - when).total_seconds() / 86400)
    return 0.5 ** (days / RECENCY_HALF_LIFE_DAYS)


def score_hit(hit: Hit, keywords: list[str], now: datetime) -> float:
    return round(
        0.65 * match_score(hit, keywords)
        + 0.20 * max(0.0, min(1.0, hit.rank))
        + 0.15 * recency(hit.when, now),
        4,
    )


def _key(h: Hit) -> str:
    if h.link:
        return h.link
    return f"{h.source}|{' '.join(h.title.casefold().split())}|{' '.join(h.snippet.casefold().split())[:60]}"


def rank_hits(hits: list[Hit], keywords: list[str], now: datetime, top: int = TOP_N) -> list[Hit]:
    """Score, drop duplicates (same link, or the same text; the better score wins and a `files`
    hit beats a plain chat hit for the same message), cap per source, keep the best `top`."""
    best: dict[str, Hit] = {}
    for h in hits:
        h.score = score_hit(h, keywords, now)
        k = _key(h)
        cur = best.get(k)
        if cur is None or h.score > cur.score:
            best[k] = h
    ordered = sorted(
        best.values(),
        key=lambda h: (h.score, h.when or datetime.min.replace(tzinfo=UTC)),
        reverse=True,
    )
    counts: dict[str, int] = {}
    out: list[Hit] = []
    for h in ordered:
        if counts.get(h.source, 0) >= PER_SOURCE_CAP:
            continue
        counts[h.source] = counts.get(h.source, 0) + 1
        out.append(h)
        if len(out) >= top:
            break
    return out


# ------------------------------------------------------------------ adapters over scoped tools
def _tool(tools: list, name: str):
    return next((f for f in tools if getattr(f, "__name__", "") == name), None)


class SharedChatSearch:
    """One `MemoryTools.chat_search` per distinct request, shared by the "chat" and "files"
    adapters. Both run in parallel with the same keywords and dates, so without this every
    search_everything call ran the same memory search twice (two identical log lines ~50 ms
    apart). The second caller waits for the first one's result."""

    def __init__(self, memory) -> None:
        self.memory = memory
        self._lock = threading.Lock()
        self._runs: dict[tuple, Future] = {}

    def __call__(self, keywords: list[str], start: date, end: date) -> dict:
        key = (tuple(keywords), start, end)
        with self._lock:
            fut = self._runs.get(key)
            owner = fut is None
            if owner:
                fut = self._runs[key] = Future()
        if owner:
            try:
                fut.set_result(
                    self.memory.chat_search(
                        keywords=keywords,
                        start_date=start.isoformat(),
                        end_date=end.isoformat(),
                        limit=15,
                    )
                )
            except BaseException as e:
                fut.set_exception(e)
        return fut.result()


def chat_adapter(memory, zone, *, files: bool, shared: SharedChatSearch | None = None) -> Callable:
    """`MemoryTools.chat_search` (already limited to the asker's readable channels and the role
    hierarchy). `files=True` keeps only messages that carry a file note, else only plain chat.
    `shared`: lets the chat and files adapters of one request share a single search."""
    shared = shared or SharedChatSearch(memory)

    def run(keywords: list[str], _query: str, start: date, end: date) -> list[Hit]:
        res = shared(keywords, start, end)
        out: list[Hit] = []
        for rank, g in enumerate(res.get("hits") or []):
            r = max(0.0, 1.0 - rank / 15)
            day = _parse_dt(g.get("date"))
            lines = g.get("summary_hits") or []
            msgs = g.get("messages") or []
            for m in msgs[:3]:
                is_file = bool(_FILE_NOTE.search(m.get("text", "")))
                if is_file != files:
                    continue
                out.append(
                    Hit(
                        "files" if is_file else "chat",
                        title=f"#{g.get('channel', '')}",
                        snippet=_cut(m.get("text", "")),
                        when=_parse_dt((m.get("time") or "").replace(" ", "T")) or day,
                        who=m.get("author", ""),
                        link=m.get("link", ""),
                        rank=r,
                    )
                )
            if lines and not files and not msgs:
                out.append(
                    Hit(
                        "chat",
                        title=f"#{g.get('channel', '')} (summary)",
                        snippet=_cut(lines[0]),
                        when=day,
                        rank=r * 0.8,
                    )
                )
        return out

    return run


def wiki_adapter(wiki_search: Callable) -> Callable:
    def run(_kws: list[str], query: str, _a: date, _b: date) -> list[Hit]:
        res = wiki_search(query)
        return [
            Hit(
                "wiki",
                title=p.get("title", ""),
                snippet=_cut(p.get("snippet", "")),
                when=_parse_dt(p.get("updated_at")),
                who=p.get("author") or "",
                rank=max(0.0, 1.0 - i / 5),
                open_with=f"wiki_get(topic={p.get('title', '')!r})",
            )
            for i, p in enumerate(res.get("pages") or [])
        ]

    return run


def tasks_adapter(list_tasks: Callable) -> Callable:
    def run(keywords: list[str], _q: str, _a: date, _b: date) -> list[Hit]:
        res = list_tasks(status="all")
        out: list[Hit] = []
        for t in res.get("tasks") or []:
            text = (
                f"{t.get('title', '')} {t.get('note') or ''} {t.get('assignee') or ''}".casefold()
            )
            if not any(k in text for k in keywords):
                continue
            tag = f"{t.get('status', '')}" + (", overdue" if t.get("overdue") else "")
            due = f", due {t['due']}" if t.get("due") else ""
            out.append(
                Hit(
                    "tasks",
                    title=_cut(t.get("title", ""), 80),
                    snippet=f"{tag}{due}" + (f" - {t['note']}" if t.get("note") else ""),
                    when=_parse_dt(t.get("due")),
                    who=t.get("assignee") or "",
                    link=t.get("link") or "",
                    rank=0.6,
                )
            )
        return out

    return run


def places_adapter(find_place: Callable) -> Callable:
    def run(keywords: list[str], _q: str, _a: date, _b: date) -> list[Hit]:
        seen: dict[str, Hit] = {}
        for k in keywords[:4]:
            for p in (find_place(k) or {}).get("places") or []:
                seen.setdefault(
                    p.get("name", ""),
                    Hit(
                        "places",
                        title=p.get("name", ""),
                        snippet=_cut(p.get("address", "")),
                        who=p.get("owner", "") if p.get("personal") else "",
                        link=p.get("maps_link") or "",
                        rank=0.7,
                    ),
                )
        return list(seen.values())

    return run


def meetings_adapter(
    list_meetings: Callable[[str, str], list[dict]], user_id: int, zone
) -> Callable:
    """Meetings the asker organised or was invited to: title, description, minutes, attendees."""

    def run(keywords: list[str], _q: str, start: date, end: date) -> list[Hit]:
        a = datetime.combine(start, datetime.min.time(), zone).astimezone(UTC).isoformat()
        # upcoming meetings count too: look UPCOMING_DAYS past the end of the range
        b = datetime.combine(end + timedelta(days=UPCOMING_DAYS), datetime.min.time(), zone)
        out: list[Hit] = []
        for m in list_meetings(a, b.astimezone(UTC).isoformat()):
            if user_id not in m.get("attendee_ids", []) and m.get("organizer_id") != user_id:
                continue
            blob = " ".join(
                [m.get("title", ""), m.get("description", ""), m.get("summary", "")]
                + list(m.get("attendee_names", []))
            ).casefold()
            if not any(k in blob for k in keywords):
                continue
            minutes = (m.get("summary") or "").strip()
            out.append(
                Hit(
                    "meetings",
                    title=m.get("title", ""),
                    snippet=_cut(
                        minutes
                        or m.get("description", "")
                        or ", ".join(m.get("attendee_names", []))
                    ),
                    when=_parse_dt(m.get("start_utc")),
                    who=m.get("organizer_name", ""),
                    link=m.get("summary_url") or m.get("link") or "",
                    rank=0.6,
                )
            )
        return out

    return run


def receipts_adapter(ledger, now: Callable[[], datetime]) -> Callable:
    """Receipts by vendor name through the ledger tool object (its own scoping applies)."""

    def run(keywords: list[str], _q: str, start: date, end: date) -> list[Hit]:
        floor = now().date() - timedelta(days=RECEIPT_LOOKBACK_DAYS)
        a, b = max(start, floor).isoformat(), end.isoformat()
        seen: dict[int, Hit] = {}
        for k in keywords[:5]:
            for r in ledger.list_receipts(a, b, vendor_contains=k, limit=8).get("receipts") or []:
                seen.setdefault(
                    r["id"],
                    Hit(
                        "receipts",
                        title=f"{r.get('vendor') or '-'} #{r['id']}",
                        snippet=_cut(
                            f"{r.get('total_fmt') or ''} {r.get('category') or ''} {r.get('notes') or ''}"
                        ),
                        when=_parse_dt(r.get("date")),
                        who=r.get("submitter") or "",
                        rank=0.6,
                    ),
                )
        return list(seen.values())

    return run


def build_sources(
    *,
    memory,
    tools: list,
    zone,
    now: Callable[[], datetime],
    user_id: int,
    list_meetings: Callable[[str, str], list[dict]] | None = None,
    ledger=None,
) -> SearchSources:
    """Adapters for whatever the asker has. `tools` = the agent tools already built for them."""
    src = SearchSources()
    if memory is not None:
        shared = SharedChatSearch(memory)  # chat + files: one memory search, not two
        src.adapters["chat"] = chat_adapter(memory, zone, files=False, shared=shared)
        src.adapters["files"] = chat_adapter(memory, zone, files=True, shared=shared)
        src.withheld = lambda: getattr(memory, "_hidden_note", lambda: None)()
    if (f := _tool(tools, "wiki_search")) is not None:
        src.adapters["wiki"] = wiki_adapter(f)
    if (f := _tool(tools, "list_tasks")) is not None:
        src.adapters["tasks"] = tasks_adapter(f)
    if (f := _tool(tools, "find_place")) is not None:
        src.adapters["places"] = places_adapter(f)
    if list_meetings is not None:
        src.adapters["meetings"] = meetings_adapter(list_meetings, user_id, zone)
    if ledger is not None:
        src.adapters["receipts"] = receipts_adapter(ledger, now)
    return src


# ------------------------------------------------------------------ the search itself
def pick_sources(sources, available: list[str]) -> tuple[list[str], list[str]]:
    """(names to search, unknown/unavailable names). "auto"/empty = everything available."""
    if sources in (None, "", "auto") or (isinstance(sources, list) and "auto" in sources):
        return list(available), []
    if isinstance(sources, str):
        sources = [s for s in re.split(r"[\s,]+", sources) if s]
    want = [s.strip().casefold() for s in sources]
    want = [
        {
            "chats": "chat",
            "messages": "chat",
            "file": "files",
            "task": "tasks",
            "meeting": "meetings",
            "receipt": "receipts",
            "ledger": "receipts",
            "place": "places",
        }.get(w, w)
        for w in want
    ]
    return [w for w in dict.fromkeys(want) if w in available], [
        w for w in want if w not in available
    ]


def search_everything(
    src: SearchSources,
    expand: Callable[[str], list[str]],
    query: str,
    *,
    sources="auto",
    since: str | None = None,
    until: str | None = None,
    now: datetime,
    zone,
    default_days: int = 365,
    top: int = TOP_N,
    pool: ThreadPoolExecutor | None = None,
) -> dict:
    query = " ".join((query or "").split())[:300]
    if not query:
        return {"error": "give a search query"}
    today = now.astimezone(zone).date()
    end = today
    if until:
        u = parse_since(until, now, zone)
        if u is None:
            return {"error": f"cannot read until={until!r}; use YYYY-MM-DD"}
        end = min(u.date(), today)
    start = today - timedelta(days=default_days)
    if since:
        s = parse_since(since, now, zone)
        if s is None:
            return {
                "error": f"cannot read since={since!r}; use YYYY-MM-DD, 'last week' or '30 days'"
            }
        start = s.date()
    if end < start:
        return {"error": "until is before since"}
    names, unknown = pick_sources(sources, [n for n in SOURCES if n in src.adapters])
    if not names:
        return {"error": "no searchable source", "unavailable": unknown}
    keywords = expand(query) or base_keywords(query)

    own = pool is None
    pool = pool or ThreadPoolExecutor(max_workers=len(names), thread_name_prefix="search")
    futures = {n: pool.submit(src.adapters[n], keywords, query, start, end) for n in names}
    hits: list[Hit] = []
    failed: list[str] = []
    counts: dict[str, int] = {}
    try:
        for n, fut in futures.items():
            try:
                got = fut.result(timeout=SOURCE_TIMEOUT_S)
            except FutureTimeout:
                failed.append(n)
                fut.cancel()
                continue
            except Exception as e:
                logger.warning("search source %s failed (%s)", n, type(e).__name__)
                failed.append(n)
                continue
            counts[n] = len(got)
            hits += got
    finally:
        if own:
            pool.shutdown(wait=False, cancel_futures=True)
    ranked = rank_hits(hits, keywords, now, top)
    out: dict = {
        "query": query,
        "keywords": keywords,
        "searched": names,
        "count": len(ranked),
        "results": [
            {
                "n": i,
                "source": h.source,
                "date": f"{h.when.astimezone(zone):%Y-%m-%d}" if h.when else None,
                "who": h.who or None,
                "title": h.title,
                "snippet": h.snippet,
                "link": h.link or None,
                **({"open_with": h.open_with} if h.open_with else {}),
            }
            for i, h in enumerate(ranked, 1)
        ],
    }
    if failed:
        out["unavailable_now"] = failed
    if unknown:
        out["not_searched"] = unknown
    if w := src.withheld():
        out["withheld"] = w
    if not ranked:
        out["note"] = (
            "nothing found in the sources you can see; try other words or ask in the channel"
        )
    return out


def search_tools(
    src: SearchSources,
    expand: Callable[[str], list[str]],
    now: Callable[[], datetime],
    zone,
    calls: list | None = None,
) -> list:
    """The `search_everything` function the agent registers (bound to one asker's sources)."""
    log = calls if calls is not None else []
    available = [n for n in SOURCES if n in src.adapters]

    def search_everything_tool(
        query: str,
        sources: str | list[str] = "auto",
        since: str | None = None,
        until: str | None = None,
    ) -> dict:
        """Search ALL the team's internal information in one call: chat history, shared files and
        attachments, wiki pages, receipts, tasks, meetings and minutes, and saved places, limited
        to what the asker may see. Returns the best ~10 hits with source, date, who, a one-line
        snippet and a link. Thai and English both work (the query is expanded with synonyms, e.g.
        ใบเสนอราคา also finds "quotation"). Use for "where is...", "who said...", "find the file
        about...", "เคยคุยเรื่อง... ที่ไหน". Then answer from the hits and give the links; call
        open_file / wiki_get for details. Not for the internet (use web_search).

        Args:
            query: what to find, in the user's words (a few words, not a sentence).
            sources: "auto" (all) or a list from: chat, files, wiki, receipts, tasks, meetings, places.
            since: optional start: YYYY-MM-DD, "last week", "30 days", "เมื่อวาน".
            until: optional end date YYYY-MM-DD.
        """
        log.append(("search_everything", {"sources": sources if sources != "auto" else "auto"}))
        try:
            return search_everything(
                src, expand, query, sources=sources, since=since, until=until, now=now(), zone=zone
            )
        except (TypeError, ValueError) as e:
            return {"error": str(e)}

    search_everything_tool.__name__ = "search_everything"
    search_everything_tool.salid_group = "search"  # type: ignore[attr-defined]
    search_everything_tool.salid_sources = available  # type: ignore[attr-defined]
    return [search_everything_tool]


SEARCH_PROMPT = (
    "\nเลือกเครื่องมือค้นหา: คำถามเรื่องภายในทีม (เคยคุยเรื่องนี้ที่ไหน ใครพูด ไฟล์/ใบเสนอราคา/สัญญาอยู่ไหน) "
    "-> search_everything ครั้งเดียว (หรือเครื่องมือเฉพาะถ้ารู้ชัด เช่น ใบเสร็จ -> ledger, ขั้นตอน -> wiki_search) "
    "แล้วตอบพร้อมลิงก์ของผลลัพธ์; ข้อมูลล่าสุด/ข่าว/ราคา/เรื่องนอกทีม -> web_search (อ้างอิงแหล่งที่มาเป็นเลข [1] [2] "
    "และลิงก์); ผู้ใช้ขอ 'ทำรีเสิร์ช/หาข้อมูลเชิงลึก/เปรียบเทียบ/วิเคราะห์ตลาด' -> deep_research (ทำเบื้องหลัง "
    "ส่งรายงานและ PDF ให้ทีหลัง บอกผู้ใช้ว่าใช้เวลาประมาณ 1-2 นาที). "
    "ห้ามแต่งข้อมูลหรือแหล่งอ้างอิง ถ้าไม่พบให้บอกตรงๆ. ห้ามใส่ข้อมูลภายในของทีม (ความจำ บัญชี แชต) ใน query ของ web_search."
)
_ = math  # (math kept for future scoring tweaks)
