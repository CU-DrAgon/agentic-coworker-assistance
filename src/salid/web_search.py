"""`web_search`: an explicit Google-Search-grounded answer with numbered citations.

Built on the same Vertex client and the same rules as `read_web` (see `web_read.py`): a SEPARATE
`generate_content` call that carries only the built-in `google_search` tool, so Gemini can ground
the answer while our servers never fetch a URL. Only the model-written query (capped) leaves the
building; chat memory, the ledger and other team data must not be put in it. Only domains are
logged.

The answer carries `[1]`, `[2]` markers placed from Gemini's grounding supports, and the result lists
the numbered sources (title, domain, link) so Salid can cite them. Today's date in Bangkok is part of
the request so "latest" means now.
"""

from __future__ import annotations

import contextlib
import logging
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

from google.genai import types

from . import llm
from .web_read import domain

logger = logging.getLogger(__name__)
QUERY_CHARS = 400
MAX_OUTPUT_TOKENS = 4096
MAX_SOURCES = 12
RECENCY = {"day": 1, "week": 7, "month": 31}
_SITE_RE = re.compile(r"^[a-z0-9][a-z0-9.\-]*\.[a-z]{2,}$")
BANGKOK = ZoneInfo("Asia/Bangkok")


def clean_site(site: str | None) -> str:
    """A bare domain, or "" (anything else, such as a URL with a path or a query, is dropped)."""
    s = (site or "").strip().casefold()
    s = re.sub(r"^https?://", "", s).split("/")[0].removeprefix("www.")
    return s if _SITE_RE.match(s) else ""


def today_bangkok(now: datetime | None = None) -> date:
    return (now or datetime.now(BANGKOK)).astimezone(BANGKOK).date()


def instruction(today: date, recency: str | None, site: str) -> str:
    parts = [
        f"Today is {today.isoformat()} (Bangkok, UTC+7). Answer the question with Google Search "
        "grounding. Treat 'latest', 'current', 'today' and 'this year' relative to this date, and "
        "prefer sources published close to it; state the date of a fact when it matters.",
        "Be concise and factual in the question's language. Do not invent facts, figures, quotes "
        "or sources; say clearly when the results do not answer the question or disagree.",
    ]
    if recency in RECENCY:
        days = RECENCY[recency]
        parts.append(
            f"Only use information published in the last {days} day(s); say so if nothing that "
            "recent was found."
        )
    if site:
        parts.append(f"Only use pages from {site}.")
    return " ".join(parts)


def search_config(today: date, recency: str | None, site: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=instruction(today, recency, site),
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.2,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def build_query(query: str, site: str) -> str:
    q = " ".join((query or "").split())[:QUERY_CHARS]
    return f"{q} site:{site}" if site and q else q


def _source_title(web) -> tuple[str, str]:
    """(title, domain). Vertex puts the site's domain in `title` and a redirect in `uri`."""
    uri = getattr(web, "uri", None) or ""
    title = (getattr(web, "title", None) or "").strip()
    host = title.casefold().removeprefix("www.") if _SITE_RE.match(title.casefold()) else ""
    host = host or domain(uri)
    if host.startswith("xn--") or ".xn--" in host:  # internationalised domain: show it readable
        with contextlib.suppress(UnicodeError):
            host = host.encode("ascii").decode("idna")
    return (title if title and not host.startswith("xn--") and "xn--" not in title else host, host)


def extract_sources(resp) -> list[dict]:
    """Numbered sources in grounding-chunk order (deduplicated by link)."""
    out: list[dict] = []
    seen: dict[str, int] = {}
    cand = next(iter(getattr(resp, "candidates", None) or []), None)
    gm = getattr(cand, "grounding_metadata", None)
    for ch in getattr(gm, "grounding_chunks", None) or []:
        web = getattr(ch, "web", None)
        uri = getattr(web, "uri", None)
        if not uri or uri in seen:
            continue
        title, host = _source_title(web)
        seen[uri] = len(out) + 1
        out.append({"n": len(out) + 1, "title": title, "domain": host, "link": uri})
    return out[:MAX_SOURCES]


def add_citations(text: str, resp) -> str:
    """Put [n] after each grounded segment. Gemini's indices are UTF-8 byte offsets."""
    cand = next(iter(getattr(resp, "candidates", None) or []), None)
    gm = getattr(cand, "grounding_metadata", None)
    chunks = list(getattr(gm, "grounding_chunks", None) or [])
    supports = getattr(gm, "grounding_supports", None) or []
    if not text or not chunks or not supports:
        return text
    # chunk index -> source number (same dedupe as extract_sources)
    number: dict[int, int] = {}
    seen: dict[str, int] = {}
    for i, ch in enumerate(chunks):
        uri = getattr(getattr(ch, "web", None), "uri", None)
        if not uri:
            continue
        number[i] = seen.setdefault(uri, len(seen) + 1)
    raw = text.encode("utf-8")
    inserts: list[tuple[int, str]] = []
    for s in supports:
        seg = getattr(s, "segment", None)
        end = getattr(seg, "end_index", None)
        ids = sorted(
            {number[i] for i in getattr(s, "grounding_chunk_indices", None) or [] if i in number}
        )
        if end is None or not ids or end > len(raw):
            continue
        inserts.append((int(end), "".join(f"[{n}]" for n in ids[:3])))
    for end, mark in sorted(inserts, key=lambda x: x[0], reverse=True):
        raw = raw[:end] + mark.encode() + raw[end:]
    return raw.decode("utf-8", errors="replace")


def web_search_call(
    client,
    model: str,
    query: str,
    *,
    recency: str | None = None,
    site: str | None = None,
    today: date | None = None,
) -> dict:
    """Blocking. {answer, sources:[{n,title,domain,link}], date, query} or {error}."""
    recency = recency if recency in RECENCY else None
    host = clean_site(site)
    q = build_query(query, host)
    if not q:
        return {"error": "give a search query"}
    day = today or today_bangkok()
    try:
        resp = llm.generate(
            client,
            model,
            [types.Content(role="user", parts=[types.Part(text=q)])],
            search_config(day, recency, host),
        )
    except Exception as e:  # never log the query
        logger.warning("web_search failed (%s)", type(e).__name__)
        return {"error": f"web search failed ({type(e).__name__})"}
    text = (getattr(resp, "text", None) or "").strip()
    sources = extract_sources(resp)
    if not text:
        return {"error": "the search returned no text", "sources": sources}
    logger.info(
        "web_search: %d source(s), domains=%s",
        len(sources),
        sorted({s["domain"] for s in sources if s["domain"]}),
    )
    return {
        "query": q,
        "date": day.isoformat(),
        "answer": add_citations(text, resp),
        "sources": sources,
        "how_to_cite": "cite as [n] with the title and link from `sources`; never add sources of your own",
    }


def format_sources(sources: list[dict]) -> str:
    """Numbered source lines for a report: `[1] Title - domain - link`."""
    return "\n".join(
        f"[{s['n']}] {s['title']} - {s['domain']} - {s['link']}"
        if s.get("domain")
        else f"[{s['n']}] {s['title']} - {s['link']}"
        for s in sources
    )


def make_web_search(client, model: str, calls: list | None = None, now=None):
    """The `web_search` agent tool (a plain function, like `read_web`)."""
    log = calls if calls is not None else []

    def web_search(query: str, recency: str | None = None, site: str | None = None) -> dict:
        """Search the internet with Google Search and get an answer with NUMBERED citations (title,
        domain, link). Use for current facts, news, prices, rules, competitors, anything outside
        the team; "latest" is relative to today in Bangkok. To read a specific link use read_web;
        for team data use search_everything. Never put chat, ledger or other team data in `query`:
        it goes to Google. Cite the sources by number and link in your answer.

        Args:
            query: what to find, in the user's language (max 400 characters).
            recency: optional "day", "week" or "month": only fresh information.
            site: optional domain to restrict to, such as "bot.or.th".
        """
        log.append(("web_search", {"recency": recency or "", "site": clean_site(site)}))
        return web_search_call(
            client,
            model,
            query,
            recency=recency,
            site=site,
            today=today_bangkok(now() if now else None),
        )

    return web_search
