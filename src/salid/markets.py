"""Stock, gold and news tools, watchlist alerts, the morning market line and company news.

Free public sources only, no keys:

* quotes / history: Yahoo Finance chart endpoint (delayed ~15-20 min, unofficial: it can rate-limit
  or block a server IP, then every tool says `unavailable` and the agent falls back to read_web);
* Thai gold: the Gold Traders Association "classic" page (the new site renders prices in the
  browser, so its HTML has none); world gold / USD-THB: Yahoo `GC=F` / `THB=X`;
* news: Google News RSS (titles, sources, times and links only; no article scraping).

Everything takes an injectable `get(url) -> str` so tests never touch the network. Numbers are
never invented: a failed source gives `{"status": "unavailable", ...}`. Pure logic and the agent
tools live here; the Discord glue is `markets_bot.py`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from urllib.parse import quote, quote_plus
from zoneinfo import ZoneInfo

import httpx
from google.genai import types

from . import llm
from .i18n import CATALOG, t

logger = logging.getLogger(__name__)

WATCHES = "salid_market_watch"  # one doc per alert condition (guild, user)
STATE = "salid_market_state"  # per guild: gold baseline of the day
TOPICS = "salid_market_topics"  # per guild: company / competitor news topics

TIMEOUT_S = 6.0
QUOTE_TTL_S = 60.0
NEWS_TTL_S = 600.0
FAIL_TTL_S = 30.0  # a failed source is not retried for a moment (rate limits)
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
YAHOO_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
GOLD_URL = "https://classic.goldtraders.or.th/"
BKK = ZoneInfo("Asia/Bangkok")
NY = ZoneInfo("America/New_York")

MAX_WATCH_SYMBOLS_PER_GUILD = 50
MAX_WATCHES_PER_USER = 20
MAX_TOPICS_PER_GUILD = 10
POLL_S = 600.0
GOLD_POLL_S = 1800.0
HISTORY_RANGES = ("1mo", "6mo", "1y")
SOURCE_YAHOO = "Yahoo Finance"
SOURCE_GOLD = "สมาคมค้าทองคำ (Gold Traders Association)"
SOURCE_NEWS = "Google News"
DELAY_NOTE = "ราคาอาจล่าช้า 15-20 นาที / may be delayed 15-20 minutes"
UNAVAILABLE_HINT = (
    "Source unreachable. Say so; do not guess any number. You may offer to look it up with read_web "
    "(Google Search)."
)

# ------------------------------------------------------------------ symbols
# Common Thai names -> SET tickers (the model may also pass a ticker directly).
THAI_NAMES: dict[str, str] = {
    "ปตท": "PTT", "ปตท.": "PTT", "ปตท สผ": "PTTEP", "ปตท.สผ.": "PTTEP", "ปตทสผ": "PTTEP",
    "ซีพี ออลล์": "CPALL", "ซีพีออลล์": "CPALL", "เซเว่น": "CPALL",
    "เอไอเอส": "ADVANC", "แอดวานซ์": "ADVANC", "ทรู": "TRUE", "ดีแทค": "DTAC",
    "กสิกร": "KBANK", "กสิกรไทย": "KBANK", "ไทยพาณิชย์": "SCB", "เอสซีบี": "SCB",
    "กรุงเทพ": "BBL", "แบงก์กรุงเทพ": "BBL", "ธนาคารกรุงเทพ": "BBL", "กรุงไทย": "KTB",
    "กรุงศรี": "BAY", "ทีเอ็มบีธนชาต": "TTB", "ทีทีบี": "TTB",
    "ปูนซิเมนต์ไทย": "SCC", "เอสซีจี": "SCC", "ปูนใหญ่": "SCC",
    "ท่าอากาศยานไทย": "AOT", "ทอท": "AOT", "การบินไทย": "THAI",
    "ไทยออยล์": "TOP", "ไออาร์พีซี": "IRPC", "บางจาก": "BCP", "บ้านปู": "BANPU",
    "ซีพีเอ็น": "CPN", "เซ็นทรัล พัฒนา": "CPN", "ซีพี แอ็กซ์ตร้า": "CPAXT",
    "โฮมโปร": "HMPRO", "โกลบอล เฮ้าส์": "GLOBAL", "ไมเนอร์": "MINT",
    "เดลต้า": "DELTA", "เดลต้า อิเล็คโทรนิคส์": "DELTA", "กัลฟ์": "GULF", "บีทีเอส": "BTS",
    "แลนด์แอนด์เฮ้าส์": "LH", "ซีพีเอฟ": "CPF", "เจริญโภคภัณฑ์อาหาร": "CPF",
    "ไทยยูเนี่ยน": "TU", "โรงพยาบาลบำรุงราษฎร์": "BH", "กรุงเทพดุสิต": "BDMS", "บีดีเอ็มเอส": "BDMS",
}  # fmt: skip
SET_TICKERS = frozenset(THAI_NAMES.values()) | {"BEM", "EA", "OR", "KTC", "TISCO", "MTC", "SAWAD"}

# name -> candidate Yahoo symbols in order of preference
INDEXES: dict[str, tuple[str, ...]] = {
    "set": ("^SET.BK",), "set index": ("^SET.BK",), "ดัชนี set": ("^SET.BK",),
    "ดัชนีหุ้นไทย": ("^SET.BK",), "เซ็ต": ("^SET.BK",), "หุ้นไทย": ("^SET.BK",),
    "set50": ("^SET50.BK",), "set 50": ("^SET50.BK",), "เซ็ต50": ("^SET50.BK",),
    "set100": ("^SET100.BK",), "set 100": ("^SET100.BK",),
    "mai": ("^MAI.BK",),
    "s&p 500": ("^GSPC",), "s&p500": ("^GSPC",), "s&p": ("^GSPC",), "sp500": ("^GSPC",),
    "nasdaq": ("^IXIC",), "แนสแด็ก": ("^IXIC",), "นาสแด็ก": ("^IXIC",),
    "dow": ("^DJI",), "dow jones": ("^DJI",), "ดาวโจนส์": ("^DJI",),
    "nikkei": ("^N225",), "nikkei 225": ("^N225",), "นิกเคอิ": ("^N225",),
    "hang seng": ("^HSI",), "ฮั่งเส็ง": ("^HSI",), "ฮั่งเซ็ง": ("^HSI",),
    "ftse": ("^FTSE",), "dax": ("^GDAXI",), "shanghai": ("000001.SS",), "เซี่ยงไฮ้": ("000001.SS",),
    "kospi": ("^KS11",), "sti": ("^STI",),
    "gold": ("GC=F",), "xau": ("GC=F",), "ทองโลก": ("GC=F",), "ทองคำโลก": ("GC=F",),
    "oil": ("CL=F",), "น้ำมันดิบ": ("CL=F",), "brent": ("BZ=F",),
    "usd/thb": ("THB=X",), "usdthb": ("THB=X",), "ดอลลาร์": ("THB=X",), "ค่าเงินบาท": ("THB=X",),
    "bitcoin": ("BTC-USD",), "btc": ("BTC-USD",), "บิตคอยน์": ("BTC-USD",),
}  # fmt: skip
_TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.\-=^&]{0,14}$")
_US_INDEXES = {"^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX"}
GOLD_WORDS = ("ทอง", "gold", "ทองคำ")


class MarketError(Exception):
    """A source could not be read (network, rate limit, bad payload)."""


def candidates(text: str, market: str = "auto") -> list[str]:
    """Yahoo symbols to try for what the user said, best first ([] = unknown name).
    market: auto | th | global (us)."""
    raw = " ".join(str(text or "").split())
    if not raw:
        return []
    key = raw.casefold().strip(" .")
    if key in INDEXES:
        return list(INDEXES[key])
    if raw in THAI_NAMES or raw.strip(" .") in THAI_NAMES:
        return [(THAI_NAMES.get(raw) or THAI_NAMES[raw.strip(" .")]) + ".BK"]
    sym = raw.upper().replace(" ", "")
    if not _TICKER.match(sym):
        return []
    if sym.endswith(".BK") or sym.startswith("^") or "=" in sym:
        return [sym]
    m = (market or "auto").casefold()
    if m in ("th", "set", "thai"):
        return [f"{sym}.BK"]
    if m in ("global", "us", "world"):
        return [sym]
    if sym in SET_TICKERS:
        return [f"{sym}.BK", sym]
    return [sym, f"{sym}.BK"]  # unknown ticker: as given first, then a Thai listing


def resolve_names(symbols: str | list[str]) -> list[str]:
    """Split "PTT, AAPL และ ทอง" style input into items."""
    if isinstance(symbols, list):
        items = [str(s) for s in symbols]
    else:
        items = re.split(r"[,;\n]|\s+และ\s+|\s+and\s+|\s*&\s+", str(symbols or ""))
    return [i.strip() for i in items if i and i.strip()][:8]


def market_kind(symbol: str) -> str:
    """Trading-hours family of a symbol: set | us | gold | global."""
    s = symbol.upper()
    if s == "GOLD":
        return "gold"
    if s.endswith(".BK") or s.startswith("^SET") or s.startswith("^MAI"):
        return "set"
    if "=" in s or "-USD" in s:
        return "global"
    if s in _US_INDEXES or (s.isalpha() and len(s) <= 5) or re.fullmatch(r"[A-Z]{1,5}[-.][A-Z]", s):
        return "us"
    return "global"


def is_open(kind: str, now: datetime) -> bool:
    """Is it worth polling this market now? SET Mon-Fri 10:00-16:30 Bangkok; US 09:30-16:00 New
    York; Thai gold announcements Mon-Sat 09:00-17:30 Bangkok; other markets on weekdays."""
    if kind == "set":
        b = now.astimezone(BKK)
        return b.weekday() < 5 and (10, 0) <= (b.hour, b.minute) <= (16, 30)
    if kind == "us":
        n = now.astimezone(NY)
        return n.weekday() < 5 and (9, 30) <= (n.hour, n.minute) <= (16, 0)
    if kind == "gold":
        b = now.astimezone(BKK)
        return b.weekday() < 6 and (9, 0) <= (b.hour, b.minute) <= (17, 30)
    return now.astimezone(UTC).weekday() < 5


# ------------------------------------------------------------------ parsing
def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def _rd(v: float | None, n: int = 4) -> float | None:
    return None if v is None else round(v, n)


def parse_yahoo_chart(payload: str | dict) -> dict:
    """The fields we use from a Yahoo chart response; raises MarketError when there is no data."""
    try:
        data = json.loads(payload) if isinstance(payload, str | bytes) else payload
        chart = data["chart"]
        if chart.get("error"):
            raise MarketError(str((chart["error"] or {}).get("description") or "no data"))
        res = chart["result"][0]
        meta = res["meta"]
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise MarketError("unexpected payload") from e
    ts = res.get("timestamp") or []
    q = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    closes = q.get("close") or []
    offset = int(meta.get("gmtoffset") or 0)
    zone = UTC
    bars: list[tuple[date, float, float | None, float | None]] = []
    highs, lows = q.get("high") or [], q.get("low") or []
    for i, stamp in enumerate(ts):
        c = _num(closes[i]) if i < len(closes) else None
        if c is None or not isinstance(stamp, int | float):
            continue
        d = (datetime.fromtimestamp(stamp, zone) + timedelta(seconds=offset)).date()
        hi = _num(highs[i]) if i < len(highs) else None
        lo = _num(lows[i]) if i < len(lows) else None
        bars.append((d, c, hi, lo))
    price = _num(meta.get("regularMarketPrice"))
    if price is None and bars:
        price = bars[-1][1]
    if price is None:
        raise MarketError("no price")
    mt = meta.get("regularMarketTime")
    local_day = (
        (datetime.fromtimestamp(mt, UTC) + timedelta(seconds=offset)).date()
        if isinstance(mt, int | float)
        else (bars[-1][0] if bars else None)
    )
    prev = None
    for d, c, _, _ in reversed(bars):  # the last close before the market's current day
        if local_day is None or d < local_day:
            prev = c
            break
    if prev is None:
        prev = _num(meta.get("previousClose")) or _num(meta.get("chartPreviousClose"))
    return {
        "symbol": meta.get("symbol", ""),
        "name": meta.get("shortName") or meta.get("longName") or meta.get("symbol", ""),
        "price": price,
        "previous_close": prev,
        "currency": meta.get("currency", ""),
        "exchange": meta.get("fullExchangeName") or meta.get("exchangeName", ""),
        "market_time": (
            datetime.fromtimestamp(mt, UTC).isoformat(timespec="minutes")
            if isinstance(mt, int | float)
            else ""
        ),
        "bars": bars,
    }


def _quote_result(p: dict) -> dict:
    price, prev = p["price"], p["previous_close"]
    chg = None if prev in (None, 0) else price - prev
    pct = None if chg is None else chg / prev * 100
    trend = [{"date": d.isoformat(), "close": _rd(c, 4)} for d, c, _, _ in p["bars"]][-5:]
    first = p["bars"][0][1] if p["bars"] else None
    out = {
        "status": "ok",
        "symbol": p["symbol"],
        "name": p["name"],
        "price": _rd(price),
        "previous_close": _rd(prev),
        "change": _rd(chg),
        "change_pct": _rd(pct, 2),
        "currency": p["currency"],
        "exchange": p["exchange"],
        "market_time_utc": p["market_time"],
        "trend_5d": trend,
        "trend_5d_pct": _rd((price - first) / first * 100, 2) if first else None,
        "source": SOURCE_YAHOO,
        "note": DELAY_NOTE,
    }
    return out


def history_summary(p: dict, rng: str) -> dict:
    bars = p["bars"]
    if len(bars) < 2:
        raise MarketError("not enough data")
    start, end = bars[0][1], bars[-1][1]
    hi = max(bars, key=lambda b: b[2] if b[2] is not None else b[1])
    lo = min(bars, key=lambda b: b[3] if b[3] is not None else b[1])
    return {
        "status": "ok",
        "symbol": p["symbol"],
        "name": p["name"],
        "range": rng,
        "from": bars[0][0].isoformat(),
        "to": bars[-1][0].isoformat(),
        "start": _rd(start),
        "end": _rd(end),
        "change_pct": _rd((end - start) / start * 100, 2) if start else None,
        "high": _rd(hi[2] if hi[2] is not None else hi[1]),
        "high_date": hi[0].isoformat(),
        "low": _rd(lo[3] if lo[3] is not None else lo[1]),
        "low_date": lo[0].isoformat(),
        "points": len(bars),
        "currency": p["currency"],
        "source": SOURCE_YAHOO,
        "note": DELAY_NOTE,
        "chart": "not available (no charting library installed)",
    }


_TAGS = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.S | re.I)
_GOLD_TIME = re.compile(
    r"(\d{1,2})/(\d{1,2})/(\d{4})\s*เวลา\s*(\d{1,2}):(\d{2})\s*น\.?\s*(?:\(\s*ครั้งที่\s*(\d+)\s*\))?"
)
_PRICE = r"([\d,]+(?:\.\d+)?)"


def parse_gold_page(html: str) -> dict:
    """Thai gold price announcement from the Gold Traders Association page text: bar 96.5% sell /
    buy, ornament 96.5% sell / tax base, time and round. Raises MarketError when it is not there."""
    text = " ".join(_TAGS.sub(" ", html or "").replace("&nbsp;", " ").replace("\xa0", " ").split())

    def after(label: str, word: str) -> float | None:
        m = re.search(rf"{label}\s*96\.5\s*%?\s*{word}\s*{_PRICE}", text)
        return _num(m.group(1).replace(",", "")) if m else None

    bar_sell = after("ทองคำแท่ง", "ขายออก")
    bar_buy = None
    m = re.search(rf"ทองคำแท่ง\s*96\.5\s*%?\s*ขายออก\s*{_PRICE}\s*รับซื้อ\s*{_PRICE}", text)
    if m:
        bar_sell, bar_buy = (_num(m.group(i).replace(",", "")) for i in (1, 2))
    om = re.search(rf"ทองรูปพรรณ\s*96\.5\s*%?\s*ขายออก\s*{_PRICE}(?:\s*ฐานภาษี\s*{_PRICE})?", text)
    orn_sell = _num(om.group(1).replace(",", "")) if om else None
    orn_tax = _num(om.group(2).replace(",", "")) if om and om.group(2) else None
    if bar_sell is None or bar_buy is None:
        raise MarketError("gold prices not found")
    tm = _GOLD_TIME.search(text)
    announced, rnd, label = "", None, ""
    if tm:
        d, mo, y, hh, mi = (int(tm.group(i)) for i in range(1, 6))
        y = y - 543 if y > 2400 else y  # Buddhist era
        try:
            announced = datetime(y, mo, d, hh, mi, tzinfo=BKK).isoformat(timespec="minutes")
        except ValueError:
            announced = ""
        rnd = int(tm.group(6)) if tm.group(6) else None
        label = tm.group(0).strip()
    return {
        "status": "ok",
        "bar_sell": bar_sell,
        "bar_buy": bar_buy,
        "ornament_sell": orn_sell,
        "ornament_tax_base": orn_tax,
        "unit": "THB per baht-weight (บาทละ)",
        "purity": "96.5%",
        "announced_at": announced,
        "announced_text": label,
        "round": rnd,
        "source": SOURCE_GOLD,
        "url": GOLD_URL,
    }


_SUFFIX = re.compile(r"\s+[-–|]\s+[^-–|]{2,60}$")


def _norm_title(title: str) -> str:
    t = _SUFFIX.sub("", title or "")
    return re.sub(r"[\W_]+", "", t.casefold())


def parse_news(xml_text: str, limit: int = 5) -> list[dict]:
    """Items of a Google News RSS feed with near-identical titles merged (first wins)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise MarketError("bad RSS") from e
    out: list[dict] = []
    seen: list[str] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not title or not link:
            continue
        src_el = item.find("source")
        source = (src_el.text or "").strip() if src_el is not None else ""
        if not source:
            m = _SUFFIX.search(title)
            source = m.group(0).lstrip(" -–|") if m else ""
        headline = _SUFFIX.sub("", title).strip() if source and title.endswith(source) else title
        key = _norm_title(headline)
        if not key or any(key == s or SequenceMatcher(None, key, s).ratio() >= 0.85 for s in seen):
            continue
        seen.append(key)
        when = ""
        with contextlib.suppress(TypeError, ValueError):
            when = (
                parsedate_to_datetime(item.findtext("pubDate") or "")
                .astimezone(UTC)
                .isoformat(timespec="minutes")
            )
        out.append({"title": headline, "source": source, "time_utc": when, "link": link})
        if len(out) >= limit:
            break
    return out


TOPIC_CODES = {
    "business": "BUSINESS", "ธุรกิจ": "BUSINESS", "เศรษฐกิจ": "BUSINESS", "การเงิน": "BUSINESS",
    "หุ้น": "BUSINESS", "technology": "TECHNOLOGY", "tech": "TECHNOLOGY", "เทคโนโลยี": "TECHNOLOGY",
    "world": "WORLD", "ต่างประเทศ": "WORLD", "nation": "NATION", "ในประเทศ": "NATION",
    "science": "SCIENCE", "วิทยาศาสตร์": "SCIENCE", "health": "HEALTH", "สุขภาพ": "HEALTH",
    "sports": "SPORTS", "sport": "SPORTS", "กีฬา": "SPORTS",
    "entertainment": "ENTERTAINMENT", "บันเทิง": "ENTERTAINMENT",
}  # fmt: skip


def news_url(query: str, topic: str, lang: str) -> str:
    th = lang != "en"
    loc = "hl=th&gl=TH&ceid=TH:th" if th else "hl=en-US&gl=US&ceid=US:en"
    q = (query or "").strip()
    if q:
        return f"https://news.google.com/rss/search?q={quote_plus(q)}&{loc}"
    code = TOPIC_CODES.get((topic or "").strip().casefold())
    if code:
        return f"https://news.google.com/rss/headlines/section/topic/{code}?{loc}"
    return f"https://news.google.com/rss?{loc}"


# ------------------------------------------------------------------ data access
def http_get(url: str, timeout: float = TIMEOUT_S) -> str:
    try:
        r = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "th,en;q=0.8"},
        )
    except httpx.HTTPError as e:
        raise MarketError(type(e).__name__) from e
    if r.status_code != 200:
        raise MarketError(f"http {r.status_code}")
    return r.text


def unavailable(what: str, why: str) -> dict:
    return {"status": "unavailable", "what": what, "reason": why, "hint": UNAVAILABLE_HINT}


SOURCE_GROUNDED = "Google Search (grounded)"
BREAKER_S = 600.0  # Yahoo is skipped this long after a 429
GROUNDED_POLL_PER_HOUR = 30  # alert polling budget per process
GROUNDED_ASK_PER_HOUR = 60  # interactive budget per process
GROUNDED_PROMPT = (
    "You look up current market prices on the web with Google Search. Reply with ONLY a JSON "
    "array, no prose and no code fence, one object per item below, in this shape: "
    '{{"symbol": str, "name": str, "price": number, "currency": str, "change": number or null, '
    '"change_pct": number or null, "as_of": "exchange local date and time of the price, e.g. '
    '2026-10-16 10:30 ICT", "source_title": str, "source_url": str}}. '
    "Use only numbers you actually found on a page; never estimate or calculate a price. If you "
    "cannot find an item, omit it. Prices may be delayed. Items:\n{items}"
)
GROUNDED_HISTORY_PROMPT = (
    "You look up price history on the web with Google Search. Reply with ONLY a JSON array with one "
    'object: {{"symbol": str, "name": str, "currency": str, "start": number, "end": number, '
    '"high": number, "low": number, "change_pct": number or null, "from": "YYYY-MM-DD", "to": '
    '"YYYY-MM-DD", "as_of": str, "source_title": str, "source_url": str}} for the period '
    "below. Use only numbers found on a page; never estimate. If you cannot find them reply []. "
    "Prices may be delayed. Item: {item}, period: {period}"
)
SEARCH_LABELS = {
    "GC=F": "gold futures GC=F, price in USD per troy ounce",
    "THB=X": "USD/THB exchange rate (Thai baht per 1 US dollar)",
    "CL=F": "WTI crude oil futures CL=F in USD per barrel",
    "^SET.BK": "SET Index (Stock Exchange of Thailand main index)",
    "^SET50.BK": "SET50 Index (Thailand)",
    "^GSPC": "S&P 500 index",
    "^IXIC": "Nasdaq Composite index",
    "^N225": "Nikkei 225 index",
}
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}[/.]\d{1,2}[/.]\d{2,4}")
_RANGE_TEXT = {"1mo": "the last 1 month", "6mo": "the last 6 months", "1y": "the last 1 year"}


def _json_array(text: str) -> list[dict]:
    t0 = (text or "").strip()
    for a, b in (("[", "]"), ("{", "}")):
        i, j = t0.find(a), t0.rfind(b)
        if i != -1 and j > i:
            try:
                v = json.loads(t0[i : j + 1])
            except ValueError:
                continue
            return [x for x in (v if isinstance(v, list) else [v]) if isinstance(x, dict)]
    return []


class Grounded:
    """Last-resort prices from Gemini with Google Search grounding (the read_web mechanism),
    validated strictly: a result needs a positive price and a time, else it is dropped."""

    def __init__(self, client, model: str, clock: Callable[[], float] = time.monotonic):
        self.client, self.model, self._clock = client, model, clock
        self._used: dict[str, list[float]] = {"poll": [], "ask": []}
        self._lock = threading.Lock()

    def _allow(self, kind: str) -> bool:
        cap = GROUNDED_POLL_PER_HOUR if kind == "poll" else GROUNDED_ASK_PER_HOUR
        now = self._clock()
        with self._lock:
            used = [x for x in self._used[kind] if now - x < 3600]
            self._used[kind] = used
            if len(used) >= cap:
                return False
            used.append(now)
            return True

    def _call(self, prompt: str, kind: str) -> tuple[list[dict], list[dict]]:
        if self.client is None or not self._allow(kind):
            raise MarketError("grounded search unavailable")
        cfg = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.0,
            max_output_tokens=2048,
        )
        try:
            resp = llm.generate(
                self.client,
                self.model,
                [types.Content(role="user", parts=[types.Part(text=prompt)])],
                cfg,
            )
        except Exception as e:
            raise MarketError(f"grounded {type(e).__name__}") from e
        from .web_read import _sources

        return _json_array(getattr(resp, "text", "") or ""), _sources(resp)[0]

    @staticmethod
    def _src(row: dict, chunks: list[dict]) -> dict:
        return {
            "source_title": str(row.get("source_title") or (chunks[0]["title"] if chunks else ""))[
                :120
            ],
            "source_url": str(row.get("source_url") or (chunks[0]["uri"] if chunks else ""))[:300],
        }

    def quotes(self, items: list[tuple[str, str]], kind: str = "ask") -> dict[str, dict]:
        """items: (key, search label). One search call for all of them. Returns key -> quote
        result (same shape as Yahoo's, minus the trend) for the items that validated."""
        if not items:
            return {}
        rows, chunks = self._call(
            GROUNDED_PROMPT.format(items="\n".join(f"- {label}" for _, label in items)), kind
        )
        out: dict[str, dict] = {}
        for row in rows:
            price, as_of = _num(row.get("price")), str(row.get("as_of") or "").strip()
            if price is None or price <= 0 or not _DATE.search(as_of):
                continue
            key = self._match(row, items, out)
            if key is None:
                continue
            chg, pct = _num(row.get("change")), _num(row.get("change_pct"))
            if pct is None and chg is not None and price - chg:
                pct = chg / (price - chg) * 100
            out[key] = {
                "status": "ok",
                "symbol": str(row.get("symbol") or key),
                "name": str(row.get("name") or ""),
                "price": _rd(price),
                "previous_close": _rd(price - chg) if chg is not None else None,
                "change": _rd(chg),
                "change_pct": _rd(pct, 2),
                "currency": str(row.get("currency") or ""),
                "as_of": as_of,
                "market_time_utc": "",
                "source": SOURCE_GROUNDED,
                "confidence": "web",
                "note": DELAY_NOTE + "; from a web search, verify before relying on it",
                **self._src(row, chunks),
            }
        return out

    @staticmethod
    def _match(row: dict, items, taken: dict) -> str | None:
        sym = re.sub(r"[^A-Z0-9]", "", str(row.get("symbol") or "").upper())
        name = str(row.get("name") or "").casefold()
        for key, label in items:
            if key in taken:
                continue
            k = re.sub(r"[^A-Z0-9]", "", key.upper().removesuffix(".BK"))
            if sym and (sym == k or sym == re.sub(r"[^A-Z0-9]", "", key.upper())):
                return key
            if name and label.casefold() in name:
                return key
        free = [k for k, _ in items if k not in taken]
        return free[0] if len(items) == 1 and free else None

    def history(self, label: str, rng: str) -> dict | None:
        rows, chunks = self._call(
            GROUNDED_HISTORY_PROMPT.format(item=label, period=_RANGE_TEXT.get(rng, rng)), "ask"
        )
        for row in rows:
            vals = [_num(row.get(k)) for k in ("start", "end", "high", "low")]
            if any(v is None or v <= 0 for v in vals) or not str(row.get("as_of") or "").strip():
                continue
            start, end, high, low = vals
            pct = _num(row.get("change_pct"))
            return {
                "status": "ok", "symbol": str(row.get("symbol") or label), "name": str(row.get("name") or ""),
                "range": rng, "from": str(row.get("from") or ""), "to": str(row.get("to") or ""),
                "start": start, "end": end, "high": high, "low": low,
                "change_pct": _rd(pct if pct is not None else (end - start) / start * 100, 2),
                "currency": str(row.get("currency") or ""), "as_of": str(row["as_of"]),
                "source": SOURCE_GROUNDED, "confidence": "web",
                "note": DELAY_NOTE + "; from a web search, verify before relying on it",
                **self._src(row, chunks),
            }  # fmt: skip
        return None


class MarketData:
    """The sources behind the tools, with a 60 s quote cache and a 10 min news cache."""

    def __init__(
        self,
        get: Callable[[str], str] = http_get,
        clock: Callable[[], float] = time.monotonic,
        grounded: Grounded | None = None,
    ):
        self._get, self._clock, self.grounded = get, clock, grounded
        self._blocked_until = -1e12  # Yahoo circuit breaker (set after a 429)
        self._cache: dict[str, tuple[float, float, object]] = {}  # key -> (at, ttl, value)
        self._lock = threading.Lock()

    def _cached(self, key: str, ttl: float, load: Callable[[], object]):
        now = self._clock()
        with self._lock:
            hit = self._cache.get(key)
        if hit and now - hit[0] < hit[1]:
            if isinstance(hit[2], MarketError):
                raise hit[2]
            return hit[2]
        try:
            value = load()
        except MarketError as e:
            with self._lock:
                self._cache[key] = (now, FAIL_TTL_S, e)
            raise
        with self._lock:
            self._cache[key] = (now, ttl, value)
            if len(self._cache) > 400:  # drop expired entries
                self._cache = {k: v for k, v in self._cache.items() if now - v[0] < v[1]}
        return value

    def _chart(self, symbol: str, rng: str, interval: str) -> dict:
        def load():
            if self._clock() < self._blocked_until:
                raise MarketError("yahoo paused after http 429")
            err: MarketError | None = None
            for host in YAHOO_HOSTS:
                url = (
                    f"https://{host}/v8/finance/chart/{quote(symbol, safe='')}"
                    f"?range={rng}&interval={interval}"
                )
                try:
                    return parse_yahoo_chart(self._get(url))
                except MarketError as e:
                    err = e
                    if "429" in str(e):  # circuit breaker: leave Yahoo alone for a while
                        self._blocked_until = self._clock() + BREAKER_S
                        break
                    if str(e) in ("no data", "unexpected payload") or "Not Found" in str(e):
                        break  # the symbol is wrong, not the host
            raise err or MarketError("unreachable")

        return self._cached(f"y:{symbol}:{rng}:{interval}", QUOTE_TTL_S, load)  # type: ignore[return-value]

    def _peek(self, key: str):
        with self._lock:
            hit = self._cache.get(key)
        return hit[2] if hit and self._clock() - hit[0] < hit[1] else None

    def _put(self, key: str, ttl: float, value) -> None:
        with self._lock:
            self._cache[key] = (self._clock(), ttl, value)

    def quotes(self, inputs: list[str], market: str = "auto", poll: bool = False) -> list[dict]:
        """Quotes for several names, in order. Yahoo first; whatever it cannot give is looked up
        together in ONE grounded Google Search call; else `unavailable`."""
        out: list[dict | None] = [None] * len(inputs)
        pending: dict[str, list[int]] = {}  # first candidate -> positions
        labels: dict[str, str] = {}
        whys: dict[str, str] = {}
        for i, text in enumerate(inputs):
            names = candidates(text, market)
            if not names:
                out[i] = {
                    "status": "unknown_symbol",
                    "input": text,
                    "hint": "Ask for the ticker, e.g. PTT, AAPL, or ^GSPC.",
                }
                continue
            last: MarketError | None = None
            for sym in names:
                try:
                    out[i] = {**_quote_result(self._chart(sym, "5d", "1d")), "input": text}
                    break
                except MarketError as e:
                    last = e
            if out[i] is None:
                key = names[0]
                hit = self._peek(f"g:{key}")
                if hit is not None:
                    out[i] = {**hit, "input": text} if hit else None
                    if not hit:
                        out[i] = {**unavailable(text, str(last)), "input": text}
                else:
                    pending.setdefault(key, []).append(i)
                    labels[key] = SEARCH_LABELS.get(key) or (
                        text if text.isascii() and text.upper() == key else f"{text} ({key})"
                    )
                    whys[key] = str(last)
        if pending:
            found: dict[str, dict] = {}
            why = "; ".join(sorted(set(whys.values())))
            if self.grounded is not None:
                try:
                    found = self.grounded.quotes(
                        [(k, labels[k]) for k in pending], "poll" if poll else "ask"
                    )
                except MarketError as e:
                    why = f"{why}; {e}"
                    logger.info("grounded lookup failed: %s", e)
            if self.grounded is not None and not found and "grounded" not in why:
                why += "; grounded search found no valid price"
            for key, positions in pending.items():
                res = found.get(key)
                self._put(f"g:{key}", QUOTE_TTL_S if res else FAIL_TTL_S, res or {})
                for i in positions:
                    out[i] = (
                        {**res, "input": inputs[i]}
                        if res
                        else {**unavailable(inputs[i], why), "input": inputs[i]}
                    )
        return out  # type: ignore[return-value]

    def quote(self, symbol: str, market: str = "auto") -> dict:
        """One quote for a name or ticker (Yahoo, then grounded search, then unavailable)."""
        return self.quotes([symbol], market)[0]

    def history(self, symbol: str, rng: str = "1mo", market: str = "auto") -> dict:
        if rng not in HISTORY_RANGES:
            rng = "1mo"
        names = candidates(symbol, market)
        if not names:
            return {"status": "unknown_symbol", "input": symbol}
        last: MarketError | None = None
        for sym in names:
            try:
                return {**history_summary(self._chart(sym, rng, "1d"), rng), "input": symbol}
            except MarketError as e:
                last = e
        if self.grounded is not None:
            key = f"gh:{names[0]}:{rng}"
            res = self._peek(key)
            if res is None:
                try:
                    res = self.grounded.history(f"{symbol} ({names[0]})", rng) or {}
                except MarketError as e:
                    res, last = {}, e
                self._put(key, QUOTE_TTL_S if res else FAIL_TTL_S, res)
            if res:
                return {**res, "input": symbol}
        return {**unavailable(symbol, str(last)), "input": symbol}

    def thai_gold(self) -> dict:
        try:
            return self._cached(
                "gold:th", QUOTE_TTL_S, lambda: parse_gold_page(self._get(GOLD_URL))
            )  # type: ignore[return-value]
        except MarketError as e:
            return unavailable("thai gold", str(e))

    def gold(self) -> dict:
        world, fx = self.quotes(["GC=F", "THB=X"])
        return {
            "thai_gold": self.thai_gold(),
            "world_gold_usd_per_oz": world,
            "usd_thb": fx,
            "note": "Thai prices are the association announcement; world gold is a futures quote.",
        }

    def news(self, query: str = "", topic: str = "", lang: str = "th", limit: int = 5) -> dict:
        lang = "en" if str(lang).casefold().startswith("en") else "th"
        limit = max(1, min(int(limit or 5), 10))
        url = news_url(query, topic, lang)
        try:
            items = self._cached(
                f"n:{url}", NEWS_TTL_S, lambda: parse_news(self._get(url), limit=10)
            )
        except MarketError as e:
            return unavailable("news", str(e))
        return {
            "status": "ok",
            "query": query or topic or "top",
            "items": list(items)[:limit],  # type: ignore[arg-type]
            "source": SOURCE_NEWS,
            "note": "Headlines only; use read_web on a link to summarise an article.",
        }


# ------------------------------------------------------------------ watchlist
CONDITIONS = ("above", "below", "up_pct", "down_pct", "move_pct")
_COND_ALIASES = {
    "over": "above", "more": "above", "gt": "above", "สูงกว่า": "above", "เกิน": "above",
    "under": "below", "less": "below", "lt": "below", "ต่ำกว่า": "below", "ต่ำ": "below",
    "up": "up_pct", "rise": "up_pct", "ขึ้น": "up_pct", "down": "down_pct", "fall": "down_pct",
    "ลง": "down_pct", "move": "move_pct", "change": "move_pct", "เคลื่อนไหว": "move_pct",
}  # fmt: skip


@dataclass
class Watch:
    id: str
    user_id: int
    symbol: str  # Yahoo symbol or "gold" (Thai bar gold sell price)
    label: str
    condition: str
    threshold: float
    created_at: str = ""
    fired_day: str = ""

    def to_doc(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_doc(cls, d: dict) -> Watch:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def triggered(condition: str, threshold: float, price: float | None, pct: float | None) -> bool:
    if condition == "above":
        return price is not None and price >= threshold
    if condition == "below":
        return price is not None and price <= threshold
    if pct is None:
        return False
    if condition == "up_pct":
        return pct >= threshold
    if condition == "down_pct":
        return pct <= -threshold
    return abs(pct) >= threshold  # move_pct


class WatchRepo:
    def __init__(self, docs):
        self.docs = docs

    def all(self, guild_id: int) -> list[Watch]:
        return [Watch.from_doc(d) for d in self.docs.find(WATCHES, guild_id)]

    def of_user(self, guild_id: int, user_id: int) -> list[Watch]:
        return [w for w in self.all(guild_id) if w.user_id == user_id]

    def add(self, guild_id: int, w: Watch) -> str:
        """ "" on success, else an error code (caps)."""
        mine = self.all(guild_id)
        if sum(1 for x in mine if x.user_id == w.user_id) >= MAX_WATCHES_PER_USER:
            return "too_many_for_user"
        symbols = {x.symbol for x in mine}
        if w.symbol not in symbols and len(symbols) >= MAX_WATCH_SYMBOLS_PER_GUILD:
            return "too_many_symbols_in_server"
        if any(
            (x.user_id, x.symbol, x.condition, x.threshold)
            == (w.user_id, w.symbol, w.condition, w.threshold)
            for x in mine
        ):
            return "duplicate"
        self.docs.put(WATCHES, guild_id, w.id, w.to_doc())
        return ""

    def remove(self, guild_id: int, user_id: int, symbol: str | None, wid: str = "") -> int:
        n = 0
        for w in self.of_user(guild_id, user_id):
            if (wid and w.id == wid) or (not wid and symbol and w.symbol == symbol):
                self.docs.delete(WATCHES, guild_id, w.id)
                n += 1
        return n

    def mark_fired(self, guild_id: int, w: Watch, day: str) -> None:
        w.fired_day = day
        self.docs.put(WATCHES, guild_id, w.id, w.to_doc())


@dataclass
class Alert:
    guild_id: int
    user_id: int
    watch: Watch
    price: float | None
    change_pct: float | None
    currency: str
    as_of: str
    source: str


class GoldBaseline:
    """Thai gold has no "previous close" in the announcement, so the move % is measured against
    the last price of the previous day we saw (else the first price seen today)."""

    def __init__(self, docs):
        self.docs = docs

    def pct(self, guild_id: int, price: float, day: str) -> float | None:
        d = self.docs.get(STATE, guild_id, "gold") or {}
        if d.get("day") != day:
            d = {"day": day, "base": d.get("last") or price, "last": price}
        d["last"] = price
        self.docs.put(STATE, guild_id, "gold", d)
        base = d["base"]
        return (price - base) / base * 100 if base else None


class Poller:
    """Evaluates watchlists. Symbols are fetched once per round across guilds and members; each
    market is polled only in its hours (SET / US 10 min, Thai gold 30 min)."""

    def __init__(
        self, repo: WatchRepo, data: MarketData, docs, clock: Callable[[], float] = time.monotonic
    ):
        self.repo, self.data, self.baseline = repo, data, GoldBaseline(docs)
        self._clock = clock
        self._last: dict[str, float] = {}

    @staticmethod
    def _snap(q: dict) -> tuple | None:
        if q.get("status") != "ok":
            return None
        return (
            q["price"], q["change_pct"], q["currency"],
            q.get("market_time_utc") or q.get("as_of", ""), q.get("source", SOURCE_YAHOO),
        )  # fmt: skip

    def snapshot(self, symbol: str) -> tuple | None:
        """(price, change %, currency, as_of, source) or None when no source has it."""
        if symbol == "gold":
            g = self.data.thai_gold()
            if g.get("status") != "ok":
                return None
            return g["bar_sell"], None, "THB/บาททอง", g.get("announced_at", ""), SOURCE_GOLD
        return self._snap(self.data.quote(symbol))

    def due(self, guild_ids: list[int], now: datetime) -> list[Alert]:
        """Alerts that fire now (each condition at most once per Bangkok day). Blocking. Quotes
        come from the same chain as the tools (Yahoo, then ONE batched grounded search call for
        everything Yahoo missed, capped per hour)."""
        today = now.astimezone(BKK).date().isoformat()
        by_guild = {g: [w for w in self.repo.all(g) if w.fired_day != today] for g in guild_ids}
        wanted = {w.symbol for ws in by_guild.values() for w in ws}
        t0 = self._clock()
        snaps: dict[str, tuple | None] = {}
        quote_syms: list[str] = []
        for sym in sorted(wanted):
            kind = market_kind(sym)
            if not is_open(kind, now):
                continue
            every = GOLD_POLL_S if kind == "gold" else POLL_S
            if t0 - self._last.get(sym, -1e9) < every:
                continue
            self._last[sym] = t0
            if sym == "gold":
                snaps[sym] = self.snapshot(sym)
            else:
                quote_syms.append(sym)
        if quote_syms:
            for sym, q in zip(quote_syms, self.data.quotes(quote_syms, poll=True), strict=True):
                snaps[sym] = self._snap(q)
                if snaps[sym] is None:
                    logger.info(
                        "market poll failed: %s (%s)", sym, q.get("reason", q.get("status"))
                    )
        out: list[Alert] = []
        for gid, ws in by_guild.items():
            for w in ws:
                snap = snaps.get(w.symbol)
                if snap is None:
                    continue
                price, pct, cur, as_of, src = snap
                if w.symbol == "gold" and price is not None:
                    pct = self.baseline.pct(gid, price, today)
                if triggered(w.condition, w.threshold, price, pct):
                    self.repo.mark_fired(gid, w, today)
                    out.append(Alert(gid, w.user_id, w, price, pct, cur, as_of, src))
        return out


def format_alert(a: Alert, lang: str = "th") -> str:
    w = a.watch
    cond = t_(f"mk_cond_{w.condition}", lang, thr=f"{w.threshold:g}")
    chg = "" if a.change_pct is None else f" ({a.change_pct:+.2f}%)"
    when = ""
    if a.as_of:
        try:
            when = datetime.fromisoformat(a.as_of).astimezone(BKK).strftime("%d/%m %H:%M")
        except ValueError:
            when = a.as_of
    return t_(
        "mk_alert", lang, label=w.label, cond=cond, price=f"{a.price:,.2f}", cur=a.currency,
        chg=chg, src=a.source, when=when,
    )  # fmt: skip


def t_(key: str, lang: str, **fmt) -> str:
    return t(key, lang, **fmt)


# ------------------------------------------------------------------ brief line / company news
DEFAULT_BRIEF_ITEMS = ("set", "gold")


def market_line(data: MarketData, items, lang: str, today: date | None = None) -> str:
    """ "ตลาดเช้านี้: SET …, ทอง …" from live data; "" when nothing could be read."""
    parts: list[str] = []
    for item in items or DEFAULT_BRIEF_ITEMS:
        key = str(item).strip()
        if key.casefold() in ("gold", "ทอง", "ทองคำ"):
            g = data.thai_gold()
            if g.get("status") == "ok":
                parts.append(
                    t_("mk_line_gold", lang, sell=f"{g['bar_sell']:,.0f}", buy=f"{g['bar_buy']:,.0f}",
                       time=(g["announced_at"][11:16] if g.get("announced_at") else "-"))
                )  # fmt: skip
            continue
        q = data.quote(key)
        if q.get("status") != "ok":
            continue
        label = (q.get("input") or q["symbol"]).upper() if key.isascii() else key
        day = ""
        if q.get("market_time_utc") and today is not None:
            d = datetime.fromisoformat(q["market_time_utc"]).astimezone(BKK).date()
            day = "" if d == today else f" {d:%d/%m}"
        pct = "" if q["change_pct"] is None else f" ({q['change_pct']:+.2f}%)"
        parts.append(f"{label} {q['price']:,.2f}{pct}{day}")
    return t_("mk_line", lang, parts=", ".join(parts)) if parts else ""


def company_news(data: MarketData, topics: list[str], lang: str, total: int = 3) -> str:
    """Up to `total` headlines about the guild's tracked topics, labelled; "" when none."""
    rows: list[tuple[str, dict]] = []
    seen: list[str] = []
    for topic in topics[:MAX_TOPICS_PER_GUILD]:
        res = data.news(query=topic, lang=lang, limit=2)
        if res.get("status") != "ok":
            continue
        for it in res["items"]:
            key = _norm_title(it["title"])
            if key and not any(SequenceMatcher(None, key, s).ratio() >= 0.85 for s in seen):
                seen.append(key)
                rows.append((topic, it))
                break  # one per topic first, so every topic gets a turn
    lines = [
        f"• [{topic}] {it['title']}"
        + (f" — {it['source']}" if it["source"] else "")
        + f"\n  <{it['link']}>"
        for topic, it in rows[:total]
    ]
    return t_("mk_news_head", lang) + "\n" + "\n".join(lines) if lines else ""


# ------------------------------------------------------------------ agent tools
@dataclass
class MarketsCtx:
    guild_id: int
    asker_id: int
    is_exec: bool
    data: MarketData
    repo: WatchRepo
    docs: object
    prefs: object | None = None  # briefings.PrefsRepo
    lang: str = "th"
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    on_prefs_changed: Callable[[], None] = field(default=lambda: None)


def _cond(raw: str) -> str:
    c = str(raw or "").strip().casefold().replace("-", "_").replace(" ", "_")
    return c if c in CONDITIONS else _COND_ALIASES.get(c, "")


def markets_tools(ctx: MarketsCtx) -> list[Callable]:
    """The functions `_agent_kwargs` appends to `extra_tools` (docstrings = tool descriptions)."""

    def stock_quote(symbols: str, market: str = "auto") -> dict:
        """Latest price of stocks, indexes, FX or commodities, from Yahoo Finance (delayed about
        15-20 minutes). Use for "ราคาหุ้น PTT", "SET วันนี้", "NVDA เท่าไหร่", "S&P 500", "Nikkei",
        "ค่าเงินบาท". Returns price, previous close, change %, currency, market time and a
        5-day trend per symbol. Never state a price that this tool did not return; if a symbol
        is `unavailable`, say so. In the answer give the source and time, and add a short "not
        investment advice" once.

        Args:
            symbols: one or several names or tickers separated by commas: Thai names (ปตท, กสิกร),
                SET tickers (PTT, KBANK), indexes (SET, SET50, S&P 500, Nasdaq, Nikkei) or global
                tickers (AAPL, NVDA).
            market: "auto" (default), "th" (force a SET listing) or "global".
        """
        names = resolve_names(symbols)
        if not names:
            return {"status": "error", "error": "no_symbol"}
        return {"quotes": ctx.data.quotes(names, market)}

    def stock_history(symbol: str, range: str = "1mo", market: str = "auto") -> dict:  # noqa: A002
        """Price history summary of one symbol: start, end, % change, high and low with dates over
        1 month, 6 months or 1 year (Yahoo Finance). Use for "หุ้น PTT 6 เดือนที่ผ่านมาเป็นยังไง".
        Quote the numbers only as returned, with the source, and note it is not investment advice
        once.

        Args:
            symbol: name or ticker, as for stock_quote.
            range: "1mo", "6mo" or "1y".
            market: "auto", "th" or "global".
        """
        return ctx.data.history(symbol, range, market)

    def gold_price() -> dict:
        """Today's gold prices: Thai gold bar and ornament 96.5% (buy/sell per baht-weight,
        announcement time and round) from the Gold Traders Association, plus world gold in
        USD/oz and USD/THB from Yahoo Finance. Use for "ราคาทองวันนี้", "gold price". Give the
        source and the announcement time; never invent a figure; if a part is `unavailable`, say
        so. Add a short "not investment advice" once.
        """
        return ctx.data.gold()

    def news(query: str = "", topic: str = "", lang: str = "th", max: int = 5) -> dict:  # noqa: A002
        """Latest news headlines from Google News: title, source, time and link (no article
        text). Use for "ข่าวล่าสุดเรื่อง …", "ข่าวหุ้น", "news about …". To summarise an article the
        asker names, call read_web on its link. Do not claim anything that is not in the
        headlines.

        Args:
            query: search words (company, person, subject). Leave empty for topic headlines.
            topic: when no query: business, technology, world, nation, science, health, sports,
                entertainment (Thai names work); empty = top stories.
            lang: "th" (default) or "en".
            max: number of headlines, 1-10 (default 5).
        """
        return ctx.data.news(query, topic, lang, max)

    def market_watch(
        action: str, symbol: str = "", condition: str = "", value: float = 0.0
    ) -> dict:
        """Price alerts for the asker. "แจ้งเตือนถ้า PTT ต่ำกว่า 30" -> add, PTT, below, 30.
        "ทองขึ้นเกิน 1% บอกด้วย" -> add, gold, up_pct, 1. Salid checks during market hours and
        DMs once per condition per day. Confirm what was set in one short line.

        Args:
            action: "add", "remove" or "list".
            symbol: ticker or name (PTT, ปตท, AAPL, SET) or "gold" for the Thai gold bar sell
                price (the % is measured against the previous day's last announcement).
            condition: for add: "above" / "below" (price) or "up_pct" / "down_pct" / "move_pct"
                (change in percent versus the previous close, either direction for move_pct).
            value: the price or the percent.
        """
        gid, uid = ctx.guild_id, ctx.asker_id
        act = (action or "").strip().casefold()
        if act == "list":
            return {
                "watches": [
                    {"id": w.id, "label": w.label, "condition": w.condition, "value": w.threshold}
                    for w in ctx.repo.of_user(gid, uid)
                ]
            }
        raw = (symbol or "").strip()
        if not raw:
            return {"status": "error", "error": "symbol_required"}
        if raw.casefold() in GOLD_WORDS or raw in ("ทองแท่ง", "ทองคำแท่ง"):
            sym, label = "gold", "ทองคำแท่ง 96.5% (สมาคม)"
        else:
            cands = candidates(raw)
            if not cands:
                return {"status": "unknown_symbol", "input": raw}
            sym = cands[0]
            if len(cands) > 1:  # ambiguous: keep the one that has data
                for c in cands:
                    if ctx.data.quote(c).get("status") == "ok":
                        sym = c
                        break
            label = raw.upper() if raw.isascii() else raw
        if act in ("remove", "delete", "cancel"):
            n = ctx.repo.remove(gid, uid, sym)
            return {"status": "ok" if n else "not_found", "removed": n}
        if act != "add":
            return {"status": "error", "error": "action must be add, remove or list"}
        cond = _cond(condition)
        v = _num(value)
        if not cond or v is None or v <= 0:
            return {
                "status": "error",
                "error": "need condition above/below/up_pct/down_pct/move_pct and a positive value",
            }
        w = Watch(
            id=uuid.uuid4().hex[:10], user_id=uid, symbol=sym, label=label, condition=cond,
            threshold=v, created_at=ctx.now().isoformat(timespec="seconds"),
        )  # fmt: skip
        err = ctx.repo.add(gid, w)
        if err:
            return {"status": "refused", "reason": err}
        return {
            "status": "ok",
            "watching": label,
            "condition": cond,
            "value": v,
            "checked": "SET 10:00-16:30 / US hours / gold 09:00-17:30, once per day per condition",
        }

    def market_brief(on: bool = True, items: str = "") -> dict:
        """Add (or remove) a market line to the asker's own morning brief DM: "ตลาดเช้านี้: SET …,
        ทอง …". Use for "ส่งราคาทองทุกเช้าด้วย", "brief me on markets". Only for the asker.

        Args:
            on: true to include it, false to stop.
            items: comma-separated, default "SET, gold"; may contain tickers (PTT, NVDA, S&P 500).
        """
        if ctx.prefs is None:
            return {"status": "error", "error": "briefings_unavailable"}
        p = ctx.prefs.get(ctx.guild_id, ctx.asker_id)
        p.markets = bool(on)
        if items.strip():
            p.market_items = resolve_names(items)[:6]
        elif on and not p.market_items:
            p.market_items = list(DEFAULT_BRIEF_ITEMS)
        ctx.prefs.save(ctx.guild_id, ctx.asker_id, p)
        ctx.on_prefs_changed()
        return {"status": "ok", "market_line_in_morning_brief": p.markets, "items": p.market_items}

    def track_news_topic(action: str, topic: str = "") -> dict:
        """Executives/admins only: follow news about a customer or competitor. The evening report
        then shows up to 3 labelled headlines. Use for "ติดตามข่าวลูกค้า ABC", "ติดตามข่าวคู่แข่ง X",
        "เลิกติดตาม X", "ตอนนี้ติดตามข่าวอะไรอยู่".

        Args:
            action: "add", "remove" or "list".
            topic: the company or subject name (as searched in the news).
        """
        if not ctx.is_exec:
            return {"status": "refused", "reason": "executives_only"}
        gid = ctx.guild_id
        key = re.sub(r"\s+", " ", (topic or "").strip())[:60]
        act = (action or "").strip().casefold()
        topics = topic_list(ctx.docs, gid)
        if act == "list":
            return {"topics": topics}
        if not key:
            return {"status": "error", "error": "topic_required"}
        slug = _norm_title(key) or uuid.uuid4().hex[:8]
        if act == "add":
            if len(topics) >= MAX_TOPICS_PER_GUILD and key not in topics:
                return {"status": "refused", "reason": "too_many_topics"}
            ctx.docs.put(TOPICS, gid, slug, {"topic": key, "added_by": ctx.asker_id})
            return {"status": "ok", "tracking": key}
        if act in ("remove", "delete", "stop"):
            ctx.docs.delete(TOPICS, gid, slug)
            return {"status": "ok", "removed": key}
        return {"status": "error", "error": "action must be add, remove or list"}

    return [
        stock_quote,
        stock_history,
        gold_price,
        news,
        market_watch,
        market_brief,
        track_news_topic,
    ]


def topic_list(docs, guild_id: int) -> list[str]:
    return [d["topic"] for d in docs.find(TOPICS, guild_id) if d.get("topic")]


MARKETS_PROMPT = (
    "\nตลาดหุ้น ทอง และข่าว: ราคาหุ้น/ดัชนี/ทอง/ค่าเงิน/ข่าวล่าสุด → ใช้ stock_quote / stock_history / gold_price / news "
    "เท่านั้น ห้ามเดาหรือใช้ตัวเลขจากความจำเด็ดขาด. ถ้าเครื่องมือคืน unavailable ให้บอกตรงๆ ว่าตอนนี้ดึงข้อมูลไม่ได้ "
    "(เสนอค้นให้ทางเว็บด้วย read_web ถ้ามี). ทุกคำตอบเรื่องราคาต้องบอกแหล่งที่มาและเวลา (ราคาหุ้นอาจล่าช้า 15-20 นาที) "
    "และปิดท้ายสั้นๆ ว่า “ไม่ใช่คำแนะนำการลงทุน” เพียงครั้งเดียวต่อคำตอบ; ห้ามชี้นำให้ซื้อ/ขาย. "
    "ข่าว: ให้หัวข้อ แหล่ง เวลา ลิงก์ ถ้าต้องสรุปเนื้อข่าวให้เรียก read_web กับลิงก์นั้น. "
    "ตั้งแจ้งเตือนราคา (“แจ้งเตือนถ้า PTT ต่ำกว่า 30”, “ทองขึ้นเกิน 1% บอกด้วย”) ด้วย market_watch; "
    "“ส่งราคาทองทุกเช้า”/“brief me on markets” ใช้ market_brief; "
    "ผู้บริหารสั่งให้ติดตามข่าวลูกค้า/คู่แข่งได้ด้วย track_news_topic."
)

STRINGS: dict[str, dict[str, str]] = {
    "mk_cond_above": {"th": "ราคาขึ้นถึง {thr}", "en": "price reached {thr}"},
    "mk_cond_below": {"th": "ราคาลงมาที่ {thr}", "en": "price fell to {thr}"},
    "mk_cond_up_pct": {"th": "ขึ้นเกิน {thr}%", "en": "up more than {thr}%"},
    "mk_cond_down_pct": {"th": "ลงเกิน {thr}%", "en": "down more than {thr}%"},
    "mk_cond_move_pct": {"th": "ขยับเกิน {thr}%", "en": "moved more than {thr}%"},
    "mk_alert": {
        "th": "🔔 {label}: {cond} — ตอนนี้ {price} {cur}{chg} · ที่มา {src} {when} · ไม่ใช่คำแนะนำการลงทุนค่ะ",
        "en": "🔔 {label}: {cond} — now {price} {cur}{chg} · source {src} {when} · not investment advice.",
    },
    "mk_line": {
        "th": "📈 ตลาดเช้านี้: {parts} (ที่มา Yahoo Finance / สมาคมค้าทองคำ อาจล่าช้า · ไม่ใช่คำแนะนำการลงทุน)",
        "en": "📈 Markets this morning: {parts} (source Yahoo Finance / Gold Traders Association, may be delayed; not investment advice)",
    },
    "mk_line_gold": {
        "th": "ทองแท่ง {sell}/{buy} บาท (ประกาศ {time})",
        "en": "gold bar {sell}/{buy} THB (announced {time})",
    },
    "mk_news_head": {
        "th": "📰 ข่าวที่ติดตาม (หัวข้อจาก Google News ยังไม่ได้ตรวจสอบ):",
        "en": "📰 Followed news (Google News headlines, not verified):",
    },
}
CATALOG.update(STRINGS)
