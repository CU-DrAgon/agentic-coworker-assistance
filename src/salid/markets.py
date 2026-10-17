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
import statistics
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

from .currency import Converter
from .i18n import CATALOG, has_thai, t
from .web_read import grounded_generate

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
WORLD_GOLD = "GOLD:WORLD"  # watch symbol: world spot XAU/USD (the default "gold" is Thai)
GOLD_UPDATES_URL = "https://classic.goldtraders.or.th/UpdatePriceList.aspx"
GOLD_API_URL = "https://api.gold-api.com/price/XAU"
FX_URL = "https://api.frankfurter.dev/v1/latest?base=USD&symbols=THB"
BAHT_GRAMS = 15.244  # one Thai baht-weight of gold
TROY_OZ_GRAMS = 31.1035
THAI_UNIT = "บาทต่อน้ำหนัก 1 บาททองคำ (15.244 กรัม) / THB per 1 baht-weight of gold (15.244 g)"
WORLD_UNIT = "USD ต่อทรอยออนซ์ (31.1035 กรัม) / USD per troy ounce (31.1035 g)"
SPOT_DISAGREE_PCT = 1.0
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
WORLD_TARGETS = ("world", "spot", "international", "global", "xau", "xauusd", "สากล", "ทองโลก", "ทองคำโลก", "ทองสากล", "ต่างประเทศ")  # fmt: skip


def gold_target(symbol: str) -> str:
    return "world" if symbol == WORLD_GOLD else "thai"


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
    if s == WORLD_GOLD:
        return "global"
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
        "unit": THAI_UNIT,
        "purity": "96.5%",
        "announced_at": announced,
        "announced_text": label,
        "round": rnd,
        "source": SOURCE_GOLD,
        "url": GOLD_URL,
    }


_UPDATE_ROW = re.compile(
    r"(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2})\s+(\d+)\s+"
    + r"\s+".join([_PRICE] * 6)
    + r"\s+([+\-\u2212]?\s*[\d,]+(?:\.\d+)?)"
)


def parse_gold_updates(html: str) -> dict:
    """The newest row of the association's intraday-changes page (UpdatePriceList.aspx): bar and
    ornament prices, the "Gold Spot" (USD/oz) and "Baht / US$" the association used, and the
    change versus the previous announcement. Raises MarketError when there is no row."""
    text = " ".join(_TAGS.sub(" ", html or "").replace("&nbsp;", " ").replace("\xa0", " ").split())
    best = None
    for m in _UPDATE_ROW.finditer(text):
        d, mo, y, hh, mi, rnd = (int(m.group(i)) for i in range(1, 7))
        y = y - 543 if y > 2400 else y
        try:
            at = datetime(y, mo, d, hh, mi, tzinfo=BKK)
        except ValueError:
            continue
        if best is None or at > best[0]:
            best = (at, rnd, m)
    if best is None:
        raise MarketError("gold updates not found")
    at, rnd, m = best
    v = [_num(m.group(i).replace(",", "")) for i in range(7, 13)]
    chg = _num(m.group(13).replace(",", "").replace("\u2212", "-").replace(" ", ""))
    return {
        "announced_at": at.isoformat(timespec="minutes"),
        "round": rnd,
        "bar_buy": v[0], "bar_sell": v[1], "ornament_buy": v[2], "ornament_sell": v[3],
        "spot_usd_oz": v[4], "usd_thb": v[5],
        "change_baht": chg,
    }  # fmt: skip


def parse_gold_api(payload: str | dict) -> dict:
    """gold-api.com /price/XAU: spot USD per troy ounce and when it was updated (UTC)."""
    try:
        d = json.loads(payload) if isinstance(payload, str) else payload
        price = _num(d["price"])
    except (ValueError, KeyError, TypeError) as e:
        raise MarketError("unexpected payload") from e
    if price is None or price <= 0 or str(d.get("symbol", "XAU")).upper() != "XAU":
        raise MarketError("unexpected payload")
    if str(d.get("currency", "USD")).upper() != "USD":
        raise MarketError("unexpected payload")
    at = ""
    try:
        at = datetime.fromisoformat(str(d["updatedAt"]).replace("Z", "+00:00")).astimezone(UTC)
        at = at.isoformat(timespec="seconds")
    except (KeyError, ValueError):
        at = ""
    return {"price": price, "updated_at": at}


def parse_frankfurter(payload: str | dict) -> dict:
    try:
        d = json.loads(payload) if isinstance(payload, str) else payload
        rate = _num(d["rates"]["THB"])
    except (ValueError, KeyError, TypeError) as e:
        raise MarketError("unexpected payload") from e
    if rate is None or rate <= 0:
        raise MarketError("unexpected payload")
    return {"rate": rate, "date": str(d.get("date") or "")}


def thb_per_baht_weight(spot_usd_oz: float, usd_thb: float) -> float:
    """International spot converted to THB per baht-weight of 96.5% gold (no premiums)."""
    return spot_usd_oz * usd_thb / TROY_OZ_GRAMS * BAHT_GRAMS * 0.965


def combine_spots(sources: list[dict]) -> dict:
    """Median headline and spread over the spot sources that gave a number."""
    vals = sorted(s["value"] for s in sources)
    if not vals:
        return {"spot_usd_oz": None, "spread_pct": None, "disagree": False}
    med = statistics.median(vals)
    spread = (vals[-1] - vals[0]) / med * 100 if len(vals) > 1 else 0.0
    return {
        "spot_usd_oz": round(med, 2),
        "spread_pct": round(spread, 2),
        "disagree": len(vals) > 1 and spread > SPOT_DISAGREE_PCT,
    }


def _bkk(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone(BKK).strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return iso or "ไม่ระบุเวลา"


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
    th = lang == "th"
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


WEB_STATUSES = ("web_reference", "estimate")
WEB_AGREE_PCT = 0.5  # web sources within this spread of the median count as agreeing


def _src_line(x: dict, lang: str) -> str:
    miss = x.get("missing") or []
    link = f" {x['url']}" if x.get("url") else ""
    note = ""
    if miss:
        note = (" [missing: " if lang != "th" else " [ขาด: ") + ", ".join(miss) + "]"
    return f"{x['name']} {x['value']:,.2f} ({x['as_of']}){link}{note}"


def estimate_text(q: dict, lang: str = "en") -> str:
    """Labelled text for a web-read price. web_reference (sources agree, each with link and time):
    "ราคาอ้างอิงจาก N แหล่ง (อ่านจากหน้าเว็บ) ..."; estimate (they differ or lack a time/link):
    "ราคาโดยประมาณ ช่วง A–B ...". Every source is listed with its value, time and link."""
    cur = q.get("currency") or ""
    rg = q.get("range") or {}
    lo, hi = rg.get("low", q["price"]), rg.get("high", q["price"])
    n = q.get("n_sources", 1)
    parts = "; ".join(_src_line(x, lang) for x in q.get("sources", []))
    if q.get("status") == "web_reference":
        if lang != "th":
            return (
                f"Reference price {q['price']:,.2f} {cur} (median) from {n} web source(s), read from "
                f"web pages, not an official feed: {parts}."
            )
        return (
            f"ราคาอ้างอิงจาก {n} แหล่ง (อ่านจากหน้าเว็บ) {q['price']:,.2f} {cur} (ค่ากลาง) "
            f"ไม่ใช่ฟีดทางการ: {parts}"
        )
    if lang != "th":
        return (
            f"Estimate ~{q['price']:,.2f} {cur} (range {lo:,.2f}-{hi:,.2f}) from {n} web source(s), "
            f"official price unconfirmed: {parts}."
        )
    return (
        f"ราคาโดยประมาณ ~{q['price']:,.2f} {cur} (ช่วง {lo:,.2f}–{hi:,.2f}) จาก {n} แหล่ง "
        f"ยังยืนยันราคาทางการไม่ได้: {parts}"
    )


SOURCE_GROUNDED = "Google Search (grounded)"
WEB_NOTE = "อ่านจากหน้าเว็บ ไม่ใช่ฟีดทางการ/เรียลไทม์ / read from web pages, not an official feed"
ESTIMATE_NOTE = (
    "ราคาโดยประมาณจากการค้นเว็บ ความแม่นยำต่ำกว่า ยังยืนยันราคาทางการไม่ได้ / estimate from web "
    "search, lower accuracy, not the official price; say so and give the range and the sources"
)
BREAKER_S = 600.0  # Yahoo is skipped this long after a 429
GROUNDED_POLL_PER_HOUR = 30  # alert polling budget per process
GROUNDED_ASK_PER_HOUR = 60  # interactive budget per process
GROUNDED_PROMPT = (
    "You look up current market prices on the web with Google Search. Reply with ONLY a JSON "
    "array, no prose and no code fence, one object per item below, in this shape: "
    '{{"symbol": str, "name": str, "price": number, "currency": str, "change": number or null, '
    '"change_pct": number or null, "as_of": "exchange local date and time of the price, e.g. '
    '2026-10-16 10:30 ICT", "source_title": str, "source_url": str}}. '
    "Use only numbers you actually found on a page; never estimate or calculate a price. Give "
    "several rows per item when several different pages show it (one row per page). If the page "
    "shows no time, use an empty as_of. If you cannot find an item, omit it. Prices may be "
    "delayed. Items:\n{items}"
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
        try:
            resp = grounded_generate(
                self.client, self.model, prompt, temperature=0.0, max_output_tokens=2048
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
        """items: (key, search label). One search call for all of them. Returns key -> a web
        result (never "ok", which is for direct feeds): "web_reference" when every source has a
        value, name, link and time and they agree within 0.5%, else "estimate" with the range.
        The median is the headline; a missing time is kept ("ไม่ระบุเวลา"); only an item with no
        usable number is absent."""
        if not items:
            return {}
        rows, chunks = self._call(
            GROUNDED_PROMPT.format(items="\n".join(f"- {label}" for _, label in items)), kind
        )
        by_key: dict[str, list[dict]] = {}
        for row in rows:
            price = _num(row.get("price"))
            if price is None or price <= 0:
                continue
            key = self._match(row, items)
            if key is not None:
                by_key.setdefault(key, []).append(row)
        out: dict[str, dict] = {}
        for key, krows in by_key.items():
            srcs: list[dict] = []
            seen: set = set()
            for row in krows:
                price = _num(row.get("price"))
                src = self._src(row, chunks)
                dedupe = (src["source_url"] or src["source_title"], round(price, 4))
                if dedupe in seen:
                    continue
                seen.add(dedupe)
                as_of = str(row.get("as_of") or "").strip()
                srcs.append(
                    {
                        "name": src["source_title"] or src["source_url"] or "web",
                        "url": src["source_url"],
                        "value": _rd(price),
                        "as_of": as_of or "ไม่ระบุเวลา",
                        "time_approximate": not _DATE.search(as_of),
                        "missing": [
                            m
                            for m, bad in (
                                ("เวลา", not _DATE.search(as_of)),
                                ("ลิงก์", not src["source_url"]),
                            )
                            if bad
                        ],
                    }
                )
            vals = sorted(x["value"] for x in srcs)
            med = statistics.median(vals)
            first = krows[0]
            chg, pct = _num(first.get("change")), _num(first.get("change_pct"))
            price0 = _num(first.get("price"))
            if pct is None and chg is not None and price0 and price0 - chg:
                pct = chg / (price0 - chg) * 100
            spread = (vals[-1] - vals[0]) / med * 100 if med else 0.0
            agree = spread <= WEB_AGREE_PCT and not any(x["missing"] for x in srcs)
            out[key] = {
                "status": "web_reference" if agree else "estimate",
                "symbol": str(first.get("symbol") or key),
                "name": str(first.get("name") or ""),
                "price": _rd(med),
                "range": {"low": vals[0], "high": vals[-1]},
                "n_sources": len(srcs),
                "spread_pct": round(spread, 2),
                "sources": srcs,
                "previous_close": _rd(price0 - chg) if chg is not None and price0 else None,
                "change": _rd(chg),
                "change_pct": _rd(pct, 2),
                "currency": str(first.get("currency") or ""),
                "as_of": srcs[0]["as_of"],
                "market_time_utc": "",
                "source": SOURCE_GROUNDED,
                "confidence": "medium" if agree else "low",
                "note": WEB_NOTE if agree else ESTIMATE_NOTE,
                **self._src(first, chunks),
            }
            out[key]["estimate_text"] = estimate_text(out[key])
        return out

    @staticmethod
    def _match(row: dict, items) -> str | None:
        sym = re.sub(r"[^A-Z0-9]", "", str(row.get("symbol") or "").upper())
        name = str(row.get("name") or "").casefold()
        for key, label in items:
            k = re.sub(r"[^A-Z0-9]", "", key.upper().removesuffix(".BK"))
            if sym and (sym == k or sym == re.sub(r"[^A-Z0-9]", "", key.upper())):
                return key
            if name and label.casefold() in name:
                return key
        return items[0][0] if len(items) == 1 else None

    def history(self, label: str, rng: str) -> dict | None:
        rows, chunks = self._call(
            GROUNDED_HISTORY_PROMPT.format(item=label, period=_RANGE_TEXT.get(rng, rng)), "ask"
        )
        for row in rows:
            vals = [_num(row.get(k)) for k in ("start", "end", "high", "low")]
            if any(v is None or v <= 0 for v in vals):
                continue
            start, end, high, low = vals
            pct = _num(row.get("change_pct"))
            return {
                "status": "ok", "symbol": str(row.get("symbol") or label), "name": str(row.get("name") or ""),
                "range": rng, "from": str(row.get("from") or ""), "to": str(row.get("to") or ""),
                "start": start, "end": end, "high": high, "low": low,
                "change_pct": _rd(pct if pct is not None else (end - start) / start * 100, 2),
                "currency": str(row.get("currency") or ""), "as_of": str(row.get("as_of") or "").strip() or "ไม่ระบุเวลา",
                "source": SOURCE_GROUNDED, "confidence": "low", "note": ESTIMATE_NOTE,
                **self._src(row, chunks),
            }  # fmt: skip
        return None


GROUNDED_LABELS = {
    "XAU_SPOT": "gold SPOT price XAU/USD in USD per troy ounce (spot, NOT futures)",
    "TH_GOLD_BAR": "Thai gold bar 96.5% selling price (ทองคำแท่ง ขายออก) in THB per baht-weight "
    "(15.244 g), from the Gold Traders Association or Thai gold shops",
}


def gold_summary(g: dict, lang: str = "en") -> str:
    """Ready text with every unit spelled out, Thai section first, world section second."""
    en = lang != "th"
    th, w = g["thai"], g["world"]
    L: list[str] = []
    if th.get("status") == "ok":
        when = _bkk(th["announced_at"]) if th.get("announced_at") else "-"
        rnd = f" (ครั้งที่ {th['round']})" if th.get("round") else ""
        c = th.get("change")
        chg = ""
        if c and c.get("baht") is not None:
            chg = (
                f"; change {c['baht']:+,.0f} THB vs previous announcement ({c['direction']})"
                if en
                else f"; {c['direction']} {abs(c['baht']):,.0f} บาท จากประกาศครั้งก่อน"
            )
        orn = f"{th['ornament_buy']:,.2f}" if th.get("ornament_buy") else "-"
        if en:
            L.append(
                f"THAI GOLD (Gold Traders Association, 96.5%, THB per 1 baht-weight = 15.244 g), "
                f"announced {when} Bangkok time{rnd}: gold bar buy {th['bar_buy']:,.0f} / sell "
                f"{th['bar_sell']:,.0f}; ornament buy (tax base) {orn} / sell "
                f"{th['ornament_sell'] or 0:,.0f}{chg}."
            )
        else:
            L.append(
                f"ราคาทองไทย (สมาคมค้าทองคำ ทอง 96.5% หน่วยบาทต่อน้ำหนัก 1 บาททองคำ = 15.244 กรัม) "
                f"ประกาศ {when} น. เวลาไทย{rnd}: ทองคำแท่ง รับซื้อ {th['bar_buy']:,.0f} / ขายออก "
                f"{th['bar_sell']:,.0f} บาท; ทองรูปพรรณ รับซื้อ (ฐานภาษี) {orn} / ขายออก "
                f"{th['ornament_sell'] or 0:,.0f} บาท{chg}."
            )
    elif th.get("status") in WEB_STATUSES:
        est = {**th, "price": th["bar_sell"], "currency": "THB per baht-weight"}
        L.append(
            ("Thai gold bar sell (96.5%): " if en else "ทองคำแท่ง 96.5% ขายออก: ")
            + estimate_text(est, lang)
        )
    else:
        L.append("Thai gold: unavailable." if en else "ราคาทองไทย: ดึงข้อมูลไม่ได้ตอนนี้")
    wl: list[str] = []
    if w.get("spot_usd_oz"):
        srcs = "; ".join(f"{x['name']} {x['value']:,.2f} ({_bkk(x['as_of']) if x['as_of'][:2] == '20' else x['as_of']})" for x in w["spot_sources"])  # fmt: skip
        if w.get("spot_basis") in WEB_STATUSES:
            est = {"status": w["spot_basis"], "price": w["spot_usd_oz"], "range": w["spot_range"],
                   "n_sources": len(w["spot_sources"]), "sources": w["spot_sources"],
                   "currency": "USD per troy ounce (spot)"}  # fmt: skip
            wl.append(estimate_text(est, lang))
        else:
            head = "Spot XAU/USD" if en else "ราคาสปอต XAU/USD"
            wl.append(f"{head} {w['spot_usd_oz']:,.2f} USD per troy ounce"
                      + ("" if en else " (ดอลลาร์สหรัฐต่อทรอยออนซ์)") + f": {srcs}")  # fmt: skip
            if len(w["spot_sources"]) > 1:
                wl.append(
                    f"(median of {len(w['spot_sources'])} sources, spread {w['spread_pct']:.2f}%)"
                    if en
                    else f"(ค่ากลางจาก {len(w['spot_sources'])} แหล่ง ต่างกัน {w['spread_pct']:.2f}%)"
                )
            if w.get("disagree"):
                wl.append(
                    "Sources differ by more than 1%, treat with caution."
                    if en
                    else "แหล่งข้อมูลต่างกันเกิน 1% ใช้ด้วยความระมัดระวัง"
                )
    if w.get("futures_usd_oz"):
        wl.append(
            f"Futures (COMEX GC=F, not spot) {w['futures_usd_oz']:,.2f} USD per troy ounce"
            f" as of {_bkk(w['futures_as_of'])} Bangkok" if en else
            f"ฟิวเจอร์ส (COMEX GC=F ไม่ใช่สปอต) {w['futures_usd_oz']:,.2f} ดอลลาร์สหรัฐต่อทรอยออนซ์ "
            f"ณ {_bkk(w['futures_as_of'])} น. เวลาไทย"
        )  # fmt: skip
    if wl:
        L.append(("INTERNATIONAL GOLD: " if en else "ราคาทองสากล: ") + " ".join(wl))
    else:
        L.append("International gold: unavailable." if en else "ราคาทองสากล: ดึงข้อมูลไม่ได้ตอนนี้")
    cv = g.get("conversion")
    if cv and cv.get("status") in ("ok", "estimate"):
        L.append(
            f"Conversion: {cv['spot_usd_oz']:,.2f} USD/oz × {cv['usd_thb']:.2f} THB/USD "
            f"({cv['usd_thb_source']}) ÷ 31.1035 × 15.244 × 0.965 = about "
            f"{cv['thb_per_baht_weight_96_5']:,} THB per baht-weight (96.5%); the Thai "
            "association price includes premiums/margins so it differs."
            if en
            else f"คำนวณจากสปอต: {cv['spot_usd_oz']:,.2f} ดอลลาร์/ออนซ์ × {cv['usd_thb']:.2f} บาท/ดอลลาร์ "
            f"({cv['usd_thb_source']}) ÷ 31.1035 × 15.244 × 0.965 = ประมาณ "
            f"{cv['thb_per_baht_weight_96_5']:,} บาทต่อน้ำหนัก 1 บาททองคำ (96.5%) "
            "ราคาสมาคมค้าทองคำรวมพรีเมียม/ส่วนต่าง จึงต่างจากตัวเลขนี้"
        )
        if cv.get("usd_thb_ecb") and cv.get("fx_disagree"):
            L.append(
                f"USD/THB differs between sources: association {cv['usd_thb']:.2f}, "
                f"ECB {cv['usd_thb_ecb']:.2f}."
                if en
                else f"อัตรา USD/THB ต่างกันระหว่างแหล่ง: สมาคม {cv['usd_thb']:.2f}, "
                f"ECB {cv['usd_thb_ecb']:.2f}"
            )
    return "\n".join(L)


class MarketData:
    """The sources behind the tools, with a 60 s quote cache and a 10 min news cache."""

    def __init__(
        self,
        get: Callable[[str], str] = http_get,
        clock: Callable[[], float] = time.monotonic,
        grounded: Grounded | None = None,
        fx: Converter | None = None,
    ):
        self._get, self._clock, self.grounded = get, clock, grounded
        # the ONE currency source (salid.currency: Frankfurter/ECB, then open.er-api), here through
        # this object's own http getter so a test's fake network covers it
        self._fx = fx or Converter(fetch=lambda url: json.loads(self._get(url)), clock=clock)
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

    def _updates(self) -> dict | None:
        try:
            return self._cached(  # type: ignore[return-value]
                "gold:upd", QUOTE_TTL_S, lambda: parse_gold_updates(self._get(GOLD_UPDATES_URL))
            )
        except MarketError:
            return None

    def thai_gold(self) -> dict:
        """Thai gold (96.5%) from the association: the main page, joined with the newest row of its
        intraday-changes page (Gold Spot, baht rate, change vs the previous announcement)."""
        why = ""
        try:
            main: dict | None = self._cached(
                "gold:th", QUOTE_TTL_S, lambda: parse_gold_page(self._get(GOLD_URL))
            )  # type: ignore[assignment]
        except MarketError as e:
            main, why = None, str(e)
        upd = self._updates()
        if main is None and upd is None:
            return unavailable("thai gold", why or "no data")
        if main is None or (upd and upd["announced_at"] > main.get("announced_at", "")):
            base = {
                "status": "ok", "unit": THAI_UNIT, "purity": "96.5%", "source": SOURCE_GOLD,
                "url": GOLD_UPDATES_URL, "ornament_tax_base": upd["ornament_buy"],  # type: ignore[index]
                **{k: upd[k] for k in ("bar_buy", "bar_sell", "ornament_buy", "ornament_sell")},  # type: ignore[index]
                "announced_at": upd["announced_at"], "round": upd["round"],  # type: ignore[index]
            }  # fmt: skip
        else:
            base = {**main, "ornament_buy": main.get("ornament_tax_base")}
        base["unit"] = THAI_UNIT
        if upd and upd["announced_at"] == base.get("announced_at"):
            c = upd["change_baht"]
            base["change"] = {
                "baht": c,
                "direction": "ขึ้น" if c and c > 0 else "ลง" if c and c < 0 else "เท่าเดิม",
                "unit": "บาท ต่อน้ำหนัก 1 บาททองคำ เทียบกับประกาศครั้งก่อน",
            }
            base["gta_spot_usd_oz"] = upd["spot_usd_oz"]
            base["gta_usd_thb"] = upd["usd_thb"]
        else:
            base["change"] = None
            base["gta_spot_usd_oz"] = base["gta_usd_thb"] = None
        return base

    def _gold_api(self) -> dict | None:
        try:
            return self._cached(  # type: ignore[return-value]
                "gold:api", QUOTE_TTL_S, lambda: parse_gold_api(self._get(GOLD_API_URL))
            )
        except MarketError:
            return None

    def _frankfurter(self) -> dict | None:
        """USD/THB from currency.py (ECB reference rate), or None."""
        r = self._fx.rate("USD", "THB")
        return None if "error" in r else r

    def _futures(self) -> dict | None:
        """COMEX GC=F from Yahoo only (never the grounded fallback: futures must not be mixed in)."""
        try:
            q = _quote_result(self._chart("GC=F", "5d", "1d"))
        except MarketError:
            return None
        return {"value": q["price"], "as_of": q.get("market_time_utc") or ""}

    def _grounded_one(self, key: str, label: str) -> dict | None:
        if self.grounded is None:
            return None
        hit = self._peek(f"g:{key}")
        if hit is not None:
            return hit or None
        try:
            res = self.grounded.quotes([(key, label)]).get(key)
        except MarketError as e:
            logger.info("grounded %s failed: %s", key, e)
            res = None
        self._put(f"g:{key}", QUOTE_TTL_S if res else FAIL_TTL_S, res or {})
        return res

    def _spot_sources(self, thai: dict) -> list[dict]:
        """Exact spot XAU/USD sources that answered, each with its own value and time."""
        sources: list[dict] = []
        if thai.get("status") == "ok" and thai.get("gta_spot_usd_oz"):
            sources.append(
                {"name": "Spot (อ้างอิงจากสมาคมค้าทองคำ)", "value": thai["gta_spot_usd_oz"],
                 "as_of": thai["announced_at"], "kind": "spot", "url": GOLD_UPDATES_URL}
            )  # fmt: skip
        api = self._gold_api()
        if api:
            sources.append(
                {"name": "Spot gold-api.com (XAU/USD)", "value": api["price"],
                 "as_of": api["updated_at"], "kind": "spot", "url": GOLD_API_URL}
            )  # fmt: skip
        return sources

    def world_spot(self) -> dict:
        """World spot for alerts: median of the exact sources only (no web estimates)."""
        sources = self._spot_sources(self.thai_gold())
        c = combine_spots(sources)
        if not sources:
            return unavailable("world gold spot", "no spot source")
        return {"status": "ok", **c, "sources": sources}

    def gold(self, compare: bool = False) -> dict:
        """Thai gold and international gold, kept apart. Exact sources first; a labelled estimate
        from web search only when no exact source has the number."""
        thai = self.thai_gold()
        thai_est = None
        if thai.get("status") != "ok":
            thai_est = self._grounded_one("TH_GOLD_BAR", GROUNDED_LABELS["TH_GOLD_BAR"])
            thai = (
                {
                    "status": thai_est["status"],
                    "unit": THAI_UNIT,
                    "purity": "96.5%",
                    "bar_sell": thai_est["price"],
                    "range": thai_est["range"],
                    "n_sources": thai_est["n_sources"],
                    "sources": thai_est["sources"],
                    "confidence": thai_est["confidence"],
                    "note": thai_est["note"],
                    "estimate_text": estimate_text(thai_est),
                    "bar_buy": None,
                    "ornament_buy": None,
                    "ornament_sell": None,
                    "announced_at": "",
                    "round": None,
                    "change": None,
                }  # fmt: skip
                if thai_est
                else thai
            )
        sources = self._spot_sources(thai)
        fut = self._futures()
        world: dict = {
            "unit": WORLD_UNIT,
            "spot_sources": sources,
            "futures_usd_oz": fut["value"] if fut else None,
            "futures_as_of": fut["as_of"] if fut else "",
            "futures_label": "Futures (COMEX GC=F) — ไม่ใช่ราคา Spot / not spot",
        }
        world.update(combine_spots(sources))
        world["spot_basis"] = "exact" if sources else None
        if not sources:
            est = self._grounded_one("XAU_SPOT", GROUNDED_LABELS["XAU_SPOT"])
            if est:
                world.update(
                    spot_usd_oz=est["price"], spot_basis=est["status"], spot_range=est["range"],
                    spot_sources=[
                        {**x, "kind": "web", "name": f"{x['name']} (จากการค้นเว็บ ความแม่นยำต่ำกว่า)"}
                        for x in est["sources"]
                    ],
                    spot_estimate_text=estimate_text({**est, "currency": "USD"}),
                    confidence=est["confidence"],
                )  # fmt: skip
        out: dict = {
            "status": "ok" if thai.get("status") != "unavailable" or world["spot_usd_oz"] or fut
            else "unavailable",
            "thai": thai,
            "world": world,
            "rule": "Never mix Thai (THB per baht-weight, 96.5%) and world (USD per troy ounce) "
            "numbers in one figure; always state unit, source and time for each.",
        }  # fmt: skip
        if compare:
            out["conversion"] = self._conversion(thai, world)
        out["summary"] = gold_summary(out, "th")
        out["summary_en"] = gold_summary(out, "en")
        return out

    def _conversion(self, thai: dict, world: dict) -> dict:
        spot = world.get("spot_usd_oz")
        assoc, ecb = thai.get("gta_usd_thb"), self._frankfurter()
        fx, fx_src, fx_at = (
            assoc,
            "สมาคมค้าทองคำ (Baht / US$)",
            thai.get("announced_at", ""),
        )
        if not fx and ecb:
            fx, fx_src, fx_at = ecb["rate"], "Frankfurter (ECB)", ecb["date"]
        if not spot or not fx:
            return {"status": "unavailable", "reason": "need spot and USD/THB"}
        v = thb_per_baht_weight(spot, fx)
        return {
            "status": "ok" if world.get("spot_basis") == "exact" else "estimate",
            "thb_per_baht_weight_96_5": round(v),
            "formula": "spot_usd_per_oz × usd_thb ÷ 31.1035 × 15.244 × 0.965",
            "spot_usd_oz": spot, "usd_thb": fx, "usd_thb_source": fx_src, "usd_thb_as_of": fx_at,
            # the second FX source (currency.py), shown next to the association's baht rate
            "usd_thb_ecb": ecb["rate"] if ecb else None,
            "usd_thb_ecb_as_of": ecb["date"] if ecb else "",
            "fx_disagree": bool(assoc and ecb and abs(assoc - ecb["rate"]) / ecb["rate"] > 0.01),
            "note": "ราคาคำนวณจากสปอตสากลล้วนๆ ราคาของสมาคมค้าทองคำรวมพรีเมียม/ส่วนต่างของตลาดไทย "
            "จึงต่างจากตัวเลขนี้ / The Thai association price also includes premiums and margins, "
            "so it differs from this figure.",
        }  # fmt: skip

    def news(self, query: str = "", topic: str = "", lang: str = "en", limit: int = 5) -> dict:
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

    def pct(self, guild_id: int, price: float, day: str, key: str = "gold") -> float | None:
        d = self.docs.get(STATE, guild_id, key) or {}
        if d.get("day") != day:
            d = {"day": day, "base": d.get("last") or price, "last": price}
        d["last"] = price
        self.docs.put(STATE, guild_id, key, d)
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
        if q.get("status") not in ("ok", *WEB_STATUSES):
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
        if symbol == WORLD_GOLD:
            w = self.data.world_spot()
            if w.get("status") != "ok":
                return None
            at = max((x["as_of"] for x in w["sources"]), default="")
            return w["spot_usd_oz"], None, "USD/troy oz (spot)", at, "Spot XAU/USD (median)"
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
            if sym in ("gold", WORLD_GOLD):
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
                if w.symbol in ("gold", WORLD_GOLD) and price is not None:
                    pct = self.baseline.pct(gid, price, today, w.symbol.lower().replace(":", "_"))
                if triggered(w.condition, w.threshold, price, pct):
                    self.repo.mark_fired(gid, w, today)
                    out.append(Alert(gid, w.user_id, w, price, pct, cur, as_of, src))
        return out


def format_alert(a: Alert, lang: str = "en") -> str:
    w = a.watch
    cond = t_(f"mk_cond_{w.condition}", lang, thr=f"{w.threshold:g}")
    chg = "" if a.change_pct is None else f" ({a.change_pct:+.2f}%)"
    when = ""
    if a.as_of:
        try:
            when = datetime.fromisoformat(a.as_of).astimezone(BKK).strftime("%d/%m %H:%M")
        except ValueError:
            when = a.as_of
    label, cur, src = w.label, a.currency, a.source
    if lang != "th":
        label, cur, src = localize_en(label), localize_en(cur), localize_en(src)
    return t_(
        "mk_alert", lang, label=label, cond=cond, price=f"{a.price:,.2f}", cur=cur,
        chg=chg, src=src, when=when,
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


# ------------------------------------------------------------------ English-mode localisation
# The data layer builds Thai/bilingual notes. In an English guild the tool results the model reads
# (and the alert text users see) are rewritten to English-only here: exact strings first, then
# substrings, then a "thai / english" bilingual string keeps only its English half.
_EN_EXACT = {
    SOURCE_GOLD: "Gold Traders Association (Thailand)",
    "ไม่ระบุเวลา": "time not stated",
    "เวลา": "time",
    "ลิงก์": "link",
    "ขึ้น": "up",
    "ลง": "down",
    "เท่าเดิม": "unchanged",
    "THB/บาททอง": "THB per baht-weight",
    "ทองคำสากล Spot XAU/USD (USD ต่อทรอยออนซ์)": "World gold spot XAU/USD (USD per troy ounce)",
    "ทองคำแท่ง 96.5% (สมาคม) ขายออก (บาทต่อ 1 บาททองคำ)": (
        "Thai gold bar 96.5% (Gold Traders Association) sell price (THB per baht-weight)"
    ),
}
_EN_SUBSTR = (
    ("Spot (อ้างอิงจากสมาคมค้าทองคำ)", "Spot (as quoted by the Gold Traders Association)"),
    (" (จากการค้นเว็บ ความแม่นยำต่ำกว่า)", " (from web search, lower accuracy)"),
    ("สมาคมค้าทองคำ", "Gold Traders Association"),
    (
        "บาท ต่อน้ำหนัก 1 บาททองคำ เทียบกับประกาศครั้งก่อน",
        "THB per baht-weight vs the previous announcement",
    ),
    ("ดอลลาร์สหรัฐต่อทรอยออนซ์", "USD per troy ounce"),
    ("บาทต่อน้ำหนัก 1 บาททองคำ", "THB per baht-weight"),
    (" (ครั้งที่ ", " (round "),
)


def localize_en(obj):
    """`obj` (nested dict / list / str) with every Thai-bearing string made English-only."""
    if isinstance(obj, dict):
        return {k: localize_en(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [localize_en(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(localize_en(v) for v in obj)
    if not isinstance(obj, str) or not has_thai(obj):
        return obj
    if obj in _EN_EXACT:
        return _EN_EXACT[obj]
    if " / " in obj:
        tail = obj.rsplit(" / ", 1)[1]
        if not has_thai(tail) and len(tail) >= 6:
            return tail
    for th, en in _EN_SUBSTR:
        obj = obj.replace(th, en)
    return obj


def _loc(ctx_lang: str, obj):
    return localize_en(obj) if ctx_lang == "en" else obj


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
    lang: str = "en"
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
        5-day trend per symbol. Never state a price that this tool did not return. status "ok" is a
        direct Yahoo feed (give source and time). status "web_reference" (sources agree) or
        "estimate" (they differ) came from web pages: say "ราคาอ้างอิงจาก N แหล่ง (อ่านจากหน้าเว็บ)"
        or "ราคาโดยประมาณ ช่วง A-B", list each source with value, time and link (`sources`), and
        that it is not an official feed. Only `unavailable` means you cannot get it. Add a short
        "not investment advice" once.

        Args:
            symbols: one or several names or tickers separated by commas: Thai names (ปตท, กสิกร),
                SET tickers (PTT, KBANK), indexes (SET, SET50, S&P 500, Nasdaq, Nikkei) or global
                tickers (AAPL, NVDA).
            market: "auto" (default), "th" (force a SET listing) or "global".
        """
        names = resolve_names(symbols)
        if not names:
            return {"status": "error", "error": "no_symbol"}
        return _loc(ctx.lang, {"quotes": ctx.data.quotes(names, market)})

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
        return _loc(ctx.lang, ctx.data.history(symbol, range, market))

    def gold_price(compare: bool = False) -> dict:
        """Today's gold prices in TWO separate parts: `thai` (Gold Traders Association, 96.5%,
        THB per baht-weight = 15.244 g: bar and ornament buy/sell, announcement time and round,
        change vs previous announcement) and `world` (spot XAU/USD in USD per troy ounce from
        several sources with median and spread; COMEX futures kept apart). Use for "ราคาทองวันนี้",
        "gold price", "ทองสากลเท่าไหร่". `summary` / `summary_en` are ready texts with all units.
        Never mix Thai and world numbers in one figure; always say unit, source and time. A part
        with status "web_reference" or "estimate" came from web pages: say "ราคาอ้างอิงจาก N แหล่ง
        (อ่านจากหน้าเว็บ)" or "ราคาโดยประมาณ ช่วง A-B", list each source with value, time and link. Only if a part is `unavailable` say
        you cannot get it. Add a short "not investment advice" once.

        Args:
            compare: true when the user asks to compare Thai vs world or to convert the world
                price to baht-weight; adds `conversion` (with the FX rate and its source).
        """
        res = ctx.data.gold(bool(compare))
        if ctx.lang != "th":
            res = localize_en(res)
            res["summary"] = res.get("summary_en") or res.get("summary", "")
        return res

    def news(query: str = "", topic: str = "", lang: str = "en", max: int = 5) -> dict:  # noqa: A002
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
        action: str, symbol: str = "", condition: str = "", value: float = 0.0, target: str = ""
    ) -> dict:
        """Price alerts for the asker. "แจ้งเตือนถ้า PTT ต่ำกว่า 30" -> add, PTT, below, 30.
        "ทองขึ้นเกิน 1% บอกด้วย" -> add, gold, up_pct, 1. Salid checks during market hours and
        DMs once per condition per day. Confirm what was set in one short line.

        Args:
            action: "add", "remove" or "list".
            symbol: ticker or name (PTT, ปตท, AAPL, SET) or "gold" for gold (see target).
            target: for gold only: "thai" (default, Thai speakers: the Thai gold bar sell price in
                THB per baht-weight, % against the previous day's last announcement) or "world"
                (international spot XAU/USD in USD per troy ounce). Ask which if unclear; it is
                stored with the alert.
            condition: for add: "above" / "below" (price) or "up_pct" / "down_pct" / "move_pct"
                (change in percent versus the previous close, either direction for move_pct).
            value: the price or the percent.
        """
        gid, uid = ctx.guild_id, ctx.asker_id
        act = (action or "").strip().casefold()
        if act == "list":
            return {
                "watches": [
                    {
                        "id": w.id,
                        "label": _loc(ctx.lang, w.label),
                        "condition": w.condition,
                        "value": w.threshold,
                        "target": gold_target(w.symbol),
                    }  # fmt: skip
                    for w in ctx.repo.of_user(gid, uid)
                ]
            }
        raw = (symbol or "").strip()
        if not raw:
            return {"status": "error", "error": "symbol_required"}
        world = (
            str(target or "").strip().casefold() in WORLD_TARGETS or raw.casefold() in WORLD_TARGETS
        )
        if (
            raw.casefold() in GOLD_WORDS
            or raw in ("ทองแท่ง", "ทองคำแท่ง")
            or (world and not raw.isascii())
            or raw.casefold() in WORLD_TARGETS
        ):
            if world:
                sym, label = WORLD_GOLD, "ทองคำสากล Spot XAU/USD (USD ต่อทรอยออนซ์)"
            else:
                sym, label = "gold", "ทองคำแท่ง 96.5% (สมาคม) ขายออก (บาทต่อ 1 บาททองคำ)"
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
            "watching": _loc(ctx.lang, label),
            "target": gold_target(sym) if sym in ("gold", WORLD_GOLD) else None,
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
    "ที่มาของตัวเลข: status ok = ฟีดตรงจากแหล่งนั้น (บอกชื่อแหล่งและเวลา); web_reference = ราคาอ้างอิงจาก N แหล่ง "
    "(อ่านจากหน้าเว็บ) ที่ตรงกันใน±0.5%; estimate = ราคาโดยประมาณ ช่วง A–B ที่แหล่งต่างกันหรือขาดเวลา/ลิงก์ "
    "ต้องบอกว่ายังยืนยันราคาทางการไม่ได้ และลิสต์ทุกแหล่งพร้อมค่า เวลา ลิงก์ (สั้นๆ); ห้ามเรียกว่า “ราคาทางการ”; "
    "มีแค่ unavailable เท่านั้นที่ตอบว่าดึงราคาไม่ได้. "
    "ราคาทอง: gold_price แยกสองส่วนเสมอ ห้ามปนกันในตัวเลขเดียว — (A) ราคาทองไทย สมาคมค้าทองคำ ทอง 96.5% หน่วย "
    "บาทต่อน้ำหนัก 1 บาททองคำ (15.244 กรัม) ทองคำแท่งรับซื้อ/ขายออก ทองรูปพรรณ รับซื้อ(ฐานภาษี)/ขายออก พร้อมวันเวลา(เวลาไทย) "
    "ครั้งที่ และขึ้น/ลงกี่บาท; (B) ราคาทองสากล Spot XAU/USD หน่วยดอลลาร์สหรัฐต่อทรอยออนซ์ (31.1035 กรัม) บอกแต่ละแหล่ง ค่า เวลา "
    "ถ้ามีหลายแหล่งให้ค่ากลางและบอกส่วนต่าง ถ้าต่างเกิน 1% ให้บอกตรงๆ; ฟิวเจอร์ส (COMEX GC=F) ให้ติดป้ายว่า Futures แยกจาก Spot. "
    "ทุกตัวเลขต้องมี หน่วย + แหล่ง + เวลา. ถามเป็นไทย (“ราคาทองวันนี้”) → ตอบทองไทยก่อน แล้วทองสากล; ถามเป็นอังกฤษ "
    "(“gold price”) → ทองสากลก่อน แล้วทองไทย. เทียบ/แปลงราคาสากลเป็นบาททอง → เรียก gold_price(compare=true) แล้วแสดงสูตร "
    "อัตราแลกเปลี่ยนและแหล่งของมัน พร้อมบอกว่าราคาสมาคมรวมพรีเมียม/ส่วนต่างจึงต่างกัน; ไม่ถามก็ไม่ต้องแสดงการแปลง. "
    "แจ้งเตือนราคาทอง: ถามผู้ใช้ว่าจะให้ดูราคาทองไทย (ทองแท่งขายออก, ค่าเริ่มต้นสำหรับคนไทย) หรือสปอตสากล แล้วส่ง target=thai/world. "
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
