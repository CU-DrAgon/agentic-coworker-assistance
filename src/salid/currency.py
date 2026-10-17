"""convert_currency: exchange rates from free keyless sources, never guessed.

Frankfurter (ECB reference rates, current and historical) first; open.er-api.com (latest only) as
the fallback for currencies the ECB does not publish. These are reference mid-market rates: banks
and card issuers buy and sell at different rates, and the answer says so. Cached for one hour.
Only currency codes and a date ever leave the server. Stored ledger data is never touched: a
converted figure is a labelled approximation next to the per-currency totals, never a replacement.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from datetime import date as date_cls

import httpx

logger = logging.getLogger(__name__)

TIMEOUT_S = 8.0
CACHE_SECONDS = 3600
FRANKFURTER = "https://api.frankfurter.dev/v1"  # api.frankfurter.app now redirects here
ER_API = "https://open.er-api.com/v6/latest"

# Thai names / symbols -> ISO 4217. Matched after trimming and lower-casing, whole string.
_NAMES: dict[str, str] = {
    "บาท": "THB", "฿": "THB", "baht": "THB", "thb": "THB",
    "ดอลลาร์": "USD", "ดอลล่าร์": "USD", "ดอลลาร์สหรัฐ": "USD", "ดอลลาร์อเมริกา": "USD",
    "เหรียญ": "USD", "เหรียญสหรัฐ": "USD", "$": "USD", "us$": "USD", "dollar": "USD",
    "dollars": "USD", "usd": "USD",
    "เยน": "JPY", "เยนญี่ปุ่น": "JPY", "円": "JPY", "¥": "JPY", "yen": "JPY", "jpy": "JPY",
    "หยวน": "CNY", "เหรินหมินปี้": "CNY", "元": "CNY", "yuan": "CNY", "rmb": "CNY",
    "cny": "CNY",
    "ยูโร": "EUR", "€": "EUR", "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "ปอนด์": "GBP", "ปอนด์สเตอร์ลิง": "GBP", "£": "GBP", "pound": "GBP", "pounds": "GBP",
    "gbp": "GBP",
    "วอน": "KRW", "₩": "KRW", "won": "KRW", "krw": "KRW",
    "ดอลลาร์สิงคโปร์": "SGD", "เหรียญสิงคโปร์": "SGD", "s$": "SGD",
    "ดอลลาร์ฮ่องกง": "HKD", "เหรียญฮ่องกง": "HKD", "hk$": "HKD",
    "ดอลลาร์ออสเตรเลีย": "AUD", "เหรียญออสเตรเลีย": "AUD", "a$": "AUD",
    "ดอลลาร์แคนาดา": "CAD", "เหรียญแคนาดา": "CAD", "c$": "CAD",
    "ดอลลาร์ไต้หวัน": "TWD", "เหรียญไต้หวัน": "TWD", "นิวไต้หวัน": "TWD", "nt$": "TWD",
    "ดอลลาร์นิวซีแลนด์": "NZD", "nz$": "NZD",
    "ริงกิต": "MYR", "ringgit": "MYR",
    "รูปี": "INR", "รูปีอินเดีย": "INR", "₹": "INR",
    "รูเปียห์": "IDR", "รูเปียะห์": "IDR", "rupiah": "IDR",
    "ดอง": "VND", "₫": "VND", "dong": "VND",
    "เปโซ": "PHP", "เปโซฟิลิปปินส์": "PHP", "₱": "PHP",
    "ฟรังก์": "CHF", "ฟรังก์สวิส": "CHF",
    "ดิรฮัม": "AED", "ริยัล": "SAR", "กีบ": "LAK", "จ๊าต": "MMK", "เรียล": "KHR",
}  # fmt: skip
_ISO = re.compile(r"^[A-Za-z]{3}$")


def resolve_currency(text: str) -> str | None:
    """'ดอลลาร์' / '$' / 'yen' / 'jpy' -> 'USD' / 'USD' / 'JPY' / 'JPY'; None when unknown."""
    s = (text or "").strip().lower()
    if not s:
        return None
    if s in _NAMES:
        return _NAMES[s]
    return s.upper() if _ISO.match(s) else None


Fetch = Callable[[str], dict]


def http_fetch(url: str) -> dict:
    """GET json with a hard timeout. Raises on any failure (the caller falls back)."""
    r = httpx.get(url, timeout=TIMEOUT_S, follow_redirects=True)
    r.raise_for_status()
    return r.json()


def _fmt(x: float) -> str:
    return f"{x:,.2f}" if abs(x) < 1e15 else str(x)


class Converter:
    """Rates with a 1 h cache. `fetch(url) -> dict` and `clock()` are injectable for tests."""

    def __init__(self, fetch: Fetch = http_fetch, clock: Callable[[], float] = time.monotonic):
        self._fetch, self._clock = fetch, clock
        self._cache: dict[tuple, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def rate(self, base: str, to: str, on: str | None = None) -> dict:
        """{rate, date, source, note?} or {error}. `on`: YYYY-MM-DD (None = latest)."""
        key = (base, to, on)
        with self._lock:
            hit = self._cache.get(key)
            if hit and self._clock() - hit[0] < CACHE_SECONDS:
                return hit[1]
        out = self._lookup(base, to, on)
        if "error" not in out:
            with self._lock:
                self._cache[key] = (self._clock(), out)
        return out

    def _lookup(self, base: str, to: str, on: str | None) -> dict:
        if base == to:
            return {"rate": 1.0, "date": on or date_cls.today().isoformat(), "source": "identity"}
        try:
            d = self._fetch(f"{FRANKFURTER}/{on or 'latest'}?base={base}&symbols={to}")
            return {"rate": float(d["rates"][to]), "date": d["date"], "source": "ECB"}
        except Exception as e:
            logger.info("frankfurter failed (%s); trying open.er-api", type(e).__name__)
        try:
            d = self._fetch(f"{ER_API}/{base}")
            if d.get("result") != "success":
                raise ValueError("er-api error")
            day = time.strftime("%Y-%m-%d", time.gmtime(d["time_last_update_unix"]))
            out = {"rate": float(d["rates"][to]), "date": day, "source": "open.er-api.com"}
            if on and on != day:
                out["note"] = f"no historical rate for {on} from this source; the latest was used"
            return out
        except Exception as e:
            logger.warning("er-api failed (%s)", type(e).__name__)
            return {"error": "exchange rate unavailable right now; do not guess a rate"}


def label(converted: float, to: str, source: str, day: str, lang: str = "en") -> str:
    src = "ECB" if source == "ECB" else source
    if lang != "th":
        return f"≈ {_fmt(converted)} {to} ({src} rate on {day})"
    return f"≈ {_fmt(converted)} {to} (อัตรา {src} วันที่ {day})"


def convert(
    conv: Converter, amount: float, from_cur: str, to_cur: str, on: str = "", lang: str = "en"
) -> dict:
    base, to = resolve_currency(from_cur), resolve_currency(to_cur)
    if base is None or to is None:
        bad = from_cur if base is None else to_cur
        return {"error": f"unknown currency '{bad}'; use an ISO code such as USD, JPY, THB"}
    try:
        amt = float(amount)
    except (TypeError, ValueError):
        return {"error": "amount must be a number"}
    on = (on or "").strip()
    if on:
        try:
            if date_cls.fromisoformat(on) > date_cls.today():
                on = ""  # a future date: use today's rate
        except ValueError:
            return {"error": "date must be YYYY-MM-DD"}
    r = conv.rate(base, to, on or None)
    if "error" in r:
        return r
    converted = round(amt * r["rate"], 2)
    out = {
        "amount": amt,
        "from": base,
        "to": to,
        "rate": r["rate"],
        "rate_date": r["date"],
        "source": r["source"],
        "converted": converted,
        "converted_fmt": f"{_fmt(converted)} {to}",
        "label": label(converted, to, r["source"], r["date"], lang),
        "disclaimer": (
            "อัตรากลาง (reference mid-market) ธนาคารรับซื้อ/ขายคนละอัตรา ตัวเลขจริงอาจต่างเล็กน้อย"
            if lang == "th"
            else "Reference mid-market rate; banks and card issuers buy/sell at different rates."
        ),
    }
    if r.get("note"):
        out["note"] = r["note"]
    return out


def currency_tools(conv: Converter, lang: str = "en") -> list:
    def convert_currency(
        amount: float, from_currency: str, to_currency: str, date: str = ""
    ) -> dict:
        """Convert money with a real exchange rate (ECB reference rates, cached 1 h). Use it for
        EVERY conversion; never estimate a rate yourself. Quote the returned `label` (it carries
        the rate date and source) and mention it is a reference mid-market rate.

        Args:
            amount: the amount in from_currency.
            from_currency: ISO code or name, e.g. USD, JPY, "ดอลลาร์", "เยน", "$", "฿".
            to_currency: ISO code or name, e.g. THB.
            date: optional YYYY-MM-DD for a past day's rate (e.g. the receipt date); empty = today.
        """
        return convert(conv, amount, from_currency, to_currency, date, lang)

    convert_currency.__name__ = "convert_currency"
    convert_currency.salid_group = "travel"
    return [convert_currency]


CURRENCY_PROMPT = (
    "\nแปลงสกุลเงิน: ทุกครั้งที่ต้องแปลงเงิน (เช่น “100 ดอลลาร์กี่บาท”, “รวมเป็นเงินบาทเท่าไหร่”) ให้เรียก convert_currency "
    "ห้ามเดาหรือประมาณอัตราเอง ถ้าเครื่องมือล้มเหลวให้บอกตรงๆ ว่าตอนนี้ดึงอัตราไม่ได้. "
    "บอกอัตรา วันที่ของอัตรา และแหล่งที่มา (ใช้ฟิลด์ label) และบอกสั้นๆ ว่าเป็นอัตรากลางอ้างอิง ธนาคารรับซื้อ/ขายต่างกัน. "
    "ยอดในบัญชี: ตัวเลขแยกตามสกุลเงินยังเป็นตัวเลขหลักเสมอ ถ้าผู้ใช้ถามว่า “รวมเป็นเงินบาทเท่าไหร่” จึงแปลงยอดสกุลอื่นด้วย "
    "convert_currency (อัตราวันนี้ หรือวันที่ใบเสร็จถ้าผู้ใช้ต้องการ) แล้วแสดงเป็น “≈ ฿… (อัตรา ECB วันที่ …)” พร้อมยอดแยกสกุล "
    "ห้ามบอกว่าเป็นยอดจริงหรือแก้ข้อมูลในบัญชี."
)
