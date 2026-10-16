"""Flights: status, indicative prices and (with a provider key) live tracking.

No free keyless flight-status or fare API exists, so the honest design is:
- `flight_status`: a flight-data provider when `SALID_AERODATABOX_KEY` (RapidAPI) or
  `SALID_AVIATIONSTACK_KEY` is set (adapters behind one interface, disabled when unset); otherwise
  Google Search grounding through the existing web_read client with a strict prompt, labelled
  `confidence: "web"`, always with FlightAware and Google links.
- `flight_prices`: Google Search grounding only, labelled indicative, with Google Flights and
  Skyscanner deep links.
- `track_flight`: only with a provider key. Polls every 15 minutes from 6 h before departure until
  landed and DMs the requester on a delay of 15 minutes or more, a gate change, cancellation and
  landing, each at most once. Tracks persist in the DocStore.

Only the flight number, route and dates go to any third party; never chat, ledger or people.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import date as date_cls
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx

from . import web_read

logger = logging.getLogger(__name__)

TIMEOUT_S = 8.0
STATUS_CACHE_S = 5 * 60
PRICES_CACHE_S = 30 * 60
TRACK_KIND = "flight_track"
POLL_EVERY_S = 15 * 60
START_BEFORE_S = 6 * 3600
GIVE_UP_AFTER_S = 24 * 3600  # after the scheduled departure
DELAY_NOTIFY_MIN = 15
MAX_TRACKS_PER_USER = 5

# airline name (lower-cased, spaces removed) -> IATA code
AIRLINE_NAMES: dict[str, str] = {
    "การบินไทย": "TG", "thaiairways": "TG", "thai": "TG",
    "ไทยแอร์เอเชีย": "FD", "แอร์เอเชีย": "FD", "thaiairasia": "FD", "airasia": "FD",
    "ไทยไลอ้อนแอร์": "SL", "ไลอ้อนแอร์": "SL", "thailionair": "SL",
    "บางกอกแอร์เวย์ส": "PG", "บางกอกแอร์เวย์": "PG", "bangkokairways": "PG",
    "นกแอร์": "DD", "nokair": "DD",
    "ไทยเวียตเจ็ท": "VZ", "เวียตเจ็ท": "VZ", "thaivietjet": "VZ",
    "ไทยสมายล์": "WE", "thaismile": "WE",
    "สิงคโปร์แอร์ไลน์": "SQ", "singaporeairlines": "SQ",
    "เจแปนแอร์ไลน์": "JL", "japanairlines": "JL", "jal": "JL",
    "แอลนิปปอน": "NH", "ana": "NH", "allnipponairways": "NH",
    "เอมิเรตส์": "EK", "emirates": "EK",
    "คาเธ่ย์แปซิฟิก": "CX", "cathaypacific": "CX",
    "อีวีเอแอร์": "BR", "evaair": "BR",
    "โคเรียนแอร์": "KE", "koreanair": "KE",
    "กาตาร์แอร์เวย์ส": "QR", "qatarairways": "QR",
    "มาเลเซียแอร์ไลน์": "MH", "malaysiaairlines": "MH",
    "ไทยไทเกอร์แอร์": "TR", "scoot": "TR", "tigerair": "TR",
    "บริติชแอร์เวย์ส": "BA", "britishairways": "BA",
    "ลุฟท์ฮันซ่า": "LH", "lufthansa": "LH",
    "แอร์ฟรานซ์": "AF", "airfrance": "AF",
    "ควอนตัส": "QF", "qantas": "QF",
    "ยูไนเต็ด": "UA", "unitedairlines": "UA",
}  # fmt: skip
# IATA -> ICAO airline designator, for FlightAware (it prefers ICAO callsigns)
ICAO: dict[str, str] = {
    "TG": "THA", "FD": "AIQ", "SL": "TLM", "PG": "BKP", "DD": "NOK", "VZ": "TVJ", "WE": "THD",
    "SQ": "SIA", "JL": "JAL", "NH": "ANA", "EK": "UAE", "CX": "CPA", "BR": "EVA", "KE": "KAL",
    "QR": "QTR", "MH": "MAS", "TR": "TGW", "3K": "JSA", "BA": "BAA", "LH": "DLH", "AF": "AFR",
    "QF": "QFA", "UA": "UAL", "AA": "AAL", "DL": "DAL", "AK": "AXM", "QZ": "AWQ", "VJ": "VJC",
}  # fmt: skip

_CODE_NUM = re.compile(r"^([a-z0-9]{2})\s*[-\s]?\s*0*(\d{1,4}[a-z]?)$")
_NAME_NUM = re.compile(r"^(.+?)\s*[-\s]?\s*0*(\d{1,4}[a-z]?)$")


def normalize_flight(text: str) -> dict | None:
    """'TG 676' / 'tg676' / 'TG-0676' / 'การบินไทย 676' -> {iata, airline, number, icao, display}."""
    s = (text or "").strip().lower()
    if not s:
        return None
    airline = number = None
    m = _CODE_NUM.match(s)
    if m and any(c.isalpha() for c in m.group(1)):
        airline, number = m.group(1).upper(), m.group(2).upper()
    else:
        m = _NAME_NUM.match(s)
        if m:
            code = AIRLINE_NAMES.get(re.sub(r"[\s.\-]+", "", m.group(1)))
            if code:
                airline, number = code, m.group(2).upper()
    if airline is None:
        return None
    icao = ICAO.get(airline)
    return {
        "iata": f"{airline}{number}",
        "airline": airline,
        "number": number,
        "icao": f"{icao}{number}" if icao else None,
        "display": f"{airline} {number}",
    }


def flight_links(flight: dict) -> dict:
    ident = flight["icao"] or flight["iata"]
    return {
        "flightaware": f"https://www.flightaware.com/live/flight/{ident}",
        "google": "https://www.google.com/search?q="
        + quote(f"{flight['display']} flight status", safe="").replace("%20", "+"),
    }


# --------------------------------------------------------------------------- airports
AIRPORTS: dict[str, str] = {
    "BKK": "BKK", "สุวรรณภูมิ": "BKK", "suvarnabhumi": "BKK", "กรุงเทพ": "BKK", "bangkok": "BKK",
    "DMK": "DMK", "ดอนเมือง": "DMK", "donmueang": "DMK", "don mueang": "DMK",
    "CNX": "CNX", "เชียงใหม่": "CNX", "chiangmai": "CNX", "chiang mai": "CNX",
    "HKT": "HKT", "ภูเก็ต": "HKT", "phuket": "HKT",
    "KBV": "KBV", "กระบี่": "KBV", "krabi": "KBV",
    "USM": "USM", "สมุย": "USM", "เกาะสมุย": "USM", "samui": "USM", "koh samui": "USM",
    "HDY": "HDY", "หาดใหญ่": "HDY", "hatyai": "HDY", "hat yai": "HDY",
    "CEI": "CEI", "เชียงราย": "CEI", "chiangrai": "CEI", "chiang rai": "CEI",
    "KKC": "KKC", "ขอนแก่น": "KKC", "khonkaen": "KKC", "UTH": "UTH", "อุดรธานี": "UTH",
    "UBP": "UBP", "อุบลราชธานี": "UBP", "อุบล": "UBP",
    "NRT": "NRT", "นาริตะ": "NRT", "narita": "NRT",
    "HND": "HND", "ฮาเนดะ": "HND", "haneda": "HND",
    "TYO": "TYO", "โตเกียว": "TYO", "tokyo": "TYO",
    "KIX": "KIX", "คันไซ": "KIX", "kansai": "KIX", "OSA": "OSA", "โอซาก้า": "OSA", "osaka": "OSA",
    "FUK": "FUK", "ฟุกุโอกะ": "FUK", "fukuoka": "FUK", "CTS": "CTS", "ซัปโปโร": "CTS",
    "sapporo": "CTS", "ซัปโปโร (ชินจิโตเซะ)": "CTS",
    "ICN": "ICN", "อินชอน": "ICN", "incheon": "ICN", "SEL": "SEL", "โซล": "SEL", "seoul": "SEL",
    "PUS": "PUS", "ปูซาน": "PUS", "busan": "PUS",
    "SIN": "SIN", "สิงคโปร์": "SIN", "singapore": "SIN",
    "HKG": "HKG", "ฮ่องกง": "HKG", "hongkong": "HKG", "hong kong": "HKG",
    "TPE": "TPE", "ไทเป": "TPE", "taipei": "TPE", "เถาหยวน": "TPE",
    "PEK": "PEK", "ปักกิ่ง": "PEK", "beijing": "PEK", "PVG": "PVG", "เซี่ยงไฮ้": "PVG",
    "shanghai": "PVG", "CAN": "CAN", "กวางโจว": "CAN", "guangzhou": "CAN",
    "KUL": "KUL", "กัวลาลัมเปอร์": "KUL", "kualalumpur": "KUL", "kuala lumpur": "KUL",
    "CGK": "CGK", "จาการ์ตา": "CGK", "jakarta": "CGK", "DPS": "DPS", "บาหลี": "DPS", "bali": "DPS",
    "SGN": "SGN", "โฮจิมินห์": "SGN", "hochiminh": "SGN", "ho chi minh": "SGN",
    "HAN": "HAN", "ฮานอย": "HAN", "hanoi": "HAN", "DAD": "DAD", "ดานัง": "DAD", "danang": "DAD",
    "MNL": "MNL", "มะนิลา": "MNL", "manila": "MNL", "REP": "REP", "เสียมเรียบ": "REP",
    "PNH": "PNH", "พนมเปญ": "PNH", "VTE": "VTE", "เวียงจันทน์": "VTE", "RGN": "RGN",
    "ย่างกุ้ง": "RGN", "yangon": "RGN",
    "DEL": "DEL", "เดลี": "DEL", "delhi": "DEL", "BOM": "BOM", "มุมไบ": "BOM", "mumbai": "BOM",
    "LHR": "LHR", "ฮีทโธรว์": "LHR", "heathrow": "LHR", "LON": "LON", "ลอนดอน": "LON",
    "london": "LON", "CDG": "CDG", "ปารีส": "CDG", "paris": "CDG", "FRA": "FRA",
    "แฟรงก์เฟิร์ต": "FRA", "frankfurt": "FRA", "AMS": "AMS", "อัมสเตอร์ดัม": "AMS",
    "amsterdam": "AMS", "FCO": "FCO", "โรม": "FCO", "rome": "FCO", "ZRH": "ZRH", "ซูริค": "ZRH",
    "zurich": "ZRH", "MUC": "MUC", "มิวนิก": "MUC", "munich": "MUC",
    "SFO": "SFO", "ซานฟรานซิสโก": "SFO", "sanfrancisco": "SFO", "san francisco": "SFO",
    "LAX": "LAX", "ลอสแอนเจลิส": "LAX", "losangeles": "LAX", "los angeles": "LAX",
    "JFK": "JFK", "นิวยอร์ก": "JFK", "newyork": "JFK", "new york": "JFK",
    "SEA": "SEA", "ซีแอตเทิล": "SEA", "seattle": "SEA",
    "DXB": "DXB", "ดูไบ": "DXB", "dubai": "DXB", "DOH": "DOH", "โดฮา": "DOH", "doha": "DOH",
    "IST": "IST", "อิสตันบูล": "IST", "istanbul": "IST",
    "SYD": "SYD", "ซิดนีย์": "SYD", "sydney": "SYD", "MEL": "MEL", "เมลเบิร์น": "MEL",
    "melbourne": "MEL", "AKL": "AKL", "ออคแลนด์": "AKL", "auckland": "AKL",
}  # fmt: skip
_AIRPORT_KEYS = {re.sub(r"[\s.\-]+", "", k.lower()): v for k, v in AIRPORTS.items()}


def resolve_airport(text: str) -> str | None:
    """Thai/English city or airport name, or an IATA code -> IATA code (None if unknown)."""
    s = re.sub(r"[\s.\-]+", "", (text or "").strip().lower())
    if not s:
        return None
    if s in _AIRPORT_KEYS:
        return _AIRPORT_KEYS[s]
    return s.upper() if re.fullmatch(r"[a-z]{3}", s) else None


def prices_links(origin: str, dest: str, depart: str, ret: str = "", adults: int = 1,
                 cabin: str = "economy") -> dict:  # fmt: skip
    q = f"Flights to {dest} from {origin} on {depart}" + (f" through {ret}" if ret else "")
    links = {"google_flights": "https://www.google.com/travel/flights?q=" + quote(q, safe="")}
    if re.fullmatch(r"[A-Z]{3}", origin) and re.fullmatch(r"[A-Z]{3}", dest):
        d = depart.replace("-", "")[2:]
        r = f"/{ret.replace('-', '')[2:]}" if ret else ""
        links["skyscanner"] = (
            f"https://www.skyscanner.co.th/transport/flights/{origin.lower()}/{dest.lower()}/{d}{r}/"
            f"?adultsv2={max(1, int(adults))}&cabinclass={cabin}"
        )
    return links


# --------------------------------------------------------------------------- providers
def http_get(url: str, headers: dict | None = None) -> dict | list:
    r = httpx.get(url, headers=headers or {}, timeout=TIMEOUT_S, follow_redirects=True)
    r.raise_for_status()
    return r.json()


def _ts(value: str | None, tz: str | None = None) -> float | None:
    """ISO-ish time -> epoch seconds. `tz`: reinterpret a naive/mislabelled time in that zone."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if tz:
            dt = dt.replace(tzinfo=ZoneInfo(tz))
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    except Exception:
        return None


class FlightProvider:
    name = ""

    def status(self, flight: dict, day: str) -> dict | None:
        """Normalised flight dict, or None when the provider does not know the flight.
        Raises on transport errors."""
        raise NotImplementedError


class AeroDataBox(FlightProvider):
    name = "AeroDataBox"
    HOST = "aerodatabox.p.rapidapi.com"
    STATUS = {"canceled": "cancelled", "cancelled": "cancelled", "arrived": "landed",
              "enroute": "active", "departed": "active", "delayed": "delayed",
              "diverted": "diverted", "expected": "scheduled", "checkin": "scheduled",
              "boarding": "scheduled", "gateclosed": "scheduled", "approaching": "active"}  # fmt: skip

    def __init__(self, key: str, get: Callable = http_get):
        self._key, self._get = key, get

    def status(self, flight: dict, day: str) -> dict | None:
        data = self._get(
            f"https://{self.HOST}/flights/number/{flight['iata']}/{day}",
            {"X-RapidAPI-Key": self._key, "X-RapidAPI-Host": self.HOST},
        )
        items = data if isinstance(data, list) else []
        if not items:
            return None
        return self.parse(items[0])

    def parse(self, f: dict) -> dict:
        dep, arr = f.get("departure") or {}, f.get("arrival") or {}

        def t(side, key, sub="local"):
            return ((side.get(key) or {}).get(sub)) or None

        dep_s, dep_r = t(dep, "scheduledTime", "utc"), t(dep, "revisedTime", "utc")
        a, b = _ts(dep_s), _ts(dep_r)
        delay = round((b - a) / 60) if a is not None and b is not None else 0
        status = self.STATUS.get(str(f.get("status", "")).replace(" ", "").lower(), "unknown")
        if status == "scheduled" and delay >= DELAY_NOTIFY_MIN:
            status = "delayed"
        return {
            "airline": (f.get("airline") or {}).get("name"),
            "origin": (dep.get("airport") or {}).get("iata"),
            "destination": (arr.get("airport") or {}).get("iata"),
            "dep_scheduled": t(dep, "scheduledTime"),
            "dep_estimated": t(dep, "revisedTime"),
            "arr_scheduled": t(arr, "scheduledTime"),
            "arr_estimated": t(arr, "revisedTime"),
            "dep_terminal": dep.get("terminal"),
            "dep_gate": dep.get("gate"),
            "arr_terminal": arr.get("terminal"),
            "arr_gate": arr.get("gate"),
            "status": status,
            "delay_min": max(delay, 0),
            "dep_ts": _ts(dep_s),
        }


class AviationStack(FlightProvider):
    name = "AviationStack"
    STATUS = {"scheduled": "scheduled", "active": "active", "landed": "landed",
              "cancelled": "cancelled", "incident": "diverted", "diverted": "diverted"}  # fmt: skip

    def __init__(self, key: str, get: Callable = http_get):
        self._key, self._get = key, get

    def status(self, flight: dict, day: str) -> dict | None:
        data = self._get(
            "https://api.aviationstack.com/v1/flights?access_key="
            f"{quote(self._key, safe='')}&flight_iata={flight['iata']}&flight_date={day}",
            None,
        )
        items = (data or {}).get("data") if isinstance(data, dict) else None
        return self.parse(items[0]) if items else None

    def parse(self, f: dict) -> dict:
        dep, arr = f.get("departure") or {}, f.get("arrival") or {}
        delay = int(dep.get("delay") or 0)  # minutes
        status = self.STATUS.get(str(f.get("flight_status", "")).lower(), "unknown")
        if status == "scheduled" and delay >= DELAY_NOTIFY_MIN:
            status = "delayed"
        return {
            "airline": (f.get("airline") or {}).get("name"),
            "origin": dep.get("iata"),
            "destination": arr.get("iata"),
            "dep_scheduled": dep.get("scheduled"),
            "dep_estimated": dep.get("estimated"),
            "arr_scheduled": arr.get("scheduled"),
            "arr_estimated": arr.get("estimated"),
            "dep_terminal": dep.get("terminal"),
            "dep_gate": dep.get("gate"),
            "arr_terminal": arr.get("terminal"),
            "arr_gate": arr.get("gate"),
            "status": status,
            "delay_min": max(delay, 0),
            # aviationstack labels local times as +00:00: reinterpret them in the airport's zone
            "dep_ts": _ts(dep.get("scheduled"), dep.get("timezone")),
        }


def select_provider(aerodatabox_key: str = "", aviationstack_key: str = "",
                    get: Callable = http_get) -> FlightProvider | None:  # fmt: skip
    """AeroDataBox first, then AviationStack; None when no key is set (feature disabled)."""
    if (aerodatabox_key or "").strip():
        return AeroDataBox(aerodatabox_key.strip(), get)
    if (aviationstack_key or "").strip():
        return AviationStack(aviationstack_key.strip(), get)
    return None


# --------------------------------------------------------------------------- the service
STATUS_INSTRUCTION = (
    "You look up flight facts on the web for a Thai assistant. Report only what the search "
    "results actually say. Never guess, never fill a gap from the usual schedule. If you cannot "
    "find a fact, write 'ไม่พบข้อมูล'. Be brief and structured."
)


def status_prompt(flight: dict, day: str, today: str) -> str:
    return (
        f"Find the status of flight {flight['display']} (IATA {flight['iata']}"
        + (f", ICAO {flight['icao']}" if flight["icao"] else "")
        + f") departing on {day}. Today is {today}.\n"
        "Return exactly these lines, in Thai:\n"
        "สายการบิน / เส้นทาง (ต้นทาง → ปลายทาง พร้อมรหัส IATA)\n"
        "ออกเดินทาง: เวลาตามตาราง และเวลาที่คาดว่าจะออกจริง (เวลาท้องถิ่นของสนามบินต้นทาง)\n"
        "ถึง: เวลาตามตาราง และเวลาที่คาดว่าจะถึงจริง (เวลาท้องถิ่นของสนามบินปลายทาง)\n"
        "เทอร์มินัล / ประตู (gate)\n"
        "สถานะ (ตรงเวลา, ล่าช้า, ยกเลิก, ออกแล้ว, ลงจอดแล้ว)\n"
        "State which page each fact comes from. Use only information for this exact date."
    )


def prices_prompt(origin: str, dest: str, depart: str, ret: str, adults: int, cabin: str,
                  today: str) -> str:  # fmt: skip
    trip = f"round trip returning {ret}" if ret else "one way"
    return (
        f"Find indicative airfares for flights from {origin} to {dest} departing {depart}, {trip}, "
        f"{adults} adult(s), {cabin} cabin. Today is {today}.\n"
        "Return in Thai: the cheapest airlines found and the price range for each (currency "
        "exactly as shown on the source, prefer THB when the source shows it), whether it is "
        "direct or has stops, and which page each price comes from. These are web-search "
        "snapshots, not bookable quotes. If you find no prices, say so; never estimate a fare."
    )


class FlightService:
    def __init__(self, client=None, model: str = "", provider: FlightProvider | None = None,
                 today: Callable[[], date_cls] = date_cls.today,
                 clock: Callable[[], float] = time.monotonic):  # fmt: skip
        self.client, self.model, self.provider = client, model, provider
        self._today, self._clock = today, clock
        self._cache: dict[tuple, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def _cached(self, key: tuple, ttl: float, make: Callable[[], dict]) -> dict:
        with self._lock:
            hit = self._cache.get(key)
            if hit and self._clock() - hit[0] < ttl:
                return hit[1]
        out = make()
        if "error" not in out:
            with self._lock:
                self._cache[key] = (self._clock(), out)
        return out

    def _ground(self, prompt: str) -> dict:
        if self.client is None:
            return {"error": "web search is not available"}
        return web_read.grounded_search(self.client, self.model, prompt, STATUS_INSTRUCTION)

    def status(self, flight_number: str, day: str = "") -> dict:
        flight = normalize_flight(flight_number)
        if flight is None:
            return {"error": "could not read the flight number; use e.g. 'TG 676'"}
        day = (day or "").strip() or self._today().isoformat()
        try:
            date_cls.fromisoformat(day)
        except ValueError:
            return {"error": "date must be YYYY-MM-DD"}
        return self._cached(("status", flight["iata"], day), STATUS_CACHE_S,
                            lambda: self._status(flight, day))  # fmt: skip

    def _status(self, flight: dict, day: str) -> dict:
        links = flight_links(flight)
        base = {"flight": flight["display"], "date": day, "links": links,
                "checked_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")}  # fmt: skip
        if self.provider is not None:
            try:
                info = self.provider.status(flight, day)
            except Exception as e:
                logger.warning("flight provider failed (%s)", type(e).__name__)
                info = None
            if info:
                return {**base, "confidence": "provider", "source": self.provider.name, **info}
        res = self._ground(status_prompt(flight, day, self._today().isoformat()))
        if "error" in res:
            return {**base, "error": res["error"], "note": "give the links so the user can check"}
        return {
            **base,
            "confidence": "web",
            "text": res["text"],
            "sources": res.get("sources", []),
            "note": "ข้อมูลจากการค้นเว็บ อาจไม่ใช่เวลาจริงล่าสุด ให้ตรวจที่ลิงก์ FlightAware/สายการบินอีกครั้ง",
        }

    def prices(self, origin: str, destination: str, depart_date: str, return_date: str = "",
               adults: int = 1, cabin: str = "economy") -> dict:  # fmt: skip
        o = resolve_airport(origin) or (origin or "").strip()
        d = resolve_airport(destination) or (destination or "").strip()
        if not o or not d:
            return {"error": "origin and destination are required"}
        try:
            dep = date_cls.fromisoformat((depart_date or "").strip())
            ret = (
                date_cls.fromisoformat(return_date.strip()) if (return_date or "").strip() else None
            )
        except ValueError:
            return {"error": "dates must be YYYY-MM-DD"}
        today = self._today()
        if dep < today:
            return {"error": "the departure date is in the past"}
        if ret is not None and ret < dep:
            return {"error": "the return date is before the departure date"}
        try:
            adults = max(1, min(9, int(adults)))
        except (TypeError, ValueError):
            adults = 1
        cabin = cabin if cabin in ("economy", "premium_economy", "business", "first") else "economy"
        dep_s, ret_s = dep.isoformat(), ret.isoformat() if ret else ""
        links = prices_links(o, d, dep_s, ret_s, adults, cabin)

        def make() -> dict:
            res = self._ground(prices_prompt(o, d, dep_s, ret_s, adults, cabin, today.isoformat()))
            base = {"origin": o, "destination": d, "depart_date": dep_s, "return_date": ret_s,
                    "adults": adults, "cabin": cabin, "searched_on": today.isoformat(),
                    "confidence": "web", "links": links}  # fmt: skip
            if "error" in res:
                return {**base, "error": res["error"], "note": "give the links instead"}
            return {
                **base,
                "text": res["text"],
                "sources": res.get("sources", []),
                "label": "ราคาโดยประมาณจากการค้นเว็บ ณ วันนี้ ราคาจริงเปลี่ยนตลอด",
                "label_en": "Indicative prices from a web search today; real fares change constantly.",
            }

        return self._cached(("prices", o, d, dep_s, ret_s, adults, cabin), PRICES_CACHE_S, make)


# --------------------------------------------------------------------------- tracking
def _hhmm(local_iso: str | None) -> str:
    if not local_iso:
        return "-"
    m = re.search(r"(\d{2}:\d{2})", local_iso)
    return m.group(1) if m else local_iso


@dataclass
class TrackEvent:
    kind: str  # delay | gate | cancelled | landed
    minutes: int = 0
    gate: str = ""
    time: str = ""


def diff_events(state: dict, info: dict) -> tuple[list[TrackEvent], dict]:
    """New events since the last notification, and the updated state (notify once)."""
    events: list[TrackEvent] = []
    st = dict(state)
    if info.get("status") == "cancelled" and not st.get("cancelled"):
        events.append(TrackEvent("cancelled"))
        st["cancelled"] = True
    delay = int(info.get("delay_min") or 0)
    last = int(st.get("delay_notified") or 0)
    if delay >= DELAY_NOTIFY_MIN and abs(delay - last) >= DELAY_NOTIFY_MIN:
        events.append(TrackEvent("delay", minutes=delay,
                                 time=_hhmm(info.get("dep_estimated"))))  # fmt: skip
        st["delay_notified"] = delay
    gate = info.get("dep_gate") or ""
    if gate and gate != (st.get("gate") or ""):
        if st.get("gate_seen"):  # the first sight of a gate is the baseline when tracking starts
            events.append(TrackEvent("gate", gate=gate))
        st["gate"] = gate
    st["gate_seen"] = True
    if info.get("status") == "landed" and not st.get("landed"):
        events.append(TrackEvent("landed", time=_hhmm(info.get("arr_estimated"))))
        st["landed"] = True
    return events, st


class FlightTracker:
    """Persisted tracks in the DocStore: one doc per (user, flight, date)."""

    def __init__(self, docs, provider: FlightProvider | None, now: Callable[[], float] = time.time):
        self.docs, self.provider, self.now = docs, provider, now

    @staticmethod
    def key(user_id: int, flight: dict, day: str) -> str:
        return f"{user_id}_{flight['iata']}_{day}"

    def mine(self, guild_id: int, user_id: int) -> list[dict]:
        return self.docs.find(TRACK_KIND, guild_id, lambda d: d.get("user_id") == user_id)

    def add(self, guild_id: int, user_id: int, flight: dict, day: str) -> dict:
        if self.provider is None:
            return {"error": "no_provider"}
        key = self.key(user_id, flight, day)
        if self.docs.get(TRACK_KIND, guild_id, key) is None and (
            len(self.mine(guild_id, user_id)) >= MAX_TRACKS_PER_USER
        ):
            return {"error": "too_many", "max": MAX_TRACKS_PER_USER}
        try:
            info = self.provider.status(flight, day)
        except Exception as e:
            logger.warning("track lookup failed (%s)", type(e).__name__)
            return {"error": "lookup_failed"}
        if not info:
            return {"error": "not_found"}
        _, state = diff_events({}, info)  # silent baseline
        dep_ts = info.get("dep_ts") or self.now() + START_BEFORE_S
        doc = {"guild_id": guild_id, "user_id": user_id, "iata": flight["iata"],
               "display": flight["display"], "date": day, "dep_ts": dep_ts, "state": state,
               "next_poll": max(self.now(), dep_ts - START_BEFORE_S)}  # fmt: skip
        self.docs.put(TRACK_KIND, guild_id, key, doc)
        return {"status": "tracking", "doc": doc, "info": info}

    def remove(self, guild_id: int, user_id: int, flight: dict, day: str) -> bool:
        key = self.key(user_id, flight, day)
        had = self.docs.get(TRACK_KIND, guild_id, key) is not None
        self.docs.delete(TRACK_KIND, guild_id, key)
        return had

    def due(self, guild_id: int) -> list[dict]:
        now = self.now()
        return self.docs.find(TRACK_KIND, guild_id, lambda d: d.get("next_poll", 0) <= now)

    def poll(self, doc: dict) -> list[TrackEvent]:
        """One poll of one track: the events to tell the requester. Finished tracks are deleted;
        the state is saved before the caller notifies, so an event is never sent twice."""
        gid, key = doc["guild_id"], f"{doc['user_id']}_{doc['iata']}_{doc['date']}"
        now = self.now()
        if self.provider is None or now > doc["dep_ts"] + GIVE_UP_AFTER_S:
            self.docs.delete(TRACK_KIND, gid, key)
            return []
        flight = normalize_flight(doc["iata"])
        try:
            info = self.provider.status(flight, doc["date"]) if flight else None
        except Exception as e:
            logger.warning("track poll failed (%s)", type(e).__name__)
            info = None
        doc = {**doc, "next_poll": now + POLL_EVERY_S}
        events: list[TrackEvent] = []
        if info:
            events, doc["state"] = diff_events(doc.get("state") or {}, info)
            if info.get("dep_ts"):
                doc["dep_ts"] = info["dep_ts"]
        if info and info.get("status") in ("landed", "cancelled"):
            self.docs.delete(TRACK_KIND, gid, key)
        else:
            self.docs.put(TRACK_KIND, gid, key, doc)
        return events


# --------------------------------------------------------------------------- agent tools
def flight_tools(svc: FlightService, tracker: FlightTracker, guild_id: int, user_id: int,
                 today: Callable[[], date_cls] = date_cls.today) -> list:  # fmt: skip
    def flight_status(flight_number: str, date: str = "") -> dict:
        """Status of a flight: airline, route, scheduled/estimated times (local), terminal/gate,
        status. Without a flight-data provider this comes from a web search (confidence "web"):
        say so, cite the sources and the time checked, and ALWAYS give the FlightAware and Google
        links. Never invent times.

        Args:
            flight_number: e.g. "TG 676", "tg676" or "การบินไทย 676".
            date: YYYY-MM-DD of departure; empty = today.
        """
        return svc.status(flight_number, date)

    def flight_prices(origin: str, destination: str, depart_date: str, return_date: str = "",
                      adults: int = 1, cabin: str = "economy") -> dict:  # fmt: skip
        """Indicative airfares from a web search (not bookable quotes). Say they are rough
        prices from a web search today and real fares change constantly; cite sources; ALWAYS
        give the Google Flights and Skyscanner links. Use convert_currency to give THB when the
        prices are in another currency. Never invent a price.

        Args:
            origin: city/airport name (Thai or English) or IATA code, e.g. "กรุงเทพ" or BKK.
            destination: city/airport name or IATA code, e.g. "โตเกียว" or NRT.
            depart_date: YYYY-MM-DD.
            return_date: YYYY-MM-DD for a round trip, empty for one way.
            adults: number of adult passengers.
            cabin: economy, premium_economy, business or first.
        """
        return svc.prices(origin, destination, depart_date, return_date, adults, cabin)

    def track_flight(flight_number: str, date: str, notify: bool = True) -> dict:
        """Watch a flight and DM the requester about a delay of 15+ minutes, a gate change,
        cancellation and landing. Needs a flight-data provider key; without one it says so.
        notify=false stops tracking that flight.

        Args:
            flight_number: e.g. "TG 676".
            date: YYYY-MM-DD of departure.
            notify: true to start tracking, false to stop.
        """
        flight = normalize_flight(flight_number)
        if flight is None:
            return {"error": "could not read the flight number; use e.g. 'TG 676'"}
        day = (date or "").strip() or today().isoformat()
        try:
            date_cls.fromisoformat(day)
        except ValueError:
            return {"error": "date must be YYYY-MM-DD"}
        links = flight_links(flight)
        if not notify:
            return {"status": "stopped" if tracker.remove(guild_id, user_id, flight, day)
                    else "not_tracking", "flight": flight["display"], "date": day}  # fmt: skip
        if tracker.provider is None:
            return {
                "status": "unavailable",
                "reason": "live tracking needs a flight-data key (not configured)",
                "flight": flight["display"],
                "links": links,
            }
        res = tracker.add(guild_id, user_id, flight, day)
        if "error" in res:
            return {"status": "failed", "reason": res["error"], "links": links,
                    **({"max": res["max"]} if "max" in res else {})}  # fmt: skip
        return {"status": "tracking", "flight": flight["display"], "date": day,
                "polls_every_minutes": POLL_EVERY_S // 60, "starts_6h_before_departure": True,
                "links": links, "current_status": res["info"].get("status")}  # fmt: skip

    track_flight.salid_group = "travel"
    flight_status.salid_group = "travel"
    flight_prices.salid_group = "travel"
    return [flight_status, flight_prices, track_flight]


FLIGHT_PROMPT = (
    "\nเที่ยวบิน: ถามสถานะ/เวลา/เกต ใช้ flight_status (ใส่เลขเที่ยวบินตามที่ผู้ใช้พิมพ์ได้เลย); ถามราคาตั๋ว ใช้ flight_prices. "
    "ถ้าผลเป็น confidence “web” ให้บอกว่าข้อมูลมาจากการค้นเว็บ พร้อมแหล่งที่มาและเวลาที่ตรวจ และแนบลิงก์ FlightAware กับ Google ทุกครั้ง; "
    "ราคาต้องบอกว่า “ราคาโดยประมาณจากการค้นเว็บ ณ วันนี้ ราคาจริงเปลี่ยนตลอด” และแนบลิงก์ Google Flights กับ Skyscanner ทุกครั้ง. "
    "ห้ามแต่งเวลา เกต หรือราคาเองเด็ดขาด ถ้าไม่พบให้บอกตรงๆ และให้ลิงก์ไปตรวจเอง. "
    "ถ้าผู้ใช้อยากให้คอยเตือนเที่ยวบินใช้ track_flight (ถ้าตอบว่า unavailable ให้บอกว่าต้องมีคีย์ข้อมูลเที่ยวบิน และให้ลิงก์ FlightAware). "
    "ถ้ามีคนพูดลอยๆ ว่าจะบินเมื่อไหร่ อย่าค้นหรือติดตามเอง รอให้ผู้ใช้ขอก่อน. ราคาที่ไม่ใช่บาท แปลงด้วย convert_currency เมื่อผู้ใช้อยากได้เป็นบาท."
)
