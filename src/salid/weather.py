"""Weather for Salid: Google Maps Platform Weather API (WeatherNext models), plus an optional
WeatherNext 2 ensemble read from BigQuery. Locations come from remembered places (visibility rules
in `geo.py`) or the Places API text search. Answers carry a place NAME, never an address."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .geo import parse_when
from .gmaps import Geocoder, GoogleHttp, MapsError
from .places import PlacesCtx

logger = logging.getLogger(__name__)

BASE = "https://weather.googleapis.com/v1"
SOURCE = "Google Weather (WeatherNext)"
BANGKOK = (13.7563, 100.5018)
CACHE_SECONDS = 600
MAX_HOURS = 240
RAIN_NOTE_PCT = 60
_DAY_WORDS = {"today": 0, "วันนี้": 0, "tomorrow": 1, "พรุ่งนี้": 1, "มะรืนนี้": 2, "มะรืน": 2}


# ---------------------------------------------------------------------------- parsing
def _g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return default if d is None else d


def norm_conditions(d: dict) -> dict:
    """One current / hourly / day-part block -> flat numbers (None when absent)."""
    prob = _g(d, "precipitation", "probability", "percent")
    qpf = _g(d, "precipitation", "qpf", "quantity")
    return {
        "condition": _g(d, "weatherCondition", "description", "text", default=""),
        "temp_c": _g(d, "temperature", "degrees"),
        "feels_c": _g(d, "feelsLikeTemperature", "degrees"),
        "rain_pct": prob,
        "rain_mm": qpf,
        "storm_pct": d.get("thunderstormProbability"),
        "wind_kmh": _g(d, "wind", "speed", "value"),
        "uv": d.get("uvIndex"),
    }


def norm_day(d: dict) -> dict:
    day, night = d.get("daytimeForecast") or {}, d.get("nighttimeForecast") or {}
    a, b = norm_conditions(day), norm_conditions(night)

    def worst(k, fn=max):
        vals = [x[k] for x in (a, b) if x[k] is not None]
        return fn(vals) if vals else None

    out = dict(a)
    out["rain_pct"] = worst("rain_pct")
    out["storm_pct"] = worst("storm_pct")
    out["rain_mm"] = round(sum(x["rain_mm"] for x in (a, b) if x["rain_mm"] is not None), 1) if (
        a["rain_mm"] is not None or b["rain_mm"] is not None
    ) else None  # fmt: skip
    out["wind_kmh"] = worst("wind_kmh")
    out["temp_c"] = _g(d, "maxTemperature", "degrees", default=a["temp_c"])
    out["min_c"] = _g(d, "minTemperature", "degrees")
    out["feels_c"] = _g(d, "feelsLikeMaxTemperature", "degrees", default=a["feels_c"])
    dd = d.get("displayDate") or {}
    out["date"] = f"{dd.get('year', 0):04d}-{dd.get('month', 0):02d}-{dd.get('day', 0):02d}"
    return out


def summarize(c: dict, lang: str) -> str:
    """One natural line from normalised conditions."""
    th = lang != "en"
    bits = [c.get("condition") or ("อากาศ" if th else "Weather")]
    t, f = c.get("temp_c"), c.get("feels_c")
    if t is not None:
        lo = c.get("min_c")
        rng = f"{round(lo)}-{round(t)}°C" if lo is not None else f"{round(t)}°C"
        bits.append(("อุณหภูมิ " if th else "") + rng)
        if f is not None and abs(f - t) >= 3 and lo is None:
            bits.append(f"รู้สึกเหมือน {round(f)}°C" if th else f"feels like {round(f)}°C")
    p = c.get("rain_pct")
    if p is not None:
        bits.append(f"โอกาสฝน {round(p)}%" if th else f"{round(p)}% chance of rain")
    if (c.get("storm_pct") or 0) >= 30:
        bits.append(f"พายุฝนฟ้าคะนอง {round(c['storm_pct'])}%" if th else "thunderstorms likely")
    if (c.get("uv") or 0) >= 8:
        bits.append(f"UV สูง ({c['uv']})" if th else f"high UV ({c['uv']})")
    line = ", ".join(str(b) for b in bits)
    if (p or 0) >= 50:
        line += " ควรพกร่มค่ะ" if th else " - bring an umbrella"
    return line


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------- BigQuery ensemble
ENSEMBLE_SQL = """
SELECT t.ensemble_member AS member, f.total_precipitation_6hr AS precip_m
FROM `{table}` AS t, UNNEST(t.forecast) AS f
WHERE t.init_time = (SELECT MAX(init_time) FROM `{table}`)
  AND ST_DWITHIN(t.geography, ST_GEOGPOINT(@lng, @lat), @radius_m)
  AND f.time > @t0 AND f.time <= @t1
"""  # assumed WeatherNext layout: one row per member and grid point, `forecast` array of steps


class WeatherNextBigQuerySource:
    """WeatherNext 2 ensemble (64 members): share of members above a rain threshold. Disabled with
    an empty dataset. `run(sql, params) -> rows` is injectable (tests use fake rows)."""

    def __init__(self, dataset: str, table: str = "weathernext_2_0_0", run: Callable | None = None):
        self.table = f"{dataset}.{table}"
        self.enabled = bool(dataset)
        self._run = run

    def _runner(self):
        if self._run:
            return self._run

        def run(sql, params):
            from google.cloud import bigquery  # optional dependency

            client = bigquery.Client()
            cfg = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        k, "TIMESTAMP" if isinstance(v, datetime) else
                        "FLOAT64" if isinstance(v, float) else "INT64", v)
                    for k, v in params.items()
                ]
            )  # fmt: skip
            return [dict(r) for r in client.query(sql, job_config=cfg).result()]

        return run

    def ensemble(self, lat: float, lng: float, start: datetime, hours: int = 6, mm: float = 1.0):
        if not self.enabled:
            raise MapsError("ensemble_unavailable", "not_configured")
        params = {
            "lat": float(lat), "lng": float(lng), "radius_m": 30000,
            "t0": start.astimezone(UTC), "t1": start.astimezone(UTC) + timedelta(hours=hours),
        }  # fmt: skip
        try:
            rows = self._runner()(ENSEMBLE_SQL.format(table=self.table), params)
        except MapsError:
            raise
        except Exception as e:
            logger.warning("weathernext bigquery failed: %s", type(e).__name__)
            raise MapsError("ensemble_unavailable", type(e).__name__) from e
        return parse_ensemble(rows, mm)


def parse_ensemble(rows: list[dict], mm: float = 1.0) -> dict:
    """Rows of {member, precip_m (metres)} -> per-member total (mm) and the share above `mm`."""
    total: dict = {}
    for r in rows:
        if r.get("precip_m") is None:
            continue
        total[r["member"]] = total.get(r["member"], 0.0) + float(r["precip_m"]) * 1000.0
    if not total:
        raise MapsError("ensemble_unavailable", "no_rows")
    vals = sorted(total.values())
    n = len(vals)
    return {
        "members": n,
        "rain_probability_pct": round(100 * sum(v > mm for v in vals) / n),
        "threshold_mm": mm,
        "median_mm": round(vals[n // 2], 1),
        "p90_mm": round(vals[min(n - 1, math.ceil(0.9 * n) - 1)], 1),
        "max_mm": round(vals[-1], 1),
    }


# ---------------------------------------------------------------------------- service
class WeatherService:
    """Long-lived: Weather API calls with a 10-minute cache."""

    def __init__(
        self, http: GoogleHttp, geo: Geocoder, bq: WeatherNextBigQuerySource | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):  # fmt: skip
        self.http, self.geo, self.bq, self.clock = http, geo, bq, clock
        self._cache: dict = {}

    def _cached(self, key, fn):
        hit = self._cache.get(key)
        if hit and hit[0] > self.clock():
            return hit[1]
        val = fn()
        self._cache[key] = (self.clock() + CACHE_SECONDS, val)
        return val

    @staticmethod
    def _loc(lat, lng, lang, **extra) -> dict:
        return {
            "location.latitude": round(lat, 4), "location.longitude": round(lng, 4),
            "languageCode": "en" if lang == "en" else "th", "unitsSystem": "METRIC", **extra,
        }  # fmt: skip

    def current(self, lat, lng, lang) -> dict:
        p = self._loc(lat, lng, lang)
        return norm_conditions(
            self._cached(("cur", *p.values()), lambda: self.http.get(f"{BASE}/currentConditions:lookup", p))
        )  # fmt: skip

    def hours(self, lat, lng, lang, ahead: int) -> list[dict]:
        """Hourly entries from now through `ahead` hours (<= 240), each with its start time."""
        ahead = max(1, min(MAX_HOURS, ahead))

        def fetch():
            out, token = [], None
            for _ in range(11):
                p = self._loc(lat, lng, lang, hours=ahead, pageSize=24)
                if token:
                    p["pageToken"] = token
                d = self.http.get(f"{BASE}/forecast/hours:lookup", p)
                out += d.get("forecastHours") or []
                token = d.get("nextPageToken")
                if not token:
                    break
            return out

        rows = self._cached(("hrs", round(lat, 3), round(lng, 3), lang, ahead), fetch)
        return [
            {**norm_conditions(r), "start": _parse_ts(_g(r, "interval", "startTime", default=""))}
            for r in rows
            if _g(r, "interval", "startTime")
        ]

    def days(self, lat, lng, lang, count: int = 10) -> list[dict]:
        p = self._loc(lat, lng, lang, days=count, pageSize=10)
        d = self._cached(
            ("day", *p.values()), lambda: self.http.get(f"{BASE}/forecast/days:lookup", p)
        )
        return [norm_day(x) for x in d.get("forecastDays") or []]

    def rain_at(self, lat, lng, when: datetime | None, now: datetime) -> float | None:
        """Rain probability (%) at `when` (None = now), or None when unknown. Never raises."""
        try:
            if when is None or when <= now + timedelta(minutes=30):
                return self.current(lat, lng, "th").get("rain_pct")
            ahead = math.ceil((when - now).total_seconds() / 3600) + 1
            if ahead > MAX_HOURS:
                return None
            rows = self.hours(lat, lng, "th", ahead)
            best = min(rows, key=lambda r: abs((r["start"] - when).total_seconds()), default=None)
            return best["rain_pct"] if best else None
        except (MapsError, ValueError, KeyError):
            return None


# ---------------------------------------------------------------------------- tools
@dataclass
class WeatherCtx:
    service: WeatherService
    places: PlacesCtx | None
    lang: str
    zone: object
    now: Callable[[], datetime]


class WeatherTools:
    def __init__(self, ctx: WeatherCtx):
        self.c = ctx

    def locate(self, place: str):
        """(label, lat, lng) for a place name / address; the office, else Bangkok, when empty."""
        from .geo import find_place

        c = self.c
        pc = c.places
        p = find_place(pc, place if place.strip() else "office")
        if p is not None:
            if p.lat is None or p.lng is None:
                got = c.service.geo.locate(p.address or p.name, c.lang)
                if got is None:
                    raise MapsError("place_not_found", "geocode")
                p.lat, p.lng = got
                try:
                    pc.repo.put(p)
                except Exception:  # best effort cache of the coordinates
                    logger.info("could not store coordinates", exc_info=True)
            return p.name, p.lat, p.lng
        if not place.strip():
            return ("กรุงเทพฯ" if c.lang != "en" else "Bangkok"), *BANGKOK
        got = c.service.geo.locate(place.strip(), c.lang)
        if got is None:
            raise MapsError("place_not_found", "geocode")
        return place.strip(), *got

    def weather(self, place: str = "", when: str = "now", detail: str = "summary") -> dict:
        """Weather and forecast for a place (Google Weather API, WeatherNext models).

        Args:
            place: a remembered place ("สาขาบางนา", "ออฟฟิศ") or any address / area name. Empty =
                the asker's office, else Bangkok.
            when: "now", "today", "tomorrow", a date YYYY-MM-DD, a time ("17:00", "พรุ่งนี้ 8 โมง")
                or ISO 8601; forecasts reach 10 days (hourly 240 h).
            detail: "summary" (default), "hourly" (next hours), or "ensemble" (rain probability from
                the 64-member WeatherNext 2 ensemble, only if configured).
        """
        c = self.c
        now = c.now()
        try:
            label, lat, lng = self.locate(place or "")
            w = (when or "now").strip()
            key = w.casefold()
            base = {"place": label, "source": SOURCE, "estimate": True}
            day_off = _DAY_WORDS.get(key)
            if day_off is None and len(w) == 10 and w[4] == "-":
                try:
                    day_off = (datetime.fromisoformat(w).date() - now.date()).days
                except ValueError:
                    day_off = None
            at = (
                None
                if day_off is not None or key in ("", "now", "ตอนนี้")
                else parse_when(w, now, c.zone)
            )
            if day_off is not None:
                if not 0 <= day_off <= 9:
                    return {"error": "out_of_range", "detail": "forecast covers the next 10 days"}
                at = (now + timedelta(days=day_off)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                at = max(at, now)
            if detail == "ensemble":
                return self._ensemble(base, lat, lng, at or now)
            if detail == "hourly":
                start = at or now
                ahead = math.ceil((start - now).total_seconds() / 3600) + 12
                if ahead > MAX_HOURS:
                    return {"error": "out_of_range", "detail": "hourly forecast covers 240 hours"}
                rows = [r for r in c.service.hours(lat, lng, c.lang, ahead)
                        if r["start"] >= start - timedelta(minutes=59)][:12]  # fmt: skip
                return {**base, "hours": [
                    {"time": f"{r['start'].astimezone(c.zone):%a %H:%M}", **_brief(r)} for r in rows
                ]}  # fmt: skip
            if at is None:
                cur = c.service.current(lat, lng, c.lang)
                return {**base, "when": "now", **_brief(cur), "summary": summarize(cur, c.lang)}
            if day_off is not None:
                days = c.service.days(lat, lng, c.lang)
                want = (now + timedelta(days=day_off)).date().isoformat()
                row = next((d for d in days if d["date"] == want), None)
                if row is None:
                    return {"error": "no_forecast", "detail": want}
                return {**base, "when": want, **_brief(row), "summary": summarize(row, c.lang)}
            ahead = math.ceil((at - now).total_seconds() / 3600) + 1
            if ahead > MAX_HOURS:
                return {"error": "out_of_range", "detail": "hourly forecast covers 240 hours"}
            rows = c.service.hours(lat, lng, c.lang, max(ahead, 1))
            row = min(rows, key=lambda r: abs((r["start"] - at).total_seconds()), default=None)
            if row is None:
                return {"error": "no_forecast"}
            return {**base, "when": f"{at.astimezone(c.zone):%Y-%m-%d %H:%M}", **_brief(row),
                    "summary": summarize(row, c.lang)}  # fmt: skip
        except MapsError as e:
            return _error(e)
        except ValueError as e:
            return {"error": "bad_time", "detail": str(e)}

    def _ensemble(self, base, lat, lng, start) -> dict:
        bq = self.c.service.bq
        if bq is None or not bq.enabled:
            return {"error": "ensemble_unavailable",
                    "detail": "the WeatherNext ensemble is not configured; use detail=summary"}  # fmt: skip
        res = bq.ensemble(lat, lng, start)
        return {**base, "when": f"{start.astimezone(self.c.zone):%Y-%m-%d %H:%M}", "ensemble": res,
                "source": "WeatherNext 2 ensemble (BigQuery)"}  # fmt: skip


def _brief(c: dict) -> dict:
    keys = (
        "condition",
        "temp_c",
        "min_c",
        "feels_c",
        "rain_pct",
        "rain_mm",
        "storm_pct",
        "wind_kmh",
        "uv",
    )
    return {k: c[k] for k in keys if c.get(k) not in (None, "")}


def _error(e: MapsError) -> dict:
    if e.code == "maps_unavailable":
        return {"error": "maps_unavailable", "detail": "the weather service is not available "
                f"right now ({e.reason}); say so briefly and do not guess the weather"}  # fmt: skip
    return {"error": e.code, "detail": e.reason}


def build_weather_tools(ctx: WeatherCtx) -> list:
    return [WeatherTools(ctx).weather]
