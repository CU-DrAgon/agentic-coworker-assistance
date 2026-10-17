"""Travel time and routes (Google Routes API v2), as one agent tool `travel_time`.

Known places (`places.py` visibility rules, via `geo.py`) can be used as origin / destination.
An address the audience may not know is used for the computation only: the answer then carries
the place NAME and no address and no map link (a link would contain the address)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

from .geo import Endpoint, parse_when, resolve_endpoint
from .gmaps import GoogleHttp, MapsError
from .i18n import map_language_code
from .places import PlacesCtx
from .weather import RAIN_NOTE_PCT, WeatherService

logger = logging.getLogger(__name__)

ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
CACHE_SECONDS = 300
MODES = {"drive": "DRIVE", "transit": "TRANSIT", "walk": "WALK", "two_wheeler": "TWO_WHEELER"}
LINK_MODE = {
    "drive": "driving",
    "transit": "transit",
    "walk": "walking",
    "two_wheeler": "two-wheeler",
}
_ALIAS = {
    "driving": "drive", "car": "drive", "รถ": "drive", "รถยนต์": "drive", "ขับรถ": "drive",
    "bts": "transit", "mrt": "transit", "bus": "transit", "รถไฟฟ้า": "transit", "รถเมล์": "transit",
    "walking": "walk", "เดิน": "walk", "motorcycle": "two_wheeler", "motorbike": "two_wheeler",
    "มอเตอร์ไซค์": "two_wheeler", "two-wheeler": "two_wheeler", "twowheeler": "two_wheeler",
}  # fmt: skip
BASE_MASK = "routes.duration,routes.staticDuration,routes.distanceMeters,routes.description,routes.travelAdvisory"
TRANSIT_MASK = BASE_MASK + ",routes.legs.steps.transitDetails"


def directions_link(origin: str, destination: str, mode: str) -> str:
    return (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={quote(origin, safe='')}&destination={quote(destination, safe='')}"
        f"&travelmode={LINK_MODE.get(mode, 'driving')}"
    )


def _secs(v) -> int | None:
    try:
        return int(float(str(v).rstrip("s")))
    except (TypeError, ValueError):
        return None


def fmt_duration(sec: int, lang: str) -> str:
    m = max(1, round(sec / 60))
    h, mi = divmod(m, 60)
    if lang != "th":
        return f"{h} h {mi} min" if h and mi else (f"{h} h" if h else f"{mi} min")
    return f"{h} ชม. {mi} นาที" if h and mi else (f"{h} ชม." if h else f"{mi} นาที")


def _round5(dt: datetime) -> str:
    return (dt - timedelta(minutes=dt.minute % 5, seconds=dt.second)).strftime("%Y%m%d%H%M")


def _utc(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class RouteService:
    """Routes API calls with a 5-minute cache of identical (origin, destination, mode, time)."""

    def __init__(self, http: GoogleHttp, clock: Callable[[], float] = time.monotonic):
        self.http, self.clock = http, clock
        self._cache: dict = {}

    def compute(self, origin, destination, mode, *, depart=None, arrive=None, lang="en") -> dict:
        when = ("a", _round5(arrive)) if arrive else ("d", _round5(depart) if depart else "now")
        key = (origin, destination, mode, when, lang)
        hit = self._cache.get(key)
        if hit and hit[0] > self.clock():
            return hit[1]
        body: dict = {
            "origin": {"address": origin}, "destination": {"address": destination},
            "travelMode": MODES[mode], "languageCode": map_language_code(lang),
            "regionCode": "TH", "computeAlternativeRoutes": False, "units": "METRIC",
        }  # fmt: skip
        if mode in ("drive", "two_wheeler"):
            body["routingPreference"] = "TRAFFIC_AWARE"
        if depart:
            body["departureTime"] = _utc(depart)
        if arrive:
            body["arrivalTime"] = _utc(arrive)
        data = self.http.post(ROUTES_URL, body, TRANSIT_MASK if mode == "transit" else BASE_MASK)
        routes = data.get("routes") or []
        if not routes:
            raise MapsError("no_route", "zero_results")
        r = routes[0]
        lines = []
        for leg in r.get("legs") or []:
            for st in leg.get("steps") or []:
                tl = (st.get("transitDetails") or {}).get("transitLine") or {}
                name = tl.get("nameShort") or tl.get("name")
                if name and name not in lines:
                    lines.append(name)
        dur = _secs(r.get("duration"))
        if dur is None:
            raise MapsError("no_route", "no_duration")
        out = {
            "duration_s": dur, "static_s": _secs(r.get("staticDuration")),
            "distance_m": r.get("distanceMeters"), "summary": r.get("description", ""),
            "lines": lines,
        }  # fmt: skip
        self._cache[key] = (self.clock() + CACHE_SECONDS, out)
        return out

    def with_arrival(self, origin, destination, mode, arrive, now, lang) -> tuple[dict, datetime]:
        """Route that gets there by `arrive`, and when to leave. Transit has arrivalTime; the others
        take two passes (guess, then depart at arrive - duration)."""
        if mode == "transit":
            r = self.compute(origin, destination, mode, arrive=arrive, lang=lang)
            return r, arrive - timedelta(seconds=r["duration_s"])
        guess = arrive - timedelta(minutes=45)
        first = self.compute(
            origin, destination, mode, depart=guess if guess > now else None, lang=lang
        )
        leave = arrive - timedelta(seconds=first["duration_s"])
        if leave <= now + timedelta(minutes=1):
            return first, leave
        second = self.compute(origin, destination, mode, depart=leave, lang=lang)
        return second, arrive - timedelta(seconds=second["duration_s"])


@dataclass
class TravelCtx:
    routes: RouteService
    places: PlacesCtx | None
    lang: str
    zone: object
    now: Callable[[], datetime]
    weather: WeatherService | None = None


class TravelTools:
    def __init__(self, ctx: TravelCtx):
        self.c = ctx

    def _coords(self, ep: Endpoint):
        p = ep.place
        if p is not None and p.lat is not None and p.lng is not None:
            return p.lat, p.lng
        return self.c.weather.geo.locate(ep.address, self.c.lang) if self.c.weather else None

    def travel_time(
        self,
        origin: str,
        destination: str,
        mode: str = "drive",
        depart_at: str = "now",
        arrive_by: str = "",
    ) -> dict:
        """Roughly how long it takes to get from A to B and the route (Google Maps), now or at a
        given time, with traffic. Always give the result as an estimate, plus maps_link.

        Args:
            origin: a place name Salid remembers ("บ้านผม", "ออฟฟิศ", "สาขาบางนา") or any address.
            destination: same.
            mode: drive (default), transit (BTS/MRT/bus), walk or two_wheeler (motorcycle).
            depart_at: "now" (default), or a time: ISO 8601, "พรุ่งนี้ 8 โมง", "17:30".
            arrive_by: when you must be there ("9 โมง", ISO); returns when to leave. Leave
                depart_at as "now" when using this.
        """
        c = self.c
        lang = c.lang
        mode = _ALIAS.get(
            (mode or "drive").strip().casefold(), (mode or "drive").strip().casefold()
        )
        if mode not in MODES:
            return {"error": "bad_mode", "detail": "use drive, transit, walk or two_wheeler"}
        if not (origin or "").strip() or not (destination or "").strip():
            return {"error": "need_info", "detail": "both origin and destination are needed"}
        now = c.now()
        try:
            depart = parse_when(depart_at, now, c.zone)
            arrive = parse_when(arrive_by, now, c.zone)
        except ValueError as e:
            return {"error": "bad_time", "detail": str(e)}
        if arrive is not None:
            depart = None
            if arrive <= now:
                return {"error": "bad_time", "detail": "arrive_by is in the past"}
        if depart is not None and depart <= now + timedelta(minutes=1):
            depart = None  # the past or now means "now"
        a = resolve_endpoint(c.places, origin)
        b = resolve_endpoint(c.places, destination)
        hidden = not (a.reveal and b.reveal)
        link = None if hidden else directions_link(a.address, b.address, mode)
        try:
            if arrive is not None:
                r, leave = c.routes.with_arrival(a.address, b.address, mode, arrive, now, lang)
            else:
                r = c.routes.compute(a.address, b.address, mode, depart=depart, lang=lang)
                leave = None
        except MapsError as e:
            if e.code == "no_route":
                return {"error": "no_route", "maps_link": link,
                        "detail": "no route found between these places"}  # fmt: skip
            if e.code != "maps_unavailable":
                return {"error": e.code, "maps_link": link, "detail": e.reason}
            return {
                "error": "maps_unavailable", "maps_link": link,
                "instruction": "Google Maps cannot be reached right now, so there is no time "
                "estimate. Say so briefly, do not guess the duration, "
                + ("and give the maps_link so they can open it." if link else
                   "and do not say the private address."),
            }  # fmt: skip
        dur, static = r["duration_s"], r["static_s"]
        out: dict = {
            "status": "ok", "mode": mode,
            "from": a.label, "to": b.label,
            "duration_min": round(dur / 60), "duration_text": fmt_duration(dur, lang),
            "when": ("arrive by " + f"{arrive.astimezone(c.zone):%H:%M}") if arrive else (
                f"depart {depart.astimezone(c.zone):%Y-%m-%d %H:%M}" if depart else "now"),
        }  # fmt: skip
        if a.reveal and a.place:
            out["from_address"] = a.address
        if b.reveal and b.place:
            out["to_address"] = b.address
        if r["distance_m"] is not None:
            out["distance_km"] = round(r["distance_m"] / 1000, 1)
        if static and mode in ("drive", "two_wheeler"):
            out["typical_min"] = round(static / 60)
            delay = dur - static
            if delay >= 300 and delay >= 0.1 * static:
                m = round(delay / 60)
                out["traffic_note"] = (
                    f"~{m} min slower than usual because of traffic" if lang != "th"
                    else f"รถติดกว่าปกติ ~{m} นาที")  # fmt: skip
        if r["summary"]:
            out["route_summary"] = ("via " if lang != "th" else "ผ่าน") + r["summary"]
        if r["lines"]:
            out["transit_lines"] = r["lines"]
        if leave is not None:
            out["leave_by"] = f"{leave.astimezone(c.zone):%H:%M}"
        if link:
            out["maps_link"] = link
        else:
            out["instruction"] = (
                "A place here may be private to someone listening: say only its NAME, never its "
                "address, and do not give a map link. The duration is fine to say."
            )
        rain = self._rain(a, leave or depart, now)
        if rain is not None and rain >= RAIN_NOTE_PCT:
            out["rain_note"] = (
                f"rain likely ({round(rain)}%), it may take longer" if lang != "th"
                else f"ฝนตก ({round(rain)}%) อาจใช้เวลานานขึ้น")  # fmt: skip
        out["estimate"] = "this is a rough estimate; real traffic can differ"
        return out

    def _rain(self, origin: Endpoint, when, now):
        if self.c.weather is None:
            return None
        try:
            loc = self._coords(origin)
        except MapsError:
            return None
        return self.c.weather.rain_at(*loc, when, now) if loc else None


def build_travel_tools(ctx: TravelCtx) -> list:
    return [TravelTools(ctx).travel_time]


def leave_line(route: dict, origin: str, title: str, place: str, lang: str) -> str | None:
    """The morning-brief line for a computed arrive_by route (see maps_bot)."""
    from .i18n import t

    if route.get("status") != "ok" or "leave_by" not in route:
        return None
    return t("bf_leave_line", lang, origin=origin, leave=route["leave_by"],
             minutes=route["duration_min"], title=title, place=place)  # fmt: skip
