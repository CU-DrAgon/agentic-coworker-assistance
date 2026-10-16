# ruff: noqa: F811, F401
"""Travel time (Routes API) and weather (Weather API / WeatherNext): requests, parsing, places
visibility, fallbacks, caches, flags and the morning-brief lines. Fake HTTP only; nothing real."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import pytest
from test_briefings import GUILD, STAFF, at, meeting, set_features
from test_coordinator import BOSS, PEER, PUBLIC, world
from test_modes import bot  # noqa: F401  (fixture)
from test_places import (
    ADDR,
    CHANNELS,
    LOW,
    MGR,
    NAMES,
    OWNER,
    RANKS,
    G,
    viewers_of,
)
from test_places import (
    BOSS as PBOSS,
)
from test_places import (
    STAFF as PSTAFF,
)
from test_places import (
    ctx as places_ctx,
)

from salid import places as pl
from salid.coo_store import SQLiteDocStore
from salid.geo import parse_when
from salid.gmaps import Geocoder, GoogleHttp, MapsError
from salid.places import PlaceRepo
from salid.travel import ROUTES_URL, RouteService, TravelCtx, TravelTools, directions_link
from salid.weather import (
    WeatherCtx,
    WeatherNextBigQuerySource,
    WeatherService,
    WeatherTools,
    parse_ensemble,
)

ZONE = ZoneInfo("Asia/Bangkok")
NOW = datetime(2026, 10, 14, 10, 0, tzinfo=ZONE)
PROJECT = "proj-x"


class Net:
    """A fake Google: records requests, answers by URL prefix (callable or JSON)."""

    def __init__(self, **routes):
        self.routes, self.calls = routes, []

    def handler(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else None
        self.calls.append(
            SimpleNamespace(url=str(req.url), req=req, body=body, headers=req.headers)
        )
        for prefix, resp in self.routes.items():
            if str(req.url).startswith(prefix):
                out = resp(req) if callable(resp) else resp
                if isinstance(out, httpx.Response):
                    return out
                return httpx.Response(200, json=out)
        return httpx.Response(404, json={"error": {"message": "no route"}})

    def http(self, **kw) -> GoogleHttp:
        return GoogleHttp(
            PROJECT, token=lambda: "tok123",
            http=httpx.Client(transport=httpx.MockTransport(self.handler)), **kw,
        )  # fmt: skip


def route(dur=1800, static=1500, km=12300, desc="ทางด่วน", **extra):
    return {"routes": [{"duration": f"{dur}s", "staticDuration": f"{static}s",
                        "distanceMeters": km, "description": desc, **extra}]}  # fmt: skip


def travel(net, pc=None, lang="th", weather=None, now=NOW):
    return TravelTools(
        TravelCtx(RouteService(net.http()), pc, lang, ZONE, lambda: now, weather)
    )  # fmt: skip


# ------------------------------------------------------------------ time phrases
@pytest.mark.parametrize(
    "text, expect",
    [
        ("พรุ่งนี้ 8 โมง", "2026-10-15 08:00"), ("พรุ่งนี้ 8 โมงเช้า", "2026-10-15 08:00"),
        ("วันนี้ 17:30", "2026-10-14 17:30"), ("บ่าย 2", "2026-10-14 14:00"),
        ("บ่ายโมงครึ่ง", "2026-10-14 13:30"), ("2 ทุ่ม", "2026-10-14 20:00"),
        ("5 โมงเย็น", "2026-10-14 17:00"), ("tomorrow 8am", "2026-10-15 08:00"),
        ("tomorrow 6pm", "2026-10-15 18:00"), ("๘:๓๐", "2026-10-15 08:30"),  # past today -> tomorrow
        ("เที่ยง", "2026-10-14 12:00"), ("2026-10-20T09:00", "2026-10-20 09:00"),
        ("2026-10-20T02:00:00Z", "2026-10-20 09:00"), ("มะรืนนี้ 9 โมง", "2026-10-16 09:00"),
    ],
)  # fmt: skip
def test_parse_when(text, expect):
    assert f"{parse_when(text, NOW, ZONE):%Y-%m-%d %H:%M}" == expect


def test_parse_when_now_and_garbage():
    assert parse_when("", NOW, ZONE) is None and parse_when("now", NOW, ZONE) is None
    assert parse_when("ตอนนี้", NOW, ZONE) is None
    with pytest.raises(ValueError):
        parse_when("sometime soon", NOW, ZONE)


# ------------------------------------------------------------------ request shape
def test_drive_now_request_headers_body_and_traffic_delta():
    net = Net(**{ROUTES_URL: route(dur=2400, static=1500)})
    r = travel(net).travel_time("สยามพารากอน", "สนามบินสุวรรณภูมิ")
    (call,) = net.calls
    h = call.headers
    assert h["authorization"] == "Bearer tok123" and h["x-goog-user-project"] == PROJECT
    assert "x-goog-api-key" not in h and "key=" not in call.url
    assert h["x-goog-fieldmask"] == (
        "routes.duration,routes.staticDuration,routes.distanceMeters,routes.description,"
        "routes.travelAdvisory"
    )
    b = call.body
    assert b["origin"] == {"address": "สยามพารากอน"} and b["travelMode"] == "DRIVE"
    assert b["routingPreference"] == "TRAFFIC_AWARE" and b["regionCode"] == "TH"
    assert b["languageCode"] == "th" and b["computeAlternativeRoutes"] is False
    assert "departureTime" not in b and "arrivalTime" not in b
    assert r["status"] == "ok" and r["duration_min"] == 40 and r["typical_min"] == 25
    assert r["traffic_note"] == "รถติดกว่าปกติ ~15 นาที" and r["distance_km"] == 12.3
    assert r["route_summary"] == "ผ่านทางด่วน" and "estimate" in r
    assert r["maps_link"] == (
        "https://www.google.com/maps/dir/?api=1&origin=%E0%B8%AA%E0%B8%A2%E0%B8%B2%E0%B8%A1"
        + r["maps_link"].split("%E0%B8%AA%E0%B8%A2%E0%B8%B2%E0%B8%A1", 1)[1]
    )
    assert r["maps_link"].endswith("&travelmode=driving")


def test_no_traffic_note_when_the_delay_is_small_and_english_text():
    net = Net(**{ROUTES_URL: route(dur=1560, static=1500)})
    r = travel(net, lang="en").travel_time("A", "B")
    assert "traffic_note" not in r and r["duration_text"] == "26 min"
    assert net.calls[0].body["languageCode"] == "en"
    net2 = Net(**{ROUTES_URL: route(dur=5400, static=3000)})
    r2 = travel(net2, lang="en").travel_time("A", "B")
    assert r2["traffic_note"] == "~40 min slower than usual because of traffic"
    assert r2["duration_text"] == "1 h 30 min"


def test_transit_has_no_routing_preference_and_asks_for_transit_details():
    steps = {"legs": [{"steps": [{}, {"transitDetails": {"transitLine": {"nameShort": "BTS สุขุมวิท"}}},
                                 {"transitDetails": {"transitLine": {"name": "MRT น้ำเงิน"}}}]}]}  # fmt: skip
    net = Net(**{ROUTES_URL: route(dur=3000, static=0, **steps)})
    r = travel(net).travel_time("A", "B", "transit")
    b, h = net.calls[0].body, net.calls[0].headers
    assert b["travelMode"] == "TRANSIT" and "routingPreference" not in b
    assert h["x-goog-fieldmask"].endswith(",routes.legs.steps.transitDetails")
    assert r["transit_lines"] == ["BTS สุขุมวิท", "MRT น้ำเงิน"] and "traffic_note" not in r
    assert r["maps_link"].endswith("travelmode=transit")


def test_walk_and_two_wheeler_modes_and_aliases():
    net = Net(**{ROUTES_URL: route()})
    t = travel(net)
    t.travel_time("A", "B", "walk")
    t.travel_time("A2", "B2", "two_wheeler")
    t.travel_time("A3", "B3", "มอเตอร์ไซค์")
    assert [c.body["travelMode"] for c in net.calls] == ["WALK", "TWO_WHEELER", "TWO_WHEELER"]
    assert "routingPreference" not in net.calls[0].body
    assert net.calls[1].body["routingPreference"] == "TRAFFIC_AWARE"
    assert t.travel_time("A", "B", "teleport")["error"] == "bad_mode"


def test_depart_at_thai_phrase_becomes_a_utc_departure_time():
    net = Net(**{ROUTES_URL: route()})
    travel(net).travel_time("A", "B", depart_at="พรุ่งนี้ 8 โมง")
    assert net.calls[0].body["departureTime"] == "2026-10-15T01:00:00Z"
    net2 = Net(**{ROUTES_URL: route()})
    r = travel(net2).travel_time("A", "B", depart_at="2026-10-14T09:00:00+07:00")  # in the past
    assert "departureTime" not in net2.calls[0].body and r["when"] == "now"
    assert travel(net2).travel_time("A", "B", depart_at="whenever")["error"] == "bad_time"


def test_arrive_by_transit_uses_arrival_time_and_computes_leave_by():
    net = Net(**{ROUTES_URL: route(dur=3600, static=0)})
    r = travel(net).travel_time("A", "B", "transit", arrive_by="พรุ่งนี้ 9 โมง")
    assert net.calls[0].body["arrivalTime"] == "2026-10-15T02:00:00Z"
    assert r["leave_by"] == "08:00" and r["when"] == "arrive by 09:00"
    assert travel(net).travel_time("A", "B", arrive_by="2026-10-14T08:00")["error"] == "bad_time"


def test_arrive_by_drive_departs_at_arrival_minus_duration():
    def answer(req):
        dep = json.loads(req.content).get("departureTime")
        return route(dur=1800 if dep is None else 2700, static=1200)

    net = Net(**{ROUTES_URL: answer})
    r = travel(net).travel_time("A", "B", arrive_by="พรุ่งนี้ 9 โมง")
    assert net.calls[0].body["departureTime"] == "2026-10-15T01:15:00Z"
    assert len(net.calls) == 1  # the second pass is the same bucket: cached
    assert "arrivalTime" not in net.calls[0].body  # drive has no arrivalTime
    assert r["leave_by"] == "08:15" and r["duration_min"] == 45


# ------------------------------------------------------------------ errors and fallback
def test_api_not_enabled_returns_maps_unavailable_with_the_link():
    net = Net(**{ROUTES_URL: httpx.Response(403, json={"error": {"status": "PERMISSION_DENIED"}})})
    r = travel(net).travel_time("สยาม", "อโศก", "transit")
    assert r["error"] == "maps_unavailable" and "duration_min" not in r
    assert r["maps_link"] == directions_link("สยาม", "อโศก", "transit")
    assert "do not guess" in r["instruction"]


def test_timeout_and_server_errors_are_maps_unavailable():
    def boom(req):
        raise httpx.ReadTimeout("slow")

    t = travel(Net(**{ROUTES_URL: boom}))
    assert t.travel_time("A", "B")["error"] == "maps_unavailable"
    assert travel(Net(**{ROUTES_URL: httpx.Response(503, json={})})).travel_time("A", "B")["error"] == "maps_unavailable"  # fmt: skip
    http = Net().http()
    assert http.timeout == 8.0


def test_no_route_and_bad_request():
    r = travel(Net(**{ROUTES_URL: {}})).travel_time("A", "B")
    assert r["error"] == "no_route" and r["maps_link"]
    assert travel(Net(**{ROUTES_URL: httpx.Response(400, json={})})).travel_time("A", "B")["error"] == "bad_request"  # fmt: skip


def test_missing_credentials_is_maps_unavailable():
    def no_creds():
        raise MapsError("maps_unavailable", "no_credentials")

    net = Net(**{ROUTES_URL: route()})
    http = GoogleHttp(PROJECT, token=no_creds, http=httpx.Client(transport=httpx.MockTransport(net.handler)))  # fmt: skip
    t = TravelTools(TravelCtx(RouteService(http), None, "th", ZONE, lambda: NOW))
    assert t.travel_time("A", "B")["error"] == "maps_unavailable" and net.calls == []


def test_api_key_is_only_a_fallback_when_oauth_is_rejected_and_is_remembered():
    def answer(req):
        if "authorization" in req.headers:
            return httpx.Response(
                401, json={"error": {"message": "API keys are not supported; use OAuth"}}
            )
        return {"currentTime": "x"}

    net = Net(**{"https://weather.googleapis.com": answer})
    http = net.http(api_key="SECRET-KEY")
    assert http.get("https://weather.googleapis.com/v1/x", {"a": 1}) == {"currentTime": "x"}
    first, second = net.calls
    assert first.headers["authorization"] == "Bearer tok123"
    assert (
        second.headers["x-goog-api-key"] == "SECRET-KEY" and "authorization" not in second.headers
    )
    http.get("https://weather.googleapis.com/v1/x", {"a": 2})
    assert "authorization" not in net.calls[2].headers  # key first from now on
    noKey = Net(**{"https://weather.googleapis.com": answer}).http()
    with pytest.raises(MapsError) as e:
        noKey.get("https://weather.googleapis.com/v1/x", {})
    assert e.value.code == "maps_unavailable" and "SECRET" not in str(e.value)


# ------------------------------------------------------------------ cache
def test_identical_requests_are_cached_for_five_minutes():
    clock = {"t": 0.0}
    net = Net(**{ROUTES_URL: route()})
    svc = RouteService(net.http(), clock=lambda: clock["t"])
    t = TravelTools(TravelCtx(svc, None, "th", ZONE, lambda: NOW))
    t.travel_time("A", "B")
    t.travel_time("A", "B")
    assert len(net.calls) == 1
    t.travel_time("A", "B", "walk")  # another mode: another request
    assert len(net.calls) == 2
    # the same 5-minute departure bucket shares the answer
    t.travel_time("A", "B", depart_at="พรุ่งนี้ 8 โมง")
    t.travel_time("A", "B", depart_at="2026-10-15T08:03")
    assert len(net.calls) == 3
    clock["t"] = 301
    t.travel_time("A", "B")
    assert len(net.calls) == 4


# ------------------------------------------------------------------ places and hiding
@pytest.fixture
def repo():
    return PlaceRepo(SQLiteDocStore(":memory:"))


def home_and_branch(repo):
    home = repo.create(
        G, name="บ้านสมชาย", address="12 ซอยลับ บางนา กรุงเทพ", kind="home", owner_id=PSTAFF,
        created_by=PSTAFF, sources=[pl.make_source(-PSTAFF, PSTAFF, NOW)],
    )  # fmt: skip
    branch = repo.create(
        G, name="สาขาบางนา", address=ADDR, kind="branch", created_by=PBOSS,
        sources=[pl.make_source(PUBLIC_CH, PBOSS, NOW)],
    )  # fmt: skip
    return home, branch


PUBLIC_CH = 10


def test_known_places_resolve_and_a_personal_address_is_hidden_in_a_shared_room(repo):
    home, branch = home_and_branch(repo)
    net = Net(**{ROUTES_URL: route()})
    # a DM: the owner alone hears -> the address may be shown, the link is complete
    dm = places_ctx(repo, PSTAFF, channel=None)
    r = travel(net, dm).travel_time("บ้านผม", "สาขาบางนา")
    assert net.calls[0].body["origin"] == {"address": home.address}
    assert net.calls[0].body["destination"] == {"address": ADDR}
    assert r["from"] == "บ้านสมชาย" and r["from_address"] == home.address
    assert home.address.split()[1] in r["maps_link"] or "%E0%B8%8B" in r["maps_link"]
    # a public channel: still computed from the home address, but only its name is said
    net2 = Net(**{ROUTES_URL: route()})
    room = places_ctx(repo, PSTAFF, channel=PUBLIC_CH)
    r2 = travel(net2, room).travel_time("บ้านผม", "สาขาบางนา")
    assert net2.calls[0].body["origin"] == {"address": home.address}  # used for the computation
    assert r2["status"] == "ok" and r2["duration_min"] == 30  # the duration is fine to say
    dumped = json.dumps(r2, ensure_ascii=False)
    assert home.address not in dumped and "ซอยลับ" not in dumped and "maps_link" not in r2
    assert r2["from"] == "บ้านสมชาย" and "NAME" in r2["instruction"]
    assert r2["to_address"] == ADDR  # the company branch is known to the whole room


def test_hidden_address_is_not_leaked_on_failure_either(repo):
    home, _ = home_and_branch(repo)
    net = Net(**{ROUTES_URL: httpx.Response(403, json={})})
    r = travel(net, places_ctx(repo, PSTAFF, channel=PUBLIC_CH)).travel_time("บ้านผม", "สาขาบางนา")
    assert r["error"] == "maps_unavailable" and r["maps_link"] is None
    assert "ซอยลับ" not in json.dumps(r, ensure_ascii=False)


def test_someone_elses_personal_place_is_never_used(repo):
    home, _ = home_and_branch(repo)
    net = Net(**{ROUTES_URL: route()})
    # Peer asks about "บ้านสมชาย": not theirs -> treated as plain words, never the real address
    r = travel(net, places_ctx(repo, 27, channel=None)).travel_time("บ้านสมชาย", "สาขาบางนา")
    assert net.calls[0].body["origin"] == {"address": "บ้านสมชาย"}
    assert home.address not in json.dumps(r, ensure_ascii=False)
    # even when the owner shared it with everyone, "personal only for its owner" holds
    home.share_everyone = True
    repo.put(home)
    net2 = Net(**{ROUTES_URL: route()})
    travel(net2, places_ctx(repo, 27, channel=None)).travel_time("บ้านสมชาย", "สาขาบางนา")
    assert net2.calls[0].body["origin"] == {"address": "บ้านสมชาย"}


def test_a_company_place_the_asker_may_not_know_is_not_used(repo):
    repo.create(
        G, name="ห้องลับ", address="1 Secret Rd Bangkok", kind="site", created_by=PBOSS,
        sources=[pl.make_source(MGR, PBOSS, NOW)],
    )  # fmt: skip
    net = Net(**{ROUTES_URL: route()})
    travel(net, places_ctx(repo, PSTAFF, channel=None)).travel_time("ห้องลับ", "สาขาบางนา")
    assert net.calls[0].body["origin"] == {"address": "ห้องลับ"}


def test_office_word_finds_the_head_office(repo):
    hq = repo.create(
        G, name="สำนักงานใหญ่", address="1 Silom Rd Bangkok", kind="hq", created_by=PBOSS,
        sources=[pl.make_source(PUBLIC_CH, PBOSS, NOW)],
    )  # fmt: skip
    net = Net(**{ROUTES_URL: route()})
    travel(net, places_ctx(repo, PSTAFF, channel=None)).travel_time("ออฟฟิศ", "A")
    assert net.calls[0].body["origin"] == {"address": hq.address}


# ------------------------------------------------------------------ weather
CUR = {
    "weatherCondition": {"description": {"text": "ฝนฟ้าคะนอง"}},
    "temperature": {"degrees": 29.4}, "feelsLikeTemperature": {"degrees": 34.0},
    "precipitation": {"probability": {"percent": 70}, "qpf": {"quantity": 2.5}},
    "thunderstormProbability": 40, "wind": {"speed": {"value": 12}}, "uvIndex": 9,
}  # fmt: skip
CUR_URL = "https://weather.googleapis.com/v1/currentConditions:lookup"
HRS_URL = "https://weather.googleapis.com/v1/forecast/hours:lookup"
DAY_URL = "https://weather.googleapis.com/v1/forecast/days:lookup"
PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
GEO = {"places": [{"location": {"latitude": 13.67, "longitude": 100.6}}]}


def hour(h, pct, day=14):
    t = (datetime(2026, 10, day, 0, 0, tzinfo=ZONE) + timedelta(hours=h)).astimezone(UTC)
    c = {**CUR, "precipitation": {"probability": {"percent": pct}}}
    return {**c, "interval": {"startTime": t.strftime("%Y-%m-%dT%H:%M:%SZ")}}


def weather_tools(net, pc=None, bq=None, lang="th"):
    http = net.http()
    svc = WeatherService(http, Geocoder(http), bq)
    return WeatherTools(WeatherCtx(svc, pc, lang, ZONE, lambda: NOW)), svc


def test_current_conditions_request_and_summary():
    net = Net(**{CUR_URL: CUR})
    w, _ = weather_tools(net)
    r = w.weather()  # no place: Bangkok
    (c,) = net.calls
    assert (
        c.headers["authorization"] == "Bearer tok123"
        and c.headers["x-goog-user-project"] == PROJECT
    )
    q = dict(c.req.url.params)
    assert q["location.latitude"] == "13.7563" and q["unitsSystem"] == "METRIC" and q["languageCode"] == "th"  # fmt: skip
    assert r["place"] == "กรุงเทพฯ" and r["source"].startswith("Google Weather (WeatherNext)")
    assert r["rain_pct"] == 70 and r["storm_pct"] == 40 and r["uv"] == 9 and r["wind_kmh"] == 12
    assert "โอกาสฝน 70%" in r["summary"] and "ควรพกร่ม" in r["summary"]


def test_a_named_area_is_geocoded_once_and_cached():
    net = Net(**{CUR_URL: CUR, PLACES_URL: GEO})
    w, _ = weather_tools(net, lang="en")
    w.weather("Bangna")
    w.weather("bangna")
    geo = [c for c in net.calls if c.url.startswith(PLACES_URL)]
    assert len(geo) == 1
    assert geo[0].headers["x-goog-fieldmask"] == "places.location,places.displayName,places.formattedAddress"  # fmt: skip
    assert geo[0].body["textQuery"] == "Bangna" and geo[0].body["regionCode"] == "TH"
    cur = [c for c in net.calls if c.url.startswith(CUR_URL)]
    assert len(cur) == 1  # the same spot within ten minutes: cached
    assert dict(cur[0].req.url.params)["location.latitude"] == "13.67"
    assert dict(cur[0].req.url.params)["languageCode"] == "en"


def test_tomorrow_uses_the_daily_forecast():
    day = {
        "displayDate": {"year": 2026, "month": 10, "day": 15},
        "daytimeForecast": {**CUR}, "nighttimeForecast": {"precipitation": {"probability": {"percent": 20}}},
        "maxTemperature": {"degrees": 33}, "minTemperature": {"degrees": 26},
    }  # fmt: skip
    other = {**day, "displayDate": {"year": 2026, "month": 10, "day": 14}}
    net = Net(**{DAY_URL: {"forecastDays": [other, day]}})
    r = weather_tools(net)[0].weather("", "พรุ่งนี้")
    assert (
        r["when"] == "2026-10-15" and r["rain_pct"] == 70 and r["temp_c"] == 33 and r["min_c"] == 26
    )
    assert "26-33°C" in r["summary"]
    assert weather_tools(net)[0].weather("", "2026-12-30")["error"] == "out_of_range"


def test_a_time_picks_the_nearest_hour_and_pages_through_results():
    pages = {
        None: {"forecastHours": [hour(h, 10) for h in range(10, 34)], "nextPageToken": "p2"},
        "p2": {"forecastHours": [hour(h, 80, day=15) for h in range(10, 14)]},
    }
    net = Net(**{HRS_URL: lambda req: pages[req.url.params.get("pageToken")]})
    r = weather_tools(net)[0].weather("", "พรุ่งนี้ 8 โมง")
    assert r["when"] == "2026-10-15 08:00" and r["rain_pct"] == 10
    r2 = weather_tools(net)[0].weather("", "2026-10-15T12:00")
    assert r2["rain_pct"] == 80 and len(net.calls) >= 3  # paged
    assert weather_tools(net)[0].weather("", "2026-11-20T12:00")["error"] == "out_of_range"


def test_hourly_detail_lists_the_next_hours():
    net = Net(**{HRS_URL: {"forecastHours": [hour(h, h) for h in range(9, 24)]}})
    r = weather_tools(net)[0].weather("", "now", "hourly")
    assert len(r["hours"]) == 12 and r["hours"][0]["time"].endswith("10:00")


def test_weather_default_place_is_the_office_and_stores_coordinates(repo):
    hq = repo.create(
        G, name="สำนักงานใหญ่", address="1 Silom Rd", kind="hq", created_by=PBOSS,
        sources=[pl.make_source(PUBLIC_CH, PBOSS, NOW)],
    )  # fmt: skip
    net = Net(**{CUR_URL: CUR, PLACES_URL: GEO})
    r = weather_tools(net, places_ctx(repo, PSTAFF, channel=PUBLIC_CH))[0].weather()
    assert r["place"] == "สำนักงานใหญ่" and "Silom" not in json.dumps(r, ensure_ascii=False)
    assert (repo.get(G, hq.id).lat, repo.get(G, hq.id).lng) == (13.67, 100.6)
    net2 = Net(**{CUR_URL: CUR})  # coordinates known now: no geocoding
    weather_tools(net2, places_ctx(repo, PSTAFF, channel=PUBLIC_CH))[0].weather("ออฟฟิศ")
    assert [c.url.split("?")[0] for c in net2.calls] == [CUR_URL]


def test_weather_never_says_a_hidden_address(repo):
    home, _ = home_and_branch(repo)
    net = Net(**{CUR_URL: CUR, PLACES_URL: GEO})
    r = weather_tools(net, places_ctx(repo, PSTAFF, channel=PUBLIC_CH))[0].weather("บ้านผม")
    assert r["place"] == "บ้านสมชาย" and "ซอยลับ" not in json.dumps(r, ensure_ascii=False)


def test_weather_api_off_is_reported_not_invented():
    net = Net(**{CUR_URL: httpx.Response(403, json={})})
    r = weather_tools(net)[0].weather()
    assert r["error"] == "maps_unavailable" and "do not guess" in r["detail"]
    net2 = Net(**{CUR_URL: CUR, PLACES_URL: {}})
    assert weather_tools(net2)[0].weather("Atlantis")["error"] == "place_not_found"


# ------------------------------------------------------------------ ensemble (BigQuery, fake rows)
def ens_rows():
    # 4 members over two 6 h steps; member totals (mm): 0.5, 1.5, 3.0, 0.0
    spec = {0: (0.0002, 0.0003), 1: (0.001, 0.0005), 2: (0.002, 0.001), 3: (0.0, 0.0)}
    return [{"member": m, "precip_m": v} for m, vs in spec.items() for v in vs]


def test_parse_ensemble_counts_members_above_the_threshold():
    res = parse_ensemble(ens_rows())
    assert res["members"] == 4 and res["rain_probability_pct"] == 50 and res["max_mm"] == 3.0
    assert parse_ensemble(ens_rows(), mm=0.4)["rain_probability_pct"] == 75
    with pytest.raises(MapsError):
        parse_ensemble([])


def test_ensemble_detail_uses_bigquery_and_is_disabled_by_default():
    seen = {}

    def run(sql, params):
        seen.update(sql=sql, params=params)
        return ens_rows()

    bq = WeatherNextBigQuerySource("proj.wn", "weathernext_2_0_0", run=run)
    net = Net()
    w, _ = weather_tools(net, bq=bq)
    r = w.weather("", "now", "ensemble")
    assert r["ensemble"]["rain_probability_pct"] == 50 and "ensemble" in r["source"].lower()
    assert "`proj.wn.weathernext_2_0_0`" in seen["sql"] and seen["params"]["lat"] == 13.7563
    assert net.calls == []  # no Google HTTP call at all
    off, _ = weather_tools(Net(), bq=WeatherNextBigQuerySource(""))
    assert off.weather("", "now", "ensemble")["error"] == "ensemble_unavailable"
    boom = WeatherNextBigQuerySource("p.d", run=lambda s, p: 1 / 0)
    assert weather_tools(Net(), bq=boom)[0].weather("", "now", "ensemble")["error"] == "ensemble_unavailable"  # fmt: skip


# ------------------------------------------------------------------ travel + weather
def test_travel_adds_a_rain_note_when_rain_is_likely_at_departure():
    net = Net(**{ROUTES_URL: route(), PLACES_URL: GEO, CUR_URL: CUR, HRS_URL: {"forecastHours": [hour(h, 90) for h in range(9, 24)]}})  # fmt: skip
    _, svc = weather_tools(net)
    r = travel(net, weather=svc).travel_time("A", "B")
    assert r["rain_note"] == "ฝนตก (70%) อาจใช้เวลานานขึ้น"
    dry = Net(**{ROUTES_URL: route(), PLACES_URL: GEO, CUR_URL: {**CUR, "precipitation": {"probability": {"percent": 10}}}})  # fmt: skip
    _, svc2 = weather_tools(dry)
    assert "rain_note" not in travel(dry, weather=svc2).travel_time("A", "B")
    # weather down: the travel answer still comes
    broken = Net(**{ROUTES_URL: route(), PLACES_URL: httpx.Response(403, json={})})
    _, svc3 = weather_tools(broken)
    assert travel(broken, weather=svc3).travel_time("A", "B")["status"] == "ok"


# ------------------------------------------------------------------ bot glue: flags and the brief
def tool_names(tools):
    return [f.__name__ for f in tools]


def fake_services(bot, net):
    http = net.http()
    bot._maps_svc = (RouteService(http), WeatherService(http, Geocoder(http)))


async def test_flag_off_means_no_tools_and_flag_on_registers_both(bot, world):
    msg = SimpleNamespace(channel=world.pub, guild=world.g, room=None, id=1, jump_url="")
    fake_services(bot, Net())
    args = (msg, world.g, world.staff, False, "th")
    assert bot._maps_tools(*args) == []  # SALID_MAPS_ENABLED false (the default)
    bot.s.maps_enabled = True
    assert tool_names(bot._maps_tools(*args)) == ["travel_time", "weather"]
    set_features(bot, maps=False)  # the guild flag
    assert bot._maps_tools(*args) == []
    set_features(bot, maps=True)
    assert len(bot._maps_tools(*args)) == 2


def test_settings_default_is_off_and_not_an_action_tool():
    from salid.config import Settings
    from salid.voice_actions import ACTION_TOOL_NAMES

    assert Settings(_env_file=None).maps_enabled is False
    assert Settings(_env_file=None).weathernext_bq_dataset == ""
    assert {"travel_time", "weather"}.isdisjoint(ACTION_TOOL_NAMES)


async def test_the_morning_brief_says_when_to_leave_and_the_office_weather(bot, world):
    bot.s.maps_enabled = True
    home = bot.places.create(
        GUILD, name="บ้านสมชาย", address="12 ซอยลับ บางนา", kind="home", owner_id=STAFF,
        created_by=STAFF, sources=[pl.make_source(-STAFF, STAFF, NOW)],
    )  # fmt: skip
    branch = bot.places.create(
        GUILD, name="สาขาบางนา", address=ADDR, kind="branch", created_by=BOSS,
        sources=[pl.make_source(PUBLIC, BOSS, NOW)],
    )  # fmt: skip
    bot.places.create(
        GUILD, name="สำนักงานใหญ่", address="1 Silom Rd", kind="hq", created_by=BOSS, lat=13.7, lng=100.5,
        sources=[pl.make_source(PUBLIC, BOSS, NOW)],
    )  # fmt: skip
    net = Net(**{ROUTES_URL: route(dur=2700, static=1800), CUR_URL: CUR,
                 "https://weather.googleapis.com/v1/forecast/days": {"forecastDays": [{
                     "displayDate": {"year": 2026, "month": 10, "day": 14},
                     "daytimeForecast": CUR, "maxTemperature": {"degrees": 33}}]}})  # fmt: skip
    fake_services(bot, net)
    meeting(bot, "m1", "ตรวจสาขา", at(14, 0), [STAFF, PEER], place_id=branch.id)
    meeting(bot, "m2", "Online", at(11, 0), [STAFF])  # no place: no line
    inp = bot._gather_brief(world.g, world.staff, "morning", at(8, 31))
    assert any("ออกจากบ้านสมชาย" in n and "13:15" in n and "45 นาที" in n for n in inp.notes), inp.notes  # fmt: skip
    assert any("อากาศที่สำนักงานใหญ่" in n for n in inp.notes)
    body = next(c for c in net.calls if c.url == ROUTES_URL).body
    assert body["origin"] == {"address": home.address}
    # Peer is invited but has no home/office they may use except the HQ: no leave line from a home
    inp2 = bot._gather_brief(world.g, world.peer, "morning", at(8, 31))
    assert not any("ออกจากบ้านสมชาย" in n for n in inp2.notes)
    # the notes are only for the private brief and only with the setting on
    bot.s.maps_enabled = False
    assert bot._gather_brief(world.g, world.staff, "morning", at(8, 31)).notes == []
