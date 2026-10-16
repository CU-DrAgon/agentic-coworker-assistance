# ruff: noqa: F811
"""Currency conversion and flights: rates parsing + fallback, Thai names, flight number and airport
normalisation, grounded prompts and links, provider selection, tracking (poll, notify once,
persistence), and the wiring (flag, voice parity, interjection)."""

from datetime import UTC, date, datetime

import pytest
from conftest import GUILD, FakeClient, scripted, text_resp
from test_coordinator import STAFF, world  # noqa: F401  (fixture)
from test_modes import bot  # noqa: F401  (fixture)

from salid import qa
from salid.coo_store import SQLiteDocStore
from salid.currency import CURRENCY_PROMPT, Converter, convert, currency_tools, resolve_currency
from salid.flights import (
    AeroDataBox,
    AviationStack,
    FlightService,
    FlightTracker,
    diff_events,
    flight_links,
    flight_tools,
    normalize_flight,
    prices_links,
    resolve_airport,
    select_provider,
)
from salid.guild_settings import FEATURES, Features, GuildSettingsUpdate
from salid.voice_actions import ACTION_TOOL_NAMES

TODAY = date(2026, 10, 16)


# ------------------------------------------------------------------ currency
class Net:
    """Fake fetch: url -> json or exception; records calls."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, url):
        self.calls.append(url)
        for prefix, val in self.routes.items():
            if url.startswith(prefix):
                if isinstance(val, Exception):
                    raise val
                return val
        raise ConnectionError(url)


ECB = {"amount": 1.0, "base": "USD", "date": "2026-10-15", "rates": {"THB": 33.54}}
ERAPI = {
    "result": "success",
    "time_last_update_unix": 1792108951,
    "rates": {"THB": 33.6, "SAR": 3.75},
}


def test_frankfurter_parsing_and_label():
    net = Net({"https://api.frankfurter.dev/v1/latest": ECB})
    r = convert(Converter(net), 100, "USD", "THB")
    assert r["converted"] == 3354.0 and r["rate"] == 33.54 and r["rate_date"] == "2026-10-15"
    assert r["source"] == "ECB" and "base=USD&symbols=THB" in net.calls[0]
    assert r["label"] == "≈ 3,354.00 THB (อัตรา ECB วันที่ 2026-10-15)"
    assert "อัตรากลาง" in r["disclaimer"]
    en = convert(Converter(net), 1, "USD", "THB", lang="en")
    assert en["label"].startswith("≈ 33.54 THB (ECB rate on")


def test_historical_date_uses_the_dated_endpoint():
    net = Net({"https://api.frankfurter.dev/v1/2026-10-01": {**ECB, "date": "2026-10-01"}})
    r = convert(Converter(net), 10, "usd", "thb", "2026-10-01")
    assert r["rate_date"] == "2026-10-01" and "/2026-10-01?" in net.calls[0]


def test_falls_back_to_open_er_api_when_the_ecb_fails_or_lacks_the_currency():
    net = Net(
        {"https://api.frankfurter.dev": TimeoutError("slow"), "https://open.er-api.com": ERAPI}
    )
    r = convert(Converter(net), 10, "SAR", "THB")
    assert r["source"] == "open.er-api.com" and r["converted"] == 336.0
    assert r["rate_date"] == "2026-10-16" and "open.er-api.com" in r["label"]
    old = convert(Converter(net), 10, "SAR", "THB", "2026-09-01")
    assert "no historical rate" in old["note"]


def test_both_sources_down_is_an_error_not_a_guess():
    r = convert(Converter(Net({})), 5, "USD", "THB")
    assert "error" in r and "do not guess" in r["error"]


def test_rates_are_cached_for_an_hour():
    now = [0.0]
    net = Net({"https://api.frankfurter.dev/v1/latest": ECB})
    conv = Converter(net, clock=lambda: now[0])
    convert(conv, 1, "USD", "THB")
    now[0] = 3599
    convert(conv, 2, "USD", "THB")
    assert len(net.calls) == 1
    now[0] = 3601
    convert(conv, 3, "USD", "THB")
    assert len(net.calls) == 2


@pytest.mark.parametrize(
    "text,code",
    [("ดอลลาร์", "USD"), ("$", "USD"), ("เยน", "JPY"), ("¥", "JPY"), ("ยูโร", "EUR"),
     ("หยวน", "CNY"), ("฿", "THB"), ("บาท", "THB"), ("jpy", "JPY"), (" Eur ", "EUR"),
     ("ดอลลาร์สิงคโปร์", "SGD"), ("ปอนด์", "GBP"), ("xyz1", None), ("", None)],
)  # fmt: skip
def test_thai_currency_names_and_symbols(text, code):
    assert resolve_currency(text) == code


def test_unknown_currency_and_bad_inputs():
    c = Converter(Net({}))
    assert "unknown currency" in convert(c, 1, "กอไก่", "THB")["error"]
    assert "number" in convert(c, "abc", "USD", "THB")["error"]
    assert "YYYY" in convert(c, 1, "USD", "THB", "yesterday")["error"]


def test_tool_is_named_and_grouped_and_ledger_totals_stay_primary_in_the_prompt():
    (tool,) = currency_tools(Converter(Net({"https://api.frankfurter.dev": ECB})))
    assert tool.__name__ == "convert_currency" and tool.salid_group == "travel"
    out = tool(131.99, "ดอลลาร์", "บาท")
    assert out["label"].startswith("≈ 4,426.94 THB")  # a THB-equivalent is always labelled
    assert "≈ ฿" in CURRENCY_PROMPT and "ตัวเลขหลัก" in CURRENCY_PROMPT
    assert "ห้ามเดา" in CURRENCY_PROMPT


# ------------------------------------------------------------------ flight numbers, airports, links
@pytest.mark.parametrize(
    "text,iata",
    [("TG 676", "TG676"), ("tg676", "TG676"), ("TG-676", "TG676"), ("การบินไทย 676", "TG676"),
     ("Thai Airways 676", "TG676"), ("ไทยแอร์เอเชีย 123", "FD123"), ("3K 512", "3K512"),
     ("TG 0676", "TG676"), ("แอร์เอเชีย 3100", "FD3100")],
)  # fmt: skip
def test_flight_number_normalisation(text, iata):
    assert normalize_flight(text)["iata"] == iata


def test_bad_flight_numbers():
    assert normalize_flight("676") is None and normalize_flight("") is None
    assert normalize_flight("สายการบินลึกลับ 12") is None


def test_flightaware_prefers_icao_and_google_link():
    links = flight_links(normalize_flight("TG 676"))
    assert links["flightaware"] == "https://www.flightaware.com/live/flight/THA676"
    assert links["google"] == "https://www.google.com/search?q=TG+676+flight+status"
    unknown = flight_links(normalize_flight("ZZ 12"))
    assert unknown["flightaware"].endswith("/ZZ12")


def test_airports_thai_english_and_codes():
    assert resolve_airport("กรุงเทพ") == "BKK" and resolve_airport("โตเกียว") == "TYO"
    assert resolve_airport("Chiang Mai") == "CNX" and resolve_airport("hnd") == "HND"
    assert resolve_airport("ซิดนีย์") == "SYD" and resolve_airport("เมืองลับ") is None


def test_price_links():
    one = prices_links("BKK", "NRT", "2026-12-20")
    assert one["google_flights"] == (
        "https://www.google.com/travel/flights?q=Flights%20to%20NRT%20from%20BKK%20on%202026-12-20"
    )
    rt = prices_links("BKK", "NRT", "2026-12-20", "2026-12-27", adults=2, cabin="business")
    assert rt["google_flights"].endswith("on%202026-12-20%20through%202026-12-27")
    assert rt["skyscanner"] == (
        "https://www.skyscanner.co.th/transport/flights/bkk/nrt/261220/261227/"
        "?adultsv2=2&cabinclass=business"
    )
    assert "skyscanner" not in prices_links("กรุงเทพ", "NRT", "2026-12-20")


# ------------------------------------------------------------------ grounded status / prices
def grounded(text="ข้อมูลเที่ยวบิน"):
    return FakeClient([text_resp(text)])


def svc(client=None, provider=None):
    return FlightService(client, "m", provider, today=lambda: TODAY)


def test_grounded_status_prompt_links_and_confidence():
    c = grounded()
    out = svc(c).status("tg676", "2026-10-17")
    assert out["confidence"] == "web" and out["text"] == "ข้อมูลเที่ยวบิน"
    assert out["links"]["flightaware"].endswith("/THA676") and "q=TG+676" in out["links"]["google"]
    assert out["checked_at"] and "ค้นเว็บ" in out["note"]
    model, contents, config = c.models.calls[0]
    prompt = contents[0].parts[0].text
    assert "TG676" in prompt and "2026-10-17" in prompt and "gate" in prompt
    assert "ไม่พบข้อมูล" in config.system_instruction and "Never guess" in config.system_instruction
    assert config.tools[0].google_search is not None and config.tools[0].url_context is None
    assert config.http_options.timeout == 15000
    assert "SALID" not in prompt  # nothing but the flight reaches Google


def test_status_is_cached_five_minutes_and_errors_still_carry_links():
    now = [0.0]
    c = FakeClient([text_resp("a"), text_resp("b")])
    s = FlightService(c, "m", None, today=lambda: TODAY, clock=lambda: now[0])
    s.status("TG 676")
    now[0] = 299
    assert s.status("TG 676")["text"] == "a"
    now[0] = 301
    assert s.status("TG 676")["text"] == "b"
    bad = FlightService(None, "m", None, today=lambda: TODAY).status("TG 676")
    assert "error" in bad and bad["links"]["flightaware"]
    assert "error" in svc().status("nonsense") and "error" in svc().status("TG 676", "tomorrow")


def test_prices_grounded_labelled_with_links_and_cache():
    c = grounded("Thai Airways 14,000-16,000 THB")
    s = svc(c)
    out = s.prices("กรุงเทพ", "โตเกียว", "2026-12-20", "2026-12-27", adults=2)
    assert out["origin"] == "BKK" and out["destination"] == "TYO" and out["confidence"] == "web"
    assert out["label"] == "ราคาโดยประมาณจากการค้นเว็บ ณ วันนี้ ราคาจริงเปลี่ยนตลอด"
    assert out["searched_on"] == "2026-10-16" and out["sources"] == []
    assert out["links"]["google_flights"].startswith(
        "https://www.google.com/travel/flights?q=Flights%20to%20TYO%20from%20BKK%20on%202026-12-20"
    )
    prompt = c.models.calls[0][1][0].parts[0].text
    assert "never estimate a fare" in prompt and "round trip returning 2026-12-27" in prompt
    assert s.prices("BKK", "TYO", "2026-12-20", "2026-12-27", adults=2) is out  # cached 30 min
    assert len(c.models.calls) == 1


def test_prices_input_checks():
    s = svc(grounded())
    assert "past" in s.prices("BKK", "NRT", "2026-10-01")["error"]
    assert "before" in s.prices("BKK", "NRT", "2026-12-20", "2026-12-01")["error"]
    assert "YYYY" in s.prices("BKK", "NRT", "20 Dec")["error"]
    out = s.prices(
        "เมืองลับ", "NRT", "2026-12-20"
    )  # unknown city: model still searches, no skyscanner
    assert "skyscanner" not in out["links"]


# ------------------------------------------------------------------ providers
def test_provider_selection_with_and_without_keys():
    assert select_provider("", "") is None and select_provider("  ", None) is None
    assert isinstance(select_provider("k1", "k2"), AeroDataBox)  # AeroDataBox wins
    assert isinstance(select_provider("", "k2"), AviationStack)


ADB = [{
    "airline": {"name": "Thai Airways"}, "status": "Expected",
    "departure": {"airport": {"iata": "BKK"}, "terminal": "1", "gate": "D5",
                  "scheduledTime": {"utc": "2026-10-17 03:00Z", "local": "2026-10-17 10:00+07:00"},
                  "revisedTime": {"utc": "2026-10-17 03:30Z", "local": "2026-10-17 10:30+07:00"}},
    "arrival": {"airport": {"iata": "NRT"},
                "scheduledTime": {"utc": "2026-10-17 11:00Z", "local": "2026-10-17 20:00+09:00"}},
}]  # fmt: skip
AVS = {"data": [{
    "flight_status": "active", "airline": {"name": "Thai Airways"},
    "departure": {"iata": "BKK", "gate": "D5", "terminal": "1", "delay": 20,
                  "timezone": "Asia/Bangkok", "scheduled": "2026-10-17T10:00:00+00:00",
                  "estimated": "2026-10-17T10:20:00+00:00"},
    "arrival": {"iata": "NRT", "scheduled": "2026-10-17T20:00:00+00:00"},
}]}  # fmt: skip


def test_aerodatabox_adapter_parses_and_sends_the_key_as_a_header():
    seen = {}

    def get(url, headers=None):
        seen.update(url=url, headers=headers)
        return ADB

    info = AeroDataBox("SECRET", get).status(normalize_flight("TG 676"), "2026-10-17")
    assert seen["url"].endswith("/flights/number/TG676/2026-10-17") and "SECRET" not in seen["url"]
    assert seen["headers"]["X-RapidAPI-Key"] == "SECRET"
    assert info["delay_min"] == 30 and info["status"] == "delayed" and info["dep_gate"] == "D5"
    assert info["origin"] == "BKK" and info["destination"] == "NRT" and info["dep_ts"]


def test_aviationstack_adapter_parses_local_times_in_the_airport_zone():
    info = AviationStack("K", lambda url, headers=None: AVS).status(
        normalize_flight("TG 676"), "2026-10-17"
    )
    assert info["status"] == "active" and info["delay_min"] == 20 and info["dep_terminal"] == "1"
    assert info["dep_ts"] == datetime(2026, 10, 17, 3, 0, tzinfo=UTC).timestamp()  # 10:00 +07:00
    assert (
        AviationStack("K", lambda u, headers=None: {"data": []}).status(
            normalize_flight("TG 1"), "2026-10-17"
        )
        is None
    )


class FakeProvider:
    name = "Fake"

    def __init__(self, infos):
        self.infos, self.calls = list(infos), 0

    def status(self, flight, day):
        self.calls += 1
        i = self.infos[min(self.calls - 1, len(self.infos) - 1)]
        if isinstance(i, Exception):
            raise i
        return i


def test_status_prefers_the_provider_then_falls_back_to_the_web():
    info = {"status": "scheduled", "dep_gate": "A1"}
    c = grounded()
    out = svc(c, FakeProvider([info])).status("TG 676")
    assert out["confidence"] == "provider" and out["source"] == "Fake" and out["dep_gate"] == "A1"
    assert out["links"]["flightaware"] and c.models.calls == []
    out = svc(c, FakeProvider([RuntimeError("boom")])).status("TG 676")
    assert out["confidence"] == "web" and len(c.models.calls) == 1
    out = svc(grounded(), FakeProvider([None])).status("TG 676")  # unknown to the provider
    assert out["confidence"] == "web"


# ------------------------------------------------------------------ tracking
T0 = datetime(2026, 10, 17, 3, 0, tzinfo=UTC).timestamp()  # scheduled departure


def info(**kw):
    return {"status": "scheduled", "delay_min": 0, "dep_gate": "A1", "dep_ts": T0,
            "dep_estimated": "2026-10-17 10:00+07:00", **kw}  # fmt: skip


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def tracker(infos, now=T0 - 10 * 3600):
    clock = Clock(now)
    docs = SQLiteDocStore(":memory:")
    return FlightTracker(docs, FakeProvider(infos), clock), clock, docs


FL = normalize_flight("TG 676")


def test_diff_events_notify_once():
    st = diff_events({}, info())[1]  # silent baseline
    ev, st = diff_events(st, info(delay_min=20, dep_estimated="2026-10-17 10:20+07:00"))
    assert [(e.kind, e.minutes, e.time) for e in ev] == [("delay", 20, "10:20")]
    assert diff_events(st, info(delay_min=20))[0] == []  # same delay: nothing
    assert diff_events(st, info(delay_min=30))[0] == []  # +10 min is below the step
    assert [e.kind for e in diff_events(st, info(delay_min=40))[0]] == ["delay"]
    assert diff_events({}, info(delay_min=10))[0] == []  # below 15
    ev, st = diff_events(st, info(dep_gate="B7"))
    assert [(e.kind, e.gate) for e in ev if e.kind == "gate"] == [("gate", "B7")]
    ev, st = diff_events(st, info(status="cancelled", dep_gate="B7"))
    assert [e.kind for e in ev] == ["cancelled"]
    assert diff_events(st, info(status="cancelled", dep_gate="B7"))[0] == []
    ev, st = diff_events({}, info(status="landed"))
    assert [e.kind for e in ev] == ["landed"] and diff_events(st, info(status="landed"))[0] == []


def test_tracking_starts_six_hours_before_polls_every_fifteen_minutes_and_persists():
    tr, clock, docs = tracker([info(), info(), info(delay_min=25), info(delay_min=25),
                               info(status="landed")])  # fmt: skip
    res = tr.add(GUILD, STAFF, FL, "2026-10-17")
    assert res["status"] == "tracking"
    doc = docs.get("flight_track", GUILD, f"{STAFF}_TG676_2026-10-17")
    assert doc["next_poll"] == T0 - 6 * 3600 and doc["user_id"] == STAFF  # persisted
    assert tr.due(GUILD) == []  # 10 h before: too early
    clock.t = T0 - 6 * 3600
    (d,) = tr.due(GUILD)
    assert tr.poll(d) == []  # nothing changed
    assert tr.due(GUILD) == []  # next poll in 15 min
    clock.t += 15 * 60
    (d,) = tr.due(GUILD)
    assert [e.kind for e in tr.poll(d)] == ["delay"]
    clock.t += 15 * 60
    (d,) = tr.due(GUILD)
    assert tr.poll(d) == []  # the same delay is not sent twice
    # a "restart": a fresh tracker over the same store keeps state
    tr2 = FlightTracker(docs, FakeProvider([info(delay_min=25), info(status="landed")]), clock)
    clock.t += 15 * 60
    (d,) = tr2.due(GUILD)
    assert tr2.poll(d) == []
    clock.t += 15 * 60
    (d,) = tr2.due(GUILD)
    assert [e.kind for e in tr2.poll(d)] == ["landed"]
    assert docs.find("flight_track", GUILD) == []  # finished: track removed


def test_tracking_limits_and_failures():
    tr, clock, docs = tracker([info()])
    assert tr.add(GUILD, STAFF, FL, "2026-10-17")["status"] == "tracking"
    assert tr.add(GUILD, STAFF, FL, "2026-10-17")["status"] == "tracking"  # same flight: idempotent
    for n in range(2, 6):
        tr.add(GUILD, STAFF, normalize_flight(f"TG {n}"), "2026-10-17")
    assert tr.add(GUILD, STAFF, normalize_flight("TG 9"), "2026-10-17")["error"] == "too_many"
    assert tr.add(GUILD, STAFF + 1, FL, "2026-10-17")["status"] == "tracking"  # per user
    assert tr.remove(GUILD, STAFF, FL, "2026-10-17") and not tr.remove(GUILD, STAFF, FL, "x")
    assert FlightTracker(docs, FakeProvider([None])).add(GUILD, 7, FL, "2026-10-17")["error"] == (
        "not_found"
    )
    assert FlightTracker(docs, None).add(GUILD, 7, FL, "2026-10-17")["error"] == "no_provider"
    stale, clock, docs = tracker([info()])
    stale.add(GUILD, STAFF, FL, "2026-10-17")
    clock.t = T0 + 25 * 3600
    (d,) = [x for x in docs.find("flight_track", GUILD)]
    assert stale.poll({**d, "next_poll": 0}) == [] and docs.find("flight_track", GUILD) == []


def test_track_flight_tool_without_a_key_gives_the_link_not_a_promise():
    tr = FlightTracker(SQLiteDocStore(":memory:"), None)
    tools = {f.__name__: f for f in flight_tools(svc(), tr, GUILD, STAFF, today=lambda: TODAY)}
    out = tools["track_flight"]("TG 676", "2026-10-17")
    assert out["status"] == "unavailable" and "key" in out["reason"]
    assert out["links"]["flightaware"].endswith("/THA676")
    assert tools["track_flight"]("??", "2026-10-17")["error"]


def test_track_flight_tool_with_a_key_tracks_and_stops():
    tr, _, docs = tracker([info()])
    tools = {f.__name__: f for f in flight_tools(svc(), tr, GUILD, STAFF, today=lambda: TODAY)}
    out = tools["track_flight"]("TG 676", "2026-10-17")
    assert out["status"] == "tracking" and out["polls_every_minutes"] == 15
    assert len(docs.find("flight_track", GUILD)) == 1
    assert tools["track_flight"]("TG 676", "2026-10-17", notify=False)["status"] == "stopped"
    assert docs.find("flight_track", GUILD) == []


# ------------------------------------------------------------------ wiring
def test_flag_and_action_tool_registration():
    assert "travel" in FEATURES and Features().travel is True
    assert "track_flight" in ACTION_TOOL_NAMES
    assert "convert_currency" not in ACTION_TOOL_NAMES  # a read: stays in the room agent


def kw_for(bot, world, **extra):
    member = world.g.members[STAFF]
    msg = bot.room_message(world.g, member, world.pub)
    return bot._agent_kwargs(
        msg, world.g, member, False, setting="th", lang="th", question="q", memory=None,
        profile=None, **extra,
    )  # fmt: skip


def names(kw):
    return {f.__name__ for f in kw.get("extra_tools", [])}


async def test_agent_kwargs_register_travel_tools_and_flag_off_removes_them(bot, world):
    assert {"convert_currency", "flight_status", "flight_prices", "track_flight"} <= names(
        kw_for(bot, world)
    )
    assert not names(kw_for(bot, world, interjection=True)) & {"convert_currency", "flight_status"}
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={"travel": False}), "x")
    bot.team._cache.clear()
    assert not names(kw_for(bot, world)) & {
        "convert_currency", "flight_status", "flight_prices", "track_flight",
    }  # fmt: skip


async def test_prompt_follows_the_tools_and_persona_still_forbids_guessing(bot, world):
    settings = bot.s
    off = qa.system_prompt(settings, scoped=False)
    on = qa.system_prompt(settings, scoped=False, travel=True)
    assert "ห้ามประมาณหรือเดาอัตราแลกเปลี่ยน" in off and "convert_currency" in off
    assert "flight_status" not in off and "flight_status" in on and "ราคาโดยประมาณ" in on
    seen = {}
    llm = scripted(lambda req: seen.update(p=req.config.system_instruction) or text_resp("ok"))
    ans = await qa.ask_async(
        bot.s, bot.store, "ขอสถานะ", guild_id=GUILD, is_exec=True, user_id=1, model=llm,
        extra_tools=kw_for(bot, world)["extra_tools"],
    )  # fmt: skip
    assert ans.text and "flight_status" in seen["p"] and "รวมเป็นเงินบาท" in seen["p"]


async def test_travel_due_dms_only_the_requester_once(bot, world, monkeypatch):
    clock = Clock(T0 - 6 * 3600)
    prov = FakeProvider([info(), info(delay_min=40, dep_estimated="2026-10-17 10:40+07:00")])
    bot.flight_tracker = FlightTracker(bot.docs, prov, clock)
    bot.flight_tracker.add(GUILD, STAFF, FL, "2026-10-17")
    monkeypatch.setattr(type(bot), "guilds", property(lambda self: [world.g]))
    world.g.get_member = lambda uid: world.g.members.get(uid)
    await bot._travel_due()
    clock.t += 15 * 60
    bot._travel_last = -1e9
    await bot._travel_due()
    bot._travel_last = -1e9
    clock.t += 15 * 60
    await bot._travel_due()
    staff = world.g.members[STAFF]
    assert len(staff.dms) == 1 and "TG 676" in staff.dms[0][0] and "40" in staff.dms[0][0]
    assert all(m.dms == [] for uid, m in world.g.members.items() if uid != STAFF)
