# ruff: noqa: F811, F401
"""Stock / gold / news tools, watchlist alerts, market hours, the opt-in morning line, company news
and the bot glue. Every HTTP call is a fake: nothing here touches the network."""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import GUILD
from test_briefings import BKK, set_features
from test_coordinator import STAFF, at, person, world
from test_modes import bot  # noqa: F401  (fixture)

from salid import markets as mk
from salid.coo_store import SQLiteDocStore
from salid.i18n import t
from salid.qa import system_prompt
from salid.voice_actions import ACTION_TOOL_NAMES

FIX = Path(__file__).parent.parent / "fixtures"


def fx(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


class Net:
    """Routes a URL (by substring) to canned text; an Exception value is raised."""

    def __init__(self, routes: dict):
        self.routes, self.urls = routes, []

    def __call__(self, url: str) -> str:
        self.urls.append(url)
        for key, val in self.routes.items():
            if key in url:
                if isinstance(val, Exception):
                    raise val
                return val
        raise mk.MarketError("http 404")


def routes(**extra) -> dict:
    base = {
        "PTT.BK": fx("yahoo_ptt_bk.json"),
        "%5ESET.BK": fx("yahoo_set.json"),
        "GC%3DF": fx("yahoo_gold.json"),
        "THB%3DX": fx("yahoo_thb.json"),
        "UpdatePriceList": fx("goldtraders_updates.html"),
        "goldtraders": fx("goldtraders_classic.html"),
        "gold-api.com": fx("gold_api_xau.json"),
        "frankfurter": fx("frankfurter_usd_thb.json"),
        "news.google.com": fx("google_news_ptt.xml"),
        "AAPL": fx("yahoo_ptt_bk.json").replace("PTT.BK", "AAPL").replace('"THB"', '"USD"'),
    }
    base.update(extra)
    return base


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def data(clock):
    return mk.MarketData(get=Net(routes()), clock=clock)


# ------------------------------------------------------------------ symbols
@pytest.mark.parametrize(
    "text, market, expected",
    [
        ("PTT", "auto", ["PTT.BK", "PTT"]),
        ("ปตท", "auto", ["PTT.BK"]),
        ("ปตท.", "auto", ["PTT.BK"]),
        ("กสิกร", "auto", ["KBANK.BK"]),
        ("SET index", "auto", ["^SET.BK"]),
        ("SET50", "auto", ["^SET50.BK"]),
        ("S&P 500", "auto", ["^GSPC"]),
        ("Nasdaq", "auto", ["^IXIC"]),
        ("Nikkei", "auto", ["^N225"]),
        ("NVDA", "auto", ["NVDA", "NVDA.BK"]),
        ("aapl", "global", ["AAPL"]),
        ("ADVANC", "th", ["ADVANC.BK"]),
        ("PTT.BK", "auto", ["PTT.BK"]),
        ("GC=F", "auto", ["GC=F"]),
        ("ค่าเงินบาท", "auto", ["THB=X"]),
        ("หุ้นอะไรก็ไม่รู้ที่ไม่มี", "auto", []),
        ("", "auto", []),
    ],
)
def test_symbol_candidates(text, market, expected):
    assert mk.candidates(text, market) == expected


def test_resolve_names_splits_lists():
    assert mk.resolve_names("PTT, AAPL และ ปตท") == ["PTT", "AAPL", "ปตท"]
    assert mk.resolve_names(["a", " "]) == ["a"]


# ------------------------------------------------------------------ Yahoo parsing
def test_quote_parses_price_previous_close_change_and_trend(data):
    q = data.quote("ปตท")
    assert q["status"] == "ok" and q["symbol"] == "PTT.BK" and q["currency"] == "THB"
    assert q["price"] == 30.5 and q["previous_close"] == 30.75  # yesterday, not the range start
    assert q["change"] == -0.25 and q["change_pct"] == -0.81
    assert [p["date"] for p in q["trend_5d"]][-1] == "2026-10-16" and len(q["trend_5d"]) == 5
    assert q["trend_5d_pct"] == round((30.5 - 31.0) / 31.0 * 100, 2)
    assert q["market_time_utc"].startswith("2026-10-16T03:30")
    assert q["source"] == "Yahoo Finance" and "15-20" in q["note"]


def test_thai_symbol_goes_to_yahoo_with_bk_suffix_and_a_user_agent_free_url(clock):
    net = Net(routes())
    d = mk.MarketData(get=net, clock=clock)
    d.quote("PTT")
    d.quote("SET index")
    assert any("/chart/PTT.BK?range=5d&interval=1d" in u for u in net.urls)
    assert any("/chart/%5ESET.BK?range=5d&interval=1d" in u for u in net.urls)


def test_unknown_ticker_falls_back_to_the_thai_listing(clock):
    ok = fx("yahoo_ptt_bk.json").replace("PTT.BK", "XYZ.BK")
    net = Net({"/XYZ.BK": ok, "/XYZ?": mk.MarketError("http 404")})
    q = mk.MarketData(get=net, clock=clock).quote("XYZ")
    assert q["status"] == "ok" and q["symbol"] == "XYZ.BK"


def test_quote_unavailable_when_endpoint_fails_and_never_invents_numbers(clock):
    d = mk.MarketData(get=Net({"yahoo": mk.MarketError("http 429")}), clock=clock)
    q = d.quote("PTT")
    assert q["status"] == "unavailable" and "price" not in q and "read_web" in q["hint"]
    assert d.quote("???ไม่รู้จัก")["status"] == "unknown_symbol"
    bad = mk.MarketData(get=Net({"yahoo": "not json"}), clock=clock).quote("AAPL")
    assert bad["status"] == "unavailable"
    nf = mk.MarketData(get=Net({"yahoo": fx("yahoo_notfound.json")}), clock=clock).quote("AAPL")
    assert nf["status"] == "unavailable"


def test_second_host_is_tried_when_the_first_fails(clock):
    calls = []

    def get(url):
        calls.append(url)
        if "query1" in url:
            raise mk.MarketError("http 503")
        return fx("yahoo_ptt_bk.json")

    q = mk.MarketData(get=get, clock=clock).quote("PTT.BK")
    assert q["status"] == "ok" and "query2" in calls[-1]


def test_quote_cache_is_60_seconds(clock):
    net = Net(routes())
    d = mk.MarketData(get=net, clock=clock)
    d.quote("PTT.BK")
    d.quote("PTT.BK")
    assert len(net.urls) == 1
    clock.t += 61
    d.quote("PTT.BK")
    assert len(net.urls) == 2


def test_failures_are_cached_briefly(clock):
    net = Net({"yahoo": mk.MarketError("http 503")})
    d = mk.MarketData(get=net, clock=clock)
    d.quote("PTT.BK")
    n = len(net.urls)
    d.quote("PTT.BK")
    assert len(net.urls) == n  # not hammered
    clock.t += 31
    d.quote("PTT.BK")
    assert len(net.urls) > n


def test_history_summary(data):
    h = data.history("PTT.BK", "6mo")
    assert h["status"] == "ok" and h["range"] == "6mo"
    assert (h["start"], h["end"], h["change_pct"]) == (31.0, 30.5, -1.61)
    assert h["high"] == 32.0 and h["high_date"] == "2026-10-13"
    assert h["low"] == 30.0 and h["low_date"] == "2026-10-16"
    assert "chart" in h and data.history("PTT.BK", "10y")["range"] == "1mo"  # bad range -> 1mo


# ------------------------------------------------------------------ gold
def test_gold_association_html_is_parsed():
    g = mk.parse_gold_page(fx("goldtraders_classic.html"))
    assert g["bar_sell"] == 66000.0 and g["bar_buy"] == 65800.0
    assert g["ornament_sell"] == 66800.0 and g["ornament_tax_base"] == 64490.64
    assert g["announced_at"] == "2026-10-16T09:03+07:00" and g["round"] == 1  # Buddhist year
    assert "สมาคมค้าทองคำ" in g["source"]


def test_gold_parse_fails_cleanly_on_other_html():
    with pytest.raises(mk.MarketError):
        mk.parse_gold_page("<html><body>Loading latest prices...</body></html>")
    with pytest.raises(mk.MarketError):
        mk.parse_gold_page("")


def test_gold_updates_page_has_spot_baht_rate_and_change():
    u = mk.parse_gold_updates(fx("goldtraders_updates.html"))
    assert u["announced_at"] == "2026-10-16T09:03+07:00" and u["round"] == 1
    assert (u["bar_buy"], u["bar_sell"]) == (65800.0, 66000.0)
    assert (u["ornament_buy"], u["ornament_sell"]) == (64490.64, 66800.0)
    assert u["spot_usd_oz"] == 4155.5 and u["usd_thb"] == 33.55 and u["change_baht"] == 50.0
    with pytest.raises(mk.MarketError):
        mk.parse_gold_updates("<html>nothing</html>")
    down = fx("goldtraders_updates.html").replace(">50<", ">-100<").replace(" 50 ", " -100 ")
    assert mk.parse_gold_updates(down)["change_baht"] == -100.0


def test_gold_api_parsing():
    g = mk.parse_gold_api(fx("gold_api_xau.json"))
    assert g["price"] == 4163.100098 and g["updated_at"] == "2026-10-16T02:29:17+00:00"
    for bad in ('{"price": 0, "symbol": "XAU"}', "{}", "nope", '{"price": 5, "symbol": "XAG"}'):
        with pytest.raises(mk.MarketError):
            mk.parse_gold_api(bad)


def test_median_spread_and_disagreement():
    one = mk.combine_spots([{"value": 4000.0}])
    assert one == {"spot_usd_oz": 4000.0, "spread_pct": 0.0, "disagree": False}
    two = mk.combine_spots([{"value": 4155.5}, {"value": 4163.1}])
    assert two["spot_usd_oz"] == 4159.3 and two["spread_pct"] == 0.18 and not two["disagree"]
    far = mk.combine_spots([{"value": 4000.0}, {"value": 4100.0}])
    assert far["disagree"] and far["spread_pct"] == 2.47


def test_conversion_math_worked_example():
    # 4,159.30 USD/oz x 33.55 THB/USD = 139,540.5 THB/oz; / 31.1035 g = 4,486.2 THB/g;
    # x 15.244 g = 68,388 THB per baht-weight of pure gold; x 0.965 = 65,998 (96.5%)
    assert round(mk.thb_per_baht_weight(4159.30, 33.55)) == 65998


def test_gold_price_has_separate_thai_and_world_sections(clock):
    g = mk.MarketData(get=Net(routes()), clock=clock).gold()
    th, w = g["thai"], g["world"]
    assert th["status"] == "ok" and "15.244" in th["unit"] and th["purity"] == "96.5%"
    assert (th["bar_buy"], th["bar_sell"]) == (65800.0, 66000.0)
    assert (th["ornament_buy"], th["ornament_sell"]) == (64490.64, 66800.0)
    assert th["announced_at"] == "2026-10-16T09:03+07:00" and th["round"] == 1
    assert th["change"]["baht"] == 50.0 and th["change"]["direction"] == "ขึ้น"
    names = [x["name"] for x in w["spot_sources"]]
    assert names == ["Spot (อ้างอิงจากสมาคมค้าทองคำ)", "Spot gold-api.com (XAU/USD)"]
    assert [x["value"] for x in w["spot_sources"]] == [4155.5, 4163.100098]
    assert w["spot_usd_oz"] == 4159.3 and w["spot_basis"] == "exact" and not w["disagree"]
    assert (
        w["futures_usd_oz"] == 4170.0 and "Futures" in w["futures_label"]
    )  # never in the spot list
    assert 4170.0 not in [x["value"] for x in w["spot_sources"]]
    assert "conversion" not in g


def test_gold_summary_states_units_sources_and_times(clock):
    g = mk.MarketData(get=Net(routes()), clock=clock).gold()
    th, en = g["summary"], g["summary_en"]
    for needle in ("บาทต่อน้ำหนัก 1 บาททองคำ", "15.244 กรัม", "96.5%", "16/10/2026 09:03", "ครั้งที่ 1",
                   "65,800", "66,000", "ฐานภาษี", "ขึ้น 50 บาท", "USD per troy ounce", "ฟิวเจอร์ส (COMEX GC=F",
                   "ไม่ใช่สปอต", "4,159.30"):  # fmt: skip
        assert needle in th, needle
    assert th.index("ราคาทองไทย") < th.index("ราคาทองสากล")
    for needle in ("THB per 1 baht-weight = 15.244 g", "Bangkok time", "USD per troy ounce",
                   "median of 2 sources", "Futures (COMEX GC=F, not spot)"):  # fmt: skip
        assert needle in en, needle


def test_spot_sources_disagreeing_over_one_percent_are_called_out(clock):
    far = json.dumps(
        {"price": 4300.0, "symbol": "XAU", "currency": "USD", "updatedAt": "2026-10-16T02:29:17Z"}
    )
    g = mk.MarketData(get=Net(routes(**{"gold-api.com": far})), clock=clock).gold()
    assert g["world"]["disagree"] and "ต่างกันเกิน 1%" in g["summary"]
    assert "4,300.00" in g["summary"] and "4,155.50" in g["summary"]  # each source shown


def test_conversion_only_when_asked_with_fx_source(clock):
    d = mk.MarketData(get=Net(routes()), clock=clock)
    c = d.gold(compare=True)["conversion"]
    assert c["status"] == "ok" and c["thb_per_baht_weight_96_5"] == 65998
    assert c["usd_thb"] == 33.55 and "สมาคม" in c["usd_thb_source"] and "×" in c["formula"]
    assert "พรีเมียม" in c["note"] and "premiums" in c["note"]
    assert (
        "65,998" in d.gold(compare=True)["summary"] and "พรีเมียม" in d.gold(compare=True)["summary"]
    )
    # without the association's baht rate, Frankfurter (ECB) is the FX source
    only_api = routes(**{"UpdatePriceList": mk.MarketError("http 500")})
    c2 = mk.MarketData(get=Net(only_api), clock=Clock()).gold(compare=True)["conversion"]
    assert c2["usd_thb"] == 33.54 and "Frankfurter" in c2["usd_thb_source"]


def test_gold_degrades_per_part(clock):
    half = mk.MarketData(
        get=Net(
            routes(goldtraders=mk.MarketError("http 500"), UpdatePriceList=mk.MarketError("x"))
        ),
        clock=clock,
    ).gold()
    assert half["thai"]["status"] == "unavailable" and "ดึงข้อมูลไม่ได้" in half["summary"]
    assert half["world"]["spot_sources"][0]["name"].startswith("Spot gold-api")
    # the updates page alone still gives the Thai prices
    only_upd = mk.MarketData(get=Net(routes(goldtraders=mk.MarketError("x"))), clock=Clock()).gold()
    assert only_upd["thai"]["bar_sell"] == 66000.0
    nothing = mk.MarketData(
        get=Net({"x": "y", "yahoo": mk.MarketError("http 500")}), clock=Clock()
    ).gold()
    assert nothing["status"] == "unavailable" and nothing["world"]["spot_usd_oz"] is None


def wgq(symbol, price, as_of="2026-10-16 10:30 ICT", url="https://a.example/x", title="A", **kw):
    return gq(symbol, price, as_of=as_of, source_url=url, source_title=title, currency="USD", **kw)


def test_three_tiers_for_a_stock_quote(clock):
    # tier 1: direct Yahoo feed
    assert mk.MarketData(get=Net(routes()), clock=clock).quote("PTT.BK")["status"] == "ok"
    # tier 2: web sources with value, name, link and time that agree within 0.5%
    agree = [
        gq("PTT.BK", 30.5, source_title="SET", source_url="https://set.or.th/x"),
        gq("PTT.BK", 30.55, source_title="Settrade", source_url="https://settrade.com/y"),
    ]
    d, _ = blocked_data(clock, GClient(json.dumps(agree)))
    q = d.quote("PTT.BK")
    assert q["status"] == "web_reference" and q["price"] == 30.525 and q["n_sources"] == 2
    assert q["confidence"] == "medium" and {x["url"] for x in q["sources"]} == {
        "https://set.or.th/x", "https://settrade.com/y"}  # fmt: skip
    assert "Reference price 30.52 THB (median) from 2 web source(s)" in q["estimate_text"]
    assert "https://set.or.th/x" in q["estimate_text"] and "2026-10-16 10:30" in q["estimate_text"]
    # tier 3: sources differ (> 0.5%) or lack a time -> range, low confidence
    differ = [
        gq("PTT.BK", 30.0),
        gq("PTT.BK", 31.0, source_url="https://b.example/z", source_title="B"),
    ]
    q = blocked_data(Clock(), GClient(json.dumps(differ)))[0].quote("PTT.BK")
    assert q["status"] == "estimate" and q["range"] == {"low": 30.0, "high": 31.0}
    assert q["price"] == 30.5 and q["confidence"] == "low"
    assert "Estimate ~30.50 THB" in q["estimate_text"] and "range 30.00-31.00" in q["estimate_text"]
    no_time = blocked_data(Clock(), GClient(json.dumps([gq("PTT.BK", 30.5, as_of="")])))[0].quote(
        "PTT.BK"
    )
    assert no_time["status"] == "estimate" and no_time["sources"][0]["as_of"] == "ไม่ระบุเวลา"
    assert (
        no_time["sources"][0]["missing"] == ["เวลา"]
        and "[missing: เวลา]" in no_time["estimate_text"]
    )
    # tier 4: nothing usable
    for bad in ("[]", "no idea", json.dumps([gq("PTT.BK", price="n/a")])):
        assert blocked_data(Clock(), GClient(bad))[0].quote("PTT.BK")["status"] == "unavailable"


def test_gold_tiers_when_exact_sources_are_missing(clock):
    # no gold-api, no association spot: world spot from web pages (two agreeing sources)
    web = [wgq("XAU", 4160.0), wgq("XAU", 4162.0, url="https://b.example/g", title="B")]
    thai = [gq("TH_GOLD_BAR", 66000.0, currency="THB"), gq("TH_GOLD_BAR", 66500.0, currency="THB", source_url="https://c.example/t", source_title="C")]  # fmt: skip
    client = GClient(json.dumps(thai), json.dumps(web))  # Thai is asked first
    net = Net({"yahoo": mk.MarketError("http 429"), "gold-api.com": mk.MarketError("x"),
               "goldtraders": mk.MarketError("x")})  # fmt: skip
    d = mk.MarketData(get=net, clock=clock, grounded=mk.Grounded(client, "m", clock))
    g = d.gold()
    w, th = g["world"], g["thai"]
    assert w["spot_basis"] == "web_reference" and w["spot_usd_oz"] == 4161.0
    assert "จากการค้นเว็บ ความแม่นยำต่ำกว่า" in w["spot_sources"][0]["name"]
    assert w["futures_usd_oz"] is None  # futures are never filled from web search
    assert (
        th["status"] == "estimate" and th["bar_sell"] == 66250.0 and th["range"]["high"] == 66500.0
    )
    assert "ราคาโดยประมาณ" in g["summary"] and "ราคาอ้างอิงจาก 2 แหล่ง" in g["summary"]
    assert "https://c.example/t" in g["summary"]


# ------------------------------------------------------------------ news
def test_news_parse_dedupes_near_identical_titles_and_keeps_fields():
    items = mk.parse_news(fx("google_news_ptt.xml"))
    titles = [i["title"] for i in items]
    assert len(items) == 3  # the "!" duplicate from another outlet is merged
    assert titles[0].startswith("ราคาน้ำมันพรุ่งนี้") and "thansettakij" not in titles[0]
    assert items[0]["source"] == "thansettakij" and items[0]["time_utc"] == "2026-10-15T10:00+00:00"
    assert items[2]["time_utc"] == "" and items[2]["link"].startswith("https://news.google.com/")
    assert len(mk.parse_news(fx("google_news_ptt.xml"), limit=1)) == 1


def test_news_urls_thai_english_and_topics():
    th = mk.news_url("ปตท", "", "th")
    assert th.startswith("https://news.google.com/rss/search?q=%E0%B8%9B%E0%B8%95%E0%B8%97")
    assert th.endswith("hl=th&gl=TH&ceid=TH:th")
    assert mk.news_url("Apple", "", "en").endswith("hl=en-US&gl=US&ceid=US:en")
    assert "/topic/BUSINESS?hl=th" in mk.news_url("", "ธุรกิจ", "th")
    assert mk.news_url("", "", "en").startswith("https://news.google.com/rss?hl=en-US")


def test_news_tool_result_cache_and_failure(clock):
    net = Net(routes())
    d = mk.MarketData(get=net, clock=clock)
    r = d.news("ปตท", limit=2)
    assert r["status"] == "ok" and len(r["items"]) == 2 and r["source"] == "Google News"
    d.news("ปตท", limit=5)
    assert len(net.urls) == 1  # 10 minute cache, shared across limits
    clock.t += 601
    d.news("ปตท")
    assert len(net.urls) == 2
    bad = mk.MarketData(get=Net({"news": mk.MarketError("http 503")}), clock=clock).news("x")
    assert bad["status"] == "unavailable"


# ------------------------------------------------------------------ market hours
def test_market_hours_set_us_gold_and_weekends():
    fri = lambda h, m=0: datetime(2026, 10, 16, h, m, tzinfo=BKK)  # noqa: E731
    sat = lambda h, m=0: datetime(2026, 10, 17, h, m, tzinfo=BKK)  # noqa: E731
    assert mk.is_open("set", fri(10, 0)) and mk.is_open("set", fri(16, 30))
    assert not mk.is_open("set", fri(9, 59)) and not mk.is_open("set", fri(16, 31))
    assert not mk.is_open("set", sat(11))
    # US cash session 09:30-16:00 New York = 20:30-03:00 Bangkok (EDT in October)
    assert mk.is_open("us", fri(21, 0)) and not mk.is_open("us", fri(15, 0))
    assert mk.is_open("us", sat(2, 0))  # still Friday afternoon in New York
    assert not mk.is_open("us", sat(21, 0))
    assert (
        mk.is_open("gold", fri(9, 0))
        and mk.is_open("gold", sat(12))
        and not mk.is_open("gold", fri(18))
    )
    assert mk.is_open("global", fri(3)) and not mk.is_open("global", sat(12))


def test_market_kinds():
    assert [mk.market_kind(s) for s in ("PTT.BK", "^SET.BK", "gold", "AAPL", "^GSPC", "GC=F", "^N225")] == [
        "set", "set", "gold", "us", "us", "global", "global",
    ]  # fmt: skip


# ------------------------------------------------------------------ watchlist + alerts
@pytest.fixture
def docs():
    return SQLiteDocStore(":memory:")


def ctx_for(docs, data, uid=STAFF, is_exec=False, gid=GUILD, prefs=None):
    return mk.MarketsCtx(
        guild_id=gid, asker_id=uid, is_exec=is_exec, data=data, repo=mk.WatchRepo(docs), docs=docs,
        prefs=prefs, now=lambda: datetime(2026, 10, 16, 3, 0, tzinfo=UTC),
    )  # fmt: skip


def tool(ctx, name):
    return next(f for f in mk.markets_tools(ctx) if f.__name__ == name)


def test_market_watch_add_list_remove(docs, data):
    c = ctx_for(docs, data)
    watch = tool(c, "market_watch")
    r = watch("add", "ปตท", "below", 30)
    assert r["status"] == "ok" and r["condition"] == "below" and r["value"] == 30
    assert watch("add", "ทอง", "up_pct", 1)["status"] == "ok"
    assert watch("add", "PTT", "ต่ำกว่า", 30)["status"] == "refused"  # duplicate
    lst = watch("list")["watches"]
    assert {w["condition"] for w in lst} == {"below", "up_pct"}
    assert watch("remove", "ปตท")["removed"] == 1 and len(watch("list")["watches"]) == 1
    assert watch("add", "PTT", "sideways", 3)["status"] == "error"
    assert watch("add", "PTT", "above", -1)["status"] == "error"
    assert watch("add", "", "above", 1)["status"] == "error"
    assert watch("add", "ที่ไม่รู้จักเลย", "above", 1)["status"] == "unknown_symbol"
    other = tool(ctx_for(docs, data, uid=99), "market_watch")
    assert other("list")["watches"] == []  # per user


def test_caps_per_user_and_per_guild(docs, data):
    repo = mk.WatchRepo(docs)
    for i in range(mk.MAX_WATCHES_PER_USER):
        assert repo.add(GUILD, mk.Watch(f"a{i}", 1, "AAPL", "AAPL", "above", 100 + i)) == ""
    assert repo.add(GUILD, mk.Watch("zz", 1, "AAPL", "AAPL", "above", 5)) == "too_many_for_user"
    # distinct symbols per guild
    for i in range(mk.MAX_WATCH_SYMBOLS_PER_GUILD - 1):  # AAPL already counts as one
        assert repo.add(GUILD, mk.Watch(f"s{i}", 100 + i, f"S{i}", "x", "above", 1)) == ""
    assert (
        repo.add(GUILD, mk.Watch("over", 7, "NEWONE", "x", "above", 1))
        == "too_many_symbols_in_server"
    )
    assert repo.add(GUILD, mk.Watch("ok", 7, "AAPL", "x", "below", 1)) == ""  # known symbol is fine


def test_trigger_conditions():
    assert mk.triggered("above", 30, 30.0, None) and not mk.triggered("above", 30, 29.9, None)
    assert mk.triggered("below", 30, 29.9, None) and not mk.triggered("below", 30, 31, None)
    assert mk.triggered("up_pct", 1, None, 1.2) and not mk.triggered("up_pct", 1, None, -2)
    assert mk.triggered("down_pct", 1, None, -1.5) and not mk.triggered("down_pct", 1, None, 0.9)
    assert mk.triggered("move_pct", 1, None, -1.5) and mk.triggered("move_pct", 1, None, 1.5)
    assert not mk.triggered("move_pct", 1, None, None)


def make_poller(docs, data, clock):
    repo = mk.WatchRepo(docs)
    return repo, mk.Poller(repo, data, docs, clock=clock)


FRI_10AM = datetime(2026, 10, 16, 10, 30, tzinfo=BKK)


def test_alert_fires_once_per_day_and_only_in_market_hours(docs, data, clock):
    repo, poller = make_poller(docs, data, clock)
    repo.add(GUILD, mk.Watch("w1", STAFF, "PTT.BK", "PTT", "below", 31))  # price 30.5 -> true
    repo.add(GUILD, mk.Watch("w2", STAFF, "PTT.BK", "PTT", "above", 99))  # never
    assert poller.due([GUILD], datetime(2026, 10, 16, 9, 0, tzinfo=BKK)) == []  # before 10:00
    assert poller.due([GUILD], datetime(2026, 10, 17, 11, 0, tzinfo=BKK)) == []  # Saturday
    fired = poller.due([GUILD], FRI_10AM)
    assert [(a.watch.id, a.user_id, a.price) for a in fired] == [("w1", STAFF, 30.5)]
    clock.t += 700  # next poll window, same day
    assert poller.due([GUILD], FRI_10AM.replace(hour=11)) == []  # fired once per day
    # the next day it can fire again
    clock.t += 90000
    nxt = datetime(2026, 10, 19, 10, 30, tzinfo=BKK)  # Monday
    assert [a.watch.id for a in poller.due([GUILD], nxt)] == ["w1"]


def test_polling_is_batched_per_symbol_and_throttled(docs, clock):
    net = Net(routes())
    d = mk.MarketData(get=net, clock=clock)
    repo, poller = make_poller(docs, d, clock)
    for uid in range(1, 6):  # five people watch PTT, in two guilds
        repo.add(GUILD, mk.Watch(f"a{uid}", uid, "PTT.BK", "PTT", "above", 99))
        repo.add(GUILD + 1, mk.Watch(f"b{uid}", uid, "PTT.BK", "PTT", "above", 99))
    poller.due([GUILD, GUILD + 1], FRI_10AM)
    assert len([u for u in net.urls if "PTT.BK" in u]) == 1
    clock.t += 120  # < 10 minutes: not polled again
    poller.due([GUILD, GUILD + 1], FRI_10AM)
    assert len([u for u in net.urls if "PTT.BK" in u]) == 1
    clock.t += 600
    poller.due([GUILD, GUILD + 1], FRI_10AM)
    assert len([u for u in net.urls if "PTT.BK" in u]) == 2


def test_gold_poll_is_every_30_minutes_and_move_is_measured_against_yesterday(docs, data, clock):
    repo, poller = make_poller(docs, data, clock)
    repo.add(GUILD, mk.Watch("g", STAFF, "gold", "ทอง", "up_pct", 1.0))
    # yesterday's last price was 65,000: today's 66,000 is +1.54%
    docs.put(mk.STATE, GUILD, "gold", {"day": "2026-10-15", "base": 64000, "last": 65000})
    fired = poller.due([GUILD], FRI_10AM)
    assert len(fired) == 1 and fired[0].price == 66000.0 and round(fired[0].change_pct, 2) == 1.54
    repo.add(GUILD, mk.Watch("g2", 2, "gold", "ทอง", "above", 70000))
    clock.t += 600
    assert poller.due([GUILD], FRI_10AM) == []  # gold is polled every 30 min, not 10
    clock.t += 1300
    assert poller.due([GUILD], FRI_10AM) == []  # polled again, nothing new matches


def test_unavailable_source_fires_nothing(docs, clock):
    d = mk.MarketData(get=Net({"yahoo": mk.MarketError("http 429")}), clock=clock)
    repo, poller = make_poller(docs, d, clock)
    repo.add(GUILD, mk.Watch("w", STAFF, "PTT.BK", "PTT", "below", 99))
    assert poller.due([GUILD], FRI_10AM) == []


def test_alert_text_is_bilingual_with_source_and_disclaimer(docs, data, clock):
    repo, poller = make_poller(docs, data, clock)
    repo.add(GUILD, mk.Watch("w", STAFF, "PTT.BK", "PTT", "below", 31))
    (a,) = poller.due([GUILD], FRI_10AM)
    th, en = mk.format_alert(a, "th"), mk.format_alert(a, "en")
    assert "PTT" in th and "30.50" in th and "Yahoo Finance" in th and "ไม่ใช่คำแนะนำการลงทุน" in th
    assert "not investment advice" in en and "fell to 31" in en and "10:30" in th


# ------------------------------------------------------------------ brief line, company news
def test_market_line_and_failure_modes(data, clock):
    line = mk.market_line(data, [], "th", today=datetime(2026, 10, 16).date())
    assert (
        line.startswith("📈 ตลาดเช้านี้: SET 1,415.50 (")
        and "ทองแท่ง 66,000/65,800 บาท (ประกาศ 09:03)" in line
    )
    assert "ไม่ใช่คำแนะนำการลงทุน" in line and "Yahoo Finance" in line
    en = mk.market_line(data, ["PTT"], "en", today=datetime(2026, 10, 17).date())
    assert en.startswith("📈 Markets this morning: PTT 30.50") and "16/10" in en  # last close date
    dead = mk.MarketData(get=Net({}), clock=clock)
    assert mk.market_line(dead, [], "th") == ""  # nothing readable: no line, no invention


def test_company_news_is_labelled_capped_and_deduped(data, docs):
    text = mk.company_news(data, ["ABC", "คู่แข่ง X", "Z"], "th", total=3)
    assert text.startswith(t("mk_news_head", "th")) and "Google News" in text
    assert text.count("•") <= 3 and "[ABC]" in text and "https://news.google.com" in text
    # all topics return the same feed: identical headlines are not repeated
    assert len({ln for ln in text.splitlines() if ln.startswith("•")}) == text.count("•")
    assert mk.company_news(mk.MarketData(get=Net({})), ["ABC"], "th") == ""


def test_track_news_topic_is_exec_only_and_capped(docs, data):
    staff = tool(ctx_for(docs, data, is_exec=False), "track_news_topic")
    assert staff("add", "ABC")["status"] == "refused"
    boss = tool(ctx_for(docs, data, is_exec=True), "track_news_topic")
    assert boss("add", "ABC Co.")["status"] == "ok" and boss("add", "คู่แข่ง X")["status"] == "ok"
    assert boss("list")["topics"] == ["ABC Co.", "คู่แข่ง X"] or set(boss("list")["topics"]) == {
        "ABC Co.",
        "คู่แข่ง X",
    }
    assert boss("remove", "ABC Co.")["status"] == "ok" and boss("list")["topics"] == ["คู่แข่ง X"]
    for i in range(mk.MAX_TOPICS_PER_GUILD):
        boss("add", f"company {i}")
    assert boss("add", "one more")["status"] == "refused"


def test_agent_tools_wrap_the_data_layer(docs, data):
    c = ctx_for(docs, data)
    q = tool(c, "stock_quote")("PTT, ปตท, S&P 500")
    assert [x["status"] for x in q["quotes"]][0] == "ok" and len(q["quotes"]) == 3
    assert tool(c, "stock_quote")("")["status"] == "error"
    assert tool(c, "gold_price")()["thai"]["status"] == "ok"
    assert "conversion" in tool(c, "gold_price")(True)
    n = tool(c, "news")("ปตท", "", "th", 2)
    assert n["status"] == "ok" and len(n["items"]) == 2
    assert tool(c, "stock_history")("PTT.BK", "1y")["range"] == "1y"


def test_prompt_demands_tools_sources_and_a_single_disclaimer():
    assert "stock_quote" in mk.MARKETS_PROMPT and "gold_price" in mk.MARKETS_PROMPT
    assert "ห้ามเดา" in mk.MARKETS_PROMPT and "ไม่ใช่คำแนะนำการลงทุน" in mk.MARKETS_PROMPT
    from salid.config import Settings

    s = Settings(_env_file=None)
    assert "stock_quote" in system_prompt(s, False, markets=True)
    assert "stock_quote" not in system_prompt(s, False)


def test_action_tools_are_rebound_in_voice_and_flag_exists():
    assert {"market_watch", "market_brief", "track_news_topic"} <= ACTION_TOOL_NAMES
    assert not {"stock_quote", "gold_price", "news"} & ACTION_TOOL_NAMES  # public reads
    from salid.guild_settings import FEATURES, Features

    assert "markets" in FEATURES and Features().markets is True


# ------------------------------------------------------------------ the bot
def bot_with_net(bot, net=None):
    bot.market_data = mk.MarketData(get=net or Net(routes()))
    bot.market_poller = mk.Poller(bot.market_watches, bot.market_data, bot.docs)
    return bot


async def test_agent_kwargs_register_market_tools_and_the_flag_turns_them_off(bot, world):
    member = world.g.members[STAFF]
    msg = bot.room_message(world.g, member, world.pub)

    def names(**kw):
        k = bot._agent_kwargs(
            msg, world.g, member, False, setting="th", lang="th", question="q", memory=None,
            profile=None, **kw,
        )  # fmt: skip
        return {f.__name__ for f in k.get("extra_tools", [])}

    on = names()
    assert {
        "stock_quote",
        "stock_history",
        "gold_price",
        "news",
        "market_watch",
        "market_brief",
    } <= on
    assert not {"stock_quote", "market_watch"} & names(interjection=True)
    set_features(bot, markets=False)
    off = names()
    assert not {"stock_quote", "gold_price", "news", "market_watch", "market_brief"} & off
    assert bot._markets_tools(world.g, member, False, "th") == []


async def test_alert_is_delivered_by_dm_once_and_not_when_flag_off(bot, world, monkeypatch):
    bot_with_net(bot)
    monkeypatch.setattr(bot, "get_guild", lambda gid: world.g)
    bot.market_watches.add(GUILD, mk.Watch("w", STAFF, "PTT.BK", "PTT", "below", 31))
    set_features(bot, markets=False)
    assert await bot._markets_due(FRI_10AM) == 0 and not world.staff.dms
    set_features(bot, markets=True)
    bot._markets_last = -1e9
    assert await bot._markets_due(FRI_10AM) == 1
    assert "PTT" in world.staff.dms[0][0] and "not investment advice" in world.staff.dms[0][0]
    bot._markets_last = -1e9
    bot.market_poller._last.clear()
    assert await bot._markets_due(FRI_10AM) == 0  # once per condition per day


async def test_morning_brief_has_the_market_line_only_when_opted_in(bot, world):
    bot_with_net(bot)
    await bot._briefings_due(at(8, 31))  # nothing to say yet: claimed, no DM
    before = len(world.staff.dms)
    tool_ = tool(
        mk.MarketsCtx(
            guild_id=GUILD, asker_id=STAFF, is_exec=False, data=bot.market_data,
            repo=bot.market_watches, docs=bot.docs, prefs=bot.prefs,
            on_prefs_changed=lambda: bot._prefs_cache.clear(),
        ),
        "market_brief",
    )  # fmt: skip
    assert tool_(True, "SET, gold")["market_line_in_morning_brief"] is True
    bot._brief_done.clear()
    bot.docs.delete("salid_coo_state", GUILD, f"brief:morning:{STAFF}:{at(8).date().isoformat()}")
    await bot._briefings_due(at(8, 32))
    texts = [d[0] for d in world.staff.dms[before:]]
    assert any("ตลาดเช้านี้" in x and "ทองแท่ง" in x for x in texts)
    assert tool_(False)["market_line_in_morning_brief"] is False
    p = bot.prefs.get(GUILD, STAFF)
    assert p.markets is False and p.market_items == ["SET", "gold"]


async def test_company_news_section_is_in_the_report_text_when_topics_exist(bot, world):
    bot_with_net(bot)
    assert bot._company_news_section(world.g, "th") == ""
    bot.docs.put(mk.TOPICS, GUILD, "abc", {"topic": "ABC", "added_by": 1})
    sec = bot._company_news_section(world.g, "th")
    assert sec.startswith(t("mk_news_head", "th")) and "[ABC]" in sec
    set_features(bot, markets=False)
    assert bot._company_news_section(world.g, "th") == ""


# ------------------------------------------------------------------ grounded fallback
class GClient:
    """A fake Gemini client: records calls and returns canned text."""

    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents[0].parts[0].text, config))
        text = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        if isinstance(text, Exception):
            raise text
        return SimpleNamespace(text=text, candidates=[])


def gq(symbol, price=30.5, as_of="2026-10-16 10:30 ICT", **kw):
    return {
        "symbol": symbol, "name": symbol, "price": price, "currency": "THB", "change": -0.25,
        "change_pct": None, "as_of": as_of, "source_title": "SET", "source_url": "https://set.or.th/x",
        **kw,
    }  # fmt: skip


def blocked_data(clock, client, extra=None):
    net = Net({"yahoo": mk.MarketError("http 429"), **(extra or {})})
    return mk.MarketData(get=net, clock=clock, grounded=mk.Grounded(client, "m", clock)), net


def test_yahoo_429_falls_back_to_one_grounded_call_and_trips_the_breaker(clock):
    client = GClient(json.dumps([gq("PTT.BK"), gq("GC=F", 4170.5, currency="USD")]))
    d, net = blocked_data(clock, client)
    res = d.quotes(["PTT", "GC=F"])
    assert len(client.calls) == 1  # batched
    cfg = client.calls[0][2]
    assert cfg.temperature == 0.0 and cfg.tools[0].google_search is not None
    assert "delayed" in client.calls[0][1] and "never estimate" in client.calls[0][1]
    a, b = res
    assert a["status"] == "web_reference" and a["price"] == 30.5
    assert a["source"] == "Google Search (grounded)"
    assert a["confidence"] == "medium" and a["source_url"] == "https://set.or.th/x"
    assert a["change_pct"] == round(-0.25 / 30.75 * 100, 2) and a["as_of"].startswith("2026-10-16")
    assert b["price"] == 4170.5 and b["currency"] == "USD"
    n = len(net.urls)
    clock.t += 100  # inside the 10 minute breaker: Yahoo is not even tried
    d.quotes(["AAPL"])
    assert len(net.urls) == n


def test_breaker_expires_after_ten_minutes(clock):
    client = GClient(json.dumps([gq("PTT.BK")]))
    d, net = blocked_data(clock, client)
    d.quote("PTT.BK")
    n = len(net.urls)
    clock.t += 601
    d._cache.clear()
    d.quote("PTT.BK")
    assert len(net.urls) > n


def test_grounded_result_is_cached_for_a_minute(clock):
    client = GClient(json.dumps([gq("PTT.BK")]))
    d, _ = blocked_data(clock, client)
    d.quote("PTT.BK")
    d.quote("PTT.BK")
    assert len(client.calls) == 1
    clock.t += 61
    d.quote("PTT.BK")
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    "bad",
    [
        json.dumps([gq("PTT.BK", price=None)]),  # no number
        json.dumps([gq("PTT.BK", price="n/a")]),
        json.dumps([gq("PTT.BK", price=-3)]),
        "I could not find it",  # prose
        "[]",
        RuntimeError("vertex down"),
    ],
)
def test_invalid_grounded_answers_are_unavailable(clock, bad):
    d, _ = blocked_data(clock, GClient(bad))
    q = d.quote("PTT.BK")
    assert q["status"] == "unavailable" and "price" not in q


def test_grounded_json_in_a_code_fence_is_accepted(clock):
    fenced = "```json\n" + json.dumps([gq("PTT.BK")]) + "\n```"
    d, _ = blocked_data(clock, GClient(fenced))
    assert d.quote("PTT.BK")["price"] == 30.5


def test_yahoo_ok_never_calls_the_grounded_client(clock):
    client = GClient("[]")
    d = mk.MarketData(get=Net(routes()), clock=clock, grounded=mk.Grounded(client, "m", clock))
    assert d.quote("PTT.BK")["source"] == "Yahoo Finance" and client.calls == []


def test_mixed_batch_asks_grounded_only_for_what_yahoo_missed(clock):
    client = GClient(json.dumps([gq("NOPE.BK", 12.0)]))
    net = Net(routes(**{"NOPE": mk.MarketError("http 404")}))
    d = mk.MarketData(get=net, clock=clock, grounded=mk.Grounded(client, "m", clock))
    res = d.quotes(["PTT.BK", "NOPE.BK"])
    assert [r["source"] for r in res] == ["Yahoo Finance", "Google Search (grounded)"]
    assert (
        len(client.calls) == 1
        and "NOPE.BK" in client.calls[0][1]
        and "PTT.BK" not in client.calls[0][1]
    )


def test_history_and_world_gold_fall_back_too(clock):
    hist = {
        "symbol": "PTT.BK", "name": "PTT", "currency": "THB", "start": 30.0, "end": 33.0,
        "high": 34.0, "low": 29.5, "change_pct": None, "from": "2026-09-16", "to": "2026-10-16",
        "as_of": "2026-10-16 16:30 ICT", "source_title": "SET", "source_url": "https://set.or.th",
    }  # fmt: skip
    client = GClient(json.dumps([hist]), json.dumps([gq("GC=F", 4170.5), gq("THB=X", 32.4)]))
    d, _ = blocked_data(clock, client, {"goldtraders": fx("goldtraders_classic.html")})
    h = d.history("PTT.BK", "1mo")
    assert h["status"] == "ok" and h["change_pct"] == 10.0 and h["confidence"] == "low"
    g = d.gold()
    assert g["thai"]["status"] == "ok"  # the association page is still reachable
    # Yahoo is blocked: no futures (never filled from the web), spot comes from gold-api absent -> web
    assert g["world"]["futures_usd_oz"] is None
    bad = blocked_data(clock, GClient(json.dumps([{**hist, "end": None}])))[0]
    assert bad.history("PTT.BK")["status"] == "unavailable"


def test_grounded_hourly_caps(clock):
    client = GClient(json.dumps([gq("PTT.BK")]))
    g = mk.Grounded(client, "m", clock)
    for _ in range(mk.GROUNDED_POLL_PER_HOUR):
        g.quotes([("PTT.BK", "PTT")], "poll")
    with pytest.raises(mk.MarketError):
        g.quotes([("PTT.BK", "PTT")], "poll")
    g.quotes([("PTT.BK", "PTT")], "ask")  # interactive budget is separate
    clock.t += 3601
    g.quotes([("PTT.BK", "PTT")], "poll")
    assert len(client.calls) == mk.GROUNDED_POLL_PER_HOUR + 2


def test_poller_batches_grounded_symbols_and_alerts_from_them(docs, clock):
    client = GClient(json.dumps([gq("PTT.BK", 29.0), gq("KBANK.BK", 150.0)]))
    d, _ = blocked_data(clock, client)
    repo, poller = make_poller(docs, d, clock)
    repo.add(GUILD, mk.Watch("a", STAFF, "PTT.BK", "PTT", "below", 30))
    repo.add(GUILD, mk.Watch("b", STAFF, "KBANK.BK", "KBANK", "above", 140))
    fired = poller.due([GUILD], FRI_10AM)
    assert sorted(a.watch.id for a in fired) == ["a", "b"] and len(client.calls) == 1
    assert fired[0].source == "Google Search (grounded)"
    assert "Google Search (grounded)" in mk.format_alert(fired[0], "th")


def test_gold_alert_target_thai_default_or_world_spot(docs, data, clock):
    c = ctx_for(docs, data)
    watch = tool(c, "market_watch")
    th = watch("add", "gold", "above", 65000)
    assert th["status"] == "ok" and th["target"] == "thai"
    wd = watch("add", "gold", "above", 4000, "world")
    assert wd["target"] == "world" and "XAU/USD" in wd["watching"]
    assert {w["target"] for w in watch("list")["watches"]} == {"thai", "world"}
    syms = {w.symbol for w in mk.WatchRepo(docs).of_user(GUILD, STAFF)}
    assert syms == {"gold", mk.WORLD_GOLD}
    repo, poller = make_poller(docs, data, clock)
    fired = poller.due([GUILD], FRI_10AM)
    by = {a.watch.symbol: a for a in fired}
    assert by["gold"].price == 66000.0 and by["gold"].currency == "THB/บาททอง"
    assert by[mk.WORLD_GOLD].price == 4159.3 and "USD" in by[mk.WORLD_GOLD].currency


def test_gold_conversion_shows_the_ecb_rate_next_to_the_associations(data):
    c = data.gold(compare=True)["conversion"]
    assert c["usd_thb"] == 33.55 and "สมาคม" in c["usd_thb_source"]
    assert c["usd_thb_ecb"] and c["usd_thb_ecb_as_of"]  # currency.py's Frankfurter rate, 2nd source
    assert isinstance(c["fx_disagree"], bool)
