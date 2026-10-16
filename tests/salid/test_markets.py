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
        "goldtraders": fx("goldtraders_classic.html"),
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
            raise mk.MarketError("http 429")
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
    net = Net({"yahoo": mk.MarketError("http 429")})
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


def test_gold_price_returns_thai_and_world_and_degrades_per_part(clock):
    g = mk.MarketData(get=Net(routes()), clock=clock).gold()
    assert g["thai_gold"]["bar_sell"] == 66000.0
    assert g["world_gold_usd_per_oz"]["price"] == 4170.0 and g["usd_thb"]["price"] == 32.45
    half = mk.MarketData(
        get=Net(routes(goldtraders=mk.MarketError("http 500"))), clock=clock
    ).gold()
    assert half["thai_gold"]["status"] == "unavailable"
    assert half["world_gold_usd_per_oz"]["status"] == "ok"
    none = mk.MarketData(get=Net(routes(goldtraders="<html>changed</html>")), clock=clock).gold()
    assert none["thai_gold"]["status"] == "unavailable"


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
    assert tool(c, "gold_price")()["thai_gold"]["status"] == "ok"
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
    assert "PTT" in world.staff.dms[0][0] and "ไม่ใช่คำแนะนำการลงทุน" in world.staff.dms[0][0]
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
