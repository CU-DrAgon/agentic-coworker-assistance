"""English mode: document generators, report renderers and tool notes emit no Thai."""

import re
from datetime import date

import pytest
from conftest import GUILD

from salid import benchmark, flights, markets, research
from salid.deep_task import DeepTaskError, DeepTaskTimeout, run_deep_task
from salid.domain_packs import PACKS, pack_text

THAI = re.compile(r"[฀-๿]")


def no_thai(text: str) -> bool:
    return THAI.search(text) is None


def test_localize_en_makes_market_constants_english():
    for s in (
        markets.THAI_UNIT,
        markets.WORLD_UNIT,
        markets.SOURCE_GOLD,
        markets.DELAY_NOTE,
        markets.WEB_NOTE,
        markets.ESTIMATE_NOTE,
        "Futures (COMEX GC=F) — ไม่ใช่ราคา Spot / not spot",
        "ไม่ระบุเวลา",
        "Spot (อ้างอิงจากสมาคมค้าทองคำ)",
    ):
        assert no_thai(markets.localize_en(s)), s
    assert (
        markets.localize_en({"a": [markets.DELAY_NOTE]})["a"][0] == "may be delayed 15-20 minutes"
    )
    assert markets.localize_en("plain 12") == "plain 12"


class FakeData:
    def gold(self, compare=False):
        return {
            "status": "ok",
            "thai": {"unit": markets.THAI_UNIT, "source": markets.SOURCE_GOLD,
                     "change": {"direction": "ขึ้น", "unit": "บาท ต่อน้ำหนัก 1 บาททองคำ เทียบกับประกาศครั้งก่อน"}},
            "world": {"unit": markets.WORLD_UNIT,
                      "futures_label": "Futures (COMEX GC=F) — ไม่ใช่ราคา Spot / not spot"},
            "summary": "ราคาทองไทย",
            "summary_en": "Thai gold: 1",
        }  # fmt: skip

    def quotes(self, names, market="auto"):
        return [{"status": "ok", "note": markets.DELAY_NOTE, "source": "Yahoo Finance"}]


def _tools(lang):
    ctx = markets.MarketsCtx(
        guild_id=GUILD, asker_id=1, is_exec=True, data=FakeData(), repo=None, docs=None, lang=lang
    )
    return {f.__name__: f for f in markets.markets_tools(ctx)}


def test_gold_and_quote_tools_english_only_in_en():
    res = _tools("en")["gold_price"]()
    assert no_thai(str(res)) and res["summary"] == "Thai gold: 1"
    assert no_thai(str(_tools("en")["stock_quote"]("PTT")))
    th = _tools("th")["gold_price"]()  # Thai guild: unchanged
    assert th["summary"] == "ราคาทองไทย" and th["thai"]["unit"] == markets.THAI_UNIT


def test_benchmark_render_english():
    clean = {
        "summary": "s", "gaps": [], "ours_wins": [], "ours_loses": [], "next_moves": [],
        "competitors": [{"name": "A", "pricing": "1", "strengths": "x", "weaknesses": "y",
                         "thai_support": "yes", "sources": [1]}],
    }  # fmt: skip
    src = [{"n": 1, "title": "t", "link": "https://x.example"}]
    en = benchmark.render("topic", clean, src, 1, "en", date(2026, 1, 2))
    assert no_thai(en)
    assert "เทียบตลาด" in benchmark.render("topic", clean, src, 1, "th", date(2026, 1, 2))


def test_research_prompt_english():
    p = research.synth_prompt("q", [], [], "", "en", date(2026, 1, 2), "deep")
    assert "Summary" in p and "ค่ะ" not in p
    assert "สรุป" in research.synth_prompt("q", [], [], "", "th", date(2026, 1, 2), "deep")


def test_domain_pack_labels_english():
    text = pack_text(next(iter(PACKS.values())), "en")
    assert "Vocabulary:" in text and "คำศัพท์:" not in text
    assert "คำศัพท์:" in pack_text(next(iter(PACKS.values())), "th")


def test_flight_prompts_and_notes_english():
    flight = flights.normalize_flight("TG 676")
    assert no_thai(flights.status_prompt(flight, "2026-01-02", "2026-01-01", "en"))
    assert "สายการบิน" in flights.status_prompt(flight, "2026-01-02", "2026-01-01")
    p = flights.prices_prompt("BKK", "NRT", "2026-02-01", "", 1, "economy", "2026-01-01", "en")
    assert "in English" in p
    svc = flights.FlightService(client=object(), model="m", today=lambda: date(2026, 1, 1))
    svc._ground = lambda prompt: {"text": "ok", "sources": []}
    assert no_thai(str(svc.status("TG 676", "2026-01-02", "en")))
    assert "ข้อมูลจาก" in svc.status("TG 676", "2026-01-02")["note"]
    assert no_thai(str(svc.prices("BKK", "NRT", "2026-02-01", lang="en")["label"]))


def test_deep_task_timeout_message_follows_language(tmp_path):
    from salid.config import Settings

    s = Settings(gemini_cli_path="/bin/true")

    def boom(*a):
        raise DeepTaskTimeout(77)

    class Store:
        def all_receipts(self, *a, **k):
            return []

    with pytest.raises(DeepTaskError) as e:
        run_deep_task(s, Store(), "task", guild_id=GUILD, runner=boom, lang="en")
    assert no_thai(str(e.value)) and "77" in str(e.value)
    with pytest.raises(DeepTaskError, match="นานเกิน"):
        run_deep_task(s, Store(), "งาน", guild_id=GUILD, runner=boom, lang="th")
