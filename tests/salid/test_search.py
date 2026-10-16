# ruff: noqa: F811, F401
"""Information search: search_everything (scope per source, merge/rank/dedupe, Thai<->English
expansion), web_search (citations), deep_research (plan -> search -> synthesize, cap, cancel, PDF)
and the wiring (flag, voice builder). Offline: fake Discord objects, scripted clients."""

import asyncio
import json
import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from conftest import GUILD, FakeClient, json_resp, mk, text_resp
from test_coordinator import (
    ADMIN_ID,
    BOSS,
    INTERN,
    OWNER_ID,
    PEER,
    PUBLIC,
    SECRET,
    STAFF,
    at,
    world,
)
from test_modes import bot  # noqa: F401  (fixture)
from test_wiki import WChan, put

from salid import research as rs
from salid import search as sr
from salid import web_search as ws
from salid.guild_settings import GuildSettingsUpdate

BKK = ZoneInfo("Asia/Bangkok")
NOW = datetime(2026, 10, 14, 12, 0, tzinfo=BKK)


# ------------------------------------------------------------------ fakes
def grounded(text, sources, supports=()):
    """A Gemini response with grounding metadata: sources = [(title, uri)], supports =
    [(end_byte_index, [chunk indices])]."""
    chunks = [SimpleNamespace(web=SimpleNamespace(title=t, uri=u)) for t, u in sources]
    sup = [
        SimpleNamespace(segment=SimpleNamespace(end_index=e), grounding_chunk_indices=i)
        for e, i in supports
    ]
    gm = SimpleNamespace(grounding_chunks=chunks, grounding_supports=sup)
    return SimpleNamespace(text=text, candidates=[SimpleNamespace(grounding_metadata=gm)])


class Dispatch:
    """A genai client that answers by kind of call: planner (JSON), search (google_search tool),
    synthesis (anything else)."""

    def __init__(self, plan=None, synth="## สรุป\nคำตอบ [1] [9]\n", fail_search=()):
        self.plan = plan or ["คำถามย่อย A", "คำถามย่อย B", "คำถามย่อย C"]
        self.synth, self.fail_search = synth, set(fail_search)
        self.calls, self.lock = [], threading.Lock()
        self.models = self

    def generate_content(self, model, contents, config):
        text = contents[0].parts[0].text
        with self.lock:
            self.calls.append((model, text, config))
        if getattr(config, "tools", None):
            if any(k in text for k in self.fail_search):
                raise RuntimeError("boom")
            n = abs(hash(text)) % 1000
            return grounded(
                f"ข้อมูลของ {text[:12]} ตัวเลข 10 [x]",
                [("example.com", f"https://r/{n}a"), ("shared.com", "https://r/shared")],
                [(10, [0, 1])],
            )
        if config.response_mime_type == "application/json":
            if "keywords" in text.lower() and "Expand" in text:
                return json_resp({"keywords": ["quotation", "ใบเสนอราคา", "quote"]})
            return json_resp({"questions": self.plan})
        return text_resp(self.synth)


# ------------------------------------------------------------------ expansion
def test_the_glossary_maps_thai_and_english_both_ways():
    assert "quotation" in sr.glossary_terms("ใบเสนอราคาบริษัท ABC")
    assert "ใบเสนอราคา" in sr.glossary_terms("where is the quotation for ABC")
    assert sr.glossary_terms("zzz") == []


def test_expander_merges_model_synonyms_glossary_and_the_original_words():
    client = Dispatch()
    kws = sr.Expander(client, "lite")("ใบเสนอราคา ABC")
    assert kws[0] == "ใบเสนอราคา abc"
    assert {"quotation", "abc", "quote"} <= set(kws)
    assert len(kws) <= sr.MAX_KEYWORDS
    assert [c[0] for c in client.calls] == ["lite"]  # one cheap call


def test_expander_caches_and_survives_a_failing_or_missing_model():
    class Boom:
        models = SimpleNamespace(
            generate_content=lambda **kw: (_ for _ in ()).throw(RuntimeError("down"))
        )

    ex = sr.Expander(Boom(), "lite", cache_size=2)
    kws = ex("ใบเสนอราคา")
    assert "quotation" in kws  # the glossary still helps
    assert {"contract", "agreement", "สัญญา"} <= set(sr.Expander(None, "lite")("contract"))
    c = Dispatch()
    e2 = sr.Expander(c, "lite")
    e2("a b")
    e2("a b")
    assert len(c.calls) == 1


# ------------------------------------------------------------------ merge / rank / dedupe
def hit(source, title, snippet="", days_ago=0, link="", rank=0.5, who=""):
    return sr.Hit(source, title, snippet, NOW - timedelta(days=days_ago), who, link, rank)


def test_ranking_prefers_matches_then_rank_then_recency():
    kws = ["quotation", "ใบเสนอราคา"]
    old_match = hit("chat", "x", "quotation for ABC ใบเสนอราคา", days_ago=200, link="l1")
    new_nomatch = hit("chat", "y", "lunch plans", days_ago=0, link="l2")
    new_match = hit("chat", "z", "quotation v2 ใบเสนอราคา", days_ago=1, link="l3")
    out = sr.rank_hits([new_nomatch, old_match, new_match], kws, NOW)
    assert [h.link for h in out] == ["l3", "l1", "l2"]
    assert out[0].score > out[1].score > out[2].score


def test_duplicates_collapse_to_the_best_score_and_each_source_is_capped():
    same = [
        hit("chat", "a", "quotation", link="L", rank=0.1),
        hit("files", "a", "quotation", link="L", rank=0.9),
    ]
    out = sr.rank_hits(same, ["quotation"], NOW)
    assert len(out) == 1 and out[0].source == "files"
    nolink = [hit("wiki", "Page", "same text"), hit("wiki", "Page", "same  text")]
    assert len(sr.rank_hits(nolink, ["same"], NOW)) == 1
    many = [hit("chat", f"t{i}", "quotation", link=f"l{i}") for i in range(9)]
    many += [hit("wiki", "w", "quotation")]
    out = sr.rank_hits(many, ["quotation"], NOW)
    assert sum(h.source == "chat" for h in out) == sr.PER_SOURCE_CAP
    assert any(h.source == "wiki" for h in out)
    assert (
        len(
            sr.rank_hits(
                [hit("chat", f"t{i}", "q", link=str(i)) for i in range(30)], ["q"], NOW, top=10
            )
        )
        <= 10
    )


def src_of(**adapters):
    return sr.SearchSources(dict(adapters))


def run(src, query="ใบเสนอราคา", **kw):
    return sr.search_everything(
        src, lambda q: ["ใบเสนอราคา", "quotation"], query, now=NOW, zone=BKK, **kw
    )


def test_search_merges_sources_formats_results_and_reports_failures():
    def chat(kws, q, a, b):
        assert "quotation" in kws  # every adapter sees the expansion
        return [hit("chat", "#ทั่วไป", "ส่งใบเสนอราคาแล้ว", 1, "https://d/1", who="Somchai")]

    def bad(kws, q, a, b):
        raise RuntimeError("down")

    wiki = lambda kws, q, a, b: [  # noqa: E731
        sr.Hit("wiki", "วิธีขอใบเสนอราคา", "ขั้นตอน", None, "Boss", open_with="wiki_get(topic='x')")
    ]
    res = run(src_of(chat=chat, wiki=wiki, tasks=bad))
    assert res["searched"] == ["chat", "wiki", "tasks"] and res["unavailable_now"] == ["tasks"]
    first = res["results"][0]
    assert first["source"] == "chat" and first["date"] == "2026-10-13"
    assert first["who"] == "Somchai" and first["link"] == "https://d/1"
    assert first["n"] == 1 and res["results"][1]["open_with"].startswith("wiki_get")


def test_search_sources_argument_dates_and_errors():
    seen = {}

    def chat(kws, q, a, b):
        seen["range"] = (a, b)
        return []

    s = src_of(chat=chat, wiki=lambda *a: [])
    res = run(s, sources=["chats", "tasks"], since="2026-10-01", until="2026-10-10")
    assert res["searched"] == ["chat"] and res["not_searched"] == ["tasks"]
    assert (
        seen["range"][0].isoformat() == "2026-10-01"
        and seen["range"][1].isoformat() == "2026-10-10"
    )
    assert "nothing found" in res["note"]
    assert "error" in run(s, since="gibberish")
    assert "error" in run(s, since="2026-10-10", until="2026-10-01")
    assert "error" in run(s, query="  ")
    assert "error" in run(s, sources=["ledger"])
    assert run(s, sources="auto")["searched"] == ["chat", "wiki"]
    run(s, since="30 days")
    assert seen["range"][0].isoformat() == "2026-09-14"


# ------------------------------------------------------------------ scope (through the bot)
async def tools_for(bot, world, who, answering=None, interjection=False):
    member = world.g.members[who]
    where = SimpleNamespace(guild=world.g if answering else None, channel=answering)
    mem = await bot._memory_tools(where, world.g, member)
    is_exec = who in (OWNER_ID, ADMIN_ID, BOSS)
    kw = bot._agent_kwargs(
        bot.room_message(world.g, member, answering or world.pub),
        world.g, member, is_exec, setting="th", lang="th", question="q", memory=mem, profile=None,
        interjection=interjection,
    )  # fmt: skip
    return {f.__name__: f for f in kw.get("extra_tools", [])}


def seed_chat(bot):
    put(bot, 1, STAFF, "Somchai", "ส่งใบเสนอราคาให้ลูกค้า ABC แล้ว", PUBLIC)
    put(bot, 2, BOSS, "Boss", "quotation ลับของลูกค้า ABC วงเงิน 900000", SECRET, day=13)
    put(bot, 3, STAFF, "Somchai", "แนบไฟล์ [PDF: quotation-abc.pdf] ให้แล้ว", PUBLIC, day=12)


async def test_a_channel_the_asker_cannot_read_never_appears(bot, world):
    seed_chat(bot)
    staff = await tools_for(bot, world, STAFF)
    res = await asyncio.to_thread(staff["search_everything"], "ใบเสนอราคา ABC")
    texts = json.dumps(res, ensure_ascii=False)
    assert "ส่งใบเสนอราคาให้ลูกค้า" in texts and "900000" not in texts and "ลับ" not in texts
    boss = await tools_for(bot, world, BOSS)
    res = await asyncio.to_thread(boss["search_everything"], "ใบเสนอราคา ABC")
    assert "900000" in json.dumps(res, ensure_ascii=False)


async def test_english_query_finds_thai_messages_through_expansion_and_files_are_typed(bot, world):
    seed_chat(bot)
    staff = await tools_for(bot, world, STAFF)
    res = await asyncio.to_thread(staff["search_everything"], "quotation for ABC")
    by_src = {r["source"]: r for r in res["results"]}
    assert "ส่งใบเสนอราคาให้ลูกค้า" in by_src["chat"]["snippet"]  # glossary: quotation -> ใบเสนอราคา
    assert "quotation-abc.pdf" in by_src["files"]["snippet"]
    assert by_src["files"]["link"].startswith("https://discord.com/channels/")
    only = await asyncio.to_thread(staff["search_everything"], "ABC", sources=["files"])
    assert {r["source"] for r in only["results"]} == {"files"}


async def test_the_ledger_is_searched_for_executives_and_own_receipts_for_the_rest(bot, world):
    mk(bot.store, 1, STAFF, "Somchai", "Makro ABC", "2026-10-10", 1500.0)
    mk(bot.store, 2, BOSS, "Boss", "Makro XYZ", "2026-10-11", 900.0)
    staff = await tools_for(bot, world, STAFF)
    boss = await tools_for(bot, world, BOSS)
    s = await asyncio.to_thread(staff["search_everything"], "Makro")
    assert "receipts" in s["searched"]  # a non-exec searches their OWN receipts only
    titles = [r["title"] for r in s["results"] if r["source"] == "receipts"]
    assert len(titles) == 1 and "ABC" in titles[0]
    b = await asyncio.to_thread(boss["search_everything"], "Makro")
    assert "receipts" in b["searched"]
    assert {r["title"][:10] for r in b["results"] if r["source"] == "receipts"} >= {
        "Makro ABC "[:10],
        "Makro XYZ "[:10],
    }


async def test_meetings_only_where_the_asker_was_invited(bot, world):
    base = {
        "guild_id": GUILD, "status": "scheduled", "organizer_id": BOSS, "organizer_name": "Boss",
        "start_utc": (NOW + timedelta(days=1)).astimezone(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip
    bot.secretary_store.put_meeting(
        {
            **base,
            "id": "m1",
            "title": "ประชุมใบเสนอราคา ABC",
            "attendee_ids": [STAFF],
            "attendee_names": ["Somchai"],
        }
    )
    bot.secretary_store.put_meeting(
        {
            **base,
            "id": "m2",
            "title": "ประชุมใบเสนอราคาลับ",
            "attendee_ids": [BOSS],
            "attendee_names": ["Boss"],
        }
    )
    staff = await tools_for(bot, world, STAFF)
    res = await asyncio.to_thread(staff["search_everything"], "ใบเสนอราคา", since="2026-10-01")
    titles = [r["title"] for r in res["results"] if r["source"] == "meetings"]
    assert titles == ["ประชุมใบเสนอราคา ABC"]


def test_wiki_tasks_and_places_go_through_the_scoped_tool_objects():
    calls = []

    def wiki_search(query):
        calls.append(("wiki", query))
        return {
            "pages": [
                {
                    "title": "ขั้นตอนใบเสนอราคา",
                    "snippet": "ทำตามนี้",
                    "updated_at": "2026-10-01",
                    "author": "Boss",
                }
            ]
        }

    def list_tasks(person=None, status="open"):
        calls.append(("tasks", status))
        return {"tasks": [
            {"title": "ส่งใบเสนอราคา ABC", "assignee": "Peer", "due": "2026-10-20", "status": "open", "link": "https://d/t"},
            {"title": "ซื้อกาแฟ", "assignee": "Peer", "status": "open"},
        ]}  # fmt: skip

    def find_place(q):
        calls.append(("place", q))
        if q != "abc":
            return {"count": 0, "places": []}
        return {
            "places": [{"name": "ออฟฟิศลูกค้า ABC", "address": "1 ถนน", "maps_link": "https://maps/x"}]
        }

    for f, n in (
        (wiki_search, "wiki_search"),
        (list_tasks, "list_tasks"),
        (find_place, "find_place"),
    ):
        f.__name__ = n
    src = sr.build_sources(
        memory=None,
        tools=[wiki_search, list_tasks, find_place],
        zone=BKK,
        now=lambda: NOW,
        user_id=1,
    )
    assert sorted(src.adapters) == ["places", "tasks", "wiki"]
    res = sr.search_everything(
        src, lambda q: ["abc", "quotation", "ใบเสนอราคา"], "ABC", now=NOW, zone=BKK
    )
    kinds = {r["source"]: r for r in res["results"]}
    assert set(kinds) == {"wiki", "tasks", "places"}
    assert kinds["tasks"]["link"] == "https://d/t" and "ซื้อกาแฟ" not in json.dumps(
        res, ensure_ascii=False
    )
    assert kinds["places"]["link"] == "https://maps/x"
    assert ("wiki", "ABC") in calls


# ------------------------------------------------------------------ web_search
def test_citations_land_after_the_grounded_text_in_utf8_offsets_and_sources_are_numbered():
    text = "ราคาทอง 66000 บาท ข่าวอื่น"
    end1 = len("ราคาทอง 66000 บาท".encode())
    resp = grounded(
        text,
        [
            ("thairath.co.th", "https://r/1"),
            ("xn--42cah7d0cxcvbbb9x.com", "https://r/2"),
            ("a.com", "https://r/1"),
        ],
        [(end1, [0, 1]), (len(text.encode()), [2])],
    )
    assert ws.add_citations(text, resp) == "ราคาทอง 66000 บาท[1][2] ข่าวอื่น[1]"
    srcs = ws.extract_sources(resp)
    assert [s["n"] for s in srcs] == [1, 2]  # the repeated link is one source
    assert srcs[0] == {
        "n": 1,
        "title": "thairath.co.th",
        "domain": "thairath.co.th",
        "link": "https://r/1",
    }
    assert srcs[1]["domain"] == "ทองคำ".join(["", ""]) or srcs[1]["domain"].endswith(".com")
    assert (
        ws.format_sources(srcs).splitlines()[0]
        == "[1] thairath.co.th - thairath.co.th - https://r/1"
    )


def test_web_search_call_puts_today_site_and_recency_in_the_request():
    client = Dispatch()
    res = ws.web_search_call(
        client,
        "m",
        "latest policy rate",
        recency="week",
        site="https://www.bot.or.th/path",
        today=NOW.date(),
    )
    model, text, cfg = client.calls[0]
    assert text == "latest policy rate site:bot.or.th"
    ins = cfg.system_instruction
    assert "2026-10-14" in ins and "Bangkok" in ins and "last 7 day" in ins and "bot.or.th" in ins
    assert cfg.tools[0].google_search is not None and len(cfg.tools) == 1  # only Google Search
    assert res["date"] == "2026-10-14" and res["sources"] and "[1]" in res["answer"]
    assert ws.clean_site("not a domain") == "" and ws.clean_site("Bot.or.th") == "bot.or.th"
    bad = ws.web_search_call(client, "m", "  ")
    assert "error" in bad
    client.fail_search = {"boom"}
    assert "error" in ws.web_search_call(client, "m", "boom please")


# ------------------------------------------------------------------ deep research pipeline
TODAY = NOW.date()


def go(client, **kw):
    return rs.run_research(
        client, question="เปรียบเทียบ POS ร้านกาแฟ", depth=kw.pop("depth", "quick"), lang="th",
        today=TODAY, search_model="main", plan_model="lite", synth_model="main", **kw,
    )  # fmt: skip


def test_plan_search_synthesize_with_global_citations_and_a_code_written_source_list():
    client = Dispatch()
    res = go(client)
    kinds = [
        ("search" if c[2].tools else "json" if c[2].response_mime_type else "synth", c[0])
        for c in client.calls
    ]
    assert kinds[0] == ("json", "lite")  # the cheap model plans first
    assert sorted(k for k, _ in kinds[1:4]) == ["search"] * 3 and kinds[-1] == ("synth", "main")
    assert res.sub_questions == client.plan and res.failed == 0
    # 3 searches x (own source + the shared one) = 4 distinct sources; the shared one is deduped
    assert len({s["link"] for s in res.sources}) == len(res.sources)
    synth_prompt = client.calls[-1][1]
    assert "Sub-question 1: คำถามย่อย A" in synth_prompt and "[1]" in synth_prompt
    assert "[9]" not in res.report.split("## แหล่งอ้างอิง")[0]  # a made-up citation is stripped
    assert "## แหล่งอ้างอิง" in res.report and "https://r/" in res.report
    assert res.report.count("[1]") >= 2  # in the text and in the list


def test_depth_sets_the_number_of_sub_questions_and_a_failed_planner_falls_back():
    six = Dispatch(plan=[f"q{i}" for i in range(8)])
    assert len(rs.plan_questions(six, "lite", "x", "thorough", TODAY)) == 6
    assert len(rs.plan_questions(Dispatch(plan=["a", "b"]), "lite", "x", "quick", TODAY)) == 2
    broken = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **k: 1 / 0))
    assert rs.plan_questions(broken, "lite", "ทำรีเสิร์ช", "quick", TODAY) == ["ทำรีเสิร์ช"]
    assert rs.clean_depth("weird") == "quick"


def test_a_failed_sub_search_is_skipped_and_a_failed_synthesis_shows_the_raw_findings():
    client = Dispatch(fail_search=["คำถามย่อย B"])
    res = go(client)
    assert res.failed == 1 and len(res.findings) == 2
    client = Dispatch()
    client.synth = ""
    res = go(client)
    assert "## ผลการค้นหา" in res.report and "คำถามย่อย A" in res.report
    allfail = Dispatch(fail_search=["คำถามย่อย"])
    res = go(allfail)
    assert "ค้นเว็บแล้วไม่ได้" in res.report and len(allfail.calls) == 4  # no synthesis call


def test_internal_context_reaches_only_the_synthesis_never_a_search():
    client = Dispatch()
    go(client, internal=lambda: "- (chat) ลูกค้า ABC สั่งซื้อเมื่อวาน")
    for _model, text, cfg in client.calls:
        has_ctx = "ลูกค้า ABC สั่งซื้อ" in text
        assert has_ctx == (not cfg.tools and not cfg.response_mime_type)


def test_cancel_stops_between_steps():
    h = rs.ResearchHandle()
    client = Dispatch()
    orig = client.generate_content

    def cancelling(model, contents, config):
        if config.tools:
            h.cancel()  # the user says "cancel" while searching
        return orig(model, contents, config)

    client.models = SimpleNamespace(generate_content=cancelling)
    with pytest.raises(rs.ResearchCancelled):
        go(client, handle=h)
    assert not any(
        not c[2].tools and not c[2].response_mime_type for c in client.calls
    )  # no synthesis


def test_the_pdf_renders_headings_bullets_and_tables():
    from salid.pdf_report import find_fonts

    body = (
        "## สรุป\nข้อความ **หนา** [1]\n\n- ข้อ 1\n- ข้อ 2\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n## Sources\n[1] x - y.com - https://"
        + "z" * 200
    )
    pdf = rs.markdown_pdf(
        org="Co", title="หัวข้อ", body=body, lang="th", generated_at=NOW, fonts=find_fonts("")
    )
    assert pdf.startswith(b"%PDF") and len(pdf) > 1000
    short = rs.short_version("## สรุป\n" + "ก" * 3000, "th")
    assert len(short) <= rs.SHORT_CHARS


# ------------------------------------------------------------------ the bot: tools, job, cap, cancel
class Out:
    """Where the report is posted."""

    def __init__(self):
        self.sent = []
        self.id = PUBLIC

    async def send(self, text=None, **kw):
        self.sent.append((text, kw))


async def start(bot, world, who=STAFF, question="ทำรีเสิร์ชเรื่อง POS ร้านกาแฟ", **kw):
    out = Out()
    bot_msg = SimpleNamespace(channel=out, guild=world.g)
    member = world.g.members[who]
    fns = bot._search_tools(bot_msg, world.g, member, False, None, [], "th")
    return {f.__name__: f for f in fns}, out


async def finish(bot, gid=GUILD):
    for _ in range(50):
        job = bot._research_jobs.get(gid)
        if job is None:
            break
        await asyncio.wait({job.task}, timeout=10)
    await asyncio.sleep(0)


async def test_deep_research_runs_in_the_background_and_posts_text_and_pdf(bot, world):
    bot.llm = Dispatch()
    fns, out = await start(bot, world)
    res = await asyncio.to_thread(fns["deep_research"], "เปรียบเทียบ POS", "thorough")
    assert res["status"] == "started" and "1–2 นาที" in res["say"]
    await finish(bot)
    ((text, kw),) = out.sent
    assert text.startswith(f"<@{STAFF}>") and "ค้นข้อมูลเสร็จแล้ว" in text and len(text) <= 2000
    assert kw["file"].filename == "research.pdf" and kw["file"].fp.read(4) == b"%PDF"
    assert GUILD not in bot._research_jobs


async def test_a_second_research_waits_and_the_daily_cap_is_per_guild(bot, world):
    bot.s.research_daily_cap = 2
    bot._research_budget = type(bot._research_budget)(2)
    bot.llm = Dispatch()
    fns, out = await start(bot, world)
    first = await asyncio.to_thread(fns["deep_research"], "หนึ่ง")
    busy = await asyncio.to_thread(fns["deep_research"], "สอง")  # the first is still running
    assert first["status"] == "started" and busy["error"] == "busy"
    await finish(bot)
    assert (await asyncio.to_thread(fns["deep_research"], "สาม"))["status"] == "started"
    await finish(bot)
    limit = await asyncio.to_thread(fns["deep_research"], "สี่")
    assert limit["error"] == "daily_limit" and "2" in limit["say"]
    assert bot._research_budget.available(GUILD + 1, bot.s.now().date().isoformat())  # other guild
    assert len(out.sent) == 2


async def test_cancel_deep_task_words_stop_a_running_research(bot, world):
    gate = threading.Event()
    client = Dispatch()
    orig = client.generate_content

    def slow(model, contents, config):
        if config.tools:
            gate.wait(5)
        return orig(model, contents, config)

    client.models = SimpleNamespace(generate_content=slow)
    bot.llm = client
    fns, out = await start(bot, world)
    await asyncio.to_thread(fns["deep_research"], "POS")
    await asyncio.sleep(0.05)
    stranger = world.g.members[PEER]
    from salid.i18n import t as tr

    assert await bot._cancel_deep(GUILD, stranger, "th") == tr("cancel_denied", "th")
    reply = await bot._cancel_deep(GUILD, world.g.members[STAFF], "th")
    gate.set()
    await finish(bot)
    assert "หยุด" in reply and out.sent == []  # cancelled: nothing is posted
    assert "ไม่มีงานใหญ่" in await bot._cancel_deep(GUILD, world.g.members[STAFF], "th")


async def test_deep_research_with_internal_context_uses_only_the_scoped_search(bot, world):
    seed_chat(bot)
    client = Dispatch()
    bot.llm = client
    member = world.g.members[STAFF]
    mem = await bot._memory_tools(SimpleNamespace(guild=None, channel=None), world.g, member)
    out = Out()
    fns = {f.__name__: f for f in bot._search_tools(
        SimpleNamespace(channel=out, guild=world.g), world.g, member, False, mem, [], "th")}  # fmt: skip
    await asyncio.to_thread(fns["deep_research"], "ใบเสนอราคา ABC", "quick", True)
    await finish(bot)
    synth = client.calls[-1][1]
    assert "ส่งใบเสนอราคาให้ลูกค้า" in synth and "900000" not in synth  # SECRET stays out
    for _m, text, cfg in client.calls:
        if cfg.tools:
            assert "ส่งใบเสนอราคา" not in text  # searches never carry team data


# ------------------------------------------------------------------ wiring
async def test_tools_reach_chat_and_voice_builder_and_follow_the_flags(bot, world):
    bot.llm = Dispatch()
    t = await tools_for(bot, world, STAFF)
    assert {"search_everything", "web_search", "deep_research"} <= set(t)
    assert t["search_everything"].salid_group == "search"
    quiet = await tools_for(bot, world, STAFF, interjection=True)
    assert not {"search_everything", "web_search", "deep_research"} & set(quiet)
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={"web": False}), "x")
    bot.team._cache.clear()
    no_web = await tools_for(bot, world, STAFF)
    assert "search_everything" in no_web and not {"web_search", "deep_research"} & set(no_web)
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={"search": False}), "x")
    bot.team._cache.clear()
    off = await tools_for(bot, world, STAFF)
    assert not {"search_everything", "web_search", "deep_research"} & set(off)


async def test_prompt_guidance_is_added_only_with_the_tools():
    from conftest import FixedClockSettings

    from salid import qa
    from salid.search import SEARCH_PROMPT

    s = FixedClockSettings()
    assert SEARCH_PROMPT in qa.system_prompt(s, scoped=False, search=True)
    assert SEARCH_PROMPT not in qa.system_prompt(s, scoped=False)
    for name in ("search_everything", "web_search", "deep_research"):
        assert name in SEARCH_PROMPT
