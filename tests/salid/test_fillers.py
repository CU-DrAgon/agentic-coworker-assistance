"""Model-written "one moment" lines (salid.fillers): generation with a fake client, the checks on
what the model says, the hand-written fallback pools and the timeout race. No network."""

import asyncio
import random
import re
from types import SimpleNamespace

import pytest
from conftest import FixedClockSettings

from salid import fillers
from salid.ack import Ack
from salid.coo_store import FirestoreDocStore, SQLiteDocStore
from salid.fillers import (
    ClipCache,
    FillerPool,
    FillerWriter,
    RecentLines,
    clean_line,
    is_generic,
    parse_reply,
    pick_fallback,
    pool_key,
    within,
)
from salid.i18n import t

THAI = re.compile(r"[฀-๿]")


def scripted(monkeypatch, *texts, boom=False):
    """Replace fillers.generate: answers with the given texts in order; records the prompts."""
    seen = []
    queue = list(texts)

    def fake(client, model, contents, config, delays=None):
        seen.append(SimpleNamespace(model=model, prompt=contents, config=config, delays=delays))
        if boom:
            raise RuntimeError("vertex down")
        return SimpleNamespace(text=queue.pop(0) if queue else "")

    monkeypatch.setattr(fillers, "generate", fake)
    return seen


def writer(**kw):
    return FillerWriter(object(), FixedClockSettings(**kw))


def call(w, **kw):
    base = dict(kind="slow", question="ยอดเดือนนี้เท่าไร", lang="th", particle="ค่ะ")
    return w.write_blocking(**{**base, **kw})


# ------------------------------------------------------------------ generation
def test_generated_line_uses_the_cheap_model_with_a_short_hot_call(monkeypatch):
    seen = scripted(monkeypatch, "ขอเช็กยอดเดือนนี้แป๊บนะคะ")
    w = writer()
    assert call(w, avoid=["เดี๋ยวดูให้เลยค่ะ"]) == "ขอเช็กยอดเดือนนี้แป๊บนะคะ"
    (c,) = seen
    assert c.model == w.s.gate_model and c.delays == ()  # no retries
    assert c.config.temperature == 1.0 and c.config.max_output_tokens <= 80
    assert "ยอดเดือนนี้เท่าไร" in c.prompt and "เดี๋ยวดูให้เลยค่ะ" in c.prompt
    assert "ค่ะ" in c.prompt


def test_english_prompt_and_stage_again(monkeypatch):
    seen = scripted(monkeypatch, "Still pulling it together, thanks for waiting.")
    out = call(writer(), lang="en", question="what did we spend?", stage="again")
    assert out and not THAI.search(out)
    assert "almost there" in seen[0].prompt.lower() and "12 words" in seen[0].prompt
    assert not THAI.search(seen[0].prompt)


def test_the_particle_is_applied(monkeypatch):
    scripted(monkeypatch, "ขอเช็กให้แป๊บนะคะ")
    assert call(writer(), particle="ครับ") == "ขอเช็กให้แป๊บนะครับ"


def test_question_is_truncated_in_the_prompt(monkeypatch):
    seen = scripted(monkeypatch, "ขอดูให้แป๊บนะคะ")
    call(writer(), question="ก" * 2000)
    assert "ก" * 300 in seen[0].prompt and "ก" * 301 not in seen[0].prompt


@pytest.mark.parametrize(
    ("raw", "lang"),
    [
        ("", "th"),
        ("   ", "th"),
        ("ก" * 91, "th"),
        ("a " * 61, "en"),
        ("ขอดูให้นะคะ\nแป๊บนึง", "th"),
        ("ดูที่ https://x.co นะคะ", "th"),
        ("```ขอดูนะคะ```", "th"),
        ("ยอดรวม 5,000 บาทค่ะ", "th"),  # an answer-like amount
        ("รวม ฿5,000 ค่ะ", "th"),
        ("Let me check.", "th"),  # wrong language
        ("ขอเช็กให้นะคะ", "en"),
    ],
)
def test_bad_lines_are_rejected(raw, lang):
    assert clean_line(raw, lang=lang, question="ยอดเดือนนี้เท่าไร") is None


def test_clean_line_strips_quotes_and_markdown_and_keeps_asked_numbers():
    assert clean_line("“**ขอเช็กให้แป๊บนะคะ**”", lang="th") == "ขอเช็กให้แป๊บนะคะ"
    assert clean_line("เช็กยอด 500 บาทให้นะคะ", lang="th", question="ยอด 500 บาทใช่ไหม")
    assert clean_line("Let me check.", lang="en") == "Let me check."
    assert clean_line("ใกล้เสร็จแล้วค่ะ", lang="th", stage="again") is None
    assert clean_line("Almost there!", lang="en", stage="again") is None


def test_lines_with_facts_may_be_longer_and_must_carry_the_numbers():
    facts = ["ใช้เวลาประมาณ 1-2 นาที", "ส่งไฟล์ PDF ที่ห้องนี้"]
    ok = "ขอค้นข้อมูลให้ประมาณ 1-2 นาทีนะคะ เสร็จแล้วส่งรายงานพร้อมไฟล์ PDF ที่ห้องนี้ค่ะ" + "ค่ะ" * 10
    assert clean_line(ok, lang="th", facts=facts) == ok
    assert clean_line("ขอค้นข้อมูลให้ก่อนนะคะ", lang="th", facts=facts) is None  # dropped 1-2


def test_any_failure_is_none(monkeypatch):
    scripted(monkeypatch, boom=True)
    assert call(writer()) is None
    scripted(monkeypatch, "")
    assert call(writer()) is None


def test_ack_llm_off_never_calls_the_model(monkeypatch):
    seen = scripted(monkeypatch, "ขอเช็กให้แป๊บนะคะ")
    w = writer(ack_llm=False)
    assert call(w) is None and asyncio.run(w.write(kind="slow", question="q", lang="th")) is None
    assert seen == [] and not w.enabled()
    assert not FillerWriter(None, FixedClockSettings()).enabled()


def test_settings_defaults():
    s = FixedClockSettings()
    assert s.ack_llm is True and s.ack_llm_timeout_s == 2.0


# ------------------------------------------------------------------ fallback pools
KINDS = [*fillers.KINDS, "again"]


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_has_eight_natural_lines_in_both_languages(kind):
    for lang in ("th", "en"):
        lines = [t(f"ack_{kind}_{n}", lang) for n in range(1, 9)]
        assert len(set(lines)) == 8 and all(lines)
    th = [t(f"ack_{kind}_{n}", "th") for n in range(1, 9)]
    assert all(x.endswith(("ค่ะ", "คะ", "นะ")) or "ค่ะ" in x or "คะ" in x for x in th)
    assert not any("ใกล้เสร็จแล้ว" in x for x in th)


def test_fallback_never_repeats_within_avoid():
    avoid = []
    for _ in range(8):  # the whole pool is walked without a repeat
        line = pick_fallback("slow", "th", avoid)
        assert line not in avoid
        avoid.append(line)
    assert pick_fallback("slow", "th", avoid) in avoid  # exhausted: the pool again, no crash
    assert pick_fallback("nonsense", "en") in [t(f"ack_slow_{n}", "en") for n in range(1, 9)]


def test_fallback_applies_the_particle():
    lines = {pick_fallback("voice", "th", particle="ครับ") for _ in range(60)}
    assert lines and all("ค่ะ" not in x and "คะ" not in x for x in lines)
    assert any("ครับ" in x for x in lines)
    en = {pick_fallback("voice", "en", particle="ครับ") for _ in range(30)}
    assert all(not THAI.search(x) for x in en)


def test_recent_lines_keep_the_last_four_per_key():
    r = RecentLines()
    for i in range(6):
        r.add("a", str(i))
    r.add("b", "x")
    r.add("b", "")
    assert r.get("a") == ("2", "3", "4", "5") and r.get("b") == ("x",) and r.get("c") == ()


# ------------------------------------------------------------------ timeout race
async def test_within_returns_a_ready_line_and_drops_a_late_one():
    async def quick():
        return "เร็ว"

    async def slow():
        await asyncio.sleep(1)
        return "ช้า"

    assert await within(asyncio.create_task(quick()), 0.5) == "เร็ว"
    late = asyncio.create_task(slow())
    assert await within(late, 0.05) is None
    await asyncio.sleep(0)
    assert late.cancelled() or late.done()  # never used afterwards
    assert await within(None, 1) is None

    async def bad():
        raise RuntimeError("x")

    assert await within(asyncio.create_task(bad()), 0.5) is None


class Msg:
    def __init__(self):
        self.replies = []

    async def reply(self, text, **kw):
        self.replies.append(text)
        return SimpleNamespace(id=len(self.replies))


async def test_ack_sends_one_message_in_every_case():
    async def line(text, delay):
        await asyncio.sleep(delay)
        return text

    for pending_text, delay, expect in [
        ("generated", 0.0, "generated"),
        ("late", 0.5, "fallback"),
        (None, 0.0, "fallback"),
    ]:
        m = Msg()
        ack = Ack(
            m,
            "fallback",
            pending=asyncio.create_task(line(pending_text, delay)),
            timeout=0.05,
        )
        await ack.now()
        await asyncio.sleep(0.6)
        assert m.replies == [expect]
    # the answer arrived before the timer: nothing is sent and the writer is cancelled
    m = Msg()
    task = asyncio.create_task(line("x", 5))

    async def never(_):
        await asyncio.sleep(10)

    ack = Ack(m, "fallback", sleep=never, pending=task)
    ack.after(6)
    await ack.finish()
    await asyncio.sleep(0)
    assert m.replies == [] and task.cancelled()


# ------------------------------------------------------------------ generic lines and the pool
def js(line, generic):
    import json

    return json.dumps({"line": line, "generic": generic}, ensure_ascii=False)


def test_json_reply_is_parsed_and_plain_text_is_not_generic():
    assert parse_reply(js("ขอดูให้แป๊บนึงนะคะ", True)) == ("ขอดูให้แป๊บนึงนะคะ", True)
    assert parse_reply("```json\n" + js("x", True) + "\n```") == ("x", True)
    assert parse_reply("ขอดูให้นะคะ") == ("ขอดูให้นะคะ", False)
    assert parse_reply('{"line": 5}') == ("", False)


def test_generic_is_double_checked_in_code():
    assert is_generic("ขอดูให้แป๊บนึงนะคะ", "ยอดขายเดือนกันยายน")
    assert not is_generic("ขอเช็กยอด 2 เดือนนะคะ", "x")  # digits
    assert not is_generic("Let me check the budget", "what is the budget")  # echoes the question


def test_only_generic_lines_are_marked_generic(monkeypatch):
    w = writer()
    scripted(monkeypatch, js("ขอดูให้แป๊บนึงนะคะ", True))
    line = w.generate_blocking(kind="slow", question="ยอดเดือนนี้", lang="th")
    assert line.text == "ขอดูให้แป๊บนึงนะคะ" and line.generic
    scripted(monkeypatch, js("ขอเช็กยอดเดือนก.ย.ให้แป๊บนะคะ", False))
    assert not w.generate_blocking(kind="slow", question="ยอดเดือนก.ย.", lang="th").generic
    scripted(monkeypatch, js("ขอเช็กให้นะคะ", True))
    facts = w.generate_blocking(kind="deep", question="q", lang="th", facts=("x",))
    assert facts is None or not facts.generic  # a line carrying facts is never reusable


def make_docs(kind, tmp_path):
    if kind == "sqlite":
        return SQLiteDocStore(str(tmp_path / "d.db"))
    from fake_firestore import FakeFirestore

    return FirestoreDocStore(client=FakeFirestore())


@pytest.fixture(params=["sqlite", "firestore"])
def docs(request, tmp_path):
    return make_docs(request.param, tmp_path)


def pool_of(docs, **kw):
    s = FixedClockSettings(**kw)
    return FillerPool(docs, s, rng=random.Random(7))


def test_pool_saves_persists_and_dedupes(docs):
    p = pool_of(docs)
    key = pool_key("slow", "th", "first")
    assert p.add(key, "ขอดูให้แป๊บนึงนะคะ") and not p.add(key, "ขอดูให้แป๊บนึงนะคะ")
    assert not p.add(key, "  ")
    assert pool_of(docs).size(key) == 1  # a new process reads it back from the store
    assert pool_of(docs).size(pool_key("slow", "en", "first")) == 0  # per language
    assert pool_key("slow", "th", "first", "ครับ") != key


def test_below_the_minimum_always_generates(docs):
    p = pool_of(docs, filler_pool_min=3)
    key = pool_key("doc", "th", "first")
    p.add(key, "a")
    p.add(key, "b")
    assert all(p.pick(key) is None for _ in range(30))


def test_at_the_minimum_only_the_fresh_ratio_generates(docs):
    p = pool_of(docs, filler_pool_min=3, filler_fresh_ratio=0.25)
    key = pool_key("doc", "th", "first")
    for x in "abc":
        p.add(key, x)
    got = [p.pick(key) for _ in range(2000)]
    fresh = sum(g is None for g in got) / len(got)
    assert 0.2 < fresh < 0.3
    p2 = pool_of(docs, filler_pool_min=3, filler_fresh_ratio=0.0)
    assert all(p2.pick(key) for _ in range(50))
    p3 = pool_of(docs, filler_pool_min=3, filler_fresh_ratio=1.0)
    assert all(p3.pick(key) is None for _ in range(50))


def test_pick_avoids_recent_lines_and_favours_the_least_used(docs):
    p = pool_of(docs, filler_pool_min=3, filler_fresh_ratio=0.0)
    key = pool_key("doc", "th", "first")
    for x in "abc":
        p.add(key, x)
    assert {p.pick(key, avoid=["a", "b"]) for _ in range(30)} == {"c"}
    p.add(key, "fresh")
    for _ in range(300):
        p.pick(key)
    uses = p._lines(key)
    assert max(uses.values()) - min(uses.values()) < 0.5 * max(uses.values())  # evened out
    assert p.pick(key, avoid=list(uses)) is None  # everything recent: generate


def test_least_used_is_preferred_statistically(docs):
    p = pool_of(docs, filler_pool_min=2, filler_fresh_ratio=0.0)
    key = pool_key("doc", "th", "first")
    p.add(key, "old")
    p.add(key, "new")
    p._lines(key)["old"] = 50
    picks = [p.pick(key) for _ in range(40)]
    assert picks.count("new") > picks.count("old") * 3


def test_pool_cap_evicts_the_most_used(docs):
    p = pool_of(docs, filler_pool_max=3)
    key = pool_key("doc", "th", "first")
    for x in "abc":
        p.add(key, x)
    p._lines(key)["b"] = 9
    p.add(key, "d")
    assert p.size(key) == 3 and not p.has(key, "b") and p.has(key, "d")
    assert (
        FixedClockSettings().filler_pool_max == 200 and FixedClockSettings().filler_pool_min == 20
    )


async def test_writer_generates_until_the_pool_is_big_then_reuses(monkeypatch, docs):
    seen = scripted(monkeypatch, *[js(f"ขอดูให้นะคะ แบบที่ {c}", True) for c in "abcdef"])
    s = FixedClockSettings(filler_pool_min=3, filler_fresh_ratio=0.0)
    w = FillerWriter(object(), s, FillerPool(docs, s, rng=random.Random(1)))
    kw = dict(kind="slow", question="เรื่องนี้", lang="th", particle="ค่ะ")
    first = [await w.write_ex(**kw) for _ in range(3)]  # pool below 3: three model calls
    assert len(seen) == 3 and all(x.generic for x in first)
    again = [await w.write_ex(**kw) for _ in range(5)]  # pool full enough: no model call
    assert len(seen) == 3 and all(x.generic and x.text.startswith("ขอดูให้นะคะ") for x in again)
    # a non-generic line is never saved
    scripted(monkeypatch, js("ขอเช็กยอดเดือนก.ย.ให้นะคะ", False))
    w2 = FillerWriter(object(), s, FillerPool(docs, s))
    assert w2.pool.size(pool_key("link", "th", "first")) == 0
    got = await w2.write_ex(kind="link", question="x", lang="th", particle="ค่ะ")
    assert got and not got.generic and w2.pool.size(pool_key("link", "th", "first")) == 0


# ------------------------------------------------------------------ clip cache on disk
def test_clip_cache_miss_then_hit_and_key_parts(tmp_path):
    c = ClipCache(tmp_path / "c", "Leda", 1.06)
    assert c.get("th", "ขอดูนะคะ") is None
    c.put("th", "ขอดูนะคะ", b"\x01\x02")
    assert c.get("th", "ขอดูนะคะ") == b"\x01\x02"
    assert c.get("en", "ขอดูนะคะ") is None
    assert ClipCache(tmp_path / "c", "Other", 1.06).get("th", "ขอดูนะคะ") is None
    assert ClipCache(tmp_path / "c", "Leda", 1.0).get("th", "ขอดูนะคะ") is None
    assert len(c.path("th", "x").stem) == 64  # sha256


def test_cache_dir_defaults(tmp_path):
    assert fillers.cache_dir(FixedClockSettings(filler_cache_dir="/x/y")) == "/x/y"
    assert fillers.cache_dir(
        FixedClockSettings(filler_cache_dir="", store="sqlite", db_path=str(tmp_path / "l.db"))
    ) == str(tmp_path / "fillers")
    assert fillers.cache_dir(FixedClockSettings(filler_cache_dir="", store="firestore")).endswith(
        "salid-fillers"
    )
