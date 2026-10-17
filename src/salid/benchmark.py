"""`market_benchmark`: grounded web research into a market -> gaps, a competitor matrix and where
"our product" wins / loses. Reuses `web_search_call` (Google-grounded, numbered citations) and
`research.merge_findings`; ONE synthesis call returns JSON that code validates:

* every competitor row and every market gap must cite at least one REAL source number from the
  searches, rows without a valid source are dropped (never shown as fact);
* the report and the sources list are rendered by code, so the model cannot invent a link;
* "our product" defaults to a short factual description of Salid (`SALID_FACTS`, features that
  exist in this repo), never invented by the model.

Only the topic (and the product description the user gave) goes to Google, no team data.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date

from .i18n import lang_name as _lang_name
from .research import _json_call, merge_findings
from .web_search import format_sources, web_search_call

logger = logging.getLogger(__name__)

TOPIC_CHARS = 300
PRODUCT_CHARS = 800
FINDING_CHARS = 3500
MAX_COMPETITORS = 8
MAX_GAPS = 8

# What Salid really does today (see CLAUDE.md). Facts only; keep it short and checkable.
SALID_FACTS = (
    "Salid (สลิด) is an AI teammate that lives in a company's Discord. Real features: "
    "(1) reads receipts, slips and invoices from photos/PDFs into an expense ledger with duplicate "
    "detection and Thai/English Q&A over it; (2) joins Discord voice meetings, transcribes Thai "
    "speech, and posts minutes with action items; the same assistant also talks live by voice; "
    "(3) team memory of channels with per-channel access rules (people only see what they could "
    "read), search across chat, files, wiki, tasks and meetings; (4) coordinator role: tracks "
    "tasks and follow-ups, morning/evening briefs, reports, finds a time everyone is free, "
    "schedules meetings, polls and forms, approvals with Approve/Decline buttons; "
    "(5) translation and web research with citations, stock/gold/news lookups; "
    "(6) idea workspace with business frameworks (BMC, SWOT...) exported as PDF/DOCX; "
    "(7) works in Thai and English, multi-company (one bot, isolated per Discord server), "
    "command center web page for settings. Limits: Discord-first (no Zoom/Google Meet bot), "
    "no native mobile app, no on-prem install."
)

QUERIES = (
    "{topic}: leading products and competitors with their key features and pricing (include "
    "plan prices in USD or THB), current as of {year}",
    "{topic}: support for Thai language and Thai-market competitors or local alternatives",
    "{topic}: common user complaints, unmet needs, limitations and gaps in the market; market size",
)


@dataclass
class Benchmark:
    status: str  # ok | no_results
    markdown: str = ""
    title: str = ""
    competitors: list[dict] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    dropped: int = 0  # rows removed because they cited no real source


def clean_topic(topic: str) -> str:
    return " ".join((topic or "").split())[:TOPIC_CHARS]


def _synth_prompt(
    topic: str, ours: str, findings: list[dict], sources: list[dict], lang: str, today: date
) -> str:
    lang_name = _lang_name(lang)
    blocks = "\n\n".join(f"### {f['question']}\n{f['answer'][:FINDING_CHARS]}" for f in findings)
    src = format_sources(sources)
    return (
        f"Today is {today.isoformat()}. You compare products in a market using ONLY the research "
        f"notes below. Write all text values in {lang_name}.\n"
        f"MARKET / TOPIC: {topic}\n\nOUR PRODUCT (use only what this says about it):\n{ours}\n\n"
        f"RESEARCH NOTES (citations like [3] refer to SOURCES):\n{blocks}\n\nSOURCES:\n{src}\n\n"
        "Reply JSON only:\n"
        '{"summary": "3-5 sentences",\n'
        ' "gaps": [{"gap": "unmet need or weakness across the market", "sources": [1]}],\n'
        ' "competitors": [{"name": "", "pricing": "plan prices exactly as found, or unknown", '
        '"strengths": "", "weaknesses": "", "thai_support": "yes|partial|no|unknown", '
        '"sources": [1]}],\n'
        ' "ours_wins": [{"point": "where OUR PRODUCT is stronger, grounded in its description", '
        '"vs": "competitor name or market"}],\n'
        ' "ours_loses": [{"point": "where a competitor is stronger or we have no such feature", '
        '"vs": "competitor name", "sources": [1]}],\n'
        ' "next_moves": ["short concrete suggestion"]}\n'
        "Rules: every competitor and every gap needs `sources` (numbers from SOURCES that really "
        f"support it); at most {MAX_COMPETITORS} competitors, {MAX_GAPS} gaps; never invent prices, "
        "numbers, companies or features; write `unknown` when the notes do not say; do not "
        "praise OUR PRODUCT beyond its description; be honest about where it loses."
    )


def _nums(v, valid: set[int]) -> list[int]:
    out = []
    for x in v if isinstance(v, list) else []:
        try:
            n = int(x)
        except (TypeError, ValueError):
            continue
        if n in valid and n not in out:
            out.append(n)
    return out


def _s(v, cap: int = 300) -> str:
    return " ".join(str(v or "").split())[:cap]


def validate(data: dict, valid: set[int]) -> tuple[dict, int]:
    """Keep only rows that cite real sources. -> (clean dict, number of dropped rows)."""
    dropped = 0
    comps: list[dict] = []
    for c in (data.get("competitors") or [])[: MAX_COMPETITORS * 2]:
        if not isinstance(c, dict):
            continue
        src = _nums(c.get("sources"), valid)
        name = _s(c.get("name"), 80)
        if not name or not src:
            dropped += 1
            continue
        thai = _s(c.get("thai_support"), 10).lower()
        comps.append(
            {
                "name": name,
                "pricing": _s(c.get("pricing")) or "unknown",
                "strengths": _s(c.get("strengths")),
                "weaknesses": _s(c.get("weaknesses")),
                "thai_support": thai if thai in ("yes", "partial", "no") else "unknown",
                "sources": src,
            }
        )
    gaps: list[dict] = []
    for g in (data.get("gaps") or [])[: MAX_GAPS * 2]:
        if not isinstance(g, dict):
            continue
        src = _nums(g.get("sources"), valid)
        text = _s(g.get("gap"), 400)
        if not text or not src:
            dropped += 1
            continue
        gaps.append({"gap": text, "sources": src})
    wins = [
        {"point": _s(w.get("point"), 400), "vs": _s(w.get("vs"), 80)}
        for w in data.get("ours_wins") or []
        if isinstance(w, dict) and _s(w.get("point"))
    ][:6]
    loses = []
    for w in data.get("ours_loses") or []:
        if isinstance(w, dict) and _s(w.get("point")):
            loses.append(
                {
                    "point": _s(w.get("point"), 400),
                    "vs": _s(w.get("vs"), 80),
                    "sources": _nums(w.get("sources"), valid),
                }
            )
    return {
        "summary": _s(data.get("summary"), 900),
        "competitors": comps[:MAX_COMPETITORS],
        "gaps": gaps[:MAX_GAPS],
        "ours_wins": wins,
        "ours_loses": loses[:6],
        "next_moves": [_s(x, 200) for x in (data.get("next_moves") or []) if _s(x)][:5],
    }, dropped


def _refs(nums: list[int]) -> str:
    return "".join(f"[{n}]" for n in nums)


THAI = {"yes": "รองรับ", "partial": "บางส่วน", "no": "ไม่รองรับ", "unknown": "ไม่ทราบ"}
H = {
    "th": {
        "title": "เทียบตลาด",
        "summary": "สรุป",
        "gaps": "ช่องว่างของตลาด",
        "matrix": "ตารางเทียบคู่แข่ง",
        "cols": ("ผลิตภัณฑ์", "ราคา", "จุดแข็ง", "จุดอ่อน", "ภาษาไทย", "อ้างอิง"),
        "ours": "เทียบกับผลิตภัณฑ์ของเรา",
        "wins": "เราเหนือกว่า",
        "loses": "เรายังเป็นรอง",
        "next": "ก้าวต่อไป",
        "sources": "แหล่งอ้างอิง",
        "note": "ผลเทียบเป็นการประเมินของสลิดจากข้อมูลเว็บที่ค้นเจอ ณ วันที่ {d} ราคา/ฟีเจอร์อาจเปลี่ยน ควรตรวจหน้าเว็บทางการอีกครั้ง",
        "dropped": "ตัดข้อมูล {n} รายการที่ไม่มีแหล่งอ้างอิงออกแล้ว",
        "none": "ไม่พบข้อมูลที่มีแหล่งอ้างอิงเพียงพอ",
    },
    "en": {
        "title": "Market benchmark",
        "summary": "Summary",
        "gaps": "Market gaps",
        "matrix": "Competitor matrix",
        "cols": ("Product", "Pricing", "Strengths", "Weaknesses", "Thai", "Sources"),
        "ours": "Versus our product",
        "wins": "Where we win",
        "loses": "Where we lose",
        "next": "Next moves",
        "sources": "Sources",
        "note": "Salid's estimate from web results found on {d}; prices and features change, check the official pages.",
        "dropped": "{n} row(s) without a source were removed.",
        "none": "Not enough sourced information was found.",
    },
}


def render(
    topic: str, clean: dict, sources: list[dict], dropped: int, lang: str, today: date
) -> str:
    h = H["en" if lang != "th" else "th"]
    cell = lambda s: (s or "-").replace("|", "/").replace("\n", " ")  # noqa: E731
    L = [f"# {h['title']}: {topic}", ""]
    if clean["summary"]:
        L += [f"## {h['summary']}", clean["summary"], ""]
    if clean["gaps"]:
        L += (
            [f"## {h['gaps']}"]
            + [f"- {g['gap']} {_refs(g['sources'])}" for g in clean["gaps"]]
            + [""]
        )
    if clean["competitors"]:
        L += [
            f"## {h['matrix']}",
            "| " + " | ".join(h["cols"]) + " |",
            "|" + "---|" * len(h["cols"]),
        ]
        for c in clean["competitors"]:
            thai = THAI[c["thai_support"]] if lang == "th" else c["thai_support"]
            L.append(
                f"| {cell(c['name'])} | {cell(c['pricing'])} | {cell(c['strengths'])} | "
                f"{cell(c['weaknesses'])} | {thai} | {_refs(c['sources'])} |"
            )
        L.append("")
    if clean["ours_wins"] or clean["ours_loses"]:
        L.append(f"## {h['ours']}")
        if clean["ours_wins"]:
            L += [f"**{h['wins']}**"] + [
                f"- {w['point']}" + (f" ({w['vs']})" if w["vs"] else "") for w in clean["ours_wins"]
            ]
        if clean["ours_loses"]:
            L += [f"**{h['loses']}**"] + [
                f"- {w['point']}" + (f" ({w['vs']})" if w["vs"] else "") + f" {_refs(w['sources'])}"
                for w in clean["ours_loses"]
            ]
        L.append("")
    if clean["next_moves"]:
        L += [f"## {h['next']}"] + [f"- {m}" for m in clean["next_moves"]] + [""]
    if not (clean["competitors"] or clean["gaps"]):
        L += [h["none"], ""]
    used = (
        {n for c in clean["competitors"] for n in c["sources"]}
        | {n for g in clean["gaps"] for n in g["sources"]}
        | {n for w in clean["ours_loses"] for n in w["sources"]}
    )
    shown = [s for s in sources if s["n"] in used] or sources
    if shown:
        L += [f"## {h['sources']}", format_sources(shown), ""]
    note = h["note"].format(d=today.isoformat())
    if dropped:
        note += " " + h["dropped"].format(n=dropped)
    L.append(f"_{note}_")
    return "\n".join(L)


def run_benchmark(
    client,
    *,
    topic: str,
    our_product: str = "",
    lang: str = "en",
    today: date,
    search_model: str,
    synth_model: str,
    search: Callable = web_search_call,
) -> Benchmark:
    """Blocking (run it in a worker thread). Never raises for model/search failures."""
    topic = clean_topic(topic)
    ours = " ".join((our_product or "").split())[:PRODUCT_CHARS] or SALID_FACTS
    queries = [q.format(topic=topic, year=today.year) for q in QUERIES]

    def one(q: str) -> tuple[str, dict]:
        return q, search(client, search_model, q, today=today)

    with ThreadPoolExecutor(max_workers=len(queries), thread_name_prefix="benchmark") as pool:
        results = list(pool.map(one, queries))
    findings, sources = merge_findings(results)
    if not findings or not sources:
        return Benchmark("no_results", title=topic)
    try:
        data = _json_call(
            client,
            synth_model,
            _synth_prompt(topic, ours, findings, sources, lang, today),
            max_tokens=6000,
        )
        if not isinstance(data, dict):
            data = {}
    except Exception as e:  # never log the topic
        logger.warning("benchmark synthesis failed (%s)", type(e).__name__)
        data = {}
    clean, dropped = validate(data, {s["n"] for s in sources})
    if not (clean["competitors"] or clean["gaps"]):
        return Benchmark("no_results", title=topic, sources=sources, dropped=dropped)
    md = render(topic, clean, sources, dropped, lang, today)
    return Benchmark(
        "ok",
        markdown=md,
        title=topic,
        competitors=clean["competitors"],
        gaps=clean["gaps"],
        sources=sources,
        dropped=dropped,
    )


def has_markup(md: str) -> bool:
    return bool(re.search(r"^#{1,3} ", md, re.M))
