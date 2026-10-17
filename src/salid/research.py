"""`deep_research`: a background, multi-step web research job.

plan (cheap model: 3-6 sub-questions) -> `web_search_call` for each (parallel, capped) ->
synthesis (the main model: summary, key findings with citations, a table when useful, open
questions) -> a sources list appended by code from the real search results (never written by the
model) -> a PDF. The pure pipeline is here; the Discord glue (ack, background task, delivery,
cancel, daily cap) is in `research_bot.py`.

Data isolation is the same as `web_search`: only the question and the planned sub-questions go to
Google. Internal context (optional, already scoped to the asker) goes only to the synthesis call,
never to a search.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime

from google.genai import types

from . import llm
from .i18n import base_lang, lang_name, t
from .pdf_report import FontSet, _finish, _Pdf
from .web_search import format_sources, web_search_call

logger = logging.getLogger(__name__)
DEPTH_QUESTIONS = {"quick": 3, "thorough": 6}
WORKERS = 3
QUESTION_CHARS = 600
FINDING_CHARS = 3500  # of each sub-answer the synthesis sees
INTERNAL_CHARS = 3000
SHORT_CHARS = 1500  # the part of the report posted as a message


class ResearchCancelled(Exception):
    pass


class ResearchHandle:
    """Lets another task stop the job between steps (the model calls themselves are short)."""

    def __init__(self) -> None:
        self._stop = threading.Event()

    def cancel(self) -> None:
        self._stop.set()

    @property
    def cancelled(self) -> bool:
        return self._stop.is_set()

    def check(self) -> None:
        if self._stop.is_set():
            raise ResearchCancelled


@dataclass
class ResearchResult:
    report: str  # markdown, sources list included
    title: str
    sub_questions: list[str]
    sources: list[dict]
    failed: int = 0
    seconds: float = 0.0
    findings: list[dict] = field(default_factory=list)


def clean_depth(depth: str | None) -> str:
    return depth if depth in DEPTH_QUESTIONS else "quick"


# ------------------------------------------------------------------ plan
def _json_call(client, model: str, prompt: str, max_tokens: int = 600):
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json", temperature=0.2, max_output_tokens=max_tokens
    )
    resp = llm.generate(
        client, model, [types.Content(role="user", parts=[types.Part(text=prompt)])], cfg
    )
    return json.loads(getattr(resp, "text", None) or "{}")


def plan_questions(client, model: str, question: str, depth: str, today: date) -> list[str]:
    """3-6 self-contained web search questions (quick=3, thorough=6). Falls back to the question
    itself when the planner fails."""
    n = DEPTH_QUESTIONS[clean_depth(depth)]
    question = " ".join((question or "").split())[:QUESTION_CHARS]
    prompt = (
        f"Today is {today.isoformat()}. Plan a web research. Split the research question into "
        f"exactly {n} different, self-contained web search questions that together cover it "
        "(background/definitions, current facts and figures, comparisons or alternatives, risks "
        "and recent developments as relevant). Each must make sense alone, keep the language of "
        'the research question and contain no private data. Reply JSON: {"questions": ["..."]}\n'
        f"Research question: {question}"
    )
    try:
        data = _json_call(client, model, prompt)
        items = data.get("questions") if isinstance(data, dict) else None
    except Exception as e:  # never log the question
        logger.warning("research plan failed (%s)", type(e).__name__)
        items = None
    qs = [" ".join(str(x).split())[:300] for x in items or [] if isinstance(x, str) and x.strip()]
    qs = list(dict.fromkeys(qs))[:6]
    return qs if len(qs) >= 1 else [question]


# ------------------------------------------------------------------ search + merge
def _renumber(text: str, mapping: dict[int, int]) -> str:
    return re.sub(
        r"\[(\d{1,2})\]",
        lambda m: f"[{mapping[int(m.group(1))]}]" if int(m.group(1)) in mapping else "",
        text,
    )


def merge_findings(results: list[tuple[str, dict]]) -> tuple[list[dict], list[dict]]:
    """([{question, answer}] with GLOBAL citation numbers, global sources deduped by link)."""
    sources: list[dict] = []
    by_link: dict[str, int] = {}
    findings: list[dict] = []
    for question, res in results:
        if res.get("error") or not res.get("answer"):
            continue
        mapping: dict[int, int] = {}
        for s in res.get("sources") or []:
            n = by_link.get(s["link"])
            if n is None:
                n = len(sources) + 1
                by_link[s["link"]] = n
                sources.append({**s, "n": n})
            mapping[s["n"]] = n
        findings.append({"question": question, "answer": _renumber(res["answer"], mapping)})
    return findings, sources


# ------------------------------------------------------------------ synthesis
def synth_prompt(
    question: str,
    findings: list[dict],
    sources: list[dict],
    internal: str,
    lang: str,
    today: date,
    depth: str,
) -> str:
    heads = {
        "th": ("สรุป", "ประเด็นสำคัญ", "ตารางเปรียบเทียบ", "คำถามที่ยังไม่มีคำตอบ"),
        "en": ("Summary", "Key findings", "Comparison table", "Open questions"),
    }[base_lang(lang)]
    body = "\n\n".join(
        f"### Sub-question {i}: {f['question']}\n{f['answer'][:FINDING_CHARS]}"
        for i, f in enumerate(findings, 1)
    )
    ctx = (
        f"\n\nInternal company context (from the team's own data; cite as (internal), never as "
        f"[n]; do not repeat private details beyond what the question needs):\n{internal[:INTERNAL_CHARS]}"
        if internal
        else ""
    )
    language = "Thai (use ค่ะ/คะ sparingly, formal report tone)" if lang == "th" else lang_name(lang)
    return (
        f"Today is {today.isoformat()}. Write a research report in {language} answering the research "
        f"question, using ONLY the findings below ({depth} depth).\n"
        f"Structure in Markdown: '## {heads[0]}' (3-5 sentences), '## {heads[1]}' (bullets, each "
        f"with [n] citations), '## {heads[2]}' ONLY if comparing options or figures helps (a "
        f"Markdown table), '## {heads[3]}' (what could not be established; bullets).\n"
        "Rules: cite sources as [n] using only the numbers in the source list; every figure or "
        "claim from the web needs a [n]; never invent facts, numbers or sources; if findings "
        "disagree or are weak or old, say so; do not write a sources section (it is appended).\n"
        f"\nResearch question: {question}\n\nSource list (numbers you may cite): "
        f"{[s['n'] for s in sources]}\n\nFindings:\n{body}{ctx}"
    )


def strip_bad_citations(report: str, valid: set[int]) -> str:
    return re.sub(
        r"\[(\d{1,2})\]", lambda m: m.group(0) if int(m.group(1)) in valid else "", report
    )


def synthesize(client, model: str, prompt: str) -> str:
    cfg = types.GenerateContentConfig(temperature=0.3, max_output_tokens=8192)
    resp = llm.generate(
        client, model, [types.Content(role="user", parts=[types.Part(text=prompt)])], cfg
    )
    return (getattr(resp, "text", None) or "").strip()


def cited_numbers(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d{1,2})\]", text)}


def fallback_report(question: str, findings: list[dict], lang: str) -> str:
    """When the synthesis call fails: the raw findings, honestly labelled."""
    head = t("research_raw_head", lang)
    parts = [f"## {head}"] + [f"### {f['question']}\n{f['answer']}" for f in findings]
    return "\n\n".join(parts)


def run_research(
    client,
    *,
    question: str,
    depth: str = "quick",
    lang: str = "th",
    today: date,
    search_model: str,
    plan_model: str,
    synth_model: str,
    internal: Callable[[], str] | None = None,
    handle: ResearchHandle | None = None,
    search: Callable = web_search_call,
    clock: Callable[[], float] | None = None,
) -> ResearchResult:
    """Blocking (run it in a worker thread). Raises ResearchCancelled when `handle` is cancelled."""
    import time

    clock = clock or time.monotonic
    started = clock()
    handle = handle or ResearchHandle()
    depth = clean_depth(depth)
    question = " ".join((question or "").split())[:QUESTION_CHARS]
    handle.check()
    subs = plan_questions(client, plan_model, question, depth, today)
    handle.check()

    def one(q: str) -> tuple[str, dict]:
        if handle.cancelled:
            return q, {"error": "cancelled"}
        return q, search(client, search_model, q, today=today)

    with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="research") as pool:
        results = list(pool.map(one, subs))
    handle.check()
    findings, sources = merge_findings(results)
    failed = len(results) - len(findings)
    ctx = ""
    if internal is not None:
        try:
            ctx = internal() or ""
        except Exception as e:
            logger.warning("research internal context failed (%s)", type(e).__name__)
    handle.check()
    if not findings:
        body = t("research_nothing", lang)
    else:
        try:
            body = synthesize(
                client,
                synth_model,
                synth_prompt(question, findings, sources, ctx, lang, today, depth),
            )
        except Exception as e:
            logger.warning("research synthesis failed (%s)", type(e).__name__)
            body = ""
        if not body:
            body = fallback_report(question, findings, lang)
    handle.check()
    body = strip_bad_citations(body, {s["n"] for s in sources})
    used = [s for s in sources if s["n"] in cited_numbers(body)] or sources
    report = body
    if used:
        report += f"\n\n## {t('research_sources', lang)}\n" + format_sources(used)
    title = question if len(question) <= 80 else question[:79] + "…"
    return ResearchResult(
        report=report,
        title=title,
        sub_questions=subs,
        sources=used,
        failed=failed,
        seconds=clock() - started,
        findings=findings,
    )


# ------------------------------------------------------------------ delivery text and PDF
def short_version(report: str, lang: str, limit: int = SHORT_CHARS) -> str:
    """The first section(s) of the report that fit the message limit (cut at a line)."""
    lines, out, size = report.splitlines(), [], 0
    for line in lines:
        if line.startswith("## ") and out and size > 300 and size + len(line) > limit:
            break
        if size + len(line) + 1 > limit:
            break
        out.append(line)
        size += len(line) + 1
    text = "\n".join(out).strip()
    return text if text else report[:limit]


_MD_NOISE = re.compile(r"(\*\*|__|`)")


def _plain(s: str) -> str:
    return _MD_NOISE.sub("", s).strip()


def markdown_pdf(
    *, org: str, title: str, body: str, lang: str, generated_at: datetime, fonts: FontSet
) -> bytes:
    """A small Markdown subset (## headings, bullets, tables, paragraphs) as a PDF."""
    pdf = _Pdf(fonts, org, lang, generated_at)
    pdf.add_page()
    pdf.h1(title or t("research_title", lang))
    pdf.p(t("research_dated", lang, at=f"{generated_at:%Y-%m-%d}"), size=9)
    table: list[list[str]] = []

    def flush() -> None:
        if not table:
            return
        cols = max(len(r) for r in table)
        rows = [r + [""] * (cols - len(r)) for r in table]
        w = 182.0 / cols
        pdf.ln(1)
        pdf.table(rows[0], rows[1:], [w] * cols)
        pdf.ln(1)
        table.clear()

    for raw in body.splitlines():
        line = raw.rstrip()
        if line.lstrip().startswith("|"):
            cells = [_plain(c) for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in cells if c.strip()):
                continue  # the |---|---| separator
            table.append(cells)
            continue
        flush()
        if not line.strip():
            continue
        if line.startswith("#"):
            pdf.h2(_plain(line.lstrip("# ")))
        elif re.match(r"\s*[-*•]\s+", line):
            pdf.p("•  " + _plain(re.sub(r"^\s*[-*•]\s+", "", line)))
        else:
            pdf.p(_plain(line))
    flush()
    return _finish(pdf)
