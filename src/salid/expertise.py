"""Expertise locator: who in the company can do X, with the reasons, and who is the backup.

Used for "who knows our tax filings?", "who can take over Jems' tasks while he is away?",
"who should fill the vacant Sales Lead position?". Pure logic (no discord, no FastAPI):

1. `Candidate` evidence is gathered by the caller from what Salid already keeps: the member's
   profile (title, department, what they do), the positions they hold, the people directory
   (what Salid OBSERVED from their messages: role, responsibilities, works with; active
   channels), and the titles of tasks they finished / have open.
2. `rank` makes ONE model call that must ground every recommendation in that evidence: it returns
   a primary pick and backups, each with a short reason and the evidence lines it used. Code then
   drops people who are not candidates (no invented names) and evidence that is not in the
   candidate's own lines.

The need may name a position: its job description, responsibilities and skills are added to
the request so the match is against what the job actually asks for.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from .i18n import lang_name

MAX_CANDIDATES = 60
MAX_LINES = 14  # evidence lines per candidate in the prompt


class Candidate(BaseModel):
    key: str  # email for members, "discord:<id>" for people without an account
    name: str
    has_account: bool = True
    evidence: list[str] = Field(default_factory=list)  # short facts, each one line


class Pick(BaseModel):
    key: str
    rank: str = "primary"  # primary | backup
    reason: str = ""
    evidence: list[str] = Field(default_factory=list)
    confidence: str = "medium"  # high | medium | low


class Ranking(BaseModel):
    need: str = ""
    picks: list[Pick] = Field(default_factory=list)
    gaps: str = ""  # what nobody seems to cover ("" = none)


class _ModelPick(BaseModel):
    key: str
    rank: str
    reason: str
    evidence: list[str] = Field(default_factory=list)
    confidence: str = "medium"


class _ModelOut(BaseModel):
    picks: list[_ModelPick] = Field(default_factory=list)
    gaps: str = ""


def candidate(
    key: str,
    name: str,
    *,
    has_account: bool = True,
    title: str = "",
    department: str = "",
    about: str = "",
    positions: list[str] | None = None,
    observed: dict | None = None,
    channels: list[str] | None = None,
    tasks_done: list[str] | None = None,
    tasks_open: list[str] | None = None,
) -> Candidate:
    """One candidate's evidence lines (only non-empty facts)."""
    ev: list[str] = []
    if title or department:
        ev.append("Title: " + ", ".join(x for x in (title, department) if x))
    for p in positions or []:
        ev.append(f"Holds position: {p}")
    if about:
        ev.append(f"Says they do: {about[:200]}")
    o = observed or {}
    if o.get("role"):
        ev.append(f"Salid observed role: {o['role'][:200]}")
    if o.get("responsibilities"):
        ev.append(f"Salid observed handles: {o['responsibilities'][:200]}")
    if o.get("works_with"):
        ev.append("Works with: " + ", ".join(o["works_with"][:6]))
    if channels:
        ev.append("Active in: " + ", ".join("#" + c for c in channels[:5]))
    for t in (tasks_done or [])[:5]:
        ev.append(f"Finished task: {t[:120]}")
    for t in (tasks_open or [])[:3]:
        ev.append(f"Working on: {t[:120]}")
    return Candidate(key=key, name=name, has_account=has_account, evidence=ev[:MAX_LINES])


def build_prompt(need: str, candidates: list[Candidate], lang: str, position: dict | None) -> str:
    job = ""
    if position:
        job = (
            "\nThe need is about this position:\n"
            f"Title: {position.get('title', '')}\n"
            + (f"Department: {position['department']}\n" if position.get("department") else "")
            + (f"Job description: {position['summary']}\n" if position.get("summary") else "")
            + (
                f"Responsibilities: {position['responsibilities']}\n"
                if position.get("responsibilities")
                else ""
            )
            + (f"Skills: {', '.join(position['skills'])}\n" if position.get("skills") else "")
        )
    people = "\n\n".join(
        f"[{c.key}] {c.name}"
        + ("" if c.has_account else " (no Salid account yet)")
        + "\n- "
        + ("\n- ".join(c.evidence) if c.evidence else "(no information)")
        for c in candidates
    )
    return (
        "You are Salid, the company's assistant, acting as an expertise locator. Recommend the "
        "people best suited for the need below: one primary pick and up to two backups (people "
        "who could take over or cover). Use ONLY the evidence listed under each person; never "
        "invent skills or experience. Each pick: key (exactly as in brackets), rank (primary or "
        "backup), reason (one or two sentences in "
        + lang_name(lang)
        + "), evidence (the evidence lines you relied on, copied verbatim), confidence (high / "
        "medium / low: low when the evidence is thin). If nobody fits, return no picks and say "
        "what is missing in gaps (in " + lang_name(lang) + "). Return JSON only.\n\n"
        f"Need: {need}\n{job}\nPeople:\n{people}"
    )


def rank(
    need: str,
    candidates: list[Candidate],
    *,
    model_call: Any,
    lang: str = "en",
    position: dict | None = None,
) -> Ranking:
    """One model call (`model_call(prompt) -> dict`), then code checks every pick."""
    cands = [c for c in candidates if c.evidence][:MAX_CANDIDATES]
    if not cands:
        return Ranking(need=need, gaps="no information about anyone yet")
    raw = model_call(build_prompt(need, cands, lang, position)) or {}
    out = _ModelOut.model_validate(raw)
    by_key = {c.key: c for c in cands}
    picks: list[Pick] = []
    seen: set[str] = set()
    for p in out.picks:
        c = by_key.get(p.key)
        if c is None or p.key in seen:
            continue  # not a candidate: never recommend an invented person
        seen.add(p.key)
        ev = [e for e in p.evidence if e in c.evidence][:5] or c.evidence[:2]
        rank_ = (
            "primary"
            if p.rank == "primary" and not any(x.rank == "primary" for x in picks)
            else "backup"
        )
        conf = p.confidence if p.confidence in ("high", "medium", "low") else "medium"
        picks.append(
            Pick(key=p.key, rank=rank_, reason=p.reason.strip()[:400], evidence=ev, confidence=conf)
        )
        if len(picks) >= 3:
            break
    if picks and not any(x.rank == "primary" for x in picks):
        picks[0].rank = "primary"
    picks.sort(key=lambda x: x.rank != "primary")
    return Ranking(need=need, picks=picks, gaps=out.gaps.strip()[:400])


def make_model_call(client: Any, model: str):
    """(prompt) -> dict, through Gemini in JSON mode."""
    from google.genai import types

    from .llm import generate

    def run(prompt: str) -> dict:
        resp = generate(
            client,
            model,
            prompt,
            types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=3000,
                response_mime_type="application/json",
                response_schema=_ModelOut,
            ),
        )
        return json.loads(resp.text or "{}")

    return run
