"""Meeting votes: validation, the stored record, ballots, tally and the one-line result.

Salid sends each voter a button ballot in their private channel, counts the presses, closes the vote
(everyone voted / the deadline / on request) and posts the result. State is one document per vote
in the `DocStore` (`salid_votes`, SQLite or Firestore) and one per offered vote (`salid_vote_offers`).
Discord glue is `salid.votes_bot`. Pure: no Discord, no clock of its own.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel

from .coo_store import DocStore
from .i18n import t

VOTES = "salid_votes"
OFFERS = "salid_vote_offers"
DEFAULT_OPTIONS = {
    "th": ["เห็นด้วย", "ไม่เห็นด้วย", "งดออกเสียง"],
    "en": ["Agree", "Disagree", "Abstain"],
}
MIN_OPTIONS, MAX_OPTIONS = 2, 6
MAX_OPTION_CHARS = 40
MAX_QUESTION_CHARS = 300
MIN_MINUTES, MAX_MINUTES, DEFAULT_MINUTES = 1, 60, 5
MAX_VOTERS = 50
MAX_OPEN_PER_GUILD = 5


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def clean_vote(question: str, options, minutes, lang: str) -> tuple[str, list[str], int] | dict:
    """(question, options, whole minutes) within the limits, or an error dict. No options = the
    agree / disagree / abstain set in `lang`."""
    q = " ".join((question or "").split())
    if not q:
        return {"error": "need_question"}
    seen: set[str] = set()
    opts: list[str] = []
    for o in options or []:
        o = " ".join(str(o).split())
        if o and o.casefold() not in seen:
            seen.add(o.casefold())
            opts.append(o)
    if not opts:
        opts = list(DEFAULT_OPTIONS["en" if lang != "th" else "th"])
    if len(opts) < MIN_OPTIONS:
        return {"error": "need_at_least_2_options"}
    if len(opts) > MAX_OPTIONS:
        return {"error": "too_many_options", "max": MAX_OPTIONS}
    if len(q) > MAX_QUESTION_CHARS or any(len(o) > MAX_OPTION_CHARS for o in opts):
        return {
            "error": "too_long",
            "max_question_chars": MAX_QUESTION_CHARS,
            "max_option_chars": MAX_OPTION_CHARS,
        }
    try:
        m = int(round(float(minutes)))
    except (TypeError, ValueError):
        m = DEFAULT_MINUTES
    return q, opts, min(max(m, MIN_MINUTES), MAX_MINUTES)


def is_yes_no(options: list[str]) -> bool:
    """The default agree / disagree / abstain set (either language): pass / fail wording applies."""
    return options in (DEFAULT_OPTIONS["th"], DEFAULT_OPTIONS["en"])


def new_vote(
    *,
    guild_id: int,
    channel_id: int,
    room: str,
    question: str,
    options: list[str],
    voters: list[int],
    opened_by: int,
    opener_name: str = "",
    minutes: int = DEFAULT_MINUTES,
    anonymous: bool = False,
    lang: str = "th",
    voice_channel_id: int = 0,
    now: datetime,
) -> dict:
    return {
        "id": uuid.uuid4().hex[:12],
        "guild_id": guild_id,
        "channel_id": channel_id,  # the text channel that gets the tally and the result
        "voice_channel_id": voice_channel_id,
        "room": room,
        "question": question,
        "options": options,
        "voters": list(dict.fromkeys(voters)),
        "anonymous": bool(anonymous),
        "ballots": {},  # str(user id) -> {"option": index, "at": iso}
        "delivered": {},  # str(user id) -> {"channel_id", "message_id"}
        "unreached": [],
        "opened_by": opened_by,
        "opener_name": opener_name,
        "opened_at": iso(now),
        "closes_at": iso(now + timedelta(minutes=minutes)),
        "status": "open",
        "closed_at": "",
        "tally_message_id": 0,
        "result_message_id": 0,
        "lang": lang,
    }


def is_open(vote: dict) -> bool:
    return vote.get("status") == "open"


def is_due(vote: dict, now: datetime) -> bool:
    return is_open(vote) and datetime.fromisoformat(vote["closes_at"]) <= now


def cast(vote: dict, user_id: int, index: int, now: datetime) -> str:
    """Record or change a ballot: "ok", "closed", "not_voter" or "bad_option"."""
    if not is_open(vote):
        return "closed"
    if user_id not in vote["voters"]:
        return "not_voter"
    if not isinstance(index, int) or not 0 <= index < len(vote["options"]):
        return "bad_option"
    vote["ballots"][str(user_id)] = {"option": index, "at": iso(now)}
    return "ok"


def counts(vote: dict) -> list[int]:
    out = [0] * len(vote["options"])
    for b in vote["ballots"].values():
        out[b["option"]] += 1
    return out


def all_voted(vote: dict) -> bool:
    return bool(vote["voters"]) and all(str(u) in vote["ballots"] for u in vote["voters"])


def outcome(vote: dict) -> dict:
    """verdict: pass | fail | tie | none (yes / no set) or win | tie | none (custom options).
    Simple majority of the ballots cast; on the agree / disagree set abstentions count toward the
    turnout but toward neither side; a tie is no conclusion."""
    c = counts(vote)
    total = sum(c)
    if total == 0:
        return {"verdict": "none", "counts": c, "total": 0, "winner": None}
    if is_yes_no(vote["options"]):
        yes, no = c[0], c[1]
        verdict = "pass" if yes > no else "fail" if no > yes else "tie"
        return {"verdict": verdict, "counts": c, "total": total, "winner": None}
    top = max(c)
    leaders = [i for i, n in enumerate(c) if n == top]
    if len(leaders) > 1:
        return {"verdict": "tie", "counts": c, "total": total, "winner": None}
    return {"verdict": "win", "counts": c, "total": total, "winner": leaders[0]}


def _counts_text(vote: dict, sep: str) -> str:
    return sep.join(f"{o} {n}" for o, n in zip(vote["options"], counts(vote), strict=False))


def hhmm(iso_text: str, tz) -> str:
    return datetime.fromisoformat(iso_text).astimezone(tz).strftime("%H:%M")


def tally_text(vote: dict, lang: str, tz) -> str:
    """The meeting-chat message, edited on every ballot. Counts only, never names."""
    key = "vote_tally" if is_open(vote) else "vote_tally_closed"
    return t(
        key,
        lang,
        question=vote["question"],
        done=len(vote["ballots"]),
        total=len(vote["voters"]),
        counts=_counts_text(vote, " · "),
        time=hhmm(vote["closes_at"], tz),
    )


def result_line(vote: dict, lang: str, *, emoji: bool = True) -> str:
    """One line: "✅ มติ: <q> → เห็นด้วย 4 / ไม่เห็นด้วย 1 / งดออกเสียง 0 (ผ่าน, ...)". The rule
    is part of the line."""
    r = outcome(vote)
    v = r["verdict"]
    if v == "win":
        verdict = t("vote_v_win", lang, option=vote["options"][r["winner"]])
    else:
        verdict = t(f"vote_v_{v}", lang)
    icon = {"pass": "✅", "win": "✅", "fail": "❌", "tie": "⚖️", "none": "🗳️"}[v] + " "
    line = t(
        "vote_result",
        lang,
        icon=icon if emoji else "",
        question=vote["question"],
        counts=_counts_text(vote, " / "),
        verdict=verdict,
    )
    if vote["anonymous"]:
        line += " " + t("vote_anon_tag", lang)
    return line


def result_text(vote: dict, names: dict[int, str], lang: str) -> str:
    """The result line plus, unless the vote is anonymous, who chose what and who did not vote.
    Salid never reveals who voted what in an anonymous vote."""
    out = [result_line(vote, lang)]
    if not vote["anonymous"]:
        for i, o in enumerate(vote["options"]):
            who = [
                names.get(int(u), str(u)) for u, b in vote["ballots"].items() if b["option"] == i
            ]
            if who:
                out.append(t("vote_names", lang, option=o, names=", ".join(who)))
        absent = [names.get(u, str(u)) for u in vote["voters"] if str(u) not in vote["ballots"]]
        if absent:
            out.append(t("vote_nonvoters", lang, names=", ".join(absent)))
    return "\n".join(out)


class VoteRepo:
    def __init__(self, docs: DocStore):
        self.docs = docs

    def put(self, rec: dict) -> dict:
        self.docs.put(VOTES, rec["guild_id"], rec["id"], rec)
        return rec

    def get(self, vote_id: str) -> dict | None:
        return self.docs.find_any(VOTES, vote_id)

    def all(self, guild_id: int) -> list[dict]:
        return sorted(self.docs.find(VOTES, guild_id), key=lambda r: r["opened_at"])

    def open_votes(self, guild_id: int) -> list[dict]:
        return [r for r in self.all(guild_id) if is_open(r)]

    def find(self, guild_id: int, ref: str | None, visible=lambda rec: True):
        """(record, candidates): `ref` = vote id, part of the question, or empty / "last" for the
        newest vote `visible` allows (an open one first)."""
        recs = [r for r in self.all(guild_id) if visible(r)]
        if not recs:
            return None, []
        key = (ref or "").strip().casefold()
        if key in ("", "last", "latest", "ล่าสุด"):
            open_ = [r for r in recs if is_open(r)]
            return (open_ or recs)[-1], []
        by_id = [r for r in recs if r["id"] == key]
        if by_id:
            return by_id[0], []
        hits = [r for r in recs if key in r["question"].casefold()]
        if len(hits) == 1:
            return hits[0], []
        return None, (hits or recs)[-5:]


# ------------------------------------------------------------------ offers from untagged talk
OFFER_GAP = timedelta(minutes=10)  # one offer per channel
OFFER_SILENCE = timedelta(hours=1)  # after "ไม่ต้อง"
OFFER_MAX_CHARS = 300

# Conservative on purpose: a decision / vote cue AND a question-like structure.
_CUE = re.compile(
    r"โหวต|ลงมติ|ยกมือ|ใคร(?:เห็นด้วย|ไม่เห็นด้วย|เอา|โอเค)|เอา(?:ไง|ยังไง)ดี"
    r"|(?:เห็นด้วย|ตกลง)\s*(?:กัน)?\s*(?:ไหม|มั้ย|หรือเปล่า)|หรือ.{1,40}ดี"
    r"|\bvote\b|who'?s\s+in\s+favou?r|who\s+agrees|show\s+of\s+hands|all\s+in\s+favou?r"
    r"|do\s+we\s+all\s+agree|which\s+one\s+should\s+we\s+pick",
    re.I,
)
_QUESTION = re.compile(
    r"\?|ไหม|มั้ย|บ้าง|ดี\s*$|ดีนะ|กันเถอะ|\bshall we\b|\bshould we\b|\blet'?s\b"
    r"|^\s*(?:who|which|do|does|can|could)\b",
    re.I,
)


def looks_like_vote_call(text: str) -> bool:
    """Free pre-filter (no LLM): the group seems to be trying to decide something or asks for a
    show of hands. An answer ("เห็นด้วยครับ", "I voted A") is not a call for a vote."""
    s = " ".join((text or "").split())
    if not 5 <= len(s) <= OFFER_MAX_CHARS:
        return False
    return bool(_CUE.search(s) and _QUESTION.search(s))


class _Offer(BaseModel):
    answer: bool
    question: str = ""
    options: list[str] = []


def extract_offer(client, model: str, history: list[str], text: str) -> dict | None:
    """Blocking. The cheap yes/no gate plus the structured {question, options}; None when the
    answer is no or anything fails (fail quiet)."""
    import logging

    from google.genai import types

    from .llm import generate

    prompt = (
        "People are chatting in a team Discord (Thai or English). Recent lines, oldest first:\n"
        + "\n".join(h[:200] for h in history[-6:])
        + f"\n\nNew message:\n{text[:OFFER_MAX_CHARS]}\n\n"
        "Is the group trying to DECIDE something between themselves, or asking for a vote / show "
        "of hands (not a factual question, not an answer to one)? If yes, give the decision as a "
        "short question in the language of the chat and, only if the options are explicit "
        "(e.g. A or B), the options; otherwise leave options empty (agree/disagree is used). "
        'Reply JSON: {"answer": true|false, "question": "...", "options": []}.'
    )
    try:
        resp = generate(
            client,
            model,
            prompt,
            types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=200,
                response_mime_type="application/json",
                response_schema=_Offer,
            ),
        )
        got = _Offer.model_validate(json.loads(resp.text))
    except Exception:
        logging.getLogger(__name__).warning("vote offer gate failed; staying quiet", exc_info=True)
        return None
    q = " ".join(got.question.split())
    if not got.answer or not q:
        return None
    return {"question": q[:MAX_QUESTION_CHARS], "options": got.options}


def question_key(question: str) -> str:
    return hashlib.sha1(" ".join(question.casefold().split()).encode()).hexdigest()[:12]


def new_offer(
    *,
    guild_id: int,
    channel_id: int,
    question: str,
    options: list[str],
    voice_channel_id: int = 0,
    lang: str = "th",
    now: datetime,
) -> dict:
    return {
        "id": uuid.uuid4().hex[:12],
        "guild_id": guild_id,
        "channel_id": channel_id,
        "voice_channel_id": voice_channel_id,
        "question": question,
        "options": options,
        "lang": lang,
        "status": "pending",  # pending | opened | declined
        "created_at": iso(now),
    }


class OfferRepo:
    def __init__(self, docs: DocStore):
        self.docs = docs

    def put(self, rec: dict[str, Any]) -> dict:
        self.docs.put(OFFERS, rec["guild_id"], rec["id"], rec)
        return rec

    def get(self, offer_id: str) -> dict | None:
        return self.docs.find_any(OFFERS, offer_id)
