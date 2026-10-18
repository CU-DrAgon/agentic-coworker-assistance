"""The website's company positions (`orgs.Org.positions`), read and written for Salid (pure logic
on an `Org`; the Discord side is `positions_bot.py`).

The central website is the ONLY truth about who holds which job: `Position.holders` (member
emails) and the chart (`reports_to`). A member's website access right (owner / admin / member) is
never a position. Functions here only read an `Org` except `put_on_site`, which writes through
`OrgStore.save_position`.
"""

from __future__ import annotations

import logging

from .orgs import Member, Org, OrgStore, Position
from .positions import level_of, norm_title

logger = logging.getLogger(__name__)

MAX_LEVEL = 7
RESP_CHARS = 300


def _by_id(org: Org) -> dict[str, Position]:
    return {p.id: p for p in org.positions}


def in_chart(org: Org, pos: Position) -> bool:
    """True when the position is linked into the chart (reports to one, or one reports to it)."""
    return bool(pos.reports_to) or any(x.reports_to == pos.id for x in org.positions)


def chart_depth(org: Org, pos: Position) -> int:
    """Steps from the top of the chart: the top = 1, each `reports_to` step +1 (capped at 7).
    0 when the position is not part of a chart."""
    if not in_chart(org, pos):
        return 0
    ids, depth, cur, seen = _by_id(org), 1, pos, {pos.id}
    while cur.reports_to and cur.reports_to in ids and cur.reports_to not in seen:
        cur = ids[cur.reports_to]
        seen.add(cur.id)
        depth += 1
    return min(depth, MAX_LEVEL)


def level_for(org: Org, pos: Position) -> int:
    """The level of a website position: the words of its title when they say (CEO, Manager...),
    otherwise its depth in the chart, otherwise 0 (unknown)."""
    return level_of(pos.title) or chart_depth(org, pos)


def held_by(org: Org, email: str) -> list[Position]:
    e = (email or "").strip().lower()
    return [p for p in org.positions if e and e in {h.lower() for h in p.holders}]


def _seniority(org: Org, p: Position) -> tuple:
    depth = chart_depth(org, p)
    return (depth == 0, depth, level_of(p.title) or 99, p.title.casefold(), p.id)


def most_senior(org: Org, positions: list[Position]) -> Position | None:
    """The position nearest the top of the chart (shortest `reports_to` chain)."""
    return min(positions, key=lambda p: _seniority(org, p), default=None)


def center_position(org: Org, member: Member) -> tuple[str, Position | None]:
    """(title, website position) of a member's company position: the most senior position that
    lists their email as a holder; else `Member.title` (what v0.15 mirrored); else ("", None).
    The access role is never consulted."""
    pos = most_senior(org, held_by(org, member.email))
    if pos is not None:
        return pos.title, pos
    return (getattr(member, "title", "") or "").strip(), None


def held_titles(org: Org, email: str) -> str:
    """All position titles a person holds, most senior first (for change detection)."""
    held = sorted(held_by(org, email), key=lambda p: _seniority(org, p))
    return ", ".join(p.title for p in held)


def find_on_site(org: Org, title: str, aliases=()) -> Position | None:
    names = {norm_title(title), *(norm_title(a) for a in aliases)} - {""}
    return next((p for p in org.positions if norm_title(p.title) in names), None)


def put_on_site(
    store: OrgStore,
    org_id: str,
    email: str,
    title: str,
    *,
    department: str = "",
    prev_id: str = "",
    aliases=(),
) -> str:
    """Make `email` a holder of the website position called `title` (found case / alias
    insensitively, else created by Salid), and drop them from `prev_id` (the position Salid last
    put them on) when that is a different one. Returns the position id ("" on failure)."""
    email = email.strip().lower()
    org = store.get(org_id)
    if org is None:
        return ""
    try:
        pos = find_on_site(org, title, aliases)
        if pos is None:
            pos = store.save_position(
                org_id, by="salid", title=title, department=department or None, holders=[email]
            )
        elif email not in {h.lower() for h in pos.holders}:
            pos = store.save_position(org_id, by="salid", pid=pos.id, holders=[*pos.holders, email])
        old = next((p for p in org.positions if p.id == prev_id), None) if prev_id else None
        if old is not None and old.id != pos.id and email in {h.lower() for h in old.holders}:
            store.save_position(
                org_id, by="salid", pid=old.id,
                holders=[h for h in old.holders if h.lower() != email],
            )  # fmt: skip
        return pos.id
    except (KeyError, ValueError):
        logger.warning("positions: could not write the website position", exc_info=True)
        return ""


# ------------------------------------------------------------------ who reports to whom
def manager_email(org: Org, email: str) -> str:
    """The person's manager: `Member.reports_to`, else the first holder of the position their
    position reports to. "" when unknown."""
    e = (email or "").strip().lower()
    mem = next((m for m in org.members if m.email.lower() == e), None)
    if mem is None:
        return ""
    if mem.reports_to:
        return mem.reports_to.lower()
    mine = most_senior(org, held_by(org, e))
    if mine is None or not mine.reports_to:
        return ""
    boss = _by_id(org).get(mine.reports_to)
    return next((h.lower() for h in (boss.holders if boss else []) if h.lower() != e), "")


def management_chain(org: Org, email: str, steps: int = 2) -> list[str]:
    """[manager, manager's manager] (as far as known, at most `steps`)."""
    out: list[str] = []
    cur = (email or "").lower()
    while len(out) < steps:
        nxt = manager_email(org, cur)
        if not nxt or nxt == (email or "").lower() or nxt in out:
            break
        out.append(nxt)
        cur = nxt
    return out


def position_card(org: Org, email: str) -> dict | None:
    """{title, department, responsibilities (first 300 chars)} of a person's senior position."""
    pos = most_senior(org, held_by(org, email))
    if pos is None:
        return None
    return {
        "title": pos.title,
        "department": pos.department,
        "responsibilities": (pos.responsibilities or pos.summary or "")[:RESP_CHARS],
    }
