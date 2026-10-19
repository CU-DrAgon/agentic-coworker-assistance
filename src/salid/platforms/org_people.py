"""Org record and people directory tools for a VERIFIED member on Slack, LINE, Telegram, WhatsApp.

Discord answers "what is my position?", "who is the CEO?", "who can approve purchases?" from the
org record. Here the sender must be a verified member of the org that owns the tenant
(`Assistant.identity_of`); a guest or unverified sender gets NO tools. Rules:

* the tools are bound to the asker's own org, taken from the verified link, never from the model;
* on LINE nothing is added unless `SALID_LINE_ORG_LINKING` is on (LINE User Data Policy 3.2.5);
* org directory facts (names, titles, departments, roles) are fine in a group room, but an email
  address is shown only in a 1:1 chat: in a group room people are named by display name or the
  part of their email before the @, and `email` fields are left out.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..orgs import Member, Org
    from .core import Assistant, Incoming

logger = logging.getLogger(__name__)

MAX_PEOPLE = 50
ADMIN_ROLES = ("owner", "admin")


def _local(email: str) -> str:
    return email.split("@", 1)[0]


def org_people_tools(org: Org, email: str, names: dict[str, str], *, private: bool) -> list:
    """The agent-facing functions (docstrings are the tool descriptions).

    `names`: lowercase email -> display name. `private`: a 1:1 chat (emails may be shown)."""
    me = email.strip().lower()
    held: dict[str, str] = {}  # member email -> title of the position they hold
    for p in org.positions:
        for h in p.holders:
            held.setdefault(h.lower(), p.title)

    def name_of(m: Member) -> str:
        return names.get(m.email.lower()) or _local(m.email)

    def card(m: Member) -> dict:
        out: dict = {
            "name": name_of(m),
            "role": m.role,
            "title": m.title or held.get(m.email.lower(), ""),
            "department": m.department,
            "finance_approver": bool(m.finance),
        }
        if private:
            out["email"] = m.email
        return out

    def my_profile() -> dict:
        """The asker's own record in their organisation: name, role (owner/admin/member), job
        title, department, level (1 founder/owner ... 7 intern, 0 unknown), whether they are an
        authorized finance approver, and the chat platforms linked to them. Use it for "what is
        my position / role / title?"."""
        m = next((x for x in org.members if x.email.lower() == me), None)
        if m is None:
            return {"error": "not a member"}
        out = card(m)
        out.update(
            organisation=org.name,
            level=m.level,
            linked_platforms=sorted(m.all_identities()),
        )
        return out

    def org_people(query: str = "") -> dict:
        """The people of the asker's organisation (up to 50): name, title, department, role,
        finance approver. `query` filters by a part of the name, title or department, e.g. "CEO",
        "finance", "marketing". Use it for "who is the CEO?" and "who is in finance?"."""
        q = (query or "").strip().lower()
        rows = []
        for m in org.members:
            c = card(m)
            hay = f"{c['name']} {c['title']} {c['department']}".lower()
            if q and q not in hay:
                continue
            rows.append(c)
        return {
            "organisation": org.name,
            "total_matches": len(rows),
            "people": rows[:MAX_PEOPLE],
        }

    def who_approves(topic: str = "purchase") -> dict:
        """Who can approve something such as a purchase or budget: the authorized finance
        approvers, plus the org owners and admins who can override. Use it for "who can approve
        purchases?"."""
        fin = [card(m) for m in org.members if m.finance]
        boss = [card(m) for m in org.members if m.role in ADMIN_ROLES and not m.finance]
        return {
            "topic": (topic or "purchase")[:80],
            "authorized_approvers": fin,
            "owners_and_admins": boss,
            "note": (
                "Authorized approvers are the members flagged as finance approvers. Owners and "
                "admins can also act. A company approval policy, if any, is in its documents."
            ),
        }

    return [my_profile, org_people, who_approves]


def _lookup(a: Assistant, msg: Incoming):
    who = a.identity_of(msg)
    if who is None or a.identity is None:
        return None
    org_id, email = who
    orgs = a.identity.orgs
    org = orgs.get(org_id)
    if org is None or org.role_of(email) is None:
        return None
    names: dict[str, str] = {}
    for m in org.members:
        try:
            n = orgs.user_profile(m.email).name
        except Exception:  # a broken profile must not hide the directory
            n = ""
        if n:
            names[m.email.lower()] = n
    return org, email, names


async def org_people_tools_for(a: Assistant, msg: Incoming) -> list:
    """The tools for this message, or [] (unverified / guest / LINE without linking)."""
    if msg.platform == "line" and not a.ps.line_org_linking:
        return []
    try:
        found = await asyncio.to_thread(_lookup, a, msg)
    except Exception as e:
        logger.warning("org people lookup failed (%s)", type(e).__name__)
        return []
    if found is None:
        return []
    org, email, names = found
    return org_people_tools(org, email, names, private=msg.is_direct)
