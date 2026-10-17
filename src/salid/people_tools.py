"""The agent tools of company knowledge: who does what, whom to ask, how we do things, onboarding
and handover. Logic only; the Discord side is injected as hooks by `people_bot.PeopleMixin`.

Everything a reader gets is scoped in code: an item learned from chat is shown only when the reader
can read at least one of its source channels (`people.evidence_visible`; an executive asking alone,
i.e. the answer reaches nobody else, reads all). A DM scope never teaches anything. Personal
preferences (`profiles.py`) are never part of a person profile. Where nothing is known the tools
say so and Salid must not make anything up.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from .company_brain import CompanyRepo
from .handover import STRINGS as _HANDOVER_STRINGS  # noqa: F401  (registers the catalog keys)
from .i18n import CATALOG, t
from .people import (
    Dossier,
    PeopleRepo,
    RosterEntry,
    build_org,
    clean,
    department_label,
    evidence_visible,
    norm,
    rank_experts,
    set_role,
    topic_match,
    view_items,
)
from .wiki import WikiPage, find_pages, slugify

logger = logging.getLogger(__name__)

MAX_HANDOVER_SHOWN = 6000
MAX_BRIEF_PEOPLE = 8
MAX_LIST = 10
MAX_BRIEF_CHARS = 6000

STRINGS = {
    "ob_title": {"th": "คู่มือต้อนรับ: {who}", "en": "Onboarding brief: {who}"},
    "ob_company": {"th": "บริษัทของเราทำอะไร", "en": "What the company does"},
    "ob_dept": {"th": "แผนกและหัวหน้า", "en": "Your department and manager"},
    "ob_people": {"th": "คนที่ควรรู้จัก", "en": "People to meet"},
    "ob_systems": {"th": "ระบบและสิทธิ์ที่ต้องใช้", "en": "Systems and access"},
    "ob_recurring": {"th": "งานประจำ", "en": "Recurring work"},
    "ob_projects": {"th": "โปรเจกต์ที่กำลังทำ", "en": "Current projects"},
    "ob_procedures": {"th": "ขั้นตอนที่บริษัทกำหนด", "en": "How we do things"},
    "ob_open": {"th": "สิ่งที่สลิดยังไม่รู้", "en": "Open questions"},
    "ob_head": {"th": "หัวหน้า: {name}", "en": "Head: {name}"},
    "ob_unknown": {"th": "ยังไม่มีข้อมูล", "en": "Nothing recorded yet"},
    "ob_gap_sop": {
        "th": "ยังไม่มีขั้นตอนมาตรฐานที่บริษัทสอนสลิด",
        "en": "No standard procedures have been taught to Salid yet",
    },
    "ob_gap_systems": {
        "th": "ยังไม่รู้ว่าแผนกนี้ใช้ระบบอะไรบ้าง",
        "en": "Salid does not know yet which systems this team uses",
    },
    "ob_gap_head": {
        "th": "ยังไม่รู้ว่าใครเป็นหัวหน้าแผนกนี้",
        "en": "Salid does not know yet who leads this team",
    },
    "ob_gap_people": {
        "th": "ยังไม่มีข้อมูลการทำงานของคนในแผนกนี้มากพอ",
        "en": "There is not enough learned yet about the people in this team",
    },
    "ob_predecessor": {
        "th": "มีเอกสารส่งมอบงานของ {names} ถามสลิดว่า “ส่งมอบงานของ {names}” ได้",
        "en": "A handover exists for {names}: ask Salid for “handover {names}”",
    },
}
CATALOG.update(STRINGS)


# Added to the system prompt when the tools exist (qa.stable_prompt, keyed on `who_to_ask`).
KNOWLEDGE_PROMPT = {
    "th": (
        "\nสลิดคือความจำของบริษัท (institutional memory): ถามว่าบริษัททำอะไร ใครทำอะไร ควรถามใครเรื่องไหน "
        "เราทำเรื่องนี้ยังไง ใครคือคนที่ควรรู้จัก หรือขอคู่มือสำหรับพนักงานใหม่ ให้ใช้ company_overview / "
        "who_to_ask / person_profile / how_do_we / onboarding_brief ก่อนตอบเสมอ ตอบจากผลที่ได้เท่านั้น "
        "อ้างที่มาเบาๆ เช่น “จากห้อง #dev เมื่อ 12 ต.ค.” ห้ามเปิดเผยห้องส่วนตัวหรือบันทึก DM และถ้าไม่มีหลักฐาน "
        "ให้บอกว่า “ยังไม่รู้ค่ะ” ห้ามเดาชื่อคนหรือแต่งขั้นตอน. คนที่ออกไปแล้ว: บอกว่าเคยทำอะไร ชี้ไปที่ผู้รับช่วง "
        "หรือเอกสารส่งมอบงาน (handover). ผู้บริหารบอกว่ามีคนลาออก → person_left; บอกตำแหน่ง/แผนก → set_role. "
        "ข้อมูลส่วนตัว/ความชอบของคนไม่อยู่ในที่นี้ (ใช้ได้แค่ข้อมูลด้านงาน).\n"
    ),
    "en": (
        "\nSalid is the company's institutional memory: when asked what the company does, who does "
        "what, whom to ask about something, how we do something, whom a newcomer should meet, or "
        "for an onboarding brief, call company_overview / who_to_ask / person_profile / how_do_we / "
        "onboarding_brief first and answer only from what they return. Cite the source lightly "
        '("from the #dev chat on 12 Oct"). Never expose private channels or DM notes. With no '
        'evidence say "I don\'t know yet"; never guess a name or invent a procedure. For someone '
        "who has left: say what they used to do and point to the successor or the handover. An "
        "executive saying someone left -> person_left; stating a role or department -> set_role. "
        "Personal preferences are not part of this: work knowledge only.\n"
    ),
}


@dataclass
class PeopleCtx:
    guild_id: int
    asker_id: int
    asker_name: str
    is_exec: bool
    lang: str
    repo: PeopleRepo
    company: CompanyRepo
    now: Callable[[], datetime]
    guild_name: str = ""
    readable: Callable[[], frozenset[int]] = lambda: frozenset()
    see_all: Callable[[], bool] = lambda: False  # an executive reading alone
    channel_name: Callable[[int], str] = lambda cid: str(cid)
    roster: Callable[[], list[RosterEntry]] = lambda: []
    channels: Callable[[], list[str]] = lambda: []
    wiki_pages: Callable[[], list[WikiPage]] = lambda: []  # the ones the asker may read
    projects: Callable[[], list[dict]] = lambda: []  # {name, status, department, owner}
    decisions: Callable[[str], list[dict]] = lambda topic: []  # {text, day, channel_id}
    resolve_member: Callable[[str], list[tuple[int, str]]] = lambda name: []
    gen_handover: Callable[[int, str, int], dict] | None = None  # (uid, name, successor) -> dict
    on_left: Callable[[int, str, int], dict] | None = None  # (uid, name, successor) -> dict
    pdf: Callable[[str, str], bytes] | None = None  # (title, markdown) -> PDF bytes
    save_wiki: Callable[[str, str, list[int]], dict] | None = None  # (title, body, channels)
    outbox: list[tuple[str, bytes]] = field(default_factory=list)


# ------------------------------------------------------------------ helpers
def _cite(ctx: PeopleCtx, src: dict | None) -> str:
    if not src:
        return ""
    cid = int(src.get("channel_id") or 0)
    where = f"#{ctx.channel_name(cid)}" if cid else ""
    kind = src.get("kind", "chat")
    return " ".join(x for x in (where, kind, src.get("day", "")) if x)


def _items(ctx: PeopleCtx, rows: list[dict]) -> list[dict]:
    for r in rows:
        src = r.pop("source", None)
        if src:
            r["from"] = _cite(ctx, src)
    return rows


def _hidden(items: list, readable, see_all: bool) -> int:
    return sum(1 for i in items if not evidence_visible(i.evidence, readable, see_all))


def _find(ctx: PeopleCtx, name: str) -> tuple[Dossier | None, dict | None]:
    """(dossier, None) | (None, error). A member Salid has not learned anything about yet gets an
    empty, unsaved dossier."""
    name = " ".join((name or "").split())
    if not name:
        return None, {"error": "give the person's name"}
    found = ctx.repo.find(ctx.guild_id, name)
    if len(found) == 1:
        return found[0], None
    if len(found) > 1:
        return None, {"error": "ambiguous", "candidates": [d.name for d in found[:6]]}
    members = ctx.resolve_member(name)
    if len(members) == 1:
        return Dossier(user_id=members[0][0], name=members[0][1]), None
    if len(members) > 1:
        return None, {"error": "ambiguous", "candidates": [n for _, n in members[:6]]}
    return None, {"error": "not_found", "note": "nobody by that name is known here"}


def _name_of(ctx: PeopleCtx, uid: int) -> str:
    """A person's name: their dossier, else the member list ("" when unknown)."""
    d = ctx.repo.get(ctx.guild_id, uid)
    return (d.name if d and d.name else "") or next(
        (r.name for r in ctx.roster() if r.user_id == uid), ""
    )


def _roster_dept(ctx: PeopleCtx, uid: int) -> str:
    return next((department_label(r.department) for r in ctx.roster() if r.user_id == uid), "")


def _org(ctx: PeopleCtx):
    roster = ctx.roster()
    if roster:
        return build_org(roster)
    return ctx.company.get(ctx.guild_id).org


def _org_view(depts) -> list[dict]:
    return [
        {
            "department": d.name or "unassigned",
            "head": d.head.name if d.head else None,
            "positions": [
                {"title": p.title or "(no title)", "people": [x.name for x in p.people]}
                for p in d.positions
            ],
        }
        for d in depts
    ]


# ------------------------------------------------------------------ the tools
def company_overview_data(ctx: PeopleCtx) -> dict:
    pb = ctx.company.get(ctx.guild_id)
    depts = _org(ctx)
    people = sum(len(p.people) for d in depts for p in d.positions)
    pages = [p.title for p in ctx.wiki_pages()][:MAX_LIST]
    projects = [p.get("name", "") for p in ctx.projects()][:MAX_LIST]
    out: dict = {
        "company": ctx.guild_name or None,
        "industry": pb.industry or None,
        "about": pb.about or None,
        "people_count": people,
        "departments": _org_view(depts),
        "key_contacts": [
            {"name": k.name, "context": k.context, "times": k.count}
            for k in pb.key_contacts[:MAX_LIST]
        ],
        "where_things_live": {
            "channels": ctx.channels()[:MAX_LIST],
            "wiki_pages": pages,
            "projects": projects,
            "procedures": [s.title for s in pb.sop][:MAX_LIST],
        },
    }
    if not (pb.about or pb.industry or depts):
        out["note"] = (
            "Salid has not learned what the company does yet; say so and invite an admin to "
            "teach it (teach_company about / industry)."
        )
    return out


def who_to_ask_data(ctx: PeopleCtx, topic: str) -> dict:
    topic = " ".join((topic or "").split())
    if not topic:
        return {"error": "give the topic"}
    readable, see_all = ctx.readable(), ctx.see_all()
    experts = rank_experts(ctx.repo.list(ctx.guild_id), topic, readable=readable, see_all=see_all)
    rows: list[dict] = []
    for e in experts:
        row: dict = {
            "name": e.name,
            "status": e.status,
            "why": [
                {
                    "what": w["what"],
                    "text": w["text"],
                    "from": _cite(ctx, {"channel_id": w["channel_id"], "day": w["day"]}),
                }
                for w in e.why
            ],
            "last_seen": e.last_seen,
        }
        if e.status == "left":
            d = ctx.repo.get(ctx.guild_id, e.user_id)
            row["left"] = True
            row["left_at"] = d.left_at if d else ""
            if e.successor_id and (succ := _name_of(ctx, e.successor_id)):
                row["successor"] = succ
            row["handover_available"] = bool(d and d.handover)
            row["note"] = (
                "this person has left: send the question to the successor if there is one, "
                "otherwise say what they used to do (the 'why' lines) and offer the handover"
            )
        rows.append(row)
    if rows:
        return {"topic": topic, "people": rows}
    # nothing in the dossiers: fall back to the company map and the roster's roles
    pb = ctx.company.get(ctx.guild_id)
    mapped = [
        {"name": p.name, "why": "listed as a go-to person for this topic"}
        for w in pb.who_to_ask
        if topic_match(topic, w.topic) > 0
        for p in w.people[:2]
    ]
    roles = [
        {"name": r.name, "why": f"role: {r.title or department_label(r.department)}"}
        for r in ctx.roster()
        if topic_match(topic, f"{r.title} {r.department}") >= 1
    ][:5]
    if mapped or roles:
        return {"topic": topic, "people": (mapped + roles)[:6]}
    return {
        "topic": topic,
        "people": [],
        "note": "Salid has no evidence yet of who handles this; say 'I don't know yet' and "
        "suggest asking a manager. Do not guess a name.",
    }


def person_profile_data(ctx: PeopleCtx, name: str) -> dict:
    d, err = _find(ctx, name)
    if err:
        return err
    readable, see_all = ctx.readable(), ctx.see_all()
    dept = d.department or _roster_dept(ctx, d.user_id)
    hidden = sum(
        _hidden(items, readable, see_all)
        for items in (
            d.responsibilities,
            d.how_they_work,
            d.systems,
            d.recurring,
            d.expertise,
            d.contacts,
        )
    )
    out: dict = {
        "name": d.name,
        "status": d.status,
        "role": d.role_title or None,
        "department": dept or None,
        "responsibilities": _items(ctx, view_items(d.responsibilities, readable, see_all)),
        "how_they_work": _items(ctx, view_items(d.how_they_work, readable, see_all)),
        "systems": _items(ctx, view_items(d.systems, readable, see_all)),
        "recurring_duties": _items(ctx, view_items(d.recurring, readable, see_all)),
        "expertise": [
            t_.topic
            for t_ in sorted(d.expertise, key=lambda x: -x.score)
            if evidence_visible(t_.evidence, readable, see_all)
        ][:MAX_LIST],
        "deals_with": [
            {"who": c.who, "colleague": bool(c.user_id), "about": c.about, "times": c.count}
            for c in sorted(d.contacts, key=lambda c: -c.count)
            if evidence_visible(c.evidence, readable, see_all)
        ][:MAX_LIST],
    }
    if hidden:
        out["hidden_items"] = hidden
        out["hidden_note"] = (
            "some details come from channels the asker cannot read: do not describe them"
        )
    if d.status == "left":
        out["left_at"] = d.left_at
        out["handover_available"] = bool(d.handover)
        if d.successor_id and (succ := _name_of(ctx, d.successor_id)):
            out["successor"] = succ
    empty = not any(
        out[k]
        for k in (
            "responsibilities",
            "how_they_work",
            "systems",
            "recurring_duties",
            "expertise",
            "deals_with",
        )
    )
    if empty and not hidden:
        out["note"] = "Salid has not learned about this person's work yet; say so, do not guess."
    return out


def how_do_we_data(ctx: PeopleCtx, topic: str) -> dict:
    topic = " ".join((topic or "").split())
    if not topic:
        return {"error": "give the topic"}
    pb = ctx.company.get(ctx.guild_id)
    sops = [
        {"title": s.title, "steps": s.steps}
        for s in pb.sop
        if topic_match(topic, s.title) > 0 or any(topic_match(topic, x) >= 1.5 for x in s.steps)
    ][:3]
    faqs = [{"q": f.q, "a": f.a} for f in pb.faq if topic_match(topic, f.q) > 0][:3]
    terms = [(s, p) for s, p in find_pages(ctx.wiki_pages(), topic)][:3]
    wiki = [
        {"title": p.title, "updated_at": p.updated_at, "snippet": " ".join(p.body.split())[:300]}
        for _, p in terms
    ]
    readable, see_all = ctx.readable(), ctx.see_all()
    people_say: list[dict] = []
    for d in ctx.repo.list(ctx.guild_id):
        for label, items in (("recurring", d.recurring), ("how", d.how_they_work)):
            for f in items:
                if topic_match(topic, f.text) > 0 and evidence_visible(
                    f.evidence, readable, see_all
                ):
                    best = max(f.evidence, key=lambda e: e.day, default=None)
                    people_say.append(
                        {
                            "who": d.name,
                            "status": d.status,
                            "kind": label,
                            "text": f.text,
                            "from": _cite(
                                ctx,
                                {"channel_id": best.channel_id, "day": best.day, "kind": best.kind}
                                if best
                                else None,
                            ),
                            "_s": topic_match(topic, f.text) * (0.5 + f.confidence),
                        }
                    )
    people_say.sort(key=lambda r: -r["_s"])
    for r in people_say:
        r.pop("_s", None)
    decisions = [
        {"text": x.get("text", ""), "day": x.get("day", ""), "from": _cite(ctx, x)}
        for x in ctx.decisions(topic)[:5]
    ]
    found = bool(sops or faqs or wiki or people_say or decisions)
    out: dict = {
        "topic": topic,
        "sops": sops,
        "faqs": faqs,
        "wiki_pages": wiki,
        "people_say": people_say[:6],
        "decisions": decisions,
        "found": found,
    }
    if not found:
        out["note"] = (
            "No procedure, wiki page, FAQ or learned habit matches. Say you don't know how the "
            "company does this yet; offer who_to_ask(topic) and to write it down once explained."
        )
    else:
        out["note"] = (
            "Answer from these sources, name them lightly (wiki page, SOP, who does it), and say "
            "what is missing instead of filling gaps."
        )
    return out


def build_brief(
    ctx: PeopleCtx, role_or_department: str, for_user: str
) -> tuple[str, str, list[int], dict]:
    """(markdown, title, source channel ids, summary info) of an onboarding brief."""
    lang = ctx.lang
    readable, see_all = ctx.readable(), ctx.see_all()
    roster = ctx.roster()
    dossiers = {d.user_id: d for d in ctx.repo.list(ctx.guild_id)}
    pb = ctx.company.get(ctx.guild_id)
    who_label, dept, title = "", "", ""
    target_uid = 0
    if for_user:
        d, err = _find(ctx, for_user)
        if err:
            return "", "", [], err
        target_uid, who_label = d.user_id, d.name
        d = dossiers.get(d.user_id) or d
        dept = d.department or _roster_dept(ctx, d.user_id)
        title = d.role_title or next((r.title for r in roster if r.user_id == d.user_id), "")
    raw = " ".join((role_or_department or "").split())
    if raw:
        known = {department_label(r.department) for r in roster if r.department}
        known |= {department_label(d.department) for d in dossiers.values() if d.department}
        hit = next((k for k in known if k.casefold() == raw.casefold()), None) or next(
            (k for k in known if raw.casefold() in k.casefold() or k.casefold() in raw.casefold()),
            None,
        )
        if hit:
            dept = hit
        else:
            title = raw
        who_label = who_label or raw
    if not (dept or title or who_label):
        who_label = ctx.asker_name
        target_uid = ctx.asker_id
        dept = (
            dossiers.get(ctx.asker_id).department if dossiers.get(ctx.asker_id) else ""
        ) or _roster_dept(ctx, ctx.asker_id)
    team = [
        r
        for r in roster
        if r.user_id != target_uid
        and (
            (dept and department_label(r.department).casefold() == dept.casefold())
            or (not dept and title and norm(r.title) == norm(title))
        )
    ]
    sources: set[int] = set()
    gaps: list[str] = []

    def keep(items) -> list:
        out = []
        for f in items:
            if evidence_visible(f.evidence, readable, see_all):
                out.append(f)
                sources.update(e.channel_id for e in f.evidence if e.channel_id > 0)
        return out

    parts: list[str] = []
    # 1. the company
    about = " ".join(x for x in (pb.industry, pb.about) if x)
    parts.append(f"## {t('ob_company', lang)}\n{about or t('ob_unknown', lang)}")
    # 2. department and manager
    head = None
    if dept:
        org = next((o for o in build_org(roster) if o.name.casefold() == dept.casefold()), None)
        head = org.head if org else None
        if head and head.user_id == target_uid:
            head = None
    dept_lines = [f"{dept}" if dept else t("ob_unknown", lang)]
    if head:
        dept_lines.append(t("ob_head", lang, name=head.name))
    elif dept:
        gaps.append(t("ob_gap_head", lang))
    parts.append(f"## {t('ob_dept', lang)}\n" + "\n".join(f"- {x}" for x in dept_lines))
    # 3. people to meet
    meet: list[str] = []
    if head:
        meet.append(f"{head.name}: {t('ob_head', lang, name='').strip(': ')}")
    for r in team[:MAX_BRIEF_PEOPLE]:
        d = dossiers.get(r.user_id)
        why = r.title or ""
        if d:
            tops = [x.topic for x in sorted(keep(d.expertise), key=lambda x: -x.score)[:2]]
            if tops:
                why = (why + "; " if why else "") + ", ".join(tops)
        meet.append(f"{r.name}" + (f": {why}" if why else ""))
    team_ids = {r.user_id for r in team} | {target_uid}
    for w in pb.who_to_ask[:6]:
        p = w.people[0] if w.people else None
        if p and p.user_id not in team_ids and len(meet) < MAX_BRIEF_PEOPLE + 4:
            meet.append(f"{p.name}: {w.topic}")
            team_ids.add(p.user_id)
    if not any(dossiers.get(r.user_id) for r in team):
        gaps.append(t("ob_gap_people", lang))
    parts.append(
        f"## {t('ob_people', lang)}\n"
        + ("\n".join(f"- {x}" for x in meet) or f"- {t('ob_unknown', lang)}")
    )
    # 4. systems
    sys_count: Counter = Counter()
    sys_text: dict[str, str] = {}
    rec_rows: list[str] = []
    for r in team:
        d = dossiers.get(r.user_id)
        if not d:
            continue
        for f in keep(d.systems):
            sys_count[norm(f.text)] += 1
            sys_text.setdefault(norm(f.text), f.text)
        for f in keep(d.recurring)[-3:]:
            rec_rows.append(f"{f.text} ({r.name})")
    systems = [sys_text[k] for k, _ in sys_count.most_common(MAX_LIST)]
    if not systems:
        gaps.append(t("ob_gap_systems", lang))
    parts.append(
        f"## {t('ob_systems', lang)}\n"
        + ("\n".join(f"- {x}" for x in systems) or f"- {t('ob_unknown', lang)}")
    )
    parts.append(
        f"## {t('ob_recurring', lang)}\n"
        + ("\n".join(f"- {x}" for x in rec_rows[:MAX_LIST]) or f"- {t('ob_unknown', lang)}")
    )
    # 5. projects
    projects = ctx.projects()
    mine = [
        p for p in projects if dept and str(p.get("department", "")).casefold() == dept.casefold()
    ]
    shown = (mine or projects)[:8]
    parts.append(
        f"## {t('ob_projects', lang)}\n"
        + (
            "\n".join(f"- {p.get('name', '')} ({p.get('status', '')})" for p in shown)
            or f"- {t('ob_unknown', lang)}"
        )
    )
    # 6. how we do things
    if pb.sop:
        parts.append(
            f"## {t('ob_procedures', lang)}\n"
            + "\n".join(f"- {s.title}" for s in pb.sop[:MAX_LIST])
        )
    else:
        gaps.append(t("ob_gap_sop", lang))
    # predecessors who left with the same role / department
    gone = [
        d.name
        for d in dossiers.values()
        if d.status == "left"
        and d.handover
        and (
            (dept and d.department.casefold() == dept.casefold())
            or (title and norm(d.role_title) == norm(title))
        )
    ][:3]
    if gone:
        gaps.append(t("ob_predecessor", lang, names=", ".join(gone)))
    parts.append(
        f"## {t('ob_open', lang)}\n"
        + ("\n".join(f"- {g}" for g in gaps) or f"- {t('ob_unknown', lang)}")
    )
    body = "\n\n".join(parts)[:MAX_BRIEF_CHARS]
    info = {
        "department": dept or None,
        "role": title or None,
        "team": [r.name for r in team][:MAX_BRIEF_PEOPLE],
    }
    return body, t("ob_title", lang, who=who_label or dept or title), sorted(sources), info


def onboarding_brief_data(
    ctx: PeopleCtx, role_or_department: str, for_user: str, save_to_wiki: bool
) -> dict:
    body, title, sources, info = build_brief(ctx, role_or_department, for_user)
    if not body:
        return info or {"error": "could not build the brief"}
    out: dict = {"title": title, "brief_markdown": body, **info}
    if ctx.pdf is not None:
        try:
            data = ctx.pdf(title, body)
            name = f"onboarding-{slugify(title)[:40]}.pdf"
            ctx.outbox.append((name, data))
            out["attached_file"] = name
        except Exception:
            logger.warning("onboarding PDF failed", exc_info=True)
    if save_to_wiki:
        if ctx.save_wiki is None:
            out["wiki"] = {"error": "the wiki is not available"}
        else:
            out["wiki"] = ctx.save_wiki(title, body, sources)
    else:
        out["offer"] = (
            "ask whether to save this as a wiki page draft (call again with save_to_wiki=True)"
        )
    out["instruction"] = (
        "Give a short summary of the sections; the full brief is in the attached PDF."
    )
    return out


def _may_see_handover(ctx: PeopleCtx, d: Dossier) -> bool:
    return (
        ctx.is_exec
        or d.user_id == ctx.asker_id
        or (d.successor_id and d.successor_id == ctx.asker_id)
    )


def handover_data(ctx: PeopleCtx, name: str, successor: str, regenerate: bool) -> dict:
    d, err = _find(ctx, name)
    if err:
        return err
    if not _may_see_handover(ctx, d):
        return {"error": "not_allowed", "note": "only executives, the person and their successor"}
    succ_uid = 0
    if successor:
        if not ctx.is_exec:
            return {"error": "not_allowed", "note": "only an executive names a successor"}
        s, serr = _find(ctx, successor)
        if serr:
            return {**serr, "which": "successor"}
        succ_uid = s.user_id
    d = ctx.repo.get(ctx.guild_id, d.user_id) or d
    text = d.handover
    made = False
    if (not text or regenerate or succ_uid) and (ctx.is_exec or d.user_id == ctx.asker_id):
        if not text or regenerate:
            if ctx.gen_handover is None:
                return {"error": "the handover writer is not available"}
            res = ctx.gen_handover(d.user_id, d.name, succ_uid)
            if res.get("error"):
                return res
            text, made = res.get("text", ""), True
        elif succ_uid:
            ctx.repo.update(
                ctx.guild_id,
                d.user_id,
                d.name,
                lambda x: setattr(x, "successor_id", succ_uid) or True,
            )
    if not text:
        return {"name": d.name, "found": False, "note": "no handover has been written yet"}
    title = t("ho_title", ctx.lang, name=d.name)
    out: dict = {
        "name": d.name,
        "status": d.status,
        "written_now": made,
        "handover": text[:MAX_HANDOVER_SHOWN],
        "instruction": "Summarise the sections briefly; the full document is attached as a PDF.",
    }
    if ctx.pdf is not None:
        try:
            fname = f"handover-{slugify(d.name)[:40]}.pdf"
            ctx.outbox.append((fname, ctx.pdf(title, text)))
            out["attached_file"] = fname
        except Exception:
            logger.warning("handover PDF failed", exc_info=True)
    return out


def set_role_data(ctx: PeopleCtx, name: str, title: str, department: str) -> dict:
    d, err = _find(ctx, name)
    if err:
        return err
    if not (ctx.is_exec or d.user_id == ctx.asker_id):
        return {"error": "not_allowed", "note": "only an executive, or the person for themself"}
    title, department = clean(title, 80), clean(department, 60)
    if not title and not department:
        return {"error": "give a title and/or a department"}
    ctx.repo.update(
        ctx.guild_id,
        d.user_id,
        d.name,
        lambda x: set_role(x, title, "stated", department or None) or True,
    )
    return {
        "status": "saved",
        "name": d.name,
        "role": title or d.role_title,
        "department": department or d.department,
    }


def person_left_data(ctx: PeopleCtx, name: str, successor: str) -> dict:
    if not ctx.is_exec:
        return {"error": "not_allowed", "note": "only executives can record that someone left"}
    d, err = _find(ctx, name)
    if err:
        return err
    succ_uid = 0
    if successor:
        s, serr = _find(ctx, successor)
        if serr:
            return {**serr, "which": "successor"}
        succ_uid = s.user_id
    if ctx.on_left is None:
        return {"error": "not available here"}
    return ctx.on_left(d.user_id, d.name, succ_uid)


# ------------------------------------------------------------------ registration
def people_tools(ctx: PeopleCtx) -> list[Callable]:
    """The functions `_agent_kwargs` appends to `extra_tools` (docstrings = tool descriptions)."""

    def safe(fn: Callable[[], dict]) -> dict:
        try:
            return fn()
        except Exception:
            logger.exception("company knowledge tool failed")
            return {"status": "error", "error": "the tool failed; try again"}

    def company_overview() -> dict:
        """What this company does, how it is organised and where things live: "บริษัทเราทำอะไร",
        "what does the company do", "who does what here", "company structure". Returns industry,
        about, departments with heads and positions, the key outside contacts, and the channels,
        wiki pages, projects and procedures. If a part is empty say Salid does not know it yet."""
        return safe(lambda: company_overview_data(ctx))

    def who_to_ask(topic: str) -> dict:
        """Whom to ask about a topic: "ใครรู้เรื่อง X", "ควรถามใครเรื่อง X", "who handles invoices",
        "who do I ask about the server". Ranks people by what Salid has seen them do or know and
        says WHY (what, from which channel and day). A person who has LEFT comes back with their
        successor or a pointer to their handover: say what they used to do and offer the handover.
        If nobody is known say "I don't know yet"; never guess a name.

        Args:
            topic: the subject in a few words (e.g. "month-end close", "เซิร์ฟเวอร์", "supplier contracts").
        """
        return safe(lambda: who_to_ask_data(ctx, topic))

    def person_profile(name: str) -> dict:
        """The WORK profile of a colleague: role, department, responsibilities, how they work,
        systems they use, recurring duties, topics they know and who they deal with. Use for "X
        ทำอะไร", "what does X do", "tell me about X's work". Never personal preferences or
        private life. Parts learned in channels the asker cannot read are hidden (only counted).

        Args:
            name: display name, @mention or id.
        """
        return safe(lambda: person_profile_data(ctx, name))

    def how_do_we(topic: str) -> dict:
        """How the company does something: "เราทำ X ยังไง", "how do we onboard a client", "what is
        our process for refunds". Assembles the company's procedures (SOPs), FAQ, wiki pages, what
        people who do it say, and past meeting decisions, each with where it comes from. When none
        exists it says so: tell the user you don't know yet instead of inventing a process.

        Args:
            topic: the procedure or task in a few words.
        """
        return safe(lambda: how_do_we_data(ctx, topic))

    def onboarding_brief(
        role_or_department: str = "", for_user: str = "", save_to_wiki: bool = False
    ) -> dict:
        """What a NEW person in a role or department must know: company overview, their
        department and manager, people to meet (with why), systems and access, recurring work,
        current projects, procedures and open questions. Attaches a PDF. Use for "ทำคู่มือต้อนรับ",
        "onboarding for the new accountant", "what should a new X know". With no arguments it is
        for the asker. Ask before saving: pass save_to_wiki=True only after the user confirms
        they want it kept as a wiki page.

        Args:
            role_or_department: a department (e.g. "finance") or a job title; empty = the person in for_user, else the asker.
            for_user: the new person (name or @mention), when the brief is for someone specific.
            save_to_wiki: save the brief as a wiki page (only after the asker said yes).
        """
        return safe(lambda: onboarding_brief_data(ctx, role_or_department, for_user, save_to_wiki))

    def handover(name: str, successor: str = "", regenerate: bool = False) -> dict:
        """The handover document of a person (what they did and how, for whoever takes over):
        "ส่งมอบงานของ X", "show X's handover", "what did X leave behind". Executives and the
        person themself may generate or read it; a named successor may read it. An executive can
        name the `successor`. Attaches a PDF.

        Args:
            name: the person (name, @mention or id).
            successor: who takes over (executives only).
            regenerate: write it again from the latest knowledge.
        """
        return safe(lambda: handover_data(ctx, name, successor, regenerate))

    def set_role(name: str, title: str = "", department: str = "") -> dict:
        """Record someone's job title and/or department explicitly: "X เป็นหัวหน้าฝ่ายบัญชี",
        "set Mali's role to accountant, finance". Executives for anyone, or a person for
        themself. What is stated this way wins over what Salid guessed.

        Args:
            name: the person.
            title: job title / position.
            department: department or team.
        """
        return safe(lambda: set_role_data(ctx, name, title, department))

    def person_left(name: str, successor: str = "") -> dict:
        """An executive says someone left or resigned ("X ลาออกแล้ว", "X left the company"). Marks
        them as left (their knowledge is kept), lists their open tasks with a hint to reassign, and
        writes the handover document, sent to the executives' private channels. Executives only.

        Args:
            name: the person who left.
            successor: who takes over, if known.
        """
        return safe(lambda: person_left_data(ctx, name, successor))

    tools = [
        company_overview,
        who_to_ask,
        person_profile,
        how_do_we,
        onboarding_brief,
        handover,
        set_role,
        person_left,
    ]
    for fn in tools:
        fn.outbox = ctx.outbox  # type: ignore[attr-defined]
    return tools


# wiki draft helper used by the bot hook and the tests
def brief_page(
    title: str, body: str, channel_ids: list[int], author_id: int, author: str
) -> WikiPage:
    return WikiPage(
        slug=slugify(title),
        title=title,
        body=body,
        channel_ids=list(channel_ids),
        channel_authors={str(c): [author_id] for c in channel_ids},
        author_id=author_id,
        author_name=author,
    )
