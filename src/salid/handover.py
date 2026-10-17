"""The handover document of someone who leaves (pure: prompt, fallback text, evidence digest).

When a member leaves, Salid writes down what the company must not lose: their role, what they were
responsible for, how to do each recurring duty, who they dealt with, which systems they used, what
is still open, and the questions they used to answer. The main model writes it from the dossier,
their open tasks and the last 90 days of channel summaries that name them; if the model is
unavailable the same sections are filled straight from the dossier (`fallback_handover`).
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import date, timedelta

from .i18n import CATALOG, LANG_NAMES, t
from .memory_access import is_dm_scope
from .people import MAX_HANDOVER, Dossier

DIGEST_DAYS = 90
DIGEST_CHARS = 9000
MAX_TASK_LINES = 25

STRINGS = {
    "ho_title": {"th": "เอกสารส่งมอบงาน: {name}", "en": "Handover: {name}"},
    "ho_role": {"th": "ตำแหน่งและหน้าที่", "en": "Role"},
    "ho_resp": {"th": "สิ่งที่รับผิดชอบ", "en": "Responsibilities"},
    "ho_recurring": {"th": "งานประจำและวิธีทำ", "en": "Recurring duties and how to do them"},
    "ho_contacts": {"th": "คนและหน่วยงานที่ต้องติดต่อ", "en": "People and parties they dealt with"},
    "ho_systems": {"th": "ระบบและเครื่องมือที่ใช้", "en": "Systems and tools"},
    "ho_open": {"th": "งานที่ยังค้างอยู่", "en": "Open items"},
    "ho_faq": {"th": "คำถามที่เคยตอบให้คนอื่น", "en": "Questions they used to answer"},
    "ho_none": {"th": "ยังไม่มีข้อมูล", "en": "Nothing recorded yet"},
    "ho_left": {"th": "ออกจากบริษัทเมื่อ {day}", "en": "Left on {day}"},
    "ho_how": {"th": "วิธีทำงาน", "en": "How they worked"},
}
CATALOG.update(STRINGS)


def _lang_name(lang: str) -> str:
    return LANG_NAMES.get(lang, ("en", "English"))[1]


def facts_text(d: Dossier, limit: int = 40) -> str:
    """The whole dossier as compact lines for a prompt (the caller decides who may see it)."""

    def lines(label: str, items) -> list[str]:
        return [f"{label}: {x.text}" for x in sorted(items, key=lambda i: i.last_seen)[-limit:]]

    out = [f"Name: {d.name}", f"Role: {d.role_title or '?'} / Department: {d.department or '?'}"]
    out += lines("Responsible for", d.responsibilities)
    out += lines("Recurring", d.recurring)
    out += lines("Works like", d.how_they_work)
    out += lines("Uses", d.systems)
    out += [
        f"Knows: {x.topic} ({x.score:g})"
        for x in sorted(d.expertise, key=lambda x: -x.score)[:limit]
    ]
    out += [
        f"Deals with: {c.who}" + (f" about {c.about}" if c.about else "") + f" ({c.count}x)"
        for c in sorted(d.contacts, key=lambda c: -c.count)[:limit]
    ]
    return "\n".join(out)


def handover_prompt(d: Dossier, tasks: list[str], digest: str, lang: str, left_day: str) -> str:
    return f"""You write the HANDOVER document for a company: {d.name} has left (on {left_day}) and \
someone new will take over their work. Use ONLY the material below; never invent. If a section has \
no material write one line saying nothing is recorded yet.
Language: {_lang_name(lang)}. Format: Markdown, one "## " heading per section, short bullet points.

Sections, in this order:
1. {t("ho_role", lang)}
2. {t("ho_resp", lang)}
3. {t("ho_recurring", lang)} (for each duty: the steps, tools and people involved, how often)
4. {t("ho_contacts", lang)} (who, and what for)
5. {t("ho_systems", lang)}
6. {t("ho_open", lang)}
7. {t("ho_faq", lang)}

Dossier (learned from the work over time):
{facts_text(d)}

Open tasks assigned to {d.name}:
{chr(10).join(tasks) if tasks else "(none)"}

Recent channel summaries that mention {d.name} (last {DIGEST_DAYS} days):
{digest or "(none)"}"""


def fallback_handover(d: Dossier, tasks: list[str], lang: str) -> str:
    """The same sections straight from the dossier (no model)."""
    none = t("ho_none", lang)

    def bullets(rows: list[str]) -> str:
        return "\n".join(f"- {r}" for r in rows) if rows else f"- {none}"

    role = " / ".join(x for x in (d.role_title, d.department) if x)
    parts = [
        f"## {t('ho_role', lang)}\n{bullets([role] if role else [])}",
        f"## {t('ho_resp', lang)}\n{bullets([f.text for f in d.responsibilities[-20:]])}",
        f"## {t('ho_recurring', lang)}\n{bullets([f.text for f in d.recurring[-20:]] + [f.text for f in d.how_they_work[-10:]])}",
        f"## {t('ho_contacts', lang)}\n"
        + bullets(
            [
                c.who + (f": {c.about}" if c.about else "")
                for c in sorted(d.contacts, key=lambda c: -c.count)[:20]
            ]
        ),
        f"## {t('ho_systems', lang)}\n{bullets([f.text for f in d.systems[-20:]])}",
        f"## {t('ho_open', lang)}\n{bullets(tasks[:MAX_TASK_LINES])}",
        f"## {t('ho_faq', lang)}\n{bullets([x.topic for x in sorted(d.expertise, key=lambda x: -x.score)[:15]])}",
    ]
    return "\n\n".join(parts)[:MAX_HANDOVER]


def evidence_channels(d: Dossier) -> list[int]:
    """Every real channel the dossier was learned from (never a DM scope, never 0)."""
    seen: set[int] = set()
    for items in (
        d.responsibilities,
        d.how_they_work,
        d.systems,
        d.recurring,
        d.expertise,
        d.contacts,
    ):
        for it in items:
            for e in it.evidence:
                if e.channel_id > 0:
                    seen.add(e.channel_id)
    return sorted(seen)


def collect_digest(
    store,
    guild_id: int,
    channel_ids: Collection[int],
    name: str,
    today: date,
    *,
    names: dict[int, str] | None = None,
    days: int = DIGEST_DAYS,
    max_chars: int = DIGEST_CHARS,
) -> str:
    """Lines of the day summaries (of `channel_ids`, last `days` days) that name the person:
    topics they took part in, action items that are theirs, decisions that mention them."""
    chans = [c for c in channel_ids if not is_dm_scope(c)]
    if not chans or not name.strip():
        return ""
    low = name.casefold()
    names = names or {}
    start = (today - timedelta(days=days)).isoformat()
    try:
        rows = store.summaries(guild_id, set(chans), start, today.isoformat())
    except Exception:
        return ""
    out: list[str] = []
    used = 0
    for s in sorted(rows, key=lambda r: (r.local_date, r.channel_id)):
        sm = s.summary or {}
        where = f"{s.local_date} #{names.get(s.channel_id, s.channel_id)}"
        lines: list[str] = []
        for tp in sm.get("topics") or []:
            if any(low in str(p).casefold() for p in tp.get("people") or []):
                lines.append(f"Topic: {tp.get('title', '')}: {tp.get('details', '')}")
        for a in sm.get("action_items") or []:
            if low in str(a.get("who", "")).casefold():
                lines.append(f"Action: {a.get('what', '')} {a.get('due', '')}".strip())
        for x in sm.get("decisions") or []:
            if low in str(x).casefold():
                lines.append(f"Decision: {x}")
        for ln in lines:
            row = f"- {where} {ln}"[:400]
            if used + len(row) > max_chars:
                return "\n".join(out)
            out.append(row)
            used += len(row) + 1
    return "\n".join(out)
