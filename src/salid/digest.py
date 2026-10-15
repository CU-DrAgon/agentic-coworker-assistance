from __future__ import annotations

from datetime import date

from .i18n import t
from .qa import baht
from .store import Store, duplicate_extras


def _duplicates_line(store: Store, guild_id: int, lang: str) -> str | None:
    """Possible duplicates still counted in the totals (the extra copies of every group)."""
    try:
        extras = duplicate_extras(store.duplicate_groups(guild_id))
    except Exception:  # the digest must go out even if the duplicate scan fails
        return None
    if not extras:
        return None
    amount = sum(r.total or 0.0 for r in extras if r.currency == "THB")
    return t("digest_dups", lang, n=len(extras), amount=baht(amount))


def build_daily_digest(store: Store, guild_id: int, day: date | str, lang: str = "th") -> str:
    d = day.isoformat() if isinstance(day, date) else day
    total = store.summarize(guild_id, d, d, "none")
    dup_line = _duplicates_line(store, guild_id, lang)
    if total["count"] == 0:
        none = f"{t('digest_title', lang, day=d)}\n{t('digest_none', lang)}"
        return f"{none}\n\n{dup_line}" if dup_line else none
    cats = store.summarize(guild_id, d, d, "category")["groups"]
    vendors = store.summarize(guild_id, d, d, "vendor")["groups"][:3]
    flagged = store.flagged(guild_id, d, d)
    lines = [
        t("digest_title", lang, day=d),
        t("digest_total", lang, total=baht(total["total"]), count=total["count"]),
        "",
        t("digest_by_category", lang),
    ]
    lines += [f"• {g['key']}: {baht(g['total'])} ({g['count']})" for g in cats]
    lines += ["", t("digest_top_vendors", lang)]
    lines += [f"{i}. {g['key']}: {baht(g['total'])}" for i, g in enumerate(vendors, 1)]
    if flagged:
        lines += ["", t("digest_flagged", lang, n=len(flagged))]
        for r in flagged[:10]:
            why = t("digest_dup" if r.possible_duplicate_of else "digest_unclear", lang)
            lines.append(
                t(
                    "digest_flag_line",
                    lang,
                    id=r.id,
                    vendor=r.vendor or t("unknown", lang),
                    amount=baht(r.total) if r.total is not None else t("digest_no_total", lang),
                    why=why,
                    who=r.submitter_name,
                )
            )
    if dup_line:
        lines += ["", dup_line]
    return "\n".join(lines)
