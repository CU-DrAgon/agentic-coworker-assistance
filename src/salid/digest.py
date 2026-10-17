from __future__ import annotations

from datetime import date

from .extract import category_label
from .i18n import t
from .qa import fmt_totals, money
from .store import Store, cur_of, duplicate_extras


def _duplicates_line(store: Store, guild_id: int, lang: str) -> str | None:
    """Possible duplicates still counted in the totals (the extra copies of every group)."""
    try:
        extras = duplicate_extras(store.duplicate_groups(guild_id))
    except Exception:  # the digest must go out even if the duplicate scan fails
        return None
    if not extras:
        return None
    by: dict[str, float] = {}
    for r in extras:  # per currency, never mixed
        by[cur_of(r)] = by.get(cur_of(r), 0.0) + (r.total or 0.0)
    amount = fmt_totals({c: round(v, 2) for c, v in sorted(by.items())})
    return t("digest_dups", lang, n=len(extras), amount=amount)


def build_daily_digest(store: Store, guild_id: int, day: date | str, lang: str = "en") -> str:
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
        t(
            "digest_total",
            lang,
            total=fmt_totals(total["totals_by_currency"]),
            count=total["count"],
        ),
        "",
        t("digest_by_category", lang),
    ]
    lines += [
        f"• {category_label(g['key'], lang)}: {fmt_totals(g['totals_by_currency'])} ({g['count']})"
        for g in cats
    ]
    lines += ["", t("digest_top_vendors", lang)]
    lines += [
        f"{i}. {g['key']}: {fmt_totals(g['totals_by_currency'])}" for i, g in enumerate(vendors, 1)
    ]
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
                    amount=money(r.total, r.currency)
                    if r.total is not None
                    else t("digest_no_total", lang),
                    why=why,
                    who=r.submitter_name,
                )
            )
    if dup_line:
        lines += ["", dup_line]
    return "\n".join(lines)
