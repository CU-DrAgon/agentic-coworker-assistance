"""Totals must never add different currencies (no FX); a missing currency counts as THB."""

from conftest import ALICE, GUILD

from salid.deep_task import _aggregates
from salid.digest import build_daily_digest
from salid.models import Receipt
from salid.qa import LedgerTools, fmt_totals, money

RANGE = ("2026-10-01", "2026-10-31")


def add(store, n, total, cur, day="2026-10-10", vendor="V", dup_sha=None):
    return store.add_receipt(
        Receipt(
            guild_id=GUILD,
            submitter_id=ALICE,
            submitter_name="Alice",
            sha256=dup_sha or f"c{n}",
            vendor=vendor,
            receipt_date=day,
            effective_date=day,
            total=total,
            currency=cur,
            category="อื่นๆ",
            confidence=0.9,
            created_at=day,
        )
    )


def seed_mixed(store):
    add(store, 1, 184885.54, "THB", vendor="Big")
    for i, amt in enumerate([1.99, 100, 10, 20]):
        add(store, 10 + i, amt, "USD", vendor=f"Cloud{i}")


def test_summarize_per_currency(store):
    seed_mixed(store)
    s = store.summarize(GUILD, *RANGE)
    assert s["total"] is None
    assert s["totals_by_currency"] == {"THB": 184885.54, "USD": 131.99}
    assert s["count"] == 5
    g = store.summarize(GUILD, *RANGE, "category")["groups"]
    assert g[0]["total"] is None and g[0]["totals_by_currency"]["USD"] == 131.99


def test_single_currency_keeps_total_and_missing_is_thb(store):
    add(store, 1, 100.0, "THB")
    add(store, 2, 50.0, "")
    add(store, 3, 5.0, "thb")
    s = store.summarize(GUILD, *RANGE, "vendor")
    assert s["total"] == 155.0 and s["totals_by_currency"] == {"THB": 155.0}
    assert s["groups"][0]["total"] == 155.0


def test_group_sort_and_day_groups_mixed(store):
    add(store, 1, 50.0, "USD", day="2026-10-02", vendor="A")
    add(store, 2, 900.0, "THB", day="2026-10-01", vendor="B")
    add(store, 3, 20.0, "USD", day="2026-10-01", vendor="B")
    days = store.summarize(GUILD, *RANGE, "day")["groups"]
    assert [d["key"] for d in days] == ["2026-10-01", "2026-10-02"]
    assert days[0]["totals_by_currency"] == {"THB": 900.0, "USD": 20.0}
    vendors = store.summarize(GUILD, *RANGE, "vendor")["groups"]
    assert [v["key"] for v in vendors] == ["B", "A"]


def test_money_formatting():
    assert money(1.5, "THB") == "฿1.50" and money(1000, None) == "฿1,000"
    assert money(1.99, "USD") == "$1.99"
    assert money(1234.5, "EUR") == "1,234.50 EUR"
    assert fmt_totals({"THB": 184885.54, "USD": 131.99}) == "฿184,885.54, $131.99"


def test_tool_output_formatting(store):
    seed_mixed(store)
    t = LedgerTools(store, GUILD)
    s = t.spending_summary(*RANGE, "category")
    assert s["total"] is None
    assert s["total_fmt"] == "฿184,885.54, $131.99"
    assert s["groups"][0]["total_fmt"] == "฿184,885.54, $131.99"
    rows = t.list_receipts(*RANGE)["receipts"]
    usd = [r for r in rows if r["currency"] == "USD"]
    assert usd and all("฿" not in r["total_fmt"] and r["total_fmt"].startswith("$") for r in usd)


def test_compare_periods_per_currency(store):
    add(store, 1, 100.0, "THB", day="2026-10-10")
    add(store, 2, 10.0, "USD", day="2026-10-11")
    add(store, 3, 80.0, "THB", day="2026-09-10")
    t = LedgerTools(store, GUILD)
    c = t.compare_periods("2026-10-01", "2026-10-31", "2026-09-01", "2026-09-30")
    assert c["period_a"]["total_fmt"] == "฿100, $10.00"
    assert c["overall"]["diff"] is None
    assert c["overall"]["by_currency"]["THB"]["diff_fmt"] == "+฿20"
    assert c["overall"]["by_currency"]["USD"]["diff_fmt"] == "+$10.00"
    # single currency keeps the flat fields
    one = t.compare_periods("2026-10-10", "2026-10-10", "2026-09-01", "2026-09-30")
    assert one["overall"]["diff_fmt"] == "+฿20"


def test_duplicate_deduction_per_currency(store):
    add(store, 1, 100.0, "USD", vendor="Same", dup_sha="x1")
    add(store, 2, 100.0, "USD", vendor="Same", dup_sha="x2")
    add(store, 3, 7.0, "THB", vendor="Other", dup_sha="x3")
    add(store, 4, 7.0, "THB", vendor="Other", dup_sha="x4")
    d = LedgerTools(store, GUILD).find_duplicates()
    assert d["total_deduct_by_currency"] == {"THB": 7.0, "USD": 100.0}
    assert d["total_deduct_fmt"] == "฿7, $100.00"


def test_digest_mixed_currencies(store):
    seed_mixed(store)
    add(store, 20, 100.0, "USD", vendor="Cloud1", dup_sha="d1")
    text = build_daily_digest(store, GUILD, "2026-10-10")
    assert "฿184,885.54, $231.99" in text
    assert "$100.00" in text and "อาจซ้ำ 1 รายการ รวม $100.00" in text
    assert "฿185" not in text.replace("฿184,885.54", "")


def test_deep_task_aggregates_mixed(store):
    seed_mixed(store)
    agg = _aggregates(store.all_receipts(GUILD, *RANGE))
    assert agg["overall"]["total"] is None
    assert agg["overall"]["totals_by_currency"] == {"THB": 184885.54, "USD": 131.99}
    assert "NEVER add" in agg["currency_note"]
