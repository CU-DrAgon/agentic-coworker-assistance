"""Seed a demo Salid ledger: python scripts/salid_seed_demo.py /path/to.db (dates relative to today)."""

import sys
from datetime import date, timedelta

from salid.models import Receipt
from salid.store import SQLiteStore

ALICE, BOB = 100, 200


def add(store, n, uid, name, vendor, day, total, cat, conf=0.9):
    return store.add_receipt(
        Receipt(
            submitter_id=uid, submitter_name=name, sha256=f"demo{n}", vendor=vendor,
            receipt_date=day, effective_date=day, total=total, category=cat,
            confidence=conf, created_at=day,
        )
    )  # fmt: skip


def main(path: str) -> None:
    store = SQLiteStore(path)
    t = date.today()

    def d(n):
        return (t - timedelta(days=n)).isoformat()

    rows = [
        ("Makro", d(0), 2450, "วัตถุดิบ", ALICE, "Alice"),
        ("Grab", d(1), 180, "ค่าเดินทาง", ALICE, "Alice"),
        ("ตลาดสด", d(2), 1320, "วัตถุดิบ", BOB, "Bob"),
        ("การไฟฟ้า", d(3), 5400, "สาธารณูปโภค", BOB, "Bob"),
        ("Makro", d(9), 1800, "วัตถุดิบ", ALICE, "Alice"),
        ("ตลาดสด", d(10), 900, "วัตถุดิบ", BOB, "Bob"),
    ]
    for i, (v, day, tot, cat, uid, name) in enumerate(rows):
        add(store, i, uid, name, v, day, float(tot), cat)
    add(store, 99, BOB, "Bob", "น้ำแข็ง", d(0), 120.0, "วัตถุดิบ", conf=0.4)
    print("seeded", path)


if __name__ == "__main__":
    main(sys.argv[1])
