from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from .models import Receipt

GROUPS = {
    "category": "category",
    "vendor": "COALESCE(vendor,'(ไม่ทราบ)')",
    "submitter": "submitter_name",
    "day": "effective_date",
    "none": "'all'",
}
UPDATABLE = {
    "vendor",
    "receipt_date",
    "currency",
    "subtotal",
    "vat",
    "total",
    "category",
    "payment_method",
    "notes",
    "confidence",
}
LOW_CONF = 0.7


class Store(Protocol):
    def add_receipt(self, r: Receipt) -> Receipt: ...
    def get_by_sha(self, sha256: str) -> Receipt | None: ...
    def get(self, receipt_id: int) -> Receipt | None: ...
    def get_by_reply_message(self, message_id: int) -> Receipt | None: ...
    def set_reply_message(self, receipt_id: int, message_id: int) -> None: ...
    def update_receipt(self, receipt_id: int, fields: dict) -> Receipt: ...
    def find_possible_duplicate(self, r: Receipt) -> Receipt | None: ...
    def summarize(
        self, start: str, end: str, group_by: str = "none", submitter_id: int | None = None
    ) -> dict: ...
    def list_receipts(
        self,
        start: str,
        end: str,
        category: str | None = None,
        vendor_contains: str | None = None,
        submitter: str | None = None,
        limit: int = 20,
        submitter_id: int | None = None,
    ) -> list[Receipt]: ...
    def flagged(
        self, start: str | None = None, end: str | None = None, submitter_id: int | None = None
    ) -> list[Receipt]: ...


SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  submitter_id INTEGER NOT NULL, submitter_name TEXT NOT NULL,
  channel_id INTEGER, message_id INTEGER, reply_message_id INTEGER,
  sha256 TEXT NOT NULL UNIQUE,
  vendor TEXT, receipt_date TEXT, effective_date TEXT NOT NULL, currency TEXT DEFAULT 'THB',
  subtotal REAL, vat REAL, total REAL, category TEXT, items TEXT, payment_method TEXT,
  confidence REAL, notes TEXT, context TEXT, possible_duplicate_of INTEGER, created_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_date ON receipts(effective_date);
CREATE INDEX IF NOT EXISTS ix_sub ON receipts(submitter_id);
"""


class SQLiteStore:
    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    @staticmethod
    def _row(row) -> Receipt | None:
        if row is None:
            return None
        d = dict(row)
        d["items"] = json.loads(d["items"] or "[]")
        return Receipt(**d)

    def add_receipt(self, r: Receipt) -> Receipt:
        d = r.model_dump(exclude={"id"})
        d["items"] = json.dumps([i for i in d["items"]], ensure_ascii=False)
        cols = ",".join(d)
        with self._conn() as c:
            cur = c.execute(
                f"INSERT INTO receipts ({cols}) VALUES ({','.join(':' + k for k in d)})", d
            )
            rid = cur.lastrowid
        return self.get(rid)

    def get_by_sha(self, sha256):
        with self._conn() as c:
            return self._row(
                c.execute("SELECT * FROM receipts WHERE sha256=?", (sha256,)).fetchone()
            )

    def get(self, receipt_id):
        with self._conn() as c:
            return self._row(
                c.execute("SELECT * FROM receipts WHERE id=?", (receipt_id,)).fetchone()
            )

    def get_by_reply_message(self, message_id):
        with self._conn() as c:
            return self._row(
                c.execute(
                    "SELECT * FROM receipts WHERE reply_message_id=?", (message_id,)
                ).fetchone()
            )

    def set_reply_message(self, receipt_id, message_id):
        with self._conn() as c:
            c.execute("UPDATE receipts SET reply_message_id=? WHERE id=?", (message_id, receipt_id))

    def update_receipt(self, receipt_id, fields):
        bad = set(fields) - UPDATABLE
        if bad:
            raise ValueError(f"fields not updatable: {bad}")
        if fields:
            f = dict(fields)
            if "receipt_date" in f and f["receipt_date"]:
                f["effective_date"] = f["receipt_date"]
            sets = ",".join(f"{k}=:{k}" for k in f)
            with self._conn() as c:
                c.execute(f"UPDATE receipts SET {sets} WHERE id=:_id", {**f, "_id": receipt_id})
        return self.get(receipt_id)

    def find_possible_duplicate(self, r):
        if not (r.vendor and r.receipt_date and r.total is not None):
            return None
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM receipts WHERE lower(vendor)=lower(?) AND receipt_date=? AND total=? AND sha256<>? ORDER BY id LIMIT 1",
                (r.vendor, r.receipt_date, r.total, r.sha256),
            ).fetchone()
        return self._row(row)

    def summarize(self, start, end, group_by="none", submitter_id=None):
        if group_by not in GROUPS:
            raise ValueError(f"bad group_by {group_by}")
        key = GROUPS[group_by]
        where, p = "effective_date BETWEEN ? AND ?", [start, end]
        if submitter_id is not None:
            where += " AND submitter_id=?"
            p.append(submitter_id)
        order = "key" if group_by == "day" else "total DESC"
        with self._conn() as c:
            rows = c.execute(
                f"SELECT {key} AS key, ROUND(COALESCE(SUM(total),0),2) AS total, COUNT(*) AS count FROM receipts WHERE {where} GROUP BY key ORDER BY {order}",
                p,
            ).fetchall()
        groups = [dict(r) for r in rows]
        return {
            "start": start,
            "end": end,
            "group_by": group_by,
            "total": round(sum(g["total"] for g in groups), 2),
            "count": sum(g["count"] for g in groups),
            "groups": [] if group_by == "none" else groups,
        }

    def list_receipts(
        self,
        start,
        end,
        category=None,
        vendor_contains=None,
        submitter=None,
        limit=20,
        submitter_id=None,
    ):
        where, p = ["effective_date BETWEEN ? AND ?"], [start, end]
        if category:
            where.append("category=?")
            p.append(category)
        if vendor_contains:
            where.append("vendor LIKE ?")
            p.append(f"%{vendor_contains}%")
        if submitter:
            where.append("submitter_name LIKE ?")
            p.append(f"%{submitter}%")
        if submitter_id is not None:
            where.append("submitter_id=?")
            p.append(submitter_id)
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM receipts WHERE {' AND '.join(where)} ORDER BY effective_date DESC, id DESC LIMIT ?",
                [*p, max(1, min(int(limit), 100))],
            ).fetchall()
        return [self._row(r) for r in rows]

    def flagged(self, start=None, end=None, submitter_id=None):
        where, p = (
            ["(confidence < ? OR total IS NULL OR possible_duplicate_of IS NOT NULL)"],
            [LOW_CONF],
        )
        if start and end:
            where.append("effective_date BETWEEN ? AND ?")
            p += [start, end]
        if submitter_id is not None:
            where.append("submitter_id=?")
            p.append(submitter_id)
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM receipts WHERE {' AND '.join(where)} ORDER BY effective_date DESC LIMIT 50",
                p,
            ).fetchall()
        return [self._row(r) for r in rows]


def make_store(settings) -> Store:
    if settings.store == "firestore":
        raise NotImplementedError("Firestore store not implemented in MVP; use STORE=sqlite")
    return SQLiteStore(settings.db_path)
