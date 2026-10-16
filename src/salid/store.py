from __future__ import annotations

import json
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import UTC, date, datetime
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

DEFAULT_CURRENCY = "THB"  # a receipt with a missing currency counts as THB


def cur_of(r: Receipt) -> str:
    return (r.currency or DEFAULT_CURRENCY).strip().upper() or DEFAULT_CURRENCY


def _single_total(by: dict[str, float]) -> float | None:
    """The plain total only when it is unambiguous (zero or one currency); never mix currencies."""
    if not by:
        return 0.0
    return next(iter(by.values())) if len(by) == 1 else None


def _size_key(g: dict) -> tuple:
    by = g["totals_by_currency"]
    return (-by.get(DEFAULT_CURRENCY, 0.0), *(-v for c, v in sorted(by.items()) if c != "THB"))


def build_summary(start, end, group_by, rows) -> dict:
    """Shared by every store. `rows` = (group key, currency, total, count) per key+currency.
    Amounts are summed PER CURRENCY (no FX conversion): `totals_by_currency` always;
    `total` is a number only when at most one currency is present, otherwise None."""
    acc: dict[str, dict] = {}
    for key, cur, total, count in rows:
        cur = (cur or DEFAULT_CURRENCY).strip().upper() or DEFAULT_CURRENCY
        g = acc.setdefault(key, {"key": key, "by": {}, "count": 0})
        g["by"][cur] = g["by"].get(cur, 0.0) + (total or 0.0)
        g["count"] += count
    groups = []
    for g in acc.values():
        by = {c: round(v, 2) for c, v in sorted(g["by"].items())}
        groups.append(
            {
                "key": g["key"],
                "total": _single_total(by),
                "totals_by_currency": by,
                "count": g["count"],
            }
        )
    if group_by == "day":
        groups.sort(key=lambda g: g["key"])
    else:  # biggest first, comparing THB first and then the other currencies
        groups.sort(key=_size_key)
    overall: dict[str, float] = {}
    for g in groups:
        for c, v in g["totals_by_currency"].items():
            overall[c] = overall.get(c, 0.0) + v
    overall = {c: round(v, 2) for c, v in sorted(overall.items())}
    return {
        "start": start,
        "end": end,
        "group_by": group_by,
        "total": _single_total(overall),
        "totals_by_currency": overall,
        "count": sum(g["count"] for g in groups),
        "groups": [] if group_by == "none" else groups,
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

# ---- duplicate detection (one deterministic rule shared by both stores)
DUP_DAYS = 3  # receipt dates up to this many days apart can still be the same bill
DUP_AMOUNT_TOL = 0.01
DUP_TOKEN_OVERLAP = 0.5
DUP_MIN_CONTAINS = (
    3  # a vendor name must be at least this long (normalized) to match by containment
)
ACTIVE, VOID = "active", "void"


def is_active(r: Receipt) -> bool:
    return (r.status or ACTIVE) != VOID


def _vendor_tokens(name: str | None) -> list[str]:
    """Lower-cased alphanumeric words (Thai combining marks kept); punctuation splits words."""
    out, cur = [], []
    for ch in (name or "").casefold():
        if ch.isalnum() or unicodedata.category(ch).startswith("M"):
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def normalize_vendor(name: str | None) -> str:
    return "".join(_vendor_tokens(name))


def vendors_similar(a: str | None, b: str | None) -> bool:
    """Case/punctuation/space-insensitive equal, one contains the other, or token overlap >= 0.5
    (overlap measured against the shorter name): "NVIDIA Corporation" ~ "NVIDIA Corp"."""
    na, nb = normalize_vendor(a), normalize_vendor(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    short, long_ = sorted((na, nb), key=len)
    if len(short) >= DUP_MIN_CONTAINS and short in long_:
        return True
    ta, tb = set(_vendor_tokens(a)), set(_vendor_tokens(b))
    return len(ta & tb) / min(len(ta), len(tb)) >= DUP_TOKEN_OVERLAP


def _same_invoice(a: Receipt, b: Receipt) -> bool:
    x, y = normalize_vendor(a.invoice_no), normalize_vendor(b.invoice_no)
    return bool(x) and x == y


def _days_apart(a: Receipt, b: Receipt) -> int | None:
    try:
        return abs(
            (
                date.fromisoformat(a.effective_date[:10])
                - date.fromisoformat(b.effective_date[:10])
            ).days
        )
    except ValueError:
        return None


def looks_duplicate(a: Receipt, b: Receipt) -> bool:
    """Same guild, same currency and total (within 0.01), dates within 3 days, and either a
    similar vendor or the same invoice number. Pure and symmetric; ignores status."""
    if a.guild_id != b.guild_id or a.total is None or b.total is None:
        return False
    if (a.currency or "THB").upper() != (b.currency or "THB").upper():
        return False
    if abs(a.total - b.total) > DUP_AMOUNT_TOL + 1e-9:
        return False
    gap = _days_apart(a, b)
    if gap is None or gap > DUP_DAYS:
        return False
    return vendors_similar(a.vendor, b.vendor) or _same_invoice(a, b)


def find_duplicate_in(r: Receipt, candidates) -> Receipt | None:
    """Oldest ACTIVE candidate (other file than `r`) that looks like the same bill."""
    hits = [
        x for x in candidates if is_active(x) and x.sha256 != r.sha256 and looks_duplicate(r, x)
    ]
    return min(hits, key=lambda x: x.id or 0, default=None)


def group_duplicates(receipts) -> list[list[Receipt]]:
    """Groups (size >= 2) of ACTIVE receipts that look like one bill: connected components of the
    pairwise rule plus stored `possible_duplicate_of` links. Each group is oldest first (date,
    id); groups are ordered by their oldest member."""
    rows = sorted(
        (r for r in receipts if is_active(r)), key=lambda r: (r.effective_date, r.id or 0)
    )
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    by_id = {r.id: i for i, r in enumerate(rows)}
    for i, r in enumerate(rows):
        if r.possible_duplicate_of in by_id:
            union(i, by_id[r.possible_duplicate_of])
        for j in range(i + 1, len(rows)):
            if looks_duplicate(r, rows[j]):
                union(i, j)
    comps: dict[int, list[Receipt]] = {}
    for i, r in enumerate(rows):
        comps.setdefault(find(i), []).append(r)
    groups = [g for g in comps.values() if len(g) > 1]
    groups.sort(key=lambda g: (g[0].effective_date, g[0].id or 0))
    return groups


def pick_keep(group: list[Receipt]) -> Receipt:
    """The copy that stays: a PDF original first, then the higher confidence, then the earliest."""
    return min(
        group,
        key=lambda r: (
            not (r.original_uri or "").lower().endswith(".pdf"),
            -(r.confidence or 0.0),
            r.effective_date,
            r.id or 0,
        ),
    )


def duplicate_extras(groups: list[list[Receipt]]) -> list[Receipt]:
    """Every receipt that would be deducted: all but the kept one of each group."""
    out = []
    for g in groups:
        keep = pick_keep(g)
        out += [r for r in g if r is not keep]
    return out


class Store(Protocol):
    """Receipt ledger. The tenant is the Discord server: every read that returns or aggregates
    receipts takes `guild_id` first, and dedupe (sha256, vendor+date+total) is per guild. Lookups
    by receipt id or reply-message id are global (those ids are unique across guilds)."""

    def add_receipt(self, r: Receipt) -> Receipt: ...
    def get_by_sha(self, guild_id: int, sha256: str) -> Receipt | None: ...
    def get(self, receipt_id: int) -> Receipt | None: ...
    def get_by_reply_message(self, message_id: int) -> Receipt | None: ...
    def set_reply_message(self, receipt_id: int, message_id: int) -> None: ...
    def update_receipt(self, receipt_id: int, fields: dict) -> Receipt: ...
    def find_possible_duplicate(self, r: Receipt) -> Receipt | None: ...
    def duplicate_groups(
        self,
        guild_id: int,
        start: str | None = None,
        end: str | None = None,
        submitter_id: int | None = None,
    ) -> list[list[Receipt]]: ...
    def set_status(
        self,
        receipt_id: int,
        status: str,
        reason: str | None = None,
        by: str | None = None,
    ) -> Receipt: ...
    def summarize(
        self,
        guild_id: int,
        start: str,
        end: str,
        group_by: str = "none",
        submitter_id: int | None = None,
    ) -> dict: ...
    def list_receipts(
        self,
        guild_id: int,
        start: str,
        end: str,
        category: str | None = None,
        vendor_contains: str | None = None,
        submitter: str | None = None,
        limit: int = 20,
        submitter_id: int | None = None,
        include_void: bool = False,
    ) -> list[Receipt]: ...
    def flagged(
        self,
        guild_id: int,
        start: str | None = None,
        end: str | None = None,
        submitter_id: int | None = None,
        include_void: bool = False,
    ) -> list[Receipt]: ...
    def all_receipts(
        self,
        guild_id: int,
        start: str,
        end: str,
        submitter_id: int | None = None,
        limit: int = 5000,
        include_void: bool = False,
    ) -> list[Receipt]: ...
    def recent(
        self, guild_id: int, limit: int = 20, include_void: bool = False
    ) -> list[Receipt]: ...


SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  guild_id INTEGER NOT NULL DEFAULT 0,
  submitter_id INTEGER NOT NULL, submitter_name TEXT NOT NULL,
  channel_id INTEGER, message_id INTEGER, reply_message_id INTEGER,
  sha256 TEXT NOT NULL,
  vendor TEXT, receipt_date TEXT, effective_date TEXT NOT NULL, currency TEXT DEFAULT 'THB',
  subtotal REAL, vat REAL, total REAL, category TEXT, items TEXT, payment_method TEXT,
  confidence REAL, notes TEXT, context TEXT, possible_duplicate_of INTEGER, created_at TEXT, original_uri TEXT,
  invoice_no TEXT, status TEXT NOT NULL DEFAULT 'active', void_reason TEXT, voided_by TEXT, voided_at TEXT,
  UNIQUE (guild_id, sha256)
);
CREATE INDEX IF NOT EXISTS ix_date ON receipts(effective_date);
CREATE INDEX IF NOT EXISTS ix_sub ON receipts(submitter_id);
"""
NEW_COLUMNS = {
    "invoice_no": "TEXT",
    "status": "TEXT NOT NULL DEFAULT 'active'",
    "void_reason": "TEXT",
    "voided_by": "TEXT",
    "voided_at": "TEXT",
}
ACTIVE_SQL = "COALESCE(status,'active')<>'void'"
GUILD_INDEX = "CREATE INDEX IF NOT EXISTS ix_guild_date ON receipts(guild_id, effective_date)"


class SQLiteStore:
    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            cols = {row["name"] for row in c.execute("PRAGMA table_info(receipts)")}
            if "original_uri" not in cols:  # ledgers created before receipt archiving
                c.execute("ALTER TABLE receipts ADD COLUMN original_uri TEXT")
                cols.add("original_uri")
            if "guild_id" not in cols:  # ledgers created before multi-tenancy
                self._migrate_to_guilds(c, cols)
            c.execute(GUILD_INDEX)
            cols = {row["name"] for row in c.execute("PRAGMA table_info(receipts)")}
            for col, ddl in NEW_COLUMNS.items():  # ledgers created before void/duplicate support
                if col not in cols:
                    c.execute(f"ALTER TABLE receipts ADD COLUMN {col} {ddl}")

    @staticmethod
    def _migrate_to_guilds(c, old_cols: set[str]) -> None:
        """Rebuild the table: `guild_id` (existing rows get 0) and UNIQUE(guild_id, sha256)
        instead of a global UNIQUE(sha256), which SQLite cannot alter in place."""
        c.executescript("DROP INDEX IF EXISTS ix_date; DROP INDEX IF EXISTS ix_sub;")
        c.execute("ALTER TABLE receipts RENAME TO receipts_old")
        c.executescript(SCHEMA)
        shared = ",".join(sorted(old_cols))
        c.execute(f"INSERT INTO receipts ({shared}) SELECT {shared} FROM receipts_old")
        c.execute("DROP TABLE receipts_old")

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

    def get_by_sha(self, guild_id, sha256):
        with self._conn() as c:
            return self._row(
                c.execute(
                    "SELECT * FROM receipts WHERE guild_id=? AND sha256=?", (guild_id, sha256)
                ).fetchone()
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
        if r.total is None:
            return None
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM receipts WHERE guild_id=? AND {ACTIVE_SQL} AND total BETWEEN ? AND ?",
                (r.guild_id, r.total - 0.02, r.total + 0.02),
            ).fetchall()
        return find_duplicate_in(r, (self._row(x) for x in rows))

    def duplicate_groups(self, guild_id, start=None, end=None, submitter_id=None):
        where, p = ["guild_id=?", ACTIVE_SQL, "total IS NOT NULL"], [guild_id]
        if start and end:
            where.append("effective_date BETWEEN ? AND ?")
            p += [start, end]
        if submitter_id is not None:
            where.append("submitter_id=?")
            p.append(submitter_id)
        with self._conn() as c:
            rows = c.execute(f"SELECT * FROM receipts WHERE {' AND '.join(where)}", p).fetchall()
        return group_duplicates(self._row(x) for x in rows)

    def set_status(self, receipt_id, status, reason=None, by=None):
        if status not in (ACTIVE, VOID):
            raise ValueError(f"bad status {status}")
        void = status == VOID
        at = datetime.now(UTC).isoformat(timespec="seconds") if void else None
        with self._conn() as c:
            c.execute(
                "UPDATE receipts SET status=?, void_reason=?, voided_by=?, voided_at=? WHERE id=?",
                (status, reason if void else None, by if void else None, at, receipt_id),
            )
        return self.get(receipt_id)

    def summarize(self, guild_id, start, end, group_by="none", submitter_id=None):
        if group_by not in GROUPS:
            raise ValueError(f"bad group_by {group_by}")
        key = GROUPS[group_by]
        where, p = (
            f"guild_id=? AND effective_date BETWEEN ? AND ? AND {ACTIVE_SQL}",
            [
                guild_id,
                start,
                end,
            ],
        )
        if submitter_id is not None:
            where += " AND submitter_id=?"
            p.append(submitter_id)
        with self._conn() as c:
            rows = c.execute(
                f"SELECT {key} AS key, COALESCE(NULLIF(UPPER(TRIM(currency)),''),'THB') AS cur, "
                f"COALESCE(SUM(total),0) AS total, COUNT(*) AS count FROM receipts WHERE {where} "
                "GROUP BY key, cur",
                p,
            ).fetchall()
        return build_summary(start, end, group_by, [(r[0], r[1], r[2], r[3]) for r in rows])

    def list_receipts(
        self,
        guild_id,
        start,
        end,
        category=None,
        vendor_contains=None,
        submitter=None,
        limit=20,
        submitter_id=None,
        include_void=False,
    ):
        where, p = ["guild_id=?", "effective_date BETWEEN ? AND ?"], [guild_id, start, end]
        if not include_void:
            where.append(ACTIVE_SQL)
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

    def flagged(self, guild_id, start=None, end=None, submitter_id=None, include_void=False):
        where, p = (
            [
                "guild_id=?",
                "(confidence < ? OR total IS NULL OR possible_duplicate_of IS NOT NULL)",
            ],
            [guild_id, LOW_CONF],
        )
        if not include_void:
            where.append(ACTIVE_SQL)
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

    def all_receipts(self, guild_id, start, end, submitter_id=None, limit=5000, include_void=False):
        """Unpaged export in date order (used by deep tasks); `limit` is a safety cap."""
        where, p = ["guild_id=?", "effective_date BETWEEN ? AND ?"], [guild_id, start, end]
        if not include_void:
            where.append(ACTIVE_SQL)
        if submitter_id is not None:
            where.append("submitter_id=?")
            p.append(submitter_id)
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM receipts WHERE {' AND '.join(where)} ORDER BY effective_date, id LIMIT ?",
                [*p, limit],
            ).fetchall()
        return [self._row(r) for r in rows]

    def recent(self, guild_id, limit=20, include_void=False):
        """Newest receipts first (by id); used by the hub activity feed."""
        extra = "" if include_void else f" AND {ACTIVE_SQL}"
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM receipts WHERE guild_id=?{extra} ORDER BY id DESC LIMIT ?",
                (guild_id, max(1, min(int(limit), 100))),
            ).fetchall()
        return [self._row(r) for r in rows]


def make_store(settings) -> Store:
    if settings.store == "firestore":
        from .firestore_store import FirestoreStore

        return FirestoreStore(
            collection=settings.firestore_collection, project=settings.gcp_project
        )
    if settings.store != "sqlite":
        raise ValueError(f"SALID_STORE must be sqlite or firestore, got {settings.store!r}")
    return SQLiteStore(settings.db_path)
