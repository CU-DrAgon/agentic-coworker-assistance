"""Firestore-backed receipt ledger (same `Store` protocol as `SQLiteStore`).

Layout (all in the project's `(default)` database, nothing else is touched):

* `{collection}/{sha256}`     one receipt per document; the doc id is the file's SHA-256, so a
                              re-upload is rejected atomically (`create()` raises AlreadyExists).
* `{collection}_meta/counter` `{last_id: n}`: allocates the small integer ids shown to users
                              ("#12") inside a transaction.

Receipt fields are stored as-is; `effective_date` is an ISO `YYYY-MM-DD` string so lexicographic
range queries work.

Scale note: Firestore does the date-range filter server-side (one field, so only the automatic
single-field index is needed). Everything else (submitter scope, category, vendor, grouping,
sums) is done in Python over that result. That is fine at shop scale (tens of receipts per day,
a few thousand per year) but loads every receipt in the range into memory; revisit with
server-side aggregation queries if a ledger grows far beyond that. `flagged()` without a date
range scans the whole collection for the same reason.

No composite indexes are required: every query has at most one range/equality filter.
"""

from __future__ import annotations

from typing import Any

from google.api_core.exceptions import AlreadyExists
from google.cloud.firestore_v1.base_query import FieldFilter

from .models import Receipt
from .store import GROUPS, LOW_CONF, UPDATABLE

UNKNOWN_VENDOR = "(ไม่ทราบ)"


def _group_key(r: Receipt, group_by: str) -> str:
    return {
        "category": r.category,
        "vendor": r.vendor or UNKNOWN_VENDOR,
        "submitter": r.submitter_name,
        "day": r.effective_date,
        "none": "all",
    }[group_by]


class FirestoreStore:
    def __init__(
        self,
        client: Any = None,
        collection: str = "salid_receipts",
        project: str | None = None,
    ) -> None:
        self._client = client
        self._project = project
        self.collection = collection
        self.meta_collection = f"{collection}_meta"

    @property
    def client(self) -> Any:
        if self._client is None:
            from google.cloud import firestore

            self._client = firestore.Client(project=self._project)
        return self._client

    @property
    def _col(self) -> Any:
        return self.client.collection(self.collection)

    # ------------------------------------------------------------ helpers
    def _next_id(self) -> int:
        from google.cloud import firestore

        ref = self.client.collection(self.meta_collection).document("counter")

        @firestore.transactional
        def bump(txn):
            snap = ref.get(transaction=txn)
            n = ((snap.to_dict() or {}).get("last_id", 0) if snap.exists else 0) + 1
            txn.set(ref, {"last_id": n})
            return n

        return bump(self.client.transaction())

    def _where(self, field: str, op: str, value: Any) -> Any:
        return self._col.where(filter=FieldFilter(field, op, value))

    def _snap_by(self, field: str, value: Any) -> Any:
        for snap in self._where(field, "==", value).limit(1).stream():
            return snap
        return None

    @staticmethod
    def _receipt(snap) -> Receipt | None:
        return Receipt(**snap.to_dict()) if snap is not None else None

    def _in_range(self, start: str, end: str, submitter_id: int | None) -> list[Receipt]:
        q = self._where("effective_date", ">=", start).where(
            filter=FieldFilter("effective_date", "<=", end)
        )
        rows = [Receipt(**s.to_dict()) for s in q.stream()]
        if submitter_id is not None:  # client-side: avoids a composite index
            rows = [r for r in rows if r.submitter_id == submitter_id]
        return rows

    # ------------------------------------------------------------ writes
    def add_receipt(self, r: Receipt) -> Receipt:
        rid = self._next_id()
        data = r.model_copy(update={"id": rid}).model_dump()
        try:
            self._col.document(r.sha256).create(data)
        except AlreadyExists as e:
            raise ValueError(f"receipt with sha256 {r.sha256} already exists") from e
        return Receipt(**data)

    def update_receipt(self, receipt_id: int, fields: dict) -> Receipt:
        bad = set(fields) - UPDATABLE
        if bad:
            raise ValueError(f"fields not updatable: {bad}")
        snap = self._snap_by("id", receipt_id)
        if snap is None:
            raise KeyError(f"receipt {receipt_id} not found")
        f = dict(fields)
        if f.get("receipt_date"):
            f["effective_date"] = f["receipt_date"]
        if f:
            snap.reference.update(f)
        return Receipt(**{**snap.to_dict(), **f})

    def set_reply_message(self, receipt_id: int, message_id: int) -> None:
        snap = self._snap_by("id", receipt_id)
        if snap is not None:
            snap.reference.update({"reply_message_id": message_id})

    # ------------------------------------------------------------ point reads
    def get_by_sha(self, sha256: str) -> Receipt | None:
        snap = self._col.document(sha256).get()
        return Receipt(**snap.to_dict()) if snap.exists else None

    def get(self, receipt_id: int) -> Receipt | None:
        return self._receipt(self._snap_by("id", receipt_id))

    def get_by_reply_message(self, message_id: int) -> Receipt | None:
        return self._receipt(self._snap_by("reply_message_id", message_id))

    def find_possible_duplicate(self, r: Receipt) -> Receipt | None:
        if not (r.vendor and r.receipt_date and r.total is not None):
            return None
        same_day = (
            Receipt(**s.to_dict())
            for s in self._where("receipt_date", "==", r.receipt_date).stream()
        )
        hits = [
            x
            for x in same_day
            if x.sha256 != r.sha256
            and x.total == r.total
            and (x.vendor or "").casefold() == r.vendor.casefold()
        ]
        return min(hits, key=lambda x: x.id or 0, default=None)

    # ------------------------------------------------------------ range reads
    def summarize(
        self, start: str, end: str, group_by: str = "none", submitter_id: int | None = None
    ) -> dict:
        if group_by not in GROUPS:
            raise ValueError(f"bad group_by {group_by}")
        acc: dict[str, list[float]] = {}
        for r in self._in_range(start, end, submitter_id):
            g = acc.setdefault(_group_key(r, group_by), [0.0, 0])
            g[0] += r.total or 0.0
            g[1] += 1
        groups = [{"key": k, "total": round(t, 2), "count": c} for k, (t, c) in acc.items()]
        groups.sort(key=(lambda g: g["key"]) if group_by == "day" else (lambda g: -g["total"]))
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
        start: str,
        end: str,
        category: str | None = None,
        vendor_contains: str | None = None,
        submitter: str | None = None,
        limit: int = 20,
        submitter_id: int | None = None,
    ) -> list[Receipt]:
        rows = self._in_range(start, end, submitter_id)
        if category:
            rows = [r for r in rows if r.category == category]
        if vendor_contains:
            rows = [r for r in rows if vendor_contains.casefold() in (r.vendor or "").casefold()]
        if submitter:
            rows = [r for r in rows if submitter.casefold() in r.submitter_name.casefold()]
        rows.sort(key=lambda r: (r.effective_date, r.id or 0), reverse=True)
        return rows[: max(1, min(int(limit), 100))]

    def flagged(
        self, start: str | None = None, end: str | None = None, submitter_id: int | None = None
    ) -> list[Receipt]:
        if start and end:
            rows = self._in_range(start, end, submitter_id)
        else:  # whole-collection scan, see the scale note in the module docstring
            rows = [Receipt(**s.to_dict()) for s in self._col.stream()]
            if submitter_id is not None:
                rows = [r for r in rows if r.submitter_id == submitter_id]
        rows = [
            r
            for r in rows
            if r.confidence < LOW_CONF or r.total is None or r.possible_duplicate_of is not None
        ]
        rows.sort(key=lambda r: r.effective_date, reverse=True)
        return rows[:50]
