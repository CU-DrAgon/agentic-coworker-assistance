"""Firestore-backed receipt ledger (same `Store` protocol as `SQLiteStore`).

Layout (all in the project's `(default)` database, nothing else is touched):

* `{collection}/{guild_id}_{sha256}`
                              one receipt per document; the doc id is the guild plus the file's
                              SHA-256, so a re-upload in the same guild is rejected atomically
                              (`create()` raises AlreadyExists) while another guild may store the
                              same file.
* `{collection}_meta/counter` `{last_id: n}`: allocates the small integer ids shown to users
                              ("#12") inside a transaction.

Receipt fields are stored as-is; `effective_date` is an ISO `YYYY-MM-DD` string so lexicographic
range queries work.

Tenancy: every receipt carries `guild_id` (int) and every query starts with the equality filter
`guild_id == X`, so one server's data is never read for another.

Scale note: Firestore does only that guild filter server-side (one field, so only the automatic
single-field index is needed). Everything else (date range, submitter scope, category, vendor,
grouping, sums) is done in Python over the guild's receipts. That is fine at shop scale (tens of
receipts per day, a few thousand per year) but loads the guild's whole ledger into memory; if one
ledger grows far beyond that, add the composite index (guild_id, effective_date) and push the
range filter into the query.

No composite indexes are required: every query has a single equality filter.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from google.api_core.exceptions import AlreadyExists
from google.cloud.firestore_v1.base_query import FieldFilter

from .models import Receipt
from .store import (
    ACTIVE,
    GROUPS,
    LOW_CONF,
    UPDATABLE,
    VOID,
    build_summary,
    cur_of,
    find_duplicate_in,
    group_duplicates,
    is_active,
)

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

    def _guild(self, guild_id: int) -> Any:
        return self._where("guild_id", "==", guild_id)

    def _snap_by(self, field: str, value: Any) -> Any:
        for snap in self._where(field, "==", value).limit(1).stream():
            return snap
        return None

    @staticmethod
    def _receipt(snap) -> Receipt | None:
        return Receipt(**snap.to_dict()) if snap is not None else None

    def _in_range(
        self,
        guild_id: int,
        start: str,
        end: str,
        submitter_id: int | None,
        include_void: bool = False,
    ) -> list[Receipt]:
        rows = [Receipt(**s.to_dict()) for s in self._guild(guild_id).stream()]
        if not include_void:  # a missing `status` field means active
            rows = [r for r in rows if is_active(r)]
        rows = [r for r in rows if start <= r.effective_date <= end]  # client-side: no index
        if submitter_id is not None:
            rows = [r for r in rows if r.submitter_id == submitter_id]
        return rows

    # ------------------------------------------------------------ writes
    def add_receipt(self, r: Receipt) -> Receipt:
        rid = self._next_id()
        data = r.model_copy(update={"id": rid}).model_dump()
        try:
            self._col.document(f"{r.guild_id}_{r.sha256}").create(data)
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
    def get_by_sha(self, guild_id: int, sha256: str) -> Receipt | None:
        snap = self._col.document(f"{guild_id}_{sha256}").get()
        return Receipt(**snap.to_dict()) if snap.exists else None

    def get(self, receipt_id: int) -> Receipt | None:
        return self._receipt(self._snap_by("id", receipt_id))

    def get_by_reply_message(self, message_id: int) -> Receipt | None:
        return self._receipt(self._snap_by("reply_message_id", message_id))

    def find_possible_duplicate(self, r: Receipt) -> Receipt | None:
        if r.total is None:
            return None
        return find_duplicate_in(
            r, (Receipt(**s.to_dict()) for s in self._guild(r.guild_id).stream())
        )

    def duplicate_groups(
        self,
        guild_id: int,
        start: str | None = None,
        end: str | None = None,
        submitter_id: int | None = None,
    ) -> list[list[Receipt]]:
        rows = self._in_range(guild_id, start or "", end or "\uffff", submitter_id)
        return group_duplicates(rows)

    def set_status(
        self, receipt_id: int, status: str, reason: str | None = None, by: str | None = None
    ) -> Receipt:
        if status not in (ACTIVE, VOID):
            raise ValueError(f"bad status {status}")
        snap = self._snap_by("id", receipt_id)
        if snap is None:
            raise KeyError(f"receipt {receipt_id} not found")
        void = status == VOID
        f = {
            "status": status,
            "void_reason": reason if void else None,
            "voided_by": by if void else None,
            "voided_at": datetime.now(UTC).isoformat(timespec="seconds") if void else None,
        }
        snap.reference.update(f)
        return Receipt(**{**snap.to_dict(), **f})

    # ------------------------------------------------------------ range reads
    def summarize(
        self,
        guild_id: int,
        start: str,
        end: str,
        group_by: str = "none",
        submitter_id: int | None = None,
    ) -> dict:
        if group_by not in GROUPS:
            raise ValueError(f"bad group_by {group_by}")
        acc: dict[tuple[str, str], list[float]] = {}
        for r in self._in_range(guild_id, start, end, submitter_id):
            g = acc.setdefault((_group_key(r, group_by), cur_of(r)), [0.0, 0])
            g[0] += r.total or 0.0
            g[1] += 1
        return build_summary(start, end, group_by, [(k, c, t, n) for (k, c), (t, n) in acc.items()])

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
    ) -> list[Receipt]:
        rows = self._in_range(guild_id, start, end, submitter_id, include_void)
        if category:
            rows = [r for r in rows if r.category == category]
        if vendor_contains:
            rows = [r for r in rows if vendor_contains.casefold() in (r.vendor or "").casefold()]
        if submitter:
            rows = [r for r in rows if submitter.casefold() in r.submitter_name.casefold()]
        rows.sort(key=lambda r: (r.effective_date, r.id or 0), reverse=True)
        return rows[: max(1, min(int(limit), 100))]

    def flagged(
        self,
        guild_id: int,
        start: str | None = None,
        end: str | None = None,
        submitter_id: int | None = None,
        include_void: bool = False,
    ) -> list[Receipt]:
        if start and end:
            rows = self._in_range(guild_id, start, end, submitter_id, include_void)
        else:  # the guild's whole ledger, see the scale note in the module docstring
            rows = self._in_range(guild_id, "", "\uffff", submitter_id, include_void)
        rows = [
            r
            for r in rows
            if r.confidence < LOW_CONF or r.total is None or r.possible_duplicate_of is not None
        ]
        rows.sort(key=lambda r: r.effective_date, reverse=True)
        return rows[:50]

    def all_receipts(
        self,
        guild_id: int,
        start: str,
        end: str,
        submitter_id: int | None = None,
        limit: int = 5000,
        include_void: bool = False,
    ) -> list[Receipt]:
        rows = self._in_range(guild_id, start, end, submitter_id, include_void)
        rows.sort(key=lambda r: (r.effective_date, r.id or 0))
        return rows[:limit]

    def recent(self, guild_id: int, limit: int = 20, include_void: bool = False) -> list[Receipt]:
        """Newest receipts first (by id). Reads the guild's ledger, see the scale note above."""
        rows = [Receipt(**s.to_dict()) for s in self._guild(guild_id).stream()]
        if not include_void:
            rows = [r for r in rows if is_active(r)]
        rows.sort(key=lambda r: r.id or 0, reverse=True)
        return rows[: max(1, min(int(limit), 100))]

    def receipt_channel_counts(self, guild_id: int, since: str) -> dict[int, int]:
        """Active receipts per channel id since a date (reads the guild's ledger, see above)."""
        counts: dict[int, int] = {}
        for snap in self._guild(guild_id).stream():
            r = Receipt(**snap.to_dict())
            if is_active(r) and r.channel_id and r.effective_date >= since:
                counts[r.channel_id] = counts.get(r.channel_id, 0) + 1
        return counts
