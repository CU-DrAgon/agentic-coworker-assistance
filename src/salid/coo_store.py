"""A tiny document store for Salid's coordinator data: tasks, disclosure grants, bot state.

One interface (`DocStore`), two backends chosen by `SALID_STORE` like the rest of Salid: SQLite
(one table, the ledger's database file) and Firestore (one collection per kind, doc id
`{guild_id}_{key}`). Every read starts from a guild id; filtering beyond that happens in Python
(guild scale, no composite indexes), the same approach as the memory store.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, TypeVar

T = TypeVar("T")
SQLITE_BUSY_RETRIES = 4

TASKS = "salid_tasks"
DISCLOSURES = "salid_disclosures"
STATE = "salid_coo_state"
PREFS = "salid_notif_prefs"  # per (guild, user) briefing / alert preferences
ASKS = "salid_asks"  # messages that mention a member and ask for something
ALERTS = "salid_doc_alerts"  # important documents and who they concern
DELIVERIES = "salid_deliveries"  # files Salid forwarded for people (audit log)
MINUTES = "salid_meeting_minutes"  # summary text of each finished voice meeting (minutes.py)
AVAILABILITY = "salid_availability"  # per (guild, user) work hours and busy blocks (freetime.py)
APPROVALS = "salid_approvals"  # approval requests (salid.approvals)
PRIVATE_CHATS = "salid_private_chats"  # per (guild, user) private Salid channel (private_chat)
PROFILES = "salid_profiles"  # per (guild, user) personal preferences (salid.profiles)
COMPANY = "salid_company"  # per guild playbook, key "playbook" (salid.company_brain)
CONTESTS = "salid_contests"  # competitions / hackathons / grants, key = contest id (salid.contests)
QUOTES = "salid_quotes"  # quotations, key = base number QT-YYYY-NNNN + counters (salid.quotation)
IDEAS = "salid_ideas"  # idea / project documents, key `{owner_id}_{slug}` (salid.ideas)
IDEA_IMAGES = "salid_idea_images"  # concept images (base64 JPEG) of an idea, key `{idea key}_{sha}`


class DocStore(Protocol):
    def put(self, kind: str, guild_id: int, key: str, doc: dict[str, Any]) -> None: ...
    def get(self, kind: str, guild_id: int, key: str) -> dict[str, Any] | None: ...
    def delete(self, kind: str, guild_id: int, key: str) -> None: ...
    def find(
        self, kind: str, guild_id: int, where: Callable[[dict], bool] | None = None
    ) -> list[dict[str, Any]]:
        """All docs of the guild in this kind (optionally filtered), oldest key first."""
        ...

    def find_items(
        self, kind: str, guild_id: int, where: Callable[[dict], bool] | None = None
    ) -> list[tuple[str, dict[str, Any]]]:
        """Like `find`, as (key, doc) pairs, for docs that do not carry their own key."""
        ...

    def find_any(self, kind: str, doc_key: str) -> dict[str, Any] | None:
        """One doc by key across guilds (a button click knows only its request id)."""
        ...

    def transact(self, fn: Callable[[Tx], T]) -> T:
        """Run `fn(tx)` as ONE atomic read-modify-write that is safe across processes.

        `tx` offers get / put / delete (same arguments as the store). `fn` may run more than
        once (Firestore retries a contended transaction), so it must have no side effects other
        than through `tx`. Any exception rolls everything back and propagates."""
        ...


class Tx(Protocol):
    def get(self, kind: str, guild_id: int, key: str) -> dict[str, Any] | None: ...
    def put(self, kind: str, guild_id: int, key: str, doc: dict[str, Any]) -> None: ...
    def delete(self, kind: str, guild_id: int, key: str) -> None: ...


class _SqliteTx:
    """The transaction handle of `SQLiteDocStore.transact`: one connection, writes immediate."""

    def __init__(self, conn: sqlite3.Connection):
        self._c = conn

    def get(self, kind, guild_id, key):
        row = self._c.execute(
            "SELECT doc FROM salid_docs WHERE kind=? AND guild_id=? AND key=?",
            (kind, guild_id, key),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, kind, guild_id, key, doc) -> None:
        self._c.execute(
            "INSERT OR REPLACE INTO salid_docs (kind, guild_id, key, doc) VALUES (?,?,?,?)",
            (kind, guild_id, key, json.dumps(doc, ensure_ascii=False)),
        )

    def delete(self, kind, guild_id, key) -> None:
        self._c.execute(
            "DELETE FROM salid_docs WHERE kind=? AND guild_id=? AND key=?", (kind, guild_id, key)
        )


class SQLiteDocStore:
    def __init__(self, path: str):
        self.path = path
        self._shared = None
        self._lock = threading.RLock()
        if path == ":memory:":
            self._shared = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS salid_docs (kind TEXT NOT NULL, guild_id INTEGER NOT "
                "NULL, key TEXT NOT NULL, doc TEXT NOT NULL, PRIMARY KEY (kind, guild_id, key))"
            )

    @contextmanager
    def _conn(self):
        with self._lock:
            if self._shared is not None:
                yield self._shared
                self._shared.commit()
                return
            c = sqlite3.connect(self.path, timeout=10)
            try:
                yield c
                c.commit()
            finally:
                c.close()

    def put(self, kind, guild_id, key, doc) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO salid_docs (kind, guild_id, key, doc) VALUES (?,?,?,?)",
                (kind, guild_id, key, json.dumps(doc, ensure_ascii=False)),
            )

    def get(self, kind, guild_id, key):
        with self._conn() as c:
            row = c.execute(
                "SELECT doc FROM salid_docs WHERE kind=? AND guild_id=? AND key=?",
                (kind, guild_id, key),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def delete(self, kind, guild_id, key) -> None:
        with self._conn() as c:
            c.execute(
                "DELETE FROM salid_docs WHERE kind=? AND guild_id=? AND key=?",
                (kind, guild_id, key),
            )

    def find(self, kind, guild_id, where=None):
        return [doc for _, doc in self.find_items(kind, guild_id, where)]

    def find_items(self, kind, guild_id, where=None):
        with self._conn() as c:
            rows = c.execute(
                "SELECT key, doc FROM salid_docs WHERE kind=? AND guild_id=? ORDER BY key",
                (kind, guild_id),
            ).fetchall()
        items = [(key, json.loads(doc)) for key, doc in rows]
        return [(k, d) for k, d in items if where is None or where(d)]

    def find_any(self, kind, doc_key):
        with self._conn() as c:
            row = c.execute(
                "SELECT doc FROM salid_docs WHERE kind=? AND key=?", (kind, doc_key)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def transact(self, fn):
        """`BEGIN IMMEDIATE` takes the database write lock, so two processes (or two store
        instances) can never interleave inside `fn`. A busy database is retried a few times."""
        with self._lock:
            for attempt in range(SQLITE_BUSY_RETRIES):
                try:
                    return self._transact_once(fn)
                except sqlite3.OperationalError as exc:
                    busy = "locked" in str(exc) or "busy" in str(exc)
                    if not busy or attempt == SQLITE_BUSY_RETRIES - 1:
                        raise
        raise AssertionError("unreachable")  # pragma: no cover

    def _transact_once(self, fn):
        shared = self._shared
        c = shared if shared is not None else sqlite3.connect(self.path, timeout=10)
        try:
            c.execute("BEGIN IMMEDIATE")
            try:
                result = fn(_SqliteTx(c))
            except BaseException:
                c.rollback()
                raise
            c.commit()
            return result
        finally:
            if shared is None:
                c.close()


class _FirestoreTx:
    """Reads go through the Firestore transaction; writes are queued and applied by the
    transaction on commit (Firestore wants every read before any write). Reads see the
    queued writes of this same transaction."""

    def __init__(self, store: FirestoreDocStore, txn: Any):
        self._store, self._txn = store, txn
        self._pending: dict[tuple[str, int, str], dict | None] = {}

    def get(self, kind, guild_id, key):
        k = (kind, guild_id, key)
        if k in self._pending:
            doc = self._pending[k]
            return None if doc is None else {kk: v for kk, v in doc.items() if kk != "_key"}
        snap = self._store._ref(kind, guild_id, key).get(transaction=self._txn)
        return self._store._clean(snap.to_dict()) if snap.exists else None

    def put(self, kind, guild_id, key, doc) -> None:
        self._pending[(kind, guild_id, key)] = {**doc, "guild_id": guild_id, "_key": key}

    def delete(self, kind, guild_id, key) -> None:
        self._pending[(kind, guild_id, key)] = None

    def flush(self) -> None:
        for (kind, guild_id, key), doc in self._pending.items():
            ref = self._store._ref(kind, guild_id, key)
            if doc is None:
                self._txn.delete(ref)
            else:
                self._txn.set(ref, doc)


class FirestoreDocStore:
    """Collection = kind, doc id = `{guild_id}_{key}`, plus `guild_id` and `_key` fields."""

    def __init__(self, client: Any = None, project: str | None = None):
        self._client, self._project = client, project

    @property
    def client(self) -> Any:
        if self._client is None:
            from google.cloud import firestore

            self._client = firestore.Client(project=self._project)
        return self._client

    def _ref(self, kind, guild_id, key):
        return self.client.collection(kind).document(f"{guild_id}_{key}")

    def put(self, kind, guild_id, key, doc) -> None:
        self._ref(kind, guild_id, key).set({**doc, "guild_id": guild_id, "_key": key})

    def get(self, kind, guild_id, key):
        snap = self._ref(kind, guild_id, key).get()
        return self._clean(snap.to_dict()) if snap.exists else None

    def delete(self, kind, guild_id, key) -> None:
        self._ref(kind, guild_id, key).delete()

    @staticmethod
    def _clean(d):
        return {k: v for k, v in d.items() if k != "_key"}

    def find(self, kind, guild_id, where=None):
        return [doc for _, doc in self.find_items(kind, guild_id, where)]

    def find_items(self, kind, guild_id, where=None):
        from google.cloud.firestore_v1.base_query import FieldFilter

        q = self.client.collection(kind).where(filter=FieldFilter("guild_id", "==", guild_id))
        docs = sorted(((s.to_dict() or {}) for s in q.stream()), key=lambda d: d.get("_key", ""))
        return [(d.get("_key", ""), self._clean(d)) for d in docs if where is None or where(d)]

    def find_any(self, kind, doc_key):
        from google.cloud.firestore_v1.base_query import FieldFilter

        q = self.client.collection(kind).where(filter=FieldFilter("_key", "==", doc_key))
        for s in q.stream():
            return self._clean(s.to_dict())
        return None

    def transact(self, fn):
        """A Firestore transaction (optimistic: a conflicting commit re-runs `fn`)."""
        from google.cloud import firestore

        @firestore.transactional
        def run(txn):
            tx = _FirestoreTx(self, txn)
            result = fn(tx)
            tx.flush()
            return result

        return run(self.client.transaction())


def make_doc_store(settings, store=None) -> DocStore:
    if settings.store == "firestore":
        return FirestoreDocStore(project=settings.gcp_project)
    return SQLiteDocStore(getattr(store, "path", None) or ":memory:")
