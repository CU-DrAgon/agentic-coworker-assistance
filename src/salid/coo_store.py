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
from typing import Any, Protocol

TASKS = "salid_tasks"
DISCLOSURES = "salid_disclosures"
STATE = "salid_coo_state"
PREFS = "salid_notif_prefs"  # per (guild, user) briefing / alert preferences
ASKS = "salid_asks"  # messages that mention a member and ask for something
ALERTS = "salid_doc_alerts"  # important documents and who they concern
DELIVERIES = "salid_deliveries"  # files Salid forwarded for people (audit log)
PROFILES = "salid_profiles"  # per (guild, user) personal preferences (salid.profiles)


class DocStore(Protocol):
    def put(self, kind: str, guild_id: int, key: str, doc: dict[str, Any]) -> None: ...
    def get(self, kind: str, guild_id: int, key: str) -> dict[str, Any] | None: ...
    def delete(self, kind: str, guild_id: int, key: str) -> None: ...
    def find(
        self, kind: str, guild_id: int, where: Callable[[dict], bool] | None = None
    ) -> list[dict[str, Any]]:
        """All docs of the guild in this kind (optionally filtered), oldest key first."""
        ...

    def find_any(self, kind: str, doc_key: str) -> dict[str, Any] | None:
        """One doc by key across guilds (a button click knows only its request id)."""
        ...


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
        with self._conn() as c:
            rows = c.execute(
                "SELECT doc FROM salid_docs WHERE kind=? AND guild_id=? ORDER BY key",
                (kind, guild_id),
            ).fetchall()
        docs = [json.loads(r[0]) for r in rows]
        return [d for d in docs if where is None or where(d)]

    def find_any(self, kind, doc_key):
        with self._conn() as c:
            row = c.execute(
                "SELECT doc FROM salid_docs WHERE kind=? AND key=?", (kind, doc_key)
            ).fetchone()
        return json.loads(row[0]) if row else None


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
        from google.cloud.firestore_v1.base_query import FieldFilter

        q = self.client.collection(kind).where(filter=FieldFilter("guild_id", "==", guild_id))
        docs = sorted(((s.to_dict() or {}) for s in q.stream()), key=lambda d: d.get("_key", ""))
        return [self._clean(d) for d in docs if where is None or where(d)]

    def find_any(self, kind, doc_key):
        from google.cloud.firestore_v1.base_query import FieldFilter

        q = self.client.collection(kind).where(filter=FieldFilter("_key", "==", doc_key))
        for s in q.stream():
            return self._clean(s.to_dict())
        return None


def make_doc_store(settings, store=None) -> DocStore:
    if settings.store == "firestore":
        return FirestoreDocStore(project=settings.gcp_project)
    return SQLiteDocStore(getattr(store, "path", None) or ":memory:")
