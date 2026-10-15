"""Storage for the secretary features: scheduled meetings (`salid_meetings`) and emails
(`salid_emails`). One tiny document-store interface with two backends, matching the rest of
Salid: SQLite (local dev, one JSON column per document) and Firestore (one document per record,
a `guild_id` field on each, every query is a single equality filter so no composite index is
needed). Records are plain dicts; the typed helpers live on `SecretaryStore`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

MEETINGS = "salid_meetings"
EMAILS = "salid_emails"


class DocBackend(Protocol):
    def put(self, collection: str, doc_id: str, data: dict) -> None: ...
    def get(self, collection: str, doc_id: str) -> dict | None: ...
    def query(self, collection: str, guild_id: int | None = None) -> list[dict]: ...


class SqliteDocs:
    """JSON documents in one table. A single shared connection keeps `:memory:` usable."""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._c: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _conn(self) -> sqlite3.Connection:
        """Opened on first use, so constructing the store never touches the disk."""
        if self._c is None:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._c = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
            self._c.execute(
                "CREATE TABLE IF NOT EXISTS secretary_docs ("
                "collection TEXT NOT NULL, id TEXT NOT NULL, guild_id INTEGER, data TEXT NOT NULL,"
                " PRIMARY KEY (collection, id))"
            )
            self._c.execute(
                "CREATE INDEX IF NOT EXISTS ix_secretary_guild "
                "ON secretary_docs(collection, guild_id)"
            )
            self._c.commit()
        return self._c

    def put(self, collection: str, doc_id: str, data: dict) -> None:
        with self._lock:
            c = self._conn()
            c.execute(
                "INSERT INTO secretary_docs(collection, id, guild_id, data) VALUES (?,?,?,?) "
                "ON CONFLICT(collection, id) DO UPDATE SET guild_id=excluded.guild_id, "
                "data=excluded.data",
                (collection, doc_id, data.get("guild_id"), json.dumps(data, ensure_ascii=False)),
            )
            c.commit()

    def get(self, collection: str, doc_id: str) -> dict | None:
        with self._lock:
            row = (
                self._conn()
                .execute(
                    "SELECT data FROM secretary_docs WHERE collection=? AND id=?",
                    (collection, doc_id),
                )
                .fetchone()
            )
        return json.loads(row[0]) if row else None

    def query(self, collection: str, guild_id: int | None = None) -> list[dict]:
        sql, args = "SELECT data FROM secretary_docs WHERE collection=?", [collection]
        if guild_id is not None:
            sql += " AND guild_id=?"
            args.append(guild_id)
        with self._lock:
            rows = self._conn().execute(sql, args).fetchall()
        return [json.loads(r[0]) for r in rows]


class FirestoreDocs:
    def __init__(self, client: Any = None, project: str | None = None) -> None:
        self._client, self._project = client, project

    @property
    def client(self) -> Any:
        if self._client is None:
            from google.cloud import firestore

            self._client = firestore.Client(project=self._project)
        return self._client

    def put(self, collection: str, doc_id: str, data: dict) -> None:
        self.client.collection(collection).document(doc_id).set(data)

    def get(self, collection: str, doc_id: str) -> dict | None:
        snap = self.client.collection(collection).document(doc_id).get()
        return snap.to_dict() if snap.exists else None

    def query(self, collection: str, guild_id: int | None = None) -> list[dict]:
        col = self.client.collection(collection)
        if guild_id is not None:
            from google.cloud.firestore_v1.base_query import FieldFilter

            col = col.where(filter=FieldFilter("guild_id", "==", guild_id))
        return [s.to_dict() for s in col.stream()]


def new_id() -> str:
    return uuid.uuid4().hex[:10]


def utc_iso(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")


class SecretaryStore:
    """Meetings and emails on top of a `DocBackend`."""

    def __init__(self, backend: DocBackend) -> None:
        self.db = backend

    # ---- meetings
    def put_meeting(self, m: dict) -> dict:
        self.db.put(MEETINGS, m["id"], m)
        return m

    def get_meeting(self, meeting_id: str) -> dict | None:
        return self.db.get(MEETINGS, str(meeting_id))

    def meeting_by_event(self, guild_id: int, event_id: int | str) -> dict | None:
        return next(
            (
                m
                for m in self.db.query(MEETINGS, guild_id)
                if str(m.get("event_id")) == str(event_id)
            ),
            None,
        )

    def list_meetings(
        self,
        guild_id: int | None = None,
        start: str | None = None,
        end: str | None = None,
        statuses: tuple[str, ...] | None = None,
    ) -> list[dict]:
        """Meetings whose start is within [start, end) (ISO instants, compared as UTC)."""
        out = []
        for m in self.db.query(MEETINGS, guild_id):
            if statuses and m.get("status") not in statuses:
                continue
            s = m.get("start_utc", "")
            if (start and s < start) or (end and s >= end):
                continue
            out.append(m)
        return sorted(out, key=lambda m: m.get("start_utc", ""))

    # ---- emails
    def put_email(self, e: dict) -> dict:
        self.db.put(EMAILS, e["id"], e)
        return e

    def get_email(self, email_id: str) -> dict | None:
        return self.db.get(EMAILS, str(email_id))

    def count_emails_since(self, guild_id: int, since: datetime, statuses: tuple[str, ...]) -> int:
        cut = utc_iso(since)
        return sum(
            1
            for e in self.db.query(EMAILS, guild_id)
            if e.get("status") in statuses and (e.get("sent_at") or "") >= cut
        )


def make_secretary_store(settings, ledger=None) -> SecretaryStore:
    if settings.store == "firestore":
        return SecretaryStore(FirestoreDocs(project=settings.gcp_project))
    # Next to the receipt ledger; a ledger without a file (fakes) gets a throwaway database.
    path = getattr(ledger, "path", None) or (":memory:" if ledger is not None else settings.db_path)
    return SecretaryStore(SqliteDocs(path))


__all__ = [
    "EMAILS",
    "MEETINGS",
    "SecretaryStore",
    "SqliteDocs",
    "FirestoreDocs",
    "make_secretary_store",
    "new_id",
    "utc_iso",
]
