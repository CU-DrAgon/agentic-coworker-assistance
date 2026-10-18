"""Storage for Salid's organisational memory: the raw message log and per-channel day summaries.

Same pattern as the receipt ledger: a `MemoryStore` protocol with a SQLite implementation (local
dev) and a Firestore one (`salid.firestore_memory`), chosen by `SALID_STORE`.

Access control lives in the callers, but the store makes it hard to get wrong: every read takes a
`guild_id` and an explicit collection of allowed channel ids and returns nothing outside them (an
empty collection returns nothing). Retention is applied by `purge` from each guild's current setting (no stored expiry).
Timestamps are ISO-8601 UTC strings with microseconds, so they
compare correctly as text in both backends. `local_date` is the day in the guild's timezone
(`SALID_TZ`), computed by the caller when the message is recorded.

Ordering inside a channel-day is `(created_at, message_id)`; a summary remembers the last message
it covered as that pair, so "messages since the last update" is exact.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Collection, Iterable
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

KINDS = ("chat", "voice", "salid", "dm", "huddle")
MAX_DAYS = 400  # safety cap on a date range
MAX_ROWS = 5000  # safety cap on rows one read returns


class MemoryMessage(BaseModel):
    guild_id: int
    channel_id: int  # the text/voice channel used for access checks (a thread's parent)
    thread_id: int | None = None
    message_id: int
    author_id: int = 0
    author_name: str = ""
    created_at: datetime  # timezone-aware UTC
    local_date: str  # YYYY-MM-DD in the guild's timezone
    text: str = ""
    jump_url: str = ""
    edited_at: datetime | None = None
    kind: str = "chat"  # chat | voice | salid | dm | huddle
    visibility: str = "normal"  # normal | private (a DM note only its author may see)

    @property
    def key(self) -> tuple[str, int]:
        """Position inside a channel-day (the summary cursor)."""
        return (iso(self.created_at), self.message_id)


class DaySummary(BaseModel):
    guild_id: int
    channel_id: int
    local_date: str
    summary: dict[str, Any] = Field(default_factory=dict)  # see salid.memory_summary
    last_message_at: str | None = None  # created_at of the last message covered (ISO UTC)
    last_message_id: int = 0
    message_count: int = 0
    updated_at: str = ""
    final: bool = False
    dirty: bool = False  # a covered message was deleted: regenerate once from the raw log
    # Embeddings of the summary's items (salid.memory.vectorize), all optional: "<model>@<dim>",
    # one short hash per item (same order as memory_summary.summary_items), packed float32.
    vec_model: str = ""
    vec_hashes: list[str] = Field(default_factory=list)
    vecs: bytes = b""

    @property
    def cursor(self) -> tuple[str, int] | None:
        return (self.last_message_at, self.last_message_id) if self.last_message_at else None


class ChannelState(BaseModel):
    """Import progress of one channel or thread (resumable, idempotent).

    The log covers `[imported_from, last_message_at]` contiguously; `gap_cursor` is the progress
    of an in-flight import of an older gap (retention was lengthened); `summary_days` are past
    channel-days with >= 3 messages that still wait for their one final summary."""

    guild_id: int
    channel_id: int  # a thread's own id for threads
    parent_id: int | None = None
    imported_from: str = ""  # ISO UTC: start of the covered range
    last_message_at: str = ""  # ISO UTC: end of the covered range (the catch-up cursor)
    gap_cursor: str | None = None
    complete: bool = False  # the last run reached the end of both phases
    messages: int = 0  # messages imported by the history import (progress display)
    summary_days: list[str] = Field(default_factory=list)
    updated_at: str = ""


class GuildMemoryState(BaseModel):
    """Per-guild import accounting. Daily budgets reset when `day` changes."""

    guild_id: int
    day: str = ""  # local date the daily counters belong to
    images_done: int = 0  # historical images described today
    summaries_done: int = 0  # past-day summaries generated today
    channels_total: int = 0
    updated_at: str = ""


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="microseconds")


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def message_to_doc(m: MemoryMessage) -> dict[str, Any]:
    d = m.model_dump()
    d["created_at"] = iso(m.created_at)
    d["edited_at"] = iso(m.edited_at) if m.edited_at else None
    return d


def message_from_doc(d: dict[str, Any]) -> MemoryMessage:
    return MemoryMessage(**d)  # pydantic parses the ISO strings


def cutoff_date(cutoff: datetime, zone) -> str:
    """Summaries of days entirely before this local date are past retention."""
    return cutoff.astimezone(zone).date().isoformat()


def date_range(start: str, end: str) -> list[str]:
    """Inclusive list of ISO dates (empty when start > end; capped at MAX_DAYS)."""
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    n = min((b - a).days + 1, MAX_DAYS)
    return [(a + timedelta(days=i)).isoformat() for i in range(max(n, 0))]


def select_messages(
    rows: Iterable[MemoryMessage],
    *,
    channel_id: int | None,
    keyword: str | None,
    limit: int,
    newest_first: bool,
) -> list[MemoryMessage]:
    """Shared post-filter: optional single channel, case-insensitive substring, order, limit."""
    kw = (keyword or "").casefold().strip()
    out = [
        m
        for m in rows
        if (channel_id is None or m.channel_id == channel_id)
        and (not kw or kw in m.text.casefold() or kw in m.author_name.casefold())
    ]
    out.sort(key=lambda m: m.key, reverse=newest_first)
    return out[: max(1, min(int(limit), MAX_ROWS))]


class MemoryStore(Protocol):
    """Raw message log + day summaries. Every read is scoped by guild and allowed channels."""

    def add_message(self, m: MemoryMessage) -> None: ...
    def update_message_text(
        self, guild_id: int, message_id: int, text: str, edited_at: datetime | None
    ) -> bool:
        """Replace a record's text (`edited_at=None` keeps the edit time: used to add image
        descriptions). False when the record does not exist."""
        ...

    def set_visibility(self, guild_id: int, message_id: int, visibility: str) -> bool:
        """Mark a record "normal" or "private". False when the record does not exist."""
        ...

    def delete_messages(self, guild_id: int, message_ids: Collection[int]) -> list[tuple[int, str]]:
        """Delete records; return the (channel_id, local_date) of each that existed and mark
        those channel-days' summaries dirty."""
        ...

    def delete_guild(self, guild_id: int) -> int: ...
    def purge(self, cutoffs: dict[int, datetime], default_cutoff: datetime) -> tuple[int, int]:
        """Retention: delete messages created before the guild's cutoff (`default_cutoff` for
        guilds not listed) and day summaries of days before that cutoff's local date (the
        `local_date` of the cutoff, see `cutoff_date`). Returns (messages, summaries). The
        cutoffs are computed from each guild's current retention setting by the caller, so
        changing the setting applies to data already stored."""
        ...

    def messages(
        self,
        guild_id: int,
        allowed_channels: Collection[int],
        start_date: str,
        end_date: str,
        *,
        channel_id: int | None = None,
        keyword: str | None = None,
        limit: int = 50,
        newest_first: bool = True,
    ) -> list[MemoryMessage]: ...
    def messages_after(
        self,
        guild_id: int,
        channel_id: int,
        local_date: str,
        after: tuple[str, int] | None = None,
        limit: int = 500,
    ) -> list[MemoryMessage]:
        """Oldest first: the channel-day's messages after the cursor `(created_at, message_id)`."""
        ...

    def day_channels(self, guild_id: int, local_date: str) -> list[int]: ...
    def latest_message_at(self, guild_id: int, channel_id: int, since_date: str) -> datetime | None:
        """created_at of the newest logged message of the channel on or after `since_date`
        (a local date), or None. Used by the downtime catch-up."""
        ...

    def get_summary(self, guild_id: int, channel_id: int, local_date: str) -> DaySummary | None: ...
    def put_summary(self, s: DaySummary) -> None: ...
    def all_summaries(self, guild_id: int) -> list[DaySummary]:
        """Every day summary of the guild (one query; the memory index and the vector backfill
        filter in Python)."""
        ...

    def put_summary_vectors(
        self,
        guild_id: int,
        channel_id: int,
        local_date: str,
        vec_model: str,
        vec_hashes: list[str],
        vecs: bytes,
    ) -> bool:
        """Set only the vector fields of an existing summary; False when it does not exist."""
        ...

    def summaries(
        self, guild_id: int, allowed_channels: Collection[int], start_date: str, end_date: str
    ) -> list[DaySummary]:
        """Day summaries to SHOW: never a dirty one (a covered message was deleted, so its text
        may still be in the summary until it is regenerated). The scheduler reads the dirty
        ones through `get_summary` / `all_summaries` / `open_summaries`."""
        ...

    def open_summaries(self, guild_id: int) -> list[DaySummary]:
        """Summaries that are not final or are dirty (work for the scheduler)."""
        ...

    def get_channel_state(self, guild_id: int, channel_id: int) -> ChannelState | None: ...
    def put_channel_state(self, s: ChannelState) -> None: ...
    def list_channel_states(self, guild_id: int) -> list[ChannelState]: ...
    def get_guild_state(self, guild_id: int) -> GuildMemoryState: ...
    def put_guild_state(self, s: GuildMemoryState) -> None: ...


# ---------------------------------------------------------------------------- SQLite
SCHEMA = """
CREATE TABLE IF NOT EXISTS salid_messages (
  guild_id INTEGER NOT NULL, message_id INTEGER NOT NULL,
  channel_id INTEGER NOT NULL, thread_id INTEGER,
  author_id INTEGER, author_name TEXT, created_at TEXT NOT NULL, local_date TEXT NOT NULL,
  text TEXT, jump_url TEXT, edited_at TEXT, kind TEXT, visibility TEXT DEFAULT 'normal',
  PRIMARY KEY (guild_id, message_id)
);
CREATE INDEX IF NOT EXISTS ix_msg_day ON salid_messages(guild_id, local_date, channel_id);
CREATE INDEX IF NOT EXISTS ix_msg_created ON salid_messages(created_at);
CREATE TABLE IF NOT EXISTS salid_memory_channels (
  guild_id INTEGER NOT NULL, channel_id INTEGER NOT NULL, doc TEXT NOT NULL,
  PRIMARY KEY (guild_id, channel_id)
);
CREATE TABLE IF NOT EXISTS salid_memory_guilds (guild_id INTEGER PRIMARY KEY, doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS salid_day_summaries (
  guild_id INTEGER NOT NULL, channel_id INTEGER NOT NULL, local_date TEXT NOT NULL,
  summary TEXT, last_message_at TEXT, last_message_id INTEGER DEFAULT 0,
  message_count INTEGER DEFAULT 0, updated_at TEXT, final INTEGER DEFAULT 0,
  dirty INTEGER DEFAULT 0, vec_model TEXT DEFAULT '', vec_hashes TEXT DEFAULT '[]', vecs BLOB,
  PRIMARY KEY (guild_id, channel_id, local_date)
);
"""
# Columns added after the first release: added to older databases on open.
SUMMARY_ADDED_COLUMNS = {
    "vec_model": "TEXT DEFAULT ''",
    "vec_hashes": "TEXT DEFAULT '[]'",
    "vecs": "BLOB",
}
MESSAGE_COLS = (
    "guild_id,message_id,channel_id,thread_id,author_id,author_name,created_at,local_date,"
    "text,jump_url,edited_at,kind,visibility"
)


def _marks(n: int) -> str:
    return ",".join("?" * n)


class SQLiteMemoryStore:
    """Tables `salid_messages` and `salid_day_summaries`, in the ledger's database file (or a
    shared in-memory database for ":memory:")."""

    def __init__(self, path: str, zone=None):
        self.path = path
        self.zone = zone or ZoneInfo("Asia/Bangkok")  # day boundary for summary retention
        self._shared = None
        self._lock = threading.RLock()
        if path == ":memory:":
            self._shared = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared.row_factory = sqlite3.Row
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            cols = {r["name"] for r in c.execute("PRAGMA table_info(salid_messages)")}
            if "visibility" not in cols:  # database from before DM notes
                c.execute("ALTER TABLE salid_messages ADD COLUMN visibility TEXT DEFAULT 'normal'")
            have = {r["name"] for r in c.execute("PRAGMA table_info(salid_day_summaries)")}
            for col, decl in SUMMARY_ADDED_COLUMNS.items():
                if col not in have:
                    c.execute(f"ALTER TABLE salid_day_summaries ADD COLUMN {col} {decl}")

    @contextmanager
    def _conn(self):
        with self._lock:
            if self._shared is not None:
                yield self._shared
                self._shared.commit()
                return
            c = sqlite3.connect(self.path, timeout=10)
            c.row_factory = sqlite3.Row
            try:
                yield c
                c.commit()
            finally:
                c.close()

    @staticmethod
    def _msg(row) -> MemoryMessage:
        d = dict(row)
        d["visibility"] = d.get("visibility") or "normal"
        return message_from_doc(d)

    @staticmethod
    def _sum(row) -> DaySummary:
        d = dict(row)
        d["summary"] = json.loads(d["summary"] or "{}")
        d["final"], d["dirty"] = bool(d["final"]), bool(d["dirty"])
        d["vec_model"] = d.get("vec_model") or ""
        d["vec_hashes"] = json.loads(d.get("vec_hashes") or "[]")
        d["vecs"] = bytes(d.get("vecs") or b"")
        return DaySummary(**d)

    # ------------------------------------------------------------ messages
    def add_message(self, m: MemoryMessage) -> None:
        d = message_to_doc(m)
        with self._conn() as c:
            c.execute(
                f"INSERT OR REPLACE INTO salid_messages ({MESSAGE_COLS}) VALUES ({_marks(13)})",
                [d[k] for k in MESSAGE_COLS.split(",")],
            )

    def update_message_text(self, guild_id, message_id, text, edited_at) -> bool:
        with self._conn() as c:
            cur = c.execute(
                "UPDATE salid_messages SET text=?, edited_at=COALESCE(?, edited_at) "
                "WHERE guild_id=? AND message_id=?",
                (text, iso(edited_at) if edited_at else None, guild_id, message_id),
            )
            return cur.rowcount > 0

    def set_visibility(self, guild_id, message_id, visibility) -> bool:
        with self._conn() as c:
            cur = c.execute(
                "UPDATE salid_messages SET visibility=? WHERE guild_id=? AND message_id=?",
                (visibility, guild_id, message_id),
            )
            return cur.rowcount > 0

    def delete_messages(self, guild_id, message_ids) -> list[tuple[int, str]]:
        ids = list(message_ids)
        affected: set[tuple[int, str]] = set()
        with self._conn() as c:
            for i in range(0, len(ids), 200):
                part = ids[i : i + 200]
                rows = c.execute(
                    "SELECT channel_id, local_date FROM salid_messages "
                    f"WHERE guild_id=? AND message_id IN ({_marks(len(part))})",
                    [guild_id, *part],
                ).fetchall()
                affected |= {(r["channel_id"], r["local_date"]) for r in rows}
                c.execute(
                    f"DELETE FROM salid_messages WHERE guild_id=? AND message_id IN ({_marks(len(part))})",
                    [guild_id, *part],
                )
            for cid, day in affected:
                c.execute(
                    "UPDATE salid_day_summaries SET dirty=1 "
                    "WHERE guild_id=? AND channel_id=? AND local_date=?",
                    (guild_id, cid, day),
                )
        return sorted(affected)

    def delete_guild(self, guild_id) -> int:
        with self._conn() as c:
            n = c.execute("DELETE FROM salid_messages WHERE guild_id=?", (guild_id,)).rowcount
            c.execute("DELETE FROM salid_day_summaries WHERE guild_id=?", (guild_id,))
            c.execute("DELETE FROM salid_memory_channels WHERE guild_id=?", (guild_id,))
            c.execute("DELETE FROM salid_memory_guilds WHERE guild_id=?", (guild_id,))
        return n

    def purge(self, cutoffs, default_cutoff) -> tuple[int, int]:
        m = d = 0
        with self._conn() as c:
            ids = list(cutoffs)
            for gid, cut in cutoffs.items():
                m += c.execute(
                    "DELETE FROM salid_messages WHERE guild_id=? AND created_at < ?",
                    (gid, iso(cut)),
                ).rowcount
                d += c.execute(
                    "DELETE FROM salid_day_summaries WHERE guild_id=? AND local_date < ?",
                    (gid, cutoff_date(cut, self.zone)),
                ).rowcount
            rest = f" AND guild_id NOT IN ({_marks(len(ids))})" if ids else ""
            m += c.execute(
                f"DELETE FROM salid_messages WHERE created_at < ?{rest}",
                [iso(default_cutoff), *ids],
            ).rowcount
            d += c.execute(
                f"DELETE FROM salid_day_summaries WHERE local_date < ?{rest}",
                [cutoff_date(default_cutoff, self.zone), *ids],
            ).rowcount
        return m, d

    def messages(
        self,
        guild_id,
        allowed_channels,
        start_date,
        end_date,
        *,
        channel_id=None,
        keyword=None,
        limit=50,
        newest_first=True,
    ):
        allowed = list(allowed_channels)
        if not allowed or (channel_id is not None and channel_id not in allowed):
            return []
        with self._conn() as c:
            rows = c.execute(
                f"SELECT {MESSAGE_COLS} FROM salid_messages WHERE guild_id=? "
                f"AND local_date BETWEEN ? AND ? AND channel_id IN ({_marks(len(allowed))})",
                [guild_id, start_date, end_date, *allowed],
            ).fetchall()
        return select_messages(
            (self._msg(r) for r in rows),
            channel_id=channel_id,
            keyword=keyword,
            limit=limit,
            newest_first=newest_first,
        )

    def messages_after(self, guild_id, channel_id, local_date, after=None, limit=500):
        with self._conn() as c:
            rows = c.execute(
                f"SELECT {MESSAGE_COLS} FROM salid_messages "
                "WHERE guild_id=? AND channel_id=? AND local_date=?",
                (guild_id, channel_id, local_date),
            ).fetchall()
        out = [self._msg(r) for r in rows]
        if after is not None:
            out = [m for m in out if m.key > after]
        out.sort(key=lambda m: m.key)
        return out[:limit]

    def day_channels(self, guild_id, local_date) -> list[int]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT channel_id FROM salid_messages WHERE guild_id=? AND local_date=?",
                (guild_id, local_date),
            ).fetchall()
        return sorted(r["channel_id"] for r in rows)

    def latest_message_at(self, guild_id, channel_id, since_date):
        with self._conn() as c:
            row = c.execute(
                "SELECT MAX(created_at) AS t FROM salid_messages "
                "WHERE guild_id=? AND channel_id=? AND local_date >= ?",
                (guild_id, channel_id, since_date),
            ).fetchone()
        return _dt(row["t"]) if row and row["t"] else None

    # ------------------------------------------------------------ summaries
    def get_summary(self, guild_id, channel_id, local_date):
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM salid_day_summaries WHERE guild_id=? AND channel_id=? "
                "AND local_date=?",
                (guild_id, channel_id, local_date),
            ).fetchone()
        return self._sum(row) if row else None

    def put_summary(self, s: DaySummary) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO salid_day_summaries (guild_id,channel_id,local_date,"
                "summary,last_message_at,last_message_id,message_count,updated_at,final,dirty,"
                "vec_model,vec_hashes,vecs) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    s.guild_id,
                    s.channel_id,
                    s.local_date,
                    json.dumps(s.summary, ensure_ascii=False),
                    s.last_message_at,
                    s.last_message_id,
                    s.message_count,
                    s.updated_at,
                    int(s.final),
                    int(s.dirty),
                    s.vec_model,
                    json.dumps(s.vec_hashes),
                    s.vecs,
                ),
            )

    def all_summaries(self, guild_id):
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM salid_day_summaries WHERE guild_id=? ORDER BY local_date, channel_id",
                (guild_id,),
            ).fetchall()
        return [self._sum(r) for r in rows]

    def put_summary_vectors(self, guild_id, channel_id, local_date, vec_model, vec_hashes, vecs):
        with self._conn() as c:
            cur = c.execute(
                "UPDATE salid_day_summaries SET vec_model=?, vec_hashes=?, vecs=? "
                "WHERE guild_id=? AND channel_id=? AND local_date=?",
                (vec_model, json.dumps(vec_hashes), vecs, guild_id, channel_id, local_date),
            )
            return cur.rowcount > 0

    def summaries(self, guild_id, allowed_channels, start_date, end_date):
        allowed = list(allowed_channels)
        if not allowed:
            return []
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM salid_day_summaries WHERE guild_id=? "
                f"AND local_date BETWEEN ? AND ? AND channel_id IN ({_marks(len(allowed))}) "
                "AND dirty=0 ORDER BY local_date, channel_id",
                [guild_id, start_date, end_date, *allowed],
            ).fetchall()
        return [self._sum(r) for r in rows]

    def open_summaries(self, guild_id):
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM salid_day_summaries WHERE guild_id=? AND (final=0 OR dirty=1)",
                (guild_id,),
            ).fetchall()
        return [self._sum(r) for r in rows]


# state methods shared by name; SQLiteMemoryStore gets them below
def _sqlite_state_methods():
    def get_channel_state(self, guild_id, channel_id):
        with self._conn() as c:
            row = c.execute(
                "SELECT doc FROM salid_memory_channels WHERE guild_id=? AND channel_id=?",
                (guild_id, channel_id),
            ).fetchone()
        return ChannelState(**json.loads(row["doc"])) if row else None

    def put_channel_state(self, s):
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO salid_memory_channels (guild_id, channel_id, doc) "
                "VALUES (?,?,?)",
                (s.guild_id, s.channel_id, s.model_dump_json()),
            )

    def list_channel_states(self, guild_id):
        with self._conn() as c:
            rows = c.execute(
                "SELECT doc FROM salid_memory_channels WHERE guild_id=?", (guild_id,)
            ).fetchall()
        return [ChannelState(**json.loads(r["doc"])) for r in rows]

    def get_guild_state(self, guild_id):
        with self._conn() as c:
            row = c.execute(
                "SELECT doc FROM salid_memory_guilds WHERE guild_id=?", (guild_id,)
            ).fetchone()
        return (
            GuildMemoryState(**json.loads(row["doc"]))
            if row
            else GuildMemoryState(guild_id=guild_id)
        )

    def put_guild_state(self, s):
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO salid_memory_guilds (guild_id, doc) VALUES (?,?)",
                (s.guild_id, s.model_dump_json()),
            )

    return (
        get_channel_state,
        put_channel_state,
        list_channel_states,
        get_guild_state,
        put_guild_state,
    )


(
    SQLiteMemoryStore.get_channel_state,
    SQLiteMemoryStore.put_channel_state,
    SQLiteMemoryStore.list_channel_states,
    SQLiteMemoryStore.get_guild_state,
    SQLiteMemoryStore.put_guild_state,
) = _sqlite_state_methods()


def make_memory_store(settings, store=None) -> MemoryStore:
    """`SALID_STORE` picks the backend. SQLite shares the ledger's file (`store.path` when the
    ledger is a SQLiteStore, else `SALID_DB_PATH`)."""
    if settings.store == "firestore":
        from .firestore_memory import FirestoreMemoryStore

        return FirestoreMemoryStore(
            messages=settings.memory_messages_collection,
            summaries=settings.memory_summaries_collection,
            channels=settings.memory_channels_collection,
            guilds=settings.memory_state_collection,
            project=settings.gcp_project,
            zone=settings.zone,
        )
    if settings.store != "sqlite":
        raise ValueError(f"SALID_STORE must be sqlite or firestore, got {settings.store!r}")
    return SQLiteMemoryStore(getattr(store, "path", None) or settings.db_path, settings.zone)
