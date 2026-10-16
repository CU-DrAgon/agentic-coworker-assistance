"""In-process index of day summaries and their vectors, per guild (meaning-based memory search).

`IndexedMemoryStore` wraps any `MemoryStore`: every protocol method is delegated, and the summary
reads (`summaries`, `all_summaries`, `summary_entries`) are served from a per-guild cache loaded
with one `all_summaries` query. The writes that touch summaries (`put_summary`,
`put_summary_vectors`, `delete_messages`, `purge`, `delete_guild`) update or drop the cache, so it
stays correct as long as every writer goes through this wrapper. That holds today: the bot is the
only process that writes summaries (one fixed Cloud Run worker instance; the hub only reads import
progress). Running two bot processes would make this cache stale.

The cache is bounded by approximate bytes (LRU over guilds). Vectors are unpacked once, when an
entry is loaded or written. Thread-safe: tool threads read while the scheduler thread writes;
readers get new lists. Logs carry ids, counts and sizes only.
"""

from __future__ import annotations

import logging
import threading
import time
from array import array
from collections import OrderedDict
from collections.abc import Collection
from dataclasses import dataclass, field

from .embed import NullEmbedder, unpack
from .memory import vectors_usable
from .memory_store import DaySummary, MemoryStore
from .memory_summary import summary_items

logger = logging.getLogger(__name__)

INDEX_MAX_BYTES = 64 * 1024 * 1024
ENTRY_OVERHEAD = 200  # rough bytes per entry beyond its texts and vectors


@dataclass
class IndexEntry:
    summary: DaySummary
    items: list[str]  # memory_summary.summary_items(summary.summary)
    vectors: list[array] = field(default_factory=list)  # one per item, or none

    @property
    def channel_id(self) -> int:
        return self.summary.channel_id

    @property
    def local_date(self) -> str:
        return self.summary.local_date


def make_entry(s: DaySummary, embedder) -> IndexEntry:
    items = summary_items(s.summary)
    usable = items and vectors_usable(s, embedder.model_id)
    return IndexEntry(s, items, unpack(s.vecs, embedder.dim) if usable else [])


def _entry_bytes(e: IndexEntry) -> int:
    return sum(len(x.encode("utf-8")) for x in e.items) + len(e.summary.vecs) + ENTRY_OVERHEAD


def entries_for(
    store, guild_id: int, allowed: Collection[int], start: str, end: str, embedder=None
) -> list[IndexEntry]:
    """Index entries for a guild, allowed channels and date range: from the cache when `store` is
    an `IndexedMemoryStore`, else built from `store.summaries` on the fly."""
    if hasattr(store, "summary_entries"):
        return store.summary_entries(guild_id, allowed, start, end)
    emb = embedder or NullEmbedder()
    return [make_entry(s, emb) for s in store.summaries(guild_id, allowed, start, end)]


class IndexedMemoryStore:
    def __init__(self, base: MemoryStore, embedder=None, max_bytes: int = INDEX_MAX_BYTES):
        self.base = base
        self.embedder = embedder or NullEmbedder()
        self.max_bytes = max_bytes
        self._guilds: OrderedDict[int, dict[tuple[int, str], IndexEntry]] = OrderedDict()
        self._sizes: dict[int, int] = {}
        self._lock = threading.RLock()

    def __getattr__(self, name):
        if name == "base":  # not set yet (during construction)
            raise AttributeError(name)
        return getattr(self.base, name)

    # ------------------------------------------------------------ cache
    def _entries(self, guild_id: int) -> dict[tuple[int, str], IndexEntry]:
        """The guild's entries (loading them on a miss). Call with the lock held."""
        got = self._guilds.get(guild_id)
        if got is not None:
            self._guilds.move_to_end(guild_id)
            return got
        started = time.monotonic()
        got = {
            (s.channel_id, s.local_date): make_entry(s, self.embedder)
            for s in self.base.all_summaries(guild_id)
        }
        self._guilds[guild_id] = got
        self._sizes[guild_id] = sum(_entry_bytes(e) for e in got.values())
        logger.info(
            "memory index load guild=%s summaries=%d bytes=%d ms=%d",
            guild_id, len(got), self._sizes[guild_id], (time.monotonic() - started) * 1000,
        )  # fmt: skip
        self._evict(keep=guild_id)
        return got

    def _evict(self, keep: int) -> None:
        while sum(self._sizes.values()) > self.max_bytes and len(self._guilds) > 1:
            gid = next(iter(self._guilds))
            if gid == keep:
                self._guilds.move_to_end(gid)
                continue
            self._guilds.pop(gid)
            self._sizes.pop(gid, None)
            logger.info("memory index evict guild=%s", gid)

    def _replace(self, s: DaySummary) -> None:
        """Update a loaded guild's entry for `s` (no-op when the guild is not loaded)."""
        entries = self._guilds.get(s.guild_id)
        if entries is None:
            return
        key = (s.channel_id, s.local_date)
        old = entries.get(key)
        new = entries[key] = make_entry(s, self.embedder)
        self._sizes[s.guild_id] += _entry_bytes(new) - (_entry_bytes(old) if old else 0)
        self._evict(keep=s.guild_id)

    # ------------------------------------------------------------ reads
    def summary_entries(
        self, guild_id: int, allowed_channels: Collection[int], start_date: str, end_date: str
    ) -> list[IndexEntry]:
        allowed = set(allowed_channels)
        if not allowed:
            return []
        with self._lock:
            rows = [
                e
                for e in self._entries(guild_id).values()
                if e.channel_id in allowed and start_date <= e.local_date <= end_date
            ]
        return sorted(rows, key=lambda e: (e.local_date, e.channel_id))

    def summaries(self, guild_id, allowed_channels, start_date, end_date) -> list[DaySummary]:
        return [
            e.summary
            for e in self.summary_entries(guild_id, allowed_channels, start_date, end_date)
        ]

    def all_summaries(self, guild_id: int) -> list[DaySummary]:
        with self._lock:
            rows = [e.summary for e in self._entries(guild_id).values()]
        return sorted(rows, key=lambda s: (s.local_date, s.channel_id))

    # ------------------------------------------------------------ writes
    def put_summary(self, s: DaySummary) -> None:
        self.base.put_summary(s)
        with self._lock:
            self._replace(s)

    def put_summary_vectors(
        self, guild_id, channel_id, local_date, vec_model, vec_hashes, vecs
    ) -> bool:
        ok = self.base.put_summary_vectors(
            guild_id, channel_id, local_date, vec_model, vec_hashes, vecs
        )
        with self._lock:
            entry = (self._guilds.get(guild_id) or {}).get((channel_id, local_date))
            if ok and entry is not None:
                update = {"vec_model": vec_model, "vec_hashes": list(vec_hashes), "vecs": vecs}
                self._replace(entry.summary.model_copy(update=update))
        return ok

    def delete_messages(self, guild_id, message_ids) -> list[tuple[int, str]]:
        affected = self.base.delete_messages(guild_id, message_ids)
        with self._lock:
            entries = self._guilds.get(guild_id) or {}
            for key in affected:
                if (e := entries.get(key)) is not None:  # a new entry: readers keep their copy
                    dirty = e.summary.model_copy(update={"dirty": True})
                    entries[key] = IndexEntry(dirty, e.items, e.vectors)
        return affected

    def purge(self, cutoffs, default_cutoff) -> tuple[int, int]:
        result = self.base.purge(cutoffs, default_cutoff)
        with self._lock:
            self._guilds.clear()
            self._sizes.clear()
        return result

    def delete_guild(self, guild_id) -> int:
        n = self.base.delete_guild(guild_id)
        with self._lock:
            self._guilds.pop(guild_id, None)
            self._sizes.pop(guild_id, None)
        return n
