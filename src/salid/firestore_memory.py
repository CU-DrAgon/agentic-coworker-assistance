"""Firestore-backed memory store (same protocol as `SQLiteMemoryStore`).

Collections (names from `SALID_MEMORY_MESSAGES_COLLECTION` / `..._SUMMARIES_COLLECTION`):

* `salid_messages/{guild_id}_{message_id}`: one raw message (see `MemoryMessage`).
* `salid_day_summaries/{guild_id}_{channel_id}_{local_date}`: one summary per channel-day.

Every query is equality-only on single fields (`guild_id`, plus `local_date` or `final` / `dirty`
where used), which Firestore serves from automatic indexes: no composite index is needed. The one
range queries of the purge are single-field too. Channel filtering, keyword
matching and ordering happen in Python over one channel-day at a time.

There is deliberately no stored expiry (and so no Firestore TTL policy): retention is a per-guild
setting that can change at any time, so the bot's daily purge deletes by the guild's current
retention (`created_at < now - days`, summaries by `local_date`). The purge queries are single-field
range queries (`created_at <` / `local_date <` the latest cutoff), filtered per guild in Python.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from google.cloud.firestore_v1.base_query import FieldFilter

from .memory_store import (
    ChannelState,
    DaySummary,
    GuildMemoryState,
    MemoryMessage,
    cutoff_date,
    date_range,
    iso,
    message_from_doc,
    message_to_doc,
    select_messages,
)


def _sum_from(d: dict[str, Any]) -> DaySummary:
    return DaySummary(**d)


class FirestoreMemoryStore:
    def __init__(
        self,
        client: Any = None,
        messages: str = "salid_messages",
        summaries: str = "salid_day_summaries",
        channels: str = "salid_memory_channels",
        guilds: str = "salid_memory_guilds",
        project: str | None = None,
        zone=None,
    ) -> None:
        self._client, self._project = client, project
        self.zone = zone or ZoneInfo("Asia/Bangkok")
        self.messages_name, self.summaries_name = messages, summaries
        self.channels_name, self.guilds_name = channels, guilds

    @property
    def client(self) -> Any:
        if self._client is None:
            from google.cloud import firestore

            self._client = firestore.Client(project=self._project)
        return self._client

    # ------------------------------------------------------------ helpers
    def _q(self, collection: str, *conds: tuple[str, str, Any]) -> Any:
        q = self.client.collection(collection)
        for f, op, v in conds:
            q = q.where(filter=FieldFilter(f, op, v))
        return q

    def _msg_ref(self, guild_id: int, message_id: int) -> Any:
        return self.client.collection(self.messages_name).document(f"{guild_id}_{message_id}")

    def _sum_ref(self, guild_id: int, channel_id: int, day: str) -> Any:
        return self.client.collection(self.summaries_name).document(
            f"{guild_id}_{channel_id}_{day}"
        )

    def _day(self, guild_id: int, day: str) -> list[MemoryMessage]:
        snaps = self._q(self.messages_name, ("guild_id", "==", guild_id), ("local_date", "==", day))
        return [message_from_doc(s.to_dict()) for s in snaps.stream()]

    # ------------------------------------------------------------ messages
    def add_message(self, m: MemoryMessage) -> None:
        self._msg_ref(m.guild_id, m.message_id).set(message_to_doc(m))

    def update_message_text(self, guild_id, message_id, text, edited_at) -> bool:
        ref = self._msg_ref(guild_id, message_id)
        if not ref.get().exists:
            return False
        ref.update({"text": text, **({"edited_at": iso(edited_at)} if edited_at else {})})
        return True

    def delete_messages(self, guild_id, message_ids) -> list[tuple[int, str]]:
        affected: set[tuple[int, str]] = set()
        for mid in message_ids:
            ref = self._msg_ref(guild_id, mid)
            snap = ref.get()
            if not snap.exists:
                continue
            d = snap.to_dict()
            affected.add((d["channel_id"], d["local_date"]))
            ref.delete()
        for cid, day in affected:
            sref = self._sum_ref(guild_id, cid, day)
            if sref.get().exists:
                sref.update({"dirty": True})
        return sorted(affected)

    def delete_guild(self, guild_id) -> int:
        n = 0
        for snap in self._q(self.messages_name, ("guild_id", "==", guild_id)).stream():
            snap.reference.delete()
            n += 1
        for name in (self.summaries_name, self.channels_name):
            for snap in self._q(name, ("guild_id", "==", guild_id)).stream():
                snap.reference.delete()
        self.client.collection(self.guilds_name).document(str(guild_id)).delete()
        return n

    def purge(self, cutoffs, default_cutoff) -> tuple[int, int]:
        def cut_for(gid: int) -> datetime:
            return cutoffs.get(gid, default_cutoff)

        latest = max([default_cutoff, *cutoffs.values()])
        m = d = 0
        for snap in self._q(self.messages_name, ("created_at", "<", iso(latest))).stream():
            doc = snap.to_dict()
            if doc["created_at"] < iso(cut_for(doc["guild_id"])):
                snap.reference.delete()
                m += 1
        for snap in self._q(
            self.summaries_name, ("local_date", "<", cutoff_date(latest, self.zone))
        ).stream():
            doc = snap.to_dict()
            if doc["local_date"] < cutoff_date(cut_for(doc["guild_id"]), self.zone):
                snap.reference.delete()
                d += 1
        return m, d

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
    ) -> list[MemoryMessage]:
        allowed = set(allowed_channels)
        if not allowed or (channel_id is not None and channel_id not in allowed):
            return []
        rows = [
            m
            for day in date_range(start_date, end_date)
            for m in self._day(guild_id, day)
            if m.channel_id in allowed
        ]
        return select_messages(
            rows, channel_id=channel_id, keyword=keyword, limit=limit, newest_first=newest_first
        )

    def messages_after(self, guild_id, channel_id, local_date, after=None, limit=500):
        out = [m for m in self._day(guild_id, local_date) if m.channel_id == channel_id]
        if after is not None:
            out = [m for m in out if m.key > after]
        out.sort(key=lambda m: m.key)
        return out[:limit]

    def day_channels(self, guild_id, local_date) -> list[int]:
        return sorted({m.channel_id for m in self._day(guild_id, local_date)})

    def latest_message_at(self, guild_id, channel_id, since_date):
        today = datetime.now(self.zone).date().isoformat()
        for day in reversed(date_range(since_date, max(today, since_date))):
            rows = [m for m in self._day(guild_id, day) if m.channel_id == channel_id]
            if rows:
                return max(m.created_at for m in rows)
        return None

    # ------------------------------------------------------------ summaries
    def get_summary(self, guild_id, channel_id, local_date):
        snap = self._sum_ref(guild_id, channel_id, local_date).get()
        return _sum_from(snap.to_dict()) if snap.exists else None

    def put_summary(self, s: DaySummary) -> None:
        self._sum_ref(s.guild_id, s.channel_id, s.local_date).set(s.model_dump())

    def summaries(self, guild_id, allowed_channels, start_date, end_date):
        allowed = set(allowed_channels)
        if not allowed:
            return []
        out = []
        for day in date_range(start_date, end_date):
            snaps = self._q(
                self.summaries_name, ("guild_id", "==", guild_id), ("local_date", "==", day)
            )
            out += [
                _sum_from(s.to_dict())
                for s in snaps.stream()
                if s.to_dict()["channel_id"] in allowed
            ]
        return sorted(out, key=lambda s: (s.local_date, s.channel_id))

    def open_summaries(self, guild_id):
        seen: dict[str, DaySummary] = {}
        for flag, want in (("final", False), ("dirty", True)):
            for s in self._q(
                self.summaries_name, ("guild_id", "==", guild_id), (flag, "==", want)
            ).stream():
                d = _sum_from(s.to_dict())
                seen[f"{d.channel_id}_{d.local_date}"] = d
        return list(seen.values())

    # ------------------------------------------------------------ import state
    def get_channel_state(self, guild_id, channel_id):
        snap = self.client.collection(self.channels_name).document(f"{guild_id}_{channel_id}").get()
        return ChannelState(**snap.to_dict()) if snap.exists else None

    def put_channel_state(self, s: ChannelState) -> None:
        self.client.collection(self.channels_name).document(f"{s.guild_id}_{s.channel_id}").set(
            s.model_dump()
        )

    def list_channel_states(self, guild_id):
        snaps = self._q(self.channels_name, ("guild_id", "==", guild_id)).stream()
        return [ChannelState(**s.to_dict()) for s in snaps]

    def get_guild_state(self, guild_id):
        snap = self.client.collection(self.guilds_name).document(str(guild_id)).get()
        return (
            GuildMemoryState(**snap.to_dict())
            if snap.exists
            else GuildMemoryState(guild_id=guild_id)
        )

    def put_guild_state(self, s: GuildMemoryState) -> None:
        self.client.collection(self.guilds_name).document(str(s.guild_id)).set(s.model_dump())
