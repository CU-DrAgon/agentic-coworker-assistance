"""Store adapters that let the single-tenant tools read several workspaces of one org.

`qa.MemoryTools(store, guild_id, allowed, ...)` and `qa.LedgerTools(store, guild_id, forced)` call
their store with ONE guild id. These adapters answer for several: each allowed channel is routed to
the workspace it belongs to (channel ids are globally unique: Discord snowflakes are below 2**62,
every other platform's ids are namespaced at or above it), each ledger read fans out over the
(workspace, submitter) pairs the asker may see. The adapters never widen anything themselves: the
caller decides the allowed channels and the pairs (`org_scope.OrgView`), and a channel / pair that
is not listed is never read.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from .embed import NullEmbedder
from .memory_index import make_entry
from .memory_store import DaySummary, MemoryMessage
from .store import Receipt, build_summary


class MultiGuildMemoryStore:
    """Read-only view of a memory store over several guilds. `routes` maps channel id -> guild
    id; a channel without a route belongs to `primary` (the asking tenant)."""

    def __init__(self, base, routes: Mapping[int, int], primary: int, embedder=None):
        self.base, self.routes, self.primary = base, dict(routes), primary
        self.embedder = embedder or NullEmbedder()

    def __getattr__(self, name):  # anything else (writes included) is the base store's
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def _guild_of(self, channel_id: int) -> int:
        return self.routes.get(channel_id, self.primary)

    def _split(self, channels: Collection[int]) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for c in channels:
            out.setdefault(self._guild_of(c), []).append(c)
        return out

    def summaries(
        self, guild_id: int, allowed_channels: Collection[int], start_date: str, end_date: str
    ) -> list[DaySummary]:
        rows: list[DaySummary] = []
        for g, chans in self._split(allowed_channels).items():
            rows += self.base.summaries(g, chans, start_date, end_date)
        return sorted(rows, key=lambda s: (s.local_date, s.channel_id))

    def summary_entries(
        self, guild_id: int, allowed_channels: Collection[int], start_date: str, end_date: str
    ):
        out = []
        for g, chans in self._split(allowed_channels).items():
            if hasattr(self.base, "summary_entries"):
                out += self.base.summary_entries(g, chans, start_date, end_date)
            else:
                out += [
                    make_entry(s, self.embedder)
                    for s in self.base.summaries(g, chans, start_date, end_date)
                ]
        return sorted(out, key=lambda e: (e.local_date, e.channel_id))

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
        rows: list[MemoryMessage] = []
        for g, chans in self._split(allowed_channels).items():
            rows += self.base.messages(
                g,
                chans,
                start_date,
                end_date,
                channel_id=channel_id,
                keyword=keyword,
                limit=limit,
                newest_first=newest_first,
            )
        rows.sort(key=lambda m: m.key, reverse=newest_first)
        return rows[:limit]

    def messages_after(
        self,
        guild_id: int,
        channel_id: int,
        local_date: str,
        after: tuple[str, int] | None = None,
        limit: int = 500,
    ) -> list[MemoryMessage]:
        return self.base.messages_after(
            self._guild_of(channel_id), channel_id, local_date, after, limit
        )


class MultiGuildLedgerStore:
    """The receipt ledger over several workspaces. `sources` = (guild id, submitter id | None)
    pairs; None means every receipt of that workspace (an executive), a number only that
    submitter's receipts. Calls for a guild outside the list read nothing. Totals stay per
    currency (`store.build_summary`); writes and lookups by receipt id are the base store's."""

    def __init__(self, base, sources: Sequence[tuple[int, int | None]]):
        self.base = base
        self.sources = list(dict.fromkeys(sources))

    def __getattr__(self, name):
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def _each(self, submitter_id: int | None):
        """(guild, submitter) pairs to read. A caller-forced submitter only narrows an
        unrestricted pair; a restricted pair keeps its own submitter."""
        for g, s in self.sources:
            yield g, (s if s is not None else submitter_id)

    def summarize(self, guild_id, start, end, group_by="none", submitter_id=None):
        parts = [
            self.base.summarize(g, start, end, group_by, submitter_id=s)
            for g, s in self._each(submitter_id)
        ]
        rows: list[tuple] = []
        for p in parts:
            if group_by == "none":
                groups = [
                    {
                        "key": "all",
                        "totals_by_currency": p["totals_by_currency"],
                        "count": p["count"],
                    }
                ]
            else:
                groups = p["groups"]
            for g in groups:
                first = True
                for cur, total in g["totals_by_currency"].items():
                    rows.append((g["key"], cur, total, g["count"] if first else 0))
                    first = False
        return build_summary(start, end, group_by, rows)

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
    ) -> list[Receipt]:
        rows: list[Receipt] = []
        for g, s in self._each(submitter_id):
            rows += self.base.list_receipts(
                g,
                start,
                end,
                category,
                vendor_contains,
                submitter,
                limit,
                submitter_id=s,
                include_void=include_void,
            )
        rows.sort(key=lambda r: (r.effective_date, r.id or 0), reverse=True)
        return rows[: max(1, min(int(limit), 100))]

    def flagged(self, guild_id, start=None, end=None, submitter_id=None, include_void=False):
        rows: list[Receipt] = []
        for g, s in self._each(submitter_id):
            rows += self.base.flagged(g, start, end, submitter_id=s, include_void=include_void)
        return rows

    def duplicate_groups(self, guild_id, start=None, end=None, submitter_id=None):
        # a bill is a duplicate only inside one workspace's ledger (dedupe is per guild)
        groups: list[list[Receipt]] = []
        for g, s in self._each(submitter_id):
            groups += self.base.duplicate_groups(g, start, end, submitter_id=s)
        return groups
