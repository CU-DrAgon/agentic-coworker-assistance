"""The multi-workspace store adapters: memory routed by channel, ledger fanned out by submitter."""

from datetime import UTC, datetime

import pytest
from conftest import FixedClockSettings

from salid.memory_store import DaySummary, MemoryMessage, SQLiteMemoryStore
from salid.org_stores import MultiGuildLedgerStore, MultiGuildMemoryStore
from salid.qa import LedgerTools, MemoryTools
from salid.store import Receipt, SQLiteStore

DISCORD, SLACK = 900, 2**62 + 5  # a Discord guild and a platform tenant
CH_D1, CH_D2, CH_S1 = 501, 502, 2**62 + 77
DAY = "2026-10-14"


@pytest.fixture
def s():
    return FixedClockSettings(exec_user_ids=set())


@pytest.fixture
def mstore(tmp_path, s):
    return SQLiteMemoryStore(str(tmp_path / "m.db"), s.zone)


def msg(guild, channel, mid, text, minute=0):
    at = datetime(2026, 10, 14, 3, minute, tzinfo=UTC)
    return MemoryMessage(
        guild_id=guild,
        channel_id=channel,
        message_id=mid,
        author_id=1,
        author_name="ann",
        created_at=at,
        local_date=DAY,
        text=text,
    )


def put_day(mstore, guild, channel, overview):
    mstore.put_summary(
        DaySummary(
            guild_id=guild,
            channel_id=channel,
            local_date=DAY,
            message_count=3,
            final=True,
            summary={"overview": overview, "topics": [], "decisions": [overview]},
        )
    )


@pytest.fixture
def world(mstore):
    mstore.add_message(msg(DISCORD, CH_D1, 1, "discord budget approved", 1))
    mstore.add_message(msg(DISCORD, CH_D2, 2, "discord secret layoffs", 2))
    mstore.add_message(msg(SLACK, CH_S1, 3, "slack budget question", 3))
    put_day(mstore, DISCORD, CH_D1, "discord budget plan")
    put_day(mstore, DISCORD, CH_D2, "discord layoffs plan")
    put_day(mstore, SLACK, CH_S1, "slack budget thread")
    return mstore


def tools(s, store, allowed, labels=None):
    return MemoryTools(store, SLACK, allowed, s, 90, embedder=None)


def adapter(base, s):
    routes = {CH_D1: DISCORD, CH_D2: DISCORD, CH_S1: SLACK}
    return MultiGuildMemoryStore(base, routes, SLACK)


def test_day_summaries_come_from_every_routed_guild(s, world):
    t = tools(s, adapter(world, s), {CH_S1: "Slack #ops", CH_D1: "Discord #general"})
    got = t.chat_days(DAY, DAY)
    assert {d["channel"] for d in got["days"]} == {"Slack #ops", "Discord #general"}
    assert "discord budget plan" in str(got) and "layoffs" not in str(got)


def test_a_channel_outside_the_allowed_scope_is_never_read(s, world):
    t = tools(s, adapter(world, s), {CH_S1: "Slack #ops", CH_D1: "Discord #general"})
    assert "layoffs" not in str(t.chat_days(DAY, DAY)) + str(
        t.chat_messages(DAY, DAY, keyword="discord")
    )
    assert t.chat_messages(DAY, DAY, channel_name="Discord #secret")["count"] == 0


def test_search_and_messages_label_the_source_platform(s, world):
    t = tools(s, adapter(world, s), {CH_S1: "Slack #ops", CH_D1: "Discord #general"})
    hits = t.chat_search(keywords=["budget"])
    assert {h["channel"] for h in hits["hits"]} == {"Slack #ops", "Discord #general"}
    msgs = t.chat_messages(DAY, DAY)["messages"]
    assert {m["channel"] for m in msgs} == {"Slack #ops", "Discord #general"}
    assert [m["text"] for m in msgs] == ["slack budget question", "discord budget approved"]


def test_the_message_limit_applies_after_merging(s, world):
    t = tools(s, adapter(world, s), {CH_S1: "Slack #ops", CH_D1: "Discord #general"})
    one = t.chat_messages(DAY, DAY, limit=1)["messages"]
    assert [m["text"] for m in one] == ["slack budget question"]  # the newest of both guilds


def test_a_plain_store_without_an_index_gets_entries_built_on_the_fly(s, world):
    a = adapter(world, s)
    rows = a.summary_entries(SLACK, {CH_S1, CH_D1}, DAY, DAY)
    assert sorted(e.channel_id for e in rows) == sorted([CH_S1, CH_D1])


def test_unrouted_channels_belong_to_the_asking_tenant(s, world):
    a = MultiGuildMemoryStore(world, {CH_D1: DISCORD}, SLACK)
    assert [m.message_id for m in a.messages(SLACK, {CH_S1}, DAY, DAY)] == [3]
    assert a.messages_after(SLACK, CH_D1, DAY)[0].message_id == 1


# ------------------------------------------------------------------ ledger
def rcpt(store, guild, submitter, n, total, cur="THB", day="2026-10-10", cat="food"):
    return store.add_receipt(
        Receipt(
            guild_id=guild,
            submitter_id=submitter,
            submitter_name=f"u{submitter}",
            sha256=f"sha{guild}{submitter}{n}",
            vendor=f"v{n}",
            receipt_date=day,
            effective_date=day,
            total=total,
            currency=cur,
            category=cat,
            confidence=0.9,
            created_at=day,
        )
    )


ME_D, ME_S, OTHER = 42, 2**62 + 9, 77


@pytest.fixture
def ledger(tmp_path):
    st = SQLiteStore(str(tmp_path / "l.db"))
    rcpt(st, DISCORD, ME_D, 1, 100.0)
    rcpt(st, DISCORD, OTHER, 2, 1000.0)
    rcpt(st, SLACK, ME_S, 3, 50.0, "USD")
    rcpt(st, SLACK, ME_S, 4, 25.0, "THB", cat="travel")
    rcpt(st, SLACK, OTHER + 1, 5, 7.0)
    rcpt(st, 12345, ME_D, 6, 9999.0)  # a workspace that is not in the org
    return st


def test_a_member_sees_only_their_own_receipts_in_every_workspace(ledger):
    store = MultiGuildLedgerStore(ledger, [(DISCORD, ME_D), (SLACK, ME_S)])
    out = LedgerTools(store, SLACK, ME_S).spending_summary("2026-10-01", "2026-10-31")
    assert out["totals_by_currency"] == {"THB": 125.0, "USD": 50.0}
    assert out["total"] is None  # currencies are never added together
    assert out["count"] == 3


def test_an_executive_sees_every_receipt_of_every_org_workspace(ledger):
    store = MultiGuildLedgerStore(ledger, [(DISCORD, None), (SLACK, None)])
    out = LedgerTools(store, SLACK, None).spending_summary("2026-10-01", "2026-10-31")
    assert out["totals_by_currency"] == {"THB": 1132.0, "USD": 50.0}  # not the outside guild
    assert out["count"] == 5


def test_grouped_summaries_merge_the_same_key_across_workspaces(ledger):
    store = MultiGuildLedgerStore(ledger, [(DISCORD, ME_D), (SLACK, ME_S)])
    out = LedgerTools(store, SLACK, ME_S).spending_summary(
        "2026-10-01", "2026-10-31", group_by="category"
    )
    by = {g["key"]: g["totals_by_currency"] for g in out["groups"]}
    assert by == {"food": {"THB": 100.0, "USD": 50.0}, "travel": {"THB": 25.0}}
    assert out["count"] == sum(g["count"] for g in out["groups"]) == 3


def test_receipts_are_listed_newest_first_across_workspaces(ledger):
    rcpt(ledger, SLACK, ME_S, 8, 5.0, day="2026-10-20")
    store = MultiGuildLedgerStore(ledger, [(DISCORD, ME_D), (SLACK, ME_S)])
    rows = LedgerTools(store, SLACK, ME_S).list_receipts("2026-10-01", "2026-10-31", limit=2)
    assert rows["count"] == 2
    got = store.list_receipts(SLACK, "2026-10-01", "2026-10-31", limit=2, submitter_id=ME_S)
    assert [r.effective_date for r in got] == ["2026-10-20", "2026-10-10"]


def test_a_workspace_that_is_not_listed_is_never_read(ledger):
    store = MultiGuildLedgerStore(ledger, [(SLACK, ME_S)])
    out = LedgerTools(store, SLACK, ME_S).spending_summary("2026-10-01", "2026-10-31")
    assert out["totals_by_currency"] == {"THB": 25.0, "USD": 50.0}


def test_a_restricted_pair_keeps_its_submitter_whatever_the_caller_forces(ledger):
    store = MultiGuildLedgerStore(ledger, [(DISCORD, ME_D)])
    out = store.summarize(DISCORD, "2026-10-01", "2026-10-31", "none", submitter_id=OTHER)
    assert out["totals_by_currency"] == {"THB": 100.0}  # the pair's own submitter, never OTHER


def test_flagged_duplicates_and_lookups_fall_through(ledger):
    store = MultiGuildLedgerStore(ledger, [(DISCORD, None), (SLACK, None)])
    assert store.get(1).guild_id == DISCORD  # lookups by id are the base store's
    assert isinstance(store.flagged(SLACK), list) and store.duplicate_groups(SLACK) == []
