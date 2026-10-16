"""IndexedMemoryStore: per-guild cache of day summaries and their vectors, kept fresh by the
writes that go through it, bounded by bytes. Offline: the Firestore fake."""

from datetime import UTC, datetime

from conftest import GUILD, FakeEmbedder
from fake_firestore import offline_memory_store
from test_memory import bkk, mk

from salid.memory import vectorize
from salid.memory_index import INDEX_MAX_BYTES, IndexedMemoryStore
from salid.memory_store import DaySummary


def summ(day, cid=10, gid=GUILD, text="งบโฆษณา"):
    return DaySummary(guild_id=gid, channel_id=cid, local_date=day, message_count=3, final=True,
                      summary={"overview": text})  # fmt: skip


def wrapped(max_bytes=INDEX_MAX_BYTES):
    """Firestore fake store whose `all_summaries` calls are counted in `base.loads`."""
    base = offline_memory_store()
    base.loads, real = [], base.all_summaries
    base.all_summaries = lambda gid: base.loads.append(gid) or real(gid)
    return base, IndexedMemoryStore(base, FakeEmbedder(), max_bytes=max_bytes)


def loads(base):
    return base.loads


def test_one_load_query_per_guild_then_served_from_memory():
    base, st = wrapped()
    base.put_summary(summ("2026-10-01"))
    base.client.streams.clear()
    st.summaries(GUILD, {10}, "2026-10-01", "2026-10-31")
    st.summary_entries(GUILD, {10}, "2026-10-01", "2026-10-31")
    assert len(loads(base)) == 1 and len(base.client.streams) == 1


def test_writes_keep_the_index_fresh(settings):
    base, st = wrapped()
    st.summaries(GUILD, {10}, "2026-10-01", "2026-10-31")  # load
    st.put_summary(summ("2026-10-02", text="ค่าเช่า"))
    v = vectorize({"overview": "ค่าเช่า"}, None, FakeEmbedder())
    st.put_summary_vectors(GUILD, 10, "2026-10-02", v.model_id, v.hashes, v.data)
    [e] = st.summary_entries(GUILD, {10}, "2026-10-01", "2026-10-31")
    assert e.items == ["ค่าเช่า"] and len(e.vectors) == 1
    st.add_message(mk(settings, 1, bkk(2, 9).astimezone(UTC)))
    st.delete_messages(GUILD, [1])
    assert st.summary_entries(GUILD, {10}, "2026-10-02", "2026-10-02")[0].summary.dirty
    st.delete_guild(GUILD)
    assert st.summary_entries(GUILD, {10}, "2026-10-01", "2026-10-31") == []
    assert len(loads(base)) == 2  # reloaded once after delete_guild


def test_purge_drops_the_cache():
    base, st = wrapped()
    base.put_summary(summ("2026-10-01"))
    st.summaries(GUILD, {10}, "2026-10-01", "2026-10-31")
    st.purge({GUILD: datetime(2026, 10, 5, tzinfo=UTC)}, datetime(2026, 1, 1, tzinfo=UTC))
    assert st.summaries(GUILD, {10}, "2026-10-01", "2026-10-31") == [] and len(loads(base)) == 2


def test_lru_evicts_by_bytes():
    base, st = wrapped(max_bytes=1)
    base.put_summary(summ("2026-10-01", gid=GUILD))
    base.put_summary(summ("2026-10-01", gid=GUILD + 1))
    st.summaries(GUILD, {10}, "2026-10-01", "2026-10-01")
    st.summaries(GUILD + 1, {10}, "2026-10-01", "2026-10-01")
    st.summaries(GUILD, {10}, "2026-10-01", "2026-10-01")
    assert len(loads(base)) == 3  # GUILD was evicted by GUILD + 1 and reloaded


def test_summaries_match_the_base_store():
    base, st = wrapped()
    for d, cid in (("2026-10-01", 10), ("2026-10-02", 11), ("2026-10-03", 10)):
        base.put_summary(summ(d, cid))
    for allowed, a, b in (
        ({10}, "2026-10-01", "2026-10-03"),
        ({10, 11}, "2026-10-02", "2026-10-02"),
        (set(), "2026-10-01", "2026-10-03"),
    ):
        assert st.summaries(GUILD, allowed, a, b) == base.summaries(GUILD, allowed, a, b)


def test_summary_entries_returns_a_snapshot():
    base, st = wrapped()
    base.put_summary(summ("2026-10-01"))
    got = st.summary_entries(GUILD, {10}, "2026-10-01", "2026-10-01")
    got.clear()
    assert len(st.summary_entries(GUILD, {10}, "2026-10-01", "2026-10-01")) == 1
