"""Role hierarchy ("higher sees lower, lower cannot see higher"): the pure rules in
`memory_access`, and the memory tools that apply them (person filter, withheld records)."""

from datetime import UTC, datetime, timedelta

import pytest
from conftest import GUILD, FixedClockSettings
from test_memory import mk  # noqa: F401  (message builder)

from salid.memory import build_message
from salid.memory_access import (
    ADMIN_RANK,
    OWNER_RANK,
    Scope,
    Visibility,
    audience_authors,
    dm_scope,
    hidden_but_askable,
    hierarchy_authors,
    member_rank,
    message_visible,
)
from salid.memory_store import SQLiteMemoryStore
from salid.qa import MemoryTools

OWNER, ADMIN, BOSS, STAFF, PEER, INTERN = 1, 2, 3, 4, 5, 6
RANKS = {
    OWNER: member_rank(is_owner=True, is_admin=True, top_role_position=9),
    ADMIN: member_rank(is_owner=False, is_admin=True, top_role_position=7),
    BOSS: member_rank(is_owner=False, is_admin=False, top_role_position=5),
    STAFF: member_rank(is_owner=False, is_admin=False, top_role_position=2),
    PEER: member_rank(is_owner=False, is_admin=False, top_role_position=2),
    INTERN: member_rank(is_owner=False, is_admin=False, top_role_position=1),
}
PUBLIC, PRIVATE_CH = 10, 11


class M:  # a stand-in for MemoryMessage
    def __init__(self, author, channel, kind="chat", visibility="normal", mid=1):
        self.author_id, self.channel_id, self.kind = author, channel, kind
        self.visibility, self.message_id = visibility, mid


def test_rank_order_owner_admin_role_position():
    assert member_rank(is_owner=True, is_admin=False, top_role_position=0) == OWNER_RANK
    assert RANKS[OWNER] > RANKS[ADMIN] > RANKS[BOSS] > RANKS[STAFF] == RANKS[PEER] > RANKS[INTERN]
    assert RANKS[ADMIN] >= ADMIN_RANK  # Administrator counts as top, above any plain role


def test_asker_sees_themself_and_strictly_lower_ranks_only():
    assert hierarchy_authors(BOSS, RANKS) == {BOSS, STAFF, PEER, INTERN}
    assert hierarchy_authors(STAFF, RANKS) == {STAFF, INTERN}  # same rank: not visible
    assert hierarchy_authors(INTERN, RANKS) == {INTERN}
    assert hierarchy_authors(OWNER, RANKS) == set(RANKS)  # the owner outranks everyone
    assert ADMIN not in hierarchy_authors(BOSS, RANKS)  # lower never sees higher
    assert hierarchy_authors(999, RANKS) == frozenset()  # unknown asker


def test_audience_rule_never_widens_the_circle():
    # a channel the boss and a staff member read: the boss's own hidden info stays out (the staff
    # member does not outrank them), the staff member's and the intern's may be shown
    assert hierarchy_authors(BOSS, RANKS, {BOSS, STAFF}) == {STAFF, INTERN}
    assert hierarchy_authors(BOSS, RANKS, {BOSS, INTERN}) == {INTERN}
    # a channel only the owner and an admin read
    assert hierarchy_authors(OWNER, RANKS, {OWNER, ADMIN}) == set(RANKS) - {OWNER}
    assert hierarchy_authors(OWNER, RANKS, {OWNER}) == set(RANKS)
    assert hierarchy_authors(OWNER, RANKS, None) == set(RANKS)  # a DM


def test_audience_authors_is_what_every_reader_may_see():
    assert audience_authors(RANKS, {OWNER}) == set(RANKS)
    assert audience_authors(RANKS, {OWNER, ADMIN}) == set(RANKS) - {OWNER}
    assert audience_authors(RANKS, {BOSS, STAFF}) == {STAFF, INTERN}
    assert audience_authors(RANKS, set()) == set(RANKS)  # nobody to protect


def scope(asker, *, audience=None, channels=(PUBLIC,), unlocked=(), dm=None):
    return Scope(
        asker_id=asker,
        channels=frozenset(channels),
        authors=hierarchy_authors(asker, RANKS, audience),
        dm_authors=hierarchy_authors(asker, RANKS, None),
        unlocked=frozenset(unlocked),
        dm=audience is None if dm is None else dm,
    )


def test_message_visibility_rules():
    boss = scope(BOSS)
    assert message_visible(M(STAFF, PRIVATE_CH), boss)  # hidden channel, lower rank
    assert message_visible(M(INTERN, PUBLIC), boss)  # a channel the asker reads
    assert not message_visible(M(ADMIN, PRIVATE_CH), boss)  # higher rank, hidden channel
    assert message_visible(M(ADMIN, PUBLIC), boss)  # readable channels stay visible as before
    staff = scope(STAFF)
    assert not message_visible(M(PEER, PRIVATE_CH), staff)  # same rank: only via shared channels
    assert not message_visible(M(BOSS, PRIVATE_CH), staff)
    assert message_visible(M(STAFF, PRIVATE_CH), staff)  # their own
    assert not message_visible(M(0, PRIVATE_CH, kind="salid"), scope(OWNER))  # Salid's own words


def test_dm_notes_author_hierarchy_and_private():
    note = M(STAFF, dm_scope(STAFF), kind="dm")
    secret = M(STAFF, dm_scope(STAFF), kind="dm", visibility="private")
    assert message_visible(note, scope(STAFF)) and message_visible(secret, scope(STAFF))
    assert message_visible(note, scope(BOSS))  # higher ranks read a lower rank's briefings
    assert not message_visible(secret, scope(BOSS))  # ... unless private (owner-only)
    assert not message_visible(secret, scope(OWNER))
    assert not message_visible(note, scope(PEER))  # same rank
    assert not message_visible(note, scope(INTERN))
    # in a channel the author also reads, the boss may bring them up (the author is the subject),
    # but not in front of someone who does not outrank the author
    assert message_visible(note, scope(BOSS, audience={BOSS, STAFF}))
    assert not message_visible(note, scope(BOSS, audience={BOSS, PEER}))
    # and the author's own notes are not repeated in front of equals
    assert not message_visible(note, scope(STAFF, audience={STAFF, PEER}))


def test_public_channel_info_is_open_to_everyone_who_reads_it():
    for who in (INTERN, STAFF, BOSS):
        assert message_visible(M(ADMIN, PUBLIC), scope(who))


def test_grants_unlock_only_those_messages_and_only_in_dms():
    hidden = M(BOSS, PRIVATE_CH, mid=77)
    other = M(BOSS, PRIVATE_CH, mid=78)
    assert not message_visible(hidden, scope(STAFF))
    s = scope(STAFF, unlocked={77})
    assert message_visible(hidden, s) and not message_visible(other, s)
    # a public channel reply would reach people who were not granted anything
    assert not message_visible(hidden, scope(STAFF, audience={STAFF, PEER}, unlocked={77}))
    # nobody else gets it
    assert not message_visible(hidden, scope(PEER))


def test_hidden_but_askable_excludes_private_notes_and_visible_ones():
    assert hidden_but_askable(M(BOSS, PRIVATE_CH), scope(STAFF))
    assert not hidden_but_askable(M(STAFF, PRIVATE_CH), scope(BOSS))
    assert not hidden_but_askable(
        M(BOSS, dm_scope(BOSS), kind="dm", visibility="private"), scope(STAFF)
    )
    assert hidden_but_askable(M(BOSS, dm_scope(BOSS), kind="dm"), scope(STAFF))


# ------------------------------------------------------------------ the memory tools
PEOPLE = {STAFF: "Somchai", BOSS: "Boss Ann", INTERN: "Ploy", PEER: "Peer"}


@pytest.fixture
def mem(tmp_path):
    settings = FixedClockSettings()
    store = SQLiteMemoryStore(str(tmp_path / "m.db"))
    at = datetime(2026, 10, 14, 5, 0, tzinfo=UTC)

    def add(mid, uid, text, cid, kind="chat", visibility="normal"):
        m = build_message(settings, guild_id=GUILD, channel_id=cid, message_id=mid, author_id=uid,
                          author_name=PEOPLE[uid], created_at=at + timedelta(minutes=mid), text=text,
                          kind=kind)  # fmt: skip
        store.add_message(m.model_copy(update={"visibility": visibility}))

    add(1, STAFF, "งบโฆษณา ส่งใบเสนอราคาลูกค้า X แล้ว", PRIVATE_CH)
    add(2, BOSS, "งบโฆษณา ปรับขึ้น 20% ห้ามบอกทีม", PRIVATE_CH)
    add(3, INTERN, "งบโฆษณา ทำโพสต์เสร็จแล้ว", PUBLIC)
    add(4, STAFF, "งบโฆษณา พรุ่งนี้ผมลา", dm_scope(STAFF), kind="dm")
    add(5, STAFF, "งบโฆษณา เรื่องส่วนตัวมาก", dm_scope(STAFF), kind="dm", visibility="private")
    names = {PUBLIC: "ทั่วไป", PRIVATE_CH: "ลับ"}
    read = {**names, **{dm_scope(u): "DM" for u in RANKS}}

    def tools(asker, audience=None, unlocked=()):
        sc = scope(asker, audience=audience, unlocked=unlocked)
        allowed = {PUBLIC: "ทั่วไป"}
        return MemoryTools(store, GUILD, allowed, settings, visibility=Visibility(sc, read, PEOPLE))

    return tools


def texts(res):
    return sorted(m["text"] for m in res["messages"])


def test_boss_reads_staff_work_in_a_dm_but_not_the_private_note(mem):
    r = mem(BOSS).chat_messages("2026-10-14", "2026-10-14", keyword="งบโฆษณา", person="somchai")
    assert sorted(m["channel"] for m in r["messages"]) == ["DM", "ลับ"]
    assert all("ส่วนตัวมาก" not in m["text"] for m in r["messages"])
    assert "withheld" not in r  # nothing above the boss matched the filter


def test_staff_cannot_read_the_boss_but_learns_something_is_withheld(mem):
    t = mem(STAFF)
    r = t.chat_messages("2026-10-14", "2026-10-14", keyword="งบโฆษณา")
    assert texts(r) == sorted(["งบโฆษณา ส่งใบเสนอราคาลูกค้า X แล้ว", "งบโฆษณา ทำโพสต์เสร็จแล้ว",
                               "งบโฆษณา พรุ่งนี้ผมลา", "งบโฆษณา เรื่องส่วนตัวมาก"])  # fmt: skip
    assert "ห้ามบอกทีม" not in str(r)  # the boss's content never appears
    assert "withheld" in r and "request_disclosure" in r["withheld"]
    assert [h.author_id for h in t.hidden] == [BOSS]


def test_withheld_is_counted_not_shown_in_search_too(mem):
    t = mem(PEER)
    r = t.chat_search(["งบโฆษณา"], "2026-10-14", "2026-10-14")
    assert "ห้ามบอกทีม" not in str(r) and "ส่งใบเสนอราคา" not in str(r)
    assert {h.author_id for h in t.hidden} == {STAFF, BOSS}  # same-rank staff is hidden too
    assert all(h.visibility != "private" for h in t.hidden)  # private notes are never hinted at
    assert "withheld" in r


def test_person_filter_by_name_and_mention(mem):
    t = mem(OWNER)
    by_name = t.chat_messages("2026-10-14", "2026-10-14", person="boss ann")
    by_mention = t.chat_messages("2026-10-14", "2026-10-14", person=f"<@{BOSS}>")
    assert texts(by_name) == texts(by_mention) == ["งบโฆษณา ปรับขึ้น 20% ห้ามบอกทีม"]
    assert mem(OWNER).chat_messages("2026-10-14", "2026-10-14", person="nobody")["count"] == 0


def test_dm_hint_when_a_dm_would_show_it(mem):
    t = mem(BOSS, audience={BOSS, STAFF})  # asked in a channel the staff member also reads
    t.chat_messages("2026-10-14", "2026-10-14", keyword="งบโฆษณา")
    assert (
        t.hidden_in_dm
        and "DM" in t.chat_messages("2026-10-14", "2026-10-14", keyword="งบโฆษณา")["withheld"]
    )


def test_grant_unlocks_exactly_the_approved_messages(mem):
    t = mem(STAFF, unlocked={2})
    r = t.chat_messages("2026-10-14", "2026-10-14", keyword="งบโฆษณา", person="boss")
    assert texts(r) == ["งบโฆษณา ปรับขึ้น 20% ห้ามบอกทีม"] and "withheld" not in r
    assert (
        mem(PEER, unlocked=()).chat_messages("2026-10-14", "2026-10-14", person="boss")["count"]
        == 0
    )


# ------------------------------------------------------------------ visibility in the stores
@pytest.fixture(params=["sqlite", "firestore"])
def mstore(request, tmp_path):
    if request.param == "sqlite":
        return SQLiteMemoryStore(str(tmp_path / "v.db"))
    from fake_firestore import offline_memory_store

    return offline_memory_store()


def test_visibility_is_stored_defaulted_and_changeable(mstore):
    s = FixedClockSettings()
    at = datetime(2026, 10, 14, 5, 0, tzinfo=UTC)
    mstore.add_message(build_message(s, guild_id=GUILD, channel_id=dm_scope(STAFF), message_id=1,
                                     author_id=STAFF, author_name="x", created_at=at, text="note",
                                     kind="dm"))  # fmt: skip
    ids = {dm_scope(STAFF)}
    (m,) = mstore.messages(GUILD, ids, "2026-10-14", "2026-10-14")
    assert (m.kind, m.visibility) == ("dm", "normal")
    assert mstore.set_visibility(GUILD, 1, "private") and not mstore.set_visibility(
        GUILD, 99, "private"
    )
    (m,) = mstore.messages(GUILD, ids, "2026-10-14", "2026-10-14")
    assert m.visibility == "private"
    assert mstore.set_visibility(GUILD + 1, 1, "private") is False  # another guild's id space


def test_an_old_sqlite_database_gets_the_visibility_column(tmp_path):
    import sqlite3

    path = str(tmp_path / "old.db")
    c = sqlite3.connect(path)
    c.execute(
        "CREATE TABLE salid_messages (guild_id INTEGER NOT NULL, message_id INTEGER NOT NULL, "
        "channel_id INTEGER NOT NULL, thread_id INTEGER, author_id INTEGER, author_name TEXT, "
        "created_at TEXT NOT NULL, local_date TEXT NOT NULL, text TEXT, jump_url TEXT, "
        "edited_at TEXT, kind TEXT, PRIMARY KEY (guild_id, message_id))"
    )
    c.execute(
        "INSERT INTO salid_messages VALUES (1000,1,10,NULL,5,'a','2026-10-14T05:00:00.000000+00:00',"
        "'2026-10-14','hello','',NULL,'chat')"
    )
    c.commit()
    c.close()
    store = SQLiteMemoryStore(path)
    (m,) = store.messages(1000, {10}, "2026-10-14", "2026-10-14")
    assert m.text == "hello" and m.visibility == "normal"
    assert store.set_visibility(1000, 1, "private")
