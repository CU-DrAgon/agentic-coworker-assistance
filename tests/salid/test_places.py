# ruff: noqa: F811, F401
"""Remembered places: who may know a company / personal place (channels, hierarchy, sharing, voice
rooms), learning from conversation, the consent flow and the scheduling location. Offline."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from conftest import FakeClient, json_resp
from test_secretary import ARGS, START, bgloop, w  # noqa: F401  (fixtures)

from salid import places as pl
from salid.coo_store import FirestoreDocStore, SQLiteDocStore
from salid.guild_settings import FEATURES, Features
from salid.places import (
    Candidate,
    PlaceRepo,
    PlacesCtx,
    PlaceTools,
    build_place_tools,
    extract_candidates,
    learn,
    looks_like_place,
    maps_link,
    may_know,
    place_audience,
    place_for_event,
)

G = 1000
OWNER, BOSS, STAFF, PEER, INTERN = 1, 25, 26, 27, 28
RANKS = {OWNER: 2_000_000, BOSS: 50, STAFF: 10, PEER: 10, INTERN: 5}
PUBLIC, MGR, LOW = 10, 11, 12
VIEWERS = {
    PUBLIC: frozenset(RANKS),
    MGR: frozenset({OWNER, BOSS}),
    LOW: frozenset({STAFF, INTERN}),
}
NAMES = {OWNER: "Owner", BOSS: "Boss", STAFF: "Somchai", PEER: "Peer", INTERN: "Ploy"}
CHANNELS = {PUBLIC: "ทั่วไป", MGR: "ผู้บริหาร", LOW: "ทีมเล็ก"}
AT = datetime(2026, 10, 14, 3, 0, tzinfo=UTC)
ADDR = "99 ถนนสุขุมวิท แขวงบางนา เขตบางนา กรุงเทพ 10260"


def viewers_of(cid):
    return VIEWERS.get(cid, frozenset())


@pytest.fixture(params=["sqlite", "firestore"])
def repo(request):
    if request.param == "sqlite":
        return PlaceRepo(SQLiteDocStore(":memory:"))
    from fake_firestore import FakeFirestore

    return PlaceRepo(FirestoreDocStore(client=FakeFirestore()))


def ctx(repo, asker, *, channel=PUBLIC, audience="auto", room=None, is_exec=False, confirm=None):
    if audience == "auto":
        audience = None if channel is None else viewers_of(channel) | {asker}
    return PlacesCtx(
        guild_id=G, asker_id=asker, is_exec=is_exec, repo=repo, ranks=RANKS,
        viewers_of=viewers_of, audience=audience, channel_id=channel, people=NAMES,
        channels=CHANNELS, confirm=confirm, room=room, now=lambda: AT,
    )  # fmt: skip


def company(repo, channel, name="สาขาบางนา", by=BOSS, address=ADDR, aliases=()):
    return repo.create(
        G, name=name, address=address, kind="branch", created_by=by,
        aliases=list(aliases), sources=[pl.make_source(channel, by, AT)],
    )  # fmt: skip


def who(repo, place):
    return pl.place_audience(place, RANKS, viewers_of)


# ------------------------------------------------------------------ company visibility
def test_company_place_from_a_private_channel_stays_there_plus_hierarchy(repo):
    p = company(repo, MGR)
    assert who(repo, p) == {OWNER, BOSS}  # readers of the channel (the owner outranks them)
    assert not may_know(p, STAFF, RANKS, viewers_of)
    assert may_know(p, BOSS, RANKS, viewers_of)


def test_a_later_public_mention_widens_it_and_never_narrows(repo):
    p = company(repo, MGR)
    cand = Candidate(name="สาขาบางนา", address=ADDR, kind="branch", scope="company")
    r = learn(repo, [cand], guild_id=G, author_id=STAFF, channel_id=PUBLIC, message_id=5,
              link="https://discord.com/x", at=AT)  # fmt: skip
    assert [x.id for x in r.widened] == [p.id] and r.created == []
    p = repo.get(G, p.id)
    assert may_know(p, INTERN, RANKS, viewers_of)
    assert [s["channel_id"] for s in p.sources] == [MGR, PUBLIC]
    # the same channel again changes nothing
    again = learn(
        repo, [cand], guild_id=G, author_id=PEER, channel_id=PUBLIC, message_id=6, link="", at=AT
    )
    assert again.widened == [] and len(repo.get(G, p.id).sources) == 2


def test_hierarchy_higher_roles_see_what_lower_roles_can_see(repo):
    p = company(repo, LOW, by=STAFF)
    assert may_know(p, STAFF, RANKS, viewers_of) and may_know(p, INTERN, RANKS, viewers_of)
    assert may_know(p, BOSS, RANKS, viewers_of) and may_know(p, OWNER, RANKS, viewers_of)
    # Peer outranks Ploy, who reads the channel, so Peer may know it too (never the reverse)
    assert may_know(p, PEER, RANKS, viewers_of)
    top = company(repo, MGR, name="อีกที่", address="other 12345678")
    assert not may_know(top, INTERN, RANKS, viewers_of)  # lower never sees higher's


def test_executive_secret_resets_the_sources(repo):
    p = company(repo, PUBLIC)
    assert may_know(p, INTERN, RANKS, viewers_of)
    t = PlaceTools(ctx(repo, OWNER, channel=None, is_exec=True))
    assert (
        PlaceTools(ctx(repo, BOSS, is_exec=False)).set_place_secret("สาขาบางนา", ["ผู้บริหาร"])[
            "error"
        ]
        == "not_allowed"
    )
    r = t.set_place_secret("สาขาบางนา", ["ผู้บริหาร"])
    assert r["status"] == "updated"
    p = repo.get(G, p.id)
    assert not may_know(p, INTERN, RANKS, viewers_of) and may_know(p, BOSS, RANKS, viewers_of)
    assert t.set_place_secret("สาขาบางนา", ["nope"])["error"] == "unknown channels"


def test_a_dm_source_is_visible_to_that_person_and_their_superiors(repo):
    p = company(repo, None, by=STAFF)  # make_source(None) -> the person's DM scope
    assert who(repo, p) >= {STAFF, BOSS, OWNER} and not may_know(p, PEER, RANKS, viewers_of)


# ------------------------------------------------------------------ personal places
def personal(repo, owner=STAFF, **kw):
    return repo.create(
        G, name="บ้านสมชาย", address=ADDR, kind="home", owner_id=owner, created_by=owner, **kw
    )


def test_personal_place_is_owner_only_even_for_the_boss(repo):
    p = personal(repo)
    assert who(repo, p) == {STAFF}
    assert not may_know(p, BOSS, RANKS, viewers_of) and not may_know(p, OWNER, RANKS, viewers_of)
    assert PlaceTools(ctx(repo, BOSS, channel=None)).find_place("บ้านสมชาย")["status"] == "not_found"
    assert (
        PlaceTools(ctx(repo, STAFF, channel=None)).find_place("บ้าน")["places"][0]["address"] == ADDR
    )


def test_owner_shares_with_superiors_named_people_and_everyone(repo):
    personal(repo)
    mine = PlaceTools(ctx(repo, STAFF, channel=None))
    assert mine.share_place("บ้านสมชาย", "superiors")["status"] == "updated"
    p = repo.search(G, "บ้านสมชาย")[0]
    assert who(repo, p) == {STAFF, BOSS, OWNER} and not may_know(p, PEER, RANKS, viewers_of)
    assert mine.share_place("บ้านสมชาย", "people", ["Peer"])["status"] == "updated"
    assert may_know(repo.search(G, "บ้าน")[0], PEER, RANKS, viewers_of)
    assert mine.share_place("บ้านสมชาย", "people", ["ghost"])["error"]
    assert mine.share_place("บ้านสมชาย", "everyone")["status"] == "updated"
    assert who(repo, repo.search(G, "บ้าน")[0]) == set(RANKS)
    assert mine.share_place("บ้านสมชาย", "owner")["status"] == "updated"  # back to private
    assert who(repo, repo.search(G, "บ้าน")[0]) == {STAFF}


def test_others_can_never_change_someone_elses_personal_place(repo):
    personal(repo)
    for uid in (BOSS, OWNER):
        t = PlaceTools(ctx(repo, uid, channel=None, is_exec=True))
        assert t.share_place("บ้านสมชาย", "everyone")["status"] == "not_found"
        assert t.forget_place("บ้านสมชาย")["status"] == "not_found"
    p = repo.search(G, "บ้านสมชาย")[0]
    p.share_everyone = True  # even when they can see it, they cannot change it
    repo.put(p)
    t = PlaceTools(ctx(repo, BOSS, channel=None, is_exec=True))
    assert t.share_place("บ้านสมชาย", "owner")["error"] == "not_allowed"
    assert t.forget_place("บ้านสมชาย")["error"] == "not_allowed"
    r = t.remember_place("บ้านสมชาย", "elsewhere 1", personal=True)  # their own, a different one
    assert r["status"] == "saved" and repo.get(G, p.id).address == ADDR


def test_someone_else_posting_a_personal_address_does_not_widen_it_the_owner_does(repo):
    p = personal(repo)
    other = Candidate(name="บ้านสมชาย", address=ADDR, scope="other")
    r = learn(
        repo, [other], guild_id=G, author_id=PEER, channel_id=PUBLIC, message_id=1, link="", at=AT
    )
    assert (r.created, r.widened, r.offers) == ([], [], [])
    assert who(repo, repo.get(G, p.id)) == {STAFF}
    mine = Candidate(name="บ้านสมชาย", address=ADDR, scope="own_home")
    # a different person claiming the same address as theirs is only an offer to them
    r = learn(
        repo, [mine], guild_id=G, author_id=PEER, channel_id=PUBLIC, message_id=2, link="", at=AT
    )
    assert r.widened == [] and who(repo, repo.get(G, p.id)) == {STAFF}
    r = learn(
        repo, [mine], guild_id=G, author_id=STAFF, channel_id=PUBLIC, message_id=3, link="", at=AT
    )
    assert [x.id for x in r.widened] == [p.id]
    assert who(repo, repo.get(G, p.id)) == set(RANKS)  # the public channel's audience


def test_owner_stating_it_in_a_private_channel_widens_only_to_that_channel(repo):
    p = personal(repo, owner=INTERN)
    p.name = "บ้านน้อง"
    repo.put(p)
    mine = Candidate(name="บ้านน้อง", address=ADDR, scope="own_home")
    learn(repo, [mine], guild_id=G, author_id=INTERN, channel_id=LOW, message_id=3, link="", at=AT)
    assert who(repo, repo.get(G, p.id)) == {STAFF, INTERN}


# ------------------------------------------------------------------ consent flow
def test_consent_nothing_stored_before_yes_and_asked_only_once(repo):
    mine = Candidate(name="บ้าน", address=ADDR, scope="own_home")
    kw = dict(guild_id=G, author_id=STAFF, channel_id=PUBLIC, message_id=1, link="", at=AT)
    r = learn(repo, [mine], **kw)
    assert len(r.offers) == 1 and repo.list(G) == []  # an offer, nothing stored
    r = learn(repo, [mine], **{**kw, "message_id": 2})
    assert r.offers == [] and repo.list(G) == []  # never ask twice for the same place
    # saying yes goes through the tool, for themselves: private by default
    t = PlaceTools(ctx(repo, STAFF, channel=None))
    saved = t.remember_place("บ้าน", ADDR, kind="home", personal=True)
    assert saved["status"] == "saved"
    (p,) = repo.list(G)
    assert p.personal and p.owner_id == STAFF and who(repo, p) == {STAFF} and p.sources == []


def test_an_addressed_message_makes_no_offer_the_agent_handles_it(repo):
    mine = Candidate(name="บ้าน", address=ADDR, scope="own_home")
    r = learn(repo, [mine], guild_id=G, author_id=STAFF, channel_id=PUBLIC, message_id=1, link="",
              at=AT, addressed=True)  # fmt: skip
    assert r.offers == [] and repo.list(G) == []


def test_a_personal_place_saved_by_tool_is_never_offered_afterwards(repo):
    PlaceTools(ctx(repo, STAFF, channel=None)).remember_place("บ้าน", ADDR, "home", True)
    mine = Candidate(name="บ้านใหม่", address=ADDR, scope="own_home")
    r = learn(
        repo, [mine], guild_id=G, author_id=STAFF, channel_id=LOW, message_id=1, link="", at=AT
    )
    assert r.offers == []


# ------------------------------------------------------------------ learning
def test_company_place_is_created_with_the_channel_as_source(repo):
    c = Candidate(name="สำนักงานใหญ่", address=ADDR, kind="hq", scope="company", aliases=["HQ"])
    r = learn(repo, [c], guild_id=G, author_id=BOSS, channel_id=MGR, message_id=7,
              link="https://discord.com/channels/1/2/7", at=AT)  # fmt: skip
    (p,) = r.created
    assert p.kind == "hq" and p.owner_id is None and p.aliases == ["HQ"]
    assert p.sources[0]["message_link"].endswith("/7") and p.sources[0]["disclosed_by"] == BOSS
    assert not may_know(p, STAFF, RANKS, viewers_of)


def test_junk_candidates_are_ignored(repo):
    bad = [
        Candidate(name="ร้านอาหาร", address=ADDR, scope="other"),
        Candidate(name="สาขา", address="", scope="company"),
        Candidate(name="", address=ADDR, scope="company"),
    ]
    r = learn(
        repo, bad, guild_id=G, author_id=BOSS, channel_id=PUBLIC, message_id=1, link="", at=AT
    )
    assert repo.list(G) == [] and not r.created


@pytest.mark.parametrize(
    "text",
    ["สาขาบางนาอยู่ที่ เลขที่ 99 ถนนบางนา", "our HQ is at 12 Silom Road", "https://maps.app.goo.gl/abc123 ไปที่นี่",
     "บ้านผมอยู่ซอยอารีย์ 5", "visit the branch office tomorrow", "รหัสไปรษณีย์ 10260"],
)  # fmt: skip
def test_prefilter_lets_address_like_text_through(text):
    assert looks_like_place(text)


@pytest.mark.parametrize(
    "text", ["โอเคครับ", "ประชุมพรุ่งนี้ 10 โมง", "", "thanks!", "ใบเสร็จค่ากาแฟ 120 บาท"]
)
def test_prefilter_drops_ordinary_chat(text):
    assert not looks_like_place(text)


def test_one_cheap_call_extracts_candidates_and_fails_quiet():
    client = FakeClient(
        [
            json_resp(
                {
                    "places": [
                        {"name": "สาขาบางนา", "address": ADDR, "kind": "branch", "scope": "company"}
                    ]
                }
            )
        ]
    )
    (c,) = extract_candidates(client, "lite", "สาขาบางนาอยู่ที่ " + ADDR, "Boss")
    assert c.scope == "company" and len(client.models.calls) == 1
    assert client.models.calls[0][0] == "lite"
    assert extract_candidates(FakeClient([SimpleNamespace(text="not json")]), "m", "x", "a") == []


# ------------------------------------------------------------------ tools and the answer audience
def test_remember_place_in_a_channel_uses_that_channel_as_the_source(repo):
    t = PlaceTools(ctx(repo, BOSS, channel=MGR))
    r = t.remember_place("สาขาบางนา", ADDR, "branch", aliases=["Bangna"])
    assert r["status"] == "saved" and r["place"]["maps_link"].startswith(pl.MAPS_SEARCH)
    (p,) = repo.list(G)
    assert [s["channel_id"] for s in p.sources] == [MGR]
    # said again in the public channel by someone who typed the same address: it widens
    t2 = PlaceTools(ctx(repo, STAFF, channel=PUBLIC))
    assert t2.remember_place("สาขาบางนา", ADDR)["status"] == "updated"
    (p,) = repo.list(G)
    assert [s["channel_id"] for s in p.sources] == [MGR, PUBLIC]
    # now everybody can see it: another address under the same name is not theirs to change
    t3 = PlaceTools(ctx(repo, PEER, channel=LOW))
    assert t3.remember_place("สาขาบางนา", "another address 999")["error"] == "not_allowed"
    # but a place nobody in this audience may know is never touched or revealed by a name alone
    company(repo, MGR, name="ที่ลับ", address="secret road 12345")
    r = PlaceTools(ctx(repo, PEER, channel=LOW)).remember_place("ที่ลับ", "my own idea 9876")
    assert r["status"] == "saved" and len(repo.list(G)) == 3
    assert repo.search(G, "ที่ลับ")[0].address in ("secret road 12345", "my own idea 9876")


def test_changing_an_address_needs_the_creator_or_an_executive(repo):
    company(repo, PUBLIC, by=BOSS)
    r = PlaceTools(ctx(repo, STAFF)).remember_place("สาขาบางนา", "somewhere else 5")
    assert r["error"] == "not_allowed"
    assert (
        PlaceTools(ctx(repo, BOSS)).remember_place("สาขาบางนา", "new addr 12345")["status"]
        == "updated"
    )


def test_find_place_matches_aliases_and_hides_what_the_asker_may_not_know(repo):
    company(repo, MGR, aliases=["Bangna branch"])
    assert PlaceTools(ctx(repo, BOSS, channel=MGR)).find_place("bangna")["count"] == 1
    r = PlaceTools(ctx(repo, STAFF)).find_place("bangna")
    assert r["status"] == "not_found" and ADDR not in str(r)
    assert PlaceTools(ctx(repo, STAFF)).list_places()["count"] == 0


def test_the_owner_sees_who_can_see_it_others_do_not(repo):
    personal(repo).share_superiors = True
    p = repo.search(G, "บ้านสมชาย")[0]
    p.share_superiors = True
    repo.put(p)
    owner_view = PlaceTools(ctx(repo, STAFF, channel=None)).find_place("บ้านสมชาย")["places"][0]
    assert "people ranked above the owner" in owner_view["who_can_see"]
    boss_view = PlaceTools(ctx(repo, BOSS, channel=None)).find_place("บ้านสมชาย")["places"][0]
    assert "who_can_see" not in boss_view and boss_view["owner"] == "Somchai"


def test_answer_in_a_public_channel_withholds_a_place_not_everyone_there_may_know(repo):
    company(repo, MGR)
    # the boss asks in #general: the staff listening may not know it
    in_public = PlaceTools(ctx(repo, BOSS, channel=PUBLIC))
    assert in_public.find_place("สาขาบางนา")["status"] == "not_found"
    assert in_public.list_places()["count"] == 0
    in_mgr = PlaceTools(ctx(repo, BOSS, channel=MGR))
    assert in_mgr.find_place("สาขาบางนา")["count"] == 1
    assert PlaceTools(ctx(repo, BOSS, channel=None)).find_place("สาขาบางนา")["count"] == 1  # a DM


def test_a_personal_place_is_not_read_out_in_a_public_channel(repo):
    personal(repo)
    assert (
        PlaceTools(ctx(repo, STAFF, channel=PUBLIC)).find_place("บ้านสมชาย")["status"] == "not_found"
    )
    assert PlaceTools(ctx(repo, STAFF, channel=None)).find_place("บ้านสมชาย")["count"] == 1


def test_voice_room_only_places_every_human_present_may_know(repo):
    company(repo, MGR)
    both_exec = ctx(
        repo, BOSS, channel=None, audience=frozenset({BOSS, OWNER}), room=frozenset({BOSS, OWNER})
    )
    assert PlaceTools(both_exec).find_place("สาขาบางนา")["count"] == 1
    mixed = ctx(
        repo, BOSS, channel=None, audience=frozenset({BOSS, STAFF}), room=frozenset({BOSS, STAFF})
    )
    assert PlaceTools(mixed).find_place("สาขาบางนา")["status"] == "not_found"
    assert PlaceTools(mixed).list_places()["count"] == 0
    # told in a room: only the people present (and their superiors) may know it afterwards
    room = ctx(
        repo,
        STAFF,
        channel=None,
        audience=frozenset({STAFF, PEER}),
        room=frozenset({STAFF, PEER}),
    )
    PlaceTools(room).remember_place("โกดังรังสิต", "5 รังสิต ปทุมธานี 12000", "warehouse")
    p = repo.search(G, "โกดังรังสิต")[0]
    assert who(repo, p) == {STAFF, PEER, BOSS, OWNER} and not may_know(p, INTERN, RANKS, viewers_of)


def test_forgetting_a_company_place_asks_first_and_personal_does_not(repo):
    company(repo, PUBLIC, by=BOSS)
    seen = []

    def confirm(key):
        first = key not in seen
        seen.append(key)
        return not first

    assert PlaceTools(ctx(repo, STAFF)).forget_place("สาขาบางนา")["error"] == "not_allowed"
    t = PlaceTools(ctx(repo, BOSS, confirm=confirm))
    assert t.forget_place("สาขาบางนา")["status"] == "needs_confirmation" and repo.list(G)
    assert t.forget_place("สาขาบางนา")["status"] == "forgotten" and repo.list(G) == []
    personal(repo)
    assert (
        PlaceTools(ctx(repo, STAFF, channel=None)).forget_place("บ้านสมชาย")["status"] == "forgotten"
    )
    assert repo.list(G) == []


def test_share_company_place_to_a_channel_widens_it(repo):
    company(repo, MGR, by=BOSS)
    t = PlaceTools(ctx(repo, BOSS, channel=MGR))
    assert t.share_place("สาขาบางนา", "superiors")["error"]  # company places are shared by channel
    assert (
        PlaceTools(ctx(repo, STAFF, channel=PUBLIC)).share_place("สาขาบางนา", "channel")["status"]
        == "not_found"
    )
    PlaceTools(ctx(repo, BOSS, channel=MGR)).share_place("สาขาบางนา", "everyone")
    assert who(repo, repo.list(G)[0]) == set(RANKS)


def test_tools_are_registered_as_a_group(repo):
    fns = build_place_tools(ctx(repo, STAFF))
    assert {f.__name__ for f in fns} == {
        "remember_place", "find_place", "list_places", "share_place", "forget_place",
        "set_place_secret",
    }  # fmt: skip
    assert all(f.salid_group == "places" for f in fns)


def test_maps_link_needs_no_api_key_and_is_url_encoded(repo):
    p = company(repo, PUBLIC, address="99 Sukhumvit Rd, Bangna & Co")
    assert maps_link(p) == (
        "https://www.google.com/maps/search/?api=1&query=99%20Sukhumvit%20Rd%2C%20Bangna%20%26%20Co"
    )
    p.maps_url = "https://maps.app.goo.gl/x"
    assert maps_link(p) == "https://maps.app.goo.gl/x"


def test_feature_flag_defaults_on():
    assert "places" in FEATURES and Features().places is True


def test_opaque_places_survive_retention_nothing_to_do_with_memory(repo):
    # places live in the doc store, not in the memory log: a deleted source message keeps the place
    p = company(repo, PUBLIC)
    assert repo.get(G, p.id) is not None and repo.get(2000, p.id) is None  # and per guild


# ------------------------------------------------------------------ scheduling
def test_place_for_event_only_when_every_invitee_may_know(repo):
    company(repo, PUBLIC)
    ok = place_for_event(ctx(repo, BOSS), "สาขาบางนา", [STAFF, PEER])
    assert ok["status"] == "ok" and ok["address"] == ADDR and ok["public"] is True
    repo.delete(repo.list(G)[0])
    company(repo, MGR)
    r = place_for_event(ctx(repo, BOSS), "สาขาบางนา", [OWNER, STAFF])
    assert r == {"status": "restricted", "place": "สาขาบางนา", "cannot_know": ["Somchai"]}
    assert place_for_event(ctx(repo, BOSS), "สาขาบางนา", [OWNER])["public"] is False
    assert place_for_event(ctx(repo, STAFF), "สาขาบางนา", [])["status"] == "not_found"
    assert place_for_event(ctx(repo, BOSS), "ที่อื่น", [])["status"] == "not_found"


def _places_for(w, repo, asker):
    c = ctx(repo, asker)
    c.ranks = {1: 100, 10: 5, 11: 5, 12: 5}
    c.people = {1: "Boss", 10: "Ann", 11: "Bob", 12: "Cat"}
    return lambda q, ids: place_for_event(c, q, ids)


def test_schedule_fills_the_location_from_a_place_all_invitees_know(w, repo):
    c = ctx(repo, 1)
    c.ranks = {1: 100, 10: 5, 11: 5, 12: 5}
    c.people = {1: "Boss", 10: "Ann", 11: "Bob", 12: "Cat"}
    c.asker_id = 1
    repo.create(G, name="สาขาบางนา", address=ADDR, kind="branch", created_by=1,
                sources=[pl.make_source(None, 1, AT, room=frozenset({1, 10, 11, 12}))])  # fmt: skip
    resolver = lambda q, ids: place_for_event(c, q, ids)  # noqa: E731
    args = {**ARGS, "place": "สาขาบางนา"}
    first = w.tools(places=resolver).schedule_meeting(**args)
    assert first["status"] == "needs_confirmation" and first["proposal"]["where"] == "สาขาบางนา"
    r = w.tools(places=resolver).schedule_meeting(**args)
    assert r["status"] == "created"
    (kw,) = w.guild.created
    assert kw["location"] == ADDR[:100] and "channel" not in kw  # everyone may know it: public
    assert any(ADDR in m and "google.com/maps" in m for m in w.ann.sent)  # the invite has it
    (m,) = w.store.list_meetings(GUILD)
    assert ADDR not in str(m)  # not kept on the meeting record


def test_schedule_asks_when_some_invitee_may_not_know_the_place(w, repo):
    c = ctx(repo, 1)
    c.ranks = {1: 100, 10: 5, 11: 5, 12: 5}
    c.people = {1: "Boss", 10: "Ann", 11: "Bob", 12: "Cat"}
    repo.create(
        G,
        name="สาขาบางนา",
        address=ADDR,
        kind="branch",
        created_by=1,
        sources=[pl.make_source(None, 1, AT)],
    )  # fmt: skip  (a DM of the boss: private)
    resolver = lambda q, ids: place_for_event(c, q, ids)  # noqa: E731
    r = w.tools(places=resolver).schedule_meeting(**{**ARGS, "place": "สาขาบางนา"})
    assert r["status"] == "needs_info" and set(r["place_restricted_for"]) == {"Ann", "Bob"}
    assert ADDR not in str(r) and w.guild.created == []
    r = w.tools(places=resolver).schedule_meeting(**{**ARGS, "place": "ไม่มีที่นี่"})
    assert r["status"] == "needs_info" and "No remembered place" in r["instruction"]
    # without a place the old behaviour is untouched
    assert w.tools(places=resolver).schedule_meeting(**ARGS)["status"] == "needs_confirmation"


from test_secretary import GUILD  # noqa: E402
