# ruff: noqa: F811, F401
"""Organizations store (SQLite + fake Firestore doc stores), license rules, grandfathering, the
bot's license gate, sharing safety for the bot tools, and the Slack signature check."""

import hashlib
import hmac
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from conftest import GUILD, FixedClockSettings
from fake_firestore import FakeFirestore
from test_bot_multitenant import STAFF, member
from test_modes import TMsg, bot, send, world  # noqa: F401  (fixtures)

from salid.connectors.slack import verify_signature
from salid.coo_store import FirestoreDocStore, SQLiteDocStore
from salid.guild_settings import MemoryGuildStore
from salid.i18n import t
from salid.licensing import Licensing, days_left, license_ok
from salid.orgs import (
    NAMESPACE_BASE,
    License,
    Member,
    OrgStore,
    Workspace,
    WorkspaceTaken,
    namespaced_id,
    trial_license,
    workspace_memory_id,
)
from salid.places import PlaceRepo, PlacesCtx, PlaceTools, make_source
from salid.sharing import Sharing, page_is_public, place_is_public, shared_sources
from salid.sharing import set_default as set_sharing_default
from salid.wiki import WikiHooks, WikiPage, WikiRepo, WikiTools

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)


@pytest.fixture(params=["sqlite", "firestore"])
def docs(request):
    if request.param == "sqlite":
        return SQLiteDocStore(":memory:")
    return FirestoreDocStore(client=FakeFirestore())


@pytest.fixture
def orgs(docs):
    return OrgStore(docs, clock=lambda: NOW)


def ws(platform, ext, name=""):
    from salid.orgs import workspace_id

    return Workspace(id=workspace_id(platform, ext), platform=platform, external_id=ext, name=name)


# ------------------------------------------------------------------ ids
def test_memory_ids_never_collide_with_discord():
    assert workspace_memory_id("discord", "1400000000000000000") == 1400000000000000000
    s = workspace_memory_id("slack", "T1")
    assert s >= NAMESPACE_BASE and s < 2**63
    assert s == workspace_memory_id("slack", "T1") != workspace_memory_id("clickup", "T1")
    assert namespaced_id("slack", "C1") != namespaced_id("slack", "C1", "user")


# ------------------------------------------------------------------ store
def test_create_lookup_and_indexes(orgs):
    org = orgs.create("Acme", "Boss@Example.com")
    assert org.owner_email == "boss@example.com" and org.role_of("BOSS@example.com") == "owner"
    assert [w.platform for w in org.workspaces] == ["web"]
    assert org.license.plan == "trial" and org.license.valid_until is None  # launch: open-ended
    assert (
        org.license.seats == 10_000 and trial_license(days=14, now=NOW).valid_until == "2026-10-30"
    )
    assert orgs.get(org.id).name == "Acme" and orgs.get("nope") is None
    assert [o.id for o in orgs.orgs_of_email("boss@example.com")] == [org.id]
    assert orgs.orgs_of_email("x@example.com") == []
    found = orgs.org_of_memory_id(org.web_workspace().memory_id)
    assert found[0].id == org.id and found[1].platform == "web"


def test_guild_belongs_to_one_org_and_detach_cleans_up(orgs):
    a, b = orgs.create("A", "a@x.com"), orgs.create("B", "b@x.com")
    orgs.attach_workspace(a.id, ws("discord", "42", "Guild"))
    assert orgs.org_of_guild(42).id == a.id
    with pytest.raises(WorkspaceTaken):
        orgs.attach_workspace(b.id, ws("discord", "42"))
    orgs.attach_workspace(a.id, ws("discord", "42", "Renamed"))  # idempotent in the same org
    assert orgs.get(a.id).workspace("discord-42").name == "Renamed"
    org = orgs.get(a.id)
    from salid.orgs import ShareLink

    org.sharing = [ShareLink(from_workspace="discord-42", to_workspace="web", categories=["wiki"])]
    orgs.save(org)
    orgs.detach_workspace(a.id, "discord-42")
    assert orgs.org_of_guild(42) is None and orgs.get(a.id).sharing == []
    orgs.attach_workspace(b.id, ws("discord", "42"))  # free again
    with pytest.raises(ValueError):
        orgs.detach_workspace(a.id, "web")
    with pytest.raises(KeyError):
        orgs.detach_workspace(a.id, "nope")


def test_member_index_follows_membership(orgs):
    org = orgs.create("A", "a@x.com")
    org.members.append(Member(email="Z@x.com", role="member"))
    orgs.save(org)
    assert [o.id for o in orgs.orgs_of_email("z@x.com")] == [org.id]
    org.members = [m for m in org.members if m.email.lower() != "z@x.com"]
    orgs.save(org)
    assert orgs.orgs_of_email("z@x.com") == []


# ------------------------------------------------------------------ license rules
def test_license_rules_grace_and_status():
    lic = License(plan="monthly", status="active", valid_until="2026-10-10")
    assert license_ok(lic, datetime(2026, 10, 12, tzinfo=UTC))  # inside the 3 day grace
    assert license_ok(lic, datetime(2026, 10, 13, 23, 0, tzinfo=UTC))  # grace ends with that day
    assert not license_ok(lic, datetime(2026, 10, 14, 1, 0, tzinfo=UTC))
    assert license_ok(License(status="active", valid_until=None), NOW)  # open ended
    assert license_ok(License(status="past_due", valid_until="2027-01-01"), NOW)
    for st in ("expired", "suspended"):
        assert not license_ok(License(status=st, valid_until="2099-01-01"), NOW)
    assert days_left(License(valid_until="2026-10-26"), NOW) == 10
    assert days_left(License(valid_until=None), NOW) is None


# ------------------------------------------------------------------ licensing / grandfathering
def mk_licensing(orgs, guilds, enforce=True, clock=None):
    return Licensing(
        orgs, guilds, enforce=enforce, now=lambda: NOW, clock=clock or (lambda: 0.0),
        hub_url="https://hub.example/",
    )  # fmt: skip


def test_enforce_off_allows_everything(orgs):
    assert mk_licensing(orgs, MemoryGuildStore(), enforce=False).is_licensed(1)


def test_unknown_guild_without_a_settings_doc_is_unlicensed(orgs):
    lic = mk_licensing(orgs, MemoryGuildStore())
    assert lic.is_licensed(5) is False
    assert lic.url == "https://hub.example/center"


def test_existing_guild_is_grandfathered_idempotently(orgs):
    guilds = MemoryGuildStore()
    guilds.upsert_guild(7, name="Old Cafe", icon_url="i", channels=[])
    code = guilds.create_link_code(7, 1, "u")
    guilds.redeem_link_code(code, "owner@x.com")
    lic = mk_licensing(orgs, guilds)
    assert lic.is_licensed(7) is True
    org = orgs.org_of_guild(7)
    assert org.id == "org_g7" and org.name == "Old Cafe" and org.grandfathered
    assert org.license.plan == "trial" and org.license.valid_until == "2026-12-31"
    assert org.role_of("owner@x.com") == "owner"
    lic.invalidate()
    assert lic.is_licensed(7) is True
    assert len(orgs.all_orgs()) == 1  # no second org
    # a second Licensing instance (another process) does not duplicate it either
    assert mk_licensing(orgs, guilds).is_licensed(7) and len(orgs.all_orgs()) == 1


def test_trial_expires_after_the_grace_period(orgs):
    guilds = MemoryGuildStore()
    guilds.upsert_guild(7, name="Old", icon_url="", channels=[])
    now = {"v": NOW}
    lic = Licensing(orgs, guilds, now=lambda: now["v"], clock=lambda: 0.0)
    assert lic.is_licensed(7)
    now["v"] = datetime(2027, 1, 5, tzinfo=UTC)
    lic.invalidate()
    assert lic.is_licensed(7) is False


def test_guild_installed_after_licensing_started_is_not_grandfathered(orgs):
    guilds = MemoryGuildStore()
    guilds.upsert_guild(1, name="Before", icon_url="", channels=[])
    lic = mk_licensing(orgs, guilds)
    assert lic.is_licensed(1)  # first lookup starts the epoch
    guilds.upsert_guild(2, name="New", icon_url="", channels=[])
    guilds._ref(2).set({"installed_at": "2026-10-17T00:00:00+00:00"}, merge=True)  # after the epoch
    assert lic.is_licensed(2) is False and orgs.org_of_guild(2) is None


def test_detached_grandfathered_guild_is_not_recreated(orgs):
    guilds = MemoryGuildStore()
    guilds.upsert_guild(7, name="Old", icon_url="", channels=[])
    lic = mk_licensing(orgs, guilds)
    assert lic.is_licensed(7)
    orgs.detach_workspace("org_g7", "discord-7")
    lic.invalidate()
    assert lic.is_licensed(7) is False


def test_cache_five_minutes_and_fail_open(orgs):
    guilds = MemoryGuildStore()
    guilds.upsert_guild(7, name="Old", icon_url="", channels=[])
    clock = {"t": 0.0}
    lic = mk_licensing(orgs, guilds, clock=lambda: clock["t"])
    assert lic.is_licensed(7)
    org = orgs.get("org_g7")
    org.license.status = "suspended"
    orgs.save(org)
    assert lic.is_licensed(7) is True  # cached
    clock["t"] = 301.0
    assert lic.is_licensed(7) is False
    orgs.docs = None  # the store breaks: keep the last known answer (fail open)
    clock["t"] = 700.0
    assert lic.is_licensed(7) is False
    assert lic.is_licensed(99) is True  # nothing known + failing lookup = allow


def test_daily_notice_claim(orgs):
    lic = mk_licensing(orgs, MemoryGuildStore())
    assert lic.claim_notice(1, "2026-10-16") is True
    assert lic.claim_notice(1, "2026-10-16") is False
    assert lic.claim_notice(2, "2026-10-16") is True  # per guild
    assert lic.claim_notice(1, "2026-10-17") is True  # per day


# ------------------------------------------------------------------ the bot gate
@pytest.fixture
def gated(bot, orgs, world):  # noqa: F811
    """The bot with the license gate ON and an empty org store (guild 1000 has no settings doc)."""
    lic = mk_licensing(orgs, bot.guilds_store)
    bot.licensing = lic
    bot.team.license_gate = lic.is_licensed
    bot.recorded, bot.calls = [], []

    async def record(message, own):
        bot.recorded.append(message.content)
        return False

    bot._memory_record = record
    return bot


def hub_line(bot):
    return t("unlicensed", "th", url=bot.licensing.url)


async def test_unlicensed_guild_is_silent_and_stores_nothing(gated, world):
    m1 = await send(gated, world, world.gen, "ยอดเท่าไร")
    m2 = await send(
        gated, world, world.gen, "photo", attachments=[SimpleNamespace(filename="a.png")]
    )
    assert gated.recorded == [] and gated.calls == []
    assert m1.replies == [] and m2.replies == [] and not m1.reactions


async def test_unlicensed_guild_gets_one_reply_per_day_when_addressed(gated, world):
    m1 = await send(gated, world, world.gen, "ยอดเท่าไร", mention=True)
    assert m1.replies == [hub_line(gated)] and "https://hub.example/center" in m1.replies[0]
    m2 = await send(gated, world, world.gen, "สลิด ช่วยหน่อย")
    assert m2.replies == []  # already told today
    assert gated.calls == [] and gated.recorded == []
    day2 = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)

    class NextDay(FixedClockSettings):
        def now(self):
            return day2.astimezone(self.zone)

    gated.s = NextDay(tz="Asia/Bangkok")
    m3 = await send(gated, world, world.gen, "สลิดอยู่ไหม")
    assert m3.replies == [hub_line(gated)]  # a new day


async def test_english_reply_when_addressed_in_english(gated, world):
    gated._lang = lambda target, text=None: "en"
    m = await send(gated, world, world.gen, "Salid how much?", mention=True)
    assert m.replies == [t("unlicensed", "en", url=gated.licensing.url)]


async def test_licensed_guild_works_normally(gated, world, orgs):
    org = orgs.create("A", "a@x.com", license=License(plan="monthly", valid_until="2099-01-01"))
    orgs.attach_workspace(org.id, ws("discord", str(GUILD)))
    gated.licensing.invalidate()
    m = await send(gated, world, world.gen, "ยอดเท่าไร", mention=True)
    assert gated.calls == [("answer", "ยอดเท่าไร")] and gated.recorded == ["ยอดเท่าไร"]
    assert hub_line(gated) not in m.replies


async def test_grandfathered_guild_works(gated, world):
    gated.guilds_store.upsert_guild(GUILD, name="A", icon_url="", channels=[])
    gated.licensing.invalidate()
    await send(gated, world, world.gen, "ยอดเท่าไร", mention=True)
    assert gated.calls == [("answer", "ยอดเท่าไร")]


def test_features_are_all_off_for_an_unlicensed_guild(gated):
    assert gated.team.enabled(GUILD, "memory") is False
    assert gated.team.enabled(GUILD, "auto_join_meetings") is False
    assert gated.team.enabled(GUILD, "receipts") is False


async def test_unlicensed_guild_does_not_auto_join_voice(gated, world):
    seen = []

    async def activity(*a):
        seen.append(a)

    gated.meetings.on_voice_activity = activity
    m = member(world.g, STAFF)
    await gated.on_voice_state_update(
        m, SimpleNamespace(channel=None), SimpleNamespace(channel=object())
    )
    assert seen == []
    gated.licensing.enforce = False
    await gated.on_voice_state_update(
        m, SimpleNamespace(channel=None), SimpleNamespace(channel=object())
    )
    assert len(seen) == 1


async def test_slash_commands_blocked_but_setup_link_flow_works(gated, world):
    from salid.bot import _LicensedTree

    class It:
        def __init__(self, name):
            self.client, self.guild = gated, world.g
            self.command = SimpleNamespace(name=name)
            self.sent = []
            self.response = SimpleNamespace(send_message=self.send)

        async def send(self, text, **kw):
            self.sent.append((text, kw))

    tree = gated.tree
    assert isinstance(tree, _LicensedTree)
    other = It("สรุป")
    assert await tree.interaction_check(other) is False
    assert other.sent == [(hub_line(gated), {"ephemeral": True})]
    for name in ("ตั้งค่า", "setup"):
        ok = It(name)
        assert await tree.interaction_check(ok) is True and ok.sent == []
    # /ตั้งค่า really hands out the link code in an unlicensed guild
    gated.s = gated.s.model_copy(update={"hub_url": "https://hub.example/"})
    manager = member(world.g, 3, manage_guild=True)
    setup = SetupInteraction(world.g, manager)
    gated.licensing.enforce = True
    await gated.handle_setup(setup)
    assert "https://hub.example/hub?link=" in setup.sent[0][0]


class SetupInteraction:
    def __init__(self, guild, user):
        self.guild, self.user, self.sent = guild, user, []
        self.response = SimpleNamespace(send_message=self._send)
        self.followup = SimpleNamespace(send=self._send)

    async def _send(self, text, ephemeral=False):
        self.sent.append((text, ephemeral))


async def _noop():
    return None


# ------------------------------------------------------------------ sharing safety (bot tools)
A, B, PUB, PRIV = 1000, 2000, 1001, 1002


@pytest.fixture
def share_world(docs, orgs):
    guilds = MemoryGuildStore()
    guilds.upsert_guild(A, name="Cafe A", icon_url="", channels=[], public_channel_ids=[str(PUB)])
    guilds.upsert_guild(B, name="Cafe B", icon_url="", channels=[], public_channel_ids=[])
    org = orgs.create("G", "a@x.com", org_id="org_s")
    orgs.attach_workspace(org.id, ws("discord", str(A), "Cafe A"))
    orgs.attach_workspace(org.id, ws("discord", str(B), "Cafe B"))
    wiki, places = WikiRepo(docs), PlaceRepo(docs)
    wiki.save(
        A, WikiPage(slug="open", title="Opening hours", body="open at 8 sharp", channel_ids=[PUB])
    )
    wiki.save(
        A, WikiPage(slug="pay", title="Payroll hours", body="secret hours", channel_ids=[PRIV])
    )
    wiki.save(A, WikiPage(slug="mix", title="Mixed hours", body="hours", channel_ids=[PUB, PRIV]))
    wiki.save(A, WikiPage(slug="dm", title="DM hours", body="hours", channel_ids=[-3]))
    wiki.save(A, WikiPage(slug="none", title="Orphan hours", body="hours", channel_ids=[]))
    at = datetime(2026, 10, 14, tzinfo=UTC)
    places.create(A, name="สาขาบางนา", address="99 Bangna", kind="branch", created_by=1,
                  sources=[make_source(PUB, 1, at)])  # fmt: skip
    places.create(A, name="โกดังลับ", address="1 Secret Rd", kind="warehouse", created_by=1,
                  sources=[make_source(PRIV, 1, at)])  # fmt: skip
    places.create(A, name="บ้านสมชาย", address="7 Home St", kind="home", owner_id=5, created_by=5,
                  sources=[make_source(PUB, 5, at)])  # fmt: skip
    places.create(A, name="ห้องประชุมเสียง", address="Voice", kind="venue", created_by=1,
                  sources=[make_source(None, 1, at, room=frozenset({1, 2}))])  # fmt: skip
    sharing = Sharing(orgs, lambda mid: {PUB} if mid == A else set(), clock=lambda: 0.0)
    return SimpleNamespace(orgs=orgs, wiki=wiki, places=places, sharing=sharing, org=org)


def link(w, cats, frm="discord-1000", to="discord-2000"):
    from salid.orgs import ShareLink

    org = w.orgs.get(w.org.id)
    org.sharing = [ShareLink(from_workspace=frm, to_workspace=to, categories=cats)]
    w.orgs.save(org)
    w.sharing.invalidate()


def wiki_tools_for(w, guild_id):
    tools = WikiTools(
        guild_id=guild_id, asker_id=1, asker_name="x", is_exec=False,
        memory=SimpleNamespace(allowed={}, vis=None), repo=w.wiki, client=None, model="m",
        lang="th", now=lambda: datetime(2026, 10, 15, tzinfo=UTC), state=None, request_id="r",
        hooks=WikiHooks(),
    )  # fmt: skip
    tools.shared = lambda: w.sharing.public_wiki_pages(guild_id, w.wiki)
    return tools


def place_tools_for(w, guild_id):
    ctx = PlacesCtx(
        guild_id=guild_id, asker_id=9, is_exec=False, repo=w.places, ranks={9: 1},
        viewers_of=lambda c: frozenset({9}), audience=frozenset({9}), channel_id=5, people={},
        channels={}, now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
        shared=lambda: w.sharing.public_places(guild_id, w.places),
    )  # fmt: skip
    return PlaceTools(ctx)


def test_page_and_place_publicness_rules():
    assert page_is_public([PUB], {PUB}) and page_is_public([PUB, 5], {PUB, 5})
    assert not page_is_public([], {PUB}) and not page_is_public([PUB, PRIV], {PUB})
    assert not page_is_public([-3], {-3})


def test_no_link_means_nothing_is_shared(share_world):
    w = share_world
    assert shared_sources(B, "wiki") == []  # no default registered
    assert w.sharing.sources(B, "wiki") == []
    assert wiki_tools_for(w, B).wiki_search("hours")["count"] == 0
    assert place_tools_for(w, B).find_place("บางนา")["status"] == "not_found"


def test_wiki_search_returns_only_public_pages_of_the_sharing_workspace(share_world):
    w = share_world
    link(w, ["wiki"])
    r = wiki_tools_for(w, B).wiki_search("hours")
    assert [p["title"] for p in r["pages"]] == ["Opening hours"]
    assert r["pages"][0]["shared_from"] == "Cafe A" and "open at 8" in r["pages"][0]["snippet"]
    for secret in ("Payroll hours", "Mixed hours", "DM hours", "Orphan hours"):
        assert secret not in str(r)
    # the reverse direction shares nothing: A only sees its own pages
    assert all("shared_from" not in p for p in wiki_tools_for(w, A).wiki_search("hours")["pages"])


def test_sharing_is_per_category_and_directional(share_world):
    w = share_world
    link(w, ["places"])  # places shared, wiki not
    assert wiki_tools_for(w, B).wiki_search("hours")["count"] == 0
    link(w, ["wiki"])
    assert place_tools_for(w, B).find_place("บางนา")["status"] == "not_found"
    link(w, ["wiki"], frm="discord-2000", to="discord-1000")  # B -> A: B has no public channels
    assert all("shared_from" not in p for p in wiki_tools_for(w, A).wiki_search("hours")["pages"])


def test_find_place_shares_only_public_company_places(share_world):
    w = share_world
    link(w, ["places"])
    t_ = place_tools_for(w, B)
    r = t_.find_place("บางนา")
    assert r["count"] == 1 and r["places"][0]["address"] == "99 Bangna"
    assert r["places"][0]["shared_from"] == "Cafe A" and "maps_link" in r["places"][0]
    for q in ("โกดังลับ", "Secret Rd", "บ้านสมชาย", "Home", "ห้องประชุมเสียง"):
        assert t_.find_place(q)["status"] == "not_found", (
            q
        )  # private / personal / voice never cross
    assert not place_is_public(w.places.search(A, "บ้านสมชาย")[0], {PUB})


def test_a_source_guild_without_public_channel_data_shares_nothing(share_world):
    w = share_world
    w.sharing._public = lambda mid: set()  # e.g. not re-synced yet
    link(w, ["wiki", "places"])
    assert wiki_tools_for(w, B).wiki_search("hours")["count"] == 0
    assert place_tools_for(w, B).find_place("บางนา")["status"] == "not_found"


def test_shared_sources_helper_and_other_orgs(share_world):
    w = share_world
    link(w, ["wiki", "documents"])
    set_sharing_default(w.sharing)
    try:
        assert shared_sources(B, "wiki") == [A] and shared_sources(B, "places") == []
        assert shared_sources(A, "wiki") == [] and shared_sources(9999, "wiki") == []
        assert shared_sources(B, "chat") == []  # there is no chat category
    finally:
        set_sharing_default(None)


# ------------------------------------------------------------------ slack signature / ids
def test_slack_signature_verification():
    secret, body, ts = "sec", b'{"a":1}', "1760000000"
    sig = "v0=" + hmac.new(b"sec", f"v0:{ts}:".encode() + body, hashlib.sha256).hexdigest()
    now = 1760000000.0
    assert verify_signature(secret, body, ts, sig, now=now)
    assert verify_signature(secret, body, ts, sig, now=now + 299)
    assert not verify_signature(secret, body, ts, sig, now=now + 301)  # replay window
    assert not verify_signature(secret, body, ts, sig, now=now - 301)
    assert not verify_signature(secret, body + b" ", ts, sig, now=now)  # tampered body
    assert not verify_signature("other", body, ts, sig, now=now)
    assert not verify_signature(secret, body, "abc", sig, now=now)
    assert not verify_signature(secret, body, ts, "", now=now)
    assert not verify_signature("", body, ts, sig, now=now)  # unset secret never verifies
    assert not verify_signature(secret, body, ts, sig.replace("v0=", "v1="), now=now)


def test_slack_message_id_includes_channel():
    from salid.orgs import NAMESPACE_BASE, message_namespaced_id

    a = message_namespaced_id("slack", "T1", "C1", "1700000000.000200")
    assert a == message_namespaced_id("slack", "T1", "C1", "1700000000.000200")
    assert a >= NAMESPACE_BASE
    # the same ts in another channel / workspace must not overwrite this record
    assert a != message_namespaced_id("slack", "T1", "C2", "1700000000.000200")
    assert a != message_namespaced_id("slack", "T2", "C1", "1700000000.000200")
    assert timedelta(0) == timedelta(0)
