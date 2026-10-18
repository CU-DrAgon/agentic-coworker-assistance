"""Org-wide scope through the assistant: a verified 1:1 reads across the org's workspaces (what the
asker may read, per platform), everyone else keeps the single-tenant scope; plus the providers."""

import itertools
from datetime import UTC, datetime

import pytest
from conftest import FixedClockSettings
from platform_fakes import NOW, Clock, FakeAsk, FakePort, inc, make_assistant
from test_slack import BOT, TEAM, FakeSlack

from salid.coo_store import SQLiteDocStore
from salid.guild_settings import GuildSettingsUpdate
from salid.identity_links import IdentityLinks
from salid.memory_store import DaySummary, MemoryMessage
from salid.org_scope import READ_MESSAGE_HISTORY, VIEW_CHANNEL
from salid.orgs import Member, OrgStore, Workspace, namespaced_id, workspace_id
from salid.platforms import ids
from salid.platforms.org_sources import (
    CachedProvider,
    DiscordRest,
    DiscordRooms,
    OrgSources,
)
from salid.platforms.slack import SlackPort
from salid.qa import LedgerTools
from salid.store import Receipt

_n = itertools.count(1)
DAY = "2026-10-14"
GUILD = 900
GENERAL, STAFF_ONLY = 5001, 5002  # Discord channels: everyone / only the Boss role
READ = VIEW_CHANNEL | READ_MESSAGE_HISTORY


@pytest.fixture
def s():
    return FixedClockSettings(exec_user_ids=set())


class StoreAsk(FakeAsk):
    """FakeAsk that also keeps the ledger store `ask` was given."""

    def __call__(self, settings, store, question, **kw):
        self.stores = getattr(self, "stores", [])
        self.stores.append(store)
        return super().__call__(settings, store, question, **kw)


class FakeDiscord:
    """A REST double for the three GETs; `fail` makes every call raise."""

    def __init__(self, roles=(), owner="1"):
        self.fail = False
        self.calls: list[str] = []
        self.member_roles = list(roles)
        self.owner = owner

    async def get(self, path):
        self.calls.append(path)
        if self.fail:
            raise RuntimeError("discord is down")
        if path.endswith("/channels"):
            return [
                {"id": str(GENERAL), "type": 0, "name": "general", "permission_overwrites": []},
                {
                    "id": str(STAFF_ONLY),
                    "type": 0,
                    "name": "boss-only",
                    "permission_overwrites": [
                        {"id": str(GUILD), "type": 0, "allow": "0", "deny": str(VIEW_CHANNEL)},
                        {"id": "77", "type": 0, "allow": str(VIEW_CHANNEL), "deny": "0"},
                    ],
                },
            ]
        if "/members/" in path:
            return {"user": {"id": path.rsplit("/", 1)[1]}, "roles": self.member_roles}
        return {
            "id": str(GUILD),
            "owner_id": self.owner,
            "roles": [
                {"id": str(GUILD), "permissions": str(READ)},
                {"id": "77", "permissions": "0"},
            ],
        }


class World:
    pass


def put(ws, store, guild, channel, mid, text, minute=0):
    store.add_message(
        MemoryMessage(
            guild_id=guild,
            channel_id=channel,
            message_id=mid,
            author_id=1,
            author_name="ann",
            created_at=datetime(2026, 10, 14, 3, minute, tzinfo=UTC),
            local_date=DAY,
            text=text,
        )
    )
    store.put_summary(
        DaySummary(
            guild_id=guild,
            channel_id=channel,
            local_date=DAY,
            message_count=3,
            final=True,
            summary={"overview": text, "topics": [], "decisions": [text]},
        )
    )


def receipt(store, guild, submitter, n, total, cur="THB"):
    store.add_receipt(
        Receipt(
            guild_id=guild,
            submitter_id=submitter,
            submitter_name=f"u{submitter}",
            sha256=f"sha{n}",
            vendor=f"v{n}",
            receipt_date="2026-10-10",
            effective_date="2026-10-10",
            total=total,
            currency=cur,
            category="food",
            confidence=0.9,
            created_at="2026-10-10",
        )
    )


@pytest.fixture
def wd(s, tmp_path):
    w = World()
    w.docs = SQLiteDocStore(":memory:")
    w.orgs = OrgStore(w.docs)
    w.orgs.create(
        "Cafe",
        "boss@example.com",
        org_id="org_a",
        members=[
            Member(email="boss@example.com", role="owner"),
            Member(email="mew@example.com", discord_user_id="42"),
            Member(email="ann@example.com"),
        ],
    )
    for platform, key in (("discord", str(GUILD)), ("slack", TEAM), ("line", "Ubot")):
        w.orgs.attach_workspace(
            "org_a",
            Workspace(id=workspace_id(platform, key), platform=platform, external_id=key),
        )
    w.orgs.link_identity("org_a", "mew@example.com", "slack", "U1")
    w.orgs.link_identity("org_a", "mew@example.com", "line", "Lmew", "Ubot")
    w.orgs.link_identity("org_a", "boss@example.com", "slack", "U2")
    w.links = IdentityLinks(w.orgs, clock=lambda: NOW)
    w.discord = FakeDiscord()
    w.clock = Clock()
    w.a = make_assistant(s, tmp_path, clock=w.clock, ask=StoreAsk())
    w.a.identity = w.links
    w.a.org_sources = OrgSources(
        w.orgs, clock=w.clock, rest=DiscordRest("token-never-used", client=None)
    )
    w.a.org_sources.providers["discord"] = CachedProvider(DiscordRooms(w.discord), w.clock)
    # a Slack team of this org with channels the member may read (stubbed provider)
    slack_rooms = {"C1": "ops"}

    async def slack(ws, account):
        return dict(slack_rooms) if account == "U1" else {"C1": "ops"}

    w.a.org_sources.providers["slack"] = CachedProvider(slack, w.clock)
    w.port = FakePort(platform="slack")
    w.slack_c1 = namespaced_id("slack", "C1", "channel")
    w.slack_gid = ids.tenant_id("slack", TEAM)
    w.line_gid = ids.tenant_id("line", "Ubot")
    put(w, w.a.memory_store, GUILD, GENERAL, 1, "discord roadmap: ship the menu")
    put(w, w.a.memory_store, GUILD, STAFF_ONLY, 2, "discord layoffs: closing the branch", 1)
    put(w, w.a.memory_store, w.slack_gid, w.slack_c1, 3, "slack ops: fridge is broken", 2)
    return w


async def ask(w, text="what happened?", *, platform="slack", tenant=TEAM, user="U1", direct=True,
              port=None, **kw):  # fmt: skip
    msg = inc(
        platform=platform,
        tenant=tenant,
        room=user if direct else "G1",
        user=user,
        text=text,
        message=f"m{next(_n)}",
        is_direct=direct,
        mentions_bot=True,
        room_name="" if direct else "team",
        **kw,
    )
    await w.a.handle(msg, port or w.port)
    await w.a.drain()
    return msg


def memory_of(w):
    return w.a.ask_fn.calls[-1][1]["memory"]


# ------------------------------------------------------------------ memory
async def test_verified_slack_dm_reads_the_discord_channel_they_may_read_not_the_other(wd):
    await ask(wd)
    mem = memory_of(wd)
    assert mem.allowed[GENERAL] == "Discord #general"
    assert STAFF_ONLY not in mem.allowed
    days = mem.chat_days(DAY, DAY)
    channels = {d["channel"] for d in days["days"]}
    assert channels == {"Discord #general", "Slack #ops"}  # the sources are named in citations
    assert "discord roadmap" in str(days) and "layoffs" not in str(days)
    assert mem.chat_search(keywords=["layoffs"])["count"] == 0
    assert mem.chat_search(keywords=["roadmap"])["count"] == 1
    assert "layoffs" not in str(mem.chat_messages(DAY, DAY))


async def test_a_member_with_the_role_reads_the_restricted_channel(wd):
    wd.discord.member_roles = ["77"]
    await ask(wd)
    mem = memory_of(wd)
    assert mem.allowed[STAFF_ONLY] == "Discord #boss-only"
    assert "layoffs" in str(mem.chat_days(DAY, DAY))


async def test_the_askers_own_tenant_scope_is_kept_and_names_are_not_replaced(wd):
    await ask(wd)
    own = ids.room_id("slack", TEAM, "U1")
    assert memory_of(wd).allowed[own] == "U1"  # today's DM label, unchanged


async def test_verified_line_one_to_one_also_reads_the_other_workspaces(wd):
    port = FakePort(platform="line")
    await ask(wd, platform="line", tenant="Ubot", user="Lmew", port=port)
    mem = memory_of(wd)
    assert mem.allowed[GENERAL] == "Discord #general" and mem.allowed[wd.slack_c1] == "Slack #ops"
    assert mem.guild_id == wd.line_gid


async def test_a_line_group_gets_only_the_group(wd):
    port = FakePort(platform="line")
    msg = await ask(wd, platform="line", tenant="Ubot", user="Lmew", direct=False, port=port)
    assert memory_of(wd).allowed == {ids.room_id("line", "Ubot", msg.room): "team"}
    assert wd.discord.calls == []  # nothing was even fetched


async def test_a_slack_channel_gets_only_its_own_scope(wd):
    msg = await ask(wd, direct=False)
    assert GENERAL not in memory_of(wd).allowed
    assert ids.room_id("slack", TEAM, msg.room) in memory_of(wd).allowed


async def test_an_unverified_dm_asker_gets_only_that_dm(wd):
    msg = await ask(wd, user="U-guest")
    assert memory_of(wd).allowed == {ids.room_id("slack", TEAM, msg.room): "U-guest"}
    assert wd.discord.calls == []


async def test_a_member_of_another_org_is_not_widened_by_this_orgs_data(wd):
    wd.orgs.create("Other", "o@example.com", org_id="org_b")
    wd.orgs.attach_workspace("org_b", Workspace(id="slack-T9", platform="slack", external_id="T9"))
    wd.orgs.link_identity("org_b", "o@example.com", "slack", "U1")  # same slack id, other org
    await ask(wd, tenant="T9", user="U1")
    assert GENERAL not in memory_of(wd).allowed and wd.discord.calls == []  # org_a's, not org_b's
    assert [g for g, _ in wd.a.ask_fn.stores[-1].sources] == [ids.tenant_id("slack", "T9")]


async def test_without_a_discord_token_discord_is_left_out(wd):
    wd.a.org_sources.providers.pop("discord")
    await ask(wd)
    mem = memory_of(wd)
    assert GENERAL not in mem.allowed and mem.allowed[wd.slack_c1] == "Slack #ops"


async def test_a_discord_rest_error_leaves_discord_out_and_fails_closed(wd):
    wd.discord.fail = True
    await ask(wd)
    mem = memory_of(wd)
    assert GENERAL not in mem.allowed and STAFF_ONLY not in mem.allowed
    assert mem.allowed[wd.slack_c1] == "Slack #ops"


async def test_discord_permissions_are_cached_for_five_minutes(wd):
    await ask(wd)
    first = len(wd.discord.calls)
    await ask(wd)
    assert len(wd.discord.calls) == first  # served from the cache
    wd.clock.advance(301)
    await ask(wd)
    assert len(wd.discord.calls) == 2 * first


async def test_a_failure_is_cached_only_briefly(wd):
    wd.discord.fail = True
    await ask(wd)
    wd.discord.fail = False
    wd.clock.advance(20)
    await ask(wd)
    assert memory_of(wd).allowed[GENERAL] == "Discord #general"


async def test_channels_in_mode_off_and_workspaces_without_memory_add_nothing(wd):
    wd.a.team.store.save_settings(
        GUILD, GuildSettingsUpdate(channel_modes={str(GENERAL): "off"}), "t"
    )
    wd.a.team.store.save_settings(wd.slack_gid, GuildSettingsUpdate(features={"memory": True}), "t")
    await ask(wd)
    assert GENERAL not in memory_of(wd).allowed
    wd.a.team.store.save_settings(GUILD, GuildSettingsUpdate(features={"memory": False}), "t")
    await ask(wd)
    assert not any(c in memory_of(wd).allowed for c in (GENERAL, STAFF_ONLY))


async def test_an_error_while_widening_keeps_the_single_tenant_scope(wd, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(wd.a, "_widen_memory", boom)
    msg = await ask(wd)
    assert memory_of(wd).allowed == {ids.room_id("slack", TEAM, msg.room): "U1"}


async def test_retention_is_the_shortest_of_the_workspaces_involved(wd):
    wd.a.team.store.save_settings(GUILD, GuildSettingsUpdate(memory_retention_days=7), "t")
    await ask(wd)
    assert memory_of(wd).retention_days == 7


# ------------------------------------------------------------------ ledger
@pytest.fixture
def money(wd):
    st = wd.a.store
    receipt(st, GUILD, 42, 1, 100.0)  # mew in Discord
    receipt(st, GUILD, 99, 2, 1000.0)  # someone else in Discord
    receipt(st, wd.slack_gid, ids.user_id("slack", "U1"), 3, 50.0, "USD")  # mew in Slack
    receipt(st, wd.slack_gid, ids.user_id("slack", "U2"), 4, 7.0)  # the boss in Slack
    receipt(st, wd.line_gid, ids.user_id("line", "Lmew"), 5, 25.0)  # mew in LINE
    receipt(st, 4242, 42, 6, 5555.0)  # a workspace outside the org
    return wd


def totals(w, who="mew"):
    store = w.a.ask_fn.stores[-1]
    forced = None if who == "exec" else 1
    out = LedgerTools(store, w.slack_gid, forced).spending_summary("2026-10-01", "2026-10-31")
    return out["totals_by_currency"]


async def test_a_member_sees_their_own_receipts_across_their_linked_accounts(money):
    await ask(money)
    assert totals(money) == {"THB": 125.0, "USD": 50.0}  # per currency, never added


async def test_an_org_owner_sees_every_receipt_of_every_org_workspace(money):
    await ask(money, user="U2")  # the boss, an org owner (not a Slack admin)
    assert totals(money, "exec") == {"THB": 1132.0, "USD": 50.0}


async def test_the_platform_executive_rule_also_gives_the_org_wide_ledger(money):
    money.a.exec_ids["slack"] = {"U1"}
    await ask(money)
    assert totals(money, "exec") == {"THB": 1132.0, "USD": 50.0}


async def test_a_group_asker_keeps_the_single_tenant_ledger(money):
    await ask(money, direct=False)
    assert money.a.ask_fn.stores[-1] is money.a.store


async def test_an_unverified_dm_keeps_the_single_tenant_ledger(money):
    await ask(money, user="U-guest")
    assert money.a.ask_fn.stores[-1] is money.a.store


async def test_without_org_sources_nothing_changes(money):
    money.a.org_sources = None
    await ask(money)
    assert money.a.ask_fn.stores[-1] is money.a.store
    assert memory_of(money).guild_id == money.slack_gid


# ------------------------------------------------------------------ the Discord provider
async def test_discord_provider_rejects_a_member_answer_for_another_user(wd):
    class Odd(FakeDiscord):
        async def get(self, path):
            got = await super().get(path)
            if "/members/" in path:
                got["user"] = {"id": "1"}
            return got

    ws = Workspace(id="discord-900", platform="discord", external_id=str(GUILD))
    assert await DiscordRooms(Odd()).__call__(ws, "42") is None


async def test_discord_provider_owner_and_admin(wd):
    ws = Workspace(id="discord-900", platform="discord", external_id=str(GUILD))
    rooms = await DiscordRooms(FakeDiscord(owner="42"))(ws, "42")
    assert set(rooms) == {str(GENERAL), str(STAFF_ONLY)}


# ------------------------------------------------------------------ the Slack provider
class FlagSlack(FakeSlack):
    """users.conversations rows with the share flags Slack sends."""

    def __init__(self):
        super().__init__()
        self.users["UG"] = {"id": "UG", "team_id": TEAM, "name": "guest", "is_restricted": True,
                            "profile": {}}  # fmt: skip
        self.channels = {
            "CP": {"id": "CP", "name": "all-hands", "is_private": False, "members": [BOT]},
            "CS": {"id": "CS", "name": "secret", "is_private": True, "members": ["U1", BOT]},
            "CO": {"id": "CO", "name": "others", "is_private": True, "members": ["U3", BOT]},
            "CX": {
                "id": "CX",
                "name": "partner",
                "is_private": False,
                "members": [BOT],
                "is_ext_shared": True,
            },  # Slack Connect  # fmt: skip
            "CY": {
                "id": "CY",
                "name": "shared-priv",
                "is_private": True,
                "members": ["U1", BOT],
                "is_shared": True,
            },  # fmt: skip
            "CM": {
                "id": "CM",
                "name": "mpdm-u1--u3",
                "is_mpim": True,
                "is_private": True,
                "members": ["U1", "U3", BOT],
            },  # fmt: skip
            "CN": {
                "id": "CN",
                "name": "noflags",
                "is_private": False,
                "members": [BOT],
                "omit_flags": True,
            },  # fmt: skip
            "CG": {"id": "CG", "name": "guest-ch", "is_private": True, "members": ["UG", BOT]},
        }

    async def users_conversations(self, user=None, types=None, exclude_archived=None,
                                  limit=200, cursor=None):  # fmt: skip
        who = user or BOT
        rows = []
        for c in self.channels.values():
            if who not in c["members"]:
                continue
            row = {"id": c["id"], "name": c["name"], "is_private": c.get("is_private", False),
                   "is_mpim": c.get("is_mpim", False)}  # fmt: skip
            if not c.get("omit_flags"):
                for k in ("is_shared", "is_ext_shared", "is_org_shared"):
                    row[k] = bool(c.get(k, False))
            rows.append(row)
        return {"channels": rows, "response_metadata": {"next_cursor": ""}}


@pytest.fixture
def sp():
    return SlackPort(FlagSlack(), TEAM, BOT, clock=Clock())


async def test_slack_readable_rooms_public_private_membership_and_never_shared(sp):
    got = await sp.readable_rooms("U1")
    assert got == {"CP": "all-hands", "CS": "secret", "CM": "mpdm-u1--u3"}
    # not: CO (private, not a member), CX / CY (Slack Connect), CN (no share flags: unknown)
    assert await sp.readable_rooms("U3") == {"CP": "all-hands", "CO": "others", "CM": "mpdm-u1--u3"}


async def test_a_slack_guest_reads_no_public_channel_only_their_own_private_ones(sp):
    assert await sp.readable_rooms("UG") == {"CG": "guest-ch"}


@pytest.mark.parametrize("who", ["UB", "UD", "UX", "UN", "U-missing"])
async def test_bots_deleted_foreign_and_unknown_accounts_get_nothing(sp, who):
    assert await sp.readable_rooms(who) is None


async def test_a_slack_error_means_nothing_is_known(sp):
    async def boom(**kw):
        raise RuntimeError("ratelimited")

    sp.client.users_conversations = boom
    assert await sp.readable_rooms("U1") is None


async def test_an_org_workspace_without_a_port_in_this_process_is_excluded(wd):
    wd.a.org_sources.providers.pop("slack")
    from salid.platforms.org_sources import SlackRooms

    rooms = SlackRooms()
    ws = Workspace(id="slack-T9", platform="slack", external_id="T9")
    assert await rooms(ws, "U1") is None
    rooms.add(SlackPort(FlagSlack(), TEAM, BOT, clock=Clock()))
    assert await rooms(Workspace(id=f"slack-{TEAM}", platform="slack", external_id=TEAM), "U1")


# ------------------------------------------------------------------ provider-scoped identities
@pytest.mark.parametrize("platform", ["line", "whatsapp"])
async def test_an_identity_verified_in_workspace_a_is_unknown_in_workspace_b(wd, platform):
    for key in ("A", "B"):
        wd.orgs.attach_workspace(
            "org_a",
            Workspace(id=workspace_id(platform, key), platform=platform, external_id=key),
        )
    wd.orgs.link_identity("org_a", "ann@example.com", platform, "Xann", "A")
    gid_a, gid_b = ids.tenant_id(platform, "A"), ids.tenant_id(platform, "B")

    port = FakePort(platform=platform)
    await ask(wd, platform=platform, tenant="A", user="Xann", port=port)
    assert wd.a.ask_fn.stores[-1] is not wd.a.store  # A: verified, the org-wide view
    assert memory_of(wd).guild_id == gid_a

    port = FakePort(platform=platform)
    await ask(wd, platform=platform, tenant="B", user="Xann", port=port)
    assert wd.a.ask_fn.stores[-1] is wd.a.store  # B: not verified there, today's scope only
    assert memory_of(wd).guild_id == gid_b
    assert wd.a.identity_of(inc(platform=platform, tenant="B", user="Xann")) is None
