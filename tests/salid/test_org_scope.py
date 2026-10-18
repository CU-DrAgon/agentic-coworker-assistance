"""Org-wide scope: the Discord permission resolver and `org_scope.resolve` (pure, no network)."""

import pytest

from salid import org_scope as osc
from salid.org_scope import (
    ADMINISTRATOR,
    CONNECT,
    READ_MESSAGE_HISTORY,
    VIEW_CHANNEL,
    channel_permissions,
    readable_channels,
)
from salid.orgs import Member, Org, Workspace, namespaced_id, workspace_memory_id

READ = VIEW_CHANNEL | READ_MESSAGE_HISTORY
GUILD = "900"  # the @everyone role has the guild's id
ME = "42"
STAFF, BOSS, MUTED = "11", "12", "13"


def guild(everyone=READ, roles=None, owner="1"):
    rows = [{"id": GUILD, "permissions": str(everyone)}]
    rows += [{"id": rid, "permissions": str(p)} for rid, p in (roles or {}).items()]
    return {"id": GUILD, "owner_id": owner, "roles": rows}


def member(*roles):
    return {"user": {"id": ME}, "roles": list(roles)}


def role_ow(rid, allow=0, deny=0):
    return {"id": rid, "type": 0, "allow": str(allow), "deny": str(deny)}


def member_ow(allow=0, deny=0):
    return {"id": ME, "type": 1, "allow": str(allow), "deny": str(deny)}


def chan(cid, *overwrites, kind=0, name=None):
    return {
        "id": str(cid),
        "type": kind,
        "name": name or f"c{cid}",
        "permission_overwrites": list(overwrites),
    }


def readable(g, m, *channels):
    return set(readable_channels(g, m, list(channels)))


# ------------------------------------------------------------------ the Discord algorithm
def test_everyone_with_view_and_history_reads_a_plain_channel():
    assert readable(guild(), member(), chan(1)) == {1}


def test_no_view_in_the_base_means_no_channel():
    assert readable(guild(everyone=READ_MESSAGE_HISTORY), member(), chan(1)) == set()
    assert readable(guild(everyone=VIEW_CHANNEL), member(), chan(1)) == set()


def test_everyone_deny_hides_a_private_channel():
    private = chan(1, role_ow(GUILD, deny=VIEW_CHANNEL))
    assert readable(guild(), member(), private) == set()


def test_a_role_overwrite_allows_what_everyone_denied():
    g = guild(roles={STAFF: 0})
    private = chan(1, role_ow(GUILD, deny=VIEW_CHANNEL), role_ow(STAFF, allow=VIEW_CHANNEL))
    assert readable(g, member(STAFF), private) == {1}
    assert readable(g, member(), private) == set()  # without the role it stays hidden


def test_overwrites_of_roles_the_member_lacks_do_not_apply():
    g = guild(roles={STAFF: 0, BOSS: 0})
    private = chan(1, role_ow(GUILD, deny=VIEW_CHANNEL), role_ow(BOSS, allow=VIEW_CHANNEL))
    assert readable(g, member(STAFF), private) == set()


def test_role_allow_wins_over_role_deny_because_denies_apply_first():
    # the member holds two roles: one denies View, one allows it. Discord removes ALL role denies
    # first and then adds ALL role allows, so the allow stands.
    g = guild(roles={STAFF: 0, BOSS: 0})
    c = chan(1, role_ow(STAFF, deny=VIEW_CHANNEL), role_ow(BOSS, allow=VIEW_CHANNEL))
    assert readable(g, member(STAFF, BOSS), c) == {1}


def test_member_allow_beats_a_role_deny():
    g = guild(roles={MUTED: 0})
    c = chan(1, role_ow(MUTED, deny=VIEW_CHANNEL), member_ow(allow=VIEW_CHANNEL))
    assert readable(g, member(MUTED), c) == {1}


def test_member_deny_beats_a_role_allow():
    g = guild(roles={STAFF: 0})
    c = chan(1, role_ow(STAFF, allow=READ), member_ow(deny=READ_MESSAGE_HISTORY))
    assert readable(g, member(STAFF), c) == set()


def test_member_overwrite_is_for_this_member_only():
    other = {"id": "7", "type": 1, "allow": str(READ), "deny": "0"}
    c = chan(1, role_ow(GUILD, deny=VIEW_CHANNEL), other)
    assert readable(guild(), member(), c) == set()


def test_overwrite_ids_are_matched_by_kind():
    # a MEMBER overwrite whose id equals a role id (and vice versa) is not that role
    g = guild(roles={STAFF: 0})
    c = chan(
        1,
        role_ow(GUILD, deny=VIEW_CHANNEL),
        {"id": STAFF, "type": 1, "allow": str(READ), "deny": "0"},
    )
    assert readable(g, member(STAFF), c) == set()


def test_history_can_be_denied_while_view_stays():
    c = chan(1, role_ow(GUILD, deny=READ_MESSAGE_HISTORY))
    assert readable(guild(), member(), c) == set()


def test_base_permissions_from_roles_are_ored():
    g = guild(everyone=0, roles={STAFF: VIEW_CHANNEL, BOSS: READ_MESSAGE_HISTORY})
    assert readable(g, member(STAFF), chan(1)) == set()
    assert readable(g, member(STAFF, BOSS), chan(1)) == {1}


def test_administrator_role_reads_everything_despite_overwrites():
    g = guild(everyone=0, roles={BOSS: ADMINISTRATOR})
    c = chan(1, role_ow(GUILD, deny=READ), member_ow(deny=READ))
    assert readable(g, member(BOSS), c) == {1}


def test_the_owner_reads_everything():
    g = guild(everyone=0, owner=ME)
    c = chan(1, role_ow(GUILD, deny=READ), member_ow(deny=READ))
    assert readable(g, member(), c) == {1}


def test_permissions_in_a_channel_follow_the_documented_order():
    g = guild(everyone=READ | CONNECT, roles={STAFF: 0})
    c = chan(
        1,
        role_ow(GUILD, deny=CONNECT),
        role_ow(STAFF, allow=CONNECT),
        member_ow(deny=VIEW_CHANNEL),
    )
    perms = channel_permissions(g, member(STAFF), c)
    assert perms & CONNECT and not perms & VIEW_CHANNEL and perms & READ_MESSAGE_HISTORY


def test_voice_text_is_readable_with_view_and_history_connect_is_not_needed():
    # the text chat of a voice / stage channel is read with View Channel + Read Message History
    voice, stage = chan(1, kind=2), chan(2, kind=13)
    assert readable(guild(everyone=READ), member(), voice, stage) == {1, 2}
    assert readable(guild(everyone=READ | CONNECT), member(), voice, stage) == {1, 2}
    assert readable(guild(everyone=VIEW_CHANNEL | CONNECT), member(), voice, stage) == set()
    denied = chan(3, role_ow(GUILD, deny=CONNECT), kind=2)
    assert readable(guild(everyone=READ | CONNECT), member(), denied) == {3}
    g = guild(everyone=READ)
    voice, stage = chan(1, kind=2), chan(2, kind=13)
    skipped = [chan(3, kind=4), chan(4, kind=11), chan(5, kind=12), chan(6, kind=10)]
    assert readable(g, member(), *skipped) == set()
    forum, announce = chan(7, kind=15), chan(8, kind=5)
    assert readable(g, member(), forum, announce) == {7, 8}


def test_names_come_back_and_permission_strings_are_tolerated():
    got = readable_channels(guild(), member(), [chan(1, name="general")])
    assert got == {1: "general"}
    g = {"id": GUILD, "owner_id": "1", "roles": [{"id": GUILD, "permissions": None}]}
    assert readable_channels(g, member(), [chan(1)]) == {}


# ------------------------------------------------------------------ resolve()
class Providers:
    """Fake per-platform providers: they record what they were asked."""

    def __init__(self, **rooms):
        self.rooms, self.calls = rooms, []

    def __getitem__(self, platform):
        return self.provider(platform)

    def get(self, platform, default=None):
        return self.provider(platform) if platform in self.rooms else default

    def provider(self, platform):
        async def run(ws, account):
            self.calls.append((platform, ws.external_id, account))
            got = self.rooms[platform]
            if isinstance(got, Exception):
                raise got
            return got

        return run


def make_org(*, role="member", identities=None, extra=None):
    ws = [
        Workspace(id="discord-900", platform="discord", external_id="900", name="Cafe"),
        Workspace(id="slack-T1", platform="slack", external_id="T1", name="Cafe Slack"),
        Workspace(id="line-Ubot", platform="line", external_id="Ubot"),
        *(extra or []),
    ]
    mew = Member(
        email="mew@x.com",
        role=role,
        discord_user_id="42",
        identities=identities if identities is not None else {"slack": "U1", "line": "Lmew"},
        identity_tenants={"line": "Ubot"},  # verified in the workspace below
    )
    return Org(
        id="o", name="Cafe", members=[mew, Member(email="boss@x.com", role="owner")], workspaces=ws
    )


def run(org, providers, *, direct=True, audience="asker", asker="asker", email="mew@x.com"):
    import asyncio

    aud = frozenset({audience}) if audience else None
    return asyncio.run(
        osc.resolve(org, email, is_direct=direct, audience=aud, asker=asker, providers=providers)
    )


FULL = dict(
    discord={"5": "general", "6": "ops"},
    slack={"C1": "general"},
    line={"Lmew": ""},
)


def test_a_verified_one_to_one_sees_every_workspace_it_may_read():
    view = run(make_org(), Providers(**FULL))
    by = {s.platform: s for s in view.sources}
    assert set(by) == {"discord", "slack", "line"}
    assert by["discord"].guild_id == 900
    assert by["discord"].channels == {5: "Discord #general", 6: "Discord #ops"}
    c1 = namespaced_id("slack", "C1", "channel")
    assert by["slack"].channels == {c1: "Slack #general"}
    assert by["slack"].guild_id == workspace_memory_id("slack", "T1")
    room = namespaced_id("line", "Lmew", "channel")
    assert by["line"].channels == {room: "LINE 1:1 chat"} and by["line"].direct == {room}
    assert view.memory_channels()[5] == (900, "Discord #general")
    assert not view.skipped


def test_a_group_or_an_unknown_audience_gets_no_widening():
    org = make_org()
    assert run(org, Providers(**FULL), direct=False) is None
    assert run(org, Providers(**FULL), audience=None) is None
    assert run(org, Providers(**FULL), audience="someone-else") is None
    assert run(org, Providers(**FULL), email="stranger@x.com") is None


def test_a_platform_without_a_verified_account_is_left_out_and_never_asked():
    org = make_org(
        identities={"slack": "U1"}
    )  # no LINE account; Discord comes from discord_user_id
    p = Providers(**FULL)
    view = run(org, p)
    assert {s.platform for s in view.sources} == {"discord", "slack"}
    assert all(c[0] != "line" for c in p.calls)
    assert ("discord", "900", "42") in p.calls and ("slack", "T1", "U1") in p.calls


def test_a_failing_provider_excludes_only_its_workspace():
    p = Providers(**{**FULL, "discord": RuntimeError("rest down"), "slack": None})
    view = run(make_org(), p)
    assert {s.platform for s in view.sources} == {"line"}
    assert set(view.skipped) == {"discord-900", "slack-T1"}


def test_a_platform_without_a_provider_is_excluded():
    p = Providers(slack=FULL["slack"], line=FULL["line"])  # no Discord token configured
    view = run(make_org(), p)
    assert {s.platform for s in view.sources} == {"slack", "line"}
    assert view.skipped == ("discord-900",)


def test_workspaces_that_are_not_connected_are_ignored():
    org = make_org()
    org.workspaces[1].status = "disconnected"
    view = run(org, Providers(**FULL))
    assert {s.platform for s in view.sources} == {"discord", "line"}
    assert all(w.platform != "slack" for w in view.workspaces)


def test_labels_name_the_workspace_when_a_platform_has_several():
    p = Providers(discord={"5": "general"}, slack={"C1": "general"}, line={"Lmew": ""})
    extra = [Workspace(id="discord-901", platform="discord", external_id="901", name="Annex")]
    view = run(make_org(extra=extra), p)
    discord = [s for s in view.sources if s.platform == "discord"]
    assert len(discord) == 1  # channel id 5 exists once; the first workspace claims it
    assert discord[0].channels[5] == "Discord Cafe #general"


def test_ledger_workspaces_carry_the_members_submitter_ids():
    view = run(make_org(), Providers(**FULL))
    ids_ = {w.platform: w.submitter_id for w in view.workspaces}
    assert ids_["discord"] == 42
    assert ids_["slack"] == namespaced_id("slack", "U1", "user")
    assert ids_["line"] == namespaced_id("line", "Lmew", "user")
    mine = view.ledger_sources(False)
    assert (900, 42) in mine and len(mine) == 3
    everything = view.ledger_sources(True)
    assert all(s is None for _, s in everything) and len(everything) == 3


def test_a_workspace_with_no_account_has_no_own_receipts_but_exec_sees_it():
    view = run(make_org(identities={"slack": "U1"}), Providers(**FULL))
    own = dict(view.ledger_sources(False))
    assert workspace_memory_id("line", "Ubot") not in own
    assert workspace_memory_id("line", "Ubot") in dict(view.ledger_sources(True))


@pytest.mark.parametrize("role,admin", [("owner", True), ("admin", True), ("member", False)])
def test_org_owner_and_admin_are_flagged(role, admin):
    assert run(make_org(role=role), Providers(**FULL)).is_admin is admin


# ------------------------------------------------------------------ provider-scoped identities
def two_workspaces(platform, key_a, key_b, identity, tenants):
    ws = [
        Workspace(id=f"{platform}-{key_a}", platform=platform, external_id=key_a),
        Workspace(id=f"{platform}-{key_b}", platform=platform, external_id=key_b),
    ]
    mew = Member(
        email="mew@x.com",
        identities={platform: identity},
        identity_tenants=tenants,
    )
    return Org(id="o", name="Cafe", members=[mew], workspaces=ws)


def run_from(org, providers, platform, tenant, asker):
    import asyncio

    return asyncio.run(
        osc.resolve(
            org,
            "mew@x.com",
            is_direct=True,
            audience=frozenset({asker}),
            asker=asker,
            providers=providers,
            asking=(platform, tenant),
        )
    )


@pytest.mark.parametrize(
    ("platform", "ident"), [("line", "Lmew"), ("whatsapp", "bsuid-1"), ("telegram", "777")]
)
def test_an_identity_linked_in_workspace_a_authorizes_only_workspace_a(platform, ident):
    org = two_workspaces(platform, "A", "B", ident, {platform: "A"})
    providers = Providers(**{platform: {ident: ""}})
    view = run_from(org, providers, platform, "A", ident)
    assert {s.guild_id for s in view.sources} == {workspace_memory_id(platform, "A")}
    by_ws = {w.guild_id: w.submitter_id for w in view.workspaces}
    assert by_ws[workspace_memory_id(platform, "A")] is not None
    assert by_ws[workspace_memory_id(platform, "B")] is None  # no ledger mapping either
    assert [c[1] for c in providers.calls] == ["A"]  # B's provider was never even asked
    assert view.ledger_sources(False) == [
        (workspace_memory_id(platform, "A"), by_ws[workspace_memory_id(platform, "A")])
    ]


def test_asking_from_workspace_b_never_adds_b_for_an_identity_linked_in_a():
    org = two_workspaces("line", "A", "B", "Lmew", {"line": "A"})
    view = run_from(org, Providers(line={"Lmew": ""}), "line", "B", "Lmew")
    assert {s.guild_id for s in view.sources} == {workspace_memory_id("line", "A")}


def test_an_identity_without_a_recorded_tenant_authorizes_only_the_asking_workspace():
    org = two_workspaces("whatsapp", "A", "B", "bsuid-1", {})
    p = Providers(whatsapp={"bsuid-1": ""})
    view = run_from(org, p, "whatsapp", "B", "bsuid-1")
    assert {s.guild_id for s in view.sources} == {workspace_memory_id("whatsapp", "B")}
    # asked from another platform: nothing to anchor it to, so nothing
    view = run_from(org, p, "slack", "T1", "U1")
    assert view.sources == ()


def test_a_slack_or_discord_identity_is_not_tenant_bound():
    org = make_org()
    view = run(org, Providers(**FULL))
    assert {s.platform for s in view.sources} == {"discord", "slack", "line"}
