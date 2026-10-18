# ruff: noqa: F811, F401
"""Positions in the bot (`salid.positions_bot`): Discord roles -> positions, administrators and
titles into the central org, stated positions -> one Discord role each (created once, assigned,
the old one removed), missing permissions, roles Salid did not create, the rank and executive
hooks, the tools in chat. Offline: fake Discord objects, SQLite, no network."""

import asyncio
from types import SimpleNamespace

import pytest
from conftest import GUILD, FakeClient, text_resp
from test_coordinator import ADMIN_ID, BOSS, INTERN, NS, OWNER_ID, PEER, STAFF, person, world
from test_modes import TMsg, bot  # noqa: F401  (fixture)

from salid import positions as pos
from salid.guild_settings import GuildSettingsUpdate
from salid.i18n import t
from salid.orgs import Member, OrgStore, Workspace
from salid.people import set_role as _set_role
from salid.positions_bot import RANK_UNIT

ME_ID = 999


# ------------------------------------------------------------------ fake Discord
class FRole(NS):
    """A role: id, name, position, permissions value (0 = a plain tag)."""

    def __init__(self, id, name, position=1, perms=0):
        super().__init__(
            id=id, name=name, position=position, permissions=SimpleNamespace(value=perms),
            managed=False,
        )  # fmt: skip

    def is_assignable(self):
        return True

    async def edit(self, **kw):
        if "position" in kw:
            self.position = kw["position"]


def wire(world, *, manage_roles=True):
    """Roles on the guild, add_roles / remove_roles on every member, Salid's own member."""
    g = world.g
    g.default_role = FRole(1, "@everyone", 0)
    g.roles = [g.default_role]
    g.created = []
    top = FRole(500, "Salid", 50)
    g.roles.append(top)
    g.me = SimpleNamespace(
        id=ME_ID, display_name="Salid", name="Salid", top_role=top,
        guild_permissions=SimpleNamespace(manage_roles=manage_roles),
    )  # fmt: skip
    g.get_role = lambda rid: next((r for r in g.roles if r.id == rid), None)

    async def create_role(*, name, permissions, hoist=False, mentionable=False, reason=None):
        role = FRole(1000 + len(g.created), name, 1, perms=permissions.value)
        role.hoist, role.mentionable, role.guild = hoist, mentionable, g
        g.created.append(role)
        g.roles.append(role)
        return role

    g.create_role = create_role
    for m in g.members.values():
        wire_member(m, g)
    g.owner_id = OWNER_ID


def wire_member(m, g):
    m.roles = [r for r in getattr(m, "roles", []) if hasattr(r, "id")] or [g.default_role]
    m.dms = getattr(m, "dms", [])

    async def add_roles(*roles, reason=None):
        for r in roles:
            if r not in m.roles:
                m.roles.append(r)

    async def remove_roles(*roles, reason=None):
        for r in roles:
            if r in m.roles:
                m.roles.remove(r)

    m.add_roles, m.remove_roles = add_roles, remove_roles


def give(world, uid, *roles):
    m = world.g.members[uid]
    for r in roles:
        if isinstance(r, str):
            r = FRole(2000 + len(world.g.roles), r, 10)
            world.g.roles.append(r)
        if r not in m.roles:
            m.roles.append(r)
    return m


def names(m):
    return sorted(r.name for r in m.roles if r.name != "@everyone")


def link(bot, *pairs, owner_role="owner"):
    """An org with the Discord workspace and members linked to the given Discord ids."""
    store = OrgStore(bot.docs)
    members = [Member(email="boss@x.com", role=owner_role, discord_user_id=str(OWNER_ID))]
    for email, uid in pairs:
        members.append(Member(email=email, role="member", discord_user_id=str(uid)))
    org = store.create("Cafe", "boss@x.com", org_id="o1", members=members)
    store.attach_workspace(
        org.id, Workspace(id=f"discord-{GUILD}", platform="discord", external_id=str(GUILD))
    )
    return store


def org_member(bot, email):
    org = OrgStore(bot.docs).get("o1")
    return next(m for m in org.members if m.email == email)


@pytest.fixture
def rig(bot, world):
    wire(world)
    give(world, OWNER_ID, "Founder")  # the owner already wears the Founder tag (see the owner test)
    return SimpleNamespace(bot=bot, w=world, g=world.g)


# ------------------------------------------------------------------ Discord -> center
async def test_a_discord_role_named_like_a_title_is_the_members_position(rig):
    give(rig.w, BOSS, "ซีทีโอ", "Gardeners")  # a Thai alias, plus a role that is no position
    give(rig.w, PEER, "manager")  # case-insensitive
    await rig.bot.positions_sync(rig.g)
    assert rig.bot.positions.of(GUILD, BOSS)[1].key == "cto"
    assert rig.bot.positions.of(GUILD, PEER)[1].key == "manager"
    assert rig.bot.positions.of(GUILD, STAFF) is None
    assert rig.bot.positions.of(GUILD, BOSS)[0].source == "discord"


async def test_the_highest_matching_role_wins(rig):
    give(rig.w, BOSS, "Intern", "CEO", "Manager")
    await rig.bot.positions_sync(rig.g)
    assert rig.bot.positions.of(GUILD, BOSS)[1].key == "ceo"


async def test_a_discord_role_never_overrides_a_stated_position(rig):
    rig.bot.positions.assign(GUILD, BOSS, "cfo", "stated")
    give(rig.w, BOSS, "CEO")
    await rig.bot.positions_sync(rig.g)
    assert rig.bot.positions.of(GUILD, BOSS)[1].key == "cfo"


async def test_the_guild_owner_is_a_founder_when_nothing_else_is_known(rig):
    owner = rig.g.members[OWNER_ID]
    owner.roles = [r for r in owner.roles if r.name != "Founder"]
    await rig.bot.positions_sync(rig.g)
    a, p = rig.bot.positions.of(GUILD, OWNER_ID)
    assert (p.key, a.source) == ("founder", "discord")
    await rig.bot.positions_sync(rig.g)  # and it stays on the next day
    assert rig.bot.positions.of(GUILD, OWNER_ID)[1].key == "founder"
    worn = [r.name for r in rig.g.members[OWNER_ID].roles]
    tags = [n for n in worn if n in ("Founder", "ผู้ก่อตั้ง")]
    assert len(tags) == 1, "the owner wears the Founder tag without anybody asking"
    await rig.bot.positions_sync(rig.g)
    assert len(rig.g.created) <= 1, "and it is made once, never again"


async def test_a_title_salid_already_knows_from_the_dossier_is_tagged_without_a_command(rig):
    rig.bot.people.update(GUILD, PEER, "Peer", lambda d: _set_role(d, "CTO", "heard"))
    await rig.bot.positions_sync(rig.g)
    assert rig.bot.positions.of(GUILD, PEER)[1].key == "cto"
    cto = [r for r in rig.g.created if r.name == "CTO"]
    assert cto and cto[0] in rig.g.members[PEER].roles


async def test_a_position_from_a_worn_discord_role_is_not_tagged_twice(rig):
    give(rig.w, BOSS, "CEO")
    await rig.bot.positions_sync(rig.g)
    assert not [r for r in rig.g.created if r.name == "CEO"]


async def test_a_position_that_came_from_a_role_goes_when_the_role_goes(rig):
    m = give(rig.w, PEER, "Director")
    await rig.bot.positions_sync(rig.g)
    assert rig.bot.positions.of(GUILD, PEER) is not None
    m.roles = [rig.g.default_role]
    await rig.bot.positions_sync(rig.g)
    assert rig.bot.positions.of(GUILD, PEER) is None
    rig.bot.positions.assign(GUILD, STAFF, "cto", "stated")
    await rig.bot.positions_sync(rig.g)
    assert rig.bot.positions.of(GUILD, STAFF) is not None  # stated stays without any role


async def test_administrator_and_manage_server_make_the_linked_org_member_admin(rig):
    link(rig.bot, ("admin@x.com", ADMIN_ID), ("boss@x.com2", BOSS), ("tom@x.com", STAFF))
    rig.g.members[BOSS].guild_permissions.manage_guild = True
    await rig.bot.positions_sync(rig.g)
    assert org_member(rig.bot, "admin@x.com").role == "admin"
    assert org_member(rig.bot, "boss@x.com2").role == "admin"
    assert org_member(rig.bot, "tom@x.com").role == "member"  # no permission, no promotion
    assert org_member(rig.bot, "boss@x.com").role == "owner"  # the owner stays the owner


async def test_nobody_is_lowered_and_unlinked_admins_change_nothing(rig):
    store = link(rig.bot, ("tom@x.com", STAFF))
    store.update_members("o1", lambda ms: [setattr(m, "role", "admin") for m in ms
                                           if m.email == "tom@x.com"] and True)  # fmt: skip
    await rig.bot.positions_sync(rig.g)  # Tom has no admin permission in Discord
    assert org_member(rig.bot, "tom@x.com").role == "admin"


# ------------------------------------------------------------------ center
async def test_titles_levels_and_departments_are_mirrored_to_linked_members(rig):
    link(rig.bot, ("ceo@x.com", BOSS), ("tom@x.com", STAFF))
    rig.bot.positions.assign(GUILD, BOSS, "ceo", "stated", "business")
    rig.bot.positions.assign(GUILD, STAFF, "staff", "stated", "Customer Care")
    rep = await rig.bot.positions_sync(rig.g)
    ceo = org_member(rig.bot, "ceo@x.com")
    assert (ceo.title, ceo.level, ceo.department) == ("CEO", 1, "business")
    tom = org_member(rig.bot, "tom@x.com")
    assert (tom.title, tom.level, tom.department) == ("Staff", 6, "")  # free text is not a code
    assert rep.mirrored >= 2


async def test_members_not_linked_to_the_center_are_listed(rig):
    link(rig.bot, ("tom@x.com", STAFF))
    rep = await rig.bot.positions_sync(rig.g)
    listed = rep.as_dict()
    assert "Somchai" not in listed["not_linked_to_center"]  # STAFF is linked
    assert "Ploy" in listed["not_linked_to_center"] and listed["not_linked_to_center_count"] >= 4


async def test_without_an_org_nothing_is_mirrored_and_nobody_is_called_unlinked(rig):
    rig.bot.positions.assign(GUILD, BOSS, "ceo", "stated")
    rep = await rig.bot.positions_sync(rig.g)
    assert rep.mirrored == 0 and rep.as_dict()["not_linked_to_center_count"] == 0


async def test_a_title_edited_in_the_center_becomes_the_position_and_is_not_reverted(rig):
    store = link(rig.bot, ("tom@x.com", STAFF))
    rig.bot.positions.assign(GUILD, STAFF, "staff", "stated")
    await rig.bot.positions_sync(rig.g)  # mirrored: "Staff"
    store.update_members("o1", lambda ms: setattr(ms[1], "title", "CFO") or True)
    await rig.bot.positions_sync(rig.g)
    a, p = rig.bot.positions.of(GUILD, STAFF)
    assert (p.key, a.source) == ("cfo", "org")
    assert org_member(rig.bot, "tom@x.com").title == "CFO"
    assert [r.name for r in rig.g.members[STAFF].roles if r.name == "CFO"] == ["CFO"]  # tagged
    await rig.bot.positions_sync(rig.g)
    assert rig.bot.positions.of(GUILD, STAFF)[1].key == "cfo"


async def test_a_new_center_title_creates_a_catalog_entry(rig):
    store = link(rig.bot, ("tom@x.com", STAFF))
    store.update_members("o1", lambda ms: setattr(ms[1], "title", "Head of Growth") or True)
    await rig.bot.positions_sync(rig.g)
    p = rig.bot.positions.of(GUILD, STAFF)[1]
    assert p.title == "Head of Growth" and p.level == 3


# ------------------------------------------------------------------ center -> Discord
async def test_a_stated_position_gets_one_role_created_once_and_assigned(rig):
    rig.bot.positions.assign(GUILD, BOSS, "cto", "stated")
    rig.bot.positions.assign(GUILD, PEER, "cto", "stated")
    rep = await rig.bot.positions_sync(rig.g)
    assert [r.name for r in rig.g.created] == ["CTO"], "one role for two people"
    role = rig.g.created[0]
    assert role.permissions.value == 0 and role.mentionable and role.hoist  # level 2: hoisted
    assert role.position == 49  # just under Salid's own top role
    assert role in rig.g.members[BOSS].roles and role in rig.g.members[PEER].roles
    assert rep.tagged == 2 and rep.created == 1
    assert rig.bot.positions.get(GUILD, "cto").discord_role_id == role.id
    again = await rig.bot.positions_sync(rig.g)
    assert rig.g.created == [role] and again.tagged == 0  # nothing new


async def test_levels_below_three_are_not_hoisted(rig):
    rig.bot.positions.assign(GUILD, STAFF, "staff", "stated")
    await rig.bot.positions_sync(rig.g)
    assert rig.g.created[0].hoist is False


async def test_a_new_position_replaces_the_previous_one_of_the_member(rig):
    rig.bot.positions.assign(GUILD, BOSS, "cto", "stated")
    await rig.bot.positions_sync(rig.g)
    rig.bot.positions.assign(GUILD, BOSS, "coo", "stated")
    rep = await rig.bot.positions_sync(rig.g)
    assert "COO" in names(rig.g.members[BOSS]) and "CTO" not in names(rig.g.members[BOSS])
    assert rep.untagged == 1 and rep.tagged == 1


async def test_roles_salid_did_not_create_are_never_taken_away(rig):
    mine = give(rig.w, BOSS, "Gardeners", "Executive")
    rig.bot.positions.assign(GUILD, BOSS, "cto", "stated")
    await rig.bot.positions_sync(rig.g)
    rig.bot.positions.assign(GUILD, BOSS, "coo", "stated")
    await rig.bot.positions_sync(rig.g)
    assert {"Gardeners", "Executive", "COO"} <= set(names(mine)) and "CTO" not in names(mine)
    assert all(r.name in {"CTO", "COO"} for r in rig.g.created)  # nothing else was made


async def test_a_same_name_role_with_permissions_is_not_a_tag_and_not_handed_out(rig):
    boss_role = FRole(77, "CTO", 20, perms=8)  # an admin-like role that happens to be called CTO
    rig.g.roles.append(boss_role)
    rig.bot.positions.assign(GUILD, PEER, "cto", "stated")
    rep = await rig.bot.positions_sync(rig.g)
    assert boss_role not in rig.g.members[PEER].roles and rig.g.created == []
    assert rep.skipped_other == 1


async def test_a_same_name_role_without_permissions_is_used_instead_of_a_twin(rig):
    plain = FRole(78, "CTO", 20, perms=0)
    rig.g.roles.append(plain)
    rig.bot.positions.assign(GUILD, PEER, "cto", "stated")
    await rig.bot.positions_sync(rig.g)
    assert rig.g.created == [] and plain in rig.g.members[PEER].roles
    assert rig.bot.positions.get(GUILD, "cto").discord_role_id == 78


async def test_a_deleted_role_is_made_again_for_the_next_tag(rig):
    rig.bot.positions.assign(GUILD, BOSS, "cto", "stated")
    await rig.bot.positions_sync(rig.g)
    role = rig.g.created[0]
    rig.g.roles.remove(role)
    await rig.bot._positions_on_role_delete(role)
    assert rig.bot.positions.get(GUILD, "cto").discord_role_id == 0
    rig.g.members[BOSS].roles.remove(role)
    await rig.bot.positions_sync(rig.g)
    assert len(rig.g.created) == 2 and rig.g.created[1] in rig.g.members[BOSS].roles


async def test_renaming_a_position_role_in_discord_renames_the_position(rig):
    rig.bot.positions.assign(GUILD, BOSS, "cto", "stated")
    await rig.bot.positions_sync(rig.g)
    role = rig.g.created[0]
    before = SimpleNamespace(name="CTO", permissions=role.permissions, guild=rig.g, id=role.id)
    role.name = "Tech chief"
    after = SimpleNamespace(
        name="Tech chief", permissions=role.permissions, guild=rig.g, id=role.id
    )
    await rig.bot._positions_on_role_update(before, after)
    assert rig.bot.positions.get(GUILD, "cto").title == "Tech chief"


async def test_missing_manage_roles_is_a_perm_line_once_a_day_not_an_exception(rig):
    wire(rig.w, manage_roles=False)
    sent = []

    async def deliver(guild, m, text, files=None, view=None):
        sent.append((m.id, text))
        return SimpleNamespace(ok=True)

    rig.bot.deliver_to_person = deliver
    rig.bot.positions.assign(GUILD, BOSS, "cto", "stated")
    rep = await rig.bot.positions_sync(rig.g)
    assert rep.skipped_no_permission == 1 and rig.g.created == []
    assert sent and "Manage Roles" in sent[0][1]
    n = len(sent)
    await rig.bot.positions_sync(rig.g)
    assert len(sent) == n, "once a day"
    rig.bot.clock.advance(0)
    rig.bot._positions_warned.clear()
    await rig.bot.positions_sync(rig.g)
    assert len(sent) > n  # the next day it may say it again


async def test_discord_refusing_the_role_is_counted_not_raised(rig):
    async def boom(**kw):
        raise __import__("discord").Forbidden(SimpleNamespace(status=403, reason="x"), "no")

    rig.g.create_role = boom
    rig.bot.positions.assign(GUILD, BOSS, "cto", "stated")
    rep = await rig.bot.positions_sync(rig.g)
    assert rep.skipped_no_permission == 1


async def test_first_start_creates_nothing_unless_a_position_was_stated(rig):
    give(rig.w, BOSS, "CEO")  # a matching role already exists
    link(rig.bot, ("ceo@x.com", BOSS))
    rep = await rig.bot.positions_sync(rig.g)
    assert rig.g.created == [] and rep.tagged == 0 and rep.untagged == 0
    assert rig.bot.positions.of(GUILD, BOSS)[0].source == "discord"
    assert org_member(rig.bot, "ceo@x.com").title == "CEO"  # but the center learns it


async def test_the_department_role_is_off_by_default_and_on_when_asked(rig):
    rig.bot.positions.assign(GUILD, BOSS, "cto", "stated", "tech")
    await rig.bot.positions_sync(rig.g)
    assert [r.name for r in rig.g.created] == ["CTO"]
    rig.bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(department_roles=True), "x")
    rig.bot.team._cache.clear()
    await rig.bot.positions_sync(rig.g)
    assert "Tech" in [r.name for r in rig.g.created]
    assert "Tech" in names(rig.g.members[BOSS])
    n = len(rig.g.created)
    await rig.bot.positions_sync(rig.g)
    assert len(rig.g.created) == n


async def test_the_flag_off_does_nothing_at_all(rig):
    rig.bot.guilds_store.save_settings(
        GUILD, GuildSettingsUpdate(features={"role_sync": False}), "x"
    )
    rig.bot.team._cache.clear()
    give(rig.w, BOSS, "CEO")
    await rig.bot._positions_due()
    assert rig.bot.positions.assignments(GUILD) == {}
    before = rank(rig.bot, rig.g, BOSS)
    rig.bot.positions.assign(GUILD, BOSS, "ceo", "stated")  # stored, but the feature is off
    rig.bot._positions_forget(GUILD)
    rig.bot._ranks_cache.clear()
    assert rank(rig.bot, rig.g, BOSS) == before
    assert not rig.bot.is_exec_member(rig.g.members[STAFF])
    rig.bot.positions.assign(GUILD, STAFF, "ceo", "stated")
    rig.bot._positions_forget(GUILD)
    assert not rig.bot.is_exec_member(rig.g.members[STAFF])


async def test_the_daily_job_runs_once_per_day(rig):
    await rig.bot._positions_due()
    assert rig.bot.positions.of(GUILD, OWNER_ID) is not None
    rig.bot.positions.unassign(GUILD, OWNER_ID)
    await rig.bot._positions_due()
    assert rig.bot.positions.of(GUILD, OWNER_ID) is None  # same day: not run again


async def test_a_role_change_of_one_member_updates_just_that_member(rig):
    before = SimpleNamespace(roles=list(rig.g.members[PEER].roles), guild=rig.g)
    m = give(rig.w, PEER, "Senior")
    after = SimpleNamespace(**{**vars(m)})
    after.guild = rig.g
    await rig.bot._positions_on_member_update(before, after)
    assert rig.bot.positions.of(GUILD, PEER)[1].key == "senior"
    assert rig.bot.positions.of(GUILD, BOSS) is None  # nobody else was looked at


# ------------------------------------------------------------------ hierarchy
def rank(bot, g, uid):
    return bot._rank_of(g, g.members[uid])


async def test_ranks_are_unchanged_while_nobody_has_a_position(rig):
    before = {u: rank(rig.bot, rig.g, u) for u in rig.g.members}
    rig.bot._positions_forget(GUILD)
    rig.bot._ranks_cache.clear()
    assert {u: rank(rig.bot, rig.g, u) for u in rig.g.members} == before
    assert before[OWNER_ID] == 2_000_000 and before[ADMIN_ID] >= 1_000_000


async def test_a_ceo_outranks_an_administrator_staffer_and_the_owner_stays_on_top(rig):
    base_admin = rank(rig.bot, rig.g, ADMIN_ID)
    rig.bot.positions.assign(GUILD, ADMIN_ID, "staff", "stated")
    rig.bot.positions.assign(GUILD, BOSS, "ceo", "stated")
    rig.bot.positions.assign(GUILD, PEER, "manager", "stated")
    rig.bot._positions_forget(GUILD)
    ceo, admin, mgr = (rank(rig.bot, rig.g, u) for u in (BOSS, ADMIN_ID, PEER))
    assert ceo > mgr > admin > rank(rig.bot, rig.g, STAFF)  # position first, then the old formula
    assert admin == base_admin + 2 * RANK_UNIT  # staff = level 6 -> two steps
    assert rank(rig.bot, rig.g, OWNER_ID) > ceo, "the owner is always on top"
    assert rank(rig.bot, rig.g, STAFF) == 2  # no position: as before


async def test_hierarchy_reading_follows_the_new_ranks(rig):
    rig.bot.positions.assign(GUILD, BOSS, "ceo", "stated")
    rig.bot.positions.assign(GUILD, ADMIN_ID, "staff", "stated")
    rig.bot._positions_forget(GUILD)
    ranks = rig.bot._ranks(rig.g)
    from salid.memory_access import hierarchy_authors

    assert ADMIN_ID in hierarchy_authors(BOSS, ranks)
    assert BOSS not in hierarchy_authors(ADMIN_ID, ranks)


async def test_executive_by_position_level(rig):
    m = rig.g.members[STAFF]
    assert not rig.bot.is_exec_member(m)
    rig.bot.positions.assign(GUILD, STAFF, "director", "stated")  # level 3
    rig.bot._positions_forget(GUILD)
    assert rig.bot.is_exec_member(m)
    rig.bot.positions.assign(GUILD, PEER, "manager", "stated")  # level 4: not at the default
    rig.bot._positions_forget(GUILD)
    assert not rig.bot.is_exec_member(rig.g.members[PEER])
    rig.bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(exec_max_level=4), "x")
    rig.bot.team._cache.clear()
    assert rig.bot.is_exec_member(rig.g.members[PEER])
    rig.bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(exec_max_level=2), "x")
    rig.bot.team._cache.clear()
    assert not rig.bot.is_exec_member(m)  # level 3 is above the new limit
    rig.bot.positions.assign(GUILD, INTERN, "ceo", "stated")
    rig.bot._positions_forget(GUILD)
    assert rig.bot.is_exec_member(rig.g.members[INTERN])


async def test_an_unknown_level_position_is_never_an_executive(rig):
    rig.bot.positions.add(GUILD, "Accountant")
    rig.bot.positions.assign(GUILD, STAFF, "accountant", "stated")
    rig.bot._positions_forget(GUILD)
    assert not rig.bot.is_exec_member(rig.g.members[STAFF])


# ------------------------------------------------------------------ dossiers and roster
async def test_the_dossier_and_the_roster_take_the_position_first(rig):
    give(rig.w, STAFF, "Senior")  # the top Discord role would say something else
    rig.g.members[STAFF].top_role = SimpleNamespace(name="Gardeners", position=2)
    rig.bot.positions.assign(GUILD, STAFF, "cto", "stated")
    await rig.bot.positions_sync(rig.g)
    assert rig.bot.people.get(GUILD, STAFF).role_title == "CTO"
    rig.bot._people_stage_roles(rig.g)
    assert rig.bot.people.get(GUILD, STAFF).role_title == "CTO"
    row = next(r for r in rig.bot._people_roster(rig.g, fresh=True) if r.user_id == STAFF)
    assert row.title == "CTO"


# ------------------------------------------------------------------ tools in chat
async def tools_for(bot, world, who):
    member = world.g.members[who]
    where = SimpleNamespace(guild=None, channel=None)
    mem = await bot._memory_tools(where, world.g, member)
    is_exec = bot.is_exec_member(member) or who in (OWNER_ID, ADMIN_ID, BOSS)
    return {f.__name__: f for f in bot._people_tools(None, world.g, member, is_exec, mem, "en")}


async def test_the_new_tools_reach_chat_and_voice_through_the_one_builder(rig):
    member = rig.g.members[STAFF]
    mem = await rig.bot._memory_tools(SimpleNamespace(guild=None, channel=None), rig.g, member)
    msg = rig.bot.room_message(rig.g, member, rig.w.pub)
    kw = rig.bot._agent_kwargs(
        msg, rig.g, member, False, setting="th", lang="th", question="q", memory=mem, profile=None,
    )  # fmt: skip
    assert {"org_chart", "sync_roles", "positions_catalog", "set_role"} <= {
        f.__name__ for f in kw["extra_tools"]
    }
    rig.bot.guilds_store.save_settings(
        GUILD, GuildSettingsUpdate(features={"role_sync": False}), "x"
    )
    rig.bot.team._cache.clear()
    off = rig.bot._agent_kwargs(
        msg, rig.g, member, False, setting="th", lang="th", question="q", memory=mem, profile=None,
    )  # fmt: skip
    got = {f.__name__ for f in off["extra_tools"]}
    assert "set_role" in got and "org_chart" not in got


async def test_set_role_tags_the_member_and_says_so(rig):
    tools = await tools_for(rig.bot, rig.w, BOSS)
    out = await asyncio.to_thread(tools["set_role"], "Peer", "CTO", "tech")
    assert out["status"] == "saved" and out["position"] == "CTO" and out["tagged"] is True
    assert "CTO" in names(rig.g.members[PEER])
    assert [r.name for r in rig.g.created] == ["CTO"]
    assert rig.bot.positions.of(GUILD, PEER)[0].source == "stated"


async def test_set_role_without_manage_roles_says_no_permission(rig):
    wire(rig.w, manage_roles=False)
    tools = await tools_for(rig.bot, rig.w, BOSS)
    out = await asyncio.to_thread(tools["set_role"], "Peer", "CTO")
    assert out["tagged"] == "no_permission" and rig.g.created == []
    assert rig.bot.positions.of(GUILD, PEER) is not None  # the position itself is kept


async def test_set_role_on_the_bots_own_loop_schedules_and_answers_pending(rig):
    tools = await tools_for(rig.bot, rig.w, BOSS)
    out = tools["set_role"]("Peer", "COO")  # called on the event loop thread: cannot wait
    assert out["tagged"] == "pending"
    for _ in range(50):
        await asyncio.sleep(0.02)
        if "COO" in names(rig.g.members[PEER]):
            break
    assert "COO" in names(rig.g.members[PEER])


async def test_sync_roles_tool_runs_the_sync_and_reports_counts(rig):
    rig.bot.positions.assign(GUILD, PEER, "cto", "stated")
    tools = await tools_for(rig.bot, rig.w, BOSS)
    out = await asyncio.to_thread(tools["sync_roles"])
    assert out["status"] == "synced" and out["tagged"] == 1 and out["roles_created"] == 1
    plain = await tools_for(rig.bot, rig.w, INTERN)
    assert plain["sync_roles"]()["error"] == "not_allowed"


async def test_org_chart_and_knowledge_status_in_chat(rig):
    link(rig.bot, ("tom@x.com", STAFF))
    rig.bot.positions.assign(GUILD, BOSS, "ceo", "stated")
    tools = await tools_for(rig.bot, rig.w, BOSS)
    chart = await asyncio.to_thread(tools["org_chart"])
    assert chart["levels"][0]["people"][0]["name"] == "Boss"
    assert chart["no_position_count"] >= 3
    assert "Ploy" in chart["not_linked_to_center"]
    status = await asyncio.to_thread(tools["knowledge_status"])
    assert status["not_linked_to_center_count"] >= 4 and status["positions"]["assigned"] == 1
