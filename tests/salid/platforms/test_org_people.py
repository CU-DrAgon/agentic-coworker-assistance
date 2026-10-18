"""Org record + people tools for a verified member on the non-Discord platforms."""

import pytest
from conftest import FixedClockSettings
from platform_fakes import NOW, FakeAsk, FakePort, inc, make_assistant

from salid.coo_store import SQLiteDocStore
from salid.identity_links import IdentityLinks
from salid.orgs import Member, OrgStore, Position, Workspace, workspace_id
from salid.platforms.config import PlatformSettings

TEAM = "T1"
NAMES = {"my_profile", "org_people", "who_approves"}


class World:
    pass


@pytest.fixture
def w(tmp_path):
    w = World()
    w.orgs = OrgStore(SQLiteDocStore(":memory:"))
    w.orgs.create(
        "Cafe",
        "boss@example.com",
        org_id="org_a",
        members=[
            Member(email="boss@example.com", role="owner", title="CEO", level=1),
            Member(email="mew@example.com", title="Accountant", department="finance", finance=True),
            Member(email="ann@example.com", department="tech"),
        ],
    )
    org = w.orgs.get("org_a")
    org.positions = [Position(id="p1", title="Head Barista", holders=["ann@example.com"])]
    w.orgs.save(org)
    for plat, tenant in (("slack", TEAM), ("line", "L1")):
        w.orgs.attach_workspace(
            "org_a",
            Workspace(id=workspace_id(plat, tenant), platform=plat, external_id=tenant),
        )
    w.orgs.link_identity("org_a", "mew@example.com", "slack", "U1")
    w.orgs.link_identity("org_a", "mew@example.com", "line", "UL1", tenant="L1")
    w.make = lambda **ps: _assistant(w, tmp_path, **ps)
    w.port = FakePort()
    return w


def _assistant(w, tmp_path, **ps):
    a = make_assistant(FixedClockSettings(exec_user_ids=set()), tmp_path, ask=FakeAsk())
    a.identity = IdentityLinks(w.orgs, clock=lambda: NOW)
    if ps:
        a.ps = PlatformSettings(**ps)
    return a


async def ask(a, port, **kw):
    kw = {"is_direct": True, "text": "what is my position?", **kw}
    await a.handle(inc(**kw), port)
    return {f.__name__: f for f in a.ask_fn.calls[-1][1].get("extra_tools", [])}


async def test_verified_slack_dm_gets_tools_and_profile(w):
    tools = await ask(w.make(), w.port, platform="slack", tenant=TEAM, user="U1")
    assert set(tools) >= NAMES
    me = tools["my_profile"]()
    assert me["title"] == "Accountant" and me["role"] == "member"
    assert me["finance_approver"] is True and me["email"] == "mew@example.com"
    assert me["organisation"] == "Cafe" and "slack" in me["linked_platforms"]
    ppl = tools["org_people"]("ceo")["people"]
    assert [p["name"] for p in ppl] == ["boss"] and ppl[0]["role"] == "owner"
    # the position catalog names the job of someone with no title of their own
    assert tools["org_people"]("barista")["people"][0]["title"] == "Head Barista"
    ap = tools["who_approves"]()
    assert [p["name"] for p in ap["authorized_approvers"]] == ["mew"]
    assert [p["name"] for p in ap["owners_and_admins"]] == ["boss"]


async def test_unverified_gets_no_tools(w):
    tools = await ask(w.make(), w.port, platform="slack", tenant=TEAM, user="UNKNOWN")
    assert not NAMES & set(tools)


async def test_line_needs_org_linking(w):
    port = FakePort(platform="line")
    off = await ask(w.make(), port, platform="line", tenant="L1", user="UL1")
    assert not NAMES & set(off)
    on = await ask(w.make(line_org_linking=True), port, platform="line", tenant="L1", user="UL1")
    assert set(on) >= NAMES


async def test_group_room_hides_emails(w):
    tools = await ask(
        w.make(),
        w.port,
        platform="slack",
        tenant=TEAM,
        user="U1",
        is_direct=False,
        mentions_bot=True,
    )
    assert set(tools) >= NAMES
    assert "email" not in tools["my_profile"]()
    assert all("email" not in p for p in tools["org_people"]()["people"])
    assert "@" not in str(tools["who_approves"]())
