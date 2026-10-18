"""Org-wide widening needs a FRESH Slack verdict: a person deleted in Slack loses the org-wide
scope within five minutes, not ten."""

import pytest
from conftest import FixedClockSettings
from platform_fakes import Clock, make_assistant
from test_org_scope_wiring import StoreAsk, ask
from test_slack import BOT, TEAM, FakeSlack

from salid.coo_store import SQLiteDocStore
from salid.identity_links import IdentityLinks
from salid.orgs import Member, OrgStore, Workspace, namespaced_id, workspace_id
from salid.platforms.core import VERIFY_TTL_S
from salid.platforms.org_sources import CachedProvider, OrgSources
from salid.platforms.slack import USER_TTL_S, SlackPort

LIMIT_S = 300.0


class World:
    pass


@pytest.fixture
def s():
    return FixedClockSettings(exec_user_ids=set())


@pytest.fixture
def wf(s, tmp_path):
    w = World()
    w.client = FakeSlack()
    w.client.users["U1"]["profile"] = {"email": "mew@example.com"}
    w.docs = SQLiteDocStore(":memory:")
    w.orgs = OrgStore(w.docs)
    w.orgs.create(
        "Cafe",
        "boss@example.com",
        org_id="org_a",
        members=[Member(email="boss@example.com", role="owner"), Member(email="mew@example.com")],
    )
    for platform, key in (("slack", TEAM), ("line", "Ubot")):
        w.orgs.attach_workspace(
            "org_a", Workspace(id=workspace_id(platform, key), platform=platform, external_id=key)
        )
    w.orgs.link_identity("org_a", "mew@example.com", "slack", "U1", TEAM)
    w.orgs.link_identity("org_a", "mew@example.com", "line", "Lmew", "Ubot")
    w.clock = Clock()
    w.a = make_assistant(s, tmp_path, clock=w.clock, ask=StoreAsk())
    w.a.identity = IdentityLinks(w.orgs, clock=w.clock)
    w.a.org_sources = OrgSources(w.orgs, clock=w.clock)

    async def slack_rooms(ws, account):
        return {"C1": "ops"}

    w.a.org_sources.providers["slack"] = CachedProvider(slack_rooms, w.clock)
    w.port = SlackPort(w.client, TEAM, BOT, clock=w.clock)
    return w


def test_the_org_wide_path_never_trusts_a_verdict_older_than_five_minutes():
    assert VERIFY_TTL_S <= LIMIT_S and USER_TTL_S <= LIMIT_S


async def test_a_deleted_slack_account_loses_the_org_wide_scope_after_five_minutes(wf):
    await ask(wf, port=wf.port)
    assert wf.a.ask_fn.stores[-1] is not wf.a.store  # verified: the multi-guild ledger
    line_room = namespaced_id("line", "Lmew", "channel")
    assert line_room in wf.a.ask_fn.calls[-1][1]["memory"].allowed
    wf.client.users["U1"]["deleted"] = True
    wf.clock.advance(LIMIT_S + 1)
    await ask(wf, port=wf.port)
    assert wf.a.ask_fn.stores[-1] is wf.a.store  # single tenant only
    assert line_room not in wf.a.ask_fn.calls[-1][1]["memory"].allowed
    assert wf.orgs.member_by_identity("org_a", "slack", "U1") is None
