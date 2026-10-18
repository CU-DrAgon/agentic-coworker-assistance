"""Verified identities on the chat platforms: link / attach codes typed in a 1:1 (never in a group,
never seen by the model or the memory), guessing limits, Slack email verification, revocation."""

import itertools
from datetime import timedelta
from pathlib import Path

import pytest
from conftest import FixedClockSettings
from platform_fakes import NOW, Clock, FakePort, inc, make_assistant, stored
from test_slack import BOT, TEAM, FakeSlack

from salid.coo_store import SQLiteDocStore
from salid.identity_links import (
    REPLY_ATTACHED,
    REPLY_INVALID,
    REPLY_LINKED,
    REPLY_ONE_CODE,
    REPLY_PRIVATE,
    REPLY_SLOW_DOWN,
    IdentityLinks,
)
from salid.orgs import Member, OrgStore, Workspace, workspace_id
from salid.platforms.core import LINK_ATTEMPTS, LINK_WINDOW_S, VERIFY_TTL_S, AccountState
from salid.platforms.slack import SlackAdapter, SlackPort

_n = itertools.count(1)


@pytest.fixture
def s():
    return FixedClockSettings(exec_user_ids=set())


class World:
    pass


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
            Member(email="mew@example.com"),
            Member(email="ann@example.com"),
        ],
    )
    w.orgs.create("Other", "o@example.com", org_id="org_b")
    w.links = IdentityLinks(w.orgs, clock=lambda: NOW)
    w.clock = Clock()
    w.a = make_assistant(s, tmp_path, clock=w.clock)
    w.a.identity = w.links
    w.port = FakePort(platform="line")
    return w


def attach(orgs, org_id, platform, tenant):
    orgs.attach_workspace(
        org_id,
        Workspace(id=workspace_id(platform, tenant), platform=platform, external_id=tenant),
    )


async def say(w, text, *, user="U-mew", direct=True, platform="line", tenant="Ubot", **kw):
    msg = inc(
        platform=platform,
        tenant=tenant,
        room=user if direct else "Group1",
        user=user,
        text=text,
        message=f"m{next(_n)}",
        is_direct=direct,
        **kw,
    )
    await w.a.handle(msg, w.port)
    await w.a.drain()
    return msg


def code_for(w, email="mew@example.com", platform="line"):
    return w.links.create_code("org_a", email, platform).code


# ------------------------------------------------------------------ redeeming in a 1:1
async def test_link_code_in_a_private_chat_links_the_account_without_the_model(wd):
    attach(wd.orgs, "org_a", "line", "Ubot")
    code = code_for(wd)
    msg = await say(wd, f"link {code}")
    assert wd.port.texts == [REPLY_LINKED]
    assert wd.a.ask_fn.calls == [] and wd.a.gate_calls == []
    assert wd.orgs.member_by_identity("org_a", "line", "U-mew").email == "mew@example.com"
    assert wd.a.identity_of(msg) == ("org_a", "mew@example.com")
    # the code is neither remembered nor in the live buffer the model would read
    assert stored(wd.a, msg) == []
    assert not wd.a.buffer.lines((0, 0, None))
    assert all(code not in line.text for k in wd.a.buffer._d for _, line in wd.a.buffer._d[k])


async def test_thai_command(wd):
    attach(wd.orgs, "org_a", "line", "Ubot")
    await say(wd, f"เชื่อมบัญชี {code_for(wd)}")
    assert wd.port.texts == [REPLY_LINKED] and not wd.a.ask_fn.calls


def assert_unseen(w, msg, *codes):
    """The message reached neither the model, the gate, the live buffer nor the memory."""
    assert w.a.ask_fn.calls == [] and w.a.gate_calls == []
    assert stored(w.a, msg) == []
    assert not w.a.buffer._d
    assert all(c not in repr(w.a.buffer._d) for c in codes)


@pytest.mark.parametrize(
    "template",
    [
        "can you link {code} for me",
        "please attach this: {code} thanks",
        "my code is {code}",
        "รหัสของฉันคือ {code} ช่วยเชื่อมให้หน่อย",
    ],
)
async def test_a_code_inside_a_sentence_in_a_private_chat_is_redeemed_not_answered(wd, template):
    attach(wd.orgs, "org_a", "line", "Ubot")
    code = code_for(wd)
    msg = await say(wd, template.format(code=code))
    assert wd.port.texts == [REPLY_LINKED]
    assert wd.orgs.member_by_identity("org_a", "line", "U-mew").email == "mew@example.com"
    assert_unseen(wd, msg, code, code.replace("-", ""))


async def test_a_sentence_with_a_code_in_a_group_is_never_seen_by_the_model_or_memory(wd):
    attach(wd.orgs, "org_a", "line", "Ubot")
    code = code_for(wd)
    msg = await say(wd, f"@Salid can you link {code} for me", direct=False, mentions_bot=True)
    assert wd.port.texts == [REPLY_PRIVATE]
    assert wd.orgs.member_by_identity("org_a", "line", "U-mew") is None
    assert_unseen(wd, msg, code)
    await say(wd, f"link {code}")  # not consumed: the private chat still works
    assert wd.port.texts[-1] == REPLY_LINKED


async def test_several_codes_in_one_message_get_the_instruction_and_redeem_nothing(wd):
    attach(wd.orgs, "org_a", "line", "Ubot")
    one, two = code_for(wd), code_for(wd)
    msg = await say(wd, f"link {one} and also {two}")
    assert wd.port.texts == [REPLY_ONE_CODE]
    assert wd.orgs.member_by_identity("org_a", "line", "U-mew") is None
    assert_unseen(wd, msg, one, two)


async def test_a_number_without_a_link_word_is_just_a_message(wd):
    await say(wd, "my number is 1234-5678")
    assert len(wd.a.ask_fn.calls) == 1


async def test_an_edited_message_with_a_code_is_not_remembered(wd):
    attach(wd.orgs, "org_a", "line", "Ubot")
    code = code_for(wd)
    msg = await say(wd, "hello")
    edited = inc(**{**msg.__dict__, "text": f"oops link {code}"})
    await wd.a.on_edit(edited)
    await wd.a.drain()
    assert all(code not in line.text for k in wd.a.buffer._d for _, line in wd.a.buffer._d[k])
    assert all(code not in (r.text or "") for r in stored(wd.a, msg))


async def test_without_an_identity_store_nothing_is_intercepted(wd):
    wd.a.identity = None
    await say(wd, "link 1234-5678")
    assert len(wd.a.ask_fn.calls) == 1


@pytest.mark.parametrize("platform", ["telegram", "whatsapp", "line"])
async def test_every_bot_platform_can_link(wd, platform):
    wd.port = FakePort(platform=platform)
    code = code_for(wd, platform=platform)
    await say(wd, f"/link {code}", platform=platform, tenant="bot9", user="1001")
    assert wd.port.texts == [REPLY_LINKED]
    assert wd.orgs.member_by_identity("org_a", platform, "1001").email == "mew@example.com"


# ------------------------------------------------------------------ refusals
async def test_group_chat_never_redeems_and_says_so(wd):
    attach(wd.orgs, "org_a", "line", "Ubot")
    code = code_for(wd)
    msg = await say(wd, f"link {code}", direct=False)
    assert wd.port.texts == [REPLY_PRIVATE]
    assert wd.orgs.member_by_identity("org_a", "line", "U-mew") is None
    assert not wd.a.ask_fn.calls and stored(wd.a, msg) == []
    # the code was not consumed: the private chat still works
    await say(wd, f"link {code}")
    assert wd.port.texts[-1] == REPLY_LINKED


async def test_group_attach_code_is_not_redeemed_either(wd):
    code = wd.links.create_attach_code("org_a", "boss@example.com", "line").code
    await say(wd, f"link {code}", direct=False)
    assert wd.port.texts == [REPLY_PRIVATE]
    assert wd.orgs.org_of_workspace("line", "Ubot") is None


async def test_group_reply_is_rate_limited_per_room(wd):
    wd.a.ps.platform_answers_per_minute = 2
    for _ in range(4):
        await say(wd, "link 1234-5678", direct=False)
    assert wd.port.texts == [REPLY_PRIVATE, REPLY_PRIVATE]


async def test_unknown_expired_and_reused_codes_all_get_the_same_vague_refusal(wd):
    attach(wd.orgs, "org_a", "line", "Ubot")
    code = code_for(wd)
    await say(wd, "link 1111-2222")  # unknown
    await say(wd, f"link {code}")  # ok
    await say(wd, f"link {code}", user="U-eve")  # reused
    old = wd.links.create_code("org_a", "ann@example.com", "line").code
    wd.links._clock = lambda: NOW + timedelta(minutes=16)
    await say(wd, f"link {old}", user="U-ann")  # expired
    assert wd.port.texts == [REPLY_INVALID, REPLY_LINKED, REPLY_INVALID, REPLY_INVALID]
    assert wd.orgs.member_by_identity("org_a", "line", "U-eve") is None
    assert wd.orgs.member_by_identity("org_a", "line", "U-ann") is None
    assert not wd.a.ask_fn.calls


async def test_wrong_platform_code_is_refused(wd):
    code = code_for(wd, platform="telegram")
    await say(wd, f"link {code}")  # typed to LINE
    assert wd.port.texts == [REPLY_INVALID]


async def test_account_already_linked_to_another_member_is_refused(wd):
    attach(wd.orgs, "org_a", "line", "Ubot")
    wd.orgs.link_identity("org_a", "ann@example.com", "line", "U-shared")
    await say(wd, f"link {code_for(wd)}", user="U-shared")
    assert wd.port.texts == [REPLY_INVALID]
    assert wd.orgs.member_by_identity("org_a", "line", "U-shared").email == "ann@example.com"


async def test_bot_of_another_org_is_refused(wd):
    attach(wd.orgs, "org_b", "line", "Ubot")
    await say(wd, f"link {code_for(wd)}")
    assert wd.port.texts == [REPLY_INVALID]
    assert wd.orgs.member_by_identity("org_a", "line", "U-mew") is None


async def test_guessing_is_rate_limited_per_person_and_a_valid_code_waits(wd):
    attach(wd.orgs, "org_a", "line", "Ubot")
    code = code_for(wd)
    for i in range(LINK_ATTEMPTS):
        await say(wd, f"link 0000-000{i}")
    assert wd.port.texts == [REPLY_INVALID] * LINK_ATTEMPTS
    await say(wd, f"link {code}")  # even the right code is held back
    assert wd.port.texts[-1] == REPLY_SLOW_DOWN
    assert wd.orgs.member_by_identity("org_a", "line", "U-mew") is None
    await say(wd, "link 0000-0009", user="U-ann")  # somebody else is not locked out
    assert wd.port.texts[-1] == REPLY_INVALID
    wd.clock.advance(LINK_WINDOW_S + 1)
    await say(wd, f"link {code}")  # the code is still live and was never burned
    assert wd.port.texts[-1] == REPLY_LINKED


async def test_many_guessers_hit_the_tenant_limit(wd):
    for i in range(30):
        await say(wd, "link 0000-0001", user=f"U{i}")
    await say(wd, "link 0000-0002", user="U-late")
    assert wd.port.texts[-1] == REPLY_SLOW_DOWN


# ------------------------------------------------------------------ attaching a bot tenant
async def test_attach_code_connects_the_bot_to_the_org(wd):
    code = wd.links.create_attach_code("org_a", "boss@example.com", "line").code
    msg = await say(wd, f"link {code}", user="U-anyone")
    assert wd.port.texts == [REPLY_ATTACHED]
    org, ws = wd.orgs.org_of_workspace("line", "Ubot")
    assert (
        org.id == "org_a"
        and ws.memory_id == wd.a.identity.orgs.get("org_a").workspaces[-1].memory_id
    )
    assert wd.a.identity_of(msg) is None  # attaching links no account: a guest still
    await say(wd, f"link {code_for(wd)}")
    assert wd.a.identity_of(msg) is None  # other user, still unknown
    assert wd.a.identity_of(inc(platform="line", tenant="Ubot", user="U-mew")) == (
        "org_a",
        "mew@example.com",
    )


async def test_attach_refuses_a_bot_that_belongs_to_another_org(wd):
    attach(wd.orgs, "org_b", "line", "Ubot")
    code = wd.links.create_attach_code("org_a", "boss@example.com", "line").code
    await say(wd, f"link {code}")
    assert wd.port.texts == [REPLY_INVALID]
    assert wd.orgs.org_of_workspace("line", "Ubot")[0].id == "org_b"


# ------------------------------------------------------------------ identity_of
async def test_identity_of_is_none_for_guests_and_unwired_assistants(wd):
    attach(wd.orgs, "org_a", "line", "Ubot")
    assert wd.a.identity_of(inc(platform="line", tenant="Ubot", user="U-guest")) is None
    wd.orgs.link_identity("org_a", "mew@example.com", "line", "U-mew")
    mine = inc(platform="line", tenant="Ubot", user="U-mew")
    assert wd.a.identity_of(mine) == ("org_a", "mew@example.com")
    assert wd.a.identity_of(inc(platform="line", tenant="Uelse", user="U-mew")) is None
    wd.a.identity = None
    assert wd.a.identity_of(mine) is None


# ------------------------------------------------------------------ Slack: automatic
@pytest.fixture
def slack(s, tmp_path):
    client = FakeSlack()
    for uid, extra in {
        "U1": {"profile": {"email": "Mew@Example.com"}},
        "U2": {"profile": {"email": "ann@example.com"}},
        "U3": {"profile": {"email": "stranger@example.com"}},
        "UX": {"profile": {"email": "mew@example.com"}},  # Slack Connect: another team
        "UB": {"profile": {"email": "mew@example.com"}},  # bot
    }.items():
        client.users[uid].update(extra)
    client.users["UG"] = {
        "id": "UG",
        "team_id": TEAM,
        "is_restricted": True,
        "profile": {"email": "mew@example.com"},
    }
    client.users["UN"]["profile"] = {}  # no email visible (scope missing)
    w = World()
    w.client, w.clock = client, Clock()
    w.docs = SQLiteDocStore(":memory:")
    w.orgs = OrgStore(w.docs)
    w.orgs.create(
        "Cafe",
        "boss@example.com",
        org_id="org_a",
        members=[
            Member(email="boss@example.com", role="owner"),
            Member(email="mew@example.com"),
            Member(email="ann@example.com"),
        ],
    )
    w.links = IdentityLinks(w.orgs, clock=lambda: NOW)
    w.a = make_assistant(s, tmp_path, clock=w.clock)
    w.a.identity = w.links
    w.port = SlackPort(client, TEAM, BOT, clock=w.clock)
    w.adapter = SlackAdapter(w.a, w.port, lambda url: b"")
    return w


async def slack_say(w, user, text="hello salid"):
    msg = inc(
        platform="slack", tenant=TEAM, room=f"D{user}", user=user, text=text,
        message=f"{next(_n)}.0", is_direct=True,
    )  # fmt: skip
    await w.a.handle(msg, w.port)
    await w.a.drain()
    return msg


async def test_slack_profile_email_only_for_real_members_of_this_team(slack):
    p = slack.port
    assert await p.profile_email("U1") == "mew@example.com"  # lower-cased
    assert await p.profile_email("UX") == ""  # another workspace
    assert await p.profile_email("UB") == ""  # bot
    assert await p.profile_email("UG") == ""  # guest
    assert await p.profile_email("UD") == ""  # deleted
    assert await p.profile_email("UN") == ""  # no team / no email
    assert await p.profile_email("U-missing") == ""  # users.info fails


async def test_slack_email_match_links_the_member_automatically(slack):
    attach(slack.orgs, "org_a", "slack", TEAM)
    msg = await slack_say(slack, "U1")
    assert slack.orgs.member_by_identity("org_a", "slack", "U1").email == "mew@example.com"
    assert slack.a.identity_of(msg) == ("org_a", "mew@example.com")
    assert len(slack.a.ask_fn.calls) == 1  # and the question was still answered


@pytest.mark.parametrize("user", ["UX", "UB", "UG", "U3", "UN"])
async def test_slack_email_match_refuses_everyone_else(slack, user):
    attach(slack.orgs, "org_a", "slack", TEAM)
    await slack_say(slack, user)
    assert slack.orgs.member_by_identity("org_a", "slack", user) is None
    assert all(not m.identities for m in slack.orgs.get("org_a").members)


async def test_slack_workspace_of_no_org_is_never_looked_up(slack):
    asked = []

    async def spy(uid):
        asked.append(uid)
        return AccountState("member", "mew@example.com")

    slack.port.account_state = spy
    await slack_say(slack, "U1")
    assert asked == []
    assert all(not m.identities for m in slack.orgs.get("org_a").members)


async def test_slack_lookup_is_throttled_and_retried_later(slack):
    attach(slack.orgs, "org_a", "slack", TEAM)
    asked = []
    real = slack.port.account_state

    async def spy(uid):
        asked.append(uid)
        return await real(uid)

    slack.port.account_state = spy
    await slack_say(slack, "U1")
    await slack_say(slack, "U1")
    assert asked == ["U1"]  # not asked again within the window
    slack.clock.advance(VERIFY_TTL_S + 1)
    await slack_say(slack, "U1")
    assert asked == ["U1", "U1"]
    assert slack.orgs.member_by_identity("org_a", "slack", "U1").email == "mew@example.com"


async def test_an_account_linked_to_another_member_is_not_taken_over(slack):
    attach(slack.orgs, "org_a", "slack", TEAM)
    slack.orgs.link_identity("org_a", "ann@example.com", "slack", "U9")  # Ann has another account
    await slack_say(slack, "U1")  # U1's address is Mew's: linked to Mew, Ann untouched
    assert who(slack, "U1") == "mew@example.com" and who(slack, "U9") == "ann@example.com"


async def test_slack_uninstall_forgets_the_member_links(slack):
    attach(slack.orgs, "org_a", "slack", TEAM)
    await slack_say(slack, "U1")
    assert slack.orgs.get("org_a").members[1].identities == {"slack": "U1"}
    await slack.adapter.on_event({"type": "app_uninstalled"}, TEAM)
    assert slack.orgs.get("org_a").members[1].identities == {}


# ------------------------------------------------------------------ Slack: revocation
def link_u1(slack):
    attach(slack.orgs, "org_a", "slack", TEAM)
    slack.orgs.link_identity("org_a", "mew@example.com", "slack", "U1")


def who(slack, uid="U1"):
    m = slack.orgs.member_by_identity("org_a", "slack", uid)
    return m.email if m else None


async def test_a_member_who_became_a_guest_is_unlinked_at_the_next_refresh(slack):
    link_u1(slack)
    first = await slack_say(slack, "U1")
    assert slack.a.identity_of(first) == ("org_a", "mew@example.com")
    slack.client.users["U1"]["is_restricted"] = True
    # within the window the cached verdict stands ...
    assert slack.a.identity_of(await slack_say(slack, "U1")) == ("org_a", "mew@example.com")
    slack.clock.advance(VERIFY_TTL_S + 1)  # (users.info is cached for the same 10 minutes)
    msg = await slack_say(slack, "U1")
    assert slack.a.identity_of(msg) is None and who(slack) is None


@pytest.mark.parametrize(
    "change",
    [
        {"deleted": True},
        {"is_ultra_restricted": True},
        {"team_id": "T-ELSEWHERE"},
        {"profile": {"email": "someone.else@example.com"}},
        {"is_bot": True},
    ],
)
async def test_deleted_restricted_other_team_or_new_email_loses_the_link(slack, change):
    link_u1(slack)
    slack.client.users["U1"].update(change)
    msg = await slack_say(slack, "U1")
    assert slack.a.identity_of(msg) is None
    assert who(slack) is None


async def test_a_new_email_that_matches_nobody_does_not_relink_to_someone_else(slack):
    link_u1(slack)
    slack.client.users["U1"]["profile"] = {"email": "ann@example.com"}
    msg = await slack_say(slack, "U1")
    assert who(slack) is None and slack.a.identity_of(msg) is None
    # the address now belongs to Ann; she links on the next refresh (not in this request)
    slack.clock.advance(VERIFY_TTL_S + 1)
    msg2 = await slack_say(slack, "U1")
    assert slack.a.identity_of(msg2) == ("org_a", "ann@example.com")


async def test_a_lookup_error_fails_closed_for_the_request_but_keeps_the_link(slack):
    link_u1(slack)
    del slack.client.users["U1"]  # users.info raises
    msg = await slack_say(slack, "U1")
    assert slack.a.identity_of(msg) is None
    assert who(slack) == "mew@example.com"  # not unlinked
    # the person comes back: the very next message (no waiting for the TTL) is verified again
    slack.client.users["U1"] = {
        "id": "U1",
        "team_id": TEAM,
        "profile": {"email": "mew@example.com"},
    }
    slack.clock.advance(31)  # only the short failure cache
    msg2 = await slack_say(slack, "U1")
    assert slack.a.identity_of(msg2) == ("org_a", "mew@example.com")


async def test_a_missing_email_scope_fails_closed_without_unlinking(slack):
    link_u1(slack)
    slack.client.users["U1"]["profile"] = {}
    msg = await slack_say(slack, "U1")
    assert slack.a.identity_of(msg) is None and who(slack) == "mew@example.com"


async def test_user_change_event_revokes_at_once(slack):
    link_u1(slack)
    assert slack.a.identity_of(await slack_say(slack, "U1")) == ("org_a", "mew@example.com")
    slack.client.users["U1"]["deleted"] = True
    await slack.adapter.on_event({"type": "user_change", "user": {"id": "U1"}}, TEAM)
    assert who(slack) is None  # unlinked before the person says another word
    msg = await slack_say(slack, "U1")  # (also within the verdict window)
    assert slack.a.identity_of(msg) is None


async def test_user_change_event_of_another_team_is_ignored(slack):
    link_u1(slack)
    slack.client.users["U1"]["deleted"] = True
    await slack.adapter.on_event({"type": "user_change", "user": {"id": "U1"}}, "T-OTHER")
    assert who(slack) == "mew@example.com"


async def test_user_change_does_not_unlink_a_person_who_is_fine(slack):
    link_u1(slack)
    await slack.adapter.on_event({"type": "user_change", "user": {"id": "U1"}}, TEAM)
    assert who(slack) == "mew@example.com"


def test_user_change_is_subscribed():
    from salid.platforms.slack import LIFECYCLE_EVENTS

    assert LIFECYCLE_EVENTS.match("user_change")
    text = (Path(__file__).parents[3] / "docs/platforms/slack.md").read_text()
    assert "- user_change" in text
