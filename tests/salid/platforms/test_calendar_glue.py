"""Google Calendar tools in chat on the non-Discord platforms: who gets them, and that the confirm
rule is bound to the conversation. No network: a fake OAuth store and a fake GoogleCalendar."""

import asyncio
from zoneinfo import ZoneInfo

import pytest
from conftest import FixedClockSettings
from platform_fakes import NOW, FakeAsk, FakePort, inc, make_assistant

from salid.coo_store import SQLiteDocStore
from salid.google_oauth import GoogleConfig
from salid.identity_links import IdentityLinks
from salid.orgs import Member, OrgStore, Workspace, workspace_id
from salid.platforms.calendar_glue import CalendarGlue
from salid.platforms.config import PlatformSettings
from salid.scheduling import GoogleEvent

KEY = "A" * 43 + "="  # a valid Fernet key
BKK = ZoneInfo("Asia/Bangkok")
TEAM = "T1"


class FakeOAuth:
    def __init__(self, connected=()):
        self.connected = set(connected)
        self.checks = 0

    def is_connected(self, org_id, email):
        self.checks += 1
        return (org_id, email.lower()) in self.connected


class FakeCalendar:
    def __init__(self):
        self.created: list[tuple] = []

    async def create_event_full(self, account, spec, emails):
        # the tool must reach us on the event loop (run_coroutine_threadsafe from the worker)
        asyncio.get_running_loop()
        self.created.append((account.org_id, account.email, spec.title, list(emails)))
        return GoogleEvent("ev1", "https://cal/x", "https://meet.google.com/x")


@pytest.fixture
def s():
    return FixedClockSettings(exec_user_ids=set())


class World:
    pass


@pytest.fixture
def w(s, tmp_path):
    w = World()
    docs = SQLiteDocStore(":memory:")
    w.orgs = OrgStore(docs)
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
    w.orgs.attach_workspace(
        "org_a", Workspace(id=workspace_id("slack", TEAM), platform="slack", external_id=TEAM)
    )
    w.orgs.link_identity("org_a", "mew@example.com", "slack", "U1")
    w.orgs.link_identity("org_a", "boss@example.com", "slack", "U2")
    w.a = make_assistant(s, tmp_path, ask=FakeAsk())
    w.a.identity = IdentityLinks(w.orgs, clock=lambda: NOW)
    w.oauth = FakeOAuth({("org_a", "mew@example.com")})
    w.cal = FakeCalendar()
    w.glue = CalendarGlue(w.oauth, w.cal, BKK)
    w.a.calendar = w.glue
    w.port = FakePort()
    return w


async def ask(w, **kw):
    kw = {"is_direct": True, "text": "book a meeting", **kw}
    await w.a.handle(inc(**kw), w.port)
    return w.a.ask_fn.calls[-1][1]


def tool_map(kw):
    return {f.__name__: f for f in kw.get("extra_tools", [])}


async def schedule(tools, **over):
    args = dict(
        title="Sync",
        start="2026-10-20T14:00:00",
        end="2026-10-20T15:00:00",
        attendees=["ann"],
        description="",
    )
    args.update(over)
    return await asyncio.to_thread(tools["schedule_google_meeting"], **args)


async def test_verified_and_connected_gets_the_tools(w):
    kw = await ask(w, platform="slack", tenant=TEAM, user="U1")
    tools = tool_map(kw)
    assert set(tools) == {"schedule_google_meeting", "cancel_google_meeting"}
    assert w.oauth.checks == 1


async def test_scope_and_request_id_come_from_the_message(w):
    from salid.platforms import calendar_glue as cg

    seen = {}
    real = cg.CalendarToolContext

    def spy(**kw):
        seen.update(kw)
        return real(**kw)

    cg.CalendarToolContext = spy
    try:
        msg = inc(platform="slack", tenant=TEAM, user="U1", message="9.1", thread="8.0")
        tools = await cg.calendar_tools_for(w.a, msg, 11, 22, "en", False)
    finally:
        cg.CalendarToolContext = real
    assert len(tools) == 2
    assert seen["scope"] == ("slack", "11", "22", "8.0")
    assert seen["request_id"] == "|".join(msg.key)
    assert seen["email"] == "mew@example.com" and seen["org"].id == "org_a"
    assert seen["is_exec"] is False  # a plain member
    other = inc(platform="slack", tenant=TEAM, user="U1", message="9.2")
    assert "|".join(other.key) != seen["request_id"]


async def test_org_owner_counts_as_executive(w):
    from salid.platforms import calendar_glue as cg

    w.oauth.connected.add(("org_a", "boss@example.com"))
    seen = {}
    real = cg.CalendarToolContext
    cg.CalendarToolContext = lambda **kw: (seen.update(kw), real(**kw))[1]
    try:
        await cg.calendar_tools_for(
            w.a, inc(platform="slack", tenant=TEAM, user="U2"), 1, 2, "en", False
        )
    finally:
        cg.CalendarToolContext = real
    assert seen["is_exec"] is True


async def test_unverified_sender_gets_nothing(w):
    kw = await ask(w, platform="slack", tenant=TEAM, user="U9")
    assert "extra_tools" not in kw
    assert w.oauth.checks == 0


async def test_verified_but_not_connected_gets_nothing(w):
    kw = await ask(w, platform="slack", tenant=TEAM, user="U2")  # boss linked, not connected
    assert "extra_tools" not in kw


async def test_tenant_without_an_org_gets_nothing(w):
    kw = await ask(w, platform="slack", tenant="T-other", user="U1")
    assert "extra_tools" not in kw


async def test_feature_off_gets_nothing(w):
    w.a.calendar = None
    kw = await ask(w, platform="slack", tenant=TEAM, user="U1")
    assert "extra_tools" not in kw
    assert w.oauth.checks == 0


async def test_connected_check_is_cached_for_a_minute(w):
    now = [1000.0]
    w.glue._mono = lambda: now[0]
    await ask(w, platform="slack", tenant=TEAM, user="U1", message="1.1")
    await ask(w, platform="slack", tenant=TEAM, user="U1", message="1.2")
    assert w.oauth.checks == 1
    now[0] += 61
    await ask(w, platform="slack", tenant=TEAM, user="U1", message="1.3")
    assert w.oauth.checks == 2


async def test_draft_then_confirm_in_a_later_message_creates_the_event(w):
    kw1 = await ask(w, platform="slack", tenant=TEAM, user="U1", message="1.1")
    draft = await schedule(tool_map(kw1))
    assert draft["status"] == "needs_confirmation" and w.cal.created == []
    kw2 = await ask(w, platform="slack", tenant=TEAM, user="U1", message="1.2", text="yes")
    # the same message cannot both draft and confirm ...
    done = await schedule(tool_map(kw2), confirm=True)
    assert done["status"] == "scheduled" and done["meet_url"].startswith("https://meet")
    assert w.cal.created == [("org_a", "mew@example.com", "Sync", ["ann@example.com"])]


async def test_confirm_in_the_same_message_does_not_create(w):
    kw = await ask(w, platform="slack", tenant=TEAM, user="U1", message="1.1")
    t = tool_map(kw)
    assert (await schedule(t))["status"] == "needs_confirmation"
    assert (await schedule(t, confirm=True))["status"] == "needs_confirmation"
    assert w.cal.created == []


async def test_confirm_from_another_room_fails(w):
    kw1 = await ask(w, platform="slack", tenant=TEAM, user="U1", room="C1", message="1.1")
    assert (await schedule(tool_map(kw1)))["status"] == "needs_confirmation"
    kw2 = await ask(w, platform="slack", tenant=TEAM, user="U1", room="C2", message="2.1")
    r = await schedule(tool_map(kw2), confirm=True)
    assert r["status"] == "needs_confirmation" and w.cal.created == []


async def test_confirm_in_another_thread_fails(w):
    kw1 = await ask(w, platform="slack", tenant=TEAM, user="U1", message="1.1", thread="t1")
    assert (await schedule(tool_map(kw1)))["status"] == "needs_confirmation"
    kw2 = await ask(w, platform="slack", tenant=TEAM, user="U1", message="1.2", thread="t2")
    assert (await schedule(tool_map(kw2), confirm=True))["status"] == "needs_confirmation"
    assert w.cal.created == []


async def test_platform_exec_does_not_allow_outside_attendees(w):
    """A plain org member who is a platform exec / Slack workspace admin (is_exec=True from the
    bot) still cannot invite an outside email: only the verified org role counts."""
    from salid.platforms import calendar_glue as cg

    msg = inc(platform="slack", tenant=TEAM, user="U1", message="3.1")
    tools = {f.__name__: f for f in await cg.calendar_tools_for(w.a, msg, 1, 2, "en", True)}
    r = await schedule(tools, attendees=["outsider@other.com"])
    assert r["error"] == "unknown_attendees" and r["unresolved"] == ["outsider@other.com"]
    # the org owner (verified role) can
    w.oauth.connected.add(("org_a", "boss@example.com"))
    msg = inc(platform="slack", tenant=TEAM, user="U2", message="3.2")
    tools = {f.__name__: f for f in await cg.calendar_tools_for(w.a, msg, 1, 2, "en", False)}
    r = await schedule(tools, attendees=["outsider@other.com"])
    assert r["status"] == "needs_confirmation"


async def test_timeout_cancels_and_reports_outcome_unknown(w, monkeypatch):
    from salid.platforms import calendar_glue as cg

    cancelled = asyncio.Event()

    async def hang(account, spec, emails):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    w.cal.create_event_full = hang
    monkeypatch.setattr(cg, "RUN_TIMEOUT_S", 0.2)
    kw1 = await ask(w, platform="slack", tenant=TEAM, user="U1", message="4.1")
    assert (await schedule(tool_map(kw1)))["status"] == "needs_confirmation"
    kw2 = await ask(w, platform="slack", tenant=TEAM, user="U1", message="4.2", text="yes")
    r = await schedule(tool_map(kw2), confirm=True)
    assert r["status"] == "outcome_unknown"
    assert "NOT" in r["instruction"] and "calendar" in r["instruction"]
    await asyncio.wait_for(cancelled.wait(), 2)


# ------------------------------------------------------------------ settings
ENV = {
    "GOOGLE_OAUTH_CLIENT_ID": "cid",
    "GOOGLE_OAUTH_CLIENT_SECRET": "csec",
    "GOOGLE_OAUTH_REDIRECT_URI": "https://x/cb",
    "SALID_TOKEN_KEY": KEY,
}


def test_settings_read_the_web_env_names(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    ps = PlatformSettings(_env_file=None)
    assert (ps.google_oauth_client_id, ps.google_oauth_redirect_uri) == ("cid", "https://x/cb")
    assert ps.salid_token_key == KEY
    assert "csec" not in repr(ps) and KEY not in repr(ps)  # secrets stay out of the repr


@pytest.mark.parametrize("missing", list(ENV))
def test_feature_off_when_any_setting_is_missing(monkeypatch, missing):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    docs = SQLiteDocStore(":memory:")
    assert CalendarGlue.from_settings(PlatformSettings(_env_file=None), docs, "Asia/Bangkok")
    monkeypatch.delenv(missing)
    ps = PlatformSettings(_env_file=None)
    assert CalendarGlue.from_settings(ps, docs, "Asia/Bangkok") is None
    assert not GoogleConfig.from_settings(ps).enabled()


async def test_line_gets_no_calendar_tools_unless_org_linking_is_on(w):
    """LINE User Data Policy 3.2.5: a LINE sender never reaches the org's tools by default."""
    from salid.platforms import calendar_glue as cg

    w.orgs.attach_workspace(
        "org_a", Workspace(id=workspace_id("line", "Ubot"), platform="line", external_id="Ubot")
    )
    w.orgs.link_identity("org_a", "mew@example.com", "line", "Ualice")
    msg = inc(platform="line", tenant="Ubot", room="Cg", user="Ualice")
    assert w.a.ps.line_org_linking is False
    assert await cg.calendar_tools_for(w.a, msg, 1, 2, "en", False) == []
    assert w.oauth.checks == 0
    w.a.ps.line_org_linking = True  # the same flag that gates the org-wide scope
    assert len(await cg.calendar_tools_for(w.a, msg, 1, 2, "en", False)) == 2
    # Slack is not touched by the LINE flag
    w.a.ps.line_org_linking = False
    slack = inc(platform="slack", tenant=TEAM, user="U1")
    assert len(await cg.calendar_tools_for(w.a, slack, 1, 2, "en", False)) == 2
