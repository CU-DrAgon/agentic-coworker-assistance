"""Google OAuth (state, PKCE, encrypted tokens), the GoogleCalendar backend and the agent tools.
No network: httpx.MockTransport."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import httpx
import pytest
from cryptography.fernet import Fernet

from salid.coo_store import SQLiteDocStore
from salid.google_oauth import (
    CALENDAR_SCOPE,
    STATE_KIND,
    TOKENS_KIND,
    Account,
    GoogleConfig,
    GoogleOAuth,
    GoogleOAuthError,
    NotConnected,
    pkce_challenge,
)
from salid.google_tools import CalendarToolContext, calendar_tools, resolve_attendees
from salid.orgs import Member, Org
from salid.scheduling import EventSpec, GoogleCalendar, GoogleCalendarError
from salid.secretary import SecretaryState

KEY = Fernet.generate_key().decode()
ORG = "org_t"
ME = "boss@example.com"
BKK = ZoneInfo("Asia/Bangkok")


def cfg(**kw):
    base = dict(client_id="cid", client_secret="csec", redirect_uri="https://x/cb", token_key=KEY)
    return GoogleConfig(**{**base, **kw})


class Google:
    """A scripted Google: records requests, answers token / userinfo / revoke / calendar."""

    def __init__(self):
        self.log = []
        self.userinfo = {"email": ME, "email_verified": True}
        self.scope = f"openid email {CALENDAR_SCOPE}"
        self.refresh_token = "RT-secret-1"
        self.token_status = 200
        self.token_error = ""
        self.calendar = lambda req: httpx.Response(404)

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.log.append(req)
        url = str(req.url)
        if url.startswith("https://oauth2.googleapis.com/token"):
            form = parse_qs(req.content.decode())
            if self.token_status != 200:
                return httpx.Response(self.token_status, json={"error": self.token_error})
            body = {"access_token": "AT-1", "expires_in": 3600, "scope": self.scope}
            if form["grant_type"] == ["authorization_code"] and self.refresh_token:
                body["refresh_token"] = self.refresh_token
            return httpx.Response(200, json=body)
        if "userinfo" in url:
            return httpx.Response(200, json=self.userinfo)
        if url.startswith("https://oauth2.googleapis.com/revoke"):
            return httpx.Response(200)
        return self.calendar(req)

    def of(self, fragment):
        return [r for r in self.log if fragment in str(r.url)]


@pytest.fixture
def g():
    return Google()


@pytest.fixture
def docs():
    return SQLiteDocStore(":memory:")


class Clock:
    def __init__(self):
        self.t = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)
        self.m = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def oauth(docs, g, clock):
    return GoogleOAuth(
        docs,
        cfg(),
        client=httpx.Client(transport=httpx.MockTransport(g)),
        clock=clock,
        mono=lambda: clock.m,
    )


def connect(oauth, code="the-code"):
    state = oauth.begin(ORG, ME)
    return oauth.exchange(code, state)


# ------------------------------------------------------------------ config
@pytest.mark.parametrize("missing", ["client_id", "client_secret", "redirect_uri", "token_key"])
def test_feature_off_when_any_setting_missing(missing):
    assert cfg().enabled()
    assert not cfg(**{missing: ""}).enabled()


def test_invalid_fernet_key_means_off():
    assert not cfg(token_key="not-a-key").enabled()


def test_secrets_not_in_repr():
    r = repr(cfg())
    assert "csec" not in r and KEY not in r


def test_begin_refused_when_off(docs):
    with pytest.raises(GoogleOAuthError) as e:
        GoogleOAuth(docs, cfg(client_id="")).begin(ORG, ME)
    assert e.value.code == "not_configured"


# ------------------------------------------------------------------ consent URL, state, PKCE
def test_auth_url_has_required_parameters(oauth):
    state = oauth.begin(ORG, ME)
    q = parse_qs(urlparse(oauth.auth_url(state)).query)
    assert q["client_id"] == ["cid"] and q["redirect_uri"] == ["https://x/cb"]
    assert q["response_type"] == ["code"]
    assert CALENDAR_SCOPE in q["scope"][0].split()
    assert q["access_type"] == ["offline"] and q["prompt"] == ["consent"]
    assert q["include_granted_scopes"] == ["true"]
    assert q["code_challenge_method"] == ["S256"] and q["state"] == [state]
    assert q["login_hint"] == [ME]


def test_pkce_challenge_matches_verifier_sent_on_exchange(oauth, docs, g):
    state = oauth.begin(ORG, ME)
    verifier = docs.get(STATE_KIND, 0, state)["verifier"]
    q = parse_qs(urlparse(oauth.auth_url(state)).query)
    assert q["code_challenge"] == [pkce_challenge(verifier)]
    oauth.exchange("c", state)
    form = parse_qs(g.of("oauth2.googleapis.com/token")[0].content.decode())
    assert form["code_verifier"] == [verifier]
    assert form["code"] == ["c"] and form["redirect_uri"] == ["https://x/cb"]


def test_pkce_rfc7636_vector():
    assert (
        pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk")
        == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    )


def test_state_is_single_use(oauth):
    state = oauth.begin(ORG, ME)
    oauth.exchange("c", state)
    with pytest.raises(GoogleOAuthError) as e:
        oauth.exchange("c", state)
    assert e.value.code == "bad_state"


def test_state_expires_after_ten_minutes_and_is_consumed(oauth, docs, clock):
    state = oauth.begin(ORG, ME)
    clock.t += timedelta(minutes=10, seconds=1)
    with pytest.raises(GoogleOAuthError) as e:
        oauth.exchange("c", state)
    assert e.value.code == "expired"
    assert docs.get(STATE_KIND, 0, state) is None


def test_unknown_state_refused(oauth):
    with pytest.raises(GoogleOAuthError) as e:
        oauth.exchange("c", "nope")
    assert e.value.code == "bad_state"


def test_state_binds_org_and_member(oauth, docs):
    state = oauth.begin(ORG, "Boss@Example.com")
    acct = oauth.exchange("c", state)
    assert acct == Account(ORG, ME)
    assert oauth.status(ORG, ME)["connected"]
    assert not oauth.status("other_org", ME)["connected"]
    assert not oauth.status(ORG, "someone@example.com")["connected"]


def test_states_are_unguessable_and_distinct(oauth):
    a, b = oauth.begin(ORG, ME), oauth.begin(ORG, ME)
    assert a != b and len(a) >= 40


# ------------------------------------------------------------------ email check
def test_email_mismatch_refused_and_grant_revoked(oauth, g, docs):
    g.userinfo = {"email": "mallory@example.com", "email_verified": True}
    with pytest.raises(GoogleOAuthError) as e:
        connect(oauth)
    assert e.value.code == "email_mismatch"
    assert docs.find_items(TOKENS_KIND, 0) == []
    assert len(g.of("/revoke")) == 1


def test_unverified_email_refused(oauth, g):
    g.userinfo = {"email": ME, "email_verified": False}
    with pytest.raises(GoogleOAuthError) as e:
        connect(oauth)
    assert e.value.code == "email_mismatch"


def test_missing_calendar_scope_refused(oauth, g, docs):
    g.scope = "openid email"
    with pytest.raises(GoogleOAuthError) as e:
        connect(oauth)
    assert e.value.code == "scope_missing"
    assert docs.find_items(TOKENS_KIND, 0) == []


def test_no_refresh_token_refused(oauth, g):
    g.refresh_token = ""
    with pytest.raises(GoogleOAuthError) as e:
        connect(oauth)
    assert e.value.code == "no_refresh_token"


# ------------------------------------------------------------------ token store
def test_refresh_token_encrypted_round_trip_never_plaintext(oauth, docs, g):
    connect(oauth)
    items = docs.find_items(TOKENS_KIND, 0)
    assert len(items) == 1
    key, rec = items[0]
    dump = json.dumps(rec) + key
    assert "RT-secret-1" not in dump and "AT-1" not in dump
    assert Fernet(KEY.encode()).decrypt(rec["enc"].encode())  # real Fernet ciphertext
    assert key == f"{ORG}:{ME}"
    # a fresh instance (new process) can use it
    fresh = GoogleOAuth(
        docs, cfg(), client=httpx.Client(transport=httpx.MockTransport(g)), mono=lambda: 0.0
    )
    assert fresh.access_token(ORG, ME) == "AT-1"
    form = parse_qs(g.of("oauth2.googleapis.com/token")[-1].content.decode())
    assert form["grant_type"] == ["refresh_token"] and form["refresh_token"] == ["RT-secret-1"]


def test_wrong_key_makes_record_unreadable_and_dropped(oauth, docs, g):
    connect(oauth)
    other = GoogleOAuth(
        docs,
        cfg(token_key=Fernet.generate_key().decode()),
        client=httpx.Client(transport=httpx.MockTransport(g)),
    )
    with pytest.raises(NotConnected):
        other.access_token(ORG, ME)
    assert docs.find_items(TOKENS_KIND, 0) == []


def test_copied_record_cannot_be_replayed_for_another_member(oauth, docs):
    connect(oauth)
    rec = docs.get(TOKENS_KIND, 0, f"{ORG}:{ME}")
    docs.put(TOKENS_KIND, 0, f"{ORG}:eve@example.com", {**rec, "email": "eve@example.com"})
    with pytest.raises(NotConnected):
        oauth.access_token(ORG, "eve@example.com")


def test_access_token_cached_until_expiry(oauth, g, clock):
    connect(oauth)  # caches AT-1 (expires_in 3600 minus skew)
    n = len(g.of("oauth2.googleapis.com/token"))
    assert oauth.access_token(ORG, ME) == "AT-1"
    assert len(g.of("oauth2.googleapis.com/token")) == n  # served from memory
    clock.m += 3600
    oauth.access_token(ORG, ME)
    assert len(g.of("oauth2.googleapis.com/token")) == n + 1


def test_invalid_grant_deletes_record(oauth, docs, g, clock):
    connect(oauth)
    clock.m += 4000
    g.token_status, g.token_error = 400, "invalid_grant"
    with pytest.raises(NotConnected):
        oauth.access_token(ORG, ME)
    assert docs.find_items(TOKENS_KIND, 0) == []
    assert not oauth.is_connected(ORG, ME)


def test_other_refresh_errors_keep_record(oauth, docs, g, clock):
    connect(oauth)
    clock.m += 4000
    g.token_status, g.token_error = 503, ""
    with pytest.raises(GoogleOAuthError) as e:
        oauth.access_token(ORG, ME)
    assert e.value.code == "http_503"
    assert oauth.is_connected(ORG, ME)


def test_revoke_calls_google_and_deletes(oauth, docs, g):
    acct = connect(oauth)
    oauth.revoke(acct)
    rev = g.of("/revoke")
    assert len(rev) == 1 and parse_qs(rev[0].content.decode())["token"] == ["RT-secret-1"]
    assert docs.find_items(TOKENS_KIND, 0) == []
    oauth.revoke(acct)  # idempotent
    with pytest.raises(NotConnected):
        oauth.access_token(ORG, ME)


def test_revoke_deletes_even_if_google_unreachable(docs, g):
    def boom(req):
        if "/revoke" in str(req.url):
            raise httpx.ConnectError("down")
        return g(req)

    o = GoogleOAuth(docs, cfg(), client=httpx.Client(transport=httpx.MockTransport(boom)))
    acct = connect(o)
    o.revoke(acct)
    assert not o.is_connected(ORG, ME)


def test_tokens_never_in_logs(oauth, docs, g, caplog):
    caplog.set_level("DEBUG")
    connect(oauth)
    g.token_status, g.token_error = 400, "invalid_grant"
    oauth._cache.clear()
    with pytest.raises(NotConnected):
        oauth.access_token(ORG, ME)
    assert "RT-secret-1" not in caplog.text and "AT-1" not in caplog.text
    assert "csec" not in caplog.text


# ------------------------------------------------------------------ calendar backend
def make_calendar(g, oauth):
    transport = httpx.MockTransport(g)
    return GoogleCalendar(oauth, "Asia/Bangkok", client=httpx.AsyncClient(transport=transport))


def run(coro):
    # not asyncio.run: it leaves no current event loop behind, which breaks later tests
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


START = datetime(2026, 10, 20, 14, 0, tzinfo=BKK)
END = START + timedelta(hours=1)
ACCT = Account(ORG, ME)


def google_event(**kw):
    ev = {
        "id": "ev1",
        "htmlLink": "https://calendar.google.com/event?eid=1",
        "hangoutLink": "https://meet.google.com/abc-defg-hij",
        "summary": "Sync",
        "start": {"dateTime": "2026-10-20T14:00:00+07:00"},
        "extendedProperties": {"private": {"salid": "1"}},
    }
    return {**ev, **kw}


def test_create_event_payload_and_meet_request(oauth, g):
    connect(oauth)
    g.calendar = lambda r: httpx.Response(200, json=google_event())
    cal = make_calendar(g, oauth)
    ev = run(
        cal.create_event_full(
            ACCT, EventSpec("Sync", START, END, "agenda"), ["a@example.com", "b@example.com"]
        )
    )
    assert (ev.id, ev.url, ev.meet_url) == (
        "ev1",
        "https://calendar.google.com/event?eid=1",
        "https://meet.google.com/abc-defg-hij",
    )
    req = g.of("/calendars/primary/events")[-1]
    assert req.method == "POST"
    assert req.headers["authorization"] == "Bearer AT-1"
    q = parse_qs(urlparse(str(req.url)).query)
    assert q["conferenceDataVersion"] == ["1"] and q["sendUpdates"] == ["all"]
    body = json.loads(req.content)
    assert body["summary"] == "Sync" and body["description"] == "agenda"
    assert body["start"] == {"dateTime": "2026-10-20T14:00:00+07:00", "timeZone": "Asia/Bangkok"}
    assert body["end"]["dateTime"] == "2026-10-20T15:00:00+07:00"
    assert body["attendees"] == [{"email": "a@example.com"}, {"email": "b@example.com"}]
    cr = body["conferenceData"]["createRequest"]
    assert cr["conferenceSolutionKey"] == {"type": "hangoutsMeet"} and len(cr["requestId"]) >= 16
    assert body["extendedProperties"]["private"] == {"salid": "1"}


def test_create_event_request_ids_differ(oauth, g):
    connect(oauth)
    g.calendar = lambda r: httpx.Response(200, json=google_event())
    cal = make_calendar(g, oauth)
    for _ in range(2):
        run(cal.create_event_full(ACCT, EventSpec("x", START, END)))
    ids = [
        json.loads(r.content)["conferenceData"]["createRequest"]["requestId"]
        for r in g.of("/calendars/primary/events")
    ]
    assert ids[0] != ids[1]


def test_naive_times_read_in_calendar_zone_and_utc_converted(oauth, g):
    connect(oauth)
    g.calendar = lambda r: httpx.Response(200, json=google_event())
    cal = make_calendar(g, oauth)
    run(
        cal.create_event_full(
            ACCT,
            EventSpec("x", datetime(2026, 10, 20, 9, 0), datetime(2026, 10, 20, 7, 0, tzinfo=UTC)),
        )
    )
    body = json.loads(g.of("/calendars/primary/events")[-1].content)
    assert body["start"]["dateTime"] == "2026-10-20T09:00:00+07:00"
    assert body["end"]["dateTime"] == "2026-10-20T14:00:00+07:00"


def test_meet_link_from_entry_points_when_no_hangout_link(oauth, g):
    connect(oauth)
    ev = google_event()
    del ev["hangoutLink"]
    ev["conferenceData"] = {
        "entryPoints": [
            {"entryPointType": "phone", "uri": "tel:+1"},
            {"entryPointType": "video", "uri": "https://meet.google.com/zzz"},
        ]
    }
    g.calendar = lambda r: httpx.Response(200, json=ev)
    cal = make_calendar(g, oauth)
    assert run(cal.create_event_full(ACCT, EventSpec("x", START, END))).meet_url.endswith("/zzz")


def test_protocol_create_event_returns_id_and_url(oauth, g):
    connect(oauth)
    g.calendar = lambda r: httpx.Response(200, json=google_event())
    cal = make_calendar(g, oauth)
    assert run(cal.create_event(ACCT, EventSpec("x", START, END))) == (
        "ev1",
        "https://calendar.google.com/event?eid=1",
    )


def test_update_event_moves_with_send_updates(oauth, g):
    connect(oauth)
    g.calendar = lambda r: httpx.Response(200, json=google_event())
    cal = make_calendar(g, oauth)
    run(cal.update_event(ACCT, "ev/1", START + timedelta(days=1), END + timedelta(days=1)))
    req = g.of("/calendars/primary/events")[-1]
    assert req.method == "PATCH" and str(req.url).split("?")[0].endswith("/events/ev%2F1")
    assert parse_qs(urlparse(str(req.url)).query)["sendUpdates"] == ["all"]
    assert json.loads(req.content)["start"]["dateTime"] == "2026-10-21T14:00:00+07:00"


def test_delete_event_notifies_attendees_and_tolerates_gone(oauth, g):
    connect(oauth)
    g.calendar = lambda r: httpx.Response(204)
    cal = make_calendar(g, oauth)
    run(cal.delete_event(ACCT, "ev1"))
    req = g.of("/calendars/primary/events")[-1]
    assert req.method == "DELETE"
    assert parse_qs(urlparse(str(req.url)).query)["sendUpdates"] == ["all"]
    g.calendar = lambda r: httpx.Response(410)
    run(cal.delete_event(ACCT, "ev1"))  # already gone: no error


def test_errors_carry_status_only(oauth, g):
    connect(oauth)
    g.calendar = lambda r: httpx.Response(
        500, json={"error": {"message": "secret body AT-1 leaked?"}}
    )
    cal = make_calendar(g, oauth)
    with pytest.raises(GoogleCalendarError) as e:
        run(cal.create_event_full(ACCT, EventSpec("x", START, END)))
    assert e.value.status == 500 and e.value.code == "failed"
    assert "secret" not in str(e.value) and "AT-1" not in str(e.value)
    g.calendar = lambda r: httpx.Response(403)
    with pytest.raises(GoogleCalendarError) as e:
        run(cal.update_event(ACCT, "e", START, END))
    assert e.value.code == "unauthorized"
    g.calendar = lambda r: httpx.Response(404)
    with pytest.raises(GoogleCalendarError) as e:
        run(cal.get_event(ACCT, "e"))
    assert e.value.code == "not_found"


def test_not_connected_is_typed(oauth, g):
    cal = make_calendar(g, oauth)
    with pytest.raises(GoogleCalendarError) as e:
        run(cal.create_event_full(ACCT, EventSpec("x", START, END)))
    assert e.value.code == "not_connected"
    assert g.of("/calendars") == []


def test_subscribers_map_response_status(oauth, g):
    connect(oauth)
    g.calendar = lambda r: httpx.Response(
        200,
        json=google_event(
            attendees=[
                {"email": "A@example.com", "responseStatus": "accepted"},
                {"email": "b@example.com", "responseStatus": "tentative"},
                {"email": "c@example.com", "responseStatus": "declined"},
                {"email": "d@example.com", "responseStatus": "needsAction"},
                {"email": ME, "responseStatus": "accepted", "self": True},
                {"email": "room@resource.calendar.google.com", "resource": True},
            ]
        ),
    )
    cal = make_calendar(g, oauth)
    assert run(cal.attendee_status(ACCT, "ev1")) == {
        "a@example.com": "accepted",
        "b@example.com": "tentative",
        "c@example.com": "declined",
        "d@example.com": "needs_action",
    }
    assert run(cal.subscribers(ACCT, "ev1")) == ["a@example.com", "b@example.com"]
    g.calendar = lambda r: httpx.Response(500)
    assert run(cal.subscribers(ACCT, "ev1")) == []


def test_free_busy_query_and_unreadable_calendars(oauth, g):
    connect(oauth)
    g.calendar = lambda r: httpx.Response(
        200,
        json={
            "calendars": {
                "a@example.com": {
                    "busy": [{"start": "2026-10-20T07:00:00Z", "end": "2026-10-20T08:00:00Z"}]
                },
                "b@example.com": {"busy": []},
                "c@example.com": {"errors": [{"reason": "notFound"}], "busy": []},
            }
        },
    )
    cal = make_calendar(g, oauth)
    out = run(
        cal.free_busy(
            ACCT, ["a@example.com", "B@example.com", "c@example.com", "d@example.com"], START, END
        )
    )
    req = g.of("/freeBusy")[-1]
    body = json.loads(req.content)
    assert body["items"] == [
        {"id": "a@example.com"},
        {"id": "b@example.com"},
        {"id": "c@example.com"},
        {"id": "d@example.com"},
    ]
    assert body["timeMin"] == "2026-10-20T14:00:00+07:00" and body["timeZone"] == "Asia/Bangkok"
    assert out["a@example.com"] == [
        (datetime(2026, 10, 20, 7, 0, tzinfo=UTC), datetime(2026, 10, 20, 8, 0, tzinfo=UTC))
    ]
    assert out["b@example.com"] == []  # readable and free
    assert out["c@example.com"] is None  # not visible to this token: NOT "free"
    assert out["d@example.com"] is None  # absent from the answer
    with pytest.raises(GoogleCalendarError):
        run(cal.free_busy(ACCT, [f"u{i}@example.com" for i in range(51)], START, END))


# ------------------------------------------------------------------ agent tools
def org():
    return Org(
        id=ORG,
        name="Cafe",
        owner_email=ME,
        created_at="2026-01-01",
        members=[
            Member(email=ME, role="owner"),
            Member(email="anna@example.com", role="member"),
            Member(email="annabel@example.com", role="member"),
            Member(email="bob@example.com", role="member"),
        ],
    )


def tools_for(oauth, g, state, *, rid, email=ME, is_exec=True, now=None):
    cal = make_calendar(g, oauth)
    ctx = CalendarToolContext(
        org=org(),
        email=email,
        is_exec=is_exec,
        calendar=cal,
        oauth=oauth,
        state=state,
        run=run,
        zone=BKK,
        request_id=rid,
        now=lambda: now or datetime(2026, 10, 18, 5, 0, tzinfo=UTC),
    )
    return {f.__name__: f for f in calendar_tools(ctx)}


def sched(t, **kw):
    args = dict(
        title="Sync",
        start="2026-10-20T14:00:00",
        end="2026-10-20T15:00:00",
        attendees=["bob@example.com"],
        description="",
    )
    return t["schedule_google_meeting"](**{**args, **kw})


def creates(g):
    return [r for r in g.of("/calendars/primary/events") if r.method == "POST"]


def test_attendee_resolution_rules():
    o = org()
    assert resolve_attendees(o, ["BOB@example.com", "bob"], False) == (["bob@example.com"], [])
    assert resolve_attendees(o, ["anna"], False) == (["anna@example.com"], [])  # exact local part
    assert resolve_attendees(o, ["ann"], False) == ([], ["ann"])  # ambiguous prefix
    assert resolve_attendees(o, ["bo"], False) == ([], ["bo"])  # boss / bob: ambiguous
    assert resolve_attendees(o, ["bos"], False) == ([ME], [])  # unique prefix
    assert resolve_attendees(o, ["x@outside.com", "nobody"], False) == (
        [],
        ["x@outside.com", "nobody"],
    )
    assert resolve_attendees(o, ["x@outside.com", "nobody"], True) == (
        ["x@outside.com"],
        ["nobody"],
    )


def test_schedule_needs_confirmation_then_creates_in_later_request(oauth, g):
    connect(oauth)
    g.calendar = lambda r: httpx.Response(200, json=google_event())
    state = SecretaryState()
    draft = sched(tools_for(oauth, g, state, rid="r1"))
    assert draft["status"] == "needs_confirmation"
    assert draft["proposal"]["attendees"] == ["bob@example.com"]
    assert creates(g) == []
    # the model "confirming" in the SAME request must not create
    same = sched(tools_for(oauth, g, state, rid="r1"), confirm=True)
    assert same["status"] == "needs_confirmation" and creates(g) == []
    done = sched(tools_for(oauth, g, state, rid="r2"), confirm=True)
    assert done["status"] == "scheduled"
    assert done["meet_url"] == "https://meet.google.com/abc-defg-hij"
    assert done["event_id"] == "ev1"
    assert len(creates(g)) == 1
    # consumed: a repeat needs a new draft and another round
    again = sched(tools_for(oauth, g, state, rid="r3"), confirm=True)
    assert again["status"] == "needs_confirmation" and len(creates(g)) == 1


def test_confirm_requires_identical_arguments(oauth, g):
    connect(oauth)
    g.calendar = lambda r: httpx.Response(200, json=google_event())
    state = SecretaryState()
    sched(tools_for(oauth, g, state, rid="r1"))
    other = sched(tools_for(oauth, g, state, rid="r2"), confirm=True, title="Different")
    assert other["status"] == "needs_confirmation" and creates(g) == []


def test_draft_again_in_later_request_keeps_proposal_alive(oauth, g):
    connect(oauth)
    g.calendar = lambda r: httpx.Response(200, json=google_event())
    state = SecretaryState()
    sched(tools_for(oauth, g, state, rid="r1"))
    assert sched(tools_for(oauth, g, state, rid="r2"))["status"] == "needs_confirmation"
    assert sched(tools_for(oauth, g, state, rid="r3"), confirm=True)["status"] == "scheduled"


def test_unknown_attendee_refused_for_members_allowed_for_execs(oauth, g):
    connect(oauth)
    g.calendar = lambda r: httpx.Response(200, json=google_event())
    state = SecretaryState()
    r = sched(
        tools_for(oauth, g, state, rid="r1", is_exec=False), attendees=["stranger@outside.com"]
    )
    assert r["error"] == "unknown_attendees" and r["unresolved"] == ["stranger@outside.com"]
    ok = sched(
        tools_for(oauth, g, state, rid="r1", is_exec=True), attendees=["stranger@outside.com"]
    )
    assert ok["status"] == "needs_confirmation"


def test_not_connected_gives_instruction_and_never_calls_calendar(docs, g):
    o = GoogleOAuth(docs, cfg(), client=httpx.Client(transport=httpx.MockTransport(g)))
    r = sched(tools_for(o, g, SecretaryState(), rid="r1"))
    assert r["status"] == "not_connected" and "Connect Google Calendar" in r["instruction"]
    assert g.of("/calendars") == []


def test_expired_connection_during_create_reports_not_connected(oauth, g, clock):
    connect(oauth)
    state = SecretaryState()
    sched(tools_for(oauth, g, state, rid="r1"))
    clock.m += 4000
    g.token_status, g.token_error = 400, "invalid_grant"
    r = sched(tools_for(oauth, g, state, rid="r2"), confirm=True)
    assert r["status"] == "not_connected" and not oauth.is_connected(ORG, ME)


def test_input_validation(oauth, g):
    connect(oauth)
    t = tools_for(oauth, g, SecretaryState(), rid="r1")
    assert "error" in sched(t, end="2026-10-20T13:00:00")
    assert "error" in sched(t, end="2026-10-21T14:00:01")  # over 12 hours
    assert "error" in sched(t, start="tomorrow")
    assert "error" in sched(t, title="  ")
    assert "error" in sched(t, start="2026-10-01T10:00:00", end="2026-10-01T11:00:00")  # past
    assert "error" in sched(t, attendees=[f"u{i}@x.com" for i in range(30)])


def test_cancel_two_step_and_notifies(oauth, g):
    connect(oauth)

    def cal(r):
        if r.method == "GET":
            return httpx.Response(200, json=google_event())
        return httpx.Response(204)

    g.calendar = cal
    state = SecretaryState()
    p = tools_for(oauth, g, state, rid="r1")["cancel_google_meeting"](event_id="ev1")
    assert p["status"] == "needs_confirmation" and p["proposal"]["title"] == "Sync"
    assert not [r for r in g.log if r.method == "DELETE"]
    same = tools_for(oauth, g, state, rid="r1")["cancel_google_meeting"]("ev1", confirm=True)
    assert same["status"] == "needs_confirmation"
    done = tools_for(oauth, g, state, rid="r2")["cancel_google_meeting"]("ev1", confirm=True)
    assert done["status"] == "cancelled"
    d = [r for r in g.log if r.method == "DELETE"]
    assert len(d) == 1 and parse_qs(urlparse(str(d[0].url)).query)["sendUpdates"] == ["all"]


def test_cancel_refuses_events_salid_did_not_create(oauth, g):
    connect(oauth)
    g.calendar = lambda r: httpx.Response(200, json=google_event(extendedProperties={}))
    state = SecretaryState()
    t = tools_for(oauth, g, state, rid="r1")["cancel_google_meeting"]
    assert t("ev1")["error"] == "not_created_by_salid"
    assert (
        tools_for(oauth, g, state, rid="r2")["cancel_google_meeting"]("ev1", confirm=True)["error"]
        == "not_created_by_salid"
    )
    assert not [r for r in g.log if r.method == "DELETE"]


def test_tools_bound_to_asker_not_arguments(oauth, g):
    """The token used is the asker's; there is no tool argument that names another account."""
    import inspect

    connect(oauth)
    t = tools_for(oauth, g, SecretaryState(), rid="r1")
    for fn in t.values():
        assert not {"org", "org_id", "email", "organizer"} & set(inspect.signature(fn).parameters)
    other = tools_for(oauth, g, SecretaryState(), rid="r1", email="bob@example.com")
    assert sched(other)["status"] == "not_connected"  # bob never connected; mine is not used
