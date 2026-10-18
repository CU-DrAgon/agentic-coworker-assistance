"""Google Calendar routes of the command center: authz, feature switch, callback."""

import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from meeting_scribe import center, hub
from meeting_scribe.app import app
from meeting_scribe.config import get_settings
from salid.coo_store import SQLiteDocStore
from salid.google_oauth import CALENDAR_SCOPES, STATE_KIND, TOKENS_KIND, GoogleConfig, GoogleOAuth
from salid.orgs import License, Member, OrgStore

CLIENT_ID = "123-abc.apps.googleusercontent.com"
KEY = Fernet.generate_key().decode()
TOKENS = {
    n: {"email": f"{n}@example.com", "email_verified": True}
    for n in ("boss", "mem", "other", "root", "stranger")
}
H = {n: {"Authorization": f"Bearer {n}"} for n in TOKENS}
URL = "/api/center/orgs/org_t/google/calendar"


class World:
    pass


@pytest.fixture
def w(monkeypatch):
    monkeypatch.setenv("STORE", "memory")
    monkeypatch.setenv("HUB_GOOGLE_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("HUB_ADMIN_EMAILS", "root@example.com")
    get_settings.cache_clear()
    w = World()
    w.docs = SQLiteDocStore(":memory:")
    w.orgs = OrgStore(w.docs)
    w.orgs.create(
        "Cafe",
        "boss@example.com",
        org_id="org_t",
        members=[
            Member(email="boss@example.com", role="owner"),
            Member(email="mem@example.com", role="member"),
        ],
        license=License(plan="monthly", status="active", seats=5, valid_until="2099-01-01"),
    )
    w.orgs.create("Other Co", "other@example.com", org_id="org_o")
    w.on = True
    w.userinfo = {"email": "mem@example.com", "email_verified": True}
    w.log = []

    def google(req):
        w.log.append(req)
        url = str(req.url)
        if "oauth2.googleapis.com/token" in url:
            return httpx.Response(
                200,
                json={
                    "access_token": "AT",
                    "refresh_token": "RT-route-secret",
                    "expires_in": 3600,
                    "scope": "openid email " + " ".join(CALENDAR_SCOPES),
                },
            )
        if "userinfo" in url:
            return httpx.Response(200, json=w.userinfo)
        return httpx.Response(200)

    def oauth():
        cfg = (
            GoogleConfig("cid", "csec", "https://salid.example/cb", KEY) if w.on else GoogleConfig()
        )
        return GoogleOAuth(w.docs, cfg, client=httpx.Client(transport=httpx.MockTransport(google)))

    def verifier():
        def verify(token, client_id):
            if token not in TOKENS:
                raise ValueError("bad token")
            return TOKENS[token]

        return verify

    app.dependency_overrides[hub.google_token_verifier] = verifier
    app.dependency_overrides[center.get_orgs] = lambda: w.orgs
    app.dependency_overrides[center.get_google_oauth] = oauth
    w.client = TestClient(app, follow_redirects=False)
    yield w
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def connect_flow(w, who="mem"):
    r = w.client.post(f"{URL}/connect", headers=H[who])
    assert r.status_code == 200, r.text
    q = parse_qs(urlparse(r.json()["auth_url"]).query)
    return q["state"][0]


def callback(w, state, code="c1", **extra):
    return w.client.get(
        "/api/center/connectors/google/callback", params={"code": code, "state": state, **extra}
    )


def test_status_default_not_connected(w):
    r = w.client.get(URL, headers=H["mem"])
    assert r.status_code == 200
    assert r.json() == {"connected": False, "email": None, "connected_at": None}


def test_connect_returns_google_consent_url(w):
    r = w.client.post(f"{URL}/connect", headers=H["mem"])
    u = urlparse(r.json()["auth_url"])
    assert u.netloc == "accounts.google.com"
    q = parse_qs(u.query)
    assert q["code_challenge_method"] == ["S256"] and q["access_type"] == ["offline"]
    assert w.docs.get(STATE_KIND, 0, q["state"][0])["email"] == "mem@example.com"


def test_full_flow_connect_status_disconnect(w):
    state = connect_flow(w)
    r = callback(w, state)
    assert r.status_code == 303
    loc = urlparse(r.headers["location"])
    assert loc.path == "/center" and parse_qs(loc.query) == {
        "calendar": ["connected"],
        "org": ["org_t"],
    }
    st = w.client.get(URL, headers=H["mem"]).json()
    assert st["connected"] is True and st["email"] == "mem@example.com"
    # the boss does not see mem's connection
    assert w.client.get(URL, headers=H["boss"]).json()["connected"] is False
    # no secret anywhere in the API answers or the store
    assert "RT-route-secret" not in json.dumps(st)
    assert "RT-route-secret" not in json.dumps(w.docs.find_items(TOKENS_KIND, 0))
    d = w.client.delete(URL, headers=H["mem"])
    assert d.status_code == 200 and d.json()["connected"] is False
    assert w.client.get(URL, headers=H["mem"]).json()["connected"] is False
    assert any("/revoke" in str(r.url) for r in w.log)


def test_callback_state_single_use(w):
    state = connect_flow(w)
    assert "calendar=connected" in callback(w, state).headers["location"]
    r = callback(w, state)
    q = parse_qs(urlparse(r.headers["location"]).query)
    assert q == {"calendar": ["error"], "reason": ["bad_state"]}


def test_callback_email_mismatch_is_an_error_and_stores_nothing(w):
    w.userinfo = {"email": "someone.else@gmail.com", "email_verified": True}
    r = callback(w, connect_flow(w))
    assert parse_qs(urlparse(r.headers["location"]).query)["reason"] == ["email_mismatch"]
    assert w.docs.find_items(TOKENS_KIND, 0) == []


def test_callback_user_denied_consumes_state(w):
    state = connect_flow(w)
    r = w.client.get(
        "/api/center/connectors/google/callback", params={"error": "access_denied", "state": state}
    )
    assert parse_qs(urlparse(r.headers["location"]).query)["reason"] == ["denied"]
    assert w.docs.get(STATE_KIND, 0, state) is None


def test_callback_for_member_removed_meanwhile_revokes(w):
    state = connect_flow(w)
    org = w.orgs.get("org_t")
    org.members = [m for m in org.members if m.email != "mem@example.com"]
    w.orgs.save(org)
    r = callback(w, state)
    assert parse_qs(urlparse(r.headers["location"]).query)["reason"] == ["forbidden"]
    assert w.docs.find_items(TOKENS_KIND, 0) == []


def test_requires_login(w):
    assert w.client.get(URL).status_code == 401
    assert w.client.post(f"{URL}/connect").status_code == 401
    assert w.client.delete(URL).status_code == 401


def test_non_member_and_cross_org_refused(w):
    for fn, path in (
        (w.client.get, URL),
        (w.client.post, f"{URL}/connect"),
        (w.client.delete, URL),
    ):
        assert fn(path, headers=H["stranger"]).status_code == 403
        assert fn(path, headers=H["other"]).status_code == 403  # member of another org only


def test_super_admin_who_is_not_a_member_cannot_connect(w):
    assert w.client.get(URL, headers=H["root"]).status_code == 403
    assert w.client.post(f"{URL}/connect", headers=H["root"]).status_code == 403


def test_feature_off_routes_503_and_callback_error(w):
    w.on = False
    assert w.client.get(URL, headers=H["mem"]).status_code == 503
    assert w.client.post(f"{URL}/connect", headers=H["mem"]).status_code == 503
    assert w.client.delete(URL, headers=H["mem"]).status_code == 503
    r = callback(w, "whatever")
    assert parse_qs(urlparse(r.headers["location"]).query)["reason"] == ["not_available"]


def test_google_callback_not_swallowed_by_generic_connector_route(w):
    """`/connectors/{platform}/callback` is registered earlier; the Google route must win."""
    state = connect_flow(w)
    assert "calendar=connected" in callback(w, state).headers["location"]
