"""/center serves center.html with its placeholders filled; /api/center/me with the auth overridden."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from meeting_scribe import center, hub
from meeting_scribe.app import app
from meeting_scribe.config import get_settings
from salid.coo_store import SQLiteDocStore
from salid.orgs import OrgStore


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("HUB_GOOGLE_CLIENT_ID", "123456-abc.apps.googleusercontent.com")
    monkeypatch.setenv("SALID_DISCORD_CLIENT_ID", "123456789012345678")
    get_settings.cache_clear()
    orgs = OrgStore(SQLiteDocStore(":memory:"))
    app.dependency_overrides[hub.require_user] = lambda: "boss@example.com"
    app.dependency_overrides[center.get_orgs] = lambda: orgs
    yield TestClient(app)
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def test_center_page_has_its_placeholders_replaced(client):
    r = client.get("/center")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "123456-abc.apps.googleusercontent.com" in r.text
    assert "__HUB_GOOGLE_CLIENT_ID__" not in r.text and "__INVITE_URL__" not in r.text
    assert "client_id=123456789012345678" in r.text  # the Add-Salid-to-Discord link


def test_center_me_with_overridden_auth(client):
    r = client.get("/api/center/me")
    assert r.status_code == 200
    body = r.json()
    assert body["email"] == "boss@example.com" and body["orgs"] == []


def test_center_page_sets_the_monitor_flag(client, monkeypatch):
    monkeypatch.setenv("MONITOR_ENABLED", "true")
    get_settings.cache_clear()
    assert 'const MONITOR = "1" === "1"' in client.get("/center").text
    monkeypatch.delenv("MONITOR_ENABLED")
    get_settings.cache_clear()
    page = client.get("/center").text
    assert 'const MONITOR = "" === "1"' in page and "__MONITOR__" not in page
