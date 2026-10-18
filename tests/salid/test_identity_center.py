# ruff: noqa: F811, F401
"""Command center API for verified identities: link codes, own accounts, unlink, attach codes."""

import pytest
from test_center_api import H, docs, w  # noqa: F401  (fixtures)

from salid.identity_links import MAX_LIVE_CODES, IdentityLinks
from salid.orgs import Workspace

BASE = "/api/center/orgs/org_t"


def code_url(platform):
    return f"{BASE}/identities/{platform}/code"


def test_member_gets_a_code_and_the_command_to_send(w):
    r = w.client.post(code_url("line"), headers=H["mem"])
    assert r.status_code == 200
    body = r.json()
    assert len(body["code"]) == 9 and body["code"][4] == "-"
    assert body["platform"] == "line" and body["kind"] == "identity"
    assert body["command"] == f"link {body['code']}" and body["expires_at"]


def test_redeeming_the_code_shows_up_in_my_identities_and_can_be_unlinked(w):
    assert w.client.get(f"{BASE}/identities", headers=H["mem"]).json() == []
    code = w.client.post(code_url("telegram"), headers=H["mem"]).json()["code"]
    IdentityLinks(w.orgs).redeem(code, "telegram", "8629001234", "bot1")
    got = w.client.get(f"{BASE}/identities", headers=H["mem"]).json()
    assert got == [{"platform": "telegram", "account_hint": "1234"}]
    # a colleague sees only their own
    assert w.client.get(f"{BASE}/identities", headers=H["adm"]).json() == []
    r = w.client.delete(f"{BASE}/identities/telegram", headers=H["mem"])
    assert r.status_code == 200 and r.json() == []
    assert w.orgs.member_by_identity("org_t", "telegram", "8629001234") is None
    assert w.client.delete(f"{BASE}/identities/telegram", headers=H["mem"]).status_code == 404


def test_the_discord_link_is_listed_but_not_managed_here(w):
    org = w.orgs.get("org_t")
    org.members[2].discord_user_id = "98765"
    w.orgs.save(org)
    got = w.client.get(f"{BASE}/identities", headers=H["mem"]).json()
    assert got == [{"platform": "discord", "account_hint": "8765"}]
    assert w.client.delete(f"{BASE}/identities/discord", headers=H["mem"]).status_code == 400
    assert w.client.post(code_url("discord"), headers=H["mem"]).status_code == 400


def test_unknown_platform_is_400(w):
    assert w.client.post(code_url("fax"), headers=H["mem"]).status_code == 400
    assert w.client.delete(f"{BASE}/identities/fax", headers=H["mem"]).status_code == 400


def test_sixth_live_code_is_429(w):
    for _ in range(MAX_LIVE_CODES):
        assert w.client.post(code_url("line"), headers=H["mem"]).status_code == 200
    assert w.client.post(code_url("line"), headers=H["mem"]).status_code == 429
    assert w.client.post(code_url("line"), headers=H["adm"]).status_code == 200


def test_outsiders_and_anonymous_are_refused(w):
    assert w.client.post(code_url("line"), headers=H["stranger"]).status_code == 403
    assert w.client.get(f"{BASE}/identities", headers=H["stranger"]).status_code == 403
    assert w.client.delete(f"{BASE}/identities/line", headers=H["other"]).status_code == 403
    assert w.client.post(code_url("line")).status_code == 401
    assert w.client.get(f"{BASE}/identities").status_code == 401


def test_super_admin_is_not_a_member_so_gets_no_link_code(w):
    assert w.client.post(code_url("line"), headers=H["root"]).status_code == 403
    assert w.client.get(f"{BASE}/identities", headers=H["root"]).json() == []


def test_admin_creates_an_attach_code_that_attaches_the_bot(w):
    url = f"{BASE}/workspaces/whatsapp/attach-code"
    r = w.client.post(url, headers=H["adm"])
    assert r.status_code == 200 and r.json()["kind"] == "attach"
    code = r.json()["code"]
    IdentityLinks(w.orgs).redeem(code, "whatsapp", "66800000000", "1555")
    listed = w.client.get(f"{BASE}/workspaces", headers=H["mem"]).json()
    ws = next(x for x in listed if x["platform"] == "whatsapp")
    assert ws["external_id"] == "1555" and ws["id"] == "whatsapp-1555"


def test_attach_code_is_admin_only_and_org_scoped(w):
    url = f"{BASE}/workspaces/line/attach-code"
    assert w.client.post(url, headers=H["mem"]).status_code == 403
    assert w.client.post(url, headers=H["stranger"]).status_code == 403
    assert w.client.post(url, headers=H["boss"]).status_code == 200
    assert w.client.post(url, headers=H["root"]).status_code == 200  # super admin
    assert (
        w.client.post(f"{BASE}/workspaces/discord/attach-code", headers=H["boss"]).status_code
        == 400
    )
    assert (
        w.client.post(
            "/api/center/orgs/org_o/workspaces/line/attach-code", headers=H["boss"]
        ).status_code
        == 403
    )


def test_attach_code_cap_is_429(w):
    url = f"{BASE}/workspaces/telegram/attach-code"
    for _ in range(MAX_LIVE_CODES):
        assert w.client.post(url, headers=H["adm"]).status_code == 200
    assert w.client.post(url, headers=H["adm"]).status_code == 429


def test_member_list_does_not_leak_platform_ids(w):
    w.orgs.link_identity("org_t", "mem@example.com", "line", "U-secret-line-id")
    members = w.client.get(f"{BASE}/members", headers=H["adm"])
    org = w.client.get(BASE, headers=H["mem"])
    assert members.status_code == 200 and org.status_code == 200
    assert "U-secret-line-id" not in members.text + org.text
