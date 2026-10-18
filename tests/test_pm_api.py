"""PM dashboard API (`/api/center/orgs/{org}/pm/*`, `/pm`): every endpoint with fake Google tokens,
a SQLite / fake-Firestore doc store, a scripted risk assessor and a fixed clock. No network."""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent / "salid"))

from fake_firestore import FakeFirestore  # noqa: E402

from meeting_scribe import center, hub, pm_api  # noqa: E402
from meeting_scribe.app import app  # noqa: E402
from meeting_scribe.config import get_settings  # noqa: E402
from salid.coo_store import FirestoreDocStore, SQLiteDocStore  # noqa: E402
from salid.guild_settings import MemoryGuildStore  # noqa: E402
from salid.orgs import Member, OrgStore, Workspace  # noqa: E402
from salid.tasks import Task, TaskRepo  # noqa: E402

CLIENT_ID = "123-abc.apps.googleusercontent.com"
GUILD = 1000
NOW = datetime(2026, 10, 14, 12, 0, tzinfo=ZoneInfo("Asia/Bangkok"))
NAMES = ("boss", "adm", "mem", "peat", "fin", "other", "root", "stranger")
TOKENS = {n: {"email": f"{n}@example.com", "email_verified": True} for n in NAMES}
H = {n: {"Authorization": f"Bearer {n}"} for n in NAMES}


def person(name, discord=None):
    return {"email": f"{name}@example.com", "discord_user_id": discord, "name": name}


@pytest.fixture(params=["sqlite", "firestore"])
def docs(request):
    if request.param == "sqlite":
        return SQLiteDocStore(":memory:")
    return FirestoreDocStore(client=FakeFirestore())


class World:
    assessed: list
    assessor_reply: object


@pytest.fixture
def w(docs, monkeypatch):
    monkeypatch.setenv("HUB_GOOGLE_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("HUB_ADMIN_EMAILS", "root@example.com")
    get_settings.cache_clear()
    monkeypatch.setattr(pm_api, "now", lambda: NOW)
    w = World()
    w.docs = docs
    w.orgs = OrgStore(docs)
    org = w.orgs.create(
        "Acme",
        "boss@example.com",
        members=[
            Member(email="boss@example.com", role="owner"),
            Member(email="adm@example.com", role="admin"),
            Member(email="mem@example.com", discord_user_id="26", department="tech"),
            Member(email="peat@example.com", department="business"),
            Member(email="fin@example.com", finance=True, department="finance"),
            Member(email="other@example.com"),
        ],
    )
    w.orgs.attach_workspace(
        org.id, Workspace(id=f"discord-{GUILD}", platform="discord", external_id=str(GUILD))
    )
    w.org_id = org.id
    w.base = f"/api/center/orgs/{org.id}/pm"
    w.assessed, w.assessor_reply = [], {"risks": []}

    def assessor():
        def run(prompt):
            w.assessed.append(prompt)
            if isinstance(w.assessor_reply, Exception):
                raise w.assessor_reply
            return (
                w.assessor_reply
                if isinstance(w.assessor_reply, str)
                else json.dumps(w.assessor_reply)
            )

        return run

    def verifier():
        def verify(token, client_id):
            if token not in TOKENS:
                raise ValueError("bad token")
            return TOKENS[token]

        return verify

    app.dependency_overrides[hub.google_token_verifier] = verifier
    app.dependency_overrides[center.get_orgs] = lambda: w.orgs
    app.dependency_overrides[pm_api.get_assessor] = assessor
    w.guilds = MemoryGuildStore()
    app.dependency_overrides[pm_api.get_pm_guilds] = lambda: w.guilds
    w.c = TestClient(app)
    w.tasks = TaskRepo(docs)
    yield w
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def call(w, who, method, path, body=None, **params):
    return w.c.request(method, w.base + path, headers=H[who], json=body, params=params or None)


def make(w, who="mem", **kw):
    body = {
        "name": "Alpha", "department": "tech", "owner": person("mem", "26"), "priority": "medium",
        **kw,
    }  # fmt: skip
    r = call(w, who, "POST", "/projects", body)
    assert r.status_code == 201, r.text
    return r.json()


# ------------------------------------------------------------------ page and access
def test_only_org_members_get_in(w):
    assert call(w, "stranger", "GET", "/home").status_code == 403
    assert w.c.get(w.base + "/home").status_code == 401
    assert w.c.get("/api/center/orgs/nope/pm/home", headers=H["mem"]).status_code == 403
    assert call(w, "root", "GET", "/home").status_code == 200  # super admin counts as admin
    assert call(w, "other", "GET", "/home").status_code == 200


# ------------------------------------------------------------------ create / list / get
def test_create_standard_project(w):
    p = make(w, target_outcome="Ship", deadline="2026-11-30")
    assert p["id"].startswith("p-") and p["workspace"] == f"discord-{GUILD}"
    assert p["status"] == "planned" and p["kind"] == "standard" and p["visibility"] == "team"
    assert p["owner"] == person("mem", "26") and p["created_by"]["email"] == "mem@example.com"
    assert (
        p["milestones"] == p["risks"] == p["links"] == p["changes"] == [] and p["hackathon"] is None
    )
    assert p["stats"]["tasks_total"] == 0 and p["stats"]["needs_review"] is False
    assert "budget" not in p  # mem is not finance
    adm = make(w, "adm", name="Admin's")
    assert adm["budget"] == {"currency": "THB", "planned": 0, "spent": 0, "notes": "", "travel": 0}


def test_create_hackathon_applies_its_template(w):
    p = make(w, name="Cup", kind="hackathon", status="in_progress")
    h = p["hackathon"]
    assert [d["title"] for d in h["deliverables"]] == ["GitHub repo", "Demo", "Video", "Pitch deck"]
    assert h["submission_status"] == "not_started" and h["contest_id"] is None


def test_create_fills_the_owner_from_the_org_and_validates(w):
    p = make(w, owner={"email": "PEAT@example.com"})
    assert p["owner"]["name"] == "peat" and p["owner"]["email"] == "peat@example.com"
    d = make(w, name="By discord", owner={"discord_user_id": "26"})["owner"]
    assert d["email"] == "mem@example.com" and d["discord_user_id"] == "26"
    for bad in (
        {"department": "sales"}, {"priority": "urgent"}, {"owner": {}}, {"name": " "},
        {"deadline": "soon"}, {"kind": "party"}, {"workspace": "nope"}, {"surprise": 1},
    ):  # fmt: skip
        body = {"name": "X", "department": "tech", "owner": person("mem"), "priority": "low", **bad}
        assert call(w, "mem", "POST", "/projects", body).status_code == 422, bad
    assert call(w, "mem", "POST", "/projects", {"name": "X"}).status_code == 422


def test_workspace_choice_web_or_discord(w):
    d = make(w)
    web = make(w, name="Webby", workspace="web")
    assert d["workspace"] == f"discord-{GUILD}" and web["workspace"] == "web"
    names = {x["name"] for x in call(w, "mem", "GET", "/projects").json()["projects"]}
    assert names == {"Alpha", "Webby"}  # lists run across workspaces
    assert call(w, "mem", "GET", f"/projects/{web['id']}").status_code == 200


def test_list_filters_and_cards(w):
    a = make(w, name="Alpha", department="tech", status="in_progress", target_outcome="web shop")
    make(w, name="Beta", department="business", owner=person("peat"))
    make(w, name="Cup", kind="hackathon", department="tech")

    def get(**q):
        return [p["name"] for p in call(w, "mem", "GET", "/projects", **q).json()["projects"]]

    assert get() == ["Alpha", "Beta", "Cup"]
    assert get(department="business") == ["Beta"] and get(status="in_progress") == ["Alpha"]
    assert get(kind="hackathon") == ["Cup"] and get(q="SHOP") == ["Alpha"]
    assert get(mine="1") == ["Alpha", "Cup"] and get(q="zzz") == []
    card = call(w, "mem", "GET", "/projects").json()["projects"][0]
    assert card["id"] == a["id"] and "overview" not in card and "milestones" not in card
    assert "budget" not in card and "changes" not in card and card["stats"]["tasks_total"] == 0
    cup = call(w, "mem", "GET", "/projects", kind="hackathon").json()["projects"][0]
    assert set(cup["hackathon"]) == {
        "submission_status",
        "official_deadline",
        "checklist_done",
        "checklist_total",
    }


def test_members_only_projects_are_hidden_from_non_members(w):
    p = make(w, visibility="members")
    assert [x["id"] for x in call(w, "mem", "GET", "/projects").json()["projects"]] == [p["id"]]
    assert call(w, "other", "GET", "/projects").json()["projects"] == []
    assert call(w, "other", "GET", f"/projects/{p['id']}").status_code == 403
    assert call(w, "adm", "GET", f"/projects/{p['id']}").status_code == 200
    assert (
        call(
            w,
            "mem",
            "PATCH",
            f"/projects/{p['id']}",
            {"members": [{"person": person("peat"), "role": "dev"}]},
        ).status_code
        == 200
    )
    assert call(w, "peat", "GET", f"/projects/{p['id']}").status_code == 200
    assert call(w, "other", "GET", "/insights").json()["progress"]["by_status"]["planned"] == 0
    assert call(w, "other", "GET", f"/projects/{p['id']}/tasks").status_code == 403
    assert call(w, "other", "GET", "/home").json()["counts"]["projects"] == 0


def test_get_project_has_tasks_and_unknown_is_404(w):
    p = make(w)
    t = call(w, "mem", "POST", f"/projects/{p['id']}/tasks", {"title": "do it"}).json()
    got = call(w, "other", "GET", f"/projects/{p['id']}").json()
    assert [x["id"] for x in got["tasks"]] == [t["id"]] and got["stats"]["tasks_total"] == 1
    assert got["stats"]["tasks_unassigned"] == 1
    assert call(w, "mem", "GET", "/projects/p-nope").status_code == 404


# ------------------------------------------------------------------ patch / delete
def test_patch_needs_a_reason_for_the_big_four(w):
    p = make(w, deadline="2026-11-30")
    url = f"/projects/{p['id']}"
    for body in (
        {"status": "in_progress"}, {"priority": "high"}, {"deadline": "2026-12-31"},
        {"owner": person("peat")},
    ):  # fmt: skip
        r = call(w, "mem", "PATCH", url, body)
        assert r.status_code == 422 and r.json() == {"detail": "reason_required"}, body
        r = call(w, "mem", "PATCH", url, {**body, "reason": "  "})
        assert r.status_code == 422
    r = call(
        w, "mem", "PATCH", url, {"status": "in_progress", "reason": "kickoff", "impact": "none"}
    )
    assert r.status_code == 200 and r.json()["status"] == "in_progress"
    ch = r.json()["changes"][-1]
    assert (ch["field"], ch["from"], ch["to"], ch["reason"], ch["impact"]) == (
        "status", "planned", "in_progress", "kickoff", "none",
    )  # fmt: skip
    assert ch["by"]["email"] == "mem@example.com" and ch["at"].endswith("+07:00")
    r = call(w, "mem", "PATCH", url, {"name": "Alpha 2", "tags": ["a"], "status": "in_progress"})
    assert r.status_code == 200 and r.json()["name"] == "Alpha 2"  # same status: no reason needed
    assert len(r.json()["changes"]) == 1
    r = call(w, "mem", "PATCH", url, {"owner": person("peat"), "reason": "handover"})
    assert r.json()["owner"]["email"] == "peat@example.com"


def test_patch_rights_and_validation(w):
    p = make(w)
    url = f"/projects/{p['id']}"
    assert (
        call(w, "other", "PATCH", url, {"overview": "x"}).status_code == 403
    )  # not on the project
    call(w, "mem", "PATCH", url, {"members": [{"person": person("peat"), "role": "dev"}]})
    assert call(w, "peat", "PATCH", url, {"overview": "# Plan"}).json()["overview"] == "# Plan"
    assert call(w, "peat", "PATCH", url, {"name": "Mine now"}).status_code == 403
    assert (
        call(
            w, "peat", "PATCH", url, {"overview": "x", "priority": "low", "reason": "r"}
        ).status_code
        == 403
    )
    assert call(w, "adm", "PATCH", url, {"priority": "high", "reason": "client"}).status_code == 200
    assert call(w, "adm", "PATCH", url, {"visibility": "secret"}).status_code == 422
    assert call(w, "adm", "PATCH", url, {"deadline": "tomorrow", "reason": "r"}).status_code == 422
    assert call(w, "adm", "PATCH", url, {"bogus": 1}).status_code == 422
    assert call(w, "adm", "PATCH", url, {}).status_code == 200  # nothing to change
    assert call(w, "adm", "PATCH", "/projects/p-nope", {"name": "x"}).status_code == 404


def test_delete_is_admin_only_and_soft(w, docs):
    p = make(w)
    url = f"/projects/{p['id']}"
    assert call(w, "mem", "DELETE", url).status_code == 403  # even the owner
    assert call(w, "adm", "DELETE", url).status_code == 204
    assert call(w, "mem", "GET", url).status_code == 404
    assert call(w, "mem", "GET", "/projects").json()["projects"] == []
    assert call(w, "adm", "DELETE", url).status_code == 404
    from salid.projects import ProjectRepo

    gone = ProjectRepo(docs).get(GUILD, p["id"], include_deleted=True)
    assert gone.deleted and gone.status == "cancelled"


# ------------------------------------------------------------------ finance
def test_finance_fields_are_stripped_for_everyone_but_finance_and_admins(w):
    p = make(w)
    assert call(w, "fin", "PATCH", f"/projects/{p['id']}/budget", {"planned": 5000, "spent": 1200.5, "travel": 300}).json() == {
        "currency": "THB", "planned": 5000, "spent": 1200.5, "notes": "", "travel": 300,
    }  # fmt: skip
    for who in ("mem", "peat", "other"):
        assert "budget" not in call(w, who, "GET", f"/projects/{p['id']}").json(), who
    assert call(w, "fin", "GET", f"/projects/{p['id']}").json()["budget"]["planned"] == 5000
    assert call(w, "adm", "GET", f"/projects/{p['id']}").json()["budget"]["spent"] == 1200.5
    assert "budget" not in call(w, "fin", "GET", "/projects").json()["projects"][0]  # cards: never
    assert call(w, "mem", "PATCH", f"/projects/{p['id']}/budget", {"planned": 1}).status_code == 403
    assert call(w, "mem", "GET", f"/projects/{p['id']}").json()["owner"]["name"] == "mem"
    assert (
        call(
            w, "adm", "PATCH", f"/projects/{p['id']}/budget", {"notes": "n", "currency": "USD"}
        ).json()["currency"]
        == "USD"
    )
    assert (
        call(w, "fin", "PATCH", f"/projects/{p['id']}/budget", {"planned": -1}).status_code == 422
    )


# ------------------------------------------------------------------ tasks
def test_task_create_and_patch_permissions(w):
    p = make(w)
    pid = p["id"]
    call(w, "mem", "PATCH", f"/projects/{pid}", {"members": [{"person": person("peat")}]})
    m = call(w, "mem", "POST", f"/projects/{pid}/milestones", {"title": "Demo"}).json()
    body = {
        "title": "Write",
        "assignee": person("peat"),
        "due": "2026-10-20",
        "milestone_id": m["id"],
        "note": "n",
    }
    r = call(w, "peat", "POST", f"/projects/{pid}/tasks", body)  # a project member may add
    assert r.status_code == 200
    t = r.json()
    assert t["project_id"] == pid and t["assignee"]["email"] == "peat@example.com"
    assert (t["due"], t["status"], t["milestone_id"], t["note"], t["created_from"]) == (
        "2026-10-20",
        "open",
        m["id"],
        "n",
        "manual",
    )
    assert set(t) == {
        "id",
        "project_id",
        "title",
        "assignee",
        "due",
        "no_deadline",
        "status",
        "note",
        "url",
        "milestone_id",
        "start",
        "step",
        "pct_done",
        "depends_on",
        "created_from",
        "updated_at",
        "confirm",
        "assigner",
    }
    assert t["confirm"] == "" and t["assigner"] == {"discord_user_id": None, "name": ""}
    assert call(w, "other", "POST", f"/projects/{pid}/tasks", {"title": "x"}).status_code == 403
    assert (
        call(
            w, "adm", "POST", f"/projects/{pid}/tasks", {"title": "x", "milestone_id": "m9"}
        ).status_code
        == 422
    )
    assert (
        call(
            w, "adm", "POST", f"/projects/{pid}/tasks", {"title": "x", "due": "friday"}
        ).status_code
        == 422
    )
    assert call(w, "adm", "POST", f"/projects/{pid}/tasks", {"title": ""}).status_code == 422
    url = f"/projects/{pid}/tasks/{t['id']}"
    assert (
        call(w, "peat", "PATCH", url, {"status": "done"}).json()["status"] == "done"
    )  # the assignee
    assert call(w, "other", "PATCH", url, {"note": "hax"}).status_code == 403
    assert call(w, "mem", "PATCH", url, {"assignee": None, "title": "Write more"}).json()[
        "assignee"
    ] == {
        "email": None,
        "discord_user_id": None,
        "name": "",
    }  # the owner may clear it
    saved = w.tasks.get(GUILD, t["id"])
    assert saved.title == "Write more" and saved.done_at and saved.assignee_id == 0
    assert call(w, "mem", "PATCH", f"/projects/{pid}/tasks/nope", {"note": "x"}).status_code == 404
    assert call(w, "mem", "PATCH", url, {"status": "doing"}).status_code == 422
    assert call(w, "mem", "PATCH", url, {"status": "open"}).json()["status"] == "open"
    assert w.tasks.get(GUILD, t["id"]).done_at == ""
    listed = call(w, "other", "GET", f"/projects/{pid}/tasks").json()["tasks"]
    assert [x["id"] for x in listed] == [t["id"]]
    assert call(w, "other", "GET", f"/projects/{pid}/tasks", status="done").json()["tasks"] == []


def test_a_task_of_another_project_is_not_reachable_through_this_one(w):
    a, b = make(w, name="A"), make(w, name="B")
    t = call(w, "mem", "POST", f"/projects/{a['id']}/tasks", {"title": "x"}).json()
    assert (
        call(w, "mem", "PATCH", f"/projects/{b['id']}/tasks/{t['id']}", {"note": "y"}).status_code
        == 404
    )


def test_discord_assignee_sets_the_discord_id(w):
    p = make(w)
    t = call(
        w,
        "mem",
        "POST",
        f"/projects/{p['id']}/tasks",
        {"title": "x", "assignee": {"discord_user_id": "26"}},
    ).json()
    assert t["assignee"] == {"email": "mem@example.com", "discord_user_id": "26", "name": "mem"}
    assert w.tasks.get(GUILD, t["id"]).assignee_id == 26


# ------------------------------------------------------------------ milestones / risks / links
def test_milestone_crud_and_task_unlink(w):
    pid = make(w)["id"]
    m1 = call(
        w,
        "mem",
        "POST",
        f"/projects/{pid}/milestones",
        {"title": "Alpha", "due": "2026-10-30", "owner": person("peat")},
    ).json()
    m2 = call(w, "mem", "POST", f"/projects/{pid}/milestones", {"title": "Beta"}).json()
    assert (m1["id"], m2["id"], m1["status"]) == ("m1", "m2", "pending") and m1["owner"][
        "email"
    ] == "peat@example.com"
    t = call(
        w, "mem", "POST", f"/projects/{pid}/tasks", {"title": "t", "milestone_id": "m1"}
    ).json()
    r = call(w, "peat", "PATCH", f"/projects/{pid}/milestones/m1", {"status": "done"})
    assert r.status_code == 403  # peat is not on the project yet
    call(w, "mem", "PATCH", f"/projects/{pid}", {"members": [{"person": person("peat")}]})
    r = call(w, "peat", "PATCH", f"/projects/{pid}/milestones/m1", {"status": "done"})
    assert r.json()["status"] == "done" and r.json()["title"] == "Alpha"
    assert (
        call(w, "mem", "PATCH", f"/projects/{pid}/milestones/m1", {"due": "oops"}).status_code
        == 422
    )
    assert (
        call(w, "mem", "PATCH", f"/projects/{pid}/milestones/m9", {"title": "x"}).status_code == 404
    )
    st = call(w, "mem", "GET", f"/projects/{pid}").json()["stats"]
    assert (st["milestones_done"], st["milestones_total"]) == (1, 2)
    assert call(w, "mem", "DELETE", f"/projects/{pid}/milestones/m1").status_code == 204
    assert w.tasks.get(GUILD, t["id"]).milestone_id == ""
    m3 = call(w, "mem", "POST", f"/projects/{pid}/milestones", {"title": "C"}).json()
    assert m3["id"] == "m3"  # ids are not reused
    assert call(w, "mem", "DELETE", f"/projects/{pid}/milestones/m1").status_code == 404


def test_risk_and_link_crud(w):
    pid = make(w)["id"]
    r = call(
        w,
        "mem",
        "POST",
        f"/projects/{pid}/risks",
        {"title": "Vendor late", "severity": "high", "note": "n"},
    ).json()
    assert (
        r["id"] == "r1"
        and r["status"] == "open"
        and r["source"] == "team"
        and r["created_at"].startswith("2026-10-14")
    )
    ai = call(
        w, "mem", "POST", f"/projects/{pid}/risks", {"title": "From AI", "source": "ai"}
    ).json()
    assert ai["source"] == "ai" and ai["severity"] == "medium"
    assert (
        call(w, "mem", "PATCH", f"/projects/{pid}/risks/r1", {"status": "mitigated"}).json()[
            "status"
        ]
        == "mitigated"
    )
    assert (
        call(w, "mem", "PATCH", f"/projects/{pid}/risks/r1", {"severity": "huge"}).status_code
        == 422
    )
    assert call(w, "other", "POST", f"/projects/{pid}/risks", {"title": "x"}).status_code == 403
    assert call(w, "mem", "DELETE", f"/projects/{pid}/risks/r2").status_code == 204
    link = call(
        w,
        "mem",
        "POST",
        f"/projects/{pid}/links",
        {"title": "Repo", "url": "https://github.com/x/y", "kind": "repo"},
    ).json()
    assert link["id"] == "l1" and link["kind"] == "repo"
    assert call(w, "mem", "POST", f"/projects/{pid}/links", {"url": "ftp://x"}).status_code == 422
    assert (
        call(w, "mem", "PATCH", f"/projects/{pid}/links/l1", {"title": "Code"}).json()["title"]
        == "Code"
    )
    assert call(w, "mem", "DELETE", f"/projects/{pid}/links/l1").status_code == 204
    got = call(w, "mem", "GET", f"/projects/{pid}").json()
    assert [x["id"] for x in got["risks"]] == ["r1"] and got["links"] == []


# ------------------------------------------------------------------ risk assessment
def test_assess_proposes_risks_without_saving_them(w):
    pid = make(w, target_outcome="Launch")["id"]
    call(w, "mem", "POST", f"/projects/{pid}/risks", {"title": "Known one"})
    w.assessor_reply = {"risks": [
        {"title": f"Risk {i}", "severity": "HIGH" if i == 0 else "weird", "note": "why"} for i in range(8)
    ] + [{"title": ""}, "junk"]}  # fmt: skip
    r = call(w, "mem", "POST", f"/projects/{pid}/risks/assess")
    assert r.status_code == 200
    out = r.json()["proposed"]
    assert len(out) == 5 and out[0]["severity"] == "high" and out[1]["severity"] == "medium"
    assert all(x["source"] == "ai" and x["id"] == "" and x["status"] == "open" for x in out)
    assert "Launch" in w.assessed[0] and "Known one" in w.assessed[0]
    assert [x["id"] for x in call(w, "mem", "GET", f"/projects/{pid}").json()["risks"]] == [
        "r1"
    ]  # not saved
    assert call(w, "other", "POST", f"/projects/{pid}/risks/assess").status_code == 403


@pytest.mark.parametrize(
    "reply", [RuntimeError("model down"), "not json", '"just a string"', TimeoutError()]
)
def test_assess_fails_soft(w, reply):
    pid = make(w)["id"]
    w.assessor_reply = reply
    r = call(w, "mem", "POST", f"/projects/{pid}/risks/assess")
    assert r.status_code == 200 and r.json()["proposed"] == []
    assert reply == '"just a string"' or "note" in r.json()


def test_assess_times_out(w, monkeypatch):
    import time

    pid = make(w)["id"]
    monkeypatch.setattr(pm_api, "ASSESS_TIMEOUT_S", 0.05)
    w.assessor_reply = {"risks": []}
    app.dependency_overrides[pm_api.get_assessor] = lambda: lambda p: time.sleep(0.3) or "{}"
    r = call(w, "mem", "POST", f"/projects/{pid}/risks/assess")
    assert r.json()["proposed"] == [] and "note" in r.json()


# ------------------------------------------------------------------ hackathon
def test_hackathon_block_edits(w):
    pid = make(w, name="Cup", kind="hackathon")["id"]
    url = f"/projects/{pid}/hackathon"
    r = call(w, "mem", "PATCH", url, {
        "competition": "Builder Cup", "official_deadline": "2026-10-18T23:59", "rules": "# Rules",
        "team_roles": [{"person": person("peat"), "role": "pitch"}],
        "deliverables": [
            {"id": "d1", "done": True, "url": "https://github.com/x/y"},
            {"id": "d4", "owner": person("peat")},
            {"title": "Poster", "kind": "other"},
            {"id": "d2", "delete": True},
        ],
    })  # fmt: skip
    assert r.status_code == 200, r.text
    h = r.json()["hackathon"]
    assert (
        h["competition"] == "Builder Cup"
        and h["team_roles"][0]["person"]["email"] == "peat@example.com"
    )
    assert [d["id"] for d in h["deliverables"]] == ["d1", "d3", "d4", "d5"]
    assert h["deliverables"][0]["done"] is True and h["deliverables"][0]["url"].endswith("x/y")
    assert (
        h["deliverables"][2]["owner"]["email"] == "peat@example.com"
        and h["deliverables"][3]["title"] == "Poster"
    )
    got = call(w, "mem", "GET", f"/projects/{pid}").json()
    assert got["hackathon"]["official_deadline"] == "2026-10-18T23:59" and got["changes"] == []


def test_submission_status_change_is_logged_and_stamps_the_time(w):
    pid = make(w, name="Cup", kind="hackathon")["id"]
    url = f"/projects/{pid}/hackathon"
    r = call(
        w,
        "mem",
        "PATCH",
        url,
        {"submission_status": "submitted", "reason": "sent", "impact": "done"},
    )
    assert r.json()["hackathon"]["submission_status"] == "submitted" and r.json()["hackathon"][
        "submitted_at"
    ].startswith("2026-10-14")
    ch = call(w, "mem", "GET", f"/projects/{pid}").json()["changes"]
    assert [(c["field"], c["from"], c["to"], c["reason"]) for c in ch] == [
        ("submission_status", "not_started", "submitted", "sent")
    ]
    call(w, "mem", "PATCH", url, {"submission_status": "submitted"})  # unchanged: no new entry
    assert len(call(w, "mem", "GET", f"/projects/{pid}").json()["changes"]) == 1
    assert call(w, "mem", "PATCH", url, {"submission_status": "lucky"}).status_code == 422
    assert call(w, "mem", "PATCH", url, {"official_deadline": "later"}).status_code == 422
    assert call(w, "other", "PATCH", url, {"results": "x"}).status_code == 403


def test_hackathon_endpoint_refuses_a_standard_project(w):
    pid = make(w)["id"]
    r = call(w, "mem", "PATCH", f"/projects/{pid}/hackathon", {"results": "x"})
    assert r.status_code == 422 and r.json()["detail"] == "not_a_hackathon"


# ------------------------------------------------------------------ my tasks / home
def seed_tasks(w):
    a = make(w, name="Alpha")["id"]
    b = make(w, name="Beta", workspace="web")["id"]
    mine = {"assignee": {"email": "mem@example.com"}}
    call(w, "mem", "POST", f"/projects/{a}/tasks", {"title": "a-late", "due": "2026-10-10", **mine})
    call(
        w, "mem", "POST", f"/projects/{a}/tasks", {"title": "a-theirs", "assignee": person("peat")}
    )
    call(w, "mem", "POST", f"/projects/{b}/tasks", {"title": "b-soon", "due": "2026-10-15", **mine})
    done = call(w, "mem", "POST", f"/projects/{b}/tasks", {"title": "b-done", **mine}).json()
    call(w, "mem", "PATCH", f"/projects/{b}/tasks/{done['id']}", {"status": "done"})
    w.tasks.save(
        Task(guild_id=GUILD, title="loose", assignee_id=26, assignee_name="mem", due="2026-10-12")
    )
    w.tasks.save(Task(guild_id=GUILD, title="loose-peat", assignee_id=99))
    return a, b


def test_my_tasks_aggregate_across_projects_and_loose_tasks(w):
    a, b = seed_tasks(w)
    r = call(w, "mem", "GET", "/my-tasks")
    assert [(t["title"], t["project_name"], t["project_id"]) for t in r.json()["tasks"]] == [
        ("a-late", "Alpha", a), ("loose", "", ""), ("b-soon", "Beta", b),
    ]  # fmt: skip
    assert [t["title"] for t in call(w, "mem", "GET", "/my-tasks", status="all").json()["tasks"]][
        -1
    ] == "b-done"
    assert [
        t["title"] for t in call(w, "mem", "GET", "/my-tasks", status="done").json()["tasks"]
    ] == ["b-done"]
    assert [t["title"] for t in call(w, "peat", "GET", "/my-tasks").json()["tasks"]] == ["a-theirs"]
    assert call(w, "other", "GET", "/my-tasks").json() == {"tasks": []}
    assert call(w, "mem", "GET", "/my-tasks", status="weird").status_code == 422


def test_my_tasks_match_by_discord_id_when_the_task_has_no_email(w):
    seed_tasks(w)
    # "loose" has only a Discord id (26): mem's member record carries it
    assert "loose" in [t["title"] for t in call(w, "mem", "GET", "/my-tasks").json()["tasks"]]
    assert "loose" not in [t["title"] for t in call(w, "peat", "GET", "/my-tasks").json()["tasks"]]


def test_home_shape(w):
    a, b = seed_tasks(w)
    call(
        w,
        "mem",
        "PATCH",
        f"/projects/{a}",
        {"priority": "high", "status": "in_progress", "reason": "go"},
    )
    call(w, "mem", "POST", f"/projects/{a}/milestones", {"title": "Demo", "due": "2026-10-16"})
    call(w, "mem", "PATCH", f"/projects/{b}", {"deadline": "2026-10-20", "reason": "r"})
    h = call(w, "mem", "GET", "/home").json()
    assert h["me"] == {"email": "mem@example.com", "discord_user_id": "26", "name": "mem"}
    assert h["counts"] == {"projects": 2, "active": 1, "needs_review": 1, "my_open_tasks": 3}
    assert [t["title"] for t in h["my_tasks"]] == ["a-late", "loose", "b-soon"]
    assert [t["title"] for t in h["urgent_tasks"]] == ["a-late", "loose", "b-soon"]
    assert [p["id"] for p in h["at_risk"]] == [a] and h["at_risk"][0]["stats"][
        "review_reasons"
    ] == ["pm_review_tasks"]
    assert [(d["kind"], d["title"]) for d in h["deadlines"]] == [
        ("milestone", "Demo"),
        ("project", "Beta"),
    ]
    assert set(h["deadlines"][0]) == {"when", "kind", "project_id", "project_name", "title"}
    other = call(w, "other", "GET", "/home").json()
    assert other["counts"]["my_open_tasks"] == 0 and other["counts"]["projects"] == 2
    assert [t["title"] for t in other["urgent_tasks"]] == []  # not theirs, not in their projects


# ------------------------------------------------------------------ insights
def test_insights_shape_and_finance_gate(w):
    a, b = seed_tasks(w)
    call(w, "fin", "PATCH", f"/projects/{a}/budget", {"planned": 100, "spent": 40})
    cup = make(w, name="Cup", kind="hackathon", department="business")["id"]
    call(
        w,
        "mem",
        "PATCH",
        f"/projects/{cup}/hackathon",
        {"competition": "C", "official_deadline": "2026-10-18T23:59"},
    )
    for who in ("mem", "other"):
        i = call(w, who, "GET", "/insights").json()
        assert set(i) == {"progress", "resources", "hackathons", "budget"} and i["budget"] is None
    i = call(w, "mem", "GET", "/insights").json()
    dept = {d["department"]: d for d in i["progress"]["by_department"]}
    assert dept["tech"]["projects"] == 2 and dept["business"]["projects"] == 1
    assert (dept["tech"]["tasks_total"], dept["tech"]["tasks_done"]) == (4, 1)
    assert i["progress"]["by_status"]["planned"] == 3
    people = {p["person"]["email"]: p for p in i["resources"]["people"]}
    assert (
        people["mem@example.com"]["open_tasks"] == 2
        and people["peat@example.com"]["open_tasks"] == 1
    )
    assert [p["person"]["email"] for p in i["resources"]["by_department_people"]["tech"]] == [
        "mem@example.com"
    ]
    assert i["hackathons"][0]["name"] == "Cup" and i["hackathons"][0]["checklist_total"] == 4
    assert i["hackathons"][0]["next_deadline"] == "2026-10-18T23:59"
    for who in ("fin", "adm", "root"):
        b_ = call(w, who, "GET", "/insights").json()["budget"]
        assert b_["total_planned"] == 100 and b_["total_spent"] == 40 and b_["currency"] == "THB", (
            who
        )
        assert {r["project_id"] for r in b_["by_project"]} >= {a}


# ------------------------------------------------------------------ settings
def test_settings_shape(w):
    s = call(w, "other", "GET", "/settings").json()
    assert {m["email"] for m in s["members"]} == {
        f"{n}@example.com" for n in ("boss", "adm", "mem", "peat", "fin", "other")
    }
    mem = next(m for m in s["members"] if m["email"] == "mem@example.com")
    assert (
        mem["department"] == "tech" and mem["finance"] is False and mem["discord_user_id"] == "26"
    )
    assert [d["key"] for d in s["departments"]] == ["tech", "business", "finance", "other"]
    assert s["departments"][0] == {"key": "tech", "label_th": "เทค", "label_en": "Tech"}
    assert [t["kind"] for t in s["templates"]] == ["standard", "hackathon"]
    assert s["defaults"] == {
        "visibility": "team",
        "task_confirm_mode": "notify",
        "personal_settings": True,
    } and s["company"] == {
        "name": "Acme",
        "logo_url": "",
    }


def test_settings_patch_is_admin_only_and_validated(w):
    body = {
        "members": [
            {"email": "peat@example.com", "department": "tech", "finance": True, "role": "admin"}
        ],
        "defaults": {"visibility": "members"},
    }
    assert call(w, "mem", "PATCH", "/settings", body).status_code == 403
    assert call(w, "fin", "PATCH", "/settings", body).status_code == 403  # finance is not admin
    s = call(w, "adm", "PATCH", "/settings", body).json()  # admins grant admin (Members page)
    peat = next(m for m in s["members"] if m["email"] == "peat@example.com")
    assert (peat["department"], peat["finance"], peat["role"]) == ("tech", True, "admin")
    assert s["defaults"]["visibility"] == "members"
    assert w.orgs.get(w.org_id).role_of("peat@example.com") == "admin"
    assert (
        call(w, "peat", "PATCH", "/settings", {"defaults": {"visibility": "team"}}).status_code
        == 200
    )  # now an admin
    assert make(w, name="Fresh")["visibility"] == "team"
    call(w, "adm", "PATCH", "/settings", {"defaults": {"visibility": "members"}})
    assert make(w, name="Quiet")["visibility"] == "members"
    assert make(w, name="Loud", visibility="team")["visibility"] == "team"  # explicit wins
    for bad in (
        {"members": [{"email": "ghost@example.com", "finance": True}]},
        {"members": [{"email": "peat@example.com", "department": "sales"}]},
        {"members": [{"email": "peat@example.com", "role": "owner"}]},
        {"members": [{"email": "boss@example.com", "role": "member"}]},
        {"defaults": {"visibility": "public"}}, {"nope": 1},
    ):  # fmt: skip
        assert call(w, "adm", "PATCH", "/settings", bad).status_code == 422, bad
    assert w.orgs.get(w.org_id).role_of("boss@example.com") == "owner"


def test_settings_role_changes_follow_the_members_page_rules(w):
    """Roles change by `center.role_change_error`, as on the Members page: an admin may grant
    admin, but only the owner (or a super admin) changes somebody who already is an admin, and
    nobody raises their own role. Other PM settings stay open to every admin."""

    def patch(who, target, **fields):
        member = {"email": f"{target}@example.com", **fields}
        return call(w, who, "PATCH", "/settings", {"members": [member]})

    def member(name):
        org = w.orgs.get(w.org_id)
        return next(m for m in org.members if m.email == f"{name}@example.com")

    assert patch("adm", "peat", role="admin").status_code == 200  # an admin grants admin
    r = patch("peat", "adm", role="member", department="finance")
    assert r.status_code == 403, r.text  # an admin cannot demote another admin
    assert (member("adm").role, member("adm").department) == ("admin", "")  # nothing saved
    assert patch("peat", "adm", role="admin", department="tech").status_code == 200  # no change
    assert (member("adm").role, member("adm").department) == ("admin", "tech")
    assert patch("boss", "peat", role="member").status_code == 200  # the owner may
    assert member("peat").role == "member"
    assert patch("adm", "peat", role="admin").status_code == 200
    assert patch("root", "peat", role="member").status_code == 200  # so may a super admin
    assert member("peat").role == "member"
    assert patch("adm", "adm", role="member").status_code == 200  # lowering yourself is fine
    assert member("adm").role == "member"


def test_finance_flag_from_settings_opens_the_budget(w):
    pid = make(w)["id"]
    assert call(w, "other", "PATCH", f"/projects/{pid}/budget", {"planned": 1}).status_code == 403
    call(
        w,
        "adm",
        "PATCH",
        "/settings",
        {"members": [{"email": "other@example.com", "finance": True}]},
    )
    assert call(w, "other", "PATCH", f"/projects/{pid}/budget", {"planned": 1}).status_code == 200
    assert call(w, "other", "GET", f"/projects/{pid}").json()["budget"]["planned"] == 1


# ------------------------------------------------------------------ updates heard in chat
def test_updates_heard_in_chat_are_served_on_the_project_not_on_the_card(w):
    from salid import projects as P

    pid = make(w)["id"]
    repo = P.ProjectRepo(w.docs)
    p = repo.get(GUILD, pid)
    assert call(w, "mem", "GET", f"/projects/{pid}").json()["updates"] == []
    p.updates = [
        P.Update(at="2026-10-14 10:00", by=P.Person(name="Toey"), text="Beta on track", quote="q")
    ]
    repo.save(GUILD, p)
    got = call(w, "mem", "GET", f"/projects/{pid}").json()
    assert (
        got["updates"][0]["text"] == "Beta on track" and got["updates"][0]["by"]["name"] == "Toey"
    )
    card = next(x for x in call(w, "mem", "GET", "/projects").json()["projects"] if x["id"] == pid)
    assert "updates" not in card
    # a normal edit keeps them
    assert call(w, "mem", "PATCH", f"/projects/{pid}", {"overview": "hi"}).status_code == 200
    assert len(call(w, "mem", "GET", f"/projects/{pid}").json()["updates"]) == 1


def test_task_json_carries_the_handshake_state(w):
    p = make(w, "adm")
    w.tasks.save(
        Task(guild_id=GUILD, id="hs000001", project_id=p["id"], title="x", assigner_id=77,
             assigner_name="Boss", confirm="pending")
    )  # fmt: skip
    rows = call(w, "adm", "GET", f"/projects/{p['id']}/tasks").json()["tasks"]
    row = next(r for r in rows if r["id"] == "hs000001")
    assert row["confirm"] == "pending" and row["assigner"] == {
        "discord_user_id": "77",
        "name": "Boss",
    }


def test_task_json_flags_a_task_without_a_deadline(w):
    p = make(w, "adm")
    w.tasks.save(Task(guild_id=GUILD, id="nd000001", project_id=p["id"], title="x"))
    w.tasks.save(
        Task(guild_id=GUILD, id="nd000002", project_id=p["id"], title="y", due="2026-10-20")
    )
    rows = call(w, "adm", "GET", f"/projects/{p['id']}/tasks").json()["tasks"]
    flags = {r["id"]: r["no_deadline"] for r in rows}
    assert flags == {"nd000001": True, "nd000002": False}


def test_task_mode_and_personal_settings_are_admin_only_and_reach_the_guild(w):
    body = {"defaults": {"task_confirm_mode": "approve", "personal_settings": False}}
    assert call(w, "mem", "PATCH", "/settings", body).status_code == 403
    s = call(w, "adm", "PATCH", "/settings", body).json()
    assert (
        s["defaults"]["task_confirm_mode"] == "approve"
        and s["defaults"]["personal_settings"] is False
    )
    g = w.guilds.get(GUILD)  # the bot reads it from the guild settings
    assert g.task_confirm_mode == "approve" and g.personal_settings is False
    w.guilds.set_task_confirm_mode(GUILD, "notify")  # a change made in the hub / by chat shows here
    assert call(w, "other", "GET", "/settings").json()["defaults"]["task_confirm_mode"] == "notify"
    bad = {"defaults": {"task_confirm_mode": "dictator"}}
    assert call(w, "adm", "PATCH", "/settings", bad).status_code == 422


# ------------------------------------------------------------------ progress vs time, plan fields
def test_stats_carry_progress_on_cards_and_in_the_full_project(w):
    a = make(w, name="Alpha", deadline="2026-10-24", status="in_progress")["id"]
    call(w, "mem", "POST", f"/projects/{a}/tasks", {"title": "x", "due": "2026-10-20"})
    done = call(w, "mem", "POST", f"/projects/{a}/tasks", {"title": "y"}).json()
    call(w, "mem", "PATCH", f"/projects/{a}/tasks/{done['id']}", {"status": "done"})
    card = call(w, "mem", "GET", "/projects").json()["projects"][0]
    pg = card["stats"]["progress"]
    assert pg["done_pct"] == 50 and pg["days_left"] == 10 and pg["state"] == "on_track"
    assert pg["time_pct"] == 0 and pg["next_due"]["title"] == "x"
    assert "tasks" not in pg  # per-task rows stay out of the cards
    full = call(w, "mem", "GET", f"/projects/{a}").json()
    assert full["stats"]["progress"] == pg
    none = make(w, name="Open")  # no deadline, no tasks
    nd = call(w, "mem", "GET", f"/projects/{none['id']}").json()["stats"]["progress"]
    assert nd["state"] == "no_deadline" and nd["done_pct"] is None and nd["days_left"] is None


def test_home_lists_a_running_project_that_is_far_behind_with_its_reason(w):
    late = make(w, name="Late", deadline="2026-10-10", status="in_progress")["id"]
    make(w, name="Fine", deadline="2026-12-10", status="in_progress")
    h = call(w, "mem", "GET", "/home").json()
    assert [p["id"] for p in h["at_risk"]] == [late] and h["counts"]["needs_review"] == 1
    st = h["at_risk"][0]["stats"]
    assert st["progress"]["state"] == "at_risk" and st["needs_review"] is False
    assert st["review_reasons"] == ["pm_review_progress"]
    assert st["review_reasons_text"][0]["en"].startswith("work is far behind")
    planned = make(w, name="Idea", deadline="2026-10-01", status="planned")["id"]
    assert planned not in [p["id"] for p in call(w, "mem", "GET", "/home").json()["at_risk"]]


def test_insights_progress_by_project(w):
    make(w, name="Late", deadline="2026-10-10", status="in_progress")
    make(w, name="Fine", deadline="2026-12-10", status="in_progress")
    rows = call(w, "mem", "GET", "/insights").json()["progress"]["by_project"]
    assert [(r["name"], r["state"]) for r in rows] == [("Late", "at_risk"), ("Fine", "on_track")]
    assert set(rows[0]) == {"project_id", "name", "state", "done_pct", "time_pct", "days_left"}


def test_task_plan_fields_through_the_api(w):
    a = make(w)["id"]
    r = call(
        w, "mem", "POST", f"/projects/{a}/tasks",
        {"title": "Step", "due": "2026-10-20", "start": "2026-10-15", "step": 2, "pct_done": 30},
    )  # fmt: skip
    t = r.json()
    assert (t["start"], t["step"], t["pct_done"], t["depends_on"]) == ("2026-10-15", 2, 30, [])
    p = call(w, "mem", "PATCH", f"/projects/{a}/tasks/{t['id']}", {"pct_done": 80, "step": 3})
    assert (
        p.json()["pct_done"] == 80 and p.json()["step"] == 3 and p.json()["start"] == "2026-10-15"
    )
    bad = [{"pct_done": 101}, {"pct_done": -1}, {"step": -2}, {"start": "friday"}]
    for body in bad:
        assert call(w, "mem", "PATCH", f"/projects/{a}/tasks/{t['id']}", body).status_code == 422
    assert (
        call(w, "mem", "POST", f"/projects/{a}/tasks", {"title": "z", "start": "soon"}).status_code
        == 422
    )
    call(w, "mem", "PATCH", f"/projects/{a}/tasks/{t['id']}", {"status": "done"})
    assert call(w, "mem", "GET", f"/projects/{a}/tasks").json()["tasks"][0]["pct_done"] == 100


def test_settings_expose_the_company_language_as_the_pages_default(w):
    assert call(w, "other", "GET", "/settings").json()["language"] in ("", "th", "en")
    w.guilds.set_server_language(GUILD, "th")
    assert call(w, "other", "GET", "/settings").json()["language"] == "th"
    w.guilds.set_server_language(GUILD, "en")
    assert call(w, "other", "GET", "/settings").json()["language"] == "en"


def test_the_risk_prompt_uses_the_company_language():
    from salid import projects as P

    p = P.Project(id="p1", name="Alpha", owner=P.Person(name="Boss"))
    stats = {"tasks_done": 0, "tasks_total": 0, "tasks_overdue": 0, "tasks_unassigned": 0}
    assert "Write the risks in Thai." in pm_api._risk_prompt(p, stats, "th")
    assert "Write the risks in English." in pm_api._risk_prompt(p, stats, "en")
    assert "language of the project text" in pm_api._risk_prompt(p, stats, "")
