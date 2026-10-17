# ruff: noqa: F811, F401
"""PM projects: models, templates, visibility, reasons for big changes, stats / needs_review,
deadline window, insights (with and without finance), person matching, the agent tools (ambiguity,
permissions), the task auto-link and the glue (tools in the agent, voice rebinding, prompt, flag).
Fakes only."""

from datetime import date, datetime
from types import SimpleNamespace

import pytest
from conftest import GUILD
from fake_firestore import FakeFirestore
from test_briefings import set_features
from test_coordinator import BKK, BOSS, PEER, STAFF, at, person, world
from test_modes import bot  # noqa: F401  (fixture)

from salid import projects as P
from salid.coo_store import FirestoreDocStore, SQLiteDocStore
from salid.i18n import CATALOG, t
from salid.orgs import Member
from salid.qa import stable_prompt
from salid.tasks import Task, TaskRepo, TaskTracker
from salid.voice_actions import ACTION_TOOL_NAMES

NOW = at(12)  # Wednesday 2026-10-14 12:00 Bangkok
TODAY = NOW.date()
ADMIN = P.Person(email="boss@x.com", discord_user_id="25", name="Boss")
TOEY = P.Person(email="toey@x.com", discord_user_id="26", name="Toey")
PEAT = P.Person(email="peat@x.com", discord_user_id="27", name="Peat")
OUT = P.Person(email="out@x.com", name="Outsider")


@pytest.fixture(params=["sqlite", "firestore"])
def docs(request):
    if request.param == "sqlite":
        return SQLiteDocStore(":memory:")
    return FirestoreDocStore(client=FakeFirestore())


def proj(**kw) -> P.Project:
    base = dict(
        name="Alpha", owner=TOEY, department="tech", priority="medium", status="in_progress"
    )
    return P.Project(**{**base, **kw})


# ------------------------------------------------------------------ models and persons
def test_person_needs_an_identity_and_normalises():
    with pytest.raises(ValueError):
        P.Person()
    p = P.Person(email=" A@B.COM ", discord_user_id=123)
    assert p.email == "a@b.com" and p.discord_user_id == "123" and p.name == ""
    assert P.Person(name="Toey").label == "Toey"


def test_person_matches_across_email_and_discord():
    p = P.Person(email="a@b.com", discord_user_id="9", name="A")
    assert P.person_matches(p, "A@b.com", None) and P.person_matches(p, None, 9)
    assert P.person_matches(p, "x@y.com", "9") and not P.person_matches(p, "x@y.com", "8")
    assert not P.person_matches(p, None, None) and not P.person_matches(None, "a@b.com")
    name_only = P.Person(name="Toey")
    assert not P.person_matches(name_only, "toey@x.com", "1")  # a bare name never matches "me"


def test_project_validates_enums_dates_and_name():
    with pytest.raises(ValueError):
        proj(priority="urgent")
    with pytest.raises(ValueError):
        proj(deadline="30/11/2026")
    with pytest.raises(ValueError):
        proj(name="  ")
    with pytest.raises(ValueError):
        P.Link(url="javascript:alert(1)")
    assert proj(deadline="2026-11-30").deadline == "2026-11-30"


def test_change_serialises_with_from_alias():
    p = proj()
    P.record_change(p, "status", "planned", "in_progress", ADMIN, reason="go", now=NOW)
    doc = p.model_dump(by_alias=True)
    assert doc["changes"][0]["from"] == "planned" and "from_" not in doc["changes"][0]
    assert P.Project.model_validate(doc).changes[0].from_ == "planned"


# ------------------------------------------------------------------ templates
def test_standard_template_is_empty_and_hackathon_gets_four_deliverables():
    s = P.apply_template(proj())
    assert s.hackathon is None
    h = P.apply_template(proj(kind="hackathon")).hackathon
    assert [d.title for d in h.deliverables] == ["GitHub repo", "Demo", "Video", "Pitch deck"]
    assert [d.kind for d in h.deliverables] == ["github", "demo", "video", "deck"]
    assert [d.id for d in h.deliverables] == ["d1", "d2", "d3", "d4"]
    assert h.submission_status == "not_started"
    again = P.apply_template(P.apply_template(proj(kind="hackathon")))
    assert len(again.hackathon.deliverables) == 4  # idempotent


def test_templates_view_lists_sections_with_labels():
    by_kind = {x["kind"]: x for x in P.templates_view()}
    assert by_kind["standard"]["sections"] == [
        "sec_overview", "sec_tasks", "sec_milestones", "sec_risks", "sec_links",
    ]  # fmt: skip
    assert (
        "sec_budget" in by_kind["hackathon"]["sections"]
        and "sec_submission" in by_kind["hackathon"]["sections"]
    )
    assert by_kind["hackathon"]["defaults"]["deliverables"][0] == "GitHub repo"
    assert by_kind["standard"]["section_labels"][0] == {
        "key": "sec_overview", "th": "ภาพรวม", "en": "Overview",
    }  # fmt: skip


# ------------------------------------------------------------------ visibility
def test_visibility_rules():
    team = proj(created_by=ADMIN)
    private = proj(visibility="members", members=[P.ProjectMember(person=PEAT, role="dev")])
    assert P.can_read(team, OUT, False)  # team: any org member
    assert not P.can_read(private, OUT, False)
    assert P.can_read(private, TOEY, False) and P.can_read(private, PEAT, False)
    assert P.can_read(private, OUT, True)  # admins see everything
    private.deleted = True
    assert not P.can_read(private, ADMIN, True)


def test_edit_and_contribute_roles():
    p = proj(members=[P.ProjectMember(person=PEAT)])
    assert P.can_edit(p, TOEY, False) and P.can_edit(p, OUT, True)
    assert not P.can_edit(p, PEAT, False) and not P.can_edit(p, OUT, False)
    assert P.can_contribute(p, PEAT, False) and not P.can_contribute(p, OUT, False)


def test_finance_flag_and_stripping():
    assert P.is_finance(None, True) and not P.is_finance(None, False)
    assert P.is_finance(Member(email="f@x.com", finance=True), False)
    assert not P.is_finance(Member(email="m@x.com"), False)
    stats = P.project_stats(proj(), [], TODAY)
    full = P.project_json(proj(), finance_ok=True, stats=stats)
    assert "budget" in full and "deleted" not in full
    assert "budget" not in P.project_json(proj(), finance_ok=False, stats=stats)
    card = P.card_json(P.apply_template(proj(kind="hackathon")), stats)
    assert "budget" not in card and "milestones" not in card and "overview" not in card
    assert set(card["hackathon"]) == {
        "submission_status",
        "official_deadline",
        "checklist_done",
        "checklist_total",
    }


def test_members_of_old_docs_load_without_the_new_fields():
    m = Member.model_validate({"email": "a@b.com", "role": "member"})
    assert m.department == "" and m.finance is False
    t_ = Task.model_validate({"guild_id": 1, "title": "x"})
    assert t_.project_id == "" and t_.assignee_email == "" and t_.milestone_id == ""


# ------------------------------------------------------------------ reasons for big changes
@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "on_hold"),
        ("owner", PEAT.model_dump()),
        ("deadline", "2026-12-01"),
        ("priority", "high"),
    ],
)
def test_big_changes_need_a_reason(field, value):
    p = proj(deadline="2026-11-30")
    with pytest.raises(P.ReasonRequired) as e:
        P.apply_updates(p, {field: value}, ADMIN, now=NOW)
    assert e.value.field == field
    with pytest.raises(P.ReasonRequired):
        P.apply_updates(p, {field: value}, ADMIN, reason="  ", now=NOW)
    new, changes = P.apply_updates(
        p, {field: value}, ADMIN, reason="client asked", impact="2 days", now=NOW
    )
    assert len(changes) == 1 and changes[0].field == field
    assert changes[0].reason == "client asked" and changes[0].impact == "2 days"
    assert changes[0].by == ADMIN and new.changes[-1] is changes[0]
    assert p.changes == []  # the original is untouched


def test_small_changes_and_unchanged_values_need_no_reason():
    p = proj(priority="high")
    new, changes = P.apply_updates(
        p, {"name": "Beta", "overview": "hi", "priority": "high"}, ADMIN, now=NOW
    )
    assert new.name == "Beta" and new.overview == "hi" and changes == []


def test_owner_change_logs_names_and_bad_values_raise():
    new, ch = P.apply_updates(proj(), {"owner": PEAT.model_dump()}, ADMIN, reason="swap", now=NOW)
    assert (ch[0].from_, ch[0].to) == ("Toey", "Peat") and new.owner == PEAT
    with pytest.raises(ValueError):
        P.apply_updates(proj(), {"status": "nope"}, ADMIN, reason="x", now=NOW)


# ------------------------------------------------------------------ stats and needs_review
def t_(pid="p1", due="", status="open", who=0, name="", email="", title="t"):
    return Task(
        guild_id=GUILD, project_id=pid, title=title, due=due, status=status,
        assignee_id=who, assignee_name=name, assignee_email=email,
    )  # fmt: skip


def test_stats_count_tasks_and_milestones():
    p = proj(
        id="p1",
        milestones=[
            P.Milestone(id="m1", title="a", status="done"),
            P.Milestone(id="m2", title="b"),
        ],
    )
    tasks = [
        t_(status="done", who=26, name="Toey"), t_(due="2026-10-01", who=26, name="Toey"),
        t_(), t_(pid="other", due="2026-01-01"),
    ]  # fmt: skip
    s = P.project_stats(p, tasks, TODAY)
    assert (s["tasks_total"], s["tasks_done"], s["tasks_overdue"], s["tasks_unassigned"]) == (
        3,
        1,
        1,
        1,
    )
    assert (s["milestones_done"], s["milestones_total"]) == (1, 2)
    assert (
        s["needs_review"] is False and s["review_reasons"] == []
    )  # medium priority: never flagged


def test_needs_review_rules_for_high_priority():
    late = proj(id="p1", priority="high", deadline="2026-10-01")
    s = P.project_stats(late, [], TODAY)
    assert s["needs_review"] and s["review_reasons"] == ["pm_review_deadline"]
    assert s["review_reasons_text"][0]["en"] == t("pm_review_deadline", "en")
    assert not P.project_stats(proj(id="p1", priority="high", deadline="2026-10-14"), [], TODAY)[
        "needs_review"
    ]
    assert not P.project_stats(
        proj(id="p1", priority="high", deadline="2026-10-01", status="completed"), [], TODAY
    )["needs_review"]  # fmt: skip
    ontime = proj(id="p1", priority="high", deadline="2026-12-01")
    assert P.project_stats(ontime, [t_(due="2026-10-10")], TODAY)["review_reasons"] == [
        "pm_review_tasks"
    ]
    assert not P.project_stats(ontime, [t_(due="2026-10-10", status="done")], TODAY)["needs_review"]


def test_on_hold_for_over_a_week_is_flagged():
    hold = proj(id="p1", priority="high", status="in_progress")
    hold, _ = P.apply_updates(hold, {"status": "on_hold"}, ADMIN, reason="wait", now=at(9, day=1))
    assert P.project_stats(hold, [], TODAY)["review_reasons"] == ["pm_review_on_hold"]  # 13 days
    assert not P.project_stats(hold, [], date(2026, 10, 8))["needs_review"]  # 7 days: not yet
    assert P.project_stats(hold, [], date(2026, 10, 9))["needs_review"]  # 8 days


# ------------------------------------------------------------------ deadlines
def test_deadlines_window_collects_project_milestone_and_hackathon_dates():
    hack = P.apply_template(proj(name="Cup", kind="hackathon", deadline="2026-11-30"))
    hack.hackathon.competition = "Builder Cup"
    hack.hackathon.official_deadline = "2026-10-18T23:59"
    a = proj(
        name="Alpha", deadline="2026-10-20",
        milestones=[
            P.Milestone(id="m1", title="Beta", due="2026-10-30"),  # beyond 14 days
            P.Milestone(id="m2", title="Demo", due="2026-10-16"),
            P.Milestone(id="m3", title="Done one", due="2026-10-15", status="done"),
            P.Milestone(id="m4", title="Past", due="2026-10-01"),
        ],
    )  # fmt: skip
    gone = proj(name="Old", deadline="2026-10-15", status="completed")
    out = P.deadlines_window([a, hack, gone], 14, TODAY)
    assert [(d["kind"], d["title"]) for d in out] == [
        ("milestone", "Demo"), ("hackathon", "Builder Cup"), ("project", "Alpha"),
    ]  # fmt: skip
    assert out[1]["when"] == "2026-10-18T23:59"
    assert P.deadlines_window([a], 1, TODAY) == []


# ------------------------------------------------------------------ insights
def insight_world():
    members = [
        Member(email="toey@x.com", discord_user_id="26", department="tech"),
        Member(email="peat@x.com", department="business"),
    ]
    a = proj(id="p1", name="Alpha", department="tech")
    a.budget = P.Budget(planned=1000, spent=400, travel=50)
    b = proj(id="p2", name="Beta", department="business", status="completed", owner=PEAT)
    h = P.apply_template(proj(id="p3", name="Cup", kind="hackathon", department="tech"))
    h.hackathon.official_deadline = "2026-10-18T23:59"
    h.hackathon.deliverables[0].done = True
    tasks = [
        t_("p1", who=26, name="Toey"), t_("p1", status="done", who=26, name="Toey"),
        t_("p1", email="peat@x.com", name="Peat"), t_("p2", status="done"), t_("zzz"),
    ]  # fmt: skip
    return [a, b, h], tasks, members


def test_insights_progress_resources_hackathons():
    projects, tasks, members = insight_world()
    out = P.insights(projects, tasks, members, False, TODAY)
    dept = {d["department"]: d for d in out["progress"]["by_department"]}
    assert dept["tech"]["projects"] == 2 and dept["tech"]["active"] == 2
    assert dept["business"]["completed"] == 1
    assert (dept["tech"]["tasks_total"], dept["tech"]["tasks_done"]) == (3, 1)
    assert (
        out["progress"]["by_status"]["in_progress"] == 2
        and out["progress"]["by_status"]["draft"] == 0
    )
    people = {p["person"]["email"]: p for p in out["resources"]["people"]}
    assert people["toey@x.com"]["open_tasks"] == 1 and people["toey@x.com"]["projects"] == 2
    assert people["peat@x.com"]["open_tasks"] == 1  # merged by email, not duplicated
    assert [p["person"]["email"] for p in out["resources"]["by_department_people"]["business"]] == [
        "peat@x.com"
    ]
    assert set(out["resources"]["by_department_people"]) == set(P.DEPARTMENTS)
    (hk,) = out["hackathons"]
    assert hk["project_id"] == "p3" and hk["next_deadline"] == "2026-10-18T23:59"
    assert (hk["checklist_done"], hk["checklist_total"]) == (1, 4)
    assert out["budget"] is None


def test_insights_budget_only_with_finance():
    projects, tasks, members = insight_world()
    out = P.insights(projects, tasks, members, True, TODAY)
    assert out["budget"]["total_planned"] == 1000 and out["budget"]["total_spent"] == 400
    assert out["budget"]["currency"] == "THB" and len(out["budget"]["by_project"]) == 3
    assert P.insights(projects, tasks, members, False, TODAY)["budget"] is None


def test_a_past_official_deadline_is_not_a_next_deadline():
    projects, tasks, members = insight_world()
    projects[2].hackathon.official_deadline = "2026-10-01"
    assert (
        P.insights(projects, tasks, members, False, TODAY)["hackathons"][0]["next_deadline"] == ""
    )


# ------------------------------------------------------------------ repo
def test_repo_create_list_save_soft_delete(docs):
    repo = P.ProjectRepo(docs, clock=lambda: NOW)
    p = repo.create(GUILD, P.apply_template(proj(name="Zed")))
    q = repo.create(GUILD, proj(name="Alpha"))
    assert p.id.startswith("p-") and len(p.id) == 8 and p.created_at.startswith("2026-10-14")
    assert [x.name for x in repo.list(GUILD)] == ["Alpha", "Zed"]
    assert repo.list(2000) == [] and repo.get(2000, p.id) is None
    p.overview = "x"
    repo.save(GUILD, p)
    assert repo.get(GUILD, p.id).overview == "x"
    assert repo.soft_delete(GUILD, q.id, ADMIN) is True
    assert repo.get(GUILD, q.id) is None and [x.name for x in repo.list(GUILD)] == ["Zed"]
    gone = repo.get(GUILD, q.id, include_deleted=True)
    assert gone.deleted and gone.status == "cancelled"
    assert repo.soft_delete(GUILD, "p-nope") is False


def test_task_repo_project_filter_and_for_person(docs):
    repo = TaskRepo(docs)
    repo.save(t_("p1", who=26, name="Toey", title="a"))
    repo.save(t_("p1", email="Peat@x.com", title="b"))
    repo.save(t_("p2", who=26, title="c"))
    repo.save(t_("", who=26, title="d", status="done"))
    assert {x.title for x in repo.list(GUILD, project_id="p1")} == {"a", "b"}
    assert {x.title for x in repo.list(GUILD, status=None, project_id="")} == {"d"}
    assert {x.title for x in repo.for_person(GUILD, None, "26")} == {"a", "c"}
    assert {x.title for x in repo.for_person(GUILD, "peat@x.com", None)} == {"b"}
    assert {x.title for x in repo.for_person(GUILD, "peat@x.com", 26, status=None)} == {
        "a",
        "b",
        "c",
        "d",
    }
    assert repo.for_person(GUILD, None, None) == []


def test_pm_settings_default_and_validation(docs):
    s = P.PmSettings(docs)
    base = {"visibility": "team", "task_confirm_mode": "notify", "personal_settings": True}
    assert s.get("org1") == base
    assert s.set_visibility("org1", "members") == {**base, "visibility": "members"}
    assert s.set_task_confirm_mode("org1", "approve")["task_confirm_mode"] == "approve"
    assert s.set_personal_settings("org1", False)["personal_settings"] is False
    assert s.get("org1")["visibility"] == "members"  # the others are kept
    with pytest.raises(ValueError):
        s.set_task_confirm_mode("org1", "boss")
    s.set_visibility("org1", "team")
    s.set_task_confirm_mode("org1", "notify")
    s.set_personal_settings("org1", True)
    s.set_visibility("org1", "members")
    assert (
        P.PmSettings(docs).get("org1")["visibility"] == "members"
        and s.get("org2")["visibility"] == "team"
    )
    with pytest.raises(ValueError):
        s.set_visibility("org1", "public")


# ------------------------------------------------------------------ finding projects
def test_find_projects_by_id_exact_and_fragment():
    ps = [
        proj(id="p-1", name="Alpha"),
        proj(id="p-2", name="Alpha Two"),
        proj(id="p-3", name="Beta"),
    ]
    assert [p.id for p in P.find_projects(ps, "p-3")] == ["p-3"]
    assert [p.id for p in P.find_projects(ps, "alpha")] == ["p-1"]  # exact wins over fragment
    assert [p.id for p in P.find_projects(ps, "alp")] == ["p-1", "p-2"]
    assert P.find_projects(ps, "") == [] and P.find_projects(ps, "zzz") == []


def test_guess_department():
    assert P.guess_department("Website redesign") == "tech"
    assert P.guess_department("แคมเปญการตลาดไตรมาส 4") == "business"
    assert P.guess_department("Q4 budget review") == "finance"
    assert P.guess_department("Offsite") == ""
    assert P.guess_department("app for sales") == ""  # a tie: ask


# ------------------------------------------------------------------ the auto-link of tasks
def test_project_named_in_text_links_the_task():
    ps = [
        proj(id="p-1", name="Alpha"),
        proj(id="p-2", name="Alpha Launch"),
        proj(id="p-3", name="Old", status="completed"),
    ]
    assert P.project_in_text(ps, "ส่งสไลด์ โปรเจกต์ Alpha Launch ให้ลูกค้า").id == "p-2"  # longest
    assert P.project_in_text(ps, "finish the Project: alpha deck").id == "p-1"
    assert P.project_in_text(ps, "โปรเจค “Alpha” เสร็จหรือยัง").id == "p-1"
    assert P.project_in_text(ps, "talk to Alpha about it") is None  # needs the word project
    assert (
        P.project_in_text(ps, "project Old cleanup") is None
    )  # finished projects do not collect tasks
    assert P.project_in_text(ps, "") is None


def test_task_tracker_sets_project_id_from_an_action_item(docs):
    repo = P.ProjectRepo(docs)
    p = repo.create(GUILD, proj(name="Alpha"))
    tr = TaskTracker(TaskRepo(docs), SimpleNamespace(now=lambda: NOW), None)
    created = tr.ingest(
        GUILD, 10, "2026-10-14",
        {"action_items": [{"what": "ส่งสไลด์ โปรเจกต์ Alpha", "who": "", "due": ""}, {"what": "ซื้อกาแฟ"}]},
        [],
    )  # fmt: skip
    by_title = {x.title: x for x in created}
    assert by_title["ส่งสไลด์ โปรเจกต์ Alpha"].project_id == p.id
    assert by_title["ซื้อกาแฟ"].project_id == ""
    saved = TaskRepo(docs).list(GUILD, project_id=p.id)
    assert [x.title for x in saved] == ["ส่งสไลด์ โปรเจกต์ Alpha"]


def test_task_tracker_survives_a_broken_project_store():
    class Broken(SQLiteDocStore):
        def find(self, kind, guild_id, where=None):
            if kind == P.PROJECTS:
                raise RuntimeError("down")
            return super().find(kind, guild_id, where)

    tr = TaskTracker(TaskRepo(Broken(":memory:")), SimpleNamespace(now=lambda: NOW), None)
    out = tr.ingest(
        GUILD, 10, "2026-10-14", {"action_items": [{"what": "project Alpha thing"}]}, []
    )
    assert len(out) == 1 and out[0].project_id == ""


# ------------------------------------------------------------------ the agent tools
def tools(docs, asker=TOEY, admin=False, lang="en", people=(TOEY, PEAT, ADMIN)):
    def resolve(name):
        low = name.strip().lstrip("@").casefold()
        return [p for p in people if low and low in p.name.casefold()]

    ctx = P.ProjectsCtx(
        guild_id=GUILD, asker=asker, is_admin=admin, lang=lang,
        repo=P.ProjectRepo(docs, clock=lambda: NOW), tasks=TaskRepo(docs),
        now=lambda: NOW, resolve=resolve,
    )  # fmt: skip
    return {f.__name__: f for f in P.project_tools(ctx)}, ctx


def test_project_create_needs_only_name_and_department(docs):
    tl, ctx = tools(docs)
    out = tl["project_create"]("Website redesign", "tech")
    assert out["ok"] and out["id"].startswith("p-")
    p = ctx.repo.get(GUILD, out["id"])
    assert p.owner == TOEY and p.priority == "medium" and p.status == "planned"
    assert p.created_by == TOEY and p.visibility == "team" and p.workspace == ""


def test_project_create_guesses_or_asks_for_the_department(docs):
    tl, ctx = tools(docs)
    assert tl["project_create"]("Mobile app launch", "")["ok"]  # guessed: tech
    assert ctx.repo.list(GUILD)[0].department == "tech"
    out = tl["project_create"]("Offsite", "")
    assert out["ok"] is False and out["reason"] == "department_required"
    assert out["say"] == t("pm_need_department", "en")
    assert (
        tl["project_create"]("Offsite", "marketing")["reason"] == "department_required"
    )  # not a department
    assert tl["project_create"]("  ", "tech")["reason"] == "name_required"


def test_project_create_owner_priority_kind_deadline(docs):
    tl, ctx = tools(docs)
    out = tl["project_create"](
        "Cup", "tech", owner="peat", priority="high", kind="hackathon", deadline="2026-10-18"
    )
    p = ctx.repo.get(GUILD, out["id"])
    assert p.owner == PEAT and p.priority == "high" and p.deadline == "2026-10-18"
    assert len(p.hackathon.deliverables) == 4
    assert tl["project_create"]("X", "tech", owner="nobody")["reason"] == "owner_not_found"
    assert tl["project_create"]("X", "tech", owner="e")["reason"] == "owner_ambiguous"  # Toey, Peat
    assert tl["project_create"]("X", "tech", priority="urgent")["reason"] == "bad_value"
    assert tl["project_create"]("X", "tech", kind="party")["reason"] == "bad_value"
    assert tl["project_create"]("X", "tech", deadline="someday")["reason"] == "bad_value"
    assert tl["project_create"]("X", "tech", deadline="2026-10-16")["ok"]


def test_project_create_refuses_a_duplicate_name(docs):
    tl, ctx = tools(docs)
    first = tl["project_create"]("Alpha", "tech")
    again = tl["project_create"]("alpha", "tech")
    assert (
        again["reason"] == "exists"
        and again["id"] == first["id"]
        and len(ctx.repo.list(GUILD)) == 1
    )


def test_project_update_requires_reason_and_rights(docs):
    tl, ctx = tools(docs)
    pid = tl["project_create"]("Alpha", "tech")["id"]
    out = tl["project_update"]("alpha", "status", "in progress")
    assert out["reason"] == "reason_required" and ctx.repo.get(GUILD, pid).status == "planned"
    ok = tl["project_update"]("alpha", "status", "in progress", reason="kickoff", impact="none")
    assert ok["ok"] and ok["changed"] and ctx.repo.get(GUILD, pid).status == "in_progress"
    ch = ctx.repo.get(GUILD, pid).changes[-1]
    assert (ch.field, ch.from_, ch.to, ch.reason, ch.by) == (
        "status",
        "planned",
        "in_progress",
        "kickoff",
        TOEY,
    )
    assert tl["project_update"]("alpha", "status", "in_progress", reason="x")["changed"] is False
    assert tl["project_update"]("alpha", "deadline", "2026-12-01", reason="r")["ok"]
    assert tl["project_update"]("alpha", "priority", "high", reason="r")["ok"]
    assert tl["project_update"]("alpha", "status", "dancing", reason="r")["reason"] == "bad_value"
    assert tl["project_update"]("alpha", "colour", "red")["reason"] == "bad_value"
    assert (
        tl["project_update"]("alpha", "owner", "ghost", reason="r")["reason"] == "owner_not_found"
    )
    assert tl["project_update"]("alpha", "deadline", "someday", reason="r")["reason"] == "bad_value"
    assert tl["project_update"]("alpha", "owner", "peat", reason="handover")["ok"]
    assert ctx.repo.get(GUILD, pid).owner == PEAT
    assert (
        tl["project_update"]("alpha", "name", "Alpha 2")["reason"] == "forbidden"
    )  # Toey no longer owns it


def test_members_edit_only_the_overview_and_admins_everything(docs):
    owner_tl, ctx = tools(docs, asker=TOEY)
    pid = owner_tl["project_create"]("Alpha", "tech")["id"]
    p = ctx.repo.get(GUILD, pid)
    p.members.append(P.ProjectMember(person=PEAT))
    ctx.repo.save(GUILD, p)
    member_tl, _ = tools(docs, asker=PEAT)
    assert member_tl["project_update"]("alpha", "overview", "Plan: ship")["ok"]
    assert ctx.repo.get(GUILD, pid).overview == "Plan: ship"
    assert (
        member_tl["project_update"]("alpha", "status", "in_progress", reason="r")["reason"]
        == "forbidden"
    )
    admin_tl, _ = tools(docs, asker=ADMIN, admin=True)
    assert admin_tl["project_update"]("alpha", "status", "on_hold", reason="budget")["ok"]


def test_project_lookup_is_ambiguous_missing_or_hidden(docs):
    tl, ctx = tools(docs)
    tl["project_create"]("Alpha One", "tech")
    tl["project_create"]("Alpha Two", "tech")
    out = tl["project_show"]("alpha")
    assert out["reason"] == "ambiguous" and {m["name"] for m in out["matches"]} == {
        "Alpha One",
        "Alpha Two",
    }
    assert "Alpha One" in out["say"]
    assert tl["project_show"]("zeta")["reason"] == "not_found"
    p = ctx.repo.list(GUILD)[0]
    p.visibility = "members"
    ctx.repo.save(GUILD, p)
    outsider, _ = tools(docs, asker=OUT)
    assert outsider["project_show"](p.name)["reason"] == "not_found"  # hidden = not found
    assert [x["name"] for x in outsider["project_list"]()["projects"]] == ["Alpha Two"]
    admin, _ = tools(docs, asker=OUT, admin=True)
    assert admin["project_show"](p.name)["ok"]


def test_project_list_filters(docs):
    tl, ctx = tools(docs)
    tl["project_create"]("Alpha", "tech")
    tl["project_create"]("Sales push", "business", owner="peat")
    assert tl["project_list"]()["total"] == 2
    assert [p["name"] for p in tl["project_list"](mine=True)["projects"]] == ["Alpha"]
    assert [p["name"] for p in tl["project_list"](department="Business")["projects"]] == [
        "Sales push"
    ]
    assert tl["project_list"](status="completed")["projects"] == []
    assert tools(SQLiteDocStore(":memory:"))[0]["project_list"]()["say"] == t("pm_none", "en")


def test_project_add_task_assigns_dates_and_links_milestones(docs):
    tl, ctx = tools(docs)
    pid = tl["project_create"]("Alpha", "tech")["id"]
    p = ctx.repo.get(GUILD, pid)
    p.milestones.append(P.Milestone(id="m1", title="Demo day"))
    ctx.repo.save(GUILD, p)
    out = tl["project_add_task"](
        "alpha", "Write the pitch", assignee="peat", due="2026-10-30", milestone="demo"
    )
    assert out["ok"] and "hint" not in out
    task = TaskRepo(docs).get(GUILD, out["id"])
    assert (task.project_id, task.assignee_id, task.assignee_email, task.due, task.milestone_id) == (
        pid, 27, "peat@x.com", "2026-10-30", "m1",
    )  # fmt: skip
    nobody = tl["project_add_task"]("alpha", "Clean up")
    assert nobody["ok"] and nobody["hint"] == t("pm_task_unassigned", "en")
    assert TaskRepo(docs).get(GUILD, nobody["id"]).assignee_id == 0
    assert tl["project_add_task"]("alpha", "x", milestone="nope")["reason"] == "bad_value"
    assert tl["project_add_task"]("alpha", "")["reason"] == "bad_value"
    assert tl["project_add_task"]("alpha", "x", assignee="ghost")["reason"] == "owner_not_found"
    stranger, _ = tools(docs, asker=OUT)
    assert stranger["project_add_task"]("alpha", "sneaky")["reason"] == "forbidden"


def test_my_projects_and_tasks_span_projects(docs):
    tl, ctx = tools(docs)
    tl["project_create"]("Alpha", "tech")
    b = tl["project_create"]("Beta", "tech", owner="peat")["id"]
    tl["project_add_task"]("alpha", "one", assignee="toey", due="2026-10-10")
    tl["project_add_task"]("beta", "two", assignee="toey")
    TaskRepo(docs).save(t_("", who=26, name="Toey", title="loose"))
    TaskRepo(docs).save(t_(b, who=27, name="Peat", title="not mine"))
    out = tl["my_projects_and_tasks"]()
    assert [(p["name"], p["role"]) for p in out["projects"]] == [("Alpha", "owner")]
    by = {x["title"]: x for x in out["tasks"]}
    assert set(by) == {"one", "two", "loose"}
    assert by["one"]["project"] == "Alpha" and by["one"]["overdue"] is True
    assert by["loose"]["project"] == ""


def test_project_show_is_a_short_markdown_summary(docs):
    tl, ctx = tools(docs)
    pid = tl["project_create"](
        "Cup", "tech", kind="hackathon", priority="high", deadline="2026-10-01"
    )["id"]
    p = ctx.repo.get(GUILD, pid)
    p.overview = "Build the thing. " * 40
    p.milestones = [
        P.Milestone(id=f"m{i}", title=f"M{i}", due=f"2026-11-0{i}") for i in range(1, 6)
    ]
    p.risks = [P.Risk(id=f"r{i}", title=f"Risk{i}", severity="high") for i in range(1, 6)]
    p.hackathon.competition, p.hackathon.official_deadline = "Builder Cup", "2026-10-18T23:59"
    p.hackathon.deliverables[0].done = True
    ctx.repo.save(GUILD, p)
    tl["project_add_task"]("cup", "late thing", due="2026-10-01")
    md = tl["project_show"]("cup")["markdown"]
    lines = md.split("\n")
    assert len(lines) <= 25 and lines[0].startswith("**Cup**")
    assert "Tasks: 0/1 done, 1 overdue, 1 unassigned" in md
    assert "- M1 (2026-11-01)" in md and "M4" not in md  # only the next three
    assert "Builder Cup" in md and "1/4" in md and "Needs review" in md
    th = tools(docs, lang="th")[0]["project_show"]("cup")["markdown"]
    assert "เจ้าของ" in th and "ต้องทบทวน" in th


# ------------------------------------------------------------------ i18n, prompt, flag
def test_every_pm_string_has_both_languages_and_placeholders_match():
    keys = [k for k in CATALOG if k.startswith("pm_")]
    assert len(keys) > 30
    import re

    for k in keys:
        assert CATALOG[k]["th"] and CATALOG[k]["en"], k
        ph = lambda s: set(re.findall(r"\{(\w+)\}", s))  # noqa: E731
        assert ph(CATALOG[k]["th"]) == ph(CATALOG[k]["en"]), k


def test_prompt_section_flag_and_voice_names():
    assert "project_create" in stable_prompt("th", False, projects=True)
    assert "project_create" not in stable_prompt("th", False)
    assert {
        "project_create",
        "project_update",
        "project_add_task",
        "my_projects_and_tasks",
    } <= ACTION_TOOL_NAMES
    assert (
        not {"project_list", "project_show"} & ACTION_TOOL_NAMES
    )  # reads stay at the room's scope
    from salid.guild_settings import FEATURES, Features

    assert "projects" in FEATURES and Features().projects is True


# ------------------------------------------------------------------ the bot glue
def agent_tools(bot, world, member, is_exec=False, here=None, **kw):
    msg = bot.room_message(world.g, member, here or world.pub)
    k = bot._agent_kwargs(
        msg, world.g, member, is_exec, setting="th", lang="th", question="q", memory=None,
        profile=None, **kw,
    )  # fmt: skip
    return {f.__name__: f for f in k.get("extra_tools", [])}


NAMES = {
    "project_create", "project_update", "project_list", "project_show", "project_add_task",
    "my_projects_and_tasks",
}  # fmt: skip


async def test_tools_are_in_the_agent_and_the_flag_turns_them_off(bot, world):
    me = world.g.members[STAFF]
    assert set(agent_tools(bot, world, me)) >= NAMES
    assert not NAMES & set(agent_tools(bot, world, me, interjection=True))
    set_features(bot, projects=False)
    assert not NAMES & set(agent_tools(bot, world, me))
    assert bot._project_tools(SimpleNamespace(channel=world.pub), world.g, me, False, "th") == []
    set_features(bot, projects=True)
    assert set(agent_tools(bot, world, me)) >= NAMES


async def test_discord_users_become_person_refs_and_rights_come_from_the_bot(bot, world):
    staff, peer = world.g.members[STAFF], world.g.members[PEER]
    staff_tools = agent_tools(bot, world, staff)
    out = staff_tools["project_create"]("Alpha", "tech", owner=peer.display_name)
    p = bot.project_repo.get(GUILD, out["id"])
    assert p.owner.discord_user_id == str(PEER) and p.owner.name == peer.display_name
    assert p.created_by.discord_user_id == str(STAFF) and p.workspace == f"discord-{GUILD}"
    # the owner may change it; staff (not the owner, not an exec) may not
    assert staff_tools["project_update"]("alpha", "name", "Alpha 2")["reason"] == "forbidden"
    boss = world.g.members[BOSS]
    assert agent_tools(bot, world, boss, is_exec=True)["project_update"](
        "alpha", "name", "Alpha 2"
    )["ok"]
    peer_tools = agent_tools(bot, world, peer)
    assert peer_tools["project_add_task"]("alpha 2", "do it", assignee=staff.display_name)["ok"]
    task = bot.project_task_repo.list(GUILD, project_id=p.id)[0]
    assert task.assignee_id == STAFF
    assert [x["title"] for x in staff_tools["my_projects_and_tasks"]()["tasks"]] == ["do it"]


async def test_org_member_linked_to_discord_gives_the_email_and_admin_role(bot, world):
    org = bot_org(bot)
    org.members.append(Member(email="staff@x.com", role="admin", discord_user_id=str(STAFF)))
    OrgStoreFor(bot).save(org)
    staff = world.g.members[STAFF]
    tl = agent_tools(bot, world, staff)
    pid = tl["project_create"]("Alpha", "tech", owner=world.g.members[PEER].display_name)["id"]
    assert bot.project_repo.get(GUILD, pid).created_by.email == "staff@x.com"
    assert tl["project_update"]("alpha", "name", "Alpha 2")["ok"]  # org admin = project admin


def OrgStoreFor(bot):
    from salid.orgs import OrgStore

    return OrgStore(bot.docs)


def bot_org(bot):
    from salid.orgs import Workspace

    store = OrgStoreFor(bot)
    org = store.create("Acme", "owner@x.com")
    store.attach_workspace(
        org.id, Workspace(id=f"discord-{GUILD}", platform="discord", external_id=str(GUILD))
    )
    return store.get(org.id)


# ------------------------------------------------------------------ progress, pictures, plan tools
def test_project_list_cards_and_my_tasks_carry_the_progress_line_and_the_picture_hint(docs):
    tl, ctx = tools(docs)
    pid = tl["project_create"]("Alpha", "tech", deadline="2026-10-30")["id"]
    tl["project_add_task"]("alpha", "one", due="2026-10-20")
    shown = tl["project_list"]()
    assert shown["picture_hint"] == "call project_picture for a chart"
    assert "Real progress 0%" in shown["projects"][0]["progress"]
    assert tl["project_show"]("alpha")["picture_hint"] == "call project_picture for a chart"
    mine = tl["my_projects_and_tasks"]()
    assert mine["picture_hint"] == "call project_picture for a chart"
    assert mine["projects"][0]["progress"].startswith("Real progress")
    assert pid == shown["projects"][0]["id"]
    th = tools(docs, lang="th")[0]["project_list"]()["projects"][0]["progress"]
    assert th.startswith("ความคืบหน้าจริง")


def picture_tools(docs, **kw):
    tl, ctx = tools(docs, **kw)
    from salid.progress_bot import progress_tools

    return {**tl, **{f.__name__: f for f in progress_tools(ctx)}}, ctx


def seed(ctx, docs):
    a = ctx.repo.create(
        GUILD,
        P.apply_template(proj(name="Alpha", deadline="2026-10-20", created_at="")),
    )
    b = ctx.repo.create(
        GUILD, P.apply_template(proj(name="Beta", owner=PEAT, deadline="2026-12-20"))
    )
    hidden = ctx.repo.create(
        GUILD, P.apply_template(proj(name="Secret", owner=PEAT, visibility="members"))
    )
    done = ctx.repo.create(GUILD, P.apply_template(proj(name="Old", status="completed")))
    for pid, title, who, due in (
        (a.id, "late one", TOEY, "2026-10-12"), (a.id, "soon", TOEY, "2026-10-16"),
        (b.id, "peat task", PEAT, "2026-11-01"), (hidden.id, "secret task", TOEY, "2026-11-02"),
    ):  # fmt: skip
        ctx.tasks.save(P.make_task(GUILD, pid, title, assignee=who, due=due))
    return a, b, hidden, done


PNG = b"\x89PNG\r\n\x1a\n"


def test_project_picture_auto_routes_by_the_arguments(docs):
    tl, ctx = picture_tools(docs, admin=False)
    a, b, hidden, done = seed(ctx, docs)
    pic = tl["project_picture"]
    out = pic()  # nothing given: every running project
    assert out["ok"] and out["attached"] and out["what"] == "portfolio"
    assert len(out["summary_lines"]) == 2  # Alpha, Beta: not the completed one, not the hidden one
    assert out["summary_lines"][0].startswith("Alpha: Real progress 0%")  # the worst first
    assert ctx.outbox[-1][0] == "progress-projects.png" and ctx.outbox[-1][1][:8] == PNG
    assert (
        pic(project="beta")["what"] == "project"
        and ctx.outbox[-1][0] == f"progress-project-{b.id}.png"
    )
    who = pic(person="peat")
    assert who["what"] == "person" and ctx.outbox[-1][0] == "progress-person.png"
    assert pic(project="alpha", person="peat")["what"] == "person"  # a person wins
    assert pic(what="portfolio", project="alpha")["what"] == "portfolio"  # explicit wins
    assert pic(what="project", project="alpha")["what"] == "project"
    assert tl["project_picture"].outbox is ctx.outbox
    assert all(png[:8] == PNG for _, png in ctx.outbox)


def test_project_picture_scoping_and_errors(docs):
    tl, ctx = picture_tools(docs, asker=PEAT, people=(TOEY, PEAT, ADMIN))
    a, b, hidden, done = seed(ctx, docs)
    # an outsider (not owner, not on the team) cannot open "Secret" at all
    out = picture_tools(docs, asker=OUT, people=(OUT, TOEY, PEAT))[0]
    ghost = out["project_picture"](project="secret")
    assert ghost["ok"] is False and ghost["reason"] == "not_found"
    nobody = out["project_picture"](person="toey")
    assert nobody["ok"]  # Toey has work in readable projects
    assert out["project_picture"](what="sunset")["reason"] == "bad_value"
    assert out["project_picture"](person="zzz")["reason"] == "owner_not_found"
    assert tl["project_picture"](project="alp")["ok"]
    assert tl["project_picture"](project="e")["reason"] == "ambiguous"  # Alpha, Beta, Secret, Old
    empty_tl, _ = picture_tools(SQLiteDocStore(":memory:"))
    assert empty_tl["project_picture"]()["reason"] == "empty"
    assert "No projects yet" in empty_tl["project_picture"]()["say"]


def test_person_picture_hides_tasks_of_unreadable_projects(docs):
    tl, ctx = picture_tools(docs, asker=OUT, people=(OUT, TOEY, PEAT))
    seed(ctx, docs)
    out = tl["project_picture"](person="toey")
    assert out["ok"] and all("Secret" not in line for line in out["summary_lines"])


def test_a_picture_failure_is_a_sentence_not_an_exception(docs, monkeypatch):
    tl, ctx = picture_tools(docs)
    seed(ctx, docs)
    import salid.progress_bot as PB

    monkeypatch.setattr(
        PB, "portfolio_png", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    )
    out = tl["project_picture"]()
    assert out["ok"] is False and "could not draw" in out["say"] and not ctx.outbox


def test_prompt_lines_for_pictures_and_plans_and_voice_names():
    th = stable_prompt("th", False, projects=True)
    assert "project_picture" in th and "project_import_plan" in th and "confirm=False" in th
    assert "two separate facts" in th and "สองเรื่องแยกกัน" in th
    assert "project_import_plan" in ACTION_TOOL_NAMES and "project_picture" not in ACTION_TOOL_NAMES


async def test_picture_and_plan_tools_are_in_the_agent_and_follow_the_flag(bot, world):
    me = world.g.members[STAFF]
    names = {"project_picture", "project_import_plan"}
    got = agent_tools(bot, world, me)
    assert names <= set(got)
    assert (
        got["project_picture"].outbox is got["project_import_plan"].outbox
    )  # one outbox per request
    assert not names & set(agent_tools(bot, world, me, interjection=True))
    set_features(bot, projects=False)
    assert not names & set(agent_tools(bot, world, me))
    set_features(bot, projects=True)


async def test_files_reach_the_plan_tool_through_the_agent_kwargs(bot, world, monkeypatch):
    me = world.g.members[STAFF]
    from salid import plan_import as PI

    seen = []

    def gen(client, model, contents, config, delays=None):
        seen.append(contents)
        raise RuntimeError("stop here")

    monkeypatch.setattr(PI, "generate", gen)
    from salid.qa import FileInput

    f = FileInput(0, "gantt.png", "image/png", data=b"\x89PNG....")
    got = agent_tools(bot, world, me, files=[f])
    out = got["project_import_plan"]()
    assert out["ok"] is False and seen and any(not isinstance(p, str) for p in seen[0])
