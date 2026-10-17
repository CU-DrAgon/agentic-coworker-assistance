"""Gantt / plan import: reading the plan (xlsx text, picture, typed), Buddhist and day-first
dates, weeks relative to the plan start, the preview, the confirm that creates milestones and
numbered tasks with owners, the owner question for names nobody matches, re-import without
duplicates, the one-hour draft, the typed "ยืนยัน" and soft failures. The model is a fake."""

import io
import json
import re
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from conftest import GUILD
from fake_firestore import FakeFirestore

from salid import plan_import as PI
from salid import progress_bot as PB
from salid import projects as P
from salid.coo_store import FirestoreDocStore, SQLiteDocStore
from salid.qa import FileInput
from salid.tasks import TaskRepo

NOW = datetime(2026, 10, 14, 12, 0).astimezone()  # any aware local time
TOEY = P.Person(email="toey@x.com", discord_user_id="26", name="Toey")
PEAT = P.Person(email="peat@x.com", discord_user_id="27", name="Peat")
PLAN = {
    "project_name": "Launch Salid",
    "plan_start": "2026-10-05",
    "phases": [
        {"title": "Build", "start": "2026-10-05", "end": "2026-10-16"},
        {"title": "Ship", "start": "", "end": ""},
    ],
    "steps": [
        {"n": 1, "title": "Write the pitch", "phase": "Build", "owner": "Toey", "start": "2026-10-05", "due": "2026-10-09", "pct_done": 100},
        {"n": 2, "title": "Record the demo", "phase": "Build", "owner": "Peat", "start": "2026-10-10", "due": "2026-10-16", "pct_done": 40, "depends_on": [1]},
        {"n": 3, "title": "Submit", "phase": "Ship", "owner": "Nobody Known", "start": "2026-10-17", "due": "2026-10-20", "depends_on": [2, 9, 3]},
        {"n": 4, "title": "Celebrate", "phase": "Ship", "owner": "", "due": "2026-10-21"},
    ],
    "unreadable": ["the last column is cut off"],
}  # fmt: skip


@pytest.fixture(params=["sqlite", "firestore"])
def docs(request):
    if request.param == "sqlite":
        return SQLiteDocStore(":memory:")
    return FirestoreDocStore(client=FakeFirestore())


def fake_model(monkeypatch, reply, calls=None):
    def gen(client, model, contents, config, delays=None):
        if calls is not None:
            calls.append(contents)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(text=reply if isinstance(reply, str) else json.dumps(reply))

    monkeypatch.setattr(PI, "generate", gen)


def xlsx_file() -> FileInput:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Step", "Task", "Owner", "Start", "End", "%"])
    ws.append([1, "Write the pitch", "Toey", "5/10/2026", "9/10/2026", 100])
    ws.append([2, "Record the demo", "Peat", "10/10/2026", "16/10/2026", 40])
    buf = io.BytesIO()
    wb.save(buf)
    return FileInput(0, "plan.xlsx", "text/plain", raw=buf.getvalue())  # text part not filled


def make_ctx(docs, asker=TOEY, files=(), people=(TOEY, PEAT), lang="en", admin=False, now=NOW):
    def resolve(name):
        low = name.strip().lstrip("@").casefold()
        return [p for p in people if low and low in p.name.casefold()]

    cards = []
    ctx = P.ProjectsCtx(
        guild_id=GUILD, asker=asker, is_admin=admin, lang=lang,
        repo=P.ProjectRepo(docs, clock=lambda: ctx.now()), tasks=TaskRepo(docs),
        now=lambda: now, resolve=resolve, workspace="w", docs=docs, llm=object(),
        model="fake", files=list(files),
        on_assigned_many=lambda ts: cards.append([x.id for x in ts]),
    )  # fmt: skip
    ctx.cards = cards
    return ctx


def tools_of(ctx):
    return {f.__name__: f for f in PB.progress_tools(ctx)}


# ------------------------------------------------------------------ reading
def test_xlsx_text_reaches_the_model_and_the_plan_is_cleaned(monkeypatch):
    calls = []
    fake_model(monkeypatch, PLAN, calls)
    draft = PI.extract_plan(object(), "m", [xlsx_file()], "en")
    body = "\n".join(p for p in calls[0] if isinstance(p, str))
    assert "Write the pitch" in body and "Record the demo" in body and "plan.xlsx" in body
    assert draft.error == "" and draft.project_name == "Launch Salid"
    assert [s.n for s in draft.steps] == [1, 2, 3, 4]
    assert draft.steps[2].depends_on == [2]  # unknown 9 and itself dropped
    assert draft.notes == ["the last column is cut off"]


def test_an_image_and_a_pdf_go_to_the_model_as_bytes_and_typed_text_too(monkeypatch):
    calls = []
    fake_model(monkeypatch, PLAN, calls)
    png = FileInput(0, "gantt.png", "image/png", data=b"\x89PNG....")
    pdf = FileInput(1, "plan.pdf", "application/pdf", data=b"%PDF-1.4 x")
    skipped = FileInput(2, "big.png", "image/png", data=b"x", skip="over limit")
    PI.extract_plan(object(), "m", [png, pdf, skipped], "th", text="step 1 do it")
    parts = calls[0]
    blobs = [p for p in parts if not isinstance(p, str)]
    assert len(blobs) == 2 and {b.inline_data.mime_type for b in blobs} == {
        "image/png",
        "application/pdf",
    }
    assert any(isinstance(p, str) and "step 1 do it" in p for p in parts)
    assert "Thai" in parts[-1]  # the prompt names the reply language


def test_nothing_to_read_and_model_failures_are_soft(monkeypatch):
    fake_model(monkeypatch, PLAN)
    assert PI.extract_plan(object(), "m", [], "en").error == "no_input"
    assert (
        PI.extract_plan(object(), "m", [FileInput(0, "a.png", "image/png", skip="x")], "en").error
        == "no_input"
    )
    fake_model(monkeypatch, RuntimeError("boom"))
    assert PI.extract_plan(object(), "m", [], "en", text="x").error == "unavailable"
    fake_model(monkeypatch, "not json at all")
    assert PI.extract_plan(object(), "m", [], "en", text="x").error == "unreadable"
    fake_model(monkeypatch, {"project_name": "x", "steps": []})
    assert PI.extract_plan(object(), "m", [], "en", text="x").error == "empty"


def test_dates_buddhist_years_day_first_and_garbage():
    assert PI.norm_date("2026-10-17") == "2026-10-17"
    assert PI.norm_date("2569-10-17") == "2026-10-17"  # Buddhist era
    assert PI.norm_date("17/10/2569") == "2026-10-17"
    assert PI.norm_date("17/10/2026") == "2026-10-17"  # day first
    assert PI.norm_date("5-1-2026") == "2026-01-05"
    assert (
        PI.norm_date("2026-02-30") == "" and PI.norm_date("soon") == "" and PI.norm_date("") == ""
    )


def test_buddhist_dates_in_the_reply_become_gregorian(monkeypatch):
    plan = {"steps": [{"n": 1, "title": "a", "start": "2569-10-05", "due": "16/10/2569"}]}
    fake_model(monkeypatch, plan)
    s = PI.extract_plan(object(), "m", [], "th", text="x").steps[0]
    assert (s.start, s.due) == ("2026-10-05", "2026-10-16")


def test_weeks_count_from_the_plan_start_or_are_left_empty_and_flagged(monkeypatch):
    steps = [
        {"n": 1, "title": "a", "start_week": 3, "due_week": 3},
        {"n": 2, "title": "b", "start_week": 1, "due_week": 2},
    ]
    fake_model(monkeypatch, {"plan_start": "2026-10-05", "steps": steps})
    a, b = PI.extract_plan(object(), "m", [], "en", text="x").steps
    assert (a.start, a.due) == ("2026-10-19", "2026-10-25")  # week 3 = Oct 19-25
    assert (b.start, b.due) == ("2026-10-05", "2026-10-18")
    fake_model(monkeypatch, {"steps": steps})
    d = PI.extract_plan(object(), "m", [], "en", text="x")
    assert all(not s.start and not s.due for s in d.steps)
    assert any("Step 1" in n and "weeks" in n for n in d.notes)
    thai = PI.extract_plan(object(), "m", [], "th", text="x")
    assert any("สัปดาห์ที่" in n for n in thai.notes)


def test_numbers_dates_and_percentages_are_checked(monkeypatch):
    steps = [
        {"n": 1, "title": "a", "start": "2026-10-20", "due": "2026-10-10", "pct_done": 250},
        {"n": 1, "title": "dup number"},
        {"n": 0, "title": "no number", "due": "tomorrowish"},
        {"title": "   "},
    ]
    fake_model(monkeypatch, {"steps": steps})
    d = PI.extract_plan(object(), "m", [], "en", text="x")
    assert [(s.n, s.title) for s in d.steps] == [(1, "a"), (2, "dup number"), (3, "no number")]
    assert d.steps[0].pct_done == 100 and d.steps[0].start == ""  # start after due: dropped
    assert d.steps[2].due == ""
    assert any("tomorrowish" in n for n in d.notes) and any("start is after" in n for n in d.notes)


def test_a_phase_without_dates_takes_the_span_of_its_steps(monkeypatch):
    fake_model(monkeypatch, PLAN)
    ship = PI.extract_plan(object(), "m", [], "en", text="x").phases[1]
    assert (ship.start, ship.end) == ("2026-10-17", "2026-10-21")


def test_confirm_words():
    for ok in ("ยืนยัน", "ยืนยันค่ะ", "confirm", "Confirm!", "ตกลง", "yes", "confirm plan"):
        assert PI.parse_plan_confirm(ok), ok
    for no in ("ยืนยันตัวตน", "confirm the budget", "", "ไม่", "no", "confirmation"):
        assert not PI.parse_plan_confirm(no), no


# ------------------------------------------------------------------ preview then confirm
def preview(ctx, monkeypatch, plan=PLAN, **kw):
    fake_model(monkeypatch, plan)
    return tools_of(ctx)["project_import_plan"](text="the plan", **kw)


def test_the_first_call_only_previews_and_keeps_the_draft(docs, monkeypatch):
    ctx = make_ctx(docs)
    out = preview(ctx, monkeypatch)
    assert out["ok"] and out["awaiting_confirm"] and out["steps"] == 4
    md = out["preview"]
    assert (
        "Launch Salid" in md
        and "| 2 | Record the demo | Peat | 2026-10-10 | 2026-10-16 | 2 d left | 40% |" in md
    )
    assert "reply “confirm”" in md and "last column is cut off" in md
    assert "no member called “Nobody Known”" in md and "1 steps have no owner" in md
    assert (
        ctx.repo.list(GUILD) == [] and TaskRepo(docs).list(GUILD, status=None) == []
    )  # nothing yet
    assert docs.get(PI.PLAN_DRAFTS, GUILD, "26")["plan"]["project_name"] == "Launch Salid"


def test_the_preview_is_capped_at_thirty_rows_and_thai(docs, monkeypatch):
    plan = {"steps": [{"n": i, "title": f"step {i}", "due": "2026-10-20"} for i in range(1, 46)]}
    out = preview(make_ctx(docs, lang="th"), monkeypatch, plan)
    assert len(re.findall(r"^\| \d+ \|", out["preview"], re.M)) == 30
    assert "| 31 |" not in out["preview"] and "…และอีก 15 ขั้น" in out["preview"]
    assert "ตอบ “ยืนยัน”" in out["preview"]


def test_confirm_creates_project_milestones_and_numbered_tasks(docs, monkeypatch):
    ctx = make_ctx(docs)
    preview(ctx, monkeypatch)
    out = tools_of(ctx)["project_import_plan"](confirm=True)
    assert out["ok"] and out["attached"] and "Imported the plan into “Launch Salid”" in out["say"]
    assert "4 new tasks, 0 updated, 2 milestones" in out["say"]
    assert out["unresolved_owners"] == ["Nobody Known"]
    assert "Real progress" in out["progress"] and "1. Write the pitch" in out["markdown"]
    p = ctx.repo.list(GUILD)[0]
    assert p.name == "Launch Salid" and p.owner == TOEY and p.deadline == "2026-10-21"
    assert [(m.id, m.title, m.due) for m in p.milestones] == [
        ("m1", "Build", "2026-10-16"),
        ("m2", "Ship", "2026-10-21"),
    ]
    tasks = {x.step: x for x in TaskRepo(docs).list(GUILD, status=None, project_id=p.id)}
    assert sorted(tasks) == [1, 2, 3, 4]
    t1, t2, t3, t4 = tasks[1], tasks[2], tasks[3], tasks[4]  # noqa: F841
    assert (t1.status, t1.pct_done, t1.start, t1.due, t1.milestone_id) == (
        "done",
        100,
        "2026-10-05",
        "2026-10-09",
        "m1",
    )
    assert t1.assignee_id == 26 and t1.confirm == "accepted"  # the asker's own: no card
    assert t2.assignee_id == 27 and t2.assigner_id == 26 and t2.confirm == "" and t2.pct_done == 40
    assert t2.depends_on == [t1.id] and t3.depends_on == [t2.id]
    assert t2.created_from == "plan" and t2.milestone_id == "m1" and t3.milestone_id == "m2"
    assert out["attached"] and ctx.outbox and ctx.outbox[0][1][:8] == b"\x89PNG\r\n\x1a\n"
    assert ctx.outbox[0][0] == f"plan-{p.id}.png"
    assert tools_of(ctx)["project_import_plan"].outbox is ctx.outbox
    assert docs.get(PI.PLAN_DRAFTS, GUILD, "26") is None  # the draft is used up


def test_owners_resolved_unresolved_and_the_owner_question(docs, monkeypatch):
    ctx = make_ctx(docs)
    preview(ctx, monkeypatch)
    tools_of(ctx)["project_import_plan"](confirm=True)
    tasks = {x.step: x for x in TaskRepo(docs).list(GUILD, status=None)}
    # Peat gets a card (assigner = the asker); "Nobody Known" is an owner question to the asker
    assert tasks[3].assignee_id == 0 and tasks[3].assigner_id == 26
    assert "Nobody Known" in tasks[3].note
    # no owner named at all: left alone, nobody is asked
    assert tasks[4].assignee_id == 0 and tasks[4].assigner_id == 0
    assert len(ctx.cards) == 1 and set(ctx.cards[0]) == {tasks[2].id, tasks[3].id}


def test_ambiguous_owner_counts_as_unresolved(docs, monkeypatch):
    twin = P.Person(email="peat2@x.com", discord_user_id="28", name="Peat Two")
    ctx = make_ctx(docs, people=(TOEY, PEAT, twin))
    preview(ctx, monkeypatch)
    tools_of(ctx)["project_import_plan"](confirm=True)
    t2 = [x for x in TaskRepo(docs).list(GUILD, status=None) if x.step == 2][0]
    assert t2.assignee_id == 0 and t2.assigner_id == 26


def test_reimport_updates_matching_steps_and_adds_only_new_ones(docs, monkeypatch):
    ctx = make_ctx(docs)
    preview(ctx, monkeypatch)
    tools_of(ctx)["project_import_plan"](confirm=True)
    again = json.loads(json.dumps(PLAN))
    again["steps"][1].update(pct_done=70, due="2026-10-18")
    again["steps"][0]["title"] = "  write THE pitch "  # same step, different spacing / case
    again["steps"].append(
        {"n": 5, "title": "Retro", "phase": "Ship", "owner": "Toey", "due": "2026-10-25"}
    )
    preview(ctx, monkeypatch, again)
    out = tools_of(ctx)["project_import_plan"](confirm=True)
    assert "1 new tasks, 4 updated" in out["say"]
    assert len(ctx.repo.list(GUILD)) == 1  # the same project, found by the plan's name
    tasks = TaskRepo(docs).list(GUILD, status=None)
    assert len(tasks) == 5 and len([m for m in ctx.repo.list(GUILD)[0].milestones]) == 2
    t2 = [x for x in tasks if x.step == 2][0]
    assert (t2.pct_done, t2.due) == (70, "2026-10-18")


def test_a_named_project_gets_the_plan_and_needs_membership(docs, monkeypatch):
    ctx = make_ctx(docs)
    owned = ctx.repo.create(
        GUILD,
        P.apply_template(
            P.Project(
                name="Hack Cup",
                owner=TOEY,
                department="tech",
                status="in_progress",
                created_by=TOEY,
            )
        ),
    )
    preview(ctx, monkeypatch, project="hack")
    out = tools_of(ctx)["project_import_plan"](confirm=True)
    assert out["ok"] and out["id"] == owned.id and len(ctx.repo.list(GUILD)) == 1
    outsider = P.Person(email="out@x.com", discord_user_id="99", name="Outsider")
    octx = make_ctx(docs, asker=outsider, people=(TOEY, PEAT, outsider))
    preview(octx, monkeypatch, project="hack")
    no = tools_of(octx)["project_import_plan"](confirm=True)
    assert no["ok"] is False and no["reason"] == "forbidden"
    assert docs.get(PI.PLAN_DRAFTS, GUILD, "99") is not None  # still waiting, nothing created
    assert [x for x in TaskRepo(docs).list(GUILD, status=None) if x.project_id == owned.id][
        0
    ].step in (1, 2, 3, 4)


def test_a_project_name_that_does_not_exist_is_created_with_that_name(docs, monkeypatch):
    ctx = make_ctx(docs)
    preview(ctx, monkeypatch, project="My own name")
    out = tools_of(ctx)["project_import_plan"](confirm=True, project="My own name")
    assert out["ok"] and ctx.repo.list(GUILD)[0].name == "My own name"


def test_confirm_without_a_draft_and_expired_drafts(docs, monkeypatch):
    ctx = make_ctx(docs)
    none = tools_of(ctx)["project_import_plan"](confirm=True)
    assert (
        none["ok"] is False and none["reason"] == "no_draft" and "No plan is waiting" in none["say"]
    )
    preview(ctx, monkeypatch)
    late = make_ctx(docs, now=NOW + timedelta(minutes=61))
    assert tools_of(late)["project_import_plan"](confirm=True)["reason"] == "no_draft"
    assert docs.get(PI.PLAN_DRAFTS, GUILD, "26") is None  # an old draft is deleted
    preview(ctx, monkeypatch)
    fresh = make_ctx(docs, now=NOW + timedelta(minutes=59))
    assert tools_of(fresh)["project_import_plan"](confirm=True)["ok"]


def test_drafts_are_per_person(docs, monkeypatch):
    a, b = make_ctx(docs), make_ctx(docs, asker=PEAT)
    preview(a, monkeypatch)
    assert tools_of(b)["project_import_plan"](confirm=True)["reason"] == "no_draft"


def test_soft_errors_come_back_as_words(docs, monkeypatch):
    ctx = make_ctx(docs)
    fake_model(monkeypatch, PLAN)
    assert tools_of(ctx)["project_import_plan"]()["reason"] == "no_input"  # no file, no text
    fake_model(monkeypatch, "garbage")
    bad = tools_of(ctx)["project_import_plan"](text="x")
    assert bad["ok"] is False and "could not read this plan" in bad["say"]
    fake_model(monkeypatch, {"steps": []})
    assert tools_of(ctx)["project_import_plan"](text="x")["reason"] == "empty"
    th = make_ctx(docs, lang="th")
    assert "ส่งไฟล์แผนงาน" in tools_of(th)["project_import_plan"]()["say"]


def test_file_index_picks_one_attachment(docs, monkeypatch):
    calls = []
    fake_model(monkeypatch, PLAN, calls)
    a = FileInput(0, "a.png", "image/png", data=b"A")
    b = FileInput(1, "b.png", "image/png", data=b"B")
    ctx = make_ctx(docs, files=[a, b])
    tools_of(ctx)["project_import_plan"](file_index=1)
    assert [p.inline_data.data for p in calls[0] if not isinstance(p, str)] == [b"B"]
    tools_of(ctx)["project_import_plan"]()
    assert len([p for p in calls[1] if not isinstance(p, str)]) == 2


# ------------------------------------------------------------------ the typed "ยืนยัน"
class Msg:
    def __init__(self, text, author, guild=None):
        self.content, self.author, self.guild = text, author, guild
        self.mentions, self.replies, self.channel = [], [], SimpleNamespace(id=5)
        self.reference = None

    async def reply(self, text=None, **kw):
        self.replies.append((text, kw))


async def test_typed_confirm_creates_the_plan_without_the_model(docs, monkeypatch):
    from salid.progress_bot import ProgressMixin

    ctx = make_ctx(docs)
    preview(ctx, monkeypatch)
    me = SimpleNamespace(id=26, display_name="Toey")
    guild = SimpleNamespace(id=GUILD)

    class FakeBot(ProgressMixin):
        docs = None
        s = SimpleNamespace(now=lambda: NOW)

        def _command_text(self, message):
            return message.content

        def _projects_on(self, gid):
            return True

        def _person_lang(self, gid, uid, text=""):
            return "th"

        def is_exec_member(self, m):
            return False

        def _project_ctx(self, message, g, member, is_exec, lang, files=()):
            c = make_ctx(docs, lang=lang)
            self.last = c
            return c

    bot = FakeBot()
    bot.docs = docs
    msg = Msg("ยืนยัน", me, guild)
    assert await bot._plan_confirm_command(msg) is True
    text, kw = msg.replies[0]
    assert "นำเข้าแผนใน “Launch Salid”" in text and "ความคืบหน้าจริง" in text
    assert kw["files"] and kw["files"][0].filename.startswith("plan-p-")
    assert len(TaskRepo(docs).list(GUILD, status=None)) == 4
    again = Msg("ยืนยัน", me, guild)  # nothing waits any more: the message goes on to the model
    assert await bot._plan_confirm_command(again) is False and not again.replies
    other = Msg("พรุ่งนี้ประชุม", me, guild)
    assert await bot._plan_confirm_command(other) is False
