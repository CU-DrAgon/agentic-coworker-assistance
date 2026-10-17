# ruff: noqa: F811, F401
"""Tasks with no deadline: the card hint, update_task(due=), the one-time question, the reports."""

from datetime import date
from zoneinfo import ZoneInfo

from conftest import GUILD
from test_coordinator import BOSS, INTERN, STAFF, bot, person, world  # noqa: F401  (fixtures)
from test_task_handshake import coo, hs, new_task  # noqa: F401  (fixtures)

from salid import reports as rep
from salid import task_handshake as H
from salid.coo_store import STATE
from salid.i18n import t
from salid.nodue_bot import NoDeadlineMixin
from salid.tasks import Task, no_deadline_due, working_days_between

BKK = ZoneInfo("Asia/Bangkok")


def nodue_task(hs, **kw):
    task = new_task(hs, **kw)
    task.due = ""
    return hs.tasks.save(task)


WED = date(2026, 10, 14)


def mk(created="2026-10-09T09:00:00+07:00", **kw):
    base = dict(guild_id=GUILD, title="x", assignee_id=STAFF, created_at=created)
    return Task(**{**base, **kw})


def test_working_days_skip_the_weekend():
    assert working_days_between(date(2026, 10, 9), date(2026, 10, 12)) == 1  # Fri -> Mon
    assert working_days_between(date(2026, 10, 10), date(2026, 10, 14)) == 3  # Sat -> Wed
    assert working_days_between(date(2026, 10, 14), date(2026, 10, 14)) == 0


def test_no_deadline_due_rules():
    assert no_deadline_due([mk()], WED, BKK) != []  # Fri -> Wed = 3 working days
    assert no_deadline_due([mk()], date(2026, 10, 13), BKK) == []  # only 2
    assert no_deadline_due([mk()], date(2026, 10, 12), BKK) == []  # over a weekend: 1
    assert no_deadline_due([mk(due="2026-10-20")], WED, BKK) == []
    assert no_deadline_due([mk(nodue_asked_at="2026-10-13T10:00:00+07:00")], WED, BKK) == []
    assert no_deadline_due([mk(assignee_id=0)], WED, BKK) == []
    assert no_deadline_due([mk(status="done")], WED, BKK) == []
    assert no_deadline_due([mk(created="")], WED, BKK) == []


def test_accepted_later_counts_from_confirm_at():
    late = mk(created="2026-10-01T09:00:00+07:00", confirm="accepted",
              confirm_at="2026-10-13T09:00:00+07:00")  # fmt: skip
    assert no_deadline_due([late], WED, BKK) == []
    old = mk(created="2026-10-01T09:00:00+07:00", confirm="pending",
             confirm_at="2026-10-13T09:00:00+07:00")  # fmt: skip
    assert no_deadline_due([old], WED, BKK) != []


def test_old_task_documents_load_without_the_field():
    assert Task(**{"id": "abcd1234", "guild_id": GUILD, "title": "x"}).nodue_asked_at == ""


def test_card_hint_only_when_no_due(hs):
    task = nodue_task(hs)
    text = H.card_text(task, "en", channel="gen")
    assert "press ✏️ Edit" in text and f"task #{task.id} due Friday" in text
    assert "ไม่มีกำหนด" in H.card_text(task, "th", channel="gen")
    dated = new_task(hs, title="อีกงาน")
    assert "ℹ️" not in H.card_text(dated, "en", channel="gen")


def test_update_task_sets_due(hs):
    task = nodue_task(hs, assigner=BOSS)
    tools, _ = coo(hs, INTERN)
    assert "not allowed" in tools.update_task(task.id, due="friday")["error"]
    tools, _ = coo(hs, STAFF)
    bad = tools.update_task(task.id, due="ไว้ก่อน")
    assert "cannot read" in bad["error"] and "friday" in bad["hint"]
    assert hs.tasks.get(GUILD, task.id).due == ""
    assert tools.update_task(task.id, due="friday")["updated"]["due"] == "2026-10-16"
    tools, _ = coo(hs, BOSS)  # the giver may too
    tools.update_task(task.id, due="2026-10-20")
    assert hs.tasks.get(GUILD, task.id).due == "2026-10-20"


async def test_the_tick_asks_the_assignee_once(hs, world):
    task = nodue_task(hs, created_at="2026-10-09T09:00:00+07:00")
    assert await hs._nodue_due() == 1
    ((text, _),) = world.staff.dms
    assert f"#{task.id}" in text and "2026-10-20" in text  # the due_formats hint
    assert hs.tasks.get(GUILD, task.id).nodue_asked_at
    assert not world.boss.dms  # the giver hears nothing
    hs._nodue_last = float("-inf")
    assert await hs._nodue_due() == 0 and len(world.staff.dms) == 1
    assert hs.docs.get(STATE, GUILD, f"nodue:{task.id}") is not None


async def test_a_young_task_is_left_alone(hs, world):
    nodue_task(hs, created_at="2026-10-13T09:00:00+07:00")
    assert await hs._nodue_due() == 0 and not world.staff.dms


def test_company_report_section_only_when_there_are_some():
    base = dict(day="2026-10-14", ledger=rep.ledger_brief.__defaults__ and {}, channels=[],
                people=[], unassigned_open=[])  # fmt: skip
    base["ledger"] = {"today_by_currency": {}, "today_count": 0, "month_by_currency": {},
                      "review": 0, "duplicates": 0}  # fmt: skip
    assert "🗓️" not in rep.render_plain(rep.CompanyInput(**base), "en")
    rows = [f"Ann: job {i} (#{i})" for i in range(13)]
    text = rep.render_plain(rep.CompanyInput(**base, no_deadline=rows), "en")
    assert t("rep_nodue", "en") in text and "job 9" in text and "job 10" not in text
    assert "+3 more" in text


def test_personal_report_section(hs):
    task = nodue_task(hs)
    data = rep.person_day(guild_id=GUILD, user_id=STAFF, day=WED, zone=BKK, memory=hs.memory_store,
                          tasks=hs.tasks, channels={}, include_memory=False)  # fmt: skip
    assert t("rep_nodue", "en") in rep.render_personal(data, "en")
    hs.tasks.save(task.model_copy(update={"due": "2026-10-20"}))
    data = rep.person_day(guild_id=GUILD, user_id=STAFF, day=WED, zone=BKK, memory=hs.memory_store,
                          tasks=hs.tasks, channels={}, include_memory=False)  # fmt: skip
    assert t("rep_nodue", "en") not in rep.render_personal(data, "en")
