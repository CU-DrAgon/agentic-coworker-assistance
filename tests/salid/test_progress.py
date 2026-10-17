"""Progress vs time: the pure math (all states, no deadline, no tasks, overdue), the one-line text
in Thai and English, step order, and the places it shows up (summary, list cards, stats, insights,
task card text). Fakes only."""

from datetime import date

import pytest

from salid import projects as P
from salid.progress import (
    ordered_tasks,
    progress,
    progress_line,
    task_pct,
)
from salid.task_handshake import card_text
from salid.tasks import Task

TODAY = date(2026, 10, 17)
TOEY = P.Person(email="toey@x.com", discord_user_id="26", name="Toey")


def proj(**kw) -> P.Project:
    base = dict(
        id="p1", name="Alpha", owner=TOEY, status="in_progress",
        created_at="2026-10-01T09:00:00+07:00", deadline="2026-10-27",
    )  # fmt: skip
    return P.Project(**{**base, **kw})


def tk(n=0, status="open", due="", **kw) -> Task:
    return Task(guild_id=1, project_id="p1", title=f"task {n}", status=status, due=due, **kw)


def tasks_done(done: int, total: int) -> list[Task]:
    return [tk(i, "done" if i < done else "open") for i in range(total)]


def test_done_pct_counts_tasks_then_milestones_else_none():
    p = proj()
    assert progress(p, tasks_done(2, 5), TODAY).done_pct == 40
    assert progress(p, [], TODAY).done_pct is None
    ms = proj(
        milestones=[P.Milestone(id="m1", title="a", status="done"), P.Milestone(id="m2", title="b")]
    )
    assert progress(ms, [], TODAY).done_pct == 50
    assert progress(ms, tasks_done(1, 4), TODAY).done_pct == 25  # tasks win over milestones


def test_time_pct_between_creation_and_deadline_and_days_left():
    pr = progress(proj(), tasks_done(0, 1), TODAY)  # Oct 1 -> Oct 27, today Oct 17
    assert pr.time_pct == 62 and pr.days_left == 10
    assert progress(proj(deadline=""), [], TODAY).time_pct is None
    assert progress(proj(deadline=""), [], TODAY).days_left is None
    before = progress(
        proj(created_at="2026-10-20T00:00:00+07:00", deadline="2026-10-27"), [], TODAY
    )
    assert before.time_pct == 0  # not started yet


def test_an_earlier_task_start_moves_the_beginning():
    early = [tk(1, start="2026-09-17")]  # plan began a month ago, deadline in 10 days
    assert progress(proj(), early, TODAY).time_pct == round(100 * 30 / 40)


def test_task_start_is_the_fallback_when_created_at_is_empty():
    p = proj(created_at="")
    assert progress(p, [tk(1, start="2026-10-07")], TODAY).time_pct == 50
    assert progress(p, [], TODAY).time_pct == 0  # no start at all: the deadline alone says 0 %


@pytest.mark.parametrize(
    "done,created,deadline,status,state",
    [
        (4, "2026-10-01", "2026-10-27", "in_progress", "on_track"),  # 40% vs 62%: within 20
        (3, "2026-10-01", "2026-12-20", "in_progress", "on_track"),
        (1, "2026-09-01", "2026-12-01", "in_progress", "behind"),  # 10% done, ~41% time, 45 d left
        (1, "2026-10-01", "2026-10-22", "in_progress", "at_risk"),  # behind and 5 days left
        (1, "2026-10-01", "2026-10-10", "in_progress", "at_risk"),  # overdue, not completed
        (5, "2026-10-01", "2026-10-10", "in_progress", "done"),  # everything done
        (1, "2026-10-01", "2026-10-10", "completed", "done"),  # status completed
        (1, "2026-10-01", "2026-10-10", "cancelled", "done"),
        (1, "2026-10-01", "", "in_progress", "no_deadline"),
    ],
)
def test_states(done, created, deadline, status, state):
    p = proj(created_at=created + "T00:00:00+07:00", deadline=deadline, status=status)
    assert progress(p, tasks_done(done, 5), TODAY).state == state


def test_no_tasks_and_no_milestones_is_on_track_until_the_deadline_passes():
    assert progress(proj(), [], TODAY).state == "on_track"
    assert progress(proj(deadline="2026-10-10"), [], TODAY).state == "at_risk"
    assert progress(proj(deadline="2026-10-10"), [], TODAY).days_left == -7


def test_next_due_is_the_nearest_open_task_or_milestone():
    p = proj(milestones=[P.Milestone(id="m1", title="Demo", due="2026-10-20")])
    ts = [tk(1, due="2026-10-25"), tk(2, due="2026-10-19"), tk(3, "done", due="2026-10-18")]
    nxt = progress(p, ts, TODAY).next_due
    assert nxt == {"kind": "task", "title": "task 2", "due": "2026-10-19", "days": 2}
    assert progress(p, [tk(1, due="2026-10-25")], TODAY).next_due["kind"] == "milestone"
    assert progress(proj(), [], TODAY).next_due is None


def test_task_gaps_pct_and_step_order():
    ts = [
        tk(3, due="2026-10-30", step=3, pct_done=10),
        tk(1, "done", due="2026-10-12", step=1),
        tk(2, due="2026-10-15", step=2, pct_done=50, start="2026-10-10"),
        tk(9),  # no step: after the numbered ones
    ]
    rows = progress(proj(), ts, TODAY).tasks
    assert [r.n for r in rows] == [1, 2, 3, 4]
    assert [r.title for r in rows] == ["task 1", "task 2", "task 3", "task 9"]
    assert [r.gap_days for r in rows] == [None, -2, 13, None]  # done: no gap
    assert [r.pct for r in rows] == [100, 50, 10, 0]
    assert rows[1].overdue and not rows[2].overdue
    assert task_pct(tk(1, "done")) == 100 and task_pct(tk(1, pct_done=250)) == 100


def test_steps_without_numbers_follow_the_milestone_order():
    p = proj(
        milestones=[
            P.Milestone(id="m1", title="Late", due="2026-11-30"),
            P.Milestone(id="m2", title="Early", due="2026-10-20"),
        ]
    )
    a, b = tk(1, milestone_id="m1"), tk(2, milestone_id="m2")
    assert [x.title for x in ordered_tasks(p, [a, b])] == ["task 2", "task 1"]


def test_progress_line_thai_and_english():
    p = proj(created_at="2026-10-07T00:00:00+07:00", deadline="2026-10-22")  # 66% time, 5 days
    pr = progress(p, tasks_done(1, 5), TODAY)
    assert pr.state == "at_risk"
    assert progress_line(pr, "th") == "ความคืบหน้าจริง 20% · เวลาผ่านไป 67% · เหลือ 5 วัน · 🔴 เสี่ยงไม่ทัน"
    assert (
        progress_line(pr, "en")
        == "Real progress 20% · 67% of the time used · 5 days left · 🔴 at risk"
    )
    behind = progress(
        proj(created_at="2026-09-01T00:00:00+07:00", deadline="2026-12-01"), tasks_done(1, 5), TODAY
    )
    assert progress_line(behind, "th").endswith("⚠️ ช้ากว่าแผน")
    assert "behind plan" in progress_line(behind, "en")


def test_progress_line_edge_cases():
    late = progress(proj(deadline="2026-10-15"), tasks_done(1, 5), TODAY)
    assert "เลยกำหนด 2 วัน" in progress_line(late, "th") and "2 days overdue" in progress_line(
        late, "en"
    )
    one = progress(proj(deadline="2026-10-18"), tasks_done(4, 5), TODAY)
    assert "เหลือ 1 วัน" in progress_line(one, "th") and "1 day left" in progress_line(one, "en")
    today = progress(proj(deadline="2026-10-17"), tasks_done(4, 5), TODAY)
    assert "ครบกำหนดวันนี้" in progress_line(today, "th")
    nodl = progress_line(progress(proj(deadline=""), [], TODAY), "en")
    assert nodl == "nothing to measure yet · ➖ no deadline"
    done = progress_line(progress(proj(), tasks_done(5, 5), TODAY), "th")
    assert (
        done.startswith("ความคืบหน้าจริง 100%") and done.endswith("🏁 เสร็จแล้ว") and "เหลือ" not in done
    )


def test_stats_carry_progress_and_the_summary_shows_the_line_and_steps():
    p = proj()
    ts = [
        tk(1, "done", due="2026-10-12", step=1, pct_done=100),
        tk(2, due="2026-10-20", step=2, pct_done=40),
    ]
    st = P.project_stats(p, ts, TODAY)
    assert st["progress"]["done_pct"] == 50 and st["progress"]["state"] == "on_track"
    assert "tasks" not in st["progress"]  # the per-task rows stay out of cards
    md = P.render_summary(p, ts, "en", TODAY)
    assert "Real progress 50%" in md
    assert "1. task 1 (100%, ✅)" in md and "2. task 2 (40%, 3 d left)" in md
    assert md.index("1. task 1") < md.index("2. task 2")
    th = P.render_summary(p, ts, "th", TODAY)
    assert "ความคืบหน้าจริง 50%" in th and "2. task 2 (40%, เหลือ 3 วัน)" in th
    assert "Steps" not in P.render_summary(p, [tk(1)], "en", TODAY)  # unnumbered tasks: no list


def test_task_card_text_mentions_the_step():
    task = tk(1, step=3, due="2026-10-20")
    assert "ขั้นที่ 3" in card_text(task, "th", channel="general")
    assert "Step 3" in card_text(task, "en", channel="general")
    assert "ขั้นที่" not in card_text(tk(1), "th", channel="general")


def test_task_json_has_the_plan_fields():
    j = P.task_json(tk(1, step=2, pct_done=30, start="2026-10-01", depends_on=["a"]))
    assert (j["step"], j["pct_done"], j["start"], j["depends_on"]) == (2, 30, "2026-10-01", ["a"])
    assert P.task_json(tk(1, "done"))["pct_done"] == 100


def test_insights_list_progress_by_project_worst_first():
    ok, bad = proj(id="p1", name="Fine"), proj(id="p2", name="Late", deadline="2026-10-10")
    out = P.insights([ok, bad], [], [], False, TODAY)
    rows = out["progress"]["by_project"]
    assert [r["name"] for r in rows] == ["Late", "Fine"] and rows[0]["state"] == "at_risk"
    assert rows[0]["days_left"] == -7
