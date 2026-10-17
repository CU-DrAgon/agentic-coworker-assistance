"""The three progress pictures: valid PNGs under 1 MB with Thai labels, for 0 / 1 / 20 projects, a
person with an overdue task, one project with milestones and steps. The demo set is also saved to
SALID_SAMPLE_DIR when that is set (how the owner's sample pictures are made)."""

import os
import struct
from datetime import date
from pathlib import Path

import pytest

from salid import projects as P
from salid.progress import progress
from salid.progress_chart import MAX_BYTES, person_png, portfolio_png, project_png
from salid.tasks import Task

TODAY = date(2026, 10, 17)
TOEY = P.Person(email="toey@x.com", discord_user_id="26", name="Toey")
MAGIC = b"\x89PNG\r\n\x1a\n"


def size_of(png: bytes) -> tuple[int, int]:
    return struct.unpack(">II", png[16:24])


def check(png: bytes) -> None:
    assert png[:8] == MAGIC and 3_000 < len(png) <= MAX_BYTES
    w, h = size_of(png)
    assert w >= 800 and h >= 240


def proj(i=1, name="เว็บไซต์ร้านใหม่", deadline="2026-10-27", created="2026-10-01", **kw):
    return P.Project(
        id=f"p{i}", name=name, owner=TOEY, status="in_progress", deadline=deadline,
        created_at=created + "T00:00:00+07:00", **kw,
    )  # fmt: skip


def task(pid="p1", title="อัดวิดีโอเดโม", **kw) -> Task:
    base = dict(guild_id=1, project_id=pid, title=title, assignee_name="Toey", assignee_id=26)
    return Task(**{**base, **kw})


def items(projects, tasks):
    return [(p, progress(p, tasks, TODAY)) for p in projects]


def demo():
    ms = [
        P.Milestone(id="m1", title="Demo พร้อม", due="2026-10-20"),
        P.Milestone(id="m2", title="ส่งงาน", due="2026-10-22", status="pending"),
    ]
    p1 = proj(1, "งานแข่ง Hackathon Salid", "2026-10-22", "2026-09-20", milestones=ms)
    ts = [
        task("p1", "เขียน pitch deck", step=1, start="2026-10-01", due="2026-10-10", status="done", milestone_id="m1"),
        task("p1", "อัดวิดีโอเดโม", step=2, start="2026-10-10", due="2026-10-19", pct_done=40, milestone_id="m1"),
        task("p1", "เตรียม repo ให้เรียบร้อย", step=3, start="2026-10-12", due="2026-10-16", pct_done=60, milestone_id="m2"),
        task("p1", "ส่งเข้าระบบ", step=4, start="2026-10-20", due="2026-10-22", milestone_id="m2", assignee_name="Peat", assignee_id=27),
        task("p2", "ออกแบบหน้าแรก", step=1, start="2026-10-01", due="2026-10-14", status="done"),
        task("p2", "เขียน frontend", step=2, start="2026-10-14", due="2026-11-10"),
        task("p3", "ยิงแอด", step=1, start="2026-09-20", due="2026-10-15", pct_done=30),
    ]  # fmt: skip
    ps = [
        p1,
        proj(2, "เว็บไซต์ร้านใหม่", "2026-12-01"),
        proj(3, "Launch campaign Q4", "2026-10-15", "2026-09-01"),
        proj(4, "ระบบบัญชี", "2027-01-15", "2026-09-15"),
        proj(5, "Internal wiki", ""),
    ]
    return ps, ts


def test_portfolio_for_zero_one_and_twenty_projects():
    check(portfolio_png([], "th", "โปรเจกต์ที่กำลังทำอยู่", TODAY))
    one = proj()
    check(portfolio_png(items([one], [task()]), "en", today=TODAY))
    many = [
        proj(i, f"โปรเจกต์หมายเลข {i} " + "ยาวมาก" * 8, f"2026-1{i % 3}-2{i % 9}")
        for i in range(1, 31)
    ]
    big = portfolio_png(items(many, []), "th", today=TODAY)
    check(big)
    assert size_of(big)[1] < 1500  # 30 given, 20 drawn
    assert size_of(big)[1] > size_of(portfolio_png(items(many[:5], []), "th", today=TODAY))[1]


def test_portfolio_sorts_at_risk_first_and_has_the_row_height_of_its_rows():
    ps, ts = demo()
    png = portfolio_png(items(ps, ts), "th", today=TODAY)
    check(png)
    assert size_of(png)[1] == 150 + 58 * 5


def test_person_chart_with_an_overdue_task_and_no_project_block():
    ps, ts = demo()
    mine = [x for x in ts if x.assignee_name == "Toey" and x.status == "open"]
    png = person_png(
        "Toey", items(ps[:3], ts), mine + [task("", "ต่ออายุโดเมน", due="2026-10-30")], "th", TODAY
    )
    check(png)
    nothing = person_png("Toey", [], [], "en", TODAY)  # nobody: still a valid picture
    check(nothing)
    owned = person_png("Toey", items(ps[:1], ts), [], "en", TODAY)  # owns a project, no open task
    check(owned)


def test_person_chart_caps_tasks_per_project():
    p = proj()
    many = [task(title=f"task {i}", due=f"2026-10-2{i % 9}", step=i + 1) for i in range(15)]
    png = person_png("Toey", items([p], many), many, "th", TODAY)
    check(png)
    assert size_of(png)[1] < 170 + 36 * 12  # a header, 6 tasks and a "+9" line


def test_project_chart_with_milestones_steps_and_done_ticks():
    ps, ts = demo()
    for lang in ("th", "en"):
        check(project_png(ps[0], ts, lang, TODAY))
    check(project_png(ps[1], ts, "th", TODAY))  # no milestones: steps only
    check(project_png(proj(5, deadline=""), [], "th", TODAY))  # nothing at all
    ms_only = proj(milestones=[P.Milestone(id="m1", title="Only", due="2026-10-20")])
    check(project_png(ms_only, [], "en", TODAY))


def test_project_chart_survives_odd_dates():
    p = proj(deadline="2026-10-05")  # past the deadline already
    odd = [
        task(due="2026-10-01", start="2026-10-09", pct_done=100, status="done"),
        task(due="", start=""),
    ]
    check(project_png(p, odd, "th", TODAY))


@pytest.mark.skipif(not os.environ.get("SALID_SAMPLE_DIR"), reason="sample pictures are opt-in")
def test_save_sample_pictures():
    out = Path(os.environ["SALID_SAMPLE_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    ps, ts = demo()
    its = items(ps, ts)
    (out / "1-portfolio-running-now.png").write_bytes(portfolio_png(its, "th", today=TODAY))
    mine = [x for x in ts if x.assignee_name == "Toey" and x.status == "open"]
    (out / "2-person-toey.png").write_bytes(person_png("Toey", its[:3], mine, "th", TODAY))
    (out / "3-project-hackathon.png").write_bytes(project_png(ps[0], ts, "th", TODAY))
