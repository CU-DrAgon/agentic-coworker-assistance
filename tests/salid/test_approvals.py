# ruff: noqa: F811, F401
"""Approval requests: approver rules, storage (SQLite and Firestore), the confirm gate, the DM with
persistent buttons, the outcome DMs, receipt links, the leave read API, report and morning brief.
Offline: fake Discord objects."""

import asyncio
from datetime import UTC, date, datetime, timedelta

import pytest
from conftest import GUILD, mk
from test_briefings import set_features
from test_coordinator import (
    BOSS,
    INTERN,
    OWNER_ID,
    PEER,
    STAFF,
    Click,
    at,
    docs,
    world,
)
from test_modes import bot  # noqa: F401  (fixture)

from salid import reports as rep
from salid.approvals import (
    Approval,
    ApprovalRepo,
    fmt_amount,
    pick_approver,
)
from salid.approvals_bot import ApprovalButton
from salid.i18n import t


# ------------------------------------------------------------------ approver rules
def test_pick_approver_nearest_higher_then_executive():
    ranks = {1: 10, 2: 5, 3: 3, 4: 3, 5: 1}
    assert pick_approver(5, ranks, set()) == 3  # nearest above (lowest id on a tie)
    assert pick_approver(5, ranks, {4}) == 4  # an executive wins the tie
    assert pick_approver(3, ranks, set()) == 2
    assert pick_approver(2, ranks, set()) == 1
    assert pick_approver(1, ranks, {1, 2}) == 2  # top of the tree: another executive
    assert pick_approver(1, {1: 10}, {1}) == 0  # nobody else
    assert pick_approver(9, ranks, set()) == 5  # unranked requester: the lowest ranked member


def test_fmt_amount():
    assert fmt_amount(350, "THB") == "350 THB" and fmt_amount(12000.5, "THB") == "12,000.50 THB"
    assert fmt_amount(None) == ""


# ------------------------------------------------------------------ repo
def test_repo_decide_once_and_filters(docs):
    now = [datetime(2026, 10, 14, 5, tzinfo=UTC)]
    repo = ApprovalRepo(docs, now=lambda: now[0])
    a = repo.create(Approval(guild_id=GUILD, requester_id=STAFF, approver_id=BOSS, title="taxi"))
    b = repo.create(Approval(guild_id=GUILD + 1, requester_id=STAFF, approver_id=BOSS, title="x"))
    assert repo.get(a.id).status == "pending" and repo.get("nope") is None
    assert [x.id for x in repo.list(GUILD, approver_id=BOSS)] == [a.id]  # per guild
    assert repo.list(GUILD, requester_id=PEER) == []
    done = repo.decide(a.id, "approved", " ok ")
    assert done.status == "approved" and done.decision_note == "ok" and done.decided_at
    assert repo.decide(a.id, "declined") is None and repo.decide(a.id, "cancelled") is None
    assert repo.decide(b.id, "bogus") is None
    assert repo.attach_receipt(a.id, 7).receipt_id == 7


def test_stale_and_counts(docs):
    now = [datetime(2026, 10, 14, 5, tzinfo=UTC)]
    repo = ApprovalRepo(docs, now=lambda: now[0])
    old = repo.create(Approval(guild_id=GUILD, requester_id=STAFF, approver_id=BOSS, title="old"))
    now[0] += timedelta(hours=30)
    repo.create(Approval(guild_id=GUILD, requester_id=PEER, approver_id=BOSS, title="new"))
    ok = repo.create(Approval(guild_id=GUILD, requester_id=PEER, approver_id=BOSS, title="ok"))
    repo.decide(ok.id, "approved")
    assert [a.id for a in repo.stale_pending(GUILD, BOSS)] == [old.id]
    assert repo.stale_pending(GUILD, STAFF) == []
    zone = at(0).tzinfo
    c = repo.counts(GUILD, now[0].astimezone(zone).date(), zone)
    assert c == {"pending": 2, "stale": 1, "approved": 1, "declined": 0}
    assert repo.counts(GUILD, date(2026, 1, 1), zone)["approved"] == 0


def test_approved_leave_read_api(docs):
    repo = ApprovalRepo(docs)

    def leave(uid, s, e, status="approved", kind="leave"):
        a = repo.create(
            Approval(guild_id=GUILD, requester_id=uid, approver_id=BOSS, title="l", kind=kind,
                     start_date=s, end_date=e)
        )  # fmt: skip
        if status != "pending":
            repo.decide(a.id, status)

    leave(STAFF, "2026-10-15", "2026-10-16")
    leave(STAFF, "2026-10-30", "")  # one day, no end date
    leave(STAFF, "2026-10-20", "2026-10-21", status="declined")
    leave(STAFF, "2026-10-22", "2026-10-22", status="pending")
    leave(PEER, "2026-10-15", "2026-10-15")
    leave(STAFF, "2026-10-18", "2026-10-18", kind="other")
    d = date.fromisoformat
    assert repo.approved_leave(GUILD, STAFF, "2026-10-01", "2026-10-31") == [
        (d("2026-10-15"), d("2026-10-16")),
        (d("2026-10-30"), d("2026-10-30")),
    ]
    assert repo.approved_leave(GUILD, STAFF, d("2026-10-16"), d("2026-10-16")) == [
        (d("2026-10-15"), d("2026-10-16"))
    ]  # overlap counts
    assert repo.approved_leave(GUILD, STAFF, "2026-10-17", "2026-10-29") == []
    assert repo.approved_leave(GUILD + 1, STAFF, "2026-10-01", "2026-10-31") == []


# ------------------------------------------------------------------ the agent tools
def tools_of(bot, world, who, is_exec=False):
    ts = bot._approval_tools(world.g, who, is_exec, "th")
    return {f.__name__: f for f in ts}


async def call(fn, *a, **kw):
    return await asyncio.to_thread(lambda: fn(*a, **kw))


EXPENSE = dict(kind="expense", title="ค่าแท็กซี่", reason="ไปหาลูกค้า", amount=350, currency="THB")


async def send_request(bot, world, who=None, **kw):
    """Propose, then confirm in a 'later message' (a new tool set = a new request id)."""
    who = who or world.staff
    args = {**EXPENSE, **kw}
    first = await call(tools_of(bot, world, who)["request_approval"], **args)
    assert first["status"] == "needs_confirmation"
    return first, await call(tools_of(bot, world, who)["request_approval"], **args)


async def test_it_asks_for_confirmation_first_and_dms_nobody_until_confirmed(bot, world):
    tools = tools_of(bot, world, world.staff)
    first = await call(tools["request_approval"], **EXPENSE)
    again = await call(tools["request_approval"], **EXPENSE)  # same message: still not confirmed
    assert first["status"] == again["status"] == "needs_confirmation"
    p = first["proposal"]
    assert (
        p["approver"] == "Boss" and p["amount"] == "350 THB" and "ยืนยันไหมคะ" in first["instruction"]
    )
    assert world.boss.dms == [] and bot.approvals.list(GUILD) == []
    sent = await call(tools_of(bot, world, world.staff)["request_approval"], **EXPENSE)
    assert sent["status"] == "sent" and sent["request"]["status"] == "pending"
    assert len(bot.approvals.list(GUILD)) == 1


async def test_changed_arguments_need_a_new_confirmation(bot, world):
    await send_request(bot, world)
    other = await call(
        tools_of(bot, world, world.peer)["request_approval"], **{**EXPENSE, "amount": 400}
    )
    assert other["status"] == "needs_confirmation"


async def test_the_approver_gets_a_dm_with_persistent_buttons(bot, world):
    _, sent = await send_request(bot, world)
    ((text, kw),) = world.boss.dms
    assert "Somchai" in text and "ค่าแท็กซี่" in text and "350 THB" in text and "ไปหาลูกค้า" in text
    ids = [b.custom_id for b in kw["view"].children]
    assert [i.split(":")[2] for i in ids] == ["ok", "no"] and kw["view"].timeout is None
    rx = ApprovalButton.__discord_ui_compiled_template__
    assert all(rx.fullmatch(i) for i in ids) and sent["request"]["id"] == ids[0].rsplit(":", 1)[1]
    assert world.staff.dms == []  # the requester gets the outcome later


async def test_default_approver_follows_the_hierarchy_and_never_the_requester(bot, world):
    assert (await send_request(bot, world, world.intern))[1]["request"]["approver"] == "Somchai"
    assert (await send_request(bot, world, world.boss, title="b"))[1]["request"][
        "approver"
    ] == "Admin"
    own = await call(
        tools_of(bot, world, world.staff)["request_approval"], **EXPENSE, approver="Somchai"
    )
    assert "own request" in own["error"]
    named = await call(
        tools_of(bot, world, world.staff)["request_approval"], **EXPENSE, approver="Peer"
    )
    assert named["proposal"]["approver"] == "Peer"
    unknown = await call(
        tools_of(bot, world, world.staff)["request_approval"], **EXPENSE, approver="Nobody"
    )
    assert unknown["status"] == "needs_info"


async def test_input_validation(bot, world):
    t_ = tools_of(bot, world, world.staff)["request_approval"]
    assert (await call(t_, kind="meal", title="x", reason="y"))["error"]
    assert (await call(t_, kind="expense", title="x", reason="y"))[
        "status"
    ] == "needs_info"  # amount
    assert (await call(t_, kind="expense", title="x", reason="", amount=5))[
        "status"
    ] == "needs_info"
    assert (await call(t_, kind="expense", title="x", reason="y", amount=-5))["error"]
    assert (await call(t_, kind="leave", title="ลา", reason="ป่วย"))["status"] == "needs_info"
    assert (await call(t_, kind="leave", title="ลา", reason="p", start_date="2026-01-01"))["error"]
    assert bot.approvals.list(GUILD) == []


async def test_a_closed_dm_cancels_the_request(bot, world):
    async def closed(text=None, **kw):
        raise discord_forbidden()

    world.boss.send = closed
    _, res = await send_request(bot, world)
    assert res["error"] == "could_not_dm_approver"
    assert [a.status for a in bot.approvals.list(GUILD)] == ["cancelled"]


def discord_forbidden():
    from types import SimpleNamespace

    import discord

    return discord.Forbidden(SimpleNamespace(status=403, reason="x"), "closed")


async def test_duplicate_pending_request_is_not_sent_twice(bot, world):
    await send_request(bot, world)
    again = await call(tools_of(bot, world, world.staff)["request_approval"], **EXPENSE)
    assert again["status"] == "already_pending" and len(world.boss.dms) == 1


# ------------------------------------------------------------------ the approver's decision
async def pending(bot, world, **kw):
    _, sent = await send_request(bot, world, **kw)
    return sent["request"]["id"]


async def test_approve_updates_edits_the_dm_and_tells_the_requester(bot, world):
    rid = await pending(bot, world)
    stranger = Click(world.peer)
    await bot.handle_approval(stranger, "ok", rid)
    assert stranger.sent == [(t("appr_not_yours", "th"), True)]
    assert bot.approvals.get(rid).status == "pending" and world.staff.dms == []
    click = Click(world.boss)
    await bot.handle_approval(click, "ok", rid)
    assert "✅" in click.edited[0]["content"] and click.edited[0]["view"] is None
    assert bot.approvals.get(rid).status == "approved"
    ((text, _),) = world.staff.dms
    assert "ค่าแท็กซี่" in text and "Boss" in text and "อนุมัติ" in text
    again = Click(world.boss)
    await bot.handle_approval(again, "no", rid)  # decided once
    assert again.sent == [(t("appr_handled", "th"), True)] and len(world.staff.dms) == 1
    assert bot.approvals.get(rid).status == "approved"


async def test_decline_carries_the_reason_to_the_requester(bot, world):
    rid = await pending(bot, world)
    click = Click(world.boss)
    await bot.handle_approval(click, "no", rid, "งบหมดแล้ว")
    assert "🚫" in click.edited[0]["content"]
    a = bot.approvals.get(rid)
    assert a.status == "declined" and a.decision_note == "งบหมดแล้ว"
    ((text, _),) = world.staff.dms
    assert "ไม่ได้รับอนุมัติ" in text and "งบหมดแล้ว" in text


async def test_decline_button_opens_the_reason_modal_and_ok_decides_directly(bot, world):
    from types import SimpleNamespace

    from salid.approvals_bot import DeclineModal

    rid = await pending(bot, world)
    shown = []

    async def send_modal(m):
        shown.append(m)

    inter = SimpleNamespace(client=bot, response=SimpleNamespace(send_modal=send_modal))
    await ApprovalButton("no", rid).callback(inter)
    assert isinstance(shown[0], DeclineModal) and shown[0].approval_id == rid
    shown[0].note._value = "ไม่จำเป็น"
    click = Click(world.boss)
    click.client = bot
    await shown[0].on_submit(click)
    assert bot.approvals.get(rid).decision_note == "ไม่จำเป็น"


async def test_cannot_decide_your_own_request_even_if_named_approver_matches(bot, world):
    rid = await pending(bot, world)
    a = bot.approvals.get(rid)
    a.approver_id = a.requester_id  # a corrupted record must still not let you self-approve
    bot.approvals._save(a)
    click = Click(world.staff)
    await bot.handle_approval(click, "ok", rid)
    assert bot.approvals.get(rid).status == "pending" and click.sent[0][1] is True


# ------------------------------------------------------------------ list, cancel, receipts
async def test_list_mine_and_to_me_and_cancel(bot, world):
    rid = await pending(bot, world)
    mine = await call(tools_of(bot, world, world.staff)["list_approvals"])
    assert [r["id"] for r in mine["approvals"]] == [rid] and mine["approvals"][0][
        "approver"
    ] == "Boss"
    to_me = await call(
        tools_of(bot, world, world.boss)["list_approvals"], status="pending", scope="to_me"
    )
    assert [r["id"] for r in to_me["approvals"]] == [rid]
    assert (await call(tools_of(bot, world, world.peer)["list_approvals"], scope="to_me"))[
        "count"
    ] == 0
    assert (await call(tools_of(bot, world, world.peer)["list_approvals"]))["count"] == 0
    assert (await call(tools_of(bot, world, world.staff)["list_approvals"], scope="x"))["error"]
    bad = await call(tools_of(bot, world, world.peer)["cancel_approval"], rid)
    assert "no such request" in bad["error"]
    ok = await call(tools_of(bot, world, world.staff)["cancel_approval"], rid)
    assert (
        ok["cancelled"]["status"] == "cancelled" and len(world.boss.dms) == 2
    )  # told the approver
    assert (
        "already cancelled"
        in (await call(tools_of(bot, world, world.staff)["cancel_approval"], rid))["error"]
    )
    click = Click(world.boss)
    await bot.handle_approval(click, "ok", rid)
    assert click.sent == [(t("appr_handled", "th"), True)]


def receipt(bot, n, uid, total, cur="THB"):
    r = mk(bot.store, n, uid, "x", "Grab", "2026-10-13", total)
    return r if cur == "THB" else bot.store.update_receipt(r.id, {"currency": cur})


async def test_receipt_link_keeps_ledger_totals(bot, world):
    r_own = receipt(bot, 1, STAFF, 350.0)
    r_other = receipt(bot, 2, PEER, 90.0)
    diff = receipt(bot, 3, STAFF, 500.0)
    before = bot.store.summarize(GUILD, "2026-10-01", "2026-10-31", "category")
    rid = await pending(bot, world)
    a_id = bot.approvals.decide(rid, "approved").id
    att = tools_of(bot, world, world.staff)["attach_receipt_to_approval"]
    ok = await call(att, a_id, r_own.id)
    assert (
        ok["linked"]["receipt_id"] == r_own.id
        and ok["ledger_totals"] == "unchanged"
        and "warning" not in ok
    )
    assert bot.approvals.get(a_id).receipt_id == r_own.id
    assert "own receipts" in (await call(att, a_id, r_other.id))["error"]
    assert "no receipt" in (await call(att, a_id, 999))["error"]
    assert "differs" in (await call(att, a_id, diff.id))["warning"]
    assert (
        await call(tools_of(bot, world, world.peer)["attach_receipt_to_approval"], a_id, r_own.id)
    )["error"]
    after = bot.store.summarize(GUILD, "2026-10-01", "2026-10-31", "category")
    assert before == after  # linking writes nothing to the ledger


async def test_receipt_can_be_attached_when_requesting_and_must_be_the_askers(bot, world):
    r = receipt(bot, 1, STAFF, 350.0)
    _, sent = await send_request(bot, world, attach_receipt_id=r.id)
    assert sent["request"]["receipt_id"] == r.id and "#" + str(r.id) in world.boss.dms[0][0]
    other = receipt(bot, 2, PEER, 10.0)
    err = await call(
        tools_of(bot, world, world.staff)["request_approval"], **EXPENSE, attach_receipt_id=other.id
    )
    assert "own receipts" in err["error"]
    leave = await call(
        tools_of(bot, world, world.staff)["request_approval"],
        kind="leave",
        title="ลา",
        reason="ป่วย",
        start_date="2026-10-15",
        attach_receipt_id=r.id,
    )
    assert "only an expense" in leave["error"]


async def test_leave_request_feeds_approved_leave(bot, world):
    _, sent = await send_request(
        bot,
        world,
        kind="leave",
        title="ลาพักร้อน",
        reason="ธุระ",
        amount=None,
        currency=None,
        start_date="2026-10-15",
        end_date="2026-10-16",
    )
    rid = sent["request"]["id"]
    assert "2026-10-15" in world.boss.dms[0][0]
    assert bot.approved_leave(GUILD, STAFF, "2026-10-14", "2026-10-31") == []  # not approved yet
    await bot.handle_approval(Click(world.boss), "ok", rid)
    assert bot.approved_leave(GUILD, STAFF, date(2026, 10, 14), date(2026, 10, 31)) == [
        (date(2026, 10, 15), date(2026, 10, 16))
    ]


async def test_feature_flag_hides_the_tools(bot, world):
    assert {"request_approval", "list_approvals", "cancel_approval"} <= set(
        tools_of(bot, world, world.staff)
    )
    set_features(bot, approvals=False)
    assert bot._approval_tools(world.g, world.staff, False, "th") == []


# ------------------------------------------------------------------ report and morning brief
async def test_evening_report_shows_pending_and_approved_counts(bot, world):
    await pending(bot, world)
    b = await pending(bot, world, who=world.peer)
    bot.approvals.decide(b, "approved")
    day = bot.s.now().date()
    counts = bot._approval_counts(GUILD, day)
    assert counts["pending"] == 1 and counts["approved"] == 1
    inp = rep.CompanyInput(day=str(day), ledger={"today_by_currency": {}, "today_count": 0,
                           "month_by_currency": {}, "review": 0, "duplicates": 0},
                           channels=[], people=[], unassigned_open=[], approvals=counts)  # fmt: skip
    plain = rep.render_plain(inp, "en")
    assert t("rep_approvals", "en", pending=1, stale=0, approved=1) in plain
    assert "APPROVAL REQUESTS" in rep.compact_text(inp)
    set_features(bot, approvals=False)
    assert bot._approval_counts(GUILD, day) == {}


async def test_morning_brief_reminds_the_approver_of_requests_pending_over_a_day(bot, world):
    rid = await pending(bot, world)
    a = bot.approvals.get(rid)
    a.created_at = (bot.s.now() - timedelta(days=2)).astimezone(UTC).isoformat(timespec="seconds")
    bot.approvals._save(a)
    fresh = await pending(bot, world, who=world.peer, title="ใหม่")  # too new to nag about
    assert fresh
    world.boss.dms.clear()
    assert await bot._briefings_due(at(8, 31, day=bot.s.now().day)) >= 1
    text = world.boss.dms[0][0]
    assert "ค่าแท็กซี่" in text and "Somchai" in text and "ใหม่" not in text
    rows = bot._approvals_waiting(GUILD, BOSS, bot.s.now())
    assert [r["id"] for r in rows] == [rid] and rows[0]["days"] >= 2
    assert bot._approvals_waiting(GUILD, STAFF, bot.s.now()) == []
