# ruff: noqa: F811, F401
"""Task handshake: assignee cards, answers, the giver's line, groups, approve mode, personal
settings. Offline: fake Discord objects, SQLite doc store, no model."""

import asyncio
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import GUILD, FixedClockSettings
from test_coordinator import (  # noqa: F401  (fixtures)
    BOSS,
    INTERN,
    OWNER_ID,
    PEER,
    PEOPLE,
    PUBLIC,
    STAFF,
    at,
    bot,
    person,
    world,
)

from salid import task_handshake as H
from salid.coo_store import STATE, SQLiteDocStore
from salid.coo_tools import CooTools, build_coo_tools
from salid.guild_settings import FEATURES, Features, GuildSettingsUpdate
from salid.i18n import t
from salid.memory_summary import ActionItem
from salid.profiles import Profile, ProfileRepo, ProfileTools, apply_change
from salid.task_handshake_bot import TaskButton, TaskEditModal
from salid.tasks import Task, TaskRepo, TaskTracker, guess_assigner, split_who

BKK_NOW = at(10)


# ------------------------------------------------------------------ fakes
class Click:
    """A component / modal interaction by `user` on a card message."""

    def __init__(self, user, message=True):
        self.user = user
        self.message = SimpleNamespace(content="CARD") if message else None
        self.edited, self.sent, self.modals, self.followups = [], [], [], []

        async def edit_message(**kw):
            self.edited.append(kw)

        async def send_message(text, ephemeral=False):
            self.sent.append((text, ephemeral))

        async def send_modal(modal):
            self.modals.append(modal)

        async def followup_send(text, ephemeral=False):
            self.followups.append((text, ephemeral))

        self.response = SimpleNamespace(
            edit_message=edit_message, send_message=send_message, send_modal=send_modal
        )
        self.followup = SimpleNamespace(send=followup_send)


@pytest.fixture
def hs(bot, world):
    """Every person's send() returns a message with a channel and an id, like Discord."""
    counter = iter(range(5000, 9000))
    for m in world.g.members.values():

        def make(m):
            async def send(text=None, **kw):
                m.dms.append((text, kw))
                return SimpleNamespace(id=next(counter), channel=SimpleNamespace(id=700 + m.id))

            return send

        m.send = make(m)
    return bot


def new_task(bot, title="ส่งใบเสนอราคา", assignee=STAFF, assigner=BOSS, **kw):
    task = Task(
        guild_id=GUILD,
        assignee_id=assignee,
        assignee_name=PEOPLE.get(assignee, "x"),
        assigner_id=assigner,
        assigner_name=PEOPLE.get(assigner, "") if assigner else "",
        title=title,
        due="2026-10-16",
        source_channel_id=PUBLIC,
        created_from="summary",
        **kw,
    )
    return bot.tasks.add(task)[0]


def set_settings(bot, **kw):
    features = kw.pop("features", {})
    bot.guilds_store.save_settings(
        GUILD,
        GuildSettingsUpdate(features=Features(**features), **{"default_language": "th", **kw}),
        "x",
    )
    bot.team._cache.clear()


def last_view(person_):
    return person_.dms[-1][1]["view"]


def button(view, act):
    return next(b for b in view.children if b.act == act)


# ------------------------------------------------------------------ tasks.py: assigner + groups
def tracker(docs, **kw):
    return TaskTracker(
        TaskRepo(docs), FixedClockSettings(), None, people_of=lambda gid: PEOPLE, **kw
    )


def msg(uid, name, text, minute=0):
    return SimpleNamespace(
        author_id=uid,
        author_name=name,
        text=text,
        kind="chat",
        jump_url="",
        created_at=at(9, minute),
    )


def test_old_task_documents_load_with_defaults():
    old = {"id": "abcd1234", "guild_id": GUILD, "title": "x", "assignee_id": STAFF}
    task = Task(**old)
    assert (task.assigner_id, task.confirm, task.group_id, task.approval_rounds) == (0, "", "", 0)
    assert task.handshake_message == "" and task.confirm_at == ""


def test_action_item_has_by():
    assert ActionItem(what="x").by == "" and ActionItem(what="x", by="Boss").by == "Boss"


def test_ingest_fills_the_assigner_from_by_from_the_heuristic_or_leaves_zero():
    docs = SQLiteDocStore(":memory:")
    tr = tracker(docs)
    items = [
        {"who": "Somchai", "what": "ส่งใบเสนอราคาลูกค้า X", "by": "Boss"},
        {"who": "Ploy", "what": "ทำโพสต์โปรโมชัน", "by": ""},
        {"who": "Somchai", "what": "จองห้องประชุมใหญ่", "by": ""},
        {"who": "Somchai", "what": "สรุปยอดขายรายเดือน", "by": "คนที่ไม่รู้จัก"},
    ]
    msgs = [
        msg(BOSS, "Boss", "Ploy ช่วยทำโพสต์โปรโมชัน ให้หน่อย", 1),
        msg(INTERN, "Ploy", "รับทราบ", 2),
    ]
    new = tr.ingest(GUILD, PUBLIC, "2026-10-14", {"action_items": items}, msgs)
    by_title = {t_.title: t_ for t_ in new}
    assert by_title["ส่งใบเสนอราคาลูกค้า X"].assigner_id == BOSS  # from `by`
    assert by_title["ส่งใบเสนอราคาลูกค้า X"].assigner_name == "Boss"
    assert (
        by_title["ทำโพสต์โปรโมชัน"].assigner_id == BOSS
    )  # newest message naming Ploy + a title word
    assert by_title["จองห้องประชุมใหญ่"].assigner_id == 0
    unknown = by_title["สรุปยอดขายรายเดือน"]
    assert unknown.assigner_id == 0 and unknown.assigner_name == "คนที่ไม่รู้จัก"


def test_guess_assigner_needs_the_name_and_a_title_word():
    people = PEOPLE
    ms = [msg(BOSS, "Boss", "Somchai ส่งรายงานยอดขาย"), msg(INTERN, "Ploy", "อะไรก็ได้")]
    assert guess_assigner(ms, people, STAFF, "Somchai", "ส่งรายงานยอดขาย") == BOSS
    assert guess_assigner(ms, people, STAFF, "Somchai", "คนละเรื่องเลย") == 0
    assert (
        guess_assigner(
            [msg(BOSS, "Boss", "ส่งรายงานยอดขาย")], people, STAFF, "Somchai", "ส่งรายงานยอดขาย"
        )
        == 0
    )
    assert guess_assigner(ms, people, 0, "", "ส่งรายงานยอดขาย") == 0


@pytest.mark.parametrize(
    "who, parts",
    [
        ("เตยกับบี", ["เตย", "บี"]),
        ("A, B and C", ["A", "B", "C"]),
        ("@A @B", ["@A", "@B"]),
        ("<@1><@2>", ["<@1>", "<@2>"]),
        ("A & B / C", ["A", "B", "C"]),
        ("Somchai", ["Somchai"]),
        ("", []),
    ],
)
def test_split_who(who, parts):
    assert split_who(who) == parts


def test_several_names_make_one_task_each_with_a_shared_group_and_both_survive_dedupe():
    docs = SQLiteDocStore(":memory:")
    tr = tracker(docs)
    s = {"action_items": [{"who": "Somchai and Ploy", "what": "เตรียมงานแสดงสินค้า", "by": "Boss"}]}
    new = tr.ingest(GUILD, PUBLIC, "2026-10-14", s, [])
    assert sorted(x.assignee_id for x in new) == [STAFF, INTERN]
    assert len({x.group_id for x in new}) == 1 and new[0].group_id
    assert all(x.assigner_id == BOSS and x.title == new[0].title for x in new)
    again = tr.ingest(GUILD, PUBLIC, "2026-10-14", s, [])  # a later summary lists it again
    assert again == [] and len(tr.repo.list(GUILD, status=None)) == 2


def test_one_name_is_unchanged_and_has_no_group():
    tr = tracker(SQLiteDocStore(":memory:"))
    (one,) = tr.ingest(
        GUILD, PUBLIC, "2026-10-14", {"action_items": [{"who": "Somchai", "what": "งานเดียว"}]}, []
    )
    assert one.assignee_id == STAFF and one.group_id == "" and not one._needs_owner


def test_a_small_role_makes_a_task_per_member_a_big_one_stays_unassigned():
    roles = {"tech": [STAFF, INTERN, BOSS], "all": list(range(100, 108))}
    tr = tracker(
        SQLiteDocStore(":memory:"),
        role_members_of=lambda gid, name: roles.get(name.split()[-1].lower()),
    )
    people = {**PEOPLE, **{i: f"p{i}" for i in range(100, 108)}}
    tr.people_of = lambda gid: people
    small = tr.ingest(
        GUILD, PUBLIC, "2026-10-14", {"action_items": [{"who": "ทีม tech", "what": "ทำเว็บ"}]}, []
    )
    assert sorted(x.assignee_id for x in small) == sorted([STAFF, INTERN, BOSS])
    assert len({x.group_id for x in small}) == 1
    (big,) = tr.ingest(
        GUILD,
        PUBLIC,
        "2026-10-14",
        {"action_items": [{"who": "ทีม all", "what": "ประชุมใหญ่", "by": "Boss"}]},
        [],
    )
    assert big.assignee_id == 0 and big.assignee_name == "ทีม all" and big._needs_owner
    assert big.assigner_id == BOSS and not H.offerable(big)


def test_a_declined_task_is_not_reassigned_by_a_later_summary():
    tr = tracker(SQLiteDocStore(":memory:"))
    s = {"action_items": [{"who": "Somchai", "what": "ส่งรายงานลูกค้า"}]}
    (task,) = tr.ingest(GUILD, PUBLIC, "2026-10-14", s, [])
    task.confirm, task.assignee_id = "declined", 0
    tr.repo.save(task)
    assert tr.ingest(GUILD, PUBLIC, "2026-10-14", s, []) == []
    assert tr.repo.get(GUILD, task.id).assignee_id == 0


# ------------------------------------------------------------------ the card
async def test_a_card_is_sent_once_per_new_task_with_persistent_buttons(hs, world):
    task = new_task(hs)
    assert await hs._handshake_offer(GUILD, [task]) == 1
    ((text, kw),) = world.staff.dms
    assert "ส่งใบเสนอราคา" in text and "Boss" in text and "2026-10-16" in text and "ทั่วไป" in text
    view = kw["view"]
    assert view.timeout is None and [b.act for b in view.children] == ["ok", "edit", "no"]
    rx = TaskButton.__discord_ui_compiled_template__
    for b in view.children:
        m = rx.fullmatch(b.item.custom_id)
        assert m and m["id"] == task.id and int(m["gid"]) == GUILD
        again = await TaskButton.from_custom_id(None, b.item, m)
        assert (again.act, again.guild_id, again.task_id) == (b.act, GUILD, task.id)
    saved = hs.tasks.get(GUILD, task.id)
    assert (
        saved.confirm == "pending" and saved.confirm_at and saved.handshake_message.count(":") == 2
    )


async def test_no_card_for_unresolved_assignees_dm_scope_or_when_the_flag_is_off(hs, world):
    nobody = new_task(hs, "ไม่มีเจ้าของ", assignee=0)
    dm = Task(
        guild_id=GUILD, assignee_id=STAFF, title="โน้ต", source_channel_id=-STAFF, created_from="dm"
    )
    dm = hs.tasks.add(dm)[0]
    assert await hs._handshake_offer(GUILD, [nobody, dm]) == 0 and world.staff.dms == []
    ok = new_task(hs, "มีเจ้าของ")
    set_settings(hs, features={"task_handshake": False})
    assert await hs._handshake_offer(GUILD, [ok]) == 0 and world.staff.dms == []
    assert hs.tasks.get(GUILD, ok.id).confirm == ""


async def test_up_to_five_cards_per_message_one_message_per_person(hs, world):
    tasks = [
        new_task(hs, f"รายการ{chr(65 + i) * 6} เรื่องพิเศษ{i * 7919}", assigner=0) for i in range(7)
    ]
    other = new_task(hs, "งานของพลอย", assignee=INTERN)
    assert await hs._handshake_offer(GUILD, [*tasks, other]) == 3
    assert [len(kw["view"].children) for _, kw in world.staff.dms] == [15, 6]
    assert len(world.intern.dms) == 1
    first = world.staff.dms[0][0]
    assert first.count("📋") == 5 and "1." in first and "5." in first
    labels = [b.item.label for b in world.staff.dms[0][1]["view"].children]
    assert labels[0].startswith("1.") and labels[-1].startswith("5.")


async def test_the_assignee_accepts_the_card_is_finished_and_the_giver_is_told(hs, world):
    task = new_task(hs)
    await hs._handshake_offer(GUILD, [task])
    click = Click(world.staff)
    await hs.handle_task_answer(click, "ok", GUILD, task.id)
    saved = hs.tasks.get(GUILD, task.id)
    assert saved.confirm == "accepted" and saved.assignee_id == STAFF
    (edit,) = click.edited
    assert edit["view"] is None and t("hs_status_ok", "th") in edit["content"]
    ((line, _),) = world.boss.dms
    assert line == t(
        "hs_giver_ok", "th", name="Somchai", title=task.title, due="2026-10-16", id=task.id
    )
    again = Click(world.staff)
    await hs.handle_task_answer(again, "ok", GUILD, task.id)
    assert again.sent == [(t("hs_handled", "th"), True)] and len(world.boss.dms) == 1


async def test_someone_else_gets_an_ephemeral_refusal_an_executive_may_answer(hs, world):
    task = new_task(hs)
    await hs._handshake_offer(GUILD, [task])
    stranger = Click(world.peer)
    await hs.handle_task_answer(stranger, "ok", GUILD, task.id)
    assert stranger.sent == [(t("hs_not_yours", "th"), True)] and not stranger.edited
    assert hs.tasks.get(GUILD, task.id).confirm == "pending" and world.boss.dms == []
    boss_click = Click(world.boss)  # an executive may answer for the assignee
    await hs.handle_task_answer(boss_click, "ok", GUILD, task.id)
    assert hs.tasks.get(GUILD, task.id).confirm == "accepted"


async def test_edit_modal_changes_title_due_and_note_and_the_giver_hears_the_edits(hs, world):
    task = new_task(hs)
    await hs._handshake_offer(GUILD, [task])
    opener = Click(world.staff)
    await hs.handle_task_modal_open(opener, "edit", GUILD, task.id)
    (modal,) = opener.modals
    assert isinstance(modal, TaskEditModal)
    assert modal.f_title.default == task.title and modal.f_due.default == "2026-10-16"
    click = Click(world.staff)
    await hs.handle_task_edit_submit(
        click, GUILD, task.id, "ส่งใบเสนอราคาแก้ไข", "2026-10-20", "รอข้อมูล"
    )
    saved = hs.tasks.get(GUILD, task.id)
    assert (saved.title, saved.due, saved.note, saved.confirm) == (
        "ส่งใบเสนอราคาแก้ไข", "2026-10-20", "รอข้อมูล", "accepted",
    )  # fmt: skip
    assert t("hs_status_edit", "th") in click.edited[0]["content"]
    ((line, _),) = world.boss.dms
    assert (
        "แก้เป็น" in line and "ส่งใบเสนอราคาแก้ไข" in line and "2026-10-20" in line and task.id in line
    )


async def test_an_unreadable_due_keeps_the_old_one(hs, world):
    task = new_task(hs)
    await hs._handshake_offer(GUILD, [task])
    click = Click(world.staff)
    await hs.handle_task_edit_submit(click, GUILD, task.id, task.title, "สักวันหนึ่ง", "")
    assert hs.tasks.get(GUILD, task.id).due == "2026-10-16"
    assert click.followups == [(t("hs_bad_due", "th"), True)]
    assert "รับงาน" in world.boss.dms[0][0] and "แก้เป็น" not in world.boss.dms[0][0]


async def test_not_mine_unassigns_keeps_the_written_name_and_tells_the_giver(hs, world):
    task = new_task(hs)
    await hs._handshake_offer(GUILD, [task])
    click = Click(world.staff)
    await hs.handle_task_answer(click, "no", GUILD, task.id)
    saved = hs.tasks.get(GUILD, task.id)
    assert (
        saved.confirm == "declined" and saved.assignee_id == 0 and saved.assignee_name == "Somchai"
    )
    assert t("hs_status_no", "th") in click.edited[0]["content"]
    ((line, _),) = world.boss.dms
    assert line == t("hs_giver_no", "th", name="Somchai", title=task.title, id=task.id)
    assert saved in hs.tasks.list(GUILD, assignee_id=0)  # counts as unassigned


async def test_no_giver_line_when_the_giver_is_the_assignee_or_unknown(hs, world):
    own = new_task(hs, "โน้ตถึงตัวเอง", assigner=STAFF)
    unknown = new_task(hs, "ไม่รู้ว่าใครสั่ง", assigner=0)
    await hs._handshake_offer(GUILD, [own, unknown])
    assert len(world.staff.dms) == 1  # the self-assigned note still gets its card
    for tk in (own, unknown):
        await hs.handle_task_answer(Click(world.staff), "ok", GUILD, tk.id)
    assert world.boss.dms == [] and world.staff.dms[1:] == []


async def test_group_members_confirm_separately_and_the_giver_sees_the_count(hs, world):
    a = new_task(hs, "เตรียมงานแสดงสินค้า", assignee=STAFF, group_id="grp00001")
    b = new_task(hs, "เตรียมงานแสดงสินค้า", assignee=INTERN, group_id="grp00001")
    c = new_task(hs, "เตรียมงานแสดงสินค้า", assignee=PEER, group_id="grp00001")
    await hs._handshake_offer(GUILD, [a, b, c])
    assert len(world.staff.dms) == len(world.intern.dms) == len(world.peer.dms) == 1
    await hs.handle_task_answer(Click(world.staff), "ok", GUILD, a.id)
    await hs.handle_task_answer(Click(world.intern), "ok", GUILD, b.id)
    lines = [text for text, _ in world.boss.dms]
    assert len(lines) == 2 and lines[0].endswith(t("hs_count", "th", n=1, total=3))
    assert lines[1].endswith(t("hs_count", "th", n=2, total=3))


async def test_a_team_nobody_could_pick_asks_the_giver_instead_of_sending_cards(hs, world):
    big = Task(guild_id=GUILD, title="ประชุมใหญ่", assignee_name="ทีม all", assigner_id=BOSS)
    big._needs_owner = True
    assert await hs._handshake_offer(GUILD, [big]) == 0
    ((line, _),) = world.boss.dms
    assert line == t("hs_team_unclear", "th", title="ประชุมใหญ่", id=big.id)


async def test_a_second_message_edit_keeps_the_other_cards_of_the_same_message(hs, world):
    a = new_task(hs, "ส่งใบเสนอราคาลูกค้า X", assigner=0)
    b = new_task(hs, "จองห้องประชุมไตรมาสหน้า", assigner=0)
    await hs._handshake_offer(GUILD, [a, b])
    click = Click(world.staff)
    await hs.handle_task_answer(click, "ok", GUILD, a.id)
    kw = click.edited[0]
    assert "ส่งใบเสนอราคาลูกค้า X" in kw["content"] and "จองห้องประชุมไตรมาสหน้า" in kw["content"]
    assert [x.item.custom_id.split(":")[-1] for x in kw["view"].children] == [b.id] * 3


# ------------------------------------------------------------------ the hook from the summary worker
async def test_the_summary_hook_schedules_the_cards(hs, world, monkeypatch):
    monkeypatch.setattr(hs, "loop", asyncio.get_running_loop())
    task = new_task(hs)
    hs._handshake_new_tasks(GUILD, [task])
    for _ in range(50):
        if world.staff.dms:
            break
        await asyncio.sleep(0.01)
    assert len(world.staff.dms) == 1


# ------------------------------------------------------------------ the nudge
async def test_one_nudge_after_24_hours_never_two(hs, world):
    task = new_task(hs)
    await hs._handshake_offer(GUILD, [task])
    stamp = hs.tasks.get(GUILD, task.id).confirm_at
    assert await hs._handshake_due(hs.s.now()) == 0  # fresh
    later = hs.s.now() + timedelta(hours=25)
    assert await hs._handshake_due(later) == 1
    assert len(world.staff.dms) == 2 and world.staff.dms[1][0] == world.staff.dms[0][0]
    assert await hs._handshake_due(later + timedelta(hours=30)) == 0
    assert hs.tasks.get(GUILD, task.id).confirm_at == stamp  # still the original time
    assert len(world.staff.dms) == 2
    assert hs.docs.get(STATE, GUILD, f"handshake-nudge:{task.id}:doer:{STAFF}") is not None


async def test_answered_tasks_are_not_nudged(hs, world):
    task = new_task(hs)
    await hs._handshake_offer(GUILD, [task])
    await hs.handle_task_answer(Click(world.staff), "ok", GUILD, task.id)
    assert await hs._handshake_due(hs.s.now() + timedelta(days=3)) == 0


# ------------------------------------------------------------------ update_task(assignee=)
def coo(bot, asker, *, is_exec=False, calls=None):
    sent = [] if calls is None else calls
    return CooTools(
        guild_id=GUILD, asker_id=asker, is_exec=is_exec, scope=None, tasks=bot.tasks,
        people=PEOPLE, today=lambda: bot.s.now().date(), reassigned=sent.append,
    ), sent  # fmt: skip


def test_reassign_by_the_giver_or_an_executive_resends_the_card(hs):
    task = new_task(hs, assigner=BOSS)
    task.confirm, task.assignee_id = "declined", 0
    hs.tasks.save(task)
    tools, sent = coo(hs, INTERN)  # not the giver, not an executive
    assert "not allowed" in tools.update_task(task.id, assignee="Somchai")["error"]
    tools, sent = coo(hs, BOSS)
    res = tools.update_task(task.id, assignee="Somchai")
    saved = hs.tasks.get(GUILD, task.id)
    assert res["updated"]["assignee"] == "Somchai" and saved.assignee_id == STAFF
    assert saved.confirm == "" and saved.assigner_id == BOSS and [x.id for x in sent] == [task.id]
    tools, sent = coo(hs, BOSS)
    assert "cannot tell" in tools.update_task(task.id, assignee="ไม่มีคนนี้")["error"]
    assert "give a status" in tools.update_task(task.id)["error"]


def test_the_giver_taking_a_task_back_for_themself_needs_no_card(hs):
    task = new_task(hs, assigner=BOSS)
    tools, sent = coo(hs, BOSS)
    tools.update_task(task.id, assignee="Boss")
    assert hs.tasks.get(GUILD, task.id).confirm == "accepted" and sent == []


def test_update_task_status_rules_are_unchanged(hs):
    task = new_task(hs, assignee=STAFF)
    tools, _ = coo(hs, STAFF)
    assert tools.update_task(task.id, "done")["updated"]["status"] == "done"
    tools, _ = coo(hs, PEER)
    assert "not allowed" in tools.update_task(task.id, "open")["error"]
    tools, _ = coo(hs, STAFF)
    assert "must be open or done" in tools.update_task(task.id, "bogus")["error"]
    fns = {f.__name__: f for f in build_coo_tools(tools)}
    assert "assignee" in fns["update_task"].__doc__


async def test_reassigning_through_the_bot_sends_the_card_to_the_new_person(hs, world, monkeypatch):
    monkeypatch.setattr(hs, "loop", asyncio.get_running_loop())
    task = new_task(hs, assignee=0, assigner=BOSS)
    mem = await hs._memory_tools(SimpleNamespace(guild=None, channel=None), world.g, world.boss)
    fns = {f.__name__: f for f in hs._coo_tools(world.g, world.boss, True, mem, "q", "th")}
    res = await asyncio.to_thread(fns["update_task"], task.id, assignee="Somchai")
    assert res["updated"]["assignee"] == "Somchai"
    for _ in range(50):
        if world.staff.dms:
            break
        await asyncio.sleep(0.01)
    assert len(world.staff.dms) == 1 and hs.tasks.get(GUILD, task.id).confirm == "pending"


# ------------------------------------------------------------------ tool-created tasks
def project_ctx(bot, world, asker_id, on_assigned):
    from salid.projects import Person, ProjectsCtx

    return ProjectsCtx(
        guild_id=GUILD, asker=Person(discord_user_id=str(asker_id), name="Boss"), is_admin=True,
        lang="th", repo=bot.project_repo, tasks=bot.tasks, now=bot.s.now,
        resolve=lambda name: [Person(discord_user_id=str(i), name=n) for i, n in PEOPLE.items() if n.lower() == name.lower()],
        on_assigned=on_assigned,
    )  # fmt: skip


def test_project_add_task_for_someone_else_sends_a_card_and_for_oneself_is_accepted(hs, world):
    from salid.projects import Person, Project, project_tools

    owner = Person(discord_user_id=str(BOSS), name="Boss")
    hs.project_repo.create(GUILD, Project(name="เว็บใหม่", department="tech", owner=owner))
    pid = hs.project_repo.list(GUILD)[0].id
    cards = []
    fn = {f.__name__: f for f in project_tools(project_ctx(hs, world, BOSS, cards.append))}
    res = fn["project_add_task"]("เว็บใหม่", "ออกแบบหน้าแรก", assignee="Somchai")
    task = hs.tasks.get(GUILD, res["id"])
    assert task.assigner_id == BOSS and task.assignee_id == STAFF and task.confirm == ""
    assert [x.id for x in cards] == [task.id] and task.project_id == pid
    mine = fn["project_add_task"]("เว็บใหม่", "ตรวจงาน", assignee="Boss")
    assert hs.tasks.get(GUILD, mine["id"]).confirm == "accepted" and len(cards) == 1
    nobody = fn["project_add_task"]("เว็บใหม่", "ยังไม่มีเจ้าของ")
    assert hs.tasks.get(GUILD, nobody["id"]).assignee_id == 0 and len(cards) == 1


# ------------------------------------------------------------------ markers
def test_pending_tasks_show_a_marker(hs, world):
    task = new_task(hs)
    assert H.pending_marker(task, "th") == ""
    task.confirm = "pending"
    assert H.pending_marker(task, "th") == t("hs_pending_marker", "th")
    task.confirm = "awaiting_approval"
    assert H.pending_marker(task, "en") == t("hs_awaiting_marker", "en")
    tools, _ = coo(hs, BOSS, is_exec=True)
    view = tools._view(hs.tasks.save(task))
    assert view["confirm"] == "awaiting_approval" and view["assigned_by"] == "Boss"


# ------------------------------------------------------------------ approve mode
APPROVE = dict(task_confirm_mode="approve")


async def accepted_waiting(hs, world, **kw):
    task = new_task(hs, **kw)
    await hs._handshake_offer(GUILD, [task])
    return task


async def test_approve_mode_accept_goes_to_the_boss_who_approves_and_the_assignee_is_told(
    hs, world
):
    set_settings(hs, **APPROVE)
    task = await accepted_waiting(hs, world)
    click = Click(world.staff)
    await hs.handle_task_answer(click, "ok", GUILD, task.id)
    saved = hs.tasks.get(GUILD, task.id)
    assert saved.confirm == "awaiting_approval" and saved.approver_id == BOSS
    assert (
        t("hs_status_wait", "th") in click.edited[0]["content"] and click.edited[0]["view"] is None
    )
    ((card, kw),) = world.boss.dms
    assert "Somchai" in card and task.title in card and task.id in card
    assert [b.act for b in kw["view"].children] == ["ap", "ae", "sb"]
    assert hs.tasks.get(GUILD, task.id).approval_message  # the card is remembered
    assert len(world.staff.dms) == 1  # only the original card: nothing yet
    stranger = Click(world.peer)
    await hs.handle_task_approve(stranger, GUILD, task.id)
    assert stranger.sent == [(t("hs_not_yours", "th"), True)]
    boss = Click(world.boss)
    await hs.handle_task_approve(boss, GUILD, task.id)
    assert hs.tasks.get(GUILD, task.id).confirm == "accepted"
    assert t("hs_boss_ok", "th") in boss.edited[0]["content"] and boss.edited[0]["view"] is None
    assert world.staff.dms[-1][0] == t(
        "hs_to_assignee_ok", "th", title=task.title, due="2026-10-16", id=task.id
    )
    await hs.handle_task_approve(Click(world.boss), GUILD, task.id)  # decided once
    assert len(world.staff.dms) == 2


async def test_the_boss_card_shows_the_assignees_edits_before_and_after(hs, world):
    set_settings(hs, **APPROVE)
    task = await accepted_waiting(hs, world)
    await hs.handle_task_edit_submit(Click(world.staff), GUILD, task.id, "ชื่อใหม่", "2026-10-22", "")
    ((card, _),) = world.boss.dms
    assert "ชื่อใหม่" in card and task.title in card and "2026-10-16" in card and "2026-10-22" in card
    assert hs.tasks.get(GUILD, task.id).title == "ชื่อใหม่"


async def test_the_boss_edit_is_final_and_the_assignee_hears_the_final_values(hs, world):
    set_settings(hs, **APPROVE)
    task = await accepted_waiting(hs, world)
    await hs.handle_task_answer(Click(world.staff), "ok", GUILD, task.id)
    opener = Click(world.boss)
    await hs.handle_task_modal_open(opener, "ae", GUILD, task.id)
    assert opener.modals[0].boss is True
    click = Click(world.boss)
    await hs.handle_task_boss_edit_submit(click, GUILD, task.id, "ชื่อจากหัวหน้า", "2026-10-30", "")
    saved = hs.tasks.get(GUILD, task.id)
    assert (saved.title, saved.due, saved.confirm) == ("ชื่อจากหัวหน้า", "2026-10-30", "accepted")
    assert t("hs_boss_edited", "th") in click.edited[0]["content"]
    final = world.staff.dms[-1][0]
    assert "ชื่อจากหัวหน้า" in final and "2026-10-30" in final and "หัวหน้า" in final


async def test_send_back_returns_the_card_with_the_note_and_stops_after_three_rounds(hs, world):
    set_settings(hs, **APPROVE)
    task = await accepted_waiting(hs, world)
    for rnd in range(1, 4):
        await hs.handle_task_answer(Click(world.staff), "ok", GUILD, task.id)
        opener = Click(world.boss)
        await hs.handle_task_modal_open(opener, "sb", GUILD, task.id)
        assert opener.modals
        click = Click(world.boss)
        await hs.handle_task_sendback_submit(click, GUILD, task.id, f"แก้รอบที่ {rnd}")
        saved = hs.tasks.get(GUILD, task.id)
        assert saved.confirm == "pending" and saved.approval_rounds == rnd
        assert t("hs_boss_back", "th") in click.edited[0]["content"]
        text, kw = world.staff.dms[-1]
        assert f"แก้รอบที่ {rnd}" in text and [b.act for b in kw["view"].children] == [
            "ok",
            "edit",
            "no",
        ]
    # the fourth send-back: no new card, the task stays pending and an executive hears once
    await hs.handle_task_answer(Click(world.staff), "ok", GUILD, task.id)
    n_staff, n_owner = len(world.staff.dms), len(world.owner.dms) + len(world.admin.dms)
    click = Click(world.boss)
    await hs.handle_task_sendback_submit(click, GUILD, task.id, "อีกแล้ว")
    saved = hs.tasks.get(GUILD, task.id)
    assert saved.confirm == "pending" and saved.approval_rounds == 3
    assert len(world.staff.dms) == n_staff
    assert click.followups == [(t("hs_back_cap", "th", n=3), True)]
    execs = [m for m in (world.owner, world.admin) if m.dms]
    assert execs and "#" + task.id in execs[0].dms[-1][0]
    await hs.handle_task_answer(Click(world.staff), "ok", GUILD, task.id)
    await hs.handle_task_sendback_submit(Click(world.boss), GUILD, task.id, "ครั้งที่ห้า")
    assert sum(len(m.dms) for m in (world.owner, world.admin)) == n_owner + 1  # told once


async def test_not_mine_behaves_the_same_in_approve_mode(hs, world):
    set_settings(hs, **APPROVE)
    task = await accepted_waiting(hs, world)
    await hs.handle_task_answer(Click(world.staff), "no", GUILD, task.id)
    assert hs.tasks.get(GUILD, task.id).confirm == "declined"
    assert world.boss.dms[-1][0] == t(
        "hs_giver_no", "th", name="Somchai", title=task.title, id=task.id
    )


async def test_the_boss_is_the_giver_then_the_project_owner_then_an_executive_then_notify(
    hs, world
):
    from salid.projects import Person, Project

    set_settings(hs, **APPROVE)
    hs.project_repo.create(
        GUILD,
        Project(
            name="เว็บใหม่", department="tech", owner=Person(discord_user_id=str(PEER), name="Peer")
        ),
    )
    pid = hs.project_repo.list(GUILD)[0].id
    boss_of = lambda task: getattr(hs._handshake_boss(world.g, task, task.assignee_id), "id", None)  # noqa: E731
    with_giver = new_task(hs, "ก", assigner=BOSS, project_id=pid)
    assert boss_of(with_giver) == BOSS
    with_project = new_task(hs, "ข", assigner=0, project_id=pid)
    assert boss_of(with_project) == PEER
    no_project = new_task(hs, "ค", assigner=0)
    assert boss_of(no_project) is None  # an executive exists but has no private channel yet
    hs.private_chats.put(GUILD, BOSS, 555)
    assert boss_of(no_project) == BOSS
    self_given = new_task(hs, "ง", assigner=STAFF)
    assert boss_of(self_given) is None  # giver == assignee never needs approval
    set_settings(hs)
    assert boss_of(with_giver) is None  # notify mode: nobody signs off


async def test_with_nobody_to_approve_the_task_is_recorded_like_notify(hs, world):
    set_settings(hs, **APPROVE)
    task = await accepted_waiting(hs, world, assigner=0)
    await hs.handle_task_answer(Click(world.staff), "ok", GUILD, task.id)
    assert hs.tasks.get(GUILD, task.id).confirm == "accepted"


async def test_the_boss_is_nudged_once_after_24_hours(hs, world):
    set_settings(hs, **APPROVE)
    task = await accepted_waiting(hs, world)
    await hs.handle_task_answer(Click(world.staff), "ok", GUILD, task.id)
    later = hs.s.now() + timedelta(hours=25)
    assert await hs._handshake_due(later) == 1
    assert len(world.boss.dms) == 2 and len(world.staff.dms) == 1
    assert await hs._handshake_due(later + timedelta(days=2)) == 0


async def test_notify_mode_is_unchanged(hs, world):
    task = await accepted_waiting(hs, world)
    await hs.handle_task_answer(Click(world.staff), "ok", GUILD, task.id)
    assert hs.tasks.get(GUILD, task.id).confirm == "accepted"
    assert [t_[0] for t_ in world.boss.dms][0].startswith("✅")


# ------------------------------------------------------------------ org mode, personal mode, policy
def test_mode_switch_through_settings_and_the_hub_flag_parity(hs):
    assert hs.team.task_confirm_mode(GUILD) == "notify" and hs.team.personal_allowed(GUILD)
    hs.team.set_task_confirm_mode(GUILD, "approve")
    assert hs.team.task_confirm_mode(GUILD) == "approve"
    with pytest.raises(ValueError):
        hs.team.set_task_confirm_mode(GUILD, "dictator")
    hs.team.set_personal_settings(GUILD, False)
    assert not hs.team.personal_allowed(GUILD)
    set_settings(hs, task_confirm_mode="nonsense", personal_settings=False)
    assert hs.team.task_confirm_mode(GUILD) == "notify"
    assert "task_handshake" in FEATURES and Features().task_handshake is True


def test_the_hub_lists_the_flag_the_mode_and_the_policy():
    html = (Path(__file__).parents[2] / "src/meeting_scribe/static/hub.html").read_text("utf-8")
    for k in ("f_task_handshake_n", "f_task_handshake_d", "f_task_handshake_w", "task_mode_l",
              "task_mode_notify_l", "task_mode_approve_l", "personal_l", "personal_s"):  # fmt: skip
        assert k in html
    assert '{ id: "task_handshake" }' in html and html.count("task_handshake: true") >= 2
    assert "task_confirm_mode:" in html and "personal_settings:" in html
    pm = (Path(__file__).parents[2] / "src/meeting_scribe/static/pm.html").read_text("utf-8")
    assert "task_confirm_mode" in pm and "personal_settings" in pm and "tk_pending" in pm


def test_effective_mode_prefers_the_givers_personal_choice_when_allowed():
    assert H.effective_mode("notify", "approve", True) == "approve"
    assert H.effective_mode("approve", "notify", True) == "notify"
    assert H.effective_mode("approve", "", True) == "approve"
    assert H.effective_mode("approve", "notify", False) == "approve"


async def test_personal_mode_wins_when_allowed_and_the_org_mode_when_the_person_has_none(hs, world):
    hs.team.set_task_confirm_mode(GUILD, "notify")
    ProfileTools(hs.profiles, GUILD, BOSS).remember("task_confirm_mode", "approve")
    task = new_task(hs, assigner=BOSS)
    assert hs._handshake_mode(GUILD, task) == "approve"
    other = new_task(hs, "งานอื่น", assigner=OWNER_ID)
    assert hs._handshake_mode(GUILD, other) == "notify"  # no personal mode: the org's
    hs.team.set_task_confirm_mode(GUILD, "approve")
    ProfileTools(hs.profiles, GUILD, BOSS).remember("task_confirm_mode", "แจ้ง")
    assert (
        hs._handshake_mode(GUILD, task) == "notify"
        and hs._handshake_mode(GUILD, other) == "approve"
    )
    hs.team.set_personal_settings(GUILD, False)  # policy off: the org mode only
    assert hs._handshake_mode(GUILD, task) == "approve"


def test_profile_round_trip_for_the_task_mode():
    p = Profile()
    assert p.task_confirm_mode == "" and p.is_empty()
    assert apply_change(p, "task_confirm_mode", "ขออนุมัติ") == "" and p.task_confirm_mode == "approve"
    assert not p.is_empty()
    assert apply_change(p, "task_confirm_mode", "notify") == "" and p.task_confirm_mode == "notify"
    assert "must be notify or approve" in apply_change(p, "task_confirm_mode", "x")
    assert apply_change(p, "task_confirm_mode", "") == "" and p.task_confirm_mode == ""
    repo = ProfileRepo(SQLiteDocStore(":memory:"))
    p.task_confirm_mode = "approve"
    repo.save(GUILD, BOSS, p)
    assert repo.get(GUILD, BOSS).task_confirm_mode == "approve"


def test_the_policy_off_refuses_personal_settings_and_ignores_stored_ones(hs, world):
    tools = ProfileTools(hs.profiles, GUILD, BOSS, lambda: hs._personal_allowed(GUILD))
    for kind, value in (("task_confirm_mode", "approve"), ("language", "en"), ("tone", "calm")):
        assert "error" not in tools.remember(kind, value)
    assert tools.remember("nickname", "พี่บอส")["status"] == "ok"
    p = hs._profile_of(GUILD, BOSS)
    assert (p.task_confirm_mode, p.language, p.tone) == ("approve", "en", "calm")
    hs.team.set_personal_settings(GUILD, False)
    gated = hs._profile_of(GUILD, BOSS)
    assert (gated.task_confirm_mode, gated.language, gated.tone) == ("", None, None)
    assert gated.nickname == "พี่บอส"  # only the three personal settings are gated
    res = tools.remember("language", "th")
    assert "shared setting" in res["error"] and "การตั้งค่ากลาง" in res["error"]
    assert hs._person_lang(GUILD, BOSS) == hs._person_lang(GUILD, INTERN)  # org language for all
    assert "error" not in tools.remember("nickname", "x")


async def reply_of(bot, world, who, text, channel=None):
    from test_modes import Chan, TMsg

    msg = TMsg(world.g, channel or world.pub, who, text)
    msg.mentions = []
    handled = await bot._task_mode_command(msg)
    return handled, msg.replies


async def test_chat_command_admins_set_the_server_mode_members_are_refused(hs, world):
    handled, replies = await reply_of(hs, world, world.admin, "โหมดมอบหมายงานแบบอนุมัติ")
    assert handled and replies == [t("hs_mode_set_approve", "th")]
    assert hs.team.task_confirm_mode(GUILD) == "approve"
    handled, replies = await reply_of(hs, world, world.admin, "task mode notify")
    assert handled and hs.team.task_confirm_mode(GUILD) == "notify"
    handled, replies = await reply_of(hs, world, world.peer, "โหมดมอบหมายงานแบบอนุมัติ")
    assert handled and replies == [t("hs_mode_denied", "th")]
    assert hs.team.task_confirm_mode(GUILD) == "notify"
    assert (await reply_of(hs, world, world.admin, "สวัสดี"))[0] is False


async def test_chat_command_personal_form_and_the_policy_refusal(hs, world):
    handled, replies = await reply_of(hs, world, world.peer, "งานที่ฉันมอบหมาย ขออนุมัติก่อน")
    assert handled and replies == [t("hs_mode_me_approve", "th")]
    assert hs._profile_of(GUILD, PEER).task_confirm_mode == "approve"
    handled, replies = await reply_of(hs, world, world.peer, "my task mode notify")
    assert handled and hs._profile_of(GUILD, PEER).task_confirm_mode == "notify"
    handled, _ = await reply_of(hs, world, world.peer, "งานที่ฉันมอบหมาย แค่แจ้งก็พอ")
    assert handled and hs._profile_of(GUILD, PEER).task_confirm_mode == "notify"
    hs.team.set_personal_settings(GUILD, False)
    handled, replies = await reply_of(hs, world, world.peer, "my task mode approve")
    assert handled and replies == [t("personal_off", "th")]
    assert hs.profiles.get(GUILD, PEER).task_confirm_mode == "notify"  # not saved


@pytest.mark.parametrize(
    "text, expected",
    [
        ("โหมดมอบหมายงานแบบอนุมัติ", ("approve", False)),
        ("โหมดมอบหมายงานแบบแจ้ง", ("notify", False)),
        ("task mode approve", ("approve", False)),
        ("task mode notify", ("notify", False)),
        ("my task mode approve", ("approve", True)),
        ("my task mode notify", ("notify", True)),
        ("งานที่ฉันมอบหมาย ขออนุมัติก่อน", ("approve", True)),
        ("งานที่ฉันมอบหมาย แค่แจ้งก็พอ", ("notify", True)),
        ("พรุ่งนี้มีประชุม", None),
        ("task mode", None),
    ],
)
def test_parse_task_mode_command(text, expected):
    assert H.parse_task_mode_command(text) == expected


def test_translations_cover_every_new_string():
    locales = Path(__file__).parents[2] / "src/salid/locales"
    for lang in ("zh", "ja"):
        strings = json.loads((locales / f"{lang}.json").read_text("utf-8"))["strings"]
        missing = [k for k in (*H.STRINGS, "personal_off") if k not in strings]
        assert not missing, (lang, missing)


# ------------------------------------------------------------------ add_task (tag / reply)
async def add_tools(hs, world, asker=None, monkeypatch=None):
    asker = asker or world.boss
    mem = await hs._memory_tools(SimpleNamespace(guild=None, channel=None), world.g, asker)
    tools = hs._coo_tools_in(PUBLIC, world.g, asker, True, mem, "q", "th")
    return {f.__name__: f for f in tools}


async def settle(*persons):
    for _ in range(80):
        if all(p.dms for p in persons):
            return
        await asyncio.sleep(0.01)


async def test_add_task_creates_a_task_for_the_giver_and_sends_the_card(hs, world, monkeypatch):
    monkeypatch.setattr(hs, "loop", asyncio.get_running_loop())
    fns = await add_tools(hs, world)
    res = await asyncio.to_thread(fns["add_task"], "ทำสไลด์นำเสนอ", "Somchai", "2026-10-23")
    task = hs.tasks.get(GUILD, res["task"]["id"])
    assert (task.assignee_id, task.assigner_id, task.created_from) == (STAFF, BOSS, "manual")
    assert (task.due, task.source_channel_id, task.project_id) == ("2026-10-23", PUBLIC, "")
    assert res["unresolved"] == [] and res["asked_to_confirm"] == ["Somchai"]
    await settle(world.staff)
    assert len(world.staff.dms) == 1 and "ทำสไลด์นำเสนอ" in world.staff.dms[0][0]
    assert hs.tasks.get(GUILD, task.id).confirm == "pending"
    # the passive summary pass later hears the same item: no second copy
    tr = TaskTracker(hs.tasks, hs.s, None, people_of=lambda gid: PEOPLE)
    again = tr.ingest(
        GUILD,
        PUBLIC,
        "2026-10-14",
        {"action_items": [{"who": "Somchai", "what": "ทำสไลด์นำเสนอ"}]},
        [],
    )
    assert again == [] and len(hs.tasks.list(GUILD, status=None)) == 1


async def test_add_task_resolves_mentions_splits_people_and_keeps_unresolved_names(
    hs, world, monkeypatch
):
    monkeypatch.setattr(hs, "loop", asyncio.get_running_loop())
    fns = await add_tools(hs, world)
    one = await asyncio.to_thread(fns["add_task"], "ตรวจบัญชีรายเดือน", f"<@{INTERN}>")
    assert hs.tasks.get(GUILD, one["task"]["id"]).assignee_id == INTERN
    many = await asyncio.to_thread(fns["add_task"], "จัดบูธงานแสดงสินค้า", "Somchai and Ploy")
    rows = [hs.tasks.get(GUILD, x["id"]) for x in many["tasks"]]
    assert (
        sorted(r.assignee_id for r in rows) == [STAFF, INTERN]
        and len({r.group_id for r in rows}) == 1
    )
    mixed = await asyncio.to_thread(fns["add_task"], "เขียนรายงานประจำปี", "Somchai, คนที่ไม่มี")
    assert mixed["unresolved"] == ["คนที่ไม่มี"] and len(mixed["tasks"]) == 1
    nobody = await asyncio.to_thread(fns["add_task"], "หาของขวัญปีใหม่", "คนที่ไม่มี")
    saved = hs.tasks.get(GUILD, nobody["task"]["id"])
    assert (
        saved.assignee_id == 0
        and saved.assignee_name == "คนที่ไม่มี"
        and nobody["unresolved"] == ["คนที่ไม่มี"]
    )
    empty = await asyncio.to_thread(fns["add_task"], "งานที่ยังไม่มีเจ้าของเลย")
    assert hs.tasks.get(GUILD, empty["task"]["id"]).assignee_id == 0 and empty["unresolved"] == []
    await asyncio.sleep(0.3)
    assert len(world.staff.dms) == 2 and len(world.intern.dms) == 2  # nobody else got a card


async def test_add_task_taking_it_yourself_is_accepted_and_bad_input_is_an_error(hs, world):
    fns = await add_tools(hs, world)
    mine = await asyncio.to_thread(fns["add_task"], "จดบันทึกประชุม", "Boss")
    assert (
        hs.tasks.get(GUILD, mine["task"]["id"]).confirm == "accepted" and "asked_to_confirm" in mine
    )
    assert world.staff.dms == []
    assert "title" in (await asyncio.to_thread(fns["add_task"], "  "))["error"]
    bad = await asyncio.to_thread(fns["add_task"], "งาน", "Somchai", "สักวันหนึ่ง")
    assert "due date" in bad["error"]


async def test_add_task_links_a_project_the_asker_may_use(hs, world):
    from salid.projects import Person, Project

    owner = Person(discord_user_id=str(BOSS), name="Boss")
    hs.project_repo.create(GUILD, Project(name="เว็บใหม่", department="tech", owner=owner))
    pid = hs.project_repo.list(GUILD)[0].id
    fns = await add_tools(hs, world)
    res = await asyncio.to_thread(fns["add_task"], "ออกแบบหน้าแรก", "Somchai", "", "เว็บ")
    assert hs.tasks.get(GUILD, res["task"]["id"]).project_id == pid
    assert (
        "no project"
        in (await asyncio.to_thread(fns["add_task"], "x", "", "", "ไม่มีโปรเจกต์นี้"))["error"]
    )
    plain_fns = {f.__name__: f for f in hs._coo_tools(world.g, world.peer, False, None, "q", "th")}
    res = await asyncio.to_thread(plain_fns["add_task"], "x", "", "", "เว็บ")
    assert "not allowed" in res["error"]


def test_add_task_is_a_voice_action_tool_and_the_prompt_mentions_it():
    from salid import qa
    from salid.voice_actions import ACTION_TOOL_NAMES

    assert "add_task" in ACTION_TOOL_NAMES
    assert "add_task" in qa.COO_PROMPT and "@Salid note the tasks above" in qa.COO_PROMPT
    assert "add_task" in H.HANDSHAKE_PROMPT
