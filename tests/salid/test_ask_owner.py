# ruff: noqa: F811, F401
"""Ask the speaker who a task is for: giver fallback, the question card, the three answers (buttons,
words, tool), the nudge and the markers. Offline, fake Discord objects."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
from conftest import GUILD
from test_coordinator import BOSS, INTERN, PEER, PEOPLE, PUBLIC, STAFF, bot, world  # noqa: F401
from test_task_handshake import (
    APPROVE,
    Click,
    TaskButton,
    button,
    coo,
    hs,
    last_view,
    msg,
    new_task,
    set_settings,
    tracker,
)

from salid import task_handshake as H
from salid.coo_store import SQLiteDocStore
from salid.hints import with_hint
from salid.i18n import t
from salid.tasks import Task


def ownerless(bot, title="ทำสไลด์นำเสนอลูกค้า", giver=STAFF, **kw):
    return new_task(bot, title, assignee=0, assigner=giver, **kw)


# ------------------------------------------------------------------ giver fallback
def test_giver_is_the_newest_speaker_sharing_words_with_the_title():
    tr = tracker(SQLiteDocStore(":memory:"))
    s = {"action_items": [{"who": "", "what": "ทำสไลด์นำเสนอลูกค้า"}]}
    ms = [
        msg(INTERN, "Ploy", "ทำสไลด์นำเสนอลูกค้า ด้วยนะ", 1),
        msg(STAFF, "Somchai", "ใครว่างบ้าง", 2),
    ]
    (task,) = tr.ingest(GUILD, PUBLIC, "2026-10-14", s, ms)
    assert task.assignee_id == 0 and task.assigner_id == INTERN and task.created_from != "dm"


def test_giver_stays_unknown_without_a_matching_message():
    tr = tracker(SQLiteDocStore(":memory:"))
    s = {"action_items": [{"who": "", "what": "ทำสไลด์นำเสนอลูกค้า"}]}
    (task,) = tr.ingest(GUILD, PUBLIC, "2026-10-14", s, [msg(STAFF, "Somchai", "สวัสดี", 1)])
    assert task.assigner_id == 0


# ------------------------------------------------------------------ the question card
async def test_question_is_sent_once_to_the_giver_with_persistent_buttons(hs, world):
    task = ownerless(hs)
    assert await hs._handshake_offer(GUILD, [task]) == 1
    ((text, kw),) = world.staff.dms
    assert "ทำสไลด์นำเสนอลูกค้า" in text and "ทั่วไป" in text and "❓" in text
    view = kw["view"]
    assert [b.act for b in view.children] == ["om", "oo", "on"] and view.timeout is None
    rx = TaskButton.__discord_ui_compiled_template__
    for b in view.children:
        m = rx.fullmatch(b.item.custom_id)
        again = await TaskButton.from_custom_id(None, b.item, m)
        assert (again.act, again.task_id) == (b.act, task.id)
    saved = hs.tasks.get(GUILD, task.id)
    assert saved.confirm == "owner_asked" and saved.owner_message.count(":") == 2
    assert saved.handshake_message == ""


async def test_no_question_for_dm_scope_unknown_giver_or_flag_off(hs, world):
    dm = hs.tasks.add(
        Task(guild_id=GUILD, assigner_id=STAFF, title="โน้ตส่วนตัว", source_channel_id=-STAFF,
             created_from="dm")
    )[0]  # fmt: skip
    nobody = ownerless(hs, "งานไม่มีคนสั่ง", giver=0)
    assert await hs._handshake_offer(GUILD, [dm, nobody]) == 0 and world.staff.dms == []
    ok = ownerless(hs, "งานที่ควรถาม")
    set_settings(hs, features={"task_handshake": False})
    assert await hs._handshake_offer(GUILD, [ok]) == 0 and world.staff.dms == []


async def test_questions_are_batched_five_per_message(hs, world):
    tasks = [ownerless(hs, f"รายการ{chr(65 + i) * 6} เรื่องพิเศษ{i * 7919}") for i in range(7)]
    assert await hs._handshake_offer(GUILD, tasks) == 2
    assert [len(kw["view"].children) for _, kw in world.staff.dms] == [15, 6]


async def test_mine_assigns_the_giver_and_sends_no_giver_line(hs, world):
    task = ownerless(hs)
    await hs._handshake_offer(GUILD, [task])
    click = Click(world.staff)
    await hs.handle_owner_answer(click, "om", GUILD, task.id)
    saved = hs.tasks.get(GUILD, task.id)
    assert (saved.assignee_id, saved.confirm) == (STAFF, "accepted")
    (edit,) = click.edited
    assert t("hs_owner_mine", "th", id=task.id) in edit["content"] and edit["view"] is None
    assert len(world.staff.dms) == 1 and world.boss.dms == []
    again = Click(world.staff)
    await hs.handle_owner_answer(again, "om", GUILD, task.id)
    assert again.sent == [(t("hs_handled", "th"), True)]


@pytest.mark.parametrize("mode", [{}, APPROVE])
async def test_someone_else_assigns_and_starts_the_normal_handshake(hs, world, mode):
    if mode:
        set_settings(hs, **mode)
    task = ownerless(hs)
    await hs._handshake_offer(GUILD, [task])
    modal_click = Click(world.staff)
    await hs.handle_owner_modal_open(modal_click, GUILD, task.id)
    assert len(modal_click.modals) == 1
    click = Click(world.staff)
    await hs.handle_owner_answer(click, "oo", GUILD, task.id, "Ploy")
    saved = hs.tasks.get(GUILD, task.id)
    assert saved.assignee_id == INTERN and saved.assigner_id == STAFF
    assert saved.confirm == "pending"  # the confirm card went out
    assert "Somchai" in world.intern.dms[-1][0] and last_view(world.intern).children
    assert t("hs_owner_other", "th", name="Ploy", id=task.id) in click.edited[0]["content"]
    await hs.handle_task_answer(Click(world.intern), "ok", GUILD, task.id)
    if mode:  # approve mode: the giver now gets the sign-off card
        assert hs.tasks.get(GUILD, task.id).confirm == "awaiting_approval"
        assert last_view(world.staff).children[0].act == "ap"
    else:
        assert hs.tasks.get(GUILD, task.id).confirm == "accepted"


async def test_an_unknown_name_keeps_the_card_waiting(hs, world):
    task = ownerless(hs)
    await hs._handshake_offer(GUILD, [task])
    click = Click(world.staff)
    await hs.handle_owner_answer(click, "oo", GUILD, task.id, "ไม่มีคนนี้")
    assert click.sent == [
        (with_hint(t("hs_owner_unknown", "th", name="ไม่มีคนนี้"), "mention", lang="th"), True)
    ]
    assert not click.edited and hs.tasks.get(GUILD, task.id).confirm == "owner_asked"


async def test_several_names_split_into_one_task_each(hs, world):
    task = ownerless(hs, giver=BOSS)
    await hs._handshake_offer(GUILD, [task])
    await hs.handle_owner_answer(Click(world.boss), "oo", GUILD, task.id, "Somchai and Ploy")
    rows = [x for x in hs.tasks.list(GUILD, status=None) if x.title == task.title]
    assert sorted(x.assignee_id for x in rows) == [STAFF, INTERN]
    assert len({x.group_id for x in rows}) == 1 and rows[0].group_id
    assert hs.tasks.get(GUILD, task.id).assignee_id in (STAFF, INTERN)
    assert world.staff.dms and world.intern.dms


async def test_not_a_task_drops_it_from_open_lists_and_reports(hs, world):
    task = ownerless(hs)
    await hs._handshake_offer(GUILD, [task])
    click = Click(world.staff)
    await hs.handle_owner_answer(click, "on", GUILD, task.id)
    saved = hs.tasks.get(GUILD, task.id)
    assert (saved.status, saved.note, saved.confirm) == ("done", "not a task", "declined")
    assert saved.done_at == "" and task.id not in [x.id for x in hs.tasks.list(GUILD)]
    assert t("hs_owner_none", "th", id=task.id) in click.edited[0]["content"]
    assert await hs._handshake_due(hs.s.now() + timedelta(days=3)) == 0


async def test_only_the_giver_or_an_executive_may_press(hs, world):
    task = ownerless(hs)
    await hs._handshake_offer(GUILD, [task])
    stranger = Click(world.peer)
    await hs.handle_owner_answer(stranger, "om", GUILD, task.id)
    assert stranger.sent == [(t("hs_owner_not_giver", "th"), True)]
    assert hs.tasks.get(GUILD, task.id).confirm == "owner_asked"
    await hs.handle_owner_answer(Click(world.boss), "on", GUILD, task.id)  # an executive
    assert hs.tasks.get(GUILD, task.id).status == "done"


async def test_one_nudge_after_24_hours_then_it_stays_unassigned(hs, world):
    task = ownerless(hs)
    await hs._handshake_offer(GUILD, [task])
    assert await hs._handshake_due(hs.s.now()) == 0
    later = hs.s.now() + timedelta(hours=25)
    assert await hs._handshake_due(later) == 1 and len(world.staff.dms) == 2
    assert await hs._handshake_due(later + timedelta(hours=30)) == 0
    saved = hs.tasks.get(GUILD, task.id)
    assert saved.confirm == "owner_asked" and saved.status == "open"


def test_marker_in_followups_and_the_company_report_input():
    task = Task(guild_id=GUILD, title="x", confirm="owner_asked")
    assert H.pending_marker(task, "th") == t("hs_owner_marker", "th")
    assert H.pending_marker(task, "en") == "❓ owner pending"


def test_old_documents_load_without_owner_message():
    assert Task(guild_id=GUILD, title="x").owner_message == ""


# ------------------------------------------------------------------ the words
@pytest.mark.parametrize(
    "text",
    ["อ๋อ อันนี้ไม่ใช่งานที่ต้องสั่งครับ", "ไม่ใช่ครับ", "ไม่ใช่งาน", "ไม่ต้องจด", "ลบทิ้ง",
     "not a task", "no, ignore that", "drop it"],
)  # fmt: skip
def test_parse_none(text):
    assert H.parse_owner_answer(text) == ("none", "")


@pytest.mark.parametrize(
    "text", ["ของฉัน", "ของผมครับ", "mine", "it's mine", "งาน #abcd1234 ของฉัน", "อ๋อ ผมเอง"]
)
def test_parse_mine(text):
    assert H.parse_owner_answer(text) == ("mine", "")


@pytest.mark.parametrize(
    "text, name",
    [("ให้บี", "บี"), ("ให้ Ploy ทำ", "ploy"), ("for Bee", "bee"), ("it's for Bee", "bee"),
     ("อ๋อ ให้เตยนะคะ", "เตย"), ("<@5>", "<@5>")],
)  # fmt: skip
def test_parse_other(text, name):
    assert H.parse_owner_answer(text) == ("other", name)


@pytest.mark.parametrize("text", ["สวัสดี", "ไม่ใช่ของฉัน", "พรุ่งนี้มีประชุมกับลูกค้า", ""])
def test_parse_nothing(text):
    assert H.parse_owner_answer(text) is None


async def say(hs, world, who, text, reply_to=None):
    from test_modes import Chan, TMsg

    m = TMsg(world.g, Chan(900 + who.id), who, text)
    m.mentions = []
    if reply_to is not None:
        m.reference = SimpleNamespace(message_id=reply_to, resolved=None)
    handled = await hs._owner_answer_command(m, world.g, who)
    return handled, m.replies


async def test_a_single_pending_task_is_dropped_by_words(hs, world):
    task = ownerless(hs)
    await hs._handshake_offer(GUILD, [task])
    handled, replies = await say(hs, world, world.staff, "อ๋อ อันนี้ไม่ใช่งานที่ต้องสั่งครับ")
    assert handled and replies == [t("hs_owner_say_none", "th")]
    saved = hs.tasks.get(GUILD, task.id)
    assert saved.status == "done" and saved.confirm == "declined"


async def test_words_for_mine_and_for_someone(hs, world):
    a, b = ownerless(hs, "งานแรกของลูกค้า"), None
    await hs._handshake_offer(GUILD, [a])
    handled, _ = await say(hs, world, world.staff, "ให้ Ploy")
    assert handled and hs.tasks.get(GUILD, a.id).assignee_id == INTERN
    b = ownerless(hs, "งานที่สองเรื่องบัญชี")
    await hs._handshake_offer(GUILD, [b])
    handled, _ = await say(hs, world, world.staff, "ของฉัน")
    assert handled and hs.tasks.get(GUILD, b.id).assignee_id == STAFF


async def test_a_reply_to_the_card_picks_that_task_among_several(hs, world):
    a, b = ownerless(hs, "งานแรกของลูกค้า"), ownerless(hs, "งานที่สองเรื่องบัญชี")
    await hs._handshake_offer(GUILD, [a])
    await hs._handshake_offer(GUILD, [b])
    card_id = int(hs.tasks.get(GUILD, b.id).owner_message.split(":")[1])
    handled, _ = await say(hs, world, world.staff, "ไม่ใช่งาน", reply_to=card_id)
    assert handled
    assert hs.tasks.get(GUILD, b.id).status == "done" and hs.tasks.get(GUILD, a.id).status == "open"


async def test_several_pending_without_a_clue_asks_which(hs, world):
    a, b = ownerless(hs, "งานแรกของลูกค้า"), ownerless(hs, "งานที่สองเรื่องบัญชี")
    await hs._handshake_offer(GUILD, [a, b])
    handled, replies = await say(hs, world, world.staff, "ของฉัน")
    assert handled and a.id in replies[0] and b.id in replies[0]
    assert {hs.tasks.get(GUILD, x.id).confirm for x in (a, b)} == {"owner_asked"}
    handled, _ = await say(hs, world, world.staff, f"งาน #{b.id} ของฉัน")
    assert handled and hs.tasks.get(GUILD, b.id).assignee_id == STAFF


async def test_other_messages_and_other_people_fall_through(hs, world):
    task = ownerless(hs)
    await hs._handshake_offer(GUILD, [task])
    assert (await say(hs, world, world.staff, "พรุ่งนี้มีประชุม"))[0] is False
    assert (await say(hs, world, world.intern, "ไม่ใช่งาน"))[0] is False  # not their question
    assert hs.tasks.get(GUILD, task.id).confirm == "owner_asked"


# ------------------------------------------------------------------ tool + text route
def tool(bot, asker, *, is_exec=False):
    calls = []
    c, _ = coo(bot, asker, is_exec=is_exec)
    c._owner_answered = lambda done, offer: calls.append((done, offer))
    return c, calls


async def test_answer_task_question_tool(hs, world):
    task = ownerless(hs)
    await hs._handshake_offer(GUILD, [task])
    c, calls = tool(hs, INTERN)
    assert "not allowed" in c.answer_task_question(task.id, "mine")["error"]
    c, calls = tool(hs, STAFF)
    assert "answer must be" in c.answer_task_question(task.id, "x")["error"]
    assert "name" in c.answer_task_question(task.id, "other")["error"]
    assert "cannot tell" in c.answer_task_question(task.id, "other", "ไม่มีคนนี้")["error"]
    res = c.answer_task_question(task.id, "other", "Ploy")
    assert res["updated"][0]["assignee"] == "Ploy" and calls[0][1][0].id == task.id
    assert "not waiting" in c.answer_task_question(task.id, "mine")["error"]
    other = ownerless(hs, "งานอื่นเรื่องคลัง")
    await hs._handshake_offer(GUILD, [other])
    c, _ = tool(hs, BOSS, is_exec=True)  # an executive may
    assert "error" not in c.answer_task_question(other.id, "none")
    assert hs.tasks.get(GUILD, other.id).status == "done"


async def test_update_task_mine_assigns_the_asker_and_listing_marks_it(hs, world):
    task = ownerless(hs)
    await hs._handshake_offer(GUILD, [task])
    c, _ = coo(hs, STAFF)
    view = c._view(hs.tasks.get(GUILD, task.id))
    assert view["owner_pending"] is True
    res = c.update_task(task.id, assignee="me")
    saved = hs.tasks.get(GUILD, task.id)
    assert res["updated"]["assignee"] == "Somchai" and saved.assignee_id == STAFF
    assert saved.confirm == "accepted"
    other = ownerless(hs, "งานอื่นเรื่องคลัง")
    c.update_task(other.id, assignee="Ploy")
    assert hs.tasks.get(GUILD, other.id).assignee_id == INTERN
