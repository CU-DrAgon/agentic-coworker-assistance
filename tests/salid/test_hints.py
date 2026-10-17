# ruff: noqa: F811, F401
"""Reply hints: the shared vocabulary, the help command, and every message that expects typing
carries its hint (cards, questions, lists, reports, reminders, welcome). Offline, fake Discord."""

from datetime import date
from pathlib import Path

import pytest
from conftest import GUILD
from test_ask_owner import ownerless
from test_coordinator import BOSS, PEOPLE, PUBLIC, STAFF, bot, world  # noqa: F401
from test_task_handshake import (
    APPROVE,
    Click,
    coo,
    hs,
    last_view,
    new_task,
    set_settings,
)

from salid import hints as HI
from salid import reminders as R
from salid import reports as rep
from salid import task_handshake as H
from salid.hints import hint, with_hint, with_legend
from salid.i18n import CATALOG, EXTRA, LANGS, t
from salid.tasks import Task

ROOT = Path(__file__).parents[2]


# ------------------------------------------------------------------ the vocabulary
@pytest.mark.parametrize("key", HI.LINE_HINTS)
def test_every_hint_exists_in_thai_and_english_on_one_line(key):
    for lang in ("th", "en"):
        line = hint(key, lang)
        assert line.strip() and "\n" not in line
    assert "ℹ️" not in hint(key, "en")  # the prefix is added by with_hint, not stored


def test_the_catalog_has_every_hint_key_and_the_translations_cover_them():
    keys = [f"hint_{k}" for k in HI.LINE_HINTS] + list(HI.STRINGS)
    for k in keys:
        assert k in CATALOG and CATALOG[k]["th"] and CATALOG[k]["en"], k
    import json

    for lang in EXTRA:
        strings = json.loads((ROOT / f"src/salid/locales/{lang}.json").read_text("utf-8"))[
            "strings"
        ]
        missing = [k for k in HI.STRINGS if k not in strings]
        assert not missing, (lang, missing)


def test_with_hint_appends_once_and_never_duplicates():
    out = with_hint("hello", "mention", "mention", "task_id", lang="en")
    assert out.splitlines() == [
        "hello",
        "ℹ️ " + hint("mention", "en"),
        "ℹ️ " + hint("task_id", "en"),
    ]
    assert with_hint(out, "mention", "task_id", lang="en") == out  # already there
    assert with_hint("hello", lang="en") == "hello"
    assert with_hint("hello\n", "task_id", lang="th").startswith("hello\nℹ️ ")


def test_the_legend_uses_the_same_emoji_and_words_as_the_markers():
    for lang in ("th", "en"):
        line = HI.legend(lang)
        for key in HI.MARKER_KEYS:
            assert t(key, lang) in line
        assert all(e in line for e in HI.MARKER_EMOJI)
    task = Task(guild_id=1, title="x", confirm="pending")
    assert H.pending_marker(task, "th") == "⏳ รอยืนยัน"
    task.confirm = "awaiting_approval"
    assert H.pending_marker(task, "en") == "🔏 awaiting approval"
    task.confirm = "owner_asked"
    assert H.pending_marker(task, "th") == "❓ รอระบุผู้รับ"
    assert HI.overdue_marker("th") == "⚠️ เลยกำหนด" and t("rep_overdue", "th").startswith("⚠️")
    assert t("cu_late", "en").startswith("⚠️") and t("rep_done", "en").startswith("✅")


def test_with_legend_only_when_a_marker_is_present():
    assert with_legend("nothing special", "en") == "nothing special"
    out = with_legend(f"- #a1 do it {HI.overdue_marker('en')}", "en")
    assert out.endswith(HI.legend("en")) and out.count("ℹ️") == 1
    assert with_legend(out, "en") == out
    assert with_legend("plain", "en", markers=True).endswith(HI.legend("en"))


def test_help_guide_is_short_and_has_the_legend_and_the_symbols():
    for lang in ("th", "en"):
        guide = HI.help_tasks(lang)
        assert len(guide.splitlines()) <= 12
        assert HI.legend(lang) in guide and "#" in guide and "@" in guide and "/pm" in guide


@pytest.mark.parametrize(
    "text",
    [
        "ใช้งานยังไง",
        "สลิด ใช้งานยังไง",
        "สลิด ใช้งานยังไงคะ",
        "ช่วยอธิบายสัญลักษณ์",
        "สลิด ช่วยอธิบายสัญลักษณ์หน่อย",
        "ตอบยังไง",
        "help",
        "Salid help",
        "salid, help please",
        "how do I answer",
        "How do I use Salid?",
        "what do the symbols mean",
        "explain the symbols",
    ],
)
def test_parse_help_command(text):
    assert HI.parse_help_command(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "สวัสดี",
        "ช่วยสรุปประชุมให้หน่อย",
        "help me write a long email to the customer about the delayed order",
        "ใช้งานยังไงกับระบบบัญชีของบริษัทเรา ช่วยเขียนขั้นตอนให้ละเอียดหน่อย",
        "#a1b2c3 เสร็จแล้ว",
    ],
)
def test_parse_help_command_ignores_other_talk(text):
    assert not HI.parse_help_command(text)


# ------------------------------------------------------------------ the cards
def hint_lines(text):
    return [x for x in text.splitlines() if x.startswith("ℹ️")]


async def test_confirm_card_has_the_answer_and_task_id_hints(hs, world):
    await hs._handshake_offer(GUILD, [new_task(hs, "งานแรก")])  # the first card is special
    task = new_task(hs)
    await hs._handshake_offer(GUILD, [task])
    text = world.staff.dms[-1][0]
    assert f"งาน #{task.id}" in text  # the card shows the id the hint talks about
    assert hint("answer_confirm", "th") in text and hint("task_id", "th") in text
    assert len(hint_lines(text)) <= 2 and len(text) < 2000
    edited = Click(world.staff)
    await hs.handle_task_answer(edited, "ok", GUILD, task.id)
    assert "ℹ️" not in edited.edited[0]["content"]  # answered: no hint left on the card


async def test_first_card_gets_one_extra_line_later_cards_do_not(hs, world):
    first = new_task(hs, "งานแรก")
    await hs._handshake_offer(GUILD, [first])
    text1 = world.staff.dms[0][0]
    assert hint("first_time", "th") in text1 and hint("answer_confirm", "th") in text1
    assert len(hint_lines(text1)) == 2
    second = new_task(hs, "งานที่สอง")
    await hs._handshake_offer(GUILD, [second])
    text2 = world.staff.dms[1][0]
    assert hint("first_time", "th") not in text2 and hint("task_id", "th") in text2
    # a nudge of the first card is the same first card
    assert hs._first_card(GUILD, STAFF, frozenset({first.id})) is False  # `second` has confirm
    only = {first.id, second.id}
    assert hs._first_card(GUILD, STAFF, frozenset(only)) is True


async def test_a_second_person_has_their_own_first_card(hs, world):
    a = new_task(hs, "ของเตย", assignee=STAFF)
    await hs._handshake_offer(GUILD, [a])
    b = new_task(hs, "ของอีกคน", assignee=world.peer.id)
    await hs._handshake_offer(GUILD, [b])
    assert hint("first_time", "th") in world.peer.dms[0][0]


async def test_owner_question_card_has_answer_and_mention_hints(hs, world):
    task = ownerless(hs)
    await hs._handshake_offer(GUILD, [task])
    ((text, _),) = world.staff.dms
    assert hint("answer_owner", "th") in text and hint("mention", "th") in text
    assert len(hint_lines(text)) == 2
    # the final line of an answered card carries no hint
    done = hs.tasks.get(GUILD, task.id)
    done.status, done.note, done.confirm = "done", H.NOT_A_TASK, H.DECLINED
    assert "ℹ️" not in hs._owner_text(world.g, [done], "th")


async def test_boss_card_has_the_answer_hint_until_it_is_decided(hs, world):
    set_settings(hs, **APPROVE)
    task = new_task(hs)
    await hs._handshake_offer(GUILD, [task])
    await hs.handle_task_answer(Click(world.staff), "ok", GUILD, task.id)
    ((card, kw),) = world.boss.dms
    assert hint("answer_boss", "th") in card and [b.act for b in kw["view"].children] == [
        "ap",
        "ae",
        "sb",
    ]
    boss = Click(world.boss)
    await hs.handle_task_approve(boss, GUILD, task.id)
    assert "ℹ️" not in boss.edited[0]["content"]


async def test_giver_lines_asking_for_an_owner_carry_reassign_and_mention(hs, world):
    task = new_task(hs)
    await hs._handshake_offer(GUILD, [task])
    await hs.handle_task_answer(Click(world.staff), "no", GUILD, task.id)
    line = world.boss.dms[-1][0]
    assert hint("reassign", "th") in line and hint("mention", "th") in line
    big = hs.tasks.add(
        Task(guild_id=GUILD, title="ประชุมใหญ่", assigner_id=BOSS, source_channel_id=PUBLIC)
    )[0]
    big._needs_owner = True
    await hs._handshake_ask_owner(world.g, big)
    assert hint("reassign", "th") in world.boss.dms[-1][0]


async def test_which_task_question_and_unknown_name_carry_hints(hs, world):
    from test_ask_owner import say

    a, b = ownerless(hs, "งานแรกของลูกค้า"), ownerless(hs, "งานที่สองเรื่องบัญชี")
    await hs._handshake_offer(GUILD, [a, b])
    handled, replies = await say(hs, world, world.staff, "ของฉัน")
    assert handled and hint("task_id", "th") in replies[0]
    hs.tasks.get(GUILD, a.id)
    handled, replies = await say(hs, world, world.staff, f"งาน #{a.id} ให้ ไม่มีคนนี้")
    assert handled and hint("mention", "th") in replies[0]


def test_modal_placeholders_show_the_formats(hs):
    task = Task(guild_id=GUILD, title="x", due="2026-10-16")
    from salid.task_handshake_bot import OwnerModal, TaskEditModal

    edit = TaskEditModal(GUILD, task, "th")
    assert edit.f_due.placeholder == hint("ph_due", "th")
    owner = OwnerModal(GUILD, task, "en")
    assert owner.f_who.placeholder == hint("ph_who", "en")


# ------------------------------------------------------------------ lists, tools, reports
def test_list_tasks_returns_a_legend_only_when_a_marker_is_shown(hs, world):
    new_task(hs, "ไม่มีเครื่องหมาย")
    tools, _ = coo(hs, STAFF)
    res = tools.list_tasks(status="all")
    assert "legend" not in res and res["tasks"][0]["marker"] is None
    waiting = new_task(hs, "รอยืนยันอยู่")
    waiting.confirm = "pending"
    hs.tasks.save(waiting)
    res = tools.list_tasks(status="all")
    assert res["legend"] == HI.legend("en")
    marked = [v for v in res["tasks"] if v["marker"]]
    assert [v["marker"] for v in marked] == [t("hs_pending_marker", "en")]
    tools.lang = "th"
    assert tools.list_tasks(status="all")["legend"] == HI.legend("th")


def test_the_prompt_tells_the_model_to_show_the_legend_and_answer_symbol_questions():
    assert "legend" in H.HANDSHAKE_PROMPT and HI.help_tasks("th") in H.HANDSHAKE_PROMPT
    assert HI.help_tasks("en") in H.HANDSHAKE_PROMPT


def company(people=(), unassigned=()):
    ledger = {
        "today_by_currency": {},
        "today_count": 0,
        "month_by_currency": {},
        "review": 0,
        "duplicates": 0,
    }
    return rep.CompanyInput("2026-10-17", ledger, [], list(people), list(unassigned))


def test_daily_report_ends_with_the_legend_only_when_a_marker_is_present():
    quiet = rep.render_plain(company(), "th")
    assert HI.legend("th") not in quiet and "ℹ️" not in quiet
    late = rep.PersonDay(1, "Boss", overdue=["ส่งงาน (2026-10-10)"], open_n=1)
    text = rep.render_plain(company([late]), "th")
    assert "⚠️ เลยกำหนด" in text and text.splitlines()[-1] == "ℹ️ " + HI.legend("th")
    ownerless_line = rep.render_plain(company(unassigned=["x [❓ owner pending]"]), "en")
    assert ownerless_line.splitlines()[-1] == "ℹ️ " + HI.legend("en")


def test_personal_report_uses_the_shared_markers_and_legend():
    data = {
        "date": "2026-10-17",
        "messages_by_channel": {},
        "messages": [],
        "tasks_done_today": [],
        "tasks_open": [
            {"id": "a1", "title": "late", "due": "2026-10-10", "overdue": True, "confirm": ""},
            {"id": "b2", "title": "wait", "due": "", "overdue": False, "confirm": "pending"},
        ],
    }
    text = rep.render_personal(data, "en")
    assert "#a1 late (2026-10-10) ⚠️ overdue" in text and "#b2 wait ⏳ awaiting confirmation" in text
    assert text.splitlines()[-1] == "ℹ️ " + HI.legend("en")
    data["tasks_open"] = []
    assert "ℹ️" not in rep.render_personal(data, "en")


async def test_followup_dm_has_task_id_and_legend(hs, world):
    task = new_task(hs)
    task.confirm = "pending"
    hs.tasks.save(task)
    await hs._send_followups(world.g, date(2026, 10, 16))
    text = world.staff.dms[-1][0]
    assert hint("task_id", "th") in text and HI.legend("th") in text


# ------------------------------------------------------------------ reminders, welcome, help
def test_reminder_time_errors_show_the_formats_other_errors_do_not():
    from test_reminders import make_ctx

    ctx = make_ctx(lang="th")
    for code in ("no_time", "bad_time"):
        res = R._err(ctx, R.ReminderError(code))
        assert hint("time_formats", "th") in res["reason"], code
    res = R._err(ctx, R.ReminderError("past"))
    assert hint("time_formats", "th") not in res["reason"]


async def say_help(bot, world, who, text, channel=None):
    from test_modes import TMsg

    msg = TMsg(world.g, channel or world.pub, who, text)
    msg.mentions = []
    return await bot._help_command(msg), msg.replies


async def test_help_command_answers_in_the_channel_language(hs, world):
    handled, replies = await say_help(hs, world, world.staff, "สลิด ใช้งานยังไง")
    assert handled and replies == [HI.help_tasks("th")]
    set_settings(hs, default_language="en")
    handled, replies = await say_help(hs, world, world.staff, "Salid help")
    assert handled and replies == [HI.help_tasks("en")]
    handled, replies = await say_help(hs, world, world.staff, "สวัสดี")
    assert not handled and replies == []


async def test_help_command_in_a_private_channel_uses_the_persons_language(hs, world):
    from test_modes import TMsg

    msg = TMsg(world.g, world.pub, world.staff, "ช่วยอธิบายสัญลักษณ์")
    msg.mentions = []
    assert await hs._help_command(msg, world.g, world.staff)
    assert msg.replies == [HI.help_tasks("th")]


def test_the_welcome_texts_point_to_the_help_command():
    for lang in ("th", "en"):
        assert "Salid help" in hint("welcome_help", "en")
        assert hint("welcome_help", lang) in with_hint(
            t("pc_welcome", lang, mention="@x"), "welcome_help", lang=lang
        )
    assert "สลิด ใช้งานยังไง" in hint("welcome_help", "th")


# ------------------------------------------------------------------ docs and /pm
def test_the_pm_settings_page_has_the_legend_block():
    page = (ROOT / "src/meeting_scribe/static/pm.html").read_text("utf-8")
    for key in HI.MARKER_KEYS:
        assert t(key, "th") in page and t(key, "en") in page, key
    assert "s_legend" in page


def test_the_readme_has_the_how_to_answer_section():
    readme = (ROOT / "README.md").read_text("utf-8")
    assert "### How to answer Salid" in readme
    for key in HI.MARKER_KEYS:
        assert t(key, "en") in readme and t(key, "th") in readme, key


def test_my_projects_and_tasks_returns_the_legend_only_with_a_marker(hs, world):
    from test_task_handshake import project_ctx

    from salid.projects import project_tools

    fn = {f.__name__: f for f in project_tools(project_ctx(hs, world, STAFF, lambda t_: None))}
    task = new_task(hs)
    res = fn["my_projects_and_tasks"]()
    assert "legend" not in res and res["tasks"][0]["marker"] is None
    task.confirm = "pending"
    hs.tasks.save(task)
    res = fn["my_projects_and_tasks"]()
    assert res["legend"] == HI.legend("th") and res["tasks"][0]["marker"] == "⏳ รอยืนยัน"
