# ruff: noqa: F811, F401
"""Polls (native Discord polls) and DM forms: validation, limits, confirm gates, scoping, the
answer gate, reminders, results visibility and files. Offline with fake Discord objects."""

import io
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import discord
import pytest
from conftest import GUILD
from test_coordinator import ADMIN_ID, BOSS, INTERN, NS, OWNER_ID, PEER, STAFF, person, world
from test_modes import Chan, TMsg, bot  # noqa: F401  (fixture)

from salid import forms as F
from salid import polls as P
from salid.guild_settings import GuildSettingsUpdate
from salid.i18n import t
from salid.permissions import SALID_PERMISSIONS

NOW = datetime(2026, 10, 16, 9, 0, tzinfo=UTC)


# ------------------------------------------------------------------ fakes
class PChan:
    def __init__(self, id, name, can_poll=True, senders=None):
        self.id, self.name, self.can_poll, self.senders = id, name, can_poll, senders
        self.posts, self.mention, self.log = [], f"<#{id}>", []

    def permissions_for(self, m):
        send = self.senders is None or m.id in self.senders
        if m.id == 999:
            return SimpleNamespace(view_channel=True, read_messages=True, send_messages=True,
                                   send_polls=self.can_poll)  # fmt: skip
        return SimpleNamespace(view_channel=True, read_messages=True, send_messages=send,
                               send_polls=True)  # fmt: skip

    async def send(self, content=None, poll=None, **kw):
        sent = SimpleNamespace(id=5000 + len(self.posts), jump_url="https://x/1")
        self.posts.append(SimpleNamespace(content=content, poll=poll, msg=sent))
        return sent


@pytest.fixture
def pw(world):
    g = world.g
    g.me = NS(id=999)
    g.roles = [NS(name="@everyone", is_default=lambda: True, members=[])]
    g.name = "A"
    g.sales = [world.staff, world.peer]
    g.roles.append(NS(name="Sales", is_default=lambda: False, members=g.sales))
    here = PChan(70, "general")
    other = PChan(71, "announce", senders={BOSS})
    for c in (here, other):
        g.text_channels.append(c)
    g.get_channel = lambda cid: next((c for c in g.text_channels if c.id == cid), None)
    g.get_member = lambda uid: g.members.get(uid)
    return SimpleNamespace(g=g, here=here, other=other, w=world)


def msg_in(pw, chan, who=STAFF):
    return SimpleNamespace(guild=pw.g, channel=chan, author=pw.g.members[who])


def conf(bot, who, req):
    from salid.secretary import fingerprint

    return lambda *parts: bot.secretary_state.confirm(
        ("pollform", GUILD, who, fingerprint(*parts)), req
    )


# ------------------------------------------------------------------ pure: polls
def test_clean_poll_limits():
    assert P.clean_poll("กินไหน", ["A", "B", "a"], 24) == ("กินไหน", ["A", "B"], 24)
    assert P.clean_poll("q", ["A"], 24)["error"] == "need_at_least_2_options"
    assert P.clean_poll("q", [str(i) for i in range(11)], 24)["error"] == "too_many_options"
    assert P.clean_poll("q", ["A", "x" * 56], 24)["error"] == "too_long"
    assert P.clean_poll("", ["A", "B"], 24)["error"] == "need_question"
    assert P.clean_poll("q", ["A", "B"], 99999)[2] == P.MAX_HOURS
    assert P.clean_poll("q", ["A", "B"], 0)[2] == P.MIN_HOURS


def test_result_line_winner_tie_none():
    assert "B" in P.result_line("q", ["A", "B"], [1, 4], "en") and "4 of 5" in P.result_line(
        "q", ["A", "B"], [1, 4], "en"
    )
    assert "A / B" in P.result_line("q", ["A", "B"], [2, 2], "en")
    assert "nobody" in P.result_line("q", ["A", "B"], [0, 0], "en")
    assert "ชนะ" in P.result_line("q", ["A", "B"], [1, 4], "th")


def test_poll_repo_find_last_text_and_id(bot):
    repo = P.PollRepo(bot.docs)
    for i, q in enumerate(["lunch?", "friday party"]):
        repo.put(P.new_record(guild_id=1, channel_id=2, message_id=100 + i, creator_id=3,
                              question=q, options=["a", "b"], hours=1, multiple=False,
                              now=NOW + timedelta(minutes=i)))  # fmt: skip
    assert repo.find(1, "last")[0]["question"] == "friday party"
    assert repo.find(1, "lunch")[0]["id"] == "100"
    assert repo.find(1, "https://discord.com/channels/1/2/101")[0]["id"] == "101"
    assert repo.find(1, "zzz")[0] is None and repo.find(2, "last") == (None, [])
    assert repo.find(1, "last", visible=lambda r: r["id"] == "100")[0]["id"] == "100"


# ------------------------------------------------------------------ polls: the tool
async def test_poll_in_current_channel(bot, pw):
    out = await bot._create_poll(
        pw.g, pw.g.members[STAFF], pw.here, "ร้านไหน", ["A", "B", "C"], None, 24, False, "th",
        conf(bot, STAFF, "r1"),
    )  # fmt: skip
    assert out["status"] == "posted"
    poll = pw.here.posts[0].poll
    assert isinstance(poll, discord.Poll) and poll.question == "ร้านไหน"
    assert [a.text for a in poll.answers] == [
        "A",
        "B",
        "C",
    ] and poll.duration.total_seconds() == 86400
    rec = bot.poll_repo.all(GUILD)[0]
    assert rec["message_id"] == pw.here.posts[0].msg.id and not rec["done"]


async def test_poll_missing_send_polls_uses_the_permission_message(bot, pw):
    pw.here.can_poll = False
    out = await bot._create_poll(
        pw.g, pw.g.members[STAFF], pw.here, "q", ["A", "B"], None, 24, False, "en",
        conf(bot, STAFF, "r1"),
    )  # fmt: skip
    assert out["error"] == "missing_permission" and "Send Polls" in out["how_to_grant"]
    assert "I'm missing the Send Polls permission" in out["how_to_grant"]
    assert not pw.here.posts


async def test_poll_forbidden_on_send_is_the_same_message(bot, pw):
    async def boom(**kw):
        raise discord.Forbidden(SimpleNamespace(status=403, reason="x"), "no")

    pw.here.send = boom
    out = await bot._create_poll(
        pw.g, pw.g.members[STAFF], pw.here, "q", ["A", "B"], None, 24, False, "th",
        conf(bot, STAFF, "r1"),
    )  # fmt: skip
    assert out["error"] == "missing_permission" and "Send Polls" in out["how_to_grant"]


async def test_poll_other_channel_needs_permission_and_confirmation(bot, pw):
    args = (pw.g, pw.g.members[STAFF], pw.here, "q", ["A", "B"], "announce", 24, False, "th")
    out = await bot._create_poll(*args, conf(bot, STAFF, "r1"))
    assert out["error"] == "not_allowed" and not pw.other.posts  # staff cannot post there
    boss = (pw.g, pw.g.members[BOSS], pw.here, "q", ["A", "B"], "announce", 24, False, "th")
    first = await bot._create_poll(*boss, conf(bot, BOSS, "r1"))
    assert first["status"] == "needs_confirmation" and not pw.other.posts
    second = await bot._create_poll(*boss, conf(bot, BOSS, "r2"))
    assert second["status"] == "posted" and len(pw.other.posts) == 1


async def test_poll_results_and_visibility(bot, pw):
    await bot._create_poll(
        pw.g, pw.g.members[STAFF], pw.here, "lunch", ["A", "B"], None, 24, False, "th",
        conf(bot, STAFF, "r1"),
    )  # fmt: skip
    answers = [SimpleNamespace(text="A", vote_count=1), SimpleNamespace(text="B", vote_count=3)]
    live = SimpleNamespace(answers=answers, is_finalised=lambda: False)

    async def fetch(mid):
        return SimpleNamespace(poll=live)

    pw.here.fetch_message = fetch
    out = await bot._poll_results(pw.g, pw.g.members[PEER], "the last poll")
    assert out["winners"] == ["B"] and out["total_votes"] == 4 and out["ended"] is False
    pw.here.permissions_for = lambda m: SimpleNamespace(view_channel=False, read_messages=False)
    assert (await bot._poll_results(pw.g, pw.g.members[PEER], "last"))["error"] == "poll_not_found"


async def test_ended_poll_posts_a_line_once_and_goes_to_memory(bot, pw, monkeypatch):
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(), "x")
    await bot._create_poll(
        pw.g, pw.g.members[STAFF], pw.here, "lunch", ["A", "B"], None, 1, False, "en",
        conf(bot, STAFF, "r1"),
    )  # fmt: skip
    poll = SimpleNamespace(
        answers=[SimpleNamespace(text="A", vote_count=0), SimpleNamespace(text="B", vote_count=2)],
        is_finalised=lambda: True,
    )

    async def fetch(mid):
        return SimpleNamespace(poll=poll)

    pw.here.fetch_message = fetch
    queued = []
    bot._memory_enqueue = queued.append
    rec = bot.poll_repo.all(GUILD)[0]
    await bot._poll_finish(pw.g, rec, NOW)
    assert "B wins" in pw.here.posts[-1].content and len(queued) == 1
    assert bot.poll_repo.all(GUILD)[0]["done"]
    later = datetime.fromisoformat(rec["ends_at"]) + timedelta(minutes=1)
    assert bot.poll_repo.due(GUILD, later) == []  # not posted twice


async def test_unfinalised_poll_waits_then_gives_up(bot, pw):
    await bot._create_poll(
        pw.g, pw.g.members[STAFF], pw.here, "q", ["A", "B"], None, 1, False, "en",
        conf(bot, STAFF, "r1"),
    )  # fmt: skip
    poll = SimpleNamespace(answers=[SimpleNamespace(text="A", vote_count=1),
                                    SimpleNamespace(text="B", vote_count=0)],
                           is_finalised=lambda: False)  # fmt: skip

    async def fetch(mid):
        return SimpleNamespace(poll=poll)

    pw.here.fetch_message = fetch
    bot._memory_enqueue = lambda job: None
    rec = bot.poll_repo.all(GUILD)[0]
    ends = datetime.fromisoformat(rec["ends_at"])
    n = len(pw.here.posts)
    await bot._poll_finish(pw.g, rec, ends + timedelta(minutes=1))
    assert len(pw.here.posts) == n and not bot.poll_repo.all(GUILD)[0]["done"]
    await bot._poll_finish(pw.g, rec, ends + P.GRACE + timedelta(minutes=1))
    assert len(pw.here.posts) == n + 1 and bot.poll_repo.all(GUILD)[0]["done"]


def test_send_polls_is_in_the_invite_permissions():
    assert SALID_PERMISSIONS & (1 << 49)


# ------------------------------------------------------------------ pure: forms
def mk(people=None, hours=48, now=NOW):
    return F.new_form(guild_id=GUILD, creator_id=1, creator_name="Boss", title="Office",
                      questions=["Q1", "Q2"], people=people or {10: "Ann", 11: "Bob"},
                      hours=hours, lang="en", now=now)  # fmt: skip


def test_recipient_limits():
    assert F.recipients_refusal(0, True)["error"] == "no_recipients"
    assert F.recipients_refusal(10, False) is None
    assert F.recipients_refusal(11, False)["error"] == "not_allowed"
    assert F.recipients_refusal(11, True) is None
    assert F.recipients_refusal(50, True) is None
    assert F.recipients_refusal(51, True)["error"] == "too_many_recipients"


def test_questions_validation():
    assert F.clean_questions([" a ", ""]) == ["a"]
    assert F.clean_questions([])["error"] == "need_questions"
    assert F.clean_questions(["q"] * 11)["error"] == "too_many_questions"


def test_state_machine_answers_skip_and_done():
    f = mk()
    F.mark_asked(f, 10, NOW)
    assert F.current_question(f, 10) == "Q1"
    assert F.record_answer(f, 10, "yes", NOW) == 1
    assert F.record_answer(f, 10, None, NOW) is None  # skipped, finished
    r = f["recipients"]["10"]
    assert r["state"] == F.DONE and r["answers"] == ["yes", None]
    assert F.current_question(f, 10) is None
    assert F.progress(f) == (1, 2) and F.waiting(f) == ["Bob"]


def test_asking_form_picks_the_latest_question_and_ignores_closed():
    a, b = mk(), mk(now=NOW + timedelta(minutes=5))
    F.mark_asked(a, 10, NOW)
    F.mark_asked(b, 10, NOW + timedelta(minutes=5))
    assert F.asking_form([a, b], 10, NOW + timedelta(hours=1))["id"] == b["id"]
    b["status"] = "closed"
    assert F.asking_form([a, b], 10, NOW + timedelta(hours=1))["id"] == a["id"]
    assert F.asking_form([a], 10, NOW + timedelta(hours=49)) is None  # past the deadline
    assert F.asking_form([a], 11, NOW) is None  # never asked


def test_reminder_once_before_deadline():
    f = mk(hours=48)
    F.mark_asked(f, 10, NOW)
    F.mark_asked(f, 11, NOW)
    assert F.due_reminders(f, NOW + timedelta(hours=10)) == []
    assert sorted(F.due_reminders(f, NOW + timedelta(hours=40))) == [10, 11]
    f["recipients"]["10"]["reminded"] = True
    F.record_answer(f, 11, "x", NOW)
    F.record_answer(f, 11, "y", NOW)
    assert F.due_reminders(f, NOW + timedelta(hours=40)) == []
    assert F.due_reminders(f, NOW + timedelta(hours=50)) == []  # closed by time


def test_results_visibility_is_hierarchy_only_upwards():
    f = mk()
    ranks = {1: 5, 2: 9, 3: 5, 4: 2}
    assert F.can_view_results(1, f, ranks)  # creator
    assert F.can_view_results(2, f, ranks)  # above
    assert not F.can_view_results(3, f, ranks)  # peer
    assert not F.can_view_results(4, f, ranks)  # below
    assert not F.can_view_results(99, f, ranks)  # unknown


def test_csv_and_text_and_pdf():
    f = mk()
    F.mark_asked(f, 10, NOW)
    F.record_answer(f, 10, "ไปได้", NOW)
    raw = F.csv_bytes(f, "en")
    assert raw.startswith(b"\xef\xbb\xbf") and "ไปได้" in raw.decode("utf-8-sig")
    assert "Ann" in F.text_table(f, "en") and "ไปได้" in F.text_table(f, "en")
    from salid.pdf_report import find_fonts

    pdf = F.form_pdf(f, org="A", lang="en", generated_at=NOW, fonts=find_fonts(""))
    assert pdf.startswith(b"%PDF")


def test_summary_hides_answers_unless_private():
    f = mk()
    F.mark_asked(f, 10, NOW)
    F.record_answer(f, 10, "secret", NOW)
    assert "people" not in F.summary_for_agent(f, "en", with_answers=False)
    assert F.summary_for_agent(f, "en", with_answers=True)["people"][0]["answers"][0] == "secret"


def test_stop_and_skip_regex():
    assert F.SKIP_RE.match("ข้าม") and F.SKIP_RE.match("Skip") and not F.SKIP_RE.match("skip lunch")
    assert (
        F.STOP_RE.match("stop") and F.STOP_RE.match("ไม่ขอตอบค่ะ") and not F.STOP_RE.match("stop by")
    )


def test_gate_fails_closed(monkeypatch):
    from salid import gate

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(gate, "_decide", boom)
    assert F.looks_like_answer(None, "m", "q", "a") is False
    monkeypatch.setattr(gate, "_decide", lambda c, m, p, s: s(answer=True))
    assert F.looks_like_answer(None, "m", "q", "a") is True


# ------------------------------------------------------------------ forms: starting
async def start(bot, pw, who=BOSS, req="r1", **kw):
    args = dict(title="Friday", questions=["ไปงานวันศุกร์ได้ไหม"], people=None, role="Sales",
                channel=None, hours=48)  # fmt: skip
    args.update(kw)
    return await bot._start_form(
        pw.g, pw.g.members[who], who == OWNER_ID, args["title"], args["questions"],
        args["people"], args["role"], args["channel"], args["hours"], "th", conf(bot, who, req),
    )  # fmt: skip


async def test_form_confirm_gate_then_sends(bot, pw):
    bot._sleep = lambda s: _noop()
    first = await start(bot, pw)
    assert first["status"] == "needs_confirmation" and first["recipient_count"] == 2
    assert not bot.form_repo.all(GUILD) and not pw.w.staff.dms
    again = await start(bot, pw, req="r1")  # same request id never confirms itself
    assert again["status"] == "needs_confirmation"
    ok = await start(bot, pw, req="r2")
    assert ok["status"] == "sending" and ok["people"] == 2
    await _drain(bot)
    assert "ไปงานวันศุกร์ได้ไหม" in pw.w.staff.dms[0][0] and "ข้อ 1/1" in pw.w.peer.dms[0][0]
    form = bot.form_repo.all(GUILD)[0]
    assert {r["state"] for r in form["recipients"].values()} == {F.ANSWERING}


async def _noop():
    return None


async def _drain(bot):
    import asyncio

    while bot._bg_tasks:
        await asyncio.gather(*list(bot._bg_tasks))


async def test_form_closed_dms_are_marked_and_the_creator_told(bot, pw):
    bot._sleep = lambda s: _noop()

    async def closed(text=None, **kw):
        raise discord.Forbidden(SimpleNamespace(status=403, reason="x"), "closed")

    pw.w.peer.send = closed
    await start(bot, pw)
    await start(bot, pw, req="r2")
    await _drain(bot)
    form = bot.form_repo.all(GUILD)[0]
    assert form["recipients"][str(PEER)]["state"] == F.UNREACHABLE
    assert any("Peer" in d[0] for d in pw.w.boss.dms)


async def test_form_limits_and_sensitive_and_unknown(bot, pw):
    big = {i: NS(id=i, bot=False, display_name=f"P{i}") for i in range(100, 112)}
    pw.g.roles.append(NS(name="Big", is_default=lambda: False, members=list(big.values())))
    refused = await start(bot, pw, who=STAFF, role="Big")
    assert refused["error"] == "not_allowed"  # 12 people, not a manager
    assert (await start(bot, pw, role="Big"))["status"] == "needs_confirmation"  # boss is exec role
    assert (await start(bot, pw, questions=["ช่วยโอนเงินให้หน่อย"]))["error"] == "refused_sensitive"
    assert (await start(bot, pw, role="Nope"))["error"] == "role_not_found"
    assert (await start(bot, pw, role=None))["error"] == "need_recipients"
    assert (await start(bot, pw, people=["Nobody"], role=None))["error"] == "person_not_found"
    assert (await start(bot, pw, questions=[]))["error"] == "need_questions"


async def test_form_hard_cap_50_even_for_executives(bot, pw):
    many = [NS(id=i, bot=False, display_name=f"P{i}") for i in range(200, 252)]
    pw.g.roles.append(NS(name="All", is_default=lambda: False, members=many))
    assert (await start(bot, pw, role="All"))["error"] == "too_many_recipients"


async def test_form_by_name_and_mention_excludes_the_asker_and_bots(bot, pw):
    out = await start(bot, pw, people=["Somchai", f"<@{PEER}>", "Boss"], role=None)
    assert out["recipient_count"] == 2  # Boss is the asker


# ------------------------------------------------------------------ forms: DM answers
async def open_form(bot, pw, questions=("Q1", "Q2")):
    bot._sleep = lambda s: _noop()
    await start(bot, pw, questions=list(questions))
    await start(bot, pw, questions=list(questions), req="r2")
    await _drain(bot)
    return bot.form_repo.all(GUILD)[0]


def dm_msg(pw, who, text):
    m = TMsg(None, Chan(900 + who), SimpleNamespace(id=who, bot=False), text)
    return m


async def test_dm_answer_flow_and_gate(bot, pw, monkeypatch):
    form = await open_form(bot, pw)
    member = pw.g.members[STAFF]
    monkeypatch.setattr(F, "looks_like_answer", lambda c, m, q, a: a != "what's the weather?")
    off = dm_msg(pw, STAFF, "what's the weather?")
    assert await bot._form_dm_answer(off, pw.g, member) is False and not off.replies  # normal chat
    a1 = dm_msg(pw, STAFF, "ไปได้ค่ะ")
    assert await bot._form_dm_answer(a1, pw.g, member) is True and "ข้อ 2/2" in a1.replies[0]
    skip = dm_msg(pw, STAFF, "ข้าม")
    assert await bot._form_dm_answer(skip, pw.g, member) is True
    assert pw.g.members[BOSS].display_name  # sanity
    saved = bot.form_repo.get(GUILD, form["id"])["recipients"][str(STAFF)]
    assert saved["answers"] == ["ไปได้ค่ะ", None] and saved["state"] == F.DONE
    assert "Boss" in skip.replies[0]  # thanks names the creator
    late = dm_msg(pw, STAFF, "hello again")
    assert await bot._form_dm_answer(late, pw.g, member) is False  # nothing open any more


async def test_dm_without_open_form_never_calls_the_gate(bot, pw, monkeypatch):
    def boom(*a):
        raise AssertionError("gate must not run")

    monkeypatch.setattr(F, "looks_like_answer", boom)
    assert await bot._form_dm_answer(dm_msg(pw, STAFF, "hi"), pw.g, pw.g.members[STAFF]) is False


async def test_dm_stop_opts_out(bot, pw):
    form = await open_form(bot, pw)
    m = dm_msg(pw, PEER, "ไม่ขอตอบค่ะ")
    assert await bot._form_dm_answer(m, pw.g, pw.g.members[PEER]) is True
    assert bot.form_repo.get(GUILD, form["id"])["recipients"][str(PEER)]["state"] == F.DECLINED


async def test_polls_feature_flag_off_means_no_tools_and_no_dm_hook(bot, pw):
    await open_form(bot, pw)
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={"polls": False}), "x")
    bot.team._cache.clear()
    assert (
        bot._poll_form_tools(msg_in(pw, pw.here), pw.g, pw.g.members[STAFF], False, "th", "r") == []
    )
    assert await bot._form_dm_answer(dm_msg(pw, STAFF, "yes"), pw.g, pw.g.members[STAFF]) is False


async def test_tools_are_registered(bot, pw):
    tools = bot._poll_form_tools(msg_in(pw, pw.here), pw.g, pw.g.members[STAFF], False, "th", "r")
    assert [f.__name__ for f in tools] == [
        "create_poll",
        "poll_results",
        "start_form",
        "form_results",
    ]


# ------------------------------------------------------------------ forms: timers
async def test_reminder_once_and_close_notifies_creator(bot, pw):
    form = await open_form(bot, pw)
    for r in pw.w.staff.dms, pw.w.peer.dms:
        r.clear()
    soon = datetime.fromisoformat(form["deadline"]) - timedelta(hours=2)
    import salid.polls_bot as pb

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return soon

    pb.datetime = Clock
    try:
        await bot._polls_forms_guild(pw.g)
        await bot._polls_forms_guild(pw.g)
    finally:
        pb.datetime = datetime
    assert len(pw.w.staff.dms) == 1 and len(pw.w.peer.dms) == 1  # exactly one reminder each
    after = datetime.fromisoformat(form["deadline"]) + timedelta(minutes=1)

    class Clock2(datetime):
        @classmethod
        def now(cls, tz=None):
            return after

    pb.datetime = Clock2
    try:
        await bot._polls_forms_guild(pw.g)
        await bot._polls_forms_guild(pw.g)
    finally:
        pb.datetime = datetime
    assert bot.form_repo.get(GUILD, form["id"])["status"] == "closed"
    assert sum("ปิดแล้ว" in d[0] for d in pw.w.boss.dms) == 1


# ------------------------------------------------------------------ forms: results
async def answered_form(bot, pw, monkeypatch):
    form = await open_form(bot, pw)
    monkeypatch.setattr(F, "looks_like_answer", lambda *a: True)
    for text in ("yes", "no"):
        await bot._form_dm_answer(dm_msg(pw, STAFF, text), pw.g, pw.g.members[STAFF])
    return form


async def test_results_creator_in_public_gets_counts_and_a_dm(bot, pw, monkeypatch):
    await answered_form(bot, pw, monkeypatch)
    pw.w.boss.dms.clear()
    out = await bot._form_results(pw.g, pw.g.members[BOSS], False, "last", "none", "en")
    assert out["finished"] == 1 and "people" not in out and out["sent_by_dm"]
    assert "Somchai" in pw.w.boss.dms[0][0] and "yes" in pw.w.boss.dms[0][0]


async def test_results_in_dm_returns_answers_and_files(bot, pw, monkeypatch):
    await answered_form(bot, pw, monkeypatch)
    pw.w.boss.dms.clear()
    out = await bot._form_results(pw.g, pw.g.members[BOSS], True, "Friday", "csv", "en")
    assert out["people"] and pw.w.boss.dms[0][1]["files"][0].filename.endswith(".csv")
    out = await bot._form_results(pw.g, pw.g.members[BOSS], True, "last", "pdf", "en")
    assert pw.w.boss.dms[-1][1]["files"][0].filename.endswith(".pdf")


async def test_results_hidden_from_peers_and_below_visible_to_above(bot, pw, monkeypatch):
    await answered_form(bot, pw, monkeypatch)
    for who in (STAFF, PEER, INTERN):
        out = await bot._form_results(pw.g, pw.g.members[who], True, "last", "none", "en")
        assert out["error"] == "form_not_found"  # does not even exist for them
    out = await bot._form_results(pw.g, pw.g.members[ADMIN_ID], True, "last", "none", "en")
    assert "people" in out
    out = await bot._form_results(pw.g, pw.g.members[OWNER_ID], True, "last", "none", "en")
    assert "people" in out
