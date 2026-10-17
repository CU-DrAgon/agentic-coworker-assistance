# ruff: noqa: F811, F401
"""Projects that update themselves from normal chat (`salid.projects_listen`): what is applied, what
is dropped, the one note per batch (throttle, DM scope, flag), model failure, the bot glue.
Fake LLM and fake Discord objects only."""

import asyncio
import re
from datetime import UTC
from types import SimpleNamespace

import pytest
from conftest import FakeClient, FixedClockSettings, json_resp
from fake_firestore import FakeFirestore
from test_briefings import set_features
from test_coordinator import PUBLIC, STAFF, at, world
from test_modes import bot  # noqa: F401  (fixture)

from salid import projects as P
from salid.bot import SecretaryBot
from salid.coo_store import FirestoreDocStore, SQLiteDocStore
from salid.i18n import CATALOG, t
from salid.memory import build_message
from salid.projects_listen import MAX_UPDATES_PER_BATCH, NOTE_GAP_S, ProjectListener
from salid.tasks import TaskRepo

GUILD = 1
CHAN = 10
DAY = "2026-10-14"
TOEY = P.Person(email="toey@x.com", discord_user_id="26", name="Toey")
PEOPLE = {26: "Toey", 27: "Peat"}


@pytest.fixture(params=["sqlite", "firestore"])
def docs(request):
    if request.param == "sqlite":
        return SQLiteDocStore(":memory:")
    return FirestoreDocStore(client=FakeFirestore())


def msg(i, text, uid=26, name="Toey", channel=CHAN, kind="chat"):
    s = FixedClockSettings()
    m = build_message(
        s, guild_id=GUILD, channel_id=channel, message_id=i, author_id=uid, author_name=name,
        created_at=at(11), text=text, kind=kind,
    )  # fmt: skip
    return m.model_copy(update={"jump_url": f"https://discord.com/channels/1/{channel}/{i}"})


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Rig:
    def __init__(self, docs, script, **kw):
        self.s = FixedClockSettings()
        self.repo = P.ProjectRepo(docs, clock=self.s.now)
        self.llm = FakeClient([json_resp(x) if isinstance(x, dict) else x for x in script])
        self.notes, self.clock = [], Clock()
        self.l = ProjectListener(
            self.repo, TaskRepo(docs), self.s, self.llm, people_of=lambda gid: PEOPLE,
            note=lambda g, c, text: self.notes.append((g, c, text)),
            lang_of=lambda g, c: "th", clock=self.clock, **kw,
        )  # fmt: skip

    def project(self, **kw):
        base = dict(name="PM Dashboard", owner=TOEY, department="tech", status="in_progress")
        return self.repo.create(GUILD, P.Project(**{**base, **kw}))

    def hear(self, *messages, channel=CHAN):
        return self.l.on_summary(GUILD, channel, DAY, {}, list(messages))

    def get(self, p):
        return self.repo.get(GUILD, p.id)


def upd(p, kind, value="", quote="", **kw):
    return {
        "project_id": p.id,
        "kind": kind,
        "value": value,
        "quote": quote,
        "speaker": "Toey",
        **kw,
    }


# ------------------------------------------------------------------ applied from chat
def test_a_status_change_is_applied_with_the_quoted_line_as_the_reason(docs):
    r = Rig(docs, [])
    p = r.project()
    line = "PM Dashboard เสร็จแล้ว ส่งงานไปแล้ว"
    r.llm.models.script = [json_resp({"updates": [upd(p, "status", "completed", line)]})]
    out = r.hear(msg(1, line))
    got = r.get(p)
    assert got.status == "completed" and len(out) == 1
    ch = got.changes[-1]
    assert ch.field == "status" and ch.from_ == "in_progress" and ch.to == "completed"
    assert line in ch.reason and ch.reason.startswith("heard in chat")
    assert ch.by.discord_user_id == "26" and ch.by.name == "Toey"  # matched to the member
    assert len(r.llm.models.calls) == 1 and r.llm.models.calls[0][0] == r.s.gate_model
    assert r.notes == [(GUILD, CHAN, t("pl_noted", "th", items="PM Dashboard → เสร็จสิ้น"))]


def test_the_prompt_carries_projects_with_ids_open_milestones_and_the_messages(docs):
    r = Rig(docs, [{"updates": []}])
    p = r.project(
        milestones=[
            P.Milestone(id="m1", title="Beta"),
            P.Milestone(id="m2", title="Old", status="done"),
        ],
        risks=[P.Risk(id="r1", title="Quota")],
        deadline="2026-11-01",
    )
    r.hear(msg(1, "คุยเรื่องอื่น"), msg(2, "salid says", kind="salid"))
    prompt = r.llm.models.calls[0][1]
    assert f"{p.id} | PM Dashboard | in_progress | medium | 2026-11-01 | m1: Beta | Quota" in prompt
    assert "Old" not in prompt and "คุยเรื่องอื่น" in prompt and "salid says" not in prompt
    assert DAY in prompt


def test_deadline_priority_and_two_statements_last_one_wins(docs):
    r = Rig(docs, [])
    p = r.project(deadline="2026-11-01")
    l1, l2, l3 = "เลื่อนเดดไลน์ไป 30 พ.ย.", "ตอนนี้เร่งด่วนมาก", "เลื่อนอีกทีเป็น 5 ธ.ค."
    r.llm.models.script = [
        json_resp(
            {
                "updates": [
                    upd(p, "deadline", "2026-11-30", l1),
                    upd(p, "priority", "high", l2),
                    upd(p, "deadline", "2026-12-05", l3),
                ]
            }
        )
    ]
    out = r.hear(msg(1, l1), msg(2, l2), msg(3, l3))
    got = r.get(p)
    assert got.deadline == "2026-12-05" and got.priority == "high" and len(out) == 2
    assert [c.field for c in got.changes] == ["priority", "deadline"] or {
        c.field for c in got.changes
    } == {"deadline", "priority"}
    assert "5 ธ.ค." in r.notes[0][2] and "30 พ.ย." not in r.notes[0][2]


def test_milestone_done_and_new_risk_and_duplicate_risk(docs):
    r = Rig(docs, [])
    p = r.project(
        milestones=[P.Milestone(id="m1", title="Beta"), P.Milestone(id="m2", title="Launch")],
        risks=[P.Risk(id="r1", title="Quota หมดบ่อย")],
    )
    a, b, c = "milestone beta เสร็จแล้ว", "ความเสี่ยง: quota Nano Banana หมดบ่อย", "ระวัง quota หมดบ่อย"
    r.llm.models.script = [
        json_resp(
            {
                "updates": [
                    upd(p, "milestone_done", "", a, milestone_id="m1"),
                    upd(p, "risk", "Nano Banana quota", b),
                    upd(p, "risk", "quota หมดบ่อย", c),
                    upd(p, "milestone_done", "", a, milestone_id="m9"),  # unknown milestone
                ]
            }
        )
    ]
    out = r.hear(msg(1, a), msg(2, b), msg(3, c))
    got = r.get(p)
    assert [m.status for m in got.milestones] == ["done", "pending"]
    assert [x.title for x in got.risks] == ["Quota หมดบ่อย", "Nano Banana quota"]
    new = got.risks[-1]
    assert new.id == "r2" and new.source == "team" and new.severity == "medium" and new.note == b
    assert len(out) == 2
    # the same risk again (any case) is not added twice
    r.llm.models.script = [json_resp({"updates": [upd(p, "risk", "NANO BANANA QUOTA", b)]})]
    assert r.hear(msg(4, b)) == [] and len(r.get(p).risks) == 2


def test_a_note_is_kept_on_the_project_capped_and_one_per_batch(docs):
    r = Rig(docs, [])
    p = r.project()
    p.updates = [P.Update(at="2026-10-01 09:00", text=f"old {i}") for i in range(50)]
    r.repo.save(GUILD, p)
    q1, q2 = "หน้า dashboard ใช้งานได้แล้ว", "เหลือแก้ไอคอน"
    r.llm.models.script = [
        json_resp(
            {"updates": [upd(p, "note", "Dashboard usable", q1), upd(p, "note", "Icons left", q2)]}
        )
    ]
    r.hear(msg(1, q1), msg(2, q2))
    ups = r.get(p).updates
    assert len(ups) == 50 and ups[-1].text == "Dashboard usable" and ups[0].text == "old 1"
    u = ups[-1]
    assert u.quote == q1 and u.channel_id == CHAN and u.jump_url.endswith("/1")
    assert u.by.name == "Toey" and u.at


def test_changes_are_saved_once_per_project(docs, monkeypatch):
    r = Rig(docs, [])
    p = r.project()
    q1, q2 = "ทำเสร็จแล้ว", "เร่งด่วนนะ"
    r.llm.models.script = [
        json_resp({"updates": [upd(p, "status", "completed", q1), upd(p, "priority", "high", q2)]})
    ]
    saves = []
    orig = r.repo.save
    monkeypatch.setattr(r.repo, "save", lambda g, x, **k: saves.append(x.id) or orig(g, x, **k))
    r.hear(msg(1, q1), msg(2, q2))
    assert saves == [p.id]
    # a completed project took both changes, and updated_at moved
    assert r.get(p).status == "completed" and r.get(p).priority == "high"


# ------------------------------------------------------------------ dropped
def test_unknown_ids_same_status_bad_values_and_unquoted_updates_are_dropped(docs):
    r = Rig(docs, [])
    p = r.project()
    q = "คุยกันว่าจะเลื่อน"
    r.llm.models.script = [
        json_resp(
            {
                "updates": [
                    {**upd(p, "status", "completed", q), "project_id": "p-nope"},
                    upd(p, "status", "in_progress", q),  # the current status
                    upd(p, "status", "finished", q),  # not an enum value
                    upd(p, "priority", "urgent", q),
                    upd(p, "deadline", "30 Nov", q),
                    upd(p, "deadline", "2026-11-30", "ไม่เคยพูดประโยคนี้"),  # not in the messages
                    upd(p, "owner", "Peat", q),  # owners never change from chat
                    upd(p, "banana", "x", q),
                ]
            }
        )
    ]
    before = r.get(p).model_dump()
    assert r.hear(msg(1, q)) == []
    assert r.get(p).model_dump() == before and r.notes == []


def test_a_completed_project_only_takes_a_note(docs):
    r = Rig(docs, [])
    done = r.project(name="Old", status="completed", deadline="2026-09-01")
    q = "ขอเลื่อนเดดไลน์ Old ไป 30 พ.ย. และสรุปงานเก็บไว้"
    r.llm.models.script = [
        json_resp(
            {
                "updates": [
                    upd(done, "deadline", "2026-11-30", q),
                    upd(done, "status", "in_progress", q),
                    upd(done, "risk", "x", q),
                    upd(done, "note", "Wrap-up written", q),
                ]
            }
        )
    ]
    r.hear(msg(1, q))
    got = r.get(done)
    assert got.deadline == "2026-09-01" and got.status == "completed" and got.risks == []
    assert [u.text for u in got.updates] == ["Wrap-up written"]


def test_at_most_eight_updates_per_batch(docs):
    r = Rig(docs, [])
    p = r.project()
    ups = [upd(p, "risk", f"risk {i}", "พูดแล้ว") for i in range(12)]
    r.llm.models.script = [json_resp({"updates": ups})]
    r.hear(msg(1, "พูดแล้ว"))
    assert len(r.get(p).risks) == MAX_UPDATES_PER_BATCH


def test_a_cancelled_project_is_not_offered_to_the_model(docs):
    r = Rig(docs, [{"updates": []}])
    r.project(name="Dead", status="cancelled")
    live = r.project(name="Live")
    r.hear(msg(1, "อะไรก็ได้"))
    prompt = r.llm.models.calls[0][1]
    assert live.id in prompt and "Dead" not in prompt


# ------------------------------------------------------------------ skipped without a call
def test_no_llm_call_without_projects_humans_or_the_flag(docs):
    r = Rig(docs, [])
    r.hear(msg(1, "hello"))
    assert r.llm.models.calls == []  # guild without projects
    p = r.project()
    r.hear(msg(2, "x", kind="salid"), msg(3, "   "))
    assert r.llm.models.calls == []  # nothing a human said
    r.project(name="Cancelled", status="cancelled")
    r.repo.soft_delete(GUILD, p.id)
    r.hear(msg(4, "hello"))
    assert r.llm.models.calls == []  # only cancelled / deleted projects left
    off = Rig(docs, [], enabled=lambda gid: False)
    off.project(name="Other")
    off.hear(msg(5, "hello"))
    assert off.llm.models.calls == []


def test_a_failing_model_applies_nothing_and_never_raises(docs):
    r = Rig(docs, [])  # empty script: the call raises
    p = r.project()
    before = r.get(p).model_dump()
    assert r.hear(msg(1, "เสร็จแล้ว")) == []
    r.llm.models.script = [json_resp({"updates": "not a list"})]
    assert r.hear(msg(2, "เสร็จแล้ว")) == [] and r.get(p).model_dump() == before
    assert r.notes == []


# ------------------------------------------------------------------ telling the room
def say_done(r, p, text="เสร็จแล้ว", n=1):
    r.llm.models.script = [json_resp({"updates": [upd(p, "note", f"progress {n}", text)]})]
    return r.hear(msg(n, text))


def test_one_note_per_batch_throttled_for_ten_minutes_per_channel(docs):
    r = Rig(docs, [])
    p = r.project()
    say_done(r, p, n=1)
    assert len(r.notes) == 1 and r.notes[0][:2] == (GUILD, CHAN)
    r.clock.t += NOTE_GAP_S - 1
    assert say_done(r, p, n=2) and len(r.notes) == 1  # applied, but no second note
    assert [u.text for u in r.get(p).updates] == ["progress 1", "progress 2"]
    r.hear(msg(3, "ห้องอื่น"), channel=77)  # another channel has its own clock
    r.llm.models.script = [json_resp({"updates": [upd(p, "note", "other room", "ห้องอื่น")]})]
    r.hear(msg(3, "ห้องอื่น"), channel=77)
    assert len(r.notes) == 2 and r.notes[1][1] == 77
    r.clock.t += 2
    say_done(r, p, n=4)
    assert len(r.notes) == 3


def test_several_changes_make_one_note_with_every_item(docs):
    r = Rig(docs, [])
    a = r.project(name="PM Dashboard")
    b = r.project(name="Salid Web", deadline="2026-11-01")
    q1, q2 = "PM Dashboard เสร็จแล้ว", "Salid Web เลื่อนเป็น 30 พ.ย."
    r.llm.models.script = [
        json_resp(
            {"updates": [upd(a, "status", "completed", q1), upd(b, "deadline", "2026-11-30", q2)]}
        )
    ]
    r.hear(msg(1, q1), msg(2, q2))
    assert len(r.notes) == 1
    text = r.notes[0][2]
    assert text.startswith("📌 บันทึกแล้ว:") and "PM Dashboard → เสร็จสิ้น" in text
    assert "Salid Web กำหนดส่งเป็น 30 พ.ย." in text and "บอกสลิดได้" in text


def test_note_language_follows_the_channel(docs):
    r = Rig(docs, [])
    r.l.lang_of = lambda g, c: "en"
    p = r.project(deadline="2026-11-01")
    q = "push the deadline to 30 Nov"
    r.llm.models.script = [json_resp({"updates": [upd(p, "deadline", "2026-11-30", q)]})]
    r.hear(msg(1, q))
    assert (
        r.notes[0][2]
        == "📌 Noted: PM Dashboard deadline set to 30 Nov (if that's wrong, tell Salid)"
    )
    r.l.lang_of = lambda g, c: "auto"  # auto follows the words
    r.clock.t += NOTE_GAP_S + 1
    r.llm.models.script = [json_resp({"updates": [upd(p, "status", "on_hold", "พักไว้ก่อน")]})]
    r.hear(msg(2, "พักไว้ก่อน"))
    assert "พักไว้" in r.notes[1][2]


def test_no_note_in_dm_scope_or_when_notes_are_off_but_the_change_is_kept(docs):
    r = Rig(docs, [])
    p = r.project()
    r.llm.models.script = [json_resp({"updates": [upd(p, "status", "on_hold", "พักไว้ก่อน")]})]
    r.hear(msg(1, "พักไว้ก่อน", channel=-26), channel=-26)
    assert r.get(p).status == "on_hold" and r.notes == []
    off = Rig(docs, [], notes_on=lambda gid: False)
    q = off.project(name="Quiet")
    off.llm.models.script = [json_resp({"updates": [upd(q, "status", "on_hold", "พักไว้ก่อน")]})]
    off.hear(msg(1, "พักไว้ก่อน"))
    assert off.get(q).status == "on_hold" and off.notes == []
    # a DM note never stores a channel or a link
    r.llm.models.script = [json_resp({"updates": [upd(p, "note", "paused", "พักไว้ก่อน")]})]
    r.hear(msg(2, "พักไว้ก่อน", channel=-26), channel=-26)
    u = r.get(p).updates[-1]
    assert u.channel_id == 0 and u.jump_url == ""


def test_a_failing_note_callback_never_undoes_the_change(docs):
    r = Rig(docs, [])
    r.l.note = lambda *a: 1 / 0
    p = r.project()
    r.llm.models.script = [json_resp({"updates": [upd(p, "status", "on_hold", "พักไว้ก่อน")]})]
    assert r.hear(msg(1, "พักไว้ก่อน")) and r.get(p).status == "on_hold"


# ------------------------------------------------------------------ project_show, strings, flag
def test_project_show_lists_the_latest_updates_newest_first(docs):
    r = Rig(docs, [])
    p = r.project()
    p.updates = [P.Update(at=f"2026-10-0{i} 09:00", text=f"step {i}") for i in range(1, 8)]
    md = P.render_summary(p, [], "en", at(12).date())
    tail = md.split("Latest updates:")[1].strip().split("\n")
    assert tail == [f"- 2026-10-0{i} step {i}" for i in (7, 6, 5, 4, 3)]
    assert "ความเคลื่อนไหวล่าสุด" in P.render_summary(p, [], "th", at(12).date())
    p.updates = []
    assert "Latest updates" not in P.render_summary(p, [], "en", at(12).date())


def test_every_listen_string_has_both_languages_and_matching_placeholders():
    keys = [k for k in CATALOG if k.startswith("pl_")]
    assert len(keys) >= 15
    ph = lambda s: set(re.findall(r"\{(\w+)\}", s))  # noqa: E731
    for k in keys:
        assert CATALOG[k]["th"] and CATALOG[k]["en"], k
        assert ph(CATALOG[k]["th"]) == ph(CATALOG[k]["en"]), k


def test_flag_exists_defaults_on_and_the_prompt_mentions_correction():
    from salid.guild_settings import FEATURES, Features
    from salid.qa import stable_prompt

    assert "project_notes" in FEATURES and Features().project_notes is True
    prompt = stable_prompt("th", False, projects=True)
    assert "correction" in prompt
    assert "list_tasks(person=X)" in stable_prompt("th", False, coo=True)


# ------------------------------------------------------------------ the bot glue
def test_both_hooks_run_and_one_failing_never_blocks_the_other(bot, world, monkeypatch):
    calls = []
    monkeypatch.setattr(
        bot.task_tracker, "on_summary", lambda *a: (calls.append("tasks"), 1 / 0)[1]
    )
    monkeypatch.setattr(bot.project_listener, "on_summary", lambda *a: calls.append("projects"))
    assert bot.memory.scheduler.on_summary == bot._on_summary_all
    bot._on_summary_all(world.g.id, PUBLIC, DAY, {}, [])
    assert calls == ["tasks", "projects"]
    monkeypatch.setattr(bot.project_listener, "on_summary", lambda *a: 1 / 0)
    bot._on_summary_all(world.g.id, PUBLIC, DAY, {}, [])  # no exception


async def test_the_note_is_posted_on_the_bot_loop_without_mentions(bot, world, monkeypatch):
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(bot, "loop", loop, raising=False)
    sent = []

    async def send(text, **kw):
        sent.append((text, kw))

    world.pub.send = send
    world.g.get_channel = lambda cid: world.pub if cid == PUBLIC else None
    await asyncio.to_thread(bot._project_note, world.g.id, PUBLIC, "📌 @everyone x")
    await asyncio.sleep(0.05)
    assert len(sent) == 1 and sent[0][0] == "📌 @everyone x"
    assert sent[0][1]["allowed_mentions"].everyone is False
    # a member's private channel and an unknown channel get nothing
    bot.private_chats.put(world.g.id, STAFF, PUBLIC)
    await asyncio.to_thread(bot._project_note, world.g.id, PUBLIC, "secret")
    await asyncio.to_thread(bot._project_note, world.g.id, 12345, "nowhere")
    await asyncio.sleep(0.05)
    assert len(sent) == 1


def test_the_listener_follows_the_projects_and_memory_flags(bot, world):
    assert bot.project_listener.enabled(world.g.id) is True
    set_features(bot, projects=False)
    assert bot.project_listener.enabled(world.g.id) is False
    set_features(bot, projects=True, memory=False)
    assert bot.project_listener.enabled(world.g.id) is False
    set_features(bot, memory=True, project_notes=False)
    assert bot.project_listener.notes_on(world.g.id) is False
    assert bot.project_listener.enabled(world.g.id) is True


def test_the_listener_runs_after_a_real_summary_update(docs):
    """End to end through the memory scheduler: summary -> action items -> project updates."""
    from salid.memory import MemoryScheduler
    from salid.memory_store import SQLiteMemoryStore

    s = FixedClockSettings()
    r = Rig(docs, [])
    p = r.project()
    q = "PM Dashboard เสร็จแล้ว"
    store = SQLiteMemoryStore(":memory:")
    r.llm.models.script = [
        json_resp({"overview": "o", "action_items": []}),
        json_resp({"updates": [upd(p, "status", "completed", q)]}),
    ]
    sched = MemoryScheduler(store, s, r.llm, now=lambda: at(10, 40), on_summary=r.l.on_summary)
    for i in range(30):
        store.add_message(
            msg(i + 1, q if i == 29 else f"m{i}").model_copy(
                update={"created_at": at(10).astimezone(UTC)}
            )
        )
        sched.note(GUILD, CHAN, DAY)
    assert sched.tick([GUILD]) == 1
    assert r.get(p).status == "completed"
