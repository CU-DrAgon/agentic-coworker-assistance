# ruff: noqa: F811, F401
"""Find a time everyone is free: the search, busy sources, availability rules, the tools bound to
the speaker, privacy (busy/free only) and the bot wiring. Offline."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from test_bot_multitenant import STAFF, Msg, bot, member, world  # noqa: F401  (fixtures)

from salid.coo_store import SQLiteDocStore
from salid.freetime import (
    AvailabilityRepo,
    AvailabilitySource,
    Busy,
    CallbackSource,
    FreetimeContext,
    GoogleCalendarSource,
    LeaveSource,
    MeetingsSource,
    Person,
    find_slots,
    freetime_tools,
    parse_minutes,
    parse_weekdays,
    parse_within,
)
from salid.qa import system_prompt
from salid.secretary_store import SecretaryStore, SqliteDocs

BKK = ZoneInfo("Asia/Bangkok")
NOW = datetime(2026, 10, 19, 8, 0, tzinfo=BKK)  # a Monday
ME, A, B = Person(1, "Me"), Person(2, "Aom"), Person(3, "Bank")


def at(day, h, m=0):
    return datetime(2026, 10, day, h, m, tzinfo=BKK)


class Fixed:
    def __init__(self, busy):
        self._busy = busy

    def busy(self, guild_id, user_ids, start, end):
        return {u: list(b) for u, b in self._busy.items() if u in user_ids}


def ctx(sources=(), repo=None, people=(A, B), now=NOW):
    repo = repo or AvailabilityRepo(SQLiteDocStore(":memory:"))
    by_name = {p.name.casefold(): p for p in people}
    return FreetimeContext(
        guild_id=7, asker=ME, zone=BKK, tz_name="Asia/Bangkok", lang="en", repo=repo,
        sources=list(sources), now=lambda: now,
        match_people=lambda n: [p for k, p in by_name.items() if k.startswith(n.casefold())],
        person_by_id=lambda i: next((p for p in people if p.id == i), None),
    )  # fmt: skip


def tools(c):
    return {f.__name__: f for f in freetime_tools(c)}


# ------------------------------------------------------------------ the search
def test_three_all_free_slots_skip_lunch_and_busy_time():
    busy = {2: [Busy(at(19, 9), at(19, 11))], 3: [Busy(at(19, 15), at(19, 18))]}
    slots = find_slots(
        people=[ME, A, B], busy=busy, work_days={}, start=NOW, end=at(20, 0), now=NOW,
        duration_min=60,
    )  # fmt: skip
    assert len(slots) == 3
    assert all(not s.busy for s in slots)
    for s in slots:
        assert (s.start >= at(19, 11) and s.end <= at(19, 12)) or (
            s.start >= at(19, 13) and s.end <= at(19, 15)
        )
        assert not (
            s.start.hour * 60 + s.start.minute < 780 and s.end.hour * 60 + s.end.minute > 720
        )
    starts = [s.start for s in slots]
    assert starts == sorted(starts)
    assert all(a.end <= b.start for a, b in zip(slots, slots[1:], strict=False))


def test_lunch_allowed_only_on_request():
    busy = {2: [Busy(at(19, 9), at(19, 12)), Busy(at(19, 13), at(19, 18))]}
    none = find_slots(
        people=[ME, A], busy=busy, work_days={}, start=NOW, end=at(20, 0), now=NOW, duration_min=60
    )
    assert none[0].busy == [2]  # nothing common without lunch: best effort shows who is busy
    ok = find_slots(
        people=[ME, A], busy=busy, work_days={}, start=NOW, end=at(20, 0), now=NOW,
        duration_min=60, allow_lunch=True,
    )  # fmt: skip
    assert ok[0].start == at(19, 12) and not ok[0].busy


def test_weekends_skipped_unless_everyone_works_them():
    sat = datetime(2026, 10, 24, 0, 0, tzinfo=BKK)
    args = dict(people=[ME, A], busy={}, start=sat, end=sat + timedelta(days=2), now=NOW, duration_min=60)  # fmt: skip
    assert find_slots(work_days={}, **args) == []
    both = {1: {5}, 2: {5}}
    assert find_slots(work_days=both, **args)[0].start.date() == sat.date()


def test_picks_are_spread_over_days_and_never_in_the_past():
    slots = find_slots(
        people=[ME], busy={}, work_days={}, start=NOW, end=at(23, 0), now=at(19, 13, 10),
        duration_min=60,
    )  # fmt: skip
    assert len({s.start.date() for s in slots}) == 3
    assert all(s.start > at(19, 13, 10) for s in slots)


def test_partial_slots_report_who_is_busy_when_no_common_time():
    busy = {2: [Busy(at(19, 0), at(20, 0))], 3: [Busy(at(19, 0), at(20, 0))]}
    slots = find_slots(
        people=[ME, A, B], busy=busy, work_days={}, start=NOW, end=at(20, 0), now=NOW,
        duration_min=60,
    )  # fmt: skip
    assert slots and slots[0].free == [1] and slots[0].busy == [2, 3]


def test_parse_within_variants():
    assert parse_within("tomorrow", NOW)[0] == at(20, 0)
    a, b = parse_within("this week", NOW)
    assert a == NOW and b == at(26, 0)
    a, b = parse_within("อาทิตย์หน้า", NOW)
    assert a == at(26, 0) and b == at(26, 0) + timedelta(days=7)
    assert parse_within("", NOW)[1] == at(26, 0)
    a, b = parse_within("2026-10-21..2026-10-22", NOW)
    assert a == at(21, 0) and b == at(23, 0)
    with pytest.raises(ValueError):
        parse_within("someday", NOW)


def test_parsers():
    assert (
        parse_minutes("10") == 600
        and parse_minutes("17:30") == 1050
        and parse_minutes("9.30") == 570
    )
    assert parse_minutes("25") is None and parse_minutes("24:00", allow_24=True) == 1440
    assert parse_weekdays(["พุธ"]) == [2] and parse_weekdays("Mon") == [0]
    assert parse_weekdays(["weekdays"]) == [0, 1, 2, 3, 4] and parse_weekdays(["xx"]) is None


# ------------------------------------------------------------------ availability + sources
def test_set_availability_rules_become_busy_blocks():
    repo = AvailabilityRepo(SQLiteDocStore(":memory:"))
    t = tools(ctx([AvailabilitySource(repo, BKK)], repo))
    assert (
        t["set_availability"]("busy_weekly", weekdays=["พุธ"], part="afternoon")["status"] == "saved"
    )
    assert (
        t["set_availability"]("work_hours", start_time="10:00", end_time="19:00")["status"]
        == "saved"
    )
    assert (
        t["set_availability"]("busy_once", date="2026-10-20", part="morning", note="ลา")["status"]
        == "saved"
    )
    src = AvailabilitySource(repo, BKK)
    b = src.busy(7, [1], NOW, at(26, 0))[1]
    spans = {(x.start, x.end) for x in b}
    assert (at(21, 13), at(21, 18)) in spans  # Wednesday afternoon
    assert (at(19, 0), at(19, 10)) in spans  # before the 10:00 start
    assert (at(20, 9), at(20, 13)) in spans  # half-day leave
    assert any(x.start == at(24, 0) for x in b)  # Saturday is a non-work day
    shown = t["my_availability"]()["availability"]
    assert shown["work_hours"].startswith("10:00-19:00")
    assert shown["busy_weekly"][0]["days"] == "Wed" and shown["busy_once"][0]["note"] == "ลา"


def test_find_free_time_uses_rules_and_returns_labels():
    repo = AvailabilityRepo(SQLiteDocStore(":memory:"))
    c = ctx([AvailabilitySource(repo, BKK)], repo)
    t = tools(c)
    t["set_availability"]("busy_once", date="2026-10-19", part="day")
    out = t["find_free_time"](people=["aom", "bank"], duration_minutes=60, within="this week")
    assert out["status"] == "ok" and len(out["slots"]) == 3
    assert out["attendees"] == ["Me", "Aom", "Bank"]
    assert all(s["start"] >= "2026-10-20" and s["everyone_free"] for s in out["slots"])
    assert out["slots"][0]["label"].startswith("Tue 2026-10-20")


def test_clear_availability_one_or_all():
    repo = AvailabilityRepo(SQLiteDocStore(":memory:"))
    t = tools(ctx([], repo))
    t["set_availability"]("busy_weekly", weekdays=["mon"], start_time="9", end_time="10")
    rid = t["my_availability"]()["availability"]["busy_weekly"][0]["id"]
    assert t["my_availability"]("clear", "nope")["error"]
    assert t["my_availability"]("clear", rid)["availability"]["busy_weekly"] == []
    t["set_availability"]("work_hours", start_time="10", end_time="19")
    assert t["my_availability"]("clear", "work_hours")["availability"]["work_hours"].startswith(
        "default"
    )
    t["set_availability"]("busy_weekly", weekdays=["mon"], start_time="9", end_time="10")
    t["my_availability"]("clear")
    assert repo.get(7, 1) == {"work": None, "weekly": [], "once": []}


def test_set_availability_validation_and_isolation():
    repo = AvailabilityRepo(SQLiteDocStore(":memory:"))
    t = tools(ctx([], repo))
    assert "error" in t["set_availability"]("busy_once", date="2026-10-01")  # past
    assert "error" in t["set_availability"]("busy_once", date="tomorrow")
    assert "error" in t["set_availability"]("busy_weekly", weekdays=["xx"])
    assert "error" in t["set_availability"]("work_hours", start_time="19:00", end_time="10:00")
    assert "error" in t["set_availability"]("nonsense")
    import inspect

    for f in t.values():  # nobody else's availability can be targeted
        assert not set(inspect.signature(f).parameters) & {
            "user",
            "user_id",
            "member",
            "guild_id",
            "who",
        }
    t["set_availability"]("busy_weekly", weekdays=["mon"], part="morning")
    assert repo.get(7, 2)["weekly"] == [] and repo.get(8, 1)["weekly"] == []  # per guild and user


def test_meetings_source_uses_salid_meetings_and_ignores_cancelled():
    store = SecretaryStore(SqliteDocs())
    base = {"guild_id": 7, "organizer_id": 2, "attendee_ids": [1], "title": "ลับสุดยอด"}
    store.put_meeting({**base, "id": "m1", "status": "scheduled",
                       "start_utc": at(19, 10).astimezone(UTC).isoformat(),
                       "end_utc": at(19, 11).astimezone(UTC).isoformat()})  # fmt: skip
    store.put_meeting({**base, "id": "m2", "status": "cancelled",
                       "start_utc": at(19, 14).astimezone(UTC).isoformat(),
                       "end_utc": at(19, 15).astimezone(UTC).isoformat()})  # fmt: skip
    got = MeetingsSource(store).busy(7, [1, 2, 3], NOW, at(20, 0))
    assert set(got) == {1, 2} and len(got[1]) == 1
    assert got[1][0].start == at(19, 10) and got[1][0].kind == "meeting"
    assert MeetingsSource(store).busy(8, [1], NOW, at(20, 0)) == {}  # other guild


def test_noop_defaults_and_pluggable_leave_and_calendar():
    assert LeaveSource(None).busy(7, [1], NOW, at(20, 0)) == {}
    assert GoogleCalendarSource().busy(7, [1], NOW, at(20, 0)) == {}
    calls = []

    def approved_leave(guild_id, user_id, start, end):
        calls.append((guild_id, user_id))
        return [(at(19, 0), at(20, 0))] if user_id == 2 else []

    c = ctx([LeaveSource(approved_leave)])
    out = tools(c)["find_free_time"](people=["aom"], within="today")
    assert out["status"] == "ok" and out["slots"][0]["busy"] == ["Aom"]  # leave -> best effort
    assert (7, 2) in calls


def test_a_failing_source_does_not_break_the_answer():
    class Boom:
        def busy(self, *a):
            raise RuntimeError("down")

    out = tools(ctx([Boom(), Fixed({2: [Busy(at(19, 9), at(19, 12))]})]))["find_free_time"](
        people=["aom"], within="today", duration_minutes=60
    )
    assert out["status"] == "ok" and out["slots"][0]["start"].startswith("2026-10-19T13:00")


def test_unknown_and_ambiguous_people_need_info_and_mentions_resolve():
    c = ctx(people=(A, B, Person(4, "Aomsin")))
    t = tools(c)
    out = t["find_free_time"](people=["Zed", "Aom"])
    assert out["status"] == "needs_info" and out["unresolved"] == ["Zed"]
    assert out["ambiguous"] == {"Aom": ["Aom", "Aomsin"]}
    ok = t["find_free_time"](people=["<@3>", "me", "ผม"])
    assert ok["attendees"] == ["Me", "Bank"]


def test_no_slot_and_bad_arguments():
    t = tools(ctx())
    assert t["find_free_time"](within="2026-10-17")["status"] == "no_slot"  # a Saturday in the past
    assert "error" in t["find_free_time"](within="blah")
    assert "error" in t["find_free_time"](earliest_hour="20", latest_hour="9")
    out = t["find_free_time"](
        within="today", earliest_hour="13", latest_hour="15", duration_minutes=60
    )
    assert out["status"] == "ok" and out["slots"][0]["start"][11:16] == "13:00"
    assert all(s["end"][11:16] <= "15:00" for s in out["slots"] if s["start"][:10] == "2026-10-19")


def test_never_leaks_titles_only_busy_and_free():
    store = SecretaryStore(SqliteDocs())
    store.put_meeting({"id": "m1", "guild_id": 7, "organizer_id": 2, "attendee_ids": [2],
                       "title": "SECRET-LAYOFF-PLAN", "description": "SECRET-DESC", "status": "scheduled",
                       "start_utc": at(19, 9).astimezone(UTC).isoformat(),
                       "end_utc": at(19, 18).astimezone(UTC).isoformat()})  # fmt: skip
    out = tools(ctx([MeetingsSource(store)]))["find_free_time"](people=["aom"], within="today")
    assert "SECRET" not in json.dumps(out, ensure_ascii=False)
    assert out["slots"][0]["busy"] == ["Aom"]


# ------------------------------------------------------------------ prompts and bot wiring
def test_prompts_mention_find_free_time():
    from salid.qa import SECRETARY_PROMPT

    assert "find_free_time" in SECRETARY_PROMPT
    p = system_prompt(SimpleNamespace(tz="Asia/Bangkok"), scoped=False, now=NOW, freetime=True)
    assert "find_free_time" in p and "set_availability" in p and "เอาอันแรก" in p
    plain = system_prompt(SimpleNamespace(tz="Asia/Bangkok"), scoped=False, now=NOW)
    assert "set_availability" not in plain


async def test_agent_kwargs_register_the_tools_for_chat_and_voice(bot, world):
    me = member(world.a, STAFF)
    other = member(world.a, STAFF + 1)
    other.display_name = "Aom"
    msg = Msg(world.a, world.a.text_channels[0], me, "hi")
    kw = bot._agent_kwargs(
        msg, world.a, me, False, setting="th", lang="th", question="q", memory=None, profile=None
    )
    names = [f.__name__ for f in kw["extra_tools"]]
    assert {"find_free_time", "set_availability", "my_availability"} <= set(names)


async def test_flag_off_registers_nothing(bot, world):
    me = member(world.a, STAFF)
    bot.team.enabled = lambda gid, f: f != "freetime"
    assert bot._freetime_tools(world.a, me, "th") == []


async def test_discord_events_busy_uses_interest_and_creator_only_time(bot, world):
    me = member(world.a, STAFF)
    aom = member(world.a, STAFF + 1)
    aom.display_name = "Aom"
    bob = member(world.a, STAFF + 2)

    class Ev:
        def __init__(self, name, start, end, creator, interested, status="scheduled"):
            self.name, self.start_time, self.end_time = name, start, end
            self.creator_id, self._int = creator, interested
            self.status = SimpleNamespace(name=status)

        async def users(self, limit=None):
            for u in self._int:
                yield SimpleNamespace(id=u)

    world.a.scheduled_events = [
        Ev("SECRET party", at(20, 9), at(20, 12), bob.id, [aom.id]),
        Ev("old", at(20, 9), at(20, 12), aom.id, [], status="completed"),
        Ev("elsewhere", at(21, 9), at(21, 12), bob.id, [bob.id]),
    ]
    got = await bot._event_busy(world.a, {me.id, aom.id}, at(20, 0), at(21, 0))
    assert set(got) == {aom.id} and len(got[aom.id]) == 1
    assert got[aom.id][0].start == at(20, 9) and "SECRET" not in repr(got)
    tool = {f.__name__: f for f in bot._freetime_tools(world.a, me, "en")}["find_free_time"]
    out = await asyncio.to_thread(tool, ["Aom"], 60, "2026-10-20..2026-10-20")
    assert out["status"] in ("ok", "no_slot")
    assert "SECRET" not in json.dumps(out)
