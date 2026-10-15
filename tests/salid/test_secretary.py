"""Secretary features: scheduling (confirmation, event payload, reminders, no-show nudges,
auto-join, attendance notes, events made in Discord), PDF documents and email. All offline: Discord
objects are small fakes, the clock is a variable, email goes to LogOnlySender or a mocked HTTP."""

import asyncio
import base64
import json
import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import discord
import httpx
import pytest
from conftest import ALICE, BOB, GUILD, OTHER_GUILD, mk

from salid import pdf_report
from salid.bot import SecretaryBot
from salid.email_out import GmailSender, LogOnlySender, OutgoingEmail, clean_addresses
from salid.guild_settings import MemoryGuildStore
from salid.i18n import t
from salid.qa import LedgerTools, build_agent, system_prompt
from salid.scheduling import (
    CalendarError,
    DiscordCalendar,
    MeetingService,
    classify_link,
    join_link,
    parse_range,
    pick_voice_channel,
    resolve_attendees,
)
from salid.secretary import SecretaryContext, SecretaryState, SecretaryTools
from salid.secretary_store import SecretaryStore, SqliteDocs

NOW = datetime(2026, 10, 14, 5, 0, tzinfo=UTC)  # 12:00 in Bangkok
START = "2026-10-19T14:00"  # Monday, 07:00 UTC


# ------------------------------------------------------------------ fakes
class Chan:
    def __init__(self, id, name, members=()):
        self.id, self.name, self.members, self.sent = id, name, list(members), []

    async def send(self, text, **kw):
        self.sent.append(text)
        return SimpleNamespace(
            jump_url=f"https://discord.com/channels/{GUILD}/{self.id}/{len(self.sent)}"
        )


class Member:
    def __init__(self, id, name, fail_dm=False, bot=False):
        self.id, self.display_name, self.name, self.bot = id, name, name.lower(), bot
        self.global_name, self.sent, self.fail_dm = None, [], fail_dm

    async def send(self, text):
        if self.fail_dm:
            raise discord.Forbidden(SimpleNamespace(status=403, reason="x"), "closed")
        self.sent.append(text)


class Event:
    def __init__(self, id, guild=None, **kw):
        self.id, self.guild, self.kw, self.edits, self.deleted = id, guild, kw, [], False
        self.url = f"https://discord.com/events/{GUILD}/{id}"
        self.subs: list = []
        self.name = kw.get("name", "x")
        self.start_time, self.end_time = kw.get("start_time"), kw.get("end_time")
        self.channel, self.location = kw.get("channel"), kw.get("location")
        self.description, self.creator = kw.get("description", ""), None
        self.status = SimpleNamespace(name="scheduled")

    async def edit(self, **kw):
        self.edits.append(kw)

    async def delete(self):
        self.deleted = True

    async def users(self):
        for u in self.subs:
            yield u


class Guild:
    def __init__(self, id=GUILD, manage_events=True):
        self.id, self.name, self.members = id, "ร้านกาแฟ", {}
        self.me = SimpleNamespace(guild_permissions=SimpleNamespace(manage_events=manage_events))
        self.voice_channels, self.chans, self.created, self.events = [], {}, [], {}
        self.system_channel = None

    def add(self, *members):
        for m in members:
            self.members[m.id] = m
        return members

    def add_channel(self, ch, voice=False):
        self.chans[ch.id] = ch
        if voice:
            self.voice_channels.append(ch)
        return ch

    def get_member(self, uid):
        return self.members.get(uid)

    def get_channel(self, cid):
        return self.chans.get(cid)

    async def create_scheduled_event(self, **kw):
        ev = Event(9000 + len(self.created), self, **kw)
        self.created.append(kw)
        self.events[ev.id] = ev
        return ev

    def get_scheduled_event(self, eid):
        return self.events.get(eid)

    async def fetch_scheduled_event(self, eid):
        raise discord.NotFound(SimpleNamespace(status=404, reason="x"), "gone")


class FakeTeam:
    def __init__(self, **off):
        self.off = off

    def enabled(self, gid, feature):
        return not self.off.get(feature, False)

    def channel_language(self, gid, cid):
        return "th"


class FakeMeetings:
    """Stands in for MeetingManager: records join() and exposes `sessions`."""

    def __init__(self, guild):
        self.guild, self.sessions, self.joins = guild, {}, []

    async def join(self, starter, **kw):
        self.joins.append((starter, kw))
        s = SimpleNamespace(
            meeting_id=kw["meeting_id"],
            voice_channel=starter.voice.channel,
            summary_text="",
            summary_url="",
        )
        self.sessions[self.guild.id] = s
        return "joined"


@pytest.fixture
def bgloop():
    loop = asyncio.new_event_loop()
    th = threading.Thread(target=loop.run_forever, daemon=True)
    th.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    th.join(5)


class World:
    def __init__(self, settings, loop, ledger_store, features=None):
        self.settings = settings
        self.now = NOW
        self.guild = Guild()
        self.ann, self.bob, self.cat = self.guild.add(
            Member(10, "Ann"), Member(11, "Bob"), Member(12, "Cat")
        )
        self.org = self.guild.add(Member(1, "Boss"))[0]
        self.lobby = self.guild.add_channel(Chan(500, "general"))
        self.room = self.guild.add_channel(Chan(600, "ห้องประชุม"), voice=True)
        self.store = SecretaryStore(SqliteDocs())
        self.state = SecretaryState()
        self.meetings = FakeMeetings(self.guild)
        self.team = FakeTeam(**(features or {}))
        self.svc = MeetingService(
            self.store, DiscordCalendar(), settings, self.team, self.meetings,
            lambda gid: self.guild if gid == GUILD else None, clock=lambda: self.now,
        )  # fmt: skip
        self.loop, self.ledger_store = loop, ledger_store

    def ctx(self, uid=1, name="Boss", is_exec=True, can_email=True, **kw) -> SecretaryContext:
        loop = self.loop
        base = dict(
            settings=self.settings, guild_id=GUILD, guild_name="ร้านกาแฟ", author_id=uid,
            author_name=name, is_exec=is_exec, can_email=can_email, lang="th",
            text_channel_id=500, store=self.store, state=self.state,
            run=lambda coro: asyncio.run_coroutine_threadsafe(coro, loop).result(10),
            service=self.svc, sender=LogOnlySender(), now=lambda: self.now,
            ledger=LedgerTools(self.ledger_store, GUILD, None if is_exec else uid),
        )  # fmt: skip
        c = SecretaryContext(**{**base, **kw})
        return c

    def tools(self, **kw) -> SecretaryTools:
        return SecretaryTools(self.ctx(**kw))


@pytest.fixture
def w(settings, bgloop, store):
    return World(settings, bgloop, store)


def propose_then_confirm(w, call, **ctx):
    """First request proposes, a LATER request (new request id) confirms."""
    first = call(w.tools(**ctx))
    assert first["status"] == "needs_confirmation", first
    return first, call(w.tools(**ctx))


# ------------------------------------------------------------------ link types, attendees, ranges
@pytest.mark.parametrize(
    ("url", "kind"),
    [
        ("https://meet.google.com/abc-defg-hij", "meet"),
        ("meet.google.com/abc-defg-hij", "meet"),
        ("https://us02web.zoom.us/j/123?pwd=x", "zoom"),
        ("https://teams.microsoft.com/l/meetup-join/abc", "teams"),
        ("https://example.com/room/1", "url"),
        ("https://notzoom.us.evil.com/x", "url"),
        ("javascript:alert(1)", None),
        ("ftp://x.y/z", None),
        ("two words", None),
        ("", None),
        (None, None),
    ],
)
def test_classify_link(url, kind):
    assert classify_link(url) == kind


def test_resolve_attendees_by_mention_name_and_ambiguity(w):
    w.guild.add(Member(13, "Annabel"), Member(99, "Robot", bot=True))
    got, bad = resolve_attendees(w.guild, ["<@11>", "cat", "ann", "annab", "ghost", "Robot"])
    # "ann" is exact for Ann (not ambiguous with Annabel); "annab" is a unique prefix
    assert [m.id for m in got] == [11, 12, 10, 13] and bad == ["ghost", "Robot"]
    _, bad2 = resolve_attendees(w.guild, ["an"])  # prefix of two people: ambiguous
    assert bad2 == ["an"]


def test_voice_channel_choice(w):
    assert pick_voice_channel(w.guild, "ห้องประชุม") is w.room
    assert pick_voice_channel(w.guild, None) is w.room  # the only voice channel
    w.guild.add_channel(Chan(601, "music"), voice=True)
    assert pick_voice_channel(w.guild, None) is w.room  # the only meeting-like one
    w.guild.add_channel(Chan(602, "meeting 2"), voice=True)
    assert pick_voice_channel(w.guild, None) is None  # no obvious choice
    assert pick_voice_channel(w.guild, "nope") is None


def test_parse_range():
    now = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
    a, b = parse_range("today", now)
    assert (a.day, b.day) == (14, 15)
    a, b = parse_range("2026-10-19..2026-10-20", now)
    assert (a.day, b.day) == (19, 21)
    with pytest.raises(ValueError):
        parse_range("whenever", now)


# ------------------------------------------------------------------ schedule: confirmation + payload
ARGS = dict(title="Weekly sync", start=START, duration_minutes=45, attendees=["ann", "<@11>"])


def test_schedule_needs_confirmation_and_never_creates_in_the_same_request(w):
    tools = w.tools()
    r = tools.schedule_meeting(**ARGS)
    assert r["status"] == "needs_confirmation" and w.guild.created == []
    p = r["proposal"]
    assert p["when"].startswith("จันทร์ 2026-10-19 14:00 (Asia/Bangkok)")  # weekday + local tz
    assert {a["mention"] for a in p["attendees"]} == {"<@1>", "<@10>", "<@11>"}
    assert "voice channel #ห้องประชุม" in p["where"]
    assert tools.schedule_meeting(**ARGS)["status"] == "needs_confirmation"  # same request again
    assert w.guild.created == []


def test_schedule_creates_a_voice_event_after_a_later_confirmation(w):
    _, r = propose_then_confirm(w, lambda tl: tl.schedule_meeting(**ARGS))
    assert r["status"] == "created" and r["dm_failed"] == []
    (kw,) = w.guild.created
    assert kw["start_time"] == datetime(2026, 10, 19, 7, 0, tzinfo=UTC)  # 14:00 Bangkok
    assert kw["start_time"].utcoffset() == timedelta(hours=7) or kw["start_time"].tzinfo
    assert kw["end_time"] - kw["start_time"] == timedelta(minutes=45)
    assert kw["channel"] is w.room and "entity_type" not in kw and kw["name"] == "Weekly sync"
    (m,) = w.store.list_meetings(GUILD)
    assert m["kind"] == "voice" and m["text_channel_id"] == 600
    assert sorted(m["attendee_ids"]) == [1, 10, 11] and m["status"] == "scheduled"
    assert "Weekly sync" in w.ann.sent[0] and join_link(m) in w.ann.sent[0]  # invite DM
    assert w.org.sent == []  # the organiser is not DM'd about their own meeting


def test_changed_arguments_need_a_fresh_confirmation(w):
    w.tools().schedule_meeting(**ARGS)
    r = w.tools().schedule_meeting(**{**ARGS, "duration_minutes": 90})
    assert r["status"] == "needs_confirmation" and w.guild.created == []


def test_schedule_with_a_meet_link_is_an_external_event(w):
    a = {**ARGS, "link": "https://meet.google.com/abc-defg-hij", "attendees": ["cat"]}
    _, r = propose_then_confirm(w, lambda tl: tl.schedule_meeting(**a))
    (kw,) = w.guild.created
    assert kw["location"] == "https://meet.google.com/abc-defg-hij" and "channel" not in kw
    assert kw["entity_type"] == discord.EntityType.external
    (m,) = w.store.list_meetings(GUILD)
    assert m["kind"] == "external" and m["link_type"] == "meet" and m["voice_channel_id"] is None
    assert join_link(m) == "https://meet.google.com/abc-defg-hij"


def test_schedule_without_an_obvious_voice_channel_is_external_discord(w):
    w.guild.voice_channels.clear()
    propose_then_confirm(w, lambda tl: tl.schedule_meeting(**ARGS))
    assert w.guild.created[0]["location"] == "Discord"


def test_schedule_rejects_bad_input_and_unknown_attendees(w):
    tl = w.tools()
    assert "ISO" in tl.schedule_meeting("x", "tomorrow at 2")["error"]
    assert "past" in tl.schedule_meeting("x", "2026-10-01T10:00")["error"]
    assert "link" in tl.schedule_meeting("x", START, link="not a url")["error"]
    r = tl.schedule_meeting("x", START, attendees=["Zed"])
    assert r["status"] == "needs_info" and r["unresolved_attendees"] == ["Zed"]
    assert w.guild.created == []


def test_dm_failures_are_reported_not_fatal(w):
    w.bob.fail_dm = True
    _, r = propose_then_confirm(w, lambda tl: tl.schedule_meeting(**ARGS))
    assert r["status"] == "created" and r["dm_failed"] == ["Bob"] and len(w.ann.sent) == 1


def test_missing_manage_events_permission_explains_how_to_grant(w):
    w.guild.me.guild_permissions.manage_events = False
    _, r = propose_then_confirm(w, lambda tl: tl.schedule_meeting(**ARGS))
    assert r["error"] == "missing_permission"
    assert "Manage Events" in r["how_to_grant"] or "จัดการกิจกรรม" in r["how_to_grant"]
    assert w.guild.created == [] and w.store.list_meetings(GUILD) == []


def test_discord_forbidden_is_also_a_missing_permission(w):
    async def boom(**kw):
        raise discord.Forbidden(SimpleNamespace(status=403, reason="x"), "Missing Permissions")

    w.guild.create_scheduled_event = boom
    _, r = propose_then_confirm(w, lambda tl: tl.schedule_meeting(**ARGS))
    assert r["error"] == "missing_permission"


def test_scheduling_can_be_switched_off(w):
    assert w.tools(scheduling=False).schedule_meeting(**ARGS) == {"status": "disabled"}


# ------------------------------------------------------------------ list / cancel / reschedule
def create_meeting(w, **over):
    a = {**ARGS, **over}
    propose_then_confirm(w, lambda tl: tl.schedule_meeting(**a))
    return w.store.list_meetings(GUILD)[-1]


def test_list_meetings(w):
    create_meeting(w)
    assert w.tools().list_meetings("today")["count"] == 0  # it is next Monday
    r = w.tools().list_meetings("week")
    assert r["count"] == 1 and r["meetings"][0]["title"] == "Weekly sync"
    assert w.tools().list_meetings("2026-10-19")["count"] == 1


def test_cancel_needs_confirmation_and_permission(w):
    m = create_meeting(w)
    r = w.tools().cancel_meeting(m["event_id"])
    assert r["status"] == "needs_confirmation" and not w.guild.events[9000].deleted
    stranger = w.tools(uid=12, name="Cat", is_exec=False)
    assert stranger.cancel_meeting(m["event_id"])["error"] == "not_allowed"
    assert w.tools().cancel_meeting(m["event_id"])["status"] == "cancelled"
    assert w.guild.events[9000].deleted
    assert w.store.get_meeting(m["id"])["status"] == "cancelled"
    assert "ยกเลิก" in w.ann.sent[-1]
    assert w.tools().cancel_meeting(m["event_id"])["error"] == "meeting not found"


def test_reschedule_moves_the_event_and_resets_reminders(w):
    m = create_meeting(w)
    w.store.put_meeting({**m, "rem_pre": True})
    new = "2026-10-20T10:30"
    r = w.tools().reschedule_meeting(m["id"], new)
    assert r["status"] == "needs_confirmation" and w.guild.events[9000].edits == []
    r = w.tools().reschedule_meeting(m["id"], new)
    assert r["status"] == "rescheduled"
    (edit,) = w.guild.events[9000].edits
    assert edit["start_time"] == datetime(2026, 10, 20, 3, 30, tzinfo=UTC)
    assert edit["end_time"] - edit["start_time"] == timedelta(minutes=45)
    got = w.store.get_meeting(m["id"])
    assert got["rem_pre"] is False and got["start_utc"].startswith("2026-10-20T03:30")
    assert "2026-10-20 10:30" in w.ann.sent[-1]


# ------------------------------------------------------------------ reminders (fake clock)
def meeting_at(w, start, kind="voice", link="", attendees=(10, 11), **over):
    s = start.astimezone(UTC)
    m = {
        "id": "m1", "guild_id": GUILD, "event_id": "9000", "source": "salid", "title": "Sync",
        "start": s.isoformat(), "start_utc": s.isoformat(timespec="seconds"),
        "end_utc": (s + timedelta(hours=1)).isoformat(timespec="seconds"),
        "duration_minutes": 60, "tz": "Asia/Bangkok", "kind": kind,
        "voice_channel_id": 600 if kind == "voice" else None, "text_channel_id": 600 if kind == "voice" else 500,
        "link": link, "link_type": classify_link(link), "organizer_id": 1, "organizer_name": "Boss",
        "attendee_ids": list(attendees), "attendee_names": [w.guild.members[i].display_name for i in attendees],
        "status": "scheduled", "rem_pre": False, "rem_start": False, "nudged": False, "joined": False,
        "attended": [], "dm_sent": {}, "dm_failed": [], "attendance_notes": {}, "summary": "", "summary_url": "",
    }  # fmt: skip
    m.update(over)
    w.store.put_meeting(m)
    return m


async def test_reminders_fire_ten_minutes_before_and_at_the_start(w):
    start = NOW + timedelta(hours=1)
    meeting_at(w, start)
    w.now = start - timedelta(minutes=11)
    await w.svc.tick()
    assert w.room.sent == [] and w.ann.sent == []
    w.now = start - timedelta(minutes=10)
    await w.svc.tick()
    (post,) = w.room.sent
    link = f"https://discord.com/channels/{GUILD}/600"
    assert "<@10>" in post and "<@11>" in post and "10 นาที" in post and link in post
    assert link in w.ann.sent[0] and link in w.bob.sent[0]  # DM with the one-click link
    await w.svc.tick()
    assert len(w.room.sent) == 1  # not twice
    w.now = start
    await w.svc.tick()
    assert len(w.room.sent) == 2 and "เริ่มแล้ว" in w.room.sent[1] and link in w.ann.sent[1]
    await w.svc.tick()
    assert len(w.room.sent) == 2


async def test_reminders_respect_dm_failures(w):
    start = NOW + timedelta(minutes=5)
    meeting_at(w, start)
    w.ann.fail_dm = True
    w.now = start - timedelta(minutes=5)
    await w.svc.tick()
    assert w.bob.sent and "Ann" in w.store.get_meeting("m1")["dm_failed"]
    assert len(w.room.sent) == 1  # the channel post still went out


async def test_meet_meeting_reminds_with_the_link_and_a_reply_hint(w):
    start = NOW + timedelta(hours=1)
    url = "https://meet.google.com/abc-defg-hij"
    meeting_at(w, start, kind="external", link=url)
    w.now = start
    await w.svc.tick()
    assert url in w.lobby.sent[0] and url in w.ann.sent[0]
    assert "ตอบกลับสลิด" in w.ann.sent[0] and "ตอบกลับสลิด" in w.lobby.sent[0]
    w.now = start + timedelta(minutes=6)
    await w.svc.tick()
    assert len(w.ann.sent) == 1  # no no-show DM for Meet: Salid cannot see who joined


async def test_a_missed_start_reminder_is_not_sent_late(w):
    start = NOW
    meeting_at(w, start)
    w.now = start + timedelta(minutes=40)
    await w.svc.tick()
    assert w.room.sent == [] and w.ann.sent == []


async def test_reminders_follow_the_scheduling_toggle(w):
    w.svc.team = FakeTeam(scheduling=True)
    start = NOW
    meeting_at(w, start)
    w.now = start
    await w.svc.tick()
    assert w.room.sent == []


# ------------------------------------------------------------------ no-show nudge + replies
async def test_no_show_dm_goes_only_to_attendees_missing_from_the_voice_channel(w):
    start = NOW + timedelta(hours=1)
    meeting_at(w, start)
    w.room.members = [w.ann]  # Ann is in; Bob is not
    w.now = start
    await w.svc.tick()
    n_ann, n_bob = len(w.ann.sent), len(w.bob.sent)
    w.now = start + timedelta(minutes=4)
    await w.svc.tick()
    assert (len(w.ann.sent), len(w.bob.sent)) == (n_ann, n_bob)  # not yet
    w.now = start + timedelta(minutes=5)
    await w.svc.tick()
    assert len(w.ann.sent) == n_ann
    nudge = w.bob.sent[-1]
    assert "เริ่มแล้ว" in nudge and "มีปัญหาอะไรหรือเปล่า" in nudge
    assert f"https://discord.com/channels/{GUILD}/600" in nudge
    await w.svc.tick()
    assert len(w.bob.sent) == n_bob + 1  # once


async def test_dm_reply_is_recorded_and_the_organiser_is_told(w):
    start = NOW
    meeting_at(w, start)
    w.now = start + timedelta(minutes=5)
    await w.svc.tick()  # reminder + nudge DMs stamped
    w.room.sent.clear()
    ack = await w.svc.handle_dm_reply(11, "ไปช้า 10 นาที")
    assert ack and "Sync" in ack
    assert w.room.sent == [
        t("sched_absence_note", "th", name="Bob", title="Sync", reason="ไปช้า 10 นาที")
    ]
    assert w.store.get_meeting("m1")["attendance_notes"]["11"]["note"] == "ไปช้า 10 นาที"
    # unrelated DMs are left to the normal agent
    assert await w.svc.handle_dm_reply(11 + 100, "hello") is None
    assert await w.svc.handle_dm_reply(11, "x" * 300) is None
    w.now = start + timedelta(hours=3)
    assert await w.svc.handle_dm_reply(11, "late reply") is None  # too long after our DM


async def test_attendance_text_lists_absence_reasons(w):
    m = meeting_at(
        w, NOW, attended=[10], attendance_notes={"11": {"name": "Bob", "note": "ติดลูกค้า"}}
    )
    text = w.svc.attendance_text(w.guild, m, "th")
    assert "Ann" in text and "Bob (ติดลูกค้า)" in text


# ------------------------------------------------------------------ auto-join and summary link
async def test_auto_join_takes_notes_when_people_are_in_the_channel(w):
    start = NOW
    meeting_at(w, start)
    w.now = start
    await w.svc.tick()
    assert w.meetings.joins == []  # nobody there yet
    w.room.members = [w.ann]
    await w.svc.tick()
    ((starter, kw),) = w.meetings.joins
    assert starter.voice.channel is w.room and kw["announce"] == "meet_announce_scheduled"
    assert kw["empty_grace"] == 120.0 and kw["meeting_id"] == "m1"
    assert "สลิดเข้ามาจดประชุมตามนัดครับ" in t("meet_announce_scheduled", "th")
    got = w.store.get_meeting("m1")
    assert got["joined"] is True and got["attended"] == [10]
    await w.svc.tick()
    assert len(w.meetings.joins) == 1


async def test_auto_join_respects_the_voice_meetings_toggle(w):
    w.svc.team = FakeTeam(voice_meetings=True)
    meeting_at(w, NOW)
    w.room.members = [w.ann]
    w.now = NOW
    await w.svc.tick()
    assert w.meetings.joins == []


async def test_voice_end_links_the_summary_and_posts_attendance(w):
    meeting_at(w, NOW, attended=[10], attendance_notes={"11": {"name": "Bob", "note": "ติดลูกค้า"}})
    session = SimpleNamespace(
        summary_text="ตกลงกันว่า...", summary_url="https://discord.com/channels/1/2/3"
    )
    await w.svc.on_voice_end("m1", session)
    m = w.store.get_meeting("m1")
    assert (
        m["status"] == "done"
        and m["summary_url"].endswith("/2/3")
        and m["summary"].startswith("ตกลง")
    )
    post = w.room.sent[-1]
    assert "Bob (ติดลูกค้า)" in post and "https://discord.com/channels/1/2/3" in post


async def test_non_voice_meeting_posts_attendance_notes_when_done(w):
    m = meeting_at(w, NOW, kind="external", attendance_notes={"11": {"name": "Bob", "note": "ลา"}})
    w.now = NOW + timedelta(hours=3)
    await w.svc.tick()
    assert w.store.get_meeting(m["id"])["status"] == "done" and "Bob (ลา)" in w.lobby.sent[-1]


# ------------------------------------------------------------------ events made in Discord
def discord_event(w, **over):
    s = NOW + timedelta(hours=2)
    ev = Event(
        7001,
        w.guild,
        name="Standup",
        start_time=s,
        end_time=s + timedelta(minutes=30),
        channel=w.room,
    )
    for k, v in over.items():
        setattr(ev, k, v)
    return ev


async def test_events_created_in_discord_are_tracked_and_reminded(w):
    ev = discord_event(w)
    ev.subs = [w.cat, w.bob]
    m = await w.svc.track_discord_event(ev)
    assert m["source"] == "discord" and m["kind"] == "voice" and m["voice_channel_id"] == 600
    assert (await w.svc.track_discord_event(ev))["id"] == m["id"]  # idempotent
    w.guild.events[7001] = ev
    start = ev.start_time
    w.now = start - timedelta(minutes=10)
    await w.svc.tick()
    assert sorted(w.store.get_meeting(m["id"])["attendee_ids"]) == [11, 12]  # interested members
    assert "Standup" in w.cat.sent[0] and "<@12>" in w.room.sent[0]
    w.now = start + timedelta(minutes=5)
    w.room.members = [w.cat]
    await w.svc.tick()
    assert "มีปัญหาอะไรหรือเปล่า" in w.bob.sent[-1] and len(w.cat.sent) == 2  # only Bob nudged


async def test_discord_event_with_a_zoom_location_is_external(w):
    ev = discord_event(w, channel=None, location="https://zoom.us/j/123")
    m = await w.svc.track_discord_event(ev)
    assert (
        m["kind"] == "external"
        and m["link_type"] == "zoom"
        and join_link(m) == "https://zoom.us/j/123"
    )


async def test_discord_event_updates_and_deletes_follow_through(w):
    ev = discord_event(w)
    m = await w.svc.track_discord_event(ev)
    ev.start_time += timedelta(hours=1)
    ev.name = "Standup v2"
    await w.svc.update_discord_event(ev)
    got = w.store.get_meeting(m["id"])
    assert got["title"] == "Standup v2" and got["start_utc"].startswith("2026-10-14T08:00")
    await w.svc.delete_discord_event(ev)
    assert w.store.get_meeting(m["id"])["status"] == "cancelled"


async def test_bot_event_handlers_track_discord_events(w, monkeypatch):
    monkeypatch.setattr(SecretaryBot, "user", property(lambda s: SimpleNamespace(id=5)))
    bot = SecretaryBot(w.settings, None, None, guilds=MemoryGuildStore(), secretary_store=w.store)
    bot.scheduler.team = w.team
    ev = discord_event(w)
    await bot.on_scheduled_event_create(ev)
    assert w.store.meeting_by_event(GUILD, 7001)["title"] == "Standup"
    ev.name = "Renamed"
    await bot.on_scheduled_event_update(ev, ev)
    assert w.store.meeting_by_event(GUILD, 7001)["title"] == "Renamed"
    await bot.on_scheduled_event_delete(ev)
    assert w.store.meeting_by_event(GUILD, 7001)["status"] == "cancelled"
    assert bot.intents.guild_scheduled_events


async def test_dm_replies_reach_the_scheduler_before_the_agent(w, monkeypatch):
    monkeypatch.setattr(SecretaryBot, "user", property(lambda s: SimpleNamespace(id=5)))
    bot = SecretaryBot(w.settings, None, None, guilds=MemoryGuildStore(), secretary_store=w.store)
    bot.scheduler = w.svc
    monkeypatch.setattr(SecretaryBot, "guilds", property(lambda s: [w.guild]))
    meeting_at(w, NOW)
    w.now = NOW
    await w.svc.tick()
    replies = []

    async def reply(text, **kw):
        replies.append(text)

    msg = SimpleNamespace(author=w.bob, content="ติดลูกค้า", reply=reply, guild=None, channel=None)
    await bot._direct_message(msg)
    assert replies and "Sync" in replies[0]
    assert w.store.get_meeting("m1")["attendance_notes"]["11"]["note"] == "ติดลูกค้า"


# ------------------------------------------------------------------ PDF
def seed_receipts(store):
    mk(store, 1, ALICE, "Alice", "Makro", "2026-10-01", 1000.0)
    mk(store, 2, ALICE, "Alice", "Grab", "2026-10-05", 250.5, "ค่าเดินทาง", conf=0.4)
    mk(store, 3, BOB, "Bob", "Makro", "2026-10-05", 9000.0)
    usd = mk(store, 4, BOB, "Bob", "AWS", "2026-10-06", 131.99, "อื่นๆ")
    store.update_receipt(usd.id, {"currency": "USD"})
    gone = mk(store, 5, BOB, "Bob", "Dup", "2026-10-07", 777.0)
    store.set_status(gone.id, "void", reason="dup", by="x")
    mk(store, 6, ALICE, "Other", "Elsewhere", "2026-10-02", 5.0, guild=OTHER_GUILD)


def thai_font():
    fonts = pdf_report.find_fonts()
    if not fonts.thai:
        pytest.skip("no Thai-capable font installed")
    return fonts


def test_expense_pdf_is_a_pdf_with_per_currency_totals_and_no_void(w):
    seed_receipts(w.ledger_store)
    thai_font()
    c = w.ctx()
    r = SecretaryTools(c).make_report_pdf(
        "expense_summary", "2026-10-01", "2026-10-31", "สรุปเดือนตุลาคม"
    )
    assert r["status"] == "created" and r["file_name"].endswith(".pdf")
    assert r["totals_by_currency"] == {"THB": 10250.5, "USD": 131.99}  # never added together
    assert r["receipts"] == 4  # the voided one is excluded
    ((name, data),) = c.attachments
    assert name == r["file_name"] and data.startswith(b"%PDF") and data.rstrip().endswith(b"%%EOF")
    assert r["thai_font"] is True


def test_pdf_embeds_a_font_with_thai_glyphs(w):
    fonts = thai_font()
    assert pdf_report.supports_thai(fonts.regular)
    data = pdf_report.custom_pdf(
        org="ร้านกาแฟ",
        title="สวัสดีครับ ใบเสร็จ",
        body="ค่าใช้จ่ายเดือนนี้ ก่ำ กิ้ง",
        generated_at=NOW,
        fonts=fonts,
    )
    assert data.startswith(b"%PDF") and b"/FontFile" in data and b"/ToUnicode" in data
    assert fonts.family.encode() in data or b"/BaseFont" in data


def test_pdf_scoping_matches_the_ledger_tools(w):
    seed_receipts(w.ledger_store)
    c = w.ctx(uid=ALICE, name="Alice", is_exec=False, can_email=False)
    r = SecretaryTools(c).make_report_pdf("receipt_list", "2026-10-01", "2026-10-31")
    assert r["totals_by_currency"] == {"THB": 1250.5} and r["receipts"] == 2
    other = SecretaryTools(w.ctx(uid=BOB, name="Bob", is_exec=False)).make_report_pdf(
        "receipt_list", "2026-10-01", "2026-10-31"
    )
    assert other["receipts"] == 2 and other["totals_by_currency"] == {"THB": 9000.0, "USD": 131.99}


def test_pdf_validates_input_and_can_be_switched_off(w):
    tl = w.tools()
    assert "required" in tl.make_report_pdf("expense_summary")["error"]
    assert "kind" in tl.make_report_pdf("poster")["error"]
    assert w.tools(documents=False).make_report_pdf("custom", notes="x") == {"status": "disabled"}


def test_meeting_minutes_pdf_uses_the_stored_summary_and_scoping(w):
    thai_font()
    m = meeting_at(w, NOW - timedelta(days=1), summary="ตกลงเปิดสาขาใหม่", status="done", attended=[10],
                   attendance_notes={"11": {"name": "Bob", "note": "ติดลูกค้า"}})  # fmt: skip
    c = w.ctx()
    r = SecretaryTools(c).make_report_pdf("meeting_minutes", "2026-10-13", "2026-10-14")
    assert (
        r["status"] == "created"
        and r["meeting"] == "Sync"
        and c.attachments[0][1].startswith(b"%PDF")
    )
    outsider = w.tools(uid=12, name="Cat", is_exec=False).make_report_pdf(
        "meeting_minutes", meeting_id=m["id"]
    )
    assert outsider["error"] == "not_allowed"
    assert (
        "no meeting summary"
        in w.tools().make_report_pdf("meeting_minutes", "2026-01-01", "2026-01-02")["error"]
    )


def test_font_search_falls_back_without_thai(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_report, "KNOWN_FONTS", [("/nope/a.ttf", None)])
    monkeypatch.setattr(pdf_report, "LATIN_FALLBACKS", [])
    f = pdf_report.find_fonts()
    assert f.regular is None and f.thai is False
    data = pdf_report.custom_pdf(
        org="Shop", title="Report", body="plain latin", generated_at=NOW, fonts=f
    )
    assert data.startswith(b"%PDF")  # still builds with the built-in font


def test_configured_font_wins(tmp_path):
    fonts = thai_font()
    assert pdf_report.find_fonts(fonts.regular).regular == fonts.regular


# ------------------------------------------------------------------ email
EMAIL = dict(to=["Client@Example.com"], subject="ใบเสนอราคา", body="สวัสดีครับ\nแนบไฟล์ตามนี้")


def test_email_draft_shows_the_draft_and_send_needs_a_later_confirmation(w):
    tl = w.tools()
    d = tl.draft_email(**EMAIL)
    assert d["status"] == "drafted" and d["draft"]["to"] == ["Client@example.com"]
    assert tl.send_email(d["draft_id"])["status"] == "needs_confirmation"  # same request
    assert w.store.get_email(d["draft_id"])["status"] == "draft"
    r = w.tools().send_email(d["draft_id"])  # a later request = the user said yes
    assert r["status"] == "not_configured" and "ยังไม่ได้ตั้งค่าการส่งอีเมล" in r["message"]
    e = w.store.get_email(d["draft_id"])
    assert e["status"] == "logged" and e["sender"] == "log_only" and e["confirmed_by"] == "user 1"
    assert e["author_id"] == 1 and e["sent_at"]
    assert w.tools().send_email(d["draft_id"])["error"].startswith("this draft is already")


def test_email_is_for_executives_managers_and_listed_roles_only(w):
    tl = w.tools(uid=12, name="Cat", is_exec=False, can_email=False)
    assert tl.draft_email(**EMAIL)["error"] == "not_allowed"
    d = w.tools().draft_email(**EMAIL)
    assert tl.send_email(d["draft_id"])["error"] == "not_allowed"
    other = w.tools(uid=2, name="Other Exec")
    assert other.send_email(d["draft_id"])["error"] == "draft not found"  # drafts are per author


def test_email_validates_addresses(w):
    tl = w.tools()
    r = tl.draft_email(["not-an-email", "a@b.co"], "s", "b")
    assert r["error"] == "invalid email address" and r["invalid"] == ["not-an-email"]
    assert tl.draft_email([], "s", "b")["error"]
    assert tl.draft_email(["a@b.co"], "", "b")["error"]
    assert tl.draft_email(["a@b.co"], "s", "b", cc=["x@@y"])["invalid"] == ["x@@y"]
    assert clean_addresses(["A@B.com", "a@b.com", " <c@d.org> ", "bad@", "x y@z.com"]) == (
        ["A@b.com", "c@d.org"],
        ["bad@", "x y@z.com"],
    )


def test_email_rate_limit_is_20_per_day_per_guild(w):
    for i in range(20):
        d = w.tools().draft_email(["a@b.co"], f"s{i}", "b")
        w.tools().send_email(d["draft_id"])
    d = w.tools().draft_email(["a@b.co"], "one more", "b")
    r = w.tools().send_email(d["draft_id"])
    assert r == {"error": "rate_limited", "limit_per_day": 20}
    w.now += timedelta(days=1, minutes=1)  # the day rolls over
    assert w.tools().send_email(d["draft_id"])["status"] == "needs_confirmation"
    assert w.tools().send_email(d["draft_id"])["status"] == "not_configured"


def test_email_can_attach_a_pdf_made_earlier(w):
    seed_receipts(w.ledger_store)
    made = w.tools().make_report_pdf("receipt_list", "2026-10-01", "2026-10-31")
    d = w.tools().draft_email(**EMAIL, attach_file=made["file_name"])  # a later message
    assert d["draft"]["attachment"] == made["file_name"]
    assert "no PDF named" in w.tools().draft_email(**EMAIL, attach_file="ghost.pdf")["error"]
    sent = []

    class Spy(LogOnlySender):
        def send(self, mail):
            sent.append(mail)
            return super().send(mail)

    SecretaryTools(w.ctx(sender=Spy())).send_email(d["draft_id"])  # a later request
    assert sent and sent[-1].attachments[0][0] == made["file_name"]
    assert (
        sent[-1].attachments[0][1].startswith(b"%PDF")
        and sent[-1].attachments[0][2] == "application/pdf"
    )


def test_gmail_sender_builds_the_api_requests():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.host == "oauth2.googleapis.com":
            return httpx.Response(200, json={"access_token": "tok-1"})
        return httpx.Response(200, json={"id": "gm-42"})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    g = GmailSender("cid", "secret", "refresh", "salid@example.com", http)
    assert g.configured and not GmailSender("", "", "", "").configured
    res = g.send(
        OutgoingEmail(
            ["a@b.co"], "หัวข้อ", "เนื้อหา", ["c@d.co"], [("r.pdf", b"%PDF-1", "application/pdf")]
        )
    )
    assert res.ok and res.status == "sent" and res.message_id == "gm-42"
    tok, send = calls
    assert dict(x.split("=") for x in tok.content.decode().split("&")) == {
        "client_id": "cid", "client_secret": "secret", "refresh_token": "refresh", "grant_type": "refresh_token",
    }  # fmt: skip
    assert str(send.url) == "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
    assert send.headers["authorization"] == "Bearer tok-1"
    raw = base64.urlsafe_b64decode(json.loads(send.content)["raw"])
    assert b"To: a@b.co" in raw and b"Cc: c@d.co" in raw and b"From: salid@example.com" in raw
    assert b'filename="r.pdf"' in raw and b"application/pdf" in raw


def test_gmail_failure_is_reported_without_leaking_contents():
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    res = GmailSender("a", "b", "c", "d@e.fr", http).send(OutgoingEmail(["a@b.co"], "s", "b"))
    assert not res.ok and res.status == "failed" and res.detail == "HTTPStatusError"


def test_send_email_with_a_working_sender_marks_it_sent(w):
    class Ok:
        name, configured = "gmail", True

        def send(self, mail):
            from salid.email_out import SendResult

            return SendResult(True, "sent", message_id="m-1")

    d = w.tools().draft_email(**EMAIL)
    r = SecretaryTools(w.ctx(sender=Ok())).send_email(d["draft_id"])
    assert r["status"] == "sent" and w.store.get_email(d["draft_id"])["message_id"] == "m-1"


def test_email_feature_off_hides_everything(w):
    assert w.tools(email=False).draft_email(**EMAIL) == {"status": "disabled"}


# ------------------------------------------------------------------ prompt + agent wiring
def test_prompts_mention_the_secretary_tools(settings):
    p = system_prompt(settings, scoped=False, secretary=True)
    for word in ("schedule_meeting", "make_report_pdf", "draft_email", "needs_confirmation"):
        assert word in p
    assert "schedule_meeting" not in system_prompt(settings, scoped=False)
    for lang in ("th", "en", "auto"):
        base = system_prompt(settings, scoped=False, language=lang)
        assert "นัดประชุม" in base and "PDF" in base and "อีเมล" in base  # capabilities lines


def test_tool_set_follows_features_and_roles(w):
    from salid.secretary import build_secretary_tools

    names = lambda **kw: {f.__name__ for f in build_secretary_tools(w.tools(**kw), [])}  # noqa: E731
    assert names() == {
        "schedule_meeting", "list_meetings", "cancel_meeting", "reschedule_meeting",
        "make_report_pdf", "draft_email", "send_email",
    }  # fmt: skip
    assert names(can_email=False, scheduling=False) == {"make_report_pdf"}
    assert names(documents=False, email=False) >= {"schedule_meeting"}


def test_agent_gets_the_secretary_tools(w, settings):
    from conftest import scripted

    from salid.secretary import build_secretary_tools

    tools = build_secretary_tools(w.tools(), [])
    agent = build_agent(settings, tools, scoped=False, secretary=True, model=scripted())
    assert {tool.__name__ for tool in agent.tools} >= {"schedule_meeting", "send_email"}


def test_ask_runs_a_scheduling_turn_end_to_end(w, settings):
    from conftest import fc_resp, scripted, text_resp

    from salid.qa import ask

    args = dict(title="Sync", start=START, attendees=["Ann"])
    ctx = w.ctx()
    llm = scripted(fc_resp(("schedule_meeting", args)), text_resp("สรุปนัดให้แล้วครับ ยืนยันไหม"))
    ans = ask(settings, w.ledger_store, "นัด Ann วันจันทร์บ่ายสอง", guild_id=GUILD, is_exec=True,
              user_id=1, model=llm, secretary=ctx)  # fmt: skip
    assert ans.text.startswith("สรุปนัด") and ("schedule_meeting", args) in [
        (n, {k: v for k, v in a.items() if k in args}) for n, a in ans.tool_calls
    ]
    assert w.guild.created == []  # one request can never create the event
    tool_result = llm.requests[1].contents[-1].parts[0].function_response.response
    assert tool_result["status"] == "needs_confirmation"
    assert ctx.ledger is not None


def test_ask_returns_the_pdfs_as_attachments(w, settings):
    from conftest import fc_resp, scripted, text_resp

    from salid.qa import ask

    seed_receipts(w.ledger_store)
    ctx = w.ctx()
    llm = scripted(
        fc_resp(
            (
                "make_report_pdf",
                dict(kind="expense_summary", start_date="2026-10-01", end_date="2026-10-31"),
            )
        ),
        text_resp("แนบรายงานให้แล้วครับ"),
    )
    ans = ask(settings, w.ledger_store, "ขอรายงานเดือนนี้เป็น PDF", guild_id=GUILD, is_exec=True,
              user_id=1, model=llm, secretary=ctx)  # fmt: skip
    assert [n for n, _ in ans.attachments] == ["expense_summary_2026-10-01_2026-10-31.pdf"]
    assert ans.attachments[0][1].startswith(b"%PDF")


def test_secretary_store_round_trips_on_both_backends():
    from fake_firestore import OfflineFirestoreStore  # noqa: F401  (ensures the fake module loads)

    class FakeFs:
        def __init__(self):
            self.docs = {}

        def collection(self, name):
            return FakeCol(self, name)

    class FakeCol:
        def __init__(self, fs, name, flt=None):
            self.fs, self.name, self.flt = fs, name, flt

        def document(self, did):
            return FakeDoc(self.fs, self.name, did)

        def where(self, filter):
            return FakeCol(self.fs, self.name, (filter.field_path, filter.value))

        def stream(self):
            for (n, _), d in list(self.fs.docs.items()):
                if n == self.name and (self.flt is None or d.get(self.flt[0]) == self.flt[1]):
                    yield SimpleNamespace(to_dict=lambda d=d: d)

    class FakeDoc:
        def __init__(self, fs, name, did):
            self.fs, self.key = fs, (name, did)

        def set(self, data):
            self.fs.docs[self.key] = data

        def get(self):
            d = self.fs.docs.get(self.key)
            return SimpleNamespace(exists=d is not None, to_dict=lambda: d)

    from salid.secretary_store import FirestoreDocs

    for backend in (SqliteDocs(), FirestoreDocs(client=FakeFs())):
        s = SecretaryStore(backend)
        s.put_meeting(
            {
                "id": "a",
                "guild_id": 1,
                "event_id": 5,
                "status": "scheduled",
                "start_utc": "2026-10-20T00:00:00+00:00",
            }
        )
        s.put_meeting(
            {
                "id": "b",
                "guild_id": 2,
                "event_id": 6,
                "status": "scheduled",
                "start_utc": "2026-10-21T00:00:00+00:00",
            }
        )
        assert [m["id"] for m in s.list_meetings(1)] == ["a"]
        assert [m["id"] for m in s.list_meetings()] == ["a", "b"]
        assert s.meeting_by_event(1, "5")["id"] == "a" and s.meeting_by_event(1, 6) is None
        assert s.get_meeting("b")["guild_id"] == 2 and s.get_meeting("zz") is None
        s.put_email(
            {"id": "e1", "guild_id": 1, "status": "sent", "sent_at": "2026-10-14T00:00:00+00:00"}
        )
        assert s.count_emails_since(1, datetime(2026, 10, 13, tzinfo=UTC), ("sent",)) == 1
        assert s.count_emails_since(2, datetime(2026, 10, 13, tzinfo=UTC), ("sent",)) == 0


def test_calendar_error_codes():
    assert CalendarError("missing_permission").code == "missing_permission"
    g = Guild(manage_events=False)
    with pytest.raises(CalendarError):
        DiscordCalendar.check_permissions(g)
