"""Voice bodies (main bot + helper accounts), auto-join by importance, preemption. Offline: fake
clients, fake voice connections, no Discord."""

import asyncio
import logging
from types import SimpleNamespace

import discord
import pytest
from conftest import FakeClient, FixedClockSettings
from test_meeting import FakeGuild, FakeVC, FakeVoiceChannel, make_member

from salid import meeting as mt
from salid.bot import SecretaryBot
from salid.guild_settings import GuildSettingsUpdate, MemoryGuildStore
from salid.i18n import t
from salid.permissions import HELPER_PERMISSIONS, helper_invite_url
from salid.voice_pool import (
    Claim,
    VoicePool,
    pick_victim,
    room_score,
)

GID = 1000
A, B, C, D = 77, 78, 79, 80


# ------------------------------------------------------------------ importance rules (pure)
def test_room_score_orders_scheduled_over_execs_over_headcount():
    assert room_score(2, 0, True) > room_score(6, 2, False) > room_score(6, 0, False)
    assert room_score(2, 1, False) == 2 * 5 + 30
    assert room_score(8, 5, False) == 8 * 5 + 2 * 30  # executives are capped
    assert room_score(1, 1, False) == 0 < room_score(2, 0, False)  # alone ranks below any room
    assert room_score(1, 0, True) == 100  # scheduled beats everything even for one person


def claim(key, humans=2, execs=0, origin="auto", scheduled=False, started=0.0):
    return Claim(key, room_score(humans, execs, scheduled), humans, started, origin, scheduled)


def test_pick_victim_margin_and_protection():
    weak = claim("w", 2)  # score 10
    strong = claim("s", 5, 1)  # 55
    assert pick_victim(claim("n", 6, 1), [weak, strong]) is weak  # 60 vs 10
    assert pick_victim(claim("n", 3), [weak, strong]) is None  # 15 vs 10: below the margin
    assert pick_victim(claim("n", 2, 1), [weak]) is weak  # 40 vs 10: >= 20 ahead
    # explicit and scheduled sessions are never victims
    assert pick_victim(claim("n", 9, 2), [claim("c", 2, origin="command")]) is None
    assert pick_victim(claim("n", 1, scheduled=True), [claim("c", 2, origin="command")]) is None
    assert pick_victim(claim("n", 1, scheduled=True), [claim("c", 2, scheduled=True)]) is None
    assert pick_victim(claim("n", 1, scheduled=True), [claim("c", 4, origin="scheduled")]) is None


def test_pick_victim_alone_yields_to_any_room_and_scheduled_beats_everything():
    alone = claim("a", 1)
    assert pick_victim(claim("n", 2), [alone]) is alone  # 10 vs 0, below the margin but alone
    assert pick_victim(claim("n", 1), [alone]) is None  # alone does not push alone
    busy = claim("b", 9, 2)
    assert pick_victim(claim("n", 1, scheduled=True), [busy]) is busy  # scheduled always wins
    # the weakest goes first; on a tie the one that started later
    early, late = claim("e", 3, started=1.0), claim("l", 3, started=2.0)
    assert pick_victim(claim("n", 1, scheduled=True), [early, late]) is late


# ------------------------------------------------------------------ fakes
class FakeHelper:
    def __init__(self, ready, guild_map, fail=None):
        self.ready, self.guild_map, self.fail = ready, guild_map, fail
        self.application_id = None
        self.closed = False

    async def start(self, token):
        if self.fail:
            raise self.fail
        self.ready()
        await asyncio.Event().wait()

    def get_guild(self, gid):
        return self.guild_map.get(gid)

    async def close(self):
        self.closed = True


def pool_with(bot, specs):
    """specs: list of dicts(guilds=..., fail=...) one per token."""
    it = iter(specs)

    def factory(ready, leave):
        s = next(it)
        return FakeHelper(ready, s.get("guilds", {}), s.get("fail"))

    return VoicePool(bot, [f"tok-{i}-secret" for i in range(len(specs))], factory=factory)


@pytest.fixture
def world(monkeypatch):
    guild = FakeGuild(GID)
    guild.afk_channel = None
    chans = {cid: FakeVoiceChannel(cid, f"room{cid}") for cid in (A, B, C, D)}
    for ch in chans.values():
        ch.guild = guild
    guild.get_channel = chans.get
    guild.voice_channels = list(chans.values())
    bot = SecretaryBot(
        FixedClockSettings(
            exec_user_ids={5}, voice_reply=False, voice_mode="classic", db_path=":memory:"
        ),
        None,
        FakeClient([]),
        guilds=MemoryGuildStore(),
    )
    monkeypatch.setattr(SecretaryBot, "user", property(lambda s: SimpleNamespace(id=999)))
    monkeypatch.setattr(SecretaryBot, "get_guild", lambda self, gid: guild if gid == GID else None)
    mgr = bot.meetings
    vcs: list[FakeVC] = []

    async def connect(body, channel):
        vc = FakeVC()
        vc.body = body.index
        vcs.append(vc)
        return vc

    monkeypatch.setattr(mgr, "_connect", connect)
    return SimpleNamespace(guild=guild, chans=chans, bot=bot, mgr=mgr, vcs=vcs)


async def with_helpers(w, n=1, in_guild=True):
    guilds = {GID: w.guild} if in_guild else {}
    w.mgr.pool = pool_with(w.bot, [{"guilds": guilds}] * n)
    w.pool = w.mgr.pool
    w.pool.start()
    await w.pool.wait_ready()
    return w.pool


def people(w, ch, *uids):
    members = [make_member(w.guild, u, f"u{u}", w.chans[ch]) for u in uids]
    w.chans[ch].members = members
    return members


async def command_join(w, ch, uid=10):
    m = make_member(w.guild, uid, f"u{uid}", w.chans[ch])
    w.chans[ch].members = [*w.chans[ch].members, m]
    return await w.mgr.join(m)


async def settle(mgr):
    await asyncio.gather(*list(mgr._auto_tasks.values()))


async def enter(w, ch, uid):
    m = make_member(w.guild, uid, f"u{uid}", w.chans[ch])
    w.chans[ch].members = [*w.chans[ch].members, m]
    await w.mgr.on_voice_activity(
        m, SimpleNamespace(channel=None), SimpleNamespace(channel=w.chans[ch])
    )
    await settle(w.mgr)
    return m


async def end_all(w):
    for ch in w.chans.values():
        ch.members = []  # nobody waits: ending frees bodies without rejoins
    for key in list(w.mgr.sessions):
        await w.mgr.end(key, "empty")
    await w.mgr.pool.close()


# ------------------------------------------------------------------ pool startup
async def test_pool_without_tokens_is_just_the_main_bot(world):
    pool = VoicePool(world.bot, [])
    pool.start()
    await pool.wait_ready()
    assert [b.name for b in pool.bodies] == ["Salid"] and pool.usable(GID) == [pool.main]
    assert world.bot.voice_pool.bodies[0].index == 0  # the bot always owns a pool


async def test_pool_with_two_tokens_logs_both_helpers_in(world):
    pool = await with_helpers(world, 2)
    assert [b.name for b in pool.bodies] == ["Salid", "Salid · 2", "Salid · 3"]
    assert [b.index for b in pool.usable(GID)] == [0, 1, 2]
    assert pool._tokens == []  # tokens are not kept after start
    await pool.close()
    assert all(b.client.closed for b in pool.bodies[1:])


async def test_failing_token_is_unavailable_and_never_logged(world, caplog):
    caplog.set_level(logging.DEBUG)
    it = iter(
        [{"fail": discord.LoginFailure("bad token tok-0-secret")}, {"guilds": {GID: world.guild}}]
    )
    pool = VoicePool(
        world.bot,
        ["tok-0-secret", "tok-1-secret"],
        factory=lambda ready, leave: (
            lambda s: FakeHelper(ready, s.get("guilds", {}), s.get("fail"))
        )(next(it)),
    )
    pool.start()
    await pool.wait_ready()
    assert [b.index for b in pool.usable(GID)] == [0, 2]
    assert "voice helper 1 unavailable" in caplog.text
    assert "secret" not in caplog.text and "tok-" not in caplog.text
    await pool.close()


async def test_helper_not_in_the_guild_is_skipped_and_gets_an_invite(world):
    pool = await with_helpers(world, 1, in_guild=False)
    pool.bodies[1].client.application_id = 4242
    assert [b.index for b in pool.usable(GID)] == [0]
    rows = pool.status_rows(GID, "en")
    assert rows[0]["in_guild"] and not rows[1]["in_guild"]
    assert pool.invite_links(GID) == [("Salid · 2", helper_invite_url(4242))]
    url = helper_invite_url(4242)
    assert "client_id=4242" in url and f"permissions={HELPER_PERMISSIONS}" in url
    assert HELPER_PERMISSIONS == (1 << 10) | (1 << 20) | (1 << 21) | (1 << 25)
    assert helper_invite_url(None) == ""
    await pool.close()


# ------------------------------------------------------------------ sessions per channel / bodies
async def test_two_simultaneous_meetings_use_two_bodies(world):
    await with_helpers(world, 1)
    assert "room77" in await command_join(world, A)
    assert "room78" in await command_join(world, B, uid=11)
    s1, s2 = world.mgr.sessions[(GID, A)], world.mgr.sessions[(GID, B)]
    assert {s1.body.index, s2.body.index} == {0, 1} and s1.vc is not s2.vc
    assert {vc.body for vc in world.vcs} == {0, 1}
    # a third command with no free body says which rooms hold the bodies
    reply = await command_join(world, C, uid=12)
    assert (
        reply == t("meet_busy", "th", rooms="#room77, #room78")
        and (GID, C) not in world.mgr.sessions
    )
    await end_all(world)


async def test_second_join_in_the_same_room_is_refused_and_other_guild_rooms_are_free(world):
    await with_helpers(world, 0)
    await command_join(world, A)
    assert "กำลังอยู่ในการประชุม" in await command_join(world, A, uid=11)
    await end_all(world)


async def test_end_frees_only_that_rooms_body(world):
    await with_helpers(world, 1)
    await command_join(world, A)
    await command_join(world, B, uid=11)
    held = world.mgr.sessions[(GID, B)].body.index
    await world.mgr.end((GID, A), "command")
    assert (GID, A) not in world.mgr.sessions and world.mgr.sessions[(GID, B)].body.index == held
    assert "room79" in await command_join(world, C, uid=12)
    await end_all(world)


# ------------------------------------------------------------------ leave per channel
async def test_leave_ends_the_session_of_the_room_the_person_is_in(world):
    await with_helpers(world, 1)
    await command_join(world, A)
    await command_join(world, B, uid=11)
    m = world.chans[B].members[0]
    assert await world.mgr.leave(m) == t("meet_left", "th")
    await asyncio.sleep(0.05)
    assert (GID, B) not in world.mgr.sessions and (GID, A) in world.mgr.sessions
    await end_all(world)


async def test_typed_leave_from_a_text_channel_uses_where_the_author_is_and_manager_needs_a_room(
    world,
):
    await with_helpers(world, 1)
    await command_join(world, A)
    await command_join(world, B, uid=11)
    outsider = make_member(world.guild, 30, "out")
    assert await world.mgr.leave(outsider, text_channel_id=555) == t("meet_leave_denied", "th")
    boss = make_member(world.guild, 31, "boss")
    boss.guild_permissions.manage_guild = True
    assert "#room77" in await world.mgr.leave(boss, text_channel_id=555)  # which room? (two)
    # the typed leave in a room's own text chat
    assert await world.mgr.leave(boss, text_channel_id=A) == t("meet_left", "th")
    await asyncio.sleep(0.05)
    assert (GID, A) not in world.mgr.sessions and (GID, B) in world.mgr.sessions
    await end_all(world)


# ------------------------------------------------------------------ auto-join
async def test_one_person_triggers_an_auto_join_with_the_announcement(world):
    await enter(world, A, 10)
    s = world.mgr.sessions[(GID, A)]
    assert s.origin == "auto" and s.body.index == 0
    assert world.chans[A].sent == [t("meet_announce_auto", "th")]
    assert (
        world.chans[A]
        .sent[0]
        .startswith("สลิดเข้ามาจดประชุมนะครับ ไม่ต้องการพิมพ์ “@Salid ออกประชุม” ได้เลย")
    )
    assert "leave meeting" in t("meet_announce_auto", "en")
    await end_all(world)


async def test_auto_join_needs_no_delay_and_duplicate_events_share_one_attempt(world, monkeypatch):
    calls = []

    async def no_sleep(s):
        calls.append(s)

    world.mgr._sleep = no_sleep
    m = make_member(world.guild, 10, "u10", world.chans[A])
    world.chans[A].members = [m]
    ev = (m, SimpleNamespace(channel=None), SimpleNamespace(channel=world.chans[A]))
    await world.mgr.on_voice_activity(*ev)
    await world.mgr.on_voice_activity(*ev)
    assert len(world.mgr._auto_tasks) == 1
    await settle(world.mgr)
    assert calls == [] and list(world.mgr.sessions) == [(GID, A)] and len(world.vcs) == 1
    await end_all(world)


async def test_optional_delay_waits_and_is_cancelled_when_the_room_empties(world):
    world.bot.s.auto_join_delay_seconds = 30.0
    gate = asyncio.Event()
    waited = []

    async def sleep(s):
        waited.append(s)
        await gate.wait()

    world.mgr._sleep = sleep
    m = make_member(world.guild, 10, "u10", world.chans[A])
    world.chans[A].members = [m]
    await world.mgr.on_voice_activity(
        m, SimpleNamespace(channel=None), SimpleNamespace(channel=world.chans[A])
    )
    await asyncio.sleep(0)
    assert waited == [30.0] and world.mgr.sessions == {}
    world.chans[A].members = []  # popped in and out
    await world.mgr.on_voice_activity(
        m, SimpleNamespace(channel=world.chans[A]), SimpleNamespace(channel=None)
    )
    await asyncio.sleep(0.01)
    gate.set()
    await asyncio.sleep(0.01)
    assert world.mgr.sessions == {} and world.vcs == []
    # staying through the wait joins
    gate.clear()
    world.chans[A].members = [m]
    await world.mgr.on_voice_activity(
        m, SimpleNamespace(channel=None), SimpleNamespace(channel=world.chans[A])
    )
    await asyncio.sleep(0)
    gate.set()
    await settle(world.mgr)
    assert (GID, A) in world.mgr.sessions
    await end_all(world)


async def test_min_humans_threshold_is_configurable(world):
    world.bot.s.auto_join_min_humans = 2
    await enter(world, A, 10)
    assert world.mgr.sessions == {}
    await enter(world, A, 11)
    assert (GID, A) in world.mgr.sessions
    await end_all(world)


async def test_mode_off_flag_off_afk_and_bots_do_not_auto_join(world):
    world.bot.guilds_store.save_settings(
        GID, GuildSettingsUpdate(channel_modes={str(A): "off"}), "x"
    )
    await asyncio.to_thread(world.bot.team.current, GID)
    await enter(world, A, 10)
    assert world.mgr.sessions == {}
    world.guild.afk_channel = world.chans[B]
    await enter(world, B, 11)
    assert world.mgr.sessions == {}
    bot_user = SimpleNamespace(id=1, bot=True)
    await world.mgr.on_voice_activity(
        bot_user, SimpleNamespace(channel=None), SimpleNamespace(channel=world.chans[C])
    )
    assert world.mgr._auto_tasks == {} and world.mgr.sessions == {}
    world.guild.afk_channel = None
    world.bot.guilds_store.save_settings(
        GID, GuildSettingsUpdate(features=dict(auto_join_meetings=False)), "x"
    )
    world.bot.team._cache.clear()
    await asyncio.to_thread(world.bot.team.current, GID)
    await enter(world, D, 12)
    assert world.mgr.sessions == {}
    # the command still works with the flag off
    assert "room79" in await command_join(world, C, uid=13)
    await end_all(world)


async def test_leave_command_blocks_auto_rejoin_for_an_hour(world):
    await enter(world, A, 10)
    m = world.chans[A].members[0]
    await world.mgr.leave(m)
    await asyncio.sleep(0.05)
    assert world.mgr.sessions == {}
    await enter(world, A, 11)  # someone else walks in: still no
    assert world.mgr.sessions == {}
    now = world.mgr._now()
    world.mgr._now = lambda: now + 61 * 60
    await enter(world, A, 12)
    assert (GID, A) in world.mgr.sessions
    await end_all(world)


async def test_spoken_leave_also_sets_the_cooldown(world, monkeypatch):
    await enter(world, A, 10)
    s = world.mgr.sessions[(GID, A)]
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สลิด ออกประชุม")
    await world.mgr.handle_segment(
        s, mt.Segment(10, mt.FRAME if hasattr(mt, "FRAME") else bytes(9600) * 100, 0.0, 2.0)
    )
    await asyncio.sleep(0.05)
    assert world.mgr.sessions == {}
    await enter(world, A, 11)
    assert world.mgr.sessions == {}


# ------------------------------------------------------------------ priority and preemption
async def test_second_room_goes_to_a_helper_without_preempting(world):
    await with_helpers(world, 1)
    await enter(world, A, 10)
    await enter(world, B, 11)
    assert {s.body.index for s in world.mgr.sessions.values()} == {0, 1}
    await end_all(world)


async def test_busy_room_with_executive_preempts_a_person_alone(world):
    # only the main body
    await enter(world, A, 10)  # one person alone holds it
    await enter(world, B, 11)  # B: another single person: no preemption
    assert list(world.mgr.sessions) == [(GID, A)]
    await enter(world, B, 5)  # an executive joins B: 2 people incl. executive (40 vs 0)
    assert list(world.mgr.sessions) == [(GID, B)]
    assert t("meet_preempted", "th") == "ขอไปช่วยประชุมที่นัดไว้ก่อนนะครับ ส่วนที่คุยมาสรุปไว้ให้แล้ว"
    assert t("meet_preempted", "th") in world.chans[A].sent  # the old room was told first
    assert any("สรุป" in x or "ไม่มี" in x for x in world.chans[A].sent[1:])  # then its summary
    await end_all(world)


async def test_a_bigger_room_needs_the_margin_to_preempt_an_auto_session(world):
    people(world, A, 10, 11)  # 2 people: 10 points
    await world.mgr.on_voice_activity(
        world.chans[A].members[0],
        SimpleNamespace(channel=None),
        SimpleNamespace(channel=world.chans[A]),
    )
    await settle(world.mgr)
    assert (GID, A) in world.mgr.sessions
    people(world, B, 20, 21, 22)  # 15 points: only +5
    await world.mgr.on_voice_activity(
        world.chans[B].members[0],
        SimpleNamespace(channel=None),
        SimpleNamespace(channel=world.chans[B]),
    )
    await settle(world.mgr)
    assert list(world.mgr.sessions) == [(GID, A)]
    people(world, C, 30, 31, 32, 33, 34)  # 25 points: +15, still short
    await world.mgr.on_voice_activity(
        world.chans[C].members[0],
        SimpleNamespace(channel=None),
        SimpleNamespace(channel=world.chans[C]),
    )
    await settle(world.mgr)
    assert list(world.mgr.sessions) == [(GID, A)]
    people(world, D, 5, 40, 41, 42)  # exec + 4 people: 50 points: +40
    await world.mgr.on_voice_activity(
        world.chans[D].members[0],
        SimpleNamespace(channel=None),
        SimpleNamespace(channel=world.chans[D]),
    )
    await settle(world.mgr)
    assert list(world.mgr.sessions) == [(GID, D)]
    await end_all(world)


async def test_scheduled_meeting_preempts_an_ad_hoc_session_even_a_single_person(world):
    await enter(world, A, 10)
    m = make_member(world.guild, 50, "host", world.chans[B])
    world.chans[B].members = [m]
    ended = []

    async def on_end(s):
        ended.append(s)

    reply = await world.mgr.join(m, meeting_id="m1", on_end=on_end)
    assert "room78" in reply
    s = world.mgr.sessions[(GID, B)]
    assert s.origin == "scheduled" and s.meeting_id == "m1" and (GID, A) not in world.mgr.sessions
    await end_all(world)


async def test_commanded_and_scheduled_sessions_are_never_preempted(world):
    await command_join(world, A)  # explicit
    # an auto candidate with an executive crowd does not take it
    people(world, B, 5, 40, 41, 42)
    await world.mgr.on_voice_activity(
        world.chans[B].members[0],
        SimpleNamespace(channel=None),
        SimpleNamespace(channel=world.chans[B]),
    )
    await settle(world.mgr)
    assert list(world.mgr.sessions) == [(GID, A)]
    # not even a scheduled meeting takes a commanded session
    m = make_member(world.guild, 50, "host", world.chans[C])
    world.chans[C].members = [m]
    reply = await world.mgr.join(m, meeting_id="m2")
    assert reply == t("meet_busy", "th", rooms="#room77") and list(world.mgr.sessions) == [(GID, A)]
    await world.mgr.end((GID, A), "command")
    await world.mgr.join(m, meeting_id="m2")
    other = make_member(world.guild, 51, "x", world.chans[D])
    world.chans[D].members = [other]
    reply = await world.mgr.join(other, meeting_id="m3")  # scheduled never preempts scheduled
    assert reply.startswith(t("meet_busy", "th", rooms="")[:10])
    assert list(world.mgr.sessions) == [(GID, C)]
    await end_all(world)


async def test_command_join_without_a_free_body_follows_the_same_rules(world):
    await enter(world, A, 10)  # single, auto
    m = make_member(world.guild, 11, "u11", world.chans[B])
    world.chans[B].members = [m, make_member(world.guild, 12, "u12", world.chans[B])]
    assert "room78" in await world.mgr.join(m)  # 2 people beat one alone
    assert (
        list(world.mgr.sessions) == [(GID, B)] and world.mgr.sessions[(GID, B)].origin == "command"
    )
    await end_all(world)


async def test_a_freed_body_is_taken_by_a_waiting_room(world):
    await enter(world, A, 10)
    await enter(world, B, 11)  # waits: no body, no preemption
    assert list(world.mgr.sessions) == [(GID, A)]
    world.chans[A].members = []
    await world.mgr.end((GID, A), "empty")  # the room emptied: the body frees up
    await asyncio.sleep(0.05)
    await settle(world.mgr)
    assert list(world.mgr.sessions) == [(GID, B)]
    await end_all(world)


async def test_scheduled_adopt_turns_an_auto_session_into_a_scheduled_one(world):
    await enter(world, A, 10)
    s = world.mgr.sessions[(GID, A)]
    await world.mgr.adopt(s, "m9", None)
    assert s.origin == "scheduled" and s.meeting_id == "m9"
    assert world.bot.secretary_store.get_active_meeting(GID, A)["meeting_id"] == "m9"
    await end_all(world)


# ------------------------------------------------------------------ persistence / resume per body
async def test_records_are_per_channel_with_the_body_index_and_resume_reuses_the_body(world):
    await with_helpers(world, 1)
    await command_join(world, A)
    await command_join(world, B, uid=11)
    store = world.bot.secretary_store
    recs = {r["voice_channel_id"]: r for r in store.list_active_meetings()}
    assert set(recs) == {A, B} and {r["body_index"] for r in recs.values()} == {0, 1}
    helper_room = next(c for c, r in recs.items() if r["body_index"] == 1)
    await world.mgr.shutdown()  # the restart
    assert len(store.list_active_meetings()) == 2 and world.mgr.sessions == {}
    world.vcs.clear()
    await world.mgr.resume_all()
    assert set(world.mgr.sessions) == {(GID, A), (GID, B)}
    assert world.mgr.sessions[(GID, helper_room)].body.index == 1
    assert {vc.body for vc in world.vcs} == {0, 1}
    await world.mgr.end((GID, A), "command")
    assert store.get_active_meeting(GID, A) is None and store.get_active_meeting(GID, B)
    await end_all(world)


async def test_resume_falls_back_to_a_free_body_when_the_helper_is_gone(world):
    await with_helpers(world, 1)
    await command_join(world, A)
    await command_join(world, B, uid=11)
    await world.mgr.shutdown()
    await world.mgr.pool.close()
    world.mgr.pool = VoicePool(world.bot, [])  # the helper no longer starts
    await world.mgr.resume_all()
    # one room keeps the main body, the other has no body: it ends with its summary
    assert len(world.mgr.sessions) == 1
    assert len(world.bot.secretary_store.list_active_meetings()) == 1
    await end_all(world)


async def test_legacy_per_guild_record_is_cleared_with_its_channel(world):
    store = world.bot.secretary_store
    store.db.put("salid_active_meetings", str(GID), {"guild_id": GID, "voice_channel_id": A})
    store.clear_active_meeting(GID, A)
    assert store.list_active_meetings() == []


# ------------------------------------------------------------------ reconnect per body
async def test_a_helper_disconnect_reconnects_that_rooms_session_only(world, monkeypatch):
    monkeypatch.setattr(mt, "RECONNECT_DELAYS", (0,))
    await with_helpers(world, 1)
    await command_join(world, A)
    await command_join(world, B, uid=11)
    s_main = next(s for s in world.mgr.sessions.values() if s.body.index == 0)
    s_help = next(s for s in world.mgr.sessions.values() if s.body.index == 1)
    old_main = s_main.vc
    world.vcs.clear()
    await world.mgr.on_disconnected(GID, 1)
    assert [vc.body for vc in world.vcs] == [1] and s_help.vc is world.vcs[0]
    assert s_main.vc is old_main
    await end_all(world)


# ------------------------------------------------------------------ real connect routing / settings
async def test_connect_uses_the_bodys_own_channel_object(world):
    seen = []

    class Chan:
        async def connect(self, cls=None):
            seen.append(cls)
            return "vc"

    hguild = SimpleNamespace(id=GID, get_channel=lambda cid: Chan() if cid == A else None)
    body = SimpleNamespace(index=1, client=SimpleNamespace(get_guild=lambda g: hguild))
    mgr = SecretaryBot.meetings.fget if False else world.mgr
    assert await mt.MeetingManager._connect(mgr, body, world.chans[A]) == "vc"
    assert seen and seen[0].__name__ == "VoiceRecvClient"
    with pytest.raises(RuntimeError):
        await mt.MeetingManager._connect(mgr, body, world.chans[B])


def test_settings_parse_tokens_and_hide_them():
    from salid.config import Settings

    s = Settings(voice_tokens="aaa, bbb,,")
    assert s.voice_tokens == ["aaa", "bbb"] and "aaa" not in repr(s)
    assert Settings().voice_tokens == [] and Settings().auto_join_delay_seconds == 0
    assert Settings().auto_join_min_humans == 1


def test_auto_join_flag_defaults_on_and_has_a_hub_toggle():
    from salid.guild_settings import FEATURES, Features

    assert Features().auto_join_meetings is True and "auto_join_meetings" in FEATURES
    from pathlib import Path

    hub = (Path(__file__).parents[2] / "src/meeting_scribe/static/hub.html").read_text()
    assert '"f_auto_join_meetings_n"' in hub and '{ id: "auto_join_meetings" }' in hub
