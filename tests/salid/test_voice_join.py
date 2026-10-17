"""Robust voice join: one join at a time per guild, connect timeout + cleanup, stale voice
client recovery, auto-join backoff (the 'Already connected to a voice channel' incidents)."""

import asyncio
import logging
from types import SimpleNamespace

import pytest
from conftest import FakeClient, FixedClockSettings
from test_meeting import FakeGuild, FakeVC, FakeVoiceChannel, make_member

from salid import meeting as mt
from salid.bot import SecretaryBot
from salid.guild_settings import MemoryGuildStore
from salid.i18n import t

ME = SimpleNamespace(id=999, mention="<@999>")


class RealishChannel(FakeVoiceChannel):
    """connect() behaves like discord.py: refuses when the guild already has a voice client."""

    def __init__(self, id=77, hang=False, **kw):
        super().__init__(id=id, **kw)
        self.connects, self.hang, self.new_vcs = 0, hang, []

    async def connect(self, cls=None):
        self.connects += 1
        if getattr(self.guild, "voice_client", None) is not None:
            raise RuntimeError("Already connected to a voice channel.")
        vc = FakeVC()
        vc.channel = self
        self.guild.voice_client = vc  # discord.py registers the client before the handshake ends
        self.new_vcs.append(vc)
        if self.hang:
            await asyncio.sleep(3600)
        return vc


@pytest.fixture
def env(monkeypatch):
    guild = FakeGuild()
    guild.voice_client = None
    chan = RealishChannel()
    chan.guild = guild
    other = RealishChannel(id=88, name="อีกห้อง")
    other.guild = guild
    guild.get_channel = {77: chan, 88: other}.get
    bot = SecretaryBot(
        FixedClockSettings(
            exec_user_ids={5},
            voice_reply=False,
            voice_mode="classic",
            voice_connect_timeout_seconds=0.05,
        ),
        None,
        FakeClient([]),
        guilds=MemoryGuildStore(),
    )
    monkeypatch.setattr(SecretaryBot, "user", property(lambda s: ME))
    return SimpleNamespace(guild=guild, chan=chan, other=other, bot=bot, mgr=bot.meetings)


def stale(env, channel, connected=True):
    vc = FakeVC()
    vc.channel, vc.connected = channel, connected
    env.guild.voice_client = vc
    return vc


async def test_concurrent_joins_connect_once(env):
    a = make_member(env.guild, 10, "แอน", env.chan)
    b = make_member(env.guild, 11, "บอม", env.chan)
    r1, r2 = await asyncio.gather(env.mgr.join(a), env.mgr.join(b))
    assert env.chan.connects == 1
    assert r1 == t("meet_joined", "th", channel=env.chan.name)
    assert r2 == t("meet_already", "th", channel=env.chan.name)  # waited, then saw the session
    assert list(env.mgr.sessions) == [(1000, 77)]
    await env.mgr.end((1000, 77), "command")


async def test_resume_and_auto_join_race_connects_once(env, monkeypatch):
    m = make_member(env.guild, 10, "แอน", env.chan)
    env.chan.members = [m]
    assert "ห้องประชุม" in await env.mgr.join(m)
    rec = env.bot.secretary_store.get_active_meeting(1000, 77)
    await env.mgr.shutdown()  # the restart: the record stays
    env.guild.voice_client = None
    env.chan.connects = 0
    monkeypatch.setattr(env.bot, "get_guild", lambda gid: env.guild)
    await asyncio.gather(env.mgr._resume(rec), env.mgr._auto_join(env.chan))
    assert env.chan.connects == 1
    assert list(env.mgr.sessions) == [(1000, 77)]
    await env.mgr.end((1000, 77), "command")


async def test_connect_timeout_disconnects_and_next_join_succeeds(env):
    env.chan.hang = True
    m = make_member(env.guild, 10, "แอน", env.chan)
    assert await env.mgr.join(m) == t("meet_connect_failed", "th")
    hung = env.chan.new_vcs[0]
    assert hung.disconnected and not env.mgr.sessions and not env.mgr._joining
    env.guild.voice_client = None if hung.disconnected else hung  # FakeVC does not unregister
    env.chan.hang = False
    assert "ห้องประชุม" in await env.mgr.join(m)
    assert (1000, 77) in env.mgr.sessions
    await env.mgr.end((1000, 77), "command")


async def test_connected_client_without_session_is_reused(env):
    old = stale(env, env.chan)
    m = make_member(env.guild, 10, "แอน", env.chan)
    assert "ห้องประชุม" in await env.mgr.join(m)
    assert env.chan.connects == 0 and env.mgr.sessions[(1000, 77)].vc is old
    assert not old.disconnected
    await env.mgr.end((1000, 77), "command")


async def test_client_in_another_channel_is_disconnected_first(env):
    old = stale(env, env.other)

    async def dropped(force=False):
        old.disconnected = True
        env.guild.voice_client = None

    old.disconnect = dropped
    m = make_member(env.guild, 10, "แอน", env.chan)
    assert "ห้องประชุม" in await env.mgr.join(m)
    assert old.disconnected and env.chan.connects == 1
    await env.mgr.end((1000, 77), "command")


async def test_disconnected_stale_client_is_replaced(env):
    old = stale(env, env.chan, connected=False)

    async def dropped(force=False):
        old.disconnected = True
        env.guild.voice_client = None

    old.disconnect = dropped
    m = make_member(env.guild, 10, "แอน", env.chan)
    assert "ห้องประชุม" in await env.mgr.join(m)
    assert old.disconnected and env.chan.connects == 1
    await env.mgr.end((1000, 77), "command")


async def test_three_failures_back_off_auto_join_but_manual_still_works(env, caplog):
    env.chan.hang = True
    env.chan.members = [m := make_member(env.guild, 10, "แอน", env.chan)]
    key = (1000, 77)
    with caplog.at_level(logging.ERROR, logger="salid.meeting"):
        for _ in range(3):
            env.guild.voice_client = None
            await env.mgr._auto_join(env.chan)
    assert env.chan.connects == 3
    assert sum("no auto-join" in r.getMessage() for r in caplog.records) == 1
    assert not env.mgr._auto_wanted(env.chan)  # backed off
    env.chan.hang = False
    env.guild.voice_client = None
    assert "ห้องประชุม" in await env.mgr.join(m)  # a manual join overrides
    assert key in env.mgr.sessions
    await env.mgr.end(key, "command")


async def test_failures_outside_the_window_do_not_back_off(env):
    now = [1000.0]
    env.mgr._now = lambda: now[0]
    key = (1000, 77)
    for _ in range(2):
        env.mgr._note_join_result(key, False)
    now[0] += mt.JOIN_FAIL_WINDOW_S + 1
    env.mgr._note_join_result(key, False)
    assert key not in env.mgr._cooldown


_ = mt
