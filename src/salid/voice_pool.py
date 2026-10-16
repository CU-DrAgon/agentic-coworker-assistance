"""Voice bodies: Discord allows one voice connection per bot account per guild, so when two
meetings happen at once Salid "splits" across helper bot accounts ("Salid · 2", "Salid · 3").

The main bot is body #0. Helpers are lightweight `discord.Client`s (guilds + voice states only,
no privileged intents) that run in the same event loop; they only hold a voice connection
(listen + play). Text, memory, answers and TTS generation all stay in the main bot.

This module also holds the importance rules used to decide which meeting gets a body when there
are fewer free bodies than meetings (`room_score`, `pick_victim`). Helper tokens are secrets:
they are never logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import discord

from .i18n import t

logger = logging.getLogger(__name__)

SCHEDULED_BONUS = 100  # a scheduled meeting happening now
EXEC_BONUS = 30  # per executive / manager in the room ...
EXEC_CAP = 2  # ... counted up to this many
HUMAN_POINTS = 5  # per person in a multi-person room
PREEMPT_MARGIN = 20  # a newcomer must outrank the weakest auto-joined session by this much
READY_WAIT_S = 20.0


def room_score(humans: int, execs: int, scheduled: bool) -> int:
    """Importance of a room. Scheduled (+100) beats executives (+30 each, capped) beats head
    count (+5 each). One person alone scores only the scheduled bonus, so they rank below any
    multi-person room and are the first to give up a body."""
    if humans <= 0 and not scheduled:
        return 0
    score = SCHEDULED_BONUS if scheduled else 0
    if humans >= 2:
        score += min(execs, EXEC_CAP) * EXEC_BONUS + humans * HUMAN_POINTS
    return score


@dataclass(frozen=True)
class Claim:
    """A room asking for (or holding) a body."""

    key: Any
    score: int
    humans: int
    started: float  # monotonic; earlier = more important on a tie
    origin: str  # command | scheduled | auto
    scheduled: bool = False


def pick_victim(candidate: Claim, active: list[Claim]) -> Claim | None:
    """The session that should give up its body for `candidate`, or None.

    Only auto-joined, non-scheduled sessions can be preempted: never one a person asked for by
    command and never a scheduled one. A scheduled candidate always wins against those; others
    must outrank the weakest by PREEMPT_MARGIN, except that a person sitting alone gives way to
    any room with at least two people."""
    pool = [c for c in active if c.origin == "auto" and not c.scheduled]
    if not pool:
        return None
    victim = min(pool, key=lambda c: (c.score, -c.started))
    if candidate.scheduled:
        return victim
    if candidate.score - victim.score >= PREEMPT_MARGIN:
        return victim
    if victim.humans <= 1 and candidate.humans >= 2 and candidate.score > victim.score:
        return victim
    return None


class VoiceBody:
    """One voice-capable account: the main bot (index 0) or a helper."""

    def __init__(self, index: int, client: Any, name: str, *, available: bool = False):
        self.index, self.client, self.name = index, client, name
        self.available = available  # logged in and ready
        self.settled = asyncio.Event()  # ready or failed: startup is over for this body
        if available:
            self.settled.set()

    @property
    def application_id(self):
        return getattr(self.client, "application_id", None)

    def in_guild(self, guild_id: int) -> bool:
        return self.index == 0 or self.client.get_guild(guild_id) is not None

    def __repr__(self) -> str:
        return f"<VoiceBody {self.index} {self.name} available={self.available}>"


class _HelperClient(discord.Client):
    """Voice-only: no commands, no messages. Reports its own voice disconnects."""

    def __init__(self, on_ready_cb: Callable[[], None], on_leave_cb) -> None:
        intents = discord.Intents.none()
        intents.guilds = True
        intents.voice_states = True
        super().__init__(intents=intents)
        self._ready_cb, self._leave_cb = on_ready_cb, on_leave_cb

    async def on_ready(self) -> None:
        self._ready_cb()

    async def on_voice_state_update(self, member, before, after) -> None:
        if member == self.user and before.channel is not None and after.channel is None:
            await self._leave_cb(before.channel.guild.id)


def default_client_factory(on_ready_cb, on_leave_cb) -> Any:
    return _HelperClient(on_ready_cb, on_leave_cb)


class VoicePool:
    """The main bot plus its helpers. `factory(on_ready, on_leave)` builds a helper client
    (injectable for tests); `on_leave(index, guild_id)` is awaited when a helper's own voice
    connection was dropped."""

    def __init__(
        self,
        main: Any,
        tokens: list[str] | None = None,
        *,
        factory: Callable[..., Any] = default_client_factory,
        on_leave=None,
    ) -> None:
        self.main = VoiceBody(0, main, "Salid", available=True)
        self.bodies: list[VoiceBody] = [self.main]
        self._tokens = list(tokens or [])
        self._factory, self._on_leave = factory, on_leave
        self._tasks: list[asyncio.Task] = []
        self._started = False

    # ---- lifecycle
    def start(self) -> None:
        """Log every helper in (as background tasks in the running loop). Idempotent."""
        if self._started:
            return
        self._started = True
        tokens, self._tokens = self._tokens, []  # the pool keeps no token after this
        for n, token in enumerate(tokens, start=1):
            index = n
            body_box: list[VoiceBody] = []

            def ready(box=body_box) -> None:
                box[0].available = True
                box[0].settled.set()

            async def leave(guild_id: int, i: int = index) -> None:
                if self._on_leave is not None:
                    await self._on_leave(i, guild_id)

            body = VoiceBody(index, self._factory(ready, leave), f"Salid · {index + 1}")
            body_box.append(body)
            self.bodies.append(body)
            self._tasks.append(asyncio.create_task(self._run(body, token)))

    async def _run(self, body: VoiceBody, token: str) -> None:
        try:
            await body.client.start(token)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # never log the exception text or traceback: they may quote it
            logger.warning("voice helper %d unavailable (%s)", body.index, type(e).__name__)
        finally:
            body.available = False
            body.settled.set()

    async def wait_ready(self, timeout: float = READY_WAIT_S) -> None:
        """Until every helper is ready or failed (bounded), so a resume can use them."""
        helpers = [b for b in self.bodies if b.index]
        if not helpers:
            return
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(timeout):
                for b in helpers:
                    await b.settled.wait()

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        for body in self.bodies[1:]:
            with contextlib.suppress(Exception):
                await body.client.close()
        for task in self._tasks:
            with contextlib.suppress(BaseException):
                await task
        self._tasks.clear()

    # ---- lookup
    def body(self, index: int | None) -> VoiceBody | None:
        return next((b for b in self.bodies if b.index == index), None)

    def usable(self, guild_id: int) -> list[VoiceBody]:
        """Bodies that are ready and are members of this guild, main bot first."""
        return [b for b in self.bodies if b.available and b.in_guild(guild_id)]

    # ---- status
    def status_rows(self, guild_id: int, lang: str) -> list[dict]:
        rows = []
        for b in self.bodies:
            if not b.available:
                state = t("voice_body_down", lang)
            elif b.in_guild(guild_id):
                state = t("voice_body_in", lang)
            else:
                state = t("voice_body_out", lang)
            rows.append(
                {
                    "name": b.name,
                    "state": state,
                    "in_guild": b.available and b.in_guild(guild_id),
                    "application_id": b.application_id,
                }
            )
        return rows

    def invite_links(self, guild_id: int) -> list[tuple[str, str]]:
        """(name, invite link) of every ready helper that is not in this guild yet."""
        from .permissions import helper_invite_url

        return [
            (b.name, helper_invite_url(b.application_id))
            for b in self.bodies[1:]
            if b.available and not b.in_guild(guild_id) and b.application_id
        ]
