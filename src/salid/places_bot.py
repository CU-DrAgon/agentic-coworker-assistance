"""Discord glue for remembered places (logic and access rules live in `places.py`).

`PlacesMixin` gives `SecretaryBot`: the agent tools (`_places_tools`, appended to `extra_tools` by
`_agent_kwargs`, so voice has them too), the learner that reuses the memory ingest hook (one
Flash-Lite call, only on messages that look like they hold an address), the one-time DM offer to
remember someone's own home, and the scheduling lookup.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from functools import partial

import discord

from . import audience as aud
from .i18n import t
from .naming import mentions_name
from .places import (
    PlaceRepo,
    PlacesCtx,
    build_place_tools,
    extract_candidates,
    learn,
    looks_like_place,
    place_for_event,
)

logger = logging.getLogger(__name__)


class PlacesMixin:
    def _places_init(self) -> None:
        self.places = PlaceRepo(self.docs)

    def _places_on(self, guild_id: int) -> bool:
        return self.team.enabled(guild_id, "places")

    # ------------------------------------------------------------- access adapter
    def _places_ctx(
        self,
        guild,
        asker,
        is_exec: bool,
        *,
        channel,
        room: frozenset[int] | None = None,
        in_guild: bool = True,
        request_id: str = "",
        message_id: int = 0,
        link: str = "",
    ) -> PlacesCtx:
        """Who is asking and who will hear the answer. `channel`: the channel the answer goes to
        (None when it cannot be told: nobody then counts as allowed)."""

        def viewers_of(cid: int) -> frozenset[int]:
            ch = guild.get_channel(cid)
            return self._channel_viewers(guild, ch) if ch is not None else frozenset()

        parent = getattr(channel, "parent_id", None) if channel is not None else None
        cid = parent or getattr(channel, "id", None)
        if room is not None:
            audience: frozenset[int] | None = frozenset(room) | {asker.id}
        elif not in_guild:
            audience = None  # a DM: the asker alone
        elif channel is None or guild.get_channel(cid) is None:
            audience = frozenset({-1})  # unknown audience: nobody qualifies
        else:
            audience = viewers_of(cid) | {asker.id}
        rid = request_id or uuid.uuid4().hex
        return PlacesCtx(
            guild_id=guild.id,
            asker_id=asker.id,
            is_exec=is_exec,
            repo=self.places,
            ranks=self._ranks(guild),
            viewers_of=viewers_of,
            audience=audience,
            channel_id=cid if in_guild and room is None else None,
            people=self._people(guild),
            channels={
                c.id: c.name
                for c in self._readable_channels(guild)
                if self.team.channel_mode(guild.id, c.id, c.name, self.s.ask_channel_ids) != "off"
            },
            confirm=lambda key: self.secretary_state.confirm(key, rid),
            room=frozenset(room) if room is not None else None,
            message_id=message_id,
            message_link=link,
            now=self.s.now,
            shared=lambda: self.sharing.public_places(guild.id, self.places),
        )

    @staticmethod
    def _readable_channels(guild) -> list:
        out: list = []
        for attr in ("text_channels", "voice_channels", "stage_channels"):
            out += list(getattr(guild, attr, None) or [])
        return out

    def _places_tools(self, message, guild, asker, is_exec: bool) -> list:
        """The place tools for one request (chat message or `room_message` stand-in). Never
        raises: without them Salid just cannot talk about places."""
        try:
            if not self._places_on(guild.id):
                return []
            ctx = self._places_ctx(
                guild,
                asker,
                is_exec,
                channel=message.channel,
                room=getattr(message, "room", None),
                in_guild=message.guild is not None,
                message_id=getattr(message, "id", 0) or 0,
                link=getattr(message, "jump_url", "") or "",
            )
            return build_place_tools(ctx)
        except Exception:
            logger.warning("place tools unavailable", exc_info=True)
            return []

    def _places_resolver(self, guild, asker, is_exec: bool):
        """For `schedule_meeting(place=...)`: (query, invitee ids) -> place_for_event result."""
        try:
            if not self._places_on(guild.id):
                return None
            ctx = self._places_ctx(guild, asker, is_exec, channel=None, in_guild=False)
            return partial(place_for_event, ctx)
        except Exception:
            logger.warning("place lookup unavailable", exc_info=True)
            return None

    # ------------------------------------------------------------- learning (memory ingest hook)
    def _places_hook(self, message, cid: int) -> None:
        """Called next to the memory record of a human message. Free regex prefilter first."""
        try:
            guild = message.guild
            if guild is None or getattr(message.author, "bot", False):
                return
            if not looks_like_place(message.content) or not self._places_on(guild.id):
                return
            task = asyncio.create_task(self._places_learn(message, cid))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
        except Exception:
            logger.warning("places hook failed", exc_info=True)

    async def _places_learn(self, message, cid: int) -> None:
        try:
            text = message.content
            cands = await asyncio.to_thread(
                extract_candidates,
                self.llm,
                self.s.gate_model,
                text,
                getattr(message.author, "display_name", ""),
            )
            if not cands:
                return
            addressed = mentions_name(text) or any(
                getattr(m, "id", None) == getattr(self.user, "id", 0) for m in message.mentions
            )
            res = await asyncio.to_thread(
                learn,
                self.places,
                cands,
                guild_id=message.guild.id,
                author_id=message.author.id,
                channel_id=cid,
                message_id=message.id,
                link=getattr(message, "jump_url", "") or "",
                at=message.created_at,
                addressed=addressed,
            )
            logger.info(
                "places: %d new, %d widened, %d offers (guild=%s)",
                len(res.created),
                len(res.widened),
                len(res.offers),
                message.guild.id,
            )
            gid = message.guild.id
            langs = self._deliver_langs(gid, message.author.id, text=text)
            for c in res.offers:  # asked once; nothing is stored until the person says yes
                with contextlib.suppress(discord.HTTPException):
                    await message.author.send(
                        aud.render(
                            lambda lang, c=c: t(
                                "place_offer_dm",
                                lang,
                                name=c.name or ("Home" if lang == "en" else "บ้าน"),
                                address=c.address,
                            ),
                            langs,
                        )
                    )
        except Exception:
            logger.warning("places learn failed", exc_info=True)

    async def _places_source_deleted(self, guild_id: int, message_ids: set[int]) -> None:
        """A deleted source message does not delete the place; we only log it."""
        try:
            if not self._places_on(guild_id):
                return
            for p in await asyncio.to_thread(self.places.list, guild_id):
                if any(s.get("message_id") in message_ids for s in p.sources):
                    logger.info("place %s: a source message was deleted; the place is kept", p.id)
        except Exception:
            logger.warning("places delete check failed", exc_info=True)
