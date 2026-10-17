"""Discord glue of the meeting minutes document (logic in `meeting_doc.py`).

`MeetingDocMixin` gives `SecretaryBot` the `meeting_document` agent tool (`_meeting_doc_tools`,
appended to `extra_tools` by `_agent_kwargs`, so voice has it too). The transcript is rebuilt from
the memory log (kind "voice") so retention, deletion and the not-recording rules apply on their own.
Who may receive the file is decided here, in code: attended / can see the room, and a shared
channel only gets the file when every reader of it can see the room, else it goes to the asker
privately.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
from datetime import datetime, timedelta

import discord

from .coo_store import MINUTES
from .i18n import LANGS
from .meeting_doc import Line, MeetingDocCtx, meeting_document_tools
from .minutes import can_see
from .pdf_report import find_fonts

logger = logging.getLogger(__name__)

END_GRACE = timedelta(seconds=120)  # the last segments are transcribed a little after the end
MAX_LINES = 5000


class MeetingDocMixin:
    def _meeting_doc_tools(self, message, guild, asker, lang: str, loop) -> list:
        """The `meeting_document` tool for this asker, or [] when the flag is off. Never raises."""
        try:
            if not self.team.enabled(guild.id, "meeting_transcript_doc"):
                return []
            return meeting_document_tools(self._meeting_doc_ctx(message, guild, asker, lang, loop))
        except Exception:
            logger.warning("meeting document tool unavailable", exc_info=True)
            return []

    def _meeting_doc_ctx(self, message, guild, asker, lang: str, loop) -> MeetingDocCtx:
        gid = guild.id

        def can_view_voice(cid: int) -> bool:
            ch = guild.get_channel(cid)
            return ch is not None and can_see(ch, asker)

        def send_private(name: str, data: bytes, caption: str) -> bool:
            fut = asyncio.run_coroutine_threadsafe(
                self.deliver_to_person(
                    guild, asker, caption, [discord.File(io.BytesIO(data), filename=name)]
                ),
                loop,
            )
            res = fut.result(timeout=60)
            return bool(getattr(res, "ok", False))

        return MeetingDocCtx(
            asker_id=asker.id,
            lang=lang if lang in LANGS and lang != "auto" else "th",
            org=getattr(guild, "name", "") or "",
            records=lambda: self.docs.find(MINUTES, gid),
            can_view_voice=can_view_voice,
            lines=lambda rec: self._meeting_lines(gid, rec),
            audience_ok=lambda rec: self._meeting_audience_ok(message, guild, asker, rec),
            send_private=send_private,
            fonts=lambda: find_fonts(self.s.pdf_font),
            now=self.s.now,
            client=self.llm,
            model=self.s.gate_model,
        )

    def _meeting_lines(self, guild_id: int, rec: dict) -> list[Line]:
        """The voice lines of a finished meeting from the memory log (empty when memory is off, the
        lines expired, or were deleted). Lines before a "not recording" pause are never in it."""
        mem = getattr(self, "memory_store", None)
        if mem is None or not self.team.enabled(guild_id, "memory"):
            return []
        vid = int(rec["voice_channel_id"])
        start = datetime.fromisoformat(str(rec.get("transcript_from") or rec["started_at"]))
        end = datetime.fromisoformat(str(rec["ended_at"])) + END_GRACE
        zone = self.s.now().tzinfo
        msgs = mem.messages(
            guild_id, [vid], start.astimezone(zone).date().isoformat(),
            end.astimezone(zone).date().isoformat(), channel_id=vid, limit=MAX_LINES,
            newest_first=False,
        )  # fmt: skip
        return [
            (m.created_at, m.author_name, m.text)
            for m in msgs
            if m.kind == "voice" and start <= m.created_at <= end
        ]

    def _meeting_audience_ok(self, message, guild, asker, rec: dict) -> bool:
        """May the file be posted where this conversation happens? Yes in a DM or the asker's own
        private Salid channel; in a guild channel only when everyone who can read it can also see
        the meeting's voice room. Never in a voice room (several listeners, no reply message)."""
        if getattr(message, "room", None) is not None:
            return False
        if getattr(message, "guild", None) is None:
            return True
        channel = message.channel
        with contextlib.suppress(Exception):
            if self.private_chats.owner_of(guild.id, channel.id) == asker.id:
                return True
        voice = guild.get_channel(int(rec["voice_channel_id"]))
        if voice is None:
            return False
        try:
            members = list(getattr(channel, "members", None) or [])
        except Exception:
            return False
        if not members:
            return False
        return all(can_see(voice, m) for m in members if not getattr(m, "bot", False))
