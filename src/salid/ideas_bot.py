"""Discord glue for the idea / project workspace (logic in `ideas.py`).

`IdeasMixin` gives `SecretaryBot` the agent tools (`_ideas_tools`, appended to `extra_tools` by
`_agent_kwargs`, so voice has them too). Who outranks whom, who is in the server and whether the
conversation is private all come from the bot; nothing here is chosen by the model.
"""

from __future__ import annotations

import logging

from .ideas import IdeaRepo, IdeasCtx, ideas_tools
from .memory_access import hierarchy_authors
from .pdf_report import find_fonts

logger = logging.getLogger(__name__)


class IdeasMixin:
    def _ideas_init(self) -> None:
        self.ideas = IdeaRepo(self.docs, clock=lambda: self.s.now())

    def _ideas_tools(self, message, guild, asker, is_exec: bool, lang: str) -> list:
        """The idea tools for one request (chat or voice). [] when the flag is off; never raises."""
        try:
            if not self.team.enabled(guild.id, "ideas"):
                return []
            ctx = IdeasCtx(
                guild_id=guild.id,
                guild_name=getattr(guild, "name", "") or "",
                asker_id=asker.id,
                asker_name=getattr(asker, "display_name", str(asker.id)),
                is_exec=is_exec,
                repo=self.ideas,
                client=self.llm,
                model=self.s.gemini_model,
                lang=lang,
                now=self.s.now,
                outranked=lambda: hierarchy_authors(asker.id, self._ranks(guild), None),
                private=lambda: self._ideas_private_here(message, guild, asker),
                resolve_person=lambda name: self._ideas_people(guild, name),
                fonts=lambda: find_fonts(self.s.pdf_font),
            )
            return ideas_tools(ctx)
        except Exception:
            logger.warning("idea tools unavailable", exc_info=True)
            return []

    def _ideas_private_here(self, message, guild, asker) -> bool:
        """A DM or the asker's own private Salid channel: nobody else reads the answer."""
        if getattr(message, "room", None) is not None:  # a voice room
            return False
        if getattr(message, "guild", None) is None:
            return True
        try:
            owner = self.private_chats.owner_of(guild.id, message.channel.id)
        except Exception:
            return False
        return owner == asker.id

    def _ideas_people(self, guild, name: str) -> list[tuple[int, str]]:
        """Members whose display name matches `name` (exact first, else contains)."""
        low = (name or "").strip().lstrip("@").casefold()
        people = self._people(guild)
        exact = [(i, n) for i, n in people.items() if n.casefold() == low]
        return exact or [(i, n) for i, n in people.items() if low and low in n.casefold()]
