"""Discord glue for quotations (logic in `quotation.py`).

`QuotationMixin` gives `SecretaryBot` the agent tools (`_quotation_tools`, appended to
`extra_tools` by `_agent_kwargs`, so voice has them too). Who may issue (executive or a member
with a `quote_roles` role) and who can read the channel a quotation was made in come from the bot;
nothing here is chosen by the model.
"""

from __future__ import annotations

import logging

from .pdf_report import find_fonts
from .quotation import QuoteCtx, QuoteRepo, quotation_tools

logger = logging.getLogger(__name__)


class QuotationMixin:
    def _quotation_init(self) -> None:
        self.quotes = QuoteRepo(self.docs, clock=lambda: self.s.now())

    def _quotation_tools(self, message, guild, asker, is_exec: bool, lang: str) -> list:
        """The quotation tools for one request (chat or voice). [] when the flag is off; never
        raises."""
        try:
            if not self.team.enabled(guild.id, "quotations"):
                return []
            roles = sorted(self.team.quote_roles(guild.id))
            mine = {getattr(r, "name", "") for r in getattr(asker, "roles", [])}
            ctx = QuoteCtx(
                guild_id=guild.id,
                guild_name=getattr(guild, "name", "") or "",
                asker_id=asker.id,
                asker_name=getattr(asker, "display_name", str(asker.id)),
                is_exec=is_exec,
                can_issue=bool(is_exec or (set(roles) & mine)),
                quote_roles=roles,
                channel_id=message.channel.id,
                repo=self.quotes,
                company=self.company,
                lang=lang,
                now=self.s.now,
                channel_readable=lambda cid: self._quotation_can_read(guild, asker, cid),
                fonts=lambda: find_fonts(self.s.pdf_font),
            )
            return quotation_tools(ctx)
        except Exception:
            logger.warning("quotation tools unavailable", exc_info=True)
            return []

    def _quotation_can_read(self, guild, asker, channel_id: int) -> bool:
        """The asker can read the channel the quotation was made in (live, never cached)."""
        ch = guild.get_channel(channel_id)
        return ch is not None and asker.id in self._channel_viewers(guild, ch, fresh=True)
