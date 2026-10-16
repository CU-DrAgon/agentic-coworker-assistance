"""Discord glue for the market tools (logic in `markets.py`).

`MarketsMixin` gives `SecretaryBot`: the agent tools (`_markets_tools`, appended to `extra_tools`
by `_agent_kwargs`, so voice has them too), the alert poller driven by the bot tick (DMs through
`dm.send_dm`), the opt-in market line of the morning brief and the company-news section of the
evening report.
"""

from __future__ import annotations

import asyncio
import logging

from . import markets as mk
from .dm import send_dm
from .i18n import resolve_lang

logger = logging.getLogger(__name__)

POLL_TICK_S = 60.0  # how often the watchlists are looked at (each market has its own interval)


class MarketsMixin:
    def _markets_init(self) -> None:
        grounded = mk.Grounded(self.llm, self.s.gemini_model) if self.llm is not None else None
        self.market_data = mk.MarketData(grounded=grounded)  # Yahoo, then grounded search
        self.market_watches = mk.WatchRepo(self.docs)
        self.market_poller = mk.Poller(self.market_watches, self.market_data, self.docs)
        self._markets_last = -1e9

    def _markets_on(self, guild_id: int) -> bool:
        return self.team.enabled(guild_id, "markets")

    def _markets_tools(self, guild, asker, is_exec: bool, lang: str) -> list:
        """The market tools for one request (chat or voice). [] when the flag is off; never
        raises."""
        try:
            if not self._markets_on(guild.id):
                return []
            ctx = mk.MarketsCtx(
                guild_id=guild.id,
                asker_id=asker.id,
                is_exec=is_exec,
                data=self.market_data,
                repo=self.market_watches,
                docs=self.docs,
                prefs=self.prefs,
                lang=lang,
                now=self.s.now,
                on_prefs_changed=lambda: self._prefs_cache.pop(guild.id, None),
            )
            return mk.markets_tools(ctx)
        except Exception:
            logger.warning("market tools unavailable", exc_info=True)
            return []

    # ------------------------------------------------------------- alerts (bot tick)
    async def _markets_due(self, now=None) -> int:
        """Evaluate the watchlists (at most once a minute) and DM the people whose condition
        fired. Returns DMs sent."""
        if self._clock() - self._markets_last < POLL_TICK_S or not self.guilds:
            return 0
        self._markets_last = self._clock()
        try:
            gids = [g.id for g in self.guilds if self._markets_on(g.id)]
            if not gids:
                return 0
            alerts = await asyncio.to_thread(self.market_poller.due, gids, now or self.s.now())
        except Exception:
            logger.exception("market poll failed")
            return 0
        sent = 0
        for a in alerts:
            guild = self.get_guild(a.guild_id)
            member = guild.get_member(a.user_id) if guild is not None else None
            if member is None:
                continue
            lang = self._profile_of(a.guild_id, a.user_id).language or self._guild_lang(a.guild_id)
            res = await send_dm(member, mk.format_alert(a, resolve_lang(lang)))
            sent += int(res.ok)
        return sent

    # ------------------------------------------------------------- brief / report
    def _market_brief_line(self, guild, member, prefs, now) -> str:
        """The member's opt-in "ตลาดเช้านี้" line for the morning brief. Blocking."""
        if not (prefs.markets and self._markets_on(guild.id)):
            return ""
        try:
            lang = self._profile_of(guild.id, member.id).language or self._guild_lang(guild.id)
            return mk.market_line(
                self.market_data, prefs.market_items, resolve_lang(lang), now.date()
            )
        except Exception:
            logger.warning("market line failed", exc_info=True)
            return ""

    def _company_news_section(self, guild, lang: str) -> str:
        """Up to 3 labelled headlines on the topics executives asked to follow, for the evening
        report. "" when there are none. Blocking."""
        if not self._markets_on(guild.id):
            return ""
        try:
            topics = mk.topic_list(self.docs, guild.id)
            return mk.company_news(self.market_data, topics, lang) if topics else ""
        except Exception:
            logger.warning("company news failed", exc_info=True)
            return ""
