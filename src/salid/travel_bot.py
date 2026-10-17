"""Discord glue for currency conversion and flights (logic: `currency.py`, `flights.py`).

`TravelMixin` gives `SecretaryBot` the agent tools (`_travel_tools`, appended to `extra_tools` by
`_agent_kwargs`, so voice has them too) and the flight-tracking poll (`_travel_due`, from `tick`),
which DMs only the person who asked for the tracking.
"""

from __future__ import annotations

import asyncio
import logging
import time

from .currency import Converter, currency_tools
from .dm import send_dm
from .flights import (
    FlightService,
    FlightTracker,
    flight_tools,
    select_provider,
)
from .i18n import resolve_lang, t

logger = logging.getLogger(__name__)
TRAVEL_TICK_SECONDS = 60.0  # how often tracks are looked at (each is polled every 15 minutes)


class TravelMixin:
    def _travel_init(self) -> None:
        provider = select_provider(self.s.aerodatabox_key, self.s.aviationstack_key)
        self.converter = Converter()
        self.flight_service = FlightService(
            self.llm, self.s.gemini_model, provider, today=lambda: self.s.now().date()
        )
        self.flight_tracker = FlightTracker(self.docs, provider)
        self._travel_last = 0.0

    def _travel_on(self, guild_id: int) -> bool:
        return self.team.enabled(guild_id, "travel")

    def _travel_tools(self, guild, asker, lang: str) -> list:
        """convert_currency, flight_status, flight_prices, track_flight for one request.
        Never raises: without them Salid just cannot do these."""
        try:
            if not self._travel_on(guild.id):
                return []
            return [
                *currency_tools(self.converter, lang),
                *flight_tools(
                    self.flight_service,
                    self.flight_tracker,
                    guild.id,
                    asker.id,
                    today=lambda: self.s.now().date(),
                    lang=lang,
                ),
            ]
        except Exception:
            logger.warning("travel tools unavailable", exc_info=True)
            return []

    async def _travel_due(self) -> None:
        """Poll the due flight tracks (only with a provider key) and DM the requester."""
        if self.flight_tracker.provider is None or not self.guilds:
            return
        now = time.monotonic()
        if now - self._travel_last < TRAVEL_TICK_SECONDS:
            return
        self._travel_last = now
        for guild in list(self.guilds):
            if not self._travel_on(guild.id):
                continue
            for doc in await asyncio.to_thread(self.flight_tracker.due, guild.id):
                events = await asyncio.to_thread(self.flight_tracker.poll, doc)
                member = guild.get_member(doc["user_id"])
                if member is None or not events:
                    continue
                lang = resolve_lang(self.team.current(guild.id).language_setting)
                for ev in events:
                    text = t(
                        f"flight_{ev.kind}_dm",
                        lang,
                        flight=doc["display"],
                        date=doc["date"],
                        minutes=ev.minutes,
                        time=ev.time or "-",
                        gate=ev.gate,
                    )
                    await send_dm(member, text)
