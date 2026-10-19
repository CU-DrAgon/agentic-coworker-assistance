"""Discord glue of the concept studio (logic in `concept.py`).

`ConceptMixin` gives `SecretaryBot` the `concept_image` and `price_check` agent tools
(`_concept_tools`, appended to `extra_tools` by `_agent_kwargs`). Needs the `concept` and `ideas`
flags (the work lands in the idea document); `price_check` also needs `web`. Voice gets
`price_check` only: an image cannot be spoken. Guild, channel and author are bound here, never
model input. The daily cap is kept in the DocStore (field `concept_images`, `_init_concept` after `self.docs`); the revision memory lives in this process.
"""

from __future__ import annotations

import logging

from . import cards
from .concept import ConceptCtx, ConceptMemory, concept_tools
from .imagegen_bot import StoredDailyBudget
from .promo import PHOTO_MIMES
from .videoad import MediaUsage

logger = logging.getLogger(__name__)


class ConceptMixin:
    def _init_concept(self) -> None:
        self.concept_budget = StoredDailyBudget(
            MediaUsage(self.docs), self.s.concept_images_per_day, field="concept_images"
        )
        self.concept_memory = ConceptMemory()

    def _concept_tools(self, message, guild, asker, files, lang: str) -> list:
        """The concept tools for one request, or [] (flag off, no `ideas`, no model). Never raises."""
        try:
            if self.llm is None or not (
                self.team.enabled(guild.id, "concept") and self.team.enabled(guild.id, "ideas")
            ):
                return []
            voice = hasattr(message, "room")  # `room_message` stand-in of a voice meeting
            photos = [
                (f.index, f.data)
                for f in (files or [])
                if f.mime in PHOTO_MIMES and f.data and not f.skip
            ]
            ctx = ConceptCtx(
                guild_id=guild.id,
                day=self.s.today().isoformat(),
                lang=lang,
                asker_id=asker.id,
                asker_name=getattr(asker, "display_name", str(asker.id)),
                key=(guild.id, message.channel.id, asker.id),
                image_client=self.llm,
                image_model=self.s.promo_image_model,
                image_size=self.s.concept_image_size,
                budget=self.concept_budget,
                memory=self.concept_memory,
                repo=self.ideas,
                photos=photos,
                fonts=cards.card_fonts(lang),  # CJK / Devanagari labels need their script font
                web_client=self.llm if self.team.enabled(guild.id, "web") else None,
                web_model=self.s.gemini_model,
            )
            tools = concept_tools(ctx, images=not voice)
            if not self.team.enabled(guild.id, "web"):  # a price check needs the web
                tools = [f for f in tools if f.__name__ != "price_check"]
            return tools
        except Exception:
            logger.warning("concept tools unavailable", exc_info=True)
            return []
