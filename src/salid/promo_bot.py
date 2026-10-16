"""Discord glue of promo post kits (logic in `promo.py`).

`PromoMixin` gives `SecretaryBot` the `make_promo_kit` agent tool (`_promo_tools`, appended to
`extra_tools` by `_agent_kwargs`). It is offered only for chat answers (voice never passes files)
that carry a usable photo or have a remembered kit for the same person in the same channel, so
ordinary chats pay nothing for its description. Guild, channel and author are bound here, never
model input. The daily cap and the revision memory live in this process (one bot process).
"""

from __future__ import annotations

import logging

from . import cards
from .media import DailyBudget
from .promo import PHOTO_MIMES, PromoCtx, PromoMemory, promo_tools

logger = logging.getLogger(__name__)

_warned = False  # "libfribidi missing" is logged once per process


class PromoMixin:
    def _init_promo(self) -> None:
        self.promo_budget = DailyBudget(self.s.promo_kits_per_day)
        self.promo_memory = PromoMemory()

    def _promo_tools(self, message, guild, asker, files) -> list:
        """`make_promo_kit` for this message, or [] (flag off, no Thai shaping, voice, or no photo
        and no remembered kit). Never raises."""
        global _warned
        try:
            if files is None or not self.team.enabled(guild.id, "promo"):
                return []
            if not cards.thai_ready():
                if not _warned:
                    logger.error("libfribidi missing: promo kits off")
                    _warned = True
                return []
            photos = [
                (f.index, f.data) for f in files if f.mime in PHOTO_MIMES and f.data and not f.skip
            ]
            key = (guild.id, message.channel.id, asker.id)
            if not photos and self.promo_memory.get(key) is None:
                return []
            return promo_tools(
                PromoCtx(
                    guild_id=guild.id,
                    day=self.s.today().isoformat(),
                    default_shop=getattr(guild, "name", "") or "",
                    photos=photos,
                    memory=self.promo_memory,
                    key=key,
                    client=self.llm,
                    model=self.s.promo_image_model,
                    budget=self.promo_budget,
                    fonts=cards.card_fonts(),
                )
            )
        except Exception:
            logger.warning("promo tool unavailable", exc_info=True)
            return []
