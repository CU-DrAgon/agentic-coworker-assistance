"""Discord glue of promo post kits (logic in `promo.py`).

`PromoMixin` gives `SecretaryBot` the `make_promo_kit` agent tool (`_promo_tools`, appended to
`extra_tools` by `_agent_kwargs`). It is offered for answers that carry a usable photo or have
a remembered kit for the same person in the same channel, so ordinary chats pay nothing for its
description. Gated on `files is None` only: voice passes `files=[]` on purpose, so a remembered
kit can be revised by voice.
Guild, channel and author are bound here, never model input. The daily cap is counted in the
DocStore (`imagegen_bot.StoredDailyBudget`, kind `salid_media_usage`, key = the day, field
`promo_kits`), so a restart keeps it; the revision memory lives in this process (one bot
process).
"""

from __future__ import annotations

import logging

from . import cards
from .imagegen_bot import StoredDailyBudget
from .promo import PromoCtx, PromoMemory, promo_tools, usable_photos
from .videoad import MediaUsage

logger = logging.getLogger(__name__)

_warned = False  # "libfribidi missing" is logged once per process
PROMO_FIELD = "promo_kits"  # the guild's AI scenes today, in the day's usage document


class PromoMixin:
    """Needs `self.s`, `self.llm`, `self.team` and `self.docs`: call `_init_promo()` once after
    `self.docs` exists."""

    def _init_promo(self) -> None:
        self.promo_budget = StoredDailyBudget(
            MediaUsage(self.docs), self.s.promo_kits_per_day, field=PROMO_FIELD
        )
        self.promo_memory = PromoMemory()

    def _promo_tools(self, message, guild, asker, files) -> list:
        """`make_promo_kit` for this message, or [] (flag off, no Thai shaping, no file list, or
        no photo and no remembered kit). Never raises."""
        global _warned
        try:
            if files is None or not self.team.enabled(guild.id, "promo"):
                return []
            if not cards.thai_ready():
                if not _warned:
                    logger.error("libfribidi missing: promo kits off")
                    _warned = True
                return []
            photos = usable_photos(files)
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
                    fonts=cards.card_fonts(self._lang(message)),
                )
            )
        except Exception:
            logger.warning("promo tool unavailable", exc_info=True)
            return []
