"""Discord glue of `create_image` (logic in `imagegen.py`).

`ImageMixin` gives `SecretaryBot` the `create_image` agent tool (`_image_tools`, appended to
`extra_tools` by `_agent_kwargs`). It is offered for chat answers with or without a photo (voice
never passes files; interjections never get it), when the guild's `images` flag is on and Thai
text can be shaped. Guild, channel and author are bound here, never model input. Both daily caps
(per guild, per person) and the revision memory live in this process (one bot process).
"""

from __future__ import annotations

import logging

from . import cards
from .imagegen import ImageCtx, ImageMemory, image_tools
from .media import DailyBudget
from .promo import PHOTO_MIMES

logger = logging.getLogger(__name__)

_warned = False  # "libfribidi missing" is logged once per process


class ImageMixin:
    def _init_images(self) -> None:
        self.image_budget = DailyBudget(self.s.images_per_day)  # keyed by guild id
        self.image_user_budget = DailyBudget(self.s.images_per_user_day)  # keyed by user id
        self.image_memory = ImageMemory()

    def _image_tools(self, message, guild, asker, files) -> list:
        """`create_image` for this message, or [] (flag off, no Thai shaping, or voice). Never
        raises."""
        global _warned
        try:
            if files is None or not self.team.enabled(guild.id, "images"):
                return []
            if not cards.thai_ready():
                if not _warned:
                    logger.error("libfribidi missing: create_image off")
                    _warned = True
                return []
            photos = [
                (f.index, f.data) for f in files if f.mime in PHOTO_MIMES and f.data and not f.skip
            ]
            return image_tools(
                ImageCtx(
                    guild_id=guild.id,
                    user_id=asker.id,
                    day=self.s.today().isoformat(),
                    photos=photos,
                    memory=self.image_memory,
                    key=(guild.id, message.channel.id, asker.id),
                    client=self.llm,
                    model=self.s.image_model,
                    guild_budget=self.image_budget,
                    user_budget=self.image_user_budget,
                    fonts=cards.card_fonts(),
                )
            )
        except Exception:
            logger.warning("image tool unavailable", exc_info=True)
            return []
