"""Discord glue of `create_image` (logic in `imagegen.py`).

`ImageMixin` gives `SecretaryBot` the `create_image` agent tool (`_image_tools`, appended to
`extra_tools` by `_agent_kwargs`). It is offered for chat answers with or without a photo (never
in a voice room, see `spoken`; interjections never get it), when the guild's `images` flag is on
and Thai text can be shaped. Guild, channel and author are bound here, never model input. Both
daily caps (per guild, and per person in that guild) are counted in the DocStore
(`StoredDailyBudget`: kind `salid_media_usage`, key = the day), so a restart keeps them; the
revision memory lives in this process (one bot process).
"""

from __future__ import annotations

import logging

from . import cards
from .imagegen import OUTBOX_MAX_BYTES, ImageCtx, ImageMemory, image_tools
from .promo import PHOTO_MIMES
from .videoad import MediaUsage

logger = logging.getLogger(__name__)

_warned = False  # "libfribidi missing" is logged once per process
GUILD_FIELD = "images"  # the guild's images today
USER_FIELD = "images_user_{}"  # one person's images today in that guild


def spoken(message, files) -> bool:
    """True for an answer in a voice room: older voice code passes no file list (`files` None),
    newer code passes `files=[]` with a room stand-in (`bot.room_message`), which has no
    `reply`. A Discord message always has one."""
    return files is None or not callable(getattr(message, "reply", None))


class StoredDailyBudget:
    """`media.DailyBudget`'s interface (`cap`, `used`, `add`) over DocStore kind
    `salid_media_usage`, key = the day (YYYY-MM-DD), so a restart keeps the count. The guild's
    budget is keyed by guild (`used(guild_id, day)`, field `images`); a per-person one
    (`per_user`) is bound to a guild with `for_guild` and keyed by person (`used(user_id, day)`,
    field `images_user_<id>` in that guild's document). Callers check and add under one lock
    (`imagegen.ImageCtx.make`); `add` itself is atomic in this process."""

    def __init__(
        self, usage: MediaUsage, cap: float, *, per_user: bool = False, guild_id: int | None = None
    ):
        self.usage, self.cap = usage, max(0.0, float(cap))
        self.per_user, self.guild_id = per_user, guild_id

    def for_guild(self, guild_id: int) -> StoredDailyBudget:
        """This per-person budget for the people of `guild_id`."""
        return StoredDailyBudget(self.usage, self.cap, per_user=True, guild_id=guild_id)

    def _where(self, key: int) -> tuple[int, str]:
        if not self.per_user:
            return key, GUILD_FIELD
        if self.guild_id is None:
            raise ValueError("a per-person image budget needs for_guild(guild_id)")
        return self.guild_id, USER_FIELD.format(key)

    def used(self, key: int, day: str) -> float:
        gid, name = self._where(key)
        return self.usage.used(gid, day, name)

    def available(self, key: int, day: str) -> bool:
        return self.used(key, day) < self.cap

    def add(self, key: int, day: str, amount: float) -> None:
        gid, name = self._where(key)
        self.usage.add(gid, day, name, max(0.0, amount))


class ImageMixin:
    """Needs `self.s`, `self.llm`, `self.team` and `self.docs`: call `_init_images()` once after
    `self.docs` exists."""

    def _init_images(self) -> None:
        usage = MediaUsage(self.docs)
        self.image_budget = StoredDailyBudget(usage, self.s.images_per_day)  # per guild
        # per person in a guild, bound to the request's guild (`for_guild`)
        self.image_user_budget = StoredDailyBudget(usage, self.s.images_per_user_day, per_user=True)
        self.image_memory = ImageMemory()

    def _image_tools(self, message, guild, asker, files) -> list:
        """`create_image` for this message, or [] (flag off, no Thai shaping, or a voice room).
        Never raises."""
        global _warned
        try:
            if spoken(message, files) or not self.team.enabled(guild.id, "images"):
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
                    user_budget=self.image_user_budget.for_guild(guild.id),
                    fonts=cards.card_fonts(),
                    # the upload limit where the reply goes (discord.py: 10 MiB unless boosted;
                    # a DM about this server has no boost) minus 1 MiB
                    max_bytes=min(
                        OUTBOX_MAX_BYTES,
                        int(getattr(message.guild, "filesize_limit", 10 * 2**20)) - 2**20,
                    ),
                )
            )
        except Exception:
            logger.warning("image tool unavailable", exc_info=True)
            return []
