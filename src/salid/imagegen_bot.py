"""Discord glue of `create_image` (logic in `imagegen.py`).

`ImageMixin` gives `SecretaryBot` the `create_image` agent tool (`_image_tools`, appended to
`extra_tools` by `_agent_kwargs`). It is offered for chat answers with or without a photo (never
in a voice room: `_agent_kwargs(spoken=True)`, with `looks_spoken` as the fallback; interjections
never get it), when the guild's `images` flag is on and Thai text can be shaped. Guild, channel
and author are bound here, never model input. Both daily caps (per guild, and per person in that
guild) are counted in the DocStore (`StoredDailyBudget`: kind `salid_media_usage`, key = the day),
so a restart keeps them; one request reads that day document once before the take and changes
both counts in one read-modify-write (`StoredPair`). The revision memory lives in this process
(one bot process).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from . import cards
from .imagegen import OUTBOX_MAX_BYTES, ImageCtx, ImageMemory, image_tools
from .promo import usable_photos
from .videoad import MediaUsage

logger = logging.getLogger(__name__)

_warned = False  # "libfribidi missing" is logged once per process
GUILD_FIELD = "images"  # the guild's images today
USER_FIELD = "{}_user_{}"  # one person's units today in that guild: images_user_<id>


def looks_spoken(message, files) -> bool:
    """True for an answer in a voice room (the fallback when a caller does not pass
    `_agent_kwargs(spoken=True)`): older voice code passes no file list (`files` None), newer
    code passes `files=[]` with a room stand-in (`bot.room_message`), which has no `reply`. A
    Discord message always has one."""
    return files is None or not callable(getattr(message, "reply", None))


class StoredDailyBudget:
    """`media.DailyBudget`'s interface (`cap`, `used`, `add`, `give_back`) plus an atomic `take`,
    over DocStore kind `salid_media_usage`, key = the day (YYYY-MM-DD), so a restart keeps the
    count. A guild's budget is keyed by guild (`used(guild_id, day)`, field `field`: `images`,
    `promo_kits`); a per-person one (`per_user`) is bound to a guild with `for_guild` and keyed by
    person (`used(user_id, day)`, field `<field>_user_<id>` in that guild's document). `pair`
    joins a guild budget and a per-person one of the same guild (`StoredPair`). Every change is
    atomic in this process (`MediaUsage`)."""

    def __init__(
        self,
        usage: MediaUsage,
        cap: float,
        *,
        field: str = GUILD_FIELD,
        per_user: bool = False,
        guild_id: int | None = None,
    ):
        self.usage, self.cap, self.field = usage, max(0.0, float(cap)), field
        self.per_user, self.guild_id = per_user, guild_id

    def for_guild(self, guild_id: int) -> StoredDailyBudget:
        """This per-person budget for the people of `guild_id`."""
        return StoredDailyBudget(
            self.usage, self.cap, field=self.field, per_user=True, guild_id=guild_id
        )

    def where(self, key: int) -> tuple[int, str]:
        """(the guild whose document holds the count, the field) of `key`."""
        if not self.per_user:
            return key, self.field
        if self.guild_id is None:
            raise ValueError("a per-person budget needs for_guild(guild_id)")
        return self.guild_id, USER_FIELD.format(self.field, key)

    def used(self, key: int, day: str) -> float:
        gid, name = self.where(key)
        return self.usage.used(gid, day, name)

    def add(self, key: int, day: str, amount: float) -> None:
        gid, name = self.where(key)
        self.usage.add(gid, day, name, max(0.0, amount))

    def take(self, key: int, day: str, amount: float) -> bool:
        """Add `amount` unless that would pass `cap` (False, nothing written, then)."""
        gid, name = self.where(key)
        return self.usage.take(gid, day, name, max(0.0, amount), self.cap)

    def give_back(self, key: int, day: str, amount: float) -> None:
        gid, name = self.where(key)
        self.usage.give_back(gid, day, name, amount)

    def pair(self, person: object, guild_id: int) -> StoredPair | None:
        """This guild budget and `person` (a per-person budget of `guild_id` in the same store)
        as one `StoredPair`; None when they do not share the guild's day document."""
        if (
            not self.per_user
            and isinstance(person, StoredDailyBudget)
            and person.per_user
            and person.guild_id == guild_id
            and person.usage is self.usage
        ):
            return StoredPair(self, person)
        return None


@dataclass(frozen=True)
class StoredPair:
    """A guild's budget and one of its per-person budgets, both in the guild's day document:
    both counts come from one read, and both are taken (or given back) in one read-modify-write
    (`imagegen.ImageCtx` uses it through its budget interface)."""

    guild: StoredDailyBudget
    person: StoredDailyBudget

    def _fields(self, guild_id: int, user_id: int) -> tuple[str, str]:
        return self.guild.where(guild_id)[1], self.person.where(user_id)[1]

    def counts(self, guild_id: int, user_id: int, day: str) -> tuple[float, float]:
        """(the guild's units today, the person's units today): one read."""
        g, u = self._fields(guild_id, user_id)
        doc = self.guild.usage.doc(guild_id, day)
        return float(doc.get(g) or 0), float(doc.get(u) or 0)

    def take(self, guild_id: int, user_id: int, day: str, amount: int) -> tuple[int, int]:
        """Take `amount` units from both, trimmed to what both have left: (taken, left after)."""
        g, u = self._fields(guild_id, user_id)
        caps = {g: self.guild.cap, u: self.person.cap}
        n, doc = self.guild.usage.take_upto(guild_id, day, caps, amount)
        left = min(cap - float(doc.get(f) or 0) for f, cap in caps.items())
        return n, max(0, int(left))

    def give_back(self, guild_id: int, user_id: int, day: str, amount: int) -> None:
        self.guild.usage.give_back(guild_id, day, list(self._fields(guild_id, user_id)), amount)


class ImageMixin:
    """Needs `self.s`, `self.llm`, `self.team` and `self.docs`: call `_init_images()` once after
    `self.docs` exists."""

    def _init_images(self) -> None:
        usage = MediaUsage(self.docs)
        self.image_budget = StoredDailyBudget(usage, self.s.images_per_day)  # per guild
        # per person in a guild, bound to the request's guild (`for_guild`)
        self.image_user_budget = StoredDailyBudget(usage, self.s.images_per_user_day, per_user=True)
        self.image_memory = ImageMemory()

    def _image_tools(self, message, guild, asker, files, *, spoken: bool = False) -> list:
        """`create_image` for this message, or [] (flag off, no Thai shaping, or a voice room:
        `spoken`, or `looks_spoken` for callers that do not say). Never raises."""
        global _warned
        try:
            if spoken or looks_spoken(message, files) or not self.team.enabled(guild.id, "images"):
                return []
            if not cards.thai_ready():
                if not _warned:
                    logger.error("libfribidi missing: create_image off")
                    _warned = True
                return []
            return image_tools(
                ImageCtx(
                    guild_id=guild.id,
                    user_id=asker.id,
                    day=self.s.today().isoformat(),
                    photos=usable_photos(files),
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
