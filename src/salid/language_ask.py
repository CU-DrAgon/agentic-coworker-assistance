"""The main-language question: each server says once which language Salid should use.

Until a server has chosen (`GuildSettings.language_chosen`), Salid runs on `auto` (the words the
person writes, English when there are none). When she joins a server, or at startup in a server
that never answered, she posts ONE message with a persistent select menu in the Salid chat (or
system) channel. Only server managers and executives may pick; the pick is saved like
"@Salid speak X everywhere" (`_apply_server_language`), the menu message turns into a confirmation
in the chosen language and the chat channel Salid created is renamed to match. The question itself
is hard-coded bilingual (English + Thai) because no language is known yet. `language_asked_at`
makes sure it is never posted twice.

`LanguageSelect` is a `discord.ui.DynamicItem` like `DisclosureButton`, so the menu keeps working
after a restart; `LanguageAskMixin` is the `SecretaryBot` glue.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import discord

from .hints import with_hint
from .i18n import LANG_NAMES, LANGS, LANGUAGE_LABELS, t, t_days
from .modes import CHAT_CHANNEL_NAME, CHAT_CHANNEL_NAME_EN, chat_channel_name, is_chat_channel_name

logger = logging.getLogger(__name__)

CUSTOM_ID = "salid:lang:ask"
QUESTION = (
    "Which language should I use as this server's main language?\n"
    "ใช้ภาษาอะไรเป็นภาษาหลักของเซิร์ฟเวอร์นี้ดีคะ"
)
PLACEHOLDER = "Choose a language · เลือกภาษา"
# English first (the default), then Thai, then the second tier.
OPTION_ORDER = ("en", "th", *(k for k in LANG_NAMES if k not in ("en", "th")))
# Names of the chat channel Salid creates itself; a reused channel has any other name.
OWN_CHAT_NAMES = {CHAT_CHANNEL_NAME, CHAT_CHANNEL_NAME_EN}


def both_hint(key: str) -> str:
    """A hint in English and Thai, each on its own ℹ️ line (the language is not known yet)."""
    return "\n".join(with_hint("", key, lang=lang).strip() for lang in ("en", "th"))


QUESTION_FULL = QUESTION + "\n" + both_hint("lang_how")


def both(key: str) -> str:
    """A catalog line in English and Thai (for people whose language is not known yet)."""
    return f"{t(key, 'en')}\n{t(key, 'th')}"


class LanguageSelect(discord.ui.DynamicItem[discord.ui.Select], template=r"salid:lang:ask"):
    """The language menu. The custom id is fixed and the bot registers the class once, so the
    menu keeps working after a restart; the guild comes from the interaction."""

    def __init__(self) -> None:
        super().__init__(
            discord.ui.Select(
                custom_id=CUSTOM_ID,
                placeholder=PLACEHOLDER,
                min_values=1,
                max_values=1,
                options=[
                    discord.SelectOption(label=LANGUAGE_LABELS[k][:100], value=k)
                    for k in OPTION_ORDER
                ],
            )
        )

    @classmethod
    async def from_custom_id(cls, interaction, item, match, /):
        return cls()

    async def callback(self, interaction: discord.Interaction) -> None:
        values = (getattr(interaction, "data", None) or {}).get("values") or []
        await interaction.client.handle_language_pick(interaction, values[0] if values else "")


def ask_view() -> discord.ui.View:
    view = discord.ui.View(timeout=None)
    view.add_item(LanguageSelect())
    return view


class LanguageAskMixin:
    async def _post_language_question(self, guild: discord.Guild, target=None) -> bool:
        """Post the question once (False when it was asked before, was answered, or there is no
        channel Salid may write in). Records `language_asked_at` only after it was posted."""
        settings = self.team.current(guild.id)
        if settings.language_chosen or settings.language_asked_at:
            return False
        if target is None:
            target = self._language_ask_channel(guild)
        if target is None:
            return False
        try:
            await target.send(QUESTION_FULL, view=ask_view())
        except discord.HTTPException:
            logger.warning("could not ask the language in guild %s", guild.id, exc_info=True)
            return False
        try:
            await asyncio.to_thread(self.team.mark_language_asked, guild.id)
        except Exception:
            logger.exception("could not save language_asked_at for guild %s", guild.id)
        return True

    @staticmethod
    def _language_ask_channel(guild: discord.Guild):
        """Salid's chat channel, else the system channel (where she can write)."""
        me = getattr(guild, "me", None)

        def ok(c) -> bool:
            return c is not None and (me is None or c.permissions_for(me).send_messages)

        chat = next(
            (c for c in guild.text_channels if is_chat_channel_name(c.name) and ok(c)), None
        )
        if chat is not None:
            return chat
        return guild.system_channel if ok(guild.system_channel) else None

    async def _ask_unanswered_languages(self) -> None:
        """Startup: ask every server that has never been asked or answered (once, ever)."""
        for guild in list(self.guilds):
            try:
                await self._post_language_question(guild)
            except Exception:
                logger.exception("language question failed in guild %s", guild.id)

    async def handle_language_pick(self, interaction: discord.Interaction, value: str) -> None:
        """A pick in the menu: managers and executives only; saves the server language."""
        guild = interaction.guild
        user = interaction.user
        if guild is None or value not in LANGS or value == "auto":
            with contextlib.suppress(discord.HTTPException):
                await interaction.response.defer()
            return
        if not hasattr(user, "guild_permissions") or not self.is_exec_member(user):
            await interaction.response.send_message(both("lang_ask_denied"), ephemeral=True)
            return
        try:
            await self._apply_server_language(guild, value, True)
        except Exception:
            logger.exception("could not save the server language")
            await interaction.response.send_message(both("lang_failed"), ephemeral=True)
            return
        texts = list(guild.text_channels)
        own = next((c for c in texts if c.name in OWN_CHAT_NAMES), None)  # Salid created it
        chat = own or next((c for c in texts if is_chat_channel_name(c.name)), None)
        where = chat.mention if chat is not None else f"#{chat_channel_name(value)}"
        text = (
            t("lang_ask_done", value, label=t(f"lang_label_{value}", value))
            + "\n\n"
            + with_hint(
                t_days(
                    "welcome",
                    value,
                    self.team.retention_shown(guild.id),
                    mention=self.user.mention,
                    chat=where,
                ),
                "welcome_help",
                lang=value,
            )
        )
        await interaction.response.edit_message(content=text, view=None)
        if own is not None and own.name != chat_channel_name(
            value
        ):  # a reused channel keeps its name
            with contextlib.suppress(discord.HTTPException):
                await own.edit(name=chat_channel_name(value), reason="Salid main language")
