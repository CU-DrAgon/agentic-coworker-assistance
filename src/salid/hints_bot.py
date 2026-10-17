"""Help on demand, Discord side (the words are in `hints.py`).

"@Salid ใช้งานยังไง" / "สลิด ช่วยอธิบายสัญลักษณ์" / "Salid help" / "how do I answer" answers with the
"how to answer Salid" guide, deterministically (no model call), in the channel's language.
"""

from __future__ import annotations

from .hints import help_tasks, parse_help_command


class HintsMixin:
    async def _help_command(self, message, guild=None, member=None) -> bool:
        """True when the message asked how to use Salid and the guide was sent. Open to everyone;
        `guild` / `member` are set for a private channel or DM (the person's own language)."""
        if not parse_help_command(self._command_text(message)):
            return False
        if guild is not None and member is not None:
            lang = self._person_lang(guild.id, member.id, message.content or "")
        else:
            lang = self._lang(message)
        await message.reply(help_tasks(lang), mention_author=False)
        return True
