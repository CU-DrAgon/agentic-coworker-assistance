"""Tone presets, Discord side (logic in `tone.py`): a mixin of `SecretaryBot`.

* `_tone_for` -> (tone, note) for one answer; `_agent_kwargs` passes it to `ask` (chat and voice);
* `set_tone` agent tool (`_tone_tools`, appended to `extra_tools`; a voice action tool);
* typed commands: shared room ("@Salid เปลี่ยนโทนเป็นทางการ", "@Salid change tone to professional",
  "โทนอะไร") sets the CHANNEL tone (or the server default with a scope word) for executives and
  server managers; in a member's private channel or DM it sets that PERSON's tone (anyone may);
* slash `/โทน` `/tone`.

One core, `tone_apply`, does the rights check, the validation and the write for all three, and
returns the confirmation already spoken in the NEW tone (`tone_set_<tone>_<scope>` catalog keys).
"""

from __future__ import annotations

import asyncio
import logging

import discord

from . import tone as T
from .i18n import t

logger = logging.getLogger(__name__)

TONE_COMMANDS = ("โทน", "tone")
_CLEAR_NOTE = {"-", "none", "clear", "off", "ลบ", "ล้าง", "ไม่มี", "ไม่ต้อง"}


class ToneMixin:
    # ------------------------------------------------------------- which tone now
    def _tone_private(self, message, asker_id: int) -> bool:
        """A DM, or the asker's own private Salid channel: where their personal tone applies."""
        if getattr(message, "guild", None) is None:
            return True
        cid = getattr(getattr(message, "channel", None), "id", 0)
        return self._pc_owners.get(cid) == asker_id

    def _tone_for(self, message, guild, asker, profile) -> tuple[str, str]:
        """(tone, owner's note) for an answer to `asker` in `message`'s channel. Never raises."""
        try:
            g = self.team.current(guild.id)
            cid = getattr(getattr(message, "channel", None), "id", 0)
            tone = T.effective_tone(g, cid, profile, self._tone_private(message, asker.id))
            return tone, g.tone_note
        except Exception:
            logger.warning("tone lookup failed; using the default tone", exc_info=True)
            return T.DEFAULT_TONE, ""

    # ------------------------------------------------------------- the one write path
    def tone_apply(
        self,
        *,
        guild_id: int,
        channel_id: int,
        user,
        private: bool,
        lang: str,
        tone: str = "",
        scope: str = "channel",
        note: str = "",
    ) -> dict:
        """Change a tone and/or the server note. Synchronous (store calls only), so tools can call
        it from the agent's thread. Returns {"status": "ok", "say": text} or {"error": code,
        "say": text}; `say` is what Salid tells the user, in `lang` and already in the new tone."""
        scope = scope if scope in T.SCOPES else "channel"
        if scope == "channel" and private:
            scope = "me"  # in your own private channel / DM, "here" means you
        key = ""
        if str(tone or "").strip():
            key = T.resolve_tone(tone, allow_default=True) or ""
            if not key:
                return {"error": "unknown_tone", "say": t("tone_unknown", lang)}
        note_text = str(note or "").strip()
        if not key and not note_text:
            return {"error": "nothing_to_change", "say": t("tone_unknown", lang)}
        if scope in ("channel", "server") and not self._may_configure(user):
            return {"error": "not_allowed", "say": t("tone_denied", lang)}
        clear_note = note_text.casefold() in _CLEAR_NOTE
        if note_text:
            if scope != "server":
                return {"error": "note_scope", "say": t("tone_note_scope", lang)}
            if not clear_note:
                why = T.note_problem(note_text)
                if why:
                    return {"error": why, "say": t(f"tone_note_{why}", lang)}
        lines: list[str] = []
        try:
            if key:
                lines.append(
                    self._tone_write(guild_id, channel_id, user, scope, key, lang, private)
                )
            if note_text:
                clean = "" if clear_note else T.clean_note(note_text)
                self.team.set_tone_note(guild_id, clean)
                lines.append(
                    t("tone_note_cleared", lang)
                    if not clean
                    else t("tone_note_set", lang, note=clean)
                )
        except Exception:
            logger.exception("could not save the tone")
            return {"error": "failed", "say": t("tone_failed", lang)}
        return {"status": "ok", "scope": scope, "tone": key or None, "say": "\n".join(lines)}

    def _tone_write(self, gid: int, cid: int, user, scope: str, key: str, lang: str, private: bool):
        """Write one tone (`key` may be RESET) at `scope`; the confirmation line."""
        reset = key == T.RESET
        if scope == "channel":
            self.team.set_channel_tone(gid, cid, None if reset else key)
        elif scope == "server":
            self.team.set_server_tone(gid, T.DEFAULT_TONE if reset else key)
        else:
            from .profiles import ProfileTools

            res = ProfileTools(
                self.profiles, gid, user.id, lambda: self._personal_allowed(gid)
            ).remember("tone", key)
            if "error" in res:
                raise ValueError(res["error"])
        if reset and scope != "server":
            now = T.effective_tone(
                self.team.current(gid), cid, self._profile_of(gid, user.id), private
            )
            return t(
                f"tone_reset_{'me' if scope == 'me' else 'channel'}", lang, label=T.label(now, lang)
            )
        text = t(f"tone_set_{T.DEFAULT_TONE if reset else key}_{scope}", lang)
        if scope == "me" and not private:
            text += " " + t("tone_me_elsewhere", lang)
        return text

    def tone_overview(self, guild_id: int, channel_id: int, user, private: bool, lang: str) -> str:
        """The "what tone" reply: current tone here, the server note, the presets and how to change."""
        g = self.team.current(guild_id)
        now = T.effective_tone(g, channel_id, self._profile_of(guild_id, user.id), private)
        head = t("tone_current_me" if private else "tone_current", lang, label=T.label(now, lang))
        lines = [head]
        if g.tone_note:
            lines.append(t("tone_current_note", lang, note=g.tone_note))
        lines += ["", t("tone_presets_h", lang), T.preset_list(lang), "", t("tone_footer", lang)]
        return "\n".join(lines)

    # ------------------------------------------------------------- agent tool
    def _tone_tools(self, message, guild, asker, is_exec: bool, lang: str) -> list:
        """`set_tone` for one request (chat or voice; in a voice room `message.channel` is the
        meeting's text channel). Never raises."""
        try:
            channel_id = getattr(getattr(message, "channel", None), "id", 0)
            private = self._tone_private(message, asker.id)

            def set_tone(tone: str = "", scope: str = "channel", note: str = "") -> dict:
                """Change how Salid talks (her tone of voice). Use when someone asks in words:
                "พูดให้ทางการหน่อย" / "speak more formally" (professional), "พูดสบายๆ เป็นกันเอง" /
                "be more casual" (friendly), "กระชับขึ้น" / "keep it short" (concise), "ใจเย็นๆ หน่อย" /
                "be calm" (calm), "กลับมาสดใสเหมือนเดิม" / "be cheerful again" (cheerful). Salid's
                identity, name and politeness particle never change, only the style. Reply with
                the `say` text as it is: it is already written in the new tone.

                Args:
                    tone: cheerful, friendly, professional, concise or calm; "default" removes
                        this channel's own tone (or the person's own tone); empty = leave the
                        tone unchanged (only a note).
                    scope: channel (this channel; the default; admins / executives only; in the
                        asker's own private channel or DMs it means "me"), server (the default for
                        the whole server; admins / executives only) or me (only the asker's own
                        private channel and DMs; anyone may).
                    note: optional wording note for the whole server (needs scope server;
                        admins / executives only; up to 200 characters, only about HOW to speak,
                        e.g. "เรียกลูกค้าว่า คุณลูกค้า เสมอ"); "clear" removes it.
                """
                return self.tone_apply(
                    guild_id=guild.id,
                    channel_id=channel_id,
                    user=asker,
                    private=private,
                    lang=lang,
                    tone=tone,
                    scope=scope,
                    note=note,
                )

            return [set_tone]
        except Exception:
            logger.warning("tone tools unavailable", exc_info=True)
            return []

    # ------------------------------------------------------------- typed commands
    async def _tone_command(self, message: discord.Message) -> bool:
        """'@Salid เปลี่ยนโทนเป็นทางการ' / 'change tone to professional' / 'โทนอะไร': the channel's
        tone (or, with a scope word, the server default). Executives and server managers; others
        get a refusal when the words were explicit. Show is open to everyone."""
        if message.guild is None:
            return False
        text = self._command_text(message)
        server_tone = T.parse_server_tone_command(text)
        cmd = server_tone or T.parse_tone_command(text)
        if cmd is None:
            return False
        lang = self._lang(message)
        gid, cid, user = message.guild.id, message.channel.id, message.author
        await asyncio.to_thread(self.team.current, gid)
        private = self._tone_private(message, user.id)
        if cmd == "show":
            reply = await asyncio.to_thread(self.tone_overview, gid, cid, user, private, lang)
            await message.reply(reply, mention_author=False)
            return True
        if not private and not self._may_configure(user) and not T.explicit_tone_words(text):
            return False  # "talk short" from a member is a request to the agent, not a refusal
        res = await asyncio.to_thread(
            self.tone_apply,
            guild_id=gid,
            channel_id=cid,
            user=user,
            private=private,
            lang=lang,
            tone=cmd,
            scope="server" if server_tone else "channel",
        )
        await message.reply(res["say"], mention_author=False)
        return True

    async def _personal_tone_command(self, message, guild, member) -> bool:
        """The same words in a DM or a member's private channel: the PERSON's tone."""
        text = self._command_text(message)
        cmd = T.parse_tone_command(text)
        if cmd is None:
            return False
        lang = self._lang(message)
        cid = getattr(message.channel, "id", 0)
        if cmd == "show":
            reply = await asyncio.to_thread(self.tone_overview, guild.id, cid, member, True, lang)
        else:
            res = await asyncio.to_thread(
                self.tone_apply,
                guild_id=guild.id,
                channel_id=cid,
                user=member,
                private=True,
                lang=lang,
                tone=cmd,
                scope="me",
            )
            reply = res["say"]
        await message.reply(reply, mention_author=False)
        return True

    # ------------------------------------------------------------- slash command
    async def handle_tone(
        self,
        interaction: discord.Interaction,
        tone: str | None,
        scope: str | None = None,
        note: str | None = None,
    ) -> None:
        """/โทน /tone: show this channel's tone with the presets, or (executives / server
        managers; anyone for their own private channel) set it. Ephemeral."""
        guild, channel, user = interaction.guild, interaction.channel, interaction.user
        send = interaction.response.send_message
        if guild is None or channel is None:
            await send(t("setup_denied", self._interaction_lang(interaction)), ephemeral=True)
            return
        await asyncio.to_thread(self.team.current, guild.id)
        lang = self._interaction_lang(interaction)
        private = await self._pc_owner(guild.id, channel.id) == user.id
        if tone is None and not note:
            reply = await asyncio.to_thread(
                self.tone_overview, guild.id, channel.id, user, private, lang
            )
            await send(reply, ephemeral=True)
            return
        res = await asyncio.to_thread(
            self.tone_apply,
            guild_id=guild.id,
            channel_id=channel.id,
            user=user,
            private=private,
            lang=lang,
            tone=tone or "",
            scope=scope or "channel",
            note=note or "",
        )
        await send(res["say"], ephemeral=True)
