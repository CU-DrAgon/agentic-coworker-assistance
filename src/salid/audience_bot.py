"""Audience languages glue: who Salid answers in which languages (see `salid.audience`).

`AudienceMixin.langs_for(guild, channel, *user_ids)` is the ONE resolution: the channel's language
(its explicit override, else the company language) first, then the personal language of each person
involved when personal settings are on and the org policy `language_mode` is `audience`. An unchosen
server returns `["auto"]` and the callers keep following the words people write.

Typed commands: personal ("talk to me in Thai" / "use the company language"), org policy
("company language only" / "allow personal languages", executives), per meeting (the chair). The
agent tool `set_language_mode` does the same by voice or chat.
"""

from __future__ import annotations

import asyncio
import logging

from . import audience as aud
from .i18n import resolve_lang

logger = logging.getLogger(__name__)
NOTICE_KIND = "salid_lang_notice"  # DocStore: the day a channel last got the mode statement


class AudienceMixin:
    # ------------------------------------------------------------- resolution
    def _audience_mode(self, gid: int) -> str:
        try:
            return aud.valid_mode(self.team.language_mode(gid))
        except Exception:
            return aud.DEFAULT_MODE

    def _person_extra(self, gid: int, uid: int) -> str:
        try:
            return aud.person_lang(self._profile_of(gid, uid).language)
        except Exception:
            return ""

    def langs_for(self, guild_id: int, channel_id: int | None, *user_ids: int) -> list[str]:
        """The ordered languages of a message in `channel_id` (None: the company language) to the
        people `user_ids`."""
        try:  # one company language for the whole server (no per-channel overrides)
            company = aud.company_lang(self.team.current(guild_id))
        except Exception:
            company = "auto"
        people = [self._person_extra(guild_id, u) for u in user_ids if u]
        return aud.langs_for(company, *people, mode=self._audience_mode(guild_id))

    def _person_langs(self, gid: int, *uids: int) -> list[str]:
        """Languages for a private message from Salid to / about people (company first); `["auto"]`
        while the server has not chosen its language."""
        return self.langs_for(gid, None, *uids)

    def _deliver_langs(self, gid: int, *uids: int, text: str | None = None) -> list[str]:
        """`_person_langs` ready to render: an unchosen server renders in the first person's own
        language (else the words, `auto`), i.e. exactly the old single-language path."""
        langs = self.langs_for(gid, None, *uids)
        if langs != ["auto"]:
            return langs
        return [self._person_lang(gid, uids[0] if uids else 0, text)]

    # ------------------------------------------------------------- the mode statement
    def _with_mode_notice(self, channel_id: int, guild_id: int, text: str, langs) -> str:
        """The first answer of Salid in a channel each day ends with ONE line: the answering mode
        and how to change it. Never on the later answers (a cheap flag per channel and day)."""
        if not aud.MODE_NOTICES or not langs or "auto" in langs or not text:
            return text
        day = self.s.now().date().isoformat()
        try:
            seen = self.docs.get(NOTICE_KIND, guild_id, str(channel_id)) or {}
            if seen.get("day") == day:
                return text
            self.docs.put(NOTICE_KIND, guild_id, str(channel_id), {"day": day})
        except Exception:  # a store hiccup must never stop an answer
            return text
        company_only = self._audience_mode(guild_id) == "company_only"
        line = aud.mode_line_with_hint(langs, company_only=company_only).replace("\n", " ")
        return f"{text}\n\nℹ️ {line}"

    # ------------------------------------------------------------- typed commands
    async def _audience_command(self, message) -> bool:
        """Personal language, org language mode: typed forms. True when the message was one."""
        if message.guild is None:
            return False
        text = self._command_text(message)
        gid = message.guild.id
        company = [self.langs_for(gid, message.channel.id)[0]]
        if (personal := aud.parse_personal_language(text)) is not None:
            await message.reply(
                self._set_personal_language(gid, message.author.id, personal, company),
                mention_author=False,
            )
            return True
        if (mode := aud.parse_language_mode(text)) is not None:
            await message.reply(
                await self._set_org_mode(message.author, gid, mode, company), mention_author=False
            )
            return True
        if (mode := aud.parse_meeting_mode(text)) is not None:
            reply = self._set_meeting_mode(message.author, gid, mode, company)
            if reply:
                await message.reply(reply, mention_author=False)
                return True
        return False

    def _set_personal_language(self, gid: int, uid: int, cmd, company: list[str]) -> str:
        kind, code = cmd
        if not self._personal_allowed(gid):
            return aud.tt("aud_personal_off", company)
        p = self.profiles.get(gid, uid)
        if kind == "unset":
            p.language = None
            self.profiles.save(gid, uid, p)
            return aud.tt("aud_personal_unset", company)
        if company[0] == code:
            p.language = None
            self.profiles.save(gid, uid, p)
            return aud.tt("aud_personal_same", company, lname=aud.name_in(code))
        p.language = code
        self.profiles.save(gid, uid, p)
        return aud.tt("aud_personal_set", [company[0], code], lname=aud.name_in(code))

    async def _set_org_mode(self, member, gid: int, mode: str, company: list[str]) -> str:
        if not self.is_exec_member(member):
            return aud.tt("aud_only_denied", company)
        await asyncio.to_thread(self.team.set_language_mode, gid, mode)
        key = "aud_mode_set_only" if mode == "company_only" else "aud_mode_set_audience"
        return aud.tt(key, company, lname=aud.name_in(company[0]))

    def _meeting_session_of(self, member):
        ch = getattr(getattr(member, "voice", None), "channel", None)
        gid = getattr(getattr(member, "guild", None), "id", None)
        if ch is None or gid is None:
            return None
        return self.meetings.session_in(gid, ch.id)

    def _set_meeting_mode(self, member, gid: int, mode: str, company: list[str]) -> str:
        """The chair (whoever invited Salid, or an executive in the room) sets the mode for this
        meeting only. "" when the speaker is in no meeting (not a meeting command then)."""
        s = self._meeting_session_of(member)
        if s is None:
            return ""
        if not (getattr(s, "starter_id", 0) == member.id or self.is_exec_member(member)):
            return aud.tt("aud_meet_denied", [s.lang])
        self.meetings.set_language_mode(s, mode)
        langs = s.langs or company
        key = "aud_meet_set_only" if mode == "company_only" else "aud_meet_set_audience"
        return aud.tt(key, langs, lname=aud.name_in(langs[0]))

    # ------------------------------------------------------------- agent tool
    def _audience_tools(self, message, guild, asker, is_exec: bool, lang: str) -> list:
        """`set_language_mode(mode, scope)`: executives set the org policy; the chair of the meeting
        the speaker sits in sets it for that meeting only."""
        gid = guild.id
        bot = self

        def set_language_mode(mode: str, scope: str = "company") -> dict:
            """Force one language or allow personal languages. mode: "company_only" (answer only in
            the company language, still listening to Thai and English) or "audience" (company
            language plus each person's own). scope: "company" (the whole organisation, executives
            only) or "meeting" (the voice meeting the speaker is in, whoever invited Salid or an
            executive). Use for "company language only" / "English only" / "ตอบหลายภาษาได้"."""
            if mode not in aud.MODES:
                return {"error": "mode must be audience or company_only"}
            company = [bot.langs_for(gid, message.channel.id)[0]]
            if scope == "meeting":
                reply = bot._set_meeting_mode(asker, gid, mode, company)
                return (
                    {"status": "ok", "say": reply}
                    if reply
                    else {"error": "no_meeting", "say": aud.tt("aud_meet_none", company)}
                )
            if not is_exec:
                return {"error": "not_executive", "say": aud.tt("aud_only_denied", company)}
            bot.team.set_language_mode(gid, mode)
            key = "aud_mode_set_only" if mode == "company_only" else "aud_mode_set_audience"
            return {"status": "ok", "say": aud.tt(key, company, lname=aud.name_in(company[0]))}

        return [set_language_mode]

    def _audience_primary(self, langs: list[str], text: str | None = None) -> str:
        return resolve_lang(aud.primary(langs), text)
