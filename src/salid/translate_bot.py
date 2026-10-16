"""Bot glue for translation (see `translate.py`): the agent tools bound to the asker, the message
context menu "Translate with Salid" and per-channel auto-translate. A `SecretaryBot` mixin."""

from __future__ import annotations

import asyncio
import logging
import re
from functools import partial

import discord

from .i18n import t
from .mentions import defang_raw_mentions
from .translate import (
    AutoTranslateRepo,
    RateLimiter,
    TranslateContext,
    TranslateResult,
    auto_translate,
    english_name,
    format_auto,
    lang_info,
    menu_target,
    neutralize,
    normalize_targets,
    script_language,
    translate_text,
    translate_tools,
    worth_translating,
)

logger = logging.getLogger(__name__)

JUMP_RE = re.compile(r"discord(?:app)?\.com/channels/(\d+)/(\d+)/(\d+)")
_LAST_RE = re.compile(
    r"^(?:the\s+)?(?:last|latest|previous|ล่าสุด)(?:\s+message)?(?:\s+(?:from|by|of)\s+(.+))?$", re.I
)
_REPLY_WORDS = {"", "reply", "this", "replied", "that", "อันนี้", "นี้", "ข้อความนี้"}
HISTORY_SCAN = 50
AUTO_REPLY_CHARS = 1900


class TranslateMixin:
    """Needs `self.docs` (a DocStore), `self.llm`, `self.s`, `self.team` and the usual bot helpers."""

    def _translate_init(self) -> None:
        self.autotranslate = AutoTranslateRepo(self.docs)
        self._auto_limit = RateLimiter()

    # ------------------------------------------------------------------ agent tools
    def _translate_tools(self, message, guild, asker, is_exec: bool, loop) -> list:
        """`translate` and `set_auto_translate`, bound to this guild, channel and asker."""

        def run(coro):
            return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=30)

        def set_auto(targets: list[str], ref: str | None) -> dict:
            if not self._may_configure(asker):
                return {"error": "not_allowed", "needs": "server manager or executive"}
            target = self._find_text_channel(guild, ref, message.channel)
            if target is None:
                return {"error": "channel_not_found"}
            self.autotranslate.set(guild.id, target.id, targets, asker.id)
            return {
                "status": "ok",
                "channel": getattr(target, "name", ""),
                "auto_translate": [english_name(x) for x in targets] or "off",
            }

        return translate_tools(
            TranslateContext(
                resolve=lambda ref: run(self._resolve_message_ref(message, guild, asker, ref)),
                translate=lambda text, target: translate_text(
                    self.llm, self.s.gemini_model, text, target
                ),
                set_auto=set_auto,
            )
        )

    async def _resolve_message_ref(self, message, guild, asker, ref: str) -> dict:
        """The text of the message `ref` names: the one `message` replies to, a message link,
        "last" or "last from <name>". Only channels the asker can read; never chosen by the model."""
        ref = (ref or "").strip()
        found = None
        link = JUMP_RE.search(ref)
        try:
            if link:
                gid, cid, mid = (int(x) for x in link.groups())
                channel = guild.get_channel_or_thread(cid) if gid == guild.id else None
                if channel is None or not self._can_read(channel, asker):
                    return {"error": "cannot_read_that_message"}
                found = await channel.fetch_message(mid)
            elif ref.casefold() in _REPLY_WORDS:
                found = await self._referenced_message(message)
                found = found or await self._last_message(message, asker, "")
            else:
                m = _LAST_RE.match(ref)
                found = await self._last_message(message, asker, (m.group(1) or "") if m else ref)
        except (discord.HTTPException, AttributeError):
            return {"error": "message_not_found"}
        if found is None:
            return {"error": "message_not_found"}
        channel = getattr(found, "channel", None)
        if channel is not None and not self._can_read(channel, asker):
            return {"error": "cannot_read_that_message"}
        text = self._strip_mentions(getattr(found, "content", "") or "", found).strip()
        if not text:
            return {"error": "message_has_no_text"}
        who = "Salid" if found.author == self.user else found.author.display_name
        return {"text": text, "author": who}

    async def _last_message(self, message, asker, name: str | None):
        """Latest text message in the channel from `name` (display name or username, partial
        match), or, with no name, from anyone but the asker and Salid."""
        channel = message.channel
        if not self._can_read(channel, asker):
            return None
        history = getattr(channel, "history", None)
        if history is None:
            return None
        want = (name or "").strip().lstrip("@").casefold()
        before = message if getattr(message, "id", 0) else None
        kw = {"limit": HISTORY_SCAN, **({"before": before} if before is not None else {})}
        fallback = None
        async for prev in history(**kw):
            if prev.author == self.user or not (prev.content or "").strip():
                continue
            if want:
                names = (prev.author.display_name, getattr(prev.author, "name", "") or "")
                if any(want in n.casefold() for n in names):
                    return prev
            elif prev.author.id != asker.id:
                return prev
            elif fallback is None:
                fallback = prev
        return None if want else fallback

    # ------------------------------------------------------------------ context menu
    async def handle_translate_menu(self, interaction: discord.Interaction, message) -> None:
        """Message context menu "Translate with Salid": an ephemeral translation into the
        clicker's language (profile, else the channel's th/en; the other of Thai/English when the
        message is already in it)."""
        guild, user = interaction.guild, interaction.user
        await interaction.response.defer(ephemeral=True)
        await asyncio.to_thread(self.team.current, guild.id)
        lang = self._interaction_lang(interaction)
        send = partial(interaction.followup.send, ephemeral=True)
        if not self.team.enabled(guild.id, "qa"):
            await send(t("summary_unavailable", lang))
            return
        text = self._strip_mentions(getattr(message, "content", "") or "", message).strip()
        if not text:
            await send(t("translate_nothing_pure", lang))
            return
        profile = self._profile_of(guild.id, user.id)
        preferred = profile.language or self._lang_setting(interaction)  # personal wins
        target = menu_target(
            script_language(text) or "", preferred if preferred != "auto" else None
        )
        try:
            r = await asyncio.to_thread(self._menu_translate, text, target)
        except Exception:
            logger.exception("translate menu failed")
            await send(t("translate_failed_pure", lang))
            return
        body = defang_raw_mentions(r.translation, guild)
        who = getattr(message.author, "display_name", "")
        src = (lang_info(r.source_language) or (r.source_language, ""))[0].upper()
        head = f"🌐 {who} ({src} → {lang_info(target)[0].upper()})" if src else f"🌐 {who}"
        for part in _chunks(f"{head}\n{body}"):
            await send(part, allowed_mentions=discord.AllowedMentions.none())

    def _menu_translate(self, text: str, target: str) -> TranslateResult:
        r = translate_text(self.llm, self.s.gemini_model, text, target)
        known = lang_info(r.source_language)
        if known and known[0] == target:  # it was already in the target: show the other one
            other = menu_target(r.source_language, None)
            if other != target:
                return translate_text(self.llm, self.s.gemini_model, text, other)
        return r

    # ------------------------------------------------------------------ auto-translate
    async def _autotranslate_maybe(self, message) -> None:
        """Per-channel auto-translate (off unless an admin turned it on by talking): a human
        message in another language than the channel's target(s) gets a short 🌐 reply. Never
        raises; fails quiet."""
        try:
            guild, channel = message.guild, message.channel
            if guild is None or getattr(message.author, "bot", False):
                return
            keys = [channel.id, getattr(channel, "parent_id", None)]
            if not self.autotranslate.is_cached(guild.id):
                await asyncio.to_thread(self.autotranslate.targets, guild.id, channel.id)
            targets = next(
                (x for k in keys if k and (x := self.autotranslate.targets(guild.id, k))), []
            )
            if not targets or self.llm is None:
                return
            text = self._strip_mentions(message.content or "", message)
            if (
                not worth_translating(text)
                or self._starts_with_name(message)
                or (self.user is not None and self.user in (message.mentions or ()))
            ):
                return
            if len(targets) == 1 and script_language(text) == lang_code(targets[0]):
                return  # already in the channel's language: no model call
            if not self._auto_limit.allow(channel.id):
                return
            self._spawn(self._auto_translate_send(message, message.content or "", targets))
        except Exception:
            logger.warning("auto-translate check failed", exc_info=True)

    async def _auto_translate_send(self, message, text: str, targets: list[str]) -> None:
        try:
            got = await asyncio.to_thread(
                auto_translate, self.llm, self.s.gate_model, text, normalize_targets(targets)
            )
            if not got:
                return
            reply = neutralize(format_auto(got))[:AUTO_REPLY_CHARS]
            await message.reply(
                reply,
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
                suppress_embeds=True,
            )
        except Exception:
            logger.warning("auto-translate failed", exc_info=True)


def lang_code(value: str) -> str:
    info = lang_info(value)
    return info[0] if info else ""


def _chunks(text: str, n: int = 1900):
    while text:
        yield text[:n]
        text = text[n:]
