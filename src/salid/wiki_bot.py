"""Discord side of the company wiki: the agent tools' hooks (who can read what, which channel is
the knowledge channel, posting and replacing a page's post) as a `SecretaryBot` mixin.

Logic and storage are in `salid.wiki`. Nothing here posts unless `wiki_publish` was confirmed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import uuid

import discord

from .channel_access import split_message
from .i18n import t
from .minutes import can_see_and_post, resolve_channel
from .permissions import channel_missing, permission_message
from .wiki import WikiHooks, WikiPage, WikiRepo, WikiTools, wiki_tools

logger = logging.getLogger(__name__)

POST_LIMIT = 1900
# Channels that look like a knowledge channel: asked about once, never picked silently.
_KNOWLEDGE = re.compile(
    r"wiki|knowledge|docs?(?![a-z])|handbook|playbook|sop|how-?to|ความรู้|คู่มือ|เอกสาร|วิธี", re.I
)


def is_forum(channel) -> bool:
    return isinstance(channel, discord.ForumChannel) or (
        getattr(channel, "type", None) == discord.ChannelType.forum
    )


def looks_like_knowledge(channel) -> bool:
    return bool(
        _KNOWLEDGE.search(getattr(channel, "name", "") or "")
        or _KNOWLEDGE.search(getattr(channel, "topic", "") or "")
    )


def page_message(page: WikiPage, lang: str) -> str:
    """The text posted for a page: title, body, and a small footer (updated, written by)."""
    day = (page.updated_at or "")[:10]
    footer = t("wiki_post_footer", lang, date=day, who=page.author_name or "Salid")
    return f"# {page.title}\n\n{page.body.strip()}\n\n-# {footer}"


class WikiMixin:
    """Needs `self.docs`, `self.llm`, `self.s`, `self.team`, `self.secretary_state`, `_people`,
    `_channel_viewers`. Call `_wiki_init()` once after `self.docs` exists."""

    def _wiki_init(self) -> None:
        self.wiki = WikiRepo(self.docs, clock=lambda: self.s.now())

    # ------------------------------------------------------------- agent tools
    def _wiki_tools(self, guild, member, is_exec: bool, memory, lang: str) -> list:
        """The wiki tools for this request, bound to the guild and the asker. Needs the memory
        tools (their scope decides what the asker may read); [] when the feature is off."""
        if memory is None or not self.team.enabled(guild.id, "wiki"):
            return []
        loop = asyncio.get_running_loop()
        w = WikiTools(
            guild_id=guild.id,
            asker_id=member.id,
            asker_name=getattr(member, "display_name", str(member.id)),
            is_exec=is_exec,
            memory=memory,
            repo=self.wiki,
            client=self.llm,
            model=self.s.gemini_model,
            lang=lang,
            now=self.s.now,
            state=self.secretary_state,
            request_id=uuid.uuid4().hex,
            hooks=WikiHooks(
                channel_names=lambda ids: [self._channel_label(guild, i) for i in ids],
                unreadable_by=lambda ids: self._wiki_unreadable(guild, member, ids),
                destination_gap=lambda dest, ids: self._wiki_dest_gap(guild, dest, ids),
                candidates=lambda: self._wiki_candidates(guild, member),
                resolve=lambda name, rid: self._wiki_resolve(guild, member, name, rid),
                post=lambda page, dest: self._wiki_run(
                    loop, self._wiki_post(guild, page, dest, lang)
                ),
                unpost=lambda page: self._wiki_run(loop, self._wiki_unpost(guild, page)),
            ),
        )
        return wiki_tools(w)

    @staticmethod
    def _wiki_run(loop, coro):
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=60)

    @staticmethod
    def _channel_label(guild, channel_id: int) -> str:
        ch = guild.get_channel(channel_id)
        return getattr(ch, "name", None) or str(channel_id)

    # ------------------------------------------------------------- who can read
    def _wiki_unreadable(self, guild, writer, channel_ids: list[int]) -> list[str]:
        """Display names of people who cannot read ALL the source channels (the writer is not
        listed). A channel Salid cannot find counts as unreadable by everyone."""
        people = self._people(guild)
        readers = set(people)
        for cid in channel_ids:
            ch = guild.get_channel(cid)
            readers &= set(self._channel_viewers(guild, ch)) if ch is not None else set()
        return sorted(n for uid, n in people.items() if uid not in readers and uid != writer.id)

    def _wiki_dest_gap(self, guild, dest_id: int, channel_ids: list[int]) -> list[str]:
        """People who can read the destination but not all the source channels."""
        dest = guild.get_channel(dest_id)
        if dest is None:
            return ["?"]
        people = self._people(guild)
        audience = set(self._channel_viewers(guild, dest))
        readers = set(audience)
        for cid in channel_ids:
            ch = guild.get_channel(cid)
            readers &= set(self._channel_viewers(guild, ch)) if ch is not None else set()
        return sorted(people.get(u, str(u)) for u in audience - readers)

    # ------------------------------------------------------------- channels
    @staticmethod
    def _wiki_all_channels(guild) -> list:
        return [*guild.text_channels, *(getattr(guild, "forums", None) or [])]

    def _wiki_candidates(self, guild, member) -> list[str]:
        out = []
        for c in self._wiki_all_channels(guild):
            if looks_like_knowledge(c) and can_see_and_post(c, member):
                out.append(("forum " if is_forum(c) else "") + f"#{c.name}")
        return out

    def _wiki_resolve(self, guild, member, name: str | None, remembered: int | None) -> dict:
        if name is None:
            ch = guild.get_channel(remembered) if remembered else None
            if ch is None:
                return {
                    "error": "channel_gone",
                    "note": "the remembered knowledge channel no longer exists; ask which to use",
                    "channel_candidates": self._wiki_candidates(guild, member)[:8],
                }
            if not can_see_and_post(ch, member):
                return {"error": "not_allowed", "note": "the requester cannot post there"}
            return {"id": ch.id, "name": ch.name}
        mention = re.fullmatch(r"<#(\d+)>", name.strip())
        if mention:
            ch = guild.get_channel(int(mention.group(1)))
            if ch is None or not can_see_and_post(ch, member):
                return {"error": "destination_not_found"}
            return {"id": ch.id, "name": ch.name}
        res = resolve_channel(
            self._wiki_all_channels(guild), name, lambda c: can_see_and_post(c, member)
        )
        if res.status == "ok":
            return {"id": res.channel.id, "name": res.channel.name}
        if res.status == "ambiguous":
            return {"error": "ambiguous", "candidates": [f"#{c.name}" for c in res.candidates]}
        return {
            "error": "not_allowed" if res.status == "not_allowed" else "destination_not_found",
            "channel_candidates": self._wiki_candidates(guild, member)[:8],
        }

    # ------------------------------------------------------------- posting
    async def _wiki_unpost(self, guild, page: WikiPage) -> None:
        """Remove the earlier post of a page (best effort: it may already be gone)."""
        post = page.post or {}
        if post.get("thread_id"):
            thread = guild.get_channel_or_thread(int(post["thread_id"]))
            if thread is not None:
                with contextlib.suppress(discord.HTTPException):
                    await thread.delete()
            return
        ch = guild.get_channel(int(post.get("channel_id") or 0))
        if ch is None:
            return
        for mid in post.get("message_ids") or []:
            with contextlib.suppress(discord.HTTPException):
                await ch.get_partial_message(int(mid)).delete()

    async def _wiki_post(self, guild, page: WikiPage, dest_id: int, lang: str) -> dict:
        dest = guild.get_channel(dest_id)
        if dest is None:
            return {"error": "destination_not_found"}
        need = ("send_messages", "create_public_threads") if is_forum(dest) else ("send_messages",)
        if bad := channel_missing(dest, guild, need):
            return {
                "error": "missing_permission",
                "how_to_grant": permission_message(bad, lang, self.application_id),
            }
        no_pings = discord.AllowedMentions.none()
        chunks = split_message(page_message(page, lang), POST_LIMIT)
        await self._wiki_unpost(guild, page)  # "update" = replace the earlier post
        try:
            if is_forum(dest):
                made = await dest.create_thread(
                    name=page.title[:100], content=chunks[0], allowed_mentions=no_pings
                )
                thread, first = made.thread, made.message
                for part in chunks[1:]:
                    await thread.send(part, allowed_mentions=no_pings)
                post = {"channel_id": dest.id, "thread_id": thread.id, "message_ids": []}
            else:
                sent = [await dest.send(c, allowed_mentions=no_pings) for c in chunks]
                first = sent[0]
                post = {
                    "channel_id": dest.id,
                    "thread_id": None,
                    "message_ids": [m.id for m in sent],
                }
        except discord.HTTPException:
            logger.warning("wiki post failed", exc_info=True)
            return {"error": "could not post there (Discord refused)"}
        return {"status": "posted", "post": post, "link": getattr(first, "jump_url", None)}
