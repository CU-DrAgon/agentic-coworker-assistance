from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timedelta

import discord

from .config import Settings
from .digest import build_daily_digest
from .extract import apply_correction, ingest_attachment
from .qa import ask
from .store import Store

logger = logging.getLogger(__name__)
ATTACH_LIMIT = 15 * 1024 * 1024


def chunks(text: str, n: int = 1900):
    while text:
        yield text[:n]
        text = text[n:]


class SecretaryBot(discord.Client):
    def __init__(self, settings: Settings, store: Store, llm):
        intents = discord.Intents(
            guilds=True, messages=True, message_content=True, members=True, dm_messages=True
        )
        super().__init__(intents=intents)
        self.s, self.store, self.llm = settings, store, llm
        self._digest_task: asyncio.Task | None = None

    async def setup_hook(self):
        self._digest_task = asyncio.create_task(self._digest_loop())

    async def on_ready(self):
        logger.info("logged in as %s", self.user)

    # ------------------------------------------------------------- permissions
    def role_names_for(self, message: discord.Message) -> list[str]:
        if isinstance(message.author, discord.Member):
            return [r.name for r in message.author.roles]
        names: list[str] = []  # DM: look the user up in shared guilds
        for g in self.guilds:
            m = g.get_member(message.author.id)
            if m:
                names += [r.name for r in m.roles]
        return names

    def is_exec(self, message: discord.Message) -> bool:
        return self.s.is_exec(message.author.id, self.role_names_for(message))

    # ------------------------------------------------------------- routing
    async def on_message(self, message: discord.Message):
        if message.author.bot or message.author == self.user:
            return
        try:
            if await self._maybe_correction(message):
                return
            if message.channel.id in self.s.receipt_channel_ids and message.attachments:
                await self._intake(message)
                return
            mentioned = self.user is not None and self.user in message.mentions
            if message.guild is None or mentioned or message.channel.id in self.s.ask_channel_ids:
                await self._answer(message)
        except Exception:
            logger.exception("handler failed")

    async def _maybe_correction(self, message: discord.Message) -> bool:
        ref = message.reference
        if not ref or not ref.message_id or not message.content.strip():
            return False
        receipt = self.store.get_by_reply_message(ref.message_id)
        if not receipt:
            return False
        if message.author.id != receipt.submitter_id and not self.is_exec(message):
            await message.reply("แก้ไขได้เฉพาะผู้ส่งใบเสร็จหรือผู้บริหารเท่านั้น", mention_author=False)
            return True
        async with message.channel.typing():
            text = await asyncio.to_thread(
                apply_correction, self.store, self.llm, self.s, receipt, message.content
            )
        await message.reply(text, mention_author=False)
        return True

    async def _intake(self, message: discord.Message):
        note = message.content.strip()
        for att in message.attachments:
            if att.size > ATTACH_LIMIT:
                continue
            data = await att.read()
            async with message.channel.typing():
                res = await asyncio.to_thread(
                    ingest_attachment,
                    self.store,
                    self.llm,
                    self.s,
                    data=data,
                    filename=att.filename,
                    content_type=att.content_type,
                    submitter_id=message.author.id,
                    submitter_name=message.author.display_name,
                    channel_id=message.channel.id,
                    message_id=message.id,
                    note=note,
                )
            if res.status == "ignored":
                continue
            if res.reaction:
                with contextlib.suppress(discord.HTTPException):
                    await message.add_reaction(res.reaction)
            sent = await message.reply(res.reply, mention_author=False)
            if res.status == "created":
                self.store.set_reply_message(res.receipt.id, sent.id)

    async def _answer(self, message: discord.Message):
        q = message.content
        if self.user:
            q = q.replace(f"<@{self.user.id}>", "").replace(f"<@!{self.user.id}>", "")
        q = q.strip()
        if not q:
            return
        is_exec = self.is_exec(message)
        async with message.channel.typing():
            ans = await asyncio.to_thread(
                ask, self.s, self.store, q, is_exec=is_exec, user_id=message.author.id
            )
        logger.info("answered in %.1fs (%d rounds, exec=%s)", ans.seconds, ans.rounds, is_exec)
        for i, part in enumerate(chunks(ans.text)):
            await (
                message.reply(part, mention_author=False) if i == 0 else message.channel.send(part)
            )

    # ------------------------------------------------------------- digest
    def next_digest_at(self, now: datetime) -> datetime:
        hh, mm = (int(x) for x in self.s.digest_time.split(":"))
        t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        return t if t > now else t + timedelta(days=1)

    async def _digest_loop(self):
        await self.wait_until_ready()
        while not self.is_closed():
            now = self.s.now()
            nxt = self.next_digest_at(now)
            await asyncio.sleep((nxt - now).total_seconds())
            if self.s.digest_channel_id:
                ch = self.get_channel(self.s.digest_channel_id)
                if ch:
                    text = await asyncio.to_thread(build_daily_digest, self.store, nxt.date())
                    for part in chunks(text):
                        await ch.send(part)
            await asyncio.sleep(1)
