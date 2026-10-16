"""Instant acknowledgement for slow work.

When Salid is addressed and the answer will take a while (files, links, audio, a deep task) it
replies at once with a short natural line ("ขอเวลาแป๊บนึงนะคะ ขออ่านเอกสารให้เข้าใจก่อน 📄"); for
anything else the same kind of line goes out if no answer is back after SLOW_AFTER_S seconds. The
bot then edits that message into the first chunk of the answer.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import Callable

import discord

from .docs import doc_kind
from .i18n import t
from .image_memory import image_mime
from .media import media_candidates
from .web_read import extract_urls

SLOW_AFTER_S = 6.0
FLAVORS = ("doc", "image", "audio", "link", "deep", "slow")
VARIANTS = 2  # ack_<flavor>_1 .. ack_<flavor>_N in the catalog


def ack_flavor(message, text: str, *, deep: bool = False) -> str | None:
    """What kind of slow work this addressed message means, or None when it is probably quick.
    Priority: deep task, audio/video, document, image, link."""
    if deep:
        return "deep"
    atts = list(getattr(message, "attachments", ()) or ())
    if media_candidates(message):
        return "audio"
    if any(doc_kind(a) and not image_mime(a) for a in atts):
        return "doc"
    if any(image_mime(a) for a in atts):
        return "image"
    if atts:
        return "doc"
    if extract_urls(text or ""):
        return "link"
    return None


def ack_text(flavor: str, lang: str, choose: Callable[[list[int]], int] = random.choice) -> str:
    return t(f"ack_{flavor}_{choose(list(range(1, VARIANTS + 1)))}", lang)


class Ack:
    """The acknowledgement message of one answer: sent now, or after a delay unless the answer
    arrives first. `sent` is the Discord message (None when nothing was posted)."""

    def __init__(self, message, text: str, sleep=asyncio.sleep) -> None:
        self.message, self.text, self._sleep = message, text, sleep
        self.sent = None
        self._task: asyncio.Task | None = None
        self._committed = False

    async def now(self) -> None:
        self._committed = True
        await self._send()

    def after(self, seconds: float = SLOW_AFTER_S) -> None:
        self._task = asyncio.create_task(self._later(seconds))

    async def _later(self, seconds: float) -> None:
        await self._sleep(seconds)
        self._committed = True  # past this point the send is not interrupted
        await self._send()

    async def _send(self) -> None:
        try:
            self.sent = await self.message.reply(self.text, mention_author=False)
        except discord.HTTPException:
            self.sent = None

    async def finish(self) -> None:
        """The answer is ready: stop a pending timer (or wait for a send already under way)."""
        task = self._task
        if task is None or task.done():
            return
        if not self._committed:
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
