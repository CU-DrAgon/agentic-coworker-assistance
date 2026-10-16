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
from .fillers import POOL_SIZE, pick_fallback, within
from .image_memory import image_mime
from .media import media_candidates
from .web_read import extract_urls

SLOW_AFTER_S = 6.0
FLAVORS = ("doc", "image", "audio", "link", "deep", "slow")
VARIANTS = POOL_SIZE  # ack_<flavor>_1 .. ack_<flavor>_N in the catalog
LLM_TIMEOUT_S = 2.0


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


def ack_text(
    flavor: str,
    lang: str,
    choose: Callable[[list[str]], str] = random.choice,
    avoid=(),
    particle: str = "ค่ะ",
) -> str:
    """The hand-written fallback line (a pool of 8 per flavor), never one of `avoid`."""
    return pick_fallback(flavor, lang, avoid, particle, choose)


class Ack:
    """The acknowledgement message of one answer: sent now, or after a delay unless the answer
    arrives first. `text` is the hand-written fallback; `pending` is a task writing a line for
    this request (`FillerWriter.write`), waited for up to `timeout` seconds when the message goes
    out, never longer, and dropped if it is late: only one acknowledgement is ever sent.
    `sent` is the Discord message (None when nothing was posted)."""

    def __init__(
        self,
        message,
        text: str,
        sleep=asyncio.sleep,
        pending: asyncio.Task | None = None,
        timeout: float = LLM_TIMEOUT_S,
        on_text: Callable[[str], None] | None = None,
    ) -> None:
        self.message, self.text, self._sleep = message, text, sleep
        self._pending, self._timeout, self._on_text = pending, timeout, on_text
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
        text = await within(self._pending, self._timeout) or self.text
        if self._on_text is not None:
            self._on_text(text)
        try:
            self.sent = await self.message.reply(text, mention_author=False)
        except discord.HTTPException:
            self.sent = None

    async def finish(self) -> None:
        """The answer is ready: stop a pending timer (or wait for a send already under way)."""
        task = self._task
        if task is None or task.done():
            self._drop_pending()
            return
        if not self._committed:
            task.cancel()
            self._drop_pending()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def _drop_pending(self) -> None:
        if self._pending is not None and not self._pending.done():
            self._pending.cancel()
