"""Discord glue for business finance, benchmarking and security checks (logic in `bizfin.py`,
`benchmark.py`, `security_check.py`, tools in `bizfin_tools.py`).

`BizfinMixin` gives `SecretaryBot`:

* `_bizfin_tools`: the agent tools, appended to `extra_tools` by `_agent_kwargs` (so voice has them);
* `_link_guard_maybe`: the passive warning when a channel message holds a link that is clearly
  dangerous (flag `link_guard`, rate-limited, never on allowlisted domains).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time

import discord

from .bizfin_tools import BizfinCtx, bizfin_tools
from .pdf_report import find_fonts
from .security_check import GuardLimiter, guard_message, link_guard_hits

logger = logging.getLogger(__name__)
SLIP_LOOKBACK_START, SLIP_LOOKBACK_END = "2000-01-01", "2100-01-01"


class BizfinMixin:
    def _bizfin_init(self) -> None:
        self._guard_limiter = GuardLimiter(time.monotonic)

    # ---- agent tools
    def _bizfin_tools(self, guild, asker, is_exec: bool, lang: str) -> list:
        """The finance / benchmark / security tools for one request. [] when the flag is off;
        never raises."""
        try:
            if not self.team.enabled(guild.id, "bizfin"):
                return []
            web = self.llm is not None and self.team.enabled(guild.id, "web")
            ideas_on = self.team.enabled(guild.id, "ideas")
            ctx = BizfinCtx(
                lang=lang,
                now=self.s.now,
                client=self.llm if web else None,
                model=self.s.gemini_model,
                org=getattr(guild, "name", "") or "",
                fonts=lambda: find_fonts(self.s.pdf_font),
                safe_browsing_key=getattr(self.s, "safe_browsing_key", "") or "",
                save_idea=(
                    (
                        lambda title, section, md: self._bizfin_save_idea(
                            guild, asker, title, section, md
                        )
                    )
                    if ideas_on
                    else None
                ),
                ledger_ref=lambda ref: self._bizfin_ledger_ref(guild, asker, is_exec, ref),
            )
            return bizfin_tools(ctx, web=web)
        except Exception:
            logger.warning("bizfin tools unavailable", exc_info=True)
            return []

    def _bizfin_save_idea(self, guild, asker, title: str, section: str, md: str) -> dict:
        try:
            doc, _key, _offer = self.ideas.save_section(
                guild.id,
                asker.id,
                getattr(asker, "display_name", str(asker.id)),
                title,
                section,
                md,
            )
        except ValueError as e:
            return {"status": "error", "error": str(e)}
        return {"status": "saved", "title": doc.title}

    def _bizfin_ledger_ref(self, guild, asker, is_exec: bool, ref: str) -> int | None:
        """The id of a ledger entry the asker may see whose invoice / reference number equals
        `ref` (executives: the whole guild; everyone else: their own entries only)."""
        want = re.sub(r"[\s-]", "", ref or "").casefold()
        if len(want) < 6:
            return None
        rows = self.store.all_receipts(
            guild.id,
            SLIP_LOOKBACK_START,
            SLIP_LOOKBACK_END,
            submitter_id=None if is_exec else asker.id,
        )
        for r in rows:
            if re.sub(r"[\s-]", "", r.invoice_no or "").casefold() == want:
                return r.id
        return None

    # ---- passive link guard
    async def _link_guard_maybe(self, message, mode: str) -> None:
        """Reply once with a short warning under a message whose link is clearly dangerous. Quiet
        in `off` / `listen` channels (and while snoozed), by flag `link_guard`, never on
        allowlisted domains, rate-limited. Never raises."""
        try:
            guild = message.guild
            content = getattr(message, "content", "") or ""
            if guild is None or mode in ("off", "listen") or "." not in content:
                return
            if not self.team.enabled(guild.id, "link_guard"):
                return
            key = getattr(self.s, "safe_browsing_key", "") or ""
            hits = (
                await asyncio.to_thread(link_guard_hits, content, safe_browsing_key=key)
                if key
                else link_guard_hits(content)
            )
            if not hits:
                return
            if not self._guard_limiter.allow(guild.id, message.channel.id, [h.host for h in hits]):
                return
            text = guard_message(hits, self._lang(message))
            with contextlib.suppress(discord.HTTPException, AttributeError):
                await message.reply(text, mention_author=False)
        except Exception:
            logger.warning("link guard failed", exc_info=True)
