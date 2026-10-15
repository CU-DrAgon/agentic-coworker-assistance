"""One place that sends a private message (DM) to a member and survives the member's settings.

Used by the DM tool, meeting invites and reminders. A member who closed server DMs raises
`discord.Forbidden` (error 50007): that is a normal outcome, reported as `closed`, never a crash.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import discord

from .i18n import t

logger = logging.getLogger(__name__)


@dataclass
class DMResult:
    ok: bool
    reason: str = ""  # "" | "closed" | "not_found" | "error"


async def send_dm(member, text: str, files=()) -> DMResult:
    """DM `member` (None = not in the server). Never raises."""
    if member is None:
        return DMResult(False, "not_found")
    try:
        await member.send(text, **({"files": list(files)} if files else {}))
        return DMResult(True)
    except discord.Forbidden:
        logger.info("DM to %s failed (DMs closed)", getattr(member, "id", "?"))
        return DMResult(False, "closed")
    except discord.HTTPException:
        logger.info("DM to %s failed", getattr(member, "id", "?"))
        return DMResult(False, "error")
    except Exception:
        logger.warning("DM to %s failed", getattr(member, "id", "?"), exc_info=True)
        return DMResult(False, "error")


def closed_dm_message(name: str, server: str, lang: str) -> str:
    """What to tell the requester when `name` has server DMs turned off, and how to open them."""
    return t("dm_closed", lang, name=name, server=server)
