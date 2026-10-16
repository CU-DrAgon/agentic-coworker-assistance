"""One place that sends a private message (DM) to a member and survives the member's settings.

Used by the DM tool, meeting invites and reminders. A member who closed server DMs raises
`discord.Forbidden` (error 50007): that is a normal outcome, reported as `closed`, never a crash.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import discord

from .i18n import t

logger = logging.getLogger(__name__)


MAX_DM_PEOPLE = 3  # different people one `send_dm` request may DM (send_dm_many reaches more)
DM_CONFIRM_OVER = 3  # a bulk send to more people than this needs a confirmation in a later message
DM_EXEC_OVER = 10  # ... and more than this also needs an executive / manager
MAX_DM_BULK = 50  # hard cap for one bulk send
SENSITIVE_DM_RE = re.compile(
    r"โอนเงิน|โอนตัง|เลขบัญชี|เลขที่บัญชี|รหัสผ่าน|พาสเวิร์ด|รหัส\s*otp|\botp\b|password|passcode"
    r"|transfer\s+(?:the\s+)?(?:money|funds?)|wire\s+(?:the\s+)?money|bank\s+account"
    r"|account\s+number|credit\s+card",
    re.I,
)


def dm_refusal(text: str, others: set[int], new_target: int, is_self: bool) -> dict | None:
    """A hard (non-prompt) limit for the DM tool: more than MAX_DM_PEOPLE different people in one
    request, or money / credential text sent to someone other than the requester."""
    if is_self:
        return None
    if SENSITIVE_DM_RE.search(text or ""):
        return {
            "error": "refused_sensitive",
            "reason": "money transfer or credential request to another person; decline politely "
            "and suggest they contact that person directly",
        }
    if new_target not in others and len(others) >= MAX_DM_PEOPLE:
        return {
            "error": "too_many_recipients",
            "max": MAX_DM_PEOPLE,
            "reason": "send_dm reaches at most 3 different people per request; for 4 or more "
            "people use send_dm_many (it asks for a confirmation first), or post_message to a "
            "channel",
        }
    return None


def bulk_refusal(text: str, count: int, may_bulk: bool) -> dict | None:
    """The hard limits of a bulk DM (`send_dm_many`) to `count` people other than the requester:
    nobody, over MAX_DM_BULK, money / credential text, or over DM_EXEC_OVER recipients without
    being an executive / manager (`may_bulk`)."""
    if count < 1:
        return {"error": "no_recipients"}
    if SENSITIVE_DM_RE.search(text or ""):
        return {
            "error": "refused_sensitive",
            "reason": "money transfer or credential request to other people; decline politely "
            "and suggest they contact those people directly",
        }
    if count > MAX_DM_BULK:
        return {
            "error": "too_many_recipients",
            "count": count,
            "max": MAX_DM_BULK,
            "reason": f"at most {MAX_DM_BULK} people per bulk DM; suggest post_message to a channel",
        }
    if count > DM_EXEC_OVER and not may_bulk:
        return {
            "error": "needs_executive",
            "count": count,
            "reason": f"DMing more than {DM_EXEC_OVER} people is for executives / server managers; "
            "suggest post_message to a channel instead",
        }
    return None


def name_list(names: list[str], lang: str, shown: int = 10) -> str:
    """ "A, B, ... and N more" with the first `shown` names."""
    head = ", ".join(names[:shown])
    rest = len(names) - shown
    if rest <= 0:
        return head
    return f"{head} {'และอีก' if lang != 'en' else 'and'} {rest} {'คน' if lang != 'en' else 'more'}"


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
