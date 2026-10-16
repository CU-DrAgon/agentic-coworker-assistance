"""Copy a receipt posted in the wrong channel to the guild's receipt home channel.

Salid has no Manage Messages by design, so the user's original message stays; the copy (files
re-uploaded from the bytes already read for intake, a header with a jump link and the usual
summary) becomes the receipt's reply message, so corrections / voiding by reply work there.

Home channel: `receipt_home_channel_id` from the guild settings; when unset, the first
configured receipt channel Salid can send in, else the channel with the most receipts recorded in
the last 30 days (learned from the ledger, never from channel names), else None.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterable, Mapping

import discord

from .i18n import t

logger = logging.getLogger(__name__)
MAX_FILES = 4
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TEXT = 1900
LEARN_MIN_RECEIPTS = 3  # receipts in the last 30 days that make a channel a "receipt channel"
LEARN_DAYS = 30


def learned_channels(counts: Mapping[int, int]) -> set[int]:
    """Channels that took at least LEARN_MIN_RECEIPTS receipts in the last LEARN_DAYS days."""
    return {int(c) for c, n in counts.items() if n >= LEARN_MIN_RECEIPTS}


def can_post(channel, me) -> bool:
    """Salid can see the channel, send messages and attach files there."""
    if channel is None or me is None or not hasattr(channel, "send"):
        return False
    perms = channel.permissions_for(me)
    return bool(perms.view_channel and perms.send_messages and perms.attach_files)


def home_channel(
    guild,
    settings,
    env_receipt_ids: Iterable[int],
    counts: Mapping[int, int],
):
    """The channel receipts are copied to, or None when there is no usable one."""
    me = guild.me
    if settings.receipt_home_channel_id:  # explicit choice: used as is, or nothing
        ch = guild.get_channel(int(settings.receipt_home_channel_id))
        return ch if can_post(ch, me) else None
    ordered = [int(x) for x in settings.receipt_channel_ids] + sorted(env_receipt_ids)
    for cid in ordered:
        ch = guild.get_channel(cid)
        if can_post(ch, me):
            return ch
    for cid, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if n < 1:
            break
        ch = guild.get_channel(int(cid))
        if can_post(ch, me):
            return ch
    return None


async def forward_receipt(
    *,
    guild,
    settings,
    env_receipt_ids: set[int],
    counts: Mapping[int, int],
    source,
    author,
    jump_url: str | None,
    files: list[tuple[str, bytes]],
    text: str | None,
    lang: str,
):
    """Post the copy and return the sent message (its `.channel` is the home), or None when
    nothing was forwarded (switch off, already in a receipt channel, no usable home channel,
    or the send failed)."""
    if not settings.receipt_forward:
        return None
    if source is not None and (
        source.id in env_receipt_ids
        or str(source.id) in settings.receipt_channel_ids
        or source.id in learned_channels(counts)
    ):
        return None
    home = home_channel(guild, settings, env_receipt_ids, counts)
    if home is None or (source is not None and home.id == source.id):
        return None
    if source is None:  # a DM: no channel or link to show
        header = t("fwd_header_dm", lang, user=author.mention)
    else:
        header = t("fwd_header", lang, user=author.mention, channel=source.name, url=jump_url or "")
    body = f"{header}\n{text}" if text else header
    uploads = [
        discord.File(io.BytesIO(data), filename=name)
        for name, data in files[:MAX_FILES]
        if len(data) <= MAX_FILE_BYTES
    ]
    try:
        return await home.send(
            body[:MAX_TEXT], files=uploads, allowed_mentions=discord.AllowedMentions.none()
        )
    except discord.HTTPException:
        logger.warning("could not forward a receipt to channel %s", home.id, exc_info=True)
        return None
