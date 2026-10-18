"""Backfill: archive the ORIGINAL documents already posted in a Discord server as real files.

Before 17 Oct 2026 the bot only wrote a text note about each PDF / Word / Excel / PowerPoint it
read, so the command center listed them as "knowledge" with nothing to download. This walks the
server's text channels (the ones the bot can read), downloads each document attachment still on
Discord and stores it with `salid.files.FileArchive` (same record as the live path; duplicates
by content are skipped), so they show up under Documents -> Files with a download button.

    SALID_STORE=firestore SALID_RECEIPT_BUCKET=<bucket> \\
      uv run python scripts/salid_backfill_files.py --guild 1548606556583501945 [--days 365] [--dry-run]

Reads the bot token from `.env` (SALID_DISCORD_TOKEN) in-process, never prints it. Needs the
same GCP access as the bot (ADC). Images are left to the receipt pipeline; this is documents only.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

DOC_EXT = {"pdf", "doc", "docx", "xls", "xlsx", "csv", "ppt", "pptx", "txt", "md"}
MAX_BYTES = 20 * 1024 * 1024


def _token() -> str:
    tok = os.environ.get("SALID_DISCORD_TOKEN", "")
    env = Path(".env")
    if not tok and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("SALID_DISCORD_TOKEN="):
                tok = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not tok:
        raise SystemExit("SALID_DISCORD_TOKEN not found (env or .env)")
    return tok


async def run(guild_id: int, days: int, dry: bool, limit_per_channel: int) -> None:
    import discord

    from salid.config import Settings
    from salid.coo_store import make_doc_store
    from salid.files import FileArchive
    from salid.library import make_blobs
    from salid.store import make_store

    s = Settings.from_env()
    if not s.receipt_bucket and not dry:
        raise SystemExit("SALID_RECEIPT_BUCKET is not set: files would land on local disk")
    store = make_store(s)
    archive = FileArchive(make_doc_store(s, store), make_blobs(s))
    since = datetime.now(UTC) - timedelta(days=days)

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    totals = {"seen": 0, "saved": 0, "skipped": 0, "channels": 0}

    @client.event
    async def on_ready():
        try:
            guild = client.get_guild(guild_id) or await client.fetch_guild(guild_id)
            for ch in guild.text_channels:
                me = guild.me
                perms = ch.permissions_for(me) if me else None
                if perms and not (perms.view_channel and perms.read_message_history):
                    continue
                totals["channels"] += 1
                try:
                    async for m in ch.history(
                        limit=limit_per_channel, after=since, oldest_first=True
                    ):
                        for att in m.attachments:
                            ext = (
                                att.filename.rsplit(".", 1)[-1] if "." in att.filename else ""
                            ).lower()
                            if ext not in DOC_EXT or (att.size or 0) > MAX_BYTES:
                                continue
                            totals["seen"] += 1
                            if dry:
                                print(f"would archive #{ch.name}: {att.filename} ({att.size} B)")
                                continue
                            data = await att.read()
                            did = await asyncio.to_thread(
                                archive.save,
                                guild.id,
                                filename=att.filename,
                                data=data,
                                mime=(att.content_type or "application/octet-stream").split(";")[0],
                                by=m.author.display_name,
                                channel_id=ch.id,
                                channel_name=ch.name,
                                source_url=m.jump_url,
                                summary="",
                                tag=ext,
                                text="",
                            )
                            if did:
                                totals["saved"] += 1
                                print(f"archived #{ch.name}: {att.filename} -> {did}")
                            else:
                                totals["skipped"] += 1
                except discord.Forbidden:
                    continue
        finally:
            print("done:", totals)
            await client.close()

    await client.start(_token())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--guild", type=int, required=True)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--limit", type=int, default=2000, help="messages per channel")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    asyncio.run(run(a.guild, a.days, a.dry_run, a.limit))


if __name__ == "__main__":
    main()
