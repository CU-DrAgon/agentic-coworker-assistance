"""Fill the person dossiers (company knowledge) from the day summaries already in the store.

    uv run python scripts/salid_people_backfill.py --guild 123 --days 60            # dry run: counts
    uv run python scripts/salid_people_backfill.py --guild 123 --days 60 --apply    # learn

Runs the same pass the bot runs on every new summary batch (`salid.people_listen`) over the stored
messages of the last N days, through the normal classes (DocStore + memory store from the
`SALID_*` settings: SQLite or Firestore). Nothing from a DM scope or a member's own private Salid
channel is used. The dry run makes no model call and writes nothing; `--apply` makes one cheap-model
call per batch of <= 40 messages. Prints counts only, never message text. Names are taken from the
stored messages' author names.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from salid.config import Settings
from salid.coo_store import make_doc_store
from salid.llm import make_client
from salid.memory_access import is_dm_scope
from salid.people import PeopleRepo
from salid.people_listen import PeopleListener, backfill
from salid.private_chat import PrivateChatRepo
from salid.store import make_store


def authors_of(store, guild_id: int, today: date, days: int) -> dict[int, str]:
    """author id -> last seen name, from the stored messages of the window."""
    from datetime import timedelta

    from salid.memory_store import date_range

    names: dict[int, str] = {}
    for day in date_range((today - timedelta(days=days)).isoformat(), today.isoformat()):
        for cid in store.day_channels(guild_id, day):
            if is_dm_scope(cid):
                continue
            for m in store.messages_after(guild_id, cid, day, None, 5000):
                if m.kind in ("chat", "voice") and m.author_id > 0 and m.author_name:
                    names[m.author_id] = m.author_name
    return names


def run(settings, docs, store, llm, guild_id: int, days: int, apply: bool) -> dict:
    today = settings.now().date()
    people = authors_of(store, guild_id, today, days)
    private = PrivateChatRepo(docs)
    listener = PeopleListener(
        PeopleRepo(docs, clock=settings.now),
        settings,
        llm,
        people_of=lambda gid: people,
        excluded=lambda gid, cid: private.owner_of(gid, cid) is not None,
    )
    out = backfill(listener, store, guild_id, today, days, apply=apply)
    out["people_in_window"] = len(people)
    out["dossiers"] = len(listener.repo.list(guild_id))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--guild", type=int, required=True, help="Discord guild id")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--apply", action="store_true", help="learn (default: dry run, counts only)")
    args = ap.parse_args(argv)
    settings = Settings()
    ledger = make_store(settings)
    docs = make_doc_store(settings, ledger)
    from salid.bot import SecretaryBot

    store = SecretaryBot._default_memory_store(settings, ledger)
    llm = make_client(settings) if args.apply else None
    counts = run(settings, docs, store, llm, args.guild, max(1, min(args.days, 400)), args.apply)
    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"{mode}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
