"""Fill the person dossiers (company knowledge) from the day summaries already in the store.

    uv run python scripts/salid_people_backfill.py --guild 123 --days 60            # dry run: counts
    uv run python scripts/salid_people_backfill.py --guild 123 --days 60 --apply    # learn

Runs the same pass the bot runs on every new summary batch (`salid.people_listen`) over the stored
messages of the last N days, through the normal classes (DocStore + memory store from the
`SALID_*` settings: SQLite or Firestore). Nothing from a DM scope or a member's own private Salid
channel is used. The dry run makes no model call and writes nothing; `--apply` makes one cheap-model
call per batch of <= 40 messages. Prints counts only, never message text. Names are taken from the
stored messages' author names.

Admin passes (no Discord, no model; counts only; dry run unless `--apply`):

    --rebuild-map           rebuild the company map (org, who-to-ask, key contacts) from the
                            dossiers alone; every evidence channel counts as readable except DM
                            scopes and members' private Salid channels
    --drop-bots --bot-ids 1,2   delete the dossiers of those user ids (bots / the app itself) and
                            remove contacts named Salid / สลิด (or pointing at those ids) from
                            the others
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from salid.company_brain import CompanyRepo
from salid.config import Settings
from salid.coo_store import make_doc_store
from salid.llm import make_client
from salid.memory_access import is_dm_scope
from salid.people import PeopleRepo, is_bot_name, map_from_dossiers
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


def rebuild_map(docs, guild_id: int, apply: bool, clock=None) -> dict:
    """Company map from the dossiers (see `map_from_dossiers`). Saves only with `apply`."""
    repo = PeopleRepo(docs, clock=clock)
    private = PrivateChatRepo(docs)
    dossiers = repo.list(guild_id)
    org, who, contacts = map_from_dossiers(
        dossiers, lambda cid: private.owner_of(guild_id, cid) is not None
    )
    if apply:
        company = CompanyRepo(docs, clock=clock)
        pb = company.get(guild_id)
        pb.org, pb.who_to_ask, pb.key_contacts = org, who, contacts
        pb.org_built_at = (clock or _utcnow)().isoformat(timespec="seconds")
        company.save(guild_id, pb)
    return {
        "dossiers": len(dossiers),
        "departments": len(org),
        "topics": len(who),
        "contacts": len(contacts),
        "saved": int(apply),
    }


def _utcnow():
    from datetime import UTC, datetime

    return datetime.now(UTC)


def drop_bots(docs, guild_id: int, bot_ids: set[int], apply: bool) -> dict:
    """Delete the dossiers of `bot_ids`; remove contacts that name the assistant or point at them."""
    repo = PeopleRepo(docs)
    out = {"dossiers_deleted": 0, "contacts_removed": 0, "dossiers_cleaned": 0}
    for d in repo.list(guild_id, include_bots=True):
        if d.user_id in bot_ids:
            out["dossiers_deleted"] += 1
            if apply:
                repo.delete(guild_id, d.user_id)
            continue
        bad = [c for c in d.contacts if c.user_id in bot_ids or is_bot_name(c.who)]
        if not bad:
            continue
        out["contacts_removed"] += len(bad)
        out["dossiers_cleaned"] += 1
        if apply:
            d.contacts = [c for c in d.contacts if c not in bad]
            repo.put(guild_id, d)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--guild", type=int, required=True, help="Discord guild id")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--apply", action="store_true", help="learn (default: dry run, counts only)")
    ap.add_argument("--rebuild-map", action="store_true", help="rebuild the company map")
    ap.add_argument("--drop-bots", action="store_true", help="delete bot dossiers and contacts")
    ap.add_argument("--bot-ids", default="", help="comma list of bot user ids (for --drop-bots)")
    args = ap.parse_args(argv)
    settings = Settings()
    ledger = make_store(settings)
    docs = make_doc_store(settings, ledger)
    if args.rebuild_map or args.drop_bots:
        mode = "APPLIED" if args.apply else "DRY RUN"
        if args.drop_bots:
            ids = {int(x) for x in args.bot_ids.split(",") if x.strip().isdigit()}
            counts = drop_bots(docs, args.guild, ids, args.apply)
            print(f"{mode} drop-bots: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        if args.rebuild_map:
            counts = rebuild_map(docs, args.guild, args.apply, settings.now)
            print(f"{mode} rebuild-map: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        return 0
    from salid.bot import SecretaryBot

    store = SecretaryBot._default_memory_store(settings, ledger)
    llm = make_client(settings) if args.apply else None
    counts = run(settings, docs, store, llm, args.guild, max(1, min(args.days, 400)), args.apply)
    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"{mode}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
