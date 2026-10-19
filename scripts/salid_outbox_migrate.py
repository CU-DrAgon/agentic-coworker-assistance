"""Move the change outbox's old records to one key per org (one-time, after the 2026-10-18 deploy).

    uv run python scripts/salid_outbox_migrate.py            # dry run: counts only
    uv run python scripts/salid_outbox_migrate.py --apply    # move them

Before the re-key every org's central changes sat under guild 0 of DocStore kind
`salid_change_outbox`; the bot and the platform bridges now read only `org_guild(org id)`, so an old
record is invisible until it is moved. Runs `salid.change_outbox.migrate_legacy` on the store from
the `SALID_*` settings (SQLite or Firestore, the same one the bot and the web center use). Each move
is ONE transaction (put under the org, delete at guild 0), so a re-run moves nothing twice; a record
without an org id is skipped and stays. The dry run writes nothing. Prints counts only, never ids,
emails or content. `--apply` writes the store: run it only with a human's approval.
"""

from __future__ import annotations

import argparse
import sys

from salid.change_outbox import migrate_legacy


def main(argv: list[str] | None = None, docs=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--apply", action="store_true", help="move them (default: dry run, counts only)"
    )
    args = ap.parse_args(argv)
    if docs is None:
        from salid.config import Settings
        from salid.coo_store import make_doc_store
        from salid.store import make_store

        settings = Settings()
        docs = make_doc_store(settings, make_store(settings))
    counts = migrate_legacy(docs, apply=args.apply)
    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"{mode}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
