#!/usr/bin/env python3
"""Copy a local Salid SQLite ledger into Firestore (merge, idempotent, dry-run by default).

    uv run python scripts/salid_migrate_sqlite_to_firestore.py --db ~/.salid-dev/ledger.db [--apply]

Policy: Firestore is the shared store of the company. Records that exist only in SQLite are added;
records that already exist in Firestore are kept (messages, summaries, receipts, documents) except the
guild settings document, where the SQLite fields are merged over the Firestore ones (the local bot's
settings are the ones the owner tuned). Nothing is deleted. Counts only are printed, never contents.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import sys
from collections import Counter

from google.api_core.exceptions import AlreadyExists
from google.cloud import firestore


def rows(c: sqlite3.Connection, sql: str) -> list[sqlite3.Row]:
    c.row_factory = sqlite3.Row
    try:
        return list(c.execute(sql))
    except sqlite3.OperationalError:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.expanduser("~/.salid-dev/ledger.db"))
    ap.add_argument(
        "--project", default=os.environ.get("SALID_GCP_PROJECT", "meeting-scribe-hackathon")
    )
    ap.add_argument("--apply", action="store_true", help="write to Firestore (default: dry run)")
    ap.add_argument("--receipts-collection", default="salid_receipts")
    ap.add_argument("--guild-collection", default="salid_guilds")
    a = ap.parse_args()
    if not os.path.exists(a.db):
        print("no such db", a.db)
        return 2
    db = firestore.Client(project=a.project)
    c = sqlite3.connect(a.db)
    added, kept, merged = Counter(), Counter(), Counter()

    def create(ref, data: dict, label: str) -> None:
        if a.apply:
            try:
                ref.create(data)
                added[label] += 1
            except AlreadyExists:
                kept[label] += 1
        else:  # dry run: count rows only (no Firestore reads; existing docs are skipped on apply)
            added[label] += 1

    # 1. DocStore kinds (tasks, projects, profiles, reminders, minutes, ...): collection = kind
    for r in rows(c, "SELECT kind, guild_id, key, doc FROM salid_docs"):
        doc = json.loads(r["doc"])
        ref = db.collection(r["kind"]).document(f"{r['guild_id']}_{r['key']}")
        create(ref, {**doc, "guild_id": r["guild_id"], "_key": r["key"]}, r["kind"])

    # 2. Guild settings: merge SQLite fields over the Firestore document
    for r in rows(c, "SELECT collection, id, doc FROM salid_guild_docs"):
        doc = json.loads(r["doc"])
        ref = db.collection(r["collection"]).document(str(r["id"]))
        if a.apply:
            ref.set(doc, merge=True)
        merged[r["collection"]] += 1

    # 3. Secretary docs (active meetings, scheduled meetings)
    for r in rows(c, "SELECT collection, id, guild_id, data FROM secretary_docs"):
        data = json.loads(r["data"])
        create(db.collection(r["collection"]).document(str(r["id"])), data, r["collection"])

    # 4. Team memory: raw messages, day summaries, channel / guild state
    for r in rows(c, "SELECT * FROM salid_messages"):
        d = dict(r)
        create(
            db.collection("salid_messages").document(f"{d['guild_id']}_{d['message_id']}"),
            d,
            "salid_messages",
        )
    for r in rows(c, "SELECT * FROM salid_day_summaries"):
        d = dict(r)
        for k in ("summary", "vec_hashes"):
            if isinstance(d.get(k), str):
                with contextlib.suppress(ValueError):
                    d[k] = json.loads(d[k])
        d["final"] = bool(d.get("final"))
        d["dirty"] = bool(d.get("dirty"))
        if d.get("vecs") is None:
            d.pop("vecs", None)
        ref = db.collection("salid_day_summaries").document(
            f"{d['guild_id']}_{d['channel_id']}_{d['local_date']}"
        )
        create(ref, d, "salid_day_summaries")
    for r in rows(c, "SELECT guild_id, channel_id, doc FROM salid_memory_channels"):
        create(
            db.collection("salid_memory_channels").document(f"{r['guild_id']}_{r['channel_id']}"),
            json.loads(r["doc"]),
            "salid_memory_channels",
        )
    for r in rows(c, "SELECT guild_id, doc FROM salid_memory_guilds"):
        create(
            db.collection("salid_memory_guilds").document(str(r["guild_id"])),
            json.loads(r["doc"]),
            "salid_memory_guilds",
        )

    # 5. Receipts: keep ids; bump the counter so new receipts never collide
    max_id = 0
    for r in rows(c, "SELECT * FROM receipts"):
        d = dict(r)
        max_id = max(max_id, int(d.get("id") or 0))
        if isinstance(d.get("items"), str):
            with contextlib.suppress(ValueError):
                d["items"] = json.loads(d["items"])
        sha = d.get("sha256")
        if not sha:
            kept["receipts(no sha)"] += 1
            continue
        create(
            db.collection(a.receipts_collection).document(f"{d['guild_id']}_{sha}"),
            d,
            a.receipts_collection,
        )
    if max_id:
        meta = db.collection(f"{a.receipts_collection}_meta").document("counter")
        snap = meta.get()
        last = int((snap.to_dict() or {}).get("last_id", 0)) if snap.exists else 0
        if max_id > last:
            if a.apply:
                meta.set({"last_id": max_id}, merge=True)
            merged[f"{a.receipts_collection}_meta"] += 1

    mode = "APPLIED" if a.apply else "DRY RUN"
    print(f"{mode}: would add / added:")
    for k, v in sorted(added.items()):
        print(f"  {k}: {v}  (kept existing: {kept.get(k, 0)})")
    for k, v in sorted(merged.items()):
        print(f"  {k}: merged {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
