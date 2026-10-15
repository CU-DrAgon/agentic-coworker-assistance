"""Guild A can never see guild B's receipts: store, dedupe, Q&A, deep-task export, digest, migration."""

import json
import sqlite3
from pathlib import Path

import pytest
from conftest import (
    ALICE,
    EXEC_ID,
    GUILD,
    OTHER_GUILD,
    FakeClient,
    fc_resp,
    json_resp,
    mk,
    scripted,
    text_resp,
)
from test_deep_task import FakeRun
from test_salid_store_extract import EX, ingest

from salid.config import Settings
from salid.deep_task import prepare_workspace, run_deep_task
from salid.digest import build_daily_digest
from salid.models import Receipt
from salid.qa import LedgerTools, ask
from salid.store import SQLiteStore

ALL = ("2000-01-01", "2100-01-01")
SECRETS = ("Zeta Corp", "7777")  # only guild B has these


@pytest.fixture
def two_guilds(store):
    mk(store, 1, ALICE, "Alice", "Makro", "2026-10-05", 100.0)
    mk(store, 2, ALICE, "Alice", "Grab", "2026-10-06", 50.0, conf=0.3)
    # guild B: same user id, a same-sha file, and a vendor/date/total twin of A's first receipt
    mk(store, 3, ALICE, "Alice", "Zeta Corp", "2026-10-05", 7777.0, guild=OTHER_GUILD)
    mk(store, 4, ALICE, "Alice", "Makro", "2026-10-05", 100.0, sha="sha1", guild=OTHER_GUILD)
    mk(store, 5, ALICE, "Alice", "Ice Zeta", "2026-10-07", 1.0, conf=0.2, guild=OTHER_GUILD)
    return store


def vendors(rows):
    return sorted(r.vendor for r in rows)


def test_every_read_is_scoped_to_its_guild(two_guilds):
    s = two_guilds
    assert s.summarize(GUILD, *ALL)["total"] == 150.0 and s.summarize(GUILD, *ALL)["count"] == 2
    assert s.summarize(OTHER_GUILD, *ALL)["total"] == 7878.0
    assert s.summarize(999, *ALL)["count"] == 0  # unknown guild: nothing
    for by in ("category", "vendor", "submitter", "day"):
        a = json.dumps(s.summarize(GUILD, *ALL, by)["groups"], ensure_ascii=False)
        assert not any(x in a for x in SECRETS)
    assert vendors(s.list_receipts(GUILD, *ALL, limit=100)) == ["Grab", "Makro"]
    assert vendors(s.list_receipts(GUILD, *ALL, vendor_contains="zeta")) == []
    assert vendors(s.all_receipts(GUILD, *ALL)) == ["Grab", "Makro"]
    assert vendors(s.all_receipts(OTHER_GUILD, *ALL)) == ["Ice Zeta", "Makro", "Zeta Corp"]
    assert vendors(s.recent(GUILD, 50)) == ["Grab", "Makro"]
    assert vendors(s.flagged(GUILD)) == ["Grab"] and vendors(s.flagged(OTHER_GUILD)) == ["Ice Zeta"]
    assert vendors(s.flagged(GUILD, *ALL)) == ["Grab"]


def test_dedupe_is_per_guild(two_guilds):
    s = two_guilds
    assert s.get_by_sha(GUILD, "sha1").id == 1 and s.get_by_sha(OTHER_GUILD, "sha1").id == 4
    assert s.get_by_sha(GUILD, "sha3") is None  # sha3 exists only in guild B
    with pytest.raises(Exception):  # noqa: B017 (sqlite IntegrityError / firestore ValueError)
        mk(s, 9, ALICE, "Alice", "Makro", "2026-10-05", 100.0, sha="sha1", guild=OTHER_GUILD)
    probe = Receipt(
        guild_id=GUILD, submitter_id=1, submitter_name="x", sha256="new", vendor="makro",
        receipt_date="2026-10-05", effective_date="2026-10-05", total=100.0,
    )  # fmt: skip
    assert s.find_possible_duplicate(probe).id == 1  # A's own Makro, never B's twin (id 4)
    other = probe.model_copy(update={"guild_id": 999})
    assert s.find_possible_duplicate(other) is None
    zeta = probe.model_copy(update={"vendor": "Zeta Corp", "total": 7777.0})
    assert s.find_possible_duplicate(zeta) is None  # B's receipt is invisible to A


def test_ingest_stamps_guild_and_same_file_in_two_guilds_is_stored_twice(store, settings):
    a = ingest(store, settings, FakeClient([json_resp(EX)]), guild_id=GUILD)
    assert a.status == "created" and a.receipt.guild_id == GUILD
    again = ingest(store, settings, FakeClient([json_resp(EX)]), guild_id=GUILD)
    assert again.status == "duplicate"
    b = ingest(store, settings, FakeClient([json_resp(EX)]), guild_id=OTHER_GUILD)
    assert b.status == "created" and b.receipt.guild_id == OTHER_GUILD  # not flagged as a dup
    assert b.receipt.possible_duplicate_of is None
    assert store.summarize(GUILD, *ALL)["count"] == 1


def test_qa_tools_cannot_cross_guilds(two_guilds):
    for forced in (None, ALICE):
        t = LedgerTools(two_guilds, GUILD, forced_submitter_id=forced)
        blob = json.dumps(
            [
                t.spending_summary(*ALL, "vendor"),
                t.list_receipts(*ALL, limit=100),
                t.list_receipts(*ALL, vendor_contains="Zeta"),
                t.compare_periods(*ALL, *ALL, group_by="vendor"),
                t.flagged_receipts(),
                # a forged guild_id from the model is dropped (not in any tool signature)
                t.call("spending_summary", {**dict(start_date=ALL[0], end_date=ALL[1]), "guild_id": OTHER_GUILD}),
            ],
            ensure_ascii=False,
        )  # fmt: skip
        assert not any(x in blob for x in SECRETS) and "Ice Zeta" not in blob


def test_ask_end_to_end_is_scoped_even_when_model_asks_for_everything(two_guilds, settings):
    window = {"start_date": ALL[0], "end_date": ALL[1]}
    llm = scripted(
        fc_resp(
            ("spending_summary", {**window, "group_by": "vendor", "guild_id": OTHER_GUILD}),
            ("list_receipts", {**window, "limit": 100}),
            ("flagged_receipts", {"guild_id": OTHER_GUILD}),
        ),
        text_resp("ok"),
    )
    ask(settings, two_guilds, "ทั้งหมด", guild_id=GUILD, is_exec=True, user_id=EXEC_ID, model=llm)
    blob = json.dumps(
        [
            str(p.function_response)
            for req in llm.requests
            for c in req.contents
            for p in c.parts or []
            if p.function_response
        ],
        ensure_ascii=False,
    )
    assert "Makro" in blob and not any(x in blob for x in (*SECRETS, "Ice Zeta"))


def test_deep_task_export_is_guild_scoped(two_guilds, tmp_path):
    s = Settings(gemini_cli_path="/opt/gemini")
    ws = tmp_path / "ws"
    ws.mkdir()
    assert prepare_workspace(ws, two_guilds, s, "t", GUILD) == 2
    exported = "".join(p.read_text(encoding="utf-8") for p in Path(ws).iterdir())
    assert "Makro" in exported and not any(x in exported for x in (*SECRETS, "Ice Zeta"))
    f = FakeRun()
    run_deep_task(s, two_guilds, "t", guild_id=OTHER_GUILD, runner=f)
    assert "Zeta Corp" in f.seen["csv"] and "Grab" not in f.seen["csv"]
    assert f.seen["agg"]["overall"]["count"] == 3


def test_digest_is_per_guild(two_guilds):
    a = build_daily_digest(two_guilds, GUILD, "2026-10-05")
    b = build_daily_digest(two_guilds, OTHER_GUILD, "2026-10-05")
    assert "฿100" in a and "Zeta" not in a and "7777" not in a
    assert "Zeta Corp" in b and "฿7,877" in b
    assert "ไม่มีใบเสร็จ" in build_daily_digest(two_guilds, 999, "2026-10-05")


def test_old_sqlite_ledger_is_migrated_to_guild_zero(tmp_path):
    p = str(tmp_path / "old.db")
    c = sqlite3.connect(p)
    c.executescript(
        "CREATE TABLE receipts (id INTEGER PRIMARY KEY AUTOINCREMENT, submitter_id INTEGER NOT NULL,"
        " submitter_name TEXT NOT NULL, channel_id INTEGER, message_id INTEGER, reply_message_id INTEGER,"
        " sha256 TEXT NOT NULL UNIQUE, vendor TEXT, receipt_date TEXT, effective_date TEXT NOT NULL,"
        " currency TEXT DEFAULT 'THB', subtotal REAL, vat REAL, total REAL, category TEXT, items TEXT,"
        " payment_method TEXT, confidence REAL, notes TEXT, context TEXT, possible_duplicate_of INTEGER,"
        " created_at TEXT, original_uri TEXT);"
        "INSERT INTO receipts (submitter_id, submitter_name, sha256, vendor, effective_date, total,"
        " confidence, items, channel_id, message_id, category, created_at)"
        " VALUES (1, 'Old', 'oldsha', 'Legacy', '2026-10-01', 42.0, 0.9, '[]', 0, 0, 'x', '');"
    )
    c.commit()
    c.close()
    s = SQLiteStore(p)
    assert s.get(1).guild_id == 0 and s.summarize(0, *ALL)["total"] == 42.0
    assert s.summarize(GUILD, *ALL)["count"] == 0
    mk(s, 2, 1, "New", "Shop", "2026-10-02", 1.0, sha="oldsha", guild=GUILD)  # per-guild unique now
    assert SQLiteStore(p).summarize(GUILD, *ALL)["count"] == 1  # reopening is a no-op
