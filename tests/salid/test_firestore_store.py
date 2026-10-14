"""FirestoreStore-specific behaviour (the shared Store contract runs via the `store` fixture)."""

import pytest
from conftest import ALICE, FakeClient, json_resp, mk
from fake_firestore import OfflineFirestoreStore
from test_salid_store_extract import EX, ingest

from salid.archive import ReceiptArchive, make_archive
from salid.config import Settings
from salid.firestore_store import FirestoreStore
from salid.store import SQLiteStore, make_store


@pytest.fixture
def fs():
    return OfflineFirestoreStore()


def test_layout_doc_id_is_sha_and_effective_date_is_iso(fs):
    r = mk(fs, 1, ALICE, "Alice", "Makro", "2026-10-01", 100.0, sha="abc123")
    doc = fs.client.collection("salid_receipts").docs["abc123"]
    assert doc["id"] == r.id == 1 and doc["effective_date"] == "2026-10-01"
    assert set(fs.client.cols) == {"salid_receipts"}  # nothing outside the configured collection


def test_collection_name_configurable():
    s = OfflineFirestoreStore(collection="salid_x")
    mk(s, 1, ALICE, "Alice", "Makro", "2026-10-01", 1.0)
    assert set(s.client.cols) == {"salid_x"} and s.meta_collection == "salid_x_meta"


def test_duplicate_sha_rejected_atomically(fs):
    mk(fs, 1, ALICE, "Alice", "Makro", "2026-10-01", 100.0, sha="same")
    with pytest.raises(ValueError):
        mk(fs, 2, ALICE, "Alice", "Makro", "2026-10-01", 100.0, sha="same")
    assert fs.summarize("2026-01-01", "2026-12-31")["count"] == 1


def test_no_query_needs_a_composite_index(seeded_fs):
    """Server-side filters are only ever on ONE field (range on effective_date, or one equality)."""
    s = seeded_fs
    s.client.queries.clear()
    s.summarize("2026-10-01", "2026-10-31", "category", submitter_id=ALICE)
    s.list_receipts(
        "2026-10-01",
        "2026-10-31",
        category="วัตถุดิบ",
        vendor_contains="m",
        submitter="Al",
        submitter_id=ALICE,
    )
    s.flagged("2026-10-01", "2026-10-31", ALICE)
    s.flagged()
    s.get(1)
    s.find_possible_duplicate(s.get(1).model_copy(update={"sha256": "zz"}))
    for filters in s.client.queries:
        assert len({f.field_path for f in filters}) <= 1, filters


@pytest.fixture
def seeded_fs(fs):
    mk(fs, 1, ALICE, "Alice", "Makro", "2026-10-01", 1000.0)
    mk(fs, 2, 200, "Bob", "Makro", "2026-10-05", 9000.0)
    mk(fs, 3, 200, "Bob", "Ice", "2026-10-14", 120.0, conf=0.4)
    return fs


def test_update_and_reply_lookup(fs):
    r = mk(fs, 1, ALICE, "Alice", "Makro", "2026-10-01", 100.0)
    fs.set_reply_message(r.id, 555)
    assert fs.get_by_reply_message(555).id == r.id and fs.get_by_reply_message(1) is None
    u = fs.update_receipt(r.id, {"total": 450.0, "receipt_date": "2026-09-30"})
    assert u.total == 450.0 and u.effective_date == "2026-09-30"
    assert fs.get(r.id).effective_date == "2026-09-30"
    with pytest.raises(ValueError):
        fs.update_receipt(r.id, {"submitter_id": 9})


def test_ingest_works_on_firestore(settings, fs):
    res = ingest(fs, settings, FakeClient([json_resp(EX)]))
    assert res.status == "created" and fs.get(res.receipt.id).vendor == "Makro"


def test_make_store_selects_backend(tmp_path):
    s = Settings(store="sqlite", db_path=str(tmp_path / "a.db"))
    assert isinstance(make_store(s), SQLiteStore)
    f = make_store(Settings(store="firestore", firestore_collection="salid_y"))
    assert isinstance(f, FirestoreStore) and f.collection == "salid_y"  # lazy: no client yet
    with pytest.raises(ValueError):
        make_store(Settings(store="mongo"))


# --------------------------------------------------------------- receipt archive


class FakeBucketClient:
    def __init__(self):
        self.objects = {}

    def bucket(self, name):
        outer = self

        class B:
            def blob(self, obj):
                class Bl:
                    def upload_from_string(self, data, content_type=None):
                        outer.objects[(name, obj)] = (data, content_type)

                    def delete(self):
                        outer.objects.pop((name, obj))

                return Bl()

        return B()


def test_archive_saves_uri_on_record(settings, fs):
    cli = FakeBucketClient()
    arch = ReceiptArchive("bkt", "salid-receipts", client=cli)
    res = ingest(fs, settings, FakeClient([json_resp(EX)]), archive=arch)
    uri = res.receipt.original_uri
    assert uri == f"gs://bkt/salid-receipts/{res.receipt.sha256}.jpg"
    assert cli.objects[("bkt", f"salid-receipts/{res.receipt.sha256}.jpg")] == (
        b"img1",
        "image/jpeg",
    )
    assert fs.get(res.receipt.id).original_uri == uri
    arch.delete(uri)
    assert not cli.objects


def test_archive_failure_does_not_lose_receipt(settings, store):
    class Boom:
        def save(self, *a):
            raise RuntimeError("gcs down")

    res = ingest(store, settings, FakeClient([json_resp(EX)]), archive=Boom())
    assert res.status == "created" and res.receipt.original_uri is None


def test_archive_disabled_by_default():
    assert make_archive(Settings()) is None
    assert make_archive(Settings(receipt_bucket="b")).bucket_name == "b"


def test_sqlite_ledger_from_before_archiving_is_migrated(tmp_path):
    import sqlite3

    p = str(tmp_path / "old.db")
    c = sqlite3.connect(p)
    c.executescript(
        "CREATE TABLE receipts (id INTEGER PRIMARY KEY AUTOINCREMENT, submitter_id INTEGER NOT NULL,"
        " submitter_name TEXT NOT NULL, channel_id INTEGER, message_id INTEGER, reply_message_id INTEGER,"
        " sha256 TEXT NOT NULL UNIQUE, vendor TEXT, receipt_date TEXT, effective_date TEXT NOT NULL,"
        " currency TEXT DEFAULT 'THB', subtotal REAL, vat REAL, total REAL, category TEXT, items TEXT,"
        " payment_method TEXT, confidence REAL, notes TEXT, context TEXT, possible_duplicate_of INTEGER,"
        " created_at TEXT);"
    )
    c.close()
    s = SQLiteStore(p)
    r = mk(s, 1, ALICE, "Alice", "Makro", "2026-10-01", 1.0)
    assert s.get(r.id).original_uri is None
