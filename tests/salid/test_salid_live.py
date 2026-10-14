import io
import time

import pytest
from PIL import Image, ImageDraw, ImageFont

from salid.config import Settings
from salid.extract import extract_receipt
from salid.llm import make_client

pytestmark = pytest.mark.live


def make_receipt_png() -> bytes:
    img = Image.new("RGB", (520, 560), "white")
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.load_default(size=22)
    except TypeError:
        f = ImageFont.load_default()
    lines = [
        "MAKRO SUKHUMVIT BRANCH",
        "TAX INVOICE / RECEIPT",
        "Date: 12/10/2026",
        "-" * 32,
        "Ice 10kg x 2        100.00",
        "Pork belly 3kg      540.00",
        "Cooking oil 5L      325.00",
        "-" * 32,
        "Subtotal           965.00",
        "VAT 7%             67.55",
        "TOTAL THB        1032.55",
        "Paid: CASH",
    ]
    for i, t in enumerate(lines):
        d.text((20, 20 + i * 38), t, fill="black", font=f)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_live_extract():
    s = Settings.from_env()
    t = time.monotonic()
    ex = extract_receipt(make_client(s), s, make_receipt_png(), "image/png", "ค่าวัตถุดิบ สาขา 2")
    print(f"\n{time.monotonic() - t:.1f}s\n{ex.model_dump_json(indent=2, ensure_ascii=False)}")
    assert ex.is_receipt and "makro" in (ex.vendor or "").lower()
    assert ex.total == pytest.approx(1032.55, abs=0.01)
    assert ex.receipt_date == "2026-10-12"
    assert ex.category == "วัตถุดิบ"


def test_live_firestore_store_and_gcs_archive():
    """Real Firestore `(default)` + the existing audio bucket, in throwaway names only.

    Writes only to `salid_receipts_test_<random>` (+ its `_meta` counter) and to
    `salid-receipts/test-<random>/` in the bucket; everything created is deleted in `finally`.
    """
    import uuid

    from google.cloud import firestore

    from salid.archive import ReceiptArchive
    from salid.firestore_store import FirestoreStore
    from salid.models import Receipt

    s = Settings.from_env()
    token = uuid.uuid4().hex[:10]
    coll = f"salid_receipts_test_{token}"
    client = firestore.Client(project=s.gcp_project)
    store = FirestoreStore(client=client, collection=coll)
    bucket = "meeting-scribe-hackathon-audio"
    arch = ReceiptArchive(bucket, f"salid-receipts/test-{token}/", project=s.gcp_project)
    uri = None
    try:

        def mk(n, uid, name, vendor, day, total, conf=0.9):
            return store.add_receipt(
                Receipt(
                    submitter_id=uid, submitter_name=name, sha256=f"{token}-{n}", vendor=vendor,
                    receipt_date=day, effective_date=day, total=total, category="วัตถุดิบ",
                    confidence=conf, created_at=day,
                )
            )  # fmt: skip

        a = mk(1, 100, "Alice", "Makro", "2026-10-01", 1000.0)
        b = mk(2, 200, "Bob", "Makro", "2026-10-05", 9000.0)
        c = mk(3, 200, "Bob", "Ice", "2026-10-14", 120.0, conf=0.4)
        assert (a.id, b.id, c.id) == (1, 2, 3)  # transactional counter
        assert store.get_by_sha(f"{token}-2").id == 2 and store.get(3).vendor == "Ice"
        with pytest.raises(ValueError):
            mk(2, 200, "Bob", "Makro", "2026-10-05", 9000.0)  # same sha -> rejected
        assert store.summarize("2026-10-01", "2026-10-31")["total"] == 10120.0
        assert store.summarize("2026-10-01", "2026-10-31", submitter_id=100)["total"] == 1000.0
        assert [r.vendor for r in store.flagged()] == ["Ice"]
        assert store.find_possible_duplicate(a.model_copy(update={"sha256": "x"})).id == 1
        store.set_reply_message(1, 42)
        assert store.get_by_reply_message(42).id == 1
        assert store.update_receipt(1, {"total": 5.0}).total == 5.0

        uri = arch.save(f"{token}-obj", b"hello", "application/pdf")
        assert uri == f"gs://{bucket}/salid-receipts/test-{token}/{token}-obj.pdf"
        blob = arch.client.bucket(bucket).blob(uri.split(f"{bucket}/", 1)[1])
        assert blob.download_as_bytes() == b"hello"
    finally:
        for name in (coll, f"{coll}_meta"):
            for snap in client.collection(name).stream():
                snap.reference.delete()
        if uri:
            arch.delete(uri)
