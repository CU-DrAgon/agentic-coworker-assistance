from conftest import ALICE, GUILD, FakeClient, json_resp

from salid.extract import apply_correction, ingest_attachment, parse_correction_simple
from salid.store import SQLiteStore

EX = {
    "is_receipt": True,
    "vendor": "Makro",
    "receipt_date": "2026-10-12",
    "currency": "THB",
    "subtotal": 420.0,
    "vat": 29.4,
    "total": 449.4,
    "category": "วัตถุดิบ",
    "items": [{"name": "ice", "qty": 2, "amount": 100}],
    "payment_method": "cash",
    "confidence": 0.95,
    "notes": None,
}


def ingest(store, settings, client, data=b"img1", **kw):
    args = dict(
        guild_id=GUILD,
        data=data,
        filename="a.jpg",
        content_type="image/jpeg",
        submitter_id=ALICE,
        submitter_name="Alice",
        channel_id=5,
        message_id=7,
        note="ค่าน้ำแข็ง",
    )
    args.update(kw)
    return ingest_attachment(store, client, settings, **args)


def test_store_roundtrip_and_persist(tmp_path, settings):
    p = str(tmp_path / "x.db")
    r = ingest(SQLiteStore(p), settings, FakeClient([json_resp(EX)]))
    again = SQLiteStore(p).get(r.receipt.id)
    assert again.vendor == "Makro" and again.items[0].name == "ice" and again.context == "ค่าน้ำแข็ง"
    assert again.submitter_id == ALICE and again.created_at


def test_create_confirms_with_check(store, settings):
    r = ingest(store, settings, FakeClient([json_resp(EX)]))
    assert (
        r.status == "created" and r.reaction == "✅" and "฿449.40" in r.reply and "Makro" in r.reply
    )


def test_low_confidence_or_no_total_warns(store, settings):
    r = ingest(store, settings, FakeClient([json_resp({**EX, "confidence": 0.5})]), data=b"a")
    assert r.reaction == "⚠️"
    r = ingest(store, settings, FakeClient([json_resp({**EX, "total": None})]), data=b"b")
    assert r.reaction == "⚠️"


def test_non_receipt_ignored_and_not_stored(store, settings):
    r = ingest(store, settings, FakeClient([json_resp({"is_receipt": False, "confidence": 0.9})]))
    assert (
        r.status == "ignored"
        and r.reply is None
        and store.summarize(GUILD, "2000-01-01", "2100-01-01")["count"] == 0
    )


def test_unsupported_file_ignored_without_gemini(store, settings):
    c = FakeClient([])
    assert (
        ingest(store, settings, c, filename="x.txt", content_type="text/plain").status == "ignored"
    )


def test_sha_dedupe_skips_gemini(store, settings):
    ingest(store, settings, FakeClient([json_resp(EX)]))
    c = FakeClient([])  # would raise if Gemini were called
    r = ingest(store, settings, c)
    assert r.status == "duplicate" and "เคยบันทึก" in r.reply and not c.models.calls
    assert store.summarize(GUILD, "2000-01-01", "2100-01-01")["count"] == 1


def test_likely_duplicate_flagged(store, settings):
    ingest(store, settings, FakeClient([json_resp(EX)]), data=b"one")
    r = ingest(store, settings, FakeClient([json_resp(EX)]), data=b"two")
    assert r.status == "created" and r.receipt.possible_duplicate_of and "อาจซ้ำ" in r.reply
    assert len(store.flagged(GUILD)) == 1


def test_missing_date_uses_submission_date(store, settings):
    r = ingest(store, settings, FakeClient([json_resp({**EX, "receipt_date": None})]))
    assert r.receipt.effective_date == "2026-10-14"


def test_buddhist_year_fixed(store, settings):
    r = ingest(store, settings, FakeClient([json_resp({**EX, "receipt_date": "2569-10-12"})]))
    assert r.receipt.receipt_date == "2026-10-12"


def test_correction_parsing():
    assert parse_correction_simple("ยอดจริง 450") == {"total": 450.0}
    assert parse_correction_simple("หมวด ค่าเดินทาง") == {"category": "ค่าเดินทาง"}
    assert parse_correction_simple("ยอดรวม 1,250.50 หมวด การตลาด") == {
        "total": 1250.5,
        "category": "การตลาด",
    }
    assert parse_correction_simple("วันที่ 2569-10-01") == {"receipt_date": "2026-10-01"}
    assert parse_correction_simple("ร้าน ตลาดสด") == {"vendor": "ตลาดสด"}
    assert parse_correction_simple("หมวด ไม่มีจริง") == {}
    assert parse_correction_simple("ขอบคุณครับ") == {}


def test_apply_correction_updates_and_clears_flag(store, settings):
    r = ingest(store, settings, FakeClient([json_resp({**EX, "confidence": 0.3})])).receipt
    msg = apply_correction(store, None, settings, r, "ยอดจริง 450")
    got = store.get(r.id)
    assert got.total == 450 and got.confidence == 1.0 and "฿450" in msg
    assert store.summarize(GUILD, "2026-10-01", "2026-10-31")["total"] == 450
    apply_correction(store, None, settings, r, "วันที่ 2026-09-30")
    assert store.get(r.id).effective_date == "2026-09-30"


def test_correction_llm_fallback(store, settings):
    r = ingest(store, settings, FakeClient([json_resp(EX)])).receipt
    c = FakeClient([json_resp({"total": None, "category": "การตลาด"})])
    apply_correction(store, c, settings, r, "น่าจะเป็นค่าโฆษณานะ")
    assert store.get(r.id).category == "การตลาด"


def test_correction_not_understood(store, settings):
    r = ingest(store, settings, FakeClient([json_resp(EX)])).receipt
    assert "ไม่เข้าใจ" in apply_correction(store, FakeClient([json_resp({})]), settings, r, "555")


def test_reply_message_lookup(store, settings):
    r = ingest(store, settings, FakeClient([json_resp(EX)])).receipt
    store.set_reply_message(r.id, 999)
    assert store.get_by_reply_message(999).id == r.id and store.get_by_reply_message(1) is None


def test_recent_returns_newest_first_and_respects_limit(seeded):
    assert [r.id for r in seeded.recent(GUILD, 3)] == [5, 4, 3]
    assert len(seeded.recent(GUILD)) == 5
    assert len(seeded.recent(GUILD, 0)) == 1  # limit is clamped to at least 1


# ---------------------------------------------------------------- reply language
def test_receipt_replies_follow_the_language(store, settings):
    th = ingest(store, settings, FakeClient([json_resp(EX)]))
    en = ingest(store, settings, FakeClient([json_resp(EX)]), data=b"img2", lang="en")
    assert "บันทึกแล้ว" in th.reply and "ร้าน:" in th.reply
    assert "Saved" in en.reply and "Vendor:" in en.reply and "บันทึกแล้ว" not in en.reply
    dup = ingest(store, settings, FakeClient([]), data=b"img2", lang="en")
    assert dup.status == "duplicate" and "already recorded" in dup.reply
    assert "เคยบันทึกแล้ว" in ingest(store, settings, FakeClient([]), data=b"img2").reply
    unclear = ingest(
        store, settings, FakeClient([json_resp({**EX, "confidence": 0.3})]), data=b"img3", lang="en"
    )
    assert "Please check" in unclear.reply and "Reply to this message" in unclear.reply


def test_corrections_reply_in_english(store, settings):
    r = ingest(store, settings, FakeClient([json_resp(EX)])).receipt
    msg = apply_correction(store, None, settings, r, "total 450", "en")
    assert "Receipt #" in msg and "total=฿450" in msg and "Now:" in msg
    assert "did not understand" in apply_correction(
        store, FakeClient([json_resp({})]), settings, r, "555", "en"
    )
