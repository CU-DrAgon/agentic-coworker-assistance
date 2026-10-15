# ruff: noqa: F811, F401
"""Void (soft delete), duplicate detection and groups, the agent tools and reply-to-void."""

import sqlite3
from types import SimpleNamespace

import pytest
from conftest import ALICE, BOB, EXEC_ID, GUILD, OTHER_GUILD, mk, scripted, text_resp
from test_bot_multitenant import BOSS, Msg, bot, member, world

from salid.digest import build_daily_digest
from salid.extract import parse_void_command
from salid.models import Receipt
from salid.qa import CAPABILITIES_PROMPT, LedgerTools, system_prompt
from salid.store import SQLiteStore, looks_duplicate, vendors_similar

ALL = ("2000-01-01", "2100-01-01")


def rc(store, n, vendor, day, total, *, uid=ALICE, name="Alice", guild=GUILD, **kw):
    return store.add_receipt(
        Receipt(
            guild_id=guild,
            submitter_id=uid,
            submitter_name=name,
            sha256=kw.pop("sha", f"s{n}"),
            vendor=vendor,
            receipt_date=day,
            effective_date=day,
            total=total,
            created_at=day,
            confidence=kw.pop("confidence", 0.9),
            channel_id=kw.pop("channel_id", 5),
            message_id=kw.pop("message_id", n),
            **kw,
        )
    )


@pytest.fixture
def pair(store):
    """The live case: one NVIDIA invoice as a PDF and as a screenshot, plus an unrelated bill."""
    a = rc(
        store, 1, "NVIDIA Corporation", "2026-10-10", 5000.0, original_uri="gs://b/x/1.pdf",
        confidence=0.95,
    )  # fmt: skip
    b = rc(store, 2, "NVIDIA Corp.", "2026-10-10", 5000.0, uid=BOB, name="Bob", confidence=0.8)
    c = rc(store, 3, "Makro", "2026-10-11", 700.0)
    return a, b, c


# ------------------------------------------------------------------ detection rule
def test_vendor_similarity():
    assert vendors_similar("NVIDIA Corporation", "NVIDIA Corp.")
    assert vendors_similar("7-Eleven", "7 eleven")
    assert vendors_similar("Makro Pro Rangsit", "makro")
    assert not vendors_similar("Makro", "Lotus")
    assert not vendors_similar(None, "Makro")


def test_possible_duplicate_positive_and_links(store, pair):
    a, b, _ = pair
    new = Receipt(
        guild_id=GUILD, submitter_id=1, submitter_name="x", sha256="new", vendor="nvidia corp",
        receipt_date="2026-10-12", effective_date="2026-10-12", total=5000.0,
    )  # fmt: skip
    assert store.find_possible_duplicate(new).id == a.id  # oldest match, 2 days apart


def test_possible_duplicate_by_invoice_number(store):
    a = rc(store, 1, "บริษัท เอ", "2026-10-10", 100.0, invoice_no="INV-001")
    new = Receipt(
        guild_id=GUILD, submitter_id=1, submitter_name="x", sha256="n", vendor="Totally Different",
        receipt_date="2026-10-10", effective_date="2026-10-10", total=100.0, invoice_no="inv 001",
    )  # fmt: skip
    assert store.find_possible_duplicate(new).id == a.id


def test_possible_duplicate_negatives(store, pair):
    base = dict(guild_id=GUILD, submitter_id=1, submitter_name="x", total=5000.0)

    def probe(**kw):
        d = {**base, "sha256": "p", "vendor": "NVIDIA Corporation", **kw}
        d.setdefault("receipt_date", "2026-10-10")
        d["effective_date"] = d["receipt_date"]
        return store.find_possible_duplicate(Receipt(**d))

    assert probe(receipt_date="2026-10-20") is None  # same vendor and amount, 10 days apart
    assert probe(currency="USD") is None
    assert probe(total=5000.5) is None
    assert probe(vendor="Other Co") is None
    assert probe(guild_id=OTHER_GUILD) is None


def test_voided_receipts_are_not_duplicate_candidates(store, pair):
    a, b, _ = pair
    store.set_status(a.id, "void", "x", "me")
    new = Receipt(
        guild_id=GUILD, submitter_id=1, submitter_name="x", sha256="n", vendor="NVIDIA Corp",
        receipt_date="2026-10-10", effective_date="2026-10-10", total=5000.0,
    )  # fmt: skip
    assert store.find_possible_duplicate(new).id == b.id


def test_duplicate_groups(store, pair):
    a, b, c = pair
    rc(store, 4, "NVIDIA Corp", "2026-10-09", 5000.0, uid=BOB, name="Bob")  # a third copy
    rc(store, 5, "NVIDIA Corp", "2026-10-10", 5000.0, guild=OTHER_GUILD)  # another tenant
    (g,) = store.duplicate_groups(GUILD)
    assert [r.id for r in g] == [4, 1, 2]  # oldest first
    assert store.duplicate_groups(GUILD, "2026-11-01", "2026-11-30") == []
    assert store.duplicate_groups(GUILD, submitter_id=ALICE) == []  # copies are Bob's
    assert [[r.id for r in g] for g in store.duplicate_groups(GUILD, submitter_id=BOB)] == [[4, 2]]


def test_duplicate_groups_follow_stored_links_and_skip_void(store):
    a = rc(store, 1, "Foo", "2026-10-01", 10.0)
    b = rc(store, 2, "Bar", "2026-10-20", 99.0, possible_duplicate_of=a.id)
    assert [[r.id for r in g] for g in store.duplicate_groups(GUILD)] == [[1, 2]]
    store.set_status(b.id, "void", "d", "x")
    assert store.duplicate_groups(GUILD) == []


# ------------------------------------------------------------------ void in every aggregate
def test_void_is_excluded_everywhere_and_reversible(store, pair, settings):
    a, b, c = pair
    assert store.summarize(GUILD, *ALL)["total"] == 10700.0
    v = store.set_status(b.id, "void", "duplicate", "Boss (1)")
    assert (v.status, v.void_reason, v.voided_by) == (
        "void",
        "duplicate",
        "Boss (1)",
    ) and v.voided_at
    assert store.summarize(GUILD, *ALL)["total"] == 5700.0
    assert store.summarize(GUILD, *ALL, "submitter", submitter_id=BOB)["count"] == 0
    assert [r.id for r in store.list_receipts(GUILD, *ALL)] == [3, 1]
    assert {r.id for r in store.list_receipts(GUILD, *ALL, include_void=True)} == {1, 2, 3}
    assert b.id not in [r.id for r in store.all_receipts(GUILD, *ALL)]
    assert b.id in [r.id for r in store.all_receipts(GUILD, *ALL, include_void=True)]
    assert b.id not in [r.id for r in store.recent(GUILD)]
    assert b.id in [r.id for r in store.recent(GUILD, include_void=True)]
    store.update_receipt(b.id, {"confidence": 0.1})
    assert b.id not in [r.id for r in store.flagged(GUILD)]
    assert b.id in [r.id for r in store.flagged(GUILD, include_void=True)]
    r = store.set_status(b.id, "active")
    assert (r.status, r.void_reason, r.voided_at) == ("active", None, None)
    assert store.summarize(GUILD, *ALL)["total"] == 10700.0


def test_void_excluded_from_digest_and_dup_line(store, pair):
    a, b, _ = pair
    text = build_daily_digest(store, GUILD, "2026-10-10")
    assert "฿10,000" in text and "อาจซ้ำ 1 รายการ รวม ฿5,000 รอตรวจ" in text
    en = build_daily_digest(store, GUILD, "2026-10-10", "en")
    assert "Possible duplicates: 1 receipts, ฿5,000 awaiting review" in en
    store.set_status(b.id, "void", "d", "x")
    text = build_daily_digest(store, GUILD, "2026-10-10")
    assert "฿5,000" in text and "รอตรวจ" not in text


def test_hub_activity_excludes_void_and_flags_duplicates(store, pair):
    from meeting_scribe.hub import build_activity

    a, b, c = pair
    items = {i.title: i.status for i in build_activity(store, GUILD, False)}
    assert items["NVIDIA Corp."] == "review" and items["Makro"] == "ok"
    store.set_status(b.id, "void", "d", "x")
    assert {i.title for i in build_activity(store, GUILD, False)} == {"NVIDIA Corporation", "Makro"}


def test_sqlite_migration_adds_void_columns(tmp_path):
    path = str(tmp_path / "old.db")
    with sqlite3.connect(path) as c:
        c.execute(
            "CREATE TABLE receipts (id INTEGER PRIMARY KEY AUTOINCREMENT, guild_id INTEGER NOT NULL "
            "DEFAULT 0, submitter_id INTEGER NOT NULL, submitter_name TEXT NOT NULL, channel_id "
            "INTEGER, message_id INTEGER, reply_message_id INTEGER, sha256 TEXT NOT NULL, vendor "
            "TEXT, receipt_date TEXT, effective_date TEXT NOT NULL, currency TEXT DEFAULT 'THB', "
            "subtotal REAL, vat REAL, total REAL, category TEXT, items TEXT, payment_method TEXT, "
            "confidence REAL, notes TEXT, context TEXT, possible_duplicate_of INTEGER, created_at "
            "TEXT, original_uri TEXT, UNIQUE (guild_id, sha256))"
        )
        c.execute(
            "INSERT INTO receipts (guild_id, submitter_id, submitter_name, sha256, vendor, "
            "effective_date, total, confidence, category, created_at, channel_id, message_id) "
            "VALUES (1000, 1, 'A', 'x', 'Old', '2026-10-01', 9, 1, 'อื่นๆ', '2026-10-01', 0, 0)"
        )
    s = SQLiteStore(path)
    assert s.get(1).status == "active" and s.summarize(1000, *ALL)["total"] == 9
    s.set_status(1, "void", "r", "me")
    assert s.summarize(1000, *ALL)["total"] == 0
    SQLiteStore(path)  # migrating twice is harmless


# ------------------------------------------------------------------ tools
def tools(store, forced=None, actor="Boss"):
    return LedgerTools(store, GUILD, forced_submitter_id=forced, actor=actor)


def test_find_duplicates_tool(store, pair):
    out = tools(store).find_duplicates()
    assert out["group_count"] == 1 and out["total_deduct_fmt"] == "฿5,000"
    g = out["groups"][0]
    assert g["keep"]["id"] == 1 and g["keep"]["file_type"] == "pdf"  # the PDF stays
    assert [d["id"] for d in g["duplicates"]] == [2] and g["deduct_fmt"] == "฿5,000"
    assert g["duplicates"][0]["source"]["jump_url"] == f"https://discord.com/channels/{GUILD}/5/2"
    assert g["duplicates"][0]["submitter"] == "Bob" and out["can_void"] is True
    assert tools(store).find_duplicates("2026-11-01", "2026-11-30")["group_count"] == 0


def test_scoped_user_sees_only_own_duplicates(store, pair):
    assert tools(store, forced=BOB).find_duplicates() == {
        "group_count": 0, "groups": [], "total_deduct_by_currency": {},
        "total_deduct_fmt": "฿0", "can_void": False,
    }  # fmt: skip
    rc(store, 9, "NVIDIA Corp", "2026-10-10", 5000.0, uid=BOB, name="Bob", sha="s9")
    out = tools(store, forced=BOB).find_duplicates()
    assert out["group_count"] == 1
    assert {e["id"] for g in out["groups"] for e in [g["keep"], *g["duplicates"]]} == {2, 4}


def test_void_and_restore_are_exec_only(store, pair):
    assert tools(store, forced=ALICE).void_receipts([2], "dup") == {"error": "not_allowed"}
    assert tools(store, forced=ALICE).restore_receipts([2]) == {"error": "not_allowed"}
    assert store.get(2).status == "active"
    out = tools(store).void_receipts([2], "duplicate of #1")
    assert [r["id"] for r in out["voided"]] == [2] and out["total_delta_fmt"] == "-฿5,000"
    assert store.get(2).voided_by == "Boss" and store.summarize(GUILD, *ALL)["total"] == 5700.0
    back = tools(store).restore_receipts([2, 2, 3])
    assert [r["id"] for r in back["restored"]] == [2] and back["total_delta_fmt"] == "+฿5,000"
    assert back["ignored"] == [{"id": 3, "reason": "not void"}]
    assert store.summarize(GUILD, *ALL)["total"] == 10700.0


def test_void_is_guild_scoped(store, pair):
    other = rc(store, 7, "Zeta", "2026-10-01", 1.0, guild=OTHER_GUILD)
    out = tools(store).void_receipts([other.id, 999], "x")
    assert out["voided"] == [] and len(out["ignored"]) == 2
    assert store.get(other.id).status == "active"
    assert tools(store).restore_receipts([other.id])["restored"] == []


def test_void_refuses_to_void_every_copy(store, pair):
    out = tools(store).void_receipts([1, 2], "dup")
    assert out["voided"] == [] and {r["id"] for r in out["refused"]} == {1, 2}
    assert store.summarize(GUILD, *ALL)["total"] == 10700.0
    one = tools(store).void_receipts([2, 3, 3], "dup")  # 3 is no duplicate: allowed
    assert {r["id"] for r in one["voided"]} == {2, 3}
    assert tools(store).void_receipts([2], "again")["ignored"][0]["reason"] == "already void"
    assert tools(store).void_receipts([1], "x")["voided"][0]["id"] == 1  # the group is gone
    assert tools(store).void_receipts([], "x")["voided"] == []
    assert tools(store).void_receipts([3], " ") == {"error": "reason is required"}


def test_list_tool_marks_void(store, pair):
    t = tools(store)
    t.void_receipts([2], "dup")
    assert t.list_receipts(*ALL)["count"] == 2
    rows = t.list_receipts(*ALL, include_void=True)["receipts"]
    assert next(r for r in rows if r["id"] == 2)["status"] == "void"
    assert "status" not in next(r for r in rows if r["id"] == 1)


def test_agent_exposes_tools_and_prompt_has_confirmation_rule(store, pair, settings):
    from salid.qa import ask

    llm = scripted(text_resp("ok"))
    ask(settings, store, "x", guild_id=GUILD, is_exec=True, user_id=EXEC_ID, model=llm)
    req = llm.requests[0]
    names = {d.name for t in req.config.tools for d in t.function_declarations}
    assert {"find_duplicates", "void_receipts", "restore_receipts"} <= names
    prompt = system_prompt(settings, scoped=False)
    assert "ถามขอคำยืนยัน" in prompt and "ข้อความถัดมา" in prompt and "find_duplicates" in prompt
    assert "กู้คืนได้" in prompt and "บิลที่ส่งซ้ำ" in CAPABILITIES_PROMPT


async def test_void_via_agent_end_to_end(store, pair, settings):
    from conftest import fc_resp

    from salid.qa import ask_async

    llm = scripted(
        fc_resp(("void_receipts", {"receipt_ids": [2], "reason": "dup"})), text_resp("done")
    )
    ans = await ask_async(
        settings, store, "ยืนยัน", guild_id=GUILD, is_exec=True, user_id=EXEC_ID, model=llm
    )
    assert ans.text == "done" and store.get(2).status == "void"
    assert store.get(2).voided_by == f"user {EXEC_ID}"
    llm2 = scripted(
        fc_resp(("void_receipts", {"receipt_ids": [3], "reason": "dup"})), text_resp("no")
    )
    await ask_async(
        settings, store, "ยืนยัน", guild_id=GUILD, is_exec=False, user_id=ALICE, model=llm2
    )
    assert store.get(3).status == "active"


# ------------------------------------------------------------------ reply to void
def test_parse_void_command():
    for t in ("ซ้ำ", "ลบ", "ยกเลิก", "duplicate", "Void", "ซ้ำครับ", "dup!"):
        assert parse_void_command(t) == "void", t
    for t in ("กู้คืน", "restore", "Restore it"):
        assert parse_void_command(t) == "restore", t
    for t in ("ไม่ซ้ำ", "not a duplicate", "ยอดจริง 450", "", "ซ้ำกับใบ 12 ที่ส่งเมื่อวานนี้ตอนบ่ายสาม"):
        assert parse_void_command(t) is None, t


async def test_reply_void_and_restore(bot, world, seeded):
    r = seeded.get(1)  # Alice's Makro receipt
    seeded.set_reply_message(r.id, 555)

    async def reply(author, text):
        m = Msg(world.a, world.a.text_channels[0], author, text)
        m.reference = SimpleNamespace(message_id=555)
        assert await bot._maybe_correction(m) is True
        return m.replies[-1]

    stranger = member(world.a, BOB)
    assert "เฉพาะผู้ส่ง" in await reply(stranger, "ซ้ำ")
    assert seeded.get(1).status == "active"
    assert "หักใบเสร็จ #1" in await reply(member(world.a, ALICE), "ซ้ำ")
    got = seeded.get(1)
    assert got.status == "void" and "u100" in got.voided_by and "(100)" in got.voided_by
    assert "ถูกหักออกไปแล้ว" in await reply(member(world.a, BOSS), "duplicate")
    assert "กู้คืนใบเสร็จ #1" in await reply(member(world.a, BOSS), "กู้คืน")
    assert seeded.get(1).status == "active"
    assert "ยังนับอยู่" in await reply(member(world.a, BOSS), "restore")
