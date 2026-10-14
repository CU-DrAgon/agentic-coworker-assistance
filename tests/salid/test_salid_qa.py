import json

from conftest import ALICE, BOB, EXEC_ID, FakeClient, fc_resp, text_resp

from salid.digest import build_daily_digest
from salid.qa import MAX_ROUNDS, LedgerTools, ask


def test_aggregation_math(seeded):
    t = LedgerTools(seeded)
    s = t.spending_summary("2026-10-01", "2026-10-31", "category")
    assert s["total"] == 10370.5 and s["count"] == 4
    assert {g["key"]: g["total"] for g in s["groups"]} == {"วัตถุดิบ": 10120.0, "ค่าเดินทาง": 250.5}
    assert s["total_fmt"] == "฿10,370.50"
    v = t.spending_summary("2026-10-01", "2026-10-31", "vendor")["groups"]
    assert v[0] == {"key": "Makro", "total": 10000.0, "count": 2, "total_fmt": "฿10,000"}
    assert [g["key"] for g in t.spending_summary("2026-10-01", "2026-10-31", "day")["groups"]] == [
        "2026-10-01",
        "2026-10-05",
        "2026-10-14",
    ]
    assert t.spending_summary("2026-10-01", "2026-10-31", "submitter")["groups"][0]["key"] == "Bob"
    assert t.spending_summary("2030-01-01", "2030-01-31")["total"] == 0


def test_compare_periods(seeded):
    c = LedgerTools(seeded).compare_periods(
        "2026-10-01", "2026-10-31", "2026-09-01", "2026-09-30", "category"
    )
    assert c["overall"]["diff"] == 7370.5 and c["overall"]["pct_change_vs_b"] == 245.7
    util = next(g for g in c["groups"] if g["key"] == "สาธารณูปโภค")
    assert util["a"] == 0 and util["b"] == 3000 and util["diff"] == -3000


def test_list_and_flagged(seeded):
    t = LedgerTools(seeded)
    assert t.list_receipts("2026-10-01", "2026-10-31", vendor_contains="mak")["count"] == 2
    assert (
        t.list_receipts("2026-10-01", "2026-10-31", category="ค่าเดินทาง")["receipts"][0]["vendor"]
        == "Grab"
    )
    assert t.list_receipts("2026-01-01", "2026-12-31", limit=2)["count"] == 2
    assert [r["vendor"] for r in t.flagged_receipts()["receipts"]] == ["Ice Co"]


def test_non_exec_scoped_in_every_tool(seeded):
    t = LedgerTools(seeded, forced_submitter_id=ALICE)
    assert t.spending_summary("2026-01-01", "2026-12-31")["total"] == 1250.5
    assert t.spending_summary("2026-01-01", "2026-12-31", "submitter")["groups"] == [
        {"key": "Alice", "total": 1250.5, "count": 2, "total_fmt": "฿1,250.50"}
    ]
    assert {
        r["submitter"]
        for r in t.list_receipts("2026-01-01", "2026-12-31", submitter="Bob")["receipts"]
    } == set()
    assert t.list_receipts("2026-01-01", "2026-12-31", vendor_contains="Makro")["count"] == 1
    c = t.compare_periods("2026-10-01", "2026-10-31", "2026-09-01", "2026-09-30")
    assert c["period_a"]["total"] == 1250.5 and c["period_b"]["total"] == 0
    assert t.flagged_receipts()["count"] == 0  # Bob's low-confidence receipt is invisible to Alice


def test_model_cannot_override_scope(seeded):
    t = LedgerTools(seeded, forced_submitter_id=ALICE)
    r = t.call(
        "spending_summary",
        {
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "submitter_id": BOB,
            "forced_submitter_id": BOB,
        },
    )
    assert r["total"] == 1250.5
    assert "error" in t.call("drop_table", {})


def test_end_to_end_scope_no_leak_to_model(seeded, settings):
    """Whatever the (fake) model asks for, data returned to it contains only the asker's receipts."""
    seen = []

    def model_turn(contents, config):
        seen.append(contents)
        return fc_resp(
            (
                "spending_summary",
                {
                    "start_date": "2026-01-01",
                    "end_date": "2026-12-31",
                    "group_by": "submitter",
                    "submitter_id": BOB,
                },
            )
        )

    client = FakeClient([model_turn, lambda c, cfg: (seen.append(c), text_resp("ok"))[1]])
    ask(client, settings, seeded, "ใครเบิกเยอะสุด", is_exec=False, user_id=ALICE)
    blob = json.dumps(
        [
            str(p.function_response)
            for c in seen[-1]
            for p in (c.parts or [])
            if p.function_response
        ],
        ensure_ascii=False,
    )
    assert "Alice" in blob and "Bob" not in blob and "9000" not in blob and "13370" not in blob
    prompt = client.models.calls[0][2].system_instruction
    assert "ไม่ใช่ผู้บริหาร" in prompt and "2026-10-14" in prompt and "Asia/Bangkok" in prompt


def test_exec_sees_everything(seeded, settings):
    client = FakeClient(
        [
            fc_resp(("spending_summary", {"start_date": "2026-01-01", "end_date": "2026-12-31"})),
            text_resp("รวม ฿13,370.50"),
        ]
    )
    a = ask(client, settings, seeded, "ทั้งหมดเท่าไร", is_exec=True, user_id=EXEC_ID)
    assert a.text == "รวม ฿13,370.50" and a.rounds == 1 and a.tool_calls[0][0] == "spending_summary"
    fr = client.models.calls[1][1][-1].parts[0].function_response.response["result"]
    assert fr["total"] == 13370.5


def test_round_cap(seeded, settings):
    def loop(c, cfg):
        return fc_resp(("flagged_receipts", {}))

    client = FakeClient([loop] * MAX_ROUNDS + [text_resp("สรุป")])
    a = ask(client, settings, seeded, "x", is_exec=True, user_id=EXEC_ID)
    assert a.rounds == MAX_ROUNDS and a.text == "สรุป"
    assert client.models.calls[-1][2].tools is None  # final call has no tools


def test_digest(seeded):
    seeded_day = "2026-10-05"
    d = build_daily_digest(seeded, seeded_day)
    assert "฿9,250.5" in d and "Makro" in d and "วัตถุดิบ" in d and "ค่าเดินทาง" in d
    d2 = build_daily_digest(seeded, "2026-10-14")
    assert "Ice Co" in d2 and "ต้องตรวจสอบ" in d2
    assert "ไม่มีใบเสร็จ" in build_daily_digest(seeded, "2026-01-01")


def test_digest_top3(store):
    from conftest import mk

    for i, (v, t) in enumerate([("A", 1), ("B", 5), ("C", 3), ("D", 4)]):
        mk(store, i, 1, "x", v, "2026-10-01", t)
    d = build_daily_digest(store, "2026-10-01")
    assert "1. B" in d and "2. D" in d and "3. C" in d and "A:" not in d.split("3 อันดับ")[1]


def test_is_exec(settings):
    assert settings.is_exec(EXEC_ID) and not settings.is_exec(5, ["staff"])
    assert settings.is_exec(5, ["ผู้บริหาร"]) and settings.is_exec(5, ["Executive"])
