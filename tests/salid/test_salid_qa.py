import json

from conftest import ALICE, BOB, EXEC_ID, GUILD, fc_resp, scripted, text_resp

from salid.digest import build_daily_digest
from salid.qa import MAX_ROUNDS, LedgerTools, ask


def test_aggregation_math(seeded):
    t = LedgerTools(seeded, GUILD)
    s = t.spending_summary("2026-10-01", "2026-10-31", "category")
    assert s["total"] == 10370.5 and s["count"] == 4
    assert {g["key"]: g["total"] for g in s["groups"]} == {"วัตถุดิบ": 10120.0, "ค่าเดินทาง": 250.5}
    assert s["total_fmt"] == "฿10,370.50"
    v = t.spending_summary("2026-10-01", "2026-10-31", "vendor")["groups"]
    assert v[0] == {
        "key": "Makro",
        "total": 10000.0,
        "totals_by_currency": {"THB": 10000.0},
        "count": 2,
        "total_fmt": "฿10,000",
    }
    assert [g["key"] for g in t.spending_summary("2026-10-01", "2026-10-31", "day")["groups"]] == [
        "2026-10-01",
        "2026-10-05",
        "2026-10-14",
    ]
    assert t.spending_summary("2026-10-01", "2026-10-31", "submitter")["groups"][0]["key"] == "Bob"
    assert t.spending_summary("2030-01-01", "2030-01-31")["total"] == 0


def test_compare_periods(seeded):
    c = LedgerTools(seeded, GUILD).compare_periods(
        "2026-10-01", "2026-10-31", "2026-09-01", "2026-09-30", "category"
    )
    assert c["overall"]["diff"] == 7370.5 and c["overall"]["pct_change_vs_b"] == 245.7
    util = next(g for g in c["groups"] if g["key"] == "สาธารณูปโภค")
    assert util["a"] == 0 and util["b"] == 3000 and util["diff"] == -3000


def test_list_and_flagged(seeded):
    t = LedgerTools(seeded, GUILD)
    assert t.list_receipts("2026-10-01", "2026-10-31", vendor_contains="mak")["count"] == 2
    assert (
        t.list_receipts("2026-10-01", "2026-10-31", category="ค่าเดินทาง")["receipts"][0]["vendor"]
        == "Grab"
    )
    assert t.list_receipts("2026-01-01", "2026-12-31", limit=2)["count"] == 2
    assert [r["vendor"] for r in t.flagged_receipts()["receipts"]] == ["Ice Co"]


def test_non_exec_scoped_in_every_tool(seeded):
    t = LedgerTools(seeded, GUILD, forced_submitter_id=ALICE)
    assert t.spending_summary("2026-01-01", "2026-12-31")["total"] == 1250.5
    assert t.spending_summary("2026-01-01", "2026-12-31", "submitter")["groups"] == [
        {
            "key": "Alice",
            "total": 1250.5,
            "totals_by_currency": {"THB": 1250.5},
            "count": 2,
            "total_fmt": "฿1,250.50",
        }
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
    t = LedgerTools(seeded, GUILD, forced_submitter_id=ALICE)
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


def _tool_results(llm):
    """Every function_response the model was shown, across all requests, as one string."""
    out = []
    for req in llm.requests:
        for c in req.contents:
            out += [str(p.function_response) for p in c.parts or [] if p.function_response]
    return json.dumps(out, ensure_ascii=False)


EVERYTHING = {"start_date": "2026-01-01", "end_date": "2026-12-31"}


def test_non_exec_cannot_see_others_even_if_model_asks_for_everything(seeded, settings):
    """A hostile/confused model asks for all data, other users, and a forged submitter scope."""
    llm = scripted(
        fc_resp(
            ("spending_summary", {**EVERYTHING, "group_by": "submitter", "submitter_id": BOB}),
            ("list_receipts", {**EVERYTHING, "submitter": "Bob", "limit": 100}),
            (
                "list_receipts",
                {**EVERYTHING, "vendor_contains": "Makro", "forced_submitter_id": BOB},
            ),
            (
                "compare_periods",
                {
                    "a_start": "2026-10-01",
                    "a_end": "2026-10-31",
                    "b_start": "2026-09-01",
                    "b_end": "2026-09-30",
                    "submitter_id": 0,
                },
            ),
            ("flagged_receipts", {"submitter_id": BOB}),
        ),
        text_resp("ok"),
    )
    a = ask(
        settings, seeded, "ใครเบิกเยอะสุด", guild_id=GUILD, is_exec=False, user_id=ALICE, model=llm
    )
    blob = _tool_results(llm)
    assert a.text == "ok" and len(a.tool_calls) == 5
    assert "Alice" in blob and "1250.5" in blob
    for secret in ("Bob", "9000", "3000", "13370", "Ice Co", "PEA"):
        assert secret not in blob, secret
    prompt = llm.requests[0].config.system_instruction
    assert "ไม่ใช่ผู้บริหาร" in prompt and "2026-10-14" in prompt and "Asia/Bangkok" in prompt
    # no tool schema offers a submitter scope parameter
    for t in llm.requests[0].config.tools:
        for d in t.function_declarations:
            schema = d.parameters_json_schema or (d.parameters and d.parameters.model_dump()) or {}
            props = set(schema.get("properties") or {})
            assert d.name != "spending_summary" or "group_by" in props  # schema is really read
            assert not {"submitter_id", "forced_submitter_id", "user_id"} & props


def test_exec_sees_everything(seeded, settings):
    llm = scripted(fc_resp(("spending_summary", EVERYTHING)), text_resp("รวม ฿13,370.50"))
    a = ask(
        settings, seeded, "ทั้งหมดเท่าไร", guild_id=GUILD, is_exec=True, user_id=EXEC_ID, model=llm
    )
    assert a.text == "รวม ฿13,370.50" and a.rounds == 1 and a.tool_calls[0][0] == "spending_summary"
    assert "13370.5" in _tool_results(llm)
    assert "ไม่ใช่ผู้บริหาร" not in llm.requests[0].config.system_instruction


def test_tools_are_the_ledger_tools(seeded, settings):
    llm = scripted(text_resp("ok"))
    ask(settings, seeded, "x", guild_id=GUILD, is_exec=True, user_id=EXEC_ID, model=llm)
    names = {d.name for t in llm.requests[0].config.tools for d in t.function_declarations}
    assert names == {
        "spending_summary",
        "list_receipts",
        "compare_periods",
        "flagged_receipts",
        "find_duplicates",
        "void_receipts",
        "restore_receipts",
    }


def test_bad_tool_args_return_error_to_model(seeded, settings):
    llm = scripted(
        fc_resp(("spending_summary", {**EVERYTHING, "group_by": "nonsense"})), text_resp("ขออภัย")
    )
    a = ask(settings, seeded, "x", guild_id=GUILD, is_exec=True, user_id=EXEC_ID, model=llm)
    assert a.text == "ขออภัย" and "bad group_by" in _tool_results(llm)


def test_round_cap(seeded, settings):
    llm = scripted(*[fc_resp(("flagged_receipts", {}))] * MAX_ROUNDS, text_resp("สรุป"))
    a = ask(settings, seeded, "x", guild_id=GUILD, is_exec=True, user_id=EXEC_ID, model=llm)
    assert a.rounds == MAX_ROUNDS and a.text == "สรุป"
    assert len(llm.requests) == MAX_ROUNDS + 1
    assert not llm.requests[-1].config.tools  # the final call has no tools


def test_digest(seeded):
    seeded_day = "2026-10-05"
    d = build_daily_digest(seeded, GUILD, seeded_day)
    assert "฿9,250.5" in d and "Makro" in d and "วัตถุดิบ" in d and "ค่าเดินทาง" in d
    d2 = build_daily_digest(seeded, GUILD, "2026-10-14")
    assert "Ice Co" in d2 and "ต้องตรวจสอบ" in d2
    assert "ไม่มีใบเสร็จ" in build_daily_digest(seeded, GUILD, "2026-01-01")


def test_digest_top3(store):
    from conftest import mk

    for i, (v, t) in enumerate([("A", 1), ("B", 5), ("C", 3), ("D", 4)]):
        mk(store, i, 1, "x", v, "2026-10-01", t)
    d = build_daily_digest(store, GUILD, "2026-10-01")
    assert "1. B" in d and "2. D" in d and "3. C" in d and "A:" not in d.split("3 อันดับ")[1]


def test_is_exec(settings):
    assert settings.is_exec(EXEC_ID) and not settings.is_exec(5, ["staff"])
    assert settings.is_exec(5, ["ผู้บริหาร"]) and settings.is_exec(5, ["Executive"])


def test_system_prompt_language_instruction(settings):
    from salid.qa import system_prompt

    th = system_prompt(settings, scoped=False)
    assert th == system_prompt(settings, scoped=False, language="th")
    assert "พูดภาษาไทยเป็นธรรมชาติ" in th and "\nLanguage:" not in th
    en = system_prompt(settings, scoped=False, language="en")
    assert "ALWAYS reply in English" in en and "พูดภาษาไทยเป็นธรรมชาติ" not in en
    auto = system_prompt(settings, scoped=True, language="auto")
    assert "same language as the user's latest message" in auto and "ไม่ใช่ผู้บริหาร" in auto


def test_system_prompt_lists_capabilities(settings):
    from salid.qa import system_prompt

    for lang in ("th", "en", "auto"):
        p = system_prompt(settings, scoped=True, language=lang)
        for cmd in ("/โหมด", "/mode", "/ภาษา", "/language", "/ความจำ", "/memory", "/join"):
            assert cmd in p
        assert "งานใหญ่:" in p and "ค้นอินเทอร์เน็ตไม่ได้" in p


def test_ask_passes_the_language_to_the_model(seeded, settings):
    llm = scripted(text_resp("Total is ฿1,000"))
    a = ask(
        settings, seeded, "total?", guild_id=GUILD, is_exec=True, user_id=EXEC_ID, model=llm,
        language="en",
    )  # fmt: skip
    assert a.text == "Total is ฿1,000"
    assert "ALWAYS reply in English" in llm.requests[0].config.system_instruction


def test_empty_answer_falls_back_in_the_users_language(seeded, settings):
    def empty(question, language):
        llm = scripted(text_resp(""))
        return ask(
            settings, seeded, question, guild_id=GUILD, is_exec=True, user_id=EXEC_ID, model=llm,
            language=language,
        ).text  # fmt: skip

    assert "ขออภัย" in empty("ยอดเท่าไร", "th")
    assert "Sorry" in empty("total?", "en")
    assert "Sorry" in empty("total?", "auto") and "ขออภัย" in empty("ยอดเท่าไร", "auto")


def test_digest_in_english(seeded):
    d = build_daily_digest(seeded, GUILD, "2026-10-14", "en")
    assert "Daily expense summary" in d and "Needs review" in d and "unclear data" in d
    assert "สรุปค่าใช้จ่าย" not in d
    assert "No receipts" in build_daily_digest(seeded, GUILD, "2026-01-01", "en")
