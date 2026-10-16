# ruff: noqa: F811, F401
"""Join-mode silence / memory hints / same-user cooldown, receipts found anywhere (describer ->
intake, history import), name-prefixed typed commands. Offline."""

import itertools
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from conftest import GUILD, FakeClient, FakeEmbedder, json_resp, scripted, text_resp
from test_bot_multitenant import ME, member
from test_memory import mk
from test_memory_import import (
    FakeAtt,
    ago,
    att,
    hmsg,
    image_calls,
    iw,
    logged,
    note_resp,
    run_import,
    set_llm,
    settle,
)
from test_modes import EXEC, STAFF, Sent, TMsg, bot, manager, say, send, set_mode, world

from salid import bot as botmod
from salid import gate, qa
from salid.config import Settings
from salid.extract import IngestResult
from salid.guild_settings import GuildSettingsUpdate
from salid.i18n import t
from salid.memory_index import IndexedMemoryStore
from salid.models import Receipt
from salid.naming import command_text


# ------------------------------------------------------------------ A: NO_REPLY
@pytest.mark.parametrize(
    "text, expected",
    [("NO_REPLY", True), ("no_reply", True), ("  No_Reply \n", True), ("`NO_REPLY`", True),
     ("NO REPLY.", True), ("", False), ("ยอดรวม 500 บาท", False),
     ("NO_REPLY " + "x" * 30, False)],
)  # fmt: skip
def test_is_no_reply(text, expected):
    assert qa.is_no_reply(text) is expected


def test_interjection_prompt_only_when_unprompted():
    s = Settings()
    assert "NO_REPLY" in qa.system_prompt(s, False, interjection=True)
    assert "NO_REPLY" not in qa.system_prompt(s, False)


def test_ask_passes_the_flag_to_the_prompt(seeded, settings):
    llm = scripted(text_resp("NO_REPLY"))
    ans = qa.ask(settings, seeded, "x", guild_id=GUILD, is_exec=True, user_id=1, model=llm,
                 interjection=True)  # fmt: skip
    assert ans.text == "NO_REPLY"
    assert "NO_REPLY" in str(llm.requests[0].config.system_instruction)


async def test_silent_interjection_posts_nothing_and_keeps_the_budget(bot, world):
    set_mode(bot, 10, "join")
    bot.gate.silent = True
    users = itertools.count(100)
    for _ in range(5):
        m = await send(bot, world, world.gen, "ยอดค่าใช้จ่ายเท่าไหร่", who=next(users))
        bot.clock.advance(60)
        await bot.tick()
        assert m.replies == []
    assert len(bot.calls) == 5 and all(k["interjection"] for k in bot.answer_kw)
    assert not bot._interjections[10] and bot._interjected_at == {}  # nothing consumed
    bot.gate.silent = False
    m = await send(bot, world, world.gen, "ยอดค่าใช้จ่ายเท่าไหร่", who=next(users))
    bot.clock.advance(60)
    await bot.tick()
    assert m.replies == ["ans"] and len(bot._interjections[10]) == 1


def _real_answer(bot, monkeypatch, text):
    seen = []
    monkeypatch.setattr(
        botmod, "ask",
        lambda s, store, q, **kw: seen.append((q, kw)) or SimpleNamespace(text=text, seconds=0, rounds=0),
    )  # fmt: skip
    bot.__dict__.pop("_answer", None)
    return seen


async def test_explicit_call_is_never_suppressed(bot, world, monkeypatch):
    seen = _real_answer(bot, monkeypatch, "NO_REPLY")
    m = say(world, world.gen, "สลิด ยอดเท่าไร")
    assert await bot._answer(m, world.g, m.author) is True
    assert m.replies == ["NO_REPLY"] and seen[0][1]["interjection"] is False


async def test_interjection_no_reply_is_suppressed(bot, world, monkeypatch):
    seen = _real_answer(bot, monkeypatch, "no_reply")
    m = say(world, world.gen, "ยอดเท่าไร")
    assert await bot._answer(m, world.g, m.author, interjection=True) is False
    assert m.replies == [] and seen[0][1]["interjection"] is True
    _real_answer(bot, monkeypatch, "ยอด 500")
    assert await bot._answer(m, world.g, m.author, interjection=True) is True and m.replies == [
        "ยอด 500"
    ]


async def test_answer_logs_one_metrics_line_without_content(bot, world, monkeypatch, caplog):
    _real_answer(bot, monkeypatch, "ลับมาก")
    m = say(world, world.gen, "สลิด ยอดเท่าไร")
    with caplog.at_level("INFO", logger="salid.bot"):
        await bot._answer(m, world.g, m.author)
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("answer "))
    for key in ("guild=", "channel=", "ms_total=", "ms_context=", "ms_agent=", "tok_in=0",
                "tok_cached=0"):  # fmt: skip
        assert key in line
    assert "ลับมาก" not in line and "ยอด" not in line


# ------------------------------------------------------------------ A: stage-2 memory hints
def test_keywords_drop_question_words_and_split_thai_on_particles():
    kws = gate.extract_keywords("ค่าน้ำมันเดือนนี้เท่าไหร่")
    assert kws[0] == "ค่าน้ำมันเดือนนี้" and all(len(k) <= 6 for k in kws[1:]) and len(kws) <= 8
    assert gate.extract_keywords("who paid for the Makro order?") == [
        "order",
        "Makro",
        "paid",
    ] or set(gate.extract_keywords("who paid for the Makro order?")) == {"paid", "Makro", "order"}
    assert gate.extract_keywords("ใคร ไหม อะไร") == []
    assert gate.extract_keywords("ok no") == []


async def test_gate_gets_free_memory_hints_scoped_to_the_candidate(bot, world):
    set_mode(bot, 10, "join")
    set_mode(bot, 12, "join")
    when = datetime(2026, 10, 13, 5, tzinfo=UTC)  # raw messages: inside the 3-day raw window
    bot.memory_store.add_message(mk(bot.s, 1, when, "ซื้อน้ำแข็งที่ Makro 1,200 บาท", cid=10))
    bot.memory_store.add_message(mk(bot.s, 2, when, "น้ำแข็งลับของห้องอื่น", cid=12))
    await send(bot, world, world.gen, "น้ำแข็งซื้อที่ไหนเมื่อวาน")
    bot.clock.advance(60)
    await bot.tick()
    ids, hints = bot.hints_seen[0]
    (line,) = hints.values()
    assert line.startswith("memory has: [2026-10-13 #ทั่วไป]") and "Makro" in line
    assert "ห้องอื่น" not in line and len(line) <= gate.HINT_TOTAL_CHARS


async def test_answers_get_the_embedder_but_join_hints_stay_keyword_only(bot, world):
    assert isinstance(bot.memory_store, IndexedMemoryStore)
    emb = bot.embedder = FakeEmbedder()
    set_mode(bot, 10, "join")
    when = datetime(2026, 10, 13, 5, tzinfo=UTC)
    bot.memory_store.add_message(mk(bot.s, 1, when, "ซื้อน้ำแข็งที่ Makro 1,200 บาท", cid=10))
    m = await send(bot, world, world.gen, "น้ำแข็งซื้อที่ไหนเมื่อวาน")
    bot.clock.advance(60)
    await bot.tick()
    assert bot.hints_seen and emb.calls == []  # hints: no embedding call
    tools = await bot._memory_tools_for(world.g, m.author, world.gen.id)
    assert tools.embedder is emb


async def test_no_hints_when_memory_is_off(bot, world):
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features=dict(memory=False)), "x")
    bot.team._cache.clear()
    set_mode(bot, 10, "join")
    bot.memory_store.add_message(
        mk(bot.s, 1, datetime(2026, 10, 10, 5, tzinfo=UTC), "น้ำแข็ง", cid=10)
    )
    await send(bot, world, world.gen, "น้ำแข็งซื้อที่ไหน")
    bot.clock.advance(60)
    await bot.tick()
    assert bot.hints_seen[0][1] is None


def test_hints_fit_the_total_budget_and_the_prompt_mentions_them():
    hints = gate.fit_hints({"1": "a" * 400, "2": "b" * 400})
    assert sum(map(len, hints.values())) == gate.HINT_TOTAL_CHARS
    client = FakeClient([json_resp({"answer": False})])
    gate.pick_interjection(client, "m", [], ["1"], channel_id=1, hints={"1": "memory has: [x] y"})
    prompt = client.models.calls[0][1]
    assert "[1] memory has: [x] y" in prompt and "irrelevant" in prompt


# ------------------------------------------------------------------ A: same-user cooldown
async def test_same_user_is_not_interjected_twice_within_ten_minutes(bot, world):
    set_mode(bot, 10, "join")

    async def attempt(who):
        n = bot.gate.pick_calls
        await send(bot, world, world.gen, "ยอดค่าใช้จ่ายเท่าไหร่", who=who)
        bot.clock.advance(60)
        await bot.tick()
        return bot.gate.pick_calls - n

    assert await attempt(STAFF) == 1 and len(bot.calls) == 1
    assert await attempt(STAFF) == 0 and len(bot.calls) == 1  # dropped before the gate: no tokens
    assert await attempt(EXEC) == 1 and len(bot.calls) == 2  # someone else is fine
    bot.clock.advance(botmod.INTERJECT_SAME_USER_COOLDOWN)
    assert await attempt(STAFF) == 1 and len(bot.calls) == 3


async def test_cooldown_does_not_affect_explicit_calls(bot, world):
    set_mode(bot, 10, "join")
    await send(bot, world, world.gen, "ยอดค่าใช้จ่ายเท่าไหร่")
    bot.clock.advance(60)
    await bot.tick()
    await send(bot, world, world.gen, "สลิด ยอดเท่าไร")
    assert [c[1] for c in bot.calls] == ["ยอดค่าใช้จ่ายเท่าไหร่", "สลิด ยอดเท่าไร"]


# ------------------------------------------------------------------ B1: receipts anywhere
def slip_receipt(bot):
    return bot.store.add_receipt(
        Receipt(guild_id=GUILD, submitter_id=STAFF, submitter_name="A", sha256="slip",
                vendor="สมชาย", total=500.0, effective_date="2026-10-14", category="อื่นๆ",
                created_at="2026-10-14")
    )  # fmt: skip


def real_pipeline(bot, monkeypatch, llm, status="created"):
    bot.__dict__.pop("_intake", None)
    bot.__dict__.pop("_answer", None)
    bot.llm = llm
    bot.store.set_reply_message = lambda *a: None
    monkeypatch.setattr(Sent, "id", 77, raising=False)
    monkeypatch.setattr(  # never reach a real model from a test
        botmod, "ask", lambda *a, **k: SimpleNamespace(text="(stub answer)", seconds=0, rounds=0)
    )
    calls = []

    def ingest(*a, **k):
        calls.append(k)
        if status == "ignored":
            return IngestResult("ignored")
        return IngestResult(status, "บันทึกแล้ว", "✅", slip_receipt(bot))

    monkeypatch.setattr(botmod, "ingest_attachment", ingest)
    return calls


def photo_msg(world, chan, text="", who=STAFF, name="IMG_1025.jpg"):
    return TMsg(world.g, chan, member(world.g, who), text,
                attachments=[att(name, "image/jpeg")])  # fmt: skip


async def test_slip_image_in_a_chat_channel_goes_to_the_agent_not_to_intake(
    bot, world, monkeypatch
):
    llm = FakeClient([note_resp("สลิปโอนเงิน", "500.00", "receipt/invoice")])
    calls = real_pipeline(bot, monkeypatch, llm)
    FakeAtt.reads = 0
    m = photo_msg(world, world.chat)
    await bot.on_message(m)
    await settle(bot)
    assert m.replies == ["(stub answer)"] and calls == []  # the agent decides via record_receipt
    assert len(image_calls(llm)) == 1  # the describer still feeds memory, without receipt intake


async def test_receipt_like_image_in_listen_mode_is_quiet(bot, world, monkeypatch):
    llm = FakeClient([note_resp("ใบเสร็จ", "Makro", "receipt/invoice")])
    calls = real_pipeline(bot, monkeypatch, llm)
    set_mode(bot, 10, "listen")
    m = photo_msg(world, world.gen, "ของวันนี้")
    await bot.on_message(m)
    await settle(bot)
    assert len(calls) == 1 and m.replies == [] and m.reactions == ["✅"]


async def test_call_mode_image_without_a_tag_is_read_when_it_is_a_receipt(bot, world, monkeypatch):
    llm = FakeClient(
        [note_resp("ใบเสร็จ", "Makro", "receipt/invoice"), note_resp("แมว", "", "photo")]
    )
    calls = real_pipeline(bot, monkeypatch, llm)
    m = photo_msg(world, world.gen)
    await bot.on_message(m)
    await settle(bot)
    assert len(calls) == 1 and m.replies == ["บันทึกแล้ว"]
    other = photo_msg(world, world.gen, who=EXEC, name="cat.jpg")
    await bot.on_message(other)
    await settle(bot)
    assert len(calls) == 1 and other.replies == []  # a photo of a cat is not a receipt


async def test_image_in_an_off_channel_is_ignored(bot, world, monkeypatch):
    llm = FakeClient([])
    calls = real_pipeline(bot, monkeypatch, llm)
    set_mode(bot, 10, "off")
    await bot.on_message(photo_msg(world, world.gen))
    await settle(bot)
    assert calls == [] and llm.models.calls == []


async def test_receipts_feature_off_means_no_intake(bot, world, monkeypatch):
    llm = FakeClient([note_resp("ใบเสร็จ", "", "receipt/invoice")])
    calls = real_pipeline(bot, monkeypatch, llm)
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features=dict(receipts=False)), "x")
    bot.team._cache.clear()
    m = photo_msg(world, world.gen)
    await bot.on_message(m)
    await settle(bot)
    assert calls == [] and m.replies == []


async def test_image_only_message_that_is_not_a_receipt_is_discussed(bot, world, monkeypatch):
    llm = FakeClient([note_resp("แมวสีส้ม", "", "photo")])
    real_pipeline(bot, monkeypatch, llm, status="ignored")
    seen = []
    monkeypatch.setattr(
        botmod, "ask",
        lambda s, store, q, **kw: seen.append((q, kw)) or SimpleNamespace(text="แมวน่ารัก", seconds=0, rounds=0),
    )  # fmt: skip
    m = photo_msg(world, world.chat, name="cat.jpg")
    await bot.on_message(m)
    assert m.replies == ["แมวน่ารัก"]
    assert [f.filename for f in seen[0][1]["files"]] == ["cat.jpg"]
    assert t("greeting", "th") not in m.replies


def _qa_and_describer_off(bot):
    bot.guilds_store.save_settings(
        GUILD, GuildSettingsUpdate(features=dict(qa=False, image_memory=False)), "x"
    )
    bot.team._cache.clear()


async def test_chat_image_with_qa_and_the_describer_off_falls_back_to_intake(
    bot, world, monkeypatch
):
    llm = FakeClient([])
    calls = real_pipeline(bot, monkeypatch, llm)
    _qa_and_describer_off(bot)
    m = photo_msg(world, world.chat)
    await bot.on_message(m)
    assert len(calls) == 1 and m.replies == ["บันทึกแล้ว"] and llm.models.calls == []


async def test_chat_image_not_a_receipt_with_qa_off_stays_silent(bot, world, monkeypatch):
    real_pipeline(bot, monkeypatch, FakeClient([]), status="ignored")
    _qa_and_describer_off(bot)
    m = photo_msg(world, world.chat)
    await bot.on_message(m)
    assert m.replies == []  # not the generic greeting


# ------------------------------------------------------------------ B2: history import
def history_ingest(monkeypatch, bot, status="created"):
    calls = []

    def ingest(*a, **k):
        calls.append(k)
        r = bot.store.add_receipt(
            Receipt(guild_id=GUILD, submitter_id=k["submitter_id"], submitter_name="A",
                    sha256=f"h{len(calls)}", vendor="Makro", total=100.0,
                    effective_date="2026-10-01", category="วัตถุดิบ", created_at="2026-10-01")
        )  # fmt: skip
        return IngestResult(status, "reply", "✅", r)

    monkeypatch.setattr(botmod, "ingest_attachment", ingest)
    return calls


async def test_old_receipt_images_are_ingested_quietly_during_import(bot, iw, world, monkeypatch):
    author = member(world.g, 7)
    m = hmsg(world, iw.a, 5, ago(5), "", [att("r.jpg", "image/jpeg")], author=author)
    cat = hmsg(world, iw.a, 6, ago(4), "", [att("c.jpg", "image/jpeg")])
    llm = FakeClient(
        [note_resp("แมว", "", "photo"), note_resp("ใบเสร็จ", "Makro 100", "receipt/invoice")]
    )
    set_llm(bot, llm)
    calls = history_ingest(monkeypatch, bot)
    await run_import(bot)
    (k,) = calls  # only the receipt-like image
    assert k["submitter_id"] == 7 and k["message_id"] == 5 and k["channel_id"] == 10
    assert k["at"] == m.created_at and k["guild_id"] == GUILD
    assert m.replies == [] and m.reactions == [] and cat.replies == []  # quiet
    texts = {r.message_id: r.text for r in logged(bot)}
    assert "[ใบเสร็จ] Makro" in texts[5] and "แมว" in texts[6]
    assert bot.memory_store.get_guild_state(GUILD).images_done == 3  # 2 described + 1 extraction


async def test_history_receipts_respect_the_daily_cap(bot, iw, world, monkeypatch):
    bot.s.memory_import_images_per_day = 2
    for i in (1, 2):
        hmsg(world, iw.a, i, ago(10 - i), "", [att(f"{i}.jpg", "image/jpeg")])
    set_llm(bot, FakeClient([note_resp("ใบเสร็จ", "", "receipt/invoice")] * 4))
    calls = history_ingest(monkeypatch, bot)
    await run_import(bot)
    assert len(calls) == 1  # newest only: describe + extract used the cap


async def test_history_receipts_need_the_receipts_feature(bot, iw, world, monkeypatch):
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features=dict(receipts=False)), "x")
    bot.team._cache.clear()
    hmsg(world, iw.a, 1, ago(3), "", [att("1.jpg", "image/jpeg")])
    set_llm(bot, FakeClient([note_resp("ใบเสร็จ", "", "receipt/invoice")]))
    calls = history_ingest(monkeypatch, bot)
    await run_import(bot)
    assert calls == []


def test_ingest_uses_the_message_time_for_old_receipts(seeded, settings, monkeypatch):
    from salid import extract
    from salid.models import ReceiptExtraction

    monkeypatch.setattr(
        extract, "extract_receipt",
        lambda *a, **k: ReceiptExtraction(is_receipt=True, vendor="V", total=5.0, confidence=0.9),
    )  # fmt: skip
    at = datetime(2026, 9, 1, 3, 0, tzinfo=UTC)
    res = extract.ingest_attachment(
        seeded, None, settings, guild_id=GUILD, data=b"old-bytes", filename="a.jpg",
        content_type="image/jpeg", submitter_id=9, submitter_name="Z", channel_id=1,
        message_id=2, at=at,
    )  # fmt: skip
    assert res.receipt.effective_date == "2026-09-01" and res.receipt.created_at.startswith(
        "2026-09-01T10:00"
    )


# ------------------------------------------------------------------ B1: slips in the extraction
def test_extraction_prompt_and_confidence_for_transfer_slips(settings):
    from salid import extract

    assert "สลิปโอนเงิน" in extract.PROMPT and "receiver" in extract.PROMPT
    client = FakeClient([json_resp({
        "is_receipt": True, "vendor": "นายสมชาย", "receipt_date": "2026-10-13", "total": 500,
        "payment_method": "โอนเงิน", "confidence": 0.95, "category": "อื่นๆ"})])  # fmt: skip
    ex = extract.extract_receipt(client, settings, b"x", "image/jpeg", "ค่าข้าว")
    assert ex.is_receipt and ex.vendor == "นายสมชาย" and ex.total == 500
    assert ex.confidence == extract.SLIP_MAX_CONF < 0.7  # shown as "please check"
    assert "ค่าข้าว" in client.models.calls[0][1][1]


def test_image_note_receipt_kinds():
    from salid.image_memory import ImageNote

    for kind in ("receipt/invoice", "Receipt", "bank transfer slip", "ใบเสร็จ"):
        assert ImageNote(type=kind).is_receipt_like
    for kind in ("photo", "meme", "screenshot", "other"):
        assert not ImageNote(type=kind).is_receipt_like


# ------------------------------------------------------------------ B3: name-prefixed commands
@pytest.mark.parametrize(
    "text, rest, named",
    [("Salid เข้าประชุมหน่อย", "เข้าประชุม", True), ("สลิด ออกประชุม", "ออกประชุม", True),
     ("salid, be quiet please", "be quiet", True), ("สลิด พูดภาษาอังกฤษครับ", "พูดภาษาอังกฤษ", True),
     ("สลิดเข้าประชุมด้วยนะคะ", "เข้าประชุม", True), ("เข้าประชุมหน่อย", "เข้าประชุม", False),
     ("ช่วยสรุปหน่อยสลิด", "ช่วยสรุป", False)],
)  # fmt: skip
def test_command_text(text, rest, named):
    got = command_text(text)
    assert got[1] is named and got[0].startswith(rest.split("หน่อย")[0][:6])


async def test_name_prefixed_meeting_commands_work_without_a_tag(bot, world, monkeypatch):
    seen = []

    async def join(who, fallback_channel=None, lang="th"):
        seen.append("join")
        return "เข้าแล้ว"

    async def leave(who, lang="th", **_):
        seen.append("leave")
        return "ออกแล้ว"

    monkeypatch.setattr(bot.meetings, "join", join)
    monkeypatch.setattr(bot.meetings, "leave", leave)
    for text in ("Salid เข้าประชุมหน่อย", "สลิด ออกประชุม", "สลิดเข้าประชุมด้วยนะครับ"):
        author = member(world.g, STAFF)
        author.voice = object()
        m = TMsg(world.g, world.gen, author, text)
        await bot.on_message(m)
        assert len(m.replies) == 1 and bot.calls == []  # a command, not a chat answer
    assert seen == ["join", "leave", "join"]


async def test_name_prefixed_snooze_and_language(bot, world):
    m = await send(bot, world, world.gen, "salid be quiet")
    assert m.replies == [t("snooze", "th")] and bot._is_snoozed(10)
    m = await send(bot, world, world.gen, "สลิด คุยได้แล้วนะ")
    assert m.replies == [t("unsnooze", "th")]
    m = await send(bot, world, world.gen, "สลิด พูดภาษาอังกฤษหน่อย", who=manager(world))
    assert bot.team.channel_language(GUILD, 10) == "en" and len(m.replies) == 1
    assert bot.calls == []


async def test_a_name_inside_a_question_is_not_a_command(bot, world):
    await send(bot, world, world.gen, "สลิด เข้าประชุมที่ไหนบ้างเมื่อวาน")
    assert bot.calls == [("answer", "สลิด เข้าประชุมที่ไหนบ้างเมื่อวาน")]
