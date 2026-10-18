# ruff: noqa: F811, F401
"""History import (retention window, resume, gap, downtime catch-up, past-day summaries) and image
memory (describer, receipts, caps, backlog). Offline: fake channels with discord.py-like history."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import discord
import pytest
from conftest import GUILD, FakeClient, json_resp
from test_bot_multitenant import ME, member
from test_memory import RChan, bkk, mk, summary_json
from test_modes import STAFF, TMsg, bot, set_mode, world

from salid import bot as botmod
from salid.extract import IngestResult
from salid.guild_settings import FOREVER_DAYS, GuildSettingsUpdate
from salid.image_memory import ImageNote, format_note, image_mime
from salid.memory_import import import_progress
from salid.models import Receipt

NOW = datetime(2026, 10, 14, 5, tzinfo=UTC)  # = FixedClockSettings.now() (12:00 Bangkok)


class FakeAtt(SimpleNamespace):
    reads = 0

    async def read(self):
        FakeAtt.reads += 1
        return b"\x89PNG-bytes"


def att(name="a.png", ctype="image/png", size=1000):
    return FakeAtt(filename=name, content_type=ctype, size=size)


class HChan(RChan):
    """History honours after / before / oldest_first / limit like discord.py."""

    def __init__(self, id, name, readers=(STAFF, 999)):
        super().__init__(id, name, readers)
        self.stored, self.fetches = [], []

    async def history(self, limit=None, after=None, before=None, oldest_first=None):
        self.fetches.append({"after": after, "before": before, "limit": limit})
        rows = [
            m
            for m in self.stored
            if (after is None or m.created_at > after) and (before is None or m.created_at < before)
        ]
        rows.sort(key=lambda m: m.created_at, reverse=not oldest_first)
        for m in rows[:limit]:
            yield m

    async def fetch_message(self, mid):
        for m in self.stored:
            if m.id == mid:
                return m
        raise discord.NotFound(SimpleNamespace(status=404, reason="gone"), "gone")


def hmsg(world, chan, mid, at, text="hello", attachments=(), author=None):
    m = TMsg(world.g, chan, author or member(world.g, STAFF), text, attachments=list(attachments))
    m.id, m.created_at = mid, at
    chan.stored.append(m)
    return m


@pytest.fixture
def iw(world, bot):
    g = world.g
    g.text_channels[:] = [HChan(10, "ทั่วไป"), HChan(12, "ads"), HChan(13, "ว่าง")]
    for c in g.text_channels:
        c.guild = g
    g.me = SimpleNamespace(id=999)
    bot.importer.page_sleep = 0
    FakeAtt.reads = 0
    return SimpleNamespace(g=g, a=g.text_channels[0], b=g.text_channels[1], c=g.text_channels[2])


def set_llm(bot, client):
    bot.llm = bot.importer.llm = bot.memory.scheduler.llm = client


async def run_import(bot, summaries=True):
    await bot._import_all()
    await bot.drain_memory()


def logged(bot, cid=10, start="2026-01-01", end="2026-12-31"):
    return bot.memory_store.messages(GUILD, {cid}, start, end, limit=5000, newest_first=False)


def ago(days, hours=0):
    return NOW - timedelta(days=days) + timedelta(hours=hours)


# ------------------------------------------------------------------ import
async def test_import_covers_the_retention_window_including_pre_memory_history(bot, iw, world):
    bot.team.set_memory_retention_days(GUILD, 90)  # a chosen window (the default keeps forever)
    hmsg(world, iw.a, 1, ago(100), "too old")
    hmsg(world, iw.a, 2, ago(40), "forty days")
    hmsg(world, iw.a, 3, ago(5), "five days")
    await run_import(bot)
    assert [m.text for m in logged(bot)] == ["forty days", "five days"]  # nothing past 90 days
    st = bot.memory_store.get_channel_state(GUILD, 10)
    assert st.complete and st.messages == 2 and st.imported_from.startswith("2026-07-16")
    after = iw.a.fetches[0]["after"]
    assert after == NOW - timedelta(days=90)  # bounded by the guild's retention


async def test_import_on_the_forever_default_reads_ten_years_of_history(bot, iw, world):
    hmsg(world, iw.a, 1, ago(400), "last year")
    hmsg(world, iw.a, 2, ago(5), "five days")
    await run_import(bot)
    assert [m.text for m in logged(bot, start="2025-01-01")] == ["last year", "five days"]
    assert iw.a.fetches[0]["after"] == NOW - timedelta(days=FOREVER_DAYS)


async def test_import_is_idempotent_and_resumes_as_a_catch_up(bot, iw, world):
    for i in range(1, 4):
        hmsg(world, iw.a, i, ago(2, i), f"m{i}")
    await run_import(bot)
    await run_import(bot)  # a restart: nothing new, nothing duplicated
    assert len(logged(bot)) == 3
    assert iw.a.fetches[-1]["after"] == datetime.fromisoformat(
        bot.memory_store.get_channel_state(GUILD, 10).last_message_at
    )
    hmsg(world, iw.a, 9, ago(0, -1) - timedelta(hours=2), "while the bot was down")
    await run_import(bot)
    assert [m.text for m in logged(bot)][-1] == "while the bot was down" and len(logged(bot)) == 4


async def test_import_resumes_after_the_guild_cap(bot, iw, world):
    bot.s.memory_import_guild_cap = 250
    for i in range(1, 701):
        hmsg(world, iw.a, i, ago(30) + timedelta(minutes=i), f"m{i}")
    await bot._import_all()
    await bot.drain_memory()
    first = len(logged(bot))
    st = bot.memory_store.get_channel_state(GUILD, 10)
    assert 250 <= first <= 350 and not st.complete  # stopped at a page boundary
    assert bot._import_resume_at is not None  # and a resume is scheduled
    for _ in range(5):
        await run_import(bot)
    st = bot.memory_store.get_channel_state(GUILD, 10)
    assert st.complete and len(logged(bot)) == 700 and st.messages == 700
    assert len({m.message_id for m in logged(bot)}) == 700


async def test_lengthening_retention_imports_only_the_older_gap(bot, iw, world):
    bot.team.set_memory_retention_days(GUILD, 30)
    hmsg(world, iw.a, 1, ago(60), "in the gap")
    hmsg(world, iw.a, 2, ago(10), "recent")
    await run_import(bot)
    assert [m.text for m in logged(bot)] == ["recent"]
    iw.a.fetches.clear()
    bot.team.set_memory_retention_days(GUILD, 90)
    await run_import(bot)
    assert [m.text for m in logged(bot)] == ["in the gap", "recent"]
    gap = [f for f in iw.a.fetches if f["before"] is not None]
    assert len(gap) == 1 and gap[0]["after"] == NOW - timedelta(days=90)
    assert gap[0]["before"] == NOW - timedelta(days=30)  # [new start, old start) only
    st = bot.memory_store.get_channel_state(GUILD, 10)
    assert st.imported_from.startswith("2026-07-16") and st.gap_cursor is None and st.complete


async def test_import_targets_follow_modes_permissions_and_threads(bot, iw, world):
    set_mode(bot, 12, "off")
    iw.c.readers = {STAFF}  # the bot cannot read #ว่าง
    hmsg(world, iw.a, 1, ago(1), "ok")
    hmsg(world, iw.b, 2, ago(1), "off channel")
    hmsg(world, iw.c, 3, ago(1), "unreadable")
    pub = HChan(500, "thread")
    pub.parent_id, pub.parent, pub.guild = 10, iw.a, world.g
    prv = HChan(501, "secret")
    prv.parent_id, prv.parent, prv.guild, prv.is_private = 10, iw.a, world.g, lambda: True
    arch = HChan(502, "archived")
    arch.parent_id, arch.parent, arch.guild = 10, iw.a, world.g
    arch.archive_timestamp = NOW - timedelta(days=5)
    world.g.threads = [pub, prv]

    async def archived(limit=None):
        yield arch

    iw.a.archived_threads = archived
    hmsg(world, pub, 4, ago(1), "in thread")
    hmsg(world, prv, 5, ago(1), "private thread")
    hmsg(world, arch, 6, ago(1), "archived thread")
    await run_import(bot)
    got = bot.memory_store.messages(GUILD, {10, 12, 13}, "2026-01-01", "2026-12-31", limit=50)
    assert {m.text for m in got} == {"ok", "in thread", "archived thread"}
    threads = {m.text: (m.channel_id, m.thread_id) for m in got}
    assert threads["in thread"] == (10, 500) and threads["archived thread"] == (10, 502)
    assert iw.b.fetches == [] and iw.c.fetches == [] and prv.fetches == []


async def test_import_skips_other_bots_and_keeps_salids_own_messages(bot, iw, world):
    robot = member(world.g, 55)
    robot.bot = True
    hmsg(world, iw.a, 1, ago(1), "beep", author=robot)
    hmsg(world, iw.a, 2, ago(1, 1), "my reply", author=ME)
    hmsg(world, iw.a, 3, ago(1, 2), "human")
    await run_import(bot)
    assert [(m.text, m.kind) for m in logged(bot)] == [("my reply", "salid"), ("human", "chat")]


async def test_import_disabled_by_the_memory_flag(bot, iw, world):
    hmsg(world, iw.a, 1, ago(1), "x")
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features=dict(memory=False)), "x")
    bot.team._cache.clear()
    await run_import(bot)
    assert iw.a.fetches == []


async def test_triggers_start_an_import(bot, iw, world, monkeypatch):
    started = []
    monkeypatch.setattr(bot, "_start_import", lambda: started.append(1))
    bot._memory_triggers()  # first look: just remembers the config (keep forever)
    assert started == []
    bot.team.set_memory_retention_days(GUILD, 60)  # shortened from forever: no import
    bot._memory_triggers()
    assert started == []
    bot.team.set_memory_retention_days(GUILD, 120)
    bot._memory_triggers()
    assert started == [1]
    bot.team.set_memory_retention_days(GUILD, 30)  # shortening needs no import
    bot._memory_triggers()
    assert started == [1]
    bot.guilds_store.save_settings(
        GUILD, GuildSettingsUpdate(features=dict(memory=False), memory_retention_days=30), "x"
    )
    bot.team._cache.clear()
    bot._memory_triggers()
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(memory_retention_days=30), "x")
    bot.team._cache.clear()
    bot._memory_triggers()  # switched on
    assert started == [1, 1]
    bot._import_resume_at = bot._clock() - 1
    bot._memory_triggers()
    assert started == [1, 1, 1] and bot._import_resume_at is None


async def test_switching_a_channel_on_starts_an_import(bot, iw, world, monkeypatch):
    from test_modes import Interaction

    started = []
    monkeypatch.setattr(bot, "_start_import", lambda: started.append(1))
    set_mode(bot, 10, "off")
    it = Interaction(world.g, member(world.g, 3, manage_guild=True), iw.a)
    await bot.handle_mode(it, "call")
    assert started == [1]


# ------------------------------------------------------------------ past-day summaries
async def test_past_days_get_one_final_summary_each(bot, iw, world):
    for i in range(4):
        hmsg(world, iw.a, 10 + i, ago(10, i), f"busy {i}")  # one busy day: 4 messages
    hmsg(world, iw.a, 20, ago(8), "lonely")  # a day with 1 message: no summary
    hmsg(world, iw.a, 21, ago(8, 1), "two")  # 2 messages: still below 3
    llm = FakeClient([summary_json("busy")])
    set_llm(bot, llm)
    await run_import(bot)
    day = (NOW - timedelta(days=10)).astimezone(bkk(1, 1).tzinfo).date().isoformat()
    s = bot.memory_store.get_summary(GUILD, 10, day)
    assert s.final and not s.dirty and s.message_count == 4 and len(llm.models.calls) == 1
    assert (
        bot.memory_store.get_summary(GUILD, 10, (NOW - timedelta(days=8)).date().isoformat())
        is None
    )
    await run_import(bot)  # a second run never touches a final summary
    assert len(llm.models.calls) == 1
    assert bot.memory_store.get_channel_state(GUILD, 10).summary_days == []


async def test_summary_queue_is_capped_per_day_newest_first(bot, iw, world):
    bot.s.memory_import_summaries_per_day = 2
    for d in (20, 15, 12, 9):  # four busy days
        for i in range(3):
            hmsg(world, iw.a, d * 10 + i, ago(d, i), f"m{d}-{i}")
    llm = FakeClient([summary_json(str(i)) for i in range(4)])
    set_llm(bot, llm)
    await run_import(bot)
    days = sorted(
        s.local_date for s in bot.memory_store.summaries(GUILD, {10}, "2026-01-01", "2026-12-31")
    )
    assert days == [
        (NOW - timedelta(days=d)).astimezone(bkk(1, 1).tzinfo).date().isoformat() for d in (12, 9)
    ]
    assert bot.memory_store.get_guild_state(GUILD).summaries_done == 2
    # the oldest two wait in the queue until the budget resets
    assert len(bot.memory_store.get_channel_state(GUILD, 10).summary_days) == 2


async def test_a_failed_past_summary_stays_queued(bot, iw, world):
    for i in range(3):
        hmsg(world, iw.a, i + 1, ago(5, i), f"m{i}")

    def boom(contents, config):
        raise RuntimeError("down")

    set_llm(bot, FakeClient([boom, summary_json("ok")]))
    await run_import(bot)
    assert len(bot.memory_store.get_channel_state(GUILD, 10).summary_days) == 1
    await run_import(bot)
    assert bot.memory_store.get_channel_state(GUILD, 10).summary_days == []


async def test_import_progress_status(bot, iw, world):
    hmsg(world, iw.a, 1, ago(1), "x")
    await run_import(bot)
    p = import_progress(bot.memory_store, GUILD, 300, bot.s.now().date().isoformat())
    assert p == {
        "channels_done": 3,
        "channels_total": 3,
        "messages": 1,
        "images_done": 0,
        "images_cap": 300,
    }
    from test_modes import Interaction

    it = Interaction(world.g, member(world.g, STAFF), iw.a)
    await bot.handle_memory(it, None)
    assert "3/3" in it.sent[0][0] and "รูป 0/300" in it.sent[0][0]


# ------------------------------------------------------------------ images
def note_resp(desc="ใบเสนอราคา", text="ยอดรวม 12,500 บาท", kind="document"):
    return json_resp({"description": desc, "text": text, "type": kind})


def image_calls(llm):
    return [c for c in llm.models.calls if isinstance(c[1], list)]


async def settle(bot):
    while bot._bg_tasks:
        await asyncio.gather(*list(bot._bg_tasks))
    await bot.drain_memory()


def test_image_helpers():
    assert image_mime(att("x.bin", "image/webp; charset=x")) == "image/webp"
    assert (
        image_mime(att("x.JPG", "")) == "image/jpeg"
        and image_mime(att("a.pdf", "application/pdf")) is None
    )
    text = format_note(
        ImageNote(description="กราฟยอดขาย", text="Q3  50,000\nQ4 60,000", type="chart/table")
    )
    assert text == "[รูป: chart/table] กราฟยอดขาย | ข้อความในรูป: Q3 50,000 Q4 60,000"
    assert format_note(ImageNote(description="แมว", type="photo")) == "[รูป: photo] แมว"


async def test_each_image_is_described_once_and_ocr_is_searchable(bot, iw, world):
    llm = FakeClient([note_resp(), note_resp("แมวนอน", "", "photo")])
    set_llm(bot, llm)
    m = await send_images(bot, world, iw.a, "ดูนี่", [att("quote.png"), att("cat.jpg", "image/jpeg")])
    assert m.id
    await settle(bot)
    calls = image_calls(llm)
    assert len(calls) == 2 and calls[0][0] == bot.s.gate_model
    assert calls[0][2].temperature == 0 and calls[0][2].max_output_tokens <= 500
    (rec,) = logged(bot)
    assert "[รูป: document] ใบเสนอราคา | ข้อความในรูป: ยอดรวม 12,500 บาท" in rec.text
    assert (
        "[รูป: photo] แมวนอน" in rec.text and "[image:" not in rec.text and rec.text.startswith("ดูนี่")
    )
    from salid.qa import MemoryTools

    tools = MemoryTools(bot.memory_store, GUILD, {10: "ทั่วไป"}, bot.s)
    hit = tools.chat_search(["12,500", "ใบเสนอราคา"])["hits"][0]
    assert hit["messages"][0]["text"].count("12,500") == 1
    assert bot.image_stats == {"described": 2, "failed": 0, "skipped": 0}


async def send_images(bot, world, chan, text, atts, who=STAFF):
    m = TMsg(world.g, chan, member(world.g, who), text, attachments=atts)
    await bot.on_message(m)
    return m


async def test_at_most_four_images_per_message_and_big_or_non_images_skipped(bot, iw, world):
    llm = FakeClient([note_resp(f"img{i}", "") for i in range(6)])
    set_llm(bot, llm)
    atts = [att(f"{i}.png") for i in range(6)] + [att("doc.pdf", "application/pdf")]
    await send_images(bot, world, iw.a, "", atts)
    await settle(bot)
    assert len(image_calls(llm)) == 4 and FakeAtt.reads == 5  # + the PDF read for memory
    (rec,) = logged(bot)
    assert (
        rec.text.count("[รูป:") == 4
        and "[image: 4.png]" in rec.text
        and "[file: doc.pdf]" in rec.text
    )
    llm2 = FakeClient([note_resp()])
    set_llm(bot, llm2)
    FakeAtt.reads = 0
    await send_images(
        bot, world, iw.a, "ใหญ่", [att("big.png", size=9 * 1024 * 1024), att("ok.png")]
    )
    await settle(bot)
    assert len(image_calls(llm2)) == 1 and FakeAtt.reads == 1  # the 9 MB one is never read
    assert bot.image_stats["skipped"] == 1
    assert "[image: big.png]" in logged(bot)[-1].text


async def test_a_failed_description_keeps_the_plain_marker(bot, iw, world, caplog):
    def boom(contents, config):
        raise RuntimeError("secret contents")

    set_llm(bot, FakeClient([boom]))
    with caplog.at_level("WARNING"):
        await send_images(bot, world, iw.a, "รูป", [att("p.png")])
        await settle(bot)
    assert logged(bot)[0].text == "รูป [image: p.png]" and bot.image_stats["failed"] == 1
    assert "image description failed (RuntimeError)" in caplog.text and "secret" not in caplog.text


async def test_images_in_off_channels_and_with_the_flag_off_are_not_described(bot, iw, world):
    llm = FakeClient([note_resp(), note_resp()])
    set_llm(bot, llm)
    set_mode(bot, 12, "off")
    await send_images(bot, world, iw.b, "x", [att()])
    bot.guilds_store.save_settings(
        GUILD, GuildSettingsUpdate(features=dict(image_memory=False)), "x"
    )
    bot.team._cache.clear()
    await send_images(bot, world, iw.a, "y", [att()])
    await settle(bot)
    # image memory is off: no note is stored, but receipts are still recognised by content, so
    # the describer runs once (only in the channel that is not off)
    assert len(image_calls(llm)) == 1 and FakeAtt.reads == 1
    assert logged(bot)[0].text == "y [image: a.png]" and logged(bot, 12) == []


async def test_receipt_posts_are_not_described_again(bot, iw, world, monkeypatch):
    llm = FakeClient([note_resp()])
    set_llm(bot, llm)
    bot.guilds_store.save_settings(  # a configured receipt channel: the pipeline reads the image
        GUILD, GuildSettingsUpdate(receipt_channel_ids=[str(iw.a.id)]), "x"
    )
    bot.team._cache.clear()
    await send_images(bot, world, iw.a, "ใบเสร็จ", [att("r.jpg", "image/jpeg")])
    await settle(bot)
    assert image_calls(llm) == []
    # the pipeline's result becomes the stored text (real _intake, stubbed extraction)
    del bot._intake
    r = bot.store.add_receipt(Receipt(guild_id=GUILD, submitter_id=1, submitter_name="A", sha256="s",
                                      vendor="Makro", total=1250.5, effective_date="2026-10-14",
                                      category="วัตถุดิบ", created_at="2026-10-14"))  # fmt: skip
    monkeypatch.setattr(
        botmod, "ingest_attachment", lambda *a, **k: IngestResult("created", "ok", "✅", r)
    )
    m = TMsg(
        world.g, iw.a, member(world.g, STAFF), "ใบเสร็จ", attachments=[att("r.jpg", "image/jpeg")]
    )
    set_mode(bot, 10, "listen")  # quiet: no text reply
    await bot.on_message(m)
    await settle(bot)
    text = logged(bot)[-1].text
    assert (
        "[ใบเสร็จ] Makro | ยอด 1,250.50 THB | 2026-10-14 | วัตถุดิบ" in text and image_calls(llm) == []
    )


async def test_a_non_receipt_image_in_a_receipt_channel_is_described(bot, iw, world, monkeypatch):
    llm = FakeClient([note_resp("ภาพกลุ่ม", "", "photo")])
    set_llm(bot, llm)
    del bot._intake
    iw.a.name = "receipts"
    monkeypatch.setattr(botmod, "ingest_attachment", lambda *a, **k: IngestResult("ignored"))
    m = TMsg(world.g, iw.a, member(world.g, STAFF), "", attachments=[att("team.jpg", "image/jpeg")])
    set_mode(bot, 10, "listen")
    await bot.on_message(m)
    await settle(bot)
    assert len(image_calls(llm)) == 1 and "ภาพกลุ่ม" in logged(bot)[-1].text


async def test_replies_never_wait_for_image_descriptions(bot, iw, world):
    gate = asyncio.Event()

    class SlowAtt(FakeAtt):
        async def read(self):
            await gate.wait()
            return b"x"

    set_llm(bot, FakeClient([note_resp()]))
    m = await send_images(
        bot,
        world,
        iw.a,
        "สลิด ช่วยหน่อย",
        [SlowAtt(filename="a.png", content_type="image/png", size=1)],
    )
    assert bot.calls == [("answer", "สลิด ช่วยหน่อย")] and m.replies == ["ans"]  # answered already
    gate.set()
    await settle(bot)
    assert "[รูป:" in logged(bot)[0].text


# ------------------------------------------------------------------ historical images
async def test_backlog_describes_newest_first_within_the_daily_cap(bot, iw, world, monkeypatch):
    bot.s.memory_import_images_per_day = 2
    for i in range(1, 5):  # i=4 is the newest
        hmsg(world, iw.a, i, ago(10 - i), f"pic{i}", attachments=[att(f"{i}.png")])
    llm = FakeClient([note_resp(f"desc{i}", f"ocr{i}") for i in range(4)])
    set_llm(bot, llm)
    await run_import(bot)
    texts = {m.message_id: m.text for m in logged(bot)}
    assert "desc0" in texts[4] and "desc1" in texts[3]  # newest two done first
    assert "[image: 2.png]" in texts[2] and "[image: 1.png]" in texts[1]  # the rest wait
    assert bot.memory_store.get_guild_state(GUILD).images_done == 2
    assert len(image_calls(llm)) == 2
    await run_import(bot)  # same day: the cap holds
    assert len(image_calls(llm)) == 2
    p = import_progress(bot.memory_store, GUILD, 2, bot.s.now().date().isoformat())
    assert (p["images_done"], p["images_cap"]) == (2, 2)
    tomorrow = datetime(2026, 10, 15, 12, tzinfo=bkk(1, 1).tzinfo)
    monkeypatch.setattr(type(bot.s), "now", lambda self: tomorrow)  # budget resets next day
    bot.importer.now = bot.s.now
    await run_import(bot)
    assert len(image_calls(llm)) == 4
    assert all("[image: " not in m.text for m in logged(bot))


async def test_backlog_marks_vanished_messages_and_unreadable_images(bot, iw, world):
    # logged earlier (e.g. live) but the Discord message is gone now
    bot.memory_store.add_message(mk(bot.s, 1, ago(3), "gone [image: g.png]", cid=10))
    hmsg(world, iw.a, 2, ago(2), "big", attachments=[att("b.png", size=20 * 1024 * 1024)])
    set_llm(bot, FakeClient([]))
    await run_import(bot)
    texts = {m.message_id: m.text for m in logged(bot)}
    assert "[image-unread: g.png]" in texts[1] and "[image-unread: b.png]" in texts[2]
    assert FakeAtt.reads == 0 and bot.image_stats["described"] == 0
    await run_import(bot)  # nothing left to retry
    assert {m.message_id: m.text for m in logged(bot)} == texts


# ------------------------------------------------------------------ access changes start an import
@pytest.fixture
def access(bot, world, monkeypatch):
    started, synced = [], []
    monkeypatch.setattr(botmod, "IMPORT_DEBOUNCE_S", 0.01)
    monkeypatch.setattr(bot, "_start_import", lambda: started.append(1))
    monkeypatch.setattr(bot, "_request_sync", lambda g: synced.append(g.id))
    return started, synced


async def debounce_wait():
    await asyncio.sleep(0.05)


async def test_channel_create_requests_import(bot, world, access):
    started, synced = access
    await bot.on_guild_channel_create(SimpleNamespace(guild=world.g))
    await debounce_wait()
    assert started == [1] and synced == [GUILD]


async def test_channel_overwrite_change_requests_import_but_other_updates_do_not(
    bot, world, access
):
    started, synced = access
    same = SimpleNamespace(guild=world.g, name="a", overwrites={})
    await bot.on_guild_channel_update(same, SimpleNamespace(**vars(same)))
    await debounce_wait()
    assert started == [] and synced == []
    changed = SimpleNamespace(guild=world.g, name="a", overwrites={"x": 1})
    await bot.on_guild_channel_update(same, changed)
    await debounce_wait()
    assert started == [1] and synced == [GUILD]


async def test_own_role_change_requests_import(bot, world, access):
    started, synced = access
    before = SimpleNamespace(id=ME.id, guild=world.g, roles=[1])
    await bot.on_member_update(before, SimpleNamespace(id=ME.id, guild=world.g, roles=[1, 2]))
    await debounce_wait()
    assert started == [1] and synced == [GUILD]
    other = SimpleNamespace(id=5, guild=world.g, roles=[1, 2])
    await bot.on_member_update(before, other)  # someone else
    await bot.on_member_update(before, SimpleNamespace(id=ME.id, guild=world.g, roles=[1]))
    await debounce_wait()
    assert started == [1]


async def test_role_permission_change_requests_import_only_for_salids_role(bot, world, access):
    started, synced = access
    role = SimpleNamespace(id=7, guild=world.g, permissions=2)
    world.g.me = SimpleNamespace(id=ME.id, roles=[role])
    await bot.on_guild_role_update(SimpleNamespace(permissions=1), role)
    await debounce_wait()
    assert started == [1] and synced == [GUILD]
    foreign = SimpleNamespace(id=8, guild=world.g, permissions=2)
    await bot.on_guild_role_update(SimpleNamespace(permissions=1), foreign)  # not Salid's role
    await bot.on_guild_role_update(SimpleNamespace(permissions=2), role)  # unchanged
    await debounce_wait()
    assert started == [1]


async def test_quick_events_start_a_single_import(bot, world, access):
    started, _ = access
    for _ in range(4):
        await bot.on_guild_channel_create(SimpleNamespace(guild=world.g))
    await debounce_wait()
    assert started == [1]
