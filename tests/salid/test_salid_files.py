# ruff: noqa: F811, F401
"""Files sent to Salid are read by the agent (multimodal parts) and it acts through tools
(record_receipt, start_deep_task); routing in the bot. Offline."""

import asyncio
import io
import json
from types import SimpleNamespace

import pytest
from conftest import GUILD, FakeClient, fc_resp, scripted, text_resp
from test_bot_multitenant import ME, member
from test_memory_import import FakeAtt, att, image_calls, note_resp, settle
from test_modes import EXEC, STAFF, Sent, TMsg, bot, set_mode, world

from salid import bot as botmod
from salid import qa
from salid.extract import IngestResult
from salid.guild_settings import GuildSettingsUpdate
from salid.models import Receipt
from salid.qa import FileActions, FileInput, ask, select_files, system_prompt

MB = 1024 * 1024


def fi(index, name, mime, data=b"x", **kw):
    return FileInput(index, name, mime, data, **kw)


# ------------------------------------------------------------------ limits and parts
def test_select_files_limits():
    files = [fi(i, f"{i}.png", "image/png") for i in range(6)]
    ok, skipped = select_files(files)
    assert [f.index for f in ok] == [0, 1, 2, 3]
    assert [f.skip for f in skipped] == ["more than 4 files"] * 2


def test_select_files_size_limits_and_unsupported_types():
    files = [
        fi(0, "big.pdf", "application/pdf", size=15 * MB + 1),
        fi(1, "a.pdf", "application/pdf", size=12 * MB),
        fi(2, "b.pdf", "application/pdf", size=9 * MB),  # total would be 21 MB
        fi(3, "c.zip", None),
        fi(4, "n.txt", "text/plain", size=201 * 1024),
        fi(5, "ok.txt", "text/plain", size=10),
        fi(6, "pre.png", "image/png", skip="too large"),
    ]
    ok, skipped = select_files(files)
    assert [f.index for f in ok] == [1, 5]
    why = {f.filename: f.skip for f in skipped}
    assert why["big.pdf"] == "over 15 MB" and why["b.pdf"] == "total size over 20 MB"
    assert why["c.zip"] == "unsupported file type" and why["n.txt"].startswith("text file over")
    assert why["pre.png"] == "too large"


def test_file_mime():
    assert qa.file_mime("a.JPG") == "image/jpeg" and qa.file_mime("x", "image/webp") == "image/webp"
    assert qa.file_mime("a.pdf") == "application/pdf" and qa.file_mime("n.csv") == "text/plain"
    assert qa.file_mime("v.mp4", "video/mp4") is None and qa.file_mime("a.zip") is None


def _user_parts(llm):
    return llm.requests[0].contents[0].parts


def test_files_are_passed_as_parts_with_skipped_notes(seeded, settings):
    llm = scripted(text_resp("เห็นแล้ว"))
    files = [
        fi(0, "slip.jpg", "image/jpeg", b"JPEGDATA"),
        fi(1, "notes.txt", "text/plain", "ประชุมพรุ่งนี้".encode()),
        fi(2, "doc.pdf", "application/pdf", b"%PDF"),
        fi(3, "movie.mp4", None),
    ]
    a = ask(settings, seeded, "ดูให้หน่อย", guild_id=GUILD, is_exec=True, user_id=1, model=llm,
            files=files)  # fmt: skip
    assert a.text == "เห็นแล้ว"
    parts = _user_parts(llm)
    assert "ดูให้หน่อย" in parts[0].text and "[file skipped: movie.mp4, unsupported file type]" in (
        parts[0].text
    )
    assert parts[1].text == "[ไฟล์ #0: slip.jpg]"
    assert (
        parts[2].inline_data.mime_type == "image/jpeg" and parts[2].inline_data.data == b"JPEGDATA"
    )
    assert parts[3].text.startswith("[ไฟล์ #1: notes.txt]\n") and "ประชุมพรุ่งนี้" in parts[3].text
    assert parts[4].text == "[ไฟล์ #2: doc.pdf]"
    assert parts[5].inline_data.mime_type == "application/pdf"
    assert len(parts) == 6  # the text file is text, not bytes


def test_gif_goes_in_as_its_first_frame_png(seeded, settings):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, format="GIF")
    llm = scripted(text_resp("ok"))
    ask(settings, seeded, "x", guild_id=GUILD, is_exec=True, user_id=1, model=llm,
        files=[fi(0, "a.gif", "image/gif", buf.getvalue())])  # fmt: skip
    assert _user_parts(llm)[2].inline_data.mime_type == "image/png"


def test_an_unreadable_gif_is_reported_not_fatal(seeded, settings):
    llm = scripted(text_resp("ok"))
    ask(settings, seeded, "x", guild_id=GUILD, is_exec=True, user_id=1, model=llm,
        files=[fi(0, "a.gif", "image/gif", b"not a gif")])  # fmt: skip
    assert "[file skipped: a.gif, could not be read]" in _user_parts(llm)[0].text


def test_a_message_without_text_gets_a_placeholder_turn(seeded, settings):
    llm = scripted(text_resp("ok"))
    ask(settings, seeded, "", guild_id=GUILD, is_exec=True, user_id=1, model=llm,
        files=[fi(0, "a.png", "image/png")])  # fmt: skip
    assert _user_parts(llm)[0].text == qa.NO_TEXT_TURN
    llm = scripted(text_resp("ok"))
    ask(settings, seeded, "", guild_id=GUILD, is_exec=True, user_id=1, model=llm, history="[12:00] a: hi",
        files=[fi(0, "a.png", "image/png")])  # fmt: skip
    assert qa.NO_TEXT_TURN in _user_parts(llm)[0].text


def test_no_files_means_no_file_parts_prompt_or_tools(seeded, settings):
    llm = scripted(text_resp("ok"))
    ask(settings, seeded, "hi", guild_id=GUILD, is_exec=True, user_id=1, model=llm)
    assert len(_user_parts(llm)) == 1 and "record_receipt" not in str(llm.requests[0].config)
    assert "record_receipt" not in str(llm.requests[0].config.system_instruction)


def test_file_guidance_is_in_the_prompt_when_files_are_sent(seeded, settings):
    p = system_prompt(settings, scoped=False, files=True)
    for needle in ("record_receipt", "chat_search", "start_deep_task", "[ไฟล์ #N", "ห้ามแต่งเนื้อหา",
                   "ไม่ต้องถามก่อน", "1-3"):  # fmt: skip
        assert needle in p, needle
    assert "record_receipt" not in system_prompt(settings, scoped=False)
    assert "อ่านรูปและไฟล์ที่ส่งมา" in qa.CAPABILITIES_PROMPT


# ------------------------------------------------------------------ tools
def _receipt(i=7):
    return Receipt(id=i, guild_id=GUILD, submitter_id=1, submitter_name="A", sha256="s",
                   vendor="Makro", receipt_date="2026-10-14", total=500.0,
                   effective_date="2026-10-14", category="วัตถุดิบ", created_at="x")  # fmt: skip


def _tool_blob(llm):
    out = []
    for req in llm.requests:
        for c in req.contents:
            out += [str(p.function_response) for p in c.parts or [] if p.function_response]
    return json.dumps(out, ensure_ascii=False)


def test_record_receipt_uses_the_chosen_attachment_only(seeded, settings):
    seen = []

    def record(f, note):
        seen.append((f.index, f.filename, f.data, note))
        return IngestResult("created", "บันทึกแล้ว (#7)", "✅", _receipt())

    llm = scripted(
        fc_resp(("record_receipt", {"file_index": 2, "note": "ค่าของ", "guild_id": 5, "submitter_id": 9})),
        text_resp("บันทึกแล้ว"),
    )  # fmt: skip
    files = [fi(0, "a.png", "image/png", b"A"), fi(2, "slip.png", "image/png", b"SLIP")]
    a = ask(settings, seeded, "สลิดนี้", guild_id=GUILD, is_exec=True, user_id=1, model=llm,
            files=files, actions=FileActions(record_receipt=record))  # fmt: skip
    assert seen == [(2, "slip.png", b"SLIP", "ค่าของ")]
    assert a.receipts == [(7, "✅")] and a.text == "บันทึกแล้ว"
    blob = _tool_blob(llm)
    assert "'receipt_id': 7" in blob and "Makro" in blob and "บันทึกแล้ว (#7)" in blob


def test_record_receipt_statuses_and_bad_inputs(seeded, settings):
    def run(actions, call, files=None):
        llm = scripted(fc_resp(call), text_resp("ok"))
        a = ask(settings, seeded, "x", guild_id=GUILD, is_exec=True, user_id=1, model=llm,
                files=files or [fi(0, "a.png", "image/png")], actions=actions)  # fmt: skip
        return a, _tool_blob(llm)

    call = ("record_receipt", {"file_index": 0})
    a, blob = run(FileActions(record_receipt=None), call)
    assert "disabled" in blob and a.receipts == []
    a, blob = run(FileActions(record_receipt=lambda f, n: IngestResult("ignored")), call)
    assert "ignored" in blob and a.receipts == []
    dup = IngestResult("duplicate", "ซ้ำ", "♻️", _receipt())
    a, blob = run(FileActions(record_receipt=lambda f, n: dup), call)
    assert "duplicate" in blob and a.receipts == []  # nothing new to link or react to
    a, blob = run(
        FileActions(record_receipt=lambda f, n: dup), ("record_receipt", {"file_index": 9})
    )
    assert "no file #9" in blob
    a, blob = run(
        FileActions(record_receipt=lambda f, n: dup), call, [fi(0, "n.txt", "text/plain")]
    )
    assert "only an image or PDF" in blob

    def boom(f, n):
        raise RuntimeError("vertex down")

    a, blob = run(FileActions(record_receipt=boom), call)
    assert "could not read this file" in blob and "vertex down" not in blob


def test_start_deep_task_tool(seeded, settings):
    started = []

    def deep(task):
        started.append(task)
        return "started"

    llm = scripted(
        fc_resp(("start_deep_task", {"task": "วิเคราะห์ทั้งไตรมาส"}),
                ("start_deep_task", {"task": "again"})),
        text_resp("เริ่มแล้ว"),
    )  # fmt: skip
    ask(settings, seeded, "x", guild_id=GUILD, is_exec=True, user_id=1, model=llm,
        files=[fi(0, "a.pdf", "application/pdf")], actions=FileActions(start_deep_task=deep))  # fmt: skip
    assert started == ["วิเคราะห์ทั้งไตรมาส"]  # a second call in the same turn does nothing
    assert "started" in _tool_blob(llm) and "already_started" in _tool_blob(llm)
    llm = scripted(fc_resp(("start_deep_task", {"task": "t"})), text_resp("ok"))
    ask(settings, seeded, "x", guild_id=GUILD, is_exec=True, user_id=1, model=llm,
        files=[fi(0, "a.pdf", "application/pdf")], actions=FileActions())  # fmt: skip
    assert "disabled" in _tool_blob(llm)


def test_file_tools_exist_only_with_usable_files_and_actions(seeded, settings):
    def tool_names(llm):
        decls = llm.requests[0].config.tools[0].function_declarations
        return {d.name for d in decls}

    llm = scripted(text_resp("ok"))
    ask(settings, seeded, "x", guild_id=GUILD, is_exec=True, user_id=1, model=llm,
        files=[fi(0, "a.png", "image/png")], actions=FileActions())  # fmt: skip
    assert {"record_receipt", "start_deep_task"} <= tool_names(llm)
    llm = scripted(text_resp("ok"))
    ask(settings, seeded, "x", guild_id=GUILD, is_exec=True, user_id=1, model=llm,
        files=[fi(0, "a.zip", None)], actions=FileActions())  # fmt: skip
    assert "record_receipt" not in tool_names(llm)


# ------------------------------------------------------------------ routing in the bot
class Agent:
    """Stand-in for qa.ask in the bot: records what the bot hands over; `script` may act."""

    def __init__(self):
        self.calls, self.script, self.receipts = [], None, []

    def __call__(self, s, store, q, **kw):
        self.calls.append((q, kw))
        if self.script:
            self.script(kw)
        return SimpleNamespace(text="agent says", seconds=0, rounds=0, receipts=self.receipts)


def wire(bot, monkeypatch, llm=None, status="created"):
    """Real _answer/_intake with the model, ingest and Discord id stubbed."""
    bot.__dict__.pop("_intake", None)
    bot.__dict__.pop("_answer", None)
    bot.llm = llm or FakeClient([])
    agent, ingested, linked = Agent(), [], []
    bot.store.set_reply_message = lambda rid, mid: linked.append((rid, mid))
    monkeypatch.setattr(Sent, "id", 77, raising=False)
    monkeypatch.setattr(botmod, "ask", agent)

    def ingest(*a, **k):
        ingested.append(k)
        if status == "ignored":
            return IngestResult("ignored")
        r = bot.store.add_receipt(_new_receipt(len(ingested)))
        return IngestResult(status, "บันทึกแล้ว", "✅", r)

    monkeypatch.setattr(botmod, "ingest_attachment", ingest)
    return SimpleNamespace(agent=agent, ingested=ingested, linked=linked)


def _new_receipt(n):
    return Receipt(guild_id=GUILD, submitter_id=STAFF, submitter_name="A", sha256=f"x{n}",
                   vendor="Makro", total=500.0, effective_date="2026-10-14", category="วัตถุดิบ",
                   created_at="2026-10-14")  # fmt: skip


def photo_msg(world, chan, text="", who=STAFF, name="IMG_1.jpg", ctype="image/jpeg", **kw):
    return TMsg(world.g, chan, member(world.g, who), text, attachments=[att(name, ctype)], **kw)


async def test_image_only_message_in_chat_mode_goes_to_the_agent(bot, world, monkeypatch):
    llm = FakeClient([note_resp("สลิป", "500", "receipt/invoice")])
    w = wire(bot, monkeypatch, llm)
    m = photo_msg(world, world.chat)
    await bot.on_message(m)
    await settle(bot)
    ((q, kw),) = w.agent.calls
    assert q == "" and [(f.index, f.filename, f.mime) for f in kw["files"]] == [
        (0, "IMG_1.jpg", "image/jpeg")
    ]
    assert kw["files"][0].data == b"\x89PNG-bytes" and isinstance(kw["actions"], FileActions)
    assert m.replies == ["agent says"]
    assert w.ingested == []  # no second, separate receipt intake
    assert len(image_calls(llm)) == 1  # memory still gets the cheap description


async def test_receipt_named_channel_keeps_direct_intake(bot, world, monkeypatch):
    w = wire(bot, monkeypatch)
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(receipt_channel_ids=["10"]), "x")
    m = photo_msg(world, world.gen)
    await bot.on_message(m)
    await settle(bot)
    assert len(w.ingested) == 1 and m.replies == ["บันทึกแล้ว"] and w.agent.calls == []
    set_mode(bot, 10, "listen")
    q = photo_msg(world, world.gen, who=EXEC)
    await bot.on_message(q)
    assert len(w.ingested) == 2 and q.replies == [] and w.agent.calls == []  # quiet


async def test_unaddressed_image_in_call_mode_keeps_the_silent_path(bot, world, monkeypatch):
    llm = FakeClient([note_resp("ใบเสร็จ", "Makro", "receipt/invoice")])
    w = wire(bot, monkeypatch, llm)
    m = photo_msg(world, world.gen)
    await bot.on_message(m)
    await settle(bot)
    assert w.agent.calls == [] and len(w.ingested) == 1  # describer -> receipt-like -> intake
    assert m.replies == ["บันทึกแล้ว"]


async def test_tagged_image_in_call_mode_goes_to_the_agent_without_double_intake(
    bot, world, monkeypatch
):
    llm = FakeClient([note_resp("ใบเสร็จ", "Makro", "receipt/invoice")])
    w = wire(bot, monkeypatch, llm)
    m = photo_msg(world, world.gen, "ช่วยดูหน่อย", mention=True)
    await bot.on_message(m)
    await settle(bot)
    assert len(w.agent.calls) == 1 and w.agent.calls[0][0] == "ช่วยดูหน่อย"
    assert w.ingested == [] and m.replies == ["agent says"]


async def test_name_call_and_reply_to_salid_with_a_file_go_to_the_agent(bot, world, monkeypatch):
    w = wire(bot, monkeypatch)
    named = photo_msg(world, world.gen, "สลิด ดูนี่")
    await bot.on_message(named)
    replying = photo_msg(world, world.gen, "อันนี้ล่ะ", who=EXEC)
    replying.reference = SimpleNamespace(resolved=SimpleNamespace(author=ME), message_id=5)
    await bot.on_message(replying)
    await settle(bot)
    assert len(w.agent.calls) == 2 and w.ingested == []


async def test_in_window_file_goes_through_the_gate_then_the_agent(bot, world, monkeypatch):
    w = wire(bot, monkeypatch)
    first = await _say(bot, world, "สลิด สวัสดี")
    assert first.replies == ["agent says"]
    bot.gate.window = True
    m = photo_msg(world, world.gen)
    await bot.on_message(m)
    assert bot.gate.window_calls == 1 and len(w.agent.calls) == 2 and m.replies == ["agent says"]


async def _say(bot, world, text):
    m = TMsg(world.g, world.gen, member(world.g, STAFF), text)
    await bot.on_message(m)
    return m


async def test_listen_mode_tag_with_an_image_is_answered_but_nothing_else(bot, world, monkeypatch):
    w = wire(bot, monkeypatch)
    set_mode(bot, 10, "listen")
    tagged = photo_msg(world, world.gen, "นี่อะไร", mention=True)
    plain = photo_msg(world, world.gen, who=EXEC)
    await bot.on_message(tagged)
    await bot.on_message(plain)
    await settle(bot)
    assert len(w.agent.calls) == 1 and tagged.replies == ["agent says"] and plain.replies == []


async def test_a_pdf_and_a_text_file_go_to_the_agent_too(bot, world, monkeypatch):
    w = wire(bot, monkeypatch)
    m = TMsg(world.g, world.chat, member(world.g, STAFF), "",
             attachments=[att("q.pdf", "application/pdf"), att("n.txt", "text/plain"),
                          att("z.zip", "application/zip", size=5)])  # fmt: skip
    await bot.on_message(m)
    files = w.agent.calls[0][1]["files"]
    assert [(f.index, f.mime, bool(f.data), f.skip) for f in files] == [
        (0, "application/pdf", True, ""),
        (1, "text/plain", True, ""),
        (2, None, False, "unsupported file type"),
    ]


async def test_oversize_attachment_is_not_downloaded(bot, world, monkeypatch):
    w = wire(bot, monkeypatch)
    FakeAtt.reads = 0
    m = TMsg(world.g, world.chat, member(world.g, STAFF), "ดู",
             attachments=[att("huge.png", "image/png", size=16 * MB)])  # fmt: skip
    await bot.on_message(m)
    (f,) = w.agent.calls[0][1]["files"]
    assert f.skip == "over 15 MB" and f.data == b"" and FakeAtt.reads == 0


async def test_record_receipt_is_bound_to_this_message_and_author(bot, world, monkeypatch):
    w = wire(bot, monkeypatch)
    results = []

    def act(kw):
        results.append(kw["actions"].record_receipt(kw["files"][0], "ค่าของ"))

    w.agent.script = act
    m = photo_msg(world, world.chat, "บันทึกให้ด้วย", who=EXEC)
    await bot.on_message(m)
    (res,) = results
    (k,) = w.ingested
    assert res.status == "created"
    assert (k["guild_id"], k["submitter_id"], k["channel_id"], k["message_id"]) == (
        GUILD, EXEC, world.chat.id, m.id,
    )  # fmt: skip
    assert k["filename"] == "IMG_1.jpg" and k["data"] == b"\x89PNG-bytes"
    assert k["note"] == "บันทึกให้ด้วย ค่าของ" and k["submitter_name"] == f"u{EXEC}"


async def test_reply_message_is_linked_after_a_created_receipt(bot, world, monkeypatch):
    w = wire(bot, monkeypatch)
    w.agent.receipts = [(41, "⚠️")]
    m = photo_msg(world, world.chat)
    await bot.on_message(m)
    assert w.linked == [(41, 77)] and m.reactions == ["⚠️"] and m.replies == ["agent says"]
    w.agent.receipts = []
    m2 = photo_msg(world, world.chat)
    await bot.on_message(m2)
    assert w.linked == [(41, 77)]  # nothing created, nothing linked


async def test_record_receipt_is_disabled_when_the_feature_is_off(bot, world, monkeypatch):
    w = wire(bot, monkeypatch)
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features=dict(receipts=False)), "x")
    bot.team._cache.clear()
    await bot.on_message(photo_msg(world, world.chat))
    actions = w.agent.calls[0][1]["actions"]
    assert actions.record_receipt is None  # the tool answers "disabled"


async def test_start_deep_task_permissions_and_flag(bot, world, monkeypatch):
    w = wire(bot, monkeypatch)
    launched = []
    monkeypatch.setattr(bot, "_launch_deep", lambda *a: launched.append(a[2]))
    statuses = []
    w.agent.script = lambda kw: statuses.append(
        kw["actions"].start_deep_task("วิเคราะห์") if kw["actions"].start_deep_task else "off"
    )
    await bot.on_message(photo_msg(world, world.chat, who=STAFF))  # not an executive
    bot.s.gemini_cli_path = ""
    await bot.on_message(photo_msg(world, world.chat, who=EXEC))  # no Gemini CLI
    bot.s.gemini_cli_path = "/bin/gemini"
    await bot.on_message(photo_msg(world, world.chat, who=EXEC))  # allowed
    await asyncio.sleep(0)
    assert statuses == ["not_executive", "disabled", "started"] and launched == ["วิเคราะห์"]
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features=dict(deep_tasks=False)), "x")
    bot.team._cache.clear()
    await bot.on_message(photo_msg(world, world.chat, who=EXEC))
    assert statuses[-1] == "off" and len(launched) == 1


async def test_busy_deep_task_is_reported(bot, world, monkeypatch):
    bot.s.gemini_cli_path = "/bin/gemini"
    pending = asyncio.get_running_loop().create_future()
    bot._deep_tasks.add(pending)
    assert bot._deep_status(GUILD, "x", True) == "busy"
    pending.set_result(None)
    assert (
        bot._deep_status(GUILD, "x", True) == "ok" and bot._deep_status(GUILD, "", True) == "empty"
    )
    bot._deep_tasks.clear()
