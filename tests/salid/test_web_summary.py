# ruff: noqa: F811, F401
"""read_web (links + Google Search, separate Gemini call) and the direct summary features
(/สรุป, reply "สรุป", message context menu). Offline: fake genai client, scripted ADK model."""

import asyncio
import json
from datetime import date
from types import SimpleNamespace

import discord
import pytest
from conftest import GUILD, FakeClient, fc_resp, scripted, text_resp
from test_bot_multitenant import member
from test_memory import RChan, rworld
from test_modes import EXEC, MANAGER, STAFF, TMsg, bot, set_mode, world

from salid import bot as botmod
from salid import qa, web_read
from salid.bot import MENU_SUMMARIZE, SalidTranslator
from salid.guild_settings import GuildSettingsUpdate
from salid.i18n import t
from salid.summary import is_summarize_request, period_range, summary_question


# ------------------------------------------------------------------ read_web
def web_resp(text="สรุปหน้าเว็บ", chunks=(), url_meta=()):
    cand = SimpleNamespace(
        grounding_metadata=SimpleNamespace(
            grounding_chunks=[
                SimpleNamespace(web=SimpleNamespace(uri=u, title=ti)) for u, ti in chunks
            ]
        ),
        url_context_metadata=SimpleNamespace(
            url_metadata=[
                SimpleNamespace(retrieved_url=u, url_retrieval_status=s) for u, s in url_meta
            ]
        ),
    )
    return SimpleNamespace(text=text, candidates=[cand])


def outbound_text(client):
    (_model, contents, _config) = client.models.calls[0]
    return " ".join(p.text or "" for c in contents for p in c.parts)


def test_read_web_uses_both_builtin_tools_and_parses_sources():
    client = FakeClient(
        [
            web_resp(
                chunks=[("https://g/redirect1", "Example"), ("https://g/redirect1", "dup")],
                url_meta=[
                    ("https://a.example/x", "URL_RETRIEVAL_STATUS_SUCCESS"),
                    ("https://b.example/paywall", "URL_RETRIEVAL_STATUS_ERROR"),
                ],
            )
        ]
    )
    out = web_read.read_web_call(client, "gemini-x", "ดูให้หน่อย https://a.example/x", "สรุป")
    model, contents, config = client.models.calls[0]
    assert model == "gemini-x"
    tools = config.tools
    assert any(t_.url_context is not None for t_ in tools)
    assert any(t_.google_search is not None for t_ in tools)
    assert config.max_output_tokens >= 2048
    assert "do not invent" in config.system_instruction and "paywalled" in config.system_instruction
    assert out["text"] == "สรุปหน้าเว็บ"
    assert out["sources"] == [
        {"title": "Example", "uri": "https://g/redirect1"},
        {"title": "a.example", "uri": "https://a.example/x"},
    ]
    assert out["unreadable"] == ["https://b.example/paywall"]


@pytest.mark.parametrize(
    "url, yes",
    [
        ("https://www.youtube.com/watch?v=abc", True),
        ("https://youtu.be/abc", True),
        ("https://m.youtube.com/shorts/abc", True),
        ("https://youtube.com/@channel", False),
        ("https://notyoutube.com/watch?v=1", False),
        ("https://example.com/a", False),
    ],
)
def test_youtube_detection(url, yes):
    assert web_read.is_youtube(url) is yes


def test_youtube_links_become_video_parts():
    client = FakeClient([web_resp()])
    web_read.read_web_call(
        client, "m", "https://youtu.be/abc และ https://example.com/p", "what is it"
    )
    (_m, contents, _c) = client.models.calls[0]
    parts = contents[0].parts
    videos = [p for p in parts if p.file_data is not None]
    assert [(v.file_data.file_uri, v.file_data.mime_type) for v in videos] == [
        ("https://youtu.be/abc", "video/*")
    ]
    assert "https://example.com/p" in parts[0].text


def test_url_cap_notes_the_ignored_links():
    urls = [f"https://s{i}.example/p" for i in range(8)]
    client = FakeClient([web_resp()])
    out = web_read.read_web_call(client, "m", " ".join(urls), "read these")
    sent = outbound_text(client)
    assert all(u in sent for u in urls[:5]) and not any(u in sent for u in urls[5:])
    assert "3 more link(s) ignored" in out["note"]


def test_failures_return_an_honest_error_not_an_exception():
    class Boom:
        class models:
            @staticmethod
            def generate_content(**_):
                raise RuntimeError("secret detail")

    out = web_read.read_web_call(Boom, "m", "https://youtu.be/abc", "x")
    assert "error" in out and "secret detail" not in out["error"]
    out = web_read.read_web_call(FakeClient([web_resp(text="")]), "m", "q", "q")
    assert "error" in out


def test_extract_urls_trims_punctuation_and_dedupes():
    assert web_read.extract_urls("see (https://a.example/x). and https://a.example/x, ok") == [
        "https://a.example/x"
    ]


def tool_names(llm):
    return {d.name for d in llm.requests[0].config.tools[0].function_declarations}


def test_read_web_registered_only_with_a_web_client(seeded, settings):
    llm = scripted(text_resp("ok"))
    qa.ask(settings, seeded, "hi", guild_id=GUILD, is_exec=True, user_id=1, model=llm)
    assert "read_web" not in tool_names(llm)
    assert "read_web" not in str(llm.requests[0].config.system_instruction)
    llm = scripted(text_resp("ok"))
    qa.ask(settings, seeded, "hi", guild_id=GUILD, is_exec=True, user_id=1, model=llm,
           web_client=FakeClient([]))  # fmt: skip
    assert "read_web" in tool_names(llm)
    assert "read_web" in str(llm.requests[0].config.system_instruction)


def test_outbound_request_contains_only_the_user_text_and_the_capped_request(seeded, settings):
    """A ledger result the model saw must not reach Google, except via the 500-char request."""
    client = FakeClient([web_resp()])
    long_request = "สรุปให้ " + "x" * 2000
    llm = scripted(
        fc_resp(("list_receipts", {"start_date": "2026-10-01", "end_date": "2026-10-31"})),
        fc_resp(("read_web", {"request": long_request})),
        text_resp("เสร็จ"),
    )
    ans = qa.ask(settings, seeded, "ดู https://a.example/x", guild_id=GUILD, is_exec=True,
                 user_id=1, history="[12:00] boss: งบลับ 999999", model=llm,
                 web_client=client)  # fmt: skip
    assert ans.text == "เสร็จ"
    sent = outbound_text(client)
    for secret in ("Makro", "Grab", "999999", "งบลับ", "total"):
        assert secret not in sent
    assert "https://a.example/x" in sent
    assert long_request[:500] in sent and long_request[:600] not in sent
    assert ("read_web", {"domains": ["a.example"]}) in ans.tool_calls  # domains only


def test_prompt_guidance_and_capabilities(settings):
    p = qa.system_prompt(settings, scoped=False, web=True)
    for needle in ("read_web", "YouTube", "ลิงก์", "ledger", "ห้ามใส่ข้อมูลบัญชี"):
        assert needle in p, needle
    assert "read_web" not in qa.system_prompt(settings, scoped=False)
    assert "ค้นอินเทอร์เน็ตไม่ได้" not in qa.CAPABILITIES_PROMPT
    assert "Google Search" in qa.CAPABILITIES_PROMPT
    assert "/สรุป" in qa.CAPABILITIES_PROMPT and "สรุปด้วยสลิด" in qa.CAPABILITIES_PROMPT


def test_web_feature_flag_in_guild_settings(bot, world):
    assert bot.team.enabled(GUILD, "web")
    bot.llm = FakeClient([])
    assert bot._web_kw(GUILD) == {"web_client": bot.llm}
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features=dict(web=False)), "x")
    bot.team._cache.clear()
    assert bot._web_kw(GUILD) == {}


# ------------------------------------------------------------------ summary pure functions
def test_period_ranges():
    today = date(2026, 10, 14)
    assert period_range("today", today) == (today, today)
    assert period_range("yesterday", today) == (date(2026, 10, 13),) * 2
    assert period_range("7d", today) == (date(2026, 10, 8), today)
    assert period_range("30d", today) == (date(2026, 9, 15), today)


def test_summary_question_per_period_language_and_topic():
    d = date(2026, 10, 14)
    q = summary_question("th", period="yesterday", today=d, tz="Asia/Bangkok", channel_name="ads")
    assert q.startswith("สรุปสิ่งที่คุยกันใน #ads ช่วง 2026-10-13 (เขตเวลา Asia/Bangkok):")
    for part in ("หัวข้อ", "สิ่งที่ตัดสินใจ", "ใครรับงานอะไร", "ตัวเลขสำคัญ", "เรื่องที่ค้าง"):
        assert part in q
    q = summary_question("en", period="7d", today=d, tz="Asia/Bangkok", channel_name="ads",
                         topic="budget")  # fmt: skip
    assert "#ads for 2026-10-08 to 2026-10-14" in q and q.endswith("[focus: budget]")
    q = summary_question("th", period="today", today=d, tz="UTC", topic="งบ")
    assert "ทุกห้องที่ฉันเห็น" in q and "[เน้น: งบ]" in q


@pytest.mark.parametrize(
    "text",
    ["สรุป", "สรุปหน่อย", "สลิด สรุปหน่อยครับ", "summarize", "Summarise", "tl;dr", "TLDR",
     "summarize this", "ช่วยสรุปหน่อย", "<@999> สรุป"],
)  # fmt: skip
def test_summarize_requests(text):
    assert is_summarize_request(text.replace("<@999>", "").strip())


@pytest.mark.parametrize(
    "text", ["สรุปยอดเดือนนี้", "summarize the meeting notes please", "", "ขอบคุณ", "สรุปว่าไงดี"]
)
def test_not_summarize_requests(text):
    assert not is_summarize_request(text)


# ------------------------------------------------------------------ /สรุป
class Interaction:
    def __init__(self, guild, user, channel, locale=None):
        self.guild, self.user, self.channel, self.locale = guild, user, channel, locale
        self.deferred, self.sent = None, []
        self.response = SimpleNamespace(defer=self._defer)
        self.followup = SimpleNamespace(send=self._send)

    async def _defer(self, ephemeral=False):
        self.deferred = ephemeral

    async def _send(self, text, ephemeral=False):
        self.sent.append((text, ephemeral))


@pytest.fixture
def agent(monkeypatch):
    seen = []

    def fake(s, store, q, **kw):
        seen.append((q, kw))
        return SimpleNamespace(text="สรุปแล้ว", seconds=0, rounds=0, tool_calls=[], receipts=[])

    monkeypatch.setattr(botmod, "ask", fake)
    return seen


def readers(bot):
    bot._viewers.clear()


async def test_summary_command_question_scope_and_followup(bot, rworld, agent):
    readers(bot)
    it = Interaction(rworld.g, rworld.staff, rworld.g.text_channels[0])  # staff in #ทั่วไป
    await bot.handle_summary(it, "yesterday", None, False, "งบ", False)
    ((q, kw),) = agent
    assert it.deferred is False and it.sent == [("สรุปแล้ว", False)]
    assert q.startswith("สรุปสิ่งที่คุยกันใน #ทั่วไป ช่วง 2026-10-13 (เขตเวลา Asia/Bangkok):")
    assert q.endswith("[เน้น: งบ]")
    # staff reads #ทั่วไป and #ads; #ads has more readers than #ทั่วไป's audience? both include
    # staff+manager, so allowed channels = the ones whose readers cover the answering audience
    assert set(kw["memory"].allowed) <= {10, 12} and 10 in kw["memory"].allowed
    assert 11 not in kw["memory"].allowed  # #ผู้บริหาร: staff cannot read it


async def test_summary_all_channels_private_uses_only_the_invokers_access(bot, rworld, agent):
    readers(bot)
    it = Interaction(rworld.g, rworld.staff, rworld.g.text_channels[0])
    await bot.handle_summary(it, "7d", None, True, None, True)
    ((q, kw),) = agent
    assert it.deferred is True and it.sent == [("สรุปแล้ว", True)]
    assert "ทุกห้องที่ฉันเห็น" in q and "2026-10-08 ถึง 2026-10-14" in q
    assert kw["memory"].allowed == {10: "ทั่วไป", 12: "ads"}  # DM-like: what staff can read
    readers(bot)
    boss = Interaction(rworld.g, rworld.boss, rworld.g.text_channels[1])
    await bot.handle_summary(boss, "today", None, True, None, True)
    assert set(agent[1][1]["memory"].allowed) == {11, 12}


async def test_summary_refuses_a_channel_the_invoker_cannot_read(bot, rworld, agent):
    readers(bot)
    it = Interaction(rworld.g, rworld.staff, rworld.g.text_channels[0])
    await bot.handle_summary(it, "today", rworld.g.text_channels[1], False, None, True)
    assert agent == [] and it.sent == [(t("summary_no_access", "th"), True)]


async def test_summary_needs_memory_and_qa(bot, rworld, agent):
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features=dict(memory=False)), "x")
    bot.team._cache.clear()
    it = Interaction(rworld.g, rworld.staff, rworld.g.text_channels[0])
    await bot.handle_summary(it, "today", None, False, None, False)
    assert agent == [] and it.sent == [(t("summary_unavailable", "th"), False)]


async def test_summary_question_follows_the_channel_language(bot, rworld, agent):
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(default_language="en"), "x")
    bot.team._cache.clear()
    readers(bot)
    it = Interaction(rworld.g, rworld.staff, rworld.g.text_channels[0])
    await bot.handle_summary(it, "today", None, False, None, False)
    assert agent[0][0].startswith("Summarize what was discussed in #ทั่วไป for 2026-10-14")
    assert agent[0][1]["language"] == "en"


async def test_summary_replies_are_chunked(bot, rworld, monkeypatch):
    monkeypatch.setattr(
        botmod, "ask",
        lambda s, st, q, **kw: SimpleNamespace(text="ก" * 4000, seconds=0, rounds=0),
    )  # fmt: skip
    readers(bot)
    it = Interaction(rworld.g, rworld.staff, rworld.g.text_channels[0])
    await bot.handle_summary(it, "today", None, False, None, False)
    assert [len(x) for x, _ in it.sent] == [1900, 1900, 200]


def test_commands_are_bilingual_with_period_choices_and_a_th_context_menu(bot):
    bot._register_commands()
    for name in ("สรุป", "summary"):
        cmd = bot.tree.get_command(name)
        assert cmd.description == t("cmd_summary")
        params = {p.name: p for p in cmd.parameters}
        assert set(params) == {"period", "channel", "all_channels", "topic", "private"}
        assert [c.name for c in params["period"].choices] == [
            "วันนี้ · Today", "เมื่อวาน · Yesterday", "7 วัน · Last 7 days", "30 วัน · Last 30 days",
        ]  # fmt: skip
        assert [c.value for c in params["period"].choices] == ["today", "yesterday", "7d", "30d"]
        assert params["period"].default == "today" and not params["period"].required
        assert all(" · " in p.description for p in params.values())
        assert cmd.guild_only is True and cmd.default_permissions is None  # open to anyone
    menu = bot.tree.get_command(MENU_SUMMARIZE, type=discord.AppCommandType.message)
    assert menu is not None and menu.guild_only
    payload = asyncio.run(menu.get_translated_payload(bot.tree, SalidTranslator()))
    assert payload["name"] == "Summarize with Salid"
    assert payload["name_localizations"] == {"th": "สรุปด้วยสลิด"}


# ------------------------------------------------------------------ summarise one message
class FakeAtt(SimpleNamespace):
    async def read(self):
        return b"\x89PNG-bytes"


def target_msg(world, text="ดูรายงาน https://a.example/r", attachments=()):
    return TMsg(world.g, world.gen, member(world.g, MANAGER), text, attachments=list(attachments))


def reply_to(world, target, text="สลิด สรุปหน่อย", **kw):
    m = TMsg(world.g, world.gen, member(world.g, STAFF), text, **kw)
    m.reference = SimpleNamespace(resolved=target, message_id=target.id)
    return m


async def test_reply_summarize_passes_text_links_and_the_referenced_attachments(bot, world, agent):
    bot.__dict__.pop("_answer", None)
    bot.llm = FakeClient([])
    att = FakeAtt(filename="r.png", content_type="image/png", size=10)
    target = target_msg(world, attachments=[att])
    m = reply_to(world, target)
    await bot.on_message(m)
    ((q, kw),) = agent
    assert "https://a.example/r" in q and "u3" in q  # text, link and the author
    assert [(f.filename, f.mime, f.data) for f in kw["files"]] == [
        ("r.png", "image/png", b"\x89PNG-bytes")
    ]
    assert "actions" not in kw and kw["web_client"] is bot.llm and "memory" not in kw
    assert m.replies == ["สรุปแล้ว"]


async def test_reply_summarize_in_chat_mode_needs_no_name_and_web_flag_off(bot, world, agent):
    bot.__dict__.pop("_answer", None)
    set_mode(bot, 11, "chat")
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features=dict(web=False)), "x")
    bot.team._cache.clear()
    bot.llm = FakeClient([])
    target = TMsg(world.g, world.chat, member(world.g, MANAGER), "ประชุมพรุ่งนี้ 10 โมง")
    m = reply_to(world, target, "tl;dr")
    m.channel = world.chat
    await bot.on_message(m)
    ((q, kw),) = agent
    assert "ประชุมพรุ่งนี้ 10 โมง" in q and "web_client" not in kw and "files" not in kw


async def test_summarize_without_a_reference_is_a_normal_question(bot, world, agent):
    bot.__dict__.pop("_answer", None)
    m = TMsg(world.g, world.gen, member(world.g, STAFF), "สลิด สรุป")
    await bot.on_message(m)
    ((q, kw),) = agent
    assert q == "สลิด สรุป" and "files" not in kw


async def test_context_menu_summarizes_the_clicked_message_ephemerally(bot, world, agent):
    bot.llm = FakeClient([])
    att = FakeAtt(filename="n.txt", content_type="text/plain", size=10)
    target = target_msg(world, "", attachments=[att])
    it = Interaction(world.g, member(world.g, STAFF), world.gen)
    await bot.handle_summarize_menu(it, target)
    ((q, kw),) = agent
    assert it.deferred is True and it.sent == [("สรุปแล้ว", True)]
    assert [f.filename for f in kw["files"]] == ["n.txt"] and "actions" not in kw
    agent.clear()
    empty = target_msg(world, "")
    it2 = Interaction(world.g, member(world.g, STAFF), world.gen)
    await bot.handle_summarize_menu(it2, empty)
    assert agent == [] and it2.sent == [(t("summary_nothing", "th"), True)]
