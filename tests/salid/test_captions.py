# ruff: noqa: F811, F401
"""Live captions: off by default, batched "**Name:** text", optional cheap-model translation,
stop with recording / the meeting, and the set_captions tools (chat agent and Live)."""

import asyncio
from types import SimpleNamespace

from conftest import text_resp
from test_live_voice import lv, tool_msg, until  # noqa: F401
from test_meeting import env, make_member, segment, start

from salid import meeting as mt
from salid.captions import (
    CaptionState,
    captions_tools,
    format_captions,
    normalize_language,
    translate_lines,
)


def test_normalize_language():
    assert normalize_language("English") == "en" and normalize_language("ไทย") == "th"
    assert normalize_language("") == "original" and normalize_language("klingon") is None


def test_format_joins_a_speakers_turn_and_splits_long_output():
    out = format_captions([("แอน", "สวัสดี"), ("แอน", "ครับ"), ("บอย", "hi")])
    # 9f83597: a caption is a quote block with a microphone, not a typed message
    assert out == ["> 🎙️ **แอน:** สวัสดี ครับ\n> 🎙️ **บอย:** hi"]
    two_lines = format_captions([("แอน", "สวัสดี\n↳ (English translation) hello")])
    assert two_lines == ["> 🎙️ **แอน:** สวัสดี\n> ↳ (English translation) hello"]
    many = format_captions([(f"p{i % 2}", "x" * 300) for i in range(20)])
    assert len(many) > 1 and all(len(m) <= 1800 for m in many)


def test_state_is_off_by_default_and_buffers_only_when_on():
    c = CaptionState()
    c.add("a", "hello")
    assert c.pending == [] and not c.due(100.0)
    c.turn_on("en", 0.0)
    c.add("a", "hello")
    assert not c.due(5.0) and c.due(13.0)
    assert c.take(13.0) == [("a", "hello")] and not c.due(14.0)
    c.turn_off()
    assert not c.on and c.pending == []


def test_translate_is_one_call_and_falls_back_to_the_original():
    from conftest import FakeClient

    client = FakeClient([text_resp('["hello", "bye"]')])
    assert translate_lines(client, "m", ["สวัสดี", "ลาก่อน"], "en") == ["hello", "bye"]
    assert len(client.models.calls) == 1
    bad = FakeClient([text_resp("not json")])
    assert translate_lines(bad, "m", ["a", "b"], "th") == ["a", "b"]
    assert translate_lines(bad, "m", ["a"], "original") == ["a"]  # no call


def test_tool_needs_a_meeting_and_a_valid_language():
    calls = []
    sess = object()
    (tool,) = captions_tools(lambda: None, lambda *a: calls.append(a) or {})
    assert tool.__name__ == "set_captions" and tool(True)["error"] == "not_in_meeting_voice_channel"
    (tool,) = captions_tools(lambda: sess, lambda *a: calls.append(a) or {"status": "on"})
    assert "error" in tool(True, "klingon") and tool(True, "English") == {"status": "on"}
    assert calls == [(sess, True, "en")]


async def test_captions_are_off_by_default_then_posted_in_batches_without_pings(env, monkeypatch):
    s = await start(env)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "เรามาเริ่มกัน")
    await env.mgr.handle_segment(s, segment(10))
    assert not s.captions.on and s.captions.pending == []
    kwargs = []
    orig = env.chan.send

    async def send(text, **kw):
        kwargs.append(kw)
        return await orig(text, **kw)

    env.chan.send = send
    assert env.mgr.set_captions(s, True, "original") == {"status": "on", "language": "original"}
    await env.mgr.handle_segment(s, segment(10))
    await env.mgr.handle_segment(s, segment(11))
    await env.mgr._caption_flush(s)
    assert env.chan.sent[-1] == "> 🎙️ **แอน:** เรามาเริ่มกัน\n> 🎙️ **บอย:** เรามาเริ่มกัน"
    assert (
        kwargs[-1]["allowed_mentions"].everyone is False
        and kwargs[-1]["allowed_mentions"].users is False
    )
    await env.mgr.end((1000, 77), "shutdown")


async def test_translated_captions_use_the_cheap_model_once_per_batch(env, monkeypatch):
    s = await start(env)
    monkeypatch.setattr(mt, "transcribe_pcm", lambda c, m, pcm: "สวัสดี")
    env.mgr.set_captions(s, True, "en")
    await env.mgr.handle_segment(s, segment(10))
    await env.mgr.handle_segment(s, segment(11))
    env.llm.models.script = [text_resp('["hello", "hi"]')]
    await env.mgr._caption_flush(s)
    assert env.chan.sent[-1] == "> 🎙️ **แอน:** hello\n> 🎙️ **บอย:** hi"
    assert env.llm.models.calls[-1][0] == env.bot.s.gate_model
    await env.mgr.end((1000, 77), "shutdown")


async def test_not_recording_stops_captions_until_asked_again(env, monkeypatch):
    s = await start(env)
    env.mgr.set_captions(s, True, "original")
    env.mgr.set_recording(s, False)
    assert not s.captions.on
    env.mgr.set_captions(s, True, "th")  # explicit request after: allowed
    assert s.captions.on and s.recording == "not_recording"
    await env.mgr.end((1000, 77), "shutdown")


async def test_captions_stop_when_the_meeting_ends(env):
    s = await start(env)
    env.mgr.set_captions(s, True, "original")
    s.captions.add("แอน", "x")
    await env.mgr.end((1000, 77), "shutdown")
    assert not s.captions.on and s.captions.pending == []
    assert env.mgr.set_captions(s, True)["error"] == "meeting_ending"


async def test_chat_tool_only_for_someone_in_the_voice_channel(env):
    s = await start(env)
    a = env.guild.members[10]
    outsider = make_member(env.guild, 99, "ซี", None)
    (tool, *_) = env.bot._captions_tools(a)
    assert tool(True, "en")["status"] == "on" and s.captions.language == "en"
    (tool, *_) = env.bot._captions_tools(outsider)
    assert tool(False)["error"] == "not_in_meeting_voice_channel" and s.captions.on
    await env.mgr.end((1000, 77), "shutdown")


async def test_live_tool_turns_captions_on_and_off(lv):
    s, live = await lv.join(5)
    live.push(tool_msg("set_captions", {"on": True, "language": "en"}))
    await until(lambda: live.tool_responses)
    assert live.tool_responses[0][0].response["status"] == "on" and s.captions.language == "en"
    live.push(tool_msg("set_captions", {"on": False}, id="c2"))
    await until(lambda: len(live.tool_responses) == 2)
    assert not s.captions.on
    await lv.mgr.end((1000, 77))
