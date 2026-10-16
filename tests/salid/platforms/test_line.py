"""LINE: signature, fast 200, event translation, reply-token vs push, splitting."""

import base64
import hashlib
import hmac
import json

import pytest
from conftest import FixedClockSettings
from fastapi.testclient import TestClient
from platform_fakes import Clock, inc, make_assistant, rid_of, stored

from salid.platforms import ids, line, textfmt
from salid.platforms.config import PlatformSettings
from salid.platforms.line import LineAdapter, LinePort, verify_signature

SECRET, BOT = "channel-secret", "Ubot"


def sign(body: bytes, secret=SECRET) -> str:
    return base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()


class FakeLineApi:
    def __init__(self):
        self.replies, self.pushes, self.loadings = [], [], []
        self.fail_reply = False
        self.blobs = {"img1": b"imgbytes", "file1": b"pdfbytes"}
        self.names = {"Ualice": "Alice", "Ubob": "Bob"}

    def reply(self, token, texts):
        if self.fail_reply:
            raise RuntimeError("Invalid reply token")
        self.replies.append((token, list(texts)))
        return [f"r{i}" for i in range(len(texts))]

    def push(self, to, texts):
        self.pushes.append((to, list(texts)))
        return [f"p{i}" for i in range(len(texts))]

    def loading(self, chat_id, seconds):
        self.loadings.append((chat_id, seconds))

    def profile(self, kind, room, user):
        return self.names[user]

    def group_name(self, kind, room):
        return "Ops team"

    def content(self, message_id):
        return self.blobs[message_id]

    def bot_info(self):
        return {"user_id": BOT}


@pytest.fixture
def s():
    return FixedClockSettings(exec_user_ids=set())


@pytest.fixture
def fx(s, tmp_path):
    api = FakeLineApi()
    clock = Clock(500.0)
    a = make_assistant(s, tmp_path)
    port = LinePort(api, a.ps, clock=clock)
    adapter = LineAdapter(a, port, BOT, clock=clock)
    return api, a, port, adapter, clock


def mev(text="hi", *, src="group", mid="m1", token="tok1", mentionees=None, **kw):
    source = {"type": src, "userId": "Ualice"}
    if src == "group":
        source["groupId"] = "Cgroup"
    if src == "room":
        source["roomId"] = "Rroom"
    message = {"type": "text", "id": mid, "text": text}
    if mentionees is not None:
        message["mention"] = {"mentionees": mentionees}
    ev = {"type": "message", "mode": "active", "timestamp": 1791954000000,
          "source": source, "replyToken": token, "message": message}  # fmt: skip
    ev.update(kw)
    return ev


# ------------------------------------------------------------------ signature


def test_signature_valid_invalid_missing():
    body = b'{"events":[]}'
    assert verify_signature(SECRET, body, sign(body))
    assert not verify_signature(SECRET, body, sign(body, "other-secret"))
    assert not verify_signature(SECRET, body + b" ", sign(body))
    assert not verify_signature(SECRET, body, None)
    assert not verify_signature(SECRET, body, "")
    assert not verify_signature("", body, sign(body, ""))  # an unset secret never verifies


def test_signature_is_compared_in_constant_time(monkeypatch):
    calls = []
    real = hmac.compare_digest
    monkeypatch.setattr(line.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    verify_signature(SECRET, b"x", sign(b"x"))
    assert calls


# ------------------------------------------------------------------ webhook


class Recorder:
    """Stands in for the adapter: records events and the order things happened in."""

    def __init__(self):
        self.events, self.order = [], []

    async def on_event(self, ev, destination=""):
        self.order.append("handled")
        self.events.append((ev, destination))


def post(client, payload, *, signature="auto", raw=None):
    body = raw if raw is not None else json.dumps(payload).encode()
    headers = {}
    if signature == "auto":
        headers["X-Line-Signature"] = sign(body)
    elif signature:
        headers["X-Line-Signature"] = signature
    return client.post(line.WEBHOOK_PATH, content=body, headers=headers)


def test_unsigned_and_badly_signed_webhooks_are_rejected():
    rec = Recorder()
    pending = []
    client = TestClient(line.create_app(SECRET, rec, spawn=pending.append))
    assert post(client, {"events": [{}]}, signature=None).status_code == 401
    assert post(client, None, raw=b" " * (line.MAX_BODY + 1), signature=None).status_code == 401
    assert post(client, {"events": [{}]}, signature="AAAA").status_code == 401
    assert post(client, {"events": [{}]}, signature=sign(b"{}", "wrong")).status_code == 401
    assert pending == [] and rec.events == []  # nothing was scheduled


def test_webhook_returns_200_before_the_event_is_handled():
    rec = Recorder()
    pending = []
    client = TestClient(line.create_app(SECRET, rec, spawn=pending.append))
    r = post(client, {"destination": BOT, "events": [{"type": "message"}, {"type": "follow"}]})
    assert r.status_code == 200 and r.json() == {}
    assert rec.order == [] and len(pending) == 2  # scheduled, not run: the answer did not wait
    import asyncio

    for coro in pending:
        asyncio.run(coro)
    assert [d for _e, d in rec.events] == [BOT, BOT]


def test_verification_request_with_no_events_is_ok_and_bad_bodies_fail():
    client = TestClient(line.create_app(SECRET, Recorder(), spawn=lambda c: c.close()))
    assert post(client, {"events": []}).status_code == 200
    assert post(client, None, raw=b"not json").status_code == 400
    assert post(client, None, raw=b" " * (line.MAX_BODY + 1)).status_code == 413
    assert client.get(line.WEBHOOK_PATH).status_code == 405
    assert client.get("/openapi.json").status_code == 404 and client.get("/docs").status_code == 404


async def test_the_default_spawn_runs_the_handler_in_the_background(s, tmp_path):
    import asyncio

    import httpx

    rec = Recorder()
    app = line.create_app(SECRET, rec)
    body = json.dumps({"destination": BOT, "events": [{"type": "message"}]}).encode()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(line.WEBHOOK_PATH, content=body, headers={"X-Line-Signature": sign(body)})
    assert r.status_code == 200
    await asyncio.gather(*list(app.state.tasks))
    assert len(rec.events) == 1


# ------------------------------------------------------------------ event translation


async def test_group_mention_is_answered_with_the_self_mention_removed(fx):
    api, a, port, adapter, clock = fx
    text = "@Salid 😀 total?"
    # index/length count UTF-16 units: "@Salid" is units 0..6
    await adapter.on_event(
        mev(text, mentionees=[{"index": 0, "length": 6, "type": "user", "isSelf": True}]), BOT
    )
    await a.drain()
    q, kw = a.ask_fn.calls[0]
    assert q == "😀 total?" and kw["guild_id"] == ids.tenant_id("line", BOT)
    assert api.replies == [("tok1", ["ok"])] and api.pushes == []


async def test_mention_offsets_after_an_emoji_are_utf16_aware():
    from salid.platforms.line import _strip_self_mentions

    text = "😀😀 @Salid what"  # two emoji = 4 units, a space, then the mention at unit 5
    out, me, others = _strip_self_mentions(
        text, [{"index": 5, "length": 6, "type": "user", "isSelf": True}]
    )
    assert out == "😀😀 what" and me and not others
    _o, me2, others2 = _strip_self_mentions("@Bob hi", [{"index": 0, "length": 4, "type": "user"}])
    assert not me2 and others2


async def test_group_chatter_is_remembered_not_answered_and_1to1_is_answered(fx):
    api, a, port, adapter, clock = fx
    await adapter.on_event(mev("see you at 3", mid="m1"), BOT)
    await a.drain()
    assert not a.ask_fn.calls and not api.replies
    msg = inc(platform="line", tenant=BOT, room="Cgroup")
    assert [m.text for m in stored(a, msg)] == ["see you at 3"]
    await adapter.on_event(mev("hello", src="user", mid="m2", token="tok2"), BOT)
    assert len(a.ask_fn.calls) == 1 and api.loadings == []  # fast answer: no loading animation


async def test_name_in_a_group_calls_salid(fx):
    api, a, port, adapter, clock = fx
    await adapter.on_event(mev("สลิด ช่วยสรุปหน่อย"), BOT)
    assert len(a.ask_fn.calls) == 1


async def test_images_and_files_are_downloaded_only_when_salid_is_addressed(fx):
    api, a, port, adapter, clock = fx
    ev = mev("", src="user")
    ev["message"] = {"type": "image", "id": "img1"}
    await adapter.on_event(ev, BOT)
    (fi,) = a.ask_fn.calls[0][1]["files"]
    assert fi.filename == "image-img1.jpg" and fi.mime == "image/jpeg" and fi.data == b"imgbytes"
    ev2 = mev("", src="user", mid="m9", token="t9")
    ev2["message"] = {"type": "file", "id": "file1", "fileName": "bill.pdf", "fileSize": 7}
    await adapter.on_event(ev2, BOT)
    (fp,) = a.ask_fn.calls[1][1]["files"]
    assert fp.filename == "bill.pdf" and fp.mime == "application/pdf" and fp.data == b"pdfbytes"


async def test_unsupported_content_and_other_events_are_ignored(fx):
    api, a, port, adapter, clock = fx
    sticker = mev("", src="user")
    sticker["message"] = {"type": "sticker", "id": "s1"}
    standby = mev("hi", src="user", mid="m5")
    standby["mode"] = "standby"
    for ev in (sticker, {"type": "follow", "source": {"type": "user", "userId": "Ux"}}, standby):
        await adapter.on_event(ev, BOT)
    await adapter.on_event(mev("hi", src="user", mid="m6"), "Uother-bot")  # not our channel
    await adapter.on_event({"type": "message"}, BOT)  # malformed: never raises
    assert not a.ask_fn.calls and not api.replies


async def test_unsend_removes_the_remembered_message(fx):
    api, a, port, adapter, clock = fx
    await adapter.on_event(mev("oops wrong chat", mid="m3"), BOT)
    await a.drain()
    msg = inc(platform="line", tenant=BOT, room="Cgroup")
    assert len(stored(a, msg)) == 1
    await adapter.on_event(
        {"type": "unsend", "source": {"type": "group", "groupId": "Cgroup", "userId": "Ualice"},
         "unsend": {"messageId": "m3"}}, BOT)  # fmt: skip
    await a.drain()
    assert stored(a, msg) == []


async def test_line_executives_are_only_the_configured_ids(s, tmp_path):
    api = FakeLineApi()
    a = make_assistant(s, tmp_path, line_exec_user_ids="Ualice")
    port = LinePort(api, a.ps)
    adapter = LineAdapter(a, port, BOT)
    await adapter.on_event(mev("hi", src="user", mid="m1"), BOT)
    ev = mev("hi", src="user", mid="m2", token="t2")
    ev["source"]["userId"] = "Ubob"
    await adapter.on_event(ev, BOT)
    assert [c[1]["is_exec"] for c in a.ask_fn.calls] == [True, False]


# ------------------------------------------------------------------ reply token vs push


async def test_reply_token_is_used_first_and_only_once(fx):
    api, a, port, adapter, clock = fx
    msg = inc(platform="line", reply_token="T", raw_at=clock(), room="Cgroup")
    sent = await port.deliver(msg, ["a", "b"], None)
    assert api.replies == [("T", ["a", "b"])] and api.pushes == [] and len(sent) == 2
    await port.deliver(msg, ["c"], None)  # the same token again: push
    assert api.pushes == [("Cgroup", ["c"])]


async def test_slow_answers_fall_back_to_push(fx):
    api, a, port, adapter, clock = fx
    msg = inc(platform="line", reply_token="T", raw_at=clock(), room="Cgroup")
    clock.advance(a.ps.line_reply_token_seconds + 1)
    await port.deliver(msg, ["late"], None)
    assert api.replies == [] and api.pushes == [("Cgroup", ["late"])]


async def test_an_expired_token_error_falls_back_to_push_without_losing_messages(fx):
    api, a, port, adapter, clock = fx
    api.fail_reply = True
    msg = inc(platform="line", reply_token="T", raw_at=clock(), room="Cgroup")
    await port.deliver(msg, ["a", "b", "c"], None)
    assert api.pushes == [("Cgroup", ["a", "b", "c"])]


async def test_at_most_five_messages_per_call(fx):
    api, a, port, adapter, clock = fx
    msg = inc(platform="line", reply_token="T", raw_at=clock(), room="Cgroup")
    await port.deliver(msg, [str(i) for i in range(12)], None)
    assert [len(t) for _tok, t in api.replies] == [5]
    assert [len(t) for _to, t in api.pushes] == [5, 2]
    assert [x for _to, t in api.pushes for x in t] == [str(i) for i in range(5, 12)]


async def test_no_token_means_push_to_the_room(fx):
    api, a, port, adapter, clock = fx
    await port.deliver(inc(platform="line", room="Ualice"), ["x"], None)
    assert api.pushes == [("Ualice", ["x"])]


def test_long_answers_are_split_below_the_5000_unit_limit():
    api = FakeLineApi()
    port = LinePort(api, PlatformSettings(_env_file=None))
    text = "สวัสดีค่ะ 😀 " * 1500
    parts = port.format("**" + text + "**")
    assert len(parts) >= 3 and all(textfmt.utf16_len(p) <= 5000 for p in parts)
    assert "**" not in "".join(parts)  # plain text


async def test_loading_animation_only_in_1to1_chats_and_never_a_text_ack(fx):
    api, a, port, adapter, clock = fx
    assert await port.ack(inc(platform="line", room="Ualice", is_direct=True), "wait") is None
    assert await port.ack(inc(platform="line", room="Cgroup"), "wait") is None
    assert api.loadings == [("Ualice", line.LOADING_SECONDS)]
    assert api.replies == [] and api.pushes == []  # the single-use token is untouched


async def test_loading_shows_while_a_1to1_answer_is_slow(s, tmp_path):
    import threading

    from platform_fakes import FakeAsk, immediately

    release = threading.Event()
    api = FakeLineApi()
    a = make_assistant(s, tmp_path, ask=FakeAsk(release=release), sleep=immediately)
    port = LinePort(api, a.ps)
    adapter = LineAdapter(a, port, BOT)
    import asyncio

    task = asyncio.create_task(adapter.on_event(mev("hello", src="user"), BOT))
    for _ in range(200):
        if api.loadings:
            break
        await asyncio.sleep(0.01)
    release.set()
    await task
    assert api.loadings and api.replies


# ------------------------------------------------------------------ memory scope on LINE


async def test_group_memory_never_leaves_the_group(fx):
    api, a, port, adapter, clock = fx
    await adapter.on_event(mev("สลิด ใครพูดเรื่องงบ", mid="m1"), BOT)
    mem = a.ask_fn.calls[0][1]["memory"]
    group = inc(platform="line", tenant=BOT, room="Cgroup")
    assert list(mem.allowed) == [rid_of(group)]
    assert mem.allowed[rid_of(group)] == "Ops team"
