"""Telegram: update mapping, addressing, polling, files, splitting, edits, secrets, smoke."""

import asyncio
import logging

import httpx
import pytest
from conftest import FixedClockSettings
from platform_fakes import NOW, FakeAsk, immediately, inc, make_assistant, rid_of, stored

from salid.platforms import ids, smoke, telegram, textfmt
from salid.platforms.config import PlatformSettings
from salid.platforms.telegram import (
    Poller,
    TelegramAdapter,
    TelegramApi,
    TelegramError,
    TelegramPort,
    strip_self_mentions,
)

TOKEN = "123456:SECRET-tg-token"
BOT_ID, BOT_NAME = "999", "SalidBot"
TS = int(NOW.timestamp())


@pytest.fixture
def s():
    return FixedClockSettings(exec_user_ids=set())


class FakeClock:
    """A clock whose sleep just moves it forward: pacing and debounce tests run instantly."""

    def __init__(self):
        self.t = 1000.0
        self.slept: list[float] = []

    def now(self):
        return self.t

    async def sleep(self, d):
        self.slept.append(d)
        self.t += d


class FakeApi:
    """Stands in for TelegramApi: records calls, serves scripted updates and files."""

    def __init__(self):
        self.sent: list[dict] = []
        self.actions: list[tuple] = []
        self.webhook_deleted = 0
        self.batches: list = []  # list of update lists, or Exceptions to raise
        self.offsets: list = []
        self.commands: list[list[dict]] = []
        self.me = {"id": int(BOT_ID), "username": BOT_NAME, "can_read_all_group_messages": True}
        self.files = {"ph_big": b"bigjpeg", "ph_small": b"small", "doc1": b"%PDF-bytes"}
        self.fetched: list[str] = []

    async def get_me(self):
        return self.me

    async def delete_webhook(self):
        self.webhook_deleted += 1

    async def set_my_commands(self, commands):
        self.commands.append(commands)

    async def get_updates(self, offset, timeout):
        self.offsets.append(offset)
        nxt = self.batches.pop(0) if self.batches else []
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    async def send_message(self, params):
        self.sent.append(params)
        return {"message_id": 5000 + len(self.sent)}

    async def send_chat_action(self, chat_id, thread):
        self.actions.append((chat_id, thread))

    def fetch_file(self, file_id):
        self.fetched.append(file_id)
        return self.files[file_id]


@pytest.fixture
def fx(s, tmp_path):
    api = FakeApi()
    a = make_assistant(s, tmp_path)
    clock = FakeClock()
    port = TelegramPort(api, clock=clock.now, sleep=clock.sleep)
    adapter = TelegramAdapter(a, port, BOT_ID, BOT_NAME, sleep=clock.sleep)
    return api, a, port, adapter


def upd(
    text="hi",
    *,
    chat="group",
    mid=10,
    uid=111,
    entities=None,
    update_id=1,
    key="message",
    **extra,
):
    ctype = {"private": "private", "group": "group", "super": "supergroup"}[chat]
    chat_obj = {"id": 111 if ctype == "private" else -100500, "type": ctype}
    if ctype != "private":
        chat_obj["title"] = "Ops team"
    m = {
        "message_id": mid,
        "date": TS,
        "from": {"id": uid, "is_bot": False, "first_name": "Alice", "last_name": "A"},
        "chat": chat_obj,
        "text": text,
    }
    if entities is not None:
        m["entities"] = entities
    m.update(extra)
    return {"update_id": update_id, key: m}


def mention(text, who=f"@{BOT_NAME}"):
    off = text.encode("utf-16-le").find(who.encode("utf-16-le")) // 2
    return {"type": "mention", "offset": off, "length": len(who.encode("utf-16-le")) // 2}


# ------------------------------------------------------------------ mapping


async def test_private_chat_maps_to_a_direct_incoming_and_is_answered(fx):
    api, a, port, adapter = fx
    await adapter.on_update(upd("hello there", chat="private"))
    await a.drain()
    q, kw = a.ask_fn.calls[0]
    assert q == "hello there" and kw["guild_id"] == ids.tenant_id("telegram", BOT_ID)
    (reply,) = api.sent
    assert reply["chat_id"] == 111 and reply["text"] == "ok"
    assert "reply_parameters" not in reply and "message_thread_id" not in reply
    assert api.actions[0] == (111, None)  # typing at once in a private chat


async def test_group_and_supergroup_fields(fx):
    api, a, port, adapter = fx
    for chat in ("group", "super"):
        msg = adapter._incoming(
            upd("@SalidBot total?", chat=chat, entities=[mention("@SalidBot total?")])["message"]
        )
        assert msg.platform == "telegram" and msg.tenant == BOT_ID
        assert msg.room == "-100500" and msg.room_name == "Ops team" and not msg.is_direct
        assert msg.user == "111" and msg.user_name == "Alice A" and msg.message == "10"
        assert msg.at == NOW and msg.mentions_bot and msg.text == "total?" and msg.thread is None


async def test_forum_topic_becomes_the_thread_but_plain_reply_chains_do_not(fx):
    api, a, port, adapter = fx
    topic = adapter._incoming(
        upd("x", chat="super", message_thread_id=77, is_topic_message=True)["message"]
    )
    assert topic.thread == "77"
    chain = adapter._incoming(upd("x", chat="super", message_thread_id=77)["message"])
    assert chain.thread is None  # a reply chain in a non-forum supergroup is not a topic


async def test_replies_go_to_the_topic_and_reply_to_the_triggering_message(fx):
    api, a, port, adapter = fx
    text = "@SalidBot hi"
    await adapter.on_update(
        upd(
            text,
            chat="super",
            mid=42,
            entities=[mention(text)],
            message_thread_id=77,
            is_topic_message=True,
        )
    )
    await a.drain()
    (reply,) = api.sent
    assert reply["chat_id"] == -100500 and reply["message_thread_id"] == 77
    assert reply["reply_parameters"] == {"message_id": 42, "allow_sending_without_reply": True}


async def test_unsupported_chats_and_empty_messages_are_ignored(fx):
    api, a, port, adapter = fx
    channel = upd("post")
    channel["message"]["chat"]["type"] = "channel"
    sticker = upd("")
    del sticker["message"]["text"]
    for u in (channel, sticker, {"update_id": 3, "poll": {}}, {"update_id": 4}, {}):
        await adapter.on_update(u)
    assert not a.ask_fn.calls and not api.sent


# ------------------------------------------------------------------ addressing


async def test_group_chatter_is_remembered_not_answered(fx):
    api, a, port, adapter = fx
    await adapter.on_update(upd("see you at 3"))
    await a.drain()
    assert not a.ask_fn.calls and not api.sent
    msg = inc(platform="telegram", tenant=BOT_ID, room="-100500")
    assert [m.text for m in stored(a, msg)] == ["see you at 3"]


async def test_mention_entity_is_answered_with_the_mention_removed_utf16_aware(fx):
    api, a, port, adapter = fx
    text = "😀 @SalidBot total?"  # the emoji is 2 UTF-16 units
    await adapter.on_update(upd(text, entities=[mention(text)]))
    q, _ = a.ask_fn.calls[0]
    assert q == "😀 total?"


async def test_text_mention_of_the_bot_and_mentions_of_others():
    text = "Salid and Bob"
    out, me, others = strip_self_mentions(
        text,
        [
            {"type": "text_mention", "offset": 0, "length": 5, "user": {"id": int(BOT_ID)}},
            {"type": "text_mention", "offset": 10, "length": 3, "user": {"id": 5}},
        ],
        BOT_ID,
        BOT_NAME,
    )
    assert out == "and Bob" and me and others
    _t, me2, others2 = strip_self_mentions(
        "@someone hi", [mention("@someone hi", "@someone")], BOT_ID, BOT_NAME
    )
    assert not me2 and others2
    out3, me3, _o = strip_self_mentions(
        "@salidbot hi", [mention("@salidbot hi", "@salidbot")], BOT_ID, BOT_NAME
    )
    assert me3 and out3 == "hi"  # usernames are case-insensitive


async def test_tagging_someone_else_is_not_for_salid(fx):
    api, a, port, adapter = fx
    text = "@bobby can you look"
    await adapter.on_update(upd(text, entities=[mention(text, "@bobby")]))
    assert not a.ask_fn.calls


async def test_reply_to_a_bot_message_calls_salid_but_a_topic_header_does_not(fx):
    api, a, port, adapter = fx
    bot_msg = {"message_id": 1, "from": {"id": int(BOT_ID), "is_bot": True}}
    await adapter.on_update(upd("and the month?", reply_to_message=bot_msg))
    assert len(a.ask_fn.calls) == 1
    header = {
        "message_id": 2,
        "from": {"id": int(BOT_ID), "is_bot": True},
        "forum_topic_created": {},
    }
    msg = adapter._incoming(upd("hey", mid=11, reply_to_message=header)["message"])
    assert not msg.replies_to_bot and not msg.mentions_bot


async def test_name_in_a_group_calls_salid(fx):
    api, a, port, adapter = fx
    await adapter.on_update(upd("สลิด ช่วยสรุปหน่อย"))
    assert len(a.ask_fn.calls) == 1


async def test_start_command_gets_a_greeting_not_a_question(fx):
    api, a, port, adapter = fx
    await adapter.on_update(upd("/start", chat="private"))
    await a.drain()
    assert not a.ask_fn.calls and len(api.sent) == 1  # the core's greeting


async def test_bots_the_bot_itself_and_channel_posts_are_ignored(fx):
    api, a, port, adapter = fx
    other_bot = upd("@SalidBot hi", entities=[mention("@SalidBot hi")], uid=7)
    other_bot["message"]["from"]["is_bot"] = True
    me = upd("@SalidBot hi", entities=[mention("@SalidBot hi")], uid=int(BOT_ID))
    as_channel = upd("@SalidBot hi", entities=[mention("@SalidBot hi")])
    as_channel["message"]["sender_chat"] = {"id": -1, "type": "channel"}
    for u in (other_bot, me, as_channel):
        await adapter.on_update(u)
    await a.drain()
    assert not a.ask_fn.calls and not api.sent
    assert stored(a, inc(platform="telegram", tenant=BOT_ID, room="-100500")) == []


async def test_executives_are_only_the_configured_ids_not_group_admins(s, tmp_path):
    api = FakeApi()
    a = make_assistant(s, tmp_path, telegram_exec_user_ids="111, 222")
    adapter = TelegramAdapter(a, TelegramPort(api), BOT_ID, BOT_NAME)
    await adapter.on_update(upd("hi", chat="private", uid=111, mid=1))
    await adapter.on_update(upd("hi", chat="private", uid=333, mid=2))
    assert [c[1]["is_exec"] for c in a.ask_fn.calls] == [True, False]
    assert a.ps.telegram_exec_user_ids == {"111", "222"}
    assert await TelegramPort(api).is_admin(inc(platform="telegram")) is False


# ------------------------------------------------------------------ memory scope


async def test_group_memory_never_leaves_the_group_and_private_sees_only_itself(fx):
    api, a, port, adapter = fx
    text = "@SalidBot who talked about the budget"
    await adapter.on_update(upd(text, entities=[mention(text)]))
    mem = a.ask_fn.calls[0][1]["memory"]
    group = inc(platform="telegram", tenant=BOT_ID, room="-100500")
    assert list(mem.allowed) == [rid_of(group)] and mem.allowed[rid_of(group)] == "Ops team"
    await adapter.on_update(upd("hello", chat="private", mid=11))
    mem2 = a.ask_fn.calls[1][1]["memory"]
    assert list(mem2.allowed) == [rid_of(inc(platform="telegram", tenant=BOT_ID, room="111"))]


# ------------------------------------------------------------------ edits


async def test_edited_message_updates_memory_and_does_not_trigger_an_answer(fx):
    api, a, port, adapter = fx
    msg = inc(platform="telegram", tenant=BOT_ID, room="-100500")
    await adapter.on_update(upd("see you at 3", mid=10))
    await a.drain()
    await adapter.on_update(upd("see you at 4", mid=10, key="edited_message", update_id=2))
    await a.drain()
    assert [m.text for m in stored(a, msg)] == ["see you at 4"]
    assert not a.ask_fn.calls and not api.sent


# ------------------------------------------------------------------ files


def photo_msg():
    u = upd(
        "",
        chat="private",
        photo=[
            {"file_id": "ph_small", "file_size": 100, "width": 90, "height": 90},
            {"file_id": "ph_big", "file_size": 5000, "width": 1280, "height": 960},
        ],
        caption="what is this",
    )
    del u["message"]["text"]  # a photo message carries a caption, not text
    return u


async def test_photo_uses_the_largest_size_and_downloads_lazily(fx):
    api, a, port, adapter = fx
    u = photo_msg()
    msg = adapter._incoming(u["message"])
    assert msg.text == "what is this"  # the caption is the text
    (f,) = msg.files
    assert f.name == "photo-10.jpg" and f.mime == "image/jpeg" and f.size == 5000
    assert api.fetched == []  # nothing downloaded for mere mapping
    await adapter.on_update(u)
    await a.drain()
    (fi,) = a.ask_fn.calls[0][1]["files"]
    assert fi.data == b"bigjpeg" and api.fetched == ["ph_big"]


async def test_files_in_unaddressed_group_messages_are_never_downloaded(fx):
    api, a, port, adapter = fx
    u = upd(
        "here is the bill",
        document={
            "file_id": "doc1",
            "file_name": "b.pdf",
            "mime_type": "application/pdf",
            "file_size": 10,
        },
    )
    await adapter.on_update(u)
    assert api.fetched == [] and not a.ask_fn.calls


async def test_document_is_passed_with_its_name_and_mime(fx):
    api, a, port, adapter = fx
    u = upd(
        "",
        chat="private",
        document={
            "file_id": "doc1",
            "file_name": "bill.pdf",
            "mime_type": "application/pdf",
            "file_size": 10,
        },
    )
    del u["message"]["text"]
    await adapter.on_update(u)
    (fi,) = a.ask_fn.calls[0][1]["files"]
    assert fi.filename == "bill.pdf" and fi.mime == "application/pdf" and fi.data == b"%PDF-bytes"


async def test_oversized_declared_files_are_skipped_without_download(fx):
    api, a, port, adapter = fx
    big = telegram.MAX_FILE_BYTES + 1
    u = upd(
        "",
        chat="private",
        document={
            "file_id": "doc1",
            "file_name": "huge.pdf",
            "mime_type": "application/pdf",
            "file_size": big,
        },
    )
    del u["message"]["text"]
    await adapter.on_update(u)
    (fi,) = a.ask_fn.calls[0][1]["files"]
    assert fi.skip and not fi.data and api.fetched == []


def tg_transport(routes, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        for suffix, fn in routes.items():
            if request.url.path.endswith(suffix):
                return fn(request)
        return httpx.Response(404, json={"ok": False, "error_code": 404})

    return httpx.MockTransport(handler)


def file_routes(size, body, path="photos/file_1.jpg"):
    return {
        "/getFile": lambda r: httpx.Response(
            200, json={"ok": True, "result": {"file_id": "x", "file_size": size, "file_path": path}}
        ),
        f"/{path}": lambda r: httpx.Response(200, content=body),
    }


def test_download_uses_the_file_endpoint_and_returns_bytes():
    seen = []
    api = TelegramApi(TOKEN, transport=tg_transport(file_routes(5, b"abcde"), seen))
    assert api.fetch_file("x") == b"abcde"
    assert seen == [
        ("POST", f"/bot{TOKEN}/getFile"),
        ("GET", f"/file/bot{TOKEN}/photos/file_1.jpg"),
    ]


def test_download_refuses_over_the_limit_declared_or_streamed():
    seen = []
    declared = TelegramApi(
        TOKEN, transport=tg_transport(file_routes(telegram.DOWNLOAD_MAX + 1, b"x"), seen)
    )
    with pytest.raises(ValueError):
        declared.fetch_file("x")
    assert all("/file/" not in p for _m, p in seen)  # never even started downloading
    liar = TelegramApi(
        TOKEN, transport=tg_transport(file_routes(1, b"z" * (telegram.MAX_FILE_BYTES + 1)), [])
    )
    with pytest.raises(ValueError):
        liar.fetch_file("x")


def test_download_failures_are_telegram_errors_without_url_or_token():
    api = TelegramApi(TOKEN, transport=tg_transport({}, []))
    with pytest.raises(TelegramError) as exc:
        api.fetch_file("x")
    assert TOKEN not in str(exc.value) and TOKEN not in repr(exc.value)

    def boom(_r):
        raise httpx.ConnectError(f"cannot reach https://api.telegram.org/bot{TOKEN}/getFile")

    api2 = TelegramApi(TOKEN, transport=httpx.MockTransport(boom))
    with pytest.raises(TelegramError) as exc2:
        api2.fetch_file("x")
    assert TOKEN not in str(exc2.value) and exc2.value.__cause__ is None


# ------------------------------------------------------------------ replies


def test_long_answers_are_split_at_4096_and_plain():
    port = TelegramPort(FakeApi())
    parts = port.format("**" + "สวัสดีค่ะ 😀 " * 1500 + "**")
    assert len(parts) >= 3 and all(textfmt.utf16_len(p) <= 4096 for p in parts)
    assert "**" not in "".join(parts)
    assert port.format("## Title\n- one\n- **two**") == ["Title\n• one\n• two"]


async def test_every_chunk_is_sent_and_only_the_first_replies_in_groups(fx):
    api, a, port, adapter = fx
    msg = inc(platform="telegram", tenant=BOT_ID, room="-100500", message="42")
    sent = await port.deliver(msg, ["a", "b", "c"], None)
    assert sent == ["5001", "5002", "5003"]
    assert ["reply_parameters" in p for p in api.sent] == [True, False, False]


async def test_typing_keeps_going_until_the_answer_is_delivered(s, tmp_path, monkeypatch):
    import threading

    monkeypatch.setattr(telegram, "TYPING_EVERY", 0.01)
    release = threading.Event()
    api = FakeApi()
    a = make_assistant(s, tmp_path, ask=FakeAsk(release=release), sleep=immediately)
    adapter = TelegramAdapter(a, TelegramPort(api), BOT_ID, BOT_NAME)
    task = asyncio.create_task(adapter.on_update(upd("hello", chat="private")))
    for _ in range(300):
        if len(api.actions) >= 3:
            break
        await asyncio.sleep(0.01)
    release.set()
    await task
    await a.drain()
    n = len(api.actions)
    assert n >= 3 and len(api.sent) == 1
    await asyncio.sleep(0.05)
    assert len(api.actions) == n  # stopped once the reply went out


# ------------------------------------------------------------------ polling


async def test_polling_deletes_the_webhook_once_and_advances_the_offset(fx):
    api, a, port, adapter = fx
    api.batches = [[upd("one", update_id=7, mid=1), upd("two", update_id=9, mid=2)], [], []]
    p = Poller(api, adapter)
    assert await p.poll_once() == 2
    assert await p.poll_once() == 0
    await p.poll_once()
    assert api.offsets == [None, 10, 10] and api.webhook_deleted == 1
    await a.drain()
    msg = inc(platform="telegram", tenant=BOT_ID, room="-100500")
    assert [m.text for m in stored(a, msg)] == ["one", "two"]


async def test_a_failed_poll_backs_off_and_loses_no_updates(fx):
    api, a, port, adapter = fx
    api.batches = [
        [upd("first", update_id=4, mid=1)],
        TelegramError("getUpdates", 502),
        TelegramError("getUpdates", 502),
        [upd("second", update_id=5, mid=2)],
    ]
    delays = []
    p = Poller(api, adapter)

    async def sleep(d):
        delays.append(d)

    p._sleep = sleep
    calls = 0
    original = api.get_updates

    async def counting(offset, timeout):
        nonlocal calls
        calls += 1
        if calls == 5:
            p.stop()
        return await original(offset, timeout)

    api.get_updates = counting
    await p.run()
    await a.drain()
    assert delays == [1.0, 2.0]  # exponential backoff
    assert api.offsets == [None, 5, 5, 5, 6]  # the failed rounds re-asked from the same offset
    msg = inc(platform="telegram", tenant=BOT_ID, room="-100500")
    assert [m.text for m in stored(a, msg)] == ["first", "second"]
    assert p.backoff == 1.0  # reset after the success


async def test_a_failed_delete_webhook_is_retried_before_polling(fx):
    api, a, port, adapter = fx
    fails = [TelegramError("deleteWebhook", 500)]
    real = api.delete_webhook

    async def flaky():
        if fails:
            raise fails.pop()
        await real()

    api.delete_webhook = flaky
    p = Poller(api, adapter)
    with pytest.raises(TelegramError):
        await p.poll_once()
    assert api.offsets == []
    await p.poll_once()
    assert api.webhook_deleted == 1 and api.offsets == [None]


# ------------------------------------------------------------------ Bot API client, secrets


def ok(result):
    return httpx.Response(200, json={"ok": True, "result": result})


async def test_get_updates_sends_offset_timeout_and_allowed_updates():
    bodies = []

    def handler(request):
        import json

        bodies.append((request.url.path, json.loads(request.content)))
        return ok([{"update_id": 1}])

    api = TelegramApi(TOKEN, transport=httpx.MockTransport(handler))
    assert await api.get_updates(41, 30) == [{"update_id": 1}]
    assert bodies == [
        (
            f"/bot{TOKEN}/getUpdates",
            {
                "timeout": 30,
                "allowed_updates": ["message", "edited_message", "my_chat_member"],
                "offset": 41,
            },
        )
    ]
    await api.aclose()


async def test_rate_limit_is_retried_once_after_retry_after():
    hits, slept = [], []

    def handler(request):
        hits.append(1)
        if len(hits) == 1:
            return httpx.Response(
                429, json={"ok": False, "error_code": 429, "parameters": {"retry_after": 3}}
            )
        return ok({"message_id": 1})

    async def sleep(d):
        slept.append(d)

    api = TelegramApi(TOKEN, transport=httpx.MockTransport(handler), sleep=sleep)
    assert await api.send_message({"chat_id": 1, "text": "x"}) == {"message_id": 1}
    assert slept == [3.0] and len(hits) == 2
    api2 = TelegramApi(
        TOKEN,
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                429, json={"ok": False, "error_code": 429, "parameters": {"retry_after": 999}}
            )
        ),
        sleep=sleep,
    )
    with pytest.raises(TelegramError) as e:
        await api2.send_message({})
    assert e.value.code == 429  # too long to wait inline: surfaced


async def test_the_token_never_reaches_the_logs_even_on_http_errors(caplog, fx):
    caplog.set_level(logging.DEBUG)
    api, a, port, adapter = fx

    def handler(request):
        if request.url.path.endswith("/getUpdates"):
            return httpx.Response(
                401, json={"ok": False, "error_code": 401, "description": f"bad {TOKEN}"}
            )
        raise httpx.ReadTimeout(f"timed out calling {request.url}")

    real = TelegramApi(TOKEN, transport=httpx.MockTransport(handler))
    p = Poller(real, adapter)
    p._sleep = lambda d: asyncio.sleep(0)
    p.api.delete_webhook = lambda: asyncio.sleep(0)  # type: ignore[method-assign]
    calls = 0
    original = real.get_updates

    async def twice(offset, timeout):
        nonlocal calls
        calls += 1
        if calls == 2:
            p.stop()
        return await original(offset, timeout)

    real.get_updates = twice  # type: ignore[method-assign]
    await p.run()
    with pytest.raises(TelegramError):
        await real.send_message({"chat_id": 1, "text": "x"})  # timeout path
    assert caplog.records  # the failed poll was logged ...
    assert (
        TOKEN not in caplog.text and "SECRET-tg-token" not in caplog.text
    )  # ... without the token
    assert "api.telegram.org" not in caplog.text  # and without any URL
    for logger_name in ("httpx", "httpcore"):
        assert logging.getLogger(logger_name).level >= logging.WARNING
    await real.aclose()


async def test_handler_failures_log_types_only(caplog, s, tmp_path):
    caplog.set_level(logging.DEBUG)

    class Exploding(FakeApi):
        async def send_message(self, params):
            raise RuntimeError(f"boom-detail {TOKEN} my iban is TH99-SECRET-4242")

    a = make_assistant(s, tmp_path)
    adapter = TelegramAdapter(a, TelegramPort(Exploding()), BOT_ID, BOT_NAME)
    await adapter.on_update(upd("my iban is TH99-SECRET-4242", chat="private"))
    await a.drain()
    assert caplog.records
    for needle in (TOKEN, "TH99-SECRET-4242", "boom-detail"):
        assert needle not in caplog.text


def test_settings_never_print_the_token():
    ps = PlatformSettings(_env_file=None, telegram_bot_token=TOKEN, telegram_exec_user_ids="1,2")
    assert TOKEN not in repr(ps) + str(ps) + repr(ps.model_fields_set)
    assert ps.telegram_enabled and ps.enabled_platforms() == ["telegram"]
    assert ps.telegram_exec_user_ids == {"1", "2"}
    assert not PlatformSettings(_env_file=None).telegram_enabled


def test_telegram_tenants_are_their_own_range():
    t = ids.tenant_id("telegram", BOT_ID)
    assert ids.platform_of(t) == "telegram" and ids.is_platform_tenant(t)
    assert t != ids.tenant_id("line", BOT_ID) and t != ids.tenant_id("slack", BOT_ID)
    assert t < 2**63  # fits a signed int64


# ------------------------------------------------------------------ smoke


async def test_smoke_check_prints_username_and_id_only(monkeypatch):
    ps = PlatformSettings(_env_file=None, telegram_bot_token=TOKEN)
    assert smoke.plan(ps)[2] == "telegram: will start"
    assert "missing SALID_TELEGRAM_BOT_TOKEN" in smoke.plan(PlatformSettings(_env_file=None))[2]

    async def good(_ps):
        return {"username": BOT_NAME, "id": int(BOT_ID)}

    async def bad(_ps):
        raise TelegramError("getMe", 401)

    assert await smoke.check(ps, telegram_check=good) == [
        f"telegram: token ok (username={BOT_NAME}, id={BOT_ID})"
    ]
    (line,) = await smoke.check(ps, telegram_check=bad)
    assert line.startswith("telegram: token check FAILED") and TOKEN not in line


async def test_check_uses_getme_and_returns_no_secrets(monkeypatch):
    real = telegram.TelegramApi
    monkeypatch.setattr(
        telegram,
        "TelegramApi",
        lambda token: real(
            token,
            transport=httpx.MockTransport(
                lambda r: ok({"id": 9, "username": "SalidBot", "first_name": "x"})
            ),
        ),
    )
    out = await telegram.check(PlatformSettings(_env_file=None, telegram_bot_token=TOKEN))
    assert out == {"username": "SalidBot", "id": 9}


# ------------------------------------------------------------------ commands (/cmd@bot, /help)


def command(text, who):
    off = text.index(who)
    return {"type": "bot_command", "offset": off, "length": len(who)}


def test_command_for_this_bot_addresses_it_and_loses_the_suffix():
    text = "/total@SalidBot this week"
    out, me, others = strip_self_mentions(
        text, [command(text, "/total@SalidBot")], BOT_ID, BOT_NAME
    )
    assert (out, me, others) == ("/total this week", True, False)
    text = "/total@salidbot"  # usernames are case-insensitive
    out, me, _o = strip_self_mentions(text, [command(text, text)], BOT_ID, BOT_NAME)
    assert (out, me) == ("/total", True)
    plain = strip_self_mentions("/total", [command("/total", "/total")], BOT_ID, BOT_NAME)
    assert plain == ("/total", False, False)  # no suffix: not addressed to us by name
    text = "/total@OtherBot"
    other = strip_self_mentions(text, [command(text, text)], BOT_ID, BOT_NAME)
    assert other == (text, False, True)  # another bot's command is not ours


async def test_command_with_our_username_is_answered_in_a_group_but_not_another_bots(fx):
    api, a, port, adapter = fx
    text = "/total@SalidBot please"
    await adapter.on_update(upd(text, entities=[command(text, "/total@SalidBot")]))
    assert a.ask_fn.calls[0][0] == "/total please"
    other = "/total@OtherBot please"
    await adapter.on_update(upd(other, mid=11, entities=[command(other, "/total@OtherBot")]))
    assert len(a.ask_fn.calls) == 1


async def test_start_at_bot_in_a_group_gets_the_greeting_not_a_question(fx):
    api, a, port, adapter = fx
    for mid, text in ((10, "/start@SalidBot"), (11, "/start@SalidBot ref123")):
        await adapter.on_update(upd(text, mid=mid, entities=[command(text, "/start@SalidBot")]))
    await a.drain()
    assert not a.ask_fn.calls and len(api.sent) == 2  # the core's greeting, twice


async def test_help_is_answered_without_the_model_in_private_and_when_addressed(fx):
    api, a, port, adapter = fx
    await adapter.on_update(upd("/help", chat="private"))
    text = "/help@SalidBot"
    await adapter.on_update(upd(text, mid=11, entities=[command(text, text)]))
    await adapter.on_update(upd("/help", mid=12))  # a bare /help in a group may be for any bot
    await a.drain()
    assert not a.ask_fn.calls and len(api.sent) == 2
    private, group = api.sent
    assert private["text"] == telegram.HELP_TEXT and "reply_parameters" not in private
    assert group["text"] == telegram.HELP_TEXT and group["reply_parameters"]["message_id"] == 11
    assert "Salid" in telegram.HELP_TEXT and "สลิด" in telegram.HELP_TEXT  # Thai and English


# ------------------------------------------------------------------ startup


async def test_startup_registers_the_commands_and_warns_when_privacy_mode_is_on(fx, caplog):
    api, a, port, adapter = fx
    api.me = {"id": int(BOT_ID), "username": BOT_NAME, "can_read_all_group_messages": False}
    p = Poller(api, adapter)
    p.stop()
    caplog.set_level(logging.WARNING)
    await p.run()
    (cmds,) = api.commands
    assert [c["command"] for c in cmds] == ["start", "help"]
    assert all(1 <= len(c["description"]) <= 256 and "/" in c["description"] for c in cmds)
    assert "privacy mode is ON" in caplog.text and "RE-ADD" in caplog.text


async def test_startup_is_quiet_when_privacy_is_off_and_never_raises(fx, caplog):
    api, a, port, adapter = fx
    caplog.set_level(logging.WARNING)
    await Poller(api, adapter).startup()
    assert "privacy" not in caplog.text
    off_and_closed = {"can_read_all_group_messages": False, "can_join_groups": False}
    assert telegram.privacy_warning(off_and_closed) is None
    assert telegram.privacy_warning({"id": 1}) is None  # unknown: say nothing

    async def boom(*_a):
        raise TelegramError("getMe", 500)

    api.get_me, api.set_my_commands = boom, boom
    await Poller(api, adapter).startup()  # logged, not raised
    assert "getMe check failed" in caplog.text and "setMyCommands failed" in caplog.text


async def test_set_my_commands_posts_the_command_list():
    seen = []

    def handler(request):
        import json

        seen.append((request.url.path, json.loads(request.content)))
        return ok(True)

    api = TelegramApi(TOKEN, transport=httpx.MockTransport(handler))
    await api.set_my_commands(telegram.COMMANDS)
    assert seen == [(f"/bot{TOKEN}/setMyCommands", {"commands": telegram.COMMANDS})]
    await api.aclose()


# ------------------------------------------------------------------ albums


def album_item(mid, file_id, caption=None, entities=None, group="g1", chat="group"):
    photo = [{"file_id": file_id, "file_size": 50}]
    u = upd("", chat=chat, mid=mid, media_group_id=group, photo=photo)
    del u["message"]["text"]
    if caption:
        u["message"]["caption"] = caption
    if entities:
        u["message"]["caption_entities"] = entities
    return u


@pytest.fixture
def album_fx(s, tmp_path):
    """An adapter whose debounce sleep yields once, so concurrently started updates can join."""
    api = FakeApi()
    api.files.update(p1=b"one", p2=b"two", p3=b"three")
    a = make_assistant(s, tmp_path)
    slept = []

    async def sleep(d):
        slept.append(d)
        await asyncio.sleep(0)

    clock = FakeClock()
    port = TelegramPort(api, clock=clock.now, sleep=clock.sleep)
    return api, a, TelegramAdapter(a, port, BOT_ID, BOT_NAME, sleep=sleep), slept


async def test_an_album_in_a_private_chat_is_one_question_with_all_files(album_fx):
    api, a, adapter, slept = album_fx
    await asyncio.gather(
        adapter.on_update(album_item(10, "p1", chat="private")),
        adapter.on_update(album_item(11, "p2", caption="three receipts", chat="private")),
        adapter.on_update(album_item(12, "p3", chat="private")),
    )
    await a.drain()
    (call,) = a.ask_fn.calls  # one answer, not three
    assert call[0] == "three receipts" and len(api.sent) == 1
    assert [f.data for f in call[1]["files"]] == [b"one", b"two", b"three"]
    assert slept and set(slept) == {telegram.ALBUM_DEBOUNCE_S}


async def test_a_group_album_is_addressed_if_any_caption_mentions_the_bot(album_fx):
    api, a, adapter, _ = album_fx
    cap = "@SalidBot record these"
    await asyncio.gather(
        adapter.on_update(album_item(10, "p1")),
        adapter.on_update(album_item(11, "p2", caption=cap, entities=[mention(cap)])),
    )
    await a.drain()
    (call,) = a.ask_fn.calls
    assert call[0] == "record these" and len(call[1]["files"]) == 2
    assert api.sent[0]["reply_parameters"]["message_id"] == 11  # replies to the captioned photo


async def test_an_unaddressed_group_album_is_only_remembered_and_downloads_nothing(album_fx):
    api, a, adapter, _ = album_fx
    await asyncio.gather(
        adapter.on_update(album_item(10, "p1", caption="lunch bills")),
        adapter.on_update(album_item(11, "p2")),
    )
    await a.drain()
    assert not a.ask_fn.calls and not api.sent and api.fetched == []


async def test_separate_albums_and_single_photos_do_not_merge(album_fx):
    api, a, adapter, _ = album_fx
    await asyncio.gather(
        adapter.on_update(album_item(20, "p1", group="A", chat="private")),
        adapter.on_update(album_item(21, "p2", group="B", chat="private")),
        adapter.on_update(photo_msg()),
    )
    await a.drain()
    assert len(a.ask_fn.calls) == 3


# ------------------------------------------------------------------ membership (my_chat_member)


def member_update(status, old="left", *, chat="group", cid=-100500, user=None):
    return {
        "update_id": 1,
        "my_chat_member": {
            "chat": {"id": cid, "type": chat, "title": "Ops team"},
            "from": {"id": 5, "is_bot": False, "first_name": "Admin"},
            "date": TS,
            "old_chat_member": {"status": old, "user": {"id": int(BOT_ID), "is_bot": True}},
            "new_chat_member": {
                "status": status,
                "user": {"id": int(user or BOT_ID), "is_bot": True},
            },
        },
    }


async def test_being_added_to_a_group_posts_the_consent_notice_once(fx):
    api, a, port, adapter = fx
    await adapter.on_update(member_update("member"))
    (note,) = api.sent
    assert note["chat_id"] == -100500 and note["text"] == telegram.WELCOME_TEXT
    text = telegram.WELCOME_TEXT
    assert "ความจำของทีม" in text and "team memory" in text  # Thai and English
    assert "Google Vertex AI" in text and "removes Salid" in text and "/help" in text
    await adapter.on_update(member_update("administrator", old="member"))  # promoted: no repeat
    assert len(api.sent) == 1


async def test_removal_or_block_stops_sending_and_a_new_message_clears_it(fx):
    api, a, port, adapter = fx
    await adapter.on_update(member_update("kicked", old="member"))
    assert -100500 in port.gone
    msg = inc(platform="telegram", tenant=BOT_ID, room="-100500", message="42")
    assert await port.deliver(msg, ["hello"], None) == [] and api.sent == []
    await port.typing(msg)
    assert api.actions == []
    await port.notice(-100500, "x")
    assert api.sent == []
    await adapter.on_update(member_update("kicked", old="member", chat="private", cid=111))
    assert 111 in port.gone  # a user blocked the bot
    await adapter.on_update(member_update("member", old="kicked", chat="private", cid=111))
    assert 111 not in port.gone and api.sent == []  # unblocked: no welcome in a private chat
    await adapter.on_update(upd("@SalidBot hi", entities=[mention("@SalidBot hi")]))
    assert -100500 not in port.gone  # it talked to us, so we are in there
    await a.drain()
    assert len(api.sent) == 1


async def test_a_403_marks_the_chat_gone_instead_of_raising_and_other_users_are_ignored(fx):
    api, a, port, adapter = fx

    async def forbidden(params):
        raise TelegramError("sendMessage", 403)

    api.send_message = forbidden
    msg = inc(platform="telegram", tenant=BOT_ID, room="-100500", message="42")
    assert await port.deliver(msg, ["a", "b"], None) == []
    assert -100500 in port.gone
    await adapter.on_update(member_update("kicked", user=777, cid=-1))  # someone else's status
    assert -1 not in port.gone
    port.mark_alive(-100500)
    muted = member_update("restricted", old="member")
    muted["my_chat_member"]["new_chat_member"]["can_send_messages"] = False
    await adapter.on_update(muted)
    assert -100500 in port.gone


# ------------------------------------------------------------------ send pacing, 429


async def test_messages_to_one_chat_are_spaced_a_second_apart(fx):
    api, a, port, adapter = fx
    clock = FakeClock()
    port = TelegramPort(api, clock=clock.now, sleep=clock.sleep)
    msg = inc(platform="telegram", tenant=BOT_ID, room="111", message="1", is_direct=True)
    other = inc(platform="telegram", tenant=BOT_ID, room="222", message="1", is_direct=True)
    await port.deliver(msg, ["a", "b", "c"], None)
    assert clock.slept == [pytest.approx(1.0)] * 2
    clock.slept.clear()
    await port.deliver(other, ["x"], None)  # another chat is not held back
    assert clock.slept == []


async def test_a_group_gets_at_most_twenty_messages_a_minute(fx):
    api, a, port, adapter = fx
    clock = FakeClock()
    sent_at: list[float] = []

    async def record(params):
        sent_at.append(clock.now())
        return {"message_id": len(sent_at)}

    api.send_message = record
    port = TelegramPort(api, clock=clock.now, sleep=clock.sleep)
    msg = inc(platform="telegram", tenant=BOT_ID, room="-100500", message="1")
    await port.deliver(msg, [str(i) for i in range(45)], None)
    assert len(sent_at) == 45  # none dropped
    assert all(b - a_ >= 1.0 - 1e-9 for a_, b in zip(sent_at, sent_at[1:], strict=False))
    for t0 in sent_at:
        assert sum(1 for x in sent_at if t0 <= x < t0 + 60) <= 20
    assert sent_at[20] - sent_at[0] >= 60  # the 21st waited for the minute to roll over


async def test_a_429_is_waited_out_up_to_two_minutes_and_later_chunks_still_go():
    slept, bodies = [], []

    def handler(request):
        import json

        body = json.loads(request.content)
        bodies.append(body["text"])
        # chunk "b" is rate limited twice (90 s each), then accepted
        if body["text"] == "b" and bodies.count("b") <= 2:
            return httpx.Response(
                429, json={"ok": False, "error_code": 429, "parameters": {"retry_after": 90}}
            )
        return ok({"message_id": len(bodies)})

    async def sleep(d):
        slept.append(d)

    api = TelegramApi(TOKEN, transport=httpx.MockTransport(handler), sleep=sleep)
    port = TelegramPort(api, sleep=sleep)
    msg = inc(platform="telegram", tenant=BOT_ID, room="111", message="1", is_direct=True)
    sent = await port.deliver(msg, ["a", "b", "c"], None)
    assert len(sent) == 3 and bodies == ["a", "b", "b", "b", "c"]
    assert slept.count(90.0) == 2
    await api.aclose()


async def test_a_long_429_is_surfaced_loudly_not_dropped_silently(caplog):
    def handler(request):
        return httpx.Response(
            429, json={"ok": False, "error_code": 429, "parameters": {"retry_after": 600}}
        )

    async def sleep(d):
        pass

    api = TelegramApi(TOKEN, transport=httpx.MockTransport(handler), sleep=sleep)
    port = TelegramPort(api, sleep=sleep)
    msg = inc(platform="telegram", tenant=BOT_ID, room="111", message="1", is_direct=True)
    caplog.set_level(logging.ERROR)
    with pytest.raises(TelegramError) as e:
        await port.deliver(msg, ["a", "b"], None)
    assert e.value.code == 429 and e.value.retry_after == 600
    assert "0 of 2 messages sent" in caplog.text
    await api.aclose()


# ------------------------------------------------------------------ private-chat topics


async def test_private_chat_topics_are_answered_inside_their_thread(fx):
    api, a, port, adapter = fx
    msg = adapter._incoming(
        upd("hi", chat="private", message_thread_id=5, is_topic_message=True)["message"]
    )
    assert msg.thread == "5" and msg.is_direct
    await adapter.on_update(
        upd("hello", chat="private", message_thread_id=5, is_topic_message=True)
    )
    await a.drain()
    (reply,) = api.sent
    assert reply["message_thread_id"] == 5 and "reply_parameters" not in reply
    assert api.actions[0] == (111, 5)  # typing goes to the topic too
    plain = adapter._incoming(upd("hi", chat="private")["message"])
    assert plain.thread is None


# ------------------------------------------------------------------ offset, 409


async def test_after_a_long_idle_the_offset_is_reset_loudly(fx, caplog):
    api, a, port, adapter = fx
    now = [1_000_000.0]
    p = Poller(api, adapter, clock=lambda: now[0])
    api.batches = [[upd("one", update_id=7, mid=1)], [], []]
    await p.poll_once()
    assert p.offset == 8
    now[0] += telegram.IDLE_RESET_S - 60  # quiet, but not yet a week
    await p.poll_once()
    assert api.offsets[-1] == 8
    now[0] += 120
    caplog.set_level(logging.WARNING)
    await p.poll_once()
    assert api.offsets[-1] is None and "resetting the offset" in caplog.text
    await a.drain()


async def test_a_409_conflict_is_logged_clearly_and_retried_with_a_floor_backoff(fx, caplog):
    api, a, port, adapter = fx
    api.batches = [TelegramError("getUpdates", 409), TelegramError("getUpdates", 409), []]
    delays = []
    p = Poller(api, adapter)

    async def sleep(d):
        delays.append(d)

    p._sleep = sleep
    calls = 0
    original = api.get_updates

    async def counting(offset, timeout):
        nonlocal calls
        calls += 1
        if calls == 3:
            p.stop()
        return await original(offset, timeout)

    api.get_updates = counting
    caplog.set_level(logging.WARNING)
    await p.run()
    assert delays == [telegram.CONFLICT_BACKOFF_MIN, 5.0]  # never hammers: 5 s floor
    assert caplog.text.count("409 Conflict") == 2 and "ONE instance per token" in caplog.text
    assert api.webhook_deleted == 3  # a webhook may be the cause: cleared again each round
