"""WhatsApp: webhook verification and signature, payload mapping, replies, media, secrets, smoke."""

import asyncio
import hashlib
import hmac
import json
import logging

import httpx
import pytest
from conftest import FixedClockSettings
from fastapi.testclient import TestClient
from platform_fakes import NOW, FakeAsk, make_assistant, rid_of, stored

from salid.platforms import ids, smoke, textfmt, whatsapp
from salid.platforms.config import PlatformSettings
from salid.platforms.whatsapp import (
    WhatsAppAdapter,
    WhatsAppApi,
    WhatsAppError,
    WhatsAppPort,
    verify_signature,
)

SECRET, VERIFY, TOKEN = "app-secret-xyz", "verify-token-abc", "EAAG-SECRET-access-token"
PHONE_ID = "1055550001"
ALICE, BOB = "66811112222", "66833334444"  # wa_ids are phone numbers: PII
TS = str(int(NOW.timestamp()))


@pytest.fixture
def s():
    return FixedClockSettings(exec_user_ids=set())


def sign(body: bytes, secret=SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class FakeApi:
    def __init__(self):
        self.sent: list[dict] = []
        self.reads: list[str] = []
        self.media = {"img1": b"jpegbytes", "doc1": b"%PDF-bytes"}
        self.fetched: list[str] = []

    async def send_text(self, to, body, reply_to):
        self.sent.append({"to": to, "body": body, "reply_to": reply_to})
        return f"wamid.OUT{len(self.sent)}"

    async def mark_read(self, message_id):
        self.reads.append(message_id)

    def fetch_media(self, media_id):
        self.fetched.append(media_id)
        return self.media[media_id]


@pytest.fixture
def fx(s, tmp_path):
    api = FakeApi()
    a = make_assistant(s, tmp_path)
    port = WhatsAppPort(api)
    return api, a, port, WhatsAppAdapter(a, port, PHONE_ID)


def msg(kind="text", mid="wamid.1", sender=ALICE, **body):
    m = {"id": mid, "from": sender, "timestamp": TS, "type": kind}
    m.update(body)
    return m


def payload(*messages, phone=PHONE_ID, statuses=None, name="Alice"):
    value = {
        "messaging_product": "whatsapp",
        "metadata": {"display_phone_number": "15550001111", "phone_number_id": phone},
        "contacts": [{"profile": {"name": name}, "wa_id": ALICE}],
        "messages": list(messages),
    }
    if statuses:
        value["statuses"] = statuses
    return {
        "object": "whatsapp_business_account",
        "entry": [{"id": "waba", "changes": [{"field": "messages", "value": value}]}],
    }


def text(body="hello there", **kw):
    return msg("text", text={"body": body}, **kw)


# ------------------------------------------------------------------ webhook: verification


class Recorder:
    def __init__(self):
        self.payloads = []

    async def on_payload(self, p):
        self.payloads.append(p)


def app_client(rec=None, pending=None):
    rec = rec or Recorder()
    pending = pending if pending is not None else []
    app = whatsapp.create_app(SECRET, VERIFY, rec, spawn=pending.append)
    return TestClient(app), rec, pending


def test_get_verification_echoes_the_challenge_only_for_the_right_token():
    client, _r, _p = app_client()
    q = {"hub.mode": "subscribe", "hub.verify_token": VERIFY, "hub.challenge": "1158201444"}
    r = client.get(whatsapp.WEBHOOK_PATH, params=q)
    assert r.status_code == 200 and r.text == "1158201444"
    assert r.headers["content-type"].startswith("text/plain")
    assert (
        client.get(whatsapp.WEBHOOK_PATH, params={**q, "hub.verify_token": "no"}).status_code == 403
    )
    assert (
        client.get(whatsapp.WEBHOOK_PATH, params={**q, "hub.mode": "unsubscribe"}).status_code
        == 403
    )
    assert client.get(whatsapp.WEBHOOK_PATH).status_code == 403


def test_an_unset_verify_token_never_verifies():
    client = TestClient(whatsapp.create_app(SECRET, "", Recorder()))
    q = {"hub.mode": "subscribe", "hub.verify_token": "", "hub.challenge": "1"}
    assert client.get(whatsapp.WEBHOOK_PATH, params=q).status_code == 403


# ------------------------------------------------------------------ webhook: signature


def post(client, body: bytes, signature="auto"):
    headers = {"Content-Type": "application/json"}
    if signature == "auto":
        headers["X-Hub-Signature-256"] = sign(body)
    elif signature:
        headers["X-Hub-Signature-256"] = signature
    return client.post(whatsapp.WEBHOOK_PATH, content=body, headers=headers)


def test_valid_signature_returns_200_and_schedules_without_running():
    client, rec, pending = app_client()
    body = json.dumps(payload(text())).encode()
    r = post(client, body)
    assert r.status_code == 200
    assert rec.payloads == [] and len(pending) == 1  # scheduled, not run
    asyncio.run(pending[0])
    assert rec.payloads[0]["entry"][0]["id"] == "waba"


def test_invalid_or_missing_signatures_are_rejected_before_parsing():
    client, rec, pending = app_client()
    body = json.dumps(payload(text())).encode()
    assert post(client, body, signature=None).status_code == 401
    assert post(client, body, signature="sha256=" + "0" * 64).status_code == 401
    assert post(client, body, signature=sign(body, "wrong")).status_code == 401
    assert post(client, body, signature="sha1=" + sign(body)[7:]).status_code == 401
    assert post(client, body, signature="sha256=").status_code == 401
    # invalid JSON with a bad signature is a 401 (not a 400): nothing was parsed
    assert post(client, b"not json", signature="sha256=00").status_code == 401
    assert pending == [] and rec.payloads == []


def test_signature_check_is_hmac_sha256_over_the_raw_bytes():
    body = b'{"a": 1}'
    assert verify_signature(SECRET, body, sign(body))
    assert verify_signature(SECRET, body, sign(body).upper().replace("SHA256", "sha256"))
    assert not verify_signature(SECRET, body + b" ", sign(body))
    assert not verify_signature("", body, sign(body, ""))
    assert not verify_signature(SECRET, body, None)


def test_oversized_bodies_are_refused_and_bad_json_is_400():
    client, _r, pending = app_client()
    big = b" " * (whatsapp.MAX_BODY + 1)
    assert post(client, big).status_code == 413
    assert post(client, b"not json").status_code == 400
    assert pending == []


def test_docs_are_not_exposed():
    client, _r, _p = app_client()
    assert client.get("/openapi.json").status_code == 404 and client.get("/docs").status_code == 404


async def test_the_default_spawn_runs_the_adapter_in_the_background():
    rec = Recorder()
    app = whatsapp.create_app(SECRET, VERIFY, rec)
    body = json.dumps({"entry": []}).encode()
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post(
            whatsapp.WEBHOOK_PATH, content=body, headers={"X-Hub-Signature-256": sign(body)}
        )
    assert r.status_code == 200
    await asyncio.gather(*app.state.tasks)
    assert rec.payloads == [{"entry": []}]


# ------------------------------------------------------------------ payload -> Incoming


async def test_text_message_is_a_direct_answered_incoming(fx):
    api, a, port, adapter = fx
    await adapter.on_payload(payload(text("how much did we spend?")))
    await a.drain()
    q, kw = a.ask_fn.calls[0]
    assert q == "how much did we spend?"
    assert kw["guild_id"] == ids.tenant_id("whatsapp", PHONE_ID)
    (out,) = api.sent
    assert out == {"to": ALICE, "body": "ok", "reply_to": "wamid.1"}
    assert api.reads == ["wamid.1"]  # read receipt (with the typing indicator)


def test_incoming_fields(fx):
    _api, _a, _port, adapter = fx
    m = adapter._incoming(text("hi", mid="wamid.7"), {ALICE: "Alice"})
    assert m.platform == "whatsapp" and m.tenant == PHONE_ID
    assert m.room == ALICE and m.user == ALICE and m.user_name == "Alice"
    assert m.is_direct and m.message == "wamid.7" and m.at == NOW and m.thread is None
    assert ids.tenant_id("whatsapp", PHONE_ID) >= ids.DISCORD_CEILING
    assert ids.platform_of(ids.tenant_id("whatsapp", PHONE_ID)) == "whatsapp"


async def test_image_with_caption_downloads_lazily(fx):
    api, a, port, adapter = fx
    img = msg("image", image={"id": "img1", "mime_type": "image/jpeg", "caption": "this receipt"})
    m = adapter._incoming(img, {})
    assert m.text == "this receipt" and api.fetched == []
    (f,) = m.files
    assert f.name == "image-wamid.1.jpg" and f.mime == "image/jpeg"
    assert f.fetch() == b"jpegbytes" and api.fetched == ["img1"]


async def test_document_keeps_its_name_and_mime(fx):
    _api, _a, _port, adapter = fx
    doc = msg(
        "document",
        document={"id": "doc1", "mime_type": "application/pdf", "filename": "bill.pdf"},
    )
    m = adapter._incoming(doc, {})
    assert m.text == ""
    (f,) = m.files
    assert (f.name, f.mime) == ("bill.pdf", "application/pdf") and f.fetch() == b"%PDF-bytes"


async def test_reply_context_to_a_salid_message_is_recognised(fx):
    api, a, port, adapter = fx
    await adapter.on_payload(payload(text("first")))
    await a.drain()
    out_id = "wamid.OUT1"
    assert out_id in port.sent_ids
    quoted = adapter._incoming(text("and then?", mid="wamid.2", context={"id": out_id}), {})
    other = adapter._incoming(text("x", mid="wamid.3", context={"id": "wamid.someoneelse"}), {})
    assert quoted.replies_to_bot and not other.replies_to_bot


async def test_other_phone_numbers_statuses_and_unsupported_types_are_ignored(fx):
    api, a, port, adapter = fx
    await adapter.on_payload(payload(text("not for us"), phone="999"))
    await adapter.on_payload(payload(statuses=[{"id": "wamid.OUT1", "status": "delivered"}]))
    await adapter.on_payload(payload(msg("sticker", mid="s"), msg("video", mid="v")))
    await adapter.on_payload(payload(msg("text", mid="empty", text={"body": "  "})))
    await adapter.on_payload({"entry": [{"changes": [{"value": None}]}]})
    await adapter.on_payload({})
    await a.drain()
    assert a.ask_fn.calls == [] and api.sent == [] and api.reads == []


# ------------------------------------------------------------------ BSUID / usernames

BSUID = "US.13491208655302741918"
BSUID2 = "US.99999999999999999999"


def bsuid_msg(mid="wamid.b1", body="hello there", **kw):
    """A user with a username: no `from`, only `from_user_id`."""
    m = {"id": mid, "from_user_id": BSUID, "timestamp": TS, "type": "text", "text": {"body": body}}
    m.update(kw)
    return m


async def test_a_message_without_a_phone_number_is_answered_to_the_bsuid(fx):
    api, a, port, adapter = fx
    p = payload(bsuid_msg())
    p["entry"][0]["changes"][0]["value"]["contacts"] = [
        {"profile": {"name": "Nok"}, "user_id": BSUID}  # no wa_id
    ]
    await adapter.on_payload(p)
    await a.drain()
    assert len(a.ask_fn.calls) == 1
    assert api.sent == [{"to": BSUID, "body": "ok", "reply_to": "wamid.b1"}]  # FakeApi's key
    m = adapter._incoming(bsuid_msg(mid="wamid.b2"), {BSUID: "Nok"})
    assert m.user == BSUID and m.room == BSUID and m.user_name == "Nok" and m.is_direct


def test_the_room_is_the_bsuid_even_when_the_phone_number_shows_up_later(fx):
    _api, _a, _port, adapter = fx
    without = adapter._incoming(bsuid_msg(mid="m1"), {})
    with_phone = adapter._incoming(bsuid_msg(mid="m2", **{"from": ALICE}), {})
    assert without.room == with_phone.room == BSUID  # one chat, one memory
    assert without.user == BSUID and with_phone.user == ALICE  # the wa_id wins for the user
    legacy = adapter._incoming(text("old payload", mid="m3"), {})
    assert legacy.room == ALICE  # no from_user_id (pre-BSUID payloads): the wa_id
    assert adapter._incoming({"id": "m4", "type": "text", "text": {"body": "x"}}, {}) is None


async def test_bsuid_is_sent_as_recipient_and_a_phone_number_as_to():
    seen = []
    t = graph_transport(
        {"/messages": lambda r: httpx.Response(200, json={"messages": [{"id": "wamid.N"}]})}, seen
    )
    api = WhatsAppApi(TOKEN, PHONE_ID, transport=t)
    await api.send_text(BSUID, "hi", None)
    await api.send_text(ALICE, "hi", None)
    await api.aclose()
    b_bsuid, b_phone = seen[0][3], seen[1][3]
    assert b_bsuid["recipient"] == BSUID and "to" not in b_bsuid
    assert b_phone["to"] == ALICE and "recipient" not in b_phone


async def test_bsuid_users_are_not_executives_unless_listed(s, tmp_path):
    ps = PlatformSettings(_env_file=None, whatsapp_exec_numbers=ALICE)
    a = make_assistant(s, tmp_path, ps=ps)
    port = WhatsAppPort(FakeApi(), exec_bsuids={BSUID2})
    adapter = WhatsAppAdapter(a, port, PHONE_ID)
    await adapter.on_payload(payload(bsuid_msg(mid="m1")))
    await adapter.on_payload(payload(bsuid_msg(mid="m2", from_user_id=BSUID2)))
    await adapter.on_payload(payload(text("hi", mid="m3")))  # Alice, listed by phone number
    await adapter.on_payload(payload(text("hi", mid="m4", sender=BOB)))
    await a.drain()
    by_text = sorted(c[1]["is_exec"] for c in a.ask_fn.calls)
    assert by_text == [False, False, True, True]
    assert not await WhatsAppPort(FakeApi()).is_admin(adapter._incoming(bsuid_msg(), {}))
    assert await port.is_admin(adapter._incoming(bsuid_msg(mid="m9", from_user_id=BSUID2), {}))


# ------------------------------------------------------------------ body cap


def test_webhook_bodies_up_to_3_mb_are_accepted():
    assert whatsapp.MAX_BODY >= 3 * 1000 * 1000
    client, _r, pending = app_client()
    pad = " " * (3 * 1000 * 1000)
    ok = json.dumps({"entry": [], "pad": pad}).encode()
    assert post(client, ok).status_code == 200 and len(pending) == 1


# ------------------------------------------------------------------ statuses and rate limits


async def test_failed_statuses_are_logged_by_error_code_only(fx, caplog):
    api, a, port, adapter = fx
    caplog.set_level(logging.DEBUG)
    failed = {
        "id": "wamid.OUT1",
        "status": "failed",
        "recipient_id": ALICE,
        "errors": [{"code": 131047, "title": f"Re-engagement for {ALICE}", "message": "secret"}],
    }
    ok = {"id": "wamid.OUT2", "status": "delivered", "recipient_id": ALICE}
    await adapter.on_payload(payload(statuses=["junk", failed, ok, {"status": "failed"}]))
    assert "131047" in caplog.text and "unknown" in caplog.text
    for needle in (ALICE, "Re-engagement", "secret", "wamid"):
        assert needle not in caplog.text
    assert api.sent == [] and a.ask_fn.calls == []


class RateLimitedApi(FakeApi):
    """Fails the first `fails` sends of every part with `code`, then succeeds."""

    def __init__(self, code, fails, always_fail_part=None):
        super().__init__()
        self.code, self.fails, self.always_fail_part = code, fails, always_fail_part
        self.attempts: dict[str, int] = {}

    async def send_text(self, to, body, reply_to):
        n = self.attempts[body] = self.attempts.get(body, 0) + 1
        if body == self.always_fail_part or n <= self.fails:
            raise WhatsAppError("send", 400, self.code)
        return await super().send_text(to, body, reply_to)


def slow_port(api):
    waited: list[int] = []

    async def sleep(seconds):
        waited.append(seconds)

    return WhatsAppPort(api, sleep=sleep), waited


@pytest.mark.parametrize("code", [130429, 131056])
async def test_rate_limit_errors_back_off_and_retry(fx, code):
    _api, _a, _port, adapter = fx
    api = RateLimitedApi(code, fails=2)
    port, waited = slow_port(api)
    m = adapter._incoming(text("q", mid="wamid.q"), {})
    out = await port.deliver(m, ["one", "two"], None)
    assert [x["body"] for x in api.sent] == ["one", "two"] and len(out) == 2
    assert waited == [4, 16, 4, 16]  # bounded 4^X backoff, per part


async def test_retries_are_bounded_and_later_parts_are_never_dropped_silently(fx, caplog):
    _api, _a, _port, adapter = fx
    api = RateLimitedApi(131056, fails=0, always_fail_part="two")
    port, waited = slow_port(api)
    m = adapter._incoming(text("q", mid="wamid.q"), {})
    with pytest.raises(WhatsAppError):
        await port.deliver(m, ["one", "two", "three"], None)
    assert api.attempts == {"one": 1, "two": 3}  # first try + two retries, then it gives up
    assert waited == [4, 16]
    assert "1 of 3 parts sent, 2 not sent" in caplog.text and "131056" in caplog.text
    assert ALICE not in caplog.text


async def test_other_send_errors_are_not_retried_but_still_reported(fx, caplog):
    _api, _a, _port, adapter = fx
    api = RateLimitedApi(131047, fails=9)
    port, waited = slow_port(api)
    m = adapter._incoming(text("q", mid="wamid.q"), {})
    with pytest.raises(WhatsAppError):
        await port.deliver(m, ["one", "two"], None)
    assert api.attempts == {"one": 1} and waited == []
    assert "0 of 2 parts sent, 2 not sent" in caplog.text


# ------------------------------------------------------------------ audio


def test_voice_notes_are_passed_to_core_as_audio_files(fx):
    api, _a, _port, adapter = fx
    api.media["voice1"] = b"oggbytes"
    voice = msg(
        "audio", audio={"id": "voice1", "mime_type": "audio/ogg; codecs=opus", "voice": True}
    )
    m = adapter._incoming(voice, {})
    assert m.text == ""
    (f,) = m.files
    assert (f.name, f.mime) == ("audio-wamid.1.ogg", "audio/ogg")
    assert f.fetch() == b"oggbytes" and api.fetched == ["voice1"]
    mp3 = adapter._incoming(
        msg("audio", mid="w2", audio={"id": "voice1", "mime_type": "audio/mpeg"}), {}
    )
    assert mp3.files[0].name == "audio-w2.mp3"
    bare = adapter._incoming(msg("audio", mid="w3", audio={"id": "voice1"}), {})
    assert bare.files[0].name == "audio-w3.ogg" and bare.files[0].mime is None


async def test_core_receives_audio_and_skips_it_as_an_unsupported_type(fx):
    # Core only reads images, PDFs and text: `file_mime` returns None for audio, so the file is
    # skipped ("unsupported file type") rather than transcribed. The adapter just passes it on.
    api, a, port, adapter = fx
    api.media["voice1"] = b"oggbytes"
    voice = msg("audio", audio={"id": "voice1", "mime_type": "audio/ogg; codecs=opus"})
    await adapter.on_payload(payload(voice))
    await a.drain()
    assert api.fetched == []  # never downloaded: not a type core can use


# ------------------------------------------------------------------ human escalation


@pytest.mark.parametrize(
    "words",
    ["human", "Human!", "คน", "ติดต่อเจ้าหน้าที่", "ขอคุยกับเจ้าหน้าที่หน่อยค่ะ", "I want to talk to a human"],
)
async def test_asking_for_a_person_gets_a_fixed_answer_without_the_model(fx, words):
    api, a, port, adapter = fx
    await adapter.on_payload(payload(text(words)))
    await a.drain()
    assert a.ask_fn.calls == []
    assert [x["body"] for x in api.sent] == [whatsapp.HUMAN_REPLY]
    assert api.reads == []  # no typing indicator for a fixed answer
    assert "administrators" in whatsapp.HUMAN_REPLY and "ผู้ดูแล" in whatsapp.HUMAN_REPLY


@pytest.mark.parametrize(
    "words",
    ["how many people came on Monday?", "คนที่จ่ายเงินเมื่อวานคือใคร", "สรุปค่าใช้จ่ายของคนในทีม"],
)
async def test_ordinary_questions_about_people_still_reach_the_model(fx, words):
    api, a, port, adapter = fx
    await adapter.on_payload(payload(text(words)))
    await a.drain()
    assert len(a.ask_fn.calls) == 1
    assert whatsapp.HUMAN_REPLY not in [x["body"] for x in api.sent]


async def test_the_human_answer_goes_to_a_bsuid_user_too_and_survives_a_send_error(fx, caplog):
    api, a, port, adapter = fx
    await adapter.on_payload(payload(bsuid_msg(body="human")))
    assert api.sent[0]["to"] == BSUID

    async def boom(*_a):
        raise WhatsAppError("send", 500)

    api.send_text = boom
    await adapter.on_payload(payload(text("human", mid="m2")))  # logged, never raised
    assert "human-contact reply failed" in caplog.text


async def test_a_retried_delivery_is_answered_once(fx):
    api, a, port, adapter = fx
    p = payload(text("once please", mid="wamid.dup"))
    await adapter.on_payload(p)
    await adapter.on_payload(p)
    await a.drain()
    assert len(a.ask_fn.calls) == 1 and len(api.sent) == 1


async def test_executives_are_only_the_configured_numbers_in_any_notation(s, tmp_path):
    ps = PlatformSettings(_env_file=None, whatsapp_exec_numbers="+66 81-111-2222")
    assert ps.whatsapp_exec_numbers == {ALICE}
    a = make_assistant(s, tmp_path, ps=ps)
    api = FakeApi()
    adapter = WhatsAppAdapter(a, WhatsAppPort(api), PHONE_ID)
    await adapter.on_payload(payload(text("hi", mid="m1")))
    await adapter.on_payload(payload(text("hi", mid="m2", sender=BOB)))
    await a.drain()
    assert sorted(c[1]["is_exec"] for c in a.ask_fn.calls) == [False, True]


async def test_each_chat_sees_only_its_own_memory(fx):
    api, a, port, adapter = fx
    await adapter.on_payload(payload(text("alice secret plan", mid="m1")))
    await adapter.on_payload(payload(text("bob question", mid="m2", sender=BOB)))
    await a.drain()
    m_alice = adapter._incoming(text("x", mid="m9"), {})
    m_bob = adapter._incoming(text("y", mid="m10", sender=BOB), {})
    alice_texts = [x.text for x in stored(a, m_alice)]
    bob_texts = [x.text for x in stored(a, m_bob)]
    assert "alice secret plan" in alice_texts and "bob question" not in alice_texts
    assert "bob question" in bob_texts and "alice secret plan" not in bob_texts
    assert rid_of(m_alice) != rid_of(m_bob)
    view = await port.room_view(m_alice)
    assert view.audience is None and view.rooms == []  # fail closed: the room alone
    assert await port.history(m_alice, 10) == [] and not await port.is_admin(m_alice)


# ------------------------------------------------------------------ replies


def test_long_answers_are_split_at_4096_and_formatted():
    port = WhatsAppPort(FakeApi())
    parts = port.format("para one. " * 700)
    assert len(parts) >= 2 and all(len(p) <= whatsapp.TEXT_LIMIT for p in parts)
    (one,) = port.format("## Title\n\n**bold** and *it* and ~~old~~\n- item")
    assert one == "*Title*\n\n*bold* and _it_ and ~old~\n• item"
    assert textfmt.to_whatsapp("`x` and\n```py\nprint(1)\n```") == "```x``` and\n```print(1)```"
    assert textfmt.to_whatsapp("[docs](https://e.com/a)") == "docs (https://e.com/a)"


async def test_every_chunk_is_sent_and_only_the_first_quotes_the_question(fx):
    api, a, port, adapter = fx
    m = adapter._incoming(text("q", mid="wamid.q"), {})
    ids_out = await port.deliver(m, ["one", "two", "three"], None)
    assert [x["body"] for x in api.sent] == ["one", "two", "three"]
    assert [x["reply_to"] for x in api.sent] == ["wamid.q", None, None]
    assert ids_out == ["wamid.OUT1", "wamid.OUT2", "wamid.OUT3"]


def graph_transport(routes, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            (
                request.method,
                request.url.path,
                request.headers.get("authorization"),
                json.loads(request.content) if request.content else None,
            )
        )
        for suffix, fn in routes.items():
            if request.url.path.endswith(suffix):
                return fn(request)
        return httpx.Response(404, json={"error": {"code": 100}})

    return httpx.MockTransport(handler)


async def test_graph_requests_have_the_documented_shape():
    seen = []
    t = graph_transport(
        {
            "/messages": lambda r: httpx.Response(200, json={"messages": [{"id": "wamid.N"}]}),
        },
        seen,
    )
    api = WhatsAppApi(TOKEN, PHONE_ID, "v26.0", transport=t)
    assert await api.send_text(ALICE, "hi", "wamid.q") == "wamid.N"
    await api.send_text(ALICE, "plain", None)
    await api.mark_read("wamid.q")
    await api.aclose()
    path = f"/v26.0/{PHONE_ID}/messages"
    (m1, p1, auth1, b1), (_m2, _p2, _a2, b2), (_m3, p3, _a3, b3) = seen
    assert (m1, p1, auth1) == ("POST", path, f"Bearer {TOKEN}") and p3 == path
    assert b1 == {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": ALICE,
        "type": "text",
        "text": {"body": "hi", "preview_url": False},
        "context": {"message_id": "wamid.q"},
    }
    assert "context" not in b2
    assert b3 == {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": "wamid.q",
        "typing_indicator": {"type": "text"},
    }


async def test_api_version_comes_from_settings():
    seen = []
    t = graph_transport({"/messages": lambda r: httpx.Response(200, json={})}, seen)
    api = WhatsAppApi(TOKEN, PHONE_ID, "v99.1", transport=t)
    await api.mark_read("x")
    await api.aclose()
    assert seen[0][1] == f"/v99.1/{PHONE_ID}/messages"
    assert PlatformSettings(_env_file=None).whatsapp_api_version == "v26.0"
    with pytest.raises(ValueError):
        PlatformSettings(_env_file=None, whatsapp_api_version="v1/../x")


async def test_a_failed_read_receipt_does_not_stop_the_answer(fx, caplog):
    api, a, port, adapter = fx

    async def boom(_mid):
        raise WhatsAppError("read", 500)

    api.mark_read = boom
    await adapter.on_payload(payload(text("still answered")))
    await a.drain()
    assert len(api.sent) == 1


# ------------------------------------------------------------------ media


def media_routes(
    size, body, url="https://lookaside.fbsbx.com/whatsapp_business/attachments/?mid=1"
):
    return {
        "/MEDIA1": lambda r: httpx.Response(
            200, json={"url": url, "mime_type": "image/jpeg", "file_size": size, "id": "MEDIA1"}
        ),
        "/attachments/": lambda r: httpx.Response(200, content=body),
    }


def test_media_fetch_is_two_steps_with_the_bearer_header_both_times():
    seen = []
    api = WhatsAppApi(TOKEN, PHONE_ID, transport=graph_transport(media_routes(5, b"abcde"), seen))
    assert api.fetch_media("MEDIA1") == b"abcde"
    assert [(m, p) for m, p, _a, _b in seen] == [
        ("GET", "/v26.0/MEDIA1"),
        ("GET", "/whatsapp_business/attachments/"),
    ]
    assert all(a == f"Bearer {TOKEN}" for _m, _p, a, _b in seen)


def test_media_over_the_limit_declared_or_streamed_is_refused():
    seen = []
    declared = WhatsAppApi(
        TOKEN,
        PHONE_ID,
        transport=graph_transport(media_routes(whatsapp.DOWNLOAD_MAX + 1, b"x"), seen),
    )
    with pytest.raises(ValueError):
        declared.fetch_media("MEDIA1")
    assert all("/attachments/" not in p for _m, p, _a, _b in seen)  # never started downloading
    liar = WhatsAppApi(
        TOKEN,
        PHONE_ID,
        transport=graph_transport(media_routes(1, b"z" * (whatsapp.MAX_FILE_BYTES + 1)), []),
    )
    with pytest.raises(ValueError):
        liar.fetch_media("MEDIA1")


def test_media_failures_and_non_https_urls_are_token_free_errors():
    api = WhatsAppApi(TOKEN, PHONE_ID, transport=graph_transport({}, []))
    with pytest.raises(WhatsAppError) as exc:
        api.fetch_media("MEDIA1")
    assert TOKEN not in str(exc.value) and "MEDIA1" not in str(exc.value)
    plain = WhatsAppApi(
        TOKEN,
        PHONE_ID,
        transport=graph_transport(media_routes(1, b"x", url="http://evil.example/a"), []),
    )
    with pytest.raises(WhatsAppError):
        plain.fetch_media("MEDIA1")

    def boom(_r):
        raise httpx.ConnectError("cannot reach https://lookaside.fbsbx.com/?token=" + TOKEN)

    api2 = WhatsAppApi(TOKEN, PHONE_ID, transport=httpx.MockTransport(boom))
    with pytest.raises(WhatsAppError) as exc2:
        api2.fetch_media("MEDIA1")
    assert TOKEN not in str(exc2.value) and exc2.value.__cause__ is None


# ------------------------------------------------------------------ secrets and PII in logs


async def test_no_secret_or_phone_number_reaches_the_logs(caplog, s, tmp_path):
    caplog.set_level(logging.DEBUG)
    leaky = f"Invalid {TOKEN} for {ALICE} app {SECRET}"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/messages") and b"read" in request.content:
            raise httpx.ReadTimeout(f"timed out calling {request.url} for {ALICE}")
        return httpx.Response(401, json={"error": {"code": 190, "message": leaky}})

    api = WhatsAppApi(TOKEN, PHONE_ID, transport=httpx.MockTransport(handler))
    a = make_assistant(s, tmp_path, ask=FakeAsk("answer to " + ALICE))
    port = WhatsAppPort(api)
    adapter = WhatsAppAdapter(a, port, PHONE_ID)
    await adapter.on_payload(payload(text("what is the plan?", mid="wamid.log")))
    await a.drain()
    with pytest.raises(WhatsAppError) as e1:
        await api.send_text(ALICE, "x", None)
    with pytest.raises(WhatsAppError) as e2:
        await api.mark_read("x")
    assert e1.value.status == 401 and e1.value.graph_code == 190 and e2.value.status is None
    for needle in (TOKEN, "SECRET-access-token", SECRET, VERIFY, ALICE, "what is the plan"):
        assert needle not in caplog.text, needle
        assert needle not in str(e1.value) + str(e2.value)
    assert "graph.facebook.com" not in caplog.text
    assert caplog.records  # the failures were logged ...
    for logger_name in ("httpx", "httpcore"):
        assert logging.getLogger(logger_name).level >= logging.WARNING
    await api.aclose()


async def test_a_handler_crash_logs_only_the_type(caplog, s, tmp_path):
    caplog.set_level(logging.DEBUG)
    a = make_assistant(s, tmp_path)
    adapter = WhatsAppAdapter(a, WhatsAppPort(FakeApi()), PHONE_ID)

    async def crash(*_a, **_k):
        raise RuntimeError(f"boom with {ALICE} and a private text")

    a.handle = crash  # type: ignore[method-assign]
    await adapter.on_payload(payload(text("private text")))
    assert "RuntimeError" in caplog.text
    assert ALICE not in caplog.text and "private text" not in caplog.text


def test_settings_never_print_the_secrets():
    ps = PlatformSettings(
        _env_file=None,
        whatsapp_access_token=TOKEN,
        whatsapp_app_secret=SECRET,
        whatsapp_verify_token=VERIFY,
        whatsapp_phone_number_id=PHONE_ID,
    )
    shown = repr(ps) + str(ps)
    for secret in (TOKEN, SECRET, VERIFY):
        assert secret not in shown
    assert ps.whatsapp_enabled and ps.enabled_platforms() == ["whatsapp"]
    assert ps.whatsapp_port == 8092 and ps.whatsapp_host == "127.0.0.1"
    # every one of the four is needed
    assert not PlatformSettings(
        _env_file=None, whatsapp_access_token=TOKEN, whatsapp_phone_number_id=PHONE_ID
    ).whatsapp_enabled


def test_whatsapp_tenants_are_their_own_range():
    t = ids.tenant_id("whatsapp", PHONE_ID)
    assert ids.DISCORD_CEILING <= t < 2**63 and ids.platform_of(t) == "whatsapp"
    for other in ("slack", "line", "telegram"):
        assert ids.tenant_id(other, PHONE_ID) != t
        assert ids.platform_of(ids.tenant_id(other, PHONE_ID)) == other
    assert ids.platform_of(12345) is None


# ------------------------------------------------------------------ smoke


async def test_smoke_prints_the_verified_name_only():
    ps = PlatformSettings(
        _env_file=None,
        whatsapp_access_token=TOKEN,
        whatsapp_app_secret=SECRET,
        whatsapp_verify_token=VERIFY,
        whatsapp_phone_number_id=PHONE_ID,
    )
    assert "whatsapp: will start (webhook port 8092)" in smoke.plan(ps)
    assert any(
        "SALID_WHATSAPP_APP_SECRET" in p for p in smoke.plan(PlatformSettings(_env_file=None))
    )

    async def good(_ps):
        return {"verified_name": "Salid Demo"}

    async def bad(_ps):
        raise WhatsAppError("profile", 401)

    assert await smoke.check(ps, whatsapp_check=good) == [
        "whatsapp: token ok (verified_name=Salid Demo)"
    ]
    (line,) = await smoke.check(ps, whatsapp_check=bad)
    assert line.startswith("whatsapp: token check FAILED") and TOKEN not in line


async def test_check_asks_for_the_name_fields_and_returns_no_secrets(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, dict(request.url.params), request.headers["authorization"]))
        return httpx.Response(
            200,
            json={
                "verified_name": "Salid Demo",
                "display_phone_number": "+1 555 000 1111",
                "id": PHONE_ID,
            },
        )

    real = whatsapp.WhatsAppApi
    monkeypatch.setattr(
        whatsapp,
        "WhatsAppApi",
        lambda token, phone, version: real(
            token, phone, version, transport=httpx.MockTransport(handler)
        ),
    )
    ps = PlatformSettings(
        _env_file=None,
        whatsapp_access_token=TOKEN,
        whatsapp_phone_number_id=PHONE_ID,
    )
    out = await whatsapp.check(ps)
    assert out == {"verified_name": "Salid Demo"}  # not the phone number
    assert seen == [
        (
            f"/v26.0/{PHONE_ID}",
            {"fields": "display_phone_number,verified_name"},
            f"Bearer {TOKEN}",
        )
    ]
