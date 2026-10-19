"""LINE: reply-token window, push quota/idempotency, redelivery, edits, quotes, notices."""

import logging
import time

import pytest
from conftest import FixedClockSettings
from platform_fakes import Clock, inc, make_assistant, stored
from test_line import BOT, FakeLineApi, api_error, big, mev

from salid.guild_settings import GuildSettingsUpdate
from salid.platforms import ids, line
from salid.platforms.line import LineAdapter, LinePort, notice_text, retry_key


@pytest.fixture
def fx(tmp_path):
    api = FakeLineApi()
    clock = Clock(500.0)
    a = make_assistant(FixedClockSettings(exec_user_ids=set()), tmp_path)
    port = LinePort(api, a.ps, clock=clock)
    return api, a, port, LineAdapter(a, port, BOT, clock=clock), clock


def gmsg(clock, **kw):
    return inc(platform="line", reply_token="T", raw_at=clock(), room="Cgroup", **kw)


# ------------------------------------------------------------------ 1. reply token window


async def test_the_reply_token_is_used_until_about_a_minute_has_passed(fx):
    api, a, port, adapter, clock = fx
    clock.advance(45)  # past the old 25 s limit, inside the minute LINE grants
    await port.deliver(gmsg(clock, message="m1"), ["x"], None)
    assert api.replies == [("T", ["x"])] and api.pushes == []
    assert port.token_window() >= 50


async def test_an_old_token_goes_straight_to_push(fx):
    api, a, port, adapter, clock = fx
    msg = gmsg(clock)
    clock.advance(line.REPLY_TOKEN_SECONDS + 1)
    await port.deliver(msg, ["x"], None)
    assert api.replies == [] and len(api.pushes) == 1


# ------------------------------------------------------------------ 2. quota / 429


async def test_push_quota_exhausted_is_logged_clearly_and_stops(fx, caplog):
    api, a, port, adapter, clock = fx
    api.push_errors = [api_error(429)]
    msg = inc(platform="line", room="Ualice", message="m1", tenant=BOT)
    with caplog.at_level(logging.ERROR):
        sent = await port.deliver(msg, [big(i) for i in range(7)], None)
    assert sent == [] and api.pushes == []
    err = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(err) == 1 and "429" in err[0].getMessage() and "NOT delivered" in err[0].getMessage()
    assert "quota" in err[0].getMessage()
    assert len(api.push_keys) == 1  # the second batch was not even tried
    # the raw LINE ids never reach the logs: only the derived tenant / room ids
    assert "Ualice" not in caplog.text and BOT not in caplog.text
    assert str(ids.room_id("line", BOT, "Ualice")) in err[0].getMessage()
    assert str(ids.tenant_id("line", BOT)) in err[0].getMessage()


async def test_a_rate_limited_reply_is_not_turned_into_a_paid_push(fx, caplog):
    api, a, port, adapter, clock = fx
    api.fail_reply = 429
    with caplog.at_level(logging.ERROR):
        await port.deliver(gmsg(clock, message="m1"), ["x"], None)
    assert api.pushes == [] and "429" in caplog.text


# ------------------------------------------------------------------ 3. retry key


def test_the_retry_key_is_stable_per_message_chunk_and_content():
    m = inc(platform="line", tenant=BOT, room="Cg", message="m1")
    k = retry_key(m, 0, ["a"])
    assert k == retry_key(m, 0, ["a"]) and len(k) == 36
    assert len({k, retry_key(m, 5, ["a"]), retry_key(m, 0, ["b"])}) == 3
    assert retry_key(inc(platform="line", tenant=BOT, room="Cg", message="m2"), 0, ["a"]) != k


async def test_a_retried_push_reuses_its_key_and_a_409_counts_as_delivered(fx, monkeypatch):
    api, a, port, adapter, clock = fx
    monkeypatch.setattr(line, "PUSH_RETRY_DELAY_S", 0)
    msg = inc(platform="line", tenant=BOT, room="Ualice", message="m1")
    api.push_errors = [TimeoutError("read timed out")]
    await port.deliver(msg, ["hello"], None)
    assert len(api.pushes) == 1 and len(api.push_keys) == 2
    assert api.push_keys[0] == api.push_keys[1] == retry_key(msg, 0, ["hello"])
    api.push_errors = [api_error(409)]
    assert await port.deliver(msg, ["hello"], None) == []  # already accepted: no error
    assert len(api.pushes) == 1


async def test_a_push_rejected_for_good_is_not_retried(fx):
    api, a, port, adapter, clock = fx
    api.push_errors = [api_error(400)]
    with pytest.raises(Exception, match="400"):
        await port.deliver(inc(platform="line", room="Ualice", message="m1"), ["x"], None)
    assert len(api.push_keys) == 1


# ------------------------------------------------------------------ 4. redelivery


async def test_the_same_webhook_event_id_is_handled_once(fx):
    api, a, port, adapter, clock = fx
    ev = mev("สลิด หวัดดี", webhookEventId="01AAA", deliveryContext={"isRedelivery": False})
    await adapter.on_event(ev, BOT)
    await adapter.on_event(
        {**ev, "deliveryContext": {"isRedelivery": True}}, BOT
    )  # LINE sent it again
    assert len(a.ask_fn.calls) == 1 and len(api.replies) == 1


async def test_a_redelivery_of_an_event_we_never_saw_is_handled(fx):
    api, a, port, adapter, clock = fx
    ev = mev("สลิด หวัดดี", webhookEventId="01BBB", deliveryContext={"isRedelivery": True})
    ev["timestamp"] = int(time.time() * 1000)
    await adapter.on_event(ev, BOT)
    assert len(a.ask_fn.calls) == 1 and api.replies


async def test_an_old_redelivery_does_not_try_its_dead_reply_token(fx):
    api, a, port, adapter, clock = fx
    ev = mev("สลิด หวัดดี", webhookEventId="01CCC", deliveryContext={"isRedelivery": True})
    ev["timestamp"] = int((time.time() - 3600) * 1000)
    await adapter.on_event(ev, BOT)
    assert api.replies == [] and len(api.pushes) == 1  # no token to try: push directly


# ------------------------------------------------------------------ 5. edits and unsend


def edit_ev(text, ts, mid="m1"):
    return {
        "type": "messageEdited",
        "mode": "active",
        "timestamp": ts,
        "replyToken": "edit-token",
        "source": {"type": "group", "groupId": "Cgroup", "userId": "Ualice"},
        "message": {"type": "text", "id": mid, "text": text},
    }


async def test_an_edited_message_updates_memory_and_is_not_answered(fx):
    api, a, port, adapter, clock = fx
    await adapter.on_event(mev("tuesday at 3", mid="m1"), BOT)
    await adapter.on_event(edit_ev("wednesday at 4", 1791954005000), BOT)
    await a.drain()
    msg = inc(platform="line", tenant=BOT, room="Cgroup")
    assert [m.text for m in stored(a, msg)] == ["wednesday at 4"]
    assert not a.ask_fn.calls and not api.replies and not api.pushes


async def test_a_stale_edit_arriving_late_does_not_overwrite_a_newer_one(fx):
    api, a, port, adapter, clock = fx
    await adapter.on_event(mev("v1", mid="m1"), BOT)
    await adapter.on_event(edit_ev("v3", 1791954009000), BOT)
    await adapter.on_event(edit_ev("v2", 1791954005000), BOT)  # older, arrived second
    await a.drain()
    assert [m.text for m in stored(a, inc(platform="line", tenant=BOT, room="Cgroup"))] == ["v3"]


# ------------------------------------------------------------------ 6. mentions, quotes


async def test_a_mention_of_all_is_not_a_call_for_salid():
    out, me, others = line._strip_self_mentions(
        "@All meeting at 3", [{"index": 0, "length": 4, "type": "all"}]
    )
    assert not me and others


def image_ev(mid, token="ti"):
    ev = mev("", mid=mid, token=token)
    ev["message"] = {"type": "image", "id": mid, "quoteToken": "qt-" + mid}
    return ev


def quote_ev(mid, quoted, *, mention=True, token="tq"):
    ev = mev(
        "@Salid is this receipt ok?" if mention else "nice picture",
        mid=mid,
        token=token,
        mentionees=[{"index": 0, "length": 6, "type": "user", "isSelf": True}] if mention else None,
    )
    ev["message"]["quotedMessageId"] = quoted
    ev["message"]["quoteToken"] = "qt-" + mid
    return ev


async def test_a_mention_quoting_an_image_applies_to_that_image(fx):
    api, a, port, adapter, clock = fx
    await adapter.on_event(image_ev("img1"), BOT)  # not addressed: nothing happens
    assert not a.ask_fn.calls
    clock.advance(120)
    await adapter.on_event(quote_ev("m2", "img1"), BOT)
    q, kw = a.ask_fn.calls[0]
    assert q == "is this receipt ok?"
    assert [f.filename for f in kw["files"]] == ["image-img1.jpg"]
    assert kw["files"][0].data == b"imgbytes"


async def test_a_quote_without_addressing_salid_attaches_nothing(fx):
    api, a, port, adapter, clock = fx
    await adapter.on_event(image_ev("img1"), BOT)
    await adapter.on_event(quote_ev("m2", "img1", mention=False), BOT)
    assert not a.ask_fn.calls


async def test_the_file_cache_forgets_after_ten_minutes(fx):
    api, a, port, adapter, clock = fx
    await adapter.on_event(image_ev("img1"), BOT)
    clock.advance(line.FILE_CACHE_S + 1)
    await adapter.on_event(quote_ev("m2", "img1"), BOT)
    assert not a.ask_fn.calls[0][1].get("files")


async def test_group_replies_quote_the_asker_and_1to1_replies_do_not(fx):
    api, a, port, adapter, clock = fx
    await adapter.on_event(quote_ev("m2", "none"), BOT)
    assert api.reply_quotes == ["qt-m2"]
    ev = mev("hello", src="user", mid="m3", token="t3")
    ev["message"]["quoteToken"] = "qt-m3"
    await adapter.on_event(ev, BOT)
    assert api.reply_quotes == ["qt-m2", None]


async def test_the_quote_token_goes_on_the_first_message_object_only():
    from linebot.v3.messaging import TextMessage  # noqa: F401

    msgs = line.SdkLineApi._messages(["a", "b"], "QT")
    assert msgs[0].quote_token == "QT" and msgs[1].quote_token is None


async def test_the_quote_moves_to_the_push_when_the_reply_token_is_refused(fx):
    api, a, port, adapter, clock = fx
    api.fail_reply = True
    port.remember_quote("Cgroup", "m1", "QT")
    await port.deliver(gmsg(clock, message="m1"), ["x"], None)
    assert api.push_quotes == ["QT"]


# ------------------------------------------------------------------ 7. loading animation


async def test_loading_starts_at_once_in_1to1_even_for_a_fast_answer(fx):
    api, a, port, adapter, clock = fx
    await adapter.on_event(mev("hello", src="user", mid="m1"), BOT)
    assert api.loadings[0] == ("Ualice", line.LOADING_SECONDS) and api.replies


async def test_no_loading_animation_in_groups(fx):
    api, a, port, adapter, clock = fx
    await adapter.on_event(mev("สลิด hi", mid="m1"), BOT)
    assert api.loadings == []


# ------------------------------------------------------------------ 8. join / follow


def join_ev(kind="join", token="jt", **src):
    source = {"type": "group", "groupId": "Cgroup"} if kind == "join" else {"type": "user"}
    if kind == "follow":
        source["userId"] = "Ualice"
    return {"type": kind, "mode": "active", "replyToken": token, "source": source, **src}


async def test_joining_a_group_sends_the_notice_through_the_free_reply_token(fx):
    api, a, port, adapter, clock = fx
    await adapter.on_event(join_ev("join"), BOT)
    ((token, (text,)),) = api.replies
    assert token == "jt" and api.pushes == []
    assert "สลิด" in text and "Salid" in text  # Thai and English
    assert "ยกเลิกการส่ง" in text and "unsend" in text and "remove me" in text
    assert "http" not in text and "www" not in text  # link-free
    assert f"{line.LINE_RETENTION_DAYS} days" in text and f"{line.LINE_RETENTION_DAYS} วัน" in text


async def test_following_in_1to1_sends_the_notice_for_a_chat(fx):
    api, a, port, adapter, clock = fx
    await adapter.on_event(join_ev("follow"), BOT)
    ((_t, (text,)),) = api.replies
    assert "block or delete this chat" in text and "remove me from the group" not in text


async def test_the_notice_is_sent_once_per_event_and_never_pushed(fx):
    api, a, port, adapter, clock = fx
    api.fail_reply = True  # the token is refused: the notice is dropped, not pushed
    await adapter.on_event(join_ev("join"), BOT)
    await adapter.on_event(join_ev("join", token=""), BOT)
    assert api.pushes == [] and api.replies == []


async def test_a_tenant_on_the_default_retention_gets_linees_shorter_default(fx):
    api, a, port, adapter, clock = fx
    gid = ids.tenant_id("line", BOT)
    assert a.team.memory_retention_days(gid) > line.LINE_RETENTION_DAYS
    await adapter.on_event(join_ev("join"), BOT)
    assert a.team.memory_retention_days(gid) == line.LINE_RETENTION_DAYS


async def test_a_retention_an_admin_chose_is_left_alone(fx):
    api, a, port, adapter, clock = fx
    gid = ids.tenant_id("line", BOT)
    a.team.set_memory_retention_days(gid, 30)
    await adapter.on_event(join_ev("join"), BOT)
    assert a.team.memory_retention_days(gid) == 30
    assert "30 days" in api.replies[0][1][0]


async def test_keep_forever_chosen_in_the_hub_is_never_cut_to_lines_default(fx):
    """0 (keep forever) is also the global default: once the hub saved the settings it is a
    company choice, and a join after a restart must not cut it to 7 days (a purge)."""
    api, a, port, adapter, clock = fx
    gid = ids.tenant_id("line", BOT)
    a.team.store.save_settings(gid, GuildSettingsUpdate(memory_retention_days=0), "boss@x.com")
    a.team._cache.clear()
    await adapter.on_event(join_ev("join"), BOT)
    assert a.team.store.get(gid).memory_retention_days == 0
    assert a.team.memory_retention_days(gid) > line.LINE_RETENTION_DAYS
    text = api.replies[0][1][0]
    assert "3650" not in text and "ถาวร" in text and "kept forever" in text


def test_the_forever_notice_uses_the_catalog_text_in_thai_then_english():
    from salid.i18n import t

    text = notice_text(0, False)
    th_stop, en_stop = line.STOP_GROUP
    assert t("line_memory_forever", "th", stop=th_stop) in text
    assert t("line_memory_forever", "en", stop=en_stop) in text
    assert "{" not in text and "3650" not in text and " 0 " not in text


def test_the_notice_without_memory_says_nothing_about_storing():
    text = notice_text(None, False)
    assert "unsend" not in text and "สลิด" in text and "http" not in text


# ------------------------------------------------------------------ 9. no userId


async def test_a_group_message_without_a_user_id_is_remembered_as_anonymous(fx):
    api, a, port, adapter, clock = fx
    ev = mev("from line for pc", mid="m1")
    del ev["source"]["userId"]
    await adapter.on_event(ev, BOT)
    await a.drain()
    (m,) = stored(a, inc(platform="line", tenant=BOT, room="Cgroup"))
    assert m.text == "from line for pc" and "anon" not in m.author_name


async def test_an_anonymous_mention_is_answered_as_a_non_executive(tmp_path):
    s = FixedClockSettings(exec_user_ids=set())
    api, clock = FakeLineApi(), Clock(500.0)
    a = make_assistant(s, tmp_path, line_exec_user_ids="Ualice")
    port = LinePort(api, a.ps, clock=clock)
    adapter = LineAdapter(a, port, BOT, clock=clock)
    ev = mev(
        "@Salid hi",
        mentionees=[{"index": 0, "length": 6, "type": "user", "isSelf": True}],
    )
    del ev["source"]["userId"]
    await adapter.on_event(ev, BOT)
    kw = a.ask_fn.calls[0][1]
    assert kw["is_exec"] is False and kw["user_id"] == ids.user_id("line", "anon:Cgroup:m1")
    assert api.replies


async def test_a_1to1_message_without_a_user_id_is_dropped_with_a_warning(fx, caplog):
    api, a, port, adapter, clock = fx
    ev = mev("hi", src="user", mid="m1")
    del ev["source"]["userId"]
    with caplog.at_level(logging.WARNING):
        await adapter.on_event(ev, BOT)
    assert not a.ask_fn.calls and "without a userId" in caplog.text
