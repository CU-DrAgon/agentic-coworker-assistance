# ruff: noqa: F811, F401
"""create_video_ad in chat: when the tool is offered, the background job (records, monthly
budget, progress message, Cancel button, delivery) and the restart sweep. The pipeline itself is
replaced by a fake `run_video_ad`; nothing touches the network or ffmpeg."""

import asyncio
import inspect
import json
import threading
from types import SimpleNamespace

import discord
import pytest
from conftest import GUILD
from promo_helpers import png_bytes
from test_coordinator import world  # fixture: world.g, .pub, .secret, .staff, .boss
from test_modes import Chan, TMsg, bot  # fixture

from salid import videoad, videoad_bot
from salid.guild_settings import FEATURES, Features, GuildSettingsUpdate
from salid.i18n import t
from salid.qa import FileInput

STARTED = {
    "status": "started",
    "note": "Reply in one short line that the video is being made and will be posted here.",
}
DOC = """Make a finished vertical video ad (15, 20 or 30 seconds) for the user's product or promotion:
Salid writes a storyboard, makes the scenes (or uses the photos the user attached), adds Thai
captions and Thai narration, and posts the MP4 here when it is ready (a few minutes). Use for
"ทำวิดีโอโฆษณา…", "ทำคลิปโปรโมท…", "make a video ad…". Pass the user's brief as is.

Args:
    brief: what to advertise and the offer, in the user's words.
    seconds: 15, 20 (default) or 30.
    use_photos: use the photos attached to this message as scenes (default true).
    hero_shots: 0-2 generated motion shots (only when enabled for this server).
    voice: add Thai narration (default true).
    music: add background music (only when enabled for this server)."""
RECORD_KEYS = {
    "guild_id", "channel_id", "requester_id", "state", "created_at", "updated_at", "cost_usd",
    "reason",
}  # fmt: skip
FILE = "ad-20261014-1200.mp4"


# ------------------------------------------------------------------ fakes
class RMsg(TMsg):
    """A request message that keeps the keyword arguments of every reply."""

    async def reply(self, text, **kw):
        self.reply_kw = [*getattr(self, "reply_kw", []), kw]
        return await super().reply(text, **kw)


class Posted:
    def __init__(self, text, kw):
        self.text, self.kw, self.edits = text, kw, []

    async def edit(self, content=None, **kw):
        self.edits.append((content, kw))
        return self


def capture_posts(channel):
    """`channel.send` returns an editable message; `channel.posted` keeps them."""
    channel.posted = []

    async def send(text=None, **kw):
        channel.sent.append(text)
        channel.posted.append(Posted(text, kw))
        return channel.posted[-1]

    channel.send = send


class Click:
    def __init__(self, guild, user, channel, client):
        self.guild, self.user, self.channel, self.client = guild, user, channel, client
        self.locale, self.sent, self.edited = "th", [], []
        self.response = SimpleNamespace(send_message=self._send, edit_message=self._edit)

    async def _send(self, text, ephemeral=False):
        self.sent.append((text, ephemeral))

    async def _edit(self, **kw):
        self.edited.append(kw)


def board():
    return videoad.Storyboard(
        title="ลาเต้เย็น",
        scenes=[
            videoad.Scene(
                image_prompt_en="a latte",
                caption_th=f"ฉาก {i}",
                voice_th="หอม",
                seconds=5,
                source="ai",
            )
            for i in range(1, 5)
        ],
    )


@pytest.fixture
def runner(monkeypatch):
    """A fake `run_video_ad`: takes the month's ad like the real one, reports each step (after
    waiting for a cancel when `wait_cancel`), then raises `error` or writes a tiny MP4. `spent`
    is the cost estimate it reports before its first step."""
    r = SimpleNamespace(calls=[], error=None, wait_cancel=False, spent=0.0)

    def fake(
        req, *, client, settings, say, workdir, budget_mib, handle, usage, month, media_client=None
    ):
        r.calls.append(SimpleNamespace(req=req, workdir=workdir, budget=budget_mib, month=month,
                                       handle=handle, say=say, client=client,
                                       media_client=media_client))  # fmt: skip
        if not usage.take(req.guild_id, month, "ads", 1, settings.video_ads_per_month):
            raise videoad.MonthLimit("month limit")
        handle.cost_usd = r.spent
        for step in ("storyboard", "stills", "audio", "render"):
            if r.wait_cancel:
                handle.cancelled.wait(5)
            videoad.check(handle)
            handle.progress(step, "")
        if r.error is not None:
            raise r.error
        out = workdir / FILE
        out.write_bytes(b"mp4")
        return videoad.VideoResult(out, board(), 0.25, {"storyboard": 1})

    monkeypatch.setattr(videoad_bot, "run_video_ad", fake)
    return r


@pytest.fixture
def ready(bot, monkeypatch):
    """ffmpeg and Thai shaping available, quick timers; the flag is still off."""
    monkeypatch.setattr(videoad_bot, "ffmpeg_ready", lambda: True)
    monkeypatch.setattr(videoad_bot.cards, "thai_ready", lambda: True)
    bot.media_clients = []

    def media_client(settings):  # never a real client in tests
        bot.media_clients.append(SimpleNamespace(settings=settings))
        return bot.media_clients[-1]

    monkeypatch.setattr(videoad_bot, "make_media_client", media_client)

    async def sleep(s):
        bot.clock.advance(s)

    bot._sleep = sleep
    return bot


def enable(bot, on=True):
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={"video_ads": on}), "x")
    bot.team._cache.clear()


@pytest.fixture
def on(ready):
    enable(ready)
    return ready


def names(tools):
    return {f.__name__ for f in tools}


def tool_for(bot, world, msg=None, files=(), who=None):
    who = who or world.staff
    msg = msg or RMsg(world.g, world.pub, who, "ทำวิดีโอโฆษณาลาเต้")
    [tool] = bot._video_tools(msg, world.g, who, list(files))
    return tool, msg


async def settle(bot):
    """Let the tool's launch callback run, then wait for every video job to end."""
    await asyncio.sleep(0)
    for _ in range(100):
        tasks = list(bot._video_tasks)
        if not tasks:
            return
        await asyncio.wait(tasks, timeout=10)
        await asyncio.sleep(0)
    raise AssertionError("video jobs did not finish")


async def until(cond):
    for _ in range(500):
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


def records(bot):
    return bot.docs.find("salid_video_jobs", GUILD)


# ------------------------------------------------------------------ when the tool is offered
def test_video_ads_flag_defaults_off():
    assert "video_ads" in FEATURES and Features().video_ads is False


async def test_not_offered_by_default_nor_in_voice_or_interjections(ready, world):
    bot, m = ready, world.staff
    msg = RMsg(world.g, world.pub, m, "ทำวิดีโอโฆษณา")
    kw = dict(setting="th", lang="th", question="x", memory=None, profile=None)
    assert bot._video_tools(msg, world.g, m, []) == []  # flag off by default
    plain = bot._agent_kwargs(msg, world.g, m, False, files=[], **kw)["extra_tools"]
    assert "create_video_ad" not in names(plain)
    enable(bot)
    assert bot._video_tools(msg, world.g, m, None) == []  # voice
    inter = bot._agent_kwargs(msg, world.g, m, False, files=[], interjection=True, **kw)
    assert "create_video_ad" not in names(inter.get("extra_tools", []))
    chat = bot._agent_kwargs(msg, world.g, m, False, files=[], **kw)["extra_tools"]
    assert "create_video_ad" in names(chat)
    order = [f.__name__ for f in chat]
    assert order.index("create_video_ad") == order.index("create_image") + 1


async def test_offered_when_an_admin_turns_it_on(ready, world):
    ready.guilds_store.save_settings(
        GUILD, GuildSettingsUpdate(features={"video_ads": True}), "admin@example.com"
    )
    ready.team._cache.clear()
    tool, _ = tool_for(ready, world)
    assert tool.__name__ == "create_video_ad" and inspect.getdoc(tool) == DOC
    params = inspect.signature(tool).parameters
    assert {k: p.default for k, p in params.items()} == dict(
        brief="", seconds=20, use_photos=True, hero_shots=0, voice=True, music=False
    )


async def test_not_offered_without_ffmpeg_or_thai_shaping(on, world, monkeypatch):
    msg = RMsg(world.g, world.pub, world.staff, "x")
    monkeypatch.setattr(videoad_bot, "ffmpeg_ready", lambda: False)
    assert on._video_tools(msg, world.g, world.staff, []) == []
    monkeypatch.setattr(videoad_bot, "ffmpeg_ready", lambda: True)
    monkeypatch.setattr(videoad_bot.cards, "thai_ready", lambda: False)
    assert on._video_tools(msg, world.g, world.staff, []) == []


async def test_tool_list_never_raises(on, world, monkeypatch):
    monkeypatch.setattr(on.team, "enabled", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    assert (
        on._video_tools(RMsg(world.g, world.pub, world.staff, "x"), world.g, world.staff, []) == []
    )


# ------------------------------------------------------------------ the job
async def test_a_started_job_posts_progress_and_delivers_the_mp4(on, world, runner):
    capture_posts(world.pub)
    states = []
    put = on.docs.put
    on.docs.put = lambda kind, gid, key, doc: (
        states.append(doc["state"]) if kind == "salid_video_jobs" else None,
        put(kind, gid, key, doc),
    )
    photo = FileInput(0, "p.png", "image/png", data=png_bytes((900, 900)))
    gif = FileInput(1, "a.gif", "image/gif", data=b"GIF89a")
    tool, msg = tool_for(on, world, files=[photo, gif])
    assert tool(brief="ลาเต้เย็น ลด 20%", seconds=15) == STARTED
    await settle(on)
    [call] = runner.calls
    req = call.req
    assert (req.guild_id, req.user_id, req.seconds, req.lang) == (GUILD, world.staff.id, 15, "th")
    assert req.brief == "ลาเต้เย็น ลด 20%" and req.photos == [photo.data]
    assert req.hero_shots == 0 and req.music is False and req.voice is True
    assert call.month == "2026-10" and call.budget == 9.0  # no known limit: 10 MiB - 1
    assert not call.workdir.exists()  # the temp dir is always removed
    [posted] = world.pub.posted
    assert posted.text == t("video_started", "th")
    assert isinstance(posted.kw["view"], videoad_bot.CancelView)
    assert posted.edits[-1] == (t("video_done", "th"), {"view": None})
    [reply], [kw] = msg.replies, msg.reply_kw
    assert reply.startswith(f"<@{world.staff.id}>") and t("video_done", "th") in reply
    assert "1. ฉาก 1" in reply and "4. ฉาก 4" in reply and len(reply) < 1900
    assert reply.rstrip().endswith("#สร้างด้วยAI")
    assert [f.filename for f in kw["files"]] == [FILE]
    [rec] = records(on)
    assert rec["state"] == "done" and rec["cost_usd"] == 0.25 and rec["reason"] == ""
    assert rec["requester_id"] == world.staff.id and rec["channel_id"] == world.pub.id
    assert set(rec) <= RECORD_KEYS and "ลาเต้" not in json.dumps(rec, ensure_ascii=False)
    assert states == ["queued", "storyboard", "stills", "audio", "render", "done"]
    assert on.docs.get("salid_media_usage", GUILD, "2026-10") == {"ads": 1}
    assert on._video_jobs == {}


async def test_photos_can_be_left_out(on, world, runner):
    photo = FileInput(0, "p.png", "image/png", data=png_bytes((900, 900)))
    tool, _ = tool_for(on, world, files=[photo])
    assert tool(brief="x", use_photos=False)["status"] == "started"
    await settle(on)
    assert runner.calls[0].req.photos == []


async def test_a_second_request_in_the_same_guild_is_busy(on, world, runner):
    tool, _ = tool_for(on, world)
    assert tool(brief="a") == STARTED
    second = tool(brief="b")
    assert second["status"] == "busy" and second["say"] == t("video_busy", "th")
    await settle(on)
    assert len(runner.calls) == 1
    assert tool(brief="c") == STARTED  # the slot is free again once the job ended
    await settle(on)
    assert len(runner.calls) == 2


async def test_the_bot_runs_at_most_video_max_jobs_at_once(on, world, runner):
    assert on.s.video_max_jobs == 2
    on._video_jobs[111] = SimpleNamespace(job_id="x", guild_id=111)
    on._video_jobs[222] = SimpleNamespace(job_id="y", guild_id=222)
    tool, _ = tool_for(on, world)
    assert tool(brief="a")["status"] == "busy"
    await settle(on)
    assert runner.calls == [] and set(on._video_jobs) == {111, 222}
    del on._video_jobs[222]
    assert tool(brief="a") == STARTED
    await settle(on)
    assert len(runner.calls) == 1 and set(on._video_jobs) == {111}


async def test_the_month_limit_refuses_before_anything_starts(on, world, runner):
    on.docs.put("salid_media_usage", GUILD, "2026-10", {"ads": 10})
    tool, _ = tool_for(on, world)
    out = tool(brief="a")
    assert out["status"] == "limit" and out["say"] == t("video_month_limit", "th", n=10)
    await settle(on)
    assert runner.calls == [] and on._video_jobs == {} and records(on) == []
    assert on.docs.get("salid_media_usage", GUILD, "2026-10") == {"ads": 10}


async def test_the_month_peek_reads_outside_the_slot_lock(on, world, runner, monkeypatch):
    held = []
    used = on.video_usage.used
    monkeypatch.setattr(
        on.video_usage, "used", lambda *a: held.append(on._video_lock.locked()) or used(*a)
    )
    tool, _ = tool_for(on, world)
    assert tool(brief="a") == STARTED
    await settle(on)
    assert held == [False]  # a DocStore read never blocks other guilds' tool calls


async def test_a_month_limit_hit_inside_the_job_is_reported(on, world, runner):
    capture_posts(world.pub)
    on.docs.put("salid_media_usage", GUILD, "2026-10", {"ads": 9})
    tool, _ = tool_for(on, world)
    assert tool(brief="a") == STARTED
    on.docs.put("salid_media_usage", GUILD, "2026-10", {"ads": 10})  # another job took the last
    await settle(on)
    [rec] = records(on)
    assert rec["state"] == "failed" and rec["reason"] == "month_limit"
    assert world.pub.posted[0].edits[-1] == (t("video_month_limit", "th", n=10), {"view": None})


@pytest.mark.parametrize(
    "kw", [dict(seconds=25), dict(seconds=0), dict(seconds="x"), dict(hero_shots=3),
           dict(hero_shots=-1)]
)  # fmt: skip
async def test_bad_arguments_are_errors(on, world, runner, kw):
    tool, _ = tool_for(on, world)
    assert tool(brief="a", **kw)["status"] == "error"
    await settle(on)
    assert runner.calls == [] and on._video_jobs == {}


async def test_hero_shots_and_music_are_ignored_while_their_models_are_empty(on, world, runner):
    tool, _ = tool_for(on, world)
    out = tool(brief="a", hero_shots=2, music=True)
    assert out["status"] == "started" and out["ignored"] == ["hero_shots", "music"]
    await settle(on)
    assert runner.calls[0].req.hero_shots == 0
    assert runner.calls[0].req.music is True  # the user's request; the runner checks the model
    on.s.video_hero_model = "omni-test"
    on.s.video_music_model = "music-test"
    out = tool(brief="a", hero_shots=2, music=True)
    assert out == STARTED
    await settle(on)
    assert runner.calls[1].req.hero_shots == 2 and runner.calls[1].req.music is True


async def test_paid_media_calls_get_their_own_client_only_when_enabled(on, world, runner):
    tool, _ = tool_for(on, world)
    tool(brief="a")
    await settle(on)
    assert runner.calls[0].media_client is None and on.media_clients == []  # add-ons off
    assert runner.calls[0].client is on.llm
    on.s.video_hero_model = "omni-test"
    for _ in range(2):
        tool(brief="b", hero_shots=1)
        await settle(on)
    [made] = on.media_clients  # made once, then reused
    assert made.settings is on.s
    assert runner.calls[1].media_client is made and runner.calls[2].media_client is made
    assert runner.calls[1].client is on.llm  # polls and everything else stay on the main client


async def test_the_budget_follows_the_message_server_and_dms_get_the_default(on, world, runner):
    world.g.filesize_limit = 100 * 2**20  # boost tier 3
    tool, _ = tool_for(on, world)
    tool(brief="a")
    await settle(on)
    dm = RMsg(None, Chan(901), world.staff, "ทำวิดีโอโฆษณา")
    tool, _ = tool_for(on, world, msg=dm)  # about this server, asked in a DM
    tool(brief="a")
    await settle(on)
    assert [c.budget for c in runner.calls] == [19.0, 9.0]


@pytest.mark.parametrize(
    "error, reason",
    [
        (videoad.StoryboardError("secret A"), "storyboard"),
        (RuntimeError("ffmpeg failed: secret B"), "render"),
        (ValueError("secret C"), "error"),
    ],
)
async def test_failures_tell_the_requester_without_details(
    on, world, runner, error, reason, caplog
):
    caplog.set_level("INFO", logger="salid.videoad_bot")
    capture_posts(world.pub)
    runner.error = error
    tool, msg = tool_for(on, world)
    tool(brief="a")
    await settle(on)
    [rec] = records(on)
    assert rec["state"] == "failed" and rec["reason"] == reason
    text = t("video_failed", "th", reason=t(f"video_reason_{reason}", "th"))
    [posted] = world.pub.posted
    assert posted.edits[-1] == (text, {"view": None})
    shown = [*world.pub.sent, *msg.replies, *(e[0] for e in posted.edits)]
    assert all("secret" not in (x or "") for x in shown)
    assert not runner.calls[0].workdir.exists()
    logged = [r.getMessage() for r in caplog.records if r.name == "salid.videoad_bot"]
    assert logged and not any(m.startswith("video guild=") for m in logged)  # step prefix only


async def test_without_a_progress_message_the_outcome_is_posted(on, world, runner):
    runner.error = videoad.StoryboardError("x")
    tool, _ = tool_for(on, world)
    tool(brief="a")
    await settle(on)
    text = t("video_failed", "th", reason=t("video_reason_storyboard", "th"))
    assert world.pub.sent[0] == t("video_started", "th") and text in world.pub.sent[-1]
    assert f"<@{world.staff.id}>" in world.pub.sent[-1]


async def test_a_crash_outside_the_pipeline_still_ends_the_job(on, world, runner, monkeypatch):
    async def broken(*a, **kw):
        raise ValueError("secret D")

    monkeypatch.setattr(on, "_video_deliver", broken)
    capture_posts(world.pub)
    tool, _ = tool_for(on, world)
    tool(brief="a")
    await settle(on)
    [rec] = records(on)
    assert rec["state"] == "failed" and rec["reason"] == "error" and on._video_jobs == {}
    text = t("video_failed", "th", reason=t("video_reason_error", "th"))
    assert world.pub.posted[0].edits[-1] == (text, {"view": None})


async def test_delivery_falls_back_to_the_channel_when_the_request_is_gone(on, world, runner):
    class Gone(RMsg):
        async def reply(self, text, **kw):
            raise discord.NotFound(SimpleNamespace(status=404, reason="x"), "gone")

    capture_posts(world.pub)
    tool, _ = tool_for(on, world, msg=Gone(world.g, world.pub, world.staff, "x"))
    tool(brief="a")
    await settle(on)
    last = world.pub.posted[-1]
    assert last.kw["file"].filename == FILE and last.text.rstrip().endswith("#สร้างด้วยAI")
    assert records(on)[0]["state"] == "done"


async def test_an_upload_refused_still_sends_the_text_with_a_note(on, world, runner):
    class TooBig(RMsg):
        async def reply(self, text, **kw):
            if kw.get("files"):
                raise discord.HTTPException(SimpleNamespace(status=413, reason="x"), "too big")
            return await super().reply(text, **kw)

    tool, msg = tool_for(on, world, msg=TooBig(world.g, world.pub, world.staff, "x"))
    tool(brief="a")
    await settle(on)
    assert msg.replies[-1].endswith(t("attach_failed", "th"))


async def test_progress_edits_at_most_every_5_s_with_the_latest_step(ready):
    bot, sleeps = ready, []

    async def sleep(s):
        sleeps.append(s)
        bot.clock.advance(s)

    bot._sleep = sleep
    job = videoad_bot.VideoJob(
        job_id="j", guild_id=GUILD, channel_id=10, requester_id=1, lang="th", created_at="x"
    )
    job.msg, job.last_edit = Posted("start", {}), bot.clock()
    bot._video_progress(job, "storyboard")
    bot._video_progress(job, "stills")  # same window: only the label changes
    await job.edit_task
    assert sleeps == [5.0] and job.msg.edits == [(t("video_step_stills", "th"), {})]
    bot.clock.advance(7)
    bot._video_progress(job, "audio")  # more than 5 s since the last edit: at once
    await job.edit_task
    assert sleeps == [5.0] and job.msg.edits[-1] == (t("video_step_audio", "th"), {})
    bot.clock.advance(1)
    bot._video_progress(job, "render")
    await job.edit_task
    assert sleeps == [5.0, 4.0] and len(job.msg.edits) == 3


# ------------------------------------------------------------------ cancel
async def start_waiting_job(bot, world, runner):
    runner.wait_cancel = True
    capture_posts(world.pub)
    tool, msg = tool_for(bot, world)
    assert tool(brief="a") == STARTED
    await until(lambda: world.pub.posted and runner.calls)
    return bot._video_jobs[GUILD], world.pub.posted[0].kw["view"], msg


async def test_cancel_by_someone_else_is_refused_and_the_job_keeps_running(on, world, runner):
    job, view, _ = await start_waiting_job(on, world, runner)
    [button] = view.children
    assert button.label == t("video_cancel_button", "th")
    assert view.timeout is None  # a long job keeps its button; the job's end stops the view
    it = Click(world.g, world.peer, world.pub, on)
    await button.callback(it)
    assert it.sent == [(t("video_not_yours", "th"), True)] and not job.handle.cancelled.is_set()
    assert on._video_jobs[GUILD] is job
    await button.callback(Click(world.g, world.staff, world.pub, on))  # the requester
    assert job.handle.cancelled.is_set()
    await settle(on)
    [rec] = records(on)
    assert rec["state"] == "cancelled"
    assert world.pub.posted[0].edits[-1] == (t("video_cancelled", "th"), {"view": None})


async def test_an_admin_may_cancel_any_job(on, world, runner):
    job, view, msg = await start_waiting_job(on, world, runner)
    it = Click(world.g, world.admin, world.pub, on)
    await view.children[0].callback(it)
    assert job.handle.cancelled.is_set() and it.sent == []
    await settle(on)
    assert records(on)[0]["state"] == "cancelled" and msg.replies == []
    assert on._video_jobs == {} and view.is_finished()


async def test_a_cancelled_job_records_what_was_already_spent(on, world, runner):
    runner.spent = 2 * videoad.STILL_USD  # stills billed before the cancel
    job, view, _ = await start_waiting_job(on, world, runner)
    await view.children[0].callback(Click(world.g, world.staff, world.pub, on))
    await settle(on)
    [rec] = records(on)
    assert rec["state"] == "cancelled" and rec["cost_usd"] == round(2 * videoad.STILL_USD, 4)


async def test_shutdown_stops_the_thread_before_its_temp_dir_goes(on, world, runner, monkeypatch):
    seen = []
    real = videoad_bot.tempfile.TemporaryDirectory

    class Dir(real):
        def __exit__(self, *exc):  # what the worker thread was told when the dir was removed
            seen.append(runner.calls[0].handle.cancelled.is_set())
            return super().__exit__(*exc)

    monkeypatch.setattr(videoad_bot.tempfile, "TemporaryDirectory", Dir)
    job, _, _ = await start_waiting_job(on, world, runner)
    job.task.cancel()  # the bot shuts down mid-job
    with pytest.raises(asyncio.CancelledError):
        await job.task
    assert seen == [True]
    assert records(on)[0]["state"] == "queued"  # left for the next start's sweep


# ------------------------------------------------------------------ restart
async def test_restart_marks_running_jobs_failed_and_tells_the_requester(
    bot, world
):  # Review Focus 5
    bot.docs.put(
        "salid_video_jobs",
        GUILD,
        "j1",
        {
            "guild_id": GUILD,
            "channel_id": world.pub.id,
            "requester_id": world.staff.id,
            "state": "stills",
        },
    )
    bot.docs.put("salid_media_usage", GUILD, "2026-10", {"ads": 1})
    await bot._sweep_video_jobs()
    assert bot.docs.get("salid_video_jobs", GUILD, "j1")["state"] == "failed"
    assert bot.docs.get("salid_video_jobs", GUILD, "j1")["reason"] == "restarted"
    assert bot.docs.get("salid_media_usage", GUILD, "2026-10") == {"ads": 1}
    assert any("video_restarted" in m or t("video_restarted", "th") in m for m in world.pub.sent)
    assert any(f"<@{world.staff.id}>" in m for m in world.pub.sent)


async def test_the_sweep_runs_once_and_skips_live_and_finished_jobs(bot, world):
    def job(state, channel=world.pub.id):
        return {"guild_id": GUILD, "channel_id": channel, "requester_id": world.staff.id,
                "state": state}  # fmt: skip

    bot.docs.put("salid_video_jobs", GUILD, "old", job("done"))
    bot.docs.put("salid_video_jobs", GUILD, "live", job("render"))
    bot.docs.put("salid_video_jobs", GUILD, "lost", job("queued", channel=4242))  # a DM, say
    bot._video_jobs[GUILD] = SimpleNamespace(job_id="live", guild_id=GUILD)
    await bot._sweep_video_jobs()
    get = lambda key: bot.docs.get("salid_video_jobs", GUILD, key)  # noqa: E731
    assert (get("old")["state"], get("live")["state"]) == ("done", "render")
    assert get("lost")["state"] == "failed" and world.pub.sent == []
    [(text, _)] = world.staff.dms  # no such channel: the requester gets a DM
    assert t("video_restarted", "th") in text
    bot.docs.put("salid_video_jobs", GUILD, "later", job("stills"))
    await bot._sweep_video_jobs()  # on_ready again after a reconnect: no second sweep
    assert get("later")["state"] == "stills"


async def test_the_sweep_never_fails_a_job_launched_while_it_runs(on, world, runner):
    runner.wait_cancel = True
    gate, real = threading.Event(), on.docs.find_items

    def slow_find(*a, **kw):  # the sweep's read is still pending when the request comes in
        gate.wait(5)
        return real(*a, **kw)

    on.docs.find_items = slow_find
    sweep = asyncio.create_task(on._sweep_video_jobs())
    await asyncio.sleep(0)
    tool, _ = tool_for(on, world)
    assert tool(brief="a") == STARTED
    job_id = on._video_jobs[GUILD].job_id
    rec = lambda: on.docs.get("salid_video_jobs", GUILD, job_id)  # noqa: E731
    await until(rec)  # the new job's record ("queued") exists; the sweep's read still waits
    gate.set()
    await sweep
    assert rec()["state"] == "queued" and rec()["reason"] == ""
    assert not any(t("video_restarted", "th") in (m or "") for m in world.pub.sent)
    on._video_jobs[GUILD].handle.cancelled.set()
    await settle(on)
    assert records(on)[0]["state"] == "cancelled"


async def test_the_sweep_leaves_a_job_that_finished_meanwhile(bot, world, monkeypatch):
    doc = {"guild_id": GUILD, "channel_id": world.pub.id, "requester_id": world.staff.id,
           "state": "render"}  # fmt: skip
    bot.docs.put("salid_video_jobs", GUILD, "j", doc)
    real = bot.docs.find_items

    def find_then_finish(*a, **kw):  # the job writes "done" right after the sweep's read
        items = real(*a, **kw)
        bot.docs.put("salid_video_jobs", GUILD, "j", {**doc, "state": "done"})
        return items

    bot.docs.find_items = find_then_finish
    await bot._sweep_video_jobs()
    assert bot.docs.get("salid_video_jobs", GUILD, "j")["state"] == "done"
    assert world.pub.sent == []


async def test_on_ready_sweeps_left_over_jobs(bot, world, monkeypatch):
    called = []

    async def sweep():
        called.append(1)

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(bot, "_sweep_video_jobs", sweep)
    monkeypatch.setattr(bot, "_sync_guild", noop)
    monkeypatch.setattr(bot, "_pc_backfill_start", lambda g: None)
    monkeypatch.setattr(bot, "_start_import", lambda: None)
    bot.meetings = SimpleNamespace(resume_all=noop, scan_all=lambda: None)
    await bot.on_ready()
    assert called == [1]
