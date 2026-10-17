"""Discord glue of `create_video_ad` (the pipeline is `videoad.py`).

`VideoAdMixin` gives `SecretaryBot`:
- `_video_tools`: the `create_video_ad` agent tool (appended to `extra_tools` by `_agent_kwargs`)
  for chat answers (voice never passes files, interjections never get it), only when the guild's
  `video_ads` flag is on (default off), ffmpeg is installed and Thai text can be shaped. Guild,
  channel, author and the attached photos are bound here, never model input;
- the background job: the tool validates, peeks at the month's ads and reserves the guild's slot
  (one job per guild, `SALID_VIDEO_MAX_JOBS` across the bot) before anything starts, then
  `run_video_ad` runs in a worker thread in a temp dir that is always removed. One progress
  message in the asker's channel, edited at most every 5 s with the current step, carries a
  Cancel button for the requester or an admin; the MP4 comes back as a reply to the request with
  the scene list and a suggested caption;
- job records in DocStore kind `salid_video_jobs` (written on every state change; never the
  brief, the storyboard or media) and `_sweep_video_jobs`: on the first `on_ready`, jobs a restart
  cut off are marked failed and their requester is told to ask again (usage is not refunded).

Jobs and their slots live in this process (one bot process, like `memory_index` and `recent`).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import discord

from . import cards, speech
from .i18n import CATALOG, t
from .promo import PHOTO_MIMES
from .videoad import (
    LENGTHS,
    Cancelled,
    MediaUsage,
    MonthLimit,
    StoryboardError,
    VideoHandle,
    VideoRequest,
    budget_mib,
    run_video_ad,
)

logger = logging.getLogger(__name__)

JOBS = "salid_video_jobs"  # DocStore kind, key = job id
FINAL = frozenset({"done", "failed", "cancelled"})
PROGRESS_EVERY_S = 5.0  # the progress message is edited at most this often
CANCEL_TIMEOUT_S = 900
MAX_HERO_SHOTS = 2
AI_HASHTAG = "#สร้างด้วยAI"
MAX_REPLY = 1900
STARTED_NOTE = "Reply in one short line that the video is being made and will be posted here."
_NOT_ENABLED = {"hero_shots": "generated motion shots", "music": "background music"}

_warned: set[str] = set()  # missing ffmpeg / libfribidi is logged once per process


def ffmpeg_ready() -> bool:
    return shutil.which("ffmpeg") is not None


def _warn_once(what: str, text: str) -> None:
    if what not in _warned:
        _warned.add(what)
        logger.error(text)


@dataclass
class VideoJob:
    job_id: str
    guild_id: int
    channel_id: int
    requester_id: int
    lang: str
    created_at: str
    handle: VideoHandle = field(default_factory=VideoHandle)
    task: asyncio.Task | None = None
    msg: Any = None  # the progress message (None: it could not be posted)
    view: discord.ui.View | None = None
    label: str = ""  # the step label the next progress edit shows
    last_edit: float = float("-inf")
    edit_task: asyncio.Task | None = None
    final: bool = False


class CancelView(discord.ui.View):
    """The progress message's one button: cancels the job (requester or admin only)."""

    def __init__(self, job_id: str, lang: str = "th"):
        super().__init__(timeout=CANCEL_TIMEOUT_S)
        self.job_id = job_id
        button = discord.ui.Button(
            label=t("video_cancel_button", lang), style=discord.ButtonStyle.danger
        )
        button.callback = self._pressed
        self.add_item(button)

    async def _pressed(self, interaction: discord.Interaction) -> None:
        await interaction.client._video_cancel_pressed(interaction, self.job_id)


class VideoAdMixin:
    """Needs `self.s`, `self.llm`, `self.team`, `self.docs`, `self._clock`, `self._sleep`, `_lang`,
    `_interaction_lang`, `_may_configure` and `_reply_with_uploads`. Call `_init_video_ads()` once
    after `self.docs` exists."""

    def _init_video_ads(self) -> None:
        self._video_jobs: dict[int, Any] = {}  # guild -> its job, reserved before the launch
        self._video_tasks: set[asyncio.Task] = set()
        self._video_lock = threading.Lock()  # agent tools may run in worker threads
        self._video_swept = False
        self.video_usage = MediaUsage(self.docs)

    # ------------------------------------------------------------- agent tool
    def _video_tools(self, message, guild, asker, files) -> list:
        """`create_video_ad` for this message, or [] (voice, flag off, no ffmpeg, no Thai
        shaping). Never raises."""
        try:
            if files is None or not self.team.enabled(guild.id, "video_ads"):
                return []
            if not ffmpeg_ready():
                _warn_once("ffmpeg", "ffmpeg missing: create_video_ad off")
                return []
            if not cards.thai_ready():
                _warn_once("raqm", "libfribidi missing: create_video_ad off")
                return []
            photos = [f.data for f in files if f.mime in PHOTO_MIMES and f.data and not f.skip]
            loop = asyncio.get_running_loop()
            return [self._video_tool(message, guild, asker, photos, self._lang(message), loop)]
        except Exception:
            logger.warning("video tool unavailable", exc_info=True)
            return []

    def _video_tool(self, message, guild, asker, photos: list[bytes], lang: str, loop):
        def create_video_ad(
            brief: str = "",
            seconds: int = 20,
            use_photos: bool = True,
            hero_shots: int = 0,
            voice: bool = True,
            music: bool = False,
        ) -> dict:
            """Make a finished vertical video ad (15, 20 or 30 seconds) for the user's product or promotion:
            Salid writes a storyboard, makes the scenes (or uses the photos the user attached), adds Thai
            captions and Thai narration, and posts the MP4 here when it is ready (a few minutes). Use for
            "ทำวิดีโอโฆษณา…", "ทำคลิปโปรโมท…", "make a video ad…". Pass the user's brief as is.

            Args:
                brief: what to advertise and the offer, in the user's words.
                seconds: 15, 20 (default) or 30.
                use_photos: use the photos attached to this message as scenes (default true).
                hero_shots: 0-2 generated motion shots (only when enabled for this server).
                voice: add Thai narration (default true).
                music: add background music (only when enabled for this server).
            """
            try:
                return self._start_video(
                    message, guild, asker, photos, lang, loop,
                    brief=brief, seconds=seconds, use_photos=use_photos, hero_shots=hero_shots,
                    voice=voice, music=music,
                )  # fmt: skip
            except Exception:
                logger.warning("create_video_ad could not start", exc_info=True)
                return {"status": "error", "error": "the video could not be started"}

        return create_video_ad

    def _start_video(
        self, message, guild, asker, photos, lang, loop, *, brief, seconds, use_photos,
        hero_shots, voice, music,
    ) -> dict:  # fmt: skip
        """Validate, peek at the month's ads, reserve the guild's slot and schedule the job."""
        try:
            secs, heroes = float(seconds), float(hero_shots)
        except (TypeError, ValueError):
            secs, heroes = -1.0, -1.0
        if secs not in LENGTHS:
            return {"status": "error", "error": "seconds must be 15, 20 or 30"}
        if not heroes.is_integer() or not 0 <= heroes <= MAX_HERO_SHOTS:
            return {"status": "error", "error": f"hero_shots must be 0 to {MAX_HERO_SHOTS}"}
        ignored = []
        if heroes and not self.s.video_hero_model:
            ignored.append("hero_shots")
            heroes = 0
        if music and not self.s.video_music_model:
            ignored.append("music")
        gid, now = guild.id, self.s.now()
        month, cap = now.strftime("%Y-%m"), self.s.video_ads_per_month
        job = VideoJob(
            job_id=uuid.uuid4().hex[:12],
            guild_id=gid,
            channel_id=message.channel.id,
            requester_id=asker.id,
            lang=lang,
            created_at=now.isoformat(timespec="seconds"),
        )
        req = VideoRequest(
            guild_id=gid,
            user_id=asker.id,
            brief=str(brief or ""),
            seconds=int(secs),
            photos=list(photos) if use_photos else [],
            hero_shots=int(heroes),
            voice=bool(voice),
            music=None,  # no music model yet
            lang=lang,
        )
        with self._video_lock:  # check and reserve together: two quick asks cannot both start
            if gid in self._video_jobs or len(self._video_jobs) >= self.s.video_max_jobs:
                return {"status": "busy", "say": t("video_busy", lang)}
            if self.video_usage.used(gid, month, "ads") + 1 > cap:
                return {"status": "limit", "say": t("video_month_limit", lang, n=cap)}
            self._video_jobs[gid] = job
        try:  # nothing between the reservation and here can raise
            loop.call_soon_threadsafe(self._launch_video, job, message, req)
        except RuntimeError:  # the event loop is closing
            self._video_release(job)
            return {"status": "error", "error": "the video could not be started"}
        out = {"status": "started", "note": STARTED_NOTE}
        if ignored:
            what = " and ".join(_NOT_ENABLED[x] for x in ignored)
            out["ignored"] = ignored
            out["note"] += f" Also say that {what} {'is' if len(ignored) == 1 else 'are'} not"
            out["note"] += " enabled on this server."
        return out

    # ------------------------------------------------------------- the background job
    def _launch_video(self, job: VideoJob, message, req: VideoRequest) -> None:
        try:
            task = asyncio.create_task(self._run_video(job, message, req))
        except Exception:
            logger.warning("video job could not start", exc_info=True)
            self._video_release(job)
            return
        job.task = task
        self._video_tasks.add(task)
        task.add_done_callback(self._video_tasks.discard)
        task.add_done_callback(lambda _t, j=job: self._video_release(j))

    def _video_release(self, job: VideoJob) -> None:
        with self._video_lock:
            if self._video_jobs.get(job.guild_id) is job:
                del self._video_jobs[job.guild_id]

    async def _run_video(self, job: VideoJob, message, req: VideoRequest) -> None:
        """The job's task: every job ends done, failed (with a reason) or cancelled, even when
        something outside the pipeline breaks. A shutdown leaves the record for the sweep."""
        try:
            await self._video_steps(job, message, req)
        except asyncio.CancelledError:  # shutdown: the thread stops before its next step
            job.handle.cancelled.set()
            raise
        except Exception:
            logger.exception("video guild=%s job crashed", job.guild_id)
            if not job.final:
                lang = job.lang
                text = t("video_failed", lang, reason=t("video_reason_error", lang))
                with contextlib.suppress(Exception):
                    await self._video_finish(job, message, "failed", text, reason="error")

    async def _video_steps(self, job: VideoJob, message, req: VideoRequest) -> None:
        lang, loop = job.lang, asyncio.get_running_loop()
        await self._video_record(job, "queued")
        job.view = CancelView(job.job_id, lang)
        try:
            job.msg = await message.channel.send(t("video_started", lang), view=job.view)
        except discord.HTTPException:
            logger.warning("video progress message not posted", exc_info=True)
        job.last_edit = self._clock()

        def progress(step: str, detail: str) -> None:  # in the job's worker thread
            self._video_save(job, step)
            with contextlib.suppress(RuntimeError):  # the loop is closing
                loop.call_soon_threadsafe(self._video_progress, job, step)

        job.handle.progress = progress

        def say(text: str) -> bytes:
            return speech.say(self.s, self.llm, text, lang)

        state, reason, text = "done", "", ""
        with tempfile.TemporaryDirectory(prefix="salid-video-") as tmp:
            try:
                res = await asyncio.to_thread(
                    run_video_ad,
                    req,
                    client=self.llm,
                    settings=self.s,
                    say=say,
                    workdir=Path(tmp),
                    # the upload limit where the reply goes (a DM: discord.py's 10 MiB default)
                    budget_mib=budget_mib(getattr(message.guild, "filesize_limit", None)),
                    handle=job.handle,
                    usage=self.video_usage,
                    month=self.s.now().strftime("%Y-%m"),
                )
            except Cancelled:
                state, text = "cancelled", t("video_cancelled", lang)
            except MonthLimit:
                state, reason = "failed", "month_limit"
                text = t("video_month_limit", lang, n=self.s.video_ads_per_month)
            except StoryboardError as e:  # problem lines only, never the storyboard text
                logger.info("video guild=%s storyboard failed: %s", job.guild_id, e)
                state, reason = "failed", "storyboard"
            except RuntimeError as e:  # our ffmpeg / budget errors
                logger.warning("video guild=%s render failed: %s", job.guild_id, e)
                state, reason = "failed", "render"
            except Exception:
                logger.exception("video guild=%s pipeline crashed", job.guild_id)
                state, reason = "failed", "error"
            if state == "done":
                job.handle.cost_usd = res.cost_usd
                await self._video_deliver(job, message, res)
                text = t("video_done", lang)
        if reason and not text:
            text = t("video_failed", lang, reason=t(f"video_reason_{reason}", lang))
        await self._video_finish(job, message, state, text, reason=reason)

    def _video_progress(self, job: VideoJob, step: str) -> None:
        """A step started (on the loop): show its label, at most one edit per PROGRESS_EVERY_S
        (a step that comes sooner waits, and the latest label wins)."""
        key = f"video_step_{step}"
        if job.final or job.msg is None or key not in CATALOG:
            return
        job.label = t(key, job.lang)
        if job.edit_task is not None and not job.edit_task.done():
            return  # the pending edit will show this label
        wait = max(0.0, job.last_edit + PROGRESS_EVERY_S - self._clock())
        job.edit_task = asyncio.create_task(self._video_edit_after(job, wait))

    async def _video_edit_after(self, job: VideoJob, wait: float) -> None:
        if wait > 0:
            await self._sleep(wait)
        if job.final:
            return
        job.last_edit = self._clock()
        try:
            await job.msg.edit(content=job.label)
        except discord.HTTPException:
            logger.info("video progress edit failed", exc_info=True)

    async def _video_finish(
        self, job: VideoJob, message, state: str, text: str, *, reason: str = ""
    ) -> None:
        """The job ended: final record, progress message edited to `text` without the button
        (posted with a mention when there is no progress message, unless the ad was delivered)."""
        job.final = True
        if job.edit_task is not None:
            job.edit_task.cancel()
        if job.view is not None:
            job.view.stop()
        await self._video_record(job, state, reason=reason)
        if job.msg is not None:
            try:
                await job.msg.edit(content=text, view=None)
                return
            except discord.HTTPException:
                logger.info("video final edit failed", exc_info=True)
        if state != "done":
            with contextlib.suppress(discord.HTTPException):
                await message.channel.send(
                    f"<@{job.requester_id}> {text}", allowed_mentions=self._video_mentions(job)
                )

    async def _video_deliver(self, job: VideoJob, message, res) -> None:
        """Reply to the request with the MP4, the scene list and a suggested caption. Upload
        refused: the text with a note. Request gone: the same in its channel."""
        text, allowed = self._video_reply(job, res), self._video_mentions(job)
        files = [discord.File(str(res.path), filename=res.path.name)]
        try:
            try:
                await self._reply_with_uploads(message, text, allowed, {"files": files})
                return
            except discord.HTTPException:
                logger.info("video reply failed; posting in the channel instead")
            files.append(discord.File(str(res.path), filename=res.path.name))
            try:
                await message.channel.send(text, file=files[-1], allowed_mentions=allowed)
                return
            except discord.HTTPException:
                logger.warning("could not post the video ad", exc_info=True)
            with contextlib.suppress(discord.HTTPException):
                note = t("attach_failed", job.lang)
                await message.channel.send(f"{text}\n{note}", allowed_mentions=allowed)
        finally:
            for f in files:
                f.close()

    @staticmethod
    def _video_mentions(job: VideoJob) -> discord.AllowedMentions:
        """Only the requester may be pinged (captions come from a model)."""
        return discord.AllowedMentions(
            everyone=False, roles=False, users=[discord.Object(job.requester_id)]
        )

    @staticmethod
    def _video_reply(job: VideoJob, res) -> str:
        lang, board = job.lang, res.storyboard
        scenes = [f"{i}. {s.caption_th}" for i, s in enumerate(board.scenes, start=1)]
        closing = board.scenes[-1].caption_th if board.scenes else ""
        words = [board.title] + ([closing] if closing != board.title else [])
        suggested = " ".join(w.strip() for w in [*words, AI_HASHTAG] if w.strip())
        head = f"<@{job.requester_id}> {t('video_done', lang)}"
        tail = f"{t('video_caption', lang)} {suggested}"
        text = f"{head}\n{t('video_scenes', lang)}\n" + "\n".join(scenes) + f"\n\n{tail}"
        return text if len(text) <= MAX_REPLY else f"{head}\n\n{tail}"

    def _video_save(self, job: VideoJob, state: str, *, reason: str = "") -> None:
        """Write the job's record (blocking). Never the brief, storyboard or media. Never
        raises."""
        try:
            self.docs.put(
                JOBS,
                job.guild_id,
                job.job_id,
                {
                    "guild_id": job.guild_id,
                    "channel_id": job.channel_id,
                    "requester_id": job.requester_id,
                    "state": state,
                    "created_at": job.created_at,
                    "updated_at": self.s.now().isoformat(timespec="seconds"),
                    "cost_usd": round(job.handle.cost_usd, 4),
                    "reason": reason,
                },
            )
        except Exception:
            logger.warning("video job record not saved", exc_info=True)

    async def _video_record(self, job: VideoJob, state: str, *, reason: str = "") -> None:
        await asyncio.to_thread(self._video_save, job, state, reason=reason)

    # ------------------------------------------------------------- cancel
    async def _video_cancel_pressed(self, interaction, job_id: str) -> None:
        """The Cancel button: the requester or an admin stops the job before its next step."""
        lang = self._interaction_lang(interaction)
        job = next(
            (j for j in list(self._video_jobs.values()) if getattr(j, "job_id", None) == job_id),
            None,
        )
        try:
            if job is None:  # already over: drop the stale button
                await interaction.response.edit_message(view=None)
                return
            user = interaction.user
            if user.id != job.requester_id and not self._may_configure(user):
                await interaction.response.send_message(t("video_not_yours", lang), ephemeral=True)
                return
            job.handle.cancelled.set()
            logger.info("video guild=%s cancel requested", job.guild_id)
            await interaction.response.edit_message(view=None)
        except discord.HTTPException:
            logger.info("video cancel response failed", exc_info=True)

    # ------------------------------------------------------------- restart
    async def _sweep_video_jobs(self) -> None:
        """Once per process (the first on_ready; reconnects fire it again): jobs a restart cut
        off are marked failed with reason "restarted" and their requester is told to ask again.
        Usage is not refunded. Jobs this process runs are skipped. Never raises."""
        if self._video_swept:
            return
        self._video_swept = True
        live = {getattr(j, "job_id", None) for j in list(self._video_jobs.values())}
        for guild in list(self.guilds):
            try:
                items = await asyncio.to_thread(
                    self.docs.find_items, JOBS, guild.id, lambda d: d.get("state") not in FINAL
                )
            except Exception:
                logger.warning("video job sweep failed for guild %s", guild.id, exc_info=True)
                continue
            for key, doc in items:
                if key in live:
                    continue
                update = {
                    **doc,
                    "state": "failed",
                    "reason": "restarted",
                    "updated_at": self.s.now().isoformat(timespec="seconds"),
                }
                try:
                    await asyncio.to_thread(self.docs.put, JOBS, guild.id, key, update)
                except Exception:
                    logger.warning("video job %s not marked failed", key, exc_info=True)
                    continue
                await self._video_tell_restarted(guild, doc)

    async def _video_tell_restarted(self, guild, doc: dict) -> None:
        try:
            cid, rid = int(doc.get("channel_id") or 0), int(doc.get("requester_id") or 0)
            channel = (guild.get_channel(cid) or self.get_channel(cid)) if cid else None
            if channel is None:
                return
            who = f"<@{rid}> " if rid else ""
            users = [discord.Object(rid)] if rid else False
            await channel.send(
                f"{who}{t('video_restarted', self._lang(channel))}",
                allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=users),
            )
        except Exception:
            logger.warning("could not tell the requester about a restarted video", exc_info=True)
