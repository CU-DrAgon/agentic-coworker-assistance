"""Discord glue of `create_video_ad` (the pipeline is `videoad.py`).

`VideoAdMixin` gives `SecretaryBot`:
- `_video_tools`: the `create_video_ad` agent tool (appended to `extra_tools` by `_agent_kwargs`)
  for chat answers (never in a voice room: `_agent_kwargs(spoken=True)`, else
  `imagegen_bot.looks_spoken`; interjections never get it), only when the guild's `video_ads`
  flag is on (default off), ffmpeg is installed and Thai text can be shaped. Guild, channel,
  author, the attached photos and the answer's language (`lang` from `_agent_kwargs`: the
  person's profile language in an auto channel) are bound here, never model input;
- the background job: the tool validates, peeks at the month's ads and reserves the guild's slot
  (one job per guild, `SALID_VIDEO_MAX_JOBS` across the bot) before anything starts, then
  `run_video_ad` runs in the bot's own render pool (`_video_pool`, `SALID_VIDEO_MAX_JOBS`
  threads, never asyncio's default pool) in a temp dir that is always removed. One progress
  message in the asker's channel, edited at most every 5 s with the current step, carries a
  Cancel button for the requester or an admin (pressed: "cancelling…" at once, and nothing is
  delivered afterwards); the MP4 comes back as a reply to the request with the scene list and a
  suggested caption ending in the AI hashtag of the job's language. A reply that went out
  without the MP4 (Discord refused the file) ends the job failed with reason `upload_refused`
  (the ad stays used: its media was billed);
- job records in DocStore kind `salid_video_jobs` (written on every state change, with the
  pending Omni interaction id `hero_id` while a hero clip is made; never the brief, the
  storyboard or media), and one active-jobs index in the same kind (guild 0, key `_active`,
  `{job id: guild id}`), changed under the slot lock when a job is reserved and when it ends;
  `_sweep_video_jobs`: on the first `on_ready`, it reads only that index and the jobs in it:
  jobs a restart cut off are marked failed, a hero clip they left running is cancelled (best
  effort), their ad is given back when the record shows nothing billed (state `storyboard`,
  `cost_usd` 0; a stopping bot sets `VideoHandle.shutdown`, so the runner never gives it back
  too), their requester is told to ask again, and their ids leave the index (records are never
  deleted). An ended job always leaves the index; a job whose index entry cannot be written
  does not start.

Jobs and their slots live in this process (one bot process, like `memory_index` and `recent`).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import re
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import discord

from . import cards, speech, videogen
from .dm import send_dm
from .i18n import CATALOG, resolve_lang, t
from .imagegen_bot import looks_spoken
from .llm import make_media_client
from .promo import usable_photos
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
INDEX_GUILD, INDEX_KEY = 0, "_active"  # in JOBS: {job id: guild id} of every job not yet ended
FINAL = frozenset({"done", "failed", "cancelled"})
PROGRESS_EVERY_S = 5.0  # the progress message is edited at most this often
MAX_HERO_SHOTS = 2
MAX_REPLY = 1900
_MONTH = re.compile(r"\d{4}-\d{2}")
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
    """The progress message's one button: cancels the job (requester or admin only). No timeout:
    a job with hero shots can run a long time; `_video_finish` stops the view when it ends."""

    def __init__(self, job_id: str, lang: str = "th"):
        super().__init__(timeout=None)
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
        self._video_media = None  # made by `_video_media_client` when an add-on is on
        # renders run here, not in asyncio's default pool (shared with every to_thread call)
        self._video_pool = ThreadPoolExecutor(
            max_workers=max(1, self.s.video_max_jobs), thread_name_prefix="salid-video"
        )

    def _video_media_client(self):
        """The client for paid media creates (`llm.make_media_client`: no SDK retry on 5xx),
        made once and only while a hero or music model is set; None otherwise (no paid create
        can happen, and the runner falls back to `self.llm`)."""
        if not (self.s.video_hero_model or self.s.video_music_model):
            return None
        with self._video_lock:
            if self._video_media is None:
                self._video_media = make_media_client(self.s)
            return self._video_media

    # ------------------------------------------------------------- agent tool
    def _video_tools(
        self, message, guild, asker, files, *, lang: str | None = None, spoken: bool = False
    ) -> list:
        """`create_video_ad` for this message, or [] (a voice room: `spoken`, or
        `looks_spoken` for callers that do not say; flag off, no ffmpeg, no Thai shaping). The
        job speaks `lang` (`_agent_kwargs` passes the answer's language, which follows the
        person's profile in an auto channel); other callers get the message's own language.
        Never raises."""
        try:
            if spoken or looks_spoken(message, files):
                return []
            if not self.team.enabled(guild.id, "video_ads"):
                return []
            if not ffmpeg_ready():
                _warn_once("ffmpeg", "ffmpeg missing: create_video_ad off")
                return []
            if not cards.thai_ready():
                _warn_once("raqm", "libfribidi missing: create_video_ad off")
                return []
            photos = [data for _, data in usable_photos(files)]
            loop = asyncio.get_running_loop()
            lang = lang or self._lang(message)
            return [self._video_tool(message, guild, asker, photos, lang, loop)]
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
            music=bool(music),  # the runner makes music only with SALID_VIDEO_MUSIC_MODEL
            lang=lang,
        )
        # a peek, outside the lock (a DocStore read): the job's own take re-checks the month
        if self.video_usage.used(gid, month, "ads") + 1 > cap:
            return {"status": "limit", "say": t("video_month_limit", lang, n=cap)}
        with self._video_lock:  # check and reserve together: two quick asks cannot both start
            if gid in self._video_jobs or len(self._video_jobs) >= self.s.video_max_jobs:
                return {"status": "busy", "say": t("video_busy", lang)}
            self._video_jobs[gid] = job
            # in the index before its first record; a job the sweep could not find never runs
            if not self._video_index(job.job_id, gid):
                del self._video_jobs[gid]
                return {"status": "error", "error": "the video could not be started"}
        try:  # nothing between the reservation and here can raise
            loop.call_soon_threadsafe(self._launch_video, job, message, req)
        except RuntimeError:  # the event loop is closing
            self._video_release(job)
            self._video_unindex(job.job_id)
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
        except Exception:  # its index entry has no record: the next start's sweep drops it
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
            job.handle.shutdown.set()  # the next start's sweep decides about the ad
            job.handle.cancelled.set()
            raise
        except Exception:
            logger.exception("video job crashed: guild=%s", job.guild_id)
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
                res = await loop.run_in_executor(
                    self._video_pool,
                    functools.partial(
                        run_video_ad,
                        req,
                        client=self.llm,
                        media_client=self._video_media_client(),
                        settings=self.s,
                        say=say,
                        workdir=Path(tmp),
                        # the upload limit where the reply goes (a DM: discord.py's 10 MiB
                        # default)
                        budget_mib=budget_mib(getattr(message.guild, "filesize_limit", None)),
                        handle=job.handle,
                        usage=self.video_usage,
                        month=self.s.now().strftime("%Y-%m"),
                    ),
                )
            except asyncio.CancelledError:
                # shutdown: tell the thread before its temp dir is removed (it stops before its
                # next step instead of writing into a deleted directory); the record stays, so
                # the runner leaves the ad's refund to the next start's sweep
                job.handle.shutdown.set()
                job.handle.cancelled.set()
                raise
            except Cancelled:
                state, text = "cancelled", t("video_cancelled", lang)
            except MonthLimit:
                state, reason = "failed", "month_limit"
                text = t("video_month_limit", lang, n=self.s.video_ads_per_month)
            except StoryboardError as e:  # problem lines only, never the storyboard text
                logger.info("video job storyboard failed: guild=%s %s", job.guild_id, e)
                state, reason = "failed", "storyboard"
            except RuntimeError as e:  # our ffmpeg / budget errors
                logger.warning("video job render failed: guild=%s %s", job.guild_id, e)
                state, reason = "failed", "render"
            except Exception:
                logger.exception("video job pipeline crashed: guild=%s", job.guild_id)
                state, reason = "failed", "error"
            if state == "done":
                job.handle.cost_usd = res.cost_usd
                if job.handle.cancelled.is_set():  # pressed after the runner's last check
                    logger.info("video job cancelled before delivery: guild=%s", job.guild_id)
                    state, text = "cancelled", t("video_cancelled", lang)
                elif await self._video_deliver(job, message, res):
                    text = t("video_done", lang)
                else:  # the reply went out without the MP4; its media was billed: no refund
                    state, reason = "failed", "upload_refused"
                    text = t("video_upload_refused", lang)
        if reason and not text:
            text = t("video_failed", lang, reason=t(f"video_reason_{reason}", lang))
        await self._video_finish(job, message, state, text, reason=reason)

    def _video_progress(self, job: VideoJob, step: str) -> None:
        """A step started (on the loop): show its label, at most one edit per PROGRESS_EVERY_S
        (a step that comes sooner waits, and the latest label wins, also one that comes while
        an edit is in flight: `_video_edit_after` edits again). Nothing after a cancel (the
        message says "cancelling…") or when the label is unchanged (a hero clip's id came or
        went)."""
        key = f"video_step_{step}"
        if job.final or job.msg is None or key not in CATALOG or job.handle.cancelled.is_set():
            return
        label = t(key, job.lang)
        if label == job.label:
            return
        job.label = label
        if job.edit_task is not None and not job.edit_task.done():
            return  # the pending edit will show this label
        wait = max(0.0, job.last_edit + PROGRESS_EVERY_S - self._clock())
        job.edit_task = asyncio.create_task(self._video_edit_after(job, wait))

    async def _video_edit_after(self, job: VideoJob, wait: float) -> None:
        """Edit the progress message to `job.label` after `wait` s; a label that changed while
        the edit was in flight gets one more edit, PROGRESS_EVERY_S after the last."""
        while True:
            if wait > 0:
                await self._sleep(wait)
            if job.final or job.handle.cancelled.is_set():
                return
            label = job.label
            job.last_edit = self._clock()
            try:
                await job.msg.edit(content=label)
            except discord.HTTPException:
                logger.info("video progress edit failed", exc_info=True)
            if job.label == label:
                return
            wait = max(0.0, job.last_edit + PROGRESS_EVERY_S - self._clock())

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
        # it ended in this process: out of the index even if the record could not be written
        await asyncio.to_thread(self._video_unindex, job.job_id)
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

    async def _video_deliver(self, job: VideoJob, message, res) -> bool:
        """Reply to the request with the MP4, the scene list and a suggested caption. Reply
        failed (request gone, or any other error): the same in its channel. Upload refused there
        too: the text with a note. True when the MP4 went out; False when only text did
        (`_reply_with_uploads` falls back to text when Discord refuses the file: its message
        then has no attachment)."""
        text, allowed = self._video_reply(job, res), self._video_mentions(job)
        files = [discord.File(str(res.path), filename=res.path.name)]
        try:
            try:
                sent = await self._reply_with_uploads(message, text, allowed, {"files": files})
                if getattr(sent, "attachments", None):
                    return True
                logger.warning("video reply went out without the MP4: guild=%s", job.guild_id)
                return False
            except Exception as e:  # gone, refused, or not a message that can be replied to
                logger.info("video reply failed (%s); posting in the channel", type(e).__name__)
            files.append(discord.File(str(res.path), filename=res.path.name))
            try:
                await message.channel.send(text, file=files[-1], allowed_mentions=allowed)
                return True
            except discord.HTTPException:
                logger.warning("could not post the video ad", exc_info=True)
            with contextlib.suppress(discord.HTTPException):
                note = t("attach_failed", job.lang)
                await message.channel.send(f"{text}\n{note}", allowed_mentions=allowed)
            return False
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
        tag = t("video_ai_hashtag", lang)
        suggested = " ".join(w.strip() for w in [*words, tag] if w.strip())
        head = f"<@{job.requester_id}> {t('video_done', lang)}"
        tail = f"{t('video_caption', lang)} {suggested}"
        text = f"{head}\n{t('video_scenes', lang)}\n" + "\n".join(scenes) + f"\n\n{tail}"
        return text if len(text) <= MAX_REPLY else f"{head}\n\n{tail}"

    def _video_save(self, job: VideoJob, state: str, *, reason: str = "") -> bool:
        """Write the job's record (blocking), with `hero_id` while a hero clip is pending. Never
        the brief, storyboard or media. Never raises; False when it was not written."""
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
                    # the Omni clip being made: a restart cancels it (`_sweep_video_jobs`)
                    **({"hero_id": job.handle.hero_id} if job.handle.hero_id else {}),
                },
            )
            return True
        except Exception:
            logger.warning("video job record not saved", exc_info=True)
            return False

    async def _video_record(self, job: VideoJob, state: str, *, reason: str = "") -> bool:
        return await asyncio.to_thread(self._video_save, job, state, reason=reason)

    # ------------------------------------------------------------- active-jobs index
    def _video_index(self, job_id: str, guild_id: int | None) -> bool:
        """Add `job_id` (of `guild_id`) to the active-jobs index, or remove it (None): one
        read-modify-write of the one index document. The caller holds `_video_lock`, so two
        changes never lose each other. Never raises: False when it could not be written (a job
        is not started then; an entry left behind is dropped by the next start's sweep)."""
        try:
            doc = self._video_active()
            if guild_id is None:
                if job_id not in doc:
                    return True
                del doc[job_id]
            else:
                doc[job_id] = guild_id
            self.docs.put(JOBS, INDEX_GUILD, INDEX_KEY, doc)
            return True
        except Exception:
            logger.warning("video job index not updated", exc_info=True)
            return False

    def _video_unindex(self, job_id: str) -> None:
        """Remove an ended job from the index, under the slot lock (blocking)."""
        with self._video_lock:
            self._video_index(job_id, None)

    def _video_active(self) -> dict[str, int]:
        """The active-jobs index: {job id: guild id} (one read; blocking)."""
        doc = dict(self.docs.get(JOBS, INDEX_GUILD, INDEX_KEY) or {})
        doc.pop("guild_id", None)  # Firestore adds it on write
        return {str(k): int(v) for k, v in doc.items()}

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
            logger.info("video job cancel requested: guild=%s", job.guild_id)
            # at once (the job stops before its next step; a step running can take minutes)
            await interaction.response.edit_message(
                content=t("video_cancelling", job.lang), view=None
            )
        except discord.HTTPException:
            logger.info("video cancel response failed", exc_info=True)

    # ------------------------------------------------------------- restart
    async def _sweep_video_jobs(self) -> None:
        """Once per process (the first on_ready; reconnects fire it again), reading only the
        active-jobs index and the job records it lists: jobs a restart cut off are marked failed
        with reason "restarted", the hero clip one left running (its record's `hero_id`) is
        cancelled, best effort, the month's ad is given back when the record shows nothing was
        billed (`_video_refundable`), the requester is told to ask again, and the id leaves the
        index (an entry whose record ended or is missing just leaves it). Never raises.

        Requests keep working while it runs, so each job is checked against this process's
        slots before its record is read afresh, and again right before it is marked. A job's
        slot is reserved (and the job indexed) before its first record and released only after
        its final one, so a job of this process that is not live has already ended."""
        if self._video_swept:
            return
        self._video_swept = True
        try:
            index = await asyncio.to_thread(self._video_active)
        except Exception:
            logger.warning("video job sweep: index not read", exc_info=True)
            return
        for key, gid in index.items():
            if self._video_is_live(gid, key):  # runs here: it leaves the index when it ends
                continue
            try:
                doc = await asyncio.to_thread(self.docs.get, JOBS, gid, key)
            except Exception:
                logger.warning("video job %s not read", key, exc_info=True)
                continue
            if not doc or doc.get("state") in FINAL:  # ended (or never written)
                await asyncio.to_thread(self._video_unindex, key)
                continue
            # no await between this check and the write below starting
            if self._video_is_live(gid, key):
                continue
            update = {
                **doc,
                "state": "failed",
                "reason": "restarted",
                "updated_at": self.s.now().isoformat(timespec="seconds"),
            }
            hero_id = str(update.pop("hero_id", "") or "")
            try:
                await asyncio.to_thread(self.docs.put, JOBS, gid, key, update)
            except Exception:
                logger.warning("video job %s not marked failed", key, exc_info=True)
                continue
            month = self._video_refundable(doc)
            if month:
                await asyncio.to_thread(self._video_give_back, gid, month)
            await asyncio.to_thread(self._video_unindex, key)
            if hero_id:
                await asyncio.to_thread(self._video_cancel_hero, hero_id)
            guild = next((g for g in self.guilds if g.id == gid), None)
            if guild is not None:
                await self._video_tell_restarted(guild, doc)

    @staticmethod
    def _video_refundable(doc: dict) -> str:
        """The month whose ad a cut-off job gives back, or "": only when it stopped in the
        `storyboard` step with `cost_usd` 0. The runner takes the ad right before reporting
        "storyboard", the one step that starts before any paid call; later records can show 0
        while stills, a hero clip or music were already billed (the cost is written at each
        step's start), and a `queued` job may not have taken its ad yet."""
        try:
            cost = float(doc["cost_usd"])
        except (KeyError, TypeError, ValueError):
            return ""
        month = str(doc.get("created_at") or "")[:7]
        if cost != 0 or doc.get("state") != "storyboard" or not _MONTH.fullmatch(month):
            return ""
        return month

    def _video_give_back(self, guild_id: int, month: str) -> None:
        """Give a cut-off job's ad back (blocking). Never raises."""
        try:
            self.video_usage.give_back(guild_id, month, "ads", 1)
            logger.info("video ad given back after a restart: guild=%s", guild_id)
        except Exception:
            logger.warning("video ad not given back after a restart", exc_info=True)

    def _video_cancel_hero(self, interaction_id: str) -> None:
        """Cancel the Omni clip a restart left running (it would run, and bill, to its end), on
        the normal client. Best effort (blocking); never raises, logs the error type only."""
        try:
            if self.llm is not None:
                videogen.cancel_hero(self.llm, interaction_id)
        except Exception as e:
            logger.warning("video hero cancel after a restart failed: %s", type(e).__name__)

    def _video_is_live(self, guild_id: int, job_id: str) -> bool:
        """This process runs (or is launching) the job `job_id` of the guild."""
        with self._video_lock:
            return getattr(self._video_jobs.get(guild_id), "job_id", None) == job_id

    async def _video_tell_restarted(self, guild, doc: dict) -> None:
        """Tell the requester in the job's channel; no such channel (a DM after a restart, a
        deleted channel): by DM (or their private Salid channel). Best effort."""
        try:
            cid, rid = int(doc.get("channel_id") or 0), int(doc.get("requester_id") or 0)
            channel = (guild.get_channel(cid) or self.get_channel(cid)) if cid else None
            if channel is None:
                member = guild.get_member(rid) if rid else None
                if member is not None:
                    lang = resolve_lang(self.team.current(guild.id).default_language)
                    await send_dm(member, t("video_restarted", lang))
                return
            who = f"<@{rid}> " if rid else ""
            users = [discord.Object(rid)] if rid else False
            await channel.send(
                f"{who}{t('video_restarted', self._lang(channel))}",
                allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=users),
            )
        except Exception:
            logger.warning("could not tell the requester about a restarted video", exc_info=True)
