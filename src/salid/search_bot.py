"""Discord glue of the search tools (logic: `search.py`, `web_search.py`, `research.py`).

`SearchMixin` gives `SecretaryBot`:
- `_search_tools`, appended to `extra_tools` by `_agent_kwargs` (so chat and voice share it): the
  internal `search_everything` over the tools the bot already built for this asker, plus
  `web_search` and `deep_research` when the `web` feature is on;
- the background research job: acknowledged at once by the agent, run in a worker thread, posted
  (message + PDF) to the channel the asker asked in, cancellable with the same words as a deep task
  ("ยกเลิกงาน" / `cancel_deep_task`), at most one per guild at a time and a daily cap per guild;
  recorded in `salid_jobs` (salid.jobs_bot: running, then done / failed / cancelled; never the
  question).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
from dataclasses import dataclass

import discord

from .i18n import resolve_lang, t
from .jobs import new_id as new_job_id
from .media import DailyBudget
from .pdf_report import find_fonts
from .qa import LedgerTools
from .research import (
    ResearchCancelled,
    ResearchHandle,
    clean_depth,
    markdown_pdf,
    run_research,
    short_version,
)
from .search import Expander, build_sources, search_tools
from .web_search import make_web_search, today_bangkok

logger = logging.getLogger(__name__)
REPORT_NAME = "research.pdf"


@dataclass
class _ResearchJob:
    task: asyncio.Task
    handle: ResearchHandle
    starter_id: int | None
    job_id: str = ""  # its `salid_jobs` record (salid.jobs_bot)


class SearchMixin:
    """Needs `self.llm`, `self.s`, `self.team`, `self.store`, `self.secretary_store`, `_deep_jobs`,
    `_job_start` / `_job_end` (`JobsMixin`). Call `_search_init()` once after the settings exist."""

    def _search_init(self) -> None:
        self._research_jobs: dict[int, _ResearchJob] = {}
        self._research_tasks: set[asyncio.Task] = set()
        self._research_budget = DailyBudget(max(0, int(self.s.research_daily_cap)))
        self._expander = Expander(self.llm, self.s.gate_model)

    def _search_on(self, guild_id: int) -> bool:
        return self.team.enabled(guild_id, "search")

    # ------------------------------------------------------------- agent tools
    def _search_tools(self, message, guild, asker, is_exec: bool, memory, tools: list, lang: str):
        """`search_everything` (+ `web_search`, `deep_research`) for one request. `tools` = the
        agent tools already built for this asker and audience: every internal source goes through
        them, so nothing is widened here. Never raises."""
        try:
            gid = guild.id
            if not self._search_on(gid):
                return []
            self._expander.client = self.llm  # the bot's current client (None: glossary only)
            src = build_sources(
                memory=memory,
                tools=tools,
                zone=self.s.zone,
                now=self.s.now,
                user_id=asker.id,
                list_meetings=lambda a, b: self.secretary_store.list_meetings(
                    gid, a, b, statuses=("scheduled", "started", "done")
                ),
                # same rule as the ledger tools: executives see everyone's receipts, anyone else
                # only the ones they submitted themselves
                ledger=LedgerTools(
                    self.store,
                    gid,
                    forced_submitter_id=None if is_exec else asker.id,
                    actor=f"user {asker.id}",
                ),
            )
            internal = search_tools(src, self._expander, self.s.now, self.s.zone)
            out = list(internal)
            if self.llm is not None and self.team.enabled(gid, "web"):
                out.append(make_web_search(self.llm, self.s.gemini_model, now=self.s.now))
                out.append(self._research_tool(message, guild, asker, lang, internal[0]))
            for f in out:
                f.salid_group = "search"
            return out
        except Exception:
            logger.warning("search tools unavailable", exc_info=True)
            return []

    def _research_tool(self, message, guild, asker, lang: str, internal_search):
        loop = asyncio.get_running_loop()
        gid = guild.id

        def deep_research(
            question: str, depth: str = "quick", include_internal: bool = False
        ) -> dict:
            """Start a background multi-step WEB research and report back: it plans 3-6
            sub-questions, searches each with Google, and writes a cited report (summary, key
            findings, table, open questions) posted here as a message plus a PDF. Use when the
            user asks for research ("ทำรีเสิร์ช", "หาข้อมูลเชิงลึก", "เปรียบเทียบ", "analyse the
            market"), not for one quick fact (use web_search). After it starts, tell the user it
            takes about 1-2 minutes; they can say "ยกเลิกงาน" to stop it. Limited per day.

            Args:
                question: the research question with everything needed, in the user's language.
                depth: "quick" (3 sub-questions, about 1 minute) or "thorough" (6, 2-3 minutes).
                include_internal: also use what the team's own chat, files and wiki say (only what
                    the asker may see) when the question is about the company or its clients.
            """
            question = " ".join((question or "").split())
            if not question:
                return {"error": "give the research question"}
            job = self._research_jobs.get(gid)
            if job is not None and not job.task.done():
                return {"error": "busy", "say": t("research_busy", lang)}
            day = self.s.now().date().isoformat()
            if not self._research_budget.available(gid, day):
                return {
                    "error": "daily_limit",
                    "say": t("research_limit", lang, n=int(self._research_budget.cap)),
                }
            self._research_budget.add(gid, day, 1)
            ctx = None
            if include_internal:
                ctx = lambda q=question: _internal_text(internal_search(q))  # noqa: E731
            loop.call_soon_threadsafe(
                self._launch_research,
                message,
                guild,
                asker,
                question,
                clean_depth(depth),
                lang,
                ctx,
            )
            return {"status": "started", "say": self._research_say(question, lang, loop)}

        return deep_research

    def _research_say(self, question: str, lang: str, loop) -> str:
        """The line the agent says when a research starts: written for this question, keeping the
        facts (about 1-2 minutes, report + PDF here, how to cancel); the catalog wording when the
        model is slow or fails. Runs in the tool's worker thread, never on the event loop."""
        fallback = t("research_ack", lang)
        writer = getattr(self, "fillers", None)
        try:
            asyncio.get_running_loop()
            return fallback  # on the event loop thread: never block it
        except RuntimeError:
            pass
        if writer is None or not writer.enabled():
            return fallback
        facts = (
            (
                "it takes about 1-2 minutes",
                "a report with a PDF will be posted in this channel",
                "say “cancel task” to stop it",
            )
            if lang != "th"
            else (
                "ใช้เวลาประมาณ 1-2 นาที",
                "เสร็จแล้วส่งรายงานพร้อมไฟล์ PDF ที่ห้องนี้",
                "สั่ง “ยกเลิกงาน” ได้ถ้าไม่ต้องการแล้ว",
            )
        )
        try:
            fut = asyncio.run_coroutine_threadsafe(
                writer.write(
                    kind="research",
                    question=question,
                    lang=lang,
                    particle=str(getattr(self.s, "persona_particle", "ค่ะ") or "ค่ะ"),
                    facts=facts,
                ),
                loop,
            )
            return fut.result(float(getattr(self.s, "ack_llm_timeout_s", 2.0))) or fallback
        except Exception:
            return fallback

    # ------------------------------------------------------------- the background job
    def _launch_research(self, message, guild, asker, question, depth, lang, ctx) -> None:
        handle = ResearchHandle()
        gid = guild.id
        job_id = new_job_id()
        task = asyncio.create_task(
            self._run_research(
                message, guild, asker, question, depth, lang, ctx, handle, job_id=job_id
            )
        )
        self._research_tasks.add(task)
        self._research_jobs[gid] = _ResearchJob(task, handle, getattr(asker, "id", None), job_id)
        task.add_done_callback(self._research_tasks.discard)
        task.add_done_callback(
            lambda _t, g=gid, me=task: (
                self._research_jobs.pop(g, None)
                if getattr(self._research_jobs.get(g), "task", None) is me
                else None
            )
        )

    async def _run_research(
        self, message, guild, asker, question, depth, lang, ctx, handle, job_id=""
    ) -> None:
        """The research's background job. With a `job_id` it is recorded (salid.jobs_bot):
        running first, then done / failed ("error", also when the report could not be posted) /
        cancelled by the user, whatever happens; a shutdown leaves it for the next start's
        sweep. Never the question."""
        channel = message.channel
        lang = resolve_lang(lang, question)
        gid = guild.id
        state: str | None = "failed"  # until it ends some other way
        reason = "error"
        try:
            try:
                await self._job_start(
                    "research", job_id, message, gid, requester_id=getattr(asker, "id", None)
                )
                res = await asyncio.to_thread(
                    run_research,
                    self.llm,
                    question=question,
                    depth=depth,
                    lang=lang,
                    today=today_bangkok(self.s.now()),
                    search_model=self.s.gemini_model,
                    plan_model=self.s.gate_model,
                    synth_model=self.s.gemini_model,
                    internal=ctx,
                    handle=handle,
                )
            except (asyncio.CancelledError, ResearchCancelled) as e:
                state, reason = self._job_cancel_state(handle, e), ""  # before handle.cancel()
                handle.cancel()
                logger.info("deep research %s", "cancelled" if state else "stopped by a shutdown")
                return
            except Exception:
                logger.exception("deep research crashed")
                with contextlib.suppress(discord.HTTPException):
                    await channel.send(t("research_failed", lang))
                return
            logger.info(
                "deep research done in %.0fs: %d sub-questions (%d failed), %d sources",
                res.seconds,
                len(res.sub_questions),
                res.failed,
                len(res.sources),
            )
            try:
                pdf = await asyncio.to_thread(
                    markdown_pdf,
                    org=getattr(guild, "name", "") or "Salid",
                    title=res.title,
                    body=res.report,
                    lang=lang,
                    generated_at=self.s.now(),
                    fonts=find_fonts(self.s.pdf_font),
                )
            except Exception:
                logger.warning("research PDF failed", exc_info=True)
                pdf = None
            who = f"<@{asker.id}> " if getattr(asker, "id", None) else ""
            text = f"{who}{t('research_done', lang)}\n\n{short_version(res.report, lang)}"
            if len(text) > 1900:
                text = text[:1890] + "…"
            file = discord.File(io.BytesIO(pdf), filename=REPORT_NAME) if pdf else None
            try:
                await channel.send(text, **({"file": file} if file else {}))
            except discord.HTTPException:
                logger.warning("could not post the research report", exc_info=True)
                return  # recorded failed ("error"): nothing reached the asker
            state, reason = "done", ""
        except asyncio.CancelledError as e:  # stopped while the report was being made or posted
            state, reason = self._job_cancel_state(handle, e), ""
            raise
        finally:
            await self._job_end(gid, job_id, state, reason)

    def _research_cancel(self, guild_id: int, user, lang: str) -> str | None:
        """Stop the guild's running research (None: there is none). Same rule as a deep task:
        whoever started it, server managers and executives."""
        job = self._research_jobs.get(guild_id)
        if job is None or job.task.done():
            return None
        if user.id != job.starter_id and not (
            hasattr(user, "guild_permissions") and self.is_exec_member(user)
        ):
            return t("cancel_denied", lang)
        job.handle.cancel()
        job.task.cancel()
        return t("research_cancelled", lang)


def _internal_text(res: dict) -> str:
    """The scoped search_everything hits as plain lines for the synthesis call."""
    lines = []
    for r in (res or {}).get("results") or []:
        lines.append(
            f"- ({r['source']}, {r.get('date') or '-'}, {r.get('who') or '-'}) {r.get('title', '')}: {r.get('snippet', '')}"
        )
    return "\n".join(lines)
