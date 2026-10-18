"""Wiring for `python -m salid.platforms`: the same stores, memory and model client as the Discord
bot, plus whichever platforms have secrets set."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta

from ..config import Settings
from ..embed import make_embedder
from ..guild_settings import BotGuildSettings, make_guild_store
from ..i18n import resolve_lang
from ..llm import make_client
from ..memory import MemoryScheduler, MemoryService
from ..memory_index import IndexedMemoryStore
from ..memory_store import make_memory_store
from ..store import make_store
from . import change_outbox_platform, ids
from .calendar_glue import CalendarGlue
from .config import PlatformSettings
from .core import Assistant, safe_exc

logger = logging.getLogger(__name__)

TICK_SECONDS = 30.0
IMPORT_RESUME_S = 600.0
IMPORT_IDLE_S = 6 * 3600.0
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class PlatformScheduler(MemoryScheduler):
    """Day summaries for the platform tenants. The daily purge differs from the Discord bot's in
    one way that matters: guilds this process does not know are NEVER purged with the default
    retention (the shared store also holds Discord guilds with their own longer retention)."""

    def maybe_purge(self, now: datetime):
        day = now.date().isoformat()
        if self._purged_on == day:
            return None
        self._purged_on = day
        try:
            utc = now.astimezone(UTC)
            cutoffs = {
                gid: utc - timedelta(days=self.retention_of(gid)) for gid in self._known_guilds
            }
            m, s = self.store.purge(cutoffs, EPOCH)  # default cutoff = the epoch: others untouched
        except Exception as e:
            logger.warning("platform memory purge failed (%s)", safe_exc(e))
            self._purged_on = None
            return None
        logger.info("platform memory purge: %d messages, %d summaries", m, s)
        return m, s


def build_assistant(
    s: Settings,
    ps: PlatformSettings,
    *,
    store=None,
    llm=None,
    guilds=None,
    memory_store=None,
    **kw,
) -> Assistant:
    store = store or make_store(s)
    llm = llm if llm is not None else make_client(s)
    team = BotGuildSettings(guilds or make_guild_store(s))
    embedder = make_embedder(s, llm)
    base = memory_store or make_memory_store(s, store)
    mstore = IndexedMemoryStore(base, embedder)
    names: dict = {}
    scheduler = PlatformScheduler(
        mstore,
        s,
        llm,
        now=s.now,
        lang_of=lambda gid: resolve_lang(team.current(gid).default_language),
        retention_of=team.fresh_memory_retention_days,
        channel_name_of=lambda gid, cid: names.get((gid, cid), str(cid)),
        enabled=lambda gid: team.enabled(gid, "memory"),
        embedder=embedder,
    )
    from ..archive import make_archive
    from ..identity_links import IdentityLinks
    from ..orgs import make_org_store
    from .org_sources import OrgSources

    options = {"identity": IdentityLinks(make_org_store(s, store))} | kw
    identity = options["identity"]
    if "org_sources" not in options and identity is not None:
        options["org_sources"] = OrgSources(identity.orgs, discord_token=ps.discord_token)

    assistant = Assistant(
        settings=s,
        ps=ps,
        store=store,
        llm=llm,
        team=team,
        memory=MemoryService(mstore, s, scheduler),
        memory_store=mstore,
        embedder=embedder,
        archive=make_archive(s),
        guilds_store=team.store,
        names=names,
        **options,
    )
    if identity is not None:  # tokens live in the org store's DocStore, shared with the web
        assistant.calendar = CalendarGlue.from_settings(ps, identity.orgs.docs, s.tz)
    return assistant


async def memory_loop(a: Assistant) -> None:
    """Summaries, vectors and the daily purge for every tenant seen so far."""
    while True:
        await asyncio.sleep(TICK_SECONDS)
        try:
            await asyncio.to_thread(a.memory.scheduler.tick, list(a.tenants))
        except Exception as e:
            logger.error("platform memory tick failed (%s)", safe_exc(e))


async def import_loop(importer, wake: asyncio.Event) -> None:
    await asyncio.sleep(5)
    while True:
        try:
            await importer.run()
        except Exception as e:
            logger.error("slack import failed (%s)", safe_exc(e))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                wake.wait(), IMPORT_RESUME_S if importer.capped else IMPORT_IDLE_S
            )
        wake.clear()


def _changes_pusher(a: Assistant, platform: str, tenant: str, push) -> None:
    """Tell the change outbox that this tenant can be messaged 1:1 (see change_outbox_platform)."""
    if a.changes is not None:
        a.changes.add_pusher(platform, tenant, push)


async def run_slack(a: Assistant, ps: PlatformSettings) -> None:
    from . import slack
    from .slack_import import SlackImporter

    handler, adapter, port = await slack.connect(ps, a)
    if a.org_sources is not None:
        a.org_sources.add_slack_port(port)  # org-wide scope reads this team's channel lists
    _changes_pusher(a, "slack", port.team_id, change_outbox_platform.slack_pusher(port))
    tasks = [asyncio.create_task(handler.start_async())]
    if ps.slack_import:
        wake = asyncio.Event()
        adapter.on_joined = lambda _channel: wake.set()
        tasks.append(asyncio.create_task(import_loop(SlackImporter(a, port), wake)))
    logger.info("slack ready team=%s tenant=%d", port.team_id, ids.tenant_id("slack", port.team_id))
    await asyncio.gather(*tasks)


async def run_line(a: Assistant, ps: PlatformSettings) -> None:
    import uvicorn

    from . import line

    api = line.SdkLineApi(ps.line_channel_access_token)
    info = await asyncio.to_thread(api.bot_info)
    port = line.LinePort(api, ps)
    _changes_pusher(a, "line", info["user_id"], change_outbox_platform.line_pusher(api))
    adapter = line.LineAdapter(a, port, info["user_id"])
    app = line.create_app(ps.line_channel_secret, adapter)
    config = uvicorn.Config(
        app, host=ps.line_host, port=ps.line_port, log_config=None, access_log=False
    )
    logger.info(
        "line webhook on %s:%d%s tenant=%d",
        ps.line_host, ps.line_port, line.WEBHOOK_PATH, ids.tenant_id("line", info["user_id"]),
    )  # fmt: skip
    await uvicorn.Server(config).serve()


async def run_telegram(a: Assistant, ps: PlatformSettings) -> None:
    from . import telegram

    api = telegram.TelegramApi(ps.telegram_bot_token)
    try:
        me = await api.get_me()
        port = telegram.TelegramPort(api)
        _changes_pusher(a, "telegram", str(me["id"]), change_outbox_platform.telegram_pusher(api))
        adapter = telegram.TelegramAdapter(a, port, str(me["id"]), me.get("username") or "")
        logger.info(
            "telegram ready bot=%s tenant=%d",
            me.get("username") or me["id"],
            ids.tenant_id("telegram", str(me["id"])),
        )
        await telegram.Poller(api, adapter).run()
    finally:
        await api.aclose()


async def run_whatsapp(a: Assistant, ps: PlatformSettings) -> None:
    import uvicorn

    from . import whatsapp

    api = whatsapp.WhatsAppApi(
        ps.whatsapp_access_token, ps.whatsapp_phone_number_id, ps.whatsapp_api_version
    )
    try:
        port = whatsapp.WhatsAppPort(api)
        _changes_pusher(
            a, "whatsapp", ps.whatsapp_phone_number_id, change_outbox_platform.whatsapp_pusher(api)
        )
        adapter = whatsapp.WhatsAppAdapter(a, port, ps.whatsapp_phone_number_id)
        app = whatsapp.create_app(ps.whatsapp_app_secret, ps.whatsapp_verify_token, adapter)
        config = uvicorn.Config(
            app, host=ps.whatsapp_host, port=ps.whatsapp_port, log_config=None, access_log=False
        )
        logger.info(
            "whatsapp webhook on %s:%d%s tenant=%d",
            ps.whatsapp_host, ps.whatsapp_port, whatsapp.WEBHOOK_PATH,
            ids.tenant_id("whatsapp", ps.whatsapp_phone_number_id),
        )  # fmt: skip
        await uvicorn.Server(config).serve()
    finally:
        await api.aclose()


async def run(s: Settings, ps: PlatformSettings) -> None:
    a = build_assistant(s, ps)
    jobs = [asyncio.create_task(memory_loop(a))]
    if a.identity is not None:  # the org store exists: central changes can be delivered
        from ..change_outbox import ChangeOutbox

        a.changes = change_outbox_platform.PlatformChanges(
            ChangeOutbox.from_settings(a.identity.orgs.docs, s), a.identity.orgs, a.team
        )
        jobs.append(asyncio.create_task(change_outbox_platform.changes_loop(a.changes)))
    if ps.slack_enabled:
        jobs.append(asyncio.create_task(run_slack(a, ps)))
    if ps.line_enabled:
        jobs.append(asyncio.create_task(run_line(a, ps)))
    if ps.telegram_enabled:
        jobs.append(asyncio.create_task(run_telegram(a, ps)))
    if ps.whatsapp_enabled:
        jobs.append(asyncio.create_task(run_whatsapp(a, ps)))
    logger.info("platforms started: %s", ", ".join(ps.enabled_platforms()))
    await asyncio.gather(*jobs)
