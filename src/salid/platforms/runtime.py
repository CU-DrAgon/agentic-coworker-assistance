"""Wiring for `python -m salid.platforms`: the same stores, memory and model client as the Discord
bot, plus whichever platforms have secrets set."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .. import monitor
from ..config import Settings
from ..embed import make_embedder
from ..guild_settings import BotGuildSettings, make_guild_store
from ..i18n import resolve_lang
from ..llm import make_client
from ..memory import MemoryScheduler, MemoryService
from ..memory_index import IndexedMemoryStore
from ..memory_store import make_memory_store
from ..monitor import HEARTBEAT_EVERY_S, Emitter, org_key_resolver
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


@dataclass
class PlatformMonitor:
    """The platform bridges' side of the workflow monitor: producers add change-log events to
    `emitter`; `monitor_loop` flushes it and writes the `platforms` heartbeat with `queues()`."""

    emitter: Emitter
    docs: Any  # the org store's DocStore, shared with the web
    queues: Callable[[], dict[str, int]]
    clock: Callable[[], float] = time.monotonic
    last_beat: float = float("-inf")


def platform_monitor(s: Settings, a: Assistant) -> PlatformMonitor | None:
    """The monitor of this process, or None when `SALID_MONITOR_ENABLED` is off or there is no
    org store (the change log and the heartbeat live in its DocStore)."""
    if a.identity is None or not s.monitor_enabled:
        return None
    docs = a.identity.orgs.docs
    return PlatformMonitor(
        emitter=Emitter(docs, enabled=True, resolve=org_key_resolver(docs)),
        docs=docs,
        queues=lambda: {"actions": 0, "outbox_pending": 0, "memory": a.memory.scheduler.backlog()},
    )


def change_events(m: PlatformMonitor | None) -> Callable[[list], None] | None:
    """The change outbox's `on_change` in this process: one event per stored or updated change,
    buffered in the platform emitter under the change's org (flushed by `monitor_loop`). None
    while the monitor is off. The hook never raises (the outbox also guards it)."""
    if m is None:
        return None
    from ..monitor_sources import adapt_change

    def on_change(changes: list) -> None:
        try:
            now = m.emitter.now()
            for c in changes:
                m.emitter.add(c.org_id, monitor.event_for(adapt_change(c, now=now)))
        except Exception as e:  # the type only: the text may quote the change
            logger.warning("platform monitor change event failed (%s)", type(e).__name__)

    return on_change


async def monitor_tick(m: PlatformMonitor) -> None:
    """Flush the buffered events and, every `HEARTBEAT_EVERY_S`, write the `platforms` heartbeat.
    The store work and `queues()` (the memory backlog takes the scheduler lock, which a seed can
    hold across a store read) run off the event loop. Never raises."""

    def beat() -> bool:
        return monitor.write_heartbeat(m.docs, "platforms", queues=m.queues(), now=m.emitter.now())

    try:
        await asyncio.to_thread(m.emitter.flush)
        now = m.clock()
        if now - m.last_beat < HEARTBEAT_EVERY_S:
            return
        m.last_beat = now  # a failed write is tried again on the next beat
        await asyncio.to_thread(beat)
    except Exception as e:  # the type only: the text may quote stored data
        logger.warning("platform monitor tick failed (%s)", type(e).__name__)


async def monitor_loop(m: PlatformMonitor) -> None:
    """The monitor tick every TICK_SECONDS, the first one at start (the heartbeat shows up at
    once)."""
    while True:
        await monitor_tick(m)
        await asyncio.sleep(TICK_SECONDS)


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


class PlatformsHttp:
    """The single HTTP port (Cloud Run): `GET /health` plus the LINE / WhatsApp webhook apps, which
    are registered by path once their (slower) startup is done. A path nobody registered yet answers
    503, so the platform retries instead of the container failing its port check."""

    def __init__(self, platforms: list[str] | None = None) -> None:
        self.apps: dict[str, object] = {}
        self.platforms: list[str] = list(platforms or [])

    def register(self, path: str, app, platform: str) -> None:
        self.apps[path] = app
        if platform not in self.platforms:
            self.platforms.append(platform)

    def drop(self, platform: str) -> None:
        if platform in self.platforms:
            self.platforms.remove(platform)

    async def __call__(self, scope, receive, send) -> None:
        from starlette.responses import JSONResponse, Response

        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif event["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        path = scope["path"]
        if path == "/health" and scope["method"] in ("GET", "HEAD"):
            resp = JSONResponse({"ok": True, "platforms": list(self.platforms)})
        elif path in self.apps:
            await self.apps[path](scope, receive, send)
            return
        elif path in WEBHOOK_PATHS:
            resp = Response(status_code=503)
        else:
            resp = Response(status_code=404)
        await resp(scope, receive, send)


WEBHOOK_PATHS = ("/line/webhook", "/whatsapp/webhook")  # checked against the modules in tests


async def serve_http(http: PlatformsHttp, port: int, started: asyncio.Event | None = None) -> None:
    import uvicorn

    config = uvicorn.Config(
        http,
        host="0.0.0.0",
        port=port,
        log_config=None,
        access_log=False,  # noqa: S104
    )
    server = uvicorn.Server(config)
    logger.info("http server on 0.0.0.0:%d (/health, platform webhooks)", port)

    async def watch() -> None:
        while not server.started:
            await asyncio.sleep(0.02)
        started.set()

    watcher = asyncio.create_task(watch()) if started is not None else None
    try:
        await server.serve()
    finally:
        if watcher is not None:
            watcher.cancel()


async def _guarded(name: str, coro: Awaitable, http: PlatformsHttp) -> None:
    """Combined mode: one platform failing is logged and leaves the others (and the port) up."""
    try:
        await coro
    except asyncio.CancelledError:
        raise
    except Exception as e:
        http.drop(name)
        logger.error("%s stopped (%s); other platforms keep running", name, safe_exc(e))


async def run_line(a: Assistant, ps: PlatformSettings, http: PlatformsHttp | None = None) -> None:
    import uvicorn

    from . import line

    api = line.SdkLineApi(ps.line_channel_access_token)
    info = await asyncio.to_thread(api.bot_info)
    port = line.LinePort(api, ps)
    _changes_pusher(a, "line", info["user_id"], change_outbox_platform.line_pusher(api))
    adapter = line.LineAdapter(a, port, info["user_id"])
    app = line.create_app(ps.line_channel_secret, adapter)
    if http is not None:
        http.register(line.WEBHOOK_PATH, app, "line")
        logger.info("line webhook ready tenant=%d", ids.tenant_id("line", info["user_id"]))
        await asyncio.Event().wait()
        return
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


async def run_whatsapp(
    a: Assistant, ps: PlatformSettings, http: PlatformsHttp | None = None
) -> None:
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
        if http is not None:
            http.register(whatsapp.WEBHOOK_PATH, app, "whatsapp")
            logger.info(
                "whatsapp webhook ready tenant=%d",
                ids.tenant_id("whatsapp", ps.whatsapp_phone_number_id),
            )
            await asyncio.Event().wait()
            return
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
    port = ps.combined_port
    http = PlatformsHttp() if port else None
    jobs: list[asyncio.Task] = []
    if http is not None:  # first: Cloud Run needs the port open before the slow startups finish
        up = asyncio.Event()
        jobs.append(asyncio.create_task(serve_http(http, port, up)))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(up.wait(), 15)
    a = build_assistant(s, ps)
    jobs.append(asyncio.create_task(memory_loop(a)))
    mon = platform_monitor(s, a)  # flag SALID_MONITOR_ENABLED and an org store
    if mon is not None:
        jobs.append(asyncio.create_task(monitor_loop(mon)))
    if a.identity is not None:  # the org store exists: central changes can be delivered
        from ..change_outbox import ChangeOutbox

        outbox = ChangeOutbox.from_settings(
            a.identity.orgs.docs, s, on_change=change_events(mon)
        )  # each stored / updated change is a monitor event (when the monitor is on)
        a.changes = change_outbox_platform.PlatformChanges(outbox, a.identity.orgs, a.team)
        jobs.append(asyncio.create_task(change_outbox_platform.changes_loop(a.changes)))

    def launch(name: str, coro_fn, *args) -> None:
        if http is None:
            jobs.append(asyncio.create_task(coro_fn(a, ps, *args)))
            return
        if name in ("slack", "telegram"):
            http.platforms.append(name)
            coro = coro_fn(a, ps)
        else:
            coro = coro_fn(a, ps, http)
        jobs.append(asyncio.create_task(_guarded(name, coro, http)))

    if ps.slack_enabled:
        launch("slack", run_slack)
    if ps.line_enabled:
        launch("line", run_line)
    if ps.telegram_enabled:
        launch("telegram", run_telegram)
    if ps.whatsapp_enabled:
        launch("whatsapp", run_whatsapp)
    logger.info("platforms started: %s", ", ".join(ps.enabled_platforms()))
    await asyncio.gather(*jobs)
