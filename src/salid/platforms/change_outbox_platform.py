"""The change outbox on Slack / LINE / Telegram / WhatsApp (logic and store: `salid.change_outbox`).

One generic consumer for the platform process. Each platform runtime registers a *pusher* for its
tenant (`add_pusher`): an async function `(person, text) -> message ref` that sends a 1:1 message.
Every 30 s `tick`:
- `apply` target of a served workspace: no adapter can change anything on its platform yet, so it
  is acked at once as `nothing_to_apply`;
- `notify` target: the notice goes to the person with "reply OK to confirm"; a reply (see
  `on_message`, called by `Assistant` before anything else in a 1:1) acks it.
A workspace nobody serves (platform not configured) keeps its targets pending; they fail after the
deadline and the alert below says so. Admins on a served platform get one alert per failed change.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from ..change_outbox import (
    ChangeOutbox,
    admins_of,
    alert_text,
    change_text,
    is_ack_text,
)
from ..i18n import base_lang, resolve_lang, t
from ..orgs import OrgStore
from . import ids
from .core import safe_exc

logger = logging.getLogger(__name__)

TICK_S = 30.0
Pusher = Callable[[str, str], Awaitable[str]]


class PlatformChanges:
    def __init__(self, outbox: ChangeOutbox, orgs: OrgStore, team) -> None:
        self.outbox, self.orgs, self.team = outbox, orgs, team
        self._pushers: dict[tuple[str, str], Pusher] = {}

    def add_pusher(self, platform: str, tenant: str, push: Pusher) -> None:
        self._pushers[(platform, str(tenant))] = push

    def _served(self, workspace_id: str, platform: str) -> tuple[str, Pusher] | None:
        key = workspace_id[len(platform) + 1 :]
        push = self._pushers.get((platform, key))
        return (key, push) if push else None

    def _lang(self, platform: str, tenant: str) -> str:
        try:
            gid = ids.tenant_id(platform, tenant)
            return base_lang(resolve_lang(self.team.current(gid).default_language))
        except Exception:
            return "en"

    # ------------------------------------------------------------- consumer
    async def tick(self) -> None:
        if not self._pushers:
            return
        try:
            await self._run()
        except Exception as e:
            logger.warning("change outbox: platform tick failed (%s)", safe_exc(e))

    async def _run(self) -> None:
        outbox = self.outbox
        await asyncio.to_thread(outbox.prune)
        names: dict[str, str] = {}
        for change, target in await asyncio.to_thread(outbox.due):
            served = self._served(target.workspace_id, target.platform)
            if served is None:
                continue  # not configured here: stays pending, fails at the deadline
            tenant, push = served
            try:
                if target.kind == "apply":
                    await asyncio.to_thread(
                        outbox.ack, change.id, target.workspace_id, "", "nothing_to_apply"
                    )
                    continue
                if change.org_id not in names:
                    org = await asyncio.to_thread(self.orgs.get, change.org_id)
                    names[change.org_id] = getattr(org, "name", "") or ""
                lang = self._lang(target.platform, tenant)
                text = change_text(change, names[change.org_id], lang, target.attempts)
                text = f"{text}\n{t('chg_hint', lang)}"
                ref = await push(target.person, text)
                await asyncio.to_thread(
                    outbox.mark_attempt,
                    change.id,
                    target.workspace_id,
                    target.person,
                    message_ref=str(ref or ""),
                )
            except Exception as e:
                logger.warning("change outbox: send failed (%s)", safe_exc(e))
                await asyncio.to_thread(
                    outbox.mark_attempt,
                    change.id,
                    target.workspace_id,
                    target.person,
                    error=type(e).__name__,
                )
        await self._alerts()

    async def _alerts(self) -> None:
        outbox = self.outbox
        for platform in sorted({p for p, _ in self._pushers}):
            for change in await asyncio.to_thread(outbox.to_alert, platform):
                org = await asyncio.to_thread(self.orgs.get, change.org_id)
                if org is None:
                    continue
                for w in org.workspaces:
                    served = self._served(w.id, w.platform) if w.platform == platform else None
                    if served is None:
                        continue
                    tenant, push = served
                    text = alert_text(
                        change, org.name, self._lang(platform, tenant), outbox.max_attempts
                    )
                    for m in admins_of(org):
                        uid = m.identity(platform)
                        if not uid:
                            continue
                        try:
                            await push(uid, text)
                        except Exception as e:
                            logger.info("change outbox: alert not sent (%s)", safe_exc(e))
                await asyncio.to_thread(outbox.mark_alerted, change.id, platform)

    # ------------------------------------------------------------- acknowledgement
    async def on_message(self, msg) -> str | None:
        """A 1:1 message "รับทราบ / ok": acks the person's unconfirmed notices. Returns the thanks
        to send back, or None when this is not an acknowledgement (the message goes on)."""
        if not is_ack_text(getattr(msg, "text", "") or ""):
            return None
        ws = f"{msg.platform}-{msg.tenant}"
        open_ = await asyncio.to_thread(self.outbox.open_for, ws, str(msg.user))
        if not open_:
            return None
        for change, _target in open_:
            await asyncio.to_thread(self.outbox.ack, change.id, ws, str(msg.user), "reply")
        return t("chg_thanks", self._lang(msg.platform, msg.tenant))


async def changes_loop(changes: PlatformChanges) -> None:
    while True:
        await asyncio.sleep(TICK_S)
        await changes.tick()


# ------------------------------------------------------------------ pushers (one per platform)
def slack_pusher(port) -> Pusher:
    async def push(user: str, text: str) -> str:
        opened = await port._call(port.client.conversations_open, users=user)
        channel = (opened.get("channel") or {}).get("id")
        if not channel:
            raise RuntimeError("no DM channel")
        sent = await port._call(port.client.chat_postMessage, channel=channel, text=text)
        return str(sent.get("ts") or "")

    return push


def line_pusher(api) -> Pusher:
    async def push(user: str, text: str) -> str:
        sent = await asyncio.to_thread(api.push, user, [text])
        return str((sent or [""])[0])

    return push


def telegram_pusher(api) -> Pusher:
    async def push(user: str, text: str) -> str:
        sent = await api.send_message({"chat_id": int(user), "text": text})
        return str((sent or {}).get("message_id") or "")

    return push


def whatsapp_pusher(api) -> Pusher:
    async def push(user: str, text: str) -> str:
        return str(await api.send_text(user, text, None) or "")

    return push
