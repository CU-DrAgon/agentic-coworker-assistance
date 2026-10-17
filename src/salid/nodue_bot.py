"""Tasks without a deadline are not forgotten: `NoDeadlineMixin` asks the assignee ONCE.

`_nodue_due` is a tick (see `tasks.no_deadline_due`): an open task with an assignee and no due date
that is 3 working days old gets one private message to the assignee asking how it is going and how
to give a deadline. The giver is not told (the daily report lists such tasks).
"""

from __future__ import annotations

import asyncio
import logging

from .hints import with_hint
from .i18n import CATALOG, t
from .tasks import no_deadline_due

logger = logging.getLogger(__name__)

NODUE_CHECK_S = 3600.0

CATALOG.update(
    {
        "nodue_ask": {
            "th": "🗓️ งาน “{title}” (#{id}) ยังไม่มีกำหนด คืบหน้าไหมคะ ถ้าพอบอกวันได้ พิมพ์ “งาน #{id} กำหนด <วัน>” ได้เลยค่ะ",
            "en": "🗓️ Task “{title}” (#{id}) has no deadline yet — how is it going? If you can, type “task #{id} due <day>”.",
        },
    }
)


class NoDeadlineMixin:
    async def _nodue_due(self, now=None) -> int:
        """Send the one-time "no deadline yet" question; returns how many were delivered."""
        if (
            not self.guilds
            or self._clock() - getattr(self, "_nodue_last", float("-inf")) < NODUE_CHECK_S
        ):
            return 0
        self._nodue_last = self._clock()
        now = now or self.s.now()
        sent = 0
        for guild in list(self.guilds):
            gid = guild.id
            try:
                if not self._handshake_on(gid):
                    continue
                tasks = await asyncio.to_thread(self.tasks.list, gid, status="open")
                for task in no_deadline_due(tasks, now.astimezone(self.s.zone).date(), self.s.zone):
                    member = guild.get_member(task.assignee_id)
                    if member is None or getattr(member, "bot", False):
                        continue
                    if not await asyncio.to_thread(self._claim, gid, f"nodue:{task.id}"):
                        continue
                    lang = self._person_lang(gid, member.id)
                    text = with_hint(
                        t("nodue_ask", lang, title=task.title, id=task.id), "due_formats", lang=lang
                    )
                    res = await self.deliver_to_person(guild, member, text)
                    if not res.ok:
                        continue
                    task.nodue_asked_at = now.isoformat(timespec="seconds")
                    await asyncio.to_thread(self.tasks.save, task)
                    sent += 1
            except Exception:
                logger.exception("no-deadline question for guild %s failed", gid)
        return sent
