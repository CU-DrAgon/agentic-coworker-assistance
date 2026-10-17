"""PM dashboard projects, Discord side (logic in `projects.py`, web API in `pm_api.py`).

`ProjectsMixin` gives `SecretaryBot` the agent tools (`_project_tools`, appended to `extra_tools`
by `_agent_kwargs`, so chat and voice share them). Projects of a Discord server live under its
guild id, next to its tasks, so the web dashboard and Discord see the same ones. The asker is a
person ref built from the Discord member (plus the email of the org member linked to that Discord
id, when there is one); admin = a Salid executive or an org owner / admin. Nothing here is chosen
by the model.
"""

from __future__ import annotations

import logging
import re

from .mentions import match_members
from .orgs import OrgStore
from .projects import Person, ProjectRepo, ProjectsCtx, project_tools
from .tasks import TaskRepo

logger = logging.getLogger(__name__)

_RAW_MENTION = re.compile(r"<@[!&]?(\d+)>")


class ProjectsMixin:
    def _projects_init(self) -> None:
        """Call after `_coo_init` (needs `self.docs`)."""
        self.project_repo = ProjectRepo(self.docs, clock=self.s.now)
        self.project_task_repo = TaskRepo(self.docs)

    def _projects_on(self, guild_id: int) -> bool:
        return self.team.enabled(guild_id, "projects")

    def _project_tools(self, message, guild, asker, is_exec: bool, lang: str) -> list:
        """The project tools for one request (chat or voice). [] when the flag is off; never
        raises."""
        try:
            if not self._projects_on(guild.id):
                return []
            org = OrgStore(self.docs).org_of_guild(guild.id)
            emails = {
                m.discord_user_id: m for m in (org.members if org else []) if m.discord_user_id
            }

            def person(member) -> Person:
                known = emails.get(str(member.id))
                return Person(
                    discord_user_id=str(member.id),
                    email=known.email if known else None,
                    name=getattr(member, "display_name", "") or str(member.id),
                )

            def resolve(name: str) -> list[Person]:
                m = _RAW_MENTION.fullmatch(name.strip())
                if m:
                    who = guild.get_member(int(m.group(1)))
                    return (
                        [person(who)] if who is not None and not getattr(who, "bot", False) else []
                    )
                return [person(x) for x in match_members(guild, name)]

            me = person(asker)
            role = org.role_of(me.email) if (org and me.email) else None
            ctx = ProjectsCtx(
                guild_id=guild.id,
                asker=me,
                is_admin=bool(is_exec or role in ("owner", "admin")),
                lang=lang,
                repo=self.project_repo,
                tasks=self.project_task_repo,
                now=self.s.now,
                resolve=resolve,
                workspace=f"discord-{guild.id}",
            )
            return project_tools(ctx)
        except Exception:
            logger.warning("project tools unavailable", exc_info=True)
            return []
