"""ACTION_TOOL_NAMES must match the real tool function names, or a renamed tool silently stays
bound to the room's lowest-ranked person instead of the speaker."""

from __future__ import annotations

from types import SimpleNamespace

from test_modes import bot  # noqa: F401  (fixture)
from test_wiki import STAFF, world  # noqa: F401  (fixture)

from salid.voice_actions import ACTION_TOOL_NAMES, speaker_tools

# Listed on purpose but only built in some situations (voice-side name, or a setup we do not make).
OPTIONAL = {"set_captions", "set_captions_tool", "deliver_minutes"}


async def test_every_action_tool_name_exists_in_the_chat_builder(bot, world):  # noqa: F811
    member = world.g.members[STAFF]
    mem = await bot._memory_tools(SimpleNamespace(guild=None, channel=None), world.g, member)
    msg = bot.room_message(world.g, member, world.pub)
    kw = bot._agent_kwargs(
        msg, world.g, member, True, setting="th", lang="th", question="q", memory=mem,
        profile=None,
    )  # fmt: skip
    names = {getattr(f, "__name__", "") for f in speaker_tools(kw)}
    assert {"list_tasks", "find_free_time", "wiki_search"} <= names  # the builder is complete
    missing = ACTION_TOOL_NAMES - names - OPTIONAL
    assert not missing, missing
