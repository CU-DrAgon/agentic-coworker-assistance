"""Who a voice agent acts for. In a room with several people the whole agent is built for the
LOWEST-ranked human present, so every READ (tasks, approvals, meetings, reports, memory) is safe to
say out loud to everyone. Only the outward ACTIONS (scheduling, DMs, posting, approvals, polls...)
are then re-bound to the person who actually asked, with their own identity and rights.

`rebind_actions(room_kwargs, speaker_kwargs, names)` swaps tools by function name, so a new
feature's action tool only needs its name added to ACTION_TOOL_NAMES."""

from __future__ import annotations

from collections.abc import Iterable

# Function names (as the model sees them) of tools that DO something for the asker. Reads are not
# listed on purpose: they stay bound to the room.
ACTION_TOOL_NAMES: frozenset[str] = frozenset(
    {
        # secretary
        "schedule_meeting",
        "cancel_meeting",
        "reschedule_meeting",
        # chat actions
        "send_dm",
        "deliver_minutes",
        "set_my_notifications",
        # coordinator
        "update_task",
        "request_disclosure",
        # feature branches (approvals, broadcast, freetime, polls/forms, captions)
        "request_approval",
        "cancel_approval",
        "attach_receipt_to_approval",
        "send_dm_many",
        "post_message",
        "set_availability",
        "create_poll",
        "start_form",
        "set_captions",
        "set_captions_tool",
        # found by the builder-parity test: also act with the asker's identity and rights.
        # Not wiki_write / wiki_publish: they read memory and return page text to the model, so in
        # a shared room they must stay at the room's scope.
        "send_email",
        "draft_email",
        "deliver_file",
        "remember_my_preference",
        "forget_my_preferences",
        "set_auto_translate",
        "wiki_delete",
        "cancel_deep_task",
        "set_channel_language",
        "set_server_language",
        "set_channel_mode",
        "set_memory_days",
        "remember_place",
        "share_place",
        "forget_place",
        "set_place_secret",
        "track_flight",
        "market_watch",
        "market_brief",
        "track_news_topic",
        "idea_save",
        "idea_export",
        "idea_share",
        "business_framework",
        "teach_company",
        "propose_lesson",
        "forget_company_item",
        "review_suggestion",
        "company_domain",
        "contest_add",
        "contest_update",
        "contest_share",
        # bizfin: calculators can save into the asker's idea; verify_slip reads the asker's ledger
        "unit_economics",
        "price_from_cost",
        "investment_metrics",
        "financial_projection",
        "verify_slip",
    }
)


def speaker_tools(speaker_kwargs: dict) -> list:
    """Every tool the agent built from `speaker_kwargs` (an `_agent_kwargs` result) would get from
    the secretary, chat-actions and extra-tools parts."""
    from .chat_tools import build_chat_tools
    from .secretary import SecretaryTools, build_secretary_tools

    tools: list = []
    if speaker_kwargs.get("secretary") is not None:
        tools += build_secretary_tools(SecretaryTools(speaker_kwargs["secretary"]), [])
    if speaker_kwargs.get("chat_actions") is not None:
        tools += build_chat_tools(speaker_kwargs["chat_actions"], [])
    tools += list(speaker_kwargs.get("extra_tools") or [])
    return tools


def rebind_actions(
    room_kwargs: dict, speaker_kwargs: dict, names: Iterable[str] = ACTION_TOOL_NAMES
) -> dict:
    """`room_kwargs` plus `tool_overrides`: the speaker's version of each action tool in `names`.
    `ask` replaces the same-named tools of the room agent with them (it never adds one the room
    agent does not have, so feature switches keep working)."""
    wanted = set(names)
    overrides = [f for f in speaker_tools(speaker_kwargs) if getattr(f, "__name__", "") in wanted]
    return {**room_kwargs, "tool_overrides": overrides}
