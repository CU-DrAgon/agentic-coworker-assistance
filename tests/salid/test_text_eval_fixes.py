# ruff: noqa: F811, F401
"""Offline tests for fixes found by the live usability evaluation (tests/salid/live_eval)."""

from test_bot_multitenant import ME, Msg, member
from test_modes import bot, world  # noqa: F401  (fixtures)

from salid import qa
from salid.chat_tools import ChatActions, build_chat_tools


async def test_other_peoples_tags_become_names_before_the_model_sees_them(bot, world):
    jemm = member(world.g, 77)
    jemm.display_name = "Jemm"
    sender = member(world.g, 78)
    m = Msg(
        world.g, world.g.text_channels[0], sender, "<@999> ฝากบอก <@77> ทาง DM ว่าส่งงานด้วย <@!78>"
    )
    m.mentions = [ME, jemm]
    q = bot._name_mentions(bot._strip_mentions(m.content, m), m)
    assert q == "ฝากบอก @Jemm ทาง DM ว่าส่งงานด้วย @u78"  # the guild member lookup is the fallback
    assert "<@" not in q


async def test_unknown_ids_and_salids_own_tag_are_left_alone(bot, world):
    m = Msg(world.g, world.g.text_channels[0], member(world.g, 78), "<@555> hi <@999>")
    assert bot._name_mentions(m.content, m) == "<@555> hi <@999>"


def test_dm_tool_says_not_to_broadcast():
    tools = {f.__name__: f for f in build_chat_tools(ChatActions(send_dm=lambda *a: {}), [])}
    assert "never DM" in tools["send_dm"].__doc__ and "credentials" in tools["send_dm"].__doc__


def test_capabilities_answer_is_told_to_stay_short():
    assert "3-4 ประโยค" in qa.CAPABILITIES_PROMPT
