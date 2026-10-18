"""SlackWorkspace over a fake Web API client (the FakeSlack of test_slack, extended)."""

import pytest
from platform_fakes import Clock
from test_slack import BOT, TEAM, FakeSlack

from salid.platforms import ids
from salid.platforms.slack import SlackPort
from salid.platforms.workspace_slack import UPLOAD_LIMIT, SlackWorkspace, blocks
from salid.workspace import (
    CAPABILITIES,
    EVENTS,
    VOICE,
    Button,
    FileOut,
    MsgRef,
    WorkspaceError,
    button_id,
)


class WsSlack(FakeSlack):
    def __init__(self):
        super().__init__()
        self.users["U1"]["profile"]["email"] = "alice@acme.co"
        self.users["U4"] = {"id": "U4", "team_id": TEAM, "real_name": "Dan", "profile": {}}
        self.users["U5"] = {"id": "U5", "team_id": TEAM, "real_name": "Dan", "profile": {}}
        self.users["USLACKBOT"] = {"id": "USLACKBOT", "team_id": TEAM, "name": "slackbot"}
        self.channels["C9"] = {
            "id": "C9",
            "name": "town",
            "is_private": False,
            "members": ["U1", BOT],
        }
        self.channels["C8"] = {
            "id": "C8",
            "name": "partners",
            "is_private": True,
            "is_ext_shared": True,
            "members": ["U1", "UX", BOT],
        }
        self.uploads: list[dict] = []
        self.opened: list[str] = []
        self.user_pages = 2

    async def users_list(self, limit=200, cursor=None):
        rows = list(self.users.values())
        size = max(len(rows) // self.user_pages, 1)
        start = int(cursor or 0)
        nxt = start + size
        return {
            "members": rows[start:nxt],
            "response_metadata": {"next_cursor": str(nxt) if nxt < len(rows) else ""},
        }

    async def users_conversations(
        self, user=None, types=None, exclude_archived=None, limit=200, cursor=None
    ):
        who = user or BOT
        rows = [
            {
                **{k: False for k in ("is_shared", "is_ext_shared", "is_org_shared")},
                **{k: v for k, v in c.items() if k != "members"},
            }
            for c in self.channels.values()
            if who in c["members"]
        ]
        return {"channels": rows, "response_metadata": {"next_cursor": ""}}

    async def conversations_open(self, users):
        self.opened.append(users)
        return {"channel": {"id": "D" + users}}

    async def files_upload_v2(self, **kw):
        self.uploads.append(kw)
        return {"ok": True}


@pytest.fixture
def fx():
    client = WsSlack()
    port = SlackPort(client, TEAM, BOT, clock=Clock(), sleep=_nosleep)
    return client, SlackWorkspace(port)


async def _nosleep(_s):
    return None


def uid(u):
    return ids.user_id("slack", u)


def rid(c):
    return ids.room_id("slack", TEAM, c)


def test_identity_ids_and_capabilities(fx):
    _, ws = fx
    assert ws.platform == "slack" and ws.tenant_id == ids.tenant_id("slack", TEAM)
    assert ws.upload_limit == UPLOAD_LIMIT == 100 * 1024 * 1024
    assert not ws.supports(VOICE) and not ws.supports(EVENTS)
    assert all(ws.supports(f) for f in CAPABILITIES if f not in (VOICE, EVENTS))


async def test_people_paginates_and_drops_bots_deleted_and_outsiders(fx):
    client, ws = fx
    ps = await ws.people()
    assert {p.platform_id for p in ps} == {"U1", "U2", "U3", "U4", "U5"}
    assert all(p.id == uid(p.platform_id) for p in ps)
    assert next(p for p in ps if p.platform_id == "U1").email == "alice@acme.co"
    assert next(p for p in ps if p.platform_id == "U2").name == "Bob Lee"
    await ws.people()  # cached: users.list is not called again
    assert client.user_pages == 2 and "users_list" not in client.calls


async def test_person_by_mention_id_name_and_email(fx):
    _, ws = fx
    assert (await ws.person("<@U1>")).name == "Alice"
    assert (await ws.person("<@U1|alice>")).platform_id == "U1"
    assert (await ws.person("U2")).name == "Bob Lee"
    assert (await ws.person("@alice")).platform_id == "U1"  # display name, case-insensitive
    assert (await ws.person("bob lee")).platform_id == "U2"  # real name
    assert (await ws.person("eve")).platform_id == "U3"  # username
    assert (await ws.person("Alice@Acme.co")).platform_id == "U1"  # email
    assert (await ws.person("U1")).id == uid("U1")


async def test_person_refuses_ambiguous_bots_outsiders_and_unknowns(fx):
    _, ws = fx
    assert await ws.person("dan") is None  # two people called Dan: never guessed
    assert await ws.person("UN") is None  # no team_id on the record: fail closed
    assert await ws.person("<@UB>") is None  # a bot
    assert await ws.person("<@UD>") is None  # deleted
    assert await ws.person("<@UX>") is None  # another workspace (Slack Connect)
    assert await ws.person("<@U999>") is None and await ws.person("") is None


async def test_exec_and_rank_follow_the_port_admin_rule(fx):
    _, ws = fx
    await ws.people()
    assert await ws.is_exec(uid("U2")) and await ws.rank(uid("U2")) == 1
    assert not await ws.is_exec(uid("U1")) and await ws.rank(uid("U1")) == 0
    assert not await ws.is_exec(12345) and await ws.rank(12345) == 0
    assert not await ws.is_exec(uid("UX"))  # never learned: refused


async def test_rooms(fx):
    _, ws = fx
    rooms = {r.name: r for r in await ws.rooms()}
    assert set(rooms) == {"general", "leads", "secret", "town", "partners"}
    assert rooms["general"].id == rid("C1") and rooms["general"].private
    assert not rooms["town"].private and not rooms["town"].shared
    assert rooms["partners"].shared and rooms["partners"].platform_id == "C8"


async def test_room_view_private_exact_public_and_shared_unknown(fx):
    _, ws = fx
    await ws.rooms()
    v = await ws.room_view(rid("C1"))  # private, not shared: exact members (bot removed)
    assert v.audience == frozenset({uid("U1"), uid("U2")})
    names = {r.name: r.viewers for r in v.rooms}
    assert names["leads"] == frozenset({uid("U1"), uid("U2"), uid("U3")})
    assert "partners" not in names and "general" not in names  # shared / the room itself
    assert "town" not in names  # a public room's readers are unknown: never in cross-room scope
    assert (await ws.room_view(rid("C9"))).audience is None  # public: unknown
    assert (await ws.room_view(rid("C8"))).audience is None  # Slack Connect: unknown
    assert (await ws.room_view(123)).audience is None  # never seen


async def test_room_view_errors_fail_closed(fx):
    client, ws = fx
    await ws.rooms()

    async def boom(**kw):
        raise RuntimeError("ratelimited")

    client.conversations_members = boom
    assert (await ws.room_view(rid("C1"))).audience is None


async def test_post_with_buttons_in_a_thread_and_file(fx):
    client, ws = fx
    await ws.rooms()
    btn = Button(button_id("task", "done", ws.tenant_id, "t1"), "Done", "success")
    ref = await ws.post(
        rid("C1"),
        "**hi** <@U2>",
        thread=MsgRef("slack", rid("C1"), "50.1"),
        files=[FileOut("r.pdf", b"%PDF", "application/pdf")],
        buttons=[btn, Button("salid:task:no:1:t1", "No", "danger"), Button("salid:a:b:1:c", "Meh")],
    )
    assert ref == MsgRef("slack", rid("C1"), "91.000")
    (p,) = client.posts
    assert p["channel"] == "C1" and p["thread_ts"] == "50.1" and "hi" in p["text"]
    section, actions = p["blocks"]
    assert section["type"] == "section"
    els = actions["elements"]
    assert [e["action_id"] for e in els] == [btn.id, "salid:task:no:1:t1", "salid:a:b:1:c"]
    assert [e.get("style") for e in els] == ["primary", "danger", None]
    (up,) = client.uploads
    assert up["channel"] == "C1" and up["thread_ts"] == "50.1"
    assert up["file_uploads"][0]["filename"] == "r.pdf" and up["file_uploads"][0]["file"] == b"%PDF"


async def test_post_without_extras_and_long_text(fx):
    client, ws = fx
    await ws.rooms()
    await ws.post(rid("C1"), "plain")
    assert "blocks" not in client.posts[0] and "thread_ts" not in client.posts[0]
    assert client.uploads == []
    await ws.post(rid("C1"), "\n".join(["y" * 1500] * 5), buttons=[Button("salid:a:b:1:c", "Go")])
    long = client.posts[1:]
    assert len(long) >= 3
    assert all("blocks" not in p for p in long[:-1]) and "blocks" in long[-1]


async def test_post_unknown_room_and_bad_buttons(fx):
    _, ws = fx
    with pytest.raises(WorkspaceError):
        await ws.post(999, "x")
    await ws.rooms()
    with pytest.raises(WorkspaceError):
        await ws.post(rid("C1"), "x", buttons=[Button("i", "L", "weird")])


async def test_post_after_uninstall_raises(fx):
    _, ws = fx
    await ws.rooms()
    ws.port.revoked = True
    with pytest.raises(WorkspaceError):
        await ws.post(rid("C1"), "x")


async def test_dm_opens_a_conversation_first(fx):
    client, ws = fx
    await ws.people()
    ref = await ws.dm(uid("U3"), "psst", buttons=[Button("salid:a:b:1:c", "Go")])
    assert client.opened == ["U3"]
    assert client.posts[0]["channel"] == "DU3" and "blocks" in client.posts[0]
    assert ref.room_id == rid("DU3") and ref.platform == "slack"
    assert await ws.dm(777, "x") is None  # unknown person


async def test_dm_failure_returns_none(fx):
    client, ws = fx
    await ws.people()

    async def closed(users):
        raise RuntimeError("cannot_dm_bot")

    client.conversations_open = closed
    assert await ws.dm(uid("U3"), "x") is None


async def test_edit_always_sends_blocks_so_buttons_can_be_removed(fx):
    client, ws = fx
    await ws.rooms()
    ref = MsgRef("slack", rid("C1"), "91.000")
    await ws.edit(ref, "done")
    (u,) = client.updates
    assert u["channel"] == "C1" and u["ts"] == "91.000"
    assert [b["type"] for b in u["blocks"]] == ["section"]  # no actions block: buttons removed
    await ws.edit(ref, "again", buttons=[Button("salid:a:b:1:c", "Go")])
    assert [b["type"] for b in client.updates[1]["blocks"]] == ["section", "actions"]


async def test_mention_pings_only_through_the_sentinel_of_a_local_human(fx):
    client, ws = fx
    await ws.people()
    await ws.rooms()
    assert ws.mention(5) == "@someone"
    assert ws.mention(uid("UX")) == "@someone"  # never a local human
    await ws.post(
        rid("C1"),
        f"hi {ws.mention(uid('U2'))} and <@U3> <!channel> <!here> <!everyone> <@UX>",
    )
    text = client.posts[0]["text"]
    assert "hi <@U2> and" in text  # Salid's own mention pings
    assert "&lt;@U3&gt;" in text and "&lt;@UX&gt;" in text  # user-supplied raw tokens do not
    for tok in ("channel", "here", "everyone"):
        assert f"&lt;!{tok}&gt;" in text and f"<!{tok}>" not in text
    await ws.edit(MsgRef("slack", rid("C1"), "91.000"), f"{ws.mention(uid('U2'))} <@U3>")
    assert client.updates[0]["text"] == "<@U2> &lt;@U3&gt;"
    await ws.dm(uid("U3"), f"psst {ws.mention(uid('U1'))}")
    assert "<@U1>" in client.posts[1]["text"]


async def test_a_sentinel_for_an_unknown_id_is_not_a_ping(fx):
    client, ws = fx
    await ws.rooms()
    await ws.post(rid("C1"), "x \ue000UX\ue001 y")
    assert "<@UX>" not in client.posts[0]["text"]


async def test_omitted_share_flags_make_the_audience_unknown(fx):
    client, ws = fx
    await ws.rooms()
    client.omit_share_fields = {"is_org_shared"}  # Slack left one flag out: not provably private
    assert (await ws.room_view(rid("C1"))).audience is None
    client.omit_share_fields = {"is_shared", "is_ext_shared", "is_org_shared"}
    assert (await ws.room_view(rid("C1"))).audience is None
    client.omit_share_fields = set()
    ws.port.forget_room("C1")
    assert (await ws.room_view(rid("C1"))).audience == frozenset({uid("U1"), uid("U2")})


async def test_other_rooms_with_omitted_or_set_share_flags_are_excluded(fx):
    client, ws = fx
    await ws.rooms()
    client.channels["C2"]["is_org_shared"] = True  # leads: shared

    client.omit_share_fields = {"is_ext_shared"}  # ... and every conversations.info omits one
    view = await ws.room_view(rid("C1"))
    assert view.audience is None  # even the answering room is no longer provable
    client.omit_share_fields = set()
    ws.port.forget_room("C1")
    view = await ws.room_view(rid("C1"))
    assert {r.name for r in view.rooms} == {"secret"}  # leads is shared, town public, partners ext


async def test_rooms_with_missing_share_flags_are_reported_shared(fx):
    client, ws = fx
    rows = await ws.rooms()
    assert not any(r.shared for r in rows if r.name != "partners")

    async def bare(**kw):
        return {"channels": [{"id": "C1", "name": "general", "is_private": True}]}

    client.users_conversations = bare
    ws._rooms = None
    assert (await ws.rooms())[0].shared is True  # flags omitted: unknown, treated as shared


async def test_group_dm_and_dm_need_explicit_flags_and_validated_members(fx):
    client, ws = fx
    client.channels["G1"] = {
        "id": "G1",
        "name": "mpdm-u1--u2",
        "is_private": True,
        "is_mpim": True,
        "members": ["U1", "U2", BOT],
    }
    client.channels["G2"] = {**client.channels["G1"], "id": "G2", "members": ["U1", "UX", BOT]}
    client.channels["D1"] = {
        "id": "D1",
        "name": "",
        "is_im": True,
        "user": "U2",
        "members": ["U2", BOT],
    }
    client.channels["D2"] = {
        "id": "D2",
        "name": "",
        "is_im": True,
        "user": "UB",
        "members": ["UB", BOT],
    }
    client.channels["D3"] = {"id": "D3", "name": "", "user": "U2", "members": ["U2", BOT]}
    for c in ("G1", "G2", "D1", "D2", "D3"):
        ws.learn_room(c)
    assert (await ws.room_view(rid("G1"))).audience == frozenset({uid("U1"), uid("U2")})
    assert (await ws.room_view(rid("G2"))).audience is None  # an outsider is a member
    assert (await ws.room_view(rid("D1"))).audience == frozenset({uid("U2")})
    assert (await ws.room_view(rid("D2"))).audience is None  # a bot peer
    assert (await ws.room_view(rid("D3"))).audience is None  # not explicitly an im


async def test_name_collisions_across_alias_fields_are_never_guessed(fx):
    client, ws = fx
    client.users["U6"] = {
        "id": "U6",
        "team_id": TEAM,
        "real_name": "Zed",
        "name": "mia",
        "profile": {},
    }
    client.users["U7"] = {
        "id": "U7",
        "team_id": TEAM,
        "real_name": "Mia Q",
        "name": "mq",
        "profile": {"display_name": "mia"},
    }
    assert await ws.person("mia") is None  # A's username == B's display name
    assert (await ws.person("zed")).platform_id == "U6"
    assert (await ws.person("mq")).platform_id == "U7"


async def test_a_failed_explicit_id_never_falls_back_to_names(fx):
    client, ws = fx
    client.users["U8"] = {
        "id": "U8",
        "team_id": TEAM,
        "real_name": "U404",
        "name": "x",
        "profile": {},
    }
    assert await ws.person("U404") is None  # users.info fails; not matched by real name
    assert await ws.person("<@U404>") is None


def test_blocks_truncates_label_and_skips_empty_actions():
    assert [b["type"] for b in blocks("hi", [])] == ["section"]
    b = blocks("", [Button("id", "L" * 200)])
    assert len(b[1]["elements"][0]["text"]["text"]) == 75 and b[0]["text"]["text"] == " "
