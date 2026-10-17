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
            {k: v for k, v in c.items() if k != "members"}
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
    assert "town" in names  # public rooms are listed with their (partial) members
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


async def test_mention(fx):
    _, ws = fx
    await ws.people()
    assert ws.mention(uid("U2")) == "<@U2>"
    assert ws.mention(5) == "@someone"


def test_blocks_truncates_label_and_skips_empty_actions():
    assert [b["type"] for b in blocks("hi", [])] == ["section"]
    b = blocks("", [Button("id", "L" * 200)])
    assert len(b[1]["elements"][0]["text"]["text"]) == 75 and b[0]["text"]["text"] == " "
