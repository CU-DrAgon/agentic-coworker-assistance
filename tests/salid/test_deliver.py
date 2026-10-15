# ruff: noqa: F811, F401
"""deliver_file: fuzzy channel matching (Thai / English), posting with a polite note and the
original link, DM and email-draft delivery, access rules, the cross-audience confirmation gate,
deleted sources, mention resolution, the size fallback and the delivery log. Offline."""

import io
from types import SimpleNamespace

import discord
import pytest
from conftest import GUILD
from test_coordinator import ADMIN_ID, BOSS, INTERN, NS, OWNER_ID, PEER, STAFF, person, world
from test_modes import Chan, TMsg, bot  # noqa: F401  (fixture)

from salid.coo_store import DELIVERIES
from salid.deliver import consonant_skeleton, match_channels
from salid.i18n import t
from salid.secretary import SecretaryContext

SRC, DEST, ARCH, PRIV = 30, 31, 32, 33


class DChan:
    """A text channel with separate read / send permissions; `posts` keeps what Salid sent."""

    def __init__(self, id, name, readers=None, senders=None, bot_attach=True):
        self.id, self.name, self.readers, self.senders = id, name, readers, senders
        self.bot_attach, self.posts, self.mention, self.log = bot_attach, [], f"<#{id}>", []

    def permissions_for(self, m):
        if m.id == 999:  # Salid
            return SimpleNamespace(read_messages=True, read_message_history=True, view_channel=True,
                                   send_messages=True, attach_files=self.bot_attach)  # fmt: skip
        ok = self.readers is None or m.id in self.readers
        send = ok and (self.senders is None or m.id in self.senders)
        return SimpleNamespace(read_messages=ok, read_message_history=ok, view_channel=ok,
                               send_messages=send, attach_files=send)  # fmt: skip

    async def send(self, content=None, files=None, allowed_mentions=None, **kw):
        self.posts.append(SimpleNamespace(content=content, files=files, allowed=allowed_mentions))
        return SimpleNamespace(jump_url=f"https://discord.com/channels/{GUILD}/{self.id}/777")

    async def fetch_message(self, mid):
        raise discord.NotFound(SimpleNamespace(status=404, reason="x"), "gone")


class FAtt:
    def __init__(self, name="minutes.pdf", data=b"%PDF-1.4 hello", size=None):
        self.filename, self.data = name, data
        self.size, self.content_type = size if size is not None else len(data), "application/pdf"

    async def read(self):
        return self.data


def source_msg(channel, atts=None, author=PEER, mid=555):
    return SimpleNamespace(
        id=mid, channel=channel, author=SimpleNamespace(id=author, bot=False),
        attachments=[FAtt()] if atts is None else atts,
        jump_url=f"https://discord.com/channels/{GUILD}/{channel.id}/{mid}",
    )  # fmt: skip


@pytest.fixture
def jemm(world):
    return person(world.g, 40, "Jemm", 2)


@pytest.fixture
def chans(world):
    """The source channel (staff + peer + boss), the destination everybody reads and writes."""
    team = {STAFF, PEER, BOSS, OWNER_ID, ADMIN_ID, 40}
    src = DChan(SRC, "minutes-source", readers=set(team))
    dest = DChan(DEST, "paper-minute", readers=set(team))
    for c in (src, dest):
        c.guild = world.g
        world.g.text_channels.append(c)
    return SimpleNamespace(src=src, dest=dest)


def reply_to(world, chans, atts=None, author=STAFF, text="ส่งต่อให้หน่อย"):
    src = source_msg(chans.src, atts)
    req = TMsg(world.g, chans.src, world.g.members[author], text)
    req.attachments, req.reference = [], SimpleNamespace(resolved=src, message_id=src.id)
    req.author = world.g.members[author]
    return req, src


async def deliver(bot, world, req, who=STAFF, request_id="r1", secretary=None, **kw):
    kw.setdefault("note", "")
    kw.setdefault("source", "this")
    kw.setdefault("file_index", None)
    kw.setdefault("mention", [])
    return await bot._deliver_file(
        req, world.g, world.g.members[who], "th", secretary, request_id, **kw
    )


# ------------------------------------------------------------------ matching
def names(chs):
    return [c.name for c in chs]


def pool():
    return [SimpleNamespace(name=n) for n in ("general", "paper-minute", "ใบเสร็จ", "random")]


def test_match_channels_english_tiers():
    got, how = match_channels(pool(), "paper-minute")
    assert names(got) == ["paper-minute"] and how == "exact"
    got, how = match_channels(pool(), "#Paper Minute")
    assert names(got) == ["paper-minute"] and how == "exact"
    got, how = match_channels(pool(), "paper")
    assert names(got) == ["paper-minute"] and how == "token"
    got, how = match_channels(pool(), "ห้อง general")
    assert names(got) == ["general"]
    got, how = match_channels(pool(), "ใบเสร็จ")
    assert names(got) == ["ใบเสร็จ"] and how == "exact"
    assert match_channels(pool(), "nothing-like-it") == ([], "")
    assert match_channels(pool(), "  ") == ([], "")


def test_match_channels_thai_to_english_is_a_phonetic_guess():
    assert consonant_skeleton("เปเปอร์") == "pp" and consonant_skeleton("paper") == "ppr"
    got, how = match_channels(pool(), "ห้องเปเปอร์")
    assert names(got) == ["paper-minute"] and how == "phonetic"
    assert match_channels(pool(), "ห้องบัญชี") == ([], "")  # nothing close


def test_match_channels_ambiguous_returns_all_best():
    chs = [SimpleNamespace(name=n) for n in ("paper-minute", "paper-draft", "general")]
    assert names(match_channels(chs, "paper")[0]) == ["paper-minute", "paper-draft"]
    assert len(match_channels(chs, "ห้องเปเปอร์")[0]) == 2


# ------------------------------------------------------------------ channel delivery
async def test_the_live_case_reply_to_a_pdf_post_it_in_the_channel_and_tag_jemm(
    bot, world, chans, jemm
):
    req, src = reply_to(world, chans)
    out = await deliver(
        bot, world, req, destination="paper-minute",
        note="เรียน @[Jemm] ส่งรายงานการประชุมมาให้ตรวจสอบครับ", mention=["Jemm"],
    )  # fmt: skip
    assert out["status"] == "posted" and out["channel"] == "paper-minute"
    (post,) = chans.dest.posts
    assert "<@40>" in post.content and "@[Jemm]" not in post.content  # a real, resolved mention
    assert post.content.count("<@40>") == 1  # not tagged twice (note + mention list)
    assert t("deliver_footer", "th", who=f"<@{STAFF}>", url=src.jump_url) in post.content
    (file,) = post.files
    assert file.filename == "minutes.pdf"
    assert [getattr(u, "id", u) for u in post.allowed.users] == [40]  # only Jemm is pinged
    assert post.allowed.everyone is False and post.allowed.roles is False
    (log,) = bot.docs.find(DELIVERIES, GUILD)
    assert log["requester_id"] == STAFF and log["source"]["message_id"] == src.id
    assert (
        log["source"]["files"] == ["minutes.pdf"] and log["destination"]["name"] == "paper-minute"
    )
    assert log["at"]


async def test_unique_thai_channel_name_match_posts_straight_away(bot, world, chans):
    req, _ = reply_to(world, chans)
    first = await deliver(bot, world, req, destination="ห้องเปเปอร์", note="รบกวนตรวจครับ")
    assert first["status"] == "posted" and len(chans.dest.posts) == 1


async def test_ambiguous_channel_asks_and_unknown_channel_lists_candidates(bot, world, chans):
    extra = DChan(ARCH, "paper-archive")
    extra.guild = world.g
    world.g.text_channels.append(extra)
    req, _ = reply_to(world, chans)
    amb = await deliver(bot, world, req, destination="paper")
    assert amb["error"] == "ambiguous" and set(amb["candidates"]) == {
        "#paper-minute",
        "#paper-archive",
    }
    none = await deliver(bot, world, req, destination="finance")
    assert none["error"] == "destination_not_found" and "paper-minute" in none["channels"]
    assert chans.dest.posts == [] and extra.posts == []


async def test_a_channel_mention_is_resolved_exactly(bot, world, chans):
    req, _ = reply_to(world, chans)
    out = await deliver(bot, world, req, destination=f"<#{DEST}>", note="ส่งมาให้ครับ")
    assert out["status"] == "posted" and out["mention"] == f"<#{DEST}>"


async def test_files_on_the_current_message_and_by_link_and_by_pdf_name(bot, world, chans):
    cur = TMsg(world.g, chans.src, world.staff, "ส่งไฟล์นี้ไปที่ paper-minute")
    cur.attachments = [FAtt("a.pdf", b"A"), FAtt("b.pdf", b"B")]
    out = await deliver(bot, world, cur, destination="paper-minute", file_index=1)
    assert out["files"] == ["b.pdf"] and chans.dest.posts[0].files[0].filename == "b.pdf"
    out = await deliver(bot, world, cur, destination="paper-minute", request_id="r2")
    assert out["files"] == ["a.pdf", "b.pdf"]  # no index: all of them
    # a message link (as chat_search returns it); the fetch goes to the channel of the link
    found = source_msg(chans.src, [FAtt("linked.pdf")], mid=888)

    async def fetch(mid):
        assert mid == 888
        return found

    chans.src.fetch_message = fetch
    link = f"https://discord.com/channels/{GUILD}/{SRC}/888"
    plain = TMsg(world.g, chans.src, world.staff, "ส่งต่อ")
    out = await deliver(bot, world, plain, destination="paper-minute", source=link, request_id="r3")
    assert out["files"] == ["linked.pdf"]
    # the name of a PDF Salid made earlier in the conversation: no source message, no link
    bot.secretary_state.remember_pdf(GUILD, STAFF, "report.pdf", b"%PDF made")
    out = await deliver(bot, world, plain, destination="paper-minute", source="report.pdf",
                        request_id="r4")  # fmt: skip
    assert out["files"] == ["report.pdf"]
    assert "ต้นฉบับ" not in chans.dest.posts[-1].content
    bad = await deliver(bot, world, plain, destination="paper-minute", source="nope.pdf",
                        request_id="r5")  # fmt: skip
    assert bad["error"] == "source_not_found"


async def test_no_files_and_bad_index_are_honest_errors(bot, world, chans):
    plain = TMsg(world.g, chans.src, world.staff, "ส่งต่อ")
    assert (await deliver(bot, world, plain, destination="paper-minute"))["error"] == "no_file"
    req, _ = reply_to(world, chans)
    bad = await deliver(bot, world, req, destination="paper-minute", file_index=5)
    assert bad["error"] == "no_file" and "out of range" in bad["note"]


async def test_deleted_source_is_an_honest_error(bot, world, chans):
    req = TMsg(world.g, chans.src, world.staff, "ส่งต่อ")
    req.attachments, req.reference = [], SimpleNamespace(resolved=None, message_id=404)
    req.channel = chans.src  # fetch_message raises NotFound
    out = await deliver(bot, world, req, destination="paper-minute")
    assert out["error"] == "source_deleted"
    link = f"https://discord.com/channels/{GUILD}/{SRC}/404"
    plain = TMsg(world.g, chans.src, world.staff, "x")
    assert (await deliver(bot, world, plain, destination="paper-minute", source=link))[
        "error"
    ] == "source_deleted"
    assert chans.dest.posts == []


# ------------------------------------------------------------------ access rules
async def test_requester_must_be_able_to_read_the_source_channel(bot, world, chans):
    secret_src = DChan(34, "board", readers={BOSS, OWNER_ID, ADMIN_ID})  # Somchai cannot read
    secret_src.guild = world.g
    world.g.text_channels.append(secret_src)
    src = source_msg(secret_src)
    req = TMsg(world.g, chans.src, world.staff, "ส่งต่อ")
    req.attachments, req.reference = [], SimpleNamespace(resolved=src, message_id=src.id)
    out = await deliver(bot, world, req, destination="paper-minute")
    assert out["error"] == "not_allowed" and chans.dest.posts == []


async def test_requester_must_be_able_to_send_in_the_destination_and_salid_must_attach(
    bot, world, chans
):
    chans.dest.senders = {BOSS}  # Somchai may read it but not write
    req, _ = reply_to(world, chans)
    out = await deliver(bot, world, req, destination="paper-minute")
    assert out["error"] == "not_allowed" and "paper-minute" in out["note"]
    chans.dest.senders, chans.dest.bot_attach = None, False
    out = await deliver(bot, world, req, destination="paper-minute", request_id="r2")
    assert out["error"] == "missing_permission" and "Attach Files" in out["how_to_grant"]
    assert chans.dest.posts == []


async def test_channels_the_requester_cannot_see_are_not_matched_or_listed(bot, world, chans):
    hidden = DChan(35, "paper-board", readers={BOSS})  # Somchai cannot see it
    hidden.guild = world.g
    world.g.text_channels.append(hidden)
    req, _ = reply_to(world, chans)
    out = await deliver(bot, world, req, destination="paper")  # only paper-minute is visible
    assert out["status"] == "posted" and hidden.posts == []
    none = await deliver(bot, world, req, destination="board", request_id="r2")
    assert none["error"] == "destination_not_found" and "paper-board" not in none["channels"]


async def test_cross_audience_delivery_needs_a_later_confirmation(bot, world, chans):
    chans.dest.readers.add(INTERN)  # Ploy reads the destination but not the source channel
    out = await deliver(bot, world, *[reply_to(world, chans)[0]], destination="paper-minute",
                        note="ส่งมาให้ครับ")  # fmt: skip
    assert out["status"] == "needs_confirmation"
    assert "minutes-source" in out["warning"] and "มองไม่เห็น ยืนยันจะส่งไหมครับ" in out["warning"]
    assert chans.dest.posts == []
    # the same request cannot confirm itself, whatever the model does
    req, _ = reply_to(world, chans)
    same = await deliver(bot, world, req, destination="paper-minute", note="ส่งมาให้ครับ")
    assert same["status"] == "needs_confirmation" and chans.dest.posts == []
    # a later message with exactly the same arguments goes through
    later, _ = reply_to(world, chans)
    done = await deliver(bot, world, later, request_id="r2", destination="paper-minute",
                         note="ส่งมาให้ครับ")  # fmt: skip
    assert done["status"] == "posted" and len(chans.dest.posts) == 1
    # different arguments are a new proposal again
    other, _ = reply_to(world, chans)
    changed = await deliver(bot, world, other, request_id="r3", destination="paper-minute",
                            note="อีกข้อความ")  # fmt: skip
    assert changed["status"] == "needs_confirmation"


async def test_no_confirmation_when_everyone_at_the_destination_can_read_the_source(
    bot, world, chans
):
    chans.dest.readers = {STAFF, PEER}
    req, _ = reply_to(world, chans)
    assert (await deliver(bot, world, req, destination="paper-minute"))["status"] == "posted"


# ------------------------------------------------------------------ people and email
async def test_delivering_to_a_person_is_a_dm_with_the_file(bot, world, chans, jemm):
    req, src = reply_to(world, chans)
    out = await deliver(bot, world, req, destination="Jemm", note="รบกวนตรวจครับ")
    assert out == {"status": "sent_dm", "to": "Jemm", "files": ["minutes.pdf"]}
    ((text, kw),) = jemm.dms
    assert "รบกวนตรวจครับ" in text and src.jump_url in text and "Somchai" in text
    assert kw["files"][0].filename == "minutes.pdf"
    assert chans.dest.posts == []
    assert bot.docs.find(DELIVERIES, GUILD)[0]["destination"]["type"] == "dm"


async def test_dm_to_someone_who_cannot_read_the_source_needs_confirmation(bot, world, chans, jemm):
    chans.src.readers.discard(40)
    req, _ = reply_to(world, chans)
    out = await deliver(bot, world, req, destination="Jemm")
    assert out["status"] == "needs_confirmation" and "Jemm" in out["warning"] and jemm.dms == []


async def test_closed_dms_are_reported_with_guidance(bot, world, chans, jemm):
    async def closed(text=None, **kw):
        raise discord.Forbidden(SimpleNamespace(status=403, reason="x"), "closed")

    jemm.send = closed
    req, _ = reply_to(world, chans)
    out = await deliver(bot, world, req, destination="Jemm")
    assert out["error"] == "dms_closed" and out["guidance"]


async def test_an_email_destination_makes_a_draft_with_the_attachment(bot, world, chans):
    ctx = SecretaryContext(
        settings=bot.s, guild_id=GUILD, guild_name="A", author_id=STAFF, author_name="Somchai",
        is_exec=True, can_email=True, lang="th", text_channel_id=SRC,
        store=bot.secretary_store, state=bot.secretary_state, run=lambda c: None,
    )  # fmt: skip
    req, _ = reply_to(world, chans)
    out = await deliver(bot, world, req, secretary=ctx, destination="email:boss@example.com",
                        note="เรียนคุณลูกค้า แนบรายงานมาครับ")  # fmt: skip
    assert out["status"] == "drafted" and out["draft"]["to"] == ["boss@example.com"]
    assert out["draft"]["attachment"] == "minutes.pdf" and "แนบรายงาน" in out["draft"]["body"]
    assert bot.secretary_state.recall_pdf(GUILD, STAFF, "minutes.pdf") == b"%PDF-1.4 hello"
    assert chans.dest.posts == []  # nothing is sent: the usual confirm flow comes next
    ctx.can_email = False
    denied = await deliver(bot, world, req, secretary=ctx, destination="email:x@example.com",
                           request_id="r2")  # fmt: skip
    assert denied["error"] == "not_allowed"
    no_ctx = await deliver(bot, world, req, destination="email:x@example.com", request_id="r3")
    assert no_ctx["error"] == "email_disabled"


# ------------------------------------------------------------------ note, size, tagging
async def test_mentions_are_resolved_only_when_unambiguous_and_raw_pings_are_stripped(
    bot, world, chans, jemm
):
    req, _ = reply_to(world, chans)
    out = await deliver(
        bot, world, req, destination="paper-minute",
        note="@[Jemm] @[Nobody] <@1> @everyone @[everyone] ครับ", mention=["Ghost"],
    )  # fmt: skip
    (post,) = chans.dest.posts
    assert "<@40>" in post.content and "<@1>" not in post.content
    assert "@everyone" not in post.content.replace("@[everyone]", "")
    assert set(out["could_not_tag"]) >= {"Nobody", "Ghost"}
    assert [getattr(u, "id", u) for u in post.allowed.users] == [40]


async def test_files_over_25_mb_are_not_attached_only_linked(bot, world, chans):
    big = FAtt("big.pdf", b"x", size=30 * 1024 * 1024)
    req, src = reply_to(world, chans, [big])
    out = await deliver(bot, world, req, destination="paper-minute", note="ไฟล์ใหญ่ครับ")
    (post,) = chans.dest.posts
    assert post.files is None and t("deliver_oversize", "th", url=src.jump_url) in post.content
    assert out["files"] == [] and "25 MB" in out["note"]


async def test_a_note_is_optional_and_the_post_stays_within_discords_limit(bot, world, chans):
    req, _ = reply_to(world, chans)
    await deliver(bot, world, req, destination="paper-minute", note="ก" * 5000)
    assert len(chans.dest.posts[0].content) <= 2000


async def test_a_failed_post_is_reported_not_raised(bot, world, chans):
    async def boom(*a, **k):
        raise discord.HTTPException(SimpleNamespace(status=500, reason="x"), "boom")

    chans.dest.send = boom
    req, _ = reply_to(world, chans)
    assert (await deliver(bot, world, req, destination="paper-minute"))["error"] == "could_not_post"
    assert bot.docs.find(DELIVERIES, GUILD) == []  # nothing delivered, nothing logged


async def test_the_deliver_tool_is_registered_with_its_prompt():
    from salid import qa
    from salid.chat_tools import ChatActions, build_chat_tools

    names = [f.__name__ for f in build_chat_tools(ChatActions(deliver_file=lambda **k: {}), [])]
    assert names == ["deliver_file"]
    assert "deliver_file" in qa.CHAT_TOOLS_PROMPT and "ส่งไฟล์ต่อให้คน" in qa.CAPABILITIES_PROMPT
    assert "แจ้งเตือนส่วนตัว" in qa.CHAT_TOOLS_PROMPT and "สรุปเช้า" in qa.CAPABILITIES_PROMPT
