# ruff: noqa: F811, F401
"""Company wiki: slugs, versions (last 5), search, confidentiality (source channels + hierarchy),
who is told they cannot see a page, publishing (channel candidates, remembered choice,
confirmation, audience check, replacing the old post), delete rules, and the agent wiring.
Offline: fake Discord objects, SQLite, scripted models."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import discord
import pytest
from conftest import GUILD, FakeClient, fc_resp, json_resp, scripted, text_resp
from test_coordinator import (
    ADMIN_ID,
    BOSS,
    INTERN,
    OWNER_ID,
    PEER,
    PUBLIC,
    SECRET,
    STAFF,
    RChan,
    at,
    person,
    world,
)
from test_modes import TMsg, bot  # noqa: F401  (fixture)

from salid import qa
from salid.coo_store import SQLiteDocStore
from salid.guild_settings import GuildSettingsUpdate
from salid.i18n import t
from salid.memory import build_message
from salid.wiki import (
    KEEP_VERSIONS,
    WIKI_PROMPT,
    WikiPage,
    WikiRepo,
    find_pages,
    page_visible,
    slugify,
)
from salid.wiki_bot import looks_like_knowledge, page_message

WIKI_CH, DOCS_CH = 40, 41


class WChan(RChan):
    """A channel Salid may post in; keeps what was sent and what was deleted."""

    def __init__(self, id, name, readers=None, topic=""):
        super().__init__(id, name, readers)
        self.topic, self.posts, self.deleted, self.n = topic, [], [], 0

    def permissions_for(self, m):
        ok = self.readers is None or m.id in self.readers or m.id == 999
        return SimpleNamespace(
            read_messages=ok,
            read_message_history=ok,
            view_channel=ok,
            send_messages=ok,
            create_public_threads=ok,
        )

    async def send(self, content=None, **kw):
        self.n += 1
        msg = SimpleNamespace(
            id=7000 + self.n, content=content,
            jump_url=f"https://discord.com/channels/{GUILD}/{self.id}/{7000 + self.n}",
        )  # fmt: skip
        self.posts.append(msg)
        return msg

    def get_partial_message(self, mid):
        async def delete():
            self.deleted.append(mid)

        return SimpleNamespace(delete=delete)


def put(bot, mid, uid, name, text, cid, day=14, kind="chat"):
    m = build_message(
        bot.s, guild_id=GUILD, channel_id=cid, message_id=mid, author_id=uid, author_name=name,
        created_at=at(10, day=day).astimezone(UTC), text=text, kind=kind,
        jump_url=f"https://discord.com/channels/{GUILD}/{cid}/{mid}",
    )  # fmt: skip
    bot.memory_store.add_message(m)


def seed(bot):
    put(bot, 1, STAFF, "Somchai", "วิธีเบิกเงิน: กรอกแบบฟอร์ม แนบใบเสร็จ ส่งบัญชีภายในวันที่ 5", PUBLIC)
    put(bot, 2, BOSS, "Boss", "ตัดสินใจ วงเงินเบิกเกิน 5000 ต้องให้บอสอนุมัติ", PUBLIC, day=13)
    put(bot, 3, BOSS, "Boss", "งบโฆษณา ลับ ไตรมาสหน้า 900000", SECRET)
    put(bot, 4, BOSS, "Boss", "เบิกเงินค่าเดินทาง ให้ส่งใบเสร็จผ่านสลิด (ลับ)", SECRET)


@pytest.fixture
def wchans(world):
    wiki = WChan(WIKI_CH, "ความรู้บริษัท")
    docs = WChan(DOCS_CH, "docs-private", readers={BOSS, OWNER_ID, ADMIN_ID})
    for c in (wiki, docs):
        c.guild = world.g
        world.g.text_channels.append(c)
    return SimpleNamespace(wiki=wiki, docs=docs)


async def tools(bot, world, who, answering=None, llm=None):
    """The wiki tools of one request by `who` (DM-style scope unless `answering` is a channel)."""
    member = world.g.members[who]
    where = SimpleNamespace(guild=world.g if answering else None, channel=answering)
    mem = await bot._memory_tools(where, world.g, member)
    if llm is not None:
        bot.llm = llm
    is_exec = who in (OWNER_ID, ADMIN_ID, BOSS)
    fns = bot._wiki_tools(world.g, member, is_exec, mem, "th")
    return {f.__name__: f for f in fns}


def writer_llm(body="## สรุป\nกรอกแบบฟอร์ม แนบใบเสร็จ ส่งบัญชีภายในวันที่ 5\n"):
    return FakeClient([json_resp({"keywords": ["เบิกเงิน", "เบิก"]}), text_resp(body)])


# ------------------------------------------------------------------ pure parts
def test_slugify_keeps_thai_and_collapses_separators():
    assert slugify("วิธีเบิกเงิน") == "วิธีเบิกเงิน"
    assert slugify("  Client  ABC!! ") == "client-abc"
    assert slugify("!!!") == "page"
    assert len(slugify("x" * 300)) <= 80


def test_repo_versions_keep_the_last_five():
    now = [datetime(2026, 10, 1, tzinfo=UTC)]
    repo = WikiRepo(SQLiteDocStore(":memory:"), clock=lambda: now[0])
    for i in range(8):
        now[0] += timedelta(days=1)
        repo.save(GUILD, WikiPage(slug="a", title="A", body=f"v{i + 1}", author_id=i))
    page = repo.get(GUILD, "a")
    assert page.version == 8 and page.body == "v8"
    assert len(page.history) == KEEP_VERSIONS
    assert [h["body"] for h in page.history] == ["v7", "v6", "v5", "v4", "v3"]
    assert repo.get(GUILD + 1, "a") is None  # per guild
    repo.delete(GUILD, "a")
    assert repo.list(GUILD) == []


def test_repo_remembers_the_knowledge_channel_per_guild():
    repo = WikiRepo(SQLiteDocStore(":memory:"))
    assert repo.channel_id(GUILD) is None
    repo.set_channel(GUILD, 40)
    assert repo.channel_id(GUILD) == 40 and repo.channel_id(GUILD + 1) is None


def test_page_visible_rules():
    p = WikiPage(slug="a", title="A", channel_ids=[10, 11], channel_authors={"11": [5, 6]})
    assert page_visible(p, {10, 11})
    assert not page_visible(p, {10})  # one source unreadable
    assert not page_visible(p, {10}, frozenset({5}))  # outranks only one of the two authors
    assert page_visible(p, {10}, frozenset({5, 6}))  # hierarchy grants the missing channel
    assert page_visible(WikiPage(slug="b", title="B"), set())  # no sources: everyone
    dm = WikiPage(slug="c", title="C", channel_ids=[-5], channel_authors={"-5": [5]})
    assert not page_visible(dm, set(), frozenset({5}))  # DM scopes are never granted


def test_find_pages_exact_then_scored():
    pages = [
        WikiPage(slug="วิธีเบิกเงิน", title="วิธีเบิกเงิน", body="กรอกแบบฟอร์ม"),
        WikiPage(slug="vpn", title="VPN setup", body="install the client then login"),
    ]
    assert [p.slug for _, p in find_pages(pages, "วิธีเบิกเงิน")] == ["วิธีเบิกเงิน"]
    assert [p.slug for _, p in find_pages(pages, "how to login to vpn")][0] == "vpn"
    assert find_pages(pages, "ลาพักร้อน") == []


def test_knowledge_channel_names():
    for n in ("wiki", "ความรู้บริษัท", "team-docs", "Knowledge-Base", "คู่มือ", "how-to"):
        assert looks_like_knowledge(SimpleNamespace(name=n, topic=""))
    assert looks_like_knowledge(SimpleNamespace(name="general", topic="our wiki lives here"))
    assert not looks_like_knowledge(SimpleNamespace(name="ทั่วไป", topic=""))
    assert not looks_like_knowledge(SimpleNamespace(name="doctors", topic=""))


def test_post_text_has_title_body_and_footer():
    p = WikiPage(slug="a", title="วิธีเบิกเงิน", body="## สรุป\nx", author_name="Boss",
                 updated_at="2026-10-14T10:00:00")  # fmt: skip
    text = page_message(p, "th")
    assert text.startswith("# วิธีเบิกเงิน\n\n## สรุป") and "2026-10-14" in text and "Boss" in text
    assert "Last updated 2026-10-14" in page_message(p, "en")


# ------------------------------------------------------------------ write
async def test_write_researches_scope_and_saves_a_page_with_sources(bot, world, wchans):
    seed(bot)
    w = await tools(bot, world, STAFF, llm=writer_llm())
    out = await asyncio.to_thread(w["wiki_write"], "วิธีเบิกเงิน", "เน้นขั้นตอน")
    assert out["status"] == "saved" and out["version"] == 1 and out["slug"] == "วิธีเบิกเงิน"
    page = bot.wiki.get(GUILD, "วิธีเบิกเงิน")
    assert page.author_id == STAFF and page.author_name == "Somchai"
    assert page.channel_ids == [PUBLIC]  # Somchai cannot read #ลับ: nothing from it was used
    assert f"https://discord.com/channels/{GUILD}/{PUBLIC}/1" in page.source_links
    assert page.body.rstrip().splitlines()[-1].startswith("- [")  # sources added by code
    assert "restricted" not in out and out["publish"]["channel_candidates"] == ["#ความรู้บริษัท"]
    prompt = bot.llm.models.calls[1][1]  # what the writer model saw
    assert (
        "กรอกแบบฟอร์ม" in prompt
        and "งบโฆษณา" not in prompt
        and "ลับ" not in prompt.split("MATERIAL")[1]
    )
    assert "เน้นขั้นตอน" in prompt


async def test_write_with_nothing_found_and_without_a_model(bot, world):
    seed(bot)
    w = await tools(bot, world, STAFF, llm=FakeClient([json_resp({"keywords": ["ลาพักร้อน"]})]))
    assert (await asyncio.to_thread(w["wiki_write"], "ลาพักร้อน"))["error"] == "nothing_found"
    assert bot.wiki.list(GUILD) == []
    bot.llm = None
    w = await tools(bot, world, STAFF)
    assert "not available" in (await asyncio.to_thread(w["wiki_write"], "x"))["error"]


async def test_instructions_alone_can_make_a_page(bot, world):
    w = await tools(bot, world, STAFF, llm=writer_llm("ลา 3 วัน แจ้งหัวหน้า"))
    out = await asyncio.to_thread(w["wiki_write"], "วิธีขอลา", "ลา 3 วัน แจ้งหัวหน้า")
    assert out["status"] == "saved" and bot.wiki.get(GUILD, "วิธีขอลา").channel_ids == []


async def test_restricted_sources_tell_the_writer_who_cannot_see_the_page(bot, world):
    seed(bot)
    w = await tools(bot, world, BOSS, llm=writer_llm())
    out = await asyncio.to_thread(w["wiki_write"], "เบิกเงิน")
    page = bot.wiki.get(GUILD, "เบิกเงิน")
    assert set(page.channel_ids) == {PUBLIC, SECRET}
    assert out["restricted"] is True and out["source_channels"] == ["ทั่วไป", "ลับ"]
    # the secret channel is readable by owner, admin and boss: the others cannot see the page
    assert set(out["cannot_see"]) == {"Somchai", "Peer", "Ploy"} and out["cannot_see_count"] == 3


async def test_update_keeps_old_text_input_and_bumps_the_version(bot, world):
    seed(bot)
    w = await tools(bot, world, STAFF, llm=writer_llm("v1 body"))
    await asyncio.to_thread(w["wiki_write"], "วิธีเบิกเงิน")
    w = await tools(bot, world, PEER, llm=writer_llm("v2 body"))
    out = await asyncio.to_thread(w["wiki_write"], "วิธีเบิกเงิน", "เพิ่มเรื่องวงเงิน")
    assert out["status"] == "updated" and out["version"] == 2
    page = bot.wiki.get(GUILD, "วิธีเบิกเงิน")
    assert page.author_id == PEER and page.history[0]["body"].startswith("v1 body")
    assert "v1 body" in bot.llm.models.calls[1][1]  # the writer sees the current page


async def test_cannot_overwrite_a_page_one_may_not_see(bot, world):
    seed(bot)
    w = await tools(bot, world, BOSS, llm=writer_llm())
    await asyncio.to_thread(w["wiki_write"], "เบิกเงิน")
    w = await tools(bot, world, STAFF, llm=writer_llm())
    out = await asyncio.to_thread(w["wiki_write"], "เบิกเงิน")
    assert out["error"] == "name_taken" and "ลับ" not in str(out)
    assert bot.wiki.get(GUILD, "เบิกเงิน").version == 1


# ------------------------------------------------------------------ read: confidentiality
async def test_readers_see_only_pages_whose_sources_they_can_read(bot, world):
    seed(bot)
    w = await tools(bot, world, BOSS, llm=writer_llm())
    await asyncio.to_thread(w["wiki_write"], "เบิกเงิน")  # SECRET + PUBLIC
    pub = await tools(bot, world, STAFF, llm=writer_llm())
    await asyncio.to_thread(pub["wiki_write"], "วิธีเบิกเงินของพนักงาน")  # PUBLIC only
    for who, expect in ((BOSS, 2), (OWNER_ID, 2), (STAFF, 1), (INTERN, 1)):
        r = await tools(bot, world, who)
        listed = (await asyncio.to_thread(r["wiki_list"]))["pages"]
        assert len(listed) == expect, who
    r = await tools(bot, world, PEER)
    got = await asyncio.to_thread(r["wiki_get"], "เบิกเงิน")
    assert got["title"] == "วิธีเบิกเงินของพนักงาน"  # "เบิกเงิน" resolves to what Peer can see
    hidden_only = await asyncio.to_thread(r["wiki_search"], "งบโฆษณา")
    assert hidden_only["count"] == 0
    r = await tools(bot, world, BOSS)
    assert (await asyncio.to_thread(r["wiki_get"], "เบิกเงิน"))["title"] == "เบิกเงิน"


async def test_hierarchy_grants_a_page_made_only_of_lower_ranked_work(bot, world):
    put(bot, 10, INTERN, "Ploy", "ขั้นตอนเบิกค่าเดินทาง ส่งใบเสร็จผ่านสลิด", SECRET)
    w = await tools(bot, world, BOSS, llm=writer_llm())
    await asyncio.to_thread(w["wiki_write"], "เบิกค่าเดินทาง")
    page = bot.wiki.get(GUILD, "เบิกค่าเดินทาง")
    assert page.channel_ids == [SECRET] and page.channel_authors == {str(SECRET): [INTERN]}
    # Somchai (2) outranks Ploy (1) and may read it in a DM, but Ploy herself cannot read #ลับ
    # and does not outrank herself... she is the author, so she may.
    for who, ok in ((STAFF, True), (INTERN, True), (PEER, True)):
        r = await tools(bot, world, who)
        assert (await asyncio.to_thread(r["wiki_list"]))["count"] == int(ok), who
    # in front of an audience (the public channel) the rule needs every reader to outrank her:
    # Ploy herself is in #ทั่วไป so it still passes, but #ลับ content is never granted by channel
    r = await tools(bot, world, STAFF, answering=world.pub)
    assert (await asyncio.to_thread(r["wiki_list"]))["count"] == 1
    # a page that also used the boss's secret message is not granted by hierarchy
    put(bot, 11, BOSS, "Boss", "เบิกค่าเดินทางต้องอนุมัติโดยบอส", SECRET)
    w = await tools(bot, world, BOSS, llm=writer_llm())
    await asyncio.to_thread(w["wiki_write"], "เบิกค่าเดินทาง")
    r = await tools(bot, world, STAFF)
    assert (await asyncio.to_thread(r["wiki_list"]))["count"] == 0


async def test_dm_notes_and_salids_own_replies_are_never_source_material(bot, world):
    from salid.memory_access import dm_scope

    put(bot, 20, STAFF, "Somchai", "เบิกเงิน เรื่องส่วนตัวใน DM", dm_scope(STAFF), kind="dm")
    put(bot, 21, 999, "Salid", "เบิกเงิน คำตอบเก่าของสลิด", PUBLIC, kind="salid")
    w = await tools(bot, world, STAFF, llm=writer_llm())
    out = await asyncio.to_thread(w["wiki_write"], "เบิกเงิน")
    assert out["error"] == "nothing_found"


async def test_get_and_search_report_staleness(bot, world):
    seed(bot)
    w = await tools(bot, world, STAFF, llm=writer_llm("ขั้นตอนเบิกเงิน: กรอกแบบฟอร์ม"))
    await asyncio.to_thread(w["wiki_write"], "วิธีเบิกเงิน")
    page = bot.wiki.get(GUILD, "วิธีเบิกเงิน")
    page.updated_at = "2026-05-01T10:00:00+07:00"  # five months old
    bot.wiki.put_raw(GUILD, page)
    r = await tools(bot, world, PEER)
    hit = await asyncio.to_thread(r["wiki_search"], "ขั้นตอนเบิกเงิน ทำยังไง")
    assert hit["count"] == 1 and hit["pages"][0]["stale"] is True
    assert hit["pages"][0]["updated_at"].startswith("2026-05-01") and "snippet" in hit["pages"][0]
    got = await asyncio.to_thread(r["wiki_get"], "วิธีเบิกเงิน")
    assert got["age_days"] > 90 and got["stale"] and got["body"].startswith("ขั้นตอน")
    assert (await asyncio.to_thread(r["wiki_get"], "ไม่มีหน้านี้"))["error"] == "not_found"


async def test_pages_survive_memory_retention(bot, world):
    seed(bot)
    w = await tools(bot, world, STAFF, llm=writer_llm())
    await asyncio.to_thread(w["wiki_write"], "วิธีเบิกเงิน")
    bot.memory_store.delete_guild(GUILD) if hasattr(bot.memory_store, "delete_guild") else None
    assert bot.wiki.get(GUILD, "วิธีเบิกเงิน") is not None  # a different collection


# ------------------------------------------------------------------ delete
async def test_delete_is_for_the_author_or_an_executive_and_needs_confirmation(bot, world):
    seed(bot)
    w = await tools(bot, world, STAFF, llm=writer_llm())
    await asyncio.to_thread(w["wiki_write"], "วิธีเบิกเงิน")
    other = await tools(bot, world, PEER)
    assert (await asyncio.to_thread(other["wiki_delete"], "วิธีเบิกเงิน"))["error"] == "not_allowed"
    mine = await tools(bot, world, STAFF)
    first = await asyncio.to_thread(mine["wiki_delete"], "วิธีเบิกเงิน")
    assert first["status"] == "needs_confirmation" and bot.wiki.get(GUILD, "วิธีเบิกเงิน")
    same = await asyncio.to_thread(mine["wiki_delete"], "วิธีเบิกเงิน")  # same request: no
    assert same["status"] == "needs_confirmation"
    later = await tools(bot, world, STAFF)  # a later message
    assert (await asyncio.to_thread(later["wiki_delete"], "วิธีเบิกเงิน"))["status"] == "deleted"
    assert bot.wiki.list(GUILD) == []


async def test_an_executive_can_delete_any_visible_page_and_its_post(bot, world, wchans):
    seed(bot)
    w = await tools(bot, world, STAFF, llm=writer_llm())
    await asyncio.to_thread(w["wiki_write"], "วิธีเบิกเงิน")
    page = bot.wiki.get(GUILD, "วิธีเบิกเงิน")
    page.post = {"channel_id": WIKI_CH, "thread_id": None, "message_ids": [7001, 7002]}
    bot.wiki.put_raw(GUILD, page)
    for _ in range(2):
        boss = await tools(bot, world, BOSS)
        out = await asyncio.to_thread(boss["wiki_delete"], "วิธีเบิกเงิน")
    assert out["status"] == "deleted" and wchans.wiki.deleted == [7001, 7002]


# ------------------------------------------------------------------ publish
async def written(bot, world, who=STAFF, topic="วิธีเบิกเงิน"):
    w = await tools(bot, world, who, llm=writer_llm())
    return await asyncio.to_thread(w["wiki_write"], topic)


async def test_publish_asks_which_channel_once_and_only_an_executive_chooses(bot, world, wchans):
    seed(bot)
    await written(bot, world)
    staff = await tools(bot, world, STAFF)
    out = await asyncio.to_thread(staff["wiki_publish"], "วิธีเบิกเงิน")
    assert out["error"] == "no_channel" and out["channel_candidates"] == ["#ความรู้บริษัท"]
    out = await asyncio.to_thread(staff["wiki_publish"], "วิธีเบิกเงิน", "ความรู้บริษัท")
    assert out["error"] == "not_allowed" and wchans.wiki.posts == []
    assert bot.wiki.channel_id(GUILD) is None
    # an executive picks it: needs a confirmation first, nothing is posted before
    for call in range(2):
        boss = await tools(bot, world, BOSS)
        out = await asyncio.to_thread(boss["wiki_publish"], "วิธีเบิกเงิน", "ความรู้บริษัท")
        if call == 0:
            assert out["status"] == "needs_confirmation" and wchans.wiki.posts == []
    assert out["status"] == "posted" and out["remembered_channel"] is True
    assert bot.wiki.channel_id(GUILD) == WIKI_CH
    (post,) = wchans.wiki.posts
    assert post.content.startswith("# วิธีเบิกเงิน") and "สลิด" in post.content
    assert out["link"] == post.jump_url
    saved = bot.wiki.get(GUILD, "วิธีเบิกเงิน").post
    assert saved == {"channel_id": WIKI_CH, "thread_id": None, "message_ids": [post.id]}
    # the choice is remembered: anyone who may post there publishes without naming it, still
    # with a confirmation, and a re-publish replaces the earlier post
    for _ in range(2):
        staff = await tools(bot, world, STAFF)
        out = await asyncio.to_thread(staff["wiki_publish"], "วิธีเบิกเงิน")
    assert out["status"] == "posted" and len(wchans.wiki.posts) == 2
    assert wchans.wiki.deleted == [post.id]


async def test_the_write_result_offers_the_remembered_channel(bot, world, wchans):
    seed(bot)
    bot.wiki.set_channel(GUILD, WIKI_CH)
    out = await written(bot, world)
    assert out["publish"]["channel"] == "ความรู้บริษัท" and "channel_candidates" not in out["publish"]
    bot.wiki.set_channel(GUILD, 9999)  # the channel was deleted: ask again
    out = await written(bot, world, topic="วิธีเบิกเงิน 2")
    assert out["publish"].get("channel_candidates") == ["#ความรู้บริษัท"]


async def test_never_publish_where_the_audience_cannot_read_the_sources(bot, world, wchans):
    seed(bot)
    out = await written(bot, world, BOSS, "เบิกเงิน")  # uses #ลับ (boss / admin / owner only)
    assert out["restricted"]
    boss = await tools(bot, world, BOSS)
    out = await asyncio.to_thread(boss["wiki_publish"], "เบิกเงิน", "ความรู้บริษัท")
    assert out["error"] == "audience_too_wide" and "Somchai" in out["cannot_see"]
    assert wchans.wiki.posts == [] and bot.wiki.channel_id(GUILD) is None
    # a knowledge channel only the leadership reads is fine
    bot.wiki.set_channel(GUILD, DOCS_CH)
    for _ in range(2):
        boss = await tools(bot, world, BOSS)
        out = await asyncio.to_thread(boss["wiki_publish"], "เบิกเงิน")
    assert out["status"] == "posted" and len(wchans.docs.posts) == 1


async def test_publish_needs_permission_to_post_and_a_visible_page(bot, world, wchans):
    seed(bot)
    await written(bot, world, BOSS, "เบิกเงิน")
    staff = await tools(bot, world, STAFF)
    assert (await asyncio.to_thread(staff["wiki_publish"], "เบิกเงิน"))["error"] == "not_found"
    bot.wiki.set_channel(GUILD, DOCS_CH)
    await written(bot, world, STAFF, "หน้าสาธารณะ")
    out = await asyncio.to_thread((await tools(bot, world, STAFF))["wiki_publish"], "หน้าสาธารณะ")
    assert out["error"] == "not_allowed"  # Somchai cannot see/post in the leadership channel
    assert wchans.docs.posts == []


async def test_a_long_page_is_posted_in_chunks_without_pings(bot, world, wchans):
    seed(bot)
    long = "## ข้อมูล\n" + "\n".join(f"- ข้อ {i} @everyone เบิกเงิน" for i in range(300))
    w = await tools(bot, world, STAFF, llm=writer_llm(long))
    await asyncio.to_thread(w["wiki_write"], "วิธีเบิกเงิน")
    bot.wiki.set_channel(GUILD, WIKI_CH)
    for _ in range(2):
        out = await asyncio.to_thread((await tools(bot, world, STAFF))["wiki_publish"], "วิธีเบิกเงิน")
    assert out["status"] == "posted" and len(wchans.wiki.posts) > 1
    assert all(len(p.content) <= 2000 for p in wchans.wiki.posts)


# ------------------------------------------------------------------ agent wiring
async def test_the_same_wiki_tools_reach_chat_and_voice_builder(bot, world):
    member = world.g.members[STAFF]
    mem = await bot._memory_tools(SimpleNamespace(guild=None, channel=None), world.g, member)
    msg = bot.room_message(world.g, member, world.pub)
    kw = bot._agent_kwargs(
        msg, world.g, member, False, setting="th", lang="th", question="q", memory=mem,
        profile=None,
    )  # fmt: skip
    names = {f.__name__ for f in kw["extra_tools"]}
    assert {
        "wiki_write",
        "wiki_get",
        "wiki_search",
        "wiki_list",
        "wiki_delete",
        "wiki_publish",
    } <= names
    assert {"list_tasks", "my_day"} <= names  # next to the coordinator tools
    # no wiki for an interjection, without memory, or when the guild switched it off
    quiet = bot._agent_kwargs(
        msg, world.g, member, False, setting="th", lang="th", question="q", memory=mem,
        profile=None, interjection=True,
    )  # fmt: skip
    assert not any(f.__name__.startswith("wiki_") for f in quiet.get("extra_tools", []))
    off = bot._agent_kwargs(
        msg, world.g, member, False, setting="th", lang="th", question="q", memory=None,
        profile=None,
    )  # fmt: skip
    assert "wiki_search" not in {f.__name__ for f in off.get("extra_tools", [])}
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={"wiki": False}), "x")
    bot.team._cache.clear()
    kw = bot._agent_kwargs(
        msg, world.g, member, False, setting="th", lang="th", question="q", memory=mem,
        profile=None,
    )  # fmt: skip
    assert "wiki_search" not in {f.__name__ for f in kw["extra_tools"]}


async def test_agent_consults_the_wiki_first_and_the_prompt_says_so(bot, world):
    seed(bot)
    w = await tools(bot, world, STAFF, llm=writer_llm("1. กรอกแบบฟอร์ม 2. แนบใบเสร็จ"))
    await asyncio.to_thread(w["wiki_write"], "วิธีเบิกเงิน")
    member = world.g.members[PEER]
    mem = await bot._memory_tools(SimpleNamespace(guild=None, channel=None), world.g, member)
    kw = bot._agent_kwargs(
        bot.room_message(world.g, member, world.pub), world.g, member, False, setting="th",
        lang="th", question="เบิกเงินยังไง", memory=mem, profile=None,
    )  # fmt: skip
    llm = scripted(
        fc_resp(("wiki_search", {"query": "เบิกเงินยังไง"})),
        fc_resp(("wiki_get", {"topic": "วิธีเบิกเงิน"})),
        text_resp("ตามหน้าความรู้ค่ะ"),
    )
    a = await asyncio.to_thread(
        qa.ask, bot.s, bot.store, "เบิกเงินยังไง", guild_id=GUILD, is_exec=False, user_id=PEER,
        model=llm, **kw,
    )  # fmt: skip
    assert a.text == "ตามหน้าความรู้ค่ะ"
    assert WIKI_PROMPT.strip()[:20] in llm.requests[0].config.system_instruction
    names = {d.name for t_ in llm.requests[0].config.tools for d in t_.function_declarations}
    assert {"wiki_search", "wiki_get", "wiki_write"} <= names
    # the second round saw the search result (with the page and its date)
    found = llm.requests[1].contents[-1].parts[0].function_response.response
    assert found["pages"][0]["title"] == "วิธีเบิกเงิน" and found["pages"][0]["updated_at"]
    plain = scripted(text_resp("ok"))
    await asyncio.to_thread(
        qa.ask, bot.s, bot.store, "hi", guild_id=GUILD, is_exec=False, user_id=PEER, model=plain
    )
    assert "wiki_search" not in plain.requests[0].config.system_instruction


class ForumChan(WChan):
    type = discord.ChannelType.forum

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.threads = []

    async def create_thread(self, name, content, **kw):
        thread = SimpleNamespace(id=9000 + len(self.threads), name=name, parts=[content], sent=[])

        async def send(part, **k):
            thread.sent.append(part)

        async def delete():
            self.deleted.append(thread.id)

        thread.send, thread.delete = send, delete
        self.threads.append(thread)
        return SimpleNamespace(
            thread=thread, message=SimpleNamespace(jump_url=f"https://discord.com/t/{thread.id}")
        )


async def test_a_forum_gets_one_thread_per_page_and_an_update_replaces_it(bot, world):
    forum = ForumChan(50, "wiki-forum")
    forum.guild = world.g
    world.g.text_channels.append(forum)
    world.g.get_channel_or_thread = lambda i: next((t_ for t_ in forum.threads if t_.id == i), None)
    seed(bot)
    await written(bot, world)
    bot.wiki.set_channel(GUILD, 50)
    for _ in range(2):
        out = await asyncio.to_thread((await tools(bot, world, STAFF))["wiki_publish"], "วิธีเบิกเงิน")
    assert out["status"] == "posted" and out["link"].startswith("https://discord.com/t/")
    (thread,) = forum.threads
    assert thread.name == "วิธีเบิกเงิน" and thread.parts[0].startswith("# วิธีเบิกเงิน")
    assert bot.wiki.get(GUILD, "วิธีเบิกเงิน").post["thread_id"] == thread.id
    for _ in range(2):
        await asyncio.to_thread((await tools(bot, world, STAFF))["wiki_publish"], "วิธีเบิกเงิน")
    assert len(forum.threads) == 2 and forum.deleted == [thread.id]  # the old thread went away
