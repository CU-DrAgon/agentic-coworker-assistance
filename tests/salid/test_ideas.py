# ruff: noqa: F811, F401
"""Idea / project workspace: sections, visibility (owner, hierarchy, private, team, shared), export,
business frameworks (complete headings, one re-ask), long-answer delivery that never cuts a
section, the library category and the feature flag. Offline: fake model, SQLite, fake Discord."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from conftest import GUILD, FakeClient, text_resp
from test_center_api import A, H, docs, w
from test_coordinator import BOSS, INTERN, PEER, STAFF, person, world
from test_human import recording
from test_modes import TMsg, bot  # noqa: F401  (fixture)

from salid import frameworks as fw
from salid import longtext
from salid.coo_store import SQLiteDocStore
from salid.guild_settings import GuildSettingsUpdate
from salid.i18n import t
from salid.ideas import (
    IDEAS_PROMPT,
    IdeaDoc,
    IdeaRepo,
    IdeasCtx,
    can_read,
    ideas_tools,
    library_visible,
    render_markdown,
    section_key,
)
from salid.ideas_pdf import canvas_blocks, markdown_docx, markdown_pdf
from salid.pdf_report import find_fonts
from salid.qa import system_prompt, tool_files

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)


def bmc_md(skip=(), lang="th"):
    out = ["# BMC: CCTV Watch", "", "intro"]
    for pair in fw.KINDS["bmc"][2]:
        if pair[1] in skip:
            continue
        out += ["", fw.heading_line(pair, lang), f"- ข้อแรกของ {pair[1]}", f"- ข้อสองของ {pair[1]}"]
    return "\n".join(out)


def ctx_for(repo, uid=STAFF, name="Somchai", outranked=(), private=True, client=None, **kw):
    return IdeasCtx(
        guild_id=GUILD, guild_name="บริษัท A", asker_id=uid, asker_name=name, is_exec=False,
        repo=repo, client=client, model="m", lang="th", now=lambda: NOW,
        outranked=lambda: frozenset(outranked), private=lambda: private,
        resolve_person=lambda n: [(i, x) for i, x in {PEER: "Peer", BOSS: "Boss"}.items() if n.lower() in x.lower()],
        fonts=lambda: find_fonts(""), **kw,
    )  # fmt: skip


@pytest.fixture
def repo():
    return IdeaRepo(SQLiteDocStore(":memory:"), clock=lambda: NOW)


# ------------------------------------------------------------------ sections and documents
def test_section_names_map_to_canonical_sections():
    assert section_key("Business Model Canvas")[0] == "bmc"
    assert section_key("ใครได้ประโยชน์")[0] == "customers"
    assert section_key("Competitors & novelty")[0] == "competitors"
    assert section_key("go-to-market")[0] == "gtm"
    assert section_key("ราคา")[0] == "pricing"
    assert section_key("Something odd")[0] is None and section_key("")[1] == ""


def test_save_get_list_roundtrip_and_section_order(repo):
    c = ctx_for(repo)
    r = c.idea_save("CCTV Watch", "แจ้งเตือนภาพจากกล้องไปที่นาฬิกา รปภ.")  # first section: problem
    assert r["status"] == "saved" and r["section"] == "ปัญหาและไอเดีย" and not r["offer_project"]
    c.idea_save("CCTV Watch", "## ใครได้ประโยชน์\n- รปภ.", "target customers")
    c.idea_save("CCTV Watch", "- Strengths: ...", "swot")
    c.idea_save("CCTV Watch", "ข้อมูลอื่น", "Funding ideas")  # unknown name: a custom section
    got = c.idea_get("cctv")  # a partial title is enough
    md = got["markdown"]
    assert got["status"] == "ok" and md.startswith("# CCTV Watch")
    order = [
        md.index(x) for x in ("## ปัญหาและไอเดีย", "## กลุ่มลูกค้าเป้าหมาย", "## SWOT", "## Funding ideas")
    ]
    assert order == sorted(order)
    assert "### ใครได้ประโยชน์" in md  # headings inside a section go one level down
    assert [i["title"] for i in c.idea_list()["ideas"]] == ["CCTV Watch"]


def test_replacing_a_section_keeps_one_copy(repo):
    c = ctx_for(repo)
    c.idea_save("X", "v1", "swot")
    c.idea_save("X", "v2", "SWOT")
    doc = repo.get(GUILD, STAFF, "x")
    assert doc.sections["swot"].body == "v2" and len(doc.sections) == 1


def test_offer_to_compile_comes_once_when_there_are_two_sections(repo):
    c = ctx_for(repo)
    assert c.idea_save("X", "a", "problem")["offer_project"] is False
    second = c.idea_save("X", "b", "swot")
    assert second["offer_project"] is True and "ONCE" in second["instruction"]
    assert c.idea_save("X", "c", "gtm")["offer_project"] is False
    assert c.idea_save("X", "d", "pricing").get("instruction") is None


def test_bad_input_is_an_error_not_a_crash(repo):
    c = ctx_for(repo)
    assert c.idea_save("", "x")["status"] == "error"
    assert c.idea_save("X", "  ")["status"] == "error"
    assert c.idea_export("nope")["status"] == "not_found"
    assert c.idea_export("X", "xlsx")["status"] == "error"


def test_per_user_cap(repo):
    c = ctx_for(repo)
    for i in range(50):
        assert c.idea_save(f"idea {i}", "x")["status"] == "saved"
    assert c.idea_save("one more", "x")["status"] == "error"


# ------------------------------------------------------------------ visibility
def test_pure_visibility_rules():
    d = IdeaDoc(slug="x", title="X", owner_id=STAFF)
    assert d.visibility == "private"  # default: owner only, not even higher roles
    assert can_read(d, STAFF) and not can_read(d, BOSS, {STAFF})
    d.visibility = "normal"  # shared with superiors
    assert not can_read(d, PEER) and can_read(d, BOSS, {STAFF})
    d.visibility, d.shared_with = "private", [PEER]  # a named person, still private otherwise
    assert can_read(d, PEER) and not can_read(d, BOSS, {STAFF})
    d.visibility, d.shared_with = "normal", [PEER]
    assert can_read(d, PEER) and not can_read(d, INTERN)
    d.shared_with, d.visibility = [], "team"
    assert can_read(d, INTERN)
    assert library_visible(d, False)
    d.visibility = "normal"
    assert library_visible(d, True) and not library_visible(d, False)
    d.visibility = "private"
    assert not library_visible(d, True)


def test_only_the_owner_sees_an_idea_by_default_peers_and_juniors_get_nothing(repo):
    ctx_for(repo).idea_save("Secret plan", "x")
    for uid in (PEER, INTERN):
        other = ctx_for(repo, uid, "Other")
        assert other.idea_list()["count"] == 0
        assert other.idea_get("Secret plan")["status"] == "not_found"
        assert other.idea_export("Secret plan")["status"] == "not_found"


def test_a_higher_role_sees_it_only_after_the_owner_shares_with_superiors(repo):
    owner = ctx_for(repo)
    owner.idea_save("Plan", "x")
    boss = ctx_for(repo, BOSS, "Boss", outranked={STAFF})
    assert boss.idea_get("Plan")["status"] == "not_found"  # private by default
    assert owner.idea_share("Plan", "หัวหน้า")["visibility"] == "normal"
    assert (
        boss.idea_get("Plan")["status"] == "ok"
        and boss.idea_list()["ideas"][0]["owner"] == "Somchai"
    )
    assert owner.idea_share("Plan", "private")["visibility"] == "private"
    assert boss.idea_get("Plan")["status"] == "not_found"
    assert owner.idea_get("Plan")["status"] == "ok"
    assert owner.idea_share("Plan", "me")["visibility"] == "private"
    assert boss.idea_get("Plan")["status"] == "not_found"


def test_sharing_with_a_person_and_with_the_team_never_lets_them_edit(repo):
    owner = ctx_for(repo)
    owner.idea_save("Plan", "x", "problem")
    r = owner.idea_share("Plan", "peer")
    assert (
        r["status"] == "ok" and r["shared_with"] == ["Peer"] and "nobody was notified" in r["note"]
    )
    peer = ctx_for(repo, PEER, "Peer")
    assert peer.idea_get("Plan")["status"] == "ok"
    assert ctx_for(repo, INTERN, "Ploy").idea_get("Plan")["status"] == "not_found"
    # the peer's own save with the same title makes their own document, not an edit
    peer.idea_save("Plan", "mine", "problem")
    assert repo.get(GUILD, STAFF, "plan").sections["problem"].body == "x"
    assert peer.idea_share("Plan", "team")["visibility"] == "team"  # their own copy only
    assert owner.idea_share("Plan", "nobody-here")["status"] == "person_not_found"
    assert ctx_for(repo, INTERN, "Ploy").idea_share("Plan", "team")["status"] == "not_found"
    owner.idea_share("Plan", "team")
    assert ctx_for(repo, INTERN, "Ploy").idea_get("Plan")["status"] == "ok"


def test_in_a_shared_room_only_team_ideas_are_shown(repo):
    owner = ctx_for(repo)
    owner.idea_save("Plan", "secret text")
    room = ctx_for(repo, private=False)
    assert room.idea_get("Plan")["status"] == "use_private_chat"
    assert "secret text" not in str(room.idea_get("Plan"))
    assert room.idea_export("Plan")["status"] == "use_private_chat" and room.outbox == []
    listed = room.idea_list()
    assert listed["count"] == 0 and listed["hidden_here"] == 1
    owner.idea_share("Plan", "team")
    assert room.idea_get("Plan")["status"] == "ok"


# ------------------------------------------------------------------ export
def test_export_md_docx_pdf(repo):
    c = ctx_for(repo)
    c.idea_save("CCTV Watch", "แจ้งเตือนภาพจากกล้อง", "problem")
    c.idea_save("CCTV Watch", bmc_md(), "bmc")
    assert c.idea_export("CCTV Watch", "md")["file"] == "cctv-watch.md"
    assert c.idea_export("CCTV Watch", "word")["format"] == "docx"
    r = c.idea_export("CCTV Watch")
    assert r["status"] == "ok" and r["format"] == "pdf"
    files = dict(c.outbox)
    assert files["cctv-watch.md"].decode().startswith("# CCTV Watch")
    assert files["cctv-watch.docx"][:2] == b"PK"
    assert files["cctv-watch.pdf"][:5] == b"%PDF-"


def test_tool_files_collects_each_outbox_once(repo):
    c = ctx_for(repo)
    tools = ideas_tools(c)
    c.idea_save("X", "a")
    c.idea_export("X", "md")
    assert tool_files(tools) == c.outbox and len(tool_files(tools)) == 1


# ------------------------------------------------------------------ frameworks
def test_all_kinds_have_their_sections():
    assert len(fw.KINDS["bmc"][2]) == 9
    assert {"bmc", "lean_canvas", "swot", "pestel", "porter5", "value_proposition",
            "pricing_tiers", "gtm", "pitch_outline"} <= set(fw.KINDS)  # fmt: skip
    assert (
        fw.resolve_kind("Business Model Canvas") == "bmc"
        and fw.resolve_kind("pricing") == "pricing_tiers"
    )
    assert fw.resolve_kind("nonsense") is None


def test_validator_finds_missing_headings_in_either_language():
    assert fw.missing_headings(bmc_md(), "bmc") == []
    assert fw.missing_headings(bmc_md(lang="en"), "bmc") == []
    assert [
        e for _t, e in fw.missing_headings(bmc_md(skip=("Channels", "Cost Structure")), "bmc")
    ] == [
        "Channels",
        "Cost Structure",
    ]


def test_a_complete_answer_is_used_as_is():
    client = FakeClient([text_resp(bmc_md())])
    res = fw.generate_framework(client, "m", "bmc", "CCTV Watch", "ctx", "th")
    assert res.complete and fw.missing_headings(res.markdown, "bmc") == []
    assert len(client.models.calls) == 1
    prompt = client.models.calls[0][1]
    assert "## กลุ่มลูกค้า (Customer Segments)" in prompt and "ctx" in prompt


def test_a_missing_heading_is_asked_for_once_then_complete():
    client = FakeClient([text_resp(bmc_md(skip=("Channels",))), text_resp(bmc_md())])
    res = fw.generate_framework(client, "m", "bmc", "CCTV Watch", "", "th")
    assert res.complete and len(client.models.calls) == 2
    again = client.models.calls[1][1]
    assert "INCOMPLETE" in again and "## ช่องทาง (Channels)" in again


def test_still_missing_after_the_retry_adds_placeholders_never_a_short_document():
    short = bmc_md(skip=("Channels", "Cost Structure"))
    client = FakeClient([text_resp(short), text_resp(short)])
    res = fw.generate_framework(client, "m", "bmc", "X", "", "th")
    assert not res.complete and res.missing == ["Channels", "Cost Structure"]
    assert fw.missing_headings(res.markdown, "bmc") == []  # the structure is complete anyway
    assert len(client.models.calls) == 2


def test_pricing_prompt_has_thai_market_defaults():
    p = fw.build_prompt("pricing_tiers", "CCTV Watch", "condo juristic", "th")
    assert "THB" in p and "VAT 7%" in p and "ASSUMPTION" in p
    assert "VAT" not in fw.build_prompt("swot", "x", "", "th")


def test_short_summary_is_at_most_eight_lines():
    s = fw.short_summary(bmc_md(), "th")
    lines = s.splitlines()
    assert len(lines) == 8 and lines[0].startswith("• กลุ่มลูกค้า: ") and "ในไฟล์แนบ" in lines[-1]
    assert len(fw.short_summary(bmc_md(lang="en"), "en").splitlines()) <= 8
    assert len(fw.short_summary(fw_md("swot"), "th").splitlines()) == 5


def fw_md(kind):
    return "# T\n\n" + "\n\n".join(
        f"{fw.heading_line(p, 'th')}\n- จุดที่ 1 ของ {p[1]}" for p in fw.KINDS[kind][2]
    )


def test_business_framework_tool_writes_attaches_saves_and_summarises(repo):
    client = FakeClient([text_resp(bmc_md())])
    c = ctx_for(repo, client=client)
    r = c.business_framework("bmc", "CCTV Watch", "แจ้งเตือนไปที่ Apple Watch")
    assert r["status"] == "saved" and r["complete"] and r["framework"] == "Business Model Canvas"
    assert len(r["summary"].splitlines()) <= 8 and "full document is attached" in r["instruction"]
    assert [n for n, _d in c.outbox] == ["cctv-watch-bmc.pdf"] and c.outbox[0][1][:5] == b"%PDF-"
    doc = repo.get(GUILD, STAFF, "cctv-watch")
    assert "bmc" in doc.sections and not doc.sections["bmc"].body.startswith("# ")
    assert fw.missing_headings(doc.sections["bmc"].body, "bmc") == []
    # a second framework for the same idea joins the same document and triggers the one offer
    client.models.script.append(text_resp(fw_md("swot")))
    r2 = c.business_framework("swot", "CCTV Watch")
    assert r2["offer_project"] is True and set(repo.get(GUILD, STAFF, "cctv-watch").sections) == {
        "bmc",
        "swot",
    }
    # the second prompt carries what the project already holds
    assert (
        "จุดแข็ง" not in client.models.calls[0][1]
        and "Customer Segments" in client.models.calls[1][1]
    )


def test_business_framework_unknown_kind_or_no_model(repo):
    c = ctx_for(repo, client=FakeClient([]))
    assert c.business_framework("tarot", "X")["status"] == "error"
    assert ctx_for(repo, client=None).business_framework("bmc", "X")["status"] == "error"


def test_the_bmc_canvas_page_and_pdf_render(repo):
    blocks, heads = canvas_blocks(bmc_md(), "th")
    assert len(blocks) == 9 and blocks["Channels"][0].startswith("ข้อแรก")
    fonts = find_fonts("")
    pdf = markdown_pdf(
        org="A", title="CCTV", markdown="# T\n\n## หัวข้อ\n- ข้อ\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nข้อความ **หนา**",
        lang="th", generated_at=NOW, fonts=fonts, canvas=bmc_md(),
    )  # fmt: skip
    assert pdf[:5] == b"%PDF-"
    import io

    from pypdf import PdfReader

    pages = PdfReader(io.BytesIO(pdf)).pages
    assert len(pages) >= 2 and float(pages[0].mediabox.width) > float(
        pages[0].mediabox.height
    )  # landscape canvas first
    assert markdown_docx("T", "# T\n\n- a\n\n| a | b |\n|---|---|\n| 1 | 2 |")[:2] == b"PK"


# ------------------------------------------------------------------ long answers
SECTIONS = [
    f"## ส่วนที่ {i}\n" + "\n".join(f"- ข้อ {j} ของส่วนที่ {i} ยาวพอสมควรเพื่อเติมพื้นที่" for j in range(12))
    for i in range(1, 13)
]


def test_split_never_cuts_a_section_that_fits():
    text = "\n\n".join(SECTIONS)
    parts = longtext.split_sections(text, 1900)
    assert len(parts) > 1 and all(len(p) <= 1900 for p in parts)
    for sec in SECTIONS:
        assert sum(sec in p for p in parts) == 1  # each section whole, in exactly one message
    assert [p.count("## ") for p in parts] != [0] * len(parts)
    assert "\n\n".join(parts).replace("\n\n", "\n") == text.replace("\n\n", "\n")


def test_a_huge_section_is_cut_between_paragraphs_and_a_table_is_kept_whole():
    paras = [(f"ย่อหน้า {i} " + "ข้อความ " * 120).strip() for i in range(6)]
    table = "\n".join(["| a | b |", "|---|---|", *[f"| {i} | x |" for i in range(15)]])
    parts = longtext.split_sections(
        "## ใหญ่\n\n" + "\n\n".join([*paras[:3], table, *paras[3:]]), 1900
    )
    assert all(len(p) <= 1900 for p in parts)
    assert sum(table in p for p in parts) == 1
    assert all(not p.rstrip().endswith("ข้อความ ข้อ") for p in parts)
    for para in paras:
        assert sum(para in p for p in parts) == 1


def test_code_fences_stay_balanced_when_a_block_is_cut():
    code = "```\n" + "\n".join(f"line {i} " + "x" * 60 for i in range(80)) + "\n```"
    parts = longtext.split_sections(code, 1900)
    assert len(parts) > 1 and all(p.count("```") % 2 == 0 for p in parts)


def test_long_means_attachment_and_the_summary_is_short():
    text = "\n\n".join(SECTIONS)
    assert longtext.needs_attachment(longtext.split_sections(text))
    assert not longtext.needs_attachment(longtext.split_sections("\n\n".join(SECTIONS[:2])))
    s = longtext.summary_of(text)
    assert len(s) < 1000 and "• ส่วนที่ 1" in s and s.count("•") <= 9


async def test_deliver_sends_sections_whole_and_one_message_per_chunk(bot, world):
    text = "\n\n".join(SECTIONS[:4])
    m = TMsg(world.g, world.pub, SimpleNamespace(id=7, bot=False, display_name="A"), "q")
    sent = []

    async def send(t_, **kw):
        sent.append(t_)
        return SimpleNamespace(id=900 + len(sent))

    world.pub.send = send
    await bot._deliver(m, world.g, None, text)
    texts = m.replies + sent
    assert len(texts) == 2 and all(len(x) <= 1900 for x in texts)
    for sec in SECTIONS[:4]:
        assert sum(sec in x for x in texts) == 1


async def test_deliver_switches_to_a_summary_and_an_attachment_when_very_long(bot, world):
    text = "สรุปคำตอบ: ภาพรวมสั้นๆ ของเรื่องนี้ค่ะ\n\n" + "\n\n".join(SECTIONS)
    m = TMsg(world.g, world.pub, SimpleNamespace(id=7, bot=False, display_name="A"), "q")
    rec = recording(m)
    more = []

    async def send(t_, **kw):
        more.append(t_)
        return SimpleNamespace(id=901)

    world.pub.send = send
    await bot._deliver(m, world.g, None, text)
    assert more == [] and len(rec.texts) == 1  # one short message, no flood
    assert t("long_answer_note", "th") in rec.texts[0] and "สรุปคำตอบ" in rec.texts[0]
    files = rec.kws[0]["files"]
    assert len(files) == 1 and files[0].filename in ("salid-answer.pdf", "salid-answer.md")
    assert len(rec.texts[0]) < 1900


async def test_deliver_keeps_other_uploads_next_to_the_long_answer_file(bot, world):
    text = "\n\n".join(SECTIONS)
    m = TMsg(world.g, world.pub, SimpleNamespace(id=7, bot=False, display_name="A"), "q")
    rec = recording(m)
    import io

    import discord

    up = discord.File(io.BytesIO(b"x"), filename="a.pdf")
    await bot._deliver(m, world.g, None, text, [up])
    assert [f.filename for f in rec.kws[0]["files"]][0] == "a.pdf" and len(rec.kws[0]["files"]) == 2


# ------------------------------------------------------------------ bot wiring and the flag
def kinds(fns):
    return {f.__name__: f for f in fns}


async def test_the_tools_are_registered_for_chat_and_voice_and_the_prompt_follows(bot, world):
    member = world.g.members[STAFF]
    msg = TMsg(world.g, world.pub, member, "x")
    kw = bot._agent_kwargs(
        msg,
        world.g,
        member,
        False,
        setting="th",
        lang="th",
        question="x",
        memory=None,
        profile=None,
    )
    names = {f.__name__ for f in kw["extra_tools"]}
    assert {
        "idea_save",
        "idea_get",
        "idea_list",
        "idea_export",
        "idea_share",
        "business_framework",
    } <= names
    from salid.voice_actions import ACTION_TOOL_NAMES

    assert {"idea_save", "idea_export", "idea_share", "business_framework"} <= ACTION_TOOL_NAMES
    assert "โปรเจกต์ไอเดีย" in system_prompt(bot.s, scoped=False, ideas=True)
    assert "โปรเจกต์ไอเดีย" not in system_prompt(bot.s, scoped=False)
    # interjections never get them
    inter = bot._agent_kwargs(
        msg, world.g, member, False, setting="th", lang="th", question="x", memory=None,
        profile=None, interjection=True,
    )  # fmt: skip
    assert "idea_save" not in {f.__name__ for f in inter.get("extra_tools", [])}


async def test_flag_off_means_no_idea_tools(bot, world):
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={"ideas": False}), "x")
    bot.team._cache.clear()
    member = world.g.members[STAFF]
    assert (
        bot._ideas_tools(TMsg(world.g, world.pub, member, "x"), world.g, member, False, "th") == []
    )
    bot.guilds_store.save_settings(GUILD, GuildSettingsUpdate(features={"ideas": True}), "x")
    bot.team._cache.clear()
    assert bot._ideas_tools(TMsg(world.g, world.pub, member, "x"), world.g, member, False, "th")


async def test_bot_context_knows_ranks_privacy_and_people(bot, world):
    boss, staff = world.g.members[BOSS], world.g.members[STAFF]
    bot._ranks_cache.clear()
    tools = kinds(
        bot._ideas_tools(TMsg(world.g, world.pub, staff, "x"), world.g, staff, False, "th")
    )
    tools["idea_save"]("Plan", "secret text")
    tools["idea_share"]("Plan", "superiors")
    # a public channel is not private: the owner is told to use a private chat
    assert tools["idea_get"]("Plan")["status"] == "use_private_chat"
    dm_msg = TMsg(None, world.pub, staff, "x")  # a DM
    mine = kinds(bot._ideas_tools(dm_msg, world.g, staff, False, "th"))
    assert mine["idea_get"]("Plan")["status"] == "ok"
    # the boss outranks the staff member: sees it in a DM, not in a room
    theirs = kinds(bot._ideas_tools(TMsg(None, world.pub, boss, "x"), world.g, boss, True, "th"))
    assert theirs["idea_get"]("Plan")["status"] == "ok"
    room = SimpleNamespace(guild=world.g, channel=world.pub, room=frozenset({BOSS}))
    in_room = kinds(bot._ideas_tools(room, world.g, boss, True, "th"))
    assert in_room["idea_get"]("Plan")["status"] == "use_private_chat"
    # people lookup for sharing
    assert [n for _i, n in bot._ideas_people(world.g, "somchai")] == ["Somchai"]


# ------------------------------------------------------------------ the command center library
def test_library_has_an_ideas_category_with_owner_and_exec_rules(w):
    from salid.library import CATEGORIES

    assert "ideas" in CATEGORIES
    repo = IdeaRepo(w.docs, clock=lambda: NOW)
    for uid, title, vis in (
        (1, "Team idea", "team"),
        (2, "Normal idea", "normal"),
        (3, "Private idea", "private"),
    ):
        ctx_for(repo, uid, "Ann").idea_save(title, "แจ้งเตือน CCTV ไป Apple Watch", "problem")
        d = repo.get(A, uid, slugify_(title))
        d.visibility = vis
        repo.put(A, d)
    # (the fixture guild is A)
    base = "/api/center/orgs/org_t/documents?category=ideas"
    member = {i["title"] for i in w.client.get(base, headers=H["mem"]).json()["items"]}
    execs = {i["title"] for i in w.client.get(base, headers=H["adm"]).json()["items"]}
    assert member == {"Team idea"}
    assert execs == {"Team idea", "Normal idea"}  # never the private one
    item = next(
        i for i in w.client.get(base, headers=H["adm"]).json()["items"] if i["title"] == "Team idea"
    )
    assert item["category"] == "ideas" and item["who"] == "Ann" and "CCTV" in item["preview"]
    detail = w.client.get(f"/api/center/orgs/org_t/documents/{item['id']}", headers=H["adm"]).json()
    assert detail["text"].startswith("# Team idea") and "แจ้งเตือน CCTV" in detail["text"]
    priv = f"/api/center/orgs/org_t/documents/discord-{A}:idea:3_{slugify_('Private idea')}"
    assert w.client.get(priv, headers=H["adm"]).status_code == 404
    ov = w.client.get("/api/center/orgs/org_t/overview", headers=H["adm"]).json()
    assert ov["documents"]["by_category"]["ideas"] == 2


def slugify_(t_):
    from salid.wiki import slugify

    return slugify(t_)
