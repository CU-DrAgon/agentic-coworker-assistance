"""Positions (`salid.positions`): the catalog, level guessing, holders and source precedence; the
agent tools (`set_role`, `org_chart`, `sync_roles`, `positions_catalog`, `knowledge_status`) with
fake hooks; the flag and the hub fields. Offline, no Discord."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from salid import company_brain as cb
from salid import people as P
from salid import people_tools as T
from salid import positions as pos
from salid.coo_store import SQLiteDocStore
from salid.guild_settings import FEATURES, Features, GuildSettings, GuildSettingsUpdate

G = 1


@pytest.fixture
def book():
    return pos.PositionBook(
        SQLiteDocStore(":memory:"), clock=lambda: datetime(2026, 10, 18, tzinfo=UTC)
    )


# ------------------------------------------------------------------ levels
@pytest.mark.parametrize(
    "title, level",
    [
        ("Founder", 1), ("Co-Founder", 1), ("CEO", 1), ("ซีอีโอ", 1), ("ผู้ก่อตั้ง", 1),
        ("Managing Director", 1), ("CTO", 2), ("coo", 2), ("Chief Marketing Officer", 2),
        ("ซีทีโอ", 2), ("Director of Sales", 3), ("VP Engineering", 3), ("ผู้อำนวยการฝ่ายขาย", 3),
        ("Head of Finance", 3), ("หัวหน้าฝ่ายบัญชี", 3), ("Manager", 4), ("Team Lead", 4),
        ("ผจก.", 4), ("ผู้จัดการ", 4), ("หัวหน้า", 4), ("Senior Developer", 5), ("ซีเนียร์", 5),
        ("Staff", 6), ("พนักงานบัญชี", 6), ("Intern", 7), ("นักศึกษาฝึกงาน", 7),
        ("Senior Manager", 4),  # the more senior word wins
        ("Accountant", 0), ("นักออกแบบ", 0), ("", 0),  # never guessed
    ],
)  # fmt: skip
def test_level_of_guesses_from_the_words_only(title, level):
    assert pos.level_of(title) == level


def test_exec_levels_default_to_one_two_three():
    assert [pos.is_exec_level(n) for n in range(0, 6)] == [False, True, True, True, False, False]
    assert pos.is_exec_level(4, 4) and not pos.is_exec_level(0, 4)


# ------------------------------------------------------------------ catalog
def test_the_defaults_serve_until_something_is_stored(book):
    assert book.docs.get(pos.POSITIONS, G, pos.CATALOG_KEY) is None
    assert {p.key for p in book.catalog(G)} >= {"founder", "ceo", "cto", "manager", "intern"}
    assert book.docs.get(pos.POSITIONS, G, pos.CATALOG_KEY) is None  # reading stores nothing


@pytest.mark.parametrize(
    "text, key",
    [("ceo", "ceo"), ("CEO", "ceo"), ("ซีอีโอ", "ceo"), ("Co founder", "co-founder"),
     ("ผู้ก่อตั้ง", "founder"), ("ผจก.", "manager"), ("ผจก", "manager"), ("หัวหน้า", "manager")],
)  # fmt: skip
def test_titles_and_aliases_find_the_same_position_in_either_language(book, text, key):
    assert book.find(G, text).key == key


def test_a_new_title_gets_a_guessed_level_and_a_stable_key(book):
    p, created = book.add(G, "Head of Growth")
    assert created and p.level == 3 and p.key == "head-of-growth"
    again, created = book.add(G, "head of growth")
    assert not created and again.key == p.key
    p2, _ = book.add(G, "CPO", level=2, aliases=["Chief Product Officer", "หัวหน้าฝ่ายผลิตภัณฑ์"])
    assert p2.level == 2 and book.find(G, "chief product officer").key == "cpo"
    unknown, _ = book.add(G, "นักออกแบบ")
    assert unknown.level == 0
    assert book.add(G, "CPO", level=1)[0].level == 1  # an executive may correct the level


def test_the_catalog_is_capped(book):
    for i in range(pos.MAX_POSITIONS - len(book.catalog(G))):
        book.add(G, f"Role {i}")
    assert len(book.catalog(G)) == pos.MAX_POSITIONS
    with pytest.raises(pos.CatalogFull):
        book.add(G, "One too many")
    assert book.add(G, "Role 1")[1] is False  # an existing one is still fine


def test_aliases_are_capped_and_merged(book):
    p, _ = book.add(G, "Lead", aliases=[f"a{i}" for i in range(20)])
    assert len(p.aliases) <= pos.MAX_ALIASES


def test_removing_a_position_removes_its_holders_only(book):
    cto = book.find(G, "CTO")
    book.assign(G, 1, cto.key, "stated")
    book.assign(G, 2, "manager", "stated")
    assert book.remove(G, cto.key) and not book.remove(G, cto.key)
    assert book.of(G, 1) is None and book.of(G, 2) is not None
    assert book.find(G, "CTO") is None


def test_the_discord_role_name_is_the_title_as_typed_or_the_thai_default(book):
    manager = book.find(G, "Manager")
    assert book.discord_role_name(manager, "en") == "Manager"
    assert book.discord_role_name(manager, "th") == "ผู้จัดการ"
    ceo = book.find(G, "CEO")
    assert book.discord_role_name(ceo, "th") == "CEO"
    custom, _ = book.add(G, "Head of Growth")
    assert book.discord_role_name(custom, "th") == "Head of Growth"
    book.rename(G, "manager", "Boss man")
    assert book.discord_role_name(book.get(G, "manager"), "th") == "Boss man"


# ------------------------------------------------------------------ holders
def test_a_weaker_source_never_overwrites_a_stronger_one(book):
    assert book.assign(G, 7, "cto", "discord")
    assert book.assign(G, 7, "ceo", "stated")  # stronger wins
    assert not book.assign(G, 7, "cto", "discord")  # weaker is ignored
    assert not book.assign(G, 7, "cto", "org")  # org (3) < stated (4)
    assert book.of(G, 7)[1].key == "ceo"
    assert book.assign(G, 7, "cto", "org", force=True)  # an explicit new edit in the center
    assert book.of(G, 7)[0].source == "org"
    assert not book.assign(G, 7, "cto", "org")  # same again: nothing changed


def test_unassign_can_be_limited_to_one_source(book):
    book.assign(G, 7, "cto", "stated")
    assert not book.unassign(G, 7, only_source="discord")
    assert book.unassign(G, 7, only_source="stated") and book.of(G, 7) is None


def test_by_level_lists_the_most_senior_first_and_unknown_last(book):
    book.add(G, "Accountant")  # level 0
    book.assign(G, 1, "intern", "stated")
    book.assign(G, 2, "ceo", "stated")
    book.assign(G, 3, "accountant", "stated")
    book.assign(G, 4, "cto", "stated")
    assert [lv for lv, _ in book.by_level(G)] == [1, 2, 7, 0]
    assert book.level_of_user(G, 4) == 2 and book.level_of_user(G, 99) == 0


def test_the_store_survives_a_second_book_on_the_same_docs(book):
    book.assign(G, 5, "ceo", "stated", "business")
    other = pos.PositionBook(book.docs)
    a, p = other.of(G, 5)
    assert (a.source, a.department, p.title) == ("stated", "business", "CEO")


# ------------------------------------------------------------------ tools
class Rig:
    def __init__(self, asker=26, is_exec=False, book=True, **kw):
        self.docs = SQLiteDocStore(":memory:")
        self.repo = P.PeopleRepo(self.docs)
        self.company = cb.CompanyRepo(self.docs)
        self.book = pos.PositionBook(self.docs) if book else None
        self.roster = [
            P.RosterEntry(1, "Mali", "finance", "", 5),
            P.RosterEntry(2, "Tom", "finance", "", 1),
            P.RosterEntry(3, "Bee", "tech", "", 3),
            P.RosterEntry(26, "Newbie", "finance", "", 0),
        ]
        self.tagged: list[int] = []
        self.unlinked_names = ["Tom"]
        self.ctx = T.PeopleCtx(
            guild_id=G, asker_id=asker, asker_name="Asker", is_exec=is_exec, lang="en",
            repo=self.repo, company=self.company, now=lambda: datetime(2026, 10, 18, tzinfo=UTC),
            roster=lambda: self.roster, pdf=lambda title, body: b"%PDF-" + title.encode(),
            resolve_member=lambda name: [(r.user_id, r.name) for r in self.roster
                                         if name.casefold() in r.name.casefold()],
            book=self.book, tag=self._tag, sync=lambda: {"tagged": 2, "untagged": 1},
            unlinked=lambda: list(self.unlinked_names), **kw,
        )  # fmt: skip
        self.tools = {f.__name__: f for f in T.people_tools(self.ctx)}

    def _tag(self, uid):
        self.tagged.append(uid)
        return "pending"


def test_without_a_book_the_eleven_base_tools_only():
    r = Rig(book=False, is_exec=True)
    assert len(r.tools) == 11 and not {"org_chart", "sync_roles", "positions_catalog"} & set(
        r.tools
    )
    assert r.tools["set_role"]("Mali", "CEO")["status"] == "saved"  # the old behaviour


def test_with_a_book_the_three_position_tools_join_and_share_the_outbox():
    r = Rig()
    assert {"org_chart", "sync_roles", "positions_catalog"} <= set(r.tools)
    assert all(f.outbox is r.ctx.outbox for f in r.tools.values())


def test_set_role_by_an_executive_assigns_the_position_and_asks_for_the_tag():
    r = Rig(is_exec=True)
    out = r.tools["set_role"]("Mali", "CTO", "tech")
    assert out["status"] == "saved" and out["position"] == "CTO" and out["level"] == 2
    assert out["tagged"] == "pending" and r.tagged == [1]
    a, p = r.book.of(G, 1)
    assert (p.key, a.source, a.department) == ("cto", "stated", "tech")
    assert r.repo.get(G, 1).role_title == "CTO"  # the dossier follows


def test_set_role_with_a_thai_alias_uses_the_same_position():
    r = Rig(is_exec=True)
    r.tools["set_role"]("Mali", "ซีอีโอ")
    assert r.book.of(G, 1)[1].key == "ceo"


def test_a_new_title_creates_a_catalog_entry_with_a_guessed_level():
    r = Rig(is_exec=True)
    out = r.tools["set_role"]("Bee", "Head of Platform")
    assert out["level"] == 3 and r.book.find(G, "Head of Platform") is not None


def test_nobody_makes_themself_a_boss_but_a_junior_title_is_fine():
    r = Rig(asker=26)
    refused = r.tools["set_role"]("Newbie", "CEO")
    assert refused["position"] == "needs_executive" and r.book.of(G, 26) is None
    assert r.repo.get(G, 26).role_title == "CEO"  # the dossier title still works as before
    ok = r.tools["set_role"]("Newbie", "Intern")
    assert ok["level"] == 7 and r.book.of(G, 26)[1].key == "intern" and r.tagged == [26]
    assert r.tools["set_role"]("Newbie", "Accountant")["position"] == "needs_executive"  # level 0
    other = r.tools["set_role"]("Mali", "Staff")
    assert other["error"] == "not_allowed"


def test_org_chart_is_top_down_and_lists_who_has_no_position():
    r = Rig(is_exec=True)
    r.tools["set_role"]("Mali", "CEO")
    r.tools["set_role"]("Bee", "CTO", "tech")
    out = r.tools["org_chart"]()
    assert [lv["level"] for lv in out["levels"]] == [1, 2]
    assert out["levels"][0]["people"] == [
        {"name": "Mali", "position": "CEO", "department": "Finance"}
    ]
    assert out["levels"][1]["people"][0]["name"] == "Bee"
    assert out["no_position"] == ["Newbie", "Tom"] and out["no_position_count"] == 2
    assert "## Founder / CEO" in out["text"] and "- Mali: CEO (Finance)" in out["text"]
    assert out["not_linked_to_center"] == ["Tom"] and "attached_file" not in out


def test_org_chart_for_a_member_hides_who_is_unlinked_and_a_big_company_gets_a_pdf():
    r = Rig()
    assert "not_linked_to_center" not in r.tools["org_chart"]() and (
        r.tools["org_chart"]()["not_linked_count"] == 1
    )
    r = Rig(is_exec=True)
    r.roster += [P.RosterEntry(100 + i, f"P{i}", "", "", 0) for i in range(30)]
    r.tools["set_role"]("Mali", "CEO")
    out = r.tools["org_chart"]()
    assert out["attached_file"] == "org-chart.pdf" and r.ctx.outbox[0][0] == "org-chart.pdf"


def test_org_chart_skips_people_who_are_not_in_the_server_any_more():
    r = Rig(is_exec=True)
    r.book.assign(G, 999, "ceo", "stated")  # left the server
    assert r.tools["org_chart"]()["levels"] == []


def test_sync_roles_and_the_catalog_are_for_executives():
    r = Rig()
    assert r.tools["sync_roles"]()["error"] == "not_allowed"
    assert r.tools["positions_catalog"]()["error"] == "not_allowed"
    r = Rig(is_exec=True)
    assert r.tools["sync_roles"]() == {"status": "synced", "tagged": 2, "untagged": 1}


def test_positions_catalog_show_add_remove():
    r = Rig(is_exec=True)
    show = r.tools["positions_catalog"]()
    assert show["positions"][0]["level"] == 1 and any(
        p["title"] == "CTO" for p in show["positions"]
    )
    add = r.tools["positions_catalog"]("add", "CPO", 2, "Chief Product Officer, หัวหน้าผลิตภัณฑ์")
    assert add["status"] == "added" and add["level"] == 2 and len(add["aliases"]) == 2
    assert r.tools["positions_catalog"]("add", "CPO")["status"] == "updated"
    assert r.tools["positions_catalog"]("add", "")["error"].startswith("give")
    gone = r.tools["positions_catalog"]("remove", "chief product officer")
    assert gone["status"] == "removed" and r.book.find(G, "CPO") is None
    assert r.tools["positions_catalog"]("remove", "CPO")["error"] == "not_found"
    assert r.tools["positions_catalog"]("explode", "x")["error"].startswith("action")


def test_knowledge_status_reports_positions_and_who_is_not_linked():
    r = Rig(is_exec=True)
    r.tools["set_role"]("Mali", "CEO")
    out = r.tools["knowledge_status"]()
    assert out["positions"] == {"catalog": len(r.book.catalog(G)), "assigned": 1,
                                "without_position_count": 3}  # fmt: skip
    assert out["not_linked_to_center"] == ["Tom"] and out["not_linked_to_center_count"] == 1
    assert "positions" not in Rig(is_exec=True, book=False).tools["knowledge_status"]()


# ------------------------------------------------------------------ flag, hub, prompt
def test_the_flag_is_default_on_with_hub_text_and_the_new_settings():
    assert "role_sync" in FEATURES and Features().role_sync is True
    s = GuildSettings()
    assert s.exec_max_level == 3 and s.department_roles is False
    assert GuildSettingsUpdate(exec_max_level=4).exec_max_level == 4
    for bad in (0, 5, "x", None):
        assert GuildSettingsUpdate(exec_max_level=bad).exec_max_level == 3
    html = (Path(__file__).parents[2] / "src/meeting_scribe/static/hub.html").read_text("utf-8")
    for needle in ('{ id: "role_sync" }', "f_role_sync_n", "f_role_sync_d", "f_role_sync_w",
                   'id="exec-max-level"', 'id="department-roles"', "exec_max_level:",
                   "department_roles:", "role_sync: features.role_sync"):  # fmt: skip
        assert needle in html, needle


def test_the_prompt_names_the_new_tools_in_both_languages():
    for lang in ("th", "en"):
        for tool in ("org_chart", "sync_roles", "positions_catalog", "set_role"):
            assert tool in T.KNOWLEDGE_PROMPT[lang]
    assert "ผังองค์กร" in T.KNOWLEDGE_PROMPT["th"] and "org chart" in T.KNOWLEDGE_PROMPT["en"]


def test_level_labels_exist_in_both_languages():
    from salid.i18n import t

    for n in range(0, 8):
        assert t(f"pos_level_{n}", "th") != t(f"pos_level_{n}", "en")
