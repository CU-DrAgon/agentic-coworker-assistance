"""The company-knowledge tools (`salid.people_tools`): routing, ranking, visibility, left people,
overview, how-do-we, onboarding brief + PDF, handover and role permissions. Hooks are fakes."""

from datetime import UTC, datetime

import pytest

from salid import company_brain as cb
from salid import people as P
from salid import people_tools as T
from salid.coo_store import SQLiteDocStore
from salid.wiki import WikiPage

G, DAY = 1, "2026-10-14"
PUB, PRIV = 10, 11
NAMES = {PUB: "general", PRIV: "mgmt"}


def ev(cid=PUB, day=DAY, mid=1, kind="chat"):
    return P.Evidence(day=day, channel_id=cid, message_id=mid, kind=kind)


class Rig:
    def __init__(self, asker=26, is_exec=False, readable=(PUB,), see_all=False, **kw):
        self.docs = SQLiteDocStore(":memory:")
        self.repo = P.PeopleRepo(self.docs)
        self.company = cb.CompanyRepo(self.docs)
        self.roster = [
            P.RosterEntry(1, "Mali", "finance", "Accountant", 5),
            P.RosterEntry(2, "Tom", "finance", "Clerk", 1),
            P.RosterEntry(3, "Bee", "tech", "Developer", 3),
            P.RosterEntry(26, "Newbie", "finance", "", 0),
        ]
        self.calls = []
        self.ctx = T.PeopleCtx(
            guild_id=G, asker_id=asker, asker_name="Asker", is_exec=is_exec, lang="en",
            repo=self.repo, company=self.company, now=lambda: datetime(2026, 10, 15, tzinfo=UTC),
            guild_name="Coffee Co", readable=lambda: frozenset(readable), see_all=lambda: see_all,
            channel_name=lambda c: NAMES.get(c, str(c)), roster=lambda: self.roster,
            channels=lambda: ["#general"], projects=lambda: [
                {"name": "Month-end tool", "status": "in_progress", "department": "finance"}],
            resolve_member=lambda name: [(r.user_id, r.name) for r in self.roster
                                         if name.casefold() in r.name.casefold()],
            pdf=lambda title, body: b"%PDF-" + title.encode(),
            **kw,
        )  # fmt: skip
        self.tools = {f.__name__: f for f in T.people_tools(self.ctx)}

    def learn(self, uid, name, **fn):
        def apply(d):
            for k, v in fn.items():
                getattr(P, k)(getattr(d, v[0]), *v[1:])
            return True

        self.repo.update(G, uid, name, apply)


@pytest.fixture
def rig():
    r = Rig()
    r.learn(1, "Mali", merge_topic=("expertise", "invoice approval", ev(), DAY, 5),
            merge_fact=("responsibilities", "approves supplier invoices", ev(), DAY))  # fmt: skip
    r.learn(2, "Tom", merge_topic=("expertise", "invoice approval", ev(), DAY, 1))
    return r


def test_the_ten_tools_exist_and_share_one_outbox(rig):
    assert set(rig.tools) == {"company_overview", "who_to_ask", "person_profile", "how_do_we",
                              "onboarding_brief", "handover", "set_role", "person_left",
                              "who_works_with_whom", "knowledge_status"}  # fmt: skip
    assert all(f.outbox is rig.ctx.outbox for f in rig.tools.values())


# ------------------------------------------------------------------ who_to_ask
def test_who_to_ask_ranks_with_reasons_and_sources(rig):
    out = rig.tools["who_to_ask"]("invoice approval")
    assert [p["name"] for p in out["people"]] == ["Mali", "Tom"]
    why = out["people"][0]["why"][0]
    assert why["text"] == "invoice approval" and why["from"] == "#general chat 2026-10-14"


def test_who_to_ask_hides_what_the_asker_cannot_read_and_never_guesses(rig):
    rig.learn(3, "Bee", merge_topic=("expertise", "the server", ev(cid=PRIV), DAY, 9))
    out = rig.tools["who_to_ask"]("the server")
    assert out["people"] == [] and "don't know" in out["note"]
    reader = Rig(readable=(PUB, PRIV))
    reader.ctx.repo = rig.repo
    assert [p["name"] for p in reader.tools["who_to_ask"]("the server")["people"]] == ["Bee"]
    boss = Rig(is_exec=True, readable=(PUB,), see_all=True)  # an executive asking alone reads all
    boss.ctx.repo = rig.repo
    assert [p["name"] for p in boss.tools["who_to_ask"]("the server")["people"]] == ["Bee"]


def test_who_to_ask_for_someone_who_left_points_to_the_successor_and_the_handover(rig):
    rig.repo.update(G, 3, "Old", lambda d: P.merge_topic(d.expertise, "tax filing", ev(), DAY, 9))
    rig.repo.update(G, 3, "Old", lambda d: setattr(d, "status", "left") or setattr(d, "left_at", "2026-10-01") or setattr(d, "successor_id", 2) or setattr(d, "handover", "# notes") or True)  # fmt: skip
    out = rig.tools["who_to_ask"]("tax filing")
    p = out["people"][0]
    assert (
        p["left"]
        and p["successor"] == "Tom"
        and p["handover_available"]
        and p["left_at"] == "2026-10-01"
    )


def test_who_to_ask_falls_back_to_roles_from_the_roster(rig):
    out = rig.tools["who_to_ask"]("accountant")
    assert out["people"][0]["name"] == "Mali" and "role" in out["people"][0]["why"]


# ------------------------------------------------------------------ person_profile
def test_person_profile_shows_work_knowledge_and_hides_unreadable_parts(rig):
    rig.learn(1, "Mali", merge_fact=("systems", "Express accounting", ev(cid=PRIV), DAY))
    rig.learn(1, "Mali", merge_fact=("recurring", "month-end close", ev(), DAY))
    out = rig.tools["person_profile"]("mali")
    assert out["name"] == "Mali" and out["department"] == "Finance"
    assert [r["text"] for r in out["responsibilities"]] == ["approves supplier invoices"]
    assert out["responsibilities"][0]["from"] == "#general chat 2026-10-14"
    assert out["recurring_duties"][0]["text"] == "month-end close"
    assert out["systems"] == [] and out["hidden_items"] == 1
    assert "Express" not in repr(out)  # counted, never described


def test_person_profile_unknown_ambiguous_and_empty(rig):
    assert rig.tools["person_profile"]("zzz")["error"] == "not_found"
    rig.roster.append(P.RosterEntry(4, "Mali B", "tech", "", 0))
    rig.repo.update(G, 4, "Mali B", lambda d: True)
    assert rig.tools["person_profile"]("Mali")["name"] == "Mali"  # exact name wins
    assert rig.tools["person_profile"]("Ma")["error"] == "ambiguous"
    empty = rig.tools["person_profile"]("Bee")  # a member Salid knows nothing about yet
    assert empty["name"] == "Bee" and "not learned" in empty["note"]


# ------------------------------------------------------------------ company_overview
def test_company_overview_composes_playbook_org_contacts_and_places():
    r = Rig()
    pb = r.company.get(G)
    pb.about, pb.industry = "We roast and sell coffee", "coffee"
    pb.sop = [cb.Sop(id="s1", title="Refund", steps=["ask", "pay"])]
    pb.key_contacts = [cb.KeyContact(name="ACME", context="green beans", count=4)]
    r.company.save(G, pb)
    r.ctx.wiki_pages = lambda: [WikiPage(slug="a", title="Opening checklist", body="x")]
    out = r.tools["company_overview"]()
    assert (
        out["company"] == "Coffee Co" and out["industry"] == "coffee" and out["people_count"] == 4
    )
    fin = next(d for d in out["departments"] if d["department"] == "Finance")
    assert fin["head"] == "Mali"
    assert out["key_contacts"][0]["name"] == "ACME"
    where = out["where_things_live"]
    assert where["channels"] == ["#general"] and where["wiki_pages"] == ["Opening checklist"]
    assert where["projects"] == ["Month-end tool"] and where["procedures"] == ["Refund"]


def test_company_overview_of_an_unknown_company_says_so():
    r = Rig()
    r.roster = []
    out = r.tools["company_overview"]()
    assert "not learned" in out["note"] and out["departments"] == []


# ------------------------------------------------------------------ how_do_we
def test_how_do_we_assembles_sops_wiki_people_and_decisions():
    r = Rig()
    pb = r.company.get(G)
    pb.sop = [cb.Sop(id="s1", title="Refund a customer", steps=["check the order", "pay back"])]
    r.company.save(G, pb)
    r.learn(1, "Mali", merge_fact=("recurring", "processes refunds every Friday", ev(), DAY))
    r.ctx.wiki_pages = lambda: [
        WikiPage(slug="r", title="Refund policy", body="Refunds within 7 days")
    ]
    r.ctx.decisions = lambda topic: [
        {"text": "Refunds over 5000 need the boss", "day": DAY, "channel_id": PUB}
    ]
    out = r.tools["how_do_we"]("refund")
    assert out["found"] and out["sops"][0]["steps"] == ["check the order", "pay back"]
    assert out["wiki_pages"][0]["title"] == "Refund policy"
    assert out["people_say"][0]["who"] == "Mali" and out["people_say"][0]["from"].startswith(
        "#general"
    )
    assert out["decisions"][0]["from"] == "#general chat 2026-10-14"


def test_how_do_we_with_nothing_says_unknown_instead_of_inventing(rig):
    out = rig.tools["how_do_we"]("launch a rocket")
    assert out["found"] is False and "don't know" in out["note"]
    assert out["sops"] == out["wiki_pages"] == out["people_say"] == []


def test_how_do_we_hides_habits_learned_in_unreadable_channels(rig):
    rig.learn(
        1, "Mali", merge_fact=("recurring", "reconciles the vault every Monday", ev(cid=PRIV), DAY)
    )
    assert rig.tools["how_do_we"]("vault")["found"] is False


# ------------------------------------------------------------------ onboarding_brief
def test_onboarding_brief_has_the_sections_a_pdf_and_offers_the_wiki(rig):
    rig.learn(1, "Mali", merge_fact=("systems", "Express accounting", ev(), DAY))
    rig.learn(1, "Mali", merge_fact=("recurring", "month-end close", ev(), DAY))
    out = rig.tools["onboarding_brief"](role_or_department="finance")
    body = out["brief_markdown"]
    for h in ("What the company does", "Your department and manager", "People to meet",
              "Systems and access", "Recurring work", "Current projects", "Open questions"):  # fmt: skip
        assert f"## {h}" in body
    assert (
        "Head: Mali" in body and "Express accounting" in body and "month-end close (Mali)" in body
    )
    assert "Month-end tool" in body and out["department"] == "Finance"
    assert out["attached_file"].endswith(".pdf") and rig.ctx.outbox[0][1].startswith(b"%PDF-")
    assert "offer" in out and "wiki" not in out


def test_onboarding_brief_for_a_person_and_the_gaps_are_named():
    r = Rig()
    out = r.tools["onboarding_brief"](for_user="Newbie")
    body = out["brief_markdown"]
    assert out["department"] == "finance" or out["department"] == "Finance"
    assert "No standard procedures" in body and "does not know yet which systems" in body


def test_onboarding_brief_saved_to_the_wiki_only_when_asked_and_keeps_its_sources(rig):
    saved = []
    rig.ctx.save_wiki = lambda title, body, chans: (
        saved.append((title, chans)) or {"status": "saved"}
    )
    rig.learn(1, "Mali", merge_fact=("systems", "Express accounting", ev(), DAY))
    out = rig.tools["onboarding_brief"]("finance", "", True)
    assert out["wiki"] == {"status": "saved"} and saved[0][1] == [PUB]


def test_onboarding_brief_mentions_a_predecessors_handover(rig):
    rig.repo.update(G, 9, "Old Acc", lambda d: setattr(d, "status", "left") or setattr(d, "department", "finance") or setattr(d, "handover", "x") or True)  # fmt: skip
    body = rig.tools["onboarding_brief"]("finance")["brief_markdown"]
    assert "handover" in body.casefold() and "Old Acc" in body


# ------------------------------------------------------------------ handover, roles, leaving
def test_handover_is_for_executives_the_person_and_the_successor_only(rig):
    rig.repo.update(G, 3, "Bee", lambda d: setattr(d, "handover", "# Handover text") or setattr(d, "successor_id", 26) or True)  # fmt: skip
    assert (
        rig.tools["handover"]("Bee")["handover"] == "# Handover text"
    )  # the asker (26) is the successor
    other = Rig(asker=77)
    other.repo = rig.repo
    other.ctx.repo = rig.repo
    assert other.tools["handover"]("Bee")["error"] == "not_allowed"
    boss = Rig(asker=77, is_exec=True)
    boss.ctx.repo = rig.repo
    out = boss.tools["handover"]("Bee")
    assert out["handover"] == "# Handover text" and out["attached_file"].startswith("handover-bee")
    assert boss.ctx.outbox


def test_an_executive_can_generate_and_name_a_successor():
    r = Rig(asker=77, is_exec=True)
    made = []

    def gen(uid, name, succ):
        made.append((uid, succ))
        r.repo.update(G, uid, name, lambda d: setattr(d, "handover", "## Role\n- x") or True)
        return {"text": "## Role\n- x"}

    r.ctx.gen_handover = gen
    r.repo.update(G, 3, "Bee", lambda d: True)
    out = r.tools["handover"]("Bee", successor="Tom")
    assert made == [(3, 2)] and out["written_now"] and "## Role" in out["handover"]
    assert (
        r.tools["handover"]("Bee", successor="Tom")["written_now"] is False
    )  # kept, not rewritten
    regen = r.tools["handover"]("Bee", regenerate=True)
    assert regen["written_now"] and len(made) == 2
    assert Rig(asker=77).tools["handover"]("Bee", successor="Tom")["error"] == "not_allowed"


def test_set_role_is_for_executives_or_the_person_and_states_the_role():
    r = Rig(asker=77, is_exec=True)
    assert r.tools["set_role"]("Tom", "Senior clerk", "finance")["status"] == "saved"
    d = r.repo.get(G, 2)
    assert (d.role_title, d.role_source, d.department, d.dept_source) == (
        "Senior clerk",
        "stated",
        "finance",
        "stated",
    )
    me = Rig(asker=3)
    me.ctx.repo = r.repo
    assert me.tools["set_role"]("Bee", "Lead dev")["status"] == "saved"
    assert me.tools["set_role"]("Tom", "Boss")["error"] == "not_allowed"
    assert me.tools["set_role"]("Bee")["error"].startswith("give")


def test_person_left_is_executive_only_and_delegates_to_the_flow():
    calls = []
    r = Rig(
        asker=77,
        is_exec=True,
        on_left=lambda uid, name, succ: calls.append((uid, name, succ)) or {"status": "recorded"},
    )
    assert r.tools["person_left"]("Tom", successor="Bee") == {"status": "recorded"}
    assert calls == [(2, "Tom", 3)]
    assert Rig(asker=5).tools["person_left"]("Tom")["error"] == "not_allowed"


def test_a_tool_never_raises(rig):
    rig.ctx.roster = lambda: 1 / 0
    assert rig.tools["company_overview"]()["status"] == "error"


# ------------------------------------------------------------------ who works with whom, status
def _contacts(rig):
    def add(uid, name, who, other, about, cid=PUB, n=1):
        for i in range(n):
            rig.repo.update(G, uid, name, lambda d, i=i: P.merge_contact(
                d.contacts, P.HeardContact(who=who, user_id=other, about=about),
                ev(cid=cid, mid=uid * 100 + i), f"2026-10-{1 + i:02d}"))  # fmt: skip

    add(2, "Tom", "Mali", 1, "invoices", n=3)
    add(3, "Bee", "Mali", 1, "payroll")
    add(1, "Mali", "ACME", 0, "supply", n=2)
    add(3, "Bee", "ACME", 0, "demo")
    add(3, "Bee", "Tom", 2, "secret plan", cid=PRIV)


def test_who_works_with_whom_overview_shows_hubs_bridges_and_pairs(rig):
    _contacts(rig)
    out = rig.tools["who_works_with_whom"]()
    assert out["hubs"][0]["name"] == "Mali" and out["hubs"][0]["named_by_others"] == 4
    assert "invoices" in out["hubs"][0]["contacted_about"]
    assert set(out["strongest_pairs"][0]["between"]) == {"Mali", "Tom"}
    assert all("secret plan" not in repr(p) for p in out["strongest_pairs"])  # PRIV unreadable
    assert out["bridges"] == [] or all(len(b["departments"]) >= 2 for b in out["bridges"])


def test_who_works_with_whom_for_a_person_and_for_nobody_known(rig):
    assert rig.tools["who_works_with_whom"]()["found"] is False
    _contacts(rig)
    out = rig.tools["who_works_with_whom"]("Bee")
    assert {c["who"] for c in out["contacts"]} == {"ACME", "Mali"}
    assert next(c for c in out["contacts"] if c["who"] == "ACME")["outside_party"] is True
    assert any(x["name"] == "Mali" for x in out["also_dealing_with_the_same_parties"])
    boss = Rig(is_exec=True, readable=(PUB,), see_all=True)
    boss.ctx.repo = rig.repo
    seen = boss.tools["who_works_with_whom"]("Bee")
    assert "Tom" in {c["who"] for c in seen["contacts"]}  # the exec alone reads the private one
    assert rig.tools["who_works_with_whom"]("zzz")["error"] == "not_found"


def test_company_overview_has_a_collaboration_line(rig):
    assert "collaboration" not in rig.tools["company_overview"]()
    _contacts(rig)
    line = rig.tools["company_overview"]()["collaboration"]
    assert line.startswith("most connected: Mali (4 contacts)")


def test_knowledge_status_is_executive_only_and_counts(rig):
    assert rig.tools["knowledge_status"]()["status"] == "error"
    boss = Rig(is_exec=True)
    boss.ctx.repo = rig.repo
    _contacts(rig)
    rig.repo.put(G, P.Dossier(user_id=999, name="สลิด", role_title="chatbot"))
    boss.company.save(G, cb.CompanyPlaybook(org_built_at="2026-10-15T01:00:00"))
    boss.ctx.company = boss.company
    out = boss.tools["knowledge_status"]()
    assert out["people_with_dossiers"] == 3 and out["bots_excluded"] == 1
    assert out["topics"] == 2 and out["contacts"] == 5 and out["map_built_at"]
    assert out["last_learned"] and out["missing"]["no_dossier_count"] == 1  # Newbie
    assert out["missing"]["no_role"] == ["Newbie"]
