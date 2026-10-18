"""Verified identities: Member.identities, the OrgStore helpers, link / attach codes and the
typed-command parser. Runs against SQLite and the fake-Firestore doc store."""

from datetime import UTC, datetime, timedelta

import pytest
from fake_firestore import FakeFirestore

from salid.coo_store import FirestoreDocStore, SQLiteDocStore
from salid.identity_links import (
    CODE_TTL,
    KIND,
    MAX_LIVE_CODES,
    CodeError,
    IdentityLinks,
    NotAMember,
    TooManyCodes,
    format_code,
    parse_command,
)
from salid.orgs import (
    IdentityTaken,
    Member,
    OrgStore,
    UnknownMember,
    Workspace,
    workspace_id,
)

T0 = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


@pytest.fixture(params=["sqlite", "firestore"])
def docs(request):
    if request.param == "sqlite":
        return SQLiteDocStore(":memory:")
    return FirestoreDocStore(client=FakeFirestore())


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def orgs(docs, clock):
    o = OrgStore(docs, clock=clock)
    o.create(
        "Cafe",
        "boss@example.com",
        org_id="org_a",
        members=[
            Member(email="boss@example.com", role="owner", discord_user_id="555"),
            Member(email="mew@example.com", role="member"),
            Member(email="ann@example.com", role="member"),
        ],
    )
    o.create("Other", "o@example.com", org_id="org_b")
    return o


@pytest.fixture
def links(orgs, clock):
    return IdentityLinks(orgs, clock=clock)


def attach(orgs, org_id, platform, tenant):
    orgs.attach_workspace(
        org_id,
        Workspace(id=workspace_id(platform, tenant), platform=platform, external_id=tenant),
    )


def digits(code: str) -> str:
    return code.replace("-", "")


# ------------------------------------------------------------------ parser
@pytest.mark.parametrize(
    "text",
    [
        "link 1234-5678",
        "Link 12345678",
        "  link   1234 5678 ",
        "/link 1234-5678",
        "/link@SalidBot 1234-5678",
        "link: 1234–5678",
        "เชื่อมบัญชี 1234-5678",
        "เชื่อมบัญชี1234-5678",
    ],
)
def test_parse_command_accepts_the_tolerant_forms(text):
    assert parse_command(text) == "12345678"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "link",
        "link 1234-567",
        "link 1234-56789",
        "please link 1234-5678",
        "link 1234-5678 thanks",
        "unlink 1234-5678",
        "what is 1234-5678",
        "link abcd-efgh",
    ],
)
def test_parse_command_ignores_everything_else(text):
    assert parse_command(text) is None


def test_format_code():
    assert format_code("00120034") == "0012-0034"


# ------------------------------------------------------------------ Member / OrgStore
def test_member_identity_reads_discord_from_the_legacy_field(orgs):
    boss = orgs.get("org_a").members[0]
    assert boss.identity("discord") == "555"
    assert boss.identity("line") is None
    assert boss.all_identities() == {"discord": "555"}


def test_link_and_lookup_identity(orgs):
    orgs.link_identity("org_a", "MEW@example.com", "line", "U-line-1")
    m = orgs.member_by_identity("org_a", "line", "U-line-1")
    assert m and m.email == "mew@example.com" and m.identities == {"line": "U-line-1"}
    assert orgs.member_by_identity("org_a", "telegram", "U-line-1") is None
    assert orgs.member_by_identity("org_b", "line", "U-line-1") is None
    assert orgs.member_by_identity("org_a", "discord", "555").email == "boss@example.com"


def test_a_second_member_cannot_claim_the_same_account(orgs):
    orgs.link_identity("org_a", "mew@example.com", "telegram", "42")
    with pytest.raises(IdentityTaken):
        orgs.link_identity("org_a", "ann@example.com", "telegram", "42")
    assert orgs.member_by_identity("org_a", "telegram", "42").email == "mew@example.com"
    # the same member re-linking is fine and replaces the old account
    orgs.link_identity("org_a", "mew@example.com", "telegram", "43")
    assert orgs.member_by_identity("org_a", "telegram", "42") is None
    assert orgs.member_by_identity("org_a", "telegram", "43").email == "mew@example.com"


def test_the_same_account_may_belong_to_members_of_different_orgs(orgs):
    orgs.create(
        "Third",
        "mew@example.com",
        org_id="org_c",
    )
    orgs.link_identity("org_a", "mew@example.com", "line", "U1")
    orgs.link_identity("org_c", "mew@example.com", "line", "U1")
    assert orgs.member_by_identity("org_c", "line", "U1").email == "mew@example.com"


def test_link_identity_validation(orgs):
    with pytest.raises(UnknownMember):
        orgs.link_identity("org_a", "nobody@example.com", "line", "U1")
    with pytest.raises(KeyError):
        orgs.link_identity("org_zzz", "mew@example.com", "line", "U1")
    with pytest.raises(ValueError):
        orgs.link_identity("org_a", "mew@example.com", "clickup", "U1")
    with pytest.raises(ValueError):
        orgs.link_identity("org_a", "mew@example.com", "line", "  ")


def test_linking_discord_keeps_using_the_legacy_field(orgs):
    orgs.link_identity("org_a", "mew@example.com", "discord", "777")
    m = orgs.get("org_a").members[1]
    assert m.discord_user_id == "777" and "discord" not in m.identities
    assert orgs.member_by_identity("org_a", "discord", "777").email == "mew@example.com"


def test_unlink_identity(orgs):
    orgs.link_identity("org_a", "mew@example.com", "whatsapp", "66800000000")
    assert orgs.unlink_identity("org_a", "mew@example.com", "whatsapp") is True
    assert orgs.member_by_identity("org_a", "whatsapp", "66800000000") is None
    assert orgs.unlink_identity("org_a", "mew@example.com", "whatsapp") is False
    assert orgs.unlink_identity("org_a", "ghost@example.com", "whatsapp") is False


def test_detaching_the_last_workspace_of_a_platform_forgets_those_links(orgs):
    attach(orgs, "org_a", "line", "Ubot")
    orgs.link_identity("org_a", "mew@example.com", "line", "U1")
    orgs.link_identity("org_a", "mew@example.com", "telegram", "9")
    orgs.detach_workspace("org_a", workspace_id("line", "Ubot"))
    m = orgs.get("org_a").members[1]
    assert m.identities == {"telegram": "9"}


def test_detaching_one_of_two_workspaces_keeps_the_links(orgs):
    attach(orgs, "org_a", "slack", "T1")
    attach(orgs, "org_a", "slack", "T2")
    orgs.link_identity("org_a", "mew@example.com", "slack", "U1")
    orgs.detach_workspace("org_a", workspace_id("slack", "T1"))
    assert orgs.get("org_a").members[1].identities == {"slack": "U1"}


def test_drop_platform_identities_for_a_revoked_workspace(orgs):
    attach(orgs, "org_a", "slack", "T1")
    orgs.link_identity("org_a", "mew@example.com", "slack", "U1")
    orgs.drop_platform_identities("org_a", "slack", keep_workspace=workspace_id("slack", "T1"))
    assert orgs.get("org_a").members[1].identities == {}
    # never touches Discord
    orgs.drop_platform_identities("org_a", "discord")
    assert orgs.get("org_a").members[0].discord_user_id == "555"


# ------------------------------------------------------------------ identity codes
def test_code_shape_and_storage(links, docs):
    issued = links.create_code("org_a", "Mew@example.com", "line")
    assert (
        len(issued.code) == 9 and issued.code[4] == "-" and issued.code.replace("-", "").isdigit()
    )
    assert issued.kind == "identity" and issued.platform == "line"
    doc = docs.get(KIND, 0, digits(issued.code))
    assert doc["org_id"] == "org_a" and doc["email"] == "mew@example.com" and doc["used"] is False
    assert datetime.fromisoformat(issued.expires_at) == T0 + CODE_TTL


def test_redeem_links_the_account_once(links, orgs):
    attach(orgs, "org_a", "line", "Ubot")
    code = links.create_code("org_a", "mew@example.com", "line").code
    done = links.redeem(code, "line", "U-mew", "Ubot")
    assert (done.kind, done.org_id, done.email) == ("identity", "org_a", "mew@example.com")
    assert orgs.member_by_identity("org_a", "line", "U-mew").email == "mew@example.com"
    with pytest.raises(CodeError) as e:
        links.redeem(code, "line", "U-other", "Ubot")  # single use
    assert e.value.reason == "code already used"
    assert orgs.member_by_identity("org_a", "line", "U-other") is None


def test_expired_code_is_refused(links, clock, orgs):
    code = links.create_code("org_a", "mew@example.com", "telegram").code
    clock.advance(minutes=15, seconds=1)
    with pytest.raises(CodeError) as e:
        links.redeem(code, "telegram", "42", "bot1")
    assert e.value.reason == "code expired"
    assert orgs.member_by_identity("org_a", "telegram", "42") is None


def test_code_works_right_up_to_the_expiry(links, clock):
    code = links.create_code("org_a", "mew@example.com", "telegram").code
    clock.advance(minutes=14, seconds=59)
    assert links.redeem(code, "telegram", "42", "bot1").email == "mew@example.com"


def test_code_for_another_platform_is_refused_and_not_burned(links):
    code = links.create_code("org_a", "mew@example.com", "line").code
    with pytest.raises(CodeError) as e:
        links.redeem(code, "telegram", "42", "bot1")
    assert e.value.reason == "wrong platform"
    assert links.redeem(code, "line", "U1", "Ubot").email == "mew@example.com"


@pytest.mark.parametrize("bad", ["", "1234", "0000-0000", "abcd-efgh", "1234-5678-9"])
def test_unknown_or_malformed_codes_are_refused(links, bad):
    with pytest.raises(CodeError):
        links.redeem(bad, "line", "U1", "Ubot")


def test_account_already_linked_to_another_member_is_refused(links, orgs):
    orgs.link_identity("org_a", "ann@example.com", "line", "U-shared")
    code = links.create_code("org_a", "mew@example.com", "line").code
    with pytest.raises(CodeError) as e:
        links.redeem(code, "line", "U-shared", "Ubot")
    assert e.value.reason == "account linked to another member"
    assert orgs.member_by_identity("org_a", "line", "U-shared").email == "ann@example.com"


def test_tenant_owned_by_another_org_is_refused(links, orgs):
    attach(orgs, "org_b", "line", "Ubot")
    code = links.create_code("org_a", "mew@example.com", "line").code
    with pytest.raises(CodeError) as e:
        links.redeem(code, "line", "U1", "Ubot")
    assert e.value.reason == "tenant belongs to another org"
    assert orgs.member_by_identity("org_a", "line", "U1") is None


def test_unattached_tenant_may_still_link_but_only_for_the_code_org(links, orgs):
    code = links.create_code("org_a", "mew@example.com", "line").code
    links.redeem(code, "line", "U1", "Unattached")
    assert orgs.member_by_identity("org_a", "line", "U1") is not None
    assert links.identity_of("line", "Unattached", "U1") is None  # not an org tenant: a guest


def test_removed_member_cannot_redeem(links, orgs):
    code = links.create_code("org_a", "ann@example.com", "line").code
    org = orgs.get("org_a")
    org.members = [m for m in org.members if m.email != "ann@example.com"]
    orgs.save(org)
    with pytest.raises(CodeError):
        links.redeem(code, "line", "U1", "Ubot")


def test_at_most_five_live_codes_per_member(links, clock):
    for _ in range(MAX_LIVE_CODES):
        links.create_code("org_a", "mew@example.com", "line")
    with pytest.raises(TooManyCodes):
        links.create_code("org_a", "mew@example.com", "telegram")
    links.create_code("org_a", "ann@example.com", "line")  # someone else is unaffected
    clock.advance(minutes=16)
    links.create_code("org_a", "mew@example.com", "line")  # the old ones expired


def test_used_codes_do_not_count_as_live(links):
    for i in range(MAX_LIVE_CODES):
        code = links.create_code("org_a", "mew@example.com", "line").code
        links.redeem(code, "line", f"U{i}", "Ubot")
    links.create_code("org_a", "mew@example.com", "line")


def test_only_members_get_codes_and_discord_takes_none(links):
    with pytest.raises(NotAMember):
        links.create_code("org_a", "stranger@example.com", "line")
    with pytest.raises(NotAMember):
        links.create_code("org_zzz", "mew@example.com", "line")
    with pytest.raises(CodeError):
        links.create_code("org_a", "mew@example.com", "discord")


def test_old_codes_are_pruned(links, clock, docs):
    old = links.create_code("org_a", "mew@example.com", "line").code
    clock.advance(days=2)
    links.create_code("org_a", "ann@example.com", "line")
    assert docs.get(KIND, 0, digits(old)) is None


# ------------------------------------------------------------------ attach codes
def test_attach_code_attaches_the_bot_tenant(links, orgs):
    code = links.create_attach_code("org_a", "boss@example.com", "telegram").code
    done = links.redeem(code, "telegram", "42", "8629")
    assert done.kind == "attach" and done.org_id == "org_a"
    org, ws = orgs.org_of_workspace("telegram", "8629")
    assert org.id == "org_a" and ws.id == "telegram-8629" and ws.connected_by == "boss@example.com"
    assert ws.memory_id == orgs.get("org_a").workspace("telegram-8629").memory_id
    # an attach code links NO account
    assert orgs.member_by_identity("org_a", "telegram", "42") is None


def test_attach_refuses_a_tenant_of_another_org(links, orgs):
    attach(orgs, "org_b", "whatsapp", "1555")
    code = links.create_attach_code("org_a", "boss@example.com", "whatsapp").code
    with pytest.raises(CodeError) as e:
        links.redeem(code, "whatsapp", "66800000000", "1555")
    assert e.value.reason == "tenant belongs to another org"
    assert orgs.org_of_workspace("whatsapp", "1555")[0].id == "org_b"


def test_attach_is_idempotent_for_the_same_org_and_single_use(links, orgs):
    attach(orgs, "org_a", "line", "Ubot")
    code = links.create_attach_code("org_a", "boss@example.com", "line").code
    assert links.redeem(code, "line", "U1", "Ubot").kind == "attach"
    with pytest.raises(CodeError):
        links.redeem(code, "line", "U1", "Ubot")


def test_attach_and_identity_codes_are_counted_separately(links):
    for _ in range(MAX_LIVE_CODES):
        links.create_attach_code("org_a", "boss@example.com", "line")
    with pytest.raises(TooManyCodes):
        links.create_attach_code("org_a", "boss@example.com", "line")
    links.create_code("org_a", "boss@example.com", "line")


def test_attach_code_for_discord_or_unknown_org_is_refused(links):
    with pytest.raises(CodeError):
        links.create_attach_code("org_a", "boss@example.com", "discord")
    with pytest.raises(CodeError):
        links.create_attach_code("org_zzz", "boss@example.com", "line")


# ------------------------------------------------------------------ who is this account
def test_identity_of_needs_a_verified_member_of_the_tenants_org(links, orgs):
    attach(orgs, "org_a", "line", "Ubot")
    orgs.link_identity("org_a", "mew@example.com", "line", "U1")
    assert links.identity_of("line", "Ubot", "U1") == ("org_a", "mew@example.com")
    assert links.identity_of("line", "Ubot", "U-guest") is None
    assert links.identity_of("line", "Uother", "U1") is None  # tenant of no org
    attach(orgs, "org_b", "line", "Ubot2")
    assert links.identity_of("line", "Ubot2", "U1") is None  # linked in org A, tenant is org B's


def test_auto_link_by_email(links, orgs):
    attach(orgs, "org_a", "slack", "T1")
    assert links.auto_link("slack", "T1", "U1", " Mew@Example.com ") == ("org_a", "mew@example.com")
    assert orgs.member_by_identity("org_a", "slack", "U1").email == "mew@example.com"
    assert links.auto_link("slack", "T1", "U1", "mew@example.com")  # idempotent


def test_auto_link_refusals(links, orgs):
    attach(orgs, "org_a", "slack", "T1")
    assert links.auto_link("slack", "T1", "U1", "stranger@example.com") is None  # not a member
    assert links.auto_link("slack", "T1", "U1", "") is None
    assert links.auto_link("slack", "T-none", "U1", "mew@example.com") is None  # no org tenant
    orgs.link_identity("org_a", "ann@example.com", "slack", "U9")
    assert links.auto_link("slack", "T1", "U9", "mew@example.com") is None  # Ann's account
    orgs.link_identity("org_a", "mew@example.com", "slack", "U-old")
    assert links.auto_link("slack", "T1", "U2", "mew@example.com") is None  # keeps the old link
    assert orgs.member_by_identity("org_a", "slack", "U-old").email == "mew@example.com"
