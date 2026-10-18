# ruff: noqa: F811, F401
"""Change outbox: the diff, the targets, the resend / fail rules, the Discord consumer (apply,
notice card, button, reply, reminder, alert), the watcher safety net, the web hook and route.
Offline: fake Discord objects, SQLite / fake Firestore, no network."""

import asyncio
import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import GUILD
from fake_firestore import FakeFirestore
from test_center_api import H, w  # noqa: F401  (fixtures)
from test_coordinator import BOSS, OWNER_ID, PEER, world
from test_modes import bot  # noqa: F401  (fixture)
from test_positions_bot import give, link, names, rig, wire

from meeting_scribe import center
from salid import change_outbox as co
from salid.change_outbox import (
    ChangeOutbox,
    change_text,
    diff_members,
    diff_org,
    diff_positions,
    is_ack_text,
    migrate_legacy,
    record_org_edit,
)
from salid.change_outbox_bot import ChangeButton
from salid.coo_store import FirestoreDocStore, SQLiteDocStore
from salid.i18n import t
from salid.orgs import Member, Org, OrgStore, Position, Workspace

T0 = datetime(2026, 10, 18, 9, 0, tzinfo=UTC)


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t

    def advance(self, minutes):
        self.t += timedelta(minutes=minutes)


@pytest.fixture(params=["sqlite", "firestore"])
def docs(request):
    if request.param == "sqlite":
        return SQLiteDocStore(":memory:")
    return FirestoreDocStore(client=FakeFirestore())


def org_with(*members, workspaces=()):
    return Org(id="o1", name="Cafe", members=list(members), workspaces=list(workspaces))


def ws(platform, key):
    return Workspace(id=f"{platform}-{key}", platform=platform, external_id=key)


# ------------------------------------------------------------------ diff (pure)
def test_diff_one_change_per_field_per_member():
    a = Member(email="a@x.com", role="member", title="Staff", level=6, department="tech")
    b = a.model_copy(update={"role": "admin", "title": "CTO", "level": 2, "department": "business"})
    c = Member(email="c@x.com")
    out = diff_members(org_with(a, c), org_with(b, c))
    assert sorted((x.kind, x.before, x.after) for x in out) == [
        ("department", "tech", "business"),
        ("level", "6", "2"),
        ("role", "member", "admin"),
        ("title", "Staff", "CTO"),
    ]
    assert {x.subject_email for x in out} == {"a@x.com"}


def test_diff_added_removed_link_and_nothing():
    a, b = Member(email="a@x.com"), Member(email="b@x.com", role="admin")
    assert diff_members(org_with(a), org_with(a)) == []
    added = diff_members(org_with(a), org_with(a, b))
    assert [(x.kind, x.after, x.subject_email) for x in added] == [
        ("member_added", "admin", "b@x.com")
    ]
    removed = diff_members(org_with(a, b), org_with(a))
    assert [(x.kind, x.before) for x in removed] == [("member_removed", "admin")]
    linked = a.model_copy(update={"identities": {"slack": "U1"}})
    out = diff_members(org_with(a), org_with(linked))
    assert [(x.kind, x.after) for x in out] == [("link", "slack:U1")]


def test_email_case_does_not_make_a_change():
    a = Member(email="A@x.com")
    assert diff_members(org_with(a), org_with(Member(email="a@X.com"))) == []


# ------------------------------------------------------------------ targets
def edit_role(org, email, role):
    new = org.model_copy(deep=True)
    for m in new.members:
        if m.email == email:
            m.role = role
    return new


def test_record_one_apply_per_workspace_and_a_notice_per_linked_platform(docs):
    ob = ChangeOutbox(docs, Clock())
    m = Member(
        email="a@x.com",
        discord_user_id="77",
        identities={"slack": "U9", "line": "L1"},
        identity_tenants={"line": "Ubot"},
    )
    base = org_with(
        m,
        workspaces=[
            ws("discord", "5"),
            ws("slack", "T1"),
            ws("line", "Uother"),
            ws("telegram", "9"),
        ],
    )
    stored = ob.record(diff_members(base, edit_role(base, "a@x.com", "admin")), base)
    assert len(stored) == 1
    got = {(x.workspace_id, x.kind, x.person) for x in stored[0].targets}
    assert got == {
        ("discord-5", "apply", ""),
        ("slack-T1", "apply", ""),
        ("line-Uother", "apply", ""),
        ("telegram-9", "apply", ""),
        ("discord-5", "notify", "77"),
        ("slack-T1", "notify", "U9"),
        # LINE id was verified in another tenant: no notice there; telegram: not linked
    }


def test_record_skips_web_workspaces_and_an_unlinked_person_gets_no_notice(docs):
    ob = ChangeOutbox(docs, Clock())
    base = org_with(Member(email="a@x.com"), workspaces=[ws("web", "o1"), ws("slack", "T1")])
    (c,) = ob.record(diff_members(base, edit_role(base, "a@x.com", "admin")), base)
    assert [(x.workspace_id, x.kind) for x in c.targets] == [("slack-T1", "apply")]


def test_a_removed_member_still_gets_the_notice(docs):
    ob = ChangeOutbox(docs, Clock())
    m = Member(email="a@x.com", discord_user_id="77")
    base = org_with(m, Member(email="b@x.com"), workspaces=[ws("discord", "5")])
    after = org_with(Member(email="b@x.com"), workspaces=[ws("discord", "5")])
    (c,) = ob.record(diff_members(base, after), after)
    assert ("discord-5", "notify", "77") in {(x.workspace_id, x.kind, x.person) for x in c.targets}


def test_the_same_edit_is_recorded_once_within_ten_minutes(docs):
    clock = Clock()
    ob = ChangeOutbox(docs, clock)
    base = org_with(Member(email="a@x.com"), workspaces=[ws("slack", "T1")])
    after = edit_role(base, "a@x.com", "admin")
    assert len(ob.record(diff_members(base, after, by="boss@x.com"), after)) == 1
    assert ob.record(diff_members(base, after, by="detected"), after) == []
    clock.advance(11)
    assert len(ob.record(diff_members(base, after, by="detected"), after)) == 1


# ------------------------------------------------------------------ due / ack / fail
def one_change(ob, workspaces=("slack", "T1"), linked=True, org_id="o1"):
    m = Member(email="a@x.com", identities={"slack": "U9"} if linked else {})
    base = org_with(m, workspaces=[ws(*workspaces)]).model_copy(update={"id": org_id})
    (c,) = ob.record(diff_members(base, edit_role(base, "a@x.com", "admin")), base)
    return c


def link_org(docs, *workspaces, org_id="o1"):
    """The org in the central store with these chat workspaces attached, so a workspace id
    resolves to its org."""
    store = OrgStore(docs)
    store.create("Cafe", "boss@x.com", org_id=org_id)
    for platform, key in workspaces:
        store.attach_workspace(org_id, ws(platform, key))
    return store


def test_due_pending_then_resent_every_fifteen_minutes_until_acked(docs):
    clock = Clock()
    ob = ChangeOutbox(docs, clock)
    c = one_change(ob)
    assert {(x.kind) for _, x in ob.due(["o1"])} == {"apply", "notify"}
    ob.mark_attempt(c.id, "slack-T1", "U9")
    clock.advance(14)
    assert [x.kind for _, x in ob.due(["o1"])] == ["apply"]  # the notice waits
    clock.advance(1)
    assert {x.kind for _, x in ob.due(["o1"])} == {"apply", "notify"}
    assert ob.ack(c.id, "slack-T1", "U9", "reply") is True
    assert [x.kind for _, x in ob.due(["o1"])] == ["apply"]
    assert ob.ack(c.id, "slack-T1", "", "applied") is True
    assert ob.due(["o1"]) == []


def test_ack_is_idempotent_and_ignores_strangers(docs):
    ob = ChangeOutbox(docs, Clock())
    c = one_change(ob)
    assert ob.ack(c.id, "slack-T1", "U9", "button") is True
    assert ob.ack(c.id, "slack-T1", "U9", "button") is False
    assert ob.ack(c.id, "slack-T1", "U-other", "button") is False
    assert ob.ack("nope", "slack-T1", "U9", "button") is False
    t9 = next(x for x in ob.get(c.id).targets if x.person == "U9")
    assert (t9.status, t9.acked_how) == ("acked", "button")


def test_a_target_fails_after_the_last_attempt_and_the_admins_are_alerted_once(docs):
    clock = Clock()
    ob = ChangeOutbox(docs, clock, max_attempts=3)
    c = one_change(ob)
    for _ in range(3):
        ob.mark_attempt(c.id, "slack-T1", "U9")
        clock.advance(15)
    assert [x.kind for _, x in ob.due(["o1"])] == ["apply"]  # notice ran out: failed, not due
    failed = ob.get(c.id)
    assert next(x for x in failed.targets if x.kind == "notify").status == "failed"
    assert [x.id for x in ob.to_alert("slack", ["o1"])] == [c.id]
    assert ob.mark_alerted(c.id, "slack") is True
    assert ob.to_alert("slack", ["o1"]) == [] and ob.mark_alerted(c.id, "slack") is False
    assert [x.id for x in ob.to_alert("line", ["o1"])] == [c.id]  # each platform's admins once


def test_a_workspace_nobody_serves_fails_at_the_deadline(docs):
    clock = Clock()
    ob = ChangeOutbox(docs, clock, max_attempts=2)
    c = one_change(ob)  # nobody ever sends anything
    clock.advance(15 * 3 - 1)
    assert len(ob.due(["o1"])) == 2
    clock.advance(1)
    assert ob.due(["o1"]) == []
    assert {x.status for x in ob.get(c.id).targets} == {"failed"}
    assert [x.id for x in ob.to_alert("slack", ["o1"])] == [c.id]


def test_a_late_reply_still_counts_after_failure(docs):
    clock = Clock()
    ob = ChangeOutbox(docs, clock, max_attempts=1)
    link_org(docs, ("slack", "T1"))
    c = one_change(ob)
    ob.mark_attempt(c.id, "slack-T1", "U9", message_ref="ts1")
    clock.advance(15)
    ob.due(["o1"])
    assert ob.open_for("slack-T1", "U9")[0][1].status == "failed"
    assert ob.ack(c.id, "slack-T1", "U9", "reply") is True


def test_errors_are_kept_and_the_try_is_counted(docs):
    clock = Clock()
    ob = ChangeOutbox(docs, clock)
    c = one_change(ob)
    ob.mark_attempt(c.id, "slack-T1", "", error="missing Manage Roles")
    x = next(x for x in ob.get(c.id).targets if x.kind == "apply")
    assert (x.status, x.attempts, x.last_error) == ("pending", 1, "missing Manage Roles")
    assert [k.kind for _, k in ob.due(["o1"])] == ["notify"]  # the apply waits 15 minutes too


def test_acked_changes_go_after_ninety_days_failed_ones_never(docs):
    clock = Clock()
    ob = ChangeOutbox(docs, clock, max_attempts=1)
    done = one_change(ob)
    ob.ack(done.id, "slack-T1", "U9", "reply")
    ob.ack(done.id, "slack-T1", "", "applied")
    clock.advance(10)
    lost = ob.record(
        diff_members(
            org_with(Member(email="z@x.com"), workspaces=[ws("slack", "T1")]),
            org_with(Member(email="z@x.com", role="admin"), workspaces=[ws("slack", "T1")]),
        ),
        org_with(Member(email="z@x.com", role="admin"), workspaces=[ws("slack", "T1")]),
    )[0]
    clock.advance(30)
    ob.due(["o1"])
    assert ob.get(lost.id).targets[0].status == "failed"
    clock.advance(60 * 24 * 91)
    assert ob.prune(["o1"], force=True) == 1
    assert ob.get(done.id) is None and ob.get(lost.id) is not None


# ------------------------------------------------------------------ one key per org + migration
def test_changes_are_stored_per_org_and_due_reads_only_those_orgs():
    docs = FirestoreDocStore(client=FakeFirestore())
    ob = ChangeOutbox(docs, Clock())
    c = one_change(ob)
    assert docs.get(co.KIND, co.org_guild("o1"), c.id) and docs.get(co.KIND, 0, c.id) is None
    docs.client.streams.clear()
    ob.due(["o1"])
    assert {s for s in docs.client.streams if s[0] == co.KIND} == {
        (co.KIND, (("guild_id", "==", co.org_guild("o1")),))
    }


def test_an_org_never_reads_another_orgs_changes(docs):
    clock = Clock()
    ob = ChangeOutbox(docs, clock, max_attempts=1)
    mine, theirs = one_change(ob), one_change(ob, org_id="o2")
    assert co.org_guild("o1") != co.org_guild("o2")
    assert [c.id for c in ob.list_changes("o1")] == [mine.id]
    assert {c.id for c, _ in ob.due(["o1"])} == {mine.id}
    assert {c.id for c, _ in ob.due(["o1", "o2"])} == {mine.id, theirs.id}
    assert ob.due([]) == []
    clock.advance(15 * 3)
    ob.due(["o1"])  # fails o1's targets only
    assert [c.id for c in ob.to_alert("slack", ["o1", "o2"])] == [mine.id]
    clock.advance(60 * 24 * 91)
    ob.ack(theirs.id, "slack-T1", "U9", "reply", org_id="o2")
    ob.ack(theirs.id, "slack-T1", "", "applied", org_id="o2")
    assert ob.prune(["o1"], force=True) == 0  # o2's acked change is not o1's to prune
    assert ob.prune(["o2"], force=True) == 1 and ob.get(theirs.id) is None


def test_get_without_org_uses_the_change_id_alone(docs):
    ob = ChangeOutbox(docs, Clock())
    c = one_change(ob)
    assert ob.get(c.id).id == c.id  # a button click knows only the change id
    assert ob.get(c.id, "o1").id == c.id
    assert ob.get(c.id, "o2") is None and ob.get("0" * 16) is None
    assert ob.ack(c.id, "slack-T1", "", "applied", org_id="o2") is False  # not that org's
    assert ob.ack(c.id, "slack-T1", "", "applied") is True  # the org is found from the id
    assert ob.mark_attempt(c.id, "slack-T1", "U9", org_id="o1") is True
    assert ob.mark_alerted(c.id, "slack") is True


def to_legacy(docs, change):
    """Put a change back where it lived before the per-org key (not migrated yet)."""
    docs.put(
        co.KIND,
        co.LEGACY_GUILD,
        change.id,
        docs.get(co.KIND, co.org_guild(change.org_id), change.id),
    )
    docs.delete(co.KIND, co.org_guild(change.org_id), change.id)


def test_a_change_not_migrated_yet_can_still_be_acked(docs):
    ob = ChangeOutbox(docs, Clock())
    c = one_change(ob)
    to_legacy(docs, c)
    assert ob.ack(c.id, "slack-T1", "U9", "reply", org_id="o2") is False  # another org's
    assert ob.ack(c.id, "slack-T1", "U9", "reply", org_id="o1") is True
    found = ob.get(c.id)  # the click path: by the change id alone
    assert found is not None and found.org_id == "o1"
    assert ob.ack(c.id, "slack-T1", "", "applied") is True  # no org: found from the id
    legacy = co.Change(**docs.get(co.KIND, co.LEGACY_GUILD, c.id))
    assert {(x.kind, x.status) for x in legacy.targets} == {
        ("notify", "acked"),
        ("apply", "acked"),
    }
    assert docs.get(co.KIND, co.org_guild("o1"), c.id) is None  # nothing written under the org
    assert migrate_legacy(docs, apply=True)["moved"] == 1
    assert {x.status for x in ob.get(c.id, "o1").targets} == {"acked"}


def test_open_for_resolves_the_org_of_the_workspace(docs):
    ob = ChangeOutbox(docs, Clock())
    c = one_change(ob)
    ob.mark_attempt(c.id, "slack-T1", "U9", org_id="o1")
    assert ob.open_for("slack-T1", "U9") == []  # the workspace is linked to no org yet
    link_org(docs, ("slack", "T1"))
    assert [(x.id, y.person) for x, y in ob.open_for("slack-T1", "U9")] == [(c.id, "U9")]
    assert ob.open_for("slack-T2", "U9") == []  # an unlinked workspace: nothing
    assert ob.open_for("web", "U9") == []


def legacy(docs, change_id, org_id):
    c = co.Change(id=change_id, org_id=org_id, at=T0.isoformat(), kind="role",
                  subject_email="a@x.com", after="admin")  # fmt: skip
    docs.put(co.KIND, co.LEGACY_GUILD, change_id, c.model_dump())


def test_migrate_legacy_moves_once_and_skips_orphans(docs):
    legacy(docs, "a" * 16, "o1")
    legacy(docs, "b" * 16, "o2")
    docs.put(co.KIND, co.LEGACY_GUILD, "c" * 16, {"id": "c" * 16, "kind": "role"})  # no org_id
    assert migrate_legacy(docs, apply=False) == {"found": 3, "movable": 2, "moved": 0, "skipped": 1}
    assert ChangeOutbox(docs).list_changes("o2") == []  # a dry run moves nothing
    assert migrate_legacy(docs, apply=True)["moved"] == 2
    assert migrate_legacy(docs, apply=True) == {"found": 1, "movable": 0, "moved": 0, "skipped": 1}
    assert [c.id for c in ChangeOutbox(docs).list_changes("o2")] == ["b" * 16]
    assert docs.get(co.KIND, co.org_guild("o1"), "a" * 16)["after"] == "admin"
    assert docs.get(co.KIND, co.LEGACY_GUILD, "a" * 16) is None
    assert docs.get(co.KIND, co.LEGACY_GUILD, "c" * 16) is not None  # the orphan stays


def _migrate_script():
    path = Path(__file__).parents[2] / "scripts" / "salid_outbox_migrate.py"
    spec = importlib.util.spec_from_file_location("salid_outbox_migrate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_migration_script_is_a_dry_run_by_default_and_prints_counts_only(docs, capsys):
    legacy(docs, "a" * 16, "o1")
    mod = _migrate_script()
    assert mod.main([], docs=docs) == 0
    out = capsys.readouterr().out
    assert out == "DRY RUN: found=1, movable=1, moved=0, skipped=0\n"
    assert docs.get(co.KIND, co.LEGACY_GUILD, "a" * 16) is not None
    assert mod.main(["--apply"], docs=docs) == 0
    out = capsys.readouterr().out
    assert out == "APPLIED: found=1, movable=1, moved=1, skipped=0\n"
    assert "a" * 16 not in out and "o1" not in out and "@" not in out
    assert docs.get(co.KIND, co.LEGACY_GUILD, "a" * 16) is None


def test_ack_words():
    for ok in ("รับทราบ", "รับทราบค่ะ", "ทราบแล้วครับ", "OK", "ok.", "Got it", "got it!", "👍", "noted"):
        assert is_ack_text(ok), ok
    for no in ("", "ok but why was I removed", "ไม่รับทราบ", "what changed?", "รับทราบแล้ว ขอถามหน่อย"):
        assert not is_ack_text(no), no


def test_record_org_edit_never_raises(docs, monkeypatch):
    base = org_with(Member(email="a@x.com"), workspaces=[ws("slack", "T1")])
    after = edit_role(base, "a@x.com", "admin")
    assert record_org_edit(docs, base, after, "boss@x.com") == 1
    monkeypatch.setattr(ChangeOutbox, "record", lambda *a, **k: 1 / 0)
    assert record_org_edit(docs, base, edit_role(base, "a@x.com", "owner"), "x") == 0


def test_notice_texts_in_both_languages():
    c = co.Change(id="1", org_id="o", at="", by="boss@x.com", kind="title", subject_email="a@x.com",
                  before="Admin", after="Founder")  # fmt: skip
    assert co.change_text(c, "Cafe", "en") == (
        "Your company position in Cafe changed: Admin → Founder (by boss@x.com)."
    )
    assert "ตำแหน่งในบริษัทของคุณใน Cafe" in co.change_text(c, "Cafe", "th")
    assert co.change_text(c, "Cafe", "en", reminder=2).endswith("(reminder 2)")
    c.by = "detected"
    assert "the central system" in co.change_text(c, "Cafe", "en")


# ------------------------------------------------------------------ Discord consumer
def capture(m):
    """The member's DM returns a message with an id (so a reply can be matched to it)."""
    orig = m.send

    async def send(text=None, **kw):
        await orig(text, **kw)
        return SimpleNamespace(id=9000 + len(m.dms))

    m.send = send


def web_edit(bot, email, **fields):
    """An edit the way the web route makes it, recorded through the hook. Returns (before, after)."""
    store = OrgStore(bot.docs)
    before = store.get("o1")
    store.update_members(
        "o1",
        lambda ms: (
            [setattr(m, k, v) for m in ms if m.email == email for k, v in fields.items()] or True
        ),
    )
    after = store.get("o1")
    bot.change_outbox.record(diff_members(before, after, by="boss@x.com"), after)
    return before, after


@pytest.fixture
def drig(rig):
    clock = Clock()
    rig.bot.change_outbox.now = clock
    rig.clock = clock
    capture(rig.g.members[PEER])
    link(rig.bot, ("peer@x.com", PEER))
    rig.lang = rig.bot._change_lang(GUILD)
    return rig


async def tick(rig, advance=0):
    rig.clock.advance(advance)
    rig.bot._change_polled = float("-inf")
    await rig.bot._change_outbox_due()


def targets(rig, kind=None):
    out = []
    for c in rig.bot.change_outbox.list_changes("o1"):
        out += [x for x in c.targets if kind in (None, x.kind)]
    return out


async def test_apply_runs_the_sync_and_is_acked_as_applied(drig):
    web_edit(drig.bot, "peer@x.com", title="CTO")
    assert "CTO" not in names(drig.g.members[PEER])
    await tick(drig)
    assert "CTO" in names(drig.g.members[PEER])  # positions_sync ran
    (apply,) = targets(drig, "apply")
    assert (apply.status, apply.acked_how) == ("acked", "applied")


async def test_apply_without_manage_roles_is_retried_not_acked(drig):
    wire(drig.w, manage_roles=False)
    give(drig.w, BOSS)
    web_edit(drig.bot, "peer@x.com", title="CTO")
    await tick(drig)
    (apply,) = targets(drig, "apply")
    assert (apply.status, apply.attempts, apply.last_error) == (
        "pending",
        1,
        "missing Manage Roles",
    )


async def test_apply_is_nothing_to_apply_when_role_sync_is_off(drig, monkeypatch):
    monkeypatch.setattr(type(drig.bot), "_positions_on", lambda self, gid: False)
    web_edit(drig.bot, "peer@x.com", title="CTO")
    await tick(drig)
    (apply,) = targets(drig, "apply")
    assert apply.acked_how == "nothing_to_apply"


async def test_notify_sends_a_card_with_a_persistent_button(drig):
    web_edit(drig.bot, "peer@x.com", title="CTO")
    await tick(drig)
    (text, kw) = drig.g.members[PEER].dms[-1]
    assert t("chg_kind_title", drig.lang) in text and "Cafe" in text and "→ CTO" in text
    assert "boss@x.com" in text
    (button,) = kw["view"].children
    (c,) = drig.bot.change_outbox.list_changes("o1")
    assert button.custom_id == f"chg:{c.id}:discord-{GUILD}"
    assert button.item.label == t("chg_btn", drig.lang)
    note = next(x for x in c.targets if x.kind == "notify")
    assert (note.status, note.attempts, note.message_ref) == ("sent", 1, "9001")
    n = len(drig.g.members[PEER].dms)
    await tick(drig, 5)  # not yet: the wait is 15 minutes
    assert len(drig.g.members[PEER].dms) == n


async def test_the_button_survives_a_restart_and_a_click_acks(drig):
    web_edit(drig.bot, "peer@x.com", title="CTO")
    await tick(drig)
    (c,) = drig.bot.change_outbox.list_changes("o1")
    cid = f"chg:{c.id}:discord-{GUILD}"
    match = ChangeButton.__discord_ui_compiled_template__.fullmatch(cid)
    item = await ChangeButton.from_custom_id(None, SimpleNamespace(label="Got it"), match)
    sent = []

    async def edit_message(**kw):
        sent.append(kw)

    inter = SimpleNamespace(
        user=SimpleNamespace(id=PEER),
        client=drig.bot,
        message=SimpleNamespace(content="the card"),
        response=SimpleNamespace(edit_message=edit_message, send_message=None),
    )
    await item.callback(inter)
    assert sent[0]["view"] is None and t("chg_done", drig.lang) in sent[0]["content"]
    note = next(x for x in drig.bot.change_outbox.get(c.id).targets if x.kind == "notify")
    assert (note.status, note.acked_how) == ("acked", "button")
    before = len(drig.g.members[PEER].dms)
    await tick(drig, 20)
    assert len(drig.g.members[PEER].dms) == before  # confirmed: never sent again


async def test_somebody_elses_click_does_nothing(drig):
    web_edit(drig.bot, "peer@x.com", title="CTO")
    await tick(drig)
    (c,) = drig.bot.change_outbox.list_changes("o1")
    said = []

    async def send_message(text, ephemeral=False):
        said.append(text)

    inter = SimpleNamespace(
        user=SimpleNamespace(id=BOSS), response=SimpleNamespace(send_message=send_message)
    )
    await drig.bot.handle_change_click(inter, c.id, f"discord-{GUILD}")
    assert said == [t("chg_not_yours", drig.lang)]
    assert (
        next(x for x in drig.bot.change_outbox.get(c.id).targets if x.kind == "notify").status
        == "sent"
    )


def reply_msg(uid, text, ref=None):
    out = []

    async def reply(text, mention_author=False):
        out.append(text)

    m = SimpleNamespace(
        content=text,
        author=SimpleNamespace(id=uid),
        reference=SimpleNamespace(message_id=ref) if ref else None,
        reply=reply,
    )
    m.said = out
    return m


async def test_a_reply_in_words_acks(drig):
    web_edit(drig.bot, "peer@x.com", title="CTO")
    await tick(drig)
    (c,) = drig.bot.change_outbox.list_changes("o1")
    msg = reply_msg(PEER, "รับทราบค่ะ", ref=9001)
    assert await drig.bot._change_ack_reply(msg, drig.g) is True
    assert msg.said == [t("chg_thanks", drig.lang)]
    note = next(x for x in drig.bot.change_outbox.get(c.id).targets if x.kind == "notify")
    assert (note.status, note.acked_how) == ("acked", "reply")


async def test_ok_without_a_notice_goes_on_as_a_normal_message(drig):
    assert await drig.bot._change_ack_reply(reply_msg(PEER, "ok"), drig.g) is False
    web_edit(drig.bot, "peer@x.com", title="CTO")
    await tick(drig)
    assert await drig.bot._change_ack_reply(reply_msg(PEER, "what is a CTO"), drig.g) is False
    assert await drig.bot._change_ack_reply(reply_msg(BOSS, "ok"), drig.g) is False  # not theirs


async def test_a_reply_to_one_notice_acks_only_that_one(drig):
    web_edit(drig.bot, "peer@x.com", title="CTO")
    await tick(drig)
    web_edit(drig.bot, "peer@x.com", department="finance")
    await tick(drig)
    notes = sorted((c for c in drig.bot.change_outbox.list_changes("o1")), key=lambda c: c.kind)
    first = next(x for c in notes if c.kind == "title" for x in c.targets if x.kind == "notify")
    assert await drig.bot._change_ack_reply(
        reply_msg(PEER, "ok", ref=int(first.message_ref)), drig.g
    )
    now = drig.bot.change_outbox.list_changes("o1")
    states = {c.kind: next(x for x in c.targets if x.kind == "notify").status for c in now}
    assert states == {"title": "acked", "department": "sent"}


async def test_the_resend_is_the_same_card_with_a_reminder_count(drig):
    web_edit(drig.bot, "peer@x.com", title="CTO")
    await tick(drig)
    await tick(drig, 15)
    await tick(drig, 15)
    dms = [text for text, _ in drig.g.members[PEER].dms]
    assert t("chg_reminder", drig.lang, n=1) not in dms[0]
    assert dms[1].endswith(t("chg_reminder", drig.lang, n=1))
    assert dms[2].endswith(t("chg_reminder", drig.lang, n=2))
    assert dms[1].startswith(dms[0])


async def test_a_notice_nobody_confirms_fails_and_the_owner_is_alerted_once(drig):
    drig.bot.change_outbox.max_attempts = 2
    web_edit(drig.bot, "peer@x.com", title="CTO")
    owner = drig.g.members[OWNER_ID]
    await tick(drig)
    await tick(drig, 15)
    await tick(drig, 15)  # the second send was the last; now it gives up
    await tick(drig, 15)
    (c,) = drig.bot.change_outbox.list_changes("o1")
    assert next(x for x in c.targets if x.kind == "notify").status == "failed"
    alerts = [text for text, _ in owner.dms if "peer@x.com" in text]
    assert len(alerts) == 1 and "⚠️" in alerts[0]
    await tick(drig, 30)
    assert len([text for text, _ in owner.dms if "peer@x.com" in text]) == 1
    assert c.id in {
        x.id for x in drig.bot.change_outbox.list_changes("o1") if "discord" in x.alerted_on
    }


async def test_a_person_who_left_the_server_is_retried_not_dropped(drig):
    web_edit(drig.bot, "peer@x.com", title="CTO")
    del drig.g.members[PEER]
    drig.g.get_member = lambda uid: drig.g.members.get(uid)
    await tick(drig)
    note = next(x for x in targets(drig, "notify"))
    assert (note.status, note.attempts, note.last_error) == ("pending", 1, "not in the server")


async def test_the_bot_reads_only_the_changes_of_its_servers_orgs(drig):
    assert await drig.bot._change_org_ids() == {"o1"}
    m = Member(email="peer@x.com", discord_user_id=str(PEER))
    other = Org(id="o9", name="Other", members=[m], workspaces=[ws("discord", str(GUILD))])
    (theirs,) = drig.bot.change_outbox.record(
        diff_members(other, edit_role(other, "peer@x.com", "admin")), other
    )
    await tick(drig)
    # o9 owns none of this bot's servers: its change is never read, so nothing was tried
    got = drig.bot.change_outbox.get(theirs.id, "o9")
    assert len(got.targets) == 2
    assert {(x.status, x.attempts) for x in got.targets} == {("pending", 0)}
    assert drig.g.members[PEER].dms == []
    web_edit(drig.bot, "peer@x.com", title="CTO")
    await tick(drig)
    assert next(x for x in targets(drig, "notify")).status == "sent"


async def test_a_server_whose_org_lookup_fails_is_skipped_and_the_others_are_served(
    drig, monkeypatch, caplog
):
    bad = SimpleNamespace(id=4242, get_member=lambda uid: None)
    monkeypatch.setattr(type(drig.bot), "guilds", property(lambda s: [bad, drig.g]))
    real = OrgStore.org_of_guild

    def lookup(self, guild_id):
        if guild_id == 4242:
            raise RuntimeError("broken org doc for boss@x.com")
        return real(self, guild_id)

    monkeypatch.setattr(OrgStore, "org_of_guild", lookup)
    assert await drig.bot._change_org_ids() == {"o1"}
    web_edit(drig.bot, "peer@x.com", title="CTO")
    await tick(drig)
    assert next(x for x in targets(drig, "notify")).status == "sent"  # o1 still delivered
    assert any(
        "4242" in r.getMessage() and "RuntimeError" in r.getMessage() for r in caplog.records
    )
    assert "boss@x.com" not in caplog.text  # only the error type and the guild id are logged


async def test_a_click_on_a_notice_sent_before_the_re_key_still_acks(drig):
    web_edit(drig.bot, "peer@x.com", title="CTO")
    await tick(drig)
    (c,) = drig.bot.change_outbox.list_changes("o1")
    to_legacy(drig.bot.docs, c)
    said = []

    async def edit_message(**kw):
        said.append(kw)

    inter = SimpleNamespace(
        user=SimpleNamespace(id=PEER),
        message=SimpleNamespace(content="the card"),
        response=SimpleNamespace(edit_message=edit_message, send_message=None),
    )
    await drig.bot.handle_change_click(inter, c.id, f"discord-{GUILD}")
    assert t("chg_done", drig.lang) in said[0]["content"]
    note = next(x for x in drig.bot.change_outbox.get(c.id).targets if x.kind == "notify")
    assert (note.status, note.acked_how) == ("acked", "button")
    assert drig.bot.docs.get(co.KIND, co.org_guild("o1"), c.id) is None  # updated where it is


# ------------------------------------------------------------------ the safety net (watcher)
async def test_the_watcher_records_an_edit_the_web_did_not(drig):
    drig.bot._positions_polled = -1e9
    await drig.bot._positions_center_watch()  # first look: snapshot only
    assert drig.bot.change_outbox.list_changes("o1") == []
    store = OrgStore(drig.bot.docs)
    store.update_members("o1", lambda ms: setattr(ms[1], "title", "CTO") or True)
    drig.bot._positions_polled = -1e9
    await drig.bot._positions_center_watch()
    (c,) = drig.bot.change_outbox.list_changes("o1")
    assert (c.kind, c.before, c.after, c.by) == ("title", "", "CTO", "detected")
    assert any(x.kind == "notify" and x.person == str(PEER) for x in c.targets)


async def test_an_edit_the_web_already_recorded_is_not_recorded_twice(drig):
    drig.bot._positions_polled = -1e9
    await drig.bot._positions_center_watch()
    web_edit(drig.bot, "peer@x.com", title="CTO")
    drig.bot._positions_polled = -1e9
    await drig.bot._positions_center_watch()
    (c,) = drig.bot.change_outbox.list_changes("o1")
    assert c.by == "boss@x.com"


async def test_salids_own_mirror_writes_are_not_center_edits(drig):
    drig.bot.positions.assign(GUILD, PEER, "cto", "stated")
    drig.bot._positions_polled = -1e9
    await drig.bot._positions_center_watch()
    await drig.bot.positions_sync(drig.g)  # mirrors "CTO" into the org member
    drig.bot._positions_polled = -1e9
    await drig.bot._positions_center_watch()
    assert drig.bot.change_outbox.list_changes("o1") == []


# ------------------------------------------------------------------ agent tool
async def test_change_status_lists_who_has_and_has_not_confirmed(drig):
    web_edit(drig.bot, "peer@x.com", title="CTO")
    await tick(drig)
    assert drig.bot._change_tools(drig.g, is_exec=False) == []
    (tool,) = drig.bot._change_tools(drig.g, is_exec=True)
    out = await asyncio.to_thread(tool)
    (row,) = out["changes"]
    assert row["who"] == "peer@x.com" and row["waiting"] == ["discord"] and row["confirmed"] == []
    assert row["applied_on"] == ["discord"]
    (c,) = drig.bot.change_outbox.list_changes("o1")
    drig.bot.change_outbox.ack(c.id, f"discord-{GUILD}", str(PEER), "button")
    row = (await asyncio.to_thread(tool, name="PEER"))["changes"][0]
    assert row["confirmed"] == ["discord"] and row["waiting"] == []
    assert (await asyncio.to_thread(tool, name="nobody"))["count"] == 0


async def test_the_agent_gets_the_tool_for_executives_in_chat_and_voice(drig):

    tools = drig.bot._change_tools(drig.g, is_exec=True)
    assert [f.__name__ for f in tools] == ["change_status"]


# ------------------------------------------------------------------ web: hook and route
def test_the_web_route_records_a_role_edit(w):
    base = "/api/center/orgs/org_t/members"
    r = w.client.post(base, json={"email": "mem@example.com", "role": "admin"}, headers=H["boss"])
    assert r.status_code == 200
    (c,) = ChangeOutbox(w.docs).list_changes("org_t")
    assert (c.kind, c.subject_email, c.before, c.after, c.by) == (
        "role", "mem@example.com", "member", "admin", "boss@example.com",
    )  # fmt: skip
    assert {x.workspace_id for x in c.targets if x.kind == "apply"} == {
        "discord-1000", "discord-2000",
    }  # fmt: skip


def test_the_web_route_records_add_and_remove(w):
    base = "/api/center/orgs/org_t/members"
    w.client.post(base, json={"email": "new@example.com"}, headers=H["boss"])
    w.client.delete(f"{base}?email=mem@example.com", headers=H["adm"])
    kinds = sorted((c.kind, c.subject_email) for c in ChangeOutbox(w.docs).list_changes("org_t"))
    assert kinds == [("member_added", "new@example.com"), ("member_removed", "mem@example.com")]


def test_a_failing_outbox_never_breaks_the_web_request(w, monkeypatch):
    monkeypatch.setattr(ChangeOutbox, "record", lambda *a, **k: 1 / 0)
    base = "/api/center/orgs/org_t/members"
    r = w.client.post(base, json={"email": "mem@example.com", "role": "admin"}, headers=H["boss"])
    assert r.status_code == 200 and w.orgs.get("org_t").role_of("mem@example.com") == "admin"
    assert w.client.delete(f"{base}?email=adm@example.com", headers=H["boss"]).status_code == 200


def test_the_changes_route_is_for_owners_and_admins_of_that_org(w):
    # ownership is handed over (never granted, owner's rule 18 Oct 2026); the transfer goes
    # through the change outbox like any role edit
    xfer = "/api/center/orgs/org_t/owner/transfer"
    r = w.client.post(xfer, json={"to": "adm@example.com", "me": "admin"}, headers=H["boss"])
    assert r.status_code == 200, r.text
    url = "/api/center/orgs/org_t/changes"
    assert w.client.get(url, headers=H["mem"]).status_code == 403
    assert w.client.get(url, headers=H["stranger"]).status_code == 403
    assert w.client.get(url, headers={}).status_code in (401, 403)
    rows = w.client.get(url + "?limit=5", headers=H["adm"]).json()
    to_owner = [x for x in rows if x["kind"] == "role" and x["after"] == "owner"]
    assert len(to_owner) == 1
    assert to_owner[0]["targets"][0]["status"] in ("pending", "sent", "acked", "failed")
    assert all("subject_ids" not in x for x in rows)
    assert w.client.get(url, headers=H["root"]).status_code == 200  # a super admin
    # another org's changes are never shown through this org
    assert w.client.get("/api/center/orgs/org_o/changes", headers=H["adm"]).status_code == 403
    assert w.client.get(url + "?limit=0", headers=H["adm"]).status_code == 422


# ------------------------------------------------------------------ company positions (holders)
def test_diff_positions_one_company_position_change_per_holder():
    a, b, c = Member(email="a@x.com"), Member(email="b@x.com"), Member(email="c@x.com")
    cook = Position(id="P-1", title="Cook", holders=["a@x.com"])
    before = org_with(a, b, c)
    before.positions = [cook, Position(id="P-2", title="Chef", holders=["c@x.com"])]
    after = org_with(a, b, c)
    after.positions = [
        cook.model_copy(update={"holders": ["b@x.com"]}),
        Position(id="P-2", title="Head Chef", holders=["c@x.com"]),  # renamed: its holder hears it
    ]
    out = diff_positions(before, after)
    assert sorted((x.subject_email, x.kind, x.before, x.after) for x in out) == [
        ("a@x.com", "title", "Cook", ""),
        ("b@x.com", "title", "", "Cook"),
        ("c@x.com", "title", "Chef", "Head Chef"),
    ]
    assert diff_positions(after, after) == []
    text = change_text(out[0], "Cafe", "en")
    assert "company position" in text and "Cook" in text
    assert "ตำแหน่งในบริษัท" in change_text(out[0], "Cafe", "th")


def test_diff_org_does_not_tell_a_person_twice_about_their_title():
    a = Member(email="a@x.com", title="Cook")
    b = a.model_copy(update={"title": "Chef"})
    before, after = org_with(a), org_with(b)
    before.positions = [Position(id="P-1", title="Cook", holders=["a@x.com"])]
    after.positions = [Position(id="P-1", title="Chef", holders=["a@x.com"])]
    assert [x.kind for x in diff_org(before, after)] == ["title"]


def test_record_org_edit_stores_a_holder_change(docs):
    a = Member(email="a@x.com", discord_user_id="5")
    before, after = (
        org_with(a, workspaces=[ws("discord", "9")]),
        org_with(a, workspaces=[ws("discord", "9")]),
    )
    before.positions = [Position(id="P-1", title="Cook")]
    after.positions = [Position(id="P-1", title="Cook", holders=["a@x.com"])]
    assert record_org_edit(docs, before, after, "boss@x.com") == 1
    (c,) = ChangeOutbox(docs).list_changes("o1")
    assert (c.kind, c.after, c.by) == ("title", "Cook", "boss@x.com")
