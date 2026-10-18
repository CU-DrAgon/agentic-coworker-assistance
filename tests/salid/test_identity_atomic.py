"""Redeeming a code, attaching a tenant and linking accounts are atomic ACROSS store instances
(two bot / web processes on one database): no shared Python lock, a real database transaction.

Two independent `SQLiteDocStore` instances on one file stand in for two processes. A store
subclass makes the old read-then-write code lose the race deterministically: both threads wait
at a barrier right after reading the document they are about to change."""

import contextlib
import threading
from datetime import UTC, datetime

import pytest
from fake_firestore import FakeFirestore

from salid.coo_store import FirestoreDocStore, SQLiteDocStore
from salid.identity_links import KIND, CodeError, IdentityLinks
from salid.orgs import ORG_WS, ORGS, IdentityTaken, Member, OrgStore, Workspace, WorkspaceTaken

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)


class RacyStore(SQLiteDocStore):
    """After reading a document of `kind` through `get`, wait for the other thread to have read
    it too (a short timeout lets a correctly serialised caller carry on)."""

    def __init__(self, path, barrier, kind):
        super().__init__(path)
        self._barrier, self._kind = barrier, kind

    def get(self, kind, guild_id, key):
        doc = super().get(kind, guild_id, key)
        if kind == self._kind:
            with contextlib.suppress(threading.BrokenBarrierError):
                self._barrier.wait(timeout=1.0)
        return doc


def in_threads(*calls):
    """Run each callable in its own thread; (result, exception) pairs in call order."""
    out = [None] * len(calls)

    def run(i, fn):
        try:
            out[i] = (fn(), None)
        except Exception as exc:
            out[i] = (None, exc)

    threads = [threading.Thread(target=run, args=(i, c)) for i, c in enumerate(calls)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
        assert not t.is_alive()
    return out


def orgs_on(store):
    return OrgStore(store, clock=lambda: NOW)


def seed(path):
    o = orgs_on(SQLiteDocStore(path))
    o.create(
        "A",
        "boss@a.com",
        org_id="org_a",
        members=[
            Member(email="boss@a.com", role="owner"),
            Member(email="mew@a.com"),
            Member(email="ann@a.com"),
        ],
    )
    o.create("B", "boss@b.com", org_id="org_b")
    return o


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "docs.db")


def test_one_code_two_processes_exactly_one_redeems(path):
    seed(path)
    code = IdentityLinks(orgs_on(SQLiteDocStore(path)), clock=lambda: NOW).create_code(
        "org_a", "mew@a.com", "line"
    )
    barrier = threading.Barrier(2)

    def redeem(user):
        links = IdentityLinks(orgs_on(RacyStore(path, barrier, KIND)), clock=lambda: NOW)
        return lambda: links.redeem(code.code, "line", user, "Ubot")

    # the tenant must belong to the org for an identity code
    orgs_on(SQLiteDocStore(path)).attach_workspace(
        "org_a", Workspace(id="line-Ubot", platform="line", external_id="Ubot")
    )
    results = in_threads(redeem("U-one"), redeem("U-two"))
    wins = [r for r, e in results if e is None]
    errors = [e for r, e in results if e is not None]
    assert len(wins) == 1 and len(errors) == 1 and isinstance(errors[0], CodeError)
    winner = "U-one" if results[0][1] is None else "U-two"
    org = orgs_on(SQLiteDocStore(path)).get("org_a")
    assert org.members[1].identities == {"line": winner}


def test_two_attach_codes_for_one_tenant_attach_it_to_one_org_only(path):
    seed(path)
    base = IdentityLinks(orgs_on(SQLiteDocStore(path)), clock=lambda: NOW)
    a = base.create_attach_code("org_a", "boss@a.com", "telegram")
    b = base.create_attach_code("org_b", "boss@b.com", "telegram")
    barrier = threading.Barrier(2)

    def redeem(code):
        links = IdentityLinks(orgs_on(RacyStore(path, barrier, KIND)), clock=lambda: NOW)
        return lambda: links.redeem(code.code, "telegram", "42", "bot9")

    results = in_threads(redeem(a), redeem(b))
    assert sum(e is None for _, e in results) == 1
    store = orgs_on(SQLiteDocStore(path))
    found = store.org_of_workspace("telegram", "bot9")
    owners = [o.id for o in store.all_orgs() if o.workspace("telegram-bot9")]
    assert found is not None and owners == [found[0].id]


def test_same_tenant_attached_to_two_orgs_concurrently_exactly_one_wins(path):
    seed(path)
    barrier = threading.Barrier(2)

    def attach(org_id):
        store = orgs_on(RacyStore(path, barrier, ORG_WS))
        ws = Workspace(id="slack-T1", platform="slack", external_id="T1")
        return lambda: store.attach_workspace(org_id, ws)

    results = in_threads(attach("org_a"), attach("org_b"))
    errors = [e for _, e in results if e is not None]
    assert sum(e is None for _, e in results) == 1
    assert len(errors) == 1 and isinstance(errors[0], WorkspaceTaken)
    store = orgs_on(SQLiteDocStore(path))
    assert [o.id for o in store.all_orgs() if o.workspace("slack-T1")] == [
        store.org_of_workspace("slack", "T1")[0].id
    ]


def test_concurrent_links_of_two_members_both_survive(path):
    seed(path)
    barrier = threading.Barrier(2)

    def link(email, uid):
        store = orgs_on(RacyStore(path, barrier, ORGS))
        return lambda: store.link_identity("org_a", email, "line", uid)

    results = in_threads(link("mew@a.com", "U-mew"), link("ann@a.com", "U-ann"))
    assert all(e is None for _, e in results)
    org = orgs_on(SQLiteDocStore(path)).get("org_a")
    assert {m.email: m.identities for m in org.members if m.identities} == {
        "mew@a.com": {"line": "U-mew"},
        "ann@a.com": {"line": "U-ann"},
    }


def test_concurrent_claims_on_one_account_exactly_one_member_gets_it(path):
    seed(path)
    barrier = threading.Barrier(2)

    def link(email):
        store = orgs_on(RacyStore(path, barrier, ORGS))
        return lambda: store.link_identity("org_a", email, "line", "U-same")

    results = in_threads(link("mew@a.com"), link("ann@a.com"))
    errors = [e for _, e in results if e is not None]
    assert len(errors) == 1 and isinstance(errors[0], IdentityTaken)


def test_a_failed_redeem_changes_nothing(path):
    seed(path)
    o = orgs_on(SQLiteDocStore(path))
    links = IdentityLinks(o, clock=lambda: NOW)
    o.attach_workspace("org_b", Workspace(id="line-Ubot", platform="line", external_id="Ubot"))
    code = links.create_code("org_a", "mew@a.com", "line")
    with pytest.raises(CodeError, match="another org"):
        links.redeem(code.code, "line", "U-mew", "Ubot")
    assert o.docs.get(KIND, 0, code.code.replace("-", ""))["used"] is False
    assert o.get("org_a").members[1].identities == {}


# ------------------------------------------------------------------ Firestore (fake client)
def test_firestore_transaction_retries_when_a_document_it_read_changed():
    client = FakeFirestore()
    a, b = FirestoreDocStore(client=client), FirestoreDocStore(client=client)
    a.put("k", 0, "n", {"v": 1})
    calls = []

    def bump(tx):
        calls.append(1)
        n = tx.get("k", 0, "n")["v"]
        if len(calls) == 1:  # another process commits between our read and our commit
            b.put("k", 0, "n", {"v": 10})
        tx.put("k", 0, "n", {"v": n + 1})
        return n + 1

    assert a.transact(bump) == 11 and len(calls) == 2 and client.aborts == 1
    assert a.get("k", 0, "n")["v"] == 11


def test_firestore_transaction_reads_its_own_writes_and_rolls_back_on_error():
    store = FirestoreDocStore(client=FakeFirestore())

    def boom(tx):
        tx.put("k", 0, "x", {"v": 1})
        assert tx.get("k", 0, "x")["v"] == 1
        tx.delete("k", 0, "x")
        assert tx.get("k", 0, "x") is None
        tx.put("k", 0, "y", {"v": 2})
        raise RuntimeError("no")

    with pytest.raises(RuntimeError):
        store.transact(boom)
    assert store.get("k", 0, "y") is None


def test_firestore_one_code_three_racers_exactly_one_redeems():
    client = FakeFirestore()
    mk = lambda: orgs_on(FirestoreDocStore(client=client))  # noqa: E731
    o = mk()
    o.create("A", "boss@a.com", org_id="org_a", members=[Member(email="mew@a.com", role="owner")])
    o.attach_workspace("org_a", Workspace(id="line-Ubot", platform="line", external_id="Ubot"))
    code = IdentityLinks(o, clock=lambda: NOW).create_code("org_a", "mew@a.com", "line")
    calls = [
        (lambda u=u: IdentityLinks(mk(), clock=lambda: NOW).redeem(code.code, "line", u, "Ubot"))
        for u in ("U1", "U2", "U3")
    ]
    results = in_threads(*calls)
    assert sum(e is None for _, e in results) == 1
    assert all(isinstance(e, CodeError) for _, e in results if e is not None)


def test_firestore_tenant_attached_to_two_orgs_exactly_one_wins():
    client = FakeFirestore()
    mk = lambda: orgs_on(FirestoreDocStore(client=client))  # noqa: E731
    o = mk()
    o.create("A", "a@a.com", org_id="org_a")
    o.create("B", "b@b.com", org_id="org_b")
    ws = Workspace(id="slack-T1", platform="slack", external_id="T1")
    results = in_threads(
        lambda: mk().attach_workspace("org_a", ws), lambda: mk().attach_workspace("org_b", ws)
    )
    assert sorted(type(e).__name__ for _, e in results if e) == ["WorkspaceTaken"]
