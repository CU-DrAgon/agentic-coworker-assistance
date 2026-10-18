"""Tiny in-memory fake of the Firestore surface FirestoreStore uses (offline tests)."""

import operator
import threading

from google.api_core.exceptions import Aborted, AlreadyExists

from salid.firestore_store import FirestoreStore

OPS = {
    "==": operator.eq,
    ">=": operator.ge,
    "<=": operator.le,
    "<": operator.lt,
    ">": operator.gt,
}


class FakeSnap:
    def __init__(self, ref, data):
        self.reference, self._data = ref, data

    @property
    def exists(self):
        return self._data is not None

    @property
    def id(self):
        return self.reference.id

    def to_dict(self):
        return None if self._data is None else dict(self._data)


class FakeRef:
    def __init__(self, col, doc_id):
        self.col, self.id = col, doc_id

    def get(self, transaction=None, **_):
        data = self.col.docs.get(self.id)
        if transaction is not None:
            transaction.reads[(self.col.name, self.id)] = None if data is None else dict(data)
        return FakeSnap(self, data)

    def create(self, data):
        if self.id in self.col.docs:
            raise AlreadyExists("exists")
        self.col.docs[self.id] = dict(data)

    def set(self, data):
        self.col.docs[self.id] = dict(data)

    def update(self, fields):
        self.col.docs[self.id].update(fields)

    def delete(self):
        self.col.docs.pop(self.id, None)


class FakeQuery:
    def __init__(self, col, filters=(), n=None):
        self.col, self.filters, self.n = col, tuple(filters), n
        col.client.queries.append(self.filters)

    def where(self, *, filter):
        return FakeQuery(self.col, (*self.filters, filter), self.n)

    def limit(self, n):
        return FakeQuery(self.col, self.filters, n)

    def stream(self):
        self.col.client.streams.append(
            (self.col.name, tuple((f.field_path, f.op_string, f.value) for f in self.filters))
        )
        out = []
        for doc_id, d in self.col.docs.items():
            ok = all(
                f.field_path in d
                and d[f.field_path] is not None
                and OPS[f.op_string](d[f.field_path], f.value)
                for f in self.filters
            )
            if ok:
                out.append(FakeSnap(FakeRef(self.col, doc_id), d))
        return iter(out[: self.n] if self.n else out)


class FakeCollection(FakeQuery):
    def __init__(self, client, name):
        self.client, self.name, self.docs = client, name, {}
        self.col, self.filters, self.n = self, (), None

    def document(self, doc_id):
        return FakeRef(self, doc_id)


class FakeTransaction:
    """Optimistic transaction with the surface `firestore.transactional` drives: reads record
    what they saw, writes are queued, and `_commit` aborts (so the decorator retries) when a
    document that was read has changed since."""

    _max_attempts = 5
    _read_only = False

    def __init__(self, client):
        self.client, self._id, self.reads, self.writes = client, None, {}, []

    def _clean_up(self):
        self._id, self.reads, self.writes = None, {}, []

    def _begin(self, retry_id=None):
        self._id = b"tx"

    def set(self, ref, data):
        self.writes.append((ref, dict(data)))

    def delete(self, ref):
        self.writes.append((ref, None))

    def _commit(self):
        with self.client.commit_lock:
            for (col, doc_id), seen in self.reads.items():
                if self.client.collection(col).docs.get(doc_id) != seen:
                    self.client.aborts += 1
                    raise Aborted("contention")
            for ref, data in self.writes:
                if data is None:
                    ref.delete()
                else:
                    ref.set(data)
        self._clean_up()

    def _rollback(self):
        self._clean_up()


class FakeFirestore:
    def __init__(self):
        self.cols, self.queries = {}, []
        self.streams = []  # one (collection, filters) entry per executed query
        self.commit_lock, self.aborts = threading.Lock(), 0

    def transaction(self):
        return FakeTransaction(self)

    def collection(self, name):
        return self.cols.setdefault(name, FakeCollection(self, name))


class OfflineFirestoreStore(FirestoreStore):
    """FirestoreStore with the transactional id counter swapped for a plain counter
    (the real transaction path is covered by the live test)."""

    def __init__(self, **kw):
        super().__init__(client=FakeFirestore(), **kw)
        self._n = 0

    def _next_id(self):
        self._n += 1
        return self._n


def offline_memory_store(**kw):
    from salid.firestore_memory import FirestoreMemoryStore

    return FirestoreMemoryStore(client=FakeFirestore(), **kw)
