import json
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from google.genai import types

from salid.config import Settings
from salid.models import Receipt
from salid.store import SQLiteStore

EXEC_ID, ALICE, BOB = 1, 100, 200


class FakeModels:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        item = self.script.pop(0)
        return item(contents, config) if callable(item) else item


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


def text_resp(text):
    return SimpleNamespace(
        text=text,
        candidates=[
            SimpleNamespace(content=types.Content(role="model", parts=[types.Part(text=text)]))
        ],
    )


def json_resp(obj):
    return text_resp(json.dumps(obj, ensure_ascii=False))


def fc_resp(*calls):
    parts = [types.Part(function_call=types.FunctionCall(name=n, args=a)) for n, a in calls]
    return SimpleNamespace(
        text=None, candidates=[SimpleNamespace(content=types.Content(role="model", parts=parts))]
    )


class FixedClockSettings(Settings):
    def now(self):
        return datetime(2026, 10, 14, 12, 0, tzinfo=ZoneInfo("Asia/Bangkok"))


@pytest.fixture
def settings():
    return FixedClockSettings(exec_user_ids={EXEC_ID})


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(str(tmp_path / "t.db"))


def mk(store, n, uid, name, vendor, day, total, cat="วัตถุดิบ", conf=0.9, sha=None, dup=None):
    return store.add_receipt(
        Receipt(
            submitter_id=uid,
            submitter_name=name,
            sha256=sha or f"sha{n}",
            vendor=vendor,
            receipt_date=day,
            effective_date=day,
            total=total,
            category=cat,
            confidence=conf,
            possible_duplicate_of=dup,
            created_at=day,
        )
    )


@pytest.fixture
def seeded(store):
    mk(store, 1, ALICE, "Alice", "Makro", "2026-10-01", 1000.0)
    mk(store, 2, ALICE, "Alice", "Grab", "2026-10-05", 250.5, "ค่าเดินทาง")
    mk(store, 3, BOB, "Bob", "Makro", "2026-10-05", 9000.0)
    mk(store, 4, BOB, "Bob", "PEA", "2026-09-20", 3000.0, "สาธารณูปโภค")
    mk(store, 5, BOB, "Bob", "Ice Co", "2026-10-14", 120.0, conf=0.4)
    return store
