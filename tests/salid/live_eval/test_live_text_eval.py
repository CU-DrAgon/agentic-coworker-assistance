"""Live text usability evaluation (real Gemini). Run: `uv run pytest tests/salid/live_eval -m live`.
One test per scenario id; the whole suite is run once per session (module fixture)."""

import asyncio

import pytest

from live_eval import text_scenarios as ts

pytestmark = pytest.mark.live

IDS = [s.id for s in ts.build_scenarios()]


@pytest.fixture(scope="module")
def rows():
    out: list[dict] = []
    world = asyncio.run(ts.run_text_suite(out))
    yield {r["id"]: r for r in out}
    world.close()


@pytest.mark.parametrize("scenario_id", IDS)
def test_scenario(rows, scenario_id):
    r = rows[scenario_id]
    assert r["passed"], f"{r['scenario']}: {r['detail']['failures']} | got: {r['got']}"
