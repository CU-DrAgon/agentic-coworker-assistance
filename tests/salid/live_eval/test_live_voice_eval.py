"""Live voice usability evaluation (real Cloud TTS audio, real Gemini transcription, a real Gemini
Live session). Skipped by default; run with `uv run pytest -m live tests/salid/live_eval -k voice`."""

import pytest

from live_eval import voice_harness as vh

pytestmark = pytest.mark.live


@pytest.mark.parametrize("mode", ["classic", "live"])
async def test_voice_meeting_behaviour(mode):
    report: list[dict] = []
    await vh.run_voice_suite(report, modes=(mode,), alt=False)
    bad = [
        f"{r['id']}: expected {r['expected']}; got {r['got']}" for r in report if not r["passed"]
    ]
    assert not bad, "\n".join(bad)
