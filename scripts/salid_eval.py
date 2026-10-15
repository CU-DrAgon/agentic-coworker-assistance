"""Live usability evaluation of Salid: real Gemini / Live / Cloud TTS, fake Discord objects.

    uv run python scripts/salid_eval.py                  # everything
    uv run python scripts/salid_eval.py --suite text     # text scenarios only
    uv run python scripts/salid_eval.py --only greeting,dm_me
    uv run python scripts/salid_eval.py --suite voice

Prints a table (scenario, expected, got, pass/fail, latency) and writes a JSON report into the
system temp dir. Costs real model calls (about 100 for the text suite) -- see the live_eval
package. No Discord network access, no Firestore writes (a temp copy of the dev ledger is used).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import textwrap
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "salid"))
warnings.filterwarnings("ignore")


def table(rows: list[dict], width: int = 44) -> str:
    def cell(s, n):
        s = " ".join(str(s).split())
        return textwrap.shorten(s, n, placeholder="...")

    head = f"{'suite':5} {'scenario':26} {'result':6} {'lat':>6}  {'expected':{width}} | got"
    lines = [head, "-" * len(head)]
    for r in rows:
        mark = "PASS" if r["passed"] else "FAIL"
        lines.append(
            f"{r['suite']:5} {cell(r['id'], 26):26} {mark:6} {r['latency_s']:>5.1f}s  "
            f"{cell(r['expected'], width):{width}} | {cell(r['got'], 110)}"
        )
        if not r["passed"]:
            for f in r.get("detail", {}).get("failures", []):
                lines.append(f"{'':5} {'':26} {'':6} {'':>6}    -> {cell(f, 150)}")
    return "\n".join(lines)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["text", "voice", "all"], default="all")
    ap.add_argument("--only", default="", help="comma separated scenario/utterance ids")
    ap.add_argument("--out", default="", help="JSON report path (default: system temp dir)")
    a = ap.parse_args()
    only = {x for x in a.only.split(",") if x} or None
    rows: list[dict] = []
    calls: dict = {}
    t0 = time.monotonic()
    if a.suite in ("text", "all"):
        from live_eval import text_scenarios as ts

        world = await ts.run_text_suite(rows, only)
        calls["text"] = dict(ts.LIVE_CALLS)
        world.close()
    if a.suite in ("voice", "all"):
        from live_eval import voice_harness as vh

        await vh.run_voice_suite(rows, only)
        calls["voice"] = dict(getattr(vh, "LIVE_CALLS", {}))
    print(table(rows))
    ok = sum(r["passed"] for r in rows)
    print(f"\n{ok}/{len(rows)} passed in {time.monotonic() - t0:.0f}s; live calls: {calls}")
    out = Path(a.out) if a.out else Path(tempfile.gettempdir()) / f"salid_eval_{int(time.time())}.json"
    out.write_text(json.dumps({"rows": rows, "live_calls": calls}, ensure_ascii=False, indent=2, default=str))
    print(f"report: {out}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
