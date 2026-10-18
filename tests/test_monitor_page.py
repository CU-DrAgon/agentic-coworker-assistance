"""The monitor page's pure feed functions (center.html), run in Node.

The block between the two markers is DOM-free on purpose: it is sliced out of the page and run with
a small harness that prints JSON (the same pattern as the hub retention test)."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).parent.parent / "src" / "meeting_scribe" / "static" / "center.html"
START = "/* ---------- monitor: pure functions (Node-tested, no DOM) ---------- */"
END = "/* ---------- monitor: end pure functions ---------- */"


def block() -> str:
    src = PAGE.read_text(encoding="utf-8")
    return src[src.index(START) : src.index(END) + len(END)]


HARNESS = r"""
const { isDeepStrictEqual } = require("node:util");
Date.now = () => { throw new Error("pure functions never read the clock"); };
const out = {};

out.backoff = [0, 1, 2, 3, 4, 5, 6].map((n) => monBackoff(n, 0.5));
out.jitter = [monBackoff(0, 0), monBackoff(0, 1)];

const feed = (f, online, paused, hidden) => monFeedState({ failures: f, online, paused, hidden });
out.feed = {
  f1: feed(1, true, false, false),
  f2: feed(2, true, false, false),
  f4: feed(4, true, false, false),
  net: feed(0, false, false, false),
  paused5: feed(5, false, true, true),
  hidden: feed(5, false, false, true),
};

const ev = (seq, kind, id) => ({ v: 1, seq, at: "2026-10-18T10:00:00+00:00", kind, id, status: "running", attention: false });
const resp = (seq, floor, reset, events) => ({ seq, floor, now: "2026-10-18T10:00:00+00:00", events, reset });
const pick = (r) => ({ seq: r.st.seq, refetch: r.refetch, reload: r.reload });
out.reduce_ok = pick(monReduce({ seq: 10 }, resp(12, 0, false, [
  ev(12, "video_ad", "j"), ev(11, "change", "c"), ev(9, "change", "old"), ev(12, "zzz", "x"),
])));
// unique seqs as the server sends them: a key changed twice is fetched once, an unknown kind is skipped
out.reduce_dedupe = pick(monReduce({ seq: 10 }, resp(15, 0, false, [
  ev(14, "zzz", "x"), ev(13, "change", "c"), ev(12, "video_ad", "j"), ev(11, "change", "c"),
])));
out.reduce_stale = pick(monReduce({ seq: 10 }, resp(8, 0, false, [ev(8, "change", "z")])));
out.reduce_reset = pick(monReduce({ seq: 10 }, resp(14, 0, true, [ev(14, "change", "c")])));
out.reduce_floor = pick(monReduce({ seq: 10 }, resp(20, 15, false, [ev(16, "change", "c")])));
// the log was recreated (the server answers `reset` because since > seq): reload, cursor back to 3
out.reduce_recreated = pick(monReduce({ seq: 50 }, resp(3, 0, true, [ev(3, "change", "c")])));
const keep = { seq: 10, other: "kept" };
out.reduce_keeps_state = monReduce(keep, resp(11, 0, false, [ev(11, "task", "t")])).st.other;

const row = (id, updated_at, rev, _req = 1) => ({ kind: "task", id, updated_at, rev, _req });
const A = row("a", "2026-10-18T10:05:00+00:00", 1);
const B = row("b", "2026-10-18T10:00:00+00:00", 1);
const frozen = monMerge([A, B], [row("b", "2026-10-18T10:06:00+00:00", 2), row("c", "2026-10-18T10:07:00+00:00", 1)], true);
out.merge_frozen = { order: frozen.rows.map((r) => r.id), b_rev: frozen.rows[1].rev, added: frozen.added };
const free = monMerge([A, B], [
  row("b", "2026-10-18T10:06:00+00:00", 2), row("c", "2026-10-18T10:07:00+00:00", 1),
  row("a", "2026-10-18T10:01:00+00:00", 0, 9),
], false);
out.merge_free = { order: free.rows.map((r) => r.id), added: free.added };
out.older_ignored = free.rows.find((r) => r.id === "a").rev === 1;
out.merge_pure = A.rev === 1 && B.rev === 1;
// same updated_at: the later request wins; microsecond and "Z" forms parse alike
out.newer_req = [
  monNewer(row("a", "2026-10-18T10:00:00+00:00", 2, 5), row("a", "2026-10-18T10:00:00+00:00", 1, 4)),
  monNewer(row("a", "2026-10-18T10:00:00+00:00", 2, 3), row("a", "2026-10-18T10:00:00+00:00", 1, 4)),
  monNewer(row("a", "2026-10-18T10:00:00.500000+00:00", 2, 1), row("a", "2026-10-18T10:00:00Z", 1, 4)),
];

const fetchedAt = Date.parse("2026-10-18T07:00:00Z");  // the local clock is 3 h behind the server
out.elapsed = monElapsed("2026-10-18T10:00:00+00:00", "", "2026-10-18T10:05:00+00:00", fetchedAt, fetchedAt + 3000);
out.elapsed_done = monElapsed("2026-10-18T10:00:00+00:00", "2026-10-18T10:01:30+00:00", "2026-10-18T11:00:00+00:00", fetchedAt, fetchedAt + 3000);
out.elapsed_unknown = monElapsed("", "", "2026-10-18T10:05:00+00:00", fetchedAt, fetchedAt);

out.toast = [monToastGate(1000, 5000), monToastGate(1000, 11000)];

const state = {
  tab: "attention", kind: ["video_ad", "change"], status: ["failed"], owner: ["discord:42", "a+b@x.co"],
  workspace: ["discord-1000"], from: "2026-10-01", to: "2026-10-18", priority: ["high"], q: "งบ, Q4 & co",
};
out.hash = monHash(state);
out.hash_roundtrip = isDeepStrictEqual(monParseHash(out.hash), state) && monHash(monParseHash(out.hash)) === out.hash;
out.hash_empty = monHash(monParseHash("#/monitor"));
out.hash_defaults = monParseHash("#/monitor");
out.hash_bad = monParseHash("#/monitor?kind=%E0%A4,task,&zzz=1&q=%&tab=active");

const gate = (o) => monNotifyGate({ optedIn: true, permission: "granted", hidden: true, mine: true, isAdmin: false, status: "failed", attention: false, ...o });
out.notify = {
  mine_failed_hidden: gate({}),
  visible_tab: gate({ hidden: false }),
  not_opted_in: gate({ optedIn: false }),
  permission_default: gate({ permission: "default" }),
  admin_other_failed: gate({ mine: false, isAdmin: true }),
  member_other_failed: gate({ mine: false }),
  mine_done: gate({ status: "done" }),
};
out.notify_mine_attention = gate({ status: "running", attention: true });
out.notify_admin_other_attention = gate({ mine: false, isAdmin: true, status: "running", attention: true });

const w = monOnError(429, 7);
out.errors = {
  401: monOnError(401).action, 402: monOnError(402).action, 403: monOnError(403).action,
  500: monOnError(500).action, 429: [w.action, w.delayMs, w.failure],
};
const net = monOnError(0), noHeader = monOnError(429);
out.errors_more = { net: [net.action, net.failure], no_retry_after: noHeader.delayMs, stop: [401, 402, 403].map((s) => monOnError(s).failure) };

console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_monitor_pure_functions_in_node(tmp_path):
    probe = tmp_path / "monitor.js"
    probe.write_text(block() + "\n" + HARNESS, encoding="utf-8")
    run = subprocess.run(["node", str(probe)], check=True, capture_output=True, text=True)
    out = json.loads(run.stdout)

    assert out["backoff"] == [5000, 10000, 20000, 40000, 60000, 60000, 60000]  # rnd 0.5, n = 0..6
    assert out["jitter"] == [4500, 5500]  # n=0, rnd 0 and 1
    assert out["feed"] == {
        "f1": "live",
        "f2": "delayed",
        "f4": "offline",
        "net": "offline",
        "paused5": "paused",
        "hidden": "hidden",
    }
    assert out["reduce_ok"] == {
        "seq": 12,
        "refetch": ["change:c", "video_ad:j"],
        "reload": False,
    }  # events 12, 11, 9(dup), zzz(unknown)
    assert out["reduce_dedupe"] == {
        "seq": 15,
        "refetch": ["change:c", "video_ad:j"],
        "reload": False,
    }
    assert out["reduce_stale"] == {"seq": 10, "refetch": [], "reload": False}  # resp.seq 8
    # reset; floor 15 > seq 10
    assert out["reduce_reset"]["reload"] and out["reduce_floor"]["reload"]
    # a reload moves the cursor to the log's seq, so the next poll does not ask to reload again
    assert out["reduce_reset"] == {"seq": 14, "refetch": [], "reload": True}
    assert out["reduce_floor"] == {"seq": 20, "refetch": [], "reload": True}
    assert out["reduce_recreated"] == {"seq": 3, "refetch": [], "reload": True}
    assert out["reduce_keeps_state"] == "kept"
    assert out["merge_frozen"] == {"order": ["a", "b"], "b_rev": 2, "added": 1}
    assert out["merge_free"]["order"] == ["c", "b", "a"] and out["older_ignored"] is True
    assert out["merge_free"]["added"] == 1 and out["merge_pure"] is True
    assert out["newer_req"] == [True, False, True]
    assert out["elapsed"] == 303000  # 5 min + 3 s, skewed Date
    assert out["elapsed_done"] == 90000 and out["elapsed_unknown"] is None
    assert out["toast"] == [False, True] and out["hash_roundtrip"] is True
    assert out["hash"] == (
        "#/monitor?tab=attention&kind=video_ad,change&status=failed&owner=discord%3A42,a%2Bb%40x.co"
        "&workspace=discord-1000&from=2026-10-01&to=2026-10-18&priority=high"
        "&q=%E0%B8%87%E0%B8%9A%2C%20Q4%20%26%20co"
    )
    assert out["hash_empty"] == "#/monitor"
    assert out["hash_defaults"] == {
        "tab": "",
        "kind": [],
        "status": [],
        "owner": [],
        "workspace": [],
        "from": "",
        "to": "",
        "priority": [],
        "q": "",
    }
    assert out["hash_bad"] == {**out["hash_defaults"], "tab": "active", "kind": ["task"]}
    assert out["notify"] == {
        "mine_failed_hidden": True,
        "visible_tab": False,
        "not_opted_in": False,
        "permission_default": False,
        "admin_other_failed": True,
        "member_other_failed": False,
        "mine_done": False,
    }
    assert out["notify_mine_attention"] is True
    assert out["notify_admin_other_attention"] is False  # admins hear about others' failures only
    assert out["errors"] == {
        "401": "auth",
        "402": "licence",
        "403": "forbidden",
        "500": "retry",
        "429": ["wait", 7000, False],
    }  # Retry-After: 7
    assert out["errors_more"] == {
        "net": ["retry", True],
        "no_retry_after": 2000,
        "stop": [False, False, False],
    }


def test_monitor_pure_block_stays_pure_and_in_place():
    src = PAGE.read_text(encoding="utf-8")
    code = block()
    assert src.count(START) == 1 and src.count(END) == 1
    assert src.index("<script>") < src.index(START) < src.index("/* ---------- state & routing")
    for banned in (
        r"\bdocument\b",
        r"\bwindow\b",
        r"\blocalStorage\b",
        r"\bfetch\(",
        r"\bt\(",
        r"\bsetTimeout\b",
        r"\bsetInterval\b",
        r"\bDate\.now\b",
        r"\bnavigator\b",
    ):
        assert not re.search(banned, code), banned
    assert re.search(
        r"MON_KINDS = \[\"video_ad\", \"deep_task\", \"research\", \"change\", \"action\", "
        r"\"meeting\", \"task\", \"approval\", \"vote\", \"form\", \"reminder\"\]",
        code,
    )
    for const in (
        "MON_BACKOFF_MS = [5000, 10000, 20000, 40000, 60000]",
        "MON_JITTER = 0.1",
        "MON_HEAL_MS = 60000",
        "MON_TOAST_GAP_MS = 10000",
        "MON_SUMMARY_MS = 60000",
    ):
        assert const in code, const
