"""Workflow monitor foundation: item model, access matrix, attention rules and error codes."""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from salid import monitor
from salid.monitor import (
    ACTIVE,
    BOT_KINDS,
    ERROR_TEXT,
    FINAL,
    KINDS,
    STATUSES,
    Audience,
    MonitorItem,
    SafeError,
    Step,
    Viewer,
    allowed,
    apply_attention,
    controls,
    heartbeat_state,
    mine,
    person_key,
    safe_error,
    stuck_after_s,
    viewer_for,
    visible,
)
from salid.orgs import Member, Org

NOW = datetime(2026, 10, 18, 9, 0, tzinfo=UTC)


def V(role: str, ids: tuple[str, ...] = (), ex: bool = False) -> Viewer:
    return Viewer("x@y.com", role, frozenset({"x@y.com", *ids}), ex)


def item(
    kind: str = "video_ad",
    status: str = "running",
    *,
    age_s: float = 0,
    audience: Audience | None = None,
    **kw,
) -> MonitorItem:
    at = (NOW - timedelta(seconds=age_s)).isoformat()
    return MonitorItem(
        kind=kind,
        id="j1",
        status=status,
        started_at=at,
        updated_at=at,
        audience=audience or Audience(),
        **kw,
    )


def test_constants_and_item_key():
    assert ACTIVE == ("queued", "running", "waiting", "paused")
    assert FINAL == ("done", "failed", "cancelled")
    assert set(STATUSES) == set(ACTIVE) | set(FINAL) and len(STATUSES) == 7
    assert set(KINDS) == {
        "video_ad",
        "deep_task",
        "research",
        "change",
        "action",
        "meeting",
        "task",
        "approval",
        "vote",
        "form",
        "reminder",
    }
    assert set(BOT_KINDS) == {"video_ad", "deep_task", "research", "action"}
    it = item("deep_task", steps=[Step(name="plan")])
    assert it.key == "deep_task:j1"
    assert it.steps[0].model_dump() == {
        "name": "plan",
        "state": "pending",
        "started_at": "",
        "ms": None,
        "attempts": 0,
        "error_code": "",
    }
    assert (it.attention, it.attention_reason, it.error, it.cost_usd) == (False, "", None, None)
    assert Audience().model_dump() == {
        "admin_only": False,
        "exec_only": False,
        "dm": False,
        "people": [],
        "channel_ids": [],
    }
    assert person_key("discord", 7) == "discord:7" and person_key("slack", "U1") == "slack:U1"


@pytest.mark.parametrize(
    "role,ex,kind,aud,ids,ok",
    [
        ("owner", False, "video_ad", Audience(people=["discord:7"]), (), True),
        ("owner", False, "video_ad", Audience(dm=True, people=["discord:7"]), (), False),
        ("admin", False, "deep_task", Audience(exec_only=True, people=["discord:7"]), (), False),
        ("admin", True, "deep_task", Audience(exec_only=True, people=["discord:7"]), (), True),
        ("member", False, "change", Audience(admin_only=True), (), False),
        ("member", False, "action", Audience(admin_only=True, people=["x@y.com"]), (), False),
        ("member", False, "video_ad", Audience(people=["discord:7"]), ("discord:7",), True),
        ("member", False, "video_ad", Audience(people=["discord:8"]), ("discord:7",), False),
        ("super", False, "deep_task", Audience(exec_only=True), (), True),
        ("super", False, "video_ad", Audience(dm=True, people=["discord:7"]), (), False),
        # beyond the pinned rows: the dm owner, admin_only for staff, a non-exec admin's own job
        (
            "member",
            False,
            "video_ad",
            Audience(dm=True, people=["discord:7"]),
            ("discord:7",),
            True,
        ),
        ("admin", False, "change", Audience(admin_only=True), (), True),
        (
            "admin",
            False,
            "deep_task",
            Audience(exec_only=True, people=["discord:7"]),
            ("discord:7",),
            True,
        ),
        # deep tasks are executive-only per spec 4.1 even when an adapter forgot the flag
        ("admin", False, "deep_task", Audience(people=["discord:7"]), (), False),
        ("admin", False, "video_ad", Audience(exec_only=True, people=["discord:7"]), (), False),
        ("", False, "video_ad", Audience(people=["discord:7"]), (), False),
    ],
)
def test_visible_roles_by_kinds_matrix(role, ex, kind, aud, ids, ok):
    viewer = V(role, ids, ex)
    assert allowed(kind, aud, viewer) is ok
    assert visible(MonitorItem(kind=kind, id="1", status="running", audience=aud), viewer) is ok


def test_controls_matrix():
    seven = Audience(people=["discord:7"])
    staff_only = Audience(admin_only=True, people=["x@y.com"])
    own_deep = Audience(exec_only=True, people=["discord:7"])
    me = ("discord:7",)
    assert controls(item("video_ad", audience=seven), V("owner")) == ["cancel"]
    assert controls(item("video_ad", "queued", audience=seven), V("owner")) == ["cancel"]
    assert controls(item("action", "failed", audience=staff_only), V("owner")) == ["retry"]
    assert controls(item("change", "failed", audience=Audience(admin_only=True)), V("owner")) == [
        "resend"
    ]
    assert controls(item("change", "failed", audience=Audience(admin_only=True)), V("admin")) == [
        "resend"
    ]
    assert controls(item("video_ad", "done", audience=seven), V("owner")) == []
    assert controls(item("research", audience=seven), V("admin")) == ["cancel"]
    mine_video = item("video_ad", audience=seven, requester_id="discord:7")
    assert controls(mine_video, V("member", me)) == ["cancel"]
    mine_deep = item("deep_task", audience=own_deep, requester_id="discord:7")
    assert controls(mine_deep, V("member", me)) == []
    assert controls(mine_deep, V("member", me, ex=True)) == ["cancel"]
    # a non-executive admin cannot see someone else's deep task, so cannot cancel it
    assert controls(item("deep_task", audience=own_deep), V("admin")) == []
    assert controls(item("deep_task", audience=own_deep), V("admin", ex=True)) == ["cancel"]
    mine_action = item("action", "failed", audience=staff_only, requester_id="x@y.com")
    assert controls(mine_action, V("member")) == []
    # in the audience but not the requester: no cancel for a member
    assert (
        controls(item("video_ad", audience=seven, assignee_id="discord:7"), V("member", me)) == []
    )
    assert controls(item("video_ad", audience=seven), V("super")) == []
    # never a control on an item the viewer cannot see (dm items are hidden from the owner)
    dm = Audience(dm=True, people=["discord:7"])
    assert controls(item("video_ad", audience=dm, requester_id="discord:7"), V("owner")) == []


def test_mine_matches_requester_assignee_or_waiting_on():
    me = V("member", ("discord:7",))
    assert mine(item(requester_id="discord:7"), me)
    assert mine(item(assignee_id="x@y.com"), me)
    assert mine(item(waiting_on="discord:7"), me)
    assert not mine(item(requester_id="discord:8"), me)
    assert not mine(item(), me)


def _org(level: int, role: str = "admin") -> Org:
    member = Member(
        email="a@x.com",
        role=role,
        discord_user_id="7",
        identities={"slack": "U1", "line": ""},
        level=level,
    )
    return Org(id="org_t", name="T", members=[member])


def test_viewer_for_collects_verified_ids_and_executive_level():
    v = viewer_for(_org(3), "A@x.com", "admin")
    assert v.ids == {"a@x.com", "discord:7", "slack:U1"}
    assert (v.email, v.role, v.executive) == ("a@x.com", "admin", True)
    assert viewer_for(_org(5), "a@x.com", "admin").executive is False
    assert viewer_for(_org(0), "a@x.com", "admin").executive is False
    assert viewer_for(_org(0, "owner"), "a@x.com", "owner").executive is True
    assert viewer_for(_org(1, "member"), "a@x.com", "member").executive is False
    ops = viewer_for(_org(3), "ops@salid.app", "super")
    assert (ops.role, ops.ids, ops.executive) == ("super", frozenset({"ops@salid.app"}), False)


def test_heartbeat_thresholds_and_clock_skew():
    ages = {90: "online", 91: "delayed", 300: "delayed", 301: "offline", -120: "online"}
    for age, want in ages.items():
        assert heartbeat_state((NOW - timedelta(seconds=age)).isoformat(), NOW) == want
    assert heartbeat_state(None, NOW) == heartbeat_state("garbage", NOW) == "offline"
    assert heartbeat_state("", NOW) == "offline"
    # a naive timestamp is read as UTC
    assert heartbeat_state("2026-10-18T08:59:00", NOW) == "online"


def test_stuck_thresholds_per_kind():
    assert stuck_after_s("video_ad") == 600
    assert stuck_after_s("deep_task") == 720
    assert stuck_after_s("deep_task", deep_timeout_s=900) == 1020
    assert stuck_after_s("research") == 900
    assert stuck_after_s("change") is None and stuck_after_s("action") is None


def test_attention_rules():
    def att(it: MonitorItem, bot: str = "online", **kw) -> tuple[bool, str]:
        out = apply_attention(it, NOW, bot, **kw)
        return out.attention, out.attention_reason

    none = (False, "")
    assert att(item("video_ad", age_s=601)) == (True, "stuck")
    assert att(item("video_ad", age_s=599)) == none
    assert att(item("deep_task", age_s=721), deep_timeout_s=600) == (True, "stuck")
    assert att(item("deep_task", age_s=719), deep_timeout_s=600) == none
    assert att(item("deep_task", age_s=1019), deep_timeout_s=900) == none
    assert att(item("research", age_s=901)) == (True, "stuck")
    assert att(item("research", age_s=899)) == none
    assert att(item("video_ad", "done", age_s=10_000)) == none
    assert att(item("action", "queued", age_s=61)) == (True, "stuck")
    assert att(item("action", "queued", age_s=61), "delayed") == none
    assert att(item("action", "queued", age_s=59)) == none
    assert att(item("video_ad"), "offline") == (True, "bot_offline")
    assert att(item("action", "queued"), "offline") == (True, "bot_offline")
    assert att(item("deep_task", "done"), "offline") == none
    assert att(item("change"), "offline") == none
    for kind in KINDS:
        assert att(item(kind, "failed", age_s=10_000), "offline") == (True, "failed")
    # adapters set their own attention (changes: stuck); it is kept when no rule applies
    flagged = item("change", "queued", attention=True, attention_reason="stuck")
    assert att(flagged) == (True, "stuck")
    # pure: the input item is never changed
    original = item("video_ad", age_s=601)
    apply_attention(original, NOW)
    assert (original.attention, original.attention_reason) == none


def test_safe_error_never_echoes_input():
    assert safe_error("discord: SECRET").model_dump() == {
        "code": "unknown",
        "message": ERROR_TEXT["unknown"],
    }
    assert safe_error("month_limit") == SafeError(
        code="month_limit", message=ERROR_TEXT["month_limit"]
    )
    assert safe_error("").code == "unknown"


def test_error_text_codes_are_short_english_sentences():
    assert set(ERROR_TEXT) == {
        "unknown",
        "restarted",
        "failed",
        "error",
        "month_limit",
        "storyboard",
        "render",
        "upload_refused",
        "not_confirmed",
        "delivery_failed",
        "missing_permission",
        "not_found",
        "bad_request",
        "discord_error",
        "expired",
    }
    for text in ERROR_TEXT.values():
        assert text.isascii() and text[0].isupper() and text.endswith(".")
        assert len(text) <= 80 and text.count(".") == 1


def test_monitor_never_imports_discord():
    tree = ast.parse(Path(monitor.__file__).read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [n for n in names if n == "discord" or n.startswith("discord.")]
