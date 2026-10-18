"""Static checks on the Command Center page (single self-contained HTML file)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).parent.parent / "src" / "meeting_scribe" / "static" / "center.html"
ALLOWED = {"accounts.google.com", "fonts.googleapis.com", "fonts.gstatic.com"}


def html() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_file_exists_and_nonempty():
    assert PAGE.is_file() and len(html()) > 10_000


def test_no_external_hosts_except_allowed():
    src = html()
    for m in re.finditer(r"""(?:src|href)=["'](https?://[^"']+)["']""", src):
        host = re.match(r"https?://([^/:]+)", m.group(1)).group(1)
        # outbound hrefs only appear in mock sample data / JS, never as page resources
        assert host in ALLOWED, host
    for m in re.finditer(r"<script[^>]+src=[\"']([^\"']+)", src):
        assert re.match(r"https://accounts\.google\.com/", m.group(1)), m.group(1)
    assert "cdn." not in src and "unpkg" not in src and "jsdelivr" not in src


def test_placeholders_present():
    src = html()
    assert "__HUB_GOOGLE_CLIENT_ID__" in src and "__INVITE_URL__" in src


def test_mock_mode_guarded_by_query_param():
    src = html()
    assert 'get("mock") === "1"' in src
    assert "if (MOCK)" in src
    assert "localStorage.setItem(LANG_KEY" in src  # one key shared by every Salid page
    assert not re.search(r"localStorage[^;\n]*token", src, re.I)


def test_served_by_fastapi_static_read():
    # The page is plain text; make sure it parses as a single document with one script block
    src = html()
    assert src.count("<script>") == 1 and src.rstrip().endswith("</html>")


def test_google_calendar_identity_and_attach_endpoints_referenced():
    src = html()
    for needle in (
        '"/google/calendar"',
        '"/connect"',
        '"/identities"',
        '"/identities/" + p + "/code"',
        '"/workspaces/" + p + "/attach-code"',
    ):
        assert needle in src, needle
    assert "window.location = r.auth_url" in src
    assert 'get("calendar")' in src and "history.replaceState" in src


def test_new_i18n_keys_in_both_languages():
    src = html()
    th, en = src.split("\nen: {", 1)
    for key in re.findall(r"\b((?:gc|la|ab)_[a-z]+):", src):
        assert f"{key}:" in th and f"{key}:" in en, key


def between(start: str, end: str) -> str:
    src = html()
    i = src.index(start) + len(start)
    return src[i : src.index(end, i)]


def test_monitor_nav_view_and_placeholder():
    s = html()
    assert '["monitor", "n_monitor"' in s and "VIEWS.monitor = async" in s and "__MONITOR__" in s


def test_monitor_keys_in_both_languages():
    src = html()
    th, en = src.split("\nen: {", 1)
    keys = set(re.findall(r"\b(mon_[a-z0-9_]+):", src))
    assert keys, "no mon_* keys"
    for k in keys:
        assert f"{k}:" in th and f"{k}:" in en, k


def test_monitor_css_is_accessible():
    css = between("/* monitor:css */", "/* monitor:css:end */")
    assert "--faint" not in css and "min-height: 24px" in css
    assert re.search(r"prefers-reduced-motion[^}]*\.mon-flash", html())


def test_monitor_live_region_and_dialog():
    s = html()
    assert 'id: "mon-live"' in s and '"aria-live": "polite"' in s and 'role: "dialog"' in s


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_inline_script_parses(tmp_path):
    # the one inline script is large: a duplicate declaration or a syntax error must fail here,
    # not in a browser
    src = html()
    code = src.split("<script>", 1)[1].split("</script>", 1)[0]
    probe = tmp_path / "center.js"
    probe.write_text(code, encoding="utf-8")
    run = subprocess.run(["node", "--check", str(probe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


def monitor_view() -> str:
    return between("/* ---------- Monitor: a calm live view", "\nVIEWS.settings = async")


def test_monitor_hidden_tab_polls_only_for_armed_notifications():
    view = monitor_view()
    # R10: sched() and poll() ask monPollDelay (hidden + armed -> every MON_HEAL_MS, hidden alone -> stop)
    assert view.count("monPollDelay(") >= 2
    assert 'Notification.permission === "granted" && monNotifyOptIn()' in view


def test_monitor_new_pill_is_inside_the_frozen_zone():
    view = monitor_view()
    for ev in ("pointerenter", "pointerleave", "focusin", "focusout"):
        assert (
            f'zone.addEventListener("{ev}"' in view and f'list.addEventListener("{ev}"' not in view
        )
    assert re.search(r"const zone = el\([^;]*newPill[^;]*list", view)


def test_monitor_row_name_has_no_ticking_time_and_no_repeated_kind():
    view = monitor_view()
    assert '"aria-label": rowName(x)' in view and "monTitleRest(" in view


def test_route_focuses_content_only_on_a_new_page():
    s = html()
    assert "const moved = cur !== id;" in s
    assert "if (moved) c.focus({ preventScroll: true });" in s
