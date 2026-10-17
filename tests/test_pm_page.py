"""/pm serves the PM dashboard page; static checks on the page and its demo dataset."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from meeting_scribe.app import app
from meeting_scribe.config import get_settings

PAGE = Path(__file__).parent.parent / "src" / "meeting_scribe" / "static" / "pm.html"
CLIENT = "123456-abc.apps.googleusercontent.com"
NAV_LABELS = [
    "หน้าหลัก", "Home", "โครงการ", "Projects", "งานของฉัน", "My Tasks",
    "ภาพรวมบริษัท", "Company Insights", "ตั้งค่าพื้นที่ทำงาน", "Workspace Settings",
]  # fmt: skip


def html() -> str:
    return PAGE.read_text(encoding="utf-8")


def between(src: str, start: str, end: str) -> str:
    i, j = src.index(start) + len(start), src.index(end)
    return src[i:j]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("HUB_GOOGLE_CLIENT_ID", CLIENT)
    get_settings.cache_clear()
    yield TestClient(app)
    get_settings.cache_clear()


def test_pm_route_fills_client_id_and_is_no_store(client):
    r = client.get("/pm")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert r.headers["cache-control"] == "no-store"
    assert CLIENT in r.text and "__HUB_GOOGLE_CLIENT_ID__" not in r.text


def test_five_pages_and_bilingual_nav():
    src = html()
    for page in ("home", "projects", "mytasks", "insights", "settings"):
        assert f'id="page-{page}"' in src
    for label in NAV_LABELS:
        assert label in src


def test_only_google_identity_script_and_no_secrets():
    src = html()
    scripts = re.findall(r"<script[^>]+src=[\"']([^\"']+)", src)
    assert scripts == ["https://accounts.google.com/gsi/client"]
    assert "cdn." not in src and "unpkg" not in src and "jsdelivr" not in src
    assert not re.search(r"localStorage[^;\n]*token", src, re.I)  # token stays in memory
    assert not re.search(r"AIza|sk-[A-Za-z0-9]{20}|BEGIN (RSA )?PRIVATE", src)
    assert 'get("demo") === "1"' in src


def test_demo_dataset_parses_and_has_the_sample_items():
    data = json.loads(between(html(), "/*DEMO-START*/", "/*DEMO-END*/"))
    projects = data["projects"]
    standard = [p for p in projects if p["kind"] == "standard"]
    hackathons = [p for p in projects if p["kind"] == "hackathon"]
    assert len(standard) >= 2 and len(hackathons) == 1
    assert len(hackathons[0]["hackathon"]["deliverables"]) == 4
    assert any(
        p["priority"] == "high" and p["status"] == "in_progress" and p["deadline"].startswith("@-")
        for p in projects
    )  # the late high-priority project that needs review
    assert all(p.get("budget") for p in projects)
    assert {t["project_id"] for t in data["tasks"]} >= {p["id"] for p in projects}
    assert any(not t["assignee"] for t in data["tasks"])
    assert data["settings"]["members"] and data["ai_proposals"]
    heard = [u for p in projects for u in p.get("updates", [])]
    assert heard and all(u["text"] and u["at"] for u in heard)  # "Latest updates" on the overview
    assert "upd_title" in html() and "P.updates" in html()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_script_is_valid_js_and_every_label_has_th_and_en(tmp_path):
    src = html()
    js = tmp_path / "pm.js"
    js.write_text(between(src, "<script>\n", "</script>\n</body>"), encoding="utf-8")
    subprocess.run(["node", "--check", str(js)], check=True)
    labels = between(src, "/*I18N-START*/", "/*I18N-END*/")
    probe = tmp_path / "probe.js"
    probe.write_text(
        f"const S={labels};"
        "const bad=Object.entries(S).filter(([k,v])=>!Array.isArray(v)||v.length!==2"
        "||!v[0].trim()||!v[1].trim());"
        "console.log(JSON.stringify({bad:bad.map(b=>b[0]),n:Object.keys(S).length}));",
        encoding="utf-8",
    )
    out = subprocess.run(["node", str(probe)], check=True, capture_output=True, text=True).stdout
    res = json.loads(out)
    assert res["bad"] == [] and res["n"] > 150
