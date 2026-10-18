"""The Salid landing page (/salid): static, bilingual, honest about the invite link."""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from meeting_scribe.config import get_settings

PAGE = Path(__file__).parents[1] / "src/meeting_scribe/static/salid.html"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("STORE", "memory")
    monkeypatch.setenv("USE_PUBSUB", "false")
    get_settings.cache_clear()
    from meeting_scribe.app import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def _dict_and_html():
    html = PAGE.read_text("utf-8")
    m = re.search(r'<script type="application/json" id="i18n-data">(.*?)</script>', html, re.S)
    assert m, "the dictionary must be an inline JSON block"
    return json.loads(m.group(1)), html


def test_landing_page_is_served_without_an_invite_when_unconfigured(client):
    r = client.get("/")
    assert r.status_code == 200
    assert 'href="" data-invite' in r.text  # the page then shows the "not configured" label
    assert "__SALID_INVITE_URL__" not in r.text


def test_landing_page_uses_the_real_invite_link(client, monkeypatch):
    monkeypatch.setenv("SALID_DISCORD_CLIENT_ID", "123456789012345678")
    get_settings.cache_clear()
    from meeting_scribe.app import SALID_PERMISSIONS

    r = client.get("/")
    url = (
        "https://discord.com/oauth2/authorize?client_id=123456789012345678"
        f"&scope=bot+applications.commands&permissions={SALID_PERMISSIONS}"
    )
    assert r.text.count(f'href="{url}" data-invite') == 1  # secondary CTA at the end
    assert r.text.count('href="/start"') >= 6  # nav, hero, teams panel, plans, entry, final


def test_dictionary_has_same_keys_in_both_languages():
    d, _ = _dict_and_html()
    assert d
    for key, entry in d.items():
        assert set(entry) == {"th", "en"}, key
        assert entry["th"].strip() and entry["en"].strip(), key


def test_every_used_key_exists_and_markup_has_no_hardcoded_thai():
    d, html = _dict_and_html()
    used = set(re.findall(r'data-i18n(?:-html)?="(\w+)"', html))
    for attrs in re.findall(r'data-i18n-attr="([^"]+)"', html):
        used |= {p.split(":")[1] for p in attrs.split(";")}
    script = html.split('id="i18n-data"', 1)[1]
    used |= set(re.findall(r'\bt\("(\w+)"\)', script))
    assert used - set(d) == set(), used - set(d)
    markup = html.split('<script type="application/json"', 1)[0].split("</head>", 1)[1]
    assert not re.search(r"[฀-๿]", markup)


def test_language_is_english_by_default_and_a_saved_choice_wins():
    """Judges read English: every page opens in English; the browser language is ignored."""
    _, html = _dict_and_html()
    assert 'const LANG_KEY = "salid-language"' in html
    assert 'let lang = readLang("get") || "en";' in html
    assert "navigator.language" not in html
    start = START.read_text("utf-8")
    assert 'let lang = readLang("get") || "en";' in start and "navigator.language" not in start
    center = (PAGE.parent / "center.html").read_text("utf-8")
    assert 'let lang = "en";' in center and '<html lang="en">' in center
    assert "localStorage.setItem(LANG_KEY" in html and "catch" in html
    assert html.count('data-lang="th"') == 1 and html.count('data-lang="en"') == 1


def test_hero_has_a_motion_safe_final_state_and_the_cat_stays_visible():
    _, html = _dict_and_html()
    assert 'data-stage="idle"' in html  # final state is the markup default (no JS / errors)
    assert "prefers-reduced-motion" in html
    for stage in ("a", "e", "idle"):
        assert f'[data-stage="{stage}"]' in html, stage
    # The official still sits under the film, so the cat is visible whatever the film does.
    assert '<img class="still" src="/static/salid/cat-hero.webp"' in html
    video_tag = re.search(r"<video[^>]*>", html).group(0)
    assert "muted" in video_tag and "playsinline" in video_tag
    assert "autoplay" not in video_tag  # the script starts it, so reduced motion can opt out
    assert 'poster="/static/salid/cat-portal.webp"' in html
    for name in ("cat-hero.webp", "cat-portal.webp"):
        assert (PAGE.parent / "salid" / name).stat().st_size > 0, name


def test_mascot_assets_are_served_but_not_the_page_templates(client):
    r = client.get("/static/salid/cat-hero.webp")
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    assert client.get("/static/salid/../hub.html").status_code == 404
    assert client.get("/static/hub.html").status_code == 404


def test_callouts_name_only_shipped_features():
    from salid.guild_settings import FEATURES

    for flag in ("voice_meetings", "receipts", "followups", "documents", "wiki"):
        assert flag in FEATURES
    _, html = _dict_and_html()
    assert html.count('class="callout c') == 4
    for href in ("#platforms", "#features", "#plans", "#teams", "/hub", "/recorder"):
        assert f'href="{href}"' in html


def test_landing_page_routes_visitors_to_every_service():
    """The entry section points at the three services; the Discord card falls back to /hub."""
    _, html = _dict_and_html()
    entry = html.split('id="start"', 1)[1].split("</section>", 1)[0]
    assert 'href="/start"' in entry and 'href="/center"' in entry and 'href="/recorder"' in entry
    for section in ("platforms", "plans"):
        assert f'id="{section}"' in html, section


def test_hero_offers_sign_up_then_login_then_a_divider_before_explore():
    """Hero order: create account (/start), log in (/center), a separator, then explore."""
    d, html = _dict_and_html()
    hero = html.split('class="copy"', 1)[1].split('class="stage"', 1)[0]
    signup, login = hero.index('href="/start"'), hero.index('href="/center"')
    sep, explore = hero.index('role="separator"'), hero.index('data-i18n="cta_explore"')
    assert signup < login < sep < explore
    assert 'data-i18n="cta_login"' in hero
    assert d["cta_login"]["en"].startswith("Log in") and "เข้าสู่ระบบ" in d["cta_login"]["th"]


START = PAGE.parent / "start.html"


def _start_dict_and_html():
    html = START.read_text("utf-8")
    m = re.search(r'<script type="application/json" id="i18n-data">(.*?)</script>', html, re.S)
    assert m
    return json.loads(m.group(1)), html


def test_start_page_is_served_with_placeholders_filled(client, monkeypatch):
    r = client.get("/start")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    assert 'const CLIENT_ID = ""' in r.text  # unconfigured: the page shows the notice
    assert "__SALID_INVITE_URL__" not in r.text and "__HUB_GOOGLE_CLIENT_ID__" not in r.text
    monkeypatch.setenv("HUB_GOOGLE_CLIENT_ID", "123-abc.apps.googleusercontent.com")
    monkeypatch.setenv("SALID_DISCORD_CLIENT_ID", "123456789012345678")
    get_settings.cache_clear()
    r = client.get("/start")
    assert 'const CLIENT_ID = "123-abc.apps.googleusercontent.com"' in r.text
    assert 'href="https://discord.com/oauth2/authorize?client_id=123456789012345678' in r.text


def test_start_page_dictionary_and_markup_are_bilingual():
    d, html = _start_dict_and_html()
    for key, entry in d.items():
        assert set(entry) == {"th", "en"}, key
        assert set(re.findall(r"\{(\w+)\}", entry["th"])) == set(
            re.findall(r"\{(\w+)\}", entry["en"])
        ), key
    used = set(re.findall(r'data-i18n(?:-html)?="(\w+)"', html))
    for attrs in re.findall(r'data-i18n-attr="([^"]+)"', html):
        used |= {p.split(":")[1] for p in attrs.split(";")}
    script = html.split('id="i18n-data"', 1)[1]
    used |= set(re.findall(r'\bt\("(\w+)"', script))
    used |= {"pl_trial"}  # t("pl_" + plan) at runtime; only the trial exists at launch
    used.discard("pl_")
    assert used - set(d) == set(), used - set(d)
    markup = html.split('<script type="application/json"', 1)[0].split("</head>", 1)[1]
    assert not re.search(r"[\u0e00-\u0e7f]", markup)
    assert 'const LANG_KEY = "salid-language"' in html  # same choice as the landing page


def test_start_page_uses_only_existing_center_routes():
    _, html = _start_dict_and_html()
    assert 'const API = "/api/center"' in html
    assert 'api("/me")' in html and 'api("/orgs", { method: "POST"' in html
    assert "license/request" not in html  # launch: the trial is the only plan, nothing is charged
    assert 'data-plan="trial"' in html and 'data-plan="monthly"' not in html
    assert 'href="/center"' in html


def test_start_hands_the_credential_to_center_once_per_tab():
    """Same tab, sessionStorage, read once, only while the ID token is still valid."""
    _, start = _start_dict_and_html()
    center = (PAGE.parent / "center.html").read_text("utf-8")
    assert 'sessionStorage.setItem("salid_center_handoff"' in start
    assert 'sessionStorage.getItem("salid_center_handoff")' in center
    assert 'sessionStorage.removeItem("salid_center_handoff")' in center
    assert "Number(h.exp) * 1000 > Date.now()" in center
    assert "localStorage" not in start.split("salid_center_handoff", 1)[1]  # never persisted


def test_every_page_shares_the_salid_theme(client):
    """One design system: every page links the shared stylesheet, which is served as CSS."""
    for name in ("salid.html", "start.html", "hub.html", "center.html", "index.html"):
        html = (PAGE.parent / name).read_text("utf-8")
        assert '<link rel="stylesheet" href="/static/salid/theme.css">' in html, name
        assert "prefers-color-scheme" not in html, (
            name
        )  # light theme only, same on every judge's machine
    r = client.get("/static/salid/theme.css")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/css")
    assert "--indigo:" in r.text and ".btn.primary" in r.text


def test_unconfigured_invite_is_hidden_not_labelled_with_an_env_var():
    """Visitors never see an internal variable name: the invite button hides and /start explains."""
    _, html = _dict_and_html()
    assert "a.hidden = !ok;" in html
    d, start = _start_dict_and_html()
    assert "a.hidden = !ok;" in start and '$("invite-soon").hidden = ok;' in start
    assert 'id="invite-soon" data-i18n="invite_soon"' in start
    assert "SALID_DISCORD_CLIENT_ID" not in d["invite_soon"]["en"] + d["invite_soon"]["th"]


def test_every_page_shares_one_language_key_and_syncs_it_with_the_account():
    """One choice follows the person: same localStorage key on every page, and after sign-in the
    language saved with the account wins (or the local choice is saved to the account)."""
    for name in ("salid.html", "start.html", "hub.html", "center.html"):
        html = (PAGE.parent / name).read_text("utf-8")
        assert 'LANG_KEY = "salid-language"' in html, name
    for name, put in (
        ("start.html", 'api("/me/language", { method: "PUT"'),
        ("hub.html", 'api("/api/hub/me/language", { method: "PUT"'),
        ("center.html", 'api("/api/center/me/language", { method: "PUT"'),
    ):
        html = (PAGE.parent / name).read_text("utf-8")
        assert put in html and "syncLang(me.language)" in html, name


def test_every_signed_in_page_keeps_one_salid_session():
    """Sign in once, stay signed in: start, hub and center exchange the Google sign-in for a
    Salid session (localStorage `salid_session`), reuse it on load and clear it on sign-out."""
    for name in ("start.html", "hub.html", "center.html"):
        html = (PAGE.parent / name).read_text("utf-8")
        assert 'const SESSION_KEY = "salid_session"' in html, name
        assert 'fetch("/api/hub/session", { method: "POST"' in html, name
        assert "exchangeSession(" in html and 'session("get")' in html, name
    assert 'session("clear")' in (PAGE.parent / "hub.html").read_text("utf-8")
    assert 'session("clear")' in (PAGE.parent / "center.html").read_text("utf-8")
