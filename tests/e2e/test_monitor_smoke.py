"""Playwright smoke of the Monitor page (`/center?mock=1#/monitor`): the keyboard path through a
filter, a row and the detail drawer (focus trap and return), names on tiles and the live region, and
an axe WCAG 2.2 AA scan of the page content and the drawer (spec section 5 and section 10)."""

import re

import pytest
from axe_playwright_python.sync_playwright import Axe
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e

MONITOR = "/center?mock=1#/monitor"
DRAWER = ".mon-drawer"  # the chat panel (#panel) is a dialog too: always pick the monitor drawer


def test_keyboard_path_filters_rows_and_drawer(page):
    page.goto(MONITOR)
    expect(page.get_by_role("heading", level=1, name="Monitor")).to_be_visible()
    expect(page.locator("#content")).to_be_focused()  # route focus
    kind = page.get_by_role("combobox", name="Kind")
    expect(kind).to_be_visible()
    for _ in range(30):  # the filter is reachable with Tab from the page start
        if kind.evaluate("n => n === document.activeElement"):
            break
        page.keyboard.press("Tab")
    expect(kind).to_be_focused()
    kind.select_option("video_ad")  # ArrowDown opens the native picker on macOS
    expect(page.get_by_role("button", name="Remove filter: Kind")).to_be_visible()  # chip
    row = page.get_by_role("button", name=re.compile("Video ad"))
    row.focus()
    page.keyboard.press("Enter")
    dialog = page.locator(DRAWER)
    expect(dialog).to_be_visible()
    expect(dialog).to_have_attribute("role", "dialog")
    for _ in range(12):
        page.keyboard.press("Tab")
    assert page.evaluate(f"document.querySelector('{DRAWER}').contains(document.activeElement)")
    page.keyboard.press("Escape")
    expect(dialog).to_be_hidden()
    expect(row).to_be_focused()


def test_tiles_and_live_region_have_names(page):
    page.goto(MONITOR)
    expect(page.get_by_role("button", name=re.compile("Needs attention"))).to_be_visible()
    expect(page.locator("#mon-live")).to_have_attribute("aria-live", "polite")


WCAG = {
    "runOnly": {
        "type": "tag",
        "values": ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"],
    },
    "resultTypes": ["violations"],
}


def test_axe_finds_no_wcag_aa_violations_in_the_monitor(page):
    page.goto(MONITOR)
    page.get_by_role("button", name=re.compile("Video ad")).click()
    expect(page.locator(DRAWER + " .mon-evs li").first).to_be_visible()  # the events have loaded
    for ctx in ("#content", DRAWER):
        res = Axe().run(page, context=ctx, options=WCAG)
        assert res.violations_count == 0, res.generate_report()
