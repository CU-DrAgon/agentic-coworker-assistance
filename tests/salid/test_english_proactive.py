"""English mode for what Salid sends on its own and in DMs: briefs, reports, receipt
confirmations, digest labels, company-playbook prompt, DM language resolution, channel names."""

import re
from types import SimpleNamespace

from salid import briefings as bf
from salid import company_brain as cb
from salid import extract, modes, reports
from salid.bot import SecretaryBot
from salid.i18n import t
from salid.models import Receipt
from salid.scheduling import fmt_when

THAI = re.compile(r"[฀-฾เ-๿]")  # Thai letters and marks (the baht sign is a currency symbol)


def test_receipt_confirmation_english_has_no_thai():
    r = Receipt(
        id=7, submitter_id=1, submitter_name="A", sha256="x", effective_date="2026-10-16",
        vendor="Cafe", total=350.0, category="วัตถุดิบ",
    )  # fmt: skip
    dup = r.model_copy(update={"id": 3})
    en = extract.confirmation_text(r, dup, True, "en")
    assert not THAI.search(en) and "Ingredients" in en
    th = extract.confirmation_text(r, dup, True, "th")
    assert "วัตถุดิบ" in th and extract.category_label("วัตถุดิบ", "th") == "วัตถุดิบ"


def test_every_receipt_category_has_an_english_name():
    from salid.models import CATEGORIES

    assert all(c in extract.CATEGORY_EN for c in CATEGORIES)


def test_brief_plain_english_has_no_thai():
    inp = bf.BriefInput(
        kind="morning", name="Ann", day="2026-10-16",
        meetings=[{"time": "09:00", "title": "Standup", "link": "https://x.test/m"}],
        tasks_due=[{"id": 1, "title": "Send deck", "due": "2026-10-16", "overdue": False}],
    )  # fmt: skip
    assert not THAI.search(bf.render_plain(inp, "en"))
    assert THAI.search(bf.render_plain(inp, "th"))


def test_report_and_brief_prompts_name_the_language():
    assert "{language}" in bf.BRIEF_PROMPT and "translate" in bf.BRIEF_PROMPT
    assert "{language}" in reports.REPORT_PROMPT and "translate" in reports.REPORT_PROMPT


def test_company_block_english_variant():
    pb = cb.CompanyPlaybook(
        industry="restaurant",
        rules=[cb.TextItem(id="a", text="Keep answers short")],
        glossary=[cb.Term(id="b", term="PM", meaning="project manager")],
    )
    en = cb.company_block(pb, "what is PM", nudge=True, lang="en")
    assert not THAI.search(en) and "Company playbook" in en and "Reply in English" in en
    empty = cb.company_block(cb.CompanyPlaybook(), "x", nudge=True, lang="en")
    assert not THAI.search(empty) and "get to know the company" in empty
    th = cb.company_block(pb, "what is PM", nudge=True, lang="th")
    assert "กฎบริษัท" in th and "สลิดยังไม่รู้จัก" in cb.company_block(
        cb.CompanyPlaybook(), "x", nudge=True, lang="th"
    )


def test_when_labels_and_catalog_pairs_have_no_thai_in_english():
    import datetime as dt

    d = dt.datetime(2026, 10, 19, 14, 0)
    assert not THAI.search(fmt_when(d, "en")) and THAI.search(fmt_when(d, "th"))
    for key in ("fu_header", "fu_line", "fu_footer", "ask_reminder", "alert_doc",
                "sec_guard_warn", "report_need_channel", "unlicensed", "welcome",
                "rc_unclear_pure", "corr_unclear_pure", "translate_nothing_pure",
                "translate_failed_pure"):  # fmt: skip
        assert not THAI.search(t(key, "en", **_fill(key))), key


def _fill(key):
    from string import Formatter

    from salid.i18n import CATALOG

    names = [f for _, f, _, _ in Formatter().parse(CATALOG[key]["en"]) if f]
    return {n: "X" for n in names}


def test_channel_names_per_language():
    assert modes.chat_channel_name("en") == "chat-with-salid"
    assert modes.chat_channel_name("th") == modes.CHAT_CHANNEL_NAME
    assert modes.is_chat_channel_name("chat-with-salid") and modes.is_chat_channel_name(
        modes.CHAT_CHANNEL_NAME
    )


def _dm_fake(shared, langs, active=None):
    guilds = {g: SimpleNamespace(id=g) for g in shared}
    return SimpleNamespace(
        _shared_guilds=lambda uid: [(guilds[g], object()) for g in shared],
        dm_guilds=SimpleNamespace(active=lambda uid: active),
        _profile_of=lambda gid, uid: SimpleNamespace(language=langs.get(("p", gid))),
        team=SimpleNamespace(current=lambda gid: SimpleNamespace(language_setting=langs[gid])),
    )


def _dm(fake):
    msg = SimpleNamespace(author=SimpleNamespace(id=5), guild=None)
    return SecretaryBot._dm_setting(fake, msg)


def test_dm_language_is_the_company_default_unless_the_person_set_one():
    assert _dm(_dm_fake([1], {1: "en"})) == "en"
    assert _dm(_dm_fake([1], {1: "th"})) == "th"  # a company that chose Thai
    assert _dm(_dm_fake([1], {1: "en", ("p", 1): "th"})) == "th"  # personal wins
    assert _dm(_dm_fake([1], {1: "th", ("p", 1): "en"})) == "en"
    assert _dm(_dm_fake([1], {1: "auto"})) == "auto"
    assert _dm(_dm_fake([], {})) == "auto"


def test_dm_language_with_two_companies_follows_the_picked_one():
    langs = {1: "en", 2: "th"}
    assert _dm(_dm_fake([1, 2], langs, active=1)) == "en"
    assert _dm(_dm_fake([1, 2], langs, active=2)) == "th"
    assert _dm(_dm_fake([1, 2], langs)) == "auto"  # undecided and they disagree
    assert _dm(_dm_fake([1, 2], {1: "en", 2: "en"})) == "en"
