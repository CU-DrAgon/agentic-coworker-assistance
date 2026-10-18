"""Reply language: setting resolution, auto detection, bilingual typed commands, the catalog.
Pure functions only, nothing touches the network."""

import re

import pytest

from salid.i18n import (
    CATALOG,
    LANGS,
    has_thai,
    parse_language_command,
    resolve_lang,
    resolve_language,
    t,
)


@pytest.mark.parametrize(
    "setting, text, expected",
    [
        ("th", "how much did we spend", "th"),
        ("en", "ยอดเท่าไร", "en"),
        ("auto", "ยอดเดือนนี้เท่าไร", "th"),
        ("auto", "how much this month?", "en"),
        ("auto", "ยอดเดือนนี้ how much", "th"),  # any Thai character wins
        ("auto", "", "en"),  # nothing to follow: the default, English
        ("auto", None, "en"),
        ("auto", "👍", "en"),
        ("auto", "500", "en"),
        ("bogus", "hello", "en"),
        (None, "ยอดเท่าไร", "en"),  # no setting: the default
    ],
)
def test_resolve_lang(setting, text, expected):
    assert resolve_lang(setting, text) == expected


def test_has_thai():
    assert has_thai("สวัสดี") and has_thai("hi สลิด") and not has_thai("hello 123")


def test_the_guild_language_applies_and_channel_overrides_are_ignored():
    # language is set for the whole server and per person, never per channel
    langs = {"10": "en", "11": "auto", "12": "bogus"}
    kw = dict(channel_languages=langs)
    assert resolve_language(10, default_language="th", **kw) == "th"
    assert resolve_language(11, default_language="th", **kw) == "th"
    assert resolve_language(12, default_language="en", **kw) == "en"
    assert resolve_language(99, default_language="en", **kw) == "en"
    assert resolve_language(99, channel_languages={}) == "en"  # no settings at all
    assert resolve_language(99, channel_languages={}, default_language="zzz") == "en"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("พูดภาษาอังกฤษ", "en"), ("ภาษาอังกฤษ", "en"), ("ตอบอังกฤษ", "en"),
        ("speak English", "en"), ("Speak English please", "en"), ("English please", "en"),
        ("  ENGLISH  ", "en"), ("reply in english", "en"), ("English!", "en"),
        ("พูดไทย", "th"), ("พูดภาษาไทย", "th"), ("ภาษาไทย", "th"), ("ภาษาไทยครับ", "th"),
        ("speak Thai", "th"), ("Thai please", "th"), ("answer in Thai", "th"),
        ("ภาษาอัตโนมัติ", "auto"), ("อัตโนมัติ", "auto"), ("auto language", "auto"),
        ("Auto", "auto"), ("auto  language!", "auto"),
        ("ภาษาไทยคืออะไร", None), ("English is hard to learn", None),
        ("how do I say thanks in Thai", None), ("เดือนนี้จ่ายเท่าไร", None), ("", None),
        ("ภาษาอังกฤษ" + "ก" * 60, None),
    ],
)  # fmt: skip
def test_parse_language_command(text, expected):
    assert parse_language_command(text) == expected


def test_t_falls_back_to_english_and_formats():
    assert t("signoff", "en") == t("signoff", "auto") == t("signoff")
    assert t("signoff", "en") != t("signoff", "th")
    assert t("meet_joined", "en", channel="Room") == "Salid has joined Room."
    with pytest.raises(KeyError):
        t("no_such_key")


def test_every_entry_has_both_languages_with_the_same_placeholders():
    field = re.compile(r"\{(\w+)(?:[:!][^}]*)?\}")
    for key, entry in CATALOG.items():
        assert set(entry) == {"th", "en"}, key
        assert set(field.findall(entry["th"])) == set(field.findall(entry["en"])), key
    assert set(LANGS) == {"th", "en", "zh", "hi", "ms", "id", "vi", "ko", "ja", "auto"}
