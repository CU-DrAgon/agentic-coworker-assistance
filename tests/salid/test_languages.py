"""Second-tier reply languages (zh hi ms id vi ko ja): detection, commands, the generated catalogs,
prompts, voice codes, document fonts. Thai and English stay the two hand-written languages."""

import contextlib
import importlib
import json
import pkgutil
import re
import string
from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import FixedClockSettings

import salid
from salid import i18n, pdf_report, qa, speech
from salid.guild_settings import GuildSettingsUpdate, MemoryGuildStore
from salid.i18n import (
    CATALOG,
    EXTRA,
    LANG_NAMES,
    LANGS,
    base_lang,
    detect_lang,
    parse_language_command,
    parse_server_language_command,
    resolve_lang,
    t,
)
from salid.ideas_pdf import markdown_pdf
from salid.live_voice import build_instruction
from salid.profiles import Profile, apply_change

LOCALES = Path(i18n.__file__).parent / "locales"
PH = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*(?:[:!][^}]*)?\}")


def test_tiers_and_names():
    assert i18n.PRIMARY == ("th", "en")
    assert EXTRA == ("zh", "hi", "ms", "id", "vi", "ko", "ja")
    assert ("th", "en", *EXTRA, "auto") == LANGS
    assert LANG_NAMES["zh"] == ("中文", "Chinese", "จีน")
    assert i18n.LANGUAGE_LABELS["ja"] == "日本語 · Japanese"
    assert [base_lang(x) for x in ("th", "en", "ja", "hi")] == ["th", "en", "en", "en"]


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("สวัสดีค่ะ", "th"),
        ("こんにちは、元気ですか", "ja"),
        ("カタカナ", "ja"),
        ("안녕하세요", "ko"),
        ("你好，请帮我看看", "zh"),
        ("नमस्ते आप कैसे हैं", "hi"),
        ("Xin chào, bạn khỏe không? đây là ví dụ", "vi"),
        ("Hello there", "en"),
        ("Selamat pagi", "en"),  # Malay / Indonesian are never auto-detected
        ("12345 :)", "en"),  # no letters: the default, English
    ],
)
def test_auto_detects_by_script(text, lang):
    assert resolve_lang("auto", text) == lang
    assert resolve_lang(None, text) == "en"  # no setting: English, whatever the text
    assert i18n.has_thai("สวัสดี") and not i18n.has_thai("こんにちは")


def test_explicit_settings_pass_through():
    for lang in (*i18n.PRIMARY, *EXTRA):
        assert resolve_lang(lang, "anything") == lang
    assert detect_lang("") is None


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("speak Japanese", "ja"),
        ("พูดญี่ปุ่น", "ja"),
        ("พูดภาษาญี่ปุ่น", "ja"),
        ("日本語で", "ja"),
        ("日本語で話して", "ja"),
        ("说中文", "zh"),
        ("中文", "zh"),
        ("พูดจีน", "zh"),
        ("speak Chinese", "zh"),
        ("한국어로", "ko"),
        ("พูดเกาหลี", "ko"),
        ("nói tiếng Việt", "vi"),
        ("speak Vietnamese", "vi"),
        ("bahasa Melayu", "ms"),
        ("speak Malay", "ms"),
        ("bahasa Indonesia", "id"),
        ("พูดอินโดนีเซีย", "id"),
        ("हिंदी में", "hi"),
        ("speak Hindi", "hi"),
        ("speak English", "en"),
        ("พูดไทย", "th"),
        ("ภาษาอัตโนมัติ", "auto"),
    ],
)
def test_language_commands(text, lang):
    assert parse_language_command(text) == lang


@pytest.mark.parametrize(
    "text", ["what is Japanese food", "จีน", "tell me about Korean history please", ""]
)
def test_questions_are_not_language_commands(text):
    assert parse_language_command(text) is None


def test_server_language_command_with_extra_languages():
    assert parse_server_language_command("ใช้ภาษาญี่ปุ่นทั้งเซิร์ฟเวอร์") == "ja"
    assert parse_server_language_command("switch the whole server to Korean") == "ko"
    assert parse_server_language_command("switch the whole server from Thai to English") == "en"
    assert parse_server_language_command("ใช้ภาษาอังกฤษทั้งเซิร์ฟเวอร์") == "en"
    assert parse_server_language_command("speak Japanese") is None  # no scope word


# ------------------------------------------------------------------ catalogs
def _locale(lang):
    return json.loads((LOCALES / f"{lang}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def full_catalog():
    """Feature modules add their strings to CATALOG when imported: import them all first, like
    scripts/salid_translate_catalog.py, so the checks below do not depend on test order."""
    for m in pkgutil.iter_modules(salid.__path__):
        if m.name != "__main__":
            with contextlib.suppress(Exception):  # a module that cannot import has no strings
                importlib.import_module(f"salid.{m.name}")
    return CATALOG


@pytest.mark.parametrize("lang", EXTRA)
def test_locale_file_is_valid(lang, full_catalog):
    data = _locale(lang)
    assert set(data) == {"_meta", "strings"}
    strings = data["strings"]
    assert set(strings) <= set(CATALOG)
    for key, text in strings.items():
        assert isinstance(text, str) and text.strip(), key
        assert sorted(PH.findall(text)) == sorted(PH.findall(CATALOG[key]["en"])), (lang, key)
        names = {f for _, f, _, _ in string.Formatter().parse(text) if f}
        text.format(**dict.fromkeys(names, 1))  # no stray braces
    assert set(data["_meta"]["source_hash"]) == set(strings)


@pytest.mark.parametrize("lang", EXTRA)
def test_locale_coverage_is_at_least_95_percent(lang, full_catalog):
    translatable = [k for k, v in CATALOG.items() if not i18n.has_thai(v["en"])]
    have = sum(1 for k in translatable if k in _locale(lang)["strings"])
    assert have / len(translatable) >= 0.95, (lang, have, len(translatable))


def test_t_tiers_and_fallbacks():
    assert t("quote_title", "th") == "ใบเสนอราคา"
    assert t("quote_title", "en") == "QUOTATION"
    for lang in EXTRA:
        out = t("quote_title", lang)
        assert out and out != "ใบเสนอราคา"
    assert t("quote_title", "ja") != "QUOTATION"  # translated, not the English fallback
    assert t("quote_rev", "ja", n=3).count("3") == 1
    assert t("quote_title", "xx") == "QUOTATION"  # unknown: the default, English


def test_missing_key_in_a_locale_falls_back_to_english_never_thai(monkeypatch):
    monkeypatch.setitem(i18n._LOCALES, "ko", {})
    assert t("quote_title", "ko") == "QUOTATION"
    monkeypatch.setitem(i18n._LOCALES, "vi", {"quote_title": "BÁO GIÁ"})
    assert t("quote_title", "vi") == "BÁO GIÁ"


def test_missing_locale_file_is_empty(monkeypatch):
    monkeypatch.setattr(i18n, "_LOCALES", {})
    monkeypatch.setattr(i18n, "EXTRA", ("zz", *EXTRA))
    assert t("quote_title", "zz") == "QUOTATION"


# ------------------------------------------------------------------ prompts
def test_extra_language_prompt_has_reply_line_and_english_blocks():
    s = FixedClockSettings()
    p = qa.stable_prompt("ko", False)
    assert p.startswith("Reply ONLY in Korean (한국어). Keep the name Salid.")
    en = qa.stable_prompt("en", False)
    assert p.split("\n", 1)[1] == en  # the same English blocks after the reply line
    v = qa.volatile_prompt(
        s, s.now(), scoped=False, interjection=False, language="ko", files=False,
        profile="", company="", people=None,
    )  # fmt: skip
    assert "write your whole reply in Korean (한국어)" in v
    assert "Reply ONLY" not in qa.stable_prompt("th", False)
    assert "Reply ONLY" not in en


def test_particle_swap_only_for_thai():
    s = FixedClockSettings(persona_particle="ครับ")
    ja = qa.system_prompt(s, False, language="ja")
    th = qa.system_prompt(s, False, language="th")
    assert "ค่ะ" in ja  # untouched Thai guidance text keeps its particles for non-Thai
    assert "ค่ะ" not in th
    assert ja.startswith("Reply ONLY in Japanese (日本語).")


def test_tone_blocks_are_english_for_extra_languages():
    from salid import tone

    assert tone.tone_block("friendly", "ja") == tone.tone_block("friendly", "en")
    assert tone.voice_line("calm", "ko") == tone.voice_line("calm", "en")
    assert tone.tone_block("friendly", "auto") != tone.tone_block("friendly", "en")


def test_live_voice_instruction_in_extra_language():
    s = FixedClockSettings()
    p = build_instruction(s, "ja", lang="ja")
    assert "always speak Japanese (日本語)" in p
    assert "Speak plain, natural Japanese." in p and "Thai particles" in p
    assert "always speak English" in build_instruction(s, "en", lang="en")
    assert "ภาษา: พูดภาษาไทย" in build_instruction(s, "th", lang="th")


def test_reply_language_line_only_for_extra():
    assert i18n.reply_language_line("th") == "" == i18n.reply_language_line("en")
    assert "Hindi (हिन्दी)" in i18n.reply_language_line("hi")


def test_filler_pool_and_prompt_for_extra_language():
    from salid import fillers

    line = fillers.pick_fallback("slow", "ja", choose=lambda xs: xs[0])
    assert line == t("ack_slow_1", "ja") and not i18n.has_thai(line)
    w = fillers.FillerWriter(None, FixedClockSettings())
    prompt = w.prompt(kind="doc", question="q", lang="ko", particle="ค่ะ")
    assert "Korean (한국어)" in prompt and "Thai particles" in prompt
    assert fillers.clean_line("読みます", lang="ja", question="") == "読みます"
    assert fillers.clean_line("สักครู่ค่ะ", lang="ja", question="") is None


# ------------------------------------------------------------------ settings, profiles
def test_validators_accept_every_language():
    store = MemoryGuildStore()
    store.save_settings(
        1, GuildSettingsUpdate(default_language="ja", channel_languages={"5": "ko"}), "t@x.com"
    )
    s = store.get(1)
    assert s.default_language == "ja" and s.channel_languages == {"5": "ko"}
    store.set_server_language(1, "zh")
    assert store.get(1).default_language == "zh"
    with pytest.raises(ValueError):
        store.set_server_language(1, "auto")
    with pytest.raises(ValueError):
        store.set_server_language(1, "xx")
    # a stored channel value is ignored: the server language applies everywhere
    assert i18n.resolve_language(5, channel_languages={"5": "ja"}, default_language="hi") == "hi"
    assert i18n.resolve_language(6, channel_languages={}, default_language="hi") == "hi"


def test_profile_language_accepts_all():
    p = Profile()
    for raw, code in [("ja", "ja"), ("Japanese", "ja"), ("ภาษาเกาหลี", "ko"), ("日本語", "ja")]:
        assert apply_change(p, "language", raw) in (None, "") or p.language == code
        assert p.language == code, raw


def test_hub_dropdown_lists_every_language():
    html = (Path(i18n.__file__).parents[1] / "meeting_scribe/static/hub.html").read_text("utf-8")
    m = re.search(r"const LANGUAGES = (\[[^\]]*\]);", html)
    assert json.loads(m.group(1)) == list(LANGS)
    for lang in EXTRA:
        assert f'"lang_{lang}"' in html


# ------------------------------------------------------------------ voice
EXPECTED_CODES = {"zh": "cmn-CN", "hi": "hi-IN", "ms": "ms-MY", "id": "id-ID", "vi": "vi-VN",
                  "ko": "ko-KR", "ja": "ja-JP"}  # fmt: skip


def test_tts_codes_and_fallback_voices():
    assert {k: speech._LANG_CODES[k] for k in EXTRA} == EXPECTED_CODES
    assert speech._cloud_voice("Leda", "ja") == ("ja-JP", "ja-JP-Chirp3-HD-Leda")
    assert speech._cloud_voice("Leda", "zh") == ("cmn-CN", "cmn-CN-Chirp3-HD-Leda")
    assert speech._cloud_voice("Leda", "th") == ("th-TH", "th-TH-Chirp3-HD-Leda")
    # Chirp 3 HD has no ms-MY voices: Malay uses the Indonesian voice of the same name
    assert speech._VOICE_FALLBACK == {"ms": "id-ID"}
    assert speech._cloud_voice("Leda", "ms") == ("id-ID", "id-ID-Chirp3-HD-Leda")
    assert speech._cloud_voice("th-TH-Chirp3-HD-Kore", "th")[1] == "th-TH-Chirp3-HD-Kore"


def test_spoken_prompt_names_the_language():
    assert "Speak Japanese only" in speech.spoken_prompt("ja")
    assert speech.spoken_prompt("en") is speech.SPOKEN_PROMPT_EN
    assert speech.spoken_prompt("th") is speech.SPOKEN_PROMPT


# ------------------------------------------------------------------ meeting prompts
def test_meeting_prompts_for_extra_languages():
    from salid import meeting

    assert meeting.transcribe_prompt("th") == meeting.TRANSCRIBE_PROMPT
    p = meeting.transcribe_prompt("ja")
    assert "Thai, English or Japanese" in p and p.endswith(meeting.TRANSCRIBE_PROMPT)

    seen = []

    class Client:
        class models:  # noqa: N801
            @staticmethod
            def generate_content(model, contents, config):
                seen.append(contents[0])
                return type("R", (), {"text": "summary"})()

    assert meeting.summarize_transcript(Client, "m", "[00:01] A: hi", "ko") == "summary"
    assert "summary of the meeting in Korean" in seen[0]
    assert "whole summary in Korean only" in seen[0]


# ------------------------------------------------------------------ fonts and documents
requires_noto = pytest.mark.skipif(
    not Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").is_file()
    or not Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf").is_file(),
    reason="fonts-noto-cjk / fonts-noto-core not installed",
)


@pytest.fixture
def font_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_report, "font_cache_dir", lambda: tmp_path / "fonts")
    return tmp_path / "fonts"


def test_font_table_by_language(monkeypatch, font_cache):
    monkeypatch.setattr(pdf_report, "script_fonts", lambda lang: None)
    for lang in ("th", "en", "vi", "ms", "id"):  # today's fonts (Kanit/Sarabun cover Vietnamese)
        assert pdf_report.find_fonts(lang=lang).script is None
        assert not pdf_report.fonts_missing(lang)
    for lang in ("zh", "ja", "ko", "hi"):
        assert pdf_report.fonts_missing(lang)
    assert pdf_report.CJK_FACES == {
        "zh": "Noto Sans CJK SC", "ja": "Noto Sans CJK JP", "ko": "Noto Sans CJK KR"
    }  # fmt: skip
    assert pdf_report.DECK_FONTS["hi"] == "Noto Sans Devanagari"


@requires_noto
def test_script_fonts_are_extracted_once(font_cache):
    for lang, face in pdf_report.CJK_FACES.items():
        fs = pdf_report.find_fonts(lang=lang)
        assert fs.script == lang and Path(fs.regular).is_file() and Path(fs.bold).is_file()
        assert Path(fs.regular).parent == font_cache
        assert face.replace(" ", "") in fs.regular
    hi = pdf_report.find_fonts(lang="hi")
    assert hi.script == "hi" and "Devanagari" in hi.regular
    before = {p: p.stat().st_mtime_ns for p in font_cache.iterdir()}
    pdf_report.find_fonts(lang="ja")
    assert {p: p.stat().st_mtime_ns for p in font_cache.iterdir()} == before


@requires_noto
@pytest.mark.parametrize(
    ("lang", "text", "marker"),
    [("ja", "こんにちは、会議のまとめです。", "NotoSansCJKJP"),
     ("hi", "नमस्ते, यह बैठक का सारांश है।", "NotoSansDevanagari")],
)  # fmt: skip
def test_pdf_with_japanese_and_hindi(font_cache, lang, text, marker):
    pdf = markdown_pdf(
        org="Salid", title="Salid", markdown=f"## {text}\n\n{text}", lang=lang,
        generated_at=datetime(2026, 10, 17, tzinfo=UTC), fonts=pdf_report.find_fonts(lang=lang),
    )  # fmt: skip
    assert pdf.startswith(b"%PDF")
    assert marker.encode() in pdf.replace(b"+", b"") or marker.encode() in pdf


def test_missing_font_falls_back_to_english_with_a_note(monkeypatch):
    monkeypatch.setattr(pdf_report, "script_fonts", lambda lang: None)
    assert "ภาษาญี่ปุ่น" not in pdf_report.font_note("th")
    note = pdf_report.font_note("ja")
    assert note and "日本語" in note
    assert pdf_report.font_note("en") == "" and pdf_report.font_note("vi") == ""
    pdf = pdf_report._Pdf(pdf_report.FontSet(None, None, False), "Salid", "ja", datetime.now(UTC))
    assert pdf.lang == "en"  # the fixed labels switch to English
    pdf_th = pdf_report._Pdf(
        pdf_report.FontSet(None, None, False), "Salid", "th", datetime.now(UTC)
    )
    assert pdf_th.lang == "th"
    assert t("docs_font_missing", "en", language="Japanese").startswith("Fonts for Japanese")


@requires_noto
def test_cards_use_script_fonts(font_cache):
    from salid import cards

    cards.card_fonts.cache_clear()
    ja = cards.card_fonts("ja")
    assert "NotoSansCJKJP" in ja.regular and ja.bold
    assert "Kanit" in cards.card_fonts().regular


@requires_noto
def test_deck_sets_east_asian_and_devanagari_font_slots():
    from pptx import Presentation

    from salid import deck

    for lang, face in (("ko", "Noto Sans CJK KR"), ("hi", "Noto Sans Devanagari")):
        spec = deck.DeckSpec.model_validate(
            {"title": "t", "slides": [{"kind": "bullets", "title": "s", "bullets": ["a"]}]}
        )
        data = deck.render_deck(spec, {}, lang=lang, today=datetime(2026, 10, 17).date())
        import io

        xml = b"".join(
            sl._element.xml.encode() for sl in Presentation(io.BytesIO(data)).slides
        ).decode()
        assert face in xml
    assert deck._DECK_LANG.get() == "th"  # reset after rendering
