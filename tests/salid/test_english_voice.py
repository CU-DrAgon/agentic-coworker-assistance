"""English guild: voice instruction, spoken-reply prompt, summary prompt and captions have no Thai."""

import re

from test_live_voice import FixedClockSettings

from salid import live_voice as lv
from salid import meeting as mt
from salid import speech
from salid.captions import translate_lines

THAI = re.compile(r"[฀-๿]")


def test_en_instruction_is_pure_english():
    s = FixedClockSettings()
    for qa in (True, False):
        for language in ("en", "auto"):
            text = lv.build_instruction(s, language, qa=qa, history="Mew: hi", lang="en")
            assert not THAI.search(text)
            assert "Salid" in text and "English" in text
    assert not THAI.search(lv.build_instruction(s, "en"))
    assert "ask_salid" not in lv.build_instruction(s, "en", qa=False)


def test_th_instruction_unchanged_by_lang_en_none():
    s = FixedClockSettings()
    assert lv.build_instruction(s, "th") == lv.build_instruction(s, "th", lang="th")
    assert "พูดภาษาไทย" in lv.build_instruction(s, "th")


def test_en_notes_and_cues_have_no_thai():
    for text in (
        lv.ROOM_NOTE_EN, lv.LEDGER_NOTE_EN, lv.STATE_LISTENING_EN, lv.STATE_ENGAGED_EN,
        lv.FORCE_CUE_EN, lv.HEARD_EN, lv.NO_ROOM_EN, lv.QA_OFF_EN, *lv.FILLERS["en"],
        mt.ACK_TEXT["en"], mt.SUMMARY_PROMPT_EN, speech.SPOKEN_PROMPT_EN,
    ):  # fmt: skip
        assert not THAI.search(text)
    assert lv.FILLERS["th"]


def test_spoken_text_uses_english_prompt_for_en():
    seen = []

    class Resp:
        text = "Twelve thousand baht."

    def fake(client, model, contents, config):
        seen.append(contents[0])
        return Resp()

    orig = speech.generate
    speech.generate = fake
    try:
        speech.spoken_text(None, "m", "q", "a", "en")
        speech.spoken_text(None, "m", "q", "a")
    finally:
        speech.generate = orig
    assert not THAI.search(seen[0]) and THAI.search(seen[1])


def test_translate_captions_en_prompt_has_no_thai():
    assert translate_lines(None, "m", [], "en") == []
