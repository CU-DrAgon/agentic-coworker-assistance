"""The gate models fail quiet: any error or odd reply means "do not answer". No network."""

import json
from types import SimpleNamespace

from conftest import FakeClient

from salid import gate
from salid.gate import GateLine

HISTORY = [GateLine("1", "สลิด", "ยอดวันนี้ 500 บาท"), GateLine("2", "พี่", "ขอบคุณ")]
LINE = GateLine("3", "พี่", "แล้วเมื่อวานล่ะ")


def reply(obj):
    return SimpleNamespace(text=obj if isinstance(obj, str) else json.dumps(obj))


def boom(contents, config):
    raise RuntimeError("vertex down")


def test_window_gate_yes_and_no():
    yes, no = FakeClient([reply({"answer": True})]), FakeClient([reply({"answer": False})])
    assert gate.addressed_to_salid(yes, "m", HISTORY, LINE, channel_id=1) is True
    assert gate.addressed_to_salid(no, "m", HISTORY, LINE, channel_id=1) is False


def test_window_gate_uses_the_cheap_json_settings_and_logs_no_text(caplog):
    client = FakeClient([reply({"answer": True})])
    with caplog.at_level("INFO", logger="salid.gate"):
        gate.addressed_to_salid(client, "gate-model", HISTORY, LINE, channel_id=42)
    model, contents, config = client.models.calls[0]
    assert model == "gate-model" and "แล้วเมื่อวานล่ะ" in contents
    assert config.temperature == 0 and config.max_output_tokens <= 50
    assert config.response_mime_type == "application/json"
    assert "channel=42" in caplog.text and "แล้วเมื่อวานล่ะ" not in caplog.text


def test_window_gate_fails_quiet():
    for script in ([boom], [reply("not json")], [reply({"nope": 1})], [reply("")]):
        assert (
            gate.addressed_to_salid(FakeClient(script), "m", HISTORY, LINE, channel_id=1) is False
        )
    assert gate.addressed_to_salid(None, "m", HISTORY, LINE, channel_id=1) is False


def test_pick_returns_a_candidate_id():
    c = FakeClient([reply({"answer": True, "message_id": "3"})])
    assert gate.pick_interjection(c, "m", HISTORY, ["2", "3"], channel_id=1) == "3"
    assert "[1] สลิด" in c.models.calls[0][1]


def test_pick_no_or_unknown_id_or_errors_mean_none():
    for script in (
        [reply({"answer": False})],
        [reply({"answer": True, "message_id": "77"})],  # not one of the candidates
        [reply({"answer": True})],
        [reply("garbage")],
        [boom],
    ):
        assert (
            gate.pick_interjection(FakeClient(script), "m", HISTORY, ["2", "3"], channel_id=1)
            is None
        )
    assert gate.pick_interjection(None, "m", HISTORY, ["2"], channel_id=1) is None


def test_gate_prompts_describe_a_bilingual_assistant():
    client = FakeClient([reply({"answer": True})])
    en = GateLine("3", "Pat", "can you check the receipt from Makro?")
    assert gate.addressed_to_salid(client, "m", HISTORY, en, channel_id=1) is True
    assert "Thai or English" in client.models.calls[0][1]
