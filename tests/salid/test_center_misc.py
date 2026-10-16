"""Smaller pieces of the command center: analyzer, blob stores, connector switches, bot sync."""

import json
from types import SimpleNamespace

import httpx
import pytest
from conftest import FakeClient, json_resp

from salid.channel_access import is_public, public_channel_ids
from salid.config import Settings
from salid.connectors import ClickUpConnector, ConnectorConfig, ConnectorError, SlackConnector
from salid.guild_settings import LinkError, MemoryGuildStore
from salid.library import GcsBlobs, Library, LocalBlobs, make_analyzer, make_blobs


def test_analyzer_reads_text_and_categorises_with_flash_lite():
    client = FakeClient(
        [json_resp({"doc_type": "contract", "summary": "Lease for the shop", "text": ""})]
    )
    out = make_analyzer(client, "gemini-3.5-flash-lite")(
        b"Lease agreement text", "lease.txt", "text/plain"
    )
    assert out == {
        "tag": "contract",
        "summary": "Lease for the shop",
        "text": "Lease agreement text",
    }
    assert client.models.calls[0][0] == "gemini-3.5-flash-lite"


def test_analyzer_describes_images_and_rejects_unknown_types():
    client = FakeClient(
        [json_resp({"description": "A receipt", "text": "Makro 100", "type": "receipt"})]
    )
    out = make_analyzer(client, "m")(b"\x89PNG", "scan.png", "image/png")
    assert out["tag"] == "receipt" and out["summary"] == "A receipt" and out["text"] == "Makro 100"
    from salid.docs import DocUnreadable

    with pytest.raises(DocUnreadable):
        make_analyzer(client, "m")(b"MZ", "x.exe", "application/octet-stream")


def test_local_blobs_roundtrip_and_path_guard(tmp_path):
    b = LocalBlobs(tmp_path / "up")
    uri = b.save("org1", "d1", "A Plan.PDF", b"data", "application/pdf")
    assert uri.endswith("org1/d1.pdf") and b.read(uri) == b"data"
    with pytest.raises(ValueError):
        b.read(str(tmp_path / "secret.txt"))


class FakeBlob:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def upload_from_string(self, data, content_type=None):
        self.store[self.name] = (data, content_type)

    def download_as_bytes(self):
        return self.store[self.name][0]


class FakeBucket:
    def __init__(self, store):
        self.store = store

    def blob(self, name):
        return FakeBlob(self.store, name)


def test_gcs_blobs_use_the_receipt_bucket_prefix():
    store = {}
    client = SimpleNamespace(bucket=lambda name: FakeBucket(store))
    b = GcsBlobs("my-bucket", "salid-docs", client=client)
    uri = b.save("org1", "d1", "x.docx", b"bytes", "application/x")
    assert uri == "gs://my-bucket/salid-docs/org1/d1.docx"
    assert list(store) == ["salid-docs/org1/d1.docx"] and b.read(uri) == b"bytes"
    with pytest.raises(ValueError):
        b.read("gs://my-bucket/salid-receipts/1/abc.pdf")  # never the receipt objects
    with pytest.raises(ValueError):
        b.read("gs://other/salid-docs/org1/d1.docx")


def test_make_blobs_follows_receipt_bucket_setting(tmp_path):
    assert isinstance(
        make_blobs(Settings(receipt_bucket="", docs_local_dir=str(tmp_path))), LocalBlobs
    )
    g = make_blobs(Settings(receipt_bucket="bkt"))
    assert isinstance(g, GcsBlobs) and g.prefix == "salid-docs/" and g.bucket_name == "bkt"


def test_connectors_are_disabled_without_credentials():
    off = ConnectorConfig(base_url="https://hub.example")
    assert not SlackConnector(off).enabled() and not ClickUpConnector(off).enabled()
    partial = ConnectorConfig(slack_client_id="a", slack_client_secret="b")  # no signing secret
    assert not SlackConnector(partial).enabled()
    with pytest.raises(ConnectorError):
        SlackConnector(off).oauth_start("org")
    with pytest.raises(ConnectorError):
        ClickUpConnector(off).oauth_callback({"code": "x"})
    full = ConnectorConfig(slack_client_id="a", slack_client_secret="b", slack_signing_secret="c")
    assert SlackConnector(full).enabled()


def test_slack_callback_rejects_failures_without_leaking_secrets():
    cfg = ConnectorConfig(
        base_url="https://h",
        slack_client_id="a",
        slack_client_secret="TOPSECRET",
        slack_signing_secret="c",
    )

    def boom(request):
        raise httpx.ConnectError("down")

    c = SlackConnector(cfg, client=httpx.Client(transport=httpx.MockTransport(boom)))
    with pytest.raises(ConnectorError) as e:
        c.oauth_callback({"code": "x"})
    assert "TOPSECRET" not in str(e.value)
    with pytest.raises(ConnectorError):
        c.oauth_callback({"error": "access_denied"})


def test_clickup_never_writes():
    c = ClickUpConnector(ConnectorConfig(clickup_client_id="a", clickup_client_secret="b"))
    with pytest.raises(ConnectorError):
        c.send("tok", "chan", "hi")
    assert c.ingest({"x": 1}) == 0


class Perm(SimpleNamespace):
    pass


class Chan:
    def __init__(self, id, public):
        self.id, self._public = id, public

    def permissions_for(self, role):
        assert role == "everyone"
        return Perm(view_channel=self._public, read_message_history=self._public)


def test_public_channel_detection_needs_a_positive_answer():
    guild = SimpleNamespace(
        default_role="everyone",
        text_channels=[Chan(1, True), Chan(2, False)],
        voice_channels=[Chan(3, True)],
    )
    assert is_public(guild.text_channels[0], guild) and not is_public(guild.text_channels[1], guild)
    assert public_channel_ids(guild) == ["1", "3"]
    assert (
        public_channel_ids(SimpleNamespace(text_channels=[Chan(1, True)])) == []
    )  # no default role
    broken = SimpleNamespace(permissions_for=lambda r: 1 / 0)
    assert is_public(broken, guild) is False


def test_guild_snapshot_syncs_public_channels():
    from salid.bot import SecretaryBot

    class Ch(Chan):
        name = "c"

        def permissions_for(self, who):
            if who == "everyone":
                return Perm(view_channel=self.id == 1, read_message_history=self.id == 1)
            return Perm(view_channel=True, read_message_history=True)

    g = SimpleNamespace(
        name="G", icon=None, default_role="everyone", me=object(),
        text_channels=[Ch(1, True), Ch(2, False)],
    )  # fmt: skip
    snap = SecretaryBot._guild_snapshot(g)
    assert snap["public_channel_ids"] == ["1"]
    store = MemoryGuildStore()
    store.upsert_guild(5, **snap)
    assert store.get(5).public_channel_ids == ["1"]
    store.upsert_guild(
        5, name="G", icon_url="", channels=[]
    )  # an upsert without the field keeps it
    assert store.get(5).public_channel_ids == ["1"]


def test_peek_link_code_does_not_consume():
    store = MemoryGuildStore()
    store.upsert_guild(5, name="G", icon_url="", channels=[])
    code = store.create_link_code(5, 1, "u")
    assert store.peek_link_code(code) == 5 and store.peek_link_code(code) == 5
    store.redeem_link_code(code, "a@x.com")
    with pytest.raises(LinkError):
        store.peek_link_code(code)
    with pytest.raises(LinkError):
        store.peek_link_code("nope")


def test_note_text_is_what_salid_searches():
    rec = {
        "filename": "p.pdf",
        "tag": "invoice",
        "summary": "Pay  100",
        "text": "line one\nline two",
    }
    out = Library.note_text(rec)
    assert out.startswith("[ไฟล์: p.pdf, invoice] Pay 100 | ข้อความสำคัญ: line one line two")
    assert json.dumps(out)
