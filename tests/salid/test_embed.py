"""Embeddings for memory search: Vertex embedder (batching, task types, retry), the kill switch,
vector packing and cosine. Offline: a fake genai client."""

from types import SimpleNamespace

import pytest
from google.genai import errors

from salid.config import Settings
from salid.embed import (
    NullEmbedder,
    VertexEmbedder,
    cosine,
    make_embedder,
    normalize,
    pack,
    unpack,
)


class FakeEmbedClient:
    def __init__(self, *script):
        self.script, self.calls = list(script), []
        self.models = self

    def embed_content(self, model, contents, config):
        self.calls.append((model, list(contents), config))
        item = self.script.pop(0) if self.script else None
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0]) for _ in contents])


def test_vertex_embedder_batches_and_sets_task_and_dim():
    c = FakeEmbedClient()
    vecs = VertexEmbedder(c, "m", 2, batch=2).embed(["a", "b", "c", "d", "e"], "document")
    assert [len(x[1]) for x in c.calls] == [2, 2, 1] and len(vecs) == 5
    cfg = c.calls[0][2]
    assert cfg.task_type == "RETRIEVAL_DOCUMENT" and cfg.output_dimensionality == 2
    assert vecs[0].typecode == "f" and abs(cosine(vecs[0], vecs[0]) - 1.0) < 1e-6  # normalised


def test_query_task_and_model_id():
    c = FakeEmbedClient()
    e = VertexEmbedder(c, "gemini-embedding-001", 2)
    e.embed(["q"], "query")
    assert c.calls[0][2].task_type == "RETRIEVAL_QUERY" and e.model_id == "gemini-embedding-001@2"


def test_transient_errors_are_retried(monkeypatch):
    monkeypatch.setattr("salid.llm.RETRY_DELAYS", (0.0,))
    c = FakeEmbedClient(errors.ClientError(429, {"error": {"message": "quota"}}))
    assert len(VertexEmbedder(c, "m", 2).embed(["a"], "document")) == 1 and len(c.calls) == 2


def test_vertex_embedder_rejects_a_short_response():
    c = FakeEmbedClient()
    c.embed_content = lambda model, contents, config: SimpleNamespace(embeddings=[])
    with pytest.raises(ValueError):
        VertexEmbedder(c, "m", 2).embed(["a"], "document")


def test_kill_switch_and_missing_client_give_the_null_embedder():
    assert isinstance(make_embedder(Settings(embed_model=""), object()), NullEmbedder)
    assert isinstance(make_embedder(Settings(), None), NullEmbedder)
    assert NullEmbedder().embed(["a"], "query") == [] and NullEmbedder().model_id == ""


def test_pack_roundtrip():
    vs = [normalize([1.0, 2.0]), normalize([0.0, 1.0])]
    back = unpack(pack(vs), 2)
    assert len(pack(vs)) == 16 and [list(v) for v in back] == [list(v) for v in vs]


def test_query_embeddings_fail_fast_without_retry(monkeypatch):
    monkeypatch.setattr("salid.llm.RETRY_DELAYS", (0.0,))
    c = FakeEmbedClient(errors.ClientError(429, {"error": {"message": "quota"}}))
    with pytest.raises(errors.ClientError):
        VertexEmbedder(c, "m", 2).embed(["q"], "query")
    assert len(c.calls) == 1  # an answer is waiting: no retry, the caller falls back to keywords
