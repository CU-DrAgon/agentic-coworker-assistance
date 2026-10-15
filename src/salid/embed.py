"""Text embeddings for meaning-based memory search (see salid.memory_index and qa.chat_search).

Vectors are L2-normalised float32 `array("f")`, so cosine similarity is a plain dot product
(`math.sumprod`, no numpy). They are stored packed (little-endian float32) on the day summary
they belong to. `NullEmbedder` (empty `SALID_EMBED_MODEL`, or no client) turns meaning-based search
off: callers then use keywords only. Blocking; run from worker threads. Only counts are logged.
"""

from __future__ import annotations

import math
import sys
from array import array
from collections.abc import Sequence
from typing import Protocol

from google.genai import types

from . import llm

EMBED_BATCH = 1  # texts per embed_content call (raised to the API limit measured by the probe)
TASKS = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}


class Embedder(Protocol):
    model_id: str  # "<model>@<dim>"; "" = embeddings off
    dim: int

    def embed(self, texts: list[str], task: str) -> list[array]:
        """One normalised vector per text; `task` is "document" or "query"."""
        ...


def normalize(values: Sequence[float]) -> array:
    """`values` scaled to unit length as float32 (a zero vector stays zero)."""
    norm = math.sqrt(math.sumprod(values, values)) or 1.0
    return array("f", (v / norm for v in values))


def cosine(a: array, b: array) -> float:
    """Cosine similarity of two normalised vectors of the same length."""
    return math.sumprod(a, b)


def pack(vecs: list[array]) -> bytes:
    out = array("f")
    for v in vecs:
        out.extend(v)
    if sys.byteorder == "big":
        out.byteswap()
    return out.tobytes()


def unpack(data: bytes, dim: int) -> list[array]:
    flat = array("f")
    flat.frombytes(data)
    if sys.byteorder == "big":
        flat.byteswap()
    return [flat[i : i + dim] for i in range(0, len(flat), dim)] if dim else []


class NullEmbedder:
    model_id = ""
    dim = 0

    def embed(self, texts: list[str], task: str) -> list[array]:
        return []


class VertexEmbedder:
    def __init__(self, client, model: str, dim: int, batch: int = EMBED_BATCH):
        self.client, self.model, self.dim, self.batch = client, model, dim, max(1, batch)
        self.model_id = f"{model}@{dim}"

    def embed(self, texts: list[str], task: str) -> list[array]:
        config = types.EmbedContentConfig(task_type=TASKS[task], output_dimensionality=self.dim)
        out: list[array] = []
        for i in range(0, len(texts), self.batch):
            part = texts[i : i + self.batch]
            resp = llm.embed(self.client, self.model, part, config)
            got = list(getattr(resp, "embeddings", None) or [])
            if len(got) != len(part):
                raise ValueError(f"expected {len(part)} embeddings, got {len(got)}")
            out += [normalize(e.values) for e in got]
        return out


def make_embedder(settings, client) -> Embedder:
    """The configured embedder; `NullEmbedder` when `SALID_EMBED_MODEL` is empty or there is no
    client (tests, offline runs). `SALID_EMBED_LOCATION` gets its own client when it differs."""
    if not settings.embed_model or client is None:
        return NullEmbedder()
    loc = settings.embed_location
    if loc and loc != settings.gemini_location:
        from google import genai

        client = genai.Client(vertexai=True, project=settings.gcp_project, location=loc)
    return VertexEmbedder(client, settings.embed_model, settings.embed_dim)
