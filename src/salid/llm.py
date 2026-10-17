from __future__ import annotations

import logging
import time

import httpx
from google import genai
from google.genai import errors, types

from .config import Settings

logger = logging.getLogger(__name__)
TRANSIENT = {408, 429, 500, 502, 503, 504}
RETRY_DELAYS = (2.0, 5.0, 10.0)
# Without a timeout one stuck Vertex call blocks forever (seen live: a voice-meeting
# transcription waited 3+ minutes for headers and froze the meeting's worker).
REQUEST_TIMEOUT_MS = 60_000


def make_client(s: Settings) -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=s.gcp_project,
        location=s.gemini_location,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )


def media_http_options(**kw) -> types.HttpOptions:
    """HttpOptions for paid media creates (Omni hero clips, Lyria music): the request timeout
    and no SDK retry on 5xx / 408 / 409, because a retry after the service accepted the job would
    start (and bill) a second one. Only 429 (refused before any work) is retried, once:
    `attempts=1` is the fewest the SDK allows (0 is rewritten to 1). The SDK's Interactions
    client still retries a connection error or timeout once; that cannot be switched off here.
    `kw` goes to HttpOptions (tests pass an `httpx_client`)."""
    return types.HttpOptions(
        timeout=REQUEST_TIMEOUT_MS,
        retry_options=types.HttpRetryOptions(attempts=1, http_status_codes=[429]),
        **kw,
    )


def make_media_client(s: Settings) -> genai.Client:
    """The client for paid media creates (see `media_http_options`); polls and everything else
    use `make_client`."""
    return genai.Client(
        vertexai=True,
        project=s.gcp_project,
        location=s.gemini_location,
        http_options=media_http_options(),
    )


def with_retry(fn, delays=None):
    """Call `fn()`, retrying transient Vertex errors (429/5xx) after each delay in `delays`
    (None = `RETRY_DELAYS`, read at call time)."""
    delays = RETRY_DELAYS if delays is None else delays
    for attempt, delay in enumerate((*delays, None), start=1):
        try:
            return fn()
        except errors.APIError as e:
            if e.code not in TRANSIENT or delay is None:
                raise
            logger.warning("gemini transient %s, retry %d in %.0fs", e.code, attempt, delay)
            time.sleep(delay)
        except httpx.TimeoutException:
            if delay is None:
                raise
            logger.warning("gemini request timed out, retry %d in %.0fs", attempt, delay)
            time.sleep(delay)


def generate(client, model: str, contents, config, delays=RETRY_DELAYS):
    """generate_content with retry on transient Vertex errors (429/5xx)."""
    return with_retry(
        lambda: client.models.generate_content(model=model, contents=contents, config=config),
        delays,
    )


def embed(client, model: str, contents, config, delays=None):
    """embed_content with the same retry policy as `generate`."""
    return with_retry(
        lambda: client.models.embed_content(model=model, contents=contents, config=config), delays
    )
