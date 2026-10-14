from __future__ import annotations

import logging
import time

from google import genai
from google.genai import errors

from .config import Settings

logger = logging.getLogger(__name__)
TRANSIENT = {408, 429, 500, 502, 503, 504}
RETRY_DELAYS = (2.0, 5.0, 10.0)


def make_client(s: Settings) -> genai.Client:
    return genai.Client(vertexai=True, project=s.gcp_project, location=s.gemini_location)


def generate(client, model: str, contents, config, delays=RETRY_DELAYS):
    """generate_content with retry on transient Vertex errors (429/5xx)."""
    for attempt, delay in enumerate((*delays, None), start=1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as e:
            if e.code not in TRANSIENT or delay is None:
                raise
            logger.warning("gemini transient %s, retry %d in %.0fs", e.code, attempt, delay)
            time.sleep(delay)
