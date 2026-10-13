"""Audio chunk storage in Google Cloud Storage."""

import logging
from functools import lru_cache

from google.cloud import storage

from .config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def _client() -> storage.Client:
    return storage.Client(project=get_settings().gcp_project)


def chunk_object_name(meeting_id: str, index: int) -> str:
    return f"meetings/{meeting_id}/chunk-{index:04d}.flac"


def upload_chunk(
    meeting_id: str, index: int, data: bytes, content_type: str = "audio/flac"
) -> str:
    """Upload one audio chunk and return its `gs://` URI."""
    bucket_name = get_settings().audio_bucket
    name = chunk_object_name(meeting_id, index)
    blob = _client().bucket(bucket_name).blob(name)
    blob.upload_from_string(data, content_type=content_type)
    logger.info("Uploaded chunk %d of meeting %s (%d bytes)", index, meeting_id, len(data))
    return f"gs://{bucket_name}/{name}"
