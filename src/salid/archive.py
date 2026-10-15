"""Optional archive of original receipt files in Cloud Storage (`SALID_RECEIPT_BUCKET`)."""

from __future__ import annotations

import logging
from typing import Any

from .config import Settings

logger = logging.getLogger(__name__)
EXT_BY_MIME = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/heic": "heic",
    "image/heif": "heif",
    "application/pdf": "pdf",
}


class ReceiptArchive:
    """Stores original bytes at `gs://{bucket}/{prefix}{guild_id}/{sha256}.{ext}` (per guild)."""

    def __init__(
        self, bucket: str, prefix: str = "salid-receipts/", client: Any = None, project=None
    ):
        self.bucket_name = bucket
        self.prefix = prefix if prefix.endswith("/") or not prefix else prefix + "/"
        self._client, self._project = client, project

    @property
    def client(self) -> Any:
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client(project=self._project)
        return self._client

    def object_name(self, guild_id: int, sha256: str, mime: str) -> str:
        return f"{self.prefix}{guild_id}/{sha256}.{EXT_BY_MIME.get(mime, 'bin')}"

    def save(self, guild_id: int, sha256: str, data: bytes, mime: str) -> str:
        name = self.object_name(guild_id, sha256, mime)
        self.client.bucket(self.bucket_name).blob(name).upload_from_string(data, content_type=mime)
        return f"gs://{self.bucket_name}/{name}"

    def delete(self, uri: str) -> None:
        prefix = f"gs://{self.bucket_name}/"
        if not uri.startswith(prefix):
            raise ValueError("uri is not in this archive's bucket")
        self.client.bucket(self.bucket_name).blob(uri[len(prefix) :]).delete()


def make_archive(settings: Settings) -> ReceiptArchive | None:
    if not settings.receipt_bucket:
        return None
    return ReceiptArchive(
        settings.receipt_bucket, settings.receipt_prefix, project=settings.gcp_project
    )
