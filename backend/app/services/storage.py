"""
S3 storage service for Meetolog audio files.

Uses ``aioboto3`` for fully async S3 interactions with streaming
multipart uploads (memory-safe for large files), presigned POST generation
for direct browser-to-S3 uploads, and exponential-backoff retry logic via
``tenacity``.
"""

import logging
from pathlib import Path
from typing import IO
from uuid import uuid4

import aioboto3
import structlog
from botocore.exceptions import ClientError
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import get_settings

logger = structlog.get_logger(__name__)

# Transient AWS errors worth retrying.
_RETRYABLE = (ClientError, ConnectionError, TimeoutError)

_ALLOWED_AUDIO_MIME_TYPES: frozenset[str] = frozenset(
    {
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/ogg",
        "audio/webm",
        "audio/aac",
        "audio/flac",
    }
)
_PRESIGNED_POST_MAX_BYTES: int = 1 * 1024 * 1024 * 1024  # 1 GiB
_PRESIGNED_POST_TTL_SECONDS: int = 900  # 15 minutes


class S3StorageService:
    """Async S3 client wrapper with retry and streaming support."""

    def __init__(
        self,
        bucket: str | None = None,
        region: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self._bucket = bucket or settings.aws_s3_bucket
        self._region = region or settings.aws_region
        self._aws_access_key_id = aws_access_key_id or settings.aws_access_key_id
        self._aws_secret_access_key = (
            aws_secret_access_key or settings.aws_secret_access_key
        )
        self._endpoint_url = endpoint_url or settings.aws_endpoint_url
        self._public_endpoint_url = settings.aws_public_endpoint_url
        self._session = aioboto3.Session(
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=self._aws_secret_access_key,
            region_name=self._region,
        )

    def _client_ctx(self):
        """Return an async context-manager for an S3 client."""
        kwargs: dict = {"service_name": "s3"}
        if self._endpoint_url:
            kwargs["endpoint_url"] = self._endpoint_url
        return self._session.client(**kwargs)

    async def generate_upload_presigned_post(
        self,
        filename: str,
        file_type: str,
        file_size: int,
    ) -> dict:
        """Generate presigned POST data for a direct browser-to-S3 upload.

        Validates the MIME type and declared file size before signing.  AWS
        enforces both constraints at the network edge via embedded policy
        conditions, so the API server never touches the file bytes.

        Args:
            filename: Client-supplied filename; used as the S3 key suffix
                      after stripping any path components.
            file_type: MIME type declared by the client (e.g. ``audio/mpeg``).
            file_size: Declared byte length of the file.  Must be in the range
                       ``[1, _PRESIGNED_POST_MAX_BYTES]``.

        Returns:
            Mapping with ``url`` (str), ``fields`` (dict[str, str]), and
            ``s3_key`` (str).

        Raises:
            ValueError: When the MIME type or file size fails validation.
            ClientError: When AWS credential or service errors prevent signing.
        """
        if file_type not in _ALLOWED_AUDIO_MIME_TYPES:
            raise ValueError(f"Unsupported MIME type: {file_type!r}")
        if not (1 <= file_size <= _PRESIGNED_POST_MAX_BYTES):
            raise ValueError(
                f"Declared file size {file_size} B is outside the allowed range "
                f"(1 – {_PRESIGNED_POST_MAX_BYTES} bytes)."
            )

        safe_name = Path(filename).name  # strip directory traversal components
        s3_key = f"uploads/{uuid4()}/{safe_name}"

        try:
            async with self._client_ctx() as s3:
                presigned = await s3.generate_presigned_post(
                    Bucket=self._bucket,
                    Key=s3_key,
                    Fields={"Content-Type": file_type},
                    Conditions=[
                        {"Content-Type": file_type},
                        ["content-length-range", 1, _PRESIGNED_POST_MAX_BYTES],
                    ],
                    ExpiresIn=_PRESIGNED_POST_TTL_SECONDS,
                )
        except ClientError as exc:
            logger.error(
                "Failed to generate presigned POST for key %s: %s", s3_key, exc
            )
            raise

        url = presigned["url"]
        if self._public_endpoint_url and self._endpoint_url:
            url = url.replace(self._endpoint_url, self._public_endpoint_url, 1)

        return {
            "url": url,
            "fields": presigned["fields"],
            "s3_key": s3_key,
        }

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def upload_stream(
        self,
        file_stream: IO[bytes],
        object_key: str,
    ) -> str:
        """Stream-upload a file-like object to S3 using multipart upload.

        The ``upload_fileobj`` call internally uses multipart upload for
        large payloads so the entire file never needs to reside in RAM.

        Args:
            file_stream: Readable binary file-like object (e.g. ``UploadFile.file``).
            object_key: Destination key inside the configured S3 bucket.

        Returns:
            The S3 URI of the uploaded object (``s3://<bucket>/<key>``).

        Raises:
            ClientError: After all retry attempts are exhausted.
        """
        logger.info("Uploading object %s to s3://%s", object_key, self._bucket)
        try:
            async with self._client_ctx() as s3:
                await s3.upload_fileobj(
                    file_stream,
                    self._bucket,
                    object_key,
                    Config=self._transfer_config(),
                )
            s3_uri = f"s3://{self._bucket}/{object_key}"
            logger.info("Upload complete: %s", s3_uri)
            return s3_uri
        except RetryError:
            logger.error(
                "S3 upload failed after retries: bucket=%s key=%s",
                self._bucket,
                object_key,
            )
            raise
        except _RETRYABLE as exc:
            logger.warning(
                "Transient S3 error during upload (will retry): %s", exc
            )
            raise

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def download_to_file(
        self,
        object_key: str,
        destination_path: str,
    ) -> None:
        """Download an S3 object to a local file path.

        Args:
            object_key: Key of the object in the configured S3 bucket.
            destination_path: Local filesystem path to write to.

        Raises:
            ClientError: After all retry attempts are exhausted.
        """
        logger.info(
            "Downloading s3://%s/%s -> %s",
            self._bucket,
            object_key,
            destination_path,
        )
        try:
            async with self._client_ctx() as s3:
                await s3.download_file(
                    self._bucket,
                    object_key,
                    destination_path,
                )
            logger.info("Download complete: %s", destination_path)
        except RetryError:
            logger.error(
                "S3 download failed after retries: bucket=%s key=%s",
                self._bucket,
                object_key,
            )
            raise
        except _RETRYABLE as exc:
            logger.warning(
                "Transient S3 error during download (will retry): %s", exc
            )
            raise

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def delete_object(self, object_key: str) -> None:
        """Delete an object from S3.

        Args:
            object_key: Key of the object to delete.
        """
        logger.info("Deleting s3://%s/%s", self._bucket, object_key)
        async with self._client_ctx() as s3:
            await s3.delete_object(Bucket=self._bucket, Key=object_key)
        logger.info("Deleted s3://%s/%s", self._bucket, object_key)

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def upload_pdf(self, file_path: str, job_id: str) -> str:
        """Upload a generated PDF report to S3.

        Args:
            file_path: Local path to the PDF file.
            job_id: Job identifier used to derive the S3 key.

        Returns:
            The S3 object key where the PDF was stored.
        """
        s3_key = f"results/{job_id}/meeting_{job_id}.pdf"
        logger.info("Uploading PDF for job %s to s3://%s/%s", job_id, self._bucket, s3_key)
        async with self._client_ctx() as s3:
            await s3.upload_file(
                file_path,
                self._bucket,
                s3_key,
                ExtraArgs={"ContentType": "application/pdf"},
            )
        logger.info("PDF upload complete: %s", s3_key)
        return s3_key

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def upload_artifacts_json(self, artifacts: dict, job_id: str) -> str:
        """Serialize and upload extracted artifacts as JSON to S3.

        Args:
            artifacts: Artifact dictionary (already JSON-serializable).
            job_id: Job identifier used to derive the S3 key.

        Returns:
            The S3 object key where the JSON was stored.
        """
        import json

        s3_key = f"results/{job_id}/artifacts.json"
        body = json.dumps(artifacts, ensure_ascii=False, indent=2).encode("utf-8")
        logger.info("Uploading artifacts JSON for job %s to s3://%s/%s", job_id, self._bucket, s3_key)
        async with self._client_ctx() as s3:
            await s3.put_object(
                Bucket=self._bucket,
                Key=s3_key,
                Body=body,
                ContentType="application/json",
            )
        logger.info("Artifacts JSON upload complete: %s", s3_key)
        return s3_key

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def generate_presigned_get_url(self, object_key: str, expires_in: int = 3600) -> str:
        """Generate a presigned GET URL for downloading an S3 object.

        Args:
            object_key: Key of the object in the configured S3 bucket.
            expires_in: URL validity in seconds (default 1 hour).

        Returns:
            Presigned URL string.
        """
        async with self._client_ctx() as s3:
            url = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": object_key},
                ExpiresIn=expires_in,
            )
        return url

    @staticmethod
    def _transfer_config():
        """Return a boto3 TransferConfig tuned for large audio files."""
        from boto3.s3.transfer import TransferConfig

        return TransferConfig(
            multipart_threshold=8 * 1024 * 1024,   # 8 MB
            multipart_chunksize=8 * 1024 * 1024,    # 8 MB
            max_concurrency=4,
        )
