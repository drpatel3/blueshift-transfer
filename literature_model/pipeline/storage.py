"""
File I/O abstraction for local filesystem and S3.

Auto-detects Lambda environment via AWS_LAMBDA_FUNCTION_NAME env var.
LocalStorage is the default for development; S3Storage is used on Lambda.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalStorage:
    """Filesystem-based storage for local development."""

    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path(".")

    def read_pdf(self, path):
        """Read a PDF file and return bytes."""
        with open(path, "rb") as f:
            return f.read()

    def save_image(self, image_data, path):
        """Save image bytes to a local path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(image_data, "save"):
            # PIL Image or pdfplumber PageImage
            image_data.save(str(path))
        else:
            with open(path, "wb") as f:
                f.write(image_data)
        return str(path)

    def read_json(self, path):
        """Read and parse a JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    def write_json(self, data, path):
        """Write data as JSON to a local path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def exists(self, path):
        """Check if a file exists."""
        return Path(path).exists()

    def list_files(self, directory, pattern="*"):
        """List files matching a pattern in a directory."""
        return sorted(Path(directory).glob(pattern))


class S3Storage:
    """S3-based storage for Lambda deployment."""

    def __init__(self, bucket_name):
        import boto3
        self.bucket_name = bucket_name
        self.s3 = boto3.client("s3")

    def read_pdf(self, s3_key):
        """Download a PDF from S3 and return bytes."""
        response = self.s3.get_object(Bucket=self.bucket_name, Key=s3_key)
        return response["Body"].read()

    def save_image(self, image_data, s3_key):
        """Upload image bytes to S3."""
        if hasattr(image_data, "save"):
            # PIL Image — save to bytes buffer first
            import io
            buf = io.BytesIO()
            image_data.save(buf, format="PNG")
            image_bytes = buf.getvalue()
        else:
            image_bytes = image_data

        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=image_bytes,
            ContentType="image/png",
        )
        return f"s3://{self.bucket_name}/{s3_key}"

    def read_json(self, s3_key):
        """Read and parse a JSON file from S3."""
        response = self.s3.get_object(Bucket=self.bucket_name, Key=s3_key)
        return json.loads(response["Body"].read().decode("utf-8"))

    def write_json(self, data, s3_key):
        """Write data as JSON to S3."""
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=json.dumps(data, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    def exists(self, s3_key):
        """Check if an object exists in S3."""
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except self.s3.exceptions.ClientError:
            return False

    def list_files(self, prefix, pattern=None):
        """List objects under a prefix in S3."""
        paginator = self.s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if pattern is None or key.endswith(pattern.lstrip("*")):
                    keys.append(key)
        return sorted(keys)

    def write_bytes(self, data, s3_key, content_type="application/octet-stream"):
        """Upload raw bytes to S3."""
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=data,
            ContentType=content_type,
        )

    def download_to_tmp(self, s3_key):
        """Download an S3 object to /tmp and return the local path."""
        filename = os.path.basename(s3_key)
        local_path = f"/tmp/{filename}"
        self.s3.download_file(self.bucket_name, s3_key, local_path)
        return local_path


def get_storage(bucket_name=None):
    """
    Return the appropriate storage backend.

    Auto-detects Lambda via AWS_LAMBDA_FUNCTION_NAME env var.
    Falls back to LocalStorage for development.
    """
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        if bucket_name is None:
            bucket_name = os.environ.get("S3_BUCKET")
        if bucket_name is None:
            raise ValueError("S3_BUCKET environment variable required on Lambda")
        logger.info(f"Using S3Storage (bucket: {bucket_name})")
        return S3Storage(bucket_name)

    logger.info("Using LocalStorage")
    return LocalStorage()
