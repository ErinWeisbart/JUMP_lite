"""Plate-level downloader for Cell Painting Gallery data from S3."""

import logging
import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

from .download_cache import DownloadCache

logger = logging.getLogger(__name__)


class PlateDownloader:
    """
    Download entire plates from S3 with caching and resume support.

    Uses anonymous S3 access (--no-sign-request) to download from the
    Cell Painting Gallery public dataset.
    """

    def __init__(
        self,
        cache_dir: Path,
        bucket: str = "cellpainting-gallery",
        use_aws_cli: bool = True,
        region: str = "us-east-1",
    ):
        """
        Initialize plate downloader.

        Args:
            cache_dir: Directory for caching downloaded plates
            bucket: S3 bucket name (default: cellpainting-gallery)
            use_aws_cli: Use AWS CLI for faster downloads (default: True)
            region: AWS region (default: us-east-1)
        """
        self.cache_dir = Path(cache_dir)
        self.bucket = bucket
        self.use_aws_cli = use_aws_cli
        self.region = region

        # Initialize download cache
        self.cache = DownloadCache(cache_dir)

        # Initialize anonymous S3 client
        self.s3_client = boto3.client(
            "s3",
            region_name=region,
            config=Config(signature_version=UNSIGNED),  # No-sign-request
        )

        # Check if AWS CLI is available
        if self.use_aws_cli and not self._check_aws_cli():
            logger.warning("AWS CLI not found, falling back to boto3")
            self.use_aws_cli = False

    def _check_aws_cli(self) -> bool:
        """Check if AWS CLI is installed and available."""
        try:
            result = subprocess.run(
                ["aws", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def download_plate(
        self,
        plate_id: str,
        s3_prefix: str,
        force: bool = False,
    ) -> Path:
        """
        Download an entire plate from S3.

        Args:
            plate_id: Plate identifier
            s3_prefix: S3 prefix path (e.g., "cpg0016-jump/source_1/images/batch/plate/")
            force: Force re-download even if already cached

        Returns:
            Path to downloaded plate cache directory

        Raises:
            RuntimeError: If download fails
        """
        # Check if already downloaded
        if not force and self.cache.is_plate_downloaded(plate_id):
            logger.info(f"Plate {plate_id} already downloaded, skipping")
            return self.cache.get_plate_cache_dir(plate_id)

        # Get cache directory for this plate
        plate_dir = self.cache.get_plate_cache_dir(plate_id)
        plate_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading plate {plate_id} from s3://{self.bucket}/{s3_prefix}")

        try:
            if self.use_aws_cli:
                num_files, size_bytes = self._download_with_aws_cli(
                    s3_prefix, plate_dir
                )
            else:
                num_files, size_bytes = self._download_with_boto3(s3_prefix, plate_dir)

            # Mark as complete
            self.cache.mark_plate_complete(
                plate_id,
                num_files=num_files,
                size_bytes=size_bytes,
                s3_prefix=s3_prefix,
            )

            logger.info(
                f"Successfully downloaded plate {plate_id}: "
                f"{num_files} files, {size_bytes / (1024**3):.2f} GB"
            )

            return plate_dir

        except Exception as e:
            logger.error(f"Failed to download plate {plate_id}: {e}")
            # Clean up incomplete download
            if plate_dir.exists():
                import shutil

                shutil.rmtree(plate_dir)
            raise RuntimeError(f"Failed to download plate {plate_id}") from e

    def _download_with_aws_cli(
        self, s3_prefix: str, dest_dir: Path
    ) -> tuple[int, int]:
        """
        Download using AWS CLI (fastest method).

        Args:
            s3_prefix: S3 prefix to download
            dest_dir: Destination directory

        Returns:
            Tuple of (num_files, total_size_bytes)

        Raises:
            RuntimeError: If AWS CLI command fails
        """
        s3_uri = f"s3://{self.bucket}/{s3_prefix}"

        logger.info(f"Using AWS CLI to download from {s3_uri}")

        # Build AWS CLI command (using cp --recursive for read-only download)
        cmd = [
            "aws",
            "s3",
            "cp",
            s3_uri,
            str(dest_dir),
            "--recursive",
            "--no-sign-request",
            "--region",
            self.region,
        ]

        logger.debug(f"Running command: {' '.join(cmd)}")

        # Run AWS CLI with progress tracking
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Show progress
            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.debug(line)

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"AWS CLI command failed with code {process.returncode}")

            # Count downloaded files and size
            num_files = sum(1 for _ in dest_dir.rglob("*.tif"))
            size_bytes = sum(f.stat().st_size for f in dest_dir.rglob("*.tif"))

            return num_files, size_bytes

        except subprocess.SubprocessError as e:
            raise RuntimeError(f"AWS CLI execution failed: {e}") from e

    def _download_with_boto3(
        self, s3_prefix: str, dest_dir: Path
    ) -> tuple[int, int]:
        """
        Download using boto3 (fallback method).

        Args:
            s3_prefix: S3 prefix to download
            dest_dir: Destination directory

        Returns:
            Tuple of (num_files, total_size_bytes)

        Raises:
            RuntimeError: If download fails
        """
        logger.info(f"Using boto3 to download from s3://{self.bucket}/{s3_prefix}")

        # List all objects with the prefix
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix)

            # Collect all TIFF files
            tiff_files = []
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if key.lower().endswith(".tif") or key.lower().endswith(".tiff"):
                            tiff_files.append((key, obj["Size"]))

            if not tiff_files:
                raise RuntimeError(f"No TIFF files found at {s3_prefix}")

            logger.info(f"Found {len(tiff_files)} TIFF files to download")

            # Download files with progress bar
            total_size = 0
            with tqdm(total=len(tiff_files), desc="Downloading files") as pbar:
                for key, size in tiff_files:
                    # Determine local path
                    # Remove prefix from key to get relative path
                    rel_path = key[len(s3_prefix):].lstrip("/")
                    local_path = dest_dir / rel_path
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    # Download file
                    try:
                        self.s3_client.download_file(
                            self.bucket, key, str(local_path)
                        )
                        total_size += size
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Failed to download {key}: {e}")
                        raise

            return len(tiff_files), total_size

        except Exception as e:
            raise RuntimeError(f"boto3 download failed: {e}") from e

    def get_s3_prefix_from_url(self, s3_url: str) -> str:
        """
        Extract S3 prefix from an S3 URL.

        Args:
            s3_url: S3 URL (e.g., "s3://bucket/prefix/file.tif")

        Returns:
            S3 prefix (e.g., "prefix/")

        Raises:
            ValueError: If URL is not a valid S3 URL
        """
        parsed = urlparse(s3_url)
        if parsed.scheme != "s3":
            raise ValueError(f"Not an S3 URL: {s3_url}")

        # Extract path and get parent directory
        path = Path(parsed.path.lstrip("/"))
        prefix = str(path.parent) + "/"

        return prefix

    def list_downloaded_plates(self) -> list[str]:
        """
        List all downloaded plates.

        Returns:
            List of plate IDs
        """
        return self.cache.list_plates()

    def get_cache_summary(self) -> dict:
        """
        Get summary of download cache.

        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_summary()

    def cleanup_incomplete(self):
        """Clean up incomplete downloads."""
        self.cache.cleanup_incomplete_downloads()
