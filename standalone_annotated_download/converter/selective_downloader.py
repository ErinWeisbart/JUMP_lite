"""
Selective downloader for downloading specific wells from Cell Painting plates.

This module provides optimized downloading of plate subsets based on metadata
filtering, using two strategies:

- Strategy A (>50% wells): Download full plate with AWS CLI (fastest)
- Strategy B (<50% wells): Selective file download with boto3 (saves bandwidth)
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

from .download_cache import DownloadCache
from .load_data_parser import LoadDataParser
from .plate_downloader import PlateDownloader

logger = logging.getLogger(__name__)


class SelectiveWellDownloader:
    """
    Download specific wells from Cell Painting plates with optimized strategies.

    Features:
    - Two-strategy approach: full plate (>50% wells) vs selective (≤50% wells)
    - Parallel downloads with connection pooling (32 workers)
    - Batch downloads by site for S3 cache locality
    - Resume support (skip existing files)
    - Exponential backoff retry logic
    - Progress tracking at plate and file level
    """

    # S3 configuration
    S3_BUCKET = "cellpainting-gallery"
    S3_PROJECT = "cpg0016-jump"

    # Download configuration
    MAX_WORKERS = 32  # Connection pool size for boto3
    BATCH_SIZE_SITES = 100  # Sites to batch together
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2.0  # Exponential backoff multiplier

    # Strategy threshold
    FULL_PLATE_THRESHOLD = 0.5  # Download full plate if >50% wells needed

    def __init__(
        self,
        cache_dir: Path,
        region: str = "us-east-1",
        max_workers: Optional[int] = None,
    ):
        """
        Initialize selective downloader.

        Args:
            cache_dir: Directory for caching downloaded files
            region: AWS region
            max_workers: Number of parallel download workers (default: 32)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.region = region
        self.max_workers = max_workers or self.MAX_WORKERS

        # Initialize cache manager
        self.cache = DownloadCache(cache_dir)

        # Initialize full plate downloader (for Strategy A)
        self.plate_downloader = PlateDownloader(
            cache_dir=cache_dir,
            bucket=self.S3_BUCKET,
            use_aws_cli=True,
            region=region,
        )

        # Initialize boto3 S3 client with connection pooling (for Strategy B)
        self.s3_client = boto3.client(
            "s3",
            region_name=region,
            config=Config(
                signature_version=UNSIGNED,  # Anonymous access
                max_pool_connections=self.max_workers,  # Connection pool
            ),
        )

        logger.info(
            f"Initialized SelectiveWellDownloader with {self.max_workers} workers"
        )

    def download_wells(
        self,
        wells_manifest: pd.DataFrame,
        force: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Path]:
        """
        Download specific wells from multiple plates.

        Args:
            wells_manifest: DataFrame with columns:
                - Metadata_Source: Source identifier
                - Metadata_Batch: Batch identifier
                - Metadata_Plate: Plate identifier
                - Metadata_Well: Well identifier (e.g., 'A01')
                Optional:
                - Metadata_Site: Site number (default: all 6 sites)
            force: Force re-download even if files exist
            show_progress: Show progress bars

        Returns:
            Dictionary mapping plate_id -> local cache directory

        Example:
            >>> downloader = SelectiveWellDownloader(cache_dir='./data')
            >>> # Download wells from metadata filter
            >>> wells_df = filter.filter_by_compounds(['JCP2022_033924'])
            >>> plate_dirs = downloader.download_wells(wells_df)
            >>> print(f"Downloaded to: {plate_dirs}")
        """
        # Group wells by plate
        plate_groups = wells_manifest.groupby("Metadata_Plate")

        logger.info(
            f"Downloading {len(wells_manifest)} wells from {len(plate_groups)} plates"
        )

        downloaded_plates = {}

        # Process each plate
        for plate_id, plate_wells in tqdm(
            plate_groups,
            desc="Downloading plates",
            disable=not show_progress,
        ):
            try:
                plate_info = plate_wells.iloc[0]
                source = plate_info["Metadata_Source"]
                batch = plate_info["Metadata_Batch"]

                # Get unique wells for this plate
                wells = plate_wells["Metadata_Well"].unique().tolist()

                # Download plate with optimal strategy
                plate_dir = self._download_plate_wells(
                    source=source,
                    batch=batch,
                    plate_id=plate_id,
                    wells=wells,
                    force=force,
                    show_progress=show_progress,
                )

                downloaded_plates[plate_id] = plate_dir

            except Exception as e:
                logger.error(f"Failed to download plate {plate_id}: {e}")
                continue

        logger.info(f"Successfully downloaded {len(downloaded_plates)} plates")
        return downloaded_plates

    def _download_plate_wells(
        self,
        source: str,
        batch: str,
        plate_id: str,
        wells: List[str],
        force: bool = False,
        show_progress: bool = True,
    ) -> Path:
        """
        Download specific wells from a single plate using optimal strategy.

        Args:
            source: Source identifier
            batch: Batch identifier
            plate_id: Plate identifier
            wells: List of well identifiers
            force: Force re-download
            show_progress: Show progress

        Returns:
            Path to local plate cache directory
        """
        # Estimate total wells in plate (384-well format: 16 rows × 24 cols)
        total_wells = 384
        wells_needed = len(wells)
        wells_fraction = wells_needed / total_wells

        logger.info(
            f"Plate {plate_id}: requesting {wells_needed}/{total_wells} wells "
            f"({wells_fraction*100:.1f}%)"
        )

        # Choose strategy based on well coverage
        if wells_fraction > self.FULL_PLATE_THRESHOLD:
            # Strategy A: Download full plate with AWS CLI
            logger.info(f"Using Strategy A (full plate download) for {plate_id}")
            return self._download_full_plate(source, batch, plate_id, force)
        else:
            # Strategy B: Selective file download with boto3
            logger.info(
                f"Using Strategy B (selective download) for {plate_id}: "
                f"{wells_needed} wells"
            )
            return self._download_selective_wells(
                source, batch, plate_id, wells, force, show_progress
            )

    def _download_full_plate(
        self,
        source: str,
        batch: str,
        plate_id: str,
        force: bool = False,
    ) -> Path:
        """
        Download full plate using AWS CLI (Strategy A).

        Args:
            source: Source identifier
            batch: Batch identifier
            plate_id: Plate identifier
            force: Force re-download

        Returns:
            Path to plate cache directory
        """
        # Construct S3 prefix for plate images
        # Format: cpg0016-jump/source_1/images/batch/plate/
        s3_prefix = f"{self.S3_PROJECT}/{source}/images/{batch}/{plate_id}/"

        # Use PlateDownloader for full plate
        return self.plate_downloader.download_plate(
            plate_id=plate_id,
            s3_prefix=s3_prefix,
            force=force,
        )

    def _download_selective_wells(
        self,
        source: str,
        batch: str,
        plate_id: str,
        wells: List[str],
        force: bool = False,
        show_progress: bool = True,
    ) -> Path:
        """
        Download only specific wells using boto3 (Strategy B).

        Args:
            source: Source identifier
            batch: Batch identifier
            plate_id: Plate identifier
            wells: List of well identifiers
            force: Force re-download
            show_progress: Show progress

        Returns:
            Path to plate cache directory
        """
        # Get plate cache directory
        plate_dir = self.cache.get_plate_cache_dir(plate_id)
        plate_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Download and parse load_data.csv to get file mappings
        load_data_csv_path = self._get_load_data_csv_path(source, batch, plate_id)
        load_data_df = self._fetch_load_data_csv(load_data_csv_path, plate_dir)

        if load_data_df is None:
            raise RuntimeError(f"Failed to fetch load_data.csv for plate {plate_id}")

        # Step 2: Filter to only files for requested wells
        well_files = self._filter_files_by_wells(load_data_df, wells)

        if not well_files:
            raise RuntimeError(f"No files found for wells {wells} in plate {plate_id}")

        logger.info(f"Found {len(well_files)} files to download for {len(wells)} wells")

        # Step 3: Filter out already-downloaded files (resume support)
        if not force:
            files_to_download = []
            for s3_url, local_rel_path in well_files:
                local_path = plate_dir / local_rel_path
                if not local_path.exists():
                    files_to_download.append((s3_url, local_rel_path))
                else:
                    logger.debug(f"Skipping existing file: {local_rel_path}")
        else:
            files_to_download = well_files

        if not files_to_download:
            logger.info(f"All files already downloaded for plate {plate_id}")
            return plate_dir

        logger.info(
            f"Downloading {len(files_to_download)}/{len(well_files)} files "
            f"({len(files_to_download) / len(well_files) * 100:.1f}% remaining)"
        )

        # Step 4: Download files in parallel with batching
        downloaded_files = self._parallel_download_files(
            files_to_download=files_to_download,
            plate_dir=plate_dir,
            show_progress=show_progress,
        )

        logger.info(
            f"Successfully downloaded {downloaded_files}/{len(files_to_download)} files"
        )

        return plate_dir

    def _get_load_data_csv_path(
        self,
        source: str,
        batch: str,
        plate_id: str,
    ) -> str:
        """
        Construct S3 path to load_data.csv.

        Args:
            source: Source identifier
            batch: Batch identifier
            plate_id: Plate identifier

        Returns:
            S3 URL to load_data.csv
        """
        return (
            f"s3://{self.S3_BUCKET}/{self.S3_PROJECT}/"
            f"{source}/workspace/load_data_csv/{batch}/{plate_id}/load_data.csv"
        )

    def _fetch_load_data_csv(self, s3_url: str, plate_dir: Path) -> Optional[pd.DataFrame]:
        """
        Download and parse load_data.csv from S3, caching it in the plate directory.

        Args:
            s3_url: S3 URL to load_data.csv
            plate_dir: Plate cache directory to save the CSV

        Returns:
            DataFrame with load_data contents, or None if failed
        """
        try:
            # Check if already cached
            cached_csv = plate_dir / "load_data.csv"

            if cached_csv.exists():
                logger.debug(f"Using cached load_data.csv: {cached_csv}")
                parser = LoadDataParser()
                return parser.parse(str(cached_csv))

            # Parse S3 URL
            parsed = urlparse(s3_url)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")

            # Download directly to cache directory
            logger.info(f"Downloading load_data.csv to {cached_csv}")
            self.s3_client.download_file(bucket, key, str(cached_csv))

            # Parse CSV
            parser = LoadDataParser()
            df = parser.parse(str(cached_csv))

            return df

        except Exception as e:
            logger.error(f"Failed to fetch load_data.csv from {s3_url}: {e}")
            # Clean up partial file
            if cached_csv.exists():
                cached_csv.unlink()
            return None

    def _filter_files_by_wells(
        self,
        load_data_df: pd.DataFrame,
        wells: List[str],
    ) -> List[Tuple[str, str]]:
        """
        Filter load_data.csv to only files for specified wells.

        Args:
            load_data_df: DataFrame from load_data.csv
            wells: List of well identifiers (e.g., ['A01', 'B02'])

        Returns:
            List of (s3_url, local_relative_path) tuples
        """
        well_files = []

        # Filter to only rows for requested wells
        if "Metadata_Well" in load_data_df.columns:
            filtered_df = load_data_df[load_data_df["Metadata_Well"].isin(wells)]
        else:
            logger.warning("No Metadata_Well column in load_data.csv, downloading all")
            filtered_df = load_data_df

        # Extract S3 URLs from load_data.csv
        # Columns can be: URL_OrigDNA, URL_OrigRNA, etc. OR FileName_OrigDNA, etc.
        # Each contains S3 URL like: s3://bucket/path/to/file.tiff
        for col in filtered_df.columns:
            if col.startswith("URL_Orig") or col.startswith("FileName_Orig"):
                for s3_url in filtered_df[col].dropna():
                    if s3_url.startswith("s3://"):
                        # Extract relative path from S3 URL
                        # s3://bucket/project/source/images/batch/plate/file.tiff
                        # -> file.tiff (or subdirectory structure)
                        parsed = urlparse(s3_url)
                        path_parts = parsed.path.split("/")
                        # Take filename only (last part)
                        filename = path_parts[-1]
                        well_files.append((s3_url, filename))

        return well_files

    def _parallel_download_files(
        self,
        files_to_download: List[Tuple[str, str]],
        plate_dir: Path,
        show_progress: bool = True,
    ) -> int:
        """
        Download files in parallel with connection pooling.

        Args:
            files_to_download: List of (s3_url, local_rel_path) tuples
            plate_dir: Local plate cache directory
            show_progress: Show progress bar

        Returns:
            Number of successfully downloaded files
        """
        downloaded_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            futures = {
                executor.submit(
                    self._download_file_with_retry,
                    s3_url,
                    plate_dir / local_rel_path,
                ): (s3_url, local_rel_path)
                for s3_url, local_rel_path in files_to_download
            }

            # Process completed downloads with progress bar
            with tqdm(
                total=len(futures),
                desc="Downloading files",
                disable=not show_progress,
                unit="file",
            ) as pbar:
                for future in as_completed(futures):
                    s3_url, local_rel_path = futures[future]
                    try:
                        success = future.result()
                        if success:
                            downloaded_count += 1
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Failed to download {local_rel_path}: {e}")
                        pbar.update(1)

        return downloaded_count

    def _download_file_with_retry(
        self,
        s3_url: str,
        local_path: Path,
    ) -> bool:
        """
        Download a single file from S3 with exponential backoff retry.

        Args:
            s3_url: S3 URL to download
            local_path: Local file path to save to

        Returns:
            True if successful, False otherwise
        """
        # Parse S3 URL
        parsed = urlparse(s3_url)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        # Create parent directory
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Retry loop with exponential backoff
        for attempt in range(self.MAX_RETRIES):
            try:
                self.s3_client.download_file(bucket, key, str(local_path))
                return True
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    # Exponential backoff
                    wait_time = self.RETRY_BACKOFF ** attempt
                    logger.debug(
                        f"Download attempt {attempt + 1} failed for {key}, "
                        f"retrying in {wait_time:.1f}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to download {key} after {self.MAX_RETRIES} attempts: {e}"
                    )
                    return False

        return False

    def get_cache_summary(self) -> Dict:
        """
        Get summary of download cache.

        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_summary()

    def cleanup_incomplete(self):
        """Clean up incomplete downloads."""
        self.cache.cleanup_incomplete_downloads()
