"""
Download manager for metadata-based selective downloading.

This module provides a high-level interface for downloading Cell Painting data
based on DuckDB metadata queries, with optional on-the-fly conversion to Zarr format.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from .metadata_filter import MetadataFilter
from .selective_downloader import SelectiveWellDownloader

logger = logging.getLogger(__name__)


class DownloadManager:
    """
    High-level manager for metadata-driven selective downloads.

    Features:
    - Query metadata to generate download manifests
    - Estimate download sizes and times
    - Execute parallel downloads with progress tracking
    - Optional on-the-fly conversion to Zarr
    - Resume support for interrupted downloads

    Example:
        >>> # Download only COMPOUND plates
        >>> manager = DownloadManager(
        ...     db_path='jump_metadata.duckdb',
        ...     cache_dir='./data'
        ... )
        >>> plates = manager.download_by_plate_type(
        ...     plate_types=['COMPOUND'],
        ...     max_plates=10
        ... )

        >>> # Download specific compounds
        >>> wells = manager.download_by_compounds(
        ...     jcp_ids=['JCP2022_033924', 'JCP2022_085227']
        ... )
    """

    # Storage estimates (average per site)
    SIZE_TIFF_PER_SITE_MB = 14.4  # Original TIFF (5 channels)
    SIZE_ZARR_PER_SITE_MB = 3.4  # Zarr with Zstd

    # Download speed estimates (baseline)
    DOWNLOAD_SPEED_MBPS = 100  # 100 MB/s typical

    def __init__(
        self,
        db_path: str,
        cache_dir: Union[str, Path],
        max_workers: Optional[int] = None,
    ):
        """
        Initialize download manager.

        Args:
            db_path: Path to DuckDB metadata database
            cache_dir: Directory for caching downloaded files
            max_workers: Number of parallel download workers (default: 32)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata filter
        self.metadata_filter = MetadataFilter(db_path)

        # Initialize selective downloader
        self.downloader = SelectiveWellDownloader(
            cache_dir=cache_dir,
            max_workers=max_workers,
        )

        logger.info(
            f"Initialized DownloadManager with cache at {cache_dir}"
        )

    def download_by_plate_type(
        self,
        plate_types: List[str],
        max_plates: Optional[int] = None,
        force: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Path]:
        """
        Download plates filtered by plate type.

        Args:
            plate_types: List of plate types (e.g., ['COMPOUND', 'ORF'])
            max_plates: Optional maximum number of plates
            force: Force re-download even if cached
            show_progress: Show progress bars

        Returns:
            Dictionary mapping plate_id -> local cache directory

        Example:
            >>> plates = manager.download_by_plate_type(
            ...     plate_types=['COMPOUND'],
            ...     max_plates=100
            ... )
            >>> print(f"Downloaded {len(plates)} plates")
        """
        logger.info(f"Filtering plates by type: {plate_types}")

        # Query metadata
        plates_df = self.metadata_filter.filter_by_plate_type(
            plate_types=plate_types,
            max_plates=max_plates,
        )

        if plates_df.empty:
            logger.warning(f"No plates found for types: {plate_types}")
            return {}

        # Estimate download size
        self._log_download_estimate(len(plates_df), sites_per_plate=384 * 6)

        # Create download manifest with S3 paths
        manifest = self.metadata_filter.get_download_manifest(
            plates_df,
            include_csv_paths=True,
        )

        # Download full plates (no well filtering)
        return self._download_full_plates(
            manifest=manifest,
            force=force,
            show_progress=show_progress,
        )

    def download_by_perturbation(
        self,
        modalities: List[str],
        max_plates: Optional[int] = None,
        force: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Path]:
        """
        Download plates filtered by perturbation modality.

        Args:
            modalities: List of modalities (e.g., ['compound', 'crispr'])
            max_plates: Optional maximum number of plates
            force: Force re-download
            show_progress: Show progress bars

        Returns:
            Dictionary mapping plate_id -> local cache directory

        Example:
            >>> plates = manager.download_by_perturbation(
            ...     modalities=['compound'],
            ...     max_plates=50
            ... )
        """
        logger.info(f"Filtering plates by perturbation: {modalities}")

        # Query metadata
        plates_df = self.metadata_filter.filter_by_perturbation(
            modalities=modalities,
            max_plates=max_plates,
        )

        if plates_df.empty:
            logger.warning(f"No plates found for modalities: {modalities}")
            return {}

        # Estimate and download
        self._log_download_estimate(len(plates_df), sites_per_plate=384 * 6)

        manifest = self.metadata_filter.get_download_manifest(
            plates_df,
            include_csv_paths=True,
        )

        return self._download_full_plates(
            manifest=manifest,
            force=force,
            show_progress=show_progress,
        )

    def download_by_compounds(
        self,
        jcp_ids: List[str],
        force: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Path]:
        """
        Download specific compounds (well-level filtering).

        Args:
            jcp_ids: List of JCP2022 compound identifiers
            force: Force re-download
            show_progress: Show progress bars

        Returns:
            Dictionary mapping plate_id -> local cache directory

        Example:
            >>> # Download only wells with specific compounds
            >>> compounds = ['JCP2022_033924', 'JCP2022_085227']
            >>> plates = manager.download_by_compounds(compounds)
            >>> # Result: only wells containing these compounds
        """
        logger.info(f"Filtering by {len(jcp_ids)} compounds")

        # Query metadata (returns well-level data)
        wells_df = self.metadata_filter.filter_by_compounds(
            jcp_ids=jcp_ids,
            return_wells=True,
        )

        if wells_df.empty:
            logger.warning(f"No wells found for compounds: {jcp_ids}")
            return {}

        # Estimate download size (well-level)
        n_wells = len(wells_df)
        n_sites = n_wells * 6  # 6 sites per well
        self._log_download_estimate(
            n_plates=wells_df["Metadata_Plate"].nunique(),
            sites_per_plate=None,
            total_sites=n_sites,
        )

        # Download with well-level filtering
        return self.downloader.download_wells(
            wells_manifest=wells_df,
            force=force,
            show_progress=show_progress,
        )

    def download_balanced_sample(
        self,
        n_per_type: int = 10,
        plate_types: Optional[List[str]] = None,
        force: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Path]:
        """
        Download balanced sample across plate types.

        Args:
            n_per_type: Number of plates per type
            plate_types: Optional list of types to include (default: all)
            force: Force re-download
            show_progress: Show progress bars

        Returns:
            Dictionary mapping plate_id -> local cache directory

        Example:
            >>> # Get 20 plates from each type (COMPOUND, ORF, CRISPR, etc.)
            >>> plates = manager.download_balanced_sample(n_per_type=20)
        """
        logger.info(f"Creating balanced sample: {n_per_type} per type")

        # Query metadata
        plates_df = self.metadata_filter.balanced_sample(
            n_per_type=n_per_type,
            plate_types=plate_types,
        )

        if plates_df.empty:
            logger.warning("No plates found for balanced sample")
            return {}

        # Estimate and download
        self._log_download_estimate(len(plates_df), sites_per_plate=384 * 6)

        manifest = self.metadata_filter.get_download_manifest(
            plates_df,
            include_csv_paths=True,
        )

        return self._download_full_plates(
            manifest=manifest,
            force=force,
            show_progress=show_progress,
        )

    def download_from_manifest(
        self,
        manifest_path: Union[str, Path],
        force: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Path]:
        """
        Download from a pre-generated manifest CSV.

        Args:
            manifest_path: Path to manifest CSV
            force: Force re-download
            show_progress: Show progress bars

        Returns:
            Dictionary mapping plate_id -> local cache directory

        Example:
            >>> # Use pre-generated manifest
            >>> plates = manager.download_from_manifest('manifest.csv')
        """
        manifest = pd.read_csv(manifest_path)

        logger.info(f"Loaded manifest with {len(manifest)} entries")

        # Check if well-level or plate-level manifest
        if "Metadata_Well" in manifest.columns:
            # Well-level manifest
            return self.downloader.download_wells(
                wells_manifest=manifest,
                force=force,
                show_progress=show_progress,
            )
        else:
            # Plate-level manifest
            return self._download_full_plates(
                manifest=manifest,
                force=force,
                show_progress=show_progress,
            )

    def _download_full_plates(
        self,
        manifest: pd.DataFrame,
        force: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Path]:
        """
        Download full plates (no well filtering).

        Args:
            manifest: DataFrame with plate information
            force: Force re-download
            show_progress: Show progress bars

        Returns:
            Dictionary mapping plate_id -> local cache directory
        """
        downloaded_plates = {}

        for _, row in manifest.iterrows():
            try:
                plate_id = row["Metadata_Plate"]
                source = row["Metadata_Source"]
                batch = row["Metadata_Batch"]

                # Construct S3 prefix for images
                # Structure: cpg0016-jump/{source}/images/{batch}/images/{plate_id}/
                s3_prefix = (
                    f"{SelectiveWellDownloader.S3_PROJECT}/"
                    f"{source}/images/{batch}/images/{plate_id}/"
                )

                # Download full plate
                plate_dir = self.downloader.plate_downloader.download_plate(
                    plate_id=plate_id,
                    s3_prefix=s3_prefix,
                    force=force,
                )

                downloaded_plates[plate_id] = plate_dir

            except Exception as e:
                logger.error(f"Failed to download plate {row['Metadata_Plate']}: {e}")
                continue

        return downloaded_plates

    def _log_download_estimate(
        self,
        n_plates: int,
        sites_per_plate: Optional[int] = None,
        total_sites: Optional[int] = None,
    ):
        """
        Log estimated download size and time.

        Args:
            n_plates: Number of plates
            sites_per_plate: Sites per plate (if known)
            total_sites: Total sites (if known, overrides n_plates × sites_per_plate)
        """
        # Calculate total sites
        if total_sites is None:
            if sites_per_plate is None:
                sites_per_plate = 384 * 6  # Default: 384 wells × 6 sites
            total_sites = n_plates * sites_per_plate

        # Estimate sizes
        size_tiff_gb = (total_sites * self.SIZE_TIFF_PER_SITE_MB) / 1024
        size_zarr_gb = (total_sites * self.SIZE_ZARR_PER_SITE_MB) / 1024

        # Estimate download time (for TIFF)
        download_time_sec = (size_tiff_gb * 1024) / self.DOWNLOAD_SPEED_MBPS
        download_time_min = download_time_sec / 60
        download_time_hours = download_time_min / 60

        logger.info(
            f"\n"
            f"Download Estimate:\n"
            f"  Plates:           {n_plates:,}\n"
            f"  Sites:            {total_sites:,}\n"
            f"  Size (TIFF):      {size_tiff_gb:.1f} GB\n"
            f"  Size (Zarr):      {size_zarr_gb:.1f} GB (after conversion)\n"
            f"  Compression:      {size_tiff_gb/size_zarr_gb:.1f}x\n"
            f"  Est. time:        {download_time_min:.1f} min "
            f"({download_time_hours:.1f} hours @ {self.DOWNLOAD_SPEED_MBPS} MB/s)"
        )

    def get_cache_summary(self) -> Dict:
        """
        Get summary of download cache.

        Returns:
            Dictionary with cache statistics
        """
        return self.downloader.get_cache_summary()

    def cleanup_incomplete(self):
        """Clean up incomplete downloads."""
        self.downloader.cleanup_incomplete()

    def close(self):
        """Close database connection."""
        self.metadata_filter.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
