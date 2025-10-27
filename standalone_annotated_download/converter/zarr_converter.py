"""Zarr converter for Cell Painting images with fast loading.

This converter stores Cell Painting images in Zarr format with Blosc compression,
optimized for fast random access and PyTorch data loading.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import zarr
from zarr.codecs import BloscCodec

from .base_converter import BaseConverter

logger = logging.getLogger(__name__)


class ZarrConverter(BaseConverter):
    """
    Convert multi-channel Cell Painting images to Zarr format.

    Zarr provides chunked array storage with fast compression (Blosc),
    enabling 3-10× faster loading than JPEG XL while maintaining good compression.

    Features:
    - Per-plate Zarr stores: Single .zarr directory per plate
    - Blosc compression: LZ4 (fastest) or Zstd (balanced)
    - Chunk optimization: Per-site chunks for random access
    - Supports 8-bit and 16-bit data
    - Metadata stored in Zarr attributes + JSON sidecar
    - PyTorch DataLoader compatible

    Storage structure:
        plate.zarr/
        ├── sites/          # Main array (N_sites, H, W, C)
        ├── .zarray         # Zarr metadata
        ├── .zattrs         # Attributes (plate info, channel names)
        ├── site_index.json # Site metadata index
        └── 0.0.0.0         # Chunk files
    """

    # Supported compressors
    COMPRESSORS = {
        "lz4": {"codec": "lz4", "level": None, "name": "LZ4 (fastest)"},
        "zstd": {"codec": "zstd", "level": 3, "name": "Zstd (balanced)"},
        "zstd_high": {"codec": "zstd", "level": 7, "name": "Zstd (high compression)"},
    }

    def __init__(
        self,
        compressor: str = "zstd",
        bit_depth: int = 8,
        zarr_path: Optional[Union[str, Path]] = None,
        plate_id: Optional[str] = None,
        overwrite: bool = False,
        channels: Optional[List[str]] = None,
        create_sidecar: bool = True,
        crop_size: Optional[tuple] = None,
    ):
        """
        Initialize Zarr converter.

        Args:
            compressor: Compression algorithm ("lz4", "zstd", "zstd_high")
            bit_depth: Bit depth (8 or 16)
            zarr_path: Path to Zarr store (plate.zarr directory)
            plate_id: Plate identifier (used to name Zarr store if zarr_path not provided)
            overwrite: Overwrite existing Zarr store
            channels: List of channel names (default: DNA, RNA, ER, AGP, Mito)
            create_sidecar: Create JSON sidecar with metadata
            crop_size: Optional (height, width) tuple for center cropping (e.g., (768, 768))

        Raises:
            ValueError: If compressor or bit_depth is invalid
        """
        super().__init__(channels=channels, create_sidecar=create_sidecar, crop_size=crop_size)

        if compressor not in self.COMPRESSORS:
            raise ValueError(
                f"Invalid compressor: {compressor}. "
                f"Choose from: {list(self.COMPRESSORS.keys())}"
            )

        if bit_depth not in [8, 16]:
            raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth}")

        self.compressor_name = compressor
        self.bit_depth = bit_depth
        self.zarr_path = Path(zarr_path) if zarr_path else None
        self.plate_id = plate_id
        self.overwrite = overwrite

        # Create Blosc compressor (Zarr 3.x API)
        comp_config = self.COMPRESSORS[compressor]
        self.compressor = BloscCodec(
            cname=comp_config["codec"],
            clevel=comp_config["level"] if comp_config["level"] else 5,
            shuffle="bitshuffle",  # Better for scientific data
        )

        # Track sites for per-plate conversion
        self.site_index = []
        self.zarr_store = None
        self.zarr_array = None

    def get_format_name(self) -> str:
        """Get format identifier."""
        return f"zarr_{self.compressor_name}_{self.bit_depth}bit"

    def get_file_extension(self) -> str:
        """Get file extension."""
        return ".zarr"

    def _initialize_zarr_store(self, first_site_shape: tuple):
        """
        Initialize Zarr store for plate-level storage.

        Args:
            first_site_shape: Shape of first site (H, W, C) to determine array structure
        """
        if self.zarr_store is not None:
            return  # Already initialized

        if self.zarr_path is None:
            raise ValueError("zarr_path must be set before conversion")

        logger.info(f"Initializing Zarr store: {self.zarr_path}")

        # Create Zarr store (Zarr 3.x API)
        self.zarr_store = zarr.open_group(
            str(self.zarr_path),
            mode="w" if self.overwrite else "a",
        )

        # Determine dtype
        dtype = np.uint8 if self.bit_depth == 8 else np.uint16

        # Create resizable array for sites
        # Shape: (num_sites, H, W, C) - will grow as sites are added
        h, w, c = first_site_shape

        self.zarr_array = self.zarr_store.create(
            "sites",
            shape=(0, h, w, c),  # Start with 0 sites
            chunks=(1, h, w, c),  # One chunk per site for random access
            dtype=dtype,
            compressors=[self.compressor],  # Zarr 3.x uses compressors list
            overwrite=self.overwrite,
        )

        # Store metadata in attributes
        self.zarr_array.attrs["channels"] = self.channels
        self.zarr_array.attrs["plate_id"] = self.plate_id or "unknown"
        self.zarr_array.attrs["bit_depth"] = self.bit_depth
        self.zarr_array.attrs["compressor"] = self.compressor_name
        self.zarr_array.attrs["format"] = self.get_format_name()
        self.zarr_array.attrs["crop_size"] = list(self.crop_size) if self.crop_size else None

        logger.info(
            f"Initialized Zarr array: shape={self.zarr_array.shape}, "
            f"chunks={self.zarr_array.chunks}, dtype={dtype}"
        )

    def _append_site(self, stacked: np.ndarray, site_metadata: Dict):
        """
        Append a site to the Zarr array.

        Args:
            stacked: Site data of shape (H, W, C)
            site_metadata: Site metadata dictionary
        """
        if self.zarr_array is None:
            self._initialize_zarr_store(stacked.shape)

        # Resize array to accommodate new site (Zarr 3.x API)
        current_size = self.zarr_array.shape[0]
        h, w, c = self.zarr_array.shape[1:]
        self.zarr_array.resize((current_size + 1, h, w, c))

        # Append site
        self.zarr_array[current_size] = stacked

        # Track site metadata
        site_info = {
            "index": current_size,
            "shape": list(stacked.shape),
            "dtype": str(stacked.dtype),
            **site_metadata,
        }
        self.site_index.append(site_info)

        logger.debug(f"Appended site {current_size}: {site_metadata}")

    def _save_site_index(self):
        """Save site index as JSON for fast lookup."""
        if self.zarr_path is None:
            return

        index_path = self.zarr_path / "site_index.json"
        with open(index_path, "w") as f:
            json.dump(
                {
                    "plate_id": self.plate_id,
                    "num_sites": len(self.site_index),
                    "channels": self.channels,
                    "bit_depth": self.bit_depth,
                    "compressor": self.compressor_name,
                    "crop_size": list(self.crop_size) if self.crop_size else None,
                    "sites": self.site_index,
                },
                f,
                indent=2,
            )

        logger.info(f"Saved site index: {index_path} ({len(self.site_index)} sites)")

    def _encode_array(self, stacked: np.ndarray) -> bytes:
        """
        Encode stacked multi-channel array to Zarr.

        For Zarr, we don't actually encode to bytes - we append to the Zarr array.
        This method is kept for compatibility with BaseConverter but returns empty bytes.

        Args:
            stacked: Multi-channel array of shape (H, W, C)

        Returns:
            Empty bytes (actual data is written to Zarr store)
        """
        # This method is overridden by convert_site for Zarr
        # We return empty bytes to satisfy the interface
        return b""

    def convert_site(
        self,
        channel_files: Dict[str, Union[str, Path]],
        output_path: Union[str, Path],
        site_metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Convert a single site to Zarr format.

        For Zarr, output_path should be the plate-level .zarr directory.
        Each site is appended to the Zarr array.

        Args:
            channel_files: Dictionary mapping channel names to file paths
            output_path: Path to Zarr store (plate.zarr directory)
            site_metadata: Optional metadata (source, batch, plate, well, site)

        Returns:
            Dictionary with conversion statistics
        """
        output_path = Path(output_path)

        # Set zarr_path if not already set
        if self.zarr_path is None:
            self.zarr_path = output_path

        # Ensure it's a directory path ending in .zarr
        if not str(self.zarr_path).endswith(".zarr"):
            self.zarr_path = self.zarr_path.with_suffix(".zarr")

        site_metadata = site_metadata or {}

        logger.debug(f"Converting site to Zarr: {self.zarr_path}")

        # Load all channels
        channel_arrays, original_sizes = self._load_channels(channel_files)

        # Validate and stack channels
        stacked = self._stack_channels(channel_arrays)

        # Apply center crop if specified
        stacked = self._crop_center(stacked)

        # Convert to target bit depth if needed
        if self.bit_depth == 8 and stacked.dtype == np.uint16:
            stacked = self._convert_16bit_to_8bit(stacked)
        elif self.bit_depth == 16 and stacked.dtype == np.uint8:
            stacked = (stacked.astype(np.uint16) * 257)  # 0-255 -> 0-65535

        logger.debug(
            f"Stacked {len(self.channels)} channels: "
            f"shape={stacked.shape}, dtype={stacked.dtype}"
        )

        # Append to Zarr array
        self._append_site(stacked, site_metadata)

        # Compute statistics
        total_original_size = sum(original_sizes.values())

        # Estimate compressed size (will be more accurate after finalize())
        compressed_size_estimate = stacked.nbytes / 4  # Rough estimate for Zstd

        stats = {
            "output_path": str(self.zarr_path),
            "format": self.get_format_name(),
            "num_channels": len(self.channels),
            "shape": list(stacked.shape),
            "dtype": str(stacked.dtype),
            "original_size_bytes": total_original_size,
            "compressed_size_bytes": compressed_size_estimate,
            "compression_ratio": total_original_size / compressed_size_estimate
            if compressed_size_estimate > 0
            else 0,
            "space_saved_percent": (1 - compressed_size_estimate / total_original_size) * 100
            if total_original_size > 0
            else 0,
            "site_index": len(self.site_index) - 1,  # Index of this site
        }

        logger.info(
            f"Added site to Zarr ({self.get_format_name()}): "
            f"index={stats['site_index']}, shape={stacked.shape}"
        )

        return stats

    def _convert_16bit_to_8bit(self, img_16bit: np.ndarray) -> np.ndarray:
        """
        Convert 16-bit image to 8-bit with per-channel percentile normalization.

        Uses 1-99 percentile clipping (same as JXL converter).

        Args:
            img_16bit: 16-bit image array of shape (H, W, C)

        Returns:
            8-bit image array of shape (H, W, C)
        """
        logger.debug("Converting 16-bit to 8-bit with per-channel percentile normalization")

        h, w, c = img_16bit.shape
        img_8bit = np.zeros((h, w, c), dtype=np.uint8)

        # Process each channel independently
        for ch_idx in range(c):
            channel = img_16bit[:, :, ch_idx]

            # Calculate per-channel percentiles
            p_low = np.percentile(channel, 1)
            p_high = np.percentile(channel, 99)

            # Avoid division by zero for flat channels
            if p_high - p_low < 1:
                logger.warning(f"Channel {ch_idx} has very low dynamic range")
                img_8bit[:, :, ch_idx] = 0
                continue

            # Clip and normalize this channel
            ch_clipped = np.clip(channel, p_low, p_high)
            ch_normalized = (ch_clipped - p_low) / (p_high - p_low)

            # Scale to 8-bit range
            img_8bit[:, :, ch_idx] = (ch_normalized * 255).astype(np.uint8)

            logger.debug(
                f"Channel {ch_idx}: [{channel.min()}, {channel.max()}] -> "
                f"[{img_8bit[:, :, ch_idx].min()}, {img_8bit[:, :, ch_idx].max()}]"
            )

        return img_8bit

    def finalize(self):
        """
        Finalize Zarr conversion.

        Saves the site index and computes final statistics.
        Call this after converting all sites in a plate.
        """
        if self.zarr_array is None:
            logger.warning("No Zarr array to finalize")
            return

        # Save site index
        self._save_site_index()

        # Log final stats
        zarr_size_mb = sum(
            f.stat().st_size for f in self.zarr_path.rglob("*") if f.is_file()
        ) / (1024**2)

        logger.info(
            f"Finalized Zarr store: {self.zarr_path}\n"
            f"  Sites: {len(self.site_index)}\n"
            f"  Size: {zarr_size_mb:.2f} MB\n"
            f"  Compressor: {self.compressor_name}\n"
            f"  Bit depth: {self.bit_depth}"
        )

    def _create_sidecar(self, output_path, stacked, site_metadata, stats):
        """
        Create JSON sidecar with Zarr-specific metadata.

        For Zarr, the sidecar is the site_index.json file.
        This method is overridden to prevent creating per-site sidecars.
        """
        # For Zarr, we don't create per-site sidecars
        # The site_index.json serves as the comprehensive metadata file
        pass
