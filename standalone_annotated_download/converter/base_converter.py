"""Base converter interface for multi-channel Cell Painting images."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class BaseConverter(ABC):
    """
    Abstract base class for multi-channel image converters.

    Defines common interface for converting Cell Painting images from
    separate channel TIFFs to various compressed formats (JPEG XL, JPEG2000, etc.).

    All converters should:
    1. Load and validate multi-channel data
    2. Encode to target format
    3. Generate metadata sidecars
    4. Track compression statistics
    """

    # Default channel order for Cell Painting
    DEFAULT_CHANNELS = ["DNA", "RNA", "ER", "AGP", "Mito"]

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        create_sidecar: bool = True,
        crop_size: Optional[tuple] = None,
    ):
        """
        Initialize base converter.

        Args:
            channels: List of channel names (default: DNA, RNA, ER, AGP, Mito)
            create_sidecar: Create JSON sidecar file with metadata
            crop_size: Optional (height, width) tuple for center cropping (e.g., (768, 768))
        """
        self.channels = channels or self.DEFAULT_CHANNELS
        self.create_sidecar = create_sidecar
        self.crop_size = crop_size

    @abstractmethod
    def get_format_name(self) -> str:
        """
        Get the format name.

        Returns:
            Format identifier (e.g., "jxl16", "jxl8", "jpeg2000")
        """
        pass

    @abstractmethod
    def get_file_extension(self) -> str:
        """
        Get the file extension for this format.

        Returns:
            File extension including dot (e.g., ".jxl", ".jp2")
        """
        pass

    @abstractmethod
    def _encode_array(self, stacked: np.ndarray) -> bytes:
        """
        Encode stacked multi-channel array to compressed format.

        Args:
            stacked: Multi-channel array of shape (H, W, C)

        Returns:
            Encoded bytes

        Raises:
            RuntimeError: If encoding fails
        """
        pass

    def convert_site(
        self,
        channel_files: Dict[str, Union[str, Path]],
        output_path: Union[str, Path],
        site_metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Convert a single site (5 channels) to compressed format.

        Args:
            channel_files: Dictionary mapping channel names to file paths
                          e.g., {"DNA": "path/to/dna.tif", "RNA": "path/to/rna.tif", ...}
            output_path: Path to output file
            site_metadata: Optional metadata to embed (source, batch, plate, well, site)

        Returns:
            Dictionary with conversion statistics

        Raises:
            ValueError: If channels are invalid or incompatible
            RuntimeError: If conversion fails
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Converting site to {output_path} using {self.get_format_name()}")

        try:
            # Load all channels
            channel_arrays, original_sizes = self._load_channels(channel_files)

            # Validate and stack channels
            stacked = self._stack_channels(channel_arrays)

            # Apply center crop if specified
            stacked = self._crop_center(stacked)

            logger.debug(
                f"Stacked {len(self.channels)} channels: "
                f"shape={stacked.shape}, dtype={stacked.dtype}"
            )

            # Encode to target format
            encoded_bytes = self._encode_array(stacked)

            # Write to file
            with open(output_path, "wb") as f:
                f.write(encoded_bytes)

            # Get output file size
            output_size = output_path.stat().st_size
            total_original_size = sum(original_sizes.values())

            # Compute statistics
            stats = {
                "output_path": str(output_path),
                "format": self.get_format_name(),
                "num_channels": len(self.channels),
                "shape": list(stacked.shape),
                "dtype": str(stacked.dtype),
                "original_size_bytes": total_original_size,
                "compressed_size_bytes": output_size,
                "compression_ratio": total_original_size / output_size if output_size > 0 else 0,
                "space_saved_percent": (1 - output_size / total_original_size) * 100
                if total_original_size > 0
                else 0,
            }

            logger.info(
                f"Converted site ({self.get_format_name()}): {stacked.shape}, "
                f"{total_original_size / (1024**2):.2f} MB → "
                f"{output_size / (1024**2):.2f} MB "
                f"({stats['compression_ratio']:.1f}x compression)"
            )

            # Create JSON sidecar if requested
            if self.create_sidecar:
                self._create_sidecar(output_path, stacked, site_metadata, stats)

            return stats

        except Exception as e:
            logger.error(f"Failed to convert site: {e}")
            raise RuntimeError(f"Conversion failed: {e}") from e

    def _load_channels(
        self, channel_files: Dict[str, Union[str, Path]]
    ) -> tuple[Dict[str, np.ndarray], Dict[str, int]]:
        """
        Load all channel files.

        Args:
            channel_files: Dictionary mapping channel names to file paths

        Returns:
            Tuple of (channel_arrays, original_sizes)

        Raises:
            ValueError: If missing channel
            FileNotFoundError: If channel file doesn't exist
        """
        import tifffile

        channel_arrays = {}
        original_sizes = {}

        for channel_name in self.channels:
            if channel_name not in channel_files:
                raise ValueError(f"Missing channel: {channel_name}")

            channel_path = Path(channel_files[channel_name])
            if not channel_path.exists():
                raise FileNotFoundError(f"Channel file not found: {channel_path}")

            # Load TIFF
            logger.debug(f"Loading {channel_name} from {channel_path}")
            img = tifffile.imread(channel_path)
            channel_arrays[channel_name] = img
            original_sizes[channel_name] = channel_path.stat().st_size

        return channel_arrays, original_sizes

    def _stack_channels(self, channel_arrays: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Stack channels into (H, W, C) array and validate.

        Args:
            channel_arrays: Dictionary of channel names to arrays

        Returns:
            Stacked array of shape (H, W, C)

        Raises:
            ValueError: If channels have mismatched shapes or dtypes
        """
        # Validate channels have same shape and dtype
        shapes = {name: arr.shape for name, arr in channel_arrays.items()}
        dtypes = {name: arr.dtype for name, arr in channel_arrays.items()}

        if len(set(shapes.values())) > 1:
            raise ValueError(f"Channel shapes don't match: {shapes}")

        if len(set(dtypes.values())) > 1:
            raise ValueError(f"Channel dtypes don't match: {dtypes}")

        # Stack channels into (H, W, C) array
        # Order channels according to self.channels
        stacked = np.stack([channel_arrays[ch] for ch in self.channels], axis=-1)

        return stacked

    def _crop_center(self, img: np.ndarray) -> np.ndarray:
        """
        Crop center region from image.

        Args:
            img: Image array of shape (H, W, C)

        Returns:
            Center-cropped image of shape (crop_h, crop_w, C)

        Raises:
            ValueError: If crop_size is larger than image dimensions
        """
        if self.crop_size is None:
            return img

        h, w = img.shape[:2]
        crop_h, crop_w = self.crop_size

        if crop_h > h or crop_w > w:
            raise ValueError(
                f"Crop size ({crop_h}, {crop_w}) is larger than "
                f"image dimensions ({h}, {w})"
            )

        # Calculate center crop offsets
        offset_h = (h - crop_h) // 2
        offset_w = (w - crop_w) // 2

        # Crop image
        cropped = img[offset_h:offset_h + crop_h, offset_w:offset_w + crop_w]

        logger.debug(f"Center cropped from {img.shape} to {cropped.shape}")

        return cropped

    def _create_sidecar(
        self,
        output_path: Path,
        stacked: np.ndarray,
        site_metadata: Optional[Dict],
        stats: Dict,
    ):
        """
        Create JSON sidecar file with metadata.

        Args:
            output_path: Output file path
            stacked: Stacked array
            site_metadata: Site metadata
            stats: Conversion statistics
        """
        sidecar_path = output_path.with_suffix(".json")
        sidecar_data = {
            "format": self.get_format_name(),
            "channels": self.channels,
            "channel_order": list(range(len(self.channels))),
            "shape": list(stacked.shape),
            "dtype": str(stacked.dtype),
            "crop_size": list(self.crop_size) if self.crop_size else None,
            "metadata": site_metadata or {},
            **stats,
        }
        with open(sidecar_path, "w") as f:
            json.dump(sidecar_data, f, indent=2)

    def convert_batch(
        self,
        sites: List[Dict],
        output_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> List[Dict]:
        """
        Convert a batch of sites.

        Args:
            sites: List of site dictionaries with "channels" and metadata
            output_dir: Output directory for compressed files
            progress_callback: Optional callback function(current, total)

        Returns:
            List of conversion statistics for each site
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        total = len(sites)

        for idx, site in enumerate(sites):
            if progress_callback:
                progress_callback(idx, total)

            # Generate output filename
            site_id = self._generate_site_id(site)
            output_path = output_dir / f"{site_id}{self.get_file_extension()}"

            try:
                stats = self.convert_site(
                    channel_files=site["channels"],
                    output_path=output_path,
                    site_metadata={k: v for k, v in site.items() if k != "channels"},
                )
                results.append({"site_id": site_id, "success": True, **stats})
            except Exception as e:
                logger.error(f"Failed to convert site {site_id}: {e}")
                results.append({"site_id": site_id, "success": False, "error": str(e)})

        if progress_callback:
            progress_callback(total, total)

        return results

    def _generate_site_id(self, site: Dict) -> str:
        """
        Generate a unique site ID from metadata.

        Args:
            site: Site dictionary with metadata

        Returns:
            Site ID string (e.g., "source1__batch1__plate1__A01__1")
        """
        parts = [
            site.get("source", "unknown"),
            site.get("batch", "unknown"),
            site.get("plate", "unknown"),
            site.get("well", "unknown"),
            site.get("site", "unknown"),
        ]
        return "__".join(str(p) for p in parts)
