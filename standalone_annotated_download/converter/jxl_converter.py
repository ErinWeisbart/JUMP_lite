"""JPEG XL converter for Cell Painting images (8-bit and 16-bit)."""

import logging
from typing import List, Optional

import numpy as np

from .base_converter import BaseConverter

logger = logging.getLogger(__name__)


class JXLConverter(BaseConverter):
    """
    Convert multi-channel Cell Painting TIFF images to JPEG XL.

    Supports both 8-bit and 16-bit encodings for compression benchmarking.

    Features:
    - 16-bit mode: Preserves full dynamic range for quantitative analysis
    - 8-bit mode: Better compression for visualization and perceptual tasks
    - Multi-channel support: Combines DNA, RNA, ER, AGP, Mito into single file
    """

    def __init__(
        self,
        bit_depth: int = 16,
        quality: int = 90,
        effort: int = 7,
        distance: float = 1.0,
        channels: Optional[List[str]] = None,
        create_sidecar: bool = True,
        crop_size: Optional[tuple] = None,
    ):
        """
        Initialize JPEG XL converter.

        Args:
            bit_depth: Output bit depth (8 or 16, default: 16)
            quality: JPEG XL quality (0-100, higher = better quality)
            effort: Encoding effort (1-9, higher = better compression but slower)
            distance: Butteraugli distance (0.0 = lossless, 1.0 = high quality)
            channels: List of channel names (default: DNA, RNA, ER, AGP, Mito)
            create_sidecar: Create JSON sidecar file with metadata
            crop_size: Optional (height, width) tuple for center cropping (e.g., (768, 768))

        Raises:
            ValueError: If bit_depth not in [8, 16]
        """
        super().__init__(channels=channels, create_sidecar=create_sidecar, crop_size=crop_size)

        if bit_depth not in [8, 16]:
            raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth}")

        self.bit_depth = bit_depth
        self.quality = quality
        self.effort = effort
        self.distance = distance

    def get_format_name(self) -> str:
        """Get format identifier."""
        return f"jxl{self.bit_depth}"

    def get_file_extension(self) -> str:
        """Get file extension."""
        return ".jxl"

    def _encode_array(self, stacked: np.ndarray) -> bytes:
        """
        Encode stacked multi-channel array to JPEG XL.

        Args:
            stacked: Multi-channel array of shape (H, W, C)

        Returns:
            Encoded JPEG XL bytes

        Raises:
            RuntimeError: If encoding fails
        """
        try:
            import imagecodecs
        except ImportError:
            raise RuntimeError(
                "imagecodecs not available. Install with: pixi add imagecodecs"
            )

        # Convert to target bit depth if needed
        if self.bit_depth == 8 and stacked.dtype == np.uint16:
            stacked = self._convert_16bit_to_8bit(stacked)
        elif self.bit_depth == 16 and stacked.dtype == np.uint8:
            # Upconvert 8-bit to 16-bit
            stacked = (stacked.astype(np.uint16) * 257)  # 0-255 -> 0-65535

        logger.debug(
            f"Encoding to JPEG XL: shape={stacked.shape}, "
            f"dtype={stacked.dtype}, quality={self.quality}"
        )

        try:
            # Encode to JPEG XL
            # imagecodecs.jpegxl_encode supports multi-channel arrays
            encoded = imagecodecs.jpegxl_encode(
                stacked,
                level=self.quality,  # Quality level
                effort=self.effort,  # Encoding effort
                distance=self.distance,  # Butteraugli distance
            )

            logger.debug(f"Encoded to JPEG XL: {len(encoded)} bytes")
            return encoded

        except Exception as e:
            raise RuntimeError(f"JPEG XL encoding failed: {e}") from e

    def _convert_16bit_to_8bit(self, img_16bit: np.ndarray) -> np.ndarray:
        """
        Convert 16-bit image to 8-bit with per-channel histogram normalization.

        Uses per-channel percentile-based stretching to maximize dynamic range
        utilization in each channel independently. This is optimal for Cell Painting
        images where different fluorescent markers have vastly different intensities.

        Args:
            img_16bit: 16-bit image array of shape (H, W, C)

        Returns:
            8-bit image array of shape (H, W, C)

        Strategy:
        - Process each channel independently
        - Clip to 1st-99th percentile per channel to remove outliers
        - Linear stretch each channel to full 0-255 range
        - Maximizes information preservation in dim channels (e.g., Mito)

        Note: This approach does NOT preserve cross-channel intensity relationships.
        For quantitative analysis requiring relative intensities, use 16-bit mode.
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

        logger.debug(
            f"16-bit -> 8-bit conversion complete: {c} channels processed"
        )

        return img_8bit

    def _create_sidecar(self, output_path, stacked, site_metadata, stats):
        """Create JSON sidecar with additional JXL-specific metadata."""
        # Call parent method to create base sidecar
        super()._create_sidecar(output_path, stacked, site_metadata, stats)

        # Add JXL-specific parameters to the JSON
        import json
        from pathlib import Path

        sidecar_path = Path(output_path).with_suffix(".json")
        with open(sidecar_path, "r") as f:
            sidecar_data = json.load(f)

        # Add JXL encoding parameters
        sidecar_data.update(
            {
                "jxl_quality": self.quality,
                "jxl_effort": self.effort,
                "jxl_distance": self.distance,
                "bit_depth": self.bit_depth,
            }
        )

        with open(sidecar_path, "w") as f:
            json.dump(sidecar_data, f, indent=2)
