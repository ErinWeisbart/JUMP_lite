"""JPEG2000 converter for Cell Painting images."""

import logging
from typing import List, Optional

import numpy as np

from .base_converter import BaseConverter

logger = logging.getLogger(__name__)


class JPEG2000Converter(BaseConverter):
    """
    Convert multi-channel Cell Painting TIFF images to JPEG2000.

    JPEG2000 is a wavelet-based compression standard widely used in medical
    and scientific imaging. It supports:
    - 16-bit multi-channel images
    - Lossless and lossy compression
    - Progressive decoding
    - Better quality at high compression ratios than JPEG

    Features:
    - Preserves full 16-bit dynamic range
    - Configurable compression ratio
    - Reversible or irreversible compression
    """

    def __init__(
        self,
        compression_ratio: float = 10.0,
        reversible: bool = False,
        quality: Optional[int] = None,
        channels: Optional[List[str]] = None,
        create_sidecar: bool = True,
        crop_size: Optional[tuple] = None,
    ):
        """
        Initialize JPEG2000 converter.

        Args:
            compression_ratio: Target compression ratio (default: 10.0)
                             Higher = more compression, lower quality
            reversible: Use reversible (lossless) compression
                       If True, compression_ratio is ignored
            quality: Optional quality parameter (0-100) for compatibility
                    If provided, maps to compression_ratio: quality 90 -> ratio 10
            channels: List of channel names (default: DNA, RNA, ER, AGP, Mito)
            create_sidecar: Create JSON sidecar file with metadata
            crop_size: Optional (height, width) tuple for center cropping (e.g., (768, 768))
        """
        super().__init__(channels=channels, create_sidecar=create_sidecar, crop_size=crop_size)

        # Map quality to compression_ratio if provided
        if quality is not None:
            # Higher quality = lower compression ratio
            # quality 100 -> ratio ~2, quality 90 -> ratio ~10, quality 50 -> ratio ~50
            self.compression_ratio = max(2.0, 100.0 / max(1, quality) * 2)
        else:
            self.compression_ratio = compression_ratio

        self.reversible = reversible

    def get_format_name(self) -> str:
        """Get format identifier."""
        return "jpeg2000"

    def get_file_extension(self) -> str:
        """Get file extension."""
        return ".jp2"

    def _encode_array(self, stacked: np.ndarray) -> bytes:
        """
        Encode stacked multi-channel array to JPEG2000.

        Args:
            stacked: Multi-channel array of shape (H, W, C)

        Returns:
            Encoded JPEG2000 bytes

        Raises:
            RuntimeError: If encoding fails
        """
        # Try imagecodecs first (preferred), fallback to OpenCV
        try:
            return self._encode_with_imagecodecs(stacked)
        except ImportError:
            logger.warning("imagecodecs not available, trying OpenCV fallback")
            return self._encode_with_opencv(stacked)
        except Exception as e:
            logger.warning(f"imagecodecs encoding failed: {e}, trying OpenCV fallback")
            return self._encode_with_opencv(stacked)

    def _encode_with_imagecodecs(self, stacked: np.ndarray) -> bytes:
        """
        Encode using imagecodecs library.

        Args:
            stacked: Multi-channel array

        Returns:
            Encoded bytes

        Raises:
            ImportError: If imagecodecs not available
            RuntimeError: If encoding fails
        """
        try:
            import imagecodecs
        except ImportError:
            raise ImportError(
                "imagecodecs not available. Install with: pixi add imagecodecs"
            )

        logger.debug(
            f"Encoding to JPEG2000 (imagecodecs): shape={stacked.shape}, "
            f"dtype={stacked.dtype}, reversible={self.reversible}"
        )

        try:
            # imagecodecs.jpeg2k_encode parameters
            # Note: Different versions of imagecodecs have different APIs
            # Try with available parameters
            if self.reversible:
                # Lossless mode
                encoded = imagecodecs.jpeg2k_encode(stacked, level=None)
            else:
                # Lossy mode with quality/compression ratio
                # Use 'level' parameter (higher = better quality, lower compression)
                # Map compression_ratio to quality level (inverse relationship)
                level = max(1, int(100 - self.compression_ratio * 5))
                encoded = imagecodecs.jpeg2k_encode(stacked, level=level)

            logger.debug(f"Encoded to JPEG2000: {len(encoded)} bytes")
            return encoded

        except Exception as e:
            raise RuntimeError(f"JPEG2000 encoding (imagecodecs) failed: {e}") from e

    def _encode_with_opencv(self, stacked: np.ndarray) -> bytes:
        """
        Encode using OpenCV (fallback method).

        OpenCV's JPEG2000 encoder has limitations:
        - May not support multi-channel images directly
        - Need to encode each channel separately and combine

        Args:
            stacked: Multi-channel array

        Returns:
            Encoded bytes

        Raises:
            ImportError: If OpenCV not available
            RuntimeError: If encoding fails
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV not available. Install with: pixi add opencv")

        logger.debug(
            f"Encoding to JPEG2000 (OpenCV): shape={stacked.shape}, "
            f"dtype={stacked.dtype}"
        )

        # OpenCV limitation: Need to save to disk first
        # We'll use a different approach: encode channels separately and stack metadata

        # For multi-channel JPEG2000 with OpenCV, we need to:
        # 1. Save each channel as a separate JPEG2000 layer
        # 2. Or, save as multi-page TIFF with JPEG2000 compression (not standard JP2)
        #
        # Since OpenCV's JPEG2000 support is limited, we'll save as a
        # compressed numpy array as a workaround for now

        raise NotImplementedError(
            "OpenCV JPEG2000 encoding for multi-channel images is not yet "
            "implemented. Please install imagecodecs: pixi add imagecodecs"
        )

    def _create_sidecar(self, output_path, stacked, site_metadata, stats):
        """Create JSON sidecar with additional JPEG2000-specific metadata."""
        # Call parent method to create base sidecar
        super()._create_sidecar(output_path, stacked, site_metadata, stats)

        # Add JPEG2000-specific parameters to the JSON
        import json
        from pathlib import Path

        sidecar_path = Path(output_path).with_suffix(".json")
        with open(sidecar_path, "r") as f:
            sidecar_data = json.load(f)

        # Add JPEG2000 encoding parameters
        sidecar_data.update(
            {
                "jpeg2000_compression_ratio": self.compression_ratio,
                "jpeg2000_reversible": self.reversible,
            }
        )

        with open(sidecar_path, "w") as f:
            json.dump(sidecar_data, f, indent=2)
