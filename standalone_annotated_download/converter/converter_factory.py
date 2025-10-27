"""Factory for creating image format converters."""

import logging
from typing import Dict, List, Optional

from .base_converter import BaseConverter
from .jpeg2000_converter import JPEG2000Converter
from .jxl_converter import JXLConverter
from .zarr_converter import ZarrConverter

logger = logging.getLogger(__name__)


# Registry of available formats
AVAILABLE_FORMATS = {
    "jxl16": {
        "name": "JPEG XL (16-bit)",
        "description": "16-bit JPEG XL with full dynamic range preservation",
        "extension": ".jxl",
        "class": JXLConverter,
        "default_params": {"bit_depth": 16, "quality": 90},
    },
    "jxl8": {
        "name": "JPEG XL (8-bit)",
        "description": "8-bit JPEG XL with better compression for visualization",
        "extension": ".jxl",
        "class": JXLConverter,
        "default_params": {"bit_depth": 8, "quality": 90},
    },
    "jpeg2000": {
        "name": "JPEG2000",
        "description": "JPEG2000 wavelet-based compression (16-bit)",
        "extension": ".jp2",
        "class": JPEG2000Converter,
        "default_params": {"compression_ratio": 10.0, "reversible": False},
    },
    "zarr_lz4": {
        "name": "Zarr with LZ4 (8-bit)",
        "description": "Zarr chunked storage with Blosc-LZ4 (fastest loading, 3-4× compression)",
        "extension": ".zarr",
        "class": ZarrConverter,
        "default_params": {"compressor": "lz4", "bit_depth": 8},
    },
    "zarr_zstd": {
        "name": "Zarr with Zstd (8-bit)",
        "description": "Zarr chunked storage with Blosc-Zstd (balanced speed/compression, 4-5×)",
        "extension": ".zarr",
        "class": ZarrConverter,
        "default_params": {"compressor": "zstd", "bit_depth": 8},
    },
    "zarr16": {
        "name": "Zarr with Zstd (16-bit)",
        "description": "Zarr chunked storage with Blosc-Zstd (16-bit, full dynamic range)",
        "extension": ".zarr",
        "class": ZarrConverter,
        "default_params": {"compressor": "zstd", "bit_depth": 16},
    },
}


def create_converter(
    format: str = "jxl16",
    channels: Optional[List[str]] = None,
    create_sidecar: bool = True,
    crop_size: Optional[tuple] = None,
    **format_specific_params,
) -> BaseConverter:
    """
    Create a converter for the specified format.

    Args:
        format: Format identifier ("jxl16", "jxl8", "jpeg2000")
        channels: List of channel names (default: DNA, RNA, ER, AGP, Mito)
        create_sidecar: Create JSON sidecar with metadata
        crop_size: Optional (height, width) tuple for center cropping (e.g., (768, 768))
        **format_specific_params: Format-specific parameters

    Returns:
        Converter instance

    Raises:
        ValueError: If format is not recognized

    Examples:
        # Create 16-bit JPEG XL converter
        converter = create_converter("jxl16", quality=95)

        # Create 8-bit JPEG XL converter with center crop
        converter = create_converter("jxl8", quality=85, crop_size=(768, 768))

        # Create JPEG2000 converter
        converter = create_converter("jpeg2000", compression_ratio=15.0)
    """
    if format not in AVAILABLE_FORMATS:
        valid_formats = ", ".join(AVAILABLE_FORMATS.keys())
        raise ValueError(
            f"Unknown format: {format}. Available formats: {valid_formats}"
        )

    format_info = AVAILABLE_FORMATS[format]
    converter_class = format_info["class"]

    # Merge default parameters with user-provided parameters
    params = {**format_info["default_params"], **format_specific_params}

    # Add common parameters
    params["channels"] = channels
    params["create_sidecar"] = create_sidecar
    params["crop_size"] = crop_size

    logger.debug(
        f"Creating converter: {format_info['name']} with params: {params}"
    )

    return converter_class(**params)


def list_formats() -> Dict[str, Dict]:
    """
    List all available compression formats.

    Returns:
        Dictionary of format information

    Example:
        >>> formats = list_formats()
        >>> for fmt_id, info in formats.items():
        ...     print(f"{fmt_id}: {info['name']} - {info['description']}")
    """
    return AVAILABLE_FORMATS.copy()


def get_format_info(format: str) -> Dict:
    """
    Get information about a specific format.

    Args:
        format: Format identifier

    Returns:
        Format information dictionary

    Raises:
        ValueError: If format is not recognized
    """
    if format not in AVAILABLE_FORMATS:
        valid_formats = ", ".join(AVAILABLE_FORMATS.keys())
        raise ValueError(
            f"Unknown format: {format}. Available formats: {valid_formats}"
        )

    return AVAILABLE_FORMATS[format].copy()


def validate_format_params(format: str, **params) -> Dict:
    """
    Validate and normalize format-specific parameters.

    Args:
        format: Format identifier
        **params: Parameters to validate

    Returns:
        Validated and normalized parameters

    Raises:
        ValueError: If parameters are invalid
    """
    if format not in AVAILABLE_FORMATS:
        valid_formats = ", ".join(AVAILABLE_FORMATS.keys())
        raise ValueError(
            f"Unknown format: {format}. Available formats: {valid_formats}"
        )

    format_info = AVAILABLE_FORMATS[format]
    validated = format_info["default_params"].copy()

    # Update with provided params
    validated.update(params)

    # Format-specific validation
    if format in ["jxl16", "jxl8"]:
        # Validate JXL parameters
        if "quality" in validated:
            quality = validated["quality"]
            if not (0 <= quality <= 100):
                raise ValueError(f"JXL quality must be 0-100, got {quality}")

        if "effort" in validated:
            effort = validated["effort"]
            if not (1 <= effort <= 9):
                raise ValueError(f"JXL effort must be 1-9, got {effort}")

        if "bit_depth" in validated:
            bit_depth = validated["bit_depth"]
            if bit_depth not in [8, 16]:
                raise ValueError(f"JXL bit_depth must be 8 or 16, got {bit_depth}")

    elif format == "jpeg2000":
        # Validate JPEG2000 parameters
        if "compression_ratio" in validated:
            ratio = validated["compression_ratio"]
            if ratio < 1.0:
                raise ValueError(
                    f"JPEG2000 compression_ratio must be >= 1.0, got {ratio}"
                )

    return validated
