"""
Convert Cell Painting TIFFs to Zarr format for fast PyTorch loading.

This script converts already-downloaded TIFF files to Zarr format with Blosc compression,
optimized for 3-10× faster loading than JPEG XL.
"""

import sys
import time
from pathlib import Path
import pandas as pd

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from converter.converter_factory import create_converter
from converter.load_data_parser import LoadDataParser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_plate_to_zarr(
    input_dir,
    output_dir,
    plate_id,
    format="zarr_zstd",
    load_data_csv=None,
    sample_sites=None,
    crop_size=None,
):
    """
    Convert a plate from TIFFs to Zarr format.

    Args:
        input_dir: Directory containing TIFF files
        output_dir: Directory for Zarr output
        plate_id: Plate identifier
        format: Zarr format ("zarr_lz4", "zarr_zstd", "zarr16")
        load_data_csv: Optional path to load_data.csv for metadata
        sample_sites: Optional number of sites to convert (for testing)
        crop_size: Optional center crop size (e.g., 768 for 768x768) or tuple (height, width)
    """
    print("=" * 80)
    print(f"CONVERTING PLATE TO ZARR: {plate_id}")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input TIFFs:   {input_dir}")
    print(f"  Output Zarr:   {output_dir}")
    print(f"  Plate:         {plate_id}")
    print(f"  Format:        {format}")
    if crop_size:
        print(f"  Crop size:     {crop_size}x{crop_size if isinstance(crop_size, int) else 'x'.join(map(str, crop_size))}")
    if sample_sites:
        print(f"  Sample sites:  {sample_sites} (testing mode)")
    print("=" * 80)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process crop_size parameter
    crop_size_tuple = None
    if crop_size is not None:
        if isinstance(crop_size, int):
            crop_size_tuple = (crop_size, crop_size)
        else:
            crop_size_tuple = tuple(crop_size)

    # Create Zarr converter
    zarr_path = output_dir / f"{plate_id}.zarr"

    converter = create_converter(
        format=format,
        zarr_path=zarr_path,
        plate_id=plate_id,
        overwrite=False,  # Set to True to overwrite existing
        crop_size=crop_size_tuple,
    )

    print(f"\n[1/3] Initializing Zarr converter...")
    print(f"  Zarr store: {zarr_path}")
    print(f"  Compressor: {converter.compressor_name}")
    print(f"  Bit depth:  {converter.bit_depth}")

    # Find all TIFF sites
    print(f"\n[2/3] Finding TIFF files...")

    sites = {}

    # Require load_data.csv for accurate file mapping
    if not load_data_csv:
        raise ValueError(
            "load_data.csv is required for Zarr conversion. "
            "This ensures correct file mapping across different filename formats. "
            "Please provide the path to load_data.csv using the load_data_csv parameter."
        )

    print(f"  Using load_data.csv for file mapping: {load_data_csv}")

    parser = LoadDataParser()
    load_data_df = parser.parse(load_data_csv)

    # Create a mapping from filename to full path
    tiff_files_by_name = {}
    for tiff_file in input_dir.glob("*.tif*"):
        tiff_files_by_name[tiff_file.name] = tiff_file

    # Build sites from load_data.csv
    for _, row in load_data_df.iterrows():
        well = row['Metadata_Well']
        site = str(row['Metadata_Site'])
        site_id = f"{well}_F{site.zfill(3)}"

        if site_id not in sites:
            sites[site_id] = {}

        # Map channels from URL columns
        channel_urls = {
            'DNA': row.get('URL_OrigDNA'),
            'RNA': row.get('URL_OrigRNA'),
            'ER': row.get('URL_OrigER'),
            'AGP': row.get('URL_OrigAGP'),
            'Mito': row.get('URL_OrigMito'),
        }

        for channel, url in channel_urls.items():
            if pd.notna(url):
                # Extract filename from S3 URL
                filename = url.split('/')[-1]
                if filename in tiff_files_by_name:
                    sites[site_id][channel] = tiff_files_by_name[filename]

    print(f"  Mapped {len(sites)} sites from load_data.csv")

    # Filter to only complete sites (all 5 channels)
    complete_sites = {
        site_id: channels
        for site_id, channels in sites.items()
        if len(channels) == 5
    }

    print(f"  Found {len(complete_sites)} complete sites")

    if sample_sites:
        # Take only first N sites for testing
        complete_sites = dict(list(complete_sites.items())[:sample_sites])
        print(f"  Using {len(complete_sites)} sites for testing")

    if not complete_sites:
        print("ERROR: No complete sites found!")
        return

    # Convert sites to Zarr
    print(f"\n[3/3] Converting {len(complete_sites)} sites to Zarr...")

    start_time = time.time()
    converted = 0
    failed = 0

    for site_id, channel_files in complete_sites.items():
        try:
            # Parse site metadata from site_id
            # Two formats:
            # 1. Old: r01c02f01p01 (r01 = row 1, c02 = col 2, f01 = field 1)
            # 2. JUMP: A01_F001 (well_field)

            if "_F" in site_id:
                # JUMP format: A01_F001
                parts = site_id.split("_F")
                well = parts[0]  # A01
                field_num = int(parts[1])  # 001 -> 1
            else:
                # Old format: r01c02f01p01
                row_num = int(site_id[1:3])
                col_num = int(site_id[4:6])
                field_num = int(site_id[7:9])
                # Convert row number to letter (1->A, 2->B, etc.)
                well = f"{chr(ord('A') + row_num - 1)}{col_num:02d}"

            site_metadata = {
                "source": "source_1",
                "batch": "Batch1_20221004",
                "plate": plate_id,
                "well": well,
                "site": str(field_num),
            }

            # Convert site
            stats = converter.convert_site(
                channel_files=channel_files,
                output_path=zarr_path,
                site_metadata=site_metadata,
            )

            converted += 1

            if converted % 100 == 0:
                elapsed = time.time() - start_time
                rate = converted / elapsed
                remaining = len(complete_sites) - converted
                eta = remaining / rate if rate > 0 else 0
                print(f"  Progress: {converted}/{len(complete_sites)} sites "
                      f"({converted/len(complete_sites)*100:.1f}%) - "
                      f"{rate:.1f} sites/sec - ETA: {eta:.0f}s")

        except Exception as e:
            logger.error(f"Failed to convert {site_id}: {e}")
            failed += 1

    # Finalize Zarr store
    print(f"\nFinalizing Zarr store...")
    converter.finalize()

    elapsed = time.time() - start_time

    # Get final Zarr size
    zarr_size_mb = sum(
        f.stat().st_size for f in zarr_path.rglob("*") if f.is_file()
    ) / (1024**2)

    # Estimate original TIFF size (rough estimate)
    tiff_size_mb = len(complete_sites) * 14.4  # ~14.4 MB per site average

    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE!")
    print("=" * 80)
    print(f"Sites converted:  {converted}")
    print(f"Sites failed:     {failed}")
    print(f"Time elapsed:     {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Conversion rate:  {converted/elapsed:.1f} sites/second")

    print(f"\nStorage Statistics:")
    print(f"  Original TIFFs (est):  {tiff_size_mb:.1f} MB")
    print(f"  Zarr compressed:       {zarr_size_mb:.1f} MB")
    print(f"  Compression ratio:     {tiff_size_mb/zarr_size_mb:.1f}x")
    print(f"  Space saved:           {(1 - zarr_size_mb/tiff_size_mb)*100:.1f}%")

    print(f"\nOutput:")
    print(f"  Zarr store:      {zarr_path}")
    print(f"  Site index:      {zarr_path}/site_index.json")
    print("=" * 80)

    print("\nNext Steps:")
    print("  1. Load with PyTorch:")
    print(f"     from dl.zarr_dataset import CellPaintingZarrDataset")
    print(f"     dataset = CellPaintingZarrDataset('{zarr_path}')")
    print()
    print("  2. Inspect Zarr:")
    print(f"     import zarr")
    print(f"     store = zarr.open('{zarr_path}', mode='r')")
    print(f"     print(store['sites'].shape)  # (N, 1080, 1080, 5)")
    print("=" * 80)


def main():
    """Main conversion function."""

    # ========================================================================
    # CONFIGURATION - Customize these paths
    # ========================================================================

    # Input directory with TIFF files
    INPUT_DIR = Path("./inputs/UL000109")

    # Output directory for Zarr files
    OUTPUT_DIR = Path("./output_zarr")

    # Plate to convert
    PLATE_ID = "UL000109"

    # Zarr format options: "zarr_lz4", "zarr_zstd", "zarr16"
    # - zarr_lz4:  Fastest loading (3-4× compression, ~10-20ms per site)
    # - zarr_zstd: Balanced (4-5× compression, ~40-60ms per site)
    # - zarr16:    16-bit with Zstd (full dynamic range)
    FORMAT = "zarr_zstd"

    # Optional: Center crop to reduce storage and improve performance
    # - None: No cropping (default 1080×1080)
    # - 768: Crop to 768×768 (50% storage reduction, 2× faster loading)
    # - (768, 768): Same as 768, tuple format
    CROP_SIZE = None  # No cropping

    # Optional: Convert only first N sites for testing
    # Set to None to convert all sites
    SAMPLE_SITES = None  # Convert all 5,888 sites

    # ========================================================================

    convert_plate_to_zarr(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        plate_id=PLATE_ID,
        format=FORMAT,
        sample_sites=SAMPLE_SITES,
        crop_size=CROP_SIZE,
    )


if __name__ == "__main__":
    main()
