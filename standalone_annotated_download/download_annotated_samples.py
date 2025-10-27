"""
Download JUMP Compound Treatments with Repurposing Hub Annotations

This script downloads Cell Painting data for compounds that are:
1. Treatments (not controls like DMSO, empty wells, or positive controls)
2. Annotated in the Repurposing Hub with target and mechanism of action information

The Repurposing Hub contains FDA-approved drugs and clinical candidates with known
targets and mechanisms - ideal for drug repurposing and supervised learning.

Dataset Stats:
- Treatment compounds with Repurposing Hub annotations: 2,721
- Total wells: ~75,000
- Total sites: ~450,000 (6 sites per well)

Quick Start:
    # Preview what will be downloaded (no actual download)
    python download_annotated_samples.py --dry-run

    # Download 10 sample compounds for testing
    python download_annotated_samples.py --sample 10

    # Download all 2,721 annotated compounds
    python download_annotated_samples.py

    # Filter by target (e.g., kinase inhibitors)
    python download_annotated_samples.py --target kinase --sample 20
"""

import sys
import argparse
import logging
from pathlib import Path

import duckdb
import pandas as pd
from converter.download_manager import DownloadManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_repurposing_hub_treatments(
    db_path: str,
    max_compounds: int = None,
    target_keyword: str = None
) -> pd.DataFrame:
    """
    Query for treatment compounds with Repurposing Hub annotations.

    Args:
        db_path: Path to DuckDB metadata database
        max_compounds: Optional limit on number of compounds (for testing)
        target_keyword: Optional keyword to filter by target/MOA (e.g., "kinase")

    Returns:
        DataFrame with compound information including:
        - Metadata_JCP2022: Compound ID
        - Metadata_repurposing_name: Drug name
        - Metadata_repurposing_target: Known target(s)
        - Metadata_repurposing_moa: Mechanism of action
        - Metadata_SMILES: Chemical structure
    """
    con = duckdb.connect(db_path, read_only=True)

    # Query for compounds that:
    # 1. Are in the repurposing_hub_targets table (have annotations)
    # 2. Are NOT in perturbation_control (i.e., are treatments, not controls)
    query = """
    SELECT DISTINCT
        c.Metadata_JCP2022,
        rh.Metadata_repurposing_name,
        rh.Metadata_repurposing_target,
        rh.Metadata_repurposing_moa,
        c.Metadata_SMILES
    FROM compound c
    JOIN repurposing_hub_targets rh
        ON c.Metadata_JCP2022 = rh.Metadata_JCP2022
    WHERE c.Metadata_JCP2022 NOT IN (
        SELECT Metadata_JCP2022
        FROM perturbation_control
    )
    ORDER BY c.Metadata_JCP2022
    """

    if max_compounds:
        query += f" LIMIT {max_compounds}"

    df = con.execute(query).fetchdf()
    con.close()

    # Filter by target keyword if provided
    if target_keyword:
        mask = (
            df['Metadata_repurposing_target'].str.contains(
                target_keyword, case=False, na=False
            ) |
            df['Metadata_repurposing_moa'].str.contains(
                target_keyword, case=False, na=False
            )
        )
        df = df[mask]
        logger.info(f"Filtered to {len(df)} compounds with '{target_keyword}' in target/MOA")

    logger.info(f"Found {len(df)} treatment compounds with Repurposing Hub annotations")

    return df


def get_wells_for_compounds(db_path: str, jcp_ids: list) -> pd.DataFrame:
    """
    Get well-level metadata for specific compounds.

    Args:
        db_path: Path to DuckDB metadata database
        jcp_ids: List of JCP2022 compound IDs

    Returns:
        DataFrame with well-level information including plate, well, source, batch
    """
    con = duckdb.connect(db_path, read_only=True)

    # Convert list to SQL IN clause
    ids_str = "','".join(jcp_ids)

    query = f"""
    SELECT
        w.Metadata_Source,
        p.Metadata_Batch,
        w.Metadata_Plate,
        w.Metadata_Well,
        w.Metadata_JCP2022,
        p.Metadata_PlateType
    FROM well w
    JOIN plate p ON w.Metadata_Plate = p.Metadata_Plate
    WHERE w.Metadata_JCP2022 IN ('{ids_str}')
    ORDER BY w.Metadata_Plate, w.Metadata_Well
    """

    df = con.execute(query).fetchdf()
    con.close()

    n_wells = len(df)
    n_plates = df['Metadata_Plate'].nunique()

    logger.info(
        f"Found {n_wells:,} wells across {n_plates:,} plates "
        f"for {len(jcp_ids):,} compounds"
    )

    return df


def save_manifest(compounds_df: pd.DataFrame, wells_df: pd.DataFrame, output_dir: Path):
    """Save compound and well manifests to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save compound information
    compound_manifest_path = output_dir / "repurposing_hub_compounds.csv"
    compounds_df.to_csv(compound_manifest_path, index=False)
    logger.info(f"Saved compound manifest: {compound_manifest_path}")

    # Save well-level information
    wells_manifest_path = output_dir / "repurposing_hub_wells.csv"
    wells_df.to_csv(wells_manifest_path, index=False)
    logger.info(f"Saved wells manifest: {wells_manifest_path}")


def print_summary(compounds_df: pd.DataFrame, wells_df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total compounds:     {len(compounds_df):,}")
    print(f"Total wells:         {len(wells_df):,}")
    print(f"Total plates:        {wells_df['Metadata_Plate'].nunique():,}")
    print(f"Wells per compound:  {len(wells_df) / len(compounds_df):.1f} (average)")

    # Estimate download size
    n_sites = len(wells_df) * 6  # 6 sites per well
    size_gb = (n_sites * 14.4) / 1024  # 14.4 MB per site (TIFF)

    print(f"\nEstimated download:")
    print(f"  Sites:      {n_sites:,}")
    print(f"  Size:       {size_gb:.1f} GB (TIFF format)")
    print(f"  Time:       ~{size_gb * 1024 / 100 / 60:.1f} minutes @ 100 MB/s")

    # Show plate type distribution
    print("\nPlate type distribution:")
    plate_dist = wells_df.groupby('Metadata_PlateType').agg({
        'Metadata_Plate': 'nunique',
        'Metadata_Well': 'count'
    }).rename(columns={'Metadata_Plate': 'num_plates', 'Metadata_Well': 'num_wells'})
    print(plate_dist)

    # Show top targets
    print("\nTop 10 most common targets:")
    target_counts = compounds_df['Metadata_repurposing_target'].value_counts().head(10)
    for i, (target, count) in enumerate(target_counts.items(), 1):
        print(f"  {i:2d}. {target}: {count} compounds")


def main():
    parser = argparse.ArgumentParser(
        description="Download JUMP compounds with Repurposing Hub annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what will be downloaded (no actual download)
  python download_annotated_samples.py --dry-run

  # Download 10 sample compounds for testing
  python download_annotated_samples.py --sample 10

  # Download all 2,721 annotated compounds
  python download_annotated_samples.py

  # Filter by target (e.g., kinase inhibitors)
  python download_annotated_samples.py --target kinase --sample 20

  # Custom output directory
  python download_annotated_samples.py --output ./my_data --sample 5
        """
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default='/data/users/jfredinh/addon_final/jump_production/data/interim/jump_metadata_augmented.duckdb',
        help='Path to DuckDB metadata database'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./data',
        help='Output directory for downloaded data (default: ./data)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        help='Number of compounds to download (for testing). Omit to download all 2,721.'
    )
    parser.add_argument(
        '--target',
        type=str,
        help='Filter compounds by target/MOA keyword (e.g., "kinase", "EGFR")'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=16,
        help='Number of parallel download workers (default: 16)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Query and show what will be downloaded, but do not download'
    )

    args = parser.parse_args()

    # Validate database path
    if not Path(args.db_path).exists():
        logger.error(f"Database not found: {args.db_path}")
        logger.error("Please update --db-path to point to your metadata database.")
        sys.exit(1)

    print("=" * 80)
    print("JUMP Repurposing Hub Annotated Compounds Download")
    print("=" * 80)

    # Step 1: Query compounds
    print("\n[Step 1/3] Querying Repurposing Hub compounds...")
    compounds_df = get_repurposing_hub_treatments(
        db_path=args.db_path,
        max_compounds=args.sample,
        target_keyword=args.target
    )

    if len(compounds_df) == 0:
        logger.error("No compounds found matching criteria")
        sys.exit(1)

    print(f"\nFound {len(compounds_df)} compounds:")
    print(compounds_df[['Metadata_JCP2022', 'Metadata_repurposing_name',
                        'Metadata_repurposing_target']].head(10))
    if len(compounds_df) > 10:
        print(f"... and {len(compounds_df) - 10} more")

    # Step 2: Get well-level information
    print("\n[Step 2/3] Getting well-level metadata...")
    jcp_ids = compounds_df['Metadata_JCP2022'].tolist()
    wells_df = get_wells_for_compounds(args.db_path, jcp_ids)

    # Save manifests
    output_dir = Path(args.output) / "manifests"
    save_manifest(compounds_df, wells_df, output_dir)

    # Print summary
    print_summary(compounds_df, wells_df)

    # Step 3: Download (unless dry-run)
    if args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN - No download performed")
        print("=" * 80)
        print("\nTo download, run the same command without --dry-run")
        return

    print("\n[Step 3/3] Downloading from S3...")
    print("-" * 80)

    cache_dir = Path(args.output) / "repurposing_hub_tiff"

    manager = DownloadManager(
        db_path=args.db_path,
        cache_dir=str(cache_dir),
        max_workers=args.workers,
    )

    plates = manager.download_by_compounds(
        jcp_ids=jcp_ids,
        force=False,
        show_progress=True,
    )

    manager.close()

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"Downloaded wells from {len(plates)} plates")
    print(f"Cache directory: {cache_dir}")

    # Get cache summary
    total_size_mb = sum(
        f.stat().st_size for f in cache_dir.rglob("*.tiff") if f.is_file()
    ) / (1024**2)

    print(f"Total size: {total_size_mb / 1024:.1f} GB")

    print("\nOutput files:")
    print(f"  Manifests:  {output_dir}")
    print(f"  TIFF data:  {cache_dir}")

    print("\nNext steps:")
    print("  1. Review manifests in", output_dir)
    print("  2. Use the TIFF files for analysis")
    print("  3. Optionally convert to Zarr for PyTorch (see README)")


if __name__ == "__main__":
    main()
