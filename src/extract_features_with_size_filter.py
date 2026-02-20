"""
Feature Extraction Script with Size Filtering

Extracts features from aliby_output profiles and creates raw_features.parquet files.
Supports both CellProfiler measurements and embedding-based features (e.g., dinov2).

NEW: Added size-based filtering to match JUMP quality standards:
- Nuclei: diameter >= 15 pixels (area >= 177 pixels)
- Cells: diameter >= 25 pixels (area >= 491 pixels)

Usage:
    # With JUMP-standard size filtering
    python src/extract_features_with_size_filter.py --input /work/datasets/aliby_output --output ./output --filter-size

    # Process specific model with size filtering
    python src/extract_features_with_size_filter.py --input /work/datasets/aliby_output --model cp_measure --output ./output --filter-size
"""

import argparse
import warnings
from pathlib import Path

import duckdb
import polars as pl
from polars import selectors as cs
from trommel.core import basic_cleanup


def discover_profile_directories(
    base_dir: Path,
    model_filter: str | None = None,
    compression_filter: str | None = None,
    dataset_filter: str | None = None,
) -> list[dict]:
    """
    Discover all MODEL/DATASET/COMPRESSION/profiles directories.

    Args:
        base_dir: Base aliby_output directory
        model_filter: Optional filter for specific model (e.g., cp_measure, dinov2)
        compression_filter: Optional filter for specific compression (e.g., zstd.zarr)
        dataset_filter: Optional filter for specific dataset (e.g., jump_target2_4plate)

    Returns:
        List of dicts with keys: model, dataset, compression, profiles_path
    """
    results = []

    # Iterate over model directories (cp_measure, dinov2, etc.)
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
        # Skip cache directories
        if "cache" in model_dir.name.lower():
            continue

        model_name = model_dir.name
        if model_filter and model_name != model_filter:
            continue

        # Iterate over dataset directories
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name
            if dataset_filter and dataset_name != dataset_filter:
                continue

            # Iterate over compression directories (*.zarr)
            for compression_dir in dataset_dir.iterdir():
                if not compression_dir.is_dir():
                    continue
                if not compression_dir.name.endswith(".zarr"):
                    continue

                compression_name = compression_dir.name
                if compression_filter and compression_name != compression_filter:
                    continue

                # Check for profiles subdirectory
                profiles_path = compression_dir / "profiles"
                if profiles_path.exists() and profiles_path.is_dir():
                    results.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "compression": compression_name,
                        "profiles_path": profiles_path,
                    })

    return results


def get_features(
    profiles_dir: Path,
    cache_dir: Path | None = None,
    filter_border_cells: bool = False,
    filter_by_size: bool = False,
    min_nuclei_diameter: float = 15.0,
    min_cell_diameter: float = 25.0,
) -> pl.DataFrame:
    """
    Extract and process features from parquet files.

    For CellProfiler data with filtering:
      Pass 1: Scan only filter-relevant metrics (EquivalentDiameter, BoundingBox*)
              with parquet predicate pushdown to build a small valid_labels table.
      Pass 2: Stream read_parquet → SEMI JOIN valid_labels → GROUP BY → well_level
              in a single pipelined query (no intermediate raw table).

    Args:
        profiles_dir: Path to profiles directory containing parquet files
        cache_dir: Optional path to cache directory for intermediate databases
        filter_border_cells: If True, exclude cells touching image borders (default: False)
        filter_by_size: If True, filter objects by minimum size (JUMP standards)
        min_nuclei_diameter: Minimum nuclei diameter in pixels (default: 15, JUMP standard)
        min_cell_diameter: Minimum cell diameter in pixels (default: 25, empirical from raw CP)

    Returns:
        Polars DataFrame with pivoted well-level features and cell counts
    """
    parquet_glob = profiles_dir / "*.parquet"

    # Peek at schema
    con = duckdb.connect()
    schema_rel = con.sql(f"SELECT * FROM read_parquet('{parquet_glob}', filename=true) LIMIT 0")
    columns = [col[0] for col in schema_rel.description]
    is_cp = "branch" in columns and "metric" in columns and "object" in columns
    has_label = "label" in columns
    has_values_list = False
    if "values" in columns:
        vtype = [x[1] for x in schema_rel.description if x[0] == "values"]
        has_values_list = bool(vtype) and vtype[0] == "list"
    con.close()

    if not is_cp:
        return _get_features_embeddings(profiles_dir, columns)

    # Source expression: handles both 'value' column and UNNEST(values) case
    if has_values_list:
        src_expr = f"(SELECT *, UNNEST(values) AS value FROM read_parquet('{parquet_glob}', filename=true))"
        filter_value_col = "UNNEST(values)"
    else:
        src_expr = f"read_parquet('{parquet_glob}', filename=true)"
        filter_value_col = "value"

    # SQL expressions for deriving well_id from filename
    site_fn = "parse_filename(r.filename, true)"
    well_expr = (
        f"string_split({site_fn}, '__')[1] || '__' || "
        f"string_split({site_fn}, '__')[2] || '__' || "
        f"string_split({site_fn}, '__')[3] || '__' || "
        f"string_split({site_fn}, '__')[4]"
    )

    needs_filtering = (filter_by_size or filter_border_cells) and has_label

    con = duckdb.connect()
    try:
        # === Pass 1: Build valid_labels from filter metrics (tiny scan) ===
        join_clause = ""
        if needs_filtering:
            filter_preds = []
            if filter_by_size:
                filter_preds.append("metric = 'EquivalentDiameter'")
            if filter_border_cells:
                filter_preds.append("metric LIKE 'BoundingBox%'")
            where = " OR ".join(f"({p})" for p in filter_preds)

            print(f"  Building filter index...", flush=True)
            con.sql(f"""
                CREATE TABLE filter_rows AS
                SELECT filename, label, object, metric, {filter_value_col} AS value
                FROM read_parquet('{parquet_glob}', filename=true)
                WHERE {where}
            """)

            if filter_by_size:
                print(f"  Size filter: nuclei >= {min_nuclei_diameter}px, cells >= {min_cell_diameter}px", flush=True)
                con.sql(f"""
                    CREATE TABLE size_valid AS
                    SELECT DISTINCT filename, label, object
                    FROM filter_rows
                    WHERE metric = 'EquivalentDiameter'
                      AND (
                        (object = 'nuclei' AND value >= {min_nuclei_diameter})
                        OR (object = 'cell' AND value >= {min_cell_diameter})
                        OR (object NOT IN ('nuclei', 'cell'))
                      )
                """)

            if filter_border_cells:
                print(f"  Border filter: excluding edge-touching objects", flush=True)
                con.sql("""
                    CREATE TABLE bbox_per_label AS
                    SELECT filename, label, object,
                        MAX(CASE WHEN metric = 'BoundingBoxMinimum_X' THEN value END) AS min_x,
                        MAX(CASE WHEN metric = 'BoundingBoxMinimum_Y' THEN value END) AS min_y,
                        MAX(CASE WHEN metric = 'BoundingBoxMaximum_X' THEN value END) AS max_x,
                        MAX(CASE WHEN metric = 'BoundingBoxMaximum_Y' THEN value END) AS max_y
                    FROM filter_rows
                    WHERE metric LIKE 'BoundingBox%'
                    GROUP BY filename, label, object
                """)
                con.sql("""
                    CREATE TABLE border_valid AS
                    SELECT DISTINCT b.filename, b.label, b.object
                    FROM bbox_per_label b
                    JOIN (
                        SELECT filename, MAX(max_x) AS img_w, MAX(max_y) AS img_h
                        FROM bbox_per_label GROUP BY filename
                    ) d ON b.filename = d.filename
                    WHERE b.min_x > 0 AND b.min_y > 0
                      AND b.max_x < d.img_w AND b.max_y < d.img_h
                """)

            if filter_by_size and filter_border_cells:
                con.sql("""
                    CREATE TABLE valid_labels AS
                    SELECT s.filename, s.label, s.object
                    FROM size_valid s
                    SEMI JOIN border_valid b
                      ON s.filename = b.filename AND s.label = b.label AND s.object = b.object
                """)
            elif filter_by_size:
                con.sql("ALTER TABLE size_valid RENAME TO valid_labels")
            else:
                con.sql("ALTER TABLE border_valid RENAME TO valid_labels")

            n_valid = con.sql("SELECT COUNT(*) FROM valid_labels").fetchone()[0]
            print(f"  {n_valid} valid (filename, label, object) tuples", flush=True)

            join_clause = """
                SEMI JOIN valid_labels v
                  ON r.filename = v.filename AND r.label = v.label AND r.object = v.object
            """

        # === Pass 2: Stream read → filter → aggregate (single pipelined query) ===
        print(f"  Aggregating to well level...", flush=True)
        con.sql(f"""
            CREATE TABLE well_level AS
            SELECT
                {well_expr} AS well_id,
                r.branch || r.metric AS full_metric_name,
                r.object,
                r.tp,
                median(r.value) AS cvalue
            FROM {src_expr} r
            {join_clause}
            GROUP BY r.tp, {well_expr}, r.branch, r.metric, r.object
        """)

        # Cell counts (from valid_labels if available — no extra parquet read)
        cell_counts_pl = None
        if has_label:
            if needs_filtering:
                vl_site = "parse_filename(filename, true)"
                vl_well = (
                    f"string_split({vl_site}, '__')[1] || '__' || "
                    f"string_split({vl_site}, '__')[2] || '__' || "
                    f"string_split({vl_site}, '__')[3] || '__' || "
                    f"string_split({vl_site}, '__')[4]"
                )
                cell_counts_pl = con.sql(f"""
                    SELECT
                        {vl_well} AS well_id,
                        COUNT(DISTINCT CASE WHEN object = 'cell' THEN label END) AS Metadata_n_cells,
                        COUNT(DISTINCT CASE WHEN object = 'nuclei' THEN label END) AS Metadata_n_nuclei
                    FROM valid_labels
                    GROUP BY {vl_well}
                """).pl()
            else:
                cell_counts_pl = con.sql(f"""
                    SELECT
                        {well_expr} AS well_id,
                        COUNT(DISTINCT CASE WHEN r.object = 'cell' THEN r.label END) AS Metadata_n_cells,
                        COUNT(DISTINCT CASE WHEN r.object = 'nuclei' THEN r.label END) AS Metadata_n_nuclei
                    FROM {src_expr} r
                    GROUP BY {well_expr}
                """).pl()

        # Pivot to wide format
        print(f"  Pivoting...", flush=True)
        pivoted_pl = con.sql(
            "PIVOT well_level ON object, full_metric_name USING any_value(cvalue)"
        ).pl()

        if cell_counts_pl is not None:
            pivoted_pl = pivoted_pl.join(cell_counts_pl, on="well_id", how="left")

    finally:
        con.close()

    return pivoted_pl


def _get_features_embeddings(profiles_dir: Path, columns: list[str]) -> pl.DataFrame:
    """Process embedding-based features (no filtering needed, load in bulk)."""
    parquet_glob = profiles_dir / "*.parquet"
    site_col = "site"
    tp_name = "tp"

    con = duckdb.connect()
    try:
        site_fn = "parse_filename(filename, true)"
        well_expr = (
            f"string_split({site_fn}, '__')[1] || '__' || "
            f"string_split({site_fn}, '__')[2] || '__' || "
            f"string_split({site_fn}, '__')[3] || '__' || "
            f"string_split({site_fn}, '__')[4]"
        )
        feature_cols = [
            col for col in columns
            if col not in [site_col, "filename", tp_name, "well_id"]
        ]
        if feature_cols:
            agg_exprs = ", ".join([f"mean({c}) AS {c}" for c in feature_cols])
            return con.sql(f"""
                SELECT {well_expr} AS well_id, {agg_exprs}
                FROM read_parquet('{parquet_glob}', filename=true)
                GROUP BY {well_expr}
            """).pl()
        else:
            return con.sql(f"""
                SELECT *, {well_expr} AS well_id
                FROM read_parquet('{parquet_glob}', filename=true)
            """).pl()
    finally:
        con.close()


def extract_metadata_from_site(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract metadata columns from well_id string.

    Well ID format: source__batch__plate__well
    """
    df = df.with_columns([
        pl.col("well_id").str.split("__").list.get(0).alias("Metadata_Source"),
        pl.col("well_id").str.split("__").list.get(1).alias("Metadata_Batch"),
        pl.col("well_id").str.split("__").list.get(2).alias("Metadata_Plate"),
        pl.col("well_id").str.split("__").list.get(3).alias("Metadata_Well"),
        pl.lit("1").alias("Metadata_Site"),
    ])

    # Rename well_id and tp to have Metadata_ prefix
    df = df.rename({"well_id": "Metadata_id"})
    if "tp" in df.columns:
        df = df.rename({"tp": "Metadata_tp"})

    return df


def process_profiles(
    profiles_info: dict,
    output_dir: Path,
    cache_dir: Path | None = None,
    filter_border_cells: bool = False,
    filter_by_size: bool = False,
    min_nuclei_diameter: float = 15.0,
    min_cell_diameter: float = 25.0,
) -> Path | None:
    """
    Process a single profiles directory and save raw_features.parquet.
    """
    model = profiles_info["model"]
    dataset = profiles_info["dataset"]
    compression = profiles_info["compression"]
    profiles_path = profiles_info["profiles_path"]

    # Create output filename with filter suffix
    compression_clean = compression.replace(".zarr", "")
    filter_parts = []
    if filter_border_cells:
        filter_parts.append("border")
    if filter_by_size:
        filter_parts.append("size")
    filter_suffix = f"_filtered_{'_'.join(filter_parts)}" if filter_parts else ""
    output_filename = f"{model}_{dataset}_{compression_clean}{filter_suffix}_raw_features.parquet"
    output_path = output_dir / output_filename

    print(f"\nProcessing: {model}/{dataset}/{compression}")
    print(f"  Input: {profiles_path}")

    try:
        # Extract features
        df = get_features(
            profiles_path,
            cache_dir,
            filter_border_cells,
            filter_by_size,
            min_nuclei_diameter,
            min_cell_diameter,
        )
        print(f"  Extracted features: {df.shape}")
        if filter_by_size:
            print(f"  Size filtering: nuclei >= {min_nuclei_diameter}px, cells >= {min_cell_diameter}px")
        if filter_border_cells:
            print(f"  Border filtering: enabled")

        # Extract metadata from site column
        df = extract_metadata_from_site(df)

        # Add model/compression info as metadata
        df = df.with_columns([
            pl.lit(model).alias("Metadata_model"),
            pl.lit(dataset).alias("Metadata_dataset"),
            pl.lit(compression).alias("Metadata_compression"),
        ])

        # Save to parquet
        df.write_parquet(output_path)
        print(f"  Output: {output_path}")
        print(f"  Shape: {df.shape}")

        return output_path

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point for feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract features from aliby_output profiles with optional size filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With JUMP-standard size filtering
  python src/extract_features_with_size_filter.py --input /work/datasets/aliby_output --output ./output --filter-size

  # Process specific model with size filtering
  python src/extract_features_with_size_filter.py --input /work/datasets/aliby_output --model cp_measure --output ./output --filter-size

  # Custom size thresholds
  python src/extract_features_with_size_filter.py --input /work/datasets/aliby_output --output ./output --filter-size --min-nuclei-diameter 17 --min-cell-diameter 26
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Base aliby_output directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for raw_features.parquet files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter for specific model (e.g., cp_measure, dinov2)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=None,
        help="Filter for specific compression (e.g., zstd.zarr, jpegxl_lossy_mq.zarr)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filter for specific dataset (e.g., jump_target2_4plate)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for intermediate databases",
    )
    parser.add_argument(
        "--filter-border-cells",
        action="store_true",
        help="Exclude cells touching image borders",
    )
    parser.add_argument(
        "--filter-size",
        action="store_true",
        help="Filter objects by minimum size (JUMP standards: nuclei >= 15px, cells >= 25px)",
    )
    parser.add_argument(
        "--min-nuclei-diameter",
        type=float,
        default=15.0,
        help="Minimum nuclei diameter in pixels (default: 15, JUMP standard)",
    )
    parser.add_argument(
        "--min-cell-diameter",
        type=float,
        default=25.0,
        help="Minimum cell diameter in pixels (default: 25, from raw CellProfiler)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Discover profile directories
    print(f"Discovering profile directories in: {args.input}")
    profiles_list = discover_profile_directories(
        args.input,
        model_filter=args.model,
        compression_filter=args.compression,
        dataset_filter=args.dataset,
    )

    print(f"\nFound {len(profiles_list)} profile directories:")
    for p in profiles_list:
        print(f"  - {p['model']}/{p['dataset']}/{p['compression']}")

    # Process each profile directory
    results = {
        "successful": [],
        "failed": [],
    }

    for profiles_info in profiles_list:
        output_path = process_profiles(
            profiles_info,
            args.output,
            args.cache_dir,
            args.filter_border_cells,
            args.filter_size,
            args.min_nuclei_diameter,
            args.min_cell_diameter,
        )

        if output_path:
            results["successful"].append(profiles_info)
        else:
            results["failed"].append(profiles_info)

    # Print summary
    print("\n" + "="*80)
    print("Feature extraction complete!")
    print(f"  Successful: {len(results['successful'])}")
    print(f"  Failed: {len(results['failed'])}")
    print(f"  Output directory: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
