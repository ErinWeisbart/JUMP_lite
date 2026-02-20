#!/usr/bin/env python3
"""
Validate that per-cell features align with segmentation masks.

Loads features from profiles parquet files and masks from npz files,
matches them by cell label, and validates by comparing area measurements.

Usage:
    python validate_feature_mask_alignment.py
    python validate_feature_mask_alignment.py --source-id <source_id> --codec zstd.zarr
"""

import argparse
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path


def load_mask(path: Path) -> np.ndarray:
    """Load segmentation mask from npz file."""
    data = np.load(path)
    return np.squeeze(data[list(data.keys())[0]]).astype(np.int32)


def load_cell_features(path: Path, object_type: str = "cell") -> pl.DataFrame:
    """Load per-cell features, pivot to one row per cell."""
    df = pl.read_parquet(path)
    # Filter to object type and first branch (values are same across branches)
    df = df.filter(
        (pl.col("object") == object_type) &
        (pl.col("branch") == "0/max/sizeshape")
    )
    # Pivot: one row per label, one column per metric
    return df.pivot(on="metric", index="label", values="value")


def compute_mask_areas(mask: np.ndarray) -> pl.DataFrame:
    """Compute pixel area for each cell in mask."""
    labels = np.unique(mask)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return pl.DataFrame({"label": pl.Series([], dtype=pl.Int64), "mask_area": pl.Series([], dtype=pl.Int64)})
    areas = [(int(lbl), int((mask == lbl).sum())) for lbl in labels]
    return pl.DataFrame(areas, schema={"label": pl.Int64, "mask_area": pl.Int64}, orient="row")


def validate_alignment(
    features_path: Path,
    mask_path: Path,
    object_type: str = "cell"
) -> dict:
    """Validate feature-mask alignment by comparing areas."""
    # Load data
    features = load_cell_features(features_path, object_type)
    mask = load_mask(mask_path)
    mask_areas = compute_mask_areas(mask)

    n_features = len(features)
    n_mask = len(mask_areas)

    # Handle empty cases
    if n_features == 0 or n_mask == 0:
        return {
            "n_cells_features": n_features,
            "n_cells_mask": n_mask,
            "n_matched": 0,
            "correlation": float("nan"),
            "exact_match": n_features == 0 and n_mask == 0,
            "feature_area": np.array([]),
            "mask_area": np.array([]),
        }

    # Cast label to same type for join
    features = features.with_columns(pl.col("label").cast(pl.Int64))

    # Join on label
    merged = features.join(mask_areas, on="label")

    if len(merged) == 0:
        return {
            "n_cells_features": n_features,
            "n_cells_mask": n_mask,
            "n_matched": 0,
            "correlation": float("nan"),
            "exact_match": False,
            "feature_area": np.array([]),
            "mask_area": np.array([]),
        }

    # Compare areas
    feature_area = merged["Area"].to_numpy()
    mask_area = merged["mask_area"].to_numpy()

    correlation = np.corrcoef(feature_area, mask_area)[0, 1]
    exact_match = np.allclose(feature_area, mask_area)

    return {
        "n_cells_features": n_features,
        "n_cells_mask": n_mask,
        "n_matched": len(merged),
        "correlation": correlation,
        "exact_match": exact_match,
        "feature_area": feature_area,
        "mask_area": mask_area,
    }


def plot_correlation(results: list[dict], source_ids: list[str], output_path: Path):
    """Plot area correlation for all samples."""
    n_samples = len(results)

    if n_samples == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        axes = [ax]
    else:
        n_cols = min(3, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = axes.flatten() if n_samples > 1 else [axes]

    for i, (result, source_id) in enumerate(zip(results, source_ids)):
        ax = axes[i]

        if len(result["feature_area"]) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(source_id[:30] + "..." if len(source_id) > 30 else source_id)
            continue

        feature_area = result["feature_area"]
        mask_area = result["mask_area"]

        ax.scatter(mask_area, feature_area, alpha=0.6, s=20, edgecolors="none")

        # Plot y=x line
        max_val = max(mask_area.max(), feature_area.max())
        ax.plot([0, max_val], [0, max_val], "r--", lw=1, label="y=x")

        ax.set_xlabel("Mask Area (pixels)")
        ax.set_ylabel("Feature Area")
        ax.set_title(f"{source_id[:25]}...\nr={result['correlation']:.4f}", fontsize=10)
        ax.set_aspect("equal", adjustable="box")

    # Hide unused axes
    for i in range(n_samples, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Feature Area vs Mask Area Correlation", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved correlation plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate feature-mask alignment")
    parser.add_argument("--base-path", type=str,
                        default="/work/datasets/aliby_output/cp_measure/jump_target2_4plate",
                        help="Base path for aliby output")
    parser.add_argument("--codec", type=str, default="zstd.zarr",
                        help="Codec/compression name")
    parser.add_argument("--source-id", type=str, default=None,
                        help="Specific source ID (random if not provided)")
    parser.add_argument("--object-type", type=str, default="cell",
                        choices=["cell", "nuclei"], help="Object type to validate")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of random samples to validate")
    parser.add_argument("--output", type=str, default="analysis/output/area_correlation.png",
                        help="Output path for correlation plot")

    args = parser.parse_args()
    base = Path(args.base_path) / args.codec

    # Find available source IDs
    profiles_dir = base / "profiles"
    if args.source_id:
        source_ids = [args.source_id]
    else:
        parquet_files = list(profiles_dir.glob("*.parquet"))
        source_ids = [p.stem for p in parquet_files]
        # Random sample
        import random
        source_ids = random.sample(source_ids, min(args.n_samples, len(source_ids)))

    segment_step = "segment_cell" if args.object_type == "cell" else "segment_nuclei"

    print(f"Validating {len(source_ids)} samples for {args.codec}...")
    print(f"Object type: {args.object_type}")
    print("-" * 60)

    all_pass = True
    all_results = []
    valid_source_ids = []

    for source_id in source_ids:
        features_path = profiles_dir / f"{source_id}.parquet"
        mask_dir = base / "steps" / source_id / segment_step
        mask_files = list(mask_dir.glob("*.npz"))

        if not mask_files:
            print(f"{source_id}: No mask files found")
            continue

        mask_path = mask_files[0]  # Use first tile
        result = validate_alignment(features_path, mask_path, args.object_type)

        all_results.append(result)
        valid_source_ids.append(source_id)

        status = "PASS" if result["exact_match"] else "FAIL"
        if not result["exact_match"]:
            all_pass = False

        print(f"{source_id}:")
        print(f"  Cells: features={result['n_cells_features']}, mask={result['n_cells_mask']}, matched={result['n_matched']}")
        print(f"  Area correlation: {result['correlation']:.6f}")
        print(f"  Exact match: {status}")

    print("-" * 60)
    print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")

    # Generate correlation plot
    if all_results:
        plot_correlation(all_results, valid_source_ids, Path(args.output))


if __name__ == "__main__":
    main()
