#!/usr/bin/env python3
"""Compute per-well phenotypic-activity AP for one (model, codec).

Calls ``calculate_phenotypic_activity`` and persists the ``activity_ap`` frame
returned by copairs — one row per (Source, Batch, Plate, Well, Site, JCP2022)
with ``normalized_average_precision``.
"""
import argparse
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from norm_3.metrics import calculate_phenotypic_activity  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, help="Best-config feature parquet")
    p.add_argument("--codec", required=True,
                   help="Value of Metadata_compression to keep (e.g. jpegxl_lossy_hq.zarr)")
    p.add_argument("--output", required=True, help="Per-well AP parquet")
    p.add_argument("--cell-count-parquet",
                   default="src/norm_3/data/features/best_configs_lite/cell_count__default.parquet",
                   help="Cell-count features parquet (used for tox-bias filter)")
    args = p.parse_args()

    df = pl.read_parquet(args.features).filter(pl.col("Metadata_compression") == args.codec)
    if len(df) == 0:
        raise SystemExit(f"no rows for codec={args.codec} in {args.features}")
    features = [c for c in df.columns if not c.startswith("Metadata_")]
    print(f"{len(df)} rows × {len(features)} features for codec={args.codec}")

    res = calculate_phenotypic_activity(
        df, features,
        compound_col="Metadata_JCP2022",
        negcon_col="Metadata_negcon",
        group_col="Metadata_Group",
    )

    ap = pl.from_pandas(res["activity_ap"])
    cmap = pl.from_pandas(
        res["activity_map"][[
            "Metadata_JCP2022", "Metadata_Group", "Metadata_reference_index",
            "mean_normalized_average_precision", "p_value", "corrected_p_value",
            "below_corrected_p", "n_replicate_pairs",
        ]]
    ).rename({"mean_normalized_average_precision": "compound_mean_nap"})
    join_cols = ["Metadata_JCP2022", "Metadata_Group", "Metadata_reference_index"]
    ap = ap.join(cmap, on=join_cols, how="left")

    # Per-well L2 distance to the plate-level DMSO median (phenotype magnitude).
    print("computing per-well DMSO distance...")
    import numpy as np
    feat_arr = df.select(features).to_numpy()
    plate_arr = df["Metadata_Plate"].to_numpy()
    negcon_arr = df["Metadata_negcon"].to_numpy()
    plate_medians: dict[str, np.ndarray] = {}
    for plate in np.unique(plate_arr):
        mask = (plate_arr == plate) & (negcon_arr == True)
        if mask.sum() > 0:
            plate_medians[plate] = np.median(feat_arr[mask], axis=0)
    # Default to global DMSO median for plates without negcons (rare).
    global_med = np.median(feat_arr[negcon_arr == True], axis=0)
    median_lookup = np.stack([plate_medians.get(p, global_med) for p in plate_arr])
    dist = np.linalg.norm(feat_arr - median_lookup, axis=1)
    well_keys = df.select([
        "Metadata_Source", "Metadata_Batch", "Metadata_Plate",
        "Metadata_Well", "Metadata_Site",
    ]).with_columns(pl.Series("dmso_distance", dist))
    well_keys = well_keys.with_columns([pl.col(c).cast(pl.Utf8)
                                        for c in well_keys.columns
                                        if c != "dmso_distance"])
    ap = ap.with_columns([pl.col(c).cast(pl.Utf8)
                          for c in ["Metadata_Source", "Metadata_Batch",
                                    "Metadata_Plate", "Metadata_Well", "Metadata_Site"]])
    ap = ap.join(
        well_keys,
        on=["Metadata_Source", "Metadata_Batch", "Metadata_Plate",
            "Metadata_Well", "Metadata_Site"],
        how="left",
    )

    # Per-well cell_count_ratio = cell_count / plate-DMSO-median cell_count.
    # Used downstream as a tox filter: ratios near 1.0 = healthy population,
    # << 1.0 = cytotoxic, >> 1.0 = over-proliferative.
    print("computing per-well cell_count_ratio...")
    cc = pl.read_parquet(args.cell_count_parquet).select([
        "Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well",
        "Metadata_negcon", "cell_count",
    ])
    plate_med = (cc.filter(pl.col("Metadata_negcon") == True)
                   .group_by("Metadata_Plate")
                   .agg(pl.col("cell_count").median().alias("plate_dmso_cell_count")))
    cc = cc.join(plate_med, on="Metadata_Plate", how="left")
    cc = cc.with_columns(
        (pl.col("cell_count") / pl.col("plate_dmso_cell_count")).alias("cell_count_ratio")
    ).select([
        "Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well",
        "cell_count", "cell_count_ratio",
    ])
    cc = cc.with_columns([pl.col(c).cast(pl.Utf8)
                          for c in ["Metadata_Source", "Metadata_Batch",
                                    "Metadata_Plate", "Metadata_Well"]])
    # cell_count is well-aggregated (no Site), so multiple ap rows map to the
    # same cell_count via (Source, Batch, Plate, Well).
    cc = cc.unique(subset=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"])
    ap = ap.join(cc,
                 on=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"],
                 how="left")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ap.write_parquet(out)
    print(f"wrote {out}  ({len(ap)} wells; cols include corrected_p_value, "
          f"n_replicate_pairs, dmso_distance, cell_count_ratio)")


if __name__ == "__main__":
    main()
