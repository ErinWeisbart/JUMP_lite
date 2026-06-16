#!/usr/bin/env python3
"""Render a random well from a zarr store as an RGB PNG.

Usage:
  visualize_well.py [output] [--rank-by AP.parquet] [--top-pct N]

If --rank-by is given, picks a random well from the top-N% by
normalized_average_precision (default top 10%).
"""
import argparse
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tifffile

# Raw TIFFs (lossless) for jump_core_annotated. Names follow
#   {source}__{batch}__{plate}__{well}__{channel}__{site}__Orig.tif
# Within a well/site, channels sort alphabetically: AGP, DNA, ER, Mito, RNA
# (this is the same order that the MQ zarr was built with — see compress_tif.py:206).
RAW_DIR = "/work/datasets/jump_lite/jump_core_annotated/raw"


def load_well_image(source: str, batch: str, plate: str, well: str, site: str) -> np.ndarray:
    """Load a 5-channel uint16 image from raw TIFFs. Returns (5, H, W)."""
    pattern = f"{RAW_DIR}/{source}__{batch}__{plate}__{well}__*__{site}__Orig.tif"
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(pattern)
    return np.stack([tifffile.imread(f) for f in files], axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", nargs="?", default="aux_figures/random_well.png")
    p.add_argument("--rank-by", default=None,
                   help="Per-well AP parquet from analysis/well_activity.py")
    p.add_argument("--top-pct", type=float, default=10.0,
                   help="Sample only from top N percent of NAP (with --rank-by)")
    p.add_argument("--n", type=int, default=1,
                   help="Number of distinct wells to render (suffixes _01, _02, ...)")
    p.add_argument("--group", default=None,
                   help="Restrict to one Metadata_Group (e.g. group_high, group_low, group_orf, group_crispr)")
    p.add_argument("--target-nap", type=float, default=None,
                   help="If set, sample N random wells with NAP in [target±tol] instead of top-ranked")
    p.add_argument("--nap-tol", type=float, default=0.02)
    p.add_argument("--cc-min", type=float, default=0.5,
                   help="Min cell_count_ratio (tox filter: 0.5 = at least 50% of plate DMSO cell count)")
    p.add_argument("--cc-max", type=float, default=2.0,
                   help="Max cell_count_ratio (also rejects over-proliferative outliers)")
    p.add_argument("--negcon-features", default=None,
                   help="Sample N random negcon wells from this best-config feature parquet "
                        "(uses --group to filter). Bypasses --rank-by.")
    args = p.parse_args()

    pool_iter = None
    if args.negcon_features:
        feats = pl.read_parquet(
            args.negcon_features,
            columns=["Metadata_Source", "Metadata_Batch", "Metadata_Plate",
                     "Metadata_Well", "Metadata_Site", "Metadata_JCP2022",
                     "Metadata_Group", "Metadata_negcon"],
        ).filter(pl.col("Metadata_negcon") == True)
        if args.group:
            feats = feats.filter(pl.col("Metadata_Group") == args.group)
            if len(feats) == 0:
                raise SystemExit(f"no negcon rows for Metadata_Group={args.group}")
        pool_iter = iter(feats.sample(len(feats), shuffle=True).iter_rows(named=True))
    elif args.rank_by:
        ap = pl.read_parquet(args.rank_by).filter(
            pl.col("normalized_average_precision").is_not_null()
        )
        if args.group:
            ap = ap.filter(pl.col("Metadata_Group") == args.group)
            if len(ap) == 0:
                raise SystemExit(f"no rows for Metadata_Group={args.group}")
        if args.target_nap is not None:
            lo, hi = args.target_nap - args.nap_tol, args.target_nap + args.nap_tol
            # Combined ranking: significant compounds with substantial replicate
            # support (kills NAP=1.0-with-2-replicates noise), healthy cell
            # count (kills cytotoxic tox bias), in target NAP band, sorted by
            # phenotype magnitude (DMSO distance) descending.
            ranked = ap.filter(
                pl.col("normalized_average_precision").is_between(lo, hi)
                & (pl.col("corrected_p_value") <= 0.05)
                & (pl.col("n_replicate_pairs") >= 5)
                & (pl.col("cell_count_ratio").is_between(args.cc_min, args.cc_max))
            )
            if len(ranked) == 0:
                raise SystemExit(
                    f"no significant wells (p_corr<=0.05, n_reps>=5, "
                    f"cell_count_ratio in [{args.cc_min}, {args.cc_max}]) with "
                    f"NAP in [{lo:.3f}, {hi:.3f}] for group={args.group}"
                )
            ranked = ranked.sort("dmso_distance", descending=True, nulls_last=True)
        else:
            best_well_per_compound = (
                ap.sort(["normalized_average_precision", "corrected_p_value"],
                        descending=[True, False], nulls_last=True)
                  .group_by("Metadata_JCP2022", maintain_order=True)
                  .head(1)
            )
            ranked = best_well_per_compound.sort(
                ["compound_mean_nap", "corrected_p_value"],
                descending=[True, False], nulls_last=True,
            )
        pool_iter = iter(ranked.iter_rows(named=True))

    def well_files_exist(key: tuple[str, str, str, str, str]) -> bool:
        s, b, p, w, st = key
        return len(glob.glob(f"{RAW_DIR}/{s}__{b}__{p}__{w}__*__{st}__Orig.tif")) > 0

    def pick() -> tuple[tuple[str, str, str, str, str], str]:
        if pool_iter is not None:
            for row in pool_iter:
                key = (
                    str(row["Metadata_Source"]), str(row["Metadata_Batch"]),
                    str(row["Metadata_Plate"]), str(row["Metadata_Well"]),
                    str(row["Metadata_Site"]),
                )
                if not well_files_exist(key):
                    continue
                if "normalized_average_precision" in row:
                    p = row.get("corrected_p_value")
                    p_str = f"  p_corr={p:.2e}" if p is not None else ""
                    d = row.get("dmso_distance")
                    d_str = f"  Δdmso={d:.2f}" if d is not None else ""
                    n = row.get("n_replicate_pairs")
                    n_str = f"  n_reps={int(n)}" if n is not None else ""
                    cc = row.get("cell_count_ratio")
                    cc_str = f"  cc={cc:.2f}" if cc is not None else ""
                    lab = (
                        f"NAP={row['normalized_average_precision']:.3f}"
                        f"{p_str}{d_str}{n_str}{cc_str}  "
                        f"{row['Metadata_JCP2022']}"
                    )
                else:
                    lab = f"NEGCON  {row['Metadata_JCP2022']}  {row['Metadata_Group']}"
                return key, lab
            raise SystemExit("pool exhausted before N wells were found")
        # Random fallback: glob a TIFF and parse its name
        files = glob.glob(f"{RAW_DIR}/*__AGP__*__Orig.tif")
        f = random.choice(files)
        parts = os.path.basename(f).split("__")
        return (parts[0], parts[1], parts[2], parts[3], parts[5]), ""

    # Always nest outputs in a subfolder named after the output basename, so
    # `aux_figures/morphem_high.png` → `aux_figures/morphem_high/morphem_high_NN.png`.
    out_dir, fname = os.path.split(args.output)
    stem, ext = os.path.splitext(fname)
    out_dir = os.path.join(out_dir, stem)
    base = os.path.join(out_dir, stem)
    os.makedirs(out_dir, exist_ok=True)
    seen: set[tuple[str, ...]] = set()
    for i in range(args.n):
        for _ in range(50):
            key, label = pick()
            if key not in seen:
                seen.add(key)
                break
        img = load_well_image(*key)
        key_str = "__".join(key)
        # Canonical Cell Painting display: alphabetical channel order is
        # AGP=0, DNA=1, ER=2, Mito=3, RNA=4. Map R=blank, G=RNA, B=DNA
        # (matches archived visualize_cell_compression.py).
        rgb = np.zeros((*img.shape[1:], 3))
        for j, ch in enumerate([99, 4, 1]):
            if ch >= img.shape[0]:
                continue
            data = img[ch].astype(np.float64)
            lo, hi = np.percentile(data, [0.1, 99.9])
            if hi > lo:
                rgb[..., j] = np.clip((data - lo) / (hi - lo), 0, 1)
        path = f"{base}{ext}" if args.n == 1 else f"{base}_{i+1:02d}{ext}"
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgb)
        ax.set_title(f"{key_str}\n{label}", fontsize=9)
        ax.axis("off")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Bare 512x512 center crop, no axes/title/whitespace
        h, w = rgb.shape[:2]
        cy, cx = h // 2, w // 2
        crop = rgb[cy - 256:cy + 256, cx - 256:cx + 256]
        crop_path = f"{base}_{i+1:02d}_crop512{ext}" if args.n > 1 else f"{base}_crop512{ext}"
        plt.imsave(crop_path, np.clip(crop, 0, 1))
        print(f"{path}  ({key_str})  {label}")
        print(f"{crop_path}  (512x512 center crop)")


if __name__ == "__main__":
    main()
