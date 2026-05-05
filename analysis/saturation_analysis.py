#!/usr/bin/env python3
"""Saturation analysis: how many treatments are needed to reliably rank models?

Subsamples treatments at various sizes, recomputes PA/PC via evaluate_all(),
and plots convergence curves with bootstrapped confidence intervals.
Marks RxRx3-core's ~2,400 treatments for comparison.

Usage:
    # Run on all best-config parquets in a directory
    python analysis/saturation_analysis.py --input-dir src/norm_3/data/features/best_configs_lite

    # Run on a single parquet
    python analysis/saturation_analysis.py --input path/to/output.parquet --label morphem
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

# Add src to path for norm_3 imports (set automatically by pixi via PYTHONPATH)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from norm_3.io import infer_columns
from norm_3.metrics import evaluate_all

SUBSAMPLE_SIZES = [50, 100, 250, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000]
N_SEEDS = 10
RXRX3_CORE_SIZE = 2400

COMPOUND_COL = "Metadata_JCP2022"
TARGET_COL = "Metadata_RefChemDB_target"
NEGCON_COL = "Metadata_negcon"
BATCH_COL = "Metadata_Plate"
GROUP_COL = "Metadata_Group"


def filter_groups(df: pl.DataFrame, groups: list[str] | None) -> pl.DataFrame:
    """Filter to specific groups, keeping negcons and dropping the group column.

    Drops Metadata_Group so that PA computes a single global metric
    rather than trying to compute per-group (which fails when only one group remains).
    """
    if not groups:
        return df
    negcons = df.filter(pl.col(NEGCON_COL) == True)
    treatments = df.filter(
        (pl.col(NEGCON_COL) == False) & pl.col(GROUP_COL).is_in(groups)
    )
    combined = pl.concat([negcons, treatments])
    return combined.drop(GROUP_COL)


def subsample_treatments(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    """Subsample n treatments, keeping all negative controls."""
    negcons = df.filter(pl.col(NEGCON_COL) == True)
    treatments = df.filter(pl.col(NEGCON_COL) == False)

    unique_ids = treatments[COMPOUND_COL].unique().to_list()
    rng = np.random.RandomState(seed)

    if n >= len(unique_ids):
        return df

    sampled_ids = rng.choice(unique_ids, size=n, replace=False).tolist()
    sampled_treatments = treatments.filter(pl.col(COMPOUND_COL).is_in(sampled_ids))

    return pl.concat([negcons, sampled_treatments])


def run_single(
    df: pl.DataFrame, features: list[str], n: int, seed: int,
    pa_only: bool = False,
    pc_only: bool = False,
) -> dict:
    """Run PA and/or PC at one subsample size and seed."""
    sub = subsample_treatments(df, n, seed)
    n_actual = sub.filter(pl.col(NEGCON_COL) == False)[COMPOUND_COL].n_unique()

    from norm_3.metrics import calculate_phenotypic_activity, calculate_phenotypic_consistency

    group_col = GROUP_COL if GROUP_COL in sub.columns else None
    # Auto-select distance: euclidean for <=2 features (e.g. cell count baseline)
    distance = "euclidean" if len(features) <= 2 else "cosine"

    result = {
        "n_requested": n,
        "n_actual": n_actual,
        "seed": seed,
        "PA_mean_nap": np.nan,
        "PA_pct": np.nan,
        "n_compounds": 0,
        "PC_mean_nap": np.nan,
        "PC_pct": np.nan,
        "n_targets": 0,
    }

    if not pc_only:
        pa_kwargs = dict(
            compound_col=COMPOUND_COL,
            negcon_col=NEGCON_COL,
            batch_col=BATCH_COL,
            distance=distance,
        )
        if group_col:
            pa_kwargs["group_col"] = group_col
        try:
            pa = calculate_phenotypic_activity(sub, features, **pa_kwargs)
            result["PA_mean_nap"] = pa.get("mean_normalized_average_precision", np.nan)
            result["PA_pct"] = pa.get("pct_compounds_active", np.nan)
            result["n_compounds"] = pa.get("n_compounds", 0)
        except Exception as e:
            print(f"    PA error: {e}")


    if not pa_only:
        pc_kwargs = dict(
            compound_col=COMPOUND_COL,
            target_col=TARGET_COL,
            negcon_col=NEGCON_COL,
            distance=distance,
        )
        if group_col:
            pc_kwargs["group_col"] = group_col
        try:
            pc = calculate_phenotypic_consistency(sub, features, **pc_kwargs)
            result["PC_mean_nap"] = pc.get("mean_normalized_average_precision", np.nan)
            result["PC_pct"] = pc.get("pct_targets_active", np.nan)
            result["n_targets"] = pc.get("n_targets_total", 0)
        except Exception as e:
            print(f"    PC error: {e}")

    return result


def run_saturation(
    df: pl.DataFrame,
    features: list[str],
    label: str,
    sizes: list[int],
    n_seeds: int,
    output_csv: Path | None = None,
    pa_only: bool = False,
    pc_only: bool = False,
    config: str = "default",
    config_id: int = 0,
) -> pd.DataFrame:
    """Run full saturation analysis for one model, saving incrementally."""
    n_total = df.filter(pl.col(NEGCON_COL) == False)[COMPOUND_COL].n_unique()
    valid_sizes = [s for s in sizes if s <= n_total] + [n_total]
    valid_sizes = sorted(set(valid_sizes))

    print(f"\n{'='*60}")
    print(f"Model: {label} ({n_total} unique treatments, {len(df)} rows)")
    print(f"Sizes: {valid_sizes}")
    print(f"Seeds: {n_seeds}")
    print(f"{'='*60}")

    # Load existing results to skip completed runs
    existing = set()
    if output_csv and output_csv.exists():
        prev = pd.read_csv(output_csv)
        prev_model = prev[(prev["model"] == label) & (prev.get("config_id", 0) == config_id)]
        for _, row in prev_model.iterrows():
            existing.add((int(row["n_requested"]), int(row["seed"])))
        if existing:
            print(f"  Resuming: {len(existing)} runs already completed")

    rows = []
    for n in valid_sizes:
        seeds = n_seeds if n < n_total else 1
        for seed in range(seeds):
            if (n, seed) in existing:
                continue
            print(f"  n={n:>6}, seed={seed}", end=" → ", flush=True)
            result = run_single(df, features, n, seed, pa_only=pa_only, pc_only=pc_only)
            result["model"] = label
            result["config"] = config
            result["config_id"] = config_id
            rows.append(result)
            print(f"PA={result['PA_mean_nap']:.4f}, PC={result['PC_mean_nap']:.4f}")

            # Save incrementally
            if output_csv:
                row_df = pd.DataFrame([result])
                row_df.to_csv(output_csv, mode="a",
                              header=not output_csv.exists() or output_csv.stat().st_size == 0,
                              index=False)

    return pd.DataFrame(rows)


def plot_saturation(results: pd.DataFrame, output_dir: Path):
    """Plot convergence curves for PA and PC."""
    has_configs = "config_id" in results.columns and results["config_id"].nunique() > 1
    models = results["model"].unique()

    # Color palette per model
    model_colors = {}
    cmap = plt.cm.tab10
    for i, model in enumerate(sorted(models)):
        model_colors[model] = cmap(i)

    for metric, label in [("PA_mean_nap", "PA (NAP)"), ("PC_mean_nap", "PC (NAP)")]:
        fig, ax = plt.subplots(figsize=(10, 6))

        for model in sorted(models):
            color = model_colors[model]
            model_data = results[results["model"] == model]

            if has_configs:
                # Plot each config as its own line (same color, different linestyle)
                config_ids = sorted(model_data["config_id"].unique())
                linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]
                for idx, cid in enumerate(config_ids):
                    config_data = model_data[model_data["config_id"] == cid]
                    summary = config_data.groupby("n_actual")[metric].agg(["mean", "std"]).reset_index()
                    summary = summary.sort_values("n_actual")
                    ls = linestyles[idx % len(linestyles)]
                    lbl = model if idx == 0 else None  # only label first config
                    ax.plot(summary["n_actual"], summary["mean"], marker="o",
                            linestyle=ls, color=color, label=lbl,
                            markersize=3, linewidth=1, alpha=0.7)
                    ax.fill_between(
                        summary["n_actual"],
                        summary["mean"] - summary["std"],
                        summary["mean"] + summary["std"],
                        color=color, alpha=0.05,
                    )
            else:
                # Original behavior: mean + std band across seeds
                summary = model_data.groupby("n_actual")[metric].agg(["mean", "std"]).reset_index()
                summary = summary.sort_values("n_actual")
                ax.plot(summary["n_actual"], summary["mean"], "o-",
                        color=color, label=model, markersize=4)
                ax.fill_between(
                    summary["n_actual"],
                    summary["mean"] - summary["std"],
                    summary["mean"] + summary["std"],
                    color=color, alpha=0.15,
                )

        ax.axvline(x=RXRX3_CORE_SIZE, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
        ax.text(
            RXRX3_CORE_SIZE + 100, ax.get_ylim()[1] * 0.95,
            f"RxRx3-core\n(~{RXRX3_CORE_SIZE})",
            color="red", fontsize=9, va="top",
        )

        ax.set_xlabel("Number of Treatments")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs. Treatment Count")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        fig.tight_layout()

        out = output_dir / f"saturation_{metric}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close(fig)

    # Combined PA vs PC scatter at different sizes
    fig, ax = plt.subplots(figsize=(7, 6))
    sizes_to_show = [250, 1000, 2500, 5000, 10000]
    for model in sorted(models):
        model_data = results[results["model"] == model]
        summary = model_data.groupby("n_actual")[["PA_mean_nap", "PC_mean_nap"]].mean().reset_index()
        ax.plot(summary["PA_mean_nap"], summary["PC_mean_nap"], "o-", label=model, markersize=5, alpha=0.7)
        # Label the full-dataset point
        if len(summary) > 0:
            last = summary.iloc[-1]
            ax.annotate(model, (last["PA_mean_nap"], last["PC_mean_nap"]),
                        fontsize=7, alpha=0.7)

    ax.set_xlabel("PA (NAP)")
    ax.set_ylabel("PC (NAP)")
    ax.set_title("PA vs PC Trajectory as Treatment Count Increases")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / "saturation_pa_vs_pc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Single output.parquet file")
    group.add_argument("--input-dir", type=Path, help="Directory with multiple parquets")
    parser.add_argument("--label", type=str, default=None,
                        help="Model label (required with --input)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("analysis/output/saturation"))
    parser.add_argument("--sizes", type=int, nargs="+", default=SUBSAMPLE_SIZES)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--groups", nargs="+", default=None,
                        help="Filter to specific groups (e.g., --groups group_crispr)")
    parser.add_argument("--pa-only", action="store_true",
                        help="Only compute PA, skip PC")
    parser.add_argument("--pc-only", action="store_true",
                        help="Only compute PC, skip PA")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip computation, replot from existing saturation_results.csv")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = args.output_dir / "saturation_results.csv"

    if args.plot_only:
        if not results_csv.exists():
            print(f"No results file found at {results_csv}")
            return
        combined = pd.read_csv(results_csv)
        print(f"Loaded {len(combined)} rows from {results_csv}")
        plot_saturation(combined, args.output_dir)
        return

    # Collect input files
    inputs = []
    if args.input:
        label = args.label or args.input.stem.split("__")[0]
        inputs.append((args.input, label, "default"))
    else:
        # Group by model, assign config_id per model
        from collections import defaultdict
        model_configs = defaultdict(list)
        for f in sorted(args.input_dir.glob("*.parquet")):
            parts = f.stem.split("__", 1)
            model = parts[0]
            config = parts[1] if len(parts) > 1 else "default"
            model_configs[model].append((f, config))

        for model, files in sorted(model_configs.items()):
            for config_id, (f, config) in enumerate(files):
                inputs.append((f, model, config, config_id))

    if not inputs:
        print("No parquet files found.")
        return

    all_results = []
    for entry in inputs:
        if len(entry) == 4:
            path, model, config, config_id = entry
            label = model
        else:
            path, label, config = entry
            config_id = 0

        print(f"\nLoading: {path}")
        df = pl.read_parquet(path)
        if args.groups:
            df = filter_groups(df, args.groups)
            print(f"  Filtered to groups {args.groups}: {len(df)} rows")
        features, _ = infer_columns(df)
        result = run_saturation(df, features, label, args.sizes, args.n_seeds,
                                output_csv=results_csv, pa_only=args.pa_only,
                                pc_only=args.pc_only,
                                config=config, config_id=config_id)
        all_results.append(result)

    # Read back the full CSV (includes resumed + new results)
    combined = pd.read_csv(results_csv)
    print(f"\nTotal results: {len(combined)} rows in {results_csv}")

    plot_saturation(combined, args.output_dir)


if __name__ == "__main__":
    main()
