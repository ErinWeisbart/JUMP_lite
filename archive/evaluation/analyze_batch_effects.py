#!/usr/bin/env python3
"""Analyze batch effects across all sweep output parquet files.

Uses random sampling to speed up the analysis:
- Well Position Effect: Uses random index (1-10) as neg_sameby instead of plate
- Plate Batch Effect: Uses random index for same_posby and same_negby, removes well location
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from copairs import map as copairs_map


def calculate_batch_effects_fast(
    df: pl.DataFrame,
    features: list[str],
    null_size: int = 10_000,
    p_threshold: float = 0.05,
    seed: int = 0,
    n_random_groups: int = 15,
    compound_col: str = "Metadata_JCP2022",
    negcon_col: str = "Metadata_negcon",
    plate_col: str = "Metadata_Plate",
    well_col: str = "Metadata_Well",
    group_col: str = "Metadata_Group",
) -> dict:
    """
    Calculate batch effect metrics using random sampling for speed.

    Well Position Effect (on treatments only):
        - Positive: same group, same well location, different plate, different compound
        - Negative: same group, same plate, same random_index, different well, different compound

    Plate Batch Effect (on treatments only, excluding negcons):
        - Positive: same group, same plate, same random_index, different compound
        - Negative: same group, same random_index, different plate, different well, different compound

    Args:
        df: Profiles with metadata
        features: Feature column names
        null_size: Size of null distribution
        p_threshold: Significance threshold
        seed: Random seed
        n_random_groups: Number of random groups (1-10 by default)
        compound_col: Column containing compound identifier
        negcon_col: Column containing negative control flag
        plate_col: Column containing plate identifier
        well_col: Column containing well identifier
        group_col: Column containing group identifier

    Returns:
        Dictionary with batch effect metrics
    """
    results = {}
    rng = np.random.default_rng(seed)

    df_pd = df.to_pandas()

    if well_col not in df_pd.columns:
        print(f"  Warning: {well_col} not found, skipping batch effect analysis")
        return {"well_position_effect": None, "plate_batch_effect": None}

    # Add random index columns for faster sampling
    df_pd["_random_index"] = rng.integers(1, n_random_groups + 1, size=len(df_pd))
    # Separate random indices for plate batch effect
    df_pd["_random_index_pos"] = rng.integers(1, 11, size=len(df_pd))  # 10 values for positive
    df_pd["_random_index_neg"] = rng.integers(1, 251, size=len(df_pd))  # 250 values for negative

    has_groups = group_col in df_pd.columns

    # Filter out negative controls for both analyses
    if negcon_col in df_pd.columns:
        df_treatments = df_pd[df_pd[negcon_col] == False].copy()
    else:
        df_treatments = df_pd.copy()

    if len(df_treatments) < 10:
        print(f"    Not enough treatment samples ({len(df_treatments)})")
        return {
            "well_position_effect": {"pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0},
            "plate_batch_effect": {"pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0},
        }

    # =========================================
    # 1. Well Position Effect
    # =========================================
    # Positive: same group, same well location, different plate, different compound
    # Negative: same group, same plate, same random_index, different well, different compound
    print("  Calculating Well Position Effect...")

    try:
        if has_groups:
            pos_sameby_well = [group_col, well_col]
            neg_sameby_well = [group_col, plate_col, "_random_index"]
        else:
            pos_sameby_well = [well_col]
            neg_sameby_well = [plate_col, "_random_index"]

        pos_diffby_well = [plate_col, compound_col]
        neg_diffby_well = [well_col, compound_col]

        # Filter to valid combinations (at least 2 different plates per well position within group)
        if has_groups:
            # Need wells that appear on multiple plates with different compounds
            well_plate_counts = df_treatments.groupby([group_col, well_col])[plate_col].nunique()
            valid_combinations = well_plate_counts[well_plate_counts >= 2].index.tolist()

            if len(valid_combinations) < 2:
                print("    Not enough valid (group, well) combinations across plates")
                results["well_position_effect"] = {
                    "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0
                }
            else:
                df_well = df_treatments[
                    df_treatments.apply(lambda r: (r[group_col], r[well_col]) in valid_combinations, axis=1)
                ].copy()

                metadata_well = df_well.filter(regex="^Metadata|^_random_index")
                profiles_well = df_well[features].values

                well_ap = copairs_map.average_precision(
                    metadata_well, profiles_well,
                    pos_sameby_well, pos_diffby_well,
                    neg_sameby_well, neg_diffby_well
                )

                if len(well_ap) > 0:
                    well_map = copairs_map.mean_average_precision(
                        well_ap, pos_sameby_well,
                        null_size=null_size, threshold=p_threshold, seed=seed
                    )
                    well_map["below_corrected_p"] = well_map["corrected_p_value"] < p_threshold

                    pct_active = (well_map["below_corrected_p"].sum() / len(well_map)) * 100 if len(well_map) > 0 else 0
                    mean_map_val = well_map["mean_average_precision"].mean() if len(well_map) > 0 else 0
                    mean_nap_val = well_map["mean_normalized_average_precision"].mean() if len(well_map) > 0 else 0

                    # Per-group stats
                    per_group_stats = {}
                    for grp in well_map[group_col].unique():
                        grp_data = well_map[well_map[group_col] == grp]
                        grp_active = grp_data["below_corrected_p"].sum()
                        grp_total = len(grp_data)
                        grp_pct = (grp_active / grp_total * 100) if grp_total > 0 else 0
                        per_group_stats[grp] = {
                            "pct_active": float(grp_pct),
                            "n_active": int(grp_active),
                            "n_total": int(grp_total),
                            "mean_map": float(grp_data["mean_average_precision"].mean()),
                            "mean_nap": float(grp_data["mean_normalized_average_precision"].mean()),
                        }

                    results["well_position_effect"] = {
                        "pct_active": float(pct_active),
                        "n_active": int(well_map["below_corrected_p"].sum()),
                        "n_total": int(len(well_map)),
                        "mean_map": float(mean_map_val),
                        "mean_nap": float(mean_nap_val),
                        "n_samples_used": int(len(df_well)),
                        "per_group": per_group_stats,
                    }
                    print(f"    Well Position Effect: {pct_active:.2f}%")
                else:
                    results["well_position_effect"] = {
                        "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0
                    }
        else:
            # No groups
            well_plate_counts = df_treatments.groupby(well_col)[plate_col].nunique()
            valid_well_locs = well_plate_counts[well_plate_counts >= 2].index.tolist()

            if len(valid_well_locs) < 2:
                results["well_position_effect"] = {
                    "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0
                }
            else:
                df_well = df_treatments[df_treatments[well_col].isin(valid_well_locs)].copy()
                metadata_well = df_well.filter(regex="^Metadata|^_random_index")
                profiles_well = df_well[features].values

                well_ap = copairs_map.average_precision(
                    metadata_well, profiles_well,
                    pos_sameby_well, pos_diffby_well,
                    neg_sameby_well, neg_diffby_well
                )

                if len(well_ap) > 0:
                    well_map = copairs_map.mean_average_precision(
                        well_ap, pos_sameby_well,
                        null_size=null_size, threshold=p_threshold, seed=seed
                    )
                    well_map["below_corrected_p"] = well_map["corrected_p_value"] < p_threshold

                    pct_active = (well_map["below_corrected_p"].sum() / len(well_map)) * 100 if len(well_map) > 0 else 0
                    mean_map_val = well_map["mean_average_precision"].mean() if len(well_map) > 0 else 0
                    mean_nap_val = well_map["mean_normalized_average_precision"].mean() if len(well_map) > 0 else 0

                    results["well_position_effect"] = {
                        "pct_active": float(pct_active),
                        "n_active": int(well_map["below_corrected_p"].sum()),
                        "n_total": int(len(well_map)),
                        "mean_map": float(mean_map_val),
                        "mean_nap": float(mean_nap_val),
                    }
                    print(f"    Well Position Effect: {pct_active:.2f}%")
                else:
                    results["well_position_effect"] = {
                        "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0
                    }

    except Exception as e:
        print(f"    Warning: Well position effect failed: {e}")
        results["well_position_effect"] = {
            "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0, "error": str(e)
        }

    # =========================================
    # 2. Plate Batch Effect (on all treatments, excluding negcons)
    # =========================================
    # Positive: same group, same plate, same random_index, different compound
    # Negative: same group, same random_index, different plate, different well, different compound
    print("  Calculating Plate Batch Effect...")

    try:
        n_plates = df_treatments[plate_col].nunique()

        if n_plates < 2:
            print(f"    Not enough plates ({n_plates})")
            results["plate_batch_effect"] = {
                "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0
            }
        else:
            metadata_treat = df_treatments.filter(regex="^Metadata|^_random_index")
            profiles_treat = df_treatments[features].values

            if has_groups:
                pos_sameby_plate = [group_col, plate_col, "_random_index_pos"]
                neg_sameby_plate = [group_col, "_random_index_neg"]
            else:
                pos_sameby_plate = [plate_col, "_random_index_pos"]
                neg_sameby_plate = ["_random_index_neg"]

            pos_diffby_plate = [compound_col]
            neg_diffby_plate = [plate_col, well_col, compound_col]

            plate_ap = copairs_map.average_precision(
                metadata_treat, profiles_treat,
                pos_sameby_plate, pos_diffby_plate,
                neg_sameby_plate, neg_diffby_plate
            )

            if len(plate_ap) > 0:
                # Group by plate for MAP calculation
                map_groupby = [group_col, plate_col] if has_groups else [plate_col]
                plate_map = copairs_map.mean_average_precision(
                    plate_ap, map_groupby,
                    null_size=null_size, threshold=p_threshold, seed=seed
                )
                plate_map["below_corrected_p"] = plate_map["corrected_p_value"] < p_threshold

                pct_active = (plate_map["below_corrected_p"].sum() / len(plate_map)) * 100 if len(plate_map) > 0 else 0
                mean_map_val = plate_map["mean_average_precision"].mean() if len(plate_map) > 0 else 0
                mean_nap_val = plate_map["mean_normalized_average_precision"].mean() if len(plate_map) > 0 else 0

                result_dict = {
                    "pct_active": float(pct_active),
                    "n_active": int(plate_map["below_corrected_p"].sum()),
                    "n_total": int(len(plate_map)),
                    "mean_map": float(mean_map_val),
                    "mean_nap": float(mean_nap_val),
                    "n_plates": int(n_plates),
                    "n_samples": int(len(df_treatments)),
                }

                # Per-group stats if groups exist
                if has_groups:
                    per_group_stats = {}
                    for grp in plate_map[group_col].unique():
                        grp_data = plate_map[plate_map[group_col] == grp]
                        grp_active = grp_data["below_corrected_p"].sum()
                        grp_total = len(grp_data)
                        grp_pct = (grp_active / grp_total * 100) if grp_total > 0 else 0
                        per_group_stats[grp] = {
                            "pct_active": float(grp_pct),
                            "n_active": int(grp_active),
                            "n_total": int(grp_total),
                            "mean_map": float(grp_data["mean_average_precision"].mean()),
                            "mean_nap": float(grp_data["mean_normalized_average_precision"].mean()),
                        }
                    result_dict["per_group"] = per_group_stats

                results["plate_batch_effect"] = result_dict
                print(f"    Plate Batch Effect: {pct_active:.2f}% ({n_plates} plates)")
            else:
                results["plate_batch_effect"] = {
                    "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0
                }

    except Exception as e:
        error_msg = str(e)
        if "No data left" in error_msg:
            print("    Warning: Plate batch effect failed: not enough valid pairs (try fewer random groups)")
        else:
            print(f"    Warning: Plate batch effect failed: {e}")
        results["plate_batch_effect"] = {
            "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0, "error": error_msg
        }

    return results


def analyze_parquet(parquet_path: Path, seed: int = 0, save_individual: bool = True) -> dict:
    """Analyze a single parquet file for batch effects."""
    config_name = parquet_path.parent.name

    # Check if results already exist
    results_dir = parquet_path.parent / "results"
    batch_effects_path = results_dir / "batch_effects.json"

    if batch_effects_path.exists():
        print(f"\n  Skipping {config_name}: batch_effects.json already exists")
        # Load existing results
        with open(batch_effects_path) as f:
            return json.load(f)

    print(f"\n  Processing: {config_name}")

    df = pl.read_parquet(parquet_path)

    # Get feature columns
    features = [c for c in df.columns if not c.startswith("Metadata")]

    if len(features) == 0:
        print("    No feature columns found")
        return None

    # Run batch effects analysis
    results = calculate_batch_effects_fast(df, features, seed=seed)

    # Save individual results to the config's results folder
    if save_individual and results:
        results_dir.mkdir(exist_ok=True)
        with open(batch_effects_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"    Saved to {batch_effects_path}")

    return results


def parse_config_name(config_name: str) -> dict:
    """Parse config name into individual settings."""
    parts = config_name.split("__")
    settings = {
        "use_wellcorr": False,
        "use_int": False,
        "use_snorm": False,
        "snorm_type": None,
        "batch_method": "none",
        "tvn_epsilon": None,
        "spherize_method": None,
        "spherize_fit_controls": None,
        "prune_thresh": None,
        "agg_method": None,
    }

    for part in parts:
        if part == "wellcorr":
            settings["use_wellcorr"] = True
        elif part == "INT":
            settings["use_int"] = True
        elif part.startswith("snorm_"):
            settings["use_snorm"] = True
            settings["snorm_type"] = part.replace("snorm_", "")
        elif part.startswith("tvn_efaar_e"):
            settings["batch_method"] = "tvn_efaar"
            settings["tvn_epsilon"] = float(part.replace("tvn_efaar_e", ""))
        elif part.startswith("robustmad"):
            settings["batch_method"] = "robustmad"
            if "_ctrl" in part:
                settings["norm_fit_controls"] = True
            elif "_all" in part:
                settings["norm_fit_controls"] = False
        elif part in ("PCA-cor_ctrl", "PCA-cor_all", "ZCA-cor_ctrl", "ZCA-cor_all"):
            if "PCA" in part:
                settings["spherize_method"] = "PCA-cor"
            else:
                settings["spherize_method"] = "ZCA-cor"
            settings["spherize_fit_controls"] = "_ctrl" in part
        elif part.startswith("prune"):
            settings["prune_thresh"] = float(part.replace("prune", ""))
        elif part.startswith("agg_"):
            settings["agg_method"] = part.replace("agg_", "")

    return settings


def main():
    parser = argparse.ArgumentParser(description="Analyze batch effects across sweep outputs")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("src/norm_3/data/features/unified_batch_sweep"),
        help="Path to sweep output directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: batch_effects.csv in sweep-dir)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Don't save individual batch_effects.json to each config's results/ folder",
    )
    args = parser.parse_args()

    # Default output to sweep directory
    if args.output is None:
        args.output = args.sweep_dir / "batch_effects.csv"

    # Find all output.parquet files
    parquet_files = list(args.sweep_dir.rglob("output.parquet"))
    print(f"Found {len(parquet_files)} output.parquet files")

    if not parquet_files:
        print("No parquet files found!")
        return

    # Analyze each parquet
    all_results = []
    for parquet_path in parquet_files:
        # Extract path components
        # Path: .../unified_batch_sweep/model_name/config_name/output.parquet
        config_name = parquet_path.parent.name
        model_name = parquet_path.parent.parent.name
        model_clean = model_name.replace("_jump_core_annotated_raw_features", "")

        try:
            batch_results = analyze_parquet(
                parquet_path, seed=args.seed, save_individual=not args.no_individual
            )

            if batch_results is None:
                continue

            # Build row
            row = {
                "model": model_clean,
                "config": config_name,
            }

            # Well position effect
            well_effect = batch_results.get("well_position_effect", {})
            row["well_effect_pct"] = well_effect.get("pct_active")
            row["well_effect_n_active"] = well_effect.get("n_active")
            row["well_effect_n_total"] = well_effect.get("n_total")
            row["well_effect_mean_map"] = well_effect.get("mean_map")
            row["well_effect_mean_nap"] = well_effect.get("mean_nap")

            # Per-group well effects
            per_group = well_effect.get("per_group", {})
            for grp, stats in per_group.items():
                row[f"well_effect_{grp}_pct"] = stats.get("pct_active")
                row[f"well_effect_{grp}_mean_map"] = stats.get("mean_map")
                row[f"well_effect_{grp}_mean_nap"] = stats.get("mean_nap")

            # Plate batch effect
            plate_effect = batch_results.get("plate_batch_effect", {})
            row["plate_effect_pct"] = plate_effect.get("pct_active")
            row["plate_effect_n_active"] = plate_effect.get("n_active")
            row["plate_effect_n_total"] = plate_effect.get("n_total")
            row["plate_effect_mean_map"] = plate_effect.get("mean_map")
            row["plate_effect_mean_nap"] = plate_effect.get("mean_nap")
            row["plate_effect_n_plates"] = plate_effect.get("n_plates")
            row["plate_effect_n_samples"] = plate_effect.get("n_samples")

            # Per-group plate effects
            plate_per_group = plate_effect.get("per_group", {})
            for grp, stats in plate_per_group.items():
                row[f"plate_effect_{grp}_pct"] = stats.get("pct_active")
                row[f"plate_effect_{grp}_mean_map"] = stats.get("mean_map")
                row[f"plate_effect_{grp}_mean_nap"] = stats.get("mean_nap")

            # Parse config settings
            settings = parse_config_name(config_name)
            row.update(settings)

            all_results.append(row)

        except Exception as e:
            print(f"  Error processing {parquet_path}: {e}")

    if not all_results:
        print("No results collected!")
        return

    # Create DataFrame
    df = pl.DataFrame(all_results, infer_schema_length=None)

    # Sort by model, then by plate_effect_pct ascending (lower is better)
    df = df.sort(["model", "plate_effect_pct"], descending=[False, False], nulls_last=True)

    # Save to CSV
    df.write_csv(args.output)
    print(f"\nSaved {len(df)} results to {args.output}")

    # Print summary
    print("\n=== Summary by Model ===")
    summary = (
        df.group_by("model")
        .agg(
            pl.len().alias("n_configs"),
            pl.col("well_effect_pct").mean().alias("mean_well_effect"),
            pl.col("plate_effect_pct").mean().alias("mean_plate_effect"),
            pl.col("plate_effect_pct").min().alias("min_plate_effect"),
        )
        .sort("model")
    )
    print(summary)

    # Print best config per model (lowest plate batch effect)
    print("\n=== Best Config per Model (lowest plate batch effect) ===")
    best = df.group_by("model").agg(
        pl.col("config").sort_by("plate_effect_pct").first().alias("best_config"),
        pl.col("plate_effect_pct").min().alias("plate_effect"),
        pl.col("well_effect_pct").sort_by("plate_effect_pct").first().alias("well_effect"),
    )
    print(best)


if __name__ == "__main__":
    main()
