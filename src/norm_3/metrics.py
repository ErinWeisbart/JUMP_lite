"""Metrics for normalization quality evaluation.

This module provides:
- Phenotypic Activity (PA): compound replicate retrieval
- Phenotypic Consistency (PC): target-level retrieval

Uses copairs library (CPU) for the actual metrics computation.
GPU is used for preprocessing only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from norm_3.io import get_numeric_features, infer_columns


def calculate_phenotypic_activity(
    df: pl.DataFrame,
    features: list[str],
    null_size: int = 10_000,
    p_threshold: float = 0.05,
    seed: int = 0,
    compound_col: str = "Metadata_pert_iname",
    negcon_col: str = "Metadata_negcon",
    negcon_value: str = "DMSO",
    batch_col: str = "Metadata_Plate",
    group_col: str = "Metadata_Group",
) -> dict[str, Any]:
    """Calculate Phenotypic Activity (compound replicate retrieval).

    Measures how well replicates of the same compound cluster together
    compared to random compounds.

    Args:
        df: Normalized profiles with metadata
        features: Feature column names
        null_size: Size of null distribution
        p_threshold: Significance threshold
        seed: Random seed
        compound_col: Column containing compound/perturbation identifier
        negcon_col: Column containing negative control flag
        negcon_value: Value in compound_col that represents negative controls
        batch_col: Column containing batch/plate identifier
        group_col: Column containing group identifier for per-group statistics

    Returns:
        Dictionary with metrics including per-group summary
    """
    import pandas as pd
    from copairs import map as copairs_map
    from copairs.matching import assign_reference_index

    df_pd = df.to_pandas()

    # Check if group column exists for per-group statistics
    has_groups = group_col in df_pd.columns

    if has_groups:
        pos_sameby = [compound_col, group_col, "Metadata_reference_index"]
        neg_sameby = [batch_col, group_col]
    else:
        pos_sameby = [compound_col, "Metadata_reference_index"]
        neg_sameby = [batch_col]

    pos_diffby = []
    neg_diffby = [compound_col, negcon_col, "Metadata_reference_index"]

    # Split by group and compute AP separately per group.
    # Since group_col is in both pos_sameby and neg_sameby, copairs only
    # compares within groups. Splitting first avoids the overhead of
    # processing all 163K rows in a single call.
    if has_groups:
        groups = sorted(df_pd[group_col].unique())
        activity_ap_parts = []
        for grp in groups:
            grp_df = df_pd[df_pd[group_col] == grp].copy().reset_index(drop=True)
            if negcon_col in grp_df.columns:
                grp_df = assign_reference_index(
                    grp_df, f"{negcon_col} == True",
                    reference_col="Metadata_reference_index", default_value=-1,
                )
            else:
                grp_df = assign_reference_index(
                    grp_df, f"{compound_col} == '{negcon_value}'",
                    reference_col="Metadata_reference_index", default_value=-1,
                )
            grp_meta = grp_df.filter(regex="^Metadata")
            grp_profiles = grp_df[features].values
            print(f"  PA {grp}: {len(grp_df)} rows")
            grp_ap = copairs_map.average_precision(
                grp_meta, grp_profiles, pos_sameby, pos_diffby, neg_sameby, neg_diffby
            )
            activity_ap_parts.append(grp_ap)
        activity_ap = pd.concat(activity_ap_parts, ignore_index=True)
    else:
        if negcon_col in df_pd.columns:
            df_pd = assign_reference_index(
                df_pd, f"{negcon_col} == True",
                reference_col="Metadata_reference_index", default_value=-1,
            )
        else:
            df_pd = assign_reference_index(
                df_pd, f"{compound_col} == '{negcon_value}'",
                reference_col="Metadata_reference_index", default_value=-1,
            )
        metadata = df_pd.filter(regex="^Metadata")
        profiles = df_pd[features].values
        activity_ap = copairs_map.average_precision(
            metadata, profiles, pos_sameby, pos_diffby, neg_sameby, neg_diffby
        )

    # Filter out negative controls
    if negcon_col in activity_ap.columns:
        activity_ap = activity_ap.query(f"{negcon_col} == False").copy()
    else:
        activity_ap = activity_ap.query(f"{compound_col} != '{negcon_value}'").copy()

    # Calculate replicate counts per compound (or compound+group)
    replicate_counts = activity_ap.groupby(pos_sameby).size()

    # Calculate mean average precision
    activity_map = copairs_map.mean_average_precision(
        activity_ap, pos_sameby, null_size=null_size, threshold=p_threshold, seed=seed
    ).copy()
    activity_map["below_corrected_p"] = activity_map["corrected_p_value"] < p_threshold

    # Merge replicate counts into activity_map
    activity_map = activity_map.merge(
        replicate_counts.rename("n_replicate_pairs"),
        on=pos_sameby,
        how="left",
    )

    pct_compounds_active = (
        activity_map["below_corrected_p"].sum() / len(activity_map)
    ) * 100

    # Calculate per-group summary if groups exist
    group_summary = None
    if has_groups and group_col in activity_map.columns:
        group_summary = activity_map.groupby(group_col).agg(
            pct_active=("below_corrected_p", "mean"),
            num_active=("below_corrected_p", "sum"),
            mean_normalized_average_precision=("mean_normalized_average_precision", "mean"),
            median_normalized_average_precision=("mean_normalized_average_precision", "median"),
            mean_n_replicates=("n_replicate_pairs", "mean"),
            median_n_replicates=("n_replicate_pairs", "median"),
            n_unique_compounds=(compound_col, "nunique"),
        ).reset_index()
        group_summary["pct_active"] *= 100  # Convert to percentage

    # Calculate overall mean normalized average precision
    mean_nap = float(activity_map["mean_normalized_average_precision"].mean())
    median_nap = float(activity_map["mean_normalized_average_precision"].median())

    return {
        "activity_ap": activity_ap,
        "activity_map": activity_map,
        "group_summary": group_summary,
        "pct_compounds_active": float(pct_compounds_active),
        "n_compounds": int(len(activity_map)),
        "mean_normalized_average_precision": mean_nap,
        "median_normalized_average_precision": median_nap,
    }


def _filter_targets_by_compound_count(
    df_consensus,
    min_compounds_per_target: int = 3,
    max_targets_per_compound: int = 50,
    exclude_unknown: bool = True,
    compound_col: str = "Metadata_pert_iname",
    negcon_col: str = "Metadata_negcon",
):
    """Filter targets based on minimum number of compounds."""
    # Filter out negative controls
    df_consensus = df_consensus[df_consensus[negcon_col] == False].copy()

    # Filter out promiscuous compounds
    if max_targets_per_compound is not None:
        target_counts_per_compound = df_consensus["Metadata_target"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        n_promiscuous = (target_counts_per_compound > max_targets_per_compound).sum()
        if n_promiscuous > 0:
            print(f"  Filtering out {n_promiscuous} promiscuous compounds")
        df_consensus = df_consensus[
            target_counts_per_compound <= max_targets_per_compound
        ].copy()

    # Explode targets
    df_exploded = df_consensus.explode("Metadata_target")

    if exclude_unknown:
        df_exploded = df_exploded[df_exploded["Metadata_target"] != "unknown"].copy()

    # Count unique compounds per target
    target_counts = df_exploded.groupby("Metadata_target")[
        compound_col
    ].nunique()
    target_counts = target_counts.sort_values(ascending=False)

    valid_targets = target_counts[
        target_counts >= min_compounds_per_target
    ].index.tolist()
    print(f"  Using min_compounds_per_target={min_compounds_per_target}: {len(valid_targets)} targets")

    def has_valid_target(target_list):
        if not isinstance(target_list, list):
            return False
        return any(t in valid_targets for t in target_list)

    df_consensus = df_consensus[
        df_consensus["Metadata_target"].apply(has_valid_target)
    ].copy()

    df_consensus["Metadata_target"] = df_consensus["Metadata_target"].apply(
        lambda targets: [t for t in targets if t in valid_targets]
        if isinstance(targets, list)
        else []
    )

    print(f"  Compounds remaining after filtering: {len(df_consensus)}")
    return df_consensus


def calculate_phenotypic_consistency(
    df: pl.DataFrame,
    features: list[str],
    null_size: int = 10_000,
    p_threshold: float = 0.05,
    seed: int = 0,
    min_compounds_per_target: int = 3,
    max_targets_per_compound: int = 50,
    compound_col: str = "Metadata_pert_iname",
    target_col: str = "Metadata_target_list",
    negcon_col: str = "Metadata_negcon",
    group_col: str = "Metadata_Group",
    pc_groups: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate Phenotypic Consistency (target-level retrieval).

    Measures how well compounds targeting the same protein cluster together.

    Args:
        df: Normalized profiles with metadata
        features: Feature column names
        null_size: Size of null distribution
        p_threshold: Significance threshold
        seed: Random seed
        min_compounds_per_target: Minimum compounds per target
        max_targets_per_compound: Maximum targets per compound
        compound_col: Column containing compound identifier
        target_col: Column containing target identifier (pipe-separated for multiple)
        negcon_col: Column containing negative control flag
        group_col: Column containing group identifier for per-group statistics
        pc_groups: If set, only compute PC for these groups (e.g. ["group_high", "group_low"])

    Returns:
        Dictionary with metrics including per-group summary
    """
    from copairs import map as copairs_map

    # Filter to specified groups before computing PC
    if pc_groups and group_col in df.columns:
        before = len(df)
        df = df.filter(pl.col(group_col).is_in(pc_groups))
        print(f"  PC: filtered to groups {pc_groups}: {before} -> {len(df)} rows")

    df_pd = df.to_pandas()

    # Check if group column exists for per-group statistics
    has_groups = group_col in df_pd.columns

    # Fill null target values with "unknown"
    if target_col in df_pd.columns:
        df_pd[target_col] = df_pd[target_col].fillna("unknown")

    # Get consensus profiles per compound (and per group if groups exist)
    if has_groups:
        groupby_cols = [compound_col, target_col, negcon_col, group_col]
    else:
        groupby_cols = [compound_col, target_col, negcon_col]

    df_consensus = (
        df_pd.groupby(
            groupby_cols,
            as_index=False,
            observed=True,
        )[features]
        .median()
        .copy()
    )
    df_consensus["Metadata_target"] = df_consensus[target_col].str.split("|")

    df_consensus = _filter_targets_by_compound_count(
        df_consensus,
        min_compounds_per_target=min_compounds_per_target,
        max_targets_per_compound=max_targets_per_compound,
        compound_col=compound_col,
        negcon_col=negcon_col,
    )

    if len(df_consensus) < 2:
        print(f"  Warning: Not enough compounds ({len(df_consensus)}) for PC")
        return {
            "target_consistency": None,
            "pct_targets_active": 0.0,
            "n_targets_active": 0,
            "n_targets_total": 0,
            "group_summary": None,
        }

    # Include group in sameby columns if groups exist
    if has_groups:
        pos_sameby_target = ["Metadata_target", group_col]
        neg_sameby_target = [group_col]
    else:
        pos_sameby_target = ["Metadata_target"]
        neg_sameby_target = []

    pos_diffby_target = []
    neg_diffby_target = ["Metadata_target"]

    try:
        # Split by group for faster copairs computation
        if has_groups:
            import pandas as pd
            groups = sorted(df_consensus[group_col].unique())
            target_ap_parts = []
            for grp in groups:
                grp_df = df_consensus[df_consensus[group_col] == grp].copy().reset_index(drop=True)
                if len(grp_df) < 2:
                    continue
                grp_meta = grp_df.filter(regex="^Metadata")
                grp_profiles = grp_df[features].values
                print(f"  PC {grp}: {len(grp_df)} rows")
                grp_ap = copairs_map.multilabel.average_precision(
                    grp_meta, grp_profiles,
                    pos_sameby_target, pos_diffby_target,
                    neg_sameby_target, neg_diffby_target,
                    multilabel_col="Metadata_target",
                )
                target_ap_parts.append(grp_ap)
            target_ap = pd.concat(target_ap_parts, ignore_index=True)
        else:
            metadata = df_consensus.filter(regex="^Metadata")
            profiles = df_consensus[features].values
            target_ap = copairs_map.multilabel.average_precision(
                metadata, profiles,
                pos_sameby_target, pos_diffby_target,
                neg_sameby_target, neg_diffby_target,
                multilabel_col="Metadata_target",
            )

        target_map = copairs_map.mean_average_precision(
            target_ap,
            pos_sameby_target,
            null_size=null_size,
            threshold=p_threshold,
            seed=seed,
        ).copy()

        target_map["below_corrected_p"] = target_map["corrected_p_value"] < p_threshold

        n_targets_active = target_map["below_corrected_p"].sum()
        n_targets_total = len(target_map)
        pct_targets_active = (
            (n_targets_active / n_targets_total * 100) if n_targets_total > 0 else 0.0
        )

        # Calculate per-group summary if groups exist
        group_summary = None
        if has_groups and group_col in target_map.columns:
            group_summary = target_map.groupby(group_col).agg(
                pct_active=("below_corrected_p", "mean"),
                num_active=("below_corrected_p", "sum"),
                mean_normalized_average_precision=("mean_normalized_average_precision", "mean"),
                median_normalized_average_precision=("mean_normalized_average_precision", "median"),
                n_targets=("Metadata_target", "nunique"),
            ).reset_index()
            group_summary["pct_active"] *= 100  # Convert to percentage

        # Calculate overall mean normalized average precision
        mean_nap = float(target_map["mean_normalized_average_precision"].mean())
        median_nap = float(target_map["mean_normalized_average_precision"].median())

        return {
            "target_consistency": target_map,
            "pct_targets_active": float(pct_targets_active),
            "n_targets_active": int(n_targets_active),
            "n_targets_total": int(n_targets_total),
            "group_summary": group_summary,
            "mean_normalized_average_precision": mean_nap,
            "median_normalized_average_precision": median_nap,
        }
    except Exception as e:
        print(f"Warning: Target consistency failed: {e}")
        return {
            "target_consistency": None,
            "pct_targets_active": 0.0,
            "n_targets_active": 0,
            "n_targets_total": 0,
            "group_summary": None,
        }


def evaluate_all(
    df: pl.DataFrame,
    features: list[str],
    output_dir: Path | str | None = None,
    skip_visualization: bool = False,
    skip_umap: bool = False,
    n_top_compounds: int = 20,
    min_compounds_per_target: int = 3,
    compound_col: str = "Metadata_pert_iname",
    target_col: str = "Metadata_target_list",
    negcon_col: str = "Metadata_negcon",
    batch_col: str = "Metadata_Plate",
    group_col: str = "Metadata_Group",
    pc_groups: list[str] | None = None,
) -> dict[str, Any]:
    """Run all metrics and optionally save results.

    Args:
        df: Normalized profiles
        features: Feature column names
        output_dir: Directory to save results (None = don't save)
        skip_visualization: Skip visualization generation
        skip_umap: Skip UMAP in visualization
        n_top_compounds: Number of compounds to highlight
        min_compounds_per_target: Minimum compounds required per target for PC
        compound_col: Column for compound identifier
        target_col: Column for target identifier
        negcon_col: Boolean column for negative control flag
        batch_col: Column for batch/plate identifier
        group_col: Column for group identifier (for per-group PA and PC statistics)
        pc_groups: If set, only compute PC for these groups (e.g. ["group_high", "group_low"])

    Returns:
        Dictionary with all metrics including per-group summaries
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    results = {}
    pa = {}  # Initialized here so PC_replicable can check it even if PA errors

    # Phenotypic Activity
    try:
        pa = calculate_phenotypic_activity(
            df, features,
            compound_col=compound_col,
            negcon_col=negcon_col,
            batch_col=batch_col,
            group_col=group_col,
        )
        results["PA"] = pa["pct_compounds_active"]
        results["n_compounds"] = pa["n_compounds"]
        results["PA_mean_nap"] = pa["mean_normalized_average_precision"]
        results["PA_median_nap"] = pa["median_normalized_average_precision"]
        print(f"  PA: {pa['pct_compounds_active']:.2f}%")
        print(f"  PA mean NAP: {pa['mean_normalized_average_precision']:.4f}")

        # Add per-group PA summary to results
        if pa.get("group_summary") is not None:
            results["PA_group_summary"] = pa["group_summary"].set_index(group_col).to_dict(orient="index")
            for group_name, group_stats in results["PA_group_summary"].items():
                print(f"    {group_name}: {group_stats['pct_active']:.2f}%")

        if output_dir is not None:
            if pa.get("activity_ap") is not None and len(pa["activity_ap"]) > 0:
                pa["activity_ap"].to_csv(output_dir / "phenotypic_activity_per_compound.csv", index=False)
            if pa.get("activity_map") is not None and len(pa["activity_map"]) > 0:
                pa["activity_map"].to_csv(output_dir / "phenotypic_activity_map.csv", index=False)
            if pa.get("group_summary") is not None and len(pa["group_summary"]) > 0:
                pa["group_summary"].to_csv(output_dir / "phenotypic_activity_group_summary.csv", index=False)
    except Exception as e:
        print(f"  PA ERROR: {e}")
        results["PA"] = 0.0
        results["n_compounds"] = 0

    # Phenotypic Consistency
    try:
        pc = calculate_phenotypic_consistency(
            df, features,
            min_compounds_per_target=min_compounds_per_target,
            compound_col=compound_col,
            target_col=target_col,
            negcon_col=negcon_col,
            group_col=group_col,
            pc_groups=pc_groups,
        )
        results["PC"] = pc["pct_targets_active"]
        results["n_targets_active"] = pc["n_targets_active"]
        results["n_targets_total"] = pc["n_targets_total"]
        results["PC_mean_nap"] = pc.get("mean_normalized_average_precision", 0.0)
        results["PC_median_nap"] = pc.get("median_normalized_average_precision", 0.0)
        print(f"  PC: {pc['pct_targets_active']:.1f}%")
        if pc.get("mean_normalized_average_precision") is not None:
            print(f"  PC mean NAP: {pc['mean_normalized_average_precision']:.4f}")

        # Add per-group PC summary to results
        if pc.get("group_summary") is not None:
            results["PC_group_summary"] = pc["group_summary"].set_index(group_col).to_dict(orient="index")
            for group_name, group_stats in results["PC_group_summary"].items():
                print(f"    {group_name}: {group_stats['pct_active']:.2f}%")

        if output_dir is not None:
            if pc.get("target_consistency") is not None:
                pc["target_consistency"].to_csv(output_dir / "phenotypic_consistency_per_target.csv", index=False)
            if pc.get("group_summary") is not None and len(pc["group_summary"]) > 0:
                pc["group_summary"].to_csv(output_dir / "phenotypic_consistency_group_summary.csv", index=False)
    except Exception as e:
        print(f"  PC ERROR: {e}")
        results["PC"] = 0.0
        results["n_targets_active"] = 0
        results["n_targets_total"] = 0
        results["PC_mean_nap"] = 0.0
        results["PC_median_nap"] = 0.0

    # Phenotypic Consistency on PA-replicable compounds only (Chandrasekaran-style gating)
    _pc_rep_defaults = {
        "PC_replicable": 0.0,
        "PC_replicable_n_targets_active": 0,
        "PC_replicable_n_targets_total": 0,
        "PC_replicable_mean_nap": 0.0,
        "PC_replicable_median_nap": 0.0,
        "PC_replicable_n_compounds": 0,
    }
    try:
        if pa.get("activity_map") is not None and len(pa["activity_map"]) > 0:
            replicable = pa["activity_map"].query("below_corrected_p == True")[compound_col].tolist()
            if len(replicable) >= 2:
                df_replicable = df.filter(pl.col(compound_col).is_in(replicable))
                print(f"  PC_replicable: {len(replicable)} PA-significant compounds")
                pc_rep = calculate_phenotypic_consistency(
                    df_replicable, features,
                    min_compounds_per_target=min_compounds_per_target,
                    compound_col=compound_col,
                    target_col=target_col,
                    negcon_col=negcon_col,
                    group_col=group_col,
                    pc_groups=pc_groups,
                )
                results["PC_replicable"] = pc_rep["pct_targets_active"]
                results["PC_replicable_n_targets_active"] = pc_rep["n_targets_active"]
                results["PC_replicable_n_targets_total"] = pc_rep["n_targets_total"]
                results["PC_replicable_mean_nap"] = pc_rep.get("mean_normalized_average_precision", 0.0)
                results["PC_replicable_median_nap"] = pc_rep.get("median_normalized_average_precision", 0.0)
                results["PC_replicable_n_compounds"] = len(replicable)
                print(f"  PC_replicable: {pc_rep['pct_targets_active']:.1f}% ({pc_rep['n_targets_active']}/{pc_rep['n_targets_total']} targets)")
                if pc_rep.get("mean_normalized_average_precision") is not None:
                    print(f"  PC_replicable mean NAP: {pc_rep['mean_normalized_average_precision']:.4f}")

                if pc_rep.get("group_summary") is not None:
                    results["PC_replicable_group_summary"] = pc_rep["group_summary"].set_index(group_col).to_dict(orient="index")
                    for group_name, group_stats in results["PC_replicable_group_summary"].items():
                        print(f"    {group_name}: {group_stats['pct_active']:.2f}%")

                if output_dir is not None:
                    if pc_rep.get("target_consistency") is not None:
                        pc_rep["target_consistency"].to_csv(
                            output_dir / "phenotypic_consistency_replicable_per_target.csv", index=False
                        )
                    if pc_rep.get("group_summary") is not None and len(pc_rep["group_summary"]) > 0:
                        pc_rep["group_summary"].to_csv(
                            output_dir / "phenotypic_consistency_replicable_group_summary.csv", index=False
                        )
            else:
                print(f"  PC_replicable: skipped (only {len(replicable)} PA-significant compounds)")
                results.update(_pc_rep_defaults)
                results["PC_replicable_n_compounds"] = len(replicable)
        else:
            results.update(_pc_rep_defaults)
    except Exception as e:
        print(f"  PC_replicable ERROR: {e}")
        results.update(_pc_rep_defaults)

    # Add TVN state
    from norm_3.core import get_tvn_state
    tvn_ill_conditioned, tvn_max_condition_number = get_tvn_state()
    results["tvn_ill_conditioned"] = tvn_ill_conditioned
    results["tvn_max_condition_number"] = float(tvn_max_condition_number) if tvn_max_condition_number > 0 else None
    if tvn_ill_conditioned:
        print(f"  WARNING: TVN encountered ill-conditioned matrix (condition number: {tvn_max_condition_number:.2e})")

    # Compute PCA variance (PC1 and PC2 explained variance)
    try:
        from norm_3.core import PCATransform
        from norm_3.utils import to_gpu, to_cpu

        X = df.select(features).to_numpy()
        if np.isnan(X).any() or np.isinf(X).any():
            print("  WARNING: NaN/Inf detected in features for PCA computation")
            results["PC1_variance"] = None
            results["PC2_variance"] = None
        else:
            X_gpu = to_gpu(X)
            pca = PCATransform(n_components=2)
            pca.fit(X_gpu)
            results["PC1_variance"] = float(pca.explained_variance_ratio_[0])
            results["PC2_variance"] = float(pca.explained_variance_ratio_[1])
            print(f"  PC1 variance: {results['PC1_variance']*100:.2f}%")
            print(f"  PC2 variance: {results['PC2_variance']*100:.2f}%")
    except Exception as e:
        print(f"  PCA variance ERROR: {e}")
        results["PC1_variance"] = None
        results["PC2_variance"] = None

    # Add feature space size
    results["n_features"] = len(features)
    print(f"  Feature space size: {len(features)}")

    # Save metrics
    if output_dir is not None:
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)

    return results
