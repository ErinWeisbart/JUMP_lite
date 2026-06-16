#!/usr/bin/env python3
"""Per-gene normalized AP via copairs (cross-modality compound vs gene).

For each MOTIVE-target gene, compute the AP of ranking target-matching
compounds above other compounds in the same compound group. Run separately
for the four (compound_group, target_modality) combinations:
    {group_high, group_low} x {orf, crispr}

Implementation note: copairs's pair logic is symmetric and column-based; an
explosion-based approach (one row per (compound, target)) introduces a self-
cancel bias for multi-target compounds (compound A's "FOO" row is positive
for gene FOO, but its "BAR" row is also a negative for gene FOO at the same
similarity, deflating compound A's signal). Multilabel avoids the bias but
is ~30x slower due to a per-label Python loop in copairs.

Instead, this script computes per-gene AP directly from the cosine similarity
matrix (gene profiles x compound profiles), with each compound counted once
per gene query (positive iff that compound's MOTIVE target list contains the
gene's symbol, otherwise negative). This matches multilabel semantics
exactly while staying fast. Normalization uses copairs.map.normalization.
normalize_ap so numbers are comparable to the rest of the eval.

Gene-as-query AP only is reported (compound-row AP is discarded post-hoc).
We do not aggregate via mean_average_precision -- with consensus profiles
each gene is one row, so per-target "mean" is over a single element. P-values
+ BH correction are opt-in via --with-pvalues (off by default; the null
sampling dominates wall time).

CLI:
    uv run python evaluation/evaluate_motive_pc_target.py \\
        --input  <profiles.parquet> \\
        --output <out_dir>/ \\
        --annotations metadata/motive_annotations.parquet \\
        --splits      metadata/motive_splits.parquet

Outputs into ``--output``:
- ``metrics.json``                       (presence triggers idempotency)
- ``motive_pc_target_per_gene.csv``      one row per (compound_group,
                                          target_modality, gene JCP)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from evaluate_cross_modality_retrieval import (
    compute_cosine_similarity_matrix,
    get_consensus_profiles,
)
from evaluate_motive import (
    COMPOUND_GROUPS,
    TARGET_MODALITIES,
    _attach_motive_target_list,
    _restrict_compounds_to_split,
)
from evaluate_phenotypic_activity import (
    get_numeric_features,
    infer_columns,
    load_profiles,
    merge_metadata,
    setup_control_columns,
)


PER_GENE_CSV = "motive_pc_target_per_gene.csv"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _attach_target_list_for_genes(
    df: pl.DataFrame, cg_ann: pl.DataFrame, compound_col: str
) -> pl.DataFrame:
    """Attach pipe-joined ``Metadata_target_list`` for gene JCPs.

    Mirror of ``_attach_motive_target_list`` but joins on ``cg_ann.partner_jcp``
    so each gene JCP gets the symbols it resolves from. One symbol per JCP is
    typical; the unique().sort().str.join('|') is defensive against a JCP
    appearing under multiple symbols in cg_ann.
    """
    target_list = (
        cg_ann.filter(pl.col("partner_jcp").is_not_null())
        .group_by(pl.col("partner_jcp").alias(compound_col))
        .agg(
            pl.col("target")
            .unique()
            .sort()
            .str.join("|")
            .alias("Metadata_target_list")
        )
    )
    return df.join(target_list, on=compound_col, how="left")


def _ap_per_query(sim: np.ndarray, pos_mask: np.ndarray) -> np.ndarray:
    """Per-query AP from a (N_q, N_r) similarity matrix and bool positive mask.

    For each query (row), rank references (columns) by descending similarity,
    walk the ranking, and compute the standard AP definition
    ``mean over positive ranks of precision@k``. Returns NaN for queries with
    zero positives.
    """
    N_q, N_r = sim.shape
    aps = np.full(N_q, np.nan, dtype=np.float64)
    ranks = np.arange(1, N_r + 1, dtype=np.float64)
    for i in range(N_q):
        n_pos = int(pos_mask[i].sum())
        if n_pos == 0:
            continue
        order = np.argsort(-sim[i], kind="stable")
        pos_at_rank = pos_mask[i][order].astype(np.float64)
        cumsum = np.cumsum(pos_at_rank)
        precisions = cumsum / ranks
        aps[i] = float((pos_at_rank * precisions).sum() / n_pos)
    return aps


def _gate_genes_by_compound_count(
    df_pair_pd: pd.DataFrame,
    min_compounds_per_target: int,
) -> pd.DataFrame:
    """Drop gene rows whose target symbol has < min compounds in this run.

    Counts unique compound JCPs per gene symbol. Genes with too few annotated
    compounds in the current compound_group are removed from ``df_pair_pd``;
    compound rows are kept (they still serve as the negative pool).

    ``Metadata_target`` is expected to be a Python list of symbol strings
    (already split by '|').
    """
    compounds = df_pair_pd[df_pair_pd["Metadata_modality"] == "compound"]
    genes = df_pair_pd[df_pair_pd["Metadata_modality"] == "gene"]

    if compounds.empty or genes.empty:
        return df_pair_pd

    # explode compound→target to count unique compounds per symbol
    comp_exploded = compounds[["Metadata_JCP2022", "Metadata_target"]].explode(
        "Metadata_target"
    )
    counts = comp_exploded.groupby("Metadata_target")["Metadata_JCP2022"].nunique()
    valid_targets = set(counts[counts >= min_compounds_per_target].index)

    def gene_has_valid_target(target_list):
        if not isinstance(target_list, list):
            return False
        return any(t in valid_targets for t in target_list)

    keep_genes = genes[genes["Metadata_target"].apply(gene_has_valid_target)]
    return pd.concat([compounds, keep_genes], ignore_index=True)


# ---------------------------------------------------------------------------
# core: one (compound_group, target_modality) run
# ---------------------------------------------------------------------------


def run_one_setting(
    df_use: pl.DataFrame,
    features: list[str],
    cg_ann: pl.DataFrame,
    compound_group: str,
    target_modality: str,
    compound_col: str,
    group_col: str,
    null_size: int,
    p_threshold: float,
    seed: int,
    min_compounds_per_target: int,
    with_pvalues: bool,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Run per-gene AP for one (compound_group, target_modality)."""
    from copairs.map.normalization import normalize_ap

    print(f"\n=== {compound_group} x {target_modality} ===")

    # 1. compound side: rows in this group with at least one MOTIVE target
    df_comp = df_use.filter(pl.col(group_col) == compound_group)
    df_comp = _attach_motive_target_list(df_comp, cg_ann, compound_col)
    df_comp = df_comp.filter(pl.col("Metadata_target_list").is_not_null())

    # 2. gene side: rows in this modality whose JCP appears as a partner_jcp
    ref_group = f"group_{target_modality}"
    df_gene = df_use.filter(pl.col(group_col) == ref_group)
    df_gene = _attach_target_list_for_genes(df_gene, cg_ann, compound_col)
    df_gene = df_gene.filter(pl.col("Metadata_target_list").is_not_null())

    if df_comp.is_empty() or df_gene.is_empty():
        print("  empty side; skipping")
        return None, {"skipped": True}

    # 3. union and mark modality
    df_pair = pl.concat([df_comp, df_gene], how="vertical_relaxed")
    df_pair = df_pair.with_columns(
        pl.when(pl.col(group_col).is_in(list(COMPOUND_GROUPS)))
        .then(pl.lit("compound"))
        .otherwise(pl.lit("gene"))
        .alias("Metadata_modality")
    )

    # 4. consensus profiles (one row per JCP). get_consensus_profiles carries
    # all Metadata_* columns through (see evaluate_cross_modality_retrieval.py:114).
    meta_pl, feat = get_consensus_profiles(df_pair, features, compound_col, group_col)
    meta_pd = meta_pl.to_pandas()
    meta_pd["Metadata_target"] = meta_pd["Metadata_target_list"].str.split("|")

    # 5. min-compounds-per-target gate
    n_genes_before = (meta_pd["Metadata_modality"] == "gene").sum()
    meta_pd = _gate_genes_by_compound_count(meta_pd, min_compounds_per_target)
    n_genes_after = (meta_pd["Metadata_modality"] == "gene").sum()
    n_compounds = (meta_pd["Metadata_modality"] == "compound").sum()
    print(
        f"  consensus: {n_compounds} compounds, "
        f"{n_genes_after}/{n_genes_before} genes after min-compounds gate"
    )
    if n_genes_after == 0:
        return None, {"skipped": True, "reason": "no genes after gating"}

    # realign feat with the kept rows
    feat = feat[meta_pd.index.values]
    meta_pd = meta_pd.reset_index(drop=True)

    # 6. split rows by modality
    is_gene = meta_pd["Metadata_modality"].to_numpy() == "gene"
    gene_meta = meta_pd[is_gene].reset_index(drop=True)
    compound_meta = meta_pd[~is_gene].reset_index(drop=True)
    gene_feat = feat[is_gene]
    compound_feat = feat[~is_gene]
    if len(gene_meta) == 0 or len(compound_meta) == 0:
        return None, {"skipped": True, "reason": "no gene or no compound rows"}

    # 7. cosine similarity: (N_genes, N_compounds)
    sim = compute_cosine_similarity_matrix(gene_feat, compound_feat)

    # 8. positive mask: True iff compound[j]'s target list contains any of
    # gene[i]'s symbols. With one symbol per gene typically, this is a simple
    # set-membership check; multi-symbol gene rows do an intersection check.
    gene_targets = gene_meta["Metadata_target"].tolist()  # list of lists
    compound_target_sets = [
        set(t) if isinstance(t, list) else set()
        for t in compound_meta["Metadata_target"].tolist()
    ]
    N_genes = len(gene_meta)
    N_compounds = len(compound_meta)
    pos_mask = np.zeros((N_genes, N_compounds), dtype=bool)
    for i, gt in enumerate(gene_targets):
        gt_set = set(gt) if isinstance(gt, list) else set()
        if not gt_set:
            continue
        for j, ct_set in enumerate(compound_target_sets):
            if gt_set & ct_set:
                pos_mask[i, j] = True

    # 9. per-gene AP + normalized AP
    ap_scores = _ap_per_query(sim, pos_mask)
    n_pos_arr = pos_mask.sum(axis=1)
    n_total_arr = np.full(N_genes, N_compounds, dtype=np.int64)
    n_neg_arr = n_total_arr - n_pos_arr

    norm_ap = np.full(N_genes, np.nan, dtype=np.float64)
    valid = (n_pos_arr > 0) & (n_neg_arr > 0)
    if valid.any():
        norm_ap[valid] = normalize_ap(
            ap_scores[valid], M=n_pos_arr[valid], N=n_neg_arr[valid]
        )

    # Clip negative normalized AP to 0 (negative = worse than random, often
    # treated as zero for downstream aggregation).
    norm_ap_clipped = np.where(np.isnan(norm_ap), np.nan, np.maximum(norm_ap, 0.0))

    gene_ap = pd.DataFrame({
        compound_col: gene_meta[compound_col].values,
        "Metadata_target": gene_meta["Metadata_target_list"].values,
        "average_precision": ap_scores,
        "normalized_average_precision": norm_ap,
        "normalized_average_precision_clipped": norm_ap_clipped,
        "n_pos_pairs": n_pos_arr.astype(np.uint32),
        "n_total_pairs": n_total_arr.astype(np.uint32),
    })

    # 8. optional p-values + BH correction (off by default; null sampling is slow)
    gene_ap["p_value"] = np.nan
    gene_ap["corrected_p_value"] = np.nan
    gene_ap["below_corrected_p"] = False
    mask = gene_ap["n_pos_pairs"] > 0
    if with_pvalues and mask.any():
        from copairs.map.average_precision import p_values as copairs_p_values
        from statsmodels.stats.multitest import multipletests

        gene_ap.loc[mask, "p_value"] = copairs_p_values(
            gene_ap.loc[mask], null_size=null_size, seed=seed,
        )
        valid = mask & gene_ap["p_value"].notna()
        if valid.any():
            gene_ap.loc[valid, "corrected_p_value"] = multipletests(
                gene_ap.loc[valid, "p_value"].values, method="fdr_bh"
            )[1]
        gene_ap["below_corrected_p"] = (
            gene_ap["corrected_p_value"] < p_threshold
        ).fillna(False)

    gene_ap["compound_group"] = compound_group
    gene_ap["target_modality"] = target_modality

    # 9. summary
    n_total = int(len(gene_ap))
    n_with_pos = int(mask.sum())
    median_norm_ap = (
        float(gene_ap.loc[mask, "normalized_average_precision"].median())
        if mask.any()
        else None
    )
    median_norm_ap_clipped = (
        float(gene_ap.loc[mask, "normalized_average_precision_clipped"].median())
        if mask.any()
        else None
    )
    mean_norm_ap_clipped = (
        float(gene_ap.loc[mask, "normalized_average_precision_clipped"].mean())
        if mask.any()
        else None
    )
    median_ap = (
        float(gene_ap.loc[mask, "average_precision"].median())
        if mask.any()
        else None
    )
    summary: dict[str, Any] = {
        "n_genes_total": n_total,
        "n_genes_with_positives": n_with_pos,
        "median_normalized_ap": median_norm_ap,
        "median_normalized_ap_clipped": median_norm_ap_clipped,
        "mean_normalized_ap_clipped": mean_norm_ap_clipped,
        "median_average_precision": median_ap,
        "n_compounds_in_pool": int(n_compounds),
    }
    if with_pvalues:
        n_active = int(gene_ap["below_corrected_p"].sum())
        summary["n_genes_active"] = n_active
        summary["pct_genes_active"] = (n_active / n_total) if n_total else None
        print(
            f"  {n_with_pos}/{n_total} genes with positives, "
            f"{n_active} active (corrected p < {p_threshold}), "
            f"median norm AP = {median_norm_ap}"
        )
    else:
        print(
            f"  {n_with_pos}/{n_total} genes with positives, "
            f"median norm AP = {median_norm_ap} (p-values skipped)"
        )
    return gene_ap, summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("metadata/motive_annotations.parquet"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("metadata/motive_splits.parquet"),
    )
    parser.add_argument("--split-eval", type=str, default="test")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("metadata/metadata_dataset_filtered_4reps.parquet"),
        help="Used only if Metadata_JCP2022 is missing from the input parquet.",
    )
    parser.add_argument("--compound-col", type=str, default="Metadata_JCP2022")
    parser.add_argument("--group-col", type=str, default="Metadata_Group")
    parser.add_argument("--negcon-col", type=str, default="Metadata_negcon")
    parser.add_argument("--null-size", type=int, default=10_000)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-compounds-per-target", type=int, default=3)
    parser.add_argument(
        "--with-pvalues",
        action="store_true",
        help=(
            "Compute null-distribution p-values and BH-corrected p-values "
            "(opt-in; null sampling dominates wall time). Off by default — "
            "the per-gene CSV still includes AP and normalized AP."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing metrics.json / per-gene CSV.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.json"
    per_gene_path = args.output / PER_GENE_CSV

    if metrics_path.exists() and not args.force:
        print(f"[skip] metrics.json already exists at {metrics_path}")
        return
    if per_gene_path.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite existing {per_gene_path}. Pass --force to opt in."
        )

    # ---------------- load profiles ----------------
    print(f"[load] {args.input}")
    df = load_profiles(args.input)
    print(f"[load] shape={df.shape}")
    if args.compound_col not in df.columns:
        print(
            f"[load] {args.compound_col} missing - merging metadata from "
            f"{args.metadata}"
        )
        df = merge_metadata(df, args.metadata)
    df = setup_control_columns(df)
    df = df.filter(pl.col(args.compound_col).is_not_null())

    features, _ = infer_columns(df)
    features = get_numeric_features(df, features)
    if not features:
        raise ValueError("no numeric feature columns found in input parquet")
    print(f"[features] {len(features)} numeric features")

    # ---------------- load MOTIVE files ----------------
    if not args.annotations.exists():
        raise FileNotFoundError(
            f"motive annotations not found: {args.annotations}. "
            "Run `just motive-curate` first."
        )
    if not args.splits.exists():
        raise FileNotFoundError(
            f"motive splits not found: {args.splits}. "
            "Run `just motive-curate <splits-path>` first."
        )
    annotations = pl.read_parquet(args.annotations)
    splits = pl.read_parquet(args.splits)
    cg_ann = annotations.filter(pl.col("source") == "cg")
    print(
        f"[motive] cg rows={cg_ann.height:,}  splits rows={splits.height:,}"
    )

    # ---------------- restrict to eval split ----------------
    df = _restrict_compounds_to_split(df, splits, args.split_eval, args.compound_col)
    if df.height == 0:
        raise ValueError("no rows remain after split restriction; aborting")

    # ---------------- run all four settings ----------------
    df_use = df.filter(pl.col(args.negcon_col) == False)

    metrics: dict[str, Any] = {
        "input_file": str(args.input),
        "split_eval": args.split_eval,
        "seed": args.seed,
        "n_features": len(features),
        "min_compounds_per_target": args.min_compounds_per_target,
        "with_pvalues": args.with_pvalues,
        "settings": {},
    }
    per_gene_dfs: list[pd.DataFrame] = []

    for compound_group in COMPOUND_GROUPS:
        for modality in TARGET_MODALITIES:
            gene_ap, summary = run_one_setting(
                df_use=df_use,
                features=features,
                cg_ann=cg_ann,
                compound_group=compound_group,
                target_modality=modality,
                compound_col=args.compound_col,
                group_col=args.group_col,
                null_size=args.null_size,
                p_threshold=args.p_threshold,
                seed=args.seed,
                min_compounds_per_target=args.min_compounds_per_target,
                with_pvalues=args.with_pvalues,
            )
            metrics["settings"][f"{compound_group}__{modality}"] = summary
            if gene_ap is not None:
                per_gene_dfs.append(gene_ap)

    # ---------------- write outputs ----------------
    if per_gene_dfs:
        out = pd.concat(per_gene_dfs, ignore_index=True)
        keep = [
            "compound_group",
            "target_modality",
            args.compound_col,
            "Metadata_target",
            "average_precision",
            "normalized_average_precision",
            "normalized_average_precision_clipped",
            "n_pos_pairs",
            "n_total_pairs",
            "p_value",
            "corrected_p_value",
            "below_corrected_p",
        ]
        keep = [c for c in keep if c in out.columns]
        out = out[keep]
        out.to_csv(per_gene_path, index=False)
        print(f"\n[write] {per_gene_path}  ({len(out)} rows)")

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[write] {metrics_path}")


if __name__ == "__main__":
    main()
