"""Numerical comparison of two sweep output.parquet files.

Prints column-by-column max-abs-diff and relative-diff stats. Exits 0 if
all numeric columns match within `--atol`, else 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def compare(slice_path: Path, ref_path: Path, atol: float = 1e-6) -> int:
    df_s = pd.read_parquet(slice_path)
    df_r = pd.read_parquet(ref_path)

    if df_s.shape != df_r.shape:
        print(f"    SHAPE MISMATCH: slice={df_s.shape} ref={df_r.shape}")
        return 1
    only_s = set(df_s.columns) - set(df_r.columns)
    only_r = set(df_r.columns) - set(df_s.columns)
    if only_s or only_r:
        print(f"    COLUMN MISMATCH: only_slice={sorted(only_s)[:5]} only_ref={sorted(only_r)[:5]}")
        return 1
    # Align column order so list-mismatch (with same set) doesn't trip us up.
    df_r = df_r[df_s.columns]

    numeric_cols = df_s.select_dtypes(include=[np.number]).columns
    worst_col = None
    worst_diff = 0.0
    n_diff = 0
    for col in numeric_cols:
        d = np.abs(df_s[col].to_numpy() - df_r[col].to_numpy())
        m = np.nanmax(d) if d.size else 0.0
        if m > worst_diff:
            worst_diff = m
            worst_col = col
        if m > atol:
            n_diff += 1

    print(f"    rows={len(df_s)} numeric_cols={len(numeric_cols)} cols_above_atol={n_diff}/{len(numeric_cols)}")
    print(f"    worst column: {worst_col} max_abs_diff={worst_diff:.3e}")
    return 0 if n_diff == 0 else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("slice_path", type=Path)
    p.add_argument("ref_path", type=Path)
    p.add_argument("--atol", type=float, default=1e-6)
    args = p.parse_args()
    sys.exit(compare(args.slice_path, args.ref_path, args.atol))


if __name__ == "__main__":
    main()
