#!/usr/bin/env python3
"""Gather all metrics.json files from a norm_3 sweep into a single CSV table with plots.

Designed for the norm_3 variance-first pipeline config naming convention:
  {norm}_ctrl__outlier{cutoff}__INT__prune{thresh}[__pca{n}][__batch_method]

Batch methods:
  - none (no suffix)
  - tvn_original_k{k}
  - tvn_efaar_e{epsilon}[_c{n_components}]  (c suffix only when != 128)
  - cascade_tvn_k{k1}_k{k2}
  - ZCA-cor_{fit}_{epsilon}  (spherize)

Usage:
  python src/norm_3/gather_sweep_results.py --sweep-dir src/norm_3/data/features/variance_first_v4 --plot
  python src/norm_3/gather_sweep_results.py --sweep-dir src/norm_3/data/features/variance_first_v4 --plot --filter-degenerate
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# Display names for compression codecs
COMPRESSION_DISPLAY = {
    "raw_jump_cp_profiles_reformatted_filtered": "raw",
    "zstd_raw": "zstd",
    "zstd_filtered_raw": "zstd_f",
    "jpegxl_lossy_hq_raw": "hq",
    "jpegxl_lossy_hq_filtered_raw": "hq_f",
    "jpegxl_lossy_effort_3_raw": "effort_3",
    "jpegxl_lossy_effort_3_filtered_raw": "effort_3_f",
    "jpegxl_lossy_mq_raw": "mq",
    "jpegxl_lossy_mq_filtered_raw": "mq_f",
    "jpegxl_lossy_lq_raw": "lq",
    "jpegxl_lossy_lq_filtered_raw": "lq_f",
    "jpegxl_lossy_d2_e8_raw": "d2_e8",
    "jpegxl_lossy_d2_e8_filtered_raw": "d2_e8_f",
    "jpegxl_lossy_d10_raw": "d10",
    "jpegxl_lossy_d10_filtered_raw": "d10_f",
    # Embedding models: DINOv2-490
    "dinov2_490_zstd_raw": "dv2_490_zstd",
    "dinov2_490_jpegxl_lossy_hq_raw": "dv2_490_hq",
    "dinov2_490_jpegxl_lossy_effort_3_raw": "dv2_490_e3",
    "dinov2_490_jpegxl_lossy_mq_raw": "dv2_490_mq",
    "dinov2_490_jpegxl_lossy_lq_raw": "dv2_490_lq",
    # Embedding models: DINOv2-random
    "dinov2_random_zstd_raw": "dv2_rand_zstd",
    "dinov2_random_jpegxl_lossy_hq_raw": "dv2_rand_hq",
    "dinov2_random_jpegxl_lossy_effort_3_raw": "dv2_rand_e3",
    "dinov2_random_jpegxl_lossy_mq_raw": "dv2_rand_mq",
    "dinov2_random_jpegxl_lossy_lq_raw": "dv2_rand_lq",
    # Embedding models: MorphEm
    "morphem_zstd_raw": "morphem_zstd",
    "morphem_jpegxl_lossy_hq_raw": "morphem_hq",
    "morphem_jpegxl_lossy_effort_3_raw": "morphem_e3",
    "morphem_jpegxl_lossy_mq_raw": "morphem_mq",
    "morphem_jpegxl_lossy_lq_raw": "morphem_lq",
    # Embedding models: SubCell
    "subcell_zstd_raw": "subcell_zstd",
    "subcell_jpegxl_lossy_hq_raw": "subcell_hq",
    "subcell_jpegxl_lossy_effort_3_raw": "subcell_e3",
    "subcell_jpegxl_lossy_mq_raw": "subcell_mq",
    "subcell_jpegxl_lossy_lq_raw": "subcell_lq",
    # Embedding models: OpenPhenom-8bit
    "openphenom_8bit_zstd_raw": "ophenom_zstd",
    "openphenom_8bit_jpegxl_lossy_hq_raw": "ophenom_hq",
    "openphenom_8bit_jpegxl_lossy_effort_3_raw": "ophenom_e3",
    "openphenom_8bit_jpegxl_lossy_mq_raw": "ophenom_mq",
    "openphenom_8bit_jpegxl_lossy_lq_raw": "ophenom_lq",
    # CellProfiler filtered_border_size (cp_measure from jump_target2_4plate_filtered)
    "zstd_filtered_border_size_raw": "cp_fbs_zstd",
    "jpegxl_lossy_hq_filtered_border_size_raw": "cp_fbs_hq",
    "jpegxl_lossy_effort_3_filtered_border_size_raw": "cp_fbs_e3",
    "jpegxl_lossy_mq_filtered_border_size_raw": "cp_fbs_mq",
    "jpegxl_lossy_lq_filtered_border_size_raw": "cp_fbs_lq",
    "jpegxl_lossy_d2_e8_filtered_border_size_raw": "cp_fbs_d2_e8",
    "jpegxl_lossy_d10_filtered_border_size_raw": "cp_fbs_d10",
    # Embedding models: DINOv2-CL (dinov2 without 490/random, from jump_target2_4plate_cl)
    "dinov2_cl_jpegxl_lossy_d2_e8_raw": "dv2_cl_d2_e8",
    "dinov2_cl_jpegxl_lossy_d10_raw": "dv2_cl_d10",
    # Embedding models: DINOv2-random d2_e8/d10 (new codecs from jump_target2_4plate_cl)
    "dinov2_random_jpegxl_lossy_d2_e8_raw": "dv2_rand_d2_e8",
    "dinov2_random_jpegxl_lossy_d10_raw": "dv2_rand_d10",
    # Embedding models: SubCell d2_e8/d10 (new codecs from jump_target2_4plate_cl)
    "subcell_jpegxl_lossy_d2_e8_raw": "subcell_d2_e8",
    "subcell_jpegxl_lossy_d10_raw": "subcell_d10",
    # Embedding models: MorphEm d2_e8/d10 (new codecs from jump_target2_4plate_cl)
    "morphem_jpegxl_lossy_d2_e8_raw": "morphem_d2_e8",
    "morphem_jpegxl_lossy_d10_raw": "morphem_d10",
    "morphem_jpegxl_lossy_d15_raw": "morphem_d15",
    "morphem_jpegxl_lossy_d20_e2_raw": "morphem_d20_e2",
    # Embedding models: OpenPhenom (non-8bit, from jump_target2_4plate_cl)
    "openphenom_zstd_raw": "ophenom_cl_zstd",
    "openphenom_jpegxl_lossy_hq_raw": "ophenom_cl_hq",
    "openphenom_jpegxl_lossy_effort_3_raw": "ophenom_cl_e3",
    "openphenom_jpegxl_lossy_mq_raw": "ophenom_cl_mq",
    "openphenom_jpegxl_lossy_lq_raw": "ophenom_cl_lq",
    "openphenom_jpegxl_lossy_d2_e8_raw": "ophenom_cl_d2_e8",
    "openphenom_jpegxl_lossy_d10_raw": "ophenom_cl_d10",
    "openphenom_jpegxl_lossy_d15_raw": "ophenom_cl_d15",
    "openphenom_jpegxl_lossy_d20_e2_raw": "ophenom_cl_d20_e2",
    # DINOv2-random d15/d20_e2/d30
    "dinov2_random_jpegxl_lossy_d15_raw": "dv2_rand_d15",
    "dinov2_random_jpegxl_lossy_d20_e2_raw": "dv2_rand_d20_e2",
    "dinov2_random_jpegxl_lossy_d30_raw": "dv2_rand_d30",
    # SubCell d15/d20_e2/d30
    "subcell_jpegxl_lossy_d15_raw": "subcell_d15",
    "subcell_jpegxl_lossy_d20_e2_raw": "subcell_d20_e2",
    "subcell_jpegxl_lossy_d30_raw": "subcell_d30",
    # MorphEm d30
    "morphem_jpegxl_lossy_d30_raw": "morphem_d30",
    # OpenPhenom d30
    "openphenom_jpegxl_lossy_d30_raw": "ophenom_cl_d30",
    # DINOv2 non-random (from jump_target2_4plate_cl)
    "dinov2_jpegxl_lossy_d2_e8_raw": "dv2_d2_e8",
    "dinov2_jpegxl_lossy_d10_raw": "dv2_d10",
    "dinov2_jpegxl_lossy_d15_raw": "dv2_d15",
    "dinov2_jpegxl_lossy_d20_e2_raw": "dv2_d20_e2",
    "dinov2_jpegxl_lossy_d30_raw": "dv2_d30",
    "dinov2_jpegxl_lossy_mq_new_raw": "dv2_mq_new",
    # mq_new codec variants
    "dinov2_random_jpegxl_lossy_mq_new_raw": "dv2_rand_mq_new",
    "morphem_jpegxl_lossy_mq_new_raw": "morphem_mq_new",
    "subcell_jpegxl_lossy_mq_new_raw": "subcell_mq_new",
    "openphenom_jpegxl_lossy_mq_new_raw": "ophenom_cl_mq_new",
    # === Rerun families (from jump_target2_4plate_cl_rerun/) ===
    # DINOv2 rerun
    "dinov2_cl_zstd_rr_raw": "dv2_rr_zstd",
    "dinov2_cl_jpegxl_lossy_hq_rr_raw": "dv2_rr_hq",
    "dinov2_cl_jpegxl_lossy_effort_3_rr_raw": "dv2_rr_e3",
    "dinov2_cl_jpegxl_lossy_mq_rr_raw": "dv2_rr_mq",
    "dinov2_cl_jpegxl_lossy_mq_new_rr_raw": "dv2_rr_mq_new",
    "dinov2_cl_jpegxl_lossy_lq_rr_raw": "dv2_rr_lq",
    "dinov2_cl_jpegxl_lossy_d2_e8_rr_raw": "dv2_rr_d2_e8",
    "dinov2_cl_jpegxl_lossy_d10_rr_raw": "dv2_rr_d10",
    "dinov2_cl_jpegxl_lossy_d15_rr_raw": "dv2_rr_d15",
    "dinov2_cl_jpegxl_lossy_d20_e2_rr_raw": "dv2_rr_d20_e2",
    "dinov2_cl_jpegxl_lossy_d30_rr_raw": "dv2_rr_d30",
    "dinov2_cl_jpegxl_lossy_d50_rr_raw": "dv2_rr_d50",
    # DINOv2-random rerun
    "dinov2_random_zstd_rr_raw": "dv2_rand_rr_zstd",
    "dinov2_random_jpegxl_lossy_hq_rr_raw": "dv2_rand_rr_hq",
    "dinov2_random_jpegxl_lossy_effort_3_rr_raw": "dv2_rand_rr_e3",
    "dinov2_random_jpegxl_lossy_mq_rr_raw": "dv2_rand_rr_mq",
    "dinov2_random_jpegxl_lossy_mq_new_rr_raw": "dv2_rand_rr_mq_new",
    "dinov2_random_jpegxl_lossy_lq_rr_raw": "dv2_rand_rr_lq",
    "dinov2_random_jpegxl_lossy_d2_e8_rr_raw": "dv2_rand_rr_d2_e8",
    "dinov2_random_jpegxl_lossy_d10_rr_raw": "dv2_rand_rr_d10",
    "dinov2_random_jpegxl_lossy_d15_rr_raw": "dv2_rand_rr_d15",
    "dinov2_random_jpegxl_lossy_d20_e2_rr_raw": "dv2_rand_rr_d20_e2",
    "dinov2_random_jpegxl_lossy_d30_rr_raw": "dv2_rand_rr_d30",
    "dinov2_random_jpegxl_lossy_d50_rr_raw": "dv2_rand_rr_d50",
    # MorphEm rerun
    "morphem_zstd_rr_raw": "morphem_rr_zstd",
    "morphem_jpegxl_lossy_hq_rr_raw": "morphem_rr_hq",
    "morphem_jpegxl_lossy_effort_3_rr_raw": "morphem_rr_e3",
    "morphem_jpegxl_lossy_mq_rr_raw": "morphem_rr_mq",
    "morphem_jpegxl_lossy_mq_new_rr_raw": "morphem_rr_mq_new",
    "morphem_jpegxl_lossy_lq_rr_raw": "morphem_rr_lq",
    "morphem_jpegxl_lossy_d2_e8_rr_raw": "morphem_rr_d2_e8",
    "morphem_jpegxl_lossy_d10_rr_raw": "morphem_rr_d10",
    "morphem_jpegxl_lossy_d15_rr_raw": "morphem_rr_d15",
    "morphem_jpegxl_lossy_d20_e2_rr_raw": "morphem_rr_d20_e2",
    "morphem_jpegxl_lossy_d30_rr_raw": "morphem_rr_d30",
    "morphem_jpegxl_lossy_d50_rr_raw": "morphem_rr_d50",
    # SubCell rerun
    "subcell_zstd_rr_raw": "subcell_rr_zstd",
    "subcell_jpegxl_lossy_hq_rr_raw": "subcell_rr_hq",
    "subcell_jpegxl_lossy_effort_3_rr_raw": "subcell_rr_e3",
    "subcell_jpegxl_lossy_mq_rr_raw": "subcell_rr_mq",
    "subcell_jpegxl_lossy_mq_new_rr_raw": "subcell_rr_mq_new",
    "subcell_jpegxl_lossy_lq_rr_raw": "subcell_rr_lq",
    "subcell_jpegxl_lossy_d2_e8_rr_raw": "subcell_rr_d2_e8",
    "subcell_jpegxl_lossy_d10_rr_raw": "subcell_rr_d10",
    "subcell_jpegxl_lossy_d15_rr_raw": "subcell_rr_d15",
    "subcell_jpegxl_lossy_d20_e2_rr_raw": "subcell_rr_d20_e2",
    "subcell_jpegxl_lossy_d30_rr_raw": "subcell_rr_d30",
    "subcell_jpegxl_lossy_d50_rr_raw": "subcell_rr_d50",
    # OpenPhenom rerun
    "openphenom_zstd_rr_raw": "ophenom_rr_zstd",
    "openphenom_jpegxl_lossy_hq_rr_raw": "ophenom_rr_hq",
    "openphenom_jpegxl_lossy_effort_3_rr_raw": "ophenom_rr_e3",
    "openphenom_jpegxl_lossy_mq_rr_raw": "ophenom_rr_mq",
    "openphenom_jpegxl_lossy_mq_new_rr_raw": "ophenom_rr_mq_new",
    "openphenom_jpegxl_lossy_lq_rr_raw": "ophenom_rr_lq",
    "openphenom_jpegxl_lossy_d2_e8_rr_raw": "ophenom_rr_d2_e8",
    "openphenom_jpegxl_lossy_d10_rr_raw": "ophenom_rr_d10",
    "openphenom_jpegxl_lossy_d15_rr_raw": "ophenom_rr_d15",
    "openphenom_jpegxl_lossy_d20_e2_rr_raw": "ophenom_rr_d20_e2",
    "openphenom_jpegxl_lossy_d30_rr_raw": "ophenom_rr_d30",
    "openphenom_jpegxl_lossy_d50_rr_raw": "ophenom_rr_d50",
    # DINOv2-CL d15/d20_e2/d30/mq_new
    "dinov2_cl_jpegxl_lossy_d15_raw": "dv2_cl_d15",
    "dinov2_cl_jpegxl_lossy_d20_e2_raw": "dv2_cl_d20_e2",
    "dinov2_cl_jpegxl_lossy_d30_raw": "dv2_cl_d30",
    "dinov2_cl_jpegxl_lossy_mq_new_raw": "dv2_cl_mq_new",
}

# Order for compression codecs (raw first, then filtered, by quality, then embedding models)
# Canonical codec ordering (lossless → heavy lossy) used to sort codecs within each model family.
# Maps codec substring patterns found in raw model names to a sort rank.
_CODEC_SORT_ORDER = {
    "raw_jump_cp_profiles": 0,  # special: reformatted CP baseline
    "zstd": 1,
    "jpegxl_lossy_hq": 2,
    "jpegxl_lossy_effort_3": 3,
    "jpegxl_lossy_d2_e8": 4,
    "jpegxl_lossy_mq_new": 5,
    "jpegxl_lossy_mq": 6,
    "jpegxl_lossy_lq": 7,
    "jpegxl_lossy_d10": 8,
    "jpegxl_lossy_d15": 9,
    "jpegxl_lossy_d20_e2": 10,
    "jpegxl_lossy_d30": 11,
    "jpegxl_lossy_d50": 12,
}


def _get_codec_sort_rank(model: str) -> int:
    """Extract codec sort rank from a raw model name.

    Tries each codec pattern (longest first to avoid partial matches like
    'jpegxl_lossy_d10' matching before 'jpegxl_lossy_d2_e8').
    """
    # Sort patterns longest-first so e.g. jpegxl_lossy_d2_e8 matches before jpegxl_lossy_d10
    for codec, rank in sorted(_CODEC_SORT_ORDER.items(), key=lambda kv: -len(kv[0])):
        if codec in model:
            return rank
    return 99  # unknown codec goes last

# Batch method display names
BATCH_DISPLAY = {
    "none": "None",
    "tvn_original": "TVN Original",
    "tvn_efaar": "TVN EFAAR",
    "cascade_tvn": "Cascade TVN",
    "spherize": "Spherize",
}

# Model family groupings for color assignment
# Each family gets a distinct hue; codecs within a family get brightness variations
MODEL_FAMILIES = {
    # CellProfiler reformatted (original JUMP CP profiles)
    "cellprofiler": [
        "raw_jump_cp_profiles_reformatted_filtered",
    ],
    # cp_measure raw
    "cp_measure": [
        "zstd_raw", "jpegxl_lossy_hq_raw", "jpegxl_lossy_effort_3_raw",
        "jpegxl_lossy_d2_e8_raw", "jpegxl_lossy_mq_raw",
        "jpegxl_lossy_lq_raw", "jpegxl_lossy_d10_raw",
    ],
    # cp_measure filtered
    "cp_measure_filtered": [
        "zstd_filtered_raw", "jpegxl_lossy_hq_filtered_raw",
        "jpegxl_lossy_effort_3_filtered_raw", "jpegxl_lossy_d2_e8_filtered_raw",
        "jpegxl_lossy_mq_filtered_raw", "jpegxl_lossy_lq_filtered_raw",
        "jpegxl_lossy_d10_filtered_raw",
    ],
    # DINOv2-490
    "dinov2_490": [
        "dinov2_490_zstd_raw", "dinov2_490_jpegxl_lossy_hq_raw",
        "dinov2_490_jpegxl_lossy_effort_3_raw", "dinov2_490_jpegxl_lossy_mq_raw",
        "dinov2_490_jpegxl_lossy_lq_raw",
    ],
    # DINOv2-random
    "dinov2_random": [
        "dinov2_random_zstd_raw", "dinov2_random_jpegxl_lossy_hq_raw",
        "dinov2_random_jpegxl_lossy_effort_3_raw", "dinov2_random_jpegxl_lossy_d2_e8_raw",
        "dinov2_random_jpegxl_lossy_mq_new_raw", "dinov2_random_jpegxl_lossy_mq_raw",
        "dinov2_random_jpegxl_lossy_lq_raw",
        "dinov2_random_jpegxl_lossy_d10_raw", "dinov2_random_jpegxl_lossy_d15_raw",
        "dinov2_random_jpegxl_lossy_d20_e2_raw", "dinov2_random_jpegxl_lossy_d30_raw",
    ],
    # DINOv2 non-random (from jump_target2_4plate_cl)
    "dinov2": [
        "dinov2_jpegxl_lossy_d2_e8_raw", "dinov2_jpegxl_lossy_mq_new_raw",
        "dinov2_jpegxl_lossy_d10_raw",
        "dinov2_jpegxl_lossy_d15_raw", "dinov2_jpegxl_lossy_d20_e2_raw",
        "dinov2_jpegxl_lossy_d30_raw",
    ],
    # MorphEm
    "morphem": [
        "morphem_zstd_raw", "morphem_jpegxl_lossy_hq_raw",
        "morphem_jpegxl_lossy_effort_3_raw", "morphem_jpegxl_lossy_d2_e8_raw",
        "morphem_jpegxl_lossy_mq_new_raw", "morphem_jpegxl_lossy_mq_raw",
        "morphem_jpegxl_lossy_lq_raw",
        "morphem_jpegxl_lossy_d10_raw", "morphem_jpegxl_lossy_d15_raw",
        "morphem_jpegxl_lossy_d20_e2_raw", "morphem_jpegxl_lossy_d30_raw",
    ],
    # SubCell
    "subcell": [
        "subcell_zstd_raw", "subcell_jpegxl_lossy_hq_raw",
        "subcell_jpegxl_lossy_effort_3_raw", "subcell_jpegxl_lossy_d2_e8_raw",
        "subcell_jpegxl_lossy_mq_new_raw", "subcell_jpegxl_lossy_mq_raw",
        "subcell_jpegxl_lossy_lq_raw",
        "subcell_jpegxl_lossy_d10_raw", "subcell_jpegxl_lossy_d15_raw",
        "subcell_jpegxl_lossy_d20_e2_raw", "subcell_jpegxl_lossy_d30_raw",
    ],
    # OpenPhenom-8bit (from output/)
    "openphenom_8bit": [
        "openphenom_8bit_zstd_raw", "openphenom_8bit_jpegxl_lossy_hq_raw",
        "openphenom_8bit_jpegxl_lossy_effort_3_raw", "openphenom_8bit_jpegxl_lossy_mq_raw",
        "openphenom_8bit_jpegxl_lossy_lq_raw",
    ],
    # OpenPhenom (from jump_target2_4plate_cl/)
    "openphenom": [
        "openphenom_zstd_raw", "openphenom_jpegxl_lossy_hq_raw",
        "openphenom_jpegxl_lossy_effort_3_raw", "openphenom_jpegxl_lossy_d2_e8_raw",
        "openphenom_jpegxl_lossy_mq_new_raw", "openphenom_jpegxl_lossy_mq_raw",
        "openphenom_jpegxl_lossy_lq_raw",
        "openphenom_jpegxl_lossy_d10_raw", "openphenom_jpegxl_lossy_d15_raw",
        "openphenom_jpegxl_lossy_d20_e2_raw", "openphenom_jpegxl_lossy_d30_raw",
    ],
    # === Rerun families (from jump_target2_4plate_cl_rerun/) ===
    "dinov2_rr": [
        "dinov2_cl_zstd_rr_raw", "dinov2_cl_jpegxl_lossy_hq_rr_raw",
        "dinov2_cl_jpegxl_lossy_effort_3_rr_raw", "dinov2_cl_jpegxl_lossy_mq_rr_raw",
        "dinov2_cl_jpegxl_lossy_mq_new_rr_raw", "dinov2_cl_jpegxl_lossy_lq_rr_raw",
        "dinov2_cl_jpegxl_lossy_d2_e8_rr_raw", "dinov2_cl_jpegxl_lossy_d10_rr_raw",
        "dinov2_cl_jpegxl_lossy_d15_rr_raw", "dinov2_cl_jpegxl_lossy_d20_e2_rr_raw",
        "dinov2_cl_jpegxl_lossy_d30_rr_raw", "dinov2_cl_jpegxl_lossy_d50_rr_raw",
    ],
    "dinov2_random_rr": [
        "dinov2_random_zstd_rr_raw", "dinov2_random_jpegxl_lossy_hq_rr_raw",
        "dinov2_random_jpegxl_lossy_effort_3_rr_raw", "dinov2_random_jpegxl_lossy_mq_rr_raw",
        "dinov2_random_jpegxl_lossy_mq_new_rr_raw", "dinov2_random_jpegxl_lossy_lq_rr_raw",
        "dinov2_random_jpegxl_lossy_d2_e8_rr_raw", "dinov2_random_jpegxl_lossy_d10_rr_raw",
        "dinov2_random_jpegxl_lossy_d15_rr_raw", "dinov2_random_jpegxl_lossy_d20_e2_rr_raw",
        "dinov2_random_jpegxl_lossy_d30_rr_raw", "dinov2_random_jpegxl_lossy_d50_rr_raw",
    ],
    "morphem_rr": [
        "morphem_zstd_rr_raw", "morphem_jpegxl_lossy_hq_rr_raw",
        "morphem_jpegxl_lossy_effort_3_rr_raw", "morphem_jpegxl_lossy_mq_rr_raw",
        "morphem_jpegxl_lossy_mq_new_rr_raw", "morphem_jpegxl_lossy_lq_rr_raw",
        "morphem_jpegxl_lossy_d2_e8_rr_raw", "morphem_jpegxl_lossy_d10_rr_raw",
        "morphem_jpegxl_lossy_d15_rr_raw", "morphem_jpegxl_lossy_d20_e2_rr_raw",
        "morphem_jpegxl_lossy_d30_rr_raw", "morphem_jpegxl_lossy_d50_rr_raw",
    ],
    "subcell_rr": [
        "subcell_zstd_rr_raw", "subcell_jpegxl_lossy_hq_rr_raw",
        "subcell_jpegxl_lossy_effort_3_rr_raw", "subcell_jpegxl_lossy_mq_rr_raw",
        "subcell_jpegxl_lossy_mq_new_rr_raw", "subcell_jpegxl_lossy_lq_rr_raw",
        "subcell_jpegxl_lossy_d2_e8_rr_raw", "subcell_jpegxl_lossy_d10_rr_raw",
        "subcell_jpegxl_lossy_d15_rr_raw", "subcell_jpegxl_lossy_d20_e2_rr_raw",
        "subcell_jpegxl_lossy_d30_rr_raw", "subcell_jpegxl_lossy_d50_rr_raw",
    ],
    "openphenom_rr": [
        "openphenom_zstd_rr_raw", "openphenom_jpegxl_lossy_hq_rr_raw",
        "openphenom_jpegxl_lossy_effort_3_rr_raw", "openphenom_jpegxl_lossy_mq_rr_raw",
        "openphenom_jpegxl_lossy_mq_new_rr_raw", "openphenom_jpegxl_lossy_lq_rr_raw",
        "openphenom_jpegxl_lossy_d2_e8_rr_raw", "openphenom_jpegxl_lossy_d10_rr_raw",
        "openphenom_jpegxl_lossy_d15_rr_raw", "openphenom_jpegxl_lossy_d20_e2_rr_raw",
        "openphenom_jpegxl_lossy_d30_rr_raw", "openphenom_jpegxl_lossy_d50_rr_raw",
    ],
    # CellProfiler filtered_border_size (from jump_target2_4plate_filtered)
    "cp_measure_fbs": [
        "zstd_filtered_border_size_raw",
        "jpegxl_lossy_hq_filtered_border_size_raw",
        "jpegxl_lossy_effort_3_filtered_border_size_raw",
        "jpegxl_lossy_d2_e8_filtered_border_size_raw",
        "jpegxl_lossy_mq_filtered_border_size_raw",
        "jpegxl_lossy_lq_filtered_border_size_raw",
        "jpegxl_lossy_d10_filtered_border_size_raw",
    ],
    # DINOv2-CL (from jump_target2_4plate_cl)
    "dinov2_cl": [
        "dinov2_cl_jpegxl_lossy_d2_e8_raw",
        "dinov2_cl_jpegxl_lossy_mq_new_raw",
        "dinov2_cl_jpegxl_lossy_d10_raw",
        "dinov2_cl_jpegxl_lossy_d15_raw",
        "dinov2_cl_jpegxl_lossy_d20_e2_raw",
        "dinov2_cl_jpegxl_lossy_d30_raw",
    ],
}

# Base hues for each family (HSV hue in [0, 1])
FAMILY_HUES = {
    "cellprofiler": 0.0,       # Red
    "cp_measure": 0.07,        # Orange-red
    "cp_measure_fbs": 0.10,    # Yellow-orange
    "cp_measure_filtered": 0.14,  # Orange
    "dinov2": 0.27,            # Yellow-green
    "dinov2_490": 0.30,        # Green
    "dinov2_cl": 0.37,         # Green-teal
    "dinov2_random": 0.45,     # Teal
    "morphem": 0.58,           # Cyan-blue
    "subcell": 0.68,           # Blue-purple
    "openphenom_8bit": 0.78,   # Purple
    "openphenom": 0.85,        # Magenta
    # Rerun families (shifted hues to distinguish from originals)
    "dinov2_rr": 0.24,         # Yellow-green (near dinov2)
    "dinov2_random_rr": 0.42,  # Teal (near dinov2_random)
    "morphem_rr": 0.55,        # Cyan (near morphem)
    "subcell_rr": 0.65,        # Blue (near subcell)
    "openphenom_rr": 0.92,     # Pink-magenta (near openphenom)
}


def _build_model_colors(models: list[str]) -> dict[str, tuple]:
    """Build a color map giving each model a unique color, grouped by family.

    Models within the same family share a base hue but vary in saturation/value
    so they are visually related yet distinguishable.
    """
    import matplotlib.colors as mcolors

    # Build reverse lookup: model -> (family, index_within_family)
    model_to_family: dict[str, tuple[str, int, int]] = {}
    for family, members in MODEL_FAMILIES.items():
        for idx, m in enumerate(members):
            model_to_family[m] = (family, idx, len(members))

    colors = {}
    for m in models:
        if m in model_to_family:
            family, idx, n = model_to_family[m]
            hue = FAMILY_HUES[family]
            # Vary saturation (0.5 to 1.0) and value (0.6 to 1.0)
            sat = 0.5 + 0.5 * (1 - idx / max(n, 1))
            val = 0.6 + 0.4 * (1 - idx / max(n, 1))
            colors[m] = mcolors.hsv_to_rgb([hue, sat, val])
        else:
            # Fallback: hash-based gray-ish color
            h = hash(m) % 360 / 360.0
            colors[m] = mcolors.hsv_to_rgb([h, 0.4, 0.7])
    return colors


def parse_model_name(folder_name: str) -> str:
    """Extract a short compression name from the model folder name.

    Examples:
      cp_measure_jump_target2_4plate_zstd_raw_features -> zstd_raw
      cp_measure_jump_target2_4plate_jpegxl_lossy_hq_filtered_raw_features -> jpegxl_lossy_hq_filtered_raw
      cp_measure_jump_target2_4plate_zstd_filtered_border_size_raw_features -> zstd_filtered_border_size_raw
      raw_jump_cp_profiles_reformatted_filtered -> raw_jump_cp_profiles_reformatted_filtered
      dinov2_490_jump_target2_4plate_zstd_raw_features -> dinov2_490_zstd_raw
      dinov2_jump_target2_4plate_jpegxl_lossy_d2_e8_raw_features -> dinov2_cl_jpegxl_lossy_d2_e8_raw
      morphem_jump_target2_4plate_jpegxl_lossy_hq_raw_features -> morphem_jpegxl_lossy_hq_raw
    """
    suffix = "_features"
    # CellProfiler prefix
    cp_prefix = "cp_measure_jump_target2_4plate_"
    if folder_name.startswith(cp_prefix):
        name = folder_name[len(cp_prefix):]
        if name.endswith(suffix):
            name = name[: -len(suffix)]
        return name

    # Embedding model prefixes: {model}_jump_target2_4plate_{codec}_raw_features
    # Map from folder prefix -> short model name used in parsed output.
    # Order matters: longer/more-specific prefixes must come before shorter ones
    # (e.g. dinov2_490_ and dinov2_random_ before bare dinov2_).
    embedding_prefixes = [
        ("dinov2_490_jump_target2_4plate_", "dinov2_490"),
        ("dinov2_random_jump_target2_4plate_", "dinov2_random"),
        ("dinov2_jump_target2_4plate_", "dinov2_cl"),  # DINOv2-CL (must be after dinov2_490_ and dinov2_random_)
        ("morphem_jump_target2_4plate_", "morphem"),
        ("subcell_jump_target2_4plate_", "subcell"),
        ("openphenom_8bit_jump_target2_4plate_", "openphenom_8bit"),  # must be before bare openphenom_
        ("openphenom_jump_target2_4plate_", "openphenom"),
    ]
    for emb_prefix, model_name in embedding_prefixes:
        if folder_name.startswith(emb_prefix):
            codec_part = folder_name[len(emb_prefix):]
            if codec_part.endswith(suffix):
                codec_part = codec_part[: -len(suffix)]
            return f"{model_name}_{codec_part}"

    return folder_name


def get_display_name(model: str) -> str:
    """Get short display name for a compression codec."""
    return COMPRESSION_DISPLAY.get(model, model)


def sort_models(models: list[str]) -> list[str]:
    """Sort models by (family order, codec order within family).

    Family order follows MODEL_FAMILIES dict insertion order.
    Codec order within each family follows _CODEC_SORT_ORDER (lossless → heavy lossy).
    """
    # Build reverse lookup: model -> (family_index, codec_rank)
    _family_index = {fam: i for i, fam in enumerate(MODEL_FAMILIES)}
    _model_to_key: dict[str, tuple[int, int]] = {}
    for family, members in MODEL_FAMILIES.items():
        fi = _family_index[family]
        for m in members:
            _model_to_key[m] = (fi, _get_codec_sort_rank(m))

    n_families = len(MODEL_FAMILIES)
    return sorted(
        models,
        key=lambda m: _model_to_key.get(m, (n_families, _get_codec_sort_rank(m))),
    )


def parse_config_name(config_name: str) -> dict:
    """Parse a norm_3 config folder name into individual settings.

    Config format: {norm}_ctrl__outlier{cutoff}__INT__prune{thresh}[__pca{n}][__batch]

    Examples:
      robustmad_ctrl__outlier100__INT__prune0.9
      std_ctrl__outlier100__INT__prune0.9__pca64__tvn_efaar_e0.5
      robustmad_ctrl__outlier100__INT__prune0.9__ZCA-cor_all_e0.5
      std_ctrl__outlier100__INT__prune0.9__pca64__cascade_tvn_k128_k32
    """
    parts = config_name.split("__")

    settings = {
        "norm_method": "unknown",
        "outlier_cutoff": None,
        "use_int": False,
        "prune_thresh": None,
        "use_pca": False,
        "pca_components": None,
        "batch_method": "none",
        "spherize_fit": None,
        "spherize_epsilon": None,
        "tvn_epsilon": None,
        "tvn_original_k": None,
        "tvn_efaar_n_components": None,
        "tvn_cascade_k1": None,
        "tvn_cascade_k2": None,
    }

    for part in parts:
        # Normalization method
        if part.startswith("robustmad"):
            settings["norm_method"] = "robustmad"
        elif part.startswith("std"):
            settings["norm_method"] = "standardize"

        # Outlier cutoff
        elif part.startswith("outlier"):
            try:
                settings["outlier_cutoff"] = int(part.replace("outlier", ""))
            except ValueError:
                pass

        # Inverse Normal Transform
        elif part == "INT":
            settings["use_int"] = True

        # Correlation pruning
        elif part.startswith("prune"):
            try:
                settings["prune_thresh"] = float(part.replace("prune", ""))
            except ValueError:
                pass

        # PCA
        elif part.startswith("pca"):
            settings["use_pca"] = True
            try:
                settings["pca_components"] = int(part.replace("pca", ""))
            except ValueError:
                pass

        # Batch correction methods
        elif part.startswith("tvn_original"):
            settings["batch_method"] = "tvn_original"
            # Extract k: tvn_original_k64 -> 64
            k_match = re.search(r"_k(\d+)$", part)
            if k_match:
                settings["tvn_original_k"] = int(k_match.group(1))
        elif part.startswith("tvn_efaar"):
            settings["batch_method"] = "tvn_efaar"
            # Extract epsilon and optional n_components:
            #   tvn_efaar_e0.5       -> epsilon=0.5, n_components=128 (default)
            #   tvn_efaar_e0.5_c256  -> epsilon=0.5, n_components=256
            efaar_match = re.search(r"_e([\d.]+)(?:_c(\d+))?$", part)
            if efaar_match:
                try:
                    settings["tvn_epsilon"] = float(efaar_match.group(1))
                except ValueError:
                    pass
                if efaar_match.group(2):
                    settings["tvn_efaar_n_components"] = int(efaar_match.group(2))
                else:
                    settings["tvn_efaar_n_components"] = 128
        elif part.startswith("cascade_tvn"):
            settings["batch_method"] = "cascade_tvn"
            # Extract k1, k2: cascade_tvn_k128_k32 -> k1=128, k2=32
            k_match = re.search(r"_k(\d+)_k(\d+)$", part)
            if k_match:
                settings["tvn_cascade_k1"] = int(k_match.group(1))
                settings["tvn_cascade_k2"] = int(k_match.group(2))
        elif part.startswith("ZCA-cor_global_"):
            settings["batch_method"] = "spherize_global"
            # Parse: ZCA-cor_global_ctrl_e0.1 or ZCA-cor_global_ctrl_e4 (=1e-4)
            remainder = part[len("ZCA-cor_global_"):]
            if "_e" in remainder:
                fit_part, eps_part = remainder.rsplit("_e", 1)
                settings["spherize_fit"] = fit_part  # "ctrl" or "all"
                try:
                    # e6 means 1e-6, e0.5 means 0.5, e100.0 means 100.0
                    eps_val = float(eps_part)
                    if "." not in eps_part and eps_val >= 2:
                        settings["spherize_epsilon"] = 10 ** (-eps_val)
                    else:
                        settings["spherize_epsilon"] = eps_val
                except ValueError:
                    pass
        elif part.startswith("ZCA-cor_"):
            settings["batch_method"] = "spherize"
            # Parse: ZCA-cor_all_e0.5 or ZCA-cor_ctrl_e6
            remainder = part[len("ZCA-cor_"):]
            if "_e" in remainder:
                fit_part, eps_part = remainder.rsplit("_e", 1)
                settings["spherize_fit"] = fit_part  # "all" or "ctrl"
                try:
                    # e6 means 1e-6, e0.5 means 0.5, e100.0 means 100.0
                    eps_val = float(eps_part)
                    if "." not in eps_part and eps_val >= 2:
                        settings["spherize_epsilon"] = 10 ** (-eps_val)
                    else:
                        settings["spherize_epsilon"] = eps_val
                except ValueError:
                    pass

    return settings


def load_metrics(json_path: Path) -> dict:
    """Load metrics from a metrics.json file and combine with path metadata."""
    with open(json_path) as f:
        data = json.load(f)

    # Path: .../sweep_dir/model_folder/config_folder/results/metrics.json
    config_name = json_path.parent.parent.name
    model_folder = json_path.parent.parent.parent.name
    model_name = parse_model_name(model_folder)

    metrics = {
        "model": model_name,
        "config": config_name,
        "PA": data.get("PA"),
        "PC": data.get("PC"),
        "PA_mean_nap": data.get("PA_mean_nap"),
        "PA_median_nap": data.get("PA_median_nap"),
        "PC_mean_nap": data.get("PC_mean_nap"),
        "PC_median_nap": data.get("PC_median_nap"),
        "n_compounds": data.get("n_compounds"),
        "n_targets_active": data.get("n_targets_active"),
        "n_targets_total": data.get("n_targets_total"),
        "n_features": data.get("n_features"),
        "tvn_ill_conditioned": data.get("tvn_ill_conditioned"),
        "tvn_condition_number": data.get("tvn_max_condition_number"),
        "PC1_variance": data.get("PC1_variance"),
        "PC2_variance": data.get("PC2_variance"),
        "PC_replicable": data.get("PC_replicable"),
        "PC_replicable_n_targets_active": data.get("PC_replicable_n_targets_active"),
        "PC_replicable_n_targets_total": data.get("PC_replicable_n_targets_total"),
        "PC_replicable_mean_nap": data.get("PC_replicable_mean_nap"),
        "PC_replicable_median_nap": data.get("PC_replicable_median_nap"),
        "PC_replicable_n_compounds": data.get("PC_replicable_n_compounds"),
    }

    # Parse config settings
    settings = parse_config_name(config_name)
    metrics.update(settings)

    return metrics


def filter_degenerate(df: pl.DataFrame) -> pl.DataFrame:
    """Filter out degenerate configs: spherize without PCA.

    Spherize (ZCA whitening) on high-dimensional features without prior PCA
    produces isotropic noise where PC1_variance ≈ 1/n_features, artificially
    inflating PA while PC stays low.
    """
    before = len(df)
    df = df.filter(
        ~((pl.col("batch_method") == "spherize") & (pl.col("use_pca") == False))
    )
    after = len(df)
    if before > after:
        print(f"  Filtered {before - after} degenerate configs (spherize + no PCA)")
    return df


def _add_best_column(pdf, best_metric="balanced"):
    """Add the selection column used to pick the best config per model.

    Args:
        pdf: pandas DataFrame (must already have PA, PC columns).
        best_metric: 'balanced' for PA%*PC%/100, 'nap_balanced' for PA_mean_nap*PC_mean_nap.

    Returns:
        (pdf, column_name) where column_name is the column to call idxmax() on.
    """
    if best_metric == "nap_balanced":
        if "PA_mean_nap" in pdf.columns and "PC_mean_nap" in pdf.columns:
            pa_max = pdf["PA_mean_nap"].max()
            pc_max = pdf["PC_mean_nap"].max()
            pa_scaled = pdf["PA_mean_nap"] / pa_max if pa_max > 0 else pdf["PA_mean_nap"]
            pc_scaled = pdf["PC_mean_nap"] / pc_max if pc_max > 0 else pdf["PC_mean_nap"]
            pdf["_best_score"] = pa_scaled * pc_scaled
        else:
            print("Warning: NAP columns not found, falling back to balanced score")
            pdf["_best_score"] = pdf["PA"] * pdf["PC"] / 100
    else:
        pdf["_best_score"] = pdf["PA"] * pdf["PC"] / 100
    return pdf, "_best_score"


def generate_all_metrics_plot(df: pl.DataFrame, output_dir: Path, model_colors: dict,
                              best_metric: str = "balanced"):
    """Generate a comprehensive grid plot with all key metrics for every model.

    Shows PA, PC, balanced score, PC1_variance, and n_features as strip plots
    with each model on the x-axis and its own color.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = df.to_pandas()
    pdf["balanced_score"] = pdf["PA"] * pdf["PC"] / 100
    pdf, best_col = _add_best_column(pdf, best_metric)

    models = sort_models(pdf["model"].unique().tolist())
    n_models = len(models)
    display_names = [get_display_name(m) for m in models]

    # Best config per model
    best_idx = {}
    for model in models:
        mdf = pdf[pdf["model"] == model]
        if len(mdf) > 0 and not mdf[best_col].isna().all():
            best_idx[model] = mdf[best_col].idxmax()

    all_metrics = [
        ("PA", "PA (%)", "PA"),
        ("PC", "PC (%)", "PC"),
        ("PC Replicable", "PC_rep (%)", "PC_replicable"),
        ("Balanced Score", "PA * PC / 100", "balanced_score"),
        ("PA Mean NAP", "NAP", "PA_mean_nap"),
        ("PA Median NAP", "NAP", "PA_median_nap"),
        ("PC Mean NAP", "NAP", "PC_mean_nap"),
        ("PC Median NAP", "NAP", "PC_median_nap"),
        ("PC Rep Mean NAP", "NAP", "PC_replicable_mean_nap"),
        ("PC Rep Median NAP", "NAP", "PC_replicable_median_nap"),
        ("n Compounds", "Count", "n_compounds"),
        ("n Targets Active", "Count", "n_targets_active"),
        ("n Targets Total", "Count", "n_targets_total"),
        ("PC Rep n Targets Active", "Count", "PC_replicable_n_targets_active"),
        ("PC Rep n Targets Total", "Count", "PC_replicable_n_targets_total"),
        ("PC Rep n Compounds", "Count", "PC_replicable_n_compounds"),
        ("n Features", "Count", "n_features"),
        ("PC1 Variance", "Variance", "PC1_variance"),
        ("PC2 Variance", "Variance", "PC2_variance"),
    ]
    available = [(t, y, c) for t, y, c in all_metrics if c in pdf.columns and not pdf[c].isna().all()]

    n_metrics = len(available)
    n_cols = min(4, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    col_width = max(8, n_models * 0.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(col_width * n_cols, 6 * n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (title, ylabel, col) in enumerate(available):
        ax = axes[i]
        for j, model in enumerate(models):
            mdf = pdf[pdf["model"] == model]
            bi = best_idx.get(model)
            other = mdf[mdf.index != bi] if bi is not None else mdf
            vals = other[col].dropna()
            if len(vals) > 0:
                x_jitter = np.random.normal(j, 0.12, len(vals))
                ax.scatter(x_jitter, vals, c=[model_colors[model]], s=30, alpha=0.5,
                           edgecolors="white", linewidths=0.2)
            if bi is not None:
                bv = pdf.loc[bi, col]
                if not np.isnan(bv):
                    ax.scatter(j, bv, c=[model_colors[model]], s=350, alpha=1.0,
                               edgecolors=[model_colors[model]], linewidths=1.5, marker="*", zorder=10)

        ax.set_xticks(range(n_models))
        ax.set_xticklabels(display_names, rotation=60, ha="right", fontsize=6)
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    metric_label = "NAP balanced" if best_metric == "nap_balanced" else "PA*PC"
    fig.suptitle(f"All Metrics (* = best by {metric_label} per model)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_all_metrics.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_all_metrics.png'}")


def generate_overview_plot(df: pl.DataFrame, output_dir: Path, model_colors: dict,
                           best_metric: str = "balanced"):
    """Generate overview plots: PA, PC, balanced score + best config table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = df.to_pandas()
    pdf["balanced_score"] = pdf["PA"] * pdf["PC"] / 100
    pdf, best_col = _add_best_column(pdf, best_metric)

    models = sort_models(pdf["model"].unique().tolist())
    n_models = len(models)
    display_names = [get_display_name(m) for m in models]

    # Best config per model
    best_idx = {}
    for model in models:
        mdf = pdf[pdf["model"] == model]
        if len(mdf) > 0 and not mdf[best_col].isna().all():
            best_idx[model] = mdf[best_col].idxmax()

    overview_metrics = [
        ("PA", "PA (%)", "PA"),
        ("PC", "PC (%)", "PC"),
        ("Balanced Score", "PA * PC / 100", "balanced_score"),
        ("PC1 Variance", "Variance", "PC1_variance"),
        ("n Features", "Count", "n_features"),
    ]
    available = [(t, y, c) for t, y, c in overview_metrics if c in pdf.columns and not pdf[c].isna().all()]

    n_metrics = len(available)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (title, ylabel, col) in enumerate(available):
        ax = axes[i]
        for j, model in enumerate(models):
            mdf = pdf[pdf["model"] == model]
            bi = best_idx.get(model)
            other = mdf[mdf.index != bi] if bi is not None else mdf
            vals = other[col].dropna()
            if len(vals) > 0:
                x_jitter = np.random.normal(j, 0.12, len(vals))
                ax.scatter(x_jitter, vals, c=[model_colors[model]], s=40, alpha=0.5,
                           edgecolors="white", linewidths=0.3)
            if bi is not None:
                bv = pdf.loc[bi, col]
                if not np.isnan(bv):
                    ax.scatter(j, bv, c=[model_colors[model]], s=350, alpha=1.0,
                               edgecolors=[model_colors[model]], linewidths=1.5, marker="*", zorder=10)

        ax.set_xticks(range(n_models))
        ax.set_xticklabels(display_names, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    metric_label = "NAP balanced" if best_metric == "nap_balanced" else "PA*PC"
    fig.suptitle(f"Overview Metrics (* = best by {metric_label} per model)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_overview.png'}")

    # Save best config table
    rows = []
    for model in models:
        bi = best_idx.get(model)
        if bi is not None:
            row = {"model": get_display_name(model)}
            for title, _, col in available:
                val = pdf.loc[bi, col]
                row[title] = f"{val:.2f}" if not np.isnan(val) else "N/A"
            row["config"] = pdf.loc[bi, "config"]
            rows.append(row)

    if rows:
        import pandas as pd
        summary = pd.DataFrame(rows)
        summary.to_csv(output_dir / "sweep_overview_best_configs.csv", index=False)
        print(f"Saved: {output_dir / 'sweep_overview_best_configs.csv'}")

        # LaTeX table
        latex_df = summary.drop(columns=["config"])
        metric_cols = [c for c in latex_df.columns if c != "model"]
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Best configuration per compression codec (by balanced score).}",
            r"\label{tab:sweep_overview}",
            r"\scriptsize",
            r"\begin{tabular}{l" + "r" * len(metric_cols) + "}",
            r"\hline\hline",
            r"\rule{0pt}{2.5ex}Compression & " + " & ".join(metric_cols) + r" \\",
            r"\hline",
        ]
        for _, row in latex_df.iterrows():
            vals = [str(row["model"])] + [str(row[c]) for c in metric_cols]
            lines.append(" & ".join(vals) + r" \\")
        lines += [r"\hline\hline", r"\end{tabular}", r"\end{table}"]
        tex_path = output_dir / "sweep_overview_best_configs.tex"
        with open(tex_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Saved: {tex_path}")


def generate_pa_vs_pc_plot(df: pl.DataFrame, output_dir: Path, model_colors: dict,
                           best_metric: str = "balanced"):
    """PA vs PC scatter plot colored by compression codec."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = df.to_pandas()
    pdf, best_col = _add_best_column(pdf, best_metric)

    models = sort_models(pdf["model"].unique().tolist())

    fig, ax = plt.subplots(figsize=(14, 12))
    model_keys_plotted = []
    for model in models:
        mdf = pdf[pdf["model"] == model]
        ax.scatter(mdf["PC"], mdf["PA"], c=[model_colors[model]], s=40, alpha=0.5,
                   edgecolors="white", linewidths=0.3,
                   label=get_display_name(model))
        model_keys_plotted.append(model)
        if len(mdf) > 0 and not mdf[best_col].isna().all():
            bi = mdf[best_col].idxmax()
            ax.scatter(pdf.loc[bi, "PC"], pdf.loc[bi, "PA"],
                       c=[model_colors[model]], s=350, edgecolors=[model_colors[model]],
                       linewidths=2, marker="*", zorder=10)

    ax.set_xlim(0, pdf["PC"].max() * 1.1)
    ax.set_ylim(0, pdf["PA"].max() * 1.1)
    _add_balanced_score_lines(ax)
    ax.set_xlabel("Phenotypic Consistency (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Phenotypic Activity (%)", fontsize=14, fontweight="bold")
    ax.set_title("PA vs PC", fontsize=16, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    _add_grouped_legend(ax, handles, labels, model_keys_plotted)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_pa_vs_pc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_pa_vs_pc.png'}")


def generate_pa_vs_pc_targets_plot(df: pl.DataFrame, output_dir: Path, model_colors: dict,
                                    best_metric: str = "balanced"):
    """PA vs n_targets_active scatter plot colored by compression codec."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = df.to_pandas()
    pdf, best_col = _add_best_column(pdf, best_metric)

    models = sort_models(pdf["model"].unique().tolist())

    fig, ax = plt.subplots(figsize=(14, 12))
    model_keys_plotted = []
    for model in models:
        mdf = pdf[pdf["model"] == model]
        ax.scatter(mdf["n_targets_active"], mdf["PA"], c=[model_colors[model]], s=40, alpha=0.5,
                   edgecolors="white", linewidths=0.3,
                   label=get_display_name(model))
        model_keys_plotted.append(model)
        if len(mdf) > 0 and not mdf[best_col].isna().all():
            bi = mdf[best_col].idxmax()
            ax.scatter(pdf.loc[bi, "n_targets_active"], pdf.loc[bi, "PA"],
                       c=[model_colors[model]], s=350, edgecolors=[model_colors[model]],
                       linewidths=2, marker="*", zorder=10)

    ax.set_xlim(0, pdf["n_targets_active"].max() * 1.1)
    ax.set_ylim(0, pdf["PA"].max() * 1.1)
    ax.set_xlabel("n Targets Active (PC)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Phenotypic Activity (%)", fontsize=14, fontweight="bold")
    ax.set_title("PA vs n Targets Active", fontsize=16, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    _add_grouped_legend(ax, handles, labels, model_keys_plotted)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_pa_vs_pc_targets.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_pa_vs_pc_targets.png'}")


def generate_batch_method_plot(df: pl.DataFrame, output_dir: Path):
    """Compare batch correction methods across all compression codecs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = df.to_pandas()
    pdf["balanced_score"] = pdf["PA"] * pdf["PC"] / 100

    # Build a readable batch label
    def batch_label(row):
        bm = row["batch_method"]
        if bm == "spherize":
            fit = row.get("spherize_fit", "?")
            eps = row.get("spherize_epsilon")
            eps_str = f"{eps:.0e}" if eps is not None and eps < 0.01 else str(eps)
            return f"Spherize({fit},{eps_str})"
        return BATCH_DISPLAY.get(bm, bm)

    pdf["batch_label"] = pdf.apply(batch_label, axis=1)

    batch_order = ["None", "TVN Original", "TVN EFAAR", "Cascade TVN",
                   "Spherize(all,0.5)", "Spherize(all,1e-06)",
                   "Spherize(ctrl,0.5)", "Spherize(ctrl,1e-06)"]
    existing = [b for b in batch_order if b in pdf["batch_label"].values]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for ax, (metric, title) in zip(axes, [("PA", "PA (%)"), ("PC", "PC (%)"), ("balanced_score", "Balanced Score")]):
        sns.boxenplot(data=pdf, x="batch_label", y=metric, order=existing,
                      palette="Set2", ax=ax, k_depth="tukey", linewidth=1)
        ax.set_xlabel("Batch Method", fontsize=12, fontweight="bold")
        ax.set_ylabel(title, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Performance by Batch Correction Method", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_batch_method_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_batch_method_comparison.png'}")


def generate_norm_pca_plot(df: pl.DataFrame, output_dir: Path):
    """Compare normalization methods and PCA on/off."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = df.to_pandas()
    pdf["balanced_score"] = pdf["PA"] * pdf["PC"] / 100
    pdf["norm_display"] = pdf["norm_method"].map({"robustmad": "RobustMAD", "standardize": "Standardize"})
    pdf["pca_display"] = pdf["use_pca"].map({True: "PCA", False: "No PCA"})

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for ax, (metric, title) in zip(axes, [("PA", "PA (%)"), ("PC", "PC (%)"), ("balanced_score", "Balanced Score")]):
        # Norm comparison
        data_long = pdf.melt(
            id_vars=["norm_display", "pca_display"],
            value_vars=[metric],
            var_name="metric_name",
            value_name="value",
        )
        data_long["group"] = data_long["norm_display"] + " / " + data_long["pca_display"]

        group_order = ["RobustMAD / No PCA", "RobustMAD / PCA",
                       "Standardize / No PCA", "Standardize / PCA"]
        existing = [g for g in group_order if g in data_long["group"].values]

        sns.boxenplot(data=data_long, x="group", y="value", order=existing,
                      palette="Set3", ax=ax, k_depth="tukey", linewidth=1)
        ax.set_xlabel("Norm / PCA", fontsize=12, fontweight="bold")
        ax.set_ylabel(title, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Performance by Normalization and PCA", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_norm_pca_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_norm_pca_comparison.png'}")


def generate_norm_batch_comparison(df: pl.DataFrame, output_dir: Path):
    """Compare normalization methods crossed with batch correction method."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = df.to_pandas()
    pdf["balanced_score"] = pdf["PA"] * pdf["PC"] / 100
    pdf["norm_display"] = pdf["norm_method"].map({"robustmad": "RobustMAD", "standardize": "Standardize"})
    pdf["batch_display"] = pdf["batch_method"].map(BATCH_DISPLAY).fillna(pdf["batch_method"])

    pdf["group"] = pdf["norm_display"] + " / " + pdf["batch_display"]

    group_order = [
        f"{norm} / {batch}"
        for norm in ["RobustMAD", "Standardize"]
        for batch in ["None", "TVN Original", "TVN EFAAR", "Cascade TVN", "Spherize"]
    ]
    existing = [g for g in group_order if g in pdf["group"].values]

    fig, axes = plt.subplots(1, 3, figsize=(28, 8))

    for ax, (metric, title) in zip(axes, [("PA", "PA (%)"), ("PC", "PC (%)"), ("balanced_score", "Balanced Score")]):
        sns.boxenplot(data=pdf, x="group", y=metric, order=existing,
                      palette="Paired", ax=ax, k_depth="tukey", linewidth=1)
        ax.set_xlabel("Norm / Batch Method", fontsize=12, fontweight="bold")
        ax.set_ylabel(title, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Performance by Normalization and Batch Correction", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_norm_batch_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_norm_batch_comparison.png'}")


# Compression level ordering for dot-size mapping (lossless → heavy lossy)
# Higher rank = more lossy = smaller dot
COMPRESSION_LEVEL = {
    "raw": 0,       # uncompressed baseline
    "zstd": 1,      # lossless
    "hq": 2,        # light lossy
    "effort_3": 3,  # moderate lossy
    "d2_e8": 4,     # distance 2 effort 8
    "mq_new": 5,    # medium quality (new settings)
    "mq": 6,        # medium quality
    "lq": 7,        # low quality (heavy lossy)
    "d10": 8,       # distance 10
    "d15": 9,       # distance 15
    "d20_e2": 10,   # distance 20 effort 2
    "d30": 11,      # distance 30
    "d50": 12,      # distance 50 (most lossy)
}

# Short codec aliases that map to canonical COMPRESSION_LEVEL keys
_CODEC_ALIASES = {
    "e3": "effort_3",
    "effort_3": "effort_3",
    "hq": "hq",
    "mq": "mq",
    "lq": "lq",
    "zstd": "zstd",
    "d10": "d10",
    "d15": "d15",
    "d20_e2": "d20_e2",
    "d2_e8": "d2_e8",
    "mq_new": "mq_new",
    "d30": "d30",
    "d50": "d50",
    "raw": "raw",
}

# Map from display name back to compression level rank
_DISPLAY_TO_LEVEL = {}
for _raw_name, _disp in COMPRESSION_DISPLAY.items():
    # Extract base codec from display name (strip model prefix like dv2_490_, morphem_, etc.)
    for _prefix in [
        "dv2_490_",
        "dv2_rand_rr_", "dv2_rand_",       # rr before non-rr
        "dv2_rr_", "dv2_cl_", "dv2_",      # rr before cl before bare dv2
        "morphem_rr_", "morphem_",
        "subcell_rr_", "subcell_",
        "ophenom_rr_", "ophenom_cl_", "ophenom_",
        "cp_fbs_",
    ]:
        if _disp.startswith(_prefix):
            _codec = _disp[len(_prefix):]
            _canonical = _CODEC_ALIASES.get(_codec, _codec)
            if _canonical in COMPRESSION_LEVEL:
                _DISPLAY_TO_LEVEL[_disp] = COMPRESSION_LEVEL[_canonical]
            break
    else:
        # CellProfiler codecs: strip _f suffix for filtered variants
        _base = _disp[:-2] if _disp.endswith("_f") else _disp
        _canonical = _CODEC_ALIASES.get(_base, _base)
        if _canonical in COMPRESSION_LEVEL:
            _DISPLAY_TO_LEVEL[_disp] = COMPRESSION_LEVEL[_canonical]


def _add_balanced_score_lines(ax, is_nap=False):
    """Add iso-balanced-score hyperbolas to a PA-vs-PC scatter plot.

    Balanced score = PA * PC / 100 (percentage axes) or PA * PC (NAP axes).
    Lines are clipped to the current axis limits.

    Args:
        ax: Matplotlib axes (x=PC, y=PA).
        is_nap: If True, axes are NAP values ([0,~0.5]) instead of percentages.
    """
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()

    # Pick score values that fall within the visible range
    if is_nap:
        candidates = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    else:
        candidates = [2, 5, 10, 15, 20, 25]

    # Only keep scores whose hyperbola intersects the visible rectangle
    score_values = []
    for s in candidates:
        if is_nap:
            # PA = s / PC  → at x=x_hi, PA_min = s / x_hi
            pa_at_xhi = s / x_hi if x_hi > 0 else float("inf")
        else:
            # PA = s * 100 / PC  → at x=x_hi, PA_min = s * 100 / x_hi
            pa_at_xhi = s * 100 / x_hi if x_hi > 0 else float("inf")
        if pa_at_xhi < y_hi:
            score_values.append(s)

    if not score_values:
        return

    pc = np.linspace(max(x_lo, 0.01), x_hi, 500)

    for score in score_values:
        if is_nap:
            pa = score / pc
        else:
            pa = score * 100 / pc

        mask = (pa >= y_lo) & (pa <= y_hi)
        if mask.sum() < 2:
            continue
        ax.plot(pc[mask], pa[mask], "--", color="gray", alpha=0.35, linewidth=0.8, zorder=1)

        # Label at the right end of the visible curve
        label_idx = np.where(mask)[0][-1]
        label_x, label_y = pc[label_idx], pa[label_idx]
        label_str = f"BS={score}" if not is_nap else f"BS={score:g}"
        ax.annotate(
            label_str,
            (label_x, label_y),
            fontsize=7, color="gray", alpha=0.7,
            xytext=(3, 3), textcoords="offset points",
        )


def _get_compression_level(display_name: str) -> int:
    """Get compression level rank for a display name. Lower = less lossy."""
    return _DISPLAY_TO_LEVEL.get(display_name, 4)  # default to middle


def _level_to_size(level: int, min_size: float = 60, max_size: float = 350) -> float:
    """Convert compression level (0=lossless, 7=heavy lossy) to dot size."""
    max_level = max(COMPRESSION_LEVEL.values())
    # Invert: lossless = big dot, heavy lossy = small dot
    return max_size - (max_size - min_size) * level / max(max_level, 1)


def _get_model_family(model: str) -> str:
    """Get the model family name for a raw model key."""
    for family, members in MODEL_FAMILIES.items():
        if model in members:
            return family
    return "unknown"


def _add_grouped_legend(ax, handles, labels, model_keys, loc="upper left",
                        fontsize=7, title_fontsize=9):
    """Add a multi-column legend where each column is one model family.

    Args:
        ax: Matplotlib axes.
        handles/labels: From ax.get_legend_handles_labels().
        model_keys: List of raw model keys in the same order as handles/labels.
    """
    from matplotlib.legend_handler import HandlerTuple

    # Group handles/labels by family, preserving family order from MODEL_FAMILIES
    family_order = list(MODEL_FAMILIES.keys())
    family_groups: dict[str, list[tuple]] = {f: [] for f in family_order}
    label_to_idx = {l: i for i, l in enumerate(labels)}

    for model_key, handle, label in zip(model_keys, handles, labels):
        fam = _get_model_family(model_key)
        if fam in family_groups:
            family_groups[fam].append((handle, label))
        else:
            family_groups.setdefault("unknown", []).append((handle, label))

    # Only keep families that have entries
    active_families = [(f, items) for f, items in family_groups.items() if items]
    if not active_families:
        return

    # Build column-aligned legend: pad shorter columns with invisible entries
    max_rows = max(len(items) + 1 for _, items in active_families)  # +1 for title row
    all_handles = []
    all_labels = []
    n_cols = len(active_families)

    for col_idx, (fam, items) in enumerate(active_families):
        # Family title as first entry (bold via invisible handle)
        title = FAMILY_DISPLAY.get(fam, fam)
        all_handles.append(ax.scatter([], [], s=0, alpha=0))  # invisible
        all_labels.append(f"$\\bf{{{title}}}$")
        # Codec entries
        for handle, label in items:
            all_handles.append(handle)
            all_labels.append(label)
        # Pad with blanks so columns align
        for _ in range(max_rows - len(items) - 1):
            all_handles.append(ax.scatter([], [], s=0, alpha=0))
            all_labels.append("")

    ax.legend(handles=all_handles, labels=all_labels,
              loc=loc, fontsize=fontsize, title_fontsize=title_fontsize,
              ncol=n_cols, labelspacing=0.4, handletextpad=0.6,
              columnspacing=1.0, framealpha=0.8)


def _build_family_colors() -> dict[str, tuple]:
    """Build one color per model family, matching the first codec in _build_model_colors."""
    import matplotlib.colors as mcolors

    # Use the same HSV formula as _build_model_colors with idx=0
    # (sat = 1.0, val = 1.0 for the first/brightest member)
    return {
        family: mcolors.hsv_to_rgb([hue, 1.0, 1.0])
        for family, hue in FAMILY_HUES.items()
    }


# Display names for model families (used in legends)
FAMILY_DISPLAY = {
    "cellprofiler": "CellProfiler",
    "cp_measure": "cp_measure",
    "cp_measure_fbs": "cp_measure_fbs",
    "cp_measure_filtered": "cp_measure_filtered",
    "dinov2_490": "DINOv2-490",
    "dinov2_cl": "DINOv2-CL",
    "dinov2_random": "DINOv2-random",
    "morphem": "MorphEm",
    "subcell": "SubCell",
    "openphenom_8bit": "OpenPhenom-8bit",
    "openphenom": "OpenPhenom",
    "dinov2_rr": "DINOv2-RR",
    "dinov2_random_rr": "DINOv2-random-RR",
    "morphem_rr": "MorphEm-RR",
    "subcell_rr": "SubCell-RR",
    "openphenom_rr": "OpenPhenom-RR",
}


def generate_pa_vs_pc_best_balanced(df: pl.DataFrame, output_dir: Path, model_colors: dict,
                                     best_metric: str = "balanced"):
    """PA% vs PC% scatter showing best config per model-codec.

    Each dot = best pipeline config for one model-codec combination.
    Color = model family (CellProfiler, MorphEm, etc.).
    Size = compression level (larger = less lossy).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = df.to_pandas()
    pdf, best_col = _add_best_column(pdf, best_metric)
    pdf["display_name"] = pdf["model"].map(lambda m: get_display_name(m))

    models = sort_models(pdf["model"].unique().tolist())

    # Get best config per model-codec
    best_rows = []
    for model in models:
        mdf = pdf[pdf["model"] == model]
        if len(mdf) > 0 and not mdf[best_col].isna().all():
            bi = mdf[best_col].idxmax()
            best_rows.append(pdf.loc[bi])
    if not best_rows:
        print("No data for PA vs PC best balanced plot.")
        return

    import pandas as pd
    best = pd.DataFrame(best_rows)
    best["comp_level"] = best["display_name"].apply(_get_compression_level)
    best["dot_size"] = best["comp_level"].apply(_level_to_size)

    fig, ax = plt.subplots(figsize=(14, 12))

    model_keys_plotted = []
    for _, row in best.iterrows():
        model = row["model"]
        color = model_colors.get(model, (0.5, 0.5, 0.5))
        ax.scatter(
            row["PC"], row["PA"],
            c=[color],
            s=row["dot_size"],
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=row["display_name"],
        )
        model_keys_plotted.append(model)
        ax.annotate(
            row["display_name"],
            (row["PC"], row["PA"]),
            fontsize=5.5, alpha=0.8,
            xytext=(4, 4), textcoords="offset points",
        )

    handles, labels = ax.get_legend_handles_labels()
    _add_grouped_legend(ax, handles, labels, model_keys_plotted)

    ax.set_xlim(0, best["PC"].max() * 1.1)
    ax.set_ylim(0, best["PA"].max() * 1.1)
    _add_balanced_score_lines(ax)
    ax.set_xlabel("Phenotypic Consistency (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Phenotypic Activity (%)", fontsize=14, fontweight="bold")
    metric_label = "NAP balanced" if best_metric == "nap_balanced" else "PA*PC"
    ax.set_title(f"PA vs PC — Best by {metric_label} per Model-Codec", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_pa_vs_pc_best_balanced.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_pa_vs_pc_best_balanced.png'}")


def generate_nap_pa_vs_pc_best_balanced(df: pl.DataFrame, output_dir: Path, model_colors: dict,
                                         best_metric: str = "balanced"):
    """Mean NAP PA vs Mean NAP PC scatter showing best config per model-codec.

    Same layout as generate_pa_vs_pc_best_balanced but using NAP metrics.
    Color = model family, size = compression level.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = df.to_pandas()
    pdf["display_name"] = pdf["model"].map(lambda m: get_display_name(m))

    if "PA_mean_nap" not in pdf.columns or "PC_mean_nap" not in pdf.columns:
        print("NAP metrics not available, skipping NAP PA vs PC plot.")
        return

    pdf, best_col = _add_best_column(pdf, best_metric)

    models = sort_models(pdf["model"].unique().tolist())

    # Get best config per model-codec
    best_rows = []
    for model in models:
        mdf = pdf[pdf["model"] == model]
        if len(mdf) > 0 and not mdf[best_col].isna().all():
            bi = mdf[best_col].idxmax()
            best_rows.append(pdf.loc[bi])
    if not best_rows:
        print("No data for NAP PA vs PC best balanced plot.")
        return

    import pandas as pd
    best = pd.DataFrame(best_rows)
    best["comp_level"] = best["display_name"].apply(_get_compression_level)
    best["dot_size"] = best["comp_level"].apply(_level_to_size)

    fig, ax = plt.subplots(figsize=(14, 12))

    model_keys_plotted = []
    for _, row in best.iterrows():
        pa_nap = row.get("PA_mean_nap")
        pc_nap = row.get("PC_mean_nap")
        if pa_nap is None or pc_nap is None or np.isnan(pa_nap) or np.isnan(pc_nap):
            continue
        model = row["model"]
        color = model_colors.get(model, (0.5, 0.5, 0.5))
        ax.scatter(
            pc_nap, pa_nap,
            c=[color],
            s=row["dot_size"],
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=row["display_name"],
        )
        model_keys_plotted.append(model)
        ax.annotate(
            row["display_name"],
            (pc_nap, pa_nap),
            fontsize=5.5, alpha=0.8,
            xytext=(4, 4), textcoords="offset points",
        )

    handles, labels = ax.get_legend_handles_labels()
    _add_grouped_legend(ax, handles, labels, model_keys_plotted)

    nap_pa_vals = best["PA_mean_nap"].dropna()
    nap_pc_vals = best["PC_mean_nap"].dropna()
    ax.set_xlim(0, nap_pc_vals.max() * 1.15 if len(nap_pc_vals) > 0 else 0.15)
    ax.set_ylim(0, nap_pa_vals.max() * 1.15 if len(nap_pa_vals) > 0 else 0.5)
    _add_balanced_score_lines(ax, is_nap=True)
    ax.set_xlabel("PC Mean NAP", fontsize=14, fontweight="bold")
    ax.set_ylabel("PA Mean NAP", fontsize=14, fontweight="bold")
    metric_label = "NAP balanced" if best_metric == "nap_balanced" else "PA*PC"
    ax.set_title(f"Mean NAP: PA vs PC — Best by {metric_label} per Model-Codec", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_nap_pa_vs_pc_best_balanced.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_nap_pa_vs_pc_best_balanced.png'}")


def generate_nap_replicable_pa_vs_pc_best_balanced(df: pl.DataFrame, output_dir: Path, model_colors: dict,
                                                     best_metric: str = "balanced"):
    """Mean NAP PA vs Mean NAP PC_replicable scatter showing best config per model-codec.

    Same layout as generate_nap_pa_vs_pc_best_balanced but using PC_replicable_mean_nap
    on the x-axis instead of PC_mean_nap.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = df.to_pandas()
    pdf["display_name"] = pdf["model"].map(lambda m: get_display_name(m))

    if "PA_mean_nap" not in pdf.columns or "PC_replicable_mean_nap" not in pdf.columns:
        print("NAP replicable metrics not available, skipping NAP PA vs PC_replicable plot.")
        return

    pdf, best_col = _add_best_column(pdf, best_metric)

    models = sort_models(pdf["model"].unique().tolist())

    # Get best config per model-codec
    best_rows = []
    for model in models:
        mdf = pdf[pdf["model"] == model]
        if len(mdf) > 0 and not mdf[best_col].isna().all():
            bi = mdf[best_col].idxmax()
            best_rows.append(pdf.loc[bi])
    if not best_rows:
        print("No data for NAP PA vs PC_replicable best balanced plot.")
        return

    import pandas as pd
    best = pd.DataFrame(best_rows)
    best["comp_level"] = best["display_name"].apply(_get_compression_level)
    best["dot_size"] = best["comp_level"].apply(_level_to_size)

    fig, ax = plt.subplots(figsize=(14, 12))

    model_keys_plotted = []
    for _, row in best.iterrows():
        pa_nap = row.get("PA_mean_nap")
        pc_rep_nap = row.get("PC_replicable_mean_nap")
        if pa_nap is None or pc_rep_nap is None or np.isnan(pa_nap) or np.isnan(pc_rep_nap):
            continue
        model = row["model"]
        color = model_colors.get(model, (0.5, 0.5, 0.5))
        ax.scatter(
            pc_rep_nap, pa_nap,
            c=[color],
            s=row["dot_size"],
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=row["display_name"],
        )
        model_keys_plotted.append(model)
        ax.annotate(
            row["display_name"],
            (pc_rep_nap, pa_nap),
            fontsize=5.5, alpha=0.8,
            xytext=(4, 4), textcoords="offset points",
        )

    handles, labels = ax.get_legend_handles_labels()
    _add_grouped_legend(ax, handles, labels, model_keys_plotted)

    nap_pa_vals = best["PA_mean_nap"].dropna()
    nap_pc_vals = best["PC_replicable_mean_nap"].dropna()
    ax.set_xlim(0, nap_pc_vals.max() * 1.15 if len(nap_pc_vals) > 0 else 0.15)
    ax.set_ylim(0, nap_pa_vals.max() * 1.15 if len(nap_pa_vals) > 0 else 0.5)
    _add_balanced_score_lines(ax, is_nap=True)
    ax.set_xlabel("PC Replicable Mean NAP", fontsize=14, fontweight="bold")
    ax.set_ylabel("PA Mean NAP", fontsize=14, fontweight="bold")
    metric_label = "NAP balanced" if best_metric == "nap_balanced" else "PA*PC"
    ax.set_title(f"Mean NAP: PA vs PC Replicable — Best by {metric_label} per Model-Codec", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_nap_pa_vs_pc_replicable_best_balanced.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_nap_pa_vs_pc_replicable_best_balanced.png'}")


def generate_nap_pa_vs_pc_targets_best_balanced(df: pl.DataFrame, output_dir: Path, model_colors: dict,
                                                  best_metric: str = "balanced"):
    """Mean NAP PA (y) vs n_targets_active (x) scatter showing best config per model-codec.

    Same layout as generate_nap_pa_vs_pc_best_balanced but with the number of
    active PC targets on the x-axis instead of PC_mean_nap.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = df.to_pandas()
    pdf["display_name"] = pdf["model"].map(lambda m: get_display_name(m))

    if "PA_mean_nap" not in pdf.columns or "n_targets_active" not in pdf.columns:
        print("PA_mean_nap or n_targets_active not available, skipping NAP PA vs targets plot.")
        return

    pdf, best_col = _add_best_column(pdf, best_metric)

    models = sort_models(pdf["model"].unique().tolist())

    # Get best config per model-codec
    best_rows = []
    for model in models:
        mdf = pdf[pdf["model"] == model]
        if len(mdf) > 0 and not mdf[best_col].isna().all():
            bi = mdf[best_col].idxmax()
            best_rows.append(pdf.loc[bi])
    if not best_rows:
        print("No data for NAP PA vs targets plot.")
        return

    import pandas as pd
    best = pd.DataFrame(best_rows)
    best["comp_level"] = best["display_name"].apply(_get_compression_level)
    best["dot_size"] = best["comp_level"].apply(_level_to_size)

    fig, ax = plt.subplots(figsize=(14, 12))

    model_keys_plotted = []
    for _, row in best.iterrows():
        pa_nap = row.get("PA_mean_nap")
        n_active = row.get("n_targets_active")
        if pa_nap is None or n_active is None or np.isnan(pa_nap) or np.isnan(n_active):
            continue
        model = row["model"]
        color = model_colors.get(model, (0.5, 0.5, 0.5))
        ax.scatter(
            n_active, pa_nap,
            c=[color],
            s=row["dot_size"],
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=row["display_name"],
        )
        model_keys_plotted.append(model)
        ax.annotate(
            row["display_name"],
            (n_active, pa_nap),
            fontsize=5.5, alpha=0.8,
            xytext=(4, 4), textcoords="offset points",
        )

    handles, labels = ax.get_legend_handles_labels()
    _add_grouped_legend(ax, handles, labels, model_keys_plotted)

    n_active_vals = best["n_targets_active"].dropna()
    nap_pa_vals = best["PA_mean_nap"].dropna()
    ax.set_xlim(0, n_active_vals.max() * 1.15 if len(n_active_vals) > 0 else 50)
    ax.set_ylim(0, nap_pa_vals.max() * 1.15 if len(nap_pa_vals) > 0 else 0.5)
    ax.set_xlabel("n Targets Active (PC)", fontsize=14, fontweight="bold")
    ax.set_ylabel("PA Mean NAP", fontsize=14, fontweight="bold")
    metric_label = "NAP balanced" if best_metric == "nap_balanced" else "PA*PC"
    ax.set_title(f"Mean NAP PA vs n Targets Active — Best by {metric_label} per Model-Codec", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_nap_pa_vs_pc_targets_best_balanced.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_nap_pa_vs_pc_targets_best_balanced.png'}")


def generate_best_per_model_plot(df: pl.DataFrame, output_dir: Path, model_colors: dict):
    """Bar chart of best PA and PC per compression codec."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = df.to_pandas()

    models = sort_models(pdf["model"].unique().tolist())
    display_names = [get_display_name(m) for m in models]
    colors = [model_colors[m] for m in models]

    best_pa = []
    best_pc = []
    for model in models:
        mdf = pdf[pdf["model"] == model]
        best_pa.append(mdf["PA"].max() if len(mdf) > 0 else 0)
        best_pc.append(mdf["PC"].max() if len(mdf) > 0 else 0)

    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(max(20, len(models) * 0.6), 8))

    axes[0].bar(x, best_pa, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display_names, rotation=60, ha="right", fontsize=8)
    axes[0].set_ylabel("Best PA (%)", fontsize=12, fontweight="bold")
    axes[0].set_title("Best Phenotypic Activity", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, best_pc, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display_names, rotation=60, ha="right", fontsize=8)
    axes[1].set_ylabel("Best PC (%)", fontsize=12, fontweight="bold")
    axes[1].set_title("Best Phenotypic Consistency", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "sweep_best_per_model.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_best_per_model.png'}")


def generate_filtered_vs_raw_plot(df: pl.DataFrame, output_dir: Path):
    """Compare filtered vs raw (unfiltered) features for each compression codec."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = df.to_pandas()
    pdf["balanced_score"] = pdf["PA"] * pdf["PC"] / 100

    # Classify as filtered or raw
    pdf["is_filtered"] = pdf["model"].str.contains("filtered")

    # Extract base codec (without raw/filtered suffix)
    def base_codec(m):
        return m.replace("_filtered_raw", "_RAW").replace("_raw", "").replace("_RAW", "_raw")

    pdf["base_codec"] = pdf["model"].apply(base_codec)

    codecs = sorted(pdf["base_codec"].unique())
    # Only keep codecs that have both filtered and raw
    paired = [c for c in codecs if
              (pdf[(pdf["base_codec"] == c) & (pdf["is_filtered"])].shape[0] > 0 and
               pdf[(pdf["base_codec"] == c) & (~pdf["is_filtered"])].shape[0] > 0)]

    if not paired:
        print("No paired filtered/raw codecs found, skipping comparison plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, (metric, title) in zip(axes, [("PA", "PA (%)"), ("PC", "PC (%)"), ("balanced_score", "Balanced Score")]):
        plot_data = pdf[pdf["base_codec"].isin(paired)].copy()
        plot_data["filter_label"] = plot_data["is_filtered"].map({True: "Filtered", False: "Raw"})

        sns.boxenplot(data=plot_data, x="base_codec", y=metric, hue="filter_label",
                      palette={"Raw": "#4daf4a", "Filtered": "#377eb8"},
                      ax=ax, k_depth="tukey", linewidth=1)
        ax.set_xlabel("Codec", fontsize=12, fontweight="bold")
        ax.set_ylabel(title, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=10)

    plt.suptitle("Filtered vs Raw Features", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_filtered_vs_raw.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_filtered_vs_raw.png'}")


def generate_degenerate_report(df_unfiltered: pl.DataFrame, output_dir: Path):
    """Flag potentially degenerate configs (spherize+noPCA, high PA + low PC)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = df_unfiltered.to_pandas()

    # Flag degenerate: spherize + no PCA, or PA > 90 and PC < 5, or PC1 variance > 0.1
    degenerate = pdf[
        ((pdf["batch_method"] == "spherize") & (pdf["use_pca"] == False))
        | ((pdf["PA"] > 90) & (pdf["PC"] < 5))
        | (pdf["PC1_variance"] > 0.1)
    ].copy()

    if len(degenerate) == 0:
        print("No degenerate configs found.")
        return

    degenerate = degenerate.sort_values("PA", ascending=False)
    cols = ["model", "config", "PA", "PC", "PC1_variance", "n_features",
            "batch_method", "use_pca", "spherize_fit", "spherize_epsilon"]
    cols = [c for c in cols if c in degenerate.columns]
    degenerate[cols].to_csv(output_dir / "degenerate_configs.csv", index=False)
    print(f"\nWARNING: {len(degenerate)} potentially degenerate configs detected!")
    print(f"Saved: {output_dir / 'degenerate_configs.csv'}")
    print(degenerate[cols].head(20).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Gather norm_3 sweep results into CSV")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("src/norm_3/data/features/variance_first_v4"),
        help="Path to sweep output directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: sweep_results.csv in sweep-dir)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for plots (default: sweep-dir/plots)",
    )
    parser.add_argument(
        "--filter-degenerate",
        action="store_true",
        help="Filter out degenerate configs (spherize + no PCA) from plots and summary",
    )
    parser.add_argument(
        "--best-metric",
        choices=["balanced", "nap_balanced"],
        default="balanced",
        help="Metric for selecting best config: 'balanced' (PA%%*PC%%/100) or 'nap_balanced' (PA_mean_nap*PC_mean_nap)",
    )
    parser.add_argument(
        "--exclude-families",
        nargs="+",
        default=[],
        metavar="FAMILY",
        help="Exclude model families from plots/summaries (e.g. cellprofiler cp_measure_filtered)",
    )
    parser.add_argument(
        "--exclude-codecs",
        nargs="+",
        default=[],
        metavar="CODEC",
        help="Exclude compression codecs from plots/summaries (e.g. d10 lq d2_e8)",
    )
    parser.add_argument(
        "--only-families",
        nargs="+",
        default=[],
        metavar="FAMILY",
        help="Only include these model families (e.g. dinov2_rr openphenom_rr)",
    )
    args = parser.parse_args()

    if not args.sweep_dir.exists():
        print(f"Error: sweep directory {args.sweep_dir} does not exist")
        return 1

    if args.output is None:
        args.output = args.sweep_dir / "sweep_results.csv"

    # Find all metrics.json files
    json_files = list(args.sweep_dir.rglob("metrics.json"))
    print(f"Found {len(json_files)} metrics.json files")

    if not json_files:
        print("No results found!")
        return 1

    # Load all metrics
    all_metrics = []
    errors = 0
    for json_path in json_files:
        try:
            metrics = load_metrics(json_path)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            errors += 1

    if errors:
        print(f"WARNING: {errors} files failed to load")

    df = pl.DataFrame(all_metrics, infer_schema_length=None)

    # Sort by model, then by PA descending
    df = df.sort(["model", "PA"], descending=[False, True], nulls_last=True)

    # Save full CSV (always unfiltered)
    df.write_csv(args.output)
    print(f"Saved {len(df)} results to {args.output}")

    # Apply degenerate filter for summaries and plots
    df_plot = filter_degenerate(df) if args.filter_degenerate else df

    # Include only specified families if requested
    if args.only_families:
        include_models = set()
        for fam in args.only_families:
            if fam in MODEL_FAMILIES:
                include_models.update(MODEL_FAMILIES[fam])
            else:
                print(f"Warning: unknown family '{fam}', available: {list(MODEL_FAMILIES.keys())}")
        if include_models:
            before = len(df_plot)
            df_plot = df_plot.filter(pl.col("model").is_in(list(include_models)))
            print(f"Only families {args.only_families}: {before} -> {len(df_plot)} rows")

    # Exclude model families if requested
    if args.exclude_families:
        exclude_models = set()
        for fam in args.exclude_families:
            if fam in MODEL_FAMILIES:
                exclude_models.update(MODEL_FAMILIES[fam])
            else:
                print(f"Warning: unknown family '{fam}', available: {list(MODEL_FAMILIES.keys())}")
        if exclude_models:
            before = len(df_plot)
            df_plot = df_plot.filter(~pl.col("model").is_in(list(exclude_models)))
            print(f"Excluded families {args.exclude_families}: {before} -> {len(df_plot)} rows")

    # Exclude codecs if requested (matches against display name codec suffix)
    if args.exclude_codecs:
        # Map codec names to canonical keys
        codec_keys = set()
        for c in args.exclude_codecs:
            canonical = _CODEC_ALIASES.get(c, c)
            if canonical in COMPRESSION_LEVEL:
                codec_keys.add(canonical)
            else:
                print(f"Warning: unknown codec '{c}', available: {list(COMPRESSION_LEVEL.keys())}")
        if codec_keys:
            # Build set of display names that match excluded codecs
            excluded_displays = set()
            for raw_name, disp in COMPRESSION_DISPLAY.items():
                level = _DISPLAY_TO_LEVEL.get(disp)
                if level is not None and any(
                    COMPRESSION_LEVEL.get(ck) == level for ck in codec_keys
                ):
                    excluded_displays.add(raw_name)
            if excluded_displays:
                before = len(df_plot)
                df_plot = df_plot.filter(~pl.col("model").is_in(list(excluded_displays)))
                print(f"Excluded codecs {args.exclude_codecs}: {before} -> {len(df_plot)} rows")

    # Print summary
    print("\n=== Summary by Compression Codec ===")
    summary = (
        df_plot.group_by("model")
        .agg(
            pl.len().alias("n_configs"),
            pl.col("PA").max().alias("best_PA"),
            pl.col("PC").max().alias("best_PC"),
            pl.col("PA").mean().alias("mean_PA"),
            pl.col("PC").mean().alias("mean_PC"),
        )
        .sort("model")
    )
    print(summary)

    print("\n=== Summary by Batch Method ===")
    batch_summary = (
        df_plot.group_by("batch_method")
        .agg(
            pl.len().alias("n_configs"),
            pl.col("PA").mean().alias("mean_PA"),
            pl.col("PC").mean().alias("mean_PC"),
            pl.col("PA").max().alias("best_PA"),
            pl.col("PC").max().alias("best_PC"),
        )
        .sort("batch_method")
    )
    print(batch_summary)

    metric_label = "NAP balanced" if args.best_metric == "nap_balanced" else "PA * PC"
    print(f"\n=== Best Config per Codec (by {metric_label}) ===")
    pdf_tmp = df_plot.to_pandas()
    pdf_tmp, best_col = _add_best_column(pdf_tmp, args.best_metric)
    for model in sort_models(pdf_tmp["model"].unique().tolist()):
        mdf = pdf_tmp[pdf_tmp["model"] == model]
        if mdf[best_col].isna().all():
            continue
        bi = mdf[best_col].idxmax()
        row = pdf_tmp.loc[bi]
        print(f"  {get_display_name(model):16s}  PA={row['PA']:.1f}%  PC={row['PC']:.1f}%  "
              f"score={row[best_col]:.3f}  config={row['config']}")

    # Generate plots
    if args.plot:
        plot_dir = args.plot_dir or args.sweep_dir / "plots"
        print(f"\nGenerating plots in {plot_dir}...")

        # Build consistent color map for all models in the (possibly filtered) data
        all_models = sort_models(df_plot.to_pandas()["model"].unique().tolist())
        model_colors = _build_model_colors(all_models)

        generate_all_metrics_plot(df_plot, plot_dir, model_colors, best_metric=args.best_metric)
        generate_overview_plot(df_plot, plot_dir, model_colors, best_metric=args.best_metric)
        generate_pa_vs_pc_plot(df_plot, plot_dir, model_colors, best_metric=args.best_metric)
        generate_pa_vs_pc_targets_plot(df_plot, plot_dir, model_colors, best_metric=args.best_metric)
        generate_pa_vs_pc_best_balanced(df_plot, plot_dir, model_colors, best_metric=args.best_metric)
        generate_nap_pa_vs_pc_best_balanced(df_plot, plot_dir, model_colors, best_metric=args.best_metric)
        generate_nap_replicable_pa_vs_pc_best_balanced(df_plot, plot_dir, model_colors, best_metric=args.best_metric)
        generate_nap_pa_vs_pc_targets_best_balanced(df_plot, plot_dir, model_colors, best_metric=args.best_metric)
        generate_batch_method_plot(df_plot, plot_dir)
        generate_norm_pca_plot(df_plot, plot_dir)
        generate_norm_batch_comparison(df_plot, plot_dir)
        generate_best_per_model_plot(df_plot, plot_dir, model_colors)
        generate_filtered_vs_raw_plot(df_plot, plot_dir)
        # Degenerate report always uses unfiltered data
        generate_degenerate_report(df, plot_dir)
        print("\nAll plots generated!")

    return 0


if __name__ == "__main__":
    exit(main())
