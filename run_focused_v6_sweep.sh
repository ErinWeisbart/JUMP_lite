#!/usr/bin/env bash

# Run from JUMP_core root directory
cd "$(dirname "$0")"

echo "==================================================="
echo "Focused Sweep v6 — CP (54 configs) + DL (12 configs)"
echo "==================================================="
echo ""
echo "Both sweeps output to variance_first_v6/ for unified summary."
echo ""

# Function to cleanup GPU memory
cleanup_gpu() {
    echo "  Cleaning up GPU memory..."
    cd src/norm_3
    pixi run python -c "import cupy as cp; cp.get_default_memory_pool().free_all_blocks(); cp.cuda.Stream.null.synchronize(); print('    GPU memory cleaned')" 2>&1 || echo "  GPU cleanup skipped"
    cd ../..
    sleep 2
    echo "  Cleanup complete"
}

# Initial GPU cleanup before starting
echo "Performing initial GPU cleanup..."
cleanup_gpu
echo ""

# ============================================================
# PART 1: CellProfiler Models — focused_cp_v6 (54 configs each)
# ============================================================

CP_FEATURE_FILES=(
    # --- CellProfiler reformatted (original JUMP CP profiles) ---
    "../../output/raw_jump_cp_profiles_reformatted_filtered.parquet"
    # --- cp_measure raw (7 codecs, from jump_target2_4plate/) ---
    "../../data/features/jump_target2_4plate/cp_measure_jump_target2_4plate_zstd_raw_features.parquet"
    "../../data/features/jump_target2_4plate/cp_measure_jump_target2_4plate_jpegxl_lossy_hq_raw_features.parquet"
    "../../data/features/jump_target2_4plate/cp_measure_jump_target2_4plate_jpegxl_lossy_effort_3_raw_features.parquet"
    "../../data/features/jump_target2_4plate/cp_measure_jump_target2_4plate_jpegxl_lossy_mq_raw_features.parquet"
    "../../data/features/jump_target2_4plate/cp_measure_jump_target2_4plate_jpegxl_lossy_lq_raw_features.parquet"
    "../../data/features/jump_target2_4plate/cp_measure_jump_target2_4plate_jpegxl_lossy_d2_e8_raw_features.parquet"
    "../../data/features/jump_target2_4plate/cp_measure_jump_target2_4plate_jpegxl_lossy_d10_raw_features.parquet"
    # --- cp_measure filtered_border_size (7 codecs, from jump_target2_4plate_filtered/) ---
    "../../data/features/jump_target2_4plate_filtered/cp_measure_jump_target2_4plate_zstd_filtered_border_size_raw_features.parquet"
    "../../data/features/jump_target2_4plate_filtered/cp_measure_jump_target2_4plate_jpegxl_lossy_hq_filtered_border_size_raw_features.parquet"
    "../../data/features/jump_target2_4plate_filtered/cp_measure_jump_target2_4plate_jpegxl_lossy_effort_3_filtered_border_size_raw_features.parquet"
    "../../data/features/jump_target2_4plate_filtered/cp_measure_jump_target2_4plate_jpegxl_lossy_mq_filtered_border_size_raw_features.parquet"
    "../../data/features/jump_target2_4plate_filtered/cp_measure_jump_target2_4plate_jpegxl_lossy_lq_filtered_border_size_raw_features.parquet"
    "../../data/features/jump_target2_4plate_filtered/cp_measure_jump_target2_4plate_jpegxl_lossy_d2_e8_filtered_border_size_raw_features.parquet"
    "../../data/features/jump_target2_4plate_filtered/cp_measure_jump_target2_4plate_jpegxl_lossy_d10_filtered_border_size_raw_features.parquet"
)

CP_COMPRESSION_NAMES=(
    "reformatted"
    "cp_raw_zstd" "cp_raw_hq" "cp_raw_effort_3" "cp_raw_mq" "cp_raw_lq" "cp_raw_d2_e8" "cp_raw_d10"
    "cp_fbs_zstd" "cp_fbs_hq" "cp_fbs_effort_3" "cp_fbs_mq" "cp_fbs_lq" "cp_fbs_d2_e8" "cp_fbs_d10"
)

echo "==================================================="
echo "PART 1: CellProfiler Models (${#CP_FEATURE_FILES[@]} datasets × 54 configs)"
echo "==================================================="
echo ""

for i in "${!CP_FEATURE_FILES[@]}"; do
    feature_file="${CP_FEATURE_FILES[$i]}"
    compression="${CP_COMPRESSION_NAMES[$i]}"

    echo "==================================================="
    echo "[CP $((i+1))/${#CP_FEATURE_FILES[@]}] Running sweep: ${compression}"
    echo "==================================================="
    echo "  Input: ${feature_file}"
    echo ""

    cd src/norm_3
    if [ ! -f "${feature_file}" ]; then
        echo "  SKIPPING: Input file not found"
        cd ../..
        continue
    fi

    pixi run python -m norm_3.pipeline --multirun +sweep=focused_cp_v6 input.path="${feature_file}" hydra/launcher=joblib hydra.launcher.n_jobs=32 || {
        echo "  Warning: Sweep encountered errors (some configs may have failed)"
    }
    cd ../..

    echo ""
    echo "Sweep complete for ${compression}"
    echo ""
    cleanup_gpu
    echo ""
done

# ============================================================
# PART 2: Deep Learning Models — focused_dl_v6 (18 configs each)
# ============================================================

DL_FEATURE_FILES=(
    # --- DINOv2 non-random (6 codecs) ---
    "../../data/features/jump_target2_4plate_cl/dinov2_jump_target2_4plate_jpegxl_lossy_d2_e8_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_jump_target2_4plate_jpegxl_lossy_d10_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_jump_target2_4plate_jpegxl_lossy_d15_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_jump_target2_4plate_jpegxl_lossy_d20_e2_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_jump_target2_4plate_jpegxl_lossy_d30_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_jump_target2_4plate_jpegxl_lossy_mq_new_raw_features.parquet"
    # --- DINOv2-random (11 codecs) ---
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_zstd_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_jpegxl_lossy_hq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_jpegxl_lossy_effort_3_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_jpegxl_lossy_mq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_jpegxl_lossy_mq_new_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_jpegxl_lossy_lq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_jpegxl_lossy_d2_e8_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_jpegxl_lossy_d10_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_jpegxl_lossy_d15_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_jpegxl_lossy_d20_e2_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_jpegxl_lossy_d30_raw_features.parquet"
    # --- SubCell (12 codecs) ---
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_zstd_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_hq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_effort_3_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_mq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_mq_new_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_lq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_d2_e8_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_d10_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_d15_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_d20_e2_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_d30_raw_features.parquet"
    # --- MorphEm (12 codecs) ---
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_zstd_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_hq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_effort_3_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_mq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_mq_new_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_lq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_d2_e8_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_d10_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_d15_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_d20_e2_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_d30_raw_features.parquet"
    # --- OpenPhenom (12 codecs) ---
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_zstd_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_hq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_effort_3_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_mq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_mq_new_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_lq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_d2_e8_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_d10_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_d15_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_d20_e2_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_d30_raw_features.parquet"
    # --- DINOv2 rerun (12 codecs) ---
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_jump_target2_4plate_zstd_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_jump_target2_4plate_jpegxl_lossy_hq_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_jump_target2_4plate_jpegxl_lossy_effort_3_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_jump_target2_4plate_jpegxl_lossy_mq_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_jump_target2_4plate_jpegxl_lossy_mq_new_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_jump_target2_4plate_jpegxl_lossy_lq_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_jump_target2_4plate_jpegxl_lossy_d2_e8_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_jump_target2_4plate_jpegxl_lossy_d10_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_jump_target2_4plate_jpegxl_lossy_d15_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_jump_target2_4plate_jpegxl_lossy_d20_e2_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_jump_target2_4plate_jpegxl_lossy_d30_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_jump_target2_4plate_jpegxl_lossy_d50_rr_raw_features.parquet"
    # --- DINOv2-random rerun (12 codecs) ---
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_random_jump_target2_4plate_zstd_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_random_jump_target2_4plate_jpegxl_lossy_hq_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_random_jump_target2_4plate_jpegxl_lossy_effort_3_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_random_jump_target2_4plate_jpegxl_lossy_mq_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_random_jump_target2_4plate_jpegxl_lossy_mq_new_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_random_jump_target2_4plate_jpegxl_lossy_lq_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_random_jump_target2_4plate_jpegxl_lossy_d2_e8_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_random_jump_target2_4plate_jpegxl_lossy_d10_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_random_jump_target2_4plate_jpegxl_lossy_d15_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_random_jump_target2_4plate_jpegxl_lossy_d20_e2_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_random_jump_target2_4plate_jpegxl_lossy_d30_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/dinov2_random_jump_target2_4plate_jpegxl_lossy_d50_rr_raw_features.parquet"
    # --- MorphEm rerun (12 codecs) ---
    "../../data/features/jump_target2_4plate_cl_rerun/morphem_jump_target2_4plate_zstd_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/morphem_jump_target2_4plate_jpegxl_lossy_hq_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/morphem_jump_target2_4plate_jpegxl_lossy_effort_3_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/morphem_jump_target2_4plate_jpegxl_lossy_mq_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/morphem_jump_target2_4plate_jpegxl_lossy_mq_new_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/morphem_jump_target2_4plate_jpegxl_lossy_lq_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/morphem_jump_target2_4plate_jpegxl_lossy_d2_e8_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/morphem_jump_target2_4plate_jpegxl_lossy_d10_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/morphem_jump_target2_4plate_jpegxl_lossy_d15_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/morphem_jump_target2_4plate_jpegxl_lossy_d20_e2_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/morphem_jump_target2_4plate_jpegxl_lossy_d30_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/morphem_jump_target2_4plate_jpegxl_lossy_d50_rr_raw_features.parquet"
    # --- SubCell rerun (12 codecs) ---
    "../../data/features/jump_target2_4plate_cl_rerun/subcell_jump_target2_4plate_zstd_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/subcell_jump_target2_4plate_jpegxl_lossy_hq_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/subcell_jump_target2_4plate_jpegxl_lossy_effort_3_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/subcell_jump_target2_4plate_jpegxl_lossy_mq_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/subcell_jump_target2_4plate_jpegxl_lossy_mq_new_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/subcell_jump_target2_4plate_jpegxl_lossy_lq_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/subcell_jump_target2_4plate_jpegxl_lossy_d2_e8_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/subcell_jump_target2_4plate_jpegxl_lossy_d10_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/subcell_jump_target2_4plate_jpegxl_lossy_d15_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/subcell_jump_target2_4plate_jpegxl_lossy_d20_e2_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/subcell_jump_target2_4plate_jpegxl_lossy_d30_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/subcell_jump_target2_4plate_jpegxl_lossy_d50_rr_raw_features.parquet"
    # --- OpenPhenom rerun (12 codecs) ---
    "../../data/features/jump_target2_4plate_cl_rerun/openphenom_jump_target2_4plate_zstd_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/openphenom_jump_target2_4plate_jpegxl_lossy_hq_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/openphenom_jump_target2_4plate_jpegxl_lossy_effort_3_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/openphenom_jump_target2_4plate_jpegxl_lossy_mq_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/openphenom_jump_target2_4plate_jpegxl_lossy_mq_new_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/openphenom_jump_target2_4plate_jpegxl_lossy_lq_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/openphenom_jump_target2_4plate_jpegxl_lossy_d2_e8_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/openphenom_jump_target2_4plate_jpegxl_lossy_d10_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/openphenom_jump_target2_4plate_jpegxl_lossy_d15_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/openphenom_jump_target2_4plate_jpegxl_lossy_d20_e2_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/openphenom_jump_target2_4plate_jpegxl_lossy_d30_rr_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl_rerun/openphenom_jump_target2_4plate_jpegxl_lossy_d50_rr_raw_features.parquet"
)

DL_COMPRESSION_NAMES=(
    "dinov2_d2_e8" "dinov2_d10" "dinov2_d15" "dinov2_d20_e2" "dinov2_d30" "dinov2_mq_new"
    "dinov2_random_zstd" "dinov2_random_hq" "dinov2_random_effort_3" "dinov2_random_mq"
    "dinov2_random_mq_new" "dinov2_random_lq" "dinov2_random_d2_e8" "dinov2_random_d10"
    "dinov2_random_d15" "dinov2_random_d20_e2" "dinov2_random_d30"
    "subcell_zstd" "subcell_hq" "subcell_effort_3" "subcell_mq" "subcell_mq_new"
    "subcell_lq" "subcell_d2_e8" "subcell_d10" "subcell_d15" "subcell_d20_e2" "subcell_d30"
    "morphem_zstd" "morphem_hq" "morphem_effort_3" "morphem_mq" "morphem_mq_new"
    "morphem_lq" "morphem_d2_e8" "morphem_d10" "morphem_d15" "morphem_d20_e2" "morphem_d30"
    "openphenom_zstd" "openphenom_hq" "openphenom_effort_3" "openphenom_mq" "openphenom_mq_new"
    "openphenom_lq" "openphenom_d2_e8" "openphenom_d10" "openphenom_d15" "openphenom_d20_e2" "openphenom_d30"
    "dinov2_rr_zstd" "dinov2_rr_hq" "dinov2_rr_effort_3" "dinov2_rr_mq" "dinov2_rr_mq_new"
    "dinov2_rr_lq" "dinov2_rr_d2_e8" "dinov2_rr_d10" "dinov2_rr_d15" "dinov2_rr_d20_e2" "dinov2_rr_d30" "dinov2_rr_d50"
    "dinov2_random_rr_zstd" "dinov2_random_rr_hq" "dinov2_random_rr_effort_3" "dinov2_random_rr_mq" "dinov2_random_rr_mq_new"
    "dinov2_random_rr_lq" "dinov2_random_rr_d2_e8" "dinov2_random_rr_d10" "dinov2_random_rr_d15" "dinov2_random_rr_d20_e2" "dinov2_random_rr_d30" "dinov2_random_rr_d50"
    "morphem_rr_zstd" "morphem_rr_hq" "morphem_rr_effort_3" "morphem_rr_mq" "morphem_rr_mq_new"
    "morphem_rr_lq" "morphem_rr_d2_e8" "morphem_rr_d10" "morphem_rr_d15" "morphem_rr_d20_e2" "morphem_rr_d30" "morphem_rr_d50"
    "subcell_rr_zstd" "subcell_rr_hq" "subcell_rr_effort_3" "subcell_rr_mq" "subcell_rr_mq_new"
    "subcell_rr_lq" "subcell_rr_d2_e8" "subcell_rr_d10" "subcell_rr_d15" "subcell_rr_d20_e2" "subcell_rr_d30" "subcell_rr_d50"
    "openphenom_rr_zstd" "openphenom_rr_hq" "openphenom_rr_effort_3" "openphenom_rr_mq" "openphenom_rr_mq_new"
    "openphenom_rr_lq" "openphenom_rr_d2_e8" "openphenom_rr_d10" "openphenom_rr_d15" "openphenom_rr_d20_e2" "openphenom_rr_d30" "openphenom_rr_d50"
)

echo "==================================================="
echo "PART 2: Deep Learning Models (${#DL_FEATURE_FILES[@]} datasets × 18 configs)"
echo "==================================================="
echo ""

for i in "${!DL_FEATURE_FILES[@]}"; do
    feature_file="${DL_FEATURE_FILES[$i]}"
    compression="${DL_COMPRESSION_NAMES[$i]}"

    echo "==================================================="
    echo "[DL $((i+1))/${#DL_FEATURE_FILES[@]}] Running sweep: ${compression}"
    echo "==================================================="
    echo "  Input: ${feature_file}"
    echo ""

    cd src/norm_3
    if [ ! -f "${feature_file}" ]; then
        echo "  SKIPPING: Input file not found"
        cd ../..
        continue
    fi

    pixi run python -m norm_3.pipeline --multirun +sweep=focused_dl_v6 input.path="${feature_file}" hydra/launcher=joblib hydra.launcher.n_jobs=32 || {
        echo "  Warning: Sweep encountered errors (some configs may have failed)"
    }
    cd ../..

    echo ""
    echo "Sweep complete for ${compression}"
    echo ""
    cleanup_gpu
    echo ""
done

echo "==================================================="
echo "All Focused v6 Sweeps Complete!"
echo "==================================================="
echo ""
echo "Total: ${#CP_FEATURE_FILES[@]} CP datasets (54 configs) + ${#DL_FEATURE_FILES[@]} DL datasets (12 configs)"
echo ""

# Count results
echo "Final counts:"
if [ -d "src/norm_3/data/features/variance_first_v6" ]; then
    cd src/norm_3/data/features/variance_first_v6
    ls -1 | xargs -I {} sh -c 'echo "  {}: $(find {} -name metrics.json 2>/dev/null | wc -l) configs"'
    cd ../../../../..
else
    echo "  Output directory not found"
fi
echo ""
echo "Run summary with:"
echo "  cd src/norm_3 && pixi run python gather_sweep_results.py --sweep-dir data/features/variance_first_v6 --plot --filter-degenerate"
