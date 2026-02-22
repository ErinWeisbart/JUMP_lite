#!/usr/bin/env bash

# Run from JUMP_core root directory
cd "$(dirname "$0")"

echo "==================================================="
echo "Variance-First Pipeline v5 - Raw CP + Filtered FBS + CL Embeddings"
echo "==================================================="
echo ""
echo "Pipeline order:"
echo "  1. NaN removal"
echo "  2. Low variance filtering (variance_threshold)"
echo "  3. Normalization (RobustMAD or Standardize)"
echo "  4. Outlier removal (cutoff=100)"
echo "  5. Inverse Normal Transform"
echo "  6. Correlation pruning (threshold=0.9)"
echo "  7. PCA (optional - min of 64 components or 95% variance)"
echo "  8. Batch correction (None, TVN_Original k64, TVN_EFAAR, Cascade_TVN k128/k32, or Spherize)"
echo ""
echo "Datasets:"
echo "  - CellProfiler reformatted (1, from output/)"
echo "  - cp_measure raw (7 codecs, from jump_target2_4plate/)"
echo "  - cp_measure filtered_border_size (7 codecs, from jump_target2_4plate_filtered/)"
echo "  - DINOv2-random (8 codecs, from jump_target2_4plate_cl/)"
echo "  - SubCell (9 codecs, from jump_target2_4plate_cl/)"
echo "  - MorphEm (9 codecs, from jump_target2_4plate_cl/)"
echo "  - OpenPhenom (9 codecs, from jump_target2_4plate_cl/)"
echo "  - Total: 52 datasets × 20 configs = 1040 runs"
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

FEATURE_FILES=(
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
    # --- DINOv2-random (7 codecs, from jump_target2_4plate_cl/) ---
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_zstd_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_jpegxl_lossy_hq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_jpegxl_lossy_effort_3_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_jpegxl_lossy_mq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_jpegxl_lossy_lq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_jpegxl_lossy_d2_e8_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_jpegxl_lossy_d10_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/dinov2_random_jump_target2_4plate_jpegxl_lossy_d15_raw_features.parquet"
    # --- SubCell (9 codecs, from jump_target2_4plate_cl/) ---
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_zstd_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_hq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_effort_3_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_mq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_lq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_d2_e8_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_d10_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_d15_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/subcell_jump_target2_4plate_jpegxl_lossy_d20_e2_raw_features.parquet"
    # --- MorphEm (9 codecs, from jump_target2_4plate_cl/) ---
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_zstd_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_hq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_effort_3_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_mq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_lq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_d2_e8_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_d10_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_d15_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/morphem_jump_target2_4plate_jpegxl_lossy_d20_e2_raw_features.parquet"
    # --- OpenPhenom (9 codecs, from jump_target2_4plate_cl/) ---
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_zstd_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_hq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_effort_3_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_mq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_lq_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_d2_e8_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_d10_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_d15_raw_features.parquet"
    "../../data/features/jump_target2_4plate_cl/openphenom_jump_target2_4plate_jpegxl_lossy_d20_e2_raw_features.parquet"
)

COMPRESSION_NAMES=(
    # CellProfiler reformatted
    "reformatted"
    # cp_measure raw
    "cp_raw_zstd"
    "cp_raw_hq"
    "cp_raw_effort_3"
    "cp_raw_mq"
    "cp_raw_lq"
    "cp_raw_d2_e8"
    "cp_raw_d10"
    # cp_measure filtered_border_size
    "cp_fbs_zstd"
    "cp_fbs_hq"
    "cp_fbs_effort_3"
    "cp_fbs_mq"
    "cp_fbs_lq"
    "cp_fbs_d2_e8"
    "cp_fbs_d10"
    # DINOv2-random
    "dinov2_random_zstd"
    "dinov2_random_hq"
    "dinov2_random_effort_3"
    "dinov2_random_mq"
    "dinov2_random_lq"
    "dinov2_random_d2_e8"
    "dinov2_random_d10"
    "dinov2_random_d15"
    # SubCell
    "subcell_zstd"
    "subcell_hq"
    "subcell_effort_3"
    "subcell_mq"
    "subcell_lq"
    "subcell_d2_e8"
    "subcell_d10"
    "subcell_d15"
    "subcell_d20_e2"
    # MorphEm
    "morphem_zstd"
    "morphem_hq"
    "morphem_effort_3"
    "morphem_mq"
    "morphem_lq"
    "morphem_d2_e8"
    "morphem_d10"
    "morphem_d15"
    "morphem_d20_e2"
    # OpenPhenom
    "openphenom_zstd"
    "openphenom_hq"
    "openphenom_effort_3"
    "openphenom_mq"
    "openphenom_lq"
    "openphenom_d2_e8"
    "openphenom_d10"
    "openphenom_d15"
    "openphenom_d20_e2"
)

TOTAL_DATASETS=${#FEATURE_FILES[@]}

# Run sweep for each dataset
for i in "${!FEATURE_FILES[@]}"; do
    feature_file="${FEATURE_FILES[$i]}"
    compression="${COMPRESSION_NAMES[$i]}"

    echo "==================================================="
    echo "[$((i+1))/${TOTAL_DATASETS}] Running sweep: ${compression}"
    echo "==================================================="
    echo "  Input: ${feature_file}"
    echo ""

    # Check if input file exists
    cd src/norm_3
    if [ ! -f "${feature_file}" ]; then
        echo "  SKIPPING: Input file not found (not yet available?)"
        cd ../..
        continue
    fi

    # Run sweep (continue on error to allow cleanup and next compression)
    pixi run python -m norm_3.pipeline --multirun \
        +sweep=simple_cellprofiler_variance_first_v5 \
        input.path="${feature_file}" \
        hydra/launcher=joblib \
        hydra.launcher.n_jobs=8 || {
        echo "  Warning: Sweep encountered errors (some configs may have failed)"
    }
    cd ../..

    echo ""
    echo "Sweep complete for ${compression}"
    echo ""

    # Cleanup GPU memory between datasets
    cleanup_gpu
    echo ""
done

echo "==================================================="
echo "All Variance-First v5 Sweeps Complete!"
echo "==================================================="
echo ""

# Count results
echo "Final counts:"
if [ -d "src/norm_3/data/features/variance_first_v5" ]; then
    cd src/norm_3/data/features/variance_first_v5
    ls -1 | xargs -I {} sh -c 'echo "  {}: $(find {} -name metrics.json 2>/dev/null | wc -l) configs"'
    cd ../../../../..
else
    echo "  Output directory not found"
fi
echo ""
