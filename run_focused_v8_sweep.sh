#!/usr/bin/env bash
set -euo pipefail

# Run from JUMP_core root directory
cd "$(dirname "$0")"

LOG_DIR="logs/sweep_v8"
mkdir -p "$LOG_DIR"

echo "==================================================="
echo "Focused Sweep v8 — jump_lite (5 datasets, 4 GPUs)"
echo "==================================================="
echo ""
# Limit numpy/scipy threading to avoid oversubscription
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

echo "  GPU 0: cellprofiler  (54 configs)"
echo "  GPU 1: subcell       (44 configs)"
echo "  GPU 2: morphem       (44 configs)"
echo "  GPU 3: openphenom    (44 configs) -> then dinov2 (44 configs)"
echo ""
echo "Logs: ${LOG_DIR}/"
echo ""

# DL sweep configs (run sequentially per model)
DL_CONFIGS="focused_dl_v8_none focused_dl_v8_tvn_efaar focused_dl_v8_tvn_original focused_dl_v8_spherize"

# ============================================================
# GPU 0: CellProfiler (background)
# ============================================================
(
    cd src/norm_3
    echo "=== cellprofiler on GPU 0 === $(date)"
    CUDA_VISIBLE_DEVICES=0 pixi run python -m norm_3.pipeline --multirun \
        +sweep=focused_cp_v8 \
        input.path="../../data/features/jump_lite/cellprofiler_raw_jump_lite_raw_features.parquet" \
        hydra/launcher=joblib hydra.launcher.n_jobs=4
    echo "=== cellprofiler DONE === $(date)"
) > "${LOG_DIR}/cellprofiler.log" 2>&1 &
echo "  Started cellprofiler on GPU 0 (PID $!)"

# ============================================================
# GPU 1: subcell (background)
# ============================================================
(
    cd src/norm_3
    for cfg in $DL_CONFIGS; do
        echo "--- ${cfg} --- $(date)"
        CUDA_VISIBLE_DEVICES=1 pixi run python -m norm_3.pipeline --multirun \
            +sweep="${cfg}" \
            input.path="../../data/features/jump_lite/subcell_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet" \
            hydra/launcher=joblib hydra.launcher.n_jobs=4
    done
    echo "=== subcell DONE === $(date)"
) > "${LOG_DIR}/subcell.log" 2>&1 &
echo "  Started subcell on GPU 1 (PID $!)"

# ============================================================
# GPU 2: morphem (background)
# ============================================================
(
    cd src/norm_3
    for cfg in $DL_CONFIGS; do
        echo "--- ${cfg} --- $(date)"
        CUDA_VISIBLE_DEVICES=2 pixi run python -m norm_3.pipeline --multirun \
            +sweep="${cfg}" \
            input.path="../../data/features/jump_lite/morphem_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet" \
            hydra/launcher=joblib hydra.launcher.n_jobs=4
    done
    echo "=== morphem DONE === $(date)"
) > "${LOG_DIR}/morphem.log" 2>&1 &
echo "  Started morphem on GPU 2 (PID $!)"

# ============================================================
# GPU 3: openphenom first, then dinov2 (background)
# ============================================================
(
    cd src/norm_3
    echo "=== openphenom on GPU 3 === $(date)"
    for cfg in $DL_CONFIGS; do
        echo "--- ${cfg} --- $(date)"
        CUDA_VISIBLE_DEVICES=3 pixi run python -m norm_3.pipeline --multirun \
            +sweep="${cfg}" \
            input.path="../../data/features/jump_lite/openphenom_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet" \
            hydra/launcher=joblib hydra.launcher.n_jobs=4
    done
    echo "=== openphenom DONE === $(date)"
    echo ""
    echo "=== dinov2 on GPU 3 === $(date)"
    for cfg in $DL_CONFIGS; do
        echo "--- ${cfg} --- $(date)"
        CUDA_VISIBLE_DEVICES=3 pixi run python -m norm_3.pipeline --multirun \
            +sweep="${cfg}" \
            input.path="../../data/features/jump_lite/dinov2_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet" \
            hydra/launcher=joblib hydra.launcher.n_jobs=4
    done
    echo "=== dinov2 DONE === $(date)"
) > "${LOG_DIR}/gpu3_openphenom_dinov2.log" 2>&1 &
echo "  Started openphenom+dinov2 on GPU 3 (PID $!)"

echo ""
echo "All 5 models launched. Waiting for completion..."
echo "  Monitor with: tail -f ${LOG_DIR}/*.log"
echo ""

wait

echo ""
echo "==================================================="
echo "All Focused v8 Sweeps Complete! $(date)"
echo "==================================================="

# Count results
echo ""
echo "Final counts:"
for d in src/norm_3/data/features/variance_first_v8/*/; do
    name=$(basename "$d")
    count=$(find "$d" -name metrics.json 2>/dev/null | wc -l)
    echo "  ${name}: ${count} configs"
done

echo ""
echo "Run summary with:"
echo "  cd src/norm_3 && pixi run python gather_sweep_results.py --sweep-dir data/features/variance_first_v8 --plot --filter-degenerate"
