#!/usr/bin/env bash
set -euo pipefail

# Quick v9 test: CellProfiler + morphem on 2 GPUs
# Run AFTER v8 sweep finishes (or use free GPUs)
cd "$(dirname "$0")"

LOG_DIR="logs/sweep_v9"
mkdir -p "$LOG_DIR"

echo "==================================================="
echo "v9 Test Run — CellProfiler + morphem (2 GPUs)"
echo "  CP: n_jobs=4, DL: n_jobs=4"
echo "  OMP_NUM_THREADS=8"
echo "==================================================="
echo ""

# Lower thread counts to reduce contention
# CP: 4 jobs * 8 threads = 32, DL: 4 jobs * 8 threads = 32
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

CP_CONFIGS="focused_cp_v9_none focused_cp_v9_tvn_efaar focused_cp_v9_tvn_original focused_cp_v9_spherize"
DL_CONFIGS="focused_dl_v9_none focused_dl_v9_tvn_efaar focused_dl_v9_tvn_original focused_dl_v9_spherize"

echo "  GPU 0: cellprofiler  (~100 configs, n_jobs=4)"
echo "  GPU 1: morphem       (~100 configs, n_jobs=4)"
echo ""

# ============================================================
# GPU 0: CellProfiler (background)
# ============================================================
(
    cd src/norm_3
    echo "=== cellprofiler on GPU 0 === $(date)"
    for cfg in $CP_CONFIGS; do
        echo "--- ${cfg} --- $(date)"
        CUDA_VISIBLE_DEVICES=0 pixi run python -m norm_3.pipeline --multirun \
            +sweep="${cfg}" \
            input.path="../../data/features/jump_lite/cellprofiler_raw_jump_lite_raw_features.parquet" \
            hydra/launcher=joblib hydra.launcher.n_jobs=4
    done
    echo "=== cellprofiler DONE === $(date)"
) > "${LOG_DIR}/cellprofiler.log" 2>&1 &
echo "  Started cellprofiler on GPU 0 (PID $!)"

# ============================================================
# GPU 1: morphem (background)
# ============================================================
(
    cd src/norm_3
    echo "=== morphem on GPU 1 === $(date)"
    for cfg in $DL_CONFIGS; do
        echo "--- ${cfg} --- $(date)"
        CUDA_VISIBLE_DEVICES=1 pixi run python -m norm_3.pipeline --multirun \
            +sweep="${cfg}" \
            input.path="../../data/features/jump_lite/morphem_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet" \
            hydra/launcher=joblib hydra.launcher.n_jobs=4
    done
    echo "=== morphem DONE === $(date)"
) > "${LOG_DIR}/morphem.log" 2>&1 &
echo "  Started morphem on GPU 1 (PID $!)"

echo ""
echo "Monitor with: tail -f ${LOG_DIR}/{cellprofiler,morphem}.log"
echo ""

wait

echo ""
echo "==================================================="
echo "v9 Test Complete! $(date)"
echo "==================================================="
echo ""
echo "Results:"
for d in src/norm_3/data/features/variance_first_v9/*/; do
    name=$(basename "$d")
    count=$(find "$d" -name metrics.json 2>/dev/null | wc -l)
    [ "$count" -gt 0 ] && echo "  ${name}: ${count} configs"
done
