#!/usr/bin/env bash

ZARR_DIRS=(
	"/work/datasets/jump_lite/images/compressed/jump_target2_4plate/jpegxl_lossy_mq_new.zarr"
	"/work/datasets/jump_lite/images/compressed/jump_target2_4plate/jpegxl_lossy_d10.zarr"
	"/work/datasets/jump_lite/images/compressed/jump_target2_4plate/jpegxl_lossy_effort_3.zarr"
	"/work/datasets/jump_lite/images/compressed/jump_target2_4plate/jpegxl_lossy_d15.zarr"
	"/work/datasets/jump_lite/images/compressed/jump_target2_4plate/zstd.zarr"
	"/work/datasets/jump_lite/images/compressed/jump_target2_4plate/jpegxl_lossy_mq.zarr"
	"/work/datasets/jump_lite/images/compressed/jump_target2_4plate/jpegxl_lossy_d2_e8.zarr"
	"/work/datasets/jump_lite/images/compressed/jump_target2_4plate/jpegxl_lossy_d30.zarr"
	"/work/datasets/jump_lite/images/compressed/jump_target2_4plate/jpegxl_lossy_lq.zarr"
	"/work/datasets/jump_lite/images/compressed/jump_target2_4plate/jpegxl_lossy_d50.zarr"
	"/work/datasets/jump_lite/images/compressed/jump_target2_4plate/jpegxl_lossy_d20_e2.zarr"
	"/work/datasets/jump_lite/images/compressed/jump_target2_4plate/jpegxl_lossy_hq.zarr"
)

echo "\begin{table}[h]"
echo "\centering"
echo "\begin{tabular}{lrr}"
echo "\toprule"
echo "Directory & Size & Decompression Time (s) \\"
echo "\midrule"

for ZARR_DIR in "${ZARR_DIRS[@]}"; do
	RESULT=$(python benchmark_zarr.py "$ZARR_DIR")
	DIR_NAME=$(echo "$RESULT" | cut -d',' -f1 | sed 's/\\.zarr//' | sed 's/_/\\_/g')
	SIZE=$(echo "$RESULT" | cut -d',' -f2)
	TIME=$(echo "$RESULT" | cut -d',' -f3)
	echo "$DIR_NAME & $SIZE & $TIME \\"
done

echo "\bottomrule"
echo "\end{tabular}"
echo "\caption{Zarr Decompression Benchmark}"
echo "\label{tab:zarr_benchmark}"
echo "\end{table}"
