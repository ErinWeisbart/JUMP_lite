#!/usr/bin/env bash
set -euo pipefail

# Validate justfile.new recipe by recipe
# Run each dry-run manually, inspect the output, then uncomment the next one.
# Dry-run prints the commands without executing them.

JF="justfile.new"

echo "=== Section 2: Setup & Environment ==="
# just --justfile "$JF" check-env
# just --justfile "$JF" check-data
# just --justfile "$JF" setup

echo ""
echo "=== Section 3: Metadata ==="
# just --justfile "$JF" metadata

echo ""
echo "=== Section 4: Image Compression ==="
# just --justfile "$JF" --dry-run compress-target2
# just --justfile "$JF" --dry-run compress-target2 zstd
# just --justfile "$JF" --dry-run compress-lite
# just --justfile "$JF" --dry-run compress-lite zstd
# just --justfile "$JF" --dry-run compress-target2-all
# just --justfile "$JF" --dry-run compress-lite-all

echo ""
echo "=== Section 5: Image Quality ==="
# just --justfile "$JF" --dry-run quality-metrics
# just --justfile "$JF" --dry-run quality-sharpness
# just --justfile "$JF" --dry-run quality-figures

echo ""
echo "=== Section 6: Segmentation ==="
# just --justfile "$JF" segmentation-compare
# just --justfile "$JF" segmentation-quick
# just --justfile "$JF" segmentation-iou-plot
# just --justfile "$JF" segmentation-cell-iou

echo ""
echo "=== Section 7: Feature Extraction ==="
# just --justfile "$JF" extract-dl-target2
# just --justfile "$JF" extract-cp-target2
# just --justfile "$JF" --dry-run extract-dl-lite
# just --justfile "$JF" --dry-run extract-cp-lite
# just --justfile "$JF" --dry-run extract-cell-count
# just --justfile "$JF" --dry-run extract model=morphem

echo ""
echo "=== Section 8: Feature Analysis ==="
# just --justfile "$JF" feature-correlation-cp 
# just --justfile "$JF" feature-correlation-raw 
# just --justfile "$JF" feature-codec-compare
# just --justfile "$JF" --dry-run feature-codec-compare mappings_dir=output/instance_mappings n_samples=100
# just --justfile "$JF" feature-cross-well

echo ""
echo "=== Section 9: Sweeps ==="
# just --justfile "$JF" sweep-v11
# just --justfile "$JF" --dry-run sweep-v11-lite
# just --justfile "$JF" --dry-run sweep-single input=test.parquet
# just --justfile "$JF" --dry-run sweep-model model=morphem
# just --justfile "$JF" --dry-run sweep-model model=morphem dataset=lite
# just --justfile "$JF" --dry-run sweep-monitor
# just --justfile "$JF" --dry-run sweep-monitor dataset=v11_lite
# just --justfile "$JF" --dry-run sweep-status
# just --justfile "$JF" --dry-run sweep-status dataset=v11_lite

echo ""
echo "=== Section 10: Results ==="
# just --justfile "$JF" results-v11
# just --justfile "$JF" --dry-run results-v11-lite
# just --justfile "$JF" --dry-run results sweep_dir=data/features/variance_first_v11

echo ""
echo "=== Section 11: Auxiliary ==="
# just --justfile "$JF" --dry-run compression-explore
# just --justfile "$JF" --dry-run sphering-demo
# just --justfile "$JF" --dry-run gather-figures

echo ""
echo "=== Done ==="
echo "Uncomment lines one by one to validate each recipe."
