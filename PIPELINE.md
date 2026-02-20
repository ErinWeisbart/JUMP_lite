# Pipeline Steps

## Step 1: Image Compression
- **Script:** `src/compress_tif.py`
- **Input:** Raw 16-bit TIF microscopy images (5-channel Cell Painting), grouped by site
- **Output:** Zarr arrays `(5, H, W)` per site, compressed with specified codec
- **Codecs:** zstd, lz4hc, zlib, jpegxl_lossless, jpegxl_lossy (effort/quality variants), brotli
- **Deps:** numpy, zarr, joblib, PIL, imagecodecs, numcodecs, scikit-image
- **Config:** Edit `input_dir`, `output_dir`, and uncomment desired codecs in the `compressors` dict directly in the script
- **Run:** `nix develop . uv run python src/compress_tif.py`

## Step 2: Image Quality Assessment
- **Script:** `analysis/image_quality/compare_codecs.py`
- **Input:** Compressed zarr files (lossy codecs) + zstd reference zarr
- **Output:** `quality_metrics.csv`, violin plots (PSNR, SSIM, LPIPS)
- **Deps:** see `analysis/image_quality/pyproject.toml` (torch, torchmetrics, lpips, zarr, imagecodecs)
- **Args:** `--data-dir <path>` (default: `/work/datasets/jump_target2_4plate`), `--figures-only` (skip computation, plot from existing CSV)
- **Run:** `cd analysis/image_quality && uv run python compare_codecs.py --data-dir /path/to/zarr/files`

## Auxiliary: Compression Parameter Exploration
- **Script:** `analysis/compression_exploration/explore.py`
- **Purpose:** Auxiliary exploration of JPEG XL compression parameters (distance vs effort grid). Not part of the main pipeline — used for ad-hoc investigation of compression trade-offs.
- **Input:** Raw TIF images + compressed zarr files
- **Output:** Comparison plots, histograms
- **Args:** `--hist-only` (only run histogram + peak comparison)
- **Run:** `nix develop . uv run python analysis/compression_exploration/explore.py`

## Step 3: Segmentation Comparison
- **Script:** `analysis/segmentation/compare_segmentations.py`
- **Input:** Segmentation masks from aliby_output for each codec + ground truth (zstd)
- **Output:** Per-site IoU, Dice, F1, panoptic quality CSVs; boxen/violin plots; sample visualizations
- **Deps:** numpy, scipy, medpy, polars, matplotlib, seaborn, cellpose, PIL, zarr, imagecodecs
- **Args:**
  - `--root <path>` (required) Root directory containing all methods
  - `--ground-truth <name>` (required) Ground truth method name (e.g., `zstd.zarr`)
  - `--methods <name> [<name> ...]` (required) Methods to compare
  - `--output <prefix>` Output file prefix (default: `segmentation_comparison`)
  - `--segment-step <name>` `segment_cell` or `segment_nuclei` (default: `segment_cell`)
  - `--both` Process both cell and nuclei together
  - `--workers <N>` Parallel workers (default: 8)
  - `--fast` Skip expensive metrics (hausdorff, asd) for ~2-3x speedup
  - `--save-mappings` Save instance ID mappings to parquet
  - `--filter-percentile <N>` Filter outlier wells by cell count percentile
  - `--samples <N>` Limit to N samples for quick testing
  - `--visualize-sample` / `--visualize-sample-grid` + `--well <id>` Single sample visualization
- **Run:** `nix develop . uv run python analysis/segmentation/compare_segmentations.py --root /work/datasets/aliby_output/cp_measure/jump_target2_4plate --ground-truth zstd.zarr --methods jpegxl_lossy_hq.zarr jpegxl_lossy_mq.zarr --both --fast`
- **Utility:** `analysis/segmentation/instance_matching.py` — Instance matching between reference and compressed masks (imported by compare_segmentations.py)

## Step 3b: Segmentation Plotting
- **Script:** `analysis/segmentation/plot_segmentation_iou.py`
- **Input:** CSV outputs from Step 3
- **Output:** Combined IoU/Dice violin and boxen plots
- **Run:** `nix develop . uv run python analysis/segmentation/plot_segmentation_iou.py`

- **Script:** `analysis/segmentation/plot_cell_level_iou.py`
- **Input:** Instance mapping parquets from Step 3 (`--save-mappings`)
- **Output:** Per-cell IoU distribution plots
- **Args:** `--mappings-dir <path>` (required), `--output <prefix>`, `--thresh <float>` (default: 0.5)
- **Run:** `nix develop . uv run python analysis/segmentation/plot_cell_level_iou.py --mappings-dir output/segmentation_comparison_with_mapping/instance_mappings`

## Auxiliary: Segmentation Visualization & Validation
- **Script:** `analysis/segmentation/validate_feature_mask_alignment.py`
- **Purpose:** Verify that extracted features correspond to the correct segmentation masks. Spot-check alignment.
- **Args:** `--base-path <path>`, `--codec <name>`, `--object-type cell|nuclei`, `--n-samples <N>`, `--output <path>`
- **Run:** `nix develop . uv run python analysis/segmentation/validate_feature_mask_alignment.py --base-path /work/datasets/aliby_output/cp_measure/jump_target2_4plate`

- **Script:** `analysis/segmentation/visualize_cell_compression.py`
- **Purpose:** Visualize how individual cells look across compression levels. Useful for qualitative assessment.
- **Args:** `--mappings-dir <path>` (required), `--zarr-root <path>`, `--masks-root <path>`, `--gt-method <name>`, `--n-samples <N>`, `--seed <int>`
- **Run:** `nix develop . uv run python analysis/segmentation/visualize_cell_compression.py --mappings-dir output/instance_mappings`

- **Script:** `analysis/segmentation/interactive_cell_count_viewer.py`
- **Purpose:** Interactive Panel dashboard for browsing cell count differences with images and masks.
- **Args:** `--csv <path>` (required), `--mask-root <path>` (required), `--zarr-root <path>`, `--port <N>`
- **Run:** `nix develop . uv run python analysis/segmentation/interactive_cell_count_viewer.py --csv output/large_cell_count_diff.csv --mask-root /work/datasets/aliby_output/cp_measure/jump_target2_4plate`

- **Script:** `analysis/segmentation/segmentation_dashboard.py`
- **Purpose:** Interactive Panel dashboard for exploring segmentation comparison results with instance mappings.
- **Args:** `--mappings-dir <path>`, `--zarr-root <path>`, `--masks-root <path>`, `--port <N>`
- **Run:** `nix develop . uv run python analysis/segmentation/segmentation_dashboard.py --mappings-dir output/instance_mappings`

- **Script:** `analysis/segmentation/test_viewer_simple.py`
- **Purpose:** Simple Panel viewer test for debugging the dashboard setup.
