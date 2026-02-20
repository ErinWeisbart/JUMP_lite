# Pipeline Steps

## Step 1a: Image Compression (batch)
- **Script:** `src/compress_tif.py`
- **Input:** Raw 16-bit TIF microscopy images (5-channel Cell Painting), grouped by site
- **Output:** Zarr arrays `(5, H, W)` per site, compressed with specified codec
- **Codecs:** zstd, lz4hc, zlib, jpegxl_lossless, jpegxl_lossy (effort/quality variants), brotli
- **Deps:** numpy, zarr, joblib, PIL, imagecodecs, numcodecs, scikit-image
- **Config:** Edit `input_dir`, `output_dir`, and uncomment desired codecs in the `compressors` dict directly in the script
- **Run:** `nix develop . uv run python src/compress_tif.py`

## Step 1b: Image Compression (single codec, CLI)
- **Script:** `src/compress_tif_single.py`
- **Input:** Raw 16-bit TIF images directory
- **Output:** Zarr array per site, compressed with the specified codec
- **Args:**
  - `--input <path>` (required) Input directory containing .tif files
  - `--output <path>` (required) Output directory for zarr files
  - `--codec <name>` (required) Codec: `zstd`, `jpegxl_lossy_hq`, `jpegxl_lossy_mq`, `jpegxl_lossy_lq`, `jpegxl_lossy_effort_3`
  - `--overwrite` Overwrite existing zarr files
  - `--n-jobs <N>` Parallel workers (default: 16)
  - `--no-skip-existing` Recompress everything
- **Run:** `nix develop . uv run python src/compress_tif_single.py --input /work/datasets/raw --output /work/datasets/compressed --codec jpegxl_lossy_hq`

## Step 1c: Feature Extraction
- **Script:** `src/extract_features.py`
- **Input:** Feature profiles from aliby_output directory tree (`MODEL/DATASET/COMPRESSION/profiles/*.parquet`)
- **Output:** Well-level aggregated features parquet (`{model}_{dataset}_{compression}_raw_features.parquet`)
- **Deps:** duckdb, polars, trommel
- **Args:**
  - `--input <path>` aliby_output directory
  - `--output <path>` Output directory
  - `--model <name>` Model name filter
  - `--compression <name>` Compression name filter
  - `--dataset <name>` Dataset name filter
  - `--cache-dir <path>` DuckDB cache directory
  - `--filter-border-cells` Exclude cells touching image borders
- **Run:** `nix develop . uv run python src/extract_features.py --input /work/datasets/aliby_output --output output/ --model cp_measure --compression zstd.zarr`

- **Script:** `src/extract_features_with_size_filter.py`
- **Purpose:** Same as extract_features.py but with additional cell size filtering
- **Additional args:**
  - `--filter-size` Enable size-based filtering
  - `--min-nuclei-diameter <px>` Minimum nuclei diameter
  - `--min-cell-diameter <px>` Minimum cell diameter
- **Run:** `nix develop . uv run python src/extract_features_with_size_filter.py --input /work/datasets/aliby_output --output output/ --model cp_measure --filter-border-cells --filter-size`

## Step 1d: Reformat Raw CellProfiler Profiles
- **Script:** `src/reformat_raw_cp_profiles.py`
- **Input:** Raw CellProfiler profiles parquet + metadata parquet with wells of interest
- **Output:** Reformatted parquet with standardized `Metadata_*` columns and model/compression tags
- **Args:**
  - `--source <path>` (required) Source profiles parquet
  - `--metadata <path>` (required) Metadata parquet with wells of interest
  - `--output <path>` (required) Output parquet file
  - `--model <name>` Metadata_model value (default: `cellprofiler_raw`)
  - `--dataset <name>` Metadata_dataset value (default: `jump_core_annotated`)
  - `--compression <name>` Metadata_compression value (default: `none`)
- **Run:** `nix develop . uv run python src/reformat_raw_cp_profiles.py --source /path/to/profiles.parquet --metadata metadata/metadata_dataset_filtered_4reps.parquet --output output/reformatted.parquet`

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

## Step 4: Feature Similarity Analysis
- **Script:** `analysis/feature_similarity/feature_correlation_cp_measure_script.py`
- **Input:** Feature profiles from aliby_output (per compression), compound metadata from `input/JUMP-Target-2_compound_metadata.tsv`
- **Output:** Correlation heatmaps, violin plots, parquet correlation results
- **Config:** Paths to aliby_output workspace and cache directory are set within the script
- **Utility:** `analysis/feature_similarity/utils_cp_measure_name_mapping.py` — Maps between CP measure naming conventions (imported by main script)
- **Run:** `nix develop . uv run python analysis/feature_similarity/feature_correlation_cp_measure_script.py`

- **Script:** `analysis/feature_similarity/correlate_vs_raw_cp.py`
- **Input:** Raw CellProfiler profiles + normalized features from the pipeline (filtered and non-filtered)
- **Output:** Spearman/Pearson correlation scatter plots, violin plots, bar charts per feature category
- **Config:** Hardcoded paths to raw CP profiles, filtered/non-filtered feature directories, and output directory — edit `RAW_CP_PATH`, `FILTERED_DIR`, `NONFILTERED_DIR`, `OUTPUT_DIR` in the script
- **Note:** Imports `scripts/map_cellprofiler_features.py:FeatureMapper` via sys.path manipulation
- **Run:** `nix develop . uv run python analysis/feature_similarity/correlate_vs_raw_cp.py`

- **Script:** `analysis/feature_similarity/compare_codec_features.py`
- **Input:** Instance mapping parquets from Step 3 (`--save-mappings`) + per-cell feature profiles from aliby_output
- **Output:** Per-cell and per-site feature correlation plots/CSVs across codecs, feature ranking
- **Args:**
  - `--mappings-dir <path>` (required) Directory with instance mapping parquet files
  - `--features-base <path>` Base path for feature profiles (default: `/work/datasets/aliby_output/cp_measure/jump_target2_4plate`)
  - `--gt-codec <name>` Ground truth codec (default: `zstd.zarr`)
  - `--codecs <name> [...]` Codecs to compare (default: jpegxl variants)
  - `--object-type cell|nuclei` (default: `cell`)
  - `--site-level` Also run site-level analysis (median of matched cells per site)
  - `--n-samples <N>` Number of random source_ids to sample (default: 5)
  - `--features <name> [...]` / `--feature-pattern <regex>` Filter specific features
  - `--list-features` List available features and exit
  - `--min-cells <N>` Minimum GT cell count per site (default: 5)
  - `--filter-percentile <N>` Filter outlier sites by cell count percentile
- **Run:** `nix develop . uv run python analysis/feature_similarity/compare_codec_features.py --mappings-dir output/instance_mappings --site-level`

### Input data
- `analysis/feature_similarity/input/JUMP-Target-2_compound_metadata.tsv` — Compound metadata
- `analysis/feature_similarity/input/JUMP-Target-2_compound_platemap.tsv` — Plate map metadata
