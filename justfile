# JUMP_core analysis pipeline
# Run `just --list` to see all recipes
# Run `just <recipe>` to execute a step

# Default data paths (override with env vars)
raw_images := env("RAW_IMAGES", "/work/datasets/jump_target2_4plate/raw")
compressed_dir := env("COMPRESSED_DIR", "/work/datasets/jump_target2_4plate")
aliby_output := env("ALIBY_OUTPUT", "/work/datasets/aliby_output")
features_lite := "data/features/jump_lite"
features_target2 := "data/features/jump_target2_4plate"
norm3_dir := "src/norm_3"
sweep_v9_dir := "src/norm_3/data/features/variance_first_v9"
sweep_v6_dir := "src/norm_3/data/features/variance_first_v6"

# ─── Environment ───────────────────────────────────────────────

# Verify the development environment
check-env:
    python --version
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
    pixi --version

# ─── Step 0: Metadata ─────────────────────────────────────────

# Build the unified metadata dataset
metadata annotations_db="/work/datasets/jump_core/annotations/jump_metadata.duckdb" annotations_cc="/work/datasets/jump_core/annotations/annotations_compound_compound.parquet" annotations_cg="/work/datasets/jump_core/annotations/annotations_compound_gene.parquet" profiles="/work/datasets/jump_core_annotated/raw_jump_CP_profiles/profiles.parquet" refchemdb="/home/jfredinh/projects/JUMP_core/metadata/refchemdb_conf_jump_matched.parquet" output_dir="metadata/":
    nix develop . uv run python scripts/build_metadata_dataset.py \
        --annotations-db {{ annotations_db }} \
        --annotations-cc {{ annotations_cc }} \
        --annotations-cg {{ annotations_cg }} \
        --profiles {{ profiles }} \
        --refchemdb {{ refchemdb }} \
        --output-dir {{ output_dir }} \
        --save-intermediates

# ─── Step 1: Image Compression ────────────────────────────────

# Batch compress all codecs (edit paths in script first)
compress-batch:
    nix develop . uv run python src/compress_tif.py

# Compress with a single codec
compress codec="jpegxl_lossy_mq" jobs="16":
    nix develop . uv run python src/compress_tif_single.py \
        --input {{ raw_images }} \
        --output {{ compressed_dir }} \
        --codec {{ codec }} \
        --n-jobs {{ jobs }}

# ─── Step 2: Image Quality ────────────────────────────────────

# Compute PSNR/SSIM/LPIPS quality metrics
quality-metrics:
    cd analysis/image_quality && uv run python compare_codecs.py \
        --data-dir {{ compressed_dir }}

# Regenerate quality violin plots from existing CSV
quality-figures:
    cd analysis/image_quality && uv run python compare_codecs.py \
        --data-dir {{ compressed_dir }} \
        --figures-only

# ─── Step 3: Segmentation Comparison ──────────────────────────

# Compare segmentation masks across codecs (cell + nuclei)
segmentation-compare methods="jpegxl_lossy_hq.zarr jpegxl_lossy_mq.zarr jpegxl_lossy_lq.zarr":
    nix develop . uv run python analysis/segmentation/compare_segmentations.py \
        --root {{ aliby_output }}/cp_measure/jump_target2_4plate \
        --ground-truth zstd.zarr \
        --methods {{ methods }} \
        --both --fast --save-mappings

# Quick segmentation test (50 samples)
segmentation-quick:
    nix develop . uv run python analysis/segmentation/compare_segmentations.py \
        --root {{ aliby_output }}/cp_measure/jump_target2_4plate \
        --ground-truth zstd.zarr \
        --methods jpegxl_lossy_hq.zarr jpegxl_lossy_mq.zarr \
        --both --fast --samples 50

# Generate combined IoU vs file-size plot
segmentation-iou-plot:
    nix develop . uv run python analysis/segmentation/plot_segmentation_iou.py

# Generate per-cell IoU distribution plots
segmentation-cell-iou mappings_dir="output/segmentation_comparison_with_mapping/instance_mappings":
    nix develop . uv run python analysis/segmentation/plot_cell_level_iou.py \
        --mappings-dir {{ mappings_dir }}

# ─── Step 4: Feature Extraction ───────────────────────────────

# Extract features for a DL model
extract-features model codec="jpegxl_lossy_mq.zarr" output=features_lite:
    nix develop . uv run python src/extract_features.py \
        --input {{ aliby_output }} \
        --output {{ output }} \
        --model {{ model }} \
        --compression {{ codec }}

# Extract CellProfiler features with size filtering
extract-cp output=features_lite:
    nix develop . uv run python src/extract_features_with_size_filter.py \
        --input {{ aliby_output }} \
        --output {{ output }} \
        --model cp_measure \
        --filter-border-cells --filter-size

# Reformat raw CellProfiler profiles
reformat-cp source metadata="metadata/metadata_dataset_filtered_4reps.parquet" output="data/features/jump_lite/cellprofiler_raw_jump_lite_raw_features.parquet":
    nix develop . uv run python src/reformat_raw_cp_profiles.py \
        --source {{ source }} \
        --metadata {{ metadata }} \
        --output {{ output }}

# Feature similarity: CellProfiler correlation heatmaps
feature-correlation-cp:
    nix develop . uv run python analysis/feature_similarity/feature_correlation_cp_measure_script.py

# Feature similarity: correlation vs raw CellProfiler
feature-correlation-raw:
    nix develop . uv run python analysis/feature_similarity/correlate_vs_raw_cp.py

# Feature similarity: per-cell codec comparison
feature-codec-compare mappings_dir="output/instance_mappings":
    nix develop . uv run python analysis/feature_similarity/compare_codec_features.py \
        --mappings-dir {{ mappings_dir }} --site-level

# ─── Step 5: Normalization — jump_lite (v9) ───────────────────

# Single normalization run (test)
norm-single input:
    cd {{ norm3_dir }} && pixi run python pipeline.py \
        +preset=gpu_base_variance_first_v9 \
        input.path={{ input }}

# Single sweep config for one model
norm-sweep-one input sweep="focused_dl_v9_none" jobs="4":
    cd {{ norm3_dir }} && pixi run python pipeline.py --multirun \
        +sweep={{ sweep }} \
        input.path={{ input }} \
        hydra/launcher=joblib hydra.launcher.n_jobs={{ jobs }}

# Full v9 sweep: 5 models, 4 GPUs, ~500 configs
sweep-v9:
    bash run_focused_v9_sweep.sh

# Quick v9 test: CellProfiler + MorphEm, 2 GPUs
sweep-v9-test:
    bash run_v9_test_cp_morphem.sh

# Monitor running v9 sweep logs
sweep-v9-monitor:
    tail -f logs/sweep_v9/*.log

# ─── Step 5-target2: Normalization — target2 (v6) ─────────────

# Full target2 sweep: CP (15 datasets x 54) + DL (110 datasets x 336)
sweep-target2:
    bash run_focused_v6_sweep.sh

# Single target2 CP sweep
sweep-target2-cp input jobs="8":
    cd {{ norm3_dir }} && pixi run python -m norm_3.pipeline --multirun \
        +sweep=focused_cp_v6 \
        input.path={{ input }} \
        hydra/launcher=joblib hydra.launcher.n_jobs={{ jobs }}

# Single target2 DL sweep
sweep-target2-dl input jobs="8":
    cd {{ norm3_dir }} && pixi run python -m norm_3.pipeline --multirun \
        +sweep=focused_dl_v6 \
        input.path={{ input }} \
        hydra/launcher=joblib hydra.launcher.n_jobs={{ jobs }}

# ─── Step 6: Results Aggregation & Figures ─────────────────────

# Aggregate v9 sweep results with plots
results-v9:
    cd {{ norm3_dir }} && pixi run python gather_sweep_results.py \
        --sweep-dir data/features/variance_first_v9 \
        --plot --filter-degenerate

# Aggregate target2 (v6) sweep results with plots
results-target2:
    cd {{ norm3_dir }} && pixi run python gather_sweep_results.py \
        --sweep-dir data/features/variance_first_v6 \
        --plot --filter-degenerate

# Aggregate any sweep dir with plots
results sweep_dir:
    cd {{ norm3_dir }} && pixi run python gather_sweep_results.py \
        --sweep-dir {{ sweep_dir }} \
        --plot --filter-degenerate

# Aggregate with custom best-metric selection
results-best sweep_dir metric="nap_balanced":
    cd {{ norm3_dir }} && pixi run python gather_sweep_results.py \
        --sweep-dir {{ sweep_dir }} \
        --plot --filter-degenerate \
        --best-metric {{ metric }}

# ─── Auxiliary ─────────────────────────────────────────────────

# Compression parameter exploration (JPEG XL distance vs effort)
compression-explore:
    nix develop . uv run python analysis/compression_exploration/explore.py

# Interactive sphering demo (Marimo)
sphering-demo:
    nix develop . uv run marimo run scripts/sphering_demo.py
