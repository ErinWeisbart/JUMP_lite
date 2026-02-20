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
