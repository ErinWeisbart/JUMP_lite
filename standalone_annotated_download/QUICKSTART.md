# Quick Start - 3 Commands

## 1. Install
```bash
pixi install
```

## 2. Preview (No Download)
```bash
pixi run preview
```

Shows what will be downloaded without downloading anything.

## 3. Download Sample
```bash
# Download TIFFs only
pixi run sample

# Or download and convert to Zarr (recommended for PyTorch)
pixi run sample-zarr

# Or download and convert to JXL (best compression for archival)
pixi run sample-jxl
```

Downloads 10 compounds for testing (~2-5 GB, ~10 minutes).

---

## That's It!

Check the output:
```bash
ls data/manifests/          # CSV files with compound info
ls data/repurposing_hub_tiff/  # Downloaded TIFF files
```

## Next Steps

**Download all 2,721 compounds:**
```bash
# TIFFs only
pixi run download-all

# Or with Zarr conversion (recommended for PyTorch)
pixi run download-all-zarr

# Or with JXL conversion (best compression for archival)
pixi run download-all-jxl
```

**Filter by target (e.g., kinase inhibitors):**
```bash
# TIFFs only
python download_annotated_samples.py --target kinase --sample 20

# With Zarr conversion (PyTorch)
python download_annotated_samples.py --target kinase --sample 20 --format zarr_zstd

# With JXL conversion (archival)
python download_annotated_samples.py --target kinase --sample 20 --format jxl16
```

**See all options:**
```bash
python download_annotated_samples.py --help
```

## Why Use Compression?

**Zarr** (recommended for PyTorch):
- **75% smaller** than TIFF
- **2-3× faster** loading
- Ready for deep learning

**JXL** (recommended for archival):
- **85% smaller** than TIFF
- **Lossless 16-bit** preservation
- Best long-term storage

See [README.md](README.md) for all format options (Zarr, JXL, JPEG2000).

---

Full documentation: See [README.md](README.md)
