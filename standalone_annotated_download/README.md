# JUMP Annotated Compounds Download - Standalone Example

This is a minimal, standalone example for downloading JUMP Cell Painting compounds with Repurposing Hub annotations and converting to compressed formats (Zarr, JXL, JPEG2000).

## What This Downloads

**2,721 FDA-approved drugs and clinical candidates** with known:
- Drug names
- Molecular targets (e.g., "EGFR", "HDAC1")
- Mechanisms of action
- Chemical structures (SMILES)

This dataset is ideal for:
- Drug repurposing studies
- Supervised learning (use targets/MOA as labels)
- Mechanism prediction from morphology
- Structure-activity relationships

## Quick Start

### 1. Install Dependencies

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Install dependencies
pixi install
```

### 2. Preview What Will Be Downloaded

```bash
# This queries the database and shows summary statistics (no download)
pixi run preview
```

**Output:**
- Compound manifest: `./data/manifests/repurposing_hub_compounds.csv`
- Wells manifest: `./data/manifests/repurposing_hub_wells.csv`
- Summary statistics printed to console

### 3. Test with Sample

```bash
# Download 10 compounds as TIFF only (~1-5 GB, 5-10 minutes)
pixi run sample

# Or download and convert to Zarr (recommended for PyTorch)
pixi run sample-zarr

# Or download and convert to JXL (best compression for archival)
pixi run sample-jxl
```

### 4. Download All

```bash
# Download all 2,721 compounds (~100-200 GB, 1-3 hours)
pixi run download-all

# Or download all and convert to Zarr
pixi run download-all-zarr

# Or download all and convert to JXL (16-bit, best compression)
pixi run download-all-jxl
```

## Command-Line Options

```bash
# Full help
python download_annotated_samples.py --help

# Common options:
python download_annotated_samples.py --sample 20         # Download 20 compounds
python download_annotated_samples.py --target kinase     # Filter by target
python download_annotated_samples.py --format zarr_zstd  # Convert to Zarr (PyTorch)
python download_annotated_samples.py --format jxl16      # Convert to JXL (archival)
python download_annotated_samples.py --crop-size 768     # Crop to 768×768
python download_annotated_samples.py --workers 32        # More parallel downloads
python download_annotated_samples.py --output ./my_data  # Custom output directory
python download_annotated_samples.py --dry-run           # Preview only
```

### Compression Format Options

Choose the format that best suits your needs:

| Format | Compression | Speed | Quality | Best For |
|--------|-------------|-------|---------|----------|
| **zarr_zstd** | **4-5×** | **Fast (30ms)** | 8-bit | **PyTorch training** |
| **zarr_lz4** | 3-4× | Fastest (20ms) | 8-bit | PyTorch (speed priority) |
| **zarr16** | 2-3× | Fast (35ms) | 16-bit lossless | Lossless PyTorch |
| **jxl16** | **5-6×** | Medium (100ms) | 16-bit lossless | **Archival storage** |
| **jxl8** | 6-7× | Medium (80ms) | 8-bit | Visualization |
| **jpeg2000** | 10× | Slow (200ms) | Lossy | Legacy compatibility |

**Recommendations:**
- **For deep learning**: Use `zarr_zstd` (best balance of speed and size)
- **For archival**: Use `jxl16` (best compression with lossless quality)
- **For speed**: Use `zarr_lz4` (fastest loading)
- **For maximum compression**: Use `jpeg2000` (smallest files, slower)

## Examples

### Download kinase inhibitors only (TIFF)
```bash
python download_annotated_samples.py --target kinase --sample 20
```

### Download EGFR inhibitors and convert to Zarr
```bash
python download_annotated_samples.py --target EGFR --format zarr_zstd
```

### Download proteasome inhibitors and convert to JXL (best compression)
```bash
python download_annotated_samples.py --target proteasome --format jxl16
```

### Download with custom crop size
```bash
python download_annotated_samples.py \
  --sample 10 \
  --format zarr_lz4 \
  --crop-size 512
```

### Download all kinase inhibitors as 8-bit JXL
```bash
python download_annotated_samples.py \
  --target kinase \
  --format jxl8 \
  --crop-size 768
```

### Custom database path
```bash
python download_annotated_samples.py \
  --db-path /path/to/your/jump_metadata_augmented.duckdb \
  --sample 10
```

## Output Structure

```
data/
├── manifests/
│   ├── repurposing_hub_compounds.csv  # Compound info (2,721 rows)
│   └── repurposing_hub_wells.csv      # Well locations (~75,000 rows)
│
├── repurposing_hub_tiff/              # Downloaded TIFFs
│   ├── PLATE_ID_1/
│   │   ├── *.tiff                     # 5-channel TIFF files
│   │   └── load_data.csv              # Metadata
│   ├── PLATE_ID_2/
│   └── ...
│
└── repurposing_hub_zarr/              # Converted Zarr (if --convert-to-zarr used)
    ├── PLATE_ID_1.zarr/
    │   ├── images/                    # Chunked array data
    │   ├── .zarray                    # Array metadata
    │   └── .zattrs                    # Attributes
    ├── PLATE_ID_2.zarr/
    └── ...
```

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Compounds | 2,721 annotated drugs |
| Wells | ~75,000 |
| Sites | ~450,000 (6 per well) |
| Plates | ~1,024 |
| Size (TIFF 16-bit) | ~100-200 GB |
| Size (Zarr 8-bit, 768×768) | ~25-50 GB (75% smaller) |
| Size (JXL 16-bit, 768×768) | ~15-30 GB (85% smaller, best compression) |
| Download time | ~1-3 hours @ 100 MB/s |
| Conversion time | ~1-3 hours (depends on format) |

## Top Targets

1. **Cyclooxygenase inhibitors** - 57 compounds (NSAIDs)
2. **Acetylcholine receptor antagonists** - 52 compounds
3. **Adrenergic receptor antagonists** - 50 compounds (cardiovascular)
4. **Histamine receptor antagonists** - 47 compounds (allergy)
5. **EGFR inhibitors** - 33 compounds (cancer)
6. **PI3K inhibitors** - 33 compounds (cancer)
7. **HDAC inhibitors** - 32 compounds (cancer)

Run `--dry-run` to see the full list!

## Manifest Files

### Compound Manifest
`data/manifests/repurposing_hub_compounds.csv` contains:

| Column | Description | Example |
|--------|-------------|---------|
| Metadata_JCP2022 | Compound ID | "JCP2022_000088" |
| Metadata_repurposing_name | Drug name | "erlotinib" |
| Metadata_repurposing_target | Protein targets | "EGFR\|NR1I2" |
| Metadata_repurposing_moa | Mechanism | "EGFR inhibitor" |
| Metadata_SMILES | Chemical structure | "COc1cc2..." |

### Wells Manifest
`data/manifests/repurposing_hub_wells.csv` contains:

| Column | Description | Example |
|--------|-------------|---------|
| Metadata_Source | Data source | "source_13" |
| Metadata_Batch | Batch ID | "CPJUMP1" |
| Metadata_Plate | Plate ID | "CP-CC9-R1-08" |
| Metadata_Well | Well position | "B02" |
| Metadata_JCP2022 | Compound ID | "JCP2022_000088" |
| Metadata_PlateType | Plate type | "COMPOUND" |

## System Requirements

- **Python**: 3.10 or 3.11
- **Disk space**:
  - Sample (10 compounds): ~1-5 GB
  - Full (2,721 compounds): ~100-200 GB
- **RAM**: 4 GB minimum
- **Network**: AWS S3 access (anonymous, no credentials needed)

## Dependencies

This standalone example has minimal dependencies:

```toml
pandas      # Data handling
duckdb      # Metadata queries
boto3       # AWS S3 downloads
s3fs        # S3 filesystem interface
tifffile    # TIFF handling
tqdm        # Progress bars
```

All installed automatically via `pixi install`.

## Database Configuration

By default, the script uses:
```
/data/users/jfredinh/addon_final/jump_production/data/interim/jump_metadata_augmented.duckdb
```

To use a different database:

```bash
python download_annotated_samples.py \
  --db-path /path/to/your/database.duckdb
```

Or edit the script's `DEFAULT_DB_PATH` variable.

## Troubleshooting

### "Database not found"

Update the database path:
```bash
python download_annotated_samples.py \
  --db-path /your/path/to/jump_metadata_augmented.duckdb
```

### "Out of disk space"

Options:
1. Start with `--sample 10` (smaller test)
2. Use `--target` to filter (fewer compounds)
3. Clean up TIFF files after analysis

### "Download too slow"

Increase parallel workers:
```bash
python download_annotated_samples.py --workers 32
```

### "AWS S3 access denied"

The dataset uses anonymous access (no credentials needed). If you get errors:
1. Check internet connection
2. Verify firewall allows AWS S3
3. Try from a different network

## Using Zarr Files with PyTorch

After converting to Zarr, you can use the data for deep learning:

### Option 1: Direct Zarr Loading (NumPy)

```python
import zarr
import numpy as np

# Open a Zarr store
store = zarr.open('data/repurposing_hub_zarr/PLATE_ID.zarr', mode='r')

# Access the images array
images = store['images']  # Shape: (num_sites, 5, 768, 768)

# Load a single site
site_image = images[0]  # Shape: (5, 768, 768)
# Channels: [DNA, RNA, ER, AGP, Mito]
```

### Option 2: PyTorch Dataset (Full Project)

For a complete PyTorch DataLoader with augmentations, see the parent directory's full project which includes:
- `dl/zarr_dataset.py` - PyTorch Dataset for Zarr files
- `dl/dataloader.py` - DataLoader with augmentations
- `dl/transforms.py` - Cell Painting-specific augmentations

## What's Different from the Full Project?

This standalone example is simplified:

**Included:**
- ✅ Metadata queries for annotated compounds
- ✅ Parallel downloads from S3
- ✅ Zarr conversion with compression
- ✅ Resume support
- ✅ Progress tracking
- ✅ Manifest generation

**Not Included:**
- ❌ PyTorch DataLoader utilities (see full project)
- ❌ Image format conversion (JXL, JPEG2000)
- ❌ Development tools (testing, linting)
- ❌ Advanced augmentations

For the full pipeline with PyTorch DataLoader and training utilities, see the parent directory.

## Next Steps

After downloading:

1. **Explore the manifests:**
   ```bash
   head data/manifests/repurposing_hub_compounds.csv
   ```

2. **Check download:**
   ```bash
   ls data/repurposing_hub_tiff/
   du -sh data/repurposing_hub_tiff/
   ```

3. **Analyze the data:**
   - Use your own analysis pipeline
   - Or use the full project for Zarr conversion + PyTorch

4. **For PyTorch training:**
   - See the full project in the parent directory
   - Includes Zarr conversion and DataLoader utilities

## Support

- **Issues**: https://github.com/your-repo/issues
- **JUMP Dataset**: https://jump-cellpainting.broadinstitute.org/
- **Documentation**: See parent directory for full pipeline docs

## File Structure

```
standalone_annotated_download/
├── README.md                        # This file
├── pixi.toml                        # Dependencies
├── download_annotated_samples.py    # Main script
└── converter/
    ├── __init__.py
    ├── download_manager.py          # High-level download API
    ├── metadata_filter.py           # Database queries
    ├── selective_downloader.py      # S3 downloads
    └── load_data_parser.py          # Metadata parsing
```

## License

This code is part of the JUMP Cell Painting project and follows the same license as the parent repository.

---

**Quick Reference:**

```bash
# Preview
pixi run preview

# Test with 10 compounds
pixi run sample

# Download all
pixi run download-all

# Filter by target
python download_annotated_samples.py --target kinase --sample 20
```
