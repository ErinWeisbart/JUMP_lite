# JUMP Annotated Compounds Download - Standalone Example

This is a minimal, standalone example for downloading JUMP Cell Painting compounds with Repurposing Hub annotations.

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
# Download 10 compounds for testing (~1-5 GB, 5-10 minutes)
pixi run sample
```

### 4. Download All

```bash
# Download all 2,721 compounds (~100-200 GB, 1-3 hours)
pixi run download-all
```

## Command-Line Options

```bash
# Full help
python download_annotated_samples.py --help

# Common options:
python download_annotated_samples.py --sample 20         # Download 20 compounds
python download_annotated_samples.py --target kinase     # Filter by target
python download_annotated_samples.py --workers 32        # More parallel downloads
python download_annotated_samples.py --output ./my_data  # Custom output directory
python download_annotated_samples.py --dry-run           # Preview only
```

## Examples

### Download kinase inhibitors only
```bash
python download_annotated_samples.py --target kinase --sample 20
```

### Download EGFR inhibitors
```bash
python download_annotated_samples.py --target EGFR
```

### Download proteasome inhibitors
```bash
python download_annotated_samples.py --target proteasome
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
└── repurposing_hub_tiff/
    ├── PLATE_ID_1/
    │   ├── *.tiff                     # 5-channel TIFF files
    │   └── load_data.csv              # Metadata
    ├── PLATE_ID_2/
    └── ...
```

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Compounds | 2,721 annotated drugs |
| Wells | ~75,000 |
| Sites | ~450,000 (6 per well) |
| Plates | ~1,024 |
| Size (TIFF) | ~100-200 GB |
| Download time | ~1-3 hours @ 100 MB/s |

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

## What's Different from the Full Project?

This standalone example is simplified:

**Included:**
- ✅ Metadata queries for annotated compounds
- ✅ Parallel downloads from S3
- ✅ Resume support
- ✅ Progress tracking
- ✅ Manifest generation

**Not Included:**
- ❌ Zarr conversion (use full project for this)
- ❌ PyTorch DataLoader (use full project for this)
- ❌ Image format conversion (JXL, JPEG2000)
- ❌ Development tools (testing, linting)

For the full pipeline including Zarr conversion and PyTorch integration, see the parent directory.

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
