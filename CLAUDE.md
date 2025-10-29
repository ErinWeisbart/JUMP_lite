# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JUMP_core is a Python project for downloading and compressing biological imaging data from the JUMP Cell Painting dataset. It provides utilities for:

1. **Image Download**: Fetching cell painting images for specific perturbations (CRISPR, ORF, compounds)
2. **Image Compression**: Testing various compression algorithms on TIFF images using Zarr storage format

## Development Environment

This project uses:
- **Nix Flake**: Complete development environment setup
- **uv**: Python package manager for dependency management
- **Python 3.13**: Required minimum version

### Environment Setup

```bash
# Enter the Nix development shell (recommended)
nix develop

# Alternative: manual setup with uv
uv sync --all-groups
source .venv/bin/activate
```

The Nix flake automatically:
- Sets up Python 3.13 environment
- Installs uv and dependencies via `uv sync --all-groups`
- Configures necessary system libraries (libz, libGL, glib)
- Activates the virtual environment

## Core Components

### src/download_images.py
Downloads cell painting images from the JUMP dataset:
- Fetches metadata for CRISPR, ORF, and compound perturbations
- Downloads specific channels (DNA, ER, Mito) and sites
- Saves images as TIFF files in `./images/raw/`
- Uses `jump-portrait` library for data access
- Supports parallel processing via joblib

Key configuration:
- Sample size: configurable (default 10 per perturbation type)
- Channels: DNA, ER, Mito
- Sites: 1-6 (currently limited to site 1)
- Output format: TIFF with structured naming `source__batch__plate__well__channel__site.tif`

### src/compress_tif.py
Benchmarks compression algorithms on downloaded images:
- Groups images by site and metadata (source, batch, plate, well)
- Tests multiple compression codecs: Blosc variants (lz4hc, zstd, zlib), Brotli, JpegXL
- Stores compressed data in Zarr format (v2 for imagecodecs, v3 for Blosc)
- Measures compression time, decompression time, and file size

Compression results (typical):
- **Best compression ratio**: JpegXL (~46% of original)
- **Fastest decompression**: lz4hc (~1.3s)
- **Balanced**: zstd (57% size, ~2s decompression)

### metadata/repurposed_compounds.tsv
Contains compound metadata with repurposing information:
- JCP2022 identifiers
- Compound names and mechanisms of action
- Target proteins

## Key Dependencies

- `jump-portrait>=0.0.29`: JUMP dataset access
- `zarr>=3.1.3`: Compressed array storage
- `imagecodecs>=2025.8.2`: Image compression codecs
- `pooch>=1.8.2`: Data downloading utilities
- Development: `jupyter>=1.1.1`

## Common Development Tasks

### Running Scripts

```bash
# Download sample images
python src/download_images.py

# Test compression algorithms
python src/compress_tif.py
```

### Testing Compression

The compression script automatically tests all available codecs and outputs:
1. Compression times
2. Decompression times  
3. File size ratios

Results are stored in `images/` directory as separate `.zarr` folders for each codec.

### Modifying Sample Parameters

Edit `src/download_images.py`:
- `sample`: Number of perturbations per type
- `seed`: Random seed for reproducible sampling
- `channels`: Image channels to download
- `sites`: Imaging sites to include
- `correction`: Image correction type

## Data Flow

1. **Metadata Retrieval**: Fetch JCP2022 IDs for perturbations
2. **Address Resolution**: Map IDs to storage locations (source/plate/well)
3. **Image Download**: Retrieve TIFF images for specified channels/sites
4. **Compression Testing**: Apply various codecs and measure performance
5. **Result Analysis**: Compare compression ratios and timing

## File Organization

```
├── src/
│   ├── download_images.py    # Image downloading script
│   └── compress_tif.py       # Compression benchmarking
├── metadata/
│   └── repurposed_compounds.tsv  # Compound annotations
├── images/                   # Generated during execution
│   ├── raw/                  # Downloaded TIFF images
│   └── *.zarr/              # Compressed data stores
├── flake.nix                 # Nix development environment
└── pyproject.toml           # Python project configuration
```