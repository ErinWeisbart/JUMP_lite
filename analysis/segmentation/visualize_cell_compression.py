#!/usr/bin/env python3
"""
Visualize a single cell across all compression levels.

Shows a 2x5 grid:
- Row 1: Image crops for GT and 4 compression levels
- Row 2: Corresponding mask crops with cell ID annotations

Usage:
    python visualize_cell_compression.py --mappings-dir /path/to/instance_mappings/
    python visualize_cell_compression.py --source-id <source_id> --file <file> --cell-id <gt_cell_id>
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from functools import lru_cache
import random

# Register JPEG-XL codec for zarr
try:
    import numcodecs
    from imagecodecs.numcodecs import Jpegxl
    numcodecs.register_codec(Jpegxl)
except (ImportError, AttributeError, ValueError):
    pass


@lru_cache(maxsize=32)
def load_zarr_image(zarr_store_path: str, source_id: str) -> np.ndarray:
    """Load image from zarr store, return 3-channel RGB for display."""
    # Cell Painting channels (alphabetically sorted): AGP=0, DNA=1, ER=2, Mito=3, RNA=4
    # Display: R=none, G=RNA(4), B=DNA(1)
    channels = [99, 4, 1]  # 99 = no channel (zeros)
    path = Path(zarr_store_path)
    if not path.exists():
        return None
    try:
        import zarr
        store = zarr.open(zarr_store_path, mode='r')
        if source_id not in store:
            return None
        img = store[source_id][:]

        if img.ndim == 3 and img.shape[0] < img.shape[1]:
            selected = np.zeros((*img.shape[1:], 3), dtype=np.float64)
            for i, ch in enumerate(channels):
                if ch < img.shape[0]:
                    ch_data = img[ch].astype(np.float64)
                    ch_min = np.percentile(ch_data, 1)
                    ch_max = np.percentile(ch_data, 99)
                    if ch_max > ch_min:
                        ch_clipped = np.clip(ch_data, ch_min, ch_max)
                        selected[..., i] = (ch_clipped - ch_min) / (ch_max - ch_min)
            return selected
    except Exception as e:
        print(f"Warning: Failed to load zarr image: {e}")
    return None


@lru_cache(maxsize=64)
def load_instance_mask(npz_path: str) -> np.ndarray:
    """Load segmentation mask keeping integer labels."""
    path = Path(npz_path)
    if not path.exists():
        return None
    data = np.load(path)
    for key in ['mask', 'segmentation', 'arr_0']:
        if key in data:
            return np.squeeze(data[key]).astype(np.int32)
    return np.squeeze(data[list(data.keys())[0]]).astype(np.int32)


def get_cell_bbox(mask: np.ndarray, cell_id: int, padding: int = 30) -> tuple:
    """Get bounding box for a cell with padding."""
    coords = np.argwhere(mask == cell_id)
    if len(coords) == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add padding
    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    y_max = min(mask.shape[0], y_max + padding)
    x_max = min(mask.shape[1], x_max + padding)

    return (y_min, y_max, x_min, x_max)


def crop_region(arr: np.ndarray, bbox: tuple) -> np.ndarray:
    """Crop array to bounding box."""
    y_min, y_max, x_min, x_max = bbox
    return arr[y_min:y_max, x_min:x_max]


def main():
    parser = argparse.ArgumentParser(description="Visualize cell across compression levels")
    parser.add_argument("--mappings-dir", type=str, required=True,
                        help="Directory containing per-codec instance mappings parquet files")
    parser.add_argument("--zarr-root", type=str, default="/work/datasets/jump_target2_4plate",
                        help="Root directory for zarr images")
    parser.add_argument("--masks-root", type=str, default="/work/datasets/aliby_output/cp_measure/jump_target2_4plate",
                        help="Root directory for mask files")
    parser.add_argument("--gt-method", type=str, default="zstd.zarr",
                        help="Ground truth method name")
    parser.add_argument("--source-id", type=str, default=None,
                        help="Specific source_id (random if not provided)")
    parser.add_argument("--file", type=str, default=None,
                        help="Specific file name (random if not provided)")
    parser.add_argument("--cell-id", type=int, default=None,
                        help="Specific GT cell ID (random if not provided)")
    parser.add_argument("--segment-step", type=str, default="segment_cell",
                        help="Segmentation step name")
    parser.add_argument("--thresh", type=float, default=0.5,
                        help="IoU threshold for matching lookup")
    parser.add_argument("--output", type=str, default="cell_compression_comparison.png",
                        help="Output file path (for multiple samples, index is appended)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Number of random samples to generate")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    mappings_dir = Path(args.mappings_dir)
    zarr_root = Path(args.zarr_root)
    masks_root = Path(args.masks_root)

    # Define compression levels (GT + 4 lossy levels)
    # Order: GT (zstd), then high quality to low quality
    compression_levels = [
        ('GT (zstd)', args.gt_method),
        ('HQ', 'jpegxl_lossy_hq.zarr'),
        ('Effort 3', 'jpegxl_lossy_effort_3.zarr'),
        ('MQ', 'jpegxl_lossy_mq.zarr'),
        ('LQ', 'jpegxl_lossy_lq.zarr'),
    ]

    # Load parquet files for each codec to get mappings
    codec_dfs = {}
    for label, codec_name in compression_levels[1:]:  # Skip GT
        # Find matching parquet file
        codec_base = codec_name.replace('.zarr', '')
        pq_pattern = f"{args.segment_step}_{codec_base}.parquet"
        pq_file = mappings_dir / pq_pattern
        if pq_file.exists():
            df = pd.read_parquet(pq_file)
            # Filter by threshold
            if 'thresh' in df.columns:
                closest_thresh = min(df['thresh'].unique(), key=lambda x: abs(x - args.thresh))
                df = df[np.isclose(df['thresh'], closest_thresh)]
            codec_dfs[codec_name] = df
            print(f"Loaded {codec_name}: {len(df)} rows")
        else:
            print(f"Warning: Parquet file not found: {pq_file}")

    if not codec_dfs:
        print("Error: No parquet files found")
        return

    # Get all unique (source_id, file) pairs
    first_df = list(codec_dfs.values())[0]
    all_pairs = first_df[['source_id', 'file']].drop_duplicates()

    # Determine samples to process
    n_samples = args.n_samples
    samples_to_process = []

    if args.source_id is not None and args.file is not None and args.cell_id is not None:
        # Specific cell requested
        samples_to_process = [(args.source_id, args.file, args.cell_id)]
        n_samples = 1
    else:
        # Random sampling
        sampled_pairs = all_pairs.sample(n=min(n_samples, len(all_pairs))).values.tolist()
        for source_id, file_name in sampled_pairs:
            # Load GT mask to get available cell IDs
            gt_mask_path = masks_root / args.gt_method / "steps" / source_id / args.segment_step / file_name
            gt_mask = load_instance_mask(str(gt_mask_path))
            if gt_mask is not None:
                gt_cell_ids = sorted(set(np.unique(gt_mask)) - {0})
                if gt_cell_ids:
                    gt_cell_id = random.choice(gt_cell_ids)
                    samples_to_process.append((source_id, file_name, gt_cell_id))

    print(f"Processing {len(samples_to_process)} samples...")

    # Process each sample
    for sample_idx, (source_id, file_name, gt_cell_id) in enumerate(samples_to_process):
        print(f"\n[{sample_idx + 1}/{len(samples_to_process)}] {source_id} / {file_name} / Cell {gt_cell_id}")

        # Load GT mask
        gt_mask_path = masks_root / args.gt_method / "steps" / source_id / args.segment_step / file_name
        gt_mask = load_instance_mask(str(gt_mask_path))
        if gt_mask is None:
            print(f"  Warning: Could not load GT mask, skipping")
            continue

        # Get bounding box from GT mask
        bbox = get_cell_bbox(gt_mask, gt_cell_id)
        if bbox is None:
            print(f"  Warning: Could not find cell {gt_cell_id} in GT mask, skipping")
            continue

        # Build mapping: codec -> pred_id for this cell
        cell_mappings = {'GT (zstd)': gt_cell_id}  # GT maps to itself
        for codec_name, df in codec_dfs.items():
            img_df = df[(df['source_id'] == source_id) & (df['file'] == file_name)]
            cell_row = img_df[img_df['gt_id'] == gt_cell_id]
            if len(cell_row) > 0:
                row = cell_row.iloc[0]
                pred_id = int(row['pred_id']) if pd.notna(row['pred_id']) else None
                iou = row['iou_score']
                match_type = row['match_type']
                cell_mappings[codec_name] = (pred_id, iou, match_type)
            else:
                cell_mappings[codec_name] = (None, 0, 'NOT_FOUND')

        # Create figure: 2 rows x 5 columns
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))

        for col_idx, (label, codec_name) in enumerate(compression_levels):
            # Load image
            zarr_path = zarr_root / codec_name
            img = load_zarr_image(str(zarr_path), source_id)
            if img is None:
                axes[0, col_idx].text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center',
                                       transform=axes[0, col_idx].transAxes, fontsize=12)
                axes[0, col_idx].axis('off')
            else:
                img_crop = crop_region(img, bbox)
                axes[0, col_idx].imshow(img_crop)
                axes[0, col_idx].axis('off')

            # Load mask
            mask_path = masks_root / codec_name / "steps" / source_id / args.segment_step / file_name
            mask = load_instance_mask(str(mask_path))

            if mask is None:
                axes[1, col_idx].text(0.5, 0.5, 'Mask\nNot Found', ha='center', va='center',
                                       transform=axes[1, col_idx].transAxes, fontsize=12)
                axes[1, col_idx].axis('off')
            else:
                mask_crop = crop_region(mask, bbox)

                # Create RGB visualization with cell of interest highlighted in red
                mask_rgb = np.zeros((*mask_crop.shape, 3), dtype=np.float32)

                # Show all cells in gray
                mask_rgb[mask_crop > 0] = [0.4, 0.4, 0.4]

                # Determine which cell ID to highlight
                if label == 'GT (zstd)':
                    highlight_id = gt_cell_id
                else:
                    mapping = cell_mappings.get(codec_name, (None, 0, 'N/A'))
                    highlight_id = mapping[0]  # pred_id

                # Highlight cell of interest in red
                if highlight_id is not None:
                    mask_rgb[mask_crop == highlight_id] = [1.0, 0.2, 0.2]

                axes[1, col_idx].imshow(mask_rgb)

                # Add cell ID text at the center of the cell
                if highlight_id is not None:
                    cell_coords = np.argwhere(mask_crop == highlight_id)
                    if len(cell_coords) > 0:
                        # Calculate centroid
                        center_y = cell_coords[:, 0].mean()
                        center_x = cell_coords[:, 1].mean()

                        # Build label text
                        if label == 'GT (zstd)':
                            cell_label = f"{gt_cell_id}"
                        else:
                            cell_label = f"{highlight_id}\n(GT:{gt_cell_id})"

                        axes[1, col_idx].text(center_x, center_y, cell_label,
                                               ha='center', va='center',
                                               fontsize=9, fontweight='bold',
                                               color='white',
                                               bbox=dict(boxstyle='round,pad=0.2',
                                                        facecolor='black', alpha=0.7))

                axes[1, col_idx].axis('off')

            # Build title with cell ID info
            if label == 'GT (zstd)':
                title_img = f"{label}"
                title_mask = f"GT ID: {gt_cell_id}"
            else:
                mapping = cell_mappings.get(codec_name, (None, 0, 'N/A'))
                pred_id, iou, match_type = mapping
                title_img = f"{label}"
                if pred_id is not None:
                    title_mask = f"GT:{gt_cell_id} → Pred:{pred_id}\nIoU:{iou:.3f} ({match_type})"
                else:
                    title_mask = f"GT:{gt_cell_id} → No match\n({match_type})"

            axes[0, col_idx].set_title(title_img, fontsize=12, fontweight='bold')
            axes[1, col_idx].set_title(title_mask, fontsize=10)

        # Add row labels
        axes[0, 0].text(-0.15, 0.5, 'Image', transform=axes[0, 0].transAxes,
                        fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)
        axes[1, 0].text(-0.15, 0.5, 'Mask', transform=axes[1, 0].transAxes,
                        fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)

        fig.suptitle(f'Cell Compression Comparison\n{source_id}\nGT Cell ID: {gt_cell_id}',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save figure
        output_base = Path(args.output)
        if len(samples_to_process) > 1:
            output_path = output_base.parent / f"{output_base.stem}_{sample_idx + 1:03d}{output_base.suffix}"
        else:
            output_path = output_base
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {output_path}")

        # Print mapping summary
        print("  Mappings:")
        for label, codec_name in compression_levels:
            if label == 'GT (zstd)':
                print(f"    {label:12s}: GT ID = {gt_cell_id}")
            else:
                mapping = cell_mappings.get(codec_name, (None, 0, 'N/A'))
                pred_id, iou, match_type = mapping
                if pred_id is not None:
                    print(f"    {label:12s}: GT {gt_cell_id} → Pred {pred_id} (IoU={iou:.3f}, {match_type})")
                else:
                    print(f"    {label:12s}: GT {gt_cell_id} → No match ({match_type})")

    print(f"\nDone! Generated {len(samples_to_process)} figures.")


if __name__ == "__main__":
    main()
