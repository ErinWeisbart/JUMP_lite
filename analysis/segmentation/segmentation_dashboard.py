#!/usr/bin/env python3
"""
Interactive dashboard for comparing segmentation results across compression methods.

Usage:
    panel serve analysis/segmentation_dashboard.py --show

Or run directly:
    python analysis/segmentation_dashboard.py
"""

import argparse
import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
from holoviews import streams
import param
from pathlib import Path
import warnings
from functools import lru_cache
import logging

# Suppress harmless Bokeh patch warnings from DynamicMap updates
logging.getLogger('bokeh').setLevel(logging.ERROR)

pn.extension()
hv.extension('bokeh')

# Register JPEG-XL codec for zarr
try:
    import numcodecs
    from imagecodecs.numcodecs import Jpegxl
    numcodecs.register_codec(Jpegxl)
except (ImportError, AttributeError, ValueError):
    pass


@lru_cache(maxsize=16)
def _load_zarr_image_cached(zarr_store_path: str, source_id: str) -> np.ndarray:
    """Cached inner function for loading zarr images."""
    # Cell Painting channels (alphabetically sorted): AGP=0, DNA=1, ER=2, Mito=3, RNA=4
    # Display: R=none, G=RNA(4), B=DNA/Nucleus(1)
    channels = [99, 4, 1]  # 99 means no channel (will be zeros)
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
        warnings.warn(f"Failed to load zarr image: {e}")
    return None


def load_zarr_image(zarr_store_path: Path, source_id: str, channels: list = [0, 1, 2]) -> np.ndarray:
    """Load an image from a zarr store and return a 3-channel float64 array."""
    return _load_zarr_image_cached(str(zarr_store_path), source_id)


@lru_cache(maxsize=32)
def _load_instance_mask_cached(npz_path: str) -> np.ndarray:
    """Cached inner function for loading instance masks."""
    path = Path(npz_path)
    if not path.exists():
        return None
    data = np.load(path)
    for key in ['mask', 'segmentation', 'arr_0']:
        if key in data:
            return np.squeeze(data[key]).astype(np.int32)
    return np.squeeze(data[list(data.keys())[0]]).astype(np.int32)


def load_instance_mask(npz_path: Path) -> np.ndarray:
    """Load segmentation mask keeping integer labels."""
    return _load_instance_mask_cached(str(npz_path))


def get_cell_bbox(mask: np.ndarray, cell_id: int, padding: int = 20) -> tuple:
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


def mask_to_rgb(mask: np.ndarray, highlight_id: int = None) -> np.ndarray:
    """Convert instance mask to RGB with optional highlighting."""
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]

    np.random.seed(42)
    colors = np.random.rand(mask.max() + 1, 3)
    colors[0] = [0, 0, 0]  # Background is black

    rgb = colors[mask]

    if highlight_id is not None and highlight_id > 0:
        # Highlight selected cell with bright yellow border
        from scipy import ndimage
        cell_mask = mask == highlight_id
        dilated = ndimage.binary_dilation(cell_mask, iterations=3)
        border = dilated & ~cell_mask
        rgb[border] = [1, 1, 0]  # Yellow border

    return rgb


def get_mask_borders(mask: np.ndarray) -> np.ndarray:
    """Extract borders of all instances in a mask using fast gradient method."""
    # Use gradient to find edges - much faster than per-cell dilation
    # A pixel is a border if any neighbor has a different label
    padded = np.pad(mask, 1, mode='edge')

    # Check all 4 neighbors
    borders = (
        (mask != padded[:-2, 1:-1]) |  # top
        (mask != padded[2:, 1:-1]) |   # bottom
        (mask != padded[1:-1, :-2]) |  # left
        (mask != padded[1:-1, 2:])     # right
    )

    # Exclude background borders
    borders &= (mask > 0)

    return borders


def overlay_borders_on_image(img: np.ndarray, mask: np.ndarray, color: tuple = (1, 0, 0)) -> np.ndarray:
    """Overlay mask borders on an image."""
    result = img.copy()
    borders = get_mask_borders(mask)
    result[borders] = color
    return result


def overlay_two_masks(img: np.ndarray, mask1: np.ndarray, mask2: np.ndarray,
                      color1: tuple = (0, 1, 0), color2: tuple = (1, 0, 0)) -> np.ndarray:
    """Overlay two mask borders on an image with different colors.

    Args:
        img: Base image
        mask1: First mask (e.g., GT) - shown in color1 (green by default)
        mask2: Second mask (e.g., predicted) - shown in color2 (red by default)

    Returns:
        Image with both mask borders overlayed. Overlapping borders shown in yellow.
    """
    result = img.copy()
    borders1 = get_mask_borders(mask1)
    borders2 = get_mask_borders(mask2)

    # Overlapping borders in yellow
    overlap = borders1 & borders2
    only1 = borders1 & ~borders2
    only2 = borders2 & ~borders1

    result[only1] = color1      # GT only - green
    result[only2] = color2      # Pred only - red
    result[overlap] = (1, 1, 0) # Both - yellow

    return result


class SegmentationDashboard(param.Parameterized):
    """Interactive dashboard for segmentation comparison."""

    # File paths
    mappings_dir = param.String(default="")
    zarr_root = param.String(default="/work/datasets/jump_target2_4plate")
    masks_root = param.String(default="/work/datasets/aliby_output/cp_measure/jump_target2_4plate")
    segment_step = param.String(default="segment_cell")
    gt_method = param.String(default="zstd.zarr")

    # Selectors
    source_id = param.Selector(default=None, objects=[])
    file_name = param.Selector(default=None, objects=[])
    codec = param.Selector(default=None, objects=[])
    cell_id = param.Selector(default=None, objects=[])
    thresh = param.Selector(default=0.5, objects=[0.5, 0.7, 0.8, 0.9])

    def __init__(self, mappings_dir: str, **params):
        super().__init__(**params)
        self.mappings_dir = mappings_dir
        mappings_path = Path(mappings_dir)

        # Discover available codec parquet files
        # Expected naming: {segment_step}_{codec}.parquet (e.g., segment_nuclei_jpegxl_lossy_lq.parquet)
        self._codec_files = {}
        for pq_file in sorted(mappings_path.glob("*.parquet")):
            stem = pq_file.stem
            # Strip segment_step prefix to get codec name
            for prefix in ["segment_cell_", "segment_nuclei_"]:
                if stem.startswith(prefix):
                    codec_name = stem[len(prefix):]
                    # Add .zarr suffix for zarr store lookup
                    self._codec_files[f"{codec_name}.zarr"] = pq_file
                    # Auto-detect segment_step from first file
                    if not hasattr(self, '_detected_segment_step'):
                        self._detected_segment_step = prefix.rstrip("_")
                    break
            else:
                # Fallback: use full stem as codec name
                self._codec_files[f"{stem}.zarr"] = pq_file

        # Cache for loaded dataframes per codec
        self._df_cache = {}
        self._current_df = None

        # Initialize for clickable masks
        self._gt_mask_cache = None
        self._mask_shape = None

        # Stable tap stream - reused across renders to avoid warnings
        self._tap_stream = streams.Tap(x=None, y=None)

        # Cache for loaded images/masks to avoid reloading
        self._image_cache = {}
        self._cache_key = None

        # Use auto-detected segment_step if available and not explicitly set
        if hasattr(self, '_detected_segment_step') and self.segment_step == "segment_cell":
            self.segment_step = self._detected_segment_step
            print(f"Auto-detected segment_step: {self.segment_step}")

        # Initialize codec selector
        codecs = sorted(self._codec_files.keys())
        self.param.codec.objects = codecs
        if codecs:
            self.codec = codecs[0]
            self._load_codec_df(self.codec)

    def _load_codec_df(self, codec: str):
        """Load the parquet file for a specific codec."""
        if codec in self._df_cache:
            self._current_df = self._df_cache[codec]
        elif codec in self._codec_files:
            df = pd.read_parquet(self._codec_files[codec])
            self._df_cache[codec] = df
            self._current_df = df

            # Update source_ids based on this codec's data
            source_ids = sorted(df['source_id'].unique().tolist())
            self.param.source_id.objects = source_ids
            if source_ids and self.source_id not in source_ids:
                self.source_id = source_ids[0]

            # Update thresholds
            if 'thresh' in df.columns:
                thresholds = sorted(df['thresh'].unique().tolist())
                self.param.thresh.objects = thresholds
                if thresholds and self.thresh not in thresholds:
                    self.thresh = thresholds[0]

    @property
    def df(self):
        """Return the current codec's dataframe."""
        return self._current_df

    @param.depends('codec', watch=True)
    def _update_codec_df(self):
        """Load new dataframe when codec changes."""
        if self.codec is None:
            return
        self._load_codec_df(self.codec)

    @param.depends('source_id', watch=True)
    def _update_files(self):
        if self.source_id is None:
            return
        files = sorted(self.df[self.df['source_id'] == self.source_id]['file'].unique().tolist())
        self.param.file_name.objects = files
        if files:
            self.file_name = files[0]

    def _filter_df(self, source_id=None, file_name=None, thresh=None, gt_id=None):
        """Filter the current dataframe. Handles both per-codec files (no 'method' column) and combined files."""
        if self.df is None:
            return pd.DataFrame()

        mask = pd.Series(True, index=self.df.index)

        if source_id is not None:
            mask &= (self.df['source_id'] == source_id)
        if file_name is not None:
            mask &= (self.df['file'] == file_name)
        if thresh is not None and 'thresh' in self.df.columns:
            mask &= (self.df['thresh'] == thresh)
        if gt_id is not None:
            mask &= (self.df['gt_id'] == gt_id)

        # Only filter by method if the column exists (for combined files)
        if 'method' in self.df.columns:
            mask &= (self.df['method'] == self.codec)

        return self.df[mask]

    @param.depends('source_id', 'file_name', 'codec', 'thresh', watch=True)
    def _update_cells(self):
        if None in (self.source_id, self.file_name, self.codec):
            return

        subset = self._filter_df(
            source_id=self.source_id,
            file_name=self.file_name,
            thresh=self.thresh
        )

        # Get GT cell IDs that have matches
        gt_ids = subset[subset['gt_id'].notna()]['gt_id'].astype(int).unique().tolist()
        gt_ids = sorted(gt_ids)

        self.param.cell_id.objects = gt_ids
        if gt_ids:
            self.cell_id = gt_ids[0]

    def _load_images_and_masks(self):
        """Load all required images and masks with caching."""
        if None in (self.source_id, self.file_name, self.codec):
            return None, None, None, None

        # Check cache
        cache_key = (self.source_id, self.file_name, self.codec)
        if cache_key == self._cache_key and self._image_cache:
            return (
                self._image_cache.get('gt_img'),
                self._image_cache.get('gt_mask'),
                self._image_cache.get('codec_img'),
                self._image_cache.get('codec_mask')
            )

        zarr_root = Path(self.zarr_root)
        masks_root = Path(self.masks_root)

        # Load GT image and mask
        gt_zarr = zarr_root / self.gt_method
        gt_img = load_zarr_image(gt_zarr, self.source_id)

        gt_mask_path = masks_root / self.gt_method / "steps" / self.source_id / self.segment_step / self.file_name
        gt_mask = load_instance_mask(gt_mask_path)

        # Load codec image and mask
        codec_zarr = zarr_root / self.codec
        codec_img = load_zarr_image(codec_zarr, self.source_id)

        codec_mask_path = masks_root / self.codec / "steps" / self.source_id / self.segment_step / self.file_name
        codec_mask = load_instance_mask(codec_mask_path)

        # Update cache
        self._cache_key = cache_key
        self._image_cache = {
            'gt_img': gt_img,
            'gt_mask': gt_mask,
            'codec_img': codec_img,
            'codec_mask': codec_mask
        }

        return gt_img, gt_mask, codec_img, codec_mask

    def _handle_tap(self, x, y):
        """Handle tap/click events on the mask image."""
        if x is None or y is None:
            return hv.Points([])
        if self._gt_mask_cache is None:
            return hv.Points([])

        # Convert coordinates to pixel indices
        h, w = self._gt_mask_cache.shape
        px = int(x)
        py = int(h - y)  # Flip y coordinate

        # Bounds check
        if 0 <= px < w and 0 <= py < h:
            clicked_id = self._gt_mask_cache[py, px]
            if clicked_id > 0 and clicked_id in self.param.cell_id.objects:
                self.cell_id = int(clicked_id)

        # Return a point marker at click location
        return hv.Points([(x, y)]).opts(size=10, color='yellow', marker='x')

    @property
    def _click_handler_view(self):
        """Status text for click handling."""
        return pn.pane.Markdown("*Click on the GT Segmentation image to select a cell*", styles={'font-style': 'italic', 'color': 'gray'})

    @param.depends('source_id', 'file_name', 'codec')
    def _view_static_images(self):
        """Static views that only change when source/file/codec changes."""
        gt_img, gt_mask, codec_img, codec_mask = self._load_images_and_masks()

        if gt_img is None:
            return pn.pane.Markdown("### No image data available")

        # Cache the GT mask for click handling
        self._gt_mask_cache = gt_mask

        h, w = gt_img.shape[:2]
        self._mask_shape = (h, w)

        # Raw images
        gt_img_hv = hv.RGB(gt_img, bounds=(0, 0, w, h)).opts(
            title=f'GT Image ({self.gt_method})', xaxis=None, yaxis=None, width=400, height=400
        )
        codec_img_hv = hv.RGB(codec_img if codec_img is not None else np.zeros_like(gt_img),
                              bounds=(0, 0, w, h)).opts(
            title=f'Codec Image ({self.codec})', xaxis=None, yaxis=None, width=400, height=400
        )

        # Border overlays (static - don't depend on cell_id)
        if gt_mask is not None:
            gt_with_borders = overlay_borders_on_image(gt_img, gt_mask, color=(0, 1, 0))
            gt_borders_hv = hv.RGB(gt_with_borders, bounds=(0, 0, w, h)).opts(
                title='GT Image + GT Borders', xaxis=None, yaxis=None, width=400, height=400
            )
        else:
            gt_borders_hv = gt_img_hv.opts(title='GT Image (no mask)')

        if codec_mask is not None and codec_img is not None:
            codec_with_borders = overlay_borders_on_image(codec_img, codec_mask, color=(1, 0, 0))
            codec_borders_hv = hv.RGB(codec_with_borders, bounds=(0, 0, w, h)).opts(
                title='Codec Image + Codec Borders', xaxis=None, yaxis=None, width=400, height=400
            )
        else:
            codec_borders_hv = codec_img_hv.opts(title='Codec Image (no mask)')

        if gt_mask is not None and codec_mask is not None:
            gt_both_overlay = overlay_two_masks(gt_img, gt_mask, codec_mask,
                                                 color1=(0, 1, 0), color2=(1, 0, 0))
            gt_comparison_hv = hv.RGB(gt_both_overlay, bounds=(0, 0, w, h)).opts(
                title='GT Image: GT(green) vs Codec(red)', xaxis=None, yaxis=None, width=400, height=400
            )
        else:
            gt_comparison_hv = gt_img_hv.opts(title='Comparison N/A')

        if gt_mask is not None and codec_mask is not None and codec_img is not None:
            codec_both_overlay = overlay_two_masks(codec_img, gt_mask, codec_mask,
                                                    color1=(0, 1, 0), color2=(1, 0, 0))
            codec_comparison_hv = hv.RGB(codec_both_overlay, bounds=(0, 0, w, h)).opts(
                title='Codec Image: GT(green) vs Codec(red)', xaxis=None, yaxis=None, width=400, height=400
            )
        else:
            codec_comparison_hv = codec_img_hv.opts(title='Comparison N/A')

        # Cache for mask views
        self._static_cache = {
            'gt_img_hv': gt_img_hv,
            'codec_img_hv': codec_img_hv,
            'h': h, 'w': w
        }

        side_layout = (gt_borders_hv + codec_borders_hv + gt_comparison_hv + codec_comparison_hv).cols(2)

        legend_md = pn.pane.Markdown(
            "**Overlay Legend:** Green = GT borders | Red = Codec borders | Yellow = Overlap",
            styles={'font-size': '12px'}
        )

        return pn.Column(pn.pane.Markdown("### Border Overlays"), pn.pane.HoloViews(side_layout), legend_md)

    @param.depends('source_id', 'file_name', 'codec', 'cell_id', 'thresh')
    def _view_dynamic_masks(self):
        """Dynamic mask views that update when cell_id/thresh changes."""
        gt_img, gt_mask, codec_img, codec_mask = self._load_images_and_masks()

        if gt_img is None:
            return pn.pane.Markdown("### No mask data")

        h, w = gt_img.shape[:2]

        # Get cached static images or create placeholders
        if hasattr(self, '_static_cache'):
            gt_img_hv = self._static_cache['gt_img_hv']
            codec_img_hv = self._static_cache['codec_img_hv']
        else:
            gt_img_hv = hv.RGB(gt_img, bounds=(0, 0, w, h)).opts(
                title=f'GT Image', xaxis=None, yaxis=None, width=400, height=400
            )
            codec_img_hv = hv.RGB(codec_img if codec_img is not None else np.zeros_like(gt_img),
                                  bounds=(0, 0, w, h)).opts(
                title=f'Codec Image', xaxis=None, yaxis=None, width=400, height=400
            )

        # GT mask with cell highlighting
        if gt_mask is not None:
            gt_rgb = mask_to_rgb(gt_mask, self.cell_id)
            gt_mask_hv = hv.RGB(gt_rgb, bounds=(0, 0, w, h)).opts(
                title='GT Segmentation (Click to select cell)', xaxis=None, yaxis=None,
                width=400, height=400
            )
            self._tap_stream.source = gt_mask_hv
            tap_dmap = hv.DynamicMap(self._handle_tap, streams=[self._tap_stream])
            gt_mask_with_tap = (gt_mask_hv * tap_dmap).opts(
                hv.opts.Points(size=15, color='yellow', marker='x', line_width=3)
            )
        else:
            gt_mask_with_tap = hv.RGB(np.zeros_like(gt_img), bounds=(0, 0, w, h)).opts(
                title='GT Segmentation', xaxis=None, yaxis=None, width=400, height=400
            )

        # Codec mask with matched cell highlighting
        if codec_mask is not None:
            subset = self._filter_df(
                source_id=self.source_id,
                file_name=self.file_name,
                thresh=self.thresh,
                gt_id=self.cell_id
            )
            pred_id = subset['pred_id'].values[0] if len(subset) > 0 and pd.notna(subset['pred_id'].values[0]) else None
            pred_id = int(pred_id) if pred_id is not None else None

            codec_rgb = mask_to_rgb(codec_mask, pred_id)
            codec_mask_hv = hv.RGB(codec_rgb, bounds=(0, 0, w, h)).opts(
                title='Codec Segmentation', xaxis=None, yaxis=None, width=400, height=400
            )
        else:
            codec_mask_hv = hv.RGB(np.zeros_like(gt_img), bounds=(0, 0, w, h)).opts(
                title='Codec Segmentation', xaxis=None, yaxis=None, width=400, height=400
            )

        main_layout = (gt_img_hv + codec_img_hv + gt_mask_with_tap + codec_mask_hv).cols(2)

        return pn.Column(pn.pane.Markdown("### Main Views"), pn.pane.HoloViews(main_layout), self._click_handler_view)

    def view_full_images(self):
        """Display full images and masks side by side with clickable masks."""
        return pn.Row(
            self._view_dynamic_masks,
            self._view_static_images
        )

    @param.depends('source_id', 'file_name', 'codec', 'cell_id', 'thresh')
    def view_cell_crops(self):
        """Display cropped cell comparison."""
        if self.cell_id is None:
            return pn.pane.Markdown("### Select a cell ID")

        gt_img, gt_mask, codec_img, codec_mask = self._load_images_and_masks()

        if gt_img is None or gt_mask is None:
            return pn.pane.Markdown("### No data available")

        # Get bbox from GT mask
        bbox = get_cell_bbox(gt_mask, self.cell_id)
        if bbox is None:
            return pn.pane.Markdown(f"### Cell {self.cell_id} not found in GT mask")

        # Find matched pred_id
        subset = self._filter_df(
            source_id=self.source_id,
            file_name=self.file_name,
            thresh=self.thresh,
            gt_id=self.cell_id
        )

        if len(subset) == 0:
            return pn.pane.Markdown(f"### No mapping found for cell {self.cell_id}")

        match_info = subset.iloc[0]
        pred_id = int(match_info['pred_id']) if pd.notna(match_info['pred_id']) else None
        iou_score = match_info['iou_score']
        match_type = match_info['match_type']

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Crop GT image
        gt_img_crop = crop_region(gt_img, bbox)
        axes[0, 0].imshow(gt_img_crop)
        axes[0, 0].set_title(f'GT Image (Cell {self.cell_id})', fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')

        # Crop Codec image
        if codec_img is not None:
            codec_img_crop = crop_region(codec_img, bbox)
            axes[0, 1].imshow(codec_img_crop)
        axes[0, 1].set_title(f'Codec Image ({self.codec})', fontsize=11, fontweight='bold')
        axes[0, 1].axis('off')

        # Crop GT mask
        gt_mask_crop = crop_region(gt_mask, bbox)
        gt_mask_rgb = mask_to_rgb(gt_mask_crop, self.cell_id)
        axes[1, 0].imshow(gt_mask_rgb)
        axes[1, 0].set_title(f'GT Mask (Cell {self.cell_id})', fontsize=11, fontweight='bold')
        axes[1, 0].axis('off')

        # Crop Codec mask
        if codec_mask is not None and pred_id is not None:
            codec_mask_crop = crop_region(codec_mask, bbox)
            codec_mask_rgb = mask_to_rgb(codec_mask_crop, pred_id)
            axes[1, 1].imshow(codec_mask_rgb)
            axes[1, 1].set_title(f'Codec Mask (Cell {pred_id})', fontsize=11, fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Match', ha='center', va='center', fontsize=14)
            axes[1, 1].set_title('Codec Mask (No Match)', fontsize=11, fontweight='bold')
        axes[1, 1].axis('off')

        fig.suptitle(f'Match Type: {match_type} | IoU: {iou_score:.4f}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.close(fig)

        return pn.pane.Matplotlib(fig, tight=True)

    @param.depends('source_id', 'file_name', 'codec', 'cell_id', 'thresh')
    def view_mapping_info(self):
        """Display mapping information for selected cell."""
        if None in (self.source_id, self.file_name, self.codec, self.cell_id):
            return pn.pane.Markdown("### Select all parameters")

        subset = self._filter_df(
            source_id=self.source_id,
            file_name=self.file_name,
            thresh=self.thresh,
            gt_id=self.cell_id
        )

        if len(subset) == 0:
            return pn.pane.Markdown("### No mapping found")

        info = subset.iloc[0]
        thresh_str = f"- **Threshold:** {info['thresh']}" if 'thresh' in info else ""
        md = f"""
### Mapping Info
- **GT Cell ID:** {int(info['gt_id'])}
- **Pred Cell ID:** {int(info['pred_id']) if pd.notna(info['pred_id']) else 'None'}
- **IoU Score:** {info['iou_score']:.4f}
- **Match Type:** {info['match_type']}
{thresh_str}
"""
        return pn.pane.Markdown(md)

    def panel(self):
        """Create the dashboard layout."""
        controls = pn.Column(
            pn.pane.Markdown("## Controls"),
            pn.Param(self.param.source_id, name="Source ID"),
            pn.Param(self.param.file_name, name="File"),
            pn.Param(self.param.codec, name="Codec"),
            pn.Param(self.param.thresh, name="IoU Threshold"),
            pn.Param(self.param.cell_id, name="Cell ID"),
            self.view_mapping_info,
            width=300
        )

        main = pn.Tabs(
            ("Full Images", self.view_full_images),
            ("Cell Crops", self.view_cell_crops),
        )

        return pn.Row(controls, main)


def main():
    parser = argparse.ArgumentParser(description="Segmentation comparison dashboard")
    parser.add_argument("--mappings-dir", type=str, help="Directory containing per-codec instance mappings parquet files")
    parser.add_argument("--mappings", type=str, help="Single parquet file (backward compatibility, will use parent dir)")
    parser.add_argument("--zarr-root", type=str, default="/work/datasets/jump_target2_4plate", help="Root directory for zarr images")
    parser.add_argument("--masks-root", type=str, default="/work/datasets/aliby_output/cp_measure/jump_target2_4plate", help="Root directory for mask files")
    parser.add_argument("--segment-step", type=str, default="segment_cell", help="Segmentation step name")
    parser.add_argument("--gt-method", type=str, default="zstd.zarr", help="Ground truth method name")
    parser.add_argument("--port", type=int, default=5006, help="Port for the server")

    args = parser.parse_args()

    # Handle both --mappings (single file) and --mappings-dir (directory)
    if args.mappings_dir:
        mappings_dir = args.mappings_dir
    elif args.mappings:
        # Use parent directory of the single file
        mappings_dir = str(Path(args.mappings).parent)
        print(f"Using directory from --mappings: {mappings_dir}")
    else:
        parser.error("Either --mappings-dir or --mappings is required")

    dashboard = SegmentationDashboard(
        mappings_dir=mappings_dir,
        zarr_root=args.zarr_root,
        masks_root=args.masks_root,
        segment_step=args.segment_step,
        gt_method=args.gt_method
    )

    app = dashboard.panel()
    app.servable()

    pn.serve(app, port=args.port, show=True)


if __name__ == "__main__":
    main()
