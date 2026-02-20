#!/usr/bin/env python3
"""
Interactive Cell Count Difference Viewer using Panel/Bokeh

Click on points in the scatter plot to view the corresponding images and segmentation masks.

Usage:
    panel serve interactive_cell_count_viewer.py --show --args \
        --csv output/segmentation_comparison_with_mapping/segmentation_comparison_with_mapping_large_cell_count_diff.csv \
        --mask-root /work/datasets/aliby_output/cp_measure/jump_target2_4plate \
        --zarr-root /work/datasets/jump_target2_4plate \
        --ground-truth zstd.zarr

Or run directly (opens browser automatically):
    python interactive_cell_count_viewer.py --csv ... --mask-root ... --zarr-root ... --ground-truth ...
"""

import argparse
import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
from holoviews import streams, opts
from pathlib import Path
import zarr
import tifffile

pn.extension()
hv.extension('bokeh')

# Register imagecodecs numcodecs for JpegXL support
try:
    from imagecodecs.numcodecs import Brotli, Jpegxl
    import numcodecs
    numcodecs.register_codec(Brotli)
    numcodecs.register_codec(Jpegxl)
except (ImportError, AttributeError) as e:
    print(f"Warning: imagecodecs.numcodecs not available: {e}")


class InteractiveCellCountViewer:
    def __init__(self, df, mask_root, zarr_root, ground_truth, segment_step='segment_cell'):
        self.df = df
        self.mask_root = Path(mask_root)
        self.zarr_root = Path(zarr_root)
        self.ground_truth = ground_truth
        self.segment_step = segment_step
        self.current_selection = None
        self.current_index = None

        # Create tap stream for click handling
        self.tap_stream = streams.Tap(x=None, y=None)

        # Caching
        self._image_cache = {}  # Cache for loaded images
        self._mask_cache = {}   # Cache for loaded masks

    def mask_to_rgb(self, mask):
        """Convert instance mask to RGB with distinct random colors."""
        if mask is None:
            return None

        # Generate random colors for each instance
        np.random.seed(42)  # Fixed seed for consistent colors
        max_id = int(mask.max())
        colors = np.random.rand(max_id + 1, 3)
        colors[0] = [0, 0, 0]  # Background is black

        # Map mask to RGB
        rgb = colors[mask]
        return rgb

    def load_zarr_image_raw(self, zarr_path, source_id):
        """Load image from zarr store and create RGB composite."""
        # Check cache first
        cache_key = (str(zarr_path), source_id)
        if cache_key in self._image_cache:
            return self._image_cache[cache_key]

        try:
            # Open zarr store
            store = zarr.storage.LocalStore(zarr_path)
            root = zarr.group(store)

            if source_id in root.keys():
                img = root[source_id][:]

                # Cell Painting channels (alphabetically sorted): AGP=0, DNA=1, ER=2, Mito=3, RNA=4
                # Create RGB composite: R=none, G=RNA(4), B=DNA/Nucleus(1)
                if img.ndim == 3 and img.shape[0] == 5:
                    # 5-channel Cell Painting image
                    channels = [99, 4, 1]  # 99 means no channel (will be zeros)
                    rgb = np.zeros((*img.shape[1:], 3), dtype=np.float64)

                    for i, ch in enumerate(channels):
                        if ch < img.shape[0]:
                            ch_data = img[ch].astype(np.float64)
                            # Per-channel percentile normalization
                            ch_min = np.percentile(ch_data, 0.1)
                            ch_max = np.percentile(ch_data, 99.9)
                            if ch_max > ch_min:
                                ch_clipped = np.clip(ch_data, ch_min, ch_max)
                                rgb[..., i] = (ch_clipped - ch_min) / (ch_max - ch_min)

                    # Cache and return
                    self._image_cache[cache_key] = rgb
                    return rgb
                elif img.ndim == 3 and img.shape[0] < img.shape[1]:
                    # Channels first, take max projection
                    img_show = np.max(img, axis=0).astype(np.float32)
                    self._image_cache[cache_key] = img_show
                    return img_show
                else:
                    img_float = img.astype(np.float32)
                    self._image_cache[cache_key] = img_float
                    return img_float
            else:
                print(f"Warning: {source_id} not found in {zarr_path}")
                return None
        except Exception as e:
            print(f"Error loading image from {zarr_path}: {e}")
            return None

    def normalize_image(self, img, p_low, p_high):
        """For RGB images, return as-is (already normalized per channel)."""
        if img is None:
            return None

        # If already RGB (3 channels), return as-is
        if img.ndim == 3 and img.shape[2] == 3:
            return img

        # Otherwise apply percentile normalization
        if p_high > p_low:
            img_norm = np.clip(img, p_low, p_high)
            img_norm = (img_norm - p_low) / (p_high - p_low)
        elif img.max() > 0:
            img_norm = (img - img.min()) / (img.max() - img.min())
        else:
            img_norm = img

        return img_norm

    def create_low_range_image(self, img):
        """Create image showing only 0th to 10th percentile range."""
        if img is None:
            return None

        if img.ndim == 3 and img.shape[2] == 3:
            # RGB image - process each channel
            low_range = np.zeros_like(img)
            for i in range(3):
                ch = img[..., i]
                p0 = np.percentile(ch, 0)
                p10 = np.percentile(ch, 10)
                if p10 > p0:
                    ch_clipped = np.clip(ch, p0, p10)
                    low_range[..., i] = (ch_clipped - p0) / (p10 - p0)
            return low_range
        else:
            # Grayscale
            p0 = np.percentile(img, 0)
            p10 = np.percentile(img, 10)
            if p10 > p0:
                img_clipped = np.clip(img, p0, p10)
                return (img_clipped - p0) / (p10 - p0)
            else:
                return img

    def load_mask(self, method, source_id, file_name, segment_step=None):
        """Load segmentation mask."""
        if segment_step is None:
            segment_step = self.segment_step

        # Check cache first
        cache_key = (method, source_id, file_name, segment_step)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        try:
            mask_path = self.mask_root / method / segment_step / source_id / file_name
            if not mask_path.exists():
                # Try alternative path with 'steps' subdirectory
                mask_path = self.mask_root / method / "steps" / source_id / segment_step / file_name

            if not mask_path.exists():
                print(f"Warning: Mask not found: {mask_path}")
                return None

            # Load mask based on file extension
            if mask_path.suffix == '.npz':
                # Numpy compressed file
                data = np.load(mask_path)
                # Try common keys
                for key in ['mask', 'segmentation', 'arr_0']:
                    if key in data:
                        mask = np.squeeze(data[key]).astype(np.int32)
                        self._mask_cache[cache_key] = mask
                        return mask
                # Fallback to first array
                mask = np.squeeze(data[list(data.keys())[0]]).astype(np.int32)
                self._mask_cache[cache_key] = mask
                return mask
            else:
                # TIFF or other image format
                mask = tifffile.imread(mask_path)
                self._mask_cache[cache_key] = mask
                return mask
        except Exception as e:
            print(f"Error loading mask ({segment_step}): {e}")
            return None

    def scatter_plot(self):
        """Create scatter plot of cell count differences."""
        # Add color and marker columns to dataframe
        df_plot = self.df.copy()

        # Vectorized color mapping (much faster than apply)
        df_plot['plot_color'] = 'gray'

        # Cell colors
        df_plot.loc[(df_plot['segmentation'] == 'Cell') & (df_plot['codec'] == 'jxl_hq'), 'plot_color'] = 'darkgreen'
        df_plot.loc[(df_plot['segmentation'] == 'Cell') & (df_plot['codec'] == 'jxl_effort_3'), 'plot_color'] = 'green'
        df_plot.loc[(df_plot['segmentation'] == 'Cell') & (df_plot['codec'] == 'jxl_mq'), 'plot_color'] = 'lightgreen'
        df_plot.loc[(df_plot['segmentation'] == 'Cell') & (df_plot['codec'] == 'jxl_lq'), 'plot_color'] = 'lime'

        # Nuclei colors
        df_plot.loc[(df_plot['segmentation'] == 'Nuclei') & (df_plot['codec'] == 'jxl_hq'), 'plot_color'] = 'darkblue'
        df_plot.loc[(df_plot['segmentation'] == 'Nuclei') & (df_plot['codec'] == 'jxl_effort_3'), 'plot_color'] = 'blue'
        df_plot.loc[(df_plot['segmentation'] == 'Nuclei') & (df_plot['codec'] == 'jxl_mq'), 'plot_color'] = 'lightblue'
        df_plot.loc[(df_plot['segmentation'] == 'Nuclei') & (df_plot['codec'] == 'jxl_lq'), 'plot_color'] = 'cyan'

        # Create single scatter plot
        scatter = hv.Scatter(
            df_plot,
            kdims=['n_true'],
            vdims=['cell_count_diff', 'source_id', 'file', 'method', 'n_pred', 'iou', 'dice', 'plot_color', 'codec', 'segmentation']
        ).opts(
            color='plot_color',
            size=8,
            alpha=0.6,
            tools=['tap', 'hover'],
            hover_tooltips=[
                ('Segmentation', '@segmentation'),
                ('Codec', '@codec'),
                ('GT Count', '@n_true'),
                ('Pred Count', '@n_pred'),
                ('Difference', '@cell_count_diff'),
                ('IoU', '@iou{0.000}'),
                ('Dice', '@dice{0.000}')
            ],
            width=600,
            height=600,
            xlabel='Ground Truth Cell Count',
            ylabel='Cell Count Difference (pred - GT)',
            title='Cell Count Differences (Click points to view)',
            show_grid=True
        )

        # Add reference lines
        hline0 = hv.HLine(0).opts(color='red', line_dash='dashed', line_width=2, alpha=0.5)
        hline_pos = hv.HLine(10).opts(color='orange', line_dash='dashed', line_width=1, alpha=0.5)
        hline_neg = hv.HLine(-10).opts(color='orange', line_dash='dashed', line_width=1, alpha=0.5)

        final_plot = (scatter * hline0 * hline_pos * hline_neg)

        return final_plot

    def on_tap(self, x, y):
        """Handle tap events on scatter plot."""
        if x is None or y is None:
            return

        # Find closest point
        distances = np.sqrt((self.df['n_true'] - x)**2 + (self.df['cell_count_diff'] - y)**2)
        idx = distances.idxmin()
        self.current_index = idx
        self.current_selection = self.df.loc[idx]

        print(f"\n=== Selected Sample ===")
        print(f"Method: {self.current_selection['method']}")
        print(f"Segmentation: {self.current_selection['segmentation']}")
        print(f"Source: {self.current_selection['source_id']}")
        print(f"File: {self.current_selection['file']}")
        print(f"GT Count: {self.current_selection['n_true']}")
        print(f"Pred Count: {self.current_selection['n_pred']}")
        print(f"Difference: {self.current_selection['cell_count_diff']}")

    def view_images(self):
        """Display selected images and masks."""
        if self.current_selection is None:
            return pn.pane.Markdown("### Click on a point in the scatter plot to view images")

        row = self.current_selection
        source_id = row['source_id']
        file_name = row['file']
        method = row['method']
        seg_type = row['segmentation']  # Cell or Nuclei

        # Load images (already normalized per-channel if RGB)
        gt_zarr = self.zarr_root / self.ground_truth
        orig_img = self.load_zarr_image_raw(gt_zarr, source_id)

        comp_zarr = self.zarr_root / method
        comp_img = self.load_zarr_image_raw(comp_zarr, source_id)

        # Load BOTH cell and nuclei masks
        gt_cell_mask = self.load_mask(self.ground_truth, source_id, file_name, 'segment_cell')
        comp_cell_mask = self.load_mask(method, source_id, file_name, 'segment_cell')

        gt_nuclei_mask = self.load_mask(self.ground_truth, source_id, file_name, 'segment_nuclei')
        comp_nuclei_mask = self.load_mask(method, source_id, file_name, 'segment_nuclei')

        if orig_img is None:
            return pn.pane.Markdown("### Failed to load images")

        h, w = orig_img.shape[:2]
        bounds = (0, 0, w, h)

        # Create holoviews images
        plots = []

        # Original image
        if orig_img.ndim == 3 and orig_img.shape[2] == 3:
            # RGB image
            orig_hv = hv.RGB(orig_img, bounds=bounds).opts(
                title=f'Original ({self.ground_truth})',
                xaxis=None, yaxis=None, width=300, height=300
            )
        else:
            # Grayscale
            orig_hv = hv.Image(orig_img, bounds=bounds).opts(
                cmap='gray', title=f'Original ({self.ground_truth})',
                xaxis=None, yaxis=None, width=300, height=300
            )
        plots.append(orig_hv)

        # Compressed image
        if comp_img is not None:
            if comp_img.ndim == 3 and comp_img.shape[2] == 3:
                # RGB image
                comp_hv = hv.RGB(comp_img, bounds=bounds).opts(
                    title=f'Compressed ({method})',
                    xaxis=None, yaxis=None, width=300, height=300
                )
            else:
                # Grayscale
                comp_hv = hv.Image(comp_img, bounds=bounds).opts(
                    cmap='gray', title=f'Compressed ({method})',
                    xaxis=None, yaxis=None, width=300, height=300
                )
        else:
            comp_hv = hv.Image(np.zeros_like(orig_img), bounds=bounds).opts(
                cmap='gray', title='Compressed (N/A)',
                xaxis=None, yaxis=None, width=300, height=300
            )
        plots.append(comp_hv)

        # Low range images (0th to 10th percentile)
        # Original low range
        orig_low = self.create_low_range_image(orig_img)
        if orig_low.ndim == 3 and orig_low.shape[2] == 3:
            orig_low_hv = hv.RGB(orig_low, bounds=bounds).opts(
                title='Original (0-10th percentile)',
                xaxis=None, yaxis=None, width=300, height=300
            )
        else:
            orig_low_hv = hv.Image(orig_low, bounds=bounds).opts(
                cmap='gray', title='Original (0-10th percentile)',
                xaxis=None, yaxis=None, width=300, height=300
            )
        plots.append(orig_low_hv)

        # Compressed low range
        if comp_img is not None:
            comp_low = self.create_low_range_image(comp_img)
            if comp_low.ndim == 3 and comp_low.shape[2] == 3:
                comp_low_hv = hv.RGB(comp_low, bounds=bounds).opts(
                    title='Compressed (0-10th percentile)',
                    xaxis=None, yaxis=None, width=300, height=300
                )
            else:
                comp_low_hv = hv.Image(comp_low, bounds=bounds).opts(
                    cmap='gray', title='Compressed (0-10th percentile)',
                    xaxis=None, yaxis=None, width=300, height=300
                )
        else:
            comp_low_hv = hv.Image(np.zeros_like(orig_img), bounds=bounds).opts(
                cmap='gray', title='Compressed Low (N/A)',
                xaxis=None, yaxis=None, width=300, height=300
            )
        plots.append(comp_low_hv)

        # GT Cell mask
        if gt_cell_mask is not None:
            gt_cell_rgb = self.mask_to_rgb(gt_cell_mask)
            gt_cell_hv = hv.RGB(gt_cell_rgb, bounds=bounds).opts(
                title=f'GT Cell Mask (n={gt_cell_mask.max()})',
                xaxis=None, yaxis=None, width=300, height=300
            )
        else:
            gt_cell_hv = hv.Image(np.zeros_like(orig_img), bounds=bounds).opts(
                cmap='gray', title='GT Cell Mask (N/A)',
                xaxis=None, yaxis=None, width=300, height=300
            )
        plots.append(gt_cell_hv)

        # Pred Cell mask
        if comp_cell_mask is not None:
            comp_cell_rgb = self.mask_to_rgb(comp_cell_mask)
            comp_cell_hv = hv.RGB(comp_cell_rgb, bounds=bounds).opts(
                title=f'Pred Cell Mask (n={comp_cell_mask.max()})',
                xaxis=None, yaxis=None, width=300, height=300
            )
        else:
            comp_cell_hv = hv.Image(np.zeros_like(orig_img), bounds=bounds).opts(
                cmap='gray', title='Pred Cell Mask (N/A)',
                xaxis=None, yaxis=None, width=300, height=300
            )
        plots.append(comp_cell_hv)

        # GT Nuclei mask
        if gt_nuclei_mask is not None:
            gt_nuclei_rgb = self.mask_to_rgb(gt_nuclei_mask)
            gt_nuclei_hv = hv.RGB(gt_nuclei_rgb, bounds=bounds).opts(
                title=f'GT Nuclei Mask (n={gt_nuclei_mask.max()})',
                xaxis=None, yaxis=None, width=300, height=300
            )
        else:
            gt_nuclei_hv = hv.Image(np.zeros_like(orig_img), bounds=bounds).opts(
                cmap='gray', title='GT Nuclei Mask (N/A)',
                xaxis=None, yaxis=None, width=300, height=300
            )
        plots.append(gt_nuclei_hv)

        # Pred Nuclei mask
        if comp_nuclei_mask is not None:
            comp_nuclei_rgb = self.mask_to_rgb(comp_nuclei_mask)
            comp_nuclei_hv = hv.RGB(comp_nuclei_rgb, bounds=bounds).opts(
                title=f'Pred Nuclei Mask (n={comp_nuclei_mask.max()})',
                xaxis=None, yaxis=None, width=300, height=300
            )
        else:
            comp_nuclei_hv = hv.Image(np.zeros_like(orig_img), bounds=bounds).opts(
                cmap='gray', title='Pred Nuclei Mask (N/A)',
                xaxis=None, yaxis=None, width=300, height=300
            )
        plots.append(comp_nuclei_hv)

        # Determine which masks to use for overlay based on selected segmentation type
        if seg_type == 'Cell':
            gt_mask, comp_mask = gt_cell_mask, comp_cell_mask
        else:
            gt_mask, comp_mask = gt_nuclei_mask, comp_nuclei_mask

        # Overlay
        if gt_mask is not None and comp_mask is not None:
            overlay = np.zeros((*gt_mask.shape, 3))
            overlay[..., 1] = (gt_mask > 0).astype(float)  # Green for GT
            overlay[..., 0] = (comp_mask > 0).astype(float)  # Red for Pred
            overlay[..., 2] = (comp_mask > 0).astype(float)  # Blue for Pred (makes magenta)

            overlay_hv = hv.RGB(overlay, bounds=bounds).opts(
                title=f'{seg_type} Overlay (GT=green, Pred=magenta)',
                xaxis=None, yaxis=None, width=300, height=300
            )
        else:
            overlay_hv = hv.Image(np.zeros_like(orig_img), bounds=bounds).opts(
                cmap='gray', title='Overlay (N/A)',
                xaxis=None, yaxis=None, width=300, height=300
            )
        plots.append(overlay_hv)

        # Difference
        if gt_mask is not None and comp_mask is not None:
            diff = (comp_mask > 0).astype(int) - (gt_mask > 0).astype(int)
            diff_hv = hv.Image(diff, bounds=bounds).opts(
                cmap='RdBu_r', clim=(-1, 1), title=f'{seg_type} Diff (red=FP, blue=FN)',
                xaxis=None, yaxis=None, width=300, height=300, colorbar=False
            )
        else:
            diff_hv = hv.Image(np.zeros_like(orig_img), bounds=bounds).opts(
                cmap='gray', title='Difference (N/A)',
                xaxis=None, yaxis=None, width=300, height=300
            )
        plots.append(diff_hv)

        # Info text
        info_text = f"""
### Selected Sample Info
- **Method**: {method}
- **Segmentation**: {row['segmentation']}
- **Source**: {source_id}
- **GT Count**: {int(row['n_true'])}
- **Pred Count**: {int(row['n_pred'])}
- **Difference**: {row['cell_count_diff']:.0f}
- **IoU**: {row['iou']:.3f}
- **Dice**: {row['dice']:.3f}
"""

        # Arrange as 5 columns x 2 rows with linked axes for synchronized zooming/panning
        # Create layout and link all x_range and y_range together
        image_layout = hv.Layout(plots).cols(5).opts(
            shared_axes=True,  # Link all axes together for synchronized zoom/pan
            merge_tools=True   # Merge toolbars
        )

        layout = pn.Column(
            pn.pane.Markdown(info_text),
            pn.pane.HoloViews(image_layout),
            pn.pane.Markdown("*Overlay and difference show the selected segmentation type only. Zoom/pan is synchronized across all images.*",
                           styles={'font-style': 'italic', 'color': 'gray', 'font-size': '10px'})
        )

        return layout

    def create_dashboard(self):
        """Create the interactive dashboard."""
        scatter = self.scatter_plot()
        self.tap_stream.source = scatter

        # Dynamic image view
        @pn.depends(self.tap_stream.param.x, self.tap_stream.param.y)
        def dynamic_images(x, y):
            if x is not None and y is not None:
                self.on_tap(x, y)
                return self.view_images()
            else:
                return pn.pane.Markdown("### Click on a point in the scatter plot to view images")

        app = pn.Row(
            pn.Column(
                pn.pane.Markdown("## Cell Count Differences"),
                pn.pane.HoloViews(scatter, sizing_mode='stretch_width'),
                pn.pane.Markdown("*Click on points to view images and masks*"),
                width=700
            ),
            pn.Column(
                pn.pane.Markdown("## Selected Sample"),
                dynamic_images,
                width=1500
            )
        )

        return app


def main():
    parser = argparse.ArgumentParser(
        description='Interactive viewer for cell count differences with images and masks'
    )
    parser.add_argument('--csv', type=str, required=True,
                       help='CSV file with cell count differences')
    parser.add_argument('--mask-root', type=str, required=True,
                       help='Root directory containing segmentation masks')
    parser.add_argument('--zarr-root', type=str, default='/work/datasets/jump_target2_4plate',
                       help='Root directory containing zarr images')
    parser.add_argument('--ground-truth', type=str, default='zstd.zarr',
                       help='Ground truth method name')
    parser.add_argument('--segment-step', type=str, default='segment_cell',
                       help='Segmentation step (segment_cell or segment_nuclei)')
    parser.add_argument('--port', type=int, default=5007,
                       help='Port for the server (default: 5007)')

    args = parser.parse_args()

    # Load CSV
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    # Create viewer
    viewer = InteractiveCellCountViewer(
        df=df,
        mask_root=args.mask_root,
        zarr_root=args.zarr_root,
        ground_truth=args.ground_truth,
        segment_step=args.segment_step
    )

    app = viewer.create_dashboard()
    app.servable()

    # Serve the app
    pn.serve(app, port=args.port, show=True, title="Cell Count Viewer")


if __name__ == "__main__":
    main()
