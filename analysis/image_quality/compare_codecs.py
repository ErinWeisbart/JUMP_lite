"""
Compare image quality metrics between lossy codecs and zstd reference.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import zarr
from lpips import LPIPS
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

# Register imagecodecs numcodecs for JpegXL support
try:
    from imagecodecs.numcodecs import Brotli, Jpegxl
    import numcodecs
    numcodecs.register_codec(Brotli)
    numcodecs.register_codec(Jpegxl)
except (ImportError, AttributeError) as e:
    print(f"Warning: imagecodecs.numcodecs not available: {e}")


def setup_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def open_zarr_store(zarr_path: Path):
    """Open a zarr store and return the root group."""
    store = zarr.storage.LocalStore(zarr_path)
    return zarr.group(store)


def normalize_image(img: np.ndarray) -> torch.Tensor:
    """Normalize image to [0, 1] range."""
    if img.dtype == np.uint16:
        img_float = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img_float = img.astype(np.float32) / 255.0
    else:
        img_float = img.astype(np.float32)
    return torch.from_numpy(img_float).unsqueeze(0)


def compute_metrics(original, compressed, device, psnr_metric, ssim_metric, lpips_metric=None):
    """Compute PSNR, SSIM, and optionally LPIPS for a pair of images."""
    orig_tensor = normalize_image(original).to(device)
    comp_tensor = normalize_image(compressed).to(device)

    psnr_value = psnr_metric(comp_tensor, orig_tensor).item()
    ssim_value = ssim_metric(comp_tensor, orig_tensor).item()

    result = {'psnr': psnr_value, 'ssim': ssim_value}

    # LPIPS per channel (repeat to 3 channels) - optional
    if lpips_metric is not None:
        lpips_values = []
        for c in range(orig_tensor.shape[1]):
            orig_rgb = orig_tensor[:, c:c+1, :, :].repeat(1, 3, 1, 1)
            comp_rgb = comp_tensor[:, c:c+1, :, :].repeat(1, 3, 1, 1)
            lpips_values.append(lpips_metric(comp_rgb, orig_rgb).item())
        result['lpips'] = np.mean(lpips_values)
    else:
        result['lpips'] = np.nan

    return result


def main():
    import argparse
    import random
    parser = argparse.ArgumentParser(description="Compare image quality metrics between lossy codecs and zstd reference")
    parser.add_argument("--figures-only", action="store_true", help="Only generate figures from existing quality_metrics.csv")
    parser.add_argument("--data-dir", type=Path, default=Path("/work/datasets/jump_target2_4plate"), help="Directory containing zarr files")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of sites to sample (default: use all sites)")
    parser.add_argument("--skip-lpips", action="store_true", help="Skip LPIPS computation (10x faster, SSIM usually sufficient)")
    parser.add_argument("--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"],
                        help="LPIPS network (alex=fast, vgg=accurate, squeeze=fastest)")
    args = parser.parse_args()

    data_dir = args.data_dir
    reference_codec = "zstd"

    # Figures-only mode: load existing results and regenerate plots
    if args.figures_only:
        csv_path = data_dir / "quality_metrics.csv"
        if not csv_path.exists():
            print(f"Error: {csv_path} not found. Run evaluation first.")
            return
        print(f"Loading existing results from {csv_path}")
        df = pd.read_csv(csv_path)
        generate_violin_plots(df, data_dir)
        return

    # Find all codecs
    codec_dirs = sorted([d for d in data_dir.glob("*.zarr") if d.is_dir()])
    codec_names = [d.stem for d in codec_dirs]

    print(f"Found codecs: {codec_names}")
    print(f"Reference: {reference_codec}")

    # Setup device and metrics
    device = setup_device()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # LPIPS is optional (slow)
    if args.skip_lpips:
        lpips_metric = None
        print("Skipping LPIPS computation (much faster!)")
    else:
        lpips_metric = LPIPS(net=args.lpips_net).to(device)
        print(f"Using LPIPS with '{args.lpips_net}' network")

    # Open reference store (lazy - no data loaded yet)
    print(f"\nOpening reference ({reference_codec})...")
    ref_path = data_dir / f"{reference_codec}.zarr"
    ref_store = open_zarr_store(ref_path)
    site_names = list(ref_store.keys())
    print(f"Found {len(site_names)} total sites")

    # Sample sites if requested
    if args.n_samples and args.n_samples < len(site_names):
        site_names = random.sample(site_names, args.n_samples)
        print(f"Sampling {args.n_samples} sites for evaluation")
    else:
        print(f"Evaluating all {len(site_names)} sites")

    # Open all codec stores upfront
    print("\nOpening all codec stores...")
    codec_stores = {}
    codecs_to_compare = [c for c in codec_names if c != reference_codec]
    for codec_name in codecs_to_compare:
        codec_path = data_dir / f"{codec_name}.zarr"
        codec_stores[codec_name] = open_zarr_store(codec_path)
        print(f"  Opened {codec_name}")

    # Process all codecs per site (cache reference, avoid reloading)
    import time
    print(f"\nProcessing {len(site_names)} sites × {len(codecs_to_compare)} codecs...")
    print(f"Total comparisons: {len(site_names) * len(codecs_to_compare)}")
    results = []

    start_time = time.time()
    for i, site_name in enumerate(tqdm(site_names, desc="Evaluating sites")):
        # Load reference once per site
        try:
            original = ref_store[site_name][:]
        except KeyError:
            continue

        # Compare against all codecs
        for codec_name in codecs_to_compare:
            codec_store = codec_stores[codec_name]

            # Load compressed version
            try:
                compressed = codec_store[site_name][:]
            except KeyError:
                continue

            # Compute metrics
            metrics = compute_metrics(
                original, compressed, device,
                psnr_metric, ssim_metric, lpips_metric
            )

            results.append({
                'site_name': site_name,
                'codec': codec_name,
                **metrics,
            })

        # Print timing estimate every 10 sites
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(site_names) - i - 1) / rate
            tqdm.write(f"  [{i+1}/{len(site_names)}] Rate: {rate:.1f} sites/sec, ETA: {remaining/60:.1f} min")

    df = pd.DataFrame(results)

    # Print summary
    print("\n" + "="*80)
    print(f"Quality Comparison vs {reference_codec} (reference)")
    print("="*80)

    summary = df.groupby('codec')[['psnr', 'ssim', 'lpips']].agg(['mean', 'std'])

    for codec in codecs_to_compare:
        codec_df = df[df['codec'] == codec]
        print(f"\n{codec}:")
        print(f"  PSNR:  {codec_df['psnr'].mean():.2f} ± {codec_df['psnr'].std():.2f} dB")
        print(f"  SSIM:  {codec_df['ssim'].mean():.4f} ± {codec_df['ssim'].std():.4f}")
        if not codec_df['lpips'].isna().all():
            print(f"  LPIPS: {codec_df['lpips'].mean():.4f} ± {codec_df['lpips'].std():.4f}")
        else:
            print(f"  LPIPS: (skipped)")

    print("\n" + "="*80)
    print("Interpretation: PSNR/SSIM higher=better, LPIPS lower=better")
    print("="*80)

    # Save results
    output_path = data_dir / "quality_metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

    # Generate violin plots
    generate_violin_plots(df, data_dir)


def generate_violin_plots(df: pd.DataFrame, output_dir: Path):
    """Generate violin plots showing distribution of metrics for each codec."""
    # Clean up codec names for display
    df = df.copy()
    df['codec_display'] = df['codec'].str.replace('jpegxl_lossy_', 'jxl_')

    # Define codec order (highest to lowest quality)
    codec_order = ['jxl_hq', 'jxl_effort_3', 'jxl_d2_e8', 'jxl_mq', 'jxl_lq', 'jxl_d10']
    # Filter to only codecs present in the data
    available_codecs = df['codec_display'].unique()
    codec_order = [c for c in codec_order if c in available_codecs]
    df_filtered = df[df['codec_display'].isin(codec_order)]

    # Stats for tick labels
    codec_stats = df_filtered.groupby('codec_display').agg(
        n_total=('psnr', 'count'),
    )
    codec_labels = {
        codec: f"{codec}\nn={int(codec_stats.loc[codec, 'n_total'])}"
        for codec in codec_order if codec in codec_stats.index
    }
    label_order = [c for c in codec_order if c in codec_stats.index]

    # Check if LPIPS data exists
    has_lpips = not df_filtered['lpips'].isna().all()

    # Create figure with 2 or 3 subplots depending on LPIPS availability
    n_plots = 3 if has_lpips else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 4.5, 5))
    if n_plots == 2:
        axes = list(axes)  # Make it subscriptable

    metrics = [('psnr', 'PSNR (dB) ↑'), ('ssim', 'SSIM ↑')]
    if has_lpips:
        metrics.append(('lpips', 'LPIPS ↓'))

    for ax, (metric, label) in zip(axes, metrics):
        sns.violinplot(
            data=df_filtered,
            x='codec_display',
            y=metric,
            order=label_order,
            palette='viridis',
            inner='box',
            cut=0,
            ax=ax
        )

        ax.set_xlabel('Codec', fontsize=14, fontweight='bold')
        ax.set_ylabel(label, fontsize=14, fontweight='bold')
        ax.set_title(f'{metric.upper()} Distribution', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)

        ax.set_xticks(range(len(label_order)))
        ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=12)

        # Set SSIM y-axis range to 0.5 to 1.05
        if metric == 'ssim':
            ax.set_ylim(0.5, 1.05)

    title = 'Image Quality Metrics: JPEG-XL Lossy vs ZSTD Reference'
    if not has_lpips:
        title += ' (LPIPS skipped)'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = output_dir / "quality_metrics_violin.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Violin plot saved to: {plot_path}")

    # Generate separate SSIM plot
    generate_ssim_plot(df, output_dir)


def generate_ssim_plot(df: pd.DataFrame, output_dir: Path):
    """Generate a separate violin plot for SSIM only."""
    df = df.copy()
    df['codec_display'] = df['codec'].str.replace('jpegxl_lossy_', 'jxl_')

    # Define codec order (highest to lowest quality)
    codec_order = ['jxl_hq', 'jxl_effort_3', 'jxl_d2_e8', 'jxl_mq', 'jxl_lq', 'jxl_d10']
    # Filter to only codecs present in the data
    available_codecs = df['codec_display'].unique()
    codec_order = [c for c in codec_order if c in available_codecs]
    df_filtered = df[df['codec_display'].isin(codec_order)]

    # Nice display names (use codec name as-is if not in predefined mapping)
    codec_labels = {
        'jxl_hq': 'High',
        'jxl_effort_3': 'Effort 3',
        'jxl_d2_e8': 'D2 E8',
        'jxl_mq': 'Medium',
        'jxl_lq': 'Low',
        'jxl_d10': 'D10'
    }
    # Add any missing codecs with their display name
    for codec in codec_order:
        if codec not in codec_labels:
            codec_labels[codec] = codec.upper()

    label_order = [c for c in codec_order if c in df_filtered['codec_display'].unique()]

    fig, ax = plt.subplots(figsize=(7, 7))

    sns.violinplot(
        data=df_filtered,
        x='codec_display',
        y='ssim',
        order=label_order,
        palette='viridis',
        inner='box',
        cut=0,
        ax=ax
    )

    ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
    ax.set_ylabel('SSIM', fontsize=24, fontweight='bold')
    ax.set_title('SSIM - Image Similarity', fontsize=26, fontweight='bold')
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20)
    ax.set_ylim(0.9, 1.01)

    plt.tight_layout()
    plot_path = output_dir / "ssim_violin.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SSIM plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
