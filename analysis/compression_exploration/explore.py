"""JPEG XL compression exploration: distance vs effort grid."""
import argparse, random
from pathlib import Path
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from imagecodecs import jpegxl_encode, jpegxl_decode

parser = argparse.ArgumentParser()
parser.add_argument("--hist-only", action="store_true", help="Only run histogram + peak comparison")
parser.add_argument("--image", type=str, help="Path to specific image (default: random)")
parser.add_argument("--threshold", type=int, default=-1, help="Only consider peaks above this value (default: -1)")
args = parser.parse_args()

RAW = Path("/work/datasets/jump_target2_4plate_bak/raw/")
DISTS = [1, 2, 3, 4, 5, 7, 10, 12, 15]
EFFORTS = [1, 2, 3, 4, 5, 6]

img_path = RAW / args.image if args.image else random.choice(list(RAW.glob("*.tif")))
img = np.array(Image.open(img_path))
raw_size = img.nbytes
vmin, vmax = np.percentile(img, [0.1, 99.9])
print(f"Image: {img_path.name}\nShape: {img.shape}, dtype: {img.dtype}, raw: {raw_size/1e6:.2f} MB\n")

presets = [("1_effort_8", 1.0, 8), ("2_hq", 1.0, 5), ("3_d2_e8", 2.0, 8), ("4_d2", 2.0, 5), ("5_effort_3", 1.0, 3), ("6_mq", 3.0, 5), ("7_lq", 5.0, 5), ("8_d10_default", 10.0, 5), ("9_d15", 15.0, 5)]

if not args.hist_only:
    out_dir = Path(__file__).parent / "full_res"
    out_dir.mkdir(exist_ok=True)
    norm_orig = np.clip((img.astype(float) - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(norm_orig).save(out_dir / "d00_e0.png")

    results = {}
    for d in DISTS:
        for e in EFFORTS:
            t0 = perf_counter(); enc = jpegxl_encode(img, lossless=False, distance=d, effort=e); t_enc = perf_counter() - t0
            t0 = perf_counter(); dec = jpegxl_decode(enc); t_dec = perf_counter() - t0
            ratio = raw_size / len(enc)
            results[(d, e)] = {"enc": enc, "dec": dec, "size": len(enc), "ratio": ratio, "t_enc": t_enc, "t_dec": t_dec}
            norm = np.clip((dec.astype(float) - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
            Image.fromarray(norm).save(out_dir / f"d{d:02d}_e{e}.png")
            print(f"d={d:2d} e={e}: {len(enc)/1e3:7.1f} KB ({ratio:5.1f}x) enc={t_enc*1e3:6.1f}ms dec={t_dec*1e3:5.1f}ms")

    # Plot grid
    fig, axes = plt.subplots(len(DISTS), len(EFFORTS), figsize=(len(EFFORTS)*2.5, len(DISTS)*2.5))
    for i, d in enumerate(DISTS):
        for j, e in enumerate(EFFORTS):
            r = results[(d, e)]
            ax = axes[i, j]
            ax.imshow(r["dec"], cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"{r['ratio']:.1f}x\n{r['t_enc']*1e3:.0f}ms", fontsize=8)
            ax.axis("off")
            if j == 0: ax.set_ylabel(f"d={d}", fontsize=9)
            if i == 0: ax.set_xlabel(f"e={e}", fontsize=9); ax.xaxis.set_label_position("top")

    plt.suptitle(f"{img_path.name}\nRaw: {raw_size/1e6:.2f} MB | Rows: distance | Cols: effort", fontsize=10)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "compression_grid.png", dpi=150)
    plt.show()
    print(f"\nSaved: compression_grid.png + full_res/*.png")

    # Save original presets comparison
    presets_dir = Path(__file__).parent / "presets"
    presets_dir.mkdir(exist_ok=True)
    Image.fromarray(norm_orig).save(presets_dir / "0_original.png")
    print("\nPresets:")
    for name, d, e in presets:
        t0 = perf_counter(); enc = jpegxl_encode(img, lossless=False, distance=d, effort=e); t_enc = perf_counter() - t0
        dec = jpegxl_decode(enc)
        norm = np.clip((dec.astype(float) - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(norm).save(presets_dir / f"{name}.png")
        print(f"  {name}: d={d} e={e} -> {len(enc)/1e3:.1f} KB ({raw_size/len(enc):.1f}x) enc={t_enc*1e3:.0f}ms")
    print(f"\nSaved: presets/*.png")

    # Save presets with per-image normalization
    presets_pernorm_dir = Path(__file__).parent / "presets_pernorm"
    presets_pernorm_dir.mkdir(exist_ok=True)
    vmin_o, vmax_o = np.percentile(img, [0.1, 99.9])
    norm_o = np.clip((img.astype(float) - vmin_o) / (vmax_o - vmin_o) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(norm_o).save(presets_pernorm_dir / "0_original.png")
    for name, d, e in presets:
        dec = jpegxl_decode(jpegxl_encode(img, lossless=False, distance=d, effort=e))
        vmin_i, vmax_i = np.percentile(dec, [0.1, 99.9])
        norm = np.clip((dec.astype(float) - vmin_i) / (vmax_i - vmin_i) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(norm).save(presets_pernorm_dir / f"{name}.png")
    print(f"Saved: presets_pernorm/*.png")

    # Save TIF versions (raw decoded data, no normalization)
    tif_dir = Path(__file__).parent / "presets_tif"
    tif_dir.mkdir(exist_ok=True)
    Image.fromarray(img).save(tif_dir / "0_original.tif")
    for name, d, e in presets:
        dec = jpegxl_decode(jpegxl_encode(img, lossless=False, distance=d, effort=e))
        Image.fromarray(dec).save(tif_dir / f"{name}.tif")
    print(f"Saved: presets_tif/*.tif")

# Histogram comparison plot
all_presets = [("0_original", 0, 0)] + presets
fig, axes = plt.subplots(len(all_presets), 1, figsize=(12, 2 * len(all_presets)))
img_min = int(img.min())
bins = np.arange(img_min - 0.5, img_min + 1024.5, 1)  # bin edges at half-integers, each bin captures one integer
raw_hist, _ = np.histogram(img.ravel(), bins=bins)
ymax = raw_hist.max() * 2
for i, (name, d, e) in enumerate(all_presets):
    ax = axes[i]
    ax.bar(bins[:-1], raw_hist, width=1, alpha=0.3, color="gray", label="original")
    if d > 0:
        dec = jpegxl_decode(jpegxl_encode(img, lossless=False, distance=d, effort=e))
        hist, _ = np.histogram(dec.ravel(), bins=bins)
        ax.bar(bins[:-1], hist, width=1, alpha=0.7, color="blue", label=f"d={d} e={e}")
    ax.set_ylabel(name, fontsize=9)
    ax.set_xlim(img_min, img_min + 1024)
    ax.set_ylim(0, ymax)
    ax.set_xticks([img_min, img_min + 256, img_min + 512, img_min + 768, img_min + 1024])
    if i == 0: ax.legend(fontsize=8, loc="upper right")
plt.suptitle("Intensity histograms (raw 16-bit)", fontsize=11)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "histogram_comparison.png", dpi=150)
plt.show()
print(f"Saved: histogram_comparison.png")

# Paired plot: histogram + peak pixels mask + compressed image
fig, axes = plt.subplots(len(all_presets), 3, figsize=(14, 2.5 * len(all_presets)), gridspec_kw={"width_ratios": [3, 1, 1], "wspace": 0.05, "hspace": 0.4})
peak_colors = ["green", "blue", "red"]
for i, (name, d, e) in enumerate(all_presets):
    ax_hist, ax_mask, ax_img = axes[i]
    # Histogram
    ax_hist.bar(bins[:-1], raw_hist, width=1, alpha=0.3, color="gray", label="original")
    if d > 0:
        t0 = perf_counter(); enc = jpegxl_encode(img, lossless=False, distance=d, effort=e); t_enc = perf_counter() - t0
        t0 = perf_counter(); dec = jpegxl_decode(enc); t_dec = perf_counter() - t0
        ratio = raw_size / len(enc)
        hist, _ = np.histogram(dec.ravel(), bins=bins)
        ax_hist.bar(bins[:-1], hist, width=1, alpha=0.7, color="blue", label=f"d={d} e={e}")
        ax_hist.set_title(f"{ratio:.1f}x | enc:{t_enc*1e3:.0f}ms | dec:{t_dec*1e3:.0f}ms", fontsize=8)
    else:
        dec = img
        hist = raw_hist
    # Top 3 peaks (above threshold)
    hist_masked = hist.copy()
    bin_centers = bins[:-1] + 0.5
    hist_masked[bin_centers <= args.threshold] = 0
    top3_idx = np.argsort(hist_masked)[-3:][::-1]
    peak_vals = [int(bins[idx] + 0.5) for idx in top3_idx]
    for pv, c in zip(peak_vals, peak_colors):
        ax_hist.axvline(pv, color=c, lw=1, ls="--")
    ax_hist.set_ylabel(name, fontsize=9)
    ax_hist.set_xlim(img_min, img_min + 1024)
    ax_hist.set_ylim(0, ymax)
    ax_hist.set_xticks([img_min, img_min + 256, img_min + 512, img_min + 768, img_min + 1024])
    if i == 0: ax_hist.legend(fontsize=7, loc="upper right")
    # Peak pixels mask (RGB)
    mask = np.zeros((*dec.shape, 3), dtype=np.uint8)
    mask[dec == peak_vals[0]] = [0, 255, 0]    # green
    mask[dec == peak_vals[1]] = [0, 0, 255]    # blue
    mask[dec == peak_vals[2]] = [255, 0, 0]    # red
    ax_mask.imshow(mask)
    ax_mask.set_title(f"peaks: {peak_vals[0]}, {peak_vals[1]}, {peak_vals[2]}", fontsize=8)
    ax_mask.axis("off")
    # Compressed image
    vmin_dec, vmax_dec = np.percentile(dec, [0.1, 99.9])
    ax_img.imshow(dec, cmap="gray", vmin=vmin_dec, vmax=vmax_dec)
    ax_img.set_title(name, fontsize=8)
    ax_img.axis("off")
plt.suptitle("Histograms + peak value pixels", fontsize=11)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "histogram_peak_comparison.png", dpi=150)
plt.show()
print(f"Saved: histogram_peak_comparison.png")
