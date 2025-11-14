"""
Script to compress pulled tifs, grouping them by their site (bringing different channels into the same group).

Things to try:
- Group also by sites
- Try a more complex filter combination, such as delta and LZMA2
- Support multiple processes encoding/decoding
- Test GPU encoding
"""

import lzma
from itertools import groupby
from joblib import Parallel, delayed
from pathlib import Path
from pprint import pprint
from shutil import rmtree

from skimage.metrics import structural_similarity as ssim
from time import perf_counter
from tqdm import tqdm
import numpy
import zarr



try:
    from imagecodecs.numcodecs import Brotli, Jpegxl
    import numcodecs
    # Register the codecs manually
    numcodecs.register_codec(Brotli)
    numcodecs.register_codec(Jpegxl)
    IMAGECODECS_AVAILABLE = True
except (ImportError, AttributeError):
    IMAGECODECS_AVAILABLE = False
    print("Warning: imagecodecs.numcodecs not available, skipping JpegXL and Brotli")
from PIL import Image
from zarr.codecs import BloscCodec


def compress_single_group(key, items, store_name, compressor, zarr_format):
    """
    Compress a single image group (site).
    """
    site_name = "__".join(key)
    nchannels = len(items)
    example_arr = numpy.array(Image.open(items[0]))
    shape = example_arr.shape
    dtype = example_arr.dtype

    store = zarr.storage.LocalStore(store_name)

    arr = zarr.create_array(
        store=store,
        name=site_name,
        shape=(nchannels, *shape),
        chunks=(nchannels, *shape),
        dtype=dtype,
        compressors=compressor,
        zarr_format=zarr_format,
    )
    tmp_arr = numpy.zeros((nchannels, *shape))
    for i, img_path in enumerate(items):
        tmp_arr[i] = numpy.array(Image.open(img_path))

    arr[:] = tmp_arr


def compress_tif(name, compressor, output_dir, groups, overwrite=False, n_jobs_inner=16):
    """
    Compress tifs using different algorithms and record time taken and resulting file size.
    Uses parallel processing for group compression.

    Args:
        n_jobs_inner: Number of parallel jobs for group compression within each codec.
                      Default 16 to avoid over-parallelization when running multiple codecs.
    """
    store_name = Path(output_dir) / f"{name}.zarr"
    if store_name.exists():  # To overwrite
        if overwrite:
            rmtree(store_name)
        else:
            print(f"Skipping {name}")
            return {name: 0}

    t_start = perf_counter()

    # The API for codecs changed with Zarr 3
    # https://github.com/cgohlke/imagecodecs/issues/123
    zarr_format = 3
    if not isinstance(compressor, zarr.codecs.blosc.BloscCodec):
        zarr_format = 2

    # Compress groups in parallel with limited parallelism to avoid thrashing
    Parallel(n_jobs=n_jobs_inner, prefer="threads")(
        delayed(compress_single_group)(key, items, store_name, compressor, zarr_format)
        for key, items in tqdm(groups.items(), total=len(groups.keys()), desc=name, leave=False)
    )

    return {name: perf_counter() - t_start}



input_dir = Path("/work/datasets/jump_toy/raw")
output_dir = input_dir.parent

print("Input dir:", input_dir)
print("Output dir:", output_dir)
overwrite = True

output_dir.mkdir(parents=True, exist_ok=True)


filters = [
    dict(id=lzma.FILTER_DELTA, dist=9),
    dict(id=lzma.FILTER_LZMA2, preset=9),
]
compressing_algs = {
    # "lz4": {"clevel": 9}, # Too similar to lz4hc, but usually worse
    "lz4hc": {"clevel": 9},
    "zstd": {"clevel": 9},
    "zlib": {"clevel": 9},
}
compressors_blosc = {
    k: BloscCodec(cname=k, shuffle="bitshuffle", **v)
    for k, v in compressing_algs.items()
}

compressors = {
    **compressors_blosc,
}

# Add imagecodecs compressors if available
if IMAGECODECS_AVAILABLE:
    compressors.update({
        # "brotli": Brotli(level=11),
        "jpegxl_lossless": Jpegxl(lossless=True, level=9),
        "jpegxl_lossy_hq": Jpegxl(lossless=False, distance=1.0),
        "jpegxl_lossy_hmq": Jpegxl(lossless=False, distance=2.0),
        "jpegxl_lossy_mq": Jpegxl(lossless=False, distance=3.0),
        "jpegxl_lossy_mlq": Jpegxl(lossless=False, distance=4.0),
        "jpegxl_lossy_lq": Jpegxl(lossless=False, distance=5.0),
        "jpegxl_lossy_effort_1": Jpegxl(lossless=False, distance=1.0, effort=1),
        "jpegxl_lossy_effort_3": Jpegxl(lossless=False, distance=1.0, effort=3),
        "jpegxl_lossy_effort_5": Jpegxl(lossless=False, distance=1.0, effort=5),
        "jpegxl_lossy_decompression_1": Jpegxl(lossless=False, distance=1.0, decodingspeed=1),
        "jpegxl_lossy_decompression_3": Jpegxl(lossless=False, distance=1.0, decodingspeed=3),
        "jpegxl_lossy_decompression_5": Jpegxl(lossless=False, distance=1.0, decodingspeed=5),
    })
# for v in {
#     "preset": {"preset": 9},
#     "filters": {"filters": filters, "format": lzma.FORMAT_RAW},
# }.values():
#     compressors[k]append(LZMA(**v))


# %% Group files based on their name

key_fn = lambda x: (*(x.name.split("__"))[:4], (x.name.split("__"))[5])

groups = {
    k: list(g)
    for k, g in groupby(sorted(input_dir.glob("*.tif"), key=key_fn), key=key_fn)
}

# Subsample groups for testing
# groups = dict(list(groups.items())[:2]) 

# %% Run compression and record time
# Use limited parallelism: outer level = codecs (12), inner level = groups (16)
# This prevents over-saturation: 12 * 16 = 192 max threads (matches CPU count)
n_jobs_codecs = 32  # Number of codecs to compress in parallel
n_jobs_groups = 16  # Number of groups to compress in parallel within each codec

compression_time = Parallel(n_jobs=n_jobs_codecs, prefer="threads")(
    delayed(compress_tif)(name, compressor, output_dir, groups, overwrite, n_jobs_groups)
    for name, compressor in compressors.items()
)
compression_time = {k: v for d in compression_time for k, v in d.items()}

# %%
decompression_time = {}
for name in compressors.keys():
    # numcodecs.register_codec(compressor)
    store_name = Path(output_dir) / f"{name}.zarr"

    store = zarr.storage.LocalStore(store_name)

    # TODO add decompression test
    t_start = perf_counter()

    root = zarr.group(store)
    for k in root.keys():
        tmp = root[k][:]

    duration = perf_counter() - t_start
    decompression_time[name] = duration

print("Decompression time (milliseconds)")
pprint(
    {k: round(v * 1000, 1) for k, v in sorted(decompression_time.items(), key=lambda x: x[1])},
    sort_dicts=False,
)

"""
Decompression time (seconds)
{'lz4hc': 1.33,
 'lz4': 1.47,
 'zstd': 1.95,
 'zlib': 4.84,
 'brotli': 8.89,
 'jpegxl': 23.93}


Decompression time (Milliseconds)
{'lz4hc': 2194.9,
 'zstd': 2992.9,
 'zlib': 4841.3,
 'jpegxl_lossy_hq': 9817.7,
 'jpegxl_lossy_mq': 10320.9,
 'jpegxl_lossy_lq': 22086.8,
 'jpegxl_lossless': 37673.3}
Filesize (fraction of raw)
{'jpegxl_lossy_lq': 0.007,
 'jpegxl_lossy_mq': 0.012,
 'jpegxl_lossy_hq': 0.031,
 'jpegxl_lossless': 0.481,
 'zstd': 0.595,
 'zlib': 0.607,
 'lz4hc': 0.628,
 'raw': 1.0}
"""
# %%
filesize = {}

for name in (*compressors.keys(), "raw"):
    if name != "raw":
        name = f"{name}.zarr"
    filesize[name] = sum(file.stat().st_size for file in (output_dir / name).rglob("*"))
print("Filesize (fraction of raw)")
max_val = max([x for x in filesize.values()])
pprint(
    {
        Path(k).stem: round(v / max_val, 3)
        for k, v in sorted(filesize.items(), key=lambda x: x[1])
    },
    sort_dicts=False,
)
"""
Filesize (fraction of raw)
{'jpegxl': 0.46,
 'zstd': 0.57,
 'zlib': 0.58,
 'brotli': 0.59,
 'lz4hc': 0.6,
 'lz4': 0.63,
 'raw': 1.0}
"""

# %% Save compression metrics to CSV and JSON for easy merging with quality metrics
import pandas as pd
import json

# Create results directory
results_dir = output_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Prepare compression metrics data
raw_size = filesize.get("raw", 0)
compression_metrics = []

for codec_name in compressors.keys():
    codec_filesize = filesize.get(f"{codec_name}.zarr", 0)
    compression_metrics.append({
        'codec': codec_name,
        'compression_time_sec': compression_time.get(codec_name, 0),
        'decompression_time_sec': decompression_time.get(codec_name, 0),
        'filesize_bytes': codec_filesize,
        'filesize_ratio': codec_filesize / raw_size if raw_size > 0 else 0,
        'compression_ratio': raw_size / codec_filesize if codec_filesize > 0 else 0,
    })

# Convert to DataFrame
df_compression = pd.DataFrame(compression_metrics)

# Sort by compression ratio (best first)
df_compression = df_compression.sort_values('filesize_ratio')

# Save to CSV
csv_path = results_dir / "compression_metrics.csv"
df_compression.to_csv(csv_path, index=False)
print(f"\nCompression metrics saved to {csv_path}")

# Save to JSON
json_path = results_dir / "compression_metrics.json"
df_compression.to_json(json_path, orient='records', indent=2)
print(f"Compression metrics saved to {json_path}")

# Print summary table
print("\nCompression Metrics Summary:")
print(df_compression.to_string(index=False))


# compare compression quality

# build raw array datastore
original_data = {}
for key, items in tqdm(groups.items(), total=len(groups.keys()), desc=name):
    
    site_name = "__".join(key)
    nchannels = len(items)
    example_arr = numpy.array(Image.open(items[0]))
    shape = example_arr.shape
    dtype = example_arr.dtype

    tmp_arr = numpy.zeros((nchannels, *shape))
    for i, img_path in enumerate(items):
        tmp_arr[i] = numpy.array(Image.open(img_path))
    
    original_data[site_name] = tmp_arr

#import pdb; pdb.set_trace()


# Compute SSIM for each compression and visualize it

for key in original_data.keys():
    org_arr = original_data[key]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=len(compressors), ncols=3, figsize=(18, 6))

    for i, name in enumerate(compressors.keys()):
        store_name = Path(output_dir) / f"{name}.zarr"
        com_arr = zarr.open(store_name, mode='r')[key][:]

        ssim_index = ssim(org_arr, com_arr, data_range=com_arr.max() - com_arr.min(), multichannel=True, channel_axis=0)
        print(f"Compression: {name}, Site: {key}, SSIM: {ssim_index}")

        # Make a visual comparison, show each channel side by side
        axes[i,0].imshow(org_arr[0,:,:], cmap='gray')
        axes[i,0].set_title(f"Original - Site: {key}")
        axes[i,1].imshow(com_arr[0,:,:], cmap='gray')
        axes[i,1].set_title(f"Name: {name}, SSIM: {ssim_index:.4f}  ")
        axes[i,2].imshow(org_arr[0,:,:] - com_arr[0,:,:], cmap='gray')
        axes[i,2].set_title(f"Difference ")
    plt.tight_layout()
    plt.savefig(f"comparison_{name}_{key}.png", dpi=300)
    break