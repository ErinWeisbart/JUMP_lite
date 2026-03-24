import os
import random
import subprocess
import time
from pathlib import Path

import numcodecs
import numpy as np
import zarr
from imagecodecs.numcodecs import Jpegxl
from tqdm import tqdm

# Register the jpegxl codec
numcodecs.register_codec(Jpegxl)


def get_dir_size(path):
    """Get the size of a directory in human-readable format."""
    result = subprocess.run(["du", "-sh", path], capture_output=True, text=True)
    return result.stdout.split()[0]


def benchmark_decompression(zarr_path, sample_size: int = 5):
    """Benchmark the decompression time for a Zarr directory, returning timings for each array."""
    root = zarr.open(zarr_path, mode="r")
    keys = list(root.array_keys())

    # If sample_size is more than available keys, use all keys
    if len(keys) < sample_size:
        sample_keys = keys
    else:
        sample_keys = keys[:sample_size]

    timings = []
    for key in sample_keys:
        start_time = time.perf_counter()
        _ = root[key][:]
        end_time = time.perf_counter()
        timings.append(end_time - start_time)
    return timings


def size_to_bytes(size_str):
    """Convert human-readable size string to bytes."""
    size_str = size_str.strip().upper()
    if size_str.endswith("K"):
        return int(float(size_str[:-1]) * 1024)
    elif size_str.endswith("M"):
        return int(float(size_str[:-1]) * 1024**2)
    elif size_str.endswith("G"):
        return int(float(size_str[:-1]) * 1024**3)
    elif size_str.endswith("T"):
        return int(float(size_str[:-1]) * 1024**4)
    else:
        return int(size_str)


def main():
    """Main function to run the benchmark and generate a LaTeX table."""
    base_dir = Path("/work/datasets/jump_lite/images/compressed/jump_target2_4plate")
    zarr_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.endswith(".zarr")]
    )
    sample_size = 200

    results = []
    for zarr_dir in tqdm(zarr_dirs, desc="Benchmarking"):
        size_str = get_dir_size(zarr_dir)
        size_bytes = size_to_bytes(size_str)

        # Get timings for each array decompression
        timings = benchmark_decompression(zarr_dir, sample_size=sample_size)

        mean_time = np.mean(timings)
        std_time = np.std(timings)

        results.append((zarr_dir.name, size_str, size_bytes, mean_time, std_time))

    # Sort results by size in descending order
    results.sort(key=lambda x: x[2], reverse=True)

    # Generate LaTeX table
    output_dir = Path("size_decompression_benchmark")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "size_decompression_benchmark.tex"
    with open(output_path, "w") as f:
        f.write(r"\begin{table}[h!]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\begin{tabular}{lrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write("Compression & Size & Time (s/array) \\\\" + "\n")
        f.write(r"\midrule" + "\n")
        for name, _, size_bytes, mean_time, std_time in results:
            size_gb = size_bytes / (1024**3)
            line = (
                f"{name.removesuffix('.zarr').replace('jpegxl_lossy', 'jxl').replace('_', '-')} & {size_gb:.2f}G & ${mean_time:.3f} \\pm {std_time:.3f}$ \\\\"
                + "\n"
            )
            f.write(line)
            print(line)
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(
            r"\caption{Benchmark of Zarr directories. Decompression time is reported as mean $\pm$ standard deviation per array.}"
            + "\n"
        )
        f.write(r"\label{tab:zarr_benchmark}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"\nLaTeX table saved to {output_path}")


main()
