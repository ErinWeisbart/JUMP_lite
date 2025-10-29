import lzma
from itertools import groupby
from pathlib import Path
from shutil import rmtree
from time.time import perf_counter

import numpy
import zarr
from imagecodecs.numcodecs import Brotli, Jpegxl
from PIL import Image
from zarr.codecs import BloscCodec

input_dir = Path("/home/amunoz/projects/JUMP_core/src/images/raw")
output_dir = Path("./images")

overwrite = True

output_dir.mkdir(parents=True, exist_ok=True)


filters = [
    dict(id=lzma.FILTER_DELTA, dist=9),
    dict(id=lzma.FILTER_LZMA2, preset=9),
    # dict(id=lzma.FILTER_DELTA, dist=2),
    # dict(id=lzma.FILTER_LZMA2, preset=9),
]

# compression_algs = {"jpegxl": Jpegxl}
compressing_algs = {
    # "lz4": {"clevel": 9},
    "lz4hc": {"clevel": 9},
    "zstd": {"clevel": 9},
    "zlib": {"clevel": 9},
}
compressors_blosc = {
    k: BloscCodec(cname=k, shuffle="bitshuffle", **v)
    for k, v in compressing_algs.items()
}

compressors = {
    "brotli": Brotli(level=11),
    "jpegxl": Jpegxl(),
    **compressors_blosc,
}
# imagecodecs_compresso  # rs = [
#     # Delta(shape=test.shape, dtype=test.dtype, axis=1, dist=5),
#     Brotli(level=11),
# ]
# for v in {
#     "preset": {"preset": 9},
#     "filters": {"filters": filters, "format": lzma.FORMAT_RAW},
# }.values():
#     compressors[k]append(LZMA(**v))


# %%

key_fn = lambda x: (*(x.name.split("__"))[:4], (x.name.split("__"))[5])

groups = {
    k: list(g)
    for k, g in groupby(sorted(input_dir.glob("*.tif"), key=key_fn), key=key_fn)
}
# %%

compression_time = {}
decompression_time = {}
for name, compressor in compressors.items():
    # numcodecs.register_codec(compressor)
    store_name = Path(output_dir) / f"{name}.zarr"
    if store_name.exists():  # To overwrite
        if overwrite:
            rmtree(store_name)
        else:
            print(f"Skipping {name}")
            continue

    t_start = perf_counter()
    store = zarr.storage.LocalStore(store_name)
    zarr_format = 3
    if not isinstance(compressor, zarr.codecs.blosc.BloscCodec):
        zarr_format = 2
    root = zarr.create_group(store=store, zarr_format=zarr_format)
    for key, items in groups.items():
        site_name = "__".join(key)
        nchannels = len(items)
        example_arr = numpy.array(Image.open(items[0]))
        shape = example_arr.shape
        dtype = example_arr.dtype

        # The API for codecs changed with Zarr 3
        # https://github.com/cgohlke/imagecodecs/issues/123
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

    compression_time[name] = perf_counter() - t_start

    # TODO add decompression test

    # TODO Add size
