import lzma
from itertools import groupby
from pathlib import Path

import numcodecs
import numpy
import zarr
from imagecodecs.numcodecs import Brotli, Jpegxl
from numcodecs import LZMA, Blosc
from PIL import Image

input_dir = Path("/home/amunoz/projects/JUMP_core/src/images/raw")
output_dir = Path("./images")

output_dir.mkdir(parents=True, exist_ok=True)


filters = [
    dict(id=lzma.FILTER_DELTA, dist=9),
    dict(id=lzma.FILTER_LZMA2, preset=9),
    # dict(id=lzma.FILTER_DELTA, dist=2),
    # dict(id=lzma.FILTER_LZMA2, preset=9),
]

# compression_algs = {"jpegxl": Jpegxl}
compressing_algs = {
    "lz4hc": {"clevel": 9},
    "zstd": {"clevel": 9},
}
compressors_blosc = {
    k: Blosc(cname=k, shuffle=-1, **v) for k, v in compressing_algs.items()
}

compressors = {
    "brotli": Brotli(level=11),
    "jpegxl": Jpegxl,
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
for name, compressor in compressors.items():
    numcodecs.register_codec(compressor)
    store_name = Path(output_dir) / f"{name}.zarr"
    if store_name.exists():
        print(f"Skipping {name}")
        continue
    store = zarr.storage.LocalStore(store_name)
    root = zarr.create_group(store=store)
    for key, items in groups.items():
        site_name = "__".join(key)
        nchannels = len(items)
        example_arr = numpy.array(Image.open(items[0]))
        shape = example_arr.shape
        dtype = example_arr.dtype
        root.create_array(
            name=site_name,
            shape=(nchannels, *shape),
            chunks=(nchannels, *shape),
            dtype=dtype,
        )

        for i, img_path in enumerate(items):
            img = numpy.array(Image.open(img_path))
            root[site_name][i] = img

            # if site_name not in root:
            #     root.create_group(site_name)
        # arr = zarr.array(img, chunks=img.shape)
        # root[site_name][i] = img

# from itertools import groupby
