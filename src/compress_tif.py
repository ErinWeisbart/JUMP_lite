from itertools import groupby
from pathlib import Path

import numcodecs
import numpy
import zarr
from imagecodecs.numcodecs import Jpegxl
from PIL import Image

input_dir = Path("/home/amunoz/projects/JUMP_core/src/output_images/")
output_dir = Path("compressed")

output_dir.mkdir(parents=True, exist_ok=True)
compression_algs = {"jpegxl": Jpegxl}
# files = [xfor x in input_dir.glob("*.tif")]

key_fn = lambda x: (*(x.name.split("__"))[:4], (x.name.split("__"))[5])

groups = {
    k: list(g)
    for k, g in groupby(sorted(input_dir.glob("*.tif"), key=key_fn), key=key_fn)
}
# %%
for name, compressor in compression_algs.items():
    numcodecs.register_codec(compressor)
    store_name = output_dir / f"{name}.zarr"
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
