#!/usr/bin/env jupyter
"""CLI tool to featurize a dataset using a specific deep learning model deployed via Nahual."""

from functools import partial
from pathlib import Path
from time import perf_counter, strftime

import numcodecs
from aliby.io.dataset import dispatch_dataset
from aliby.pipe import run_pipeline_and_post
from imagecodecs.numcodecs import Jpegxl
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

# Register the codecs manually
numcodecs.register_codec(Jpegxl)

# dataset = "jump_target2_subset_BR00121438"
dataset = "jump_target2_4plate"
# model_name = "dinov2"  # dinov2 dinov3
datasets_path = Path(f"/work/datasets/{dataset}")
compression_paths = [x for x in datasets_path.glob("*/") if x.name != "raw"]

# model_name [tile_size, selected_channels, address]
model_params = {
    "dinov2": [420, [0, 1, 2]],
    # "vit": [256, [0, 1, 2, 3, 4, 5]],  # openphenom
    # "dinov3": [420, [0, 1, 2]],
    # "subcell": [420, [0, 1, 2]],
    # "deepprofiler": [420, [0, 1, 2]],
    # "scdino": [420, [0, 1, 2]],
}
model_params = {
    model_name: [
        *v,
        i % 4,
        f"ipc:///tmp/{model_name}.ipc",
    ]
    for i, (model_name, v) in enumerate(model_params.items())
}


def process_input_path(
    input_path: str,
    output_path: str,
    tile_size: int = 420,
    selected_channels: tuple[int] = (0, 1, 2),
    device: int = 0,
    address: str = "ipc:///tmp/dinov2.ipc",
):
    fluo_base_config = {
        "input_path": input_path,
        "image_kwargs": {
            "capture_order": "CYX",
            # "regex": ".*(r.+c.+)f([0-9][0-9])p01-rgb.tiff",
            # "input_dimensions": "YXC",
        },
        "ntps": 1,
        "tile": {
            "kind": "crop",
            "tile_size": 420,
            "calculate_drift": False,
        },
    }
    embed_params = dict(
        address=address,
        setup_params=dict(
            repo_or_dir="facebookresearch/dinov2",
            model_name="dinov2_vits14_lc",
            device=device,
        ),
        selected_channels=selected_channels,
    )
    base_pipeline = {
        "io": {**fluo_base_config},
        "steps": dict(
            tile=dict(
                **fluo_base_config["tile"],
                **dict(
                    image_kwargs=dict(
                        source=input_path,
                        **fluo_base_config["image_kwargs"],
                    )
                ),
            ),
            nahual_embed_dinov2=embed_params,
        ),
        "passed_data": dict(nahual_embed_dinov2=[("pixels", "tile", "data")]),
        "save": (),
        "save_interval": 1,
    }

    # try:
    result, _ = run_pipeline_and_post(
        pipeline=base_pipeline,
        img_source=input_path,
        output_path=output_path,
        fov=input_path.path,
        overwrite=False,
    )
    # except Exception as e:
    #     print(f"Error: {e}")


# %%
dsets = list(
    map(partial(dispatch_dataset, is_zarr=True, is_monozarr=True), compression_paths)
)

# %%
for model_name, v in model_params.items():
    for compression_dir, dset in tqdm(zip(compression_paths, dsets), total=len(dsets)):
        input_paths = list(dset.get_position_ids().values())
        assert len(input_paths), "No files found in input dataset"

        if __name__ == "__main__":  # Add logging
            timestamp = strftime("%s%m%d%H%M")
            output_path = (
                Path("/work/datasets/aliby_output")
                / model_name
                / dataset
                / compression_dir.name
            )

            logger.remove()
            logger.add(output_path / f"{timestamp}_{dataset}.log")
            # shutil.copy(__file__, output_path / f"{timestamp}_script.py")

            # if False:
            #     result = Parallel(30)(delayed(process_input_path)(x) for x in input_paths)
            # else:
            #     from tqdm import tqdm
            # t0 = perf_counter()
        result = [
            process_input_path(input_path, output_path, *v)
            for input_path in tqdm(input_paths)
        ]
        # print(f"Processing took {perf_counter() - t0} seconds")
