from functools import partial

import polars as pl
from joblib import Parallel, delayed
from jump_portrait.fetch import get_item_location_metadata, get_jump_image_batch
from pooch import retrieve

# Pull JCP ids
crispr = (
    pl.read_csv(
        retrieve(
            "https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/crispr.csv.gz",
            known_hash="55e36e6802c6fc5f8e5d5258554368d64601f1847205e0fceb28a2c246c8d1ed",
        ),
    )
    .select(pl.col("Metadata_JCP2022"))
    .with_columns(dataset=pl.lit("orf"))
)
orf = (
    pl.scan_csv(
        retrieve(
            "https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/orf.csv.gz",
            known_hash="9c7ec4b0fa460a3a30f270a15f11b5e85cef9dd105c8a0ab8ab50f6cc98894b8",
        ),
    )
    .select(pl.col("Metadata_JCP2022"))
    .with_columns(dataset=pl.lit("orf"))
    .collect()
)

compounds_selection = pl.scan_csv("metadata/repurposed_compounds.tsv", separator="\t")

vals = crispr.to_numpy()[:, 0]

metadata = Parallel(n_jobs=-1)(
    delayed(partial(get_item_location_metadata, input_column="JCP2022"))(x)
    for x in vals
)
concat = pl.concat(metadata)
cols = (
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
)
uniq = concat.select(
    pl.col(
        *cols,
        "Metadata_JCP2022",
    )
).unique(cols)


channels = ["DNA"]
sites = ["1"]
correction = "Orig"
rows = uniq.select(cols)

# %% Expensive!
addressed, images = get_jump_image_batch(
    rows, channel=channels, site=sites, correction=correction
)

# for k in zip(addressed, images):
