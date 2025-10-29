from functools import partial
from pathlib import Path

import polars as pl
from joblib import Parallel, delayed
from jump_portrait.fetch import get_item_location_metadata, get_jump_image_batch
from PIL import Image
from pooch import retrieve


def get_metadata_batch(
    perturbations: tuple[str],
    cols=(
        "Metadata_Source",
        "Metadata_Batch",
        "Metadata_Plate",
        "Metadata_Well",
    ),
) -> pl.DataFrame:
    """
    Pull metadata tables using as many processes as possible. Maps JCP id -> address (source, plate, well, sites)
    """
    metadata = Parallel(n_jobs=-1)(
        delayed(partial(get_item_location_metadata, input_column="JCP2022"))(x)
        for x in perturbations
    )
    concat = pl.concat(metadata)

    return concat.select((*cols, "Metadata_JCP2022"))


out_path = Path("./images/raw")
out_path.mkdir(parents=True, exist_ok=True)

sample = 10  # No. of CRISPR and ORF to test
seed = 1

# Pull JCP ids
crispr = (
    pl.scan_csv(
        retrieve(
            "https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/crispr.csv.gz",
            known_hash="55e36e6802c6fc5f8e5d5258554368d64601f1847205e0fceb28a2c246c8d1ed",
        ),
    )
    .select(pl.col("Metadata_JCP2022"))
    .collect()
    .to_series()
    .sample(sample, seed=seed)
)
orf = (
    pl.scan_csv(
        retrieve(
            "https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/orf.csv.gz",
            known_hash="9c7ec4b0fa460a3a30f270a15f11b5e85cef9dd105c8a0ab8ab50f6cc98894b8",
        ),
    )
    .select(pl.col("Metadata_JCP2022"))
    .collect()
    .to_series()
    .sample(sample, seed=seed)
)

compound_selection = (
    pl.scan_csv(
        "../metadata/repurposed_compounds.tsv",
        separator="\t",
    )
    .select(pl.col("Metadata_JCP2022"))
    .collect()
    .to_series()
)

# %%
gene_list = (*crispr, *orf)
gene_rows = get_metadata_batch(gene_list).select(pl.exclude("Metadata_JCP2022"))

channels = ["DNA", "ER", "Mito"]
sites = [str(i) for i in range(1, 7) if i < 2]  # 1->6
correction = "Orig"


# %% Expensive!
addresses, images = get_jump_image_batch(
    gene_rows, channel=channels, site=sites, correction=correction
)

# Re-do it for compounds
# compound_rows = get_metadata_batch(compound_selection)

# %% Save files

for i, (address, image) in enumerate(zip(addresses, images)):
    # `address` is a tuple of (source, plate, well, channel, site)
    fullname = "__".join(address)
    pil_img = Image.fromarray(image)
    pil_img.save(out_path / f"{fullname}.tif")
