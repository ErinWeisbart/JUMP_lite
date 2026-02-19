from functools import partial
from pathlib import Path

import boto3
import duckdb
from joblib import Parallel, delayed

# meta_file = out_path.parent / "metadata.parquet"
# progress_file = out_path.parent / "progress.txt"

# Logic to pull specific JCP ids
# Pull JCP ids
# crispr = (
#     pl.scan_csv(
#         retrieve(
#             "https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/crispr.csv.gz",
#             known_hash="55e36e6802c6fc5f8e5d5258554368d64601f1847205e0fceb28a2c246c8d1ed",
#         ),
#     )
#     .select(pl.col("Metadata_JCP2022"))
#     .collect()
#     .to_series()
# )
# orf = (
#     pl.scan_csv(
#         retrieve(
#             "https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/orf.csv.gz",
#             known_hash="9c7ec4b0fa460a3a30f270a15f11b5e85cef9dd105c8a0ab8ab50f6cc98894b8",
#         ),
#     )
#     .select(pl.col("Metadata_JCP2022"))
#     .collect()
#     .to_series()
#     # .sample(sample, seed=seed)
# )

# compound_selection = (
#     pl.scan_csv(
#         "../metadata/repurposed_compounds.tsv",
#         separator="\t",
#     )
#     .select(pl.col("Metadata_JCP2022"))
#     .collect()
#     .to_series()
#     .unique()
# )

# # %%

# channels = ["DNA", "AGP", "Mito", "RNA", "ER"]
# sites = [str(i) for i in range(1, 7)]  # 1->6
# correction = "Orig"

# # Do not pull the mapper unless explicitly told to
# if not (meta_file).exists():
#     # %% Download metadata tables
#     print("Downloading gene metadata")
#     gene_list = (*crispr, *orf)
#     t_start = perf_counter()
#     gene_rows = get_metadata_batch(gene_list)
#     print(f"Done downloading gene metadata in {int(perf_counter() - t_start)} seconds")

#     # Compounds too
#     print("Downloading compound metadata")
#     t_start = perf_counter()
#     compound_selection = get_metadata_batch(compound_selection)

#     # Add whole plates if necessary
#     whole_plate = get_whole_plate_location_info("110000293081")
#     compound_rows = pl.concat((whole_plate, compound_selection)).unique()
#     print(
#         f"Done downloading compound metadata in {int(perf_counter() - t_start)} seconds"
#     )
#     all_rows_data = (*gene_rows, *compound_rows)

#     metadata_all = pl.concat(all_rows_data)
#     metadata_all.write_parquet(meta_file)
#     print("Parquet saved. Will download images now.")

# else:
#     metadata_all = pl.read_parquet(meta_file)


out_dir = Path("/work/datasets/jump_lite/imgs/raw")
print("Loading list of files")
with duckdb.connect() as con:
    uris_list = [
        (
            *list(x.values())[:-3],
            str(x["Metadata_Site"]),
            x["Metadata_Channel"].removeprefix("URL_Orig"),
            x["uri"].removeprefix("s3://cellpainting-gallery/"),
        )
        for x in con.sql(
            "FROM read_parquet('/work/datasets/jump_lite/misc/jl_index_tidy.parquet')"
        )
        .to_arrow_table()
        .to_pylist()
    ]
print("File list ready")


# %%
def download_uri(meta: list[str], out_dir: str):
    from botocore import UNSIGNED
    from botocore.config import Config
    from loguru import logger

    *location, key = meta
    local_name = "__".join(location) + ".tif"
    local_file = out_dir / Path(local_name)
    Path(local_file).parent.mkdir(exist_ok=True, parents=True)
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    try:
        if not local_file.exists():
            logger.add(out_dir / "../../misc" / "download_log.txt")
            text = f"Downloading {key} into {local_file}"
            logger.info(text)
            s3_client.download_file("cellpainting-gallery", key, str(local_file))
            logger.info(f"{key} was successfully downloaded")

    except Exception as e:
        logger.error(f"{key} Failed: {e}")


curried = partial(download_uri, out_dir=out_dir)


print("Downloads will start now")
result = list(Parallel(n_jobs=192)(delayed((curried))(x) for x in uris_list))
