# JUMP-Lite

JUMP-Lite is a compact, source-spanning subset of the JUMP Cell Painting
dataset. It provides lossy-compressed, multichannel image arrays together with
per-site deep-learning feature Parquets, image provenance metadata, perturbation
metadata, and curated RefChemDB-derived annotations.

The original TIFF images are already public under `cpg0016-jump`. This release
is deposited as an aggregated multi-source subset beneath
`cpg0016-jump/source_all`; it adds compressed derivatives and links every
compressed site back to its five original TIFF URLs.

## Dataset scope

| Item | Count |
|---|---:|
| JUMP sources | 6 |
| Batches | 34 |
| Plates | 557 |
| Wells with images | 213,881 |
| Sites | 855,519 |
| Channels per site | 5 |
| Original image references | 4,277,595 |

Included sources are `source_2`, `source_4`, `source_6`, `source_7`, `source_8`,
and `source_13`.

The plate collection was defined from the JUMP-Lite plate list and filtered
against the JUMP redlist and graylist. Six negative-control-only graylisted
plates were excluded, leaving 557 plates. At most four sites are retained per
well; 213,876 wells have four sites and five wells have three available sites.
The release freezes the exact site keys represented by the MQ image store.

## Compressed images

Each image dataset is a Zarr v2 group containing one array per site. Site keys
use:

```text
<source>__<batch>__<plate>__<well>__<site>
```

Each site array has dimensions `(channel, y, x)`, uses unsigned 16-bit values,
and stores all five channels in one chunk. Channel order is:

```text
AGP, DNA, ER, Mito, RNA
```

Three JPEG XL variants are included:

| Dataset | JPEG XL distance | Description |
|---|---:|---|
| `jpegxl_lossy_hq.zarr` | 1.0 | High quality |
| `jpegxl_lossy_mq.zarr` | 3.0 | Medium quality and canonical site manifest |
| `jpegxl_lossy_d20.zarr` | 20.0 | High-compression comparison variant |

These arrays are lossy derivatives and should not be interpreted as replacing
the original JUMP TIFFs. Decoding requires a Zarr-compatible registration of
the `imagecodecs_jpegxl` codec, such as `imagecodecs.numcodecs.Jpegxl`.

The experimental, incomplete `zstd.zarr` store in the working filesystem is
not part of this release.

## Per-site Parquet outputs

Feature outputs intentionally use one Parquet file per image site. Their file
stem is identical to the corresponding Zarr site key. Each Parquet is in long
form with the columns:

```text
tile, label, branch, metric, value, object, tp
```

The release contains outputs from the following model families and image
compression variants:

| Model family | Image variants |
|---|---|
| DINOv2 ViT-S/14 | MQ, HQ, D20 |
| Randomly initialized DINOv2 ViT-S/14 | MQ, HQ, D20 |
| MorphEm | MQ, HQ, D20 |
| OpenPhenom | MQ, HQ, D20 |
| SubCell | MQ |
| SubCell clipped-input variant | MQ, HQ, D20 |

Broadly, DINOv2 uses AGP/DNA/ER 224-pixel tiles; MorphEm uses all five channels
with 224-pixel tiles; OpenPhenom uses all five channels with 256-pixel tiles,
outlier clipping, and 8-bit conversion; and SubCell uses Mito/ER/DNA/AGP with
448-pixel tiles. The processing scripts deposited with the model outputs are
the authoritative record of run parameters.

## Metadata files

The release metadata are generated from the exact canonical MQ keys rather than
by resampling:

- `jump_lite_site_index.parquet`: one row per compressed site, including source,
  batch, plate, well, site, and the five original JUMP TIFF URLs.
- `jump_lite_image_index.parquet`: tidy expansion with one row per site/channel
  and 4,277,595 total rows.
- `jump_lite_perturbation_metadata.parquet`: 161,926 annotated wells with JUMP
  identifiers, perturbation type, symbols, and grouping information. Empty or
  otherwise unannotated wells remain represented in the site index.
- `jump_lite_plate_manifest.parquet`: per-plate well and site counts.
- `metadata_manifest.json`: counts, channel order, artifact sizes, and the
  SHA-256 digest of the frozen site-key set.

The canonical sorted site-key digest for this release is:

```text
399e703bc924a19f7c3827db3c711373306e3d943d2f12cf56d0a368f5d13961
```

## Annotations

`jump_lite_refchem_annotations.parquet` contains the release-relevant subset of
curated RefChemDB/JUMP confidence matches:

- 29,681 annotation rows
- 1,576 distinct JUMP perturbation identifiers
- target genes, target type, mode and activity fields
- cross-modality and within-modality confidence tiers
- compound/perturbation direction-match indicators

Annotations whose query perturbation is absent from the frozen JUMP-Lite
release are excluded from this deposited table.

## Data integrity

Before upload, a fail-closed validator requires:

1. MQ, HQ, and D20 to have exactly 855,519 identical site keys.
2. Every per-site Parquet collection to have exactly the keys of its associated
   image dataset.
3. The frozen site and image indices to match the canonical keys and contain all
   five original image URLs.
4. Metadata, plate, source, well, and annotation invariants to pass.

The upload is blocked if any check fails. Verification after staging compares
object counts between local sources and S3, following Cell Painting Gallery
upload guidance.

## Provenance

The index-generation inputs and related JUMP/JUMP-Lite tables are described at:

- Zenodo: <https://doi.org/10.5281/zenodo.18705140>
- Cell Painting Gallery JUMP project:
  <https://registry.opendata.aws/cellpainting-gallery/>

JUMP-Lite is derived from `cpg0016-jump`; users should cite the primary JUMP
Cell Painting dataset and the feature-model publications appropriate to their
use.
