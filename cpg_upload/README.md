# JUMP-Lite CPG upload machinery

This directory contains the preparation and fail-closed upload tools for the
JUMP-Lite contribution beneath `cpg0016-jump/source_all`.

No script stores AWS credentials in the repository. The long-lived CPG grant
keys remain in `~/.cpg_key_id` and `~/.cpg_access_key`; temporary credentials
are requested through S3 Access Grants.

## Safety invariant

All compressed image datasets and every corresponding per-site Parquet variant
must contain the exact same frozen set of 855,519 site keys. The MQ Zarr is the
canonical set for this release.

Uploads must use either `upload_to_staging.sh` for a single unchanged directory
or `run_background_upload.sh` for the complete release. Both workflows are
gated by `validate_release.py` and exit before S3 writes if any inconsistency is
found.

## Files

- `reconcile_site_sets.py`: audits MQ/HQ/D20 and reversibly quarantines surplus
  image arrays and per-site Parquets. It never deletes data.
- `build_cpg_metadata.py`: joins the exact MQ keys to the full JUMP-Lite index
  and writes frozen release metadata under
  `/work/datasets/jump_lite/cpg_release/metadata/`.
- `validate_release.py`: read-only, fail-closed release preflight.
- `activate_cpg_credentials.sh`: exchanges the CPG grant keys for temporary,
  prefix-scoped READWRITE credentials. This file must be sourced.
- `upload_to_staging.sh`: validates and then runs one constrained CPG staging
  sync. It is an AWS dry run unless `--apply` is supplied.
- `run_background_upload.sh`: supervises the complete v1.0 upload, renews
  temporary credentials, and safely resumes interrupted image syncs.
- `upload_profiles_to_staging.py`: concurrently uploads flat local Parquets
  into the CPG model/source/batch/plate/well-site hierarchy without creating
  millions of local hard links. Per-variant checkpoints make it resumable.
- `upload_status.sh`: reports supervisor, component, log, and profile-checkpoint
  progress for the active background run.
- `rebuild_zstd_from_originals.py`: streams the five original public TIFFs for
  each frozen MQ site into memory and writes a matching site-major Zarr v3 array
  with lossless Blosc/Zstd compression. It does not cache TIFFs.
- `run_zstd_rebuild.sh`: background wrapper with durable logs and completion
  markers for the resumable Zstd rebuild.
- `zstd_rebuild_status.sh`: reports the active rebuild checkpoint and log tail.
- `systemd/*.service`: persistent user-service definitions that resume the CPG
  upload and Zstd rebuild after user-service-manager or host restarts.
- `verify_staging.sh`: compares local file count with the recursive staging
  object count after each sync.
- `JUMP_LITE_README.md`: dataset-facing README copied to the release root by
  the metadata builder and uploaded with the release.

## 1. Reconcile the current site discrepancy

Audit only:

```bash
python cpg_upload/reconcile_site_sets.py
```

The current repair requires write ACLs on the 274 surplus HQ/D20 image-array
directories. Generate their paths with:

```bash
python cpg_upload/reconcile_site_sets.py \
  --print-surplus-image-paths > /tmp/jump_lite_surplus_image_paths.txt
```

After an administrator grants access to those paths, apply the reversible
repair:

```bash
python cpg_upload/reconcile_site_sets.py --apply
```

Surplus entries are moved under `/work/datasets/jump_lite/quarantine/` with a
JSON manifest.

## 2. Build frozen release metadata

```bash
python cpg_upload/build_cpg_metadata.py
```

This deliberately uses the existing MQ keys rather than resampling sites. It
writes wide and tidy image indices, perturbation metadata, release-filtered
annotations, a plate manifest, and a metadata manifest.

## 3. Validate

```bash
python cpg_upload/validate_release.py
```

A successful validation reports `CPG release status: ready` and exits zero.
Anything else blocks upload.

## 4. CPG object layout

The release uses the following agreed `source_all` namespace:

```text
cpg0016-jump/source_all/
├── images_compressed/jump_lite/v1.0/<codec>.zarr/
├── workspace/metadata/jump_lite/v1.0/<release metadata files>
└── workspace_dl/embeddings/
    └── <model>-<codec>/jump_lite/v1.0/
        └── <source>/<batch>/<plate>/<well>-<site>/embedding.parquet
```

The image codecs are `jpegxl_lossy_mq`, `jpegxl_lossy_hq`, and
`jpegxl_lossy_d20`. The experimental `zstd.zarr` is not uploaded. Metadata
includes the dataset README, wide and tidy image indices, perturbation metadata,
RefChemDB annotations, plate manifest, and release manifest.

Local embedding files are flat and named:

```text
<source>__<batch>__<plate>__<well>__<site>.parquet
```

`upload_profiles_to_staging.py` parses that identity and writes the standard CPG
well-site object key shown above. It maps internal names
`openphenom_confusing` to `openphenom` and `subcell__clip01` to
`subcell_clip01`; local data are not renamed.

The original TIFFs and source-specific `load_data_csv` files already exist in
the six contributing JUMP source folders. They are referenced by the deposited
indices and are not duplicated under `source_all`.

## 5. Activate temporary CPG credentials

Install AWS CLI v2 first, then source:

```bash
source cpg_upload/activate_cpg_credentials.sh
```

The credentials are scoped to:

```text
s3://staging-cellpainting-gallery/cpg0016-jump/*
```

and are requested with `READWRITE` permission in `us-east-1` for 12 hours.
AWS CLI v2 can be used without permanent installation through:

```bash
nix shell nixpkgs#awscli2
```

## 6. Run the complete upload in the background

Install and start the persistent user service from the repository root:

```bash
mkdir -p ~/.config/systemd/user
ln -sfn "$PWD/cpg_upload/systemd/jump-lite-cpg-upload.service" \
  ~/.config/systemd/user/jump-lite-cpg-upload.service
systemctl --user daemon-reload
systemctl --user enable --now jump-lite-cpg-upload.service
```

The enabled service survives user-service-manager and host restarts. Check its
state with `systemctl --user status jump-lite-cpg-upload.service`.

The supervisor validates once before writes, creates a clean metadata view,
starts metadata plus three image syncs, and starts the transformed embedding
upload. Image syncs stop after 11 hours, renew their 12-hour credentials, and
resume. The Python embedding uploader refreshes credentials proactively and
stores deterministic checkpoints under:

```text
/work/datasets/jump_lite/cpg_upload_state/v1.0/profiles/
```

Run logs are stored under a UTC timestamp in:

```text
/work/datasets/jump_lite/cpg_upload_logs/
```

The `latest` symlink points to the active or most recent run. Monitor without
changing S3:

```bash
cpg_upload/upload_status.sh
```

Re-running the supervisor is safe: `aws s3 sync` skips matching image objects,
and profile checkpoints skip successfully uploaded Parquets. No upload command
uses `--delete` or follows symlinks.

## 7. Rebuild the optional lossless Zstd store

The legacy local `zstd.zarr` is an interrupted one-plate experiment with a
well/channel layout and is not part of v1.0. Install the persistent rebuild
service to create a clean replacement from original TIFFs while the v1.0 upload
continues:

```bash
mkdir -p ~/.config/systemd/user
ln -sfn "$PWD/cpg_upload/systemd/jump-lite-zstd-rebuild.service" \
  ~/.config/systemd/user/jump-lite-zstd-rebuild.service
systemctl --user daemon-reload
systemctl --user enable --now jump-lite-zstd-rebuild.service
```

Check service state with
`systemctl --user status jump-lite-zstd-rebuild.service`.

The frozen site index supplies exactly the 855,519 MQ keys and five original
TIFF URLs per site. For each site, the builder:

1. downloads AGP, DNA, ER, Mito, and RNA directly from the public CPG;
2. decodes them in memory without retaining raw TIFFs;
3. checks shape and dtype against the corresponding MQ array;
4. writes one `(5, y, x)` Zarr v3 array and one Blosc/Zstd level-9 chunk; and
5. checkpoints progress after each bounded batch.

The original image-store parent is not writable by the uploader, so the
resumable building store, validated replacement, and state are kept at:

```text
/work/datasets/jump_lite/zstd_rebuild/v1.0/zstd.building.zarr/
/work/datasets/jump_lite/zstd_rebuild/v1.0/zstd.zarr/
/work/datasets/jump_lite/zstd_rebuild_state/v1.0/checkpoint.json
```

Monitor it with:

```bash
cpg_upload/zstd_rebuild_status.sh
```

On completion, the builder verifies the full site count and canonical SHA-256
digest and atomically renames the writable building store to `zstd.zarr`. The
legacy incomplete store in the protected image directory remains untouched; an
administrator can swap in the validated replacement afterward. A truncated
test run never performs final renaming. The rebuilt Zstd is not automatically
added to or uploaded with v1.0.

## 8. Upload one unchanged directory

For a one-component dry run:

```bash
cpg_upload/upload_to_staging.sh LOCAL_PATH RELATIVE_DESTINATION
```

Add `--apply` only after reviewing its destination. Destination arguments are
relative to `cpg0016-jump/source_all/`.

## 9. Verify staging

After transfer, compare local and staging object counts:

```bash
cpg_upload/verify_staging.sh LOCAL_PATH RELATIVE_DESTINATION
```

Embedding variants require counting their transformed S3 prefix and comparing
against 855,519 objects each. Only notify the CPG maintainer after all 16
embedding variants, all three image stores, and metadata pass verification and
the complete release passes `validate_release.py` again.

## Deterministic prevention

Site sampling in `prep/build_jl_index.sql` now partitions on the complete
source/batch/plate/well identity and orders by `Metadata_Site` before retaining
at most four sites. This prevents reruns from selecting a different four-site
subset. Compression jobs should additionally start from a clean store whenever
the input manifest changes; skip-existing mode must not be used to merge data
from different site manifests.
