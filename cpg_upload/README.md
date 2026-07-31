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

An upload must only be run through `upload_to_staging.sh`, which executes
`validate_release.py` first and exits before AWS is called if any inconsistency
is found.

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
- `verify_staging.sh`: compares local file count with the recursive staging
  object count after each sync.
- `JUMP_LITE_README.md`: dataset-facing README copied to the release root by
  the metadata builder and intended for upload with the release.

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

## 4. Activate temporary CPG credentials

Install AWS CLI first, then source:

```bash
source cpg_upload/activate_cpg_credentials.sh
```

The credentials are scoped to:

```text
s3://staging-cellpainting-gallery/cpg0016-jump/*
```

and are requested with `READWRITE` permission in `us-east-1` for 12 hours.

## 5. Upload one release component

Dry run:

```bash
cpg_upload/upload_to_staging.sh LOCAL_PATH RELATIVE_DESTINATION
```

Actual sync:

```bash
cpg_upload/upload_to_staging.sh --apply LOCAL_PATH RELATIVE_DESTINATION
```

The wrapper never uses `--delete`, never follows symlinks, and rejects
out-of-prefix destinations. Destination arguments are relative to
`cpg0016-jump/source_all/`; the final image, metadata, and `workspace_dl`
subpaths should be agreed with the CPG maintainer before the first upload.

## 6. Verify staging

After each applied sync, compare local file count with the staging object count:

```bash
cpg_upload/verify_staging.sh LOCAL_PATH RELATIVE_DESTINATION
```

Only notify the CPG maintainer after every component passes this check and the
complete release passes `validate_release.py` again.

## Deterministic prevention

Site sampling in `prep/build_jl_index.sql` now partitions on the complete
source/batch/plate/well identity and orders by `Metadata_Site` before retaining
at most four sites. This prevents reruns from selecting a different four-site
subset. Compression jobs should additionally start from a clean store whenever
the input manifest changes; skip-existing mode must not be used to merge data
from different site manifests.
