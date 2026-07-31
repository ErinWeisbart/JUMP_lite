#!/usr/bin/env bash
# Validate the complete JUMP-Lite release, then sync one component to CPG staging.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGING_ROOT="s3://staging-cellpainting-gallery/cpg0016-jump/source_all"
REGION="us-east-1"
VALIDATION_REPORT="/work/datasets/jump_lite/cpg_release/validation_report.json"
APPLY=false

usage() {
  cat <<'EOF'
Usage:
  upload_to_staging.sh [--apply] LOCAL_PATH DESTINATION_RELATIVE_PATH

Without --apply, AWS CLI runs with --dryrun. With --apply, the command first
writes a successful validation report and then uploads. The destination is
always constrained beneath:

  s3://staging-cellpainting-gallery/cpg0016-jump/source_all/

Temporary CPG credentials, including AWS_SESSION_TOKEN, must already be active.
Use: source cpg_upload/activate_cpg_credentials.sh
EOF
}

if [[ ${1:-} == "--apply" ]]; then
  APPLY=true
  shift
fi
if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

SOURCE=$1
RELATIVE_DESTINATION=${2#/}

if [[ ! -d "$SOURCE" ]]; then
  echo "ERROR: local source must be a directory: $SOURCE" >&2
  exit 1
fi
if [[ -z "$RELATIVE_DESTINATION" || "$RELATIVE_DESTINATION" == *".."* ]]; then
  echo "ERROR: unsafe or empty destination: $RELATIVE_DESTINATION" >&2
  exit 1
fi
if [[ ! "$RELATIVE_DESTINATION" =~ ^[A-Za-z0-9_./-]+$ ]]; then
  echo "ERROR: destination contains characters outside [A-Za-z0-9_./-]" >&2
  exit 1
fi
if ! command -v aws >/dev/null; then
  echo "ERROR: AWS CLI is not available" >&2
  exit 1
fi
if [[ -z ${AWS_ACCESS_KEY_ID:-} || -z ${AWS_SECRET_ACCESS_KEY:-} || -z ${AWS_SESSION_TOKEN:-} ]]; then
  echo "ERROR: temporary CPG credentials are not active (AWS_SESSION_TOKEN is required)" >&2
  exit 1
fi

if $APPLY; then
  python "$SCRIPT_DIR/validate_release.py" --json-output "$VALIDATION_REPORT"
else
  python "$SCRIPT_DIR/validate_release.py"
fi

DESTINATION="$STAGING_ROOT/$RELATIVE_DESTINATION/"
ARGS=(
  s3 sync "$SOURCE" "$DESTINATION"
  --region "$REGION"
  --no-follow-symlinks
  --only-show-errors
)
if ! $APPLY; then
  ARGS+=(--dryrun)
fi

printf 'Validated source: %s\nDestination: %s\nMode: %s\n' \
  "$SOURCE" "$DESTINATION" "$($APPLY && echo APPLY || echo DRY-RUN)"
aws "${ARGS[@]}"
