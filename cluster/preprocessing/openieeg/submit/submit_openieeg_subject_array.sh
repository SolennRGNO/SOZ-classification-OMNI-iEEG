#!/usr/bin/env bash
set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly PYTHON="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags/bin/python"
manifest=""
output_root=""
mode=""
while (( $# )); do
    case "$1" in
        --manifest) manifest="$2"; shift 2 ;;
        --output-root) output_root="$2"; shift 2 ;;
        --mode) mode="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$mode" == "pilot" || "$mode" == "full" ]] || { echo "--mode must be pilot or full." >&2; exit 2; }
[[ -f "$manifest" ]] || { echo "Missing manifest: $manifest" >&2; exit 1; }
[[ -n "$output_root" ]] || { echo "--output-root is required." >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
case "$mode" in
    pilot) expected_root="/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/pilot_v1/openieeg" ;;
    full) expected_root="/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/full_v1/openieeg" ;;
esac
[[ "$output_root" == "$expected_root" ]] || { echo "Output root must be $expected_root." >&2; exit 2; }

cd "$REPOSITORY"
export PYTHONPATH="$REPOSITORY${PYTHONPATH:+:$PYTHONPATH}"
subject_count=$("$PYTHON" scripts/preprocessing/build_subject_manifests.py count --manifest "$manifest")
(( subject_count > 0 )) || { echo "Manifest is empty." >&2; exit 1; }
array_spec="0-$((subject_count - 1))"
mkdir -p /CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs "$output_root"
submission=$(sbatch --parsable --array="$array_spec" \
    cluster/preprocessing/openieeg/stages/preprocess_openieeg_subject.sh \
    --manifest "$manifest" --output-root "$output_root" --mode "$mode")
job_id="${submission%%;*}"
echo "Submitted OpenIEEG $mode array job ID: $job_id (array $array_spec)"
