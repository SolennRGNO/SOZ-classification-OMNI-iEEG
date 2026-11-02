#!/usr/bin/env bash
set -euo pipefail
umask 027

# Single-recording OpenIEEG pilot. Writes to its own pilot root so it can never
# overwrite the full-cohort derivatives, exactly like the HUP/UCLA pilots.
#
# Run on the login node (it only calls sbatch):
#   bash cluster/preprocessing/openieeg/pilot/submit_openieeg_pilot.sh

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly MANIFEST="/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/manifests/full_v1/openieeg_multisubject_pilot.csv"
readonly OUTPUT_ROOT="/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/pilot_v1/openieeg"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run this lightweight submitter on the login node, not inside a job." >&2
    exit 2
fi

cd "$REPOSITORY"
[[ -f "$MANIFEST" ]] || {
    echo "Missing pilot manifest: $MANIFEST" >&2
    echo "Run cluster/preprocessing/openieeg/stages/build_subject_manifests.sh first." >&2
    exit 1
}

bash cluster/preprocessing/openieeg/submit/submit_openieeg_subject_array.sh \
    --manifest "$MANIFEST" \
    --output-root "$OUTPUT_ROOT" \
    --mode pilot
