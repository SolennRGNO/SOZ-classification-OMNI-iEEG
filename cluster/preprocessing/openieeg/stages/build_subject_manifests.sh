#!/usr/bin/env bash
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --job-name=openieeg_subject_manifests
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/openieeg_subject_manifests_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/openieeg_subject_manifests_%j.err

# Build the OpenIEEG scale-out subject manifests with the same builder the
# HUP/UCLA workflow uses.
#
# The HUP/UCLA workflow inserts build_nonictal_feature_cohort.py between the
# inventory and the manifests to split HUP recordings into interictal/ictal and
# to apply the UCLA all-interictal dataset rule. OpenIEEG carries no such
# recording-state metadata (every recording is ses-01, task-sleep, one recording
# per subject), so there is nothing for that step to decide and the eligible
# inventory is the cohort. The >= 1000 Hz rule is applied by the inventory and
# re-checked by the manifest builder.

set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
readonly COHORT_CSV="/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/inventory/eligible_recordings_openieeg.csv"
readonly OUTPUT_DIRECTORY="/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/manifests/full_v1"
readonly LOG_DIRECTORY="/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIRECTORY" "$LOG_DIRECTORY"
[[ -d "$REPOSITORY" ]] || { echo "Missing repository: $REPOSITORY" >&2; exit 1; }
[[ -f "$COHORT_CSV" ]] || {
    echo "Missing eligible inventory: $COHORT_CSV" >&2
    echo "Run cluster/preprocessing/openieeg/stages/build_inventory.sh first." >&2
    exit 1
}
[[ -f "$VIRTUAL_ENVIRONMENT/bin/activate" ]] || {
    echo "Missing virtual environment: $VIRTUAL_ENVIRONMENT" >&2
    exit 1
}

source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"

# One subject in the pilot manifest: each OpenIEEG subject holds exactly one
# recording, so the pilot is a single-recording pilot.
python -u scripts/preprocessing/build_subject_manifests.py --log-level INFO build \
    --cohort-csv "$COHORT_CSV" \
    --output-dir "$OUTPUT_DIRECTORY" \
    --pilot-subjects-per-dataset 1

for manifest in \
    "$OUTPUT_DIRECTORY/openieeg_full_subjects.csv" \
    "$OUTPUT_DIRECTORY/openieeg_multisubject_pilot.csv"
do
    [[ -s "$manifest" ]] || { echo "Missing or empty manifest: $manifest" >&2; exit 1; }
done

echo "OpenIEEG subject manifests completed successfully."
