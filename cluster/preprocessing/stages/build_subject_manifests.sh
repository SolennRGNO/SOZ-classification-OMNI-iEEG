#!/usr/bin/env bash
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --job-name=hup_ucla_subject_manifests
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/hup_ucla_subject_manifests_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/hup_ucla_subject_manifests_%j.err

set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
readonly COHORT_CSV="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/feature_cohort/nonictal_feature_cohort_hup_ucla.csv"
readonly OUTPUT_DIRECTORY="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/manifests/full_nonictal_v1"
readonly LOG_DIRECTORY="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIRECTORY" "$LOG_DIRECTORY"
[[ -d "$REPOSITORY" ]] || { echo "Missing repository: $REPOSITORY" >&2; exit 1; }
[[ -f "$COHORT_CSV" ]] || { echo "Missing cohort CSV: $COHORT_CSV" >&2; exit 1; }
[[ -f "$VIRTUAL_ENVIRONMENT/bin/activate" ]] || {
    echo "Missing virtual environment: $VIRTUAL_ENVIRONMENT" >&2
    exit 1
}

source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"

python -u scripts/preprocessing/build_subject_manifests.py --log-level INFO build \
    --cohort-csv "$COHORT_CSV" \
    --output-dir "$OUTPUT_DIRECTORY" \
    --pilot-subjects-per-dataset 3

for manifest in \
    "$OUTPUT_DIRECTORY/hup_full_subjects.csv" \
    "$OUTPUT_DIRECTORY/ucla_full_subjects.csv" \
    "$OUTPUT_DIRECTORY/excluded_recordings.csv"
do
    [[ -s "$manifest" ]] || { echo "Missing or empty manifest: $manifest" >&2; exit 1; }
done

echo "Full-cohort subject manifests completed successfully."
