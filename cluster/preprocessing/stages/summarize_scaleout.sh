#!/usr/bin/env bash
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --job-name=hup_ucla_preproc_summary
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/hup_ucla_preproc_summary_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/hup_ucla_preproc_summary_%j.err

set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
readonly MANIFEST_DIRECTORY="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/manifests/full_nonictal_v1"
readonly PREPROCESSED_ROOT="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/full_nonictal_v1"
readonly OUTPUT_DIRECTORY="$PREPROCESSED_ROOT/summary"
readonly LOG_DIRECTORY="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIRECTORY" "$LOG_DIRECTORY"
[[ -d "$REPOSITORY" ]] || { echo "Missing repository: $REPOSITORY" >&2; exit 1; }
[[ -f "$MANIFEST_DIRECTORY/hup_full_subjects.csv" ]] || { echo "Missing HUP manifest." >&2; exit 1; }
[[ -f "$MANIFEST_DIRECTORY/ucla_full_subjects.csv" ]] || { echo "Missing UCLA manifest." >&2; exit 1; }
[[ -f "$VIRTUAL_ENVIRONMENT/bin/activate" ]] || {
    echo "Missing virtual environment: $VIRTUAL_ENVIRONMENT" >&2
    exit 1
}

source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"

python -u scripts/preprocessing/summarize_scaleout.py \
    --manifest "$MANIFEST_DIRECTORY/hup_full_subjects.csv" \
    --manifest "$MANIFEST_DIRECTORY/ucla_full_subjects.csv" \
    --status-root "$PREPROCESSED_ROOT/hup" \
    --status-root "$PREPROCESSED_ROOT/ucla" \
    --excluded-recordings "$MANIFEST_DIRECTORY/excluded_recordings.csv" \
    --output-dir "$OUTPUT_DIRECTORY"

for output in \
    "$OUTPUT_DIRECTORY/completed_subjects.csv" \
    "$OUTPUT_DIRECTORY/failed_subjects.csv" \
    "$OUTPUT_DIRECTORY/preprocessing_resource_summary.csv" \
    "$OUTPUT_DIRECTORY/preprocessing_scaleout_report.html"
do
    [[ -f "$output" ]] || { echo "Missing summary output: $output" >&2; exit 1; }
done

echo "Preprocessing scale-out summary completed successfully."
