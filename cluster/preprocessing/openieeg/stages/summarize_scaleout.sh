#!/usr/bin/env bash
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --job-name=openieeg_scaleout_summary
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/openieeg_scaleout_summary_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/openieeg_scaleout_summary_%j.err

set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
readonly ANALYSIS_ROOT="/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1"
readonly MANIFEST="$ANALYSIS_ROOT/manifests/full_v1/openieeg_full_subjects.csv"
readonly STATUS_ROOT="$ANALYSIS_ROOT/full_v1/openieeg"
readonly OUTPUT_DIR="$ANALYSIS_ROOT/scaleout_summary"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR" "$ANALYSIS_ROOT/logs"
[[ -f "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 1; }
[[ -d "$STATUS_ROOT" ]] || { echo "Missing status root: $STATUS_ROOT" >&2; exit 1; }

source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"

summary_arguments=(
    --manifest "$MANIFEST"
    --status-root "$STATUS_ROOT"
    --output-dir "$OUTPUT_DIR"
)
if [[ -f "$ANALYSIS_ROOT/manifests/full_v1/excluded_recordings.csv" ]]; then
    summary_arguments+=(--excluded-recordings "$ANALYSIS_ROOT/manifests/full_v1/excluded_recordings.csv")
fi

python -u scripts/preprocessing/summarize_scaleout.py "${summary_arguments[@]}"

echo "OpenIEEG scale-out summary: $OUTPUT_DIR/preprocessing_scaleout_report.html"
