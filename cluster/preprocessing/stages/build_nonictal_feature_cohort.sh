#!/usr/bin/env bash
#SBATCH --job-name=hup_ucla_feature_cohort
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/hup_ucla_feature_cohort_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/hup_ucla_feature_cohort_%j.err

set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly DATA_ROOT="/CONFIGURE_DATA_ROOT"
readonly INVENTORY="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/inventory/eligible_recordings_hup_ucla.csv"
readonly OUTPUT_DIR="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/feature_cohort"
readonly LOG_DIR="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
[[ -d "$REPOSITORY" ]] || { echo "Missing repository: $REPOSITORY" >&2; exit 1; }
[[ -f "$INVENTORY" ]] || { echo "Missing inventory: $INVENTORY" >&2; exit 1; }
[[ -f "$VIRTUAL_ENVIRONMENT/bin/activate" ]] || {
    echo "Missing virtual environment: $VIRTUAL_ENVIRONMENT" >&2
    exit 1
}

source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"

python scripts/preprocessing/build_nonictal_feature_cohort.py \
    --inventory-csv "$INVENTORY" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --log-level INFO
