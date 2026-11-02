#!/usr/bin/env bash
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --job-name=omni_openieeg_inventory
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/omni_openieeg_inventory_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/omni_openieeg_inventory_%j.err

# Read-only EDF-header inventory of the OpenIEEG recordings, using exactly the
# same script and the same >= 1000 Hz eligibility rule as the HUP/UCLA
# inventory. sub-openieegUCLA* is excluded by the `openieeg` entry of
# Neurodags/datasets2.yml, because those subjects belong to the `ucla` cohort.

set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly DATA_ROOT=/CONFIGURE_DATA_ROOT
readonly OUTPUT_DIR=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/inventory
readonly LOG_DIR=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this inventory with sbatch; do not run it on a login node." >&2
    exit 2
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
[[ -d "$REPOSITORY" ]] || { echo "Cluster repository does not exist: $REPOSITORY" >&2; exit 2; }
[[ -d "$DATA_ROOT" ]] || { echo "Dataset root does not exist: $DATA_ROOT" >&2; exit 2; }
[[ -f "$VIRTUAL_ENVIRONMENT/bin/activate" ]] || {
    echo "Virtual environment does not exist: $VIRTUAL_ENVIRONMENT" >&2
    exit 2
}

source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"

export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -u scripts/preprocessing/build_inventory.py \
    --data-root "$DATA_ROOT" \
    --datasets openieeg \
    --output-dir "$OUTPUT_DIR" \
    --output-suffix openieeg \
    --check-edf-header \
    --log-level INFO

echo "OpenIEEG sampling-frequency inventory completed successfully."
echo "Inventory report: $OUTPUT_DIR/inventory_openieeg.html"
