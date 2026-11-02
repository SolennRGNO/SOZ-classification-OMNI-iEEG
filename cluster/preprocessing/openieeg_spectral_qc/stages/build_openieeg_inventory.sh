#!/usr/bin/env bash
#SBATCH --job-name=openieeg_inventory
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_openieeg_qc_v1/logs/%x_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_openieeg_qc_v1/logs/%x_%j.err

# Read-only EDF-header inventory of the OpenIEEG recordings, excluding the
# sub-openieegUCLA* subjects that the `ucla` datasets2.yml entry already covers.
#
# This inventory feeds the raw spectral diagnostic only. Unlike the HUP/UCLA
# inventory it is NOT filtered to >= 1 kHz: the diagnostic deliberately reads
# every recording at its own sampling rate, so the *full* recording_inventory
# table is the input downstream, not the eligible_* subset.

set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
readonly DATA_ROOT="/CONFIGURE_DATA_ROOT"
readonly ANALYSIS_ROOT="/CONFIGURE_SCRATCH/omni_openieeg_qc_v1"
readonly OUTPUT_DIR="$ANALYSIS_ROOT/inventory"
readonly LOG_DIR="$ANALYSIS_ROOT/logs"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
[[ -d "$REPOSITORY" ]] || { echo "Missing repository: $REPOSITORY" >&2; exit 1; }
[[ -d "$DATA_ROOT" ]] || { echo "Missing dataset root: $DATA_ROOT" >&2; exit 1; }
[[ -f "$VIRTUAL_ENVIRONMENT/bin/activate" ]] || {
    echo "Missing virtual environment: $VIRTUAL_ENVIRONMENT" >&2
    exit 1
}

source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"

export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

python -u scripts/preprocessing/build_inventory.py \
    --data-root "$DATA_ROOT" \
    --datasets openieeg \
    --output-dir "$OUTPUT_DIR" \
    --output-suffix openieeg \
    --check-edf-header \
    --log-level INFO

echo "OpenIEEG inventory written to $OUTPUT_DIR/recording_inventory_openieeg.csv"
