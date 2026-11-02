#!/usr/bin/env bash
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --job-name=omni_hup_ucla_inventory
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/omni_hup_ucla_inventory_%j.out

set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly DATA_ROOT=/CONFIGURE_DATA_ROOT
readonly OUTPUT_DIR=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/inventory
readonly LOG_DIR=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this inventory with sbatch; do not run it on a login node." >&2
    exit 2
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [[ ! -d "$REPOSITORY" ]]; then
    echo "Cluster repository does not exist: $REPOSITORY" >&2
    exit 2
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Dataset root does not exist: $DATA_ROOT" >&2
    exit 2
fi
if [[ ! -f "$VIRTUAL_ENVIRONMENT/bin/activate" ]]; then
    echo "Virtual environment does not exist: $VIRTUAL_ENVIRONMENT" >&2
    exit 2
fi

source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"

export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -u scripts/preprocessing/build_inventory.py \
    --data-root "$DATA_ROOT" \
    --datasets ucla hup \
    --output-dir "$OUTPUT_DIR" \
    --check-edf-header \
    --log-level INFO

echo "HUP/UCLA sampling-frequency inventory completed successfully."
echo "Inventory report: $OUTPUT_DIR/inventory_hup_ucla.html"
