#!/usr/bin/env bash
#SBATCH --job-name=hup_ucla_raw_line_noise_qc
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/hup_ucla_raw_line_noise_qc_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/hup_ucla_raw_line_noise_qc_%j.err

set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
readonly DATA_ROOT="/CONFIGURE_DATA_ROOT"
readonly INVENTORY_CSV="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/inventory/eligible_recordings_hup_ucla.csv"
readonly OUTPUT_DIRECTORY="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/line_noise_qc"
readonly LOG_DIRECTORY="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIRECTORY" "$LOG_DIRECTORY"

[[ -d "$REPOSITORY" ]] || { echo "Missing repository: $REPOSITORY" >&2; exit 1; }
[[ -f "$INVENTORY_CSV" ]] || { echo "Missing inventory: $INVENTORY_CSV" >&2; exit 1; }
[[ -f "$VIRTUAL_ENVIRONMENT/bin/activate" ]] || {
    echo "Missing virtual environment: $VIRTUAL_ENVIRONMENT" >&2
    exit 1
}

# Keep numerical libraries single-threaded: recording-level parallelism is
# controlled explicitly by --n-jobs.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:?}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"

python -u scripts/preprocessing/run_raw_line_noise_qc.py \
    --inventory-csv "$INVENTORY_CSV" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIRECTORY" \
    --datasets ucla hup \
    --window-seconds 8 \
    --overlap-seconds 0 \
    --duration-per-recording 600 \
    --n-jobs "${SLURM_CPUS_PER_TASK:-8}" \
    --log-level INFO
