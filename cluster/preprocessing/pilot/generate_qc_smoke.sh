#!/usr/bin/env bash
#SBATCH --job-name=hup_ucla_qc_smoke
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.err

set -euo pipefail

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
[[ -d "$REPOSITORY" ]] || { echo "Missing repository: $REPOSITORY" >&2; exit 1; }
cd "$REPOSITORY"

# Environnement NeuroDAGs installé sur scratch
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
source "$SCRATCH/venvs/neurodags/bin/activate"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

OUTPUT_DIR="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/full_nonictal_v1/summary"
OUTPUT_HTML="${OUTPUT_DIR}/hup_ucla_preprocessing_qc_smoke.html"

mkdir -p "/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs"
mkdir -p "$UV_CACHE_DIR"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Working directory: $(pwd)"
echo "SCRATCH: ${SCRATCH}"
echo "Virtual environment: ${VIRTUAL_ENVIRONMENT:-$SCRATCH/venvs/neurodags}"
echo "Python: $(which python)"
python --version
echo "========================================"

if [[ ! -f scripts/preprocessing/aggregate_qc.py ]]; then
    echo "ERROR: script not found:"
    echo "$(pwd)/scripts/preprocessing/aggregate_qc.py"
    exit 1
fi

python -u scripts/preprocessing/aggregate_qc.py \
    --max-subjects-per-dataset 2 \
    --output-html "$OUTPUT_HTML"

echo "========================================"
echo "Finished: $(date)"
echo "Report: ${OUTPUT_HTML}"
ls -lh "$OUTPUT_HTML"
echo "========================================"
