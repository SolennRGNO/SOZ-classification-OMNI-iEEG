#!/usr/bin/env bash
#SBATCH --job-name=hup_ucla_qc_full
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.err

set -euo pipefail

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
[[ -d "$REPOSITORY" ]] || { echo "Missing repository: $REPOSITORY" >&2; exit 1; }
cd "$REPOSITORY"

export UV_CACHE_DIR="$SCRATCH/.cache/uv"
source "$SCRATCH/venvs/neurodags/bin/activate"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

OUTPUT_DIR="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/full_nonictal_v1/summary"
OUTPUT_HTML="${OUTPUT_DIR}/hup_ucla_preprocessing_qc_full.html"

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
echo "Output: ${OUTPUT_HTML}"
echo "========================================"

SCRIPT="scripts/preprocessing/aggregate_qc.py"

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: script not found: $(pwd)/${SCRIPT}"
    exit 1
fi

python -m py_compile "$SCRIPT"

# The cohort is v3 recomputed derivatives PLUS derivatives reused in place from
# the read-only v2 root; both must be in the report or the counts under-report.
REUSE_ROOT="${OMNI_REUSE_PREPROCESSED_ROOT:-/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v2/full_nonictal_v1}"
qc_arguments=(--output-html "$OUTPUT_HTML")
qc_arguments+=(--preprocessed-root "/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/full_nonictal_v1")
if [[ -d "$REUSE_ROOT" ]]; then
    qc_arguments+=(--reuse-root "$REUSE_ROOT")
    echo "Including read-only reused derivatives from $REUSE_ROOT"
fi

python -u "$SCRIPT" "${qc_arguments[@]}"

echo "========================================"
echo "Finished: $(date)"
echo "Report generated:"
ls -lh "$OUTPUT_HTML"
echo "========================================"
