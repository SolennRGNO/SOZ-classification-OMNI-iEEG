#!/usr/bin/env bash
#SBATCH --job-name=omni_ml_dataset_v1
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.err

set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
readonly ANALYSIS_ROOT="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3"

task="${1:-}"
case "$task" in
    interictal|ictal) ;;
    *) echo "Usage: $0 {interictal|ictal}; exactly one task per experiment." >&2; exit 2 ;;
esac

readonly TASK_ROOT="$ANALYSIS_ROOT/ml_logistic_baseline_v1/$task"
readonly DATASET_DIR="$TASK_ROOT/dataset"

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: submit with sbatch; do not run on a login node." >&2; exit 2; }
[[ -x "$VIRTUAL_ENVIRONMENT/bin/python" && -d "$REPOSITORY/.git" ]] || { echo "Missing repository or Python environment." >&2; exit 1; }
[[ ! -e "$DATASET_DIR" ]] || { echo "Dataset output already exists: $DATASET_DIR" >&2; exit 1; }
mkdir -p "$TASK_ROOT" "$ANALYSIS_ROOT/logs"
source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"
export PYTHONPATH="$REPOSITORY${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${SLURM_TMPDIR:?}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -u scripts/ml/dataset.py \
    --task "$task" \
    --output-dir "$DATASET_DIR" \
    --repository "$REPOSITORY"
