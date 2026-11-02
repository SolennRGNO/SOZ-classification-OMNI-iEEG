#!/usr/bin/env bash
#SBATCH --job-name=omni_ml_logistic_v1
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
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
readonly RESULTS_DIR="$TASK_ROOT/results"

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: submit with sbatch; do not run on a login node." >&2; exit 2; }
[[ -x "$VIRTUAL_ENVIRONMENT/bin/python" && -f "$DATASET_DIR/ml_participant_channel_dataset.parquet" && -f "$DATASET_DIR/feature_list.json" ]] || { echo "Missing ML environment or dataset inputs." >&2; exit 1; }
[[ ! -e "$RESULTS_DIR" ]] || { echo "Results output already exists: $RESULTS_DIR" >&2; exit 1; }
source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"
export PYTHONPATH="$REPOSITORY${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${SLURM_TMPDIR:?}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -u scripts/ml/logistic_baseline.py \
    --dataset "$DATASET_DIR/ml_participant_channel_dataset.parquet" \
    --feature-list "$DATASET_DIR/feature_list.json" \
    --output-dir "$RESULTS_DIR" \
    --repository "$REPOSITORY" \
    --n-bootstrap 5000
