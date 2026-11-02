#!/usr/bin/env bash
#SBATCH --job-name=omni_ml_report_v1
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
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
readonly REPORT="$TASK_ROOT/logistic_baseline_report.html"

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: submit with sbatch; do not run on a login node." >&2; exit 2; }
[[ -x "$VIRTUAL_ENVIRONMENT/bin/python" && -d "$DATASET_DIR" && -d "$RESULTS_DIR" ]] || { echo "Missing report inputs." >&2; exit 1; }
[[ ! -e "$REPORT" ]] || { echo "Report already exists: $REPORT" >&2; exit 1; }
source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"
export PYTHONPATH="$REPOSITORY${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${SLURM_TMPDIR:?}"

python -u scripts/ml/generate_report.py \
    --dataset-dir "$DATASET_DIR" \
    --results-dir "$RESULTS_DIR" \
    --output-html "$REPORT"
