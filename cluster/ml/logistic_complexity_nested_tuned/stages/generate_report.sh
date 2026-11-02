#!/usr/bin/env bash
#SBATCH --job-name=omni_cxnt_report
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
[[ "$task" =~ ^(interictal|ictal)$ ]] || { echo "Usage: $0 {interictal|ictal}" >&2; exit 2; }
readonly ROOT="$ANALYSIS_ROOT/ml_logistic_complexity_c1_nested_tuned_v1/$task"
readonly DATASET_DIR="$ROOT/dataset" RESULTS_DIR="$ROOT/results" REPORT="$ROOT/logistic_complexity_c1_nested_tuned_report.html"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: submit with sbatch; do not run on a login node." >&2; exit 2; }
[[ -x "$VIRTUAL_ENVIRONMENT/bin/python" && -d "$DATASET_DIR" && -d "$RESULTS_DIR" && ! -e "$REPORT" ]] || { echo "Missing report inputs or existing report output." >&2; exit 1; }
source "$VIRTUAL_ENVIRONMENT/bin/activate"; cd "$REPOSITORY"
export PYTHONPATH="$REPOSITORY${PYTHONPATH:+:$PYTHONPATH}" TMPDIR="${SLURM_TMPDIR:?}"
python -u scripts/ml/tuned_threshold_report.py --dataset-dir "$DATASET_DIR" --results-dir "$RESULTS_DIR" --output-html "$REPORT"
