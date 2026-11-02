#!/usr/bin/env bash
#SBATCH --job-name=omni_detroit_report
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/%x_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/%x_%j.err
set -euo pipefail
readonly REPOSITORY=/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor ANALYSIS_ROOT=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1 VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
experiment="${1:?}"; kind=complexity
case "$experiment" in baseline) kind=baseline; ROOT="$ANALYSIS_ROOT/ml_logistic_baseline_detroit_v1";; b|c1|c2) ROOT="$ANALYSIS_ROOT/ml_logistic_complexity_detroit_v1/$experiment";; c1_nested) ROOT="$ANALYSIS_ROOT/ml_logistic_complexity_c1_nested_detroit_v1";; c1_nested_tuned) ROOT="$ANALYSIS_ROOT/ml_logistic_complexity_c1_nested_tuned_detroit_v1";; random_forest_nested) ROOT="$ANALYSIS_ROOT/ml_random_forest_complexity_nested_detroit_v1";; *) exit 2;; esac
DATASET_DIR="$ANALYSIS_ROOT/ml_detroit_datasets_v1/$kind"; REPORT="$ROOT/${experiment}_report.html"; [[ -n "${SLURM_JOB_ID:-}" && -x "$VIRTUAL_ENVIRONMENT/bin/python" && ! -e "$REPORT" ]] || exit 1
source "$VIRTUAL_ENVIRONMENT/bin/activate"; cd "$REPOSITORY"; export PYTHONPATH="$REPOSITORY" TMPDIR="${SLURM_TMPDIR:?}"
if [[ "$experiment" == random_forest_nested ]]; then python -u scripts/ml/random_forest_report.py --dataset-dir "$DATASET_DIR" --results-dir "$ROOT/results" --output-html "$REPORT"; elif [[ "$experiment" == c1_nested_tuned ]]; then python -u scripts/ml/tuned_threshold_report.py --dataset-dir "$DATASET_DIR" --results-dir "$ROOT/results" --output-html "$REPORT"; else python -u scripts/ml/generate_report.py --dataset-dir "$DATASET_DIR" --results-dir "$ROOT/results" --output-html "$REPORT"; fi
