#!/usr/bin/env bash
#SBATCH --job-name=omni_detroit_ml
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/%x_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/%x_%j.err
set -euo pipefail
readonly REPOSITORY=/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor ANALYSIS_ROOT=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1 VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
experiment="${1:?}"; selected="${2:-}"; minimum_tpr="${3:-0.50}"; kind=complexity
case "$experiment" in baseline) kind=baseline; ROOT="$ANALYSIS_ROOT/ml_logistic_baseline_detroit_v1";; b|c1|c2) ROOT="$ANALYSIS_ROOT/ml_logistic_complexity_detroit_v1/$experiment";; c1_nested) ROOT="$ANALYSIS_ROOT/ml_logistic_complexity_c1_nested_detroit_v1";; c1_nested_tuned) ROOT="$ANALYSIS_ROOT/ml_logistic_complexity_c1_nested_tuned_detroit_v1";; random_forest_nested) ROOT="$ANALYSIS_ROOT/ml_random_forest_complexity_nested_detroit_v1";; *) exit 2;; esac
DATASET_DIR="$ANALYSIS_ROOT/ml_detroit_datasets_v1/$kind"; [[ -n "${SLURM_JOB_ID:-}" && -x "$VIRTUAL_ENVIRONMENT/bin/python" && ! -e "$ROOT/results" ]] || exit 1
source "$VIRTUAL_ENVIRONMENT/bin/activate"; cd "$REPOSITORY"; export PYTHONPATH="$REPOSITORY" TMPDIR="${SLURM_TMPDIR:?}" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
feature_list="$DATASET_DIR/complexity_feature_list.json"; dataset="$DATASET_DIR/ml_complexity_participant_channel_dataset.parquet"; [[ "$kind" == baseline ]] && { feature_list="$DATASET_DIR/feature_list.json"; dataset="$DATASET_DIR/ml_participant_channel_dataset.parquet"; }
args=(--experiment "$experiment" --dataset "$dataset" --feature-list "$feature_list" --output-dir "$ROOT/results" --repository "$REPOSITORY" --minimum-tpr "$minimum_tpr" --n-jobs "$SLURM_CPUS_PER_TASK")
[[ "$experiment" == c1 ]] && args+=(--n-selected-features "$selected")
python -u scripts/ml/run_detroit_ml.py "${args[@]}"
