#!/usr/bin/env bash
# Login-node submitter for the C1-nested experiment. Plans by default and submits only with --submit.
# It never reads or writes the ml_logistic_complexity_v1 tree that holds B, C1 and C2.
set -euo pipefail
umask 027
readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly ANALYSIS_ROOT="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3"

task="${1:-}"
[[ "$task" =~ ^(interictal|ictal)$ ]] || { echo "Usage: $0 {interictal|ictal} [--submit]" >&2; exit 2; }
shift
[[ "$#" -eq 0 || ( "$#" -eq 1 && "$1" == "--submit" ) ]] || { echo "Usage: $0 {interictal|ictal} [--submit]" >&2; exit 2; }
submit="${1:-}"
readonly ROOT="$ANALYSIS_ROOT/ml_logistic_complexity_c1_nested_v1/$task"
readonly DATASET_DIR="$ROOT/dataset" RESULTS_DIR="$ROOT/results" REPORT="$ROOT/logistic_complexity_c1_nested_report.html"
[[ -z "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: run this submitter on a login node." >&2; exit 2; }
[[ -d "$REPOSITORY/.git" ]] || { echo "Missing repository: $REPOSITORY" >&2; exit 1; }
for input in "$ANALYSIS_ROOT/feature_cohort/nonictal_feature_cohort_hup_ucla.csv" "$ANALYSIS_ROOT/feature_cohort/channel_label_audit_hup_ucla.csv" "$ANALYSIS_ROOT/complexity_neurodags_cohort_v1/complexity_epochs_hup_ucla.parquet"; do [[ -f "$input" ]] || { echo "Missing Complexity ML input: $input" >&2; exit 1; }; done
cd "$REPOSITORY"
for path in scripts/ml/{common,complexity_dataset,logistic_complexity,nested_mrmr,generate_report}.py cluster/ml/logistic_complexity_nested/stages/{build_dataset,run_model,generate_report}.sh; do [[ -f "$path" ]] || { echo "Missing workflow file: $path" >&2; exit 1; }; done
[[ ! -e "$DATASET_DIR" && ! -e "$RESULTS_DIR" && ! -e "$REPORT" ]] || { echo "Outputs already exist for $task/c1_nested; refusing to overwrite." >&2; exit 1; }
echo "Planned Complexity C1-nested ML workflow: $task"
echo "Experiment: ml_logistic_complexity_c1_nested_v1 (nested-CV mRMR, N tuned per outer fold)"
echo "N grid: 5 8 10 15 20 30 | outer CV: 5 grouped folds | inner CV: 3 grouped folds | tuning metric: average_precision"
echo "Dataset: 8 CPUs, 128G, 08:00:00 -> $DATASET_DIR"
echo "Model: 8 CPUs, 32G, 12:00:00 -> $RESULTS_DIR"
echo "Report: 1 CPU, 8G, 00:30:00 -> $REPORT"
echo "Dependency chain: dataset -> model -> report (afterok)"
echo "Untouched by this workflow: $ANALYSIS_ROOT/ml_logistic_complexity_v1 (models B, C1, C2)"
echo "Exact submit command: bash cluster/ml/logistic_complexity_nested/submit/submit_full_pipeline.sh $task --submit"
[[ "$submit" == "--submit" ]] || { echo "Plan only: no jobs submitted. Re-run with --submit after review."; exit 0; }
mkdir -p "$ANALYSIS_ROOT/logs"
dataset_id="$(sbatch --parsable cluster/ml/logistic_complexity_nested/stages/build_dataset.sh "$task")"; dataset_id="${dataset_id%%;*}"
model_id="$(sbatch --parsable --dependency="afterok:${dataset_id}" cluster/ml/logistic_complexity_nested/stages/run_model.sh "$task")"; model_id="${model_id%%;*}"
report_id="$(sbatch --parsable --dependency="afterok:${model_id}" cluster/ml/logistic_complexity_nested/stages/generate_report.sh "$task")"; report_id="${report_id%%;*}"
echo "Submitted dataset job: $dataset_id"
echo "Submitted model job: $model_id (afterok $dataset_id)"
echo "Submitted report job: $report_id (afterok $model_id)"
