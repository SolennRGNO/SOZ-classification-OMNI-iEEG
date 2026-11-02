#!/usr/bin/env bash
# Login-node submitter. Plans by default and submits only with --submit.
set -euo pipefail
umask 027
readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly ANALYSIS_ROOT="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3"

task="${1:-}"; experiment="${2:-}"; selected=""
[[ "$task" =~ ^(interictal|ictal)$ && "$experiment" =~ ^(b|c1|c2)$ ]] || { echo "Usage: $0 {interictal|ictal} {b|c1|c2} [n-selected-features for c1] [--submit]" >&2; exit 2; }
shift 2
if [[ "$experiment" == c1 ]]; then
    selected="${1:-}"; [[ "$selected" =~ ^[1-9][0-9]*$ ]] || { echo "C1 requires a positive n-selected-features." >&2; exit 2; }; shift
fi
[[ "$#" -eq 0 || ( "$#" -eq 1 && "$1" == "--submit" ) ]] || { echo "Usage: $0 {interictal|ictal} {b|c1|c2} [n-selected-features for c1] [--submit]" >&2; exit 2; }
submit="${1:-}"
readonly ROOT="$ANALYSIS_ROOT/ml_logistic_complexity_v1/$task/$experiment"
readonly DATASET_DIR="$ROOT/dataset" RESULTS_DIR="$ROOT/results" REPORT="$ROOT/logistic_complexity_${experiment}_report.html"
[[ -z "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: run this submitter on a login node." >&2; exit 2; }
[[ -d "$REPOSITORY/.git" ]] || { echo "Missing repository: $REPOSITORY" >&2; exit 1; }
for input in "$ANALYSIS_ROOT/feature_cohort/nonictal_feature_cohort_hup_ucla.csv" "$ANALYSIS_ROOT/feature_cohort/channel_label_audit_hup_ucla.csv" "$ANALYSIS_ROOT/complexity_neurodags_cohort_v1/complexity_epochs_hup_ucla.parquet"; do [[ -f "$input" ]] || { echo "Missing Complexity ML input: $input" >&2; exit 1; }; done
cd "$REPOSITORY"
for path in scripts/ml/{common,complexity_dataset,logistic_complexity,generate_report}.py cluster/ml/logistic_complexity/stages/{build_dataset,run_model,generate_report}.sh; do [[ -f "$path" ]] || { echo "Missing workflow file: $path" >&2; exit 1; }; done
[[ ! -e "$DATASET_DIR" && ! -e "$RESULTS_DIR" && ! -e "$REPORT" ]] || { echo "Outputs already exist for $task/$experiment; refusing to overwrite." >&2; exit 1; }
echo "Planned Complexity ML workflow: $task / $experiment"
echo "Dataset: 8 CPUs, 128G, 08:00:00 -> $DATASET_DIR"
echo "Model: 4 CPUs, 16G, 04:00:00 -> $RESULTS_DIR"
echo "Report: 1 CPU, 8G, 00:30:00 -> $REPORT"
echo "Dependency chain: dataset -> model -> report (afterok)"
[[ "$experiment" != c1 ]] || echo "C1 selected predictors per outer fold: $selected"
echo "Exact submit command: bash cluster/ml/logistic_complexity/submit/submit_full_pipeline.sh $task $experiment${selected:+ $selected} --submit"
[[ "$submit" == "--submit" ]] || { echo "Plan only: no jobs submitted. Re-run with --submit after review."; exit 0; }
mkdir -p "$ANALYSIS_ROOT/logs"
dataset_id="$(sbatch --parsable cluster/ml/logistic_complexity/stages/build_dataset.sh "$task" "$experiment")"; dataset_id="${dataset_id%%;*}"
model_id="$(sbatch --parsable --dependency="afterok:${dataset_id}" cluster/ml/logistic_complexity/stages/run_model.sh "$task" "$experiment" "$selected")"; model_id="${model_id%%;*}"
report_id="$(sbatch --parsable --dependency="afterok:${model_id}" cluster/ml/logistic_complexity/stages/generate_report.sh "$task" "$experiment")"; report_id="${report_id%%;*}"
echo "Submitted dataset job: $dataset_id"
echo "Submitted model job: $model_id (afterok $dataset_id)"
echo "Submitted report job: $report_id (afterok $model_id)"
