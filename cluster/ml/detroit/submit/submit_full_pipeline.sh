#!/usr/bin/env bash
# Detroit/OpenIEEG launcher. Plans by default; submits only with --submit.
set -euo pipefail
umask 027
readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly ANALYSIS_ROOT="/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1"
readonly FEATURE_ROOT="$ANALYSIS_ROOT/features_v1"
experiment="${1:-}"; shift || true
case "$experiment" in baseline|b|c1|c1_nested|c2|c1_nested_tuned|random_forest_nested) ;; *) echo "Usage: $0 {baseline|b|c1 N|c1_nested|c2|c1_nested_tuned [minimum_tpr]|random_forest_nested} [--submit]" >&2; exit 2;; esac
selected=""; minimum_tpr="0.50"
if [[ "$experiment" == c1 ]]; then selected="${1:-}"; [[ "$selected" =~ ^[1-9][0-9]*$ ]] || { echo "c1 requires N" >&2; exit 2; }; shift; fi
if [[ "$experiment" == c1_nested_tuned && "${1:-}" != --submit && -n "${1:-}" ]]; then minimum_tpr="$1"; shift; fi
[[ "$#" -eq 0 || ( "$#" -eq 1 && "$1" == --submit ) ]] || { echo "invalid arguments" >&2; exit 2; }
submit="${1:-}"
readonly DATA_ROOT="$ANALYSIS_ROOT/ml_detroit_datasets_v1"
case "$experiment" in
 baseline) ROOT="$ANALYSIS_ROOT/ml_logistic_baseline_detroit_v1"; kind=baseline ;;
 b|c1|c2) ROOT="$ANALYSIS_ROOT/ml_logistic_complexity_detroit_v1/$experiment"; kind=complexity ;;
 c1_nested) ROOT="$ANALYSIS_ROOT/ml_logistic_complexity_c1_nested_detroit_v1"; kind=complexity ;;
 c1_nested_tuned) ROOT="$ANALYSIS_ROOT/ml_logistic_complexity_c1_nested_tuned_detroit_v1"; kind=complexity ;;
 random_forest_nested) ROOT="$ANALYSIS_ROOT/ml_random_forest_complexity_nested_detroit_v1"; kind=complexity ;;
esac
DATASET_DIR="$DATA_ROOT/$kind"; RESULTS_DIR="$ROOT/results"; REPORT="$ROOT/${experiment}_report.html"
inputs=("$FEATURE_ROOT/relative_bandpower_neurodags_full_cohort_v1/relative_bandpower_channels_detroit.parquet" "$FEATURE_ROOT/time_domain_neurodags_full_cohort_v1/time_domain_channels_detroit.parquet" "$FEATURE_ROOT/frequency_domain_neurodags_full_cohort_v1/frequency_domain_channels_detroit.parquet" "$FEATURE_ROOT/complexity_neurodags_full_cohort_v1/complexity_channels_detroit.parquet")
[[ -z "${SLURM_JOB_ID:-}" && -d "$REPOSITORY/.git" ]] || { echo "run on login node with repository available" >&2; exit 1; }
for input in "${inputs[@]}"; do [[ -f "$input" ]] || { echo "missing Detroit input: $input" >&2; exit 1; }; done
[[ ! -e "$RESULTS_DIR" && ! -e "$REPORT" ]] || { echo "refusing to overwrite $ROOT" >&2; exit 1; }
echo "Planned unchanged HUP/UCLA evaluator for Detroit: $experiment"
echo "Dataset root: $DATA_ROOT | results: $RESULTS_DIR | report: $REPORT"
echo "Exact submit command: bash cluster/ml/detroit/submit/submit_full_pipeline.sh $experiment${selected:+ $selected} --submit"
[[ "$submit" == --submit ]] || exit 0
mkdir -p "$ANALYSIS_ROOT/logs"
dataset_id=""
if [[ ! -d "$DATASET_DIR" ]]; then dataset_id="$(sbatch --parsable cluster/ml/detroit/stages/build_dataset.sh)"; dataset_id="${dataset_id%%;*}"; fi
model_id="$(sbatch --parsable ${dataset_id:+--dependency=afterok:$dataset_id} cluster/ml/detroit/stages/run_model.sh "$experiment" "$selected" "$minimum_tpr")"; model_id="${model_id%%;*}"
report_id="$(sbatch --parsable --dependency="afterok:$model_id" cluster/ml/detroit/stages/generate_report.sh "$experiment")"; report_id="${report_id%%;*}"
echo "Submitted dataset=${dataset_id:-reused} model=$model_id report=$report_id"
