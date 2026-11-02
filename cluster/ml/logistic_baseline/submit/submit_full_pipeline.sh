#!/usr/bin/env bash
# Lightweight login-node submitter. It is never run automatically.
# Exactly one task per invocation: interictal and ictal are separate experiments.
set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly ANALYSIS_ROOT="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3"

task="${1:-}"
case "$task" in
    interictal|ictal) ;;
    *)
        echo "Usage: $0 {interictal|ictal} [--submit]" >&2
        echo "There is no 'both' task; run each experiment separately." >&2
        exit 2
        ;;
esac
shift

readonly TASK_ROOT="$ANALYSIS_ROOT/ml_logistic_baseline_v1/$task"

[[ -z "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: run this submitter on a login node." >&2; exit 2; }
[[ -d "$REPOSITORY/.git" ]] || { echo "Missing repository: $REPOSITORY" >&2; exit 1; }
cd "$REPOSITORY"
for input in \
    "$ANALYSIS_ROOT/feature_cohort/nonictal_feature_cohort_hup_ucla.csv" \
    "$ANALYSIS_ROOT/feature_cohort/channel_label_audit_hup_ucla.csv" \
    "$ANALYSIS_ROOT/time_domain_neurodags_cohort_v1/time_domain_epochs_hup_ucla.parquet" \
    "$ANALYSIS_ROOT/relative_bandpower_neurodags_cohort_v1/relative_bandpower_epochs_hup_ucla.parquet" \
    "$ANALYSIS_ROOT/frequency_domain_neurodags_cohort_v1/frequency_domain_epochs_hup_ucla.parquet"
do
    [[ -f "$input" ]] || { echo "Missing ML input: $input" >&2; exit 1; }
done
for path in \
    scripts/ml/{common,dataset,logistic_baseline,generate_report}.py \
    cluster/ml/logistic_baseline/stages/{build_dataset,run_baseline,generate_report}.sh
do
    [[ -f "$path" ]] || { echo "Missing workflow file: $path" >&2; exit 1; }
done
readonly DATASET_DIR="$TASK_ROOT/dataset"
readonly RESULTS_DIR="$TASK_ROOT/results"
readonly REPORT="$TASK_ROOT/logistic_baseline_report.html"
dataset_ready=false
if [[ -d "$DATASET_DIR" && -f "$DATASET_DIR/ml_participant_channel_dataset.parquet" && -f "$DATASET_DIR/feature_list.json" ]]; then
    dataset_ready=true
elif [[ -e "$DATASET_DIR" ]]; then
    echo "Existing ML dataset directory is incomplete; refusing to continue: $DATASET_DIR" >&2
    exit 1
fi
[[ ! -e "$RESULTS_DIR" && ! -e "$REPORT" ]] || { echo "Model/report outputs already exist; refusing to overwrite." >&2; exit 1; }

echo "Planned logistic baseline v1 submission"
echo "Baseline families: relative_bandpower + time_domain + frequency_domain (32 epoch features -> 96 predictors)"
echo "Held out: complexity (extracted and stored, reserved for a separate model)"
echo "Task: $task (single recording state; ictal and interictal are never pooled)"
echo "Account: CONFIGURE_SLURM_ACCOUNT"
echo "Partition/GPU: default CPU partition; no GPU requested"
if [[ "$dataset_ready" == true ]]; then
    echo "Dataset: reuse validated existing output $DATASET_DIR"
else
    echo "Dataset: 8 CPUs, 128G, 08:00:00; output $DATASET_DIR"
fi
echo "Model: 4 CPUs, 16G, 04:00:00; output $RESULTS_DIR"
echo "Report: 1 CPU, 8G, 00:30:00; output $REPORT"
echo "Dependency chain: dataset -> model -> report (afterok)"
echo "Exact submit command: bash cluster/ml/logistic_baseline/submit/submit_full_pipeline.sh $task --submit"

if [[ "${1:-}" != "--submit" ]]; then
    echo "Plan only: no jobs submitted. Re-run with --submit after reviewing the plan."
    exit 0
fi
[[ "$#" -eq 1 ]] || { echo "Usage: $0 {interictal|ictal} [--submit]" >&2; exit 2; }

mkdir -p "$ANALYSIS_ROOT/logs"
dataset_id=""
if [[ "$dataset_ready" == false ]]; then
    dataset_submission=$(sbatch --parsable cluster/ml/logistic_baseline/stages/build_dataset.sh "$task")
    dataset_id="${dataset_submission%%;*}"
fi
if [[ -n "$dataset_id" ]]; then
    model_submission=$(sbatch --parsable --dependency="afterok:${dataset_id}" cluster/ml/logistic_baseline/stages/run_baseline.sh "$task")
else
    model_submission=$(sbatch --parsable cluster/ml/logistic_baseline/stages/run_baseline.sh "$task")
fi
model_id="${model_submission%%;*}"
report_submission=$(sbatch --parsable --dependency="afterok:${model_id}" cluster/ml/logistic_baseline/stages/generate_report.sh "$task")
report_id="${report_submission%%;*}"

[[ -n "$dataset_id" ]] && echo "Submitted dataset job: $dataset_id" || echo "Dataset stage: reused existing validated artifacts"
[[ -n "$dataset_id" ]] && echo "Submitted model job: $model_id (afterok $dataset_id)" || echo "Submitted model job: $model_id"
echo "Submitted report job: $report_id (afterok $model_id)"
