#!/usr/bin/env bash
set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly OUTPUT_DIR="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run this lightweight submitter on the login node, not inside a job." >&2
    exit 2
fi

[[ -e "$REPOSITORY/.git" ]] || {
    echo "ERROR: expected Git repository is missing: $REPOSITORY" >&2
    exit 1
}
cd "$REPOSITORY"
mkdir -p "$OUTPUT_DIR/logs"

required_files=(
    cluster/preprocessing/stages/build_inventory.sh
    cluster/preprocessing/stages/build_nonictal_feature_cohort.sh
    cluster/preprocessing/stages/raw_line_noise_qc.sh
    cluster/preprocessing/stages/build_subject_manifests.sh
    cluster/preprocessing/stages/fan_out_full_pipeline.sh
    cluster/preprocessing/stages/preprocess_hup_subject.sh
    cluster/preprocessing/stages/preprocess_ucla_subject.sh
    cluster/preprocessing/stages/summarize_scaleout.sh
    cluster/preprocessing/stages/generate_qc_report.sh
    scripts/preprocessing/build_subject_manifests.py
    scripts/preprocessing/summarize_scaleout.py
    scripts/preprocessing/aggregate_qc.py
    Neurodags/pipeline_hup_ucla_1khz.yml
    Neurodags/config/line_noise_hup_ucla.yml
)
for path in "${required_files[@]}"; do
    [[ -f "$path" ]] || { echo "Missing required workflow file: $path" >&2; exit 1; }
done

inventory_submission=$(sbatch --parsable cluster/preprocessing/stages/build_inventory.sh)
inventory_job_id="${inventory_submission%%;*}"

cohort_submission=$(sbatch --parsable \
    --dependency="afterok:${inventory_job_id}" \
    cluster/preprocessing/stages/build_nonictal_feature_cohort.sh)
cohort_job_id="${cohort_submission%%;*}"

raw_qc_submission=$(sbatch --parsable \
    --dependency="afterok:${inventory_job_id}" \
    cluster/preprocessing/stages/raw_line_noise_qc.sh)
raw_qc_job_id="${raw_qc_submission%%;*}"

manifests_submission=$(sbatch --parsable \
    --dependency="afterok:${cohort_job_id}" \
    cluster/preprocessing/stages/build_subject_manifests.sh)
manifests_job_id="${manifests_submission%%;*}"

# This short controller waits for both the manifests and raw line-noise QC.
# It then derives exact array sizes and submits all remaining dependent jobs.
fanout_submission=$(sbatch --parsable \
    --dependency="afterok:${manifests_job_id}:${raw_qc_job_id}" \
    cluster/preprocessing/stages/fan_out_full_pipeline.sh)
fanout_job_id="${fanout_submission%%;*}"

echo "Full-cohort HUP/UCLA workflow submitted."
echo "Inventory: $inventory_job_id"
echo "Feature cohort (interictal + ictal): $cohort_job_id (afterok inventory)"
echo "Raw line-noise QC: $raw_qc_job_id (afterok inventory)"
echo "Subject manifests: $manifests_job_id (afterok cohort)"
echo "Array fan-out controller: $fanout_job_id (afterok manifests and raw QC)"
echo "The fan-out log will report the HUP array, UCLA array, summary, and final QC job IDs."
