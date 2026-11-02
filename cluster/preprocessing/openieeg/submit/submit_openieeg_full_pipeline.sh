#!/usr/bin/env bash
set -euo pipefail
umask 027

# Full OpenIEEG 1 kHz preprocessing: inventory -> subject manifests -> array
# fan-out -> scale-out summary. Submit only after the single-recording pilot has
# been verified.
#
# Run on the login node (it only calls sbatch):
#   bash cluster/preprocessing/openieeg/submit/submit_openieeg_full_pipeline.sh

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly ANALYSIS_ROOT="/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run this lightweight submitter on the login node, not inside a job." >&2
    exit 2
fi

[[ -e "$REPOSITORY/.git" ]] || { echo "ERROR: missing Git repository: $REPOSITORY" >&2; exit 1; }
cd "$REPOSITORY"
mkdir -p "$ANALYSIS_ROOT/logs"

required_files=(
    cluster/preprocessing/openieeg/stages/build_inventory.sh
    cluster/preprocessing/openieeg/stages/build_subject_manifests.sh
    cluster/preprocessing/openieeg/stages/preprocess_openieeg_subject.sh
    cluster/preprocessing/openieeg/stages/summarize_scaleout.sh
    cluster/preprocessing/openieeg/submit/submit_openieeg_subject_array.sh
    scripts/preprocessing/build_inventory.py
    scripts/preprocessing/build_subject_manifests.py
    scripts/preprocessing/summarize_scaleout.py
    Neurodags/pipeline_openieeg_1khz.yml
    Neurodags/config/line_noise_openieeg.yml
)
for path in "${required_files[@]}"; do
    [[ -f "$path" ]] || { echo "Missing required workflow file: $path" >&2; exit 1; }
done

inventory_job_id=""
if [[ -s "$ANALYSIS_ROOT/inventory/eligible_recordings_openieeg.csv" ]]; then
    echo "Reusing existing inventory: $ANALYSIS_ROOT/inventory/eligible_recordings_openieeg.csv"
else
    submission=$(sbatch --parsable cluster/preprocessing/openieeg/stages/build_inventory.sh)
    inventory_job_id="${submission%%;*}"
    echo "Inventory: $inventory_job_id"
fi

manifest_dependency=()
[[ -n "$inventory_job_id" ]] && manifest_dependency=(--dependency="afterok:${inventory_job_id}")
submission=$(sbatch --parsable "${manifest_dependency[@]}" \
    cluster/preprocessing/openieeg/stages/build_subject_manifests.sh)
manifests_job_id="${submission%%;*}"
echo "Subject manifests: $manifests_job_id"

submission=$(sbatch --parsable --dependency="afterok:${manifests_job_id}" \
    cluster/preprocessing/openieeg/stages/fan_out_full_pipeline.sh)
fanout_job_id="${submission%%;*}"
echo "Array fan-out controller: $fanout_job_id (afterok manifests)"
echo "The fan-out log reports the OpenIEEG array and scale-out summary job IDs."
