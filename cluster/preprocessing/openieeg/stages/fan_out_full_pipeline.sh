#!/usr/bin/env bash
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --job-name=openieeg_fan_out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/openieeg_fan_out_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/openieeg_fan_out_%j.err

# Short controller: once the manifests exist it knows the exact array size, so
# it submits the subject array and the dependent scale-out summary.

set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly ANALYSIS_ROOT="/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1"
readonly MANIFEST="$ANALYSIS_ROOT/manifests/full_v1/openieeg_full_subjects.csv"
readonly OUTPUT_ROOT="$ANALYSIS_ROOT/full_v1/openieeg"

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Submit this controller with sbatch." >&2; exit 2; }
cd "$REPOSITORY"
[[ -f "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 1; }

array_output=$(bash cluster/preprocessing/openieeg/submit/submit_openieeg_subject_array.sh \
    --manifest "$MANIFEST" --output-root "$OUTPUT_ROOT" --mode full)
echo "$array_output"
array_job_id=$(sed -n 's/.*array job ID: \([0-9]\+\).*/\1/p' <<<"$array_output")
[[ -n "$array_job_id" ]] || { echo "Could not parse array job ID." >&2; exit 1; }

summary_submission=$(sbatch --parsable --dependency="afterany:${array_job_id}" \
    cluster/preprocessing/openieeg/stages/summarize_scaleout.sh)
echo "Scale-out summary: ${summary_submission%%;*} (afterany array $array_job_id)"
