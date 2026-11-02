#!/usr/bin/env bash
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --job-name=hup_ucla_fanout
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/hup_ucla_fanout_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/hup_ucla_fanout_%j.err

set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly PYTHON="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags/bin/python"
readonly MANIFEST_DIRECTORY="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/manifests/full_nonictal_v1"
readonly HUP_MANIFEST="$MANIFEST_DIRECTORY/hup_full_subjects.csv"
readonly UCLA_MANIFEST="$MANIFEST_DIRECTORY/ucla_full_subjects.csv"
readonly OUTPUT_ROOT="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/full_nonictal_v1"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: this fan-out controller must be submitted with sbatch." >&2
    exit 2
fi

[[ -d "$REPOSITORY" ]] || { echo "Missing repository: $REPOSITORY" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
[[ -f "$HUP_MANIFEST" ]] || { echo "Missing HUP manifest: $HUP_MANIFEST" >&2; exit 1; }
[[ -f "$UCLA_MANIFEST" ]] || { echo "Missing UCLA manifest: $UCLA_MANIFEST" >&2; exit 1; }

cd "$REPOSITORY"
export PYTHONPATH="$REPOSITORY${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p /CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs "$OUTPUT_ROOT/hup" "$OUTPUT_ROOT/ucla"

hup_count=$("$PYTHON" scripts/preprocessing/build_subject_manifests.py count --manifest "$HUP_MANIFEST")
ucla_count=$("$PYTHON" scripts/preprocessing/build_subject_manifests.py count --manifest "$UCLA_MANIFEST")
(( hup_count > 0 )) || { echo "HUP manifest is empty." >&2; exit 1; }
(( ucla_count > 0 )) || { echo "UCLA manifest is empty." >&2; exit 1; }

hup_submission=$(sbatch --parsable --array="0-$((hup_count - 1))" \
    cluster/preprocessing/stages/preprocess_hup_subject.sh \
    --manifest "$HUP_MANIFEST" \
    --output-root "$OUTPUT_ROOT/hup" \
    --mode full)
hup_job_id="${hup_submission%%;*}"

ucla_submission=$(sbatch --parsable --array="0-$((ucla_count - 1))" \
    cluster/preprocessing/stages/preprocess_ucla_subject.sh \
    --manifest "$UCLA_MANIFEST" \
    --output-root "$OUTPUT_ROOT/ucla" \
    --mode full)
ucla_job_id="${ucla_submission%%;*}"

# The summary runs even when an array fails so its failed_subjects.csv records
# incomplete work. The final QC requires both arrays and the summary to succeed.
summary_submission=$(sbatch --parsable \
    --dependency="afterany:${hup_job_id}:${ucla_job_id}" \
    cluster/preprocessing/stages/summarize_scaleout.sh)
summary_job_id="${summary_submission%%;*}"

qc_submission=$(sbatch --parsable \
    --dependency="afterok:${hup_job_id}:${ucla_job_id}:${summary_job_id}" \
    cluster/preprocessing/stages/generate_qc_report.sh)
qc_job_id="${qc_submission%%;*}"

echo "Submitted HUP full array: $hup_job_id (0-$((hup_count - 1)))"
echo "Submitted UCLA full array: $ucla_job_id (0-$((ucla_count - 1)))"
echo "Submitted preprocessing summary: $summary_job_id (afterany both arrays)"
echo "Submitted final preprocessing QC: $qc_job_id (afterok arrays and summary)"
