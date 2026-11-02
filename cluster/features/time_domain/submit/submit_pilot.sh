#!/usr/bin/env bash
# Submit the three frozen pilot-manifest recordings. It is never run automatically.
set -euo pipefail
umask 027
readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly OUTPUT_ROOT="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3"
# The pilot has its own cohort, its own frozen manifest and its own
# *_neurodags_pilot_v1 derivatives; it never touches the full outputs.
readonly COHORT_CSV="$OUTPUT_ROOT/feature_cohort/feature_pilot_manifest_v1.csv"
readonly MANIFEST_DIRECTORY="$OUTPUT_ROOT/manifests/hup_ucla_feature_neurodags_pilot_v1"
readonly DATASETS_CONFIG="$MANIFEST_DIRECTORY/datasets_hup_ucla_time_domain.yml"
readonly INDEX_MANIFEST="$MANIFEST_DIRECTORY/hup_ucla_feature_neurodags_index.csv"
[[ -z "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: run this submitter on a login node." >&2; exit 2; }
[[ -d "$REPOSITORY/.git" && -f "$COHORT_CSV" ]] || { echo "Missing repository or pilot cohort: $COHORT_CSV" >&2; exit 1; }
cd "$REPOSITORY"
if [[ -d "$MANIFEST_DIRECTORY" && -f "$DATASETS_CONFIG" && -f "$INDEX_MANIFEST" ]]; then
    echo "Using existing frozen shared feature manifest: $MANIFEST_DIRECTORY"
else
    echo "ERROR: valid frozen shared feature manifest is required; refusing to build or replace it: $MANIFEST_DIRECTORY" >&2
    exit 1
fi
array=$(sbatch --parsable --array="0-2%3" cluster/features/time_domain/neurodags/extract_array.sh pilot)
array_id="${array%%;*}"
echo "Time-domain pilot submitted."
echo "Manifest: reused existing frozen manifest"
echo "Array: $array_id (task 0: HUP ictal; task 1: HUP interictal; task 2: UCLA sleep / recording_state=interictal; max 3 concurrent; 1 CPU/task; 8G/task; 12h/task)"
echo "Inspect later with: sacct -j $array_id --format=JobID,State,Elapsed,MaxRSS"
