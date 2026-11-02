#!/usr/bin/env bash
# Lightweight login-node submitter. It is never run automatically.
set -euo pipefail
umask 027
readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly OUTPUT_ROOT="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3"
readonly COHORT_CSV="$OUTPUT_ROOT/feature_cohort/nonictal_feature_cohort_hup_ucla.csv"
readonly MANIFEST_DIRECTORY="$OUTPUT_ROOT/manifests/hup_ucla_feature_neurodags_v1"
readonly DATASETS_CONFIG="$MANIFEST_DIRECTORY/datasets_hup_ucla_time_domain.yml"
readonly INDEX_MANIFEST="$MANIFEST_DIRECTORY/hup_ucla_feature_neurodags_index.csv"

[[ -z "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: run this submitter on a login node." >&2; exit 2; }
[[ -d "$REPOSITORY/.git" && -f "$COHORT_CSV" ]] || { echo "Missing repository or cohort input." >&2; exit 1; }
cd "$REPOSITORY"
for path in Neurodags/pipeline_hup_ucla_time_domain.yml Neurodags/custom_nodes_time_domain.py scripts/features/time_domain.py scripts/features/assemble_time_domain_cohort.py scripts/features/generate_time_domain_report.py cluster/features/time_domain/neurodags/{extract_array,assemble_cohort,generate_report}.sh; do
    [[ -f "$path" ]] || { echo "Missing workflow file: $path" >&2; exit 1; }
done
if [[ -d "$MANIFEST_DIRECTORY" && -f "$DATASETS_CONFIG" && -f "$INDEX_MANIFEST" ]]; then
    echo "Using existing frozen shared feature manifest: $MANIFEST_DIRECTORY"
else
    echo "ERROR: valid frozen shared feature manifest is required; refusing to build or replace it: $MANIFEST_DIRECTORY" >&2
    exit 1
fi
recording_count=$(awk 'END { print NR - 1 }' "$INDEX_MANIFEST")
array_size_source="frozen manifest $INDEX_MANIFEST"
(( recording_count > 0 )) || { echo "No recordings to extract; source: $array_size_source" >&2; exit 1; }
last_index=$((recording_count - 1))
echo "Array size: $recording_count task(s) from $array_size_source"

extraction=$(sbatch --parsable --array="0-${last_index}%8" cluster/features/time_domain/neurodags/extract_array.sh full)
extraction_id="${extraction%%;*}"
assembly=$(sbatch --parsable --dependency="afterok:${extraction_id}" cluster/features/time_domain/neurodags/assemble_cohort.sh)
assembly_id="${assembly%%;*}"
report=$(sbatch --parsable --dependency="afterok:${assembly_id}" cluster/features/time_domain/neurodags/generate_report.sh)
report_id="${report%%;*}"

echo "Time-domain NeuroDAGs pipeline submitted."
echo "Manifest: reused existing frozen manifest"
echo "Extraction: $extraction_id (array 0-${last_index}%8; 1 CPU/task; 8G/task; 12h/task)"
echo "Assembly: $assembly_id (afterok extraction; 4 CPUs; 64G; 12h)"
echo "Report: $report_id (afterok assembly; 4 CPUs; 32G; 2h)"
echo "Expected HTML: $OUTPUT_ROOT/time_domain_neurodags_cohort_v1/time_domain_extraction_report.html"
