#!/usr/bin/env bash
#SBATCH --job-name=hup_ucla_1khz_two_subjects
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/hup_ucla_1khz_two_subjects_%A_%a.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/hup_ucla_1khz_two_subjects_%A_%a.err

set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
readonly ELIGIBLE_MANIFEST="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/inventory/eligible_recordings_hup_ucla.csv"
readonly MANIFEST_DIRECTORY="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/manifests/two_subjects_v1"
readonly DATASETS_CONFIG="$MANIFEST_DIRECTORY/datasets_two_subjects_v1.yml"
readonly SUBJECT_TASKS="$MANIFEST_DIRECTORY/subject_tasks_v1.csv"
readonly OUTPUT_ROOT="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/two_subjects"
readonly LOG_ROOT="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit this subject-level array with sbatch; do not run it on a login node." >&2
    exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 1 )); then
    echo "ERROR: subject array index must be 0 or 1." >&2
    exit 2
fi

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT" "$(dirname "$MANIFEST_DIRECTORY")"
[[ -d "$REPOSITORY" ]] || { echo "Missing repository: $REPOSITORY" >&2; exit 1; }
[[ -f "$ELIGIBLE_MANIFEST" ]] || { echo "Missing manifest: $ELIGIBLE_MANIFEST" >&2; exit 1; }
[[ -f "$VIRTUAL_ENVIRONMENT/bin/activate" ]] || {
    echo "Missing virtual environment: $VIRTUAL_ENVIRONMENT" >&2
    exit 1
}

source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"

(
    flock -x 9
    python scripts/preprocessing/build_neurodags_index.py \
        --eligible-manifest "$ELIGIBLE_MANIFEST" \
        --output-directory "$MANIFEST_DIRECTORY" \
        --derivatives-path "$OUTPUT_ROOT" \
        --selection two-subjects \
        --version v1
) 9>"$(dirname "$MANIFEST_DIRECTORY")/.two_subjects_v1.lock"

[[ -f "$SUBJECT_TASKS" && -f "$DATASETS_CONFIG" ]] || {
    echo "Subject index creation failed." >&2
    exit 1
}

index_list=$(awk -F, -v task="$SLURM_ARRAY_TASK_ID" '$1 == task {print $4}' "$SUBJECT_TASKS")
if [[ -z "$index_list" ]]; then
    echo "No frozen NeuroDAGs indices for subject task $SLURM_ARRAY_TASK_ID." >&2
    exit 1
fi
IFS=';' read -r -a target_indices <<< "$index_list"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:?}/matplotlib"
export NUMBA_CACHE_DIR="$SLURM_TMPDIR/numba"
export TMPDIR="$SLURM_TMPDIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

echo "Subject task manifest: $SUBJECT_TASKS"
echo "Task $SLURM_ARRAY_TASK_ID processes frozen indices: ${target_indices[*]}"

neurodags --log-level INFO run Neurodags/pipeline_hup_ucla_1khz.yml \
    -d "$DATASETS_CONFIG" \
    --derivative Preprocessed1kHzNoBandpass \
    --only-index "${target_indices[@]}" \
    --raise-on-error \
    --n-jobs 1
