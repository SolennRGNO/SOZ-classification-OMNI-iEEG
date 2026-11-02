#!/usr/bin/env bash
#SBATCH --job-name=hup_ucla_bp_nd
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.err

set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
readonly OUTPUT_ROOT="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3"

mode="${1:-full}"
case "$mode" in
    full) readonly MANIFEST_DIRECTORY="$OUTPUT_ROOT/manifests/hup_ucla_feature_neurodags_v1" ;;
    pilot) readonly MANIFEST_DIRECTORY="$OUTPUT_ROOT/manifests/hup_ucla_feature_neurodags_pilot_v1" ;;
    *) echo "mode must be full or pilot" >&2; exit 2 ;;
esac
# The datasets configuration is generated beside the frozen manifest it indexes,
# so the derivatives root always matches this manifest's mode (pilot or full).
readonly DATASETS_CONFIG="$MANIFEST_DIRECTORY/datasets_hup_ucla_relative_bandpower.yml"
readonly INDEX_MANIFEST="$MANIFEST_DIRECTORY/hup_ucla_feature_neurodags_index.csv"

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: submit with sbatch; do not run on a login node." >&2; exit 2; }
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "Missing SLURM_ARRAY_TASK_ID." >&2; exit 2; }
[[ -d "$REPOSITORY" && -x "$VIRTUAL_ENVIRONMENT/bin/python" ]] || { echo "Missing repository or environment." >&2; exit 1; }
[[ -f "$DATASETS_CONFIG" && -f "$INDEX_MANIFEST" ]] || { echo "Missing frozen relative-bandpower manifest: $MANIFEST_DIRECTORY" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT/logs"
source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"
export PYTHONPATH="$REPOSITORY${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:?}/matplotlib"
export NUMBA_CACHE_DIR="$SLURM_TMPDIR/numba"
export TMPDIR="$SLURM_TMPDIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

# For both pilot and full runs, the task ID is the frozen manifest row index.
target_index="$SLURM_ARRAY_TASK_ID"

# The frozen manifest is the single source of truth for array index -> recording.
# NeuroDAGs expands file_pattern with an unsorted glob and numbers the result
# positionally, so resolve the exact recording here and hand NeuroDAGs a
# one-recording configuration (the pattern the preprocessing scale-out uses).
readonly TASK_DATASETS_CONFIG="${SLURM_TMPDIR:?}/datasets_index_${target_index}.yml"
target_fif=$(python scripts/features/resolve_manifest_index.py \
    --index-manifest "$INDEX_MANIFEST" \
    --datasets-config "$DATASETS_CONFIG" \
    --index "$target_index" \
    --output "$TASK_DATASETS_CONFIG")

echo "Mode: $mode; array task: $SLURM_ARRAY_TASK_ID; frozen manifest index: $target_index"
echo "Recording: $target_fif"
neurodags --log-level INFO run Neurodags/pipeline_hup_ucla_relative_bandpower.yml \
    -d "$TASK_DATASETS_CONFIG" \
    --derivative RelativeBandPower \
    --only-index 0 \
    --raise-on-error \
    --n-jobs 1
