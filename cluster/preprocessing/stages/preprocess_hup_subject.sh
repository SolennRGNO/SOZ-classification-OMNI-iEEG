#!/usr/bin/env bash
#SBATCH --job-name=hup_nonictal_subject
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/hup_nonictal_%A_%a.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/hup_nonictal_%A_%a.err

set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
readonly PYTHON="$VIRTUAL_ENVIRONMENT/bin/python"
# Already-valid derivatives from a previous analysis root are reused in place
# (read-only) so interictal preprocessing is never recomputed.
readonly REUSE_PREPROCESSED_ROOT="${OMNI_REUSE_PREPROCESSED_ROOT:-/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v2/full_nonictal_v1/hup}"

manifest=""
output_root=""
mode=""
while (( $# )); do
    case "$1" in
        --manifest) manifest="$2"; shift 2 ;;
        --output-root) output_root="$2"; shift 2 ;;
        --mode) mode="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "Missing SLURM_ARRAY_TASK_ID." >&2; exit 2; }
[[ -f "$manifest" ]] || { echo "Missing manifest: $manifest" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 1; }
case "$mode" in
    pilot) expected_root="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/multisubject_pilot_v1/hup" ;;
    full) expected_root="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/full_nonictal_v1/hup" ;;
    *) echo "--mode must be pilot or full." >&2; exit 2 ;;
esac
[[ "$output_root" == "$expected_root" ]] || {
    echo "Output root must be $expected_root for mode $mode." >&2
    exit 2
}

mkdir -p "$output_root" /CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs
cd "$REPOSITORY"
export PYTHONPATH="$REPOSITORY${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:?}/matplotlib"
export NUMBA_CACHE_DIR="$SLURM_TMPDIR/numba"
export TMPDIR="$SLURM_TMPDIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

run_arguments=(
    --manifest "$manifest"
    --subject-index "$SLURM_ARRAY_TASK_ID"
    --output-root "$output_root"
    --repository "$REPOSITORY"
    --dataset hup
    --mode "$mode"
)
if [[ -d "$REUSE_PREPROCESSED_ROOT" ]]; then
    run_arguments+=(--reuse-root "$REUSE_PREPROCESSED_ROOT")
fi
"$PYTHON" scripts/preprocessing/build_subject_manifests.py run-subject "${run_arguments[@]}"
