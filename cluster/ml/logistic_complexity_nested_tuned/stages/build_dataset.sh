#!/usr/bin/env bash
#SBATCH --job-name=omni_cxnt_dataset
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.err
set -euo pipefail
umask 027
readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
readonly ANALYSIS_ROOT="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3"
task="${1:-}"
[[ "$task" =~ ^(interictal|ictal)$ ]] || { echo "Usage: $0 {interictal|ictal}" >&2; exit 2; }
readonly ROOT="$ANALYSIS_ROOT/ml_logistic_complexity_c1_nested_tuned_v1/$task"
readonly DATASET_DIR="$ROOT/dataset"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: submit with sbatch; do not run on a login node." >&2; exit 2; }
[[ -x "$VIRTUAL_ENVIRONMENT/bin/python" && -d "$REPOSITORY/.git" && ! -e "$DATASET_DIR" ]] || { echo "Missing environment/repository or existing dataset output." >&2; exit 1; }
mkdir -p "$ROOT" "$ANALYSIS_ROOT/logs"
source "$VIRTUAL_ENVIRONMENT/bin/activate"; cd "$REPOSITORY"
export PYTHONPATH="$REPOSITORY${PYTHONPATH:+:$PYTHONPATH}" TMPDIR="${SLURM_TMPDIR:?}" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python -u scripts/ml/complexity_dataset.py --task "$task" --aggregations mean std median --output-dir "$DATASET_DIR" --repository "$REPOSITORY"
