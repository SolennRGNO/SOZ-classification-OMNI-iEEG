#!/usr/bin/env bash
#SBATCH --job-name=omni_cxn_model
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.err
set -euo pipefail
umask 027
readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
readonly ANALYSIS_ROOT="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3"
task="${1:-}"
[[ "$task" =~ ^(interictal|ictal)$ ]] || { echo "Usage: $0 {interictal|ictal}" >&2; exit 2; }
readonly ROOT="$ANALYSIS_ROOT/ml_logistic_complexity_c1_nested_v1/$task"
readonly DATASET_DIR="$ROOT/dataset" RESULTS_DIR="$ROOT/results"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: submit with sbatch; do not run on a login node." >&2; exit 2; }
[[ -x "$VIRTUAL_ENVIRONMENT/bin/python" && -f "$DATASET_DIR/ml_complexity_participant_channel_dataset.parquet" && -f "$DATASET_DIR/complexity_feature_list.json" && ! -e "$RESULTS_DIR" ]] || { echo "Missing model inputs or existing results output." >&2; exit 1; }
source "$VIRTUAL_ENVIRONMENT/bin/activate"; cd "$REPOSITORY"
export PYTHONPATH="$REPOSITORY${PYTHONPATH:+:$PYTHONPATH}" TMPDIR="${SLURM_TMPDIR:?}" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
# N is tuned by the inner grouped CV; passing a fixed selection count here is rejected by construction.
python -u scripts/ml/logistic_complexity.py --dataset "$DATASET_DIR/ml_complexity_participant_channel_dataset.parquet" --feature-list "$DATASET_DIR/complexity_feature_list.json" --output-dir "$RESULTS_DIR" --repository "$REPOSITORY" --n-bootstrap 5000 --experiment c1_nested
