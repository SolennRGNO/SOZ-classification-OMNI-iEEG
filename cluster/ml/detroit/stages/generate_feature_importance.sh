#!/usr/bin/env bash
# Read-only Mantilla-style Logistic Regression coefficient post-processing.
# Submit from a login node; no dataset or model fitting occurs in this job.
#SBATCH --job-name=omni_detroit_feature_importance
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/%x_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/%x_%j.err
set -euo pipefail
readonly REPOSITORY=/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor
readonly ANALYSIS_ROOT=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
[[ -n "${SLURM_JOB_ID:-}" && -x "$VIRTUAL_ENVIRONMENT/bin/python" ]] || exit 1
source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"
export PYTHONPATH="$REPOSITORY"
"$VIRTUAL_ENVIRONMENT/bin/python" -u scripts/ml/generate_detroit_feature_importance.py \
  --output-dir "$ANALYSIS_ROOT/ml_feature_importance_detroit_v1"
