#!/usr/bin/env bash
# Read-only Detroit performance-figure render after every Detroit evaluator has published results.
#SBATCH --job-name=omni_detroit_figures
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/%x_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/%x_%j.err
set -euo pipefail

readonly REPOSITORY=/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor
readonly ANALYSIS_ROOT=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1
readonly OUTPUT_DIR="$ANALYSIS_ROOT/ml_performance_figures_detroit_v1"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"

[[ -n "${SLURM_JOB_ID:-}" && -x "$VIRTUAL_ENVIRONMENT/bin/python" ]] || {
    echo "run via sbatch with the neurodags environment available" >&2
    exit 1
}
source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"
export PYTHONPATH="$REPOSITORY" TMPDIR="${SLURM_TMPDIR:?}"
python -u scripts/ml/plot_detroit_experiment_performance.py \
    --analysis-root "$ANALYSIS_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --overwrite
