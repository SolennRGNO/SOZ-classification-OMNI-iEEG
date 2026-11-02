#!/usr/bin/env bash
#SBATCH --job-name=omni_detroit_dataset
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/%x_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1/logs/%x_%j.err
set -euo pipefail
readonly REPOSITORY=/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor
readonly ANALYSIS_ROOT=/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
readonly FEATURE_ROOT="$ANALYSIS_ROOT/features_v1"
[[ -n "${SLURM_JOB_ID:-}" && -x "$VIRTUAL_ENVIRONMENT/bin/python" && ! -e "$ANALYSIS_ROOT/ml_detroit_datasets_v1" ]] || { echo "missing environment or dataset output exists" >&2; exit 1; }
source "$VIRTUAL_ENVIRONMENT/bin/activate"; cd "$REPOSITORY"; export PYTHONPATH="$REPOSITORY" TMPDIR="${SLURM_TMPDIR:?}"
python -u scripts/ml/detroit_dataset.py --relative-bandpower "$FEATURE_ROOT/relative_bandpower_neurodags_full_cohort_v1/relative_bandpower_channels_detroit.parquet" --time-domain "$FEATURE_ROOT/time_domain_neurodags_full_cohort_v1/time_domain_channels_detroit.parquet" --frequency-domain "$FEATURE_ROOT/frequency_domain_neurodags_full_cohort_v1/frequency_domain_channels_detroit.parquet" --complexity "$FEATURE_ROOT/complexity_neurodags_full_cohort_v1/complexity_channels_detroit.parquet" --baseline-output-dir "$ANALYSIS_ROOT/ml_detroit_datasets_v1/baseline" --complexity-output-dir "$ANALYSIS_ROOT/ml_detroit_datasets_v1/complexity" --repository "$REPOSITORY"
