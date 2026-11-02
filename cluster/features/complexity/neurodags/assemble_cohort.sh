#!/usr/bin/env bash
#SBATCH --job-name=hup_ucla_cx_assemble
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.err

set -euo pipefail
umask 027
readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
readonly OUTPUT_ROOT="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3"
readonly COHORT_CSV="$OUTPUT_ROOT/feature_cohort/nonictal_feature_cohort_hup_ucla.csv"
readonly DERIVATIVES_ROOT="$OUTPUT_ROOT/complexity_neurodags_v1"
readonly ASSEMBLED_ROOT="$OUTPUT_ROOT/complexity_neurodags_cohort_v1"

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: submit with sbatch; do not run on a login node." >&2; exit 2; }
[[ -x "$VIRTUAL_ENVIRONMENT/bin/python" && -f "$COHORT_CSV" && -d "$DERIVATIVES_ROOT" ]] || { echo "Missing assembly inputs." >&2; exit 1; }
[[ ! -e "$ASSEMBLED_ROOT/complexity_epochs_hup_ucla.parquet" ]] || { echo "Assembly outputs already exist." >&2; exit 1; }
mkdir -p "$ASSEMBLED_ROOT" "$OUTPUT_ROOT/logs"
source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"
export PYTHONPATH="$REPOSITORY${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${SLURM_TMPDIR:?}"
python -u scripts/features/assemble_complexity_cohort.py \
    --derivatives-root "$DERIVATIVES_ROOT" \
    --cohort-csv "$COHORT_CSV" \
    --output-dir "$ASSEMBLED_ROOT" \
    --repository "$REPOSITORY"
