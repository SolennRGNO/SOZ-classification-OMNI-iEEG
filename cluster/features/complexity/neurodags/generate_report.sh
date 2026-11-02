#!/usr/bin/env bash
#SBATCH --job-name=hup_ucla_cx_report
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.err

set -euo pipefail
umask 027
readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
readonly FEATURE_DIR="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/complexity_neurodags_cohort_v1"
readonly OUTPUT_HTML="$FEATURE_DIR/complexity_extraction_report.html"

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: submit with sbatch; do not run on a login node." >&2; exit 2; }
[[ -x "$VIRTUAL_ENVIRONMENT/bin/python" && -d "$FEATURE_DIR" ]] || { echo "Missing report inputs." >&2; exit 1; }
source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"
export PYTHONPATH="$REPOSITORY${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:?}/matplotlib"
mkdir -p "$MPLCONFIGDIR"
python -u scripts/features/generate_complexity_report.py \
    --feature-dir "$FEATURE_DIR" \
    --output-html "$OUTPUT_HTML"
