#!/usr/bin/env bash
#SBATCH --job-name=openieeg_raw_spectral_qc
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_openieeg_qc_v1/logs/%x_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_openieeg_qc_v1/logs/%x_%j.err

# Full-dataset raw spectral / line-noise inspection of OpenIEEG.
#
# DIAGNOSTIC ONLY. No notch, band-pass, high-pass, low-pass, re-reference or
# resampling operation is applied, and no preprocessing decision is made. The
# outputs live under $ANALYSIS_ROOT/line_noise_qc and are not derivatives.

set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
readonly DATA_ROOT="/CONFIGURE_DATA_ROOT"
readonly ANALYSIS_ROOT="/CONFIGURE_SCRATCH/omni_openieeg_qc_v1"
readonly INVENTORY_CSV="$ANALYSIS_ROOT/inventory/recording_inventory_openieeg.csv"
readonly OUTPUT_DIRECTORY="$ANALYSIS_ROOT/line_noise_qc"
readonly LOG_DIRECTORY="$ANALYSIS_ROOT/logs"
# Raw-only MNE report, same style as the HUP/UCLA preprocessing QC report but
# with the before-preprocessing half only. Detroit is the whole of OpenIEEG once
# sub-openieegUCLA* is excluded, which the inventory already does.
readonly MNE_REPORT_SITE="Detroit"
readonly MNE_REPORT_HTML="$OUTPUT_DIRECTORY/openieeg_detroit_raw_spectral_qc_full.html"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: submit this script with sbatch; do not run it on a login node." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIRECTORY" "$LOG_DIRECTORY"
[[ -d "$REPOSITORY" ]] || { echo "Missing repository: $REPOSITORY" >&2; exit 1; }
[[ -f "$INVENTORY_CSV" ]] || {
    echo "Missing OpenIEEG inventory: $INVENTORY_CSV" >&2
    echo "Run cluster/preprocessing/openieeg_spectral_qc/stages/build_openieeg_inventory.sh first." >&2
    exit 1
}
[[ -f "$VIRTUAL_ENVIRONMENT/bin/activate" ]] || {
    echo "Missing virtual environment: $VIRTUAL_ENVIRONMENT" >&2
    exit 1
}

# Keep numerical libraries single-threaded: recording-level parallelism is
# controlled explicitly by --n-jobs.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:?}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"

python -u scripts/preprocessing/run_openieeg_raw_spectral_qc.py \
    --inventory-csv "$INVENTORY_CSV" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIRECTORY" \
    --window-seconds 8 \
    --overlap-seconds 0 \
    --duration-per-recording 600 \
    --n-jobs "${SLURM_CPUS_PER_TASK:-8}" \
    --mne-report-site "$MNE_REPORT_SITE" \
    --mne-report-html "$MNE_REPORT_HTML" \
    --log-level INFO

echo "OpenIEEG raw spectral QC report: $OUTPUT_DIRECTORY/openieeg_raw_spectral_qc.html"
echo "Detroit raw-only MNE spectral report: $MNE_REPORT_HTML"
