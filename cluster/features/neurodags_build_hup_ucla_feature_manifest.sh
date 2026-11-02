#!/usr/bin/env bash
#SBATCH --job-name=hup_ucla_feature_manifest
#SBATCH --account=CONFIGURE_SLURM_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.out
#SBATCH --error=/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/logs/%x_%j.err

set -euo pipefail
umask 027

readonly REPOSITORY="/CONFIGURE_SCRATCH/omni_ieeg_analysis_refactor"
readonly VIRTUAL_ENVIRONMENT="${SCRATCH:-/CONFIGURE_SCRATCH}/venvs/neurodags"
readonly OUTPUT_ROOT="/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3"
readonly CHANNEL_LABELS_CSV="$OUTPUT_ROOT/feature_cohort/channel_label_audit_hup_ucla.csv"
readonly PREPROCESSED_ROOT="$OUTPUT_ROOT/full_nonictal_v1"
# Derivatives that a previous analysis root already computed are indexed in
# place so valid interictal preprocessing is never recomputed. This root is
# read-only: it is only ever supplied as a preprocessing INPUT, and it is listed
# after the v3 root so v3 wins for any recording present in both.
readonly REUSE_PREPROCESSED_ROOT="${OMNI_REUSE_PREPROCESSED_ROOT:-/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v2/full_nonictal_v1}"

# Pilot and full are fully disjoint: their own cohort, their own frozen
# manifest, and their own derivatives roots for every feature family. A pilot
# can therefore never write into the production derivatives tree.
mode="${1:-full}"
case "$mode" in
    full)
        readonly MANIFEST_DIRECTORY="$OUTPUT_ROOT/manifests/hup_ucla_feature_neurodags_v1"
        readonly COHORT_CSV="$OUTPUT_ROOT/feature_cohort/nonictal_feature_cohort_hup_ucla.csv"
        readonly DERIVATIVES_SUFFIX="neurodags_v1"
        ;;
    pilot)
        readonly MANIFEST_DIRECTORY="$OUTPUT_ROOT/manifests/hup_ucla_feature_neurodags_pilot_v1"
        readonly COHORT_CSV="$OUTPUT_ROOT/feature_cohort/feature_pilot_manifest_v1.csv"
        readonly DERIVATIVES_SUFFIX="neurodags_pilot_v1"
        ;;
    *) echo "mode must be full or pilot" >&2; exit 2 ;;
esac
readonly RELATIVE_DERIVATIVES_ROOT="$OUTPUT_ROOT/relative_bandpower_$DERIVATIVES_SUFFIX"
readonly TIME_DOMAIN_DERIVATIVES_ROOT="$OUTPUT_ROOT/time_domain_$DERIVATIVES_SUFFIX"
readonly FREQUENCY_DOMAIN_DERIVATIVES_ROOT="$OUTPUT_ROOT/frequency_domain_$DERIVATIVES_SUFFIX"
readonly COMPLEXITY_DERIVATIVES_ROOT="$OUTPUT_ROOT/complexity_$DERIVATIVES_SUFFIX"

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "ERROR: submit with sbatch; do not run on a login node." >&2; exit 2; }
[[ -x "$VIRTUAL_ENVIRONMENT/bin/python" && -f "$COHORT_CSV" && -f "$CHANNEL_LABELS_CSV" ]] || { echo "Missing manifest inputs ($mode cohort: $COHORT_CSV)." >&2; exit 1; }
[[ -d "$PREPROCESSED_ROOT/hup" && -d "$PREPROCESSED_ROOT/ucla" ]] || { echo "Missing preprocessing outputs." >&2; exit 1; }
[[ ! -e "$MANIFEST_DIRECTORY" ]] || { echo "Frozen manifest already exists: $MANIFEST_DIRECTORY" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT/logs" \
    "$RELATIVE_DERIVATIVES_ROOT" "$TIME_DOMAIN_DERIVATIVES_ROOT" \
    "$FREQUENCY_DOMAIN_DERIVATIVES_ROOT" "$COMPLEXITY_DERIVATIVES_ROOT"
source "$VIRTUAL_ENVIRONMENT/bin/activate"
cd "$REPOSITORY"
export PYTHONPATH="$REPOSITORY${PYTHONPATH:+:$PYTHONPATH}"
echo "Mode: $mode"
echo "Cohort: $COHORT_CSV"
echo "Frozen manifest: $MANIFEST_DIRECTORY"
echo "Derivatives roots: *_$DERIVATIVES_SUFFIX"
manifest_arguments=(
    --cohort-csv "$COHORT_CSV"
    --preprocessed-root "$PREPROCESSED_ROOT"
    --output-directory "$MANIFEST_DIRECTORY"
    --relative-bandpower-derivatives-path "$RELATIVE_DERIVATIVES_ROOT"
    --time-domain-derivatives-path "$TIME_DOMAIN_DERIVATIVES_ROOT"
    --frequency-domain-derivatives-path "$FREQUENCY_DOMAIN_DERIVATIVES_ROOT"
    --complexity-derivatives-path "$COMPLEXITY_DERIVATIVES_ROOT"
    --channel-labels-csv "$CHANNEL_LABELS_CSV"
)
if [[ -d "$REUSE_PREPROCESSED_ROOT" ]]; then
    manifest_arguments+=(--preprocessed-root "$REUSE_PREPROCESSED_ROOT")
    echo "Reusing already-valid derivatives (read-only) from $REUSE_PREPROCESSED_ROOT"
fi
python scripts/features/build_hup_ucla_feature_manifest.py "${manifest_arguments[@]}"
