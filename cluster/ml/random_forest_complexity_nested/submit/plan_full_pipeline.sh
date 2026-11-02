#!/usr/bin/env bash
# Compatibility plan-only entrypoint for the nested Random Forest ML submitter.
set -euo pipefail

for argument in "$@"; do
    [[ "$argument" != "--submit" ]] || {
        echo "This entrypoint is plan-only; use submit_full_pipeline.sh to submit." >&2
        exit 2
    }
done
exec bash cluster/ml/random_forest_complexity_nested/submit/submit_full_pipeline.sh "$@"
