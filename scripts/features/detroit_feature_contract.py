"""Detroit's explicit feature-family and predictor-schema contract.

This module contains no estimators.  It deliberately imports the frozen
HUP/UCLA constants so Detroit cannot drift in feature definitions, aggregation
statistics, or final baseline predictor order.  Extraction remains separated:
the three baseline source families and complexity are never joined here.
"""

from __future__ import annotations

from scripts.ml import dataset as baseline


COHORT_NAME = "detroit"
BASELINE_FAMILIES = baseline.FAMILY_ORDER
COMPLEXITY_FAMILIES = baseline.HELD_OUT_FAMILIES
CHANNEL_AGGREGATIONS = baseline.AGGREGATIONS


def baseline_predictor_names() -> list[str]:
    """The canonical 96 predictor names in HUP/UCLA's deterministic order."""
    return baseline.predictor_names()


def baseline_source_feature_names() -> dict[str, list[str]]:
    """Source epoch columns for the three separately extracted baseline families."""
    return baseline.source_feature_names()


def validate_contract() -> None:
    """Fail loudly if an upstream change would make Detroit incomparable."""
    counts = baseline.feature_counts()
    if BASELINE_FAMILIES != (
        "frequency_domain",
        "relative_bandpower",
        "time_domain",
    ):
        raise AssertionError(f"unexpected baseline family order: {BASELINE_FAMILIES}")
    if COMPLEXITY_FAMILIES != ("complexity",):
        raise AssertionError(f"complexity must remain separate: {COMPLEXITY_FAMILIES}")
    if CHANNEL_AGGREGATIONS != ("mean", "std", "median"):
        raise AssertionError(f"unexpected channel aggregations: {CHANNEL_AGGREGATIONS}")
    if counts["epoch_feature_count"] != 32 or counts["predictor_count"] != 96:
        raise AssertionError(f"unexpected baseline feature count: {counts}")

