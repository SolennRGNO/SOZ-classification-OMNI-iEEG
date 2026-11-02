"""Construct the strict seizure-free participant-by-channel Complexity dataset.

This is a *separate* dataset from the logistic baseline. The baseline matrix is
frozen at three families (RelativeBandPower + TimeDomainFeatures +
FrequencyDomainFeatures = 32 epoch features / 96 predictors); ComplexityFeatures
is held out of it on purpose (see ``scripts.ml.dataset.HELD_OUT_FAMILIES``).
This module builds the complexity matrix on its own, from the 14 Mantilla-Ramos
ComplexityFeatures, without touching the baseline pipeline or its outputs.

Everything clinical is reused unchanged from :mod:`scripts.ml.dataset`: task
selection, the fixed seizure-free protocol filter, the strict channel-label
audit, channel identity, the participant-by-channel unit, the cross-recording
conflict checks and the finite-value checks are the *same function objects*, so
the two datasets can never drift apart on cohort definition or labels.

Two things are deliberately different:

* Input is the raw **epoch-level** table ``complexity_epochs_hup_ucla.parquet``.
  The already-aggregated ``complexity_channels_hup_ucla.parquet`` is never read:
  it aggregates within a recording, which would average-of-averages across
  recordings. Raw epoch rows are pooled across every eligible recording of the
  selected task *before* any aggregation, exactly like the baseline.
* Aggregation statistics are an explicit, required choice (``--aggregations``).
  Nothing here silently picks mean/std over mean/std/median.

No feature selection, no model and no multiscale-entropy policy decision is made
in this module.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.ml.common import (
    DEFAULT_ANALYSIS_ROOT,
    TASKS,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    git_commit,
    new_output_directory,
    resolve_task,
    sha256,
    timestamp_utc,
)

# Reused unchanged from the validated baseline builder. These are imported, not
# reimplemented: the complexity dataset must apply byte-for-byte the same task,
# seizure-free, channel-label and validity rules as the baseline.
from scripts.ml.dataset import (
    AGREEMENT_COLUMNS,
    EPOCH_KEY,
    EXPECTED_COHORT_COUNTS,
    FAMILY_COLUMN_PREFIXES,
    FAMILY_DIRECTORIES,
    FAMILY_FILES,
    FINAL_KEY,
    _clean,
    _validate_upstream_artifacts,
    cohort_count_tables,
    filter_epochs_to_task,
    select_task_recordings,
    strict_channel_labels,
    strict_seizure_free_participants,
    validate_feature_family,
)


REPOSITORY = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "ml_complexity_dataset_v1"
FAMILY = "complexity"

# Four of the 14 complexity names collide with baseline feature names
# (spectral_entropy, zero_crossing_rate, hjorth_mobility, hjorth_complexity), so
# the family prefix is applied here as well even though this matrix is
# complexity-only. It keeps a column name unambiguous wherever the two datasets
# are ever compared side by side.
COLUMN_PREFIX = FAMILY_COLUMN_PREFIXES[FAMILY]
COMPLEXITY_EPOCH_FEATURE_COUNT = 14

# The aggregated per-channel artifact produced by assemble_complexity_cohort.py.
# It is an explicit non-input: it aggregates within each recording, so reading it
# would produce an average of per-recording averages instead of pooled epochs.
AGGREGATED_CHANNEL_ARTIFACT = "complexity_channels_hup_ucla.parquet"
AGGREGATED_CHANNEL_ARTIFACT_POLICY = (
    "never used as an ML input: it aggregates epochs within a recording, so "
    "pooling it across recordings would be an average of averages; this builder "
    "always pools raw epoch rows first"
)

# Aggregation statistics are a required, explicit choice. The canonical order
# below fixes the predictor order regardless of the order given on the command
# line, so a run is reproducible from its recorded aggregation set alone.
SUPPORTED_AGGREGATIONS = ("mean", "std", "median")
AGGREGATION_DEFINITIONS = {
    "mean": "arithmetic mean over all pooled valid epochs",
    "std": "sample standard deviation over all pooled valid epochs (ddof=1)",
    "median": "median over all pooled valid epochs",
}
POOLING_POLICY = (
    "all valid epochs are pooled across eligible recordings of the selected task "
    "before aggregation; recordings with more valid epochs therefore contribute "
    "proportionally more observations"
)


def source_feature_names() -> list[str]:
    """The frozen 14 ComplexityFeatures names, read from the extractor."""
    core = importlib.import_module("scripts.features.complexity")
    names = list(core.FEATURE_NAMES)
    if not names:
        raise RuntimeError("complexity exposes no feature constants")
    if len(set(names)) != len(names):
        raise RuntimeError(f"complexity feature constants contain duplicates: {names}")
    if len(names) != COMPLEXITY_EPOCH_FEATURE_COUNT:
        raise AssertionError(
            f"complexity is fixed at {COMPLEXITY_EPOCH_FEATURE_COUNT} epoch features; "
            f"the extractor defines {len(names)}"
        )
    return names


def base_feature_names() -> list[str]:
    """Namespaced ML column names for the 14 complexity features."""
    return [f"{COLUMN_PREFIX}{name}" for name in source_feature_names()]


def resolve_aggregations(aggregations: object) -> tuple[str, ...]:
    """Validate an explicit aggregation choice; there is no default anywhere."""
    if aggregations is None:
        raise ValueError(
            "aggregation statistics must be chosen explicitly; "
            f"supported: {list(SUPPORTED_AGGREGATIONS)}"
        )
    if isinstance(aggregations, str):
        raise TypeError("aggregations must be a sequence of names, not a single string")
    requested = [str(name).strip().lower() for name in aggregations]
    if not requested:
        raise ValueError("at least one aggregation statistic is required")
    duplicates = sorted({name for name in requested if requested.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate aggregation statistics requested: {duplicates}")
    unsupported = sorted(set(requested) - set(SUPPORTED_AGGREGATIONS))
    if unsupported:
        raise ValueError(
            f"unsupported aggregation statistics {unsupported}; "
            f"supported: {list(SUPPORTED_AGGREGATIONS)}"
        )
    # Canonical order, so the predictor order does not depend on argument order.
    return tuple(name for name in SUPPORTED_AGGREGATIONS if name in requested)


def predictor_names(aggregations: object) -> list[str]:
    statistics = resolve_aggregations(aggregations)
    return [
        f"{feature}_{statistic}"
        for feature in base_feature_names()
        for statistic in statistics
    ]


def minimum_epochs_per_channel(aggregations: object) -> int:
    """Sample standard deviation (ddof=1) needs two pooled epochs; nothing else does."""
    return 2 if "std" in resolve_aggregations(aggregations) else 1


def feature_counts(aggregations: object) -> dict[str, Any]:
    statistics = resolve_aggregations(aggregations)
    base = base_feature_names()
    predictors = predictor_names(statistics)
    if len(predictors) != len(base) * len(statistics):
        raise AssertionError(
            "predictor count must equal epoch features times aggregations: "
            f"{len(predictors)} != {len(base)} x {len(statistics)}"
        )
    if len(set(predictors)) != len(predictors):
        raise AssertionError("aggregated predictor names are not unique")
    return {
        "family": FAMILY,
        "epoch_feature_count": len(base),
        "aggregations": list(statistics),
        "aggregation_definitions": {name: AGGREGATION_DEFINITIONS[name] for name in statistics},
        "predictor_count": len(predictors),
        "column_prefix": COLUMN_PREFIX,
        "minimum_pooled_epochs_per_channel": minimum_epochs_per_channel(statistics),
        "separate_from_baseline": (
            "this matrix is never merged into the logistic baseline's 96 predictors"
        ),
    }


def read_complexity_epochs(path: Path) -> pd.DataFrame:
    """Read the epoch-level complexity table, refusing the aggregated artifact."""
    path = Path(path)
    expected = FAMILY_FILES[FAMILY]["epochs"]
    if path.name == AGGREGATED_CHANNEL_ARTIFACT:
        raise ValueError(
            f"refusing to read {AGGREGATED_CHANNEL_ARTIFACT}: "
            f"{AGGREGATED_CHANNEL_ARTIFACT_POLICY}; pass {expected} instead"
        )
    if path.name != expected:
        raise ValueError(f"unexpected complexity epoch filename: {path.name} (expected {expected})")
    frame = pd.read_parquet(path)
    missing_identity = sorted({"epoch_index", "epoch_start_s"} - set(frame.columns))
    if missing_identity:
        raise ValueError(
            f"{path.name} lacks epoch identity columns {missing_identity}; this builder "
            "requires the raw epoch-level table, not a per-channel aggregate"
        )
    aggregated_columns = sorted(
        column
        for column in frame.columns
        if column.endswith(tuple(f"_{name}" for name in SUPPORTED_AGGREGATIONS))
        and column.removesuffix("_mean").removesuffix("_std").removesuffix("_median")
        in set(source_feature_names())
    )
    if aggregated_columns:
        raise ValueError(
            f"{path.name} already contains aggregated columns {aggregated_columns[:5]}; "
            "this builder aggregates raw epochs itself"
        )
    return frame


def validate_complexity_epochs(frame: pd.DataFrame) -> pd.DataFrame:
    """Reuse the baseline family validator: schema, identity, validity, finiteness."""
    return validate_feature_family(frame, FAMILY, base_feature_names())


def label_epochs(
    epochs: pd.DataFrame,
    cohort: pd.DataFrame,
    labels: pd.DataFrame,
    task: str,
) -> pd.DataFrame:
    """Select one task, apply the fixed seizure-free protocol and attach labels.

    Returns the raw, still-unaggregated epoch rows, pooled across every eligible
    recording of the selected task. Every rule applied here comes from
    :mod:`scripts.ml.dataset` unchanged.
    """
    task = resolve_task(task)
    task_cohort = select_task_recordings(cohort, task)
    task_epochs = filter_epochs_to_task(epochs, task_cohort, task)
    participants = strict_seizure_free_participants(task_cohort)
    channel_labels = strict_channel_labels(labels, participants)
    work = task_epochs[task_epochs["participant_id"].isin(participants)].copy()
    work = work.rename(columns={"channel": "channel_id", "channel_type": "mne_signal_type"})
    work = work.merge(
        channel_labels,
        on=FINAL_KEY,
        how="inner",
        validate="many_to_one",
        suffixes=("", "_audit"),
    )
    if work.empty:
        raise ValueError("no strict seizure-free high-confidence channel epochs remain")
    if not work["dataset"].eq(work["dataset_audit"]).all():
        raise ValueError("feature and channel-audit site disagree")
    observed_label = pd.to_numeric(work["supervised_label"], errors="coerce")
    if observed_label.isna().any() or not observed_label.astype(int).eq(work["target"]).all():
        raise ValueError("existing epoch supervised labels disagree with channel-label audit")
    metadata_conflicts = (
        work.groupby(FINAL_KEY, dropna=False)[
            ["dataset", "mne_signal_type", "channel_type", "target"]
        ]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if metadata_conflicts.any():
        examples = metadata_conflicts[metadata_conflicts].index.tolist()[:10]
        raise ValueError(f"cross-recording metadata conflicts: {examples}")
    return work.reset_index(drop=True)


def aggregate_participant_channels(
    epochs: pd.DataFrame,
    cohort: pd.DataFrame,
    labels: pd.DataFrame,
    task: str,
    aggregations: object,
) -> pd.DataFrame:
    """Pool raw epochs across recordings, then aggregate to participant x channel."""
    task = resolve_task(task)
    statistics = resolve_aggregations(aggregations)
    minimum_epochs = minimum_epochs_per_channel(statistics)
    work = label_epochs(epochs, cohort, labels, task)
    features = base_feature_names()
    grouped = work.groupby(FINAL_KEY, sort=True, dropna=False)
    valid_epoch_counts = grouped.size()
    insufficient = valid_epoch_counts[valid_epoch_counts < minimum_epochs]
    if not insufficient.empty:
        examples = [
            {
                "participant_id": participant,
                "channel_id": channel,
                "n_valid_epochs": int(count),
            }
            for (participant, channel), count in insufficient.head(20).items()
        ]
        raise ValueError(
            f"aggregations {list(statistics)} require at least {minimum_epochs} pooled "
            f"valid epoch(s) per participant/channel; found {len(insufficient)} "
            f"insufficient channels; examples={examples}"
        )
    aggregated = grouped[features].agg(list(statistics))
    aggregated.columns = [f"{feature}_{statistic}" for feature, statistic in aggregated.columns]
    aggregated = aggregated.reset_index()
    metadata = grouped.agg(
        dataset=("dataset", "first"),
        mne_signal_type=("mne_signal_type", "first"),
        channel_type=("channel_type", "first"),
        target=("target", "first"),
        n_recordings=("recording_id", "nunique"),
        n_valid_epochs=("epoch_index", "size"),
    ).reset_index()
    metadata["task"] = task
    final = metadata.merge(aggregated, on=FINAL_KEY, validate="one_to_one")
    expected_predictors = predictor_names(statistics)
    final = final[
        FINAL_KEY
        + [
            "dataset", "task", "mne_signal_type", "channel_type", "target",
            "n_recordings", "n_valid_epochs",
        ]
        + expected_predictors
    ]
    if final.duplicated(FINAL_KEY).any():
        raise AssertionError("final participant-by-channel identities are not unique")
    counts = feature_counts(statistics)
    if len(expected_predictors) != counts["predictor_count"] or not set(
        expected_predictors
    ).issubset(final.columns):
        raise AssertionError(
            f"the final dataset must contain exactly {counts['predictor_count']} predictors"
        )
    return final


def dataset_qc(final: pd.DataFrame, task: str, aggregations: object) -> dict[str, Any]:
    task = resolve_task(task)
    statistics = resolve_aggregations(aggregations)
    observed_tasks = sorted(set(final["task"].map(_clean).str.lower()))
    if observed_tasks != [task]:
        raise ValueError(f"dataset must hold exactly the {task} task; found {observed_tasks}")
    counts_spec = feature_counts(statistics)
    predictors = predictor_names(statistics)
    values = final[predictors]
    if not all(pd.api.types.is_numeric_dtype(values[column]) for column in predictors):
        raise TypeError(f"all {counts_spec['predictor_count']} predictors must have numeric dtypes")
    array = values.to_numpy(dtype=float)
    invalid = {
        "nan": int(np.isnan(array).sum()),
        "positive_inf": int(np.isposinf(array).sum()),
        "negative_inf": int(np.isneginf(array).sum()),
    }
    if any(invalid.values()):
        per_feature = pd.DataFrame({
            "feature": predictors,
            "nan": np.isnan(array).sum(axis=0),
            "positive_inf": np.isposinf(array).sum(axis=0),
            "negative_inf": np.isneginf(array).sum(axis=0),
        })
        bad = per_feature[(per_feature[["nan", "positive_inf", "negative_inf"]] > 0).any(axis=1)]
        raise ValueError(f"final predictors contain invalid valueCONFIGURE_WINDOWS_PATH")
    counts = {
        "participants": int(final["participant_id"].nunique()),
        "sites": int(final["dataset"].nunique()),
        "observations": int(len(final)),
        "pathological": int(final["target"].eq(1).sum()),
        "non_pathological": int(final["target"].eq(0).sum()),
    }
    # The baseline's frozen reference counts are reported for comparison only.
    # Complexity may legitimately differ (for example through aperiodic-fit
    # channel-epoch exclusions), so nothing here is ever enforced.
    reference_counts = EXPECTED_COHORT_COUNTS.get(task, {})
    pooled = pd.to_numeric(final["n_valid_epochs"], errors="raise")
    channel_type_mapping = (
        final.groupby(["mne_signal_type", "channel_type"], sort=True, dropna=False)
        .agg(
            participant_channel_count=("channel_id", "size"),
            pooled_epoch_rows=("n_valid_epochs", "sum"),
        )
        .reset_index()
        .rename(columns={"channel_type": "clinical_channel_type"})
    )
    return {
        "task": task,
        "family": FAMILY,
        "counts": counts,
        "feature_counts": counts_spec,
        "baseline_expected_counts_comparison_only": reference_counts,
        "count_differences_observed_minus_baseline_expected": {
            key: counts[key] - value for key, value in reference_counts.items()
        },
        "duplicate_final_rows": int(final.duplicated(FINAL_KEY).sum()),
        "cross_recording_label_or_metadata_inconsistencies": 0,
        "upstream_extraction_failures": 0,
        "predictor_count": len(predictors),
        "all_predictors_numeric": True,
        "invalid_values": invalid,
        "repeated_recording_channels": int(final["n_recordings"].gt(1).sum()),
        "pooled_epochs_per_channel": {
            "minimum": int(pooled.min()),
            "median": float(pooled.median()),
            "mean": float(pooled.mean()),
            "maximum": int(pooled.max()),
            "total": int(pooled.sum()),
            "required_minimum": counts_spec["minimum_pooled_epochs_per_channel"],
        },
        "pooling_policy": POOLING_POLICY,
        "aggregated_channel_artifact_policy": AGGREGATED_CHANNEL_ARTIFACT_POLICY,
        "channel_type_semantics": {
            "mne_signal_type": (
                "feature-Parquet MNE internal signal kind from epochs.get_channel_types()"
            ),
            "channel_type": (
                "authoritative clinical electrode modality from the original BIDS "
                "channels.tsv via the channel-label audit"
            ),
        },
        "mne_to_clinical_channel_type_mapping": channel_type_mapping.to_dict(orient="records"),
        "qc_metadata_excluded_from_predictors": ["n_recordings", "n_valid_epochs"],
        "metadata_excluded_from_predictors": [
            "participant_id", "channel_id", "dataset", "task", "mne_signal_type",
            "channel_type", "target", "n_recordings", "n_valid_epochs",
        ],
        "task_policy": (
            "exactly one recording state per experiment; interictal and ictal "
            "epochs are never pooled"
        ),
        "feature_selection": False,
        "imputation": False,
        "model_fitted": False,
    }


def feature_qc_tables(final: pd.DataFrame, aggregations: object) -> dict[str, pd.DataFrame]:
    predictors = predictor_names(aggregations)
    distributions = final[predictors].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).T.reset_index(names="feature")
    rows = []
    for feature in predictors:
        values = final[feature].to_numpy(dtype=float)
        scale = max(1.0, float(np.max(np.abs(values))))
        rows.append({
            "feature": feature,
            "constant": bool(np.ptp(values) == 0.0),
            "near_constant": bool(np.std(values, ddof=0) <= 1e-12 * scale),
            "dtype": str(final[feature].dtype),
        })
    quality = pd.DataFrame(rows)
    duplicates = []
    for index, left in enumerate(predictors):
        for right in predictors[index + 1:]:
            if final[left].equals(final[right]):
                duplicates.append({"feature_a": left, "feature_b": right})
    correlations = final[predictors].corr(method="pearson")
    redundant = []
    for index, left in enumerate(predictors):
        for right in predictors[index + 1:]:
            correlation = correlations.loc[left, right]
            if pd.notna(correlation) and abs(float(correlation)) >= 0.95:
                redundant.append({
                    "feature_a": left, "feature_b": right,
                    "pearson_correlation": float(correlation),
                })
    # Descriptive only: nothing is dropped here. Feature selection is out of
    # scope for this stage.
    return {
        "complexity_feature_distributions": distributions,
        "complexity_feature_quality": quality,
        "complexity_exact_duplicate_features": pd.DataFrame(
            duplicates, columns=["feature_a", "feature_b"]
        ),
        "complexity_high_correlations": pd.DataFrame(
            redundant, columns=["feature_a", "feature_b", "pearson_correlation"]
        ),
    }


def complexity_task_output_root(
    task: str, analysis_root: Path = DEFAULT_ANALYSIS_ROOT
) -> Path:
    """Own output root, so a complexity run can never overwrite a baseline run."""
    return Path(analysis_root) / PROTOCOL_VERSION / resolve_task(task)


def validate_inputs_only(
    epoch_path: Path, cohort_csv: Path, channel_labels_csv: Path, aggregations: object
) -> dict[str, Any]:
    """Cheap real-data smoke check: sidecars and Parquet footer schema only."""
    import pyarrow.parquet as pq

    statistics = resolve_aggregations(aggregations)
    epoch_path = Path(epoch_path)
    if epoch_path.name == AGGREGATED_CHANNEL_ARTIFACT:
        raise ValueError(
            f"refusing to read {AGGREGATED_CHANNEL_ARTIFACT}: {AGGREGATED_CHANNEL_ARTIFACT_POLICY}"
        )
    if not epoch_path.is_file():
        raise FileNotFoundError(f"missing epoch Parquet: {epoch_path}")
    if not Path(channel_labels_csv).is_file():
        raise FileNotFoundError(f"missing channel-label audit: {channel_labels_csv}")
    schema = set(pq.ParquetFile(epoch_path).schema_arrow.names)
    required = set(EPOCH_KEY + AGREEMENT_COLUMNS + source_feature_names())
    missing = sorted(required - schema)
    if missing:
        raise ValueError(f"complexity Parquet schema lacks: {missing}")
    cohort_hash = sha256(Path(cohort_csv))
    qc_path = epoch_path.parent / FAMILY_FILES[FAMILY]["qc"]
    summary_path = epoch_path.parent / FAMILY_FILES[FAMILY]["summary"]
    if not qc_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("missing complexity QC/run-summary sidecar")
    qc = pd.read_csv(qc_path)
    if qc.empty or not qc["status"].astype(str).eq("completed").all():
        raise RuntimeError("complexity QC is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("cohort_csv_sha256") != cohort_hash:
        raise ValueError("complexity provenance has a different cohort hash")
    if int(summary.get("recordings_failed", 0)) != 0:
        raise RuntimeError("complexity summary reports extraction failures")
    return {
        "status": "passed",
        "validation_scope": (
            "Parquet footer schema, row count, extraction sidecars, and cohort provenance"
        ),
        "epoch_rows": pq.ParquetFile(epoch_path).metadata.num_rows,
        "channel_epochs_excluded_aperiodic_fit": summary.get(
            "channel_epochs_excluded_aperiodic_fit"
        ),
        "feature_counts": feature_counts(statistics),
        "base_feature_count": len(base_feature_names()),
        "predictor_count_after_aggregation": len(predictor_names(statistics)),
    }


def build_dataset(
    epoch_path: Path,
    cohort_csv: Path,
    channel_labels_csv: Path,
    output_dir: Path,
    task: str,
    aggregations: object,
    repository: Path = REPOSITORY,
) -> Path:
    task = resolve_task(task)
    statistics = resolve_aggregations(aggregations)
    epoch_path = Path(epoch_path)
    # Same upstream gate as the baseline: QC completeness, zero failed windows /
    # NaN / Inf counters, and a cohort hash matching this cohort CSV.
    provenance = _validate_upstream_artifacts({FAMILY: epoch_path}, Path(cohort_csv))
    extraction_summary = json.loads(
        (epoch_path.parent / FAMILY_FILES[FAMILY]["summary"]).read_text(encoding="utf-8")
    )
    epochs = validate_complexity_epochs(read_complexity_epochs(epoch_path))
    cohort = pd.read_csv(cohort_csv, dtype=str, keep_default_na=False)
    labels = pd.read_csv(channel_labels_csv, dtype=str, keep_default_na=False)
    final = aggregate_participant_channels(epochs, cohort, labels, task, statistics)
    qc = dataset_qc(final, task, statistics)
    tables = {**feature_qc_tables(final, statistics), **cohort_count_tables(final)}
    with new_output_directory(Path(output_dir)) as staging:
        atomic_parquet(final, staging / "ml_complexity_participant_channel_dataset.parquet")
        atomic_csv(final, staging / "ml_complexity_participant_channel_dataset.csv")
        atomic_json(
            {
                "task": task,
                "family": FAMILY,
                "source_features": source_feature_names(),
                "base_features": base_feature_names(),
                "aggregations": list(statistics),
                "predictors": predictor_names(statistics),
                **feature_counts(statistics),
            },
            staging / "complexity_feature_list.json",
        )
        atomic_json(qc, staging / "complexity_dataset_qc_summary.json")
        for name, table in tables.items():
            atomic_csv(table, staging / f"{name}.csv")
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "stage": "complexity_dataset_construction",
            "task": task,
            "timestamp_utc": timestamp_utc(),
            "repository": str(Path(repository).resolve()),
            "git_commit": git_commit(Path(repository)),
            "python_version": platform.python_version(),
            "input_provenance": {
                "cohort_csv": str(Path(cohort_csv).resolve()),
                "cohort_csv_sha256": sha256(Path(cohort_csv)),
                "channel_labels_csv": str(Path(channel_labels_csv).resolve()),
                "channel_labels_csv_sha256": sha256(Path(channel_labels_csv)),
                "feature_families": provenance,
            },
            "epoch_merge_key": EPOCH_KEY,
            "final_observation_key": FINAL_KEY,
            "epoch_rows_read": len(epochs),
            "counts": qc["counts"],
            "feature_schema": {
                **feature_counts(statistics),
                "base_features": base_feature_names(),
                "predictors": predictor_names(statistics),
            },
            "aggregation_selection": {
                "requested": list(statistics),
                "supported": list(SUPPORTED_AGGREGATIONS),
                "definitions": {name: AGGREGATION_DEFINITIONS[name] for name in statistics},
                "explicitly_specified": True,
                "minimum_pooled_epochs_per_channel": minimum_epochs_per_channel(statistics),
            },
            "channel_type_semantics": qc["channel_type_semantics"],
            "mne_to_clinical_channel_type_mapping": qc["mne_to_clinical_channel_type_mapping"],
            "pooling_policy": POOLING_POLICY,
            "aggregated_channel_artifact_policy": AGGREGATED_CHANNEL_ARTIFACT_POLICY,
            "pooled_epochs_per_channel": qc["pooled_epochs_per_channel"],
            "upstream_extraction": {
                "channel_epochs_excluded_aperiodic_fit": extraction_summary.get(
                    "channel_epochs_excluded_aperiodic_fit"
                ),
                "aperiodic_fit_exclusion_policy": extraction_summary.get(
                    "aperiodic_fit_exclusion_policy"
                ),
                "invalid_epoch_policy": extraction_summary.get("invalid_epoch_policy"),
                "total_epoch_rows": extraction_summary.get("total_epoch_rows"),
            },
            "task_selection": (
                f"cohort recording_state == {task!r} only; the other recording state is "
                "never merged into this dataset"
            ),
            "family_selection": {
                "families": [FAMILY],
                "baseline_relationship": (
                    "separate dataset; the logistic baseline's three families are not "
                    "read here and this matrix is never merged into the baseline"
                ),
            },
            "clinical_selection": (
                "surgical_outcome==1 participants only; y=1 existing high-confidence SOZ; "
                "y=0 existing high-confidence preserved non-SOZ; ambiguous/unlabelled excluded"
            ),
            "shared_with_baseline": (
                "task selection, seizure-free filter, channel-label audit, channel "
                "identity, epoch validity and finite-value checks are imported from "
                "scripts.ml.dataset and applied unchanged"
            ),
            "not_performed_in_this_stage": [
                "feature_selection", "scaling", "model_fitting", "cross_validation",
                "multiscale_entropy_policy_decision",
            ],
        }
        atomic_json(summary, staging / "complexity_dataset_run_summary.json")
    return Path(output_dir) / "ml_complexity_participant_channel_dataset.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASKS),
        help=(
            "recording state for this experiment; exactly one is allowed. "
            "The seizure-free cohort restriction is a fixed protocol rule and "
            "is deliberately not configurable."
        ),
    )
    parser.add_argument(
        "--aggregations",
        nargs="+",
        required=True,
        choices=list(SUPPORTED_AGGREGATIONS),
        help=(
            "epoch-to-channel aggregation statistics; required, with no default, so "
            "mean/std is never silently chosen over mean/std/median. The predictor "
            f"order always follows {list(SUPPORTED_AGGREGATIONS)}."
        ),
    )
    parser.add_argument(
        "--complexity-epochs",
        type=Path,
        default=(
            DEFAULT_ANALYSIS_ROOT
            / FAMILY_DIRECTORIES[FAMILY]
            / FAMILY_FILES[FAMILY]["epochs"]
        ),
        help="raw epoch-level table; the aggregated channels Parquet is rejected",
    )
    parser.add_argument(
        "--cohort-csv", type=Path,
        default=DEFAULT_ANALYSIS_ROOT / "feature_cohort" / "nonictal_feature_cohort_hup_ucla.csv",
    )
    parser.add_argument(
        "--channel-labels-csv", type=Path,
        default=DEFAULT_ANALYSIS_ROOT / "feature_cohort" / "channel_label_audit_hup_ucla.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"defaults to <analysis root>/{PROTOCOL_VERSION}/<task>/dataset",
    )
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--validate-inputs-only", action="store_true")
    args = parser.parse_args()
    task = resolve_task(args.task)
    statistics = resolve_aggregations(args.aggregations)
    epoch_path = args.complexity_epochs.resolve()
    if args.validate_inputs_only:
        result = validate_inputs_only(
            epoch_path,
            args.cohort_csv.resolve(),
            args.channel_labels_csv.resolve(),
            statistics,
        )
        result["task"] = task
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    output_dir = args.output_dir or (complexity_task_output_root(task) / "dataset")
    output = build_dataset(
        epoch_path,
        args.cohort_csv.resolve(),
        args.channel_labels_csv.resolve(),
        output_dir.resolve(),
        task,
        statistics,
        args.repository.resolve(),
    )
    print(output)


if __name__ == "__main__":
    main()
