"""Construct the strict seizure-free participant-by-channel ML dataset.

The baseline matrix is exactly three feature families -- RelativeBandPower (18),
TimeDomainFeatures (12) and FrequencyDomainFeatures (2) -- giving 32 epoch
features and 96 mean/std/median predictors. ComplexityFeatures is extracted and
stored in full but is held out of the baseline for a separate model; see
:data:`HELD_OUT_FAMILIES`.

Exactly one recording state (``--task interictal`` or ``--task ictal``) is used
per experiment; ictal and interictal epochs are never pooled. The seizure-free
participant restriction is a fixed protocol rule and is therefore not
configurable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.ml.common import (
    DEFAULT_ANALYSIS_ROOT,
    PROTOCOL_VERSION,
    TASKS,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    git_commit,
    new_output_directory,
    resolve_task,
    sha256,
    task_output_root,
    timestamp_utc,
)


REPOSITORY = Path(__file__).resolve().parents[2]
EPOCH_KEY = [
    "dataset",
    "site",
    "participant_id",
    "recording_id",
    "channel",
    "channel_type",
    "epoch_index",
    "epoch_start_s",
]
FINAL_KEY = ["participant_id", "channel_id"]
AGGREGATIONS = ("mean", "std", "median")
# Validation-only reference counts from the frozen interictal v2 run. They are
# never enforced; they only document expected drift per task.
EXPECTED_COHORT_COUNTS: dict[str, dict[str, int]] = {
    "interictal": {
        "participants": 29,
        "observations": 2189,
        "pathological": 228,
        "non_pathological": 1961,
    },
}
FAMILY_FILES = {
    "time_domain": {
        "epochs": "time_domain_epochs_hup_ucla.parquet",
        "qc": "time_domain_extraction_qc.csv",
        "summary": "time_domain_run_summary.json",
    },
    "relative_bandpower": {
        "epochs": "relative_bandpower_epochs_hup_ucla.parquet",
        "qc": "relative_bandpower_extraction_qc.csv",
        "summary": "relative_bandpower_run_summary.json",
    },
    "frequency_domain": {
        "epochs": "frequency_domain_epochs_hup_ucla.parquet",
        "qc": "frequency_domain_extraction_qc.csv",
        "summary": "frequency_domain_run_summary.json",
    },
    "complexity": {
        "epochs": "complexity_epochs_hup_ucla.parquet",
        "qc": "complexity_extraction_qc.csv",
        "summary": "complexity_run_summary.json",
    },
}
# Default cohort sub-directory produced by each assemble_*_cohort.py stage.
FAMILY_DIRECTORIES = {
    "time_domain": "time_domain_neurodags_cohort_v1",
    "relative_bandpower": "relative_bandpower_neurodags_cohort_v1",
    "frequency_domain": "frequency_domain_neurodags_cohort_v1",
    "complexity": "complexity_neurodags_cohort_v1",
}
# Baseline family order, and the deterministic predictor order.
#
# The baseline is exactly three families: RelativeBandPower (18) +
# TimeDomainFeatures (12) + FrequencyDomainFeatures (2) = 32 epoch features,
# aggregated by {mean, sample std, median} into 96 predictors.
FAMILY_ORDER = ("frequency_domain", "relative_bandpower", "time_domain")
BASELINE_EPOCH_FEATURE_COUNT = 32
BASELINE_PREDICTOR_COUNT = 96

# ComplexityFeatures is extracted, assembled and QC-reported in full (all 14
# Mantilla-Ramos features), but is deliberately NOT part of the baseline ML
# matrix: it is held out for a separate model. Merging it here would silently
# change the baseline's dimensionality, so it is named explicitly rather than
# simply omitted.
HELD_OUT_FAMILIES = ("complexity",)
HELD_OUT_REASON = (
    "extracted and stored in full, reserved for a separate complexity model; "
    "never merged into the baseline predictor matrix"
)
# Namespacing that a held-out family would need if it were ever added: four of
# its names collide with the frequency- and time-domain families. Unused while
# the family is held out.
FAMILY_COLUMN_PREFIXES = {"complexity": "complexity_"}

if set(FAMILY_ORDER) & set(HELD_OUT_FAMILIES):
    raise AssertionError(
        f"held-out families must not appear in the baseline: {FAMILY_ORDER}"
    )
AGREEMENT_COLUMNS = [
    "supervised_label",
    "label_reason",
    "outcome",
    "soz",
    "resected",
    "task",
    "epoch_valid",
    "channel_epoch_valid",
]


def _source_feature_names() -> dict[str, list[str]]:
    """Read the frozen scientific feature constants; never redefine them here."""
    frequency = importlib.import_module("scripts.features.frequency_domain")
    power = importlib.import_module("scripts.features.relative_bandpower")
    time_domain = importlib.import_module("scripts.features.time_domain")
    names = {
        "frequency_domain": list(frequency.FEATURE_NAMES),
        "relative_bandpower": (
            list(power.TOTAL_POWER_NAMES)
            + list(power.PAPER_FEATURE_NAMES)
            + list(power.EXTENDED_FEATURE_NAMES)
            + [power.HF_FEATURE_NAME]
        ),
        "time_domain": list(time_domain.FEATURE_NAMES),
    }
    if set(names) != set(FAMILY_ORDER):
        raise AssertionError(f"family order {FAMILY_ORDER} does not cover {sorted(names)}")
    for family, features in names.items():
        if not features:
            raise RuntimeError(f"{family} exposes no feature constants")
        if len(set(features)) != len(features):
            raise RuntimeError(f"{family} feature constants contain duplicates: {features}")
    return names


def source_feature_names() -> dict[str, list[str]]:
    """Per-family feature names exactly as the extraction Parquet stores them."""
    return _source_feature_names()


def _feature_names() -> dict[str, list[str]]:
    """Per-family ML column names, namespaced where families reuse a name."""
    names = {
        family: [
            f"{FAMILY_COLUMN_PREFIXES.get(family, '')}{feature}"
            for feature in features
        ]
        for family, features in _source_feature_names().items()
    }
    flat = [name for family in FAMILY_ORDER for name in names[family]]
    duplicates = sorted({name for name in flat if flat.count(name) > 1})
    if duplicates:
        raise RuntimeError(
            "feature families expose colliding ML column names; add a prefix in "
            f"FAMILY_COLUMN_PREFIXES for: {duplicates}"
        )
    return names


def base_feature_names() -> list[str]:
    names = _feature_names()
    return [name for family in FAMILY_ORDER for name in names[family]]


def predictor_names() -> list[str]:
    return [f"{feature}_{statistic}" for feature in base_feature_names() for statistic in AGGREGATIONS]


def feature_counts() -> dict[str, Any]:
    """Feature counts derived from the constants; nothing here is hard-coded."""
    names = _feature_names()
    base = base_feature_names()
    predictors = predictor_names()
    if len(predictors) != len(base) * len(AGGREGATIONS):
        raise AssertionError(
            "predictor count must equal epoch features times aggregations: "
            f"{len(predictors)} != {len(base)} x {len(AGGREGATIONS)}"
        )
    if len(set(predictors)) != len(predictors):
        raise AssertionError("aggregated predictor names are not unique")
    if len(base) != BASELINE_EPOCH_FEATURE_COUNT or len(predictors) != BASELINE_PREDICTOR_COUNT:
        raise AssertionError(
            "the baseline is fixed at "
            f"{BASELINE_EPOCH_FEATURE_COUNT} epoch features / "
            f"{BASELINE_PREDICTOR_COUNT} predictors "
            f"({', '.join(FAMILY_ORDER)}); derived "
            f"{len(base)}/{len(predictors)} from "
            f"{ {family: len(names[family]) for family in FAMILY_ORDER} }"
        )
    return {
        "epoch_features_per_family": {family: len(names[family]) for family in FAMILY_ORDER},
        "epoch_feature_count": len(base),
        "aggregations": list(AGGREGATIONS),
        "predictor_count": len(predictors),
        "baseline_families": list(FAMILY_ORDER),
        "held_out_families": list(HELD_OUT_FAMILIES),
        "held_out_reason": HELD_OUT_REASON,
    }


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def surgical_outcome(value: object) -> int | None:
    """Canonical surgical-outcome parser, shared with cohort and extraction.

    Loaded from the feature core so the ML filter, the channel audit and the
    epoch labels can never disagree about what "seizure-free" means.
    """
    core = importlib.import_module("scripts.features.relative_bandpower")
    return core.parse_surgical_outcome(value)


def _strict_binary(value: object) -> int | None:
    text = _clean(value).lower()
    if text in {"1", "1.0", "true", "yes"}:
        return 1
    if text in {"0", "0.0", "false", "no"}:
        return 0
    return None


def _canonical_metadata(column: str, series: pd.Series) -> pd.Series:
    if column in {"supervised_label", "outcome", "soz", "resected"}:
        return series.map(_strict_binary).astype("Int64")
    if column in {"epoch_valid", "channel_epoch_valid"}:
        return series.map(_strict_binary).astype("Int64")
    return series.map(_clean).str.lower()


def validate_feature_family(frame: pd.DataFrame, family: str, features: list[str]) -> pd.DataFrame:
    """Validate one family's epoch table and return it with ML column names.

    ``features`` are the namespaced ML names; the Parquet stores the source
    names, so the family prefix is applied here and nowhere else.
    """
    prefix = FAMILY_COLUMN_PREFIXES.get(family, "")
    source_features = [name.removeprefix(prefix) if prefix else name for name in features]
    renames = dict(zip(source_features, features))
    required = set(EPOCH_KEY + AGREEMENT_COLUMNS + source_features)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{family} epoch table lacks required columns: {missing}")
    selected = frame[EPOCH_KEY + AGREEMENT_COLUMNS + source_features].rename(columns=renames).copy()
    selected["dataset"] = selected["dataset"].map(_clean).str.lower()
    selected["site"] = selected["site"].map(_clean).str.lower()
    if not selected["dataset"].equals(selected["site"]):
        raise ValueError(f"{family} has disagreeing dataset and site values")
    selected["epoch_index"] = pd.to_numeric(selected["epoch_index"], errors="raise").astype("int64")
    selected["epoch_start_s"] = pd.to_numeric(selected["epoch_start_s"], errors="raise")
    if selected.duplicated(EPOCH_KEY).any():
        examples = selected.loc[selected.duplicated(EPOCH_KEY, keep=False), EPOCH_KEY].head(5)
        raise ValueError(f"{family} epoch identities are not uniquCONFIGURE_WINDOWS_PATH")
    for validity in ("epoch_valid", "channel_epoch_valid"):
        if not _canonical_metadata(validity, selected[validity]).eq(1).all():
            raise ValueError(f"{family} contains rows not explicitly marked valid in {validity}")
    for feature in features:
        selected[feature] = pd.to_numeric(selected[feature], errors="raise")
    values = selected[features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = pd.DataFrame({
            "feature": features,
            "nan": np.isnan(values).sum(axis=0),
            "positive_inf": np.isposinf(values).sum(axis=0),
            "negative_inf": np.isneginf(values).sum(axis=0),
        })
        bad = bad[(bad[["nan", "positive_inf", "negative_inf"]] > 0).any(axis=1)]
        raise ValueError(f"{family} contains invalid feature valueCONFIGURE_WINDOWS_PATH")
    return selected


def merge_epoch_feature_families(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    family_features = _feature_names()
    if set(frames) != set(family_features):
        raise ValueError(f"expected feature families {sorted(family_features)}, got {sorted(frames)}")
    validated = {
        family: validate_feature_family(frames[family], family, features)
        for family, features in family_features.items()
    }
    reference_name = "relative_bandpower"
    reference = validated[reference_name].sort_values(EPOCH_KEY, kind="stable").reset_index(drop=True)
    reference_keys = pd.MultiIndex.from_frame(reference[EPOCH_KEY])
    for family, current in validated.items():
        if family == reference_name:
            continue
        current = current.sort_values(EPOCH_KEY, kind="stable").reset_index(drop=True)
        current_keys = pd.MultiIndex.from_frame(current[EPOCH_KEY])
        if len(current) != len(reference) or not current_keys.equals(reference_keys):
            missing = reference_keys.difference(current_keys)
            extra = current_keys.difference(reference_keys)
            raise ValueError(
                f"epoch identities differ between {reference_name} and {family}: "
                f"rows={len(reference)}/{len(current)}, missing={len(missing)}, extra={len(extra)}"
            )
        for column in AGREEMENT_COLUMNS:
            left = _canonical_metadata(column, reference[column])
            right = _canonical_metadata(column, current[column])
            if not left.equals(right):
                disagreements = int((left.astype("string").fillna("<NA>") != right.astype("string").fillna("<NA>")).sum())
                raise ValueError(
                    f"metadata column {column!r} disagrees for {disagreements} epochs "
                    f"between {reference_name} and {family}"
                )
    merged = reference[EPOCH_KEY + AGREEMENT_COLUMNS + family_features[reference_name]].copy()
    for family in (name for name in FAMILY_ORDER if name != reference_name):
        merged = merged.merge(
            validated[family][EPOCH_KEY + family_features[family]],
            on=EPOCH_KEY,
            how="inner",
            validate="one_to_one",
        )
    ordered = EPOCH_KEY + AGREEMENT_COLUMNS + base_feature_names()
    merged = merged[ordered]
    if len(merged) != len(reference):
        raise AssertionError("one-to-one epoch merge unexpectedly changed the row count")
    return merged


def _recording_key(row: pd.Series) -> str:
    """Recording identity used by the feature extractors (EDF stem fallback)."""
    explicit = _clean(row.get("recording_id"))
    if explicit:
        return explicit
    name = Path(_clean(row.get("edf_path"))).name
    return name[:-4] if name.lower().endswith(".edf") else Path(name).stem


def select_task_recordings(cohort: pd.DataFrame, task: str) -> pd.DataFrame:
    """Keep only the cohort recordings whose canonical state is the given task.

    ``recording_state`` is authoritative: it already maps every UCLA task label
    (for example ``sleep``) onto ``interictal`` via the frozen cohort protocol,
    while the raw BIDS ``task`` column is preserved as metadata.
    """
    task = resolve_task(task)
    required = {"participant_id", "dataset", "recording_state", "feature_cohort_eligible"}
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise ValueError(f"cohort lacks required columns: {missing}")
    work = cohort.copy()
    work["recording_state"] = work["recording_state"].map(_clean).str.lower()
    unknown = sorted(set(work["recording_state"]) - set(TASKS))
    if unknown:
        raise ValueError(
            f"cohort contains recordings whose state is outside {list(TASKS)}: {unknown}"
        )
    work["recording_key"] = work.apply(_recording_key, axis=1)
    if work["recording_key"].eq("").any():
        raise ValueError("cohort contains a recording without an identifiable recording_id")
    selected = work[work["recording_state"].eq(task)].copy()
    if selected.empty:
        raise ValueError(f"cohort contains no {task} recording")
    return selected


def filter_epochs_to_task(
    merged_epochs: pd.DataFrame, task_cohort: pd.DataFrame, task: str
) -> pd.DataFrame:
    """Restrict merged epochs to the selected task's recordings only."""
    task = resolve_task(task)
    keys = set(
        zip(
            task_cohort["dataset"].map(_clean).str.lower(),
            task_cohort["participant_id"].map(_clean),
            task_cohort["recording_key"],
        )
    )
    work = merged_epochs.copy()
    identity = pd.MultiIndex.from_arrays(
        [
            work["dataset"].map(_clean).str.lower(),
            work["participant_id"].map(_clean),
            work["recording_id"].map(_clean),
        ]
    )
    selected = work[identity.isin(keys)].copy()
    if selected.empty:
        raise ValueError(f"no epoch rows belong to a {task} cohort recording")
    # Defence in depth: any epoch whose own BIDS task is an explicit recording
    # state must agree with the selected task, so the two are never pooled.
    epoch_task = selected["task"].map(_clean).str.lower()
    leaked = sorted(set(epoch_task[epoch_task.isin(TASKS) & epoch_task.ne(task)]))
    if leaked:
        raise ValueError(
            f"{task} selection retained epochs labelled {leaked}; tasks must never be mixed"
        )
    return selected.reset_index(drop=True)


def strict_seizure_free_participants(cohort: pd.DataFrame) -> set[str]:
    required = {
        "participant_id", "dataset", "original_sampling_frequency_hz",
        "surgical_outcome", "recording_state", "feature_cohort_eligible",
    }
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise ValueError(f"cohort lacks required columns: {missing}")
    work = cohort.copy()
    work["participant_id"] = work["participant_id"].map(_clean)
    if work["participant_id"].eq("").any():
        raise ValueError("cohort contains an empty participant_id")
    work["dataset"] = work["dataset"].map(_clean).str.lower()
    if not work["dataset"].isin({"hup", "ucla"}).all():
        raise ValueError("cohort contains a dataset outside HUP/UCLA")
    states = set(work["recording_state"].map(_clean).str.lower())
    if len(states) != 1 or not states.issubset(set(TASKS)):
        raise ValueError(
            f"the seizure-free filter must receive exactly one recording state; got {sorted(states)}"
        )
    if not work["feature_cohort_eligible"].map(_strict_binary).eq(1).all():
        raise ValueError("cohort contains a recording not marked feature-cohort eligible")
    sfreq = pd.to_numeric(work["original_sampling_frequency_hz"], errors="coerce")
    if sfreq.isna().any() or (sfreq < 1000.0).any():
        raise ValueError("cohort contains missing/malformed or <1000 Hz sampling frequency")
    # Canonical parsing: a present but unrecognised encoding raises here rather
    # than becoming "unknown" and silently removing the participant.
    outcome = work["surgical_outcome"].map(surgical_outcome)
    work["_outcome"] = outcome
    conflicts = []
    for participant, group in work.groupby("participant_id", sort=True):
        observed = set(
            group["_outcome"].map(
                lambda value: "unknown" if pd.isna(value) else str(int(value))
            )
        )
        if len(observed) > 1:
            conflicts.append(participant)
    if conflicts:
        raise ValueError(f"conflicting participant surgical outcomes: {conflicts}")
    selected = set(work.loc[work["_outcome"].eq(1), "participant_id"])
    # Excluded participants are recorded, never silently dropped.
    excluded = {
        "not_seizure_free": sorted(set(work.loc[work["_outcome"].eq(0), "participant_id"])),
        "outcome_absent": sorted(set(work.loc[work["_outcome"].isna(), "participant_id"])),
    }
    strict_seizure_free_participants.last_exclusions = excluded
    return selected


def strict_channel_labels(labels: pd.DataFrame, participants: set[str]) -> pd.DataFrame:
    required = {
        "participant_id", "dataset", "channel_name", "channel_type",
        "participant_surgical_outcome", "high_confidence_positive",
        "high_confidence_negative", "ambiguous_supervised_label",
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"channel-label audit lacks required columns: {missing}")
    work = labels.copy()
    work["participant_id"] = work["participant_id"].map(_clean)
    work = work[work["participant_id"].isin(participants)].copy()
    work["dataset"] = work["dataset"].map(_clean).str.lower()
    work["channel_id"] = work["channel_name"].map(_clean)
    if work["channel_id"].eq("").any():
        raise ValueError("selected channel-label audit contains an empty channel identity")
    work["channel_type"] = work["channel_type"].map(_clean).str.upper()
    work["participant_surgical_outcome"] = work["participant_surgical_outcome"].map(
        surgical_outcome
    )
    for column in (
        "high_confidence_positive", "high_confidence_negative",
        "ambiguous_supervised_label",
    ):
        work[column] = work[column].map(_strict_binary)
    if not work["participant_surgical_outcome"].eq(1).all():
        raise ValueError("selected participant has non-seizure-free/unknown outcome in label audit")
    label_flags = work[
        ["high_confidence_positive", "high_confidence_negative", "ambiguous_supervised_label"]
    ]
    if label_flags.isna().any().any() or not label_flags.sum(axis=1).eq(1).all():
        raise ValueError(
            "every channel-label audit row must have exactly one explicit positive, "
            "negative, or ambiguous state"
        )
    work["label_state"] = np.select(
        [
            work["high_confidence_positive"].eq(1),
            work["high_confidence_negative"].eq(1),
        ],
        ["pathological", "non_pathological"],
        default="ambiguous",
    )
    conflict_columns = ["dataset", "channel_type", "label_state"]
    conflicts = (
        work.groupby(FINAL_KEY, dropna=False)[conflict_columns]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if conflicts.any():
        examples = conflicts[conflicts].index.tolist()[:10]
        raise ValueError(f"cross-recording channel-label/type/site conflicts: {examples}")
    labelled = work[work["label_state"].ne("ambiguous")].copy()
    labelled["target"] = labelled["label_state"].map(
        {"pathological": 1, "non_pathological": 0}
    )
    return (
        labelled.sort_values(FINAL_KEY, kind="stable")
        .drop_duplicates(FINAL_KEY)[FINAL_KEY + ["dataset", "channel_type", "target"]]
        .astype({"target": "int64"})
        .reset_index(drop=True)
    )


def aggregate_participant_channels(
    merged_epochs: pd.DataFrame,
    cohort: pd.DataFrame,
    labels: pd.DataFrame,
    task: str,
) -> pd.DataFrame:
    """Select one task, then apply the fixed seizure-free protocol and aggregate."""
    task = resolve_task(task)
    task_cohort = select_task_recordings(cohort, task)
    task_epochs = filter_epochs_to_task(merged_epochs, task_cohort, task)
    participants = strict_seizure_free_participants(task_cohort)
    channel_labels = strict_channel_labels(labels, participants)
    work = task_epochs[task_epochs["participant_id"].isin(participants)].copy()
    work = work.rename(
        columns={"channel": "channel_id", "channel_type": "mne_signal_type"}
    )
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
    features = base_feature_names()
    grouped = work.groupby(FINAL_KEY, sort=True, dropna=False)
    valid_epoch_counts = grouped.size()
    insufficient = valid_epoch_counts[valid_epoch_counts < 2]
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
            "sample standard deviation (ddof=1) requires at least two pooled valid "
            f"epochs per participant/channel; found {len(insufficient)} insufficient "
            f"channels; examples={examples}"
        )
    aggregated = grouped[features].agg(["mean", "std", "median"])
    aggregated.columns = [f"{feature}_{stat}" for feature, stat in aggregated.columns]
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
    expected_predictors = predictor_names()
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
    counts = feature_counts()
    if len(expected_predictors) != counts["predictor_count"] or not set(
        expected_predictors
    ).issubset(final.columns):
        raise AssertionError(
            f"the final dataset must contain exactly {counts['predictor_count']} predictors"
        )
    return final


def dataset_qc(final: pd.DataFrame, task: str) -> dict[str, Any]:
    task = resolve_task(task)
    observed_tasks = sorted(set(final["task"].map(_clean).str.lower()))
    if observed_tasks != [task]:
        raise ValueError(f"dataset must hold exactly the {task} task; found {observed_tasks}")
    counts_spec = feature_counts()
    predictors = predictor_names()
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
    reference_counts = EXPECTED_COHORT_COUNTS.get(task, {})
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
        "counts": counts,
        "feature_counts": counts_spec,
        "expected_counts_are_validation_only": reference_counts,
        "count_differences_observed_minus_expected": {
            key: counts[key] - value for key, value in reference_counts.items()
        },
        "duplicate_final_rows": int(final.duplicated(FINAL_KEY).sum()),
        "cross_recording_label_or_metadata_inconsistencies": 0,
        "upstream_extraction_failures": 0,
        "predictor_count": len(predictors),
        "all_predictors_numeric": True,
        "invalid_values": invalid,
        "repeated_recording_channels": int(final["n_recordings"].gt(1).sum()),
        "pooling_policy": (
            "all valid epochs are pooled across eligible recordings before mean, sample "
            "standard deviation, and median aggregation; recordings with more valid epochs "
            "therefore contribute proportionally more observations"
        ),
        "channel_type_semantics": {
            "mne_signal_type": (
                "feature-Parquet MNE internal signal kind from epochs.get_channel_types()"
            ),
            "channel_type": (
                "authoritative clinical electrode modality from the original BIDS "
                "channels.tsv via the channel-label audit"
            ),
        },
        "mne_to_clinical_channel_type_mapping": channel_type_mapping.to_dict(
            orient="records"
        ),
        "qc_metadata_excluded_from_predictors": ["n_recordings", "n_valid_epochs"],
        "metadata_excluded_from_predictors": [
            "participant_id", "channel_id", "dataset", "task", "mne_signal_type",
            "channel_type", "target", "n_recordings", "n_valid_epochs",
        ],
        "task_policy": (
            "exactly one recording state per experiment; interictal and ictal "
            "epochs are never pooled"
        ),
    }


def feature_qc_tables(final: pd.DataFrame) -> dict[str, pd.DataFrame]:
    predictors = predictor_names()
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
    return {
        "feature_distributions": distributions,
        "feature_quality": quality,
        "exact_duplicate_features": pd.DataFrame(duplicates, columns=["feature_a", "feature_b"]),
        "high_correlations": pd.DataFrame(
            redundant, columns=["feature_a", "feature_b", "pearson_correlation"]
        ),
    }


def cohort_count_tables(final: pd.DataFrame) -> dict[str, pd.DataFrame]:
    per_participant = (
        final.groupby(["participant_id", "dataset"], sort=True)["target"]
        .agg(
            n_channels="size",
            pathological=lambda values: int((values == 1).sum()),
            non_pathological=lambda values: int((values == 0).sum()),
        )
        .reset_index()
    )
    per_site = (
        final.groupby("dataset", sort=True)["target"]
        .agg(
            n_channels="size",
            pathological=lambda values: int((values == 1).sum()),
            non_pathological=lambda values: int((values == 0).sum()),
        )
        .reset_index()
    )
    return {"counts_per_participant": per_participant, "counts_per_site": per_site}


def _validate_upstream_artifacts(
    epoch_paths: dict[str, Path], cohort_csv: Path
) -> dict[str, Any]:
    cohort_hash = sha256(cohort_csv)
    provenance: dict[str, Any] = {}
    for family, epoch_path in epoch_paths.items():
        expected_name = FAMILY_FILES[family]["epochs"]
        if epoch_path.name != expected_name:
            raise ValueError(f"unexpected {family} epoch filename: {epoch_path.name}")
        qc_path = epoch_path.parent / FAMILY_FILES[family]["qc"]
        summary_path = epoch_path.parent / FAMILY_FILES[family]["summary"]
        for path in (epoch_path, qc_path, summary_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing required {family} artifact: {path}")
        qc = pd.read_csv(qc_path)
        if qc.empty or "status" not in qc or not qc["status"].astype(str).eq("completed").all():
            raise RuntimeError(f"{family} extraction QC contains incomplete/failed recordings")
        failures = int(pd.to_numeric(qc.get("n_failed_windows", 0), errors="coerce").fillna(0).sum()) if "n_failed_windows" in qc else 0
        nan_features = int(pd.to_numeric(qc.get("n_nan_features", 0), errors="coerce").fillna(0).sum()) if "n_nan_features" in qc else 0
        inf_features = int(pd.to_numeric(qc.get("n_infinite_features", 0), errors="coerce").fillna(0).sum()) if "n_infinite_features" in qc else 0
        if failures or nan_features or inf_features:
            raise RuntimeError(
                f"{family} extraction reports failures/invalid features: "
                f"failed_windows={failures}, nan={nan_features}, inf={inf_features}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("cohort_csv_sha256") != cohort_hash:
            raise ValueError(f"{family} run summary was built from a different cohort CSV")
        if int(summary.get("recordings_failed", 0)) != 0:
            raise RuntimeError(f"{family} run summary reports failed recordings")
        provenance[family] = {
            "epoch_parquet": str(epoch_path.resolve()),
            "epoch_parquet_sha256": sha256(epoch_path),
            "extraction_qc": str(qc_path.resolve()),
            "extraction_qc_sha256": sha256(qc_path),
            "run_summary": str(summary_path.resolve()),
            "run_summary_sha256": sha256(summary_path),
            "protocol_version": summary.get("protocol_version"),
            "git_commit": summary.get("git_commit"),
            "total_epoch_rows": summary.get("total_epoch_rows"),
        }
    return provenance


def validate_inputs_only(
    epoch_paths: dict[str, Path], cohort_csv: Path, channel_labels_csv: Path
) -> dict[str, Any]:
    """Cheap real-data smoke check: sidecars and Parquet footer schemas only."""
    import pyarrow.parquet as pq

    family_features = _feature_names()
    source_features = _source_feature_names()
    cohort_hash = sha256(cohort_csv)
    if not channel_labels_csv.is_file():
        raise FileNotFoundError(f"missing channel-label audit: {channel_labels_csv}")
    rows: dict[str, int] = {}
    for family, epoch_path in epoch_paths.items():
        if not epoch_path.is_file():
            raise FileNotFoundError(f"missing epoch Parquet: {epoch_path}")
        schema = set(pq.ParquetFile(epoch_path).schema_arrow.names)
        required = set(EPOCH_KEY + AGREEMENT_COLUMNS + source_features[family])
        missing = sorted(required - schema)
        if missing:
            raise ValueError(f"{family} Parquet schema lacks: {missing}")
        rows[family] = pq.ParquetFile(epoch_path).metadata.num_rows
        qc_path = epoch_path.parent / FAMILY_FILES[family]["qc"]
        summary_path = epoch_path.parent / FAMILY_FILES[family]["summary"]
        if not qc_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"missing {family} QC/run-summary sidecar")
        qc = pd.read_csv(qc_path)
        if qc.empty or not qc["status"].astype(str).eq("completed").all():
            raise RuntimeError(f"{family} QC is incomplete")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("cohort_csv_sha256") != cohort_hash:
            raise ValueError(f"{family} provenance has a different cohort hash")
        if int(summary.get("recordings_failed", 0)) != 0:
            raise RuntimeError(f"{family} summary reports extraction failures")
    if len(set(rows.values())) != 1:
        raise ValueError(f"feature-family Parquet row counts are incompatible: {rows}")
    return {
        "status": "passed",
        "validation_scope": "Parquet footer schemas, row counts, extraction sidecars, and cohort provenance",
        "epoch_rows": rows,
        "feature_counts": feature_counts(),
        "base_feature_count": len(base_feature_names()),
        "predictor_count_after_aggregation": len(predictor_names()),
    }


def build_dataset(
    epoch_paths: dict[str, Path],
    cohort_csv: Path,
    channel_labels_csv: Path,
    output_dir: Path,
    task: str,
    repository: Path = REPOSITORY,
) -> Path:
    task = resolve_task(task)
    provenance = _validate_upstream_artifacts(epoch_paths, cohort_csv)
    frames = {family: pd.read_parquet(path) for family, path in epoch_paths.items()}
    merged = merge_epoch_feature_families(frames)
    cohort = pd.read_csv(cohort_csv, dtype=str, keep_default_na=False)
    labels = pd.read_csv(channel_labels_csv, dtype=str, keep_default_na=False)
    final = aggregate_participant_channels(merged, cohort, labels, task)
    qc = dataset_qc(final, task)
    tables = {**feature_qc_tables(final), **cohort_count_tables(final)}
    with new_output_directory(output_dir) as staging:
        atomic_parquet(final, staging / "ml_participant_channel_dataset.parquet")
        atomic_csv(final, staging / "ml_participant_channel_dataset.csv")
        atomic_json(
            {
                "task": task,
                "families": {
                    family: _feature_names()[family] for family in FAMILY_ORDER
                },
                "base_features": base_feature_names(),
                "aggregations": list(AGGREGATIONS),
                "predictors": predictor_names(),
                **feature_counts(),
            },
            staging / "feature_list.json",
        )
        atomic_json(qc, staging / "dataset_qc_summary.json")
        for name, table in tables.items():
            atomic_csv(table, staging / f"{name}.csv")
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "stage": "dataset_construction",
            "task": task,
            "timestamp_utc": timestamp_utc(),
            "repository": str(repository.resolve()),
            "git_commit": git_commit(repository),
            "python_version": platform.python_version(),
            "input_provenance": {
                "cohort_csv": str(cohort_csv.resolve()),
                "cohort_csv_sha256": sha256(cohort_csv),
                "channel_labels_csv": str(channel_labels_csv.resolve()),
                "channel_labels_csv_sha256": sha256(channel_labels_csv),
                "feature_families": provenance,
            },
            "epoch_merge_key": EPOCH_KEY,
            "final_observation_key": FINAL_KEY,
            "epoch_rows_merged": len(merged),
            "counts": qc["counts"],
            "feature_schema": {
                **feature_counts(),
                "base_features": base_feature_names(),
                "predictors": predictor_names(),
            },
            "channel_type_semantics": qc["channel_type_semantics"],
            "mne_to_clinical_channel_type_mapping": qc[
                "mne_to_clinical_channel_type_mapping"
            ],
            "pooling_policy": qc["pooling_policy"],
            "task_selection": (
                f"cohort recording_state == {task!r} only; the other recording state is "
                "never merged into this dataset"
            ),
            "family_selection": {
                "baseline_families": list(FAMILY_ORDER),
                "held_out_families": list(HELD_OUT_FAMILIES),
                "held_out_reason": HELD_OUT_REASON,
            },
            "clinical_selection": (
                "surgical_outcome==1 participants only; y=1 existing high-confidence SOZ; "
                "y=0 existing high-confidence preserved non-SOZ; ambiguous/unlabelled excluded"
            ),
        }
        atomic_json(summary, staging / "dataset_run_summary.json")
    return output_dir / "ml_participant_channel_dataset.parquet"


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
    for family in FAMILY_ORDER:
        parser.add_argument(
            f"--{family.replace('_', '-')}-epochs",
            type=Path,
            default=(
                DEFAULT_ANALYSIS_ROOT
                / FAMILY_DIRECTORIES[family]
                / FAMILY_FILES[family]["epochs"]
            ),
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
        help="defaults to <analysis root>/ml_logistic_baseline_v1/<task>/dataset",
    )
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--validate-inputs-only", action="store_true")
    args = parser.parse_args()
    task = resolve_task(args.task)
    epoch_paths = {
        family: getattr(args, f"{family}_epochs").resolve() for family in FAMILY_ORDER
    }
    if args.validate_inputs_only:
        result = validate_inputs_only(
            epoch_paths, args.cohort_csv.resolve(), args.channel_labels_csv.resolve()
        )
        result["task"] = task
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    output_dir = args.output_dir or (task_output_root(task) / "dataset")
    output = build_dataset(
        epoch_paths,
        args.cohort_csv.resolve(),
        args.channel_labels_csv.resolve(),
        output_dir.resolve(),
        task,
        args.repository.resolve(),
    )
    print(output)


if __name__ == "__main__":
    main()
