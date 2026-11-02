#!/usr/bin/env python3
"""Scientific core for per-recording OMNI-iEEG time-domain features."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


REPOSITORY = Path(__file__).resolve().parents[2]
SHARED_SCRIPT = REPOSITORY / "scripts" / "features" / "relative_bandpower.py"
SHARED_MODULE_NAME = "omni_feature_workflow_shared"

PROTOCOL_VERSION = "1.0.0"
DERIVATIVE_NAME = "Preprocessed1kHzNoBandpass"
EXPECTED_PREPROCESSING_PROTOCOL_VERSION = "1.1.0"
EXPECTED_SFREQ_HZ = 1000.0
EXPECTED_SAMPLES_PER_EPOCH = 8000
EPOCH_DURATION_S = 8.0
NEARLY_CONSTANT_RELATIVE_TOLERANCE = 1e-12
# The channel-epoch validity policy is defined once, in relative_bandpower.py,
# and re-exported here so every family reports the same rule.
INVALID_EPOCH_POLICY = (
    "reject_only_the_degenerate_channel_epoch: a channel-epoch is used when it is "
    "one-dimensional, has exactly the expected sample count, is entirely finite, is "
    "not flat, and is not near-flat (sample std > 1e-12 * max(1, rms)); other "
    "channels in the same epoch are unaffected"
)

FEATURE_NAMES = [
    "mean",
    "median",
    "variance",
    "standard_deviation",
    "rms",
    "line_length",
    "zero_crossing_rate",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "skewness",
    "kurtosis",
]


def _shared_module():
    """Load workflow infrastructure shared with RelativeBandPower read-only."""
    if SHARED_MODULE_NAME in sys.modules:
        return sys.modules[SHARED_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(SHARED_MODULE_NAME, SHARED_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared feature infrastructure: {SHARED_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[SHARED_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _safe_variance(values: np.ndarray) -> float:
    return float(np.mean(np.square(values - np.mean(values))))


def _hjorth(values: np.ndarray) -> tuple[float, float, float]:
    activity = _safe_variance(values)
    first_difference = np.diff(values)
    first_variance = _safe_variance(first_difference) if first_difference.size else 0.0
    if activity <= 0.0 or first_variance <= 0.0:
        return activity, 0.0, 0.0
    mobility = math.sqrt(first_variance / activity)
    second_difference = np.diff(first_difference)
    second_variance = _safe_variance(second_difference) if second_difference.size else 0.0
    derivative_mobility = math.sqrt(second_variance / first_variance)
    complexity = derivative_mobility / mobility if mobility > 0.0 else 0.0
    return activity, float(mobility), float(complexity)


def compute_epoch_time_domain_features(
    signal: np.ndarray, sfreq: float = EXPECTED_SFREQ_HZ
) -> dict[str, float]:
    """Compute the frozen temporal feature protocol for one channel epoch."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("one-dimensional channel data are required")
    if values.size != EXPECTED_SAMPLES_PER_EPOCH:
        raise ValueError(
            f"epoch must contain exactly {EXPECTED_SAMPLES_PER_EPOCH} samples; "
            f"received {values.size}"
        )
    if not math.isclose(float(sfreq), EXPECTED_SFREQ_HZ, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"sampling frequency must be exactly {EXPECTED_SFREQ_HZ:g} Hz")
    if not np.all(np.isfinite(values)):
        raise ValueError("epoch contains NaN or infinite samples")

    variance = _safe_variance(values)
    standard_deviation = math.sqrt(max(variance, 0.0))
    activity, mobility, complexity = _hjorth(values)
    finite_rms = float(np.sqrt(np.mean(np.square(values))))
    nearly_constant = standard_deviation <= (
        NEARLY_CONSTANT_RELATIVE_TOLERANCE * max(1.0, finite_rms)
    )
    skewness = math.nan if nearly_constant else float(skew(values, bias=False))
    excess_kurtosis = (
        math.nan if nearly_constant else float(kurtosis(values, fisher=True, bias=False))
    )
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "variance": variance,
        "standard_deviation": standard_deviation,
        "rms": finite_rms,
        "line_length": float(np.sum(np.abs(np.diff(values)))),
        "zero_crossing_rate": float(
            np.count_nonzero(values[:-1] * values[1:] < 0.0) / (values.size - 1)
        ),
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "skewness": skewness,
        "kurtosis": excess_kurtosis,
    }


def feature_definitions() -> dict[str, str]:
    return {
        "mean": "arithmetic mean; V",
        "median": "sample median; V",
        "variance": "population mean squared deviation; V^2",
        "standard_deviation": "square root of population variance; V",
        "rms": "sqrt(mean(x^2)); V",
        "line_length": "sum(abs(diff(x))); V",
        "zero_crossing_rate": "strict sign changes divided by n-1; unitless",
        "hjorth_activity": "population variance(x); V^2",
        "hjorth_mobility": "sqrt(var(diff(x))/var(x)); unitless",
        "hjorth_complexity": "mobility(diff(x))/mobility(x); unitless",
        "skewness": "bias-corrected Fisher-Pearson skewness; unitless",
        "kurtosis": "bias-corrected Fisher excess kurtosis; unitless",
    }


def process_preprocessed_recording(
    rec_row: dict[str, str],
    fif_path: Path,
    metadata_path: Path,
    meta_json: dict[str, Any],
    channel_labels: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream one preprocessed recording and return epoch rows plus QC."""
    shared = _shared_module()
    part_id = shared._clean(rec_row.get("participant_id"))
    dataset = shared._clean(rec_row.get("dataset")).lower()
    rec_id = shared._clean(rec_row.get("recording_id")) or Path(
        rec_row.get("edf_path", "")
    ).stem
    qc: dict[str, Any] = {
        "participant_id": part_id,
        "dataset": dataset,
        "site": dataset,
        "recording_id": rec_id,
        "source_file": rec_row.get("edf_path", ""),
        "status": "failed",
        "exclusion_reason": "",
        "n_channels_input": 0,
        "n_channels_used": 0,
        "n_bad_channels": 0,
        "n_epochs_input": 0,
        "n_epochs_valid": 0,
        "n_epochs_rejected": 0,
        "n_channel_epochs_input": 0,
        "n_channel_epochs_valid": 0,
        "n_channel_epochs_rejected": 0,
        "sampling_frequency_hz": math.nan,
        "n_nan_features": 0,
        "n_infinite_features": 0,
        "n_epochs_rejected_flat_or_non_finite": 0,
        "n_flat_channel_epochs": 0,
        "n_non_finite_channel_epochs": 0,
        "n_flat_or_non_finite_channel_epochs": 0,
        "n_failed_windows": 0,
        "failed_window_errors": [],
    }
    if dataset not in shared.SUPPORTED_FEATURE_DATASETS:
        qc["exclusion_reason"] = f"unsupported_dataset_{dataset}"
        return [], qc

    try:
        epochs = mne.read_epochs(fif_path, preload=False, verbose="ERROR")
    except Exception as error:
        raise ValueError(f"cannot read derivative as MNE Epochs: {fif_path}: {error}") from error
    sfreq = float(epochs.info["sfreq"])
    qc["sampling_frequency_hz"] = sfreq
    if not math.isclose(sfreq, EXPECTED_SFREQ_HZ, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"sampling frequency must be exactly 1000 Hz, got {sfreq:g}")

    all_ch_names = list(epochs.ch_names)
    qc["n_channels_input"] = len(all_ch_names)
    bad_channels = set(epochs.info.get("bads", []))
    bad_channels.update(meta_json.get("bad_channels_removed", []))
    qc["n_bad_channels"] = len(bad_channels)
    valid_picks: list[int] = []
    valid_ch_names: list[str] = []
    valid_ch_types: list[str] = []
    for index, (name, channel_type) in enumerate(
        zip(all_ch_names, epochs.get_channel_types())
    ):
        upper = channel_type.upper()
        if name in bad_channels or upper in shared.EXCLUDED_CHANNEL_TYPES:
            continue
        if upper in shared.VALID_IEEG_TYPES:
            valid_picks.append(index)
            valid_ch_names.append(name)
            valid_ch_types.append(upper)
    qc["n_channels_used"] = len(valid_picks)
    if not valid_picks:
        qc["exclusion_reason"] = "no_valid_ieeg_channels"
        return [], qc

    samples_per_epoch = len(epochs.times)
    if samples_per_epoch != EXPECTED_SAMPLES_PER_EPOCH:
        raise ValueError(
            f"each epoch must contain exactly {EXPECTED_SAMPLES_PER_EPOCH} samples, "
            f"got {samples_per_epoch}"
        )
    if not math.isclose(samples_per_epoch / sfreq, EPOCH_DURATION_S, abs_tol=1e-12):
        raise ValueError(f"epoch duration must be exactly {EPOCH_DURATION_S:g} seconds")

    n_epochs = len(epochs)
    qc["n_epochs_input"] = n_epochs
    qc["n_channel_epochs_input"] = n_epochs * len(valid_picks)
    channel_metadata = {
        item.get("name"): item
        for item in meta_json.get("channel_metadata", [])
        if isinstance(item, dict)
    }
    rows: list[dict[str, Any]] = []
    valid_epochs = 0
    invalid_channel_epochs = 0
    rejection_counts = {
        reason: 0 for reason in shared.CHANNEL_EPOCH_REJECTION_REASONS
    }
    # Retained for schema stability: a feature failure now raises instead of
    # being counted, so these stay zero in every completed recording.
    failed_windows = 0
    errors: list[str] = []
    n_nan = 0
    n_inf = 0

    task, recording_state = shared.require_recording_state(rec_row)

    for epoch_index in range(n_epochs):
        data = np.asarray(epochs.get_data(item=epoch_index, picks=valid_picks))
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        expected_shape = (len(valid_picks), EXPECTED_SAMPLES_PER_EPOCH)
        if data.shape != expected_shape:
            raise RuntimeError(
                f"MNE returned epoch shape {data.shape}, expected {expected_shape}"
            )
        epoch_contributed = False
        for channel_index, (channel, channel_type) in enumerate(
            zip(valid_ch_names, valid_ch_types)
        ):
            signal = data[channel_index]
            # Shared policy: reject only this channel-epoch, never the epoch.
            rejection = shared.channel_epoch_rejection_reason(
                signal, shared.EXPECTED_SAMPLES_PER_EPOCH
            )
            if rejection:
                rejection_counts[rejection] += 1
                invalid_channel_epochs += 1
                continue
            try:
                values = compute_epoch_time_domain_features(signal, sfreq)
            except Exception as error:
                raise shared.feature_failure(
                    "TimeDomainFeatures", rec_row, rec_id, epoch_index, channel,
                    "<all time-domain features>",
                    f"{type(error).__name__}: {error}",
                ) from error
            shared.check_feature_values(
                "TimeDomainFeatures", rec_row, rec_id, epoch_index, channel, values
            )
            epoch_contributed = True
            audited = channel_labels.get((part_id, channel))
            channel_meta = channel_metadata.get(channel, {})
            outcome = rec_row.get("surgical_outcome") or rec_row.get("outcome")
            label, reason = shared.determine_supervised_label(
                part_id, channel, channel_meta, audited, outcome
            )
            rows.append(
                {
                    "participant_id": part_id,
                    "dataset": dataset,
                    "site": dataset,
                    "recording_id": rec_id,
                    "source_file": str(fif_path),
                    "preprocessing_metadata_file": str(metadata_path),
                    "channel": channel,
                    "channel_type": channel_type,
                    "epoch_index": epoch_index,
                    "epoch_start_s": epoch_index * EPOCH_DURATION_S,
                    "epoch_duration_s": EPOCH_DURATION_S,
                    "epoch_valid": True,
                    "channel_epoch_valid": True,
                    "final_sampling_frequency_hz": sfreq,
                    "notch_frequencies": json.dumps(
                        meta_json.get("applied_notch_frequencies", [])
                    ),
                    "task": task,
                    "recording_state": recording_state,
                    "soz": channel_meta.get(
                        "soz", audited.get("channel_soz", "") if audited else ""
                    ),
                    "resected": channel_meta.get(
                        "resection",
                        audited.get("channel_resection", "") if audited else "",
                    ),
                    "outcome": rec_row.get(
                        "surgical_outcome",
                        audited.get("participant_surgical_outcome", "") if audited else "",
                    ),
                    "supervised_label": label,
                    "label_reason": reason,
                    **values,
                }
            )

        if epoch_contributed:
            valid_epochs += 1

    rejected = n_epochs - valid_epochs
    qc.update(
        status="completed",
        channel_epoch_validity_policy=shared.CHANNEL_EPOCH_VALIDITY_POLICY,
        n_epochs_valid=valid_epochs,
        n_epochs_rejected=rejected,
        n_channel_epochs_valid=len(rows),
        n_channel_epochs_rejected=invalid_channel_epochs,
        n_nan_features=n_nan,
        n_infinite_features=n_inf,
        n_epochs_rejected_flat_or_non_finite=rejected,
        n_flat_channel_epochs=rejection_counts["flat"],
        n_near_flat_channel_epochs=rejection_counts["near_flat"],
        n_non_finite_channel_epochs=rejection_counts["non_finite_samples"],
        n_flat_or_non_finite_channel_epochs=invalid_channel_epochs,
        channel_epoch_rejections=dict(rejection_counts),
        n_failed_windows=failed_windows,
        failed_window_errors=errors,
    )
    return rows, qc


def aggregate_channel_features(epoch_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid epochs by participant, recording, and channel."""
    if epoch_df.empty:
        return pd.DataFrame()
    valid = epoch_df
    for column in ("epoch_valid", "channel_epoch_valid"):
        if column in valid.columns:
            valid = valid[valid[column].astype(bool)]
    if valid.empty:
        return pd.DataFrame()
    # task and recording_state are carried into the channel table so a
    # per-task breakdown never has to be reconstructed from recording ids.
    group_columns = [
        "participant_id",
        "dataset",
        "site",
        "recording_id",
        "task",
        "recording_state",
        "channel",
        "channel_type",
    ]
    if "supervised_label" in valid.columns:
        group_columns.append("supervised_label")
    grouped = valid.groupby(group_columns, dropna=False)
    aggregated = grouped.agg({name: ["mean", "std", "median"] for name in FEATURE_NAMES})
    aggregated.columns = [f"{name}_{stat}" for name, stat in aggregated.columns]
    aggregated = aggregated.reset_index()
    return aggregated.merge(
        grouped.size().reset_index(name="n_valid_epochs"),
        on=group_columns,
        how="left",
    )


# Re-export only the infrastructure needed by the parallel workflow.
def load_channel_labels(path: Path | None):
    return _shared_module().load_channel_labels(path)


def index_preprocessed_artifacts(root: Path):
    return _shared_module().index_preprocessed_artifacts(root)


def resolve_preprocessed_artifacts(rec_row, root: Path, artifact_index=None):
    return _shared_module().resolve_preprocessed_artifacts(rec_row, root, artifact_index)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    _shared_module()._atomic_parquet(frame, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    _shared_module()._atomic_csv(frame, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    _shared_module()._atomic_json(payload, path)


def git_commit(repository: Path) -> str:
    return _shared_module()._git_commit(repository)
