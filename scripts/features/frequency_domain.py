#!/usr/bin/env python3
"""Scientific core for per-recording OMNI-iEEG frequency-domain features."""

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
from scipy.signal import welch


REPOSITORY = Path(__file__).resolve().parents[2]
SHARED_SCRIPT = REPOSITORY / "scripts" / "features" / "relative_bandpower.py"
SHARED_MODULE_NAME = "omni_frequency_domain_shared"

PROTOCOL_VERSION = "1.0.0"
DERIVATIVE_NAME = "Preprocessed1kHzNoBandpass"
EXPECTED_PREPROCESSING_PROTOCOL_VERSION = "1.1.0"
EXPECTED_SFREQ_HZ = 1000.0
EXPECTED_SAMPLES_PER_EPOCH = 8000
EPOCH_DURATION_S = 8.0
WELCH_N_FFT = 2048
WELCH_N_PER_SEG = 2048
WELCH_N_OVERLAP = 1024
WELCH_WINDOW = "hann"
WELCH_AVERAGE = "mean"
WELCH_DETREND = "constant"
WELCH_SCALING = "density"
WELCH_FMIN_HZ = 0.5
WELCH_FMAX_HZ = 450.0
# The channel-epoch validity policy is defined once, in relative_bandpower.py,
# and re-exported here so every family reports the same rule.
INVALID_EPOCH_POLICY = (
    "reject_only_the_degenerate_channel_epoch: a channel-epoch is used when it is "
    "one-dimensional, has exactly the expected sample count, is entirely finite, is "
    "not flat, and is not near-flat (sample std > 1e-12 * max(1, rms)); other "
    "channels in the same epoch are unaffected"
)
FEATURE_NAMES = ["spectral_entropy", "peak_frequency"]


def _shared_module():
    """Load read-only labeling and artifact infrastructure from RelativeBandPower."""
    if SHARED_MODULE_NAME in sys.modules:
        return sys.modules[SHARED_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(SHARED_MODULE_NAME, SHARED_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared feature infrastructure: {SHARED_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[SHARED_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def compute_epoch_frequency_domain_features(
    signal: np.ndarray, sfreq: float = EXPECTED_SFREQ_HZ
) -> dict[str, float]:
    """Compute normalized PSD entropy and peak frequency for one channel epoch.

    Welch settings and the retained 0.5--450 Hz range are identical to the
    validated RelativeBandPower implementation.
    """
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

    frequencies, psd = welch(
        values,
        fs=float(sfreq),
        window=WELCH_WINDOW,
        nperseg=WELCH_N_PER_SEG,
        noverlap=WELCH_N_OVERLAP,
        nfft=WELCH_N_FFT,
        detrend=WELCH_DETREND,
        scaling=WELCH_SCALING,
        average=WELCH_AVERAGE,
        axis=-1,
    )
    retained = (frequencies >= WELCH_FMIN_HZ) & (frequencies <= WELCH_FMAX_HZ)
    frequencies = frequencies[retained]
    psd = psd[retained]
    if frequencies.size == 0 or not np.all(np.isfinite(psd)):
        raise ValueError("Welch PSD has no finite bins in the configured frequency range")
    psd_sum = float(np.sum(psd))
    if psd_sum > 0.0 and frequencies.size > 1:
        probabilities = psd / psd_sum
        positive = probabilities > 0.0
        entropy = float(
            -np.sum(probabilities[positive] * np.log(probabilities[positive]))
            / np.log(probabilities.size)
        )
    else:
        entropy = 0.0
    return {
        "spectral_entropy": entropy,
        "peak_frequency": float(frequencies[int(np.argmax(psd))]),
    }


def feature_definitions() -> dict[str, str]:
    return {
        "spectral_entropy": "normalized Shannon entropy of Welch PSD bins from 0.5 through 450 Hz; unitless",
        "peak_frequency": "frequency of the maximum Welch PSD bin from 0.5 through 450 Hz; Hz",
    }


def process_preprocessed_recording(
    rec_row: dict[str, str],
    fif_path: Path,
    metadata_path: Path,
    meta_json: dict[str, Any],
    channel_labels: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream one recording and return frequency-domain epoch rows plus QC."""
    shared = _shared_module()
    participant = shared._clean(rec_row.get("participant_id"))
    dataset = shared._clean(rec_row.get("dataset")).lower()
    recording = shared._clean(rec_row.get("recording_id")) or Path(
        rec_row.get("edf_path", "")
    ).stem
    qc: dict[str, Any] = {
        "participant_id": participant,
        "dataset": dataset,
        "site": dataset,
        "recording_id": recording,
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

    all_channels = list(epochs.ch_names)
    qc["n_channels_input"] = len(all_channels)
    bad_channels = set(epochs.info.get("bads", []))
    bad_channels.update(meta_json.get("bad_channels_removed", []))
    qc["n_bad_channels"] = len(bad_channels)
    picks: list[int] = []
    channels: list[str] = []
    channel_types: list[str] = []
    for index, (name, channel_type) in enumerate(zip(all_channels, epochs.get_channel_types())):
        channel_type = channel_type.upper()
        if name in bad_channels or channel_type in shared.EXCLUDED_CHANNEL_TYPES:
            continue
        if channel_type in shared.VALID_IEEG_TYPES:
            picks.append(index)
            channels.append(name)
            channel_types.append(channel_type)
    qc["n_channels_used"] = len(picks)
    if not picks:
        qc["exclusion_reason"] = "no_valid_ieeg_channels"
        return [], qc
    if len(epochs.times) != EXPECTED_SAMPLES_PER_EPOCH:
        raise ValueError(
            f"each epoch must contain exactly {EXPECTED_SAMPLES_PER_EPOCH} samples, got {len(epochs.times)}"
        )
    if not math.isclose(len(epochs.times) / sfreq, EPOCH_DURATION_S, abs_tol=1e-12):
        raise ValueError(f"epoch duration must be exactly {EPOCH_DURATION_S:g} seconds")

    n_epochs = len(epochs)
    qc["n_epochs_input"] = n_epochs
    qc["n_channel_epochs_input"] = n_epochs * len(picks)
    channel_metadata = {
        item.get("name"): item
        for item in meta_json.get("channel_metadata", [])
        if isinstance(item, dict)
    }
    rows: list[dict[str, Any]] = []
    valid_epochs = invalid_channel_epochs = 0
    rejection_counts = {
        reason: 0 for reason in shared.CHANNEL_EPOCH_REJECTION_REASONS
    }
    # Retained for schema stability: a feature failure now raises instead of
    # being counted, so these stay zero in every completed recording.
    failed_windows = 0
    failed_errors: list[str] = []
    n_nan = n_inf = 0
    task, recording_state = shared.require_recording_state(rec_row)

    for epoch_index in range(n_epochs):
        data = np.asarray(epochs.get_data(item=epoch_index, picks=picks))
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        expected_shape = (len(picks), EXPECTED_SAMPLES_PER_EPOCH)
        if data.shape != expected_shape:
            raise RuntimeError(f"MNE returned epoch shape {data.shape}, expected {expected_shape}")
        epoch_contributed = False
        for channel_index, (channel, channel_type) in enumerate(
            zip(channels, channel_types)
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
                features = compute_epoch_frequency_domain_features(signal, sfreq)
            except Exception as error:
                raise shared.feature_failure(
                    "FrequencyDomainFeatures", rec_row, recording, epoch_index, channel,
                    "<all frequency-domain features>",
                    f"{type(error).__name__}: {error}",
                ) from error
            shared.check_feature_values(
                "FrequencyDomainFeatures", rec_row, recording, epoch_index, channel, features
            )
            epoch_contributed = True
            audited = channel_labels.get((participant, channel))
            metadata = channel_metadata.get(channel, {})
            outcome = rec_row.get("surgical_outcome") or rec_row.get("outcome")
            label, label_reason = shared.determine_supervised_label(
                participant, channel, metadata, audited, outcome
            )
            rows.append(
                {
                    "participant_id": participant,
                    "dataset": dataset,
                    "site": dataset,
                    "recording_id": recording,
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
                    "notch_frequencies": json.dumps(meta_json.get("applied_notch_frequencies", [])),
                    "task": task,
                    "recording_state": recording_state,
                    "soz": metadata.get("soz", audited.get("channel_soz", "") if audited else ""),
                    "resected": metadata.get("resection", audited.get("channel_resection", "") if audited else ""),
                    "outcome": rec_row.get("surgical_outcome", audited.get("participant_surgical_outcome", "") if audited else ""),
                    "supervised_label": label,
                    "label_reason": label_reason,
                    **features,
                }
            )

        if epoch_contributed:
            valid_epochs += 1

    rejected_epochs = n_epochs - valid_epochs
    qc.update(
        status="completed",
        channel_epoch_validity_policy=shared.CHANNEL_EPOCH_VALIDITY_POLICY,
        n_epochs_valid=valid_epochs,
        n_epochs_rejected=rejected_epochs,
        n_channel_epochs_valid=len(rows),
        n_channel_epochs_rejected=invalid_channel_epochs,
        n_nan_features=n_nan,
        n_infinite_features=n_inf,
        n_epochs_rejected_flat_or_non_finite=rejected_epochs,
        n_flat_channel_epochs=rejection_counts["flat"],
        n_near_flat_channel_epochs=rejection_counts["near_flat"],
        n_non_finite_channel_epochs=rejection_counts["non_finite_samples"],
        n_flat_or_non_finite_channel_epochs=invalid_channel_epochs,
        channel_epoch_rejections=dict(rejection_counts),
        n_failed_windows=failed_windows,
        failed_window_errors=failed_errors,
    )
    return rows, qc


def aggregate_channel_features(epoch_df: pd.DataFrame) -> pd.DataFrame:
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
    output = grouped.agg({feature: ["mean", "std", "median"] for feature in FEATURE_NAMES})
    output.columns = [f"{feature}_{stat}" for feature, stat in output.columns]
    output = output.reset_index()
    return output.merge(grouped.size().reset_index(name="n_valid_epochs"), on=group_columns, how="left")


def load_channel_labels(path: Path | None):
    return _shared_module().load_channel_labels(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    _shared_module()._atomic_parquet(frame, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    _shared_module()._atomic_csv(frame, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    _shared_module()._atomic_json(payload, path)


def git_commit(repository: Path) -> str:
    return _shared_module()._git_commit(repository)
