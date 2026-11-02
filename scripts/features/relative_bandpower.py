#!/usr/bin/env python3
"""Scientific core for the NeuroDAGs HUP/UCLA relative-bandpower derivative."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import re
import os
import subprocess
import tempfile
import csv
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import welch

LOGGER = logging.getLogger("hup_ucla_relative_bandpower")
PROTOCOL_VERSION = "2.0.0"
DERIVATIVE_NAME = "Preprocessed1kHzNoBandpass"
EXPECTED_PREPROCESSING_PROTOCOL_VERSION = "1.1.0"
EXPECTED_PREPROCESSING_PROTOCOL_VERSIONS = {
    "hup": EXPECTED_PREPROCESSING_PROTOCOL_VERSION,
    "ucla": EXPECTED_PREPROCESSING_PROTOCOL_VERSION,
    "openieeg": "openieeg-1.0.0",
    "detroit": "openieeg-1.0.0",
}

EXPECTED_SFREQ_HZ = 1000.0
MIN_NYQUIST_HZ = 450.0
WELCH_N_FFT = 2048
WELCH_N_PER_SEG = 2048
WELCH_N_OVERLAP = 1024
WELCH_WINDOW = "hann"
WELCH_AVERAGE = "mean"
WELCH_DETREND = "constant"
WELCH_SCALING = "density"
WELCH_FMIN_HZ = 0.5
WELCH_FMAX_HZ = 450.0
SUM_TOLERANCE = 1e-3
EPOCH_DURATION_S = 8.0
EXPECTED_SAMPLES_PER_EPOCH = 8000
# One channel-epoch validity policy, shared by all four feature families so a
# degenerate signal never produces different epoch identity sets between them.
# See channel_epoch_rejection_reason() for the exact rule.
CHANNEL_EPOCH_VALIDITY_POLICY = "channel_epoch_validity_v1"
INVALID_EPOCH_POLICY = (
    "reject_only_the_degenerate_channel_epoch: a channel-epoch is used when it is "
    "one-dimensional, has exactly the expected sample count, is entirely finite, is "
    "not flat, and is not near-flat (sample std > 1e-12 * max(1, rms)); other "
    "channels in the same epoch are unaffected"
)
NEARLY_CONSTANT_RELATIVE_TOLERANCE = 1e-12
CHANNEL_EPOCH_REJECTION_REASONS = (
    "not_one_dimensional",
    "unexpected_sample_count",
    "non_finite_samples",
    "flat",
    "near_flat",
)

VALID_IEEG_TYPES = frozenset({"SEEG", "ECOG", "EEG", "IEEG"})
EXCLUDED_CHANNEL_TYPES = frozenset({"ECG", "EOG", "EMG", "STIM", "TRIG", "MISC", "REF", "OTHER"})

CANONICAL_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
    "high_gamma": (80.0, 150.0),
}

EXPLORATORY_BANDS: dict[str, tuple[float, float]] = {
    "hf_150_250": (150.0, 250.0),
    "hf_250_350": (250.0, 350.0),
    "hf_350_450": (350.0, 450.0),
}

ALL_BANDS: dict[str, tuple[float, float]] = {**CANONICAL_BANDS, **EXPLORATORY_BANDS}

PAPER_FEATURE_NAMES = [
    "rel_paper_delta_0p5_4",
    "rel_paper_theta_4_8",
    "rel_paper_alpha_8_13",
    "rel_paper_beta_13_30",
    "rel_paper_gamma_30_80",
    "rel_paper_high_gamma_80_150",
]

EXTENDED_FEATURE_NAMES = [
    "rel_ext_delta_0p5_4",
    "rel_ext_theta_4_8",
    "rel_ext_alpha_8_13",
    "rel_ext_beta_13_30",
    "rel_ext_gamma_30_80",
    "rel_ext_high_gamma_80_150",
    "rel_ext_hf_150_250",
    "rel_ext_hf_250_350",
    "rel_ext_hf_350_450",
]

HF_FEATURE_NAME = "rel_ext_hf_150_450"

TOTAL_POWER_NAMES = [
    "total_power_0p5_150",
    "total_power_0p5_450",
]

# Flat list of every scientific column this family writes, for cohort-level
# validation. Composition is unchanged; nothing new is defined here.
FEATURE_NAMES = [
    *TOTAL_POWER_NAMES,
    *PAPER_FEATURE_NAMES,
    *EXTENDED_FEATURE_NAMES,
    HF_FEATURE_NAME,
]

ArtifactIndex = list[tuple[Path, Path, dict[str, Any]]]
# Canonical recording states, matching the cohort protocol.
RECORDING_STATES = ("interictal", "ictal")
# Feature definitions are shared across cohorts.  Cohort-specific manifests
# decide which recordings enter a run; this allow-list only prevents an
# accidental unrelated derivative from being assembled as a feature cohort.
SUPPORTED_FEATURE_DATASETS = frozenset({"hup", "ucla", "openieeg", "detroit"})


def require_recording_state(rec_row: dict[str, str]) -> tuple[str, str]:
    """Return (task, recording_state) from the manifest row, or fail loudly.

    Both come from the frozen feature manifest. There is no default: a missing
    value silently relabelling epochs as interictal is exactly what the ML
    task guard cannot detect.
    """
    task = _clean(rec_row.get("task"))
    state = _clean(rec_row.get("recording_state")).lower()
    if not task:
        raise ValueError(
            f"recording {_clean(rec_row.get('recording_id'))!r} has no task; "
            "it must come from the frozen feature manifest"
        )
    if state not in RECORDING_STATES:
        raise ValueError(
            f"recording {_clean(rec_row.get('recording_id'))!r} has recording_state "
            f"{state!r}; expected one of {list(RECORDING_STATES)}"
        )
    return task, state


def channel_epoch_rejection_reason(
    signal: Any, expected_samples: int = EXPECTED_SAMPLES_PER_EPOCH
) -> str:
    """Return "" when a channel-epoch is usable, else its rejection reason.

    This is the single validity policy for every feature family. It inspects the
    signal only -- never a feature value -- so all four families accept and
    reject exactly the same channel-epochs and therefore produce identical epoch
    identity sets. Rejection is per channel-epoch: the other channels of the
    same epoch are unaffected.
    """
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        return "not_one_dimensional"
    if values.size != expected_samples:
        return "unexpected_sample_count"
    if not np.all(np.isfinite(values)):
        return "non_finite_samples"
    if np.ptp(values) == 0.0:
        return "flat"
    standard_deviation = float(np.sqrt(np.mean(np.square(values - np.mean(values)))))
    root_mean_square = float(np.sqrt(np.mean(np.square(values))))
    if standard_deviation <= NEARLY_CONSTANT_RELATIVE_TOLERANCE * max(1.0, root_mean_square):
        return "near_flat"
    return ""


def is_valid_channel_epoch(
    signal: Any, expected_samples: int = EXPECTED_SAMPLES_PER_EPOCH
) -> bool:
    """True when :func:`channel_epoch_rejection_reason` finds nothing wrong."""
    return channel_epoch_rejection_reason(signal, expected_samples) == ""


def feature_failure(
    family: str,
    rec_row: dict[str, str],
    recording_id: str,
    epoch_index: int,
    channel: str,
    feature: str,
    detail: str,
) -> RuntimeError:
    """Build the one error shape every family raises for a failed feature.

    Carries participant, recording, epoch, channel and feature so a failure is
    traceable to a single value without re-running the extraction.
    """
    return RuntimeError(
        f"{family} feature computation failed: "
        f"participant={_clean(rec_row.get('participant_id'))!r} "
        f"dataset={_clean(rec_row.get('dataset')).lower()!r} "
        f"recording={recording_id!r} epoch={epoch_index} channel={channel!r} "
        f"feature={feature!r}: {detail}"
    )


def check_feature_values(
    family: str,
    rec_row: dict[str, str],
    recording_id: str,
    epoch_index: int,
    channel: str,
    values: dict[str, float],
) -> None:
    """Raise on the first non-finite feature instead of writing NaN/Inf."""
    for feature, value in values.items():
        number = float(value)
        if math.isnan(number):
            raise feature_failure(
                family, rec_row, recording_id, epoch_index, channel, feature,
                "produced NaN on a channel-epoch that passed the shared validity policy",
            )
        if math.isinf(number):
            raise feature_failure(
                family, rec_row, recording_id, epoch_index, channel, feature,
                f"produced {'+' if number > 0 else '-'}Inf on a channel-epoch that "
                "passed the shared validity policy",
            )


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    val = float(trapezoid(y, x))
    return val if math.isfinite(val) else math.nan


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".csv", prefix=f".{path.name}.", dir=path.parent,
        encoding="utf-8", newline="", delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix=f".{path.name}.", dir=path.parent,
        encoding="utf-8", delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _strict_binary(value: object) -> int | None:
    text = _clean(value).lower()
    if text in {"1", "1.0", "true", "yes"}:
        return 1
    if text in {"0", "0.0", "false", "no"}:
        return 0
    return None


# ---------------------------------------------------------------------------
# Canonical surgical-outcome parser
# ---------------------------------------------------------------------------
# One definition of "seizure-free", shared by feature labelling, the cohort and
# channel audit, and the ML seizure-free filter. Three divergent parsers used to
# exist; a value one accepted and another did not silently changed which
# participants survived the ML filter.

SURGICAL_OUTCOME_UNKNOWN_TOKENS = frozenset({"", "-1"})

_SEIZURE_FREE_TOKENS = frozenset({
    "1", "1.0", "true", "yes", "y",
    "seizure free", "seizurefree", "free of seizures", "free",
})
_NOT_SEIZURE_FREE_TOKENS = frozenset({
    "0", "0.0", "false", "no", "n",
    "not seizure free", "notseizurefree", "not free of seizures", "not free",
})
_ILAE_PATTERN = re.compile(r"^ilae\s*([1-6])\s*([a-d])?$")
_ENGEL_PATTERN = re.compile(r"^engel\s*(iv|iii|ii|i|[1-4])\s*([a-d])?$")
_ENGEL_CLASS = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "4": 4}


def _normalise_outcome(value: object) -> str:
    """Lower-case and turn word separators into spaces.

    A hyphen is only a separator between letters ("seizure-free"); a leading
    "-" belongs to the number, so the unknown-outcome sentinel "-1" remains
    distinguishable from the seizure-free value "1".
    """
    text = _clean(value).lower().replace("_", " ")
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)
    return " ".join(text.split())


def parse_surgical_outcome(value: object) -> int | None:
    """Return 1 (seizure-free), 0 (not seizure-free), or None when absent.

    ``None`` means the outcome is genuinely absent (empty/missing), which is a
    legitimate state for a participant who had no surgery. Any *present but
    unrecognised* encoding raises, so a value can never be silently reduced to
    "unknown" and quietly remove a participant from the cohort.

    Recognised encodings: 1/0, true/false, yes/no, "seizure-free" and
    "not seizure-free" (any separator), the ``-1`` unknown/unavailable
    sentinel, the ILAE scale (1[a-d] seizure-free,
    2-6 not), and the Engel scale (class I seizure-free, II-IV not; roman or
    arabic, with or without a sub-class letter).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        if float(value) == -1.0:
            return None
        if float(value) in (0.0, 1.0):
            return int(value)
        raise ValueError(f"unrecognised surgical outcome encoding: {value!r}")
    token = _normalise_outcome(value)
    if token in SURGICAL_OUTCOME_UNKNOWN_TOKENS:
        return None
    if token in _SEIZURE_FREE_TOKENS:
        return 1
    if token in _NOT_SEIZURE_FREE_TOKENS:
        return 0
    ilae = _ILAE_PATTERN.match(token)
    if ilae:
        return 1 if ilae.group(1) == "1" else 0
    engel = _ENGEL_PATTERN.match(token)
    if engel:
        return 1 if _ENGEL_CLASS[engel.group(1)] == 1 else 0
    raise ValueError(
        f"unrecognised surgical outcome encoding: {value!r}. Recognised: 1/0, "
        "true/false, yes/no, seizure-free / not seizure-free, ILAE 1-6, Engel I-IV."
    )


def is_seizure_free(value: object) -> bool | None:
    """True/False from :func:`parse_surgical_outcome`; None when absent."""
    parsed = parse_surgical_outcome(value)
    return None if parsed is None else bool(parsed)


def _is_seizure_free(value: object) -> bool | None:
    return is_seizure_free(value)


def compute_epoch_relative_bandpower(
    signal: np.ndarray,
    sfreq: float = EXPECTED_SFREQ_HZ,
) -> dict[str, float]:
    """Compute Welch PSD and relative power features for one 8-second channel epoch."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("signal must be 1D")

    # Compute Welch PSD
    freqs, psd = welch(
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

    # Frequency masks
    mask_0p5_150 = (freqs >= 0.5) & (freqs < 150.0)
    mask_0p5_450 = (freqs >= 0.5) & (freqs <= 450.0)

    total_power_150 = _trapezoid(psd[mask_0p5_150], freqs[mask_0p5_150])
    total_power_450 = _trapezoid(psd[mask_0p5_450], freqs[mask_0p5_450])

    features: dict[str, float] = {
        "total_power_0p5_150": total_power_150,
        "total_power_0p5_450": total_power_450,
    }

    if not math.isfinite(total_power_150) or total_power_150 <= 0.0:
        for name in PAPER_FEATURE_NAMES:
            features[name] = math.nan
    else:
        for name, (low, high) in CANONICAL_BANDS.items():
            mask = (freqs >= low) & (freqs < high)
            band_pow = _trapezoid(psd[mask], freqs[mask])
            # Map canonical band name to standardized column name
            paper_col_map = {
                "delta": "rel_paper_delta_0p5_4",
                "theta": "rel_paper_theta_4_8",
                "alpha": "rel_paper_alpha_8_13",
                "beta": "rel_paper_beta_13_30",
                "gamma": "rel_paper_gamma_30_80",
                "high_gamma": "rel_paper_high_gamma_80_150",
            }
            features[paper_col_map[name]] = band_pow / total_power_150

    if not math.isfinite(total_power_450) or total_power_450 <= 0.0:
        for name in EXTENDED_FEATURE_NAMES:
            features[name] = math.nan
    else:
        for name, (low, high) in ALL_BANDS.items():
            if high < 450.0:
                mask = (freqs >= low) & (freqs < high)
            else:
                mask = (freqs >= low) & (freqs <= high)
            band_pow = _trapezoid(psd[mask], freqs[mask])
            ext_col_map = {
                "delta": "rel_ext_delta_0p5_4",
                "theta": "rel_ext_theta_4_8",
                "alpha": "rel_ext_alpha_8_13",
                "beta": "rel_ext_beta_13_30",
                "gamma": "rel_ext_gamma_30_80",
                "high_gamma": "rel_ext_high_gamma_80_150",
                "hf_150_250": "rel_ext_hf_150_250",
                "hf_250_350": "rel_ext_hf_250_350",
                "hf_350_450": "rel_ext_hf_350_450",
            }
            features[ext_col_map[name]] = band_pow / total_power_450

    features[HF_FEATURE_NAME] = sum(
        features[name]
        for name in (
            "rel_ext_hf_150_250",
            "rel_ext_hf_250_350",
            "rel_ext_hf_350_450",
        )
    )

    return features


def validate_synthetic() -> None:
    """Validate relative bandpower computation against synthetic sinusoids."""
    def require(condition: bool, message: str) -> None:
        # Do not use ``assert`` here: Python's -O flag removes assertions, while
        # this guard must remain mandatory for production extraction.
        if not condition:
            raise RuntimeError(f"synthetic validation failed: {message}")

    sfreq = 1000.0
    t = np.arange(8000) / sfreq

    # 1. 10 Hz sinusoid (Alpha band 8-13 Hz)
    sig_alpha = np.sin(2 * np.pi * 10.0 * t)
    feat_alpha = compute_epoch_relative_bandpower(sig_alpha, sfreq)
    require(
        feat_alpha["rel_paper_alpha_8_13"] > 0.85,
        f"expected dominant paper alpha, got {feat_alpha['rel_paper_alpha_8_13']}",
    )
    require(
        feat_alpha["rel_ext_alpha_8_13"] > 0.85,
        f"expected dominant extended alpha, got {feat_alpha['rel_ext_alpha_8_13']}",
    )

    # 2. 200 Hz sinusoid (hf_150_250 band)
    sig_200 = np.sin(2 * np.pi * 200.0 * t)
    feat_200 = compute_epoch_relative_bandpower(sig_200, sfreq)
    require(
        feat_200["rel_ext_hf_150_250"] > 0.85,
        f"expected dominant 150-250 Hz, got {feat_200['rel_ext_hf_150_250']}",
    )
    require(feat_200[HF_FEATURE_NAME] > 0.85, "200 Hz did not dominate HF150-450")

    # 3. 300 Hz sinusoid (hf_250_350 band)
    sig_300 = np.sin(2 * np.pi * 300.0 * t)
    feat_300 = compute_epoch_relative_bandpower(sig_300, sfreq)
    require(
        feat_300["rel_ext_hf_250_350"] > 0.85,
        f"expected dominant 250-350 Hz, got {feat_300['rel_ext_hf_250_350']}",
    )
    require(feat_300[HF_FEATURE_NAME] > 0.85, "300 Hz did not dominate HF150-450")

    # 4. Check relative power sums
    paper_sum = sum(feat_alpha[col] for col in PAPER_FEATURE_NAMES)
    require(
        abs(paper_sum - 1.0) < SUM_TOLERANCE,
        f"paper relative sum error: {paper_sum}",
    )
    for frequency, features in ((10, feat_alpha), (200, feat_200), (300, feat_300)):
        ext_sum = sum(features[col] for col in EXTENDED_FEATURE_NAMES)
        require(
            abs(ext_sum - 1.0) < SUM_TOLERANCE,
            f"extended relative sum error at {frequency} Hz: {ext_sum}",
        )

    LOGGER.info("Synthetic validation checks passed successfully!")


def load_channel_labels(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    """Load optional audited channel label table."""
    if path is None or not path.is_file():
        return {}
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            part_id = _clean(row.get("participant_id"))
            ch_name = _clean(row.get("channel_name") or row.get("channel"))
            if part_id and ch_name:
                labels[(part_id, ch_name)] = row
    return labels


def _get_first_key(dicts: list[dict[str, Any] | None], keys: list[str]) -> Any:
    for d in dicts:
        if not d:
            continue
        for k in keys:
            if k in d and d[k] not in (None, ""):
                return d[k]
    return None


def determine_supervised_label(
    part_id: str,
    ch_name: str,
    ch_metadata: dict[str, Any],
    audited_row: dict[str, Any] | None,
    rec_outcome: object = None,
) -> tuple[float, str]:
    """Determine supervised_label (1.0, 0.0, or NaN) and label_reason."""
    sources = [audited_row, ch_metadata]

    soz_raw = _get_first_key(sources, ["high_confidence_positive", "channel_soz", "soz"])
    resect_raw = _get_first_key(sources, ["channel_resection", "resection", "resected"])
    outcome_raw = _get_first_key(sources, ["participant_surgical_outcome", "surgical_outcome", "outcome"]) or rec_outcome

    soz_val = _strict_binary(soz_raw)
    resect_val = _strict_binary(resect_raw)
    outcome_val = _is_seizure_free(outcome_raw)

    if soz_val == 1:
        return 1.0, "high_confidence_positive_soz"
    if soz_val == 0 and resect_val == 0 and outcome_val is True:
        return 0.0, "high_confidence_negative_preserved_seizure_free"
    if soz_val == 0 and resect_val == 1:
        return math.nan, "ambiguous_resected_non_soz"
    if soz_val == 0 and outcome_val is False:
        return math.nan, "ambiguous_non_seizure_free"
    return math.nan, "unlabeled_or_ambiguous"


def _recording_identifier_from_path(value: object) -> str:
    name = Path(_clean(value)).name
    return name[:-4] if name.lower().endswith(".edf") else Path(name).stem


def _expected_preprocessing_protocol_version(dataset: object) -> str:
    """Expected preprocessing protocol for one feature-cohort dataset."""
    return EXPECTED_PREPROCESSING_PROTOCOL_VERSIONS.get(
        _clean(dataset).lower(), EXPECTED_PREPROCESSING_PROTOCOL_VERSION
    )


def _normalize_recording_identity(value: object) -> str:
    """Normalize missing values and numeric BIDS session/run entities for matching."""
    text = _clean(value)
    return re.sub(
        r"(?<=_)(ses|run)-0*(\d+)(?=_|$)",
        lambda match: f"{match.group(1)}-{int(match.group(2))}",
        text,
        flags=re.IGNORECASE,
    )


def _metadata_disagreements(
    metadata: dict[str, Any], rec_row: dict[str, str]
) -> list[str]:
    disagreements: list[str] = []
    expected_dataset = _clean(rec_row.get("dataset")).lower()
    expected_participant = _clean(rec_row.get("participant_id"))
    expected_recording = _normalize_recording_identity(
        rec_row.get("recording_id")
    ) or _normalize_recording_identity(_recording_identifier_from_path(rec_row.get("edf_path")))
    observed_dataset = _clean(metadata.get("dataset")) or _clean(metadata.get("site"))
    observed_dataset = observed_dataset.lower()
    observed_participant = _clean(metadata.get("participant_id"))
    if observed_dataset != expected_dataset:
        disagreements.append(f"dataset {observed_dataset!r} != {expected_dataset!r}")
    if observed_participant != expected_participant:
        disagreements.append(
            f"participant_id {observed_participant!r} != {expected_participant!r}"
        )
    metadata_identifiers = {
        _normalize_recording_identity(metadata.get("recording_id")),
        _normalize_recording_identity(_recording_identifier_from_path(metadata.get("edf_path"))),
        _normalize_recording_identity(_recording_identifier_from_path(metadata.get("source_edf_path"))),
    }
    metadata_identifiers.discard("")
    if expected_recording not in metadata_identifiers:
        disagreements.append(
            f"recording {expected_recording!r} not in metadata identifiers "
            f"{sorted(metadata_identifiers)!r}"
        )
    expected_protocol = _expected_preprocessing_protocol_version(expected_dataset)
    if metadata.get("protocol_version") != expected_protocol:
        disagreements.append(
            f"preprocessing protocol {metadata.get('protocol_version')!r} != "
            f"{expected_protocol!r}"
        )
    return disagreements


def _recording_identity_keys(metadata: dict[str, Any]) -> set[tuple[str, str, str]]:
    """Recording identity keys used to de-duplicate across preprocessing roots.

    Deduplication is by recording identity rather than by metadata path: the
    same recording reached through two different analysis roots is two distinct
    files, and indexing both makes ``resolve_preprocessed_artifacts`` fail with
    "found 2". A recording is addressed by several interchangeable identifiers,
    so every one of them is claimed.
    """
    dataset = _clean(metadata.get("dataset") or metadata.get("site")).lower()
    participant = _clean(metadata.get("participant_id"))
    identifiers = {
        _clean(metadata.get("recording_id")),
        _recording_identifier_from_path(metadata.get("edf_path")),
        _recording_identifier_from_path(metadata.get("source_edf_path")),
    }
    identifiers.discard("")
    if not dataset or not participant or not identifiers:
        return set()
    return {(dataset, participant, identifier) for identifier in identifiers}


def index_preprocessed_artifacts(
    preprocessed_root: Path | Sequence[Path],
) -> ArtifactIndex:
    """Read every candidate metadata JSON once for an efficient full-cohort run.

    Several roots may be supplied so a new analysis root can reuse derivatives
    that a previous root already computed instead of recomputing them.
    """
    roots = (
        [preprocessed_root]
        if isinstance(preprocessed_root, (str, Path))
        else list(preprocessed_root)
    )
    metadata_pattern = f"*@{DERIVATIVE_NAME}.metadata.json"
    indexed: ArtifactIndex = []
    seen_paths: set[Path] = set()
    claimed: set[tuple[str, str, str]] = set()
    for root in roots:
        # Roots are scanned in the order given and sorted within each root, so
        # the result is deterministic and an earlier root (the current analysis
        # root) wins over a later one (a read-only reuse root) for any recording
        # present in both. Claims are only published after a root is finished,
        # so two copies of one recording *inside a single root* stay ambiguous
        # and still fail loudly in resolve_preprocessed_artifacts.
        root_entries: ArtifactIndex = []
        root_claims: set[tuple[str, str, str]] = set()
        for metadata_path in sorted(Path(root).rglob(metadata_pattern)):
            resolved = metadata_path.resolve()
            if resolved in seen_paths:
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"cannot read preprocessing metadata {metadata_path}: {error}"
                ) from error
            if not isinstance(metadata, dict):
                raise ValueError(f"preprocessing metadata is not an object: {metadata_path}")
            identity = _recording_identity_keys(metadata)
            # Only a derivative this workflow would actually accept may claim a
            # recording; otherwise a stale copy in an earlier root would shadow
            # the valid one a later root still holds.
            metadata_dataset = _clean(metadata.get("dataset")) or _clean(metadata.get("site"))
            usable = metadata.get("protocol_version") == _expected_preprocessing_protocol_version(
                metadata_dataset
            )
            if usable and identity & claimed:
                continue
            seen_paths.add(resolved)
            if usable:
                root_claims |= identity
            fif_path = metadata_path.with_name(
                metadata_path.name[: -len(".metadata.json")] + ".fif"
            )
            root_entries.append((fif_path, metadata_path, metadata))
        indexed.extend(root_entries)
        claimed |= root_claims
    return indexed


def resolve_preprocessed_artifacts(
    rec_row: dict[str, str],
    preprocessed_root: Path,
    artifact_index: ArtifactIndex | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve exactly one participant/dataset/recording-compatible epoch FIF."""
    indexed = (
        artifact_index
        if artifact_index is not None
        else index_preprocessed_artifacts(preprocessed_root)
    )
    matches: ArtifactIndex = []
    for fif_path, metadata_path, metadata in indexed:
        if _metadata_disagreements(metadata, rec_row):
            continue
        if not fif_path.is_file():
            raise FileNotFoundError(
                f"metadata-compatible derivative has no FIF: {fif_path}"
            )
        matches.append((fif_path, metadata_path, metadata))
    dataset = _clean(rec_row.get("dataset")).lower()
    participant = _clean(rec_row.get("participant_id"))
    recording = _clean(rec_row.get("recording_id")) or _recording_identifier_from_path(
        rec_row.get("edf_path")
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one compatible {DERIVATIVE_NAME} derivative for "
            f"{dataset}/{participant}/{recording}; found {len(matches)}"
        )
    return matches[0]


def process_preprocessed_recording(
    rec_row: dict[str, str],
    fif_path: Path,
    metadata_path: Path,
    meta_json: dict[str, Any],
    channel_labels: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract features from one explicitly supplied preprocessing derivative.

    The NeuroDAGs node supplies the exact FIF and metadata sidecar, so this
    function never scans a derivative directory itself.
    """
    part_id = _clean(rec_row.get("participant_id"))
    dataset = _clean(rec_row.get("dataset")).lower()
    rec_id = _clean(rec_row.get("recording_id")) or Path(rec_row.get("edf_path", "")).stem

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
        "nyquist_hz": math.nan,
        "paper_relative_sum_mean": math.nan,
        "paper_relative_sum_max_abs_error": math.nan,
        "extended_relative_sum_mean": math.nan,
        "extended_relative_sum_max_abs_error": math.nan,
        "n_nan_features": 0,
        "n_infinite_features": 0,
        "n_zero_total_power_epochs": 0,
        "n_epochs_rejected_flat_or_non_finite": 0,
        "n_flat_channel_epochs": 0,
        "n_near_flat_channel_epochs": 0,
        "n_non_finite_channel_epochs": 0,
        "n_flat_or_non_finite_channel_epochs": 0,
        "channel_epoch_validity_policy": CHANNEL_EPOCH_VALIDITY_POLICY,
        "channel_epoch_rejections": {},
    }

    if dataset not in SUPPORTED_FEATURE_DATASETS:
        qc["exclusion_reason"] = f"unsupported_dataset_{dataset}"
        return [], qc

    # This derivative is contractually an MNE Epochs FIF, never continuous Raw.
    try:
        # Keep only one epoch in memory at a time.  This preserves the v1
        # Welch calculation while making one-recording Slurm array tasks much
        # less memory-hungry than loading the entire recording at once.
        epochs = mne.read_epochs(fif_path, preload=False, verbose="ERROR")
    except Exception as error:
        raise ValueError(f"cannot read derivative as MNE Epochs: {fif_path}: {error}") from error

    sfreq = float(epochs.info["sfreq"])
    nyquist = sfreq / 2.0
    qc["sampling_frequency_hz"] = sfreq
    qc["nyquist_hz"] = nyquist

    if not math.isclose(sfreq, EXPECTED_SFREQ_HZ, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"sampling frequency must be exactly 1000 Hz, got {sfreq:g}")

    if nyquist < MIN_NYQUIST_HZ:
        raise ValueError(f"Nyquist frequency is below 450 Hz: {nyquist:g}")

    # Channel selection & bad channel filtering
    all_ch_names = list(epochs.ch_names)
    qc["n_channels_input"] = len(all_ch_names)

    bad_channels = set(epochs.info.get("bads", []))
    if "bad_channels_removed" in meta_json:
        bad_channels.update(meta_json["bad_channels_removed"])

    qc["n_bad_channels"] = len(bad_channels)

    ch_types = epochs.get_channel_types()
    valid_picks = []
    valid_ch_names = []
    valid_ch_types = []

    for idx, (name, ctype) in enumerate(zip(all_ch_names, ch_types)):
        if name in bad_channels:
            continue
        ctype_upper = ctype.upper()
        if ctype_upper in EXCLUDED_CHANNEL_TYPES:
            continue
        if ctype_upper in VALID_IEEG_TYPES or ctype_upper == "EEG":
            valid_picks.append(idx)
            valid_ch_names.append(name)
            valid_ch_types.append(ctype_upper)

    qc["n_channels_used"] = len(valid_picks)
    if not valid_picks:
        qc["exclusion_reason"] = "no_valid_ieeg_channels"
        return [], qc

    n_epochs = len(epochs)
    samples_per_epoch = len(epochs.times)
    if samples_per_epoch != EXPECTED_SAMPLES_PER_EPOCH:
        raise ValueError(
            f"each epoch must contain exactly {EXPECTED_SAMPLES_PER_EPOCH} samples, "
            f"got {samples_per_epoch}"
        )
    if len(epochs.times) != EXPECTED_SAMPLES_PER_EPOCH:
        raise ValueError(
            f"epoch time axis must contain exactly {EXPECTED_SAMPLES_PER_EPOCH} values, "
            f"got {len(epochs.times)}"
        )
    observed_duration = samples_per_epoch / sfreq
    if not math.isclose(observed_duration, EPOCH_DURATION_S, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"epoch duration must be exactly {EPOCH_DURATION_S:g} seconds, "
            f"got {observed_duration:g}"
        )

    qc["n_epochs_input"] = n_epochs
    qc["n_channel_epochs_input"] = n_epochs * len(valid_picks)

    ch_metadata_lookup = {
        item.get("name"): item for item in meta_json.get("channel_metadata", []) if isinstance(item, dict)
    }

    rows: list[dict[str, Any]] = []
    paper_sums: list[float] = []
    ext_sums: list[float] = []
    n_nan = 0
    n_inf = 0
    rejection_counts = {reason: 0 for reason in CHANNEL_EPOCH_REJECTION_REASONS}
    n_invalid_channel_epochs = 0
    n_valid_ep = 0

    task, recording_state = require_recording_state(rec_row)

    for ep_idx in range(n_epochs):
        ep_start_s = ep_idx * EPOCH_DURATION_S
        epoch_data = np.asarray(epochs.get_data(item=ep_idx, picks=valid_picks))
        if epoch_data.ndim == 3 and epoch_data.shape[0] == 1:
            epoch_data = epoch_data[0]
        if epoch_data.ndim != 2 or epoch_data.shape != (len(valid_picks), samples_per_epoch):
            raise RuntimeError(
                "MNE did not return one channel-by-sample epoch for relative-bandpower extraction"
            )
        epoch_contributed = False
        for ch_idx, (ch_name, ch_type) in enumerate(zip(valid_ch_names, valid_ch_types)):
            signal = epoch_data[ch_idx]
            # Shared policy: reject only this channel-epoch, never the epoch.
            rejection = channel_epoch_rejection_reason(signal, samples_per_epoch)
            if rejection:
                rejection_counts[rejection] += 1
                n_invalid_channel_epochs += 1
                continue

            try:
                feat = compute_epoch_relative_bandpower(signal, sfreq)
            except Exception as error:
                raise feature_failure(
                    "RelativeBandPower", rec_row, rec_id, ep_idx, ch_name,
                    "<all relative-bandpower features>",
                    f"{type(error).__name__}: {error}",
                ) from error
            check_feature_values(
                "RelativeBandPower", rec_row, rec_id, ep_idx, ch_name, feat
            )
            epoch_contributed = True

            p_sum = sum(feat[col] for col in PAPER_FEATURE_NAMES if math.isfinite(feat[col]))
            e_sum = sum(feat[col] for col in EXTENDED_FEATURE_NAMES if math.isfinite(feat[col]))

            if math.isfinite(p_sum):
                paper_sums.append(p_sum)
            if math.isfinite(e_sum):
                ext_sums.append(e_sum)

            for val in feat.values():
                if math.isnan(val):
                    n_nan += 1
                elif math.isinf(val):
                    n_inf += 1

            audited_row = channel_labels.get((part_id, ch_name))
            ch_meta = ch_metadata_lookup.get(ch_name, {})
            rec_outcome = rec_row.get("surgical_outcome") or rec_row.get("outcome")
            sup_label, label_reason = determine_supervised_label(part_id, ch_name, ch_meta, audited_row, rec_outcome)

            row = {
                "participant_id": part_id,
                "dataset": dataset,
                "site": dataset,
                "recording_id": rec_id,
                "source_file": str(fif_path),
                "preprocessing_metadata_file": str(metadata_path),
                "channel": ch_name,
                "channel_type": ch_type,
                "epoch_index": ep_idx,
                "epoch_start_s": ep_start_s,
                "epoch_duration_s": EPOCH_DURATION_S,
                "epoch_valid": True,
                "channel_epoch_valid": True,
                "final_sampling_frequency_hz": sfreq,
                "notch_frequencies": json.dumps(meta_json.get("applied_notch_frequencies", [])),
                "task": task,
                "recording_state": recording_state,
                "soz": ch_meta.get("soz", audited_row.get("channel_soz", "") if audited_row else ""),
                "resected": ch_meta.get("resection", audited_row.get("channel_resection", "") if audited_row else ""),
                "outcome": rec_row.get("surgical_outcome", audited_row.get("participant_surgical_outcome", "") if audited_row else ""),
                "supervised_label": sup_label,
                "label_reason": label_reason,
                **feat,
            }
            rows.append(row)

        if epoch_contributed:
            n_valid_ep += 1

    qc["status"] = "completed"
    qc["channel_epoch_validity_policy"] = CHANNEL_EPOCH_VALIDITY_POLICY
    qc["n_epochs_valid"] = n_valid_ep
    qc["n_epochs_rejected"] = n_epochs - n_valid_ep
    qc["n_channel_epochs_valid"] = len(rows)
    qc["n_channel_epochs_rejected"] = n_invalid_channel_epochs
    qc["n_nan_features"] = n_nan
    qc["n_infinite_features"] = n_inf
    qc["n_epochs_rejected_flat_or_non_finite"] = n_epochs - n_valid_ep
    qc["n_flat_channel_epochs"] = rejection_counts["flat"]
    qc["n_near_flat_channel_epochs"] = rejection_counts["near_flat"]
    qc["n_non_finite_channel_epochs"] = rejection_counts["non_finite_samples"]
    qc["n_flat_or_non_finite_channel_epochs"] = n_invalid_channel_epochs
    qc["channel_epoch_rejections"] = dict(rejection_counts)

    if paper_sums:
        qc["paper_relative_sum_mean"] = float(np.mean(paper_sums))
        qc["paper_relative_sum_max_abs_error"] = float(np.max(np.abs(np.array(paper_sums) - 1.0)))

    if ext_sums:
        qc["extended_relative_sum_mean"] = float(np.mean(ext_sums))
        qc["extended_relative_sum_max_abs_error"] = float(np.max(np.abs(np.array(ext_sums) - 1.0)))

    return rows, qc


def aggregate_channel_features(epoch_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate epoch-level features per participant x recording x channel."""
    if epoch_df.empty:
        return pd.DataFrame()

    valid_epoch_df = epoch_df
    for validity_column in ("epoch_valid", "channel_epoch_valid"):
        if validity_column in valid_epoch_df.columns:
            valid_epoch_df = valid_epoch_df[valid_epoch_df[validity_column].astype(bool)]
    if valid_epoch_df.empty:
        return pd.DataFrame()

    # task and recording_state are carried into the channel table so a
    # per-task breakdown never has to be reconstructed from recording ids.
    group_cols = [
        "participant_id",
        "dataset",
        "site",
        "recording_id",
        "task",
        "recording_state",
        "channel",
        "channel_type",
    ]
    if "supervised_label" in valid_epoch_df.columns:
        group_cols.append("supervised_label")

    feature_cols = (
        PAPER_FEATURE_NAMES
        + EXTENDED_FEATURE_NAMES
        + [HF_FEATURE_NAME]
        + TOTAL_POWER_NAMES
    )

    agg_dict: dict[str, list[str]] = {col: ["mean", "std", "median"] for col in feature_cols}

    grouped = valid_epoch_df.groupby(group_cols, dropna=False)
    aggregated = grouped.agg(agg_dict)

    # Flatten column MultiIndex
    aggregated.columns = [f"{col}_{stat}" for col, stat in aggregated.columns]
    aggregated = aggregated.reset_index()

    # Add valid epoch count per channel
    counts = grouped.size().reset_index(name="n_valid_epochs")
    aggregated = pd.merge(aggregated, counts, on=group_cols, how="left")

    return aggregated
