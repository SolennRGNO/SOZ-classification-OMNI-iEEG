#!/usr/bin/env python3
"""Assemble per-recording FrequencyDomainFeatures artifacts into cohort tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[2]
CORE_SCRIPT = REPOSITORY / "scripts" / "features" / "frequency_domain.py"
MODULE_NAME = "omni_frequency_domain_assembly_reference"
OUTPUT_FILENAMES = {
    "epochs": "frequency_domain_epochs_hup_ucla.parquet",
    "channels": "frequency_domain_channels_hup_ucla.parquet",
    "qc": "frequency_domain_extraction_qc.csv",
    "summary": "frequency_domain_run_summary.json",
}


def output_filenames(cohort_name: str) -> dict[str, str]:
    """Return the unchanged artifact schema with a cohort-specific filename."""
    if not cohort_name or not cohort_name.replace("_", "").isalnum():
        raise ValueError(f"invalid cohort name: {cohort_name!r}")
    return {key: name.replace("hup_ucla", cohort_name) for key, name in OUTPUT_FILENAMES.items()}


def _core_module():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, CORE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frequency-domain core: {CORE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_invalid_features(qc: pd.DataFrame, epochs: pd.DataFrame) -> None:
    """Fail assembly when completed artifacts still contain NaN/Inf features.

    Both the recording QC counters and the assembled feature columns are
    checked, because a counter can only report what its own extraction build
    knew about.
    """
    for column in ("n_nan_features", "n_infinite_features", "n_failed_windows"):
        if column not in qc:
            continue
        total = int(pd.to_numeric(qc[column], errors="coerce").fillna(0).sum())
        if total:
            offenders = qc.loc[
                pd.to_numeric(qc[column], errors="coerce").fillna(0) > 0, "recording_id"
            ].tolist()
            raise RuntimeError(
                f"completed extraction reports {column}={total} in {len(offenders)} "
                f"recording(s): {offenders[:10]}"
            )
    core = _core_module()
    features = [name for name in core.FEATURE_NAMES if name in epochs.columns]
    missing = sorted(set(core.FEATURE_NAMES) - set(epochs.columns))
    if missing:
        raise RuntimeError(f"assembled epoch table lacks feature columns: {missing}")
    values = epochs[features].to_numpy(dtype=float)
    invalid = ~np.isfinite(values)
    if invalid.any():
        counts = invalid.sum(axis=0)
        worst = sorted(
            ((features[i], int(counts[i])) for i in range(len(features)) if counts[i]),
            key=lambda item: -item[1],
        )
        row_index = int(np.argmax(invalid.any(axis=1)))
        example = epochs.iloc[row_index]
        raise RuntimeError(
            f"assembled epoch table contains {int(invalid.sum())} non-finite feature "
            f"values; per-feature counts={worst[:10]}; first offending row: "
            f"participant={example.get('participant_id')!r} "
            f"recording={example.get('recording_id')!r} "
            f"channel={example.get('channel')!r} epoch={example.get('epoch_index')!r}"
        )


def assemble(
    derivatives_root: Path,
    cohort_csv: Path,
    output_dir: Path,
    repository: Path = REPOSITORY,
    cohort_name: str = "hup_ucla",
) -> dict[str, Path]:
    feature_files = sorted(
        derivatives_root.rglob("*@FrequencyDomainFeatures.epoch_features.parquet")
    )
    qc_files = sorted(derivatives_root.rglob("*@FrequencyDomainFeatures.qc.json"))
    with cohort_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        cohort_rows = list(csv.DictReader(stream))
    if not cohort_rows:
        raise ValueError(f"cohort CSV is empty: {cohort_csv}")
    if len(feature_files) != len(cohort_rows) or len(qc_files) != len(cohort_rows):
        raise RuntimeError(
            "incomplete NeuroDAGs extraction: expected one feature and QC artifact per "
            f"cohort recording ({len(cohort_rows)}), found {len(feature_files)} feature "
            f"and {len(qc_files)} QC artifacts"
        )
    epochs = pd.concat([pd.read_parquet(path) for path in feature_files], ignore_index=True)
    if epochs.empty:
        raise RuntimeError("NeuroDAGs extraction produced no valid channel-epoch rows")
    qc_rows: list[dict[str, Any]] = []
    for path in qc_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"QC artifact is not an object: {path}")
        qc_rows.append(payload)
    qc = pd.DataFrame(qc_rows)
    incomplete = qc.loc[~qc["status"].eq("completed"), "recording_id"].tolist()
    recordings_failed = len(incomplete)
    if incomplete:
        raise RuntimeError(f"NeuroDAGs extraction has failed recordings: {incomplete}")
    # A "completed" recording must still carry no invalid feature value: the
    # extractors raise on non-finite features, so anything counted here means
    # the artifact predates that rule or was produced by a different build.
    _reject_invalid_features(qc, epochs)

    core = _core_module()
    channels = core.aggregate_channel_features(epochs)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: output_dir / filename for key, filename in output_filenames(cohort_name).items()}
    core.atomic_parquet(epochs, paths["epochs"])
    core.atomic_parquet(channels, paths["channels"])
    core.atomic_csv(qc, paths["qc"])
    summary = {
        "protocol_version": core.PROTOCOL_VERSION,
        "execution_model": "neurodags_per_recording_frequency_domain_v1",
        "preprocessing_derivative": core.DERIVATIVE_NAME,
        "expected_preprocessing_protocol_version": core.EXPECTED_PREPROCESSING_PROTOCOL_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_csv": str(cohort_csv.resolve()),
        "cohort_csv_sha256": _sha256(cohort_csv),
        "neurodags_derivatives_root": str(derivatives_root.resolve()),
        "repository": str(repository.resolve()),
        "git_commit": core.git_commit(repository),
        "invalid_epoch_policy": core.INVALID_EPOCH_POLICY,
        "sampling_frequency_hz": core.EXPECTED_SFREQ_HZ,
        "epoch_duration_s": core.EPOCH_DURATION_S,
        "samples_per_epoch": core.EXPECTED_SAMPLES_PER_EPOCH,
        "welch_parameters": {
            "sfreq_hz": core.EXPECTED_SFREQ_HZ,
            "n_fft": core.WELCH_N_FFT,
            "n_per_seg": core.WELCH_N_PER_SEG,
            "n_overlap": core.WELCH_N_OVERLAP,
            "window": core.WELCH_WINDOW,
            "average": core.WELCH_AVERAGE,
            "detrend": core.WELCH_DETREND,
            "scaling": core.WELCH_SCALING,
            "fmin_hz": core.WELCH_FMIN_HZ,
            "fmax_hz": core.WELCH_FMAX_HZ,
        },
        "feature_definitions": core.feature_definitions(),
        "recordings_attempted": len(cohort_rows),
        "recordings_completed": len(qc),
        "recordings_failed": recordings_failed,
        "epochs_input": int(qc["n_epochs_input"].sum()),
        "epochs_valid": int(qc["n_epochs_valid"].sum()),
        "epochs_rejected": int(qc["n_epochs_rejected"].sum()),
        "failed_windows": int(qc["n_failed_windows"].sum()),
        "total_epoch_rows": len(epochs),
        "total_channel_rows": len(channels),
        "output_files": {key: str(path.resolve()) for key, path in paths.items()},
    }
    core.atomic_json(summary, paths["summary"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derivatives-root", type=Path, required=True)
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--cohort-name", default="hup_ucla")
    args = parser.parse_args()
    paths = assemble(
        args.derivatives_root.resolve(), args.cohort_csv.resolve(),
        args.output_dir.resolve(), args.repository.resolve(), args.cohort_name,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
