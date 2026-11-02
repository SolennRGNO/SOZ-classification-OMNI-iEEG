#!/usr/bin/env python3
"""Build scale-out subject manifests and run one subject sequentially."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LOGGER = logging.getLogger("hup_ucla_preprocessing_scaleout")
DATASETS = ("hup", "ucla", "openieeg")
# Per-dataset NeuroDAGs protocol and reviewed line-noise configuration. HUP and
# UCLA share both; OpenIEEG uses the same node with its own reviewed site list.
PIPELINES = {
    "hup": "pipeline_hup_ucla_1khz.yml",
    "ucla": "pipeline_hup_ucla_1khz.yml",
    "openieeg": "pipeline_openieeg_1khz.yml",
}
LINE_NOISE_CONFIGS = {
    "hup": "config/line_noise_hup_ucla.yml",
    "ucla": "config/line_noise_hup_ucla.yml",
    "openieeg": "config/line_noise_openieeg.yml",
}
DERIVATIVE = "Preprocessed1kHzNoBandpass"
MINIMUM_SFREQ_HZ = 1000.0
# Technically eligible recording states. Preprocessing is state-agnostic: the
# task is metadata, and the surgical outcome is never used to filter here.
RECORDING_STATES = ("interictal", "ictal")
PROTOCOL_VERSION = "scaleout-v1"
SUBJECT_COLUMNS = (
    "subject_index",
    "participant_id",
    "dataset",
    "number_of_recordings",
    "recording_identifiers",
    "edf_paths",
    "original_sampling_frequencies_hz",
    "selection_reason",
    "recordings_json",
    "source_cohort_sha256",
)


def _clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truthy(value: object) -> bool:
    return _clean(value).lower() in {"1", "1.0", "true", "yes", "y"}


def _float(value: object) -> float | None:
    try:
        result = float(_clean(value))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), [
            {str(key): _clean(value) for key, value in row.items()} for row in reader
        ]


def _write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recording_identifier(row: dict[str, str]) -> str:
    explicit = _clean(row.get("recording_id"))
    if explicit:
        return explicit
    name = Path(row["edf_path"]).name
    return name[:-4] if name.lower().endswith(".edf") else Path(name).stem


def classify_cohort_rows(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    paths = [str(Path(row.get("edf_path", "")).expanduser().absolute()) for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate EDF paths are not allowed")
    included: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for source in rows:
        row = dict(source)
        dataset = _clean(row.get("dataset")).lower()
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset in cohort: {dataset!r}")
        row["dataset"] = dataset
        sfreq = _float(row.get("original_sampling_frequency_hz"))
        reason = ""
        if sfreq is None:
            reason = "missing_or_malformed_original_sampling_frequency"
        elif sfreq < MINIMUM_SFREQ_HZ:
            reason = "original_sampling_frequency_below_1000_hz"
        elif dataset == "hup" and _clean(row.get("task")).lower() not in RECORDING_STATES:
            reason = "hup_missing_or_unknown_task"
        elif dataset == "openieeg":
            # OpenIEEG recordings carry no ictal/interictal labelling (the BIDS
            # task entity is the recording condition, not a recording state), so
            # the sampling-frequency rule above is the only eligibility rule.
            pass
        elif dataset == "ucla":
            state = _clean(row.get("recording_state")).lower()
            source_rule = _clean(row.get("recording_state_source")).lower()
            if state and state != "interictal":
                reason = "ucla_recording_state_not_interictal"
            elif source_rule and source_rule != "dataset_rule_ucla_all_interictal":
                reason = "ucla_recording_state_source_disagrees_with_dataset_rule"
        if reason:
            row["scaleout_exclusion_reason"] = reason
            excluded.append(row)
        else:
            row["scaleout_exclusion_reason"] = ""
            included.append(row)
    return included, excluded


def _validated_subjects(roots: Iterable[Path]) -> set[tuple[str, str]]:
    validated: set[tuple[str, str]] = set()
    pattern = f"*@{DERIVATIVE}.metadata.json"
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob(pattern):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue
            dataset = _clean(metadata.get("dataset") or metadata.get("site")).lower()
            participant = _clean(metadata.get("participant_id"))
            if dataset in DATASETS and participant:
                validated.add((dataset, participant))
    return validated


def _identifier_from_path(value: object) -> str:
    name = Path(_clean(value)).name
    return name[:-4] if name.lower().endswith(".edf") else Path(name).stem


def index_existing_derivatives(
    roots: Iterable[Path],
) -> dict[tuple[str, str, str], dict[str, Path]]:
    """Index already-computed derivatives so valid work is never repeated.

    Keys are ``(dataset, participant_id, recording_identifier)``; the recording
    directory name is not used because adding ictal recordings renumbers it.
    """
    index: dict[tuple[str, str, str], dict[str, Path]] = {}
    pattern = f"*@{DERIVATIVE}.metadata.json"
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob(pattern)):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue
            dataset = _clean(metadata.get("dataset") or metadata.get("site")).lower()
            participant = _clean(metadata.get("participant_id"))
            identifiers = {
                _clean(metadata.get("recording_id")),
                _identifier_from_path(metadata.get("edf_path")),
                _identifier_from_path(metadata.get("source_edf_path")),
            }
            identifiers.discard("")
            if dataset not in DATASETS or not participant or not identifiers:
                continue
            prefix = path.name[: -len(".metadata.json")]
            artifacts = {
                "fif": path.with_name(f"{prefix}.fif"),
                "metadata": path,
                "channels": path.with_name(f"{prefix}.channels.tsv"),
                "report": path.with_name(f"{prefix}.report.html"),
            }
            for identifier in identifiers:
                index.setdefault((dataset, participant, identifier), artifacts)
    return index


def _subject_score(
    dataset: str,
    participant: str,
    recordings: list[dict[str, str]],
    validated: set[tuple[str, str]],
) -> tuple[int, int, float, int]:
    clinical = sum(
        all(
            _truthy(row.get(field))
            for field in (
                "has_soz_metadata",
                "has_resection_metadata",
                "has_outcome_metadata",
            )
        )
        for row in recordings
    )
    valid_channels = sum(
        _float(row.get("number_of_good_ieeg_channels")) or 0.0 for row in recordings
    )
    return (
        int((dataset, participant) in validated),
        clinical,
        valid_channels,
        len(recordings),
    )


def datasets_present(recordings: Iterable[dict[str, str]]) -> tuple[str, ...]:
    """Return the supported datasets this cohort actually contains, in order."""
    present = {row["dataset"] for row in recordings}
    return tuple(dataset for dataset in DATASETS if dataset in present)


def group_subjects(
    recordings: list[dict[str, str]],
    source_hash: str,
    validated: set[tuple[str, str]] | None = None,
    datasets: Iterable[str] | None = None,
) -> dict[str, list[dict[str, object]]]:
    validated = validated or set()
    datasets = tuple(datasets) if datasets is not None else datasets_present(recordings)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in recordings:
        key = (row["dataset"], _clean(row.get("participant_id")))
        if not key[1]:
            raise ValueError("recording is missing participant_id")
        grouped.setdefault(key, []).append(row)
    output: dict[str, list[dict[str, object]]] = {dataset: [] for dataset in datasets}
    for (dataset, participant), rows in grouped.items():
        rows = sorted(rows, key=lambda row: (recording_identifier(row), row["edf_path"]))
        score = _subject_score(dataset, participant, rows, validated)
        output[dataset].append(
            {
                "participant_id": participant,
                "dataset": dataset,
                "number_of_recordings": len(rows),
                "recording_identifiers": ";".join(recording_identifier(row) for row in rows),
                "edf_paths": ";".join(row["edf_path"] for row in rows),
                "original_sampling_frequencies_hz": ";".join(
                    f"{float(row['original_sampling_frequency_hz']):g}" for row in rows
                ),
                "selection_reason": (
                    f"validated_preprocessing={score[0]};"
                    f"recordings_with_complete_clinical_metadata={score[1]};"
                    f"total_good_ieeg_channels={score[2]:g};"
                    f"number_of_recordings={score[3]}"
                ),
                "recordings_json": json.dumps(rows, sort_keys=True, separators=(",", ":")),
                "source_cohort_sha256": source_hash,
                "_score": score,
            }
        )
    for dataset in datasets:
        output[dataset].sort(
            key=lambda row: (
                -row["_score"][0],
                -row["_score"][1],
                -row["_score"][2],
                -row["_score"][3],
                row["participant_id"],
            )
        )
    return output


def _indexed(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for index, source in enumerate(rows):
        row = {key: value for key, value in source.items() if not key.startswith("_")}
        row["subject_index"] = index
        result.append(row)
    return result


def validate_subject_manifest(rows: list[dict[str, object]], dataset: str) -> None:
    participants = [_clean(row.get("participant_id")) for row in rows]
    if len(participants) != len(set(participants)):
        raise ValueError("subject manifest contains duplicated subjects")
    all_paths: list[str] = []
    for expected_index, row in enumerate(rows):
        if int(row["subject_index"]) != expected_index:
            raise ValueError("subject_index must be contiguous and zero-based")
        if row["dataset"] != dataset:
            raise ValueError("subject manifest contains a different dataset")
        recordings = json.loads(str(row["recordings_json"]))
        if int(row["number_of_recordings"]) != len(recordings) or not recordings:
            raise ValueError("subject manifest recording count is invalid")
        identifiers = [recording_identifier(recording) for recording in recordings]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("subject manifest contains duplicated recording identifiers")
        for recording in recordings:
            sfreq = _float(recording.get("original_sampling_frequency_hz"))
            if sfreq is None or sfreq < MINIMUM_SFREQ_HZ:
                raise ValueError("subject manifest includes a recording below 1000 Hz")
            if dataset == "hup" and _clean(recording.get("task")).lower() not in RECORDING_STATES:
                raise ValueError(
                    "subject manifest includes a HUP recording without a known recording state"
                )
            all_paths.append(str(Path(recording["edf_path"]).expanduser().absolute()))
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("subject manifest contains duplicated EDF paths")


def build_manifests(
    cohort_csv: Path,
    output_dir: Path,
    pilot_subjects_per_dataset: int = 3,
    validated_roots: Iterable[Path] = (),
) -> dict[str, Path]:
    if pilot_subjects_per_dataset < 1:
        raise ValueError("pilot subject count must be positive")
    fields, source_rows = _read_csv(cohort_csv)
    required = {"participant_id", "dataset", "task", "edf_path", "original_sampling_frequency_hz"}
    missing = required - set(fields)
    if missing:
        raise ValueError("cohort lacks required columns: " + ", ".join(sorted(missing)))
    included, excluded = classify_cohort_rows(source_rows)
    source_hash = _sha256(cohort_csv)
    # Only the datasets this cohort contains are grouped and written, so a
    # single-dataset cohort (OpenIEEG) and the HUP/UCLA cohort both build here
    # without either changing the other's manifests.
    datasets = datasets_present(included)
    if not datasets:
        raise ValueError("cohort contains no eligible recordings")
    grouped = group_subjects(
        included, source_hash, _validated_subjects(validated_roots), datasets
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for dataset in datasets:
        full = _indexed(grouped[dataset])
        if len(full) < pilot_subjects_per_dataset:
            raise ValueError(
                f"{dataset} has only {len(full)} eligible subjects; "
                f"cannot select {pilot_subjects_per_dataset}"
            )
        pilot = _indexed(grouped[dataset][:pilot_subjects_per_dataset])
        validate_subject_manifest(full, dataset)
        validate_subject_manifest(pilot, dataset)
        paths[f"{dataset}_pilot"] = output_dir / f"{dataset}_multisubject_pilot.csv"
        paths[f"{dataset}_full"] = output_dir / f"{dataset}_full_subjects.csv"
        _write_csv(paths[f"{dataset}_pilot"], SUBJECT_COLUMNS, pilot)
        _write_csv(paths[f"{dataset}_full"], SUBJECT_COLUMNS, full)
    excluded_fields = list(dict.fromkeys([*fields, "scaleout_exclusion_reason"]))
    paths["excluded"] = output_dir / "excluded_recordings.csv"
    _write_csv(paths["excluded"], excluded_fields, excluded)
    return paths


def count_subject_rows(manifest: Path) -> int:
    _, rows = _read_csv(manifest)
    if not rows:
        raise ValueError("subject manifest is empty")
    return len(rows)


def subject_array_spec(manifest: Path) -> str:
    return f"0-{count_subject_rows(manifest) - 1}"


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"


def artifact_paths(subject_dir: Path, recording_directory_name: str) -> dict[str, Path]:
    """Return the flat subject-directory derivative artifact paths.

    NeuroDAGs writes the derivative beside the temporary recording input
    directory, using that directory's name as the recording prefix.
    """
    prefix = subject_dir / f"{recording_directory_name}@{DERIVATIVE}"
    return {
        "fif": Path(str(prefix) + ".fif"),
        "metadata": Path(str(prefix) + ".metadata.json"),
        "channels": Path(str(prefix) + ".channels.tsv"),
        "report": Path(str(prefix) + ".report.html"),
    }


def validate_preprocessed_artifacts(
    artifacts: dict[str, Path],
    *,
    read_fif: bool = True,
    expected_protocol_version: str | None = None,
    expected_notch_frequencies: tuple[float, ...] | None = None,
) -> tuple[bool, str]:
    for name, path in artifacts.items():
        if not path.is_file() or path.stat().st_size == 0:
            return False, f"missing_or_empty_{name}:{path}"
    try:
        metadata = json.loads(artifacts["metadata"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"unreadable_metadata:{error}"
    try:
        final_sfreq = float(metadata.get("final_sfreq", 0))
        epoch_duration = float(metadata.get("epoch_duration_seconds", 0))
        epoch_overlap = float(metadata.get("epoch_overlap_seconds", -1))
    except (TypeError, ValueError) as error:
        return False, f"invalid_preprocessing_metadata_values:{error}"
    if not math.isclose(final_sfreq, 1000.0, abs_tol=1e-9):
        return False, "metadata_final_sampling_frequency_is_not_1000_hz"
    if epoch_duration != 8.0:
        return False, "metadata_epoch_duration_is_not_8_seconds"
    if epoch_overlap != 0.0:
        return False, "metadata_epoch_overlap_is_not_zero"
    if metadata.get("l_freq") not in (None, "") or metadata.get("h_freq") not in (None, ""):
        return False, "metadata_records_bandpass_filtering"
    if (
        expected_protocol_version is not None
        and metadata.get("protocol_version") != expected_protocol_version
    ):
        return False, "metadata_protocol_version_disagrees_with_configuration"
    if expected_notch_frequencies is not None:
        try:
            configured = tuple(float(value) for value in metadata["configured_notch_frequencies"])
            original_sfreq = float(metadata["original_sfreq"])
            applied = tuple(float(value) for value in metadata["applied_notch_frequencies"])
            skipped = tuple(float(value) for value in metadata["skipped_notch_frequencies"])
        except (KeyError, TypeError, ValueError) as error:
            return False, f"invalid_notch_metadata:{error}"
        expected_applied = tuple(
            frequency
            for frequency in expected_notch_frequencies
            if frequency < original_sfreq / 2.0
        )
        expected_skipped = tuple(
            frequency
            for frequency in expected_notch_frequencies
            if frequency >= original_sfreq / 2.0
        )
        if configured != expected_notch_frequencies:
            return False, "metadata_configured_notches_disagree_with_configuration"
        if applied != expected_applied:
            return False, "metadata_applied_notches_disagree_with_configuration"
        if skipped != expected_skipped:
            return False, "metadata_skipped_notches_disagree_with_configuration"
    if read_fif:
        try:
            import mne

            epochs = mne.read_epochs(artifacts["fif"], preload=False, verbose="ERROR")
            try:
                if not math.isclose(float(epochs.info["sfreq"]), 1000.0, abs_tol=1e-9):
                    return False, "fif_sampling_frequency_is_not_1000_hz"
                if len(epochs.times) != 8000:
                    return False, "fif_epoch_does_not_contain_8000_samples"
                if len(epochs) < 1:
                    return False, "fif_contains_no_epochs"
            finally:
                del epochs
        except Exception as error:
            return False, f"unreadable_fif:{type(error).__name__}:{error}"
    return True, ""


def _load_expected_line_noise_configuration(
    repository: Path, dataset: str
) -> tuple[str, tuple[float, ...]]:
    """Load the reviewed configuration that this dataset's pipeline will use."""
    module_path = repository / "Neurodags" / "line_noise_config.py"
    module_name = "hup_ucla_scaleout_line_noise_config"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load line-noise configuration module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    configuration = module.load_line_noise_config(
        repository / "Neurodags" / LINE_NOISE_CONFIGS[dataset],
        datasets_path=repository / "Neurodags" / "datasets2.yml",
        required_sites={dataset},
    )
    return configuration.protocol_version, configuration.frequencies_for(dataset)


def _archive_invalid_artifacts(artifacts: dict[str, Path], subject_dir: Path) -> None:
    prefix = artifacts["fif"].name.removesuffix(".fif")
    existing = sorted(
        {path for path in artifacts.values() if path.exists()}
        | set(subject_dir.glob(f"{prefix}.*"))
    )
    if not existing:
        return
    archive = subject_dir / "invalid_previous" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    archive.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.move(str(path), archive / path.name)


def _prepare_recording_configuration(
    recording: dict[str, str], dataset: str, recording_dir: Path
) -> Path:
    source = Path(recording["edf_path"]).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"source EDF does not exist: {source}")
    input_dir = recording_dir / "_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    link = input_dir / f"{dataset}_ieeg.edf"
    if link.is_symlink() and link.resolve() != source.resolve():
        link.unlink()
    if not link.exists():
        link.symlink_to(source.resolve())
    eligible_manifest = input_dir / "eligible_recording.csv"
    manifest_row = dict(recording)
    manifest_row.update(
        dataset=dataset,
        source_edf_path=str(source.resolve()),
        eligible_sfreq_ge_1000="True",
    )
    _write_csv(eligible_manifest, manifest_row.keys(), [manifest_row])
    datasets_yaml = input_dir / "datasets.yml"
    datasets_yaml.write_text(
        "\n".join(
            [
                f"{dataset}:",
                f"  name: {dataset}",
                "  skip: False",
                f"  file_pattern: {json.dumps(str(link))}",
                f"  derivatives_path: {json.dumps(str(recording_dir))}",
                "  vars:",
                f"    site: {dataset}",
                f"    dataset: {dataset}",
                f"    eligible_manifest_path: {json.dumps(str(eligible_manifest))}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return datasets_yaml


def _size_total(artifacts: dict[str, Path]) -> int:
    return sum(path.stat().st_size for path in artifacts.values() if path.is_file())


def _reusable_artifacts(
    index: dict[tuple[str, str, str], dict[str, Path]],
    dataset: str,
    participant: str,
    recording: dict[str, str],
    expected_protocol_version: str | None,
    expected_notch_frequencies: tuple[float, ...] | None,
) -> dict[str, Path] | None:
    """Return an already-valid derivative for this recording, if one exists."""
    if not index:
        return None
    identifier = recording_identifier(recording)
    candidates = [
        index.get((dataset, participant, identifier)),
        index.get((dataset, participant, _identifier_from_path(recording.get("edf_path")))),
    ]
    for artifacts in candidates:
        if artifacts is None:
            continue
        valid, _ = validate_preprocessed_artifacts(
            artifacts,
            expected_protocol_version=expected_protocol_version,
            expected_notch_frequencies=expected_notch_frequencies,
        )
        if valid:
            return artifacts
    return None


def run_subject(
    manifest: Path,
    subject_index: int,
    output_root: Path,
    repository: Path,
    expected_dataset: str,
    mode: str,
    reuse_roots: Iterable[Path] = (),
) -> Path:
    _, subjects = _read_csv(manifest)
    if subject_index < 0 or subject_index >= len(subjects):
        raise IndexError(f"subject index {subject_index} is outside manifest")
    subject = subjects[subject_index]
    if int(subject["subject_index"]) != subject_index:
        raise ValueError("SLURM array index disagrees with manifest subject_index")
    if subject["dataset"] != expected_dataset:
        raise ValueError("worker dataset disagrees with subject manifest")
    participant = subject["participant_id"]
    recordings = json.loads(subject["recordings_json"])
    expected_protocol_version, expected_notches = _load_expected_line_noise_configuration(
        repository, expected_dataset
    )
    reuse_roots = [Path(root) for root in reuse_roots]
    reusable = index_existing_derivatives(reuse_roots) if reuse_roots else {}
    subject_dir = output_root / _safe_component(participant)
    status_path = subject_dir / "subject_status.json"
    started = time.monotonic()
    status: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "manifest_path": str(manifest.resolve()),
        "mode": mode,
        "subject_index": subject_index,
        "participant_id": participant,
        "dataset": expected_dataset,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "reuse_roots": [str(root) for root in reuse_roots],
        "number_of_recordings_reused": 0,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "number_of_recordings_attempted": 0,
        "number_of_recordings_completed": 0,
        "number_of_recordings_failed": 0,
        "recordings": [],
        "completed": False,
        "failure_reason": "",
    }
    subject_dir.mkdir(parents=True, exist_ok=True)
    _write_json(status_path, status)
    try:
        for ordinal, recording in enumerate(recordings):
            identifier = recording_identifier(recording)
            recording_dir = subject_dir / f"recording-{ordinal:04d}-{_safe_component(identifier)}"
            recording_dir.mkdir(parents=True, exist_ok=True)
            artifacts = artifact_paths(subject_dir, recording_dir.name)
            record_started = time.monotonic()
            record_status: dict[str, Any] = {
                "recording_index_within_subject": ordinal,
                "recording_id": identifier,
                "edf_path": recording["edf_path"],
                "output_directory": str(recording_dir),
                "artifacts": {key: str(value) for key, value in artifacts.items()},
                "status": "attempted",
                "failure_reason": "",
            }
            status["number_of_recordings_attempted"] += 1
            valid, reason = validate_preprocessed_artifacts(
                artifacts,
                expected_protocol_version=expected_protocol_version,
                expected_notch_frequencies=expected_notches,
            )
            if valid:
                record_status["status"] = "already_completed"
                status["number_of_recordings_completed"] += 1
            elif (reused := _reusable_artifacts(reusable, expected_dataset, participant, recording, expected_protocol_version, expected_notches)) is not None:
                # A previous analysis root already holds a valid derivative for
                # this recording; reuse it in place instead of recomputing.
                record_status["status"] = "reused_from_previous_root"
                record_status["reused_artifacts"] = {
                    key: str(value) for key, value in reused.items()
                }
                artifacts = reused
                record_status["artifacts"] = record_status["reused_artifacts"]
                status["number_of_recordings_reused"] += 1
                status["number_of_recordings_completed"] += 1
            else:
                try:
                    _archive_invalid_artifacts(artifacts, subject_dir)
                    datasets_yaml = _prepare_recording_configuration(
                        recording, expected_dataset, recording_dir
                    )
                    neurodags = Path(sys.executable).with_name("neurodags")
                    command = [
                        str(neurodags),
                        "--log-level",
                        "INFO",
                        "run",
                        str(repository / "Neurodags" / PIPELINES[expected_dataset]),
                        "-d",
                        str(datasets_yaml),
                        "--derivative",
                        DERIVATIVE,
                        "--only-index",
                        "0",
                        "--raise-on-error",
                        "--n-jobs",
                        "1",
                    ]
                    result = subprocess.run(command, cwd=repository, check=False)
                    if result.returncode != 0:
                        raise RuntimeError(f"NeuroDAGs exited with code {result.returncode}")
                    valid, reason = validate_preprocessed_artifacts(
                        artifacts,
                        expected_protocol_version=expected_protocol_version,
                        expected_notch_frequencies=expected_notches,
                    )
                    if not valid:
                        raise RuntimeError(f"output validation failed: {reason}")
                    record_status["status"] = "completed"
                    status["number_of_recordings_completed"] += 1
                except Exception as error:
                    record_status["status"] = "failed"
                    record_status["failure_reason"] = f"{type(error).__name__}: {error}"
                    status["number_of_recordings_failed"] += 1
            record_status["elapsed_seconds"] = time.monotonic() - record_started
            record_status["fif_size_bytes"] = (
                artifacts["fif"].stat().st_size if artifacts["fif"].is_file() else 0
            )
            record_status["report_size_bytes"] = (
                artifacts["report"].stat().st_size if artifacts["report"].is_file() else 0
            )
            record_status["total_output_size_bytes"] = _size_total(artifacts)
            status["recordings"].append(record_status)
            _write_json(status_path, status)
        status["completed"] = status["number_of_recordings_failed"] == 0 and (
            status["number_of_recordings_completed"] == len(recordings)
        )
        if not status["completed"]:
            status["failure_reason"] = "one_or_more_recordings_failed"
    except Exception as error:
        status["failure_reason"] = f"{type(error).__name__}: {error}"
        status["completed"] = False
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        status["elapsed_seconds"] = time.monotonic() - started
        _write_json(status_path, status)
    if not status["completed"]:
        raise RuntimeError(status["failure_reason"])
    return status_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--cohort-csv", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--pilot-subjects-per-dataset", type=int, default=3)
    build.add_argument("--validated-preprocessing-root", type=Path, action="append", default=[])
    count = subparsers.add_parser("count")
    count.add_argument("--manifest", type=Path, required=True)
    worker = subparsers.add_parser("run-subject")
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--subject-index", type=int, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--repository", type=Path, required=True)
    worker.add_argument("--dataset", choices=DATASETS, required=True)
    worker.add_argument("--mode", choices=("pilot", "full"), required=True)
    worker.add_argument(
        "--reuse-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "previous analysis preprocessing root to reuse already-valid derivatives "
            "from; repeatable. Reused derivatives are read in place and never modified."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.command == "build":
        paths = build_manifests(
            args.cohort_csv.expanduser().resolve(),
            args.output_dir.expanduser(),
            args.pilot_subjects_per_dataset,
            [path.expanduser() for path in args.validated_preprocessing_root],
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
    elif args.command == "count":
        print(count_subject_rows(args.manifest.expanduser()))
    else:
        run_subject(
            args.manifest.expanduser(),
            args.subject_index,
            args.output_root.expanduser(),
            args.repository.expanduser().resolve(),
            args.dataset,
            args.mode,
            [path.expanduser() for path in args.reuse_root],
        )


if __name__ == "__main__":
    main()
