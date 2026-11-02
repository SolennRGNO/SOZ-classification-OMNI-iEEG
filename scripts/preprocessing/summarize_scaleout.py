#!/usr/bin/env python3
"""Summarize HUP/UCLA subject-scale preprocessing outputs and resources."""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Iterable


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
BUILDER_PATH = SCRIPT_DIRECTORY / "build_subject_manifests.py"
SPEC = importlib.util.spec_from_file_location("hup_ucla_subject_manifest_builder", BUILDER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load subject manifest builder: {BUILDER_PATH}")
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


SUBJECT_FIELDS = (
    "manifest_path",
    "mode",
    "subject_index",
    "participant_id",
    "dataset",
    "completed",
    "number_of_recordings_expected",
    "number_of_recordings_attempted",
    "number_of_recordings_completed",
    "number_of_recordings_reused",
    "number_of_recordings_failed",
    "elapsed_seconds",
    "max_rss_mb",
    "total_fif_size_bytes",
    "total_report_size_bytes",
    "total_output_size_bytes",
    "failure_reason",
    "status_json_path",
)


# These are terminal success states emitted by the preprocessing worker.  A
# reused derivative is a validated completed recording whose artifacts reside
# in an earlier output root, so it must count as completed without losing its
# provenance in ``number_of_recordings_reused``.
SUCCESSFUL_RECORDING_STATUSES = frozenset(
    {
        "completed",
        "reused_from_previous_root",
        # Retain compatibility with status files written by earlier workers.
        "already_completed",
        "skipped_valid_existing",
    }
)

RESOURCE_FIELDS = (
    "row_type",
    "manifest_path",
    "mode",
    "participant_id",
    "dataset",
    "recording_id",
    "elapsed_seconds",
    "max_rss_mb",
    "fif_size_bytes",
    "report_size_bytes",
    "total_output_size_bytes",
    "status",
    "failure_reason",
)


def _clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _max_rss_mb(value: object) -> float | None:
    text = _clean(value).upper()
    if not text:
        return None
    match = __import__("re").fullmatch(r"([0-9.]+)\s*([KMGT]?)B?", text)
    if not match:
        return None
    number = float(match.group(1))
    factor = {"": 1 / (1024 * 1024), "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2}
    return number * factor[match.group(2)]


def load_sacct(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.is_file():
        return {}
    _, rows = _read_csv(path)
    output = {}
    for row in rows:
        job = _clean(row.get("JobID") or row.get("JobIDRaw") or row.get("job_id"))
        if not job or ".batch" in job or ".extern" in job:
            continue
        elapsed = row.get("ElapsedRaw") or row.get("elapsed_seconds")
        output[job] = {
            "elapsed_seconds": float(elapsed) if _clean(elapsed) else None,
            "max_rss_mb": _max_rss_mb(row.get("MaxRSS") or row.get("max_rss")),
        }
    return output


def _status_job_key(status: dict[str, Any]) -> str:
    job = _clean(status.get("slurm_job_id"))
    task = _clean(status.get("slurm_array_task_id"))
    return f"{job}_{task}" if job and task else job


def _artifact_validator_from_status(
    record_status: dict[str, Any]
) -> tuple[bool, str]:
    artifacts = {
        key: Path(value)
        for key, value in record_status.get("artifacts", {}).items()
        if key in {"fif", "metadata", "channels", "report"}
    }
    if set(artifacts) != {"fif", "metadata", "channels", "report"}:
        return False, "status_does_not_list_all_required_artifacts"
    return builder.validate_preprocessed_artifacts(artifacts)


def classify_subject(
    manifest_row: dict[str, str],
    manifest_path: Path,
    status: dict[str, Any] | None,
    status_path: Path | None,
    artifact_validator: Callable[[dict[str, Any]], tuple[bool, str]] = _artifact_validator_from_status,
    sacct: dict[str, dict[str, object]] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    expected = json.loads(manifest_row["recordings_json"])
    mode = "pilot" if "pilot" in manifest_path.name else "full"
    base = {
        "manifest_path": str(manifest_path),
        "mode": mode,
        "subject_index": int(manifest_row["subject_index"]),
        "participant_id": manifest_row["participant_id"],
        "dataset": manifest_row["dataset"],
        "number_of_recordings_expected": len(expected),
        "status_json_path": str(status_path) if status_path else "",
    }
    if status is None:
        subject = {
            **base,
            "completed": False,
            "number_of_recordings_attempted": 0,
            "number_of_recordings_completed": 0,
            "number_of_recordings_reused": 0,
            "number_of_recordings_failed": len(expected),
            "elapsed_seconds": "",
            "max_rss_mb": "",
            "total_fif_size_bytes": 0,
            "total_report_size_bytes": 0,
            "total_output_size_bytes": 0,
            "failure_reason": "missing_subject_status_json",
        }
        return subject, []

    records_by_id = {
        _clean(record.get("recording_id")): record for record in status.get("recordings", [])
    }
    resources: list[dict[str, object]] = []
    failures = []
    completed_count = 0
    reused_count = 0
    fif_size = report_size = total_size = 0
    for recording in expected:
        identifier = builder.recording_identifier(recording)
        record_status = records_by_id.get(identifier)
        if record_status is None:
            failures.append(f"missing_status_for_recording:{identifier}")
            continue
        valid, reason = artifact_validator(record_status)
        state = _clean(record_status.get("status"))
        completed_state = state in SUCCESSFUL_RECORDING_STATUSES
        if valid and completed_state:
            completed_count += 1
            if state == "reused_from_previous_root":
                reused_count += 1
        else:
            failures.append(reason or _clean(record_status.get("failure_reason")) or f"invalid_status:{state}")
        record_fif = int(record_status.get("fif_size_bytes", 0) or 0)
        record_report = int(record_status.get("report_size_bytes", 0) or 0)
        record_total = int(record_status.get("total_output_size_bytes", 0) or 0)
        fif_size += record_fif
        report_size += record_report
        total_size += record_total
        resources.append(
            {
                "row_type": "recording",
                "manifest_path": str(manifest_path),
                "mode": mode,
                "participant_id": manifest_row["participant_id"],
                "dataset": manifest_row["dataset"],
                "recording_id": identifier,
                "elapsed_seconds": record_status.get("elapsed_seconds", ""),
                "max_rss_mb": "",
                "fif_size_bytes": record_fif,
                "report_size_bytes": record_report,
                "total_output_size_bytes": record_total,
                "status": state,
                "failure_reason": record_status.get("failure_reason", ""),
            }
        )
    sacct_row = (sacct or {}).get(_status_job_key(status), {})
    elapsed = sacct_row.get("elapsed_seconds") or status.get("elapsed_seconds", "")
    max_rss = sacct_row.get("max_rss_mb", "")
    completed = completed_count == len(expected) and not failures
    subject = {
        **base,
        "completed": completed,
        "number_of_recordings_attempted": int(status.get("number_of_recordings_attempted", 0)),
        "number_of_recordings_completed": completed_count,
        "number_of_recordings_reused": reused_count,
        "number_of_recordings_failed": len(expected) - completed_count,
        "elapsed_seconds": elapsed,
        "max_rss_mb": max_rss,
        "total_fif_size_bytes": fif_size,
        "total_report_size_bytes": report_size,
        "total_output_size_bytes": total_size,
        "failure_reason": ";".join(dict.fromkeys(failures)) or _clean(status.get("failure_reason")),
    }
    resources.append(
        {
            "row_type": "subject",
            "manifest_path": str(manifest_path),
            "mode": mode,
            "participant_id": manifest_row["participant_id"],
            "dataset": manifest_row["dataset"],
            "recording_id": "",
            "elapsed_seconds": elapsed,
            "max_rss_mb": max_rss,
            "fif_size_bytes": fif_size,
            "report_size_bytes": report_size,
            "total_output_size_bytes": total_size,
            "status": "completed" if completed else "failed",
            "failure_reason": subject["failure_reason"],
        }
    )
    return subject, resources


def _status_index(roots: Iterable[Path]) -> dict[tuple[str, str, str], tuple[Path, dict[str, Any]]]:
    output = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("subject_status.json"):
            try:
                status = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            key = (
                str(Path(_clean(status.get("manifest_path"))).resolve(strict=False)),
                _clean(status.get("dataset")),
                _clean(status.get("participant_id")),
            )
            if key not in output or path.stat().st_mtime > output[key][0].stat().st_mtime:
                output[key] = (path, status)
    return output


def _median(values: Iterable[object]) -> float | str:
    numbers = [float(value) for value in values if _clean(value)]
    return statistics.median(numbers) if numbers else ""


def _write_html(path: Path, subjects: list[dict[str, object]]) -> None:
    rows = []
    # Only the datasets this run actually covers, so a single-dataset scale-out
    # does not report empty rows for the datasets it did not touch.
    datasets = [
        dataset
        for dataset in builder.DATASETS
        if any(row["dataset"] == dataset for row in subjects)
    ] or list(builder.DATASETS)
    for dataset in datasets:
        selected = [row for row in subjects if row["dataset"] == dataset]
        failures = sum(not bool(row["completed"]) for row in selected)
        rows.append(
            {
                "dataset": dataset,
                "subjects": len(selected),
                "median_runtime_seconds": _median(row["elapsed_seconds"] for row in selected),
                "median_max_rss_mb": _median(row["max_rss_mb"] for row in selected),
                "median_fif_size_bytes": _median(row["total_fif_size_bytes"] for row in selected),
                "failure_rate_percent": 100 * failures / len(selected) if selected else "",
                "median_recordings_per_subject": _median(
                    row["number_of_recordings_expected"] for row in selected
                ),
            }
        )
    headings = list(rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in headings) + "</tr>"
        for row in rows
    )
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Preprocessing scale-out</title>"
        "<style>body{font-family:sans-serif;margin:2rem}table{border-collapse:collapse}th,td{border:1px solid #aaa;padding:.4rem}</style>"
        "</head><body><h1>HUP/UCLA preprocessing scale-out</h1><table><thead><tr>"
        + "".join(f"<th>{html.escape(key)}</th>" for key in headings)
        + f"</tr></thead><tbody>{body}</tbody></table></body></html>\n",
        encoding="utf-8",
    )


def summarize(
    manifests: list[Path],
    status_roots: list[Path],
    output_dir: Path,
    excluded_paths: list[Path] | None = None,
    sacct_path: Path | None = None,
) -> dict[str, Path]:
    statuses = _status_index(status_roots)
    sacct = load_sacct(sacct_path)
    subjects = []
    resources = []
    for manifest in manifests:
        _, rows = _read_csv(manifest)
        for row in rows:
            key = (str(manifest.resolve()), row["dataset"], row["participant_id"])
            status_entry = statuses.get(key)
            subject, subject_resources = classify_subject(
                row,
                manifest,
                status_entry[1] if status_entry else None,
                status_entry[0] if status_entry else None,
                sacct=sacct,
            )
            subjects.append(subject)
            resources.extend(subject_resources)
    completed = [row for row in subjects if row["completed"]]
    failed = [row for row in subjects if not row["completed"]]
    excluded_rows = []
    excluded_fields = []
    for path in excluded_paths or []:
        fields, rows = _read_csv(path)
        excluded_fields = list(dict.fromkeys([*excluded_fields, *fields]))
        excluded_rows.extend(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "completed": output_dir / "completed_subjects.csv",
        "failed": output_dir / "failed_subjects.csv",
        "excluded": output_dir / "excluded_recordings.csv",
        "resources": output_dir / "preprocessing_resource_summary.csv",
        "html": output_dir / "preprocessing_scaleout_report.html",
    }
    _write_csv(paths["completed"], SUBJECT_FIELDS, completed)
    _write_csv(paths["failed"], SUBJECT_FIELDS, failed)
    _write_csv(paths["excluded"], excluded_fields or ["scaleout_exclusion_reason"], excluded_rows)
    _write_csv(paths["resources"], RESOURCE_FIELDS, resources)
    _write_html(paths["html"], subjects)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--status-root", type=Path, action="append", required=True)
    parser.add_argument("--excluded-recordings", type=Path, action="append", default=[])
    parser.add_argument("--sacct-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(
        [path.expanduser() for path in args.manifest],
        [path.expanduser() for path in args.status_root],
        args.output_dir.expanduser(),
        [path.expanduser() for path in args.excluded_recordings],
        args.sacct_csv.expanduser() if args.sacct_csv else None,
    )


if __name__ == "__main__":
    main()
