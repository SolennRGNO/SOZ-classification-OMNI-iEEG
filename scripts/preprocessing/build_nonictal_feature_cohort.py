#!/usr/bin/env python3
"""Build the recording and channel manifests for the HUP/UCLA feature cohort.

The cohort is technical only: every HUP/UCLA recording at >= 1000 Hz whose
recording state is explicitly known is eligible, whether it is interictal or
ictal. The recording state and the surgical outcome are carried as metadata and
are never used to filter here; the seizure-free restriction is applied later, in
the ML dataset stage. Output filenames keep their historical ``nonictal`` names
so existing wrappers and v2 provenance stay valid.

This script is metadata-only: it never opens an EDF or performs preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


LOGGER = logging.getLogger("hup_ucla_nonictal_feature_cohort")
PROTOCOL_VERSION = "1.1.0"
SITE_ORDER = ("hup", "ucla")
MINIMUM_ORIGINAL_SFREQ_HZ = 1000.0
RECORDING_STATES = ("interictal", "ictal")
PILOT_SELECTION = (
    ("hup", "interictal"),
    ("hup", "ictal"),
    ("ucla", "interictal"),
)

OUTPUT_NAMES = {
    "cohort": "nonictal_feature_cohort_hup_ucla.csv",
    "excluded": "excluded_feature_recordings_hup_ucla.csv",
    "audit": "feature_cohort_recording_audit_hup_ucla.csv",
    "summary": "feature_cohort_summary_hup_ucla.csv",
    "html": "feature_cohort_summary_hup_ucla.html",
    "pilot": "feature_pilot_manifest_v1.csv",
    "protocol": "feature_cohort_protocol_v1.json",
    "channels": "channel_label_audit_hup_ucla.csv",
}

DETROIT_OUTPUT_NAMES = {
    "cohort": "openieeg_detroit_final_eligible_participants.csv",
    "excluded": "openieeg_detroit_exclusions.csv",
    "qc": "openieeg_detroit_participant_qc.csv",
    "channels": "openieeg_detroit_channel_label_audit.csv",
    "funnel": "openieeg_detroit_cohort_funnel.csv",
    "summary": "openieeg_detroit_run_summary.json",
}

REQUIRED_INPUT_COLUMNS = {
    "participant_id",
    "dataset",
    "task",
    "edf_path",
    "original_sampling_frequency_hz",
}

ADDED_RECORDING_COLUMNS = (
    "recording_state",
    "recording_state_source",
    "feature_cohort_eligible",
    "exclusion_reason",
    "manual_review_required",
    "manual_review_reason",
    "surgical_outcome",
)

CHANNEL_AUDIT_COLUMNS = (
    "participant_id",
    "dataset",
    "institution",
    "session",
    "task",
    "run",
    "edf_path",
    "channels_tsv_path",
    "participants_tsv_path",
    "channel_name",
    "channel_type",
    "channel_status",
    "channel_soz",
    "channel_resection",
    "participant_surgical_outcome",
    "high_confidence_positive",
    "high_confidence_negative",
    "ambiguous_supervised_label",
)


def _clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def _normalise_site(value: object) -> str:
    return _clean(value).lower()


def _parse_sfreq(value: object) -> float | None:
    try:
        parsed = float(_clean(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _strict_binary(value: object) -> int | None:
    text = _clean(value).lower()
    if text in {"1", "1.0", "true", "yes"}:
        return 1
    if text in {"0", "0.0", "false", "no"}:
        return 0
    return None


def _shared_module():
    """Load the canonical surgical-outcome parser shared with feature labelling."""
    import importlib.util
    import sys

    name = "omni_cohort_surgical_outcome_reference"
    if name in sys.modules:
        return sys.modules[name]
    script = Path(__file__).resolve().parents[1] / "features" / "relative_bandpower.py"
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared feature infrastructure: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_surgical_outcome(value: object) -> int | None:
    """Canonical parser; raises on a present but unrecognised encoding."""
    return _shared_module().parse_surgical_outcome(value)


def _is_seizure_free(value: object) -> bool:
    """True only when the canonical parser says seizure-free."""
    return parse_surgical_outcome(value) == 1


def _first_available(mapping: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = _clean(mapping.get(name, ""))
        if value:
            return value
    return ""


def _read_delimited(path: Path, delimiter: str = ",") -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        fieldnames = list(reader.fieldnames or [])
        rows = [
            {str(key): _clean(value) for key, value in row.items()}
            for row in reader
        ]
    return fieldnames, rows


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(fieldnames))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _count_values(values: Iterable[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = _clean(value) or "missing"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _resolved_edf_path(row: dict[str, str], data_root: Path | None) -> Path:
    path = Path(row["edf_path"]).expanduser()
    if not path.is_absolute():
        if data_root is None:
            raise ValueError(f"relative EDF path requires --data-root: {path}")
        path = data_root / path
    return path.absolute()


def _channels_path(edf_path: Path) -> Path:
    name = re.sub(r"_ieeg\.edf$", "_channels.tsv", edf_path.name, flags=re.IGNORECASE)
    if name == edf_path.name:
        name = f"{edf_path.stem}_channels.tsv"
    return edf_path.with_name(name)


def _participants_path(edf_path: Path, data_root: Path | None) -> Path | None:
    stop = data_root.absolute() if data_root is not None else None
    for directory in (edf_path.parent, *edf_path.parents):
        candidate = directory / "participants.tsv"
        if candidate.is_file():
            return candidate
        if stop is not None and directory == stop:
            break
    if stop is not None:
        candidate = stop / "participants.tsv"
        if candidate.is_file():
            return candidate
    return None


def _participant_row(path: Path, participant_id: str) -> dict[str, str] | None:
    fields, rows = _read_delimited(path, "\t")
    identifier = next(
        (name for name in ("participant_id", "subject", "subject_id") if name in fields),
        None,
    )
    if identifier is None:
        return None
    matches = [row for row in rows if _clean(row.get(identifier)) == participant_id]
    if not matches:
        return None
    return matches[0]


def _classify_recording(row: dict[str, str]) -> dict[str, object]:
    classified: dict[str, object] = dict(row)
    site = _normalise_site(row.get("dataset"))
    task = _clean(row.get("task"))
    task_normalised = task.lower()
    sfreq = _parse_sfreq(row.get("original_sampling_frequency_hz"))

    classified.update(
        recording_state="",
        recording_state_source="",
        feature_cohort_eligible=False,
        exclusion_reason="",
        manual_review_required=False,
        manual_review_reason="",
    )

    if site not in SITE_ORDER:
        classified["exclusion_reason"] = "dataset_not_hup_or_ucla"
        return classified
    classified["dataset"] = site

    if sfreq is None:
        classified["exclusion_reason"] = "missing_or_malformed_original_sampling_frequency"
        return classified
    if sfreq < MINIMUM_ORIGINAL_SFREQ_HZ:
        classified["exclusion_reason"] = "original_sampling_frequency_below_1000_hz"
        return classified

    if site == "ucla":
        classified.update(
            recording_state="interictal",
            recording_state_source="dataset_rule_ucla_all_interictal",
            feature_cohort_eligible=True,
        )
        return classified

    if task_normalised == "interictal":
        classified.update(
            recording_state="interictal",
            recording_state_source="task_metadata",
            feature_cohort_eligible=True,
        )
    elif task_normalised == "ictal":
        classified.update(
            recording_state="ictal",
            recording_state_source="task_metadata",
            feature_cohort_eligible=True,
        )
    else:
        classified.update(
            recording_state="unknown",
            recording_state_source="task_metadata_missing_or_unknown",
            exclusion_reason="hup_missing_or_unknown_task",
            manual_review_required=True,
            manual_review_reason="verify_hup_recording_state",
        )
    return classified


def _source_outcome(row: dict[str, str], participant: dict[str, str] | None) -> str:
    value = _first_available(row, ("surgical_outcome", "outcome", "outcome_participant"))
    if value or participant is None:
        return value
    return _first_available(participant, ("surgical_outcome", "outcome", "seizure_free"))


def _channel_audit_rows(
    recording: dict[str, object], data_root: Path | None
) -> list[dict[str, object]]:
    row = {key: _clean(value) for key, value in recording.items()}
    edf_path = _resolved_edf_path(row, data_root)
    channels_path = _channels_path(edf_path)
    participants_path = _participants_path(edf_path, data_root)
    if not channels_path.is_file() or participants_path is None:
        return []
    participant = _participant_row(participants_path, row["participant_id"])
    if participant is None:
        return []
    fields, channels = _read_delimited(channels_path, "\t")
    channel_name_field = next((name for name in ("name", "channel", "channel_name") if name in fields), None)
    outcome = _source_outcome(row, participant)
    seizure_free = _is_seizure_free(outcome)
    output: list[dict[str, object]] = []
    for channel_index, channel in enumerate(channels):
        soz = _strict_binary(channel.get("soz"))
        resection = _strict_binary(channel.get("resection"))
        positive = soz == 1
        negative = (not positive) and resection == 0 and seizure_free
        output.append(
            {
                "participant_id": row["participant_id"],
                "dataset": row["dataset"],
                "institution": row.get("institution", ""),
                "session": row.get("session", ""),
                "task": row.get("task", ""),
                "run": row.get("run", ""),
                "edf_path": row["edf_path"],
                "channels_tsv_path": str(channels_path),
                "participants_tsv_path": str(participants_path),
                "channel_name": channel.get(channel_name_field, "") if channel_name_field else str(channel_index),
                "channel_type": _first_available(channel, ("type", "channel_type")),
                "channel_status": _first_available(channel, ("status", "good")),
                "channel_soz": channel.get("soz", ""),
                "channel_resection": channel.get("resection", ""),
                "participant_surgical_outcome": outcome,
                "high_confidence_positive": positive,
                "high_confidence_negative": negative,
                "ambiguous_supervised_label": not positive and not negative,
            }
        )
    return output


def _stable_key(row: dict[str, object]) -> tuple[str, int, str]:
    try:
        index = int(float(_clean(row.get("recording_index"))))
    except ValueError:
        index = 2**63 - 1
    return (_clean(row.get("participant_id")), index, _clean(row.get("edf_path")))


def _select_pilots(included: list[dict[str, object]]) -> list[dict[str, object]]:
    pilots = []
    for site, recording_state in PILOT_SELECTION:
        candidates = sorted(
            (
                row
                for row in included
                if row["dataset"] == site
                and _clean(row.get("recording_state")).lower() == recording_state
            ),
            key=_stable_key,
        )
        if not candidates:
            raise ValueError(
                "pilot manifest requires one eligible "
                f"{site} {recording_state} recording"
            )
        pilots.append(candidates[0])
    if [
        (row["dataset"], _clean(row.get("recording_state")).lower())
        for row in pilots
    ] != list(PILOT_SELECTION):
        raise ValueError(
            "pilot manifest must contain HUP interictal, HUP ictal, and UCLA interictal recordings"
        )
    return pilots


def _validate(
    audit: list[dict[str, object]],
    included: list[dict[str, object]],
    pilots: list[dict[str, object]],
) -> None:
    for row in included:
        sfreq = _parse_sfreq(row.get("original_sampling_frequency_hz"))
        if sfreq is None or sfreq < MINIMUM_ORIGINAL_SFREQ_HZ:
            raise ValueError("recording below 1000 Hz was included")
        state = _clean(row.get("recording_state")).lower()
        if state not in RECORDING_STATES:
            raise ValueError(
                f"included recording has unknown recording state {state!r}; "
                f"only {list(RECORDING_STATES)} are eligible"
            )
        if (
            row["dataset"] == "hup"
            and _clean(row.get("task")).lower() != state
        ):
            raise ValueError("HUP recording state disagrees with its BIDS task")
    if any(
        row["dataset"] == "ucla"
        and _clean(row.get("task")).lower() == "sleep"
        and (_parse_sfreq(row.get("original_sampling_frequency_hz")) or -1)
        >= MINIMUM_ORIGINAL_SFREQ_HZ
        and not bool(row["feature_cohort_eligible"])
        for row in audit
    ):
        raise ValueError("eligible UCLA sleep recording was excluded because of task")
    if len(pilots) != len(PILOT_SELECTION) or [
        (row["dataset"], _clean(row.get("recording_state")).lower())
        for row in pilots
    ] != list(PILOT_SELECTION):
        raise ValueError(
            "pilot manifest must contain HUP interictal, HUP ictal, and UCLA interictal recordings"
        )
    if any(not bool(row.get("feature_cohort_eligible")) for row in pilots):
        raise ValueError("pilot manifest contains an ineligible recording")
    for row, (site, recording_state) in zip(pilots, PILOT_SELECTION):
        if row["dataset"] != site or _clean(row.get("recording_state")).lower() != recording_state:
            raise ValueError("pilot manifest recording does not match its required site and state")
        if site == "hup" and _clean(row.get("task")).lower() != recording_state:
            raise ValueError("pilot manifest HUP recording state disagrees with its task")


def _summary_rows(audit: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for site in (*SITE_ORDER, "all"):
        rows = audit if site == "all" else [row for row in audit if row["dataset"] == site]
        included = [row for row in rows if bool(row["feature_cohort_eligible"])]
        durations = [
            value
            for row in included
            if (value := _parse_sfreq(row.get("duration_seconds"))) is not None
        ]
        by_state = {
            f"number_of_recordings_included_{state}": sum(
                _clean(row.get("recording_state")).lower() == state for row in included
            )
            for state in RECORDING_STATES
        }
        output.append(
            {
                "dataset": site,
                "number_of_recordings_audited": len(rows),
                "number_of_recordings_included": len(included),
                **by_state,
                "number_of_recordings_excluded": len(rows) - len(included),
                "number_requiring_manual_review": sum(bool(row["manual_review_required"]) for row in rows),
                "number_of_participants_included": len({_clean(row["participant_id"]) for row in included}),
                "included_duration_hours": sum(durations) / 3600,
            }
        )
    return output


def _write_html(path: Path, summary: list[dict[str, object]], source: Path) -> None:
    headings = list(summary[0])
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in headings) + "</tr>"
        for row in summary
    )
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>HUP/UCLA feature cohort</title>"
        "<style>body{font-family:sans-serif;margin:2rem}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:.4rem}</style>"
        "</head><body><h1>HUP/UCLA feature cohort (interictal and ictal)</h1>"
        f"<p>Protocol {PROTOCOL_VERSION}. Metadata-only selection from {html.escape(str(source))}.</p>"
        "<table><thead><tr>" + "".join(f"<th>{html.escape(key)}</th>" for key in headings) + "</tr></thead>"
        f"<tbody>{rows}</tbody></table></body></html>\n",
        encoding="utf-8",
    )


def build_cohort(
    inventory_csv: Path, output_dir: Path, data_root: Path | None = None
) -> dict[str, Path]:
    input_fields, source_rows = _read_delimited(inventory_csv)
    missing = REQUIRED_INPUT_COLUMNS - set(input_fields)
    if missing:
        raise ValueError("inventory lacks required columns: " + ", ".join(sorted(missing)))
    edf_paths = [str(_resolved_edf_path(row, data_root)) for row in source_rows]
    if len(edf_paths) != len(set(edf_paths)):
        raise ValueError("duplicate EDF paths exist in the input inventory")

    audit = [_classify_recording(row) for row in source_rows]
    for row in audit:
        edf = _resolved_edf_path({key: _clean(value) for key, value in row.items()}, data_root)
        participant_path = _participants_path(edf, data_root)
        participant = (
            _participant_row(participant_path, _clean(row["participant_id"]))
            if participant_path is not None
            else None
        )
        row["surgical_outcome"] = _source_outcome(
            {key: _clean(value) for key, value in row.items()}, participant
        )

    included = [row for row in audit if bool(row["feature_cohort_eligible"])]
    excluded = [row for row in audit if not bool(row["feature_cohort_eligible"])]
    pilots = _select_pilots(included)
    _validate(audit, included, pilots)
    channel_rows = [
        channel
        for recording in included
        for channel in _channel_audit_rows(recording, data_root)
    ]
    summary = _summary_rows(audit)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: output_dir / name for key, name in OUTPUT_NAMES.items()}
    recording_fields = list(dict.fromkeys([*input_fields, *ADDED_RECORDING_COLUMNS]))
    _write_csv(paths["cohort"], recording_fields, included)
    _write_csv(paths["excluded"], recording_fields, excluded)
    _write_csv(paths["audit"], recording_fields, audit)
    _write_csv(paths["pilot"], recording_fields, pilots)
    _write_csv(paths["summary"], summary[0].keys(), summary)
    _write_csv(paths["channels"], CHANNEL_AUDIT_COLUMNS, channel_rows)
    _write_html(paths["html"], summary, inventory_csv)

    source_hash = hashlib.sha256(inventory_csv.read_bytes()).hexdigest()
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_inventory": str(inventory_csv),
        "source_inventory_sha256": source_hash,
        "scope": list(SITE_ORDER),
        "minimum_original_sampling_frequency_hz": MINIMUM_ORIGINAL_SFREQ_HZ,
        "eligible_recording_states": list(RECORDING_STATES),
        "hup_rule": "include task=interictal and task=ictal; manually review missing/unknown task",
        "ucla_rule": "all eligible recordings are interictal regardless of original task",
        "outcome_rule": (
            "surgical outcome is recorded as metadata only; it never filters the "
            "preprocessing cohort. The seizure-free restriction is applied in the ML stage."
        ),
        "pilot_rule": (
            "first deterministic eligible HUP interictal, HUP ictal, and UCLA interictal "
            "recording by participant, recording index, and EDF path"
        ),
        "channel_label_rules": {
            "high_confidence_positive": "channel soz equals 1",
            "high_confidence_negative": "channel resection equals 0 and participant outcome explicitly indicates seizure-free; SOZ positives take precedence",
            "ambiguous_supervised_label": "all other channels",
        },
        "signals_opened": False,
    }
    paths["protocol"].write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Included %d recordings; excluded %d", len(included), len(excluded))
    return paths


def _detroit_channel_audit(
    recording: dict[str, str], data_root: Path | None
) -> list[dict[str, object]]:
    """Add Detroit usability metadata while retaining the canonical label audit."""
    labels = _channel_audit_rows(recording, data_root)
    if not labels:
        return []
    edf_path = _resolved_edf_path(recording, data_root)
    _, channels = _read_delimited(_channels_path(edf_path), "\t")
    by_name = {str(row["channel_name"]): row for row in labels}
    for index, channel in enumerate(channels):
        name = _first_available(channel, ("name", "channel", "channel_name")) or str(index)
        audit = by_name.get(name)
        if audit is None:
            continue
        channel_type = _first_available(channel, ("type", "channel_type")).upper()
        good = _strict_binary(channel.get("good"))
        # Some BIDS exports use status instead of the non-standard good column.
        status = _clean(channel.get("status")).lower()
        is_bad = status == "bad" or good == 0
        audit["usable_ecog"] = channel_type == "ECOG" and not is_bad and (good in (None, 1))
        audit["ecog_type"] = channel_type == "ECOG"
        audit["bad_channel"] = is_bad
    return labels


def _preprocessed_participants(derivatives_root: Path) -> set[str]:
    """Read derivative sidecars once; EDF/FIF signals are never opened."""
    if not derivatives_root.is_dir():
        raise ValueError(f"preprocessing derivatives root does not exist: {derivatives_root}")
    participants: set[str] = set()
    for path in derivatives_root.rglob("*@Preprocessed1kHzNoBandpass.metadata.json"):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        participant_id = _clean(metadata.get("participant_id"))
        if participant_id:
            participants.add(participant_id)
    return participants


def build_openieeg_detroit_cohort(
    inventory_csv: Path,
    output_dir: Path,
    data_root: Path,
    derivatives_root: Path | None,
) -> dict[str, Path]:
    """Build the outcome-restricted Detroit pre-feature cohort, metadata only.

    This deliberately uses the HUP/UCLA outcome parser and channel-label audit;
    it neither opens EDF signals nor performs feature QC.
    """
    fields, rows = _read_delimited(inventory_csv)
    missing = REQUIRED_INPUT_COLUMNS - set(fields)
    if missing:
        raise ValueError("inventory lacks required columns: " + ", ".join(sorted(missing)))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    preprocessed_participants = _preprocessed_participants(derivatives_root) if derivatives_root else set()

    participant_rows: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        participant_rows.setdefault(_clean(row.get("participant_id")), []).append(row)
    qc: list[dict[str, object]] = []
    channel_audit: list[dict[str, object]] = []
    funnel_names = (
        "Detroit", "sfreq", "seizure-free", "pure ECoG", "channels.tsv",
        "usable ECoG", "SOZ metadata", "resection metadata", "SOZ+",
        "high-confidence negative", ">=1 complete 8-s epoch", "task=sleep", "final cohort",
    )
    funnel = {name: 0 for name in funnel_names}
    eligible: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []

    for participant_id, participant_recordings in sorted(participant_rows.items()):
        row = participant_recordings[0]
        reason = ""
        checks: dict[str, bool] = {}
        detroit = (
            participant_id.lower().startswith("sub-openieegdetroit")
            and all("ucla" not in _clean(recording.get("participant_id")).lower() for recording in participant_recordings)
        )
        checks["Detroit"] = detroit
        sfreq = all((_parse_sfreq(recording.get("original_sampling_frequency_hz")) or -1) >= 1000 for recording in participant_recordings)
        checks["sfreq"] = detroit and sfreq
        participant_path = _participants_path(_resolved_edf_path(row, data_root), data_root)
        participant = _participant_row(participant_path, participant_id) if participant_path else None
        outcome = _source_outcome(row, participant)
        seizure_free = _is_seizure_free(outcome)
        checks["seizure-free"] = checks["sfreq"] and seizure_free
        audits = [audit for recording in participant_recordings for audit in _detroit_channel_audit(recording, data_root)]
        channel_audit.extend(audits)
        types = {_clean(audit.get("channel_type")).upper() for audit in audits}
        pure_ecog = bool(types) and types == {"ECOG"}
        checks["pure ECoG"] = checks["seizure-free"] and pure_ecog
        has_channels = all(_channels_path(_resolved_edf_path(recording, data_root)).is_file() for recording in participant_recordings)
        checks["channels.tsv"] = checks["pure ECoG"] and has_channels
        usable = [audit for audit in audits if bool(audit.get("usable_ecog"))]
        checks["usable ECoG"] = checks["channels.tsv"] and bool(usable)
        has_soz = all(_strict_binary(recording.get("has_soz_metadata")) == 1 for recording in participant_recordings)
        checks["SOZ metadata"] = checks["usable ECoG"] and has_soz
        has_resection = all(_strict_binary(recording.get("has_resection_metadata")) == 1 for recording in participant_recordings)
        checks["resection metadata"] = checks["SOZ metadata"] and has_resection
        positives = [audit for audit in usable if bool(audit.get("high_confidence_positive"))]
        checks["SOZ+"] = checks["resection metadata"] and bool(positives)
        negatives = [audit for audit in usable if bool(audit.get("high_confidence_negative"))]
        checks["high-confidence negative"] = checks["SOZ+"] and bool(negatives)
        complete_epoch = any((_parse_sfreq(recording.get("duration_seconds")) or 0) >= 8 for recording in participant_recordings)
        checks[">=1 complete 8-s epoch"] = checks["high-confidence negative"] and complete_epoch
        sleep = all(_clean(recording.get("task")).lower() == "sleep" for recording in participant_recordings)
        checks["task=sleep"] = checks[">=1 complete 8-s epoch"] and sleep
        has_derivative = participant_id in preprocessed_participants
        checks["final cohort"] = checks["task=sleep"] and has_derivative
        for name in funnel_names:
            funnel[name] += int(checks[name])
        for name in funnel_names:
            if not checks[name]:
                reason = {
                    "Detroit": "not_openieeg_detroit",
                    "sfreq": "original_sampling_frequency_below_1000_hz",
                    "seizure-free": "not_seizure_free_or_missing_outcome",
                    "pure ECoG": "not_pure_clinical_ecog",
                    "channels.tsv": "missing_channels_tsv",
                    "usable ECoG": "no_usable_ecog_channel",
                    "SOZ metadata": "missing_soz_metadata",
                    "resection metadata": "missing_resection_metadata",
                    "SOZ+": "no_usable_soz_positive_channel",
                    "high-confidence negative": "no_usable_high_confidence_negative_channel",
                    ">=1 complete 8-s epoch": "no_complete_8_second_epoch",
                    "task=sleep": "task_not_sleep",
                    "final cohort": "missing_successful_preprocessing_derivative",
                }[name]
                break
        qc_row = {
            "participant_id": participant_id, "dataset": _clean(row.get("dataset")),
            "institution": _clean(row.get("institution")), "surgical_outcome": outcome,
            "number_of_recordings": len(participant_recordings), "number_of_channels": len(audits),
            "number_of_usable_ecog_channels": len(usable), "number_of_soz_positive_channels": len(positives),
            "number_of_high_confidence_negative_channels": len(negatives),
            "has_successful_preprocessing_derivative": has_derivative,
            "detroit_interictal_condition": "sleep" if sleep else "",
            **{f"passes_{name.replace(' ', '_').replace('>=', 'ge_').replace('+', 'positive').replace('-', '_').replace('8-s', '8s').replace('=', '_')}": value for name, value in checks.items()},
            "exclusion_reason": reason,
        }
        qc.append(qc_row)
        if reason:
            exclusions.append(qc_row)
        else:
            eligible.append({**row, "recording_state": "interictal", "recording_state_source": "dataset_rule_openieeg_detroit_sleep_interictal", "detroit_interictal_condition": "sleep", "surgical_outcome": outcome})

    if len({row["participant_id"] for row in eligible}) != len(eligible):
        raise ValueError("duplicate participants in final Detroit cohort")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: output_dir / value for key, value in DETROIT_OUTPUT_NAMES.items()}
    _write_csv(paths["cohort"], list(dict.fromkeys([*fields, "recording_state", "recording_state_source", "detroit_interictal_condition", "surgical_outcome"])), eligible)
    _write_csv(paths["excluded"], qc[0].keys() if qc else ("participant_id", "exclusion_reason"), exclusions)
    _write_csv(paths["qc"], qc[0].keys() if qc else ("participant_id",), qc)
    _write_csv(paths["channels"], [*CHANNEL_AUDIT_COLUMNS, "usable_ecog", "ecog_type", "bad_channel"], channel_audit)
    _write_csv(paths["funnel"], ("stage", "participants"), ({"stage": name, "participants": funnel[name]} for name in funnel_names))
    _write_json(paths["summary"], {"protocol": "openieeg-detroit-pre-feature-v1", "source_inventory": str(inventory_csv), "data_root": str(data_root), "derivatives_root": str(derivatives_root) if derivatives_root else None, "signals_opened": False, "funnel": funnel, "final_participants": len(eligible), "final_recordings": len(eligible), "usable_ecog_channels": sum(row["number_of_usable_ecog_channels"] for row in qc if not row["exclusion_reason"]), "soz_positive_channels": sum(row["number_of_soz_positive_channels"] for row in qc if not row["exclusion_reason"]), "high_confidence_negative_channels": sum(row["number_of_high_confidence_negative_channels"] for row in qc if not row["exclusion_reason"]), "task_distribution": _count_values(row.get("task") for row in eligible), "outcome_distribution": _count_values(row.get("surgical_outcome") for row in eligible), "modality_distribution": _count_values(row.get("channel_type") for row in channel_audit if bool(row.get("usable_ecog")))})
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("hup-ucla", "openieeg-detroit"), default="hup-ucla"
    )
    parser.add_argument("--inventory-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--derivatives-root", type=Path,
        help="Required for the OpenIEEG Detroit profile to verify preprocessing artifacts.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.profile == "openieeg-detroit":
        if args.data_root is None or args.derivatives_root is None:
            raise ValueError("OpenIEEG Detroit requires --data-root and --derivatives-root")
        build_openieeg_detroit_cohort(
            args.inventory_csv.expanduser(), args.output_dir.expanduser(),
            args.data_root.expanduser(), args.derivatives_root.expanduser(),
        )
    else:
        build_cohort(
            args.inventory_csv.expanduser(), args.output_dir.expanduser(),
            args.data_root.expanduser() if args.data_root else None,
        )


if __name__ == "__main__":
    main()
