#!/usr/bin/env python3
"""Create small deterministic HUP/UCLA fixtures for the 1-kHz workflow."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mne
import numpy as np


FIXTURE_DEFINITIONS = (
    {
        "participant_id": "sub-hupfixture",
        "dataset": "hup",
        "institution": "HUP synthetic",
        "session": "01",
        "task": "interictal",
        "run": "01",
        "sampling_frequency": 1000,
        "duration_seconds": 16,
        "line_frequencies": (60.0, 180.0),
        "eligible": True,
    },
    {
        "participant_id": "sub-uclafixture",
        "dataset": "ucla",
        "institution": "UCLA synthetic",
        "session": "01",
        "task": "interictal",
        "run": "01",
        "sampling_frequency": 1024,
        "duration_seconds": 20,
        "line_frequencies": (60.0, 180.0),
        "eligible": True,
    },
    {
        "participant_id": "sub-hupineligible",
        "dataset": "hup",
        "institution": "HUP synthetic",
        "session": "01",
        "task": "interictal",
        "run": "01",
        "sampling_frequency": 500,
        "duration_seconds": 16,
        "line_frequencies": (60.0,),
        "eligible": False,
    },
)


def _write_csv(
    path: Path, rows: list[dict[str, object]], *, delimiter: str = ","
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def build_fixture(root: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for recording_index, definition in enumerate(FIXTURE_DEFINITIONS):
        participant = str(definition["participant_id"])
        site = str(definition["dataset"])
        session = str(definition["session"])
        task = str(definition["task"])
        run = str(definition["run"])
        sfreq = int(definition["sampling_frequency"])
        duration = int(definition["duration_seconds"])
        directory = root / participant / f"ses-{session}" / "ieeg"
        directory.mkdir(parents=True, exist_ok=True)
        base = f"{participant}_ses-{session}_task-{task}_run-{run}_ieeg"
        signal_path = directory / f"{base}.fif"
        time = np.arange(sfreq * duration, dtype=float) / sfreq
        rng = np.random.default_rng(1000 + recording_index)
        data = np.zeros((4, time.size), dtype=float)
        for channel in range(3):
            data[channel] = 1e-5 * np.sin(2 * np.pi * 10.0 * time)
            for line_frequency in definition["line_frequencies"]:
                data[channel] += 5e-5 * np.sin(2 * np.pi * line_frequency * time)
            data[channel] += 1e-7 * rng.standard_normal(time.size)
        data[3] = 1e-4 * np.sin(2 * np.pi * 1.0 * time)
        info = mne.create_info(
            ["A1", "A2", "A3", "ECG"], sfreq, ["seeg", "seeg", "seeg", "ecg"]
        )
        raw = mne.io.RawArray(data, info, verbose="ERROR")
        raw.save(signal_path, overwrite=True, verbose="ERROR")
        channel_rows = [
            {
                "name": "A1",
                "type": "SEEG",
                "status": "good",
                "good": 1,
                "soz": 1,
                "resection": 1,
            },
            {
                "name": "A2",
                "type": "SEEG",
                "status": "good",
                "good": 1,
                "soz": 0,
                "resection": 0,
            },
            {
                "name": "A3",
                "type": "SEEG",
                "status": "bad",
                "good": 0,
                "soz": 0,
                "resection": 0,
            },
            {
                "name": "ECG",
                "type": "ECG",
                "status": "good",
                "good": 1,
                "soz": "n/a",
                "resection": "n/a",
            },
        ]
        _write_csv(
            directory / f"{base.removesuffix('_ieeg')}_channels.tsv",
            channel_rows,
            delimiter="\t",
        )
        (directory / f"{base}.json").write_text(
            json.dumps({"InstitutionName": definition["institution"], "TaskName": task})
            + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "recording_index": recording_index,
                "participant_id": participant,
                "dataset": site,
                "institution": definition["institution"],
                "session": session,
                "task": task,
                "run": run,
                "edf_path": str(signal_path),
                "original_sampling_frequency_hz": sfreq,
                "eligible_sfreq_ge_1000": definition["eligible"],
                "duration_seconds": duration,
                "has_channels_tsv": True,
                "has_soz_metadata": True,
                "has_resection_metadata": True,
                "has_outcome_metadata": True,
                "soz": "A1",
                "resection": "A1",
                "outcome": "ILAE 1",
            }
        )
    inventory_path = root / "synthetic_recording_inventory.csv"
    eligible_path = root / "synthetic_eligible_recordings.csv"
    _write_csv(inventory_path, rows)
    _write_csv(eligible_path, [row for row in rows if bool(row["eligible_sfreq_ge_1000"])])
    _write_csv(
        root / "participants.tsv",
        [
            {
                "participant_id": row["participant_id"],
                "outcome": row["outcome"],
                "clinical_group": "synthetic_epilepsy",
            }
            for row in rows
        ],
        delimiter="\t",
    )
    return inventory_path, eligible_path, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inventory, eligible, _ = build_fixture(args.output_dir)
    print(f"Synthetic inventory: {inventory}")
    print(f"Synthetic eligible manifest: {eligible}")


if __name__ == "__main__":
    main()
