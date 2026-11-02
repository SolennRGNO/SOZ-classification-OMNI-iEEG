#!/usr/bin/env python3
"""Build a reproducible NeuroDAGs file index from the verified eligible CSV."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


SITE_ORDER = ("hup", "ucla")
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
OUTPUT_NAMES = {
    "full": ("eligible_index_manifest_{version}.csv", "datasets_full_{version}.yml"),
    "pilots": ("pilot_manifest_{version}.csv", "datasets_pilots_{version}.yml"),
    "two-subjects": (
        "two_subjects_manifest_{version}.csv",
        "datasets_two_subjects_{version}.yml",
    ),
}


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _source_path(row: dict[str, str], data_root: Path | None) -> Path:
    path = Path(row["edf_path"]).expanduser()
    if not path.is_absolute():
        if data_root is None:
            raise ValueError(f"relative EDF path requires --data-root: {path}")
        path = data_root / path
    return path.resolve()


def load_verified_eligible_rows(
    path: Path, data_root: Path | None = None
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "recording_index",
            "participant_id",
            "dataset",
            "edf_path",
            "original_sampling_frequency_hz",
            "eligible_sfreq_ge_1000",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("eligible manifest lacks columns: " + ", ".join(sorted(missing)))
        rows = []
        for raw_row in reader:
            row = {str(key): str(value or "") for key, value in raw_row.items()}
            site = row["dataset"].strip().lower()
            if site not in SITE_ORDER:
                continue
            try:
                sampling_frequency = float(row["original_sampling_frequency_hz"])
            except ValueError as error:
                raise ValueError(
                    f"malformed original sampling frequency for {row['edf_path']!r}"
                ) from error
            if not _truthy(row["eligible_sfreq_ge_1000"]) or sampling_frequency < 1000:
                raise ValueError(
                    "eligible manifest contains an ineligible HUP/UCLA row: "
                    + row["edf_path"]
                )
            row["dataset"] = site
            row["source_edf_path"] = str(_source_path(row, data_root))
            rows.append(row)
    if not rows:
        raise ValueError("eligible manifest contains no HUP/UCLA recordings")
    if {row["dataset"] for row in rows} != set(SITE_ORDER):
        raise ValueError("eligible manifest must contain both hup and ucla")
    paths = [row["source_edf_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("eligible manifest contains duplicated EDF paths")
    return rows


def _stable_row_key(row: dict[str, str]) -> tuple[str, int, str]:
    try:
        index = int(float(row.get("recording_index", "")))
    except ValueError:
        index = 2**63 - 1
    return (row.get("participant_id", ""), index, row["source_edf_path"])


def _pilot_score(row: dict[str, str]) -> tuple[int, int, int, int, int, float]:
    task = row.get("task", "").lower()
    interictal = int("interictal" in task or "rest" in task)
    channels = int(_truthy(row.get("has_channels_tsv", "")))
    soz = int(_truthy(row.get("has_soz_metadata", "")))
    resection = int(_truthy(row.get("has_resection_metadata", "")))
    outcome = int(_truthy(row.get("has_outcome_metadata", "")))
    try:
        duration = float(row.get("duration_seconds", 0) or 0)
    except ValueError:
        duration = 0.0
    return (interictal, channels, soz, resection, outcome, duration)


def select_rows(rows: list[dict[str, str]], selection: str) -> list[dict[str, str]]:
    grouped = {
        site: sorted((row for row in rows if row["dataset"] == site), key=_stable_row_key)
        for site in SITE_ORDER
    }
    if selection == "full":
        return [row for site in SITE_ORDER for row in grouped[site]]
    if selection == "pilots":
        selected = []
        for site in SITE_ORDER:
            site_rows = grouped[site]
            if not site_rows:
                raise ValueError(f"no eligible pilot candidate for {site}")
            selected.append(max(site_rows, key=_pilot_score))
        return selected
    if selection == "two-subjects":
        selected = []
        for site in SITE_ORDER:
            participants = sorted({row["participant_id"] for row in grouped[site]})
            if not participants:
                raise ValueError(f"no eligible subject for {site}")
            participant = participants[0]
            selected.extend(
                row for row in grouped[site] if row["participant_id"] == participant
            )
        return selected
    raise ValueError(f"unsupported selection: {selection}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signal_suffix(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".fif.gz"):
        return ".fif.gz"
    return path.suffix.lower()


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _yaml_string(value: str | Path) -> str:
    return json.dumps(str(value))


def _write_datasets_yaml(
    path: Path,
    *,
    final_directory: Path,
    derivatives_path: Path,
    manifest_path: Path,
) -> None:
    lines = [
        "# Generated from a verified eligible manifest. Do not reorder entries.",
        "# NeuroDAGs indices are frozen in the adjacent versioned CSV.",
    ]
    for site in SITE_ORDER:
        lines.extend(
            [
                f"{site}:",
                f"  name: {site}",
                "  skip: False",
                "  file_pattern: "
                + _yaml_string(final_directory / "indexed_inputs" / site / "*_ieeg.*"),
                "  derivatives_path: " + _yaml_string(derivatives_path / site),
                "  vars:",
                f"    site: {site}",
                f"    dataset: {site}",
                "    eligible_manifest_path: " + _yaml_string(manifest_path),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _actual_indexed_paths(index_root: Path) -> list[str]:
    """Return indexed links in a deterministic, filesystem-order-independent order.

    ``glob.glob`` returns directory entries in filesystem order, which changes
    whenever entries are recreated or reordered. The persisted
    ``neurodags_index`` is derived from this list, so it must be sorted: link
    names are ``index-<ordinal>__<name>``, therefore sorting within a site
    reproduces the ordinal order exactly.
    """
    paths = []
    for site in SITE_ORDER:
        paths.extend(sorted(glob.glob(str(index_root / site / "*_ieeg.*"))))
    return paths


def _ordinal_from_link(path: str) -> int | None:
    """Ordinal encoded in an ``index-<ordinal>__<name>`` link name, if present."""
    match = re.match(r"^index-(\d+)__", Path(path).name)
    return int(match.group(1)) if match else None


def _verify_existing(
    output_directory: Path,
    manifest_name: str,
    datasets_name: str,
    source_hash: str,
) -> tuple[Path, Path]:
    manifest_path = output_directory / manifest_name
    datasets_path = output_directory / datasets_name
    if not manifest_path.is_file() or not datasets_path.is_file():
        raise ValueError(f"incomplete existing versioned index: {output_directory}")
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row.get("source_manifest_sha256") != source_hash for row in rows):
        raise ValueError("existing index was built from a different eligible manifest")
    actual = [
        str(Path(path).resolve(strict=False))
        for path in _actual_indexed_paths(output_directory / "indexed_inputs")
    ]
    recorded = [str(Path(row["indexed_edf_path"]).resolve(strict=False)) for row in rows]
    if actual != recorded or [int(row["neurodags_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("existing NeuroDAGs index order differs from its versioned manifest")
    return manifest_path, datasets_path


def build_index(
    *,
    eligible_manifest: Path,
    output_directory: Path,
    derivatives_path: Path,
    selection: str,
    version: str,
    data_root: Path | None = None,
) -> tuple[Path, Path]:
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"invalid manifest version: {version!r}")
    manifest_template, datasets_template = OUTPUT_NAMES[selection]
    manifest_name = manifest_template.format(version=version)
    datasets_name = datasets_template.format(version=version)
    source_hash = _sha256(eligible_manifest)
    if output_directory.exists():
        return _verify_existing(
            output_directory, manifest_name, datasets_name, source_hash
        )

    rows = select_rows(load_verified_eligible_rows(eligible_manifest, data_root), selection)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent)
    )
    try:
        link_to_row: dict[str, dict[str, str]] = {}
        for ordinal, row in enumerate(rows):
            site_directory = temporary / "indexed_inputs" / row["dataset"]
            site_directory.mkdir(parents=True, exist_ok=True)
            source = Path(row["source_edf_path"])
            if not source.is_file():
                raise FileNotFoundError(f"eligible EDF does not exist: {source}")
            suffix = _signal_suffix(source)
            base = source.name[: -len(suffix)]
            link = site_directory / f"index-{ordinal:06d}__{base}{suffix}"
            link.symlink_to(source)
            final_link = output_directory / link.relative_to(temporary)
            link_to_row[str(final_link)] = row

        final_manifest = output_directory / manifest_name
        actual_temporary = _actual_indexed_paths(temporary / "indexed_inputs")
        ordinals = [_ordinal_from_link(path) for path in actual_temporary]
        if ordinals != list(range(len(actual_temporary))):
            raise ValueError(
                "indexed link order does not match the ordinals encoded in the link "
                f"names; refusing to persist a non-deterministic index: {ordinals[:10]}"
            )
        indexed_rows: list[dict[str, str]] = []
        for index, temporary_link_text in enumerate(actual_temporary):
            temporary_link = Path(temporary_link_text)
            final_link = output_directory / temporary_link.relative_to(temporary)
            row = dict(link_to_row[str(final_link)])
            row.update(
                {
                    "neurodags_index": str(index),
                    "indexed_edf_path": str(final_link),
                    "index_manifest_version": version,
                    "index_selection": selection,
                    "source_manifest_path": str(eligible_manifest),
                    "source_manifest_sha256": source_hash,
                }
            )
            indexed_rows.append(row)
        temporary_manifest = temporary / manifest_name
        _write_csv(temporary_manifest, indexed_rows)
        _write_datasets_yaml(
            temporary / datasets_name,
            final_directory=output_directory,
            derivatives_path=derivatives_path,
            manifest_path=final_manifest,
        )
        if selection == "two-subjects":
            task_rows = []
            for task_id, site in enumerate(SITE_ORDER):
                site_rows = [row for row in indexed_rows if row["dataset"] == site]
                participants = {row["participant_id"] for row in site_rows}
                if len(participants) != 1:
                    raise ValueError(f"two-subject selection is not one subject for {site}")
                task_rows.append(
                    {
                        "array_task_id": str(task_id),
                        "site": site,
                        "participant_id": next(iter(participants)),
                        "neurodags_indices": ";".join(
                            row["neurodags_index"] for row in site_rows
                        ),
                    }
                )
            _write_csv(temporary / f"subject_tasks_{version}.csv", task_rows)
        temporary.rename(output_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_directory / manifest_name, output_directory / datasets_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligible-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--derivatives-path", type=Path, required=True)
    parser.add_argument("--selection", choices=tuple(OUTPUT_NAMES), required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--data-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path, datasets_path = build_index(
        eligible_manifest=args.eligible_manifest.expanduser().resolve(),
        output_directory=args.output_directory.expanduser(),
        derivatives_path=args.derivatives_path.expanduser(),
        selection=args.selection,
        version=args.version,
        data_root=args.data_root.expanduser() if args.data_root else None,
    )
    print(f"Versioned index manifest: {manifest_path}")
    print(f"NeuroDAGs datasets configuration: {datasets_path}")


if __name__ == "__main__":
    main()
