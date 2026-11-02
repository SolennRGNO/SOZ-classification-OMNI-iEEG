#!/usr/bin/env python3
"""Resolve one frozen-manifest row into a single-recording NeuroDAGs config.

The frozen feature manifest is the only source of truth for the
``SLURM_ARRAY_TASK_ID -> recording`` mapping.

NeuroDAGs expands ``file_pattern`` with an unsorted ``glob.glob``
(``neurodags/iterators.py``) and then numbers the result positionally, so a
directory glob combined with ``--only-index N`` does **not** reliably select
manifest row ``N``: the run-time order is filesystem order, which changes when
directory entries are recreated or reordered.

Every array task therefore resolves its exact recording here, from the frozen
manifest, and runs NeuroDAGs against a configuration that contains exactly one
recording with ``--only-index 0``. This is the same one-recording pattern the
preprocessing scale-out already uses in
``scripts/preprocessing/build_subject_manifests.py``.

The generated configuration copies the site's stanza from the frozen/checked-in
datasets configuration verbatim and overrides only two keys, so the derivative
output paths are byte-identical to those produced by the directory glob:

* ``file_pattern``    -> the manifest row's ``indexed_fif_path``
* ``derivatives_path`` -> ``<original derivatives_path>/<indexed FIF basename>``

The second override is required because NeuroDAGs derives an artifact's name
from ``os.path.relpath(file_path, common_root)``, and ``common_root`` for a
single-file dataset is the file itself (relative path ``"."``). Appending the
basename to ``derivatives_path`` restores the exact reference base the
multi-file glob produced.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


INDEX_COLUMN = "neurodags_index"
PATH_COLUMN = "indexed_fif_path"
SITE_COLUMN = "dataset"
# Passed through to the NeuroDAGs node so every epoch row carries the task and
# the canonical recording state straight from the frozen manifest.
RECORDING_VARS = ("task", "recording_state")
REQUIRED_COLUMNS = (INDEX_COLUMN, PATH_COLUMN, SITE_COLUMN, *RECORDING_VARS)


def read_manifest(index_manifest: Path) -> list[dict[str, str]]:
    """Return the frozen manifest rows ordered by ``neurodags_index``.

    The index column is validated to be exactly ``0..len(rows) - 1`` so a row
    can never be addressed by a position that disagrees with its recorded
    index.
    """
    with index_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"frozen manifest is empty: {index_manifest}")
    missing = sorted(set(REQUIRED_COLUMNS) - set(rows[0]))
    if missing:
        raise ValueError(f"frozen manifest lacks required columns {missing}: {index_manifest}")
    indices: list[int] = []
    for row in rows:
        raw = str(row.get(INDEX_COLUMN, "")).strip()
        try:
            indices.append(int(raw))
        except ValueError as error:
            raise ValueError(
                f"frozen manifest has a non-integer {INDEX_COLUMN} {raw!r}: {index_manifest}"
            ) from error
    if sorted(indices) != list(range(len(rows))):
        raise ValueError(
            f"frozen manifest {INDEX_COLUMN} values are not exactly 0..{len(rows) - 1} "
            f"(duplicated or missing indices): {index_manifest}"
        )
    return [row for _, row in sorted(zip(indices, rows), key=lambda item: item[0])]


def manifest_length(index_manifest: Path) -> int:
    """Number of recordings the frozen manifest defines."""
    return len(read_manifest(index_manifest))


def resolve_row(index_manifest: Path, index: int) -> dict[str, str]:
    """Return the manifest row whose ``neurodags_index`` equals ``index``.

    Raises loudly when the index is outside the frozen manifest; NeuroDAGs
    itself only logs a warning for an unknown ``--only-index`` and then exits
    successfully having processed nothing.
    """
    rows = read_manifest(index_manifest)
    if index < 0 or index >= len(rows):
        raise IndexError(
            f"array index {index} is outside the frozen manifest "
            f"(valid range 0..{len(rows) - 1}): {index_manifest}"
        )
    row = rows[index]
    if int(str(row[INDEX_COLUMN]).strip()) != index:
        raise AssertionError(f"frozen manifest row {index} does not carry index {index}")
    if not str(row.get(PATH_COLUMN, "")).strip():
        raise ValueError(f"frozen manifest row {index} has an empty {PATH_COLUMN}")
    for name in RECORDING_VARS:
        if not str(row.get(name, "")).strip():
            raise ValueError(f"frozen manifest row {index} has an empty {name}")
    return row


def _replace_scalar(original: Any, value: str) -> Any:
    """Set ``value`` while preserving a mount-point mapping shape if present."""
    if isinstance(original, dict):
        return {key: value for key in original}
    return value


def single_recording_datasets_config(
    datasets_config: Path, row: dict[str, str]
) -> dict[str, Any]:
    """Build a datasets configuration holding exactly the manifest row's FIF."""
    import yaml  # deferred so --print-count stays standard-library only

    loaded = yaml.safe_load(datasets_config.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"datasets configuration is not a mapping: {datasets_config}")
    site = str(row[SITE_COLUMN]).strip().lower()
    stanza = next(
        (value for key, value in loaded.items() if str(key).strip().lower() == site),
        None,
    )
    if not isinstance(stanza, dict):
        raise ValueError(f"datasets configuration has no {site!r} dataset: {datasets_config}")
    fif_path = Path(str(row[PATH_COLUMN]).strip())
    if "derivatives_path" not in stanza:
        raise ValueError(
            f"{site!r} dataset has no derivatives_path: {datasets_config}"
        )
    original_derivatives = stanza["derivatives_path"]
    if isinstance(original_derivatives, dict):
        derivatives = {
            key: str(Path(str(value)) / fif_path.name)
            for key, value in original_derivatives.items()
        }
    else:
        derivatives = str(Path(str(original_derivatives)) / fif_path.name)
    single = dict(stanza)
    single["file_pattern"] = _replace_scalar(stanza.get("file_pattern"), str(fif_path))
    single["derivatives_path"] = derivatives
    variables = dict(stanza.get("vars") or {})
    for name in RECORDING_VARS:
        value = str(row.get(name, "")).strip()
        if not value:
            raise ValueError(f"frozen manifest row has no {name}: {row.get(PATH_COLUMN)}")
        variables[name] = value
    single["vars"] = variables
    return {site: single}


def write_single_recording_config(
    index_manifest: Path, datasets_config: Path, index: int, output: Path
) -> tuple[Path, Path]:
    """Write the one-recording configuration and return (config, resolved FIF)."""
    import yaml

    row = resolve_row(index_manifest, index)
    fif_path = Path(str(row[PATH_COLUMN]).strip())
    if not fif_path.exists():
        raise FileNotFoundError(
            f"frozen manifest row {index} points at a missing FIF: {fif_path}"
        )
    config = single_recording_datasets_config(datasets_config, row)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated per array task from the frozen manifest. One recording only;\n"
        "# run NeuroDAGs against this file with --only-index 0.\n"
        f"# frozen manifest: {index_manifest}\n"
        f"# {INDEX_COLUMN}: {index}\n"
    )
    output.write_text(
        header + yaml.safe_dump(config, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return output, fif_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--datasets-config", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--print-count",
        action="store_true",
        help="print the number of frozen-manifest recordings and exit",
    )
    parser.add_argument(
        "--print-row",
        action="store_true",
        help="print the resolved manifest row as JSON on stderr",
    )
    args = parser.parse_args()

    if args.print_count:
        print(manifest_length(args.index_manifest))
        return
    if args.datasets_config is None or args.index is None or args.output is None:
        parser.error("--datasets-config, --index and --output are required")
    config, fif_path = write_single_recording_config(
        args.index_manifest, args.datasets_config, args.index, args.output
    )
    if args.print_row:
        print(
            json.dumps(resolve_row(args.index_manifest, args.index), sort_keys=True),
            file=sys.stderr,
        )
    print(f"one-recording datasets configuration: {config}", file=sys.stderr)
    print(fif_path)


if __name__ == "__main__":
    main()
