#!/usr/bin/env python3
"""Freeze one deterministic FIF manifest shared by feature workflows.

The historical HUP/UCLA command-line interface is retained.  ``build_manifest``
also accepts a single-site Detroit cohort, using the identical family topology
without mixing its artifacts with HUP/UCLA artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
EXTRACTION_SCRIPT = REPOSITORY / "scripts" / "features" / "relative_bandpower.py"
MODULE_NAME = "omni_hup_ucla_feature_manifest_reference"
SITE_ORDER = ("hup", "ucla")
# Every feature family uses the same NeuroDAGs architecture: the datasets
# configuration is generated here, beside the frozen manifest it indexes, so a
# family can never be pointed at a manifest it was not frozen against.
FEATURE_FAMILIES = ("relative_bandpower", "time_domain", "frequency_domain", "complexity")


def datasets_config_name(family: str, cohort_name: str = "hup_ucla") -> str:
    if family not in FEATURE_FAMILIES:
        raise ValueError(f"unknown feature family {family!r}; expected {list(FEATURE_FAMILIES)}")
    return f"datasets_{cohort_name}_{family}.yml"


def _core_module():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, EXTRACTION_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared feature infrastructure: {EXTRACTION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _yaml_path(path: Path) -> str:
    return json.dumps(str(path))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_datasets_config(
    path: Path,
    output_directory: Path,
    derivatives_path: Path,
    channel_labels_csv: Path,
    sites: tuple[str, ...],
) -> None:
    lines = [
        "# Generated from the frozen feature cohort. Do not edit or reorder.",
    ]
    for site in sites:
        lines.extend(
            [
                f"{site}:",
                f"  name: {site}",
                "  skip: False",
                "  file_pattern: "
                + _yaml_path(output_directory / "indexed_inputs" / site / "*.fif"),
                "  derivatives_path: " + _yaml_path(derivatives_path / site),
                "  vars:",
                "    channel_labels_csv: " + _yaml_path(channel_labels_csv),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_manifest(
    cohort_csv: Path,
    preprocessed_root: Path | list[Path],
    output_directory: Path,
    derivatives_paths: dict[str, Path],
    channel_labels_csv: Path,
    *,
    sites: tuple[str, ...] = SITE_ORDER,
    cohort_name: str = "hup_ucla",
) -> tuple[Path, dict[str, Path]]:
    """Build one shared FIF index and every feature family's NeuroDAGs config.

    ``preprocessed_root`` may be a list so a new analysis root reuses
    derivatives an earlier root already produced. The first root that supplies a
    given recording wins, so the current analysis root must be listed first and
    a read-only reuse root last.

    ``derivatives_paths`` maps every entry of :data:`FEATURE_FAMILIES` to that
    family's derivatives root for this run (pilot roots for a pilot manifest,
    full roots for a full manifest), keeping pilot and full outputs disjoint.
    """
    if not sites or len(set(sites)) != len(sites):
        raise ValueError(f"sites must be a non-empty unique sequence, got {sites}")
    if not cohort_name or not cohort_name.replace("_", "").isalnum():
        raise ValueError(f"invalid cohort name: {cohort_name!r}")
    missing_families = sorted(set(FEATURE_FAMILIES) - set(derivatives_paths))
    unknown_families = sorted(set(derivatives_paths) - set(FEATURE_FAMILIES))
    if missing_families or unknown_families:
        raise ValueError(
            "derivatives_paths must cover exactly "
            f"{list(FEATURE_FAMILIES)}; missing={missing_families}, unknown={unknown_families}"
        )
    distinct = {str(Path(path)) for path in derivatives_paths.values()}
    if len(distinct) != len(FEATURE_FAMILIES):
        raise ValueError(
            f"each feature family needs its own derivatives root; got {sorted(distinct)}"
        )
    if output_directory.exists():
        raise FileExistsError(
            f"refusing to alter existing frozen manifest directory: {output_directory}"
        )
    with cohort_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        cohort = list(csv.DictReader(stream))
    if not cohort:
        raise ValueError(f"cohort CSV is empty: {cohort_csv}")

    roots = (
        [preprocessed_root]
        if isinstance(preprocessed_root, (str, Path))
        else list(preprocessed_root)
    )
    core = _core_module()
    artifacts = core.index_preprocessed_artifacts(roots)
    if not artifacts:
        raise FileNotFoundError(f"no preprocessing metadata below {roots}")

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent)
    )
    try:
        resolved: list[tuple[str, dict[str, str], Path, dict[str, object]]] = []
        for cohort_row in cohort:
            fif_path, _, metadata = core.resolve_preprocessed_artifacts(
                cohort_row, roots, artifacts
            )
            site = str(cohort_row.get("dataset", "")).strip().lower()
            if site not in sites:
                raise ValueError(f"unsupported cohort dataset: {site!r}")
            # The manifest is the audit trail for which array index is ictal, so
            # both the raw BIDS task and the canonical recording state must be
            # present. A missing value is a cohort defect, never a default.
            for column in ("task", "recording_state"):
                if not str(cohort_row.get(column, "")).strip():
                    raise ValueError(
                        f"cohort row for {site}/{cohort_row.get('participant_id')}/"
                        f"{cohort_row.get('recording_id')} has no {column}"
                    )
            resolved.append((site, cohort_row, fif_path, metadata))

        # Match each datasets configuration's glob expansion and sorting exactly.
        resolved.sort(key=lambda item: (sites.index(item[0]), item[2].name))
        rows: list[dict[str, str]] = []
        cohort_hash = _sha256(cohort_csv)
        for index, (site, cohort_row, fif_path, metadata) in enumerate(resolved):
            link_directory = temporary / "indexed_inputs" / site
            link_directory.mkdir(parents=True, exist_ok=True)
            link = link_directory / f"index-{index:06d}__{fif_path.name}"
            link.symlink_to(fif_path)
            rows.append(
                {
                    "neurodags_index": str(index),
                    "indexed_fif_path": str(output_directory / link.relative_to(temporary)),
                    "source_fif_path": str(fif_path),
                    "participant_id": str(cohort_row.get("participant_id", "")),
                    "dataset": site,
                    "recording_id": str(cohort_row.get("recording_id", "")),
                    "task": str(cohort_row.get("task", "")).strip(),
                    "recording_state": str(cohort_row.get("recording_state", "")).strip().lower(),
                    "cohort_csv": str(cohort_csv),
                    "cohort_csv_sha256": cohort_hash,
                    "preprocessing_protocol_version": str(
                        metadata.get("protocol_version", "")
                    ),
                }
            )

        index_path = temporary / f"{cohort_name}_feature_neurodags_index.csv"
        _write_csv(index_path, rows)
        for family in FEATURE_FAMILIES:
            _write_datasets_config(
                temporary / datasets_config_name(family, cohort_name),
                output_directory,
                Path(derivatives_paths[family]),
                channel_labels_csv,
                sites,
            )
        temporary.rename(output_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return (
        output_directory / f"{cohort_name}_feature_neurodags_index.csv",
        {
            family: output_directory / datasets_config_name(family, cohort_name)
            for family in FEATURE_FAMILIES
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        action="append",
        required=True,
        help=(
            "preprocessing root to index; repeat to reuse derivatives from a "
            "previous analysis root without recomputing them"
        ),
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    for family in FEATURE_FAMILIES:
        parser.add_argument(
            f"--{family.replace('_', '-')}-derivatives-path",
            type=Path,
            required=True,
            help=f"{family} derivatives root for this manifest (pilot or full)",
        )
    parser.add_argument("--channel-labels-csv", type=Path, required=True)
    args = parser.parse_args()
    derivatives_paths = {
        family: getattr(args, f"{family}_derivatives_path") for family in FEATURE_FAMILIES
    }
    index, configs = build_manifest(
        args.cohort_csv.resolve(),
        [path.resolve() for path in args.preprocessed_root],
        args.output_directory,
        derivatives_paths,
        args.channel_labels_csv.resolve(),
    )
    print(f"Shared HUP/UCLA feature index: {index}")
    for family in FEATURE_FAMILIES:
        print(f"{family} NeuroDAGs config: {configs[family]}")


if __name__ == "__main__":
    main()
