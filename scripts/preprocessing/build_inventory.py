#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import html
import logging
import re
import shlex
from pathlib import Path

import mne
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("hup_ucla_sampling_inventory")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASETS_CONFIG = REPOSITORY_ROOT / "Neurodags" / "datasets2.yml"
ALLOWED_DATASETS = {"ucla", "hup", "openieeg"}
MINIMUM_SAMPLING_FREQUENCY_HZ = 1000.0
PILOT_WINDOW_SECONDS = 8.0
MINIMUM_PILOT_WINDOWS = 4
FREQUENCY_ABSOLUTE_TOLERANCE_HZ = 1e-6

INVENTORY_COLUMNS = [
    "recording_index",
    "participant_id",
    "dataset",
    "institution",
    "session",
    "task",
    "run",
    "edf_path",
    "metadata_sampling_frequency_hz",
    "edf_header_sampling_frequency_hz",
    "original_sampling_frequency_hz",
    "sampling_frequency_provenance",
    "sampling_frequency_disagreement",
    "duration_seconds",
    "number_of_channels",
    "number_of_good_ieeg_channels",
    "has_channels_tsv",
    "has_soz_metadata",
    "has_resection_metadata",
    "has_outcome_metadata",
    "eligible_sfreq_ge_1000",
    "exclusion_reason",
]


def extract_entity(filename: str, entity: str) -> str | None:
    """
    Extrait une entité BIDS depuis un nom de fichier.

    Exemple :
    task-sleep -> sleep
    run-01     -> 01
    acq-ecog   -> ecog
    """
    match = re.search(
        rf"(?:^|_){re.escape(entity)}-([^_]+)",
        filename,
    )
    return match.group(1) if match else None

def infer_dataset_name(
    participant_id: str,
    dataset_value,
) -> str:
    """
    Retourne le nom du dataset.

    Priorité :
    1. valeur trouvée dans final_split.csv ou participants.tsv ;
    2. déduction depuis le nom du participant ;
    3. Unknown si aucune source ne permet de conclure.
    """
    if pd.notna(dataset_value):
        dataset_text = str(dataset_value).strip()

        if (
            dataset_text
            and dataset_text.lower()
            not in {"unknown", "nan", "none"}
        ):
            return dataset_text

    participant_lower = participant_id.lower()

    if "multicenter" in participant_lower:
        return "Multicenter"

    if "openieeg" in participant_lower:
        return "OpeniEEG"

    if "sourcesink" in participant_lower:
        return "SourceSink"

    if "zurich" in participant_lower:
        return "Zurich"

    if participant_lower.startswith("sub-hup"):
        return "HUP"

    return "Unknown"

def infer_site_code(participant_id: str, dataset: str) -> str:
    """
    Extrait un code de site depuis le nom du sujet.

    Les codes ne sont pas automatiquement convertis en noms
    d'institutions lorsqu'ils ne sont pas parfaitement explicites.
    """
    participant_lower = participant_id.lower()
    dataset_lower = str(dataset).lower()

    if dataset_lower == "openieeg":
        if "ucla" in participant_lower:
            return "UCLA"
        if "detroit" in participant_lower:
            return "Detroit"
        return "OpeniEEG-other"

    if dataset_lower == "hup":
        return "HUP"

    if dataset_lower == "zurich":
        return "Zurich"

    if dataset_lower == "sourcesink":
        for code in ("nih", "umf", "rns", "py", "jh", "pt"):
            if f"sourcesink{code}" in participant_lower:
                return code.upper()
        return "SourceSink-other"

    if dataset_lower == "multicenter":
        for code in ("ummc", "umf", "jh", "pt"):
            if f"multicenter{code}" in participant_lower:
                return code.upper()
        return "Multicenter-other"

    return "Unknown"


def count_csv_rows(path: Path) -> int | None:
    """
    Compte les lignes d'un CSV sans charger tout le fichier en mémoire.

    La première ligne correspond à l'en-tête.
    """
    if not path.exists():
        return None

    with path.open("rb") as file:
        number_of_lines = sum(1 for _ in file)

    return max(number_of_lines - 1, 0)


def numeric_first(
    dataframe: pd.DataFrame,
    column: str,
) -> float | None:
    if column not in dataframe.columns:
        return None

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(values.iloc[0])


def join_unique_values(series: pd.Series) -> str:
    values = sorted(
        {
            str(value)
            for value in series.dropna()
            if str(value).strip()
        }
    )
    return ";".join(values)


def join_unique_numeric(series: pd.Series) -> str:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    unique_values = sorted(set(values.astype(float)))

    return ";".join(
        f"{value:g}"
        for value in unique_values
    )


def read_channels_table(
    channels_path: Path,
) -> dict:
    """
    Lit les métadonnées d'un enregistrement depuis channels.tsv.
    """
    default_result = {
        "has_channels_tsv": False,
        "n_channels": np.nan,
        "n_good_channels": np.nan,
        "n_bad_channels": np.nan,
        "channel_types": "",
        "channel_sampling_frequency_hz": np.nan,
        "channel_edf_length_seconds": np.nan,
        "n_soz_channels": np.nan,
        "n_resected_channels": np.nan,
        "n_anatomically_labeled_channels": np.nan,
        "channels_read_error": "",
    }

    if not channels_path.exists():
        return default_result

    try:
        channels = pd.read_csv(
            channels_path,
            sep="\t",
        )

        result = default_result.copy()
        result["has_channels_tsv"] = True
        result["n_channels"] = int(len(channels))

        if "type" in channels.columns:
            result["channel_types"] = join_unique_values(
                channels["type"].astype(str).str.upper()
            )

        if "good" in channels.columns:
            good = pd.to_numeric(
                channels["good"],
                errors="coerce",
            )

            result["n_good_channels"] = int(
                good.eq(1).sum()
            )
            result["n_bad_channels"] = int(
                good.eq(0).sum()
            )

        if "sampling_frequency" in channels.columns:
            result["channel_sampling_frequency_hz"] = (
                numeric_first(
                    channels,
                    "sampling_frequency",
                )
            )

        if "edf_length" in channels.columns:
            result["channel_edf_length_seconds"] = (
                numeric_first(
                    channels,
                    "edf_length",
                )
            )

        if "soz" in channels.columns:
            soz = pd.to_numeric(
                channels["soz"],
                errors="coerce",
            )
            result["n_soz_channels"] = int(
                soz.eq(1).sum()
            )

        if "resection" in channels.columns:
            resection = pd.to_numeric(
                channels["resection"],
                errors="coerce",
            )
            result["n_resected_channels"] = int(
                resection.eq(1).sum()
            )

        if "anatomical" in channels.columns:
            anatomical = (
                channels["anatomical"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            missing_labels = {
                "",
                "nan",
                "n/a",
                "na",
                "none",
                "-1",
                "unknown",
            }

            result["n_anatomically_labeled_channels"] = int(
                (~anatomical.isin(missing_labels)).sum()
            )

        return result

    except Exception as error:
        result = default_result.copy()
        result["has_channels_tsv"] = True
        result["channels_read_error"] = repr(error)
        return result


def make_summary(
    dataframe: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    return (
        dataframe.groupby(
            group_columns,
            dropna=False,
        )
        .agg(
            n_subjects=(
                "participant_id",
                "nunique",
            ),
            n_recordings=(
                "edf_relative_path",
                "count",
            ),
            total_duration_hours=(
                "duration_seconds",
                lambda values: (
                    pd.to_numeric(
                        values,
                        errors="coerce",
                    ).sum()
                    / 3600
                ),
            ),
            duration_min_minutes=(
                "duration_minutes",
                "min",
            ),
            duration_median_minutes=(
                "duration_minutes",
                "median",
            ),
            duration_max_minutes=(
                "duration_minutes",
                "max",
            ),
            recordings_per_subject_median=(
                "participant_id",
                lambda values: (
                    values.value_counts().median()
                ),
            ),
            channels_min=(
                "n_channels",
                "min",
            ),
            channels_median=(
                "n_channels",
                "median",
            ),
            channels_max=(
                "n_channels",
                "max",
            ),
            sampling_frequencies_hz=(
                "sampling_frequency_hz",
                join_unique_numeric,
            ),
            modalities_present=(
                "channel_types",
                join_unique_values,
            ),
            recordings_with_hfo_csv=(
                "has_hfo_csv",
                "sum",
            ),
            candidate_hfo_total=(
                "n_candidate_hfo",
                "sum",
            ),
            candidate_hfo_median_per_recording=(
                "n_candidate_hfo",
                "median",
            ),
            candidate_hfo_rate_median_per_minute=(
                "candidate_hfo_per_minute",
                "median",
            ),
        )
        .reset_index()
    )


def legacy_main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/inventory"),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite optionnelle pour tester le script.",
    )

    args = parser.parse_args()

    root = args.root.expanduser()
    output_dir = args.out.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    participants_path = root / "participants.tsv"
    final_split_path = (
        root
        / "derivatives"
        / "datasplit"
        / "final_split.csv"
    )

    participants = pd.read_csv(
        participants_path,
        sep="\t",
    )

    participants_by_id = (
        participants
        .set_index("participant_id")
        .to_dict(orient="index")
    )

    final_split = pd.read_csv(final_split_path)

    split_by_edf = (
        final_split
        .drop_duplicates(subset=["edf_name"])
        .set_index("edf_name")
        .to_dict(orient="index")
    )

    edf_files = sorted(
        path
        for path in root.rglob("*.edf")
        if "derivatives" not in path.parts
    )

    if args.limit is not None:
        edf_files = edf_files[: args.limit]

    print(
        f"{len(edf_files)} EDF à inventorier.",
        flush=True,
    )

    rows: list[dict] = []

    for index, edf_path in enumerate(
        edf_files,
        start=1,
    ):
        relative_path = (
            edf_path.relative_to(root).as_posix()
        )

        participant_id = relative_path.split("/")[0]
        filename = edf_path.name

        participant_metadata = (
            participants_by_id.get(
                participant_id,
                {},
            )
        )

        split_metadata = split_by_edf.get(
            relative_path,
            {},
        )

        dataset_value = split_metadata.get(
            "dataset",
            participant_metadata.get(
                "dataset",
                "Unknown",
            ),
        )

        dataset = infer_dataset_name(
            participant_id,
            dataset_value,
        )

        if split_metadata:
            split_value = split_metadata.get(
                "split",
                "split_missing"
            )
            in_final_split_value = True

        elif str(dataset).lower() == "multicenter":
            split_value = "not_applicable"
            in_final_split_value = False

        else:
            split_value = "not_in_final_split"
            in_final_split_value = False

        session = extract_entity(
            filename,
            "ses",
        )
        task = extract_entity(
            filename,
            "task",
        )
        acquisition = extract_entity(
            filename,
            "acq",
        )
        run = extract_entity(
            filename,
            "run",
        )

        channels_path = edf_path.with_name(
            filename.replace(
                "_ieeg.edf",
                "_channels.tsv",
            )
        )

        channels_metadata = read_channels_table(
            channels_path
        )

        duration_seconds = split_metadata.get(
            "length",
            channels_metadata[
                "channel_edf_length_seconds"
            ],
        )

        sampling_frequency = split_metadata.get(
            "frequency",
            channels_metadata[
                "channel_sampling_frequency_hz"
            ],
        )

        duration_seconds = pd.to_numeric(
            pd.Series([duration_seconds]),
            errors="coerce",
        ).iloc[0]

        sampling_frequency = pd.to_numeric(
            pd.Series([sampling_frequency]),
            errors="coerce",
        ).iloc[0]

        hfo_path = (
            root
            / "derivatives"
            / "hfo"
            / Path(relative_path).with_suffix(".csv")
        )

        number_of_candidate_hfo = count_csv_rows(
            hfo_path
        )

        if (
            number_of_candidate_hfo is not None
            and pd.notna(duration_seconds)
            and duration_seconds > 0
        ):
            candidate_hfo_per_minute = (
                number_of_candidate_hfo
                / (duration_seconds / 60)
            )
        else:
            candidate_hfo_per_minute = np.nan

        row = {
            "participant_id": participant_id,
            "dataset": dataset,
            "site_code": infer_site_code(
                participant_id,
                dataset,
            ),
            "is_main_302_cohort": (
                str(dataset).lower()
                != "multicenter"
            ),
            "session": session,
            "task": task,
            "acquisition": acquisition,
            "run": run,
            "split": split_value,
            "in_final_split": in_final_split_value,
            "edf_relative_path": relative_path,
            "edf_absolute_path": str(
                edf_path.resolve()
            ),
            "duration_seconds": duration_seconds,
            "duration_minutes": (
                duration_seconds / 60
                if pd.notna(duration_seconds)
                else np.nan
            ),
            "sampling_frequency_hz": (
                sampling_frequency
            ),
            "has_hfo_csv": hfo_path.exists(),
            "hfo_csv_relative_path": (
                hfo_path.relative_to(root).as_posix()
                if hfo_path.exists()
                else ""
            ),
            "n_candidate_hfo": (
                number_of_candidate_hfo
                if number_of_candidate_hfo
                is not None
                else np.nan
            ),
            "candidate_hfo_per_minute": (
                candidate_hfo_per_minute
            ),
            "expert_annotation_available": (
                participant_metadata.get(
                    "event_annotation",
                    0,
                )
            ),
            "expert_annotation_count_subject": (
                participant_metadata.get(
                    "event_annotation_count",
                    0,
                )
            ),
            "participant_task_metadata": (
                participant_metadata.get(
                    "task",
                    "",
                )
            ),
            "participant_method": (
                participant_metadata.get(
                    "methods",
                    "",
                )
            ),
            "outcome": participant_metadata.get(
                "outcome",
                np.nan,
            ),
            **channels_metadata,
        }

        rows.append(row)

        if index % 100 == 0:
            print(
                f"{index}/{len(edf_files)} traités",
                flush=True,
            )

    inventory = pd.DataFrame(rows)

    inventory.to_csv(
        output_dir / "recordings_inventory.csv",
        index=False,
    )

    make_summary(
        inventory,
        ["dataset"],
    ).to_csv(
        output_dir / "summary_by_dataset.csv",
        index=False,
    )

    make_summary(
        inventory,
        ["dataset", "task"],
    ).to_csv(
        output_dir
        / "summary_by_dataset_and_task.csv",
        index=False,
    )

    make_summary(
        inventory,
        ["dataset", "site_code"],
    ).to_csv(
        output_dir
        / "summary_by_dataset_and_site.csv",
        index=False,
    )

    make_summary(
        inventory,
        ["dataset", "split"],
    ).to_csv(
        output_dir
        / "summary_by_dataset_and_split.csv",
        index=False,
    )

    make_summary(
        inventory,
        ["dataset", "channel_types"],
    ).to_csv(
        output_dir
        / "summary_by_dataset_and_modality.csv",
        index=False,
    )

    subject_summary = (
        inventory.groupby(
            [
                "participant_id",
                "dataset",
                "site_code",
            ],
            dropna=False,
        )
        .agg(
            n_recordings=(
                "edf_relative_path",
                "count",
            ),
            total_duration_minutes=(
                "duration_minutes",
                "sum",
            ),
            min_duration_minutes=(
                "duration_minutes",
                "min",
            ),
            median_duration_minutes=(
                "duration_minutes",
                "median",
            ),
            max_duration_minutes=(
                "duration_minutes",
                "max",
            ),
            tasks=(
                "task",
                join_unique_values,
            ),
            modalities=(
                "channel_types",
                join_unique_values,
            ),
            candidate_hfo_total=(
                "n_candidate_hfo",
                "sum",
            ),
            expert_annotation_available=(
                "expert_annotation_available",
                "max",
            ),
            expert_annotation_count_subject=(
                "expert_annotation_count_subject",
                "max",
            ),
        )
        .reset_index()
    )

    subject_summary.to_csv(
        output_dir / "summary_by_subject.csv",
        index=False,
    )

    recordings_not_in_split = inventory[
        inventory["split"].eq(
            "not_in_final_split"
        )
    ]

    recordings_not_in_split.to_csv(
        output_dir
        / "recordings_not_in_final_split.csv",
        index=False,
    )

    recordings_split_not_applicable = inventory[
        inventory["split"].eq(
            "not_applicable"
        )
    ]

    recordings_split_not_applicable.to_csv(
        output_dir
        / "recordings_split_not_applicable.csv",
        index=False,
    )

    recordings_without_hfo = inventory[
        ~inventory["has_hfo_csv"]
    ]

    recordings_without_hfo.to_csv(
        output_dir
        / "recordings_without_hfo_csv.csv",
        index=False,
    )

    recordings_without_channels = inventory[
        ~inventory["has_channels_tsv"]
    ]

    recordings_without_channels.to_csv(
        output_dir
        / "recordings_without_channels_tsv.csv",
        index=False,
    )

    channel_errors = inventory[
        inventory["channels_read_error"]
        .fillna("")
        .ne("")
    ]

    channel_errors.to_csv(
        output_dir
        / "channels_read_errors.csv",
        index=False,
    )

    split_consistency = (
        final_split.groupby("patient_name")["split"]
        .nunique()
        .reset_index(
            name="n_distinct_splits"
        )
    )

    split_conflicts = split_consistency[
        split_consistency[
            "n_distinct_splits"
        ] > 1
    ]

    split_conflicts.to_csv(
        output_dir
        / "subject_split_conflicts.csv",
        index=False,
    )

    expert_summary = (
        participants[
            participants["event_annotation"].eq(1)
        ]
        .groupby("dataset")
        .agg(
            n_expert_annotated_subjects=(
                "participant_id",
                "nunique",
            ),
            n_expert_annotated_events=(
                "event_annotation_count",
                "sum",
            ),
            median_expert_events_per_subject=(
                "event_annotation_count",
                "median",
            ),
            min_expert_events_per_subject=(
                "event_annotation_count",
                "min",
            ),
            max_expert_events_per_subject=(
                "event_annotation_count",
                "max",
            ),
        )
        .reset_index()
    )

    expert_summary.to_csv(
        output_dir
        / "expert_annotation_summary_by_dataset.csv",
        index=False,
    )

    print("\n=== INVENTAIRE TERMINÉ ===")
    print(
        "EDF inventoriés :",
        len(inventory),
    )
    print(
        "Sujets :",
        inventory["participant_id"].nunique(),
    )
    print(
        "EDF absents de final_split :",
        len(recordings_not_in_split),
    )
    print(
        "EDF sans CSV HFO :",
        len(recordings_without_hfo),
    )
    print(
        "EDF sans channels.tsv :",
        len(recordings_without_channels),
    )
    print(
        "Conflits train/test au niveau sujet :",
        len(split_conflicts),
    )

    print("\n=== RÉSUMÉ PAR DATASET ===")
    print(
        make_summary(
            inventory,
            ["dataset"],
        ).to_string(index=False)
    )

    print(
        "\nRésultats :",
        output_dir.resolve(),
    )


def _simple_yaml_scalar(value: str):
    """Parse the scalar subset used by datasets2.yml without requiring PyYAML."""
    tokens = shlex.split(value, comments=True, posix=True)
    if not tokens:
        return ""
    scalar = " ".join(tokens)
    lowered = scalar.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null", "~"}:
        return None
    return scalar


def load_datasets_configuration(path: Path = DATASETS_CONFIG) -> dict:
    """Load the shallow dataset/file-pattern structure used by datasets2.yml."""
    config: dict[str, dict] = {}
    current_dataset: dict | None = None
    current_section: str | None = None

    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            expanded = line.expandtabs(8).rstrip()
            stripped = expanded.lstrip()
            if not stripped or stripped.startswith("#"):
                continue

            indentation = len(expanded) - len(stripped)
            if indentation == 0 and stripped.endswith(":"):
                current_dataset = {}
                config[stripped[:-1].strip().lower()] = current_dataset
                current_section = None
                continue
            if current_dataset is None or ":" not in stripped:
                continue

            key, value = (part.strip() for part in stripped.split(":", 1))
            if indentation == 2:
                if value:
                    current_dataset[key] = _simple_yaml_scalar(value)
                    current_section = None
                else:
                    current_dataset[key] = {}
                    current_section = key
            elif indentation == 4 and current_section is not None:
                section = current_dataset[current_section]
                if isinstance(section, dict):
                    section[key] = _simple_yaml_scalar(value)

    return config


def parse_sampling_frequency(value: object) -> tuple[float | None, str]:
    """Return a positive finite frequency and a structured parse status."""
    if value is None:
        return None, "missing"
    try:
        if pd.isna(value):
            return None, "missing"
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a", "na"}:
        return None, "missing"
    try:
        frequency = float(text)
    except (TypeError, ValueError):
        return None, "malformed"
    if not np.isfinite(frequency) or frequency <= 0:
        return None, "malformed"
    return frequency, "valid"


def resolve_sampling_frequency(
    metadata_value: object,
    header_value: object,
    *,
    header_status: str = "valid",
    header_checked: bool = True,
) -> dict:
    """Resolve provenance and eligibility, using the EDF header as authority."""
    metadata_frequency, _ = parse_sampling_frequency(metadata_value)
    header_frequency, parsed_header_status = parse_sampling_frequency(header_value)
    effective_header_status = (
        parsed_header_status if header_status == "valid" else header_status
    )

    disagreement = bool(
        metadata_frequency is not None
        and header_frequency is not None
        and not np.isclose(
            metadata_frequency,
            header_frequency,
            rtol=0.0,
            atol=FREQUENCY_ABSOLUTE_TOLERANCE_HZ,
        )
    )

    if header_frequency is not None and effective_header_status == "valid":
        original_frequency = header_frequency
        provenance = "edf_header"
    elif metadata_frequency is not None:
        original_frequency = metadata_frequency
        provenance = "metadata_only_unverified"
    else:
        original_frequency = None
        provenance = "unavailable"

    if not header_checked or effective_header_status == "not_checked":
        eligible = False
        exclusion_reason = "edf_header_not_checked"
    elif effective_header_status == "unreadable":
        eligible = False
        exclusion_reason = "edf_header_unreadable"
    elif effective_header_status == "missing":
        eligible = False
        exclusion_reason = "edf_header_sampling_frequency_missing"
    elif effective_header_status == "malformed" or header_frequency is None:
        eligible = False
        exclusion_reason = "edf_header_sampling_frequency_malformed"
    elif header_frequency < MINIMUM_SAMPLING_FREQUENCY_HZ:
        eligible = False
        exclusion_reason = "original_sampling_frequency_below_1000"
    else:
        eligible = True
        exclusion_reason = ""

    return {
        "metadata_sampling_frequency_hz": metadata_frequency,
        "edf_header_sampling_frequency_hz": header_frequency,
        "original_sampling_frequency_hz": original_frequency,
        "sampling_frequency_provenance": provenance,
        "sampling_frequency_disagreement": disagreement,
        "eligible_sfreq_ge_1000": eligible,
        "exclusion_reason": exclusion_reason,
    }


def _pattern_below_data_root(pattern: str, data_root: Path) -> str:
    """Relocate a datasets2.yml subject glob below the caller's data root."""
    parts = Path(str(pattern)).parts
    subject_position = next(
        (index for index, part in enumerate(parts) if part.startswith("sub-")),
        None,
    )
    if subject_position is None:
        raise ValueError(f"dataset pattern has no sub-* component: {pattern}")
    return str(data_root.joinpath(*parts[subject_position:]))


def discover_scoped_recordings(
    data_root: Path,
    datasets: list[str],
    *,
    config_path: Path = DATASETS_CONFIG,
    limit: int | None = None,
) -> list[dict]:
    """Expand only requested ucla/hup/openieeg entries, preserving normalized keys."""
    requested = [str(dataset).strip().lower() for dataset in datasets]
    unsupported = sorted(set(requested) - ALLOWED_DATASETS)
    if unsupported:
        raise ValueError(
            "unsupported dataset entries: "
            + ", ".join(unsupported)
            + "; only "
            + ", ".join(sorted(ALLOWED_DATASETS))
            + " are permitted"
        )

    config = load_datasets_configuration(config_path)
    discovered: list[dict] = []
    identities: dict[str, str] = {}

    for dataset_name in requested:
        entry = config.get(dataset_name)
        if not isinstance(entry, dict):
            raise ValueError(f"dataset entry not found in {config_path}: {dataset_name}")
        if bool(entry.get("skip", False)):
            LOGGER.info("Skipping disabled datasets2.yml entry %s", dataset_name)
            continue
        patterns = entry.get("file_pattern")
        cluster_pattern = patterns.get("cluster") if isinstance(patterns, dict) else None
        if not cluster_pattern:
            raise ValueError(f"dataset {dataset_name} has no cluster file pattern")

        relocated = _pattern_below_data_root(str(cluster_pattern), data_root)
        # An optional `cluster_exclude` glob is subtracted from `cluster`. It lets a
        # broad entry (openieeg) stay disjoint from a narrower one (ucla, whose glob
        # is a strict subset of it) so both can be inventoried in the same run.
        exclude_pattern = (
            patterns.get("cluster_exclude") if isinstance(patterns, dict) else None
        )
        excluded_identities: set[str] = set()
        if exclude_pattern:
            excluded_relocated = _pattern_below_data_root(str(exclude_pattern), data_root)
            excluded_identities = {
                str(Path(match).resolve(strict=False))
                for match in glob.glob(excluded_relocated, recursive=True)
            }
            LOGGER.info(
                "Dataset %s excludes %d recording(s) matching %s",
                dataset_name,
                len(excluded_identities),
                exclude_pattern,
            )
        for match in sorted(glob.glob(relocated, recursive=True)):
            path = Path(match)
            if not path.is_file() or not path.name.endswith("_ieeg.edf"):
                continue
            identity = str(path.resolve(strict=False))
            if identity in excluded_identities:
                continue
            previous = identities.get(identity)
            if previous is not None and previous != dataset_name:
                raise ValueError(
                    f"recording matched multiple dataset entries: {path} "
                    f"({previous}, {dataset_name})"
                )
            if previous is None:
                identities[identity] = dataset_name
                discovered.append(
                    {
                        "dataset": dataset_name,
                        "institution": dataset_name,
                        "edf_path": path,
                    }
                )

    discovered.sort(key=lambda item: (item["dataset"], str(item["edf_path"])))
    if limit is not None:
        discovered = discovered[:limit]
    return discovered


def _single_numeric_value(values: pd.Series | list[object]) -> tuple[float | None, str]:
    parsed = [parse_sampling_frequency(value) for value in values]
    valid = sorted({value for value, status in parsed if status == "valid"})
    if len(valid) == 1:
        return valid[0], "valid"
    if len(valid) > 1:
        return None, "malformed"
    if any(status == "malformed" for _, status in parsed):
        return None, "malformed"
    return None, "missing"


def _is_informative(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() not in {
        "",
        "nan",
        "none",
        "null",
        "n/a",
        "na",
        "unknown",
        "-1",
    }


def read_channels_metadata(channels_path: Path) -> dict:
    """Read sidecar metadata without touching EDF signal samples."""
    result = {
        "has_channels_tsv": channels_path.is_file(),
        "channels_tsv_valid": False,
        "metadata_sampling_frequency_hz": None,
        "metadata_sampling_frequency_status": "missing",
        "metadata_duration_seconds": None,
        "number_of_channels": None,
        "number_of_good_ieeg_channels": 0,
        "has_soz_metadata": False,
        "has_resection_metadata": False,
        "has_soz_and_non_soz_channels": False,
    }
    if not channels_path.is_file():
        return result

    try:
        channels = pd.read_csv(channels_path, sep="\t")
    except Exception as error:
        LOGGER.warning("Cannot read channels sidecar %s: %s", channels_path, error)
        return result

    result["channels_tsv_valid"] = True
    result["number_of_channels"] = int(len(channels))

    if "sampling_frequency" in channels.columns:
        frequency, status = _single_numeric_value(channels["sampling_frequency"])
        result["metadata_sampling_frequency_hz"] = frequency
        result["metadata_sampling_frequency_status"] = status

    if "edf_length" in channels.columns:
        durations = [parse_sampling_frequency(value) for value in channels["edf_length"]]
        valid_durations = sorted(
            {value for value, status in durations if status == "valid"}
        )
        if len(valid_durations) == 1:
            result["metadata_duration_seconds"] = valid_durations[0]

    if "type" in channels.columns:
        ieeg = channels["type"].astype(str).str.upper().isin({"SEEG", "ECOG", "EEG"})
    else:
        ieeg = pd.Series(True, index=channels.index)
    good = pd.Series(True, index=channels.index)
    if "good" in channels.columns:
        good &= pd.to_numeric(channels["good"], errors="coerce").eq(1)
    if "status" in channels.columns:
        good &= ~channels["status"].astype(str).str.lower().eq("bad")
    result["number_of_good_ieeg_channels"] = int((ieeg & good).sum())

    if "soz" in channels.columns:
        soz = pd.to_numeric(channels["soz"], errors="coerce")
        result["has_soz_metadata"] = bool(soz.isin([0, 1]).any())
        result["has_soz_and_non_soz_channels"] = bool(
            soz.eq(0).any() and soz.eq(1).any()
        )
    if "resection" in channels.columns:
        resection = pd.to_numeric(channels["resection"], errors="coerce")
        result["has_resection_metadata"] = bool(resection.isin([0, 1]).any())
    return result


def read_edf_header(edf_path: Path, *, check: bool) -> dict:
    """Read only EDF metadata; never preload or request signal samples."""
    if not check:
        return {
            "sampling_frequency_hz": None,
            "sampling_frequency_status": "not_checked",
            "duration_seconds": None,
            "number_of_channels": None,
        }

    raw = None
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        frequency, status = parse_sampling_frequency(raw.info.get("sfreq"))
        duration = (
            float(raw.n_times / frequency)
            if frequency is not None and int(raw.n_times) >= 0
            else None
        )
        return {
            "sampling_frequency_hz": frequency,
            "sampling_frequency_status": status,
            "duration_seconds": duration,
            "number_of_channels": int(len(raw.ch_names)),
        }
    except Exception as error:
        LOGGER.warning("Cannot read EDF header %s: %s", edf_path, error)
        return {
            "sampling_frequency_hz": None,
            "sampling_frequency_status": "unreadable",
            "duration_seconds": None,
            "number_of_channels": None,
        }
    finally:
        if raw is not None:
            raw.close()


def _load_final_split(data_root: Path) -> dict[str, list[dict]]:
    path = data_root / "derivatives" / "datasplit" / "final_split.csv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    if "edf_name" not in frame.columns:
        return {}
    lookup: dict[str, list[dict]] = {}
    for row in frame.to_dict(orient="records"):
        identity = str(row.get("edf_name", "")).strip().replace("\\", "/")
        if not identity:
            continue
        for key in {identity, Path(identity).name}:
            lookup.setdefault(key, []).append(row)
    return lookup


def _lookup_final_split(
    edf_path: Path,
    data_root: Path,
    lookup: dict[str, list[dict]],
) -> dict:
    try:
        relative = edf_path.relative_to(data_root).as_posix()
    except ValueError:
        relative = edf_path.name
    for key in (relative, edf_path.name):
        matches = lookup.get(key, [])
        if len(matches) == 1:
            return matches[0]
    return {}


def _load_participant_outcomes(data_root: Path) -> dict[str, object]:
    path = data_root / "participants.tsv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, sep="\t")
    if "participant_id" not in frame.columns or "outcome" not in frame.columns:
        return {}
    return dict(zip(frame["participant_id"].astype(str), frame["outcome"]))


def _metadata_value(
    split_row: dict,
    split_column: str,
    sidecar_value: object,
) -> object:
    split_value = split_row.get(split_column)
    if parse_sampling_frequency(split_value)[1] == "valid":
        return split_value
    return sidecar_value


def build_hup_ucla_inventory(
    *,
    data_root: Path,
    datasets: list[str],
    check_edf_header: bool,
    limit: int | None = None,
    config_path: Path = DATASETS_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the required inventory and an internal frame used for pilot ranking."""
    selected = discover_scoped_recordings(
        data_root,
        datasets,
        config_path=config_path,
        limit=limit,
    )
    final_split = _load_final_split(data_root)
    outcomes = _load_participant_outcomes(data_root)
    rows: list[dict] = []

    for recording_index, selected_item in enumerate(selected):
        edf_path = Path(selected_item["edf_path"])
        participant_id = next(
            (part for part in edf_path.parts if part.startswith("sub-")),
            "",
        )
        channels_path = edf_path.with_name(
            edf_path.name[: -len("_ieeg.edf")] + "_channels.tsv"
        )
        channels = read_channels_metadata(channels_path)
        split_row = _lookup_final_split(edf_path, data_root, final_split)
        metadata_frequency = _metadata_value(
            split_row,
            "frequency",
            channels["metadata_sampling_frequency_hz"],
        )
        header = read_edf_header(edf_path, check=check_edf_header)
        frequency = resolve_sampling_frequency(
            metadata_frequency,
            header["sampling_frequency_hz"],
            header_status=header["sampling_frequency_status"],
            header_checked=check_edf_header,
        )

        metadata_duration = _metadata_value(
            split_row,
            "length",
            channels["metadata_duration_seconds"],
        )
        parsed_duration, _ = parse_sampling_frequency(metadata_duration)
        duration_seconds = (
            header["duration_seconds"]
            if header["duration_seconds"] is not None
            else parsed_duration
        )
        number_of_channels = (
            header["number_of_channels"]
            if header["number_of_channels"] is not None
            else channels["number_of_channels"]
        )

        row = {
            "recording_index": recording_index,
            "participant_id": participant_id,
            "dataset": selected_item["dataset"],
            "institution": selected_item["institution"],
            "session": extract_entity(edf_path.name, "ses") or "",
            "task": extract_entity(edf_path.name, "task") or "",
            "run": extract_entity(edf_path.name, "run") or "",
            "edf_path": str(edf_path.resolve(strict=False)),
            **frequency,
            "duration_seconds": duration_seconds,
            "number_of_channels": number_of_channels,
            "number_of_good_ieeg_channels": channels[
                "number_of_good_ieeg_channels"
            ],
            "has_channels_tsv": channels["has_channels_tsv"],
            "has_soz_metadata": channels["has_soz_metadata"],
            "has_resection_metadata": channels["has_resection_metadata"],
            "has_outcome_metadata": _is_informative(outcomes.get(participant_id)),
            "channels_tsv_valid": channels["channels_tsv_valid"],
            "has_soz_and_non_soz_channels": channels[
                "has_soz_and_non_soz_channels"
            ],
        }
        rows.append(row)

        if (recording_index + 1) % 50 == 0:
            LOGGER.info("Inspected %d/%d EDF headers", recording_index + 1, len(selected))

    internal_columns = INVENTORY_COLUMNS + [
        "channels_tsv_valid",
        "has_soz_and_non_soz_channels",
    ]
    internal = pd.DataFrame(rows, columns=internal_columns)
    return internal[INVENTORY_COLUMNS].copy(), internal


def select_pilot_candidates(internal: pd.DataFrame) -> pd.DataFrame:
    """Rank up to five eligible, well-described recordings per normalized site."""
    pilot_columns = INVENTORY_COLUMNS + [
        "pilot_rank",
        "complete_8s_windows",
        "interictal_priority",
        "has_soz_and_non_soz_channels",
    ]
    if internal.empty:
        return pd.DataFrame(columns=pilot_columns)

    work = internal.copy()
    work["complete_8s_windows"] = np.floor(
        pd.to_numeric(work["duration_seconds"], errors="coerce")
        / PILOT_WINDOW_SECONDS
    ).fillna(0).astype(int)
    work["interictal_priority"] = (
        work["task"].astype(str).str.lower().eq("interictal")
    )
    work = work[
        work["eligible_sfreq_ge_1000"].astype(bool)
        & work["channels_tsv_valid"].astype(bool)
        & (work["number_of_good_ieeg_channels"] > 0)
        & (work["complete_8s_windows"] >= MINIMUM_PILOT_WINDOWS)
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=pilot_columns)

    ranked: list[pd.DataFrame] = []
    for dataset, group in work.groupby("dataset", sort=True):
        group = group.sort_values(
            [
                "interictal_priority",
                "has_soz_and_non_soz_channels",
                "has_resection_metadata",
                "has_outcome_metadata",
                "number_of_good_ieeg_channels",
                "duration_seconds",
                "edf_path",
            ],
            ascending=[False, False, False, False, False, False, True],
            kind="stable",
        ).head(5)
        group["pilot_rank"] = np.arange(1, len(group) + 1)
        ranked.append(group)
    pilots = pd.concat(ranked, ignore_index=True)
    return pilots[pilot_columns]


def make_sampling_frequency_counts(inventory: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "institution",
        "original_sampling_frequency_hz",
        "eligible_sfreq_ge_1000",
        "exclusion_reason",
        "number_of_recordings",
    ]
    if inventory.empty:
        return pd.DataFrame(columns=columns)
    return (
        inventory.groupby(
            columns[:-1],
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "number_of_recordings"})
        .sort_values(
            ["dataset", "original_sampling_frequency_hz", "exclusion_reason"],
            na_position="last",
        )
    )


def make_sampling_frequency_by_site(inventory: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "institution",
        "number_of_recordings",
        "number_eligible",
        "number_excluded",
        "number_with_metadata_frequency",
        "number_with_edf_header_frequency",
        "number_with_sampling_frequency_disagreement",
        "minimum_original_sampling_frequency_hz",
        "median_original_sampling_frequency_hz",
        "maximum_original_sampling_frequency_hz",
    ]
    if inventory.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (dataset, institution), group in inventory.groupby(
        ["dataset", "institution"], sort=True
    ):
        original = pd.to_numeric(
            group["original_sampling_frequency_hz"], errors="coerce"
        )
        eligible = group["eligible_sfreq_ge_1000"].astype(bool)
        rows.append(
            {
                "dataset": dataset,
                "institution": institution,
                "number_of_recordings": len(group),
                "number_eligible": int(eligible.sum()),
                "number_excluded": int((~eligible).sum()),
                "number_with_metadata_frequency": int(
                    group["metadata_sampling_frequency_hz"].notna().sum()
                ),
                "number_with_edf_header_frequency": int(
                    group["edf_header_sampling_frequency_hz"].notna().sum()
                ),
                "number_with_sampling_frequency_disagreement": int(
                    group["sampling_frequency_disagreement"].astype(bool).sum()
                ),
                "minimum_original_sampling_frequency_hz": original.min(),
                "median_original_sampling_frequency_hz": original.median(),
                "maximum_original_sampling_frequency_hz": original.max(),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def make_subject_counts_by_site(inventory: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "institution",
        "number_of_subjects",
        "number_of_recordings",
        "number_of_eligible_subjects",
        "number_of_eligible_recordings",
        "number_of_excluded_recordings",
    ]
    if inventory.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (dataset, institution), group in inventory.groupby(
        ["dataset", "institution"], sort=True
    ):
        eligible = group[group["eligible_sfreq_ge_1000"].astype(bool)]
        rows.append(
            {
                "dataset": dataset,
                "institution": institution,
                "number_of_subjects": group["participant_id"].nunique(),
                "number_of_recordings": len(group),
                "number_of_eligible_subjects": eligible["participant_id"].nunique(),
                "number_of_eligible_recordings": len(eligible),
                "number_of_excluded_recordings": len(group) - len(eligible),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def write_inventory_html(
    output_path: Path,
    *,
    inventory: pd.DataFrame,
    frequency_counts: pd.DataFrame,
    frequency_by_site: pd.DataFrame,
    subject_counts: pd.DataFrame,
    pilots: pd.DataFrame,
    datasets: list[str],
    check_edf_header: bool,
) -> None:
    excluded = inventory[~inventory["eligible_sfreq_ge_1000"].astype(bool)]
    exclusion_counts = (
        excluded.groupby(["dataset", "exclusion_reason"], as_index=False)
        .size()
        .rename(columns={"size": "number_of_recordings"})
        if not excluded.empty
        else pd.DataFrame(
            columns=["dataset", "exclusion_reason", "number_of_recordings"]
        )
    )

    def table(frame: pd.DataFrame) -> str:
        return frame.to_html(index=False, escape=True, border=0, na_rep="")

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>HUP/UCLA sampling-frequency inventory</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1f2937; }}
table {{ border-collapse: collapse; margin-bottom: 2rem; width: 100%; }}
th, td {{ border: 1px solid #d1d5db; padding: .4rem .55rem; text-align: left; }}
th {{ background: #eef2ff; }}
code {{ background: #f3f4f6; padding: .1rem .25rem; }}
</style>
</head>
<body>
<h1>HUP/UCLA sampling-frequency inventory</h1>
<p>Datasets: <code>{html.escape(' '.join(datasets))}</code>. EDF headers checked:
<code>{str(check_edf_header).lower()}</code>. Eligibility requires a verified EDF-header
sampling frequency of at least 1000 Hz. No EDF signal samples were loaded.</p>
<p>Pilot candidates require a valid channels.tsv, at least one good iEEG channel, and at
least {MINIMUM_PILOT_WINDOWS} complete non-overlapping {PILOT_WINDOW_SECONDS:g}-second windows.</p>
<h2>Sampling frequency by site</h2>
{table(frequency_by_site)}
<h2>Subject counts by site</h2>
{table(subject_counts)}
<h2>Exclusion reasons</h2>
{table(exclusion_counts)}
<h2>Sampling-frequency counts</h2>
{table(frequency_counts)}
<h2>Pilot candidates</h2>
{table(pilots)}
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def write_hup_ucla_outputs(
    output_dir: Path,
    *,
    inventory: pd.DataFrame,
    internal: pd.DataFrame,
    datasets: list[str],
    check_edf_header: bool,
    output_suffix: str = "hup_ucla",
) -> dict[str, Path]:
    """Write the inventory tables. ``output_suffix`` names the cohort in the filenames.

    The default reproduces the HUP/UCLA filenames exactly; a scoped run (for
    example the OpenIEEG spectral diagnostic) passes its own suffix so its
    tables can never be mistaken for the 1 kHz preprocessing cohort's.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible = inventory[inventory["eligible_sfreq_ge_1000"].astype(bool)].copy()
    excluded = inventory[~inventory["eligible_sfreq_ge_1000"].astype(bool)].copy()
    frequency_counts = make_sampling_frequency_counts(inventory)
    frequency_by_site = make_sampling_frequency_by_site(inventory)
    subject_counts = make_subject_counts_by_site(inventory)
    pilots = select_pilot_candidates(internal)

    outputs = {
        "inventory": output_dir / f"recording_inventory_{output_suffix}.csv",
        "eligible": output_dir / f"eligible_recordings_{output_suffix}.csv",
        "excluded": output_dir / f"excluded_recordings_{output_suffix}.csv",
        "frequency_counts": output_dir / f"sampling_frequency_counts_{output_suffix}.csv",
        "frequency_by_site": output_dir / f"sampling_frequency_by_site_{output_suffix}.csv",
        "subject_counts": output_dir / f"subject_counts_by_site_{output_suffix}.csv",
        "pilots": output_dir / f"pilot_candidates_{output_suffix}.csv",
        "html": output_dir / f"inventory_{output_suffix}.html",
    }
    inventory.to_csv(outputs["inventory"], index=False)
    eligible.to_csv(outputs["eligible"], index=False)
    excluded.to_csv(outputs["excluded"], index=False)
    frequency_counts.to_csv(outputs["frequency_counts"], index=False)
    frequency_by_site.to_csv(outputs["frequency_by_site"], index=False)
    subject_counts.to_csv(outputs["subject_counts"], index=False)
    pilots.to_csv(outputs["pilots"], index=False)
    write_inventory_html(
        outputs["html"],
        inventory=inventory,
        frequency_counts=frequency_counts,
        frequency_by_site=frequency_by_site,
        subject_counts=subject_counts,
        pilots=pilots,
        datasets=datasets,
        check_edf_header=check_edf_header,
    )
    return outputs


def parse_hup_ucla_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only EDF-header sampling-frequency inventory for the "
            "ucla and hup entries in Neurodags/datasets2.yml."
        )
    )
    parser.add_argument("--data-root", "--root", dest="data_root", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ucla", "hup"],
        help="datasets2.yml entries to inspect (only ucla and hup are allowed)",
    )
    parser.add_argument(
        "--output-dir", "--out", dest="output_dir", type=Path, required=True
    )
    parser.add_argument(
        "--check-edf-header",
        action="store_true",
        help="read each EDF header with MNE preload=False; required for eligibility",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-suffix",
        default="hup_ucla",
        help=(
            "cohort name used in the output filenames; keep the default for the "
            "HUP/UCLA cohort and pass e.g. openieeg for a scoped diagnostic run"
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()
    args.datasets = [str(value).strip().lower() for value in args.datasets]
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    return args


def main() -> None:
    args = parse_hup_ucla_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"data root is not a directory: {data_root}")

    inventory, internal = build_hup_ucla_inventory(
        data_root=data_root,
        datasets=args.datasets,
        check_edf_header=args.check_edf_header,
        limit=args.limit,
    )
    outputs = write_hup_ucla_outputs(
        output_dir,
        inventory=inventory,
        internal=internal,
        datasets=args.datasets,
        check_edf_header=args.check_edf_header,
        output_suffix=args.output_suffix,
    )
    eligible = int(inventory["eligible_sfreq_ge_1000"].astype(bool).sum())
    LOGGER.info("Inventory recordings: %d", len(inventory))
    LOGGER.info("Eligible recordings: %d", eligible)
    LOGGER.info("Excluded recordings: %d", len(inventory) - eligible)
    for path in outputs.values():
        LOGGER.info("Wrote %s", path)


if __name__ == "__main__":
    main()
