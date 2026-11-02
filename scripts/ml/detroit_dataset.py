"""Build Detroit channel-level ML inputs for the existing frozen evaluators.

Detroit is supplied as already aggregated participant-by-channel feature
tables.  This adapter only validates, joins and labels those tables; all model
selection, fitting and reporting remain in the existing HUP/UCLA ML modules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.ml.common import atomic_csv, atomic_json, atomic_parquet, git_commit, new_output_directory, sha256, timestamp_utc
from scripts.ml.dataset import (
    FAMILY_ORDER, FINAL_KEY, base_feature_names, cohort_count_tables,
    feature_counts as baseline_feature_counts, feature_qc_tables as baseline_feature_qc,
    predictor_names as baseline_predictors,
)
from scripts.ml.complexity_dataset import (
    COMPLEXITY_EPOCH_FEATURE_COUNT, base_feature_names as complexity_base_features,
    feature_counts as complexity_feature_counts, feature_qc_tables as complexity_feature_qc,
    predictor_names as complexity_predictors,
)

REPOSITORY = Path(__file__).resolve().parents[2]
DETROIT_TASK = "detroit"
EXPECTED = {"participants": 85, "observations": 7244, "pathological": 854, "non_pathological": 6390}
EXPECTED_TOTAL_CHANNELS = 9028
EXPECTED_AMBIGUOUS = 1784
AGGREGATIONS = ("mean", "std", "median")


def _required_channel_columns(frame: pd.DataFrame, path: Path) -> tuple[str, list[str]]:
    channel_column = "channel_id" if "channel_id" in frame else "channel" if "channel" in frame else ""
    missing = sorted({"participant_id", "supervised_label"} - set(frame.columns))
    if not channel_column:
        missing.append("channel or channel_id")
    if missing:
        raise ValueError(f"{path} lacks required Detroit identity/label columns: {missing}")
    return channel_column, ["participant_id", channel_column]


def _strict_labels(frame: pd.DataFrame, path: Path) -> pd.Series:
    labels = pd.to_numeric(frame["supervised_label"], errors="raise")
    finite = labels.dropna()
    if not finite.isin([0, 1]).all():
        raise ValueError(f"{path} supervised_label must be 0, 1 or NaN")
    return labels


def _load_family(path: Path, predictors: list[str], family: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    channel_column, identity = _required_channel_columns(frame, path)
    missing = sorted(set(predictors) - set(frame.columns))
    unexpected = sorted(
        column for column in frame.columns
        if column.endswith(("_mean", "_std", "_median")) and column not in predictors
    )
    if missing or unexpected:
        raise ValueError(f"{family} predictor contract mismatch; missing={missing}, unexpected={unexpected}")
    labels = _strict_labels(frame, path)
    selected = frame[["participant_id", channel_column, "supervised_label"] + predictors].copy()
    selected = selected.rename(columns={channel_column: "channel_id"})
    selected["participant_id"] = selected["participant_id"].astype(str).str.strip()
    selected["channel_id"] = selected["channel_id"].astype(str).str.strip()
    if selected[["participant_id", "channel_id"]].eq("").any().any():
        raise ValueError(f"{path} contains an empty participant or channel identity")
    if selected.duplicated(FINAL_KEY).any():
        raise ValueError(f"{path} is not a participant-by-channel table (duplicate {FINAL_KEY})")
    selected["supervised_label"] = labels
    values = selected[predictors].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{family} contains predictor NaN/Inf")
    return selected.sort_values(FINAL_KEY, kind="stable").reset_index(drop=True)


def _assert_same_rows(reference: pd.DataFrame, current: pd.DataFrame, family: str) -> None:
    reference_key = pd.MultiIndex.from_frame(reference[FINAL_KEY])
    current_key = pd.MultiIndex.from_frame(current[FINAL_KEY])
    if not reference_key.equals(current_key):
        raise ValueError(f"Detroit channel rows differ between baseline reference and {family}")
    if not reference["supervised_label"].equals(current["supervised_label"]):
        raise ValueError(f"Detroit supervised labels differ between baseline reference and {family}")


def _supervised(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[frame["supervised_label"].notna()].copy()
    if result["supervised_label"].isna().any():
        raise AssertionError("ambiguous labels reached supervised ML")
    result["target"] = result["supervised_label"].astype(int)
    return result.drop(columns="supervised_label")


def _finalize(frame: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    result = _supervised(frame)
    result.insert(2, "dataset", "detroit")
    result.insert(3, "task", DETROIT_TASK)
    result.insert(4, "mne_signal_type", "unknown")
    result.insert(5, "channel_type", "unknown")
    result.insert(7, "n_recordings", 1)
    result.insert(8, "n_valid_epochs", 0)
    result = result[FINAL_KEY + ["dataset", "task", "mne_signal_type", "channel_type", "target", "n_recordings", "n_valid_epochs"] + predictors]
    return result


def _validation(final: pd.DataFrame, predictors: list[str], expected_predictor_count: int) -> dict[str, Any]:
    counts = {
        "participants": int(final["participant_id"].nunique()), "sites": int(final["dataset"].nunique()),
        "observations": int(len(final)), "pathological": int(final["target"].eq(1).sum()),
        "non_pathological": int(final["target"].eq(0).sum()),
    }
    if {key: counts[key] for key in EXPECTED} != EXPECTED:
        raise ValueError(f"Detroit supervised cohort counts differ from the frozen contract: {counts}")
    if len(predictors) != expected_predictor_count:
        raise AssertionError(f"Detroit predictor contract is {len(predictors)}, expected {expected_predictor_count}")
    values = final[predictors].to_numpy(float)
    if not np.isfinite(values).all() or final["target"].isna().any() or final["participant_id"].eq("").any():
        raise ValueError("Detroit validation failed: invalid predictors, labels, or participant grouping")
    return {"status": "passed", "task": DETROIT_TASK, "counts": counts, "predictor_count": len(predictors), "ambiguous_labels_in_ml": 0, "participant_grouping": "participant_id", "invalid_predictor_values": 0}


def build_datasets(relative_bandpower: Path, time_domain: Path, frequency_domain: Path, complexity: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    # Select each family's frozen columns rather than inferring from names.
    from scripts.ml.dataset import _feature_names
    family_columns = {family: [f"{name}_{stat}" for name in _feature_names()[family] for stat in AGGREGATIONS] for family in FAMILY_ORDER}
    baseline_by_family = {
        "relative_bandpower": _load_family(relative_bandpower, family_columns["relative_bandpower"], "relative_bandpower"),
        "time_domain": _load_family(time_domain, family_columns["time_domain"], "time_domain"),
        "frequency_domain": _load_family(frequency_domain, family_columns["frequency_domain"], "frequency_domain"),
    }
    reference = baseline_by_family["relative_bandpower"]
    for family in ("time_domain", "frequency_domain"):
        _assert_same_rows(reference, baseline_by_family[family], family)
    if len(reference) != EXPECTED_TOTAL_CHANNELS or int(reference["supervised_label"].isna().sum()) != EXPECTED_AMBIGUOUS:
        raise ValueError(
            "Detroit assembled-channel contract differs from the confirmed cohort: "
            f"rows={len(reference)}, ambiguous={int(reference['supervised_label'].isna().sum())}"
        )
    baseline = reference[FINAL_KEY + ["supervised_label"]].copy()
    for family in FAMILY_ORDER:
        baseline = baseline.merge(baseline_by_family[family][FINAL_KEY + family_columns[family]], on=FINAL_KEY, validate="one_to_one")
    complexity_source = [f"{name.removeprefix('complexity_')}_{stat}" for name in complexity_base_features() for stat in AGGREGATIONS]
    complexity_loaded = _load_family(complexity, complexity_source, "complexity")
    _assert_same_rows(reference, complexity_loaded, "complexity")
    complexity_frame = complexity_loaded[FINAL_KEY + ["supervised_label"] + complexity_source].rename(columns=dict(zip(complexity_source, complexity_predictors(AGGREGATIONS))))
    baseline_final = _finalize(baseline, baseline_predictors())
    complexity_final = _finalize(complexity_frame, complexity_predictors(AGGREGATIONS))
    return baseline_final, complexity_final, _validation(baseline_final, baseline_predictors(), 96), _validation(complexity_final, complexity_predictors(AGGREGATIONS), 42)


def write_dataset(final: pd.DataFrame, validation: dict[str, Any], kind: str, output_dir: Path, repository: Path, inputs: dict[str, Path]) -> Path:
    if kind == "baseline":
        predictors, schema, qc_tables, filename, feature_file, qc_file = baseline_predictors(), baseline_feature_counts(), baseline_feature_qc, "ml_participant_channel_dataset.parquet", "feature_list.json", "dataset_qc_summary.json"
    else:
        predictors, schema, qc_tables, filename, feature_file, qc_file = complexity_predictors(AGGREGATIONS), complexity_feature_counts(AGGREGATIONS), lambda x: complexity_feature_qc(x, AGGREGATIONS), "ml_complexity_participant_channel_dataset.parquet", "complexity_feature_list.json", "complexity_dataset_qc_summary.json"
    with new_output_directory(output_dir) as staging:
        atomic_parquet(final, staging / filename)
        atomic_csv(final, staging / filename.replace(".parquet", ".csv"))
        atomic_json({"task": DETROIT_TASK, "predictors": predictors, **schema}, staging / feature_file)
        atomic_json({**validation, "feature_counts": schema, "counts": validation["counts"]}, staging / qc_file)
        for name, table in {**qc_tables(final), **cohort_count_tables(final)}.items():
            atomic_csv(table, staging / f"{name}.csv")
        atomic_json({"protocol_version": "ml_detroit_dataset_v1", "stage": f"detroit_{kind}_dataset_construction", "task": DETROIT_TASK, "timestamp_utc": timestamp_utc(), "repository": str(repository.resolve()), "git_commit": git_commit(repository), "counts": validation["counts"], "validation": validation, "input_provenance": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in inputs.items()}}, staging / f"{kind}_dataset_run_summary.json")
    return output_dir / filename


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relative-bandpower", type=Path, required=True); parser.add_argument("--time-domain", type=Path, required=True); parser.add_argument("--frequency-domain", type=Path, required=True); parser.add_argument("--complexity", type=Path, required=True)
    parser.add_argument("--baseline-output-dir", type=Path); parser.add_argument("--complexity-output-dir", type=Path); parser.add_argument("--repository", type=Path, default=REPOSITORY); parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    inputs = {"relative_bandpower": args.relative_bandpower.resolve(), "time_domain": args.time_domain.resolve(), "frequency_domain": args.frequency_domain.resolve(), "complexity": args.complexity.resolve()}
    baseline, complexity, baseline_validation, complexity_validation = build_datasets(**inputs)
    if args.validate_only:
        print(json.dumps({"baseline": baseline_validation, "complexity": complexity_validation}, indent=2, sort_keys=True)); return
    if args.baseline_output_dir is None or args.complexity_output_dir is None:
        raise ValueError("both output directories are required unless --validate-only is used")
    print(write_dataset(baseline, baseline_validation, "baseline", args.baseline_output_dir.resolve(), args.repository.resolve(), inputs))
    print(write_dataset(complexity, complexity_validation, "complexity", args.complexity_output_dir.resolve(), args.repository.resolve(), inputs))


if __name__ == "__main__":
    main()
