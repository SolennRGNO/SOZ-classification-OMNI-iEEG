"""CLI runner for ``ml_logistic_complexity_c1_nested_tuned_v1``.

This experiment reuses the Complexity schema, cohort, labels, participant
groups and outer folds of the existing complexity experiments and adds joint
inner tuning of the number of mRMR predictors ``N`` and the L2 strength ``C``,
plus a PPV-maximizing threshold selected under a minimum-TPR constraint from
inner out-of-fold probabilities.  It writes only into its own experiment root
and never reads or writes any other experiment's results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.ml.common import DEFAULT_ANALYSIS_ROOT, TASKS, resolve_task
from scripts.ml.complexity_dataset import feature_counts, predictor_names
from scripts.ml.logistic_baseline import N_BOOTSTRAP, REPOSITORY
from scripts.ml.nested_tuned import (
    DEFAULT_MINIMUM_TPR,
    EXPERIMENT_ID,
    PROTOCOL_VERSION,
    REPORT_FILENAME,
    STAGE,
    TUNED_C_GRID,
    TUNED_N_GRID,
    run_nested_tuned_evaluation,
    tuned_task_output_root,
)


COMPLEXITY_AGGREGATIONS = ("mean", "std", "median")
COMPLEXITY_PREDICTOR_COUNT = 42
DATASET_FILENAME = "ml_complexity_participant_channel_dataset.parquet"
FEATURE_LIST_FILENAME = "complexity_feature_list.json"


def complexity_predictors() -> list[str]:
    predictors = predictor_names(COMPLEXITY_AGGREGATIONS)
    if len(predictors) != COMPLEXITY_PREDICTOR_COUNT:
        raise AssertionError(
            f"this experiment requires exactly {COMPLEXITY_PREDICTOR_COUNT} predictors"
        )
    return predictors


def default_paths(task: str, analysis_root: Path = DEFAULT_ANALYSIS_ROOT) -> dict[str, Path]:
    """Project path helpers only; no absolute cluster path is hard-coded here."""
    root = tuned_task_output_root(task, analysis_root)
    dataset_dir = root / "dataset"
    return {
        "root": root,
        "dataset_dir": dataset_dir,
        "dataset": dataset_dir / DATASET_FILENAME,
        "feature_list": dataset_dir / FEATURE_LIST_FILENAME,
        "results_dir": root / "results",
        "report": root / REPORT_FILENAME,
    }


def run_complexity_tuned(
    dataset_path: Path,
    feature_list_path: Path,
    output_dir: Path,
    repository: Path = REPOSITORY,
    n_bootstrap: int = N_BOOTSTRAP,
    minimum_tpr: float = DEFAULT_MINIMUM_TPR,
    reference_fold_assignments: Path | None = None,
) -> Path:
    """Run the tuned nested experiment on the frozen 42-predictor schema."""
    schema = feature_counts(COMPLEXITY_AGGREGATIONS)
    if schema["predictor_count"] != COMPLEXITY_PREDICTOR_COUNT:
        raise AssertionError("Complexity feature schema no longer has 42 predictors")
    return run_nested_tuned_evaluation(
        dataset_path,
        feature_list_path,
        output_dir,
        expected_predictors=complexity_predictors(),
        feature_schema=schema,
        protocol_version=PROTOCOL_VERSION,
        stage=STAGE,
        repository=repository,
        n_bootstrap=n_bootstrap,
        n_grid=TUNED_N_GRID,
        c_grid=TUNED_C_GRID,
        minimum_tpr=minimum_tpr,
        reference_fold_assignments=reference_fold_assignments,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASKS),
        help="recording state for this experiment; exactly one is allowed",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT,
        help="root under which this experiment's own output tree is created",
    )
    parser.add_argument(
        "--dataset", type=Path,
        help=f"defaults to <analysis root>/{EXPERIMENT_ID}/<task>/dataset/{DATASET_FILENAME}",
    )
    parser.add_argument(
        "--feature-list", type=Path,
        help=f"defaults to <analysis root>/{EXPERIMENT_ID}/<task>/dataset/{FEATURE_LIST_FILENAME}",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help=f"defaults to <analysis root>/{EXPERIMENT_ID}/<task>/results",
    )
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument(
        "--minimum-tpr",
        type=float,
        default=DEFAULT_MINIMUM_TPR,
        help=(
            "minimum inner out-of-fold TPR a candidate threshold must reach; the "
            "threshold search maximizes PPV subject to this constraint"
        ),
    )
    parser.add_argument(
        "--reference-fold-assignments",
        type=Path,
        help=(
            "optional fold_assignments.csv of an existing complexity experiment; when "
            "given, the run fails unless (participant_id, channel_id, fold_id) matches"
        ),
    )
    args = parser.parse_args()
    task = resolve_task(args.task)
    defaults = default_paths(task, args.analysis_root)
    dataset = (args.dataset or defaults["dataset"]).resolve()
    feature_list = (args.feature_list or defaults["feature_list"]).resolve()
    output_dir = (args.output_dir or defaults["results_dir"]).resolve()
    print(
        run_complexity_tuned(
            dataset,
            feature_list,
            output_dir,
            args.repository.resolve(),
            args.n_bootstrap,
            args.minimum_tpr,
            args.reference_fold_assignments.resolve()
            if args.reference_fold_assignments is not None else None,
        )
    )


if __name__ == "__main__":
    main()
