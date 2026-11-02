"""CLI runner for the nested Random Forest Complexity experiment.

Experiment ID: ``ml_random_forest_complexity_nested_v1``.

The cohort, labels, participant-by-channel observations and the 42 aggregated
complexity predictors are exactly those of the existing Complexity experiments;
only the estimator and its nested hyperparameter search differ.  No feature
selection and no scaling take place before the forest.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from scripts.ml.complexity_dataset import feature_counts
from scripts.ml.logistic_baseline import N_BOOTSTRAP, REPOSITORY
from scripts.ml.logistic_complexity import (
    COMPLEXITY_AGGREGATIONS,
    COMPLEXITY_PREDICTOR_COUNT,
    complexity_predictors,
)
from scripts.ml.random_forest_nested import (
    PERMUTATION_REPEATS,
    RF_MAX_DEPTH_GRID,
    RF_MAX_FEATURES_GRID,
    RF_MIN_SAMPLES_LEAF_GRID,
    RF_N_ESTIMATORS,
    run_nested_random_forest_evaluation,
)


RANDOM_FOREST_PROTOCOL_VERSION = "ml_random_forest_complexity_nested_v1"
RANDOM_FOREST_STAGE = "nested_random_forest_complexity"


def run_random_forest_complexity(
    dataset_path: Path,
    feature_list_path: Path,
    output_dir: Path,
    repository: Path = REPOSITORY,
    n_bootstrap: int = N_BOOTSTRAP,
    n_estimators: int = RF_N_ESTIMATORS,
    permutation_repeats: int = PERMUTATION_REPEATS,
    n_jobs: int | None = None,
    *,
    max_depth_grid: Iterable[int | None] = RF_MAX_DEPTH_GRID,
    min_samples_leaf_grid: Iterable[int] = RF_MIN_SAMPLES_LEAF_GRID,
    max_features_grid: Iterable[object] = RF_MAX_FEATURES_GRID,
) -> Path:
    """Run the nested Random Forest on all 42 aggregated complexity predictors.

    The grids are keyword-only and default to the frozen protocol grid; the CLI
    never overrides them, so a scientific run cannot silently tune a different
    hyperparameter space.
    """
    schema = feature_counts(COMPLEXITY_AGGREGATIONS)
    if schema["predictor_count"] != COMPLEXITY_PREDICTOR_COUNT:
        raise AssertionError("Complexity feature schema no longer has 42 predictors")
    return run_nested_random_forest_evaluation(
        dataset_path,
        feature_list_path,
        output_dir,
        expected_predictors=complexity_predictors(),
        feature_schema=schema,
        protocol_version=RANDOM_FOREST_PROTOCOL_VERSION,
        stage=RANDOM_FOREST_STAGE,
        repository=repository,
        n_bootstrap=n_bootstrap,
        n_estimators=n_estimators,
        max_depth_grid=max_depth_grid,
        min_samples_leaf_grid=min_samples_leaf_grid,
        max_features_grid=max_features_grid,
        permutation_repeats=permutation_repeats,
        n_jobs=n_jobs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--feature-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=RF_N_ESTIMATORS,
        help="Trees per forest; the scientific run must keep the default of 500",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=PERMUTATION_REPEATS,
        help="Repeats of the post-hoc permutation importance on each outer test fold",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Tree-fitting parallelism; defaults to SLURM_CPUS_PER_TASK or OMP_NUM_THREADS. "
            "Candidate configurations are always evaluated sequentially."
        ),
    )
    args = parser.parse_args()
    print(
        run_random_forest_complexity(
            args.dataset.resolve(),
            args.feature_list.resolve(),
            args.output_dir.resolve(),
            args.repository.resolve(),
            args.n_bootstrap,
            args.n_estimators,
            args.permutation_repeats,
            args.n_jobs,
        )
    )


if __name__ == "__main__":
    main()
