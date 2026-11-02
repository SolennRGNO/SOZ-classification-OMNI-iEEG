"""CLI runner for the fixed grouped-CV Complexity logistic evaluation.

This stage intentionally reuses the Baseline evaluator's scientific protocol.
Only the schema is different: 14 ComplexityFeatures aggregated by mean, std and
median, yielding exactly 42 supplied predictors.  By default this remains the
unchanged all-feature model B; C1 can explicitly request fold-local mRMR.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.ml.complexity_dataset import feature_counts, predictor_names
from scripts.ml.logistic_baseline import (
    N_BOOTSTRAP,
    REPOSITORY,
    run_logistic_evaluation,
    run_nested_elasticnet_evaluation,
)
from scripts.ml.feature_selection import MRMR_MI_PARAMETERS, MRMRSelector
from scripts.ml.nested_mrmr import MRMR_N_GRID, run_nested_mrmr_evaluation


COMPLEXITY_PROTOCOL_VERSION = "ml_logistic_complexity_v1"
COMPLEXITY_C1_PROTOCOL_VERSION = "ml_logistic_complexity_c1_v1"
COMPLEXITY_C1_NESTED_PROTOCOL_VERSION = "ml_logistic_complexity_c1_nested_v1"
COMPLEXITY_C2_PROTOCOL_VERSION = "ml_logistic_complexity_c2_v1"
COMPLEXITY_AGGREGATIONS = ("mean", "std", "median")
COMPLEXITY_PREDICTOR_COUNT = 42


def complexity_predictors() -> list[str]:
    predictors = predictor_names(COMPLEXITY_AGGREGATIONS)
    if len(predictors) != COMPLEXITY_PREDICTOR_COUNT:
        raise AssertionError(
            f"Complexity evaluation requires exactly {COMPLEXITY_PREDICTOR_COUNT} predictors"
        )
    return predictors


def run_complexity(
    dataset_path: Path,
    feature_list_path: Path,
    output_dir: Path,
    repository: Path = REPOSITORY,
    n_bootstrap: int = N_BOOTSTRAP,
    n_selected_features: int | None = None,
    experiment: str = "b",
) -> Path:
    """Run B by default, C1 with mRMR, C1-nested with tuned N, or C2 Elastic Net."""
    schema = feature_counts(COMPLEXITY_AGGREGATIONS)
    if schema["predictor_count"] != COMPLEXITY_PREDICTOR_COUNT:
        raise AssertionError("Complexity feature schema no longer has 42 predictors")
    experiment = str(experiment).strip().lower()
    if experiment not in {"b", "c1", "c1_nested", "c2"}:
        raise ValueError("experiment must be one of: b, c1, c1_nested, c2")
    if experiment == "c1_nested":
        if n_selected_features is not None:
            raise ValueError(
                "C1-nested tunes N inside each outer training fold and cannot be "
                "combined with a fixed n_selected_features"
            )
        return run_nested_mrmr_evaluation(
            dataset_path,
            feature_list_path,
            output_dir,
            expected_predictors=complexity_predictors(),
            feature_schema=schema,
            protocol_version=COMPLEXITY_C1_NESTED_PROTOCOL_VERSION,
            stage="nested_mrmr_logistic_complexity_c1_nested",
            repository=repository,
            n_bootstrap=n_bootstrap,
            n_grid=MRMR_N_GRID,
        )
    if experiment == "c2":
        if n_selected_features is not None:
            raise ValueError("C2 Elastic Net cannot be combined with C1 n_selected_features")
        return run_nested_elasticnet_evaluation(
            dataset_path,
            feature_list_path,
            output_dir,
            expected_predictors=complexity_predictors(),
            feature_schema=schema,
            protocol_version=COMPLEXITY_C2_PROTOCOL_VERSION,
            stage="nested_elasticnet_logistic_complexity_c2",
            repository=repository,
            n_bootstrap=n_bootstrap,
        )
    if experiment == "c1" and n_selected_features is None:
        raise ValueError("C1 requires n_selected_features")
    if n_selected_features is not None:
        try:
            is_positive_integer = (
                not isinstance(n_selected_features, bool)
                and int(n_selected_features) == n_selected_features
                and n_selected_features >= 1
            )
        except (TypeError, ValueError, OverflowError):
            is_positive_integer = False
        if not is_positive_integer:
            raise ValueError("n_selected_features must be a positive integer when supplied")
        n_selected_features = int(n_selected_features)
        selector_factory = lambda: MRMRSelector(n_selected_features)
        protocol_version = COMPLEXITY_C1_PROTOCOL_VERSION
        stage = "fixed_logistic_complexity_c1_mrmr"
        selection_summary = {
            "method": "mRMR mutual-information difference (MID)",
            "implementation": "scripts.ml.feature_selection.MRMRSelector",
            "dependency": "scikit-learn mutual_info_classif and mutual_info_regression",
            "mutual_information_estimator": MRMR_MI_PARAMETERS,
            "requested_n_features": n_selected_features,
            "fit_scope": "outer training rows only after fold-local StandardScaler",
            "test_data_seen_by_selector": False,
        }
    else:
        selector_factory = None
        protocol_version = COMPLEXITY_PROTOCOL_VERSION
        stage = "fixed_logistic_complexity"
        selection_summary = None
    return run_logistic_evaluation(
        dataset_path,
        feature_list_path,
        output_dir,
        expected_predictors=complexity_predictors(),
        feature_schema=schema,
        protocol_version=protocol_version,
        stage=stage,
        repository=repository,
        n_bootstrap=n_bootstrap,
        selector_factory=selector_factory,
        feature_selection_summary=selection_summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--feature-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument(
        "--n-selected-features",
        type=int,
        help="Enable C1 fold-local mRMR and select this many predictors per outer fold",
    )
    parser.add_argument(
        "--experiment",
        choices=("b", "c1", "c1_nested", "c2"),
        default="b",
        help=(
            "Complexity model: B (fixed L2), C1 (fixed-N mRMR + L2), "
            "C1-nested (inner-CV-tuned N mRMR + L2), or C2 (nested Elastic Net)"
        ),
    )
    args = parser.parse_args()
    print(
        run_complexity(
            args.dataset.resolve(), args.feature_list.resolve(), args.output_dir.resolve(),
            args.repository.resolve(), args.n_bootstrap, args.n_selected_features, args.experiment,
        )
    )


if __name__ == "__main__":
    main()
