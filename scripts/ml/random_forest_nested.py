"""Nested-CV Random Forest evaluation (experiment ``ml_random_forest_complexity_nested_v1``).

This module answers one question only: does a nonlinear classifier improve
pathological-versus-non-pathological channel classification relative to the
existing logistic models?  To keep that comparison honest, every scientific
primitive that is not the estimator itself is imported verbatim from the
logistic experiments: the outer grouped folds, the participant/class sample
weighting, the metrics, the 0.5 decision threshold, the participant-cluster
bootstrap and the fold audits.

The only deliberate protocol differences are:

* no ``StandardScaler`` -- axis-aligned trees are invariant to monotone
  per-feature rescaling, so standardization would be inert machinery;
* no feature selection -- all supplied predictors enter every forest;
* the inner grouped CV tunes ``max_depth``, ``min_samples_leaf`` and
  ``max_features`` instead of a penalty or a number of selected features.

Feature importances (impurity and permutation) are published as post-hoc
descriptive analyses.  They are computed after the outer fold's model is frozen
and never enter tuning, fitting or thresholding.
"""

from __future__ import annotations

import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from scripts.ml.common import (
    atomic_csv,
    atomic_json,
    git_commit,
    new_output_directory,
    sha256,
    timestamp_utc,
)
from scripts.ml.dataset import FINAL_KEY

# The scientific core is shared with the logistic experiments on purpose; the
# leading-underscore helpers are internal to the ML package, not private API.
from scripts.ml.logistic_baseline import (
    BOOTSTRAP_RANDOM_STATE,
    CONFIDENCE_LEVEL,
    CV_RANDOM_STATE,
    DECISION_THRESHOLD,
    INNER_N_SPLITS,
    N_BOOTSTRAP,
    N_SPLITS,
    REPOSITORY,
    _fold_audit_row,
    _load_predictor_schema,
    _validate_predictor_schema,
    calculate_metrics,
    evaluation_tables,
    participant_and_class_weights,
)


# Frozen Random Forest configuration.  ``n_estimators`` is the only quantity a
# smoke test may lower; the scientific run must keep 500.
RF_N_ESTIMATORS = 500
RF_CRITERION = "gini"
RF_BOOTSTRAP = True
RF_CLASS_WEIGHT = None  # imbalance is handled by the shared sample weights
RF_MAX_DEPTH_GRID: tuple[int | None, ...] = (None, 5, 10, 20)
RF_MIN_SAMPLES_LEAF_GRID: tuple[int, ...] = (1, 5, 10, 20)
RF_MAX_FEATURES_GRID: tuple[object, ...] = ("sqrt", 0.3, 0.5, 1.0)
PERMUTATION_REPEATS = 20
PERMUTATION_SCORING = "average_precision"
TIE_BREAK_RULE = (
    "highest mean inner Average Precision; then smallest max_depth (None ranks last "
    "because it is unbounded); then largest min_samples_leaf; then most restrictive "
    "max_features (smallest fraction of predictors offered per split); then frozen grid order"
)


def resolve_n_jobs(n_jobs: int | None = None) -> int:
    """Resolve one parallelism level from the cluster's thread-control variables.

    Only the forest's tree fitting is parallelized.  Candidate configurations are
    evaluated sequentially, so the process never oversubscribes its cores.
    """
    if n_jobs is not None:
        resolved = int(n_jobs)
        if resolved == 0 or resolved < -1:
            raise ValueError("n_jobs must be -1 or a positive integer")
        return resolved
    for variable in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        value = os.environ.get(variable, "").strip()
        if value.isdigit() and int(value) > 0:
            return int(value)
    return 1


def max_features_fraction(candidate: object, n_features: int) -> float:
    """Fraction of predictors a split may consider; also the restrictiveness order."""
    if n_features < 1:
        raise ValueError("max_features requires at least one predictor")
    if isinstance(candidate, str):
        if candidate == "sqrt":
            return math.sqrt(n_features) / n_features
        if candidate == "log2":
            return math.log2(n_features) / n_features
        raise ValueError(f"unsupported max_features string: {candidate!r}")
    if isinstance(candidate, bool):
        raise ValueError("max_features must not be a boolean")
    if isinstance(candidate, (int, np.integer)):
        if int(candidate) < 1 or int(candidate) > n_features:
            raise ValueError(f"integer max_features outside 1..{n_features}: {candidate}")
        return int(candidate) / n_features
    fraction = float(candidate)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fractional max_features must lie in (0, 1]: {candidate}")
    return fraction


def _label(value: object) -> str:
    """Stable CSV/JSON label for a hyperparameter value (``None`` included)."""
    return "None" if value is None else str(value)


def _depth_order(candidate: int | None) -> float:
    return math.inf if candidate is None else float(candidate)


def validate_grids(
    max_depth_grid: Iterable[int | None],
    min_samples_leaf_grid: Iterable[int],
    max_features_grid: Iterable[object],
    n_features: int,
) -> tuple[tuple[int | None, ...], tuple[int, ...], tuple[object, ...]]:
    depths = tuple(max_depth_grid)
    leaves = tuple(min_samples_leaf_grid)
    features = tuple(max_features_grid)
    if not depths or not leaves or not features:
        raise ValueError("every Random Forest grid must be non-empty")
    for candidate in depths:
        if candidate is None:
            continue
        if isinstance(candidate, bool) or int(candidate) != candidate or int(candidate) < 1:
            raise ValueError(f"max_depth candidates must be None or positive integers: {candidate!r}")
    for candidate in leaves:
        if isinstance(candidate, bool) or int(candidate) != candidate or int(candidate) < 1:
            raise ValueError(f"min_samples_leaf candidates must be positive integers: {candidate!r}")
    for candidate in features:
        max_features_fraction(candidate, n_features)
    if len({_label(candidate) for candidate in depths}) != len(depths):
        raise ValueError("max_depth grid contains duplicates")
    if len(set(leaves)) != len(leaves):
        raise ValueError("min_samples_leaf grid contains duplicates")
    if len({_label(candidate) for candidate in features}) != len(features):
        raise ValueError("max_features grid contains duplicates")
    return (
        tuple(None if candidate is None else int(candidate) for candidate in depths),
        tuple(int(candidate) for candidate in leaves),
        features,
    )


def candidate_grid(
    max_depth_grid: Sequence[int | None],
    min_samples_leaf_grid: Sequence[int],
    max_features_grid: Sequence[object],
) -> list[dict[str, Any]]:
    """Full factorial grid in a frozen, reproducible order."""
    return [
        {"max_depth": depth, "min_samples_leaf": leaf, "max_features": features}
        for depth in max_depth_grid
        for leaf in min_samples_leaf_grid
        for features in max_features_grid
    ]


def make_random_forest(
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: object,
    n_estimators: int = RF_N_ESTIMATORS,
    n_jobs: int = 1,
) -> RandomForestClassifier:
    """Create the frozen estimator; sample weights are always passed to ``fit``."""
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        criterion=RF_CRITERION,
        max_depth=max_depth,
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        bootstrap=RF_BOOTSTRAP,
        class_weight=RF_CLASS_WEIGHT,
        random_state=CV_RANDOM_STATE,
        n_jobs=n_jobs,
    )
    if model.class_weight is not None:
        raise AssertionError(
            "the Random Forest must use explicit sample weights, not class_weight; "
            "class imbalance is already handled by the shared weighting procedure"
        )
    return model


def _pathological_scores(model: RandomForestClassifier, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, int(np.flatnonzero(model.classes_ == 1)[0])]


def _prepare_matrices(
    dataset: pd.DataFrame, predictors: Iterable[str]
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Validate the frozen schema and return the modelling matrices."""
    features = _validate_predictor_schema(dataset, predictors)
    required = set(FINAL_KEY + ["dataset", "task", "target"] + features)
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"ML dataset lacks required columns: {missing}")
    tasks = sorted(set(dataset["task"].astype(str).str.strip().str.lower()))
    if len(tasks) != 1:
        raise ValueError(f"ML dataset must hold exactly one task; found {tasks}")
    if dataset.duplicated(FINAL_KEY).any():
        raise ValueError("ML dataset has duplicate participant-by-channel rows")
    if set(dataset["target"].astype(int).unique()) != {0, 1}:
        raise ValueError("ML dataset must contain both classes")
    x = dataset[features].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("ML predictors contain NaN or infinity; no imputation is permitted")
    y = dataset["target"].to_numpy(dtype=int)
    groups = dataset["participant_id"].astype(str).to_numpy()
    return features, x, y, groups


def outer_splits(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The outer protocol of the complexity experiments, byte-for-byte.

    ``StratifiedGroupKFold`` partitions on ``y`` and ``groups`` only, so the
    assignment cannot depend on predictor values.  The check below turns that
    documented property into an executed assertion, which is what makes this
    experiment's outer folds identical to the logistic experiments' folds on the
    same dataset rows.
    """
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    splits = [(train, test) for train, test in splitter.split(x, y, groups)]
    reference = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    neutral = np.zeros_like(x)
    for (train, test), (neutral_train, neutral_test) in zip(
        splits, reference.split(neutral, y, groups)
    ):
        if not np.array_equal(train, neutral_train) or not np.array_equal(test, neutral_test):
            raise AssertionError(
                "the outer grouped folds depend on predictor values; this experiment would "
                "no longer be comparable with the existing complexity experiments"
            )
    return splits


def run_nested_random_forest_cross_validation(
    dataset: pd.DataFrame,
    predictors: Iterable[str],
    *,
    n_estimators: int = RF_N_ESTIMATORS,
    max_depth_grid: Iterable[int | None] = RF_MAX_DEPTH_GRID,
    min_samples_leaf_grid: Iterable[int] = RF_MIN_SAMPLES_LEAF_GRID,
    max_features_grid: Iterable[object] = RF_MAX_FEATURES_GRID,
    permutation_repeats: int = PERMUTATION_REPEATS,
    n_jobs: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Nested grouped CV: Random Forest hyperparameters tuned inside each outer TRAIN.

    The outer test fold never reaches the hyperparameter search, the sample
    weights, the fitting or the threshold.  Feature importances are computed only
    after the outer model is frozen and are descriptive output, never feedback.
    """
    features, x, y, groups = _prepare_matrices(dataset, predictors)
    depths, leaves, feature_fractions = validate_grids(
        max_depth_grid, min_samples_leaf_grid, max_features_grid, len(features)
    )
    grid = candidate_grid(depths, leaves, feature_fractions)
    if int(n_estimators) < 1:
        raise ValueError("n_estimators must be a positive integer")
    if int(permutation_repeats) < 1:
        raise ValueError("permutation_repeats must be a positive integer")
    n_estimators = int(n_estimators)
    permutation_repeats = int(permutation_repeats)
    jobs = resolve_n_jobs(n_jobs)

    splits = outer_splits(x, y, groups)
    fold_assignment = np.full(len(dataset), -1, dtype=int)
    fold_audit, predictions, fold_metrics = [], [], []
    weight_rows, factor_rows, status_rows = [], [], []
    tuning_rows, inner_audit_rows, summary_rows, chosen_rows = [], [], [], []
    impurity_rows, permutation_rows = [], []

    for fold, (outer_train_index, outer_test_index) in enumerate(splits, start=1):
        outer_train = dataset.iloc[outer_train_index].copy()
        outer_test = dataset.iloc[outer_test_index].copy()
        outer_train_ids = set(outer_train["participant_id"].astype(str))
        outer_test_ids = set(outer_test["participant_id"].astype(str))
        if outer_train_ids & outer_test_ids:
            raise AssertionError(f"fold {fold}: outer train/test participant overlap")
        if set(outer_train_index) & set(outer_test_index):
            raise AssertionError(f"fold {fold}: outer train/test row overlap")
        fold_audit.append(_fold_audit_row(fold, outer_train, outer_test))
        if set(y[outer_train_index]) != {0, 1} or set(y[outer_test_index]) != {0, 1}:
            raise ValueError(
                f"outer fold {fold} lacks both classes; the frozen grouped CV protocol "
                "was not changed and no model was fit for that fold"
            )

        # Inner indices are local to the outer-training partition, which makes it
        # impossible for an outer-test row or label to enter the tuning.
        inner_splitter = StratifiedGroupKFold(
            n_splits=INNER_N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE
        )
        try:
            inner_splits = list(
                inner_splitter.split(
                    x[outer_train_index], y[outer_train_index], groups[outer_train_index]
                )
            )
        except ValueError as error:
            raise ValueError(
                f"outer fold {fold}: a valid {INNER_N_SPLITS}-fold grouped inner CV "
                "cannot be constructed from its training data"
            ) from error
        if len(inner_splits) != INNER_N_SPLITS:
            raise ValueError(
                f"outer fold {fold}: expected exactly {INNER_N_SPLITS} inner grouped folds, "
                f"got {len(inner_splits)}"
            )

        candidate_scores: dict[int, list[float]] = {index: [] for index in range(len(grid))}
        invalid_candidates: dict[int, list[str]] = {index: [] for index in range(len(grid))}

        for inner_fold, (inner_train_local, inner_validation_local) in enumerate(inner_splits, start=1):
            inner_train_index = outer_train_index[inner_train_local]
            inner_validation_index = outer_train_index[inner_validation_local]
            inner_train_ids = set(dataset.iloc[inner_train_index]["participant_id"].astype(str))
            inner_validation_ids = set(
                dataset.iloc[inner_validation_index]["participant_id"].astype(str)
            )
            if inner_train_ids & inner_validation_ids:
                raise AssertionError(
                    f"outer fold {fold}, inner fold {inner_fold}: participant overlap"
                )
            if not inner_train_ids <= outer_train_ids or not inner_validation_ids <= outer_train_ids:
                raise AssertionError("inner split escaped the outer training fold")
            if (inner_train_ids | inner_validation_ids) & outer_test_ids:
                raise AssertionError("an outer-test participant reached the inner tuning CV")
            if set(inner_train_index) & set(outer_test_index) or set(inner_validation_index) & set(
                outer_test_index
            ):
                raise AssertionError("an outer-test row reached the inner tuning CV")
            if len(inner_train_index) == 0 or len(inner_validation_index) == 0:
                raise ValueError(
                    f"outer fold {fold}, inner fold {inner_fold}: a valid grouped inner CV "
                    "cannot be constructed because a split is empty"
                )
            if set(y[inner_train_index]) != {0, 1} or set(y[inner_validation_index]) != {0, 1}:
                raise ValueError(
                    f"outer fold {fold}, inner fold {inner_fold}: a valid grouped inner CV "
                    "cannot be constructed because both training and validation must contain both classes"
                )
            # Weights depend only on the inner-training partition, never on the
            # hyperparameters, so they are computed once for all candidates.
            inner_weights, _ = participant_and_class_weights(
                dataset.iloc[inner_train_index]["participant_id"], y[inner_train_index]
            )
            if set(inner_weights["participant_id"].astype(str)) != inner_train_ids:
                raise AssertionError("inner weights used a validation participant")
            inner_audit_rows.append(
                {
                    "outer_fold": fold,
                    "inner_fold": inner_fold,
                    "n_inner_train_rows": len(inner_train_index),
                    "n_inner_validation_rows": len(inner_validation_index),
                    "inner_train_row_positions": json.dumps(inner_train_index.tolist()),
                    "inner_validation_row_positions": json.dumps(inner_validation_index.tolist()),
                    "inner_train_participant_ids": json.dumps(sorted(inner_train_ids)),
                    "inner_validation_participant_ids": json.dumps(sorted(inner_validation_ids)),
                    "weight_fit_participant_ids": json.dumps(sorted(inner_train_ids)),
                    "inner_train_classes": json.dumps(sorted(set(y[inner_train_index].tolist()))),
                    "inner_validation_classes": json.dumps(
                        sorted(set(y[inner_validation_index].tolist()))
                    ),
                    "inner_train_validation_participant_overlap_count": len(
                        inner_train_ids & inner_validation_ids
                    ),
                    "outer_test_participant_overlap_count": len(
                        (inner_train_ids | inner_validation_ids) & outer_test_ids
                    ),
                    "outer_test_row_overlap_count": len(
                        (set(inner_train_index) | set(inner_validation_index))
                        & set(outer_test_index)
                    ),
                    "scaler_used": False,
                }
            )
            x_inner_train = x[inner_train_index]
            x_inner_validation = x[inner_validation_index]
            for index, candidate in enumerate(grid):
                model = make_random_forest(
                    candidate["max_depth"],
                    candidate["min_samples_leaf"],
                    candidate["max_features"],
                    n_estimators=n_estimators,
                    n_jobs=jobs,
                )
                try:
                    model.fit(
                        x_inner_train,
                        y[inner_train_index],
                        sample_weight=inner_weights["final_sample_weight"].to_numpy(dtype=float),
                    )
                    score = _pathological_scores(model, x_inner_validation)
                    average_precision = float(
                        average_precision_score(y[inner_validation_index], score)
                    )
                except Exception as error:  # fail loud: a failed candidate is a bug, not a result
                    raise RuntimeError(
                        f"outer fold {fold}, inner fold {inner_fold}: Random Forest candidate "
                        f"max_depth={_label(candidate['max_depth'])}, "
                        f"min_samples_leaf={candidate['min_samples_leaf']}, "
                        f"max_features={_label(candidate['max_features'])} failed: {error}"
                    ) from error
                valid = bool(math.isfinite(average_precision))
                if valid:
                    candidate_scores[index].append(average_precision)
                else:
                    invalid_candidates[index].append(
                        f"inner fold {inner_fold}: Average Precision is undefined"
                    )
                tuning_rows.append(
                    {
                        "outer_fold": fold,
                        "inner_fold": inner_fold,
                        "candidate_index": index,
                        "max_depth": _label(candidate["max_depth"]),
                        "min_samples_leaf": int(candidate["min_samples_leaf"]),
                        "max_features": _label(candidate["max_features"]),
                        "n_estimators": n_estimators,
                        "average_precision": average_precision,
                        "n_inner_train_rows": len(inner_train_index),
                        "n_inner_validation_rows": len(inner_validation_index),
                        "inner_train_validation_participant_overlap_count": len(
                            inner_train_ids & inner_validation_ids
                        ),
                        "outer_test_participant_overlap_count": len(
                            (inner_train_ids | inner_validation_ids) & outer_test_ids
                        ),
                        "status": "ok" if valid else "undefined_average_precision",
                        "valid_for_tuning": valid,
                    }
                )

        eligible = [
            (
                float(np.mean(candidate_scores[index])),
                _depth_order(grid[index]["max_depth"]),
                -int(grid[index]["min_samples_leaf"]),
                max_features_fraction(grid[index]["max_features"], len(features)),
                index,
            )
            for index in range(len(grid))
            if not invalid_candidates[index] and len(candidate_scores[index]) == INNER_N_SPLITS
        ]
        if not eligible:
            raise RuntimeError(
                f"outer fold {fold}: no Random Forest candidate produced a defined Average "
                f"Precision in all {INNER_N_SPLITS} inner folds; tuning is invalid"
            )
        # ``sorted`` states the documented tie-break rule explicitly: descending
        # score, then ascending max_depth (None last), then descending
        # min_samples_leaf, then ascending max_features fraction, then grid order.
        ordered = sorted(eligible, key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
        inner_mean_ap, _, _, _, best_index = ordered[0]
        best = grid[best_index]
        rank_by_index = {item[4]: rank for rank, item in enumerate(ordered, start=1)}
        for index, candidate in enumerate(grid):
            scores = candidate_scores[index]
            is_eligible = index in rank_by_index
            summary_rows.append(
                {
                    "outer_fold": fold,
                    "candidate_index": index,
                    "max_depth": _label(candidate["max_depth"]),
                    "min_samples_leaf": int(candidate["min_samples_leaf"]),
                    "max_features": _label(candidate["max_features"]),
                    "mean_inner_average_precision": (
                        float(np.mean(scores)) if is_eligible else math.nan
                    ),
                    "std_inner_average_precision": (
                        float(np.std(scores, ddof=1)) if is_eligible and len(scores) > 1 else math.nan
                    ),
                    "n_inner_folds_scored": len(scores),
                    "n_inner_folds": INNER_N_SPLITS,
                    "rank": rank_by_index.get(index, math.nan),
                    "selected_best": index == best_index,
                    "tie_break_rule": TIE_BREAK_RULE,
                }
            )

        # The hyperparameters are now frozen: the weights are recomputed from
        # scratch on the full outer training fold, and one final forest is fit.
        outer_weights, factors = participant_and_class_weights(
            outer_train["participant_id"], y[outer_train_index]
        )
        outer_weights.insert(1, "channel_id", outer_train["channel_id"].to_numpy())
        outer_weights.insert(0, "fold_id", fold)
        if set(outer_weights["participant_id"].astype(str)) != outer_train_ids:
            raise AssertionError("training weights were not computed solely from train participants")
        model = make_random_forest(
            best["max_depth"],
            best["min_samples_leaf"],
            best["max_features"],
            n_estimators=n_estimators,
            n_jobs=jobs,
        )
        started = time.perf_counter()
        model.fit(
            x[outer_train_index],
            y[outer_train_index],
            sample_weight=outer_weights["final_sample_weight"].to_numpy(dtype=float),
        )
        fit_seconds = time.perf_counter() - started
        if len(model.estimators_) != n_estimators:
            raise AssertionError("the fitted forest does not hold the requested number of trees")
        score = _pathological_scores(model, x[outer_test_index])
        predicted = (score >= DECISION_THRESHOLD).astype(int)

        # Post-hoc descriptive importances.  They are produced after the outer
        # model and its predictions are frozen and feed nothing.
        impurity_rows.extend(
            {
                "outer_fold": fold,
                "feature": feature,
                "impurity_importance": float(importance),
                "analysis": "post-hoc descriptive",
            }
            for feature, importance in zip(features, model.feature_importances_)
        )
        permutation = permutation_importance(
            model,
            x[outer_test_index],
            y[outer_test_index],
            scoring=PERMUTATION_SCORING,
            n_repeats=permutation_repeats,
            random_state=CV_RANDOM_STATE,
            n_jobs=1,
        )
        permutation_rows.extend(
            {
                "outer_fold": fold,
                "feature": feature,
                "permutation_importance_mean": float(mean),
                "permutation_importance_std": float(std),
                "scoring": PERMUTATION_SCORING,
                "n_repeats": permutation_repeats,
                "evaluated_on": "outer test fold only",
                "analysis": "post-hoc descriptive",
            }
            for feature, mean, std in zip(
                features, permutation.importances_mean, permutation.importances_std
            )
        )

        chosen_rows.append(
            {
                "outer_fold": fold,
                "selected_max_depth": _label(best["max_depth"]),
                "selected_min_samples_leaf": int(best["min_samples_leaf"]),
                "selected_max_features": _label(best["max_features"]),
                "mean_inner_average_precision": inner_mean_ap,
                "std_inner_average_precision": (
                    float(np.std(candidate_scores[best_index], ddof=1))
                    if len(candidate_scores[best_index]) > 1
                    else math.nan
                ),
                "n_candidates_evaluated": len(grid),
                "n_eligible_candidates": len(eligible),
                "n_estimators": n_estimators,
                "tie_break_rule": TIE_BREAK_RULE,
            }
        )
        status_rows.append(
            {
                "fold_id": fold,
                "status": "ok",
                "n_estimators": n_estimators,
                "n_trees_fitted": len(model.estimators_),
                "max_depth": _label(best["max_depth"]),
                "min_samples_leaf": int(best["min_samples_leaf"]),
                "max_features": _label(best["max_features"]),
                "criterion": RF_CRITERION,
                "bootstrap": RF_BOOTSTRAP,
                "class_weight": _label(RF_CLASS_WEIGHT),
                "sample_weight_used": True,
                "scaler_used": False,
                "n_train_rows": len(outer_train_index),
                "n_test_rows": len(outer_test_index),
                "fit_seconds": float(fit_seconds),
                "n_jobs": jobs,
            }
        )
        fold_assignment[outer_test_index] = fold
        fold_predictions = (
            outer_test[FINAL_KEY + ["dataset", "target"]]
            .copy()
            .rename(columns={"target": "true_label"})
        )
        fold_predictions["pathological_score"] = score
        fold_predictions["predicted_label"] = predicted
        fold_predictions["fold_id"] = fold
        predictions.append(fold_predictions)
        metrics = calculate_metrics(y[outer_test_index], score)
        fold_metrics.extend(
            {"fold_id": fold, "metric": name, "value": value} for name, value in metrics.items()
        )
        weight_rows.append(outer_weights)
        factor_rows.extend(
            {
                "fold_id": fold,
                "class": class_id,
                "class_factor": factor,
                "total_participant_weight_mass": float(outer_weights["participant_weight"].sum()),
                "class_participant_weight_mass": float(
                    outer_weights.loc[
                        outer_weights["true_label"].eq(class_id), "participant_weight"
                    ].sum()
                ),
                "final_class_weight_mass": float(
                    outer_weights.loc[
                        outer_weights["true_label"].eq(class_id), "final_sample_weight"
                    ].sum()
                ),
                "final_weight_min": float(outer_weights["final_sample_weight"].min()),
                "final_weight_median": float(outer_weights["final_sample_weight"].median()),
                "final_weight_mean": float(outer_weights["final_sample_weight"].mean()),
                "final_weight_max": float(outer_weights["final_sample_weight"].max()),
            }
            for class_id, factor in factors.items()
        )

    if (fold_assignment < 1).any():
        raise AssertionError("some OOF observations were never assigned to an outer test fold")
    oof = pd.concat(predictions, ignore_index=True)
    if len(oof) != len(dataset) or oof.duplicated(FINAL_KEY).any():
        raise AssertionError("every included observation must be predicted exactly once")
    assignments = dataset[FINAL_KEY + ["dataset", "target"]].copy()
    assignments["fold_id"] = fold_assignment
    impurity = pd.DataFrame(impurity_rows)
    permutation_table = pd.DataFrame(permutation_rows)
    return {
        "fold_assignments": assignments,
        "fold_audit": pd.DataFrame(fold_audit),
        "sample_weights": pd.concat(weight_rows, ignore_index=True),
        "class_factors": pd.DataFrame(factor_rows),
        "fit_status": pd.DataFrame(status_rows),
        "oof_predictions": oof.sort_values(FINAL_KEY, kind="stable").reset_index(drop=True),
        "fold_metrics": pd.DataFrame(fold_metrics),
        "rf_nested_tuning": pd.DataFrame(tuning_rows),
        "rf_nested_summary": pd.DataFrame(summary_rows),
        "rf_inner_fold_audit": pd.DataFrame(inner_audit_rows),
        "rf_selected_hyperparameters_by_outer_fold": pd.DataFrame(chosen_rows),
        "rf_impurity_importance_by_fold": impurity,
        "rf_permutation_importance_by_fold": permutation_table,
        "rf_feature_importance_summary": feature_importance_summary(
            impurity, permutation_table, features
        ),
    }


def feature_importance_summary(
    impurity: pd.DataFrame, permutation: pd.DataFrame, predictors: Sequence[str]
) -> pd.DataFrame:
    """Aggregate the two post-hoc importance families across outer folds."""
    impurity = impurity.copy()
    permutation = permutation.copy()
    impurity["impurity_rank"] = impurity.groupby("outer_fold")["impurity_importance"].rank(
        ascending=False, method="average"
    )
    permutation["permutation_rank"] = permutation.groupby("outer_fold")[
        "permutation_importance_mean"
    ].rank(ascending=False, method="average")
    impurity_summary = (
        impurity.groupby("feature", sort=False)
        .agg(
            mean_impurity_importance=("impurity_importance", "mean"),
            std_impurity_importance=("impurity_importance", "std"),
            mean_impurity_rank=("impurity_rank", "mean"),
        )
        .reset_index()
    )
    permutation_summary = (
        permutation.groupby("feature", sort=False)
        .agg(
            mean_permutation_importance=("permutation_importance_mean", "mean"),
            std_permutation_importance=("permutation_importance_mean", "std"),
            mean_permutation_rank=("permutation_rank", "mean"),
            n_outer_folds_positive_permutation=("permutation_importance_mean", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )
    summary = (
        pd.DataFrame({"feature": list(predictors), "feature_index": range(len(predictors))})
        .merge(impurity_summary, on="feature", how="left", validate="one_to_one")
        .merge(permutation_summary, on="feature", how="left", validate="one_to_one")
    )
    summary["n_outer_folds"] = N_SPLITS
    summary["analysis"] = "post-hoc descriptive; never used for tuning or selection"
    return summary.sort_values("mean_impurity_importance", ascending=False).reset_index(drop=True)


def run_nested_random_forest_evaluation(
    dataset_path: Path,
    feature_list_path: Path,
    output_dir: Path,
    *,
    expected_predictors: Iterable[str],
    feature_schema: dict[str, Any],
    protocol_version: str,
    stage: str,
    repository: Path = REPOSITORY,
    n_bootstrap: int = N_BOOTSTRAP,
    n_estimators: int = RF_N_ESTIMATORS,
    max_depth_grid: Iterable[int | None] = RF_MAX_DEPTH_GRID,
    min_samples_leaf_grid: Iterable[int] = RF_MIN_SAMPLES_LEAF_GRID,
    max_features_grid: Iterable[object] = RF_MAX_FEATURES_GRID,
    permutation_repeats: int = PERMUTATION_REPEATS,
    n_jobs: int | None = None,
) -> Path:
    """Publish the nested Random Forest artifacts into a fresh output directory."""
    _, schema_predictors = _load_predictor_schema(feature_list_path)
    expected = list(expected_predictors)
    if schema_predictors != expected:
        raise ValueError(
            "dataset feature order differs from the expected frozen schema "
            f"({len(expected)} predictors)"
        )
    dataset = pd.read_parquet(dataset_path)
    task = sorted(set(dataset["task"].astype(str).str.strip().str.lower()))[0]
    depths, leaves, feature_fractions = validate_grids(
        max_depth_grid, min_samples_leaf_grid, max_features_grid, len(schema_predictors)
    )
    jobs = resolve_n_jobs(n_jobs)
    cv_tables = run_nested_random_forest_cross_validation(
        dataset,
        schema_predictors,
        n_estimators=n_estimators,
        max_depth_grid=depths,
        min_samples_leaf_grid=leaves,
        max_features_grid=feature_fractions,
        permutation_repeats=permutation_repeats,
        n_jobs=jobs,
    )
    evaluation = evaluation_tables(cv_tables["oof_predictions"], n_bootstrap=n_bootstrap)
    chosen = cv_tables["rf_selected_hyperparameters_by_outer_fold"]
    assignments = cv_tables["fold_assignments"]
    with new_output_directory(output_dir) as staging:
        for name, table in {**cv_tables, **evaluation}.items():
            atomic_csv(table, staging / f"{name}.csv")
        summary = {
            "experiment_id": protocol_version,
            "protocol_version": protocol_version,
            "stage": stage,
            "task": task,
            "timestamp_utc": timestamp_utc(),
            "repository": str(repository.resolve()),
            "git_commit": git_commit(repository),
            "input_dataset": str(dataset_path.resolve()),
            "input_dataset_sha256": sha256(dataset_path),
            "feature_list": str(feature_list_path.resolve()),
            "feature_list_sha256": sha256(feature_list_path),
            "n_observations": len(dataset),
            "n_participants": int(dataset["participant_id"].nunique()),
            "n_predictors": len(schema_predictors),
            "predictor_order": schema_predictors,
            "primary_metric": "auroc",
            "complementary_imbalance_sensitive_metric": "average_precision_score",
            "outer_cv": {
                "class": "StratifiedGroupKFold",
                "n_splits": N_SPLITS,
                "shuffle": True,
                "random_state": CV_RANDOM_STATE,
                "groups": "participant_id",
                "stratification": "target",
            },
            "inner_cv": {
                "class": "StratifiedGroupKFold",
                "n_splits": INNER_N_SPLITS,
                "shuffle": True,
                "random_state": CV_RANDOM_STATE,
                "groups": "participant_id",
                "stratification": "target",
                "scope": "each outer training fold",
                "purpose": "Random Forest hyperparameter selection only",
            },
            "outer_fold_participant_assignment": {
                str(int(fold)): sorted(group["participant_id"].astype(str).unique().tolist())
                for fold, group in assignments.groupby("fold_id", sort=True)
            },
            "outer_folds_match_complexity_protocol": True,
            "normalization": "none; tree-based model, no StandardScaler is fit or applied",
            "standard_scaler_used": False,
            "weighting": (
                "inner-training-only and outer-training-only inverse participant channel count "
                "followed by equal participant-weight class mass and normalization to mean one, "
                "passed to RandomForestClassifier.fit(sample_weight=...)"
            ),
            "model": {
                "class": "RandomForestClassifier",
                "criterion": RF_CRITERION,
                "n_estimators": int(n_estimators),
                "bootstrap": RF_BOOTSTRAP,
                "class_weight": RF_CLASS_WEIGHT,
                "random_state": CV_RANDOM_STATE,
                "n_jobs": jobs,
            },
            "n_estimators": int(n_estimators),
            "scientific_full_configuration": int(n_estimators) == RF_N_ESTIMATORS,
            "hyperparameter_grid": {
                "max_depth": [None if depth is None else int(depth) for depth in depths],
                "min_samples_leaf": [int(leaf) for leaf in leaves],
                "max_features": [
                    candidate if isinstance(candidate, str) else float(candidate)
                    for candidate in feature_fractions
                ],
                "n_candidates": len(depths) * len(leaves) * len(feature_fractions),
            },
            "tuning_metric": "average_precision",
            "tie_breaking": TIE_BREAK_RULE,
            "selected_hyperparameters_by_outer_fold": {
                str(int(row["outer_fold"])): {
                    "max_depth": str(row["selected_max_depth"]),
                    "min_samples_leaf": int(row["selected_min_samples_leaf"]),
                    "max_features": str(row["selected_max_features"]),
                    "mean_inner_average_precision": float(row["mean_inner_average_precision"]),
                }
                for _, row in chosen.iterrows()
            },
            "hyperparameters_chosen_from_outer_performance": False,
            "decision_threshold": DECISION_THRESHOLD,
            "threshold_was_optimized": False,
            "feature_selection": False,
            "all_predictors_retained": True,
            "feature_counts": feature_schema,
            "imputation": False,
            "resampling": False,
            "dimensionality_reduction": False,
            "feature_importance": {
                "impurity": "RandomForestClassifier.feature_importances_ of each frozen outer model",
                "permutation": {
                    "scoring": PERMUTATION_SCORING,
                    "n_repeats": int(permutation_repeats),
                    "random_state": CV_RANDOM_STATE,
                    "evaluated_on": "outer test fold only",
                },
                "role": "post-hoc descriptive only; never used for tuning, selection or thresholding",
            },
            "bootstrap": {
                "unit": "participant",
                "refit_models": False,
                "n_replicates": n_bootstrap,
                "random_state": BOOTSTRAP_RANDOM_STATE,
                "confidence_level": CONFIDENCE_LEVEL,
                "interval": "percentile",
            },
            "software_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "all_folds_fit": bool(cv_tables["fit_status"]["status"].eq("ok").all()),
        }
        atomic_json(summary, staging / "model_run_summary.json")
    return output_dir / "oof_predictions.csv"
