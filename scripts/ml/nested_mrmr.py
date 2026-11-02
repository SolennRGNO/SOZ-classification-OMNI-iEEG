"""Nested-CV mRMR evaluation (Complexity C1-nested): inner selection of N.

C1 fixes the number of mRMR-selected predictors on the command line.  This
module keeps C1's estimator, protocol and audits verbatim and only replaces the
fixed ``N`` by a value chosen inside each outer training fold with a grouped
inner cross-validation.  Every scientific primitive (folds, scaler, weights,
metrics, bootstrap, mRMR) is imported from the existing modules so that the two
experiments differ solely in where ``N`` comes from.
"""

from __future__ import annotations

import json
import math
import platform
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
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
from scripts.ml.feature_selection import MRMR_MI_PARAMETERS, MRMRSelector

# The scientific core is shared with the fixed-N experiments on purpose; the
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
    fit_training_scaler,
    make_model,
    participant_and_class_weights,
)


# Frozen candidate numbers of aggregated complexity predictors (not of the 14
# raw complexity measures).  Tuned per outer fold, never across outer folds.
MRMR_N_GRID = (5, 8, 10, 15, 20, 30)
TIE_BREAK_RULE = "highest mean inner Average Precision; then smallest N"


def _validate_n_grid(n_grid: Iterable[int], n_available: int) -> tuple[int, ...]:
    grid = [int(candidate) for candidate in n_grid]
    if not grid or len(set(grid)) != len(grid):
        raise ValueError("the mRMR N grid must be a non-empty set of distinct values")
    if any(candidate < 1 for candidate in grid):
        raise ValueError("every mRMR N candidate must be a positive integer")
    if any(candidate > n_available for candidate in grid):
        raise ValueError(
            f"mRMR N candidates exceed the {n_available} available predictors: "
            f"{sorted(candidate for candidate in grid if candidate > n_available)}"
        )
    return tuple(sorted(grid))


def _fit_mrmr(x_train: np.ndarray, y_train: np.ndarray, n_features: int) -> MRMRSelector:
    """Fit the existing mRMR selector on train-only rows and audit its output."""
    selector = MRMRSelector(n_features)
    selector.fit(x_train, y_train)
    indices = np.asarray(selector.selected_indices_, dtype=int)
    if len(indices) != selector.actual_n_features_ or len(indices) == 0:
        raise AssertionError("selector did not report its selected training predictors")
    if (indices < 0).any() or (indices >= x_train.shape[1]).any():
        raise AssertionError("selector returned a predictor outside the frozen schema")
    if len(np.unique(indices)) != len(indices):
        raise AssertionError("selector returned duplicate predictors")
    if selector.fit_n_rows_ != x_train.shape[0]:
        raise AssertionError("selector was not fit on exactly the supplied training rows")
    return selector


def _fit_scored_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: Iterable[float],
    x_evaluate: np.ndarray,
) -> tuple[Any, np.ndarray, bool, str]:
    """Fit the frozen C1 L2 model and return its pathological scores."""
    model = make_model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train, sample_weight=sample_weight)
    convergence_warnings = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    converged = not convergence_warnings and int(model.n_iter_[0]) < model.max_iter
    message = " | ".join(str(item.message) for item in convergence_warnings)
    if not convergence_warnings and int(model.n_iter_[0]) >= model.max_iter:
        message = f"max_iter exhausted ({model.max_iter})"
    score = model.predict_proba(x_evaluate)[:, int(np.flatnonzero(model.classes_ == 1)[0])]
    return model, score, converged, message


def selection_frequencies(selected: pd.DataFrame, predictors: Sequence[str]) -> pd.DataFrame:
    """Per-predictor outer-fold selection frequency across the frozen schema."""
    observed = (
        selected.groupby(["feature", "feature_index"], sort=True)
        .agg(n_folds_selected=("fold_id", "nunique"))
        .reset_index()
    )
    # Zero-frequency predictors are retained: this is an audit of the whole
    # schema, not only of the predictors that happened to be selected.
    frequency = pd.DataFrame(
        {"feature": list(predictors), "feature_index": range(len(predictors))}
    ).merge(observed, on=["feature", "feature_index"], how="left", validate="one_to_one")
    frequency["n_folds_selected"] = frequency["n_folds_selected"].fillna(0).astype(int)
    frequency["n_outer_folds"] = N_SPLITS
    frequency["selection_frequency"] = frequency["n_folds_selected"] / N_SPLITS
    return frequency


def run_nested_mrmr_cross_validation(
    dataset: pd.DataFrame,
    predictors: Iterable[str],
    n_grid: Iterable[int] = MRMR_N_GRID,
) -> dict[str, pd.DataFrame]:
    """Run C1-nested grouped CV: N is chosen inside each outer training fold.

    The outer protocol is byte-for-byte the C1 protocol.  Within each outer
    training fold a 3-fold grouped inner CV scores every candidate ``N`` by mean
    Average Precision; ties are resolved deterministically by the smallest
    ``N``.  The outer test fold never reaches the scaler, the weights, the mRMR
    statistics, the choice of ``N`` or the final fit.
    """
    features = _validate_predictor_schema(dataset, predictors)
    grid = _validate_n_grid(n_grid, len(features))
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
    outer_splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE
    )
    splits = list(outer_splitter.split(x, y, groups))
    fold_assignment = np.full(len(dataset), -1, dtype=int)
    fold_audit, predictions, fold_metrics = [], [], []
    scaler_rows, weight_rows, factor_rows = [], [], []
    coefficient_rows, intercept_rows, convergence_rows = [], [], []
    selected_feature_rows, tuning_rows, summary_rows, chosen_rows = [], [], [], []

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

        # Inner indices are local to the outer-training partition, which makes
        # it impossible for an outer-test row or label to enter the tuning of N.
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
        candidate_scores: dict[int, list[float]] = {candidate: [] for candidate in grid}
        invalid_candidates: dict[int, list[str]] = {candidate: [] for candidate in grid}

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
            inner_scaler = fit_training_scaler(
                x[inner_train_index], inner_train_ids, inner_validation_ids
            )
            x_inner_train = inner_scaler.transform(x[inner_train_index])
            x_inner_validation = inner_scaler.transform(x[inner_validation_index])
            inner_weights, _ = participant_and_class_weights(
                dataset.iloc[inner_train_index]["participant_id"], y[inner_train_index]
            )
            if set(inner_weights["participant_id"].astype(str)) != inner_train_ids:
                raise AssertionError("inner weights used a validation participant")
            # Greedy MID selection is prefix-nested: step k of the greedy loop
            # does not depend on how many steps are requested, so the first N
            # picks of a rank-max(grid) fit are exactly the picks of a rank-N
            # fit on the same inner-training data.  One fit therefore reproduces
            # every candidate without any candidate seeing different rows.
            inner_selector = _fit_mrmr(x_inner_train, y[inner_train_index], max(grid))
            inner_ranking = np.asarray(inner_selector.selected_indices_, dtype=int)
            common_audit = {
                "outer_fold": fold,
                "inner_fold": inner_fold,
                "inner_train_row_positions": json.dumps(inner_train_index.tolist()),
                "inner_validation_row_positions": json.dumps(inner_validation_index.tolist()),
                "inner_train_participant_ids": json.dumps(sorted(inner_train_ids)),
                "inner_validation_participant_ids": json.dumps(sorted(inner_validation_ids)),
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
                "scaler_fit_participant_ids": json.dumps(sorted(inner_scaler.fit_participant_ids_)),
                "weight_fit_participant_ids": json.dumps(sorted(inner_train_ids)),
                "selector_fit_n_rows": int(inner_selector.fit_n_rows_),
            }
            for candidate in grid:
                selected_indices = inner_ranking[:candidate]
                if len(selected_indices) != candidate:
                    raise AssertionError(
                        f"outer fold {fold}, inner fold {inner_fold}: mRMR returned "
                        f"{len(selected_indices)} predictors for N={candidate}"
                    )
                _, score, converged, message = _fit_scored_model(
                    x_inner_train[:, selected_indices],
                    y[inner_train_index],
                    inner_weights["final_sample_weight"],
                    x_inner_validation[:, selected_indices],
                )
                if converged:
                    average_precision = float(
                        average_precision_score(y[inner_validation_index], score)
                    )
                    candidate_scores[candidate].append(average_precision)
                else:
                    average_precision = math.nan
                    invalid_candidates[candidate].append(f"inner fold {inner_fold}: {message}")
                tuning_rows.append(
                    {
                        **common_audit,
                        "N": candidate,
                        "average_precision": average_precision,
                        "selected_features": json.dumps(
                            [features[index] for index in selected_indices]
                        ),
                        "selected_feature_indices": json.dumps(selected_indices.tolist()),
                        "converged": converged,
                        "convergence_warning": message,
                        "valid_for_tuning": converged,
                    }
                )

        eligible = [
            (float(np.mean(scores)), candidate)
            for candidate, scores in candidate_scores.items()
            if not invalid_candidates[candidate] and len(scores) == INNER_N_SPLITS
        ]
        if not eligible:
            raise RuntimeError(
                f"outer fold {fold}: no candidate N converged across all "
                f"{INNER_N_SPLITS} inner folds; tuning is invalid"
            )
        # ``sorted`` makes the stated descending-score/ascending-N tie rule explicit.
        ordered = sorted(eligible, key=lambda item: (-item[0], item[1]))
        inner_mean_ap, best_n = ordered[0]
        rank_by_candidate = {candidate: rank for rank, (_, candidate) in enumerate(ordered, start=1)}
        for candidate in grid:
            scores = candidate_scores[candidate]
            is_eligible = candidate in rank_by_candidate
            summary_rows.append(
                {
                    "outer_fold": fold,
                    "N": candidate,
                    "mean_inner_average_precision": float(np.mean(scores)) if is_eligible else math.nan,
                    "std_inner_average_precision": (
                        float(np.std(scores, ddof=1)) if is_eligible and len(scores) > 1 else math.nan
                    ),
                    "n_inner_folds_scored": len(scores),
                    "n_inner_folds": INNER_N_SPLITS,
                    "all_inner_folds_converged": is_eligible,
                    "rank": rank_by_candidate.get(candidate, math.nan),
                    "selected_N_best": candidate == best_n,
                    "tie_break_rule": TIE_BREAK_RULE,
                }
            )

        # N_best is now frozen: the scaler, the mRMR statistics and the weights
        # are all recomputed from scratch on the full outer training fold.
        outer_scaler = fit_training_scaler(x[outer_train_index], outer_train_ids, outer_test_ids)
        if (
            set(outer_scaler.fit_participant_ids_) != outer_train_ids
            or set(outer_scaler.fit_participant_ids_) & outer_test_ids
        ):
            raise AssertionError("scaler audit contains held-out participants")
        x_outer_train = outer_scaler.transform(x[outer_train_index])
        x_outer_test = outer_scaler.transform(x[outer_test_index])
        outer_selector = _fit_mrmr(x_outer_train, y[outer_train_index], best_n)
        selected_indices = np.asarray(outer_selector.selected_indices_, dtype=int)
        if outer_selector.fit_n_rows_ != len(outer_train_index):
            raise AssertionError("selector was not fit on exactly the outer training rows")
        outer_weights, factors = participant_and_class_weights(
            outer_train["participant_id"], y[outer_train_index]
        )
        outer_weights.insert(1, "channel_id", outer_train["channel_id"].to_numpy())
        outer_weights.insert(0, "fold_id", fold)
        if set(outer_weights["participant_id"].astype(str)) != outer_train_ids:
            raise AssertionError("training weights were not computed solely from train participants")
        model, score, converged, message = _fit_scored_model(
            x_outer_train[:, selected_indices],
            y[outer_train_index],
            outer_weights["final_sample_weight"],
            x_outer_test[:, selected_indices],
        )
        predicted = (score >= DECISION_THRESHOLD).astype(int)
        selected_names = [features[index] for index in selected_indices]
        selected_feature_rows.extend(
            {
                "fold_id": fold,
                "selection_rank": rank,
                "feature": features[feature_index],
                "feature_index": int(feature_index),
                "selected_n": int(best_n),
                "requested_n_features": int(outer_selector.requested_n_features),
                "actual_n_features": int(outer_selector.actual_n_features_),
                "n_available_predictors": len(features),
                "selector_fit_n_rows": int(outer_selector.fit_n_rows_),
                "selector_fit_row_positions": json.dumps(outer_train_index.tolist()),
                "outer_test_row_positions": json.dumps(outer_test_index.tolist()),
                "fit_test_row_overlap_count": int(
                    len(set(outer_train_index) & set(outer_test_index))
                ),
                "selector_fit_participant_ids": json.dumps(sorted(outer_train_ids)),
                "outer_test_participant_ids": json.dumps(sorted(outer_test_ids)),
                "fit_test_participant_overlap_count": int(len(outer_train_ids & outer_test_ids)),
            }
            for rank, feature_index in enumerate(selected_indices, start=1)
        )
        chosen_rows.append(
            {
                "outer_fold": fold,
                "selected_N": int(best_n),
                "actual_n_features": int(outer_selector.actual_n_features_),
                "mean_inner_average_precision": inner_mean_ap,
                "std_inner_average_precision": (
                    float(np.std(candidate_scores[best_n], ddof=1))
                    if len(candidate_scores[best_n]) > 1
                    else math.nan
                ),
                "n_candidates_evaluated": len(grid),
                "n_eligible_candidates": len(eligible),
                "tie_break_rule": TIE_BREAK_RULE,
                "selected_features": json.dumps(selected_names),
                "converged": converged,
                "convergence_warning": message,
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
        scaler_rows.extend(
            {
                "fold_id": fold,
                "feature": feature,
                "mean": float(mean),
                "scale": float(scale),
                "fit_participant_ids": json.dumps(sorted(outer_train_ids)),
            }
            for feature, mean, scale in zip(features, outer_scaler.mean_, outer_scaler.scale_)
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
        coefficient_rows.extend(
            {"fold_id": fold, "feature": feature, "coefficient": float(coefficient)}
            for feature, coefficient in zip(selected_names, model.coef_[0])
        )
        intercept_rows.append({"fold_id": fold, "intercept": float(model.intercept_[0])})
        convergence_rows.append(
            {
                "fold_id": fold,
                "solver": model.solver,
                "max_iter": model.max_iter,
                "selected_N": int(best_n),
                "n_iter": int(model.n_iter_[0]),
                "converged": converged,
                "convergence_warning": message,
            }
        )

    if (fold_assignment < 1).any():
        raise AssertionError("some OOF observations were never assigned to an outer test fold")
    oof = pd.concat(predictions, ignore_index=True)
    if len(oof) != len(dataset) or oof.duplicated(FINAL_KEY).any():
        raise AssertionError("every included observation must be predicted exactly once")
    assignments = dataset[FINAL_KEY + ["dataset", "target"]].copy()
    assignments["fold_id"] = fold_assignment
    return {
        "fold_assignments": assignments,
        "fold_audit": pd.DataFrame(fold_audit),
        "scaler_parameters": pd.DataFrame(scaler_rows),
        "sample_weights": pd.concat(weight_rows, ignore_index=True),
        "class_factors": pd.DataFrame(factor_rows),
        "model_coefficients": pd.DataFrame(coefficient_rows),
        "model_intercepts": pd.DataFrame(intercept_rows),
        "convergence": pd.DataFrame(convergence_rows),
        "oof_predictions": oof.sort_values(FINAL_KEY, kind="stable").reset_index(drop=True),
        "fold_metrics": pd.DataFrame(fold_metrics),
        "selected_features_by_fold": pd.DataFrame(selected_feature_rows),
        "mrmr_nested_tuning": pd.DataFrame(tuning_rows),
        "mrmr_nested_summary": pd.DataFrame(summary_rows),
        "mrmr_selected_n_by_outer_fold": pd.DataFrame(chosen_rows),
    }


def run_nested_mrmr_evaluation(
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
    n_grid: Iterable[int] = MRMR_N_GRID,
) -> Path:
    """Publish C1-nested artifacts plus the standard metric/bootstrap artifacts."""
    _, schema_predictors = _load_predictor_schema(feature_list_path)
    expected = list(expected_predictors)
    if schema_predictors != expected:
        raise ValueError(
            "dataset feature order differs from the expected frozen schema "
            f"({len(expected)} predictors)"
        )
    dataset = pd.read_parquet(dataset_path)
    task = sorted(set(dataset["task"].astype(str).str.strip().str.lower()))[0]
    grid = _validate_n_grid(n_grid, len(schema_predictors))
    cv_tables = run_nested_mrmr_cross_validation(dataset, schema_predictors, n_grid=grid)
    cv_tables["selection_frequencies"] = selection_frequencies(
        cv_tables["selected_features_by_fold"], schema_predictors
    )
    evaluation = evaluation_tables(cv_tables["oof_predictions"], n_bootstrap=n_bootstrap)
    selected_n = cv_tables["mrmr_selected_n_by_outer_fold"]
    with new_output_directory(output_dir) as staging:
        for name, table in {**cv_tables, **evaluation}.items():
            atomic_csv(table, staging / f"{name}.csv")
        summary = {
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
            },
            "inner_cv": {
                "class": "StratifiedGroupKFold",
                "n_splits": INNER_N_SPLITS,
                "shuffle": True,
                "random_state": CV_RANDOM_STATE,
                "groups": "participant_id",
                "scope": "each outer training fold",
            },
            "normalization": (
                "StandardScaler fit independently on each inner-training split during the "
                "selection of N and on each full outer training fold for final fitting"
            ),
            "weighting": (
                "inner-training-only and outer-training-only inverse participant channel count "
                "followed by equal participant-weight class mass and normalization to mean one"
            ),
            "model": {
                "class": "LogisticRegression",
                "penalty": "l2",
                "C": 1.0,
                "class_weight": None,
                "solver": "lbfgs",
                "max_iter": 5000,
            },
            "feature_selection": True,
            "all_predictors_retained": False,
            "feature_selection_details": {
                "method": "mRMR mutual-information difference (MID)",
                "implementation": "scripts.ml.feature_selection.MRMRSelector",
                "dependency": "scikit-learn mutual_info_classif and mutual_info_regression",
                "mutual_information_estimator": MRMR_MI_PARAMETERS,
                "n_grid": list(grid),
                "n_selection": "nested grouped inner CV inside each outer training fold",
                "tuning_metric": "average_precision",
                "tie_breaking": TIE_BREAK_RULE,
                "fit_scope": (
                    "inner-training rows only while tuning N, then outer-training rows only "
                    "for the final selection, both after their own StandardScaler"
                ),
                "test_data_seen_by_selector": False,
                "selected_n_by_outer_fold": {
                    str(int(row["outer_fold"])): int(row["selected_N"])
                    for _, row in selected_n.iterrows()
                },
                "selected_n_unique": sorted(set(selected_n["selected_N"].astype(int).tolist())),
            },
            "hyperparameter_grid": {"n_selected_features": list(grid)},
            "tuning_metric": "average_precision",
            "tie_breaking": TIE_BREAK_RULE,
            "final_n_chosen_from_outer_performance": False,
            "decision_threshold": DECISION_THRESHOLD,
            "threshold_was_optimized": False,
            "feature_counts": feature_schema,
            "imputation": False,
            "resampling": False,
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
            "all_folds_converged": bool(cv_tables["convergence"]["converged"].all()),
        }
        atomic_json(summary, staging / "model_run_summary.json")
    return output_dir / "oof_predictions.csv"
