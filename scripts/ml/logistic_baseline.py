"""Scientific core and CLI for the fixed grouped-CV logistic baseline."""

from __future__ import annotations

import argparse
import json
import math
import platform
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from scripts.ml.common import (
    PROTOCOL_VERSION,
    atomic_csv,
    atomic_json,
    git_commit,
    new_output_directory,
    sha256,
    timestamp_utc,
)
from scripts.ml.dataset import FINAL_KEY, feature_counts, predictor_names
from scripts.ml.feature_selection import FeatureSelector


REPOSITORY = Path(__file__).resolve().parents[2]
CV_RANDOM_STATE = 42
N_SPLITS = 5
INNER_N_SPLITS = 3
DECISION_THRESHOLD = 0.5
N_BOOTSTRAP = 5000
BOOTSTRAP_RANDOM_STATE = 42
CONFIDENCE_LEVEL = 0.95
METRIC_NAMES = (
    "auroc",
    "average_precision",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "balanced_accuracy",
)
ELASTICNET_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
ELASTICNET_L1_RATIO_GRID = (0.1, 0.25, 0.5, 0.75, 0.9)


def make_model() -> LogisticRegression:
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight=None,
        solver="lbfgs",
        max_iter=5000,
    )
    if model.class_weight is not None:
        raise AssertionError("baseline must use explicit sample weights, not class_weight")
    return model


def make_elasticnet_model(C: float, l1_ratio: float) -> LogisticRegression:
    """Create the frozen C2 estimator (weights are passed explicitly to fit)."""
    return LogisticRegression(
        penalty="elasticnet",
        C=C,
        l1_ratio=l1_ratio,
        class_weight=None,
        solver="saga",
        max_iter=10000,
        random_state=CV_RANDOM_STATE,
    )


def _elasticnet_convergence_status(
    model: LogisticRegression, caught: list[warnings.WarningMessage]
) -> tuple[bool, str]:
    """Return strict convergence status for a fitted C2 Elastic Net model."""
    convergence_warnings = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    exhausted = int(model.n_iter_[0]) >= model.max_iter
    messages = [str(item.message) for item in convergence_warnings]
    if exhausted and not messages:
        messages.append(f"max_iter exhausted ({model.max_iter})")
    return not convergence_warnings and not exhausted, " | ".join(messages)


def participant_and_class_weights(
    participant_ids: Iterable[object], labels: Iterable[int]
) -> tuple[pd.DataFrame, dict[int, float]]:
    participants = pd.Series(list(participant_ids), dtype="string", name="participant_id")
    target = pd.Series(list(labels), dtype="int64", name="true_label")
    if len(participants) != len(target) or len(target) == 0:
        raise ValueError("training participant IDs and labels must be non-empty and aligned")
    if set(target.unique()) != {0, 1}:
        raise ValueError("both classes are required in every training fold")
    channel_counts = participants.value_counts()
    participant_weight = participants.map((1.0 / channel_counts).to_dict()).astype(float)
    mass_by_participant = participant_weight.groupby(participants).sum()
    if not np.allclose(mass_by_participant.to_numpy(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("participant weights do not give equal pre-class-correction mass")
    total_mass = float(participant_weight.sum())
    class_mass = {
        class_id: float(participant_weight[target.eq(class_id)].sum()) for class_id in (0, 1)
    }
    if any(mass <= 0.0 for mass in class_mass.values()):
        raise ValueError("both classes must have positive participant-weight mass")
    factors = {
        class_id: total_mass / (2.0 * class_mass[class_id]) for class_id in (0, 1)
    }
    class_factor = target.map(factors).astype(float)
    final_weight = participant_weight * class_factor
    final_weight = final_weight / float(final_weight.mean())
    if not math.isclose(float(final_weight.mean()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("final sample weights do not have mean one")
    final_mass = {
        class_id: float(final_weight[target.eq(class_id)].sum()) for class_id in (0, 1)
    }
    if not math.isclose(final_mass[0], final_mass[1], rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("final sample weights do not give equal class mass")
    return (
        pd.DataFrame(
            {
                "participant_id": participants,
                "true_label": target,
                "participant_weight": participant_weight,
                "class_factor": class_factor,
                "final_sample_weight": final_weight,
            }
        ),
        factors,
    )


def fit_training_scaler(
    x_train: np.ndarray,
    train_participants: Iterable[object],
    test_participants: Iterable[object],
) -> StandardScaler:
    train_ids = set(map(str, train_participants))
    test_ids = set(map(str, test_participants))
    if train_ids & test_ids:
        raise AssertionError("participant leakage before scaler fitting")
    scaler = StandardScaler()
    scaler.fit(x_train)
    if not np.allclose(scaler.mean_, np.mean(x_train, axis=0), rtol=1e-12, atol=1e-12):
        raise AssertionError("scaler parameters did not originate exclusively from X_train")
    scaler.fit_participant_ids_ = tuple(sorted(train_ids))
    return scaler


def calculate_metrics(y_true: Iterable[int], scores: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    pathological_score = np.asarray(list(scores), dtype=float)
    if y.size != pathological_score.size or y.size == 0:
        raise ValueError("metric inputs must be non-empty and aligned")
    predicted = (pathological_score >= DECISION_THRESHOLD).astype(int)
    classes = set(y.tolist())
    true_positive = int(((y == 1) & (predicted == 1)).sum())
    false_positive = int(((y == 0) & (predicted == 1)).sum())
    false_negative = int(((y == 1) & (predicted == 0)).sum())
    precision_denominator = true_positive + false_positive
    f1_denominator = 2 * true_positive + false_positive + false_negative
    metrics = {
        "auroc": float(roc_auc_score(y, pathological_score)) if classes == {0, 1} else math.nan,
        "average_precision": (
            float(average_precision_score(y, pathological_score)) if classes == {0, 1} else math.nan
        ),
        "sensitivity": (
            float(recall_score(y, predicted, pos_label=1, zero_division=0)) if 1 in classes else math.nan
        ),
        "specificity": (
            float(recall_score(y, predicted, pos_label=0, zero_division=0)) if 0 in classes else math.nan
        ),
        "precision": (
            float(precision_score(y, predicted, pos_label=1, zero_division=0))
            if precision_denominator > 0 else math.nan
        ),
        "f1": (
            float(f1_score(y, predicted, pos_label=1, zero_division=0))
            if f1_denominator > 0 else math.nan
        ),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y, predicted)) if classes == {0, 1} else math.nan
        ),
    }
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    metrics.update(
        true_negative=float(tn), false_positive=float(fp),
        false_negative=float(fn), true_positive=float(tp),
    )
    return metrics


def resample_participant_clusters(
    predictions: pd.DataFrame, sampled_participants: Iterable[str]
) -> pd.DataFrame:
    """Retain every channel for every sampled participant occurrence."""
    pieces = []
    for occurrence, participant in enumerate(sampled_participants):
        cluster = predictions[predictions["participant_id"].astype(str).eq(str(participant))].copy()
        if cluster.empty:
            raise ValueError(f"bootstrap participant is absent: {participant}")
        cluster["bootstrap_occurrence"] = occurrence
        pieces.append(cluster)
    return pd.concat(pieces, ignore_index=True)


def cluster_bootstrap_metrics(
    predictions: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = BOOTSTRAP_RANDOM_STATE,
) -> pd.DataFrame:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    participants = np.asarray(sorted(predictions["participant_id"].astype(str).unique()))
    if participants.size == 0:
        raise ValueError("bootstrap requires participants")
    generator = np.random.default_rng(random_state)
    rows = []
    for replicate in range(n_bootstrap):
        sampled = generator.choice(participants, size=len(participants), replace=True)
        sample = resample_participant_clusters(predictions, sampled)
        metrics = calculate_metrics(sample["true_label"], sample["pathological_score"])
        for metric in METRIC_NAMES:
            rows.append(
                {
                    "bootstrap_replicate": replicate,
                    "metric": metric,
                    "value": metrics[metric],
                    "defined": bool(math.isfinite(metrics[metric])),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_confidence_intervals(distribution: pd.DataFrame) -> pd.DataFrame:
    alpha = 1.0 - CONFIDENCE_LEVEL
    rows = []
    for metric, group in distribution.groupby("metric", sort=False):
        valid = pd.to_numeric(group.loc[group["defined"], "value"], errors="coerce").dropna()
        count = len(valid)
        rows.append(
            {
                "metric": metric,
                "confidence_level": CONFIDENCE_LEVEL,
                "lower_percentile": float(valid.quantile(alpha / 2.0)) if count else math.nan,
                "upper_percentile": float(valid.quantile(1.0 - alpha / 2.0)) if count else math.nan,
                "n_valid_replicates": count,
                "n_total_replicates": len(group),
                "proportion_valid": count / len(group),
            }
        )
    return pd.DataFrame(rows)


def _metric_rows(
    predictions: pd.DataFrame, scope: str, scope_value: str
) -> list[dict[str, Any]]:
    metrics = calculate_metrics(predictions["true_label"], predictions["pathological_score"])
    return [
        {
            "scope": scope,
            "scope_value": scope_value,
            "metric": metric,
            "value": value,
            "n_channels": len(predictions),
            "n_participants": predictions["participant_id"].nunique(),
        }
        for metric, value in metrics.items()
    ]


def evaluation_tables(
    predictions: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP
) -> dict[str, pd.DataFrame]:
    metric_rows = _metric_rows(predictions, "overall", "all")
    for site, group in predictions.groupby("dataset", sort=True):
        metric_rows.extend(_metric_rows(group, "site", str(site)))
    overall_and_site = pd.DataFrame(metric_rows)

    participant_rows = []
    for (participant, site), group in predictions.groupby(["participant_id", "dataset"], sort=True):
        metrics = calculate_metrics(group["true_label"], group["pathological_score"])
        participant_rows.append(
            {
                "participant_id": participant,
                "dataset": site,
                "n_channels": len(group),
                "pathological": int(group["true_label"].eq(1).sum()),
                "non_pathological": int(group["true_label"].eq(0).sum()),
                **metrics,
            }
        )
    participant_metrics = pd.DataFrame(participant_rows)
    summary_rows = []
    for metric in METRIC_NAMES:
        valid = pd.to_numeric(participant_metrics[metric], errors="coerce").dropna()
        summary_rows.append(
            {
                "metric": metric,
                "mean_across_defined_participants": float(valid.mean()) if len(valid) else math.nan,
                "median_across_defined_participants": float(valid.median()) if len(valid) else math.nan,
                "n_participants_defined": len(valid),
                "n_participants_total": len(participant_metrics),
            }
        )
    participant_summary = pd.DataFrame(summary_rows)

    bootstrap_parts = []
    ci_parts = []
    scopes = [("overall", "all", predictions)] + [
        ("site", str(site), group) for site, group in predictions.groupby("dataset", sort=True)
    ]
    for scope, value, group in scopes:
        distribution = cluster_bootstrap_metrics(group, n_bootstrap=n_bootstrap)
        distribution.insert(0, "scope_value", value)
        distribution.insert(0, "scope", scope)
        confidence = bootstrap_confidence_intervals(distribution)
        confidence.insert(0, "scope_value", value)
        confidence.insert(0, "scope", scope)
        bootstrap_parts.append(distribution)
        ci_parts.append(confidence)
    return {
        "overall_site_metrics": overall_and_site,
        "participant_metrics": participant_metrics,
        "participant_metric_summary": participant_summary,
        "bootstrap_distribution": pd.concat(bootstrap_parts, ignore_index=True),
        "bootstrap_confidence_intervals": pd.concat(ci_parts, ignore_index=True),
    }


def _fold_audit_row(
    fold: int, train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, Any]:
    def site_participants(frame: pd.DataFrame, site: str) -> int:
        return int(frame.loc[frame["dataset"].eq(site), "participant_id"].nunique())

    def channels_per_participant(frame: pd.DataFrame) -> str:
        return json.dumps(
            frame.groupby("participant_id").size().sort_index().astype(int).to_dict(),
            sort_keys=True,
        )

    return {
        "fold_id": fold,
        "train_participant_ids": json.dumps(sorted(train["participant_id"].unique())),
        "test_participant_ids": json.dumps(sorted(test["participant_id"].unique())),
        "n_train_participants": train["participant_id"].nunique(),
        "n_test_participants": test["participant_id"].nunique(),
        "train_hup_participants": site_participants(train, "hup"),
        "train_ucla_participants": site_participants(train, "ucla"),
        "test_hup_participants": site_participants(test, "hup"),
        "test_ucla_participants": site_participants(test, "ucla"),
        "train_pathological_channels": int(train["target"].eq(1).sum()),
        "train_non_pathological_channels": int(train["target"].eq(0).sum()),
        "test_pathological_channels": int(test["target"].eq(1).sum()),
        "test_non_pathological_channels": int(test["target"].eq(0).sum()),
        "train_channels_per_participant": channels_per_participant(train),
        "test_channels_per_participant": channels_per_participant(test),
    }


def _validate_predictor_schema(dataset: pd.DataFrame, predictors: Iterable[str]) -> list[str]:
    """Validate the ordered, dataset-produced predictor schema before any fitting."""
    features = list(predictors)
    if not features or len(features) != len(set(features)):
        raise ValueError("predictor schema must be a non-empty ordered list of unique columns")
    if any(not isinstance(feature, str) or not feature for feature in features):
        raise ValueError("predictor schema contains an invalid column name")
    missing = sorted(set(features) - set(dataset.columns))
    if missing:
        raise ValueError(f"ML dataset lacks schema predictor columns: {missing}")
    return features


def run_cross_validation(
    dataset: pd.DataFrame,
    predictors: Iterable[str],
    selector_factory: Callable[[], FeatureSelector] | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate one fixed predictor schema with the frozen grouped-CV protocol."""
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
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE
    )
    fold_assignment = np.full(len(dataset), -1, dtype=int)
    fold_audit = []
    predictions = []
    fold_metrics = []
    scaler_rows = []
    weight_rows = []
    factor_rows = []
    coefficient_rows = []
    intercept_rows = []
    convergence_rows = []
    selected_feature_rows = []

    splits = list(splitter.split(x, y, groups))
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        train = dataset.iloc[train_index].copy()
        test = dataset.iloc[test_index].copy()
        train_ids = set(train["participant_id"].astype(str))
        test_ids = set(test["participant_id"].astype(str))
        if train_ids & test_ids:
            raise AssertionError(f"fold {fold}: train/test participant overlap")
        fold_audit.append(_fold_audit_row(fold, train, test))
    invalid_folds = [
        row["fold_id"]
        for row in fold_audit
        if row["train_pathological_channels"] == 0
        or row["train_non_pathological_channels"] == 0
        or row["test_pathological_channels"] == 0
        or row["test_non_pathological_channels"] == 0
    ]
    if invalid_folds:
        audit = pd.DataFrame(fold_audit)
        raise ValueError(
            "StratifiedGroupKFold produced folds without both classes; no model was fit "
            f"and the validation method was not changed. Invalid folds={invalid_folds}. "
            f"Fold compositionCONFIGURE_WINDOWS_PATH"
        )

    for fold, (train_index, test_index) in enumerate(splits, start=1):
        train = dataset.iloc[train_index].copy()
        test = dataset.iloc[test_index].copy()
        train_ids = set(train["participant_id"].astype(str))
        test_ids = set(test["participant_id"].astype(str))
        scaler = fit_training_scaler(x[train_index], train_ids, test_ids)
        if set(scaler.fit_participant_ids_) != train_ids or set(scaler.fit_participant_ids_) & test_ids:
            raise AssertionError("scaler audit contains held-out participants")
        x_train_scaled = scaler.transform(x[train_index])
        x_test_scaled = scaler.transform(x[test_index])
        selected_indices = np.arange(len(features), dtype=int)
        if selector_factory is not None:
            selector = selector_factory()
            selector.fit(x_train_scaled, y[train_index])
            selected_indices = np.asarray(selector.selected_indices_, dtype=int)
            if len(selected_indices) != selector.actual_n_features_ or len(selected_indices) == 0:
                raise AssertionError("selector did not report its selected training predictors")
            if (selected_indices < 0).any() or (selected_indices >= len(features)).any():
                raise AssertionError("selector returned a predictor outside the frozen schema")
            if len(np.unique(selected_indices)) != len(selected_indices):
                raise AssertionError("selector returned duplicate predictors")
            # The selector API never receives x_test, and this audit binds its
            # fitted row count and exact input row positions to the outer train split.
            if selector.fit_n_rows_ != len(train_index):
                raise AssertionError("selector was not fit on exactly the outer training rows")
            selected_feature_rows.extend(
                {
                    "fold_id": fold,
                    "selection_rank": rank,
                    "feature": features[feature_index],
                    "feature_index": int(feature_index),
                    "requested_n_features": int(selector.requested_n_features),
                    "actual_n_features": int(selector.actual_n_features_),
                    "n_available_predictors": len(features),
                    "selector_fit_n_rows": int(selector.fit_n_rows_),
                    "selector_fit_row_positions": json.dumps(train_index.tolist()),
                    "outer_test_row_positions": json.dumps(test_index.tolist()),
                    "fit_test_row_overlap_count": int(len(set(train_index) & set(test_index))),
                    "selector_fit_participant_ids": json.dumps(sorted(train_ids)),
                    "outer_test_participant_ids": json.dumps(sorted(test_ids)),
                    "fit_test_participant_overlap_count": int(len(train_ids & test_ids)),
                }
                for rank, feature_index in enumerate(selected_indices, start=1)
            )
        x_train_model = x_train_scaled[:, selected_indices]
        x_test_model = x_test_scaled[:, selected_indices]
        weights, factors = participant_and_class_weights(train["participant_id"], y[train_index])
        weights.insert(1, "channel_id", train["channel_id"].to_numpy())
        weights.insert(0, "fold_id", fold)
        if set(weights["participant_id"].astype(str)) != train_ids:
            raise AssertionError("training weights were not computed solely from train participants")
        model = make_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_train_model, y[train_index], sample_weight=weights["final_sample_weight"])
        convergence_warnings = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
        score = model.predict_proba(x_test_model)[:, int(np.flatnonzero(model.classes_ == 1)[0])]
        predicted = (score >= DECISION_THRESHOLD).astype(int)
        fold_assignment[test_index] = fold
        fold_predictions = test[FINAL_KEY + ["dataset", "target"]].copy()
        fold_predictions = fold_predictions.rename(columns={"target": "true_label"})
        fold_predictions["pathological_score"] = score
        fold_predictions["predicted_label"] = predicted
        fold_predictions["fold_id"] = fold
        predictions.append(fold_predictions)
        metrics = calculate_metrics(y[test_index], score)
        fold_metrics.extend(
            {"fold_id": fold, "metric": name, "value": value}
            for name, value in metrics.items()
        )
        scaler_rows.extend(
            {
                "fold_id": fold,
                "feature": feature,
                "mean": float(mean),
                "scale": float(scale),
                "fit_participant_ids": json.dumps(sorted(train_ids)),
            }
            for feature, mean, scale in zip(features, scaler.mean_, scaler.scale_)
        )
        weight_rows.append(weights)
        factor_rows.extend(
            {
                "fold_id": fold,
                "class": class_id,
                "class_factor": factor,
                "total_participant_weight_mass": float(weights["participant_weight"].sum()),
                "class_participant_weight_mass": float(
                    weights.loc[weights["true_label"].eq(class_id), "participant_weight"].sum()
                ),
                "final_class_weight_mass": float(
                    weights.loc[weights["true_label"].eq(class_id), "final_sample_weight"].sum()
                ),
                "final_weight_min": float(weights["final_sample_weight"].min()),
                "final_weight_median": float(weights["final_sample_weight"].median()),
                "final_weight_mean": float(weights["final_sample_weight"].mean()),
                "final_weight_max": float(weights["final_sample_weight"].max()),
            }
            for class_id, factor in factors.items()
        )
        coefficient_rows.extend(
            {"fold_id": fold, "feature": feature, "coefficient": float(coefficient)}
            for feature, coefficient in zip(np.asarray(features)[selected_indices], model.coef_[0])
        )
        intercept_rows.append({"fold_id": fold, "intercept": float(model.intercept_[0])})
        convergence_rows.append(
            {
                "fold_id": fold,
                "solver": model.solver,
                "max_iter": model.max_iter,
                "n_iter": int(model.n_iter_[0]),
                "converged": not convergence_warnings and int(model.n_iter_[0]) < model.max_iter,
                "convergence_warning": " | ".join(str(item.message) for item in convergence_warnings),
            }
        )

    if (fold_assignment < 1).any():
        raise AssertionError("some OOF observations were never assigned to a test fold")
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
        **({"selected_features_by_fold": pd.DataFrame(selected_feature_rows)} if selector_factory else {}),
    }


def run_nested_elasticnet_cross_validation(
    dataset: pd.DataFrame, predictors: Iterable[str]
) -> dict[str, pd.DataFrame]:
    """Run C2 nested grouped CV without exposing outer tests to tuning.

    Ties in mean inner Average Precision are resolved deterministically by the
    numerically smallest ``C`` and then the numerically smallest ``l1_ratio``.
    """
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
    outer_splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE
    )
    splits = list(outer_splitter.split(x, y, groups))
    fold_assignment = np.full(len(dataset), -1, dtype=int)
    fold_audit, predictions, fold_metrics = [], [], []
    scaler_rows, weight_rows, factor_rows = [], [], []
    coefficient_rows, intercept_rows, convergence_rows = [], [], []
    hyperparameter_rows, inner_rows = [], []

    for fold, (outer_train_index, outer_test_index) in enumerate(splits, start=1):
        outer_train = dataset.iloc[outer_train_index].copy()
        outer_test = dataset.iloc[outer_test_index].copy()
        outer_train_ids = set(outer_train["participant_id"].astype(str))
        outer_test_ids = set(outer_test["participant_id"].astype(str))
        if outer_train_ids & outer_test_ids:
            raise AssertionError(f"fold {fold}: outer train/test participant overlap")
        fold_audit.append(_fold_audit_row(fold, outer_train, outer_test))
        if set(y[outer_train_index]) != {0, 1} or set(y[outer_test_index]) != {0, 1}:
            raise ValueError(
                f"outer fold {fold} lacks both classes; the frozen grouped CV protocol "
                "was not changed and no model was fit for that fold"
            )

        # Inner indices are local to the outer-training partition.  This makes
        # it impossible for an outer-test row or label to enter tuning.
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
        candidate_scores: dict[tuple[float, float], list[float]] = {
            (C, l1_ratio): [] for C in ELASTICNET_C_GRID for l1_ratio in ELASTICNET_L1_RATIO_GRID
        }
        invalid_candidates: dict[tuple[float, float], list[str]] = {
            (C, l1_ratio): [] for C in ELASTICNET_C_GRID for l1_ratio in ELASTICNET_L1_RATIO_GRID
        }
        for inner_fold, (inner_train_local, inner_validation_local) in enumerate(inner_splits, start=1):
            inner_train_index = outer_train_index[inner_train_local]
            inner_validation_index = outer_train_index[inner_validation_local]
            inner_train_ids = set(dataset.iloc[inner_train_index]["participant_id"].astype(str))
            inner_validation_ids = set(dataset.iloc[inner_validation_index]["participant_id"].astype(str))
            if inner_train_ids & inner_validation_ids:
                raise AssertionError(f"outer fold {fold}, inner fold {inner_fold}: participant overlap")
            if not inner_train_ids <= outer_train_ids or not inner_validation_ids <= outer_train_ids:
                raise AssertionError("inner split escaped the outer training fold")
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
            common_audit = {
                "outer_fold": fold,
                "inner_fold": inner_fold,
                "inner_train_row_positions": json.dumps(inner_train_index.tolist()),
                "inner_validation_row_positions": json.dumps(inner_validation_index.tolist()),
                "inner_train_participant_ids": json.dumps(sorted(inner_train_ids)),
                "inner_validation_participant_ids": json.dumps(sorted(inner_validation_ids)),
                "inner_train_classes": json.dumps(sorted(set(y[inner_train_index].tolist()))),
                "inner_validation_classes": json.dumps(sorted(set(y[inner_validation_index].tolist()))),
                "inner_train_validation_participant_overlap_count": len(inner_train_ids & inner_validation_ids),
                "outer_test_participant_overlap_count": len(
                    (inner_train_ids | inner_validation_ids) & outer_test_ids
                ),
                "scaler_fit_participant_ids": json.dumps(sorted(inner_scaler.fit_participant_ids_)),
                "weight_fit_participant_ids": json.dumps(sorted(inner_train_ids)),
            }
            for C, l1_ratio in candidate_scores:
                model = make_elasticnet_model(C, l1_ratio)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(x_inner_train, y[inner_train_index], sample_weight=inner_weights["final_sample_weight"])
                converged, convergence_message = _elasticnet_convergence_status(model, caught)
                if converged:
                    score = model.predict_proba(x_inner_validation)[:, int(np.flatnonzero(model.classes_ == 1)[0])]
                    average_precision = float(average_precision_score(y[inner_validation_index], score))
                    candidate_scores[(C, l1_ratio)].append(average_precision)
                else:
                    average_precision = math.nan
                    invalid_candidates[(C, l1_ratio)].append(
                        f"inner fold {inner_fold}: {convergence_message}"
                    )
                inner_rows.append(
                    {
                        **common_audit,
                        "C": C,
                        "l1_ratio": l1_ratio,
                        "average_precision": average_precision,
                        "converged": converged,
                        "convergence_warning": convergence_message,
                        "valid_for_tuning": converged,
                    }
                )
        candidate_means = [
            (float(np.mean(scores)), C, l1_ratio)
            for (C, l1_ratio), scores in candidate_scores.items()
            if not invalid_candidates[(C, l1_ratio)] and len(scores) == INNER_N_SPLITS
        ]
        if not candidate_means:
            raise RuntimeError(
                f"outer fold {fold}: no fully converged Elastic Net candidate across all "
                f"{INNER_N_SPLITS} inner folds; tuning is invalid"
            )
        # ``min`` makes the stated descending-score/ascending-parameter tie rule explicit.
        inner_mean_ap, best_C, best_l1_ratio = min(
            candidate_means, key=lambda item: (-item[0], item[1], item[2])
        )

        outer_scaler = fit_training_scaler(x[outer_train_index], outer_train_ids, outer_test_ids)
        x_outer_train = outer_scaler.transform(x[outer_train_index])
        x_outer_test = outer_scaler.transform(x[outer_test_index])
        outer_weights, factors = participant_and_class_weights(outer_train["participant_id"], y[outer_train_index])
        outer_weights.insert(1, "channel_id", outer_train["channel_id"].to_numpy())
        outer_weights.insert(0, "fold_id", fold)
        model = make_elasticnet_model(best_C, best_l1_ratio)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_outer_train, y[outer_train_index], sample_weight=outer_weights["final_sample_weight"])
        converged, convergence_message = _elasticnet_convergence_status(model, caught)
        if not converged:
            raise RuntimeError(
                f"outer fold {fold}: final Elastic Net fit did not converge "
                f"(C={best_C}, l1_ratio={best_l1_ratio}, {convergence_message})"
            )
        score = model.predict_proba(x_outer_test)[:, int(np.flatnonzero(model.classes_ == 1)[0])]
        predicted = (score >= DECISION_THRESHOLD).astype(int)
        n_nonzero = int(np.count_nonzero(model.coef_[0]))
        hyperparameter_rows.append({
            "outer_fold": fold, "best_C": best_C, "best_l1_ratio": best_l1_ratio,
            "inner_mean_average_precision": inner_mean_ap, "converged": converged,
            "convergence_warning": convergence_message,
            "n_nonzero_coefficients": n_nonzero,
        })
        fold_assignment[outer_test_index] = fold
        fold_predictions = outer_test[FINAL_KEY + ["dataset", "target"]].copy().rename(columns={"target": "true_label"})
        fold_predictions["pathological_score"] = score
        fold_predictions["predicted_label"] = predicted
        fold_predictions["fold_id"] = fold
        predictions.append(fold_predictions)
        metrics = calculate_metrics(y[outer_test_index], score)
        fold_metrics.extend({"fold_id": fold, "metric": name, "value": value} for name, value in metrics.items())
        scaler_rows.extend({"fold_id": fold, "feature": feature, "mean": float(mean), "scale": float(scale), "fit_participant_ids": json.dumps(sorted(outer_train_ids))} for feature, mean, scale in zip(features, outer_scaler.mean_, outer_scaler.scale_))
        weight_rows.append(outer_weights)
        factor_rows.extend({"fold_id": fold, "class": class_id, "class_factor": factor} for class_id, factor in factors.items())
        coefficient_rows.extend({"outer_fold": fold, "feature": feature, "coefficient": float(coefficient), "nonzero": bool(coefficient != 0.0)} for feature, coefficient in zip(features, model.coef_[0]))
        intercept_rows.append({"fold_id": fold, "intercept": float(model.intercept_[0])})
        convergence_rows.append({"fold_id": fold, "solver": model.solver, "max_iter": model.max_iter, "C": best_C, "l1_ratio": best_l1_ratio, "n_iter": int(model.n_iter_[0]), "converged": converged, "convergence_warning": convergence_message})

    if (fold_assignment < 1).any():
        raise AssertionError("some OOF observations were never assigned to an outer test fold")
    oof = pd.concat(predictions, ignore_index=True)
    if len(oof) != len(dataset) or oof.duplicated(FINAL_KEY).any():
        raise AssertionError("every included observation must be predicted exactly once")
    coefficients = pd.DataFrame(coefficient_rows)
    stability = coefficients.groupby("feature", sort=False).agg(n_outer_folds_nonzero=("nonzero", "sum")).reset_index()
    stability["n_outer_folds"] = N_SPLITS
    stability["fraction_outer_folds_nonzero"] = stability["n_outer_folds_nonzero"] / N_SPLITS
    assignments = dataset[FINAL_KEY + ["dataset", "target"]].copy()
    assignments["fold_id"] = fold_assignment
    return {
        "fold_assignments": assignments,
        "fold_audit": pd.DataFrame(fold_audit),
        "scaler_parameters": pd.DataFrame(scaler_rows),
        "sample_weights": pd.concat(weight_rows, ignore_index=True),
        "class_factors": pd.DataFrame(factor_rows),
        "model_coefficients": coefficients.rename(columns={"outer_fold": "fold_id"}),
        "model_intercepts": pd.DataFrame(intercept_rows),
        "convergence": pd.DataFrame(convergence_rows),
        "oof_predictions": oof.sort_values(FINAL_KEY, kind="stable").reset_index(drop=True),
        "fold_metrics": pd.DataFrame(fold_metrics),
        "elasticnet_hyperparameters_by_fold": pd.DataFrame(hyperparameter_rows),
        "elasticnet_inner_cv_results": pd.DataFrame(inner_rows),
        "elasticnet_coefficients_by_fold": coefficients,
        "elasticnet_feature_stability": stability,
    }


def _load_predictor_schema(feature_list_path: Path) -> tuple[dict[str, Any], list[str]]:
    payload = json.loads(feature_list_path.read_text(encoding="utf-8"))
    predictors = payload.get("predictors")
    if not isinstance(predictors, list):
        raise ValueError("feature list must contain a predictors list")
    return payload, predictors


def run_logistic_evaluation(
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
    selector_factory: Callable[[], FeatureSelector] | None = None,
    feature_selection_summary: dict[str, Any] | None = None,
) -> Path:
    """Run the fixed protocol using a predictor order supplied by a dataset schema."""
    _, schema_predictors = _load_predictor_schema(feature_list_path)
    expected = list(expected_predictors)
    if schema_predictors != expected:
        raise ValueError(
            "dataset feature order differs from the expected frozen schema "
            f"({len(expected)} predictors)"
        )
    dataset = pd.read_parquet(dataset_path)
    task = sorted(set(dataset["task"].astype(str).str.strip().str.lower()))[0]
    cv_tables = run_cross_validation(dataset, schema_predictors, selector_factory=selector_factory)
    selection_details = feature_selection_summary
    if selector_factory is not None:
        selected = cv_tables["selected_features_by_fold"]
        observed_frequency = (
            selected.groupby(["feature", "feature_index"], sort=True)
            .agg(n_folds_selected=("fold_id", "nunique"))
            .reset_index()
        )
        # Keep zero-frequency predictors too: the table is an audit of the full
        # frozen schema, not only of features that happened to be selected.
        frequency = pd.DataFrame(
            {"feature": schema_predictors, "feature_index": range(len(schema_predictors))}
        ).merge(observed_frequency, on=["feature", "feature_index"], how="left", validate="one_to_one")
        frequency["n_folds_selected"] = frequency["n_folds_selected"].fillna(0).astype(int)
        frequency["n_outer_folds"] = N_SPLITS
        frequency["selection_frequency"] = frequency["n_folds_selected"] / N_SPLITS
        cv_tables["selection_frequencies"] = frequency
        actual_by_fold = (
            selected.groupby("fold_id", sort=True)["actual_n_features"].first().astype(int).to_dict()
        )
        selection_details = {
            **(selection_details or {}),
            "actual_n_features_by_fold": {str(fold): count for fold, count in actual_by_fold.items()},
            "actual_n_features_unique": sorted(set(actual_by_fold.values())),
        }
    evaluation = evaluation_tables(cv_tables["oof_predictions"], n_bootstrap=n_bootstrap)
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
            "cv": {
                "class": "StratifiedGroupKFold",
                "n_splits": N_SPLITS,
                "shuffle": True,
                "random_state": CV_RANDOM_STATE,
                "groups": "participant_id",
            },
            "normalization": "StandardScaler fit independently on X_train in each fold",
            "weighting": (
                "training-fold-only inverse participant channel count followed by equal "
                "participant-weight class mass and normalization to mean one"
            ),
            "model": {
                "class": "LogisticRegression", "penalty": "l2", "C": 1.0,
                "class_weight": None, "solver": "lbfgs", "max_iter": 5000,
            },
            "decision_threshold": DECISION_THRESHOLD,
            "threshold_was_optimized": False,
            "feature_selection": selector_factory is not None,
            "feature_counts": feature_schema,
            "all_predictors_retained": selector_factory is None,
            "imputation": False,
            "resampling": False,
            "bootstrap": {
                "unit": "participant", "refit_models": False,
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
        if selector_factory is not None:
            summary["feature_selection_details"] = selection_details
        atomic_json(summary, staging / "model_run_summary.json")
    return output_dir / "oof_predictions.csv"


def run_nested_elasticnet_evaluation(
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
) -> Path:
    """Publish C2 artifacts plus the standard metric/bootstrap artifacts."""
    _, schema_predictors = _load_predictor_schema(feature_list_path)
    expected = list(expected_predictors)
    if schema_predictors != expected:
        raise ValueError(
            "dataset feature order differs from the expected frozen schema "
            f"({len(expected)} predictors)"
        )
    dataset = pd.read_parquet(dataset_path)
    task = sorted(set(dataset["task"].astype(str).str.strip().str.lower()))[0]
    cv_tables = run_nested_elasticnet_cross_validation(dataset, schema_predictors)
    evaluation = evaluation_tables(cv_tables["oof_predictions"], n_bootstrap=n_bootstrap)
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
            "outer_cv": {"class": "StratifiedGroupKFold", "n_splits": N_SPLITS, "shuffle": True, "random_state": CV_RANDOM_STATE, "groups": "participant_id"},
            "inner_cv": {"class": "StratifiedGroupKFold", "n_splits": INNER_N_SPLITS, "shuffle": True, "random_state": CV_RANDOM_STATE, "groups": "participant_id", "scope": "each outer training fold"},
            "normalization": "StandardScaler fit independently on each inner-training split during tuning and on each full outer training fold for final fitting",
            "weighting": "inner-training-only and outer-training-only inverse participant channel count followed by equal participant-weight class mass and normalization to mean one",
            "model": {"class": "LogisticRegression", "penalty": "elasticnet", "solver": "saga", "max_iter": 10000, "random_state": CV_RANDOM_STATE, "class_weight": None},
            "hyperparameter_grid": {"C": list(ELASTICNET_C_GRID), "l1_ratio": list(ELASTICNET_L1_RATIO_GRID)},
            "tuning_metric": "average_precision",
            "tie_breaking": "highest mean inner Average Precision; then lowest C; then lowest l1_ratio",
            "decision_threshold": DECISION_THRESHOLD,
            "threshold_was_optimized": False,
            "feature_selection": False,
            "all_predictors_retained": True,
            "feature_counts": feature_schema,
            "imputation": False,
            "resampling": False,
            "bootstrap": {"unit": "participant", "refit_models": False, "n_replicates": n_bootstrap, "random_state": BOOTSTRAP_RANDOM_STATE, "confidence_level": CONFIDENCE_LEVEL, "interval": "percentile"},
            "software_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
            "all_folds_converged": bool(cv_tables["convergence"]["converged"].all()),
        }
        atomic_json(summary, staging / "model_run_summary.json")
    return output_dir / "oof_predictions.csv"


def run_baseline(
    dataset_path: Path,
    feature_list_path: Path,
    output_dir: Path,
    repository: Path = REPOSITORY,
    n_bootstrap: int = N_BOOTSTRAP,
) -> Path:
    """Backward-compatible Baseline entry point with its frozen 96-column schema."""
    return run_logistic_evaluation(
        dataset_path,
        feature_list_path,
        output_dir,
        expected_predictors=predictor_names(),
        feature_schema=feature_counts(),
        protocol_version=PROTOCOL_VERSION,
        stage="fixed_logistic_baseline",
        repository=repository,
        n_bootstrap=n_bootstrap,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--feature-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()
    output = run_baseline(
        args.dataset.resolve(), args.feature_list.resolve(), args.output_dir.resolve(),
        args.repository.resolve(), args.n_bootstrap,
    )
    print(output)


if __name__ == "__main__":
    main()
