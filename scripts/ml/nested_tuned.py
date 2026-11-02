"""Nested-CV tuned experiment ``ml_logistic_complexity_c1_nested_tuned_v1``.

The C1-nested experiment tunes only the number of mRMR predictors ``N``.  This
module extends it, without touching it, by tuning ``N`` and the L2 inverse
regularization strength ``C`` jointly inside each outer training fold, and by
selecting a classification threshold from inner out-of-fold probabilities that
never see the outer test fold.  Every scientific primitive (outer folds, inner
folds, scaler, participant/class weights, mRMR, metrics, participant-cluster
bootstrap) is imported from the existing modules, so this experiment differs
from C1-nested only in what is tuned and in the extra threshold artifacts.

Threshold optimization changes threshold-dependent metrics only.  AUROC and
Average Precision are computed from continuous probabilities and are therefore
identical at threshold 0.5 and at the optimized threshold; the evaluator
asserts this rather than assuming it.
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

from scripts.ml.common import (
    DEFAULT_ANALYSIS_ROOT,
    atomic_csv,
    atomic_json,
    git_commit,
    new_output_directory,
    resolve_task,
    sha256,
    timestamp_utc,
)
from scripts.ml.dataset import FINAL_KEY
from scripts.ml.feature_selection import MRMR_MI_PARAMETERS

# The scientific core is shared with the fixed-N and C1-nested experiments on
# purpose; the leading-underscore helpers are internal to the ML package.
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
    participant_and_class_weights,
    resample_participant_clusters,
)
from scripts.ml.nested_mrmr import _fit_mrmr, selection_frequencies


EXPERIMENT_ID = "ml_logistic_complexity_c1_nested_tuned_v1"
PROTOCOL_VERSION = EXPERIMENT_ID
STAGE = "nested_tuned_logistic_complexity_c1_nested_tuned"
REPORT_FILENAME = "logistic_complexity_c1_nested_tuned_report.html"

# Frozen candidate grids.  Both are tuned per outer fold, never across folds.
TUNED_N_GRID = (5, 8, 10, 15, 20, 30)
TUNED_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_MINIMUM_TPR = 0.50

MODEL_TIE_BREAK_RULE = (
    "highest mean inner Average Precision; then smallest N; then smallest C"
)
THRESHOLD_SELECTION_RULE = (
    "maximize PPV subject to TPR >= minimum_TPR on inner out-of-fold probabilities; "
    "then highest TPR; then lowest threshold"
)
THRESHOLD_CANDIDATE_RULE = (
    "every distinct inner out-of-fold predicted probability, applied as "
    "predict positive when probability >= threshold"
)
# Metrics published for both thresholds.  auroc and average_precision are
# threshold-free and are repeated so the two tables can be compared row by row.
PROBABILITY_METRIC_NAMES = ("auroc", "average_precision")
THRESHOLD_METRIC_NAMES = (
    "sensitivity",
    "specificity",
    "precision",
    "fpr",
    "fnr",
    "f1",
    "balanced_accuracy",
)
TUNED_METRIC_NAMES = PROBABILITY_METRIC_NAMES + THRESHOLD_METRIC_NAMES


def tuned_task_output_root(
    task: str, analysis_root: Path = DEFAULT_ANALYSIS_ROOT
) -> Path:
    """Own output root, so this experiment can never overwrite another one."""
    return Path(analysis_root) / EXPERIMENT_ID / resolve_task(task)


def make_tuned_model(C: float) -> LogisticRegression:
    """Create the frozen estimator; sample weights are passed explicitly to fit."""
    model = LogisticRegression(
        penalty="l2",
        C=float(C),
        class_weight=None,
        solver="lbfgs",
        max_iter=5000,
    )
    if model.class_weight is not None:
        raise AssertionError("this experiment must use explicit sample weights")
    return model


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


def _validate_c_grid(c_grid: Iterable[float]) -> tuple[float, ...]:
    grid = [float(candidate) for candidate in c_grid]
    if not grid or len(set(grid)) != len(grid):
        raise ValueError("the C grid must be a non-empty set of distinct values")
    if any(not math.isfinite(candidate) or candidate <= 0.0 for candidate in grid):
        raise ValueError("every C candidate must be a positive finite value")
    return tuple(sorted(grid))


def validate_minimum_tpr(minimum_tpr: float) -> float:
    """Reject a minimum TPR constraint that cannot define an eligible threshold."""
    value = float(minimum_tpr)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"minimum_tpr must lie in (0, 1]; got {minimum_tpr!r}")
    return value


def _fit_scored_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: Iterable[float],
    x_evaluate: np.ndarray,
    C: float,
) -> tuple[Any, np.ndarray, bool, str]:
    """Fit the L2 model at ``C`` and return its pathological probabilities."""
    model = make_tuned_model(C)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train, sample_weight=sample_weight)
    convergence_warnings = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    converged = not convergence_warnings and int(model.n_iter_[0]) < model.max_iter
    message = " | ".join(str(item.message) for item in convergence_warnings)
    if not convergence_warnings and int(model.n_iter_[0]) >= model.max_iter:
        message = f"max_iter exhausted ({model.max_iter})"
    probability = model.predict_proba(x_evaluate)[
        :, int(np.flatnonzero(model.classes_ == 1)[0])
    ]
    return model, probability, converged, message


def metrics_from_labels(
    y_true: Iterable[int], scores: Iterable[float], predicted: Iterable[int]
) -> dict[str, float]:
    """Metrics for an arbitrary decision rule already applied to the scores.

    The threshold-free metrics are read from ``scores`` and the
    threshold-dependent ones from ``predicted``.  At ``predicted = scores >=
    0.5`` this reproduces :func:`scripts.ml.logistic_baseline.calculate_metrics`
    exactly, which keeps the optimized-threshold tables comparable with every
    existing experiment.
    """
    y = np.asarray(list(y_true), dtype=int)
    probability = np.asarray(list(scores), dtype=float)
    label = np.asarray(list(predicted), dtype=int)
    if y.size != probability.size or y.size != label.size or y.size == 0:
        raise ValueError("metric inputs must be non-empty and aligned")
    if not set(np.unique(label).tolist()) <= {0, 1}:
        raise ValueError("predicted labels must be binary")
    classes = set(y.tolist())
    true_positive = int(((y == 1) & (label == 1)).sum())
    false_positive = int(((y == 0) & (label == 1)).sum())
    false_negative = int(((y == 1) & (label == 0)).sum())
    precision_denominator = true_positive + false_positive
    f1_denominator = 2 * true_positive + false_positive + false_negative
    sensitivity = (
        float(recall_score(y, label, pos_label=1, zero_division=0)) if 1 in classes else math.nan
    )
    specificity = (
        float(recall_score(y, label, pos_label=0, zero_division=0)) if 0 in classes else math.nan
    )
    metrics = {
        "auroc": float(roc_auc_score(y, probability)) if classes == {0, 1} else math.nan,
        "average_precision": (
            float(average_precision_score(y, probability)) if classes == {0, 1} else math.nan
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": (
            float(precision_score(y, label, pos_label=1, zero_division=0))
            if precision_denominator > 0 else math.nan
        ),
        "fpr": 1.0 - specificity if 0 in classes else math.nan,
        "fnr": 1.0 - sensitivity if 1 in classes else math.nan,
        "f1": (
            float(f1_score(y, label, pos_label=1, zero_division=0))
            if f1_denominator > 0 else math.nan
        ),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y, label)) if classes == {0, 1} else math.nan
        ),
    }
    tn, fp, fn, tp = confusion_matrix(y, label, labels=[0, 1]).ravel()
    metrics.update(
        true_negative=float(tn), false_positive=float(fp),
        false_negative=float(fn), true_positive=float(tp),
        n_predicted_positive=float(int((label == 1).sum())),
        n_observations=float(y.size),
    )
    return metrics


def threshold_trace(
    y_true: Iterable[int],
    scores: Iterable[float],
    minimum_tpr: float,
    candidates: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Confusion counts and rates for every candidate threshold.

    Candidates default to every distinct predicted probability, which is the
    complete set of thresholds that produce a distinct decision rule under
    ``probability >= threshold``; no coarse grid is imposed.
    """
    y = np.asarray(list(y_true), dtype=int)
    probability = np.asarray(list(scores), dtype=float)
    if y.size != probability.size or y.size == 0:
        raise ValueError("threshold search inputs must be non-empty and aligned")
    if set(y.tolist()) != {0, 1}:
        raise ValueError("threshold search requires both classes")
    if not np.isfinite(probability).all():
        raise ValueError("threshold search requires finite probabilities")
    minimum = validate_minimum_tpr(minimum_tpr)
    grid = (
        np.unique(probability)
        if candidates is None
        else np.unique(np.asarray(list(candidates), dtype=float))
    )
    if grid.size == 0:
        raise ValueError("threshold search requires at least one candidate threshold")
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    # Vectorized over candidates: predicted positive when probability >= threshold.
    predicted_positive = probability[None, :] >= grid[:, None]
    true_positive = (predicted_positive & (y == 1)[None, :]).sum(axis=1).astype(int)
    false_positive = (predicted_positive & (y == 0)[None, :]).sum(axis=1).astype(int)
    false_negative = positives - true_positive
    true_negative = negatives - false_positive
    with np.errstate(divide="ignore", invalid="ignore"):
        ppv = np.where(
            true_positive + false_positive > 0,
            true_positive / np.maximum(true_positive + false_positive, 1),
            np.nan,
        )
    tpr = true_positive / positives
    specificity = true_negative / negatives
    trace = pd.DataFrame(
        {
            "threshold": grid.astype(float),
            "ppv": ppv.astype(float),
            "tpr": tpr.astype(float),
            "specificity": specificity.astype(float),
            "fpr": (1.0 - specificity).astype(float),
            "fnr": (1.0 - tpr).astype(float),
            "tp": true_positive,
            "fp": false_positive,
            "tn": true_negative,
            "fn": false_negative,
            "n_predicted_positive": (true_positive + false_positive).astype(int),
        }
    )
    # An undefined PPV (no predicted positive) can never maximize PPV, so it is
    # reported in the trace but excluded from eligibility.
    trace["eligible_under_minimum_tpr"] = (
        trace["tpr"].ge(minimum) & np.isfinite(trace["ppv"])
    )
    trace["minimum_tpr"] = minimum
    return trace.sort_values("threshold", kind="stable").reset_index(drop=True)


def select_threshold(
    y_true: Iterable[int],
    scores: Iterable[float],
    minimum_tpr: float = DEFAULT_MINIMUM_TPR,
    candidates: Iterable[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Choose the highest-PPV threshold whose TPR meets the constraint.

    Ties are resolved deterministically: highest PPV, then higher TPR, then the
    lower threshold.  If no candidate satisfies the constraint the call fails
    loudly; it never falls back to 0.5.
    """
    minimum = validate_minimum_tpr(minimum_tpr)
    trace = threshold_trace(y_true, scores, minimum, candidates=candidates)
    eligible = trace[trace["eligible_under_minimum_tpr"]]
    if eligible.empty:
        raise RuntimeError(
            f"no candidate threshold reaches the minimum TPR of {minimum}; "
            f"the highest achievable TPR over {len(trace)} candidates was "
            f"{float(trace['tpr'].max())}. No fallback threshold is applied."
        )
    ordered = eligible.sort_values(
        ["ppv", "tpr", "threshold"], ascending=[False, False, True], kind="stable"
    )
    chosen = ordered.iloc[0]
    return trace, {
        "threshold": float(chosen["threshold"]),
        "ppv": float(chosen["ppv"]),
        "tpr": float(chosen["tpr"]),
        "specificity": float(chosen["specificity"]),
        "fpr": float(chosen["fpr"]),
        "fnr": float(chosen["fnr"]),
        "tp": int(chosen["tp"]),
        "fp": int(chosen["fp"]),
        "tn": int(chosen["tn"]),
        "fn": int(chosen["fn"]),
        "n_predicted_positive": int(chosen["n_predicted_positive"]),
        "n_candidates": int(len(trace)),
        "n_eligible_candidates": int(len(eligible)),
    }


def assert_matching_fold_assignments(
    assignments: pd.DataFrame, reference: pd.DataFrame
) -> None:
    """Fail loudly unless (participant_id, channel_id, fold_id) matches a reference."""
    columns = FINAL_KEY + ["fold_id"]
    for name, frame in (("new", assignments), ("reference", reference)):
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} fold assignments lack columns: {missing}")
    left = (
        assignments[columns]
        .astype({"participant_id": str, "channel_id": str, "fold_id": int})
        .sort_values(FINAL_KEY, kind="stable")
        .reset_index(drop=True)
    )
    right = (
        reference[columns]
        .astype({"participant_id": str, "channel_id": str, "fold_id": int})
        .sort_values(FINAL_KEY, kind="stable")
        .reset_index(drop=True)
    )
    if not left.equals(right):
        raise AssertionError(
            "this experiment must reuse the existing (participant_id, channel_id, "
            "fold_id) assignment of the complexity experiments, but it differs"
        )


def _label_metric_rows(
    predictions: pd.DataFrame, label_column: str, scope: str, scope_value: str
) -> list[dict[str, Any]]:
    metrics = metrics_from_labels(
        predictions["true_label"],
        predictions["pathological_score"],
        predictions[label_column],
    )
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


def overall_and_site_metrics(predictions: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Overall and per-site metrics for one already applied decision rule.

    The shared evaluator does not publish FPR and FNR, so this table is what
    makes those two rates available at threshold 0.5 as well as at the
    optimized threshold.  It adds no bootstrap: the 0.5 bootstrap is the shared
    one and is not recomputed here.
    """
    rows = _label_metric_rows(predictions, label_column, "overall", "all")
    for site, group in predictions.groupby("dataset", sort=True):
        rows.extend(_label_metric_rows(group, label_column, "site", str(site)))
    return pd.DataFrame(rows)


def _optimized_metric_rows(
    predictions: pd.DataFrame, scope: str, scope_value: str
) -> list[dict[str, Any]]:
    return _label_metric_rows(
        predictions, "predicted_label_optimized_threshold", scope, scope_value
    )


def optimized_threshold_evaluation_tables(
    predictions: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP
) -> dict[str, pd.DataFrame]:
    """Mirror :func:`evaluation_tables` using the fold-specific thresholds.

    Each observation keeps the threshold of the outer fold in which it was
    tested; the bootstrap resamples participant clusters of those already frozen
    predicted labels and never re-optimizes a threshold.
    """
    required = {"predicted_label_optimized_threshold", "optimized_threshold"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"optimized-threshold evaluation needs columns: {missing}")
    metric_rows = _optimized_metric_rows(predictions, "overall", "all")
    for site, group in predictions.groupby("dataset", sort=True):
        metric_rows.extend(_optimized_metric_rows(group, "site", str(site)))

    participant_rows = []
    for (participant, site), group in predictions.groupby(["participant_id", "dataset"], sort=True):
        metrics = metrics_from_labels(
            group["true_label"],
            group["pathological_score"],
            group["predicted_label_optimized_threshold"],
        )
        participant_rows.append(
            {
                "participant_id": participant,
                "dataset": site,
                "n_channels": len(group),
                "pathological": int(group["true_label"].eq(1).sum()),
                "non_pathological": int(group["true_label"].eq(0).sum()),
                "optimized_threshold": float(group["optimized_threshold"].iloc[0]),
                **metrics,
            }
        )
    participant_metrics = pd.DataFrame(participant_rows)
    summary_rows = []
    for metric in TUNED_METRIC_NAMES:
        valid = pd.to_numeric(participant_metrics[metric], errors="coerce").dropna()
        summary_rows.append(
            {
                "metric": metric,
                "mean_across_defined_participants": float(valid.mean()) if len(valid) else math.nan,
                "median_across_defined_participants": (
                    float(valid.median()) if len(valid) else math.nan
                ),
                "n_participants_defined": len(valid),
                "n_participants_total": len(participant_metrics),
            }
        )

    bootstrap_parts, ci_parts = [], []
    scopes = [("overall", "all", predictions)] + [
        ("site", str(site), group) for site, group in predictions.groupby("dataset", sort=True)
    ]
    for scope, value, group in scopes:
        distribution = _optimized_cluster_bootstrap(group, n_bootstrap=n_bootstrap)
        distribution.insert(0, "scope_value", value)
        distribution.insert(0, "scope", scope)
        confidence = _optimized_confidence_intervals(distribution)
        confidence.insert(0, "scope_value", value)
        confidence.insert(0, "scope", scope)
        bootstrap_parts.append(distribution)
        ci_parts.append(confidence)
    return {
        "optimized_threshold_overall_site_metrics": pd.DataFrame(metric_rows),
        "optimized_threshold_participant_metrics": participant_metrics,
        "optimized_threshold_participant_metric_summary": pd.DataFrame(summary_rows),
        "optimized_threshold_bootstrap_distribution": pd.concat(
            bootstrap_parts, ignore_index=True
        ),
        "optimized_threshold_bootstrap_confidence_intervals": pd.concat(
            ci_parts, ignore_index=True
        ),
    }


def _optimized_cluster_bootstrap(
    predictions: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = BOOTSTRAP_RANDOM_STATE,
) -> pd.DataFrame:
    """Participant-cluster bootstrap over frozen optimized-threshold labels."""
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
        metrics = metrics_from_labels(
            sample["true_label"],
            sample["pathological_score"],
            sample["predicted_label_optimized_threshold"],
        )
        for metric in TUNED_METRIC_NAMES:
            rows.append(
                {
                    "bootstrap_replicate": replicate,
                    "metric": metric,
                    "value": metrics[metric],
                    "defined": bool(math.isfinite(metrics[metric])),
                }
            )
    return pd.DataFrame(rows)


def _optimized_confidence_intervals(distribution: pd.DataFrame) -> pd.DataFrame:
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


def evaluation_tables_agreement(
    shared: pd.DataFrame, companion: pd.DataFrame
) -> list[str]:
    """Return the metrics on which two overall/site metric tables disagree."""
    keys = ["scope", "scope_value", "metric"]
    merged = shared.merge(companion, on=keys, how="inner", suffixes=("_shared", "_companion"))
    disagreeing = merged[
        ~np.isclose(
            pd.to_numeric(merged["value_shared"], errors="coerce"),
            pd.to_numeric(merged["value_companion"], errors="coerce"),
            rtol=0.0, atol=1e-12, equal_nan=True,
        )
    ]
    return sorted(set(disagreeing["metric"].astype(str)))


def run_nested_tuned_cross_validation(
    dataset: pd.DataFrame,
    predictors: Iterable[str],
    *,
    n_grid: Iterable[int] = TUNED_N_GRID,
    c_grid: Iterable[float] = TUNED_C_GRID,
    minimum_tpr: float = DEFAULT_MINIMUM_TPR,
) -> dict[str, pd.DataFrame]:
    """Run the tuned nested grouped CV.

    The outer protocol is the frozen complexity protocol, so the
    ``(participant_id, channel_id, fold_id)`` assignment is identical to the
    existing experiments.  Inside each outer training fold a 3-fold grouped
    inner CV scores every ``(N, C)`` pair by mean Average Precision, the inner
    out-of-fold probabilities of the winning pair select the threshold, and only
    then is the outer test fold scored, once, at both 0.5 and the selected
    threshold.  The outer test fold never reaches the scaler, the weights, the
    mRMR statistics, the tuning of ``(N, C)``, the threshold search or the fit.
    """
    features = _validate_predictor_schema(dataset, predictors)
    grid = _validate_n_grid(n_grid, len(features))
    regularization = _validate_c_grid(c_grid)
    minimum = validate_minimum_tpr(minimum_tpr)
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
    threshold_rows, trace_rows, inner_oof_rows = [], [], []
    standard_metric_rows, optimized_metric_rows = [], []

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
        # it impossible for an outer-test row or label to enter any tuning.
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
        candidates = [(n, C) for n in grid for C in regularization]
        candidate_scores: dict[tuple[int, float], list[float]] = {key: [] for key in candidates}
        invalid_candidates: dict[tuple[int, float], list[str]] = {key: [] for key in candidates}
        # Inner-validation probabilities are cached per (inner fold, N, C) while
        # the grid is scored.  Reusing the cache for the winning pair is exactly
        # the refit the protocol describes: same rows, same scaler, same mRMR,
        # same deterministic solver, therefore the same probabilities.
        probability_cache: dict[tuple[int, int, float], np.ndarray] = {}
        inner_validation_positions: dict[int, np.ndarray] = {}

        for inner_fold, (inner_train_local, inner_validation_local) in enumerate(
            inner_splits, start=1
        ):
            inner_train_index = outer_train_index[inner_train_local]
            inner_validation_index = outer_train_index[inner_validation_local]
            inner_validation_positions[inner_fold] = inner_validation_index
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
            if set(inner_train_index) & set(outer_test_index) or set(
                inner_validation_index
            ) & set(outer_test_index):
                raise AssertionError("an outer-test row reached the inner tuning CV")
            if len(inner_train_index) == 0 or len(inner_validation_index) == 0:
                raise ValueError(
                    f"outer fold {fold}, inner fold {inner_fold}: a valid grouped inner CV "
                    "cannot be constructed because a split is empty"
                )
            if set(y[inner_train_index]) != {0, 1} or set(y[inner_validation_index]) != {0, 1}:
                raise ValueError(
                    f"outer fold {fold}, inner fold {inner_fold}: a valid grouped inner CV "
                    "cannot be constructed because both training and validation must contain "
                    "both classes"
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
            # Greedy MID selection is prefix-nested: the first N picks of a
            # rank-max(grid) fit are the picks of a rank-N fit on the same rows,
            # so one fit per inner split reproduces every candidate N exactly.
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
                "outer_test_row_overlap_count": len(
                    (set(inner_train_index) | set(inner_validation_index))
                    & set(outer_test_index)
                ),
                "scaler_fit_participant_ids": json.dumps(sorted(inner_scaler.fit_participant_ids_)),
                "weight_fit_participant_ids": json.dumps(sorted(inner_train_ids)),
                "selector_fit_n_rows": int(inner_selector.fit_n_rows_),
            }
            for n_candidate in grid:
                selected_indices = inner_ranking[:n_candidate]
                if len(selected_indices) != n_candidate:
                    raise AssertionError(
                        f"outer fold {fold}, inner fold {inner_fold}: mRMR returned "
                        f"{len(selected_indices)} predictors for N={n_candidate}"
                    )
                selected_names = [features[index] for index in selected_indices]
                for c_candidate in regularization:
                    _, probability, converged, message = _fit_scored_model(
                        x_inner_train[:, selected_indices],
                        y[inner_train_index],
                        inner_weights["final_sample_weight"],
                        x_inner_validation[:, selected_indices],
                        c_candidate,
                    )
                    probability_cache[(inner_fold, n_candidate, c_candidate)] = probability
                    key = (n_candidate, c_candidate)
                    if converged:
                        average_precision = float(
                            average_precision_score(y[inner_validation_index], probability)
                        )
                        candidate_scores[key].append(average_precision)
                    else:
                        average_precision = math.nan
                        invalid_candidates[key].append(f"inner fold {inner_fold}: {message}")
                    tuning_rows.append(
                        {
                            **common_audit,
                            "N": n_candidate,
                            "C": c_candidate,
                            "average_precision": average_precision,
                            "selected_features": json.dumps(selected_names),
                            "selected_feature_indices": json.dumps(selected_indices.tolist()),
                            "converged": converged,
                            "convergence_warning": message,
                            "valid_for_tuning": converged,
                        }
                    )

        eligible = [
            (float(np.mean(scores)), n_candidate, c_candidate)
            for (n_candidate, c_candidate), scores in candidate_scores.items()
            if not invalid_candidates[(n_candidate, c_candidate)]
            and len(scores) == INNER_N_SPLITS
        ]
        if not eligible:
            raise RuntimeError(
                f"outer fold {fold}: no candidate (N, C) converged across all "
                f"{INNER_N_SPLITS} inner folds; tuning is invalid"
            )
        # ``sorted`` makes the descending-AP, ascending-N, ascending-C rule explicit.
        ordered = sorted(eligible, key=lambda item: (-item[0], item[1], item[2]))
        inner_mean_ap, best_n, best_c = ordered[0]
        rank_by_candidate = {
            (n_candidate, c_candidate): rank
            for rank, (_, n_candidate, c_candidate) in enumerate(ordered, start=1)
        }
        for n_candidate, c_candidate in candidates:
            scores = candidate_scores[(n_candidate, c_candidate)]
            is_eligible = (n_candidate, c_candidate) in rank_by_candidate
            summary_rows.append(
                {
                    "outer_fold": fold,
                    "N": n_candidate,
                    "C": c_candidate,
                    "mean_inner_average_precision": (
                        float(np.mean(scores)) if is_eligible else math.nan
                    ),
                    "std_inner_average_precision": (
                        float(np.std(scores, ddof=1))
                        if is_eligible and len(scores) > 1 else math.nan
                    ),
                    "n_inner_folds_scored": len(scores),
                    "n_inner_folds": INNER_N_SPLITS,
                    "all_inner_folds_converged": is_eligible,
                    "rank": rank_by_candidate.get((n_candidate, c_candidate), math.nan),
                    "selected_best": (n_candidate, c_candidate) == (best_n, best_c),
                    "tie_break_rule": MODEL_TIE_BREAK_RULE,
                }
            )

        # Threshold selection: inner out-of-fold probabilities of the winning
        # pair, covering every outer-training row exactly once.
        inner_oof_positions = np.concatenate(
            [inner_validation_positions[inner_fold] for inner_fold in range(1, INNER_N_SPLITS + 1)]
        )
        inner_oof_probability = np.concatenate(
            [
                probability_cache[(inner_fold, best_n, best_c)]
                for inner_fold in range(1, INNER_N_SPLITS + 1)
            ]
        )
        if len(inner_oof_positions) != len(inner_oof_probability):
            raise AssertionError("inner out-of-fold probabilities are misaligned")
        if sorted(inner_oof_positions.tolist()) != sorted(outer_train_index.tolist()):
            raise AssertionError(
                f"outer fold {fold}: inner out-of-fold probabilities must cover every "
                "outer-training observation exactly once"
            )
        if set(inner_oof_positions.tolist()) & set(outer_test_index.tolist()):
            raise AssertionError("an outer-test row reached the threshold search")
        inner_oof_labels = y[inner_oof_positions]
        trace, chosen_threshold = select_threshold(
            inner_oof_labels, inner_oof_probability, minimum
        )
        trace.insert(0, "outer_fold", fold)
        trace_rows.append(trace)
        inner_oof = dataset.iloc[inner_oof_positions][FINAL_KEY + ["dataset", "target"]].copy()
        inner_oof = inner_oof.rename(columns={"target": "true_label"})
        inner_oof.insert(0, "outer_fold", fold)
        inner_oof["inner_fold"] = np.concatenate(
            [
                np.full(len(inner_validation_positions[inner_fold]), inner_fold, dtype=int)
                for inner_fold in range(1, INNER_N_SPLITS + 1)
            ]
        )
        inner_oof["inner_oof_probability"] = inner_oof_probability
        inner_oof["selected_N"] = int(best_n)
        inner_oof["selected_C"] = float(best_c)
        inner_oof_rows.append(inner_oof)
        threshold_rows.append(
            {
                "outer_fold": fold,
                "selected_N": int(best_n),
                "selected_C": float(best_c),
                "minimum_TPR": minimum,
                "selected_threshold": chosen_threshold["threshold"],
                "inner_oof_ppv": chosen_threshold["ppv"],
                "inner_oof_tpr": chosen_threshold["tpr"],
                "inner_oof_specificity": chosen_threshold["specificity"],
                "inner_oof_fpr": chosen_threshold["fpr"],
                "inner_oof_fnr": chosen_threshold["fnr"],
                "n_predicted_positive": chosen_threshold["n_predicted_positive"],
                "inner_oof_tp": chosen_threshold["tp"],
                "inner_oof_fp": chosen_threshold["fp"],
                "inner_oof_tn": chosen_threshold["tn"],
                "inner_oof_fn": chosen_threshold["fn"],
                "n_candidate_thresholds": chosen_threshold["n_candidates"],
                "n_eligible_thresholds": chosen_threshold["n_eligible_candidates"],
                "n_inner_oof_observations": int(len(inner_oof_probability)),
                "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
                "threshold_candidate_rule": THRESHOLD_CANDIDATE_RULE,
                "threshold_selected_from": "inner out-of-fold probabilities of the outer TRAIN",
            }
        )

        # N_best, C_best and the threshold are now frozen: the scaler, the mRMR
        # statistics and the weights are recomputed from scratch on the full
        # outer training fold before the single outer-test prediction.
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
        model, probability, converged, message = _fit_scored_model(
            x_outer_train[:, selected_indices],
            y[outer_train_index],
            outer_weights["final_sample_weight"],
            x_outer_test[:, selected_indices],
            best_c,
        )
        standard_predicted = (probability >= DECISION_THRESHOLD).astype(int)
        optimized_predicted = (probability >= chosen_threshold["threshold"]).astype(int)
        selected_names = [features[index] for index in selected_indices]
        selected_feature_rows.extend(
            {
                "fold_id": fold,
                "selection_rank": rank,
                "feature": features[feature_index],
                "feature_index": int(feature_index),
                "selected_n": int(best_n),
                "selected_c": float(best_c),
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
                "selected_C": float(best_c),
                "actual_n_features": int(outer_selector.actual_n_features_),
                "mean_inner_average_precision": inner_mean_ap,
                "std_inner_average_precision": (
                    float(np.std(candidate_scores[(best_n, best_c)], ddof=1))
                    if len(candidate_scores[(best_n, best_c)]) > 1 else math.nan
                ),
                "n_candidates_evaluated": len(candidates),
                "n_eligible_candidates": len(eligible),
                "tie_break_rule": MODEL_TIE_BREAK_RULE,
                "selected_threshold": chosen_threshold["threshold"],
                "minimum_TPR": minimum,
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
        # ``pathological_score`` is the column name the shared evaluators expect;
        # ``pathological_probability`` is the same vector under the name this
        # experiment publishes.
        fold_predictions["pathological_score"] = probability
        fold_predictions["pathological_probability"] = probability
        fold_predictions["predicted_label"] = standard_predicted
        fold_predictions["predicted_label_threshold_0p5"] = standard_predicted
        fold_predictions["optimized_threshold"] = chosen_threshold["threshold"]
        fold_predictions["predicted_label_optimized_threshold"] = optimized_predicted
        fold_predictions["minimum_TPR"] = minimum
        fold_predictions["selected_N"] = int(best_n)
        fold_predictions["selected_C"] = float(best_c)
        fold_predictions["fold_id"] = fold
        predictions.append(fold_predictions)

        standard_metrics = calculate_metrics(y[outer_test_index], probability)
        optimized_metrics = metrics_from_labels(
            y[outer_test_index], probability, optimized_predicted
        )
        for name in PROBABILITY_METRIC_NAMES:
            first, second = standard_metrics[name], optimized_metrics[name]
            if not (
                math.isnan(first) and math.isnan(second)
            ) and not math.isclose(first, second, rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError(
                    f"outer fold {fold}: {name} must not depend on the threshold"
                )
        fold_metrics.extend(
            {"fold_id": fold, "metric": name, "value": value}
            for name, value in standard_metrics.items()
        )
        standard_metric_rows.append(
            {
                "outer_fold": fold,
                "threshold": DECISION_THRESHOLD,
                "threshold_source": "frozen protocol threshold",
                "selected_N": int(best_n),
                "selected_C": float(best_c),
                "n_channels": int(len(outer_test_index)),
                "n_participants": int(outer_test["participant_id"].nunique()),
                **metrics_from_labels(y[outer_test_index], probability, standard_predicted),
            }
        )
        optimized_metric_rows.append(
            {
                "outer_fold": fold,
                "threshold": chosen_threshold["threshold"],
                "threshold_source": "inner out-of-fold selection on the outer TRAIN",
                "minimum_TPR": minimum,
                "selected_N": int(best_n),
                "selected_C": float(best_c),
                "n_channels": int(len(outer_test_index)),
                "n_participants": int(outer_test["participant_id"].nunique()),
                **optimized_metrics,
            }
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
                "selected_C": float(best_c),
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
        "outer_fold_metrics_threshold_0p5": pd.DataFrame(standard_metric_rows),
        "outer_fold_metrics_optimized_threshold": pd.DataFrame(optimized_metric_rows),
        "selected_features_by_fold": pd.DataFrame(selected_feature_rows),
        "nested_model_tuning": pd.DataFrame(tuning_rows),
        "nested_model_tuning_summary": pd.DataFrame(summary_rows),
        "selected_hyperparameters_by_outer_fold": pd.DataFrame(chosen_rows),
        "threshold_tuning_by_outer_fold": pd.DataFrame(threshold_rows),
        "threshold_tuning_trace": pd.concat(trace_rows, ignore_index=True),
        "threshold_inner_oof_predictions": pd.concat(inner_oof_rows, ignore_index=True),
    }


def run_nested_tuned_evaluation(
    dataset_path: Path,
    feature_list_path: Path,
    output_dir: Path,
    *,
    expected_predictors: Iterable[str],
    feature_schema: dict[str, Any],
    protocol_version: str = PROTOCOL_VERSION,
    stage: str = STAGE,
    repository: Path = REPOSITORY,
    n_bootstrap: int = N_BOOTSTRAP,
    n_grid: Iterable[int] = TUNED_N_GRID,
    c_grid: Iterable[float] = TUNED_C_GRID,
    minimum_tpr: float = DEFAULT_MINIMUM_TPR,
    reference_fold_assignments: Path | None = None,
) -> Path:
    """Publish the tuned artifacts plus the standard metric/bootstrap artifacts."""
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
    regularization = _validate_c_grid(c_grid)
    minimum = validate_minimum_tpr(minimum_tpr)
    cv_tables = run_nested_tuned_cross_validation(
        dataset,
        schema_predictors,
        n_grid=grid,
        c_grid=regularization,
        minimum_tpr=minimum,
    )
    if reference_fold_assignments is not None:
        assert_matching_fold_assignments(
            cv_tables["fold_assignments"], pd.read_csv(Path(reference_fold_assignments))
        )
    cv_tables["selection_frequencies"] = selection_frequencies(
        cv_tables["selected_features_by_fold"], schema_predictors
    )
    evaluation = evaluation_tables(cv_tables["oof_predictions"], n_bootstrap=n_bootstrap)
    optimized = optimized_threshold_evaluation_tables(
        cv_tables["oof_predictions"], n_bootstrap=n_bootstrap
    )
    optimized["standard_threshold_overall_site_metrics"] = overall_and_site_metrics(
        cv_tables["oof_predictions"], "predicted_label_threshold_0p5"
    )
    # The companion 0.5 table must agree with the shared evaluator on every
    # metric they have in common; it only adds FPR and FNR.
    shared = evaluation_tables_agreement(
        evaluation["overall_site_metrics"],
        optimized["standard_threshold_overall_site_metrics"],
    )
    if shared:
        raise AssertionError(
            f"the companion threshold-0.5 table disagrees with the shared evaluator: {shared}"
        )
    # AUROC and Average Precision come from continuous probabilities and must be
    # byte-identical between the two evaluations; this asserts it rather than
    # asserting it in prose only.
    standard_overall = optimized["standard_threshold_overall_site_metrics"]
    optimized_overall = optimized["optimized_threshold_overall_site_metrics"]
    for metric in PROBABILITY_METRIC_NAMES:
        left = standard_overall[standard_overall["metric"].eq(metric)].set_index(
            ["scope", "scope_value"]
        )["value"]
        right = optimized_overall[optimized_overall["metric"].eq(metric)].set_index(
            ["scope", "scope_value"]
        )["value"]
        if not np.allclose(
            left.sort_index().to_numpy(dtype=float),
            right.sort_index().to_numpy(dtype=float),
            rtol=0.0, atol=1e-12, equal_nan=True,
        ):
            raise AssertionError(f"{metric} changed with the threshold; this is impossible")
    selected = cv_tables["selected_hyperparameters_by_outer_fold"]
    thresholds = cv_tables["threshold_tuning_by_outer_fold"]
    with new_output_directory(output_dir) as staging:
        for name, table in {**cv_tables, **evaluation, **optimized}.items():
            atomic_csv(table, staging / f"{name}.csv")
        summary = {
            "experiment_id": EXPERIMENT_ID,
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
                "purpose": "joint (N, C) tuning and threshold selection",
            },
            "outer_folds_match_complexity_protocol": True,
            "reference_fold_assignments": (
                str(Path(reference_fold_assignments).resolve())
                if reference_fold_assignments is not None else None
            ),
            "normalization": (
                "StandardScaler fit independently on each inner-training split during "
                "tuning and on each full outer training fold for final fitting"
            ),
            "weighting": (
                "inner-training-only and outer-training-only inverse participant channel count "
                "followed by equal participant-weight class mass and normalization to mean one"
            ),
            "model": {
                "class": "LogisticRegression",
                "penalty": "l2",
                "class_weight": None,
                "solver": "lbfgs",
                "max_iter": 5000,
                "C": "tuned per outer fold",
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
                "tie_breaking": MODEL_TIE_BREAK_RULE,
                "fit_scope": (
                    "recomputed inside every inner training split while tuning, then from "
                    "scratch on the outer training rows only for the final fit"
                ),
                "preselected_on_full_outer_train_before_inner_cv": False,
                "test_data_seen_by_selector": False,
                "selected_n_by_outer_fold": {
                    str(int(row["outer_fold"])): int(row["selected_N"])
                    for _, row in selected.iterrows()
                },
                "selected_n_unique": sorted(set(selected["selected_N"].astype(int).tolist())),
            },
            "hyperparameter_grid": {
                "n_selected_features": list(grid),
                "C": list(regularization),
                "n_candidates": len(grid) * len(regularization),
            },
            "tuning_metric": "average_precision",
            "tie_breaking": MODEL_TIE_BREAK_RULE,
            "selected_c_by_outer_fold": {
                str(int(row["outer_fold"])): float(row["selected_C"])
                for _, row in selected.iterrows()
            },
            "hyperparameters_chosen_from_outer_performance": False,
            "final_n_chosen_from_outer_performance": False,
            "decision_threshold": DECISION_THRESHOLD,
            "threshold_was_optimized": True,
            "threshold_optimization": {
                "objective": "maximize PPV subject to TPR >= minimum_TPR",
                "minimum_tpr": minimum,
                "minimum_tpr_is_configurable": True,
                "selection_rule": THRESHOLD_SELECTION_RULE,
                "candidate_rule": THRESHOLD_CANDIDATE_RULE,
                "selected_from": (
                    "inner out-of-fold probabilities produced by the same grouped inner "
                    "CV inside the outer training fold"
                ),
                "outer_test_used_for_threshold": False,
                "fallback_threshold_on_failure": None,
                "selected_threshold_by_outer_fold": {
                    str(int(row["outer_fold"])): float(row["selected_threshold"])
                    for _, row in thresholds.iterrows()
                },
                "affects": "threshold-dependent metrics only",
                "auroc_and_average_precision_are_threshold_free": True,
            },
            "feature_counts": feature_schema,
            "imputation": False,
            "resampling": False,
            "dimensionality_reduction": False,
            "bootstrap": {
                "unit": "participant",
                "refit_models": False,
                "reoptimizes_threshold": False,
                "threshold_per_observation": "the threshold of the outer fold that tested it",
                "n_replicates": n_bootstrap,
                "random_state": BOOTSTRAP_RANDOM_STATE,
                "confidence_level": CONFIDENCE_LEVEL,
                "interval": "percentile",
                "interpretation": (
                    "descriptive uncertainty conditional on the fixed out-of-fold "
                    "predictions and the already selected thresholds"
                ),
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
