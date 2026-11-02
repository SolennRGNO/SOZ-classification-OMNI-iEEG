"""Cross-modality generalization experiment ``ml_crossmodality_v1``.

Train on every participant of one recording modality (ECoG or SEEG) and
evaluate **once** on every participant of the opposite modality.  Nothing is
preprocessed, extracted or refitted here that the published experiments do not
already own: the participant-by-channel matrices, the clinical labels, the
scaler, the participant/class weighting, mRMR, the Elastic Net grid, the
metric definitions, the threshold search and the
participant-cluster bootstrap are all imported from the existing modules.

What is new is only the *split*: instead of :class:`StratifiedGroupKFold` over
all participants, the held-out set is the whole opposite modality.  Because
every participant in this cohort carries exactly one clinical
``channel_type`` (verified at runtime, not assumed), the modality split is also
a participant split, so the grouped-CV leakage guarantees carry over unchanged.

Leakage contract
----------------
Everything that is *learned* -- the scaler, the feature selection, the inner
cross-validation, the hyperparameters and the classification threshold -- is
learned on the training modality alone.  The threshold is chosen from
out-of-fold probabilities produced by a grouped cross-validation *inside* the
training modality, then frozen.  The final model is refit on all training
participants and applied to the opposite modality exactly once.

Confounding
-----------
Site and modality are confounded in this cohort (HUP is almost entirely SEEG,
UCLA almost entirely ECoG), so a modality contrast is partly a site contrast.
This experiment is exploratory and is labelled as such in every artifact.
"""

from __future__ import annotations

import json
import math
import platform
import warnings
import argparse
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
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
from scripts.ml.dataset import FINAL_KEY, predictor_names as baseline_predictor_names
from scripts.ml.feature_selection import MRMR_MI_PARAMETERS
from scripts.ml.logistic_baseline import (
    BOOTSTRAP_RANDOM_STATE,
    CONFIDENCE_LEVEL,
    CV_RANDOM_STATE,
    DECISION_THRESHOLD,
    ELASTICNET_C_GRID,
    ELASTICNET_L1_RATIO_GRID,
    INNER_N_SPLITS,
    N_BOOTSTRAP,
    N_SPLITS,
    _elasticnet_convergence_status,
    _validate_predictor_schema,
    fit_training_scaler,
    make_elasticnet_model,
    participant_and_class_weights,
    resample_participant_clusters,
)
from scripts.ml.nested_mrmr import MRMR_N_GRID, _fit_mrmr
from scripts.ml.nested_tuned import (
    DEFAULT_MINIMUM_TPR,
    TUNED_METRIC_NAMES,
    _fit_scored_model,
    metrics_from_labels,
    select_threshold,
    validate_minimum_tpr,
)
from scripts.ml.random_forest_nested import (
    RF_MAX_DEPTH_GRID,
    RF_MAX_FEATURES_GRID,
    RF_MIN_SAMPLES_LEAF_GRID,
    RF_N_ESTIMATORS,
    _depth_order,
    _label,
    candidate_grid,
    make_random_forest,
    max_features_fraction,
    resolve_n_jobs,
    validate_grids,
)


EXPERIMENT_ID = "ml_crossmodality_v1"
PROTOCOL_VERSION = EXPERIMENT_ID
STAGE = "crossmodality_train_one_modality_test_the_other"

# The authoritative clinical electrode modality lives in ``channel_type``,
# which the dataset builders copy from the original BIDS channels.tsv through
# the channel-label audit.  ``mne_signal_type`` is an MNE internal signal kind
# ("EEG" for both) and is never used to define a modality here.
MODALITY_COLUMN = "channel_type"
MODALITIES = ("ECOG", "SEEG")
DIRECTIONS = (("ECOG", "SEEG"), ("SEEG", "ECOG"))

# The threshold procedure uses the published five grouped folds, so each source
# modality must have at least five participants.  The target-side bootstrap
# additionally requires at least two participants; the source requirement is
# stricter in this two-direction design.
MINIMUM_SIDE_PARTICIPANTS = N_SPLITS

# EXP3A reuses the published fixed-N mRMR operating point.
FIXED_MRMR_N = 8

CONFOUND_NOTE = (
    "Site and modality are confounded in this cohort: HUP contributes about 1 ECoG and "
    "9 SEEG participants, UCLA about 15 ECoG and 4 SEEG participants. A cross-modality "
    "contrast is therefore also, to a large extent, a cross-site contrast. These results "
    "are exploratory cross-modality generalization, not an isolated modality effect."
)
THRESHOLD_SELECTION_RULE = (
    "maximize PPV subject to TPR >= minimum_TPR on out-of-fold probabilities produced by "
    "a grouped cross-validation inside the training modality only; then highest TPR; then "
    "lowest threshold"
)
FIXED_THRESHOLD_TYPE = "fixed_0.5"
OPTIMIZED_THRESHOLD_TYPE = "optimized_train_modality"
THRESHOLD_TYPES = (FIXED_THRESHOLD_TYPE, OPTIMIZED_THRESHOLD_TYPE)

# Reported metrics.  ``sensitivity`` is TPR and ``precision`` is PPV; the names
# are the ones the existing evaluators already publish, so the cross-modality
# tables can be read next to the pooled-OOF tables without a translation table.
# Include the complete requested held-out confusion matrix in the participant
# bootstrap as well as the probability/rate metrics.
METRIC_NAMES = TUNED_METRIC_NAMES + (
    "true_positive", "false_positive", "true_negative", "false_negative",
)
METRIC_ALIASES = {"sensitivity": "TPR", "precision": "PPV", "auroc": "AUROC",
                  "average_precision": "AUPRC / AP", "specificity": "specificity",
                  "f1": "F1", "fpr": "FPR", "fnr": "FNR",
                  "balanced_accuracy": "balanced accuracy"}


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
class CrossModalityExperiment(NamedTuple):
    """One published model definition, reused verbatim across the modality split."""

    key: str
    short: str
    label: str
    family: str  # "baseline" (96 predictors) or "complexity" (42 predictors)
    fitter: str  # key into FITTERS
    description: str


EXPERIMENTS = (
    CrossModalityExperiment(
        "exp1_baseline", "EXP1", "EXP1 - Conventional baseline", "baseline", "l2_all",
        "L2 logistic regression (C=1) on all baseline predictors; no selection, no tuning",
    ),
    CrossModalityExperiment(
        "exp2_complexity_all", "EXP2", "EXP2 - Complexity, no selection", "complexity", "l2_all",
        "L2 logistic regression (C=1) on all 42 complexity predictors; no selection, no tuning",
    ),
    CrossModalityExperiment(
        "exp3a_mrmr_fixed_n8", "EXP3A", "EXP3A - mRMR fixed N=8", "complexity", "mrmr_fixed",
        f"train-only mRMR with a fixed N={FIXED_MRMR_N} then L2 logistic regression (C=1)",
    ),
    CrossModalityExperiment(
        "exp3b_mrmr_nested", "EXP3B", "EXP3B - Nested mRMR", "complexity", "mrmr_nested",
        "train-only grouped inner CV tunes the mRMR N, then L2 logistic regression (C=1)",
    ),
    CrossModalityExperiment(
        "exp4_elasticnet_nested", "EXP4", "EXP4 - Nested Elastic Net", "complexity",
        "elasticnet_nested",
        "train-only grouped inner CV tunes Elastic Net C and l1_ratio on all 42 predictors",
    ),
)
DEFAULT_EXPERIMENT_KEYS = tuple(experiment.key for experiment in EXPERIMENTS)
EXPERIMENTS_BY_KEY = {experiment.key: experiment for experiment in EXPERIMENTS}


def resolve_experiments(keys: Iterable[str] | None) -> tuple[CrossModalityExperiment, ...]:
    """Validate an explicit experiment selection, keeping the frozen registry order."""
    requested = list(DEFAULT_EXPERIMENT_KEYS) if keys is None else [str(key).strip() for key in keys]
    if not requested:
        raise ValueError("at least one experiment key is required")
    unknown = sorted(set(requested) - set(EXPERIMENTS_BY_KEY))
    if unknown:
        raise ValueError(
            f"unknown experiment keys {unknown}; available: {sorted(EXPERIMENTS_BY_KEY)}"
        )
    duplicates = sorted({key for key in requested if requested.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate experiment keys requested: {duplicates}")
    selected = set(requested)
    return tuple(item for item in EXPERIMENTS if item.key in selected)


# ---------------------------------------------------------------------------
# Cohort: modality split and its audit
# ---------------------------------------------------------------------------
def normalize_modality(dataset: pd.DataFrame) -> pd.DataFrame:
    """Upper-case the authoritative ``channel_type`` without touching anything else."""
    if MODALITY_COLUMN not in dataset.columns:
        raise ValueError(
            f"the dataset lacks the authoritative {MODALITY_COLUMN!r} column; the modality "
            "split cannot be derived from mne_signal_type, which is 'EEG' for every channel"
        )
    work = dataset.copy()
    work[MODALITY_COLUMN] = (
        work[MODALITY_COLUMN].astype("string").str.strip().str.upper().fillna("")
    )
    return work


def select_modalities(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only ECoG and SEEG channels; return the kept rows and an audit of the rest."""
    work = normalize_modality(dataset)
    keep = work[MODALITY_COLUMN].isin(MODALITIES)
    excluded = (
        work.loc[~keep]
        .groupby(MODALITY_COLUMN, dropna=False)
        .agg(n_channels=("channel_id", "size"), n_participants=("participant_id", "nunique"))
        .reset_index()
        .rename(columns={MODALITY_COLUMN: "excluded_channel_type"})
    )
    kept = work.loc[keep].reset_index(drop=True)
    if kept.empty:
        raise ValueError(f"no {list(MODALITIES)} channels remain after the modality filter")
    observed = sorted(set(kept[MODALITY_COLUMN]))
    if observed != sorted(MODALITIES):
        raise ValueError(
            f"the cross-modality experiment needs both modalities; found only {observed}"
        )
    return kept, excluded


def participant_modalities(dataset: pd.DataFrame) -> pd.DataFrame:
    """One row per participant with its modality, site and label counts."""
    grouped = dataset.groupby("participant_id", sort=True)
    table = grouped.agg(
        n_channel_types=(MODALITY_COLUMN, "nunique"),
        channel_type=(MODALITY_COLUMN, "first"),
        n_sites=("dataset", "nunique"),
        site=("dataset", "first"),
        n_channels=("channel_id", "size"),
        n_pathological=("target", "sum"),
    ).reset_index()
    table["n_non_pathological"] = table["n_channels"] - table["n_pathological"]
    mixed = table[table["n_channel_types"].gt(1)]
    if not mixed.empty:
        raise ValueError(
            "the cross-modality split requires each participant to carry exactly one "
            "clinical channel_type, because the participant is the independent unit; "
            f"mixed-modality participants: {sorted(mixed['participant_id'])}"
        )
    if table["n_sites"].gt(1).any():
        raise ValueError("a participant belongs to more than one site; the cohort is inconsistent")
    return table.drop(columns=["n_channel_types", "n_sites"])


def cohort_tables(dataset: pd.DataFrame, excluded: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Participants / channels / labels per modality, and site x modality counts."""
    participants = participant_modalities(dataset)
    by_modality = (
        participants.groupby("channel_type", sort=True)
        .agg(
            n_participants=("participant_id", "size"),
            n_channels=("n_channels", "sum"),
            n_pathological=("n_pathological", "sum"),
            n_non_pathological=("n_non_pathological", "sum"),
        )
        .reset_index()
    )
    by_modality["prevalence"] = by_modality["n_pathological"] / by_modality["n_channels"]
    site_by_modality = (
        participants.groupby(["site", "channel_type"], sort=True)
        .agg(
            n_participants=("participant_id", "size"),
            n_channels=("n_channels", "sum"),
            n_pathological=("n_pathological", "sum"),
            n_non_pathological=("n_non_pathological", "sum"),
        )
        .reset_index()
    )
    return {
        "cohort_participants": participants,
        "cohort_by_modality": by_modality,
        "cohort_site_by_modality": site_by_modality,
        "cohort_excluded_channel_types": excluded,
    }


def assert_disjoint(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Fail loudly unless the two modality sides share no participant at all."""
    train_ids = set(train["participant_id"].astype(str))
    test_ids = set(test["participant_id"].astype(str))
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise AssertionError(f"train and test modality share participants: {overlap}")
    return {
        "n_train_participants": len(train_ids),
        "n_test_participants": len(test_ids),
        "participant_overlap_count": 0,
        "train_participant_ids": sorted(train_ids),
        "test_participant_ids": sorted(test_ids),
    }


def format_cohort_report(
    task: str, tables: dict[str, pd.DataFrame], directions: Sequence[dict[str, Any]]
) -> str:
    """Human-readable pre-training report; printed before any model is fit."""
    lines = [
        f"Cross-modality cohort report ({EXPERIMENT_ID}, task={task})",
        "",
        "Participants / channels / labels per modality:",
        tables["cohort_by_modality"].to_string(index=False),
        "",
        "Site x modality counts:",
        tables["cohort_site_by_modality"].to_string(index=False),
        "",
    ]
    excluded = tables["cohort_excluded_channel_types"]
    lines.append(
        "Excluded channel types: none"
        if excluded.empty
        else "Excluded channel typeCONFIGURE_WINDOWS_PATH" + excluded.to_string(index=False)
    )
    lines.append("")
    for direction in directions:
        lines.append(
            f"{direction['train_modality']} -> {direction['test_modality']}: "
            f"train {direction['n_train_participants']} participants, "
            f"test {direction['n_test_participants']} participants, "
            f"participant overlap = {direction['participant_overlap_count']} (confirmed zero)"
        )
    lines.extend(["", CONFOUND_NOTE])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fitting: every quantity below is learned on the supplied training rows only
# ---------------------------------------------------------------------------
class FitContext(NamedTuple):
    """Everything a fitter may see.  ``x_eval`` carries no labels, by design."""

    x_train: np.ndarray
    y_train: np.ndarray
    train_participants: np.ndarray
    x_eval: np.ndarray
    train_ids: frozenset
    eval_ids: frozenset
    features: list[str]
    n_jobs: int


class Fitted(NamedTuple):
    scores: np.ndarray
    record: dict[str, Any]


def _weights(participants: Iterable[object], labels: np.ndarray) -> np.ndarray:
    frame, _ = participant_and_class_weights(participants, labels)
    return frame["final_sample_weight"].to_numpy(dtype=float)


def _scale(context: FitContext) -> tuple[Any, np.ndarray, np.ndarray]:
    """Fit the shared StandardScaler on training rows only and apply it to both sides."""
    scaler = fit_training_scaler(context.x_train, context.train_ids, context.eval_ids)
    return scaler, scaler.transform(context.x_train), scaler.transform(context.x_eval)


def _inner_splits(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped inner splits of the training rows; both classes required on both sides."""
    splitter = StratifiedGroupKFold(
        n_splits=INNER_N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE
    )
    try:
        splits = list(splitter.split(x, y, groups))
    except ValueError as error:
        raise ValueError(
            f"a valid {INNER_N_SPLITS}-fold grouped inner CV cannot be built from the "
            "training modality"
        ) from error
    if len(splits) != INNER_N_SPLITS:
        raise ValueError(f"expected {INNER_N_SPLITS} inner folds, got {len(splits)}")
    for index, (train_local, validation_local) in enumerate(splits, start=1):
        train_ids = set(map(str, groups[train_local]))
        validation_ids = set(map(str, groups[validation_local]))
        if train_ids & validation_ids:
            raise AssertionError(f"inner fold {index}: participant overlap")
        if set(y[train_local]) != {0, 1} or set(y[validation_local]) != {0, 1}:
            raise ValueError(f"inner fold {index}: both classes are required on both sides")
    return splits


def _inner_prepared(
    context: FitContext, train_local: np.ndarray, validation_local: np.ndarray, scale: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inner-training-only scaler and weights; returns (x_train, x_validation, weights)."""
    train_ids = frozenset(map(str, context.train_participants[train_local]))
    validation_ids = frozenset(map(str, context.train_participants[validation_local]))
    x_train = context.x_train[train_local]
    x_validation = context.x_train[validation_local]
    if scale:
        scaler = fit_training_scaler(x_train, train_ids, validation_ids)
        x_train = scaler.transform(x_train)
        x_validation = scaler.transform(x_validation)
    elif train_ids & validation_ids:
        raise AssertionError("inner split participant overlap")
    weights = _weights(context.train_participants[train_local], context.y_train[train_local])
    return x_train, x_validation, weights


def fit_l2_all(context: FitContext) -> Fitted:
    """EXP1 / EXP2: the frozen L2 model at C=1 on every supplied predictor."""
    _, x_train, x_eval = _scale(context)
    weights = _weights(context.train_participants, context.y_train)
    model, scores, converged, message = _fit_scored_model(
        x_train, context.y_train, weights, x_eval, C=1.0
    )
    return Fitted(
        scores,
        {
            "model": "LogisticRegression(penalty=l2, C=1.0, solver=lbfgs)",
            "scaled": True,
            "feature_selection": False,
            "n_selected_features": len(context.features),
            "selected_features": list(context.features),
            "hyperparameters": {"C": 1.0},
            "converged": bool(converged),
            "convergence_warning": message,
            "intercept": float(model.intercept_[0]),
        },
    )


def fit_mrmr_fixed(context: FitContext) -> Fitted:
    """EXP3A: train-only mRMR at the published fixed N, then the frozen L2 model."""
    _, x_train, x_eval = _scale(context)
    selector = _fit_mrmr(x_train, context.y_train, FIXED_MRMR_N)
    indices = np.asarray(selector.selected_indices_, dtype=int)
    weights = _weights(context.train_participants, context.y_train)
    model, scores, converged, message = _fit_scored_model(
        x_train[:, indices], context.y_train, weights, x_eval[:, indices], C=1.0
    )
    return Fitted(
        scores,
        {
            "model": "LogisticRegression(penalty=l2, C=1.0, solver=lbfgs)",
            "scaled": True,
            "feature_selection": "mRMR (MID) fit on training rows only",
            "mutual_information_estimator": MRMR_MI_PARAMETERS,
            "requested_n_features": FIXED_MRMR_N,
            "n_selected_features": int(selector.actual_n_features_),
            "selected_features": [context.features[index] for index in indices],
            "hyperparameters": {"C": 1.0, "N": FIXED_MRMR_N},
            "converged": bool(converged),
            "convergence_warning": message,
            "intercept": float(model.intercept_[0]),
        },
    )


def fit_mrmr_nested(context: FitContext) -> Fitted:
    """EXP3B: grouped inner CV on the training rows tunes the mRMR N, then refits."""
    grid = tuple(n for n in MRMR_N_GRID if n <= len(context.features))
    if not grid:
        raise ValueError("no mRMR N candidate fits the supplied predictor schema")
    splits = _inner_splits(context.x_train, context.y_train, context.train_participants)
    scores_by_n: dict[int, list[float]] = {n: [] for n in grid}
    inner_rows: list[dict[str, Any]] = []
    for inner_fold, (train_local, validation_local) in enumerate(splits, start=1):
        x_inner, x_validation, weights = _inner_prepared(context, train_local, validation_local, True)
        y_inner = context.y_train[train_local]
        y_validation = context.y_train[validation_local]
        for n_features in grid:
            selector = _fit_mrmr(x_inner, y_inner, n_features)
            indices = np.asarray(selector.selected_indices_, dtype=int)
            _, probability, converged, message = _fit_scored_model(
                x_inner[:, indices], y_inner, weights, x_validation[:, indices], C=1.0
            )
            average_precision = float(average_precision_score(y_validation, probability))
            scores_by_n[n_features].append(average_precision)
            inner_rows.append(
                {
                    "inner_fold": inner_fold,
                    "N": int(n_features),
                    "average_precision": average_precision,
                    "converged": bool(converged),
                    "convergence_warning": message,
                }
            )
    # Frozen tie rule of the published nested-mRMR experiment: highest mean
    # inner Average Precision, then the smallest N.
    best_mean, best_n = min(
        ((float(np.mean(values)), n) for n, values in scores_by_n.items()),
        key=lambda item: (-item[0], item[1]),
    )
    _, x_train, x_eval = _scale(context)
    selector = _fit_mrmr(x_train, context.y_train, best_n)
    indices = np.asarray(selector.selected_indices_, dtype=int)
    weights = _weights(context.train_participants, context.y_train)
    model, scores, converged, message = _fit_scored_model(
        x_train[:, indices], context.y_train, weights, x_eval[:, indices], C=1.0
    )
    return Fitted(
        scores,
        {
            "model": "LogisticRegression(penalty=l2, C=1.0, solver=lbfgs)",
            "scaled": True,
            "feature_selection": "mRMR (MID) with N tuned by grouped inner CV on training rows only",
            "mutual_information_estimator": MRMR_MI_PARAMETERS,
            "n_grid": list(grid),
            "inner_cv": {"class": "StratifiedGroupKFold", "n_splits": INNER_N_SPLITS},
            "tuning_metric": "average_precision",
            "tie_breaking": "highest mean inner Average Precision; then smallest N",
            "best_N": int(best_n),
            "inner_mean_average_precision": float(best_mean),
            "inner_cv_results": inner_rows,
            "n_selected_features": int(selector.actual_n_features_),
            "selected_features": [context.features[index] for index in indices],
            "hyperparameters": {"C": 1.0, "N": int(best_n)},
            "converged": bool(converged),
            "convergence_warning": message,
            "intercept": float(model.intercept_[0]),
        },
    )


def fit_elasticnet_nested(context: FitContext) -> Fitted:
    """EXP4: grouped inner CV on the training rows tunes C and l1_ratio, then refits."""
    splits = _inner_splits(context.x_train, context.y_train, context.train_participants)
    candidates = [(C, l1) for C in ELASTICNET_C_GRID for l1 in ELASTICNET_L1_RATIO_GRID]
    scores_by_candidate: dict[tuple[float, float], list[float]] = {key: [] for key in candidates}
    invalid: dict[tuple[float, float], list[str]] = {key: [] for key in candidates}
    inner_rows: list[dict[str, Any]] = []
    for inner_fold, (train_local, validation_local) in enumerate(splits, start=1):
        x_inner, x_validation, weights = _inner_prepared(context, train_local, validation_local, True)
        y_inner = context.y_train[train_local]
        y_validation = context.y_train[validation_local]
        for C, l1_ratio in candidates:
            model = make_elasticnet_model(C, l1_ratio)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x_inner, y_inner, sample_weight=weights)
            converged, message = _elasticnet_convergence_status(model, caught)
            if converged:
                probability = model.predict_proba(x_validation)[
                    :, int(np.flatnonzero(model.classes_ == 1)[0])
                ]
                average_precision = float(average_precision_score(y_validation, probability))
                scores_by_candidate[(C, l1_ratio)].append(average_precision)
            else:
                average_precision = math.nan
                invalid[(C, l1_ratio)].append(f"inner fold {inner_fold}: {message}")
            inner_rows.append(
                {
                    "inner_fold": inner_fold, "C": C, "l1_ratio": l1_ratio,
                    "average_precision": average_precision, "converged": bool(converged),
                    "convergence_warning": message, "valid_for_tuning": bool(converged),
                }
            )
    usable = [
        (float(np.mean(values)), C, l1_ratio)
        for (C, l1_ratio), values in scores_by_candidate.items()
        if not invalid[(C, l1_ratio)] and len(values) == INNER_N_SPLITS
    ]
    if not usable:
        raise RuntimeError(
            f"no Elastic Net candidate converged across all {INNER_N_SPLITS} inner folds; "
            "tuning on the training modality is invalid"
        )
    best_mean, best_C, best_l1_ratio = min(usable, key=lambda item: (-item[0], item[1], item[2]))
    _, x_train, x_eval = _scale(context)
    weights = _weights(context.train_participants, context.y_train)
    model = make_elasticnet_model(best_C, best_l1_ratio)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, context.y_train, sample_weight=weights)
    converged, message = _elasticnet_convergence_status(model, caught)
    if not converged:
        raise RuntimeError(
            f"the final Elastic Net fit did not converge (C={best_C}, "
            f"l1_ratio={best_l1_ratio}, {message})"
        )
    scores = model.predict_proba(x_eval)[:, int(np.flatnonzero(model.classes_ == 1)[0])]
    nonzero = np.flatnonzero(model.coef_[0])
    return Fitted(
        scores,
        {
            "model": "LogisticRegression(penalty=elasticnet, solver=saga)",
            "scaled": True,
            "feature_selection": False,
            "hyperparameter_grid": {
                "C": list(ELASTICNET_C_GRID), "l1_ratio": list(ELASTICNET_L1_RATIO_GRID)
            },
            "inner_cv": {"class": "StratifiedGroupKFold", "n_splits": INNER_N_SPLITS},
            "tuning_metric": "average_precision",
            "tie_breaking": "highest mean inner Average Precision; then lowest C; then lowest l1_ratio",
            "hyperparameters": {"C": float(best_C), "l1_ratio": float(best_l1_ratio)},
            "inner_mean_average_precision": float(best_mean),
            "inner_cv_results": inner_rows,
            "n_selected_features": len(context.features),
            "n_nonzero_coefficients": int(nonzero.size),
            "selected_features": [context.features[index] for index in nonzero],
            "converged": bool(converged),
            "convergence_warning": message,
            "intercept": float(model.intercept_[0]),
        },
    )


def fit_random_forest_nested(context: FitContext) -> Fitted:
    """EXP6: grouped inner CV tunes the published forest grid; no scaling is applied."""
    depths, leaves, features_grid = validate_grids(
        RF_MAX_DEPTH_GRID, RF_MIN_SAMPLES_LEAF_GRID, RF_MAX_FEATURES_GRID, len(context.features)
    )
    grid = candidate_grid(depths, leaves, features_grid)
    splits = _inner_splits(context.x_train, context.y_train, context.train_participants)
    scores_by_candidate: list[list[float]] = [[] for _ in grid]
    inner_rows: list[dict[str, Any]] = []
    for inner_fold, (train_local, validation_local) in enumerate(splits, start=1):
        x_inner, x_validation, weights = _inner_prepared(context, train_local, validation_local, False)
        y_inner = context.y_train[train_local]
        y_validation = context.y_train[validation_local]
        for position, candidate in enumerate(grid):
            model = make_random_forest(
                candidate["max_depth"], candidate["min_samples_leaf"], candidate["max_features"],
                n_estimators=RF_N_ESTIMATORS, n_jobs=context.n_jobs,
            )
            model.fit(x_inner, y_inner, sample_weight=weights)
            probability = model.predict_proba(x_validation)[
                :, int(np.flatnonzero(model.classes_ == 1)[0])
            ]
            average_precision = float(average_precision_score(y_validation, probability))
            scores_by_candidate[position].append(average_precision)
            inner_rows.append(
                {
                    "inner_fold": inner_fold,
                    "max_depth": _label(candidate["max_depth"]),
                    "min_samples_leaf": int(candidate["min_samples_leaf"]),
                    "max_features": _label(candidate["max_features"]),
                    "average_precision": average_precision,
                }
            )
    # Frozen tie rule of the published nested Random Forest experiment.
    ranked = sorted(
        range(len(grid)),
        key=lambda position: (
            -float(np.mean(scores_by_candidate[position])),
            _depth_order(grid[position]["max_depth"]),
            -int(grid[position]["min_samples_leaf"]),
            max_features_fraction(grid[position]["max_features"], len(context.features)),
            position,
        ),
    )
    best_position = ranked[0]
    best = grid[best_position]
    weights = _weights(context.train_participants, context.y_train)
    model = make_random_forest(
        best["max_depth"], best["min_samples_leaf"], best["max_features"],
        n_estimators=RF_N_ESTIMATORS, n_jobs=context.n_jobs,
    )
    model.fit(context.x_train, context.y_train, sample_weight=weights)
    scores = model.predict_proba(context.x_eval)[
        :, int(np.flatnonzero(model.classes_ == 1)[0])
    ]
    return Fitted(
        scores,
        {
            "model": f"RandomForestClassifier(n_estimators={RF_N_ESTIMATORS})",
            "scaled": False,
            "feature_selection": False,
            "hyperparameter_grid": {
                "max_depth": [_label(value) for value in depths],
                "min_samples_leaf": list(leaves),
                "max_features": [_label(value) for value in features_grid],
            },
            "inner_cv": {"class": "StratifiedGroupKFold", "n_splits": INNER_N_SPLITS},
            "tuning_metric": "average_precision",
            "tie_breaking": (
                "highest mean inner Average Precision; then smallest max_depth (None last); "
                "then largest min_samples_leaf; then most restrictive max_features; then grid order"
            ),
            "hyperparameters": {
                "max_depth": _label(best["max_depth"]),
                "min_samples_leaf": int(best["min_samples_leaf"]),
                "max_features": _label(best["max_features"]),
            },
            "inner_mean_average_precision": float(np.mean(scores_by_candidate[best_position])),
            "inner_cv_results": inner_rows,
            "n_selected_features": len(context.features),
            "selected_features": list(context.features),
            "converged": True,
            "convergence_warning": "",
        },
    )


FITTERS: dict[str, Callable[[FitContext], Fitted]] = {
    "l2_all": fit_l2_all,
    "mrmr_fixed": fit_mrmr_fixed,
    "mrmr_nested": fit_mrmr_nested,
    "elasticnet_nested": fit_elasticnet_nested,
}


# ---------------------------------------------------------------------------
# Threshold selection inside the training modality
# ---------------------------------------------------------------------------
def train_out_of_fold_scores(
    train: pd.DataFrame, features: Sequence[str], fitter: Callable[[FitContext], Fitted], n_jobs: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grouped out-of-fold probabilities computed *within* the training modality.

    These probabilities are the only input to the threshold search, and the same
    fitter -- including its own inner tuning -- is re-run inside every fold, so
    no held-out training participant ever informs its own probability.
    """
    x = train[list(features)].to_numpy(dtype=float)
    y = train["target"].to_numpy(dtype=int)
    groups = train["participant_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    try:
        splits = list(splitter.split(x, y, groups))
    except ValueError as error:
        raise ValueError(
            f"a valid {N_SPLITS}-fold grouped CV cannot be built inside the training modality; "
            "the classification threshold cannot be selected without leaking"
        ) from error
    pieces, audit = [], []
    for fold, (train_index, test_index) in enumerate(splits, start=1):
        train_ids = frozenset(map(str, groups[train_index]))
        test_ids = frozenset(map(str, groups[test_index]))
        if train_ids & test_ids:
            raise AssertionError(f"threshold fold {fold}: participant overlap")
        if set(y[train_index]) != {0, 1} or set(y[test_index]) != {0, 1}:
            raise ValueError(
                f"threshold fold {fold} lacks both classes; the threshold cannot be selected "
                "from the training modality without changing the protocol"
            )
        fitted = fitter(
            FitContext(
                x_train=x[train_index], y_train=y[train_index],
                train_participants=groups[train_index], x_eval=x[test_index],
                train_ids=train_ids, eval_ids=test_ids, features=list(features), n_jobs=n_jobs,
            )
        )
        piece = train.iloc[test_index][FINAL_KEY + ["dataset", MODALITY_COLUMN, "target"]].copy()
        piece = piece.rename(columns={"target": "true_label"})
        piece["pathological_score"] = fitted.scores
        piece["fold_id"] = fold
        pieces.append(piece)
        audit.append(
            {
                "fold_id": fold,
                "n_train_participants": len(train_ids),
                "n_test_participants": len(test_ids),
                "n_train_channels": int(len(train_index)),
                "n_test_channels": int(len(test_index)),
                "train_pathological_channels": int((y[train_index] == 1).sum()),
                "test_pathological_channels": int((y[test_index] == 1).sum()),
                "participant_overlap_count": 0,
                "hyperparameters": json.dumps(fitted.record.get("hyperparameters", {}), sort_keys=True),
            }
        )
    oof = pd.concat(pieces, ignore_index=True)
    if len(oof) != len(train) or oof.duplicated(FINAL_KEY).any():
        raise AssertionError("every training observation must be scored out of fold exactly once")
    return oof.sort_values(FINAL_KEY, kind="stable").reset_index(drop=True), pd.DataFrame(audit)


# ---------------------------------------------------------------------------
# Evaluation on the held-out modality
# ---------------------------------------------------------------------------
def _metric_rows(
    predictions: pd.DataFrame, label_column: str, threshold_type: str,
    threshold_value: float, scope: str, scope_value: str,
) -> list[dict[str, Any]]:
    metrics = metrics_from_labels(
        predictions["true_label"], predictions["pathological_score"], predictions[label_column]
    )
    return [
        {
            "scope": scope, "scope_value": scope_value, "threshold_type": threshold_type,
            "threshold_value": float(threshold_value), "metric": metric, "value": value,
            "n_channels": int(len(predictions)),
            "n_participants": int(predictions["participant_id"].nunique()),
        }
        for metric, value in metrics.items()
    ]


def test_metric_table(
    predictions: pd.DataFrame, label_column: str, threshold_type: str, threshold_value: float
) -> pd.DataFrame:
    """Overall and per-site metrics on the held-out modality for one decision rule."""
    rows = _metric_rows(
        predictions, label_column, threshold_type, threshold_value, "overall", "all"
    )
    for site, group in predictions.groupby("dataset", sort=True):
        rows.extend(
            _metric_rows(group, label_column, threshold_type, threshold_value, "site", str(site))
        )
    return pd.DataFrame(rows)


def participant_cluster_bootstrap(
    predictions: pd.DataFrame,
    label_column: str,
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = BOOTSTRAP_RANDOM_STATE,
) -> pd.DataFrame:
    """Participant-cluster bootstrap of already frozen labels on the test modality only.

    Nothing is refit and no threshold is re-optimized: this resamples the
    independent unit (the participant) and recomputes the published metrics.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    participants = np.asarray(sorted(predictions["participant_id"].astype(str).unique()))
    if participants.size < 2:
        raise ValueError("a participant bootstrap needs at least two test participants")
    generator = np.random.default_rng(random_state)
    rows = []
    for replicate in range(n_bootstrap):
        sampled = generator.choice(participants, size=len(participants), replace=True)
        sample = resample_participant_clusters(predictions, sampled)
        metrics = metrics_from_labels(
            sample["true_label"], sample["pathological_score"], sample[label_column]
        )
        rows.extend(
            {
                "bootstrap_replicate": replicate, "metric": metric,
                "value": metrics[metric], "defined": bool(math.isfinite(metrics[metric])),
            }
            for metric in METRIC_NAMES
        )
    return pd.DataFrame(rows)


def confidence_intervals(distribution: pd.DataFrame) -> pd.DataFrame:
    alpha = 1.0 - CONFIDENCE_LEVEL
    rows = []
    for metric, group in distribution.groupby("metric", sort=False):
        valid = pd.to_numeric(group.loc[group["defined"], "value"], errors="coerce").dropna()
        count = len(valid)
        rows.append(
            {
                "metric": metric, "confidence_level": CONFIDENCE_LEVEL,
                "lower_percentile": float(valid.quantile(alpha / 2.0)) if count else math.nan,
                "upper_percentile": float(valid.quantile(1.0 - alpha / 2.0)) if count else math.nan,
                "n_valid_replicates": count, "n_total_replicates": len(group),
                "proportion_valid": count / len(group),
            }
        )
    return pd.DataFrame(rows)


PREDICTION_COLUMNS = [
    "experiment", "experiment_label", "train_modality", "test_modality",
    "participant_id", "channel_id", "site", "channel_type",
    "y_true", "pathological_score", "predicted_label", "threshold_type", "threshold_value",
]


def build_prediction_table(
    predictions: pd.DataFrame, experiment: CrossModalityExperiment,
    train_modality: str, test_modality: str, optimized_threshold: float,
) -> pd.DataFrame:
    """Channel-level test predictions, one row per (channel x decision rule)."""
    base = predictions.rename(
        columns={"dataset": "site", "true_label": "y_true"}
    )
    pieces = []
    for threshold_type, threshold_value, label_column in (
        (FIXED_THRESHOLD_TYPE, float(DECISION_THRESHOLD), "predicted_label"),
        (OPTIMIZED_THRESHOLD_TYPE, float(optimized_threshold),
         "predicted_label_optimized_threshold"),
    ):
        piece = base.copy()
        piece["experiment"] = experiment.short
        piece["experiment_label"] = experiment.label
        piece["train_modality"] = train_modality
        piece["test_modality"] = test_modality
        piece["threshold_type"] = threshold_type
        piece["threshold_value"] = threshold_value
        piece["predicted_label"] = piece[label_column].astype(int)
        pieces.append(piece[PREDICTION_COLUMNS])
    return pd.concat(pieces, ignore_index=True)


def run_direction(
    dataset: pd.DataFrame,
    features: Sequence[str],
    experiment: CrossModalityExperiment,
    train_modality: str,
    test_modality: str,
    minimum_tpr: float,
    n_bootstrap: int,
    n_jobs: int,
) -> dict[str, Any]:
    """Fit on one modality, freeze the threshold there, evaluate once on the other."""
    features = list(features)
    train = dataset[dataset[MODALITY_COLUMN].eq(train_modality)].reset_index(drop=True)
    test = dataset[dataset[MODALITY_COLUMN].eq(test_modality)].reset_index(drop=True)
    split_audit = assert_disjoint(train, test)
    fitter = FITTERS[experiment.fitter]

    # 1. Threshold: grouped out-of-fold probabilities inside the training modality.
    train_oof, threshold_fold_audit = train_out_of_fold_scores(train, features, fitter, n_jobs)
    trace, chosen = select_threshold(
        train_oof["true_label"], train_oof["pathological_score"], minimum_tpr
    )
    optimized_threshold = float(chosen["threshold"])

    # 2. Final model: refit on every training participant, applied once to the test modality.
    x_train = train[features].to_numpy(dtype=float)
    x_test = test[features].to_numpy(dtype=float)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("predictors contain NaN or infinity; no imputation is permitted")
    train_ids = frozenset(train["participant_id"].astype(str))
    test_ids = frozenset(test["participant_id"].astype(str))
    fitted = fitter(
        FitContext(
            x_train=x_train, y_train=train["target"].to_numpy(dtype=int),
            train_participants=train["participant_id"].astype(str).to_numpy(),
            x_eval=x_test, train_ids=train_ids, eval_ids=test_ids,
            features=features, n_jobs=n_jobs,
        )
    )

    predictions = test[FINAL_KEY + ["dataset", MODALITY_COLUMN, "target"]].copy()
    predictions = predictions.rename(columns={"target": "true_label"})
    predictions["pathological_score"] = fitted.scores
    predictions["predicted_label"] = (fitted.scores >= DECISION_THRESHOLD).astype(int)
    predictions["optimized_threshold"] = optimized_threshold
    predictions["predicted_label_optimized_threshold"] = (
        fitted.scores >= optimized_threshold
    ).astype(int)

    metrics = pd.concat(
        [
            test_metric_table(
                predictions, "predicted_label", FIXED_THRESHOLD_TYPE, DECISION_THRESHOLD
            ),
            test_metric_table(
                predictions, "predicted_label_optimized_threshold",
                OPTIMIZED_THRESHOLD_TYPE, optimized_threshold,
            ),
        ],
        ignore_index=True,
    )
    bootstrap_parts, ci_parts = [], []
    for threshold_type, label_column in (
        (FIXED_THRESHOLD_TYPE, "predicted_label"),
        (OPTIMIZED_THRESHOLD_TYPE, "predicted_label_optimized_threshold"),
    ):
        distribution = participant_cluster_bootstrap(
            predictions, label_column, n_bootstrap=n_bootstrap
        )
        distribution.insert(0, "threshold_type", threshold_type)
        interval = confidence_intervals(distribution)
        interval.insert(0, "threshold_type", threshold_type)
        bootstrap_parts.append(distribution)
        ci_parts.append(interval)

    record = {
        "experiment": experiment.short,
        "experiment_key": experiment.key,
        "experiment_label": experiment.label,
        "train_modality": train_modality,
        "test_modality": test_modality,
        "n_predictors": len(features),
        "predictor_order": features,
        "split_audit": split_audit,
        "threshold": {
            "selection_rule": THRESHOLD_SELECTION_RULE,
            "minimum_tpr": float(minimum_tpr),
            "selected_on": "training modality out-of-fold probabilities only",
            "cv_for_threshold": {
                "class": "StratifiedGroupKFold", "n_splits": N_SPLITS, "shuffle": True,
                "random_state": CV_RANDOM_STATE, "groups": "participant_id",
                "scope": "training modality only",
            },
            **{key: value for key, value in chosen.items()},
        },
        "final_fit": fitted.record,
        "test_evaluated_once": True,
    }
    return {
        "record": record,
        "predictions": predictions,
        "prediction_table": build_prediction_table(
            predictions, experiment, train_modality, test_modality, optimized_threshold
        ),
        "metrics": metrics,
        "bootstrap_distribution": pd.concat(bootstrap_parts, ignore_index=True),
        "bootstrap_confidence_intervals": pd.concat(ci_parts, ignore_index=True),
        "train_oof_predictions": train_oof,
        "threshold_fold_audit": threshold_fold_audit,
        "threshold_trace": trace,
    }


def crossmodality_task_output_root(
    task: str, analysis_root: Path = DEFAULT_ANALYSIS_ROOT
) -> Path:
    """Own output root, so no published experiment tree can ever be touched."""
    return Path(analysis_root) / EXPERIMENT_ID / resolve_task(task)


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


# ---------------------------------------------------------------------------
# Publishing and deliberately small, test-only visualizations
# ---------------------------------------------------------------------------
def _load_schema(path: Path, expected: Sequence[str] | None = None) -> list[str]:
    """Load an existing dataset-produced schema; never infer columns from data."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictors = payload.get("predictors")
    if not isinstance(predictors, list) or not all(isinstance(item, str) for item in predictors):
        raise ValueError(f"feature list lacks a valid predictors list: {path}")
    if expected is not None and list(predictors) != list(expected):
        raise ValueError(f"feature list does not match its frozen expected schema: {path}")
    return list(predictors)


def _validate_input_dataset(dataset: pd.DataFrame, features: Sequence[str], task: str) -> None:
    required = set(FINAL_KEY + ["dataset", "task", "target", MODALITY_COLUMN] + list(features))
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"dataset lacks required cross-modality columns: {missing}")
    observed_tasks = sorted(set(dataset["task"].astype(str).str.strip().str.lower()))
    if observed_tasks != [task]:
        raise ValueError(f"dataset must contain exactly task={task!r}; found {observed_tasks}")
    if dataset.duplicated(FINAL_KEY).any():
        raise ValueError("dataset has duplicate participant-by-channel rows")
    if set(dataset["target"].astype(int).unique()) != {0, 1}:
        raise ValueError("dataset must contain both pathological and non-pathological channels")
    _validate_predictor_schema(dataset, features)


def _save_figure(figure: Any, stem: Path) -> list[Path]:
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    return [stem.with_suffix(".png"), stem.with_suffix(".pdf")]


def make_plots(predictions: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write the requested compact plots from frozen held-out predictions only."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    directions = list(dict.fromkeys(zip(predictions["train_modality"], predictions["test_modality"])))
    experiments = list(dict.fromkeys(predictions["experiment"].astype(str)))
    colors = {"ECOG": "#0072B2", "SEEG": "#D55E00"}

    # Four metrics, with the selected decision rule visibly distinguished from 0.5.
    selected = metrics[metrics["scope"].eq("overall")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, metric, title in zip(
        axes.ravel(), ("auroc", "average_precision", "precision", "sensitivity"),
        ("AUROC", "Average Precision", "PPV", "TPR / sensitivity"),
    ):
        subset = selected[selected["metric"].eq(metric)]
        labels = [f"{exp}\n{train}→{test}" for exp, train, test in zip(
            subset["experiment"], subset["train_modality"], subset["test_modality"]
        )]
        values = subset["value"].to_numpy(dtype=float)
        axis.bar(np.arange(len(subset)), values, color=[colors.get(value, "#666666") for value in subset["train_modality"]])
        axis.set_xticks(np.arange(len(subset)), labels, rotation=45, ha="right", fontsize=8)
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Held-out cross-modality performance (threshold rows shown separately)")
    paths += _save_figure(fig, output_dir / "performance_comparison")
    plt.close(fig)

    # Curves have one panel per experiment; direction is the only visual comparison.
    for kind in ("roc", "precision_recall"):
        fig, axes = plt.subplots(1, len(experiments), figsize=(4.1 * len(experiments), 3.8), squeeze=False,
                                 constrained_layout=True)
        for axis, experiment in zip(axes.ravel(), experiments):
            for train_modality, test_modality in directions:
                rows = predictions[
                    predictions["experiment"].eq(experiment)
                    & predictions["train_modality"].eq(train_modality)
                    & predictions["test_modality"].eq(test_modality)
                    & predictions["threshold_type"].eq(FIXED_THRESHOLD_TYPE)
                ]
                if kind == "roc":
                    x, y, _ = roc_curve(rows["y_true"], rows["pathological_score"])
                    axis.plot(x, y, label=f"{train_modality}→{test_modality}", color=colors[train_modality])
                    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.05))
                else:
                    precision, recall, _ = precision_recall_curve(rows["y_true"], rows["pathological_score"])
                    axis.plot(recall, precision, label=f"{train_modality}→{test_modality}", color=colors[train_modality])
                    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.05))
            axis.set_title(experiment)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8, loc="best")
        fig.suptitle("Held-out " + ("ROC" if kind == "roc" else "precision-recall") + " curves")
        paths += _save_figure(fig, output_dir / f"{kind}_curves")
        plt.close(fig)

    fig, axes = plt.subplots(1, len(experiments), figsize=(4.1 * len(experiments), 3.8), squeeze=False,
                             constrained_layout=True)
    for axis, experiment in zip(axes.ravel(), experiments):
        for train_modality, test_modality in directions:
            rows = predictions[
                predictions["experiment"].eq(experiment)
                & predictions["train_modality"].eq(train_modality)
                & predictions["test_modality"].eq(test_modality)
                & predictions["threshold_type"].eq(FIXED_THRESHOLD_TYPE)
            ]
            for label, linestyle in ((1, "-"), (0, "--")):
                values = rows.loc[rows["y_true"].eq(label), "pathological_score"]
                axis.hist(values, bins=20, density=True, histtype="step", linewidth=1.5,
                          linestyle=linestyle, color=colors[train_modality],
                          label=f"{train_modality}→{test_modality}, " + ("pathological" if label else "non-pathological"))
        axis.set(title=experiment, xlabel="Predicted pathological probability", ylabel="Density", xlim=(0, 1))
        axis.legend(fontsize=7, loc="best")
    fig.suptitle("Held-out pathological-score distributions (solid = pathological)")
    paths += _save_figure(fig, output_dir / "pathological_probability_distributions")
    plt.close(fig)
    return paths


def run_crossmodality(
    *, task: str, baseline_dataset_path: Path, baseline_feature_list_path: Path,
    complexity_dataset_path: Path, complexity_feature_list_path: Path, output_dir: Path,
    repository: Path, minimum_tpr: float = DEFAULT_MINIMUM_TPR, n_bootstrap: int = N_BOOTSTRAP,
    n_jobs: int = 1, experiment_keys: Iterable[str] | None = None,
) -> Path:
    """Run the two held-out modality directions and publish a fresh output tree."""
    task = resolve_task(task)
    minimum_tpr = validate_minimum_tpr(minimum_tpr)
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    selected_experiments = resolve_experiments(experiment_keys)
    baseline_features = _load_schema(baseline_feature_list_path, baseline_predictor_names())
    complexity_features = _load_schema(complexity_feature_list_path)
    baseline = pd.read_parquet(baseline_dataset_path)
    complexity = pd.read_parquet(complexity_dataset_path)
    _validate_input_dataset(baseline, baseline_features, task)
    _validate_input_dataset(complexity, complexity_features, task)
    baseline, baseline_excluded = select_modalities(baseline)
    complexity, complexity_excluded = select_modalities(complexity)
    datasets = {"baseline": (baseline, baseline_features, baseline_excluded),
                "complexity": (complexity, complexity_features, complexity_excluded)}
    all_predictions, all_metrics, all_bootstrap, all_ci, index_rows = [], [], [], [], []

    # This reporting and split validation intentionally happen before the first
    # fitter invocation.
    tables = {family: cohort_tables(data, excluded) for family, (data, _, excluded) in datasets.items()}
    audits_by_family: dict[str, list[dict[str, Any]]] = {}
    for family, (data, _, _) in datasets.items():
        direction_audits = []
        for train_modality, test_modality in DIRECTIONS:
            train = data[data[MODALITY_COLUMN].eq(train_modality)]
            test = data[data[MODALITY_COLUMN].eq(test_modality)]
            audit = {"train_modality": train_modality, "test_modality": test_modality,
                     **assert_disjoint(train, test)}
            if audit["n_train_participants"] < MINIMUM_SIDE_PARTICIPANTS:
                raise ValueError(
                    f"{family} {train_modality} has {audit['n_train_participants']} participants; "
                    f"at least {MINIMUM_SIDE_PARTICIPANTS} are required for train-only {N_SPLITS}-fold threshold CV"
                )
            if audit["n_test_participants"] < 2:
                raise ValueError(f"{family} {test_modality} has fewer than two participants for bootstrap")
            direction_audits.append(audit)
        audits_by_family[family] = direction_audits
        print(format_cohort_report(task, tables[family], direction_audits), flush=True)

    with new_output_directory(output_dir) as staging:
        for family, family_tables in tables.items():
            for name, table in family_tables.items():
                atomic_csv(table, staging / "cohort" / f"{family}_{name}.csv")
        for experiment in selected_experiments:
            data, features, _ = datasets[experiment.family]
            for train_modality, test_modality in DIRECTIONS:
                result = run_direction(data, features, experiment, train_modality, test_modality,
                                       minimum_tpr, n_bootstrap, n_jobs)
                relative = Path("experiments") / experiment.key / f"{train_modality.lower()}_to_{test_modality.lower()}"
                destination = staging / relative
                for name in ("prediction_table", "metrics", "bootstrap_distribution",
                             "bootstrap_confidence_intervals", "train_oof_predictions",
                             "threshold_fold_audit", "threshold_trace"):
                    atomic_csv(result[name], destination / f"{name}.csv")
                atomic_json(result["record"], destination / "run_metadata.json")
                leakage = pd.DataFrame([result["record"]["split_audit"]])
                atomic_csv(leakage, destination / "leakage_validation.csv")
                for name, frame in (("prediction_table", result["prediction_table"]), ("metrics", result["metrics"]),
                                    ("bootstrap_distribution", result["bootstrap_distribution"]),
                                    ("bootstrap_confidence_intervals", result["bootstrap_confidence_intervals"])):
                    tagged = frame.copy()
                    tagged.insert(0, "experiment", experiment.short) if "experiment" not in tagged else None
                    tagged.insert(1 if "experiment" in tagged else 0, "train_modality", train_modality) if "train_modality" not in tagged else None
                    tagged.insert(2 if "experiment" in tagged else 1, "test_modality", test_modality) if "test_modality" not in tagged else None
                    {"prediction_table": all_predictions, "metrics": all_metrics,
                     "bootstrap_distribution": all_bootstrap, "bootstrap_confidence_intervals": all_ci}[name].append(tagged)
                index_rows.append({"experiment": experiment.short, "experiment_key": experiment.key,
                                   "train_modality": train_modality, "test_modality": test_modality,
                                   "output": str(relative), "test_evaluated_once": True})
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        combined_bootstrap = pd.concat(all_bootstrap, ignore_index=True)
        combined_ci = pd.concat(all_ci, ignore_index=True)
        atomic_csv(combined_predictions, staging / "test_predictions.csv")
        atomic_csv(combined_metrics, staging / "test_metrics.csv")
        atomic_csv(combined_bootstrap, staging / "test_bootstrap_distribution.csv")
        atomic_csv(combined_ci, staging / "test_bootstrap_confidence_intervals.csv")
        atomic_csv(pd.DataFrame(index_rows), staging / "experiment_index.csv")
        plot_paths = make_plots(combined_predictions, combined_metrics, staging / "plots")
        atomic_json({
            "protocol_version": PROTOCOL_VERSION, "stage": STAGE, "created_utc": timestamp_utc(),
            "task": task, "exploratory_caveat": CONFOUND_NOTE, "minimum_tpr": minimum_tpr,
            "experiments": [item._asdict() for item in selected_experiments],
            "directions": audits_by_family, "input_datasets": {
                "baseline": str(baseline_dataset_path.resolve()), "complexity": str(complexity_dataset_path.resolve()),
            }, "input_sha256": {"baseline_dataset": sha256(baseline_dataset_path),
                                  "complexity_dataset": sha256(complexity_dataset_path)},
            "software_versions": software_versions(), "git_commit": git_commit(repository),
            "plots": [str(path.relative_to(staging)) for path in plot_paths],
            "leakage_contract": "All fitting, tuning and threshold selection use training-modality rows only; target modality is scored once and bootstrapped without refitting.",
        }, staging / "run_metadata.json")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("interictal", "ictal"))
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--baseline-dataset", type=Path)
    parser.add_argument("--baseline-feature-list", type=Path)
    parser.add_argument("--complexity-dataset", type=Path)
    parser.add_argument("--complexity-feature-list", type=Path)
    parser.add_argument("--minimum-tpr", type=float, default=DEFAULT_MINIMUM_TPR)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--experiments", nargs="+", choices=sorted(EXPERIMENTS_BY_KEY))
    args = parser.parse_args()
    root = Path(args.analysis_root).resolve()
    task = resolve_task(args.task)
    baseline_dir = root / "ml_logistic_baseline_v1" / task / "dataset"
    complexity_dir = root / "ml_logistic_complexity_v1" / task / "b" / "dataset"
    output = args.output_dir or crossmodality_task_output_root(task, root)
    result = run_crossmodality(
        task=task,
        baseline_dataset_path=(args.baseline_dataset or baseline_dir / "ml_participant_channel_dataset.parquet").resolve(),
        baseline_feature_list_path=(args.baseline_feature_list or baseline_dir / "feature_list.json").resolve(),
        complexity_dataset_path=(args.complexity_dataset or complexity_dir / "ml_complexity_participant_channel_dataset.parquet").resolve(),
        complexity_feature_list_path=(args.complexity_feature_list or complexity_dir / "complexity_feature_list.json").resolve(),
        output_dir=Path(output).resolve(), repository=Path(__file__).resolve().parents[2],
        minimum_tpr=args.minimum_tpr, n_bootstrap=args.n_bootstrap, n_jobs=args.n_jobs,
        experiment_keys=args.experiments,
    )
    print(result)


if __name__ == "__main__":
    main()
