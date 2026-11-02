"""Plot pooled-OOF ROC curves and fold-wise metric comparisons for the published ML experiments.

Read-only: every number is recomputed from the frozen out-of-fold predictions
already published by the evaluators. Nothing is refitted, resampled, bootstrapped
or cached. Site-stratified figures are pure evaluation slices of those same
predictions: no HUP-only or UCLA-only model is ever trained, and ``overall``
always means every HUP+UCLA out-of-fold observation pooled together (never the
arithmetic mean of the two site estimates).

Two different AUC estimators are produced here and are never mixed:

* bar plots  -- mean of the per-outer-fold AUC across the outer folds, with
  error bars spanning the fold-wise minimum and maximum. These error bars are
  fold variability, *not* confidence intervals.
* ROC panels -- one pooled out-of-fold ROC computed directly from all outer-fold
  test predictions at once, with its pooled-OOF AUC shown in the legend.

Decision rules
--------------
Every experiment is shown at the shared decision threshold 0.5 so the standard
comparison figures stay directly comparable. The tuned nested experiment
``ml_logistic_complexity_c1_nested_tuned_v1`` additionally publishes a
PPV-maximizing threshold selected inside each outer training fold; that decision
rule is drawn only in the separate, explicitly labelled threshold-optimized
comparison, and it reuses the frozen ``predicted_label_optimized_threshold``
column rather than re-optimizing anything here.

AUC is threshold-free: it is always computed from the continuous
``pathological_score`` column, so the tuned experiment has, by construction, the
same AUC at threshold 0.5 and at its optimized threshold. That identity is
asserted at runtime rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

from scripts.ml.common import DEFAULT_ANALYSIS_ROOT, atomic_csv, resolve_task
from scripts.ml.dataset import FINAL_KEY
from scripts.ml.logistic_baseline import DECISION_THRESHOLD, N_SPLITS


FIXED_THRESHOLD_MODE = "fixed_0.5"
OPTIMIZED_THRESHOLD_MODE = "optimized_per_outer_fold"
OPTIMIZED_LABEL_COLUMN = "predicted_label_optimized_threshold"
OPTIMIZED_THRESHOLD_COLUMN = "optimized_threshold"


class Experiment(NamedTuple):
    """One published decision rule and the results directory that holds its OOF table.

    ``label`` is the full, unabbreviated experiment name and is the only name
    written to the CSV metadata columns. ``display`` is the shortened figure
    label; shortening happens in the figures only, never in the tables.
    """

    key: str  # stable file/CSV identifier
    short: str  # figure code, e.g. EXP3A
    label: str  # full name for CSV metadata and captions
    display: str  # shortened figure label (figures only)
    groups: tuple[str, ...]  # figure groups this decision rule appears in
    template: str  # results directory relative to the analysis root
    label_column: str | None = None  # None -> recompute (score >= 0.5)
    threshold_column: str | None = None  # per-fold threshold column, when tuned
    threshold_mode: str = FIXED_THRESHOLD_MODE

    def tick_label(self, width: int = 17) -> str:
        """Two-or-more-line x tick label; wrapped so long names stay on the canvas."""
        lines = [self.short]
        lines.extend(textwrap.wrap(self.display, width=width) or [""])
        lines.append(
            "@ threshold 0.5"
            if self.threshold_mode == FIXED_THRESHOLD_MODE
            else "@ optimized thr."
        )
        return "\n".join(lines)


class Scope(NamedTuple):
    """One evaluation slice of the frozen predictions."""

    key: str
    label: str
    site: str | None  # None == pooled HUP+UCLA


# The tuned experiment appears twice: once under the shared 0.5 rule (so the
# nested group stays directly comparable) and once under its own optimized
# threshold, which is confined to the threshold-optimized figures.
TUNED_TEMPLATE = "ml_logistic_complexity_c1_nested_tuned_v1/{task}/results"

# One entry per published decision rule; ``{task}`` is filled in at resolution time.
EXPERIMENTS = (
    Experiment(
        "exp1_baseline", "EXP1", "EXP1 - Conventional baseline", "Conventional baseline",
        ("non_nested",), "ml_logistic_baseline_v1/{task}/results",
    ),
    Experiment(
        "exp2_complexity_all", "EXP2", "EXP2 - Complexity, no selection",
        "Complexity (no selection)", ("non_nested",),
        "ml_logistic_complexity_v1/{task}/b/results",
    ),
    Experiment(
        "exp3a_mrmr_fixed_n8", "EXP3A", "EXP3A - mRMR fixed N=8", "mRMR fixed N=8",
        ("non_nested",), "ml_logistic_complexity_v1/{task}/c1/results",
    ),
    Experiment(
        "exp3b_mrmr_nested", "EXP3B", "EXP3B - Nested mRMR", "Nested mRMR",
        ("nested", "threshold_optimized"), "ml_logistic_complexity_c1_nested_v1/{task}/results",
    ),
    Experiment(
        "exp4_elasticnet_nested", "EXP4", "EXP4 - Nested Elastic Net", "Nested Elastic Net",
        ("nested", "threshold_optimized"), "ml_logistic_complexity_v1/{task}/c2/results",
    ),
    Experiment(
        "exp5_mrmr_c_tuned", "EXP5",
        "EXP5 - Tuned nested mRMR + tuned C (threshold 0.5)",
        "Tuned nested mRMR + tuned C",
        ("nested", "threshold_optimized"), TUNED_TEMPLATE,
    ),
    Experiment(
        "exp5_mrmr_c_tuned_optimized_threshold", "EXP5-OPT",
        "EXP5 - Tuned nested mRMR + tuned C (optimized threshold)",
        "Tuned nested mRMR + tuned C",
        ("threshold_optimized",), TUNED_TEMPLATE,
        label_column=OPTIMIZED_LABEL_COLUMN,
        threshold_column=OPTIMIZED_THRESHOLD_COLUMN,
        threshold_mode=OPTIMIZED_THRESHOLD_MODE,
    ),
)
# The tuned pair whose AUC must agree exactly; also drives the comparison CSV.
TUNED_FIXED_KEY = "exp5_mrmr_c_tuned"
TUNED_OPTIMIZED_KEY = "exp5_mrmr_c_tuned_optimized_threshold"
GROUPS = (
    ("non_nested", "non-nested experiments (EXP1, EXP2, EXP3A) at threshold 0.5"),
    ("nested", "nested experiments (EXP3B, EXP4, EXP5) at threshold 0.5"),
    (
        "threshold_optimized",
        "threshold-optimized comparison: EXP5 at its optimized threshold "
        "against the same experiments at threshold 0.5",
    ),
)
# Bar/CSV metrics: (csv column, axis label, figure title fragment).
METRICS = (
    ("auc", "AUC", "AUC (mean across outer folds)"),
    ("ppv", "PPV", "PPV / precision (mean across outer folds)"),
    ("tpr", "TPR", "TPR / recall (mean across outer folds)"),
)
THRESHOLD_FREE_METRICS = ("auc",)
SITE_VALUES = ("hup", "ucla")
SCOPES = (
    Scope("overall", "Overall (HUP + UCLA)", None),
    Scope("hup", "HUP", "hup"),
    Scope("ucla", "UCLA", "ucla"),
)
# Same site colours in every figure; the pooled scope is the wide shaded bar.
SCOPE_COLORS = {"overall": "#b8c6dc", "hup": "#3f6899", "ucla": "#d1743c"}
SCOPE_EDGE_COLORS = {"overall": "#7d8ba3", "hup": "#2c496b", "ucla": "#9c5327"}
# One line colour per decision rule, reused across the ROC panels.
EXPERIMENT_COLORS = {
    "EXP1": "#4c72b0",
    "EXP2": "#dd8452",
    "EXP3A": "#55a868",
    "EXP3B": "#c44e52",
    "EXP4": "#8172b3",
    "EXP5": "#937860",
    "EXP5-OPT": "#da8bc3",
}
FOLD_ERROR_BAR_NOTE = (
    "Bars show mean performance across the {n_folds} outer folds; "
    "error bars indicate fold-wise minimum and maximum (not confidence intervals)."
)
POOLED_ROC_NOTE = (
    "Pooled out-of-fold ROC: all outer-fold test predictions are used at once; "
    "the {n_folds} fold ROC curves are never averaged or interpolated. "
    "Legend AUC is pooled-OOF AUC, which differs from the fold-mean AUC in the bar figures."
)
THRESHOLD_FREE_NOTE = (
    "AUC is threshold-free: it is computed from the continuous pathological_score, so "
    "EXP5 has an identical AUC at threshold 0.5 and at its optimized threshold."
)
THRESHOLD_OPTIMIZED_NOTE = (
    "EXP5-OPT applies the PPV-maximizing threshold that EXP5 selected inside each outer "
    "training fold; every other bar uses the shared threshold 0.5."
)
ROC_THRESHOLD_NOTE = (
    "EXP5 and EXP5-OPT share one ROC curve because the threshold does not change the "
    "underlying probabilities."
)
EMPTY_COUNTS = {
    "n_participants": 0,
    "n_channels": 0,
    "n_pathological": 0,
    "n_non_pathological": 0,
    "prevalence": math.nan,
}

# ---------------------------------------------------------------------------
# Figure geometry. Every panel is laid out in inches and the margins are sized
# from the actual number of title, tick, legend and note lines, so nothing is
# ever pushed off the canvas. ``bbox_inches="tight"`` on save is a second net:
# it can only grow the saved area, never crop it.
# ---------------------------------------------------------------------------
FIGURE_DPI = 300
AXES_HEIGHT_IN = 4.6
AXES_WIDTH_PER_EXPERIMENT_IN = 1.58
MIN_BAR_AXES_WIDTH_IN = 6.4
LEFT_MARGIN_IN = 0.88
RIGHT_MARGIN_IN = 0.32
TITLE_LINE_IN = 0.24
SUBTITLE_LINE_IN = 0.21
TITLE_PAD_IN = 0.30
TICK_LINE_IN = 0.175
LEGEND_ROW_IN = 0.26
NOTE_LINE_IN = 0.19
BOTTOM_PAD_IN = 0.16
ROC_PANEL_WIDTH_IN = 4.15
ROC_PANEL_HEIGHT_IN = 4.15
ROC_PANEL_GAP_IN = 0.34
ROC_TITLE_LINES = 3
# Space reserved above each ROC panel for its three-line cohort title, and below
# it for the tick labels plus the axis label.
ROC_TITLE_BLOCK_IN = 0.62
ROC_TICK_LINES = 3
# Captions stay readable on very wide canvases by wrapping well before the edge.
NOTE_MAX_WIDTH_IN = 10.0
TITLE_FONTSIZE = 11.5
SUBTITLE_FONTSIZE = 10.0
NOTE_FONTSIZE = 7.5
TICK_FONTSIZE = 8.0
LEGEND_FONTSIZE = 8.5
BAR_LABEL_FONTSIZE = 7.5
# Bars are drawn against a 0-1 metric range, but the axis reaches a little
# higher so error-bar caps and the numeric labels above short bars stay inside.
Y_AXIS_TOP = 1.06


class ScopeResult(NamedTuple):
    """Everything one decision rule x scope contributes to the figures and CSVs."""

    available: bool
    counts: dict[str, float]
    fold_rows: list[dict[str, object]]
    summary: dict[str, tuple[float, float, float, int]]  # metric -> (mean, min, max, n_defined)
    n_valid_folds: int
    pooled_auc: float
    roc: tuple[np.ndarray, np.ndarray] | None
    fold_thresholds: dict[int, float]
    notes: list[str]


def load_oof_predictions(results_dir: Path) -> pd.DataFrame:
    """Load one frozen OOF table, validating the real site and outer-fold columns."""
    path = results_dir / "oof_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing out-of-fold predictions: {path}")
    frame = pd.read_csv(path)
    required = set(FINAL_KEY) | {"dataset", "true_label", "pathological_score", "fold_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")
    frame["dataset"] = frame["dataset"].astype(str).str.strip().str.lower()
    unexpected = sorted(set(frame["dataset"]) - set(SITE_VALUES))
    if unexpected:
        raise ValueError(f"{path} contains site labels outside {list(SITE_VALUES)}: {unexpected}")
    frame["fold_id"] = frame["fold_id"].astype(int)
    frame["true_label"] = frame["true_label"].astype(int)
    frame["pathological_score"] = frame["pathological_score"].astype(float)
    if frame.duplicated(FINAL_KEY).any():
        raise ValueError(f"{path} predicts some observations more than once")
    return frame.sort_values(FINAL_KEY, kind="stable").reset_index(drop=True)


def check_predicted_labels(predictions: pd.DataFrame, results_dir: Path) -> str:
    """Cross-check the stored labels against ``score >= threshold`` when published."""
    if "predicted_label" not in predictions.columns:
        return f"{results_dir.name}: no predicted_label column to cross-check"
    stored = predictions["predicted_label"].astype(int).to_numpy()
    recomputed = (predictions["pathological_score"].to_numpy(dtype=float) >= DECISION_THRESHOLD).astype(int)
    mismatches = int((stored != recomputed).sum())
    if mismatches:
        raise AssertionError(
            f"{results_dir / 'oof_predictions.csv'}: {mismatches} of {len(stored)} stored "
            f"predicted_label values disagree with (pathological_score >= {DECISION_THRESHOLD}); "
            "the published predictions and labels are inconsistent."
        )
    return f"stored predicted_label matches (score >= {DECISION_THRESHOLD}) for all {len(stored)} rows"


def check_optimized_labels(
    predictions: pd.DataFrame, experiment: Experiment, results_dir: Path
) -> str:
    """Cross-check the frozen optimized-threshold labels against their own threshold.

    Nothing is re-optimized here: the published per-fold threshold is simply
    re-applied to the published probabilities and must reproduce the published
    labels exactly.
    """
    path = results_dir / "oof_predictions.csv"
    missing = sorted(
        {experiment.label_column, experiment.threshold_column} - set(predictions.columns)
    )
    if missing:
        raise KeyError(
            f"{path} does not publish the optimized-threshold columns {missing} that "
            f"{experiment.short} needs; the threshold-optimized comparison cannot be drawn."
        )
    stored = predictions[experiment.label_column].astype(int).to_numpy()
    thresholds = predictions[experiment.threshold_column].astype(float).to_numpy()
    if not np.isfinite(thresholds).all():
        raise ValueError(f"{path} has non-finite values in {experiment.threshold_column}")
    recomputed = (predictions["pathological_score"].to_numpy(dtype=float) >= thresholds).astype(int)
    mismatches = int((stored != recomputed).sum())
    if mismatches:
        raise AssertionError(
            f"{path}: {mismatches} of {len(stored)} stored {experiment.label_column} values "
            f"disagree with (pathological_score >= {experiment.threshold_column})."
        )
    per_fold = {
        int(fold): sorted(set(group[experiment.threshold_column].astype(float)))
        for fold, group in predictions.groupby("fold_id", sort=True)
    }
    ambiguous = {fold: values for fold, values in per_fold.items() if len(values) != 1}
    if ambiguous:
        raise AssertionError(
            f"{path}: {experiment.threshold_column} is not constant within outer folds "
            f"{sorted(ambiguous)}; one threshold per outer fold is required."
        )
    rendered = ", ".join(f"fold {fold}: {values[0]:.4f}" for fold, values in per_fold.items())
    return (
        f"stored {experiment.label_column} matches its own per-fold threshold for all "
        f"{len(stored)} rows ({rendered})"
    )


def assert_shared_evaluation(loaded: dict[str, pd.DataFrame]) -> list[int]:
    """Every experiment must be evaluated on the same channels, labels, sites and outer folds."""
    reference_key, reference = next(iter(loaded.items()))
    reference_observations = reference[FINAL_KEY].astype(str).reset_index(drop=True)
    for key, predictions in loaded.items():
        observations = predictions[FINAL_KEY].astype(str).reset_index(drop=True)
        if not observations.equals(reference_observations):
            raise AssertionError(
                f"experiment {key!r} does not share the out-of-fold observations of "
                f"{reference_key!r}: {len(observations)} vs {len(reference_observations)} rows; "
                "the experiments are not comparable."
            )
        if not (predictions["true_label"].to_numpy() == reference["true_label"].to_numpy()).all():
            raise AssertionError(f"experiment {key!r} disagrees with {reference_key!r} on true labels")
        if not predictions["dataset"].equals(reference["dataset"]):
            raise AssertionError(f"experiment {key!r} disagrees with {reference_key!r} on site labels")
        if not (predictions["fold_id"].to_numpy() == reference["fold_id"].to_numpy()).all():
            raise AssertionError(
                f"experiment {key!r} does not use the same outer-fold assignment as "
                f"{reference_key!r}; fold-wise comparison would be meaningless."
            )
    folds = sorted(int(fold) for fold in reference["fold_id"].unique())
    if len(folds) != N_SPLITS:
        raise AssertionError(
            f"expected {N_SPLITS} outer folds in the frozen predictions, found {folds}"
        )
    return folds


def count_cohort(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return dict(EMPTY_COUNTS)
    labels = frame["true_label"].astype(int)
    n_channels = int(len(frame))
    n_pathological = int(labels.eq(1).sum())
    return {
        "n_participants": int(frame["participant_id"].nunique()),
        "n_channels": n_channels,
        "n_pathological": n_pathological,
        "n_non_pathological": int(labels.eq(0).sum()),
        "prevalence": n_pathological / n_channels,
    }


def metric_values(frame: pd.DataFrame, label_column: str | None = None) -> tuple[dict[str, float], str]:
    """AUC / PPV / TPR for one slice; undefined metrics stay NaN and are reported.

    AUC is always taken from the continuous ``pathological_score`` and therefore
    never depends on ``label_column``. Only PPV and TPR read the decision rule:
    the recomputed ``score >= 0.5`` rule when ``label_column`` is ``None``, and
    the frozen published labels otherwise.
    """
    y_true = frame["true_label"].to_numpy(dtype=int)
    scores = frame["pathological_score"].to_numpy(dtype=float)
    if label_column is None:
        predicted = (scores >= DECISION_THRESHOLD).astype(int)
    else:
        predicted = frame[label_column].to_numpy(dtype=int)
    classes = set(y_true.tolist())
    predicted_positives = int((predicted == 1).sum())
    values = {
        "auc": float(roc_auc_score(y_true, scores)) if classes == {0, 1} else math.nan,
        "ppv": (
            float(precision_score(y_true, predicted, pos_label=1, zero_division=0))
            if predicted_positives > 0
            else math.nan
        ),
        "tpr": (
            float(recall_score(y_true, predicted, pos_label=1, zero_division=0))
            if 1 in classes
            else math.nan
        ),
    }
    problems = []
    if classes != {0, 1}:
        only = "pathological only" if classes == {1} else "non-pathological only"
        problems.append(f"single-class slice ({only}); AUC undefined")
    if predicted_positives == 0:
        problems.append("no predicted positives; PPV undefined")
    if 1 not in classes:
        problems.append("no true positives; TPR undefined")
    return values, "; ".join(problems)


def aggregate_folds(fold_rows: list[dict[str, object]], metric: str) -> tuple[float, float, float, int]:
    values = [
        float(row[metric])
        for row in fold_rows
        if isinstance(row[metric], float) and math.isfinite(float(row[metric]))
    ]
    if not values:
        return (math.nan, math.nan, math.nan, 0)
    return (float(np.mean(values)), float(np.min(values)), float(np.max(values)), len(values))


def fold_decision_threshold(fold_frame: pd.DataFrame, experiment: Experiment) -> float:
    """The threshold actually applied in one outer fold, straight from the frozen table."""
    if experiment.threshold_column is None:
        return float(DECISION_THRESHOLD)
    if fold_frame.empty:
        return math.nan
    values = sorted(set(fold_frame[experiment.threshold_column].astype(float)))
    if len(values) != 1:
        raise AssertionError(
            f"{experiment.short}: {experiment.threshold_column} is not constant within an "
            f"outer fold; found {values}"
        )
    return float(values[0])


def evaluate_scope(
    predictions: pd.DataFrame, experiment: Experiment, scope: Scope, folds: list[int], task: str
) -> ScopeResult:
    """Fold-wise metrics, pooled-OOF ROC and cohort counts for one decision rule x scope."""
    frame = (
        predictions
        if scope.site is None
        else predictions[predictions["dataset"].eq(scope.site)].reset_index(drop=True)
    )
    notes: list[str] = []
    if frame.empty:
        notes.append(
            f"{experiment.short} / {scope.label}: no out-of-fold observations for task "
            f"{task!r}; this scope is reported as unavailable rather than estimated."
        )
        return ScopeResult(
            available=False,
            counts=dict(EMPTY_COUNTS),
            fold_rows=[],
            summary={metric: (math.nan, math.nan, math.nan, 0) for metric, _, _ in METRICS},
            n_valid_folds=0,
            pooled_auc=math.nan,
            roc=None,
            fold_thresholds={},
            notes=notes,
        )

    fold_rows: list[dict[str, object]] = []
    fold_thresholds: dict[int, float] = {}
    n_valid_folds = 0
    for fold in folds:
        fold_frame = frame[frame["fold_id"].eq(fold)]
        counts = count_cohort(fold_frame)
        threshold = fold_decision_threshold(fold_frame, experiment)
        if math.isfinite(threshold):
            fold_thresholds[int(fold)] = threshold
        if fold_frame.empty:
            values = {metric: math.nan for metric, _, _ in METRICS}
            warning = "no out-of-fold observations in this fold"
        else:
            values, warning = metric_values(fold_frame, experiment.label_column)
            if set(fold_frame["true_label"].astype(int).tolist()) == {0, 1}:
                n_valid_folds += 1
        if warning:
            notes.append(f"{experiment.short} / {scope.label} / fold {fold}: {warning}")
        fold_rows.append(
            {
                "experiment": experiment.short,
                "experiment_key": experiment.key,
                "experiment_label": experiment.label,
                "task": task,
                "scope": scope.label,
                "scope_key": scope.key,
                "fold": fold,
                "threshold_mode": experiment.threshold_mode,
                "decision_threshold": threshold,
                **counts,
                **values,
                "fold_note": warning,
            }
        )

    summary = {metric: aggregate_folds(fold_rows, metric) for metric, _, _ in METRICS}
    pooled_values, pooled_warning = metric_values(frame, experiment.label_column)
    pooled_auc = pooled_values["auc"]
    roc = None
    if math.isfinite(pooled_auc):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            frame["true_label"].to_numpy(dtype=int), frame["pathological_score"].to_numpy(dtype=float)
        )
        roc = (false_positive_rate, true_positive_rate)
    elif pooled_warning:
        notes.append(f"{experiment.short} / {scope.label} / pooled OOF: {pooled_warning}")

    return ScopeResult(
        available=True,
        counts=count_cohort(frame),
        fold_rows=fold_rows,
        summary=summary,
        n_valid_folds=n_valid_folds,
        pooled_auc=pooled_auc,
        roc=roc,
        fold_thresholds=fold_thresholds,
        notes=notes,
    )


def assert_threshold_free_auc(results: dict[str, dict[str, ScopeResult]]) -> list[str]:
    """AUC must be identical at 0.5 and at the optimized threshold, scope by scope."""
    if TUNED_FIXED_KEY not in results or TUNED_OPTIMIZED_KEY not in results:
        return []
    checked = []
    for scope in SCOPES:
        fixed = results[TUNED_FIXED_KEY][scope.key]
        optimized = results[TUNED_OPTIMIZED_KEY][scope.key]
        for metric in THRESHOLD_FREE_METRICS:
            for index, name in enumerate(("mean", "min", "max")):
                left, right = fixed.summary[metric][index], optimized.summary[metric][index]
                if math.isnan(left) and math.isnan(right):
                    continue
                if left != right:
                    raise AssertionError(
                        f"{metric.upper()} fold-{name} for {scope.label} differs between the "
                        f"0.5 and optimized-threshold views of the tuned experiment "
                        f"({left!r} vs {right!r}); AUC must be threshold-free."
                    )
        if not (
            math.isnan(fixed.pooled_auc) and math.isnan(optimized.pooled_auc)
        ) and fixed.pooled_auc != optimized.pooled_auc:
            raise AssertionError(
                f"pooled-OOF AUC for {scope.label} differs between the 0.5 and "
                f"optimized-threshold views of the tuned experiment "
                f"({fixed.pooled_auc!r} vs {optimized.pooled_auc!r}); AUC must be threshold-free."
            )
        checked.append(scope.label)
    return [
        "[validation] AUC is identical at threshold 0.5 and at the optimized threshold for "
        f"{', '.join(checked)} (fold mean/min/max and pooled OOF)"
    ]


def cohort_caption(reference: pd.DataFrame) -> str:
    """One concise cohort line; every experiment shares these exact observations."""
    pieces = []
    for scope in SCOPES:
        frame = reference if scope.site is None else reference[reference["dataset"].eq(scope.site)]
        counts = count_cohort(frame)
        if counts["n_channels"] == 0:
            pieces.append(f"{scope.label}: none")
            continue
        pieces.append(
            f"{scope.label}: n={counts['n_participants']} participants, "
            f"{counts['n_channels']} channels ({counts['n_pathological']} pathological)"
        )
    return "  |  ".join(pieces)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


# Conservative advance width of one character, in inches per point of font size.
# Used only to decide where to wrap; matplotlib still does the real typesetting.
CHARACTER_WIDTH_PER_POINT = 1.0 / 118.0


def _wrap_text(texts: list[str], width_in: float, fontsize: float) -> list[str]:
    """Wrap text to the drawable width so no line runs off the canvas.

    The wrap width is font-aware: a title set at ``TITLE_FONTSIZE`` fits far
    fewer characters per inch than a caption set at ``NOTE_FONTSIZE``, and
    wrapping both at the same character count is what pushes long titles over
    the edge of the figure.
    """
    characters = max(24, int((width_in - 0.4) / (fontsize * CHARACTER_WIDTH_PER_POINT)))
    lines: list[str] = []
    for text in texts:
        lines.extend(textwrap.wrap(text, width=characters) or [""])
    return lines


def _legend_columns(labels: list[str], width_in: float) -> int:
    """Fit the legend entries into the figure width, spilling into extra rows if needed."""
    if not labels:
        return 1
    longest = max(len(label) for label in labels)
    entry_in = 0.5 + 0.062 * longest  # marker + text at LEGEND_FONTSIZE
    columns = int((width_in - 0.4) // entry_in)
    return max(1, min(len(labels), columns))


class Layout(NamedTuple):
    """Figure-fraction anchors for the blocks stacked around the axes."""

    figure: plt.Figure
    axis: plt.Axes
    title_top: float  # top of the title block
    legend_top: float  # top of the legend box, below the axes
    note_top: float  # top of the caption block, below the legend


def _title_block_in(title_lines: int, subtitle_lines: int) -> float:
    return title_lines * TITLE_LINE_IN + subtitle_lines * SUBTITLE_LINE_IN + TITLE_PAD_IN


def _stacked_figure(
    axes_width_in: float,
    axes_height_in: float,
    title_block_in: float,
    tick_lines: int,
    legend_rows: int,
    note_lines: int,
) -> Layout:
    """Figure whose margins are sized from the text it actually has to hold.

    Every block -- title, axes, tick labels, legend, captions -- is given its
    own height in inches, so the canvas is always large enough for the text it
    carries no matter how many experiments or caption lines a figure has.
    """
    width_in = LEFT_MARGIN_IN + axes_width_in + RIGHT_MARGIN_IN
    note_block = note_lines * NOTE_LINE_IN + (0.12 if note_lines else 0.0)
    legend_block = legend_rows * LEGEND_ROW_IN + (0.16 if legend_rows else 0.0)
    tick_block = tick_lines * TICK_LINE_IN + 0.14
    height_in = (
        BOTTOM_PAD_IN + note_block + legend_block + tick_block + axes_height_in + title_block_in
    )

    figure = plt.figure(figsize=(width_in, height_in))
    axes_bottom = BOTTOM_PAD_IN + note_block + legend_block + tick_block
    axis = figure.add_axes(
        (
            LEFT_MARGIN_IN / width_in,
            axes_bottom / height_in,
            axes_width_in / width_in,
            axes_height_in / height_in,
        )
    )
    return Layout(
        figure=figure,
        axis=axis,
        title_top=1.0 - (TITLE_PAD_IN * 0.40) / height_in,
        legend_top=(BOTTOM_PAD_IN + note_block + legend_block) / height_in,
        note_top=(BOTTOM_PAD_IN + note_block - 0.02) / height_in,
    )


def _draw_title(
    figure: plt.Figure, title_lines: list[str], subtitle_lines: list[str], title_top: float
) -> None:
    """Title and subtitle as separate texts so each keeps its own font size."""
    height_in = figure.get_figheight()
    figure.text(
        0.5, title_top, "\n".join(title_lines), ha="center", va="top",
        fontsize=TITLE_FONTSIZE, color="#1f1f1f",
    )
    if subtitle_lines:
        figure.text(
            0.5,
            title_top - (len(title_lines) * TITLE_LINE_IN) / height_in,
            "\n".join(subtitle_lines),
            ha="center", va="top", fontsize=SUBTITLE_FONTSIZE, color="#3a3a3a",
        )


def _draw_notes(figure: plt.Figure, lines: list[str], note_top: float) -> None:
    height_in = figure.get_figheight()
    for index, line in enumerate(lines):
        figure.text(
            0.5,
            note_top - index * NOTE_LINE_IN / height_in,
            line,
            ha="center",
            va="top",
            fontsize=NOTE_FONTSIZE,
            color="#333333",
        )


def _bar_arrays(
    results: list[ScopeResult], metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bar heights plus asymmetric mean-to-min / max-to-mean error bars."""
    heights, lower, upper, defined = [], [], [], []
    for result in results:
        mean, minimum, maximum, n_defined = result.summary[metric]
        if n_defined == 0 or not math.isfinite(mean):
            heights.append(0.0)
            lower.append(0.0)
            upper.append(0.0)
            defined.append(False)
            continue
        heights.append(mean)
        lower.append(max(0.0, mean - minimum))
        upper.append(max(0.0, maximum - mean))
        defined.append(True)
    return (
        np.asarray(heights, dtype=float),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        np.asarray(defined, dtype=bool),
    )


def _bar_label(
    axis, x: float, height: float, defined: bool, result: ScopeResult, metric: str,
    n_folds: int, inside_color: str, alignment: str = "center",
) -> None:
    """Print the fold-mean inside the bar, which never collides with the error bars."""
    if not defined:
        axis.text(
            x, 0.015, "n/a", ha=alignment, va="bottom", fontsize=BAR_LABEL_FONTSIZE,
            color="#7a7a7a", zorder=6,
        )
        return
    mean, _, _, n_defined = result.summary[metric]
    text = f"{mean:.3f}"
    if n_defined < n_folds:
        text += f"\n{n_defined}/{n_folds} folds"
    if height >= 0.16:
        axis.text(
            x, height - 0.022, text, ha=alignment, va="top", fontsize=BAR_LABEL_FONTSIZE,
            color=inside_color, zorder=6,
        )
    else:
        axis.text(
            x, height + 0.015, text, ha=alignment, va="bottom", fontsize=BAR_LABEL_FONTSIZE,
            color="#1f1f1f", zorder=6,
        )


def comparison_figure(
    task: str,
    metric: str,
    metric_label: str,
    metric_title: str,
    group_key: str,
    group_label: str,
    experiments: tuple[Experiment, ...],
    results: dict[str, dict[str, ScopeResult]],
    cohort_line: str,
    n_folds: int,
    output_stem: Path,
) -> list[Path]:
    positions = np.arange(len(experiments), dtype=float)
    axes_width_in = max(MIN_BAR_AXES_WIDTH_IN, AXES_WIDTH_PER_EXPERIMENT_IN * len(experiments))
    figure_width_in = LEFT_MARGIN_IN + axes_width_in + RIGHT_MARGIN_IN

    # Legend labels are known before the figure exists, so the legend block can be
    # sized exactly and can never be clipped.
    legend_labels = []
    for scope in SCOPES:
        label = scope.label
        if scope.site is not None and not any(
            results[experiment.key][scope.key].available for experiment in experiments
        ):
            label = f"{scope.label} (no observations in the {task} task)"
        legend_labels.append(label)
    legend_columns = _legend_columns(legend_labels, figure_width_in)
    legend_rows = math.ceil(len(legend_labels) / legend_columns)

    notes = [cohort_line, FOLD_ERROR_BAR_NOTE.format(n_folds=n_folds)]
    if group_key == "threshold_optimized":
        notes.append(
            THRESHOLD_FREE_NOTE if metric in THRESHOLD_FREE_METRICS else THRESHOLD_OPTIMIZED_NOTE
        )
    note_lines = _wrap_text(
        notes, min(figure_width_in, NOTE_MAX_WIDTH_IN), NOTE_FONTSIZE
    )

    tick_labels = [experiment.tick_label() for experiment in experiments]
    tick_lines = max(label.count("\n") + 1 for label in tick_labels)
    title_lines = _wrap_text([metric_title], figure_width_in, TITLE_FONTSIZE)
    # A one-scope renderer is used by Detroit.  It has no site comparison to
    # explain, so retain just the concise cohort name instead of inheriting the
    # HUP/UCLA "<group> - <task> task" subtitle.
    subtitle_lines = _wrap_text(
        [SCOPES[0].label if len(SCOPES) == 1 else f"{group_label} - {task} task"],
        figure_width_in,
        SUBTITLE_FONTSIZE,
    )

    layout = _stacked_figure(
        axes_width_in,
        AXES_HEIGHT_IN,
        _title_block_in(len(title_lines), len(subtitle_lines)),
        tick_lines,
        legend_rows,
        len(note_lines),
    )
    figure, axis = layout.figure, layout.axis

    if len(SCOPES) == 1:
        # Detroit has one cohort, so every bar and its min/max whisker belongs
        # exactly at its experiment tick.  No empty site slots or pooled-bar
        # shoulder offsets are retained from the HUP/UCLA comparison layout.
        scope = SCOPES[0]
        scoped_results = [results[experiment.key][scope.key] for experiment in experiments]
        heights, lower, upper, defined = _bar_arrays(scoped_results, metric)
        bars = axis.bar(
            positions, heights, width=0.60, color=SCOPE_COLORS[scope.key],
            edgecolor=SCOPE_EDGE_COLORS[scope.key], linewidth=0.9, zorder=3,
            label=legend_labels[0],
        )
        for bar, experiment in zip(bars, experiments):
            if experiment.threshold_mode != FIXED_THRESHOLD_MODE:
                bar.set_hatch("//")
                bar.set_linewidth(1.3)
        if defined.any():
            axis.errorbar(
                positions[defined], heights[defined], yerr=[lower[defined], upper[defined]],
                fmt="none", ecolor="#1f1f1f", elinewidth=1.2, capsize=4, capthick=1.2, zorder=5,
            )
        for index in range(len(experiments)):
            _bar_label(
                axis, positions[index], heights[index], bool(defined[index]),
                scoped_results[index], metric, n_folds, "#ffffff",
            )
    else:
        # Wide shaded bar behind: the pooled HUP+UCLA scope.
        overall_results = [results[experiment.key]["overall"] for experiment in experiments]
        heights, lower, upper, defined = _bar_arrays(overall_results, metric)
        axis.bar(
            positions, heights, width=0.90, color=SCOPE_COLORS["overall"], alpha=0.60,
            edgecolor=SCOPE_EDGE_COLORS["overall"], linewidth=0.9, zorder=1, label=legend_labels[0],
        )
        # The pooled mean and its fold range stay legible even where the site bars overlap it:
        # a rule across the full bar width, and the error bar parked on the free right shoulder.
        for index in np.flatnonzero(defined):
            axis.hlines(
                heights[index], positions[index] - 0.45, positions[index] + 0.45,
                color=SCOPE_EDGE_COLORS["overall"], linewidth=1.5, zorder=5,
            )
        if defined.any():
            axis.errorbar(
                positions[defined] + 0.37, heights[defined],
                yerr=[lower[defined], upper[defined]], fmt="none",
                ecolor=SCOPE_EDGE_COLORS["overall"], elinewidth=1.4, capsize=4, capthick=1.4, zorder=5,
            )
        for index in range(len(experiments)):
            _bar_label(
                axis, positions[index] - 0.44, heights[index], bool(defined[index]),
                overall_results[index], metric, n_folds, "#22364f", alignment="left",
            )

        # Two narrower opaque bars in front: the site slices of those same predictions.
        for offset, scope, legend_label in zip(
            (-0.155, 0.155), SCOPES[1:], legend_labels[1:]
        ):
            site_results = [results[experiment.key][scope.key] for experiment in experiments]
            heights, lower, upper, defined = _bar_arrays(site_results, metric)
            axis.bar(
                positions + offset, heights, width=0.27, color=SCOPE_COLORS[scope.key],
                edgecolor=SCOPE_EDGE_COLORS[scope.key], linewidth=0.6, zorder=3, label=legend_label,
            )
            if defined.any():
                axis.errorbar(
                    positions[defined] + offset, heights[defined],
                    yerr=[lower[defined], upper[defined]], fmt="none",
                    ecolor="#1f1f1f", elinewidth=1.1, capsize=4, capthick=1.1, zorder=4,
                )
            for index in range(len(experiments)):
                _bar_label(
                    axis, positions[index] + offset, heights[index], bool(defined[index]),
                    site_results[index], metric, n_folds, "#ffffff",
                )

    axis.set_xticks(list(positions))
    axis.set_xticklabels(tick_labels, fontsize=TICK_FONTSIZE)
    axis.set_xlim(-0.65, len(experiments) - 0.35)
    axis.set_ylim(0.0, Y_AXIS_TOP)
    axis.set_yticks(np.round(np.arange(0.0, 1.01, 0.1), 2))
    axis.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    axis.set_ylabel(f"{metric_label} (mean across {n_folds} outer folds)", fontsize=10)
    axis.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    # Multi-site figures retain their existing threshold band.  The Detroit
    # single-site layout uses the subtle hatch above instead.
    if len(SCOPES) != 1:
        for index, experiment in enumerate(experiments):
            if experiment.threshold_mode != FIXED_THRESHOLD_MODE:
                axis.axvspan(
                    positions[index] - 0.5, positions[index] + 0.5,
                    color="#f3e4ef", zorder=0,
                )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, layout.legend_top),
        bbox_transform=figure.transFigure,
        ncol=legend_columns,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.6,
        columnspacing=1.4,
    )

    _draw_title(figure, title_lines, subtitle_lines, layout.title_top)
    _draw_notes(figure, note_lines, layout.note_top)
    return _save(figure, output_stem)


def roc_figure(
    task: str,
    group_key: str,
    group_label: str,
    experiments: tuple[Experiment, ...],
    results: dict[str, dict[str, ScopeResult]],
    n_folds: int,
    output_stem: Path,
) -> list[Path]:
    """Three pooled-OOF ROC panels (overall, HUP, UCLA) on one canvas.

    Threshold variants are not drawn as separate curves: they share the
    probabilities, therefore the curve, of their 0.5 counterpart. They appear in
    the legend so the shared curve is explicit rather than silently dropped.
    """
    curves = tuple(experiment for experiment in experiments if experiment.label_column is None)
    variants = tuple(experiment for experiment in experiments if experiment.label_column is not None)

    n_panels = len(SCOPES)
    axes_width_in = n_panels * ROC_PANEL_WIDTH_IN + (n_panels - 1) * ROC_PANEL_GAP_IN
    figure_width_in = LEFT_MARGIN_IN + axes_width_in + RIGHT_MARGIN_IN
    notes = [POOLED_ROC_NOTE.format(n_folds=n_folds)]
    if variants:
        notes.append(ROC_THRESHOLD_NOTE)
    note_width_in = min(figure_width_in, NOTE_MAX_WIDTH_IN)
    note_lines = _wrap_text(notes, note_width_in, NOTE_FONTSIZE)
    title_lines = _wrap_text(["Pooled out-of-fold ROC"], figure_width_in, TITLE_FONTSIZE)
    subtitle_lines = _wrap_text(
        [SCOPES[0].label if len(SCOPES) == 1 else f"{group_label} - {task} task"],
        figure_width_in,
        SUBTITLE_FONTSIZE,
    )

    layout = _stacked_figure(
        axes_width_in,
        ROC_PANEL_HEIGHT_IN + ROC_TITLE_BLOCK_IN,
        _title_block_in(len(title_lines), len(subtitle_lines)),
        tick_lines=ROC_TICK_LINES,
        legend_rows=0,
        note_lines=len(note_lines),
    )
    figure = layout.figure
    # ``_stacked_figure`` reserves the block; the ROC panels replace its single
    # axes and give back the strip at the top that their cohort titles need.
    host = layout.axis
    left, bottom, width, height = host.get_position().bounds
    host.remove()
    gap = ROC_PANEL_GAP_IN / figure_width_in
    panel_width = (width - (n_panels - 1) * gap) / n_panels
    panel_height = height - ROC_TITLE_BLOCK_IN / figure.get_figheight()
    axes = [
        figure.add_axes((left + index * (panel_width + gap), bottom, panel_width, panel_height))
        for index in range(n_panels)
    ]

    for index, (axis, scope) in enumerate(zip(axes, SCOPES)):
        reference = results[curves[0].key][scope.key]
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("False positive rate", fontsize=9.5)
        axis.tick_params(labelsize=8.5)
        if index > 0:
            axis.tick_params(labelleft=False)
        if not reference.available:
            axis.text(
                0.5, 0.5, f"No out-of-fold observations\nfor {scope.label}\nin the {task} task",
                ha="center", va="center", fontsize=9.5, color="#7a7a7a", transform=axis.transAxes,
            )
            axis.set_title(scope.label, fontsize=10)
            continue
        for experiment in curves:
            result = results[experiment.key][scope.key]
            if result.roc is None:
                axis.plot(
                    [], [], color=EXPERIMENT_COLORS[experiment.short], linewidth=1.8,
                    label=f"{experiment.short} (pooled AUC undefined)",
                )
                continue
            false_positive_rate, true_positive_rate = result.roc
            axis.plot(
                false_positive_rate, true_positive_rate, color=EXPERIMENT_COLORS[experiment.short],
                linewidth=1.8, label=f"{experiment.short} (pooled OOF AUC = {result.pooled_auc:.3f})",
            )
        for experiment in variants:
            result = results[experiment.key][scope.key]
            auc_text = (
                f"AUC = {result.pooled_auc:.3f}"
                if math.isfinite(result.pooled_auc)
                else "AUC undefined"
            )
            axis.plot(
                [], [], color=EXPERIMENT_COLORS[experiment.short], linewidth=1.8, linestyle=":",
                label=f"{experiment.short} (same curve, {auc_text})",
            )
        axis.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1.0, label="Chance")
        counts = reference.counts
        axis.set_title(
            f"{scope.label}\nn={counts['n_participants']} participants,\n"
            f"{counts['n_channels']} channels ({counts['n_pathological']} pathological)",
            fontsize=9.5,
        )
        axis.grid(color="#eeeeee", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.legend(loc="lower right", frameon=False, fontsize=7.5, handlelength=1.6, borderpad=0.2)
    axes[0].set_ylabel("True positive rate", fontsize=9.5)

    _draw_title(figure, title_lines, subtitle_lines, layout.title_top)
    _draw_notes(figure, note_lines, layout.note_top)
    return _save(figure, output_stem)


def _save(figure, output_stem: Path) -> list[Path]:
    """Write the 300 dpi PNG and the vector PDF, never cropping any artist.

    ``bbox_inches="tight"`` only ever grows the saved area, so a legend, tick
    label or caption that overhangs the computed canvas is still written out in
    full rather than clipped.
    """
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    figure.savefig(png, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.06)
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.06)
    plt.close(figure)
    return [png, pdf]


FOLD_COLUMNS = [
    "experiment", "task", "scope", "fold", "threshold_mode", "decision_threshold",
    "n_participants", "n_channels", "n_pathological", "n_non_pathological", "prevalence",
    "auc", "ppv", "tpr",
    "experiment_key", "experiment_label", "scope_key", "fold_note",
]
SUMMARY_COLUMNS = [
    "experiment", "task", "scope", "threshold_mode",
    "n_participants_total", "n_channels_total",
    "n_pathological_total", "n_non_pathological_total", "prevalence",
    "auc_fold_mean", "auc_fold_min", "auc_fold_max",
    "ppv_fold_mean", "ppv_fold_min", "ppv_fold_max",
    "tpr_fold_mean", "tpr_fold_min", "tpr_fold_max",
    "auc_pooled_oof", "n_valid_folds",
    "n_folds_total", "auc_n_folds_defined", "ppv_n_folds_defined", "tpr_n_folds_defined",
    "experiment_key", "experiment_label", "scope_key",
    "decision_threshold", "optimized_threshold_by_fold", "notes",
]
THRESHOLD_COMPARISON_COLUMNS = [
    "task", "scope", "scope_key",
    "experiment_key_threshold_0p5", "experiment_label_threshold_0p5",
    "experiment_key_optimized", "experiment_label_optimized",
    "n_participants_total", "n_channels_total", "n_pathological_total", "prevalence",
    "auc_fold_mean_threshold_0p5", "auc_fold_mean_optimized",
    "auc_pooled_oof_threshold_0p5", "auc_pooled_oof_optimized", "auc_is_identical",
    "ppv_fold_mean_threshold_0p5", "ppv_fold_min_threshold_0p5", "ppv_fold_max_threshold_0p5",
    "ppv_fold_mean_optimized", "ppv_fold_min_optimized", "ppv_fold_max_optimized",
    "ppv_fold_mean_delta",
    "tpr_fold_mean_threshold_0p5", "tpr_fold_min_threshold_0p5", "tpr_fold_max_threshold_0p5",
    "tpr_fold_mean_optimized", "tpr_fold_min_optimized", "tpr_fold_max_optimized",
    "tpr_fold_mean_delta",
    "n_folds_total", "decision_threshold_0p5", "optimized_threshold_by_fold",
]


def _threshold_by_fold(result: ScopeResult) -> str:
    if not result.fold_thresholds:
        return ""
    return json.dumps({str(fold): result.fold_thresholds[fold] for fold in sorted(result.fold_thresholds)})


def _delta(left: float, right: float) -> float:
    if not (math.isfinite(left) and math.isfinite(right)):
        return math.nan
    return float(right - left)


def threshold_comparison_rows(
    task: str, results: dict[str, dict[str, ScopeResult]], n_folds: int
) -> list[dict[str, object]]:
    """One row per scope contrasting the tuned experiment's two decision rules."""
    if TUNED_FIXED_KEY not in results or TUNED_OPTIMIZED_KEY not in results:
        return []
    fixed_experiment = next(item for item in EXPERIMENTS if item.key == TUNED_FIXED_KEY)
    optimized_experiment = next(item for item in EXPERIMENTS if item.key == TUNED_OPTIMIZED_KEY)
    rows = []
    for scope in SCOPES:
        fixed = results[TUNED_FIXED_KEY][scope.key]
        optimized = results[TUNED_OPTIMIZED_KEY][scope.key]
        row: dict[str, object] = {
            "task": task,
            "scope": scope.label,
            "scope_key": scope.key,
            "experiment_key_threshold_0p5": fixed_experiment.key,
            "experiment_label_threshold_0p5": fixed_experiment.label,
            "experiment_key_optimized": optimized_experiment.key,
            "experiment_label_optimized": optimized_experiment.label,
            "n_participants_total": fixed.counts["n_participants"],
            "n_channels_total": fixed.counts["n_channels"],
            "n_pathological_total": fixed.counts["n_pathological"],
            "prevalence": fixed.counts["prevalence"],
            "auc_fold_mean_threshold_0p5": fixed.summary["auc"][0],
            "auc_fold_mean_optimized": optimized.summary["auc"][0],
            "auc_pooled_oof_threshold_0p5": fixed.pooled_auc,
            "auc_pooled_oof_optimized": optimized.pooled_auc,
            "auc_is_identical": bool(
                (fixed.summary["auc"][0] == optimized.summary["auc"][0])
                or (
                    math.isnan(fixed.summary["auc"][0]) and math.isnan(optimized.summary["auc"][0])
                )
            ),
            "n_folds_total": n_folds,
            "decision_threshold_0p5": DECISION_THRESHOLD,
            "optimized_threshold_by_fold": _threshold_by_fold(optimized),
        }
        for metric in ("ppv", "tpr"):
            for index, name in enumerate(("mean", "min", "max")):
                row[f"{metric}_fold_{name}_threshold_0p5"] = fixed.summary[metric][index]
                row[f"{metric}_fold_{name}_optimized"] = optimized.summary[metric][index]
            row[f"{metric}_fold_mean_delta"] = _delta(
                fixed.summary[metric][0], optimized.summary[metric][0]
            )
        rows.append(row)
    return rows


def plot_experiments(task: str, analysis_root: Path, output_dir: Path) -> list[Path]:
    task = resolve_task(task)
    results_dirs = {
        experiment.key: analysis_root / experiment.template.format(task=task)
        for experiment in EXPERIMENTS
    }
    # Two decision rules can share one frozen table; read it once.
    tables: dict[Path, pd.DataFrame] = {}
    loaded: dict[str, pd.DataFrame] = {}
    for key, path in results_dirs.items():
        if path not in tables:
            tables[path] = load_oof_predictions(path)
        loaded[key] = tables[path]
    folds = assert_shared_evaluation(loaded)
    n_folds = len(folds)

    validation = [
        f"[validation] {len(EXPERIMENTS)} decision rules over {len(tables)} frozen prediction "
        f"tables share {len(next(iter(loaded.values())))} out-of-fold observations, true labels, "
        "site labels and outer-fold assignments",
        f"[validation] outer folds in use: {folds}",
    ]
    for experiment in EXPERIMENTS:
        validation.append(
            f"[validation] {experiment.short}: "
            f"{check_predicted_labels(loaded[experiment.key], results_dirs[experiment.key])}"
        )
        if experiment.label_column is not None:
            validation.append(
                f"[validation] {experiment.short}: "
                f"{check_optimized_labels(loaded[experiment.key], experiment, results_dirs[experiment.key])}"
            )

    results: dict[str, dict[str, ScopeResult]] = {}
    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        results[experiment.key] = {}
        for scope in SCOPES:
            result = evaluate_scope(loaded[experiment.key], experiment, scope, folds, task)
            results[experiment.key][scope.key] = result
            fold_rows.extend(result.fold_rows)
            validation.extend(f"[warning] {note}" for note in result.notes)
            row: dict[str, object] = {
                "experiment": experiment.short,
                "experiment_key": experiment.key,
                "experiment_label": experiment.label,
                "task": task,
                "scope": scope.label,
                "scope_key": scope.key,
                "threshold_mode": experiment.threshold_mode,
                "n_participants_total": result.counts["n_participants"],
                "n_channels_total": result.counts["n_channels"],
                "n_pathological_total": result.counts["n_pathological"],
                "n_non_pathological_total": result.counts["n_non_pathological"],
                "prevalence": result.counts["prevalence"],
                "auc_pooled_oof": result.pooled_auc,
                "n_valid_folds": result.n_valid_folds,
                "n_folds_total": n_folds,
                "decision_threshold": (
                    DECISION_THRESHOLD if experiment.threshold_column is None else math.nan
                ),
                "optimized_threshold_by_fold": (
                    "" if experiment.threshold_column is None else _threshold_by_fold(result)
                ),
                "notes": "; ".join(result.notes),
            }
            for metric, _, _ in METRICS:
                mean, minimum, maximum, n_defined = result.summary[metric]
                row[f"{metric}_fold_mean"] = mean
                row[f"{metric}_fold_min"] = minimum
                row[f"{metric}_fold_max"] = maximum
                row[f"{metric}_n_folds_defined"] = n_defined
            summary_rows.append(row)

    validation.extend(assert_threshold_free_auc(results))

    cohort_line = cohort_caption(next(iter(loaded.values())))
    written: list[Path] = []
    for group_key, group_label in GROUPS:
        group_experiments = tuple(
            experiment for experiment in EXPERIMENTS if group_key in experiment.groups
        )
        for metric, metric_label, metric_title in METRICS:
            written.extend(
                comparison_figure(
                    task, metric, metric_label, metric_title, group_key, group_label,
                    group_experiments, results, cohort_line, n_folds,
                    output_dir / f"{task}_comparison_{group_key}_{metric}",
                )
            )
        written.extend(
            roc_figure(
                task, group_key, group_label, group_experiments, results, n_folds,
                output_dir / f"{task}_roc_pooled_oof_{group_key}",
            )
        )

    fold_table = pd.DataFrame(fold_rows)[FOLD_COLUMNS]
    fold_path = output_dir / f"{task}_experiment_performance_folds.csv"
    atomic_csv(fold_table, fold_path)
    written.append(fold_path)

    summary_table = pd.DataFrame(summary_rows)[SUMMARY_COLUMNS]
    summary_path = output_dir / f"{task}_experiment_performance_summary.csv"
    atomic_csv(summary_table, summary_path)
    written.append(summary_path)

    comparison_rows = threshold_comparison_rows(task, results, n_folds)
    if comparison_rows:
        comparison_table = pd.DataFrame(comparison_rows)[THRESHOLD_COMPARISON_COLUMNS]
        comparison_path = output_dir / f"{task}_threshold_optimized_comparison.csv"
        atomic_csv(comparison_table, comparison_path)
        written.append(comparison_path)

    for line in validation:
        print(line)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("interictal", "ictal"))
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="defaults to <analysis-root>/ml_performance_figures_v1/<task>",
    )
    args = parser.parse_args()
    analysis_root = args.analysis_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else analysis_root / "ml_performance_figures_v1" / resolve_task(args.task)
    )
    for path in plot_experiments(args.task, analysis_root, output_dir):
        print(path)


if __name__ == "__main__":
    main()
