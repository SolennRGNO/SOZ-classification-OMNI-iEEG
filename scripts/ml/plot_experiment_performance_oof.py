"""Pooled out-of-fold (OOF) performance figures for the published ML experiments.

This is the pooled-OOF counterpart of :mod:`scripts.ml.plot_experiment_performance`.
That script summarises each experiment as the *mean / min / max across the five
outer folds*. This one never averages across folds: the five outer-fold **test**
predictions are concatenated into one complete out-of-fold dataset and every
metric is computed **once** on that pooled dataset.

Read-only: nothing is refitted, resampled, bootstrapped, re-thresholded or
cached. Every number comes from the frozen ``oof_predictions.csv`` tables the
evaluators already published, and the outputs are written to their own
directory (``ml_performance_figures_oof_v1``) so the fold-wise figures are never
overwritten.

Estimators used here
--------------------
* ``auc`` -- pooled-OOF AUROC from the continuous ``pathological_score``.
* ``ap``  -- pooled-OOF Average Precision from the same continuous scores.
* ``ppv`` / ``tpr`` -- pooled-OOF precision / recall at the shared decision
  threshold 0.5 for the standard comparisons.

Decision rules
--------------
Every standard figure uses the shared threshold 0.5, EXP5 included. The tuned
nested experiment additionally publishes a PPV-maximizing threshold selected
inside each outer training fold (``EXP5-OPT``); that rule appears only in the
separate, explicitly labelled threshold-optimized comparison and as an extra
operating-point marker on the nested PR figure. No threshold is recomputed or
retuned here -- the frozen ``predicted_label_optimized_threshold`` column is
re-applied as published.

Threshold-independence
----------------------
ROC / AUROC and PR / AP depend only on the continuous scores, so EXP5 and
EXP5-OPT share one ROC curve, one PR curve, one AUROC and one AP. The curves are
therefore drawn once, not duplicated, and the identity is asserted at runtime
rather than assumed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from scripts.ml.common import DEFAULT_ANALYSIS_ROOT, atomic_csv, resolve_task
from scripts.ml.dataset import FINAL_KEY
from scripts.ml.logistic_baseline import DECISION_THRESHOLD, N_SPLITS

# The experiment definitions, cohort scopes, palettes, validation checks and
# layout primitives are reused verbatim from the fold-wise plotting stage so the
# two figure families stay label-, colour- and order-consistent. That module is
# imported, never modified.
from scripts.ml.plot_experiment_performance import (
    EMPTY_COUNTS,
    EXPERIMENT_COLORS,
    EXPERIMENTS,
    FIXED_THRESHOLD_MODE,
    GROUPS,
    LEFT_MARGIN_IN,
    LEGEND_FONTSIZE,
    MIN_BAR_AXES_WIDTH_IN,
    NOTE_FONTSIZE,
    NOTE_MAX_WIDTH_IN,
    RIGHT_MARGIN_IN,
    SCOPE_COLORS,
    SCOPE_EDGE_COLORS,
    SCOPES,
    SUBTITLE_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    TUNED_FIXED_KEY,
    TUNED_OPTIMIZED_KEY,
    AXES_HEIGHT_IN,
    AXES_WIDTH_PER_EXPERIMENT_IN,
    BAR_LABEL_FONTSIZE,
    Experiment,
    ROC_PANEL_GAP_IN,
    ROC_PANEL_HEIGHT_IN,
    ROC_PANEL_WIDTH_IN,
    ROC_TICK_LINES,
    ROC_TITLE_BLOCK_IN,
    Y_AXIS_TOP,
    _draw_notes,
    _draw_title,
    _legend_columns,
    _save,
    _stacked_figure,
    _title_block_in,
    _wrap_text,
    assert_shared_evaluation,
    check_optimized_labels,
    check_predicted_labels,
    cohort_caption,
    count_cohort,
    load_oof_predictions,
    metric_values,
)


# Figure/CSV metrics. ``auc`` and ``ap`` are threshold-free; ``ppv`` and ``tpr``
# are read at the decision rule of the experiment (0.5 everywhere except the
# explicitly labelled threshold-optimized comparison).
OOF_METRICS = (
    ("auc", "AUROC", "Pooled out-of-fold AUROC"),
    ("ppv", "PPV", "Pooled out-of-fold PPV / precision"),
    ("tpr", "TPR", "Pooled out-of-fold TPR / recall"),
)
THRESHOLD_FREE_METRICS = ("auc", "ap")
# Only these two groups get curves and distributions; the threshold-optimized
# group is a bar comparison only, because its curves would be duplicates.
CURVE_GROUPS = ("non_nested", "nested")
GROUP_LABELS = dict(GROUPS)

SUMMARY_COLUMNS = [
    "experiment", "scope", "n", "n_positive", "n_negative", "prevalence",
    "auc", "ap", "ppv", "tpr",
]

POOLED_NOTE = (
    "Pooled out-of-fold: the {n_folds} outer-fold test predictions are concatenated into one "
    "complete OOF dataset and each metric is computed once on it. Nothing is averaged across "
    "folds, so these values differ from the fold-mean figures."
)
FIXED_THRESHOLD_NOTE = "PPV and TPR are computed at the shared decision threshold 0.5."
THRESHOLD_FREE_NOTE = (
    "AUROC and AP are threshold-free: they come from the continuous pathological_score, so EXP5 "
    "has an identical AUROC and AP at threshold 0.5 and at its optimized threshold."
)
THRESHOLD_OPTIMIZED_NOTE = (
    "EXP5-OPT applies the PPV-maximizing threshold that EXP5 selected inside each outer training "
    "fold; every other bar uses the shared threshold 0.5. No threshold is retuned here."
)
SHARED_CURVE_NOTE = (
    "EXP5 and EXP5-OPT share one curve: the threshold changes the operating point, never the "
    "underlying probabilities."
)
PR_AXIS_NOTE = (
    "Precision-Recall is the PPV-versus-TPR trade-off; the dotted horizontal line is the no-skill "
    "baseline (pathological prevalence of that panel). Circles mark the threshold-0.5 operating "
    "point."
)
OPTIMIZED_POINT_NOTE = (
    "Diamonds mark the pooled operating point of EXP5's frozen per-outer-fold optimized threshold "
    "(EXP5-OPT); it need not lie on the EXP5 curve, because it is a mixture of five thresholds."
)
DISTRIBUTION_NOTE = (
    "Pooled out-of-fold predicted pathological probability over all HUP+UCLA channels, normalised "
    "within each true class so the two classes stay comparable despite the class imbalance."
)

PATHOLOGICAL_COLOR = "#c44e52"
NON_PATHOLOGICAL_COLOR = "#4c72b0"
DISTRIBUTION_BINS = np.linspace(0.0, 1.0, 41)
PANEL_WIDTH_IN = ROC_PANEL_WIDTH_IN
PANEL_HEIGHT_IN = ROC_PANEL_HEIGHT_IN
PANEL_GAP_IN = ROC_PANEL_GAP_IN


# ``Scope`` is a private NamedTuple of the fold-wise module; alias it for typing
# without importing a name the other module does not export as public API.
Scope_t = type(SCOPES[0])


class OofResult(NamedTuple):
    """Everything one decision rule x scope contributes, computed once on pooled OOF."""

    available: bool
    counts: dict[str, float]
    auc: float  # pooled-OOF AUROC from continuous scores
    ap: float  # pooled-OOF Average Precision from continuous scores
    ppv: float  # pooled-OOF precision at this experiment's decision rule
    tpr: float  # pooled-OOF recall at this experiment's decision rule
    roc: tuple[np.ndarray, np.ndarray] | None  # (fpr, tpr)
    pr: tuple[np.ndarray, np.ndarray] | None  # (recall, precision)
    notes: list[str]

    def value(self, metric: str) -> float:
        return {"auc": self.auc, "ap": self.ap, "ppv": self.ppv, "tpr": self.tpr}[metric]

    @property
    def operating_point(self) -> tuple[float, float] | None:
        """(TPR, PPV) at this experiment's decision rule, or ``None`` if undefined."""
        if math.isfinite(self.tpr) and math.isfinite(self.ppv):
            return (self.tpr, self.ppv)
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def assert_pooled_oof_partition(loaded: dict[str, pd.DataFrame], folds: list[int]) -> list[str]:
    """Each experiment's OOF table must hold every observation exactly once.

    ``load_oof_predictions`` already rejects duplicated participant x channel
    rows; this re-states the requirement explicitly for the pooled view, where a
    single duplicated observation would silently double-count a channel, and
    checks that the outer-fold test sets really do partition the cohort.
    """
    lines = []
    for key, predictions in loaded.items():
        duplicated = predictions.duplicated(FINAL_KEY)
        if duplicated.any():
            offenders = predictions.loc[duplicated, FINAL_KEY].head(5).to_dict("records")
            raise AssertionError(
                f"experiment {key!r} has {int(duplicated.sum())} duplicate participant x channel "
                f"out-of-fold observations, e.g. {offenders}; pooling them would double-count "
                "channels."
            )
        observed_folds = sorted(int(fold) for fold in predictions["fold_id"].unique())
        if observed_folds != folds:
            raise AssertionError(
                f"experiment {key!r} covers outer folds {observed_folds}, expected {folds}; the "
                "pooled out-of-fold dataset would be incomplete."
            )
        per_fold = predictions.groupby("fold_id").size()
        if int(per_fold.sum()) != len(predictions):
            raise AssertionError(
                f"experiment {key!r}: the outer-fold test sets do not partition its "
                f"{len(predictions)} out-of-fold observations."
            )
        lines.append(
            f"[validation] {key}: {len(predictions)} out-of-fold observations, each exactly once, "
            f"partitioned over outer folds {observed_folds} "
            f"({', '.join(f'{fold}:{int(count)}' for fold, count in per_fold.items())})"
        )
    return lines


def assert_fixed_threshold_rule(experiments: tuple[Experiment, ...]) -> list[str]:
    """Standard experiments must read the recomputed ``score >= 0.5`` rule."""
    for experiment in experiments:
        if experiment.threshold_mode != FIXED_THRESHOLD_MODE:
            continue
        if experiment.label_column is not None or experiment.threshold_column is not None:
            raise AssertionError(
                f"{experiment.short} claims the shared threshold {DECISION_THRESHOLD} but reads a "
                f"stored decision column ({experiment.label_column!r}); the standard comparison "
                "must be computed at threshold 0.5."
            )
    return [
        "[validation] PPV and TPR for the standard comparisons are computed from the pooled OOF "
        f"predictions at threshold {DECISION_THRESHOLD} (pathological_score >= "
        f"{DECISION_THRESHOLD}), never from a stored optimized label"
    ]


def assert_threshold_free_metrics(results: dict[str, dict[str, OofResult]]) -> list[str]:
    """AUROC and AP must be bit-identical between the two views of the tuned experiment."""
    if TUNED_FIXED_KEY not in results or TUNED_OPTIMIZED_KEY not in results:
        return []
    checked = []
    for scope in SCOPES:
        fixed = results[TUNED_FIXED_KEY][scope.key]
        optimized = results[TUNED_OPTIMIZED_KEY][scope.key]
        for metric in THRESHOLD_FREE_METRICS:
            left, right = fixed.value(metric), optimized.value(metric)
            if math.isnan(left) and math.isnan(right):
                continue
            if left != right:
                raise AssertionError(
                    f"pooled-OOF {metric.upper()} for {scope.label} differs between the 0.5 and "
                    f"optimized-threshold views of the tuned experiment ({left!r} vs {right!r}); "
                    "it must be threshold-free."
                )
        checked.append(scope.label)
    return [
        "[validation] pooled-OOF AUROC and AP are identical at threshold 0.5 and at the optimized "
        f"threshold for {', '.join(checked)}"
    ]


# ---------------------------------------------------------------------------
# Pooled evaluation
# ---------------------------------------------------------------------------


def evaluate_oof_scope(
    predictions: pd.DataFrame, experiment: Experiment, scope: Scope_t, task: str
) -> OofResult:
    """Compute every metric once on the complete pooled OOF slice of one scope."""
    frame = (
        predictions
        if scope.site is None
        else predictions[predictions["dataset"].eq(scope.site)].reset_index(drop=True)
    )
    notes: list[str] = []
    if frame.empty:
        notes.append(
            f"{experiment.short} / {scope.label}: no out-of-fold observations for task {task!r}; "
            "this scope is reported as unavailable rather than estimated."
        )
        return OofResult(
            available=False,
            counts=dict(EMPTY_COUNTS),
            auc=math.nan,
            ap=math.nan,
            ppv=math.nan,
            tpr=math.nan,
            roc=None,
            pr=None,
            notes=notes,
        )

    y_true = frame["true_label"].to_numpy(dtype=int)
    scores = frame["pathological_score"].to_numpy(dtype=float)
    classes = set(y_true.tolist())
    if classes != {0, 1}:
        only = "pathological only" if classes == {1} else "non-pathological only"
        raise AssertionError(
            f"{experiment.short} / {scope.label}: the pooled out-of-fold slice is single-class "
            f"({only}, n={len(y_true)}); ROC/AUROC and PR/AP are undefined and no curve may be "
            "drawn for it."
        )

    # AUROC/PPV/TPR come from the shared fold-wise helper so the two figure
    # families can never disagree on how a metric is defined; only the slice
    # differs (here: the complete pooled OOF dataset instead of one fold).
    values, warning = metric_values(frame, experiment.label_column)
    if warning:
        notes.append(f"{experiment.short} / {scope.label} / pooled OOF: {warning}")

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return OofResult(
        available=True,
        counts=count_cohort(frame),
        auc=values["auc"],
        ap=float(average_precision_score(y_true, scores)),
        ppv=values["ppv"],
        tpr=values["tpr"],
        roc=(false_positive_rate, true_positive_rate),
        pr=(recall, precision),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _panel_axes(
    figure: plt.Figure, host: plt.Axes, n_panels: int, figure_width_in: float
) -> list[plt.Axes]:
    """Replace the single stacked-layout axes with ``n_panels`` side-by-side panels.

    ``_stacked_figure`` reserves the block including the strip that the per-panel
    cohort titles need; the panels give that strip back at the top.
    """
    left, bottom, width, height = host.get_position().bounds
    host.remove()
    gap = PANEL_GAP_IN / figure_width_in
    panel_width = (width - (n_panels - 1) * gap) / n_panels
    panel_height = height - ROC_TITLE_BLOCK_IN / figure.get_figheight()
    return [
        figure.add_axes((left + index * (panel_width + gap), bottom, panel_width, panel_height))
        for index in range(n_panels)
    ]


def _panel_title(scope_label: str, counts: dict[str, float]) -> str:
    return (
        f"{scope_label}\nn={counts['n_participants']} participants,\n"
        f"{counts['n_channels']} channels ({counts['n_pathological']} pathological)"
    )


def _bar_value_label(axis, x: float, value: float, inside_color: str) -> None:
    """Print the pooled value on the bar; short bars get the label above instead."""
    if not math.isfinite(value):
        axis.text(
            x, 0.015, "n/a", ha="center", va="bottom", fontsize=BAR_LABEL_FONTSIZE,
            color="#7a7a7a", zorder=6,
        )
        return
    if value >= 0.16:
        axis.text(
            x, value - 0.022, f"{value:.3f}", ha="center", va="top",
            fontsize=BAR_LABEL_FONTSIZE, color=inside_color, zorder=6,
        )
    else:
        axis.text(
            x, value + 0.015, f"{value:.3f}", ha="center", va="bottom",
            fontsize=BAR_LABEL_FONTSIZE, color="#1f1f1f", zorder=6,
        )


def oof_comparison_figure(
    task: str,
    metric: str,
    metric_label: str,
    metric_title: str,
    group_key: str,
    group_label: str,
    experiments: tuple[Experiment, ...],
    results: dict[str, dict[str, OofResult]],
    cohort_line: str,
    n_folds: int,
    output_stem: Path,
) -> list[Path]:
    """One grouped bar per experiment x scope; every bar is a single pooled-OOF value."""
    positions = np.arange(len(experiments), dtype=float)
    axes_width_in = max(MIN_BAR_AXES_WIDTH_IN, AXES_WIDTH_PER_EXPERIMENT_IN * len(experiments))
    figure_width_in = LEFT_MARGIN_IN + axes_width_in + RIGHT_MARGIN_IN

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

    notes = [cohort_line, POOLED_NOTE.format(n_folds=n_folds)]
    if metric in THRESHOLD_FREE_METRICS:
        if group_key == "threshold_optimized":
            notes.append(THRESHOLD_FREE_NOTE)
    elif group_key == "threshold_optimized":
        notes.append(THRESHOLD_OPTIMIZED_NOTE)
    else:
        notes.append(FIXED_THRESHOLD_NOTE)
    note_lines = _wrap_text(notes, min(figure_width_in, NOTE_MAX_WIDTH_IN), NOTE_FONTSIZE)

    tick_labels = [experiment.tick_label() for experiment in experiments]
    tick_lines = max(label.count("\n") + 1 for label in tick_labels)
    title_lines = _wrap_text([metric_title], figure_width_in, TITLE_FONTSIZE)
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
        scope_offsets = ((0.0, SCOPES[0], legend_labels[0]),)
        bar_width = 0.60
    else:
        scope_offsets = zip((-0.27, 0.0, 0.27), SCOPES, legend_labels)
        bar_width = 0.25
    for offset, scope, legend_label in scope_offsets:
        scoped = [results[experiment.key][scope.key] for experiment in experiments]
        heights = np.asarray(
            [result.value(metric) if math.isfinite(result.value(metric)) else 0.0 for result in scoped],
            dtype=float,
        )
        bars = axis.bar(
            positions + offset, heights, width=bar_width, color=SCOPE_COLORS[scope.key],
            edgecolor=SCOPE_EDGE_COLORS[scope.key], linewidth=0.6, zorder=3, label=legend_label,
        )
        if len(SCOPES) == 1:
            for bar, experiment in zip(bars, experiments):
                if experiment.threshold_mode != FIXED_THRESHOLD_MODE:
                    bar.set_hatch("//")
                    bar.set_linewidth(1.3)
        inside = "#ffffff" if len(SCOPES) == 1 else (
            "#22364f" if scope.key == "overall" else "#ffffff"
        )
        for index, result in enumerate(scoped):
            _bar_value_label(axis, positions[index] + offset, result.value(metric), inside)

    axis.set_xticks(list(positions))
    axis.set_xticklabels(tick_labels, fontsize=TICK_FONTSIZE)
    axis.set_xlim(-0.65, len(experiments) - 0.35)
    axis.set_ylim(0.0, Y_AXIS_TOP)
    axis.set_yticks(np.round(np.arange(0.0, 1.01, 0.1), 2))
    axis.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    axis.set_ylabel(f"{metric_label} (pooled out-of-fold)", fontsize=10)
    axis.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    if len(SCOPES) != 1:
        for index, experiment in enumerate(experiments):
            if experiment.threshold_mode != FIXED_THRESHOLD_MODE:
                axis.axvspan(positions[index] - 0.5, positions[index] + 0.5, color="#f3e4ef", zorder=0)
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


def _curve_figure_skeleton(
    task: str, title: str, group_label: str, notes: list[str], legend_rows: int = 0
) -> tuple[plt.Figure, list[plt.Axes], object, float]:
    n_panels = len(SCOPES)
    axes_width_in = n_panels * PANEL_WIDTH_IN + (n_panels - 1) * PANEL_GAP_IN
    figure_width_in = LEFT_MARGIN_IN + axes_width_in + RIGHT_MARGIN_IN
    note_lines = _wrap_text(notes, min(figure_width_in, NOTE_MAX_WIDTH_IN), NOTE_FONTSIZE)
    title_lines = _wrap_text([title], figure_width_in, TITLE_FONTSIZE)
    subtitle_lines = _wrap_text(
        [SCOPES[0].label if len(SCOPES) == 1 else f"{group_label} - {task} task"],
        figure_width_in,
        SUBTITLE_FONTSIZE,
    )
    layout = _stacked_figure(
        axes_width_in,
        PANEL_HEIGHT_IN + ROC_TITLE_BLOCK_IN,
        _title_block_in(len(title_lines), len(subtitle_lines)),
        tick_lines=ROC_TICK_LINES,
        legend_rows=legend_rows,
        note_lines=len(note_lines),
    )
    axes = _panel_axes(layout.figure, layout.axis, n_panels, figure_width_in)
    _draw_title(layout.figure, title_lines, subtitle_lines, layout.title_top)
    _draw_notes(layout.figure, note_lines, layout.note_top)
    return layout.figure, axes, layout, figure_width_in


def _shares_table(experiment: Experiment, group: tuple[Experiment, ...]) -> bool:
    return any(other.template == experiment.template for other in group)


def _optimized_overlays(group: tuple[Experiment, ...]) -> tuple[Experiment, ...]:
    """Optimized-threshold rules that belong to a table already drawn in this group."""
    return tuple(
        experiment
        for experiment in EXPERIMENTS
        if experiment.label_column is not None
        and experiment not in group
        and _shares_table(experiment, group)
    )


def oof_roc_figure(
    task: str,
    group_key: str,
    group_label: str,
    experiments: tuple[Experiment, ...],
    results: dict[str, dict[str, OofResult]],
    n_folds: int,
    output_stem: Path,
) -> list[Path]:
    """Three pooled-OOF ROC panels (Overall, HUP, UCLA), one curve per experiment."""
    overlays = _optimized_overlays(experiments)
    notes = [POOLED_NOTE.format(n_folds=n_folds)]
    if overlays:
        notes.append(SHARED_CURVE_NOTE)
    figure, axes, _, _ = _curve_figure_skeleton(
        task, "Pooled out-of-fold ROC", group_label, notes
    )

    for index, (axis, scope) in enumerate(zip(axes, SCOPES)):
        reference = results[experiments[0].key][scope.key]
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
        for experiment in experiments:
            result = results[experiment.key][scope.key]
            false_positive_rate, true_positive_rate = result.roc
            axis.plot(
                false_positive_rate, true_positive_rate,
                color=EXPERIMENT_COLORS[experiment.short], linewidth=1.8,
                label=f"{experiment.short} (pooled OOF AUROC = {result.auc:.3f})",
            )
        for experiment in overlays:
            # Threshold-independent curve: identical to its 0.5 counterpart, so
            # it is named in the legend instead of being drawn twice.
            result = results[experiment.key][scope.key]
            axis.plot(
                [], [], color=EXPERIMENT_COLORS[experiment.short], linewidth=1.8, linestyle=":",
                label=f"{experiment.short} (same curve, AUROC = {result.auc:.3f})",
            )
        axis.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1.0, label="Chance")
        axis.set_title(_panel_title(scope.label, reference.counts), fontsize=9.5)
        axis.grid(color="#eeeeee", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.legend(loc="lower right", frameon=False, fontsize=7.5, handlelength=1.6, borderpad=0.2)
    axes[0].set_ylabel("True positive rate", fontsize=9.5)
    return _save(figure, output_stem)


def oof_pr_figure(
    task: str,
    group_key: str,
    group_label: str,
    experiments: tuple[Experiment, ...],
    results: dict[str, dict[str, OofResult]],
    n_folds: int,
    output_stem: Path,
) -> list[Path]:
    """Three pooled-OOF Precision-Recall panels: the PPV-versus-TPR trade-off."""
    overlays = _optimized_overlays(experiments)
    notes = [POOLED_NOTE.format(n_folds=n_folds), PR_AXIS_NOTE]
    if overlays:
        notes.append(SHARED_CURVE_NOTE)
        notes.append(OPTIMIZED_POINT_NOTE)
    figure, axes, _, _ = _curve_figure_skeleton(
        task, "Pooled out-of-fold Precision-Recall (PPV vs TPR)", group_label, notes
    )

    for index, (axis, scope) in enumerate(zip(axes, SCOPES)):
        reference = results[experiments[0].key][scope.key]
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("TPR / Recall (Sensitivity)", fontsize=9.5)
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
        for experiment in experiments:
            result = results[experiment.key][scope.key]
            recall, precision = result.pr
            axis.plot(
                recall, precision, color=EXPERIMENT_COLORS[experiment.short], linewidth=1.8,
                label=f"{experiment.short} (pooled OOF AP = {result.ap:.3f})",
            )
        # Operating points last so the markers sit on top of every curve.
        for experiment in experiments:
            point = results[experiment.key][scope.key].operating_point
            if point is None:
                continue
            axis.plot(
                point[0], point[1], marker="o", markersize=6.5,
                color=EXPERIMENT_COLORS[experiment.short], markeredgecolor="#ffffff",
                markeredgewidth=1.1, linestyle="none", zorder=6,
            )
        for experiment in overlays:
            result = results[experiment.key][scope.key]
            point = result.operating_point
            if point is None:
                continue
            axis.plot(
                point[0], point[1], marker="D", markersize=6.5,
                color=EXPERIMENT_COLORS[experiment.short], markeredgecolor="#ffffff",
                markeredgewidth=1.1, linestyle="none", zorder=7,
            )
        prevalence = float(reference.counts["prevalence"])
        axis.axhline(
            prevalence, color="#888888", linestyle=":", linewidth=1.0,
            label=f"No skill (prevalence = {prevalence:.3f})",
        )
        axis.set_title(_panel_title(scope.label, reference.counts), fontsize=9.5)
        axis.grid(color="#eeeeee", linewidth=0.6)
        axis.set_axisbelow(True)
        handles, labels = axis.get_legend_handles_labels()
        handles.append(
            Line2D([], [], marker="o", color="#4a4a4a", markeredgecolor="#ffffff",
                   linestyle="none", markersize=6.5)
        )
        labels.append("Operating point @ threshold 0.5")
        if overlays:
            handles.append(
                Line2D([], [], marker="D", color=EXPERIMENT_COLORS[overlays[0].short],
                       markeredgecolor="#ffffff", linestyle="none", markersize=6.5)
            )
            labels.append(f"{overlays[0].short} @ optimized threshold")
        # Lower-left is the empty corner of a PR panel; the curves live top-left
        # to bottom-right, so the legend never sits on top of them.
        axis.legend(
            handles, labels, loc="lower left", frameon=False, fontsize=7.5,
            handlelength=1.6, borderpad=0.2,
        )
    axes[0].set_ylabel("PPV (Precision)", fontsize=9.5)
    return _save(figure, output_stem)


def oof_distribution_figure(
    task: str,
    group_key: str,
    group_label: str,
    experiments: tuple[Experiment, ...],
    loaded: dict[str, pd.DataFrame],
    n_folds: int,
    output_stem: Path,
) -> list[Path]:
    """Pooled-OOF predicted-probability distributions, pathological vs non-pathological."""
    n_panels = len(experiments)
    axes_width_in = n_panels * PANEL_WIDTH_IN + (n_panels - 1) * PANEL_GAP_IN
    figure_width_in = LEFT_MARGIN_IN + axes_width_in + RIGHT_MARGIN_IN
    legend_labels = ["Pathological (true label = 1)", "Non-pathological (true label = 0)"]
    legend_columns = _legend_columns(
        legend_labels + [f"Decision threshold {DECISION_THRESHOLD}"], figure_width_in
    )
    legend_rows = math.ceil((len(legend_labels) + 1) / legend_columns)
    notes = [POOLED_NOTE.format(n_folds=n_folds), DISTRIBUTION_NOTE]
    note_lines = _wrap_text(notes, min(figure_width_in, NOTE_MAX_WIDTH_IN), NOTE_FONTSIZE)
    title_lines = _wrap_text(
        ["Pooled out-of-fold predicted pathological probability"], figure_width_in, TITLE_FONTSIZE
    )
    subtitle_lines = _wrap_text(
        [SCOPES[0].label if len(SCOPES) == 1 else f"{group_label} - {task} task"],
        figure_width_in,
        SUBTITLE_FONTSIZE,
    )
    layout = _stacked_figure(
        axes_width_in,
        PANEL_HEIGHT_IN + ROC_TITLE_BLOCK_IN,
        _title_block_in(len(title_lines), len(subtitle_lines)),
        tick_lines=ROC_TICK_LINES,
        legend_rows=legend_rows,
        note_lines=len(note_lines),
    )
    figure = layout.figure
    axes = _panel_axes(figure, layout.axis, n_panels, figure_width_in)

    for index, (axis, experiment) in enumerate(zip(axes, experiments)):
        frame = loaded[experiment.key]
        scores = frame["pathological_score"].to_numpy(dtype=float)
        labels = frame["true_label"].to_numpy(dtype=int)
        for mask, color, legend_label in (
            (labels == 0, NON_PATHOLOGICAL_COLOR, legend_labels[1]),
            (labels == 1, PATHOLOGICAL_COLOR, legend_labels[0]),
        ):
            values = scores[mask]
            if values.size == 0:
                continue
            weights = np.full(values.size, 1.0 / values.size)
            axis.hist(
                values, bins=DISTRIBUTION_BINS, weights=weights, color=color, alpha=0.45,
                label=legend_label if index == 0 else None, zorder=2,
            )
            axis.hist(
                values, bins=DISTRIBUTION_BINS, weights=weights, histtype="step", color=color,
                linewidth=1.4, zorder=3,
            )
        axis.axvline(
            DECISION_THRESHOLD, color="#4a4a4a", linestyle="--", linewidth=1.0, zorder=4,
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Predicted pathological probability", fontsize=9.5)
        axis.tick_params(labelsize=8.5)
        counts = count_cohort(frame)
        axis.set_title(
            f"{experiment.short} - {experiment.display}\n"
            f"{counts['n_channels']} channels\n({counts['n_pathological']} pathological)",
            fontsize=9.5,
        )
        axis.grid(color="#eeeeee", linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Fraction of channels within class", fontsize=9.5)
    top = max(axis.get_ylim()[1] for axis in axes)
    for index, axis in enumerate(axes):
        axis.set_ylim(0.0, top)
        if index > 0:
            axis.tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="#4a4a4a", linestyle="--", linewidth=1.0))
    labels.append(f"Decision threshold {DECISION_THRESHOLD}")
    figure.legend(
        handles,
        labels,
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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def plot_oof_experiments(task: str, analysis_root: Path, output_dir: Path) -> list[Path]:
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

    # Same observations, true labels, site labels and outer folds everywhere.
    folds = assert_shared_evaluation(loaded)
    n_folds = len(folds)
    if n_folds != N_SPLITS:
        raise AssertionError(f"expected {N_SPLITS} outer folds, found {folds}")

    validation = [
        f"[validation] {len(EXPERIMENTS)} decision rules over {len(tables)} frozen prediction "
        f"tables share {len(next(iter(loaded.values())))} out-of-fold observations, true labels, "
        "site labels and outer-fold assignments",
        f"[validation] outer folds pooled: {folds}",
    ]
    validation.extend(assert_pooled_oof_partition(loaded, folds))
    validation.extend(assert_fixed_threshold_rule(EXPERIMENTS))
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

    results: dict[str, dict[str, OofResult]] = {}
    summary_rows: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        results[experiment.key] = {}
        for scope in SCOPES:
            result = evaluate_oof_scope(loaded[experiment.key], experiment, scope, task)
            results[experiment.key][scope.key] = result
            validation.extend(f"[warning] {note}" for note in result.notes)
            summary_rows.append(
                {
                    "experiment": experiment.label,
                    "scope": scope.key,
                    "n": int(result.counts["n_channels"]),
                    "n_positive": int(result.counts["n_pathological"]),
                    "n_negative": int(result.counts["n_non_pathological"]),
                    "prevalence": result.counts["prevalence"],
                    "auc": result.auc,
                    "ap": result.ap,
                    "ppv": result.ppv,
                    "tpr": result.tpr,
                }
            )

    validation.extend(assert_threshold_free_metrics(results))

    cohort_line = cohort_caption(next(iter(loaded.values())))
    written: list[Path] = []
    for group_key, group_label in GROUPS:
        group_experiments = tuple(
            experiment for experiment in EXPERIMENTS if group_key in experiment.groups
        )
        for metric, metric_label, metric_title in OOF_METRICS:
            if group_key == "threshold_optimized" and metric in THRESHOLD_FREE_METRICS:
                # AUROC is threshold-free, so EXP5 and EXP5-OPT have identical
                # bars by construction; the bar figure would carry no
                # information. The value is still published in the summary CSV,
                # in the standard comparisons and in the ROC panels.
                continue
            written.extend(
                oof_comparison_figure(
                    task, metric, metric_label, metric_title, group_key, group_label,
                    group_experiments, results, cohort_line, n_folds,
                    output_dir / f"{task}_oof_comparison_{group_key}_{metric}",
                )
            )
        if group_key not in CURVE_GROUPS:
            # The threshold-optimized group would only duplicate curves that are
            # threshold-independent by construction; it stays a bar comparison.
            continue
        written.extend(
            oof_roc_figure(
                task, group_key, group_label, group_experiments, results, n_folds,
                output_dir / f"{task}_oof_roc_{group_key}",
            )
        )
        written.extend(
            oof_pr_figure(
                task, group_key, group_label, group_experiments, results, n_folds,
                output_dir / f"{task}_oof_pr_{group_key}",
            )
        )
        written.extend(
            oof_distribution_figure(
                task, group_key, group_label, group_experiments, loaded, n_folds,
                output_dir / f"{task}_oof_probability_distribution_{group_key}",
            )
        )

    summary_table = pd.DataFrame(summary_rows)[SUMMARY_COLUMNS]
    summary_path = output_dir / f"{task}_oof_performance_summary.csv"
    atomic_csv(summary_table, summary_path)
    written.append(summary_path)

    for line in validation:
        print(line)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("interictal", "ictal"))
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="defaults to <analysis-root>/ml_performance_figures_oof_v1/<task>",
    )
    args = parser.parse_args()
    analysis_root = args.analysis_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else analysis_root / "ml_performance_figures_oof_v1" / resolve_task(args.task)
    )
    for path in plot_oof_experiments(args.task, analysis_root, output_dir):
        print(path)


if __name__ == "__main__":
    main()
