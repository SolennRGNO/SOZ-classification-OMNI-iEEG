"""Dedicated read-only HTML report for ``ml_logistic_complexity_c1_nested_tuned_v1``.

The shared logistic report renders a single frozen threshold, which cannot
describe an experiment that publishes results at two thresholds.  This
experiment therefore publishes its own renderer.  It reads the frozen artifacts
and never modifies them.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from scripts.ml.common import atomic_text


# Result artifacts the tuned-threshold report requires.
RESULT_ARTIFACTS = (
    "model_run_summary.json",
    "fold_audit.csv",
    "fold_metrics.csv",
    "convergence.csv",
    "model_coefficients.csv",
    "overall_site_metrics.csv",
    "participant_metrics.csv",
    "participant_metric_summary.csv",
    "bootstrap_confidence_intervals.csv",
    "standard_threshold_overall_site_metrics.csv",
    "optimized_threshold_overall_site_metrics.csv",
    "optimized_threshold_participant_metrics.csv",
    "optimized_threshold_participant_metric_summary.csv",
    "optimized_threshold_bootstrap_confidence_intervals.csv",
    "outer_fold_metrics_threshold_0p5.csv",
    "outer_fold_metrics_optimized_threshold.csv",
    "nested_model_tuning.csv",
    "nested_model_tuning_summary.csv",
    "selected_hyperparameters_by_outer_fold.csv",
    "threshold_tuning_by_outer_fold.csv",
    "threshold_tuning_trace.csv",
    "selected_features_by_fold.csv",
    "selection_frequencies.csv",
)
# Dataset artifacts, keyed exactly as the HTML body indexes them.
DATASET_ARTIFACTS = {
    "dataset_qc": ("complexity_dataset_qc_summary.json", "dataset_qc_summary.json"),
    "counts_participant": ("counts_per_participant.csv",),
    "counts_site": ("counts_per_site.csv",),
    "quality": ("complexity_feature_quality.csv", "feature_quality.csv"),
    "distributions": ("complexity_feature_distributions.csv", "feature_distributions.csv"),
    "duplicates": ("complexity_exact_duplicate_features.csv", "exact_duplicate_features.csv"),
    "correlations": ("complexity_high_correlations.csv", "high_correlations.csv"),
}
# The trade-off table is rendered for the metrics a threshold can move, plus the
# two threshold-free metrics that must stay identical.
COMPARED_METRICS = (
    "auroc",
    "average_precision",
    "precision",
    "sensitivity",
    "specificity",
    "fpr",
    "fnr",
    "f1",
    "balanced_accuracy",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
)
TRACE_ROWS_PER_FOLD = 25


def _table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    display = frame if max_rows is None else frame.head(max_rows)
    return display.to_html(index=False, border=0, na_rep="NA", escape=True)


def _first_existing(directory: Path, names: tuple[str, ...], label: str) -> Path:
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    expected = [str(directory / name) for name in names]
    raise FileNotFoundError(f"missing {label}; expected one of: {expected}")


def _threshold_comparison(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Overall and per-site metrics at 0.5 next to the optimized threshold."""
    # The companion 0.5 table is used rather than the shared one because only it
    # carries FPR and FNR, which the trade-off has to show at both thresholds.
    standard = tables["standard_threshold_overall_site_metrics"]
    optimized = tables["optimized_threshold_overall_site_metrics"]
    merged = standard.merge(
        optimized,
        on=["scope", "scope_value", "metric"],
        how="outer",
        suffixes=("_threshold_0p5", "_optimized_threshold"),
        validate="one_to_one",
    )
    merged = merged[merged["metric"].isin(COMPARED_METRICS)].copy()
    merged["metric"] = pd.Categorical(
        merged["metric"], categories=list(COMPARED_METRICS), ordered=True
    )
    merged["delta"] = (
        pd.to_numeric(merged["value_optimized_threshold"], errors="coerce")
        - pd.to_numeric(merged["value_threshold_0p5"], errors="coerce")
    )
    merged["threshold_free"] = merged["metric"].isin(["auroc", "average_precision"])
    return merged.sort_values(["scope", "scope_value", "metric"], kind="stable")[
        [
            "scope", "scope_value", "metric",
            "value_threshold_0p5", "value_optimized_threshold", "delta", "threshold_free",
        ]
    ].reset_index(drop=True)


def _trace_excerpt(trace: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Highest-PPV eligible candidates per outer fold, with the selected one flagged.

    The complete candidate trace stays in ``threshold_tuning_trace.csv``; only an
    excerpt is embedded so the report remains readable.
    """
    chosen = dict(
        zip(selected["outer_fold"].astype(int), selected["selected_threshold"].astype(float))
    )
    pieces = []
    for fold, group in trace.groupby("outer_fold", sort=True):
        eligible = group[group["eligible_under_minimum_tpr"].astype(bool)]
        excerpt = eligible.sort_values(
            ["ppv", "tpr", "threshold"], ascending=[False, False, True], kind="stable"
        ).head(TRACE_ROWS_PER_FOLD).copy()
        excerpt["selected"] = excerpt["threshold"].eq(chosen.get(int(fold)))
        pieces.append(excerpt)
    return pd.concat(pieces, ignore_index=True) if pieces else trace.head(0)


def _leakage_audit(tables: dict[str, pd.DataFrame], model_run: dict) -> pd.DataFrame:
    """Summarize the executed leakage assertions from the published artifacts."""
    audit = tables["fold_audit"]
    tuning = tables["nested_model_tuning"]
    selected = tables["selected_features_by_fold"]
    threshold = tables["threshold_optimization_summary"]
    disjoint = all(
        set(json.loads(row["train_participant_ids"])).isdisjoint(
            json.loads(row["test_participant_ids"])
        )
        for _, row in audit.iterrows()
    )
    optimization = model_run.get("threshold_optimization", {})
    return pd.DataFrame(
        [
            {
                "check": "outer TRAIN / TEST participant disjointness",
                "observed": "disjoint in all folds" if disjoint else "VIOLATED",
            },
            {
                "check": "inner TRAIN / VALIDATION participant overlap (max)",
                "observed": int(tuning["inner_train_validation_participant_overlap_count"].max()),
            },
            {
                "check": "outer TEST participants reaching inner CV (max)",
                "observed": int(tuning["outer_test_participant_overlap_count"].max()),
            },
            {
                "check": "outer TEST rows reaching inner CV (max)",
                "observed": int(tuning["outer_test_row_overlap_count"].max()),
            },
            {
                "check": "inner scaler fit on inner TRAIN participants only",
                "observed": bool(
                    (tuning["inner_train_participant_ids"] == tuning["scaler_fit_participant_ids"]).all()
                ),
            },
            {
                "check": "inner weights fit on inner TRAIN participants only",
                "observed": bool(
                    (tuning["inner_train_participant_ids"] == tuning["weight_fit_participant_ids"]).all()
                ),
            },
            {
                "check": "final mRMR fit / outer TEST row overlap (max)",
                "observed": int(selected["fit_test_row_overlap_count"].max()),
            },
            {
                "check": "final mRMR fit / outer TEST participant overlap (max)",
                "observed": int(selected["fit_test_participant_overlap_count"].max()),
            },
            {
                "check": "(N, C) candidates scored per inner fold",
                "observed": int(tuning.groupby(["outer_fold", "inner_fold"]).size().max()),
            },
            {
                "check": "candidates valid for tuning",
                "observed": f"{int(tuning['valid_for_tuning'].sum())} / {len(tuning)}",
            },
            {
                "check": "threshold selected from",
                "observed": str(threshold["threshold_selected_from"].iloc[0]),
            },
            {
                "check": "outer TEST used for threshold selection",
                "observed": bool(optimization.get("outer_test_used_for_threshold", True)),
            },
            {
                "check": "mRMR preselected on the full outer TRAIN before inner CV",
                "observed": bool(
                    (model_run.get("feature_selection_details") or {}).get(
                        "preselected_on_full_outer_train_before_inner_cv", True
                    )
                ),
            },
            {
                "check": "hyperparameters chosen from outer performance",
                "observed": bool(
                    model_run.get("hyperparameters_chosen_from_outer_performance", True)
                ),
            },
            {
                "check": "outer folds match the complexity-experiment protocol",
                "observed": bool(model_run.get("outer_folds_match_complexity_protocol", False)),
            },
            {
                "check": "bootstrap re-optimizes the threshold",
                "observed": bool(
                    (model_run.get("bootstrap") or {}).get("reoptimizes_threshold", True)
                ),
            },
            {
                "check": "imputation / resampling / dimensionality reduction",
                "observed": (
                    f"{model_run.get('imputation', True)} / {model_run.get('resampling', True)} / "
                    f"{model_run.get('dimensionality_reduction', True)}"
                ),
            },
        ]
    )


def generate_tuned_threshold_report(
    dataset_dir: Path, results_dir: Path, output_html: Path
) -> None:
    """Render the tuned nested artifacts without modifying them."""
    if output_html.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_html}")
    dataset_paths = {
        key: _first_existing(dataset_dir, names, f"dataset artifact '{key}'")
        for key, names in DATASET_ARTIFACTS.items()
    }
    result_paths = {
        Path(name).stem: _first_existing(results_dir, (name,), "tuned report input")
        for name in RESULT_ARTIFACTS
    }
    dataset_qc = json.loads(dataset_paths["dataset_qc"].read_text(encoding="utf-8"))
    model_run = json.loads(result_paths["model_run_summary"].read_text(encoding="utf-8"))
    tables = {
        name: pd.read_csv(path)
        for name, path in {**dataset_paths, **result_paths}.items()
        if path.suffix == ".csv"
    }
    tables["threshold_optimization_summary"] = tables["threshold_tuning_by_outer_fold"]
    site_label = " / ".join(tables["counts_site"]["dataset"].astype(str).str.upper())
    counts = pd.DataFrame(
        [{"measure": key, "observed": value} for key, value in dataset_qc["counts"].items()]
    )
    outer_cv = model_run.get("outer_cv", {})
    inner_cv = model_run.get("inner_cv", {})
    grid = model_run.get("hyperparameter_grid", {})
    model = model_run.get("model", {})
    bootstrap = model_run.get("bootstrap", {})
    optimization = model_run.get("threshold_optimization", {})
    selection = model_run.get("feature_selection_details", {})
    protocol = pd.DataFrame(
        [
            {"setting": "Experiment ID", "value": str(model_run.get("experiment_id", ""))},
            {"setting": "Predictors", "value": f"{model_run.get('n_predictors', '?')} aggregated complexity predictors, reduced per outer fold by mRMR"},
            {"setting": "Outer CV", "value": f"{outer_cv.get('class', 'StratifiedGroupKFold')}, {outer_cv.get('n_splits', '?')} folds, grouped by {outer_cv.get('groups', 'participant_id')}, shuffle={outer_cv.get('shuffle', '?')}, random_state={outer_cv.get('random_state', '?')}"},
            {"setting": "Inner CV", "value": f"{inner_cv.get('class', 'StratifiedGroupKFold')}, {inner_cv.get('n_splits', '?')} folds inside {inner_cv.get('scope', 'each outer training fold')}, grouped by {inner_cv.get('groups', 'participant_id')}, random_state={inner_cv.get('random_state', '?')} — {inner_cv.get('purpose', '')}"},
            {"setting": "N grid", "value": ", ".join(str(value) for value in grid.get("n_selected_features", []))},
            {"setting": "C grid", "value": ", ".join(str(value) for value in grid.get("C", []))},
            {"setting": "Candidate (N, C) pairs", "value": str(grid.get("n_candidates", "?"))},
            {"setting": "Tuning metric", "value": str(model_run.get("tuning_metric", "average_precision"))},
            {"setting": "Model tie-breaking", "value": str(model_run.get("tie_breaking", ""))},
            {"setting": "Estimator", "value": f"{model.get('class', 'LogisticRegression')}, penalty={model.get('penalty', '?')}, solver={model.get('solver', '?')}, max_iter={model.get('max_iter', '?')}, class_weight={model.get('class_weight', None)}, C={model.get('C', '?')}"},
            {"setting": "Standardization", "value": str(model_run.get("normalization", ""))},
            {"setting": "Sample weighting", "value": str(model_run.get("weighting", ""))},
            {"setting": "mRMR fit scope", "value": str(selection.get("fit_scope", ""))},
            {"setting": "Minimum TPR constraint", "value": f"{optimization.get('minimum_tpr', '?')} (configurable: {optimization.get('minimum_tpr_is_configurable', False)})"},
            {"setting": "Threshold objective", "value": str(optimization.get("objective", ""))},
            {"setting": "Threshold selection rule", "value": str(optimization.get("selection_rule", ""))},
            {"setting": "Threshold candidates", "value": str(optimization.get("candidate_rule", ""))},
            {"setting": "Threshold selected from", "value": str(optimization.get("selected_from", ""))},
            {"setting": "Fallback threshold on failure", "value": str(optimization.get("fallback_threshold_on_failure", None))},
            {"setting": "Bootstrap", "value": f"{bootstrap.get('n_replicates', '?')} participant-cluster replicates, {bootstrap.get('confidence_level', '?')} {bootstrap.get('interval', 'percentile')} interval, random_state={bootstrap.get('random_state', '?')}, refit_models={bootstrap.get('refit_models', False)}, reoptimizes_threshold={bootstrap.get('reoptimizes_threshold', False)}"},
            {"setting": "Imputation / resampling / PCA", "value": f"{model_run.get('imputation', False)} / {model_run.get('resampling', False)} / {model_run.get('dimensionality_reduction', False)}"},
            {"setting": "Git commit", "value": str(model_run.get("git_commit", "unavailable"))},
        ]
    )
    selected_by_fold = pd.DataFrame(
        [
            {
                "outer_fold": int(fold),
                "selected_N": value,
                "selected_C": (model_run.get("selected_c_by_outer_fold") or {}).get(fold),
                "selected_threshold": (
                    optimization.get("selected_threshold_by_outer_fold") or {}
                ).get(fold),
                "minimum_TPR": optimization.get("minimum_tpr"),
            }
            for fold, value in sorted(
                (selection.get("selected_n_by_outer_fold") or {}).items(),
                key=lambda item: int(item[0]),
            )
        ]
    )
    comparison = _threshold_comparison(tables)
    trace_excerpt = _trace_excerpt(
        tables["threshold_tuning_trace"], tables["threshold_tuning_by_outer_fold"]
    )
    task = html.escape(str(dataset_qc.get("task", "unknown")))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>OMNI iEEG nested tuned logistic (Complexity) report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;color:#1f2937}}table{{border-collapse:collapse;margin:1rem 0 2rem;font-size:.86rem;width:100%}}th,td{{border:1px solid #d1d5db;padding:.35rem .5rem;text-align:left}}th{{background:#eef2ff}}code{{background:#f3f4f6;padding:.1rem .25rem}}.note{{background:#fffbeb;border-left:4px solid #f59e0b;padding:.8rem}}</style>
</head><body><h1>Nested tuned logistic (Complexity, mRMR + L2) — {task} task</h1>
<p>Experiment <code>{html.escape(str(model_run.get('experiment_id', '')))}</code>. Predictors available: <code>{html.escape(str(model_run.get('n_predictors', '?')))}</code>. Git commit: <code>{html.escape(str(model_run.get('git_commit', 'unavailable')))}</code>.</p>
<p class="note"><strong>Threshold optimization changes threshold-dependent metrics only.</strong> AUROC and Average Precision are calculated from the continuous pathological probabilities and are therefore identical at threshold 0.5 and at the optimized threshold; the evaluator asserts this equality rather than assuming it. PPV, TPR/sensitivity, specificity, FPR, FNR, F1, balanced accuracy and the confusion counts are the only quantities the threshold can move. Every threshold was selected inside the corresponding outer training fold from inner out-of-fold probabilities, never from the outer test fold.</p>
<h2>Cohort and dataset summary</h2>{_table(counts)}<h3>Per site</h3>{_table(tables['counts_site'])}<h3>Per participant</h3>{_table(tables['counts_participant'])}
<h2>Protocol</h2>{_table(protocol)}
<h2>Nested tuning of N and C</h2>
<p class="note">N and C are tuned jointly inside each outer training fold by mean inner Average Precision over {html.escape(str(inner_cv.get('n_splits', '?')))} grouped inner folds. mRMR is recomputed inside every inner training split, never preselected on the full outer training fold. Outer folds may select different (N, C); the out-of-fold predictions below are the unbiased evaluation of the whole nested procedure.</p>
<h3>Selected N, C and threshold by outer fold</h3>{_table(selected_by_fold)}
{_table(tables['selected_hyperparameters_by_outer_fold'])}
<h3>Inner performance of every (N, C) pair</h3>{_table(tables['nested_model_tuning_summary'])}
<h3>Inner tuning trace</h3>{_table(tables['nested_model_tuning'])}
<h2>Threshold optimization</h2>
<p class="note">Objective: {html.escape(str(optimization.get('objective', '')))}. Rule: {html.escape(str(optimization.get('selection_rule', '')))}. Candidates: {html.escape(str(optimization.get('candidate_rule', '')))}. If no candidate reaches the minimum TPR the run fails; there is no fallback to 0.5.</p>
<h3>Selected threshold by outer fold</h3>{_table(tables['threshold_tuning_by_outer_fold'])}
<h3>Highest-PPV eligible candidates per outer fold (excerpt; full trace in threshold_tuning_trace.csv)</h3>{_table(trace_excerpt)}
<h2>PPV/TPR trade-off: threshold 0.5 versus optimized threshold</h2>
<p class="note">Rows flagged <code>threshold_free</code> are AUROC and Average Precision; their delta is zero by construction. FNR at both thresholds is included so the cost of the PPV gain in missed pathological channels is explicit.</p>
{_table(comparison)}
<h2>Out-of-fold performance at threshold 0.5</h2>
<h3>Overall and by site ({html.escape(site_label)})</h3>{_table(tables['standard_threshold_overall_site_metrics'])}
<h3>Overall and by site, shared evaluator tables</h3>{_table(tables['overall_site_metrics'])}
<h3>Per outer fold</h3>{_table(tables['outer_fold_metrics_threshold_0p5'])}
<h3>Participant-level metric summary</h3>{_table(tables['participant_metric_summary'])}
<h3>Participant-level metrics</h3>{_table(tables['participant_metrics'])}
<h3>Participant-cluster bootstrap 95% percentile intervals</h3>{_table(tables['bootstrap_confidence_intervals'])}
<h2>Out-of-fold performance at the optimized threshold</h2>
<p class="note">Each observation keeps the threshold of the outer fold in which it was tested. The participant-cluster bootstrap resamples those already frozen predictions and never re-optimizes a threshold, so it is descriptive uncertainty conditional on the fixed out-of-fold predictions and the selected thresholds.</p>
<h3>Overall and by site ({html.escape(site_label)})</h3>{_table(tables['optimized_threshold_overall_site_metrics'])}
<h3>Per outer fold</h3>{_table(tables['outer_fold_metrics_optimized_threshold'])}
<h3>Participant-level metric summary</h3>{_table(tables['optimized_threshold_participant_metric_summary'])}
<h3>Participant-level metrics</h3>{_table(tables['optimized_threshold_participant_metrics'])}
<h3>Participant-cluster bootstrap 95% percentile intervals</h3>{_table(tables['optimized_threshold_bootstrap_confidence_intervals'])}
<h2>Fold metrics, convergence and coefficients</h2>
<h3>Fold metrics (threshold 0.5)</h3>{_table(tables['fold_metrics'])}
<h3>Convergence</h3>{_table(tables['convergence'])}
<h3>Coefficients of the selected predictors</h3>{_table(tables['model_coefficients'])}
<h2>Selected predictors</h2>
<h3>Final selected predictors by outer fold</h3>{_table(tables['selected_features_by_fold'])}
<h3>Selection frequencies</h3>{_table(tables['selection_frequencies'])}
<h2>Leakage and fold audit</h2>{_table(_leakage_audit(tables, model_run))}
<h3>Outer fold composition</h3>{_table(tables['fold_audit'])}
<h2>Feature QC</h2><h3>Constant and near-constant predictors</h3>{_table(tables['quality'])}<h3>Exact duplicated predictors</h3>{_table(tables['duplicates'])}<h3>Highly correlated predictors</h3>{_table(tables['correlations'])}<h3>Feature distributions</h3>{_table(tables['distributions'])}
</body></html>\n"""
    atomic_text(document, output_html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    generate_tuned_threshold_report(
        args.dataset_dir.resolve(), args.results_dir.resolve(), args.output_html.resolve()
    )
    print(args.output_html.resolve())


if __name__ == "__main__":
    main()
