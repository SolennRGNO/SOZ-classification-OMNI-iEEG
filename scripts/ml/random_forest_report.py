"""Dedicated read-only HTML report for ``ml_random_forest_complexity_nested_v1``.

The logistic report renders coefficients and solver convergence, neither of
which exists for a forest, so this experiment publishes its own renderer.  It
reads the frozen artifacts and never modifies them.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from scripts.ml.common import atomic_text


# Result artifacts the Random Forest report requires.
RESULT_ARTIFACTS = (
    "model_run_summary.json",
    "fold_audit.csv",
    "fold_metrics.csv",
    "fit_status.csv",
    "overall_site_metrics.csv",
    "participant_metrics.csv",
    "participant_metric_summary.csv",
    "bootstrap_confidence_intervals.csv",
    "rf_selected_hyperparameters_by_outer_fold.csv",
    "rf_nested_summary.csv",
    "rf_nested_tuning.csv",
    "rf_inner_fold_audit.csv",
    "rf_impurity_importance_by_fold.csv",
    "rf_permutation_importance_by_fold.csv",
    "rf_feature_importance_summary.csv",
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


def _leakage_audit(tables: dict[str, pd.DataFrame], model_run: dict) -> pd.DataFrame:
    """Summarize the executed leakage assertions from the published artifacts."""
    audit = tables["fold_audit"]
    inner = tables["rf_inner_fold_audit"]
    tuning = tables["rf_nested_tuning"]
    assignments_are_disjoint = all(
        set(json.loads(row["train_participant_ids"])).isdisjoint(
            json.loads(row["test_participant_ids"])
        )
        for _, row in audit.iterrows()
    )
    return pd.DataFrame(
        [
            {
                "check": "outer TRAIN / TEST participant disjointness",
                "observed": f"{'disjoint in all folds' if assignments_are_disjoint else 'VIOLATED'}",
            },
            {
                "check": "inner TRAIN / VALIDATION participant overlap (max)",
                "observed": int(inner["inner_train_validation_participant_overlap_count"].max()),
            },
            {
                "check": "outer TEST participants reaching inner CV (max)",
                "observed": int(inner["outer_test_participant_overlap_count"].max()),
            },
            {
                "check": "outer TEST rows reaching inner CV (max)",
                "observed": int(inner["outer_test_row_overlap_count"].max()),
            },
            {
                "check": "inner-fold weights fit on inner TRAIN participants only",
                "observed": bool(
                    (inner["weight_fit_participant_ids"] == inner["inner_train_participant_ids"]).all()
                ),
            },
            {
                "check": "candidate configurations scored per inner fold",
                "observed": int(tuning.groupby(["outer_fold", "inner_fold"]).size().max()),
            },
            {
                "check": "candidates valid for tuning",
                "observed": f"{int(tuning['valid_for_tuning'].sum())} / {len(tuning)}",
            },
            {
                "check": "StandardScaler fit anywhere",
                "observed": bool(model_run.get("standard_scaler_used", False)),
            },
            {
                "check": "hyperparameters chosen from outer performance",
                "observed": bool(model_run.get("hyperparameters_chosen_from_outer_performance", False)),
            },
            {
                "check": "threshold optimized",
                "observed": bool(model_run.get("threshold_was_optimized", False)),
            },
            {
                "check": "outer folds match the complexity-experiment protocol",
                "observed": bool(model_run.get("outer_folds_match_complexity_protocol", False)),
            },
        ]
    )


def generate_random_forest_report(dataset_dir: Path, results_dir: Path, output_html: Path) -> None:
    """Render the nested Random Forest artifacts without modifying them."""
    if output_html.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_html}")
    dataset_paths = {
        key: _first_existing(dataset_dir, names, f"dataset artifact '{key}'")
        for key, names in DATASET_ARTIFACTS.items()
    }
    result_paths = {
        Path(name).stem: _first_existing(results_dir, (name,), "Random Forest report input")
        for name in RESULT_ARTIFACTS
    }
    dataset_qc = json.loads(dataset_paths["dataset_qc"].read_text(encoding="utf-8"))
    model_run = json.loads(result_paths["model_run_summary"].read_text(encoding="utf-8"))
    tables = {
        name: pd.read_csv(path)
        for name, path in {**dataset_paths, **result_paths}.items()
        if path.suffix == ".csv"
    }
    site_label = " / ".join(tables["counts_site"]["dataset"].astype(str).str.upper())
    counts = pd.DataFrame(
        [{"measure": key, "observed": value} for key, value in dataset_qc["counts"].items()]
    )
    outer_cv = model_run.get("outer_cv", {})
    inner_cv = model_run.get("inner_cv", {})
    grid = model_run.get("hyperparameter_grid", {})
    model = model_run.get("model", {})
    bootstrap = model_run.get("bootstrap", {})
    importance = model_run.get("feature_importance", {})
    permutation = importance.get("permutation", {})
    protocol = pd.DataFrame(
        [
            {"setting": "Experiment ID", "value": str(model_run.get("experiment_id", ""))},
            {"setting": "Predictors", "value": f"{model_run.get('n_predictors', '?')} aggregated complexity predictors (no feature selection)"},
            {"setting": "Standardization", "value": str(model_run.get("normalization", ""))},
            {"setting": "Sample weighting", "value": str(model_run.get("weighting", ""))},
            {"setting": "Outer CV", "value": f"{outer_cv.get('class', 'StratifiedGroupKFold')}, {outer_cv.get('n_splits', '?')} folds, grouped by {outer_cv.get('groups', 'participant_id')}, stratified on {outer_cv.get('stratification', 'target')}, random_state={outer_cv.get('random_state', '?')}"},
            {"setting": "Inner CV", "value": f"{inner_cv.get('class', 'StratifiedGroupKFold')}, {inner_cv.get('n_splits', '?')} folds inside {inner_cv.get('scope', 'each outer training fold')}, random_state={inner_cv.get('random_state', '?')} — {inner_cv.get('purpose', '')}"},
            {"setting": "Tuning metric", "value": str(model_run.get("tuning_metric", "average_precision"))},
            {"setting": "Tie-breaking", "value": str(model_run.get("tie_breaking", ""))},
            {"setting": "Estimator", "value": f"{model.get('class', 'RandomForestClassifier')}, criterion={model.get('criterion', '?')}, n_estimators={model.get('n_estimators', '?')}, bootstrap={model.get('bootstrap', '?')}, class_weight={model.get('class_weight', None)}, random_state={model.get('random_state', '?')}"},
            {"setting": "max_depth grid", "value": ", ".join(str(value) for value in grid.get("max_depth", []))},
            {"setting": "min_samples_leaf grid", "value": ", ".join(str(value) for value in grid.get("min_samples_leaf", []))},
            {"setting": "max_features grid", "value": ", ".join(str(value) for value in grid.get("max_features", []))},
            {"setting": "Candidate configurations", "value": str(grid.get("n_candidates", "?"))},
            {"setting": "Decision threshold", "value": f"{model_run.get('decision_threshold', 0.5)} (optimized: {model_run.get('threshold_was_optimized', False)})"},
            {"setting": "Bootstrap", "value": f"{bootstrap.get('n_replicates', '?')} participant-cluster replicates, {bootstrap.get('confidence_level', '?')} {bootstrap.get('interval', 'percentile')} interval, random_state={bootstrap.get('random_state', '?')}, refit_models={bootstrap.get('refit_models', False)}"},
            {"setting": "Permutation importance", "value": f"scoring={permutation.get('scoring', '?')}, n_repeats={permutation.get('n_repeats', '?')}, random_state={permutation.get('random_state', '?')}, {permutation.get('evaluated_on', '')}"},
            {"setting": "Full scientific configuration", "value": str(model_run.get("scientific_full_configuration", False))},
            {"setting": "Git commit", "value": str(model_run.get("git_commit", "unavailable"))},
        ]
    )
    selected = pd.DataFrame(
        [
            {"outer_fold": fold, **values}
            for fold, values in sorted(
                (model_run.get("selected_hyperparameters_by_outer_fold") or {}).items(),
                key=lambda item: int(item[0]),
            )
        ]
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>OMNI iEEG nested Random Forest (Complexity) report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;color:#1f2937}}table{{border-collapse:collapse;margin:1rem 0 2rem;font-size:.86rem;width:100%}}th,td{{border:1px solid #d1d5db;padding:.35rem .5rem;text-align:left}}th{{background:#eef2ff}}code{{background:#f3f4f6;padding:.1rem .25rem}}.note{{background:#fffbeb;border-left:4px solid #f59e0b;padding:.8rem}}</style>
</head><body><h1>Nested Random Forest (Complexity, all predictors) — {html.escape(str(dataset_qc.get('task', 'unknown')))} task</h1>
<p>Experiment <code>{html.escape(str(model_run.get('experiment_id', '')))}</code>. Predictors: <code>{html.escape(str(model_run.get('n_predictors', '?')))}</code>. Git commit: <code>{html.escape(str(model_run.get('git_commit', 'unavailable')))}</code>.</p>
<p class="note">Channels are final observations; valid epochs are pooled before aggregation. No StandardScaler, no imputation, no resampling, no feature selection and no dimensionality reduction are applied. All threshold metrics use the frozen pathological-score threshold of 0.5. This report evaluates the Random Forest on its own; it performs no automatic comparison against, or selection between, the logistic experiments.</p>
<h2>Cohort and dataset summary</h2>{_table(counts)}<h3>Per site</h3>{_table(tables['counts_site'])}<h3>Per participant</h3>{_table(tables['counts_participant'])}
<h2>Protocol</h2>{_table(protocol)}
<h2>Nested hyperparameter selection</h2>
<p class="note">Hyperparameters are selected inside each outer training fold only, by mean inner Average Precision. Outer folds may select different configurations; the out-of-fold predictions below are the unbiased evaluation of the whole nested procedure.</p>
<h3>Selected Random Forest parameters by outer fold</h3>{_table(selected)}
{_table(tables['rf_selected_hyperparameters_by_outer_fold'])}
<h3>Inner performance of candidate configurations</h3>{_table(tables['rf_nested_summary'])}
<h3>Inner tuning trace</h3>{_table(tables['rf_nested_tuning'])}
<h2>Final out-of-fold performance</h2>
<h3>Overall and by site ({html.escape(site_label)})</h3>{_table(tables['overall_site_metrics'])}
<h3>Participant-level metric summary</h3>{_table(tables['participant_metric_summary'])}
<h3>Participant-level metrics</h3>{_table(tables['participant_metrics'])}
<h3>Participant-cluster bootstrap 95% percentile intervals</h3>{_table(tables['bootstrap_confidence_intervals'])}
<h2>Fold metrics and fit status</h2><h3>Fold metrics</h3>{_table(tables['fold_metrics'])}<h3>Fit status</h3>{_table(tables['fit_status'])}
<h2>Feature importance (post-hoc descriptive)</h2>
<p class="note">Both importance families are computed after each outer fold's model and predictions are frozen. They never influence hyperparameter tuning, feature selection, model fitting or the decision threshold. Impurity importance is biased towards high-cardinality and correlated predictors; permutation importance is measured on the outer test fold only and is reported for description, not inference.</p>
<h3>Importance summary across outer folds</h3>{_table(tables['rf_feature_importance_summary'])}
<h3>Impurity importance by outer fold</h3>{_table(tables['rf_impurity_importance_by_fold'])}
<h3>Permutation importance by outer fold</h3>{_table(tables['rf_permutation_importance_by_fold'])}
<h2>Leakage and fold audit</h2>{_table(_leakage_audit(tables, model_run))}
<h3>Outer fold composition</h3>{_table(tables['fold_audit'])}
<h3>Inner fold audit</h3>{_table(tables['rf_inner_fold_audit'])}
<h2>Feature QC</h2><h3>Constant and near-constant predictors</h3>{_table(tables['quality'])}<h3>Exact duplicated predictors</h3>{_table(tables['duplicates'])}<h3>Highly correlated predictors</h3>{_table(tables['correlations'])}<h3>Feature distributions</h3>{_table(tables['distributions'])}
</body></html>\n"""
    atomic_text(document, output_html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    generate_random_forest_report(
        args.dataset_dir.resolve(), args.results_dir.resolve(), args.output_html.resolve()
    )
    print(args.output_html.resolve())


if __name__ == "__main__":
    main()
