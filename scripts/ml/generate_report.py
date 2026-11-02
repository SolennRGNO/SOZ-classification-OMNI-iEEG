"""Generate a compact, read-only HTML report for logistic ML artifacts."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from scripts.ml.common import atomic_text


# Result artifacts every model report (baseline, Complexity B/C1/C1-nested/C2) must load.
RESULT_ARTIFACTS = (
    "model_run_summary.json", "fold_audit.csv", "fold_metrics.csv", "convergence.csv",
    "model_coefficients.csv", "overall_site_metrics.csv", "participant_metric_summary.csv",
    "bootstrap_confidence_intervals.csv",
)
# Every table the HTML body renders, keyed exactly as ``tables`` is keyed.
RENDERED_TABLES = (
    "counts_site", "counts_participant", "quality", "duplicates", "correlations", "distributions",
    "fold_audit", "fold_metrics", "convergence", "model_coefficients", "overall_site_metrics",
    "participant_metric_summary", "bootstrap_confidence_intervals",
)


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


def _result_table(results_dir: Path, name: str) -> str:
    return _table(pd.read_csv(_first_existing(results_dir, (name,), "model report input")))


def generate_report(dataset_dir: Path, results_dir: Path, output_html: Path) -> None:
    """Render baseline or Complexity B/C1/C1-nested/C2 artifacts without modifying them."""
    if output_html.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_html}")
    dataset_paths = {
        "dataset_qc": _first_existing(dataset_dir, ("dataset_qc_summary.json", "complexity_dataset_qc_summary.json"), "dataset QC summary"),
        "counts_participant": _first_existing(dataset_dir, ("counts_per_participant.csv",), "participant counts"),
        "counts_site": _first_existing(dataset_dir, ("counts_per_site.csv",), "site counts"),
        "quality": _first_existing(dataset_dir, ("feature_quality.csv", "complexity_feature_quality.csv"), "feature quality"),
        "distributions": _first_existing(dataset_dir, ("feature_distributions.csv", "complexity_feature_distributions.csv"), "feature distributions"),
        "duplicates": _first_existing(dataset_dir, ("exact_duplicate_features.csv", "complexity_exact_duplicate_features.csv"), "duplicate-feature audit"),
        "correlations": _first_existing(dataset_dir, ("high_correlations.csv", "complexity_high_correlations.csv"), "correlation audit"),
    }
    # Results are published as ``<table name>.csv`` by the evaluators, so key them
    # by stem: the HTML below indexes ``tables`` by table name, never by filename.
    result_paths = {
        Path(name).stem: _first_existing(results_dir, (name,), "model report input")
        for name in RESULT_ARTIFACTS
    }
    dataset_qc = json.loads(dataset_paths["dataset_qc"].read_text(encoding="utf-8"))
    model_run = json.loads(result_paths["model_run_summary"].read_text(encoding="utf-8"))
    tables = {
        name: pd.read_csv(path) for name, path in {**dataset_paths, **result_paths}.items()
        if path.suffix == ".csv"
    }
    missing_tables = [name for name in RENDERED_TABLES if name not in tables]
    if missing_tables:
        raise KeyError(f"report tables were loaded but not registered for rendering: {missing_tables}")
    counts = pd.DataFrame([{"measure": key, "observed": value} for key, value in dataset_qc["counts"].items()])
    stage = str(model_run.get("stage", ""))
    is_c1_nested = stage.endswith("_c1_nested")
    is_c1 = stage.endswith("_c1_mrmr")
    is_c2 = stage.endswith("_c2")
    model_name = (
        "Complexity C1-nested (nested mRMR + L2)" if is_c1_nested
        else "Complexity C1 (mRMR + L2)" if is_c1
        else "Complexity C2 (Elastic Net)" if is_c2
        else "Complexity B (L2)" if "complexity" in stage
        else "Logistic baseline"
    )
    extra_sections = ""
    if is_c1_nested:
        selection = model_run.get("feature_selection_details", {})
        outer_cv = model_run.get("outer_cv", {})
        inner_cv = model_run.get("inner_cv", {})
        protocol = pd.DataFrame(
            [
                {"setting": "N grid", "value": ", ".join(str(candidate) for candidate in selection.get("n_grid", []))},
                {"setting": "Outer CV", "value": f"{outer_cv.get('class', 'StratifiedGroupKFold')}, {outer_cv.get('n_splits', '?')} folds, grouped by {outer_cv.get('groups', 'participant_id')}, random_state={outer_cv.get('random_state', '?')}"},
                {"setting": "Inner CV", "value": f"{inner_cv.get('class', 'StratifiedGroupKFold')}, {inner_cv.get('n_splits', '?')} folds inside {inner_cv.get('scope', 'each outer training fold')}, random_state={inner_cv.get('random_state', '?')}"},
                {"setting": "Tuning metric", "value": str(model_run.get("tuning_metric", "average_precision"))},
                {"setting": "Tie-breaking", "value": str(model_run.get("tie_breaking", ""))},
                {"setting": "Selected N by outer fold", "value": ", ".join(f"fold {fold}: N={value}" for fold, value in sorted((selection.get("selected_n_by_outer_fold") or {}).items()))},
                {"setting": "N chosen from outer performance", "value": str(model_run.get("final_n_chosen_from_outer_performance", False))},
            ]
        )
        extra_sections = f"""<h2>C1-nested selection of N</h2>
<p class="note">N is selected inside each outer training fold only. Outer folds may select different N, and the out-of-fold predictions below are the unbiased evaluation of the whole procedure; N is never compared on outer out-of-fold performance.</p>
{_table(protocol)}
<h3>Selected N by outer fold</h3>{_result_table(results_dir, "mrmr_selected_n_by_outer_fold.csv")}
<h3>Inner performance for each N</h3>{_result_table(results_dir, "mrmr_nested_summary.csv")}
<h3>Inner tuning trace</h3>{_result_table(results_dir, "mrmr_nested_tuning.csv")}
<h3>Final selected predictors by outer fold</h3>{_result_table(results_dir, "selected_features_by_fold.csv")}
<h3>Selection frequencies</h3>{_result_table(results_dir, "selection_frequencies.csv")}"""
    elif is_c1:
        extra_sections = f"""<h2>C1 feature selection</h2>
<h3>Selected features by outer fold</h3>{_result_table(results_dir, "selected_features_by_fold.csv")}
<h3>Selection frequencies</h3>{_result_table(results_dir, "selection_frequencies.csv")}"""
    elif is_c2:
        extra_sections = f"""<h2>C2 nested Elastic Net tuning</h2>
<h3>Best C and l1_ratio by outer fold</h3>{_result_table(results_dir, "elasticnet_hyperparameters_by_fold.csv")}
<h3>Inner cross-validation results</h3>{_result_table(results_dir, "elasticnet_inner_cv_results.csv")}
<h3>Nonzero feature frequencies</h3>{_result_table(results_dir, "elasticnet_feature_stability.csv")}"""
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>OMNI iEEG {html.escape(model_name)} report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;color:#1f2937}}table{{border-collapse:collapse;margin:1rem 0 2rem;font-size:.86rem;width:100%}}th,td{{border:1px solid #d1d5db;padding:.35rem .5rem;text-align:left}}th{{background:#eef2ff}}code{{background:#f3f4f6;padding:.1rem .25rem}}.note{{background:#fffbeb;border-left:4px solid #f59e0b;padding:.8rem}}</style>
</head><body><h1>{html.escape(model_name)} — {html.escape(str(dataset_qc.get('task', 'unknown')))} task</h1>
<p>Generated from fixed out-of-fold predictions. Predictors: <code>{html.escape(str(model_run.get('n_predictors', '?')))}</code>. Git commit: <code>{html.escape(str(model_run.get('git_commit', 'unavailable')))}</code>.</p>
<p class="note">Channels are final observations; valid epochs are pooled before aggregation. All threshold metrics use the frozen pathological-score threshold of 0.5.</p>
<h2>Dataset and cohort summary</h2>{_table(counts)}<h3>Per site</h3>{_table(tables['counts_site'])}<h3>Per participant</h3>{_table(tables['counts_participant'])}
<h2>OOF performance</h2>{_table(tables['overall_site_metrics'])}<h3>Participant-level metric summary</h3>{_table(tables['participant_metric_summary'])}
<h3>Participant-cluster bootstrap 95% percentile intervals</h3>{_table(tables['bootstrap_confidence_intervals'])}
<h2>Fold metrics and convergence</h2><h3>Fold audit</h3>{_table(tables['fold_audit'])}<h3>Fold metrics</h3>{_table(tables['fold_metrics'])}<h3>Convergence</h3>{_table(tables['convergence'])}
<h2>Coefficients</h2>{_table(tables['model_coefficients'])}
{extra_sections}
<h2>Feature QC</h2><h3>Constant and near-constant predictors</h3>{_table(tables['quality'])}<h3>Exact duplicated predictors</h3>{_table(tables['duplicates'])}<h3>Highly correlated predictors</h3>{_table(tables['correlations'])}<h3>Feature distributions</h3>{_table(tables['distributions'])}
</body></html>\n"""
    atomic_text(document, output_html)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    generate_report(args.dataset_dir.resolve(), args.results_dir.resolve(), args.output_html.resolve())
    print(args.output_html.resolve())


if __name__ == "__main__":
    main()
