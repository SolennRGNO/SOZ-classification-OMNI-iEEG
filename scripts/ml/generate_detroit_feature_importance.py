"""Create Detroit Mantilla-style LR importance figures from completed CV artifacts.

This is intentionally a read-only post-processing step: it loads the stored
outer-training LogisticRegression coefficients and mRMR selections and never
loads a dataset, predictions, or an estimator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.ml.common import atomic_csv


N_OUTER_FOLDS = 5
DEFAULT_ANALYSIS_ROOT = Path("/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1")
DEFAULT_OUTPUT_DIR = DEFAULT_ANALYSIS_ROOT / "ml_feature_importance_detroit_v1"
DEFAULT_RESULTS = {
    "EXP2": DEFAULT_ANALYSIS_ROOT / "ml_logistic_complexity_detroit_v1/b/results",
    "EXP3B": DEFAULT_ANALYSIS_ROOT / "ml_logistic_complexity_c1_nested_detroit_v1/results",
    "EXP4": DEFAULT_ANALYSIS_ROOT / "ml_logistic_complexity_detroit_v1/c2/results",
    "EXP5": DEFAULT_ANALYSIS_ROOT / "ml_logistic_complexity_c1_nested_tuned_detroit_v1/results",
}


def _predictors(results_dir: Path) -> list[str]:
    summary_path = results_dir / "model_run_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    predictors = payload.get("predictor_order")
    if not isinstance(predictors, list) or len(predictors) != 42:
        raise ValueError(f"{summary_path}: expected exactly 42 predictor_order entries")
    if len(set(predictors)) != 42:
        raise ValueError(f"{summary_path}: predictor_order contains duplicates")
    return [str(feature) for feature in predictors]


def _coefficients(results_dir: Path, predictors: list[str]) -> pd.DataFrame:
    path = results_dir / "model_coefficients.csv"
    frame = pd.read_csv(path)
    required = {"fold_id", "feature", "coefficient"}
    if not required <= set(frame):
        raise ValueError(f"{path}: missing {sorted(required - set(frame))}")
    frame = frame.loc[:, ["fold_id", "feature", "coefficient"]].copy()
    frame["fold_id"] = pd.to_numeric(frame["fold_id"], errors="raise").astype(int)
    frame["coefficient"] = pd.to_numeric(frame["coefficient"], errors="raise")
    if set(frame["fold_id"]) != set(range(1, N_OUTER_FOLDS + 1)):
        raise ValueError(f"{path}: expected exactly outer folds 1--{N_OUTER_FOLDS}")
    if (
        not set(frame["feature"]) <= set(predictors)
        or frame.duplicated(["fold_id", "feature"]).any()
    ):
        raise ValueError(f"{path}: invalid or duplicate coefficient rows")
    return frame


def _sign_stability(values: pd.Series) -> float:
    """Largest same-sign proportion among non-zero fold coefficients; zero if all zero."""
    signs = np.sign(values.to_numpy(dtype=float))
    nonzero = signs[signs != 0]
    if not len(nonzero):
        return 0.0
    return float(max((nonzero > 0).mean(), (nonzero < 0).mean()))


def _summary(frame: pd.DataFrame, *, conditional: bool) -> pd.DataFrame:
    grouped = frame.groupby("feature", sort=False)["coefficient"]
    summary = grouped.agg(
        mean_coefficient="mean",
        median_coefficient="median",
        mean_abs_coefficient=lambda values: values.abs().mean(),
        median_abs_coefficient=lambda values: values.abs().median(),
        min_abs_coefficient=lambda values: values.abs().min(),
        max_abs_coefficient=lambda values: values.abs().max(),
        sign_stability=_sign_stability,
    ).reset_index()
    if conditional:
        summary = summary.rename(columns={
            "mean_coefficient": "conditional_mean_coefficient",
            "median_coefficient": "conditional_median_coefficient",
            "mean_abs_coefficient": "conditional_mean_abs_coefficient",
            "median_abs_coefficient": "conditional_median_abs_coefficient",
            "min_abs_coefficient": "conditional_min_abs_coefficient",
            "max_abs_coefficient": "conditional_max_abs_coefficient",
            "sign_stability": "conditional_sign_stability",
        })
    return summary


def _descending_rank(table: pd.DataFrame, column: str, name: str) -> pd.DataFrame:
    table[name] = table[column].rank(method="min", ascending=False).astype(int)
    return table


def _all_fold_table(coefficients: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    expected = pd.MultiIndex.from_product(
        [predictors, range(1, N_OUTER_FOLDS + 1)], names=["feature", "fold"]
    ).to_frame(index=False)
    table = expected.merge(
        coefficients.rename(columns={"fold_id": "fold"}), on=["feature", "fold"], how="left",
        validate="one_to_one",
    )
    if table["coefficient"].isna().any():
        raise ValueError("all-feature experiment is missing an outer-fold coefficient")
    table["abs_coefficient"] = table["coefficient"].abs()
    summary = _summary(table, conditional=False)
    summary = _descending_rank(
        summary, "median_abs_coefficient", "rank_median_abs_coefficient"
    )
    summary = _descending_rank(
        summary, "mean_abs_coefficient", "rank_mean_abs_coefficient"
    )
    return table.merge(summary, on="feature", validate="many_to_one")


def _mrmr_table(
    results_dir: Path, coefficients: pd.DataFrame, predictors: list[str]
) -> pd.DataFrame:
    selected_path = results_dir / "selected_features_by_fold.csv"
    selected = pd.read_csv(selected_path)
    if not {"fold_id", "feature"} <= set(selected):
        raise ValueError(f"{selected_path}: needs fold_id and feature")
    selected = selected.loc[:, ["fold_id", "feature"]].drop_duplicates().copy()
    selected["fold_id"] = pd.to_numeric(selected["fold_id"], errors="raise").astype(int)
    if not set(selected["feature"]) <= set(predictors):
        raise ValueError(f"{selected_path}: selected feature outside frozen schema")
    expected = pd.MultiIndex.from_product(
        [predictors, range(1, N_OUTER_FOLDS + 1)], names=["feature", "fold"]
    ).to_frame(index=False)
    table = expected.merge(selected.rename(columns={"fold_id": "fold"}).assign(selected=True),
                           on=["feature", "fold"], how="left", validate="one_to_one")
    table["selected"] = table["selected"].fillna(False).astype(bool)
    table = table.merge(
        coefficients.rename(columns={"fold_id": "fold"}),
        on=["feature", "fold"],
        how="left",
        validate="one_to_one",
    )
    if (
        not table.loc[table["selected"], "coefficient"].notna().all()
        or table.loc[~table["selected"], "coefficient"].notna().any()
    ):
        raise ValueError("mRMR coefficient rows and selected_features_by_fold.csv disagree")
    table["abs_coefficient"] = table["coefficient"].abs()
    table["effective_coefficient"] = table["coefficient"].fillna(0.0)
    table["effective_abs_coefficient"] = table["effective_coefficient"].abs()
    counts = (
        table.groupby("feature", sort=False)["selected"]
        .sum()
        .astype(int)
        .rename("selection_count")
    )
    table = table.merge(counts, on="feature", validate="many_to_one")
    table["selection_frequency"] = table["selection_count"] / N_OUTER_FOLDS
    conditional = _summary(table.loc[table["selected"]], conditional=True)
    effective = _summary(
        table.assign(coefficient=table["effective_coefficient"]), conditional=False
    ).rename(
        columns={
            "mean_coefficient": "effective_mean_coefficient",
            "median_coefficient": "effective_median_coefficient",
            "mean_abs_coefficient": "effective_mean_abs_coefficient",
            "median_abs_coefficient": "effective_median_abs_coefficient",
            "min_abs_coefficient": "effective_min_abs_coefficient",
            "max_abs_coefficient": "effective_max_abs_coefficient",
            "sign_stability": "effective_sign_stability",
        }
    )
    effective = _descending_rank(
        effective,
        "effective_mean_abs_coefficient",
        "rank_effective_mean_abs_coefficient",
    )
    return table.merge(conditional, on="feature", how="left", validate="many_to_one").merge(
        effective, on="feature", validate="many_to_one"
    )


def _elasticnet_table(coefficients: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    table = _all_fold_table(coefficients, predictors)
    table["nonzero"] = table["coefficient"].ne(0.0)
    counts = (
        table.groupby("feature", sort=False)["nonzero"]
        .sum()
        .astype(int)
        .rename("nonzero_count")
    )
    table = table.merge(counts, on="feature", validate="many_to_one")
    table["nonzero_frequency"] = table["nonzero_count"] / N_OUTER_FOLDS
    return table


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _ordered(table: pd.DataFrame, value: str) -> list[str]:
    return table.groupby("feature", sort=False)[value].first().sort_values().index.tolist()


def _coefficient_boxplot(table: pd.DataFrame, output_dir: Path, stem: str, title: str) -> None:
    order = _ordered(table, "median_abs_coefficient")
    values = [
        table.loc[table["feature"].eq(feature), "coefficient"].to_numpy()
        for feature in order
    ]
    figure, axis = plt.subplots(figsize=(11, max(8, len(order) * 0.24)))
    axis.boxplot(values, vert=False, tick_labels=order, showfliers=False, patch_artist=True,
                 boxprops={"facecolor": "#82b5d4"}, medianprops={"color": "#111111"})
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set_xlabel("Standardized logistic-regression coefficient (outer training fit)")
    axis.set_title(title)
    _save_figure(figure, output_dir, stem)


def _mrmr_plots(table: pd.DataFrame, output_dir: Path, experiment: str) -> None:
    summary = table.drop_duplicates("feature").sort_values("selection_frequency")
    figure, axis = plt.subplots(figsize=(11, max(8, len(summary) * 0.24)))
    axis.barh(summary["feature"], summary["selection_frequency"], color="#c44e52")
    axis.set(
        xlim=(0, 1),
        xlabel="Outer-fold selection frequency",
        title=f"{experiment}: mRMR selection frequency",
    )
    _save_figure(figure, output_dir, f"{experiment.lower()}_selection_frequency")

    order = summary.sort_values("effective_mean_abs_coefficient")["feature"].tolist()
    values = [
        table.loc[table["feature"].eq(feature), "effective_abs_coefficient"].to_numpy()
        for feature in order
    ]
    figure, axis = plt.subplots(figsize=(11, max(8, len(order) * 0.24)))
    axis.boxplot(values, vert=False, tick_labels=order, showfliers=False, patch_artist=True,
                 boxprops={"facecolor": "#c44e52"}, medianprops={"color": "#111111"})
    axis.set_xlabel("Effective |coefficient| across five folds (0 when not selected)")
    axis.set_title(f"{experiment}: effective coefficient importance")
    _save_figure(figure, output_dir, f"{experiment.lower()}_effective_abs_coefficient")

    figure, axis = plt.subplots(figsize=(10, max(8, len(summary) * 0.24)))
    axis.barh(
        summary["feature"],
        summary["selection_frequency"],
        color="#c44e52",
        label="Selection frequency",
    )
    twin = axis.twiny()
    twin.scatter(
        summary["effective_mean_abs_coefficient"],
        summary["feature"],
        color="#222222",
        s=18,
        label="Effective mean |coefficient|",
    )
    axis.set(
        xlim=(0, 1),
        xlabel="Selection frequency",
        title=f"{experiment}: selection and effective magnitude",
    )
    twin.set_xlabel("Effective mean |coefficient|")
    _save_figure(figure, output_dir, f"{experiment.lower()}_combined_importance")


def generate(output_dir: Path, result_dirs: dict[str, Path]) -> None:
    if any("random_forest" in str(path).lower() for path in result_dirs.values()):
        raise ValueError("Random Forest results are explicitly excluded from this analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    exp2_predictors = _predictors(result_dirs["EXP2"])
    for experiment, results_dir in result_dirs.items():
        predictors = _predictors(results_dir)
        if predictors != exp2_predictors:
            raise ValueError(f"{experiment}: predictor order differs from EXP2")
        coefficients = _coefficients(results_dir, predictors)
        if experiment == "EXP2":
            table = _all_fold_table(coefficients, predictors)
            _coefficient_boxplot(
                table,
                output_dir,
                "exp2_coefficient_distribution",
                "EXP2: coefficient distributions",
            )
        elif experiment == "EXP4":
            table = _elasticnet_table(coefficients, predictors)
            _coefficient_boxplot(
                table,
                output_dir,
                "exp4_coefficient_distribution",
                "EXP4: Elastic Net coefficient distributions",
            )
        else:
            table = _mrmr_table(results_dir, coefficients, predictors)
            _mrmr_plots(table, output_dir, experiment)
        atomic_csv(table, output_dir / f"{experiment.lower()}_feature_importance.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    for experiment in DEFAULT_RESULTS:
        parser.add_argument(
            f"--{experiment.lower()}-results",
            type=Path,
            default=DEFAULT_RESULTS[experiment],
        )
    args = parser.parse_args()
    generate(args.output_dir.resolve(), {
        experiment: getattr(args, f"{experiment.lower()}_results").resolve()
        for experiment in DEFAULT_RESULTS
    })


if __name__ == "__main__":
    main()
