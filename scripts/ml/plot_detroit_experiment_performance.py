"""Render the published Detroit ML comparison figures from frozen result files.

This is a Detroit configuration of the HUP/UCLA performance renderers, not a
second plotting framework.  It deliberately uses one ``Detroit`` scope only:
there is no duplicated Overall panel because all Detroit OOF observations are
already that one cohort.  The command is read-only and never fits, retunes, or
overwrites any ML result.
"""

from __future__ import annotations

import argparse
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from scripts.ml import plot_experiment_performance as fold_plots
from scripts.ml import plot_experiment_performance_oof as oof_plots


DETROIT_TASK = "detroit"
DEFAULT_ANALYSIS_ROOT = Path("/CONFIGURE_SCRATCH/omni_openieeg_1khz_v1")
DEFAULT_OUTPUT_DIR = DEFAULT_ANALYSIS_ROOT / "ml_performance_figures_detroit_v1"

# Detroit comparison figures use the published logistic experiment order.
# EXP5-OPT is a second decision rule for the tuned experiment, not an
# additional fitted model.  Frozen Random Forest results deliberately remain
# on disk but are not loaded or visualised by this comparison renderer.
DETROIT_EXPERIMENTS = (
    fold_plots.Experiment(
        "exp1_baseline", "EXP1", "EXP1 - Conventional baseline", "Conventional baseline",
        ("non_nested",), "ml_logistic_baseline_detroit_v1/results",
    ),
    fold_plots.Experiment(
        "exp2_complexity_all", "EXP2", "EXP2 - Complexity, no selection",
        "Complexity (no selection)", ("non_nested",),
        "ml_logistic_complexity_detroit_v1/b/results",
    ),
    fold_plots.Experiment(
        "exp3a_mrmr_fixed_n8", "EXP3A", "EXP3A - mRMR fixed N=8", "mRMR fixed N=8",
        ("non_nested",), "ml_logistic_complexity_detroit_v1/c1/results",
    ),
    fold_plots.Experiment(
        "exp3b_mrmr_nested", "EXP3B", "EXP3B - Nested mRMR", "Nested mRMR",
        ("nested", "threshold_optimized"),
        "ml_logistic_complexity_c1_nested_detroit_v1/results",
    ),
    fold_plots.Experiment(
        "exp4_elasticnet_nested", "EXP4", "EXP4 - Nested Elastic Net", "Nested Elastic Net",
        ("nested", "threshold_optimized"), "ml_logistic_complexity_detroit_v1/c2/results",
    ),
    fold_plots.Experiment(
        "exp5_mrmr_c_tuned", "EXP5",
        "EXP5 - Tuned nested mRMR + tuned C (threshold 0.5)", "Tuned nested mRMR + tuned C",
        ("nested", "threshold_optimized"),
        "ml_logistic_complexity_c1_nested_tuned_detroit_v1/results",
    ),
    fold_plots.Experiment(
        "exp5_mrmr_c_tuned_optimized_threshold", "EXP5-OPT",
        "EXP5 - Tuned nested mRMR + tuned C (optimized threshold)",
        "Tuned nested mRMR + tuned C", ("threshold_optimized",),
        "ml_logistic_complexity_c1_nested_tuned_detroit_v1/results",
        label_column=fold_plots.OPTIMIZED_LABEL_COLUMN,
        threshold_column=fold_plots.OPTIMIZED_THRESHOLD_COLUMN,
        threshold_mode=fold_plots.OPTIMIZED_THRESHOLD_MODE,
    ),
)
# ``overall`` is the shared renderer's pooled key.  Its displayed label is
# Detroit and it is the only scope, so it cannot create an Overall duplicate.
DETROIT_SCOPES = (fold_plots.Scope("overall", "Detroit", None),)
DETROIT_SCOPE_COLORS = {"overall": "#3f6899"}
DETROIT_SCOPE_EDGE_COLORS = {"overall": "#2c496b"}
DETROIT_EXPERIMENT_COLORS = {
    **fold_plots.EXPERIMENT_COLORS,
}


@contextmanager
def _detroit_renderer_configuration() -> Iterator[None]:
    """Temporarily supply the shared renderers with the single-site cohort map.

    The established renderers keep their HUP/UCLA constants as module defaults.
    Rebinding them only in this short-lived, standalone process keeps that CLI
    byte-for-byte behaviour intact while avoiding a copied figure implementation.
    """
    replacements = {
        fold_plots: {
            "EXPERIMENTS": DETROIT_EXPERIMENTS,
            "SCOPES": DETROIT_SCOPES,
            "SITE_VALUES": ("detroit",),
            "SCOPE_COLORS": DETROIT_SCOPE_COLORS,
            "SCOPE_EDGE_COLORS": DETROIT_SCOPE_EDGE_COLORS,
            "EXPERIMENT_COLORS": DETROIT_EXPERIMENT_COLORS,
            "resolve_task": lambda task: _resolve_detroit_task(task),
        },
        oof_plots: {
            "EXPERIMENTS": DETROIT_EXPERIMENTS,
            "SCOPES": DETROIT_SCOPES,
            "SCOPE_COLORS": DETROIT_SCOPE_COLORS,
            "SCOPE_EDGE_COLORS": DETROIT_SCOPE_EDGE_COLORS,
            "EXPERIMENT_COLORS": DETROIT_EXPERIMENT_COLORS,
            "DISTRIBUTION_NOTE": (
                "Pooled out-of-fold predicted pathological probability over all Detroit channels, "
                "normalised within each true class so the two classes stay comparable despite "
                "the class imbalance."
            ),
            "resolve_task": lambda task: _resolve_detroit_task(task),
        },
    }
    originals = {
        module: {name: getattr(module, name) for name in values}
        for module, values in replacements.items()
    }
    try:
        for module, values in replacements.items():
            for name, value in values.items():
                setattr(module, name, value)
        yield
    finally:
        for module, values in originals.items():
            for name, value in values.items():
                setattr(module, name, value)


def _resolve_detroit_task(task: object) -> str:
    if str(task).strip().lower() != DETROIT_TASK:
        raise ValueError(f"Detroit plotting requires task={DETROIT_TASK!r}; got {task!r}")
    return DETROIT_TASK


def _result_directories(analysis_root: Path) -> dict[str, Path]:
    return {
        experiment.key: analysis_root / experiment.template
        for experiment in DETROIT_EXPERIMENTS
    }


def validate_detroit_comparability(analysis_root: Path) -> list[int]:
    """Require identical Detroit participant/channel observations and fold IDs."""
    tables: dict[Path, pd.DataFrame] = {}
    loaded: dict[str, pd.DataFrame] = {}
    for key, directory in _result_directories(analysis_root).items():
        if directory not in tables:
            tables[directory] = fold_plots.load_oof_predictions(directory)
        loaded[key] = tables[directory]
    folds = fold_plots.assert_shared_evaluation(loaded)
    reference = next(iter(loaded.values()))
    sites = set(reference["dataset"])
    if sites != {"detroit"}:
        raise AssertionError(f"Detroit comparison requires only dataset='detroit', found {sorted(sites)}")
    print(
        "[validation] Detroit comparability passed: "
        f"{reference['participant_id'].nunique()} participants, {len(reference)} channels, "
        f"and identical participant/channel observations, labels and outer-fold assignments "
        f"across {len(DETROIT_EXPERIMENTS)} decision rules (folds {folds})."
    )
    return folds


def plot_detroit_experiments(analysis_root: Path, output_dir: Path) -> list[Path]:
    analysis_root = analysis_root.resolve()
    output_dir = output_dir.resolve()
    with _detroit_renderer_configuration():
        validate_detroit_comparability(analysis_root)
        written = fold_plots.plot_experiments(DETROIT_TASK, analysis_root, output_dir)
        written.extend(oof_plots.plot_oof_experiments(DETROIT_TASK, analysis_root, output_dir))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only the requested Detroit figure directory; ML result directories are never touched",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.name != DEFAULT_OUTPUT_DIR.name:
        raise ValueError(
            "Detroit plotting output must use the Detroit figure-directory name "
            f"{DEFAULT_OUTPUT_DIR.name!r}; got {output_dir.name!r}"
        )
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"refusing to overwrite an existing Detroit figure directory: {output_dir}; "
                "rerun with --overwrite to replace only these plotting outputs"
            )
        shutil.rmtree(output_dir)
    for path in plot_detroit_experiments(args.analysis_root, output_dir):
        print(path)


if __name__ == "__main__":
    main()
