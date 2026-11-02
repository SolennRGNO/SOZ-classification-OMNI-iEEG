"""Dispatch Detroit datasets to the unchanged HUP/UCLA ML evaluators."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.ml.logistic_baseline import run_baseline
from scripts.ml.logistic_complexity import run_complexity
from scripts.ml.logistic_complexity_tuned import run_complexity_tuned
from scripts.ml.random_forest_complexity import run_random_forest_complexity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("baseline", "b", "c1", "c1_nested", "c2", "c1_nested_tuned", "random_forest_nested"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--feature-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--n-selected-features", type=int)
    parser.add_argument("--minimum-tpr", type=float, default=0.50)
    parser.add_argument("--n-jobs", type=int)
    args = parser.parse_args()
    common = (args.dataset.resolve(), args.feature_list.resolve(), args.output_dir.resolve(), args.repository.resolve())
    if args.experiment == "baseline":
        result = run_baseline(*common)
    elif args.experiment in {"b", "c1", "c1_nested", "c2"}:
        result = run_complexity(*common, n_selected_features=args.n_selected_features, experiment=args.experiment)
    elif args.experiment == "c1_nested_tuned":
        result = run_complexity_tuned(*common, minimum_tpr=args.minimum_tpr)
    else:
        result = run_random_forest_complexity(*common, n_jobs=args.n_jobs)
    print(result)


if __name__ == "__main__":
    main()
