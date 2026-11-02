#!/usr/bin/env python3
"""Freeze Detroit's feature manifest using the HUP/UCLA feature contract.

This is intentionally a thin adapter: it reuses the same manifest ordering,
preprocessed-artifact resolution, and one-derivative-root-per-family guards as
HUP/UCLA while constraining the cohort to the ``detroit`` dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.features.build_hup_ucla_feature_manifest import (
    FEATURE_FAMILIES,
    build_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--preprocessed-root", type=Path, action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    for family in FEATURE_FAMILIES:
        parser.add_argument(
            f"--{family.replace('_', '-')}-derivatives-path", type=Path, required=True
        )
    parser.add_argument("--channel-labels-csv", type=Path, required=True)
    args = parser.parse_args()
    index, configs = build_manifest(
        args.cohort_csv.resolve(),
        [path.resolve() for path in args.preprocessed_root],
        args.output_directory,
        {family: getattr(args, f"{family}_derivatives_path") for family in FEATURE_FAMILIES},
        args.channel_labels_csv.resolve(),
        sites=("openieeg",),
        cohort_name="detroit",
    )
    print(f"Detroit feature index: {index}")
    for family in FEATURE_FAMILIES:
        print(f"{family} NeuroDAGs config: {configs[family]}")


if __name__ == "__main__":
    main()
