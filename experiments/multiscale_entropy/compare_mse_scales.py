#!/usr/bin/env python3
"""Standalone MSE scale-range comparison: scales 1..20 vs 1..40 vs 1..50.

Exploratory only. Reads real preprocessed OMNI channel-epochs (8 s, 1000 Hz),
computes composite multiscale entropy with the *same* NeuroKit call as
``scripts/features/complexity.py`` (``method="MSEn"``, ``dimension=3``,
``tolerance="sd"``) for each candidate scale range, and reports finite
proportion, runtime, value distributions and Spearman correlations.

No labels are used and no production code is modified: the only thing imported
from the repository is the shared channel-epoch validity policy, so exactly the
channel-epochs that feature extraction would accept are the ones compared here.

Sampling is fully deterministic: recordings are taken at evenly spaced positions
in the site-sorted frozen manifest, and channel-epochs are drawn with a seed
derived from the recording path (so a recording's sample does not depend on how
many other recordings were requested).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import mne
import numpy as np
import scipy.integrate
from scipy.stats import spearmanr

# Same numpy 2.0 shim as complexity.py, so neurokit behaves identically.
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", scipy.integrate.trapezoid)

import neurokit2 as nk

REPOSITORY = Path(__file__).resolve().parents[2]
SHARED_SCRIPT = REPOSITORY / "scripts" / "features" / "relative_bandpower.py"

# Fixed by the current preprocessing runtime (8 s epochs at 1000 Hz).
EXPECTED_SAMPLES_PER_EPOCH = 8000
EXPECTED_SFREQ_HZ = 1000.0

# Candidate scale ranges. NeuroKit's _get_scales() maps an int S to
# np.arange(1, S + 1), i.e. exactly scales 1..S.
SCALE_RANGES = (20, 40, 50)

DIMENSION = 3
TOLERANCE = "sd"
METHOD = "MSEn"


def _shared_module() -> Any:
    """Load the single shared channel-epoch validity policy without side effects."""
    spec = importlib.util.spec_from_file_location("omni_shared_policy", SHARED_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared feature infrastructure: {SHARED_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["omni_shared_policy"] = module
    spec.loader.exec_module(module)
    return module


def select_recordings(manifest: Path, per_site: int) -> list[dict[str, str]]:
    """Return evenly spaced recordings per site from the site-sorted manifest."""
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream)]
    selected: list[dict[str, str]] = []
    for site in ("hup", "ucla"):
        site_rows = sorted(
            (row for row in rows if str(row.get("dataset", "")).strip().lower() == site),
            key=lambda row: row["source_fif_path"],
        )
        if not site_rows:
            continue
        take = min(per_site, len(site_rows))
        positions = np.unique(np.round(np.linspace(0, len(site_rows) - 1, take)).astype(int))
        selected.extend(site_rows[position] for position in positions)
    return selected


def sample_channel_epochs(
    fif_path: Path, n_channel_epochs: int, shared: Any
) -> list[tuple[str, int, np.ndarray]]:
    """Return up to ``n_channel_epochs`` valid (channel, epoch_index, signal) triples."""
    epochs = mne.read_epochs(fif_path, preload=False, verbose="ERROR")
    sfreq = float(epochs.info["sfreq"])
    if abs(sfreq - EXPECTED_SFREQ_HZ) > 1e-9 or len(epochs.times) != EXPECTED_SAMPLES_PER_EPOCH:
        raise ValueError(f"unexpected sampling/epoch geometry in {fif_path}: {sfreq} Hz, {len(epochs.times)} samples")

    bads = set(epochs.info.get("bads", []))
    picks: list[tuple[int, str]] = []
    for index, (name, channel_type) in enumerate(zip(epochs.ch_names, epochs.get_channel_types())):
        upper = channel_type.upper()
        if name in bads or upper in shared.EXCLUDED_CHANNEL_TYPES:
            continue
        if upper in shared.VALID_IEEG_TYPES:
            picks.append((index, name))
    if not picks:
        return []

    grid = [(epoch_index, pick) for epoch_index in range(len(epochs)) for pick in picks]
    # Seed from the recording path so each recording's draw is independent of
    # the surrounding selection and reproducible across runs.
    seed = int.from_bytes(fif_path.name.encode("utf-8"), "little") % (2**32)
    order = np.random.default_rng(seed).permutation(len(grid))

    sampled: list[tuple[str, int, np.ndarray]] = []
    for position in order:
        if len(sampled) >= n_channel_epochs:
            break
        epoch_index, (channel_index, channel_name) = grid[int(position)]
        signal = np.asarray(
            epochs.get_data(item=epoch_index, picks=[channel_index], verbose="ERROR")
        ).ravel()
        if shared.channel_epoch_rejection_reason(signal, EXPECTED_SAMPLES_PER_EPOCH):
            continue
        sampled.append((channel_name, epoch_index, signal.astype(np.float64)))
    return sampled


def multiscale_entropy(signal: np.ndarray, max_scale: int) -> tuple[float, int, int, float]:
    """Run the production MSEn call restricted to scales 1..max_scale."""
    start = time.perf_counter()
    value, info = nk.entropy_multiscale(
        signal,
        scale=max_scale,
        dimension=DIMENSION,
        tolerance=TOLERANCE,
        method=METHOD,
    )
    elapsed = time.perf_counter() - start
    per_scale = np.asarray(info["Value"], dtype=np.float64)
    return float(value), int(np.sum(np.isfinite(per_scale))), int(per_scale.size), elapsed


def _distribution(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {}
    quartiles = np.percentile(finite, [25, 50, 75])
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan"),
        "min": float(np.min(finite)),
        "q25": float(quartiles[0]),
        "median": float(quartiles[1]),
        "q75": float(quartiles[2]),
        "max": float(np.max(finite)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-manifest",
        type=Path,
        default=Path(
            "/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v2/manifests/"
            "hup_ucla_feature_neurodags_v1/hup_ucla_feature_neurodags_index.csv"
        ),
    )
    parser.add_argument("--recordings-per-site", type=int, default=3)
    parser.add_argument("--channel-epochs-per-recording", type=int, default=10)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()

    shared = _shared_module()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    recordings = select_recordings(arguments.index_manifest, arguments.recordings_per_site)
    print(f"Selected {len(recordings)} recordings from {arguments.index_manifest}", flush=True)

    rows: list[dict[str, Any]] = []
    for recording in recordings:
        fif_path = Path(recording["source_fif_path"])
        sampled = sample_channel_epochs(fif_path, arguments.channel_epochs_per_recording, shared)
        print(
            f"{recording['participant_id']} ({recording['dataset']}): "
            f"{len(sampled)} channel-epochs",
            flush=True,
        )
        for channel, epoch_index, signal in sampled:
            row: dict[str, Any] = {
                "participant_id": recording["participant_id"],
                "dataset": recording["dataset"],
                "source_file": str(fif_path),
                "channel": channel,
                "epoch_index": epoch_index,
            }
            for max_scale in SCALE_RANGES:
                value, n_finite, n_scales, elapsed = multiscale_entropy(signal, max_scale)
                row[f"mse_1_{max_scale}"] = value
                row[f"n_finite_scales_1_{max_scale}"] = n_finite
                row[f"n_scales_1_{max_scale}"] = n_scales
                row[f"seconds_1_{max_scale}"] = elapsed
            rows.append(row)

    if not rows:
        print("No valid channel-epochs sampled.", file=sys.stderr)
        return 1

    per_epoch_csv = arguments.output_directory / "mse_scale_comparison_channel_epochs.csv"
    with per_epoch_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "index_manifest": str(arguments.index_manifest),
        "n_recordings": len(recordings),
        "n_channel_epochs": len(rows),
        "configuration": {
            "dimension": DIMENSION,
            "tolerance": TOLERANCE,
            "method": METHOD,
            "samples_per_epoch": EXPECTED_SAMPLES_PER_EPOCH,
            "sampling_frequency_hz": EXPECTED_SFREQ_HZ,
        },
        "neurokit2_version": nk.__version__,
        "scale_ranges": {},
        "spearman": {},
    }
    values = {
        max_scale: np.array([row[f"mse_1_{max_scale}"] for row in rows], dtype=np.float64)
        for max_scale in SCALE_RANGES
    }
    for max_scale in SCALE_RANGES:
        seconds = np.array([row[f"seconds_1_{max_scale}"] for row in rows], dtype=np.float64)
        finite_scales = np.array(
            [row[f"n_finite_scales_1_{max_scale}"] for row in rows], dtype=np.float64
        )
        summary["scale_ranges"][f"1_{max_scale}"] = {
            "finite_proportion": float(np.mean(np.isfinite(values[max_scale]))),
            "n_non_finite": int(np.sum(~np.isfinite(values[max_scale]))),
            "seconds_per_channel_epoch_mean": float(np.mean(seconds)),
            "seconds_per_channel_epoch_median": float(np.median(seconds)),
            "seconds_total": float(np.sum(seconds)),
            "finite_scales_mean": float(np.mean(finite_scales)),
            "distribution": _distribution(values[max_scale]),
        }
    for first_index, first in enumerate(SCALE_RANGES):
        for second in SCALE_RANGES[first_index + 1 :]:
            usable = np.isfinite(values[first]) & np.isfinite(values[second])
            if usable.sum() < 3:
                summary["spearman"][f"1_{first}__vs__1_{second}"] = None
                continue
            rho, p_value = spearmanr(values[first][usable], values[second][usable])
            summary["spearman"][f"1_{first}__vs__1_{second}"] = {
                "rho": float(rho),
                "p_value": float(p_value),
                "n": int(usable.sum()),
            }

    summary_json = arguments.output_directory / "mse_scale_comparison_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n" + json.dumps(summary, indent=2), flush=True)
    print(f"\nPer-channel-epoch table: {per_epoch_csv}")
    print(f"Summary: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
