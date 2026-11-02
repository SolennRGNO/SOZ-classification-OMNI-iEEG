#!/usr/bin/env python3
"""Raw, pre-preprocessing spectral / line-noise inspection of OpenIEEG recordings.

This is a **diagnostic** stage. It answers one question -- which narrow-band
electrical frequencies are actually present in OpenIEEG before anything is done
to the data -- so that notch frequencies can later be chosen from evidence
rather than assumed. It writes no preprocessed derivative and makes no
preprocessing decision.

Scientific guarantees
---------------------
* The PSD is computed from raw EDF samples. The recording is opened lazily
  (``preload=False``) and the only operation applied to the samples is Welch's
  method itself: no notch, no high-pass, low-pass or band-pass filter, no
  re-referencing and no resampling. The shared implementation is
  :func:`run_raw_line_noise_qc.analyze_raw_recording`, which contains no filter
  or resample call at all.
* The sampling rate is read from each recording's own EDF header. Recordings
  with different rates are analysed at their own rate; when the inventory also
  carries a rate the two must agree or the recording is reported as failed.
* Nothing at or above a recording's Nyquist frequency is ever evaluated. The
  Welch grid is truncated at ``sfreq/2`` minus one native bin, per recording,
  and candidate frequencies above that bound are recorded as ``not_evaluable``
  rather than silently dropped.
* 50/60 Hz harmonics are never assumed to be noise. Every candidate is reported
  with its measured prominence in dB above the local spectral background, and
  peaks are additionally discovered from the data so that non-mains narrow
  peaks are visible too.
* A recording that cannot be analysed is written to the failure table with its
  reason and counted in the report; it is never silently skipped.

Reused from the existing HUP/UCLA line-noise QC without modification: the Welch
settings, the channels.tsv channel-type selection, the median channel
reduction, the local-background prominence definition, and the 50/60 Hz
candidate grid.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load_shared_module(name: str, relative_path: str):
    """Import a sibling pipeline script that is not an importable package module."""
    path = REPOSITORY_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


QC = _load_shared_module(
    "run_raw_line_noise_qc", "scripts/preprocessing/run_raw_line_noise_qc.py"
)
INVENTORY = _load_shared_module(
    "build_inventory", "scripts/preprocessing/build_inventory.py"
)

DATASET = "openieeg"
# Diagnostic scope: read every recording at its own rate. The >=1 kHz admission
# rule of the preprocessing cohort deliberately does not apply here.
MINIMUM_SAMPLING_FREQUENCY_HZ = 0.0
LOWEST_ANALYSED_FREQUENCY_HZ = 0.5
# A candidate counts as "near" a harmonic when it is within this many Hz of it.
HARMONIC_TOLERANCE_HZ = 1.0
# Prominence needed to flag a *mains-grid* candidate. This mirrors the HUP/UCLA
# stage's REVIEW_FLAG_THRESHOLD_DB and, like it, only flags for review.
MAINS_REPORT_THRESHOLD_DB = QC.REVIEW_FLAG_THRESHOLD_DB
# Prominence needed to *discover* an unknown peak. This must be stricter than the
# mains threshold: the mains grid tests ~20 known frequencies, whereas discovery
# scans every Welch bin, so a 3 dB rule there would report ordinary noise-floor
# fluctuation as narrow-band peaks.
DISCOVERY_THRESHOLD_DB = 6.0
# A discovered peak must be narrow; wide bumps are physiology, not mains.
PEAK_HALF_WIDTH_HZ = 1.0
MAINS_FAMILIES = ("50_hz", "60_hz")
# A notch filter applied upstream leaves the mirror image of a peak: a sharp dip
# well *below* the local background at a mains harmonic. Raw data should never
# show this, so it is checked and reported rather than assumed absent.
UPSTREAM_NOTCH_DIP_DB = -3.0

LOGGER = logging.getLogger("openieeg_raw_spectral_qc")


def harmonics_up_to(base_frequency: float, nyquist_frequency_hz: float) -> list[float]:
    """Every harmonic of ``base_frequency`` strictly below Nyquist."""
    if base_frequency <= 0 or not np.isfinite(nyquist_frequency_hz):
        return []
    count = int(math.floor((nyquist_frequency_hz - 1e-9) / base_frequency))
    return [float(base_frequency * order) for order in range(1, count + 1)]


def nearest_harmonic_order(
    frequency: float, base_frequency: float, tolerance_hz: float = HARMONIC_TOLERANCE_HZ
) -> int | None:
    """Harmonic order of ``base_frequency`` within ``tolerance_hz``, else ``None``.

    A tolerance is used rather than exact integer equality because the Welch bin
    grid rarely lands exactly on a mains harmonic.
    """
    if base_frequency <= 0 or frequency <= 0:
        return None
    order = int(round(frequency / base_frequency))
    if order < 1:
        return None
    return order if abs(frequency - order * base_frequency) <= tolerance_hz else None


def local_background_db(
    frequencies_hz: np.ndarray,
    power: np.ndarray,
    frequency: float,
    *,
    inner_hz: float = QC.LOCAL_BACKGROUND_INNER_HZ,
    outer_hz: float = QC.LOCAL_BACKGROUND_OUTER_HZ,
) -> tuple[float, float, float]:
    """Power at ``frequency``, its local background, and the dB difference.

    The background is the median of a ring on both sides of the candidate --
    exactly the definition the HUP/UCLA stage uses -- so a broad physiological
    slope is not mistaken for a narrow electrical peak.
    """
    distance = np.abs(frequencies_hz - frequency)
    ring = (distance >= inner_hz) & (distance <= outer_hz)
    if not ring.any():
        return (math.nan, math.nan, math.nan)
    value = float(np.interp(frequency, frequencies_hz, power))
    background = float(np.nanmedian(power[ring]))
    if not (np.isfinite(value) and np.isfinite(background)) or background <= 0 or value <= 0:
        return (value, background, math.nan)
    return (value, background, float(10.0 * np.log10(value / background)))


def discover_narrow_peaks(
    frequencies_hz: np.ndarray,
    power: np.ndarray,
    *,
    threshold_db: float = DISCOVERY_THRESHOLD_DB,
    half_width_hz: float = PEAK_HALF_WIDTH_HZ,
) -> list[float]:
    """Frequencies of narrow local maxima that clear the local background.

    This finds the "any other strong narrow-band peak" cases; it is independent
    of the 50/60 Hz grid and makes no assumption about mains frequency.
    """
    if frequencies_hz.size < 3:
        return []
    resolution = float(np.median(np.diff(frequencies_hz))) if frequencies_hz.size > 1 else 0.0
    if resolution <= 0:
        return []
    span = max(1, int(round(half_width_hz / resolution)))
    found: list[float] = []
    for index in range(span, frequencies_hz.size - span):
        window = power[index - span : index + span + 1]
        if power[index] < np.nanmax(window):
            continue
        _, _, prominence = local_background_db(
            frequencies_hz, power, float(frequencies_hz[index])
        )
        if np.isfinite(prominence) and prominence >= threshold_db:
            found.append(float(frequencies_hz[index]))
    return found


def candidate_peak_table(
    spectrum: QC.RecordingSpectrum,
    *,
    site_code: str,
    tolerance_hz: float = HARMONIC_TOLERANCE_HZ,
    mains_threshold_db: float = MAINS_REPORT_THRESHOLD_DB,
    discovery_threshold_db: float = DISCOVERY_THRESHOLD_DB,
) -> pd.DataFrame:
    """Per-recording candidate peaks: the 50/60 Hz grids plus discovered peaks.

    The mains grid is the same one the HUP/UCLA stage uses
    (``QC.LINE_NOISE_CANDIDATES``). Grid points above this recording's Nyquist
    frequency are kept in the table with ``evaluable_below_nyquist = False`` and
    no metric, so a reader can tell "absent" from "not measurable at this
    sampling rate". Nothing above Nyquist is ever given a value.
    """
    frequencies = np.asarray(spectrum.frequencies_hz, dtype=float)
    power = np.asarray(spectrum.power, dtype=float)
    nyquist = spectrum.sampling_frequency_hz / 2.0
    highest_evaluable = float(frequencies[-1]) if frequencies.size else math.nan

    sources: dict[float, set[str]] = {}
    for family, family_frequencies in QC.LINE_NOISE_CANDIDATES.items():
        for frequency in family_frequencies:
            sources.setdefault(round(float(frequency), 6), set()).add(family)
    for frequency in discover_narrow_peaks(
        frequencies, power, threshold_db=discovery_threshold_db
    ):
        sources.setdefault(round(frequency, 6), set()).add("discovered")

    rows: list[dict[str, object]] = []
    for frequency in sorted(sources):
        families = sources[frequency]
        on_mains_grid = bool(families & set(MAINS_FAMILIES))
        applied_threshold = mains_threshold_db if on_mains_grid else discovery_threshold_db
        order_50 = nearest_harmonic_order(frequency, 50.0, tolerance_hz)
        order_60 = nearest_harmonic_order(frequency, 60.0, tolerance_hz)
        evaluable = bool(np.isfinite(highest_evaluable) and frequency <= highest_evaluable)
        value, background, prominence = (
            local_background_db(frequencies, power, frequency)
            if evaluable
            else (math.nan, math.nan, math.nan)
        )
        rows.append(
            {
                "participant_id": spectrum.participant_id,
                "site_code": site_code,
                "recording_index": spectrum.recording_index,
                "edf_path": spectrum.edf_path,
                "original_sampling_frequency_hz": spectrum.sampling_frequency_hz,
                "nyquist_frequency_hz": nyquist,
                "highest_evaluated_frequency_hz": highest_evaluable,
                "number_of_analyzed_channels": spectrum.number_of_channels,
                "frequency_resolution_hz": spectrum.frequency_resolution_hz,
                "analyzed_duration_seconds": spectrum.analyzed_duration_seconds,
                "candidate_frequency_hz": frequency,
                "candidate_source": ";".join(sorted(families)),
                "on_mains_grid": on_mains_grid,
                "evaluable_below_nyquist": evaluable,
                "power_at_candidate_frequency": value,
                "local_background_power": background,
                "peak_prominence_db": prominence,
                "near_50_hz_harmonic": order_50 is not None,
                "harmonic_order_50_hz": order_50 if order_50 is not None else "",
                "near_60_hz_harmonic": order_60 is not None,
                "harmonic_order_60_hz": order_60 if order_60 is not None else "",
                "harmonic_tolerance_hz": tolerance_hz,
                "applied_threshold_db": applied_threshold,
                "exceeds_report_threshold_db": bool(
                    np.isfinite(prominence) and prominence >= applied_threshold
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_recording_mains(
    peaks: pd.DataFrame, *, threshold_db: float = MAINS_REPORT_THRESHOLD_DB
) -> dict[str, object]:
    """Describe one recording's mains pattern without deciding anything.

    A recording is called 50 Hz or 60 Hz only when that family's fundamental or
    harmonics clear ``threshold_db`` **at the expected grid frequency**; when
    both families do, or neither, that is reported as such rather than resolved
    arbitrarily. Discovered non-mains peaks are summarised separately and never
    drive the verdict, so a noise-floor bump that happens to land within the
    harmonic tolerance cannot change a recording's mains classification.
    """
    evaluable = peaks[peaks["evaluable_below_nyquist"].astype(bool)]
    strong = evaluable[
        pd.to_numeric(evaluable["peak_prominence_db"], errors="coerce") >= threshold_db
    ]
    on_grid = strong[strong["on_mains_grid"].astype(bool)]
    fifty = on_grid[on_grid["near_50_hz_harmonic"].astype(bool)]
    sixty = on_grid[on_grid["near_60_hz_harmonic"].astype(bool)]
    # 300 Hz and its multiples are harmonics of both 50 and 60 Hz; they cannot
    # discriminate between the two families and are excluded from the verdict.
    fifty_only = fifty[~fifty["near_60_hz_harmonic"].astype(bool)]
    sixty_only = sixty[~sixty["near_50_hz_harmonic"].astype(bool)]
    if not fifty_only.empty and not sixty_only.empty:
        pattern = "both_50_and_60_hz"
    elif not fifty_only.empty:
        pattern = "50_hz"
    elif not sixty_only.empty:
        pattern = "60_hz"
    elif not on_grid.empty:
        pattern = "ambiguous_shared_harmonic_only"
    else:
        pattern = "no_obvious_mains_peak"

    def _peak_list(frame: pd.DataFrame) -> str:
        return ";".join(f"{value:g}" for value in sorted(frame["candidate_frequency_hz"]))

    non_mains = strong[
        ~strong["on_mains_grid"].astype(bool)
        & ~strong["near_50_hz_harmonic"].astype(bool)
        & ~strong["near_60_hz_harmonic"].astype(bool)
    ]
    # Upstream-notch check: a raw recording must not show a *dip* at a mains
    # harmonic. If it does, the "raw" file was filtered before it reached us and
    # its spectrum cannot be used to choose notch frequencies.
    on_grid_all = evaluable[evaluable["on_mains_grid"].astype(bool)]
    dips = on_grid_all[
        pd.to_numeric(on_grid_all["peak_prominence_db"], errors="coerce")
        <= UPSTREAM_NOTCH_DIP_DB
    ]
    return {
        "mains_pattern": pattern,
        "possible_upstream_notch": not dips.empty,
        "notch_like_dip_frequencies_hz": _peak_list(dips),
        "strong_50_hz_frequencies_hz": _peak_list(fifty),
        "strong_60_hz_frequencies_hz": _peak_list(sixty),
        "strong_non_mains_frequencies_hz": _peak_list(non_mains),
        "has_harmonics_above_fundamental": bool(
            (
                pd.to_numeric(on_grid["harmonic_order_50_hz"], errors="coerce").fillna(0) > 1
            ).any()
            or (
                pd.to_numeric(on_grid["harmonic_order_60_hz"], errors="coerce").fillna(0) > 1
            ).any()
        ),
        "number_of_strong_mains_peaks": int(len(on_grid)),
        "number_of_strong_non_mains_peaks": int(len(non_mains)),
        "classification_threshold_db": threshold_db,
        "upstream_notch_dip_threshold_db": UPSTREAM_NOTCH_DIP_DB,
    }


def summarize_dataset(recording_summary: pd.DataFrame) -> pd.DataFrame:
    """Dataset-level and per-site verdict rows built from the recording verdicts."""
    if recording_summary.empty:
        return pd.DataFrame()

    def _rows(scope: str, scope_value: str, frame: pd.DataFrame) -> dict[str, object]:
        counts = Counter(frame["mains_pattern"])
        distinct = {
            pattern
            for pattern in counts
            if pattern in {"50_hz", "60_hz", "both_50_and_60_hz"}
        }
        if not distinct:
            verdict = (
                "no_obvious_mains_peak"
                if set(counts) <= {"no_obvious_mains_peak"}
                else "ambiguous_shared_harmonic_only"
            )
        elif distinct == {"50_hz"}:
            verdict = "predominantly_50_hz"
        elif distinct == {"60_hz"}:
            verdict = "predominantly_60_hz"
        else:
            verdict = "mixed_across_recordings"
        return {
            "scope": scope,
            "scope_value": scope_value,
            "number_of_recordings": int(len(frame)),
            "number_of_participants": int(frame["participant_id"].nunique()),
            "verdict": verdict,
            "n_recordings_50_hz": int(counts.get("50_hz", 0)),
            "n_recordings_60_hz": int(counts.get("60_hz", 0)),
            "n_recordings_both": int(counts.get("both_50_and_60_hz", 0)),
            "n_recordings_ambiguous": int(counts.get("ambiguous_shared_harmonic_only", 0)),
            "n_recordings_no_mains_peak": int(counts.get("no_obvious_mains_peak", 0)),
            "n_recordings_with_harmonics": int(
                frame["has_harmonics_above_fundamental"].astype(bool).sum()
            ),
            "n_recordings_with_possible_upstream_notch": int(
                frame["possible_upstream_notch"].astype(bool).sum()
            ),
            "sampling_frequencies_hz": ";".join(
                f"{value:g}"
                for value in sorted(
                    pd.to_numeric(
                        frame["original_sampling_frequency_hz"], errors="coerce"
                    ).dropna().unique()
                )
            ),
        }

    rows = [_rows("dataset", DATASET, recording_summary)]
    for site_code, group in recording_summary.groupby("site_code", sort=True):
        rows.append(_rows("site", str(site_code), group))
    return pd.DataFrame(rows)


def notch_candidate_summary(
    peaks: pd.DataFrame, *, threshold_db: float = MAINS_REPORT_THRESHOLD_DB
) -> pd.DataFrame:
    """Frequencies worth considering for a future notch, with the evidence.

    Diagnostic only. Nothing here is applied, and no frequency is approved.
    """
    if peaks.empty:
        return pd.DataFrame()
    evaluable = peaks[peaks["evaluable_below_nyquist"].astype(bool)].copy()
    evaluable["peak_prominence_db"] = pd.to_numeric(
        evaluable["peak_prominence_db"], errors="coerce"
    )
    rows = []
    for frequency, group in evaluable.groupby("candidate_frequency_hz", sort=True):
        # `exceeds_report_threshold_db` already applies the threshold appropriate to
        # each row's source: the mains rule on grid frequencies, the stricter
        # discovery rule elsewhere.
        strong = group[group["exceeds_report_threshold_db"].astype(bool)]
        if strong.empty:
            continue
        rows.append(
            {
                "candidate_frequency_hz": float(frequency),
                "n_recordings_evaluable": int(len(group)),
                "n_recordings_above_threshold": int(len(strong)),
                "fraction_above_threshold": float(len(strong) / len(group)),
                "median_prominence_db": float(strong["peak_prominence_db"].median()),
                "max_prominence_db": float(strong["peak_prominence_db"].max()),
                "on_mains_grid": bool(group["on_mains_grid"].iloc[0]),
                "near_50_hz_harmonic": bool(group["near_50_hz_harmonic"].iloc[0]),
                "near_60_hz_harmonic": bool(group["near_60_hz_harmonic"].iloc[0]),
                "applied_threshold_db": float(group["applied_threshold_db"].iloc[0]),
                "sites": ";".join(sorted(set(strong["site_code"].astype(str)))),
                "decision": "diagnostic_only_no_notch_applied",
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(
        ["n_recordings_above_threshold", "median_prominence_db"], ascending=False
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _mark_harmonics(axis, nyquist_frequency_hz: float, upper_hz: float) -> None:
    """Draw 50/60 Hz harmonic guides, clipped to what was actually evaluated."""
    for base, colour, label in ((50.0, "#3f6899", "50 Hz harmonics"), (60.0, "#d1743c", "60 Hz harmonics")):
        drawn = False
        for frequency in harmonics_up_to(base, min(nyquist_frequency_hz, upper_hz)):
            axis.axvline(
                frequency, color=colour, linestyle=":", linewidth=0.8, alpha=0.7,
                label=None if drawn else label, zorder=1,
            )
            drawn = True


def plot_recording_spectrum(
    spectrum: QC.RecordingSpectrum,
    peaks: pd.DataFrame,
    site_code: str,
    destination: Path,
) -> Path:
    """Median-across-channel PSD for one recording, with harmonic guides."""
    figure, axis = plt.subplots(figsize=(13, 6), layout="constrained")
    upper = float(spectrum.frequencies_hz[-1])
    _mark_harmonics(axis, spectrum.sampling_frequency_hz / 2.0, upper)
    if spectrum.q25_power is not None and spectrum.q75_power is not None:
        axis.fill_between(
            spectrum.frequencies_hz, spectrum.q25_power, spectrum.q75_power,
            alpha=0.25, color="#3f6899", linewidth=0, zorder=2,
            label="Across-channel IQR",
        )
    axis.plot(
        spectrum.frequencies_hz, spectrum.power, linewidth=1.0, color="#333333",
        label=f"Median across {spectrum.number_of_channels} iEEG channels", zorder=3,
    )
    flagged = peaks[
        peaks["evaluable_below_nyquist"].astype(bool)
        & peaks["exceeds_report_threshold_db"].astype(bool)
    ]
    if not flagged.empty:
        axis.scatter(
            flagged["candidate_frequency_hz"],
            np.interp(
                flagged["candidate_frequency_hz"].to_numpy(dtype=float),
                spectrum.frequencies_hz,
                spectrum.power,
            ),
            s=26, facecolors="none", edgecolors="#c44e52", linewidths=1.2, zorder=4,
            label="Candidate peak above its reporting threshold",
        )
    axis.set_yscale("log")
    axis.set_xlim(LOWEST_ANALYSED_FREQUENCY_HZ, upper)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Power spectral density (V**2/Hz)")
    axis.set_title(
        f"Raw unfiltered spectrum - {spectrum.participant_id} [{site_code}]\n"
        f"original sfreq {spectrum.sampling_frequency_hz:g} Hz "
        f"(Nyquist {spectrum.sampling_frequency_hz / 2.0:g} Hz), "
        f"resolution {spectrum.frequency_resolution_hz:g} Hz, "
        f"{spectrum.analyzed_duration_seconds:g} s analysed",
        fontsize=10,
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_site_overview(
    spectra: Sequence[QC.RecordingSpectrum],
    site_codes: dict[str, str],
    destination: Path,
) -> Path:
    """Dataset overview: one median curve per site on a shared frequency grid.

    Recordings are interpolated onto a grid that stops at the lowest Nyquist
    bound present, so no curve is ever extrapolated past its own Nyquist.
    """
    figure, axis = plt.subplots(figsize=(13, 6), layout="constrained")
    upper = min(float(item.frequencies_hz[-1]) for item in spectra)
    grid = np.linspace(LOWEST_ANALYSED_FREQUENCY_HZ, upper, 4000)
    _mark_harmonics(axis, upper, upper)
    by_site: dict[str, list[np.ndarray]] = {}
    for item in spectra:
        site = site_codes.get(item.edf_path, "Unknown")
        by_site.setdefault(site, []).append(
            np.interp(grid, item.frequencies_hz, item.power)
        )
    for site, curves in sorted(by_site.items()):
        stacked = np.vstack(curves)
        # Each recording is normalised by its own total power before pooling, so
        # amplitude-scale differences between centres cannot dominate the shape.
        normalised = stacked / np.nansum(stacked, axis=1, keepdims=True)
        axis.plot(
            grid, np.nanmedian(normalised, axis=0), linewidth=1.2, zorder=3,
            label=f"{site} (median of {len(curves)} recordings)",
        )
    axis.set_yscale("log")
    axis.set_xlim(LOWEST_ANALYSED_FREQUENCY_HZ, upper)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Normalised power spectral density (shape only)")
    axis.set_title(
        "OpenIEEG raw unfiltered spectra by site\n"
        f"shared grid truncated at {upper:g} Hz (lowest Nyquist bound in the cohort)",
        fontsize=10,
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


# ---------------------------------------------------------------------------
# Site-restricted MNE report (same style as the HUP/UCLA aggregate QC report)
# ---------------------------------------------------------------------------
#
# This section reuses the PSDs already computed above -- no recording is read a
# second time -- and renders them with :class:`mne.Report`, the same reporting
# object ``scripts/preprocessing/aggregate_qc.py`` uses for the "HUP & UCLA
# Preprocessing QC Report". Only the *raw, before-preprocessing* half of that
# report exists here, because nothing has been preprocessed at this stage.

# Site whose participants the MNE report describes. OpenIEEG's inventory already
# excludes ``sub-openieegUCLA*`` (that cohort is covered by the separate ``ucla``
# dataset entry), and this filter is the second, explicit guarantee that no UCLA
# participant reaches the Detroit report.
DEFAULT_REPORT_SITE = "Detroit"
# Points per Hz on the pooled report grid. 8 gives 0.125 Hz, the native Welch
# resolution of an 8 s window, so a mains peak is never blunted by resampling.
REPORT_GRID_POINTS_PER_HZ = 8.0
REPORT_GRID_MAXIMUM_POINTS = 40000


@dataclass
class ParticipantRawSpectrum:
    """One participant's raw PSD, pooled across their own recordings.

    ``power`` is NaN above this participant's own highest evaluated frequency,
    so a participant recorded at a low rate contributes nothing to the pooled
    curve above their Nyquist frequency instead of being extrapolated.
    """

    participant_id: str
    site_code: str
    frequencies_hz: np.ndarray
    power: np.ndarray
    number_of_recordings: int
    sampling_frequencies_hz: list[float]
    nyquist_frequencies_hz: list[float]
    highest_evaluated_frequency_hz: float
    numbers_of_analyzed_channels: list[int]


def report_frequency_grid(highest_frequency_hz: float) -> np.ndarray:
    """Shared plotting grid from ``LOWEST_ANALYSED_FREQUENCY_HZ`` upwards.

    The grid stops at the highest frequency any recording actually evaluated, so
    it never reaches a frequency at or above the largest valid Nyquist bound.
    """
    if not np.isfinite(highest_frequency_hz) or highest_frequency_hz <= LOWEST_ANALYSED_FREQUENCY_HZ:
        raise ValueError("no frequency range to plot")
    points = int(
        round((highest_frequency_hz - LOWEST_ANALYSED_FREQUENCY_HZ) * REPORT_GRID_POINTS_PER_HZ)
    )
    points = int(np.clip(points, 16, REPORT_GRID_MAXIMUM_POINTS))
    return np.linspace(LOWEST_ANALYSED_FREQUENCY_HZ, highest_frequency_hz, points)


def _project_onto_grid(spectrum: QC.RecordingSpectrum, grid: np.ndarray) -> np.ndarray:
    """Interpolate one recording onto ``grid``, NaN outside its evaluated range.

    ``np.interp`` clamps to the end values outside the sample range, which would
    silently draw a flat line above a recording's Nyquist frequency. Masking
    instead keeps the guarantee that nothing above true Nyquist is ever plotted.
    """
    values = np.full(grid.shape, np.nan, dtype=float)
    frequencies = np.asarray(spectrum.frequencies_hz, dtype=float)
    if frequencies.size == 0:
        return values
    inside = (grid >= frequencies[0]) & (grid <= frequencies[-1])
    values[inside] = np.interp(
        grid[inside], frequencies, np.asarray(spectrum.power, dtype=float)
    )
    return values


def participant_raw_spectra(
    spectra: Sequence[QC.RecordingSpectrum],
    site_codes: dict[str, str],
    *,
    site: str,
    grid: np.ndarray,
) -> list[ParticipantRawSpectrum]:
    """Pool the already-computed recording PSDs into one curve per participant.

    Recordings of the same participant may have different sampling rates; each is
    projected onto ``grid`` only where it was actually evaluated and the pooled
    curve is the median over the recordings available at each frequency.
    """
    selected = [
        spectrum
        for spectrum in spectra
        if site_codes.get(spectrum.edf_path, "Unknown") == site
    ]
    by_participant: dict[str, list[QC.RecordingSpectrum]] = {}
    for spectrum in selected:
        by_participant.setdefault(str(spectrum.participant_id), []).append(spectrum)

    pooled: list[ParticipantRawSpectrum] = []
    for participant_id, items in sorted(by_participant.items()):
        stacked = np.vstack([_project_onto_grid(item, grid) for item in items])
        with np.errstate(invalid="ignore"):
            curve = np.nanmedian(stacked, axis=0) if stacked.shape[0] > 1 else stacked[0]
        pooled.append(
            ParticipantRawSpectrum(
                participant_id=participant_id,
                site_code=site,
                frequencies_hz=grid,
                power=np.asarray(curve, dtype=float),
                number_of_recordings=len(items),
                sampling_frequencies_hz=sorted(
                    {float(item.sampling_frequency_hz) for item in items}
                ),
                nyquist_frequencies_hz=sorted(
                    {float(item.sampling_frequency_hz) / 2.0 for item in items}
                ),
                highest_evaluated_frequency_hz=max(
                    float(item.frequencies_hz[-1]) for item in items
                ),
                numbers_of_analyzed_channels=sorted(
                    {int(item.number_of_channels) for item in items}
                ),
            )
        )
    return pooled


def _mark_nyquist_limits(axis, nyquist_frequencies_hz: Sequence[float]) -> None:
    """Mark every distinct Nyquist bound present, so the reader can see where
    curves necessarily stop. Frequencies differ per recording, so there is no
    single Nyquist limit to annotate."""
    drawn = False
    for frequency in sorted({float(value) for value in nyquist_frequencies_hz}):
        axis.axvline(
            frequency, color="#666666", linestyle="--", linewidth=0.9, alpha=0.8, zorder=2,
            label=None if drawn else "Nyquist limit of a sampling rate in the cohort",
        )
        drawn = True


def _positive_log_limits(values: np.ndarray) -> tuple[float, float]:
    """Symmetric log-axis padding around the finite positive values.

    Same rule as the HUP/UCLA aggregate report, so both reports read alike.
    """
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return 1e-12, 1.0
    lower = float(positive.min())
    upper = float(positive.max())
    if lower == upper:
        return lower / 10.0, upper * 10.0
    padding = 10.0 ** (0.04 * (np.log10(upper) - np.log10(lower)))
    return lower / padding, upper * padding


def plot_site_dataset_raw_psd(
    participants: Sequence[ParticipantRawSpectrum],
    *,
    site: str,
    nyquist_frequencies_hz: Sequence[float],
    ylim: tuple[float, float],
):
    """Dataset-level raw PSD for one site: median and IQR across participants.

    At each frequency the median is taken over the participants whose own
    Nyquist frequency reaches it; the count of contributing participants is drawn
    on a second axis so a step in the curve cannot be mistaken for a spectral
    feature.
    """
    figure, axis = plt.subplots(figsize=(13, 6), layout="constrained")
    grid = participants[0].frequencies_hz
    upper = float(grid[-1])
    _mark_harmonics(axis, upper, upper)
    _mark_nyquist_limits(axis, nyquist_frequencies_hz)

    matrix = np.vstack([participant.power for participant in participants])
    contributing = np.sum(np.isfinite(matrix), axis=0)
    with np.errstate(invalid="ignore"):
        median_curve = np.nanmedian(matrix, axis=0)
        q25_curve = np.nanpercentile(matrix, 25, axis=0)
        q75_curve = np.nanpercentile(matrix, 75, axis=0)

    axis.fill_between(
        grid, q25_curve, q75_curve, color="#4c78a8", alpha=0.25, linewidth=0, zorder=3,
        label="IQR across participants",
    )
    axis.plot(
        grid, median_curve, color="#4c78a8", linewidth=2.0, zorder=4,
        label="Median across participants",
    )
    axis.set_yscale("log")
    axis.set_xlim(LOWEST_ANALYSED_FREQUENCY_HZ, upper)
    axis.set_ylim(ylim)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Power spectral density (V**2/Hz)")
    axis.set_title(
        f"{site} raw PSD before preprocessing (N = {len(participants)} participants)\n"
        "unfiltered, unresampled, no notch applied",
        fontsize=11,
    )
    axis.grid(True, which="both", alpha=0.2)

    counts = axis.twinx()
    counts.step(grid, contributing, where="mid", color="#999999", linewidth=0.8, alpha=0.7)
    counts.set_ylabel("Participants contributing at this frequency", fontsize=8, color="#666666")
    counts.set_ylim(0, len(participants) * 1.05)
    counts.tick_params(axis="y", labelsize=8, colors="#666666")

    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="upper right", fontsize=8, frameon=False, ncol=2)
    return figure


def plot_participant_raw_psd_overlay(
    participants: Sequence[ParticipantRawSpectrum],
    *,
    site: str,
    nyquist_frequencies_hz: Sequence[float],
    ylim: tuple[float, float],
):
    """One raw PSD curve per participant, all on a single figure.

    Each curve is drawn only over the frequencies that participant actually
    evaluated, so curves from lower sampling rates simply end at their own
    Nyquist bound rather than being extended.
    """
    figure, axis = plt.subplots(figsize=(13, 6), layout="constrained")
    grid = participants[0].frequencies_hz
    upper = float(grid[-1])
    _mark_harmonics(axis, upper, upper)
    _mark_nyquist_limits(axis, nyquist_frequencies_hz)
    # Viridis rather than the default colour cycle: its first two colours would
    # otherwise be the blue and orange of the 50/60 Hz harmonic guides.
    colours = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, max(len(participants), 1)))
    for participant, colour in zip(participants, colours):
        axis.plot(grid, participant.power, linewidth=0.8, alpha=0.5, color=colour, zorder=3)
    axis.set_yscale("log")
    axis.set_xlim(LOWEST_ANALYSED_FREQUENCY_HZ, upper)
    axis.set_ylim(ylim)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Power spectral density (V**2/Hz)")
    axis.set_title(
        f"{site} raw PSD before preprocessing - one curve per participant "
        f"({len(participants)} participants)",
        fontsize=11,
    )
    axis.grid(True, which="both", alpha=0.2)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="upper right", fontsize=8, frameon=False)
    return figure


def build_site_summary_html(
    participants: Sequence[ParticipantRawSpectrum],
    *,
    site: str,
    args: argparse.Namespace,
    excluded_sites: dict[str, int],
) -> str:
    """Short dataset summary: subjects, recordings, sampling rates, channels."""

    def _counts(values: Sequence[float]) -> str:
        counter = Counter(values)
        return ", ".join(
            f"{value:g} Hz ({count} participant{'s' if count != 1 else ''})"
            for value, count in sorted(counter.items())
        ) or "N/A"

    rates = [rate for item in participants for rate in item.sampling_frequencies_hz]
    channels = [
        count for item in participants for count in item.numbers_of_analyzed_channels
    ]
    nyquists = sorted({rate / 2.0 for rate in rates})
    # A participant recorded at two rates is counted under both.
    rate_rows = _counts(rates)
    excluded = (
        ", ".join(f"{name} ({count} recordings)" for name, count in sorted(excluded_sites.items()))
        or "none"
    )
    rows = [
        ("Site", site),
        ("Participants (subjects)", f"{len(participants)}"),
        ("Recordings", f"{sum(item.number_of_recordings for item in participants)}"),
        ("Sampling rates present", ", ".join(f"{value:g} Hz" for value in sorted(set(rates))) or "N/A"),
        ("Participants per sampling rate", rate_rows),
        ("Nyquist limits present", ", ".join(f"{value:g} Hz" for value in nyquists) or "N/A"),
        (
            "Analyzed iEEG channels per recording",
            f"min {min(channels)}, median {int(np.median(channels))}, max {max(channels)}"
            if channels
            else "N/A",
        ),
        (
            "Raw samples analysed per recording",
            f"{args.duration_per_recording:g} s (centred)"
            if args.duration_per_recording
            else "whole recording",
        ),
        (
            "Welch settings",
            f"{args.window_seconds:g} s window, {args.overlap_seconds:g} s overlap, "
            f"{1.0 / args.window_seconds:g} Hz resolution, channel {args.channel_reduction}",
        ),
        ("Preprocessing applied", "None: no notch, filter, resampling or re-referencing"),
        ("OpenIEEG sites analysed but not shown here", excluded),
    ]
    body = "".join(
        '<tr><td style="padding:8px;border:1px solid #ddd;font-weight:bold;">'
        f"{name}</td><td style=\"padding:8px;border:1px solid #ddd;\">{value}</td></tr>"
        for name, value in rows
    )
    return f"""
    <div style="font-family: sans-serif; margin: 1em 0;">
      <h3>{site} raw dataset summary (before preprocessing)</h3>
      <p>Every curve in this report is a raw, unfiltered, unresampled PSD computed
      directly from the EDF samples. <strong>No notch frequency has been chosen or
      applied.</strong> The 50 Hz and 60 Hz guides are drawn up to the Nyquist limit
      of each sampling rate present; nothing is plotted or evaluated at or above a
      recording's own Nyquist frequency.</p>
      <table style="border-collapse: collapse; width: 100%; max-width: 900px;
                    text-align: left; border: 1px solid #ddd;">
        <tbody>{body}</tbody>
      </table>
    </div>
    """


def build_site_raw_report(
    spectra: Sequence[QC.RecordingSpectrum],
    site_codes: dict[str, str],
    destination: Path,
    *,
    site: str,
    args: argparse.Namespace,
) -> Path:
    """Write the raw-only MNE report for one OpenIEEG site.

    Reuses the PSDs computed by :func:`run`; nothing is read from disk again.
    """
    kept = [
        spectrum
        for spectrum in spectra
        if site_codes.get(spectrum.edf_path, "Unknown") == site
    ]
    if not kept:
        raise ValueError(
            f"no {site} recording produced a raw PSD; cannot build the {site} report"
        )
    excluded_sites = Counter(
        site_codes.get(spectrum.edf_path, "Unknown")
        for spectrum in spectra
        if site_codes.get(spectrum.edf_path, "Unknown") != site
    )

    highest = max(float(spectrum.frequencies_hz[-1]) for spectrum in kept)
    grid = report_frequency_grid(highest)
    participants = participant_raw_spectra(kept, site_codes, site=site, grid=grid)
    nyquists = sorted({spectrum.sampling_frequency_hz / 2.0 for spectrum in kept})
    ylim = _positive_log_limits(
        np.concatenate([participant.power for participant in participants])
    )

    report = mne.Report(
        title=f"OpenIEEG {site} Raw Spectral QC Report (before preprocessing)",
        verbose="error",
    )
    report.add_html(
        build_site_summary_html(
            participants, site=site, args=args, excluded_sites=dict(excluded_sites)
        ),
        title="Dataset summary",
        section="1. Dataset summary",
    )

    dataset_figure = plot_site_dataset_raw_psd(
        participants, site=site, nyquist_frequencies_hz=nyquists, ylim=ylim
    )
    report.add_figure(
        dataset_figure,
        title=f"{site} dataset-level raw PSD",
        section="2. Dataset-level PSD before preprocessing",
    )
    plt.close(dataset_figure)

    overlay_figure = plot_participant_raw_psd_overlay(
        participants, site=site, nyquist_frequencies_hz=nyquists, ylim=ylim
    )
    report.add_figure(
        overlay_figure,
        title=f"All {site} participants before preprocessing",
        section="3. Participant-level PSD before preprocessing",
    )
    plt.close(overlay_figure)

    destination.parent.mkdir(parents=True, exist_ok=True)
    report.save(destination, overwrite=True, open_browser=False, verbose="error")
    LOGGER.info(
        "%s raw spectral report: %d participants, %d recordings -> %s",
        site,
        len(participants),
        sum(item.number_of_recordings for item in participants),
        destination,
    )
    return destination


# ---------------------------------------------------------------------------
# Selection and runner
# ---------------------------------------------------------------------------


def select_openieeg_rows(
    inventory: pd.DataFrame, *, limit: int | None = None, participant_ids: Sequence[str] | None = None
) -> pd.DataFrame:
    """Every OpenIEEG recording in the inventory, at whatever rate it was sampled.

    The >=1 kHz eligibility flag is intentionally *not* applied: excluding low-rate
    recordings would bias the very diagnostic this stage exists to produce.
    """
    required = {
        "recording_index", "participant_id", "dataset", "edf_path",
        "original_sampling_frequency_hz",
    }
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise ValueError(f"inventory is missing columns: {', '.join(missing)}")
    selected = inventory.copy()
    selected["dataset"] = selected["dataset"].astype(str).str.strip().str.lower()
    selected = selected[selected["dataset"].eq(DATASET)]
    if selected.empty:
        raise ValueError(f"inventory contains no {DATASET} recordings")
    if participant_ids:
        wanted = {str(value) for value in participant_ids}
        selected = selected[selected["participant_id"].astype(str).isin(wanted)]
        if selected.empty:
            raise ValueError(f"no {DATASET} recording matches {sorted(wanted)}")
    selected = selected.sort_values(["participant_id", "recording_index"], kind="stable")
    if limit is not None:
        selected = selected.head(limit)
    return selected.reset_index(drop=True)


def _safe_stem(participant_id: str, recording_index: object) -> str:
    """Filesystem-safe, collision-free stem for one recording's figure."""
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(participant_id)
    )
    return f"{safe}_recording-{recording_index}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Raw pre-preprocessing spectral / line-noise inspection of OpenIEEG. "
            "Diagnostic only: no filtering, no resampling, no notch decision."
        )
    )
    parser.add_argument("--inventory-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--window-seconds", type=float, default=8.0,
        help="Welch segment length; 8 s gives the 0.125 Hz resolution used for HUP/UCLA",
    )
    parser.add_argument("--overlap-seconds", type=float, default=0.0)
    parser.add_argument(
        "--duration-per-recording", type=float, default=600.0,
        help="centred span of raw samples to analyse per recording, in seconds",
    )
    parser.add_argument("--limit", type=int, help="analyse only the first N recordings")
    parser.add_argument(
        "--participant-id", action="append", dest="participant_ids",
        help="restrict to these participants; repeatable (used by the pilot)",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--harmonic-tolerance-hz", type=float, default=HARMONIC_TOLERANCE_HZ,
        help="how close a peak must be to a harmonic to be tagged as one",
    )
    parser.add_argument(
        "--peak-threshold-db", type=float, default=MAINS_REPORT_THRESHOLD_DB,
        help=(
            "prominence in dB above the local background needed to flag a 50/60 Hz "
            "grid candidate; a reporting threshold only, never a notch decision"
        ),
    )
    parser.add_argument(
        "--discovery-threshold-db", type=float, default=DISCOVERY_THRESHOLD_DB,
        help=(
            "prominence in dB needed to report an unknown narrow peak; stricter than "
            "--peak-threshold-db because discovery scans every Welch bin"
        ),
    )
    parser.add_argument("--channel-reduction", choices=("median", "mean"), default="median")
    parser.add_argument(
        "--mne-report-html", type=Path,
        help=(
            "also write a raw-only MNE report, in the style of the HUP/UCLA "
            "preprocessing QC report, restricted to --mne-report-site. The PSDs "
            "already computed for the CSV outputs are reused; nothing is re-read."
        ),
    )
    parser.add_argument(
        "--mne-report-site", default=DEFAULT_REPORT_SITE,
        help=(
            "site code the MNE report covers. OpenIEEG's inventory already excludes "
            "sub-openieegUCLA*, and this is the second, explicit guarantee that only "
            "that site's participants appear in the report"
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _write_report(
    destination: Path,
    *,
    args: argparse.Namespace,
    dataset_summary: pd.DataFrame,
    recording_summary: pd.DataFrame,
    notch_candidates: pd.DataFrame,
    failures: pd.DataFrame,
    figure_names: list[str],
    overview_name: str | None,
) -> None:
    def _table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "<p><em>none</em></p>"
        return frame.to_html(index=False, border=0)

    flagged = (
        int(recording_summary["possible_upstream_notch"].astype(bool).sum())
        if "possible_upstream_notch" in recording_summary.columns
        else 0
    )
    _upstream_notch_note = (
        "<p><strong>No recording shows a notch-like dip at a mains harmonic.</strong></p>"
        if flagged == 0
        else f'<p class="warn"><strong>{flagged} recording(s) show a notch-like dip at a '
        "mains harmonic and may already have been filtered upstream. Their spectra must not "
        "be used to choose notch frequencies; inspect them individually.</strong></p>"
    )
    images = "".join(
        f'<figure><img src="{name}" alt="{name}"><figcaption>{name}</figcaption></figure>'
        for name in figure_names
    )
    overview = (
        f'<h2>Dataset overview</h2><img src="{overview_name}" alt="site overview">'
        if overview_name
        else ""
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>OpenIEEG raw spectral QC</title>
<style>body{{font-family:sans-serif;max-width:1200px;margin:2rem auto}}
table{{border-collapse:collapse;font-size:.8rem}} th,td{{border:1px solid #ccc;padding:.3rem}}
img{{max-width:100%}} figure{{margin:1rem 0}} code{{background:#eee;padding:.15rem}}
.warn{{background:#fff4e5;border-left:4px solid #d1743c;padding:.6rem 1rem}}</style></head>
<body><h1>OpenIEEG raw spectral / line-noise QC</h1>
<p class="warn"><strong>Diagnostic only.</strong> This report is evidence for choosing notch
frequencies later. No notch, band-pass, high-pass, low-pass or resampling operation was applied,
and no preprocessing decision is made or implied by any number below.</p>
<h2>Method</h2>
<p>EDF opened with <code>preload=False</code>; the only operation applied to the samples is
Welch's method. Sampling rate is read from each recording's own EDF header. Channels are selected
by <em>type</em> from the <code>*_channels.tsv</code> sidecar (SEEG/ECOG/EEG/IEEG, status not bad),
which excludes auxiliary channels without hard-coding any channel name. Channels are reduced by
{args.channel_reduction}.</p>
<p>Welch: window {args.window_seconds:g} s, overlap {args.overlap_seconds:g} s, hann,
<code>detrend="constant"</code>, <code>scaling="density"</code>, nominal resolution
{1.0 / args.window_seconds:g} Hz, up to {args.duration_per_recording:g} s of raw samples per
recording. The spectrum is truncated at that recording's Nyquist frequency minus one native bin;
candidates above it are listed as not evaluable rather than dropped.</p>
<p>Prominence is the dB ratio of the power at a candidate to the median of a
{QC.LOCAL_BACKGROUND_INNER_HZ:g}-{QC.LOCAL_BACKGROUND_OUTER_HZ:g} Hz ring on either side of it.
A peak is tagged as a mains harmonic when it lies within
{args.harmonic_tolerance_hz:g} Hz of a 50 Hz or 60 Hz multiple; peaks are also discovered directly
from the data, so non-mains narrow peaks appear too. The
{args.peak_threshold_db:g} dB threshold on the mains grid, and the stricter
{args.discovery_threshold_db:g} dB threshold used for discovering unknown peaks, are reporting
thresholds only. The mains verdict is decided from the 50/60 Hz grid frequencies alone, so an
incidental noise-floor bump near a harmonic cannot change a recording's classification.</p>
<h2>Upstream-notch check</h2>
<p>A recording that was already notch-filtered before it reached this stage shows a sharp
<em>dip</em> at a mains harmonic instead of a peak. Any such recording is counted in
<code>n_recordings_with_possible_upstream_notch</code> below and flagged per recording in
<code>possible_upstream_notch</code>; a count of zero is the evidence that these spectra are
genuinely unfiltered.</p>
{_upstream_notch_note}
<h2>Dataset and site verdict</h2>{_table(dataset_summary)}
<h2>Candidate frequencies for future notch review</h2>
<p>Ranked by how many recordings show them. <strong>Nothing here has been applied.</strong></p>
{_table(notch_candidates)}
<h2>Per-recording verdict</h2>{_table(recording_summary)}
<h2>Failed recordings</h2>
<p>Recordings that could not be analysed, with the reason. These are reported, not skipped.</p>
{_table(failures)}
{overview}
<h2>Per-recording spectra</h2>{images}
</body></html>"""
    destination.write_text(page, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.window_seconds <= 0 or args.overlap_seconds < 0:
        raise ValueError("window must be positive and overlap cannot be negative")
    if args.n_jobs < 1:
        raise ValueError("n-jobs must be at least one")
    output_dir = args.output_dir.expanduser()
    spectra_dir = output_dir / "recording_spectra"
    output_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)

    inventory = pd.read_csv(args.inventory_csv.expanduser())
    selected = select_openieeg_rows(
        inventory, limit=args.limit, participant_ids=args.participant_ids
    )
    LOGGER.info("Selected %d OpenIEEG recording(s) for raw spectral QC", len(selected))

    def process(item: tuple[int, pd.Series]):
        _, row = item
        participant_id = str(row.get("participant_id", ""))
        site_code = INVENTORY.infer_site_code(participant_id, DATASET)
        base = {
            "recording_index": row.get("recording_index", ""),
            "participant_id": participant_id,
            "site_code": site_code,
            "edf_path": row.get("edf_path", ""),
            "inventory_sampling_frequency_hz": row.get("original_sampling_frequency_hz"),
        }
        try:
            spectrum = QC.analyze_raw_recording(
                row,
                data_root=args.data_root.expanduser(),
                window_seconds=args.window_seconds,
                overlap_seconds=args.overlap_seconds,
                duration_per_recording=args.duration_per_recording,
                channel_reduction=args.channel_reduction,
                # Diagnostic scope: accept every rate, and let each recording's own
                # Nyquist frequency be the only upper bound.
                minimum_sampling_frequency_hz=MINIMUM_SAMPLING_FREQUENCY_HZ,
                require_eligibility_flag=False,
                maximum_frequency_hz=None,
            )
            return spectrum, site_code, base
        except Exception as error:  # reported in the failure table, never swallowed
            LOGGER.warning("FAILED %s: %s", base["edf_path"], error)
            base.update({"failure_reason": f"{type(error).__name__}: {error}"})
            return None, site_code, base

    items = list(selected.iterrows())
    if args.n_jobs == 1:
        processed = [process(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
            processed = list(executor.map(process, items))

    spectra = [spectrum for spectrum, _, _ in processed if spectrum is not None]
    failures = pd.DataFrame(
        [base for spectrum, _, base in processed if spectrum is None]
    )
    site_codes = {
        spectrum.edf_path: site
        for spectrum, site, _ in processed
        if spectrum is not None
    }
    LOGGER.info("Analysed %d recording(s); %d failed", len(spectra), len(failures))
    if not spectra:
        raise RuntimeError(
            "no OpenIEEG recording produced a valid raw PSD; see the failure table"
        )

    peak_frames, summary_rows, figure_names = [], [], []
    for spectrum in spectra:
        site_code = site_codes[spectrum.edf_path]
        peaks = candidate_peak_table(
            spectrum,
            site_code=site_code,
            tolerance_hz=args.harmonic_tolerance_hz,
            mains_threshold_db=args.peak_threshold_db,
            discovery_threshold_db=args.discovery_threshold_db,
        )
        peak_frames.append(peaks)
        stem = _safe_stem(spectrum.participant_id, spectrum.recording_index)
        name = f"{stem}.png"
        plot_recording_spectrum(spectrum, peaks, site_code, spectra_dir / name)
        figure_names.append(f"recording_spectra/{name}")
        summary_rows.append(
            {
                "participant_id": spectrum.participant_id,
                "site_code": site_code,
                "recording_index": spectrum.recording_index,
                "edf_path": spectrum.edf_path,
                "original_sampling_frequency_hz": spectrum.sampling_frequency_hz,
                "nyquist_frequency_hz": spectrum.sampling_frequency_hz / 2.0,
                "highest_evaluated_frequency_hz": float(spectrum.frequencies_hz[-1]),
                "number_of_analyzed_channels": spectrum.number_of_channels,
                "frequency_resolution_hz": spectrum.frequency_resolution_hz,
                "analyzed_duration_seconds": spectrum.analyzed_duration_seconds,
                "spectrum_figure": f"recording_spectra/{name}",
                **classify_recording_mains(peaks, threshold_db=args.peak_threshold_db),
            }
        )

    peaks_table = pd.concat(peak_frames, ignore_index=True)
    recording_summary = pd.DataFrame(summary_rows)
    dataset_summary = summarize_dataset(recording_summary)
    notch_candidates = notch_candidate_summary(
        peaks_table, threshold_db=args.peak_threshold_db
    )

    overview_name = None
    if len(spectra) > 1:
        overview_name = "openieeg_site_overview.png"
        plot_site_overview(spectra, site_codes, output_dir / overview_name)

    outputs = {
        "peaks": output_dir / "openieeg_candidate_peaks.csv",
        "recordings": output_dir / "openieeg_recording_summary.csv",
        "dataset": output_dir / "openieeg_dataset_summary.csv",
        "notch_candidates": output_dir / "openieeg_notch_candidates.csv",
        "failures": output_dir / "openieeg_failed_recordings.csv",
        "report": output_dir / "openieeg_raw_spectral_qc.html",
    }
    peaks_table.to_csv(outputs["peaks"], index=False)
    recording_summary.to_csv(outputs["recordings"], index=False)
    dataset_summary.to_csv(outputs["dataset"], index=False)
    notch_candidates.to_csv(outputs["notch_candidates"], index=False)
    failures.to_csv(outputs["failures"], index=False)
    _write_report(
        outputs["report"],
        args=args,
        dataset_summary=dataset_summary,
        recording_summary=recording_summary,
        notch_candidates=notch_candidates,
        failures=failures,
        figure_names=figure_names,
        overview_name=overview_name,
    )
    if overview_name:
        outputs["overview"] = output_dir / overview_name

    # Raw-only MNE report for one site, built from the spectra already in memory.
    if getattr(args, "mne_report_html", None) is not None:
        outputs["mne_report"] = build_site_raw_report(
            spectra,
            site_codes,
            args.mne_report_html.expanduser(),
            site=getattr(args, "mne_report_site", DEFAULT_REPORT_SITE),
            args=args,
        )

    for _, row in dataset_summary.iterrows():
        LOGGER.info(
            "%s %s: %s (%d recordings)",
            row["scope"], row["scope_value"], row["verdict"], row["number_of_recordings"],
        )
    if not failures.empty:
        LOGGER.warning(
            "%d recording(s) failed and are listed in %s", len(failures), outputs["failures"]
        )
    for path in outputs.values():
        LOGGER.info("Wrote %s", path)
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
