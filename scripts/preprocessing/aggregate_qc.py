#!/usr/bin/env python3
"""Aggregate HUP and UCLA preprocessing QC into a standalone MNE Report.

Minimal adaptation of Neurodags/aggregate_preprocessing_qc_v3.py for 1 kHz no-bandpass
preprocessing outputs (HUP and UCLA cohorts).
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("aggregate_hup_ucla_preprocessing_qc")

FREQUENCY_MIN = 0.5
FREQUENCY_MAX = 500.0
FREQUENCY_STEP = 0.5

FREQUENCY_GRID = np.arange(
    FREQUENCY_MIN,
    FREQUENCY_MAX + FREQUENCY_STEP / 2.0,
    FREQUENCY_STEP,
)

# Cohérent avec les epochs de 8 secondes du preprocessing
WELCH_WINDOW_SECONDS = 8.0
SUBJECT_RE = re.compile(r"(sub-[^_/\\]+)")


@dataclass
class PreprocessedRecording:
    dataset: str
    participant_id: str
    recording_id: str
    metadata_path: Path
    fif_path: Path
    edf_path: Path | None
    metadata: dict[str, Any]
    source_root: Path | None = None
    reused: bool = False


def _clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_subject(text: str | Path) -> str:
    value = text.as_posix() if isinstance(text, Path) else str(text)
    match = SUBJECT_RE.search(value)
    if match:
        return match.group(1)
    return Path(value).stem


def _channels_sidecar(raw_path: Path) -> Path | None:
    name = raw_path.name
    lower = name.lower()
    for suffix in ("_ieeg.edf", "_ieeg.fif.gz", "_ieeg.fif"):
        if lower.endswith(suffix):
            sidecar = raw_path.with_name(name[: -len(suffix)] + "_channels.tsv")
            if sidecar.is_file():
                return sidecar
    sidecar = raw_path.with_suffix("").with_name(raw_path.stem + "_channels.tsv")
    return sidecar if sidecar.is_file() else None


def _bad_channels(raw_path: Path, ch_names: list[str]) -> list[str]:
    sidecar = _channels_sidecar(raw_path)
    if sidecar is None or not sidecar.is_file():
        return []
    try:
        channels = pd.read_csv(sidecar, sep="\t")
    except Exception:
        return []
    if "name" not in channels.columns:
        return []
    bad = pd.Series(False, index=channels.index)
    if "good" in channels.columns:
        bad |= pd.to_numeric(channels["good"], errors="coerce").eq(0)
    if "status" in channels.columns:
        bad |= channels["status"].astype(str).str.lower().eq("bad")
    flagged = set(channels.loc[bad, "name"].astype(str))
    return [name for name in ch_names if name in flagged]


def _apply_bads(inst: Any, bads: list[str]) -> None:
    present = set(inst.ch_names)
    inst.info["bads"] = sorted(set(inst.info.get("bads", [])) | (set(bads) & present))


def _resolve_edf_path(
    metadata: dict[str, Any],
    metadata_path: Path,
    raw_root: Path | None,
    manifest_lookup: dict[str, Path] | None,
) -> Path | None:
    candidates = [
        metadata.get("source_edf_path"),
        metadata.get("edf_path"),
        metadata.get("source_path"),
    ]
    for cand in candidates:
        if cand:
            path = Path(str(cand)).expanduser()
            if path.is_file():
                return path.resolve()
            if raw_root:
                resolved_raw = (raw_root / path.name).resolve()
                if resolved_raw.is_file():
                    return resolved_raw
                # Try relative matching under raw_root
                try:
                    rel_idx = str(path).find("sub-")
                    if rel_idx >= 0:
                        sub_path = (raw_root / str(path)[rel_idx:]).resolve()
                        if sub_path.is_file():
                            return sub_path
                except Exception:
                    pass

    # Check manifest lookup if provided
    if manifest_lookup:
        part_id = metadata.get("participant_id")
        if part_id and part_id in manifest_lookup:
            return manifest_lookup[part_id]
        stem = metadata_path.name.split("@")[0]
        if stem in manifest_lookup:
            return manifest_lookup[stem]

    # Check beside metadata or parent
    if raw_root:
        stem = metadata_path.name.split("@")[0]
        matches = list(raw_root.rglob(f"{stem}.edf"))
        if len(matches) == 1:
            return matches[0].resolve()

    return None


def discover_preprocessed_recordings(
    preprocessed_roots: list[Path],
    raw_root: Path | None = None,
    manifest_path: Path | None = None,
) -> list[PreprocessedRecording]:
    manifest_lookup: dict[str, Path] = {}
    if manifest_path and manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    edf = row.get("edf_path") or row.get("source_edf_path")
                    if edf:
                        p = Path(edf).expanduser().resolve()
                        part_id = row.get("participant_id")
                        if part_id:
                            manifest_lookup[part_id] = p
                        manifest_lookup[p.stem] = p
        except Exception as err:
            LOGGER.warning("Could not parse manifest %s: %s", manifest_path, err)

    recordings: list[PreprocessedRecording] = []
    seen_metadata_paths: set[Path] = set()
    # Recording identity, not metadata path: a recording reused in place from a
    # previous analysis root is the same recording, and must be counted once.
    # Roots are scanned in order, so the current root wins over a reuse root.
    claimed_identities: set[tuple[str, str, str]] = set()

    for root_index, root in enumerate(preprocessed_roots):
        if not root.exists():
            LOGGER.warning("Preprocessed root does not exist: %s", root)
            continue
        root_identities: set[tuple[str, str, str]] = set()

        # Look for metadata JSON files; ignore split FIF files (-1.fif, -2.fif)
        for meta_file in sorted(root.rglob("*.metadata.json")):
            resolved_meta = meta_file.resolve()
            if resolved_meta in seen_metadata_paths:
                continue
            seen_metadata_paths.add(resolved_meta)

            try:
                metadata = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception as err:
                LOGGER.warning("Could not read metadata JSON %s: %s", meta_file, err)
                continue

            # Determine primary FIF file
            fif_name = meta_file.name.replace(".metadata.json", ".fif")
            fif_file = meta_file.with_name(fif_name)
            if not fif_file.is_file():
                # Try finding without suffix replacement if named differently
                stem = meta_file.name.split("@")[0]
                candidates = list(meta_file.parent.glob(f"{stem}*.fif"))
                # Exclude split FIFs (-1.fif, -2.fif) from primary selection if main exists
                primary_cands = [c for c in candidates if not re.search(r"-\d+\.fif$", c.name)]
                if primary_cands:
                    fif_file = primary_cands[0]
                elif candidates:
                    fif_file = candidates[0]
                else:
                    LOGGER.warning("No FIF file found for metadata %s", meta_file)
                    continue

            dataset = _clean(metadata.get("dataset") or metadata.get("site")).lower()
            if not dataset:
                if "hup" in meta_file.parts or "hup" in root.parts:
                    dataset = "hup"
                elif "ucla" in meta_file.parts or "ucla" in root.parts:
                    dataset = "ucla"
                else:
                    dataset = "unknown"

            participant_id = _clean(metadata.get("participant_id"))
            if not participant_id:
                participant_id = _extract_subject(meta_file)

            rec_id = meta_file.name.split("@")[0]
            edf_path = _resolve_edf_path(metadata, meta_file, raw_root, manifest_lookup)

            identity = (dataset, participant_id, rec_id)
            if identity in claimed_identities:
                LOGGER.debug(
                    "Recording %s already supplied by an earlier root; skipping %s",
                    identity, meta_file,
                )
                continue
            root_identities.add(identity)

            recordings.append(
                PreprocessedRecording(
                    dataset=dataset,
                    participant_id=participant_id,
                    recording_id=rec_id,
                    metadata_path=meta_file,
                    fif_path=fif_file,
                    edf_path=edf_path,
                    metadata=metadata,
                    source_root=root,
                    reused=root_index > 0,
                )
            )

        claimed_identities |= root_identities
        LOGGER.info(
            "Root %s contributed %d recording(s)%s",
            root, len(root_identities), " (reused in place)" if root_index else "",
        )

    return recordings


def provenance_summary(recordings: list[PreprocessedRecording]) -> pd.DataFrame:
    """Recomputed vs reused counts, so the report covers the whole cohort."""
    if not recordings:
        return pd.DataFrame()
    rows = [
        {
            "dataset": recording.dataset,
            "participant_id": recording.participant_id,
            "recording_id": recording.recording_id,
            "origin": "reused_read_only" if recording.reused else "recomputed",
            "source_root": str(recording.source_root or ""),
        }
        for recording in recordings
    ]
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["dataset", "origin", "source_root"], sort=True)
        .agg(participants=("participant_id", "nunique"), recordings=("recording_id", "nunique"))
        .reset_index()
    )
    totals = (
        frame.groupby(["origin"], sort=True)
        .agg(participants=("participant_id", "nunique"), recordings=("recording_id", "nunique"))
        .reset_index()
    )
    totals["dataset"] = "(all sites)"
    totals["source_root"] = "(all roots)"
    return pd.concat([summary, totals[summary.columns]], ignore_index=True)

def _compute_welch_psd(inst: Any) -> tuple[np.ndarray, np.ndarray]:
    sfreq = float(inst.info["sfreq"])
    nyquist = sfreq / 2.0

    if nyquist + 1e-6 < FREQUENCY_MAX:
        raise ValueError(
            f"Sampling frequency {sfreq:g} Hz gives a Nyquist frequency "
            f"of only {nyquist:g} Hz; cannot compute a PSD up to "
            f"{FREQUENCY_MAX:g} Hz."
        )

    n_times = len(inst.times) if hasattr(inst, "times") else inst.n_times

    n_per_seg = min(
        n_times,
        max(8, int(round(WELCH_WINDOW_SECONDS * sfreq))),
    )

    if n_per_seg < 8:
        raise ValueError("recording too short")

    spectrum = inst.compute_psd(
        method="welch",
        fmin=0.0,
        fmax=FREQUENCY_MAX,
        picks="data",
        exclude="bads",
        n_fft=n_per_seg,
        n_per_seg=n_per_seg,
        n_overlap=0,
        average="mean",
        verbose="error",
    )

    values = np.asarray(spectrum.get_data(), dtype=float)

    if values.size == 0:
        raise ValueError("no valid data channels remain")

    # Moyenne des canaux et, pour les Epochs, des epochs.
    # Le dernier axe correspond aux fréquences.
    psd_raw = np.nanmean(
        values,
        axis=tuple(range(values.ndim - 1)),
    )

    freqs = np.asarray(spectrum.freqs, dtype=float)

    if not np.all(np.isfinite(psd_raw)):
        raise ValueError("PSD contains non-finite values")

    psd_interp = np.interp(
        FREQUENCY_GRID,
        freqs,
        psd_raw,
    )

    return FREQUENCY_GRID, psd_interp

def _read_raw_file(path: Path) -> Any:
    if path.name.lower().endswith((".fif", ".fif.gz")):
        return mne.io.read_raw_fif(path, preload=False, verbose="error")
    return mne.io.read_raw_edf(path, preload=False, verbose="error")


def compute_recording_psds(
    rec: PreprocessedRecording,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute before (raw EDF/FIF) and after (FIF) PSDs for a single recording sequentially."""
    before_psd = None
    after_psd = None

    # Compute Before (Raw EDF / FIF)
    if rec.edf_path and rec.edf_path.is_file():
        raw = None
        try:
            raw = _read_raw_file(rec.edf_path)
            bads = _bad_channels(rec.edf_path, list(raw.ch_names))
            _apply_bads(raw, bads)
            _, before_psd = _compute_welch_psd(raw)
        except Exception as err:
            LOGGER.warning("Could not compute raw PSD for %s: %s", rec.edf_path, err)
        finally:
            if raw is not None:
                try:
                    raw.close()
                except Exception:
                    pass
                del raw

    # Compute After (Preprocessed FIF)
    epochs = None
    try:
        epochs = mne.read_epochs(rec.fif_path, preload=False, verbose="error")
        _, after_psd = _compute_welch_psd(epochs)
    except Exception as err:
        LOGGER.warning("Could not compute FIF PSD for %s: %s", rec.fif_path, err)
    finally:
        if epochs is not None:
            del epochs

    gc.collect()
    return before_psd, after_psd


def _positive_log_limits(values: np.ndarray) -> tuple[float, float]:
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return 1e-3, 1.0
    lower = float(positive.min())
    upper = float(positive.max())
    if lower == upper:
        return lower / 10.0, upper * 10.0
    padding = 10.0 ** (0.04 * (np.log10(upper) - np.log10(lower)))
    return lower / padding, upper * padding


def build_parameters_html(
    recordings: list[PreprocessedRecording],
) -> str:
    stats: dict[str, dict[str, Any]] = {
        "hup": {"subjects": set(), "recordings": 0, "orig_sfreqs": set()},
        "ucla": {"subjects": set(), "recordings": 0, "orig_sfreqs": set()},
    }

    final_sfreqs: set[float] = set()
    notch_freqs: list[float] = []
    l_freqs: set[Any] = set()
    h_freqs: set[Any] = set()
    epoch_durations: set[float] = set()
    epoch_overlaps: set[float] = set()
    samples_per_epochs: set[int] = set()

    for rec in recordings:
        ds = rec.dataset if rec.dataset in stats else "hup"
        stats[ds]["subjects"].add(rec.participant_id)
        stats[ds]["recordings"] += 1

        meta = rec.metadata
        if "original_sfreq" in meta:
            stats[ds]["orig_sfreqs"].add(float(meta["original_sfreq"]))
        elif "original_sampling_frequency_hz" in meta:
            stats[ds]["orig_sfreqs"].add(float(meta["original_sampling_frequency_hz"]))

        if "final_sfreq" in meta:
            final_sfreqs.add(float(meta["final_sfreq"]))
        if "configured_notch_frequencies" in meta and not notch_freqs:
            notch_freqs = meta["configured_notch_frequencies"]
        elif "applied_notch_frequencies" in meta and not notch_freqs:
            notch_freqs = meta["applied_notch_frequencies"]

        l_freqs.add(meta.get("l_freq"))
        h_freqs.add(meta.get("h_freq"))

        if "epoch_duration_seconds" in meta:
            dur = float(meta["epoch_duration_seconds"])
            epoch_durations.add(dur)
            sf = float(meta.get("final_sfreq", 1000.0))
            samples_per_epochs.add(int(round(dur * sf)))
        if "epoch_overlap_seconds" in meta:
            epoch_overlaps.add(float(meta["epoch_overlap_seconds"]))

    def _fmt_sfreqs(sf_set: set[float]) -> str:
        if not sf_set:
            return "N/A"
        return ", ".join(f"{sf:g} Hz" for sf in sorted(sf_set))

    hup_orig = _fmt_sfreqs(stats["hup"]["orig_sfreqs"])
    ucla_orig = _fmt_sfreqs(stats["ucla"]["orig_sfreqs"])
    final_sf = _fmt_sfreqs(final_sfreqs) or "1000 Hz"
    notch_str = ", ".join(f"{f:g} Hz" for f in notch_freqs) if notch_freqs else "Not available"
    dur_str = ", ".join(f"{d:g} s" for d in epoch_durations) if epoch_durations else "8 s"
    overlap_str = ", ".join(f"{o:g} s" for o in epoch_overlaps) if epoch_overlaps else "0 s"
    samples_str = ", ".join(str(s) for s in samples_per_epochs) if samples_per_epochs else "8000"

    html_content = f"""
    <div style="font-family: sans-serif; margin: 1em 0;">
      <h3>Preprocessing Parameters Summary</h3>
      <table style="border-collapse: collapse; width: 100%; max-width: 800px; text-align: left; border: 1px solid #ddd;">
        <thead>
          <tr style="background-color: #f2f2f2;">
            <th style="padding: 10px; border: 1px solid #ddd;">Parameter</th>
            <th style="padding: 10px; border: 1px solid #ddd;">HUP</th>
            <th style="padding: 10px; border: 1px solid #ddd;">UCLA</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Number of Subjects</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{len(stats['hup']['subjects'])}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{len(stats['ucla']['subjects'])}</td>
          </tr>
          <tr>
            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Number of Recordings</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{stats['hup']['recordings']}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{stats['ucla']['recordings']}</td>
          </tr>
          <tr>
            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Original Sampling Frequency</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{hup_orig}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{ucla_orig}</td>
          </tr>
          <tr>
            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Final Sampling Frequency</td>
            <td style="padding: 8px; border: 1px solid #ddd;" colspan="2">{final_sf}</td>
          </tr>
          <tr>
            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Notch Frequencies</td>
            <td style="padding: 8px; border: 1px solid #ddd;" colspan="2">{notch_str}</td>
          </tr>
          <tr>
            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Band-pass Filtering</td>
            <td style="padding: 8px; border: 1px solid #ddd;" colspan="2">None (Absence of band-pass filtering)</td>
          </tr>
          <tr>
            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Epoch Duration</td>
            <td style="padding: 8px; border: 1px solid #ddd;" colspan="2">{dur_str}</td>
          </tr>
          <tr>
            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Epoch Overlap</td>
            <td style="padding: 8px; border: 1px solid #ddd;" colspan="2">{overlap_str}</td>
          </tr>
          <tr>
            <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Samples per Epoch</td>
            <td style="padding: 8px; border: 1px solid #ddd;" colspan="2">{samples_str}</td>
          </tr>
        </tbody>
      </table>
    </div>
    """
    return html_content

def plot_dataset_psd(
    dataset_subject_psds: dict[str, np.ndarray],
    title: str,
    color: str,
    ylim: tuple[float, float],
) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(10, 5))

    if dataset_subject_psds:
        matrix = np.asarray(
            list(dataset_subject_psds.values()),
            dtype=float,
        )

        median_curve = np.nanmedian(matrix, axis=0)
        q25_curve = np.nanpercentile(matrix, 25, axis=0)
        q75_curve = np.nanpercentile(matrix, 75, axis=0)

        axis.fill_between(
            FREQUENCY_GRID,
            q25_curve,
            q75_curve,
            color=color,
            alpha=0.25,
            label="IQR across subjects",
        )

        axis.plot(
            FREQUENCY_GRID,
            median_curve,
            color=color,
            linewidth=2.0,
            label="Median across subjects",
        )

        n_subj = len(dataset_subject_psds)
        axis.set_title(f"{title} (N = {n_subj} subjects)")
    else:
        axis.set_title(f"{title} (No data)")

    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Power Spectral Density")
    axis.set_xlim(FREQUENCY_MIN, FREQUENCY_MAX)
    axis.set_ylim(ylim)
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.2)

    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    return fig


def plot_subject_before_after(
    dataset: str,
    participant_id: str,
    n_recordings: int,
    before_psd: np.ndarray | None,
    after_psd: np.ndarray | None,
    ylim: tuple[float, float],
) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(9, 4.5))

    if before_psd is not None and np.all(np.isfinite(before_psd)):
        axis.plot(FREQUENCY_GRID, before_psd, color="#4c78a8", linewidth=1.5, label="Before preprocessing (Raw EDF)")
    if after_psd is not None and np.all(np.isfinite(after_psd)):
        axis.plot(FREQUENCY_GRID, after_psd, color="#e45756", linewidth=1.5, label="After preprocessing (1kHz FIF)")

    title_str = f"[{dataset.upper()}] Participant: {participant_id} ({n_recordings} recording{'s' if n_recordings != 1 else ''})"
    axis.set_title(title_str)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Power Spectral Density")
    axis.set_xlim(FREQUENCY_MIN, FREQUENCY_MAX)
    axis.set_ylim(ylim)
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.2)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    return fig


def run_aggregation(
    preprocessed_roots: list[Path],
    output_html: Path,
    raw_root: Path | None = None,
    manifest_path: Path | None = None,
    max_subjects_per_dataset: int | None = None,
) -> Path:
    output_html.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Discovering preprocessed recordings...")
    recordings = discover_preprocessed_recordings(preprocessed_roots, raw_root, manifest_path)
    if not recordings:
        raise SystemExit("No valid preprocessed recordings found.")

    LOGGER.info("Discovered %d recordings.", len(recordings))

    # Apply max subjects filter if specified (for smoke testing)
    if max_subjects_per_dataset is not None and max_subjects_per_dataset > 0:
        dataset_subjects: dict[str, set[str]] = defaultdict(set)
        filtered_recordings: list[PreprocessedRecording] = []
        for rec in recordings:
            if len(dataset_subjects[rec.dataset]) < max_subjects_per_dataset or rec.participant_id in dataset_subjects[rec.dataset]:
                dataset_subjects[rec.dataset].add(rec.participant_id)
                filtered_recordings.append(rec)
        recordings = filtered_recordings
        LOGGER.info("Filtered to max %d subjects per dataset (%d recordings total).", max_subjects_per_dataset, len(recordings))

    # Group recordings by (dataset, participant_id)
    subject_rec_map: dict[tuple[str, str], list[PreprocessedRecording]] = defaultdict(list)
    for rec in recordings:
        subject_rec_map[(rec.dataset, rec.participant_id)].append(rec)

    # Compute PSDs per subject (hierarchical aggregation: rec -> subject)
    # Stores mean subject PSD for before and after
    subject_before_psds: dict[tuple[str, str], np.ndarray] = {}
    subject_after_psds: dict[tuple[str, str], np.ndarray] = {}
    subject_rec_counts: dict[tuple[str, str], int] = {}

    all_psd_values: list[float] = []

    total_subjects = len(subject_rec_map)
    for idx, ((ds, part_id), rec_list) in enumerate(subject_rec_map.items(), 1):
        LOGGER.info("[%d/%d] Processing subject %s (%s, %d recordings)...", idx, total_subjects, part_id, ds, len(rec_list))

        before_list: list[np.ndarray] = []
        after_list: list[np.ndarray] = []

        for rec in rec_list:
            b_psd, a_psd = compute_recording_psds(rec)
            if b_psd is not None:
                before_list.append(b_psd)
            if a_psd is not None:
                after_list.append(a_psd)

        subject_rec_counts[(ds, part_id)] = len(rec_list)

        if before_list:
            mean_b = np.mean(before_list, axis=0)
            subject_before_psds[(ds, part_id)] = mean_b
            all_psd_values.extend(mean_b)

        if after_list:
            mean_a = np.mean(after_list, axis=0)
            subject_after_psds[(ds, part_id)] = mean_a
            all_psd_values.extend(mean_a)

    # Global Y-limits across all before and after curves for consistent scale
    ylim = _positive_log_limits(np.array(all_psd_values))

    # Dataset-level aggregations
    hup_subj_before = {part_id: psd for (ds, part_id), psd in subject_before_psds.items() if ds == "hup"}
    ucla_subj_before = {part_id: psd for (ds, part_id), psd in subject_before_psds.items() if ds == "ucla"}

    hup_subj_after = {part_id: psd for (ds, part_id), psd in subject_after_psds.items() if ds == "hup"}
    ucla_subj_after = {part_id: psd for (ds, part_id), psd in subject_after_psds.items() if ds == "ucla"}

    hf_mask = (
    (FREQUENCY_GRID >= 300.0)
    & (FREQUENCY_GRID <= 500.0)
)

    hup_hf_scores = {
        participant_id: float(np.nanmedian(psd[hf_mask]))
        for participant_id, psd in hup_subj_before.items()
    }

    LOGGER.info("HUP subjects with highest median PSD from 300 to 500 Hz:")

    for participant_id, score in sorted(
        hup_hf_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:10]:
        LOGGER.info(
            "  %s: %.6e",
            participant_id,
            score,
        )

    # Build MNE Report
    report = mne.Report(title="HUP & UCLA Preprocessing QC Report", verbose="error")

    # Section 1: Preprocessing parameters
    params_html = build_parameters_html(recordings)
    report.add_html(params_html, title="Preprocessing Parameters", section="1. Preprocessing parameters")

    # Cohort provenance: recomputed derivatives plus derivatives reused in place
    # from a previous, read-only analysis root. Both are part of the cohort, so
    # the counts below describe the complete preprocessing cohort.
    provenance = provenance_summary(recordings)
    provenance_html = (
        "<p>Complete preprocessing cohort: derivatives recomputed in this analysis "
        "root plus valid derivatives reused in place (read-only) from a previous "
        "root. Each recording is counted exactly once, by recording identity.</p>"
        + provenance.to_html(index=False, border=0, na_rep="")
    )
    report.add_html(
        provenance_html,
        title="Cohort provenance (recomputed + reused)",
        section="1. Preprocessing parameters",
    )

    # Section 2: Dataset-level PSD before preprocessing
    fig_hup_before = plot_dataset_psd(hup_subj_before, "HUP Raw PSD (Before Preprocessing)", "#4c78a8", ylim)
    fig_ucla_before = plot_dataset_psd(ucla_subj_before, "UCLA Raw PSD (Before Preprocessing)", "#4c78a8", ylim)
    report.add_figure(fig_hup_before, title="HUP Mean Raw PSD", section="2. Dataset-level PSD before preprocessing")
    report.add_figure(fig_ucla_before, title="UCLA Mean Raw PSD", section="2. Dataset-level PSD before preprocessing")
    plt.close(fig_hup_before)
    plt.close(fig_ucla_before)

    # Section 3: Dataset-level PSD after preprocessing
    fig_hup_after = plot_dataset_psd(hup_subj_after, "HUP Preprocessed PSD (After Preprocessing)", "#e45756", ylim)
    fig_ucla_after = plot_dataset_psd(ucla_subj_after, "UCLA Preprocessed PSD (After Preprocessing)", "#e45756", ylim)
    report.add_figure(fig_hup_after, title="HUP Mean Preprocessed PSD", section="3. Dataset-level PSD after preprocessing")
    report.add_figure(fig_ucla_after, title="UCLA Mean Preprocessed PSD", section="3. Dataset-level PSD after preprocessing")
    plt.close(fig_hup_after)
    plt.close(fig_ucla_after)

    # Section 4: All subject-level PSD curves before preprocessing
    fig_subjects_before = plot_subject_psd_overlay(
        subject_psds=subject_before_psds,
        title="Subject-level PSD before preprocessing",
        ylim=ylim,
    )

    report.add_figure(
        fig_subjects_before,
        title="All subjects before preprocessing",
        section="4. Subject-level PSD before preprocessing",
    )

    plt.close(fig_subjects_before)


    # Section 5: All subject-level PSD curves after preprocessing
    fig_subjects_after = plot_subject_psd_overlay(
        subject_psds=subject_after_psds,
        title="Subject-level PSD after preprocessing",
        ylim=ylim,
    )

    report.add_figure(
        fig_subjects_after,
        title="All subjects after preprocessing",
        section="5. Subject-level PSD after preprocessing",
    )

    plt.close(fig_subjects_after)



    LOGGER.info("Saving MNE report to %s...", output_html)
    report.save(output_html, overwrite=True, open_browser=False, verbose="error")
    LOGGER.info("Report successfully generated.")
    return output_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        action="append",
        dest="preprocessed_roots",
        help="Root directory containing preprocessed FIF/metadata files. Can be passed multiple times.",
    )
    parser.add_argument(
        "--hup-root",
        type=Path,
        default=Path("/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/full_nonictal_v1/hup"),
        help="HUP preprocessed root directory.",
    )
    parser.add_argument(
        "--ucla-root",
        type=Path,
        default=Path("/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3/full_nonictal_v1/ucla"),
        help="UCLA preprocessed root directory.",
    )
    parser.add_argument(
        "--reuse-root",
        type=Path,
        action="append",
        dest="reuse_roots",
        default=[],
        help=(
            "Read-only previous analysis root whose valid derivatives were reused "
            "in place. Repeatable. Listed after the current roots, so a recording "
            "present in both is counted once, from the current root."
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="Optional root directory containing original EDF files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional manifest CSV containing edf_path mappings.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("hup_ucla_preprocessing_qc_report.html"),
        help="Path for output HTML report.",
    )
    parser.add_argument(
        "--max-subjects-per-dataset",
        type=int,
        help="Maximum number of subjects to process per dataset (e.g. 2 for smoke testing).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level.",
    )
    return parser.parse_args()

def plot_subject_psd_overlay(
    subject_psds: dict[tuple[str, str], np.ndarray],
    title: str,
    ylim: tuple[float, float],
) -> plt.Figure:
    """Plot one PSD curve per subject on a single figure."""

    fig, axis = plt.subplots(figsize=(12, 6))

    valid_subjects: list[tuple[tuple[str, str], np.ndarray]] = []

    for subject_key, psd in sorted(subject_psds.items()):
        psd = np.asarray(psd, dtype=float)

        if psd.shape != FREQUENCY_GRID.shape:
            LOGGER.warning(
                "Ignoring %s because PSD shape %s does not match grid shape %s.",
                subject_key,
                psd.shape,
                FREQUENCY_GRID.shape,
            )
            continue

        if not np.all(np.isfinite(psd)):
            LOGGER.warning(
                "Ignoring %s because its PSD contains non-finite values.",
                subject_key,
            )
            continue

        valid_subjects.append((subject_key, psd))

    # Une courbe par participant, sans légende individuelle
    for (_, _), psd in valid_subjects:
        axis.plot(
            FREQUENCY_GRID,
            psd,
            linewidth=0.8,
            alpha=0.35,
        )

    n_subjects = len(valid_subjects)

    axis.set_title(f"{title} ({n_subjects} subjects)")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("PSD")
    axis.set_xlim(FREQUENCY_MIN, FREQUENCY_MAX)
    axis.set_ylim(ylim)
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    roots: list[Path] = []
    if args.preprocessed_roots:
        roots.extend(args.preprocessed_roots)
    else:
        roots.extend([args.hup_root, args.ucla_root])
    # Derivatives reused in place from a previous analysis root are part of the
    # cohort; without them the report silently covers only what was recomputed.
    roots.extend(args.reuse_roots or [])

    roots = [r.expanduser().resolve() for r in roots]

    run_aggregation(
        preprocessed_roots=roots,
        output_html=args.output_html.expanduser().resolve(),
        raw_root=args.raw_root.expanduser().resolve() if args.raw_root else None,
        manifest_path=args.manifest.expanduser().resolve() if args.manifest else None,
        max_subjects_per_dataset=args.max_subjects_per_dataset,
    )


if __name__ == "__main__":
    main()
