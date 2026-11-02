"""Custom node for the 1-kHz, no-band-pass workflow (HUP, UCLA, OpenIEEG)."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import matplotlib

matplotlib.use("Agg", force=False)

import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import welch

from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import register_node


MODULE_DIRECTORY = Path(__file__).resolve().parent
VALID_IEEG_TYPES = frozenset({"SEEG", "ECOG", "EEG", "IEEG"})
REPORT_WELCH_WINDOW_SECONDS = 8.0
REPORT_WELCH_OVERLAP_SECONDS = 0.0


def _load_line_noise_module():
    module_name = "line_noise_config"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = MODULE_DIRECTORY / "line_noise_config.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load line-noise configuration module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class IneligibleSamplingFrequencyError(ValueError):
    """Raised before signal loading when the original rate is ineligible."""

    def __init__(self, sampling_frequency_hz: float, minimum_hz: float):
        self.metadata = {
            "original_sfreq": sampling_frequency_hz,
            "eligibility_decision": "excluded",
            "exclusion_reason": "original_sampling_frequency_below_minimum",
            "min_original_sfreq": minimum_hz,
        }
        super().__init__(
            f"original sampling frequency {sampling_frequency_hz:g} Hz is below "
            f"the required {minimum_hz:g} Hz"
        )


def _read_raw_header(path: str | os.PathLike[str]):
    path = Path(path)
    if path.name.lower().endswith((".fif", ".fif.gz")):
        return mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    return mne.io.read_raw_edf(path, preload=False, verbose="ERROR")


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def _channels_sidecar(source_path: Path) -> Path:
    source_path = source_path.resolve()
    name = source_path.name
    lower_name = name.lower()
    for suffix in ("_ieeg.edf", "_ieeg.fif.gz", "_ieeg.fif"):
        if lower_name.endswith(suffix):
            return source_path.with_name(name[: -len(suffix)] + "_channels.tsv")
    return source_path.with_suffix("").with_name(source_path.stem + "_channels.tsv")


def _retain_valid_ieeg_and_apply_bads(
    raw, source_path: Path
) -> tuple[int, list[str], pd.DataFrame]:
    """Retain iEEG channels, mark sidecar bads, then remove only those bads."""
    sidecar = _channels_sidecar(source_path)
    bad_names: list[str] = []
    if sidecar.is_file():
        channels = pd.read_csv(sidecar, sep="\t")
        if not {"name", "type"}.issubset(channels.columns):
            raise ValueError(f"channels.tsv lacks name/type columns: {sidecar}")
        types = channels["type"].fillna("").astype(str).str.upper()
        valid_rows = types.isin(VALID_IEEG_TYPES)
        valid_names = set(channels.loc[valid_rows, "name"].astype(str))
        picks = [name for name in raw.ch_names if name in valid_names]
        if "status" in channels.columns:
            bad_rows = channels["status"].fillna("good").astype(str).str.lower().eq("bad")
        else:
            bad_rows = pd.Series(False, index=channels.index)
        if "good" in channels.columns:
            bad_rows |= ~channels["good"].map(_truthy)
        flagged = set(channels.loc[valid_rows & bad_rows, "name"].astype(str))
        bad_names = [name for name in picks if name in flagged]
    else:
        channel_types = raw.get_channel_types()
        picks = [
            name
            for name, channel_type in zip(raw.ch_names, channel_types)
            if channel_type.upper() in VALID_IEEG_TYPES
        ]
        bad_names = [name for name in raw.info.get("bads", []) if name in picks]
        channels = pd.DataFrame(
            {
                "name": raw.ch_names,
                "type": [channel_type.upper() for channel_type in channel_types],
                "status": [
                    "bad" if name in raw.info.get("bads", []) else "good"
                    for name in raw.ch_names
                ],
            }
        )
    if not picks:
        raise ValueError("recording contains no valid iEEG channels")
    raw.pick(picks)
    raw.info["bads"] = bad_names
    if bad_names:
        raw.drop_channels(bad_names)
    if not raw.ch_names:
        raise ValueError("all valid iEEG channels are marked bad")
    retained = channels[channels["name"].astype(str).isin(raw.ch_names)].copy()
    retained["_output_order"] = retained["name"].astype(str).map(
        {name: index for index, name in enumerate(raw.ch_names)}
    )
    retained = retained.sort_values("_output_order").drop(columns="_output_order")
    return len(picks), bad_names, retained.reset_index(drop=True)


def _load_manifest_row(manifest_path: Path, source_path: Path) -> dict[str, str]:
    source_resolved = source_path.resolve()
    matches: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            candidates = [row.get("source_edf_path"), row.get("edf_path")]
            if any(
                candidate
                and Path(candidate).expanduser().resolve() == source_resolved
                for candidate in candidates
            ):
                matches.append(dict(row))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one manifest row for {source_resolved}, found {len(matches)}"
        )
    return matches[0]


def _resolve_configuration_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else MODULE_DIRECTORY / candidate


def _load_bids_metadata(source_path: Path, participant_id: str) -> dict[str, Any]:
    """Preserve participant and recording sidecar values when present."""
    source_path = source_path.resolve()
    participant_metadata: dict[str, Any] = {}
    for parent in source_path.parents:
        participants_path = parent / "participants.tsv"
        if not participants_path.is_file():
            continue
        participants = pd.read_csv(participants_path, sep="\t")
        if "participant_id" in participants.columns:
            matching = participants[
                participants["participant_id"].astype(str).eq(participant_id)
            ]
            if len(matching) == 1:
                participant_metadata = matching.iloc[0].to_dict()
        break

    channels_path = _channels_sidecar(source_path)
    recording_json_path = channels_path.with_name(
        channels_path.name.removesuffix("_channels.tsv") + "_ieeg.json"
    )
    recording_metadata: dict[str, Any] = {}
    if recording_json_path.is_file():
        loaded = json.loads(recording_json_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"recording JSON sidecar is not an object: {recording_json_path}")
        recording_metadata = loaded
    return {
        "participant_metadata": participant_metadata,
        "recording_sidecar_metadata": recording_metadata,
    }


def _spectrum_figure(raw, title: str, applied_notches: tuple[float, ...]):
    sfreq = float(raw.info["sfreq"])
    n_per_seg = int(round(REPORT_WELCH_WINDOW_SECONDS * sfreq))
    n_overlap = int(round(REPORT_WELCH_OVERLAP_SECONDS * sfreq))
    if raw.n_times < n_per_seg:
        raise ValueError("recording is too short for an 8-second report PSD window")
    frequencies, channel_psd = welch(
        raw.get_data(),
        fs=sfreq,
        window="hann",
        nperseg=n_per_seg,
        noverlap=n_overlap,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    resolution = sfreq / n_per_seg
    maximum = min(500.0, sfreq / 2.0 - resolution)
    keep = (frequencies >= 0.5) & (frequencies <= maximum) & (frequencies < 500.0)
    central = np.median(channel_psd[:, keep], axis=0)
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(frequencies[keep], central, linewidth=1.0)
    for frequency in applied_notches:
        axis.axvline(
            frequency,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label=f"Applied notch {frequency:g} Hz",
        )
    axis.set_yscale("log")
    axis.set_xlim(0.5, float(frequencies[keep][-1]))
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Power spectral density")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    if applied_notches:
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), loc="best")
    figure.tight_layout()
    return figure, resolution


def _jsonable_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            result[str(key)] = ""
        elif isinstance(value, (str, int, float, bool)):
            result[str(key)] = value
        else:
            result[str(key)] = json.dumps(value, sort_keys=True, default=_json_default)
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


@register_node
def preprocess_1khz_no_bandpass_report(
    source_path,
    site: str,
    dataset: str,
    eligible_manifest_path,
    line_noise_config="config/line_noise_hup_ucla.yml",
    min_original_sfreq: float = 1000.0,
    target_sfreq: float = 1000.0,
    l_freq=None,
    h_freq=None,
    epoch_duration_seconds: float = 8.0,
    epoch_overlap_seconds: float = 0.0,
    _raw_reader: Callable[[str | os.PathLike[str]], Any] | None = None,
    _recording_metadata: Mapping[str, Any] | None = None,
) -> NodeResult:
    """Notch raw continuous iEEG, resample to 1 kHz, and make 8-s epochs."""
    if l_freq is not None or h_freq is not None:
        raise ValueError("l_freq and h_freq must both be null for this protocol")
    if epoch_overlap_seconds != 0:
        raise ValueError("epoch_overlap_seconds must be zero for this protocol")
    if epoch_duration_seconds <= 0:
        raise ValueError("epoch_duration_seconds must be positive")

    indexed_source = Path(source_path).expanduser()
    real_source = indexed_source.resolve()
    reader = _raw_reader or _read_raw_header
    raw = reader(real_source)
    original_sfreq = float(raw.info["sfreq"])
    if not math.isfinite(original_sfreq) or original_sfreq < float(min_original_sfreq):
        close = getattr(raw, "close", None)
        if callable(close):
            close()
        raise IneligibleSamplingFrequencyError(original_sfreq, float(min_original_sfreq))

    try:
        raw.load_data(verbose="ERROR")
        number_of_input_channels = len(raw.ch_names)
        _, bad_channels, retained_channel_metadata = _retain_valid_ieeg_and_apply_bads(
            raw, real_source
        )
        number_of_output_channels = len(raw.ch_names)

        normalized_site = str(site).strip().lower()
        normalized_dataset = str(dataset).strip().lower()
        if (
            normalized_site not in {"hup", "ucla", "openieeg"}
            or normalized_dataset != normalized_site
        ):
            raise ValueError(
                "site/dataset must be matching normalized hup, ucla or openieeg keys: "
                f"{site!r}/{dataset!r}"
            )
        if _recording_metadata is None:
            manifest_path = Path(eligible_manifest_path).expanduser()
            recording_metadata = _load_manifest_row(manifest_path, real_source)
        else:
            recording_metadata = dict(_recording_metadata)
        manifest_site = str(recording_metadata.get("dataset", "")).strip().lower()
        manifest_sfreq = float(
            recording_metadata.get("original_sampling_frequency_hz", original_sfreq)
        )
        if manifest_site != normalized_site:
            raise ValueError(
                f"manifest site {manifest_site!r} disagrees with configured site "
                f"{normalized_site!r}"
            )
        if not math.isclose(manifest_sfreq, original_sfreq, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"manifest/header sampling-frequency disagreement: {manifest_sfreq:g} "
                f"versus {original_sfreq:g} Hz"
            )
        if "eligible_sfreq_ge_1000" in recording_metadata and not _truthy(
            recording_metadata["eligible_sfreq_ge_1000"]
        ):
            raise ValueError("manifest row is not marked sampling-frequency eligible")

        line_noise = _load_line_noise_module()
        reviewed = line_noise.load_line_noise_config(
            _resolve_configuration_path(line_noise_config),
            required_sites={normalized_site},
        )
        configured_notches = reviewed.frequencies_for(normalized_site)
        nyquist_report = line_noise.report_frequencies_at_or_above_nyquist(
            reviewed, site=normalized_site, sampling_frequency_hz=original_sfreq
        )
        skipped_notches = nyquist_report.frequencies_at_or_above_nyquist_hz
        applied_notches = tuple(
            frequency for frequency in configured_notches if frequency < original_sfreq / 2.0
        )

        report = mne.Report(
            title=f"{normalized_site.upper()} 1 kHz preprocessing", verbose="error"
        )
        before_figure, before_resolution = _spectrum_figure(
            raw, "Raw PSD before notch", applied_notches
        )
        try:
            report.add_figure(before_figure, title="Raw PSD before notch")
        finally:
            plt.close(before_figure)

        if applied_notches:
            raw.notch_filter(
                freqs=np.asarray(applied_notches),
                method="fir",
                phase="zero",
                fir_window="hamming",
                fir_design="firwin",
                verbose="ERROR",
            )
        after_notch_figure, notch_resolution = _spectrum_figure(
            raw, "PSD after notch", applied_notches
        )
        try:
            report.add_figure(after_notch_figure, title="PSD after notch")
        finally:
            plt.close(after_notch_figure)

        if not math.isclose(original_sfreq, float(target_sfreq), rel_tol=0.0, abs_tol=1e-6):
            raw.resample(float(target_sfreq), verbose="ERROR")
        final_sfreq = float(raw.info["sfreq"])
        after_resample_figure, resampled_resolution = _spectrum_figure(
            raw, "PSD after resampling", applied_notches
        )
        try:
            report.add_figure(after_resample_figure, title="PSD after resampling")
        finally:
            plt.close(after_resample_figure)

        samples_per_epoch = int(round(epoch_duration_seconds * final_sfreq))
        number_of_complete_epochs = int(raw.n_times // samples_per_epoch)
        if number_of_complete_epochs < 1:
            raise ValueError("recording has no complete 8-second epoch")
        used_samples = number_of_complete_epochs * samples_per_epoch
        dropped_final_duration_seconds = (raw.n_times - used_samples) / final_sfreq

        metadata: dict[str, Any] = dict(recording_metadata)
        preserved_bids_metadata = _load_bids_metadata(
            real_source, str(recording_metadata.get("participant_id", ""))
        )
        metadata.update(
            {
                "protocol_version": reviewed.protocol_version,
                "participant_id": str(recording_metadata.get("participant_id", "")),
                "dataset": normalized_dataset,
                "site": normalized_site,
                "institution": str(recording_metadata.get("institution", "")),
                "session": str(recording_metadata.get("session", "")),
                "task": str(recording_metadata.get("task", "")),
                "run": str(recording_metadata.get("run", "")),
                "original_sfreq": original_sfreq,
                "final_sfreq": final_sfreq,
                "eligibility_decision": "eligible",
                "exclusion_reason": "",
                "configured_notch_frequencies": list(configured_notches),
                "applied_notch_frequencies": list(applied_notches),
                "skipped_notch_frequencies": list(skipped_notches),
                "l_freq": None,
                "h_freq": None,
                "epoch_duration_seconds": float(epoch_duration_seconds),
                "epoch_overlap_seconds": float(epoch_overlap_seconds),
                "number_of_input_channels": number_of_input_channels,
                "number_of_output_channels": number_of_output_channels,
                "number_of_complete_epochs": number_of_complete_epochs,
                "dropped_final_duration_seconds": dropped_final_duration_seconds,
                "bad_channels_removed": bad_channels,
                "notch_method": "fir",
                "notch_phase": "zero",
                "notch_fir_window": "hamming",
                "notch_fir_design": "firwin",
                "channel_metadata": retained_channel_metadata.to_dict(orient="records"),
                **preserved_bids_metadata,
                "report_welch_n_per_seg_before_notch": int(
                    round(REPORT_WELCH_WINDOW_SECONDS * original_sfreq)
                ),
                "report_welch_n_overlap": 0,
                "report_welch_resolution_hz_before_notch": before_resolution,
                "report_welch_resolution_hz_after_notch": notch_resolution,
                "report_welch_resolution_hz_after_resampling": resampled_resolution,
            }
        )
        epoch_metadata_row = _jsonable_metadata(metadata)
        epoch_metadata = pd.DataFrame(
            [epoch_metadata_row.copy() for _ in range(number_of_complete_epochs)]
        )
        events = np.column_stack(
            [
                raw.first_samp + np.arange(number_of_complete_epochs) * samples_per_epoch,
                np.zeros(number_of_complete_epochs, dtype=int),
                np.ones(number_of_complete_epochs, dtype=int),
            ]
        ).astype(int)
        epochs = mne.Epochs(
            raw,
            events,
            event_id={"window": 1},
            tmin=0.0,
            tmax=(samples_per_epoch - 1) / final_sfreq,
            baseline=None,
            preload=True,
            reject_by_annotation=False,
            metadata=epoch_metadata,
            verbose="ERROR",
        )
        if len(epochs) != number_of_complete_epochs or epochs.get_data().shape[-1] != samples_per_epoch:
            raise RuntimeError("epoch construction did not preserve complete fixed-length windows")

        report_payload = {
            "protocol_version": reviewed.protocol_version,
            "site": normalized_site,
            "eligibility_decision": "eligible",
            "original_sampling_frequency_hz": original_sfreq,
            "final_sampling_frequency_hz": final_sfreq,
            "configured_notch_frequencies_hz": list(configured_notches),
            "applied_notch_frequencies_hz": list(applied_notches),
            "skipped_at_or_above_nyquist_hz": list(skipped_notches),
            "welch_window_seconds": REPORT_WELCH_WINDOW_SECONDS,
            "welch_overlap_seconds": REPORT_WELCH_OVERLAP_SECONDS,
            "epoch_duration_seconds": float(epoch_duration_seconds),
            "epoch_overlap_seconds": float(epoch_overlap_seconds),
            "number_of_complete_epochs": number_of_complete_epochs,
            "dropped_final_duration_seconds": dropped_final_duration_seconds,
        }
        report.add_html(
            f"<pre>{json.dumps(report_payload, indent=2)}</pre>",
            title="Protocol and recording metadata",
        )
        metadata_json = json.dumps(
            metadata, indent=2, sort_keys=True, default=_json_default
        )
        return NodeResult(
            artifacts={
                ".fif": Artifact(
                    item=epochs,
                    writer=lambda path, value=epochs: value.save(
                        path, overwrite=True, verbose="ERROR"
                    ),
                ),
                ".report.html": Artifact(
                    item=report,
                    writer=lambda path, value=report: value.save(
                        path, overwrite=True, open_browser=False
                    ),
                ),
                ".metadata.json": Artifact(
                    item=metadata,
                    writer=lambda path, value=metadata_json: Path(path).write_text(
                        value + "\n", encoding="utf-8"
                    ),
                ),
                ".channels.tsv": Artifact(
                    item=retained_channel_metadata,
                    writer=lambda path, value=retained_channel_metadata: value.to_csv(
                        path, sep="\t", index=False
                    ),
                ),
            }
        )
    finally:
        close = getattr(raw, "close", None)
        if callable(close):
            close()
