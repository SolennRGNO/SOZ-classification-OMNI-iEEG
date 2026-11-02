#!/usr/bin/env python3
"""Generate a standalone QC and exploratory HTML report for relative bandpower.

This is a read-only analysis step. It consumes the epoch-level and channel-level
Parquet files plus the extraction QC CSV and never modifies feature tables.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import logging
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

LOGGER = logging.getLogger("hup_ucla_relative_bandpower_report")
REPORT_PROTOCOL_VERSION = "1.0.0"
SITE_ORDER = ("hup", "ucla")
CLASS_LABELS = {0.0: "Normal", 1.0: "Pathological"}
CLASS_COLORS = {0.0: "#3182bd", 1.0: "#de2d26"}

PAPER_FEATURES = {
    "Delta": "rel_paper_delta_0p5_4",
    "Theta": "rel_paper_theta_4_8",
    "Alpha": "rel_paper_alpha_8_13",
    "Beta": "rel_paper_beta_13_30",
    "Gamma": "rel_paper_gamma_30_80",
    "High-gamma": "rel_paper_high_gamma_80_150",
}
EXTENDED_FEATURES = {
    "Delta": "rel_ext_delta_0p5_4",
    "Theta": "rel_ext_theta_4_8",
    "Alpha": "rel_ext_alpha_8_13",
    "Beta": "rel_ext_beta_13_30",
    "Gamma": "rel_ext_gamma_30_80",
    "High-gamma": "rel_ext_high_gamma_80_150",
    "150-250": "rel_ext_hf_150_250",
    "250-350": "rel_ext_hf_250_350",
    "350-450": "rel_ext_hf_350_450",
}
def _site_column(frame: pd.DataFrame) -> str:
    if "site" in frame.columns:
        return "site"
    if "dataset" in frame.columns:
        return "dataset"
    raise ValueError("input table has neither a site nor a dataset column")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _require_columns(frame: pd.DataFrame, columns: set[str], table: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{table} is missing required columns: {', '.join(missing)}")


def _figure_uri(figure: plt.Figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=145, bbox_inches="tight")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _figure_html(figure: plt.Figure, alt: str) -> str:
    return (
        f'<figure><img src="{_figure_uri(figure)}" alt="{html.escape(alt)}">'
        f"<figcaption>{html.escape(alt)}</figcaption></figure>"
    )


def _table_html(frame: pd.DataFrame, *, max_height: int | None = None) -> str:
    if frame.empty:
        return '<p class="notice">No compatible data available.</p>'
    style = f' style="max-height:{max_height}px"' if max_height else ""
    return (
        f'<div class="table-wrap"{style}>'
        + frame.to_html(index=False, border=0, escape=True, na_rep="")
        + "</div>"
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".html",
        prefix=f".{path.name}.",
        dir=path.parent,
        encoding="utf-8",
        delete=False,
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_site_qc(qc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate technical extraction QC without hiding the recording-level table."""
    site_col = _site_column(qc)
    rows: list[dict[str, Any]] = []
    for site, group in qc.groupby(site_col, dropna=False, sort=True):
        statuses = group.get("status", pd.Series("", index=group.index)).astype(str).str.lower()
        rows.append(
            {
                "site": str(site).lower(),
                "recordings_total": len(group),
                "recordings_completed": int((statuses == "completed").sum()),
                "recordings_failed": int((statuses != "completed").sum()),
                "channels_input": int(_numeric(group, "n_channels_input").fillna(0).sum()),
                "channels_used": int(_numeric(group, "n_channels_used").fillna(0).sum()),
                "channels_excluded": int(
                    (_numeric(group, "n_channels_input") - _numeric(group, "n_channels_used"))
                    .clip(lower=0)
                    .fillna(0)
                    .sum()
                ),
                "epochs_input": int(_numeric(group, "n_epochs_input").fillna(0).sum()),
                "epochs_valid": int(_numeric(group, "n_epochs_valid").fillna(0).sum()),
                "epochs_rejected": int(_numeric(group, "n_epochs_rejected").fillna(0).sum()),
                "nan_features": int(_numeric(group, "n_nan_features").fillna(0).sum()),
                "infinite_features": int(_numeric(group, "n_infinite_features").fillna(0).sum()),
                "flat_channel_epochs": int(_numeric(group, "n_flat_channel_epochs").fillna(0).sum()),
                "non_finite_channel_epochs": int(
                    _numeric(group, "n_non_finite_channel_epochs").fillna(0).sum()
                ),
                "paper_sum_max_abs_error": float(
                    _numeric(group, "paper_relative_sum_max_abs_error").max()
                ),
                "extended_sum_max_abs_error": float(
                    _numeric(group, "extended_relative_sum_max_abs_error").max()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_qc(site_qc: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sites = site_qc["site"].astype(str).str.upper().tolist()
    x = np.arange(len(site_qc))

    axes[0, 0].bar(x, site_qc["recordings_completed"], label="Completed", color="#31a354")
    axes[0, 0].bar(
        x,
        site_qc["recordings_failed"],
        bottom=site_qc["recordings_completed"],
        label="Failed",
        color="#de2d26",
    )
    axes[0, 0].set_title("Recording status")
    axes[0, 0].legend()

    axes[0, 1].bar(x, site_qc["channels_used"], label="Used", color="#3182bd")
    axes[0, 1].bar(
        x,
        site_qc["channels_excluded"],
        bottom=site_qc["channels_used"],
        label="Excluded",
        color="#bdbdbd",
    )
    axes[0, 1].set_title("Channels")
    axes[0, 1].legend()

    axes[1, 0].bar(x, site_qc["epochs_valid"], label="Valid", color="#756bb1")
    axes[1, 0].bar(
        x,
        site_qc["epochs_rejected"],
        bottom=site_qc["epochs_valid"],
        label="Rejected",
        color="#fd8d3c",
    )
    axes[1, 0].set_title("Eight-second epochs")
    axes[1, 0].legend()

    width = 0.36
    paper_error = site_qc["paper_sum_max_abs_error"].fillna(0).to_numpy(float)
    extended_error = site_qc["extended_sum_max_abs_error"].fillna(0).to_numpy(float)
    axes[1, 1].bar(x - width / 2, paper_error, width, label="Paper")
    axes[1, 1].bar(x + width / 2, extended_error, width, label="Extended")
    axes[1, 1].axhline(1e-3, color="black", linestyle="--", linewidth=1, label="Tolerance 1e-3")
    axes[1, 1].set_yscale("symlog", linthresh=1e-7)
    axes[1, 1].set_title("Maximum |relative sum - 1|")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.set_xticks(x, sites)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Technical extraction QC by site", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_spectral_composition(channels: pd.DataFrame) -> plt.Figure:
    site_col = _site_column(channels)
    fig, axes = plt.subplots(len(SITE_ORDER), 1, figsize=(12, 7), squeeze=False)
    colors = plt.get_cmap("tab10").colors
    for row_index, site in enumerate(SITE_ORDER):
        axis = axes[row_index, 0]
        subset = channels[channels[site_col].astype(str).str.lower() == site]
        bottoms = np.zeros(2, dtype=float)
        families = (("Paper (0.5-150)", PAPER_FEATURES), ("Extended (0.5-450)", EXTENDED_FEATURES))
        for band_index, (band, _) in enumerate(EXTENDED_FEATURES.items()):
            values = []
            for _, mapping in families:
                base = mapping.get(band)
                values.append(float(_numeric(subset, f"{base}_mean").mean()) if base else 0.0)
            axis.bar(
                [0, 1],
                values,
                bottom=bottoms,
                label=band,
                color=colors[band_index % len(colors)],
            )
            bottoms += np.nan_to_num(values)
        axis.set_xticks([0, 1], [families[0][0], families[1][0]])
        axis.set_ylim(0, max(1.02, float(np.nanmax(bottoms)) * 1.05 if len(bottoms) else 1.02))
        axis.set_ylabel("Mean relative power")
        axis.set_title(f"{site.upper()} — channel means")
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.12, 0.5))
    fig.suptitle("Paper versus extended spectral composition", fontweight="bold")
    fig.tight_layout()
    return fig


def _boxplot_two_classes(axis: plt.Axes, frame: pd.DataFrame, column: str, title: str) -> None:
    values = [
        _numeric(frame[frame["supervised_label"] == label], column).dropna().to_numpy(float)
        for label in (0.0, 1.0)
    ]
    if not any(len(item) for item in values):
        axis.text(0.5, 0.5, "No labeled data", ha="center", va="center")
        axis.set_axis_off()
        return
    axis.boxplot(values, tick_labels=["Normal", "Pathological"], showfliers=False)
    for position, (label, samples) in enumerate(zip((0.0, 1.0), values), start=1):
        if len(samples):
            offsets = np.linspace(-0.08, 0.08, len(samples)) if len(samples) > 1 else np.array([0.0])
            axis.scatter(
                position + offsets,
                samples,
                s=9,
                alpha=0.35,
                color=CLASS_COLORS[label],
                edgecolors="none",
            )
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.2)


def plot_class_comparison(channels: pd.DataFrame, site: str) -> plt.Figure:
    site_col = _site_column(channels)
    subset = channels[channels[site_col].astype(str).str.lower() == site]
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for axis, (band, feature) in zip(axes.flat, EXTENDED_FEATURES.items()):
        _boxplot_two_classes(axis, subset, f"{feature}_median", band)
        axis.set_ylabel("Extended relative power")
    fig.suptitle(
        f"{site.upper()} — normal versus pathological (extended denominator)",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def compute_subject_deltas(channels: pd.DataFrame) -> pd.DataFrame:
    """Return per-patient pathological-minus-normal channel medians."""
    site_col = _site_column(channels)
    rows: list[dict[str, Any]] = []
    for (site, participant), group in channels.groupby(
        [site_col, "participant_id"], dropna=False, sort=True
    ):
        normal = group[group["supervised_label"] == 0.0]
        pathological = group[group["supervised_label"] == 1.0]
        if normal.empty or pathological.empty:
            continue
        row: dict[str, Any] = {
            "site": str(site).lower(),
            "participant_id": str(participant),
            "n_normal_channels": len(normal),
            "n_pathological_channels": len(pathological),
        }
        for band, feature in EXTENDED_FEATURES.items():
            column = f"{feature}_median"
            row[band] = float(_numeric(pathological, column).median() - _numeric(normal, column).median())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_subject_deltas(deltas: pd.DataFrame, site: str) -> plt.Figure:
    subset = deltas[deltas["site"] == site].sort_values("participant_id").reset_index(drop=True)
    fig_height = max(3.5, min(28.0, 0.24 * max(len(subset), 1) + 2.0))
    fig, axis = plt.subplots(figsize=(12, fig_height))
    if subset.empty:
        axis.text(0.5, 0.5, "No patient has both labeled classes", ha="center", va="center")
        axis.set_axis_off()
        return fig
    matrix = subset[list(EXTENDED_FEATURES)].to_numpy(float)
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = float(np.quantile(finite, 0.98)) if finite.size else 1.0
    limit = max(limit, 1e-9)
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    axis.set_xticks(np.arange(len(EXTENDED_FEATURES)), list(EXTENDED_FEATURES), rotation=35, ha="right")
    if len(subset) <= 80:
        axis.set_yticks(np.arange(len(subset)), subset["participant_id"], fontsize=7)
    else:
        step = math.ceil(len(subset) / 60)
        positions = np.arange(0, len(subset), step)
        axis.set_yticks(positions, subset.loc[positions, "participant_id"], fontsize=6)
    axis.set_title(f"{site.upper()} — Δ = median(pathological) - median(normal)")
    fig.colorbar(image, ax=axis, label="Δ relative power", shrink=0.6)
    fig.tight_layout()
    return fig


def _recording_qc_table(qc: pd.DataFrame) -> pd.DataFrame:
    desired = [
        "dataset",
        "site",
        "participant_id",
        "recording_id",
        "status",
        "exclusion_reason",
        "n_channels_input",
        "n_channels_used",
        "n_epochs_input",
        "n_epochs_valid",
        "n_epochs_rejected",
        "n_nan_features",
        "n_infinite_features",
        "n_flat_channel_epochs",
        "n_non_finite_channel_epochs",
        "paper_relative_sum_mean",
        "paper_relative_sum_max_abs_error",
        "extended_relative_sum_mean",
        "extended_relative_sum_max_abs_error",
    ]
    return qc[[column for column in desired if column in qc.columns]].copy()


def per_task_counts(channels: pd.DataFrame) -> pd.DataFrame:
    """Per-task cohort counts from the channel table.

    The feature cohort deliberately holds interictal and ictal recordings in one
    extraction tree, so the split is reported rather than materialised as
    separate directories.
    """
    if channels.empty:
        return pd.DataFrame()
    frame = channels.copy()
    for column in ("recording_state", "task"):
        if column not in frame.columns:
            return pd.DataFrame(
                [{"note": f"channel table has no {column}; re-run the assembly stage"}]
            )
        frame[column] = frame[column].astype(str).str.strip().str.lower()
    site = "site" if "site" in frame.columns else "dataset"
    aggregations = {
        "participants": ("participant_id", "nunique"),
        "recordings": ("recording_id", "nunique"),
        "channels": ("channel", "nunique"),
        "channel_rows": ("channel", "size"),
    }
    if "n_valid_epochs" in frame.columns:
        aggregations["channel_epochs"] = ("n_valid_epochs", "sum")
    grouped = (
        frame.groupby(["recording_state", "task", site], sort=True, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )
    totals = (
        frame.groupby(["recording_state"], sort=True, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )
    totals["task"] = "(all tasks)"
    totals[site] = "(all sites)"
    return pd.concat([grouped, totals[grouped.columns]], ignore_index=True)


def generate_report(
    epoch_parquet: Path,
    channel_parquet: Path,
    qc_csv: Path,
    run_summary_json: Path,
    output_html: Path,
) -> Path:
    """Generate the report and return its path."""
    for path in (epoch_parquet, channel_parquet, qc_csv, run_summary_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    channels = pd.read_parquet(channel_parquet)
    qc = pd.read_csv(qc_csv)
    summary = json.loads(run_summary_json.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("run summary JSON must contain an object")

    required_channel_columns = {
        "participant_id",
        "recording_id",
        "channel",
        "supervised_label",
        "n_valid_epochs",
        *(f"{feature}_mean" for feature in PAPER_FEATURES.values()),
        *(f"{feature}_mean" for feature in EXTENDED_FEATURES.values()),
        *(f"{feature}_median" for feature in EXTENDED_FEATURES.values()),
    }
    _require_columns(channels, required_channel_columns, "channel-level Parquet")
    _require_columns(
        qc,
        {"participant_id", "recording_id", "status", "n_epochs_valid", "n_epochs_rejected"},
        "extraction QC CSV",
    )
    channels["supervised_label"] = pd.to_numeric(channels["supervised_label"], errors="coerce")

    site_qc = build_site_qc(qc)
    deltas = compute_subject_deltas(channels)

    label_counts = (
        channels.assign(
            site=channels[_site_column(channels)].astype(str).str.lower(),
            class_name=channels["supervised_label"].map(CLASS_LABELS).fillna("Unlabeled/ambiguous"),
        )
        .groupby(["site", "class_name"], dropna=False)
        .size()
        .reset_index(name="channels")
    )
    provenance = pd.DataFrame(
        [
            {"parameter": "Report protocol", "value": REPORT_PROTOCOL_VERSION},
            {"parameter": "Extraction protocol", "value": summary.get("protocol_version", "")},
            {"parameter": "Preprocessing protocol", "value": summary.get("expected_preprocessing_protocol_version", "")},
            {"parameter": "Git commit", "value": summary.get("git_commit", "")},
            {"parameter": "Cohort SHA-256", "value": summary.get("cohort_csv_sha256", "")},
            {"parameter": "Epoch Parquet", "value": str(epoch_parquet.resolve())},
            {"parameter": "Channel Parquet", "value": str(channel_parquet.resolve())},
            {"parameter": "QC CSV", "value": str(qc_csv.resolve())},
        ]
    )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HUP/UCLA relative bandpower extraction report</title>
<style>
body{{font-family:Arial,sans-serif;margin:0;color:#222;background:#f5f6f8;line-height:1.45}}
header{{background:#172554;color:white;padding:28px 5vw}}
main{{max-width:1500px;margin:auto;padding:24px}}
section{{background:white;padding:22px;margin:18px 0;border-radius:8px;box-shadow:0 1px 4px #0002}}
h1,h2,h3{{margin-top:0}} figure{{margin:18px 0}} img{{max-width:100%;height:auto;display:block;margin:auto}}
figcaption{{text-align:center;color:#555;font-size:.9rem;margin-top:6px}}
.table-wrap{{overflow:auto}} table{{border-collapse:collapse;width:100%;font-size:.86rem}}
th,td{{padding:6px 8px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} th{{position:sticky;top:0;background:#e9eef8}}
.notice{{padding:10px;background:#edf2f7}} .warning{{color:#a61b1b}} code{{word-break:break-all}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}}
</style></head><body>
<header><h1>Relative bandpower extraction report — HUP &amp; UCLA</h1>
<p>QC and exploratory analysis between feature extraction and baseline ML.</p>
<p>Generated {html.escape(datetime.now(timezone.utc).isoformat())}</p></header><main>
<section id="scope"><h2>Scope and provenance</h2>
<p>This report is diagnostic. It does not replace or modify either Parquet file.</p>
{_table_html(provenance)}</section>

<section id="technical-qc"><h2>1. Technical extraction QC</h2>
<p>Counts are shown separately by site and retained at recording level. Relative sums should remain near 1.</p>
{_figure_html(plot_qc(site_qc), "Technical extraction QC by site")}
<h3>Site summary</h3>{_table_html(site_qc)}
<h3>Recording-level QC</h3>{_table_html(_recording_qc_table(qc), max_height=600)}</section>

<section id="composition"><h2>2. Paper versus extended spectral composition</h2>
<p>Paper features use total 0.5–150 Hz power as denominator. Extended features use total 0.5–450 Hz power; only the extended family contains the three bands above 150 Hz.</p>
{_figure_html(plot_spectral_composition(channels), "Paper versus extended spectral composition by site")}</section>

<section id="classes"><h2>3. Pathological versus normal channels</h2>
<p>Each point is a channel-level median across valid epochs. HUP and UCLA are never pooled in these plots.</p>
<h3>Label availability</h3>{_table_html(label_counts)}
<div class="grid">
{_figure_html(plot_class_comparison(channels, "hup"), "HUP normal versus pathological bandpowers")}
{_figure_html(plot_class_comparison(channels, "ucla"), "UCLA normal versus pathological bandpowers")}
</div></section>

<section id="subjects"><h2>4. Subject-level differences</h2>
<p>For every patient with both classes: Δ = median(pathological channels) − median(normal channels). Positive values indicate larger relative power in pathological channels.</p>
{_figure_html(plot_subject_deltas(deltas, "hup"), "HUP patient-level pathological-minus-normal differences")}
{_figure_html(plot_subject_deltas(deltas, "ucla"), "UCLA patient-level pathological-minus-normal differences")}
<h3>All subject-level values</h3>{_table_html(deltas, max_height=700)}</section>

<section><h2>Per-task cohort counts</h2><p>One combined feature cohort holds both recording states; this is the split, not a separate extraction tree.</p>{_table_html(per_task_counts(channels))}</section>
</main></body></html>"""
    _atomic_write_text(output_html, document)
    LOGGER.info("Saved relative bandpower report to %s", output_html)
    return output_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch-parquet", type=Path, required=True)
    parser.add_argument("--channel-parquet", type=Path, required=True)
    parser.add_argument("--qc-csv", type=Path, required=True)
    parser.add_argument("--run-summary-json", type=Path, required=True)
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("relative_bandpower_extraction_report.html"),
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    generate_report(
        epoch_parquet=args.epoch_parquet.expanduser().resolve(),
        channel_parquet=args.channel_parquet.expanduser().resolve(),
        qc_csv=args.qc_csv.expanduser().resolve(),
        run_summary_json=args.run_summary_json.expanduser().resolve(),
        output_html=args.output_html.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
