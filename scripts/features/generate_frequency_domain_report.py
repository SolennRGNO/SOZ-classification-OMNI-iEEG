#!/usr/bin/env python3
"""Generate a standalone HTML QC report for frequency-domain features."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("hup_ucla_frequency_domain_report")
REPORT_PROTOCOL_VERSION = "1.0.0"
FEATURES = ["spectral_entropy", "peak_frequency"]


def _site_column(frame: pd.DataFrame) -> str:
    for column in ("site", "dataset"):
        if column in frame.columns:
            return column
    raise ValueError("input table has neither a site nor a dataset column")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _table(frame: pd.DataFrame, max_height: int | None = None) -> str:
    if frame.empty:
        return '<p class="notice">No compatible data available.</p>'
    style = f' style="max-height:{max_height}px"' if max_height else ""
    return f'<div class="table-wrap"{style}>' + frame.to_html(index=False, border=0, escape=True, na_rep="") + "</div>"


def _figure(figure: plt.Figure, caption: str) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=145, bbox_inches="tight")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".html", prefix=f".{path.name}.", dir=path.parent, encoding="utf-8", delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def site_qc(qc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site, group in qc.groupby(_site_column(qc), dropna=False, sort=True):
        status = group["status"].astype(str).str.lower()
        rows.append({
            "site": str(site).lower(), "recordings": len(group),
            "completed": int(status.eq("completed").sum()),
            "failed": int(status.ne("completed").sum()),
            "epochs_input": int(_numeric(group, "n_epochs_input").fillna(0).sum()),
            "epochs_valid": int(_numeric(group, "n_epochs_valid").fillna(0).sum()),
            "epochs_rejected": int(_numeric(group, "n_epochs_rejected").fillna(0).sum()),
            "failed_windows": int(_numeric(group, "n_failed_windows").fillna(0).sum()),
            "nan_features": int(_numeric(group, "n_nan_features").fillna(0).sum()),
            "infinite_features": int(_numeric(group, "n_infinite_features").fillna(0).sum()),
        })
    return pd.DataFrame(rows)


def plot_qc(summary: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    positions = np.arange(len(summary))
    labels = summary["site"].str.upper().tolist()
    axes[0].bar(positions, summary["epochs_valid"], label="Valid", color="#756bb1")
    axes[0].bar(positions, summary["epochs_rejected"], bottom=summary["epochs_valid"], label="Rejected", color="#fd8d3c")
    axes[0].set_title("Eight-second epochs")
    axes[1].bar(positions, summary["failed_windows"], label="Failed", color="#de2d26")
    axes[1].bar(positions, summary["nan_features"], bottom=summary["failed_windows"], label="NaN features", color="#969696")
    axes[1].set_title("Feature/QC exceptions")
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return figure


def plot_features(channels: pd.DataFrame) -> plt.Figure:
    site_column = _site_column(channels)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, feature in zip(axes, FEATURES):
        data, labels = [], []
        for site in ("hup", "ucla"):
            values = _numeric(channels[channels[site_column].astype(str).str.lower().eq(site)], f"{feature}_median").dropna()
            if len(values):
                data.append(values.to_numpy(float))
                labels.append(site.upper())
        if data:
            axis.boxplot(data, tick_labels=labels, showfliers=False)
        else:
            axis.text(0.5, 0.5, "No data", ha="center", va="center")
        axis.set_title(feature.replace("_", " "))
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Channel-level epoch medians by site", fontweight="bold")
    figure.tight_layout()
    return figure


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


def generate_report(epoch_parquet: Path, channel_parquet: Path, qc_csv: Path, run_summary_json: Path, output_html: Path) -> Path:
    for path in (epoch_parquet, channel_parquet, qc_csv, run_summary_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    channels = pd.read_parquet(channel_parquet)
    qc = pd.read_csv(qc_csv)
    run = json.loads(run_summary_json.read_text(encoding="utf-8"))
    required = {"participant_id", "recording_id", "channel", "n_valid_epochs"}
    required.update(f"{feature}_median" for feature in FEATURES)
    missing = sorted(required - set(channels.columns))
    if missing:
        raise ValueError(f"channel-level Parquet is missing: {', '.join(missing)}")
    summary = site_qc(qc)
    provenance = pd.DataFrame([
        {"parameter": "Report protocol", "value": REPORT_PROTOCOL_VERSION},
        {"parameter": "Extraction protocol", "value": run.get("protocol_version", "")},
        {"parameter": "Preprocessing protocol", "value": run.get("expected_preprocessing_protocol_version", "")},
        {"parameter": "Git commit", "value": run.get("git_commit", "")},
        {"parameter": "Cohort SHA-256", "value": run.get("cohort_csv_sha256", "")},
    ])
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HUP/UCLA frequency-domain feature report</title><style>
body{{font-family:Arial,sans-serif;margin:0;background:#f5f6f8;color:#222;line-height:1.45}}header{{background:#172554;color:white;padding:28px 5vw}}main{{max-width:1400px;margin:auto;padding:24px}}section{{background:white;padding:22px;margin:18px 0;border-radius:8px;box-shadow:0 1px 4px #0002}}figure{{margin:18px 0}}img{{max-width:100%;display:block;margin:auto}}figcaption{{text-align:center;color:#555}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:.86rem}}th,td{{padding:6px 8px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{background:#e9eef8;position:sticky;top:0}}</style></head><body>
<header><h1>Frequency-domain feature extraction report — HUP &amp; UCLA</h1><p>Welch PSD entropy and peak-frequency QC for 8-second, 1 kHz epochs.</p><p>Generated {html.escape(datetime.now(timezone.utc).isoformat())}</p></header><main>
<section><h2>Scope and provenance</h2>{_table(provenance)}</section>
<section><h2>1. Technical extraction QC</h2>{_figure(plot_qc(summary), "Technical extraction QC by site")}{_table(summary)}</section>
<section><h2>2. Feature distributions</h2>{_figure(plot_features(channels), "Channel-level frequency-domain features by site")}</section>
<section><h2>Per-task cohort counts</h2><p>One combined feature cohort holds both recording states; this is the split, not a separate extraction tree.</p>{_table(per_task_counts(channels))}</section>
</main></body></html>"""
    _atomic_write(output_html, document)
    LOGGER.info("Saved frequency-domain report to %s", output_html)
    return output_html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch-parquet", type=Path, required=True)
    parser.add_argument("--channel-parquet", type=Path, required=True)
    parser.add_argument("--qc-csv", type=Path, required=True)
    parser.add_argument("--run-summary-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    generate_report(args.epoch_parquet.resolve(), args.channel_parquet.resolve(), args.qc_csv.resolve(), args.run_summary_json.resolve(), args.output_html.resolve())


if __name__ == "__main__":
    main()
