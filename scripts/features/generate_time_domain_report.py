#!/usr/bin/env python3
"""Generate a standalone HTML QC report for OMNI-iEEG time-domain features."""

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


LOGGER = logging.getLogger("hup_ucla_time_domain_report")
REPORT_PROTOCOL_VERSION = "1.0.0"
FEATURES = [
    "mean",
    "median",
    "variance",
    "standard_deviation",
    "rms",
    "line_length",
    "zero_crossing_rate",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "skewness",
    "kurtosis",
]


def _site_column(frame: pd.DataFrame) -> str:
    for column in ("site", "dataset"):
        if column in frame.columns:
            return column
    raise ValueError("input table has neither a site nor a dataset column")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _table(frame: pd.DataFrame, max_height: int | None = None) -> str:
    if frame.empty:
        return '<p class="notice">No compatible data available.</p>'
    style = f' style="max-height:{max_height}px"' if max_height else ""
    return f'<div class="table-wrap"{style}>' + frame.to_html(
        index=False, border=0, escape=True, na_rep=""
    ) + "</div>"


def _figure(figure: plt.Figure, caption: str) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=145, bbox_inches="tight")
    plt.close(figure)
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        f'<figure><img src="{uri}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", prefix=f".{path.name}.", dir=path.parent,
        encoding="utf-8", delete=False,
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_site_qc(qc: pd.DataFrame) -> pd.DataFrame:
    site_column = _site_column(qc)
    rows = []
    for site, group in qc.groupby(site_column, dropna=False, sort=True):
        status = group["status"].astype(str).str.lower()
        rows.append(
            {
                "site": str(site).lower(),
                "recordings": len(group),
                "completed": int(status.eq("completed").sum()),
                "failed": int(status.ne("completed").sum()),
                "channels_input": int(_numeric(group, "n_channels_input").sum()),
                "channels_used": int(_numeric(group, "n_channels_used").sum()),
                "epochs_input": int(_numeric(group, "n_epochs_input").sum()),
                "epochs_valid": int(_numeric(group, "n_epochs_valid").sum()),
                "epochs_rejected": int(_numeric(group, "n_epochs_rejected").sum()),
                "flat_windows": int(_numeric(group, "n_flat_channel_epochs").sum()),
                "non_finite_windows": int(
                    _numeric(group, "n_non_finite_channel_epochs").sum()
                ),
                "failed_windows": int(_numeric(group, "n_failed_windows").sum()),
                "nan_features": int(_numeric(group, "n_nan_features").sum()),
                "infinite_features": int(_numeric(group, "n_infinite_features").sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_qc(summary: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(summary))
    labels = summary["site"].str.upper().tolist()
    axes[0].bar(x, summary["completed"], label="Completed", color="#31a354")
    axes[0].bar(x, summary["failed"], bottom=summary["completed"], label="Failed", color="#de2d26")
    axes[0].set_title("Recordings")
    axes[1].bar(x, summary["epochs_valid"], label="Valid", color="#756bb1")
    axes[1].bar(x, summary["epochs_rejected"], bottom=summary["epochs_valid"], label="Rejected", color="#fd8d3c")
    axes[1].set_title("Eight-second epochs")
    width = 0.25
    for offset, column, label in (
        (-width, "flat_windows", "Flat"),
        (0.0, "non_finite_windows", "NaN/Inf input"),
        (width, "failed_windows", "Failed"),
    ):
        axes[2].bar(x + offset, summary[column], width, label=label)
    axes[2].set_title("Invalid channel windows")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def feature_summary(channels: pd.DataFrame) -> pd.DataFrame:
    site_column = _site_column(channels)
    rows = []
    for site, group in channels.groupby(site_column, sort=True):
        for feature in FEATURES:
            values = _numeric(group, f"{feature}_median").dropna()
            rows.append(
                {
                    "site": str(site).lower(),
                    "feature": feature,
                    "channels": len(values),
                    "median": float(values.median()) if len(values) else np.nan,
                    "q05": float(values.quantile(0.05)) if len(values) else np.nan,
                    "q95": float(values.quantile(0.95)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_feature_distributions(channels: pd.DataFrame) -> plt.Figure:
    site_column = _site_column(channels)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for axis, feature in zip(axes.flat, FEATURES):
        data, labels = [], []
        for site in ("hup", "ucla"):
            values = _numeric(
                channels[channels[site_column].astype(str).str.lower().eq(site)],
                f"{feature}_median",
            ).dropna()
            if len(values):
                data.append(values.to_numpy(float))
                labels.append(site.upper())
        if data:
            axis.boxplot(data, tick_labels=labels, showfliers=False)
        else:
            axis.text(0.5, 0.5, "No data", ha="center", va="center")
        axis.set_title(feature.replace("_", " "))
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Channel-level epoch medians by site", fontweight="bold")
    fig.tight_layout()
    return fig


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
    site_qc = build_site_qc(qc)
    definitions = pd.DataFrame(
        [{"feature": key, "definition": value} for key, value in run.get("feature_definitions", {}).items()]
    )
    provenance = pd.DataFrame(
        [
            {"parameter": "Report protocol", "value": REPORT_PROTOCOL_VERSION},
            {"parameter": "Extraction protocol", "value": run.get("protocol_version", "")},
            {"parameter": "Preprocessing protocol", "value": run.get("expected_preprocessing_protocol_version", "")},
            {"parameter": "Git commit", "value": run.get("git_commit", "")},
            {"parameter": "Cohort SHA-256", "value": run.get("cohort_csv_sha256", "")},
        ]
    )
    recording_columns = [
        "site", "participant_id", "recording_id", "status", "exclusion_reason",
        "n_channels_input", "n_channels_used", "n_epochs_input", "n_epochs_valid",
        "n_epochs_rejected", "n_flat_channel_epochs", "n_non_finite_channel_epochs",
        "n_failed_windows", "n_nan_features", "n_infinite_features",
    ]
    recording_qc = qc[[column for column in recording_columns if column in qc.columns]]
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HUP/UCLA time-domain feature report</title><style>
body{{font-family:Arial,sans-serif;margin:0;background:#f5f6f8;color:#222;line-height:1.45}}
header{{background:#172554;color:white;padding:28px 5vw}}main{{max-width:1500px;margin:auto;padding:24px}}
section{{background:white;padding:22px;margin:18px 0;border-radius:8px;box-shadow:0 1px 4px #0002}}
figure{{margin:18px 0}}img{{max-width:100%;display:block;margin:auto}}figcaption{{text-align:center;color:#555}}
.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:.86rem}}th,td{{padding:6px 8px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{background:#e9eef8;position:sticky;top:0}}
</style></head><body><header><h1>Time-domain feature extraction report — HUP &amp; UCLA</h1>
<p>Technical QC for 8-second, 1 kHz epochs. No machine learning is included.</p>
<p>Generated {html.escape(datetime.now(timezone.utc).isoformat())}</p></header><main>
<section><h2>Scope and provenance</h2>{_table(provenance)}<h3>Feature definitions</h3>{_table(definitions)}</section>
<section><h2>1. Technical extraction QC</h2>{_figure(plot_qc(site_qc), "Technical extraction QC by site")}{_table(site_qc)}
<h3>Recording-level QC</h3>{_table(recording_qc, 700)}</section>
<section><h2>2. Feature distributions</h2>{_figure(plot_feature_distributions(channels), "Channel-level temporal feature distributions by site")}
<h3>Distribution summary</h3>{_table(feature_summary(channels), 700)}</section>
<section><h2>Per-task cohort counts</h2><p>One combined feature cohort holds both recording states; this is the split, not a separate extraction tree.</p>{_table(per_task_counts(channels))}</section>
</main></body></html>"""
    _atomic_write(output_html, document)
    LOGGER.info("Saved time-domain report to %s", output_html)
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
    generate_report(
        args.epoch_parquet.resolve(), args.channel_parquet.resolve(), args.qc_csv.resolve(),
        args.run_summary_json.resolve(), args.output_html.resolve(),
    )


if __name__ == "__main__":
    main()
