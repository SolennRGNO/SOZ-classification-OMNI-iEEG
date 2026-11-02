#!/usr/bin/env python3
"""Generate a standalone HTML QC report for OMNI-iEEG Mantilla-Ramos 2023 complexity features."""

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


LOGGER = logging.getLogger("hup_ucla_complexity_report")
REPORT_PROTOCOL_VERSION = "1.0.0"
FEATURES = [
    "aperiodic_slope",
    "approximate_entropy",
    "sample_entropy",
    "permutation_entropy",
    "spectral_entropy",
    "svd_entropy",
    "multiscale_entropy",
    "lempel_ziv_complexity",
    "zero_crossing_rate",
    "hjorth_mobility",
    "hjorth_complexity",
    "katz_fractal_dimension",
    "petrosian_fractal_dimension",
    "higuchi_fractal_dimension",
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
    return (
        f'<div class="table-wrap"{style}>'
        + frame.to_html(index=False, border=0, escape=True, na_rep="")
        + "</div>"
    )


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
                "input_epochs": int(_numeric(group, "n_epochs_input").sum()),
                "valid_epochs": int(_numeric(group, "n_epochs_valid").sum()),
                "rejected_epochs": int(_numeric(group, "n_epochs_rejected").sum()),
                "valid_channel_epochs": int(
                    _numeric(group, "n_channel_epochs_valid").sum()
                ),
                "nan_features": int(_numeric(group, "n_nan_features").sum()),
                "infinite_features": int(
                    _numeric(group, "n_infinite_features").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_distributions(channels: pd.DataFrame) -> str:
    site_column = _site_column(channels)
    sites = sorted(channels[site_column].dropna().unique())
    if not sites:
        return ""
    ncols = 3
    nrows = int(np.ceil(len(FEATURES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows), constrained_layout=True)
    flat_axes = axes.flatten()
    colors = {"hup": "#1b9e77", "ucla": "#d95f02"}
    for idx, feature in enumerate(FEATURES):
        ax = flat_axes[idx]
        mean_col = f"{feature}_mean"
        if mean_col not in channels.columns:
            ax.set_visible(False)
            continue
        data = [
            channels.loc[channels[site_column] == s, mean_col].dropna().values
            for s in sites
        ]
        bp = ax.boxplot(data, tick_labels=[s.upper() for s in sites], patch_artist=True)
        for patch, s in zip(bp["boxes"], sites):
            patch.set_facecolor(colors.get(str(s).lower(), "#7570b3"))
            patch.set_alpha(0.7)
        ax.set_title(feature.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    for idx in range(len(FEATURES), len(flat_axes)):
        flat_axes[idx].set_visible(False)
    return _figure(fig, "Complexity Feature Distributions by Site (Channel-Level Means)")


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


def generate_report(feature_dir: Path, output_html: Path) -> Path:
    qc_path = feature_dir / "complexity_extraction_qc.csv"
    channels_path = feature_dir / "complexity_channels_hup_ucla.parquet"
    summary_path = feature_dir / "complexity_run_summary.json"

    if not qc_path.is_file() or not channels_path.is_file():
        raise FileNotFoundError(
            f"Missing required assembly files in {feature_dir} (expected {qc_path.name} and {channels_path.name})"
        )
    qc = pd.read_csv(qc_path)
    channels = pd.read_parquet(channels_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}

    site_qc = build_site_qc(qc)
    dist_html = plot_distributions(channels)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OMNI-iEEG Complexity Feature Extraction Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.5; color: #24292e; max-width: 1200px; margin: 0 auto; padding: 2rem; background: #fafbfc; }}
  h1, h2, h3 {{ color: #1b1f23; border-bottom: 1px solid #e1e4e8; padding-bottom: 0.3em; }}
  .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin: 1.5rem 0; }}
  .card {{ background: #fff; border: 1px solid #e1e4e8; border-radius: 6px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
  .card-label {{ font-size: 0.85rem; color: #586069; text-transform: uppercase; letter-spacing: 0.05em; }}
  .card-value {{ font-size: 1.6rem; font-weight: 600; color: #0366d6; margin-top: 0.2rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; background: #fff; border: 1px solid #e1e4e8; border-radius: 6px; overflow: hidden; }}
  th, td {{ padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid #e1e4e8; font-size: 0.9rem; }}
  th {{ background: #f6f8fa; font-weight: 600; }}
  tr:last-child td {{ border-bottom: none; }}
  .table-wrap {{ overflow-x: auto; margin: 1rem 0; }}
  figure {{ margin: 2rem 0; text-align: center; background: #fff; border: 1px solid #e1e4e8; border-radius: 6px; padding: 1rem; }}
  figcaption {{ font-size: 0.9rem; color: #586069; margin-top: 0.5rem; }}
  img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
<h1>OMNI-iEEG Complexity Feature Extraction Report</h1>
<p>Generated: <strong>{timestamp}</strong> | Protocol Version: <strong>{REPORT_PROTOCOL_VERSION}</strong></p>

<div class="meta-grid">
  <div class="card"><div class="card-label">Recordings Attempted</div><div class="card-value">{summary.get("recordings_attempted", len(qc))}</div></div>
  <div class="card"><div class="card-label">Recordings Completed</div><div class="card-value">{summary.get("recordings_completed", len(qc))}</div></div>
  <div class="card"><div class="card-label">Valid Channel-Epochs</div><div class="card-value">{len(channels):,}</div></div>
  <div class="card"><div class="card-label">Features Extracted</div><div class="card-value">{len(FEATURES)}</div></div>
</div>

<h2>Site Quality Control Summary</h2>
{_table(site_qc)}

<h2>Feature Distributions</h2>
{dist_html}

<h2>Feature Protocol Definitions</h2>
<table>
  <thead><tr><th>Feature Name</th><th>Scientific Specification & Reference</th></tr></thead>
  <tbody>
    {"".join(f"<tr><td><code>{k}</code></td><td>{v}</td></tr>" for k, v in summary.get("feature_definitions", {}).items())}
  </tbody>
</table>

<section><h2>Per-task cohort counts</h2><p>One combined feature cohort holds both recording states; this is the split, not a separate extraction tree.</p>{_table(per_task_counts(channels))}</section>
</body>
</html>
"""
    _atomic_write(output_html, html_content)
    return output_html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    report = generate_report(args.feature_dir.resolve(), args.output_html.resolve())
    print(f"Report generated: {report}")


if __name__ == "__main__":
    main()
